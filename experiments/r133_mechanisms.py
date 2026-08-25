"""R-133 mechanisms — the two frozen branches, implemented verbatim.

Both wrap `kelly_regime_v4`'s own `prepare()` latch loop and change only
WHETHER / HOW MUCH the latched position moves, conditioned on the strategy's
own trailing realized turnover. Neither touches the vote, the volatility
scale, the regime state machine, or anything the signal is made of.

The trailing-turnover EWM must be computed ONLINE inside the loop, not
precomputed: the throttle changes which bars fire a rebalance, so the EWM is
a function of the throttle's own decisions. The recursion below reproduces
`pandas.Series.ewm(span=..., adjust=True).mean()` exactly (verified in
`_selftest`), so the census in `r133_stepA_turnover_census.py` and the live
throttle read the same scale.

Causality: every quantity at bar `i` is a function of events at bars `<= i`
only, and the EWM the throttle reads at bar `i` contains events at `< i`
only (it is advanced *after* bar `i`'s own decision). `vol` is already
`.shift(1)`-ed by v4 itself. No scaler, quantile, mean or std is computed
over the whole series; `_selftest` includes a truncation probe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

from r131_shared import (
    ETA,
    LAMBDA_MAX,
    OVERRIDE_MULT,
    TURNOVER_EWM_SPAN_DAYS,
    TURNOVER_UPPER,
)


class _ThrottledV4(KellyRegimeV4):
    """v4's exact prepare() loop with one throttle hook spliced into the latch.

    Subclasses implement `_apply(pos, desired, ewm_now) -> new_pos` and may
    override `_per_bar(ewm_now)`, called on EVERY bar before `_apply`.
    Everything else — anchors, vote, volatility state machine, deadband — is
    copied verbatim from `KellyRegimeV3.prepare`.
    """

    def __init__(self, *, span_days: float = TURNOVER_EWM_SPAN_DAYS,
                 upper: float = TURNOVER_UPPER, **kwargs) -> None:
        super().__init__(**kwargs)
        self.span_days = span_days
        self.upper = upper
        self.diag: dict = {}

    # -- hooks ----------------------------------------------------------
    def _reset(self) -> None:
        pass

    def _per_bar(self, ewm_now: float) -> None:
        pass

    def _apply(self, pos: float, desired: float, ewm_now: float) -> float:
        raise NotImplementedError

    def _state_scalar(self) -> float:
        return 0.0

    # -------------------------------------------------------------------
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        ewm_trace = np.zeros(n)
        state_trace = np.zeros(n)
        pos, state = 0.0, 0

        alpha = 2.0 / (self.span_days * BARS_PER_DAY + 1.0)
        num = den = 0.0          # pandas adjust=True numerator / denominator

        self._reset()
        n_pending = n_intervened = 0

        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = frac[i] * scale

            ewm_now = (num / den * BARS_PER_DAY) if den > 0 else 0.0
            self._per_bar(ewm_now)

            fired = 0.0
            if abs(desired - pos) > self.deadband:
                n_pending += 1
                new_pos = self._apply(pos, desired, ewm_now)
                if abs(new_pos - pos) > 1e-12:
                    fired = 1.0
                if abs(new_pos - desired) > 1e-12:
                    n_intervened += 1
                pos = new_pos

            num = alpha * fired + (1.0 - alpha) * num
            den = alpha + (1.0 - alpha) * den
            ewm_trace[i] = num / den * BARS_PER_DAY if den > 0 else 0.0
            state_trace[i] = self._state_scalar()
            target[i] = pos

        df["target"] = target
        self.diag = {
            "n_pending": n_pending,
            "n_intervened": n_intervened,
            "ewm_trace": ewm_trace,
            "state_trace": state_trace,
            "index": df.index,
        }
        return df


# ----------------------------------------------------------------------
# CONSERVATIVE — band turnover regularization (Khubiev et al. 2025).
# ----------------------------------------------------------------------

class ConservativeTurnoverBand(_ThrottledV4):
    """v4 that defers rebalances while its own trailing turnover runs hot.

    Inside the corridor `[0, upper]` this IS `kelly_regime_v4`, bar for bar.
    At or above the corridor's upper edge the pending rebalance is deferred,
    with two pre-registered overrides so the mechanism can never indefinitely
    block a de-risking trade: a full exit (`desired == 0`) always executes,
    and so does any move larger than `override_mult * upper`.
    """

    name = "r133_conservative_turnover_band"

    def __init__(self, *, override_mult: float = OVERRIDE_MULT, **kwargs) -> None:
        super().__init__(**kwargs)
        self.override_mult = override_mult

    def _reset(self) -> None:
        self.n_defer = 0
        self.n_override_exit = 0
        self.n_override_size = 0
        self.defer_bars: list[int] = []
        self._bar = -1

    def _per_bar(self, ewm_now: float) -> None:
        self._bar += 1

    def _apply(self, pos: float, desired: float, ewm_now: float) -> float:
        if ewm_now < self.upper:
            return desired
        if desired == 0.0:
            self.n_override_exit += 1
            return desired
        if abs(desired - pos) > self.override_mult * self.upper:
            self.n_override_size += 1
            return desired
        self.n_defer += 1
        self.defer_bars.append(self._bar)
        return pos


# ----------------------------------------------------------------------
# NOVEL — online dual-ascent shadow price on turnover, the causal analogue
# of Boyd et al. (2017)'s resource-constrained multi-period control:
#     lambda_{t+1} = clip(lambda_t + eta * (ewm_t - upper), 0, lam_max)
# and the pending rebalance is SHRUNK by 1/(1+lambda_t), not skipped.
# ----------------------------------------------------------------------

class NovelTurnoverThrottle(_ThrottledV4):
    """v4 whose rebalance size is shrunk by a self-regulating shadow price on turnover."""

    name = "r133_novel_turnover_throttle"

    def __init__(self, *, eta: float = ETA, lam_max: float = LAMBDA_MAX,
                 lam_const: float | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eta = eta
        self.lam_max = lam_max
        # ABLATION CONTROL (added post-freeze, before any Step-B number was
        # read, per R-130's methodological finding): with `lam_const` set,
        # lambda is FROZEN and the turnover feedback channel is deleted
        # entirely. What remains is plain constant-rate partial adjustment —
        # i.e. exactly the Gârleanu-Pedersen smooth trading rate R-64 closed.
        # If this control matches the live branch, the branch's mechanism is
        # inert and its result is a re-derivation, not a new finding.
        self.lam_const = lam_const

    def _reset(self) -> None:
        self.lam = 0.0 if self.lam_const is None else float(self.lam_const)
        self._lam_sum = 0.0
        self._lam_n = 0
        self._lam_pos = 0
        self._lam_max_seen = 0.0

    def _per_bar(self, ewm_now: float) -> None:
        if self.lam_const is None:
            self.lam = float(np.clip(self.lam + self.eta * (ewm_now - self.upper),
                                     0.0, self.lam_max))
        self._lam_sum += self.lam
        self._lam_n += 1
        self._lam_pos += self.lam > 0.0
        self._lam_max_seen = max(self._lam_max_seen, self.lam)

    def _apply(self, pos: float, desired: float, ewm_now: float) -> float:
        return pos + (desired - pos) / (1.0 + self.lam)

    def _state_scalar(self) -> float:
        return self.lam

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = super().prepare(df)
        self.diag.update({
            "lam_mean": self._lam_sum / max(self._lam_n, 1),
            "lam_frac_positive": self._lam_pos / max(self._lam_n, 1),
            "lam_max_seen": self._lam_max_seen,
        })
        return out


# ----------------------------------------------------------------------

def _selftest() -> None:
    rng = np.random.default_rng(131)
    ev = (rng.random(200_000) < 0.0013).astype(float)
    span = TURNOVER_EWM_SPAN_DAYS * BARS_PER_DAY
    alpha = 2.0 / (span + 1.0)
    num = den = 0.0
    mine = np.empty(len(ev))
    for i, e in enumerate(ev):
        num = alpha * e + (1 - alpha) * num
        den = alpha + (1 - alpha) * den
        mine[i] = num / den
    ref = pd.Series(ev).ewm(span=span, min_periods=1).mean().to_numpy()
    assert np.allclose(mine, ref, rtol=1e-10, atol=1e-12), "online EWM != pandas ewm"
    print("online EWM recursion == pandas ewm(span, adjust=True): PASS")

    # Truncation probe: prepare() on a truncated frame must reproduce the
    # full frame's targets on every bar of the shorter frame. Any full-series
    # statistic (scaler, quantile, mean, std) would break this.
    from r131_shared import load_btc_train
    df, _ = load_btc_train()
    df = df.iloc[:120_000]
    cut = 90_000
    for cls, kw in ((ConservativeTurnoverBand, {}),
                    (NovelTurnoverThrottle, {}),
                    (NovelTurnoverThrottle, {"lam_const": 3.0})):
        full = cls(**kw).prepare(df.copy())["target"].to_numpy()
        trunc = cls(**kw).prepare(df.iloc[:cut].copy())["target"].to_numpy()
        ok = np.allclose(full[:cut], trunc, rtol=0, atol=1e-12)
        print(f"truncation probe {cls.__name__}{kw or ''}: {'PASS' if ok else 'FAIL'}")
        assert ok, f"{cls.__name__} reads ahead of its own truncation point"


if __name__ == "__main__":
    _selftest()
