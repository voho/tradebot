#!/usr/bin/env python
"""Continuous funding-aware Kelly exposure haircut (backlog B-05, novel variant).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. The sibling ``experiments/
funding_gate_decile.py`` implements the simpler binary-gate variant named
directly in B-05 ("stand flat when funding is in its top decile"); this
file implements the alternative the backlog description gestures at but
does not spell out.

**Mechanism, in one sentence.** A levered long paying a continuous funding
cost at rate ``r_funding`` per unit time has net growth rate
``g(f) = f*(mu - r_funding) - f^2*sigma^2/2`` instead of
``f*mu - f^2*sigma^2/2``, so its growth-optimal exposure shrinks from
``f* = mu/sigma^2`` to ``f*_funding = f* - r_funding/sigma^2`` — the exact
same algebra ``kelly_regime_ev.py`` (this project's precedent) used to
derive a no-trade band from a *fee*, applied here to a second,
continuously-accruing cost instead of a per-trade one.

``kelly_regime`` never estimates ``mu`` directly — it proxies ``f*`` with
``target_vol / realized_vol`` under the latched anchor-vote gate, precisely
*because* estimating drift is unreliable. This variant applies the same
structural correction to that proxy rather than to a fitted ``mu``: it
subtracts a causal, trailing, annualized estimate of the funding rate
(divided by realized variance) from the pre-deadband desired exposure,
every bar, continuously — floored at 0 so funding never pushes the
strategy short, consistent with kelly_regime's existing philosophy of
standing flat rather than shorting a historically upward-drifting asset.
This is smoother than a hard decile gate: no discontinuous flatten/
re-enter, just a continuous downward tilt whenever funding is rich,
applied *before* the existing 10% deadband/latch loop so it does not
defeat the existing turnover control.

**Constraint attacked.** COST — specifically the part of costs that scales
with the signal (funding on a levered long) rather than with turnover
(fees, already handled by ``kelly_regime_ev``).

**Not a duplicate of.** R-16/B-05's own framing is a binary decile gate;
this is the continuous, growth-theoretic alternative, structurally
parallel to ``kelly_regime_ev`` (fee no-trade band) rather than to a
predictive on/off switch. Not a duplicate of R-14 (which only measured
the adverse timing, did not act on it) or R-28/R-31 (evidence-gated
*regime* sizing, unrelated cost channel).

**Pre-registered falsification test (chosen before any result was read).**
Restrict the inner-validation Monte-Carlo-style robustness check to random
~90-day windows drawn entirely from 2020-01-01..2022-12-31 (the funding-
covered span) and check whether the selected config beats
``kelly_regime_v4`` in a *majority* of them on futures, paired per window.
Outcome that kills it: a majority loss rate, or a win rate driven by a
handful of windows while most are worse (median at or below zero net of
v4).

Usage::

    python experiments/funding_gate_continuous.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

SETTLEMENTS_PER_DAY = 3
SETTLEMENTS_PER_YEAR = SETTLEMENTS_PER_DAY * 365.25

# This project's usual inner split (2017-2020 / 2021-2022) predates the
# funding file entirely. Adapted to what the funding data actually covers
# (2020-01-01..2023-12-31): inner-train shortens to two years and
# inner-validation moves to 2022 so that BOTH inner slices carry real,
# observed funding rather than the assumed/blended fallback the rest of
# the project uses outside this window. Nothing here reads 2023-01-01
# onward (the holdout).
INNER_TRAIN = ("2020-01-01", "2021-12-31")
INNER_VAL = ("2022-01-01", "2022-12-31")

FUNDING_SPAN_GRID = (14, 30, 60)
HAIRCUT_STRENGTH_GRID = (0.5, 1.0, 2.0)


# --------------------------------------------------------------------------- strategy

class FundingAwareKelly(KellyRegimeV4):
    """kelly_regime_v4 with a continuous, analytically-derived funding haircut.

    See module docstring for the derivation. Constructor parameters:

    ``funding_span_days``  span (in days) of the causal EWM used to
                           estimate the trailing expected funding rate.
    ``haircut_strength``   multiplies the theoretically-derived coefficient
                           of 1.0; MacLean-Thorp-Ziemba fractional-Kelly
                           logic argues for shrinking below full-Kelly
                           anyway, and the mapping from "trailing EWM
                           funding" to "the mu-equivalent drift correction"
                           is itself an approximation, so this is swept.
    ``funding``            optional pre-loaded funding Series (positive =
                           longs pay); lazily loaded from ``data/`` via
                           ``tradebot.data.load_funding`` if omitted, so
                           a driver running many configs need only load
                           the file once and inject it.

    **Design choice, stated explicitly**: a *negative* funding rate means
    longs are being PAID, not charged — this implementation applies the
    haircut only when the trailing estimate is positive and leaves it at
    exactly 0 otherwise (no symmetric exposure *boost* for rich negative
    funding). That is the conservative reading: rewarding negative funding
    would mean leaning further into a mu-equivalent correction estimated
    from a noisy trailing average, in the same direction R-28/R-31 warned
    raising an exposure knob on stale evidence is dangerous. It is a
    choice, not a derivation, and a symmetric variant is a natural next
    step if this one is promoted.

    ``prepare()`` calls ``super().prepare(df)`` first to get
    ``kelly_regime_v4``'s finished ``target`` (v4 = v3 with different
    anchors, so v3's conditional-vol-targeting loop, inlined below, is the
    parent's own algorithm — not a re-derivation), then independently
    recomputes the identical vote/vol/scale arrays used inside it so the
    pre-deadband desired exposure ``frac[i]*scale[i]`` can be recovered,
    haircut, and re-fed through the SAME deadband/latch loop. Where the
    funding estimate is exactly 0 (outside 2020-2023, or before either
    trailing EWM has enough history), ``desired_adjusted`` equals
    ``desired`` exactly and the re-run deadband loop reproduces v4's own
    ``target`` bit-for-bit — verified in ``main()``.
    """

    def __init__(self, funding_span_days: float = 30.0, haircut_strength: float = 1.0,
                 funding: pd.Series | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.funding_span_days = funding_span_days
        self.haircut_strength = haircut_strength
        self._funding = funding

    def _funding_series(self) -> pd.Series | None:
        if self._funding is not None:
            return self._funding
        self._funding = load_funding(ROOT / "data")
        return self._funding

    def _funding_annualized_on_grid(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Causal trailing EWM funding estimate, annualized, ffilled onto ``index``.

        Exactly 0 outside the committed 2020-2023 coverage window and
        before the EWM has enough history to mean anything — missing
        funding data is treated as zero expected funding cost, never
        fabricated or extrapolated.
        """
        funding = self._funding_series()
        if funding is None or len(funding) == 0:
            return np.zeros(len(index))

        span = max(self.funding_span_days * SETTLEMENTS_PER_DAY, 1.0)
        # causal: shift(1) so a settlement never sees itself, mirroring
        # kelly_regime.py's own `r.ewm(span=...).std().shift(1)` convention
        trailing = funding.ewm(span=span, min_periods=SETTLEMENTS_PER_DAY).mean().shift(1)
        annualized = trailing * SETTLEMENTS_PER_DAY * 365.25

        grid = annualized.reindex(index, method="ffill")
        in_coverage = (index >= funding.index[0]) & (index <= funding.index[-1])
        grid = grid.where(in_coverage, 0.0)
        return grid.fillna(0.0).to_numpy()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4's finished target: the no-funding fallback

        close = df["close"]
        r = np.log(close).diff()

        # --- mirror kelly_regime_v3's vote + conditional-vol-targeting
        # exactly (v4 IS v3 with different anchors) so we can intercept
        # `desired = frac * scale` before ITS deadband/latch loop runs.
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
        scale = np.empty(n)
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale[i] = full[i] if state != 0 else steady[i]

        desired = frac * scale  # pre-deadband desired exposure, == v4's own

        funding_annualized = self._funding_annualized_on_grid(df.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            haircut = np.where(
                funding_annualized > 0.0,
                self.haircut_strength * funding_annualized / np.where(vol > 0, vol, np.nan) ** 2,
                0.0,
            )
        haircut = np.nan_to_num(haircut, nan=0.0, posinf=0.0, neginf=0.0)
        desired_adjusted = np.maximum(0.0, desired - haircut)

        # Re-apply the SAME deadband/latch loop v4 uses, now on the
        # funding-adjusted desired exposure, before this reaches the market.
        pos = 0.0
        target = np.empty(n)
        for i in range(n):
            if abs(desired_adjusted[i] - pos) > self.deadband:
                pos = desired_adjusted[i]
            target[i] = pos

        df["target"] = target
        df["_fa_desired"] = desired
        df["_fa_desired_adj"] = desired_adjusted
        df["_fa_funding_annualized"] = funding_annualized
        df["_fa_vol"] = vol
        return df

    # on_bar inherited unchanged from KellyRegimeV4 -> v3 -> KellyRegime:
    # it just follows df["target"] with ctx.order_notional(t).


# --------------------------------------------------------------------------- data / windows

DF, LABEL = load_dataset(ROOT / "data", "spot")
FUNDING = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()


def _period(strategy, market: MarketSpec, start=None, end=None, funding=None):
    """Backtest over a date range, warmed on the bars before it (funding_study.py's pattern).

    tradebot.window.run_period does not accept `funding=`, which is why
    this mirrors funding_study.py's `_period` instead of using it.
    """
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    metrics = compute_metrics(trimmed)
    mean_exposure = float(np.mean(np.abs(trimmed.df["target"].to_numpy()))) \
        if "target" in trimmed.df.columns else float("nan")
    return {
        "final_balance": metrics.final_balance,
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "sharpe": metrics.sharpe,
        "num_trades": metrics.num_trades,
        "fees_paid": metrics.fees_paid,
        "funding_paid": raw.funding_paid,
        "mean_exposure": mean_exposure,
    }


def run_sweep() -> pd.DataFrame:
    splits = {"inner-train": INNER_TRAIN, "inner-val": INNER_VAL}
    markets = {"spot": SPOT, "futures": FUTURES}

    rows = []
    n_backtests = 0

    # baselines, run once per split/market and reused across every config row
    for split_name, (start, end) in splits.items():
        for market_name, market in markets.items():
            for strat_name in ("kelly_regime_v4", "buy_and_hold"):
                strat = get_strategy(strat_name)
                stats = _period(strat, market, start, end, funding=FUNDING)
                n_backtests += 1
                rows.append({"config": strat_name, "span": None, "strength": None,
                            "split": split_name, "market": market_name, **stats})

    # the pre-registered grid: 3 spans x 3 strengths x 2 splits x 2 markets = 36
    for span in FUNDING_SPAN_GRID:
        for strength in HAIRCUT_STRENGTH_GRID:
            cfg_name = f"span={span} str={strength}"
            for split_name, (start, end) in splits.items():
                for market_name, market in markets.items():
                    strat = FundingAwareKelly(funding_span_days=span,
                                              haircut_strength=strength, funding=FUNDING)
                    stats = _period(strat, market, start, end, funding=FUNDING)
                    n_backtests += 1
                    rows.append({"config": cfg_name, "span": span, "strength": strength,
                                "split": split_name, "market": market_name, **stats})

    print(f"\nbacktests run: {n_backtests} "
          f"(pre-registered grid = {len(FUNDING_SPAN_GRID) * len(HAIRCUT_STRENGTH_GRID)} configs "
          f"x 2 splits x 2 markets = "
          f"{len(FUNDING_SPAN_GRID) * len(HAIRCUT_STRENGTH_GRID) * 4} sweep backtests, "
          f"plus {n_backtests - len(FUNDING_SPAN_GRID) * len(HAIRCUT_STRENGTH_GRID) * 4} "
          "baseline backtests for kelly_regime_v4 / buy_and_hold)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- lookahead self-check

def lookahead_check(span: float, strength: float) -> bool:
    """Two-opposite-tampers check (test_causality_strict.py's pattern), pre-2023 data only.

    Perturb all bars after a cutoff two different ways (x3 / /3); the
    `target` column before the cutoff must be byte-identical in both
    copies, or the strategy is reading the future.
    """
    df_pre = DF.loc[:"2022-12-31"]
    tail = df_pre.iloc[-40_000:].copy()
    cut = len(tail) - 5_000

    up, down = tail.copy(), tail.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    strat_up = FundingAwareKelly(funding_span_days=span, haircut_strength=strength, funding=FUNDING)
    strat_down = FundingAwareKelly(funding_span_days=span, haircut_strength=strength, funding=FUNDING)
    target_up = strat_up.prepare(up)["target"].to_numpy()
    target_down = strat_down.prepare(down)["target"].to_numpy()

    identical = np.array_equal(target_up[:cut], target_down[:cut])
    first_diff = None
    if not identical:
        diffs = np.nonzero(target_up[:cut] != target_down[:cut])[0]
        first_diff = int(diffs[0]) if len(diffs) else None
    return identical, first_diff, cut


def byte_identical_to_v4_check() -> bool:
    """Outside 2020-2023, or wherever funding is exactly 0, target must match v4 exactly."""
    window = DF.loc["2018-01-01":"2019-12-31"]  # well before funding coverage, well before OOS
    v4 = get_strategy("kelly_regime_v4")
    fa = FundingAwareKelly(funding_span_days=30, haircut_strength=1.0, funding=FUNDING)
    t_v4 = v4.prepare(window.copy())["target"].to_numpy()
    t_fa = fa.prepare(window.copy())["target"].to_numpy()
    return bool(np.array_equal(t_v4, t_fa))


# --------------------------------------------------------------------------- falsification test

def falsification_test(span: float, strength: float, n_windows: int = 12,
                        min_days: int = 60, max_days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Random ~90-day windows drawn entirely from 2020-01-01..2022-12-31, futures, paired."""
    lo_ts, hi_ts = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2022-12-31", tz="UTC")
    lo = int(DF.index.searchsorted(lo_ts))
    hi = int(DF.index.searchsorted(hi_ts, side="right"))

    warmup = max(get_strategy("kelly_regime_v4").warmup,
                 FundingAwareKelly(funding_span_days=span, haircut_strength=strength).warmup) + 10
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_windows):
        length = int(rng.integers(min_days, max_days + 1) * BARS_PER_DAY)
        start = int(rng.integers(lo + warmup, hi - length))
        window = DF.iloc[start - warmup: start + length]
        eval_start = warmup

        best_bal, v4_bal = {}, {}
        for label, strat in (
            ("best", FundingAwareKelly(funding_span_days=span, haircut_strength=strength, funding=FUNDING)),
            ("v4", get_strategy("kelly_regime_v4")),
        ):
            result = run_backtest(strat, window, FUTURES, 1_000.0,
                                  trade_start=eval_start, funding=FUNDING)
            eq = result.equity.to_numpy(dtype=float)
            base = eq[eval_start]
            ret = 100.0 * (eq[-1] / base - 1.0) if np.isfinite(base) and base > 0 else -100.0
            (best_bal if label == "best" else v4_bal)["return_pct"] = ret

        rows.append({
            "window": k + 1,
            "start": window.index[eval_start],
            "days": length // BARS_PER_DAY,
            "best_return_pct": best_bal["return_pct"],
            "v4_return_pct": v4_bal["return_pct"],
            "best_beats_v4": best_bal["return_pct"] > v4_bal["return_pct"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main

def main() -> None:
    if FUNDING is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")

    print("=" * 90)
    print("funding_gate_continuous: continuous funding-aware Kelly exposure haircut (B-05)")
    print(f"grid: funding_span_days in {FUNDING_SPAN_GRID} x "
          f"haircut_strength in {HAIRCUT_STRENGTH_GRID} = "
          f"{len(FUNDING_SPAN_GRID) * len(HAIRCUT_STRENGTH_GRID)} configs")
    print(f"inner-train {INNER_TRAIN}, inner-val {INNER_VAL} (funding-coverage-forced split)")
    print("=" * 90)

    results = run_sweep()
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 200):
        print(results.round(4).to_string(index=False))

    csv_path = ROOT / "reports" / "funding_gate_continuous_sweep.csv"
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(csv_path, index=False)
        print(f"\n(sweep results also written to {csv_path} for convenience; "
              "not required by the task, harmless to leave/delete)")
    except OSError:
        pass

    print("\n" + "=" * 90)
    print("byte-identical-to-v4 check (2018-2019, well outside funding coverage)")
    print("=" * 90)
    identical = byte_identical_to_v4_check()
    print(f"target column identical to kelly_regime_v4: {identical}")

    print("\n" + "=" * 90)
    print("lookahead self-check (two opposite tampers, pre-2023 data only)")
    print("=" * 90)
    for span, strength in ((30, 1.0),):
        ok, first_diff, cut = lookahead_check(span, strength)
        print(f"span={span} strength={strength}: target[:cut] byte-identical "
              f"under x3 vs /3 tamper: {ok}"
              + ("" if ok else f"  FIRST DIFF at bar {first_diff} (cut={cut})"))

    print("\n" + "=" * 90)
    print("falsification test: best config vs kelly_regime_v4, random ~90d windows, "
          "2020-2022 only, futures, paired")
    print("=" * 90)
    # Selection happens after seeing the sweep table above (inner-validation
    # only) but the test procedure itself was fixed before any result was
    # read. Fill in the chosen (span, strength) here once selected.
    SELECTED_SPAN, SELECTED_STRENGTH = 30, 1.0
    fz = falsification_test(SELECTED_SPAN, SELECTED_STRENGTH)
    with pd.option_context("display.width", 200):
        print(fz.round(2).to_string(index=False))
    win_rate = fz["best_beats_v4"].mean()
    print(f"\nselected config span={SELECTED_SPAN} strength={SELECTED_STRENGTH}: "
          f"beats kelly_regime_v4 in {fz['best_beats_v4'].sum()}/{len(fz)} windows "
          f"({win_rate:.0%})")


if __name__ == "__main__":
    main()
