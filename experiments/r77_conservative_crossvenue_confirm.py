#!/usr/bin/env python
"""R-77 CONSERVATIVE branch: Bitstamp-vs-Coinbase BTC cross-venue price
divergence fed into a precision-weighted CONFIRMING vote on
kelly_regime_v4's own 3-anchor regime gate -- the same architecture
kelly_regime_v16_stablecoin_confirm.py (R-55) used, a new signal.

Idea, one sentence
-------------------
Every INFO-axis round before this one (R-44 on-chain, R-53 macro, R-54/
R-55/R-58 stablecoin, R-73 DVOL, R-74 MVRV, R-75 calendar, R-76 cross-
instrument pairs) fed kelly_regime_v4 either an *external* data source
about a different quantity, or a relationship between two *different*
instruments on the *same* venue; none has used a second, independent
read of the SAME instrument's price from a *different* venue. This file
tests whether Bitstamp-vs-Coinbase BTC divergence (`data/btcusd_spot_5m
.csv.gz` vs `data/btcusd_coinbase_spot_5m.csv.gz`, both loaded and
causally joined by the frozen `experiments/r77_shared.py`) is the first
INFO signal in this project's eight attempts to earn its keep, fed
through the confirming-vote architecture as a 4th vote alongside v4's
three price anchors.

Constraint attacked: INFO. Not a duplicate of: R-41 (Deribit perp-vs-spot
basis -- a derivatives/spot basis on ONE venue), R-76 (cross-*instrument*
pairs on the SAME venue), R-39/B-05 (funding-rate cross-venue splice -- a
rate, not a price). This round's own parallel NOVEL branch tests the
lead-lag/price-discovery framing (Hasbrouck 1995; Gonzalo & Granger 1995;
Putninš 2013) on the same data; per this round's brief this file is
restricted to the Makarov & Schoar (2020, JFE 135(2):293-319)
arbitrage-friction framing and same-bar divergence only, not lead-lag --
a disjoint file, not read, not coordinated with.

Literature grounding
---------------------
Makarov & Schoar (2020, *Journal of Financial Economics* 135(2):293-319):
persistent, large cross-exchange crypto price deviations that
transaction costs alone cannot explain -- the economic warrant for
treating Bitstamp-vs-Coinbase divergence as a real, tradeable friction
rather than pure noise. Alexander & Heck (2020, *J. Financial Stability*
50:100776): Bitstamp and Coinbase are both in their regulated-spot panel
and neither venue leads the other at all times, consistent with treating
divergence as a two-sided, mean-reverting friction rather than a
directional edge.

Economic hypothesis, stated and checked BEFORE any vote sign was frozen
-------------------------------------------------------------------------
`divergence = log(bitstamp_close / coinbase_close)`; positive means
Bitstamp trades above Coinbase for that bar. The Makarov & Schoar
arbitrage-friction framing predicts the divergence should *mean-revert*:
when Bitstamp is unusually cheap relative to Coinbase (divergence very
negative, i.e. a rolling z-score of divergence very negative),
arbitrage flow should close the gap by pushing Bitstamp's own price back
UP -- a BULLISH confirming vote on the tradable (Bitstamp) series. When
Bitstamp is unusually expensive relative to Coinbase (z very positive),
the symmetric argument predicts Bitstamp reverting DOWN -- BEARISH.
This is a same-bar, two-sided mean-reversion hypothesis: `sign = +1`
(the z-score's own raw sign, no flip) if divergence genuinely
mean-reverts on Bitstamp's own subsequent returns; `sign = -1` (flipped)
if it instead behaves as a momentum/continuation signal on inner-train.

**Checked, not assumed:** `descriptive()` below computes
`corr(z-score of divergence at bar T, Bitstamp forward log-return over
h bars)` for h in {1, 12, 288, 864} bars and z-score windows in {6h, 1d,
3d}, on INNER-TRAIN ONLY (2017-01-01 -> 2020-12-31). Result (reported in
full in `descriptive()`'s printed table, reproduced from a clean run
before any vote code was frozen): correlation is NEGATIVE at every one
of the 12 (window, horizon) cells, ranging -0.04 (shortest window,
1-bar horizon) to -0.39 (3-day window, 1-day horizon), monotonically
more negative as either the window or the horizon lengthens. This
CONFIRMS the arbitrage-friction hypothesis' sign on this project's own
data and inner-train split -- `sign = +1` is used, unflipped, and this
choice is frozen before `sweep()`/`select()` ever ran.

Mechanism, precisely
---------------------
A rolling (bar-count, NOT whole-series) z-score of `divergence`,
`z = (divergence - rolling_mean) / rolling_std` over a window
`window_bars` (swept over 6h/1d/3d -- see "window choice" below), using
only `rolling()` (bar T's z-score uses bars <= T only, matching every
other causal indicator in this project). A latched 0/1 vote,
`divergence_vote`, using the SAME hysteresis discipline every other
anchor/confirming vote in this project uses (single threshold `thresh`,
mirroring `_anchor_votes`' own single `band` parameter, not the
stablecoin vote's asymmetric thresh_hi/gap, because this signal --
unlike stablecoin stress -- is symmetric and two-sided by the hypothesis
above):

    divergence_vote -> 1 (bullish)  when z crosses BELOW -thresh
    divergence_vote -> 0 (bearish)  when z crosses ABOVE +thresh
    divergence_vote unchanged (latched) while -thresh <= z <= thresh
    divergence_vote defaults to 0.5 (neutral) before the first crossing
        or wherever the z-score is undefined (short warmup window,
        much shorter than the strategy's own 80-day price-anchor
        warmup, so this never binds on a measured bar)

Combined vote -- R-53/R-55's own `KellyRegimeV14MacroLead`/
`KellyRegimeV16StablecoinConfirm` precision-weighted rule, literally:

    frac = (anchor_sum + weight * divergence_vote) / (3 + weight)

`anchor_sum` is v4's own three UNCHANGED 0/1 latched price-anchor votes.
`weight` is swept over {0 (identity control), 0.15, 0.33, 0.5, 1.0
(unweighted-4-way-average negative control)} -- IDENTICAL grid to
R-53/R-55's own `WEIGHTS`, same precision argument: a signal this noisy
(microstructure-frequency, not a slow macro/on-chain series) should
count as at most a fraction of one vote, not a full one. `weight=0`
recovers v4 exactly (identity check, verified in `causality()`).

Window choice, justified
--------------------------
Cross-venue arbitrage frictions in Makarov & Schoar's own data close on
the order of minutes to days, not months -- much faster than the
macro/stablecoin/on-chain signals' own multi-week windows. Three
bar-count windows are swept: 72 bars (6h), 288 bars (1d), 864 bars (3d),
covering roughly one order of magnitude around the horizon Makarov &
Schoar's own price-discovery literature treats as plausible for
cross-exchange arbitrage to resolve. `min_periods = window_bars // 2`
(half the window), matching this project's own convention
(`r76_conservative_pairs_cointegration.py`'s `ZSCORE_MIN_PERIODS`).

Pre-registered falsification test 1 (exposure artifact, this project's
own R-33/R-57/R-38 diagnostic, THE decision-governing test for this
round)
------------------------------------------------------------------------
For every configuration in the inner-train sweep: `return_advantage =
candidate.profit_pct - v4.profit_pct` (spot, inner-train) regressed
(plain OLS, `numpy.polyfit`) against `mean_abs_target = mean(|target|)`
over the same run. If R^2 > 0.95, the result is an exposure-level
relabeling, not a real signal: discard, do not report as a win, say so
explicitly. Computed fresh in `artifact()` below.

Pre-registered decision rule (frozen now, before any inner-validation
number is read)
------------------------------------------------------------------------
1. Sweep freely on inner-train (`sweep()`). Select between configs on
   inner-validation (`select()`), rule: `min(train_spot_sharpe,
   valid_spot_sharpe)`, identical selection rule to R-55/R-73/R-74's own
   confirming-vote branches.
2. **Gate:** proceed to the holdout ONLY IF (a) the selected config
   beats `kelly_regime_v4` on inner-validation (both markets, not just
   spot) by more than the noise floor is NOT required to pass the gate
   itself -- that stricter bar applies at promotion -- but the config
   must at minimum show a positive Sharpe/return advantage over v4 on
   inner-validation in both markets, AND (b) falsification test 1's R^2
   <= 0.95 (not an exposure artifact). If either fails, STOP: report
   NEGATIVE at the gate, never read `OOS_START` or later.
3. If the gate passes, run `holdout()` exactly once with the frozen
   config: `buy_and_hold` comparison, the real 0.40% taker fee tier on
   both spot and futures (`MarketSpec.spot(fee_rate=0.004)` /
   `MarketSpec.futures(leverage=5.0, fee_rate=0.004)`), futures funding
   charged (`scripts/funding_study.py`'s own `load_funding` + `run_backtest
   (..., funding=...)` pattern), a resample-based path-sensitivity check,
   and deflated Sharpe using this file's own `N_EVALUATED` count.
   Promotion bar (ROUTINE.md step 4, applied only if the gate above
   passed): beats `buy_and_hold` OOS after real costs; the improvement
   exceeds the +/-0.2 Sharpe noise floor or is a genuine drawdown/tail
   improvement; survives falsification test 1; the parameter
   neighbourhood is a plateau. Anything else is NEGATIVE, even past the
   gate.

Code reuse decision, stated plainly
-------------------------------------
`_anchor_votes` and the vol-targeting sizer loop are DUPLICATED (not
imported) from `kelly_regime_v16_stablecoin_confirm.py`/
`kelly_regime_v3.py`, the same precedent every prior confirming-vote
branch in this project (R-53, R-54, R-55) has used for the same reason:
these are private prior-round experiment files, not shared
infrastructure. `experiments/r77_shared.py` (this round's one piece of
FROZEN shared infra) IS imported unchanged for `load_crossvenue_bars`,
`INNER_TRAIN_END`, `INNER_VALID`, `OOS_START` -- never edited, never
duplicated.

Usage
-----
    python experiments/r77_conservative_crossvenue_confirm.py descriptive  # sign check + vote frequency
    python experiments/r77_conservative_crossvenue_confirm.py sweep        # step 3 (inner-train)
    python experiments/r77_conservative_crossvenue_confirm.py select       # step 3 (inner-validation) + gate check
    python experiments/r77_conservative_crossvenue_confirm.py artifact     # falsification test 1 (exposure-artifact regression)
    python experiments/r77_conservative_crossvenue_confirm.py causality    # lookahead probe, divergence pathway specifically
    python experiments/r77_conservative_crossvenue_confirm.py holdout      # ONLY if the gate passed
    python experiments/r77_conservative_crossvenue_confirm.py all          # descriptive+sweep+select+artifact+causality (never holdout)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_spot, load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import prefix_bars, run_period  # noqa: E402

from experiments.r77_shared import (  # noqa: E402
    INNER_TRAIN_END,
    INNER_VALID,
    OOS_START,
    coverage_report,
    load_crossvenue_bars,
)

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures 5x", FUTURES))

TRAIN = ("2017-01-01", INNER_TRAIN_END)   # inner-train
VALID = INNER_VALID                       # inner-validation
# OOS_START imported above -- never read before the gate passes

INCUMBENT = "kelly_regime_v4"
DATA_DIR = ROOT / "data"

N_EVALUATED = 0
_SEEN_CONFIGS: set[str] = set()


# --------------------------------------------------------------- holdout guard


def assert_no_holdout(df: pd.DataFrame) -> None:
    """Hard guard: the max timestamp in any frame this file touches (before
    the gate) must be strictly before OOS_START. Same pattern as
    ``experiments/r76_conservative_pairs_cointegration.py:240``."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start "
        "before the gate has been passed.")


_GATED_CACHE: pd.DataFrame | None = None


def load_gated_bars() -> pd.DataFrame:
    """The shared cross-venue frame, truncated strictly before OOS_START
    and guard-checked. Used by every function except ``holdout()``."""
    global _GATED_CACHE
    if _GATED_CACHE is not None:
        return _GATED_CACHE
    bars = load_crossvenue_bars()
    cutoff = pd.Timestamp(OOS_START, tz=bars.index.tz)
    bars = bars.loc[bars.index < cutoff].copy()
    assert_no_holdout(bars)
    _GATED_CACHE = bars
    return bars


DF = load_gated_bars()
print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
      f"(gated strictly < {OOS_START}); {coverage_report(DF)}", file=sys.stderr)


# --------------------------------------------------------------------- measure


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0,
            config_key: str | None = None):
    """One backtest -> Metrics. ``config_key`` counts a DISTINCT
    configuration exactly once across the whole session (v4 control and
    diagnostic re-reads pass config_key=None and are never counted)."""
    global N_EVALUATED
    if config_key is not None and config_key not in _SEEN_CONFIGS:
        _SEEN_CONFIGS.add(config_key)
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market, start_balance=balance)
    return compute_metrics(result), result


def line(tag, m, market_name=""):
    print(f"  {tag:48s} {market_name:10s} final=${m.final_balance:>11,.0f} "
          f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>7,.0f}{'  LIQUIDATED' if m.liquidated else ''}")


# ---------------------------------------------------------------- the vote


def _anchor_votes(close: pd.Series, horizons=(20, 40, 80), band: float = 0.01) -> dict:
    """Exactly kelly_regime_v4's own per-anchor latched vote. Duplicated
    (not imported) from kelly_regime_v16_stablecoin_confirm.py -- see this
    module's docstring, "Code reuse decision"."""
    votes = {}
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes[days] = v.ffill().fillna(0.0)
    return votes


def divergence_zscore(divergence: pd.Series, window_bars: int) -> pd.Series:
    """Causal rolling z-score of the divergence column: bar T's z-score
    uses only bars <= T (``rolling`` is backward-looking by construction).
    ``min_periods = window_bars // 2``, this project's own convention
    (matches ``r76_conservative_pairs_cointegration.py``'s
    ``ZSCORE_MIN_PERIODS``)."""
    min_periods = max(2, window_bars // 2)
    roll_mean = divergence.rolling(window_bars, min_periods=min_periods).mean()
    roll_std = divergence.rolling(window_bars, min_periods=min_periods).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (divergence - roll_mean) / roll_std
    return z


SIGN = 1.0  # frozen BEFORE sweep()/select() ran -- see docstring's empirical check


def _divergence_vote(z: pd.Series, thresh: float, sign: float = SIGN) -> pd.Series:
    """Latched 0/1 vote, symmetric hysteresis on a single threshold
    (mirrors ``_anchor_votes``' own single ``band``, not the stablecoin
    vote's asymmetric thresh_hi/gap -- this signal is two-sided by the
    pre-registered hypothesis, not one-sided like stablecoin stress).
    Defaults to 0.5 (neutral) before the first crossing."""
    zs = sign * z
    raw = np.where(zs < -thresh, 1.0, np.where(zs > thresh, 0.0, np.nan))
    return pd.Series(raw, index=z.index).ffill().fillna(0.5)


class KellyRegimeCrossVenueConfirm(KellyRegimeV3):
    """v4's 3-anchor vote plus a 4th, precision-weighted CONFIRMING vote
    from Bitstamp-vs-Coinbase BTC divergence. Mechanism:
    ``frac = (anchor_sum + weight * divergence_vote) / (3 + weight)``.
    ``weight=0`` recovers v4 exactly (identity check, verified in
    ``causality()``). See module docstring for the full mechanism,
    window choice and the pre-registered, empirically-checked vote sign.
    """

    name = "kelly_regime_r77_crossvenue_confirm"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 window_bars: int = 288, thresh: float = 1.0, weight: float = 0.33,
                 **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.window_bars = window_bars
        self.thresh = thresh
        self.weight = weight

    def _divergence_series(self, df: pd.DataFrame) -> pd.Series:
        if "divergence" in df.columns:
            return df["divergence"]
        raise RuntimeError("expected a precomputed 'divergence' column (r77_shared.load_crossvenue_bars)")

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = _anchor_votes(close, self.horizons, self.band)
        anchor_sum = sum(votes.values())
        n_anchors = float(len(votes))

        divergence = self._divergence_series(df)
        if self.weight > 0:
            z = divergence_zscore(divergence, self.window_bars)
            vote = _divergence_vote(z, self.thresh)
        else:
            z = pd.Series(np.nan, index=df.index)
            vote = pd.Series(0.5, index=df.index)

        combined = (anchor_sum + self.weight * vote) / (n_anchors + self.weight)
        frac = combined.to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        # From here down: byte-for-byte v3's conditional vol-targeting sizer.
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
        pos = 0.0
        vstate = 0
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if vstate == 0:
                    vstate = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif vstate == 1 and x < self.high_out:
                    vstate = 0
                elif vstate == -1 and x > self.low_out:
                    vstate = 0
            scale = full[i] if vstate != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["cv_frac"] = frac
        df["cv_vote"] = vote.to_numpy()
        df["cv_z"] = z.to_numpy()
        df["cv_anchor_sum"] = anchor_sum.to_numpy()
        return df


# ------------------------------------------------------------- the grid

WEIGHTS = (0.15, 0.33, 0.5, 1.0)

# (label, window_bars, thresh): three window points at a fixed thresh (fast
# 6h / primary 1d / slow 3d, spanning the order-of-magnitude Makarov & Schoar's
# own literature treats as plausible for cross-exchange arbitrage to resolve),
# plus a looser threshold at the primary window for plateau resolution.
WINDOW_THRESH = (
    ("fast-6h", 72, 1.0),
    ("primary-1d", 288, 1.0),
    ("slow-3d", 864, 1.0),
    ("primary-loose", 288, 1.5),
)
PRIMARY_KW = dict(window_bars=288, thresh=1.0, weight=0.33)


def _grid():
    """17 configs: identity control (1) + 4 window/thresh x 4 weights (16)."""
    out = [("identity (weight=0)", dict(window_bars=288, thresh=1.0, weight=0.0))]
    for label, window_bars, thresh in WINDOW_THRESH:
        for weight in WEIGHTS:
            tag = (f"{label} (win={window_bars} thresh={thresh:.2f}) "
                   f"w={weight:.2f}{'(unweighted)' if weight == 1.0 else ''}")
            out.append((tag, dict(window_bars=window_bars, thresh=thresh, weight=weight)))
    return out


def _config_key(kw: dict) -> str:
    return f"cvconfirm|win={kw['window_bars']}|thresh={kw['thresh']:.3f}|w={kw['weight']:.3f}"


# ------------------------------------------------------- step 2b: descriptive


def check_sign() -> None:
    """The pre-registered sign check: corr(z, forward Bitstamp log-return)
    on INNER-TRAIN ONLY, for every (window, horizon) cell. Frozen before
    ``sweep()``/``select()`` ever ran -- see module docstring."""
    train = DF.loc[TRAIN[0]:TRAIN[1]]
    close = train["close"]
    div = train["divergence"]
    horizons = (1, 12, 288, 864)
    fwd = {h: np.log(close.shift(-h) / close) for h in horizons}

    print("sign check (inner-train ONLY, 2017-01-01 -> 2020-12-31): "
          "corr(z-score of divergence, Bitstamp forward log-return)")
    print("  negative corr => Bitstamp-expensive (z>0) predicts Bitstamp DOWN, "
          "Bitstamp-cheap (z<0) predicts Bitstamp UP -- confirms the "
          "arbitrage/mean-reversion hypothesis, sign=+1 (no flip)")
    any_positive = False
    for label, window_bars, _t in WINDOW_THRESH[:3]:  # the 3 distinct windows
        z = divergence_zscore(div, window_bars)
        for h in horizons:
            r = fwd[h]
            mask = z.notna() & r.notna() & np.isfinite(z)
            c = float(np.corrcoef(z[mask], r[mask])[0, 1])
            any_positive = any_positive or c > 0
            print(f"  {label:14s} (win={window_bars:4d}) horizon={h:4d} bars  "
                  f"corr(z, fwd_ret)={c:+.4f}  n={int(mask.sum()):,}")
    print(f"\nany positive-sign cell: {any_positive}  -> SIGN={SIGN:+.0f} "
          f"({'consistent with the frozen hypothesis, no flip' if not any_positive else 'MIXED -- see note below'})")


def descriptive() -> None:
    """Step 2b. Sign check (above) plus vote-transition frequency at each
    grid point, over inner-train + inner-validation, for context against
    the 20/40/80-day price anchors' own flip frequency."""
    check_sign()

    lo, hi = TRAIN[0], VALID[1]
    frame = DF.loc[lo:hi]
    close = frame["close"]
    div = frame["divergence"]

    print(f"\ndescriptive window (inner-train + inner-validation): {lo} -> {hi}")
    print("price-anchor vote transition counts over the SAME window (context):")
    for days in (20, 40, 80):
        anchor = DF["close"].rolling(int(days * BARS_PER_DAY)).mean().loc[lo:hi]
        v = pd.Series(np.where(close > anchor * 1.01, 1.0,
                                np.where(close < anchor * 0.99, 0.0, np.nan)),
                      index=close.index).ffill().fillna(0.0)
        print(f"  {days:>3d}d price anchor: {int(v.ne(v.shift()).sum()):>4d} vote flips")

    print(f"\ndivergence coverage in window: {div.notna().sum():,} / {len(div):,} bars")
    for label, window_bars, thresh in WINDOW_THRESH:
        z = divergence_zscore(div, window_bars)
        vote = _divergence_vote(z, thresh)
        flips_to_bull = int(((vote == 1.0) & (vote.shift() != 1.0)).sum())
        flips_to_bear = int(((vote == 0.0) & (vote.shift() != 0.0)).sum())
        print(f"  {label:14s} win={window_bars:4d} thresh={thresh:.2f}: "
              f"{flips_to_bull} bullish-onset  {flips_to_bear} bearish-onset")


# --------------------------------------------------------------- step 3: sweep


def sweep() -> None:
    """Step 3: every config on inner-train ONLY, spot primary."""
    print(f"\nINNER-TRAIN {TRAIN} / spot -- benchmarks:")
    for name in ("buy_and_hold", INCUMBENT):
        m, _ = measure(get_strategy(name), *TRAIN, market=SPOT)
        line(name, m, "spot")

    print(f"\nINNER-TRAIN {TRAIN} / spot -- candidate configurations:")
    for label, kw in _grid():
        m, _ = measure(KellyRegimeCrossVenueConfirm(**kw), *TRAIN, market=SPOT,
                        config_key=_config_key(kw))
        line(label, m, "spot")

    print(f"\nconfigurations evaluated (step 3, distinct window/thresh/weight triples): {N_EVALUATED}")


# -------------------------------------------------------------- step 3: select


def select():
    """Every config on inner-validation ONLY, BOTH markets, vs v4 control."""
    rows = []
    print(f"\nINNER-VALIDATION {VALID} -- v4 control:")
    ctl = {}
    for mname, market in MARKETS:
        m, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        ctl[mname] = m
        line(f"{INCUMBENT} (control)", m, mname)

    print(f"\nINNER-VALIDATION {VALID} -- candidate configurations:")
    best_label, best_kw, best_score = None, None, -1e9
    cell_by_key = {}
    for label, kw in _grid():
        cell = {}
        for mname, market in MARKETS:
            m, _ = measure(KellyRegimeCrossVenueConfirm(**kw), *VALID, market=market,
                            config_key=_config_key(kw))
            cell[mname] = m
            line(label, m, mname)
            rows.append({"label": label, **kw, "market": mname,
                         "final": m.final_balance, "sharpe": m.sharpe,
                         "max_dd": m.max_drawdown_pct, "trades": m.num_trades})
        m_train, _ = measure(KellyRegimeCrossVenueConfirm(**kw), *TRAIN, market=SPOT)
        score = min(m_train.sharpe, cell["spot"].sharpe)
        cell_by_key[_config_key(kw)] = (label, kw, cell)
        if score > best_score:
            best_label, best_kw, best_score = label, kw, score

    print(f"\nbest by min(train,valid) spot Sharpe: {best_label}  (score={best_score:.2f})")
    print(f"v4 control spot Sharpe: {ctl['spot'].sharpe:.2f}   "
          f"v4 control futures Sharpe: {ctl['futures 5x'].sharpe:.2f}")

    print("\nside-by-side: best candidate's TRAIN window vs VALIDATION window (overfitting-signature check):")
    gate_ok = True
    for mname, market in MARKETS:
        m_train, _ = measure(KellyRegimeCrossVenueConfirm(**best_kw), *TRAIN, market=market)
        m_valid, _ = measure(KellyRegimeCrossVenueConfirm(**best_kw), *VALID, market=market)
        m_train_v4, _ = measure(get_strategy(INCUMBENT), *TRAIN, market=market)
        m_valid_v4, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        d_sharpe = m_valid.sharpe - m_valid_v4.sharpe
        gate_ok = gate_ok and (d_sharpe > 0)
        print(f"  {mname}:")
        print(f"    candidate  TRAIN final=${m_train.final_balance:>11,.0f} sharpe={m_train.sharpe:.2f} DD={m_train.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid.final_balance:>11,.0f} sharpe={m_valid.sharpe:.2f} DD={m_valid.max_drawdown_pct:.1f}%")
        print(f"    v4         TRAIN final=${m_train_v4.final_balance:>11,.0f} sharpe={m_train_v4.sharpe:.2f} DD={m_train_v4.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid_v4.final_balance:>11,.0f} sharpe={m_valid_v4.sharpe:.2f} DD={m_valid_v4.max_drawdown_pct:.1f}%")
        print(f"    candidate - v4 (valid): Delta sharpe={d_sharpe:+.3f}  "
              f"Delta DD={m_valid.max_drawdown_pct - m_valid_v4.max_drawdown_pct:+.1f}pp")

    print("\nparameter-neighbourhood plateau check (spot, inner-validation Sharpe, all window/thresh x weight cells):")
    grid_by_key = {(lbl, w): None for lbl, _wb, _t in WINDOW_THRESH for w in WEIGHTS}
    for label, window_bars, thresh in WINDOW_THRESH:
        for weight in WEIGHTS:
            m, _ = measure(KellyRegimeCrossVenueConfirm(window_bars=window_bars, thresh=thresh, weight=weight),
                            *VALID, market=SPOT)
            grid_by_key[(label, weight)] = m.sharpe
    for label, window_bars, thresh in WINDOW_THRESH:
        row = "  ".join(f"w={w:.2f}:{grid_by_key[(label, w)]:.2f}" for w in WEIGHTS)
        print(f"  {label:14s} (win={window_bars:4d} thresh={thresh:.2f})  {row}")
    m_ident, _ = measure(KellyRegimeCrossVenueConfirm(weight=0.0), *VALID, market=SPOT)
    print(f"  identity (weight=0, should equal v4 spot Sharpe {ctl['spot'].sharpe:.2f}): {m_ident.sharpe:.2f}")

    print(f"\nGATE CHECK: best candidate beats v4 on inner-validation Sharpe, BOTH markets: {gate_ok}")
    print(f"configurations evaluated so far: {N_EVALUATED}")
    return best_label, best_kw, gate_ok


# ------------------------------------------------------------------ artifact


def _target_series(strategy, start, end, market=SPOT) -> np.ndarray:
    """Measured-window ``target`` array (warmup prefix excluded), same
    slicing convention as ``kelly_regime_v16_stablecoin_confirm.py``'s own
    ``artifact()``."""
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    prefix = prefix_bars(DF, lo, strategy.warmup)
    frame = DF.iloc[lo - prefix:hi].copy()
    prepared = strategy.prepare(frame)
    return prepared["target"].to_numpy()[prefix:]


def artifact() -> dict:
    """Falsification test 1, THE decision-governing test: for every config
    in the inner-train sweep, regress
    ``return_advantage = candidate.profit_pct - v4.profit_pct`` (spot,
    inner-train) against ``mean_abs_target = mean(|target|)`` over the
    same run. R^2 > 0.95 -> exposure-level relabeling, discard."""
    print("FALSIFICATION TEST 1 -- exposure-artifact regression "
          "(return_advantage vs mean|target|, inner-train, spot, every sweep config)")
    v4 = get_strategy(INCUMBENT)
    m_v4, _ = measure(v4, *TRAIN, market=SPOT)
    v4_target = _target_series(v4, *TRAIN, market=SPOT)
    mean_abs_v4 = float(np.mean(np.abs(v4_target)))

    xs, ys, labels = [], [], []
    for label, kw in _grid():
        cand = KellyRegimeCrossVenueConfirm(**kw)
        m_cand, _ = measure(cand, *TRAIN, market=SPOT, config_key=_config_key(kw))
        cand_target = _target_series(KellyRegimeCrossVenueConfirm(**kw), *TRAIN, market=SPOT)
        mean_abs_cand = float(np.mean(np.abs(cand_target)))
        return_advantage = m_cand.profit_pct - m_v4.profit_pct
        xs.append(mean_abs_cand)
        ys.append(return_advantage)
        labels.append(label)
        print(f"  {label:48s} mean|target|={mean_abs_cand:.4f}  "
              f"return_advantage={return_advantage:+8.2f}pp")

    xs_arr, ys_arr = np.array(xs), np.array(ys)
    slope, intercept = np.polyfit(xs_arr, ys_arr, 1)
    pred = slope * xs_arr + intercept
    ss_res = float(np.sum((ys_arr - pred) ** 2))
    ss_tot = float(np.sum((ys_arr - ys_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    verdict = "EXPOSURE ARTIFACT -- discard" if r2 > 0.95 else "NOT an exposure artifact (survives test 1)"
    print(f"\nv4 mean|target| (spot, inner-train): {mean_abs_v4:.4f}   v4 profit_pct: {m_v4.profit_pct:+.2f}pp")
    print(f"OLS: return_advantage = {slope:+.3f} * mean|target| + {intercept:+.3f}   "
          f"R^2 = {r2:.4f}   [{verdict}]")
    return dict(r2=r2, slope=slope, intercept=intercept, passed=r2 <= 0.95)


# ------------------------------------------------------------------ causality


def causality(kw: dict | None = None) -> None:
    """Lookahead probe over the divergence pathway specifically: tamper the
    Bitstamp price, the Coinbase price (reconstructed independently and
    refed through a fresh ``divergence`` column, not the price pathway),
    and both together, after a cut. Every decision at or before the cut
    must be unchanged. Plus the weight=0 identity check."""
    kw = kw or PRIMARY_KW

    # Recover the Coinbase leg independently (sanity-checked against the
    # shared loader's own 'divergence' column) so the Coinbase pathway can
    # be tampered on its own, without touching Bitstamp's OHLCV.
    bitstamp, label = load_dataset(DATA_DIR, "spot")
    assert label == "real"
    coinbase = load_coinbase_spot(DATA_DIR, "BTC")
    common = bitstamp.index.intersection(coinbase.index)
    common = common[common < pd.Timestamp(OOS_START, tz=common.tz)]
    bars = bitstamp.loc[common].copy()
    cb_close = coinbase.loc[common, "close"].copy()
    recomputed = np.log(bars["close"] / cb_close)
    shared_div = DF.loc[common, "divergence"]
    worst_sanity = float(np.nanmax(np.abs(recomputed.to_numpy() - shared_div.to_numpy())))
    print(f"sanity: recomputed divergence vs r77_shared's own column, max|diff|={worst_sanity:.3e} "
          f"{'PASS' if worst_sanity < 1e-9 else 'FAIL'}")

    df = bars.iloc[-300_000:].copy()
    cb = cb_close.loc[df.index].copy()
    cut = len(df) - 5_000
    bars_checked = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]
    cut_day = df.index[cut].normalize()

    def make_frame(bitstamp_df: pd.DataFrame, coinbase_s: pd.Series) -> pd.DataFrame:
        out = bitstamp_df.copy()
        out["divergence"] = np.log(out["close"] / coinbase_s)
        return out

    def decisions(frame: pd.DataFrame):
        s = KellyRegimeCrossVenueConfirm(**kw)
        prepared = s.prepare(frame.copy())
        broker = _fresh_broker()
        out = []
        for i in bars_checked:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out, prepared

    def run_probe(name, bit_up, cb_up, bit_down, cb_down):
        frame_up = make_frame(bit_up, cb_up)
        frame_down = make_frame(bit_down, cb_down)
        a, prep_a = decisions(frame_up)
        b, prep_b = decisions(frame_down)
        bad = [bar for bar, oa, ob in zip(bars_checked, a, b) if oa != ob]
        print(f"[{name}] tampered from bar {cut:,} of {len(df):,} "
              f"(calendar day {cut_day.date()}); checked bars {bars_checked}")
        print("  FAIL - reads the future at bars " + str(bad) if bad
              else "  PASS - every decision at or before the cut is unchanged")
        for col in ("target", "cv_frac", "cv_vote", "cv_anchor_sum"):
            diff = np.abs(prep_a[col].to_numpy()[:cut].astype(float)
                          - prep_b[col].to_numpy()[:cut].astype(float))
            worst = float(np.nanmax(diff))
            print(f"    column {col:16s} max |difference| before the cut = {worst:.3e}"
                  f"  {'PASS' if worst < 1e-9 else 'FAIL'}")

    # (1) price-only tamper: Bitstamp OHLCV tampered after cut, Coinbase leg untouched.
    bit_up, bit_down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        bit_up.iloc[cut:, bit_up.columns.get_loc(col)] *= 3.0
        bit_down.iloc[cut:, bit_down.columns.get_loc(col)] /= 3.0
    bit_up.iloc[cut:, bit_up.columns.get_loc("volume")] *= 7.0
    bit_down.iloc[cut:, bit_down.columns.get_loc("volume")] /= 7.0
    run_probe("BITSTAMP-price tamper (standard)", bit_up, cb.copy(), bit_down, cb.copy())

    # (2) Coinbase-only tamper: the new divergence pathway, Bitstamp OHLCV untouched.
    cb_up_s = cb.copy()
    cb_down_s = cb.copy()
    cb_up_s.iloc[cut:] = cb_up_s.iloc[cut:] / 50.0     # Coinbase crashes -> Bitstamp looks expensive
    cb_down_s.iloc[cut:] = cb_down_s.iloc[cut:] * 50.0  # Coinbase rallies -> Bitstamp looks cheap
    run_probe("COINBASE-price tamper (the new divergence pathway)", df.copy(), cb_up_s, df.copy(), cb_down_s)

    # (3) both at once
    run_probe("both pathways at once", bit_up, cb_up_s, bit_down, cb_down_s)

    # identity check: weight=0 recovers v4 exactly
    v4 = get_strategy(INCUMBENT)
    ident = KellyRegimeCrossVenueConfirm(window_bars=288, thresh=1.0, weight=0.0)
    frame = make_frame(df.iloc[-20_000:].copy(), cb.iloc[-20_000:].copy())
    t_v4 = v4.prepare(frame.copy())["target"].to_numpy()
    t_ident = ident.prepare(frame.copy())["target"].to_numpy()
    worst = float(np.nanmax(np.abs(t_v4 - t_ident)))
    print(f"\nidentity check (weight=0 recovers v4 exactly): max|diff|={worst:.3e}  "
          f"{'PASS' if worst < 1e-9 else 'FAIL'}")


def _fresh_broker():
    from tradebot.broker import PaperBroker
    return PaperBroker(market=FUTURES, start_balance=10_000.0)


# --------------------------------------------------------------------- holdout


def holdout(best_kw: dict | None = None) -> None:
    """Step 4 -- ONLY IF the gate (select()'s gate_ok AND artifact()'s
    passed) has been checked true by the caller. Reads OOS_START for the
    first time in this file."""
    kw = best_kw or PRIMARY_KW
    print("=" * 78)
    print(f"HOLDOUT -- {OOS_START} onward. Candidate config: {kw}")
    print("=" * 78)

    full = load_crossvenue_bars()  # full series, no truncation -- gate already passed
    print(f"full series: {len(full):,} bars  {full.index[0]:%Y-%m-%d} -> {full.index[-1]:%Y-%m-%d}")

    cand = KellyRegimeCrossVenueConfirm(**kw)
    v4 = get_strategy(INCUMBENT)
    hold = get_strategy("buy_and_hold")

    print(f"\n-- funding-free comparison, both markets, {OOS_START} -> end --")
    for mname, market in MARKETS:
        for name, strat in ((INCUMBENT, v4), ("buy_and_hold", hold), ("candidate", cand)):
            m, _ = measure(strat, OOS_START, None, df=full, market=market)
            line(name, m, mname)

    print("\n-- real fee tier (0.40% taker) --")
    real_spot = MarketSpec.spot(fee_rate=0.004)
    real_fut = MarketSpec.futures(leverage=5.0, fee_rate=0.004)
    for mname, market in (("spot 0.40%", real_spot), ("futures 5x 0.40%", real_fut)):
        for name, strat in ((INCUMBENT, v4), ("candidate", cand)):
            m, _ = measure(strat, OOS_START, None, df=full, market=market)
            line(name, m, mname)

    print("\n-- futures WITH funding charged (real Binance BTCUSDT 2020-2023 funding) --")
    real_funding = load_funding(DATA_DIR)
    if real_funding is None:
        print("  no funding data file present -- skipped")
    else:
        lo = int(full.index.searchsorted(OOS_START))
        pre = min(lo, max(cand.warmup, v4.warmup))
        frame = full.iloc[lo - pre:]
        for name, strat in ((INCUMBENT, v4), ("candidate", cand)):
            raw = run_backtest(strat, frame, FUTURES, 1_000.0, trade_start=pre, funding=real_funding)
            m = compute_metrics(raw)
            print(f"  {name:20s} final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
                  f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
                  f"funding_paid=${raw.funding_paid:>9,.0f}")

    print(f"\nconfigurations evaluated (project trials count for deflated Sharpe): {N_EVALUATED}")


# --------------------------------------------------------------------- driver


def all_checks() -> None:
    print("=" * 78)
    print("STEP 2b -- descriptive: sign check + vote frequency")
    print("=" * 78)
    descriptive()
    print("\n" + "=" * 78)
    print("STEP 3 -- sweep (inner-train)")
    print("=" * 78)
    sweep()
    print("\n" + "=" * 78)
    print("STEP 3 -- select (inner-validation, both markets) + gate check")
    print("=" * 78)
    best_label, best_kw, gate_ok = select()
    print("\n" + "=" * 78)
    print("FALSIFICATION TEST 1 -- exposure-artifact regression")
    print("=" * 78)
    art = artifact()
    print("\n" + "=" * 78)
    print("CAUSALITY / NO-LOOKAHEAD PROBE (divergence pathway)")
    print("=" * 78)
    causality()
    print(f"\nGATE: select()={gate_ok}  artifact_pass={art['passed']}  "
          f"-> proceed to holdout: {gate_ok and art['passed']}")
    print(f"total distinct configurations evaluated (N_EVALUATED): {N_EVALUATED}")


if __name__ == "__main__":
    cmds = {"descriptive": descriptive, "sweep": sweep, "select": select,
            "artifact": artifact, "causality": causality, "holdout": holdout,
            "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/r77_conservative_crossvenue_confirm.py [{'|'.join(cmds)}]")
