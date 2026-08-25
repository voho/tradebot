"""Shared, read-only utilities and pre-registration for the R-136 round (08-25).

DIRECTION, in one sentence: substitute `kelly_regime_v4`'s single fast
realized-vol estimator (an 8-day EWM of 5m log returns) -- the series that
drives BOTH the breakout-state sizing (`full = target_vol / vol`) and the
regime-state machine itself (`ratio = vol / slow`) -- with an a-priori,
non-fitted multi-timescale (Corsi 2009 HAR-structured) blend, in two
constructions: pure realized-vol components (conservative) and the same
blend augmented with Deribit's DVOL implied-volatility index (novel),
testing whether a genuinely new external data channel helps when it is fed
into the SCALE estimator rather than used as a vote or gate.

**Why this exact axis, and why now.** Per docs/LEDGER.md's 08-25 backlog
re-ranking (after R-135), the ranked backlog is empty of anything but B-06
(forward paper trading, already running unattended). This round targets
`kelly_regime_v4` directly -- this project's current leader ($66.8K spot /
$156.2K futures_5x) -- rather than moving to a weaker off-backlog object,
because the operating task for this session was explicitly to improve the
best strategy. The single-asset axis's closed lists (19+ INFO, 28+ SIZE, ERR
across 5 uncertainty notions, 11 regime-timing mechanisms, 4 N-approx-3
procedures, 5 COST-model families -- see the 08-25-after-R-135 backlog
re-ranking) cover the VOTE (`frac`) and the vol-TARGETING architecture
(continuous vs conditional, drawdown cushions, CRRA, response shape) exhaustively. A
survey of docs/RESEARCH.md's own findings 1-11 and docs/LEDGER.md's ruled-out
table turned up exactly one untouched slot: the raw vol ESTIMATOR feeding
both `full` and `ratio` has only ever been an EWM of realized returns, never
a forecast blending multiple horizons or an external (implied-vol) input, and
the one prior attempt at "better vol forecasting" (R-08, 08-15) pre-dates
`kelly_regime_v3`/`v4`'s conditional (band-gated, hysteresis) targeting
architecture by several rounds -- it tested a continuous-targeting mechanism
that no longer exists in any registered strategy.

**The prior, stated honestly before any code runs.** R-08 found a genuinely
better vol forecast (a timescale blend, 8% better on QLIKE) made the OLD
continuous-targeting mechanism WORSE ($52K vs $115K, sign-inverting) because
BTC's inverse leverage effect (Baur & Dimpfl 2018, Economics Letters 173)
makes high-vol states the HIGHEST-forward-Sharpe ones, and a better forecast
de-levers into them more promptly. `kelly_regime_v3`'s whole reason to exist
(Bongaerts, Kang & van Dijk 2020, FAJ 76(4)) is that CONDITIONAL (extremes-
only, hysteresis-latched) targeting resists exactly this failure mode by
construction -- it only re-sizes off the fast estimate during a breakout
state, holding steady otherwise. Whether that resistance survives a forecast
that is not just "better" (R-08's framing) but structurally different
(multi-timescale, possibly IV-augmented) is this round's actual question, and
the modal a-priori expectation, stated now, is that R-08's finding
generalizes and the round closes NEGATIVE -- which is still a real answer:
it would be the first time this project confirms a mechanism explicitly
survives a documented failure mode from a materially different angle, closing
the "vol-estimator substitution" sub-branch by name for the first time on the
CURRENT (v3/v4) architecture rather than the pre-v3 one R-08 tested.

Literature:
- Corsi, F. (2009), "A Simple Approximate Long-Memory Model of Realized
  Volatility," Journal of Financial Econometrics 7(2), 174-196 -- the HAR
  structure: next-period volatility as a function of trailing daily, weekly
  and monthly realized-vol components, motivated by the heterogeneous-market
  hypothesis (agents acting on different horizons), already the explicit
  citation for v4's own doubling anchor ladder (docs/STRATEGIES.md). This
  round applies the SAME structural idea to the vol axis, not the trend axis,
  for the first time.
- Busch, T., Christensen, B.J. & Nielsen, M.O. (2011), "The Role of Implied
  Volatility in Forecasting Future Realized Volatility and Jumps in Foreign
  Exchange, Stock, and Bond Markets," Journal of Econometrics 160(1), 48-57
  -- HAR-RV forecasts improve when augmented with an implied-vol term (their
  HAR-RV-CJ-IV family); the novel branch's motivating citation for adding
  DVOL as a fourth component rather than a replacement.
- Bollerslev, T., Tauchen, G. & Zhou, H. (2009), RFS 22(11) -- variance risk
  premium framing, already used by R-73/R-135 for DVOL as a directional vote;
  not reused here, since this round uses DVOL as a LEVEL (an alternative vol
  estimate), not a premium (implied minus realized, a directional signal).
- Patton, A.J. (2011), Journal of Econometrics 160(1) -- QLIKE, this
  project's own standing loss function for vol-forecast quality (R-08,
  RESEARCH.md finding 2's neighbour), reused here unchanged to check the new
  estimators are genuinely better/worse forecasts, not merely different.

**A-priori, not fitted -- and why.** Corsi's own HAR is normally OLS-fitted.
This round instead uses a fixed, EQUAL-WEIGHT blend of the components (an
un-fitted mean), for two reasons stated now rather than discovered after
looking: (1) it matches this project's own stated preference for structure
over fit (v4's doubling ladder, chosen "for its structure rather than
fitted" -- docs/STRATEGIES.md); (2) DVOL coverage starts 2021-03-24, INSIDE
inner-validation and entirely ABSENT from inner-train (2017-2020), so an
OLS fit of an IV coefficient on inner-train is not even possible -- an
equal-weight blend sidesteps a fit-window argument entirely rather than
papering over it.

**Mechanism, one sentence per branch:**
- CONSERVATIVE (`r136_conservative_har_rv_scale.py`): replace the vol
  estimator with the equal-weight mean of three REALIZED-vol components only
  (daily/weekly/monthly trailing windows over 5m squared log returns,
  annualized) -- no new data source, pure re-engineering of the existing
  OHLCV-only estimator.
- NOVEL (`r136_novel_har_iv_scale.py`): the same three components PLUS a
  fourth, causally-aligned DVOL level (already in annualized-vol-fraction
  units, `close / 100` -- confirmed empirically, mean 60.2 against this
  project's own `target_vol=0.55` default), equal-weighted with the other
  three, falling back to the conservative branch's three-way mean whenever
  DVOL is unavailable (NaN, all bars before 2021-03-24) -- never fit,
  backfilled or extrapolated.

Both inherit `HARVolMixin` below, which is `KellyRegimeV3.prepare()` copied
verbatim except the single line computing `vol` is replaced by a call to
`self._vol_series(df, r)` -- every other line (vote, `slow`, `ratio`, the
state machine, `full`/`steady`, the deadband) is bit-for-bit identical to
the registered `kelly_regime_v3`/`v4`, so any measured difference is
attributable to the vol estimator alone.

**Not a duplicate of:**
- R-08 (08-15, "better volatility forecasting"): different object entirely
  (pre-v3/v4's continuous-targeting mechanism, since removed) and a
  different, single-component (not HAR-structured) forecast; this round's
  own module docstring states R-08's finding as the prior it is testing
  against on the CURRENT conditional-targeting architecture.
- R-09 (range volatility estimators -- Parkinson/GK/RS/YZ): a different
  ESTIMATOR family (intrabar range vs multi-timescale realized-vol
  combination) answering a different question (does using each bar's
  H/L/O/C beat close-to-close) -- orthogonal to this round's question (does
  combining several TRAILING WINDOWS, one of them externally priced, beat
  one EWM window).
- R-73/R-81/R-135 (DVOL/positioning as `kelly_regime_v4` confirming votes or
  `hedge_experts` panel members): DVOL used there as a directional
  vote/gate/expert; here it is a LEVEL substituted into the vol estimator
  that feeds `scale`, never touching `frac`/the vote at all -- the INFO-axis
  distinction this project has used before (R-132 vs R-81) to separate a
  standalone-signal attempt from a structurally different application point.
- R-62/R-87 (`frac` vs `scale` factor decomposition): confirms `scale`
  carries none of v4's drawdown-replication signature and is the correct,
  already-identified place to test a vol-forecast change without touching
  the vote.

**What would make this fail, named now, before any code:**
1. R-08's inversion reproduces: better/different vol forecast still
   de-levers faster into BTC's inverse-leverage high-Sharpe states even
   under conditional targeting, net-negative despite the mechanism's
   theoretical resistance to it. This is the SINGLE MOST LIKELY outcome per
   the prior stated above.
2. NO-OP: the state machine's hysteresis already absorbs most of the fast
   estimator's noise (the whole point of `high_in`/`high_out`/`low_in`/
   `low_out`), so any smoother/different vol series changes almost nothing
   -- a plateau at zero, not a real test.
3. Novel-branch-specific -- COVERAGE ARTIFACT: any apparent edge of NOVEL
   over CONSERVATIVE that is not concentrated in the DVOL-covered window
   (2021-03-24 onward) cannot be coming from DVOL, since the two branches
   are literally identical (three-way mean) before that date by
   construction -- checked directly (gate B7 below).
4. BTC-PASS/ETH-INVERT: this project's now nine-plus constructions across
   objects and axes share this signature; B4 is a real, decisive test.

**Falsification test, pre-registered:** does the QLIKE-superior candidate
(if either is QLIKE-superior; checked descriptively first) still lose to
`kelly_regime_v4` on risk-adjusted return, reproducing R-08's inversion on
the modern architecture? A YES closes this sub-branch definitively; a NO
(a candidate that is a better forecast AND does not lose) would be the
first time this project has found ANY vol-forecast change that helps.

**Decision rule, pre-registered, before any branch's code was run:**
PROMOTE-candidate (as a new variant, `kelly_regime_v5`, alongside the
unchanged incumbent, matching the v2/v3/v4 precedent of never overwriting a
registered strategy's own record) only if ALL of:
  (a) causal-truncation probe passes;
  (b) B1 -- both markets, `full_inner` AND `inner_val` both show `d_log_return`
      point estimate > 0 against frozen `kelly_regime_v4`, with at least one
      of the two periods' 95% CI excluding zero on at least one market;
  (c) B3 (plateau across the HAR window-neighbourhood grid: a majority of
      variants agree in sign with the frozen candidate);
  (d) B4 (sign replicates on ETH spot, inner-validation);
  (e) B5 (no sign flip at the 0.40% fee tier);
  (f) B6 (MANDATORY, R-33-style risk match) -- time-in-market and realized
      volatility reported for candidate vs baseline on every decisive B1
      cell, and the candidate's high-vol-quartile mean exposure vs baseline's
      is reported explicitly so a reviewer can judge whether any gain is a
      pure de-leveraging artifact rather than a timing improvement;
  (g) (NOVEL only, additional) B7 -- any `d_log_return` advantage of NOVEL
      over CONSERVATIVE is concentrated in the DVOL-covered sub-window
      (2021-03-24 onward), not present at comparable magnitude before it.
Anything else is NEGATIVE. No bar at or after `OOS_START = 2023-01-01` may
be read by either branch unless (b)-(g) all clear first; if they do, the
operator (not either branch) makes exactly one holdout consultation and
records it as such.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import (  # noqa: E402
    align_dvol_causal,
    load_coinbase_eth_spot,
    load_dataset,
    load_dvol_index,
)
from tradebot.inference import daily_returns, paired_bootstrap, total_log_return  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# ----------------------------------------------------------------------
# Splits. Identical convention to every prior round on this object.
# ----------------------------------------------------------------------
INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"  # do not read unless the pre-registered gates clear
DVOL_COVERAGE_START = "2021-03-24"  # first DVOL row (empirically confirmed)

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT_HIGH_FEE = MarketSpec.spot(fee_rate=0.0040)      # B5: 0.40% taker tier
FUTURES_HIGH_FEE = MarketSpec.futures(leverage=5.0, fee_rate=0.0040)

B1_PERIODS = [
    ("inner_train", INNER_TRAIN_START, INNER_TRAIN_END),
    ("full_inner", INNER_TRAIN_START, INNER_VAL_END),
    ("inner_val", INNER_VAL_START, INNER_VAL_END),
]

# B3: neighbourhood of HAR window choices (bars = days * BARS_PER_DAY),
# the default (1, 5, 22) is Corsi's own daily/weekly/monthly convention.
B3_WINDOW_SETS_DAYS = [(1, 5, 22), (1, 5, 20), (1, 10, 22), (2, 5, 22), (1, 5, 30)]


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {OOS_START}")


def load_btc_train(kind: str = "spot"):
    df, label = load_dataset(ROOT / "data", kind)
    train = df.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(train)
    return train, label


def load_eth_train():
    eth = load_coinbase_eth_spot(ROOT / "data")
    assert eth is not None, "ETH spot data not committed"
    eth = eth.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(eth)
    return eth


def load_dvol_causal_train(bars: pd.DataFrame) -> pd.Series:
    """Causally-aligned DVOL close, rescaled to annualized-vol-FRACTION units
    (raw index is percentage points, e.g. 60.2 == 0.602), reindexed onto
    ``bars``. NaN before DVOL_COVERAGE_START, by construction (align_dvol_causal
    never backfills)."""
    dvol = load_dvol_index(ROOT / "data")
    assert dvol is not None, "DVOL data not committed"
    aligned = align_dvol_causal(dvol[["close"]], bars)
    return (aligned["close"] / 100.0).rename("dvol")


# ----------------------------------------------------------------------
# HAR components, shared bit-for-bit by both branches.
# ----------------------------------------------------------------------

def compute_rv_components(df: pd.DataFrame, windows_days=(1, 5, 22)) -> pd.DataFrame:
    """Causal daily/weekly/monthly realized-vol components (Corsi 2009 HAR
    structure). Each window's RV = sqrt(mean(r**2 in window) * BARS_PER_YEAR),
    then `.shift(1)` so bar i sees only returns strictly before i -- the same
    causal convention `KellyRegimeV3.prepare()` already uses on its own `vol`.
    """
    close = df["close"]
    r = np.log(close).diff()
    cols = {}
    for days in windows_days:
        w = int(days * BARS_PER_DAY)
        cols[f"rv_{days}d"] = np.sqrt(
            r.pow(2).rolling(w, min_periods=BARS_PER_DAY).mean() * BARS_PER_YEAR
        ).shift(1)
    return pd.DataFrame(cols, index=df.index)


def har_rv_vol(df: pd.DataFrame, windows_days=(1, 5, 22)) -> pd.Series:
    """CONSERVATIVE vol substitute: a-priori equal-weight mean of the three
    realized-vol components. `skipna` so the estimate is live as soon as the
    shortest covered component is (the daily one, after one warmup day),
    rather than waiting for the monthly window -- matching v4's own
    `.ffill().fillna(0.0)` philosophy of using the best information
    available rather than blocking on the slowest input."""
    return compute_rv_components(df, windows_days).mean(axis=1, skipna=True)


def har_iv_vol(df: pd.DataFrame, dvol_causal: pd.Series, windows_days=(1, 5, 22)) -> pd.Series:
    """NOVEL vol substitute: the same components plus causally-aligned DVOL
    (already vol-fraction units), a-priori equal-weight mean of all four.
    Falls back to the three-way mean whenever DVOL is NaN (pre-2021-03-24) --
    never fit, never backfilled."""
    comp = compute_rv_components(df, windows_days)
    comp = comp.assign(dvol=dvol_causal.reindex(df.index))
    return comp.mean(axis=1, skipna=True)


class HARVolMixin:
    """`KellyRegimeV3.prepare()` copied verbatim except the line computing
    `vol` is replaced by `self._vol_series(df, r)`. Every other line (vote,
    `slow`, `ratio`, the state machine, `full`/`steady`, the deadband) is
    bit-for-bit identical to the registered `kelly_regime_v3`/`v4`, so any
    measured difference is attributable to the vol estimator alone. See
    src/tradebot/strategies/kelly_regime_v3.py for the original.
    """

    def _vol_series(self, df: pd.DataFrame, r: pd.Series) -> np.ndarray:
        raise NotImplementedError

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

        vol = np.asarray(self._vol_series(df, r), dtype=float)
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
        state = 0
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
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df = df.copy()
        df["target"] = target
        return df


# ----------------------------------------------------------------------
# Baseline (unmodified kelly_regime_v4) and generic-candidate run/metric helpers.
# ----------------------------------------------------------------------

def run_baseline(df: pd.DataFrame, market: MarketSpec, start: str, end: str,
                  label: str = ""):
    strat = get_strategy("kelly_regime_v4")
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


def run_strategy(strat, df: pd.DataFrame, market: MarketSpec, start: str, end: str,
                  label: str = ""):
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


def log_growth_diff(res_a, res_b, mean_block: float = 30.0, n_boot: int = 2000, seed: int = 136):
    """Paired-bootstrap total-log-growth difference (a - b)."""
    ra = daily_returns(res_a.equity)
    rb = daily_returns(res_b.equity)
    n = min(len(ra), len(rb))
    ra = ra.iloc[-n:].to_numpy()
    rb = rb.iloc[-n:].to_numpy()
    return paired_bootstrap(ra, rb, total_log_return, mean_block=mean_block, n_boot=n_boot, seed=seed)


def sharpe_diff(res_a, res_b, mean_block: float = 30.0, n_boot: int = 2000, seed: int = 136):
    """Paired-bootstrap Sharpe difference (a - b) on aligned daily returns."""
    ra = daily_returns(res_a.equity)
    rb = daily_returns(res_b.equity)
    n = min(len(ra), len(rb))
    ra = ra.iloc[-n:].to_numpy()
    rb = rb.iloc[-n:].to_numpy()

    def _sharpe(x):
        x = np.asarray(x, dtype=float)
        sd = x.std(axis=-1, ddof=1)
        sd = np.where(sd <= 0, np.nan, sd)
        return np.nan_to_num(x.mean(axis=-1) / sd * np.sqrt(365.25))

    return paired_bootstrap(ra, rb, _sharpe, mean_block=mean_block, n_boot=n_boot, seed=seed)


def qlike_loss(forecast: np.ndarray, realized_next_day_rv: np.ndarray) -> float:
    """Patton (2011) QLIKE, this project's own standing vol-forecast-quality
    metric (R-08). Lower is better. Both series must already be aligned and
    NaN-free (caller's responsibility -- see each branch's own report)."""
    f = np.asarray(forecast, dtype=float)
    y = np.asarray(realized_next_day_rv, dtype=float)
    ratio = y / f
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def exposure_by_vol_quartile(target: np.ndarray, vol_for_quartiles: np.ndarray) -> dict:
    """B6 helper: mean |exposure| conditional on the realized-vol quartile of
    each bar, for the high-vol-de-leveraging-artifact check named in this
    module's docstring. `vol_for_quartiles` should be the SAME vol series
    used to drive `ratio`/state for a fair conditional split."""
    v = np.asarray(vol_for_quartiles, dtype=float)
    t = np.abs(np.asarray(target, dtype=float))
    finite = np.isfinite(v) & np.isfinite(t)
    v, t = v[finite], t[finite]
    if len(v) == 0:
        return {"q1": float("nan"), "q2": float("nan"), "q3": float("nan"), "q4": float("nan")}
    q = np.quantile(v, [0.25, 0.5, 0.75])
    bins = np.digitize(v, q)
    return {f"q{i + 1}": float(t[bins == i].mean()) if np.any(bins == i) else float("nan")
            for i in range(4)}


if __name__ == "__main__":
    # Self-test: causal truncation probe on THIS module's own baseline
    # plumbing, plus a print of baseline kelly_regime_v4 numbers this
    # round's own decision rule compares against.
    df, label = load_btc_train("spot")
    m_full, _ = run_baseline(df, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)
    df_trunc = df.loc[:INNER_VAL_END]
    m_trunc, _ = run_baseline(df_trunc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)
    ok = np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9)
    print(f"causal truncation probe (r136_shared baseline plumbing): "
          f"{'PASS' if ok else 'FAIL'} ({m_full.final_balance} vs {m_trunc.final_balance})")
    assert ok, "run_baseline reads ahead of its own truncation point"

    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        for per_name, start, end in B1_PERIODS:
            m, _ = run_baseline(df, mkt, start, end, label)
            print(f"baseline {mkt_name}/{per_name}: trades={m.num_trades} "
                  f"final={m.final_balance:.1f} sharpe={m.sharpe:.3f} dd={m.max_drawdown_pct:.1f}%")

    dvol = load_dvol_causal_train(df)
    print(f"DVOL causal series: {dvol.notna().sum()} non-NaN bars "
          f"(first {dvol.dropna().index.min()}), mean={dvol.dropna().mean():.3f}")
