"""R-142 NOVEL branch: a continuous, TWO-SIDED SIZE-axis exposure
multiplier on `kelly_regime_v4`'s own `scale`, driven by the Deribit
front-vs-next-quarter futures term-structure SLOPE (the same signal the
sibling CONSERVATIVE branch tests as an INFO-axis confirming vote --
these two branches share `experiments/r142_shared.py` and nothing else;
neither edits it, and this file does not read the conservative branch's
Step-A gate result before running its own Step-0/B1-B5 battery).

=====================================================================
PRE-REGISTRATION (frozen before any backtest number in this file was
computed -- docs/ROUTINE.md steps 1-2/4). If anything below is later
contradicted by what actually happened, that is stated in the results
section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence). Modulate `kelly_regime_v4`'s existing
   `scale` by a bounded, two-sided function of the term-structure slope's
   own z-score: damp exposure when the curve is steeply, richly
   contangoed (a crowded, blown-off-looking curve shape -- Bianchi, Fan,
   Miffre & Zhang 2023's own finding that slope-momentum profitability
   "increases with investor sentiment" reads this direction as
   euphoria-adjacent), and modestly AMPLIFY exposure when the curve is
   deeply inverted/backwardated (a capitulation-adjacent curve shape --
   CF Benchmarks 2025, "Revisiting the Bitcoin Basis", and contemporaneous
   reporting on 2025-11 BTC backwardation preceding a local bottom; this
   specific 2025-11 episode is NOT used anywhere in this branch's gates,
   per the discipline note in r142_shared.py's module docstring, only as
   motivating color found during this round's own literature search).

   FORMULA: `scale_novel = scale_v4 * (1 - kappa * tanh(slope_z / 2))`,
   `slope_z` from `r142_shared.slope_zscore` (20-day causal rolling
   window, identical construction to the conservative branch's Step-A
   feature). `tanh(slope_z/2)` bounds the multiplier's own input to
   (-1, 1) regardless of how extreme a single `slope_z` print is (this
   round's own Step-0 data check found raw annualized-slope outliers as
   large as +23/-15 on rare near-expiry ticks; `tanh` prevents one such
   tick from producing an extreme, economically meaningless exposure
   swing -- a numerical-stability choice made before any backtest number,
   not a fitted clip).

   THIS IS DELIBERATELY TWO-SIDED (damp ranges over `[1-kappa, 1+kappa]`,
   both directions reachable, not bounded only above or only below 1) and
   DELIBERATELY NOT CALIBRATED BY EQUALITY MEAN-EXPOSURE MATCHING.
   R-141's own LPPLS dampener (`scale_novel = scale_v4 * max(0.1, 1 -
   kappa*confidence)`) is ONE-SIDED (damp <= 1 always), so matching
   `mean(scale_novel)` to `mean(scale_v4)` under exact equality is
   mathematically forced to kappa=0 (R-141, ruled out -- see
   docs/LEDGER.md's section C, R^2=1.000000 exactly). This construction
   avoids that trap by construction (kappa's effect is symmetric around
   1, so equality-matching does not collapse to a corner solution) --
   verified directly in Step 0 below before it is trusted, exactly as
   R-141's own post-hoc analytical proof should have been run pre-hoc.
   kappa is NOT solved for by any matching procedure at all: it is swept
   on the FIXED, PRE-REGISTERED grid `r142_shared.NOVEL_KAPPA_GRID =
   (0.0, 0.10, 0.20, 0.30)`, and every grid point is reported as a B3
   plateau cell -- none is "the" selected candidate before the battery
   runs.

   Citations: same term-structure-slope citation trail as the
   conservative branch (Bianchi, Fan, Miffre & Zhang 2023; Erb & Harvey
   2006; Schmeling, Schrimpf & Todorov 2023/2025); MacLean, Thorp & Ziemba
   (2010) on fractional-Kelly exposure shrinkage under parameter
   uncertainty, for the general shape of a bounded multiplicative
   dampener (this project's own standing justification for every
   SIZE-axis dampener since R-59).

   CONSTRAINT ATTACKED: SIZE (this is the axis that has actually worked in
   this project, per the standing diagnosis) via a genuinely new
   informational input (the curve slope) no prior SIZE-axis attempt has
   used -- 28+ prior dampeners (R-59 through R-141) all transform either
   the vote, the raw volatility estimate, or a model-derived
   confidence/hazard signal computed from spot alone.

   NOT A DUPLICATE OF: any prior SIZE-axis dampener (grepped
   docs/LEDGER.md's ruled-out table: CVaR, robust/shrinkage Kelly, CPPI,
   uncertainty-shrink, ladders, HAR-vol substitution, LPPLS crash-hazard --
   none uses a futures-curve input); R-141 specifically, distinguished
   above by sidedness and calibration method; the sibling CONSERVATIVE
   branch, which tests the same underlying slope signal as a discrete
   INFO-axis confirming VOTE inside the anchor gate, not a continuous
   SIZE-axis multiplier on `scale`.

2. STEP 0 -- MANDATORY PRE-FLIGHT CHECKS, run before any backtest number:
   (a) IDENTITY-RECOVERY: `kappa=0` must reproduce `kelly_regime_v4`'s own
       `target` array exactly (`np.array_equal`, not merely `allclose`).
   (b) NOT-DEGENERATE-BY-CONSTRUCTION: verify analytically (as R-141's own
       post-hoc proof should have been run pre-hoc) that this dampener's
       mean exposure is NOT monotonic in kappa across the pre-registered
       grid -- i.e., confirm this construction cannot fall into R-141's
       trap before trusting any downstream number. Report
       `mean(scale_novel)` at every grid point next to `mean(scale_v4)`.
   (c) CAUSAL TRUNCATION PROBE on the full `slope_z -> scale_novel`
       pipeline (`r142_shared.truncation_causality_probe`-style check),
       matching every prior round's convention.
   Any failure here stops the round before Step 3/4, same as R-141's own
   Step-0 gate stopping its dampener before B1-B5 ran.

3. STEP 3 (inner-train/inner-validation only, holdout untouched) --
   BATTERY, run for every kappa in the grid, BTC and ETH, spot and
   futures:
   - B1: full-period and inner-validation Sharpe, Sortino, max drawdown,
     turnover, vs `kelly_regime_v4` unmodified, for every kappa grid point
     (a plateau report, not a single winner).
   - B3: the plateau check itself -- report all 4 kappa cells together;
     a candidate is only taken seriously if improvement is monotonic-ish
     or flat across kappa in {0.10, 0.20, 0.30}, not a spike at one value.
   - B4: ETH sign-replication -- the BTC-selected kappa (if any looks
     promising) must not invert sign on ETH, this project's single most
     repeated failure mode (R-33/R-57/R-62/R-64/R-113/R-127/R-137 and
     others).
   - B5 / R-33 RISK-MATCHING: report time-in-market and realized
     annualized volatility for `kelly_regime_v4` and every kappa
     candidate, side by side. A candidate whose realized volatility
     diverges from v4's own by more than 15% is reported but treated as
     UNMATCHED and void for promotion purposes, per this project's
     standing "match risk before comparing anything" rule -- exposure
     divergence is not this round's own multiplier design intent (kappa
     is meant to REDISTRIBUTE risk across regimes, not change its
     average level), so a large divergence signals something is off
     rather than a real finding.

4. STEP 4 -- FALSIFICATION TEST (the one chosen now, before any holdout
   number): the 0.40% taker fee tier (`scripts/fee_study.py`'s real-cost
   convention, R-73/R-141's own choice for an analogous continuous
   dampener). A candidate is only promotable if at least one kappa > 0
   grid point still beats `kelly_regime_v4` at the 0.40% tier, on BOTH
   BTC and ETH, out-of-sample (>= OOS_START) -- the coverage-extension
   fetch this round required (see r142_shared.py) makes this a full,
   unbroken holdout for the first time this signal type has had one.

5. DECISION RULE (frozen now): promote only if ALL of docs/ROUTINE.md's
   standing promotion-bar clauses hold for at least one kappa grid point
   -- beats buy_and_hold OOS after real costs; improvement exceeds the
   +/-0.2 Sharpe noise floor OR is a genuine drawdown/tail improvement;
   survives the 0.40% fee tier; the kappa neighbourhood is a plateau
   (B3); AND passes the R-33 risk-match check (B5) without the 15%
   divergence flag. Failing any one clause is NEGATIVE, reported with the
   same care as a promotion, per docs/ROUTINE.md's own standard.

6. WHAT WOULD MAKE IT FAIL (named now): mean mechanism damps the wrong
   direction relative to buy_and_hold (a Sharpe/drawdown result worse than
   v4 alone at every non-zero kappa); or a real effect exists on BTC alone
   but inverts sign on ETH (this ledger's single most common failure
   mode); or the effect only appears at one isolated kappa rather than
   across the grid (a peak, not a plateau); or it requires an unmatched
   exposure change to show up at all (the R-33/B5 check).

CONFIGURATIONS EVALUATED: to be filled in by the implementing session,
counting every kappa-grid cell (4) x market (BTC/ETH) x period
(full/inner-val) combination in B1/B3, every B4/B5 cell, and the Step-4
fee-tier check.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments import r142_shared as sh  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.inference import daily_returns  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# ----------------------------------------------------------------------
# Markets. SPOT/FUTURES at this project's standard 0.10% tier; the
# *_HIGH_FEE pair is the 0.40% Bitstamp taker tier used only by the
# CONDITIONAL Step-4 falsification test, per r125_shared.py's convention
# (reused verbatim -- r142_shared.py does not define these, so they are
# declared here rather than imported).
# ----------------------------------------------------------------------
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT_HIGH_FEE = MarketSpec.spot(fee_rate=0.0040)
FUTURES_HIGH_FEE = MarketSpec.futures(leverage=5.0, fee_rate=0.0040)

INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = sh.INNER_TRAIN_END      # "2020-12-31"
INNER_VAL_START = sh.INNER_VAL_START      # "2021-01-01"
INNER_VAL_END = sh.INNER_VAL_END          # "2022-12-31"
OOS_START = sh.OOS_START                  # "2023-01-01"

KAPPA_GRID = sh.NOVEL_KAPPA_GRID          # (0.0, 0.10, 0.20, 0.30), frozen


def _assert_no_holdout(df: pd.DataFrame) -> None:
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    assert df.index.max() < cutoff, (
        f"holdout bar read: max timestamp {df.index.max()} >= {OOS_START}")


# ======================================================================
# (1) DATA: spot OHLCV + the causal slope/slope_z columns, merged once per
# asset. This merged frame spans the asset's FULL available history
# (through the present, per r142_shared's coverage-extension fetch) --
# truncation to inner-train / inner-validation / holdout happens only at
# call sites (via run_period's start/end), never here, so every period
# reads the identical, already-verified-causal slope_z values.
# ======================================================================

def build_slope_frame(asset: str) -> pd.DataFrame:
    if asset == "BTC":
        spot, _ = load_dataset(ROOT / "data", "spot")
    elif asset == "ETH":
        spot = load_coinbase_eth_spot(ROOT / "data")
        assert spot is not None, "ETH spot data not committed"
    else:
        raise ValueError(asset)
    quarterly = sh.load_deribit_quarterly(ROOT / "data", asset)
    assert quarterly is not None, f"{asset} quarterly futures data not committed"
    slope_df = sh.dual_quarter_slope(spot, quarterly)
    z = sh.slope_zscore(slope_df["slope"])
    out = spot.copy()
    out["slope"] = slope_df["slope"]
    out["slope_z"] = z
    return out


# ======================================================================
# (2) v4's own intermediate `scale` (pre-frac, pre-deadband, post-
# hysteresis) is never exposed by kelly_regime_v4.prepare(). Extracted
# verbatim from KellyRegimeV3.prepare()'s own published source, identical
# pattern to R-125/R-141's own `v4_scale_series` -- and verified exact by
# `self_test_v4_scale_matches_v4_target` before any calibration/Step-0
# number trusts it.
# ======================================================================

def v4_scale_series(df: pd.DataFrame, v4: KellyRegimeV4) -> np.ndarray:
    close = df["close"]
    r = np.log(close).diff()
    vol = (r.ewm(span=v4.vol_span, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    slow = (pd.Series(vol).ewm(span=v4.anchor_span_days * BARS_PER_DAY,
                                min_periods=BARS_PER_DAY).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        full = np.minimum(v4.target_vol / vol, v4.max_leverage)
        steady = np.minimum(v4.target_vol / slow, v4.max_leverage)
        ratio = np.where(slow > 0, vol / slow, np.nan)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    n = len(df)
    scale = np.zeros(n)
    state = 0
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > v4.high_in else (-1 if x < v4.low_in else 0)
            elif state == 1 and x < v4.high_out:
                state = 0
            elif state == -1 and x > v4.low_out:
                state = 0
        scale[i] = full[i] if state != 0 else steady[i]
    return scale


def self_test_v4_scale_matches_v4_target(df: pd.DataFrame) -> bool:
    v4 = get_strategy("kelly_regime_v4")
    v4_target = v4.prepare(df.copy())["target"].to_numpy()
    scale = v4_scale_series(df, v4)

    close = df["close"]
    votes = []
    for days in v4.horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + v4.band), 1.0,
                     np.where(close < anchor * (1.0 - v4.band), 0.0, np.nan)),
            index=df.index,
        )
        votes.append(v.ffill().fillna(0.0))
    frac = (sum(votes) / len(votes)).to_numpy()
    if v4.vote_gamma != 1.0:
        frac = frac ** v4.vote_gamma

    n = len(df)
    target = np.zeros(n)
    pos = 0.0
    for i in range(n):
        desired = frac[i] * scale[i]
        if abs(desired - pos) > v4.deadband:
            pos = desired
        target[i] = pos

    return bool(np.allclose(target, v4_target, equal_nan=True))


# ======================================================================
# (3) NovelSlopeDampener: kelly_regime_v4.prepare(), copied verbatim, with
# exactly one post-hoc substitution -- scale_novel = scale_v4 * (1 -
# kappa*tanh(slope_z/2)). NOT @register'd -- experiments/-only.
# ======================================================================

class NovelSlopeDampener(KellyRegimeV4):
    """kelly_regime_v4's exact architecture (3-anchor vote, 20/40/80-day
    anchors, 1% band, extremes-only hysteresis latch, 10% deadband) with
    the ONE substitution this round tests: the final, hysteresis-selected
    ``scale`` is multiplied by a TWO-SIDED term-structure-slope dampener,
    ``1 - kappa*tanh(slope_z/2)``, before being combined with the vote.

    Requires ``df`` to carry a ``slope_z`` column (see
    ``build_slope_frame``). Where ``slope_z`` is NaN -- fewer than two
    quarterlies simultaneously listed, 20.5%/15.0% of BTC/ETH bars per
    r142_shared's own disclosed coverage measurement -- the multiplier
    defaults to 1.0 (v4's own scale, unmodified) rather than propagating
    NaN into a strategy `target`: a disclosed causal design choice (the
    mechanism has no input to react to on those bars), not a fitted patch.

    kappa=0 recovers v4 EXACTLY regardless of slope_z's own value (`1 -
    0*x == 1.0` for every finite `x`, and the NaN branch is independently
    forced to 1.0), which is this round's required identity-recovery
    check. Not registered in ``src/tradebot/strategies/`` -- experiments/
    -only, per this round's instructions.
    """

    name = "r142_novel_slope_dampener"

    def __init__(self, kappa: float = 0.0, horizons: tuple[int, ...] = (20, 40, 80),
                 **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.kappa = kappa

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        # Vote: byte-identical to KellyRegimeV3.prepare() / KellyRegimeV4.
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

        # Risk measure / hysteresis latch: byte-identical to
        # KellyRegimeV3.prepare() / KellyRegimeV4.
        r = np.log(close).diff()
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

        # ---- THE ONE SUBSTITUTION: two-sided slope dampener, applied to
        # the hysteresis-selected scale, post-hoc.
        if "slope_z" in df.columns:
            slope_z = df["slope_z"].to_numpy(dtype=float)
        else:
            slope_z = np.full(len(df), np.nan)
        with np.errstate(invalid="ignore"):
            mult = 1.0 - self.kappa * np.tanh(slope_z / 2.0)
        mult = np.where(np.isfinite(mult), mult, 1.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
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
            scale_v4 = full[i] if state != 0 else steady[i]
            scale_novel = scale_v4 * mult[i]
            desired = frac[i] * scale_novel
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df


def make_candidate_factory(kappa: float):
    def _factory():
        return NovelSlopeDampener(kappa=kappa)
    return _factory


# ======================================================================
# (4) Causal-truncation self-test on THIS round's own new code: the whole
# raw-OHLCV+quarterly -> slope -> slope_z -> scale_novel -> target
# pipeline, not merely the shared module's own dual_quarter_slope
# causality claim taken on faith.
# ======================================================================

def causal_truncation_probe(asset: str, kappa: float, cut: int = 400_000) -> bool:
    if asset == "BTC":
        spot, _ = load_dataset(ROOT / "data", "spot")
    else:
        spot = load_coinbase_eth_spot(ROOT / "data")
    quarterly = sh.load_deribit_quarterly(ROOT / "data", asset)

    def build_and_run(spot_slice: pd.DataFrame) -> np.ndarray:
        slope_df = sh.dual_quarter_slope(spot_slice, quarterly)
        z = sh.slope_zscore(slope_df["slope"])
        frame = spot_slice.copy()
        frame["slope_z"] = z
        return NovelSlopeDampener(kappa=kappa).prepare(frame)["target"].to_numpy()

    full_target = build_and_run(spot)
    trunc_target = build_and_run(spot.iloc[:cut].copy())
    n_check = min(len(trunc_target), cut) - BARS_PER_DAY * 90  # skip the fresh 80d-anchor warmup tail
    return bool(np.allclose(full_target[:n_check], trunc_target[:n_check], equal_nan=True, rtol=1e-9))


# ======================================================================
# (5) Backtest / metrics harness (r125_shared.b1_signal-style, written
# locally since r142_shared.py -- read-only, shared with the sibling
# conservative branch -- does not itself provide one).
# ======================================================================

def sortino_ratio(rets: np.ndarray, periods_per_year: float = 365.25) -> float:
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 3:
        return 0.0
    downside = np.minimum(rets, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd == 0.0 or not np.isfinite(dd):
        return 0.0
    return float(np.mean(rets) / dd * np.sqrt(periods_per_year))


def realized_vol(rets: np.ndarray, periods_per_year: float = 365.25) -> float:
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 3:
        return 0.0
    sd = rets.std(ddof=1)
    return float(sd * np.sqrt(periods_per_year)) if np.isfinite(sd) else 0.0


def run_cell(strategy_factory, df: pd.DataFrame, market: MarketSpec,
             start: str, end: str) -> dict:
    strat = strategy_factory()
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label="")
    m = compute_metrics(res)
    rets = daily_returns(res.equity).to_numpy()
    return dict(
        sharpe=m.sharpe, sortino=sortino_ratio(rets), dd=m.max_drawdown_pct,
        trades=m.num_trades, fills=len(res.fills),
        time_in_market=m.time_in_market_pct, vol=realized_vol(rets),
        final=m.final_balance,
    )


# ======================================================================
# (6) Main: Step 0 (identity / non-degeneracy / causal probe) -> B1/B3
# (kappa-grid plateau, both assets, both markets, both periods) -> B4 (ETH
# sign-replication) -> B5 (R-33 risk-match) -> CONDITIONAL Step 4 (0.40%
# fee tier, holdout) -> verdict.
# ======================================================================

def main() -> dict:
    t0 = time.time()
    n_configs = 0
    max_ts_seen: list[pd.Timestamp] = []

    print("=" * 78)
    print("R-142 NOVEL: NovelSlopeDampener -- kelly_regime_v4's own architecture, scale")
    print("post-multiplied by a two-sided term-structure-slope dampener")
    print("1 - kappa*tanh(slope_z/2), kappa on the FIXED grid", KAPPA_GRID)
    print("=" * 78)

    print("\n-- BUILDING slope-augmented frames (BTC, ETH) --")
    frames = {}
    for asset in ("BTC", "ETH"):
        tb = time.time()
        frames[asset] = build_slope_frame(asset)
        max_ts_seen.append(frames[asset].index.max())
        nan_frac = float(frames[asset]["slope_z"].isna().mean())
        print(f"  {asset}: {len(frames[asset]):,} bars, {frames[asset].index[0]} -> "
              f"{frames[asset].index[-1]}  slope_z NaN frac={nan_frac:.4f}  [{time.time()-tb:.0f}s]")

    btc_train = frames["BTC"].loc[:INNER_VAL_END].copy()
    _assert_no_holdout(btc_train)
    eth_train = frames["ETH"].loc[:INNER_VAL_END].copy()
    _assert_no_holdout(eth_train)

    # ---------------------------------------------------------- wiring self-test
    print("\n-- PRE-FLIGHT SELF-TEST: v4_scale_series extraction matches "
          "kelly_regime_v4.prepare()'s own target exactly --")
    wiring_ok = self_test_v4_scale_matches_v4_target(btc_train)
    print(f"  self_test_v4_scale_matches_v4_target (BTC): {'PASS' if wiring_ok else 'FAIL'}")
    if not wiring_ok:
        print("\nWIRING SELF-TEST FAILURE -- stopping before any Step-0 number is trusted.")
        return dict(verdict="ABORTED (wiring self-test failure)")

    # ============================================================ STEP 0
    print("\n" + "=" * 78)
    print("STEP 0 (a) -- IDENTITY-RECOVERY: kappa=0 must reproduce kelly_regime_v4 EXACTLY")
    print("=" * 78)
    v4 = get_strategy("kelly_regime_v4")
    v4_target_btc = v4.prepare(btc_train.copy())["target"].to_numpy()
    kappa0_target_btc = NovelSlopeDampener(kappa=0.0).prepare(btc_train.copy())["target"].to_numpy()
    identity_ok = bool(np.array_equal(kappa0_target_btc, v4_target_btc))
    print(f"  kappa=0 target array_equal to kelly_regime_v4's own target (BTC, np.array_equal): {identity_ok}")
    if not identity_ok:
        print("\nIDENTITY-RECOVERY FAILURE -- stopping before Step 3.")
        return dict(verdict="ABORTED (identity-recovery failure)")

    print("\n" + "=" * 78)
    print("STEP 0 (b) -- NOT-DEGENERATE-BY-CONSTRUCTION: mean(scale_novel) at every kappa, "
          "BTC inner-train, vs mean(scale_v4) -- must NOT collapse toward v4's own mean the way "
          "R-141's one-sided, equality-matched dampener did")
    print("=" * 78)
    btc_inner_train_only = frames["BTC"].loc[INNER_TRAIN_START:INNER_TRAIN_END]
    v4_scale_check = v4_scale_series(btc_inner_train_only, v4)
    slope_z_check = btc_inner_train_only["slope_z"].to_numpy(dtype=float)
    target_mean = float(np.nanmean(v4_scale_check))
    print(f"  mean(scale_v4) on BTC inner-train ({INNER_TRAIN_START}:{INNER_TRAIN_END}) = {target_mean:.6f}")
    step0b_rows = []
    for kappa in KAPPA_GRID:
        with np.errstate(invalid="ignore"):
            mult = 1.0 - kappa * np.tanh(slope_z_check / 2.0)
        mult = np.where(np.isfinite(mult), mult, 1.0)
        scale_novel = v4_scale_check * mult
        mean_novel = float(np.nanmean(scale_novel))
        step0b_rows.append(dict(kappa=kappa, mean_scale_novel=mean_novel, gap=mean_novel - target_mean))
        print(f"    kappa={kappa:.2f}  mean(scale_novel)={mean_novel:.6f}  "
              f"gap_vs_v4={mean_novel - target_mean:+.6f}")
    gaps = [row["gap"] for row in step0b_rows]
    monotone_collapsing = all(g <= 1e-9 for g in gaps) and (
        len(gaps) < 2 or all(gaps[i] >= gaps[i + 1] - 1e-9 for i in range(len(gaps) - 1)))
    print(f"  gaps are all <=0 AND monotonically non-increasing (the R-141 degeneracy signature): "
          f"{monotone_collapsing}")
    step0b_ok = not monotone_collapsing
    print(f"  STEP 0(b) PASS (not degenerate by this signature): {step0b_ok}")
    if not step0b_ok:
        print("\nSTEP 0(b) FAILURE -- construction shows the same degeneracy signature as R-141. Stopping.")
        return dict(verdict="ABORTED (Step 0b degeneracy)", step0b_rows=step0b_rows,
                    identity_ok=identity_ok)

    print("\n" + "=" * 78)
    print("STEP 0 (c) -- CAUSAL TRUNCATION PROBE on the full slope_z -> scale_novel pipeline "
          "(BTC and ETH, diagnostic kappa=0.20 so the probe exercises the actual multiplication, "
          "not the kappa=0 no-op)")
    print("=" * 78)
    probe_btc = causal_truncation_probe("BTC", kappa=0.20)
    probe_eth = causal_truncation_probe("ETH", kappa=0.20)
    print(f"  BTC causal_truncation_probe (kappa=0.20 diagnostic): {'PASS' if probe_btc else 'FAIL'}")
    print(f"  ETH causal_truncation_probe (kappa=0.20 diagnostic): {'PASS' if probe_eth else 'FAIL'}")
    probe_ok = bool(probe_btc and probe_eth)
    if not probe_ok:
        print("\nCAUSAL PROBE FAILURE -- a result that looks too good is a bug report first. Stopping.")
        return dict(verdict="NEGATIVE (causal probe failure)", identity_ok=identity_ok,
                    step0b_rows=step0b_rows, probe_btc=probe_btc, probe_eth=probe_eth)

    print("\nSTEP 0 -- ALL CHECKS PASS. Proceeding to Step 3 (inner-train/inner-validation only).")

    # ============================================================ STEP 3: B1/B3
    print("\n" + "=" * 78)
    print("B1/B3 -- kappa-grid plateau: 4 kappa x 2 assets (BTC, ETH) x 2 markets (spot, futures_5x) "
          "x 2 periods (full inner 2017-2022, inner-validation-only 2021-2022)")
    print("=" * 78)

    assets = {"BTC": btc_train, "ETH": eth_train}
    markets = {"spot": SPOT, "futures_5x": FUTURES}
    periods = {"full_inner": (INNER_TRAIN_START, INNER_VAL_END),
               "inner_val": (INNER_VAL_START, INNER_VAL_END)}

    # v4 reference cells: 2 assets x 2 markets x 2 periods = 8, computed once,
    # shared across all 4 kappa candidates.
    v4_cells = {}
    for asset_name, df in assets.items():
        for market_name, market in markets.items():
            for period_name, (pstart, pend) in periods.items():
                key = (asset_name, market_name, period_name)
                v4_cells[key] = run_cell(lambda: get_strategy("kelly_regime_v4"), df, market, pstart, pend)
                n_configs += 1
                print(f"  [v4 ref]   {asset_name:>3s} {market_name:>10s} {period_name:>10s}  "
                      f"sharpe={v4_cells[key]['sharpe']:+.4f}  dd={v4_cells[key]['dd']:.2f}%  "
                      f"tim={v4_cells[key]['time_in_market']:.1f}%  vol={v4_cells[key]['vol']:.4f}")

    # Candidate cells: 4 kappa x 2 assets x 2 markets x 2 periods = 32.
    cand_cells = {}
    for kappa in KAPPA_GRID:
        factory = make_candidate_factory(kappa)
        for asset_name, df in assets.items():
            for market_name, market in markets.items():
                for period_name, (pstart, pend) in periods.items():
                    key = (kappa, asset_name, market_name, period_name)
                    cand_cells[key] = run_cell(factory, df, market, pstart, pend)
                    n_configs += 1
                    v4c = v4_cells[(asset_name, market_name, period_name)]
                    c = cand_cells[key]
                    d_sharpe = c["sharpe"] - v4c["sharpe"]
                    print(f"  kappa={kappa:.2f}  {asset_name:>3s} {market_name:>10s} {period_name:>10s}  "
                          f"sharpe={c['sharpe']:+.4f} (v4={v4c['sharpe']:+.4f}, d={d_sharpe:+.4f})  "
                          f"sortino={c['sortino']:+.4f}  dd={c['dd']:.2f}% (v4={v4c['dd']:.2f}%)  "
                          f"trades={c['trades']} fills={c['fills']}  tim={c['time_in_market']:.1f}%  "
                          f"vol={c['vol']:.4f}")

    print(f"\n  B1/B3 configurations evaluated so far: {n_configs} (8 v4-reference + 32 candidate cells)")

    # Second identity check, at the backtest level: kappa=0 cells must be
    # bit-identical to the v4 reference cells (same strategy, same data).
    identity_backtest_ok = True
    for asset_name in assets:
        for market_name in markets:
            for period_name in periods:
                v4c = v4_cells[(asset_name, market_name, period_name)]
                c0 = cand_cells[(0.0, asset_name, market_name, period_name)]
                same = (v4c["sharpe"] == c0["sharpe"] and v4c["dd"] == c0["dd"]
                        and v4c["trades"] == c0["trades"] and v4c["fills"] == c0["fills"])
                identity_backtest_ok = identity_backtest_ok and same
    print(f"\n  SECOND IDENTITY CHECK (backtest level): kappa=0 cells bit-identical to v4-reference "
          f"cells on all 8 (asset, market, period) combinations: {identity_backtest_ok}")

    # ---------------------------------------------------------------- B3: plateau view
    print("\n" + "=" * 78)
    print("B3 -- PLATEAU: the 4 kappa-grid d_sharpe cells together (inner-validation only), "
          "not a single winner")
    print("=" * 78)
    b3_table = []
    for asset_name in assets:
        for market_name in markets:
            v4c = v4_cells[(asset_name, market_name, "inner_val")]
            row = dict(asset=asset_name, market=market_name, v4_sharpe=v4c["sharpe"])
            for kappa in KAPPA_GRID:
                c = cand_cells[(kappa, asset_name, market_name, "inner_val")]
                row[f"d_sharpe_k{kappa:.2f}"] = c["sharpe"] - v4c["sharpe"]
            b3_table.append(row)
            print(f"  {asset_name:>3s} {market_name:>10s}  v4_sharpe={v4c['sharpe']:+.4f}   "
                  + "  ".join(f"k={kappa:.2f}: d={row[f'd_sharpe_k{kappa:.2f}']:+.4f}" for kappa in KAPPA_GRID))

    # ---------------------------------------------------------------- B4: ETH sign-replication
    print("\n" + "=" * 78)
    print("B4 -- ETH sign-replication: for each non-zero kappa, does BTC's d_sharpe sign "
          "(inner-validation) match ETH's?")
    print("=" * 78)
    b4_rows = []
    for market_name in markets:
        for kappa in KAPPA_GRID:
            if kappa == 0.0:
                continue
            btc_c = cand_cells[(kappa, "BTC", market_name, "inner_val")]
            eth_c = cand_cells[(kappa, "ETH", market_name, "inner_val")]
            btc_v4 = v4_cells[("BTC", market_name, "inner_val")]
            eth_v4 = v4_cells[("ETH", market_name, "inner_val")]
            d_btc = btc_c["sharpe"] - btc_v4["sharpe"]
            d_eth = eth_c["sharpe"] - eth_v4["sharpe"]
            same_sign = bool(np.sign(d_btc) == np.sign(d_eth) and d_btc != 0)
            b4_rows.append(dict(market=market_name, kappa=kappa, d_btc=d_btc, d_eth=d_eth,
                                 same_sign=same_sign))
            print(f"  {market_name:>10s}  kappa={kappa:.2f}  d_sharpe_BTC={d_btc:+.4f}  "
                  f"d_sharpe_ETH={d_eth:+.4f}  SAME SIGN: {same_sign}")
    n_same_sign = sum(1 for row in b4_rows if row["same_sign"])
    b4_pass = n_same_sign == len(b4_rows) and len(b4_rows) > 0
    print(f"  B4: {n_same_sign}/{len(b4_rows)} (kappa, market) cells replicate sign on ETH. "
          f"FULL PASS (all cells): {b4_pass}")

    # ---------------------------------------------------------------- B5: R-33 risk-match
    print("\n" + "=" * 78)
    print("B5 / R-33 RISK-MATCH -- time-in-market and realized annualized volatility, "
          "v4 vs every kappa candidate, inner-validation")
    print("=" * 78)
    b5_rows = []
    for asset_name in assets:
        for market_name in markets:
            v4c = v4_cells[(asset_name, market_name, "inner_val")]
            print(f"  {asset_name:>3s} {market_name:>10s}  v4: tim={v4c['time_in_market']:.1f}%  "
                  f"vol={v4c['vol']:.4f}")
            for kappa in KAPPA_GRID:
                c = cand_cells[(kappa, asset_name, market_name, "inner_val")]
                vol_divergence = abs(c["vol"] - v4c["vol"]) / v4c["vol"] if v4c["vol"] > 0 else float("nan")
                flagged = bool(np.isfinite(vol_divergence) and vol_divergence > 0.15)
                b5_rows.append(dict(asset=asset_name, market=market_name, kappa=kappa,
                                     tim=c["time_in_market"], vol=c["vol"],
                                     vol_divergence=vol_divergence, flagged=flagged))
                print(f"      kappa={kappa:.2f}  tim={c['time_in_market']:.1f}%  vol={c['vol']:.4f}  "
                      f"vol_divergence={vol_divergence:+.1%}  {'FLAGGED (>15%)' if flagged else ''}")

    # ============================================================ decision rule (Step 3)
    print("\n" + "=" * 78)
    print("STEP 3 DECISION: is there a genuinely plausible candidate for Step 4?")
    print("(at least one kappa>0 beats v4 on Sharpe OR drawdown on BOTH BTC and ETH "
          "inner-validation, without the 15% risk-mismatch flag, on at least one market)")
    print("=" * 78)
    plausible_candidates = []
    for market_name in markets:
        for kappa in KAPPA_GRID:
            if kappa == 0.0:
                continue
            btc_c = cand_cells[(kappa, "BTC", market_name, "inner_val")]
            eth_c = cand_cells[(kappa, "ETH", market_name, "inner_val")]
            btc_v4 = v4_cells[("BTC", market_name, "inner_val")]
            eth_v4 = v4_cells[("ETH", market_name, "inner_val")]
            btc_better = btc_c["sharpe"] > btc_v4["sharpe"] or btc_c["dd"] < btc_v4["dd"]
            eth_better = eth_c["sharpe"] > eth_v4["sharpe"] or eth_c["dd"] < eth_v4["dd"]
            btc_flag = next(r["flagged"] for r in b5_rows if r["asset"] == "BTC"
                             and r["market"] == market_name and r["kappa"] == kappa)
            eth_flag = next(r["flagged"] for r in b5_rows if r["asset"] == "ETH"
                             and r["market"] == market_name and r["kappa"] == kappa)
            if btc_better and eth_better and not btc_flag and not eth_flag:
                plausible_candidates.append((market_name, kappa))
    print(f"  Plausible (kappa, market) candidates found: {plausible_candidates}")
    run_step4 = len(plausible_candidates) > 0

    result = dict(
        verdict=None, n_configs=n_configs, identity_ok=identity_ok,
        identity_backtest_ok=identity_backtest_ok, step0b_rows=step0b_rows,
        probe_ok=probe_ok, v4_cells=v4_cells, cand_cells=cand_cells,
        b3_table=b3_table, b4_rows=b4_rows, b4_pass=b4_pass, b5_rows=b5_rows,
        plausible_candidates=plausible_candidates, run_step4=run_step4,
        max_ts=max(max_ts_seen),
    )

    if not run_step4:
        print("\nNo plausible candidate found on Step 3 -- per pre-registration, STOPPING here. "
              "The holdout is NOT touched.")
        result["verdict"] = "NEGATIVE (Step 3: no plausible candidate)"
        print(f"\nconfigurations evaluated (Step 3 total): {n_configs}")
        print(f"max timestamp read anywhere in this branch (bounded by inner-train/inner-val "
              f"slicing, though the underlying frame spans further): "
              f"{result['max_ts']} -- inner-train/inner-validation frames used end at "
              f"{INNER_VAL_END} < {OOS_START}")
        print(f"\n[{time.time() - t0:.0f}s]")
        return result

    # ============================================================ CONDITIONAL STEP 4
    print("\n" + "=" * 78)
    print("STEP 4 (CONDITIONAL, TRIGGERED) -- 0.40% taker fee tier, HOLDOUT "
          f"(start={OOS_START}), BTC and ETH")
    print("=" * 78)
    fee_markets = {"spot": (SPOT_HIGH_FEE, SPOT), "futures_5x": (FUTURES_HIGH_FEE, FUTURES)}
    step4_rows = []
    tested_markets = sorted({m for m, _ in plausible_candidates})
    tested_kappas = sorted({k for _, k in plausible_candidates})
    for market_name in tested_markets:
        fee_market, base_market = fee_markets[market_name]
        for asset_name in ("BTC", "ETH"):
            df = frames[asset_name]
            v4_hold = run_cell(lambda: get_strategy("kelly_regime_v4"), df, fee_market, OOS_START, None)
            n_configs += 1
            for kappa in tested_kappas:
                cand_hold = run_cell(make_candidate_factory(kappa), df, fee_market, OOS_START, None)
                n_configs += 1
                beats = cand_hold["sharpe"] > v4_hold["sharpe"]
                step4_rows.append(dict(market=market_name, asset=asset_name, kappa=kappa,
                                        sharpe_cand=cand_hold["sharpe"], sharpe_v4=v4_hold["sharpe"],
                                        dd_cand=cand_hold["dd"], dd_v4=v4_hold["dd"], beats=beats))
                print(f"  {market_name:>10s} {asset_name:>3s}  kappa={kappa:.2f}  0.40% tier, holdout  "
                      f"sharpe_cand={cand_hold['sharpe']:+.4f}  sharpe_v4={v4_hold['sharpe']:+.4f}  "
                      f"BEATS v4: {beats}")

    step4_pass_by_kappa_market = {}
    for market_name in tested_markets:
        for kappa in tested_kappas:
            rows = [r for r in step4_rows if r["market"] == market_name and r["kappa"] == kappa]
            step4_pass_by_kappa_market[(market_name, kappa)] = all(r["beats"] for r in rows) and len(rows) == 2

    any_promotable = any(step4_pass_by_kappa_market.values())
    print(f"\n  Step 4 pass (beats v4 at 0.40% tier, holdout, BOTH BTC and ETH), by (market, kappa): "
          f"{step4_pass_by_kappa_market}")
    print(f"  ANY (market, kappa) combination promotable: {any_promotable}")

    result["step4_rows"] = step4_rows
    result["step4_pass_by_kappa_market"] = step4_pass_by_kappa_market
    result["n_configs"] = n_configs
    result["verdict"] = "PROMOTE-candidate" if any_promotable else "NEGATIVE (Step 4: fails 0.40% fee tier)"

    print(f"\nconfigurations evaluated (total): {n_configs}")
    print(f"\nVERDICT: {result['verdict']}")
    print(f"\n[{time.time() - t0:.0f}s]")
    return result


if __name__ == "__main__":
    main()
