"""Shared, read-only utilities and pre-registration for the R-118 round (08-24).

DIRECTION, in one sentence: does selecting `kelly_regime_v4`'s own already-
swept free parameters (anchor-ladder base, `target_vol`, `max_leverage`) by a
ROBUST criterion measured across many SYNTHETIC alternative price paths --
rather than across the three real calendar folds R-45 used, or the single
realized path R-06/R-07/R-37 used -- find a configuration that generalizes
better than R-45's own calendar-fold-robust winner, which still failed
because its three folds were all drawn from the one window being fit?

**Direct precedent, and the reason this round exists.** R-45 (08-19) is the
only prior round in this ledger that attacked N=3 by CHANGING THE SELECTION
PROCEDURE rather than adding a new signal: it split 2017-2022 into three
calendar-purged folds (one per this project's own N approx 3 regime-event
count) and chose the (ladder, target_vol, max_leverage) triple that
maximised the WORST-fold Sharpe (minimax) instead of the pooled point
estimate. That genuinely beat naive point-estimate selection on ITS OWN
falsification pair (ETH: wins both markets; BTC pre-2020 control: retains
62% of v4's balance vs the naive winner's 37%) -- but still failed the
project's promotion bar, because it still underperformed v4 on the BTC
control outright. R-45's own diagnosis, quoted directly: "the three purged
folds are all drawn from 2017-2022, so robustness across them cannot buy
robustness against the 2016-2019 BTC-control period, which none of the
folds ever sampled." That sentence names an untried experiment: a
resampling unit that is NOT three sub-intervals of the one realized window.
This round supplies two structurally different candidates for that unit.

**Which constraint each branch attacks: N approx 3** (effective sample size
is approx 3 regime events, not 1.01M bars) -- directly, via the selection
PROCEDURE, exactly as R-45 did, not via a new trading signal. Both branches
leave `kelly_regime_v4`'s vote/scale MECHANISM completely unchanged (the
same `frac * scale` architecture, R-62's finding); only which POINT on the
already-validated (ladder, target_vol, max_leverage) grid is selected can
differ.

**Literature grounding, fetched via WebSearch this round:**

- Politis, D. N., & Romano, J. P. (1994), "The Stationary Bootstrap,"
  *Journal of the American Statistical Association* 89(428), 1303-1313.
  This project's OWN `tradebot.inference.stationary_bootstrap_indices` is
  already this exact estimator, used since R-29 for INFERENCE (confidence
  intervals on a fixed, already-shipped strategy's realized P&L). This
  round reuses the identical function for a purpose it has never been used
  for in this ledger: CALIBRATION -- building synthetic alternative
  histories the SELECTION procedure is scored against, not just describing
  uncertainty about an already-frozen choice.
- Merton, R. C. (1976), "Option Pricing When Underlying Stock Returns Are
  Discontinuous," *Journal of Financial Economics* 3(1-2), 125-144
  (jump-diffusion); Hamilton, J. D. (1989), "A New Approach to the
  Economic Analysis of Nonstationary Time Series and the Business Cycle,"
  *Econometrica* 57(2), 357-384 (regime-switching). R-01 (08-15) already
  read and rejected Hamilton's HMM for the DETECTION role (inferring the
  live regime from data as it arrives) -- this round reuses the same
  regime-switching mathematics for a structurally different role, pure
  forward SIMULATION with parameters fit once from training data, never
  updated online and never used to time a trade. This project's own
  `tradebot.data.generate_synthetic_pair` (in `src/tradebot/data.py`,
  shipped since before this ledger's R-01) is already a 3-state
  Markov-switching, clustered-vol, jump-diffusion generator in exactly
  this shape -- but with fixed, illustrative constants (drift/vol per
  regime, transition rate) chosen by hand as plausible-looking fallback
  test data, never fit to real BTC data and never used to test anything.
  This round's novel branch is the first to fit that SAME generative shape
  to real training-period returns and use the result for calibration.
- Two 2024-2025 papers on synthetic-data-for-backtesting, found this round
  and directly on point: "Backtest overfitting in the machine learning
  era: A comparison of out-of-sample testing methods in a synthetic
  controlled environment" (*Expert Systems with Applications*, 2024,
  ScienceDirect S0950705124011110) argues synthetic controlled
  environments are the only way to KNOW the ground-truth edge a selection
  procedure recovers, which real markets never provide; and "Enhancing
  Equity Strategy Backtesting with Synthetic Data: An Agent-Based Model
  Approach" (AWS/industry writeup summarizing the practitioner case,
  2024-2025) makes the calibration-robustness argument this round tests
  directly: synthetic scenario generation lets a selection procedure be
  scored against "a wide range of plausible market scenarios" instead of
  the one path that happened. Motivating, not load-bearing -- like every
  paper this ledger cites without a verified cost model on this project's
  own data (R-105's Baltas & Kosowski, R-116's Zaremba et al.), both
  branches re-measure everything from scratch on this project's own data,
  fee tier and promotion bar.

**Not a duplicate of:**

- R-45 (the direct precedent, see above): resampled the ONE REAL WINDOW
  into three non-overlapping CALENDAR folds and selected by worst-fold
  Sharpe. Neither branch below constructs a calendar fold, and neither
  scores a config on any REAL sub-interval during selection -- selection
  here happens entirely on SYNTHETIC paths, real data is touched only at
  the frozen falsification step (Step 4), exactly once per branch, after
  the config is already chosen.
- R-40 (bagging the anchor-ladder plateau): blends several REAL-data
  ladder points' SIGNALS into one traded ensemble vote, using each
  member's own real historical performance. Neither branch below ever
  trades an ensemble or blends signals -- both select ONE single
  (ladder, target_vol, max_leverage) triple from the existing v4-family
  grid and trade it alone, unchanged from how v3/v4 already trade a single
  triple.
- R-06/R-07/R-37 (the original point-estimate grid searches that
  established the 18-28 day plateau and the target_vol/max_leverage
  frontier): single-path, pooled-window optimization -- the baseline both
  this round and R-45 are trying to improve on, not a resampling or
  robustness method at all.
- `scripts/inference.py`'s own `bootstrap()`/`stress_test.py`: both
  resample or window REAL, already-realized history to describe
  UNCERTAINTY about an already-frozen, already-shipped strategy's
  performance (a reporting/inference use). Neither is ever used here to
  CHOOSE a configuration -- this round is the first to point either kind
  of resampling (stationary block bootstrap; regime-switching Monte Carlo)
  at the SELECTION step instead of the reporting step.
- `tradebot.data.generate_synthetic_pair`: ships today as a fixed-constant
  fallback DATA SOURCE (used only when no real file is present, per
  README's data-priority table) -- never fit to real data, never touched
  by any strategy-selection code. The novel branch below is structurally
  the same GENERATIVE FORM, genuinely fit via MLE/moment-matching to real
  training-period returns and used for calibration -- the first time this
  round's underlying mathematics has been used for anything but filler
  data.

**Is it simulable here?** Yes, no new data channel. Both branches consume
only the committed 5-minute OHLCV file, the same one `kelly_regime_v4`
already trades on. Every synthetic path is itself a plain OHLCV frame fed
through the UNCHANGED real engine (`tradebot.engine.run_backtest`,
`tradebot.window.run_period`) -- no proxy metric, no shortcut scoring path.
The final, decision-bearing numbers (Step 4 below) are measured on 100%
REAL market data, exactly like every prior round; synthetic paths are used
ONLY to select which configuration reaches that real evaluation.

**What would make each branch fail, named now, before any code ran:**

- Conservative (stationary block bootstrap of real inner-train bars): this
  is a resampling of the SAME realized window's own blocks, just reordered
  -- every synthetic path is built entirely out of pieces of 2017-2020, so
  its regime-TYPE content (how many bull/bear/chop episodes exist to draw
  blocks from, and how extreme each one got) is bounded by exactly the
  same three-ish events R-45's calendar folds already had. The
  pre-registered expectation, following R-45's own diagnosis to its
  logical conclusion, is that this branch reproduces R-45's exact ceiling:
  real robustness gain over the naive point estimate (a lower-variance,
  less lucky-path-dependent choice), but no real gain on the pre-2020 BTC
  control, because no amount of reshuffling 2017-2020's own blocks can
  manufacture exposure to what 2016-2019 or a genuinely different bear
  looked like. A clean NEGATIVE that reproduces R-45's mechanism through a
  structurally different (finer-grained, stochastic block, not
  fixed-calendar) resampling unit is the fully expected, fully successful
  outcome of this branch.
- Novel (fitted 3-state Markov-switching jump-diffusion Monte Carlo): even
  though this branch CAN synthesize regime sequences/durations that never
  literally occurred (unlike the conservative branch, which can only
  recombine what did), its regime-TYPE statistics (how many states, their
  drift/vol, the transition rate, the jump distribution) are themselves
  ESTIMATED from the same narrow 2017-2020 window. If the fitted model's
  bear-state parameters never span anything worse than what 2018 actually
  did, sampling many sequences from it still cannot expose the selection
  criterion to a bear regime meaningfully different in KIND from the one
  the window already contained -- a more precise version of R-45's same
  diagnosis (bounded by the window's PARAMETRIC family even when not
  bounded by its literal REALIZED sequence). If this happens, the informative
  finding is that both branches (and, transitively, R-45) share one
  ceiling: no resampling or low-order generative method fit to ONE
  training window can manufacture genuine exposure to a regime the window
  never contained in even approximate form, and only real additional data
  (ETH, the six-asset panel -- both already closed) or forward evidence
  (B-06) can.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither may edit it. Nothing here reads a bar
at or after OOS_START (2023-01-01); every function that walks the main BTC
frame is explicitly restricted to INNER_TRAIN/INNER_VAL slices, and the
falsification pair (`btcusd_bitfinex_5m.csv.gz`, `ethusd_bitfinex_5m.csv.gz`)
is used whole-file because both files end 2019-12-31, entirely pre-2020 --
the identical convention R-17/R-28/R-31/R-33/R-37/R-38/R-40/R-45 already
established.
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
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import (  # noqa: E402
    annualized_sharpe,
    daily_returns,
    max_drawdown_from_returns,
    paired_bootstrap,
    stationary_bootstrap_indices,
    total_log_return,
)
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch was dispatched.
# ------------------------------------------------------------------------
BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures_5x", FUTURES))

INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

FEE_TIER = 0.0040                 # 0.40% taker, the real-tier robustness check (B5-style)
SHARPE_NOISE_FLOOR = 0.2          # ROUTINE.md's own promotion bar

# Grid: the SAME three axes R-06/R-07 (ladder base, 18-28d plateau),
# R-37 (target_vol/max_leverage) and R-45 (identical three-axis grid,
# 6x3x3=54 points, calendar-fold-minimax-selected) already swept.
# Coarsened here to 3x2x2=12 -- disclosed now, before any run -- because
# this round evaluates every grid point across N_DRAWS synthetic paths
# apiece (12 x 40 = 480 backtests per branch's selection sweep) rather
# than once, and the two extra grid axes R-45 had (28 further ladder
# bases and vol/leverage combinations) sit inside the *plateau* R-06/R-07
# already established, not at a boundary the coarsening could plausibly
# hide a different optimum outside of.
LADDER_BASES = (20, 26, 32)
TARGET_VOL_GRID = (0.45, 0.55)
MAX_LEV_GRID = (2.0, 2.5)
GRID = tuple((b, tv, ml) for b in LADDER_BASES for tv in TARGET_VOL_GRID for ml in MAX_LEV_GRID)
assert len(GRID) == 12

N_DRAWS = 40                       # synthetic paths per grid point, per branch (selection sweep)
CVAR_FRACTION = 0.25               # robust-selection criterion: mean of worst 25% of N_DRAWS
MEAN_BLOCK_DAYS = 30.0             # stationary-bootstrap block length (R-20's noise-floor setting, reused)

V4_DEFAULT = (20, 0.55, 2.0)       # kelly_regime_v4's own shipped (base, target_vol, max_leverage)


# ------------------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------------------

def load_btc_full() -> pd.DataFrame:
    df, _label = load_dataset(ROOT / "data", "spot")
    return df


def load_inner_train_btc() -> pd.DataFrame:
    df = load_btc_full()
    return df.loc[INNER_TRAIN_START:INNER_TRAIN_END]


def load_bitfinex_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(BTC control, ETH test), both Bitfinex, whole-file -- both end
    2019-12-31, entirely pre-2020, the R-17/R-28/.../R-45 convention."""
    btc = load_ohlcv_csv(ROOT / "data" / "btcusd_bitfinex_5m.csv.gz")
    eth = load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")
    assert btc.index[-1] < pd.Timestamp("2020-01-01", tz="UTC")
    assert eth.index[-1] < pd.Timestamp("2020-01-01", tz="UTC")
    return btc, eth


def assert_no_holdout(df: pd.DataFrame, label: str = "") -> None:
    if len(df) and df.index[-1] >= pd.Timestamp(OOS_START, tz="UTC"):
        raise AssertionError(f"holdout touched: {label} reaches {df.index[-1]}")


# ------------------------------------------------------------------------
# The v4-family strategy factory both branches select over
# ------------------------------------------------------------------------

def build_kelly(base: int, target_vol: float, max_leverage: float) -> KellyRegimeV3:
    """`kelly_regime_v4`'s OWN class (KellyRegimeV3's extreme-only
    conditional vol-targeting mechanism, V4's doubling ladder), with only
    (base, target_vol, max_leverage) varied -- everything else (band,
    vol_span, deadband, vote_gamma, anchor_span_days, the hi/lo hysteresis
    bands) left at v4's own shipped defaults. Verified by self-test below
    to reproduce `kelly_regime_v4` bit-for-bit at `V4_DEFAULT`.

    `KellyRegimeV3` (unlike `KellyRegimeV4`) does not override `warmup` --
    it inherits the base `KellyRegime.warmup = 100*BARS_PER_DAY+10`, sized
    for the DEFAULT (30,50,100) ladder, not whatever `base` this factory is
    given. Left alone, every grid point except base=25 would warm up on
    the wrong number of days (too little for base>25, so the slowest
    anchor's rolling mean is not yet fully populated when trading starts).
    Set explicitly here to `4*base` days, matching V4's own convention of
    sizing warmup to its own longest anchor.
    """
    strat = KellyRegimeV3(horizons=(base, 2 * base, 4 * base),
                          target_vol=target_vol, max_leverage=max_leverage)
    strat.warmup = 4 * base * BARS_PER_DAY + 10
    return strat


# ------------------------------------------------------------------------
# Scoring on one OHLCV path (real or synthetic) -- IDENTICAL code path for
# both, since a synthetic path is just another DataFrame the real engine
# does not distinguish from real data.
# ------------------------------------------------------------------------

def daily_rets_for(strategy, df: pd.DataFrame, market: MarketSpec,
                   balance: float = 1_000.0) -> tuple[np.ndarray, float]:
    """Daily returns and final equity for one whole-``df`` backtest."""
    result = run_backtest(strategy, df, market, balance)
    rets = daily_returns(result.equity).to_numpy(dtype=float)
    final = float(result.equity.iloc[-1]) if len(result.equity) else balance
    return rets, final


def score_on_path(config: tuple[int, float, float], df_path: pd.DataFrame,
                  market: MarketSpec = SPOT) -> float:
    """Annualized Sharpe of one (base, target_vol, max_leverage) config on
    one path (real or synthetic). A liquidated/ruined run scores -10.0 --
    a fixed, worse-than-anything-observed sentinel, never NaN'd away by a
    downstream mean/quantile."""
    base, tv, ml = config
    strat = build_kelly(base, tv, ml)
    rets, final = daily_rets_for(strat, df_path, market)
    if not np.isfinite(final) or final <= 0 or len(rets) < 30:
        return -10.0
    return float(annualized_sharpe(rets))


def robust_score(sharpes: np.ndarray, frac: float = CVAR_FRACTION) -> float:
    """CVaR-style robust criterion: mean of the worst ``frac`` of draws.
    Shared verbatim by both branches -- the ONLY selection rule either may
    use, so a difference in outcome reflects the path generator, not a
    difference in how "robust" is scored."""
    s = np.sort(np.asarray(sharpes, dtype=float))
    k = max(1, int(np.ceil(frac * len(s))))
    return float(np.mean(s[:k]))


def select_config(path_generator, n_draws: int = N_DRAWS, grid=GRID,
                  market: MarketSpec = SPOT) -> tuple[tuple, dict]:
    """Generic robust-selection loop, IDENTICAL for both branches: draw
    ``n_draws`` synthetic paths once from ``path_generator(seed)`` (seed =
    0..n_draws-1), score every grid config on every path, select by
    ``robust_score``. Selection runs on SPOT only (compute budget,
    disclosed) -- the selected config is then evaluated for real on BOTH
    markets in ``evaluate_candidate`` below."""
    paths = [path_generator(seed) for seed in range(n_draws)]
    table = {}
    for cfg in grid:
        sharpes = np.array([score_on_path(cfg, p, market) for p in paths])
        table[cfg] = dict(mean=float(sharpes.mean()), std=float(sharpes.std()),
                          robust=robust_score(sharpes))
    best = max(table, key=lambda c: table[c]["robust"])
    return best, table


# ------------------------------------------------------------------------
# Step 4: the frozen, real-data-only evaluation. IDENTICAL for both
# branches -- called exactly once per branch, on the ONE config
# `select_config` returned, after selection is complete and frozen.
# ------------------------------------------------------------------------

def evaluate_candidate(config: tuple[int, float, float], label: str) -> dict:
    """All pre-registered real-data checks for one frozen candidate,
    against kelly_regime_v4's own shipped defaults as the control. Never
    reads a bar at or after OOS_START."""
    base, tv, ml = config
    out = {"label": label, "config": config}

    btc = load_btc_full()
    assert_no_holdout(btc.loc[:INNER_VAL_END], f"{label} inner_val")

    # --- inner-validation, both markets, real data, paired bootstrap ---
    inner_val = []
    for mname, market in MARKETS:
        cand = build_kelly(base, tv, ml)
        ctrl = get_strategy("kelly_regime_v4")
        rc = run_period(cand, btc, INNER_VAL_START, INNER_VAL_END, market=market)
        rv = run_period(ctrl, btc, INNER_VAL_START, INNER_VAL_END, market=market)
        dr_c = daily_returns(rc.equity).to_numpy(dtype=float)
        dr_v = daily_returns(rv.equity).to_numpy(dtype=float)
        n = min(len(dr_c), len(dr_v))
        dr_c, dr_v = dr_c[:n], dr_v[:n]
        sharpe_pair = paired_bootstrap(dr_c, dr_v, annualized_sharpe,
                                       mean_block=MEAN_BLOCK_DAYS, seed=118)
        dd_c = max_drawdown_from_returns(dr_c)
        dd_v = max_drawdown_from_returns(dr_v)
        inner_val.append(dict(
            market=mname, sharpe_cand=sharpe_pair.stat_a, sharpe_ctrl=sharpe_pair.stat_b,
            d_sharpe=sharpe_pair.diff.point, d_sharpe_lo=sharpe_pair.diff.lo,
            d_sharpe_hi=sharpe_pair.diff.hi, dd_cand=dd_c, dd_ctrl=dd_v,
            final_cand=float(rc.equity.iloc[-1]), final_ctrl=float(rv.equity.iloc[-1]),
            liquidated_cand=bool(rc.equity.iloc[-1] <= 0),
        ))
    out["inner_val"] = inner_val
    out["b1_pass"] = all(
        (r["d_sharpe"] > SHARPE_NOISE_FLOOR) or (r["d_sharpe_lo"] > 0) for r in inner_val
    )

    # --- 0.40% fee tier, inner-validation, both markets (B5-style) ---
    fee_rows = []
    for mname, market in MARKETS:
        fee_market = (MarketSpec.spot(fee_rate=FEE_TIER) if mname == "spot"
                     else MarketSpec.futures(leverage=5.0, fee_rate=FEE_TIER))
        cand = build_kelly(base, tv, ml)
        rc = run_period(cand, btc, INNER_VAL_START, INNER_VAL_END, market=fee_market)
        dr_c = daily_returns(rc.equity).to_numpy(dtype=float)
        base_row = next(r for r in inner_val if r["market"] == mname)
        fee_rows.append(dict(market=mname, sharpe=float(annualized_sharpe(dr_c)),
                             base_sharpe=base_row["sharpe_cand"],
                             no_reversal=bool(np.sign(annualized_sharpe(dr_c)) ==
                                              np.sign(base_row["sharpe_cand"])
                                              or base_row["sharpe_cand"] == 0)))
    out["b5_fee_tier"] = fee_rows
    out["b5_pass"] = all(r["no_reversal"] for r in fee_rows)

    # --- falsification: pre-2020 BTC control (Bitfinex) vs ETH (Bitfinex) ---
    btc_ctrl, eth_test = load_bitfinex_pair()
    falsification = []
    for asset, df_f in (("BTC_control", btc_ctrl), ("ETH_test", eth_test)):
        for mname, market in MARKETS:
            cand = build_kelly(base, tv, ml)
            ctrl = get_strategy("kelly_regime_v4")
            rc = run_backtest(cand, df_f, market, 1_000.0)
            rv = run_backtest(ctrl, df_f, market, 1_000.0)
            dr_c = daily_returns(rc.equity).to_numpy(dtype=float)
            dr_v = daily_returns(rv.equity).to_numpy(dtype=float)
            d_sharpe = (float(annualized_sharpe(dr_c)) - float(annualized_sharpe(dr_v))
                       if len(dr_c) >= 3 and len(dr_v) >= 3 else float("nan"))
            d_profit = (100.0 * (rc.equity.iloc[-1] - rv.equity.iloc[-1]) / 1_000.0)
            falsification.append(dict(asset=asset, market=mname, d_sharpe=d_sharpe,
                                      d_profit_pp=float(d_profit)))
    out["falsification"] = falsification
    # R-45's own bar, reused verbatim: candidate must not visibly
    # underperform v4 on the BTC control, and must be at least comparable
    # on ETH -- both by the identical margin R-45 pre-registered.
    out["falsification_pass"] = all(
        (r["d_sharpe"] > -0.05 and r["d_profit_pp"] > -2.0) for r in falsification
    )

    out["promote"] = bool(out["b1_pass"] and out["b5_pass"] and out["falsification_pass"])
    return out


def print_report(result: dict) -> None:
    print(f"\n{'=' * 78}\n{result['label']}  config={result['config']}\n{'=' * 78}")
    print("-- inner-validation (real, 2021-2022) --")
    for r in result["inner_val"]:
        print(f"  {r['market']:11s} sharpe cand={r['sharpe_cand']:+.2f} "
              f"ctrl={r['sharpe_ctrl']:+.2f}  dSharpe={r['d_sharpe']:+.2f} "
              f"[{r['d_sharpe_lo']:+.2f},{r['d_sharpe_hi']:+.2f}]  "
              f"DD cand={r['dd_cand']:.1f}% ctrl={r['dd_ctrl']:.1f}%")
    print(f"  B1 (beats v4 by noise floor or CI excludes zero, both markets): "
          f"{'PASS' if result['b1_pass'] else 'FAIL'}")
    print("-- 0.40% fee tier (B5) --")
    for r in result["b5_fee_tier"]:
        print(f"  {r['market']:11s} sharpe@0.40%={r['sharpe']:+.2f} "
              f"(base {r['base_sharpe']:+.2f})  no_reversal={r['no_reversal']}")
    print(f"  B5: {'PASS' if result['b5_pass'] else 'FAIL'}")
    print("-- falsification: BTC pre-2020 control vs ETH (Bitfinex, whole-file) --")
    for r in result["falsification"]:
        print(f"  {r['asset']:11s} {r['market']:11s} dSharpe={r['d_sharpe']:+.3f} "
              f"dProfit={r['d_profit_pp']:+.1f}pp")
    print(f"  falsification: {'PASS' if result['falsification_pass'] else 'FAIL'}")
    print(f"\n  VERDICT: {'PROMOTE-candidate' if result['promote'] else 'NEGATIVE'}")


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    # robust_score: worst-25%-of-40 is the 10 lowest draws' mean.
    rng = np.random.default_rng(118)
    x = rng.normal(0, 1, 40)
    expect = np.sort(x)[:10].mean()
    assert abs(robust_score(x) - expect) < 1e-9

    # score_on_path is finite and deterministic on a tiny synthetic frame.
    idx = pd.date_range("2017-01-01", periods=120_000, freq="5min", tz="UTC")
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    df = pd.DataFrame({"open": close, "high": close * 1.0006, "low": close * 0.9994,
                       "close": close, "volume": 1.0}, index=idx)
    s1 = score_on_path(V4_DEFAULT, df)
    s2 = score_on_path(V4_DEFAULT, df)
    assert s1 == s2
    assert np.isfinite(s1)

    # GRID / bitfinex-pair sanity.
    assert len(GRID) == 12
    assert V4_DEFAULT[0] == 20 and V4_DEFAULT[1] == 0.55 and V4_DEFAULT[2] == 2.0

    # build_kelly(*V4_DEFAULT) must reproduce kelly_regime_v4 bit-for-bit:
    # same prepare()-computed target array on the same frame.
    v4 = get_strategy("kelly_regime_v4")
    cand = build_kelly(*V4_DEFAULT)
    df_v4 = v4.prepare(df.copy())
    df_cand = cand.prepare(df.copy())
    assert np.allclose(df_v4["target"].to_numpy(), df_cand["target"].to_numpy(),
                       equal_nan=True), "build_kelly(V4_DEFAULT) != kelly_regime_v4"


_self_test()
