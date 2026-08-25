"""R-135 NOVEL branch: a data-driven breadth-optimized slot-selection
procedure applied to `hedge_experts`'s own ten-expert panel (08-25). See
`experiments/r135_shared.py`'s module docstring for the full round-level
pre-registration (direction, literature, both branches' mechanisms,
non-duplicate check, named failure modes, falsification test, decision
rule) -- this file freezes the NOVEL branch's own exact mechanism and
reports its evaluation. Nothing here deviates from that pre-registration
without saying so explicitly.

**The two-step mechanism, frozen before any performance number was read:**

**Step 1 -- breadth-based slot selection**, computed ONCE via the pure
function `select_drop_slots()` below, on INNER-TRAIN ONLY BTC spot bars
(2017-01-01 to 2020-12-31), never inner-validation or later:
`HedgeExperts._experts(HedgeExperts(), df_inner_train, r, sig1)` (the
unmodified base-class method, called statelessly) produces the original
ten expert position columns; their pairwise Pearson correlation matrix
(`np.corrcoef`, columns as variables) is computed over inner-train; for
each of experts 0-7 (excluding the two structurally-constant columns,
always-flat (8) and buy-and-hold (9), whose correlation with everything
is undefined/NaN and which are therefore not candidates for "redundant"
removal), the mean absolute correlation to every OTHER expert is
computed (NaN-excluding `nanmean`); the TWO experts with the HIGHEST
mean absolute correlation -- the two contributing the least marginal
Grinold breadth -- are dropped. The result is frozen via
`freeze_drop_slots()`, which refuses (`assert`s) to overwrite an
already-frozen selection, a deliberate guard against ever moving this
goalpost after seeing a performance number. `__main__` below calls
`select_drop_slots`/`freeze_drop_slots` and prints the full selection
FIRST, before any evaluation code runs.

**Step 2 -- two new experts fill the two freed slots:**

1. **Positioning/crowding vote** (`_positioning_vote`) -- IDENTICAL
   construction to the CONSERVATIVE branch, so results are comparable
   across branches: `tradebot.data.load_binance_metrics`'s
   `count_long_short_ratio` column ONLY (not the toptrader columns,
   documented 37.6%-missing in 2022), log-transformed, rolling z-score
   (90-day-in-bars = 25920, min_periods 30-day-in-bars = 8640) computed on
   the raw 5-minute-native metrics frame BEFORE causal alignment
   (`align_metrics_causal`), contrarian sign
   (`vote = -tanh(z / POSITIONING_DIVISOR)`, De Roon/Nijman/Veld 2000):
   crowd net-long is risk-off, crowd net-short is risk-on. `.fillna(0.0)`
   for gaps/pre-coverage (BTC metrics start 2020-09-01, ETH 2021-12-01).

2. **Implied-vol variance-risk-premium (VRP) vote** (`_vrp_vote`) --
   `tradebot.data.load_dvol_index` for BTC; this file's own
   `_load_eth_dvol_index` (mirrors `load_dvol_index`'s exact
   implementation, reading `data/eth_dvol_daily.csv.gz` instead, does NOT
   modify `src/tradebot/data.py`) for ETH. `implied_vol_frac =
   dvol_close / 100.0`, causally aligned onto the bar grid via the
   asset-agnostic `align_dvol_causal` BEFORE subtracting;
   `realized_vol_ann = sig1 * sqrt(BARS_PER_YEAR)` (the strategy's own
   trailing EWM(span=288) log-return std, passed into `_experts` exactly
   as `HedgeExperts.prepare` computes it); `vrp = implied_vol_frac_aligned
   - realized_vol_ann.shift(1)` (shift(1) matches `_experts`'s own
   `sig1.shift(1)` convention used everywhere else in the base panel);
   THEN a rolling z-score of `vrp` (90-day-in-bars = 25920, min_periods
   8640) computed directly on the already-aligned bar-grid series (there
   is no separate raw daily frame to roll on here, unlike the positioning
   vote); positive sign (`vote = +tanh(z / VRP_DIVISOR)`,
   Bollerslev/Tauchen/Zhou 2009): positive VRP predicts near-term positive
   returns, a risk-on lean. `.fillna(0.0)` for the pre-2021-03-24 DVOL gap
   and any other gaps -- DVOL is flat (vote=0, acting as the always-flat
   expert) for essentially all of `inner_train` (2017-2020), stated here
   explicitly per failure mode #4 in `r135_shared.py`'s docstring, not
   hidden; see the B1 ablation split below for the direct check.

Both new votes are ordinary `tanh`-bounded `[-1, 1]` columns, exactly like
every other `HedgeExperts._experts` column, so the Hedge blend treats them
identically to the eight technical experts -- no special-casing.

`prepare()`/`on_bar()` are inherited verbatim from `HedgeExperts`, never
touched here; both new classes only override `_experts`.

**Ablation, pre-registered, not optional (isolates drop vs. add):**
`BreadthDropOnlyPanel` drops the SAME two breadth-selected experts and
adds NOTHING (8-expert panel). Comparing its B1 numbers against
`BreadthOptimizedPanel`'s separates "did the breadth-optimized drop alone
help or hurt" from "did the two new experts, on top of that drop, help or
hurt" -- critical given `inner_train` predates almost all coverage for
both new signals (failure mode #4).

**Causality.** Every new computation here (`rolling()`, `.shift()`) is
backward-looking only; verified below via the same truncation-probe idiom
`r135_shared.py`'s own `__main__` uses, applied separately to both
`BreadthOptimizedPanel` and `BreadthDropOnlyPanel`. No bar at or after
`OOS_START = 2023-01-01` is read anywhere in this file.

**Decision rule and falsification test:** verbatim from `r135_shared.py`
-- PROMOTE-candidate only if the causal-truncation probe AND A2 (both new
experts) AND B1 (both markets, majority of {inner_train, full_inner})
AND B3 (plateau majority) AND B4 (ETH sign replication) AND B5 (0.40% fee,
no sign flip) AND B6 (mandatory turnover-matched-control beat) all pass.
Nothing here moves that goalpost.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import (  # noqa: E402
    align_dvol_causal,
    align_metrics_causal,
    load_binance_metrics,
    load_dvol_index,
)
from tradebot.strategies.hedge_experts import HedgeExperts  # noqa: E402

from r135_shared import (  # noqa: E402
    B1_PERIODS,
    B3_MULTIPLIERS,
    BARS_PER_YEAR,
    FUTURES,
    FUTURES_HIGH_FEE,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SPOT,
    SPOT_HIGH_FEE,
    a2_non_inertness,
    effective_breadth,
    load_btc_train,
    load_eth_train,
    log_growth_diff,
    replay_hedge_weights,
    run_baseline,
    run_strategy,
    sharpe_diff,
    turnover_matched_control,
)

# ----------------------------------------------------------------------
# Frozen, a-priori parameters. Never swept for a "best" value -- fixed
# before any performance number was read; B3 sweeps AROUND these points
# purely as a plateau/peak diagnostic, not a search for a better one.
# ----------------------------------------------------------------------
POSITIONING_DIVISOR = 3.0
VRP_DIVISOR = 3.0

# 90-day / 30-day windows expressed in native 5-minute bars.
BARS_PER_DAY = 288
POSITIONING_ZSCORE_WINDOW_BARS = 90 * BARS_PER_DAY   # 25920
POSITIONING_ZSCORE_MIN_PERIODS_BARS = 30 * BARS_PER_DAY  # 8640
VRP_ZSCORE_WINDOW_BARS = 90 * BARS_PER_DAY           # 25920
VRP_ZSCORE_MIN_PERIODS_BARS = 30 * BARS_PER_DAY      # 8640

EXPERT_NAMES = [
    "mom_1h", "mom_6h", "mom_1d", "mom_1w",
    "macd", "rsi_ramp", "reversion_1bar", "donchian",
    "flat", "buy_hold",
]
_CANDIDATE_DROP_IDX = tuple(range(8))  # momentum x4, macd, rsi, reversion, donchian


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {OOS_START}")


# ----------------------------------------------------------------------
# Step 1: breadth-based slot selection. Pure function of the inner-train-
# only frame; frozen exactly once via freeze_drop_slots() before any
# evaluation code below depends on it.
# ----------------------------------------------------------------------

def select_drop_slots(df_inner_train: pd.DataFrame):
    """Compute, on the given (already inner-train-only) BTC spot frame,
    the original 10 experts' pairwise position correlation and return the
    two indices (from 0-7) with the highest mean-|correlation| to the rest
    of the panel, plus the full per-expert mean-|correlation| array for
    reporting. Pure: reads only `df_inner_train`, no global state."""
    r = np.log(df_inner_train["close"]).diff()
    sig1 = r.ewm(span=288, min_periods=250).std()
    a = HedgeExperts._experts(HedgeExperts(), df_inner_train, r, sig1)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(a, rowvar=False)
    n = corr.shape[0]
    mean_abs = np.full(n, np.nan)
    for i in range(n):
        others = np.array([corr[i, j] for j in range(n) if j != i], dtype=float)
        mean_abs[i] = np.nanmean(np.abs(others))
    ranked = sorted(_CANDIDATE_DROP_IDX, key=lambda i: mean_abs[i], reverse=True)
    drop = tuple(sorted(ranked[:2]))
    return drop, mean_abs


_DROP_SLOTS: tuple[int, int] | None = None


def freeze_drop_slots(drop: tuple[int, int]) -> None:
    """Freeze the Step-1 selection exactly once. Refuses to overwrite an
    already-frozen selection -- a deliberate programmatic guard against
    moving this goalpost after seeing any performance number."""
    global _DROP_SLOTS
    assert _DROP_SLOTS is None, (
        "slot selection already frozen -- refusing to overwrite "
        "(this would be moving the goalpost; see ROUTINE.md discipline)")
    _DROP_SLOTS = drop


def _drop_slots() -> tuple[int, int]:
    assert _DROP_SLOTS is not None, (
        "slot selection not yet frozen -- call select_drop_slots()/"
        "freeze_drop_slots() before running any candidate class")
    return _DROP_SLOTS


def _kept_experts(df: pd.DataFrame, r: pd.Series, sig1: pd.Series) -> np.ndarray:
    """Unmodified 10-expert base panel with the two frozen-dropped slots
    removed -> 8 columns, order-preserving."""
    base = HedgeExperts._experts(HedgeExperts(), df, r, sig1)
    drop = _drop_slots()
    keep_idx = [i for i in range(10) if i not in drop]
    return base[:, keep_idx]


# ----------------------------------------------------------------------
# Step 2: the two new experts.
# ----------------------------------------------------------------------

def _load_eth_dvol_index(data_dir) -> pd.DataFrame | None:
    """ETH DVOL loader. Mirrors `tradebot.data.load_dvol_index`'s exact
    implementation, reading `eth_dvol_daily.csv.gz` instead of the
    hardcoded BTC file. Does not modify `src/tradebot/data.py`."""
    path = Path(data_dir) / "eth_dvol_daily.csv.gz"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = df.index.tz_localize("UTC")
    return df.astype(float).sort_index()


def _positioning_vote(df: pd.DataFrame, data_dir, asset: str, divisor: float) -> pd.Series:
    """Derivatives-positioning ("crowding") contrarian vote. Identical
    construction to the CONSERVATIVE branch. See module docstring."""
    metrics = load_binance_metrics(data_dir, asset=asset)
    if metrics is None:
        return pd.Series(0.0, index=df.index)
    log_ratio = np.log(metrics["count_long_short_ratio"].clip(lower=1e-9))
    mean = log_ratio.rolling(POSITIONING_ZSCORE_WINDOW_BARS,
                              min_periods=POSITIONING_ZSCORE_MIN_PERIODS_BARS).mean()
    std = log_ratio.rolling(POSITIONING_ZSCORE_WINDOW_BARS,
                             min_periods=POSITIONING_ZSCORE_MIN_PERIODS_BARS).std()
    z = ((log_ratio - mean) / std).rename("pos_z").to_frame()
    z_aligned = align_metrics_causal(z, df)["pos_z"]
    vote = -np.tanh(z_aligned / divisor)
    return vote.fillna(0.0)


def _vrp_vote(df: pd.DataFrame, sig1: pd.Series, data_dir, asset: str, divisor: float) -> pd.Series:
    """Implied-vol variance-risk-premium vote. See module docstring."""
    if asset == "BTC":
        dvol = load_dvol_index(data_dir)
    elif asset == "ETH":
        dvol = _load_eth_dvol_index(data_dir)
    else:
        raise ValueError(f"unsupported asset {asset!r}")
    if dvol is None:
        return pd.Series(0.0, index=df.index)
    implied_vol_frac = (dvol[["close"]] / 100.0).rename(columns={"close": "iv_frac"})
    iv_aligned = align_dvol_causal(implied_vol_frac, df)["iv_frac"]
    realized_vol_ann = sig1 * np.sqrt(BARS_PER_YEAR)
    vrp = iv_aligned - realized_vol_ann.shift(1)
    mean = vrp.rolling(VRP_ZSCORE_WINDOW_BARS, min_periods=VRP_ZSCORE_MIN_PERIODS_BARS).mean()
    std = vrp.rolling(VRP_ZSCORE_WINDOW_BARS, min_periods=VRP_ZSCORE_MIN_PERIODS_BARS).std()
    z = (vrp - mean) / std
    vote = np.tanh(z / divisor)
    return vote.fillna(0.0)


# ----------------------------------------------------------------------
# The two candidate classes. Only `_experts` is overridden; `prepare`/
# `on_bar` are inherited verbatim from `HedgeExperts`.
# ----------------------------------------------------------------------

class BreadthDropOnlyPanel(HedgeExperts):
    """Ablation (pre-registered, not optional): drop the SAME two
    breadth-selected experts, add NOTHING. 8-expert panel."""

    name = "r135_novel_breadth_drop_only_ablation"

    def _experts(self, df: pd.DataFrame, r: pd.Series, sig1: pd.Series) -> np.ndarray:
        kept = _kept_experts(df, r, sig1)
        return np.nan_to_num(kept, nan=0.0)


class BreadthOptimizedPanel(HedgeExperts):
    """Full NOVEL construction: breadth-optimized drop of the two least-
    marginal-breadth experts (Step 1), replaced by the positioning and
    VRP votes (Step 2). 10-expert panel, same count as the original."""

    name = "r135_novel_breadth_optimized_panel"

    def __init__(self, eta: float = 0.05, fixed_share: float = 1e-4,
                 hysteresis: float = 0.05, fee_rate: float = 0.0005,
                 data_dir=None, asset: str = "BTC",
                 positioning_divisor: float = POSITIONING_DIVISOR,
                 vrp_divisor: float = VRP_DIVISOR) -> None:
        super().__init__(eta=eta, fixed_share=fixed_share, hysteresis=hysteresis,
                          fee_rate=fee_rate)
        self.data_dir = Path(data_dir) if data_dir is not None else ROOT / "data"
        self.asset = asset
        self.positioning_divisor = positioning_divisor
        self.vrp_divisor = vrp_divisor

    def _experts(self, df: pd.DataFrame, r: pd.Series, sig1: pd.Series) -> np.ndarray:
        kept = _kept_experts(df, r, sig1)
        pos_vote = _positioning_vote(df, self.data_dir, self.asset, self.positioning_divisor)
        vrp_vote = _vrp_vote(df, sig1, self.data_dir, self.asset, self.vrp_divisor)
        a = np.column_stack([kept,
                              np.asarray(pos_vote, dtype=np.float64),
                              np.asarray(vrp_vote, dtype=np.float64)])
        return np.nan_to_num(a, nan=0.0)


# ----------------------------------------------------------------------
# Diagnostic helpers (mirrors r132_novel_diversified_panel.py's own).
# ----------------------------------------------------------------------

def _expert_position_matrix(strat, df: pd.DataFrame, start: str, end: str) -> np.ndarray:
    r = np.log(df["close"]).diff()
    sig1 = r.ewm(span=288, min_periods=250).std()
    a = strat._experts(df, r, sig1)
    idx = df.index
    tz = idx.tz
    mask = (idx >= pd.Timestamp(start, tz=tz)) & (idx <= pd.Timestamp(end, tz=tz))
    return a[mask]


def _report_diff(name: str, res_cand, res_base) -> dict:
    sh = sharpe_diff(res_cand, res_base)
    lg = log_growth_diff(res_cand, res_base)
    print(f"  [{name}] d_sharpe={sh.diff.point:+.4f} CI=[{sh.diff.lo:+.4f},{sh.diff.hi:+.4f}] "
          f"p_pos={sh.p_positive:.3f} sig={sh.significant} | "
          f"d_logret={lg.diff.point:+.4f} CI=[{lg.diff.lo:+.4f},{lg.diff.hi:+.4f}] "
          f"p_pos={lg.p_positive:.3f} sig={lg.significant}")
    return {"name": name, "d_sharpe": sh.diff.point, "sharpe_lo": sh.diff.lo,
            "sharpe_hi": sh.diff.hi, "sharpe_sig": sh.significant,
            "d_logret": lg.diff.point, "logret_lo": lg.diff.lo,
            "logret_hi": lg.diff.hi, "logret_sig": lg.significant}


if __name__ == "__main__":
    n_configs = 0
    results: dict[str, dict] = {}

    df_btc, label_btc = load_btc_train("spot")

    # ------------------------------------------------------------
    # STEP 1: breadth-based slot selection. Printed first, before any
    # evaluation code runs. Pure function of inner-train-only BTC spot.
    # ------------------------------------------------------------
    print("=" * 78)
    print("STEP 1: breadth-based slot selection (inner-train ONLY, 2017-2020)")
    print("=" * 78)
    df_it = df_btc.loc[INNER_TRAIN_START:INNER_TRAIN_END].copy()
    _assert_no_holdout(df_it)
    drop, mean_abs = select_drop_slots(df_it)
    for i in range(10):
        marker = "  <== DROPPED (highest mean|corr|, least marginal breadth)" if i in drop else ""
        print(f"  [{i}] {EXPERT_NAMES[i]:16s} mean|corr to rest|={mean_abs[i]:.4f}{marker}")
    freeze_drop_slots(drop)
    print(f"  FROZEN drop slots: {drop} -> {[EXPERT_NAMES[i] for i in drop]}")
    print("  (computed once on inner-train only; will NOT be recomputed after this point)")

    # ------------------------------------------------------------
    # Causal-truncation probe (must PASS) -- both classes.
    # ------------------------------------------------------------
    print()
    print("=" * 78)
    print("Causal-truncation probe")
    print("=" * 78)
    probe_ok = True
    for cls_name, strat_factory in (
        ("BreadthOptimizedPanel", lambda: BreadthOptimizedPanel()),
        ("BreadthDropOnlyPanel", lambda: BreadthDropOnlyPanel()),
    ):
        m_full, _ = run_strategy(strat_factory(), df_btc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label_btc)
        n_configs += 1
        df_trunc = df_btc.loc[:INNER_VAL_END].copy()
        _assert_no_holdout(df_trunc)
        m_trunc, _ = run_strategy(strat_factory(), df_trunc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label_btc)
        n_configs += 1
        ok = np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9)
        probe_ok = probe_ok and ok
        print(f"  [{cls_name}] {'PASS' if ok else 'FAIL'} "
              f"({m_full.final_balance} vs {m_trunc.final_balance})")
    print(f"CAUSAL PROBE OVERALL: {'PASS' if probe_ok else 'FAIL'}")
    assert probe_ok, "a candidate class reads ahead of its own truncation point"

    # ------------------------------------------------------------
    # A2 (non-inertness): each new expert's own Hedge weight p_i,
    # inner-validation, BTC spot.
    # ------------------------------------------------------------
    print()
    print("=" * 78)
    print("A2: non-inertness of the two new experts (BTC spot, inner-validation)")
    print("=" * 78)
    full_panel = BreadthOptimizedPanel()
    p_hist = replay_hedge_weights(full_panel, df_btc)
    idx = df_btc.index
    mask_val = (idx >= pd.Timestamp(INNER_VAL_START, tz=idx.tz)) & (idx <= pd.Timestamp(INNER_VAL_END, tz=idx.tz))
    num_experts = p_hist.shape[1]
    pos_col, vrp_col = num_experts - 2, num_experts - 1
    a2_pos = a2_non_inertness(p_hist[mask_val, pos_col], num_experts)
    a2_vrp = a2_non_inertness(p_hist[mask_val, vrp_col], num_experts)
    print(f"  Positioning expert: frac_bars_above_2x_uniform={a2_pos['frac_bars_above_2x_uniform']:.4f} "
          f"pass={a2_pos['pass']}")
    print(f"  VRP expert:         frac_bars_above_2x_uniform={a2_vrp['frac_bars_above_2x_uniform']:.4f} "
          f"pass={a2_vrp['pass']}")
    print("  (replay computation, not a broker backtest -- not added to n_configs)")
    a2_gate = bool(a2_pos["pass"]) and bool(a2_vrp["pass"])

    # ------------------------------------------------------------
    # Diagnostic (reported, not gating): pairwise expert-position
    # correlation + Grinold effective breadth, original 10-expert panel
    # vs full novel 10-expert panel, inner-validation, BTC spot.
    # ------------------------------------------------------------
    print()
    print("=" * 78)
    print("Diagnostic: expert-position correlation / effective breadth (inner-val, BTC spot)")
    print("=" * 78)
    a_orig = _expert_position_matrix(HedgeExperts(), df_btc, INNER_VAL_START, INNER_VAL_END)
    a_novel = _expert_position_matrix(full_panel, df_btc, INNER_VAL_START, INNER_VAL_END)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_orig = np.corrcoef(a_orig, rowvar=False)
        corr_novel = np.corrcoef(a_novel, rowvar=False)
    br_orig = effective_breadth(corr_orig)
    br_novel = effective_breadth(corr_novel)
    print(f"  original 10-expert panel:  BR_eff = {br_orig:.4f}  "
          f"(mean |off-diag corr| = {np.nanmean(np.abs(corr_orig[~np.eye(10, dtype=bool)])):.4f})")
    print(f"  full novel 10-expert panel: BR_eff = {br_novel:.4f}  "
          f"(mean |off-diag corr| = {np.nanmean(np.abs(corr_novel[~np.eye(10, dtype=bool)])):.4f})")
    print(f"  breadth {'ROSE' if br_novel > br_orig else 'DID NOT RISE'} "
          f"({br_orig:.4f} -> {br_novel:.4f})")
    print("  (correlation/breadth diagnostic -- not a broker backtest, not added to n_configs)")

    # ------------------------------------------------------------ B1
    print()
    print("=" * 78)
    print("B1: full construction AND breadth-drop-only ablation vs frozen hedge_experts")
    print("   (both markets x all THREE B1_PERIODS, both classes -- 24 backtest configs)")
    print("=" * 78)
    b1_metrics: dict[tuple[str, str, str], dict] = {}  # (cls,mkt,per) -> {"m_base","m_cand","res_base","res_cand"}
    for cls_name, strat_factory in (
        ("full", lambda: BreadthOptimizedPanel()),
        ("drop_only", lambda: BreadthDropOnlyPanel()),
    ):
        for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
            for per_name, start, end in B1_PERIODS:
                m_base, res_base = run_baseline(df_btc, mkt, start, end, label_btc)
                n_configs += 1
                m_cand, res_cand = run_strategy(strat_factory(), df_btc, mkt, start, end, label_btc)
                n_configs += 1
                print(f"  [{cls_name} {mkt_name}/{per_name}] trades={m_cand.num_trades} "
                      f"final={m_cand.final_balance:.1f} sharpe={m_cand.sharpe:.3f} | "
                      f"baseline trades={m_base.num_trades} sharpe={m_base.sharpe:.3f}")
                key = f"B1_{cls_name}_{mkt_name}_{per_name}"
                results[key] = _report_diff(key, res_cand, res_base)
                b1_metrics[(cls_name, mkt_name, per_name)] = {
                    "m_base": m_base, "m_cand": m_cand, "res_base": res_base, "res_cand": res_cand,
                }

    # ------------------------------------------------------------ B3
    print()
    print("=" * 78)
    print("B3: plateau sweep -- positioning/VRP divisor at {0.5,1,2,4}x simultaneously, "
          "full construction, inner-validation")
    print("=" * 78)
    b3_base = {}
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        m_base, res_base = run_baseline(df_btc, mkt, INNER_VAL_START, INNER_VAL_END, label_btc)
        n_configs += 1
        b3_base[mkt_name] = (m_base, res_base)
    for mult in B3_MULTIPLIERS:
        pos_d = POSITIONING_DIVISOR * mult
        vrp_d = VRP_DIVISOR * mult
        for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
            m_base, res_base = b3_base[mkt_name]
            strat = BreadthOptimizedPanel(positioning_divisor=pos_d, vrp_divisor=vrp_d)
            m_cand, res_cand = run_strategy(strat, df_btc, mkt, INNER_VAL_START, INNER_VAL_END, label_btc)
            n_configs += 1
            print(f"  [{mult}x divisor={pos_d:.2f} {mkt_name}] trades={m_cand.num_trades} "
                  f"sharpe={m_cand.sharpe:.3f} | base sharpe={m_base.sharpe:.3f}")
            key = f"B3_{mult}x_{mkt_name}"
            results[key] = _report_diff(key, res_cand, res_base)

    # ------------------------------------------------------------ B4
    print()
    print("=" * 78)
    print("B4 (pre-registered falsification): full construction on ETH spot, inner-val")
    print("=" * 78)
    df_eth = load_eth_train()
    strat_eth = BreadthOptimizedPanel(asset="ETH")
    m_cand_eth, res_cand_eth = run_strategy(strat_eth, df_eth, SPOT, INNER_VAL_START, INNER_VAL_END, "eth_spot")
    n_configs += 1
    m_base_eth, res_base_eth = run_baseline(df_eth, SPOT, INNER_VAL_START, INNER_VAL_END, "eth_spot")
    n_configs += 1
    print(f"  [ETH spot / inner-val] cand: trades={m_cand_eth.num_trades} "
          f"final={m_cand_eth.final_balance:.1f} sharpe={m_cand_eth.sharpe:.3f} | "
          f"base: trades={m_base_eth.num_trades} final={m_base_eth.final_balance:.1f} "
          f"sharpe={m_base_eth.sharpe:.3f}")
    results["B4_eth_spot_inner_val"] = _report_diff("B4_eth_spot_inner_val", res_cand_eth, res_base_eth)
    b1_btc_sign = np.sign(results["B1_full_spot_inner_val"]["d_sharpe"])
    b4_sign = np.sign(results["B4_eth_spot_inner_val"]["d_sharpe"])
    b4_replicates = bool(b1_btc_sign == b4_sign)
    print(f"  BTC spot inner-val d_sharpe sign = {b1_btc_sign:+.0f}, "
          f"ETH spot inner-val d_sharpe sign = {b4_sign:+.0f} -> "
          f"{'REPLICATES' if b4_replicates else 'DOES NOT REPLICATE'}")

    # ------------------------------------------------------------ B5
    print()
    print("=" * 78)
    print("B5: 0.40% taker fee tier, full construction, BTC both markets, inner-val")
    print("=" * 78)
    b5_keys = []
    for mkt_name, mkt in (("spot_hi_fee", SPOT_HIGH_FEE), ("futures_5x_hi_fee", FUTURES_HIGH_FEE)):
        m_base, res_base = run_baseline(df_btc, mkt, INNER_VAL_START, INNER_VAL_END, label_btc)
        n_configs += 1
        strat = BreadthOptimizedPanel()
        m_cand, res_cand = run_strategy(strat, df_btc, mkt, INNER_VAL_START, INNER_VAL_END, label_btc)
        n_configs += 1
        print(f"  [{mkt_name} / inner-val] cand: trades={m_cand.num_trades} "
              f"final={m_cand.final_balance:.1f} sharpe={m_cand.sharpe:.3f} | "
              f"base: trades={m_base.num_trades} final={m_base.final_balance:.1f} sharpe={m_base.sharpe:.3f}")
        key = f"B5_{mkt_name}"
        results[key] = _report_diff(key, res_cand, res_base)
        b5_keys.append(key)

    # ------------------------------------------------------------ B6 (MANDATORY)
    print()
    print("=" * 78)
    print("B6 (MANDATORY): turnover-matched, information-free control vs full construction, "
          "decisive B1 cells (inner_train, full_inner), both markets")
    print("=" * 78)
    HYST_GRID = (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
    decisive_periods = {"inner_train": (INNER_TRAIN_START, INNER_TRAIN_END),
                         "full_inner": (INNER_TRAIN_START, INNER_VAL_END)}
    b6_results = {}
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        target_trades = b1_metrics[("full", mkt_name, "inner_val")]["m_cand"].num_trades
        for per_name, (start, end) in decisive_periods.items():
            m_ctrl, res_ctrl, best_h, best_trades = turnover_matched_control(
                df_btc, mkt, target_trades, start, end, label_btc, hyst_grid=HYST_GRID)
            n_configs += len(HYST_GRID) + 1  # grid search for matching h + 1 final backtest
            res_base = b1_metrics[("full", mkt_name, per_name)]["res_base"]
            d_sharpe_ctrl = sharpe_diff(res_ctrl, res_base).diff.point
            d_sharpe_cand = results[f"B1_full_{mkt_name}_{per_name}"]["d_sharpe"]
            beats = d_sharpe_cand > d_sharpe_ctrl
            print(f"  [{mkt_name}/{per_name}] target_trades(inner_val)={target_trades} "
                  f"control h={best_h:.2f} (inner_val trades={best_trades}) | "
                  f"candidate d_sharpe={d_sharpe_cand:+.4f}  control d_sharpe={d_sharpe_ctrl:+.4f} "
                  f"-> {'BEATS control' if beats else 'MATCHES-OR-TRAILS control'}")
            b6_results[(mkt_name, per_name)] = {
                "cand": d_sharpe_cand, "ctrl": d_sharpe_ctrl, "beats": beats, "best_h": best_h,
            }
    b6_gate = all(v["beats"] for v in b6_results.values())

    # ------------------------------------------------------------ SUMMARY
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for key, r in results.items():
        print(f"{key:36s} d_sharpe={r['d_sharpe']:+.4f} CI=[{r['sharpe_lo']:+.4f},{r['sharpe_hi']:+.4f}] "
              f"sig={r['sharpe_sig']} | d_logret={r['d_logret']:+.4f} sig={r['logret_sig']}")

    print()
    print("=" * 78)
    print("GATES (pre-registered decision rule, r135_shared.py)")
    print("=" * 78)

    # (a) causal-truncation probe
    gate_a = probe_ok
    print(f"  (a) causal-truncation probe:              {'PASS' if gate_a else 'FAIL'}")

    # (b) A2 non-inertness, both new experts
    gate_b = a2_gate
    print(f"  (b) A2 non-inertness (both new experts):  {'PASS' if gate_b else 'FAIL'}")

    # (c) B1: both markets, majority (>=2 of 2) of {inner_train, full_inner}, full construction
    b1_market_ok = {}
    for mkt_name in ("spot", "futures_5x"):
        n_pass = sum(1 for per in ("inner_train", "full_inner")
                     if results[f"B1_full_{mkt_name}_{per}"]["d_sharpe"] > 0)
        b1_market_ok[mkt_name] = n_pass >= 2
        print(f"      B1 {mkt_name}: {n_pass}/2 periods beat baseline "
              f"({'PASS' if b1_market_ok[mkt_name] else 'FAIL'})")
    gate_c = all(b1_market_ok.values())
    print(f"  (c) B1 (both markets, majority>=2/2):     {'PASS' if gate_c else 'FAIL'}")

    # (d) B3 plateau majority, per market, across the 4 multipliers
    b3_market_ok = {}
    for mkt_name in ("spot", "futures_5x"):
        n_pass = sum(1 for mult in B3_MULTIPLIERS if results[f"B3_{mult}x_{mkt_name}"]["d_sharpe"] > 0)
        b3_market_ok[mkt_name] = n_pass >= 3  # majority of 4
        print(f"      B3 {mkt_name}: {n_pass}/4 multipliers beat baseline "
              f"({'PASS' if b3_market_ok[mkt_name] else 'FAIL'})")
    gate_d = all(b3_market_ok.values())
    print(f"  (d) B3 (plateau majority, both markets):  {'PASS' if gate_d else 'FAIL'}")

    # (e) B4 sign replication
    gate_e = b4_replicates
    print(f"  (e) B4 (ETH sign replicates):             {'PASS' if gate_e else 'FAIL'}")

    # (f) B5 no sign flip at 0.40% fee tier vs same-market B1 inner_val sign
    b5_market_map = {"spot_hi_fee": "spot", "futures_5x_hi_fee": "futures_5x"}
    b5_ok = {}
    for hi_key, base_mkt in b5_market_map.items():
        s_hi = np.sign(results[f"B5_{hi_key}"]["d_sharpe"])
        s_lo = np.sign(results[f"B1_full_{base_mkt}_inner_val"]["d_sharpe"])
        b5_ok[hi_key] = bool(s_hi == s_lo)
        print(f"      B5 {hi_key}: sign={s_hi:+.0f} vs low-fee inner_val sign={s_lo:+.0f} "
              f"({'PASS' if b5_ok[hi_key] else 'FAIL'})")
    gate_f = all(b5_ok.values())
    print(f"  (f) B5 (no sign flip, 0.40% fee):         {'PASS' if gate_f else 'FAIL'}")

    # (g) B6 mandatory: candidate must EXCEED control on ALL decisive cells
    gate_g = b6_gate
    for (mkt_name, per_name), v in b6_results.items():
        print(f"      B6 {mkt_name}/{per_name}: cand={v['cand']:+.4f} ctrl={v['ctrl']:+.4f} "
              f"({'BEATS' if v['beats'] else 'MATCHES-OR-TRAILS'})")
    print(f"  (g) B6 (mandatory, candidate beats control on ALL decisive cells): "
          f"{'PASS' if gate_g else 'FAIL'}")

    overall = gate_a and gate_b and gate_c and gate_d and gate_e and gate_f and gate_g
    print()
    print(f"OVERALL VERDICT: {'PROMOTE-candidate' if overall else 'NEGATIVE'}")
    print()
    print(f"TOTAL BACKTEST CONFIGURATIONS EVALUATED: {n_configs}")
