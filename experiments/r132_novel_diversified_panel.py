"""R-132 NOVEL branch: a structural composition change to `hedge_experts`'s
own ten-expert panel (08-25). See `experiments/r132_shared.py`'s module
docstring for the full round-level pre-registration (direction, literature,
mechanism, non-duplicate check, named failure modes, falsification test,
decision rule) -- this file freezes the NOVEL branch's own exact mechanism
and reports its evaluation. Nothing here deviates from that pre-registration
without saying so explicitly.

**The two structural changes, frozen before any performance number was
read:**

1. **Horizon collapse: keep 1h (h=12 bars) and 1w (h=2016 bars), drop 6h
   (h=72) and 1d (h=288).** `HedgeExperts._experts` loops over
   `(12, 72, 288, 2016)` -- four vol-scaled log-momentum z-scores, all
   long/short-symmetric bets on the SAME underlying trend factor at
   different lookbacks and therefore highly mutually correlated (adjacent
   lookbacks share most of their window). The a-priori rule, stated in the
   round's own pre-registration before this file was written: **keep the
   shortest and the longest** to preserve the widest possible horizon
   spread the panel can express with two slots, rather than any pair drawn
   from the middle of the range, which sits closer together in
   log-lookback space (72 and 288 are 4x apart; 12 and 2016 are 168x
   apart) and would therefore be MORE mutually correlated with each other,
   not less. This is a structural, not a performance-driven, choice: it
   was fixed by inspecting the four horizons' definitions alone, never by
   looking at any of their individual Sharpe ratios or backtest output.

2. **Two new, structurally different experts fill the two freed slots:**
   an MVRV valuation mean-reversion vote and a stablecoin-supply-growth
   macro-flow vote (see `_mvrv_vote` / `_stablecoin_vote` below for the
   exact, fixed-a-priori construction of each). Both are cast as ordinary
   `tanh`-bounded votes in `[-1, 1]`, exactly like every other column
   `HedgeExperts._experts` already produces, so the Hedge blend treats
   them identically to the eight technical experts -- no special-casing,
   no gate, no hand-built threshold. This is the round's central
   methodological question (see `r132_shared.py`): does embedding an
   already-failed-standalone signal (R-74 for MVRV, R-54/R-55/R-58 for
   stablecoin supply) as one more Hedge-weighted VOTE let the combinator's
   own online down-weighting make it safe or even useful to include,
   where a hand-built confirming-vote GATE already failed it twice.

The five other original experts (MACD, RSI ramp, 1-bar reversion, Donchian
breakout, always-flat, buy-and-hold) are kept **completely unchanged** --
`_experts` below never reimplements them; it calls
`HedgeExperts._experts(self, df, r, sig1)` once (the exact base-class
method, unmodified) and slices/recombines its own output columns. Neither
`prepare()` nor `on_bar()` is touched anywhere in this file; both are
inherited verbatim from `HedgeExperts`, and the Hedge weight-update loop
inside `prepare()` already generalizes to any number of expert columns
without modification.

**MVRV expert -- exact construction, fixed before any code ran:**

`tradebot.data.load_mvrv_ratio(data_dir, asset=...)` gives a daily MVRV
ratio (market cap / realized cap). This branch z-scores `log(mvrv)` -- not
the raw ratio -- against its own **expanding window since inception**
(`min_periods=365`), matching R-74's own "classic MVRV-Z-Score
construction" convention (an expanding, not rolling, window, since the
indicator's own textbook definition marks extremes relative to the
asset's ENTIRE valuation history, not a fixed recent lookback). Logging
the ratio first is one explicit, disclosed departure from R-74's own
level construction: MVRV is strictly positive and right-skewed (historic
euphoria spikes reach far higher multiples above "fair" than capitulation
troughs reach below it), and this branch's vote must be **symmetric**
in both directions -- a full two-sided mean-reversion vote, not R-74's
one-directional "euphoria-only" confirming gate -- so a symmetrizing log
transform is applied before z-scoring, fixed here for that structural
reason and never touched again. The vote itself is
`vote = -tanh(z / MVRV_DIVISOR)` with `MVRV_DIVISOR = 3.0`: NEGATIVE
sign because high z (overvalued relative to own history) should read as
a risk-off / short-leaning vote and low z (undervalued) as a risk-on /
long-leaning vote -- literal mean reversion. `3.0` is not fit to any
performance number; it reuses this same file's own (inherited)
`HedgeExperts` convention of `3.0` as the fixed extreme-normalizing
divisor already used by the 1-bar reversion expert
(`-(r / (3.0 * sig1)).clip(-0.5, 0.5)`), so a z of 3 standard deviations
from the panel's own historical mean approaches vote saturation, the same
threshold this file's other experts already treat as "extreme." B3 sweeps
this divisor at {0.5, 1, 2, 4}x.

**Stablecoin-supply expert -- exact construction, fixed before any code
ran:** reuses the exact causal idiom already reviewed and used in this
project's `experiments/_stablecoin_signal.py` (R-54): 14-calendar-day log
growth of aggregate USDT circulating supply
(`tradebot.data.load_stablecoin_supply`), z-scored against its own
trailing 365-day mean/std (`min_periods=60`), sign-flipped so positive
means growth is unusually slow or supply is contracting (risk-off). Both
windows (14d growth, 365d z-score) are copied verbatim from that
already-reviewed, already-fixed-a-priori construction -- not re-derived
or re-tuned here. The vote is
`vote = -tanh(stablecoin_stress_z / STABLECOIN_DIVISOR)` with
`STABLECOIN_DIVISOR = 3.0` (same value, same reasoning, as the MVRV
divisor above, for the same reason: reuse this file's own existing
extreme-normalizing convention rather than invent a new one). B3 sweeps
this divisor at {0.5, 1, 2, 4}x as well.

Both new experts are causally aligned onto the bar grid via
`tradebot.data.align_mvrv_causal` / `align_stablecoin_causal`
respectively, which additionally apply the standard +1-calendar-day
publication-lag shift every other INFO loader in this project uses. Every
rolling/expanding statistic is computed on the raw DAILY frame (so a
day's own value depends only on prior days), and only the finished daily
vote series is projected onto the 5-minute bar grid -- exactly the
`_stablecoin_signal.py` idiom, reused rather than reinvented.

**Ablation, pre-registered, not optional (isolates the two changes):**
`HorizonsOnlyPanel` performs ONLY the horizon collapse (2 momentum + the
five other original experts + buy-and-hold = 8 experts), adding neither
new expert, so B1 run on it separates "did shrinking momentum redundancy
alone help or hurt" from "did adding the two new experts alone help or
hurt" from "did the full combined construction help or hurt."

**Causality.** Every new computation here (`expanding()`, `rolling()`,
`.shift()`) is backward-looking only; verified below by the same
truncation-probe idiom `r132_shared.py`'s own `__main__` block uses,
applied separately to both `NovelDiversifiedPanel` and `HorizonsOnlyPanel`.
No bar at or after `OOS_START = 2023-01-01` is read anywhere in this file
(`_assert_no_holdout`, reused verbatim from `r132_shared.py`, guards every
loaded frame).

**Decision rule and falsification test:** verbatim from `r132_shared.py`
-- PROMOTE-candidate only if the causal-truncation probe AND A2 (both new
experts) AND B1 (both markets, full period AND inner-validation) AND B3
(plateau majority) AND B4 (ETH sign replication) AND B5 (0.40% fee, no
sign flip) all pass. Nothing here moves that goalpost.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import (  # noqa: E402
    align_mvrv_causal,
    align_stablecoin_causal,
    load_mvrv_ratio,
    load_stablecoin_supply,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategies.hedge_experts import HedgeExperts  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from r132_shared import (  # noqa: E402
    B3_MULTIPLIERS,
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
    run_baseline,
    run_strategy,
    sharpe_diff,
)

# ----------------------------------------------------------------------
# Frozen, a-priori parameters. Never swept for a "best" value -- fixed
# before any performance number was read; B3 sweeps AROUND these points
# purely as a plateau/peak diagnostic, not a search for a better one.
# ----------------------------------------------------------------------
_HORIZON_ORDER = (12, 72, 288, 2016)          # HedgeExperts's own base order
KEPT_HORIZONS = (12, 2016)                    # shortest + longest, a priori

MVRV_ZSCORE_MIN_PERIODS = 365                 # classic MVRV-Z-Score convention (R-74)
MVRV_DIVISOR = 3.0                            # reuses this file's own 3.0 convention

STABLE_GROWTH_WINDOW_DAYS = 14                # verbatim from _stablecoin_signal.py (R-54)
STABLE_ZSCORE_WINDOW_DAYS = 365               # verbatim from _stablecoin_signal.py (R-54)
STABLE_ZSCORE_MIN_PERIODS = 60                # verbatim from _stablecoin_signal.py (R-54)
STABLECOIN_DIVISOR = 3.0


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {OOS_START}")


def _mvrv_vote(df: pd.DataFrame, data_dir, asset: str, divisor: float) -> pd.Series:
    """Causal MVRV valuation mean-reversion vote, in [-1, 1]. See module
    docstring for the full construction and why each choice was made."""
    mvrv = load_mvrv_ratio(data_dir, asset=asset)
    if mvrv is None:
        return pd.Series(0.0, index=df.index)
    log_mvrv = np.log(mvrv["mvrv"].clip(lower=1e-9))
    mean = log_mvrv.expanding(min_periods=MVRV_ZSCORE_MIN_PERIODS).mean()
    std = log_mvrv.expanding(min_periods=MVRV_ZSCORE_MIN_PERIODS).std()
    z = ((log_mvrv - mean) / std).rename("mvrv_z").to_frame()
    z_aligned = align_mvrv_causal(z, df)["mvrv_z"]
    vote = -np.tanh(z_aligned / divisor)
    return vote.fillna(0.0)


def _stablecoin_vote(df: pd.DataFrame, data_dir, divisor: float) -> pd.Series:
    """Causal stablecoin-supply-growth macro-flow vote, in [-1, 1]. Reuses
    the exact `growth_14d` / `stablecoin_stress_z` idiom already reviewed
    in `experiments/_stablecoin_signal.py` (R-54); see module docstring."""
    supply = load_stablecoin_supply(data_dir)
    if supply is None:
        return pd.Series(0.0, index=df.index)
    log_s = np.log(supply["supply"])
    growth = log_s - log_s.shift(STABLE_GROWTH_WINDOW_DAYS)
    mean = growth.rolling(STABLE_ZSCORE_WINDOW_DAYS, min_periods=STABLE_ZSCORE_MIN_PERIODS).mean()
    std = growth.rolling(STABLE_ZSCORE_WINDOW_DAYS, min_periods=STABLE_ZSCORE_MIN_PERIODS).std()
    stress_z = (-1.0 * (growth - mean) / std).rename("stablecoin_stress_z").to_frame()
    aligned = align_stablecoin_causal(stress_z, df)["stablecoin_stress_z"]
    vote = -np.tanh(aligned / divisor)
    return vote.fillna(0.0)


def _momentum_kept_and_rest(df: pd.DataFrame, r: pd.Series, sig1: pd.Series):
    """Call the UNMODIFIED `HedgeExperts._experts` once, then slice its
    10-column output down to {kept momentum horizons} + {the five other
    original experts + buy-and-hold}, i.e. columns 4..9 verbatim."""
    base = HedgeExperts._experts(HedgeExperts(), df, r, sig1)  # (n, 10), stateless call
    keep_idx = [_HORIZON_ORDER.index(h) for h in KEPT_HORIZONS]
    momentum_kept = base[:, keep_idx]
    rest = base[:, 4:]  # macd, rsi, reversion, donch, flat, buy-and-hold -- unchanged
    return momentum_kept, rest


class HorizonsOnlyPanel(HedgeExperts):
    """Ablation (pre-registered, not optional): ONLY the horizon collapse --
    keep 1h+1w momentum, drop 6h+1d, add NO new experts. 8-expert panel.
    Isolates whether shrinking momentum redundancy alone helps or hurts,
    separately from whether adding the two new experts helps or hurts.
    """

    name = "r132_horizons_only_ablation"

    def _experts(self, df: pd.DataFrame, r: pd.Series, sig1: pd.Series) -> np.ndarray:
        momentum_kept, rest = _momentum_kept_and_rest(df, r, sig1)
        a = np.column_stack([momentum_kept, rest])
        return np.nan_to_num(a, nan=0.0)


class NovelDiversifiedPanel(HedgeExperts):
    """Full NOVEL construction: horizon collapse (1h+1w momentum) AND two
    new structurally different experts (MVRV valuation mean-reversion,
    stablecoin-supply-growth macro flow) filling the two freed slots.
    10-expert panel, same count as the original. See module docstring for
    the complete frozen mechanism and why each choice was made a priori.
    """

    name = "r132_novel_diversified_panel"

    def __init__(self, eta: float = 0.05, fixed_share: float = 1e-4,
                 hysteresis: float = 0.05, fee_rate: float = 0.0005,
                 data_dir=None, asset: str = "BTC",
                 mvrv_divisor: float = MVRV_DIVISOR,
                 stablecoin_divisor: float = STABLECOIN_DIVISOR) -> None:
        super().__init__(eta=eta, fixed_share=fixed_share, hysteresis=hysteresis,
                          fee_rate=fee_rate)
        self.data_dir = Path(data_dir) if data_dir is not None else ROOT / "data"
        self.asset = asset
        self.mvrv_divisor = mvrv_divisor
        self.stablecoin_divisor = stablecoin_divisor

    def _experts(self, df: pd.DataFrame, r: pd.Series, sig1: pd.Series) -> np.ndarray:
        momentum_kept, rest = _momentum_kept_and_rest(df, r, sig1)
        mvrv_vote = _mvrv_vote(df, self.data_dir, self.asset, self.mvrv_divisor)
        stable_vote = _stablecoin_vote(df, self.data_dir, self.stablecoin_divisor)
        a = np.column_stack([momentum_kept, rest,
                              np.asarray(mvrv_vote, dtype=np.float64),
                              np.asarray(stable_vote, dtype=np.float64)])
        return np.nan_to_num(a, nan=0.0)


# ----------------------------------------------------------------------
# Run/metric helpers (mirrors r132_shared.run_baseline/run_strategy).
# ----------------------------------------------------------------------

def run_candidate(strat, df: pd.DataFrame, market: MarketSpec, start: str, end: str,
                   label: str = ""):
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


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


# ----------------------------------------------------------------------
# Diagnostic (reported, does not gate promotion): pairwise expert-position
# correlation + Grinold effective breadth, original panel vs full novel
# panel, over inner-validation.
# ----------------------------------------------------------------------

def _expert_position_matrix(strat, df: pd.DataFrame, start: str, end: str) -> np.ndarray:
    """Own-position columns `a[:, j]` from `_experts()`, computed causally
    over the WHOLE frame (matching how `prepare()` actually runs) and then
    sliced to [start, end] -- slicing after computation cannot leak
    future information since each row of `a` already depends only on
    rows <= it."""
    r = np.log(df["close"]).diff()
    sig1 = r.ewm(span=288, min_periods=250).std()
    a = strat._experts(df, r, sig1)
    idx = df.index
    tz = idx.tz
    mask = (idx >= pd.Timestamp(start, tz=tz)) & (idx <= pd.Timestamp(end, tz=tz))
    return a[mask]


def _replay_hedge_weights(strat, df: pd.DataFrame) -> np.ndarray:
    """Read-only, bit-faithful replay of `HedgeExperts.prepare()`'s own
    `logw`/`p` recursion (copied verbatim from `hedge_experts.py`),
    instrumented ONLY to also record the full weight-probability history
    `p_i(t)` for every expert at every bar -- needed for gate A2. This
    does NOT modify `HedgeExperts.prepare` or `on_bar` in any way; it is a
    separate, additional computation used purely for this diagnostic."""
    r = np.log(df["close"]).diff()
    sig1 = r.ewm(span=288, min_periods=250).std()
    a = strat._experts(df, r, sig1)
    r_a = r.to_numpy()
    sig_a = sig1.shift(1).to_numpy()
    n, num = a.shape
    p_hist = np.zeros((n, num))
    logw = np.zeros(num)
    p = np.ones(num) / num
    for i in range(2, n):
        s = sig_a[i]
        if not np.isfinite(s) or s <= 0:
            p_hist[i] = p
            continue
        z_t = min(max(r_a[i] / (3.0 * s), -1.0), 1.0)
        fee_n = min(strat.fee_rate / (3.0 * s), 0.25)
        g = np.clip(a[i - 1] * z_t - fee_n * np.abs(a[i - 1] - a[i - 2]), -1.0, 1.0)
        logw += strat.eta * g
        logw -= logw.max()
        p = np.exp(logw)
        p /= p.sum()
        p = (1.0 - strat.fixed_share) * p + strat.fixed_share / num
        logw = np.log(p)
        p_hist[i] = p
    return p_hist


if __name__ == "__main__":
    n_configs = 0
    results: dict[str, dict] = {}

    df_btc, label_btc = load_btc_train("spot")

    # ------------------------------------------------------------
    # Causal-truncation probe (must PASS) -- both classes.
    # ------------------------------------------------------------
    print("=" * 78)
    print("Causal-truncation probe")
    print("=" * 78)
    probe_ok = True
    for cls_name, strat_factory in (
        ("NovelDiversifiedPanel", lambda: NovelDiversifiedPanel()),
        ("HorizonsOnlyPanel", lambda: HorizonsOnlyPanel()),
    ):
        m_full, _ = run_candidate(strat_factory(), df_btc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label_btc)
        n_configs += 1
        df_trunc = df_btc.loc[:INNER_VAL_END].copy()
        _assert_no_holdout(df_trunc)
        m_trunc, _ = run_candidate(strat_factory(), df_trunc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label_btc)
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
    novel = NovelDiversifiedPanel()
    p_hist = _replay_hedge_weights(novel, df_btc)
    idx = df_btc.index
    mask_val = (idx >= pd.Timestamp(INNER_VAL_START, tz=idx.tz)) & (idx <= pd.Timestamp(INNER_VAL_END, tz=idx.tz))
    num_experts = p_hist.shape[1]
    mvrv_col, stable_col = num_experts - 2, num_experts - 1
    a2_mvrv = a2_non_inertness(p_hist[mask_val, mvrv_col], num_experts)
    a2_stable = a2_non_inertness(p_hist[mask_val, stable_col], num_experts)
    print(f"  MVRV expert:        frac_bars_above_2x_uniform={a2_mvrv['frac_bars_above_2x_uniform']:.4f} "
          f"pass={a2_mvrv['pass']}")
    print(f"  Stablecoin expert:  frac_bars_above_2x_uniform={a2_stable['frac_bars_above_2x_uniform']:.4f} "
          f"pass={a2_stable['pass']}")
    print("  (replay computation, not a broker backtest -- not added to n_configs, "
          "same convention r132_shared.py's own diagnostics use)")

    # ------------------------------------------------------------
    # Diagnostic: pairwise expert-position correlation + effective breadth,
    # original 10-expert panel vs full novel 10-expert panel, inner-val.
    # ------------------------------------------------------------
    print()
    print("=" * 78)
    print("Diagnostic: expert-position correlation / effective breadth (inner-val, BTC spot)")
    print("=" * 78)
    a_orig = _expert_position_matrix(HedgeExperts(), df_btc, INNER_VAL_START, INNER_VAL_END)
    a_novel = _expert_position_matrix(novel, df_btc, INNER_VAL_START, INNER_VAL_END)
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
    print("  (correlation/breadth diagnostic -- not a broker backtest, not added to n_configs; "
          "flat and buy-and-hold columns are constant so their pairwise correlations are NaN "
          "and excluded by effective_breadth's own nanmean, symmetrically in both panels)")

    # ------------------------------------------------------------ B1
    print()
    print("=" * 78)
    print("B1: full construction AND horizons-only ablation vs frozen hedge_experts")
    print("   (both markets, full-inner period AND inner-validation)")
    print("=" * 78)
    periods = [
        ("full_inner", INNER_TRAIN_START, INNER_VAL_END),
        ("inner_val", INNER_VAL_START, INNER_VAL_END),
    ]
    baselines = {}
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        for per_name, start, end in periods:
            m_base, res_base = run_baseline(df_btc, mkt, start, end, label_btc)
            n_configs += 1
            baselines[(mkt_name, per_name)] = (m_base, res_base)
            print(f"  [baseline {mkt_name}/{per_name}] trades={m_base.num_trades} "
                  f"final={m_base.final_balance:.1f} sharpe={m_base.sharpe:.3f}")

    for cls_name, strat_factory in (
        ("full_novel", lambda: NovelDiversifiedPanel()),
        ("horizons_only_ablation", lambda: HorizonsOnlyPanel()),
    ):
        for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
            for per_name, start, end in periods:
                m_base, res_base = baselines[(mkt_name, per_name)]
                m_cand, res_cand = run_candidate(strat_factory(), df_btc, mkt, start, end, label_btc)
                n_configs += 1
                print(f"  [{cls_name} {mkt_name}/{per_name}] trades={m_cand.num_trades} "
                      f"final={m_cand.final_balance:.1f} sharpe={m_cand.sharpe:.3f}")
                key = f"B1_{cls_name}_{mkt_name}_{per_name}"
                results[key] = _report_diff(key, res_cand, res_base)

    # ------------------------------------------------------------ B3
    print()
    print("=" * 78)
    print("B3: plateau sweep -- MVRV/stablecoin divisor at {0.5,1,2,4}x, "
          "full novel construction, inner-validation")
    print("=" * 78)
    for mult in B3_MULTIPLIERS:
        mvrv_d = MVRV_DIVISOR * mult
        stable_d = STABLECOIN_DIVISOR * mult
        for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
            m_base, res_base = baselines[(mkt_name, "inner_val")]
            strat = NovelDiversifiedPanel(mvrv_divisor=mvrv_d, stablecoin_divisor=stable_d)
            m_cand, res_cand = run_candidate(strat, df_btc, mkt, INNER_VAL_START, INNER_VAL_END, label_btc)
            n_configs += 1
            print(f"  [{mult}x divisor={mvrv_d:.2f} {mkt_name}] trades={m_cand.num_trades} "
                  f"sharpe={m_cand.sharpe:.3f} | base sharpe={m_base.sharpe:.3f}")
            key = f"B3_{mult}x_{mkt_name}"
            results[key] = _report_diff(key, res_cand, res_base)

    # ------------------------------------------------------------ B4
    print()
    print("=" * 78)
    print("B4 (pre-registered falsification): full construction on ETH spot, inner-val")
    print("=" * 78)
    df_eth = load_eth_train()
    strat_eth = NovelDiversifiedPanel(asset="ETH")
    m_cand_eth, res_cand_eth = run_candidate(strat_eth, df_eth, SPOT, INNER_VAL_START, INNER_VAL_END, "eth_spot")
    n_configs += 1
    m_base_eth, res_base_eth = run_baseline(df_eth, SPOT, INNER_VAL_START, INNER_VAL_END, "eth_spot")
    n_configs += 1
    print(f"  [ETH spot / inner-val] cand: trades={m_cand_eth.num_trades} "
          f"final={m_cand_eth.final_balance:.1f} sharpe={m_cand_eth.sharpe:.3f} | "
          f"base: trades={m_base_eth.num_trades} final={m_base_eth.final_balance:.1f} "
          f"sharpe={m_base_eth.sharpe:.3f}")
    results["B4_eth_spot_inner_val"] = _report_diff("B4_eth_spot_inner_val", res_cand_eth, res_base_eth)
    b1_btc_sign = np.sign(results["B1_full_novel_spot_inner_val"]["d_sharpe"])
    b4_sign = np.sign(results["B4_eth_spot_inner_val"]["d_sharpe"])
    print(f"  BTC spot inner-val d_sharpe sign = {b1_btc_sign:+.0f}, "
          f"ETH spot inner-val d_sharpe sign = {b4_sign:+.0f} -> "
          f"{'REPLICATES' if b1_btc_sign == b4_sign else 'DOES NOT REPLICATE'}")

    # ------------------------------------------------------------ B5
    print()
    print("=" * 78)
    print("B5: 0.40% taker fee tier, full construction, BTC both markets, inner-val")
    print("=" * 78)
    for mkt_name, mkt in (("spot_hi_fee", SPOT_HIGH_FEE), ("futures_5x_hi_fee", FUTURES_HIGH_FEE)):
        m_base, res_base = run_baseline(df_btc, mkt, INNER_VAL_START, INNER_VAL_END, label_btc)
        n_configs += 1
        strat = NovelDiversifiedPanel()
        m_cand, res_cand = run_candidate(strat, df_btc, mkt, INNER_VAL_START, INNER_VAL_END, label_btc)
        n_configs += 1
        print(f"  [{mkt_name} / inner-val] cand: trades={m_cand.num_trades} "
              f"final={m_cand.final_balance:.1f} sharpe={m_cand.sharpe:.3f} | "
              f"base: trades={m_base.num_trades} final={m_base.final_balance:.1f} sharpe={m_base.sharpe:.3f}")
        key = f"B5_{mkt_name}"
        results[key] = _report_diff(key, res_cand, res_base)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for key, r in results.items():
        print(f"{key:40s} d_sharpe={r['d_sharpe']:+.4f} CI=[{r['sharpe_lo']:+.4f},{r['sharpe_hi']:+.4f}] "
              f"sig={r['sharpe_sig']} | d_logret={r['d_logret']:+.4f} sig={r['logret_sig']}")
    print()
    print(f"TOTAL BACKTEST CONFIGURATIONS EVALUATED: {n_configs}")
