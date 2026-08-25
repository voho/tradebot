"""R-135 CONSERVATIVE branch: append exactly ONE new expert -- a derivatives-
positioning ("crowding") contrarian vote -- to `hedge_experts`'s existing
ten-expert panel (08-25). See `experiments/r135_shared.py`'s module docstring
for the full round-level pre-registration (direction, literature, both
branches' mechanisms, non-duplicate check, named failure modes,
falsification test, decision rule) -- this file freezes THIS branch's own
exact mechanism and reports its evaluation. Nothing here deviates from that
pre-registration without saying so explicitly.

**The one structural change, frozen before any performance number was
read:** append ONE new expert column to the unmodified ten-column output of
`HedgeExperts._experts` (called once, statelessly, via
`HedgeExperts._experts(HedgeExperts(), df, r, sig1)`, exactly as R-132's own
conservative/novel branches did), producing an 11-expert panel. Every other
line of `HedgeExperts` -- `eta`/`fixed_share`/`hysteresis`, the ten original
experts, the Hedge weight-update loop inside `prepare()` -- is held
bit-for-bit fixed. Neither `prepare()` nor `on_bar()` is touched anywhere in
this file; both are inherited verbatim from `HedgeExperts`.

**Positioning ("crowding") expert -- exact construction, fixed before any
code ran:**

`tradebot.data.load_binance_metrics(data_dir, asset=...)` gives 5-minute-
native-cadence Binance USDⓈ-M futures positioning metrics. This branch uses
ONLY `count_long_short_ratio` (the broadest, all-account positioning ratio),
deliberately NOT `count_toptrader_long_short_ratio` /
`sum_toptrader_long_short_ratio` -- R-81 already found those genuinely
missing (empty in the raw CSV) for 37.6% of the committed BTC window,
concentrated in 2022 and entirely covering the FTX collapse (see
`load_binance_metrics`'s own docstring). `count_long_short_ratio` and
`sum_taker_long_short_vol_ratio` are NOT affected by that gap;
`count_long_short_ratio` is preferred over the taker-volume ratio
specifically to keep this a pure positioning-STOCK signal, structurally
distinct from R-88's already-closed order-FLOW construction.

1. `log_ratio = log(count_long_short_ratio)` -- the ratio is strictly
   positive and right-skewed, the same reasoning R-132 used before
   z-scoring MVRV.
2. Rolling (NOT expanding -- positioning is a flow/regime quantity, not an
   inception-anchored valuation level) z-score of `log_ratio` against its
   own trailing window, computed directly on the raw 5-minute-cadence
   metrics series: a 90-calendar-day window in BAR units
   (`POSITIONING_ZSCORE_WINDOW_BARS = 90 * 288 = 25920`),
   `min_periods = 30 calendar days` in bars
   (`POSITIONING_ZSCORE_MIN_PERIODS_BARS = 30 * 288 = 8640`). The rolling
   mean/std is computed BEFORE alignment, on the raw metrics frame (so the
   statistic only ever depends on the metrics series' own past, never on
   the bars grid), and only the finished z-score is aligned onto the bar
   grid via `tradebot.data.align_metrics_causal` (which forward-fills
   causally at the bars' own 5-minute cadence -- no publication-lag shift,
   unlike the daily INFO feeds, because this feed already shares the bars'
   own cadence; see that function's own docstring).
3. `vote = -tanh(z / POSITIONING_DIVISOR)` with `POSITIONING_DIVISOR = 3.0`
   (same fixed-a-priori convention as R-132's `MVRV_DIVISOR` /
   `STABLECOIN_DIVISOR`: reuses `HedgeExperts`'s own existing
   extreme-normalizing constant, the 1-bar reversion expert's
   `3.0 * sig1`, rather than inventing a new one). NEGATIVE sign, fixed
   from De Roon/Nijman/Veld (2000) hedging-pressure theory, not fit to any
   backtest number: high z (the crowd is unusually net-long relative to
   its own trailing 90-day history) reads as a risk-off / short-leaning
   contrarian vote; low z (unusually net-short) reads as risk-on /
   long-leaning. B3 sweeps this divisor at {0.5, 1, 2, 4}x.
4. `.fillna(0.0)` for any NaN -- before Binance-metrics coverage starts
   (BTC 2020-09-01, ETH 2021-12-01) or through any residual gap -- matching
   `HedgeExperts._experts`'s own `nan_to_num(a, nan=0.0)` convention
   ("NaN warmup rows act as flat experts").

**Causality.** The new computation (`rolling()`, no `.shift()` needed since
`align_metrics_causal` itself only ever looks at metrics rows at-or-before a
bar's own timestamp) is backward-looking only; verified below by the same
truncation-probe idiom `r135_shared.py`'s own `__main__` block uses. No bar
at or after `OOS_START = 2023-01-01` is read anywhere in this file
(`_assert_no_holdout`, mirrors `r135_shared.py`'s own guard).

**Decision rule and falsification test:** verbatim from `r135_shared.py`'s
module docstring (reproduced there in full) -- PROMOTE-candidate only if the
causal-truncation probe, A2, B1 (both markets, majority of
{inner_train, full_inner}), B3 (plateau majority), B4 (ETH sign
replication), B5 (0.40% fee, no sign flip), AND B6 (MANDATORY
turnover-matched-control beat) all pass. Nothing here moves that goalpost.
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
from tradebot.data import align_metrics_causal, load_binance_metrics  # noqa: E402
from tradebot.strategies.hedge_experts import HedgeExperts  # noqa: E402

from r135_shared import (  # noqa: E402
    B1_PERIODS,
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
# before any performance number was read; B3 sweeps AROUND this point
# purely as a plateau/peak diagnostic, not a search for a better one.
# ----------------------------------------------------------------------
POSITIONING_ZSCORE_WINDOW_BARS = 90 * 288    # 90 calendar days, in 5-min bars
POSITIONING_ZSCORE_MIN_PERIODS_BARS = 30 * 288  # 30 calendar days, in 5-min bars
POSITIONING_DIVISOR = 3.0                    # reuses HedgeExperts's own 3.0 convention


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {OOS_START}")


def _positioning_vote(df: pd.DataFrame, data_dir, asset: str, divisor: float) -> pd.Series:
    """Causal derivatives-positioning ("crowding") contrarian vote, in
    [-1, 1]. See module docstring for the full construction and why each
    choice was made."""
    metrics = load_binance_metrics(data_dir, asset=asset)
    if metrics is None:
        return pd.Series(0.0, index=df.index)
    log_ratio = np.log(metrics["count_long_short_ratio"].clip(lower=1e-9))
    mean = log_ratio.rolling(POSITIONING_ZSCORE_WINDOW_BARS,
                              min_periods=POSITIONING_ZSCORE_MIN_PERIODS_BARS).mean()
    std = log_ratio.rolling(POSITIONING_ZSCORE_WINDOW_BARS,
                             min_periods=POSITIONING_ZSCORE_MIN_PERIODS_BARS).std()
    z = ((log_ratio - mean) / std).rename("positioning_z").to_frame()
    z_aligned = align_metrics_causal(z, df)["positioning_z"]
    vote = -np.tanh(z_aligned / divisor)
    return vote.fillna(0.0)


class ConservativePositioningPanel(HedgeExperts):
    """Full CONSERVATIVE construction: the unmodified 10-expert
    `HedgeExperts` panel plus ONE new derivatives-positioning ("crowding")
    contrarian vote. 11-expert panel. See module docstring for the complete
    frozen mechanism and why each choice was made a priori.
    """

    name = "r135_conservative_positioning_expert"

    def __init__(self, eta: float = 0.05, fixed_share: float = 1e-4,
                 hysteresis: float = 0.05, fee_rate: float = 0.0005,
                 data_dir=None, asset: str = "BTC",
                 positioning_divisor: float = POSITIONING_DIVISOR) -> None:
        super().__init__(eta=eta, fixed_share=fixed_share, hysteresis=hysteresis,
                          fee_rate=fee_rate)
        self.data_dir = Path(data_dir) if data_dir is not None else ROOT / "data"
        self.asset = asset
        self.positioning_divisor = positioning_divisor

    def _experts(self, df: pd.DataFrame, r: pd.Series, sig1: pd.Series) -> np.ndarray:
        base = HedgeExperts._experts(HedgeExperts(), df, r, sig1)  # (n, 10), stateless call
        pos_vote = _positioning_vote(df, self.data_dir, self.asset, self.positioning_divisor)
        a = np.column_stack([base, np.asarray(pos_vote, dtype=np.float64)])
        return np.nan_to_num(a, nan=0.0)


# ----------------------------------------------------------------------
# Report helper (mirrors r132_novel_diversified_panel.py's `_report_diff`).
# ----------------------------------------------------------------------

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
    gates: dict[str, bool] = {}

    df_btc, label_btc = load_btc_train("spot")

    # ------------------------------------------------------------
    # (a) Causal-truncation probe (MUST PASS).
    # ------------------------------------------------------------
    print("=" * 78)
    print("(a) Causal-truncation probe")
    print("=" * 78)
    cand = ConservativePositioningPanel()
    m_full, _ = run_strategy(cand, df_btc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label_btc)
    n_configs += 1
    df_trunc = df_btc.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(df_trunc)
    m_trunc, _ = run_strategy(ConservativePositioningPanel(), df_trunc, SPOT,
                               INNER_TRAIN_START, INNER_TRAIN_END, label_btc)
    n_configs += 1
    probe_ok = bool(np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9))
    gates["(a) causal_truncation_probe"] = probe_ok
    print(f"  [ConservativePositioningPanel] {'PASS' if probe_ok else 'FAIL'} "
          f"({m_full.final_balance} vs {m_trunc.final_balance})")
    assert probe_ok, "ConservativePositioningPanel reads ahead of its own truncation point"

    # ------------------------------------------------------------
    # (b) A2 (non-inertness): the new expert's own Hedge weight p_i,
    # inner-validation, BTC spot.
    # ------------------------------------------------------------
    print()
    print("=" * 78)
    print("(b) A2: non-inertness of the new positioning expert (BTC spot, inner-val)")
    print("=" * 78)
    p_hist = replay_hedge_weights(cand, df_btc)
    idx = df_btc.index
    mask_val = (idx >= pd.Timestamp(INNER_VAL_START, tz=idx.tz)) & (idx <= pd.Timestamp(INNER_VAL_END, tz=idx.tz))
    num_experts = p_hist.shape[1]
    pos_col = num_experts - 1  # appended last
    a2_pos = a2_non_inertness(p_hist[mask_val, pos_col], num_experts)
    gates["(b) A2_non_inertness"] = bool(a2_pos["pass"])
    print(f"  Positioning expert: frac_bars_above_2x_uniform={a2_pos['frac_bars_above_2x_uniform']:.4f} "
          f"pass={a2_pos['pass']}  (num_experts={num_experts})")
    print("  (replay computation, not a broker backtest -- not added to n_configs, "
          "same convention r135_shared.py's own diagnostics use)")

    # ------------------------------------------------------------ B1
    print()
    print("=" * 78)
    print("B1: full construction vs frozen hedge_experts")
    print("   (both markets, all THREE B1_PERIODS: inner_train, full_inner, inner_val)")
    print("=" * 78)
    baselines = {}
    b1_dsharpe = {}       # (mkt_name, per_name) -> d_sharpe point
    cand_trades_inner_val = {}  # mkt_name -> candidate's own inner_val trade count (for B6)
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        for per_name, start, end in B1_PERIODS:
            m_base, res_base = run_baseline(df_btc, mkt, start, end, label_btc)
            n_configs += 1
            baselines[(mkt_name, per_name)] = (m_base, res_base)
            m_cand, res_cand = run_strategy(ConservativePositioningPanel(), df_btc, mkt,
                                             start, end, label_btc)
            n_configs += 1
            print(f"  [{mkt_name}/{per_name}] base: trades={m_base.num_trades} "
                  f"final={m_base.final_balance:.1f} sharpe={m_base.sharpe:.3f} | "
                  f"cand: trades={m_cand.num_trades} final={m_cand.final_balance:.1f} "
                  f"sharpe={m_cand.sharpe:.3f}")
            key = f"B1_{mkt_name}_{per_name}"
            r = _report_diff(key, res_cand, res_base)
            results[key] = r
            b1_dsharpe[(mkt_name, per_name)] = r["d_sharpe"]
            if per_name == "inner_val":
                cand_trades_inner_val[mkt_name] = m_cand.num_trades

    # Decision rule (c): both markets, majority (>=2 of 2) of
    # {inner_train, full_inner} beat baseline by d_sharpe's point estimate.
    # With exactly 2 cells per market, majority(>=2 of 2) means BOTH must be
    # positive, and this must hold on BOTH markets.
    print()
    b1_pass = True
    for mkt_name in ("spot", "futures_5x"):
        cells = [b1_dsharpe[(mkt_name, "inner_train")], b1_dsharpe[(mkt_name, "full_inner")]]
        n_pos = sum(1 for c in cells if c > 0)
        mkt_pass = n_pos >= 2
        b1_pass = b1_pass and mkt_pass
        print(f"  [{mkt_name}] decisive-cell d_sharpe: inner_train={cells[0]:+.4f} "
              f"full_inner={cells[1]:+.4f} -> {n_pos}/2 positive -> "
              f"{'PASS' if mkt_pass else 'FAIL'}")
    gates["(c) B1_majority"] = b1_pass
    print(f"  (c) B1 gate overall: {'PASS' if b1_pass else 'FAIL'} "
          f"(inner_val reported above but does NOT count toward this majority)")

    # ------------------------------------------------------------ B3
    print()
    print("=" * 78)
    print("B3: plateau sweep -- positioning divisor at {0.5,1,2,4}x, "
          "full construction, inner-validation")
    print("=" * 78)
    b3_signs = []
    for mult in B3_MULTIPLIERS:
        div = POSITIONING_DIVISOR * mult
        for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
            m_base, res_base = baselines[(mkt_name, "inner_val")]
            strat = ConservativePositioningPanel(positioning_divisor=div)
            m_cand, res_cand = run_strategy(strat, df_btc, mkt, INNER_VAL_START, INNER_VAL_END, label_btc)
            n_configs += 1
            print(f"  [{mult}x divisor={div:.2f} {mkt_name}] trades={m_cand.num_trades} "
                  f"sharpe={m_cand.sharpe:.3f} | base sharpe={m_base.sharpe:.3f}")
            key = f"B3_{mult}x_{mkt_name}"
            r = _report_diff(key, res_cand, res_base)
            results[key] = r
            b3_signs.append(np.sign(r["d_sharpe"]))

    b1_spot_inner_val_sign = np.sign(b1_dsharpe[("spot", "inner_val")])
    n_match = sum(1 for s in b3_signs if s == b1_spot_inner_val_sign)
    b3_pass = n_match >= (len(b3_signs) / 2.0)
    gates["(d) B3_plateau_majority"] = bool(b3_pass)
    print(f"  (d) B3 gate: {n_match}/{len(b3_signs)} cells share the base construction's "
          f"BTC-spot-inner_val sign ({b1_spot_inner_val_sign:+.0f}) -> "
          f"{'PASS' if b3_pass else 'FAIL'}")

    # ------------------------------------------------------------ B4
    print()
    print("=" * 78)
    print("B4 (pre-registered falsification): full construction on ETH spot, inner-val")
    print("=" * 78)
    df_eth = load_eth_train()
    strat_eth = ConservativePositioningPanel(asset="ETH")
    m_cand_eth, res_cand_eth = run_strategy(strat_eth, df_eth, SPOT, INNER_VAL_START, INNER_VAL_END, "eth_spot")
    n_configs += 1
    m_base_eth, res_base_eth = run_baseline(df_eth, SPOT, INNER_VAL_START, INNER_VAL_END, "eth_spot")
    n_configs += 1
    print(f"  [ETH spot / inner-val] cand: trades={m_cand_eth.num_trades} "
          f"final={m_cand_eth.final_balance:.1f} sharpe={m_cand_eth.sharpe:.3f} | "
          f"base: trades={m_base_eth.num_trades} final={m_base_eth.final_balance:.1f} "
          f"sharpe={m_base_eth.sharpe:.3f}")
    results["B4_eth_spot_inner_val"] = _report_diff("B4_eth_spot_inner_val", res_cand_eth, res_base_eth)
    b4_sign = np.sign(results["B4_eth_spot_inner_val"]["d_sharpe"])
    b4_pass = bool(b1_spot_inner_val_sign == b4_sign)
    gates["(e) B4_eth_replication"] = b4_pass
    print(f"  BTC spot inner-val d_sharpe sign = {b1_spot_inner_val_sign:+.0f}, "
          f"ETH spot inner-val d_sharpe sign = {b4_sign:+.0f} -> "
          f"{'REPLICATES' if b4_pass else 'DOES NOT REPLICATE'}")

    # ------------------------------------------------------------ B5
    print()
    print("=" * 78)
    print("B5: 0.40% taker fee tier, full construction, BTC both markets, inner-val")
    print("=" * 78)
    b5_pass = True
    for mkt_name, mkt, base_mkt_name in (("spot_hi_fee", SPOT_HIGH_FEE, "spot"),
                                          ("futures_5x_hi_fee", FUTURES_HIGH_FEE, "futures_5x")):
        m_base, res_base = run_baseline(df_btc, mkt, INNER_VAL_START, INNER_VAL_END, label_btc)
        n_configs += 1
        strat = ConservativePositioningPanel()
        m_cand, res_cand = run_strategy(strat, df_btc, mkt, INNER_VAL_START, INNER_VAL_END, label_btc)
        n_configs += 1
        print(f"  [{mkt_name} / inner-val] cand: trades={m_cand.num_trades} "
              f"final={m_cand.final_balance:.1f} sharpe={m_cand.sharpe:.3f} | "
              f"base: trades={m_base.num_trades} final={m_base.final_balance:.1f} sharpe={m_base.sharpe:.3f}")
        key = f"B5_{mkt_name}"
        r = _report_diff(key, res_cand, res_base)
        results[key] = r
        base_sign = np.sign(b1_dsharpe[(base_mkt_name, "inner_val")])
        hi_fee_sign = np.sign(r["d_sharpe"])
        flip = base_sign != 0 and hi_fee_sign != base_sign
        b5_pass = b5_pass and (not flip)
        print(f"    normal-fee inner_val sign={base_sign:+.0f} vs hi-fee sign={hi_fee_sign:+.0f} -> "
              f"{'SIGN FLIP' if flip else 'no flip'}")
    gates["(f) B5_no_sign_flip"] = b5_pass
    print(f"  (f) B5 gate: {'PASS' if b5_pass else 'FAIL'}")

    # ------------------------------------------------------------ B6 (MANDATORY)
    print()
    print("=" * 78)
    print("B6 (MANDATORY): turnover-matched, information-free control vs candidate, "
          "on the decisive B1 cells (inner_train, full_inner), both markets")
    print("=" * 78)
    b6_decisive_periods = [
        ("inner_train", INNER_TRAIN_START, INNER_TRAIN_END),
        ("full_inner", INNER_TRAIN_START, INNER_VAL_END),
    ]
    b6_pass = True
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        target_trades = cand_trades_inner_val[mkt_name]
        for per_name, start, end in b6_decisive_periods:
            m_base, res_base = baselines[(mkt_name, per_name)]
            m_ctrl, res_ctrl, best_h, best_trades = turnover_matched_control(
                df_btc, mkt, target_trades, start, end, label_btc)
            n_configs += 14  # 13-point hysteresis grid search + 1 final run, internal to the helper
            sh_ctrl = sharpe_diff(res_ctrl, res_base)
            cand_d = b1_dsharpe[(mkt_name, per_name)]
            ctrl_d = sh_ctrl.diff.point
            beats = cand_d > ctrl_d
            b6_pass = b6_pass and beats
            print(f"  [{mkt_name}/{per_name}] control(h={best_h:.2f}, trades={best_trades}, "
                  f"target={target_trades}) d_sharpe={ctrl_d:+.4f}  |  "
                  f"candidate d_sharpe={cand_d:+.4f}  -> "
                  f"{'CANDIDATE BEATS CONTROL' if beats else 'CANDIDATE DOES NOT BEAT CONTROL'}")
            results[f"B6_control_{mkt_name}_{per_name}"] = {
                "name": f"B6_control_{mkt_name}_{per_name}", "d_sharpe": ctrl_d,
                "sharpe_lo": sh_ctrl.diff.lo, "sharpe_hi": sh_ctrl.diff.hi,
                "sharpe_sig": sh_ctrl.significant, "d_logret": float("nan"),
                "logret_lo": float("nan"), "logret_hi": float("nan"), "logret_sig": None,
            }
    gates["(g) B6_beats_turnover_matched_control"] = b6_pass
    print(f"  (g) B6 gate: {'PASS' if b6_pass else 'FAIL'} "
          f"(candidate must EXCEED the control's own d_sharpe on every decisive cell, "
          f"not merely share its sign)")

    # ------------------------------------------------------------ SUMMARY
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for key, r in results.items():
        print(f"{key:40s} d_sharpe={r['d_sharpe']:+.4f} CI=[{r['sharpe_lo']:+.4f},{r['sharpe_hi']:+.4f}] "
              f"sig={r['sharpe_sig']} | d_logret={r['d_logret']:+.4f} sig={r['logret_sig']}")

    print()
    print("=" * 78)
    print("DECISION RULE (verbatim from r135_shared.py's module docstring)")
    print("=" * 78)
    for gname, gpass in gates.items():
        print(f"  {gname:45s} {'PASS' if gpass else 'FAIL'}")
    overall = all(gates.values())
    print()
    print(f"OVERALL VERDICT: {'PROMOTE-candidate' if overall else 'NEGATIVE'}")
    print()
    print(f"TOTAL BACKTEST CONFIGURATIONS EVALUATED: {n_configs}")
