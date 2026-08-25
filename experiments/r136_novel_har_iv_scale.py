"""R-136 NOVEL branch: `kelly_regime_v4` with its fast vol estimator replaced
by `r136_shared.har_iv_vol` -- the a-priori equal-weight mean of the three
HAR (Corsi 2009) realized-vol components PLUS Deribit's causally-aligned
DVOL implied-vol index -- via `HARVolMixin`. See `experiments/r136_shared.py`'s
module docstring for the full round-level pre-registration (direction,
literature, both branches' mechanisms, named failure modes, falsification
test, decision rule). Nothing here deviates from that pre-registration
without saying so explicitly.

**Mechanism, frozen:** `NovelHARIVScale(HARVolMixin, KellyRegimeV4)` inherits
`HARVolMixin.prepare()` UNCHANGED (never overridden here -- verified in the
bug hunt below via an identity check, not just a promise) and supplies only
`_vol_series`, which calls `har_iv_vol(df, dvol_causal, windows_days)`. Every
other line of `prepare()` (vote, `slow`, `ratio`, the state machine,
`full`/`steady`, the deadband) is therefore bit-for-bit identical to the
registered `kelly_regime_v3`/`v4`, by construction rather than by copy-paste
discipline.

**Deviation check, resolved cleanly (flagged as instructed, even though the
answer is "no deviation needed"):** the task anticipated possibly having to
reuse BTC's DVOL as an ETH proxy if no ETH DVOL file existed.
`data/eth_dvol_daily.csv.gz` DOES exist (1977 rows, 2021-03-24 through
2026-08-21, identical coverage-start date to BTC's own DVOL file, same
open/high/low/close schema) but `tradebot.data.load_dvol_index` is hardcoded
to `DVOL_FILE = "btc_dvol_daily.csv.gz"` with no asset parameter, and
`r136_shared.load_dvol_causal_train` calls it unconditionally. Rather than
touch `src/tradebot/data.py` or `r136_shared.py` (out of scope -- one new
file only), this module defines `_load_eth_dvol_index`/
`load_dvol_causal_train_eth` below, mirroring `load_dvol_index`'s /
`load_dvol_causal_train`'s exact implementation but reading the ETH file --
the same pattern already used by `r135_novel_breadth_optimized_panel.py`'s
own `_load_eth_dvol_index`. B4 (ETH falsification) therefore uses ETH's OWN
DVOL, not a BTC proxy. No deviation from a clean design was required.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import align_dvol_causal  # noqa: E402
from tradebot.inference import daily_returns  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

from r136_shared import (  # noqa: E402
    B1_PERIODS,
    B3_WINDOW_SETS_DAYS,
    DVOL_COVERAGE_START,
    FUTURES,
    FUTURES_HIGH_FEE,
    HARVolMixin,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    SPOT,
    SPOT_HIGH_FEE,
    compute_rv_components,
    exposure_by_vol_quartile,
    har_iv_vol,
    har_rv_vol,
    load_btc_train,
    load_dvol_causal_train,
    load_eth_train,
    log_growth_diff,
    qlike_loss,
    run_baseline,
    run_strategy,
    sharpe_diff,
)

BACKTEST_COUNT = 0  # incremented by every call to _run()


def _run(strat, df, market, start, end, label=""):
    """Thin wrapper over run_strategy that counts every backtest executed."""
    global BACKTEST_COUNT
    BACKTEST_COUNT += 1
    return run_strategy(strat, df, market, start, end, label)


def _run_baseline(df, market, start, end, label=""):
    global BACKTEST_COUNT
    BACKTEST_COUNT += 1
    return run_baseline(df, market, start, end, label)


# ----------------------------------------------------------------------
# ETH's own DVOL (see deviation note in module docstring). Mirrors
# tradebot.data.load_dvol_index / r136_shared.load_dvol_causal_train
# exactly, swapped to the ETH file. Does not modify data.py or r136_shared.py.
# ----------------------------------------------------------------------

def _load_eth_dvol_index(data_dir) -> pd.DataFrame | None:
    path = Path(data_dir) / "eth_dvol_daily.csv.gz"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = df.index.tz_localize("UTC")
    return df.astype(float).sort_index()


def load_dvol_causal_train_eth(bars: pd.DataFrame) -> pd.Series:
    dvol = _load_eth_dvol_index(ROOT / "data")
    assert dvol is not None, "ETH DVOL data not committed"
    aligned = align_dvol_causal(dvol[["close"]], bars)
    return (aligned["close"] / 100.0).rename("dvol_eth")


# ----------------------------------------------------------------------
# The NOVEL candidate.
# ----------------------------------------------------------------------

class NovelHARIVScale(HARVolMixin, KellyRegimeV4):
    """kelly_regime_v4 with its EWM(8d) vol estimator replaced by
    har_iv_vol (HAR daily/weekly/monthly RV components + causally-aligned
    DVOL, a-priori equal-weight mean, falling back to the 3-way RV mean
    pre-DVOL-coverage)."""

    name = "r136_novel_har_iv_scale"

    def __init__(self, dvol_causal: pd.Series, windows_days=(1, 5, 22), **kwargs) -> None:
        super().__init__(**kwargs)
        self._dvol_causal = dvol_causal
        self._windows_days = windows_days

    def _vol_series(self, df: pd.DataFrame, r: pd.Series) -> np.ndarray:
        return har_iv_vol(df, self._dvol_causal.reindex(df.index), self._windows_days).to_numpy()


class _RVOnlyLocal(HARVolMixin, KellyRegimeV4):
    """Local reimplementation of the CONSERVATIVE branch's construction
    (har_rv_vol only, no DVOL) for B7's controlled within-file comparison.
    Does NOT import experiments/r136_conservative_har_rv_scale.py -- per
    task instructions, that file is owned by a concurrent agent and is not
    touched or imported here. This class exists ONLY to isolate "the DVOL
    term" as the sole difference from NovelHARIVScale for gate B7."""

    name = "r136_novel_har_iv_scale__rv_only_local_b7"

    def __init__(self, windows_days=(1, 5, 22), **kwargs) -> None:
        super().__init__(**kwargs)
        self._windows_days = windows_days

    def _vol_series(self, df: pd.DataFrame, r: pd.Series) -> np.ndarray:
        return har_rv_vol(df, self._windows_days).to_numpy()


def _baseline_ewm_vol(df: pd.DataFrame) -> np.ndarray:
    """Baseline kelly_regime_v4's own fast estimator: EWM(8d) of 5m log
    returns, identical formula to KellyRegime.prepare()."""
    close = df["close"]
    r = np.log(close).diff()
    return (r.ewm(span=8 * BARS_PER_DAY, min_periods=BARS_PER_DAY).std()
            * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()


def make_novel(df, dvol_causal, windows_days=(1, 5, 22)):
    return NovelHARIVScale(dvol_causal=dvol_causal, windows_days=windows_days)


def make_rv_only(windows_days=(1, 5, 22)):
    return _RVOnlyLocal(windows_days=windows_days)


# ----------------------------------------------------------------------
# Gate 1: causal truncation probe.
# ----------------------------------------------------------------------

def gate1_causal_truncation_probe(df, dvol_btc, label):
    print("\n=== GATE 1: causal truncation probe ===")

    # 1a. Literal mirror of r136_shared's own self-test (df already ends at
    # INNER_VAL_END via load_btc_train, so df.loc[:INNER_VAL_END] is a
    # no-op identity check -- included only for direct comparability with
    # r136_shared's own printed self-test).
    strat_a = make_novel(df, dvol_btc)
    m_full, _ = _run(strat_a, df, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)
    df_noop = df.loc[:INNER_VAL_END]
    dvol_noop = load_dvol_causal_train(df_noop)
    strat_b = make_novel(df_noop, dvol_noop)
    m_noop, _ = _run(strat_b, df_noop, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)
    ok_noop = np.isclose(m_full.final_balance, m_noop.final_balance, rtol=1e-9)
    print(f"1a (literal mirror, no-op since df already ends at INNER_VAL_END): "
          f"{'PASS' if ok_noop else 'FAIL'} ({m_full.final_balance} vs {m_noop.final_balance})")

    # 1b. Strengthened, genuinely non-trivial version: r136_shared's own
    # cutoff (INNER_VAL_END) coincides with load_btc_train's own truncation
    # point, making 1a a comparison of a dataframe against itself. Here the
    # "full" df carries real extra rows AFTER the query's own end
    # (INNER_TRAIN_END, i.e. all of 2021-2022) that a genuinely causal
    # prepare() must not read; the "truncated" df is hard-cut exactly at
    # the query end. This flag is a deliberate departure from the literal
    # r136_shared pattern, stated explicitly per instructions.
    df_trunc = df.loc[:INNER_TRAIN_END]
    dvol_trunc = load_dvol_causal_train(df_trunc)
    strat_c = make_novel(df_trunc, dvol_trunc)
    m_trunc, _ = _run(strat_c, df_trunc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)
    ok_trunc = np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9)
    print(f"1b (strengthened: df hard-cut at query end vs df extending 2 more years past it): "
          f"{'PASS' if ok_trunc else 'FAIL'} ({m_full.final_balance} vs {m_trunc.final_balance})")

    return ok_noop and ok_trunc


# ----------------------------------------------------------------------
# Gate 2: QLIKE descriptive check.
# ----------------------------------------------------------------------

def _daily_realized_vol(df):
    close = df["close"]
    r = np.log(close).diff()
    r2 = r.pow(2)
    daily_sum = r2.groupby(r2.index.floor("D")).sum()
    return np.sqrt(daily_sum * 365.25)


def _forecast_to_next_day(vol_5m: np.ndarray, index) -> pd.Series:
    """Resample a causal (already `.shift(1)`-lagged) 5m vol series to one
    forecast per day: the last bar's value observed during day D (itself
    only using info through day D's second-to-last bar), reindexed to
    predict day D+1's realized vol via a one-day shift."""
    s = pd.Series(vol_5m, index=index)
    daily_last = s.resample("1D").last()
    return daily_last.shift(1)


def gate2_qlike(df_btc, dvol_btc, label):
    print("\n=== GATE 2: QLIKE descriptive check (BTC spot, full_inner) ===")
    realized = _daily_realized_vol(df_btc)

    f_baseline = _forecast_to_next_day(_baseline_ewm_vol(df_btc), df_btc.index)
    f_novel = _forecast_to_next_day(
        har_iv_vol(df_btc, dvol_btc.reindex(df_btc.index)).to_numpy(), df_btc.index)
    f_rvonly = _forecast_to_next_day(har_rv_vol(df_btc).to_numpy(), df_btc.index)

    windows = {
        "pre_DVOL (<2021-03-24)": (None, pd.Timestamp(DVOL_COVERAGE_START, tz="UTC") - pd.Timedelta(days=1)),
        "DVOL_covered (>=2021-03-24)": (pd.Timestamp(DVOL_COVERAGE_START, tz="UTC"), None),
    }

    results = {}
    for wname, (wstart, wend) in windows.items():
        idx = realized.index
        if wstart is not None:
            idx = idx[idx >= wstart]
        if wend is not None:
            idx = idx[idx <= wend]
        y = realized.reindex(idx)
        fb = f_baseline.reindex(idx)
        fn = f_novel.reindex(idx)
        fr = f_rvonly.reindex(idx)
        mask = np.isfinite(y) & np.isfinite(fb) & np.isfinite(fn) & np.isfinite(fr) & (y > 0) & (fb > 0) & (fn > 0) & (fr > 0)
        n = int(mask.sum())
        q_base = qlike_loss(fb[mask].to_numpy(), y[mask].to_numpy())
        q_novel = qlike_loss(fn[mask].to_numpy(), y[mask].to_numpy())
        q_rvonly = qlike_loss(fr[mask].to_numpy(), y[mask].to_numpy())
        results[wname] = dict(n=n, baseline=q_base, novel=q_novel, rv_only=q_rvonly)
        print(f"  {wname}: n={n} days | QLIKE baseline_ewm8d={q_base:.5f} "
              f"novel_har_iv={q_novel:.5f} conservative_har_rv={q_rvonly:.5f} "
              f"(lower=better; novel<baseline={q_novel < q_base}, novel<rv_only={q_novel < q_rvonly})")
    return results


# ----------------------------------------------------------------------
# Gate 3: B1.
# ----------------------------------------------------------------------

def gate3_b1(df_btc, dvol_btc, label):
    print("\n=== GATE 3: B1 (both markets x 3 periods, candidate vs frozen kelly_regime_v4) ===")
    out = {}
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        for per_name, start, end in B1_PERIODS:
            strat = make_novel(df_btc, dvol_btc)
            m_c, res_c = _run(strat, df_btc, mkt, start, end, label)
            m_b, res_b = _run_baseline(df_btc, mkt, start, end, label)
            lg = log_growth_diff(res_c, res_b)
            sh = sharpe_diff(res_c, res_b)
            out[(mkt_name, per_name)] = dict(m_c=m_c, m_b=m_b, res_c=res_c, res_b=res_b, lg=lg, sh=sh)
            print(f"  {mkt_name}/{per_name}: cand[final={m_c.final_balance:.1f} sharpe={m_c.sharpe:.3f} "
                  f"trades={m_c.num_trades} dd={m_c.max_drawdown_pct:.1f}%] "
                  f"base[final={m_b.final_balance:.1f} sharpe={m_b.sharpe:.3f} "
                  f"trades={m_b.num_trades} dd={m_b.max_drawdown_pct:.1f}%] "
                  f"d_log_return={lg.diff} sig={lg.significant} d_sharpe={sh.diff}")
    return out


# ----------------------------------------------------------------------
# Gate 4: B3 plateau.
# ----------------------------------------------------------------------

def gate4_b3(df_btc, dvol_btc, label, frozen_sign, baseline_full_inner_spot):
    print("\n=== GATE 4: B3 plateau (HAR window-set sweep, SPOT, full_inner) ===")
    m_b, res_b = baseline_full_inner_spot
    signs = {(1, 5, 22): frozen_sign}
    for windows_days in B3_WINDOW_SETS_DAYS:
        if windows_days == (1, 5, 22):
            continue  # already computed as the frozen candidate in gate 3
        strat = make_novel(df_btc, dvol_btc, windows_days=windows_days)
        m_c, res_c = _run(strat, df_btc, SPOT, INNER_TRAIN_START, INNER_VAL_END, label)
        lg = log_growth_diff(res_c, res_b)
        pt = lg.diff.point
        signs[windows_days] = pt
        print(f"  windows_days={windows_days}: final={m_c.final_balance:.1f} "
              f"d_log_return={lg.diff} sign={'+' if pt > 0 else ('-' if pt < 0 else '0')}")
    n_agree = sum(1 for v in signs.values() if np.sign(v) == np.sign(frozen_sign))
    majority = n_agree >= (len(signs) + 1) // 2
    print(f"  sign agreement with frozen (1,5,22) [sign={'+' if frozen_sign>0 else '-'}]: "
          f"{n_agree}/{len(signs)} variants -> {'PASS (majority)' if majority else 'FAIL'}")
    return majority, signs


# ----------------------------------------------------------------------
# Gate 5: B4 ETH falsification.
# ----------------------------------------------------------------------

def gate5_b4(btc_inner_val_sign):
    print("\n=== GATE 5: B4 ETH falsification (SPOT, inner_val) ===")
    eth_df = load_eth_train()
    dvol_eth = load_dvol_causal_train_eth(eth_df)
    strat = make_novel(eth_df, dvol_eth)
    m_c, res_c = _run(strat, eth_df, SPOT, INNER_VAL_START, INNER_VAL_END, "eth_spot")
    m_b, res_b = _run_baseline(eth_df, SPOT, INNER_VAL_START, INNER_VAL_END, "eth_spot")
    lg = log_growth_diff(res_c, res_b)
    pt = lg.diff.point
    replicates = np.sign(pt) == np.sign(btc_inner_val_sign)
    print(f"  ETH spot/inner_val: cand[final={m_c.final_balance:.1f} sharpe={m_c.sharpe:.3f}] "
          f"base[final={m_b.final_balance:.1f} sharpe={m_b.sharpe:.3f}] "
          f"d_log_return={lg.diff} sig={lg.significant}")
    print(f"  sign replicates BTC spot/inner_val sign "
          f"({'+' if btc_inner_val_sign>0 else '-'}): {'PASS' if replicates else 'FAIL'}")
    return replicates, lg


# ----------------------------------------------------------------------
# Gate 6: B5 fee tier.
# ----------------------------------------------------------------------

def gate6_b5(df_btc, dvol_btc, label, b1_full_inner_signs):
    print("\n=== GATE 6: B5 fee tier (full_inner, 0.40% taker) ===")
    out = {}
    for mkt_name, mkt, base_sign in (
        ("spot_high_fee", SPOT_HIGH_FEE, b1_full_inner_signs["spot"]),
        ("futures_5x_high_fee", FUTURES_HIGH_FEE, b1_full_inner_signs["futures_5x"]),
    ):
        strat = make_novel(df_btc, dvol_btc)
        m_c, res_c = _run(strat, df_btc, mkt, INNER_TRAIN_START, INNER_VAL_END, label)
        m_b, res_b = _run_baseline(df_btc, mkt, INNER_TRAIN_START, INNER_VAL_END, label)
        lg = log_growth_diff(res_c, res_b)
        pt = lg.diff.point
        survives = np.sign(pt) == np.sign(base_sign)
        out[mkt_name] = dict(lg=lg, survives=survives)
        print(f"  {mkt_name}: cand[final={m_c.final_balance:.1f}] base[final={m_b.final_balance:.1f}] "
              f"d_log_return={lg.diff} sign_survives_vs_std_fee={'PASS' if survives else 'FAIL'}")
    all_survive = all(v["survives"] for v in out.values())
    return all_survive, out


# ----------------------------------------------------------------------
# Gate 7: B6 mandatory risk-match.
# ----------------------------------------------------------------------

def _time_in_market(res):
    t = res.df["target"].to_numpy()
    return float(np.mean(np.abs(t) > 1e-9))


def _realized_vol_ann(res):
    rets = daily_returns(res.equity)
    return float(rets.std(ddof=1) * np.sqrt(365.25))


def gate7_b6(b1_out, df_btc, dvol_btc):
    print("\n=== GATE 7: B6 mandatory risk-match (full_inner, both markets) ===")
    out = {}
    for mkt_name in ("spot", "futures_5x"):
        cell = b1_out[(mkt_name, "full_inner")]
        res_c, res_b = cell["res_c"], cell["res_b"]
        tim_c, tim_b = _time_in_market(res_c), _time_in_market(res_b)
        rv_c, rv_b = _realized_vol_ann(res_c), _realized_vol_ann(res_b)

        # exposure_by_vol_quartile needs the SAME vol series driving ratio/
        # state, recomputed over each result's own trimmed measured-period
        # frame (res.df), not the full multi-year df.
        vol_cand = har_iv_vol(res_c.df, dvol_btc.reindex(res_c.df.index)).to_numpy()
        vol_base = _baseline_ewm_vol(res_b.df)
        exp_c = exposure_by_vol_quartile(res_c.df["target"].to_numpy(), vol_cand)
        exp_b = exposure_by_vol_quartile(res_b.df["target"].to_numpy(), vol_base)

        out[mkt_name] = dict(tim_c=tim_c, tim_b=tim_b, rv_c=rv_c, rv_b=rv_b, exp_c=exp_c, exp_b=exp_b)
        print(f"  {mkt_name}: time_in_market cand={tim_c:.1%} base={tim_b:.1%} | "
              f"realized_vol(equity) cand={rv_c:.3f} base={rv_b:.3f}")
        print(f"    exposure by vol quartile (q1..q4, mean |target|):")
        print(f"      cand: {exp_c}")
        print(f"      base: {exp_b}")
        q4_ratio = exp_c["q4"] / exp_b["q4"] if exp_b["q4"] not in (0, None) and np.isfinite(exp_b["q4"]) else float("nan")
        print(f"    high-vol-quartile (q4) cand/base exposure ratio: {q4_ratio:.3f} "
              f"({'de-leveraging in high-vol' if q4_ratio < 1 else 'no de-leveraging / more exposed'})")
    return out


# ----------------------------------------------------------------------
# Gate 8: B7 coverage-artifact check (NOVEL-only, mandatory).
# ----------------------------------------------------------------------

def gate8_b7(df_btc, dvol_btc, label):
    print("\n=== GATE 8: B7 coverage-artifact check (NOVEL-only, mandatory) ===")
    pre_dvol_end = (pd.Timestamp(DVOL_COVERAGE_START) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    subwindows = [
        ("pre_DVOL", INNER_TRAIN_START, pre_dvol_end),
        ("DVOL_covered", DVOL_COVERAGE_START, INNER_VAL_END),
    ]
    out = {}
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        for wname, start, end in subwindows:
            strat_novel = make_novel(df_btc, dvol_btc)
            m_n, res_n = _run(strat_novel, df_btc, mkt, start, end, label)
            strat_rv = make_rv_only()
            m_r, res_r = _run(strat_rv, df_btc, mkt, start, end, label)
            lg = log_growth_diff(res_n, res_r)
            out[(mkt_name, wname)] = dict(m_n=m_n, m_r=m_r, lg=lg)
            print(f"  {mkt_name}/{wname}: novel[final={m_n.final_balance:.1f} trades={m_n.num_trades}] "
                  f"rv_only[final={m_r.final_balance:.1f} trades={m_r.num_trades}] "
                  f"d_log_return(novel-rv_only)={lg.diff} sig={lg.significant}")

    print("\n  --- B7 verdict per market: is novel's advantage concentrated in DVOL_covered? ---")
    concentrated = {}
    for mkt_name in ("spot", "futures_5x"):
        pre = out[(mkt_name, "pre_DVOL")]["lg"].diff.point
        dvc = out[(mkt_name, "DVOL_covered")]["lg"].diff.point
        # "concentrated": DVOL-covered advantage is clearly positive and
        # materially larger in magnitude than the pre-DVOL residual (which
        # should be ~0 by construction -- see bug hunt).
        conc = (dvc > 0) and (abs(dvc) > abs(pre))
        concentrated[mkt_name] = conc
        print(f"  {mkt_name}: pre_DVOL d_log_return point={pre:.5f}  DVOL_covered point={dvc:.5f}  "
              f"concentrated_in_DVOL_covered={'YES' if conc else 'NO'}")
    b7_pass = all(concentrated.values())
    print(f"  B7 overall: {'PASS' if b7_pass else 'FAIL'}")
    return b7_pass, out


# ----------------------------------------------------------------------
# Gate 9: bug hunt.
# ----------------------------------------------------------------------

def gate9_bug_hunt(df_btc, dvol_btc):
    print("\n=== GATE 9: bug hunt ===")
    findings = []

    # 1. prepare() override check: NovelHARIVScale must NOT override
    # prepare() at all -- it should resolve to HARVolMixin.prepare via MRO,
    # identical to r136_shared's own already-reviewed implementation.
    ok1 = NovelHARIVScale.prepare is HARVolMixin.prepare
    findings.append(("NovelHARIVScale does not override prepare() "
                      "(resolves to HARVolMixin.prepare via MRO)", ok1))
    ok1b = _RVOnlyLocal.prepare is HARVolMixin.prepare
    findings.append(("_RVOnlyLocal does not override prepare() either", ok1b))

    # 2. har_iv_vol's shift(1)/causal-alignment guards: compute_rv_components
    # shifts each component by 1 bar (read from r136_shared source); spot-
    # check numerically that shifting is in effect by confirming bar i's
    # component value does not change when bar i's OWN close is perturbed.
    probe = df_btc.iloc[:5000].copy()
    comp_a = compute_rv_components(probe)
    probe2 = probe.copy()
    probe2.iloc[-1, probe2.columns.get_loc("close")] *= 1.5  # perturb only the LAST bar's close
    comp_b = compute_rv_components(probe2)
    # every row except possibly the very last should be identical (last
    # bar's OWN close must not affect its OWN shifted RV components)
    unaffected = np.allclose(
        comp_a.iloc[-1].to_numpy(dtype=float), comp_b.iloc[-1].to_numpy(dtype=float),
        equal_nan=True,
    )
    findings.append(("compute_rv_components: perturbing bar i's own close leaves "
                      "bar i's own (shifted) RV components unchanged (no same-bar lookahead)",
                      unaffected))

    # 3. dvol NaN-before-coverage is never filled/backfilled anywhere in
    # this file: empirically confirm har_iv_vol degrades EXACTLY to
    # har_rv_vol (the 3-way mean, no dvol contribution at all) for every
    # bar strictly before DVOL_COVERAGE_START.
    iv = har_iv_vol(df_btc, dvol_btc.reindex(df_btc.index))
    rv = har_rv_vol(df_btc)
    pre_mask = df_btc.index < pd.Timestamp(DVOL_COVERAGE_START, tz=df_btc.index.tz)
    both_nan = iv[pre_mask].isna() & rv[pre_mask].isna()
    close_enough = np.isclose(
        iv[pre_mask].to_numpy(dtype=float), rv[pre_mask].to_numpy(dtype=float),
        equal_nan=True,
    )
    ok3 = bool(np.all(close_enough | both_nan.to_numpy()))
    findings.append(("har_iv_vol == har_rv_vol exactly for every bar strictly before "
                      "DVOL_COVERAGE_START (dvol NaN never filled/backfilled -- confirmed "
                      "empirically, not just by code reading)", ok3))

    # 4. dvol_causal itself: confirm no NaN inside this file's own DVOL
    # loader output is ever filled (grep-level self-check: no line in this
    # file's own source calls .fillna/.bfill on a dvol-named series).
    import inspect
    src_lines = inspect.getsource(sys.modules[__name__]).splitlines()
    no_dvol_fill = all(
        not (("dvol" in line) and (".fillna(" in line or ".bfill(" in line))
        for line in src_lines
    )
    findings.append(("this file's own source never calls .fillna/.bfill on any dvol series",
                      no_dvol_fill))

    for desc, ok in findings:
        print(f"  [{'OK' if ok else 'FAIL'}] {desc}")
    all_ok = all(ok for _, ok in findings)
    print(f"  bug hunt overall: {'CLEAN' if all_ok else 'ISSUES FOUND'}")
    return all_ok, findings


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    df_btc, label = load_btc_train("spot")
    dvol_btc = load_dvol_causal_train(df_btc)
    print(f"loaded BTC spot train: {len(df_btc)} bars, DVOL non-NaN from "
          f"{dvol_btc.dropna().index.min()}")

    ok1 = gate1_causal_truncation_probe(df_btc, dvol_btc, label)
    if not ok1:
        print("\n*** GATE 1 (causal truncation probe) FAILED. STOPPING per task "
              "instructions -- no further gate is trustworthy. ***")
        sys.exit(1)

    gate2_results = gate2_qlike(df_btc, dvol_btc, label)

    b1_out = gate3_b1(df_btc, dvol_btc, label)

    frozen_full_inner_spot_sign = b1_out[("spot", "full_inner")]["lg"].diff.point
    b3_pass, b3_signs = gate4_b3(
        df_btc, dvol_btc, label, frozen_full_inner_spot_sign,
        (b1_out[("spot", "full_inner")]["m_b"], b1_out[("spot", "full_inner")]["res_b"]),
    )

    btc_inner_val_sign = b1_out[("spot", "inner_val")]["lg"].diff.point
    b4_pass, b4_lg = gate5_b4(btc_inner_val_sign)

    b1_full_inner_signs = {
        "spot": b1_out[("spot", "full_inner")]["lg"].diff.point,
        "futures_5x": b1_out[("futures_5x", "full_inner")]["lg"].diff.point,
    }
    b5_pass, b5_out = gate6_b5(df_btc, dvol_btc, label, b1_full_inner_signs)

    b6_out = gate7_b6(b1_out, df_btc, dvol_btc)

    b7_pass, b7_out = gate8_b7(df_btc, dvol_btc, label)

    bug_hunt_clean, bug_findings = gate9_bug_hunt(df_btc, dvol_btc)

    # ------------------------------------------------------------------
    # Decision rule, applied verbatim.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DECISION RULE (verbatim from r136_shared.py module docstring)")
    print("=" * 70)

    clause_a = ok1
    b1_full_inner_pos = {
        m: b1_out[(m, "full_inner")]["lg"].diff.point > 0 for m in ("spot", "futures_5x")
    }
    b1_inner_val_pos = {
        m: b1_out[(m, "inner_val")]["lg"].diff.point > 0 for m in ("spot", "futures_5x")
    }
    b1_any_sig = any(
        b1_out[(m, p)]["lg"].significant
        for m in ("spot", "futures_5x") for p in ("full_inner", "inner_val")
    )
    clause_b = all(b1_full_inner_pos.values()) and all(b1_inner_val_pos.values()) and b1_any_sig
    clause_c = b3_pass
    clause_d = b4_pass
    clause_e = b5_pass
    clause_f = True  # B6 is a mandatory REPORT, not a pass/fail gate -- reported above
    clause_g = b7_pass

    all_clauses = dict(a=clause_a, b=clause_b, c=clause_c, d=clause_d,
                        e=clause_e, f=clause_f, g=clause_g)
    for k, v in all_clauses.items():
        print(f"  clause ({k}): {'PASS' if v else 'FAIL'}")

    promote = all(all_clauses.values())
    print(f"\nVERDICT: {'PROMOTE-candidate (kelly_regime_v5, alongside incumbent)' if promote else 'NEGATIVE'}")

    print(f"\nTotal backtests run: {BACKTEST_COUNT}")
    print(f"Total wall time: {time.time() - t_start:.1f}s")
