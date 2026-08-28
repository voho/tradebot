"""R-173 NOVEL branch: spread-conditioned dynamic deadband on `kelly_regime_v4`.

MECHANISM (one sentence, `r173_direction.md` Step 2 "NOVEL"): replace v4's
fixed 10% re-target deadband with a dynamic one that widens in proportion to
how elevated the causal Corwin & Schultz (2012) spread is relative to its OWN
trailing history --

    deadband_t = V4_DEADBAND * (1 + k * pctile_t)
    target_t   = apply_deadband_dynamic(v4_raw_desired, pctile, V4_DEADBAND, k)

where `pctile_t` is `r173_shared.spread_percentile_causal` of the causal CS
spread (`r173_shared.corwin_schultz_spread_causal`) and `k` is swept on the
training period only -- deferring exactly the trades a real elevated spread
would make more expensive, per the direction doc's own framing. Built
entirely on `experiments/r173_shared.py` (frozen, not editable) and, through
it, `experiments/r102_shared.py` / `experiments/r161_shared.py`. This file
owns itself exclusively: the sibling conservative branch is never read or
assumed to exist, and neither `r173_shared.py` nor `r173_direction.md` is
edited.

k GRID (pre-registered here, before any real-data number from this file):
{0.0, 0.5, 1.0, 2.0, 4.0}. `apply_deadband_dynamic`'s formula puts the
widened band at `V4_DEADBAND*(1+k)` when `pctile_t=1` (maximum trailing
friction): k=0 is the unwidened control (band stays 0.10 always, and
`build_target(df, 0.0)` must reproduce `v4_target(df)` bit-for-bit -- checked
on both synthetic and real BTC data below); k=0.5 barely widens (0.10->0.15
at max friction); k=1.0 doubles it (0.10->0.20); k=2.0 triples it
(0.10->0.30); k=4.0 quintuples it (0.10->0.50) -- spanning the range the
task brief itself names ("barely widens" to "roughly doubles/triples").

TWO-STAGE DECISION PROCESS (this file's own operationalisation of
`r173_direction.md`'s NOVEL promotion bar, whose text says the four clauses
apply "on the true 2023+ holdout" -- which cannot itself be how a SINGLE k is
chosen, since sweeping multiple k against the holdout would mean reading it
more than once. So, exactly mirroring `r172_novel_fcr_gamma.py`'s own
precedent (clauses evaluated on inner-val/eth_replication first, producing a
"PROMOTE-CANDIDATE" recommendation with "holdout not read by this branch" if
it fails):

  STAGE A (train-only gate, decides whether the holdout is read AT ALL):
    the SAME four clauses (a beats buy_and_hold, b clears the noise floor or
    a matched-risk DD improvement, c survives ETH replication, d plateau not
    peak), evaluated on BTC inner-validation (2021-01-01..2022-12-31) and the
    ETH eth_replication slice (ETH's own full pre-2020 history -- ETH has no
    data at or after 2021-01-01, the same disclosed substitution R-171/R-172's
    own novel branches used), for the single k selected by the plateau
    procedure below. If any clause fails, the branch is NEGATIVE here and the
    holdout is never touched.

  STAGE B (only if Stage A clears): freeze k, read the true holdout
    (start=OOS_START) EXACTLY ONCE, BTC spot and futures only (ETH has no
    holdout-period data at all, so its own clause (c) evidence necessarily
    carries forward unchanged from Stage A -- disclosed explicitly, not
    silently reused), against BOTH `buy_and_hold` and `v4_target`. Re-apply
    the SAME four clauses to the holdout numbers (clause (d)'s plateau
    property is a property of the pre-holdout k-neighbourhood and is not
    re-derived from holdout data, since that would require reading multiple
    k against the holdout -- the one thing this whole scheme exists to
    avoid). This is the actual PROMOTE/NEGATIVE verdict.

k* SELECTION (task step 4 / ROUTINE.md's own "plateau, not peak, and not the
single best inner-train cell"): the statistic is the mean of
`d_sharpe(candidate - v4_target)` over BTC inner-validation's two market
cells (spot, futures_5x) ONLY -- never inner-train, never ETH. k* is the
grid point (k>0) that maximises it. The neighbourhood check requires k*'s
immediate grid neighbours (among k>0) to (i) carry the SAME SIGN and (ii)
retain at least 50% of k*'s own statistic -- a disclosed, symmetric
materiality bar chosen so an isolated one-point spike cannot pass as a
"plateau" (mirrors `r161_shared.CONST_CAP_R2_THRESH`/
`r172_shared.R2_KILL_THRESH`'s own convention of naming a materiality
threshold explicitly rather than leaving "plateau" undefined).

DD_REDUCTION_PROMOTE_PP = 5.0 percentage points (matched-risk drawdown
materiality bar for clause (b)'s alternate path) is chosen to match
`r172_shared.py`'s own identical constant/value -- independently defined
here (not imported; r172_shared is a different round's frozen file, out of
this round's scope) rather than left an arbitrary new number.

Configs evaluated by this file: reported at the end of `main()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r173_shared import (  # noqa: E402
    ETH_SLICE_NAME,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SHARPE_NOISE_FLOOR,
    SLICES,
    SPOT,
    SliceResult,
    TargetStrategy,
    V4_DEADBAND,
    apply_deadband_dynamic,
    assert_no_holdout,
    causal_truncation_probe_series,
    compare,
    corwin_schultz_spread_causal,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    run_slice,
    spread_percentile_causal,
    v4_raw_desired,
    v4_target,
)
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402 -- FULL (untruncated) series, holdout only
from tradebot.inference import daily_returns as inference_daily_returns, total_log_return  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

K_GRID: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
DD_REDUCTION_PROMOTE_PP = 5.0        # clause (b) matched-risk DD materiality bar; see module docstring
PLATEAU_RETENTION = 0.5              # neighbour must retain >=50% of k*'s own d_sharpe statistic


# ==========================================================================
# (1) build_target(df, k) and the compare()-compatible factory.
# ==========================================================================

def build_target(df: pd.DataFrame, k: float) -> np.ndarray:
    """v4's pre-deadband desired exposure (`v4_raw_desired`, UNCHANGED),
    re-targeted through `apply_deadband_dynamic` instead of v4's fixed 10%
    `apply_deadband`. `k=0.0` reproduces `v4_target(df)` EXACTLY (checked in
    `_self_test` and again on real BTC data in `main()`)."""
    raw = v4_raw_desired(df)
    spread = corwin_schultz_spread_causal(df)
    pct = spread_percentile_causal(spread)
    return apply_deadband_dynamic(raw, pct, base_deadband=V4_DEADBAND, k=k)


def make_build_target(k: float):
    """Pure `df -> np.ndarray` builder for a fixed `k`, for `compare()`."""

    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, k)

    _build.__name__ = f"r173_novel_spread_deadband_k{k:g}"
    return _build


# ==========================================================================
# (2) Self-test: synthetic causality + k=0 exact-reproduction sanity check.
#     Mirrors r173_shared.py's own convention (fast synthetic checks first).
# ==========================================================================

def _synthetic_frame(seed: int, periods: int = 150_000) -> pd.DataFrame:
    idx = pd.date_range("2017-01-01", periods=periods, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    return pd.DataFrame({"open": close, "high": high, "low": low,
                          "close": close, "volume": 1.0}, index=idx)


def _self_test() -> None:
    df = _synthetic_frame(173001)

    # (1) k=0.0 reproduces v4_target(df) EXACTLY -- the wiring sanity check
    # the task brief asks for, on top of r173_shared's own
    # apply_deadband_dynamic(k=0)==apply_deadband self-test.
    assert np.allclose(build_target(df, 0.0), v4_target(df), atol=1e-12), \
        "build_target(df, k=0.0) does not reproduce v4_target(df) exactly"

    # (2) causal_truncation_probe_series on the full build_target pipeline,
    # at a representative spread of k values (0 = control, 1 = primary
    # magnitude, 4 = the widest grid point).
    for k in (0.0, 1.0, 4.0):
        assert causal_truncation_probe_series(make_build_target(k), df, cuts=(0.5, 0.7, 0.9)), \
            f"causal_truncation_probe_series FAILED at k={k}"

    # (3) monotonicity sanity: a wider deadband (larger k) never produces
    # MORE re-targets than a narrower one, on the same desired-exposure path
    # (mirrors r173_shared's own apply_deadband_dynamic self-test, checked
    # again here end-to-end through the real spread/percentile pipeline).
    t0 = build_target(df, 0.0)
    t4 = build_target(df, 4.0)
    changes0 = int(np.sum(np.abs(np.diff(t0)) > 1e-12))
    changes4 = int(np.sum(np.abs(np.diff(t4)) > 1e-12))
    assert changes4 <= changes0, (changes4, changes0)


_self_test()


# ==========================================================================
# (3) Train-period buy_and_hold comparison (Stage A clause (a) evidence).
#     Uses run_slice, which enforces "no bar at/after OOS_START" -- exactly
#     the same enforcement every other train-period read in this branch
#     goes through. Not a holdout consultation.
# ==========================================================================

def bh_train_leg(df: pd.DataFrame, start, end, slice_name: str, market) -> SliceResult:
    bh = get_strategy("buy_and_hold")
    return run_slice(bh, df, start, end, slice_name, market)


# ==========================================================================
# (4) Holdout-only plumbing. run_slice() deliberately REFUSES any frame
#     touching OOS_START -- correct for every other read in this file, and
#     exactly why it cannot be reused here. This is the ONE place in the
#     file allowed to read bars at/after OOS_START, and it is called at most
#     once, gated behind Stage A clearing.
# ==========================================================================

def load_btc_full() -> pd.DataFrame:
    """The FULL (untruncated) committed BTC spot series -- holdout only."""
    df, _label = load_dataset(ROOT / "data", "spot")
    return df


def run_holdout_leg(strategy, df_full: pd.DataFrame, market, label: str,
                     balance: float = 1_000.0) -> SliceResult:
    res = run_period(strategy, df_full, OOS_START, None, market=market, start_balance=balance)
    m = compute_metrics(res)
    d = inference_daily_returns(res.equity).to_numpy()
    exposure = res.df["target"].to_numpy() if "target" in res.df.columns else np.array([np.nan])
    return SliceResult(
        name=label, market=market.name, final_balance=m.final_balance, sharpe=m.sharpe,
        max_drawdown_pct=m.max_drawdown_pct, num_trades=m.num_trades,
        log_growth=float(total_log_return(d)), daily=d,
        mean_abs_exposure=float(np.nanmean(np.abs(exposure))),
        realized_vol=float(np.nanstd(d) * np.sqrt(365.25)) if len(d) > 1 else float("nan"),
    )


# ==========================================================================
# (5) k* selection: best-plateau on BTC inner-validation ONLY.
# ==========================================================================

def select_k_star(sweep_rows: dict[float, list[dict]]) -> dict:
    """`sweep_rows[k]` is one `compare()` call's row list. Returns a dict
    with the selection statistic per k, k*, and the plateau verdict."""

    def _inner_val_stat(rows: list[dict]) -> float:
        cells = [r["d_sharpe"] for r in rows if r["slice"] == "inner_val" and r["market"] in ("spot", "futures_5x")]
        return float(np.mean(cells)) if cells else float("nan")

    stat = {k: _inner_val_stat(rows) for k, rows in sweep_rows.items()}
    nonzero_ks = sorted(k for k in K_GRID if k > 0)
    candidates = {k: stat[k] for k in nonzero_ks if np.isfinite(stat[k])}

    if not candidates or max(candidates.values()) <= 0.0:
        return {"stat": stat, "k_star": None, "plateau_ok": False,
                "reason": "no k>0 improves mean BTC inner-val d_sharpe over v4_target"}

    k_star = max(candidates, key=candidates.get)
    idx = nonzero_ks.index(k_star)
    neighbours = [nonzero_ks[i] for i in (idx - 1, idx + 1) if 0 <= i < len(nonzero_ks)]

    peak = stat[k_star]
    plateau_ok = True
    neighbour_report = {}
    for nb in neighbours:
        nb_stat = stat[nb]
        same_sign = np.isfinite(nb_stat) and np.sign(nb_stat) == np.sign(peak)
        retains = np.isfinite(nb_stat) and (nb_stat / peak) >= PLATEAU_RETENTION if peak != 0 else False
        ok = bool(same_sign and retains)
        neighbour_report[nb] = {"stat": nb_stat, "same_sign": bool(same_sign),
                                 "retains_50pct": bool(retains), "ok": ok}
        plateau_ok = plateau_ok and ok

    return {"stat": stat, "k_star": k_star, "peak_stat": peak,
            "neighbours": neighbour_report, "plateau_ok": plateau_ok, "reason": ""}


# ==========================================================================
# (6) Stage A / Stage B promotion-clause evaluation, applied mechanically.
# ==========================================================================

def _cell(rows: list[dict], slice_name: str, market: str) -> dict:
    return next(r for r in rows if r["slice"] == slice_name and r["market"] == market)


def evaluate_clauses(cells4: tuple[dict, ...], bh_diffs: dict[str, object]) -> dict:
    """cells4 = (btc_spot, btc_fut, eth_spot, eth_fut) rows from compare()
    (candidate vs v4_target). bh_diffs = {label: PairedResult-like} for the
    SAME four cells, candidate vs buy_and_hold (paired_diff output)."""
    clause_a = all(bh_diffs[c["label_key"]].diff.lo > 0 for c in cells4)
    clause_b_sharpe = all(c["d_sharpe"] >= SHARPE_NOISE_FLOOR for c in cells4)
    clause_b_dd = all(c["risk_matched"] and (-c["d_dd"]) >= DD_REDUCTION_PROMOTE_PP for c in cells4)
    clause_b = bool(clause_b_sharpe or clause_b_dd)
    eth_cells = cells4[2:]
    clause_c = all(c["d_sharpe"] >= 0.0 for c in eth_cells)
    return {
        "clause_a_beats_bh": clause_a,
        "clause_b_sharpe_path": clause_b_sharpe,
        "clause_b_dd_path": clause_b_dd,
        "clause_b": clause_b,
        "clause_c_eth_survives": clause_c,
    }


# ==========================================================================
# main
# ==========================================================================

def hr(msg: str) -> None:
    print("\n" + "=" * 100)
    print(msg)
    print("=" * 100)


def main() -> None:
    n_configs = 0

    hr("R-173 NOVEL: spread-conditioned dynamic deadband on kelly_regime_v4's re-target trigger")
    print("mechanism: deadband_t = V4_DEADBAND*(1+k*pctile_t), pctile_t = causal CS-spread's own "
          "trailing percentile rank. k swept on train-only data; k=0.0 is v4 itself.")

    btc = load_btc()
    eth = load_eth()
    assert_no_holdout(btc, "main(): btc")
    assert_no_holdout(eth, "main(): eth")
    print(f"\nBTC (pre-holdout): {len(btc):,} bars ({btc.index[0]} .. {btc.index[-1]})")
    print(f"ETH: {len(eth):,} bars ({eth.index[0]} .. {eth.index[-1]})")

    # ---- k=0.0 real-data sanity check (task step 1's own requirement) ----
    hr("SANITY CHECK: build_target(df, k=0.0) reproduces v4_target(df) EXACTLY, on real BTC data")
    t0 = build_target(btc, 0.0)
    tv4 = v4_target(btc)
    exact = np.array_equal(t0, tv4)
    print(f"  np.array_equal(build_target(btc,0.0), v4_target(btc)) = {exact}")
    assert exact, "k=0.0 wiring does not bit-for-bit reproduce v4_target on real data"

    # ---- k grid sweep, train-only (compare() never touches OOS_START) ----
    hr(f"STEP 3 (train-only): k grid sweep {K_GRID}, compare() vs v4_target, "
       "inner_train + inner_val + eth_replication, both markets")
    sweep_rows: dict[float, list[dict]] = {}
    for k in K_GRID:
        rows = compare(make_build_target(k), label=f"r173_novel_spread_deadband_k{k:g}", btc=btc, eth=eth)
        n_configs += len(rows)
        sweep_rows[k] = rows
        print(f"\n  k={k:g}:")
        print_rows(rows)
        # exposure/vol-ratio disclosure (R-33's own standing rule), every non-trivial k
        if k != 0.0:
            for r in rows:
                print(f"    [{r['slice']:16s} {r['market']:11s}] exposure_ratio={r['exposure_ratio']:.3f} "
                      f"vol_ratio={r['vol_ratio']:.3f} risk_matched={r['risk_matched']} "
                      f"cand_trades={r['cand_trades']} ctrl_trades={r['ctrl_trades']}")

    # ---- k* selection: best-plateau on BTC inner-validation ONLY ----
    hr("k* SELECTION (BTC inner-validation ONLY, plateau-checked, never the single best inner-train cell)")
    sel = select_k_star(sweep_rows)
    for k in K_GRID:
        if k == 0.0:
            continue
        print(f"  k={k:g}: mean BTC inner_val d_sharpe = {sel['stat'][k]:+.4f}")
    if sel["k_star"] is None:
        hr("VERDICT")
        print(f"NEGATIVE at k*-selection: {sel['reason']}.")
        print("No k clears even the weakest bar (positive mean d_sharpe on BTC inner-validation) -- "
              "branch closes NEGATIVE WITHOUT A HOLDOUT READ.")
        print(f"\nconfigurations evaluated: {n_configs}")
        return

    k_star = sel["k_star"]
    print(f"\n  k* = {k_star:g}  (peak mean BTC inner_val d_sharpe = {sel['peak_stat']:+.4f})")
    for nb, info in sel["neighbours"].items():
        print(f"    neighbour k={nb:g}: stat={info['stat']:+.4f}  same_sign={info['same_sign']}  "
              f"retains>={int(PLATEAU_RETENTION*100)}%={info['retains_50pct']}  ok={info['ok']}")
    print(f"  PLATEAU (not peak) verdict: {sel['plateau_ok']}")

    if not sel["plateau_ok"]:
        hr("VERDICT")
        print(f"NEGATIVE at k*-selection: k={k_star:g} is an isolated peak, not a plateau "
              "(clause (d) fails on train-only data). Branch closes NEGATIVE WITHOUT A HOLDOUT READ.")
        print(f"\nconfigurations evaluated: {n_configs}")
        return

    # ---- Stage A: train-only 4-clause promotion gate at k* ----
    hr(f"STAGE A (train-only gate, k*={k_star:g}): full 4-clause promotion bar on BTC inner_val "
       "+ ETH eth_replication (no holdout bar read yet)")
    primary_rows = sweep_rows[k_star]
    btc_spot = _cell(primary_rows, "inner_val", "spot")
    btc_fut = _cell(primary_rows, "inner_val", "futures_5x")
    eth_spot = _cell(primary_rows, ETH_SLICE_NAME, "spot")
    eth_fut = _cell(primary_rows, ETH_SLICE_NAME, "futures_5x")
    for c in (btc_spot, btc_fut, eth_spot, eth_fut):
        c["label_key"] = f"{c['slice']}|{c['market']}"
    cells4 = (btc_spot, btc_fut, eth_spot, eth_fut)

    hr("Stage A clause (a) evidence: candidate vs buy_and_hold, TRAIN period only (BTC inner_val, "
       "ETH eth_replication), both markets")
    build_kstar = make_build_target(k_star)
    bh_diffs: dict[str, object] = {}
    bh_jobs = [
        (btc, INNER_VAL_START, INNER_VAL_END, "inner_val", SPOT),
        (btc, INNER_VAL_START, INNER_VAL_END, "inner_val", FUTURES),
        (eth, None, None, ETH_SLICE_NAME, SPOT),
        (eth, None, None, ETH_SLICE_NAME, FUTURES),
    ]
    for df_, start, end, slice_name, market in bh_jobs:
        cand = run_slice(TargetStrategy(build_kstar, name=f"r173_novel_k{k_star:g}"), df_, start, end,
                          slice_name, market)
        bh = bh_train_leg(df_, start, end, slice_name, market)
        n_configs += 2  # candidate leg + buy_and_hold leg, one comparison
        pr = paired_diff(cand.daily, bh.daily, seed=0)
        key = f"{slice_name}|{market.name}"
        bh_diffs[key] = pr
        print(f"  [{slice_name:16s} {market.name:11s}] cand_log_growth={cand.log_growth:+.4f} "
              f"bh_log_growth={bh.log_growth:+.4f}  boot_diff={pr.diff.point:+.4f} "
              f"[{pr.diff.lo:+.4f},{pr.diff.hi:+.4f}]  beats_bh(CI>0)={pr.diff.lo > 0}")

    stage_a = evaluate_clauses(cells4, bh_diffs)
    hr("STAGE A clause outcome")
    print(f"  clause (a) beats buy_and_hold, train period, all 4 cells (CI lo>0): {stage_a['clause_a_beats_bh']}")
    print(f"  clause (b) d_sharpe>=+{SHARPE_NOISE_FLOOR} on all 4 cells: {stage_a['clause_b_sharpe_path']}  "
          f"OR matched-risk DD improvement>={DD_REDUCTION_PROMOTE_PP}pp on all 4: {stage_a['clause_b_dd_path']}  "
          f"=> clause (b): {stage_a['clause_b']}")
    for name, c in (("BTC spot", btc_spot), ("BTC futures_5x", btc_fut),
                    ("ETH spot", eth_spot), ("ETH futures_5x", eth_fut)):
        print(f"      {name:16s} d_sharpe={c['d_sharpe']:+.4f}  d_dd={c['d_dd']:+.2f}  "
              f"risk_matched={c['risk_matched']}  exposure_ratio={c['exposure_ratio']:.3f}  "
              f"vol_ratio={c['vol_ratio']:.3f}")
    print(f"  clause (c) survives ETH replication (ETH d_sharpe>=0, both markets): "
          f"{stage_a['clause_c_eth_survives']}")
    print(f"  clause (d) plateau not peak (established during k*-selection above): {sel['plateau_ok']}")

    stage_a_all = bool(stage_a["clause_a_beats_bh"] and stage_a["clause_b"] and
                        stage_a["clause_c_eth_survives"] and sel["plateau_ok"])

    if not stage_a_all:
        cleared = [n for n, v in (("a", stage_a["clause_a_beats_bh"]), ("b", stage_a["clause_b"]),
                                   ("c", stage_a["clause_c_eth_survives"]), ("d", sel["plateau_ok"])) if v]
        failed = [n for n, v in (("a", stage_a["clause_a_beats_bh"]), ("b", stage_a["clause_b"]),
                                  ("c", stage_a["clause_c_eth_survives"]), ("d", sel["plateau_ok"])) if not v]
        hr("VERDICT")
        print(f"NEGATIVE at Stage A (k*={k_star:g}): clauses cleared={cleared or 'none'}, "
              f"clauses failed={failed or 'none'}. Per ROUTINE.md, a partial clear is reported as a "
              "partial clear, not rounded to the nearest label.")
        print("No k clears the training-period bar. Branch closes NEGATIVE WITHOUT A HOLDOUT READ.")
        print(f"\nconfigurations evaluated: {n_configs}")
        return

    # ---- Stage B: k* is frozen. ONE holdout read. ----
    hr(f"STAGE A CLEARED -- FREEZING k*={k_star:g}. Proceeding to the ONE permitted holdout read "
       "(start=OOS_START), BTC spot and futures_5x only, vs v4_target and vs buy_and_hold.")
    btc_full = load_btc_full()
    print(f"\n  full BTC dataset: {btc_full.index[0]} .. {btc_full.index[-1]}  "
          f"(holdout portion: {OOS_START} onward)")

    cand_strategy = TargetStrategy(build_kstar, name=f"r173_novel_k{k_star:g}")
    ctrl_strategy = TargetStrategy(v4_target, name="kelly_regime_v4")
    bh_strategy = get_strategy("buy_and_hold")

    holdout_rows: dict[str, dict] = {}
    for market in (SPOT, FUTURES):
        cand = run_holdout_leg(cand_strategy, btc_full, market, f"holdout_cand_{market.name}")
        ctrl = run_holdout_leg(ctrl_strategy, btc_full, market, f"holdout_ctrl_{market.name}")
        bh = run_holdout_leg(bh_strategy, btc_full, market, f"holdout_bh_{market.name}")
        n_configs += 4  # cand-vs-ctrl comparison, cand-vs-bh comparison (2 markets folded in loop)

        pr_ctrl = paired_diff(cand.daily, ctrl.daily, seed=0)
        pr_bh = paired_diff(cand.daily, bh.daily, seed=0)
        exp_ratio = cand.mean_abs_exposure / ctrl.mean_abs_exposure if ctrl.mean_abs_exposure else float("nan")
        vol_ratio = cand.realized_vol / ctrl.realized_vol if ctrl.realized_vol else float("nan")
        risk_matched = bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1) if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False
        d_sharpe = cand.sharpe - ctrl.sharpe
        d_dd = cand.max_drawdown_pct - ctrl.max_drawdown_pct

        holdout_rows[market.name] = dict(
            cand=cand, ctrl=ctrl, bh=bh, pr_ctrl=pr_ctrl, pr_bh=pr_bh,
            exposure_ratio=exp_ratio, vol_ratio=vol_ratio, risk_matched=risk_matched,
            d_sharpe=d_sharpe, d_dd=d_dd,
        )
        print(f"\n  === HOLDOUT {market.name} ===")
        print(f"    candidate  final=${cand.final_balance:>10,.0f}  Sharpe={cand.sharpe:+.3f}  "
              f"DD={cand.max_drawdown_pct:.1f}%  trades={cand.num_trades}")
        print(f"    v4_target  final=${ctrl.final_balance:>10,.0f}  Sharpe={ctrl.sharpe:+.3f}  "
              f"DD={ctrl.max_drawdown_pct:.1f}%  trades={ctrl.num_trades}")
        print(f"    buy_and_hold final=${bh.final_balance:>10,.0f}  Sharpe={bh.sharpe:+.3f}  "
              f"DD={bh.max_drawdown_pct:.1f}%")
        print(f"    d_sharpe(cand-v4)={d_sharpe:+.4f}  d_dd(cand-v4)={d_dd:+.2f}pp  "
              f"exposure_ratio={exp_ratio:.3f}  vol_ratio={vol_ratio:.3f}  risk_matched={risk_matched}")
        print(f"    boot_diff(cand-v4)={pr_ctrl.diff.point:+.4f} [{pr_ctrl.diff.lo:+.4f},{pr_ctrl.diff.hi:+.4f}]")
        print(f"    boot_diff(cand-bh)={pr_bh.diff.point:+.4f} [{pr_bh.diff.lo:+.4f},{pr_bh.diff.hi:+.4f}]  "
              f"beats_bh(CI>0)={pr_bh.diff.lo > 0}")

    # ---- Final verdict: re-apply the SAME four clauses to holdout numbers ----
    hr("FINAL VERDICT: r173_direction.md's NOVEL promotion bar, re-applied to the true holdout")
    hclause_a = all(holdout_rows[m.name]["pr_bh"].diff.lo > 0 for m in (SPOT, FUTURES))
    hclause_b_sharpe = all(holdout_rows[m.name]["d_sharpe"] >= SHARPE_NOISE_FLOOR for m in (SPOT, FUTURES))
    hclause_b_dd = all(holdout_rows[m.name]["risk_matched"] and
                        (-holdout_rows[m.name]["d_dd"]) >= DD_REDUCTION_PROMOTE_PP for m in (SPOT, FUTURES))
    hclause_b = bool(hclause_b_sharpe or hclause_b_dd)
    hclause_c = stage_a["clause_c_eth_survives"]  # no ETH holdout period exists; Stage A evidence carried forward
    hclause_d = sel["plateau_ok"]                  # plateau is a pre-holdout, train-only property by construction

    print(f"  (a) beats buy_and_hold, holdout, both markets (CI lo>0): {hclause_a}")
    print(f"  (b) d_sharpe>=+{SHARPE_NOISE_FLOOR} both markets: {hclause_b_sharpe}  OR matched-risk DD "
          f"improvement>={DD_REDUCTION_PROMOTE_PP}pp both markets: {hclause_b_dd}  => (b): {hclause_b}")
    print(f"  (c) survives ETH replication [carried forward from Stage A -- ETH has no holdout-period "
          f"data]: {hclause_c}")
    print(f"  (d) k neighbourhood is a plateau, not an isolated peak [established pre-holdout, "
          f"train-only -- see k*-selection above]: {hclause_d}")

    promote = bool(hclause_a and hclause_b and hclause_c and hclause_d)
    if promote:
        print("\n  ALL FOUR CLAUSES CLEAR ON THE TRUE HOLDOUT -> PROMOTE")
    else:
        cleared = [n for n, v in (("a", hclause_a), ("b", hclause_b), ("c", hclause_c), ("d", hclause_d)) if v]
        failed = [n for n, v in (("a", hclause_a), ("b", hclause_b), ("c", hclause_c), ("d", hclause_d)) if not v]
        print(f"\n  FALL-THROUGH -> NEGATIVE. Clauses cleared: {cleared or 'none'}. "
              f"Clauses failed: {failed or 'none'}.")

    print(f"\nconfigurations evaluated: {n_configs}")


if __name__ == "__main__":
    main()
