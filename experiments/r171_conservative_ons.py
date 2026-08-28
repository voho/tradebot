#!/usr/bin/env python
"""R-171 CONSERVATIVE branch: literal single-scalar Online Newton Step (ONS)
sizing for kelly_regime_v4's SCALE factor. Direction, citations, the
non-duplication argument, the eps/beta provenance, and the pre-registered
promotion rule/kill switches all live in `experiments/r171_shared.py`'s
module docstring and in `docs_scratch_direction.md` at the repo root (read
those first -- this file does not repeat that reasoning and does not edit
either frozen document).

THE MECHANISM, exactly (design doc S3 "Conservative"): a SINGLE shared ONS
accumulator over the whole series being scored. `scale_ONS = b_t`, from
`experiments.r171_shared.ons_scale(frac, asset_simple_return(df),
V4_MAX_LEVERAGE, eps, beta, b0=ONS_B0)`, replaces v4's own
`min(target_vol/vol, max_leverage)` OUTRIGHT. `desired = frac_t * scale_ONS`,
then v4's own unmodified 10% deadband (`apply_deadband`). `frac` (the vote)
is v4's own unmodified `v4_vote_frac` -- byte-identical to v4, per R-62's
isolation discipline, which this file touches nowhere.

eps/beta are NOT swept for the primary result (design doc: "not swept,
since the whole point of the conservative arm is the paper's literal,
parameter-light construction") -- `ONS_EPS_BTC`/`ONS_BETA_BTC` from
r171_shared.py are used as-is. The ONLY sweep in this file is the
pre-registered PLATEAU check (design doc S4 item 3): a 3-point bracket of
`eps` (0.5x, 1x, 2x `ONS_EPS_BTC`) at fixed `beta=ONS_BETA_BTC` -- a
structural robustness check on the regularization floor, not a performance
search; the result is reported for all 3 points, not cherry-picked.

Run: `. .venv/bin/activate && python experiments/r171_conservative_ons.py`
(from the repo root).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import r171_shared as shared  # noqa: E402

from tradebot.data import load_dataset  # noqa: E402
from tradebot.inference import daily_returns as _daily_returns  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# ================================================================== (1)
# Pre-registered sweep: 3 configs total (the eps plateau bracket). eps/beta
# themselves are not fitted -- see r171_shared.py's module docstring for
# the derivation of ONS_EPS_BTC/ONS_BETA_BTC from Hazan OCO Thm 4.5.
# ==================================================================
BETA = shared.ONS_BETA_BTC  # fixed across the bracket -- only eps is swept (design doc S4 item 3)
EPS_BRACKET = (0.5 * shared.ONS_EPS_BTC, 1.0 * shared.ONS_EPS_BTC, 2.0 * shared.ONS_EPS_BTC)
PRIMARY_EPS = shared.ONS_EPS_BTC
FEE_TIER_040 = 0.0040  # design doc S4 item 4 / scripts/fee_study.py's own Bitstamp taker tier

CONFIGS: list[dict] = [
    dict(eps=EPS_BRACKET[0], beta=BETA, kind="plateau_low (0.5x eps)"),
    dict(eps=EPS_BRACKET[1], beta=BETA, kind="PRIMARY (1x eps)"),
    dict(eps=EPS_BRACKET[2], beta=BETA, kind="plateau_high (2x eps)"),
]
assert len(CONFIGS) == 3, len(CONFIGS)


def config_label(cfg: dict) -> str:
    return f"r171_cons_ons_eps{cfg['eps']:.4g}_beta{cfg['beta']:.4g}"


# ================================================================== (2)
# Candidate: v4's own unmodified vote, scaled by a single-accumulator ONS
# b_t path in place of v4's conditional target-vol scale, then v4's own
# unmodified 10% deadband. This is the ONLY thing this file changes vs v4.
# ==================================================================

def ons_b_path(df: pd.DataFrame, eps: float, beta: float) -> np.ndarray:
    """The learned b_t path alone (pre-vote, pre-deadband) -- used by the
    kill switches and the plateau diagnostic, which inspect b_t itself."""
    frac = shared.v4_vote_frac(df).to_numpy()
    ret = shared.asset_simple_return(df)
    return shared.ons_scale(frac, ret, shared.V4_MAX_LEVERAGE, eps, beta, b0=shared.ONS_B0)


def make_build(eps: float, beta: float):
    """Candidate `build_target(df) -> np.ndarray`, matching compare()'s
    expected `candidate_build` signature."""

    def build(df: pd.DataFrame) -> np.ndarray:
        frac = shared.v4_vote_frac(df).to_numpy()
        ret = shared.asset_simple_return(df)
        b = shared.ons_scale(frac, ret, shared.V4_MAX_LEVERAGE, eps, beta, b0=shared.ONS_B0)
        desired = frac * b
        return shared.apply_deadband(desired)

    build.__name__ = f"ons_eps{eps:.4g}_beta{beta:.4g}"
    return build


BUILD_PRIMARY = make_build(PRIMARY_EPS, BETA)


# ================================================================== (3)
# Holdout helpers (compare()/run_slice() refuse any bar >= OOS_START by
# design -- this section is the only place in this file that reads one,
# and only if the pre-registered gate below clears).
# ==================================================================

def _load_full_btc() -> pd.DataFrame:
    df, _label = load_dataset(ROOT / "data", "spot")
    return df


def holdout_cell(build_fn, name: str, df_full: pd.DataFrame, market, start: str) -> dict:
    strat = shared.TargetStrategy(build_fn, name=name)
    res = run_period(strat, df_full, start=start, market=market, start_balance=1_000.0)
    m = compute_metrics(res)
    daily = _daily_returns(res.equity).to_numpy()
    exposure = res.df["target"].to_numpy() if "target" in res.df.columns else np.array([np.nan])
    return dict(name=name, sharpe=m.sharpe, max_dd=m.max_drawdown_pct,
                final_balance=m.final_balance, daily=daily,
                mean_abs_exposure=float(np.nanmean(np.abs(exposure))),
                realized_vol=float(np.nanstd(daily) * np.sqrt(365.25)) if len(daily) > 1 else float("nan"))


def holdout_compare(build_fn, label: str, df_full: pd.DataFrame, market, start: str) -> dict:
    cand = holdout_cell(build_fn, f"r171_{label}", df_full, market, start)
    ctrl = holdout_cell(shared.v4_target, "kelly_regime_v4", df_full, market, start)
    pr = shared.paired_diff(cand["daily"], ctrl["daily"])
    exp_ratio = (cand["mean_abs_exposure"] / ctrl["mean_abs_exposure"]
                 if ctrl["mean_abs_exposure"] else float("nan"))
    vol_ratio = (cand["realized_vol"] / ctrl["realized_vol"]
                 if ctrl["realized_vol"] else float("nan"))
    risk_matched = bool(shared.EXPOSURE_MATCH_BAND[0] <= exp_ratio <= shared.EXPOSURE_MATCH_BAND[1]
                         and shared.EXPOSURE_MATCH_BAND[0] <= vol_ratio <= shared.EXPOSURE_MATCH_BAND[1]) \
        if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False
    return dict(label=label, market=market.name,
                cand_sharpe=cand["sharpe"], ctrl_sharpe=ctrl["sharpe"],
                d_sharpe=cand["sharpe"] - ctrl["sharpe"],
                cand_dd=cand["max_dd"], ctrl_dd=ctrl["max_dd"],
                d_dd=cand["max_dd"] - ctrl["max_dd"],
                cand_final=cand["final_balance"], ctrl_final=ctrl["final_balance"],
                exposure_ratio=exp_ratio, vol_ratio=vol_ratio, risk_matched=risk_matched,
                boot_d_loggrowth=pr.diff.point, boot_lo=pr.diff.lo, boot_hi=pr.diff.hi,
                excludes_zero=bool(pr.diff.lo > 0 or pr.diff.hi < 0))


def print_holdout_row(r: dict) -> None:
    print(f"    {r['label']:28s} {r['market']:11s} "
          f"cand$={r['cand_final']:>12,.0f} ctrl$={r['ctrl_final']:>12,.0f} "
          f"dSharpe={r['d_sharpe']:+.3f} dDD={r['d_dd']:+.2f} "
          f"expR={r['exposure_ratio']:.2f} volR={r['vol_ratio']:.2f} "
          f"RM={'Y' if r['risk_matched'] else 'n'} "
          f"dlogG={r['boot_d_loggrowth']:+.4f} [{r['boot_lo']:+.4f},{r['boot_hi']:+.4f}] "
          f"excl0={'YES' if r['excludes_zero'] else 'no'}")


# ================================================================== (4)
# Reporting helpers over compare()'s row dicts.
# ==================================================================

def cell(rows: list[dict], label: str, slice_name: str, market: str) -> dict | None:
    for r in rows:
        if r["label"] == label and r["slice"] == slice_name and r["market"] == market:
            return r
    return None


def hr(title: str = "") -> None:
    print("\n" + "=" * 90)
    if title:
        print(title)
        print("=" * 90)


def main() -> None:
    t0 = time.time()
    configs_evaluated = 0
    backtest_cells = 0
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-171 CONSERVATIVE -- single-scalar Online Newton Step (ONS) leverage "
       "replacing kelly_regime_v4's\nconditional target-vol SCALE. See "
       "r171_shared.py / docs_scratch_direction.md S3 for the full mechanism.")
    print(f"\nCONFIGS ({len(CONFIGS)} total, eps plateau bracket at fixed beta={BETA}):")
    for cfg in CONFIGS:
        print(f"    {config_label(cfg):40s} eps={cfg['eps']:.4g}  kind={cfg['kind']}")

    # ========================================================== STEP 0
    hr("STEP 0 -- load data (pre-holdout truncated), causal truncation probe "
       "on the FULL strategy build (PRIMARY config)")
    btc = shared.load_btc()
    eth = shared.load_eth()
    max_ts_seen += [btc.index.max(), eth.index.max()]
    shared.assert_no_holdout(btc, "BTC full")
    shared.assert_no_holdout(eth, "ETH full")
    print(f"BTC (truncated < {shared.OOS_START}): {len(btc):,} bars, {btc.index[0]} -> {btc.index[-1]}")
    print(f"ETH (truncated < {shared.OOS_START}): {len(eth):,} bars, {eth.index[0]} -> {eth.index[-1]}")
    print("NOTE: ETH (Bitfinex) physically ends 2019-12-31 -- entirely BEFORE "
          f"INNER_VAL_START ({shared.INNER_VAL_START}). It cannot supply a literal "
          "'ETH inner-validation' window. Per this ledger's own established "
          "convention (r102/r147/r161-shared's compare(), used unmodified by "
          "every prior conservative/novel branch on this slot), the ETH "
          "falsification/replication check below uses ETH's FULL available "
          "history (compare()'s 'eth_replication' slice) as the cross-asset "
          "test -- disclosed explicitly here, not silently substituted.")

    btc_train = btc.loc[shared.INNER_TRAIN_START:shared.INNER_TRAIN_END]
    shared.assert_no_holdout(btc_train, "BTC inner-train")

    def _build_probe(d: pd.DataFrame) -> np.ndarray:
        return make_build(PRIMARY_EPS, BETA)(d)

    causal_ok = True
    try:
        ok = shared.causal_truncation_probe_series(_build_probe, btc_train)
        print(f"\ncausal_truncation_probe_series(full ONS strategy build, PRIMARY "
              f"eps={PRIMARY_EPS}, BTC inner-train): {'PASS' if ok else 'FAIL'}")
    except AssertionError as e:
        ok = False
        print(f"\ncausal_truncation_probe_series: FAIL ({e})")
    causal_ok = causal_ok and ok

    if not causal_ok:
        hr("VERDICT")
        print("    VERDICT: NEGATIVE (causal truncation probe FAILED end-to-end "
              "through the strategy wiring -- stopping before any promotion-bar "
              "evaluation, per the same convention as every prior round on this "
              "slot).")
        print(f"    Configurations evaluated: {configs_evaluated} (stopped before the sweep).")
        print("    The holdout (>= 2023-01-01) was NOT read.")
        return

    # ========================================================== STEP 1
    hr("STEP 1 -- Kill switches KS-a / KS-c (BTC inner-train, PRIMARY config "
       f"only: eps={PRIMARY_EPS}, beta={BETA})")
    b_ons_train = ons_b_path(btc_train, PRIMARY_EPS, BETA)
    b_incumbent_train = shared.v4_scale(btc_train)  # v4's own full/steady conditional-target-vol path,
                                                      # same bars, same hysteresis machine -- KS-c's "b_incumbent"

    corner_frac = shared.corner_lockin_fraction(b_ons_train, shared.V4_MAX_LEVERAGE)
    r2_artifact = shared.exposure_artifact_r2(b_ons_train, b_incumbent_train)

    ks_a_tripped = corner_frac > shared.CORNER_LOCKIN_THRESH
    ks_c_tripped = r2_artifact > shared.R2_KILL_THRESH

    print(f"\nKS-a (corner lock-in): fraction of BTC inner-train bars with b_t pinned "
          f"at {{0, {shared.V4_MAX_LEVERAGE}}} = {corner_frac:.4f}  "
          f"(threshold: > {shared.CORNER_LOCKIN_THRESH})")
    print(f"    KS-a TRIPPED: {ks_a_tripped}")
    print(f"\nKS-c (exposure-collapse artifact): R^2(b_ons, v4_scale) on BTC inner-train "
          f"= {r2_artifact:.4f}  (threshold: > {shared.R2_KILL_THRESH})")
    print(f"    b_incumbent = v4_scale(btc_train) -- v3/v4's own conditional "
          f"target-vol full/steady hysteresis path, computed directly via "
          f"r171_shared's v4_scale re-export on the SAME bars as b_ons.")
    print(f"    KS-c TRIPPED: {ks_c_tripped}")
    print(f"\nb_t summary (PRIMARY, BTC inner-train): mean={np.mean(b_ons_train):.4f} "
          f"median={np.median(b_ons_train):.4f} std={np.std(b_ons_train):.4f} "
          f"min={np.min(b_ons_train):.4f} max={np.max(b_ons_train):.4f}")

    if ks_a_tripped or ks_c_tripped:
        hr("VERDICT")
        tripped = []
        if ks_a_tripped:
            tripped.append(f"KS-a (corner lock-in={corner_frac:.4f} > {shared.CORNER_LOCKIN_THRESH})")
        if ks_c_tripped:
            tripped.append(f"KS-c (exposure-artifact R^2={r2_artifact:.4f} > {shared.R2_KILL_THRESH})")
        print(f"    VERDICT: NEGATIVE. Kill switch(es) tripped on BTC inner-train, "
              f"PRIMARY config: {', '.join(tripped)}.")
        print("    Per design doc S2(4) / this ledger's standing convention, a "
              "tripped kill switch stops evaluation here -- the main sweep, "
              "falsification test, plateau check and holdout are NOT run.")
        print(f"    Configurations evaluated: {configs_evaluated} (stopped before the sweep; "
              "only the PRIMARY config's b_t path was inspected, via the kill-switch "
              "diagnostics above -- not a backtest cell).")
        print("    The holdout (>= 2023-01-01) was NOT read.")
        return

    print("\nNeither kill switch tripped on BTC inner-train (PRIMARY config) -- "
          "proceeding to the main sweep.")

    # ========================================================== STEP 2
    hr("STEP 2 -- main sweep: full compare() per eps-bracket config "
       "(inner_train / inner_val / eth_replication x SPOT/FUTURES_5x)")
    all_rows: dict[str, list[dict]] = {}
    for cfg in CONFIGS:
        label = config_label(cfg)
        build = make_build(cfg["eps"], cfg["beta"])
        rows = shared.compare(build, label=label, btc=btc, eth=eth)
        configs_evaluated += 1
        backtest_cells += len(rows)
        all_rows[label] = rows
        print(f"\n  -- {label} ({cfg['kind']}) --")
        shared.print_rows(rows)

    primary_label = config_label(CONFIGS[1])
    primary_rows = all_rows[primary_label]

    # ========================================================== STEP 3
    hr("STEP 3 -- Falsification test (design doc S3): sign(dSharpe BTC "
       "inner-val) vs sign(dSharpe ETH replication), PRIMARY config")
    falsification_notes = []
    falsification_ok = True
    for market_name in (shared.SPOT.name, shared.FUTURES.name):
        btc_row = cell(primary_rows, primary_label, "inner_val", market_name)
        eth_row = cell(primary_rows, primary_label, shared.ETH_SLICE_NAME, market_name)
        btc_sign = np.sign(btc_row["d_sharpe"])
        eth_sign = np.sign(eth_row["d_sharpe"])
        note = (f"market={market_name:11s} BTC inner_val d_sharpe={btc_row['d_sharpe']:+.4f} "
                f"(sign {btc_sign:+.0f})   ETH replication d_sharpe={eth_row['d_sharpe']:+.4f} "
                f"(sign {eth_sign:+.0f})")
        print(f"    {note}")
        falsification_notes.append(note)
        if btc_sign != 0 and eth_sign != 0 and btc_sign != eth_sign:
            falsification_ok = False
    print(f"\nFALSIFICATION TEST (same sign, BTC vs ETH, on every market -- design doc's "
          f"exact kill outcome): {'PASS' if falsification_ok else 'FAIL -- KILLED'}")

    # ========================================================== STEP 4
    hr("STEP 4 -- Plateau check (design doc S4 item 3): eps bracket "
       f"{[round(e, 4) for e in EPS_BRACKET]} at fixed beta={BETA}")
    print("\n(4a) b_t path sensitivity, BTC inner-train:")
    b_paths = {}
    for cfg in CONFIGS:
        b_paths[cfg["eps"]] = ons_b_path(btc_train, cfg["eps"], cfg["beta"])
        print(f"    eps={cfg['eps']:.4g}  mean={np.mean(b_paths[cfg['eps']]):.4f} "
              f"std={np.std(b_paths[cfg['eps']]):.4f} "
              f"corner_lockin={shared.corner_lockin_fraction(b_paths[cfg['eps']], shared.V4_MAX_LEVERAGE):.4f}")
    lo_b, hi_b = b_paths[EPS_BRACKET[0]], b_paths[EPS_BRACKET[2]]
    mean_abs_diff = float(np.mean(np.abs(lo_b - hi_b)))
    r2_lo_hi = shared.r_squared(lo_b, hi_b)
    print(f"    b_t(eps=0.5x) vs b_t(eps=2x): mean|diff|={mean_abs_diff:.4f}  R^2={r2_lo_hi:.4f}")

    print("\n(4b) dSharpe across the eps bracket, per slice/market (no cherry-picking -- "
          "all 3 reported):")
    plateau_signs = set()
    for cfg in CONFIGS:
        label = config_label(cfg)
        for slice_name in ("inner_val", shared.ETH_SLICE_NAME):
            for market_name in (shared.SPOT.name, shared.FUTURES.name):
                row = cell(all_rows[label], label, slice_name, market_name)
                print(f"    eps={cfg['eps']:.4g}  {slice_name:16s} {market_name:11s} "
                      f"d_sharpe={row['d_sharpe']:+.4f}")
                if slice_name == "inner_val" and market_name == shared.FUTURES.name:
                    plateau_signs.add(np.sign(row["d_sharpe"]))
    plateau_ok = len(plateau_signs) <= 1
    print(f"\nPLATEAU CHECK (BTC inner_val, futures_5x, sign of dSharpe consistent "
          f"across the eps bracket): {'PASS' if plateau_ok else 'FAIL -- sign flips within the bracket'}")

    # ========================================================== STEP 5
    hr("STEP 5 -- Promotion-rule clauses 1-3 (design doc S4), inner-validation, "
       "PRIMARY config")
    val_spot = cell(primary_rows, primary_label, "inner_val", shared.SPOT.name)
    val_fut = cell(primary_rows, primary_label, "inner_val", shared.FUTURES.name)
    eth_spot = cell(primary_rows, primary_label, shared.ETH_SLICE_NAME, shared.SPOT.name)
    eth_fut = cell(primary_rows, primary_label, shared.ETH_SLICE_NAME, shared.FUTURES.name)

    def clause1_sharpe(row_btc, row_eth) -> bool:
        return row_btc["d_sharpe"] >= shared.SHARPE_DELTA_PROMOTE and \
            row_eth["d_sharpe"] >= shared.SHARPE_DELTA_PROMOTE

    def clause1_dd(row_btc, row_eth) -> bool:
        return (row_btc["risk_matched"] and row_btc["d_dd"] <= -shared.DD_REDUCTION_PROMOTE_PP
                and row_eth["risk_matched"] and row_eth["d_dd"] <= -shared.DD_REDUCTION_PROMOTE_PP)

    clause1_spot = clause1_sharpe(val_spot, eth_spot) or clause1_dd(val_spot, eth_spot)
    clause1_fut = clause1_sharpe(val_fut, eth_fut) or clause1_dd(val_fut, eth_fut)
    print(f"\nClause 1 (dSharpe>=+{shared.SHARPE_DELTA_PROMOTE} both BTC&ETH, OR matched-exposure "
          f"DD reduction>={shared.DD_REDUCTION_PROMOTE_PP}pp both):")
    print(f"    spot:       BTC dSharpe={val_spot['d_sharpe']:+.4f}  ETH dSharpe={eth_spot['d_sharpe']:+.4f}  "
          f"BTC dDD={val_spot['d_dd']:+.2f} (RM={val_spot['risk_matched']})  "
          f"ETH dDD={eth_spot['d_dd']:+.2f} (RM={eth_spot['risk_matched']})  -> {clause1_spot}")
    print(f"    futures_5x: BTC dSharpe={val_fut['d_sharpe']:+.4f}  ETH dSharpe={eth_fut['d_sharpe']:+.4f}  "
          f"BTC dDD={val_fut['d_dd']:+.2f} (RM={val_fut['risk_matched']})  "
          f"ETH dDD={eth_fut['d_dd']:+.2f} (RM={eth_fut['risk_matched']})  -> {clause1_fut}")
    clause1_ok = clause1_spot or clause1_fut
    print(f"    Clause 1 satisfied on >=1 market: {clause1_ok}")
    print(f"\nClause 2 (survives falsification test): {falsification_ok}")
    print(f"Clause 3 (plateau, not peak): {plateau_ok}")
    print("Clause 4 (0.40% taker fee sign check) is evaluated at the holdout step below, "
          "per this file's own convention -- see STEP 7.")

    gate_ok = (not ks_a_tripped) and (not ks_c_tripped) and falsification_ok
    print(f"\nGATE TO HOLDOUT (no kill switch tripped AND falsification survives -- "
          f"clause 1's threshold is NOT required to advance, per the operator's "
          f"instruction to default to running the holdout unless a kill switch or "
          f"falsification already answered the question): {gate_ok}")

    print(f"\nConfigurations evaluated so far: {configs_evaluated} "
          f"({backtest_cells} total compare() cells: {len(CONFIGS)} configs x 6 cells each).")

    if not gate_ok:
        hr("VERDICT")
        print("    VERDICT: NEGATIVE (falsification test failed -- sign inverts "
              "between BTC and ETH on at least one market). Per design doc S3's "
              "exact kill outcome, the branch is killed regardless of the BTC number.")
        print(f"    Configurations evaluated: {configs_evaluated} ({backtest_cells} backtest cells).")
        print("    The holdout (>= 2023-01-01) was NOT read.")
        return

    # ========================================================== STEP 6
    hr("STEP 6 -- HOLDOUT (pre-registered, design doc S4; run exactly once, "
       f"start={shared.OOS_START}, PRIMARY config only)")
    print("Running the holdout because no kill switch tripped and the falsification "
          "test survived (STEP 3), per the operator's default-to-run instruction. "
          "This is the ONE holdout evaluation for this branch.")

    btc_full = _load_full_btc()
    max_ts_seen.append(btc_full.index.max())
    print(f"\nBTC full (untruncated): {len(btc_full):,} bars, {btc_full.index[0]} -> {btc_full.index[-1]}")
    print("ETH (Bitfinex) has NO bars at or after OOS_START (physically ends "
          "2019-12-31) -- the holdout-period falsification check is N/A for ETH, "
          "disclosed here rather than silently skipped.")

    holdout_rows = []
    for market in (shared.SPOT, shared.FUTURES):
        r = holdout_compare(BUILD_PRIMARY, "conservative_ons_primary", btc_full, market, shared.OOS_START)
        holdout_rows.append(r)
        print_holdout_row(r)

    # ---- clause 4: 0.40% taker fee sign check (futures_5x, holdout) -------
    hr("STEP 7 -- Clause 4: 0.40% taker fee sign check (holdout, futures_5x)")
    fee_market_std = shared.FUTURES
    fee_market_040 = shared.fee_at(shared.FUTURES, FEE_TIER_040)
    r_std = next(r for r in holdout_rows if r["market"] == shared.FUTURES.name)
    r_040 = holdout_compare(BUILD_PRIMARY, "conservative_ons_primary_fee040",
                             btc_full, fee_market_040, shared.OOS_START)
    print(f"    futures_5x @ standard fee ({fee_market_std.fee_rate:.4%}): dSharpe={r_std['d_sharpe']:+.4f}")
    print_holdout_row(r_std)
    print(f"    futures_5x @ {FEE_TIER_040:.2%} taker: dSharpe={r_040['d_sharpe']:+.4f}")
    print_holdout_row(r_040)
    sign_std = np.sign(r_std["d_sharpe"])
    sign_040 = np.sign(r_040["d_sharpe"])
    clause4_ok = bool(sign_std == 0 or sign_040 == 0 or sign_std == sign_040)
    print(f"\nClause 4 (sign does not reverse at 0.40% taker): {clause4_ok}")

    configs_evaluated += 1  # the fee-tier holdout cell is one more evaluated configuration
    backtest_cells += 6     # 2 holdout markets (candidate+control each) + 1 fee-tier cell pair, approx accounting below

    # ========================================================== STEP 8
    hr("STEP 8 -- Final promotion-rule check (design doc S4, ALL FOUR clauses)")
    holdout_spot = next(r for r in holdout_rows if r["market"] == shared.SPOT.name)
    holdout_fut = next(r for r in holdout_rows if r["market"] == shared.FUTURES.name)

    holdout_clause1_sharpe = (holdout_spot["d_sharpe"] >= shared.SHARPE_DELTA_PROMOTE
                               and holdout_fut["d_sharpe"] >= shared.SHARPE_DELTA_PROMOTE)
    holdout_clause1_dd = (holdout_spot["risk_matched"] and holdout_spot["d_dd"] <= -shared.DD_REDUCTION_PROMOTE_PP
                           and holdout_fut["risk_matched"] and holdout_fut["d_dd"] <= -shared.DD_REDUCTION_PROMOTE_PP)
    holdout_clause1_ok = holdout_clause1_sharpe or holdout_clause1_dd
    print(f"\nHoldout clause 1 (dSharpe>=+{shared.SHARPE_DELTA_PROMOTE} both markets OR matched-exposure "
          f"DD reduction>={shared.DD_REDUCTION_PROMOTE_PP}pp both): {holdout_clause1_ok}")
    print(f"    spot:       dSharpe={holdout_spot['d_sharpe']:+.4f}  dDD={holdout_spot['d_dd']:+.2f}  "
          f"RM={holdout_spot['risk_matched']}")
    print(f"    futures_5x: dSharpe={holdout_fut['d_sharpe']:+.4f}  dDD={holdout_fut['d_dd']:+.2f}  "
          f"RM={holdout_fut['risk_matched']}")
    print(f"\nClause 2 (falsification, inner-validation -- ETH holdout N/A, no data): {falsification_ok}")
    print(f"Clause 3 (plateau): {plateau_ok}")
    print(f"Clause 4 (0.40% fee sign check, holdout): {clause4_ok}")

    all_four = holdout_clause1_ok and falsification_ok and plateau_ok and clause4_ok
    print(f"\nALL FOUR CLAUSES MET: {all_four}")
    print("Note: design doc S4's clause 1 is written for inner-validation; this "
          "file additionally reports the identical clause computed on the holdout "
          "(the decisive window) rather than substituting the inner-validation "
          "number for it. If these two disagree, that is a fall-through the "
          "pre-registered rule does not resolve, and is reported as such below, "
          "not rounded to whichever looks better.")

    hr("VERDICT")
    if all_four:
        print("    VERDICT: PROMOTE-CANDIDATE -- all four design-doc S4 clauses are "
              "met on the holdout/inner-validation evidence above.")
    else:
        failed = []
        if not holdout_clause1_ok:
            failed.append("clause 1 (holdout Sharpe/DD bar)")
        if not falsification_ok:
            failed.append("clause 2 (falsification)")
        if not plateau_ok:
            failed.append("clause 3 (plateau)")
        if not clause4_ok:
            failed.append("clause 4 (0.40% fee sign)")
        print(f"    VERDICT: NEGATIVE. Clause(s) not met: {', '.join(failed)}.")
        print("    Per design doc S4: 'Any other outcome is NEGATIVE, including "
              "partial passes' -- a partial pass is reported as a partial pass, "
              "not rounded up.")

    print(f"\nConfigurations evaluated (total): {configs_evaluated} "
          f"({len(CONFIGS)} eps-bracket configs x 6 compare() cells = {len(CONFIGS) * 6} inner cells, "
          f"plus the holdout PRIMARY run at 2 markets x [candidate,control] "
          f"and the 0.40%-fee holdout cell x [candidate,control]).")
    print(f"\nMax timestamp read anywhere in this run: {max(max_ts_seen)}")
    print(f"Total wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
