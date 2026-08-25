#!/usr/bin/env python
"""R-126 CONSERVATIVE branch: replace ``champions_council``'s own Hedge /
multiplicative-weights allocation across its 6 members with a periodically-
rebalanced Equal-Risk-Contribution special case -- inverse trailing
volatility weighting (Maillard, Roncalli & Teiletche 2010, "The Properties
of Equally Weighted Risk Contribution Portfolios", J. Portfolio Management
36(4), 60-70) -- applied here, for the first time in this project, to a
panel of *strategy signals* rather than a panel of *assets* (R-107 used the
same paper on BCH/LTC/ETC/DASH/LINK/XTZ). Full literature grounding, the
non-duplication argument, the named failure mode and the pre-registered
decision rule / falsification test all live in ``experiments/r126_shared.py``'s
own module docstring (read in full before this file was written); not
re-derived here beyond the summary above. This file NEVER edits
``r126_shared.py`` (frozen, shared with the parallel NOVEL branch, a
disjoint file this session does not read or coordinate with), and never
reads a bar at or after ``r126_shared.OOS_START`` (2023-01-01) unless the
pre-registered decision rule (below) authorizes it after every other clause
has passed.

MECHANISM (exact, per the operator's brief):

1. ``a = r126_shared.member_signal_matrix(df)``,
   ``payoff = r126_shared.member_daily_payoffs(df, a)`` -- both frozen,
   shared functions, byte-identical to ``champions_council``'s own member
   construction.
2. ``fit_weights_erc(payoff, asof_day, lookback_days, eps_frac=0.05)``: using
   only ``payoff`` rows strictly BEFORE ``asof_day``, take the trailing
   ``lookback_days`` window, compute each member's sample std (ddof=1).
   Floor every std at ``eps_frac * median(std of members with positive
   std)`` (handles the ``flat`` member, whose payoff is identically 0).
   ``w_i = (1/std_i_floored) / sum_j(1/std_j_floored)``. If there is not
   yet ``lookback_days`` of history before ``asof_day``, return equal
   weights (1/6 each) -- a structural fallback, not a fitted one.
3. ``build_weight_schedule``: on a fixed rebalance calendar (every
   ``REBALANCE_DAYS`` calendar days, starting from the first day of the
   loaded frame), call ``fit_weights_erc`` using only data strictly before
   that rebalance day, then forward-fill the resulting weight vector over
   every calendar day until the next rebalance day. Index spans the full
   loaded frame's calendar days; columns are ``r126_shared.member_names()``.
4. ``target = r126_shared.weights_to_target(df, a, weight_schedule)`` --
   frozen, shared function; only the weight vector differs from
   ``champions_council``'s own Hedge blend.

CONFIGURATIONS EVALUATED: 1 (Step-0 gate, primary config) + 2 (B1: BTC spot
+ futures, inner-validation) + 12 (B3: 4 REBALANCE_DAYS x 3 LOOKBACK_DAYS,
BTC spot only) + 1 (B4: ETH spot, inner-validation, frozen primary config)
+ 2 (B5: BTC spot + futures at the 0.40% taker tier) = 18 total (+2 if the
decision rule authorizes a holdout read).

DECISION RULE (pre-registered, verbatim from ``r126_shared.py``, unaltered
after seeing any number): PROMOTE-candidate only if the causal-truncation
probe AND B1 (both markets) AND B3 (plateau majority) AND B4 (full, both
markets -- ETH spot only, no ETH futures data) AND B5 all pass. B2
(drawdown) is diagnostic only and never gates promotion by itself.

USAGE
-----
    python experiments/r126_conservative_erc_council.py
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

from experiments import r126_shared  # noqa: E402

BARS_PER_DAY = r126_shared.BARS_PER_DAY

# Primary configuration, pre-registered before any inner-validation number
# was read: matches r126_shared's own REBALANCE_DAYS / LOOKBACK_DAYS module
# defaults (30-day rebalance, 90-day fitting window).
PRIMARY_REBALANCE_DAYS = r126_shared.REBALANCE_DAYS   # 30
PRIMARY_LOOKBACK_DAYS = r126_shared.LOOKBACK_DAYS     # 90
EPS_FRAC = 0.05

REBALANCE_GRID = (14, 30, 60, 90)
LOOKBACK_GRID = (60, 90, 180)


# ================================================================== (1)
# fit_weights_erc: causal, inverse-trailing-vol ERC special case.
# ==================================================================

def fit_weights_erc(payoff: pd.DataFrame, asof_day, lookback_days: int = 90,
                     eps_frac: float = 0.05) -> pd.Series:
    """Equal-Risk-Contribution special case (Maillard, Roncalli & Teiletche
    2010, Section 2): ``w_i proportional to 1/std_i`` of member ``i``'s
    trailing daily payoff, using ONLY ``payoff`` rows strictly before
    ``asof_day`` (never ``asof_day`` itself -- causal). Every std is floored
    at ``eps_frac * median(std of members with positive std)`` before
    inverting, so a member with identically-zero payoff (``flat``) cannot
    divide by zero and cannot dominate the blend either. Falls back to
    equal weights (1/N each) -- a structural fallback, not a fitted one --
    when fewer than ``lookback_days`` rows of history exist before
    ``asof_day``.
    """
    names = payoff.columns
    n = len(names)
    equal = pd.Series(1.0 / n, index=names)

    hist = payoff.loc[payoff.index < asof_day]
    if len(hist) < lookback_days:
        return equal

    window = hist.iloc[-lookback_days:]
    stds = window.std(ddof=1)
    positive = stds[stds > 0]
    if len(positive) == 0:
        return equal

    floor = eps_frac * positive.median()
    stds_floored = stds.clip(lower=floor)
    inv = 1.0 / stds_floored
    w = inv / inv.sum()
    return w


# ================================================================== (2)
# weight_schedule construction: fixed rebalance calendar, causal fit at
# each rebalance day, forward-filled across every calendar day in between.
# ==================================================================

def build_weight_schedule(payoff: pd.DataFrame, rebalance_days: int = PRIMARY_REBALANCE_DAYS,
                           lookback_days: int = PRIMARY_LOOKBACK_DAYS,
                           eps_frac: float = EPS_FRAC) -> pd.DataFrame:
    names = payoff.columns
    first_day = payoff.index.min()
    last_day = payoff.index.max()
    full_index = pd.date_range(first_day, last_day, freq="1D", tz=payoff.index.tz)
    rebalance_dates = pd.date_range(first_day, last_day, freq=f"{rebalance_days}D",
                                     tz=payoff.index.tz)

    sched = pd.DataFrame(index=full_index, columns=names, dtype=float)
    for d in rebalance_dates:
        sched.loc[d] = fit_weights_erc(payoff, d, lookback_days=lookback_days,
                                        eps_frac=eps_frac).values
    # rebalance_dates[0] == first_day (rebalance calendar starts at the
    # frame's first day), so every row is covered by a forward-fill from
    # that first assigned row -- no backward-fill, no lookahead.
    sched = sched.ffill()
    assert not sched.isna().any().any(), "weight_schedule has unfilled rows"
    return sched


def build_target(df: pd.DataFrame, rebalance_days: int = PRIMARY_REBALANCE_DAYS,
                  lookback_days: int = PRIMARY_LOOKBACK_DAYS):
    """Full pipeline (steps 1-4): returns (target, weight_schedule, a, payoff)."""
    a = r126_shared.member_signal_matrix(df)
    payoff = r126_shared.member_daily_payoffs(df, a)
    weight_schedule = build_weight_schedule(payoff, rebalance_days=rebalance_days,
                                             lookback_days=lookback_days)
    target = r126_shared.weights_to_target(df, a, weight_schedule)
    return target, weight_schedule, a, payoff


# ================================================================== (3)
# Own causal-truncation self-test (this round's own new code, not the
# shared module's -- must PASS before any inner-validation number is read).
# ==================================================================

def causal_truncation_probe(df: pd.DataFrame, rebalance_days: int = PRIMARY_REBALANCE_DAYS,
                             lookback_days: int = PRIMARY_LOOKBACK_DAYS,
                             cut: int = 400_000) -> dict:
    target_full, sched_full, _, _ = build_target(df, rebalance_days, lookback_days)

    df_trunc = df.iloc[:cut].copy()
    target_trunc, sched_trunc, _, _ = build_target(df_trunc, rebalance_days, lookback_days)

    # weight_schedule: compare on days common to both, dropping the last
    # few days near the truncated frame's own end (partial-day / edge
    # effects there are expected, not a leak -- mirrors r126_shared.py's
    # own __main__ convention of dropping the last 2 days near its cut).
    common_days = sched_trunc.index[sched_trunc.index.isin(sched_full.index)]
    buffer_days = 3
    common_days = common_days[:-buffer_days] if len(common_days) > buffer_days else common_days
    sched_ok = bool(np.allclose(sched_full.loc[common_days].to_numpy(),
                                 sched_trunc.loc[common_days].to_numpy(), atol=1e-12))

    # target: compare bars strictly before the cut, minus a buffer for the
    # same partial-day effect at the truncated frame's own tail.
    buffer_bars = 2 * BARS_PER_DAY
    n_check = max(cut - buffer_bars, 0)
    target_ok = bool(np.allclose(target_full[:n_check], target_trunc[:n_check], atol=1e-9))

    return {"sched_ok": sched_ok, "target_ok": target_ok, "ok": sched_ok and target_ok,
            "n_check_bars": n_check, "n_check_days": len(common_days)}


# ================================================================== (3b)
# Holdout read: b1_signal-equivalent logic, but from OOS_START onward
# instead of inner-validation, and on the FULL (untruncated) dataset. Only
# ever called from inside the `if all_pass:` guard below -- the only path
# authorized by the pre-registered decision rule to read a bar >= OOS_START.
# ==================================================================

def holdout_signal(candidate_target: np.ndarray, df: pd.DataFrame, market) -> dict:
    m_cand, res_cand = r126_shared.run_target_series(
        candidate_target, df, market, r126_shared.OOS_START, None)
    m_council, res_council = r126_shared.run_candidate_council(
        df, market, start=r126_shared.OOS_START, end=None)
    r_cand = r126_shared.daily_returns(res_cand.equity)
    r_council = r126_shared.daily_returns(res_council.equity)
    n = min(len(r_cand), len(r_council))
    paired = r126_shared.paired_bootstrap(
        r_cand.to_numpy()[:n], r_council.to_numpy()[:n],
        stat=r126_shared.total_log_return, seed=126)
    return {
        "sharpe_cand": m_cand.sharpe, "sharpe_council": m_council.sharpe,
        "d_sharpe": m_cand.sharpe - m_council.sharpe,
        "paired_diff": paired.diff.point, "paired_lo": paired.diff.lo, "paired_hi": paired.diff.hi,
        "significant": paired.significant,
        "dd_cand": m_cand.max_drawdown_pct, "dd_council": m_council.max_drawdown_pct,
    }


# ================================================================== (4)
# Main: Step-0 -> causal probe -> B1 -> B3 -> B4 -> B5 -> verdict.
# ==================================================================

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []
    n_configs = 0

    print("=" * 78)
    print("R-126 CONSERVATIVE: ERC (inverse trailing volatility) council weighting --")
    print("champions_council's own Hedge allocation replaced, member set unchanged.")
    print("=" * 78)

    btc, _ = r126_shared.load_btc_train("spot")
    max_ts_seen.append(btc.index.max())
    print(f"\nBTC spot (truncated < {r126_shared.OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    # -------------------------------------------------------------- primary target
    print(f"\n-- BUILDING PRIMARY CONFIG (rebalance={PRIMARY_REBALANCE_DAYS}d, "
          f"lookback={PRIMARY_LOOKBACK_DAYS}d), BTC full train frame --")
    target_primary, sched_primary, a_btc, payoff_btc = build_target(
        btc, PRIMARY_REBALANCE_DAYS, PRIMARY_LOOKBACK_DAYS)
    n_rebalances = len(pd.date_range(payoff_btc.index.min(), payoff_btc.index.max(),
                                      freq=f"{PRIMARY_REBALANCE_DAYS}D"))
    print(f"  weight_schedule: {len(sched_primary)} calendar days, "
          f"{n_rebalances} rebalance points")
    print(f"  weight_schedule head:\n{sched_primary.head(3).to_string()}")
    print(f"  weight_schedule tail:\n{sched_primary.tail(3).to_string()}")

    # -------------------------------------------------------------- Step 0
    print("\n" + "=" * 78)
    print("STEP 0 -- sanity gate: is the candidate genuinely different from "
          "champions_council's own Hedge blend, or a rescaled copy?")
    print("=" * 78)
    council_target_full = r126_shared.council_reference_target(btc)
    step0 = r126_shared.step0_gate(target_primary, council_target_full)
    n_configs += 1
    print(f"  R^2 vs champions_council (BTC inner-train + inner-validation): "
          f"{step0['r2_vs_council']:.6f}")
    print(f"  KILL (R^2 > 0.98)?  {step0['kill']}")
    if step0["kill"]:
        print("\nSTEP-0 KILL: candidate is numerically a rescaled copy of "
              "champions_council's own Hedge blend. Reporting honestly rather than "
              "proceeding to claim a result. Not building additional configurations.")
        max_ts = max(max_ts_seen)
        print(f"\nconfigurations evaluated: {n_configs} (Step-0 only)")
        print(f"max timestamp read anywhere in this branch: {max_ts} "
              f"(< {r126_shared.OOS_START}: {max_ts < pd.Timestamp(r126_shared.OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(verdict="NEGATIVE (Step-0 kill)", step0=step0, n_configs=n_configs,
                    max_ts=max_ts)

    # -------------------------------------------------------------- causal probe
    print("\n" + "=" * 78)
    print("CAUSAL-TRUNCATION SELF-TEST (this round's own new code, real BTC data, cut=400,000)")
    print("=" * 78)
    probe = causal_truncation_probe(btc)
    print(f"  weight_schedule bit-identical (< cut, minus {3}-day buffer): "
          f"{'PASS' if probe['sched_ok'] else 'FAIL'} ({probe['n_check_days']} days checked)")
    print(f"  target bit-identical (< cut, minus {2 * BARS_PER_DAY}-bar buffer): "
          f"{'PASS' if probe['target_ok'] else 'FAIL'} ({probe['n_check_bars']:,} bars checked)")
    probe_ok = probe["ok"]
    print(f"  CAUSAL PROBE OVERALL: {'PASS' if probe_ok else 'FAIL'}")
    if not probe_ok:
        print("\nCAUSAL PROBE FAILURE -- a result that looks too good is a bug report "
              "first. Stopping before any inner-validation number is trusted.")
        max_ts = max(max_ts_seen)
        print(f"\nconfigurations evaluated: {n_configs} (Step-0 only; promotion bar not run)")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(verdict="NEGATIVE (causal probe failure)", step0=step0, probe=probe,
                    n_configs=n_configs, max_ts=max_ts)

    # -------------------------------------------------------------- B1
    print("\n" + "=" * 78)
    print("B1 -- BTC signal, inner-validation, spot + futures")
    print("=" * 78)
    b1_spot = r126_shared.b1_signal(target_primary, btc, r126_shared.SPOT)
    b1_fut = r126_shared.b1_signal(target_primary, btc, r126_shared.FUTURES)
    n_configs += 2
    for name, r in (("spot", b1_spot), ("futures", b1_fut)):
        print(f"  {name:>8s}  sharpe_cand={r['sharpe_cand']:+.4f}  "
              f"sharpe_council={r['sharpe_council']:+.4f}  d_sharpe={r['d_sharpe']:+.4f}  "
              f"boot=[{r['paired_lo']:+.4f},{r['paired_hi']:+.4f}]  "
              f"significant={r['significant']}  dd_cand={r['dd_cand']:.2f}%  "
              f"dd_council={r['dd_council']:.2f}%")
    b1_pass = (b1_spot["d_sharpe"] > 0.2 or b1_spot["paired_lo"] > 0.0) and \
              (b1_fut["d_sharpe"] > 0.2 or b1_fut["paired_lo"] > 0.0)
    print(f"  B1 PASS (both markets, d_sharpe > +0.2 noise floor OR bootstrap excludes "
          f"zero positively): {b1_pass}")

    # -------------------------------------------------------------- B3
    print("\n" + "=" * 78)
    print(f"B3 -- plateau: REBALANCE_DAYS in {REBALANCE_GRID} x LOOKBACK_DAYS in "
          f"{LOOKBACK_GRID}, BTC spot only")
    print("=" * 78)
    grid_rows = []
    primary_sign = float(np.sign(b1_spot["d_sharpe"]))
    for reb in REBALANCE_GRID:
        for lb in LOOKBACK_GRID:
            if reb == PRIMARY_REBALANCE_DAYS and lb == PRIMARY_LOOKBACK_DAYS:
                r = b1_spot  # reuse the primary cell's own number, not a re-fit
            else:
                tgt, _, _, _ = build_target(btc, reb, lb)
                r = r126_shared.b1_signal(tgt, btc, r126_shared.SPOT)
            n_configs += 1
            sign = float(np.sign(r["d_sharpe"]))
            grid_rows.append(dict(rebalance_days=reb, lookback_days=lb,
                                   d_sharpe=r["d_sharpe"], boot_lo=r["paired_lo"],
                                   boot_hi=r["paired_hi"], sign=sign))
            tag = "  <- PRIMARY" if (reb, lb) == (PRIMARY_REBALANCE_DAYS, PRIMARY_LOOKBACK_DAYS) else ""
            print(f"  rebalance={reb:>3d}d  lookback={lb:>3d}d  d_sharpe={r['d_sharpe']:+.4f}  "
                  f"boot=[{r['paired_lo']:+.4f},{r['paired_hi']:+.4f}]{tag}")
    n_same = sum(1 for row in grid_rows if row["sign"] == primary_sign)
    b3_pass = n_same >= len(grid_rows) / 2.0
    print(f"  B3 (majority same-signed as primary, spot): {b3_pass} ({n_same}/{len(grid_rows)})")

    # -------------------------------------------------------------- B4
    print("\n" + "=" * 78)
    print("B4 -- ETH falsification (pre-registered), spot only (no ETH futures data), "
          "PRIMARY config only")
    print("=" * 78)
    eth = r126_shared.load_eth_train()
    max_ts_seen.append(eth.index.max())
    print(f"ETH spot (truncated < {r126_shared.OOS_START}): {len(eth):,} bars, "
          f"{eth.index[0]} -> {eth.index[-1]}")
    target_eth, sched_eth, a_eth, payoff_eth = build_target(
        eth, PRIMARY_REBALANCE_DAYS, PRIMARY_LOOKBACK_DAYS)
    b4_spot = r126_shared.b1_signal(target_eth, eth, r126_shared.SPOT)
    n_configs += 1
    print(f"  spot  ETH d_sharpe={b4_spot['d_sharpe']:+.4f}  "
          f"boot=[{b4_spot['paired_lo']:+.4f},{b4_spot['paired_hi']:+.4f}]  "
          f"significant={b4_spot['significant']}")
    btc_spot_sign = float(np.sign(b1_spot["d_sharpe"]))
    eth_spot_sign = float(np.sign(b4_spot["d_sharpe"]))
    b4_full_pass = bool(btc_spot_sign != 0 and eth_spot_sign == btc_spot_sign)
    print(f"  BTC spot d_sharpe sign = {btc_spot_sign:+.0f}   ETH spot d_sharpe sign = "
          f"{eth_spot_sign:+.0f}   SAME SIGN (B4 full pass, spot-only since no ETH futures "
          f"data exists): {b4_full_pass}")

    # -------------------------------------------------------------- B5
    print("\n" + "=" * 78)
    print("B5 -- fee-tier survival (0.40% taker), primary config, BTC spot + futures")
    print("=" * 78)
    b5_spot = r126_shared.b1_signal(target_primary, btc, r126_shared.SPOT_HIGH_FEE)
    b5_fut = r126_shared.b1_signal(target_primary, btc, r126_shared.FUTURES_HIGH_FEE)
    n_configs += 2
    spot_no_flip = np.sign(b5_spot["d_sharpe"]) == np.sign(b1_spot["d_sharpe"]) or b1_spot["d_sharpe"] == 0
    fut_no_flip = np.sign(b5_fut["d_sharpe"]) == np.sign(b1_fut["d_sharpe"]) or b1_fut["d_sharpe"] == 0
    for name, r0, r1, ok in (("spot", b1_spot, b5_spot, spot_no_flip),
                              ("futures", b1_fut, b5_fut, fut_no_flip)):
        print(f"  {name:>8s}  @0.10% d_sharpe={r0['d_sharpe']:+.4f}   "
              f"@0.40% d_sharpe={r1['d_sharpe']:+.4f}   no_flip={ok}")
    b5_pass = bool(spot_no_flip and fut_no_flip)
    print(f"  B5 PASS (no sign flip, either market): {b5_pass}")

    # -------------------------------------------------------------- decision rule
    print("\n" + "=" * 78)
    print("DECISION RULE (pre-registered, r126_shared.py)")
    print("=" * 78)
    all_pass = probe_ok and b1_pass and b3_pass and b4_full_pass and b5_pass
    print(f"causal probe={probe_ok}  B1={b1_pass}  B2=diagnostic-only  B3={b3_pass}  "
          f"B4(full)={b4_full_pass}  B5={b5_pass}")
    print(f"ALL GATING CLAUSES PASS: {all_pass}")

    verdict = "PROMOTE-candidate" if all_pass else "NEGATIVE"
    if not all_pass:
        failed = [name for name, ok in (("causal probe", probe_ok), ("B1", b1_pass),
                                         ("B3", b3_pass), ("B4 (full)", b4_full_pass),
                                         ("B5", b5_pass)) if not ok]
        print(f"VERDICT: {verdict}  --  failing clause(s): {', '.join(failed)}")
    else:
        print(f"VERDICT: {verdict}")

    # ---------------------------------------------------------- holdout (gated)
    holdout_spot = None
    holdout_fut = None
    if all_pass:
        print("\n" + "=" * 78)
        print("ALL GATING CLAUSES PASS -- decision rule authorizes a holdout read.")
        print(f"HOLDOUT (>= {r126_shared.OOS_START}) -- PRIMARY config, BTC spot + futures, "
              "full (untruncated) dataset")
        print("=" * 78)
        btc_full, btc_full_label = r126_shared.load_dataset(r126_shared.ROOT / "data", "spot")
        max_ts_seen.append(btc_full.index.max())
        print(f"  full BTC frame ({btc_full_label}): {len(btc_full):,} bars, "
              f"{btc_full.index[0]} -> {btc_full.index[-1]}")
        target_full_series, _, _, _ = build_target(
            btc_full, PRIMARY_REBALANCE_DAYS, PRIMARY_LOOKBACK_DAYS)
        holdout_spot = holdout_signal(target_full_series, btc_full, r126_shared.SPOT)
        holdout_fut = holdout_signal(target_full_series, btc_full, r126_shared.FUTURES)
        n_configs += 2
        for name, r in (("spot", holdout_spot), ("futures", holdout_fut)):
            print(f"  {name:>8s}  sharpe_cand={r['sharpe_cand']:+.4f}  "
                  f"sharpe_council={r['sharpe_council']:+.4f}  d_sharpe={r['d_sharpe']:+.4f}  "
                  f"boot=[{r['paired_lo']:+.4f},{r['paired_hi']:+.4f}]  "
                  f"significant={r['significant']}  dd_cand={r['dd_cand']:.2f}%  "
                  f"dd_council={r['dd_council']:.2f}%")

    max_ts = max(max_ts_seen)
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(1 Step-0 + 2 B1 + {len(grid_rows)} B3 + 1 B4 + 2 B5"
          f"{' + 2 holdout' if all_pass else ''})")
    print(f"max timestamp read anywhere in this branch: {max_ts}"
          + ("" if not all_pass else " (holdout read was authorized this round)"))
    if not all_pass:
        print(f"(< {r126_shared.OOS_START}: "
              f"{max_ts < pd.Timestamp(r126_shared.OOS_START, tz='UTC')}) -- "
              "NO bar at or after 2023-01-01 was read by this file.")
    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(
        verdict=verdict, n_configs=n_configs, max_ts=max_ts,
        step0=step0, probe=probe,
        b1_spot=b1_spot, b1_fut=b1_fut, b1_pass=b1_pass,
        b3_grid=grid_rows, b3_pass=b3_pass,
        b4_spot=b4_spot, b4_full_pass=b4_full_pass,
        b5_spot=b5_spot, b5_fut=b5_fut, b5_pass=b5_pass,
        holdout_spot=holdout_spot, holdout_fut=holdout_fut,
    )


if __name__ == "__main__":
    main()
