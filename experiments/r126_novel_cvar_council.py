"""R-126 NOVEL branch: CVaR-budgeted convex reallocation across
`champions_council`'s six members, replacing its Hedge/multiplicative-
weights allocation.

Frozen mechanism (see `r126_shared.py`'s module docstring for the full
pre-registration -- this file only implements it):

  Every `REBALANCE_DAYS` (30) calendar days, using only the trailing
  `LOOKBACK_DAYS` (90) days of member daily payoffs strictly BEFORE the
  rebalance day, solve for the simplex weight vector `w` minimizing the
  portfolio's Conditional Value-at-Risk at `CVAR_ALPHA` (0.05), subject to
  a minimum expected daily payoff floor (`mu_floor` = trailing cross-member
  median of per-member mean daily payoff), via Rockafellar & Uryasev
  (2000)'s LP characterization:

      CVaR_alpha(w) = min_zeta { zeta + 1/((1-alpha)*T) * sum_t max(0, L_t(w) - zeta) }
      L_t(w) = -(payoff_window[t] @ w)

  solved by dependency-free projected subgradient descent on (w, zeta),
  with the floor enforced via a Lagrange multiplier `lam` found by
  bisection in [0, 50]. Hold the weight fixed until the next rebalance
  (forward-fill). Before `LOOKBACK_DAYS` of history exists, fall back to
  equal weights (1/6).

Run this file directly to execute the full pre-registered battery
(Step-0 gate, causal-truncation self-test, B1/B3/B4/B5, decision rule,
holdout only if authorized) and print every number.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r126_shared as sh  # noqa: E402

# ----------------------------------------------------------------------
# Execution-performance memoization only -- NOT part of the mechanism.
# `member_signal_matrix` (~50s on the full BTC frame, called by both
# `fit_novel_council` and `r126_shared.council_reference_target`) and the
# engine backtests inside `b1_signal` (~25-30s each) are pure, deterministic
# functions of (dataset, config), so a resumed run of this script (e.g.
# split across several shell invocations because the full battery exceeds
# one command's timeout) can skip stages already computed rather than
# recomputing member_signal_matrix from scratch every time. Disable with
# R126_NOVEL_NO_CACHE=1.
# ----------------------------------------------------------------------
CACHE_DIR = Path(os.environ.get("R126_NOVEL_CACHE_DIR",
                                 Path(tempfile.gettempdir()) / "r126_novel_cache"))
_CACHE_DISABLED = os.environ.get("R126_NOVEL_NO_CACHE") == "1"


def cached(key: str, fn):
    if _CACHE_DISABLED:
        return fn()
    path = CACHE_DIR / f"{key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            print(f"  [cache hit] {key}", flush=True)
            return pickle.load(f)
    result = fn()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(result, f)
    tmp.replace(path)
    return result

# ----------------------------------------------------------------------
# The convex program: dependency-free projected subgradient descent on
# (w, zeta), Lagrangian relaxation of the return floor via bisection on
# lam. Pure numpy, seeded fixed (no randomness used, but the structure is
# fully deterministic anyway -- noted for the causal-truncation probe).
# ----------------------------------------------------------------------

def project_simplex(v: np.ndarray) -> np.ndarray:
    """Duchi, Shalev-Shwartz, Singer & Chandra (2008), projection onto the
    probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - css / ind > 0
    rho = ind[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0)


def _inner_solve(payoff_window: np.ndarray, alpha: float, lam: float,
                  lr: float = 0.05, iters: int = 800) -> tuple[np.ndarray, float]:
    """One Lagrangian-relaxed solve of the CVaR LP for a fixed multiplier
    `lam` on the return-floor constraint. `payoff_window` is a plain
    (T, N) numpy array (no pandas inside the hot loop)."""
    T, N = payoff_window.shape
    w = np.ones(N) / N
    zeta = 0.0
    mean_payoff = payoff_window.mean(axis=0)
    for _ in range(iters):
        port = payoff_window @ w
        loss = -port
        active = (loss - zeta) > 0  # subgradient support set
        if active.any():
            grad_w_cvar = -(1.0 / ((1 - alpha) * T)) * payoff_window[active].sum(axis=0)
        else:
            grad_w_cvar = np.zeros(N)
        grad_w = grad_w_cvar - lam * mean_payoff
        grad_zeta = 1.0 - active.sum() / ((1 - alpha) * T)
        w = project_simplex(w - lr * grad_w)
        zeta = zeta - lr * grad_zeta
    return w, zeta


def solve_cvar_weights(payoff_window: np.ndarray, alpha: float, mu_floor: float,
                        lr: float = 0.05, iters: int = 800,
                        lam_hi: float = 50.0, bisect_steps: int = 20,
                        tol: float = 1e-4) -> dict:
    """Returns a dict with the solved weight vector and solver diagnostics:
    `w`, `zeta`, `lam`, `bound` (bool: did the floor constraint bind),
    `mean_payoff_achieved`, `floor_met` (bool, within `tol`).
    """
    w0, zeta0 = _inner_solve(payoff_window, alpha, lam=0.0, lr=lr, iters=iters)
    mean0 = float((payoff_window @ w0).mean())

    if mean0 >= mu_floor - tol:
        # unconstrained CVaR optimum already clears the floor -- lam=0 suffices
        return {"w": w0, "zeta": zeta0, "lam": 0.0, "bound": False,
                "mean_payoff_achieved": mean0, "floor_met": True}

    # bisect lam in [0, lam_hi] for the smallest lam whose resulting w's
    # mean trailing payoff >= mu_floor
    lo, hi = 0.0, lam_hi
    w_hi, zeta_hi = _inner_solve(payoff_window, alpha, lam=hi, lr=lr, iters=iters)
    mean_hi = float((payoff_window @ w_hi).mean())
    # if even lam_hi cannot clear the floor, return the best achievable
    if mean_hi < mu_floor - tol:
        return {"w": w_hi, "zeta": zeta_hi, "lam": hi, "bound": True,
                "mean_payoff_achieved": mean_hi, "floor_met": False}

    w_mid, zeta_mid, mean_mid = w_hi, zeta_hi, mean_hi
    for _ in range(bisect_steps):
        mid = 0.5 * (lo + hi)
        w_mid, zeta_mid = _inner_solve(payoff_window, alpha, lam=mid, lr=lr, iters=iters)
        mean_mid = float((payoff_window @ w_mid).mean())
        if mean_mid >= mu_floor - tol:
            hi = mid
        else:
            lo = mid
    # final solve at hi (guaranteed to clear the floor, smallest such lam found)
    w_final, zeta_final = _inner_solve(payoff_window, alpha, lam=hi, lr=lr, iters=iters)
    mean_final = float((payoff_window @ w_final).mean())
    return {"w": w_final, "zeta": zeta_final, "lam": hi,
            "bound": True, "mean_payoff_achieved": mean_final,
            "floor_met": bool(mean_final >= mu_floor - tol)}


# ----------------------------------------------------------------------
# Weight schedule construction: rebalance every REBALANCE_DAYS calendar
# days, causal (only payoff rows strictly before the rebalance day),
# forward-filled between rebalances. Equal weights until LOOKBACK_DAYS of
# history exists.
# ----------------------------------------------------------------------

def build_weight_schedule(payoff: pd.DataFrame, lookback_days: int, alpha: float,
                           rebalance_days: int = None) -> tuple[pd.DataFrame, list[dict]]:
    if rebalance_days is None:
        rebalance_days = sh.REBALANCE_DAYS
    names = sh.member_names()
    n_members = len(names)
    days = payoff.index
    payoff_arr = payoff.to_numpy()  # (n_days, n_members) plain numpy, sliced by position
    n_days = len(days)

    rebalance_positions = list(range(0, n_days, rebalance_days))
    equal_w = np.ones(n_members) / n_members

    sched_rows = []
    sched_index = []
    diagnostics = []

    for pos in rebalance_positions:
        day = days[pos]
        sched_index.append(day)
        if pos < lookback_days:
            sched_rows.append(equal_w.copy())
            diagnostics.append({"day": day, "fallback": True})
            continue
        window = payoff_arr[pos - lookback_days:pos, :]  # strictly before `day`
        mu_floor = float(np.median(window.mean(axis=0)))
        result = solve_cvar_weights(window, alpha, mu_floor)
        w = result["w"]
        sched_rows.append(w)
        wp = w[w > 1e-12]
        entropy = float(-np.sum(wp * np.log(wp))) if len(wp) else 0.0
        diagnostics.append({
            "day": day, "fallback": False,
            "lam": result["lam"], "bound": result["bound"],
            "floor_met": result["floor_met"],
            "mu_floor": mu_floor, "mean_payoff_achieved": result["mean_payoff_achieved"],
            "max_weight": float(w.max()), "entropy": entropy,
        })

    schedule = pd.DataFrame(sched_rows, index=pd.DatetimeIndex(sched_index, tz=days.tz),
                             columns=names)
    return schedule, diagnostics


def fit_novel_council(df: pd.DataFrame, lookback_days: int = sh.LOOKBACK_DAYS,
                       alpha: float = sh.CVAR_ALPHA, a: np.ndarray | None = None,
                       payoff: pd.DataFrame | None = None) -> dict:
    """Full pipeline: signal matrix -> payoffs -> weight schedule -> target.

    `a` (member signal matrix) and `payoff` (member daily payoffs) do not
    depend on `lookback_days`/`alpha`, so a caller sweeping the B3 grid on
    a fixed `df` may pass them in precomputed rather than paying
    `member_signal_matrix`'s ~seconds-per-member cost on every grid cell.
    """
    if a is None:
        a = sh.member_signal_matrix(df)
    if payoff is None:
        payoff = sh.member_daily_payoffs(df, a)
    schedule, diagnostics = build_weight_schedule(payoff, lookback_days, alpha)
    target = sh.weights_to_target(df, a, schedule)
    return {"a": a, "payoff": payoff, "weight_schedule": schedule,
            "diagnostics": diagnostics, "target": target}


# ----------------------------------------------------------------------
# Solver diagnostics summary (step 4)
# ----------------------------------------------------------------------

def summarize_diagnostics(diagnostics: list[dict]) -> dict:
    fit = [d for d in diagnostics if not d["fallback"]]
    n_fallback = len(diagnostics) - len(fit)
    if not fit:
        return {"n_fit": 0, "n_fallback": n_fallback}
    n_bound = sum(1 for d in fit if d["bound"])
    n_floor_met = sum(1 for d in fit if d["floor_met"])
    max_weights = np.array([d["max_weight"] for d in fit])
    entropies = np.array([d["entropy"] for d in fit])
    return {
        "n_fit": len(fit), "n_fallback": n_fallback,
        "n_bound_lam_gt_0": n_bound, "frac_bound": n_bound / len(fit),
        "n_floor_met": n_floor_met, "frac_floor_met": n_floor_met / len(fit),
        "max_weight_mean": float(max_weights.mean()), "max_weight_p90": float(np.percentile(max_weights, 90)),
        "max_weight_max": float(max_weights.max()),
        "entropy_mean": float(entropies.mean()), "entropy_min": float(entropies.min()),
        "log6": float(np.log(6)),
    }


def check_lam_monotonicity(payoff: pd.DataFrame, lookback_days: int, alpha: float,
                            n_probe_points: int = 8) -> dict:
    """Independently verify (per step 3's instruction): does increasing
    `lam` monotonically increase the achieved mean trailing payoff? Probed
    at a handful of real rebalance windows drawn from the actual data."""
    payoff_arr = payoff.to_numpy()
    n_days = len(payoff_arr)
    positions = [p for p in range(lookback_days, n_days, sh.REBALANCE_DAYS)]
    if len(positions) > n_probe_points:
        step = len(positions) // n_probe_points
        positions = positions[::step][:n_probe_points]
    lam_grid = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    all_monotone = True
    violations = []
    for pos in positions:
        window = payoff_arr[pos - lookback_days:pos, :]
        means = []
        for lam in lam_grid:
            w, _ = _inner_solve(window, alpha, lam=lam)
            means.append(float((window @ w).mean()))
        means = np.array(means)
        diffs = np.diff(means)
        if np.any(diffs < -1e-9):
            all_monotone = False
            violations.append({"day_pos": pos, "means": means.tolist()})
    return {"n_probed": len(positions), "all_monotone": all_monotone,
            "violations": violations}


# ----------------------------------------------------------------------
# Own causal-truncation self-test (step 9)
# ----------------------------------------------------------------------

def causal_truncation_probe(df: pd.DataFrame, cut: int = 400_000,
                             fit_full: dict | None = None) -> bool:
    if fit_full is None:
        fit_full = fit_novel_council(df)
    df_trunc = df.iloc[:cut].copy()
    fit_trunc = fit_novel_council(df_trunc)

    sched_full, sched_trunc = fit_full["weight_schedule"], fit_trunc["weight_schedule"]
    common_days = sched_trunc.index[sched_trunc.index.isin(sched_full.index)]
    # drop the last rebalance point: it may be within `LOOKBACK_DAYS` of the
    # truncated frame's end and its own trailing window could differ in bar
    # count if the cut lands mid-day (partial-day effects), matching the
    # buffer r126_shared's own self-test uses on member_daily_payoffs.
    common_days = common_days[:-1] if len(common_days) > 1 else common_days
    sched_ok = np.allclose(sched_full.loc[common_days].to_numpy(),
                            sched_trunc.loc[common_days].to_numpy(), atol=1e-10)

    n_check = len(df_trunc) - 10 * sh.BARS_PER_DAY  # buffer for the last rebalance day's bars
    n_check = max(n_check, 0)
    target_full = fit_full["target"][:n_check]
    target_trunc = fit_trunc["target"][:n_check]
    target_ok = np.allclose(target_full, target_trunc, atol=1e-9)

    print(f"  weight_schedule match (excl. last point): {sched_ok}", flush=True)
    print(f"  target match (first {n_check} bars, buffer=10 days): {target_ok}", flush=True)
    return bool(sched_ok and target_ok)


# ----------------------------------------------------------------------
# B3 plateau grid
# ----------------------------------------------------------------------

def run_b3(df: pd.DataFrame, market, a: np.ndarray, payoff: pd.DataFrame) -> list[dict]:
    """`a`/`payoff` are precomputed once on `df` (they don't depend on the
    (alpha, lookback) grid cell) and reused across all 6 cells."""
    rows = []
    for alpha in (0.05, 0.10):
        for lookback in (60, 90, 120):
            fit = fit_novel_council(df, lookback_days=lookback, alpha=alpha, a=a, payoff=payoff)
            b1 = sh.b1_signal(fit["target"], df, market)
            rows.append({"alpha": alpha, "lookback": lookback, "d_sharpe": b1["d_sharpe"]})
    return rows


# ----------------------------------------------------------------------
# Main: run the full pre-registered battery
# ----------------------------------------------------------------------

def main() -> None:
    n_configs = 0
    t0 = time.time()

    print("=" * 70, flush=True)
    print("R-126 NOVEL branch: CVaR-budgeted council reallocation", flush=True)
    print(f"cache dir: {CACHE_DIR} (disabled={_CACHE_DISABLED})", flush=True)
    print("=" * 70, flush=True)

    print("\n[load] BTC train (spot bars, up to INNER_VAL_END)", flush=True)
    df_btc, label_btc = sh.load_btc_train("spot")
    print(f"  loaded {len(df_btc)} bars, label={label_btc}", flush=True)

    # ---------------- Step-0 gate ----------------
    print("\n[Step-0] fit primary config (alpha=0.05, lookback=90) on full BTC train", flush=True)
    fit_primary = cached("btc_primary_fit", lambda: fit_novel_council(df_btc))
    n_configs += 1
    diag_summary = summarize_diagnostics(fit_primary["diagnostics"])
    print(f"  solver diagnostics: {diag_summary}", flush=True)

    council_ref = cached("btc_council_ref_target", lambda: sh.council_reference_target(df_btc))
    gate = sh.step0_gate(fit_primary["target"], council_ref)
    print(f"  Step-0 gate: R^2 vs champions_council Hedge blend = {gate['r2_vs_council']:.6f}, "
          f"kill={gate['kill']}", flush=True)

    if gate["kill"]:
        print("\nStep-0 KILL: candidate's target is numerically indistinguishable "
              "(R^2 > 0.98) from champions_council's own Hedge blend. STOPPING per "
              "pre-registered protocol -- this is the round's result.", flush=True)
        print(f"\nTotal configs evaluated: {n_configs}", flush=True)
        print("\nVERDICT: NEGATIVE (Step-0 kill)", flush=True)
        return

    # ---------------- lam monotonicity check ----------------
    print("\n[monotonicity] probing lam -> mean-payoff monotonicity on real windows", flush=True)
    mono = cached("mono_check", lambda: check_lam_monotonicity(
        fit_primary["payoff"], sh.LOOKBACK_DAYS, sh.CVAR_ALPHA))
    print(f"  {mono}", flush=True)

    # ---------------- causal truncation self-test ----------------
    print("\n[causal-truncation probe] own self-test (primary config)", flush=True)
    probe_ok = cached("causal_probe_ok", lambda: causal_truncation_probe(df_btc, fit_full=fit_primary))
    print(f"  causal truncation probe: {'PASS' if probe_ok else 'FAIL'}", flush=True)
    if not probe_ok:
        print("  FAIL -- refusing to read any inner-validation number. STOPPING.", flush=True)
        print(f"\nTotal configs evaluated: {n_configs}", flush=True)
        print("\nVERDICT: NEGATIVE (causal-truncation probe FAIL)", flush=True)
        return

    # ---------------- B1: BTC spot + futures ----------------
    print("\n[B1] primary config vs champions_council, inner-validation, BTC", flush=True)
    b1_spot = cached("b1_spot", lambda: sh.b1_signal(fit_primary["target"], df_btc, sh.SPOT))
    n_configs += 1
    print(f"  SPOT: {b1_spot}", flush=True)
    b1_fut = cached("b1_fut", lambda: sh.b1_signal(fit_primary["target"], df_btc, sh.FUTURES))
    n_configs += 1
    print(f"  FUTURES: {b1_fut}", flush=True)

    noise_floor = 0.2
    b1_spot_pass = (b1_spot["d_sharpe"] > noise_floor) or b1_spot["significant"] or \
        (b1_spot["dd_cand"] < b1_spot["dd_council"] - 1.0)
    b1_fut_pass = (b1_fut["d_sharpe"] > noise_floor) or b1_fut["significant"] or \
        (b1_fut["dd_cand"] < b1_fut["dd_council"] - 1.0)
    b1_pass = b1_spot_pass and b1_fut_pass
    print(f"  B1 pass (both markets, d_sharpe>{noise_floor} OR significant paired-bootstrap "
          f"OR clear drawdown win): spot={b1_spot_pass}, futures={b1_fut_pass}, overall={b1_pass}",
          flush=True)

    # ---------------- B3: plateau grid ----------------
    print("\n[B3] plateau grid: CVAR_ALPHA x LOOKBACK_DAYS, BTC spot inner-validation", flush=True)
    b3_rows = cached("b3_rows", lambda: run_b3(df_btc, sh.SPOT, fit_primary["a"], fit_primary["payoff"]))
    n_configs += len(b3_rows)
    primary_sign = np.sign(b1_spot["d_sharpe"])
    same_signed = sum(1 for r in b3_rows if np.sign(r["d_sharpe"]) == primary_sign and primary_sign != 0)
    for r in b3_rows:
        print(f"  alpha={r['alpha']:.2f} lookback={r['lookback']:3d}  d_sharpe={r['d_sharpe']:+.4f}"
              f"  same_sign_as_primary={np.sign(r['d_sharpe']) == primary_sign}", flush=True)
    b3_pass = same_signed >= 4  # majority of 6
    print(f"  same-signed cells: {same_signed}/6 (primary sign="
          f"{'+' if primary_sign>0 else '-' if primary_sign<0 else '0'})", flush=True)
    print(f"  B3 pass (majority same-signed plateau): {b3_pass}", flush=True)

    # ---------------- B4: ETH falsification ----------------
    print("\n[B4] ETH falsification, primary config only (alpha=0.05, lookback=90), spot", flush=True)
    df_eth = sh.load_eth_train()
    fit_eth = cached("eth_primary_fit", lambda: fit_novel_council(df_eth))
    n_configs += 1
    b1_eth = cached("b1_eth", lambda: sh.b1_signal(fit_eth["target"], df_eth, sh.SPOT))
    n_configs += 1
    print(f"  ETH SPOT: {b1_eth}", flush=True)
    eth_sign = np.sign(b1_eth["d_sharpe"])
    btc_sign = np.sign(b1_spot["d_sharpe"])
    b4_pass = bool(eth_sign == btc_sign and btc_sign != 0)
    print(f"  BTC primary-cell sign (spot) = {'+' if btc_sign>0 else '-' if btc_sign<0 else '0'}, "
          f"ETH sign = {'+' if eth_sign>0 else '-' if eth_sign<0 else '0'}", flush=True)
    print(f"  B4 pass (ETH replicates BTC sign): {b4_pass}", flush=True)

    # ---------------- B5: fee tier ----------------
    print("\n[B5] fee tier robustness, primary config, BTC", flush=True)
    b5_spot_hi = cached("b5_spot_hi", lambda: sh.b1_signal(fit_primary["target"], df_btc, sh.SPOT_HIGH_FEE))
    n_configs += 1
    b5_fut_hi = cached("b5_fut_hi", lambda: sh.b1_signal(fit_primary["target"], df_btc, sh.FUTURES_HIGH_FEE))
    n_configs += 1
    print(f"  SPOT high-fee (0.40%): d_sharpe={b5_spot_hi['d_sharpe']:+.4f} "
          f"(0.10% tier was {b1_spot['d_sharpe']:+.4f})", flush=True)
    print(f"  FUTURES high-fee (0.40%): d_sharpe={b5_fut_hi['d_sharpe']:+.4f} "
          f"(0.10% tier was {b1_fut['d_sharpe']:+.4f})", flush=True)
    b5_spot_flip = np.sign(b5_spot_hi["d_sharpe"]) != np.sign(b1_spot["d_sharpe"])
    b5_fut_flip = np.sign(b5_fut_hi["d_sharpe"]) != np.sign(b1_fut["d_sharpe"])
    b5_pass = not (b5_spot_flip or b5_fut_flip)
    print(f"  B5 pass (no sign flip): spot_flip={b5_spot_flip}, futures_flip={b5_fut_flip}, "
          f"overall_pass={b5_pass}", flush=True)

    # ---------------- Decision rule ----------------
    print("\n" + "=" * 70, flush=True)
    print("DECISION RULE (pre-registered in r126_shared.py)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Step-0 not killed: {not gate['kill']}", flush=True)
    print(f"  B1 pass (both markets): {b1_pass}", flush=True)
    print(f"  B3 pass (plateau majority): {b3_pass}", flush=True)
    print(f"  B4 pass (ETH replicates): {b4_pass}", flush=True)
    print(f"  B5 pass (no fee-tier flip): {b5_pass}", flush=True)

    authorized = (not gate["kill"]) and b1_pass and b3_pass and b4_pass and b5_pass
    print(f"\n  HOLDOUT READ AUTHORIZED: {authorized}", flush=True)

    if authorized:
        print("\n[HOLDOUT] all clauses passed -- reading OOS per pre-registered rule", flush=True)
        from tradebot.data import load_dataset as _load_dataset

        df_btc_full, _ = _load_dataset(sh.ROOT / "data", "spot")
        # Fit the mechanism exactly as in Step-0/B1 (causal weight schedule
        # built from payoffs), but now let weights_to_target build the
        # target series over the FULL frame (including holdout bars) so
        # run_target_series has holdout-period target values to trade --
        # the fit itself never looks past a rebalance day's own history.
        fit_primary_full = cached("btc_primary_fit_full_incl_holdout",
                                   lambda: fit_novel_council(df_btc_full))
        n_configs += 1

        def _ho_spot():
            m_cand, _ = sh.run_target_series(fit_primary_full["target"], df_btc_full,
                                              sh.SPOT, sh.OOS_START, None, label="")
            m_council, _ = sh.run_candidate_council(df_btc_full, sh.SPOT,
                                                      start=sh.OOS_START, end=None)
            return m_cand, m_council

        m_cand_ho_spot, m_council_ho_spot = cached("holdout_spot", _ho_spot)
        n_configs += 1
        print(f"  HOLDOUT SPOT (candidate):          sharpe={m_cand_ho_spot.sharpe:.4f}, "
              f"max_dd={m_cand_ho_spot.max_drawdown_pct:.2f}%", flush=True)
        print(f"  HOLDOUT SPOT (champions_council):  sharpe={m_council_ho_spot.sharpe:.4f}, "
              f"max_dd={m_council_ho_spot.max_drawdown_pct:.2f}%", flush=True)
        print(f"  HOLDOUT SPOT d_sharpe = {m_cand_ho_spot.sharpe - m_council_ho_spot.sharpe:+.4f}",
              flush=True)

        def _ho_fut():
            m_cand, _ = sh.run_target_series(fit_primary_full["target"], df_btc_full,
                                              sh.FUTURES, sh.OOS_START, None, label="")
            m_council, _ = sh.run_candidate_council(df_btc_full, sh.FUTURES,
                                                      start=sh.OOS_START, end=None)
            return m_cand, m_council

        m_cand_ho_fut, m_council_ho_fut = cached("holdout_fut", _ho_fut)
        n_configs += 1
        print(f"  HOLDOUT FUTURES (candidate):         sharpe={m_cand_ho_fut.sharpe:.4f}, "
              f"max_dd={m_cand_ho_fut.max_drawdown_pct:.2f}%", flush=True)
        print(f"  HOLDOUT FUTURES (champions_council):  sharpe={m_council_ho_fut.sharpe:.4f}, "
              f"max_dd={m_council_ho_fut.max_drawdown_pct:.2f}%", flush=True)
        print(f"  HOLDOUT FUTURES d_sharpe = {m_cand_ho_fut.sharpe - m_council_ho_fut.sharpe:+.4f}",
              flush=True)
        print(f"\nTotal configs evaluated: {n_configs}", flush=True)
        print(f"Elapsed: {time.time() - t0:.1f}s", flush=True)
        print("\nVERDICT: PROMOTE-candidate (all pre-registered clauses passed; see holdout numbers above)",
              flush=True)
    else:
        failing = [name for name, ok in [
            ("Step-0", not gate["kill"]), ("B1", b1_pass), ("B3", b3_pass),
            ("B4", b4_pass), ("B5", b5_pass)] if not ok]
        print(f"\n  NEGATIVE: failing clause(s): {failing}", flush=True)
        print("  Holdout NOT read, per pre-registered decision rule.", flush=True)
        print(f"\nTotal configs evaluated: {n_configs}", flush=True)
        print(f"Elapsed: {time.time() - t0:.1f}s", flush=True)
        print(f"\nVERDICT: NEGATIVE (failing clause(s): {failing})", flush=True)


if __name__ == "__main__":
    main()
