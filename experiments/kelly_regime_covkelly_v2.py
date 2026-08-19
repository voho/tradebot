#!/usr/bin/env python
"""B-16, novel half: de-noised covariance-aware Kelly allocation between the
same two unchanged `kelly_regime_v4` sub-books (BTC, ETH) as
`kelly_regime_covkelly.py` (the R-42 "novel" branch this file extends), with
the raw trailing-mean term that R-42 diagnosed as the failure mechanism
replaced or shrunk, per that branch's own author's prescribed fix (recorded
verbatim in the ledger's B-16 backlog entry).

R-42's finding, restated precisely (this is the bug this file exists to
fix): the original allocator solves the classical two-asset Kelly weight
`raw = Sigma^-1 mu` from causal trailing EWM estimates of each asset's own
RAW mean return. Sigma was fine; mu was not. Through the 2022 bear the
allocator concentrated ~97% of its shrunken stake into ETH on a stale,
slow-decaying sample mean, even though ETH did not outperform BTC through
that bear -- monthly-cadence validation Sharpe went negative (-0.16) while
an otherwise-identical weekly-cadence re-run of the same hyperparameters
flipped to modestly beating the baselines. That flip is the
cadence-driven train/validation inconsistency this file re-tests for every
candidate below.

Why the mean is the problem, not the covariance (citations)
-------------------------------------------------------------
- Chopra, V.K. & Ziemba, W.T. (1993), "The Effect of Errors in Means,
  Variances, and Covariances on Optimal Portfolio Choice", J. Portfolio
  Management: estimation error in the mean vector has roughly an order of
  magnitude more impact on realized portfolio performance than an
  equal-sized error in the covariance matrix. This is exactly consistent
  with what R-42 found by inspection -- a noisy `mu_btc - mu_eth`
  difference dominating a Sigma that was, by the exposure-artifact
  diagnostic, actually fine (R^2 0.005-0.60 against a flat-rescaled
  BTC-only book).
- Rising, J.K. & Wyner, A.J. (Wharton working paper; also circulated on
  arXiv/academia.edu), "Partial Kelly Portfolios and Shrinkage
  Estimators": a fractional-Kelly bet is formally a shrinkage estimator of
  the mean -- betting a fraction f of the full-Kelly stake is equivalent,
  in its effect on realized growth, to shrinking the plug-in mean estimate
  toward zero by (1-f) before solving the same optimization. This is the
  direct motivation for the `lambda` shrinkage-intensity knob below: it
  makes that equivalence a literal, sweepable parameter instead of an
  implicit byproduct of `kelly_frac`.
- MacLean, Thorp & Ziemba (eds.) (2011), "The Kelly Capital Growth
  Investment Criterion", World Scientific -- the multivariate Sigma^-1 mu
  form and its fragility to mean-estimation error (cited already in the
  file this extends).
- Ledoit, O. & Wolf, M. (2003/2004, various venues), shrinkage estimation
  of the covariance matrix toward a structured target -- the standard
  reference for stabilizing Sigma itself. NOT implemented here: this round
  is scoped to the mean side, which is where R-42 localized the fault and
  where Chopra/Ziemba says the leverage is an order of magnitude larger;
  Sigma-shrinkage is a natural next round if this one is promising and a
  further-improvement pass is wanted, not attempted this session.

Three weight-estimation treatments, each a `weight_mode`
-----------------------------------------------------------
1. `"raw"` -- UNCHANGED control. Literally the same closed-form
   `Sigma^-1 mu` as `kelly_regime_covkelly.py`, reimplemented here (not
   imported) so the weight-computation function is genuinely rewritten
   per this task's own instruction, then numerically cross-checked
   bit-for-bit against the original file's `build_weight_series` --
   see `verify_raw_matches_v1()`. Rerun in full here (not cited from the
   old file's report) so it is a true apples-to-apples control on this
   file's own code path.
2. `"equal_sharpe"` -- the noisy own-asset mean `mu_i` is replaced by a
   PRIOR mean `mu_i_prior = s_hat * sigma_i`: each asset's own
   (well-estimated) trailing vol, times a single POOLED Sharpe ratio
   shared by both assets. This removes exactly the `mu_btc - mu_eth`
   difference term R-42 diagnosed as the failure mechanism -- the two
   assets' prior means now differ only through their (better-behaved)
   vol ratio, not through two independently noisy sample means -- while
   still letting the (accurately estimated) vol difference and the
   covariance drive the allocation.

   Precise causal construction of `s_hat` (stated exactly, as required):
   let `mu_btc, mu_eth, sigma_btc, sigma_eth` be the SAME
   `.ewm(halflife=...).shift(1)` trailing estimates used everywhere else
   in this file (so already causal -- see `build_stats`). Define, at
   every row T (a deterministic, lookahead-free elementwise function of
   already-shifted columns, so introduces no additional leakage):

       s_hat_T = 0.5 * ( mu_btc_T / sigma_btc_T  +  mu_eth_T / sigma_eth_T )

   i.e. the pooled Sharpe is the simple average of each asset's own
   trailing Sharpe ratio, not a Sharpe computed on the two return series
   pooled into one longer series (which would let BTC's larger early
   history dominate a dimensionless quantity that should not care about
   which asset happened to have more history). Then
   `mu_i_prior = s_hat * sigma_i` for each asset separately.

3. `"sigma_only"` -- mean-free, minimum-variance-style allocation:
   `mu` is dropped from the formula ENTIRELY (not set to zero inside
   `Sigma^-1 mu` -- that would just collapse both weights to zero and
   move everything to cash, which is a different and strictly worse
   thing than diversifying by risk). Instead: `raw propto Sigma^-1 @
   [1, 1]` (the classical unconstrained minimum-variance weight
   direction), clipped to no-shorting and normalized to sum to an
   exposure budget `kelly_frac * total_cap` (so `kelly_frac` here means
   "what fraction of capital to deploy", not "what fraction of the
   Kelly-optimal edge to take" -- a genuinely different role than in
   modes 1-2, documented precisely at `_solve_minvar`). This is the
   purest form of "give up on estimating which asset will do better, and
   just diversify by risk" -- per MacLean/Thorp/Ziemba's own warning
   about Kelly's fragility to mean estimation error, and Chopra/Ziemba's
   order-of-magnitude finding above.

4. `"blend"` -- convex shrinkage-intensity knob `lam in [0, 1]` between
   the RAW sample mean and whichever of (2)/(3) early testing (this
   file's own `run_phase1`) finds more promising; see
   `verify_lambda_boundaries` for the numerical sanity check that
   `lam=0` exactly reproduces `"raw"` (hence the already-measured R-42
   novel branch, up to the cross-check in point 1 above) and `lam=1`
   exactly reproduces the chosen target mode. Two blend mechanics are
   implemented depending on the chosen target, because target (2) is
   itself a "mean" (so blending happens on `mu` before a single
   `Sigma^-1 mu` solve) while target (3) is not (so blending happens on
   the final, already-capped WEIGHT vectors of the two extremes -- a
   convex combination of two feasible points under linear leg-cap /
   total-cap constraints is itself feasible, so no re-capping is
   needed and the `lam=0`/`lam=1` boundary equality is exact). See
   `build_weight_series_v2` for both code paths.

Hard rules honored (same as the file this extends)
------------------------------------------------------
- Only `experiments/kelly_regime_covkelly_v2.py` is touched. Neither
  `kelly_regime_covkelly.py` nor the parallel conservative branch's
  `kelly_regime_dual_fixed.py` is modified.
- Data discipline: `load_assets` is IMPORTED UNCHANGED from
  `kelly_regime_covkelly.py`, which hard-slices both series to
  `LOAD_CUTOFF` (2022-12-31 23:55, i.e. strictly before 2023-01-01)
  immediately after loading, before this file's code ever sees a frame.
  No literal "2023"/"2024"/"2025"/"2026" appears anywhere in this file's
  *code* (only, where unavoidable, inside this docstring's prose
  describing that cutoff -- confirmed by grepping this file before
  finishing).
- Every mu/Sigma/s_hat estimator is `.ewm(...).shift(1)` or a
  deterministic function of already-shifted columns; verified with a
  truncation/tamper causality probe (`causality_check_v2`, modeled on the
  original file's `causality_check`) run against every finalist
  candidate, not just the best one.
- No holdout read. No git add/commit/push -- an orchestrating session
  handles that after both parallel branches report.

Usage::

    python experiments/kelly_regime_covkelly_v2.py all     # everything, in order (the report this file exists to produce)
    python experiments/kelly_regime_covkelly_v2.py phase1   # just the raw/equal_sharpe/sigma_only monthly sweep (36 configs)
    python experiments/kelly_regime_covkelly_v2.py causality  # mandatory no-lookahead + equivalence checks
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.metrics import sharpe_ratio  # noqa: E402

# Reused, UNCHANGED, from the R-42 novel branch this file extends -- data
# loading/slicing, the segment-stitching engine, the baselines, the metrics
# helpers and the selection rule. None of these implement the mean/Sigma
# weight formula itself, which is what this file rewrites.
from experiments.kelly_regime_covkelly import (  # noqa: E402
    FUTURES5X,  # noqa: F401  (imported for parity; futures not run this round -- see report)
    SPOT,
    TRAIN_END,
    TRAIN_START,
    VALID_END,
    VALID_START,
    SWEEP_GRID,
    build_weight_series as build_weight_series_v1,
    daily_log_returns,
    load_assets,
    portfolio_metrics,
    r_squared,
    run_naive_5050_buyhold,
    run_v4_solo,
    select_best,
    weight_at,
    _run_leg,
    _segment_bounds,
)

N_EVALUATED = 0  # dynamic-allocator (raw/equal_sharpe/sigma_only/blend) configurations backtested


# ================================================================ causal stats

def build_stats(
    btc_df: pd.DataFrame, eth_df: pd.DataFrame,
    halflife_days: float = 60.0, min_periods_days: int | None = None,
) -> pd.DataFrame:
    """Causal EWM mean/var/cov of each asset's own raw daily log return,
    plus the derived pooled-Sharpe prior mean for `equal_sharpe`/`blend`.

    Every column is `.ewm(...).shift(1)` or a deterministic elementwise
    function of already-shifted columns -- see `causality_check_v2`.
    """
    r_btc = daily_log_returns(btc_df).rename("btc")
    r_eth = daily_log_returns(eth_df).rename("eth")
    rets = pd.concat([r_btc, r_eth], axis=1).dropna(how="all").ffill()
    rets = rets.dropna(how="any")

    mp = int(min_periods_days if min_periods_days is not None else max(20, halflife_days))
    ewm_btc = rets["btc"].ewm(halflife=halflife_days, min_periods=mp)
    ewm_eth = rets["eth"].ewm(halflife=halflife_days, min_periods=mp)

    mu_btc = ewm_btc.mean().shift(1)
    mu_eth = ewm_eth.mean().shift(1)
    var_btc = ewm_btc.var().shift(1)
    var_eth = ewm_eth.var().shift(1)
    cov = rets["btc"].ewm(halflife=halflife_days, min_periods=mp).cov(rets["eth"]).shift(1)

    out = pd.DataFrame({
        "mu_btc": mu_btc, "mu_eth": mu_eth,
        "var_btc": var_btc, "var_eth": var_eth, "cov": cov,
    })
    with np.errstate(invalid="ignore", divide="ignore"):
        out["corr"] = out["cov"] / np.sqrt(out["var_btc"] * out["var_eth"])
        out["sigma_btc"] = np.sqrt(out["var_btc"])
        out["sigma_eth"] = np.sqrt(out["var_eth"])
        # pooled/shared Sharpe prior -- see docstring point (2) for the
        # exact, stated-in-advance construction.
        out["s_hat"] = 0.5 * (out["mu_btc"] / out["sigma_btc"] + out["mu_eth"] / out["sigma_eth"])
        out["mu_prior_btc"] = out["s_hat"] * out["sigma_btc"]
        out["mu_prior_eth"] = out["s_hat"] * out["sigma_eth"]
    return out


# ============================================================ weight formulas

def _cap_scale(raw_b: float, raw_e: float, max_leg_weight: float, total_cap: float) -> tuple[float, float]:
    """No-shorting clip, per-leg cap, then total-cap rescale -- identical
    post-processing for every weight_mode so the lam=0/lam=1 blend
    boundaries can be exact (see docstring point 4)."""
    raw_b = min(max(raw_b, 0.0), max_leg_weight)
    raw_e = min(max(raw_e, 0.0), max_leg_weight)
    s = raw_b + raw_e
    if s > total_cap and s > 0:
        scale = total_cap / s
        raw_b *= scale
        raw_e *= scale
    return raw_b, raw_e


def _solve_kelly(mu: np.ndarray, Sigma: np.ndarray, kelly_frac: float) -> tuple[float, float]:
    """Classical growth-optimal `raw = Sigma^-1 mu`, fractional-Kelly
    discounted. Used by weight_mode "raw", "equal_sharpe" and the
    mu-level blend against "equal_sharpe"."""
    det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
    trace = Sigma[0, 0] + Sigma[1, 1]
    eps = 1e-8 * max(trace, 1e-12)
    if not np.isfinite(det) or abs(det) < eps:
        Sigma = Sigma + eps * np.eye(2)
        det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
    raw_b = (Sigma[1, 1] * mu[0] - Sigma[0, 1] * mu[1]) / det
    raw_e = (Sigma[0, 0] * mu[1] - Sigma[1, 0] * mu[0]) / det
    return raw_b * kelly_frac, raw_e * kelly_frac


def _solve_minvar(Sigma: np.ndarray, exposure: float) -> tuple[float, float]:
    """Mean-free minimum-variance direction `Sigma^-1 @ [1, 1]`, clipped to
    no-shorting and NORMALIZED (not fractional-Kelly-discounted, because
    there is no Kelly edge here to discount -- `exposure` directly sets
    how much of capital this mode deploys, i.e. `kelly_frac * total_cap`
    plays a different role in this mode than in "raw"/"equal_sharpe": it
    is a deployed-capital fraction, not a discount on an estimated edge).
    Degenerate case (both components clip to zero, e.g. strong negative
    covariance driving both numerators negative) falls back to an even
    split of the exposure budget, not to zero -- consistent with every
    other fallback in this file being 0.5/0.5, never "go to cash"."""
    det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
    trace = Sigma[0, 0] + Sigma[1, 1]
    eps = 1e-8 * max(trace, 1e-12)
    if not np.isfinite(det) or abs(det) < eps:
        Sigma = Sigma + eps * np.eye(2)
        det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
    raw_b = (Sigma[1, 1] - Sigma[0, 1]) / det
    raw_e = (Sigma[0, 0] - Sigma[1, 0]) / det
    raw_b = max(raw_b, 0.0)
    raw_e = max(raw_e, 0.0)
    s = raw_b + raw_e
    if s <= 0:
        return 0.5 * exposure, 0.5 * exposure
    return raw_b / s * exposure, raw_e / s * exposure


def build_weight_series_v2(
    btc_df: pd.DataFrame, eth_df: pd.DataFrame,
    weight_mode: str = "raw",
    halflife_days: float = 60.0,
    kelly_frac: float = 0.5,
    max_leg_weight: float = 1.0,
    total_cap: float = 1.0,
    min_periods_days: int | None = None,
    lam: float = 0.0,
    blend_target: str = "equal_sharpe",
) -> pd.DataFrame:
    """Causal 2x2 weight series, one row per calendar day, for
    `weight_mode` in {"raw", "equal_sharpe", "sigma_only", "blend"}.

    Row T is computed from data strictly before T for every mode: `Sigma`
    and the two candidate means (`mu` and `mu_prior`) are already
    `.ewm(...).shift(1)`'d by `build_stats`; the per-row weight solve
    below is a deterministic function of those already-causal inputs, so
    it introduces no additional lookahead. See `causality_check_v2`.
    """
    stats = build_stats(btc_df, eth_df, halflife_days, min_periods_days)
    n = len(stats)
    w_btc = np.full(n, np.nan)
    w_eth = np.full(n, np.nan)
    fallback = np.zeros(n, dtype=bool)

    mb = stats["mu_btc"].to_numpy()
    me = stats["mu_eth"].to_numpy()
    vb = stats["var_btc"].to_numpy()
    ve = stats["var_eth"].to_numpy()
    cv = stats["cov"].to_numpy()
    mpb = stats["mu_prior_btc"].to_numpy()
    mpe = stats["mu_prior_eth"].to_numpy()

    for i in range(n):
        if not (np.isfinite(mb[i]) and np.isfinite(me[i]) and np.isfinite(vb[i])
                and np.isfinite(ve[i]) and np.isfinite(cv[i])):
            fallback[i] = True
            w_btc[i], w_eth[i] = 0.5, 0.5
            continue
        Sigma = np.array([[vb[i], cv[i]], [cv[i], ve[i]]])

        if weight_mode == "raw":
            rb, re = _solve_kelly(np.array([mb[i], me[i]]), Sigma, kelly_frac)
            rb, re = _cap_scale(rb, re, max_leg_weight, total_cap)

        elif weight_mode == "equal_sharpe":
            if not (np.isfinite(mpb[i]) and np.isfinite(mpe[i])):
                fallback[i] = True
                w_btc[i], w_eth[i] = 0.5, 0.5
                continue
            rb, re = _solve_kelly(np.array([mpb[i], mpe[i]]), Sigma, kelly_frac)
            rb, re = _cap_scale(rb, re, max_leg_weight, total_cap)

        elif weight_mode == "sigma_only":
            exposure = kelly_frac * total_cap
            rb, re = _solve_minvar(Sigma, exposure)
            rb, re = _cap_scale(rb, re, max_leg_weight, total_cap)

        elif weight_mode == "blend":
            mu_raw = np.array([mb[i], me[i]])
            rb_raw, re_raw = _solve_kelly(mu_raw, Sigma, kelly_frac)
            rb_raw, re_raw = _cap_scale(rb_raw, re_raw, max_leg_weight, total_cap)
            if blend_target == "equal_sharpe":
                if not (np.isfinite(mpb[i]) and np.isfinite(mpe[i])):
                    fallback[i] = True
                    w_btc[i], w_eth[i] = 0.5, 0.5
                    continue
                mu_used = (1.0 - lam) * mu_raw + lam * np.array([mpb[i], mpe[i]])
                rb, re = _solve_kelly(mu_used, Sigma, kelly_frac)
                rb, re = _cap_scale(rb, re, max_leg_weight, total_cap)
            elif blend_target == "sigma_only":
                exposure = kelly_frac * total_cap
                rb_t, re_t = _solve_minvar(Sigma, exposure)
                rb_t, re_t = _cap_scale(rb_t, re_t, max_leg_weight, total_cap)
                # convex combo of two already-feasible points; both linear
                # constraints (leg cap, total cap) are preserved exactly,
                # so no re-capping -- this is what makes lam=0/lam=1 exact.
                rb, re = (1.0 - lam) * rb_raw + lam * rb_t, (1.0 - lam) * re_raw + lam * re_t
            else:
                raise ValueError(f"unknown blend_target: {blend_target!r}")
        else:
            raise ValueError(f"unknown weight_mode: {weight_mode!r}")

        w_btc[i], w_eth[i] = rb, re

    out = stats.copy()
    out["w_btc"] = w_btc
    out["w_eth"] = w_eth
    out["fallback"] = fallback
    return out


# =========================================================== segment runner

def run_portfolio_v2(
    btc_df: pd.DataFrame, eth_df: pd.DataFrame,
    start: str, end: str, market, start_balance: float,
    rebalance_freq: str, weight_mode: str, weight_params: dict | None = None,
    v4_kwargs: dict | None = None,
) -> dict:
    """Same block/periodic-reallocation engine as `kelly_regime_covkelly.py`
    (segment bounds, per-leg `kelly_regime_v4` sub-runs, pooled-capital
    bookkeeping are all reused unchanged via `_segment_bounds`/`_run_leg`);
    only the weight lookup is generalized to the four `weight_mode`s above
    plus the original "fixed5050" control."""
    v4_kwargs = v4_kwargs or {}
    weights_df = None
    if weight_mode == "fixed5050":
        pass
    else:
        weights_df = build_weight_series_v2(btc_df, eth_df, weight_mode=weight_mode, **(weight_params or {}))

    bounds = _segment_bounds(start, end, rebalance_freq)
    pooled = start_balance
    equity_pieces = []
    log_rows = []
    fees_total = 0.0

    for i in range(len(bounds) - 1):
        seg_start = bounds[i]
        seg_end = bounds[i + 1] - pd.Timedelta(minutes=5)
        if seg_end < seg_start:
            continue
        if weight_mode == "fixed5050":
            w_b, w_e, fb = 0.5, 0.5, False
        else:
            w_b, w_e, fb = weight_at(weights_df, seg_start)

        dollars_b = pooled * w_b
        dollars_e = pooled * w_e
        cash_leftover = pooled * max(0.0, 1.0 - w_b - w_e)

        eq_b, fees_b = _run_leg(btc_df, seg_start, seg_end, market, dollars_b, v4_kwargs)
        eq_e, fees_e = _run_leg(eth_df, seg_start, seg_end, market, dollars_e, v4_kwargs)
        fees_total += fees_b + fees_e

        idx = eq_b.index.union(eq_e.index)
        eq_b_r = eq_b.reindex(idx).ffill().bfill()
        eq_e_r = eq_e.reindex(idx).ffill().bfill()
        combined = eq_b_r + eq_e_r + cash_leftover
        equity_pieces.append(combined)

        pooled = float(combined.iloc[-1]) if len(combined) else pooled
        log_rows.append({
            "date": seg_start, "w_btc": w_b, "w_eth": w_e, "fallback": fb,
            "dollars_btc": dollars_b, "dollars_eth": dollars_e,
            "cash": cash_leftover, "pooled_end": pooled,
        })

    equity = pd.concat(equity_pieces).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    return {
        "equity": equity,
        "weights_log": pd.DataFrame(log_rows),
        "fees_paid": fees_total,
        "final_balance": float(equity.iloc[-1]) if len(equity) else start_balance,
    }


def eval_config_v2(
    btc_df: pd.DataFrame, eth_df: pd.DataFrame,
    weight_mode: str, weight_params: dict, rebalance_freq: str = "MS",
    v4_kwargs: dict | None = None,
) -> dict:
    """One counted configuration = one (weight_mode, weight_params,
    rebalance_freq) triple, evaluated on both inner-train and
    inner-validation -- same counting convention as the file this
    extends (`N_EVALUATED` increments once per call here, not once per
    window)."""
    global N_EVALUATED
    N_EVALUATED += 1
    out = {}
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)), ("valid", (VALID_START, VALID_END))):
        res = run_portfolio_v2(btc_df, eth_df, s, e, SPOT, 1000.0, rebalance_freq,
                               weight_mode, weight_params, v4_kwargs)
        out[label] = portfolio_metrics(res["equity"], 1000.0)
        out[label]["_res"] = res
    return out


# ============================================================ phase 1: sweep

def run_phase1(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> dict:
    """Monthly-cadence sweep of the same 12-point hyperparameter grid
    (`SWEEP_GRID`, imported unchanged from the file this extends) for each
    of the three named treatments: raw (control), equal_sharpe,
    sigma_only. 36 configurations. Selects a per-mode winner with the
    SAME selection rule as the original branch (`select_best`, imported
    unchanged: rank by min(train, valid) Sharpe, tie-break on valid max
    DD)."""
    modes = ("raw", "equal_sharpe", "sigma_only")
    results = {}
    for mode in modes:
        rows = []
        for params in SWEEP_GRID:
            r = eval_config_v2(btc_df, eth_df, mode, params, rebalance_freq="MS")
            rows.append({"params": params, "train": r["train"], "valid": r["valid"]})
            print(f"[{mode:<12}] {params} | train final={r['train']['final_balance']:.0f} "
                  f"Sharpe={r['train']['sharpe']:.2f} DD={r['train']['max_dd_pct']:.1f}% "
                  f"|| valid final={r['valid']['final_balance']:.0f} "
                  f"Sharpe={r['valid']['sharpe']:.2f} DD={r['valid']['max_dd_pct']:.1f}%")
        best = select_best(rows)
        results[mode] = {"rows": rows, "best": best}
        print(f"  -> best[{mode}]: {best['params']} "
              f"(train Sharpe={best['train']['sharpe']:.2f}, valid Sharpe={best['valid']['sharpe']:.2f})\n")
    print(f"phase 1 configs evaluated: {len(modes) * len(SWEEP_GRID)} (N_EVALUATED so far: {N_EVALUATED})")
    return results


def choose_blend_target(phase1: dict) -> str:
    """Early-testing decision, made explicitly and reported: which of
    equal_sharpe / sigma_only scores higher (same min(train,valid) Sharpe
    metric as `select_best`) becomes the blend target for phase 3."""
    def score(best):
        return min(best["train"]["sharpe"], best["valid"]["sharpe"])
    s_es = score(phase1["equal_sharpe"]["best"])
    s_so = score(phase1["sigma_only"]["best"])
    target = "equal_sharpe" if s_es >= s_so else "sigma_only"
    print(f"blend-target decision: equal_sharpe score={s_es:.3f}, sigma_only score={s_so:.3f} "
          f"-> chosen target = {target!r}")
    return target


# ============================================================ phase 2: blend

BLEND_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def run_phase2_blend(btc_df: pd.DataFrame, eth_df: pd.DataFrame, base_params: dict, blend_target: str) -> dict:
    """Sweep the shrinkage-intensity knob `lam` at the raw-control's own
    winning hyperparameters (halflife/kelly_frac/max_leg_weight held
    fixed at `best[raw]`'s values, only `lam` varies), monthly cadence.
    5 configurations."""
    rows = []
    for lam in BLEND_LAMBDAS:
        params = {**base_params, "lam": lam, "blend_target": blend_target}
        r = eval_config_v2(btc_df, eth_df, "blend", params, rebalance_freq="MS")
        rows.append({"params": params, "train": r["train"], "valid": r["valid"]})
        print(f"[blend lam={lam:.2f} -> {blend_target}] train final={r['train']['final_balance']:.0f} "
              f"Sharpe={r['train']['sharpe']:.2f} || valid final={r['valid']['final_balance']:.0f} "
              f"Sharpe={r['valid']['sharpe']:.2f}")
    best = select_best(rows)
    print(f"  -> best[blend]: lam={best['params']['lam']} "
          f"(train Sharpe={best['train']['sharpe']:.2f}, valid Sharpe={best['valid']['sharpe']:.2f})")
    print(f"phase 2 configs evaluated: {len(BLEND_LAMBDAS)} (N_EVALUATED so far: {N_EVALUATED})")
    return {"rows": rows, "best": best}


# ==================================================== phase 3: cadence check

def run_phase3_cadence(btc_df: pd.DataFrame, eth_df: pd.DataFrame, finalists: dict) -> dict:
    """The mandatory cadence-driven train/validation inconsistency
    diagnostic, re-run for every finalist candidate: weekly (W-MON)
    repeat of each candidate's already-selected winning hyperparameters
    (monthly numbers come from phase 1/2, already computed -- not
    re-run). 4 configurations (one weekly re-run per finalist)."""
    weekly = {}
    for name, entry in finalists.items():
        mode = entry["mode"]
        params = entry["params"]
        r = eval_config_v2(btc_df, eth_df, mode, params, rebalance_freq="W-MON")
        weekly[name] = {"train": r["train"], "valid": r["valid"]}
        print(f"[{name} weekly] train final={r['train']['final_balance']:.0f} "
              f"Sharpe={r['train']['sharpe']:.2f} || valid final={r['valid']['final_balance']:.0f} "
              f"Sharpe={r['valid']['sharpe']:.2f}")
    print(f"phase 3 configs evaluated: {len(finalists)} (N_EVALUATED so far: {N_EVALUATED})")
    return weekly


def cadence_flip_report(finalists: dict, weekly: dict) -> None:
    """For each finalist: does monthly lose to baselines-free comparison
    (train Sharpe vs valid Sharpe sign / ranking) while weekly does not
    -- the exact R-42 signature -- restated as: does validation Sharpe
    go from negative-or-losing at monthly cadence to positive-or-winning
    at weekly cadence, holding hyperparameters fixed?"""
    print("\n=== cadence-driven train/validation inconsistency, per finalist ===")
    for name, entry in finalists.items():
        m_train = entry["train"]["sharpe"]
        m_valid = entry["valid"]["sharpe"]
        w_train = weekly[name]["train"]["sharpe"]
        w_valid = weekly[name]["valid"]["sharpe"]
        flip = (m_valid < 0 <= w_valid) or (m_train > m_valid and w_valid > w_train)
        print(f"{name:<14} monthly: train Sharpe={m_train:+.2f} valid Sharpe={m_valid:+.2f}  |  "
              f"weekly: train Sharpe={w_train:+.2f} valid Sharpe={w_valid:+.2f}  "
              f"-> {'SAME SIGNATURE (flip present)' if flip else 'flip not reproduced'}")


# ============================================================== diagnostics

def causality_check_v2(btc_df: pd.DataFrame, eth_df: pd.DataFrame, finalists: dict) -> bool:
    """Truncation/tamper causality probe (modeled on the original file's
    `causality_check`), applied to EVERY finalist candidate's weight_mode
    + params, not just the best one."""
    cut = pd.Timestamp("2021-06-30", tz="UTC")
    K = 137.0

    def tamper(df: pd.DataFrame, factor: float) -> pd.DataFrame:
        out = df.copy()
        mask = out.index > cut
        for col in ("open", "high", "low", "close"):
            out.loc[mask, col] = out.loc[mask, col] * factor
        return out

    btc_up, eth_up = tamper(btc_df, K), tamper(eth_df, K)
    btc_dn, eth_dn = tamper(btc_df, 1.0 / K), tamper(eth_df, 1.0 / K)

    all_ok = True
    print("\n=== causality probe (truncation/tamper), per finalist ===")
    for name, entry in finalists.items():
        mode, params = entry["mode"], {k: v for k, v in entry["params"].items()}
        base = build_weight_series_v2(btc_df, eth_df, weight_mode=mode, **params)
        up = build_weight_series_v2(btc_up, eth_up, weight_mode=mode, **params)
        dn = build_weight_series_v2(btc_dn, eth_dn, weight_mode=mode, **params)
        pre = base.index <= cut
        cols = ["mu_btc", "mu_eth", "var_btc", "var_eth", "cov", "w_btc", "w_eth"]
        b = base.loc[pre, cols].to_numpy()
        u = up.loc[pre, cols].to_numpy()
        d = dn.loc[pre, cols].to_numpy()
        max_up = np.nanmax(np.abs(b - u))
        max_dn = np.nanmax(np.abs(b - d))
        ok = max_up < 1e-9 and max_dn < 1e-9
        all_ok = all_ok and ok
        print(f"{name:<14} mode={mode:<12} max|base-up|={max_up:.3e}  max|base-down|={max_dn:.3e}  PASS={ok}")
    return all_ok


def verify_raw_matches_v1(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> bool:
    """Sanity check #1: this file's from-scratch "raw" weight computation
    reproduces `kelly_regime_covkelly.py`'s `build_weight_series`
    bit-for-bit (same formula, independently re-typed per this task's
    instruction that the weight-estimation function be genuinely
    rewritten -- this is the check that it was rewritten CORRECTLY)."""
    params = {"halflife_days": 60.0, "kelly_frac": 0.5, "max_leg_weight": 1.0, "total_cap": 1.0}
    v1_out = build_weight_series_v1(btc_df, eth_df, **params)
    v2_out = build_weight_series_v2(btc_df, eth_df, weight_mode="raw", **params)
    cols = ["mu_btc", "mu_eth", "var_btc", "var_eth", "cov", "w_btc", "w_eth"]
    diff = np.nanmax(np.abs(v1_out[cols].to_numpy() - v2_out[cols].to_numpy()))
    ok = diff < 1e-9
    print(f"verify_raw_matches_v1: max abs diff across {cols} = {diff:.3e}  PASS={ok}")
    return ok


def verify_lambda_boundaries(btc_df: pd.DataFrame, eth_df: pd.DataFrame, base_params: dict, blend_target: str) -> bool:
    """Sanity check #2 (the one explicitly required by this task): lam=0
    exactly reproduces "raw", and lam=1 exactly reproduces the chosen
    blend target, on the weight series itself (not just downstream
    equity, which could mask a real difference through path-dependent
    v4 rounding)."""
    raw_out = build_weight_series_v2(btc_df, eth_df, weight_mode="raw", **base_params)
    target_out = build_weight_series_v2(btc_df, eth_df, weight_mode=blend_target, **base_params)
    blend_lo = build_weight_series_v2(btc_df, eth_df, weight_mode="blend", lam=0.0,
                                       blend_target=blend_target, **base_params)
    blend_hi = build_weight_series_v2(btc_df, eth_df, weight_mode="blend", lam=1.0,
                                       blend_target=blend_target, **base_params)
    cols = ["w_btc", "w_eth"]
    diff_lo = np.nanmax(np.abs(raw_out[cols].to_numpy() - blend_lo[cols].to_numpy()))
    diff_hi = np.nanmax(np.abs(target_out[cols].to_numpy() - blend_hi[cols].to_numpy()))
    ok = diff_lo < 1e-9 and diff_hi < 1e-9
    print(f"verify_lambda_boundaries: lam=0 vs raw max diff={diff_lo:.3e}; "
          f"lam=1 vs {blend_target} max diff={diff_hi:.3e}  PASS={ok}")
    return ok


def artifact_diagnostics_v2(finalists: dict, weekly: dict, fixed_wk: dict, solo: dict) -> dict:
    """Mandatory exposure-artifact R^2 diagnostics, computed at the
    weekly cadence (matching the original file's `run_headline` default)
    for every finalist against (a) a flat-rescaled BTC-only v4 book and
    (b) the fixed 50/50 control, both windows. Reuses the equity curves
    already produced by `run_phase3_cadence` / the baseline computation
    in `run_all` -- no backtest is re-run here."""
    print("\n=== exposure-artifact diagnostics (weekly cadence) ===")
    r2 = {}
    for name in finalists:
        r2[name] = {}
        for label in ("train", "valid"):
            dyn_eq = weekly[name][label]["_res"]["equity"]
            r2_solo = r_squared(dyn_eq, solo[label]["_res"].equity)
            r2_fixed = r_squared(dyn_eq, fixed_wk[label]["_res"]["equity"])
            r2[name][label] = {"r2_solo": r2_solo, "r2_fixed": r2_fixed}
            flag_solo = "FLAT-RESCALE ARTIFACT" if r2_solo > 0.95 else "ok"
            flag_fixed = "SAME AS FIXED SPLIT" if r2_fixed > 0.95 else "ok"
            print(f"[{name}/{label}] vs solo R^2={r2_solo:.4f} ({flag_solo})  "
                  f"vs fixed5050 R^2={r2_fixed:.4f} ({flag_fixed})")
    return r2


def bear2022_check_v2(btc_df: pd.DataFrame, eth_df: pd.DataFrame, finalists: dict) -> None:
    """The pre-registered 2022-joint-bear concentration check, re-run for
    every finalist: does de-noising still let the allocator degrade
    sensibly toward the better-prospect asset, or does shrinking the
    mean just collapse it to an undifferentiated flat split (report
    plainly if so -- that is the "threw out the baby with the
    bathwater" failure mode this task named explicitly)."""
    print("\n=== 2022 joint-bear concentration check, per finalist ===")
    for name, entry in finalists.items():
        mode, params = entry["mode"], entry["params"]
        weights = build_weight_series_v2(btc_df, eth_df, weight_mode=mode, **params)
        print(f"-- {name} (mode={mode}) --")
        for year in (2019, 2020, 2021, 2022):
            seg = weights.loc[f"{year}-01-01":f"{year}-12-31"]
            seg = seg[~seg["fallback"]]
            if len(seg) == 0:
                print(f"  {year}: no non-fallback rows")
                continue
            invested = seg["w_btc"] + seg["w_eth"]
            near_single = 100 * np.mean(np.maximum(seg["w_btc"], seg["w_eth"]) / (invested + 1e-9) > 0.85)
            print(f"  {year}: mean corr={seg['corr'].mean():.3f}  mean w_btc={seg['w_btc'].mean():.3f}  "
                  f"mean w_eth={seg['w_eth'].mean():.3f}  mean invested={invested.mean():.3f}  "
                  f"pct near-single-asset(>85%)={near_single:.1f}%")


# ===================================================================== CLI

def run_all(data_dir: str = "data") -> None:
    btc_df, eth_df = load_assets(data_dir)

    print("=== sanity check: raw mode reproduces the R-42 novel branch's own formula ===")
    verify_raw_matches_v1(btc_df, eth_df)

    print("\n=== phase 1: monthly sweep, raw / equal_sharpe / sigma_only (36 configs) ===")
    phase1 = run_phase1(btc_df, eth_df)
    blend_target = choose_blend_target(phase1)

    print(f"\n=== phase 2: blend lambda sweep against target={blend_target!r} (5 configs) ===")
    raw_best_params = phase1["raw"]["best"]["params"]
    phase2 = run_phase2_blend(btc_df, eth_df, raw_best_params, blend_target)

    print("\n=== sanity check: blend lambda boundaries reproduce raw / target exactly ===")
    verify_lambda_boundaries(btc_df, eth_df, raw_best_params, blend_target)

    finalists = {
        "raw": {"mode": "raw", "params": phase1["raw"]["best"]["params"],
                "train": phase1["raw"]["best"]["train"], "valid": phase1["raw"]["best"]["valid"]},
        "equal_sharpe": {"mode": "equal_sharpe", "params": phase1["equal_sharpe"]["best"]["params"],
                         "train": phase1["equal_sharpe"]["best"]["train"], "valid": phase1["equal_sharpe"]["best"]["valid"]},
        "sigma_only": {"mode": "sigma_only", "params": phase1["sigma_only"]["best"]["params"],
                      "train": phase1["sigma_only"]["best"]["train"], "valid": phase1["sigma_only"]["best"]["valid"]},
        "blend": {"mode": "blend", "params": phase2["best"]["params"],
                 "train": phase2["best"]["train"], "valid": phase2["best"]["valid"]},
    }

    print("\n=== phase 3: weekly-cadence repeat of every finalist (4 configs) ===")
    weekly = run_phase3_cadence(btc_df, eth_df, finalists)
    cadence_flip_report(finalists, weekly)

    ok_causal = causality_check_v2(btc_df, eth_df, finalists)

    # baselines, computed ONCE and reused below (not counted in
    # N_EVALUATED, same convention as the file this extends)
    naive = {}
    fixed_ms, fixed_wk = {}, {}
    solo = {}
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)), ("valid", (VALID_START, VALID_END))):
        res_naive = run_naive_5050_buyhold(btc_df, eth_df, s, e, SPOT, 1000.0)
        naive[label] = portfolio_metrics(res_naive["equity"], 1000.0)
        res_fixed_ms = run_portfolio_v2(btc_df, eth_df, s, e, SPOT, 1000.0, "MS", "fixed5050")
        fixed_ms[label] = portfolio_metrics(res_fixed_ms["equity"], 1000.0)
        res_fixed_wk = run_portfolio_v2(btc_df, eth_df, s, e, SPOT, 1000.0, "W-MON", "fixed5050")
        fixed_wk[label] = {**portfolio_metrics(res_fixed_wk["equity"], 1000.0), "_res": res_fixed_wk}
        res_solo = run_v4_solo(btc_df, s, e, SPOT, 1000.0)
        solo[label] = {"final_balance": float(res_solo.equity.iloc[-1]),
                      "sharpe": sharpe_ratio(res_solo.equity.to_numpy()),
                      "max_dd_pct": None, "_res": res_solo}

    r2 = artifact_diagnostics_v2(finalists, weekly, fixed_wk, solo)

    bear2022_check_v2(btc_df, eth_df, finalists)

    print("\n=== FINAL HEADLINE TABLE (spot) ===")
    header = f"{'candidate':<16} {'cadence':<8} {'period':<6} {'final':>10} {'sharpe':>8} {'maxDD%':>8}"
    print(header)
    for name, entry in finalists.items():
        for label in ("train", "valid"):
            m = entry[label]
            print(f"{name:<16} {'MS':<8} {label:<6} {m['final_balance']:>10.1f} {m['sharpe']:>8.2f} {m['max_dd_pct']:>8.1f}")
        for label in ("train", "valid"):
            m = weekly[name][label]
            print(f"{name:<16} {'W-MON':<8} {label:<6} {m['final_balance']:>10.1f} {m['sharpe']:>8.2f} {m['max_dd_pct']:>8.1f}")
    for label in ("train", "valid"):
        m = fixed_ms[label]
        print(f"{'fixed5050':<16} {'MS':<8} {label:<6} {m['final_balance']:>10.1f} {m['sharpe']:>8.2f} {m['max_dd_pct']:>8.1f}")
        m = fixed_wk[label]
        print(f"{'fixed5050':<16} {'W-MON':<8} {label:<6} {m['final_balance']:>10.1f} {m['sharpe']:>8.2f} {m['max_dd_pct']:>8.1f}")
        m = naive[label]
        print(f"{'naive_buyhold':<16} {'--':<8} {label:<6} {m['final_balance']:>10.1f} {m['sharpe']:>8.2f} {m['max_dd_pct']:>8.1f}")
        m = solo[label]
        print(f"{'v4_btc_solo':<16} {'--':<8} {label:<6} {m['final_balance']:>10.1f} {m['sharpe']:>8.2f} {'n/a':>8}")

    print(f"\nblend target chosen: {blend_target!r}")
    print(f"causality PASS (all finalists): {ok_causal}")
    print(f"TOTAL dynamic-allocator configurations evaluated (N_EVALUATED): {N_EVALUATED}")
    print("(+ 4 baseline backtests not counted above: fixed5050 x2 cadences, naive buy&hold, v4-BTC-solo)")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    data_dir = "data"
    if cmd == "phase1":
        btc_df, eth_df = load_assets(data_dir)
        run_phase1(btc_df, eth_df)
    elif cmd == "causality":
        btc_df, eth_df = load_assets(data_dir)
        verify_raw_matches_v1(btc_df, eth_df)
        phase1 = run_phase1(btc_df, eth_df)
        blend_target = choose_blend_target(phase1)
        raw_best_params = phase1["raw"]["best"]["params"]
        verify_lambda_boundaries(btc_df, eth_df, raw_best_params, blend_target)
        finalists = {name: {"mode": name, "params": phase1[name]["best"]["params"]}
                    for name in ("raw", "equal_sharpe", "sigma_only")}
        causality_check_v2(btc_df, eth_df, finalists)
    elif cmd == "all":
        run_all(data_dir)
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
