"""R-118 NOVEL branch: robust selection via Monte Carlo draws from a
FITTED regime-switching jump-diffusion generative model.

See `experiments/r118_shared.py`'s module docstring for the full
pre-registration (direction, literature, falsification test, "not a
duplicate of" list, and the pre-registered EXPECTED outcome for this
branch). This file only implements the novel branch's two novel pieces:
(1) fitting a regime-switching jump-diffusion model to REAL, causal
2017-2020 BTC training data, and (2) a `path_generator(seed)` that draws
synthetic OHLCV paths from that fitted model, fed into the frozen
`select_config`/`evaluate_candidate` machinery in `r118_shared.py`
unmodified.

FITTING METHOD (method-of-moments / empirical-quantile labeling, not
full EM -- `scipy` is not installed in this environment, verified by
`python -c "import scipy"` failing before any code below was written;
this method needs only numpy/pandas and is fully causal, using only
`load_inner_train_btc()`, 2017-01-01..2020-12-31):

  1. Compute 5-minute log returns from the real inner-train BTC close
     series (420,768 bars, exactly 1,461 calendar days x 288 bars/day,
     zero gaps -- verified).
  2. For each calendar day, compute realized volatility
     RV_d = sqrt(sum of that day's squared 5-min log returns) -- a
     standard, assumption-light realized-vol estimator.
  3. LABEL each day into one of N_STATES regimes (default 3, matching
     `generate_synthetic_pair`'s state count) by TERTILE of RV_d. This is
     the "rolling realized-vol quantile labeling" method the round's
     brief explicitly names as a defensible, fast method-of-moments
     alternative to full Baum-Welch EM. Unlike a hand-picked bull/bear/
     chop label, the regime identity here is purely a vol-rank label;
     each regime's DRIFT is then estimated empirically from the data
     assigned to it (not assumed), so regime 1 need not be "the bull
     regime" a priori -- it is whatever drift the data conditional on
     that vol tertile actually shows.
  4. For each regime k, estimate:
       mu_k    = mean(5-min log return | day in regime k) * BARS_PER_YEAR
       sigma_k = std(5-min log return  | day in regime k) * sqrt(BARS_PER_YEAR)
     (annualized drift/vol, pooling all bars from days sharing that
     label -- the direct sample-moment estimator, no distributional
     assumption beyond first/second moments.)
  5. Estimate day-to-day TRANSITION probabilities empirically: count
     transitions between consecutive days' labels over the whole
     1,461-day sequence, Laplace-smoothed (+1 pseudo-count per cell) to
     avoid zero-probability transitions the training window's finite
     length just didn't happen to realize, and row-normalize to a
     N_STATES x N_STATES stochastic matrix. This IS the transition
     matrix's maximum-likelihood estimate for a first-order Markov chain
     observed through a fixed labeling -- the standard MOM/MLE estimator
     when states are given rather than latent (which is what distinguishes
     this from full Hamilton-filter EM: here the "E-step" is a one-shot
     vol-quantile label rather than a jointly-optimized latent-state
     posterior. Documented as a simplification, not hidden.)
  6. Estimate CLUSTERED VOLATILITY beyond the discrete regime label: let
     resid_d = log(RV_d) - mean(log(RV_d) | label_d) (the log-vol
     residual after removing the regime's own average). Fit an AR(1) to
     this residual by simple lag-1 OLS-equivalent moment estimators:
       rho        = corr(resid_d[1:], resid_d[:-1])   (lag-1 autocorr)
       innov_std  = std(resid_d[1:] - rho * resid_d[:-1])
     This is exactly `generate_synthetic_pair`'s own AR(1)-log-vol
     mechanism, but with (rho, innov_std) FIT to real data instead of
     hand-set (0.999, 0.02) constants.
  7. Estimate the JUMP component at bar (5-min) granularity: for each
     bar, compute that DAY's local per-bar sigma estimate
     local_sigma_d = RV_d / sqrt(BARS_PER_DAY), flag a bar as a "jump"
     if |bar log-return| > K_JUMP * local_sigma_d (K_JUMP = 4.0, a
     standard tail-outlier multiple). jump_rate = fraction of bars
     flagged; jump_mean/jump_std = sample mean/std of the flagged bars'
     own returns (treated as one Normal jump-size distribution -- a
     simplification; real crypto jump sizes are more likely fat-tailed/
     skewed, documented as a limitation, not hidden).

  All seven numbers/matrices above are estimated ONCE from
  `load_inner_train_btc()` and then FROZEN before any synthetic path is
  drawn or any grid point is scored -- exactly as the round's brief
  requires (fit first, select second, never re-fit after seeing scores).

SIMULATION METHOD (`path_generator(seed)`): reuses the mathematical
SHAPE of `tradebot.data.generate_synthetic_pair` (regime-switching drift,
AR(1) clustered log-vol multiplier, additive jump component, GBM-style
per-substep log-return accumulation, 4 substeps/bar for OHLC), but:
  (a) regime switches at DAY granularity via the fitted N_STATES x
      N_STATES transition matrix (a proper multi-state Markov chain
      with asymmetric transition probabilities -- the shipped function
      instead runs a single-scalar "stay" probability with uniform
      random reassignment on switch, at a much finer/arbitrary substep
      granularity that was never fit to anything), starting from the
      empirical stationary (marginal label-frequency) distribution;
  (b) the AR(1) vol multiplier uses the fitted (rho, innov_std), also
      at day granularity (matching what was actually fit);
  (c) jump arrivals are bar-level Bernoulli(fitted jump_rate) with size
      ~ Normal(fitted jump_mean, fitted jump_std), applied once per
      flagged bar;
  (d) the per-substep innovation loop is fully VECTORIZED (no
      substep-level Python loop over ~1.68M elements, unlike the
      original's per-substep regime loop) since regime/vol-mult are
      now day-level arrays broadcast up to substeps -- this is what
      keeps the N_DRAWS=40-path sweep inside the round's time budget.

Each `path_generator(seed)` draws a genuinely different path via
`np.random.default_rng(seed)`; output length is fixed at exactly
`len(load_inner_train_btc())` bars (420,768 = 1,461 days x 288
bars/day) so every synthetic path spans the same duration as the real
training window, matching the round's brief.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import r118_shared as sh

ROOT = Path(__file__).resolve().parents[1]

N_STATES = 3
K_JUMP = 4.0
STEPS_PER_BAR = 4          # substeps/bar, for OHLC shaping (matches generate_synthetic_pair)
START_PRICE = 10_000.0     # arbitrary; strategy scoring is scale-invariant (log returns)
SECONDS_PER_YEAR = 365.25 * 24 * 3600


# ------------------------------------------------------------------------
# Step 1: fit the regime-switching jump-diffusion model to REAL, causal
# training-period data.
# ------------------------------------------------------------------------

def fit_regime_model(df_train: pd.DataFrame, n_states: int = N_STATES,
                     k_jump: float = K_JUMP) -> dict:
    """Fit all parameters documented in the module docstring above from
    ``df_train`` (must be `load_inner_train_btc()` or a strict causal
    subset of it). Returns a plain-JSON-able dict."""
    close = df_train["close"].to_numpy(dtype=float)
    n_bars = len(close)
    assert n_bars % sh.BARS_PER_DAY == 0, "expected a whole number of days"
    n_days = n_bars // sh.BARS_PER_DAY
    bars_per_year = 365.25 * sh.BARS_PER_DAY

    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    logret[0] = 0.0

    # per-day realized vol and per-day label (bar-index reshape, since
    # load_inner_train_btc is verified gap-free and exactly n_days*288).
    r_by_day = logret.reshape(n_days, sh.BARS_PER_DAY)
    RV = np.sqrt(np.sum(r_by_day ** 2, axis=1))          # (n_days,)

    edges = np.quantile(RV, np.linspace(0, 1, n_states + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    labels = np.searchsorted(edges, RV, side="right") - 1
    labels = np.clip(labels, 0, n_states - 1)              # (n_days,) ints in [0, n_states)

    labels_bar = np.repeat(labels, sh.BARS_PER_DAY)         # (n_bars,)

    mu = np.zeros(n_states)
    sigma = np.zeros(n_states)
    n_days_per_state = np.zeros(n_states, dtype=int)
    for k in range(n_states):
        rk = logret[labels_bar == k]
        mu[k] = rk.mean() * bars_per_year
        sigma[k] = rk.std() * np.sqrt(bars_per_year)
        n_days_per_state[k] = int((labels == k).sum())

    # transition matrix (Laplace-smoothed MLE)
    counts = np.ones((n_states, n_states))
    for a, b in zip(labels[:-1], labels[1:]):
        counts[a, b] += 1.0
    P = counts / counts.sum(axis=1, keepdims=True)

    # stationary / initial distribution: empirical label marginal frequency
    init_dist = np.array([(labels == k).mean() for k in range(n_states)])
    init_dist = init_dist / init_dist.sum()

    # AR(1) log-vol clustering residual
    log_rv = np.log(RV)
    regime_mean_logvol = np.array([log_rv[labels == k].mean() for k in range(n_states)])
    resid = log_rv - regime_mean_logvol[labels]
    rho = float(np.corrcoef(resid[1:], resid[:-1])[0, 1])
    innov = resid[1:] - rho * resid[:-1]
    innov_std = float(innov.std())

    # jump component (bar-level, local per-day vol scale)
    local_bar_sigma = np.repeat(RV / np.sqrt(sh.BARS_PER_DAY), sh.BARS_PER_DAY)
    thresh = k_jump * local_bar_sigma
    flagged = np.abs(logret) > thresh
    jump_rate = float(flagged.mean())
    jump_sizes = logret[flagged]
    jump_mean = float(jump_sizes.mean()) if flagged.any() else 0.0
    jump_std = float(jump_sizes.std()) if flagged.any() else 0.0

    return dict(
        n_states=n_states,
        n_bars=int(n_bars),
        n_days=int(n_days),
        mu_annual=mu.tolist(),
        sigma_annual=sigma.tolist(),
        n_days_per_state=n_days_per_state.tolist(),
        transition_matrix=P.tolist(),
        init_dist=init_dist.tolist(),
        ar1_rho=rho,
        ar1_innov_std=innov_std,
        jump_rate=jump_rate,
        jump_mean=jump_mean,
        jump_std=jump_std,
        k_jump=k_jump,
        rv_quantile_edges=edges[1:-1].tolist(),
    )


# ------------------------------------------------------------------------
# Step 2: simulate a synthetic OHLCV path from the fitted model.
# ------------------------------------------------------------------------

def simulate_path(params: dict, n_bars: int, seed: int,
                  start_price: float = START_PRICE,
                  start: str = sh.INNER_TRAIN_START) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_states = params["n_states"]
    mu = np.asarray(params["mu_annual"])
    sigma = np.asarray(params["sigma_annual"])
    P = np.asarray(params["transition_matrix"])
    init_dist = np.asarray(params["init_dist"])
    rho = params["ar1_rho"]
    innov_std = params["ar1_innov_std"]
    jump_rate = params["jump_rate"]
    jump_mean = params["jump_mean"]
    jump_std = params["jump_std"]

    assert n_bars % sh.BARS_PER_DAY == 0
    n_days = n_bars // sh.BARS_PER_DAY

    # --- day-level regime Markov chain (fitted transition matrix) ---
    regime_day = np.empty(n_days, dtype=np.int64)
    regime_day[0] = rng.choice(n_states, p=init_dist)
    u = rng.random(n_days - 1)
    cum_P = np.cumsum(P, axis=1)
    for t in range(1, n_days):
        prev = regime_day[t - 1]
        regime_day[t] = np.searchsorted(cum_P[prev], u[t - 1])

    # --- day-level AR(1) clustered-vol multiplier (fitted rho/innov_std) ---
    lv = np.empty(n_days)
    lv[0] = 0.0
    eps = rng.normal(0.0, innov_std, size=n_days)
    for t in range(1, n_days):
        lv[t] = rho * lv[t - 1] + eps[t]
    vol_mult_day = np.exp(lv)

    mu_day = mu[regime_day]                       # (n_days,)
    sigma_day = sigma[regime_day] * vol_mult_day   # (n_days,)

    # broadcast day-level params to bar and then substep granularity
    mu_bar = np.repeat(mu_day, sh.BARS_PER_DAY)          # (n_bars,)
    sigma_bar = np.repeat(sigma_day, sh.BARS_PER_DAY)    # (n_bars,)

    mu_sub = np.repeat(mu_bar, STEPS_PER_BAR)            # (n_sub,)
    sigma_sub = np.repeat(sigma_bar, STEPS_PER_BAR)      # (n_sub,)

    dt_bar_years = (5 * 60) / SECONDS_PER_YEAR
    dt_sub_years = dt_bar_years / STEPS_PER_BAR

    z = rng.normal(size=len(mu_sub))
    diffusion = ((mu_sub - 0.5 * sigma_sub ** 2) * dt_sub_years
                 + sigma_sub * np.sqrt(dt_sub_years) * z)

    # jump component: bar-level Bernoulli(jump_rate), applied to each
    # flagged bar's first substep.
    jump_flag_bar = rng.random(n_bars) < jump_rate
    if jump_flag_bar.any():
        jump_vals = rng.normal(jump_mean, jump_std, size=int(jump_flag_bar.sum()))
        sub_idx_of_bar0 = np.arange(n_bars)[jump_flag_bar] * STEPS_PER_BAR
        diffusion[sub_idx_of_bar0] += jump_vals

    log_price = np.log(start_price) + np.cumsum(diffusion)
    price = np.exp(log_price)
    sub = price.reshape(n_bars, STEPS_PER_BAR)

    idx = pd.date_range(start=start, periods=n_bars, freq="5min", tz="UTC")
    ohlc = pd.DataFrame({
        "open": np.concatenate([[start_price], sub[:-1, -1]]),
        "high": sub.max(axis=1),
        "low": sub.min(axis=1),
        "close": sub[:, -1],
    }, index=idx)
    ohlc["high"] = ohlc[["open", "high", "close"]].max(axis=1)
    ohlc["low"] = ohlc[["open", "low", "close"]].min(axis=1)

    bar_ret = np.abs(np.diff(np.log(sub[:, -1]), prepend=np.log(start_price)))
    ohlc["volume"] = 50.0 * (1.0 + 400.0 * bar_ret) * np.exp(rng.normal(0, 0.5, n_bars))

    # sanity checks -- own synthetic output, per the round's brief.
    assert len(ohlc) == n_bars
    assert np.isfinite(ohlc.to_numpy()).all(), "NaN/Inf in synthetic path"
    assert (ohlc[["open", "high", "low", "close"]] > 0).to_numpy().all(), "non-positive price"
    assert (ohlc["high"] >= ohlc[["open", "close", "low"]].max(axis=1) - 1e-9).all()
    assert (ohlc["low"] <= ohlc[["open", "close", "high"]].min(axis=1) + 1e-9).all()

    return ohlc


# ------------------------------------------------------------------------
# Fit once (module load), build the path_generator select_config expects.
# ------------------------------------------------------------------------

_TRAIN_DF = sh.load_inner_train_btc()
sh.assert_no_holdout(_TRAIN_DF, "r118_novel fit input")
FITTED_PARAMS = fit_regime_model(_TRAIN_DF, n_states=N_STATES, k_jump=K_JUMP)
N_BARS_TARGET = len(_TRAIN_DF)


def path_generator(seed: int) -> pd.DataFrame:
    return simulate_path(FITTED_PARAMS, N_BARS_TARGET, seed)


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

def main() -> None:
    sh.hr("R-118 NOVEL: fitted regime-switching jump-diffusion Monte Carlo selection")
    print("Fitted parameters (from load_inner_train_btc(), 2017-01-01..2020-12-31):")
    print(json.dumps(FITTED_PARAMS, indent=2))

    # time a couple of path draws + backtests before committing to full sweep
    t0 = time.time()
    p0 = path_generator(0)
    t_sim = time.time() - t0
    t1 = time.time()
    _ = sh.score_on_path(sh.V4_DEFAULT, p0, sh.SPOT)
    t_bt = time.time() - t1
    print(f"\n[timing] one simulate_path: {t_sim:.2f}s, one score_on_path: {t_bt:.2f}s")
    per_draw = t_sim + t_bt * len(sh.GRID)
    est_total_min = per_draw * sh.N_DRAWS / 60.0
    print(f"[timing] estimated full sweep ({sh.N_DRAWS} draws x {len(sh.GRID)} configs "
          f"= {sh.N_DRAWS * len(sh.GRID)} backtests + {sh.N_DRAWS} sims): "
          f"~{est_total_min:.1f} min")

    n_draws = sh.N_DRAWS
    if per_draw * n_draws > 70 * 60:  # > ~70 min projected: scale down, disclose
        n_draws = max(10, int(70 * 60 / per_draw))
        print(f"[timing] scaling n_draws down from {sh.N_DRAWS} to {n_draws} "
              f"to stay inside the ~60-75 min budget (disclosed).")

    sh.hr(f"Selection sweep: n_draws={n_draws}, grid={len(sh.GRID)} points, "
         f"{n_draws * len(sh.GRID)} backtests")
    t_sweep0 = time.time()
    best_config, table = sh.select_config(path_generator, n_draws=n_draws,
                                          grid=sh.GRID, market=sh.SPOT)
    sweep_time = time.time() - t_sweep0
    print(f"[timing] actual sweep wall time: {sweep_time / 60.0:.1f} min")

    print("\nSelection table (config -> mean, std, robust/CVaR Sharpe over synthetic draws):")
    for cfg in sh.GRID:
        row = table[cfg]
        marker = "  <== SELECTED" if cfg == best_config else ""
        print(f"  base={cfg[0]:3d} tv={cfg[1]:.2f} ml={cfg[2]:.1f}  "
              f"mean={row['mean']:+.3f} std={row['std']:.3f} robust={row['robust']:+.3f}{marker}")
    print(f"\nBest config (max robust/CVaR score): {best_config}")

    sh.hr("Step 4: frozen real-data evaluate_candidate (called exactly once)")
    result = sh.evaluate_candidate(best_config, "R118_novel_regimeswitch")
    sh.print_report(result)

    out = dict(
        branch="novel_regimeswitch_calibration",
        n_states=N_STATES,
        k_jump=K_JUMP,
        fitted_params=FITTED_PARAMS,
        n_draws_used=n_draws,
        n_draws_preregistered=sh.N_DRAWS,
        total_backtests_selection_sweep=n_draws * len(sh.GRID),
        timing=dict(sim_seconds=t_sim, backtest_seconds=t_bt,
                   sweep_wall_seconds=sweep_time),
        selection_table={f"{c[0]}_{c[1]}_{c[2]}": table[c] for c in sh.GRID},
        best_config=list(best_config),
        evaluate_candidate_result=result,
    )
    out_path = ROOT / "experiments" / "r118_novel_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved fitted params + selection table + evaluate_candidate result -> {out_path}")


if __name__ == "__main__":
    main()
