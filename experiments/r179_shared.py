"""Shared, read-only pre-registration engine for the R-179 round (08-29).

DIRECTION, one sentence: wrap `kelly_regime_v4`'s vote+scale signal in a
supervised meta-label (Lopez de Prado 2018) -- a secondary classifier fit
on triple-barrier-labeled historical outcomes -- and use its probability
to gate (conservative) or continuously size (novel) the primary signal,
rather than acting on the vote unconditionally. Full Step 1/Step 2 design,
citations and both branches' frozen falsification rules are in
`experiments/r179_direction.md`.

This module is DELIBERATELY neutral between the two branches: it exposes
`vote_frac`/`conditional_scale` (v4's own two factors, reproduced read-only
so neither branch re-derives them differently) and `walk_forward_meta_prob`
(the shared, causal, purged/embargoed walk-forward classifier) as three
primitives neither branch may edit -- mirroring r175-r178_shared.py's own
convention.

Mechanics of `walk_forward_meta_prob`, stated once so neither branch
re-derives it differently:

- Labels are built at a DAILY checkpoint (last bar of each UTC day), not at
  v4's own actual rebalance bars. v4 rebalances only ~150-280 times in nine
  years (its own docstring), which would leave single-digit samples per
  walk-forward refit -- untestable, and exactly the N~3 problem this
  ledger's standing diagnosis already names. Daily checkpoints give ~3,400
  labels over the full history instead. This is a disclosed design choice,
  not a free escape from N~3: `r179_direction.md` Step 1 Q4's falsification
  clause A fires if it does not actually produce enough resolved samples.
- Each daily checkpoint `t` is labeled by a triple barrier over the next
  `horizon_days` (Lopez de Prado 2018, ch. 3): profit-take and stop-loss set
  at `+-k * sigma_t * sqrt(horizon_days)` log-return from that checkpoint's
  close, `sigma_t` the same causal EWM realized-vol estimator v4 already
  uses (shifted, never using bar t's own return). label=1 if the upper
  barrier is hit first, or neither is hit and the terminal return is
  positive; label=0 otherwise (lower barrier hit first, or neither hit and
  terminal return <=0). Long-only, matching v4's own long-only exposure.
- A label at checkpoint `t` is RESOLVED at `t + horizon_days` (its own
  vertical barrier) plus an `embargo_days` buffer on top (Lopez de Prado
  2018 ch. 7's purge/embargo, applied here as a walk-forward-in-time
  boundary rather than a k-fold boundary, since live deployment is
  forward-only anyway: a label is never used by a fit whose "now" is
  earlier than the label's own resolution instant).
- The classifier is refit every `refit_days` on ALL labels resolved as of
  that refit instant (expanding window, not a fixed lookback), using
  features causal as of each label's own checkpoint bar (never later):
  `vol_ratio` (v4's own SCALE-state input, current/slow realized vol),
  `vote_strength` (|frac-0.5|*2, i.e. 0=a transitional/disagreeing vote
  state, 1=full anchor agreement), and `log1p(regime_duration_days)`
  (bars since the vote last changed, log-scaled). A ridge-penalized
  Newton-Raphson logistic regression (pure numpy -- this environment has no
  scipy/sklearn, R-118/R-125's finding re-confirmed here) is fit on the
  standardized features; the fitted model's probability for the CURRENT
  feature vector at each bar between refits is forward-filled from the
  refit instant, so no bar ever reads a probability from a model fit in
  its own future.
- Before the first refit has >= `min_samples` resolved labels, the
  probability is NaN -- both branches must define their own neutral
  (no-op) behaviour for that warmup stretch, disclosed in their own files.

Configs evaluated by this file: 0 (shared infrastructure only, per
R-163/R-168/R-178's convention -- each branch counts and reports its own).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# ------------------------------------------------------------- v4's own two factors


def vote_frac(close: pd.Series, horizons: tuple[int, ...] = (20, 40, 80),
              band: float = 0.01) -> np.ndarray:
    """`kelly_regime_v4`'s own 3-anchor latched vote, verbatim, in [0, 1].

    Reproduced read-only (identical construction to `kelly_regime.py`'s
    `prepare()`) purely so both branches condition on the vote v4 already
    computes without touching detection itself.
    """
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return (sum(votes) / len(votes)).to_numpy()


def conditional_scale(close: pd.Series, target_vol: float = 0.55, max_leverage: float = 2.0,
                       vol_span: int = 8 * BARS_PER_DAY, anchor_span_days: int = 180,
                       high_in: float = 1.70, high_out: float = 1.20,
                       low_in: float = 0.55, low_out: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
    """`kelly_regime_v3`/`v4`'s own conditional-volatility SCALE factor, verbatim.

    Returns ``(scale, vol_ratio)`` -- `vol_ratio` (current/slow realized
    vol) is also one of the meta-classifier's own features below, so it is
    exposed rather than recomputed a second time.
    """
    r = np.log(close).diff()
    vol = (r.ewm(span=vol_span, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    slow = (pd.Series(vol).ewm(span=anchor_span_days * BARS_PER_DAY,
                                min_periods=BARS_PER_DAY).mean().to_numpy())

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(target_vol / vol, max_leverage)
        steady = np.minimum(target_vol / slow, max_leverage)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    n = len(close)
    scale = np.zeros(n)
    state = 0
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > high_in else (-1 if x < low_in else 0)
            elif state == 1 and x < high_out:
                state = 0
            elif state == -1 and x > low_out:
                state = 0
        scale[i] = full[i] if state != 0 else steady[i]
    return scale, np.where(np.isfinite(ratio), ratio, 1.0)


# ------------------------------------------------------------- triple-barrier labels


def daily_checkpoints(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer positions of the last bar of every UTC day, ascending."""
    days = pd.Series(index.date, index=index)
    last_of_day = days.groupby(days).apply(lambda s: s.index[-1])
    idx = pd.Index(index)
    return np.sort(idx.get_indexer(last_of_day.to_numpy()))


def daily_triple_barrier_labels(close: np.ndarray, vol_daily: np.ndarray,
                                 checkpoints: np.ndarray, k: float = 1.0,
                                 horizon_days: int = 3) -> np.ndarray:
    """Long-only triple-barrier label at each daily checkpoint, in {0, 1, nan}.

    ``vol_daily`` is v4's own causal EWM annualized vol (already
    ``shift(1)``-ed, so `vol_daily[t]` uses no information at or after `t`).
    Barrier width is `k * sigma_t * sqrt(horizon_days)` in log-return space,
    `sigma_t` de-annualized from `vol_daily[t]`. label=1 if the upper
    barrier is hit before the lower one; if neither is hit within
    `horizon_days`, label = 1 if the terminal log-return is positive else 0.
    nan where the horizon runs past the end of the series or `vol_daily[t]`
    is not finite/positive.
    """
    n = len(close)
    h_bars = horizon_days * BARS_PER_DAY
    log_close = np.log(close)
    out = np.full(len(checkpoints), np.nan)
    for j, t in enumerate(checkpoints):
        sigma = vol_daily[t]
        if not (np.isfinite(sigma) and sigma > 0) or t + h_bars >= n:
            continue
        sigma_h = (sigma / np.sqrt(BARS_PER_YEAR / BARS_PER_DAY)) * np.sqrt(horizon_days)
        path = log_close[t + 1: t + 1 + h_bars] - log_close[t]
        up = k * sigma_h
        down = -k * sigma_h
        hit_up = np.argmax(path >= up) if np.any(path >= up) else -1
        hit_down = np.argmax(path <= down) if np.any(path <= down) else -1
        if hit_up == -1 and hit_down == -1:
            out[j] = 1.0 if path[-1] > 0 else 0.0
        elif hit_down == -1:
            out[j] = 1.0
        elif hit_up == -1:
            out[j] = 0.0
        else:
            out[j] = 1.0 if hit_up <= hit_down else 0.0
    return out


# ------------------------------------------------------------- pure-numpy logistic regression


def newton_logreg(X: np.ndarray, y: np.ndarray, ridge_lambda: float = 1.0,
                   iters: int = 25) -> np.ndarray:
    """Ridge-penalized logistic regression via Newton-Raphson, numpy-only.

    `X` excludes the intercept column (added here, unpenalized). Small
    feature counts (<=5) and iters=25 converge to numerical precision well
    within that budget for this problem size; ridge_lambda keeps the
    Hessian well-conditioned even in the low-sample early-refit regime.
    """
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(p + 1)
    penalty = ridge_lambda * np.eye(p + 1)
    penalty[0, 0] = 0.0  # do not penalize the intercept
    for _ in range(iters):
        z = Xb @ w
        z = np.clip(z, -30, 30)
        pr = 1.0 / (1.0 + np.exp(-z))
        grad = Xb.T @ (y - pr) - penalty @ w
        W = pr * (1.0 - pr)
        H = -(Xb.T * W) @ Xb - penalty
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        w = w - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def predict_logreg(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    z = np.clip(Xb @ w, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------- walk-forward driver


def walk_forward_meta_prob(index: pd.DatetimeIndex, close: np.ndarray, vol_daily: np.ndarray,
                            frac: np.ndarray, vol_ratio: np.ndarray, *,
                            k: float = 1.0, horizon_days: int = 3, refit_days: int = 60,
                            embargo_days: int = 3, min_samples: int = 50,
                            ridge_lambda: float = 1.0) -> tuple[np.ndarray, dict]:
    """The shared causal walk-forward classifier. Returns (prob_per_bar, diag).

    `prob_per_bar[i]` is this bar's forward-filled probability from the
    most recent refit whose fit set contains only labels already resolved
    strictly before that refit's own checkpoint -- no bar ever reads a
    model fit using its own future. `diag` reports, per refit, the sample
    count and the max |Wald z-score| among the non-intercept coefficients,
    which is exactly what `r179_direction.md`'s falsification clause A
    checks.
    """
    n = len(close)
    checkpoints = daily_checkpoints(index)
    labels = daily_triple_barrier_labels(close, vol_daily, checkpoints, k=k, horizon_days=horizon_days)

    duration_days = np.zeros(n)
    last_change = 0
    for i in range(1, n):
        if frac[i] != frac[i - 1]:
            last_change = i
        duration_days[i] = (i - last_change) / BARS_PER_DAY

    vote_strength = np.abs(frac - 0.5) * 2.0
    feat_full = np.column_stack([vol_ratio, vote_strength, np.log1p(duration_days)])

    cp_vol_ratio = vol_ratio[checkpoints]
    cp_vote_strength = vote_strength[checkpoints]
    cp_duration = np.log1p(duration_days[checkpoints])
    cp_features = np.column_stack([cp_vol_ratio, cp_vote_strength, cp_duration])

    valid = np.isfinite(labels) & np.all(np.isfinite(cp_features), axis=1)
    resolve_bar = checkpoints + horizon_days * BARS_PER_DAY + embargo_days * BARS_PER_DAY

    prob_per_bar = np.full(n, np.nan)
    refit_positions = list(range(0, n, refit_days * BARS_PER_DAY))
    diag = {"refits": 0, "n_at_refit": [], "max_abs_z": [], "clause_a_bad_refits": 0}

    mu = std = None
    w = None
    for start in refit_positions:
        end = min(start + refit_days * BARS_PER_DAY, n)
        eligible = valid & (resolve_bar <= start)
        n_eligible = int(eligible.sum())
        if n_eligible >= min_samples:
            Xtr = cp_features[eligible]
            ytr = labels[eligible]
            mu, std = Xtr.mean(axis=0), Xtr.std(axis=0)
            std = np.where(std > 1e-9, std, 1.0)
            Xtr_s = (Xtr - mu) / std
            w = newton_logreg(Xtr_s, ytr, ridge_lambda=ridge_lambda)
            Xb = np.hstack([np.ones((Xtr_s.shape[0], 1)), Xtr_s])
            pr = predict_logreg(w, Xtr_s)
            Wd = pr * (1.0 - pr)
            H = (Xb.T * Wd) @ Xb + ridge_lambda * np.eye(Xb.shape[1])
            try:
                cov = np.linalg.inv(H)
                se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
                z = w / se
                max_abs_z = float(np.max(np.abs(z[1:])))
            except np.linalg.LinAlgError:
                max_abs_z = 0.0
            diag["refits"] += 1
            diag["n_at_refit"].append(n_eligible)
            diag["max_abs_z"].append(max_abs_z)
            if max_abs_z < 1.0:
                diag["clause_a_bad_refits"] += 1
        if w is not None:
            Xlive = (feat_full[start:end] - mu) / std
            prob_per_bar[start:end] = predict_logreg(w, Xlive)
        else:
            diag["clause_a_bad_refits"] += 1

    return prob_per_bar, diag
