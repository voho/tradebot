#!/usr/bin/env python
"""R-80 (novel branch): a hand-rolled logistic meta-labeling confidence
vote, refit periodically and walk-forward, confirming (not braking)
`kelly_regime_v4`'s 3-anchor vote. Attacks ERR (no error control anywhere
in the signal path -- see docs/LEDGER.md's standing diagnosis).

=============================================================================
WHAT THIS FILE IS AND WHY (ROUTINE.md step 1-2, done before any number ran)
=============================================================================

Mechanism, one sentence: train a small causal secondary model to estimate
P(the primary vote's implied bet would have paid off | the vote's own
recent state and market context), and use that estimate as a CONFIRMING
vote (R-53/R-55's bidirectional architecture, `weight=0` recovers v4
exactly) rather than a brake, so a low-confidence period pulls exposure
toward -- not just down from -- the anchor vote, matching how a bounded
brake cannot (a brake can only ever reduce exposure).

Which constraint: ERR. v4 acts on its 3-anchor vote with full weight
regardless of whether that vote has recently been reliable -- there is no
secondary model anywhere in this signal path (docs/LEDGER.md's standing
diagnosis, "ERR" row).

Not a duplicate of R-04 ("Meta-labeling + triple barrier", 08-15, the
project's very first negative result): R-04's own diagnosis was "their
trend-scanning label looks *forward* and is inadmissible here" -- a
specific, named lookahead bug in the LABEL construction, not a refutation
of meta-labeling as a mechanism, and it predates every methodology tool
this repo now has. The fix, concretely: every training sample used at
refit time `t` has its label's forward-return window verified fully
resolved before `t`, with an explicit embargo on top (`EMBARGO_DAYS`
below) -- built with `tradebot.inference.group_bounds` /
`purged_train_mask` (purge+embargo, Lopez de Prado 2018 ch. 7), not
hand-waved. Step A below is itself a purged, embargoed cross-validation
using those exact primitives.

Not a duplicate of the 4-for-4 failed never-increase-only bounded brake
(R-34, R-41, R-53-conservative, R-73-conservative, docs/LEDGER.md): this
uses R-53/R-55's validated CONFIRMING-VOTE combination rule
(`r80_shared.confirming_vote_frac`), which can move exposure in EITHER
direction relative to the raw anchor vote, not a brake that can only
shrink it.

Taken seriously in advance (this round's primary suspected failure mode,
per the brief and per B-09 in docs/LEDGER.md, which demoted plain
conformal prediction on exactly this diagnosis): under N~=3 (effective
regime-event sample size), a secondary model may have too little
genuinely independent signal to learn anything real, and can look
calibrated while adding pure noise. Step A is designed to actually test
for this -- OUT-OF-FOLD discriminative skill against a label-permutation
null, not an in-sample fit metric, which would not distinguish a real
model from an intercept-only classifier exploiting the label's base rate.

Pre-registered falsification / stop rule (frozen BEFORE any Step-A number
was computed -- see `STEP_A_STOP_RULE` docstring below): proceed to Step B
(build the actual strategy) only if the purged, embargoed out-of-fold AUC
on inner-train clears the 95th percentile of a matched label-permutation
null. If it does not, STOP -- report the gate result as the round's
negative outcome and do not build a strategy just to have something to
show (this project's own convention: R-53/54/73/74/75/79's novel arms all
ran this exact pattern and R-75/R-79 both stopped here).

What would make this fail, named now: (a) the meta-model's OOF skill does
not clear the permutation null (B-09's predicted failure mode, tested
directly by Step A); (b) if it clears Step A, the live walk-forward
version still fails to beat v4 net of the exposure-artifact check
(R-33/R-37-family: does the win merely track higher average exposure,
R^2 > 0.9?); (c) the periodic-refit loop turns out to have a subtle
lookahead the simple per-feature truncation probe does not catch
(explicitly why a SECOND, bespoke truncation probe is run against the
refit loop's own OUTPUT below, not just the raw features).

=============================================================================
FEATURES (Step 2 design, causal at bar i, justified individually)
=============================================================================

Five features, all a function of data available at or before bar i only:

  f1  vote_frac      = anchor_sum(i) / 3, anchor_sum in {0,1,2,3}
      -- v4's own latched vote value (rolling mean + ffill/latch: row i
      depends only on rows <= i). The primary thing being evaluated for
      reliability -- is a 2/3 vote as trustworthy as a 3/3 vote?

  f2  vote_age_log   = log1p(days since anchor_sum last changed value)
      -- a strictly backward-looking counter (cumulative max of a
      lagging change marker: `np.maximum.accumulate`). Tests whether a
      freshly-flipped vote behaves differently from a long-held one
      (regime "staleness" as a confidence proxy).

  f3  vol_z          = (vol - roll_mean_90d(vol)) / roll_std_90d(vol)
      -- `vol` is byte-identical to kelly_regime_v3/v4's own realized-vol
      construction (EWM std of log returns, span=8 trading days,
      annualized, `.shift(1)` so bar i never uses its own bar's return).
      The z-score's own rolling window (90 days, `min_periods=30d`) is
      likewise backward-only. Tests whether the vote means something
      different when volatility is elevated vs depressed relative to its
      OWN recent history (R-3's lesson: in this market, vol bursts and
      big up-moves are not independent).

  f4  price_dev      = close(i) / roll_mean_40d(close) - 1
      -- v4's own MIDDLE anchor horizon (40 days), a rolling mean using
      only rows <= i. Overextension of price from the regime anchor the
      vote itself is built on -- a classic exhaustion/continuation proxy.

  f5  mom_5d         = log(close(i) / close(i - 5 days))
      -- a fixed-lag difference, causal by construction. Short-horizon
      continuation/reversal, independent of the slow vote's own horizons.

All five verified causal by (1) individual construction (rolling / ewm /
shift / cumulative-max only, no centered window, no full-series fit) and
(2) `r80_shared.truncation_causality_probe` run directly against the
feature-builder in `main()` below, before any model is fit on them.

=============================================================================
LABEL
=============================================================================

For a candidate training sample at day j (ONE sample per calendar day --
the first bar of each UTC day -- not every 5-minute bar, exactly as
instructed, both to control the overlapping-labels problem
`purged_train_mask` exists for and to keep refitting cheap):

    y[j] = 1  if log(close[j + H_DAYS] / close[j]) > 0   else 0

H_DAYS = 10, chosen before any label was computed: half of the anchor
ladder's SHORTEST horizon (20 days) -- long enough to be a real forward
bet horizon (not noise at the 5-minute-bar scale), short enough that a
6-year inner-train window still yields a four-figure number of daily
samples for a 5-feature logistic fit (measured below: ~1,360).

=============================================================================
STEP A -- pre-registered measurement gate (frozen before any number ran)
=============================================================================

STEP_A_STOP_RULE: fit a hand-rolled L2-penalized logistic regression (IRLS
/ Newton-Raphson -- no sklearn, per this project's convention) via purged,
embargoed K-fold cross-validation (`tradebot.inference.group_bounds` /
`purged_train_mask` / `fold_mask`; PRIMARY: n_groups=8, purge=embargo=
H_DAYS=10, L2=1.0), using ONLY inner-train data (bars < 2021-01-01,
`r80_shared.INNER_TRAIN_END`). Compute out-of-fold AUC (rank-based,
tie-averaged Mann-Whitney U construction -- no sklearn). Build a
label-permutation null: shuffle y (not X), refit and re-score through the
IDENTICAL purged K-fold procedure, 500 times. PROCEED TO STEP B ONLY IF
the true OOF AUC exceeds the null distribution's 95th percentile. This bar
is not relaxed after seeing the number. Robustness diagnostics (varying
n_groups, L2, and H_DAYS) are run AFTER the primary gate result and
reported alongside it, but do not change the frozen PRIMARY decision --
per ROUTINE.md step 4, moving the goalposts after looking would downgrade
the result to in-sample; checking whether a decision already made is
fragile does not.

=============================================================================
STEP B (only if Step A passes) -- periodic-refit walk-forward strategy
=============================================================================

Refit every REFIT_EVERY_DAYS=180 days (after an initial finer 30-day
search purely to find the FIRST viable refit sooner -- see
`_walk_forward_meta_vote`'s docstring for why), using an EXPANDING window
of (feature, label) pairs whose label window is fully resolved AND
embargoed (>= H_DAYS + EMBARGO_DAYS before the refit time). Frozen
coefficients + frozen feature standardization (mean/std from the SAME
training window -- never the full series, the specific lookahead this
project's own skeptic-review checklist calls out) score every subsequent
bar causally until the next refit. Before the first successful refit,
`weight` is forced to 0 for those bars (not merely "meta_vote defaults to
0.5 with weight left on" -- see the note in `_walk_forward_meta_vote` on
why that is the stronger, literally-no-effect reading of the brief's
instruction), so `frac` is bit-identical to v4's own `anchor_sum/3` until
a real model exists.

Run: ``python experiments/r80_novel_metalabel_logistic.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.inference import fold_mask, group_bounds, purged_train_mask  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
import r80_shared as shared  # noqa: E402
from scripts.experiment import DF, FUTURES, OOS_START, SPOT, ev  # noqa: E402

# ----------------------------------------------------------------- constants

H_DAYS = 10                 # label horizon (Step 2)
N_GROUPS_PRIMARY = 8        # purged K-fold groups, Step A primary config
L2_PRIMARY = 1.0            # logistic L2 penalty, Step A primary config
N_PERMUTATIONS = 500        # label-permutation null draws

FEATURE_LOOKBACK_DAYS = 90  # f3's own trailing window
MIN_TRAIN_DAYS = 180        # admissible daily samples required before a refit can fire
EMBARGO_DAYS = 10           # >= H_DAYS, per the brief
REFIT_EVERY_DAYS = 180      # Step B refit cadence (once a model exists)
WARMUP_DAYS = 300           # >= FEATURE_LOOKBACK_DAYS + MIN_TRAIN_DAYS + H_DAYS + EMBARGO_DAYS (290)


def assert_no_holdout(df: pd.DataFrame) -> None:
    """Mirrors R-75/R-79's own guard: never read a bar on/after OOS_START."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    assert df.index.max() < cutoff, (
        f"holdout bar read: max timestamp {df.index.max()} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


# ======================================================================
# Hand-rolled logistic regression (no sklearn, per project convention --
# see experiments/bayes_confidence.py, _stablecoin_signal.py). L2-penalized
# Newton-Raphson / IRLS; the intercept (column 0) is never penalized.
# ======================================================================

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                  max_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
    """L2-penalized logistic regression via Newton-Raphson (IRLS).

    ``X`` must already include an intercept column (column 0, all ones).
    Column 0 is excluded from the penalty. Cheap by design: this project's
    feature counts are a handful, never hundreds, so a dense Newton step
    every iteration is not a performance concern.
    """
    n, d = X.shape
    w = np.zeros(d)
    mask = np.ones(d)
    mask[0] = 0.0
    for _ in range(max_iter):
        p = _sigmoid(X @ w)
        grad = X.T @ (p - y) + l2 * mask * w
        wt = np.clip(p * (1.0 - p), 1e-8, None)
        hess = (X * wt[:, None]).T @ X + np.diag(l2 * mask)
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(hess, grad, rcond=None)[0]
        w_new = w - delta
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std computed on the given rows ONLY -- caller must pass a
    single training fold/window, never the full series (the exact
    lookahead this project's skeptic-review checklist calls out)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based (Mann-Whitney U) AUC, tie-averaged via pandas `.rank()`.

    Equivalent to sklearn's `roc_auc_score` for the binary case; no
    sklearn dependency (project convention).
    """
    y_true = np.asarray(y_true, dtype=float)
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ======================================================================
# Causal feature construction (shared by Step A and Step B)
# ======================================================================

def build_features(df: pd.DataFrame) -> np.ndarray:
    """The five causal features (f1..f5), full-length arrays aligned to df.

    Every column here is rolling / ewm / shift / cumulative-max: row i
    depends only on rows <= i. Verified directly by
    `r80_shared.truncation_causality_probe` in `main()`.
    """
    close = df["close"]
    r = np.log(close).diff()
    n = len(df)

    anchor_sum = sum(shared.anchor_votes(df)).to_numpy()  # in {0,1,2,3}
    f1 = anchor_sum / 3.0

    changed = np.zeros(n, dtype=bool)
    changed[1:] = anchor_sum[1:] != anchor_sum[:-1]
    idx_arr = np.arange(n)
    last_change = np.maximum.accumulate(np.where(changed, idx_arr, 0))
    f2 = np.log1p((idx_arr - last_change) / BARS_PER_DAY)

    vol = (r.ewm(span=8 * BARS_PER_DAY, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1)
    win = FEATURE_LOOKBACK_DAYS * BARS_PER_DAY
    vol_mean = vol.rolling(win, min_periods=30 * BARS_PER_DAY).mean()
    vol_std = vol.rolling(win, min_periods=30 * BARS_PER_DAY).std()
    f3 = ((vol - vol_mean) / vol_std).to_numpy()

    anchor_mid = close.rolling(40 * BARS_PER_DAY).mean()
    f4 = (close / anchor_mid - 1.0).to_numpy()

    f5 = np.log(close / close.shift(5 * BARS_PER_DAY)).to_numpy()

    return np.column_stack([f1, f2, f3, f4, f5])


def build_anchor_sum(df: pd.DataFrame) -> np.ndarray:
    return sum(shared.anchor_votes(df)).to_numpy()


def day_start_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """Positions (bar indices) of the first bar of each UTC calendar day."""
    dates = index.tz_convert("UTC").date if index.tz is not None else index.date
    dates = np.asarray(dates)
    mask = np.empty(len(dates), dtype=bool)
    mask[0] = True
    mask[1:] = dates[1:] != dates[:-1]
    return np.flatnonzero(mask)


def forward_label_positions(index: pd.DatetimeIndex, day_pos: np.ndarray,
                             horizon_days: float) -> np.ndarray:
    """Bar position of the first bar at/after `day_ts + horizon_days` for
    each day-start position; `len(index)` (out of range) where unresolved."""
    day_ts = index[day_pos]
    target_ts = day_ts + pd.Timedelta(days=horizon_days)
    return index.searchsorted(target_ts, side="left")


def build_daily_dataset(df: pd.DataFrame, features: np.ndarray,
                         horizon_days: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per UTC calendar day: (day_pos, X, y), dropping days whose
    forward-return window is not resolvable inside `df` or whose features
    are not yet finite (feature warm-up)."""
    close_np = df["close"].to_numpy()
    n = len(df)
    day_pos = day_start_positions(df.index)
    fwd_pos = forward_label_positions(df.index, day_pos, horizon_days)
    valid = fwd_pos < n
    fwd_ret = np.full(len(day_pos), np.nan)
    fwd_ret[valid] = np.log(close_np[fwd_pos[valid]] / close_np[day_pos[valid]])
    y_all = (fwd_ret > 0).astype(float)

    X_all = features[day_pos]
    feat_valid = valid & np.all(np.isfinite(X_all), axis=1) & np.isfinite(fwd_ret)
    return day_pos[feat_valid], X_all[feat_valid], y_all[feat_valid]


# ======================================================================
# STEP A: purged, embargoed out-of-fold cross-validation + permutation null
# ======================================================================

def purged_kfold_oof(X: np.ndarray, y: np.ndarray, n_groups: int,
                      purge: int, embargo: int, l2: float) -> np.ndarray:
    """Out-of-fold predicted probabilities via purged K-fold
    (`tradebot.inference.group_bounds`/`purged_train_mask`/`fold_mask`).

    Standardization is fit on the TRAINING fold only (never the full
    series) and applied to that fold's test rows -- the specific lookahead
    R-04 predates the tooling to prevent and this project's skeptic-review
    checklist calls out (a scaler fit on the whole series and applied to
    early rows).
    """
    n = len(y)
    bounds = group_bounds(n, n_groups)
    oof = np.full(n, np.nan)
    for g in range(n_groups):
        test_mask = fold_mask(n, bounds, (g,))
        train_mask = purged_train_mask(n, bounds, (g,), purge=purge, embargo=embargo)
        if train_mask.sum() < 20 or test_mask.sum() == 0:
            continue
        mu, sd = standardize_fit(X[train_mask])
        Xtr = add_intercept((X[train_mask] - mu) / sd)
        Xte = add_intercept((X[test_mask] - mu) / sd)
        w = fit_logistic(Xtr, y[train_mask], l2=l2)
        oof[test_mask] = _sigmoid(Xte @ w)
    return oof


def step_a_gate(X: np.ndarray, y: np.ndarray, n_groups: int, l2: float,
                 n_perm: int, seed: int) -> dict:
    oof = purged_kfold_oof(X, y, n_groups, purge=H_DAYS, embargo=H_DAYS, l2=l2)
    have = np.isfinite(oof)
    true_auc = auc_score(y[have], oof[have])
    true_acc = float(((oof[have] > 0.5).astype(float) == y[have]).mean())
    base_rate_acc = float(max(y.mean(), 1 - y.mean()))

    rng = np.random.default_rng(seed)
    null_auc = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        oof_p = purged_kfold_oof(X, yp, n_groups, purge=H_DAYS, embargo=H_DAYS, l2=l2)
        hp = np.isfinite(oof_p)
        null_auc[i] = auc_score(yp[hp], oof_p[hp])

    p95 = float(np.percentile(null_auc, 95))
    pval = float((null_auc >= true_auc).mean())
    return {
        "n_groups": n_groups, "l2": l2, "n_obs": int(have.sum()),
        "true_auc": true_auc, "true_acc": true_acc, "base_rate_acc": base_rate_acc,
        "null_mean": float(null_auc.mean()), "null_std": float(null_auc.std()),
        "null_p95": p95, "pvalue": pval, "pass": bool(true_auc > p95),
    }


# ======================================================================
# STEP B: periodic-refit walk-forward meta-vote
# ======================================================================

def walk_forward_meta_vote(index: pd.DatetimeIndex, close_np: np.ndarray,
                            features: np.ndarray, weight: float, *,
                            h_days: float = H_DAYS, embargo_days: float = EMBARGO_DAYS,
                            min_train_days: int = MIN_TRAIN_DAYS,
                            refit_every_days: int = REFIT_EVERY_DAYS, l2: float = L2_PRIMARY,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Frozen-between-refits logistic meta-vote, scored causally bar by bar.

    Refit schedule: a FINE 30-day grid is searched purely to find the
    first day on which enough admissible history exists to fit a first
    model at all (an isolated `ev(strategy, start=..., end=...)` call only
    gets `warmup` days of pre-period history -- see the module docstring
    and this file's report for the honest accounting of what that costs
    on an isolated inner-validation call); once a model exists, subsequent
    refit attempts are spaced >= `refit_every_days` apart, matching the
    "refit every ~180 days" spec. This is a warm-up accelerant only -- it
    changes WHEN the first refit can fire, never what data any refit is
    allowed to see.

    Causality, stated precisely: a refit attempted at day-index `ri` (bar
    position `day_pos[ri]`) may only use day-samples `j < ri` whose label
    window is fully resolved AND embargoed before that day's timestamp
    (`day_ts[j] + h_days + embargo_days <= day_ts[ri]`). The fitted
    (`w`, `mu`, `sd`) are then frozen and used to score bars
    `[day_pos[ri], next_refit)` -- every one of those bars uses ONLY its
    own already-causal `features[i]` row, never anything from the refit
    step itself beyond the frozen coefficients. `meta_vote` defaults to
    0.5 and `eff_weight` (the ACTUAL weight applied at each bar, distinct
    from the `weight` argument) defaults to 0.0 until the first refit
    succeeds, so `confirming_vote_frac(anchor_sum, meta_vote, eff_weight)`
    is bit-identical to v4's own `anchor_sum/3` before a real model
    exists -- the literal reading of "no effect", not merely "a neutral
    input to a nonzero weight" (which would still perturb `frac` slightly
    via `weight * 0.5` in the numerator).
    """
    n = len(close_np)
    meta_vote = np.full(n, 0.5)
    eff_weight = np.zeros(n)

    day_pos = day_start_positions(index)
    D = len(day_pos)
    fwd_pos = forward_label_positions(index, day_pos, h_days)
    day_ts = index[day_pos]
    resolved_ts = day_ts + pd.Timedelta(days=h_days + embargo_days)

    candidate_days = sorted(set(range(30, D, 30)) | set(range(refit_every_days, D, refit_every_days)))

    have_model = False
    last_refit_day = -10 ** 9
    for ri in candidate_days:
        if have_model and (ri - last_refit_day) < refit_every_days:
            continue
        refit_ts = day_ts[ri]
        admissible = (resolved_ts[:ri] <= refit_ts) & (fwd_pos[:ri] < n)
        adm_idx = np.flatnonzero(admissible)
        if len(adm_idx) < min_train_days:
            continue

        X_train = features[day_pos[adm_idx]]
        y_train = (np.log(close_np[fwd_pos[adm_idx]] / close_np[day_pos[adm_idx]]) > 0).astype(float)
        feat_ok = np.all(np.isfinite(X_train), axis=1)
        X_train, y_train = X_train[feat_ok], y_train[feat_ok]
        if len(y_train) < min_train_days or len(np.unique(y_train)) < 2:
            continue

        mu, sd = standardize_fit(X_train)
        w = fit_logistic(add_intercept((X_train - mu) / sd), y_train, l2=l2)

        refit_bar_pos = day_pos[ri]
        seg_feat = features[refit_bar_pos:]
        finite_rows = np.all(np.isfinite(seg_feat), axis=1)
        seg_z = np.where(finite_rows[:, None], (seg_feat - mu) / sd, 0.0)
        seg_scores = _sigmoid(add_intercept(seg_z) @ w)

        meta_vote[refit_bar_pos:] = np.where(finite_rows, seg_scores, 0.5)
        eff_weight[refit_bar_pos:] = np.where(finite_rows, weight, 0.0)

        have_model = True
        last_refit_day = ri

    return meta_vote, eff_weight


class KellyRegimeMetaLabel(Strategy):
    """v4's 3-anchor vote confirmed by a periodically-refit, purged/embargoed
    logistic meta-labeling confidence vote (ERR); `weight=0` reproduces
    `kelly_regime_v4` exactly.

    NOT registered (`@register` deliberately omitted) -- R-80 novel branch,
    experiments/-only per docs/ROUTINE.md. Everything except the `frac`
    line is copied verbatim from `kelly_regime_v3`/`kelly_regime_v4`
    (the doubling 20/40/80-day anchor ladder, conditional-volatility-target
    `scale` with its high/low hysteresis, the 10% deadband).
    """

    name = "kelly_regime_v80_metalabel"
    warmup = WARMUP_DAYS * BARS_PER_DAY + 10

    def __init__(self, weight: float = 1.0, horizons: tuple[int, ...] = (20, 40, 80),
                 band: float = 0.01, target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 vote_gamma: float = 1.0, anchor_span_days: int = 180,
                 high_in: float = 1.70, high_out: float = 1.20,
                 low_in: float = 0.55, low_out: float = 0.85,
                 h_days: float = H_DAYS, embargo_days: float = EMBARGO_DAYS,
                 min_train_days: int = MIN_TRAIN_DAYS, refit_every_days: int = REFIT_EVERY_DAYS,
                 l2: float = L2_PRIMARY) -> None:
        self.weight = weight
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.vote_gamma = vote_gamma
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        self.h_days = h_days
        self.embargo_days = embargo_days
        self.min_train_days = min_train_days
        self.refit_every_days = refit_every_days
        self.l2 = l2

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()
        n = len(df)

        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index)
            votes.append(v.ffill().fillna(0.0))
        anchor_sum = sum(votes).to_numpy()

        features = build_features(df)
        meta_vote, eff_weight = walk_forward_meta_vote(
            df.index, close.to_numpy(), features, self.weight,
            h_days=self.h_days, embargo_days=self.embargo_days,
            min_train_days=self.min_train_days, refit_every_days=self.refit_every_days,
            l2=self.l2)

        frac = shared.confirming_vote_frac(anchor_sum, meta_vote, eff_weight)
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                    min_periods=BARS_PER_DAY).mean().to_numpy())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        target = np.zeros(n)
        pos = 0.0
        state = 0
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["meta_vote"] = meta_vote
        df["eff_weight"] = eff_weight
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# ======================================================================
# main
# ======================================================================

def main() -> None:
    n_configs = 0
    print("=" * 78)
    print("R-80 novel branch: Step A measurement gate (logistic meta-labeling)")
    print("=" * 78)

    df_train = DF.loc[:shared.INNER_TRAIN_END].copy()
    assert_no_holdout(df_train)
    print(f"\nInner-train: {df_train.index.min()} -> {df_train.index.max()} "
          f"({len(df_train):,} bars)")

    # ---- causality probe on the raw feature builder, before any model ----
    # truncation_causality_probe expects a 1-D target; probe each of the 5
    # feature columns separately via a thin per-column wrapper.
    print("\n--- causality probe: feature builder, 5 columns x 2 check points ---")
    feat_names = ["f1_vote_frac", "f2_vote_age", "f3_vol_z", "f4_price_dev", "f5_mom5d"]
    for j, fname in enumerate(feat_names):
        def col_fn(frame: pd.DataFrame, _j=j) -> np.ndarray:
            return build_features(frame)[:, _j]
        for check_at in (250_000, 350_000):
            ok = shared.truncation_causality_probe(col_fn, df_train, check_at)
            print(f"  {fname:14s} check_at={check_at:>7d}: "
                  f"{'PASS (causal)' if ok else 'FAIL (LOOKAHEAD)'}")
            assert ok, f"{fname} is not causal -- stop, do not trust Step A"

    features_train = build_features(df_train)
    day_pos, X, y = build_daily_dataset(df_train, features_train, H_DAYS)
    D = len(y)
    print(f"\nDaily samples (H_DAYS={H_DAYS}): D={D}, base rate y=1: {y.mean():.4f}")

    print(f"\n--- Step A PRIMARY: n_groups={N_GROUPS_PRIMARY}, purge=embargo={H_DAYS}d, "
          f"L2={L2_PRIMARY}, {N_PERMUTATIONS} permutations ---")
    t0 = time.time()
    primary = step_a_gate(X, y, N_GROUPS_PRIMARY, L2_PRIMARY, N_PERMUTATIONS, seed=8053)
    n_configs += 1
    print(f"  OOF n={primary['n_obs']}  true AUC={primary['true_auc']:.4f}  "
          f"acc={primary['true_acc']:.4f}  base_rate_acc={primary['base_rate_acc']:.4f}")
    print(f"  null AUC: mean={primary['null_mean']:.4f} std={primary['null_std']:.4f} "
          f"p95={primary['null_p95']:.4f}")
    print(f"  true AUC {primary['true_auc']:.4f} vs null p95 {primary['null_p95']:.4f} -> "
          f"{'EXCEEDS' if primary['pass'] else 'does NOT exceed'}, "
          f"empirical p={primary['pvalue']:.4f}  [{time.time()-t0:.1f}s]")

    print("\n--- Step A robustness diagnostics (run AFTER the primary gate; "
          "do not change the frozen PRIMARY decision) ---")
    for ng, l2v in [(6, 1.0), (10, 1.0), (8, 0.3), (8, 3.0), (8, 10.0)]:
        r = step_a_gate(X, y, ng, l2v, 150, seed=9001 + ng * 10 + int(l2v * 10))
        n_configs += 1
        print(f"  n_groups={ng:2d} l2={l2v:4.1f}: true AUC={r['true_auc']:.4f} "
              f"null_p95={r['null_p95']:.4f} -> {'PASS' if r['pass'] else 'fail'}")

    print("\n--- Step A horizon diligence (H=5, H=20; NOT part of the frozen "
          "gate, reported honestly as a fragility check, not re-selected on) ---")
    for h_alt in (5, 20):
        day_pos_h, X_h, y_h = build_daily_dataset(df_train, features_train, h_alt)
        r = step_a_gate(X_h, y_h, N_GROUPS_PRIMARY, L2_PRIMARY, 150, seed=1234 + h_alt)
        n_configs += 1
        print(f"  H={h_alt:2d}d: D={len(y_h)} base_rate={y_h.mean():.3f} "
              f"true AUC={r['true_auc']:.4f} null_p95={r['null_p95']:.4f} -> "
              f"{'PASS' if r['pass'] else 'fail'}")

    gate_pass = primary["pass"]
    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE (frozen before any Step-A number was computed):")
    print("  proceed to Step B only if the PRIMARY purged/embargoed OOF AUC exceeds")
    print("  the label-permutation null's 95th percentile.")
    print(f"  GATE: {'PASS -> proceed to Step B' if gate_pass else 'FAIL -> STOP, report negative'}")
    print("=" * 78)

    if not gate_pass:
        print(f"\nTotal Step-A configurations evaluated: {n_configs}")
        print("Stopping here per the pre-registered rule. No strategy code is built.")
        return

    # ================================================================
    # STEP B -- only reached if the gate passed
    # ================================================================
    print("\n" + "=" * 78)
    print("STEP B: periodic-refit walk-forward strategy")
    print("=" * 78)

    # ---- identity check: weight=0 must reproduce kelly_regime_v4 exactly ----
    cand0 = KellyRegimeMetaLabel(weight=0.0)
    v4 = KellyRegimeV4()
    t_cand = cand0.prepare(df_train.copy())["target"].to_numpy()
    t_v4 = v4.prepare(df_train.copy())["target"].to_numpy()
    max_diff = float(np.max(np.abs(t_cand - t_v4)))
    print(f"\nIdentity check (weight=0 vs kelly_regime_v4), inner-train: "
          f"max abs diff = {max_diff:.3e}")
    df_val_check = DF.loc[:"2022-12-31 23:55:00"].copy()
    assert_no_holdout(df_val_check)
    t_cand2 = cand0.prepare(df_val_check.copy())["target"].to_numpy()
    t_v42 = v4.prepare(df_val_check.copy())["target"].to_numpy()
    max_diff2 = float(np.max(np.abs(t_cand2 - t_v42)))
    print(f"Identity check (weight=0 vs kelly_regime_v4), inner-train+validation: "
          f"max abs diff = {max_diff2:.3e}")

    # ---- bespoke causality probe on the REFIT LOOP's own output ----
    print("\n--- bespoke causality probe: full refit-loop meta_vote output "
          "(3 check points past warmup) ---")

    def build_meta_vote_w1(frame: pd.DataFrame) -> np.ndarray:
        feats = build_features(frame)
        mv, _ = walk_forward_meta_vote(frame.index, frame["close"].to_numpy(), feats, weight=1.0)
        return mv

    for check_at in (150_000, 300_000, 400_000):
        ok = shared.truncation_causality_probe(build_meta_vote_w1, df_val_check, check_at)
        print(f"  check_at={check_at:>7d} ({df_val_check.index[check_at]}): "
              f"{'PASS (causal)' if ok else 'FAIL (LOOKAHEAD)'}")

    # ---- diagnostics: when does the meta-model actually go live? ----
    print("\n--- diagnostic: first refit timing (meta_vote goes live) ---")
    feats_train = build_features(df_train)
    mv_train, ew_train = walk_forward_meta_vote(df_train.index, df_train["close"].to_numpy(),
                                                 feats_train, weight=1.0)
    first_live = np.flatnonzero(ew_train > 0)
    if len(first_live):
        print(f"  inner-train-only call: first live bar at "
              f"{df_train.index[first_live[0]]} (bar {first_live[0]:,}/{len(df_train):,})")
    n_step_b = 0

    # ---- sweep: weight, plus a small refit-cadence check ----
    print("\n" + "=" * 78)
    print("Step B sweep (inner-train and inner-validation, spot and 5x futures)")
    print("=" * 78)

    from tradebot.registry import get_strategy  # noqa: E402  (local import, avoids polluting Step A)

    v4_default = KellyRegimeV4()
    v4_matched = KellyRegimeV4()
    v4_matched.warmup = KellyRegimeMetaLabel().warmup  # fairness control, see report
    bh = get_strategy("buy_and_hold")

    def run_both_splits(strategy, tag):
        nonlocal n_step_b
        n_step_b += 1
        print(f"\n[{tag}]")
        ev(strategy, market=SPOT, tag=f"{tag:32s} spot IS ", end=shared.INNER_TRAIN_END)
        ev(strategy, market=FUTURES, tag=f"{tag:32s} fut IS ", end=shared.INNER_TRAIN_END)
        ev(strategy, market=SPOT, tag=f"{tag:32s} spot VAL",
           start=shared.INNER_VAL_START, end=shared.INNER_VAL_END)
        ev(strategy, market=FUTURES, tag=f"{tag:32s} fut VAL",
           start=shared.INNER_VAL_START, end=shared.INNER_VAL_END)

    print("\n--- baselines ---")
    run_both_splits(bh, "buy_and_hold")
    run_both_splits(v4_default, "kelly_regime_v4 (default warmup)")
    run_both_splits(v4_matched, "kelly_regime_v4 (matched warmup)")

    print("\n--- candidate: weight sweep (refit_every_days=180 fixed) ---")
    for w in (0.5, 1.0, 2.0):
        run_both_splits(KellyRegimeMetaLabel(weight=w), f"metalabel weight={w}")

    print("\n--- candidate: refit-cadence check at weight=1.0 (plateau-vs-peak) ---")
    for cadence in (90, 365):
        run_both_splits(KellyRegimeMetaLabel(weight=1.0, refit_every_days=cadence),
                         f"metalabel w=1.0 refit={cadence}d")

    print(f"\nTotal Step-B configurations evaluated: {n_step_b} "
          f"(x4 cells each: spot/futures x inner-train/inner-validation)")
    print(f"Total Step-A configurations evaluated: {n_configs}")

    # ---- exposure-profile diagnostic: does the candidate ever go flat? ----
    print("\n--- exposure-profile diagnostic (inner-train, post-warmup bars) ---")
    print("  Does the confirming-vote formula preserve v4's own most robust, "
          "ledger-documented property -- full de-risk to cash on unanimous "
          "bearish consensus (anchor_sum=0)? ")
    for w in (0.5, 1.0, 2.0):
        c = KellyRegimeMetaLabel(weight=w)
        t = c.prepare(df_train.copy())["target"].to_numpy()[c.warmup:]
        print(f"  weight={w}: mean|target|={np.mean(np.abs(t)):.4f}  "
              f"frac_time_at_flat(<1e-6)={np.mean(t < 1e-6):.4f}")
    t_v4m = v4_matched.prepare(df_train.copy())["target"].to_numpy()[v4_matched.warmup:]
    print(f"  v4 (matched warmup): mean|target|={np.mean(np.abs(t_v4m)):.4f}  "
          f"frac_time_at_flat(<1e-6)={np.mean(t_v4m < 1e-6):.4f}")


if __name__ == "__main__":
    main()
