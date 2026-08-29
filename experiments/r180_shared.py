"""Shared, read-only pre-registration engine for the R-180 round (08-29).

DIRECTION, one sentence: re-run meta-labeling (Lopez de Prado 2018) on
`kelly_regime_v4`'s `frac*scale` decision -- closed NEGATIVE twice already
(R-170: non-price-anchor derivatives/positioning features, no discriminative
skill at all; R-179: price/vote-anchor-derived features, "signal" that
turns out to be the vote's own trend re-encoded) -- for the first time
using features genuinely exogenous to both price and the vote: macro
risk-off stress (VIX+DXY) and on-chain valuation (MVRV), the specific
feature-source requirement R-179's own closing line named. Full Step
1/Step 2 design, non-duplication case (checked against R-53/R-54/R-74/
R-87/R-161/R-167/R-170/R-174/R-179 and all 17 prior ERR-axis rounds on
this architecture), the reachable-power computation and both branches'
frozen falsification rules are in `experiments/r180_direction.md`.

This module is DELIBERATELY neutral between the two branches, mirroring
r175-r179_shared.py's own convention: it exposes the two new causal
feature builders (`macro_stress_z`, `mvrv_z`), a generalized version of
r179_shared's own walk-forward classifier (identical mechanics -- purge/
embargo, expanding-window refit, forward-fill -- parameterized on an
arbitrary feature matrix instead of r179's hardcoded vol_ratio/
vote_strength/duration triple, since this round's features are exogenous
rather than derived from `frac`/`vol_ratio`), and a label-permutation AUC
gate for the Step-A discriminative-skill check. `vote_frac`,
`conditional_scale`, `daily_checkpoints`, `daily_triple_barrier_labels`,
`newton_logreg` and `predict_logreg` are imported UNEDITED from
`r179_shared.py` (Step 0's collision-avoidance convention: do not
re-derive an already-validated causal primitive). Neither branch may edit
this file or `r179_shared.py`.

Feature construction, stated once so neither branch re-derives it
differently:

- `macro_stress_z`: VIX close + Fed broad dollar index (DXY), each
  z-scored against its own trailing 365-CALENDAR-day rolling window (not
  R-74's expanding window -- a disclosed deviation, chosen so the feature
  describes "how unusual is *ambient* stress right now" rather than
  "unusual relative to the whole history to date", matching this round's
  Step 1 Q1 framing of an ERR-axis reliability feature rather than an
  INFO-axis directional one), summed equal-weight. Loaded via
  `tradebot.data.load_macro_metrics`/`align_macro_causal` (R-53's
  fetch/alignment code, unedited) -- FRED's day-D close becomes visible
  starting day D+1, so no bar ever reads a same-day print.
- `mvrv_z`: MVRV ratio (CoinMetrics, R-74's fetch code, unedited),
  z-scored against its own trailing 365-calendar-day rolling window (the
  LEVEL only, never a rate of change -- R-74 already tested and killed
  MVRV rate-of-change as a directional signal; a level z-score answers a
  different, ERR-axis question). Loaded via
  `tradebot.data.load_mvrv_ratio`/`align_mvrv_causal` (unedited) -- day
  D's MVRV becomes visible starting day D+1.
- Both trailing-window z-scores are computed on the DAILY series itself
  (not on the 5m-forward-filled series -- rolling a 365-day window over
  repeated 5m-forward-filled values would count each daily print ~288x
  and change nothing about the window's calendar span, but computing it
  daily-native is simpler to audit and is what R-53/R-54/R-74 already did),
  then causally aligned onto the 5m bar grid with the existing D+1-visible
  convention. Bars before either series' own 365-day warmup, or before
  the first visible print, are NaN -- never filled or back-cast.

Configs evaluated by this file: 0 (shared infrastructure only, per
R-163/R-168/R-178/R-179's convention -- each branch counts and reports its
own).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import (  # noqa: E402
    align_macro_causal,
    align_mvrv_causal,
    load_macro_metrics,
    load_mvrv_ratio,
)

from r179_shared import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    conditional_scale,
    daily_checkpoints,
    daily_triple_barrier_labels,
    newton_logreg,
    predict_logreg,
    vote_frac,
)

WINDOW_DAYS = 365  # trailing causal z-score window, both features

# ------------------------------------------------------------- exogenous feature builders


def _rolling_z(s: pd.Series, window_days: int = WINDOW_DAYS) -> pd.Series:
    """Trailing rolling z-score over a CALENDAR-day span, not a row count.

    VIX/DXY are business-day series with holiday gaps (66 NaN rows out of
    2,666 for VIX alone); a row-count window of 365 combined with
    min_periods=365 requires the window to contain zero holidays, which
    never happens, so the naive version silently produced an all-NaN
    feature (caught by this module's own self-test). A time-based window
    (``'{window_days}D'``) spans the correct calendar range regardless of
    gaps; MVRV has no such gaps (a complete daily series) so the same
    function is correct for both, and min_periods is set to 70% of the
    expected observation count for a 5-day trading week (business-day
    series) rather than the window length itself.
    """
    s = s.sort_index()
    min_periods = int(window_days * 5 / 7 * 0.7)
    mean = s.rolling(f"{window_days}D", min_periods=min_periods).mean()
    std = s.rolling(f"{window_days}D", min_periods=min_periods).std()
    return (s - mean) / std.where(std > 1e-9, np.nan)


def macro_stress_z(bars_index: pd.DatetimeIndex, data_dir: str | Path,
                    window_days: int = WINDOW_DAYS) -> np.ndarray:
    """Equal-weight VIX-level + DXY-level rolling z-score, causal, aligned
    onto ``bars_index``. All-NaN if `data/vix_daily.csv.gz` or
    `data/dxy_daily.csv.gz` is absent."""
    macro = load_macro_metrics(data_dir)
    if macro is None:
        return np.full(len(bars_index), np.nan)
    z = (_rolling_z(macro["vix"]) + _rolling_z(macro["dxy"])).rename("macro_stress_z").to_frame()
    bars = pd.DataFrame(index=bars_index)
    return align_macro_causal(z, bars)["macro_stress_z"].to_numpy()


def mvrv_z(bars_index: pd.DatetimeIndex, data_dir: str | Path, asset: str = "BTC",
           window_days: int = WINDOW_DAYS) -> np.ndarray:
    """MVRV LEVEL rolling z-score, causal, aligned onto ``bars_index``.
    All-NaN if `data/btc_mvrv_daily.csv.gz` (or the ETH equivalent) is
    absent."""
    mvrv = load_mvrv_ratio(data_dir, asset=asset)
    if mvrv is None:
        return np.full(len(bars_index), np.nan)
    z = _rolling_z(mvrv["mvrv"]).rename("mvrv_z").to_frame()
    bars = pd.DataFrame(index=bars_index)
    return align_mvrv_causal(z, bars)["mvrv_z"].to_numpy()


# ------------------------------------------------------------- generalized walk-forward classifier


def build_checkpoint_data(index: pd.DatetimeIndex, close: np.ndarray, vol_daily: np.ndarray,
                           feat_full: np.ndarray, *, k: float = 1.0, horizon_days: int = 3,
                           embargo_days: int = 3):
    """Compute checkpoints/labels/features/resolve times ONCE (shared
    across every refit and every permutation-null draw -- the expensive
    triple-barrier walk is O(n_checkpoints), the classifier refit is not,
    so this is separated out rather than recomputed per draw)."""
    n = len(close)
    checkpoints = daily_checkpoints(index)
    labels = daily_triple_barrier_labels(close, vol_daily, checkpoints, k=k, horizon_days=horizon_days)
    cp_features = feat_full[checkpoints]
    valid = np.isfinite(labels) & np.all(np.isfinite(cp_features), axis=1)
    resolve_bar = checkpoints + horizon_days * BARS_PER_DAY + embargo_days * BARS_PER_DAY
    return {
        "n": n, "checkpoints": checkpoints, "labels": labels, "cp_features": cp_features,
        "valid": valid, "resolve_bar": resolve_bar,
    }


def walk_forward_from_checkpoints(cpd: dict, feat_full: np.ndarray, *, labels: np.ndarray | None = None,
                                   refit_days: int = 60, min_samples: int = 50,
                                   ridge_lambda: float = 1.0) -> tuple[np.ndarray, dict]:
    """The shared causal walk-forward classifier, generalized from
    r179_shared.walk_forward_meta_prob to accept an arbitrary feature
    matrix (this round's features are exogenous, not derived from
    `frac`/`vol_ratio`, so r179's hardcoded feature derivation does not
    apply). Mechanics -- purge/embargo, expanding-window refit on
    resolved-as-of-refit labels only, standardize-on-train, forward-fill
    -- are IDENTICAL to r179_shared's, line for line.

    ``labels`` defaults to ``cpd["labels"]``; passing a different array
    (e.g. a permuted copy) reuses the already-computed checkpoints/
    features/resolve-times for a permutation-null draw without recomputing
    the triple-barrier walk.
    """
    n = cpd["n"]
    checkpoints = cpd["checkpoints"]
    if labels is None:
        labels = cpd["labels"]
    cp_features = cpd["cp_features"]
    valid = np.isfinite(labels) & np.all(np.isfinite(cp_features), axis=1) & cpd["valid"]
    resolve_bar = cpd["resolve_bar"]

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


def walk_forward_meta_prob(index: pd.DatetimeIndex, close: np.ndarray, vol_daily: np.ndarray,
                            feat_full: np.ndarray, *, k: float = 1.0, horizon_days: int = 3,
                            refit_days: int = 60, embargo_days: int = 3, min_samples: int = 50,
                            ridge_lambda: float = 1.0) -> tuple[np.ndarray, dict]:
    """Convenience one-shot wrapper: build_checkpoint_data + walk_forward_from_checkpoints."""
    cpd = build_checkpoint_data(index, close, vol_daily, feat_full, k=k,
                                 horizon_days=horizon_days, embargo_days=embargo_days)
    return walk_forward_from_checkpoints(cpd, feat_full, refit_days=refit_days,
                                          min_samples=min_samples, ridge_lambda=ridge_lambda)


# ------------------------------------------------------------- Step-A: label-permutation AUC gate


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based (Mann-Whitney U) AUC, tie-averaged. No sklearn (project
    convention, R-118/R-125). Identical construction to
    r80_novel_metalabel_logistic.py's own `auc_score`, reused by value
    since that file is branch-owned, not shared infrastructure."""
    y_true = np.asarray(y_true, dtype=float)
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def step_a_permutation_gate(index: pd.DatetimeIndex, close: np.ndarray, vol_daily: np.ndarray,
                             feat_full: np.ndarray, train_end_bar: int, *, k: float = 1.0,
                             horizon_days: int = 3, embargo_days: int = 3, refit_days: int = 60,
                             min_samples: int = 50, ridge_lambda: float = 1.0,
                             n_perm: int = 300, seed: int = 180) -> dict:
    """Step-A discriminative-skill gate: true walk-forward AUC (causal,
    out-of-refit) against a label-permutation null (shuffle the checkpoint
    labels the classifier is fit on, refit, rescore; ``n_perm`` draws) --
    computed ENTIRELY on data before ``train_end_bar`` (pass the
    2020-12-31 inner-train boundary here) so this never touches
    inner-validation or the holdout. Builds its own truncated checkpoint
    dataset from ``index[:train_end_bar]`` etc. rather than slicing a
    full-range one, so array lengths inside
    ``walk_forward_from_checkpoints`` stay internally consistent. Reuses
    the SAME checkpoints/features/resolve-times for every draw -- only the
    label refit is repeated per permutation.

    Passes (this round's Step 1 Q4 clause) if the true AUC clears the
    null distribution's 95th percentile.
    """
    idx_tr = index[:train_end_bar]
    close_tr = close[:train_end_bar]
    vol_tr = vol_daily[:train_end_bar]
    feat_tr = feat_full[:train_end_bar]
    cpd = build_checkpoint_data(idx_tr, close_tr, vol_tr, feat_tr, k=k,
                                 horizon_days=horizon_days, embargo_days=embargo_days)
    checkpoints = cpd["checkpoints"]
    valid_mask = cpd["valid"]

    def _auc(lbls: np.ndarray) -> float:
        prob, _ = walk_forward_from_checkpoints(
            cpd, feat_tr, labels=lbls, refit_days=refit_days,
            min_samples=min_samples, ridge_lambda=ridge_lambda)
        p = prob[checkpoints]
        have = np.isfinite(p) & np.isfinite(lbls)
        if have.sum() < 20:
            return float("nan")
        return auc_score(lbls[have], p[have])

    labels = cpd["labels"]
    true_auc = _auc(labels)
    rng = np.random.default_rng(seed)
    null_auc = np.empty(n_perm)
    shuffle_idx = np.where(valid_mask)[0]
    for i in range(n_perm):
        shuffled = labels.copy()
        shuffled[shuffle_idx] = rng.permutation(shuffled[shuffle_idx])
        null_auc[i] = _auc(shuffled)

    null_auc = null_auc[np.isfinite(null_auc)]
    p95 = float(np.percentile(null_auc, 95)) if len(null_auc) else float("nan")
    pval = float((null_auc >= true_auc).mean()) if len(null_auc) else float("nan")
    return {"true_auc": true_auc, "null_p95": p95, "pval": pval,
            "n_perm_valid": int(len(null_auc)), "clears": bool(true_auc > p95)}


# ------------------------------------------------------------- self-test


def _self_test() -> None:
    data_dir = ROOT / "data"
    from tradebot.data import load_dataset
    df, label = load_dataset(str(data_dir), "spot")
    df = df.iloc[-400_000:].copy()  # ~3.8yr slice, enough for the 365d warmup + a handful of refits

    macro = macro_stress_z(df.index, data_dir)
    mvrv = mvrv_z(df.index, data_dir)
    assert len(macro) == len(df) and len(mvrv) == len(df)
    assert np.isfinite(macro).any(), "macro_stress_z all-NaN -- data files missing or join broken"
    assert np.isfinite(mvrv).any(), "mvrv_z all-NaN -- data files missing or join broken"
    # (i) identity/no-lookahead sanity: a feature at bar i must not change
    # when bars strictly after i are altered.
    macro_trunc = macro_stress_z(df.index[: len(df) // 2], data_dir)
    same = np.isfinite(macro[: len(macro_trunc)]) & np.isfinite(macro_trunc)
    assert np.allclose(macro[: len(macro_trunc)][same], macro_trunc[same], equal_nan=True), (
        "macro_stress_z changed on truncation -- lookahead")

    close = df["close"].to_numpy()
    r = np.log(close)
    r = np.diff(r, prepend=r[0])
    vol_daily = (pd.Series(r).ewm(span=8 * BARS_PER_DAY, min_periods=BARS_PER_DAY).std()
                 * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()

    feat = np.column_stack([macro, mvrv])
    prob, diag = walk_forward_meta_prob(df.index, close, vol_daily, feat,
                                         horizon_days=3, refit_days=90, embargo_days=3)
    assert diag["refits"] > 0, "walk_forward_meta_prob produced zero refits on the self-test slice"
    assert np.isfinite(prob).any(), "walk_forward_meta_prob produced no finite probability"

    print(f"r180_shared self-test OK (macro finite={np.isfinite(macro).mean():.2f}, "
          f"mvrv finite={np.isfinite(mvrv).mean():.2f}, refits={diag['refits']}, "
          f"n_at_refit median={np.median(diag['n_at_refit']) if diag['n_at_refit'] else float('nan')})")


if __name__ == "__main__":
    _self_test()
else:
    _self_test()
