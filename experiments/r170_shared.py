"""Shared, read-only pre-registration for the R-170 round (08-28).

DIRECTION, one sentence: give `kelly_regime_v4`'s already-decided
`frac * scale` exposure a multiplicative meta-labeling confidence layer
(Lopez de Prado 2018, *Advances in Financial Machine Learning*, ch. 3) --
a purged/embargoed, out-of-fold-validated probability that THIS
particular vote-flip event will resolve profitably net of fees -- applied
strictly as `desired' = frac * scale * m`, `m in [0,1]`, ahead of v4's own
unmodified 10% deadband. Neither the vote nor the volatility-target scale
is touched by either branch.

Full Step 1/Step 2 design (constraint attacked, non-duplication against
R-04/R-80/the bounded-brake family/R-87/R-160/R-161/R-162/R-141/R-163/
B-09/R-100/R-145, simulability, named failure modes, noise-floor sanity
check) is in `docs_scratch_direction.md` at the repo root, written by the
research sub-agent that proposed this round and reviewed by the operator
before this file was frozen. Read that file for the full argument; this
module is the FROZEN, executable form of it -- written by the operator
BEFORE either branch is dispatched, and neither branch may edit it or
each other's file (R-89-through-R-168's own convention).

**Two disclosed deviations from that design doc, made HERE, before either
branch is dispatched, per ROUTINE.md's allowance to fix a bug/tighten a
spec before freezing (never to loosen one after looking at a result):**

1. **Purge/embargo is applied in BAR-TIME directly, not via
   `tradebot.inference.purged_train_mask`/`fold_mask`.** Those primitives
   assume one training ROW per regularly-spaced unit (R-80 used one row
   per calendar day, so "10 days" purge == "10 rows"). This round's
   training rows are vote-FLIP events, which are irregularly spaced in
   time (v4 makes ~150-260 trades over the whole history per README), so
   "N rows of purge" and "N days of purge" are not the same quantity.
   `purged_kfold_time` below reimplements the identical purge/embargo
   LOGIC (exclude any training sample whose [entry, resolution] interval
   overlaps the test fold's bar-time window, expanded by `purge`/`embargo`
   bars on each side) directly against each sample's own entry/resolution
   BAR POSITION, which is the correct generalization for irregular
   sampling and collapses to the standard row-count version when sampling
   is in fact regular.
2. **Feature availability is asset-conditional, decided by DATA COVERAGE
   alone, never by a performance number.** BTC has funding, DVOL and
   OHLCV; ETH has DVOL and OHLCV but NO funding series anywhere in this
   project's committed data. `feature_matrix()` drops a column for a
   given dataframe if fewer than `MIN_FEATURE_COVERAGE` (30%) of its
   post-warmup bars are finite -- an objective, pre-registered,
   data-availability rule applied identically regardless of what a model
   fit on the surviving columns would show, so the ETH falsification
   test trains a (necessarily poorer) 3-feature model on ETH rather than
   silently never firing at all (0 admissible ETH training samples would
   make the ETH check vacuous, not a real falsification test).
3. **Step-A's own gate and every periodic refit fit on bars < OOS_START
   (inner-train + inner-validation combined), not on inner-train alone.**
   Checked empirically before freezing anything else: DVOL starts
   2021-03-24, which is INSIDE inner-validation (2021-2022), not
   inner-train (2017-2020) -- an inner-train-only Step-A gate would see
   `vrp` coverage of exactly 0.0% and never test it at all. ROUTINE.md
   Step 3's own table names inner-validation a training resource ("select
   between variants... as often as you like, it's a training resource"),
   and `compare()` itself already scores both inner slices, so fitting a
   classifier on the same combined non-holdout window is not a new
   liberty taken by this round. Measured before dispatch (BTC,
   bars < OOS_START, post-warmup): funding_z coverage 51.4%, vrp 30.6%
   (just above MIN_FEATURE_COVERAGE), illiq/excursion 100%. On the ETH
   replication file (`ethusd_bitfinex_5m.csv.gz`, which ends 2019-12-31 --
   entirely before both funding and DVOL exist), funding_z and vrp
   coverage are 0.0% regardless of window: the ETH branch necessarily
   trains a 2-feature (illiq, excursion) model. Disclosed, not proxied
   around -- see `STEP_A_END` below.

============================================================================
FEATURES -- four causal channels, none derived from the anchors v4's vote
or scale already read (design doc Step 1(b), distinguishing this from
R-162's Kaufman-ER and R-141/R-163's tanh dampeners, which all failed the
A2 non-collinearity kill switch computing statistics of `close` alone).
============================================================================

  funding_z   causal z-score of Binance+Deribit-extended daily-summed
              perpetual funding against its own trailing 30-day history
              (`tradebot.data.load_funding_extended`). BTC only; NaN for
              ETH (no ETH funding data exists in this project).
  vrp         Deribit DVOL (30-day implied vol) minus v4's own trailing
              realized volatility (`experiments.r102_shared.v4_symmetric_vol`,
              reused, not recomputed differently). BTC from 2021-03-24,
              ETH from its own DVOL file's own start (checked at runtime,
              not assumed identical to BTC's).
  illiq       Amihud (2002) illiquidity: |bar log return| / dollar volume,
              trailing 20-day rolling mean, `.shift(1)` so bar i never uses
              its own forming bar. Available on every asset (native OHLCV).
  excursion   Episode-relative price excursion in ATR units since the
              current bullish episode began (`r163_shared.bullish_episode_state`
              + `r163_shared.atr_n`, reused verbatim -- R-163's own SIZE-axis
              state variable, used here as a classifier FEATURE rather than,
              as R-163 built it, a hand-set multiplier). Available on every
              asset.

============================================================================
LABEL -- triple-barrier meta-label at each non-flat vote-flip event
============================================================================

A "flip" is any bar where v4's own `vote_frac` (`r161_shared.v4_vote_frac`,
in {0, 1/3, 2/3, 1}) changes value -- finer-grained than a bullish-episode
boundary (frac crossing 0.5), so most flips are NOT episode starts and the
excursion feature is not systematically ~0 at the sampled bars. Flips TO
flat (new frac == 0) are excluded -- there is no forward "position" whose
profitability can be evaluated. For each remaining flip at bar `i`
(entry = close[i]):

  - horizontal barriers: entry*(1+FEE), entry*(1-FEE);
  - vertical barrier: `min(i + H_DAYS, next_flip_bar, n-1)` -- the
    earliest of the fixed horizon or the NEXT flip of any kind (matching
    the design doc's "next opposing flip" vertical barrier, generalized to
    "next flip" since v4 is long-only and has no opposing/short side);
  - label = 1 if the upper barrier is touched first, 0 if the lower is
    touched first, else 1 iff `close` at the vertical barrier exceeds
    entry (this already nets the label against FEE via the horizontal
    barriers' own placement, matching "profitable net of the fee").

============================================================================
STEP-0 GATES -- mandatory, run before any strategy backtest, IDENTICAL for
both branches (design doc S2's "mandatory Step-0 gate"):
============================================================================

  (i)   identity-recovery: m===1 must reproduce kelly_regime_v4 exactly
        (`identity_check` below).
  (ii)  exposure-artifact kill switch: R^2 of the candidate's raw
        pre-deadband `frac*scale*m` path against v4's own unmodified
        `frac*scale`, on inner-train, must be < R2_KILL_THRESH (0.98) --
        else this is the fifth confirmation of the bounded-`[0,1]`-
        multiplier collapse-to-flat-rescale pattern (R-34/R-41/R-53-cons/
        R-73-cons), not a tested mechanism.
  (iii) Step-A gate: purged/embargoed out-of-fold AUC on bars < STEP_A_END
        (deviation #3) must exceed the 95th percentile of a 500-draw
        label-permutation null (`step_a_gate` below, reusing R-80's
        validated design). A branch
        that fails (ii) or (iii) STOPS before Step B -- no strategy code
        is run past that point, per R-166's own precedent.

Configs evaluated by this file: 0 (shared infrastructure only; each
branch's own count is logged in its own module and summed in the ledger
entry, per R-163/R-168's convention).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r102_shared import v4_symmetric_vol  # noqa: E402
from experiments.r161_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    ETH_SLICE_NAME,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SLICES,
    SPOT,
    TargetStrategy,
    V4_BAND,
    V4_HORIZONS,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_series,
    compare,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_target,
    v4_vote_frac,
)
from experiments.r163_shared import atr_n, bullish_episode_state  # noqa: E402
from tradebot.data import align_dvol_causal, load_dvol_index, load_funding_extended  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert abs(V4_BAND - 0.01) < 1e-12, V4_BAND

DATA_DIR = ROOT / "data"
ETH_DVOL_FILE = "eth_dvol_daily.csv.gz"

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch was dispatched.
# ------------------------------------------------------------------------
H_DAYS = 10                        # label horizon (half v4's shortest 20d anchor)
EMBARGO_DAYS = 10                  # >= H_DAYS
STEP_A_END = INNER_VAL_END         # Step-A gate + refit training window: bars < OOS_START
                                    # (inner-train + inner-validation combined -- deviation #3)
REFIT_DAYS_GRID = (90, 180, 365)   # conservative branch's own sweep
REFIT_DAYS_PRIMARY = 180
N_GROUPS_PRIMARY = 8               # purged K-fold groups, Step-A primary
L2_PRIMARY = 1.0                   # logistic L2 penalty
N_PERMUTATIONS = 500               # label-permutation null draws
MIN_TRAIN_FLIPS = 20               # minimum admissible flip-events before ANY refit
FUNDING_Z_WINDOW_DAYS = 30
ILLIQ_WINDOW_DAYS = 20
MIN_FEATURE_COVERAGE = 0.30        # asset-conditional column-drop threshold (data-driven only)
FEE_TIER_PRIMARY = 0.0010          # nets the meta-label itself (spot default)
FEE_TIER_STRESS = 0.0040           # cost-robustness re-run, both branches
SHARPE_NOISE_FLOOR = 0.2           # ROUTINE.md's own promotion bar (R-20)
R2_KILL_THRESH = 0.98              # Step-0(ii) exposure-artifact kill switch
AUC_GATE_SEED = 17021
FEATURE_NAMES = ("funding_z", "vrp", "illiq", "excursion")

# ==========================================================================
# (1) Causal feature builders. Each returns a full-length array aligned to
#     df's index; NaN wherever the underlying data does not yet exist.
# ==========================================================================


def _causal_zscore(s: pd.Series, window_days: int) -> pd.Series:
    mean = s.rolling(window_days, min_periods=max(5, window_days // 3)).mean()
    std = s.rolling(window_days, min_periods=max(5, window_days // 3)).std()
    return (s - mean) / std.replace(0.0, np.nan)


def _align_daily_causal(daily: pd.Series, bars: pd.DataFrame) -> pd.Series:
    """A bar at time T may only see the row for the most recent day that
    closed strictly before T's own day -- identical shift+ffill convention
    to `tradebot.data.align_dvol_causal` / every `align_*_causal` helper
    in this project."""
    shifted = daily.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    return shifted.reindex(shifted.index.union(bars.index)).sort_index().ffill().reindex(bars.index)


def funding_z_feature(df: pd.DataFrame) -> pd.Series:
    """Causal z-score of daily-summed extended funding vs its own trailing
    30-day history. All-NaN when no funding file is present (ETH)."""
    rate, _source = load_funding_extended(DATA_DIR)
    if rate is None:
        return pd.Series(np.nan, index=df.index)
    daily = rate.resample("1D").sum()
    z = _causal_zscore(daily, FUNDING_Z_WINDOW_DAYS)
    return _align_daily_causal(z, df)


def _load_dvol(filename: str) -> pd.DataFrame | None:
    path = DATA_DIR / filename
    if not path.exists():
        return None
    d = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    d.index = d.index.tz_localize("UTC") if d.index.tz is None else d.index
    return d.astype(float).sort_index()


def vrp_feature(df: pd.DataFrame, asset: str = "BTC") -> pd.Series:
    """DVOL/100 minus v4's own trailing realized vol -- variance risk
    premium. All-NaN before the asset's own DVOL history starts."""
    dvol = load_dvol_index(DATA_DIR) if asset == "BTC" else _load_dvol(ETH_DVOL_FILE)
    if dvol is None:
        return pd.Series(np.nan, index=df.index)
    dvol_aligned = align_dvol_causal(dvol, df)["close"] / 100.0
    realized = pd.Series(v4_symmetric_vol(df), index=df.index)
    return dvol_aligned - realized


def illiq_feature(df: pd.DataFrame, window_days: int = ILLIQ_WINDOW_DAYS) -> pd.Series:
    """Amihud (2002) illiquidity, trailing rolling mean, `.shift(1)` causal."""
    close = df["close"]
    r = np.log(close).diff()
    dollar_vol = (close * df["volume"]).replace(0.0, np.nan)
    bar_illiq = (r.abs() / dollar_vol)
    window_bars = window_days * BARS_PER_DAY
    return bar_illiq.rolling(window_bars, min_periods=BARS_PER_DAY).mean().shift(1)


def excursion_feature(df: pd.DataFrame) -> pd.Series:
    """Episode-relative excursion in ATR units since the current bullish
    episode began (r163_shared's own state variable, reused verbatim)."""
    bullish, entry_price = bullish_episode_state(df)
    close = df["close"].to_numpy(dtype=float)
    N = atr_n(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        exc = np.where(N > 0, (close - entry_price) / N, np.nan)
    return pd.Series(exc, index=df.index)


def coverage_fraction(s: pd.Series, warmup: int) -> float:
    tail = s.iloc[warmup:]
    return float(np.isfinite(tail.to_numpy()).mean()) if len(tail) else 0.0


def feature_matrix(df: pd.DataFrame, asset: str = "BTC", warmup: int = 0,
                    ) -> tuple[np.ndarray, list[str]]:
    """All four causal features, columns with < MIN_FEATURE_COVERAGE
    finite post-warmup values DROPPED (asset-conditional, data-driven
    only -- see module docstring deviation #2). Returns (X, kept_names)."""
    cols = {
        "funding_z": funding_z_feature(df),
        "vrp": vrp_feature(df, asset=asset),
        "illiq": illiq_feature(df),
        "excursion": excursion_feature(df),
    }
    kept = [name for name in FEATURE_NAMES if coverage_fraction(cols[name], warmup) >= MIN_FEATURE_COVERAGE]
    X = np.column_stack([cols[name].to_numpy() for name in kept]) if kept else np.zeros((len(df), 0))
    return X, kept


# ==========================================================================
# (2) Triple-barrier meta-label at each non-flat vote-flip event.
# ==========================================================================


def flip_events(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Bar positions where v4's own vote_frac changes value, and the new
    value at each -- finer-grained than a bullish-episode boundary."""
    frac = v4_vote_frac(df).to_numpy()
    n = len(frac)
    changed = np.zeros(n, dtype=bool)
    changed[1:] = frac[1:] != frac[:-1]
    pos = np.flatnonzero(changed)
    return pos, frac[pos]


def triple_barrier_labels(df: pd.DataFrame, h_days: float = H_DAYS,
                           fee: float = FEE_TIER_PRIMARY,
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(entry_bar, label, resolved_bar) at each non-flat flip event.

    label = 1 if the +fee horizontal barrier is touched before the -fee
    one; 0 if the reverse; else 1 iff close at the vertical barrier
    (min(entry+h_days, next flip, end of data)) exceeds entry. See module
    docstring for the full construction."""
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    all_pos, all_frac = flip_events(df)
    horizon_bars = int(h_days * BARS_PER_DAY)
    out_pos, out_label, out_resolved = [], [], []
    for k, i in enumerate(all_pos):
        if all_frac[k] <= 0:
            continue
        next_flip = int(all_pos[k + 1]) if k + 1 < len(all_pos) else n
        end = min(i + horizon_bars, next_flip - 1, n - 1)
        if end <= i:
            continue
        entry = close[i]
        upper, lower = entry * (1.0 + fee), entry * (1.0 - fee)
        window = close[i + 1:end + 1]
        hit_up = np.flatnonzero(window >= upper)
        hit_dn = np.flatnonzero(window <= lower)
        t_up = hit_up[0] if len(hit_up) else None
        t_dn = hit_dn[0] if len(hit_dn) else None
        if t_up is not None and (t_dn is None or t_up <= t_dn):
            label, resolved = 1.0, i + 1 + int(t_up)
        elif t_dn is not None:
            label, resolved = 0.0, i + 1 + int(t_dn)
        else:
            label, resolved = (1.0 if close[end] > entry else 0.0), end
        out_pos.append(i)
        out_label.append(label)
        out_resolved.append(resolved)
    return np.array(out_pos, dtype=int), np.array(out_label, dtype=float), np.array(out_resolved, dtype=int)


# ==========================================================================
# (3) Hand-rolled logistic regression (no sklearn, project convention;
#     verbatim from r80_novel_metalabel_logistic.py's own proven code).
# ==========================================================================


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                  max_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
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
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ==========================================================================
# (4) Purged K-fold in BAR-TIME (deviation #1 -- see module docstring).
# ==========================================================================


def purged_kfold_time(entry_bar: np.ndarray, resolved_bar: np.ndarray, n_bars: int,
                       n_groups: int, purge_bars: int, embargo_bars: int,
                       ) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_mask, test_mask) over the SAMPLE array (len == len(entry_bar))
    for each of n_groups contiguous bar-time folds. A sample is purged from
    training if its [entry_bar, resolved_bar] interval overlaps the test
    fold's bar-time window expanded by purge/embargo on each side."""
    edges = np.linspace(0, n_bars, n_groups + 1).astype(int)
    folds = []
    for g in range(n_groups):
        lo, hi = int(edges[g]), int(edges[g + 1])
        test_mask = (entry_bar >= lo) & (entry_bar < hi)
        excl_lo, excl_hi = lo - purge_bars, hi + embargo_bars
        overlap = (resolved_bar >= excl_lo) & (entry_bar < excl_hi)
        train_mask = ~test_mask & ~overlap
        folds.append((train_mask, test_mask))
    return folds


def purged_kfold_oof(X: np.ndarray, y: np.ndarray, entry_bar: np.ndarray,
                      resolved_bar: np.ndarray, n_bars: int, n_groups: int,
                      purge_bars: int, embargo_bars: int, l2: float,
                      ) -> np.ndarray:
    n = len(y)
    oof = np.full(n, np.nan)
    for train_mask, test_mask in purged_kfold_time(entry_bar, resolved_bar, n_bars,
                                                     n_groups, purge_bars, embargo_bars):
        if train_mask.sum() < 15 or test_mask.sum() == 0 or len(np.unique(y[train_mask])) < 2:
            continue
        mu, sd = standardize_fit(X[train_mask])
        Xtr = add_intercept((X[train_mask] - mu) / sd)
        Xte = add_intercept((X[test_mask] - mu) / sd)
        w = fit_logistic(Xtr, y[train_mask], l2=l2)
        oof[test_mask] = _sigmoid(Xte @ w)
    return oof


def step_a_gate(X: np.ndarray, y: np.ndarray, entry_bar: np.ndarray, resolved_bar: np.ndarray,
                 n_bars: int, n_groups: int, purge_bars: int, embargo_bars: int, l2: float,
                 n_perm: int, seed: int) -> dict:
    """Step-A pre-registered measurement gate: purged OOF AUC vs a
    label-permutation null (R-80's own validated design, reused)."""
    oof = purged_kfold_oof(X, y, entry_bar, resolved_bar, n_bars, n_groups, purge_bars, embargo_bars, l2)
    have = np.isfinite(oof)
    true_auc = auc_score(y[have], oof[have])
    rng = np.random.default_rng(seed)
    null_auc = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        oof_p = purged_kfold_oof(X, yp, entry_bar, resolved_bar, n_bars, n_groups,
                                  purge_bars, embargo_bars, l2)
        hp = np.isfinite(oof_p)
        null_auc[i] = auc_score(yp[hp], oof_p[hp]) if hp.sum() else float("nan")
    null_auc = null_auc[np.isfinite(null_auc)]
    p95 = float(np.percentile(null_auc, 95)) if len(null_auc) else float("nan")
    pval = float((null_auc >= true_auc).mean()) if len(null_auc) else float("nan")
    return {
        "n_groups": n_groups, "l2": l2, "n_obs": int(have.sum()),
        "true_auc": true_auc, "null_mean": float(null_auc.mean()) if len(null_auc) else float("nan"),
        "null_p95": p95, "pvalue": pval,
        "pass": bool(np.isfinite(true_auc) and np.isfinite(p95) and true_auc > p95),
    }


# ==========================================================================
# (5) Meta-multiplier application, identity check, decision predicate.
# ==========================================================================


def apply_meta_multiplier(df: pd.DataFrame, m: np.ndarray) -> np.ndarray:
    """desired' = frac * scale * clip(m, 0, 1), v4's own unmodified deadband."""
    m = np.clip(np.asarray(m, dtype=float), 0.0, 1.0)
    raw = v4_raw_desired(df) * m
    return apply_deadband(raw)


def identity_check(df: pd.DataFrame) -> float:
    """m===1 everywhere must reproduce kelly_regime_v4 exactly. Returns max abs diff."""
    cand = apply_meta_multiplier(df, np.ones(len(df)))
    ctrl = v4_target(df)
    return float(np.max(np.abs(cand - ctrl)))


def exposure_artifact_r2(df: pd.DataFrame, m: np.ndarray) -> float:
    """Step-0(ii): R^2 of the candidate's raw pre-deadband path vs v4's own."""
    m = np.clip(np.asarray(m, dtype=float), 0.0, 1.0)
    cand_raw = v4_raw_desired(df) * m
    return r_squared(cand_raw, v4_raw_desired(df))


def clears_bar(row: dict, sharpe_floor: float = SHARPE_NOISE_FLOOR) -> bool:
    """ROUTINE.md's own promotion-bar predicate on one compare() row."""
    if row["d_log_growth"] > 0 and row["excludes_zero"]:
        return True
    if row["d_sharpe"] >= sharpe_floor:
        return True
    if row["risk_matched"] and row["d_dd"] < 0:
        return True
    return False
