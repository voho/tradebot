#!/usr/bin/env python
"""R-94 CONSERVATIVE branch: Wikipedia "Bitcoin" pageview attention as a
confirming vote on `kelly_regime_v4`'s 3-anchor gate, via R-53/R-55's
already-validated `confirming_vote_frac` combination rule -- Step A
measurement gate first, this project's established discipline for every
INFO-axis round since R-53 (R-53/R-73/R-74/R-79/R-81/R-84/R-88).

=====================================================================
PRE-REGISTRATION (frozen before any lead/lag number in this file was
computed -- docs/ROUTINE.md steps 1-2/4, and the operator's dispatch
prompt for this branch). If anything below is later contradicted by what
actually happened, that is stated in the results section, not edited
back into this banner.
=====================================================================

1. MECHANISM (one sentence, fixed by the dispatch prompt, not chosen by
   this branch). Da, Engelberg & Gao (2011, J. Finance 66(5)) find
   abnormal retail search/attention predicts short-horizon return
   CONTINUATION -- a price move crossing one of `kelly_regime_v4`'s
   anchors is a more trustworthy confirming vote when it co-occurs with
   unusually elevated Wikipedia "Bitcoin" pageview attention than on
   ordinary readership. Full citation set, the not-a-duplicate-of
   argument against the eleven prior INFO-axis signals, and the data-
   coverage verification are in `experiments/r94_shared.py`'s module
   docstring -- frozen, not re-derived here (this branch's own dispatch
   instructions are explicit that this is not this branch's job).

   Constraint attacked: INFO. Architecture: reuses R-53/R-55's validated
   CONFIRMING-VOTE combination rule (`confirming_vote_frac`), exactly as
   the dispatch prompt specifies -- not a new architecture, so none of
   the four independent never-increase-only-BRAKE failures (R-34, R-41,
   R-53-conservative, R-73-conservative) apply here.

2. STEP A -- THE MANDATORY MEASUREMENT GATE, run BEFORE any strategy
   code, on the FULL R-82/R-83/R-84 six-episode table
   (`r94_shared.STRESS_EPISODES`), using `r94_shared`'s own frozen
   `nearest_transition` (v4's own anchor-vote flip, first bar at/after
   onset where `anchor_majority` crosses 0.5 from either side),
   `nearest_crossing` (first bar at/after onset where `attention_z`
   crosses above `thresh`, direction="above"), `episode_window`
   (onset - 60d, onset + 30d, r94_shared's own default), and
   `episode_null_leads` (500-draw, 5-day-block circular block-bootstrap
   of the LOCAL attention_z array, recomputing the crossing against the
   SAME fixed, real flip time).

   PRIMARY FEATURE: `r94_shared.attention_z(views_5m, window_days)` --
   causal log-pageview z-score against its own trailing `window_days`
   mean/std (necessary given pageviews' own secular 2017-2026 growth
   trend; a fixed level threshold would conflate "unusual today" with
   "the encyclopedia has grown", exactly r94_shared's own stated
   reasoning for building the feature this way).

   GRID (fixed now, before any lead number in this file was computed;
   this is a measurement-gate sensitivity sweep, not a search for a
   passing cell -- every cell is reported, none is dropped):
   tau (threshold) in {1.0, 1.5, 2.0} x window_days in {10, 20, 30},
   9 cells x 6 episodes = 54 episode-lead measurements. No `ev()` call,
   no strategy backtest, in any Step-A cell -- per this project's
   standing accounting convention (R-53/R-73/R-74/R-79/R-81/R-84), Step
   A's own configuration count is 0 for the deflated-Sharpe trials
   ledger; the 9 grid cells are gate diagnostics, not backtests.

   PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed, exactly as specified in this branch's dispatch prompt): an
   episode PASSES iff (a) LEAD = (flip_time - crossing_time) in days is
   > 0 (attention crossed before the gate's own nearest reaction), AND
   (b) LEAD exceeds the 90th percentile of that same episode's own
   500-draw block-bootstrap null lead distribution. PROCEED TO STEP B
   ONLY IF >= 4 of 6 episodes PASS on some reasonable parameterization,
   and "reasonable" is read conservatively here: an isolated single
   passing cell out of 9, surrounded by failing neighbours, is read as
   noise from running 9 gate cells (the dispatch prompt's own explicit
   "not a search for a passing cell -- report all cells honestly, do
   not cherry-pick" instruction), not as grounds to proceed. A genuine
   pass needs a PLATEAU of cells clearing the bar, not one lucky cell.
   If no reasonable parameterization clears >= 4/6 (plateau reading):
   STOP, report the gate result as this branch's whole product, do not
   build any strategy or backtest code -- this project's own established
   convention for a Step-A failure (R-53/R-73/R-74/R-79/R-81's own
   INFO-axis history: 0 of 11 prior signals led the anchor gate).

3. WHAT WOULD MAKE STEP A FAIL, named now: the same failure every one of
   the 11 prior INFO-axis signals in this ledger hit -- attention
   extremity is reached AFTER (not before) the anchor gate's own nearest
   reaction, or a positive lead (if it occurs in a minority of episodes
   or cells) is not distinguishable from an arbitrary time-shift of the
   same series (i.e. generic autocorrelation/persistence structure in a
   slow-moving z-score, not a genuine early-warning property). Given the
   base rate (0 of 11 prior INFO signals led), the modal, pre-registered
   outcome IS failure, and a clean negative is this branch's fully
   successful, complete product if that is what happens.

4. STEP B -- CONTINGENT PRE-REGISTRATION (design frozen now, BEFORE Step
   A's numbers exist; only executed if Step A's stop rule passes).

   CONFIRMING-VOTE CONSTRUCTION, modelled directly on
   `r84_conservative_volume_confirm.py`'s `VolumeConfirmKelly` (same
   round-template family, same gate, per this branch's own dispatch
   instructions): `confirming_vote_frac` (imported unchanged from
   `r94_shared.py`) requires a DISCRETE {0,1} `meta_vote` (R-80's lesson,
   preserved). Attention itself carries no direction, so the vote's
   direction comes from v4's own FASTEST (20-day) anchor vote -- the
   most reactive anchor v4 has, matching the mechanism's own "a price
   move crossing an anchor" framing: `meta_vote[i] = fast_anchor_vote[i]`
   on any bar where `attention_z[i] >= tau` ("confirmed"); otherwise
   `meta_vote[i] = meta_vote[i-1]` (carries forward, R-53/R-54/R-55's
   hysteresis-latch pattern, keyed on an attention gate instead of a
   level threshold-band). Before the first confirmed bar, defaults to
   the fast anchor's own then-current value (no dilution while
   unconfirmed, matching R-84's identical default).

   frac = confirming_vote_frac(anchor_sum, meta_vote, weight)
        = (anchor_sum + weight * meta_vote) / (3 + weight)

   `weight=0` must recover `kelly_regime_v4` bit-for-bit (identity
   check, run first, before any swept configuration).

   SWEEP GRID (fixed a priori, not tuned to any inner-validation number,
   modelled on R-84's own grid shape): `weight` in {0.5, 1.0, 2.0, 4.0}
   x `window_days` (attention_z's own trailing baseline) in {10, 20, 30}
   -- 12 configurations, at the Step-A primary `tau`. `tau` sensitivity
   at the pre-registered primary point (weight=1.0, window_days=20) over
   the other two grid values of `tau` -- 2 configurations. Identity check
   (weight=0): 1 configuration. Total Step B sweep configurations, if
   reached: 15 (matching R-84's own count, by construction, for a
   like-for-like comparison).

   MANDATORY CHECKS (this project's standing discipline for every
   confirming-vote round that reaches Step B): (i) exposure-artifact R^2
   -- regress the candidate's `target` series against a mean-notional-
   matched flat rescale of `kelly_regime_v4`'s own `target` series on
   inner-validation, both markets; R^2 > 0.95 = "just a rescale" (this
   project's "flat-rescale artifact"), fail; (ii) ETH falsification,
   frozen rule applied unmodified to `ethusd_coinbase_spot_5m.csv.gz` per
   this branch's dispatch instructions (disclosed here rather than
   switched to the Bitfinex pair R-84/R-77 used, because pageviews are a
   BTC-specific Wikipedia article with no natural ETH analogue in this
   round's own data -- the falsification asks whether the CONSTRUCTION,
   applied to a different coin's price action gated by the SAME BTC
   attention series, still improves on v4's own analogous gate; this is
   a weaker, differently-shaped test than R-84's ETH-specific-signal
   check, and is reported as such rather than silently reframed as
   equivalent); (iii) truncation causality probe on the attention-
   confirmed meta-vote construction, implemented locally in this file
   (r94_shared.py does not ship one; same construction as R-84's).

   PRE-REGISTERED HOLDOUT DECISION RULE (fixed now, contingent on
   reaching Step B and everything above clearing): read the 2023+
   holdout ONLY IF ALL of (a) the pre-registered primary configuration's
   inner-validation Sharpe improvement over `kelly_regime_v4` exceeds the
   +/-0.2 noise floor (R-20) on BOTH markets, on a genuine parameter
   PLATEAU (neighbouring grid cells, not an isolated peak); (b) exposure-
   artifact R^2 <= 0.95 on both markets; (c) the ETH falsification
   replicates the same sign of edge (not decisively reversed); (d) the
   truncation causality probe passes on all checkpoints. If ANY of these
   fail, this branch reports NEGATIVE and the holdout is never read -- an
   honest negative at any stage is this project's own definition of a
   complete, successful piece of work.

5. CONFIGS EVALUATED IN STEP A: 0 for the backtest trials ledger (9 grid
   cells x 6 episodes = 54 gate diagnostics, no `ev()` calls -- this
   project's standing accounting convention, R-53/R-73/R-74/R-79/R-81/
   R-84). Step B's count, if reached, is 15 as itemized above, plus 1 for
   the identity check already counted in that 15.
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

from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.data import load_wikipedia_pageviews, align_wikipedia_causal  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

from experiments.r94_shared import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    STRESS_EPISODES,
    V4_BAND,
    V4_HORIZONS,
    anchor_majority,
    anchor_votes,
    attention_z,
    confirming_vote_frac,
    episode_null_leads,
    episode_window,
    nearest_crossing,
    nearest_transition,
)

DATA_DIR = ROOT / "data"

# ---- Step A constants, fixed a priori (see banner item 2) -----------------
TAU_GRID = (1.0, 1.5, 2.0)
WINDOW_DAYS_GRID = (10, 20, 30)
TAU_PRIMARY = 1.5
WINDOW_DAYS_PRIMARY = 20
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 93
MIN_EPISODES_PASS = 4  # of 6, majority


# --------------------------------------------------------------------- data

def assert_no_holdout(df, oos_start: str = OOS_START) -> None:
    """Hard guard: the max timestamp in any frame this file touches must be
    strictly before `oos_start`. Same convention as r79/r81/r84's own guard.
    r94_shared.py does not ship one (its loaders leave truncation to the
    caller), so it is implemented locally here."""
    if len(df) == 0:
        return
    idx = df.index if hasattr(df, "index") else df
    cutoff = pd.Timestamp(oos_start, tz=idx.tz)
    max_ts = idx.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {oos_start}. "
        "This file must never read data on or after the holdout start.")


def load_btc_bars() -> pd.DataFrame:
    """BTC spot, truncated strictly before OOS_START at load time -- before
    the Wikipedia pageviews are ever aligned onto it, so the holdout is
    never touched even transiently."""
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def load_pageviews_aligned(bars: pd.DataFrame) -> pd.Series:
    """Daily Wikipedia pageviews, causally aligned onto `bars`' own
    (already holdout-truncated) index. Deliberately does NOT use
    `r94_shared.load_pageviews_5m` here, because that helper aligns onto
    the FULL committed bar file (which extends past OOS_START) before the
    caller gets a chance to truncate -- truncating `bars` first, as this
    function does, means the pageviews series itself is never aligned
    against any post-OOS_START timestamp at all."""
    pv = load_wikipedia_pageviews(DATA_DIR)
    assert pv is not None, "Wikipedia pageviews file missing from data/"
    aligned = align_wikipedia_causal(pv, bars)
    assert_no_holdout(aligned)
    return aligned["views"]


# --------------------------------------------------------------------- gate

def gate_cell(bars: pd.DataFrame, majority: pd.Series, views_5m: pd.Series,
              tau: float, window_days: int, verbose: bool = True) -> list[dict]:
    """One (tau, window_days) grid cell: run all 6 episodes, return their
    per-episode result dicts."""
    z = attention_z(views_5m, window_days=window_days)
    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str)
        if len(window) == 0:
            results.append(dict(label=label, onset=onset_str, lead=float("nan"),
                                 pass_b=False, null_p90=float("nan")))
            continue

        flip_time = nearest_transition(majority, window, onset)
        cross_time = nearest_crossing(z, window, onset, thresh=tau, direction="above")

        if flip_time is None or cross_time is None:
            if verbose:
                print(f"  [{label}] tau={tau} win={window_days}d: "
                      f"{'no anchor-gate transition' if flip_time is None else 'no attention_z crossing'} "
                      f"found. FAIL by construction (lead undefined).")
            results.append(dict(label=label, onset=onset_str, lead=float("nan"),
                                 pass_b=False, null_p90=float("nan"),
                                 flip=flip_time, cross=cross_time))
            continue

        lead = (flip_time - cross_time).total_seconds() / 86400.0
        null_leads = episode_null_leads(z, window, onset, flip_time, thresh=tau,
                                         direction="above", n_draws=N_DRAWS,
                                         block_days=BLOCK_DAYS, seed=NULL_SEED)
        valid_null = null_leads[~np.isnan(null_leads)]
        null_p90 = float(np.percentile(valid_null, 90)) if len(valid_null) else float("nan")
        pass_a = lead > 0
        pass_b = pass_a and (not np.isnan(null_p90)) and (lead > null_p90)

        if verbose:
            print(f"  [{label}] tau={tau} win={window_days}d  flip={flip_time}  "
                  f"cross={cross_time}  LEAD={lead:+.2f}d  null_p90={null_p90:+.2f}d  "
                  f"PASS={pass_b}")

        results.append(dict(label=label, onset=onset_str, lead=lead, pass_b=pass_b,
                             null_p90=null_p90, flip=flip_time, cross=cross_time,
                             valid_draws=len(valid_null)))
    return results


def gate() -> dict:
    print("=" * 78)
    print("R-94 CONSERVATIVE: Wikipedia attention confirming vote -- "
          "STEP A lead-time gate")
    print("=" * 78)

    bars = load_btc_bars()
    majority = anchor_majority(bars)
    views_5m = load_pageviews_aligned(bars)

    print(f"\ngrid: tau in {TAU_GRID}  x  window_days in {WINDOW_DAYS_GRID}  "
          f"({len(TAU_GRID) * len(WINDOW_DAYS_GRID)} cells x 6 episodes = "
          f"{len(TAU_GRID) * len(WINDOW_DAYS_GRID) * len(STRESS_EPISODES)} "
          f"episode-lead measurements)")
    print(f"null: {N_DRAWS} draws, block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    cells = {}
    for window_days in WINDOW_DAYS_GRID:
        for tau in TAU_GRID:
            print(f"--- cell tau={tau} window_days={window_days} ---")
            cell_results = gate_cell(bars, majority, views_5m, tau, window_days)
            n_pass = sum(1 for r in cell_results if r["pass_b"])
            cells[(tau, window_days)] = dict(results=cell_results, n_pass=n_pass)
            print(f"  cell totals: {n_pass}/6 episodes PASS\n")

    print("=" * 78)
    print("STEP A SUMMARY TABLE (episodes passing / 6, per grid cell):")
    print("=" * 78)
    header = "window_days \\ tau  " + "  ".join(f"{t:>6.1f}" for t in TAU_GRID)
    print(header)
    for window_days in WINDOW_DAYS_GRID:
        row = f"{window_days:>17d}  "
        row += "  ".join(f"{cells[(tau, window_days)]['n_pass']:>6d}" for tau in TAU_GRID)
        print(row)

    any_pass = any(c["n_pass"] >= MIN_EPISODES_PASS for c in cells.values())
    n_cells_passing_bar = sum(1 for c in cells.values() if c["n_pass"] >= MIN_EPISODES_PASS)
    # Plateau reading (banner item 2): a genuine pass needs neighbouring
    # cells to also clear the bar, not one isolated cell out of 9.
    plateau_pass = n_cells_passing_bar >= 3

    print(f"\ncells clearing >= {MIN_EPISODES_PASS}/6: {n_cells_passing_bar}/9")
    print(f"PLATEAU reading (>=3 cells must clear the bar, not an isolated "
          f"cell, per this branch's own pre-registered cherry-pick guard): "
          f"{'PASS' if plateau_pass else 'FAIL'}")
    print(f"\nGATE VERDICT: "
          f"{'PASS -> proceed to Step B' if plateau_pass else 'FAIL -> STOP, no strategy built'}")
    print(f"\nconfigurations evaluated in this file's Step A toward the backtest "
          f"trials ledger: 0 (fixed measurement gate; 9 grid cells x 6 episodes "
          f"= 54 gate diagnostics, no ev() calls, per this project's standing "
          f"accounting convention)")
    print(f"max timestamp read anywhere in this session so far: "
          f"{bars.index.max()}  (< {OOS_START})")

    return dict(cells=cells, any_pass=any_pass, n_cells_passing_bar=n_cells_passing_bar,
                plateau_pass=plateau_pass)


# ==========================================================================
# STEP B -- built only if the gate above passes (banner item 4).
# ==========================================================================

def compute_meta_vote(df: pd.DataFrame, views_5m: pd.Series, window_days: int,
                       tau: float, horizons: tuple[int, ...] = V4_HORIZONS,
                       band: float = V4_BAND) -> np.ndarray:
    """Attention-confirmed latch on the FASTEST anchor's own 0/1 vote
    (banner item 4): `meta_vote[i] = fast_anchor_vote[i]` on any bar where
    `attention_z[i] >= tau` ("confirmed"); otherwise `meta_vote[i] =
    meta_vote[i-1]` (carries forward). Before the first confirmed bar,
    defaults to the fast anchor's own then-current value.

    Causal: `attention_z` and each anchor vote are both causal; the latch
    update at i depends only on values at <= i.
    """
    fast_vote = anchor_votes(df, horizons=horizons, band=band)[0].to_numpy()
    z = attention_z(views_5m, window_days=window_days).reindex(df.index).to_numpy()
    confirmed = z >= tau

    n = len(df)
    meta = np.empty(n)
    meta[0] = fast_vote[0]
    for i in range(1, n):
        meta[i] = fast_vote[i] if confirmed[i] else meta[i - 1]
    return meta


class AttentionConfirmKelly(Strategy):
    """kelly_regime_v4 + a Wikipedia-attention-confirmed fast-anchor vote
    (R-94 conservative, unregistered). Structurally v3/v4's own prepare(),
    with the plain 3-anchor average `frac = anchor_sum/3` replaced by
    `confirming_vote_frac(anchor_sum, meta_vote, weight)`. `weight=0` must
    recover v4 bit-for-bit. Modelled directly on
    `r84_conservative_volume_confirm.py`'s `VolumeConfirmKelly`. Not
    `@register`ed -- stays in experiments/ per docs/ROUTINE.md.
    """

    name = "r94_conservative_attention_confirm"
    warmup = 80 * BARS_PER_DAY + 10  # identical to kelly_regime_v4

    def __init__(self, views_5m: pd.Series, weight: float = 1.0,
                 window_days: int = WINDOW_DAYS_PRIMARY, tau: float = TAU_PRIMARY,
                 horizons: tuple[int, ...] = V4_HORIZONS, band: float = V4_BAND,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85) -> None:
        self.views_5m = views_5m
        self.weight = weight
        self.window_days = window_days
        self.tau = tau
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = anchor_votes(df, horizons=self.horizons, band=self.band)
        anchor_sum = sum(v.to_numpy() for v in votes)

        views = self.views_5m.reindex(df.index)
        meta_vote = compute_meta_vote(df, views, self.window_days, self.tau,
                                       horizons=self.horizons, band=self.band)
        frac = confirming_vote_frac(anchor_sum, meta_vote, self.weight)

        # Identical conditional-volatility-targeting scale to kelly_regime_v3/_v4.
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

        n = len(df)
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
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# --------------------------------------------------------------------- checks

def truncation_causality_probe(build_target_fn, df: pd.DataFrame,
                                check_at: int, shorter_by: int = 20_000) -> bool:
    """Standard truncation probe: does `target[check_at]` change if bars
    after it are dropped? Returns True if causal (identical both ways).
    Implemented locally -- r94_shared.py does not ship one."""
    full = build_target_fn(df)
    short = build_target_fn(df.iloc[:check_at + shorter_by].copy())
    a, b = full[check_at], short[check_at]
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, equal_nan=True))


def build_target_primary(df: pd.DataFrame, views_5m: pd.Series) -> np.ndarray:
    """Target-construction function for the truncation causality probe,
    frozen at the pre-registered primary candidate."""
    return AttentionConfirmKelly(views_5m=views_5m, weight=1.0,
                                  window_days=WINDOW_DAYS_PRIMARY,
                                  tau=TAU_PRIMARY).prepare(df.copy())["target"].to_numpy()


def run_identity_check(df_full: pd.DataFrame, views_5m: pd.Series) -> float:
    """weight=0 must recover kelly_regime_v4 bit-for-bit. Returns max|diff|."""
    df = df_full.loc[:INNER_TRAIN_END].copy()
    v4_target = get_strategy("kelly_regime_v4").prepare(df.copy())["target"].to_numpy()
    cand_target = AttentionConfirmKelly(views_5m=views_5m, weight=0.0).prepare(df.copy())["target"].to_numpy()
    max_diff = float(np.max(np.abs(v4_target - cand_target)))
    print(f"[identity] weight=0 vs kelly_regime_v4, {len(df):,} bars: "
          f"max|diff| = {max_diff:.3e}")
    return max_diff


def run_causality_probe(df_full: pd.DataFrame, views_5m: pd.Series) -> list[bool]:
    df = df_full.loc[:INNER_TRAIN_END].copy()
    results = []
    for check_at in (150_000, 250_000, 350_000):
        ok = truncation_causality_probe(lambda d: build_target_primary(d, views_5m), df, check_at)
        print(f"[causality] check_at={check_at}: {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    return results


def eval_config(ev, SPOT, FUTURES, views_5m: pd.Series, weight: float,
                 window_days: int, tau: float, tag: str) -> dict:
    out = {}
    for split_name, kw in (
        ("train", dict(end=INNER_TRAIN_END)),
        ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END)),
    ):
        for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
            strat = AttentionConfirmKelly(views_5m=views_5m, weight=weight,
                                           window_days=window_days, tau=tau)
            m = ev(strat, market=mkt, tag=f"{tag} {split_name} {mkt_name}", **kw)
            out[(split_name, mkt_name)] = m
    return out


def run_sweep(ev, SPOT, FUTURES, views_5m: pd.Series) -> dict:
    results = {}
    for weight in (0.5, 1.0, 2.0, 4.0):
        for window_days in (10, 20, 30):
            tag = f"w{weight} win{window_days} tau{TAU_PRIMARY}"
            results[("main", weight, window_days, TAU_PRIMARY)] = eval_config(
                ev, SPOT, FUTURES, views_5m, weight, window_days, TAU_PRIMARY, tag)
    for tau in (1.0, 2.0):
        tag = f"w1.0 win{WINDOW_DAYS_PRIMARY} tau{tau}"
        results[("tausens", 1.0, WINDOW_DAYS_PRIMARY, tau)] = eval_config(
            ev, SPOT, FUTURES, views_5m, 1.0, WINDOW_DAYS_PRIMARY, tau, tag)
    return results


def exposure_artifact_check(ev, DF, SPOT, FUTURES, views_5m: pd.Series,
                             weight: float = 1.0, window_days: int = WINDOW_DAYS_PRIMARY,
                             tau: float = TAU_PRIMARY) -> dict:
    """Diagnostic: regress the candidate's target series against a
    mean-notional-matched flat rescale of v4's own target series, on
    inner-validation, both markets. R^2 > 0.95 -> "just a rescale"."""
    v4 = get_strategy("kelly_regime_v4")
    cand = AttentionConfirmKelly(views_5m=views_5m, weight=weight,
                                  window_days=window_days, tau=tau)
    out = {}
    print(f"\nexposure-artifact check (weight={weight}, window_days={window_days}, "
          f"tau={tau}):")
    for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
        lo = int(DF.index.searchsorted(INNER_VAL_START))
        hi = int(DF.index.searchsorted(INNER_VAL_END, side="right"))
        prefix = min(lo, max(cand.warmup, v4.warmup))
        frame = DF.iloc[lo - prefix:hi]

        v4_prepared = v4.prepare(frame.copy())
        cand_prepared = cand.prepare(frame.copy())
        v4_t = v4_prepared["target"].to_numpy()[prefix:]
        cand_t = cand_prepared["target"].to_numpy()[prefix:]

        mean_abs_v4 = np.mean(np.abs(v4_t))
        mean_abs_cand = np.mean(np.abs(cand_t))
        alpha = mean_abs_cand / mean_abs_v4 if mean_abs_v4 > 0 else 0.0
        rescaled = alpha * v4_t
        ss_res = np.sum((cand_t - rescaled) ** 2)
        ss_tot = np.sum((cand_t - cand_t.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  {mkt_name:9s} mean|v4|={mean_abs_v4:.3f} mean|cand|={mean_abs_cand:.3f} "
              f"alpha={alpha:.3f}  R^2(cand vs alpha*v4)={r2:.4f}  "
              f"{'JUST A RESCALE' if r2 > 0.95 else 'genuinely different exposure shape'}")
        out[mkt_name] = r2
    return out


def run_eth_falsification(ev, weight: float = 1.0, window_days: int = WINDOW_DAYS_PRIMARY,
                           tau: float = TAU_PRIMARY) -> dict:
    """ETH falsification: the SAME frozen construction (BTC Wikipedia
    attention gating v4's fast anchor) applied UNMODIFIED to ETH price
    action, per this branch's dispatch instructions ("does the same frozen
    rule ... replicate qualitatively on ethusd_coinbase_spot_5m.csv.gz").
    Disclosed explicitly (banner item 4): this uses the Coinbase ETH file
    named in the dispatch prompt, not R-84/R-77's Bitfinex pair, and there
    is no ETH-specific attention series available in this round's data --
    the gate stays keyed to BTC Wikipedia readership even when priced
    against ETH. This tests whether the CONSTRUCTION generalizes across
    the asset being traded, not whether a symmetric ETH attention signal
    would work; that distinction is reported alongside the result, not
    silently equated with R-84's stronger, genuinely cross-instrument
    check."""
    from tradebot.broker import MarketSpec

    spot = MarketSpec.spot()
    path = DATA_DIR / "ethusd_coinbase_spot_5m.csv.gz"
    if not path.exists():
        print(f"  ETH falsification: {path.name} not found in data/ -- SKIPPED, "
              f"reported as not run, not as a pass.")
        return {"skipped": True}

    eth_df = load_ohlcv_csv(path)
    eth_df = eth_df.loc[eth_df.index < pd.Timestamp(OOS_START, tz=eth_df.index.tz)].copy()
    assert_no_holdout(eth_df)

    # Align BTC Wikipedia attention onto ETH's own bar grid.
    pv = load_wikipedia_pageviews(DATA_DIR)
    views_on_eth = align_wikipedia_causal(pv, eth_df)["views"]
    assert_no_holdout(views_on_eth)

    cand = AttentionConfirmKelly(views_5m=views_on_eth, weight=weight,
                                  window_days=window_days, tau=tau)
    v4 = get_strategy("kelly_regime_v4")
    m_v4 = ev(v4, df=eth_df, market=spot, tag="ETH (coinbase): v4")
    m_cand = ev(cand, df=eth_df, market=spot, tag="ETH (coinbase): candidate")
    delta = m_cand.sharpe - m_v4.sharpe
    print(f"  ETH (coinbase, < {OOS_START}): v4 sharpe={m_v4.sharpe:.2f}  "
          f"candidate sharpe={m_cand.sharpe:.2f}  delta={delta:+.2f}")
    return dict(v4=m_v4.sharpe, cand=m_cand.sharpe, delta=delta,
                bars=len(eth_df), start=str(eth_df.index[0]), end=str(eth_df.index[-1]))


def run_step_b() -> None:
    from scripts.experiment import DF, FUTURES, SPOT, ev

    print("\n" + "=" * 78)
    print("STEP B (gate passed): sweep + mandatory checks")
    print("=" * 78)

    df_trunc = DF.loc[DF.index < pd.Timestamp(OOS_START, tz=DF.index.tz)]
    views_5m = load_pageviews_aligned(df_trunc)
    # Reindex onto the full DF's index (train+val only will be used by ev()
    # via start/end, but prepare() is called on whatever frame ev() passes
    # it, so the Series must cover the full pre-OOS range DF itself does).
    views_5m_full = views_5m.reindex(DF.index)

    n_configs = 0
    print("\n=== identity check ===")
    run_identity_check(DF, views_5m_full)
    n_configs += 1

    print("\n=== causality probe ===")
    run_causality_probe(DF, views_5m_full)

    print("\n=== baselines (kelly_regime_v4, buy_and_hold) ===")
    for name in ("kelly_regime_v4", "buy_and_hold"):
        for split_name, kw in (("train", dict(end=INNER_TRAIN_END)),
                                ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END))):
            for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
                ev(get_strategy(name), market=mkt, tag=f"{name} {split_name} {mkt_name}", **kw)

    print("\n=== sweep ===")
    sweep_results = run_sweep(ev, SPOT, FUTURES, views_5m_full)
    n_configs += len(sweep_results)

    print("\n=== exposure-artifact check (primary) ===")
    exposure_artifact_check(ev, DF, SPOT, FUTURES, views_5m_full)

    print("\n=== ETH falsification ===")
    run_eth_falsification(ev)

    print(f"\nTotal Step B configurations evaluated: {n_configs}")


# ------------------------------------------------------------------------ main

def main() -> None:
    t0 = time.time()
    gate_result = gate()
    if gate_result["plateau_pass"]:
        run_step_b()
    else:
        print("\nSTEP A FAILED the pre-registered stop rule. Per this file's own "
              "pre-registration, no strategy is built and no Step-B code runs. "
              "This gate result is this branch's whole product.")
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmds = {"gate": gate, "main": main}
    choice = sys.argv[1] if len(sys.argv) > 1 else "main"
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/r94_conservative_attention_confirm.py [{'|'.join(cmds)}]")
