#!/usr/bin/env python
"""R-95 CONSERVATIVE branch: the alternative.me Crypto Fear & Greed Index
(FGI) EXTREMITY as a confirming vote on `kelly_regime_v4`'s 3-anchor gate,
via R-53/R-55's already-validated `confirming_vote_frac` combination rule --
Step A measurement gate first, this project's established discipline for
every INFO-axis round since R-53 (R-53/R-73/R-74/R-79/R-81/R-84/R-88/R-94).

=====================================================================
PRE-REGISTRATION (frozen before any lead/lag number in this file was
computed -- docs/ROUTINE.md steps 1-2/4, and the operator's dispatch prompt
for this branch, 2026-08-22). If anything below is later contradicted by
what actually happened, that is stated in the results section, not edited
back into this banner.
=====================================================================

1. MECHANISM (one sentence, fixed by the dispatch prompt). He, Shen, Zhang &
   Zhang (2023, Finance Research Letters 58(PA)) find alternative.me's FGI
   has genuine in-sample and out-of-sample return-CONTINUATION predictive
   power at 1-day-to-1-week horizons -- so a price move crossing one of
   `kelly_regime_v4`'s anchors is a more trustworthy confirming vote when it
   co-occurs with FGI in an EXTREME state (either extreme greed or extreme
   fear) than when FGI sits near neutral (50). Full citation set, the
   not-a-duplicate-of argument against the twelve prior INFO-axis signals,
   and the data-coverage verification are in `experiments/r95_shared.py`'s
   module docstring -- frozen, not re-derived here.

   Constraint attacked: INFO. Architecture: reuses R-53/R-55's validated
   CONFIRMING-VOTE combination rule (`confirming_vote_frac`), exactly as the
   dispatch prompt specifies -- not a new architecture, so none of the four
   independent never-increase-only-BRAKE failures (R-34, R-41,
   R-53-conservative, R-73-conservative) apply here.

2. STEP A -- THE MANDATORY MEASUREMENT GATE, run BEFORE any strategy code,
   on the FULL six-episode table (`r95_shared.STRESS_EPISODES`; episode 1,
   2018-01-17, is a disclosed automatic coverage FAIL since FGI starts
   2018-02-01 -- scored as a fail, never dropped), using `r95_shared`'s own
   frozen `nearest_transition` (v4's own anchor-vote flip), `nearest_crossing`
   (first bar at/after onset where the extremity series crosses above
   `tau`), `episode_window` (onset-60d, onset+30d), and `episode_null_leads`
   (500-draw, 5-day-block circular block-bootstrap).

   PRIMARY FEATURE: `extremity(t) = abs(fgi_level(t) - 50) / 50` (0 at
   neutral, 1 at either extreme 0 or 100). FGI is already a fixed 0-100
   provider-normalized scale, so -- UNLIKE every prior INFO signal in this
   project (pageviews, volume, on-chain counts, all of which carry a
   secular trend and need a rolling z-score against their own trailing
   history) -- no rolling z-score is needed here; a fixed threshold on
   `extremity` is meaningful across the whole 2018-2026 span by
   construction. This is a genuine methodological difference from every
   prior INFO round, stated explicitly, not a simplification of
   convenience. `fgi_level` is optionally smoothed first with a trailing
   EWM over `smooth_days` (causal: `.ewm(...).mean()` depends only on
   current and past rows) to damp day-to-day noise before the abs/50 step.

   GRID (fixed now, before any lead number in this file was computed --
   sensitivity sweep, not a search for a passing cell; every cell reported,
   none dropped): tau in {0.2, 0.3, 0.4} (FGI outside [40,60]/[35,65]/
   [30,70]) x smooth_days in {1 (no smoothing), 3, 7} -- 9 cells x 6
   episodes = 54 episode-lead measurements. direction="above" on the
   extremity series. No `ev()` call, no strategy backtest, in any Step-A
   cell -- per this project's standing accounting convention, Step A's own
   configuration count is 0 for the deflated-Sharpe trials ledger.

   PRE-REGISTERED STOP RULE: an episode PASSES iff (a) LEAD =
   (flip_time - crossing_time) in days is > 0, AND (b) LEAD exceeds the
   90th percentile of that episode's own 500-draw block-bootstrap null.
   PROCEED TO STEP B ONLY IF a PLATEAU of cells (>= 3 of the 9 grid cells)
   each clear >= 4 of 6 episodes passing -- read conservatively: one
   isolated passing cell surrounded by failing neighbours is noise from
   running 9 cells, not grounds to proceed (identical plateau-reading
   convention to R-94). If no plateau clears the bar: STOP, report the gate
   result as this branch's whole product, build no strategy code -- 0 of
   twelve prior INFO signals have led the anchor gate at this bar; a clean
   negative here is a fully successful, complete result.

3. WHAT WOULD MAKE STEP A FAIL, named now: the same failure every one of
   the twelve prior INFO signals hit -- FGI extremity is reached AFTER (not
   before) the anchor gate's own nearest reaction, or a positive lead is
   not distinguishable from an arbitrary time-shift of the same series.
   This is also the literature's OWN split verdict on this exact data
   source (see r95_shared.py's docstring: a 2026 VAR-model paper using the
   same alternative.me series finds FGI does NOT Granger-cause BTC returns
   and is reactive, not predictive). Given the base rate (0 of 12 prior
   INFO signals led) and this specific literature split, the modal,
   pre-registered expectation IS failure, and a clean negative is this
   branch's fully successful, complete product if that is what happens.

4. STEP B -- CONTINGENT PRE-REGISTRATION (design frozen now, BEFORE Step
   A's numbers exist; only executed if Step A's plateau stop rule passes).

   CONFIRMING-VOTE CONSTRUCTION: `confirming_vote_frac` (imported unchanged
   from r95_shared.py) requires a DISCRETE {0,1} `meta_vote`. FGI extremity
   carries no direction of its own, so the vote's direction comes from v4's
   own FASTEST (20-day) anchor vote: `meta_vote[i] = fast_anchor_vote[i]` on
   any bar where `extremity[i] >= tau` ("confirmed"); otherwise
   `meta_vote[i] = meta_vote[i-1]` (carry forward, R-53/54/55's
   hysteresis-latch pattern). Before the first confirmed bar, defaults to
   the fast anchor's own then-current value (no dilution while
   unconfirmed, R-84's convention).

   frac = confirming_vote_frac(anchor_sum, meta_vote, weight)
        = (anchor_sum + weight * meta_vote) / (3 + weight)

   `weight=0` must recover `kelly_regime_v4` bit-for-bit (identity check,
   run first, before any swept configuration).

   SWEEP GRID (fixed a priori): weight in {0.5, 1.0, 2.0, 4.0} x
   smooth_days (at the Step-A primary tau, whichever cell is closest to the
   grid center of the passing plateau) in {1, 3, 7} -- 12 configurations;
   plus tau sensitivity at the primary point over the other 2 grid values
   of tau -- 2 more configurations; plus the identity check -- 1. Total 15
   configurations, matching R-84/R-94's own count for a like-for-like
   comparison.

   MANDATORY CHECKS: (i) exposure-artifact R^2 -- regress the candidate's
   `target` series (inner-validation, both markets) against a
   mean-notional-matched flat rescale of v4's own target; R^2 > 0.95 = fail
   ("just a rescale"); (ii) ETH falsification on
   `ethusd_coinbase_spot_5m.csv.gz`, same frozen construction -- FGI is
   BTC-specific so there is no ETH-specific sentiment series, disclosed
   explicitly, exactly as R-94 did for pageviews; (iii) causal-truncation
   probe (does `target[check_at]` change if later bars are dropped -- must
   NOT change).

   PRE-REGISTERED HOLDOUT DECISION RULE (only reached if everything above
   clears): read the 2023+ holdout ONLY IF ALL of (a) primary
   configuration's inner-validation Sharpe improvement over
   `kelly_regime_v4` exceeds the +/-0.2 noise floor on BOTH markets, on a
   genuine plateau; (b) exposure-artifact R^2 <= 0.95 both markets; (c) ETH
   falsification replicates the same sign; (d) causality probe passes all
   checkpoints. If ANY fail: report NEGATIVE, never read the holdout -- a
   hard rule, not a suggestion.

5. CONFIGS EVALUATED IN STEP A: 0 for the backtest trials ledger (9 grid
   cells x 6 episodes = 54 gate diagnostics, no ev() calls). Step B's
   count, if reached: 15, as itemized above.
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

from tradebot.data import align_fear_greed_causal, load_dataset, load_fear_greed_index, load_ohlcv_csv  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

from experiments.r95_shared import (  # noqa: E402
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
    confirming_vote_frac,
    episode_coverage_ok,
    episode_null_leads,
    episode_window,
    fgi_level,
    nearest_crossing,
    nearest_transition,
)

DATA_DIR = ROOT / "data"

# ---- Step A constants, fixed a priori (see banner item 2) -----------------
TAU_GRID = (0.2, 0.3, 0.4)
SMOOTH_GRID = (1, 3, 7)
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 95
MIN_EPISODES_PASS = 4  # of 6
MIN_PLATEAU_CELLS = 3  # of 9

# Grid-center fallback, used only if run_step_b() is invoked standalone
# without a gate_result to derive the actual primary cell from (banner
# item 4: "whichever cell is closest to the grid center of the passing
# plateau"). Overridden dynamically in main()'s call path.
TAU_PRIMARY_DEFAULT = 0.3
SMOOTH_PRIMARY_DEFAULT = 3


# --------------------------------------------------------------------- data

def assert_no_holdout(df, oos_start: str = OOS_START) -> None:
    """Hard guard: the max timestamp in any frame this file touches must be
    strictly before `oos_start`. Same convention as r79/r81/r84/r94's own
    guard. r95_shared.py does not ship one (its loaders leave truncation to
    the caller), so it is implemented locally here."""
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
    FGI is ever aligned onto it, so the holdout is never touched even
    transiently."""
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def load_fgi_aligned(bars: pd.DataFrame) -> pd.Series:
    """Daily FGI, causally aligned onto `bars`' own (already
    holdout-truncated) index. Deliberately does NOT use
    `r95_shared.load_fgi_5m` here, because that helper aligns onto the FULL
    committed BTC bar file (which extends past OOS_START) before the caller
    gets a chance to truncate -- truncating `bars` first, as this function
    does, means the FGI series itself is never aligned against any
    post-OOS_START timestamp at all (same fix r94's `load_pageviews_aligned`
    made over its own shared helper)."""
    fgi = load_fear_greed_index(DATA_DIR)
    assert fgi is not None, "Fear & Greed Index file missing from data/"
    aligned = align_fear_greed_causal(fgi, bars)
    assert_no_holdout(aligned)
    return aligned["value"]


def fgi_extremity(value_5m: pd.Series, smooth_days: int) -> pd.Series:
    """extremity(t) = abs(fgi_level(t) - 50) / 50, optionally smoothed by a
    trailing EWM over `smooth_days` before the abs/50 step. Causal:
    `fgi_level` is ffill only; `.ewm(...).mean()` at row i depends only on
    rows <= i."""
    level = fgi_level(value_5m)
    if smooth_days is not None and smooth_days > 1:
        span_bars = int(smooth_days * BARS_PER_DAY)
        level = level.ewm(span=span_bars, min_periods=1).mean()
    return (level - 50.0).abs() / 50.0


# --------------------------------------------------------------------- gate

def gate_cell(bars: pd.DataFrame, majority: pd.Series, value_5m: pd.Series,
              tau: float, smooth_days: int, verbose: bool = True) -> list[dict]:
    """One (tau, smooth_days) grid cell: run all 6 episodes, return their
    per-episode result dicts."""
    extremity = fgi_extremity(value_5m, smooth_days)
    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str)
        coverage_ok = episode_coverage_ok(onset_str)
        if len(window) == 0:
            results.append(dict(label=label, onset=onset_str, lead=float("nan"),
                                 pass_b=False, null_p90=float("nan"),
                                 coverage_ok=coverage_ok))
            continue

        flip_time = nearest_transition(majority, window, onset)
        cross_time = nearest_crossing(extremity, window, onset, thresh=tau, direction="above")

        if flip_time is None or cross_time is None:
            if verbose:
                note = " [disclosed FGI coverage fail]" if not coverage_ok else ""
                print(f"  [{label}] tau={tau} smooth={smooth_days}d: "
                      f"{'no anchor-gate transition' if flip_time is None else 'no extremity crossing'} "
                      f"found. FAIL by construction (lead undefined).{note}")
            results.append(dict(label=label, onset=onset_str, lead=float("nan"),
                                 pass_b=False, null_p90=float("nan"),
                                 flip=flip_time, cross=cross_time, coverage_ok=coverage_ok))
            continue

        lead = (flip_time - cross_time).total_seconds() / 86400.0
        null_leads = episode_null_leads(extremity, window, onset, flip_time, thresh=tau,
                                         direction="above", n_draws=N_DRAWS,
                                         block_days=BLOCK_DAYS, seed=NULL_SEED)
        valid_null = null_leads[~np.isnan(null_leads)]
        null_p90 = float(np.percentile(valid_null, 90)) if len(valid_null) else float("nan")
        pass_a = lead > 0
        pass_b = pass_a and (not np.isnan(null_p90)) and (lead > null_p90)

        if verbose:
            note = " [disclosed FGI coverage fail]" if not coverage_ok else ""
            print(f"  [{label}] tau={tau} smooth={smooth_days}d  flip={flip_time}  "
                  f"cross={cross_time}  LEAD={lead:+.2f}d  null_p90={null_p90:+.2f}d  "
                  f"PASS={pass_b}{note}")

        results.append(dict(label=label, onset=onset_str, lead=lead, pass_b=pass_b,
                             null_p90=null_p90, flip=flip_time, cross=cross_time,
                             valid_draws=len(valid_null), coverage_ok=coverage_ok))
    return results


def gate() -> dict:
    print("=" * 78)
    print("R-95 CONSERVATIVE: Fear & Greed Index extremity confirming vote -- "
          "STEP A lead-time gate")
    print("=" * 78)

    bars = load_btc_bars()
    majority = anchor_majority(bars)
    value_5m = load_fgi_aligned(bars)

    print(f"\ngrid: tau in {TAU_GRID}  x  smooth_days in {SMOOTH_GRID}  "
          f"({len(TAU_GRID) * len(SMOOTH_GRID)} cells x 6 episodes = "
          f"{len(TAU_GRID) * len(SMOOTH_GRID) * len(STRESS_EPISODES)} "
          f"episode-lead measurements)")
    print(f"null: {N_DRAWS} draws, block={BLOCK_DAYS}d, seed={NULL_SEED}")
    print("episode 1 (2018-01-17) is a disclosed automatic FGI coverage fail "
          "(FGI starts 2018-02-01) -- included, scored as a fail, never dropped.\n")

    cells = {}
    for smooth_days in SMOOTH_GRID:
        for tau in TAU_GRID:
            print(f"--- cell tau={tau} smooth_days={smooth_days} ---")
            cell_results = gate_cell(bars, majority, value_5m, tau, smooth_days)
            n_pass = sum(1 for r in cell_results if r["pass_b"])
            cells[(tau, smooth_days)] = dict(results=cell_results, n_pass=n_pass)
            print(f"  cell totals: {n_pass}/6 episodes PASS\n")

    print("=" * 78)
    print("STEP A SUMMARY TABLE (episodes passing / 6, per grid cell):")
    print("=" * 78)
    header = "smooth_days \\ tau  " + "  ".join(f"{t:>6.2f}" for t in TAU_GRID)
    print(header)
    for smooth_days in SMOOTH_GRID:
        row = f"{smooth_days:>17d}  "
        row += "  ".join(f"{cells[(tau, smooth_days)]['n_pass']:>6d}" for tau in TAU_GRID)
        print(row)

    n_cells_passing_bar = sum(1 for c in cells.values() if c["n_pass"] >= MIN_EPISODES_PASS)
    # Plateau reading (banner item 2): a genuine pass needs neighbouring
    # cells to also clear the bar, not one isolated cell out of 9.
    plateau_pass = n_cells_passing_bar >= MIN_PLATEAU_CELLS

    print(f"\ncells clearing >= {MIN_EPISODES_PASS}/6: {n_cells_passing_bar}/9")
    print(f"PLATEAU reading (>= {MIN_PLATEAU_CELLS} cells must clear the bar, not an "
          f"isolated cell, per this branch's own pre-registered cherry-pick guard): "
          f"{'PASS' if plateau_pass else 'FAIL'}")
    print(f"\nGATE VERDICT: "
          f"{'PASS -> proceed to Step B' if plateau_pass else 'FAIL -> STOP, no strategy built'}")
    print(f"\nconfigurations evaluated in this file's Step A toward the backtest "
          f"trials ledger: 0 (fixed measurement gate; 9 grid cells x 6 episodes "
          f"= 54 gate diagnostics, no ev() calls, per this project's standing "
          f"accounting convention)")
    print(f"max timestamp read anywhere in this session so far: "
          f"{bars.index.max()}  (< {OOS_START})")

    return dict(cells=cells, n_cells_passing_bar=n_cells_passing_bar,
                plateau_pass=plateau_pass)


# ==========================================================================
# STEP B -- built only if the gate above passes (banner item 4).
# ==========================================================================

def select_primary_cell(cells: dict) -> tuple[float, int]:
    """Among the grid cells clearing the >= 4/6 bar, pick the one closest
    (in grid-index space) to the grid center -- banner item 4's "whichever
    cell is closest to the grid center of the passing plateau". Falls back
    to the grid center itself if called with no passing cells (should not
    happen when this is only invoked after plateau_pass is True)."""
    tau_center_idx = len(TAU_GRID) // 2
    smooth_center_idx = len(SMOOTH_GRID) // 2
    passing = [(tau, smooth) for (tau, smooth), c in cells.items()
               if c["n_pass"] >= MIN_EPISODES_PASS]
    if not passing:
        return TAU_PRIMARY_DEFAULT, SMOOTH_PRIMARY_DEFAULT

    def dist(cell):
        tau, smooth = cell
        ti = TAU_GRID.index(tau)
        si = SMOOTH_GRID.index(smooth)
        return ((ti - tau_center_idx) ** 2 + (si - smooth_center_idx) ** 2, ti, si)

    tau, smooth = min(passing, key=dist)
    return tau, smooth


def compute_meta_vote(df: pd.DataFrame, value_5m: pd.Series, smooth_days: int,
                       tau: float, horizons: tuple[int, ...] = V4_HORIZONS,
                       band: float = V4_BAND) -> np.ndarray:
    """FGI-extremity-confirmed latch on the FASTEST anchor's own 0/1 vote
    (banner item 4): `meta_vote[i] = fast_anchor_vote[i]` on any bar where
    `extremity[i] >= tau` ("confirmed"); otherwise `meta_vote[i] =
    meta_vote[i-1]` (carries forward). Before the first confirmed bar,
    defaults to the fast anchor's own then-current value.

    Causal: `extremity` and each anchor vote are both causal; the latch
    update at i depends only on values at <= i.
    """
    fast_vote = anchor_votes(df, horizons=horizons, band=band)[0].to_numpy()
    extremity = fgi_extremity(value_5m, smooth_days).reindex(df.index).to_numpy()
    confirmed = extremity >= tau

    n = len(df)
    meta = np.empty(n)
    meta[0] = fast_vote[0]
    for i in range(1, n):
        meta[i] = fast_vote[i] if confirmed[i] else meta[i - 1]
    return meta


class FgiConfirmKelly(Strategy):
    """kelly_regime_v4 + an FGI-extremity-confirmed fast-anchor vote (R-95
    conservative, unregistered). Structurally v3/v4's own prepare(), with
    the plain 3-anchor average `frac = anchor_sum/3` replaced by
    `confirming_vote_frac(anchor_sum, meta_vote, weight)`. `weight=0` must
    recover v4 bit-for-bit. Modelled directly on
    `r94_conservative_attention_confirm.py`'s `AttentionConfirmKelly`. Not
    `@register`ed -- stays in experiments/ per docs/ROUTINE.md.
    """

    name = "r95_conservative_fgi_confirm"
    warmup = 80 * BARS_PER_DAY + 10  # identical to kelly_regime_v4

    def __init__(self, value_5m: pd.Series, weight: float = 1.0,
                 smooth_days: int = SMOOTH_PRIMARY_DEFAULT, tau: float = TAU_PRIMARY_DEFAULT,
                 horizons: tuple[int, ...] = V4_HORIZONS, band: float = V4_BAND,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85) -> None:
        self.value_5m = value_5m
        self.weight = weight
        self.smooth_days = smooth_days
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

        value = self.value_5m.reindex(df.index)
        meta_vote = compute_meta_vote(df, value, self.smooth_days, self.tau,
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
    Implemented locally -- r95_shared.py does not ship one."""
    full = build_target_fn(df)
    short = build_target_fn(df.iloc[:check_at + shorter_by].copy())
    a, b = full[check_at], short[check_at]
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, equal_nan=True))


def build_target_primary(df: pd.DataFrame, value_5m: pd.Series,
                          tau: float, smooth_days: int) -> np.ndarray:
    """Target-construction function for the truncation causality probe,
    frozen at the pre-registered primary candidate."""
    return FgiConfirmKelly(value_5m=value_5m, weight=1.0,
                            smooth_days=smooth_days, tau=tau).prepare(df.copy())["target"].to_numpy()


def run_identity_check(df_full: pd.DataFrame, value_5m: pd.Series) -> float:
    """weight=0 must recover kelly_regime_v4 bit-for-bit. Returns max|diff|."""
    df = df_full.loc[:INNER_TRAIN_END].copy()
    v4_target = get_strategy("kelly_regime_v4").prepare(df.copy())["target"].to_numpy()
    cand_target = FgiConfirmKelly(value_5m=value_5m, weight=0.0).prepare(df.copy())["target"].to_numpy()
    max_diff = float(np.max(np.abs(v4_target - cand_target)))
    print(f"[identity] weight=0 vs kelly_regime_v4, {len(df):,} bars: "
          f"max|diff| = {max_diff:.3e}")
    return max_diff


def run_causality_probe(df_full: pd.DataFrame, value_5m: pd.Series,
                         tau: float, smooth_days: int) -> list[bool]:
    df = df_full.loc[:INNER_TRAIN_END].copy()
    results = []
    for check_at in (150_000, 250_000, 350_000):
        ok = truncation_causality_probe(
            lambda d: build_target_primary(d, value_5m, tau, smooth_days), df, check_at)
        print(f"[causality] check_at={check_at}: {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    return results


def eval_config(ev, SPOT, FUTURES, value_5m: pd.Series, weight: float,
                 smooth_days: int, tau: float, tag: str) -> dict:
    out = {}
    for split_name, kw in (
        ("train", dict(end=INNER_TRAIN_END)),
        ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END)),
    ):
        for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
            strat = FgiConfirmKelly(value_5m=value_5m, weight=weight,
                                     smooth_days=smooth_days, tau=tau)
            m = ev(strat, market=mkt, tag=f"{tag} {split_name} {mkt_name}", **kw)
            out[(split_name, mkt_name)] = m
    return out


def run_sweep(ev, SPOT, FUTURES, value_5m: pd.Series,
              tau_primary: float, smooth_primary: int) -> dict:
    other_taus = [t for t in TAU_GRID if t != tau_primary]
    results = {}
    for weight in (0.5, 1.0, 2.0, 4.0):
        for smooth_days in SMOOTH_GRID:
            tag = f"w{weight} smooth{smooth_days} tau{tau_primary}"
            results[("main", weight, smooth_days, tau_primary)] = eval_config(
                ev, SPOT, FUTURES, value_5m, weight, smooth_days, tau_primary, tag)
    for tau in other_taus:
        tag = f"w1.0 smooth{smooth_primary} tau{tau}"
        results[("tausens", 1.0, smooth_primary, tau)] = eval_config(
            ev, SPOT, FUTURES, value_5m, 1.0, smooth_primary, tau, tag)
    return results


def exposure_artifact_check(ev, DF, SPOT, FUTURES, value_5m: pd.Series,
                             weight: float, smooth_days: int, tau: float) -> dict:
    """Diagnostic: regress the candidate's target series against a
    mean-notional-matched flat rescale of v4's own target series, on
    inner-validation, both markets. R^2 > 0.95 -> "just a rescale"."""
    v4 = get_strategy("kelly_regime_v4")
    cand = FgiConfirmKelly(value_5m=value_5m, weight=weight,
                            smooth_days=smooth_days, tau=tau)
    out = {}
    print(f"\nexposure-artifact check (weight={weight}, smooth_days={smooth_days}, "
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


def run_eth_falsification(ev, weight: float, smooth_days: int, tau: float) -> dict:
    """ETH falsification: the SAME frozen construction (BTC FGI extremity
    gating v4's fast anchor) applied UNMODIFIED to ETH price action, per
    this branch's dispatch instructions. Disclosed explicitly (banner item
    4): FGI is BTC-specific -- alternative.me publishes one crowd-sentiment
    index for the whole crypto market keyed to BTC, and there is no
    ETH-specific sentiment series in this round's data. This tests whether
    the CONSTRUCTION generalizes across the asset being traded, not whether
    a symmetric ETH sentiment signal would work; that distinction is
    reported alongside the result, not silently equated with a genuine
    cross-instrument signal check (same disclosure R-94 made for
    Wikipedia pageviews)."""
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

    # Align BTC-specific FGI onto ETH's own bar grid.
    fgi = load_fear_greed_index(DATA_DIR)
    value_on_eth = align_fear_greed_causal(fgi, eth_df)["value"]
    assert_no_holdout(value_on_eth)

    cand = FgiConfirmKelly(value_5m=value_on_eth, weight=weight,
                            smooth_days=smooth_days, tau=tau)
    v4 = get_strategy("kelly_regime_v4")
    m_v4 = ev(v4, df=eth_df, market=spot, tag="ETH (coinbase): v4")
    m_cand = ev(cand, df=eth_df, market=spot, tag="ETH (coinbase): candidate")
    delta = m_cand.sharpe - m_v4.sharpe
    print(f"  ETH (coinbase, < {OOS_START}): v4 sharpe={m_v4.sharpe:.2f}  "
          f"candidate sharpe={m_cand.sharpe:.2f}  delta={delta:+.2f}")
    return dict(v4=m_v4.sharpe, cand=m_cand.sharpe, delta=delta,
                bars=len(eth_df), start=str(eth_df.index[0]), end=str(eth_df.index[-1]))


def run_step_b(gate_result: dict) -> None:
    from scripts.experiment import DF, FUTURES, SPOT, ev

    tau_primary, smooth_primary = select_primary_cell(gate_result["cells"])
    print("\n" + "=" * 78)
    print(f"STEP B (gate passed): sweep + mandatory checks -- primary cell "
          f"tau={tau_primary}, smooth_days={smooth_primary}")
    print("=" * 78)

    df_trunc = DF.loc[DF.index < pd.Timestamp(OOS_START, tz=DF.index.tz)]
    value_5m = load_fgi_aligned(df_trunc)
    # Reindex onto the full DF's index (train+val only will be used by ev()
    # via start/end, but prepare() is called on whatever frame ev() passes
    # it, so the Series must cover the full pre-OOS range DF itself does).
    value_5m_full = value_5m.reindex(DF.index)

    n_configs = 0
    print("\n=== identity check ===")
    run_identity_check(DF, value_5m_full)
    n_configs += 1

    print("\n=== causality probe ===")
    run_causality_probe(DF, value_5m_full, tau_primary, smooth_primary)

    print("\n=== baselines (kelly_regime_v4, buy_and_hold) ===")
    for name in ("kelly_regime_v4", "buy_and_hold"):
        for split_name, kw in (("train", dict(end=INNER_TRAIN_END)),
                                ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END))):
            for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
                ev(get_strategy(name), market=mkt, tag=f"{name} {split_name} {mkt_name}", **kw)

    print("\n=== sweep ===")
    sweep_results = run_sweep(ev, SPOT, FUTURES, value_5m_full, tau_primary, smooth_primary)
    n_configs += len(sweep_results)

    print("\n=== exposure-artifact check (primary) ===")
    exposure_artifact_check(ev, DF, SPOT, FUTURES, value_5m_full,
                             weight=1.0, smooth_days=smooth_primary, tau=tau_primary)

    print("\n=== ETH falsification ===")
    run_eth_falsification(ev, weight=1.0, smooth_days=smooth_primary, tau=tau_primary)

    print(f"\nTotal Step B configurations evaluated: {n_configs}")


# ------------------------------------------------------------------------ main

def main() -> None:
    t0 = time.time()
    gate_result = gate()
    if gate_result["plateau_pass"]:
        run_step_b(gate_result)
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
        print(f"usage: python experiments/r95_conservative_fgi_confirm.py [{'|'.join(cmds)}]")
