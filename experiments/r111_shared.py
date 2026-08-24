"""R-111: does the CROSS-SECTIONAL SCORE ITSELF -- not its timing, not its
allocation weighting, both closed -- carry unexploited information on R-63's
own eight-asset panel?

Shared, frozen infrastructure for a two-branch parallel round. Per
ROUTINE.md's parallelism rules this file is neutral ground: both branches
import from it, neither branch edits it, and it does not itself compute a
verdict. It exists so the pre-registration and both branches' SCORE FORMULAS
are committed once, before either branch reads a single decisive number.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **INFO** -- the same constraint R-63 opened ("relative
information across contemporaneous price series... it is information this
repo has had committed to `data/` since R-57 and has never once used as a
cross-section"). R-63 through R-68 froze ONE way of extracting that relative
information (a multi-horizon trailing-return-vs-moving-average vote,
`kelly_regime_v4`'s own vote made continuous) and spent five rounds on WHEN
the portfolio acts on it (R-65 holding period, R-67 hysteresis, R-68 band
decomposition) and R-107/R-110 spent two more on HOW MUCH of the total goes
to each eligible name (equal-weight vs risk-parity vs a blend, 12
configurations, equal-weight winning every one). Neither axis is the SCORE
itself. R-110's own closing line, filed the same day this round starts,
names this directly: "should genuinely reconsider whether R-63's
cross-sectional score itself -- not its timing, not its allocation
weighting, both now exhausted -- is the thing worth varying next on the
multi-asset panel, since nothing on either previously-untouched dimension
has moved the verdict." This round is that reconsideration.

**Not a duplicate of:**

- R-63/65/67/68 (`cross_sectional_score`, imported unmodified everywhere
  since). Every one of those four rounds treats the score as frozen and
  varies membership timing. This round is the first to touch the score
  formula itself since R-63 wrote it.
- R-107/R-110 (allocation weighting: equal-weight vs risk-parity vs blend).
  Disjoint step -- they change how a FIXED eligible set's total notional is
  split; this round changes which assets ARE the eligible set in the first
  place. Both branches here keep k=1 (see below), which makes the R-107/
  R-110 allocation question mathematically vacuous on this round's own
  construction, by the identical logic R-107's own docstring used to justify
  sweeping `k` for ITS question -- the two axes cannot be conflated by
  accident here.
- R-57/R-59/R-60/R-61/R-62 (single-asset SIZE-axis replications on the
  panel, no cross-section formed) and every single-instrument round on
  `kelly_regime_v4`'s own vote (R-06, R-07, R-34, R-40, R-45, ...). None of
  them rank one asset against another; this round's two scores exist only as
  a cross-sectional ranking quantity.
- No prior ledger entry cites George & Hwang (2004), Da/Gurun/Warachka
  (2014), Blitz/Huij/Martens (2011) or Kim (2025/26) -- grepped, zero hits.

**Is it simulable here?** Yes, with zero new data: the same eight committed
5m spot series R-63 uses, through the identical `simulate_portfolio` engine,
the identical R-68 ENTRY_ONLY band-selection loop (`band_selection`, R-67/
R-68's winning architecture), and the identical D1/D2/D3/D5/scramble battery
R-63->R-68 and R-107/R-110 all report against. A candidate here and R-63's
own published baseline differ ONLY in the DataFrame handed to
`build_targets_from_score` as `score` -- exactly the property that made
every prior round's comparisons interpretable.

**What would make it fail (named now, before either score's numbers were
read).**

  (F1, both branches) **Near-duplication.** The new score could just be a
      noisier restatement of the SAME information the old one already
      extracts, in which case it should (a) rank assets almost identically
      to the old score bar-for-bar and (b) not move D1/D2 outside R-68's own
      published interval. Measured directly: cross-sectional Spearman rank
      correlation between the new and old scores, pooled over W_TRAIN, and
      the fraction of bars on which the two scores' own argmax (top-ranked
      asset) agree. Reported for both branches whether or not the decisive
      battery is reached.
  (F2, conservative / 52-week-high proximity) George & Hwang's (2004) result
      is a MONTHLY-rebalanced US-equity anomaly with a genuine 12-month
      lookback; this round's panel is six mid/small-cap crypto alts on a
      three-horizon 20/40/80-DAY structure, an order of magnitude shorter
      and a different asset class entirely (no earnings, no analyst
      coverage -- the anchoring-on-a-salient-reference-price behavioural
      story is unverified in crypto). Predicted failure mode: because
      `close <= rolling_max` by construction, proximity-to-high is
      structurally a MORE ONE-SIDED statistic than a moving-average
      deviation (it saturates at 0 on every new high, however strong the
      trend, while the old score keeps rising) -- so the new score may
      simply select the SAME leading asset less discriminatingly and
      **increase**, not decrease, membership churn as several assets
      cluster near their own local highs simultaneously. This is the
      opposite of R-68's own COST-axis finding and is watched for directly
      via `rebalances_per_day` against R-68's own published 0.102/day.
  (F3, novel / path-consistency) Da, Gurun & Warachka's (2014) effect and
      its 2025/26 crypto extension (Kim, SSRN #6889877) are measured on
      DAILY return signs over weeks-to-months, on samples of hundreds to
      thousands of names; Kim's own crypto result is reported STRONGEST
      OUTSIDE the largest-cap coins and WEAKEST for majors. This project's
      U6 panel (BCH, LTC, ETC, DASH, LINK, XTZ) is entirely the segment Kim's
      own result favours, which is why this is the arm with a same-era,
      same-asset-class prior behind it -- but the daily-sign-count statistic
      needs real degrees of freedom (an 80-day window is 80 daily
      observations per asset, not the hundreds Kim's own weekly-formation
      study used), so the predicted failure mode is INERT: too noisy at
      these window lengths to reweight anything, converging to a near-copy
      of the base score (high F1 rank correlation, no daily gate ever
      binding differently) rather than actively harmful.

=====================================================================
LITERATURE (Step 2 sources; full detail in each branch's own docstring)
=====================================================================

- George, T. J., & Hwang, C.-Y. (2004), "The 52-Week High and Momentum
  Investing," The Journal of Finance 59(5), 2145-2176. Ranks by nearness to
  a trailing high (`P_t / rolling_max`) rather than by trailing RETURN
  MAGNITUDE; finds it dominates and subsumes standard momentum's forecasting
  power, and that portfolios formed on it do not exhibit magnitude
  momentum's long-run reversal. US equities, monthly rebalance, 12-month
  high. The conservative branch's mechanism.
- Da, Z., Gurun, U. G., & Warachka, M. (2014), "Frog in the Pan: Continuous
  Information and Momentum," Review of Financial Studies 27(7), 2171-2218.
  Information Discreteness `ID = sign(PRET) * (%neg_days - %pos_days)` over
  the formation period; momentum built from smooth (low/negative-ID,
  many-small-same-sign-days) paths earns 5.94% vs -2.07% for jumpy
  (high-ID, few-large-days) paths at IDENTICAL cumulative return. US
  equities. The novel branch's mechanism.
- Kim, W. B. (2025/2026 working paper), "Price Path Continuity and the
  Cross-Section of Cryptocurrency Returns," SSRN #6889877. Direct crypto
  extension of Da/Gurun/Warachka, survivorship-bias-mitigated crypto sample,
  Jan 2020 - Apr 2026 -- overlapping this project's own window almost
  exactly. Short-horizon (~14-day) past return reverses on its own but
  becomes a continuation signal as the path generating it is smoother;
  effect concentrated OUTSIDE the largest-cap coins. Cited for (a) the
  closest same-era, same-asset-class prior this axis has ever had and (b)
  named failure mode F3.
- Blitz, D., Huij, J., & Martens, M. (2011), "Residual Momentum," Journal of
  Empirical Finance 18(3), 506-521. Considered and NOT chosen for either
  branch this round: it requires a factor-model regression (market beta)
  re-estimated per asset per bar on a panel of six thinly-correlated-to-
  uncorrelated alts against a single crypto "market" proxy this project does
  not have a validated construction for, which is a materially bigger
  methodological commitment than either chosen mechanism and was judged not
  worth the estimation risk for a first score-axis round. Named here so a
  future round does not have to re-discover why it was passed over.
- R-63/65/67/68/107/110's own citations (Moskowitz-Ooi-Pedersen 2012,
  Han-Kang-Ryu 2024, Grinold 1989, Clarke-de Silva-Thorley 2002) are the
  standing baseline this round's numbers are read against and are not
  re-cited in full here.

=====================================================================
UNIVERSE, WINDOWS, COSTS, BENCHMARKS, BATTERY -- INHERITED, NOT RESTATED
=====================================================================

Imported unmodified from `r63_shared.py` / `r63_novel_xsmom_rank.py` /
`r65_shared.py` / `r68_shared.py`, exactly as R-107/R-110 did:

  UNIVERSE_6, UNIVERSE_8, W_TRAIN, W_VAL, W_FULL6, W_HOLD
  SPOT_BASE (0.10%), SPOT_REAL (0.40%), SPOT_FREE (0bps, D5 only)
  simulate_portfolio, matched_hold_targets, static_hold_equity, align_frames,
  load_universe, check_causality, check_against_engine, compare,
  excludes_zero, config_count
  volmatched_hold_equity, realized_vol, turnover_stats, frontier_row
  d1_pass, d2_pass, d3_pass, d5_pass, D5_BAR_R68, scramble_fixed_perm
  conditional_vol_scale, basket_log_returns, DEADBAND (sizing constants)
  BARS_PER_DAY, BARS_PER_YEAR, HORIZONS (20, 40, 80), WARM_DAYS (91)

`band_selection` and its sizing half (`_size`) -- R-67/R-68's winning
membership architecture -- are REIMPLEMENTED below as
`build_targets_from_score`, byte-identical in LOGIC to
`r68_conservative_band_decomposition.band_selection`/`_size`
(cross-checked by `check_band_selection_matches_r68` below, which requires
this function to reproduce R-63's own baseline targets to full float
precision when handed R-63's own score and R-68's own frozen
delta_in=0.080, delta_out=0.0, k=1, buffer=0.05, hold_days=1). It is
reimplemented rather than imported because the R-68 file computes its own
score internally (`cross_sectional_score(aligned)`, hardcoded) with no
score parameter to inject -- this round's ENTIRE point is to inject a
different one.

FROZEN ARCHITECTURE (identical for both branches, identical to R-68's own
ENTRY_ONLY winner): k=1, hold_days=1 day, delta_out=0.0. The two things that
CANNOT simply be copied at R-68's raw numeric values are delta_in=0.080 and
buffer=0.05 -- both are RAW SCORE UNITS, fitted to the old score's own
distribution (`SIGMA_SCORE_W_TRAIN=0.2295`), and neither new score shares
that scale or, for the conservative score, even that SIGN CONVENTION (it is
bounded above by 0, not unbounded). Re-fitting either threshold on the new
score would reopen a sweep/selection this round is not scoped to run (it
would also inflate the trials count against a result this axis has never
once cleared). Instead:

  **δ_in and buffer are TRANSFERRED as R-68's/R-65's own SELECTED MULTIPLE
  OF THE SCORE'S OWN W_TRAIN STANDARD DEVIATION**, re-expressed on each new
  score's own measured scale. R-68 selected delta_in=0.080 against a
  measured `SIGMA_SCORE_W_TRAIN=0.2295` -- i.e. 0.3486 sigma. R-65 selected
  buffer=0.05 against the same sigma -- i.e. 0.2179 sigma. Those two
  RELATIVE multiples (REL_DELTA_IN, REL_BUFFER below), not the raw numbers,
  are what transfers: `delta_in_new = REL_DELTA_IN * sigma_new`,
  `buffer_new = REL_BUFFER * sigma_new`, where `sigma_new` is that SAME
  pooled-over-W_TRAIN standard deviation, measured fresh for each new score
  by `pooled_std_w_train` below. This is a ZERO-NEW-FITTED-PARAMETER
  design: no grid, no selection step, one frozen configuration per branch,
  decided by a rule stated here before either sigma was measured.

  **Both new scores are RECENTERED by their own pooled W_TRAIN mean before
  delta_in/delta_out are applied** (`recenter_score`) -- found necessary
  during this round's own pre-dispatch validation (a literal, uncentered
  transfer collapses `conservative_score` to permanently flat, since it is
  bounded above by 0 and measures pooled mean -0.203, so `delta_out=0.0`'s
  "hold while score > 0" is nearly never true). This is a construction fix,
  not a re-selection -- see `recenter_score`'s own docstring for the full
  reasoning and the pre-fix diagnostic that caught it. Applied uniformly to
  BOTH new scores (not only the one it visibly broke), never to the old
  score, which keeps its own already-published, uncentered convention.

  A NEIGHBOURHOOD CHECK (not a selection) is still run and reported, per
  ROUTINE step 4's plateau requirement: delta_in at 0.5x, 1.0x (frozen) and
  1.5x the transferred value, held fixed once measured, reported for every
  window this round touches. The DECISIVE read is the 1.0x point ONLY,
  frozen before any of the three neighbours was computed; the neighbours are
  transparency, not a re-selection, and the round's config count includes
  all three per branch regardless.

=====================================================================
DECISION RULES -- INHERITED, RESTATED FOR THIS ROUND'S OWN CELLS
=====================================================================

D1 (PRIMARY). W_FULL6, U6, spot @0.10%, candidate vs VOLMATCH_HOLD (R-65's
    risk-matched arm). PASS iff point estimate > 0 and 95% interval excludes
    zero (`d1_pass`, imported).
D2 (PRIMARY-2). Same cell, max drawdown, PASS iff point estimate < 0 and
    interval excludes zero (`d2_pass`, imported). D1 or D2 must pass.
D3 (INNER-VALIDATION). W_VAL, U8, spot @0.10%, directional only (`d3_pass`,
    imported).
D5 (SIGNAL RETENTION). W_FULL6, U6, SPOT_FREE (0bps), gross growth vs
    VOLMATCH_HOLD >= D5_BAR_R68 (+0.342, R-68's own corrected bar, imported
    unmodified -- this round inherits it rather than re-deriving one,
    because it measures whether ENOUGH signal survives fees at all, a
    property of the axis's OWN cost structure, not of which score feeds it).
SCRAMBLE (FALSIFICATION, can kill a branch on its own). `scramble_fixed_perm`
    on the D1 cell, 10 fixed seeds, imported unmodified. FAIL if the
    candidate's D1 point estimate does not exceed the 90th percentile of the
    10 scrambled runs.

FURTHER-WORK BAR (does not by itself authorize a holdout read): R-107/
R-110's own four-clause form, `(d1 or d2) and d3 and d5 and
scramble_survived` -- no M1' clause, for the identical reason R-107 gave:
this round changes neither entry nor exit TIMING at all (k, hold_days,
delta_out are all inherited unchanged), so there is nothing for a
membership-change-rate gate to measure that D1/D2/D3/D5 do not already
cover more directly.

PROMOTION BAR (only reachable after an authorized holdout read; default
REJECT): R-63's own, unmodified -- beats EW_HOLD/BTC_HOLD/VOLMATCH_HOLD
out-of-sample after real costs; improvement exceeds the +/-0.2 Sharpe noise
floor (R-20) or is a drawdown/tail improvement; survives the scramble
control on the holdout cell too; the delta_in neighbourhood is a plateau.
Nothing found by either branch can be REGISTERED as a new multi-asset
strategy regardless of verdict on the SAME grounds R-107/R-110 both state:
this is a bar-by-bar cross-asset allocator and the one thing that WOULD be
registrable on this axis (`xsmom_entry_band`, R-68's own winner) already is.

CONFIGURATIONS EVALUATED. Counted by `config_count()`, shared process-wide
with every prior round descended from `r63_shared.py`. Each branch reports
the DELTA (before/after its own run), and this round's total is the SUM
across both branches, per ROUTINE's parallelism rules.

HOLDOUT-READ CONVENTION. Same disclosed departure R-63->R-68/R-107/R-110 all
use: W_FULL6 already spans through the last committed bar on U6, counted as
"+0" by the project-wide convention that the RESERVED BTC/ETH 2023+ holdout,
not the U6 panel, is what "+0" refers to. A genuine W_HOLD (U8, BTC/ETH
included) read, only if the further-work bar is cleared, is the only kind
that increments the running total.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r63_shared import (  # noqa: E402,F401
    BOOT_KW,
    OUT_DIR as R63_OUT_DIR,
    SCRAMBLE_SEEDS,
    SPOT_BASE,
    SPOT_REAL,
    START_BALANCE,
    TOTAL_NOTIONAL_DEADBAND,
    UNIVERSE_6,
    UNIVERSE_8,
    W_FULL6,
    W_HOLD,
    W_TRAIN,
    W_VAL,
    align_frames,
    check_against_engine,
    check_causality,
    compare,
    config_count,
    excludes_zero,
    load_universe,
    matched_hold_targets,
    mean_total_notional,
    scramble_targets,
    simulate_portfolio,
    static_hold_equity,
)
from experiments.r63_novel_xsmom_rank import (  # noqa: E402,F401
    ANCHOR_SPAN_DAYS,
    BARS_PER_DAY,
    BARS_PER_YEAR,
    DEADBAND,
    HIGH_IN,
    HIGH_OUT,
    HORIZONS,
    LOW_IN,
    LOW_OUT,
    MAX_LEVERAGE,
    TARGET_VOL,
    VOL_SPAN,
    WARM_DAYS,
    basket_log_returns,
    conditional_vol_scale,
    cross_sectional_score as r63_cross_sectional_score,
    warm_window,
)
from experiments.r63_novel_xsmom_rank import build_targets as r63_baseline_targets  # noqa: E402,F401
from experiments.r65_shared import (  # noqa: E402,F401
    R63_GROSS_EDGE,
    R63_GROSS_EDGE_VS_VOLMATCH,
    R63_NET_D1,
    R63_TURNOVER_PER_DAY,
    SPOT_FREE,
    frontier_row,
    holding_period_days,
    realized_vol,
    turnover_stats,
    volmatched_hold_equity,
)
from experiments.r68_shared import (  # noqa: E402,F401
    D5_BAR_R68,
    SIGMA_SCORE_W_TRAIN,
    d1_pass,
    d2_pass,
    d3_pass,
    d5_pass,
    scramble_fixed_perm,
)

OUT_DIR = ROOT / "reports" / "r111_score_variants"

# ------------------------------------------------------------- transfer rule

REL_DELTA_IN = 0.080 / SIGMA_SCORE_W_TRAIN   # R-68's own selected multiple
REL_BUFFER = 0.05 / SIGMA_SCORE_W_TRAIN      # R-65's own selected multiple
K_FIXED = 1
HOLD_DAYS_FIXED = 1.0
DELTA_OUT_FIXED = 0.0
NEIGHBOUR_MULTIPLIERS = (0.5, 1.0, 1.5)      # 1.0 is the frozen decisive point


def _w_train_slice(score: pd.DataFrame, window=W_TRAIN) -> np.ndarray:
    start, end = window
    sub = score.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")
                    + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)]
    vals = sub.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        raise ValueError("no finite score values in W_TRAIN")
    return vals


def pooled_std_w_train(score: pd.DataFrame, window=W_TRAIN) -> float:
    """The new score's own W_TRAIN standard deviation, pooled across assets
    and bars -- the exact quantity `SIGMA_SCORE_W_TRAIN` measured for the old
    score, reproduced here for a new score so `REL_DELTA_IN`/`REL_BUFFER`
    have something to multiply.
    """
    return float(np.std(_w_train_slice(score, window), ddof=1))


def pooled_mean_w_train(score: pd.DataFrame, window=W_TRAIN) -> float:
    """The new score's own W_TRAIN pooled mean -- see `recenter_score` for
    why this is measured at all."""
    return float(np.mean(_w_train_slice(score, window)))


def recenter_score(score: pd.DataFrame, window=W_TRAIN) -> tuple[pd.DataFrame, float]:
    """CORRECTION, found during this round's own pre-dispatch validation,
    before either branch's decisive battery ran -- the ROUTINE-permitted
    "fix a bug" case (step 3), not a re-selection: the old score's own
    `delta_out=0.0` convention means "hold_eligible = score > 0", which is a
    genuinely loose condition for the OLD score because it is
    APPROXIMATELY ZERO-CENTERED over W_TRAIN (pooled mean +0.070) -- but
    `conservative_score` is bounded ABOVE by 0 by construction (a rolling
    max is never smaller than the current close) and measures pooled mean
    -0.203 over W_TRAIN, so a literal `score > 0` transfer would require
    being at an exact simultaneous new high on all three horizons to hold
    ANYTHING -- collapsing the arm to permanently flat, an artifact of the
    new score's bounded range, not a finding about its economic content.
    (Verified directly: pre-fix, `build_targets_from_score` on
    `conservative_score` at the literal transferred `delta_in`/`delta_out=0`
    produces zero trades over the whole W_TRAIN window.)

    FIX: both new scores are recentered by their OWN pooled W_TRAIN mean
    before `delta_in`/`delta_out` are applied, so `delta_out=0.0` means the
    same thing for every score -- "hold while at least as good as this
    score's own W_TRAIN-average level" -- regardless of where that score's
    raw values happen to sit. This is NOT a new fitted parameter: the mean
    is measured, not selected against any objective, and it collapses to a
    ~+0.07 no-op shift for a score that is already close to zero-centered
    (as `novel_score` measures: pooled mean +0.043, negligible next to its
    own sigma of ~0.134). The OLD score's own already-published R-63/65/67/
    68/107/110 numbers are UNTOUCHED by this -- centering is applied only to
    this round's two new scores, never to `r63_cross_sectional_score`.

    Returns ``(recentered_score, center)``.
    """
    center = pooled_mean_w_train(score, window)
    return score - center, center


def transferred_thresholds(score: pd.DataFrame) -> dict:
    """The zero-new-fitted-parameter transfer this round's whole design
    rests on, applied to the RECENTERED score (see `recenter_score`).
    Returns sigma, the measured center, and the three (delta_in, buffer)
    pairs at NEIGHBOUR_MULTIPLIERS, keyed by multiplier, with delta_out/k/
    hold_days restated for convenience. `sigma` is invariant to centering
    (subtracting a constant does not change a standard deviation), so it is
    measured on the raw score directly.
    """
    sigma = pooled_std_w_train(score)
    center = pooled_mean_w_train(score)
    out = {"sigma": sigma, "center": center, "delta_out": DELTA_OUT_FIXED,
           "k": K_FIXED, "hold_days": HOLD_DAYS_FIXED}
    for m in NEIGHBOUR_MULTIPLIERS:
        out[m] = {
            "delta_in": m * REL_DELTA_IN * sigma,
            "buffer": m * REL_BUFFER * sigma,
        }
    return out


# ------------------------------------------------- band selection (R-67/R-68)


def band_selection(s: np.ndarray, k: int, buffer: float, hold_days: float,
                   delta_in: float, delta_out: float):
    """R-68's `band_selection`, reimplemented byte-identical in LOGIC
    (cross-checked below) so it can take an arbitrary score matrix rather
    than calling R-63's hardcoded `cross_sectional_score` internally.
    """
    n, n_assets = s.shape
    finite = np.isfinite(s)
    enter_eligible = finite & (s > float(delta_in))
    hold_eligible = finite & (s > -float(delta_out))
    hold_bars = int(round(float(hold_days) * BARS_PER_DAY))
    buf = float(buffer)

    sel = np.zeros((n, n_assets), dtype=bool)
    held: list[int] = []
    last_change = -(1 << 60)
    keys = ("forced_exit", "entry", "swap", "blocked_by_timer",
            "blocked_by_buffer", "flat_bars")
    ev = {key: 0 for key in keys}
    ev_bars = {key: np.zeros(n, dtype=np.int32) for key in keys}

    for i in range(n):
        row = s[i]
        elig_in = enter_eligible[i]
        elig_hold = hold_eligible[i]
        changed = False

        if held:
            keep = [a for a in held if elig_hold[a]]
            if len(keep) != len(held):
                ev["forced_exit"] += 1
                ev_bars["forced_exit"][i] = 1
                held = keep
                changed = True

        if len(held) < k:
            free = [a for a in range(n_assets) if elig_in[a] and a not in held]
            if free:
                free.sort(key=lambda a: -row[a])
                for a in free[: k - len(held)]:
                    held.append(a)
                    changed = True
                ev["entry"] += 1
                ev_bars["entry"][i] = 1

        elif not changed:
            worst = min(held, key=lambda a: row[a])
            out = [a for a in range(n_assets) if elig_in[a] and a not in held]
            if out:
                best = max(out, key=lambda a: row[a])
                if row[best] > row[worst] + buf:
                    if (i - last_change) >= hold_bars:
                        held.remove(worst)
                        held.append(best)
                        changed = True
                        ev["swap"] += 1
                        ev_bars["swap"][i] = 1
                    else:
                        ev["blocked_by_timer"] += 1
                        ev_bars["blocked_by_timer"][i] = 1
                elif row[best] > row[worst]:
                    ev["blocked_by_buffer"] += 1
                    ev_bars["blocked_by_buffer"][i] = 1

        if changed:
            last_change = i
        if held:
            sel[i, held] = True
        else:
            ev["flat_bars"] += 1
            ev_bars["flat_bars"][i] = 1

    return sel, ev, ev_bars


def _size(sel: np.ndarray, aligned: dict[str, pd.DataFrame], k: int,
          index, assets) -> pd.DataFrame:
    """R-63's sizing block, copied byte-for-byte -- unmodified by this
    round, which changes only WHICH assets `sel` marks eligible."""
    n = sel.shape[0]
    m = sel.sum(axis=1)
    scale = conditional_vol_scale(basket_log_returns(aligned))

    desired = scale * (m / float(k))
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        d = desired[i]
        if abs(d - cur) > DEADBAND:
            cur = d
        pos[i] = cur

    total = np.minimum(pos, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(m > 0, total / np.maximum(m, 1), 0.0)
    w = sel * per[:, None]
    return pd.DataFrame(w, index=index, columns=assets)


def build_targets_from_score(aligned: dict[str, pd.DataFrame], score: pd.DataFrame,
                             k: int, buffer: float, hold_days: float,
                             delta_in: float, delta_out: float):
    """The one function both branches call: score in, target-weight matrix
    out, via R-68's ENTRY_ONLY architecture. Returns
    ``(targets, event_ledger, event_bars)``.
    """
    assets = list(score.columns)
    s = score.to_numpy(dtype=float)
    sel, ev, ev_bars = band_selection(s, k, buffer, hold_days, delta_in, delta_out)
    return _size(sel, aligned, k, score.index, assets), ev, ev_bars


def check_band_selection_matches_r68(frames=None, bars: int = 60_000) -> tuple[bool, float]:
    """Correctness gate: this file's `band_selection`/`_size`, fed R-63's OWN
    score at R-68's OWN frozen (k=1, buffer=0.05, hold_days=1, delta_in=0.080,
    delta_out=0.0), must reproduce `r63_baseline_targets`-family output.
    Compared via a full W_TRAIN simulate_portfolio run (SPOT_BASE) rather
    than the raw matrix, since R-63's own `build_targets` applies its own
    k=1 equal-weight sizing through the identical `_size` block reused here
    -- so this is exactly R-68's own `check_against_engine`-style transitive
    check, extended one more hop.
    """
    if frames is None:
        frames = load_universe(UNIVERSE_8)
    aligned = align_frames(frames, warm_window(W_TRAIN))
    idx = aligned[UNIVERSE_8[0]].index

    score = r63_cross_sectional_score(aligned)
    mine, _, _ = build_targets_from_score(aligned, score, k=1, buffer=0.05,
                                          hold_days=1.0, delta_in=0.080,
                                          delta_out=0.0)
    theirs, _, _, _ = __import__(
        "experiments.r68_conservative_band_decomposition", fromlist=["x"]
    ).build_band_targets_ev(aligned, k=1, buffer=0.05, hold_days=1.0,
                            delta_in=0.080, delta_out=0.0)

    eval_start = pd.Timestamp(W_TRAIN[0], tz="UTC")
    mine_e = mine.loc[mine.index >= eval_start]
    theirs_e = theirs.loc[theirs.index >= eval_start]
    err = float(np.max(np.abs(mine_e.to_numpy() - theirs_e.to_numpy())))
    return bool(err <= 1e-9), err


# ------------------------------------------------- THE TWO SCORE FORMULAS
#
# Both are this round's ONLY new machinery besides the threshold-transfer
# rule above. Written here, in the shared file, rather than split one per
# branch file: the daily-causal lookup the novel score needs is exactly the
# kind of computation this project's own history (R-21's i+1 lookahead, R-63's
# own `_hi()` off-by-one, R-72's B-33 finding) shows is easy to get subtly
# wrong, so it is written once, causality-checked once (`check_causality`,
# imported, run against BOTH scores below before either branch's decisive
# battery), and both branches import rather than each re-deriving it.


def conservative_score(aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """George & Hwang (2004): rank by NEARNESS TO THE TRAILING HIGH, not by
    trailing return magnitude. Structurally parallel to R-63's own
    `cross_sectional_score` -- same three horizons, same per-horizon
    averaging -- swapping the anchor (`rolling(h).mean()`) for a trailing
    high (`rolling(h).max()`) and the deviation term accordingly:

        score_i(t) = mean_h ( close_i(t) / rolling_max_h(close_i)(t) - 1 )

    Bounded above by 0 (a new high on every horizon simultaneously); more
    negative the further price sits below its own recent high on each
    horizon. `rolling(...).max()` at row t uses rows <= t only -- causal by
    the identical construction as the old score's `rolling(...).mean()`.
    """
    cols = {}
    for t, df in aligned.items():
        close = df["close"]
        acc = None
        for h in HORIZONS:
            roll_high = close.rolling(int(h * BARS_PER_DAY)).max()
            term = close / roll_high - 1.0
            acc = term if acc is None else acc + term
        cols[t] = acc / len(HORIZONS)
    return pd.DataFrame(cols, index=next(iter(aligned.values())).index)


def _daily_consistency_by_horizon(aligned: dict[str, pd.DataFrame], universe,
                                  horizon_days: int) -> pd.DataFrame:
    """Da/Gurun/Warachka (2014) path-consistency, one horizon, DAILY grid.

    For calendar day D, uses ONLY daily log returns strictly BEFORE D (the
    window ending at day D-1's close) -- day D's own not-yet-realized return
    never enters its own day's value, the identical per-day-lookup causality
    convention `r107_shared.build_cov_lookup` already uses in this repo for
    a rolling covariance. Per asset, per day D:

        trend_sign  = sign(sum of the trailing `horizon_days` daily log
                      returns, strictly before D)
        consistency = fraction of those days whose OWN sign matches
                      trend_sign (0.5 on a day with no prior history, or
                      when trend_sign == 0 -- genuinely uninformative, not a
                      missing value, so it must not silently drop out of the
                      mean the caller takes across horizons)

    Equivalent, up to the known affine map, to Da/Gurun/Warachka's own
    `ID = sign(PRET) * (%neg - %pos)`: `consistency = (1 - ID) / 2`. A smooth,
    continuous trend (their "frog in the pan" case, ID very negative) maps to
    consistency near 1; a jumpy, discrete trend (ID positive) maps to
    consistency near 0.
    """
    closes = pd.DataFrame({t: aligned[t]["close"] for t in universe})
    daily_close = closes.resample("1D").last()
    daily_ret = np.log(daily_close).diff()
    dates = daily_ret.index
    vals = daily_ret.to_numpy(dtype=float)
    n, k = vals.shape

    out = np.full((n, k), 0.5)
    for i in range(n):
        lo = max(0, i - horizon_days)
        window = vals[lo:i]  # STRICTLY before day i
        if len(window) < max(5, horizon_days // 4):
            continue
        finite = np.isfinite(window)
        count = finite.sum(axis=0)
        trend = np.where(count > 0, np.nansum(np.where(finite, window, 0.0), axis=0), 0.0)
        trend_sign = np.sign(trend)
        same = np.where(finite, np.sign(window) == trend_sign, False)
        frac = np.divide(same.sum(axis=0), np.maximum(count, 1),
                         out=np.full(k, 0.5), where=count > 0)
        frac = np.where(trend_sign == 0, 0.5, frac)
        out[i] = frac

    return pd.DataFrame(out, index=dates, columns=universe)


def novel_score(aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Da/Gurun/Warachka (2014) path-consistency, applied as a multiplicative
    weight on R-63's own per-horizon magnitude term (Kim 2025/26's crypto
    extension, same era, same asset class as this project's own window):

        score_i(t) = mean_h ( (close_i(t)/anchor_{i,h}(t) - 1)
                              * consistency_i(t, h) )

    `consistency_i(t, h)` is day D's causal value (built from days strictly
    before D, see `_daily_consistency_by_horizon`) held constant across every
    5-minute bar inside day D -- forward-filled from the daily grid onto the
    bar grid, which cannot leak day D's own return into day D's own bars
    since the value itself never depends on day D at all.
    """
    universe = list(aligned.keys())
    idx = next(iter(aligned.values())).index
    day_key = idx.floor("D")

    cons_by_h = {h: _daily_consistency_by_horizon(aligned, universe, h) for h in HORIZONS}

    cols = {}
    for t in universe:
        close = aligned[t]["close"]
        acc = None
        for h in HORIZONS:
            anchor = close.rolling(int(h * BARS_PER_DAY)).mean()
            base_term = (close / anchor - 1.0)
            cons_daily = cons_by_h[h][t]
            cons_bar = cons_daily.reindex(day_key).to_numpy()
            weighted = pd.Series(base_term.to_numpy() * cons_bar, index=idx)
            acc = weighted if acc is None else acc + weighted
        cols[t] = acc / len(HORIZONS)
    return pd.DataFrame(cols, index=idx)


# ------------------------------------------------------ F1 rank-agreement

def rank_agreement(score_a: pd.DataFrame, score_b: pd.DataFrame, window=W_TRAIN) -> dict:
    """Named failure mode F1: how similar is a new score's cross-sectional
    ranking to R-63's own, on the SAME window? Pooled Spearman rank
    correlation (bar-by-bar rank vectors, concatenated) and the fraction of
    bars whose single top-ranked asset (argmax) agrees. Both scores must
    share the same columns/index.
    """
    start, end = window
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    a = score_a.loc[lo:hi]
    b = score_b.loc[lo:hi].reindex(columns=a.columns)

    a_np = a.to_numpy(dtype=float)
    b_np = b.to_numpy(dtype=float)
    finite = np.isfinite(a_np).all(axis=1) & np.isfinite(b_np).all(axis=1)
    a_np, b_np = a_np[finite], b_np[finite]

    a_rank = pd.DataFrame(a_np).rank(axis=1).to_numpy().ravel()
    b_rank = pd.DataFrame(b_np).rank(axis=1).to_numpy().ravel()
    rho = float(np.corrcoef(a_rank, b_rank)[0, 1]) if len(a_rank) > 1 else float("nan")

    a_arg = np.argmax(a_np, axis=1)
    b_arg = np.argmax(b_np, axis=1)
    agree = float(np.mean(a_arg == b_arg)) if len(a_arg) else float("nan")

    return {"n_bars": int(finite.sum()), "spearman_rho": rho, "argmax_agreement": agree}


__all__ = [
    "ANCHOR_SPAN_DAYS", "BARS_PER_DAY", "BARS_PER_YEAR", "BOOT_KW",
    "DEADBAND", "DELTA_OUT_FIXED", "D5_BAR_R68", "HIGH_IN", "HIGH_OUT",
    "HOLD_DAYS_FIXED", "HORIZONS", "K_FIXED", "LOW_IN", "LOW_OUT",
    "MAX_LEVERAGE", "NEIGHBOUR_MULTIPLIERS", "OUT_DIR", "REL_BUFFER",
    "REL_DELTA_IN", "R63_GROSS_EDGE", "R63_GROSS_EDGE_VS_VOLMATCH",
    "R63_NET_D1", "R63_TURNOVER_PER_DAY", "SCRAMBLE_SEEDS",
    "SIGMA_SCORE_W_TRAIN", "SPOT_BASE", "SPOT_FREE", "SPOT_REAL",
    "START_BALANCE", "TARGET_VOL", "TOTAL_NOTIONAL_DEADBAND", "UNIVERSE_6",
    "UNIVERSE_8", "VOL_SPAN", "WARM_DAYS", "W_FULL6", "W_HOLD", "W_TRAIN",
    "W_VAL", "align_frames", "band_selection", "basket_log_returns",
    "build_targets_from_score", "check_against_engine",
    "check_band_selection_matches_r68", "check_causality",
    "conditional_vol_scale", "compare", "config_count",
    "conservative_score", "novel_score", "d1_pass", "d2_pass",
    "d3_pass", "d5_pass", "excludes_zero", "frontier_row",
    "holding_period_days", "load_universe", "matched_hold_targets",
    "mean_total_notional", "pooled_mean_w_train", "pooled_std_w_train",
    "recenter_score", "r63_baseline_targets",
    "r63_cross_sectional_score", "rank_agreement", "realized_vol",
    "scramble_fixed_perm", "scramble_targets", "simulate_portfolio",
    "static_hold_equity", "transferred_thresholds", "turnover_stats",
    "volmatched_hold_equity", "warm_window",
]
