#!/usr/bin/env python
"""R-55 CONSERVATIVE branch, closing backlog item B-22: a magnitude-AND-
DURATION filter on the aggregate-USDT-stablecoin-supply-deceleration hard
veto R-54's novel branch built (`kelly_regime_v15_stablecoin_veto.py`).

Idea, one sentence
------------------
Transient day-to-day wobbles in aggregate USDT supply growth reverse
within a few days and carry no forward information about price weakness,
so requiring the latched stablecoin-stress vote to stay continuously
above `thresh_hi` for a minimum number of consecutive days (`persist_days`)
before it is allowed to actually veto (force `frac=0`), rather than
firing on any single-day threshold crossing, should filter out most of
the transient stress-onset false positives R-54 diagnosed (24 onset
events at the tightest threshold, thresh_hi=0.75/gap=0.0) while
preserving most of the genuine multi-week early-warning lead time R-54
confirmed against v4's own 3-anchor majority gate (9/12 matched episodes,
median +16.5 days -- the first INFO-axis signal in this project's history
to lead rather than lag).

Constraint attacked: INFO, same axis as R-54 (third attempt: B-07/R-44
on-chain activity failed, R-53 external macro failed, R-54's own hard
veto on this exact signal confirmed the lead-time hypothesis but still
lost on precision). This round does not introduce a new information
channel -- it is a minimal architectural variant of R-54's own hard veto,
adding exactly one new free parameter (`persist_days`) and touching
nothing else.

Not a duplicate of, cited precisely
------------------------------------
- `kelly_regime_v15_stablecoin_veto.py` (R-54 NOVEL, this file's direct
  ancestor): SAME hard-override architecture (`frac=0` while latched vote
  reads "stress", v4's own unmodified 3-anchor average otherwise), SAME
  signal (`stablecoin_stress_z`, imported unchanged from
  `experiments._stablecoin_signal.compute_stablecoin_stress`), but the
  vote's ENTRY into "stress" additionally requires `persist_days` of
  continuous confirmation above `thresh_hi` -- R-54 fired on any single
  bar/day crossing. `persist_days=0` is this file's exact negative
  control and is verified below to reproduce R-54's `_stable_vote` output
  bit-for-bit (Step 0 sanity check, before anything else runs).
- `kelly_regime_v15_macro_veto.py` (R-54 CONSERVATIVE, VIX/DXY-fed): not
  read, not touched, disjoint signal and disjoint files.
- This round's own disjoint parallel NOVEL branch
  (`experiments/kelly_regime_v16_stablecoin_confirm.py`, presumably a
  confirming/non-overriding combination-rule attack on the SAME B-22
  backlog item, per the operator's brief): NOT read, not coordinated
  with, per ROUTINE.md's parallelism isolation rule.
- **B-22** (this project's own backlog item, filed by R-54): "a
  magnitude-and-duration filter... require the vote to persist for a
  minimum number of days before it can veto, rather than firing on any
  single-bar threshold crossing" -- this file is exactly that, the first
  of B-22's two named next steps.

Sources
-------
- Shu, Yu & Mulvey (2024), "Downside Risk Reduction Using Regime-
  Switching Signals: A Statistical Jump Model Approach," arXiv:2402.05272
  -- already cited in this project's own R-02 ledger row for its jump-
  penalty mechanism; cited HERE specifically for its independent use of a
  minimum-duration/persistence requirement before confirming a regime
  change, precisely to trade detection speed against false-positive rate
  -- the general form of the idea this file tests concretely.
- Industry practice on stablecoin-flow persistence windows for
  confirmation commonly falls in a 3-7 day range (web research for this
  round) -- motivation for this file's parameter-grid CENTER, not a
  number copied blindly; the actual grid spans 0-14 days specifically so
  the 3-7 day literature-motivated region is interior to the grid, not an
  edge choice.

Mechanism, precisely (minimal change from v15)
-----------------------------------------------
v15's `_stable_vote` latches to "stress" (0) the instant
`stablecoin_stress_z` crosses above `thresh_hi`, and back to "calm" (1)
the instant it crosses below `thresh_lo = thresh_hi - gap`. This file's
`_stable_vote_persist` changes ONLY the entry condition: instead of the
raw single-day crossing, entry requires `stablecoin_stress_z` to have
read continuously above `thresh_hi` for `persist_days` consecutive
calendar days (inclusive of the crossing day itself, so `persist_days=0`
means "no confirmation delay," the exact single-day-crossing behavior
v15 used). The EXIT condition (recovery to "calm") is left exactly as
v15 had it, unfiltered -- R-54's own diagnosis was that the false
positives are fast-REVERSING ONSETS, not slow recoveries, so filtering
only the entry side is the minimal, targeted change; filtering the exit
side too would be a second, untested mechanism and is deliberately not
attempted here.

Code reuse decision, stated plainly (per this round's instruction)
--------------------------------------------------------------------
The anchor-vote helper (`_anchor_votes`), the base latched-hysteresis
vote (`_stable_vote`, kept here ONLY as the reference implementation for
the persist_days=0 identity check, not used by the strategy class
itself), and the lead-time diagnostic helpers (`_daily_transitions`,
`nearest`) are DUPLICATED (not imported) from
`kelly_regime_v15_stablecoin_veto.py`, which itself duplicated them from
`kelly_regime_v14_macro_lead.py` -- the same norm both R-53 and R-54
established of not sharing code across branches except through an
explicitly-designated shared module (here, `_stablecoin_signal.py`,
imported unchanged via `compute_stablecoin_stress`, exactly as R-54's
own file did). Neither `kelly_regime_v15_stablecoin_veto.py` nor
`kelly_regime_v14_macro_lead.py` is edited anywhere in this session.

Pre-registered falsification test (named before any code ran)
-----------------------------------------------------------------
(a) Does adding the persistence filter PRESERVE OR IMPROVE R-54's
    lead-time-vs-3-anchor-majority result? Checked explicitly across the
    full `persist_days` grid (0, 1, 2, 3, 5, 7, 10, 14) at the primary
    thresh/gap, reporting median lead time and the leads/lags ratio at
    each persist_days value, to see explicitly WHERE (if anywhere) the
    lead-time advantage collapses as confirmation delay grows.
(b) Does the best surviving configuration (by inner-validation Sharpe,
    subject to the plateau requirement) pass the SAME pre-2020
    BTC-control vs ETH falsification test R-54 used?
Both (a) and (b) must hold, AND the candidate must beat v4 on
inner-validation Sharpe outside the +/-0.2 noise floor with a plateau
neighbourhood, before any holdout read is considered -- the exact
decision rule is restated in full, with its own section, before the
holdout is (or is not) touched, at the bottom of this file's companion
report.

Usage
-----
    python experiments/kelly_regime_v16_stablecoin_persist.py identity    # persist_days=0 == v15 sanity check
    python experiments/kelly_regime_v16_stablecoin_persist.py descriptive
    python experiments/kelly_regime_v16_stablecoin_persist.py leadtime    # THE (a) falsification check
    python experiments/kelly_regime_v16_stablecoin_persist.py sweep       # step 3 (inner-train)
    python experiments/kelly_regime_v16_stablecoin_persist.py select      # step 3 (inner-validation)
    python experiments/kelly_regime_v16_stablecoin_persist.py artifact    # exposure-artifact check
    python experiments/kelly_regime_v16_stablecoin_persist.py causality   # lookahead probe (price + stablecoin pathway)
    python experiments/kelly_regime_v16_stablecoin_persist.py eth         # ETH falsification -- THE (b) check
    python experiments/kelly_regime_v16_stablecoin_persist.py all         # everything, in order
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments._stablecoin_signal import compute_stablecoin_stress  # noqa: E402

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures 5x", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")     # inner-train
VALID = ("2021-01-01", "2022-12-31")     # inner-validation
OOS_START = "2023-01-01"                 # never read in this file

INCUMBENT = "kelly_regime_v4"
DATA_DIR = ROOT / "data"

N_EVALUATED = 0  # distinct configurations evaluated, project-trials count
_SEEN_CONFIGS: set[str] = set()


# --------------------------------------------------------------------- data


def build_stablecoin_dataframe() -> tuple[pd.DataFrame, str]:
    """Canonical spot OHLCV with a causal ``stablecoin_stress_z_visible`` column merged on.
    Identical construction to v15's own -- same data, same alignment."""
    spot, label = load_dataset(DATA_DIR, "spot")
    stress = compute_stablecoin_stress(spot, DATA_DIR)
    out = spot.copy()
    out["stablecoin_stress_z_visible"] = stress
    return out, label


DF, LABEL = build_stablecoin_dataframe()
print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
      f"(data: {LABEL}); stablecoin-stress coverage {DF['stablecoin_stress_z_visible'].notna().sum():,} bars "
      f"from {DF['stablecoin_stress_z_visible'].dropna().index[0]:%Y-%m-%d}", file=sys.stderr)


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0,
            config_key: str | None = None):
    """One backtest -> Metrics. ``config_key`` counts a DISTINCT configuration
    exactly once across the whole session, however many market/period cells
    it is subsequently re-scored on (v4 control and diagnostic re-reads pass
    config_key=None and are never counted)."""
    global N_EVALUATED
    if config_key is not None and config_key not in _SEEN_CONFIGS:
        _SEEN_CONFIGS.add(config_key)
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                         start_balance=balance, data_label=LABEL)
    return compute_metrics(result), result


def line(tag, m, market_name=""):
    print(f"  {tag:42s} {market_name:10s} final=${m.final_balance:>11,.0f} "
          f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>7,.0f}{'  LIQUIDATED' if m.liquidated else ''}")


# ---------------------------------------------------------------- the vote


def _anchor_votes(close: pd.Series, horizons=(20, 40, 80), band: float = 0.01) -> dict:
    """Exactly kelly_regime_v4's own per-anchor latched vote. Duplicated
    (not imported) from kelly_regime_v15_stablecoin_veto.py, which itself
    duplicated it from kelly_regime_v14_macro_lead.py -- see this module's
    docstring, "Code reuse decision"."""
    votes = {}
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes[days] = v.ffill().fillna(0.0)
    return votes


def _stable_vote(stress_z: pd.Series, thresh_hi: float, gap: float) -> pd.Series:
    """R-54's ORIGINAL single-day-crossing latched vote -- duplicated
    unchanged from ``kelly_regime_v15_stablecoin_veto.py`` and kept here
    ONLY as the reference implementation for this file's persist_days=0
    identity check (not called anywhere in the strategy class itself,
    which always goes through ``_stable_vote_persist`` below)."""
    thresh_lo = thresh_hi - gap
    raw = np.where(stress_z > thresh_hi, 0.0,
                    np.where(stress_z < thresh_lo, 1.0, np.nan))
    vote = pd.Series(raw, index=stress_z.index).ffill().fillna(1.0)
    return vote


def _persistence_confirmed(stress_z: pd.Series, thresh_hi: float, persist_days: int) -> pd.Series:
    """Bar-level boolean, True at bar t iff ``stress_z`` has read
    continuously above ``thresh_hi`` for at least ``persist_days``
    consecutive CALENDAR DAYS up to and including t's calendar day.
    ``persist_days=0`` means "no confirmation delay" -- reduces to the
    raw single-day crossing v15 used (this file's exact negative
    control). Causal by construction: the daily resample only uses
    ``.last()`` within each already-elapsed day, and the rolling window
    only looks backward from the current day; ``reindex(..., ffill)``
    only ever projects an already-known day's confirmed state forward
    onto later bars of the SAME day, never sideways or backward in time.
    """
    daily = stress_z.resample("1D").last().ffill()
    above_daily = (daily > thresh_hi)
    if persist_days <= 0:
        confirmed_daily = above_daily
    else:
        window = persist_days + 1  # crossing day + persist_days confirmation days
        confirmed_daily = (
            above_daily.astype(float).rolling(window, min_periods=window).min() == 1.0
        )
        confirmed_daily = confirmed_daily.fillna(False)
    return confirmed_daily.reindex(stress_z.index, method="ffill").fillna(False)


def _stable_vote_persist(stress_z: pd.Series, thresh_hi: float, gap: float,
                          persist_days: int) -> pd.Series:
    """Same latched hysteresis vote as v15's ``_stable_vote``, except entry
    into "stress" additionally requires ``persist_days`` of confirmation
    via ``_persistence_confirmed`` above. Exit back to "calm" is
    UNCHANGED (raw ``stress_z`` crossing below ``thresh_lo``, immediate,
    no persistence requirement) -- per this file's pre-registered
    mechanism, R-54's diagnosed false positives are fast-reversing
    ONSETS, not slow recoveries, so only the entry side is filtered.
    """
    thresh_lo = thresh_hi - gap
    confirmed_above = _persistence_confirmed(stress_z, thresh_hi, persist_days)
    below = (stress_z < thresh_lo).to_numpy()
    raw = np.where(confirmed_above.to_numpy(), 0.0, np.where(below, 1.0, np.nan))
    vote = pd.Series(raw, index=stress_z.index).ffill().fillna(1.0)
    return vote


class KellyRegimeV16StablecoinPersist(KellyRegimeV3):
    """v4's 3-anchor vote with a hard veto identical to R-54's
    ``KellyRegimeV15StablecoinVeto`` architecture (``frac=0`` while the
    latched stablecoin-supply-deceleration vote reads "stress"; v4's own
    unmodified 3-anchor average otherwise), except the vote's ENTRY into
    "stress" additionally requires ``persist_days`` consecutive days of
    confirmation above ``thresh_hi`` before it is allowed to fire.
    ``enabled=False`` recovers v4 exactly (identity check, verified in
    ``causality()``); ``persist_days=0`` recovers R-54's v15 behaviour
    (identity check, verified in ``identity_check()``).
    """

    name = "kelly_regime_v16_stablecoin_persist"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 thresh_hi: float = 1.0, gap: float = 0.75, persist_days: int = 3,
                 enabled: bool = True, **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.thresh_hi = thresh_hi
        self.gap = gap
        self.persist_days = persist_days
        self.enabled = enabled

    def _stress_series(self, df: pd.DataFrame) -> pd.Series:
        if "stablecoin_stress_z_visible" in df.columns:
            return df["stablecoin_stress_z_visible"]
        return compute_stablecoin_stress(df, DATA_DIR)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = _anchor_votes(close, self.horizons, self.band)
        anchor_sum = sum(votes.values())
        n_anchors = float(len(votes))
        anchor_frac = (anchor_sum / n_anchors).to_numpy()

        stress_z = self._stress_series(df)
        if self.enabled:
            stable_vote = _stable_vote_persist(
                stress_z, self.thresh_hi, self.gap, self.persist_days
            ).to_numpy()
        else:
            stable_vote = np.ones(len(df))  # identity check: never vetoes

        frac = np.where(stable_vote == 0.0, 0.0, anchor_frac)
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        # From here down: byte-for-byte v3's/v15's conditional vol-targeting
        # sizer, unchanged -- only the vote fraction feeding it differs.
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
        vstate = 0
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if vstate == 0:
                    vstate = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif vstate == 1 and x < self.high_out:
                    vstate = 0
                elif vstate == -1 and x > self.low_out:
                    vstate = 0
            scale = full[i] if vstate != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["v16_frac"] = frac
        df["v16_stable_vote"] = stable_vote
        df["v16_stress_z"] = stress_z.to_numpy()
        df["v16_anchor_sum"] = anchor_sum.to_numpy()
        return df


# ------------------------------------------------------------- the grid

# Three (thresh_hi, gap) combos: the PRIMARY (R-54's chosen point) and
# R-54's two worst/tightest configs (most false-positive stress-onsets,
# per R-54's own descriptive() count of 24 onset events at thresh=0.75/
# gap=0.0 vs 12 at the primary) -- exactly where a duration filter should
# help most if the mechanism works, per this round's instruction.
THRESH_GAP_COMBOS = (
    (1.0, 0.75, "primary"),
    (0.75, 0.0, "worst-tightest-nohys"),
    (0.75, 0.75, "worst-primary-gap"),
)
PERSIST_DAYS = (0, 1, 2, 3, 5, 7, 10, 14)
PRIMARY_KW = dict(thresh_hi=1.0, gap=0.75, persist_days=3)  # literature-motivated center (3-7d)


def _grid():
    out = []
    for thresh_hi, gap, tag in THRESH_GAP_COMBOS:
        for pd_ in PERSIST_DAYS:
            label = f"thresh={thresh_hi:.2f} gap={gap:.2f} persist={pd_:>2d}d ({tag})"
            out.append((label, dict(thresh_hi=thresh_hi, gap=gap, persist_days=pd_)))
    return out


def _config_key(kw: dict) -> str:
    return f"persist|thresh={kw['thresh_hi']:.3f}|gap={kw['gap']:.3f}|persist_days={kw['persist_days']}"


print(f"grid size: {len(THRESH_GAP_COMBOS)} thresh/gap combos x {len(PERSIST_DAYS)} persist_days values "
      f"= {len(THRESH_GAP_COMBOS) * len(PERSIST_DAYS)} configurations", file=sys.stderr)


# ------------------------------------------------------- step 0: identity check


def identity_check() -> None:
    """MANDATORY sanity check, run first: does persist_days=0 reproduce
    R-54's v15 ``_stable_vote`` bit-for-bit (or near-bit-for-bit)? Compared
    on the full inner-train+inner-validation window, for all three
    (thresh_hi, gap) combos in this file's grid."""
    lo, hi = TRAIN[0], VALID[1]
    frame = DF.loc[lo:hi]
    stress = frame["stablecoin_stress_z_visible"]

    print("identity check A: persist_days=0 vs v15's original _stable_vote (same thresh/gap)")
    for thresh_hi, gap, tag in THRESH_GAP_COMBOS:
        v15_vote = _stable_vote(stress, thresh_hi, gap).to_numpy()
        v16_vote = _stable_vote_persist(stress, thresh_hi, gap, persist_days=0).to_numpy()
        diff = np.abs(v15_vote - v16_vote)
        worst = float(np.nanmax(diff))
        n_diff = int(np.sum(diff > 1e-9))
        print(f"  thresh={thresh_hi:.2f} gap={gap:.2f} ({tag}): max|diff|={worst:.3e}  "
              f"bars differing={n_diff}/{len(diff)}  {'PASS' if worst < 1e-9 else 'FAIL/near-identical, see count'}")

    print("\nidentity check B: thresh_hi set so the veto never fires (thresh_hi=1e9) recovers v4 exactly")
    v4 = get_strategy(INCUMBENT)
    never_fires = KellyRegimeV16StablecoinPersist(thresh_hi=1e9, gap=0.0, persist_days=0)
    df_slice = DF.loc[lo:hi].copy()
    t_v4 = v4.prepare(df_slice.copy())["target"].to_numpy()
    t_never = never_fires.prepare(df_slice.copy())["target"].to_numpy()
    worst = float(np.nanmax(np.abs(t_v4 - t_never)))
    print(f"  max|diff|={worst:.3e}  {'PASS' if worst < 1e-9 else 'FAIL'}")

    print("\nidentity check C: enabled=False recovers v4 exactly")
    disabled = KellyRegimeV16StablecoinPersist(enabled=False)
    t_disabled = disabled.prepare(df_slice.copy())["target"].to_numpy()
    worst = float(np.nanmax(np.abs(t_v4 - t_disabled)))
    print(f"  max|diff|={worst:.3e}  {'PASS' if worst < 1e-9 else 'FAIL'}")


# ------------------------------------------------------- step 2b: descriptive


def descriptive() -> None:
    """Step 2b -- purely descriptive: how many stress-onset events does
    each persist_days value produce, at each of the three thresh/gap
    combos, over inner-train+inner-validation? Directly tests the
    mechanism's premise (persistence should shrink onset counts toward
    the count of GENUINE episodes, ~3-4, not the ~12-24 R-54 found)."""
    lo, hi = TRAIN[0], VALID[1]
    frame = DF.loc[lo:hi]
    stress = frame["stablecoin_stress_z_visible"]

    print(f"descriptive window (inner-train + inner-validation): {lo} -> {hi}")
    print(f"stablecoin_stress_z summary: mean={stress.mean():.2f} std={stress.std():.2f} "
          f"min={stress.min():.2f} max={stress.max():.2f}")

    for thresh_hi, gap, tag in THRESH_GAP_COMBOS:
        print(f"\n  thresh={thresh_hi:.2f} gap={gap:.2f} ({tag}):")
        for pd_ in PERSIST_DAYS:
            vote = _stable_vote_persist(stress, thresh_hi, gap, pd_)
            flips_to_stress = int(((vote == 0.0) & (vote.shift() == 1.0)).sum())
            flips_to_calm = int(((vote == 1.0) & (vote.shift() == 0.0)).sum())
            print(f"    persist_days={pd_:>2d}: {flips_to_stress:>3d} stress-onset event(s), "
                  f"{flips_to_calm:>3d} calm-return event(s)")


# ------------------------------------------------------- failure mode: lead time


def _daily_transitions(series: pd.Series, target_value: float, min_gap_days: int = 14) -> list:
    """Daily-resampled transition INTO ``target_value``, deduplicated so
    transitions within ``min_gap_days`` of a prior one count as one
    episode's onset. Uses ``shift(fill_value=False)`` (NOT
    ``.shift().fillna(False)``, the object-dtype bug R-53 found) --
    duplicated unchanged from ``kelly_regime_v15_stablecoin_veto.py``."""
    daily = series.resample("1D").last().ffill()
    is_target = (daily == target_value)
    prev = is_target.shift(fill_value=False)
    onsets = daily.index[is_target & (~prev)]
    kept = []
    for d in onsets:
        if not kept or (d - kept[-1]).days >= min_gap_days:
            kept.append(d)
    return kept


def _nearest(target_date, candidates, window_days=180):
    best, best_dist = None, None
    for c in candidates:
        dist = (c - target_date).days
        if abs(dist) <= window_days and (best_dist is None or abs(dist) < abs(best_dist)):
            best, best_dist = c, dist
    return best, best_dist


def leadtime() -> None:
    """THE (a) pre-registered falsification check: does the persistence
    filter preserve R-54's lead-time-vs-3-anchor-majority result, and
    where (if anywhere) does it collapse? Primary thresh/gap fixed
    (1.0/0.75), persist_days swept over the FULL grid. Descriptive, not a
    fit, not counted toward N_EVALUATED."""
    lo, hi = TRAIN[0], VALID[1]
    frame = DF.loc[lo:hi]
    close = frame["close"]

    votes = _anchor_votes(close, (20, 40, 80), 0.01)
    anchor_sum = sum(votes.values())
    majority_bear = (anchor_sum < 1.5).astype(float)
    majority_onsets = _daily_transitions(majority_bear, 1.0)
    print(f"3-anchor MAJORITY bear-onset episodes (reference, fixed across all persist_days): "
          f"{len(majority_onsets)}")

    stress = frame["stablecoin_stress_z_visible"]
    thresh_hi, gap = 1.0, 0.75
    print(f"\npersist_days sweep at primary thresh_hi={thresh_hi} gap={gap} "
          f"(THE (a) falsification check -- does lead time survive?):\n")

    for pd_ in PERSIST_DAYS:
        stable_vote = _stable_vote_persist(stress, thresh_hi, gap, pd_)
        stable_bear = 1.0 - stable_vote
        stable_onsets = _daily_transitions(stable_bear, 1.0)
        leads = []
        for d in stable_onsets:
            match, dist = _nearest(d, majority_onsets)
            if dist is not None:
                leads.append(-dist)
        if leads:
            n_lead = sum(1 for x in leads if x > 0)
            print(f"  persist_days={pd_:>2d}: {len(stable_onsets):>3d} onset(s), "
                  f"{len(leads)} matched, {n_lead}/{len(leads)} lead, "
                  f"median_lead={float(np.median(leads)):>+6.1f}d, "
                  f"leads={[round(x, 1) for x in leads]}")
        else:
            print(f"  persist_days={pd_:>2d}: {len(stable_onsets):>3d} onset(s), no matched pairs")

    print(f"\n(reference) R-54's own persist_days=0 result, this branch's negative control: "
          f"9/12 matched episodes lead, median +16.5 days")
    print("\nleadtime step: 0 configurations counted toward N_EVALUATED (descriptive, no free parameter fitted)")


# --------------------------------------------------------------- step 3: sweep


def sweep() -> None:
    """Step 3: every (thresh_hi, gap, persist_days) config on inner-train
    ONLY, spot primary."""
    print(f"\nINNER-TRAIN {TRAIN} / spot -- benchmarks:")
    for name in ("buy_and_hold", INCUMBENT):
        m, _ = measure(get_strategy(name), *TRAIN, market=SPOT)
        line(name, m, "spot")

    print(f"\nINNER-TRAIN {TRAIN} / spot -- candidate configurations ({len(_grid())} total):")
    for label, kw in _grid():
        m, _ = measure(KellyRegimeV16StablecoinPersist(**kw), *TRAIN, market=SPOT,
                        config_key=_config_key(kw))
        line(label, m, "spot")

    print(f"\nconfigurations evaluated (step 3, distinct thresh/gap/persist triples): {N_EVALUATED}")


# -------------------------------------------------------------- step 3: select


def select():
    """Every config on inner-validation ONLY, BOTH markets, vs v4 control."""
    rows = []
    print(f"\nINNER-VALIDATION {VALID} -- v4 control:")
    ctl = {}
    for mname, market in MARKETS:
        m, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        ctl[mname] = m
        line(f"{INCUMBENT} (control)", m, mname)

    print(f"\nINNER-VALIDATION {VALID} -- candidate configurations ({len(_grid())} total):")
    best_label, best_kw, best_score = None, None, -1e9
    for label, kw in _grid():
        cell = {}
        for mname, market in MARKETS:
            m, _ = measure(KellyRegimeV16StablecoinPersist(**kw), *VALID, market=market,
                            config_key=_config_key(kw))
            cell[mname] = m
            line(label, m, mname)
            rows.append({"label": label, **kw, "market": mname,
                         "final": m.final_balance, "sharpe": m.sharpe,
                         "max_dd": m.max_drawdown_pct, "trades": m.num_trades})
        # selection rule fixed in advance: min(train, valid) spot sharpe,
        # same rule R-54's own select() used
        m_train, _ = measure(KellyRegimeV16StablecoinPersist(**kw), *TRAIN, market=SPOT)
        score = min(m_train.sharpe, cell["spot"].sharpe)
        if score > best_score:
            best_label, best_kw, best_score = label, kw, score

    print(f"\nbest by min(train,valid) spot Sharpe: {best_label}  (score={best_score:.2f})")
    print(f"v4 control spot Sharpe: {ctl['spot'].sharpe:.2f}   "
          f"v4 control futures Sharpe: {ctl['futures 5x'].sharpe:.2f}")

    print("\nside-by-side: best candidate's TRAIN window vs VALIDATION window (overfitting-signature check):")
    for mname, market in MARKETS:
        m_train, _ = measure(KellyRegimeV16StablecoinPersist(**best_kw), *TRAIN, market=market)
        m_valid, _ = measure(KellyRegimeV16StablecoinPersist(**best_kw), *VALID, market=market)
        m_train_v4, _ = measure(get_strategy(INCUMBENT), *TRAIN, market=market)
        m_valid_v4, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        print(f"  {mname}:")
        print(f"    candidate  TRAIN final=${m_train.final_balance:>11,.0f} sharpe={m_train.sharpe:.2f} DD={m_train.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid.final_balance:>11,.0f} sharpe={m_valid.sharpe:.2f} DD={m_valid.max_drawdown_pct:.1f}%")
        print(f"    v4         TRAIN final=${m_train_v4.final_balance:>11,.0f} sharpe={m_train_v4.sharpe:.2f} DD={m_train_v4.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid_v4.final_balance:>11,.0f} sharpe={m_valid_v4.sharpe:.2f} DD={m_valid_v4.max_drawdown_pct:.1f}%")
        print(f"    candidate - v4 (valid): Delta sharpe={m_valid.sharpe - m_valid_v4.sharpe:+.3f}  "
              f"Delta DD={m_valid.max_drawdown_pct - m_valid_v4.max_drawdown_pct:+.1f}pp")

    print("\nparameter-neighbourhood plateau check (spot, inner-validation Sharpe, per thresh/gap combo, all persist_days):")
    grid_by_key = {}
    for label, kw in _grid():
        m, _ = measure(KellyRegimeV16StablecoinPersist(**kw), *VALID, market=SPOT)
        grid_by_key[(kw["thresh_hi"], kw["gap"], kw["persist_days"])] = m.sharpe
    for thresh_hi, gap, tag in THRESH_GAP_COMBOS:
        row = "  ".join(f"p={p}:{grid_by_key[(thresh_hi, gap, p)]:.2f}" for p in PERSIST_DAYS)
        print(f"  thresh={thresh_hi:.2f} gap={gap:.2f} ({tag})  {row}")

    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    return best_label, best_kw


# ------------------------------------------------------------------ artifact


def artifact(kw: dict | None = None) -> None:
    """Mandatory exposure-artifact check: R^2 of the candidate's target
    series against a mean-notional-matched flat rescale of v4's own
    target, on inner-validation, both markets. R^2 > 0.95 -> artifact."""
    kw = kw or PRIMARY_KW
    print(f"exposure-artifact check, candidate={kw}")
    v4 = get_strategy(INCUMBENT)
    for mname, market in MARKETS:
        cand = KellyRegimeV16StablecoinPersist(**kw)
        lo = int(DF.index.searchsorted(VALID[0]))
        hi = int(DF.index.searchsorted(VALID[1], side="right"))
        prefix = min(lo, max(cand.warmup, v4.warmup))
        frame = DF.iloc[lo - prefix:hi]

        v4_prepared = v4.prepare(frame.copy())
        cand_prepared = cand.prepare(frame.copy())
        v4_t = v4_prepared["target"].to_numpy()[prefix:]
        cand_t = cand_prepared["target"].to_numpy()[prefix:]

        mean_abs_v4 = float(np.mean(np.abs(v4_t)))
        mean_abs_cand = float(np.mean(np.abs(cand_t)))
        alpha = mean_abs_cand / mean_abs_v4 if mean_abs_v4 > 0 else 0.0
        rescaled = alpha * v4_t
        ss_res = float(np.sum((cand_t - rescaled) ** 2))
        ss_tot = float(np.sum((cand_t - cand_t.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        corr = float(np.corrcoef(cand_t, v4_t)[0, 1])
        verdict = "EXPOSURE-LEVEL ARTIFACT" if r2 > 0.95 else "genuinely different exposure shape"
        print(f"  {mname:10s} mean|v4|={mean_abs_v4:.3f} mean|cand|={mean_abs_cand:.3f} "
              f"alpha={alpha:.3f}  R^2(cand vs alpha*v4)={r2:.4f}  raw corr={corr:.4f}  {verdict}")


# ------------------------------------------------------------------ causality


def _make_tampered_stablecoin_dir(cut_day: pd.Timestamp, factor: float, tmp_root: Path) -> Path:
    """Copy the real stablecoin-supply CSV into a fresh dir, multiplying
    every row dated on/after ``cut_day`` by ``factor``. Used only for the
    causality probe below -- never writes into the real ``data/`` dir."""
    out_dir = tmp_root / f"stable_x{factor:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_DIR / "stablecoin_supply_daily.csv.gz")
    dates = pd.to_datetime(raw["timestamp"])
    mask = dates >= cut_day.tz_localize(None)
    raw.loc[mask, "usdt_SplyCur"] = raw.loc[mask, "usdt_SplyCur"] * factor
    raw.to_csv(out_dir / "stablecoin_supply_daily.csv.gz", index=False, compression="gzip")
    return out_dir


def causality(kw: dict | None = None) -> None:
    """Two-independent-pathway tamper probe: price OHLCV AND the
    stablecoin-supply input tampered independently after a cut. Every
    decision at or before the cut must be unchanged. Restricted to
    strictly pre-2023 bars. Structure duplicated from
    ``kelly_regime_v15_stablecoin_veto.py``'s ``causality()``."""
    kw = kw or PRIMARY_KW

    pre2023 = DF[DF.index < OOS_START]
    df = pre2023.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]
    cut_day = df.index[cut].normalize()

    def strategy_for(data_dir: Path | None):
        s = KellyRegimeV16StablecoinPersist(**kw)
        if data_dir is not None:
            def patched(frame, _dd=data_dir):
                if "stablecoin_stress_z_visible" in frame.columns:
                    # price-only tamper path: reuse the precomputed (real-stablecoin) column
                    return frame["stablecoin_stress_z_visible"]
                return compute_stablecoin_stress(frame, _dd)
            s._stress_series = patched  # noqa: SLF001 (deliberate, test-only monkeypatch)
        return s

    def run_probe(name, tamper_ohlcv_fn=None, stable_dir_up=None, stable_dir_down=None):
        up, down = df.copy(), df.copy()
        if tamper_ohlcv_fn is not None:
            tamper_ohlcv_fn(up, down)
        if stable_dir_up is not None:
            up = up.drop(columns=["stablecoin_stress_z_visible"])
            down = down.drop(columns=["stablecoin_stress_z_visible"])

        def decisions(frame, data_dir):
            s = strategy_for(data_dir)
            prepared = s.prepare(frame.copy())
            broker = _fresh_broker()
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        a = decisions(up, stable_dir_up)
        b = decisions(down, stable_dir_down)
        bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
        print(f"[{name}] tampered from bar {cut:,} of {len(df):,} "
              f"(calendar day {cut_day.date()}); checked bars {bars}")
        print("  FAIL - reads the future at bars " + str(bad) if bad
              else "  PASS - every decision at or before the cut is unchanged")

        pa = strategy_for(stable_dir_up).prepare(up.copy())
        pb = strategy_for(stable_dir_down).prepare(down.copy())
        for col in ("target", "v16_frac", "v16_stable_vote", "v16_anchor_sum"):
            diff = np.abs(pa[col].to_numpy()[:cut].astype(float)
                          - pb[col].to_numpy()[:cut].astype(float))
            worst = float(np.nanmax(diff))
            print(f"    column {col:16s} max |difference| before the cut = {worst:.3e}"
                  f"  {'PASS' if worst < 1e-9 else 'FAIL'}")

    def tamper_ohlcv(up, down):
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    tmp_root = Path(tempfile.mkdtemp(prefix="v16_stablecoin_persist_causality_"))
    try:
        run_probe("PRICE tamper (standard)", tamper_ohlcv_fn=tamper_ohlcv)

        stable_dir_up = _make_tampered_stablecoin_dir(cut_day, 50.0, tmp_root)
        stable_dir_down = _make_tampered_stablecoin_dir(cut_day, 1.0 / 50.0, tmp_root)
        run_probe("STABLECOIN tamper (the new supply-data pathway)",
                   stable_dir_up=stable_dir_up, stable_dir_down=stable_dir_down)

        run_probe("both at once", tamper_ohlcv_fn=tamper_ohlcv,
                   stable_dir_up=stable_dir_up, stable_dir_down=stable_dir_down)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # identity checks: enabled=False, and thresh_hi=1e9 (never fires), both recover v4 exactly
    v4 = get_strategy(INCUMBENT)
    frame = df.iloc[-20_000:].copy()
    t_v4 = v4.prepare(frame.copy())["target"].to_numpy()

    ident_disabled = KellyRegimeV16StablecoinPersist(**{**PRIMARY_KW, "enabled": False})
    t_ident_disabled = ident_disabled.prepare(frame.copy())["target"].to_numpy()
    worst_a = float(np.nanmax(np.abs(t_v4 - t_ident_disabled)))
    print(f"\nidentity check (enabled=False recovers v4 exactly): "
          f"max|diff|={worst_a:.3e}  {'PASS' if worst_a < 1e-9 else 'FAIL'}")

    ident_never = KellyRegimeV16StablecoinPersist(thresh_hi=1e9, gap=0.0, persist_days=0)
    t_ident_never = ident_never.prepare(frame.copy())["target"].to_numpy()
    worst_b = float(np.nanmax(np.abs(t_v4 - t_ident_never)))
    print(f"identity check (thresh_hi=1e9, veto never fires, recovers v4 exactly): "
          f"max|diff|={worst_b:.3e}  {'PASS' if worst_b < 1e-9 else 'FAIL'}")


def _fresh_broker():
    from tradebot.broker import PaperBroker
    return PaperBroker(market=FUTURES, start_balance=10_000.0)


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Falsification test (b), pre-registered rule below, fixed before running.

    ETH-USD Bitfinex spot (2016-03 -> 2019-12-31) against USDT-supply
    coverage (2017-01-01 ->). Same rule as R-54's ``eth()``: if the
    candidate is not at least comparable to v4 on ETH, or is visibly
    worse on ETH than on the BTC control through the identical code, this
    direction fails. An ETH-only failure must be reported, not hidden.
    """
    eth_spot = load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")
    eth_stress = compute_stablecoin_stress(eth_spot, DATA_DIR)
    eth_df = eth_spot.copy()
    eth_df["stablecoin_stress_z_visible"] = eth_stress

    overlap = eth_df["stablecoin_stress_z_visible"].dropna()
    print(f"ETH spot file: {len(eth_spot):,} bars  {eth_spot.index[0]:%Y-%m-%d} -> "
          f"{eth_spot.index[-1]:%Y-%m-%d}")
    print(f"stablecoin stress coverage overlapping ETH spot: {len(overlap):,} bars "
          f"visible from {overlap.index[0]:%Y-%m-%d} to {overlap.index[-1]:%Y-%m-%d}")

    frames = {"BTC (control)": DF[DF.index < OOS_START], "ETH (test)": eth_df}
    results = {}
    for asset, frame in frames.items():
        print(f"\n{asset}  {len(frame):,} bars  {frame.index[0]:%Y-%m-%d} -> {frame.index[-1]:%Y-%m-%d}")
        results[asset] = {}
        for mname, market in MARKETS:
            m_v4, _ = measure(get_strategy(INCUMBENT), None, None, df=frame, market=market)
            line(f"{INCUMBENT} (control)", m_v4, mname)
            asset_results = {"v4": m_v4}
            for label, kw in _grid():
                cand = KellyRegimeV16StablecoinPersist(**kw)
                m_c, _ = measure(cand, None, None, df=frame, market=market)
                line(f"v16[{label}]", m_c, mname)
                asset_results[label] = m_c
            results[asset][mname] = asset_results

    print("\nfalsification verdict per config (candidate/v4 final-balance ratio, BTC control vs ETH test):")
    any_fail = False
    for label, kw in _grid():
        for mname, market in MARKETS:
            btc_r = results["BTC (control)"][mname][label].final_balance / results["BTC (control)"][mname]["v4"].final_balance
            eth_r = results["ETH (test)"][mname][label].final_balance / results["ETH (test)"][mname]["v4"].final_balance
            worse_on_eth = eth_r < btc_r - 0.02
            eth_beats_v4 = eth_r >= 1.0
            flag = "FAIL" if (worse_on_eth and not eth_beats_v4) else ("caution" if worse_on_eth else "ok")
            any_fail = any_fail or flag == "FAIL"
            print(f"  {label:38s} {mname:10s} BTC ratio={btc_r:.3f}x  ETH ratio={eth_r:.3f}x  [{flag}]")
    print(f"\nprimary candidate ({PRIMARY_KW}) checked above; "
          f"overall falsification verdict: {'FAIL' if any_fail else 'no outright FAIL by the pre-registered rule'}")


# --------------------------------------------------------------------- driver


def all_checks() -> None:
    print("=" * 78)
    print("STEP 0 -- identity checks (persist_days=0 vs v15, thresh=1e9 vs v4, enabled=False vs v4)")
    print("=" * 78)
    identity_check()
    print("\n" + "=" * 78)
    print("STEP 2b -- descriptive: stress-onset frequency vs persist_days")
    print("=" * 78)
    descriptive()
    print("\n" + "=" * 78)
    print("PRE-REGISTERED FALSIFICATION (a) -- lead-time vs persist_days")
    print("=" * 78)
    leadtime()
    print("\n" + "=" * 78)
    print("STEP 3 -- sweep (inner-train)")
    print("=" * 78)
    sweep()
    print("\n" + "=" * 78)
    print("STEP 3 -- select (inner-validation, both markets)")
    print("=" * 78)
    select()
    print("\n" + "=" * 78)
    print("EXPOSURE-ARTIFACT CHECK")
    print("=" * 78)
    artifact()
    print("\n" + "=" * 78)
    print("CAUSALITY / NO-LOOKAHEAD PROBE")
    print("=" * 78)
    causality()
    print("\n" + "=" * 78)
    print("PRE-REGISTERED FALSIFICATION (b) -- ETH TEST")
    print("=" * 78)
    eth()
    print(f"\ntotal distinct configurations evaluated (N_EVALUATED): {N_EVALUATED}")


if __name__ == "__main__":
    cmds = {"identity": identity_check, "descriptive": descriptive, "leadtime": leadtime,
            "sweep": sweep, "select": select, "artifact": artifact, "causality": causality,
            "eth": eth, "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/kelly_regime_v16_stablecoin_persist.py [{'|'.join(cmds)}]")
