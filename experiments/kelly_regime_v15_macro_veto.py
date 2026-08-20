#!/usr/bin/env python
"""CONSERVATIVE branch, parallel round R-54: pre-registration and full
falsification battery for B-21 -- the hard, unweighted macro-veto override
of kelly_regime_v4's regime gate that surfaced only as R-53 novel's
ablation comparison arm and was never itself pre-registered, lead-time
checked, ETH-falsified, causality-probed as its own subject, or plateau
checked.

Idea, one sentence
------------------
While VIX/DXY-derived macro stress (`stress_z`, imported unchanged from
the shared, operator-authored `experiments/_macro_signal.py`) is latched
"elevated" (with hysteresis), force `frac = 0` -- a full stand-down --
overriding kelly_regime_v4's own three-anchor vote completely; otherwise
`frac` is v4's own unmodified 3-anchor average, with no weighting, no
averaging, no continuous haircut.

Constraint(s) attacked
-----------------------
**INFO** -- VIX/DXY are the second genuinely new, price-independent data
channel this project has used (after on-chain metrics, B-07/R-44), and
the specific exploitation tested here is architecturally different from
both R-53 attempts at the same channel (see "not a duplicate of" below).
**SIZE** -- a hard override that forces `frac=0` is, by definition, a
decision about *how much* to hold (the project's one repeatedly-working
axis per the standing diagnosis), layered on top of, not replacing, v4's
existing SIZE machinery (its conditional-vol-target scale is untouched).

Not a duplicate of, cited precisely
-------------------------------------
- **L-04/L-01** (`kelly_regime_v4`, the incumbent): this candidate's
  `frac` degenerates to v4's own unmodified 3-anchor average on every bar
  where macro data is absent or `stress_z` never crosses `thresh_hi` --
  verified as an explicit identity check in `causality()` (a frame with
  `stress_z` forced all-NaN must produce a byte-identical `target` series
  to v4). It is an override layered on the vote, not a replacement of any
  v4 machinery.
- **R-53's two rows** (this same ledger, 08-20): the conservative row
  (`kelly_regime_v14_macro_brake.py`) tested a *continuous, never-increase
  multiplicative haircut* on the SIZE scale fed by the identical
  `stress_z` signal, and it collapsed into R-34's exposure-artifact
  failure mode (R^2=0.974-0.999) -- this file's mechanism is not a
  continuous multiplier at all; it is a binary override applied to the
  VOTE, before the SIZE formula runs, and the mandatory exposure-artifact
  check below is run precisely because that is not obviously safe just
  because the architecture differs. The novel row
  (`kelly_regime_v14_macro_lead.py`) tested a *precision-weighted 4th
  vote averaged with the three anchors*
  (`frac=(anchor_sum+macro_weight*macro_vote)/(3+macro_weight)`) and it
  lost to its own hard-override ablation arm in 10/12 matched cells by
  0.25-0.48 Sharpe -- THIS file promotes that ablation arm itself
  (`KellyRegimeV14MacroOverride` in that file, not imported here, not
  edited -- reimplemented standalone per this file's own self-contained
  requirement) to be the pre-registered primary subject of its own
  falsification battery, rather than a side comparison arm inside a round
  pre-registered for a different mechanism.
- **R-44's on-chain rows** (B-07, hash-ribbon confirmation vote,
  `kelly_regime_v10_hashribbon_vote.py`): same general architecture family
  (a latched auxiliary vote combined with v4's own anchor votes) and the
  same INFO constraint, but (1) a different, price-independent-but-
  BTC-network-specific signal (on-chain hashrate) vs. this file's
  market-wide macro signal, and (2) the opposite sign discipline --
  R-44's vote only ever pushes exposure UP (capitulation-recovery =
  bullish); this file's veto only ever pushes exposure DOWN to exactly
  zero (stress = full stand-down), matching the spillover literature's
  risk-off-specific claim, never manufacturing bullish exposure the
  anchors would not already grant.

Pre-registered falsification test (named now, before any code below runs)
----------------------------------------------------------------------------
**Primary test -- resolves B-21's own named, unresolved tension.** R-53's
own lead-time finding for the AVERAGED version of this same signal found
the macro vote lags the 3-anchor majority on net (4/12 matched episodes
lead, median offset -5.5 days) -- a mechanism whose entire value
proposition is faster gate-flipping should not obviously win just because
its combination rule got blunter. This file re-runs the identical
`leadtime()` methodology (flip TIMESTAMPS, not aggregate Sharpe) against
THIS file's own veto latch (same signal, same hysteresis discipline,
different combination rule), using the corrected
`.shift(fill_value=False)` (see the `_daily_transitions` docstring below
for the exact bug R-53 found and how this file avoids re-introducing it).

**Stated failure outcome, named in advance:** if the veto's own bear-onset
flips do NOT lead the 3-anchor majority on net (median lead_days <= 0,
replicating R-53's finding for the averaged version), then the
"faster-flip" rationale this mechanism was built on does not hold for the
hard-override form either, and the "blunter combination rule can win
despite lagging" hypothesis from B-21's backlog note is REJECTED unless
the candidate's inner-validation edge (a) clears the +/-0.2 Sharpe noise
floor decisively on BOTH markets AND (b) survives every other mandatory
gate below (ETH, BTC control, causality both pathways, exposure-artifact
R^2<0.95, gap-neighbourhood plateau) -- in which case the honest reading
is that the edge comes from some OTHER property of the mechanism (e.g.
avoiding whipsaw re-entries the averaged vote's precision weighting
smoothed away) and that must be stated explicitly, not credited to
"faster reaction," which the lead-time data would have just disproven.

**Secondary, standard-menu test (ROUTINE.md step 2):** does it survive on
ETH (pre-2020, same convention as every prior round)? Stated failure
outcome: candidate underperforms v4 on ETH spot while matching or beating
v4 on the identical-pipeline BTC control (an asset-specific signature a
market-wide signal should not produce) -> FAIL, per the identical rule
R-53's novel branch used.

Neither test above is optional and neither is run in isolation: a PASS on
one and a FAIL on the other is reported as a FAIL overall, per ROUTINE.md's
"all must hold" promotion bar.

Grid, fixed before any run
----------------------------
`thresh_hi = 1.0` (one trailing std of stress; a-priori, never swept --
matches R-53 and R-54's own `KellyRegimeV14MacroOverride` convention).
`gap` swept over the pre-declared grid `{0.0, 0.5, 0.75, 1.0, 1.25}`
(task brief's own grid; a superset of R-53's `{0.0, 0.75, 1.25}` with two
new points added specifically to test whether the region is a genuine
plateau rather than R-53's 3-point read of it). `gap=0.0` is the explicit
no-hysteresis negative control (a single memoryless threshold). Note for
the record: R-53's ledger row (B-21) cites the `gap=0.0` cell's numbers
as the headline ("spot Sharpe 0.34 vs 0.14 ... futures Sharpe 0.39 vs
0.25") -- i.e. the cited "unvetted lead" number is itself the negative
control. This file does NOT treat `gap=0.0` as the pre-registered primary
candidate for that reason; the primary candidate is fixed at `gap=0.75`
(a genuine hysteresis band, matching R-53's own `KellyRegimeV14MacroLead`
default), and the full 5-point grid is reported so the `gap=0.0` result
is visible in context rather than singled out.

**5 distinct configurations** (the `gap` grid; `thresh_hi` fixed and not
counted as a searched axis) -- see `N_EVALUATED` / the counting
convention in `measure()`.

Iteration boundary
--------------------
Inner-train (2017-01-01 -> 2020-12-31) for iteration, inner-validation
(2021-01-01 -> 2022-12-31) for selection, per ROUTINE.md step 3. No bar
dated 2023-01-01 or later is read anywhere in this file -- grepped and
confirmed (see the report's own grep transcript).

Usage
-----
    python experiments/kelly_regime_v15_macro_veto.py descriptive  # step 2b, vote-frequency context
    python experiments/kelly_regime_v15_macro_veto.py leadtime     # primary pre-registered falsification test
    python experiments/kelly_regime_v15_macro_veto.py sweep        # step 3 (inner-train)
    python experiments/kelly_regime_v15_macro_veto.py select       # step 3 (inner-validation) + plateau
    python experiments/kelly_regime_v15_macro_veto.py artifact     # exposure-artifact R^2 check
    python experiments/kelly_regime_v15_macro_veto.py causality    # lookahead probe (price + macro pathway)
    python experiments/kelly_regime_v15_macro_veto.py eth          # secondary falsification test (+ BTC control)
    python experiments/kelly_regime_v15_macro_veto.py all          # everything, in order
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

from experiments._macro_signal import compute_macro_stress  # noqa: E402

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


def build_macro_dataframe() -> tuple[pd.DataFrame, str]:
    """Canonical spot OHLCV with a causal ``stress_z_visible`` column merged
    on. Identical construction to R-53's ``build_macro_dataframe`` -- same
    shared, unedited signal module, computed once."""
    spot, label = load_dataset(DATA_DIR, "spot")
    stress = compute_macro_stress(spot, DATA_DIR)
    out = spot.copy()
    out["stress_z_visible"] = stress
    return out, label


DF, LABEL = build_macro_dataframe()
print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
      f"(data: {LABEL}); macro-stress coverage {DF['stress_z_visible'].notna().sum():,} bars "
      f"from {DF['stress_z_visible'].dropna().index[0]:%Y-%m-%d}", file=sys.stderr)


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0,
            config_key: str | None = None):
    """One backtest -> Metrics. ``config_key`` counts a DISTINCT
    configuration exactly once across the whole session, however many
    market/period cells it is subsequently re-scored on (v4 control and
    diagnostic re-reads pass config_key=None and are never counted) --
    identical convention to R-53's ``kelly_regime_v14_macro_lead.py``."""
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
    """Exactly kelly_regime_v4's own per-anchor latched vote."""
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


def _macro_veto_vote(stress_z: pd.Series, thresh_hi: float, gap: float) -> pd.Series:
    """Latched macro-stress veto state, 0/1, hysteresis on ``stress_z``.
    Identical construction and identical hysteresis discipline to R-53's
    ``_macro_vote`` (reimplemented standalone here, not imported, per this
    file's self-contained requirement -- the sibling R-53 file is never
    edited or depended on).

    ``veto -> 0`` ("stress", the only direction this ever fires) requires
    ``stress_z`` to cross ABOVE ``thresh_hi``. ``veto -> 1`` ("calm", the
    default and the re-arming condition) requires it to fall back BELOW
    ``thresh_lo = thresh_hi - gap``. ``gap=0.0`` collapses ``thresh_lo``
    onto ``thresh_hi``: a single memoryless threshold, kept in the swept
    grid as the explicit negative control. Defaults to 1.0 ("calm")
    wherever ``stress_z`` is NaN (before the macro series' own ~60-day
    warmup, or if the column is entirely absent) -- absence of macro
    information means no veto, not an assumed worst case, so a candidate
    with no macro data at all recovers v4's anchor-only vote exactly."""
    thresh_lo = thresh_hi - gap
    raw = np.where(stress_z > thresh_hi, 0.0,
                    np.where(stress_z < thresh_lo, 1.0, np.nan))
    vote = pd.Series(raw, index=stress_z.index).ffill().fillna(1.0)
    return vote


class KellyRegimeV15MacroVeto(KellyRegimeV3):
    """B-21's mechanism: a hard, unweighted macro-veto override of v4's
    regime gate.

    While the latched macro veto reads 0 ("stress"), force ``frac = 0`` --
    a full stand-down, overriding the anchors completely. Otherwise
    ``frac`` is v4's own unmodified 3-anchor average -- no weighting, no
    averaging, no scale multiplier. Everything else -- v3/v4's conditional
    vol-target scale, the 2x cap, the 10% deadband -- is copied verbatim,
    unchanged, exactly as in R-53's ``KellyRegimeV14MacroOverride`` (not
    imported; reimplemented here so this file has no runtime dependency on
    a sibling round's file).
    """

    name = "kelly_regime_v15_macro_veto"
    warmup = 80 * BARS_PER_DAY + 10  # same as v4; macro's own ~60d z-score warmup is shorter

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 thresh_hi: float = 1.0, gap: float = 0.75, **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.thresh_hi = thresh_hi
        self.gap = gap

    def _stress_series(self, df: pd.DataFrame) -> pd.Series:
        if "stress_z_visible" in df.columns:
            return df["stress_z_visible"]
        return compute_macro_stress(df, DATA_DIR)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = _anchor_votes(close, self.horizons, self.band)
        anchor_sum = sum(votes.values())
        n_anchors = float(len(votes))
        anchor_frac = (anchor_sum / n_anchors).to_numpy()

        stress_z = self._stress_series(df)
        veto = _macro_veto_vote(stress_z, self.thresh_hi, self.gap).to_numpy()

        frac = np.where(veto == 0.0, 0.0, anchor_frac)
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
        df["v15_frac"] = frac
        df["v15_veto"] = veto
        df["v15_anchor_frac"] = anchor_frac
        df["v15_stress_z"] = stress_z.to_numpy()
        return df


# ------------------------------------------------------------- the grid

# gap=0.0 is the explicit negative control: no hysteresis dead zone at all.
GAPS = (0.0, 0.5, 0.75, 1.0, 1.25)
THRESH_HI = 1.0  # fixed a-priori (one trailing std of stress), never swept
PRIMARY_KW = dict(thresh_hi=THRESH_HI, gap=0.75)


def _grid():
    out = []
    for gap in GAPS:
        label = f"gap={gap:.2f}" + ("(naive-nohys)" if gap == 0.0 else "")
        out.append((label, dict(thresh_hi=THRESH_HI, gap=gap)))
    return out


def _config_key(kw: dict) -> str:
    return f"veto|thresh={kw['thresh_hi']:.3f}|gap={kw['gap']:.3f}"


# ------------------------------------------------------- step 2b: descriptive


def descriptive() -> None:
    """Step 2b -- purely descriptive, no free parameter fitted: how often
    does the latched veto actually fire over inner-train+validation, vs
    each of v4's own three price anchors flipping over the identical
    window."""
    lo, hi = TRAIN[0], VALID[1]
    frame = DF.loc[lo:hi]
    close = frame["close"]

    print(f"descriptive window (inner-train + inner-validation): {lo} -> {hi}")
    print("\nprice-anchor vote transition counts over the SAME window:")
    for days in (20, 40, 80):
        anchor = DF["close"].rolling(int(days * BARS_PER_DAY)).mean().loc[lo:hi]
        v = pd.Series(np.where(close > anchor * 1.01, 1.0,
                                np.where(close < anchor * 0.99, 0.0, np.nan)),
                      index=close.index).ffill().fillna(0.0)
        print(f"  {days:>3d}d price anchor: {int(v.ne(v.shift()).sum()):>4d} vote flips")

    stress = frame["stress_z_visible"]
    print(f"\nstress_z coverage in window: {stress.notna().sum():,} / {len(stress):,} bars")
    print(f"stress_z summary: mean={stress.mean():.2f} std={stress.std():.2f} "
          f"min={stress.min():.2f} max={stress.max():.2f}")

    for gap in GAPS:
        veto = _macro_veto_vote(stress, THRESH_HI, gap)
        flips_to_stress = int(((veto == 0.0) & (veto.shift() == 1.0)).sum())
        flips_to_calm = int(((veto == 1.0) & (veto.shift() == 0.0)).sum())
        # fraction of bars the veto is actively engaged (frac forced to 0)
        engaged_pct = 100.0 * float((veto == 0.0).mean())
        label = f"gap={gap:.2f}" + (" (naive, no hysteresis)" if gap == 0.0 else "")
        print(f"  veto {label}: {flips_to_stress} stress-onset event(s), "
              f"{flips_to_calm} calm-return event(s), engaged {engaged_pct:.1f}% of bars")


# ------------------------------------------------------- primary falsification test: lead time


def _daily_transitions(series: pd.Series, target_value: float, min_gap_days: int = 14) -> list:
    """Daily-resampled transition INTO ``target_value``, deduplicated so
    transitions within ``min_gap_days`` of a prior one count as one
    episode's onset, not repeated bar-level noise.

    NOTE (R-53's own documented gotcha, deliberately avoided here):
    ``is_target.shift().fillna(False)`` upcasts to object dtype (the
    leading NaN forces it), so ``~`` on the filled Python-bool objects
    does BITWISE invert (``~True == -2``, truthy) rather than logical
    negation -- every day after the first "target" day would wrongly
    count as a fresh onset, silently inflating onset counts (R-53 caught
    this by hand: 41 "onsets" vs. 12 genuine transitions). This file uses
    ``shift(fill_value=False)`` instead, which keeps the Series
    boolean-dtype so ``~`` is logical negation as intended."""
    daily = series.resample("1D").last().ffill()
    is_target = (daily == target_value)
    prev = is_target.shift(fill_value=False)
    onsets = daily.index[is_target & (~prev)]
    kept = []
    for d in onsets:
        if not kept or (d - kept[-1]).days >= min_gap_days:
            kept.append(d)
    return kept


def leadtime() -> None:
    """PRIMARY PRE-REGISTERED FALSIFICATION TEST: does the veto's own
    bear-onset (stress engaging, frac forced to 0) actually come BEFORE
    kelly_regime_v4's own price-anchor gate flips bearish, in the handful
    of stress episodes available (2018, 2020-03, 2022)? Compares flip
    TIMESTAMPS directly, not aggregate Sharpe -- same methodology as
    R-53's ``leadtime()``, re-run here as this file's own subject rather
    than assumed to transfer. Uses the primary config
    (thresh_hi=1.0, gap=0.75); descriptive check, not a fit, not counted
    toward N_EVALUATED."""
    lo, hi = TRAIN[0], VALID[1]
    frame = DF.loc[lo:hi]
    close = frame["close"]

    votes = _anchor_votes(close, (20, 40, 80), 0.01)
    anchor20_bear = 1.0 - votes[20]           # fastest single anchor, 1=bearish
    anchor_sum = sum(votes.values())
    majority_bear = (anchor_sum < 1.5).astype(float)  # 1 when >=2 of 3 anchors bearish

    stress = frame["stress_z_visible"]
    veto = _macro_veto_vote(stress, PRIMARY_KW["thresh_hi"], PRIMARY_KW["gap"])
    veto_bear = 1.0 - veto

    veto_onsets = _daily_transitions(veto_bear, 1.0)
    anchor20_onsets = _daily_transitions(anchor20_bear, 1.0)
    majority_onsets = _daily_transitions(majority_bear, 1.0)

    print(f"veto bear-onset episodes (gap={PRIMARY_KW['gap']}, thresh_hi={PRIMARY_KW['thresh_hi']}): "
          f"{len(veto_onsets)}")
    print(f"20d-anchor bear-onset episodes: {len(anchor20_onsets)}")
    print(f"3-anchor MAJORITY bear-onset episodes: {len(majority_onsets)}")

    def nearest(target_date, candidates, window_days=180):
        best, best_dist = None, None
        for c in candidates:
            dist = (c - target_date).days
            if abs(dist) <= window_days and (best_dist is None or abs(dist) < abs(best_dist)):
                best, best_dist = c, dist
        return best, best_dist

    print("\nveto onset vs nearest 20d-anchor onset (positive lead_days = veto flips FIRST):")
    leads_vs_20d = []
    for d in veto_onsets:
        match, dist = nearest(d, anchor20_onsets)
        lead = -dist if dist is not None else None
        if lead is not None:
            leads_vs_20d.append(lead)
        print(f"  veto onset {d.date()}  ->  nearest 20d-anchor onset "
              f"{match.date() if match else 'none within 180d'}  "
              f"lead_days={lead if lead is not None else 'n/a'}")

    print("\nveto onset vs nearest 3-anchor MAJORITY onset (positive lead_days = veto flips FIRST):")
    leads_vs_majority = []
    for d in veto_onsets:
        match, dist = nearest(d, majority_onsets)
        lead = -dist if dist is not None else None
        if lead is not None:
            leads_vs_majority.append(lead)
        print(f"  veto onset {d.date()}  ->  nearest majority-anchor onset "
              f"{match.date() if match else 'none within 180d'}  "
              f"lead_days={lead if lead is not None else 'n/a'}")

    def summarize(name, leads):
        if not leads:
            print(f"\n{name}: no matched pairs within window -- cannot assess lead/lag")
            return None
        n_lead = sum(1 for x in leads if x > 0)
        median = float(np.median(leads))
        print(f"\n{name}: {len(leads)} matched episode(s), "
              f"{n_lead}/{len(leads)} veto-leads, "
              f"median lead_days={median:.1f}, "
              f"individual leads={[round(x, 1) for x in leads]}")
        return median

    summarize("SUMMARY vs 20d anchor (fastest single anchor)", leads_vs_20d)
    median_vs_majority = summarize(
        "SUMMARY vs 3-anchor majority (the actual gate-flip proxy -- the primary test's metric)",
        leads_vs_majority)

    print("\nPRE-REGISTERED VERDICT on the primary falsification test:")
    if median_vs_majority is None:
        print("  INCONCLUSIVE -- no matched episodes; cannot resolve the stated failure outcome directly.")
    elif median_vs_majority <= 0:
        print(f"  Median lead_days={median_vs_majority:.1f} <= 0: the veto does NOT lead the 3-anchor "
              "majority on net -- replicates R-53's finding for the averaged version. Per the "
              "pre-registered failure outcome, the 'faster-flip' rationale does not hold; the "
              "'blunter combination rule wins despite lagging' hypothesis is REJECTED unless "
              "every other mandatory gate below passes decisively AND the source of any inner-"
              "validation edge is attributed to something other than faster reaction.")
    else:
        print(f"  Median lead_days={median_vs_majority:.1f} > 0: the veto DOES lead the 3-anchor "
              "majority on net -- the averaged version's lag finding does NOT transfer to the "
              "hard-override form. This is a genuinely different outcome from R-53's novel branch.")

    print("\nleadtime step: 0 configurations counted toward N_EVALUATED (descriptive, no free parameter fitted)")


# --------------------------------------------------------------- step 3: sweep


def sweep() -> None:
    """Step 3: every gap config on inner-train ONLY, spot primary."""
    print(f"\nINNER-TRAIN {TRAIN} / spot -- benchmarks:")
    for name in ("buy_and_hold", INCUMBENT):
        m, _ = measure(get_strategy(name), *TRAIN, market=SPOT)
        line(name, m, "spot")

    print(f"\nINNER-TRAIN {TRAIN} / spot -- candidate configurations:")
    for label, kw in _grid():
        m, _ = measure(KellyRegimeV15MacroVeto(**kw), *TRAIN, market=SPOT,
                        config_key=_config_key(kw))
        line(label, m, "spot")

    print(f"\nconfigurations evaluated (step 3, distinct gap values): {N_EVALUATED}")


# -------------------------------------------------------------- step 3: select


def select():
    """Every config on inner-validation ONLY, BOTH markets, vs v4 control,
    plus the parameter-neighbourhood plateau check."""
    rows = []
    print(f"\nINNER-VALIDATION {VALID} -- v4 control:")
    ctl = {}
    for mname, market in MARKETS:
        m, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        ctl[mname] = m
        line(f"{INCUMBENT} (control)", m, mname)

    print(f"\nINNER-VALIDATION {VALID} -- candidate configurations:")
    best_label, best_kw, best_score = None, None, -1e9
    for label, kw in _grid():
        cell = {}
        for mname, market in MARKETS:
            m, _ = measure(KellyRegimeV15MacroVeto(**kw), *VALID, market=market,
                            config_key=_config_key(kw))
            cell[mname] = m
            line(label, m, mname)
            rows.append({"label": label, **kw, "market": mname,
                         "final": m.final_balance, "sharpe": m.sharpe,
                         "max_dd": m.max_drawdown_pct, "trades": m.num_trades})
        # selection rule fixed in advance: min(train, valid) spot sharpe,
        # guards against a train-loses/validation-wins overfit signature
        m_train, _ = measure(KellyRegimeV15MacroVeto(**kw), *TRAIN, market=SPOT)
        score = min(m_train.sharpe, cell["spot"].sharpe)
        if score > best_score:
            best_label, best_kw, best_score = label, kw, score

    print(f"\nbest by min(train,valid) spot Sharpe: {best_label}  (score={best_score:.2f})")
    print(f"v4 control spot Sharpe: {ctl['spot'].sharpe:.2f}   "
          f"v4 control futures Sharpe: {ctl['futures 5x'].sharpe:.2f}")

    print("\nside-by-side: primary candidate's TRAIN window vs VALIDATION window (overfitting-signature check):")
    for mname, market in MARKETS:
        m_train, _ = measure(KellyRegimeV15MacroVeto(**PRIMARY_KW), *TRAIN, market=market)
        m_valid, _ = measure(KellyRegimeV15MacroVeto(**PRIMARY_KW), *VALID, market=market)
        m_train_v4, _ = measure(get_strategy(INCUMBENT), *TRAIN, market=market)
        m_valid_v4, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        print(f"  {mname}:")
        print(f"    candidate  TRAIN final=${m_train.final_balance:>11,.0f} sharpe={m_train.sharpe:.2f} DD={m_train.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid.final_balance:>11,.0f} sharpe={m_valid.sharpe:.2f} DD={m_valid.max_drawdown_pct:.1f}%")
        print(f"    v4         TRAIN final=${m_train_v4.final_balance:>11,.0f} sharpe={m_train_v4.sharpe:.2f} DD={m_train_v4.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid_v4.final_balance:>11,.0f} sharpe={m_valid_v4.sharpe:.2f} DD={m_valid_v4.max_drawdown_pct:.1f}%")
        print(f"    candidate - v4 (valid): Delta sharpe={m_valid.sharpe - m_valid_v4.sharpe:+.3f}  "
              f"Delta DD={m_valid.max_drawdown_pct - m_valid_v4.max_drawdown_pct:+.1f}pp")

    print("\nparameter-neighbourhood plateau check (spot, inner-validation Sharpe, all 5 gap values):")
    grid_by_gap = {}
    for label, kw in _grid():
        m, _ = measure(KellyRegimeV15MacroVeto(**kw), *VALID, market=SPOT)
        grid_by_gap[kw["gap"]] = m.sharpe
    row = "  ".join(f"gap={g:.2f}:{grid_by_gap[g]:.2f}" for g in GAPS)
    print(f"  {row}")
    sharpes = list(grid_by_gap.values())
    spread = max(sharpes) - min(sharpes)
    print(f"  spread across grid: {spread:.2f}  "
          f"({'PLATEAU (spread < 0.2)' if spread < 0.2 else 'NOT a plateau (spread >= 0.2 noise floor)'})")

    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    return best_label, best_kw


# ------------------------------------------------------------------ artifact


def artifact(kw: dict | None = None, full_grid: bool = True) -> None:
    """Mandatory exposure-artifact check: R^2 of the candidate's target
    series against a mean-notional-matched flat rescale of v4's own
    target, on inner-validation, both markets. R^2 > 0.95 -> artifact.
    Run for the primary config, and (diagnostic, uncounted) across the
    full gap grid for robustness."""
    v4 = get_strategy(INCUMBENT)

    def _r2_for(cand_kw):
        out = {}
        for mname, market in MARKETS:
            cand = KellyRegimeV15MacroVeto(**cand_kw)
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
            out[mname] = (mean_abs_v4, mean_abs_cand, alpha, r2, corr)
        return out

    kw = kw or PRIMARY_KW
    print(f"exposure-artifact check, PRIMARY candidate={kw}")
    for mname, (mean_abs_v4, mean_abs_cand, alpha, r2, corr) in _r2_for(kw).items():
        verdict = "EXPOSURE-LEVEL ARTIFACT" if r2 > 0.95 else "genuinely different exposure shape"
        print(f"  {mname:10s} mean|v4|={mean_abs_v4:.3f} mean|cand|={mean_abs_cand:.3f} "
              f"alpha={alpha:.3f}  R^2(cand vs alpha*v4)={r2:.4f}  raw corr={corr:.4f}  {verdict}")

    if full_grid:
        print("\nexposure-artifact R^2 across the full gap grid (diagnostic, uncounted):")
        for label, gkw in _grid():
            for mname, (_, _, _, r2, _) in _r2_for(gkw).items():
                flag = "ARTIFACT" if r2 > 0.95 else "ok"
                print(f"  {label:30s} {mname:10s} R^2={r2:.4f}  [{flag}]")


# ------------------------------------------------------------------ causality


def _make_tampered_macro_dir(cut_day: pd.Timestamp, factor: float, tmp_root: Path) -> Path:
    """Copy the three real macro CSVs into a fresh dir, multiplying every
    row dated on/after ``cut_day`` by ``factor``. Never writes into the
    real ``data/`` dir."""
    out_dir = tmp_root / f"macro_x{factor:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"spx_daily.csv.gz": "sp500", "vix_daily.csv.gz": "vixcls", "dxy_daily.csv.gz": "dtwexbgs"}
    for fname, col in files.items():
        raw = pd.read_csv(DATA_DIR / fname)
        dates = pd.to_datetime(raw["date"])
        mask = dates >= cut_day.tz_localize(None)
        raw.loc[mask, col] = raw.loc[mask, col] * factor
        raw.to_csv(out_dir / fname, index=False, compression="gzip")
    return out_dir


def _fresh_broker():
    from tradebot.broker import PaperBroker
    return PaperBroker(market=FUTURES, start_balance=10_000.0)


def causality(kw: dict | None = None) -> None:
    """Two-independent-pathway tamper probe: price OHLCV AND the macro
    stress input tampered independently after a cut. Every decision at or
    before the cut must be unchanged. Restricted to strictly pre-2023
    bars. Plus a no-macro-data identity check (candidate must degenerate
    to v4's own 3-anchor average exactly when stress_z is entirely
    unavailable)."""
    kw = kw or PRIMARY_KW

    pre2023 = DF[DF.index < OOS_START]
    df = pre2023.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]
    cut_day = df.index[cut].normalize()

    def strategy_for(data_dir: Path | None):
        s = KellyRegimeV15MacroVeto(**kw)
        if data_dir is not None:
            def patched(frame, _dd=data_dir):
                if "stress_z_visible" in frame.columns:
                    return frame["stress_z_visible"]
                return compute_macro_stress(frame, _dd)
            s._stress_series = patched  # noqa: SLF001 (deliberate, test-only monkeypatch)
        return s

    def run_probe(name, tamper_ohlcv_fn=None, macro_dir_up=None, macro_dir_down=None):
        up, down = df.copy(), df.copy()
        if tamper_ohlcv_fn is not None:
            tamper_ohlcv_fn(up, down)
        if macro_dir_up is not None:
            up = up.drop(columns=["stress_z_visible"])
            down = down.drop(columns=["stress_z_visible"])

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

        a = decisions(up, macro_dir_up)
        b = decisions(down, macro_dir_down)
        bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
        print(f"[{name}] tampered from bar {cut:,} of {len(df):,} "
              f"(calendar day {cut_day.date()}); checked bars {bars}")
        print("  FAIL - reads the future at bars " + str(bad) if bad
              else "  PASS - every decision at or before the cut is unchanged")

        pa = strategy_for(macro_dir_up).prepare(up.copy())
        pb = strategy_for(macro_dir_down).prepare(down.copy())
        for col in ("target", "v15_frac", "v15_veto"):
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

    tmp_root = Path(tempfile.mkdtemp(prefix="v15_macro_veto_causality_"))
    try:
        run_probe("PRICE tamper (standard)", tamper_ohlcv_fn=tamper_ohlcv)

        macro_dir_up = _make_tampered_macro_dir(cut_day, 50.0, tmp_root)
        macro_dir_down = _make_tampered_macro_dir(cut_day, 1.0 / 50.0, tmp_root)
        run_probe("MACRO tamper (the stress_z pathway)",
                   macro_dir_up=macro_dir_up, macro_dir_down=macro_dir_down)

        run_probe("both at once", tamper_ohlcv_fn=tamper_ohlcv,
                   macro_dir_up=macro_dir_up, macro_dir_down=macro_dir_down)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # identity check: no macro data at all -> candidate degenerates to v4's
    # own unmodified 3-anchor average exactly (veto never fires when
    # stress_z is entirely unavailable, per _macro_veto_vote's NaN-> "calm"
    # default).
    v4 = get_strategy(INCUMBENT)
    frame = df.iloc[-20_000:].copy()
    frame_no_macro = frame.drop(columns=["stress_z_visible"]).copy()
    frame_no_macro["stress_z_visible"] = np.nan  # force entirely unavailable

    cand = KellyRegimeV15MacroVeto(**kw)
    t_v4 = v4.prepare(frame.copy())["target"].to_numpy()
    t_cand_no_macro = cand.prepare(frame_no_macro.copy())["target"].to_numpy()
    worst = float(np.nanmax(np.abs(t_v4 - t_cand_no_macro)))
    print(f"\nidentity check (no macro data available -> candidate recovers v4 exactly): "
          f"max|diff|={worst:.3e}  {'PASS' if worst < 1e-9 else 'FAIL'}")


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """SECONDARY pre-registered falsification test (ROUTINE.md's standard
    step-2 menu item). ETH-USD Bitfinex spot (2016-03 -> 2019-12-31)
    overlaps macro data coverage (2016-06 ->) almost entirely -- the
    macro signal is market-wide, not BTC- or ETH-specific, so it is
    aligned onto the ETH bar grid exactly as it is onto BTC's. Every
    candidate config vs v4 control, on BTC (control, identical
    pipeline/window, pre-2020 only) and ETH (test).

    PRE-REGISTERED RULE, fixed before results are read: if the candidate
    underperforms v4 on ETH, or is visibly worse on ETH than on the BTC
    control through the identical code, this direction fails. Since the
    macro signal is asset-agnostic by construction, an ETH-only failure
    is itself informative and must be reported, not hidden.
    """
    eth_spot = load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")
    eth_stress = compute_macro_stress(eth_spot, DATA_DIR)
    eth_df = eth_spot.copy()
    eth_df["stress_z_visible"] = eth_stress

    overlap = eth_df["stress_z_visible"].dropna()
    print(f"ETH spot file: {len(eth_spot):,} bars  {eth_spot.index[0]:%Y-%m-%d} -> "
          f"{eth_spot.index[-1]:%Y-%m-%d}")
    print(f"macro stress coverage overlapping ETH spot: {len(overlap):,} bars "
          f"visible from {overlap.index[0]:%Y-%m-%d} to {overlap.index[-1]:%Y-%m-%d}")

    # BTC control is deliberately restricted to the identical pre-2020
    # window ETH covers, matching R-53's own convention (never the full
    # inner-train+valid BTC frame for this specific comparison).
    btc_control = DF[DF.index < "2020-01-01"]

    frames = {"BTC (control, pre-2020)": btc_control, "ETH (test)": eth_df}
    results = {}
    for asset, frame in frames.items():
        print(f"\n{asset}  {len(frame):,} bars  {frame.index[0]:%Y-%m-%d} -> {frame.index[-1]:%Y-%m-%d}")
        results[asset] = {}
        for mname, market in MARKETS:
            m_v4, _ = measure(get_strategy(INCUMBENT), None, None, df=frame, market=market)
            line(f"{INCUMBENT} (control)", m_v4, mname)
            asset_results = {"v4": m_v4}
            for label, kw in _grid():
                cand = KellyRegimeV15MacroVeto(**kw)
                m_c, _ = measure(cand, None, None, df=frame, market=market)
                line(f"v15[{label}]", m_c, mname)
                asset_results[label] = m_c
            results[asset][mname] = asset_results

    print("\nfalsification verdict per config (candidate/v4 final-balance ratio, BTC control vs ETH test):")
    any_fail = False
    for label, kw in _grid():
        for mname, market in MARKETS:
            btc_r = (results["BTC (control, pre-2020)"][mname][label].final_balance
                     / results["BTC (control, pre-2020)"][mname]["v4"].final_balance)
            eth_r = results["ETH (test)"][mname][label].final_balance / results["ETH (test)"][mname]["v4"].final_balance
            worse_on_eth = eth_r < btc_r - 0.02  # small tolerance for noise
            eth_beats_v4 = eth_r >= 1.0
            flag = "FAIL" if (worse_on_eth and not eth_beats_v4) else ("caution" if worse_on_eth else "ok")
            if flag == "FAIL":
                any_fail = True
            print(f"  {label:30s} {mname:10s} BTC ratio={btc_r:.3f}x  ETH ratio={eth_r:.3f}x  [{flag}]")
    print(f"\nprimary candidate ({PRIMARY_KW}) checked above; "
          f"overall secondary-falsification verdict: {'FAIL' if any_fail else 'no outright FAIL by the pre-registered rule'}")


# --------------------------------------------------------------------- driver


def all_checks() -> None:
    print("=" * 78)
    print("STEP 2b -- descriptive: veto vote frequency vs anchor votes")
    print("=" * 78)
    descriptive()
    print("\n" + "=" * 78)
    print("PRIMARY PRE-REGISTERED FALSIFICATION TEST -- lead-time check")
    print("=" * 78)
    leadtime()
    print("\n" + "=" * 78)
    print("STEP 3 -- sweep (inner-train)")
    print("=" * 78)
    sweep()
    print("\n" + "=" * 78)
    print("STEP 3 -- select (inner-validation, both markets) + plateau")
    print("=" * 78)
    select()
    print("\n" + "=" * 78)
    print("EXPOSURE-ARTIFACT CHECK")
    print("=" * 78)
    artifact()
    print("\n" + "=" * 78)
    print("CAUSALITY / NO-LOOKAHEAD PROBE (both pathways + identity check)")
    print("=" * 78)
    causality()
    print("\n" + "=" * 78)
    print("SECONDARY FALSIFICATION TEST -- ETH + BTC control")
    print("=" * 78)
    eth()
    print(f"\ntotal distinct configurations evaluated (N_EVALUATED): {N_EVALUATED}")


if __name__ == "__main__":
    cmds = {"descriptive": descriptive, "leadtime": leadtime, "sweep": sweep, "select": select,
            "artifact": artifact, "causality": causality, "eth": eth, "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/kelly_regime_v15_macro_veto.py [{'|'.join(cmds)}]")
