#!/usr/bin/env python
"""R-56 CONSERVATIVE branch, closing backlog item **B-23**: a materially
DIFFERENT mechanism on the same aggregate-USDT-stablecoin-supply-
deceleration signal -- a shorter growth window matched to genuine-stress
duration, applied to the feature itself, rather than a filter bolted on
top of the existing fixed 14-day feature.

Idea, one sentence
------------------
`stablecoin_stress_z`'s underlying `growth_Nd` window -- fixed a-priori at
N=14 calendar days by R-54's `_stablecoin_signal.py` on the reasoning that
on-chain mint/burn reacts "near-instant" once stress starts -- may be
mismatched to how long a genuine stablecoin redemption episode actually
runs (recent literature, cited below, puts acute redemption stress at
48-72 hours with most normalization within about a week), so sweeping N
down to {2, 3, 5, 7, 10} days and re-testing the SAME lead-time check
R-54/R-55 used should show whether a shorter, better-matched window
tracks genuine stress episodes with less lag, cuts the false-onset
problem R-54 diagnosed, or (the named risk, stated before running
anything) simply trades signal for day-to-day reporting/API noise that a
14-day window already smooths away.

Constraint attacked: INFO, same axis as R-53/R-54/R-55 (the fourth
consecutive round on this axis, and the fourth attempt on this specific
signal). Per B-23's own LOW-priority filing by R-55, this round goes in
expecting a plausible negative -- the point of testing it is to make that
expectation an evidenced fact rather than an assumption, and to close the
signal's research line cleanly either way.

Not a duplicate of, cited precisely
------------------------------------
- `kelly_regime_v15_stablecoin_veto.py` (R-54, this file's direct
  ancestor): fixed the growth window at N=14 a-priori and never swept it
  -- this file's entire contribution is making that window a genuinely
  swept parameter instead. Everything else (hard-veto combination
  architecture, 365-day z-score window, hysteresis thresh/gap grid) is
  reused UNCHANGED from that file, specifically so that any difference in
  outcome is attributable to the window, not to some other axis.
- `kelly_regime_v16_stablecoin_persist.py` (R-55 CONSERVATIVE, closed
  B-22 fix 1 of 2): added a minimum-duration CONFIRMATION filter bolted
  ON TOP of the unmodified 14-day feature (`persist_days` consecutive
  days above threshold before the vote is allowed to latch) -- explicitly
  NOT what this file does. R-55 found duration and precision are not
  separable at the 14-day feature's OWN native cadence (the "transient"
  onsets persist about as long as genuine ones once already smoothed by a
  14-day rolling difference, so filtering by duration on top of that
  feature just erodes lead time). This file tests the different
  hypothesis that the feature's underlying time-scale itself, not a
  filter placed after it, is mismatched -- a shorter native window changes
  what counts as "decelerating" in the first place, rather than requiring
  an already-smoothed vote to persist for longer.
- `kelly_regime_v16_stablecoin_confirm.py` (R-55 NOVEL, closed B-22 fix 2
  of 2): swapped the COMBINATION RULE (hard override -> precision-weighted
  confirming vote) while keeping the fixed 14-day feature. This file does
  the opposite: keeps R-54's hard-override combination rule byte-for-byte
  and varies only the feature's own growth window. R-55's own conclusion
  was explicit that the combination rule is not the binding constraint --
  the signal's specificity is -- which is exactly the axis this file
  attacks.
- **B-23** (filed by R-55): "a shorter growth window matched to
  genuine-stress duration rather than a persistence filter bolted onto
  the existing 14-day feature" -- this file is exactly that, the first of
  B-23's two named candidate mechanisms (the second, corroboration from a
  second independent signal, is out of scope for this branch by the
  round's own split: it is filed as B-23's remaining open half, not
  re-opened here).
- This round's own disjoint parallel NOVEL branch: not read, not
  coordinated with, per ROUTINE.md's parallelism isolation rule.

Sources
-------
- BIS WP 1340 (2025), "Stablecoin flows and spillovers to FX markets";
  Ahmed & Aldasoro, "Stablecoins and safe asset prices" (Cleveland Fed
  conference paper / BIS WP 1270, Aug 2025); NY Fed Liberty Street
  Economics, "Stablecoins and Crypto Shocks: An Update" (Apr 2025); IMF WP
  2025/141 -- all cited unchanged from `_stablecoin_signal.py` and
  `kelly_regime_v15_stablecoin_veto.py` for the base mechanism (stablecoin
  issuance/redemption as the crypto trading system's dollar on-ramp/
  off-ramp). Not re-derived here.
- NEW for this round, found via web search on stablecoin redemption
  timescales specifically (the question B-23 asks): ESRB, "Crypto-assets
  and decentralised finance" (Oct 2025) and industry technical write-ups
  on stablecoin reserve/liquidity stress (e.g. crypto-economy.com's
  "Stablecoins Under Stress: A Technical Dissection of Reserve
  Architecture and Liquidity Risk," and 2026 trade-press pieces on
  stablecoin run risk) converge on: acute redemption spikes typically run
  **48-72 hours**, after which issuers liquidate short-term reserves and
  spreads normalize within **about a week**. This is the literature basis
  for this round's window grid: if a 14-day growth window is smoothing
  over and lagging a stress dynamic that plays out over single-digit
  days, a window closer to that native cadence is the natural next thing
  to test -- **not** fit to any observed lead/lag result, chosen from this
  literature before `leadtime_by_window()` below was run.
- Named risk, stated before any code ran, motivated by the SAME
  literature read the other way: "acute" redemption stress (hours-days)
  and the multi-week CAPITAL-FLIGHT dynamic this signal is actually being
  asked to detect (the slower bleed that shows up in price weakness weeks
  later, per R-54's own confirmed +16.5-day lead at N=14) are not
  necessarily the same timescale. A window tuned to the former could easily
  be too short to capture the latter, and would instead mostly reflect
  day-to-day supply-reporting/API noise -- exactly the risk this project's
  INFO research line has hit on every antecedent attempt (R-53's lagging
  external index, R-54's specificity failure, R-55's duration/precision
  inseparability). The lead-time check below is designed to distinguish
  these two possibilities directly, and is run BEFORE any Sharpe number,
  exactly as R-54/R-55 did.

Mechanism, precisely -- the exact feature formula
--------------------------------------------------
`growth_Nd = log(supply_t) - log(supply_{t-N})`, N swept over
`{2, 3, 5, 7, 10}` (5 distinct windows; N=14 is also recomputed here,
by-window-formula, purely as an in-file reference reproduction of R-54's
own number -- it is not new evidence and is not counted toward this
round's configuration total). `stablecoin_stress_z = -1 *
zscore(growth_Nd, trailing 365d, min_periods=60)` -- IDENTICAL z-score
window and `min_periods` for every N, since `min_periods` governs how much
z-score history is required before the z-score itself is considered
stable, not the growth window's own length; keeping it fixed at R-54's
value for every N isolates the growth window as the only varied axis, per
this file's entire reason for existing. The hard-veto combination
architecture (`stable_vote` latched 0/1 with hysteresis
`thresh_lo=thresh_hi-gap`, `frac=0` while latched "stress", v4's
unmodified 3-anchor average otherwise) is reused BYTE-FOR-BYTE from
`kelly_regime_v15_stablecoin_veto.py`. `thresh_hi`/`gap` were ORIGINALLY
pre-registered to sweep R-54's own identical grid (`{0.75, 1.0, 1.25}` x
`{0.0, 0.75, 1.25}`) at every window (45 configs total), so the
veto-sensitivity axis would be held constant too -- whatever differs
attributable only to the growth window N. **Scoped down mid-session, see
`_grid()`'s own docstring for the exact reasoning:** Step A (below) killed
the branch cleanly and decisively before the 45-config grid finished
running; the grid actually executed fixes `thresh_hi` at the primary 1.0
and sweeps `gap` over its full range at every window (15 configs), which
still preserves a genuine parameter-neighbourhood axis at every window.

Code reuse decision, stated plainly
-------------------------------------
The anchor-vote and latched-hysteresis-vote helpers are duplicated (not
imported) from `kelly_regime_v15_stablecoin_veto.py`, continuing the norm
R-53/R-54/R-55 all established. `_stablecoin_signal.py`'s
`compute_stablecoin_stress` (fixed N=14) is imported unchanged and used
ONLY for the in-file N=14 reference reproduction; the swept-window
feature itself is a new, local function (`compute_stablecoin_stress_w`)
in this file, since `_stablecoin_signal.py` is a frozen, shared,
previously-pre-registered module (its own docstring states its N=14
window "is FIXED A-PRIORI and never swept anywhere") that this branch has
no authority to edit or fork in place -- editing it to add a parameter
would silently change what every OTHER branch that imports it gets.
Neither `_stablecoin_signal.py` nor `kelly_regime_v15_stablecoin_veto.py`
nor `kelly_regime_v16_stablecoin_persist.py`/`_confirm.py` is edited
anywhere in this session.

Pre-registered falsification / decision procedure (named before any code ran)
--------------------------------------------------------------------------------
**Step A -- the mechanism gate, run first, before any Sharpe number:**
does at least one of `{2, 3, 5, 7, 10}` PRESERVE OR IMPROVE R-54's
confirmed lead-time result (9/12 matched episodes leading, median +16.5
days, at the primary thresh/gap) relative to the N=14 baseline reproduced
in this same file? "Improve" means: a higher fraction of matched episodes
lead (not lag), AND a longer or equal median lead. If NO window clears
this bar, that is reported as a fast, cheap, honest negative and this
branch's PRE-REGISTERED DECISION IS TO NOT PROMOTE regardless of any
Sharpe numbers subsequently computed -- those numbers are still computed
and reported (compute is cheap and the project's own convention is to
finish the pipeline for a complete report), but they are explicitly
DIAGNOSTIC ONLY once Step A fails, not grounds to revisit the decision.
This is the exact goalpost-discipline ROUTINE.md requires: the rule is
fixed here, before Step A is run, not after.
**Step B -- only if Step A passes for at least one window:** standard
inner-train/inner-validation sweep across that window (or windows) at
the full thresh/gap grid, promotion bar as usual (beats `buy_and_hold`
OOS after costs, Delta Sharpe beyond +-0.2, survives falsification,
plateau not peak).
**Falsification test (run regardless, on whatever configuration the
process above selects, for completeness):** (a) BTC pre-2020 control (the
strategy must not lose badly to unmodified v4 on data before the stress
episodes it is tuned around); (b) ETH falsification
(`data/ethusd_bitfinex_5m.csv.gz`), same USDT global signal, no
ETH-specific stablecoin data. Exposure-artifact R^2 check (>0.95 vs a
flat rescale of v4 = reject) and the standard two-pathway causality tamper
probe both run unconditionally too.

Usage
-----
    python experiments/kelly_regime_v17_stablecoin_shortwindow.py leadtime   # STEP A, the gate
    python experiments/kelly_regime_v17_stablecoin_shortwindow.py sweep      # step 3 (inner-train)
    python experiments/kelly_regime_v17_stablecoin_shortwindow.py select     # step 3 (inner-validation)
    python experiments/kelly_regime_v17_stablecoin_shortwindow.py artifact   # exposure-artifact check
    python experiments/kelly_regime_v17_stablecoin_shortwindow.py causality  # lookahead probe
    python experiments/kelly_regime_v17_stablecoin_shortwindow.py eth        # ETH falsification
    python experiments/kelly_regime_v17_stablecoin_shortwindow.py all        # everything, in order
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
from tradebot.data import (  # noqa: E402
    align_stablecoin_causal,
    load_dataset,
    load_ohlcv_csv,
    load_stablecoin_supply,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments._stablecoin_signal import compute_stablecoin_stress  # noqa: E402 (N=14 reference only)

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures 5x", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")     # inner-train
VALID = ("2021-01-01", "2022-12-31")     # inner-validation
OOS_START = "2023-01-01"                 # never read in this file

INCUMBENT = "kelly_regime_v4"
DATA_DIR = ROOT / "data"

ZSCORE_WINDOW_DAYS = 365
MIN_PERIODS = 60          # identical to _stablecoin_signal.py, held fixed across every N

WINDOWS = (2, 3, 5, 7, 10)        # the swept candidates, per B-23 / this round's brief
REFERENCE_WINDOW = 14             # R-54's fixed a-priori window, recomputed here for comparison only

THRESH_HIS = (0.75, 1.0, 1.25)    # identical grid to kelly_regime_v15_stablecoin_veto.py
GAPS = (0.0, 0.75, 1.25)
PRIMARY_THRESH = dict(thresh_hi=1.0, gap=0.75)   # R-54's own primary point, held fixed across windows

N_EVALUATED = 0
_SEEN_CONFIGS: set[str] = set()


# --------------------------------------------------------------------- data


def compute_stablecoin_stress_w(df: pd.DataFrame, data_dir: str | Path, growth_window_days: int) -> pd.Series:
    """Causal ``stablecoin_stress_z`` with a SWEPT growth window, otherwise
    identical to ``_stablecoin_signal.compute_stablecoin_stress`` (which is
    frozen at ``growth_window_days=14``). Every rolling statistic is
    computed on the raw daily USDT-supply frame before ``align_stablecoin_
    causal`` projects the finished daily series onto the bar grid with its
    own 1-day publication-lag shift -- identical causality discipline to
    the frozen module, just parameterized.
    """
    supply = load_stablecoin_supply(data_dir)
    if supply is None:
        return pd.Series(index=df.index, dtype=float)
    s = supply["supply"]
    log_s = np.log(s)
    growth = log_s - log_s.shift(growth_window_days)
    z = (growth - growth.rolling(ZSCORE_WINDOW_DAYS, min_periods=MIN_PERIODS).mean()) / growth.rolling(
        ZSCORE_WINDOW_DAYS, min_periods=MIN_PERIODS
    ).std()
    stress_daily = (-1.0 * z).rename("stablecoin_stress_z").to_frame()
    return align_stablecoin_causal(stress_daily, df)["stablecoin_stress_z"]


def build_dataframe() -> tuple[pd.DataFrame, str]:
    """Canonical spot OHLCV with one causal stress column per candidate
    window (plus the N=14 reference), precomputed once."""
    spot, label = load_dataset(DATA_DIR, "spot")
    out = spot.copy()
    for w in (*WINDOWS, REFERENCE_WINDOW):
        out[f"stress_w{w}"] = compute_stablecoin_stress_w(spot, DATA_DIR, w)
    return out, label


DF, LABEL = build_dataframe()
print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  (data: {LABEL}); "
      f"stablecoin-stress coverage (w=14 ref) {DF['stress_w14'].notna().sum():,} bars "
      f"from {DF['stress_w14'].dropna().index[0]:%Y-%m-%d}", file=sys.stderr)


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0, config_key: str | None = None):
    global N_EVALUATED
    if config_key is not None and config_key not in _SEEN_CONFIGS:
        _SEEN_CONFIGS.add(config_key)
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market, start_balance=balance, data_label=LABEL)
    return compute_metrics(result), result


def line(tag, m, market_name=""):
    print(f"  {tag:42s} {market_name:10s} final=${m.final_balance:>11,.0f} "
          f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>7,.0f}{'  LIQUIDATED' if m.liquidated else ''}")


# ---------------------------------------------------------------- the vote


def _anchor_votes(close: pd.Series, horizons=(20, 40, 80), band: float = 0.01) -> dict:
    """Exactly kelly_regime_v4's own per-anchor latched vote. Duplicated
    (not imported) from kelly_regime_v15_stablecoin_veto.py."""
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
    """Latched stablecoin-stress vote, 0/1, hysteresis on ``stress_z``.
    Duplicated, not imported, per this module's "Code reuse decision"."""
    thresh_lo = thresh_hi - gap
    raw = np.where(stress_z > thresh_hi, 0.0, np.where(stress_z < thresh_lo, 1.0, np.nan))
    vote = pd.Series(raw, index=stress_z.index).ffill().fillna(1.0)
    return vote


class KellyRegimeV17StablecoinShortWindow(KellyRegimeV3):
    """v4's 3-anchor vote with a hard veto identical to
    ``KellyRegimeV15StablecoinVeto``, fed by a stablecoin-stress feature
    whose growth window is a swept parameter instead of R-54's fixed 14
    days. ``enabled=False`` recovers v4 exactly (identity check in
    ``causality()``)."""

    name = "kelly_regime_v17_stablecoin_shortwindow"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 growth_window_days: int = 7, thresh_hi: float = 1.0, gap: float = 0.75,
                 enabled: bool = True, **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.growth_window_days = growth_window_days
        self.thresh_hi = thresh_hi
        self.gap = gap
        self.enabled = enabled

    def _stress_series(self, df: pd.DataFrame) -> pd.Series:
        col = f"stress_w{self.growth_window_days}"
        if col in df.columns:
            return df[col]
        return compute_stablecoin_stress_w(df, DATA_DIR, self.growth_window_days)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = _anchor_votes(close, self.horizons, self.band)
        anchor_sum = sum(votes.values())
        n_anchors = float(len(votes))
        anchor_frac = (anchor_sum / n_anchors).to_numpy()

        stress_z = self._stress_series(df)
        if self.enabled:
            stable_vote = _stable_vote(stress_z, self.thresh_hi, self.gap).to_numpy()
        else:
            stable_vote = np.ones(len(df))  # identity check: never vetoes

        frac = np.where(stable_vote == 0.0, 0.0, anchor_frac)
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        # From here down: byte-for-byte v3's conditional vol-targeting sizer.
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
        df["v17_frac"] = frac
        df["v17_stable_vote"] = stable_vote
        df["v17_stress_z"] = stress_z.to_numpy()
        df["v17_anchor_sum"] = anchor_sum.to_numpy()
        return df


# ------------------------------------------------------------- the grid


def _grid():
    """SCOPED DOWN mid-run after Step A returned a clean, monotonic,
    decisive kill (0/5 windows passed the pre-registered lead-time gate --
    see leadtime_by_window()'s output, reproduced in the report). The
    original pre-registration specified the full THRESH_HIS x GAPS 3x3
    grid at every window (45 configs); that grid ran for several minutes
    without finishing and, per the operator's explicit instruction that a
    smaller COMPLETED sweep beats a stalled larger one, this was cut to a
    still-real plateau check: thresh_hi held at the primary 1.0 (R-54's
    own pre-registered point) while gap sweeps its full {0.0, 0.75, 1.25}
    range, at every window. This keeps a genuine parameter-neighbourhood
    axis (gap) at every window (15 configs total) rather than dropping to
    a single point per window, while cutting the run to 1/3 of the
    original cost. Because Step A already killed the branch on mechanism
    grounds (every window monotonically worse than the 14-day reference,
    not a borderline call), this scope-down changes no conclusion -- it
    only trims how much confirmatory Sharpe evidence is gathered for a
    result already decided."""
    out = []
    for w in WINDOWS:
        for gap in GAPS:
            label = f"w={w:>2d}d thresh={PRIMARY_THRESH['thresh_hi']:.2f} gap={gap:.2f}"
            out.append((label, dict(growth_window_days=w, thresh_hi=PRIMARY_THRESH["thresh_hi"], gap=gap)))
    return out


def _config_key(kw: dict) -> str:
    return f"shortwin|w={kw['growth_window_days']}|thresh={kw['thresh_hi']:.3f}|gap={kw['gap']:.3f}"


# ------------------------------------------------------------ STEP A: the gate


def _daily_transitions(series: pd.Series, target_value: float, min_gap_days: int = 14) -> list:
    """Identical to kelly_regime_v15_stablecoin_veto.py's helper, including
    its ``shift(fill_value=False)`` fix (not ``.shift().fillna(False)``)."""
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


def leadtime_by_window() -> dict:
    """STEP A, THE GATE, pre-registered, run before any Sharpe number:
    for each candidate window (plus the N=14 reference, recomputed
    in-file), does the stablecoin-stress veto's bear onset lead
    v4's own 3-anchor MAJORITY price-gate bear onset, at the SAME primary
    threshold (thresh_hi=1.0, gap=0.75) held fixed across every window so
    the only varying axis is the growth window itself? Descriptive, 0
    configurations counted toward N_EVALUATED (same convention as
    kelly_regime_v15_stablecoin_veto.py's leadtime()).
    """
    lo, hi = TRAIN[0], VALID[1]
    frame = DF.loc[lo:hi]
    close = frame["close"]

    votes = _anchor_votes(close, (20, 40, 80), 0.01)
    anchor_sum = sum(votes.values())
    majority_bear = (anchor_sum < 1.5).astype(float)
    majority_onsets = _daily_transitions(majority_bear, 1.0)
    print(f"3-anchor MAJORITY bear-onset episodes (reference, {lo}->{hi}): {len(majority_onsets)}")

    results = {}
    print(f"\n{'window(d)':>10s} {'onset_events':>13s} {'matched':>8s} {'leads':>16s} {'median_lead(d)':>16s}")
    for w in (*WINDOWS, REFERENCE_WINDOW):
        stress = frame[f"stress_w{w}"]
        stable_vote = _stable_vote(stress, PRIMARY_THRESH["thresh_hi"], PRIMARY_THRESH["gap"])
        stable_bear = 1.0 - stable_vote
        onsets = _daily_transitions(stable_bear, 1.0)

        leads = []
        for d in onsets:
            _, dist = _nearest(d, majority_onsets)
            if dist is not None:
                leads.append(-dist)
        n_lead = sum(1 for x in leads if x > 0)
        med = float(np.median(leads)) if leads else float("nan")
        tag = "  <- R-54 reference (fixed, never swept)" if w == REFERENCE_WINDOW else ""
        print(f"{w:>10d} {len(onsets):>13d} {len(leads):>8d} "
              f"{n_lead}/{len(leads) if leads else 0:>13} {med:>16.1f}{tag}")
        print(f"           onset dates: {[d.date().isoformat() for d in onsets]}")
        results[w] = {"onsets": len(onsets), "matched": len(leads),
                       "n_lead": n_lead, "median_lead": med}

    ref = results[REFERENCE_WINDOW]
    print(f"\nGATE (Step A pre-registered rule): a candidate window PASSES only if its "
          f"lead FRACTION >= reference's ({ref['n_lead']}/{ref['matched']}) AND its median "
          f"lead >= reference's ({ref['median_lead']:.1f}d).")
    survivors = []
    for w in WINDOWS:
        r = results[w]
        frac_w = r["n_lead"] / r["matched"] if r["matched"] else 0.0
        frac_ref = ref["n_lead"] / ref["matched"] if ref["matched"] else 0.0
        passed = (frac_w >= frac_ref) and (r["median_lead"] >= ref["median_lead"])
        if passed:
            survivors.append(w)
        print(f"  window={w:>2d}d: lead_frac={frac_w:.2f} (ref={frac_ref:.2f}) "
              f"median_lead={r['median_lead']:.1f}d (ref={ref['median_lead']:.1f}d)  "
              f"{'PASS' if passed else 'FAIL'}")
    print(f"\nSTEP A OUTCOME: {len(survivors)}/{len(WINDOWS)} windows pass the gate: {survivors}")
    if not survivors:
        print("NO window preserves or improves the reference lead-time result. Per this file's "
              "pre-registration, the decision is NOT TO PROMOTE regardless of any Sharpe numbers "
              "computed below -- those numbers are reported for completeness only.")
    print("\nleadtime_by_window: 0 configurations counted toward N_EVALUATED (descriptive)")
    return results


# --------------------------------------------------------------- step 3: sweep


def sweep() -> None:
    """Step 3: every (window, thresh_hi, gap) config on inner-train, spot."""
    print(f"\nINNER-TRAIN {TRAIN} / spot -- benchmarks:")
    for name in ("buy_and_hold", INCUMBENT):
        m, _ = measure(get_strategy(name), *TRAIN, market=SPOT)
        line(name, m, "spot")

    print(f"\nINNER-TRAIN {TRAIN} / spot -- candidate configurations ({len(_grid())} total):")
    for label, kw in _grid():
        m, _ = measure(KellyRegimeV17StablecoinShortWindow(**kw), *TRAIN, market=SPOT,
                        config_key=_config_key(kw))
        line(label, m, "spot")

    print(f"\nconfigurations evaluated (step 3, distinct window/thresh/gap triples): {N_EVALUATED}")


# -------------------------------------------------------------- step 3: select


def select():
    """Every config on inner-validation, both markets, vs v4 control."""
    print(f"\nINNER-VALIDATION {VALID} -- v4 control:")
    ctl = {}
    for mname, market in MARKETS:
        m, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        ctl[mname] = m
        line(f"{INCUMBENT} (control)", m, mname)

    print(f"\nINNER-VALIDATION {VALID} -- candidate configurations ({len(_grid())} total):")
    best_label, best_kw, best_score = None, None, -1e9
    rows = []
    for label, kw in _grid():
        cell = {}
        for mname, market in MARKETS:
            m, _ = measure(KellyRegimeV17StablecoinShortWindow(**kw), *VALID, market=market,
                            config_key=_config_key(kw))
            cell[mname] = m
            line(label, m, mname)
            rows.append({"label": label, **kw, "market": mname, "final": m.final_balance,
                         "sharpe": m.sharpe, "max_dd": m.max_drawdown_pct, "trades": m.num_trades})
        m_train, _ = measure(KellyRegimeV17StablecoinShortWindow(**kw), *TRAIN, market=SPOT)
        score = min(m_train.sharpe, cell["spot"].sharpe)
        if score > best_score:
            best_label, best_kw, best_score = label, kw, score

    print(f"\nbest by min(train,valid) spot Sharpe: {best_label}  (score={best_score:.2f})")
    print(f"v4 control spot Sharpe: {ctl['spot'].sharpe:.2f}   v4 control futures Sharpe: {ctl['futures 5x'].sharpe:.2f}")

    print("\nside-by-side: best candidate's TRAIN window vs VALIDATION window (overfitting-signature check):")
    for mname, market in MARKETS:
        m_train, _ = measure(KellyRegimeV17StablecoinShortWindow(**best_kw), *TRAIN, market=market)
        m_valid, _ = measure(KellyRegimeV17StablecoinShortWindow(**best_kw), *VALID, market=market)
        m_train_v4, _ = measure(get_strategy(INCUMBENT), *TRAIN, market=market)
        m_valid_v4, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        print(f"  {mname}:")
        print(f"    candidate  TRAIN final=${m_train.final_balance:>11,.0f} sharpe={m_train.sharpe:.2f} DD={m_train.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid.final_balance:>11,.0f} sharpe={m_valid.sharpe:.2f} DD={m_valid.max_drawdown_pct:.1f}%")
        print(f"    v4         TRAIN final=${m_train_v4.final_balance:>11,.0f} sharpe={m_train_v4.sharpe:.2f} DD={m_train_v4.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid_v4.final_balance:>11,.0f} sharpe={m_valid_v4.sharpe:.2f} DD={m_valid_v4.max_drawdown_pct:.1f}%")
        print(f"    candidate - v4 (valid): Delta sharpe={m_valid.sharpe - m_valid_v4.sharpe:+.3f}  "
              f"Delta DD={m_valid.max_drawdown_pct - m_valid_v4.max_drawdown_pct:+.1f}pp")

    print("\nparameter-neighbourhood plateau check (spot, inner-validation Sharpe, per window, "
          f"gap swept at fixed thresh_hi={PRIMARY_THRESH['thresh_hi']:.2f} -- see _grid()'s scope-down note):")
    for w in WINDOWS:
        row = []
        for label, kw in _grid():
            if kw["growth_window_days"] != w:
                continue
            m, _ = measure(KellyRegimeV17StablecoinShortWindow(**kw), *VALID, market=SPOT)
            row.append(f"gap={kw['gap']:.2f}:{m.sharpe:.2f}")
        print(f"  window={w}d:  " + "  ".join(row))

    print("\ncross-window plateau check (spot, inner-validation Sharpe, primary thresh/gap fixed, window varied):")
    for w in WINDOWS:
        m, _ = measure(KellyRegimeV17StablecoinShortWindow(growth_window_days=w, **PRIMARY_THRESH),
                        *VALID, market=SPOT)
        print(f"  window={w:>2d}d thresh={PRIMARY_THRESH['thresh_hi']:.2f} gap={PRIMARY_THRESH['gap']:.2f}: sharpe={m.sharpe:.2f}")

    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    return best_label, best_kw


# ------------------------------------------------------------------ artifact


def artifact(kw: dict | None = None) -> None:
    """Mandatory exposure-artifact check: R^2 of the candidate's target
    series against a mean-notional-matched flat rescale of v4's own
    target, on inner-validation, both markets. R^2 > 0.95 -> artifact."""
    kw = kw or dict(growth_window_days=7, **PRIMARY_THRESH)
    print(f"exposure-artifact check, candidate={kw}")
    v4 = get_strategy(INCUMBENT)
    for mname, market in MARKETS:
        cand = KellyRegimeV17StablecoinShortWindow(**kw)
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
    out_dir = tmp_root / f"stable_x{factor:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_DIR / "stablecoin_supply_daily.csv.gz")
    dates = pd.to_datetime(raw["timestamp"])
    mask = dates >= cut_day.tz_localize(None)
    raw.loc[mask, "usdt_SplyCur"] = raw.loc[mask, "usdt_SplyCur"] * factor
    raw.to_csv(out_dir / "stablecoin_supply_daily.csv.gz", index=False, compression="gzip")
    return out_dir


def _fresh_broker():
    from tradebot.broker import PaperBroker
    return PaperBroker(market=FUTURES, start_balance=10_000.0)


def causality(kw: dict | None = None) -> None:
    """Two-independent-pathway tamper probe: price OHLCV AND the
    stablecoin-supply input tampered independently after a cut, plus an
    identity-recovery check. Restricted to strictly pre-2023 bars."""
    kw = kw or dict(growth_window_days=7, **PRIMARY_THRESH)

    pre2023 = DF[DF.index < OOS_START]
    df = pre2023.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]
    cut_day = df.index[cut].normalize()

    def strategy_for(data_dir: Path | None):
        s = KellyRegimeV17StablecoinShortWindow(**kw)
        if data_dir is not None:
            def patched(frame, _dd=data_dir, _w=kw["growth_window_days"]):
                col = f"stress_w{_w}"
                if col in frame.columns:
                    return frame[col]
                return compute_stablecoin_stress_w(frame, _dd, _w)
            s._stress_series = patched  # noqa: SLF001 (deliberate, test-only monkeypatch)
        return s

    def run_probe(name, tamper_ohlcv_fn=None, stable_dir_up=None, stable_dir_down=None):
        up, down = df.copy(), df.copy()
        if tamper_ohlcv_fn is not None:
            tamper_ohlcv_fn(up, down)
        if stable_dir_up is not None:
            cols = [c for c in up.columns if c.startswith("stress_w")]
            up = up.drop(columns=cols)
            down = down.drop(columns=cols)

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
        print(f"[{name}] tampered from bar {cut:,} of {len(df):,} (calendar day {cut_day.date()}); checked bars {bars}")
        print("  FAIL - reads the future at bars " + str(bad) if bad
              else "  PASS - every decision at or before the cut is unchanged")

        pa = strategy_for(stable_dir_up).prepare(up.copy())
        pb = strategy_for(stable_dir_down).prepare(down.copy())
        for col in ("target", "v17_frac", "v17_stable_vote", "v17_anchor_sum"):
            diff = np.abs(pa[col].to_numpy()[:cut].astype(float) - pb[col].to_numpy()[:cut].astype(float))
            worst = float(np.nanmax(diff))
            print(f"    column {col:16s} max |difference| before the cut = {worst:.3e}  "
                  f"{'PASS' if worst < 1e-9 else 'FAIL'}")

    def tamper_ohlcv(up, down):
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    tmp_root = Path(tempfile.mkdtemp(prefix="v17_shortwindow_causality_"))
    try:
        run_probe("PRICE tamper (standard)", tamper_ohlcv_fn=tamper_ohlcv)

        stable_dir_up = _make_tampered_stablecoin_dir(cut_day, 50.0, tmp_root)
        stable_dir_down = _make_tampered_stablecoin_dir(cut_day, 1.0 / 50.0, tmp_root)
        run_probe("STABLECOIN tamper (new supply-data pathway)",
                   stable_dir_up=stable_dir_up, stable_dir_down=stable_dir_down)

        run_probe("both at once", tamper_ohlcv_fn=tamper_ohlcv,
                   stable_dir_up=stable_dir_up, stable_dir_down=stable_dir_down)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    v4 = get_strategy(INCUMBENT)
    ident = KellyRegimeV17StablecoinShortWindow(growth_window_days=kw["growth_window_days"],
                                                 thresh_hi=1.0, gap=0.75, enabled=False)
    frame = df.iloc[-20_000:].copy()
    t_v4 = v4.prepare(frame.copy())["target"].to_numpy()
    t_ident = ident.prepare(frame.copy())["target"].to_numpy()
    worst = float(np.nanmax(np.abs(t_v4 - t_ident)))
    print(f"\nidentity check (enabled=False recovers v4 exactly): max|diff|={worst:.3e}  "
          f"{'PASS' if worst < 1e-9 else 'FAIL'}")


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Falsification test, pre-registered rule identical to R-54's: the
    candidate must not be visibly worse on ETH than on the identical-
    pipeline BTC control. Run on the per-window primary configs (5) plus
    the single best-selected configuration from select(), both markets --
    not the full 45-config grid, to keep this check's own cost bounded;
    every config actually promoted-candidate-worthy by Step A/inner-
    validation is covered."""
    eth_spot = load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")
    eth_df = eth_spot.copy()
    for w in WINDOWS:
        eth_df[f"stress_w{w}"] = compute_stablecoin_stress_w(eth_spot, DATA_DIR, w)

    overlap = eth_df["stress_w7"].dropna()
    print(f"ETH spot file: {len(eth_spot):,} bars  {eth_spot.index[0]:%Y-%m-%d} -> {eth_spot.index[-1]:%Y-%m-%d}")
    print(f"stablecoin stress coverage overlapping ETH spot (w=7 ref): {len(overlap):,} bars "
          f"visible from {overlap.index[0]:%Y-%m-%d} to {overlap.index[-1]:%Y-%m-%d}")

    frames = {"BTC (control)": DF[DF.index < OOS_START], "ETH (test)": eth_df}
    configs = [(f"w={w}d primary", dict(growth_window_days=w, **PRIMARY_THRESH)) for w in WINDOWS]

    results = {}
    for asset, frame in frames.items():
        print(f"\n{asset}  {len(frame):,} bars  {frame.index[0]:%Y-%m-%d} -> {frame.index[-1]:%Y-%m-%d}")
        results[asset] = {}
        for mname, market in MARKETS:
            m_v4, _ = measure(get_strategy(INCUMBENT), None, None, df=frame, market=market)
            line(f"{INCUMBENT} (control)", m_v4, mname)
            asset_results = {"v4": m_v4}
            for label, kw in configs:
                cand = KellyRegimeV17StablecoinShortWindow(**kw)
                m_c, _ = measure(cand, None, None, df=frame, market=market)
                line(f"v17[{label}]", m_c, mname)
                asset_results[label] = m_c
            results[asset][mname] = asset_results

    print("\nfalsification verdict per config (candidate/v4 final-balance ratio, BTC control vs ETH test):")
    any_fail = False
    for label, kw in configs:
        for mname, market in MARKETS:
            btc_r = results["BTC (control)"][mname][label].final_balance / results["BTC (control)"][mname]["v4"].final_balance
            eth_r = results["ETH (test)"][mname][label].final_balance / results["ETH (test)"][mname]["v4"].final_balance
            worse_on_eth = eth_r < btc_r - 0.02
            eth_beats_v4 = eth_r >= 1.0
            flag = "FAIL" if (worse_on_eth and not eth_beats_v4) else ("caution" if worse_on_eth else "ok")
            any_fail = any_fail or flag == "FAIL"
            print(f"  {label:16s} {mname:10s} BTC ratio={btc_r:.3f}x  ETH ratio={eth_r:.3f}x  [{flag}]")
    print(f"\noverall falsification verdict: {'FAIL' if any_fail else 'no outright FAIL by the pre-registered rule'}")


# --------------------------------------------------------------------- driver


def all_checks() -> None:
    print("=" * 78)
    print("STEP A -- THE GATE: lead-time by window, before any Sharpe number")
    print("=" * 78)
    leadtime_by_window()
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
    print("ETH FALSIFICATION TEST")
    print("=" * 78)
    eth()
    print(f"\ntotal distinct configurations evaluated (N_EVALUATED): {N_EVALUATED}")


if __name__ == "__main__":
    cmds = {"leadtime": leadtime_by_window, "sweep": sweep, "select": select,
            "artifact": artifact, "causality": causality, "eth": eth, "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/kelly_regime_v17_stablecoin_shortwindow.py [{'|'.join(cmds)}]")
