#!/usr/bin/env python
"""R-116 NOVEL branch: ``PanelDisagreementBrakeKellyV4`` -- a bounded,
monotonic, NEVER-INCREASE-ONLY discount on ``kelly_regime_v4``'s own
``frac * scale`` product, driven by the Bomberger (1996) cross-sectional
standard deviation of ``kelly_regime_v4``'s own, byte-identical 20/40/80-day
anchor vote, computed INDEPENDENTLY per instrument across a panel of BTC plus
the six other Coinbase spot instruments this project already has 5-minute
data for (BCH, LTC, ETC, DASH, LINK, XTZ). The flipped construction of R-106
(disagreement across MODELS on one asset): here the model is held fixed at
v4's own vote and the panel is varied by ASSET instead.

Full citation trail, literature grounding (Zarnowitz & Lambros 1987;
Bomberger 1996; Zaremba et al. 2019/2020; Mercik et al. 2025; Zweig 1986),
the constraint attacked (INFO primary, ERR secondary), and the exhaustive
non-duplication argument against every related prior round all live in
``experiments/r116_shared.py``'s own module docstring (read in full before
this file was written) -- not re-derived here beyond the one-paragraph
summary above. This file never edits, and never reads a holdout bar from,
``r116_shared.py`` or any CONSERVATIVE-branch file; the one disclosed
exception is the pre-registered HOLDOUT CONSULT in section 6 below, run
once, after the decision rule in section 4 is frozen.

=====================================================================
PRE-REGISTRATION, dated 2026-08-24, frozen BEFORE any bar at or after
``OOS_START = 2023-01-01`` was read by this file. Anything below later
contradicted by what actually happened is stated in the run's own printed
output and in the final report, never edited back into this banner.
=====================================================================

1. MECHANISM (one sentence): at every bar, compute the cross-sectional
   standard deviation of a panel of ``kelly_regime_v4``-style anchor votes
   (BTC's own plus the six alts', each built independently by
   ``r116_shared.build_panel_votes`` with v4's unmodified 20/40/80-day
   anchors), map it through a bounded, monotonic, NEVER-INCREASE-ONLY
   clipped-linear discount into ``[floor, 1.0]``, and multiply v4's own
   unchanged ``frac * scale`` product by that discount before v4's own 10%
   deadband is (re-)applied.

2. CONSTRUCTION (exact):

       votes[t]      = r116_shared.build_panel_votes(tickers, window)[t]  # N cols, in {0,1/3,2/3,1}
       d[t]          = r116_shared.panel_disagreement(votes)[t]           # Bomberger disagreement
       D_MAX(N)      = sqrt(p(1-p)), p = floor(N/2)/N   # analytic max population std of
                                                          # N numbers in [0,1] -- a fixed,
                                                          # PRE-REGISTERED constant, never fit
       frac[t]       = clip(d[t] / D_MAX(N), 0, 1)
       discount[t]   = 1.0 - (1.0 - floor) * frac[t]     # NaN disagreement (panel warmup,
                                                          # see r116_shared.attach_to_btc's own
                                                          # documented 0.0-neutral fallback) ->
                                                          # d=0 -> discount=1.0, no invented number
       raw[t]        = v4_raw_desired(df)[t] * discount[t]   # frac*scale, UNCHANGED, discounted
       target[t]     = apply_deadband(raw)[t]                # v4's own 10% deadband, AFTER discount

   ``v4_raw_desired`` / ``apply_deadband`` are reused VERBATIM from
   ``experiments/r102_shared.py`` (read-only reuse of a prior round's
   already-tested, exact reproduction of ``KellyRegimeV3.prepare``'s
   internal ``frac`` / ``scale`` construction -- verified below against
   ``r116_shared.v4_targets``, the byte-identical benchmark computed by the
   ACTUAL registered strategy, before being trusted). Both functions are
   generic in ``df["close"]`` alone, so the SAME code path builds the BTC
   candidate and the ETH falsification candidate in section 5.

   BTC's OWN PANEL (primary construction): ``PANEL_BTC = ("BTC",) + UNIVERSE_6``
   -- 7 votes total (BTC's own + the six alts), per this round's own
   pre-registration in ``r116_shared.py``'s docstring, which frames the
   panel as "BTC + the six alts" and deliberately excludes ETH from BTC's
   own confirming panel.

   ETH'S PANEL (falsification construction, section 5): ``PANEL_ETH =
   UNIVERSE_8`` (ETH's own vote + BTC + the six alts) -- 8 votes total, i.e.
   ``UNIVERSE_8 minus ETH`` (BTC + the six alts, 7 members) as the
   CONFIRMING panel for ETH's own home vote, the natural symmetric flip
   given the data this project actually has (ETH did not have its own
   separate "six other instruments"; BTC is one of the "other instruments
   this project has 5-minute data for" once ETH, not BTC, is home).

   FLOOR GRID (pre-registered, before any real-data number below was
   computed): ``FLOOR_GRID = (0.3, 0.5, 0.7)`` -- reused verbatim from this
   project's own standing 3-cell grid (R-104/R-105/R-106), not widened.
   ``SELECTION_ORDER = (0.5, 0.3, 0.7)`` (grid centre preferred).

3. STEP-0 COLLINEARITY GATE (before any economic backtest number, per
   docs/ROUTINE.md step 3 / this round's own task brief): ``r_squared``
   (r116_shared's own helper) between the disagreement statistic and (a)
   v4's own vote fraction and (b) v4's own raw pre-deadband exposure, BOTH
   restricted to ``W_TRAIN`` (2020-04-01 -> 2021-12-31, the earliest window
   with a fully warmed panel -- all six alts start 2020-01-01, so their
   80-day anchors are not valid until ~2020-03-21). KILL (stop, report
   NEGATIVE, do not touch W_VAL/holdout) if EITHER R^2 >= 0.95 -- this
   project's own standing "flat rescale, not new information" artifact
   signature (R-73).

4. DECISION RULE (frozen here, before W_VAL is read for selection and
   BEFORE any holdout bar). Promote the primary (grid-centre-first)
   floor ONLY if ALL of:
     (P1) Step-0 gate above: NOT killed.
     (P2) Causal truncation probe (section 5's ``causal_probe``, on real
          BTC data) passes exactly.
     (P3) W_VAL SELECTION: a floor in FLOOR_GRID satisfies
          ``r116_shared.d3_pass`` (growth_diff > 0 AND dd_diff < 0, point
          estimates -- W_VAL is one year, 365 daily observations, too few
          for a bootstrap CI to be read as evidence rather than as a
          directional gate, exactly r63_shared's own documented reasoning
          for using d3_pass this way) against ``v4_targets`` on BTC spot,
          W_VAL. Primary = first floor in SELECTION_ORDER that passes; if
          none passes, this file still proceeds to section 5/6 to report
          the whole configuration honestly (per ROUTINE's parallelism rule
          that a branch report all evaluated configurations, not only the
          winner) but the round is scored NEGATIVE regardless of what
          follows -- default floor 0.5 is then used purely so sections 5/6
          have something concrete to report.
     (P4) PLATEAU: the primary floor's immediate grid neighbour(s) on W_VAL
          share the SAME SIGN of ``growth_diff`` as the primary cell.
     (P5) LEAD-TIME FALSIFICATION TEST (pre-registered NOW): does the panel
          disagreement statistic's own up-crossing of a fixed alarm
          threshold (``ALARM_FRAC = 0.5`` of ``D_MAX(7)``, i.e. "half of
          the theoretically maximum possible 7-way split", fixed a priori,
          never fit) LEAD BTC's own v4 anchor-vote downward transition
          (``r116_shared.nearest_transition(..., direction="down")``,
          identical rule to R-82's own gate) on at least 4 of the 6 named
          ``STRESS_EPISODES``, at the STRONG bar this project's INFO-axis
          gates use for an independent-signal claim (R-81/R-115): PASS
          requires BOTH lead >= 0 AND lead exceeds the 90th percentile of
          a 500-draw, 5-day-block circular-shift null
          (``r116_shared.block_bootstrap_shifts``) -- not merely the WEAKER
          "not worse than chance" bar R-82's own detector-vs-heuristic gate
          used, because THIS round's own claim (in r116_shared's docstring)
          is that cross-asset disagreement is a structurally NEW, possibly
          independently-arriving kind of information, the stronger claim
          R-81/R-115 were built to test.
          DISCLOSED CONSTRAINT, stated now rather than after seeing the
          number: the six-alt panel's own data starts 2020-01-01, so 2 of
          the 6 STRESS_EPISODES (2018-01-17, 2018-12-15) predate panel
          coverage entirely and are EXCLUDED as unmeasurable (matching
          R-115's own precedent for a coverage-limited panel), not counted
          as failures. The effective bar is therefore >=4 of the 4
          measurable (2020+) episodes, disclosed as a real tightening of
          the "4 of 6" language, not a relaxation.
     (P6) ETH SYMMETRY FALSIFICATION TEST (pre-registered NOW, this round's
          own decisive test per the task brief): the IDENTICAL construction
          and the SAME frozen primary floor, with ETH as home
          (PANEL_ETH = UNIVERSE_8, discount computed against ETH's own
          v4-style vote, D_MAX(8) instead of D_MAX(7)), against ETH's own
          unmodified v4-style vote (``v4_targets`` applied to ETH's own
          Coinbase spot series -- the same generic function, a different
          ``df``) on ETH's OWN W_VAL window. PASS requires
          ``r116_shared.d3_pass`` on this ETH cell too -- i.e. the SAME
          sign of improvement BTC's own W_VAL selection showed. FAIL
          (inversion or no effect) is read exactly as R-109/R-112's own
          kNN novelty brake was read: BTC-specific, not asset-general, a
          decisive kill for this construction regardless of what BTC's own
          numbers show.
   Any single failed clause among (P1)-(P6) is sufficient for NEGATIVE.
   Default is REJECT.

5. CAUSAL TRUNCATION PROBE: composed ``build_target`` (BTC construction),
   real BTC data, truncate-and-compare per ``docs/ROUTINE.md``'s own R-21
   warning (a too-good result is a bug report first). Additionally, an
   algebraic identity check: ``v4_raw_desired``/``apply_deadband``
   (imported from r102_shared) must reproduce ``r116_shared.v4_targets``
   (the ACTUAL registered strategy's own output) exactly, on both BTC and
   ETH data, before either is trusted as this file's control.

6. HOLDOUT CONSULT (run exactly once, after section 4's rule is frozen, per
   this round's own task brief -- reported as the round's primary evidence,
   never with the decision rule in section 4 relaxed afterwards): OOS
   (``OOS_START = 2023-01-01`` onward) vs ``buy_and_hold`` and vs
   ``v4_targets``, BTC, BOTH spot (@0.10%) and futures_5x, using
   ``r116_shared.compare`` for D1 (growth)/D2 (drawdown) paired-bootstrap
   CIs and ``d3_pass`` as a point-estimate corroboration; 0.40% taker fee
   tier (no sign reversal check); real BTC funding charged on the futures
   leg (Deribit BTC-PERP funding, which -- unlike the committed Binance
   file's 2020-2023 coverage -- spans the WHOLE holdout through 2026-08).
   ALSO, the ETH symmetry check repeated on ETH's OWN holdout period
   (2023-01-01 onward), same frozen floor, same comparison framework,
   against ETH's own unmodified v4-style vote -- spot only (no funding data
   exists for ETH in this repo, disclosed as a real, time-boxed
   limitation).

CONFIGURATIONS EVALUATED: counted programmatically via
``r116_shared.note_config()`` / ``config_count()`` on every backtest this
file runs (Step-0 uses none: it is a pure statistic on already-built
panel votes), printed at the end. This round's ledger trials count is this
number PLUS whatever the CONSERVATIVE branch (a separate session, same
round) reports, per ``docs/ROUTINE.md``'s parallelism rules.

----------------------------------------------------------------------
Run: python experiments/r116_novel_breadth_disagreement.py
(from the repo root, with the project venv active)
----------------------------------------------------------------------
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_funding_deribit  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import prefix_bars, run_period  # noqa: E402

from experiments.r116_shared import (  # noqa: E402
    UNIVERSE_6,
    UNIVERSE_8,
    W_TRAIN,
    W_VAL,
    W_HOLD,
    OOS_START,
    STRESS_EPISODES,
    agree_frac,  # noqa: F401  (available, unused -- see section 5 note)
    anchor_majority,
    attach_to_btc,
    block_bootstrap_shifts,
    build_panel_votes,
    check_causality,  # noqa: F401  (aligned-dict probe; this file's own causal_probe is used instead)
    compare,
    d1_pass,
    d2_pass,
    d3_pass,
    episode_window,
    excludes_zero,  # noqa: F401
    further_work,  # noqa: F401
    load_universe,
    nearest_transition,
    note_config,
    config_count,
    panel_disagreement,
    r_squared,
    v4_targets,
)

# Read-only reuse of a PRIOR round's already-tested, exact reproduction of
# kelly_regime_v4's internal frac/scale construction (verified against the
# real registered strategy below, before being trusted). Not r116_shared,
# not touched.
from experiments.r102_shared import (  # noqa: E402
    apply_deadband,
    fee_at,
    v4_raw_desired,
    v4_target,
)

BARS_PER_DAY = 288
V4_WARMUP_BARS = 80 * BARS_PER_DAY + 10
DATA_DIR = ROOT / "data"

PANEL_BTC = ("BTC",) + UNIVERSE_6          # 7 votes total (BTC's own + six alts)
PANEL_ETH = UNIVERSE_8                     # 8 votes total (ETH's own + BTC + six alts)

FLOOR_GRID = (0.3, 0.5, 0.7)
SELECTION_ORDER = (0.5, 0.3, 0.7)
R2_KILL_THRESH = 0.95
ALARM_FRAC = 0.5           # fraction of D_MAX(7) that counts as "alarm" for the lead-time gate
N_NULL_DRAWS = 500
NULL_BLOCK_DAYS = 5
NULL_SEED = 116
EPISODE_WINDOW_DAYS = 60
PANEL_COVERAGE_START = pd.Timestamp("2020-01-01", tz="UTC")  # all six alts' own first bar

FUNDING_HOLDOUT_END = None  # Deribit funding spans the whole holdout; no truncation needed

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def assert_no_holdout(df: pd.DataFrame, label: str) -> None:
    if len(df) == 0:
        return
    if df.index.max() >= pd.Timestamp(OOS_START, tz="UTC"):
        raise AssertionError(f"{label}: holdout bar read, max={df.index.max()}")


# ================================================================== (1)
# Panel disagreement -> discount, on an arbitrary home asset's own df.
# ==================================================================

def analytic_dmax(n: int) -> float:
    """Analytic max of the POPULATION std (ddof=0, matching
    r116_shared.panel_disagreement) of n numbers each bounded in [0, 1]:
    attained by a floor(n/2)/ceil(n/2) split at the two extremes."""
    k = n // 2
    p = k / n
    return float(np.sqrt(p * (1.0 - p)))


D_MAX_BTC = analytic_dmax(len(PANEL_BTC))   # n=7 -> sqrt(12/49) ~= 0.4949
D_MAX_ETH = analytic_dmax(len(PANEL_ETH))   # n=8 -> 0.5 exactly

_DISAGREEMENT_CACHE: dict[tuple, pd.Series] = {}


def disagreement_series(tickers: tuple[str, ...], end: str) -> pd.Series:
    """Panel disagreement over ``tickers``, from each ticker's own first
    bar through ``end`` (a date string) -- causal by construction (see
    r116_shared.build_panel_votes/panel_disagreement's own docstrings).
    Memoized: many backtests in this file share the same (tickers, end)
    pair (every floor in a sweep re-reads the identical disagreement path)."""
    key = (tickers, end)
    if key not in _DISAGREEMENT_CACHE:
        votes = build_panel_votes(tickers=tickers, window=(None, end))
        _DISAGREEMENT_CACHE[key] = panel_disagreement(votes)
    return _DISAGREEMENT_CACHE[key]


def disagreement_for_home(df: pd.DataFrame, tickers: tuple[str, ...]) -> pd.Series:
    """The panel disagreement statistic, reindexed causally onto ``df``'s
    own index (``r116_shared.attach_to_btc`` -- asset-agnostic despite the
    name; it only ever forward-fills a past panel value onto ``df``'s own
    grid)."""
    end = df.index.max().strftime("%Y-%m-%d")
    d = disagreement_series(tickers, end)
    return attach_to_btc(df, d, "disagreement")["disagreement"]


def discount_from_disagreement(d: np.ndarray, floor: float, dmax: float) -> np.ndarray:
    """Bounded [floor, 1.0], monotonic NON-INCREASING (never scales UP
    exposure) clipped-linear map. NaN/undefined disagreement -> 1.0 (no
    brake); in practice this never fires because
    r116_shared.attach_to_btc's own warmup fallback is 0.0, not NaN."""
    d = np.asarray(d, dtype=float)
    frac = np.clip(d / dmax, 0.0, 1.0)
    disc = 1.0 - (1.0 - floor) * frac
    return np.where(np.isfinite(d), disc, 1.0)


def make_build_target(tickers: tuple[str, ...], floor: float, dmax: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        d = disagreement_for_home(df, tickers).to_numpy()
        disc = discount_from_disagreement(d, floor, dmax)
        raw = v4_raw_desired(df) * disc
        return apply_deadband(raw)
    _build.__name__ = f"panel_disagreement_brake_{'_'.join(tickers)}_floor{floor:g}"
    return _build


# ================================================================== (2)
# TargetStrategy -- duplicated per-round, per this repo's own convention
# (experiments/r102_shared.py:510's pattern).
# ==================================================================

class TargetStrategy(Strategy):
    """Wrap a pure ``build_target(df) -> np.ndarray`` as a runnable strategy."""

    name = "r116_novel_control"
    warmup = V4_WARMUP_BARS

    def __init__(self, build_target, name: str = "r116_novel_control",
                 warmup: int | None = None) -> None:
        self._build = build_target
        self.name = name
        if warmup is not None:
            self.warmup = warmup

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["target"] = np.asarray(self._build(df), dtype=float)
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


def v4_targets_fn(df: pd.DataFrame) -> np.ndarray:
    return v4_targets(df).to_numpy()


def run_leg(build_fn, df: pd.DataFrame, start, end, market: MarketSpec,
            label: str, warmup: int = V4_WARMUP_BARS, balance: float = 1_000.0):
    strat = TargetStrategy(build_fn, name=label, warmup=warmup)
    res = run_period(strat, df, start, end, market=market, start_balance=balance)
    note_config()
    return res


def run_leg_funded(build_fn, df: pd.DataFrame, start, end, market: MarketSpec,
                    label: str, funding: pd.Series | None = None,
                    warmup: int = V4_WARMUP_BARS, balance: float = 1_000.0):
    """Same as run_leg but supports charging funding -- run_period itself has
    no funding parameter, so this reproduces its warm-prefix/trim logic by
    hand (identical to r106_conservative_disagreement_brake.py's own
    ``holdout_run``)."""
    strat = TargetStrategy(build_fn, name=label, warmup=warmup)
    lo = 0 if start is None else int(df.index.searchsorted(pd.Timestamp(start, tz="UTC")))
    hi = len(df) if end is None else int(df.index.searchsorted(pd.Timestamp(end, tz="UTC"), side="right"))
    prefix = prefix_bars(df, lo, strat.warmup)
    frame = df.iloc[lo - prefix: hi]
    res = run_backtest(strat, frame, market, balance, trade_start=prefix, funding=funding)
    note_config()
    if prefix:
        res = replace(res, equity=res.equity.iloc[prefix:])
    return res


# ================================================================== (3)
# Causal truncation probe + algebraic identity checks.
# ==================================================================

def causal_probe(build_fn, df: pd.DataFrame, cut_from_end: int = 5_000,
                  probe_len: int = 100_000) -> bool:
    """Truncate-and-compare: rows strictly before the cut must be bit-for-bit
    identical whether or not later rows exist. Run on a moderate real-BTC
    window (not the full multi-year series) purely for speed -- the
    construction's causality does not depend on series length."""
    window = df.iloc[-probe_len:].copy()
    cut = len(window) - cut_from_end
    full = np.asarray(build_fn(window), dtype=float)
    trunc_df = window.iloc[:cut + 2_000].copy()
    trunc = np.asarray(build_fn(trunc_df), dtype=float)
    m = min(cut, len(trunc))
    a = np.nan_to_num(full[:m])
    b = np.nan_to_num(trunc[:m])
    return bool(np.allclose(a, b, atol=1e-9, rtol=0.0))


def identity_check(df: pd.DataFrame, label: str) -> bool:
    """v4_raw_desired/apply_deadband (r102_shared) must reproduce
    v4_targets (the ACTUAL registered kelly_regime_v4, r116_shared) exactly."""
    a = v4_target(df)
    b = v4_targets(df).to_numpy()
    ok = bool(np.allclose(a, b, equal_nan=True))
    print(f"  {label}: v4_raw_desired/apply_deadband == v4_targets exactly? {ok}")
    return ok


# ================================================================== (4)
# Lead-time falsification gate.
# ==================================================================

def alarm_crossings(d: pd.Series, window: pd.DatetimeIndex, thresh: float) -> np.ndarray:
    vals = d.reindex(window).to_numpy()
    above = np.nan_to_num(vals, nan=-1.0) >= thresh
    cross = np.zeros(len(vals), dtype=bool)
    cross[1:] = above[1:] & ~above[:-1]
    cross[0] = bool(above[0])
    return cross


def nearest_alarm_crossing(d: pd.Series, window: pd.DatetimeIndex,
                            onset: pd.Timestamp, thresh: float) -> pd.Timestamp | None:
    cross = alarm_crossings(d, window, thresh)
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def null_leads_for_episode(d: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                            flip_time: pd.Timestamp, thresh: float) -> np.ndarray:
    local = d.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=NULL_BLOCK_DAYS,
                                     n_draws=N_NULL_DRAWS, seed=NULL_SEED)
    leads = np.full(N_NULL_DRAWS, np.nan)
    for k, shift in enumerate(shifts):
        shifted = local[shift]
        above = np.nan_to_num(shifted, nan=-1.0) >= thresh
        cross = np.zeros(n_bars, dtype=bool)
        cross[1:] = above[1:] & ~above[:-1]
        cross[0] = bool(above[0])
        idx = np.where(cross)[0]
        if len(idx) == 0:
            continue
        times = window[idx]
        deltas = np.abs((times - onset).to_numpy())
        cross_time = times[int(np.argmin(deltas))]
        leads[k] = (flip_time - cross_time).total_seconds() / 86400.0
    return leads


def lead_time_gate(btc: pd.DataFrame) -> list[dict]:
    majority = anchor_majority(btc)
    d = disagreement_for_home(btc, PANEL_BTC)
    thresh = ALARM_FRAC * D_MAX_BTC

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset = pd.Timestamp(onset_str, tz="UTC")
        if onset < PANEL_COVERAGE_START:
            print(f"  [{label}] onset={onset_str}: predates panel coverage "
                  f"({PANEL_COVERAGE_START.date()}) -- EXCLUDED, not measured.")
            results.append(dict(label=label, onset=onset_str, measurable=False, pass_ep=False))
            continue
        onset, window = episode_window(btc, onset_str, EPISODE_WINDOW_DAYS)
        if len(window) == 0:
            print(f"  [{label}] onset={onset_str}: zero bars in window -- EXCLUDED.")
            results.append(dict(label=label, onset=onset_str, measurable=False, pass_ep=False))
            continue

        flip_time = nearest_transition(majority, window, onset, direction="down")
        cross_time = nearest_alarm_crossing(d, window, onset, thresh)
        if flip_time is None or cross_time is None:
            reason = "no v4 downward vote transition" if flip_time is None else "no alarm crossing"
            print(f"  [{label}] onset={onset_str}: {reason} found in window -- UNMATCHED.")
            results.append(dict(label=label, onset=onset_str, measurable=True, matched=False, pass_ep=False))
            continue

        lead = (flip_time - cross_time).total_seconds() / 86400.0
        null_leads = null_leads_for_episode(d, window, onset, flip_time, thresh)
        valid = null_leads[~np.isnan(null_leads)]
        null_p90 = float(np.percentile(valid, 90)) if len(valid) else float("nan")
        pass_ep = bool(lead >= 0 and np.isfinite(null_p90) and lead >= null_p90)

        print(f"  [{label}] onset={onset_str}  v4 flip={flip_time}  alarm cross={cross_time}")
        print(f"    LEAD={lead:+.2f}d  null_p90={null_p90:+.2f}d "
              f"(valid draws {len(valid)}/{N_NULL_DRAWS})  PASS={pass_ep}")
        results.append(dict(label=label, onset=onset_str, measurable=True, matched=True,
                             lead=lead, null_p90=null_p90, pass_ep=pass_ep))
    return results


# ================================================================== (5)
# Step-0 gate + train sweep + W_VAL selection.
# ==================================================================

def step0_gate(btc: pd.DataFrame) -> dict:
    mask = (btc.index >= pd.Timestamp(W_TRAIN[0], tz="UTC")) & \
           (btc.index <= pd.Timestamp(W_TRAIN[1], tz="UTC"))
    d = disagreement_for_home(btc, PANEL_BTC).to_numpy()
    vote = anchor_majority(btc).to_numpy()
    raw = v4_raw_desired(btc)

    r2_vote = r_squared(pd.Series(d[mask]), pd.Series(vote[mask]))
    r2_raw = r_squared(pd.Series(d[mask]), pd.Series(raw[mask]))
    killed = bool((np.isfinite(r2_vote) and r2_vote >= R2_KILL_THRESH) or
                  (np.isfinite(r2_raw) and r2_raw >= R2_KILL_THRESH))
    return dict(r2_vs_vote=r2_vote, r2_vs_raw_exposure=r2_raw, killed=killed,
                disagreement_mean=float(np.nanmean(d[mask])),
                disagreement_std=float(np.nanstd(d[mask])))


def train_sweep(btc: pd.DataFrame) -> list[dict]:
    ctrl = run_leg(v4_targets_fn, btc, W_TRAIN[0], W_TRAIN[1], SPOT, "kelly_regime_v4")
    ctrl_m = compute_metrics(ctrl)
    rows = []
    for floor in FLOOR_GRID:
        build_fn = make_build_target(PANEL_BTC, floor, D_MAX_BTC)
        cand = run_leg(build_fn, btc, W_TRAIN[0], W_TRAIN[1], SPOT, f"floor{floor:g}")
        cand_m = compute_metrics(cand)
        row = compare(cand.equity, ctrl.equity)
        row.update(floor=floor, cand_sharpe=cand_m.sharpe, ctrl_sharpe=ctrl_m.sharpe,
                   d_sharpe=cand_m.sharpe - ctrl_m.sharpe)
        rows.append(row)
        print(f"  floor={floor:g}  cand_final=${cand_m.final_balance:>9,.0f} (Sh {cand_m.sharpe:+.2f})  "
              f"ctrl_final=${ctrl_m.final_balance:>9,.0f} (Sh {ctrl_m.sharpe:+.2f})  "
              f"growth_diff={row['growth_diff']:+.4f} [{row['growth_lo']:+.4f},{row['growth_hi']:+.4f}]  "
              f"dd_diff={row['dd_diff']:+.4f} [{row['dd_lo']:+.4f},{row['dd_hi']:+.4f}]")
    return rows


def val_select(btc: pd.DataFrame) -> tuple[float | None, list[dict]]:
    ctrl = run_leg(v4_targets_fn, btc, W_VAL[0], W_VAL[1], SPOT, "kelly_regime_v4")
    rows = []
    for floor in FLOOR_GRID:
        build_fn = make_build_target(PANEL_BTC, floor, D_MAX_BTC)
        cand = run_leg(build_fn, btc, W_VAL[0], W_VAL[1], SPOT, f"floor{floor:g}")
        row = compare(cand.equity, ctrl.equity)
        row["floor"] = floor
        row["passes_d3"] = d3_pass(row)
        rows.append(row)
        print(f"  floor={floor:g}  growth_diff={row['growth_diff']:+.4f}  dd_diff={row['dd_diff']:+.4f}  "
              f"d3_pass={row['passes_d3']}")

    by_floor = {r["floor"]: r for r in rows}
    primary = None
    for f in SELECTION_ORDER:
        if by_floor[f]["passes_d3"]:
            primary = f
            break
    return primary, rows


def plateau_check(primary: float, val_rows: list[dict]) -> bool:
    order = sorted(FLOOR_GRID)
    idx = order.index(primary)
    neighbours = [order[i] for i in (idx - 1, idx + 1) if 0 <= i < len(order) and i != idx]
    by_floor = {r["floor"]: r for r in val_rows}
    prim_sign = np.sign(by_floor[primary]["growth_diff"])
    ok = True
    for nb in neighbours:
        same = np.sign(by_floor[nb]["growth_diff"]) == prim_sign
        print(f"  neighbour floor={nb:g}: growth_diff={by_floor[nb]['growth_diff']:+.4f}  "
              f"same_sign_as_primary={same}")
        ok = ok and bool(same)
    return ok


# ================================================================== (6)
# ETH symmetry falsification check (pre-holdout, W_VAL).
# ==================================================================

def eth_val_check(eth: pd.DataFrame, primary_floor: float) -> dict:
    ctrl = run_leg(v4_targets_fn, eth, W_VAL[0], W_VAL[1], SPOT, "eth_v4_style")
    build_fn = make_build_target(PANEL_ETH, primary_floor, D_MAX_ETH)
    cand = run_leg(build_fn, eth, W_VAL[0], W_VAL[1], SPOT, f"eth_floor{primary_floor:g}")
    row = compare(cand.equity, ctrl.equity)
    row["passes_d3"] = d3_pass(row)
    print(f"  ETH W_VAL, floor={primary_floor:g}: growth_diff={row['growth_diff']:+.4f}  "
          f"dd_diff={row['dd_diff']:+.4f}  d3_pass={row['passes_d3']}")
    return row


# ================================================================== (7)
# Holdout consult.
# ==================================================================

def holdout_leg_group(build_fn, ctrl_fn, df_full: pd.DataFrame, market: MarketSpec,
                       label: str, funding: pd.Series | None = None) -> dict:
    cand = run_leg_funded(build_fn, df_full, OOS_START, None, market, f"{label}_cand", funding=funding)
    ctrl = run_leg_funded(ctrl_fn, df_full, OOS_START, None, market, f"{label}_ctrl", funding=funding)
    row = compare(cand.equity, ctrl.equity)
    cand_m, ctrl_m = compute_metrics(cand), compute_metrics(ctrl)
    row.update(cand_sharpe=cand_m.sharpe, ctrl_sharpe=ctrl_m.sharpe,
               d_sharpe=cand_m.sharpe - ctrl_m.sharpe,
               cand_final=cand_m.final_balance, ctrl_final=ctrl_m.final_balance,
               d1=d1_pass(row), d2=d2_pass(row), d3=d3_pass(row))
    return row


def run_registered_leg(name: str, df: pd.DataFrame, start, end, market: MarketSpec,
                        balance: float = 1_000.0):
    strat = get_strategy(name)
    res = run_period(strat, df, start, end, market=market, start_balance=balance)
    note_config()
    return res


def holdout_vs_hold(build_fn, df_full: pd.DataFrame, market: MarketSpec, label: str) -> dict:
    cand = run_leg_funded(build_fn, df_full, OOS_START, None, market, f"{label}_cand")
    hold = run_registered_leg("buy_and_hold", df_full, OOS_START, None, market)
    row = compare(cand.equity, hold.equity)
    row["d1_vs_hold"] = d1_pass(row)
    return row


def run_holdout(btc_full: pd.DataFrame, eth_full: pd.DataFrame, primary_floor: float) -> dict:
    hr("SECTION 6 -- HOLDOUT CONSULT (run once, decision rule already frozen in section 4)")
    build_btc = make_build_target(PANEL_BTC, primary_floor, D_MAX_BTC)
    build_eth = make_build_target(PANEL_ETH, primary_floor, D_MAX_ETH)

    print(f"\nfull BTC dataset: {btc_full.index[0]} -> {btc_full.index[-1]}  "
          f"(holdout portion: {OOS_START} onward)")
    print(f"full ETH dataset: {eth_full.index[0]} -> {eth_full.index[-1]}")

    hr("6a -- BTC holdout vs kelly_regime_v4 and vs buy_and_hold, both markets")
    btc_rows = {}
    for market in (SPOT, FUTURES):
        row = holdout_leg_group(build_btc, v4_targets_fn, btc_full, market, f"btc_{market.name}")
        hold_row = holdout_vs_hold(build_btc, btc_full, market, f"btc_{market.name}")
        row["d1_vs_hold"] = hold_row["d1_vs_hold"]
        row["growth_diff_vs_hold"] = hold_row["growth_diff"]
        btc_rows[market.name] = row
        print(f"  {market.name:>12s}  cand_final=${row['cand_final']:>10,.0f} (Sh {row['cand_sharpe']:+.2f})  "
              f"ctrl_final=${row['ctrl_final']:>10,.0f} (Sh {row['ctrl_sharpe']:+.2f})  "
              f"d_sharpe={row['d_sharpe']:+.3f}")
        print(f"    growth_diff={row['growth_diff']:+.4f} [{row['growth_lo']:+.4f},{row['growth_hi']:+.4f}]  "
              f"dd_diff={row['dd_diff']:+.4f} [{row['dd_lo']:+.4f},{row['dd_hi']:+.4f}]")
        print(f"    D1(growth)={row['d1']}  D2(drawdown)={row['d2']}  D3(point)={row['d3']}  "
              f"beats_buy_and_hold(D1)={row['d1_vs_hold']}  growth_vs_hold={row['growth_diff_vs_hold']:+.4f}")

    hr("6b -- 0.40% taker fee tier, BTC holdout, both markets")
    fee_rows = {}
    for market in (SPOT, FUTURES):
        fee_mkt = fee_at(market, 0.004)
        row = holdout_leg_group(build_btc, v4_targets_fn, btc_full, fee_mkt, f"btc_fee40_{market.name}")
        base = btc_rows[market.name]
        no_reversal = not (np.sign(row["d_sharpe"]) != np.sign(base["d_sharpe"])
                            and row["d_sharpe"] != 0 and base["d_sharpe"] != 0)
        row["no_reversal"] = no_reversal
        fee_rows[market.name] = row
        print(f"  {market.name:>12s}  @0.10-0.05% d_sharpe={base['d_sharpe']:+.3f}   "
              f"@0.40% d_sharpe={row['d_sharpe']:+.3f}   no_reversal={no_reversal}")

    hr("6c -- real BTC funding charged (Deribit BTC-PERP, spans the whole holdout), futures only")
    funding = load_funding_deribit(DATA_DIR)
    funding_row = None
    if funding is None:
        print("  no Deribit funding data committed -- skipped")
    else:
        free = btc_rows["futures_5x"]
        paid = holdout_leg_group(build_btc, v4_targets_fn, btc_full, FUTURES, "btc_funded", funding=funding)
        no_reversal = bool(np.sign(free["d_sharpe"]) == np.sign(paid["d_sharpe"])
                            or free["d_sharpe"] == 0 or paid["d_sharpe"] == 0)
        funding_row = dict(d_sharpe_free=free["d_sharpe"], d_sharpe_paid=paid["d_sharpe"],
                            cand_final_paid=paid["cand_final"], ctrl_final_paid=paid["ctrl_final"],
                            no_reversal=no_reversal)
        print(f"  funding-free d_sharpe={free['d_sharpe']:+.3f}   funding-charged d_sharpe={paid['d_sharpe']:+.3f}   "
              f"no_reversal={no_reversal}")

    hr("6d -- ETH symmetry check, ETH's own holdout period, spot only (no ETH funding data committed)")
    eth_row = holdout_leg_group(build_eth, v4_targets_fn, eth_full, SPOT, "eth_holdout")
    eth_hold_row = holdout_vs_hold(build_eth, eth_full, SPOT, "eth_holdout")
    eth_row["d1_vs_hold"] = eth_hold_row["d1_vs_hold"]
    print(f"  ETH spot  cand_final=${eth_row['cand_final']:>10,.0f} (Sh {eth_row['cand_sharpe']:+.2f})  "
          f"ctrl_final=${eth_row['ctrl_final']:>10,.0f} (Sh {eth_row['ctrl_sharpe']:+.2f})  "
          f"d_sharpe={eth_row['d_sharpe']:+.3f}")
    print(f"    D1={eth_row['d1']}  D2={eth_row['d2']}  D3={eth_row['d3']}  "
          f"beats_buy_and_hold(D1)={eth_row['d1_vs_hold']}")
    print(f"    same_sign_as_btc_spot={np.sign(eth_row['d_sharpe']) == np.sign(btc_rows['spot']['d_sharpe'])}")

    return dict(btc_rows=btc_rows, fee_rows=fee_rows, funding_row=funding_row, eth_row=eth_row)


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()

    hr("R-116 NOVEL: PanelDisagreementBrakeKellyV4 -- Bomberger (1996) cross-ASSET "
       "disagreement (BTC + six-alt / ETH + BTC + six-alt panels of v4's own vote) "
       "as a bounded, never-increase-only brake")

    btc = load_universe(("BTC",))["BTC"]
    eth = load_universe(("ETH",))["ETH"]
    btc_pretrunc = btc[btc.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    eth_pretrunc = eth[eth.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    assert_no_holdout(btc_pretrunc, "btc_pretrunc")
    assert_no_holdout(eth_pretrunc, "eth_pretrunc")
    print(f"\nBTC: {len(btc_pretrunc):,} bars < {OOS_START}  "
          f"{btc_pretrunc.index[0]} -> {btc_pretrunc.index[-1]}")
    print(f"ETH: {len(eth_pretrunc):,} bars < {OOS_START}  "
          f"{eth_pretrunc.index[0]} -> {eth_pretrunc.index[-1]}")
    print(f"D_MAX(BTC panel, n={len(PANEL_BTC)}) = {D_MAX_BTC:.4f}   "
          f"D_MAX(ETH panel, n={len(PANEL_ETH)}) = {D_MAX_ETH:.4f}")

    hr("PRE-FLIGHT IDENTITY CHECKS")
    id_ok = identity_check(btc_pretrunc, "BTC") and identity_check(eth_pretrunc, "ETH")
    print(f"identity checks pass: {id_ok}")

    hr("STEP 0 -- COLLINEARITY GATE (R^2 vs v4's own vote and raw exposure, W_TRAIN)")
    step0 = step0_gate(btc_pretrunc)
    print(f"  R^2(disagreement, v4_vote_frac)   = {step0['r2_vs_vote']:.4f}")
    print(f"  R^2(disagreement, v4_raw_exposure) = {step0['r2_vs_raw_exposure']:.4f}")
    print(f"  disagreement mean={step0['disagreement_mean']:.4f} std={step0['disagreement_std']:.4f}")
    print(f"  KILL (either R^2 >= {R2_KILL_THRESH}): {step0['killed']}")

    hr("TRAIN SWEEP (W_TRAIN, iterate freely, 3 floors)")
    train_rows = train_sweep(btc_pretrunc)

    hr("W_VAL SELECTION (3 floors, d3_pass = growth_diff>0 AND dd_diff<0, point estimates)")
    primary, val_rows = val_select(btc_pretrunc)
    if primary is None:
        print("\n  NO floor in the pre-registered grid passes d3_pass on W_VAL.")
        primary_for_report = 0.5
    else:
        primary_for_report = primary
    print(f"\n  PRIMARY floor selected: "
          f"{primary if primary is not None else f'{primary_for_report:g} (fallback -- none qualified)'}")

    hr("PLATEAU CHECK (primary floor's grid neighbour(s), W_VAL)")
    plateau_ok = plateau_check(primary_for_report, val_rows)
    print(f"\n  plateau (neighbour(s) share primary's sign): {plateau_ok}")

    hr("CAUSAL TRUNCATION PROBE (composed BTC build_target, real BTC data)")
    build_primary = make_build_target(PANEL_BTC, primary_for_report, D_MAX_BTC)
    probe_ok = causal_probe(build_primary, btc_pretrunc)
    print(f"  PASS: {probe_ok}")

    hr("LEAD-TIME FALSIFICATION GATE (STRESS_EPISODES, alarm up-crossing vs v4's own vote flip)")
    print(f"alarm threshold = {ALARM_FRAC:g} * D_MAX(7) = {ALARM_FRAC * D_MAX_BTC:.4f}")
    lead_results = lead_time_gate(btc_pretrunc)
    measurable = [r for r in lead_results if r.get("measurable")]
    n_pass_lead = sum(1 for r in lead_results if r.get("pass_ep"))
    print(f"\n  episodes passing: {n_pass_lead}/{len(measurable)} measurable "
          f"({len(lead_results) - len(measurable)} excluded, pre-2020 panel coverage)")
    lead_gate_pass = n_pass_lead >= 4

    hr("ETH SYMMETRY FALSIFICATION TEST (pre-holdout, ETH's own W_VAL)")
    eth_val_row = eth_val_check(eth_pretrunc, primary_for_report)
    eth_symmetry_pass = bool(eth_val_row["passes_d3"])

    max_ts = max(btc_pretrunc.index.max(), eth_pretrunc.index.max())
    print(f"\nmax timestamp read so far (must be < {OOS_START}): {max_ts}  "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")

    # ==================================================== FROZEN DECISION
    hr("FROZEN DECISION (section 4 of this file's own pre-registration, evaluated NOW, "
       "BEFORE any holdout bar below is read)")
    p1 = not step0["killed"]
    p2 = probe_ok
    p3 = primary is not None
    p4 = plateau_ok
    p5 = lead_gate_pass
    p6 = eth_symmetry_pass
    pre_holdout_pass = p1 and p2 and p3 and p4 and p5 and p6
    print(f"  P1 (Step-0 not killed)        = {p1}")
    print(f"  P2 (causal probe)             = {p2}")
    print(f"  P3 (a floor passes W_VAL d3)  = {p3}")
    print(f"  P4 (plateau)                  = {p4}")
    print(f"  P5 (lead-time falsification)  = {p5}")
    print(f"  P6 (ETH symmetry, pre-holdout)= {p6}")
    print(f"\n  ALL PRE-HOLDOUT GATES PASS: {pre_holdout_pass}")
    print("  Per section 4: any single failed clause is sufficient for NEGATIVE. The holdout "
          "below is read regardless (this round's own task brief requires it as evidence), "
          "but promotion requires the FULL set (P1-P6) plus the holdout's own bar in section 6.")

    # ==================================================== HOLDOUT
    holdout = run_holdout(btc, eth, primary_for_report)

    hr("SUMMARY / FINAL VERDICT")
    btc_spot, btc_fut = holdout["btc_rows"]["spot"], holdout["btc_rows"]["futures_5x"]
    fee_spot_ok = holdout["fee_rows"]["spot"]["no_reversal"]
    fee_fut_ok = holdout["fee_rows"]["futures_5x"]["no_reversal"]
    funding_ok = holdout["funding_row"]["no_reversal"] if holdout["funding_row"] else True
    eth_hold_row = holdout["eth_row"]
    eth_holdout_same_sign = bool(np.sign(eth_hold_row["d_sharpe"]) == np.sign(btc_spot["d_sharpe"]))

    p7_holdout_beats_v4 = bool((btc_spot["d1"] or btc_spot["d2"]) and (btc_fut["d1"] or btc_fut["d2"]))
    p8_holdout_beats_hold = bool(btc_spot["d1_vs_hold"] and btc_fut["d1_vs_hold"])
    p9_sharpe_floor = bool(abs(btc_spot["d_sharpe"]) > 0.2 or abs(btc_fut["d_sharpe"]) > 0.2
                            or btc_spot["d2"] or btc_fut["d2"])
    p10_fee_survives = bool(fee_spot_ok and fee_fut_ok)
    p11_funding_survives = bool(funding_ok)
    p12_eth_holdout_confirms = eth_holdout_same_sign

    all_pass = (pre_holdout_pass and p7_holdout_beats_v4 and p8_holdout_beats_hold and
                p9_sharpe_floor and p10_fee_survives and p11_funding_survives and p12_eth_holdout_confirms)
    verdict = "PROMOTE" if all_pass else "NEGATIVE"

    print(f"  P7  (holdout beats v4, D1 or D2, both markets)   = {p7_holdout_beats_v4}")
    print(f"  P8  (holdout beats buy_and_hold, both markets)   = {p8_holdout_beats_hold}")
    print(f"  P9  (|d_sharpe| > 0.2 noise floor OR D2 drawdown)= {p9_sharpe_floor}")
    print(f"  P10 (0.40% fee tier, no sign reversal)           = {p10_fee_survives}")
    print(f"  P11 (funding-charged, no sign reversal)          = {p11_funding_survives}")
    print(f"  P12 (ETH holdout same sign as BTC holdout)       = {p12_eth_holdout_confirms}")
    print(f"\n  VERDICT: {verdict}")
    if not all_pass:
        failed = [name for name, ok in (
            ("P1-P6 pre-holdout", pre_holdout_pass), ("P7", p7_holdout_beats_v4),
            ("P8", p8_holdout_beats_hold), ("P9", p9_sharpe_floor),
            ("P10", p10_fee_survives), ("P11", p11_funding_survives), ("P12", p12_eth_holdout_confirms),
        ) if not ok]
        print(f"  Reason(s): {', '.join(failed)}")

    print(f"\nconfigurations evaluated (this file, note_config() counter): {config_count()}")
    print(f"[{time.time() - t0:.0f}s]")

    return dict(step0=step0, train_rows=train_rows, val_rows=val_rows, primary=primary,
                primary_for_report=primary_for_report, plateau_ok=plateau_ok, probe_ok=probe_ok,
                lead_results=lead_results, lead_gate_pass=lead_gate_pass,
                eth_val_row=eth_val_row, eth_symmetry_pass=eth_symmetry_pass,
                pre_holdout_pass=pre_holdout_pass, holdout=holdout, verdict=verdict,
                n_configs=config_count())


if __name__ == "__main__":
    main()
