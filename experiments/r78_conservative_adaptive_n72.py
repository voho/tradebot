#!/usr/bin/env python
"""R-78 (conservative branch): R-77's own regime-adaptive execution-urgency
mechanism, re-scoped so ``n_base`` genuinely reaches into the N>=72 danger
zone R-56 found.

Not registered: lives under ``experiments/`` so it is not auto-discovered,
per ROUTINE.md step 5. Does not modify ``kelly_regime_v4.py``,
``kelly_regime_v3.py``, ``kelly_regime.py``, ``engine.py`` or ``broker.py``
-- all reused read-only. Does not modify either R-56 file or either R-77
file (``experiments/r77_conservative_narrowcap_execution.py``,
``experiments/r77_novel_execution_regime_adaptive.py``) -- the latter is
imported read-only as a library (see IMPORTS below), never edited. Zero
overlap with the sibling parallel branch's file
(``experiments/r78_novel_execution_directional_asymmetry.py``, a separate,
independent agent's file) -- not read, not touched, per this round's
explicit instruction.

THE IDEA, ONE SENTENCE
-----------------------
R-77's regime-adaptive execution-urgency mechanism (patience shrinks
continuously as a causal stress ratio rises, collapsing to near-immediate
taker fills for de-risking orders under stress) was tested twice this
project (R-77 conservative's fixed-N ablation up to N=24, R-77 novel's
adaptive arm with ``n_base`` capped at 24) and both failed cleanly, but
neither branch's ``n_base`` ceiling of 24 ever reached the N>=72 zone that
R-56's *original* N in {1,...,288} sweep is the one place this project has
found a genuine, systematic crash-transition-lag danger ("futures
inner-validation reverses sign at N>=72 ... several 25-71 bar delays
during the 2021-22 trend that net worse than earlier forced fallback
would have been") -- so this round asks the one question R-77's own
"next step" note named explicitly and no round has yet tested: does
adaptive urgency actually help, specifically inside the region where fixed
patience is known to be dangerous, once the adaptive mechanism's own
``n_base`` is finally allowed to reach that region?

CONSTRAINT ATTACKED: COST (costs that scale with the signal), same
constraint as R-56 and both R-77 branches, via the same "change HOW an
already-decided trade fills, not WHEN or how MUCH" mechanism established by
R-56 and reused unmodified by R-77 novel.

NOT A DUPLICATE OF (stated precisely):
- R-56 (``experiments/kelly_regime_exec_limit_conservative.py`` /
  ``_novel.py``): tested FIXED patience N in {1,2,3,6,12,24,72,288}. This
  round tests ADAPTIVE patience whose ceiling (``n_base``) spans the exact
  same N>=72 region R-56 flagged as dangerous -- R-56 never tested an
  adaptive mechanism at all.
- R-77 conservative (``experiments/r77_conservative_narrowcap_execution.py``,
  B-24): tested FIXED patience N in {2,3,6,12,24} only -- by design, a
  pre-registered subset that deliberately stayed inside the "safe"
  residual and never approached N>=72.
- R-77 novel (``experiments/r77_novel_execution_regime_adaptive.py``):
  introduced the exact adaptive-urgency formula this round reuses, but its
  own pre-registered grid capped ``n_base`` at 24 (Stage A: n_base in
  {12,24}) and its own thesis-test-4 fixed-N comparison set was N in
  {3,6,9,12,18,24} -- also never entering N>=72. This round's precise
  difference from R-77 novel, stated as the task brief requires: same
  formula, same shape, NOT redesigned -- only ``n_base`` (and the matched
  fixed-N comparison set) is re-scoped to {72,144,216,288}, so the
  mechanism is finally tested against the region it was motivated by.

THE MECHANISM -- reused verbatim, not redesigned
--------------------------------------------------
This file imports ``AdaptiveConfig``, the stress proxy
(``compute_v4_target_and_ratio``), and the fill simulator
(``run_adaptive_backtest`` / ``run_adaptive_period`` /
``run_taker_baseline`` / ``crash_lag_for_config`` /
``find_flip_to_flat_events`` / ``causality_probe``) directly from
``experiments/r77_novel_execution_regime_adaptive.py`` (added to
``sys.path`` by file path, the same "reuse a sibling experiment file as a
library" pattern the assignment names -- see IMPORTS below). Not one line
of the mechanism, the stress-proxy formula, or the fill model is
reproduced or modified here; only the *parameters swept* and the
*reporting* are new code. See that file's own docstring for the full
mechanism derivation (Almgren & Chriss 2000/2001; Cartea, Jaimungal &
Penalva 2015) and the fill-model justification (R-56 conservative's
100%-fill-on-touch model, not R-56 novel's penetration-probability model)
-- both carried over unchanged by importing the same functions rather than
re-deriving them.

Restated for convenience, the formula this round sweeps over (unchanged):

    is_derisking(i) = target[i] < target[i-1] - eps
    if is_derisking(i) and ratio[i] >= s_override:
        N(i) = n_min                                  # crash override
    else:
        urgency_mult(i) = 1 + kappa * max(0, ratio[i] - 1)
        N(i) = max(n_min, round(n_base / urgency_mult(i)))

PRE-REGISTRATION (written before any run in this file)
===========================================================

1. GRID (small, pre-registered, ~20-40 range applies to the SWEPT grid
   itself -- consistent with the assignment's guidance and this project's
   own convention that falsification/causality/thesis-test diagnostics are
   counted separately from the "grid"):

       n_base      in {72, 144, 216, 288}      # R-56's own original ceiling
       kappa       in {0.5, 1.0}                # the assignment's directed
                                                  # range -- deliberately
                                                  # narrower than R-77
                                                  # novel's {1,2,4}, since
                                                  # this round's question is
                                                  # about the SCOPE of
                                                  # n_base, not re-deriving
                                                  # kappa's own sensitivity
                                                  # (already characterized
                                                  # by R-77 novel's Phase 1)
       s_override  = 1.70                        # FIXED at R-77's own
                                                  # winning value (its
                                                  # Stage-B winner, the
                                                  # config the R-78 task
                                                  # brief names explicitly).
                                                  # Not varied: this round's
                                                  # question is whether
                                                  # entering N>=72 changes
                                                  # the crash-lag verdict,
                                                  # not whether a different
                                                  # override threshold
                                                  # would; re-deriving
                                                  # s_override here would
                                                  # re-open a question R-77
                                                  # already answered.
       n_min       = 2                           # FIXED at R-77's own
                                                  # winning value (its
                                                  # Stage-C winner), for the
                                                  # same reason as
                                                  # s_override above --
                                                  # reused, not re-derived.

   That is 4 x 2 = 8 SWEPT configurations (Phase 1, inner-train, spot,
   entry tier). Deliberately small and inside the guideline: the round's
   actual evidentiary weight lives in the crash-transition-lag /
   thesis-test comparison (section 4 below), not in a wide Sharpe search
   over a mechanism R-77 already characterized.

2. NOT A DUPLICATE (repeated from above, for the ledger entry that will
   quote this section directly): R-56 tested fixed N up to 288 but no
   adaptive mechanism. R-77 conservative tested fixed N only up to 24. R-77
   novel tested this exact adaptive mechanism but with n_base capped at 24.
   This round is the first to test the adaptive mechanism with n_base
   genuinely spanning R-56's own N>=72 danger zone.

3. PRE-REGISTERED PROMOTION DECISION RULE (mirrors R-77 conservative's
   verbatim, applied mechanically, before any run):

   PROMOTE only if ALL of:
     (a) beats kelly_regime_v4's existing, UNMODIFIED taker-fills-
         immediately execution on inner-validation Sharpe by more than the
         +-0.2 noise floor, OR is a clear drawdown/tail improvement, on
         BOTH spot and futures_5x (entry fee tier, the tier this project
         actually operates at);
     (b) ETH falsification (Bitfinex, entirely pre-2020) passes
         directionally (same sign as BTC, not opposite);
     (c) BTC pre-2020 control (Bitfinex) does not decisively fail;
     (d) THIS ROUND'S ACTUAL THESIS, stated precisely: crash-transition-lag
         (R-56/R-77's own methodology and event set, reused verbatim via
         import -- ``find_flip_to_flat_events`` / ``crash_lag_for_config``)
         has mean <=1-2 bars with NO SYSTEMATIC BLOWUP now that ``n_base``
         genuinely reaches 72-288 -- i.e., does entering the true danger
         zone finally show adaptive patience beating fixed patience on
         crash-lag safety, at a MATCHED fee-savings level (same comparison
         construction as R-77 novel's own thesis test 4, re-scoped: fixed
         N in {72,144,216,288} instead of {3,...,24}). Bar for "materially
         better", fixed now, identical in spirit to R-77 novel's own bar:
         the adaptive rule's crash-transition-lag violation count (lag > 2
         bars) must be at least 50% lower than the fee-savings-matched
         fixed-N baseline's, AND its mean lag must not be worse (<=) than
         that baseline's mean lag. "Both zero" proves nothing about the
         mechanism and is reported as INCONCLUSIVE, not quietly counted as
         a pass -- same convention as R-77 novel.
     (e) the swept neighbourhood (n_base x kappa) is a plateau, not a
         lucky single point.
   Anything else -> NEGATIVE. Any deviation from this plan after seeing
   results is stated explicitly and downgrades the result to in-sample,
   per ROUTINE.md step 4 -- never applied quietly.

4. THE FALSIFICATION TEST THAT WOULD KILL THE IDEA (named before running):
   if crash-transition-lag violations at large ``n_base`` (>=72) are just
   as bad for the adaptive mechanism as for a FIXED N of the same
   magnitude -- i.e. the adaptive mechanism does not show a materially
   lower violation rate than the matched fixed-N baseline per the bar in
   3(d) -- then the adaptive mechanism has not solved anything here; it
   has merely delayed where the fixed-patience failure mode reappears
   (e.g. by collapsing to n_min under the override, which fixed-N cannot
   do, but if that override is not enough to avoid the same blowup, the
   adaptivity bought nothing on the one axis this round exists to test).

DATA DISCIPLINE
----------------
Every backtest in this file is restricted to:
  - inner-train:       2017-01-01 -> 2020-12-31 (BTC, committed spot file)
  - inner-validation:  2021-01-01 -> 2022-12-31 (BTC, committed spot file)
  - combined window (crash-lag / thesis-test only): 2017-01-01 -> 2022-12-31
  - ETH falsification: data/ethusd_bitfinex_5m.csv.gz (physically ends
    2019-12, cannot leak the holdout even by accident)
  - BTC control:       data/btcusd_bitfinex_5m.csv.gz (Bitfinex venue,
    same file R-56/R-77 used, physically ends 2019-12)
No code path in this file ever loads, slices past, prints, or otherwise
touches a bar dated 2023-01-01 or later. The working BTC frame is loaded
via ``r77n.load_working_frame()`` (R-77 novel's own function), which cuts
to inner-validation's end and asserts ``cut.index[-1] < OOS_START`` at
load time -- reused unmodified, so this file inherits the exact same
runtime guard rather than re-deriving one. Grepped by the author for any
``202[3-9]`` date literal before finishing (see the report at the end of
this docstring's implementation); the operator should re-grep
independently, as this project's own practice requires.

USAGE
-----
    python experiments/r78_conservative_adaptive_n72.py causality
    python experiments/r78_conservative_adaptive_n72.py sweep
    python experiments/r78_conservative_adaptive_n72.py validate
    python experiments/r78_conservative_adaptive_n72.py falsify
    python experiments/r78_conservative_adaptive_n72.py crashlag
    python experiments/r78_conservative_adaptive_n72.py thesis
    python experiments/r78_conservative_adaptive_n72.py all      # everything, in order
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))   # so r77_novel_... is importable as a library

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import r77_novel_execution_regime_adaptive as r77n  # noqa: E402  -- reused, not edited

from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

DATA_DIR = r77n.DATA_DIR
OOS_START = r77n.OOS_START
INNER_TRAIN = r77n.INNER_TRAIN
INNER_VAL = r77n.INNER_VAL
COMBINED_WINDOW = r77n.COMBINED_WINDOW
TAKER_ENTRY, MAKER_ENTRY = r77n.TAKER_ENTRY, r77n.MAKER_ENTRY
TAKER_TOP, MAKER_TOP = r77n.TAKER_TOP, r77n.MAKER_TOP
FEE_TIERS = r77n.FEE_TIERS
SPOT, FUT = r77n.SPOT, r77n.FUT
MARKETS = r77n.MARKETS
AdaptiveConfig = r77n.AdaptiveConfig
fixed_n_config = r77n.fixed_n_config

# ---------------------------------------------------------------- pre-registered grid
N_BASE_VALUES = (72, 144, 216, 288)     # R-56's own original ceiling -- the whole point
KAPPA_VALUES = (0.5, 1.0)               # the assignment's directed range
S_OVERRIDE = 1.70                       # R-77 novel's own Stage-B winner, fixed not swept
N_MIN = 2                               # R-77 novel's own Stage-C winner, fixed not swept

# fixed-N comparison set for the thesis test, matching the n_base grid exactly
# so "fee-savings-matched" and "same magnitude" are the same comparison
MATCHED_FIXED_NS = N_BASE_VALUES

CONFIG_COUNTER: dict[str, int] = {}
_PHASE = {"name": "unassigned"}


def _count(k: int = 1) -> None:
    CONFIG_COUNTER[_PHASE["name"]] = CONFIG_COUNTER.get(_PHASE["name"], 0) + k


def _cfg(n_base: int, kappa: float) -> "AdaptiveConfig":
    return AdaptiveConfig(n_base=n_base, kappa=kappa, s_override=S_OVERRIDE, n_min=N_MIN)


# ============================================================ counted wrappers around r77n's engine
def _adaptive_period(df, start, end, market, taker, maker, cfg, label):
    res, diag = r77n.run_adaptive_period(df, start, end, market, taker, maker, cfg, data_label=label)
    _count()
    return res, diag


def _taker_baseline(df, start, end, market, taker, label):
    res = r77n.run_taker_baseline(df, start, end, market, taker, data_label=label)
    _count()
    return res


def _crash_lag(df, cfg, market, start, end):
    rows, diag = r77n.crash_lag_for_config(df, cfg, market, start=start, end=end)
    _count()
    return rows, diag


def _raw_baseline(df, market, taker, label):
    res = run_backtest(KellyRegimeV4(), df, replace(market, fee_rate=taker), 1_000.0, data_label=label)
    _count()
    return res


# ============================================================ 0. causality
def causality() -> bool:
    """Reuses R-77 novel's own ``causality_probe`` verbatim (tamper probe +
    synthetic ratio-peek guard-the-guard + N=1 identity), run against a
    representative N>=72 config (``n_base=144, kappa=1.0``) rather than
    R-77 novel's small-N default, so the probe exercises the actual regime
    this round cares about. That function performs exactly 4 real backtest
    evaluations (up-tamper, down-tamper, N=1-taker-baseline, N=1-adaptive-
    identity) plus one deterministic synthetic check counted separately;
    counted here as 4 diagnostic configurations, matching R-77 novel's own
    convention for the same probe.
    """
    _PHASE["name"] = "causality"
    df, label = r77n.load_working_frame()
    print(f"{len(df):,} bars {df.index[0]} -> {df.index[-1]} (data: {label})")
    probe_df = df.iloc[-160_000:]
    rep_cfg = _cfg(144, 1.0)
    ok = r77n.causality_probe(probe_df, rep_cfg, SPOT)
    _count(4)
    print(f"\nCAUSALITY (R-78 conservative, config={rep_cfg.tag()}): {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================ 1. sweep (Phase 1)
def sweep() -> tuple[list[dict], "AdaptiveConfig"]:
    """n_base in {72,144,216,288} x kappa in {0.5,1.0} = 8 configs, inner-
    train, spot, entry tier, + 1 baseline = 9 configurations. Winner chosen
    by inner-train Sharpe among the 8 (same selection convention R-77
    novel's own Phase 1 used); the neighbourhood is reported in full below
    so plateau-vs-peak (criterion e) can be read directly from the table.
    """
    _PHASE["name"] = "sweep"
    df, label = r77n.load_working_frame()
    start, end = INNER_TRAIN
    print("=" * 100)
    print(f"SWEEP -- n_base in {N_BASE_VALUES} x kappa in {KAPPA_VALUES}, s_override={S_OVERRIDE}, "
          f"n_min={N_MIN}, inner-train, spot, entry tier (8 configs + 1 baseline)")
    print("=" * 100)
    base = _taker_baseline(df, start, end, SPOT, TAKER_ENTRY, label)
    base_m = compute_metrics(base)
    base_row = r77n._row("BASELINE taker-only", base_m)
    r77n._print_row(base_row)
    rows = [dict(base_row, n_base=None, kappa=None)]

    candidates: list[tuple["AdaptiveConfig", dict]] = []
    for n_base in N_BASE_VALUES:
        for kappa in KAPPA_VALUES:
            cfg = _cfg(n_base, kappa)
            res, diag = _adaptive_period(df, start, end, SPOT, TAKER_ENTRY, MAKER_ENTRY, cfg, label)
            m = compute_metrics(res)
            row = r77n._row(cfg.tag(), m, diag, base_m.fees_paid)
            r77n._print_row(row)
            row.update(n_base=n_base, kappa=kappa)
            rows.append(row)
            candidates.append((cfg, row))

    winner_cfg, winner_row = max(candidates, key=lambda cr: cr[1]["sharpe"])
    print(f"\nSWEEP WINNER: {winner_cfg.tag()}  sharpe={winner_row['sharpe']:.3f}  "
          f"(baseline sharpe={base_m.sharpe:.3f})")
    sharpes = [r["sharpe"] for _, r in candidates]
    print(f"neighbourhood sharpe range: [{min(sharpes):.3f}, {max(sharpes):.3f}]  "
          f"(spread={max(sharpes) - min(sharpes):.3f})")
    return rows, winner_cfg


# ============================================================ 2. full-matrix validation
def validate(winner: "AdaptiveConfig") -> list[dict]:
    """winner x {spot, futures_5x} x {inner-train, inner-val} x {entry, top}
    = 8 configs, + 8 baselines = 16. This is where promotion criterion (a)
    is actually evaluated (inner-validation Sharpe, both markets).
    """
    _PHASE["name"] = "validate"
    df, label = r77n.load_working_frame()
    print("\n" + "=" * 100)
    print(f"FULL-MATRIX VALIDATION -- {winner.tag()} x 2 markets x 2 periods x 2 fee tiers")
    print("=" * 100)
    rows = []
    for mname, market in MARKETS.items():
        for pname, (start, end) in (("inner-train", INNER_TRAIN), ("inner-val", INNER_VAL)):
            for tname, (taker, maker) in FEE_TIERS.items():
                base = _taker_baseline(df, start, end, market, taker, label)
                base_m = compute_metrics(base)
                print(f"\n-- {mname} / {pname} / {tname} tier --")
                base_row = r77n._row("BASELINE taker-only", base_m)
                r77n._print_row(base_row)
                rows.append(dict(base_row, market=mname, period=pname, tier=tname, kind="baseline"))

                res, diag = _adaptive_period(df, start, end, market, taker, maker, winner, label)
                m = compute_metrics(res)
                row = r77n._row(winner.tag(), m, diag, base_m.fees_paid)
                r77n._print_row(row)
                row.update(market=mname, period=pname, tier=tname, kind="adaptive",
                            sharpe_delta=m.sharpe - base_m.sharpe,
                            dd_delta=m.max_drawdown_pct - base_m.max_drawdown_pct)
                print(f"    sharpe_delta = {row['sharpe_delta']:+.3f}   dd_delta = {row['dd_delta']:+.2f}pp")
                rows.append(row)
    return rows


# ============================================================ 3. falsification (ETH + BTC control)
def falsify(winner: "AdaptiveConfig") -> list[dict]:
    """winner x {ETH-falsification, BTC-control} x {spot, futures_5x},
    entry tier = 4 + 4 baselines = 8.
    """
    _PHASE["name"] = "falsification"
    print("\n" + "=" * 100)
    print("FALSIFICATION -- ETH (Bitfinex, pre-2020) + BTC control (Bitfinex, pre-2020), entry tier")
    print("=" * 100)
    from tradebot.data import load_ohlcv_csv
    eth = load_ohlcv_csv(DATA_DIR / "ethusd_bitfinex_5m.csv.gz")
    btc = load_ohlcv_csv(DATA_DIR / "btcusd_bitfinex_5m.csv.gz")
    assert eth.index.max() < pd.Timestamp("2020-01-01", tz="UTC")
    assert btc.index.max() < pd.Timestamp("2020-01-01", tz="UTC")

    rows = []
    for dname, dset in (("ETH-falsification", eth), ("BTC-control", btc)):
        for mname, market in MARKETS.items():
            base_res = _raw_baseline(dset, market, TAKER_ENTRY, dname)
            base_m = compute_metrics(base_res)
            print(f"\n-- {dname} / {mname} --")
            base_row = r77n._row("BASELINE taker-only", base_m)
            r77n._print_row(base_row)
            rows.append(dict(base_row, dataset=dname, market=mname, kind="baseline"))

            res, diag = _adaptive_period(dset, None, None, market, TAKER_ENTRY, MAKER_ENTRY, winner, dname)
            m = compute_metrics(res)
            row = r77n._row(winner.tag(), m, diag, base_m.fees_paid)
            r77n._print_row(row)
            row.update(dataset=dname, market=mname, kind="adaptive", sharpe_delta=m.sharpe - base_m.sharpe)
            print(f"    sharpe_delta = {row['sharpe_delta']:+.3f}")
            rows.append(row)
    return rows


# ============================================================ 4. crash-transition-lag (falsification test 3)
def crashlag(winner: "AdaptiveConfig") -> list[dict]:
    """winner on the combined window, spot, entry tier -- 1 configuration.
    Reuses R-77 novel's own event set (``find_flip_to_flat_events``:
    target[i-1] > 0.05 and target[i] < 1e-9) and its own resolution
    bookkeeping (``crash_lag_for_config``), imported unmodified.
    """
    _PHASE["name"] = "crashlag"
    df, label = r77n.load_working_frame()
    print("\n" + "=" * 100)
    print(f"CRASH-TRANSITION-LAG -- {winner.tag()}, combined window {COMBINED_WINDOW}, spot, entry tier")
    print("=" * 100)
    rows, diag = _crash_lag(df, winner, SPOT, COMBINED_WINDOW[0], COMBINED_WINDOW[1])
    if not rows:
        print("no flip-to-flat events in this window")
        return rows
    for r in rows:
        print(f"  {r['event_ts']}  lag={r['lag_bars']:>3} bars  kind={r['kind']:16s} "
              f"n_eff={r['n_eff']}  override={r['override_fired']}")
    lags = [r["lag_bars"] for r in rows]
    violations = [r for r in rows if r["lag_bars"] > 2]
    print(f"\n{len(rows)} flip-to-flat events; mean lag={np.mean(lags):.2f} bars "
          f"max lag={max(lags)} bars; violations(>2 bars)={len(violations)}/{len(rows)} "
          f"(baseline is always 1 bar / 5 minutes)")
    return rows


# ============================================================ 5. thesis test -- adaptive vs matched fixed-N in N>=72
def thesis(winner: "AdaptiveConfig") -> dict:
    """The round's actual question. Combined window, spot, entry tier.

    winner: crash_lag (1) + economic run (1) = 2
    baseline: 1
    fixed-N sweep over {72,144,216,288}: crash_lag (1) + economic run (1)
      each = 8
    total = 11 configurations.

    Pre-registered bar (restated verbatim from the module docstring):
    adaptive PASSES criterion (d) only if its violation count (lag > 2
    bars) is >=50% lower than the fee-savings-matched fixed-N baseline's
    AND its mean lag is <= that baseline's mean lag. "Both zero" is
    reported as INCONCLUSIVE, not a pass.
    """
    _PHASE["name"] = "thesis"
    df, label = r77n.load_working_frame()
    print("\n" + "=" * 100)
    print(f"THESIS TEST -- {winner.tag()} vs. fee-savings-matched fixed-N in {MATCHED_FIXED_NS}, "
          "combined window, spot, entry tier")
    print("=" * 100)
    start, end = COMBINED_WINDOW
    base = _taker_baseline(df, start, end, SPOT, TAKER_ENTRY, label)
    base_m = compute_metrics(base)

    winner_rows, _ = _crash_lag(df, winner, SPOT, start, end)
    winner_res, _ = _adaptive_period(df, start, end, SPOT, TAKER_ENTRY, MAKER_ENTRY, winner, label)
    winner_m = compute_metrics(winner_res)
    winner_fee_saved_pct = 100.0 * (base_m.fees_paid - winner_m.fees_paid) / base_m.fees_paid
    winner_mean_lag = float(np.mean([r["lag_bars"] for r in winner_rows])) if winner_rows else 0.0
    winner_violations = sum(1 for r in winner_rows if r["lag_bars"] > 2)
    print(f"\nADAPTIVE {winner.tag()}: fee_saved={winner_fee_saved_pct:+.1f}%  "
          f"events={len(winner_rows)}  mean_lag={winner_mean_lag:.2f}  "
          f"violations(>2)={winner_violations}  sharpe={winner_m.sharpe:.3f}")

    sweep_rows = []
    for nfix in MATCHED_FIXED_NS:
        cfg = fixed_n_config(nfix)
        rows, _ = _crash_lag(df, cfg, SPOT, start, end)
        res, _ = _adaptive_period(df, start, end, SPOT, TAKER_ENTRY, MAKER_ENTRY, cfg, label)
        m = compute_metrics(res)
        fee_saved_pct = 100.0 * (base_m.fees_paid - m.fees_paid) / base_m.fees_paid
        lags = [r["lag_bars"] for r in rows]
        mean_lag = float(np.mean(lags)) if lags else 0.0
        violations = sum(1 for lag in lags if lag > 2)
        sweep_rows.append({"n": nfix, "fee_saved_pct": fee_saved_pct, "events": len(rows),
                            "mean_lag": mean_lag, "violations": violations, "sharpe": m.sharpe})
        print(f"  fixed N={nfix:>3d}: fee_saved={fee_saved_pct:+6.1f}%  events={len(rows):>3d}  "
              f"mean_lag={mean_lag:>5.2f}  violations(>2)={violations:>3d}  sharpe={m.sharpe:.3f}")

    matched = min(sweep_rows, key=lambda r: abs(r["fee_saved_pct"] - winner_fee_saved_pct))
    print(f"\nMatched fixed-N: N={matched['n']} (fee_saved={matched['fee_saved_pct']:+.1f}% vs. "
          f"adaptive's {winner_fee_saved_pct:+.1f}%)")
    print(f"  adaptive:      violations={winner_violations}  mean_lag={winner_mean_lag:.2f}")
    print(f"  fixed N={matched['n']:>3d}:   violations={matched['violations']}  mean_lag={matched['mean_lag']:.2f}")

    both_zero = matched["violations"] == 0 and winner_violations == 0
    violation_drop_ok = (winner_violations <= 0.5 * matched["violations"]) if matched["violations"] else \
        (winner_violations == 0)
    mean_lag_ok = winner_mean_lag <= matched["mean_lag"] + 1e-9
    verdict = "INCONCLUSIVE (both zero violations)" if both_zero else (
        "PASS" if (violation_drop_ok and mean_lag_ok) else "FAIL")
    print(f"\nTHESIS TEST VERDICT (promotion criterion d): {verdict}")
    return {"winner_fee_saved_pct": winner_fee_saved_pct, "winner_mean_lag": winner_mean_lag,
            "winner_violations": winner_violations, "matched_n": matched["n"],
            "matched_fee_saved_pct": matched["fee_saved_pct"], "matched_mean_lag": matched["mean_lag"],
            "matched_violations": matched["violations"], "sweep": sweep_rows, "verdict": verdict}


# ============================================================ main
def main() -> None:
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"

    if choice in ("causality", "all"):
        causality()

    winner = _cfg(N_BASE_VALUES[0], KAPPA_VALUES[0])  # placeholder until sweep runs
    if choice in ("sweep", "all"):
        _, winner = sweep()

    if choice in ("validate", "all"):
        validate(winner)

    if choice in ("falsify", "all"):
        falsify(winner)

    if choice in ("crashlag", "all"):
        crashlag(winner)

    if choice in ("thesis", "all"):
        thesis(winner)

    total = sum(CONFIG_COUNTER.values())
    print(f"\nCONFIGS EVALUATED THIS RUN: {CONFIG_COUNTER}  total={total}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[{time.time() - t0:.0f}s]")
