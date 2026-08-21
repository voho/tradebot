#!/usr/bin/env python
"""R-78 (novel branch): directional-asymmetry execution urgency for kelly_regime_v4.

Not registered: lives under ``experiments/`` so it is not auto-discovered,
per ROUTINE.md step 5. Does not modify ``kelly_regime_v4.py``,
``kelly_regime_v3.py``, ``kelly_regime.py``, ``engine.py`` or ``broker.py``
-- all reused read-only as libraries. Does not modify either of R-56's own
files or either of R-77's own files (``experiments/r77_conservative_
narrowcap_execution.py`` / ``experiments/r77_novel_execution_regime_
adaptive.py``), which this module only *imports* (the latter) for its
fill-simulation/backtest-loop machinery, per this round's own brief. Zero
overlap with the sibling parallel branch's file
(``experiments/r78_conservative_adaptive_n72.py``) -- not read, not
touched, so the two R-78 branches stay independent evidence.

THE IDEA, ONE SENTENCE
-----------------------
R-56 found that a resting limit order with any FIXED patience N >= ~72
bars genuinely delays v4's flip-toward-flat de-risking events during
crash transitions (a real, if diffuse, over-patience cost R-56 measured
directly -- futures inner-validation reversing sign at N>=72, several
25-71 bar delays) even though the maker-fee saving from patience is real
and grows with N -- but R-56 never showed, and no round since has tested,
whether large patience is ALSO dangerous for the opposite trade direction
(adding to a position, or flipping further into an existing long) that
this project's regime-vote signal also issues; if the danger is specific
to de-risking, then the fix does not need to be "cap N low everywhere" (R-56's
implicit answer) or "shrink N smoothly as a stress proxy rises" (R-77's
answer) -- it can be a hard, binary split by trade PURPOSE: de-risking
orders always execute immediately (N=1, unconditionally), risk-increasing
orders always rest for a large, fixed N_add, with no continuous covariate
and no threshold to tune.

CONSTRAINT ATTACKED: COST (costs that scale with the signal), the same
constraint R-56 and R-77 attacked, via the same "change HOW an
already-decided trade fills, not WHEN or how MUCH" mechanism this branch
inherits unmodified from both.

WHY THIS IS NOT A DUPLICATE (stated precisely, per ROUTINE.md step 1.2)
--------------------------------------------------------------------------
- **Not R-56**: R-56 (both branches) applied ONE fixed patience N to every
  order regardless of direction, and its N=72/288 sweep is exactly the
  region where it found the danger -- but it never asked whether that
  danger is direction-specific. This round tests N_add in that same
  N>=72 danger zone, but ONLY for risk-increasing orders; de-risking
  orders never enter that zone at all (they are always N=1). If R-56's
  danger really is a de-risking-specific phenomenon, this construction
  should show zero crash-lag violations even at N_add=288 -- a
  qualitatively different claim than "a smaller N is safer for
  everyone," which is all R-56 itself tested.
- **Not R-77 novel**: R-77 novel's mechanism is `N(i) = max(n_min,
  round(n_base / (1 + kappa * max(0, ratio[i]-1))))`, a CONTINUOUS
  function of a causal stress ratio, with a soft override that only
  fires above a tuned threshold `s_override` -- three free parameters
  (n_base, kappa, s_override), and its own tested range (n_base in
  {12,24}) never reached N>=72. This round has ONE free parameter
  (N_add) and NO continuous scaling at all: the stress ratio is not
  computed or read anywhere in this file's decision rule. Direction of
  the pending order -- not the market's current volatility state -- is
  the entire input to the rule. An order placed on a quiet day and an
  order placed during a spike get the same N if they are the same
  direction; R-77's rule would treat them differently. That is the
  structural difference this round is built to test.
- **Not the parallel conservative branch** (`r78_conservative_adaptive_
  n72.py`, not read here): per this round's own brief, that branch
  re-tests R-77's continuous kappa-scaled formula with `n_base` raised
  into the N>=72 range -- a rescoping of the SAME mechanism, not a
  different one. This branch's mechanism has no kappa, no s_override, and
  no n_base; it cannot collapse to or from the conservative branch's
  formula by any parameter setting (the conservative formula is smooth in
  ratio[i] even at its override boundary; this rule is a step function of
  direction alone, discontinuous in nothing but which way the trade
  points).

CITATIONS (grounding the mechanism, not decorating it)
---------------------------------------------------------
Perold (1988, "The Implementation Shortfall", *J. Portfolio Management*
14(3)) frames execution cost as the gap between a paper decision and its
realized fill, and explicitly separates the COST of delay from its
DIRECTION -- a delayed purchase and a delayed sale are not symmetric
because the alternative (not trading) has a different opportunity cost in
each case. Almgren & Chriss (2000/2001, cited identically by R-77) build
urgency as a risk-aversion trade-off between impact cost (favors patience)
and holding-period risk (favors speed) -- but their own framework's risk
term is about the RESIDUAL POSITION being carried while the order rests,
not the market's ambient volatility per se: for a de-risking order, the
residual position being carried while patient is the very same exposure
the strategy is trying to shed, so the "holding risk" term is large by
construction, independent of whatever the stress proxy says on that
particular bar. For a risk-increasing order, the residual position while
patient is the OLD (smaller) exposure, not the new, larger one -- the
holding-risk term is small by construction. That asymmetry is structural,
not conditional on a stress reading, which is the whole argument for a
hard split by direction rather than a continuous throttle on a proxy that
can be low right up to the bar a crash starts (exactly R-56's diagnosed
failure mode: N>=72 delays de-risking BECAUSE the stress signal that would
have shortened patience had not yet fired).

THE MECHANISM, EXACT RULE (pre-registered before any run)
---------------------------------------------------------------
For each rebalance decision at bar i (`target[i] != target[i-1]`, v4's own
already-causal, already-hysteresis-gated signal, untouched here -- reused
via `compute_v4_target_and_ratio`, imported read-only from R-77 novel):

    is_derisking(i) = target[i] < target[i-1] - eps    # same definition
                                                          # R-77 novel uses,
                                                          # and for the
                                                          # same reason:
                                                          # v4 never shorts
                                                          # (frac in [0,1],
                                                          # scale >= 0), so
                                                          # de-risking here
                                                          # is always
                                                          # flip-toward-flat,
                                                          # never
                                                          # flip-to-short --
                                                          # re-verified below
                                                          # via
                                                          # _assert_v4_never_shorts,
                                                          # imported, not
                                                          # re-derived

    N(i) = N_DERISK = 1                 if is_derisking(i)     # ALWAYS
                                                                  # immediate
                                                                  # taker,
                                                                  # unconditionally
    N(i) = N_add                        otherwise               # ALWAYS
                                                                  # patient,
                                                                  # fixed

No continuous term, no market-state input, no override threshold. The
causal stress ratio `ratio[i]` that R-77's own `compute_v4_target_and_ratio`
also returns is IMPORTED (because the shared library function returns it
alongside target) but is never read by this file's own decision rule --
grep this file for "ratio_i" inside `DirectionalConfig.n_eff` to confirm
it is an unused, ignored parameter, present only because it is required by
the shared `run_adaptive_backtest` call signature this round reuses
verbatim.

FILL MECHANISM (reused verbatim, not re-implemented): R-77 novel's own
`run_adaptive_backtest` -- post a resting maker-fee limit at the decision
bar's close, check bars i+1..i+N-1's full high/low for a touch (100%
fill-on-touch), forced taker fallback at bar i+N's open if untouched. That
function accepts any object exposing `.n_eff(ratio_i, is_derisking) ->
(n, flag)`; `DirectionalConfig` below is that object, and it is this
round's only new code inside the simulation path (~10 lines). Everything
else -- broker, fee application, liquidation checks, warmup handling,
crash-lag event bookkeeping, causality-probe scaffolding -- is R-77
novel's, imported, not copied, so any future correctness fix to that
machinery benefits both files identically. Justification for reusing THAT
fill model rather than R-56 novel's penetration-probability model is
identical to R-77's own (see that file's docstring): stacking a second
uncertain-fill layer on top of the one new variable this round isolates
(who gets N=1 vs N_add) would make a negative result impossible to
attribute.

WHAT THIS ROUND DELIBERATELY DOES NOT MODEL: same as R-56/R-77 -- no order
book depth, no queue position, no informed-flow avoidance beyond the
touch-based fill rule inherited from R-77 novel (itself inherited from
R-56 conservative). See those files' own "WHAT IS NOT MODELLED" sections;
unchanged here.

DEVIATION FROM THE BRIEF, STATED PLAINLY: identical to R-77 novel's own
deviation note. `kelly_regime_v4`'s `target[i]` is provably non-negative
(re-verified here via the imported `_assert_v4_never_shorts`), so
"de-risking" can only mean flip-toward-flat for this specific strategy,
never flip-to-short. The mechanism above is written for that reality.

PRE-REGISTERED GRID (before any run)
----------------------------------------
Core: N_add in {72, 144, 216, 288} -- four values spanning and exceeding
R-56's own N=72 danger threshold up to its largest tested value (288 = 1
full day of patience). The de-risking side has NO free parameter (always
N=1) -- one tunable dial total, versus R-77's three (n_base, kappa,
s_override). This is itself a design property worth reporting (Occam's
razor: does the simpler mechanism do as well or better on the one
question that matters -- crash-transition lag -- than R-77's more
elaborate formula, if both branches' Sharpe/drawdown results come back
comparably negative, which is this project's modal outcome for
execution-only mechanisms per R-56/B-24/R-77).

Planned run buckets (counts fixed here, before any number is computed):
  1. causality/tamper probe                       ~6 diagnostic configs
  2. core validation matrix: N_add(4) x market(2) x
     period(2), entry fee tier                    16 core  + 4 baseline
  3. top-fee-tier robustness spot-check: N_add=288
     (most extreme), both markets, inner-val only  2 core  + 2 baseline
  4. falsification: ETH + BTC pre-2020 control,
     N_add(4) x dataset(2), spot only, entry tier  8 core  + 4 baseline
  5. crash-transition-lag: N_add(4) on the combined
     inner-train+inner-validation window, spot,
     entry tier (all four MUST be lag-identical by
     construction -- de-risking never depends on
     N_add -- itself a falsifiable prediction)      4 core
  6. thesis comparison: R-77's OWN fixed-N=24
     baseline, re-measured via R-77's own
     `fixed_n_config`/`crash_lag_for_config` on the
     SAME combined window, for a same-methodology
     violation-rate reference point                 1 core
Target total: ~20-40 in the project's stated range (see ROUTINE.md's
"fee study ran 32" precedent); actual total printed and reported below.

PRE-REGISTERED FALSIFICATION TESTS (before any run)
------------------------------------------------------
1. ETH falsification (Bitfinex, entirely pre-2020) -- does the fee-saving
   / Sharpe pattern replicate directionally (same sign as BTC inner-train)?
2. BTC pre-2020 control (Bitfinex, entirely pre-2020) -- same directional-
   replication bar, does not decisively fail.
3. Crash-transition-lag, same construction as R-56/R-77 (flip-to-flat
   events, lag in bars vs. the always-taker baseline's fixed 1 bar).
4. **This round's own thesis test, stated precisely before running**: at
   every N_add in {72,144,216,288}, mean crash-transition lag must stay
   near the baseline's 1 bar (target: <=1-2 bars) and the violation count
   (lag > 2 bars) must not be WORSE than R-77's own matched-methodology
   N<=24 fixed-N reference measured fresh in this file (bucket 6 above) --
   because de-risking orders never wait on N_add by construction, the
   pre-registered PREDICTION is that all four N_add cells produce
   IDENTICAL crash-lag numbers to each other, and equal-or-better numbers
   than the N=24 reference. Any N_add cell that shows a WORSE violation
   count than a smaller N_add cell would be a construction bug (the rule
   has no mechanism by which N_add should affect de-risking lag at all)
   and is treated as such, not reported as a soft finding.
5. **The falsification-of-the-round's-own-thesis test, named explicitly
   per the brief**: if crash-transition-lag violations appear ANYWAY, at
   ANY N_add, despite de-risking orders always filling at N=1, that would
   mean R-56's original danger was never really a pure de-risking-order-
   patience phenomenon (something else -- e.g. adverse selection on the
   FORCED taker fill's own next-bar-open price, or a case where the
   "de-risking" classification itself lags the true regime turn by more
   than one bar for reasons unrelated to execution patience) -- a
   genuinely informative failure mode, reported as its own finding, not
   folded into a flat "NEGATIVE".

PRE-REGISTERED PROMOTION DECISION RULE (before any run, applied mechanically)
--------------------------------------------------------------------------------
PROMOTE only if ALL of:
  (a) beats kelly_regime_v4 (unmodified, immediate-taker) on inner-
      validation Sharpe by > +-0.2, or is a clear drawdown/tail win, BOTH
      spot and futures_5x, AND at a fee-savings level comparable to what
      R-56/R-77 measured for similar N (i.e. not achieved by an
      accounting artifact -- fee dollars actually saved, cross-checked
      against baseline fees paid);
  (b) ETH falsification passes directionally;
  (c) BTC pre-2020 control does not decisively fail;
  (d) **the round's actual thesis**: crash-transition-lag shows zero or
      near-zero violations at every N_add up to 288 -- fixed bar: mean lag
      <=1-2 bars, violation rate (lag>2 bars) not worse than R-77's own
      N<=24 fixed baseline measured fresh in bucket 6;
  (e) the swept neighbourhood across N_add in {72,144,216,288} is a
      plateau, not a spike (materially similar Sharpe/drawdown/fee-saving
      pattern across the four values, since they differ only in how much
      EXTRA patience risk-increasing orders get).
Anything else -> NEGATIVE. Any deviation from this plan after seeing
results is stated explicitly and downgrades the result to in-sample, per
ROUTINE.md step 4 -- not applied quietly.

DATA DISCIPLINE
----------------
Every backtest in this file is restricted to:
  - inner-train:       2017-01-01 -> 2020-12-31 (BTC, committed spot file)
  - inner-validation:  2021-01-01 -> 2022-12-31 (BTC, committed spot file)
  - combined window (crash-lag/thesis-ref only): 2017-01-01 -> 2022-12-31
  - ETH falsification: data/ethusd_bitfinex_5m.csv.gz (whole file, entirely
    pre-2020 -- physically cannot contain a holdout bar)
  - BTC control:       data/btcusd_bitfinex_5m.csv.gz (whole file, entirely
    pre-2020, same file R-56/R-77 used)
Working BTC frame is loaded via R-77 novel's own `load_working_frame`,
which cuts to inner-validation's end immediately on load and asserts the
cut with a runtime `assert`, so no bar dated OOS_START or later can reach
any computation, print, or report in this module even by accident. This
file's own `main()` re-asserts the same bound independently (defense in
depth, not reliance on the import alone). Grepped by the author for any
`202[3-9]` date literal before reporting -- see the final report for the
grep command and its output; the operator should re-grep independently,
as this project's own practice requires.

USAGE
-----
    python experiments/r78_novel_execution_directional_asymmetry.py causality
    python experiments/r78_novel_execution_directional_asymmetry.py core
    python experiments/r78_novel_execution_directional_asymmetry.py toptier
    python experiments/r78_novel_execution_directional_asymmetry.py falsify
    python experiments/r78_novel_execution_directional_asymmetry.py crashlag
    python experiments/r78_novel_execution_directional_asymmetry.py thesisref
    python experiments/r78_novel_execution_directional_asymmetry.py all      # everything, in order
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # experiments/, for the r77 import below

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.metrics import Metrics, compute_metrics  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.data import load_ohlcv_csv  # noqa: E402

# ---------------------------------------------------------------------------
# Reused, read-only, from R-77's own novel branch: the fill-simulation /
# backtest-loop machinery and the causal target+stress-ratio computation.
# The DECISION RULE (DirectionalConfig below) is NOT imported -- it is this
# file's own new code, per the round's brief.
# ---------------------------------------------------------------------------
from r77_novel_execution_regime_adaptive import (  # noqa: E402
    DATA_DIR, OOS_START, INNER_TRAIN, INNER_VAL, COMBINED_WINDOW,
    TAKER_ENTRY, MAKER_ENTRY, TAKER_TOP, MAKER_TOP, FEE_TIERS,
    SPOT, FUT, MARKETS, V4_WARMUP,
    compute_v4_target_and_ratio, _assert_v4_never_shorts,
    run_adaptive_backtest, run_adaptive_period, run_taker_baseline,
    find_flip_to_flat_events, crash_lag_for_config, fixed_n_config,
    load_working_frame, EPS_TARGET,
)

CONFIG_COUNTER = {"core": 0, "diagnostic": 0}


def _count(kind: str = "core", k: int = 1) -> None:
    CONFIG_COUNTER[kind] += k


# ============================================================ the decision rule
@dataclass(frozen=True)
class DirectionalConfig:
    """This round's ONLY new decision rule: a hard, binary split by trade
    direction. Duck-type compatible with R-77 novel's `AdaptiveConfig`
    (same `.tag()` / `.n_eff(ratio_i, is_derisking)` interface), which is
    the ONLY reason `run_adaptive_backtest` (imported, unmodified) accepts
    it without any change to that function -- structural reuse, not a
    subclass or a monkeypatch.

    ``ratio_i`` is accepted (the shared engine always passes it) but never
    read: this rule has no continuous term at all. That is the entire
    point of the design, not an oversight -- see the module docstring's
    "not R-77 novel" section.
    """

    n_add: int = 72           # patience (bars) for RISK-INCREASING orders only -- the one free parameter
    n_derisk: int = 1         # patience for DE-RISKING orders -- always 1 (immediate taker), NOT swept

    def tag(self) -> str:
        return f"nadd{self.n_add}_nderisk{self.n_derisk}"

    def n_eff(self, ratio_i: float, is_derisking: bool) -> tuple[int, bool]:
        """Pure function of DIRECTION ONLY. ``ratio_i`` (the causal stress
        proxy the shared engine always computes and passes) is accepted
        for interface compatibility and deliberately never referenced --
        confirm by inspection: no ``ratio_i`` token appears below this
        docstring in this method body.
        """
        if is_derisking:
            return self.n_derisk, True    # "flag" reused here to mean "de-risking branch taken" (diagnostic only)
        return self.n_add, False


N_ADD_GRID = (72, 144, 216, 288)


# ============================================================ reporting helpers (own, R-77-novel-style)
def _row(tag: str, m: Metrics, diag: dict | None = None, base_fees: float | None = None) -> dict:
    out = {"tag": tag, "final_balance": m.final_balance, "profit_pct": m.profit_pct,
           "num_trades": m.num_trades, "max_dd_pct": m.max_drawdown_pct,
           "sharpe": m.sharpe, "fees_paid": m.fees_paid, "liquidated": m.liquidated}
    if diag is not None:
        total = diag["maker_fills"] + diag["taker_fallback_fills"]
        out["maker_fill_rate_pct"] = 100.0 * diag["maker_fills"] / total if total else float("nan")
        out["cancels"] = diag["cancels"]
        out["derisk_fires"] = diag["override_fires"]   # count of orders that took the N=1 de-risking branch
        out["decisions"] = diag["decisions"]
        out["mean_n_eff"] = diag["mean_n_eff"]
    if base_fees is not None:
        out["fees_saved"] = base_fees - m.fees_paid
        out["fees_saved_pct"] = 100.0 * (base_fees - m.fees_paid) / base_fees if base_fees else float("nan")
    return out


def _print_row(r: dict) -> None:
    extra = ""
    if "maker_fill_rate_pct" in r:
        extra = (f" maker%={r['maker_fill_rate_pct']:>5.1f} cancel={r['cancels']:>2d} "
                 f"derisk_fires={r['derisk_fires']:>3d} decisions={r['decisions']:>3d} "
                 f"meanN={r['mean_n_eff']:>5.1f}")
    fs = f" fee$saved={r['fees_saved']:>+8.2f}({r['fees_saved_pct']:>+5.1f}%)" if "fees_saved" in r else ""
    print(f"  {r['tag']:44s} final=${r['final_balance']:>11,.1f} ({r['profit_pct']:>+8.1f}%) "
          f"trades={r['num_trades']:>4d} DD={r['max_dd_pct']:>5.1f}% sharpe={r['sharpe']:>5.2f} "
          f"fees=${r['fees_paid']:>8.2f}{fs}{extra}{' LIQUIDATED' if r['liquidated'] else ''}")


# ============================================================ 1. causality probe
def _synthetic_direction_check() -> bool:
    """Deterministic, hand-built proof this round's OWN new code (the
    ``DirectionalConfig.n_eff`` split) does what it claims: a de-risking
    order gets N=1 (resolves at the very next bar's open, taker fallback,
    regardless of ``n_add``), and a risk-increasing order gets N=n_add
    (rests up to n_add-1 bars before falling back), on a hand-constructed
    frame where the fill outcome is unambiguous either way.
    """
    n = 400
    idx = pd.date_range("2018-01-01", periods=n, freq="5min", tz="UTC")
    # Bars 0-10 sit at 100 (bar 10's close = 100 becomes the risk-increasing
    # BUY order's limit price); bars 11+ jump to and hold at 110, so the buy
    # limit (100) is NEVER touched (touched = low <= limit_price, and
    # low=110 > 100 for every remaining bar) -- the order is forced all the
    # way to its taker-fallback deadline, isolating N(i) itself with zero
    # ambiguity from the fill-on-touch branch. The de-risking order at bar
    # 200 has n_derisk=1 (empty touch-check window by construction), so it
    # is forced to the very next bar regardless of price -- unaffected by
    # this same price path.
    price = np.full(n, 110.0)
    price[:11] = 100.0
    df = pd.DataFrame({"open": price, "high": price, "low": price,
                        "close": price, "volume": 1.0}, index=idx)
    target = np.zeros(n)
    target[10:] = 1.0     # risk-increasing flip at bar 10 (0 -> 1)
    target[200:] = 0.30   # de-risking flip at bar 200 (1 -> 0.30)

    class _FixedTargetStrategy:
        name = "synthetic_directional"
        warmup = 0

        def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.copy()
            frame["target"] = target
            return frame

    cfg = DirectionalConfig(n_add=50, n_derisk=1)
    strat = _FixedTargetStrategy()
    tgt, ratio = compute_v4_target_and_ratio(df, KellyRegimeV4())  # unused values, just exercising the shared call
    result, diag = run_adaptive_backtest(df, SPOT, TAKER_ENTRY, MAKER_ENTRY, cfg, 1_000.0,
                                          strategy=strat, target=target, ratio=np.full(n, np.nan))
    by_placed = {e["placed_at"]: e for e in diag["events"]}
    risk_up_event = by_placed.get(10)
    derisk_event = by_placed.get(200)
    ok_up = (risk_up_event is not None and risk_up_event["kind"] == "taker_fallback"
             and risk_up_event["resolved_at"] == 10 + 50 and risk_up_event["n_eff"] == 50)
    ok_down = (derisk_event is not None and derisk_event["kind"] == "taker_fallback"
               and derisk_event["resolved_at"] == 200 + 1 and derisk_event["n_eff"] == 1)
    print(f"  synthetic: risk-increasing order (bar 10) -> {risk_up_event['kind'] if risk_up_event else None} "
          f"at bar {risk_up_event['resolved_at'] if risk_up_event else None} (n_eff={risk_up_event['n_eff'] if risk_up_event else None}, "
          f"expected resolved_at=60, n_eff=50): {'PASS' if ok_up else 'FAIL'}")
    print(f"  synthetic: de-risking order (bar 200)   -> {derisk_event['kind'] if derisk_event else None} "
          f"at bar {derisk_event['resolved_at'] if derisk_event else None} (n_eff={derisk_event['n_eff'] if derisk_event else None}, "
          f"expected resolved_at=201, n_eff=1): {'PASS' if ok_down else 'FAIL'}")
    return ok_up and ok_down


def causality_probe(df: pd.DataFrame, config: DirectionalConfig, market) -> bool:
    """Tamper probe, R-77 novel's own pattern (two opposite tampers from a
    cut bar onward; every order whose deadline is at or before the cut
    must fill identically under both). Plus the N_add=1 identity check
    (n_add=1, n_derisk=1 collapses to the as-shipped, always-taker
    baseline byte-for-byte) and the synthetic directional-split proof
    above, which is specific to THIS round's own new code (R-77 novel's
    causality probe instead had a synthetic ratio-peek check, specific to
    ITS own new code -- the stress-ratio computation -- which this file
    does not add anything new to and therefore does not re-test).
    """
    cut = len(df) - 5_000
    print(f"\ncausality probe: frame={len(df)} bars, cut at bar {cut} ({df.index[cut]})")

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    ok = True
    res_up, diag_up = run_adaptive_backtest(up, market, TAKER_ENTRY, MAKER_ENTRY, config, 1_000.0)
    res_down, diag_down = run_adaptive_backtest(down, market, TAKER_ENTRY, MAKER_ENTRY, config, 1_000.0)
    pre_cut_up = [{k: v for k, v in e.items() if k != "resolved_at"} for e in diag_up["events"]
                  if e["resolved_at"] < cut]
    pre_cut_down = [{k: v for k, v in e.items() if k != "resolved_at"} for e in diag_down["events"]
                    if e["resolved_at"] < cut]
    match_events = pre_cut_up == pre_cut_down
    fills_up = [(f.ts, f.side, round(f.qty, 8), round(f.price, 6), round(f.fee, 8))
                for f in res_up.fills if f.ts < df.index[cut]]
    fills_down = [(f.ts, f.side, round(f.qty, 8), round(f.price, 6), round(f.fee, 8))
                  for f in res_down.fills if f.ts < df.index[cut]]
    match_fills = fills_up == fills_down
    print(f"  pre-cut order events identical under up/down tamper: {match_events} ({len(pre_cut_up)} events)")
    print(f"  pre-cut fills identical under up/down tamper: {match_fills} ({len(fills_up)} fills)")
    ok = ok and match_events and match_fills

    diverges_after = round(res_up.equity.iloc[-1], 2) != round(res_down.equity.iloc[-1], 2)
    print(f"  post-cut final equity differs (proves the probe isn't vacuous): {diverges_after} "
          f"(up=${res_up.equity.iloc[-1]:,.2f} down=${res_down.equity.iloc[-1]:,.2f})")
    ok = ok and diverges_after
    _count("diagnostic", 2)   # up, down

    identity_cfg = DirectionalConfig(n_add=1, n_derisk=1)
    base = run_taker_baseline(df, None, None, market, TAKER_ENTRY, data_label="probe")
    lim1, _ = run_adaptive_backtest(df, market, TAKER_ENTRY, MAKER_ENTRY, identity_cfg, 1_000.0)
    n1_match = round(base.equity.iloc[-1], 6) == round(lim1.equity.iloc[-1], 6)
    print(f"  identity (n_add=1,n_derisk=1) reduces exactly to the taker baseline: {n1_match} "
          f"(baseline=${base.equity.iloc[-1]:,.6f} directional=${lim1.equity.iloc[-1]:,.6f})")
    ok = ok and n1_match
    _count("diagnostic", 2)   # baseline run, identity run

    direction_ok = _synthetic_direction_check()
    ok = ok and direction_ok
    _count("diagnostic", 2)   # the two synthetic events double as two checks

    print(f"\nCAUSALITY PROBE: {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================ 2. core validation matrix
def core_matrix(df: pd.DataFrame) -> list[dict]:
    """N_add(4) x market(2) x period(2), entry fee tier -- the pre-
    registered decision matrix. 16 core + 4 baseline = 20 configurations."""
    print("=" * 100)
    print("CORE MATRIX -- N_add in {72,144,216,288} x {spot,futures_5x} x {inner-train,inner-val}, entry tier")
    print("=" * 100)
    rows = []
    for mname, market in MARKETS.items():
        for pname, (start, end) in (("inner-train", INNER_TRAIN), ("inner-val", INNER_VAL)):
            base = run_taker_baseline(df, start, end, market, TAKER_ENTRY, data_label="real")
            base_m = compute_metrics(base)
            print(f"\n-- {mname} / {pname} / entry tier --")
            base_row = _row("BASELINE taker-only", base_m)
            _print_row(base_row)
            _count("core")
            rows.append(dict(base_row, market=mname, period=pname, tier="entry", kind="baseline", n_add=None))

            for n_add in N_ADD_GRID:
                cfg = DirectionalConfig(n_add=n_add)
                res, diag = run_adaptive_period(df, start, end, market, TAKER_ENTRY, MAKER_ENTRY, cfg,
                                                 data_label="real")
                m = compute_metrics(res)
                row = _row(cfg.tag(), m, diag, base_m.fees_paid)
                _print_row(row)
                _count("core")
                row.update(market=mname, period=pname, tier="entry", kind="directional", n_add=n_add,
                           sharpe_delta=m.sharpe - base_m.sharpe)
                print(f"    sharpe_delta = {row['sharpe_delta']:+.3f}")
                rows.append(row)
    return rows


# ============================================================ 3. top-fee-tier robustness spot-check
def top_tier_check(df: pd.DataFrame) -> list[dict]:
    """N_add=288 (most extreme -- if fee-tier sensitivity matters anywhere,
    it matters most here), both markets, inner-validation only, top fee
    tier. 2 core + 2 baseline = 4 configurations."""
    print("\n" + "=" * 100)
    print("TOP-TIER ROBUSTNESS SPOT-CHECK -- N_add=288, both markets, inner-val, top fee tier")
    print("=" * 100)
    rows = []
    start, end = INNER_VAL
    cfg = DirectionalConfig(n_add=288)
    for mname, market in MARKETS.items():
        base = run_taker_baseline(df, start, end, market, TAKER_TOP, data_label="real")
        base_m = compute_metrics(base)
        print(f"\n-- {mname} / inner-val / top tier --")
        base_row = _row("BASELINE taker-only", base_m)
        _print_row(base_row)
        _count("core")
        rows.append(dict(base_row, market=mname, period="inner-val", tier="top", kind="baseline", n_add=None))

        res, diag = run_adaptive_period(df, start, end, market, TAKER_TOP, MAKER_TOP, cfg, data_label="real")
        m = compute_metrics(res)
        row = _row(cfg.tag(), m, diag, base_m.fees_paid)
        _print_row(row)
        _count("core")
        row.update(market=mname, period="inner-val", tier="top", kind="directional", n_add=288,
                    sharpe_delta=m.sharpe - base_m.sharpe)
        rows.append(row)
    return rows


# ============================================================ 4. falsification
def falsification() -> list[dict]:
    """N_add(4) x {ETH-falsification, BTC-control}, spot only, entry tier.
    8 core + 4 baseline = 12 configurations."""
    print("\n" + "=" * 100)
    print("FALSIFICATION -- ETH (Bitfinex, pre-2020) + BTC control (Bitfinex, pre-2020), spot, entry tier")
    print("=" * 100)
    eth = load_ohlcv_csv(DATA_DIR / "ethusd_bitfinex_5m.csv.gz")
    btc = load_ohlcv_csv(DATA_DIR / "btcusd_bitfinex_5m.csv.gz")
    assert eth.index.max() < pd.Timestamp("2020-01-01", tz="UTC")
    assert btc.index.max() < pd.Timestamp("2020-01-01", tz="UTC")

    rows = []
    for dname, dset in (("ETH-falsification", eth), ("BTC-control", btc)):
        from tradebot.engine import run_backtest
        base_res = run_backtest(KellyRegimeV4(), dset, replace(SPOT, fee_rate=TAKER_ENTRY),
                                 1_000.0, data_label=dname)
        base_m = compute_metrics(base_res)
        print(f"\n-- {dname} / spot --")
        base_row = _row("BASELINE taker-only", base_m)
        _print_row(base_row)
        _count("core")
        rows.append(dict(base_row, dataset=dname, market="spot", kind="baseline", n_add=None))

        for n_add in N_ADD_GRID:
            cfg = DirectionalConfig(n_add=n_add)
            res, diag = run_adaptive_period(dset, None, None, SPOT, TAKER_ENTRY, MAKER_ENTRY, cfg,
                                             data_label=dname)
            m = compute_metrics(res)
            row = _row(cfg.tag(), m, diag, base_m.fees_paid)
            _print_row(row)
            _count("core")
            row.update(dataset=dname, market="spot", kind="directional", n_add=n_add,
                        sharpe_delta=m.sharpe - base_m.sharpe)
            print(f"    sharpe_delta = {row['sharpe_delta']:+.3f}")
            rows.append(row)
    return rows


# ============================================================ 5. crash-transition-lag (the round's thesis)
def crash_lag_all(df: pd.DataFrame) -> list[dict]:
    """N_add(4) on the combined inner-train+inner-validation window, spot,
    entry tier. Pre-registered PREDICTION: all four cells are identical
    (de-risking never depends on n_add) -- 4 configurations, and this
    function checks that prediction explicitly rather than just reporting
    numbers."""
    print("\n" + "=" * 100)
    print("CRASH-TRANSITION-LAG -- N_add in {72,144,216,288}, combined window, spot, entry tier")
    print("=" * 100)
    start, end = COMBINED_WINDOW
    all_rows = {}
    for n_add in N_ADD_GRID:
        cfg = DirectionalConfig(n_add=n_add)
        rows, diag = crash_lag_for_config(df, cfg, SPOT, start=start, end=end)
        _count("core")
        all_rows[n_add] = rows
        if not rows:
            print(f"  N_add={n_add}: no flip-to-flat events resolvable in this window")
            continue
        lags = [r["lag_bars"] for r in rows]
        violations = [r for r in rows if r["lag_bars"] > 2]
        print(f"  N_add={n_add:>4d}: events={len(rows):>3d}  mean_lag={np.mean(lags):>5.2f} bars  "
              f"max_lag={max(lags):>3d} bars  violations(>2)={len(violations):>3d}/{len(rows)}")

    keys = [n for n in N_ADD_GRID if all_rows.get(n)]
    identical = True
    if len(keys) >= 2:
        ref = [(r["event_bar"], r["lag_bars"], r["kind"]) for r in all_rows[keys[0]]]
        for n_add in keys[1:]:
            cur = [(r["event_bar"], r["lag_bars"], r["kind"]) for r in all_rows[n_add]]
            if cur != ref:
                identical = False
    print(f"\nPRE-REGISTERED PREDICTION CHECK -- all N_add cells produce identical crash-lag events "
          f"(de-risking is independent of n_add by construction): {'CONFIRMED' if identical else 'VIOLATED'}")
    if not identical:
        print("  VIOLATED would indicate a bug in this file's own decision rule or fill loop -- "
              "investigate before trusting any other result in this run.")
    return [r for rows in all_rows.values() for r in rows]


# ============================================================ 6. thesis reference (R-77's own N<=24 baseline)
def thesis_reference(df: pd.DataFrame) -> dict:
    """R-77's OWN fixed-N=24 configuration (`fixed_n_config(24)`, imported
    unmodified), re-measured HERE on the SAME combined window with THIS
    file's own `crash_lag_for_config` call, for a same-methodology
    violation-rate reference point for criterion (d). One configuration."""
    print("\n" + "=" * 100)
    print("THESIS REFERENCE -- R-77's own fixed N=24, re-measured here, combined window, spot, entry tier")
    print("=" * 100)
    start, end = COMBINED_WINDOW
    ref_cfg = fixed_n_config(24)
    rows, diag = crash_lag_for_config(df, ref_cfg, SPOT, start=start, end=end)
    _count("core")
    if not rows:
        print("  no flip-to-flat events resolvable in this window")
        return {"n": 24, "events": 0, "mean_lag": 0.0, "violations": 0}
    lags = [r["lag_bars"] for r in rows]
    violations = sum(1 for lag in lags if lag > 2)
    mean_lag = float(np.mean(lags))
    print(f"  fixed N=24 (R-77's own): events={len(rows)}  mean_lag={mean_lag:.2f}  violations(>2)={violations}")
    return {"n": 24, "events": len(rows), "mean_lag": mean_lag, "violations": violations}


# ============================================================ main
def main() -> None:
    df, label = load_working_frame()
    assert df.index[-1] < pd.Timestamp(OOS_START, tz="UTC"), (
        "data discipline violated: a holdout bar leaked into the working frame "
        "(redundant re-assertion of R-77 novel's own load_working_frame guard)")
    print(f"{len(df):,} bars  {df.index[0]} -> {df.index[-1]}  (data: {label})")
    print(f"restricted to inner-train {INNER_TRAIN} and inner-validation {INNER_VAL}\n")

    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    default_cfg = DirectionalConfig(n_add=72)

    if choice in ("causality", "all"):
        probe_df = df.iloc[-160_000:]
        causality_probe(probe_df, default_cfg, SPOT)

    if choice in ("core", "all"):
        core_matrix(df)

    if choice in ("toptier", "all"):
        top_tier_check(df)

    if choice in ("falsify", "all"):
        falsification()

    if choice in ("crashlag", "all"):
        crash_lag_all(df)

    if choice in ("thesisref", "all"):
        thesis_reference(df)

    print(f"\nCONFIGS EVALUATED THIS RUN: core={CONFIG_COUNTER['core']} "
          f"diagnostic={CONFIG_COUNTER['diagnostic']} "
          f"total={CONFIG_COUNTER['core'] + CONFIG_COUNTER['diagnostic']}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[{time.time() - t0:.0f}s]")
