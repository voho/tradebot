"""R-60: does changing `kelly_regime_v4`'s vote/gate TIMING — rather than its
exposure SCALE, which R-59 tested twice and twice found not to be the binding
constraint — restore the matched-exposure drawdown property on R-57's
six-instrument panel? (backlog **B-26**, filed by R-59.)

Shared, frozen infrastructure for a two-branch parallel round. Per
ROUTINE.md's parallelism rules this file is neutral ground: both branches
import from it, neither branch edits it, and it does not itself run any
backtest. It exists so the pre-registration below is committed once, before
either branch reads a single strategy number on the panel.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **SIZE** (this strategy family's vote/gate mechanism)
and **N≈3** (the panel is still the only route past two correlated assets).

R-59 tested B-25's own hypothesis twice — a per-asset-calibrated
`target_vol` (conservative) and a self-normalizing, dimensionless relative-
volatility scale (novel) — and both failed to restore the matched-exposure
drawdown property on the panel (D1 0/6, both branches), while both cleanly
passed the falsification control confirming the fix does not regress the
BTC/ETH numbers the mechanism already works on. R-59's own write-up
localizes the failure: not the sizing constant's magnitude, not its
dimensional form — both branches converge independently on the same
alternative, that a rebalanced constant-exposure hold behaves like a
buy-the-dip rule on these higher-volatility, more mean-reverting
instruments, and neither branch touched the vote/gate's *timing*, only its
*scale*. B-26 asks the question R-59 explicitly left open: does the timing
axis fare any better?

**Not a duplicate of nineteen prior SIZE-axis rounds** (R-34, R-37, R-38,
R-40, R-41, R-45, R-46, R-53–R-56, R-59×2 — every one NEGATIVE, see
LEDGER.md section C): every one of those retuned the exposure *scale* —
`target_vol`, `max_leverage`, a sizing formula, a haircut multiplier, a CPPI
cushion — while leaving the three-anchor vote's horizons (20/40/80 fixed
calendar days) and hysteresis (1% band, latched) exactly as v4 shipped them.
None varied *when* the vote flips, only *how much* exposure the flipped vote
buys. This round is the first to vary the vote/gate's timing at all, on any
axis, temporal or cross-sectional. Not a duplicate of R-57 (changed nothing)
or R-59 (changed only scale terms).

**Simulable here**: yes. Reuses R-57's committed panel loaders
(`tradebot.data.load_coinbase_spot`), R-57's matched-exposure harness
(`experiments/matched_hold.py`'s `ConstantExposureHold`/`mean_notional`,
`experiments/r57_cross_asset_panel.py`'s bootstrap/binomial helpers), R-59's
D1/D2 structure, and `tradebot.inference`. No new data.

**What would make this fail, named now, before either branch is coded**: if
neither adaptive-timing mechanism restores a majority (>= 5 of 6)
matched-exposure drawdown advantage on PANEL_TRAIN, OR if fixing
generalization regresses v4's own already-established BTC/ETH
matched-exposure numbers (R-57's control cells: BTC dDD -5.6pp
[-20.0,+16.4], ETH -11.5pp [-17.3,+19.6], both vs. the mean-notional-matched
hold, 2020-04..2022-12) by more than 5 percentage points on either asset.
Either outcome is NEGATIVE and closes B-26, extending this family's
SIZE-axis record to 0-for-20 or 0-for-21. A third, structurally different
possibility, named because R-59's own branches predicted it: if both
branches instead find the panel's price dynamics reward the matched hold
regardless of the vote's timing too, that corroborates R-59's "buy-the-dip"
explanation on an axis neither prior round could rule in or out, which is
itself informative even though it is also a NEGATIVE result for B-26.

=====================================================================
LITERATURE (Step 2 sources, read and cited before either branch was coded)
=====================================================================

- Kaufman, P. (1995, "Smarter Trading", McGraw-Hill; the Efficiency Ratio
  and Kaufman's Adaptive Moving Average, KAMA) — a moving average whose
  effective smoothing constant is a function of a trailing Efficiency
  Ratio, ER = |close_t - close_{t-N}| / sum(|close_i - close_{i-1}|) over
  the trailing N bars: ER -> 1 in an efficient trend (the average tracks
  price almost immediately), ER -> 0 in noise/chop (the average slows
  toward its own long bound). Structural, thirty years old, and — the
  property that matters here — asset-agnostic by construction: the same
  formula adapts its own speed to whatever the asset's own recent price
  action looks like, with no per-asset constant to fit. This is the
  literature basis for the **conservative** branch: replace each of v4's
  three fixed-calendar-day anchors with a KAMA-style average bracketed
  around that anchor's own nominal horizon, so a choppier/more
  mean-reverting instrument's anchor naturally runs faster (shorter
  effective memory) without any asset-specific parameter being touched.
- Dai, Zhang & Zhu (2010, "Trend Following Trading under a Regime
  Switching Model", SIAM J. Financial Mathematics 1(1), 780-810) and Dai,
  Yang, Zhang & Zhu (2016, "Optimal Trend Following Trading Rules", Math.
  of Operations Research 41(2), 626-642) — in a two-state (bull/bear)
  regime-switching model, the *optimal* buy/sell trigger is a pair of
  threshold curves that are functions of the regime's own transition
  intensities: a market that switches states more often warrants a wider
  no-trade band around the belief boundary (acting on every switch is
  ruinous when switches are frequent and partly noise), a market with more
  persistent regimes warrants a narrower one. v4's 1% latch band is a
  single global constant, identical for BTC and for six other instruments
  with different regime-switching frequencies by construction (R-59: the
  panel is measurably more mean-reverting). This is the literature basis
  for the **novel** branch: derive each asset's hysteresis band from its
  own measured vote-flip frequency (the discrete-time analogue of a
  transition intensity) rather than sharing one band-width fitted
  (implicitly, by inheritance) to BTC alone.
- Corsi (2009, "A Simple Approximate Long-Memory Model of Realized
  Volatility", J. Financial Econometrics 7(2)) / Müller et al. (1997) —
  already v4's own cited basis for the fixed doubling ladder (20/40/80);
  cited again here because both branches keep that ladder's structure
  (three anchors, doubling spacing) and change only how each anchor tracks
  price or how its vote latches, not how many anchors there are or their
  nominal spacing.
- Lo & MacKinlay (1988, "Stock Market Prices Do Not Follow Random Walks:
  Evidence from a Simple Specification Test", Rev. Financial Studies 1(1))
  — the variance-ratio test for mean reversion vs. momentum in a return
  series; cited for context on why the panel's instruments plausibly flip
  their vote more often than BTC (R-59's own finding), not run as a new
  test in this round (the novel branch's own flip-rate measurement is a
  direct, simpler empirical analogue of the same question, measured on
  PANEL_TRAIN only).

=====================================================================
WINDOWS (fixed before either branch is coded, identical to R-59's)
=====================================================================

PANEL_TRAIN  2020-04-01 -> 2022-12-31   fit / select here ONLY. Includes the
                                        panel's own 2021 top and 2022 bear.
PANEL_TEST   2023-01-01 -> 2026-08-20   panel assets only, held untouched
                                        until each branch's configuration is
                                        frozen on PANEL_TRAIN. Reported as a
                                        generalization check, NOT a gate.
CONTROL      2020-04-01 -> 2022-12-31   BTC and ETH, R-57's own control
                                        window. No 2023+ BTC/ETH bar is read
                                        anywhere in this round.

Same convention as R-57/R-59: reading panel-asset data (train or test) costs
the program's BTC/ETH holdout counter **+0** — none of these are the
reserved BTC/ETH 2023+ holdout.

=====================================================================
ARMS, COSTS, METHODOLOGY (identical to R-57's D1, mean-notional axis only)
=====================================================================

Arms: candidate strategy; `buy_and_hold`; `ConstantExposureHold(c =
candidate's own mean clipped notional over the SAME window/asset/market)` —
the mean-notional-matched hold, R-57's primary axis, so each candidate is
matched against a hold carrying *its own* exposure, not v4's. The
equal-realized-volatility axis is NOT re-run here (R-57 already established
both axes agree in direction on this panel); reused only if a branch's
PANEL_TRAIN result is ambiguous enough to need it, and any such use is
reported as robustness, not as the decision rule.

Costs: spot 0.10% primary (`SPOT_BASE`), spot 0.40% Bitstamp falsification
tier (`SPOT_REAL`) — both re-exported below from `r57_cross_asset_panel`.

=====================================================================
DECISION RULES, FROZEN (default is REJECT) — identical structure to R-59
=====================================================================

D1 (PRIMARY). On PANEL_TRAIN, spot @0.10%: count panel assets (of 6) where
the candidate's max drawdown is strictly below the mean-notional-matched
hold's. Same binomial convention as R-57/R-59 (n=6, one-sided, p=0.5 null):
    6/6  -> REPLICATES     (p = 0.0156)
    5/6  -> SUGGESTIVE, not established (p = 0.109)
    <=4/6 -> FAILS

D2 (FALSIFICATION, chosen now). CONTROL window, BTC and ETH: the
candidate's matched-exposure drawdown advantage must not be worse than v4's
own R-57 control numbers (BTC -5.6pp, ETH -11.5pp) by more than 5
percentage points on either asset. A fix for six instruments that breaks
the two instruments the mechanism already works on is not a fix.

D3 (GENERALIZATION CHECK, reported, not a gate). PANEL_TEST, spot @0.10%,
same D1 methodology on the panel's 2023-2026 window. Descriptive: the
configuration was frozen on PANEL_TRAIN only, so a pass here is
corroborating, a fail is informative but does not by itself reverse a D1
promotion (n=6 assets x 1 window, same small-n caveat as D1 itself).

D4 (0.40% FALSIFICATION, from ROUTINE step 2's menu). PANEL_TRAIN, spot
@0.40%: candidate beats `buy_and_hold`'s final balance in >= 5 of 6 panel
assets. PREDICTION, recorded now: FAILS — R-13, R-47, R-57's own D2 and
R-59's D4 all say the return edge does not survive the real entry tier, and
a timing change to the vote's anchors/hysteresis does not touch the
strategy's turnover-vs-signal economics enough to expect a different
outcome here.

PROMOTION BAR: D1 >= 5/6 AND D2 passes on BOTH BTC and ETH. Anything else is
NEGATIVE. A branch that clears the bar is a candidate for further work
(inner-validation on BTC/ETH's own pre-2023 data, then — only if that also
clears — a holdout consultation); this round's own pre-registration does
NOT authorize a BTC/ETH 2023+ holdout read, because the promotion bar above
must be cleared on the panel and the control before that consultation is
worth spending.

Configurations evaluated: counted per branch via the same `measure()`-style
counter pattern as R-57/R-59, reported honestly including any fitting
iterations, and summed across both branches for the round's total (per
ROUTINE.md: parallel trials count is the total across all parallel
branches, not per branch).

Causality: each branch must run a tamper probe on its own code path
(R-57's `cmd_causality` opposite-tamper pattern — construct the strategy
directly, tamper post-cut bars in opposite directions, confirm identical
pre-cut decisions) since the strategy classes here are new and are not
covered automatically by `tests/test_causality_strict.py` (which hard-codes
the registered BTC loader).

Holdout cost: +0 (no BTC/ETH 2023+ bar is read anywhere in this round).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.matched_hold import ConstantExposureHold, mean_notional  # noqa: E402,F401
from experiments.r57_cross_asset_panel import (  # noqa: E402,F401
    RANKED_CANDIDATES,
    SPOT_BASE,
    SPOT_REAL,
    Asset,
    binomial_tail,
    load_candidates,
    realized_vol,
    select_panel,
)

PANEL_TRAIN = ("2020-04-01", "2022-12-31")
PANEL_TEST = ("2023-01-01", "2026-08-20")
CONTROL = ("2020-04-01", "2022-12-31")

# R-57's own recorded control-cell results (LEDGER.md R-57), cited here as
# already-published evidence, not a new backtest read.
R57_CONTROL_DD_ADVANTAGE = {"BTC": -5.6, "ETH": -11.5}  # percentage points
D2_REGRESSION_TOLERANCE_PP = 5.0

PANEL_TICKERS = ["BCH", "LTC", "ETC", "DASH", "LINK", "XTZ"]


def load_panel() -> list[Asset]:
    """The frozen six-asset panel R-57 selected, loaded the identical way."""
    return select_panel(load_candidates())


def d1_verdict(k: int, n: int = 6) -> str:
    if k == n:
        return "REPLICATES"
    if k == n - 1:
        return "SUGGESTIVE (not established)"
    return "FAILS"


def d2_passes(candidate_dd_advantage: dict[str, float]) -> bool:
    """candidate_dd_advantage: {'BTC': dDD_pp, 'ETH': dDD_pp}, same sign
    convention as R-57 (negative = candidate draws down less than the
    matched hold, i.e. better)."""
    for ticker, base in R57_CONTROL_DD_ADVANTAGE.items():
        cand = candidate_dd_advantage[ticker]
        if cand > base + D2_REGRESSION_TOLERANCE_PP:
            return False
    return True


def promoted(k1: int, dd_advantage: dict[str, float], n: int = 6) -> bool:
    return k1 >= n - 1 and d2_passes(dd_advantage)
