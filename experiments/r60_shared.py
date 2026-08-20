"""R-60: does changing `kelly_regime_v4`'s vote/gate TIMING — rather than
its exposure SCALE, which R-59 tested twice and twice found not to be the
binding constraint — restore the matched-exposure drawdown property on
R-57's six-asset panel? (backlog **B-26**, filed by R-59.)

Shared, frozen infrastructure for a two-branch parallel round. Per
ROUTINE.md's parallelism rules this file is neutral ground: both branches
import from it, neither branch edits it, and it does not itself run any
backtest. It exists so the pre-registration below is committed once, before
either branch reads a single strategy number on the panel.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **SIZE** (this strategy family's vote/gate mechanism)
and **N≈3** (the panel is still the only route past two correlated
instruments).

R-57 found `kelly_regime_v4`'s one surviving property — a matched-exposure
drawdown advantage — is present on BTC and ETH and inverts on 6 of 6 further
Coinbase instruments (BCH, LTC, ETC, DASH, LINK, XTZ). R-59 tested R-57's own
named hypothesis on the SCALE axis twice — per-asset `target_vol`
calibration (conservative) and a self-normalizing, dimensionless relative-vol
scale (novel) — and both failed identically (D1 0/6, both branches), while
both cleanly passed the BTC/ETH control. R-59's own write-up localizes the
failure: neither the sizing constant's magnitude nor its dimensional form is
the binding constraint, and both branches independently converge on R-57's
own alternative explanation — the matched hold's advantage on these
higher-volatility, more mean-reverting instruments looks like a
buy-the-dip effect the panel's own price dynamics reward, and
`kelly_regime_v4`'s vote-gated trend rule stands aside after drops the
matched hold is quietly buying. Neither R-59 branch touched the vote's
*timing* — only its *scale*. This round asks the question R-59 filed as
B-26: does the vote arriving faster (or asset-adaptively) let the strategy
participate in more of those dips, rather than a bigger or differently-
shaped notional doing it?

**Caution, carried forward from B-26's own filing.** This is the strategy
family's SIZE-axis attempt number **twenty** (R-34 → R-46, R-53 → R-56,
R-59 × 2, all NEGATIVE — see LEDGER.md section C and R-59's own count of
0-for-19). R-57 itself suggested a genuinely different strategy family run
on the panel is at least as promising as a twentieth vote/gate variant. This
round proceeds anyway because, unlike the prior nineteen, it is the first to
vary *timing* rather than *scale* — a materially different axis on the same
family, not a further division of one already-exhausted axis — and because
it is cheap: no new data, the panel/control/harness are already built, and a
NEGATIVE result here is itself informative (it would mean the family's SIZE
axis, both scale and timing, is exhausted, sharpening R-57's own
recommendation into a near-certainty rather than a suggestion).

**Not a duplicate of R-59** (scale, not timing) **or of R-57 itself** (pure
replication, nothing about the strategy changes). **Not a duplicate of any
prior anchor-horizon round**: R-07 (cited by `kelly_regime_v4`'s own
docstring) swept horizons 18-28 days on BTC alone, evaluated only on
BTC/ETH, never asset-adaptively and never against the panel. This round is
the first to make the anchor *cadence* — not just its calendar length — a
function of which instrument is being traded, and the first to try a
non-moving-average timing mechanism (CUSUM) on this strategy family at all.

**Simulable here**: yes. Reuses R-57's committed panel loaders
(`tradebot.data.load_coinbase_spot`), R-57's matched-exposure harness
(`experiments/matched_hold.py`'s `ConstantExposureHold`/`mean_notional`,
`experiments/r57_cross_asset_panel.py`'s `measure`/`solve_c`/
`binomial_tail`), and `tradebot.inference`. No new data.

**What would make this fail, named now, before either branch is coded**: if
neither branch restores a majority (>= 5 of 6) matched-exposure drawdown
advantage on PANEL_TRAIN, OR if a timing change regresses v4's own
already-established BTC/ETH control numbers (R-57's cells: BTC dDD -5.6pp
[-20.0,+16.4], ETH -11.5pp [-17.3,+19.6]) by more than 5 percentage points
on either asset, OR if a branch delays a crash flip-to-flat beyond the
project's three marquee crash windows relative to v4's own baseline lag
(the specific risk R-56 found a "clean-looking-in-aggregate" change can
hide). Any of these is NEGATIVE and closes B-26, extending this family's
SIZE-axis record to 0-for-20 or 0-for-21.

=====================================================================
LITERATURE (Step 2 sources, read and cited before either branch was coded)
=====================================================================

- Page, E. S. (1954, "Continuous Inspection Schemes", Biometrika 41(1/2),
  100-115) — the CUSUM sequential change-point test: accumulate deviations
  from an expected value and signal the first time the running sum crosses
  a threshold. Sequential and causal by construction (a running statistic
  updated bar-by-bar, no batch reprocessing, no future data), unlike a
  fixed-horizon moving-average crossing which only detects a regime change
  once enough of the old regime has rolled out of the window. This is the
  literature basis for the novel branch's vote-timing mechanism: replace
  the (20,40,80)-day anchor-crossing vote with a CUSUM statistic on
  log-returns that latches bullish/bearish on a threshold breach.
- Ornstein-Uhlenbeck half-life estimation via AR(1) regression (standard
  quant-finance calibration method; see e.g. Chan, E. (2013), *Algorithmic
  Trading: Winning Strategies and Their Rationale*, Wiley, ch. 2, and the
  underlying Ornstein & Uhlenbeck (1930, Phys. Rev. 36) mean-reverting
  diffusion) — fit close_t - close_(t-1) = theta * (mu - close_(t-1)) + eps
  by OLS on log-price, giving a mean-reversion speed theta and a half-life
  ln(2)/theta specific to one instrument's own price history. This is the
  literature basis for the conservative branch: rescale v4's fixed
  (20,40,80)-day anchor ladder by each instrument's own OU half-life
  relative to BTC's, a structural (data-derived, not backtest-fit)
  per-asset constant, in the same spirit as R-59's conservative branch but
  applied to the timing axis instead of the scale axis.
- Baltas & Kosowski (2013/2017, "Demystifying Time-Series Momentum
  Strategies", J. Investment Management) and Moskowitz, Ooi & Pedersen
  (2012, "Time Series Momentum", J. Financial Economics) — already cited by
  R-59; the general principle that a cross-instrument strategy should adapt
  to each instrument's own measured dynamics rather than sharing one
  constant carries over from the scale axis (R-59) to the timing axis
  (this round) without re-derivation.
- R-56 (this ledger) — the specific methodological lesson this round
  inherits directly: a change that looks clean in inner-validation
  aggregate can still be quietly wrong at exactly the moments (crash
  de-risking) that make up the whole strategy's edge, and only an explicit
  crash-transition-lag check surfaces that. Both branches below must run
  one.

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
CRASH_WINDOWS  Nov 2018, COVID Mar 2020, FTX Oct/Nov 2022 (BTC) — the same
                                        three marquee windows R-56 used, for
                                        the crash-transition-lag check.

Same convention as R-57/R-59: reading panel-asset data (train or test)
costs the program's BTC/ETH holdout counter **+0** — none of these are the
reserved BTC/ETH 2023+ holdout, they are new-instrument evidence. The three
crash windows are all pre-2023 and are read by every registered strategy's
own backtest already, so they cost +0 too.

=====================================================================
ARMS, COSTS, METHODOLOGY (identical to R-57/R-59's D1, mean-notional axis)
=====================================================================

Arms: candidate strategy; `buy_and_hold`; `ConstantExposureHold(c =
candidate's own mean clipped notional over the SAME window/asset/market)` —
the mean-notional-matched hold, R-57's primary axis. The equal-realized-
volatility axis is not re-run here (R-57/R-59 already established both axes
agree in direction on this panel); reused only if a branch's PANEL_TRAIN
result is ambiguous enough to need it, and any such use is reported as
robustness, not the decision rule.

Costs: spot 0.10% primary (`SPOT_BASE`), spot 0.40% Bitstamp falsification
tier (`SPOT_REAL`) — both re-exported below from `r57_cross_asset_panel`.

=====================================================================
DECISION RULES, FROZEN (default is REJECT)
=====================================================================

D1 (PRIMARY). On PANEL_TRAIN, spot @0.10%: count panel assets (of 6) where
the candidate's max drawdown is strictly below the mean-notional-matched
hold's. Same binomial convention as R-57/R-59 (n=6, one-sided, p=0.5 null):
    6/6  -> REPLICATES     (p = 0.0156)
    5/6  -> SUGGESTIVE, not established (p = 0.109)
    <=4/6 -> FAILS

D2 (FALSIFICATION, chosen now). CONTROL window, BTC and ETH: the
candidate's matched-exposure drawdown advantage must not be worse than
v4's own R-57 control numbers (BTC -5.6pp, ETH -11.5pp) by more than 5
percentage points on either asset. A fix for six instruments that breaks
the two instruments the mechanism already works on is not a fix.

D3 (CRASH-TRANSITION-LAG, chosen now, per R-56's lesson). BTC, the three
CRASH_WINDOWS above: measure bars-to-flatten from the candidate's own
flip-to-flat signal vs. v4's unmodified baseline signal, same window, same
market (spot). PASS requires the candidate's mean lag across the three
windows does not exceed v4's own baseline lag by more than 2 bars (10
minutes) — a change whose faster-elsewhere timing quietly slows down
exactly the de-risking events that are the strategy's edge is a failure
regardless of what D1 says.

D4 (GENERALIZATION CHECK, reported, not a gate). PANEL_TEST, spot @0.10%,
same D1 methodology on the panel's 2023-2026 window. Descriptive: the
configuration was frozen on PANEL_TRAIN only, so a pass here is
corroborating, a fail is informative but does not by itself reverse a D1
promotion (n=6 assets x 1 window, same small-n caveat as D1 itself).

D5 (0.40% FALSIFICATION, from ROUTINE step 2's menu). PANEL_TRAIN, spot
@0.40%: candidate beats `buy_and_hold`'s final balance in >= 5 of 6 panel
assets. PREDICTION, recorded now: FAILS — R-13, R-47, R-57's own D2 and
R-59's own D4 all say the return edge does not survive the real entry tier,
and nothing in this round changes the strategy's return mechanism, only
its risk timing.

PROMOTION BAR: D1 >= 5/6 AND D2 passes on BOTH BTC and ETH AND D3 passes.
Anything else is NEGATIVE. A branch that clears the bar is a candidate for
further work (inner-validation on BTC/ETH's own pre-2023 data, then — only
if that also clears — a holdout consultation); this round's own
pre-registration does NOT authorize a BTC/ETH 2023+ holdout read, because
the promotion bar above must be cleared on the panel, the control and the
crash-lag check before that consultation is worth spending.

Configurations evaluated: counted per branch via the same `measure()`-style
counter pattern as R-57/R-59, reported honestly including any solver/fit
iterations, and summed across both branches for the round's total (per
ROUTINE.md: parallel trials count is the total across all branches, not per
branch).

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
    measure,
    realized_vol,
    select_panel,
    solve_c,
)

PANEL_TRAIN = ("2020-04-01", "2022-12-31")
PANEL_TEST = ("2023-01-01", "2026-08-20")
CONTROL = ("2020-04-01", "2022-12-31")

# The three marquee crash windows used throughout this ledger (R-19, R-28,
# R-56) for a crash-transition-lag check. All pre-2023: reading them costs
# the BTC/ETH holdout counter +0.
CRASH_WINDOWS = {
    "2018-11": ("2018-11-01", "2018-12-15"),
    "2020-03-covid": ("2020-02-15", "2020-04-15"),
    "2022-11-ftx": ("2022-10-15", "2022-12-15"),
}

# R-57's own recorded control-cell results (LEDGER.md R-57), cited here as
# already-published evidence, not a new backtest read.
R57_CONTROL_DD_ADVANTAGE = {"BTC": -5.6, "ETH": -11.5}  # percentage points
D2_REGRESSION_TOLERANCE_PP = 5.0
D3_MAX_EXTRA_LAG_BARS = 2  # 2 bars = 10 minutes at 5m resolution


def load_panel() -> list:
    """The frozen six-asset panel R-57 selected, loaded the identical way."""
    return select_panel(load_candidates())


def d1_verdict(k: int, n: int = 6) -> str:
    if k == n:
        return "REPLICATES"
    if k == n - 1:
        return "SUGGESTIVE (not established)"
    return "FAILS"


def d2_passes(candidate_dd_advantage: dict) -> bool:
    """candidate_dd_advantage: {'BTC': dDD_pp, 'ETH': dDD_pp}, same sign
    convention as R-57/R-59 (negative = candidate draws down less than the
    matched hold, i.e. better)."""
    for ticker, base in R57_CONTROL_DD_ADVANTAGE.items():
        cand = candidate_dd_advantage[ticker]
        if cand > base + D2_REGRESSION_TOLERANCE_PP:
            return False
    return True


def d3_passes(candidate_mean_lag_bars: float, baseline_mean_lag_bars: float) -> bool:
    """Candidate's mean crash-transition flip-to-flat lag, across the three
    CRASH_WINDOWS, must not exceed the baseline's by more than
    D3_MAX_EXTRA_LAG_BARS."""
    return candidate_mean_lag_bars <= baseline_mean_lag_bars + D3_MAX_EXTRA_LAG_BARS


def promoted(k1: int, dd_advantage: dict, candidate_lag: float, baseline_lag: float,
             n: int = 6) -> bool:
    return (
        k1 >= n - 1
        and d2_passes(dd_advantage)
        and d3_passes(candidate_lag, baseline_lag)
    )
