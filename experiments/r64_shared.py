"""R-64: the holding-period axis -- can the ONE signal this project has
priced be bought at a price it is worth?

Shared, frozen infrastructure for a two-branch parallel round. Per
ROUTINE.md's parallelism rules this file is neutral ground: both branches
import from it, NEITHER BRANCH EDITS IT, and it does not itself define a
candidate strategy or compute a verdict. It exists so the pre-registration
below is committed once, before either branch reads a single candidate
number.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **COST** (primary), INFO (secondary).

COST is stated in the standing diagnosis as "costs scale *with* the
signal". Every previous attack on it has tried to make the *cost* smaller:
L-05/L-06 derived a no-trade band on a single asset's exposure fraction
(Constantinides 1986; Davis & Norman 1990), R-13/R-12 studied fee tiers and
turnover reduction, R-56 tried to pay maker rather than taker. None of them
had a signal whose value was independently known, so none could say whether
the surviving turnover was worth its price -- they could only observe that
the net number was bad.

**R-63 changed that.** It is the first round in this project to *price* a
signal against its own cost rather than merely watch it lose:

- the cross-sectional trend ranking is **genuinely real** -- frictionless
  it scores +0.480 log units against scrambles spanning -1.88 to +0.175
  (p90 +0.016) and beats **10 of 10** permutations of its own weights;
- it is **unaffordable by 16 to 1** -- at k=1 the top-ranked asset changes
  **2.86 times per day** (3.44 round-trip turnover/day), an implied drag of
  0.00344 log/day, which over 2,332 days is **8.02 log units**, exactly the
  gap between the frictionless +0.480 and the charged -7.537.

R-63's own closing line names the two bars for reopening this axis: a
universe whose *breadth* is materially above 1.47, **or a holding period
long enough that 2.86 leader-changes per day stops being the binding
cost.** The first is not available from these eight assets. The second is,
and it has never been varied as a first-class design axis anywhere in this
project: every strategy in section A decides at every 5-minute bar close
and is banded only on exposure *magnitude*, never on decision *frequency*.

The round in one sentence: **measure the value and the cost of a known-real
signal as functions of holding period, and find out whether the two curves
cross.**

=====================================================================
WHY THIS IS NOT A DUPLICATE
=====================================================================

- **L-05 / L-06 (`kelly_regime_ev`, `kelly_regime_ev_fast`).** Those band a
  *continuous single-asset exposure fraction* -- how much BTC to hold --
  and the banded quantity is a scalar with a natural metric. The quantity
  here is a *discrete cross-sectional selection*: which asset is held. A
  deadband on "how much" says nothing about "which", and R-63's turnover
  bill is entirely the "which" (its exposure magnitude already carries v4's
  shipped 0.10 deadband, and still turned over 3.44x/day).
- **R-56 (maker/limit execution).** Reduces the price paid per unit of
  turnover. This round reduces the number of units. Orthogonal, and R-56's
  own conservative branch failure modes (N>=72 near-miss, trend-drift) do
  not apply to a rule that never posts an order.
- **R-52 / R-51 (calendar and drift-band rebalancing of a fixed 50/50
  BTC+ETH split).** Those cadences rebalance *weights toward a fixed
  target* and contain no signal in the rebalance decision at all. Here the
  cadence IS the signal-consumption rate.
- **R-63 novel arm.** Swept `k` (how many assets), never a holding period,
  a hysteresis band or a trading rate. Its k-sweep slope toward
  "hold everything" is a statement about concentration, not about time.
- **R-59 / R-60 / R-62.** SIZE-axis retunes of `frac` or `scale`. This
  round changes neither: both branches consume R-63's frozen score
  unchanged and alter only WHEN the portfolio acts on it.

=====================================================================
THE TWO BRANCHES
=====================================================================

Both start from R-63's novel arm **byte-for-byte** (its score, its vol
scale, its deadband, its constants, all re-exported from this file so the
two branches provably share them) and change only the trading rule.

**CONSERVATIVE -- rank buffering plus a minimum holding period**
(`experiments/r64_conservative_rank_buffer.py`). The standard practitioner
mitigation, and the one the cost literature says works best. An incumbent
holding is retained until a challenger beats it by a margin, and is never
swapped before H bars have elapsed. Two parameters (`buffer`, `H`), swept
on W_TRAIN and selected on W_VAL before any D-cell is touched.

**NOVEL -- partial adjustment toward an aim portfolio**
(`experiments/r64_novel_aim_portfolio.py`). Garleanu & Pedersen's (2013)
closed form: with quadratic costs and mean-reverting signals the optimal
policy is not "trade to the target" but "trade a constant fraction of the
way toward an *aim* portfolio", where the fraction is set by the signal's
own decay rate and the cost. Structurally different from buffering: it
never makes a discrete swap, it smooths rather than filters, and its
trading rate is **derived from a measured signal half-life rather than
fitted**.

=====================================================================
THE FRONTIER -- THE ROUND'S OUTPUT REGARDLESS OF VERDICT
=====================================================================

Every configuration either branch evaluates emits the same row via
:func:`frontier_row`: turnover per day, implied cost drag, GROSS (0 bps)
log growth against the benchmark, and NET (10 bps) log growth against the
same benchmark. Two branches x a grid of trading rates traces the value and
cost curves R-63 measured at exactly one point. **That frontier is this
round's deliverable whether or not anything is promoted**, because it says
what a 16-to-1 deficit costs to close and whether closing it is even
geometrically possible.

The 0 bps column is a DIAGNOSTIC, never a decision cell. It exists to
separate the two ways a buffered arm can look better: by keeping the signal
and paying less for it (the hypothesis), or by destroying the signal and
therefore having less to pay for (the null). D5 below is what enforces
that distinction.

=====================================================================
THE FROZEN DECISION RULES
=====================================================================

Windows are R-63's, unchanged, so the frontier's endpoint is comparable to
the number it extends:

    W_TRAIN  2020-04-01 -> 2021-12-31   sweep, fit, iterate freely
    W_VAL    2022-01-01 -> 2022-12-31   select between configurations
    W_FULL6  2020-04-01 -> last bar     the decision window (U6 only)
    W_HOLD   2023-01-01 -> last bar     NOT READ BY EITHER BRANCH

The benchmark. R-63 filed a correction against its own convention and this
round is the first to implement it: `MATCHED_HOLD` matches the exposure
*level* (the candidate's own mean total notional), which is the standing
R-33 rule and is **not** a risk match for a *concentrated* candidate -- on
R-63's D1 cell the novel arm ran 86.5% annualized volatility against
MATCHED_HOLD's 42.9% at an identical 0.525 mean notional. Both arms here
are concentrated by construction, so the primary benchmark is
**`VOLMATCH_HOLD`**: an equal-weight basket held at the constant notional
whose *realized volatility* matches the candidate's own (R-31's
convention). `MATCHED_HOLD` is reported alongside it for continuity with
R-63, and `EW_HOLD` / `BTC_HOLD` as context.

    D1  W_FULL6, 0.10%, growth vs VOLMATCH_HOLD:
        point > 0 AND the 95% bootstrap interval excludes zero.
    D2  W_FULL6, 0.10%, max drawdown vs VOLMATCH_HOLD:
        point < 0 AND the 95% bootstrap interval excludes zero.
    D3  W_VAL on U8, 0.10%: growth > 0 AND drawdown < 0 vs VOLMATCH_HOLD.
    D4  W_FULL6, 0.40% (the falsification tier): beats EW_HOLD outright.
    D5  SIGNAL RETENTION, this round's own pre-registered falsification
        test and the one that matters. At 0 bps the arm must retain at
        least **half** of R-63's measured frictionless edge -- i.e. gross
        growth vs VOLMATCH_HOLD >= +0.240 log units on W_FULL6.
        Without D5 an arm can pass D1 by trading so rarely that it becomes
        the benchmark; "converges to a hold" is the null hypothesis of this
        entire round and D5 is what rejects it.
    SCRAMBLE  R-63's control, unchanged: same weights, same sizes, same
        times, wrong assets, 10 fixed seeds. Survives only if the
        candidate's D1 point estimate exceeds the scrambles' 90th
        percentile.

    FURTHER-WORK BAR = (D1 or D2) and D3 and D5 and scramble_survived.

Clearing it authorizes exactly ONE holdout read on W_HOLD, which is a
further-work bar and NOT the promotion bar in ROUTINE.md step 4.

=====================================================================
WHAT WOULD MAKE THIS FAIL -- NAMED NOW, BEFORE ANY CODE
=====================================================================

**(F1) The ceiling is already below the bar.** R-63's frictionless +0.480
carries the interval [-2.58, +3.65], which contains zero. A perfect cost
fix -- turnover driven to zero at no loss of signal -- lands on an edge
that was never significant. If D5 is calibrated at half of a
non-significant number, then an arm can pass D5 and still fail D1 on width
alone. **This is the predicted outcome for both branches** and it is
recorded here so that a pass reads as a genuine surprise rather than as
confirmation.

**(F2) Buffering trades signal for turnover at worse than 1:1.** The
selection is the signal. A margin large enough to cut 2.86 leader-changes
per day to 0.15 may well discard most of the information along with the
trades, in which case the frontier is monotone and never crosses. Novy-Marx
& Velikov's own result is that mitigation *reduces* the cost of an anomaly,
not that it rescues one whose gross alpha is smaller than its gross cost --
and this one's is smaller by 16x.

**(F3) The frontier's far end is a hold.** As H -> infinity both arms
converge on "buy the initially-strongest asset and never trade", which is a
concentrated buy-and-hold. It may well beat VOLMATCH_HOLD by luck of which
asset was strongest in 2020-04, and D5 plus the scramble control are what
stop that being read as a signal result.

None of the three is a reason not to run it. R-63 priced the signal at one
point; this round measures the whole curve, and a curve that never crosses
is a decisive answer to "would it work if it were cheaper", which this
project has asked implicitly since R-12 and never once measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.inference import (  # noqa: E402
    daily_returns,
    max_drawdown_from_returns,
    total_log_return,
)

# The frozen R-63 substrate. Imported, never copied: if these drift, the
# frontier stops extending the number it claims to extend.
from experiments.r63_shared import (  # noqa: E402
    BOOT_KW,
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
from experiments.r63_novel_xsmom_rank import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    DEADBAND,
    HORIZONS,
    WARM_DAYS,
    basket_log_returns,
    conditional_vol_scale,
    cross_sectional_score,
    warm_window,
)
from experiments.r63_novel_xsmom_rank import build_targets as r63_baseline_targets  # noqa: E402

OUT_DIR = ROOT / "reports" / "r64_holding_period"

# Zero-fee spec for the D5 diagnostic column. Not a decision cell.
SPOT_FREE = SPOT_BASE.__class__.spot(fee_rate=0.0)

# R-63's measured frictionless edge of the cross-sectional ranking, and the
# D5 bar at half of it. Both frozen here, from the committed R-63 report.
R63_GROSS_EDGE = 0.480
D5_BAR = 0.5 * R63_GROSS_EDGE  # +0.240 log units

# R-63's measured cost side, for the frontier's reference point.
R63_TURNOVER_PER_DAY = 3.44
R63_NET_D1 = -7.537


# ------------------------------------------------------------- turnover


def turnover_stats(targets: pd.DataFrame, fee_rate: float = 0.001) -> dict:
    """Round-trip traded notional per day and the log drag it implies.

    Charged on the SAME terms as :func:`simulate_portfolio` -- a rebalance
    is skipped entirely unless the requested change in total traded notional
    exceeds the 5% band -- so this is the cost the simulator actually pays,
    not an idealized upper bound on it.
    """
    w = np.clip(np.nan_to_num(targets.to_numpy(dtype=float), nan=0.0), 0.0, 1.0)
    gross = w.sum(axis=1)
    over = gross > 1.0
    if over.any():
        w[over] = w[over] / gross[over][:, None]

    held = w[0].copy()
    traded_total = 0.0
    n_rebalances = 0
    for i in range(1, len(w)):
        traded = float(np.abs(w[i - 1] - held).sum())
        if traded > TOTAL_NOTIONAL_DEADBAND:
            traded_total += traded
            n_rebalances += 1
            held = w[i - 1].copy()

    days = max(len(w) / BARS_PER_DAY, 1e-9)
    per_day = traded_total / days
    return {
        "turnover_per_day": per_day,
        "rebalances_per_day": n_rebalances / days,
        "n_rebalances": n_rebalances,
        "implied_log_drag_per_day": fee_rate * per_day,
        "implied_log_drag_total": fee_rate * traded_total,
        "days": days,
    }


def holding_period_days(targets: pd.DataFrame) -> float:
    """Mean time between changes in the SET of held assets, in days.

    The direct analogue of R-63's 2.86 leader-changes per day, and the axis
    this round varies. Membership, not weight: a weight moved by the
    volatility scale is not a holding-period event.
    """
    w = np.nan_to_num(targets.to_numpy(dtype=float), nan=0.0)
    held = w > 0.0
    changes = int((held[1:] != held[:-1]).any(axis=1).sum())
    days = max(len(w) / BARS_PER_DAY, 1e-9)
    return days / max(changes, 1)


# ------------------------------------------------------- vol-matched hold


def realized_vol(equity: pd.Series) -> float:
    """Annualized realized volatility of an equity curve's daily returns."""
    r = daily_returns(equity).to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(365.25))


def volmatched_hold_equity(cand_eq: pd.Series, aligned: dict, assets, market,
                           tol: float = 0.02, max_iter: int = 8):
    """R-31's convention: the equal-weight basket held at the CONSTANT total
    notional whose realized volatility matches the candidate's own.

    R-63 filed this as the fix its own MATCHED_HOLD needed. Equal notional is
    not equal risk when one arm is concentrated: holding 1 of 6 assets at 52%
    notional is roughly twice the volatility of holding 6 of them at 52%, and
    a drawdown comparison against the wrong one is partly arithmetic.

    Returns ``(equity, c, vol, matched)``. ``matched`` is False when the
    proportional iteration could not land inside ``tol`` -- in which case the
    cell is VOIDED, not scored, per ROUTINE.md's standing rule.
    """
    target_vol = realized_vol(cand_eq)
    if not np.isfinite(target_vol) or target_vol <= 0:
        return None, float("nan"), float("nan"), False

    idx = aligned[list(assets)[0]].index
    c = 0.5
    eq = simulate_portfolio(matched_hold_targets(idx, assets, c), aligned, market)
    vol = realized_vol(eq)
    for _ in range(max_iter):
        if not np.isfinite(vol) or vol <= 0:
            return eq, c, vol, False
        if abs(vol - target_vol) <= tol * target_vol:
            return eq, c, vol, True
        c = float(np.clip(c * (target_vol / vol), 1e-3, 1.0))
        eq = simulate_portfolio(matched_hold_targets(idx, assets, c), aligned, market)
        vol = realized_vol(eq)
        if c >= 1.0 and vol < target_vol:  # the long-only cap binds
            return eq, c, vol, False
    return eq, c, vol, abs(vol - target_vol) <= tol * target_vol


# ---------------------------------------------------------- frontier row


def frontier_row(arm: str, params: dict, targets: pd.DataFrame,
                 net_cmp: dict, gross_cmp: dict, bench: str,
                 window: str, universe: str, **extra) -> dict:
    """The one schema both branches emit, for every configuration.

    ``net_cmp`` is :func:`compare` at 0.10%; ``gross_cmp`` the same at 0 bps.
    The gross column is a diagnostic (D5), never a decision cell.
    """
    tstats = turnover_stats(targets)
    row = {
        "arm": arm,
        "window": window,
        "universe": universe,
        "bench": bench,
        "hold_days": holding_period_days(targets),
        "turnover_per_day": tstats["turnover_per_day"],
        "rebalances_per_day": tstats["rebalances_per_day"],
        "implied_log_drag_total": tstats["implied_log_drag_total"],
        "mean_notional": mean_total_notional(targets),
        "gross_growth_diff": gross_cmp["growth_diff"],
        "gross_growth_lo": gross_cmp["growth_lo"],
        "gross_growth_hi": gross_cmp["growth_hi"],
        "net_growth_diff": net_cmp["growth_diff"],
        "net_growth_lo": net_cmp["growth_lo"],
        "net_growth_hi": net_cmp["growth_hi"],
        "net_dd_diff": net_cmp["dd_diff"],
        "net_dd_lo": net_cmp["dd_lo"],
        "net_dd_hi": net_cmp["dd_hi"],
        "cand_final": net_cmp["cand_final"],
        "bench_final": net_cmp["bench_final"],
        "cand_dd": net_cmp["cand_dd"],
        "bench_dd": net_cmp["bench_dd"],
        "n_days": net_cmp["n_days"],
    }
    row.update({f"p_{k}": v for k, v in params.items()})
    row.update(extra)
    return row


# ------------------------------------------------------- decision rules


def d1_pass(row: dict) -> bool:
    """W_FULL6, 0.10%, growth vs VOLMATCH_HOLD: positive and interval clear."""
    return (row["net_growth_diff"] > 0.0
            and excludes_zero(row["net_growth_lo"], row["net_growth_hi"]))


def d2_pass(row: dict) -> bool:
    """W_FULL6, 0.10%, max drawdown vs VOLMATCH_HOLD: negative and clear."""
    return (row["net_dd_diff"] < 0.0
            and excludes_zero(row["net_dd_lo"], row["net_dd_hi"]))


def d3_pass(row: dict) -> bool:
    """W_VAL on U8: same sign on both statistics, intervals not required."""
    return row["net_growth_diff"] > 0.0 and row["net_dd_diff"] < 0.0


def d5_pass(row: dict) -> bool:
    """SIGNAL RETENTION. At 0 bps the arm keeps >= half of R-63's +0.480.

    The rule this round exists to enforce: an arm that buys its turnover
    reduction by discarding the signal has not solved the cost problem, it
    has left the market.
    """
    return row["gross_growth_diff"] >= D5_BAR


def further_work(d1: bool, d2: bool, d3: bool, d5: bool,
                 scramble_survived: bool) -> bool:
    """The frozen further-work bar. NOT the promotion bar, and clearing it
    authorizes exactly one holdout read on W_HOLD."""
    return (d1 or d2) and d3 and d5 and scramble_survived


__all__ = [
    "BARS_PER_DAY", "BARS_PER_YEAR", "BOOT_KW", "D5_BAR", "DEADBAND",
    "HORIZONS", "OUT_DIR", "R63_GROSS_EDGE", "R63_NET_D1",
    "R63_TURNOVER_PER_DAY", "SCRAMBLE_SEEDS", "SPOT_BASE", "SPOT_FREE",
    "SPOT_REAL", "START_BALANCE", "TOTAL_NOTIONAL_DEADBAND", "UNIVERSE_6",
    "UNIVERSE_8", "WARM_DAYS", "W_FULL6", "W_HOLD", "W_TRAIN", "W_VAL",
    "align_frames", "basket_log_returns", "compare", "conditional_vol_scale",
    "config_count", "cross_sectional_score", "d1_pass", "d2_pass", "d3_pass",
    "d5_pass", "excludes_zero", "frontier_row", "further_work",
    "holding_period_days", "load_universe", "matched_hold_targets",
    "mean_total_notional", "r63_baseline_targets", "realized_vol",
    "scramble_targets", "simulate_portfolio", "static_hold_equity",
    "turnover_stats", "volmatched_hold_equity", "warm_window",
]
