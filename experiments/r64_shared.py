"""R-64: stop improving *what* the incumbent wants to hold and fix *how it
gets there* -- the trading-rate axis.

Shared, frozen infrastructure for a two-branch parallel round. Per ROUTINE.md's
parallelism rules this file is neutral ground: both branches import from it,
neither branch edits it, and it does not itself define a candidate strategy or
compute a verdict. It exists so the pre-registration below is committed once,
before either branch reads a single number.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **COST** ("costs scale *with* the signal").

`kelly_regime_v4` decides a target exposure every bar as `desired = frac x
scale` and then answers a second, separate question that nobody in this
project has ever examined: **given that I am at `pos` and want `desired`, how
far do I move?** Its answer, unchanged since L-04 and inherited by every
variant since, is three lines in `KellyRegime.prepare`:

    if abs(desired - pos) > self.deadband:
        pos = desired

That is: sit still inside a +/-10% no-trade band, and on breaching it **jump
all the way to the target**. Twenty-two attempts across R-34 -> R-63 have
retuned `frac` or `scale` -- the two factors of *what* to hold. Not one has
touched the third line.

Why now, and why this is not just the twenty-third variant. Two measured
findings from the last two rounds compose into this one:

- **R-62** decomposed `frac x scale` and found the incumbent's one surviving
  property lives entirely in the directional **vote**; the volatility target
  reproduces none of it. But R-62's novel arm also measured what the vol
  target *is* doing: deleting it left a bare vote-only rule that lost by wide
  margins at the 0.40% tier (D4 0/6, LTC $121 vs $1,190), because "deleting
  the vol target removes what makes a trend rule's turnover affordable." So
  the surviving factor carries the edge and the other factor is, empirically,
  a **cost-control device**. That makes cost control the live axis, and it has
  never been attacked directly.
- **R-63** priced a signal against its own turnover bill for the first time in
  this project: a cross-sectional ranking that is *genuinely real* (it beats
  10 of 10 permutations of its own weights once fees are removed) and worth
  ~0.5 log units over 6.4 years, against an **8.0** log-unit turnover bill.
  Unaffordable by 16:1. The lesson generalizes past the panel: on this data
  the binding constraint on a real signal is what it costs to track, not
  whether it exists.

If a real signal's problem is the bill, the object to attack is the trading
policy, not the signal. That object has a mature literature, and -- this is
the round's actual question -- **that literature has two different answers
depending on the cost structure, and they disagree about what this repo
should do.**

- Under **proportional** costs (a taker fee: you pay f x |traded notional|,
  and there is no impact term), the optimal policy is a **no-trade region**
  plus, on exit, a trade only **to the nearest boundary of that region** --
  never to the target. Constantinides (1986), Davis & Norman (1990),
  Shreve & Soner (1994), Liu (2004). Trading to the target from the boundary
  buys nothing and pays for the whole distance.
- Under **quadratic / temporary-impact** costs, the optimal policy is a
  **smooth partial adjustment**: `x_t = (1 - a) x_{t-1} + a x_t^aim`, closing
  a constant fraction `a` of the distance to an "aim" portfolio each period,
  where the aim over-weights signals that **decay slowly** because they are
  worth paying to chase. Garleanu & Pedersen (2013, J. Finance 68(6)).

`kelly_regime_v4` implements **neither**. It has the no-trade region of the
first and the trade-to-target destination of neither, which is the one
combination both literatures agree is suboptimal. This round runs the two
canonical policies against each other on data whose cost structure is
**purely proportional** (a 0.10% taker fee, no order book, no impact model --
see SIMULATOR below), where the theory makes a **falsifiable prediction about
which one should win**. That prediction is pre-registered in D2 below.

**Not a duplicate of:**

- **L-05 / L-06 (`kelly_regime_ev`, `kelly_regime_ev_fast`)**. These are the
  closest prior art and the difference is exact: they derive the **width** of
  the no-trade band analytically (`|df| > 2 fee / (H sigma^2)`, Constantinides
  1986; Davis & Norman 1990) and leave the destination as "jump to target".
  This round holds the width question fixed and attacks the **destination**
  and the **rate**. The two are orthogonal: L-05 asks *when* to trade, R-64
  asks *how far*. Note L-05's own recorded lesson -- at 0.40% its band
  "exceeds 1.0, no rebalance is ever worth its cost" -- is a statement about
  width that says nothing about destination.
- **R-56 (maker/limit execution model, B-24)**. That round changed the
  **microstructure of each fill** (patient limit orders, taker fallback) at a
  fixed set of rebalance instants. This round changes **which rebalances
  happen and how large they are**, at a fixed (taker) fill model. R-56's
  conservative branch found the no-trade band "already limits it to ~150-260
  non-urgent rebalances" -- this round asks what those rebalances should be
  *sized* at.
- **R-34 -> R-63 (twenty-two SIZE/INFO variants)**. Every one modifies `frac`
  (vote timing, gates, vetoes, exogenous confirmations) or `scale` (target_vol
  magnitude, dimensional form, per-state Kelly, CPPI floors) or the universe.
  None modifies the position-update rule that consumes both. R-62's factor
  isolation is what makes that gap visible as a gap.
- **R-12 / R-13 ("tuning turnover to fit a fee tier", section C)**. Section C
  rules out *fitting* a turnover level to a fee tier -- 28 of 32 in-sample, 0
  of 28 out-of-sample. This round does not fit turnover to a tier: the
  conservative arm has **zero free parameters** (it re-uses v4's own shipped
  deadband and changes only the destination), and D2 below requires the
  advantage to **grow** with the fee, which is the opposite of a tier fit and
  is a test that a tier-fitted rule fails by construction.

**Is it simulable here?** Yes, with zero new data and no new simulator
capability. The change is entirely inside `prepare`'s position-update loop,
which consumes only bar-close information already available to v4. Bar-close
signals, next-open fills, taker fees on traded notional. Critically, the
cost model here **is** proportional -- `MarketSpec` charges `fee_rate x
notional` with no impact, no queue and no spread term -- which is precisely
the regime in which the two literatures diverge, and is why this comparison
is decidable on this data at all.

**What would make it fail (named now, before any code ran).** Three named
mechanisms, and note that the first predicts failure for the *novel* arm on
theoretical grounds, so that a novel-arm pass is read as a genuine surprise:

1. **The novel arm is a category error and the fees say so.** A smooth
   partial-adjustment rule trades a little bit *every bar*. At 288 bars a day
   and 10bps a side, "a little bit every bar" is the exact failure mode of
   L-14 / L-15 / L-16 / L-18 (1,605 / 9,039 trades). If the novel arm's
   turnover explodes, the round has confirmed the theory's own scope
   condition rather than discovered anything -- which is a legitimate,
   reportable result, and is why the novel arm is REQUIRED to carry a
   minimum-step filter so that it is tested at its strongest rather than as a
   strawman.
2. **The deadband is already doing the work.** v4 trades 174 times in 9.6
   years. If the no-trade band is wide enough that exits from it are rare and
   the position is far outside it when they happen, then "to the boundary"
   and "to the target" differ by ~one band-width per rebalance on ~174
   rebalances, which may be worth a few basis points a year and nothing more.
   This is the most likely outcome and it is named here so that a null is not
   dressed up afterwards.
3. **Lower turnover, worse tracking, worse returns.** Both arms hold a
   position closer to the *previous* position than v4 does. On a trend signal
   that is a lag, and lag on a trend rule is the one thing this project has
   repeatedly measured as expensive (R-53's macro veto led only 4/12 episodes;
   R-60's CUSUM timing). Saving fees while lagging the signal is a trade this
   data may well refuse.

=====================================================================
DATA, SPLITS, AND WHAT EACH IS FOR
=====================================================================

BTC, the canonical Bitstamp 5m spot series (`data/btcusd_spot_5m.csv.gz`,
2017-01 -> 2026-08), via `scripts.experiment`'s already-loaded frame:

    inner-train       ...        -> 2020-12-31   fit, sweep, iterate freely
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              frozen config only, ONCE

ETH for the pre-registered falsification test, in two forms, deliberately:

    ETH-A (primary)   `data/ethusd_bitfinex_5m.csv.gz`, 2016-03 -> 2019-12.
                      Costs **zero** holdout consultations (pre-2020 entirely,
                      the R-19/R-28 convention), and is the falsification arm
                      every ETH check in this ledger before R-57 used.
    ETH-B (secondary) `data/ethusd_coinbase_spot_5m.csv.gz`, restricted to
                      2023-01-01 ->. A genuine out-of-sample second instrument.
                      Read ONCE, in the same holdout pass as D1, and counted.

=====================================================================
DECISION RULES, FROZEN (default is REJECT)
=====================================================================

All comparisons are **arm vs `kelly_regime_v4`**, paired on identical daily
return series, because the round's question is whether the trading-rate change
improves the incumbent -- not whether the incumbent is any good, which R-29 /
R-30 already answered ("not distinguishably").

**D0 (RISK-MATCH GATE, and it binds).** ROUTINE.md's first standing rule:
match risk before comparing anything. Both arms hold a position closer to
their previous one than v4 does, so their mean notional CAN drift away from
v4's -- and if it does, any growth difference is an exposure statement, which
is how three of this project's findings died (R-31, R-32, R-33). So: report
`mean_notional` for the arm and for v4 on every window. If
`|c_arm - c_v4| / c_v4 > 0.10` on the holdout window, the D1 head-to-head is
**VOID as a growth claim** and must be re-reported against
`ConstantExposureHold(c_arm)` instead, flagged as such. A void D1 cannot
promote. This is a gate, not a diagnostic.

**D1 (PRIMARY, holdout, spot @ 0.10%).** Difference in total log growth,
arm minus v4, on daily returns, paired stationary block bootstrap
(mean_block=30, n_boot=2000, seed=7, the project-standard `BOOT_KW`).
Promotion requires **point estimate > 0 AND the 95% interval excluding zero**.

**D2 (COST-MECHANISM TEST -- the discriminating one, and it can fail an arm
on its own).** The same holdout comparison re-run at the **0.40%** taker tier
(`scripts/fee_study.py`'s tier). Both arms claim a *cost* mechanism. A cost
mechanism's advantage must therefore **grow with the cost**:

        REQUIRE  Delta_logret(0.40%)  >  Delta_logret(0.10%)

If an arm's advantage shrinks or inverts as the fee quadruples, then whatever
it is doing, it is not saving fees -- it is a lag or an exposure difference
wearing a cost story. That arm is **NEGATIVE regardless of D1**, and this is
the test that a turnover level fitted to a fee tier (section C, R-12) fails by
construction. This rule is why the round is worth running even if D1 is null.

**D3 (PRE-REGISTERED FALSIFICATION, ROUTINE step 2).** ETH-A (2016-03 ->
2019-12, +0 holdout): the sign of the arm-minus-v4 log-growth difference must
be **positive**. A mechanical improvement to a position-update rule is not
supposed to be an asset-specific effect -- R-57's whole lesson is that this
project's claims have been n=1 -- so a sign flip on the one other instrument
with pre-2020 coverage refutes the mechanism. ETH-B (Coinbase, 2023+) is
reported alongside as the out-of-sample second instrument, and a sign flip
there is recorded but is **not** an independent gate (n=1 asset, one period).

**D4 (TURNOVER SANITY -- did the mechanism operate at all?).** Fees paid and
trade count, arm vs v4, on every window. Both arms are turnover-reduction
devices; if turnover does not FALL, the arm did not do the thing it was built
to do and any performance difference is something else. Report, and state
plainly if it rises. For the novel arm specifically, a trade count above
**1,000** on the full period is the failure-mode-1 signature named above.

**D5 (PLATEAU, NOT PEAK).** Any arm carrying a fitted parameter must report
its neighbourhood: at least 4 neighbours around the selected value, on
inner-validation, with the Sharpe spread across the plateau stated against
the +/-0.2 noise floor (R-20). The conservative arm carries **zero** fitted
parameters by construction, so its D5 is a robustness sweep of v4's own
shipped deadband, reported for context rather than as a selection.

**D6 (FUNDING, futures).** If and only if an arm survives D1-D3, the futures
comparison is re-run with funding charged (`scripts/funding_study.py`). The
table's futures column is an upper bound until this is done (standing rule).

**PROMOTION BAR (ROUTINE step 4, default REJECT).** Promote only if ALL hold:
D0 not void; D1 positive with interval excluding zero; D2 satisfied; D3
positive on ETH-A; D4 shows turnover falling; D5 a plateau; and the arm still
beats `buy_and_hold` out-of-sample after real costs. Anything else is
NEGATIVE and is written up with the same care.

**HOLDOUT BUDGET, DECLARED IN ADVANCE.** Each arm gets: 1 read on BTC spot
@0.10%, 1 at 0.40%, 1 on ETH-B, plus the paired v4 baselines on the same
windows (which are re-runs of a strategy whose holdout numbers are already on
record, but are counted anyway). Estimated **+8 consultations** across both
arms; the exact count is reported in the ledger entry and added to the
running total. No branch reads a bar dated 2023-01-01 or later before its
configuration is frozen and committed.

**TRIALS COUNT.** The total across BOTH branches, per ROUTINE.md's parallelism
rule. Each branch reports every configuration it evaluated, including the ones
it discarded.

=====================================================================
CITATIONS (the two policies being tested)
=====================================================================

Proportional-cost / no-trade-region -> **trade to the boundary**:
  Constantinides, G. M. (1986). "Capital Market Equilibrium with Transaction
    Costs." Journal of Political Economy 94(4), 842-862.
  Davis, M. H. A. & Norman, A. R. (1990). "Portfolio Selection with
    Transaction Costs." Mathematics of Operations Research 15(4), 676-713.
  Shreve, S. E. & Soner, H. M. (1994). "Optimal Investment and Consumption
    with Transaction Costs." Annals of Applied Probability 4(3), 609-692.
  Liu, H. (2004). "Optimal Consumption and Investment with Transaction Costs
    and Multiple Risky Assets." Journal of Finance 59(1), 289-338.

Quadratic-cost / impact -> **smooth partial adjustment toward an aim**:
  Garleanu, N. & Pedersen, L. H. (2013). "Dynamic Trading with Predictable
    Returns and Transaction Costs." Journal of Finance 68(6), 2309-2340.
  Garleanu, N. & Pedersen, L. H. (2016). "Dynamic portfolio choice with
    frictions." Journal of Economic Theory 165, 487-516.

Cost-aware anomaly implementation (why the tier matters more than the signal):
  Novy-Marx, R. & Velikov, M. (2016). "A Taxonomy of Anomalies and Their
    Trading Costs." Review of Financial Studies 29(1), 104-147.
  Frazzini, A., Israel, R. & Moskowitz, T. J. (2018). "Trading Costs."
    Working paper, AQR / SSRN 3229719.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.matched_hold import ConstantExposureHold, mean_notional  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_spot, load_ohlcv_csv  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns,
    max_drawdown_from_returns,
    paired_bootstrap,
    total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DATA = ROOT / "data"

# ------------------------------------------------------------------- splits
INNER_TRAIN = (None, "2020-12-31")
INNER_VAL = ("2021-01-01", "2022-12-31")
HOLDOUT = ("2023-01-01", None)
OOS_START = "2023-01-01"

BOOT_KW = dict(mean_block=30.0, n_boot=2_000, seed=7)

# Fee tiers. 0.0010 is the table's convention; 0.0040 is fee_study.py's
# stress tier and the one D2 is decided on.
FEE_BASE = 0.0010
FEE_STRESS = 0.0040

RISK_MATCH_TOL = 0.10  # D0: |c_arm - c_v4| / c_v4
NOVEL_TURNOVER_FLAG = 1_000  # D4: novel-arm failure-mode-1 signature

# Counts every backtest run through `measure`, so the round's trials count is
# an observed number rather than a remembered one.
_CONFIGS = [0]


def spot(fee: float = FEE_BASE) -> MarketSpec:
    """Spot market at an explicit taker fee (default = the table's 0.10%)."""
    return MarketSpec.spot(fee_rate=fee)


def futures(fee: float = 0.0005, leverage: float = 5.0) -> MarketSpec:
    return MarketSpec.futures(leverage=leverage, fee_rate=fee)


# --------------------------------------------------------------------- data

def load_btc() -> pd.DataFrame:
    return load_ohlcv_csv(DATA / "btcusd_spot_5m.csv.gz")


def load_eth_a() -> pd.DataFrame:
    """ETH-A: Bitfinex 2016-03 -> 2019-12. Zero holdout cost."""
    return load_ohlcv_csv(DATA / "ethusd_bitfinex_5m.csv.gz")


def load_eth_b() -> pd.DataFrame:
    """ETH-B: Coinbase 2020-04 -> 2026-08. Only read from 2023-01-01."""
    return load_coinbase_spot(DATA, "ETH")


# ---------------------------------------------------------------- measuring

def measure(strategy, df: pd.DataFrame, window, market: MarketSpec,
            balance: float = 1_000.0):
    """One backtest over a window, warmed on the bars before it. Counted."""
    start, end = window
    _CONFIGS[0] += 1
    res = run_period(strategy, df, start, end, market=market,
                     start_balance=balance)
    return res, compute_metrics(res)


def configs_evaluated() -> int:
    return _CONFIGS[0]


def v4():
    """A fresh, unmodified incumbent. Never mutate its defaults."""
    return get_strategy("kelly_regime_v4")


def compare(arm, df: pd.DataFrame, window, market: MarketSpec,
            label: str = "") -> dict:
    """Arm vs v4 on one window/market, with everything D0-D4 needs.

    Returns a flat dict: point estimates, the paired-bootstrap interval on
    total log growth and on max drawdown, both arms' mean notional and the
    D0 risk-match verdict, fees paid and trade counts.
    """
    arm_res, arm_m = measure(arm, df, window, market)
    v4_res, v4_m = measure(v4(), df, window, market)

    a = daily_returns(arm_res.equity).to_numpy(dtype=float)
    b = daily_returns(v4_res.equity).to_numpy(dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    growth = paired_bootstrap(a, b, total_log_return, **BOOT_KW)
    dd = paired_bootstrap(a, b, max_drawdown_from_returns, **BOOT_KW)

    c_arm = mean_notional(arm_res)
    c_v4 = mean_notional(v4_res)
    mismatch = abs(c_arm - c_v4) / c_v4 if c_v4 > 0 else float("nan")

    return dict(
        label=label,
        market=market.name,
        fee=market.fee_rate,
        n_days=n,
        arm_final=arm_m.final_balance,
        v4_final=v4_m.final_balance,
        arm_sharpe=arm_m.sharpe,
        v4_sharpe=v4_m.sharpe,
        arm_dd=arm_m.max_drawdown_pct,
        v4_dd=v4_m.max_drawdown_pct,
        arm_trades=arm_m.num_trades,
        v4_trades=v4_m.num_trades,
        arm_fees=arm_m.fees_paid,
        v4_fees=v4_m.fees_paid,
        d_logret=growth.diff.point,
        d_logret_lo=growth.diff.lo,
        d_logret_hi=growth.diff.hi,
        d_logret_excludes_zero=(growth.diff.lo > 0.0 or growth.diff.hi < 0.0),
        d_dd=dd.diff.point,
        d_dd_lo=dd.diff.lo,
        d_dd_hi=dd.diff.hi,
        c_arm=c_arm,
        c_v4=c_v4,
        risk_mismatch=mismatch,
        d0_void=bool(np.isfinite(mismatch) and mismatch > RISK_MATCH_TOL),
    )


def matched_hold_cell(arm, df: pd.DataFrame, window, market: MarketSpec,
                      label: str = "") -> dict:
    """The D0-void fallback: arm vs a passive long carrying the arm's own
    mean notional. Used only when `compare` returns d0_void=True.
    """
    arm_res, arm_m = measure(arm, df, window, market)
    c = mean_notional(arm_res)
    mh_res, mh_m = measure(ConstantExposureHold(c), df, window, market)

    a = daily_returns(arm_res.equity).to_numpy(dtype=float)
    b = daily_returns(mh_res.equity).to_numpy(dtype=float)
    n = min(len(a), len(b))
    growth = paired_bootstrap(a[:n], b[:n], total_log_return, **BOOT_KW)
    dd = paired_bootstrap(a[:n], b[:n], max_drawdown_from_returns, **BOOT_KW)
    return dict(label=label, market=market.name, fee=market.fee_rate, c=c,
                arm_final=arm_m.final_balance, hold_final=mh_m.final_balance,
                d_logret=growth.diff.point, d_logret_lo=growth.diff.lo,
                d_logret_hi=growth.diff.hi, d_dd=dd.diff.point,
                d_dd_lo=dd.diff.lo, d_dd_hi=dd.diff.hi)


# ------------------------------------------------------------------ verdicts

def d2_satisfied(d_logret_base: float, d_logret_stress: float) -> bool:
    """The cost-mechanism test: the advantage must GROW with the fee."""
    return d_logret_stress > d_logret_base


def promotion(d0_void: bool, d1_point: float, d1_excludes_zero: bool,
              d2_ok: bool, d3_eth_a: float, turnover_fell: bool,
              plateau: bool, beats_hold_oos: bool) -> str:
    """The frozen bar. Returns 'PROMOTE' or the first reason it is NEGATIVE."""
    if d0_void:
        return "NEGATIVE: D0 void (risk mismatch > 10%)"
    if not (d1_point > 0.0 and d1_excludes_zero):
        return "NEGATIVE: D1 (holdout growth vs v4 not established)"
    if not d2_ok:
        return "NEGATIVE: D2 (advantage does not grow with the fee)"
    if not d3_eth_a > 0.0:
        return "NEGATIVE: D3 (ETH-A falsification: sign flips)"
    if not turnover_fell:
        return "NEGATIVE: D4 (turnover did not fall)"
    if not plateau:
        return "NEGATIVE: D5 (peak, not plateau)"
    if not beats_hold_oos:
        return "NEGATIVE: fails the standing bar vs buy_and_hold OOS"
    return "PROMOTE"


def fmt(row: dict) -> str:
    return (f"{row['label']:38s} {row['market']:11s} fee={row['fee']:.4f} "
            f"arm=${row['arm_final']:>12,.0f} v4=${row['v4_final']:>12,.0f} "
            f"dlog={row['d_logret']:+7.3f} [{row['d_logret_lo']:+.3f}, "
            f"{row['d_logret_hi']:+.3f}] "
            f"trades={row['arm_trades']:>5d}/{row['v4_trades']:<5d} "
            f"fees=${row['arm_fees']:>10,.0f}/${row['v4_fees']:<10,.0f} "
            f"c={row['c_arm']:.3f}/{row['c_v4']:.3f}"
            f"{' VOID' if row['d0_void'] else ''}")
