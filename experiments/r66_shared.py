"""R-66: the no-trade band's *shape*, and the one consequence R-64 could not
separate from its own mechanism -- that a band never lets the position reach
exactly flat.

Shared, frozen infrastructure for a two-branch parallel round. Per ROUTINE.md's
parallelism rules this file is neutral ground: both branches import from it,
neither branch edits it, and it does not itself define a candidate strategy or
compute a verdict. It exists so the pre-registration below is committed once,
before either branch reads a single number.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **COST** ("costs scale *with* the signal").
Backlog item: **B-29**, filed by R-64, ranked top of the actionable list.

R-64 attacked the position-update rule for the first time in this project and
came back NEGATIVE on both arms -- but its conservative arm failed in a way
that is *diagnostic rather than terminal*, and B-29 is the one-line experiment
that separates the two things it confounded.

What R-64's conservative arm did. `kelly_regime_v4`'s update rule, unchanged
since L-04, is a no-trade band with a **trade-to-target** destination::

    if abs(desired - pos) > band:   pos = desired          # band = 0.10

Under a purely proportional fee -- which is exactly what this simulator
charges -- the optimal destination is the **nearest boundary of the band**,
not the target (Constantinides 1986; Davis & Norman 1990; Shreve & Soner
1994; Liu 2004). R-64 implemented that, indexed by `k`::

    if desired - pos > band:   pos = desired - k * band
    elif pos - desired > band: pos = desired + k * band

Three of its measured findings matter here, and all three are *favourable* to
the mechanism:

- **The turnover saving is real and large.** Median overshoot at a band exit
  is **2.13 band-widths**, so 43% of all turnover is genuinely removable. The
  pre-registration's own most-likely-outcome ("the deadband already does the
  work; this will be a rounding error") was **refuted**.
- **D2, the cost-mechanism test, passed cleanly.** The arm's advantage *grows*
  as the fee quadruples to 0.40%, at every k>0 on both splits (inner-train
  -0.036 -> +0.145; inner-val +0.021 -> +0.128). Whatever it is doing, it is
  saving fees, which is what it claimed.
- **It still lost, and the cause was named precisely.** ETH-A log-growth
  difference vs v4 was **-0.430 [-0.791, -0.101]** at k=1 -- the only interval
  in the entire round excluding zero, and on the wrong side. The mechanism was
  a **side effect nobody had named**: a no-trade region means the position
  never returns to *exactly* flat, so the arm carries a residual long of up to
  `k*band` = 10% of equity through every bear regime v4 sits out. Fees explain
  ~13% of the ETH gap; the residual long explains the rest. The same mechanism
  prints on BTC as max drawdown rising monotonically with k on the 2022 bear
  split (33.2 -> 33.9 -> 34.3 -> 35.3 -> 37.0%).

So R-64's conservative arm confounded two independent consequences of one
line: a **destination** change (worth a real 43% of turnover, with the cost
slope in its favour) and a **never-goes-flat** consequence (fatal). B-29 is
the observation that only one of them is fatal and they are separable in one
conditional.

**This round is the separation, run two ways.** Both arms attack the same
object -- the geometry of the no-trade region -- and both make the position
able to reach exactly flat again, by two mechanisms that are independent of
each other and that this project can tell apart:

- **Conservative (B-29 literal).** Keep R-64's boundary destination; add one
  conditional that snaps to exactly flat when the target *is* exactly flat.
  Zero new fitted parameters; `band` is still v4's own shipped `deadband=0.10`
  and `k` is still R-64's mechanism index with `k=0` reproducing v4.
- **Novel.** Leave the destination alone (trade to target, as v4 does) and
  change the band's **width** into a function of the exposure it is a band
  *around*: `band(f) = deadband * f**p`. This is the shape the asymptotic
  transaction-cost literature actually derives -- the no-trade half-width is
  proportional to `(f^2 (1-f)^2 eps)^(1/3)`, i.e. `~ f^(2/3)` near f=0
  (Janecek & Shreve 2004; Gerhold, Guasoni, Muhle-Karbe & Schachermayer 2014)
  -- and it delivers the snap-to-flat property **as a derived consequence
  rather than as a hand-added conditional**: the band vanishes where the
  target vanishes, so a bear-regime exit lands on exactly zero with no
  residual long, without anyone writing a special case for zero.

That is why the pairing is worth a round rather than one branch. The two arms
reach the *same* fix from opposite directions -- one by fiat, one by theory --
so their agreement or disagreement is itself evidence:

    both fail  -> the residual long was NOT what killed R-64's arm, and this
                  project's own published diagnosis of R-64 is wrong.
    both pass  -> the destination change is bankable once flat is reachable,
                  and the 43% turnover figure R-64 measured was real money.
    split      -> width and destination are not interchangeable, and the one
                  that passes says which of the two the geometry needs.

**A sharp, pre-registered prediction, which is this round's actual content.**
R-64 published a specific causal claim: the residual long, not the fee saving
or the lag, is what sank the boundary policy. That claim makes a numerical
prediction on data already read, and it is committed here before any code
runs: **the conservative arm's ETH-A log-growth difference vs v4 must be
materially better than R-64's -0.430**, and the residual-long diagnosis is
refuted if it is not. This is the rare case where a prior round's *explanation*
is falsifiable rather than merely plausible, and the cost of testing it is one
conditional.

**Not a duplicate of:**

- **R-64 (conservative)**: this arm *is* R-64's arm plus one conditional. That
  is the point -- B-29 exists because R-64's own write-up says the confound is
  separable in one line and asks for exactly this. The comparison inherits
  R-64's entire pre-registered battery unchanged, so the two rounds' numbers
  are directly comparable rather than merely similar.
- **R-64 (novel, Garleanu-Pedersen)**: that arm replaced the band with a
  smooth *rate*. This round's novel arm keeps the band and changes its
  *width profile*. R-64 measured the smooth-rate object as a confirmed
  category error under a proportional fee; nothing here re-tries it.
- **L-05 / L-06 (`kelly_regime_ev`, `kelly_regime_ev_fast`)**. These derive a
  band width from fee, volatility and holding horizon (`|df| > 2 fee/(H
  sigma^2)`) -- a **scalar** that is the same at every exposure level, and
  they leave the destination as "jump to target". The novel arm here derives
  the width's **dependence on the exposure level itself**, which is a
  different axis of the same object: L-05 asks *how wide*, this asks *how does
  the width vary with f*. L-05's band is flat in f; the literature's is not,
  and at f=0 the difference is the whole question.
- **R-56 (maker/limit execution, B-24)**: microstructure of each fill at a
  fixed set of rebalance instants. This changes which rebalances happen.
- **R-34 -> R-63**: twenty-two variants of `frac` or `scale` -- *what* to
  hold. Neither arm here touches either factor.
- **R-12 / R-13 (section C, "tuning turnover to fit a fee tier")**: neither
  arm fits a turnover level. The conservative arm has zero free parameters;
  the novel arm's exponent `p` is a mechanism index with a literal theoretical
  value (2/3) and an identity point (p=0 == v4), not a selection grid. D2
  below requires the advantage to **grow** with the fee, which a tier-fitted
  rule fails by construction.

**Is it simulable here?** Yes, with zero new data and no new simulator
capability. Both changes live inside `prepare`'s position-update loop, which
consumes only bar-close information already available to v4. The cost model is
proportional (`MarketSpec` charges `fee_rate x |traded notional|`, no impact,
no queue, no spread), which is the regime the cited theory is derived in.

**What would make it fail (named now, before any code ran).**

1. **Snap-to-flat gives back exactly what the boundary saved.** B-29's own
   named failure mode. The largest steps this strategy ever takes are regime
   exits -- vote to zero -- and those are precisely the steps the conservative
   arm now forces to complete in full. If the 43% turnover saving was mostly
   *in* those steps, D4 will show turnover flat or rising and the 43% figure
   was never bankable.
2. **The residual long was never the cause.** If ETH-A stays near -0.430 with
   flat reachable again, R-64's published diagnosis is wrong and the boundary
   destination is simply worse on this data for a reason not yet identified.
   This is a reportable answer, not a null.
3. **The novel arm's width profile is a turnover confound.** `band(f) =
   deadband * f**p` with p>0 makes the band *narrower everywhere below f=1*,
   and v4's exposure is below 1 most of the time. A narrower band trades more.
   If the novel arm simply trades more and earns more by tracking better, that
   is not a cost mechanism and D2 will say so -- its advantage will shrink,
   not grow, as the fee quadruples. The mean-band-width diagnostic below is
   reported alongside so the reader can see the confound directly rather than
   infer it.
4. **Both arms lag.** Both hold a position closer to the previous one than v4
   does in at least some states. Lag on a trend rule is the one thing this
   project has repeatedly measured as expensive (R-53's macro veto, R-60's
   CUSUM timing).

=====================================================================
DATA, SPLITS, AND WHAT EACH IS FOR
=====================================================================

BTC, the canonical Bitstamp 5m spot series (`data/btcusd_spot_5m.csv.gz`,
2017-01 -> 2026-08):

    inner-train       ...        -> 2020-12-31   fit, sweep, iterate freely
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              frozen config only, ONCE

ETH for the pre-registered falsification test, in two forms:

    ETH-A (primary)   `data/ethusd_bitfinex_5m.csv.gz`, 2016-03 -> 2019-12.
                      Costs **zero** holdout consultations (pre-2020 entirely,
                      the R-19/R-28 convention). It is also the exact window
                      R-64's conservative arm was killed on, which is what
                      makes the -0.430 prediction above checkable.
    ETH-B (secondary) `data/ethusd_coinbase_spot_5m.csv.gz`, 2023-01-01 ->.
                      Read ONLY in a holdout pass, and counted.

Both branches hard-truncate BTC at 2022-12-31 on load until their
configuration is frozen and committed.

=====================================================================
DECISION RULES, FROZEN (default is REJECT)
=====================================================================

Inherited from `r64_shared.py` **verbatim and deliberately** -- an arm that is
R-64's arm plus one conditional must be judged by R-64's bar, or the two
rounds' verdicts are not comparable. All comparisons are **arm vs
`kelly_regime_v4`**, paired on identical daily return series.

**D0 (RISK-MATCH GATE, and it binds).** Report `mean_notional` for the arm and
for v4 on every window. If `|c_arm - c_v4| / c_v4 > 0.10` on the decision
window, the D1 head-to-head is **VOID as a growth claim** and must be
re-reported against `ConstantExposureHold(c_arm)` instead, flagged as such. A
void D1 cannot promote. Three of this project's findings died for want of this
gate (R-31, R-32, R-33).

**D1 (PRIMARY, holdout, spot @ 0.10%).** Difference in total log growth, arm
minus v4, on daily returns, paired stationary block bootstrap (mean_block=30,
n_boot=2000, seed=7). Promotion requires **point estimate > 0 AND the 95%
interval excluding zero**.

**D2 (COST-MECHANISM TEST).** The same comparison re-run at the **0.40%**
taker tier. Both arms claim a cost mechanism, so:

        REQUIRE  Delta_logret(0.40%)  >  Delta_logret(0.10%)

An arm whose advantage shrinks or inverts as the fee quadruples is NEGATIVE
regardless of D1: it is a lag or an exposure difference wearing a cost story.

**D3 (PRE-REGISTERED FALSIFICATION).** ETH-A (2016-03 -> 2019-12, +0 holdout):
the sign of the arm-minus-v4 log-growth difference must be **positive**. A
mechanical improvement to a position-update rule is not supposed to be
asset-specific (R-57's lesson), so a sign flip on the one other instrument
with pre-2020 coverage refutes the mechanism.

**D3b (THE R-64 DIAGNOSIS TEST, conservative arm only, reported either way).**
On the identical ETH-A window, the conservative arm's difference vs v4 must be
**materially better than R-64's measured -0.430 at k=1**. "Materially" is
frozen here as **at least half the gap closed, i.e. > -0.215**. This is not a
promotion gate -- an arm can close the gap fully and still fail D1 -- it is a
test of a published causal claim, and its outcome is reported whichever way it
falls. If it fails, R-64's residual-long explanation is refuted and section C's
R-64 row must be annotated in place (nothing is deleted).

**D4 (TURNOVER SANITY).** Fees paid and trade count, arm vs v4, on every
window. Both arms are no-trade-region devices. State plainly if turnover
rises; for the conservative arm a rise is failure mode 1 above, and for the
novel arm it is failure mode 3.

**D5 (PLATEAU, NOT PEAK).** Any arm carrying a fitted parameter reports its
neighbourhood -- at least 4 neighbours -- on inner-validation, with the Sharpe
spread stated against the +/-0.2 noise floor (R-20). Neither arm here selects
a parameter on returns: the conservative arm's `k` and the novel arm's `p`
both have an a-priori frozen value and an identity point equal to v4, so D5 is
a shape check on the mechanism's own index. **The response must be monotone in
that index.** A non-monotone response is the signature that something other
than the named mechanism is moving the numbers -- exactly what R-64 found on
three of four cells, and the reason it is a stated check rather than a plot.

**D6 (FUNDING, futures).** If and only if an arm survives D1-D3, the futures
comparison is re-run with funding charged. **And note B-30**: `broker.py`'s
`REBALANCE_DEADBAND = 0.05` is 25% of equity at 5x and silently discards about
half of v4's own intended futures rebalances, so a futures figure for any arm
routing through `order_notional` is partly a measurement of that band. Both
arms therefore report `fill_through` (intended position changes -> actual
fills) on every futures cell, and no futures number is quoted without it.

**PROMOTION BAR (ROUTINE step 4, default REJECT).** Promote only if ALL hold:
D0 not void; D1 positive with interval excluding zero; D2 satisfied; D3
positive on ETH-A; D4 shows turnover falling; D5 monotone in the mechanism
index; and the arm still beats `buy_and_hold` out-of-sample after real costs.
Anything else is NEGATIVE and is written up with the same care.

**HOLDOUT BUDGET, DECLARED IN ADVANCE.** Zero reads before a configuration is
frozen and committed. An arm that fails D3 or D5 on the inner splits is
NEGATIVE and never reads the holdout at all. An arm that survives gets: 1 read
on BTC spot @0.10%, 1 at 0.40%, 1 on ETH-B, plus the paired v4 baselines on
the same windows (counted anyway). Estimated **+4 consultations per surviving
arm**; the exact count is reported in the ledger entry.

**TRIALS COUNT.** The total across BOTH branches, per ROUTINE.md's parallelism
rule. Every configuration evaluated is counted by `measure` below, including
the discarded ones.

=====================================================================
OPERATOR AMENDMENTS, DISCLOSED
=====================================================================

Two changes were made to this round's setup **after** the pre-registration was
first written and **before** either branch reported any performance number.
Both are recorded here rather than quietly applied, per this file's own
standard: what corrupts a pre-registration is moving a goalpost after seeing a
result, not fixing a specification before one exists.

1. **Renumbered R-65 -> R-66.** A concurrently-running session claimed the ID
   R-65 on `main` for a different round (holding period / rank buffer) while
   this file was being written. Renumbered on the R-31/R-32 same-day-parallel
   precedent. No content change.

2. **The conservative arm's snap conditional was corrected before it was
   measured.** The first draft of the branch brief specified

       if desired == 0.0 and pos != 0.0 and abs(desired - pos) > band:
           pos = 0.0

   which does not do what B-29 asks. The boundary rule parks the position at
   exactly `k*band` when the target hits zero, and `|0 - 0.10|` is not
   *greater than* `0.10`, so the snap never fires in precisely the state it
   exists for. An operator-side diagnostic run before either branch reported
   measured the consequence directly: that gated form still holds a nonzero
   position on 15.0% of v4's exactly-flat bars on BTC-inner and 34.6% on
   ETH-A. The frozen primary is therefore the unconditional form, which is
   also B-29's literal wording ("still snaps to exactly flat when
   `desired == 0`") and matches `kelly_regime_ev`'s own standing exception
   ("a full exit is always allowed"):

       if desired == 0.0:            pos = 0.0
       elif desired - pos > band:    pos = desired - k*band
       elif pos - desired > band:    pos = desired + k*band

   The gated form is retained as a labelled ablation, because the difference
   between the two isolates the residual long specifically.

The same operator diagnostic produced one fact that belongs in the record
regardless of either arm's verdict, and that no prior round had noticed:
**`kelly_regime_v4` itself carries a residual long on flat bars.** Its own
rule does not fire when the target falls from ~0.09 to 0.0, because that step
is inside its own 10% band. Measured on ETH-A, v4 is nonzero on 3.3% of its
own exactly-flat bars (mean 0.0032, max 0.0995); on BTC-inner it is 0.0%. The
disease R-64 diagnosed in the boundary arm is present in the incumbent, more
rarely and asset-dependently.

=====================================================================
CITATIONS
=====================================================================

No-trade region under **proportional** costs, and its boundary destination:
  Constantinides, G. M. (1986). "Capital Market Equilibrium with Transaction
    Costs." Journal of Political Economy 94(4), 842-862.
  Davis, M. H. A. & Norman, A. R. (1990). "Portfolio Selection with
    Transaction Costs." Mathematics of Operations Research 15(4), 676-713.
  Shreve, S. E. & Soner, H. M. (1994). "Optimal Investment and Consumption
    with Transaction Costs." Annals of Applied Probability 4(3), 609-692.
  Liu, H. (2004). "Optimal Consumption and Investment with Transaction Costs
    and Multiple Risky Assets." Journal of Finance 59(1), 289-338.

The band's **width profile in the exposure level** -- the novel arm's actual
content, i.e. that the half-width is proportional to (f^2 (1-f)^2 eps)^(1/3)
and therefore vanishes as f -> 0:
  Janecek, K. & Shreve, S. E. (2004). "Asymptotic analysis for optimal
    investment and consumption with transaction costs." Finance and
    Stochastics 8(2), 181-206.
  Rogers, L. C. G. (2004). "Why is the effect of proportional transaction
    costs O(delta^(2/3))?" In Mathematics of Finance, Contemporary
    Mathematics 351, 303-308.
  Gerhold, S., Guasoni, P., Muhle-Karbe, J. & Schachermayer, W. (2014).
    "Transaction costs, trading volume, and the liquidity premium." Finance
    and Stochastics 18(1), 1-37.
  Muhle-Karbe, J., Reppen, M. & Soner, H. M. (2017). "A primer on portfolio
    choice with small transaction costs." Annual Review of Financial
    Economics 9, 301-331. (Survey; sect. 6.3 is the proportional-vs-quadratic
    scope statement R-64 already leaned on.)

Growth-rate asymmetry around the log-optimal fraction (why the de-risking side
of a band is the expensive side to widen):
  MacLean, L. C., Thorp, E. O. & Ziemba, W. T. (2010). "Good and bad
    properties of the Kelly criterion." Quantitative Finance 10(7), 681-687.

Cost-aware implementation, and the one published band-vs-smoothing head-to-head
this project has:
  Novy-Marx, R. & Velikov, M. (2016). "A Taxonomy of Anomalies and Their
    Trading Costs." Review of Financial Studies 29(1), 104-147.

    **A CORRECTION THIS ROUND OWES ITS PREDECESSOR.** R-64's ledger entry
    reports this paper as "a hysteresis band preserves 0.77 of gross return
    against 0.62 for trade-smoothing, i.e. 38% more net return." An
    independent survey commissioned for this round read Table 5 directly and
    those numbers are **not** fractions of gross return preserved -- they are
    monthly gross excess returns in percent. The row reads: trading hysteresis
    gross 0.77, t-costs 0.26, **net 0.51 [t=2.87]**; staggered quarterly
    rebalancing gross 0.62, t-costs 0.26, **net 0.37 [t=2.34]**. The honest
    version is *stronger* than the version R-64 carried -- the two mitigations
    cut costs identically and the band preserves far more gross signal -- but
    the ledger's phrasing is wrong and is annotated in place there, not
    deleted. Also: NMV's rule is itself **asymmetric** (an sS rule; the entry
    threshold is strictly stricter than the exit threshold), which R-64 did
    not note and which is directly relevant to this round.

    Their own stated mechanism is cross-sectional substitution -- "holding
    (not selling) close substitutes to the stocks you would have bought" --
    so the 41%/42% turnover and cost reductions should be assumed to be ~0 on
    one instrument with zero netting. What transfers is the *ordering* and the
    sS shape, not the magnitude.

The single-instrument, proportional-cost, non-asymptotic band literature -- the
closest published model to this repo, and the reason the novel arm's width
profile is the right object even though its specific f**p form is a
simplification:
  Martin, R. J. (2012). "Optimal multifactor trading under proportional
    transaction costs." arXiv:1204.6488. (Preprint; no journal publication
    verified. Band half-width ~ (cost)^(1/3) x (volatility of the TARGET
    position)^(2/3), so a faster signal is buffered more; and the band's
    *centre* is displaced in the direction the target is expected to move,
    at order cost^(2/3). Martin judges that displacement negligible -- but
    explicitly for a small cost-per-unit-volatility, and on 5m BTC bars this
    project is not in that regime.)
  de Lataillade, J. & Chaouki, A. (2020). "Equations and Shape of the Optimal
    Band Strategy." arXiv:2003.04646. (Preprint/CFM working paper. Single
    asset, linear proportional cost, OU predictor, non-asymptotic. Two results
    that bear directly on both arms: at zero signal the band is exactly
    symmetric with half-width ~ (3/2 x Gamma x beta^2)^(1/3); at large signal
    it becomes "completely asymmetric", the risk-reducing edge collapsing onto
    the target while the risk-increasing edge sits far away. Their simulated
    head-to-head against a grid-search-optimal *constant symmetric* band shows
    the advantage growing monotonically with the cost ratio -- the same shape
    as this round's D2.)

**AND THE ONE THAT CUTS AGAINST THE CONSERVATIVE ARM, RECORDED BEFORE ITS
NUMBER IS READ.** The static-target result that the band width vanishes as the
target goes to zero (Gerhold et al. 2014 sect. 3.3; Muhle-Karbe, Reppen & Soner
2017 sect. 5.1, "both of these expressions vanish if zero or full investment is
optimal in the frictionless model") does **not** survive a signal-driven
target. The Primer's general formula (eqs. 4.14-4.15) has a strictly positive
band-width floor everywhere once the target is sensitive to a factor and that
factor is not perfectly correlated with price, and its sect. 5.2 says so
outright: "the no-trade region no longer vanishes if the frictionless risky
weight is zero or one." de Lataillade & Chaouki put it plainly: "even if we
trade, the optimal policy is not to trade directly towards zero: indeed, once
close enough from zero, one can afford to wait a little bit to see whether the
predictor becomes positive or negative."

So **transaction-cost theory does not license snapping to exactly flat.** That
does not retract B-29 -- B-29's motivation was never cost-optimality, it was
that R-64's arm carried a residual long through bear regimes, which is a
*regime-risk* argument -- but it does mean the conservative arm must be
defended on regime-risk grounds and must NOT be described as the
Constantinides/Davis-Norman optimum with a fix. What the theory licenses is the
novel arm's shape: a band that *tapers* toward zero rather than a
discontinuous snap. That the two arms disagree on this point is now a
pre-registered expectation rather than a post-hoc reading.
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

#: Both branches truncate BTC here until their config is frozen and committed.
INNER_CUTOFF = "2022-12-31"

BOOT_KW = dict(mean_block=30.0, n_boot=2_000, seed=7)

FEE_BASE = 0.0010
FEE_STRESS = 0.0040

RISK_MATCH_TOL = 0.10  # D0

#: D3b: R-64's conservative arm scored this on ETH-A at k=1. The residual-long
#: diagnosis predicts a materially better number once flat is reachable.
R64_ETH_A_K1 = -0.430
D3B_BAR = -0.215  # "at least half the gap closed"

SHARPE_NOISE_FLOOR = 0.2  # R-20

_CONFIGS = [0]


def spot(fee: float = FEE_BASE) -> MarketSpec:
    return MarketSpec.spot(fee_rate=fee)


def futures(fee: float = 0.0005, leverage: float = 5.0) -> MarketSpec:
    return MarketSpec.futures(leverage=leverage, fee_rate=fee)


# --------------------------------------------------------------------- data

def load_btc() -> pd.DataFrame:
    return load_ohlcv_csv(DATA / "btcusd_spot_5m.csv.gz")


def load_btc_inner() -> pd.DataFrame:
    """BTC hard-truncated at 2022-12-31: the holdout cannot be read by
    accident from a frame that does not contain it."""
    df = load_btc()
    return df.loc[:INNER_CUTOFF]


def load_eth_a() -> pd.DataFrame:
    """ETH-A: Bitfinex 2016-03 -> 2019-12. Zero holdout cost."""
    return load_ohlcv_csv(DATA / "ethusd_bitfinex_5m.csv.gz")


def load_eth_b() -> pd.DataFrame:
    """ETH-B: Coinbase. Only read from 2023-01-01, and counted when read."""
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
    """Arm vs v4 on one window/market, with everything D0-D4 needs."""
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
    mean notional."""
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


def d3b_satisfied(d_logret_eth_a: float) -> bool:
    """R-64's residual-long diagnosis, made falsifiable. Not a promotion
    gate; reported whichever way it falls."""
    return d_logret_eth_a > D3B_BAR


def monotone(values) -> bool:
    """D5's shape check: the response must be monotone in the mechanism's own
    index. Ties are allowed; a sign change is not."""
    d = np.diff(np.asarray(values, dtype=float))
    d = d[np.isfinite(d)]
    return bool(np.all(d >= -1e-12) or np.all(d <= 1e-12))


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
        return "NEGATIVE: D5 (non-monotone in the mechanism index)"
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
