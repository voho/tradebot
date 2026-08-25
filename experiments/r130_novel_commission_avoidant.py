"""R-130 NOVEL branch: a commission-avoidant meta-ensemble wrapping the
frozen `hedge_experts` blend (08-25).

**Direction for this round** (see the dispatch prompt / `docs/LEDGER.md`
R-128, R-129): two prior rounds bolted a Kelly quadratic-cost no-trade BAND
(Constantinides 1986 / Davis & Norman 1990) onto `hedge_experts`'s output at
every application point this project's framework can construct -- banding
the final blended signal (R-128, both a fixed and an adaptive horizon) and
banding the raw pre-blend experts or bucket sub-blends (R-129, both
branches). All four failed. R-129's own closing line named the needed next
step explicitly: a cost model **outside the Kelly-quadratic-cost family
entirely**. This round answers that with a different MECHANISM, not a
third application point of the same algebra.

**Citation and what it says.** Uziel, G., & El-Yaniv, R. (2016), "Online
Learning of Commission Avoidant Portfolio Ensembles," arXiv:1605.00788
(fetched and read in full this session, both the arXiv HTML/PDF source and
its own references; no part of the mechanism below is guessed at). Their
**CAPE** algorithm wraps a set of `d` commission-oblivious online
portfolio-selection base algorithms. At every round it introduces one
extra, *artificial* expert whose recommended portfolio is defined to be
`b_hat_t` -- the ensemble's OWN current (post-drift) holding. Geometrically
(their Figure 1), this "stay" expert extends next round's feasible convex
hull to include the region between the ensemble's current holding and the
base algorithms' new recommendations, so a convex combination that leans on
the artificial expert can absorb a chase-worthy move partially, or not at
all, without ever paying to leave the current holding. CAPE then learns a
weight vector over {base algorithms, artificial stay expert} via a
regularized online Newton step (a COMID-style update extended to
exp-concave losses, Algorithm 1), where the regularizer `R(w') =
||w'_1..d||_1` explicitly penalizes weight mass AWAY from the stay expert
every single round -- not only when a trade is actually taken -- and the
accumulated-curvature matrix `A_t` gives the update a shrinking,
confidence-weighted step size as evidence accrues (their Lemma 2 is exactly
what turns this into a logarithmic, not merely sublinear, regret bound
against the best fixed regularized mixture in hindsight, Corollary 1).
Empirically (their Table 1) CAPE, wrapping strong-but-commission-fragile
mean-reversion base algorithms (OLMAR, Anticor, PAMR) plus a CRP-driven
control (EG, included specifically to show CAPE learns to ignore it),
beats every existing commission-aware method across four commission tiers
on six public equity datasets, allocating almost all of its live weight to
whichever base algorithm is currently paying off and using the artificial
expert to sit out the rest.

**What had to be adapted, disclosed plainly (not silently).** Three of the
paper's assumptions do not hold in this project's setting, and each
adaptation is a deliberate, disclosed choice, not a silent substitution:

1. **The paper's objects are full stock-simplex portfolios (`b in
   Delta^n`); this project's object is a single scalar leveraged directional
   position in `[-1, 1]`** (`ctx.order_target`'s own convention -- this
   project trades one instrument, not a cross-section). CAPE's simplex
   machinery (the `n`-stock allocation vector, the log-wealth loss
   `-log(<X_t, P_t^+ w>)` built for multiplicative per-asset relative
   prices) does not apply to a directional bet. The adaptation keeps CAPE's
   OWN two central objects -- (a) a meta-weight over {base algorithm(s),
   artificial "stay" expert}, learned online; (b) a played output that is a
   CONVEX COMBINATION of the base recommendation and the current holding,
   not a hard switch -- and replaces the log-wealth loss with the
   position's own realized bar-level reward net of the ACTUAL transaction
   cost this project's broker charges (`fee_rate * leverage * |Delta
   position|`, matching `PaperBroker._transact`'s own fee model exactly),
   regularized every round by `lambda * w` (the direct scalar analogue of
   `R(w') = ||w'||_1`, since with one real base algorithm and one artificial
   expert the "weight on the non-artificial experts" is just the scalar
   `w`).
2. **One base algorithm, not `d`.** The paper's own empirical section
   ensembles four base algorithms so CAPE has something nontrivial to pick
   among; `hedge_experts` already IS an ensemble (a discounted-Hedge blend
   of ten technical experts). Re-decomposing it into ten raw signals for
   CAPE to re-weight would collide with R-129's own pre-blend construction
   (banding the ten raw experts individually) -- a different mechanism
   wrapping the SAME object CAPE's own paper never assumed was already an
   ensemble. Per the dispatch brief's own instruction, this branch instead
   treats the frozen, already-blended `hedge_experts` output itself as
   CAPE's ONE commission-oblivious base algorithm (`d=1`), and adds
   CAPE's own artificial stay expert as the second and only other "expert"
   -- so the object being wrapped is the single already-blended target `x`,
   never the ten `HedgeExperts._experts` columns or `HedgeExperts.prepare`'s
   own `logw`/`p` recursion, both left completely untouched and
   unimported-into by this file's own update logic (this file only calls
   the registered `HedgeExperts.prepare()` once, verbatim, to read its
   `target` column). This is what keeps the construction STRUCTURALLY
   DISTINCT from a downstream band, named concretely: a band is a
   deterministic, memoryless (beyond "last held value") geometric rule --
   `if |target - held| > f(fee, horizon, sigma): jump fully to target, else
   freeze exactly` -- with no notion of whether jumping would actually have
   paid off. CAPE's meta-weight `w` here is instead PERSISTENT LEARNING
   STATE (a scalar weight plus an accumulated curvature term `A_t`,
   Algorithm 1's own ONS machinery, here specialized to a 1-D weight since
   the base-algorithm-plus-artificial-expert simplex collapses to an
   interval when `d=1`) that moves by a REGRET-MINIMIZING GRADIENT STEP
   evaluated against the REALIZED, REALIZED-COST-NET reward of last round's
   choice -- i.e. it can only drift toward "chase" when chasing has
   recently been paying off net of the regularization pressure, and toward
   "stay" otherwise, continuously, not via a one-shot distance threshold.
   See "What would make this collapse into a band" below for the concrete,
   named test of whether this distinction actually held up empirically.
3. **The regularized Newton step is specialized to its own 1-D case.**
   With one base algorithm and one artificial expert the probability
   simplex CAPE optimizes over is exactly the interval `w in [0, 1]`
   (`1-w` on the artificial expert), so Algorithm 1's `n`x`n` Bregman/Newton
   matrix machinery over the full simplex collapses to a scalar
   AdaGrad/Online-Newton-Step update (`A_t = A_{t-1} + grad_t^2`, `w_{t+1} =
   clip(w_t - eta/sqrt(A_t) * grad_t, 0, 1)`) -- the standard scalar
   specialization of exactly the diagonal-approximation family Algorithm 1
   belongs to, not a different algorithm. `eta` and the curvature seed `a0`
   are fixed (not swept; the paper's own empirical section fixes its
   analogous `eta = epsilon = 1` and treats only `lambda` as the free
   "commission avoidance" knob -- this branch follows that same convention
   and sweeps `lambda` for B3), calibrated once by inspecting the resulting
   `w_t` trajectory on BTC spot inner-train for gross pathology (immediate,
   permanent boundary saturation within the first handful of bars) rather
   than fit to any realized Sharpe or return.

**Mechanism, one sentence:** learn a single online, regret-minimizing
meta-weight `w_t in [0,1]` over {`hedge_experts`'s own frozen blended
target, an artificial "stay exactly where you currently are" expert}, play
the convex combination `x_t = w_t * hedge_target_t + (1-w_t) * held_t`, and
let `w_t` drift toward "stay" under a constant per-round regularization
pressure `lambda` unless the REALIZED, cost-net reward of chasing clears
it.

**Why this should cut cost without destroying the signal.** Unlike a band,
which freezes purely on distance and is blind to whether the frozen expert
mix is currently earning its keep, `w_t` here is pulled toward "chase" only
when doing so has recently been paying off (the gradient's `-r_t` term)
by more than the fixed regularization drag (`+lambda` every round,
matching CAPE's own unconditional per-round `R(w)` penalty) net of the
ACTUAL fee/leverage-scaled transaction cost of the move that produced the
realized outcome being scored. So it should give back less turnover than
`hedge_experts`'s own bare hysteresis exactly in the regimes where
`hedge_experts`'s edge is weak or absent (where a band, blind to realized
P&L, cannot distinguish "expensive because pointless" from "expensive
because it's currently working"), while tracking the target closely when
the edge is real. `A_t`'s accumulated-curvature shrinkage additionally lets
`w_t` become progressively less reactive to single-bar noise as evidence
accrues within a run, the source of Algorithm 1's own logarithmic- (rather
than merely sublinear-) regret guarantee against the best fixed mixture in
hindsight.

**What would make this fail, named now, before any code was run:**

1. **The sharpest named risk -- collapse into a band.** If `w_t` in
   practice saturates at the `{0, 1}` boundary almost every bar (a
   near-bang-bang process), the "continuously learned convex combination"
   claim above is cosmetic: the mechanism has degenerated into a
   deterministic on/off switch under a different name, and this branch has
   NOT actually tested anything structurally different from R-128/R-129's
   closed band axis, whatever its parameters are called. Concretely
   falsifiable: report the fraction of live bars with `w_t` strictly
   interior (`0.01 < w_t < 0.99`) against baseline's own turnover; if that
   fraction is near zero, say so plainly rather than calling the result a
   new mechanism.
2. **Too aggressive a `lambda` destroys the edge; too weak reproduces the
   baseline exactly.** Because `lambda` is now the one free knob (matching
   the paper's own single free "commission avoidance" parameter), a result
   that only clears the promotion bar at one exact `lambda` and flips sign
   at 0.5x or 2x of it is a peak, not a plateau -- B3 is the pre-registered
   check.
3. **The same repeated BTC-pass/ETH-invert signature** this project has
   now measured independently 6+ times (R-109, R-113, R-115-conservative,
   R-125-conservative, R-126 both branches, R-128 conservative weakly, and
   R-129's own closing diagnosis) is a real, standing prior that a NEW
   mechanism on the SAME underlying object (`hedge_experts`) does the same
   thing regardless of what wraps it. B4 is the test built to catch this.
4. **The broker's own `REBALANCE_DEADBAND = 0.05`** already ignores small
   same-sign target adjustments regardless of what this strategy emits
   (`tradebot/broker.py`, `_execute_target`) -- so a candidate that only
   shrinks the SIZE of `hedge_experts`'s own moves (without changing sign
   or reaching a full close) might show little or no improvement in
   `num_trades` (executed round trips) even if the meta-weight is doing
   real work; `fees_paid` and the raw `ctx.order_target` call count are
   reported alongside `num_trades` so a real cost reduction is not missed
   just because the round-trip counter is insensitive to it, and equally so
   a lack of improvement in `num_trades` is not over-read as "the mechanism
   did nothing" without checking the finer-grained numbers first.
5. **Cold-start convention, disclosed.** `self._held`, `self._x_prev`,
   `self._x_prevprev`, `self._g_prev`, `self._held_prev` are all seeded, on
   the FIRST live `on_bar` call only (`warmup` = 2,500 of ~404,000
   inner-train bars, 0.6%, same order as R-129's own disclosed cold-start
   artifact), to that bar's own `hedge_target` value (there is no prior
   position or prior recommendation to reference yet) and `w` is seeded at
   `0.5`, CAPE's own paper's literal uniform prior over `{base algorithm,
   artificial expert}` at `d=1`. This is a minor, bounded artifact, not a
   lookahead: every subsequent update uses only that bar's and earlier
   bars' realized data.

**Not a duplicate of:** R-128 (single band on the blended output, fixed or
AR(1)-adaptive horizon) or R-129 (per-expert or per-bucket bands, all
pre-blend) -- neither of those constructions has any persistent LEARNED
weight or state beyond "last accepted value"; both are one-shot geometric
distance rules recomputed fresh every bar from `fee`/`horizon`/`sigma`
alone, with no dependence on realized reward. This branch's `w_t` is
online-learned FROM realized, cost-net reward, carries genuine multi-bar
state (`A_t`, `w_t`) that a band construction has no analogue of, and its
update rule is a regret-minimizing gradient step (Uziel & El-Yaniv 2016)
rather than the Kelly quadratic-cost algebra (Constantinides 1986; Davis &
Norman 1990) both R-128 and R-129 used in every one of their four
constructions.

**Implementation note, causality.** `prepare()` calls the registered,
UNMODIFIED `HedgeExperts().prepare(df.copy())` once, verbatim -- this
branch never reimplements or edits `HedgeExperts`'s own `_experts`/
`logw`/`p` construction -- and copies its `target` column plus a plain
`log_ret = diff(log(close))` column (both causal: row `i` depends only on
rows `<= i`) into this strategy's own frame. All CAPE meta-weight state
(`w`, `A`, `held`, `x_prev`, `x_prevprev`, `g_prev`, `held_prev`) lives on
`self`, updated once per live `on_bar` call in strict bar order (matching
`HedgeExperts.prepare`'s own `logw`/`p` loop's own causal discipline),
never read from or written to the dataframe -- there is no broker-side
"held" state to read back for a synthetic artificial-expert construction,
so it must be tracked by the strategy itself, exactly as R-129's own
per-expert `_held` state was.

No bar at or after `OOS_START = 2023-01-01` is read anywhere in this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.inference import annualized_sharpe, daily_returns, paired_bootstrap  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.hedge_experts import HedgeExperts  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# ----------------------------------------------------------------------
# Splits -- identical convention to every prior round. The holdout
# (>= OOS_START) is never read by this file.
# ----------------------------------------------------------------------
INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"  # do not read; guarded by _assert_no_holdout below

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT_HIGH_FEE = MarketSpec.spot(fee_rate=0.0040)          # B5: 0.40% taker tier
FUTURES_HIGH_FEE = MarketSpec.futures(leverage=5.0, fee_rate=0.0040)

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# B3 sweep multipliers on this branch's own key parameter (lambda, the
# commission-avoidance regularization strength) -- identical convention to
# R-128/R-129's own B3.
B3_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)
LAMBDA_BASE = 5e-5  # calibrated once (this session) by inspecting the
                     # resulting w_t trajectory on BTC spot inner-train for
                     # gross pathology, per docstring point 3. A first guess
                     # of 5e-4 (HedgeExperts's own internal `fee_rate`
                     # constant, reused naively for scale) collapsed w_t to
                     # the w=0 boundary on 84.5% of live bars (median
                     # gradient == lambda almost exactly, i.e. the constant
                     # per-round regularization pressure dominated realized-
                     # reward noise at this project's 5-minute bar frequency
                     # -- CAPE's own paper applies its regularizer once per
                     # TRADING DAY, ~288x less often than once per 5-min
                     # bar, so a lambda calibrated by naive constant-reuse
                     # from a daily-round paper is structurally too strong
                     # here). 5e-5 (100x smaller) was the point at which
                     # w_t stopped saturating the {0,1} boundary and became
                     # fully interior on this same probe; this is a scale
                     # sanity check on the MECHANISM (does it behave like a
                     # continuous learner at all, per failure mode 1 above),
                     # not a fit to any realized Sharpe or return -- B3 below
                     # sweeps around this calibration point at {0.5,1,2,4}x.


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {OOS_START}")


def load_btc_train(kind: str = "spot"):
    df, label = load_dataset(ROOT / "data", kind)
    train = df.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(train)
    return train, label


def load_eth_train():
    eth = load_coinbase_eth_spot(ROOT / "data")
    assert eth is not None, "ETH spot data not committed"
    eth = eth.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(eth)
    return eth


# ----------------------------------------------------------------------
# Baseline (unmodified hedge_experts) run/metric helper.
# ----------------------------------------------------------------------

def run_baseline(df: pd.DataFrame, market: MarketSpec, start: str, end: str,
                  label: str = ""):
    strat = get_strategy("hedge_experts")
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


# ----------------------------------------------------------------------
# The candidate strategy: CAPE-style commission-avoidant meta-ensemble
# wrapping the frozen hedge_experts blend. NOT registered (experimental).
# ----------------------------------------------------------------------

class CAPEHedge(Strategy):
    """Commission-avoidant meta-ensemble (Uziel & El-Yaniv 2016, CAPE)
    wrapping the frozen `hedge_experts` blend as its one commission-
    oblivious base algorithm, plus CAPE's own artificial "stay" expert.

    See this module's own docstring for the full mechanism, the exact
    adaptation from the paper's simplex-over-stocks setting to this
    project's scalar leveraged-position setting, and the pre-registered
    failure modes.
    """

    name = "r130_cape_commission_avoidant"  # experimental; NOT @register-ed
    warmup = 2500  # matches HedgeExperts.warmup: this strategy is downstream of it

    def __init__(self, lam: float = LAMBDA_BASE, eta: float = 0.01,
                 a0: float = 1.0) -> None:
        self.lam = lam
        self.eta = eta
        self.a0 = a0
        # Instance state, seeded on the first live on_bar() call.
        self._w = None
        self._A = None
        self._held = None
        self._x_prev = None
        self._x_prevprev = None
        self._g_prev = None
        self._held_prev = None
        self._last_target = None
        # Diagnostics (not used for any decision, reported only).
        self.n_retarget_calls = 0
        self.w_history: list[float] = []

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # Verbatim, unmodified HedgeExperts.prepare() -- this branch never
        # touches HedgeExperts's own _experts/logw/p construction. Market-
        # independent, exactly like HedgeExperts's own prepare().
        base = HedgeExperts()
        prepared = base.prepare(df.copy())
        df = df.copy()
        df["hedge_target"] = prepared["target"].to_numpy()
        df["log_ret"] = np.log(df["close"]).diff().to_numpy()
        return df

    def on_bar(self, ctx: Context) -> None:
        g = float(ctx.bar["hedge_target"])
        r = ctx.bar["log_ret"]
        r = float(r) if np.isfinite(r) else 0.0
        fee = ctx.market.fee_rate
        lev = ctx.market.leverage

        if self._w is None:
            # Cold start: see docstring point 5. Seed "stay" with this
            # bar's own hedge target (nothing to stay at yet) and w at
            # CAPE's own uniform prior over {base algorithm, artificial
            # expert} at d=1.
            self._w = 0.5
            self._A = self.a0
            self._held = g
            self._x_prev = g
            self._x_prevprev = g
            self._g_prev = g
            self._held_prev = g
        else:
            # Regret-minimizing (scalar ONS/AdaGrad) step on last round's
            # realized, cost-net reward, regularized every round toward the
            # artificial "stay" expert (CAPE's own R(w) penalty, applied
            # unconditionally, matching Algorithm 1's "suffer loss g_t(w_t)
            # + lambda*R(w_t)" every round).
            grad = ((self._g_prev - self._held_prev)
                    * (fee * lev * float(np.sign(self._x_prev - self._x_prevprev)) - r)
                    + self.lam)
            self._A += grad * grad
            step = self.eta / np.sqrt(self._A)
            self._w = float(np.clip(self._w - step * grad, 0.0, 1.0))

        self.w_history.append(self._w)

        x = self._w * g + (1.0 - self._w) * self._held
        if self._last_target is None or abs(x - self._last_target) > 1e-9:
            ctx.order_target(x)
            self._last_target = x
            self.n_retarget_calls += 1

        # Roll state forward for next round.
        self._g_prev = g
        self._held_prev = self._held
        self._x_prevprev = self._x_prev
        self._x_prev = x
        self._held = x


def run_candidate(df: pd.DataFrame, market: MarketSpec, start: str, end: str,
                   label: str = "", lam: float = LAMBDA_BASE):
    strat = CAPEHedge(lam=lam)
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res, strat


def _delta_sharpe(res_cand, res_base, label: str) -> dict:
    """Paired-bootstrap Delta-Sharpe (candidate - baseline) on daily returns."""
    da = daily_returns(res_cand.equity)
    db = daily_returns(res_base.equity)
    n = min(len(da), len(db))
    da, db = da.iloc[-n:].to_numpy(), db.iloc[-n:].to_numpy()
    pr = paired_bootstrap(da, db, annualized_sharpe, mean_block=30.0, seed=7)
    print(f"  [{label}] cand_sharpe(daily)={pr.stat_a:.3f} base_sharpe(daily)={pr.stat_b:.3f} "
          f"d_sharpe={pr.diff.point:+.3f} CI=[{pr.diff.lo:+.3f}, {pr.diff.hi:+.3f}] "
          f"p_positive={pr.p_positive:.3f} significant={pr.significant}")
    return {"label": label, "d_sharpe": pr.diff.point, "lo": pr.diff.lo, "hi": pr.diff.hi,
            "p_positive": pr.p_positive, "significant": pr.significant}


if __name__ == "__main__":
    n_configs = 0
    results = {}

    print("=" * 70)
    print("Cold-start / mechanism sanity check (BTC spot, inner-train)")
    print("=" * 70)
    df_btc, label_btc = load_btc_train("spot")
    m_c, res_c, strat_c = run_candidate(df_btc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label_btc)
    n_configs += 1
    w_hist = np.array(strat_c.w_history)
    frac_interior = float(np.mean((w_hist > 0.01) & (w_hist < 0.99)))
    print(f"w_t: mean={w_hist.mean():.4f} std={w_hist.std():.4f} "
          f"frac_interior(0.01,0.99)={frac_interior:.4f} "
          f"frac_at_0={np.mean(w_hist < 0.01):.4f} frac_at_1={np.mean(w_hist > 0.99):.4f}")
    print(f"candidate: trades={m_c.num_trades} retarget_calls={strat_c.n_retarget_calls} "
          f"final={m_c.final_balance:.1f} sharpe={m_c.sharpe:.3f} fees={m_c.fees_paid:.2f}")

    print()
    print("=" * 70)
    print("Causal-truncation probe (mandatory hygiene, not a promotion gate)")
    print("=" * 70)
    m_full, _, _ = run_candidate(df_btc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label_btc)
    n_configs += 1
    df_trunc = df_btc.loc[:INNER_TRAIN_END].copy()
    m_trunc, _, _ = run_candidate(df_trunc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label_btc)
    n_configs += 1
    ok = np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9)
    print(f"causal truncation probe: {'PASS' if ok else 'FAIL'} "
          f"({m_full.final_balance} vs {m_trunc.final_balance})")
    assert ok, "CAPEHedge reads ahead of its own truncation point"

    # ------------------------------------------------------------ B1
    print()
    print("=" * 70)
    print("B1: candidate vs frozen hedge_experts, both markets, both periods (BTC)")
    print("=" * 70)
    periods = [
        ("full inner (train+val)", INNER_TRAIN_START, INNER_VAL_END),
        ("inner-val alone", INNER_VAL_START, INNER_VAL_END),
    ]
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        for per_name, start, end in periods:
            m_cand, res_cand, strat = run_candidate(df_btc, mkt, start, end, label_btc)
            n_configs += 1
            m_base, res_base = run_baseline(df_btc, mkt, start, end, label_btc)
            n_configs += 1
            print(f"[BTC {mkt_name} / {per_name}] "
                  f"cand: trades={m_cand.num_trades} retargets={strat.n_retarget_calls} "
                  f"final={m_cand.final_balance:.1f} sharpe={m_cand.sharpe:.3f} fees={m_cand.fees_paid:.1f} | "
                  f"base: trades={m_base.num_trades} final={m_base.final_balance:.1f} "
                  f"sharpe={m_base.sharpe:.3f} fees={m_base.fees_paid:.1f}")
            key = f"B1_btc_{mkt_name}_{per_name.replace(' ', '_')}"
            results[key] = _delta_sharpe(res_cand, res_base, key)

    # ------------------------------------------------------------ B3
    print()
    print("=" * 70)
    print("B3: plateau check -- sweep lambda at {0.5,1,2,4}x on BTC, inner-val")
    print("=" * 70)
    m_base_spot, res_base_spot = run_baseline(df_btc, SPOT, INNER_VAL_START, INNER_VAL_END, label_btc)
    n_configs += 1
    m_base_fut, res_base_fut = run_baseline(df_btc, FUTURES, INNER_VAL_START, INNER_VAL_END, label_btc)
    n_configs += 1
    for mult in B3_MULTIPLIERS:
        lam = LAMBDA_BASE * mult
        for mkt_name, mkt, m_base, res_base in (
            ("spot", SPOT, m_base_spot, res_base_spot),
            ("futures_5x", FUTURES, m_base_fut, res_base_fut),
        ):
            m_cand, res_cand, strat = run_candidate(df_btc, mkt, INNER_VAL_START, INNER_VAL_END,
                                                     label_btc, lam=lam)
            n_configs += 1
            print(f"[BTC {mkt_name} / lam={lam:.2e} ({mult}x)] "
                  f"cand sharpe={m_cand.sharpe:.3f} trades={m_cand.num_trades} | "
                  f"base sharpe={m_base.sharpe:.3f} trades={m_base.num_trades}")
            key = f"B3_btc_{mkt_name}_{mult}x"
            results[key] = _delta_sharpe(res_cand, res_base, key)

    # ------------------------------------------------------------ B4
    print()
    print("=" * 70)
    print("B4 (pre-registered falsification): ETH spot, inner-val")
    print("=" * 70)
    df_eth = load_eth_train()
    m_cand_eth, res_cand_eth, strat_eth = run_candidate(df_eth, SPOT, INNER_VAL_START, INNER_VAL_END, "eth_spot")
    n_configs += 1
    m_base_eth, res_base_eth = run_baseline(df_eth, SPOT, INNER_VAL_START, INNER_VAL_END, "eth_spot")
    n_configs += 1
    print(f"[ETH spot / inner-val] cand: trades={m_cand_eth.num_trades} "
          f"final={m_cand_eth.final_balance:.1f} sharpe={m_cand_eth.sharpe:.3f} | "
          f"base: trades={m_base_eth.num_trades} final={m_base_eth.final_balance:.1f} "
          f"sharpe={m_base_eth.sharpe:.3f}")
    results["B4_eth_spot_inner_val"] = _delta_sharpe(res_cand_eth, res_base_eth, "B4_eth_spot_inner_val")

    # ------------------------------------------------------------ B5
    print()
    print("=" * 70)
    print("B5: 0.40% taker fee tier, BTC, both markets, inner-val")
    print("=" * 70)
    for mkt_name, mkt in (("spot_hi_fee", SPOT_HIGH_FEE), ("futures_5x_hi_fee", FUTURES_HIGH_FEE)):
        m_cand, res_cand, strat = run_candidate(df_btc, mkt, INNER_VAL_START, INNER_VAL_END, label_btc)
        n_configs += 1
        m_base, res_base = run_baseline(df_btc, mkt, INNER_VAL_START, INNER_VAL_END, label_btc)
        n_configs += 1
        print(f"[BTC {mkt_name} / inner-val] cand: trades={m_cand.num_trades} "
              f"final={m_cand.final_balance:.1f} sharpe={m_cand.sharpe:.3f} | "
              f"base: trades={m_base.num_trades} final={m_base.final_balance:.1f} sharpe={m_base.sharpe:.3f}")
        key = f"B5_btc_{mkt_name}"
        results[key] = _delta_sharpe(res_cand, res_base, key)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key, r in results.items():
        print(f"{key:40s} d_sharpe={r['d_sharpe']:+.3f} CI=[{r['lo']:+.3f},{r['hi']:+.3f}] "
              f"significant={r['significant']}")
    print()
    print(f"TOTAL CONFIGURATIONS EVALUATED: {n_configs}")
