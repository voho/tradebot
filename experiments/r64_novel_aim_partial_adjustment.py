#!/usr/bin/env python
"""R-64 NOVEL branch: Garleanu & Pedersen (2013) applied to kelly_regime_v4 --
a partial-adjustment trading rate toward a **persistence-weighted aim**,
carried with a minimum-step filter so the quadratic-cost policy is tested at
its strongest on a purely proportional-cost simulator.

Not registered: lives under ``experiments/`` per ROUTINE.md step 5;
``src/tradebot/strategies/`` and ``src/tradebot/broker.py`` are untouched by
this branch. The round's pre-registration -- constraint attacked, the
not-a-duplicate list, the frozen decision rules D0-D6, the promotion bar and
the three named failure modes -- lives in ``experiments/r64_shared.py`` and is
NOT restated here, NOT edited here, and NOT reinterpreted here. The parallel
CONSERVATIVE branch (trade-to-the-boundary, Constantinides 1986 / Davis &
Norman 1990 / Liu 2004) owns a disjoint file which this one never reads.

=====================================================================
THE MECHANISM, AND THE TWO SEPARABLE CLAIMS IT CONTAINS
=====================================================================

Garleanu, N. & Pedersen, L. H. (2013), "Dynamic Trading with Predictable
Returns and Transaction Costs", *Journal of Finance* 68(6), 2309-2340, solve
the dynamic portfolio problem under **quadratic** (temporary-impact) trading
costs and get a closed form with two separate parts. Both are implemented
here, and -- this is the point of the file -- both are switchable, because
they are separate claims and the round is worth more if it can say which half
(if either) does anything.

**(i) The trading rate.** The optimal position does not jump to the Markowitz
portfolio and does not sit in a no-trade region. It closes a constant fraction
``a`` of the distance to an *aim* portfolio each period:

        x_t = (1 - a) x_{t-1} + a * aim_t

``a`` is increasing in risk aversion times risk and decreasing in the cost
coefficient: larger costs => smaller ``a`` => trade more slowly. Here ``a`` is
carried as an explicit swept parameter rather than reverse-engineered from a
cost coefficient this simulator does not have (there is no impact term to
calibrate ``Lambda`` against -- see CATEGORY ERROR below). ``a = 1`` recovers
"jump to the target", which is v4's own destination rule, so the incumbent's
update rule is literally a cell of this file's grid.

**(ii) The persistence-weighted aim.** GP's aim is *not* the myopic target.
It is a persistence-weighted blend of the current and expected-future
targets. For a return-predicting signal ``i`` with mean-reversion (decay)
rate ``phi_i``, GP's weight on that signal inside the aim is proportional to

        w_i  =  1 / (1 + phi_i / a)

so a signal that **decays slowly** (small ``phi``) is barely shrunk -- it will
still be there tomorrow, so it is worth paying to chase -- while a
fast-decaying signal is shrunk toward zero, because by the time you have
finished trading into it, it is gone.

v4's directional vote ``frac`` is the **equal-weighted** mean of three latched
anchor votes at 20 / 40 / 80 days. Under GP's own result equal weighting is
wrong: the 20-day anchor flips far more often than the 80-day one, therefore
decays faster, therefore should be down-weighted in the aim. Replacing the
flat mean with the ``w_k``-weighted mean is the genuinely novel half of this
branch -- half (i) is a re-derivation of a position-update rule, half (ii) is
a claim about the incumbent's own signal blend that nobody here has tested.

The blend is **normalised** (``sum_k w_k v_k / sum_k w_k``) rather than
GP-literal (un-normalised, which shrinks the whole aim toward zero). The
reason is D0, not aesthetics: an un-normalised aim is a uniform exposure cut
wearing a signal-weighting label, and this project has killed three findings
that turned out to be exposure statements (R-31, R-32, R-33). Normalising
isolates the *relative* re-weighting of the three anchors, which is the actual
claim. The un-normalised form is carried as ``normalize_aim=False`` and is
reported in the ablation so the choice is measured rather than asserted.

=====================================================================
CAUSALITY OF THE phi ESTIMATOR -- THE MOST DANGEROUS PART OF THIS FILE
=====================================================================

A decay rate computed over the whole series and applied to early rows is
exactly the full-series fit that ROUTINE.md's skeptic paragraph warns about,
and the truncation test does **not** catch it if the estimate is computed in
``prepare`` from the whole column. It is therefore computed as an
**expanding, strictly-lagged, Bayesian** flip-rate, per anchor:

1. ``v_k`` is the latched 0/1 vote of anchor ``k`` -- v4's exact loop.
2. ``valid_k[i]`` is True once anchor ``k``'s rolling mean exists at row i.
   Rows before that carry ``v_k = 0`` from ``fillna(0.0)`` and are *excluded*
   from the counts, otherwise a not-yet-defined anchor looks infinitely
   persistent.
3. ``flips_k[i] = cumsum(valid & valid.shift(1) & (v_k != v_k.shift(1)))``
   and ``n_k[i] = cumsum(valid_k)`` -- both cumulative sums, so row i is a
   function of rows <= i only.
4. ``p_k[i] = (flips_k + PRIOR_FLIPS) / (n_k + PRIOR_BARS)``, then
   ``.shift(1)`` so row i uses only rows **strictly before** i. Row 0 gets
   the prior.
5. ``phi_k[i] = -log(1 - 2 p_k[i])`` (approximately ``2 p_k``), which is the
   exact AR(1) decay rate of a two-state chain with per-bar flip hazard
   ``p``: its autocorrelation is ``rho(h) = (1 - 2p)^h``.

``PRIOR_FLIPS = 1.0`` and ``PRIOR_BARS = 30 * 288`` (one flip per thirty
days) are fixed constants, chosen a priori, **identical across the three
anchors**, and never swept. That last property is what makes them safe: a
prior that is the same for all three anchors cannot manufacture a difference
between them; it can only delay the point at which the data reveals one. In
the first months of a series the prior dominates, the three weights are
therefore equal, and the arm degenerates to v4's flat vote -- which is the
correct behaviour for "I have not yet observed how fast these signals decay".

Why this cannot see the future, stated plainly: every quantity entering
``phi_k[i]`` is a cumulative sum over rows ``< i`` of a per-row indicator that
itself depends only on rows ``<= i``. There is no full-series mean, std,
quantile, scaler, normalisation constant or fitted parameter anywhere in the
estimator. The ``causality`` subcommand proves this operationally two ways:
a **truncation probe** (prepare the full frame, prepare a truncated prefix of
it, require the ``target`` column to be bit-identical over the shared prefix)
and a **tamper probe** (multiply every post-cut bar by 3 and by 1/3 and
require identical pre-cut decisions).

=====================================================================
THE CATEGORY ERROR, NAMED BEFORE ANY NUMBER WAS READ
=====================================================================

This is failure mode 1 of the pre-registration and it is a theoretical
prediction against this branch, not an excuse written afterwards. GP's smooth
rate is derived for **quadratic** costs. ``MarketSpec`` here charges
``fee_rate * |traded notional|`` -- purely **proportional**, no impact term,
no queue, no spread. Under proportional costs the optimum is a no-trade
region, not a smooth rate, and a rule that trades a little every bar pays a
fee on each of 288 bars a day. That is the exact mechanism that killed
L-14 / L-15 / L-16 / L-18 (1,605 and 9,039 trades).

So the arm is required to be tested **at its strongest rather than as a
strawman**: the partial adjustment is paired with a **minimum-step filter**.
The GP path ``x_t`` is carried as a shadow variable that updates every bar;
the *executed* position ``pos`` moves to ``x_t`` only once ``|x_t - pos|``
exceeds ``min_step``. The rule therefore keeps GP's destination logic (the
shadow path is a genuine partial adjustment toward the persistence-weighted
aim) without paying 288 fees a day.

The honest reading of the resulting 2-D grid is pre-committed here: if the
only cells that survive fees are the ones where ``min_step`` dominates and
``a`` barely matters, then the smooth-rate object is doing nothing and the
no-trade region is doing everything, which is a genuine result about the
scope condition of GP's theory on proportional-cost data and will be reported
in exactly those terms.

Note ``min_step`` at ``a = 1`` **is** v4's deadband: the cell
``(a=1.0, min_step=0.10, phi_weighted=False, route="notional")`` reproduces
``kelly_regime_v4`` bit-for-bit, which the ``verify`` subcommand asserts. The
grid is thus a strict superset of the incumbent, and "does ``a`` matter" is
answerable *within* the grid rather than across files.

=====================================================================
TRAP 2 -- THE BROKER HAS ITS OWN DEADBAND
=====================================================================

``tradebot.broker.REBALANCE_DEADBAND = 0.05``: ``_execute_target`` silently
drops same-sign adjustments smaller than 5% of **max** notional (equity x
leverage). On spot that is 5% of equity; on 5x futures it is **25% of
equity**. v4 routes through ``ctx.order_notional`` -> ``order_target`` and so
inherits that band. Small smooth GP steps are exactly what it swallows, so a
naive implementation of this arm silently becomes a coarser strategy wearing
this file's label -- the failure ``experiments/matched_hold.py`` documents at
length for the constant-exposure arm, where routing through
``order_notional`` turned the rebalanced benchmark into the static one on
futures.

The fix is the same one: emit an explicit **quantity** (``ctx.buy`` /
``ctx.sell``) sized off equity and price, so the only deadband in force is
this strategy's own ``min_step``. ``route="qty"`` (the default) does that;
``route="notional"`` reproduces v4's routing and exists so the two can be
measured against each other. The ``futures`` subcommand reports, on spot AND
on 5x futures, how many *intended* steps (bars where the ``target`` column
moved) became *actual* fills under each route. The broker is not edited.

=====================================================================
WHAT THIS BRANCH READS
=====================================================================

BTC is truncated to ``<= 2022-12-31`` immediately on load, with an assertion,
so no bar dated 2023-01-01 or later can enter any computation in this file.
Inner-train and inner-validation come from ``r64_shared``. The falsification
instrument is ETH-A (``load_eth_a()``, Bitfinex 2016-03 -> 2019-12), which is
entirely pre-2020 and costs zero holdout consultations. **Holdout
consultations added by this branch: 0.**

Usage::

    python experiments/r64_novel_aim_partial_adjustment.py verify
    python experiments/r64_novel_aim_partial_adjustment.py persistence
    python experiments/r64_novel_aim_partial_adjustment.py causality
    python experiments/r64_novel_aim_partial_adjustment.py grid
    python experiments/r64_novel_aim_partial_adjustment.py ablation
    python experiments/r64_novel_aim_partial_adjustment.py futures
    python experiments/r64_novel_aim_partial_adjustment.py eth
    python experiments/r64_novel_aim_partial_adjustment.py all
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiments.r64_shared as shared  # noqa: E402
from experiments.matched_hold import mean_notional  # noqa: E402
from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Context  # noqa: E402

OUT_DIR = ROOT / "experiments" / "reports"

# Bayesian prior on the per-bar flip hazard of a latched anchor vote. Fixed a
# priori, identical for all three anchors, never swept. See the module
# docstring: a shared prior cannot manufacture a difference between anchors.
PRIOR_FLIPS = 1.0
PRIOR_BARS = 30.0 * BARS_PER_DAY  # "one flip per thirty days" as a starting belief


# ============================================================== the strategy


class GPAimPartialAdjustment(KellyRegimeV4):
    """kelly_regime_v4 with Garleanu-Pedersen partial adjustment toward a persistence-weighted aim.

    Subclasses ``KellyRegimeV4`` so the anchors (20/40/80), the 1% band, the
    latching hysteresis, the fractional-Kelly ``target_vol/realized_vol``
    scale, the v3 volatility-breakout state machine and every default
    constructor argument are the incumbent's own, unmodified. Only
    ``prepare``'s vote blend and position-update loop, and ``on_bar``'s order
    routing, differ -- and each difference is switchable so the halves can be
    ablated apart.

    Parameters
    ----------
    a
        GP trading rate: the fraction of the distance to the aim closed each
        bar by the shadow path. ``a=1.0`` is "jump to the aim", i.e. v4's own
        destination rule.
    min_step
        Minimum-step filter, in the same units as v4's ``deadband`` (fraction
        of equity notional). The executed position moves to the shadow path
        only when they differ by more than this. At ``a=1.0`` this *is* v4's
        deadband.
    phi_weighted
        False -> v4's flat equal-weighted vote. True -> the GP
        persistence-weighted aim vote.
    normalize_aim
        True -> weights are renormalised to sum to 1 (a pure relative
        re-weighting). False -> GP-literal un-normalised shrinkage, which is
        also an exposure cut.
    route
        ``"qty"`` -> explicit ``ctx.buy``/``ctx.sell`` quantities, bypassing
        the broker's 5%-of-max-notional rebalance deadband (Trap 2).
        ``"notional"`` -> v4's own ``ctx.order_notional`` routing.
    """

    name = "r64_novel_gp_aim_partial_adjustment"
    warmup = 80 * BARS_PER_DAY + 10  # identical to kelly_regime_v4

    def __init__(self, a: float = 1.0, min_step: float = 0.10,
                 phi_weighted: bool = True, normalize_aim: bool = True,
                 route: str = "qty", **kwargs) -> None:
        super().__init__(**kwargs)
        if not (0.0 < a <= 1.0):
            raise ValueError(f"a must be in (0, 1], got {a!r}")
        if min_step < 0.0:
            raise ValueError(f"min_step must be >= 0, got {min_step!r}")
        if route not in ("qty", "notional"):
            raise ValueError(f"route must be 'qty' or 'notional', got {route!r}")
        self.a = float(a)
        self.min_step = float(min_step)
        self.phi_weighted = bool(phi_weighted)
        self.normalize_aim = bool(normalize_aim)
        self.route = route

    # ------------------------------------------------------------- internals

    def _anchor_votes(self, df: pd.DataFrame):
        """v4's latched anchor votes, plus each anchor's validity mask.

        The vote loop is copied from ``KellyRegime.prepare`` unchanged; the
        only addition is ``valid``, which records where the rolling anchor
        actually exists so the flip-rate estimator can exclude the
        pre-warmup rows that ``fillna(0.0)`` would otherwise make look
        perfectly persistent.
        """
        close = df["close"]
        votes, valids = [], []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
            valids.append(anchor.notna())
        return votes, valids

    @staticmethod
    def _causal_phi(v: pd.Series, valid: pd.Series) -> np.ndarray:
        """Expanding, strictly-lagged, Bayesian per-bar decay rate of one vote.

        Row i is a function of rows < i only: cumulative counts (rows <= i)
        followed by ``.shift(1)``. No full-series statistic is used anywhere.
        """
        vv = v.to_numpy(dtype=float)
        ok = valid.to_numpy(dtype=bool)

        flip = np.zeros(len(vv), dtype=float)
        flip[1:] = ((vv[1:] != vv[:-1]) & ok[1:] & ok[:-1]).astype(float)

        cum_flips = np.cumsum(flip)
        cum_bars = np.cumsum(ok.astype(float))

        p = (cum_flips + PRIOR_FLIPS) / (cum_bars + PRIOR_BARS)
        # Strict lag: bar i may only use information from bars < i.
        p_lag = np.empty_like(p)
        p_lag[0] = PRIOR_FLIPS / PRIOR_BARS
        p_lag[1:] = p[:-1]

        # Exact AR(1) decay of a two-state chain with flip hazard p:
        # rho(h) = (1 - 2p)^h  =>  phi = -log(1 - 2p) ~= 2p.
        p_lag = np.clip(p_lag, 0.0, 0.499999)
        return -np.log1p(-2.0 * p_lag)

    def aim_components(self, df: pd.DataFrame) -> dict:
        """Everything the aim is built from, exposed for measurement.

        Returns the per-anchor votes, causal phi paths, GP weights and the
        resulting aim vote. Used by the ``persistence`` subcommand so the
        anchor decay rates can be reported as a measurement of the incumbent,
        independently of whether this arm wins anything.
        """
        votes, valids = self._anchor_votes(df)
        phis = [self._causal_phi(v, ok) for v, ok in zip(votes, valids)]
        flat = (sum(votes) / len(votes)).to_numpy(dtype=float)

        if self.phi_weighted:
            w = [1.0 / (1.0 + phi / self.a) for phi in phis]
            num = sum(wk * v.to_numpy(dtype=float) for wk, v in zip(w, votes))
            den = sum(w)
            aim = num / den if self.normalize_aim else num / len(votes)
        else:
            w = [np.ones(len(df)) for _ in votes]
            aim = flat

        return dict(votes=votes, valids=valids, phis=phis, weights=w,
                    flat=flat, aim=aim)

    # -------------------------------------------------------------- prepare

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        comp = self.aim_components(df)
        frac = comp["aim"]
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        close = df["close"]
        r = np.log(close).diff()

        # --- v3/v4's conditional volatility target, copied unchanged --------
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

        # --- the only changed lines: GP partial adjustment + min-step -------
        n = len(df)
        target = np.zeros(n)
        x = 0.0    # GP shadow path: closes fraction `a` of the gap every bar
        pos = 0.0  # executed position: follows x only past the min-step filter
        state = 0
        for i in range(n):
            q = ratio[i]
            if np.isfinite(q):
                if state == 0:
                    state = 1 if q > self.high_in else (-1 if q < self.low_in else 0)
                elif state == 1 and q < self.high_out:
                    state = 0
                elif state == -1 and q > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = frac[i] * scale
            # GP's literal form rather than `x += a*(desired - x)`: at a=1.0
            # this is exactly `desired`, so the a=1 cell reproduces v4
            # bit-for-bit instead of drifting by one ulp (verified by `verify`).
            x = (1.0 - self.a) * x + self.a * desired
            if abs(x - pos) > self.min_step:
                pos = x
            target[i] = pos

        df["target"] = target
        return df

    # --------------------------------------------------------------- on_bar

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) <= 1e-9:
            return  # the mechanism did not move; do not churn on equity drift

        if self.route == "notional":
            ctx.order_notional(t)  # v4's own routing, inherits the 5% broker band
            return

        # Trap 2: explicit quantity, so the only deadband in force is this
        # strategy's own min_step rather than the broker's 5%-of-max-notional
        # (25% of equity on 5x futures) rebalance band.
        lev = max(ctx.market.leverage, 1e-9)
        lo = -lev if ctx.can_short else 0.0
        t = min(lev, max(lo, t))
        equity, price = ctx.equity, ctx.close
        if not np.isfinite(equity) or equity <= 0.0 or price <= 0.0:
            return
        delta = t * equity / price - ctx.position
        if abs(delta) * price < 1e-9:
            return
        if delta > 0:
            ctx.buy(delta)
        else:
            ctx.sell(-delta)


# ================================================================== helpers


def load_btc_pre2023() -> pd.DataFrame:
    """BTC, hard-truncated so no bar dated 2023-01-01 or later can be read."""
    df = shared.load_btc().loc[:"2022-12-31"]
    assert df.index[-1] < pd.Timestamp("2023-01-01", tz=df.index.tz), df.index[-1]
    return df


def intended_steps(res) -> int:
    """Bars inside the measured period where the target column moved."""
    tgt = res.df["target"].to_numpy(dtype=float)
    if len(tgt) < 2:
        return 0
    return int(np.count_nonzero(np.abs(np.diff(tgt)) > 1e-9))


def cell(strategy, df, window, market, label: str, rows: list,
         echo: bool = True) -> dict:
    """One measured backtest, flattened into a reportable row."""
    res, m = shared.measure(strategy, df, window, market)
    row = dict(
        label=label, market=market.name, fee=market.fee_rate,
        window=f"{window[0]}:{window[1]}",
        final=m.final_balance, sharpe=m.sharpe, maxdd=m.max_drawdown_pct,
        episodes=m.num_trades, fills=len(res.fills), fees=m.fees_paid,
        mean_notional=mean_notional(res),
        intended=intended_steps(res),
        time_in_market=m.time_in_market_pct,
    )
    row["exec_rate"] = row["fills"] / row["intended"] if row["intended"] else float("nan")
    rows.append(row)
    if echo:
        # Printed as each cell lands rather than per block: these sweeps run
        # for tens of minutes and a silent log is indistinguishable from a
        # hung one.
        print(_line(row), flush=True)
    return row


def _hdr(text: str) -> None:
    print("\n" + "=" * 108)
    print(text)
    print("=" * 108, flush=True)


GRID_HEAD = (f"{'label':44s} {'final':>11s} {'sharpe':>7s} {'maxDD%':>7s} "
             f"{'episod':>7s} {'fills':>8s} {'fees':>11s} {'meanC':>6s} "
             f"{'intend':>7s} {'exec%':>6s}")


def _line(r: dict) -> str:
    ex = "" if not np.isfinite(r["exec_rate"]) else f"{100 * r['exec_rate']:6.1f}"
    return (f"{r['label']:44s} {r['final']:11,.0f} {r['sharpe']:7.2f} "
            f"{r['maxdd']:7.1f} {r['episodes']:7d} {r['fills']:8d} "
            f"{r['fees']:11,.0f} {r['mean_notional']:6.3f} {r['intended']:7d} {ex:>6s}")


def show(rows: list) -> None:
    """Re-print a finished block as a contiguous table (cells also echo live)."""
    print("-" * 108)
    print(GRID_HEAD)
    print("-" * 108)
    for r in rows:
        print(_line(r))


# ============================================================ 0. verification


def cmd_verify(df=None) -> bool:
    """The grid is a strict superset of the incumbent: prove the cell.

    ``a=1.0, min_step=0.10, phi_weighted=False, route='notional'`` must
    reproduce ``kelly_regime_v4``'s ``target`` column bit-for-bit. If it does
    not, this file's re-implementation of v3's volatility machinery is
    unfaithful and every comparison below is against a strawman.
    """
    df = load_btc_pre2023() if df is None else df
    _hdr("VERIFY -- (a=1, min_step=0.10, flat vote, notional route) == kelly_regime_v4")
    sub = df.iloc[-250_000:].copy()
    t_v4 = shared.v4().prepare(sub.copy())["target"].to_numpy(dtype=float)
    t_me = GPAimPartialAdjustment(a=1.0, min_step=0.10, phi_weighted=False,
                                  route="notional").prepare(sub.copy())["target"].to_numpy(float)
    ok = np.array_equal(t_v4, t_me)
    print(f"  identical target columns over {len(sub):,} bars: {'PASS' if ok else 'FAIL'}")
    if not ok:
        bad = int(np.argmax(t_v4 != t_me))
        print(f"  first difference at row {bad}: v4={t_v4[bad]!r} arm={t_me[bad]!r}")
    return ok


# ============================================================ 1. persistence


def cmd_persistence(df=None) -> pd.DataFrame:
    """Item 7: the measured decay rates of v4's own 20/40/80-day anchor votes.

    A measurement of the incumbent, worth having whether or not this arm
    wins. Reported at the end of inner-train (so it is the number the
    inner-validation runs actually start from) and at the end of
    inner-validation.
    """
    df = load_btc_pre2023() if df is None else df
    _hdr("ANCHOR PERSISTENCE -- causal expanding flip rates of v4's latched votes (BTC)")

    arm = GPAimPartialAdjustment(phi_weighted=True)
    comp = arm.aim_components(df.copy())
    idx = df.index

    marks = [("end inner-train (2020-12-31)", int(idx.searchsorted(pd.Timestamp("2021-01-01", tz=idx.tz))) - 1),
             ("end inner-val   (2022-12-31)", len(df) - 1)]

    rows = []
    for mark_label, i in marks:
        print(f"\n  as of {mark_label}, row {i:,}")
        print(f"    {'anchor':>8s} {'flips':>7s} {'valid bars':>12s} "
              f"{'p/bar':>10s} {'flips/yr':>9s} {'phi/bar':>10s} "
              f"{'half-life(d)':>13s}")
        phis_here = []
        for k, days in enumerate(arm.horizons):
            v = comp["votes"][k].to_numpy(dtype=float)
            ok = comp["valids"][k].to_numpy(dtype=bool)
            flips = int(np.count_nonzero((v[1:i + 1] != v[:i]) & ok[1:i + 1] & ok[:i]))
            nbars = int(np.count_nonzero(ok[:i + 1]))
            phi = float(comp["phis"][k][i])
            phis_here.append(phi)
            p = (flips + PRIOR_FLIPS) / (nbars + PRIOR_BARS)
            hl = float(np.log(2.0) / phi / BARS_PER_DAY) if phi > 0 else float("inf")
            fpy = flips / (nbars / (365.25 * BARS_PER_DAY)) if nbars else float("nan")
            print(f"    {days:6d}d {flips:7d} {nbars:12,d} {p:10.2e} "
                  f"{fpy:9.2f} {phi:10.2e} {hl:13.1f}")
            rows.append(dict(mark=mark_label, anchor_days=days, flips=flips,
                             valid_bars=nbars, p_per_bar=p, flips_per_year=fpy,
                             phi_per_bar=phi, half_life_days=hl))

        print(f"\n    implied GP aim weights  w_k = 1/(1 + phi_k/a),  normalised:")
        print(f"    {'a':>8s} " + " ".join(f"{d:>10d}d" for d in arm.horizons))
        for a in (1.0, 0.5, 0.2, 0.05, 0.01, 0.002, 0.0005):
            w = np.array([1.0 / (1.0 + phi / a) for phi in phis_here])
            wn = w / w.sum()
            print(f"    {a:8.4f} " + " ".join(f"{x:11.4f}" for x in wn))
            for k, days in enumerate(arm.horizons):
                rows.append(dict(mark=mark_label, anchor_days=days, a=a,
                                 weight_norm=float(wn[k])))
    return pd.DataFrame(rows)


# ============================================================== 2. causality


def cmd_causality(df=None) -> bool:
    """Truncation probe + tamper probe against the phi estimator."""
    df = load_btc_pre2023() if df is None else df
    _hdr("CAUSALITY -- truncation probe and tamper probe on the causal phi estimator")

    all_ok = True
    variants = [
        ("phi-weighted, a=0.02, min_step=0.05",
         dict(a=0.02, min_step=0.05, phi_weighted=True)),
        ("phi-weighted, a=1.0,  min_step=0.10",
         dict(a=1.0, min_step=0.10, phi_weighted=True)),
        ("flat vote,    a=0.02, min_step=0.05",
         dict(a=0.02, min_step=0.05, phi_weighted=False)),
    ]

    # --- truncation: the shared prefix must be bit-identical ---------------
    full = df.iloc[-300_000:].copy()
    for cut in (120_000, 200_000, 260_000):
        trunc = full.iloc[:cut].copy()
        for lbl, kw in variants:
            t_full = GPAimPartialAdjustment(**kw).prepare(full.copy())["target"].to_numpy(float)
            t_cut = GPAimPartialAdjustment(**kw).prepare(trunc.copy())["target"].to_numpy(float)
            ok = np.array_equal(t_full[:cut], t_cut)
            all_ok &= ok
            print(f"  truncation @{cut:>7,}  {lbl:38s} bit-identical prefix: "
                  f"{'PASS' if ok else 'FAIL'}")

    # --- tamper: opposite post-cut futures, identical pre-cut decisions ----
    tail = df.iloc[-120_000:].copy()
    cut = len(tail) - 10_000
    probe_bars = [cut - k for k in (1, 2, 3, 5, 10, 50, 200, 1_000)]
    up, down = tail.copy(), tail.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0
    market = MarketSpec.futures(leverage=5.0)

    for lbl, kw in variants:
        def decisions(frame):
            s = GPAimPartialAdjustment(**kw)
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=market, start_balance=10_000.0)
            out = []
            for i in probe_bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        ok = decisions(up) == decisions(down)
        all_ok &= ok
        print(f"  tamper probe          {lbl:38s} decisions identical:  "
              f"{'PASS' if ok else 'FAIL'}")

    print(f"\n  CAUSALITY: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ==================================================================== 3. grid

A_GRID = (1.0, 0.5, 0.2, 0.05, 0.01, 0.002)
STEP_GRID = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20)


def _grid_rows(df, window, market, rows, phi_weighted: bool, tag: str) -> None:
    for a in A_GRID:
        for s in STEP_GRID:
            lbl = f"{tag} a={a:<6g} step={s:<5g}"
            cell(GPAimPartialAdjustment(a=a, min_step=s, phi_weighted=phi_weighted),
                 df, window, market, lbl, rows)


def cmd_grid(df=None) -> pd.DataFrame:
    """The 2-D sweep: trading rate `a` x minimum-step threshold.

    Run for BOTH vote modes so the whole grid doubles as the ablation of
    half (ii), and on inner-train first, inner-validation second. The whole
    grid is reported, not the winner.
    """
    df = load_btc_pre2023() if df is None else df
    rows: list[dict] = []

    # Order matters only for how soon a partial log is readable: the two
    # 0.10% blocks first, then the 0.40% stress tier.
    blocks = [("INNER-TRAIN", shared.INNER_TRAIN, shared.FEE_BASE),
              ("INNER-VAL", shared.INNER_VAL, shared.FEE_BASE),
              ("INNER-VAL", shared.INNER_VAL, shared.FEE_STRESS),
              ("INNER-TRAIN", shared.INNER_TRAIN, shared.FEE_STRESS)]
    for wlabel, window, fee in blocks:
        market = shared.spot(fee)
        _hdr(f"GRID -- {wlabel} {window}  spot @{fee:.2%}")
        print(GRID_HEAD, flush=True)
        here: list[dict] = []
        cell(shared.v4(), df, window, market, "kelly_regime_v4 (incumbent)", here)
        cell(GPAimPartialAdjustment(a=1.0, min_step=0.10, phi_weighted=False,
                                    route="qty"),
             df, window, market, "route-control a=1 step=0.10 flat qty", here)
        _grid_rows(df, window, market, here, False, "flat")
        _grid_rows(df, window, market, here, True, "phiw")
        show(here)
        for r in here:
            r["window_name"] = wlabel
        rows.extend(here)

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "r64_novel_grid.csv", index=False)
    print(f"\n  written: {OUT_DIR / 'r64_novel_grid.csv'}")
    return out


# ================================================================ 4. ablation


def ablation_specs(a: float, step: float) -> list[tuple[str, GPAimPartialAdjustment | None]]:
    return [
        ("v4 incumbent (baseline)", None),
        ("route-control: v4 rule, qty routing",
         GPAimPartialAdjustment(a=1.0, min_step=0.10, phi_weighted=False, route="qty")),
        (f"(a) partial adj only, flat vote a={a:g} s={step:g}",
         GPAimPartialAdjustment(a=a, min_step=step, phi_weighted=False)),
        ("(b) phi-weighted aim only, v4 update rule",
         GPAimPartialAdjustment(a=1.0, min_step=0.10, phi_weighted=True)),
        ("(b') phi-weighted aim, un-normalised",
         GPAimPartialAdjustment(a=1.0, min_step=0.10, phi_weighted=True,
                                normalize_aim=False)),
        (f"(c) both a={a:g} s={step:g}",
         GPAimPartialAdjustment(a=a, min_step=step, phi_weighted=True)),
    ]


def cmd_ablation(df=None, a: float = 0.02, step: float = 0.05) -> pd.DataFrame:
    """(a) rate only, (b) aim only, (c) both -- against v4 and the route control."""
    df = load_btc_pre2023() if df is None else df
    rows: list[dict] = []
    for wlabel, window in (("INNER-TRAIN", shared.INNER_TRAIN),
                           ("INNER-VAL", shared.INNER_VAL)):
        for fee in (shared.FEE_BASE, shared.FEE_STRESS):
            market = shared.spot(fee)
            _hdr(f"ABLATION -- {wlabel} {window}  spot @{fee:.2%}")
            print(GRID_HEAD, flush=True)
            here: list[dict] = []
            for lbl, strat in ablation_specs(a, step):
                cell(shared.v4() if strat is None else strat, df, window, market,
                     lbl, here)
            show(here)
            for r in here:
                r["window_name"] = wlabel
            rows.extend(here)
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "r64_novel_ablation.csv", index=False)
    print(f"\n  written: {OUT_DIR / 'r64_novel_ablation.csv'}")
    return out


# ================================================= 5. futures and Trap 2


def cmd_futures(df=None, a: float = 0.02, step: float = 0.05) -> pd.DataFrame:
    """Futures 5x plus the Trap 2 measurement, on spot AND futures.

    ``intended`` counts bars where the strategy's own ``target`` column moved
    inside the measured period; ``fills`` counts executions that actually
    happened. Under ``route="notional"`` the broker's 5%-of-max-notional
    rebalance band (25% of equity at 5x) eats the difference.
    """
    df = load_btc_pre2023() if df is None else df
    rows: list[dict] = []
    combos = [("spot @0.10%", shared.spot(shared.FEE_BASE)),
              ("futures 5x", shared.futures())]
    for mlabel, market in combos:
        for wlabel, window in (("INNER-TRAIN", shared.INNER_TRAIN),
                               ("INNER-VAL", shared.INNER_VAL)):
            _hdr(f"TRAP 2 / FUTURES -- {wlabel} {window}  {mlabel} "
                 f"(leverage {market.leverage:g}x, broker band = "
                 f"{0.05 * market.leverage:.0%} of equity)")
            print(GRID_HEAD, flush=True)
            here: list[dict] = []
            cell(shared.v4(), df, window, market, "kelly_regime_v4 (notional route)", here)
            for route in ("notional", "qty"):
                for lbl, kw in (("flat  a=1    s=0.10", dict(a=1.0, min_step=0.10, phi_weighted=False)),
                                (f"phiw  a={a:g} s={step:g}", dict(a=a, min_step=step, phi_weighted=True)),
                                (f"flat  a={a:g} s={step:g}", dict(a=a, min_step=step, phi_weighted=False)),
                                ("phiw  a=0.002 s=0.0 (pure GP)", dict(a=0.002, min_step=0.0, phi_weighted=True))):
                    cell(GPAimPartialAdjustment(route=route, **kw), df, window,
                         market, f"{lbl} [{route}]", here)
            show(here)
            for r in here:
                r["window_name"] = wlabel
            rows.extend(here)
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "r64_novel_trap2_futures.csv", index=False)
    print(f"\n  written: {OUT_DIR / 'r64_novel_trap2_futures.csv'}")
    return out


# ======================================================= 6. ETH-A (D3 sign)


def cmd_eth(a: float = 0.02, step: float = 0.05) -> pd.DataFrame:
    """D3 falsification on ETH-A: Bitfinex 2016-03 -> 2019-12, +0 holdout."""
    eth = shared.load_eth_a()
    assert eth.index[-1] < pd.Timestamp("2020-01-01", tz=eth.index.tz), eth.index[-1]
    _hdr(f"D3 FALSIFICATION -- ETH-A Bitfinex {eth.index[0].date()} -> "
         f"{eth.index[-1].date()} ({len(eth):,} bars, +0 holdout consultations)")
    rows: list[dict] = []
    for fee in (shared.FEE_BASE, shared.FEE_STRESS):
        market = shared.spot(fee)
        print(f"\n  spot @{fee:.2%}")
        print(GRID_HEAD, flush=True)
        here: list[dict] = []
        cell(shared.v4(), eth, (None, None), market, "kelly_regime_v4", here)
        for lbl, kw in ((f"(a) flat a={a:g} s={step:g}", dict(a=a, min_step=step, phi_weighted=False)),
                        ("(b) phiw a=1 s=0.10", dict(a=1.0, min_step=0.10, phi_weighted=True)),
                        (f"(c) phiw a={a:g} s={step:g}", dict(a=a, min_step=step, phi_weighted=True))):
            cell(GPAimPartialAdjustment(**kw), eth, (None, None), market, lbl, here)
        show(here)
        base = here[0]["final"]
        for r in here[1:]:
            r["d_log_vs_v4"] = float(np.log(r["final"] / base))
            print(f"    D3 sign  {r['label']:44s} "
                  f"dlog(arm - v4) = {r['d_log_vs_v4']:+.4f}  "
                  f"{'POSITIVE' if r['d_log_vs_v4'] > 0 else 'NEGATIVE (refutes)'}")
        rows.extend(here)
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "r64_novel_eth_a.csv", index=False)
    print(f"\n  written: {OUT_DIR / 'r64_novel_eth_a.csv'}")
    return out


# =========================================== 7. finalists: paired bootstrap


def cmd_finalists(df=None, a: float = 0.02, step: float = 0.05) -> pd.DataFrame:
    """Paired block-bootstrap vs v4 on inner-validation, both fee tiers.

    Uses ``r64_shared.compare`` unchanged, so D0 / D2 / D4 are evaluated by
    the frozen shared code rather than reimplemented here. Inner-validation
    only: the holdout is not read by this branch.
    """
    df = load_btc_pre2023() if df is None else df
    _hdr("FINALISTS -- paired block bootstrap vs v4, INNER-VALIDATION only")
    rows: list[dict] = []
    arms = [
        (f"(a) flat a={a:g} s={step:g}", dict(a=a, min_step=step, phi_weighted=False)),
        ("(b) phiw a=1 s=0.10", dict(a=1.0, min_step=0.10, phi_weighted=True)),
        (f"(c) phiw a={a:g} s={step:g}", dict(a=a, min_step=step, phi_weighted=True)),
    ]
    for lbl, kw in arms:
        d = {}
        for fee in (shared.FEE_BASE, shared.FEE_STRESS):
            r = shared.compare(GPAimPartialAdjustment(**kw), df, shared.INNER_VAL,
                               shared.spot(fee), label=lbl)
            print("  " + shared.fmt(r))
            d[fee] = r
            rows.append(r)
        ok = shared.d2_satisfied(d[shared.FEE_BASE]["d_logret"],
                                 d[shared.FEE_STRESS]["d_logret"])
        print(f"    D2 (advantage grows with fee): {d[shared.FEE_BASE]['d_logret']:+.4f} "
              f"-> {d[shared.FEE_STRESS]['d_logret']:+.4f}  "
              f"{'SATISFIED' if ok else 'FAILS'}\n")
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "r64_novel_finalists.csv", index=False)
    print(f"  written: {OUT_DIR / 'r64_novel_finalists.csv'}")
    return out


# ====================================================================== main


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    step = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

    if cmd == "verify":
        cmd_verify()
    elif cmd == "persistence":
        cmd_persistence()
    elif cmd == "causality":
        cmd_causality()
    elif cmd == "grid":
        cmd_grid()
    elif cmd == "ablation":
        cmd_ablation(a=a, step=step)
    elif cmd == "futures":
        cmd_futures(a=a, step=step)
    elif cmd == "eth":
        cmd_eth(a=a, step=step)
    elif cmd == "finalists":
        cmd_finalists(a=a, step=step)
    elif cmd == "all":
        df = load_btc_pre2023()
        ok_v = cmd_verify(df)
        cmd_persistence(df)
        ok_c = cmd_causality(df)
        cmd_grid(df)
        cmd_ablation(df, a=a, step=step)
        cmd_futures(df, a=a, step=step)
        cmd_finalists(df, a=a, step=step)
        cmd_eth(a=a, step=step)
        _hdr("BRANCH SUMMARY")
        print(f"  verify (grid contains v4 exactly): {'PASS' if ok_v else 'FAIL'}")
        print(f"  causality (truncation + tamper):   {'PASS' if ok_c else 'FAIL'}")
        print(f"  configurations evaluated (this branch): "
              f"{shared.configs_evaluated()}")
        print("  holdout consultations added by this branch: 0 "
              "(BTC hard-truncated at 2022-12-31; ETH-A ends 2019-12)")
        return
    else:
        raise SystemExit(f"unknown command {cmd!r}")

    print(f"\nconfigurations evaluated (this branch): {shared.configs_evaluated()}")


if __name__ == "__main__":
    main()
