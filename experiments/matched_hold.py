"""Matched-risk benchmark: a **de-levered** buy-and-hold (backlog B-13).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

The question, and why it is pointed at this project's own headline
-------------------------------------------------------------------
R-31 retired R-28's risk finding with one argument: the e-process gate
was compared against an arm carrying **2.4x** its volatility, so what
looked like a property of the *gate* was a property of the *exposure
level*. Hold risk fixed and the finding dissolved.

That argument applies, unchanged, to the claim this project has been
leaning on since L-04. "Regime-gated sizing cuts drawdown" is measured
by comparing `kelly_regime_v4` — which holds roughly half the notional,
and is flat a third of the time — against a **fully-invested**
`buy_and_hold`. R-29's −41.1pp [−54.8, −18.4] and R-17's ETH replication
are both measured that way. Nobody has yet asked what a `buy_and_hold`
de-levered to v4's own realized volatility does to that gap.

This file supplies the missing arm: a passive long that holds a constant
fraction ``c`` of equity in the asset and makes no other decision. It has
no regime gate, no volatility estimate, no anchors — it cannot time
anything. If v4's drawdown advantage survives against *that*, the
advantage is the gating. If it does not, the project's one robust finding
is the same arithmetic R-28 fell for.

Two arms, because "de-levered hold" is ambiguous and the difference matters
--------------------------------------------------------------------------
``ConstantExposureHold(c, static=False)`` — **rebalanced.** Holds ``c`` x
    equity in notional, rebalancing back to it when the realized fraction
    drifts more than ``deadband`` (relative) away. This is the constant-*risk*
    reading of "de-levered hold": its volatility is ``c`` x the asset's,
    always. It pays fees for the privilege, and it is the arm the backlog
    row asks for.

``ConstantExposureHold(c, static=True)`` — **static.** Buys ``c`` x equity
    once and never trades again; the rest sits in cash. Zero turnover,
    zero further fees — but its weight *drifts*, up toward 1.0 in a bull
    and down in a bear, so it is not a constant-risk arm. This is what an
    actual lazy investor does, which is why it is worth carrying: it is
    the cheapest possible benchmark and it is not obviously the weaker one.

Both are deliberately stateless — the decision is a function of the
current bar and the account, never of a counter — so re-using one
instance across backtests, windows or markets cannot leak anything from
one run into the next.

Matching, and the axis problem that comes with it
-------------------------------------------------
Two matching axes are solved, because they answer subtly different
questions and this project has been sloppy about which one it means:

**equal realized volatility** (the R-31 convention) — solve ``c`` so the
    hold's realized annualized volatility equals v4's over the same span.

**equal mean notional** — set ``c`` to v4's own mean notional fraction.
    This is the literal form of the "it just holds less" critique, and it
    needs no solver.

The two disagree, and they must: v4 *targets* constant volatility, so its
realized volatility is roughly regime-independent while a constant-exposure
hold's tracks the market's. The exposure that equalizes risk is therefore
a property of the period it was solved on — the same instability R-31 and
R-32 both hit from the other side. That is a pre-registered failure mode
here, not a surprise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategy import Context, Strategy

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


class ConstantExposureHold(Strategy):
    """Buy-and-hold de-levered to a constant fraction ``c`` of equity."""

    name = "constant_hold"
    # A passive long needs no history. Kept at zero deliberately: run_period
    # takes each strategy's warmup from the bars *before* the measured
    # period, so v4 enters warm and this arm enters with nothing to warm,
    # and both start trading on the period's first bar (R-22).
    warmup = 0

    def __init__(self, c: float = 0.5, deadband: float = 0.10,
                 static: bool = False) -> None:
        if not np.isfinite(c) or c <= 0.0:
            raise ValueError(f"c must be positive and finite, got {c!r}")
        if deadband < 0.0:
            raise ValueError(f"deadband must be non-negative, got {deadband!r}")
        self.c = float(c)
        self.deadband = float(deadband)
        self.static = bool(static)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # A constant "target" column so the same clamp diagnostic that reads
        # v4's target reads this arm's too. For the static arm this is the
        # entry weight, not the running one - the running one drifts, which
        # is the whole difference between the two arms.
        df["target"] = self.c
        return df

    def on_bar(self, ctx: Context) -> None:
        if self.static:
            # Keyed on being flat, like buy_and_hold: a window that opens the
            # account partway in still gets its entry, and once filled this
            # never fires again.
            if not ctx.in_market:
                ctx.order_notional(self.c)
            return

        equity = ctx.equity
        if not np.isfinite(equity) or equity <= 0.0:
            return
        price = ctx.close
        current = abs(ctx.position) * price / equity
        # Relative band, so the turnover policy is comparable across
        # exposures instead of making a small-c arm rebalance on every tick.
        if abs(current - self.c) <= self.deadband * self.c:
            return

        # Explicit quantity rather than order_notional, and the reason is
        # not cosmetic. The broker ignores same-sign target adjustments
        # below 5% of MAX notional (equity x leverage) so that strategies
        # may re-emit a target every bar without churning fees. On 5x
        # futures that band is 25% of equity, which is wider than anything
        # a constant-exposure arm ever asks for: routed through
        # order_notional this arm never rebalances on futures at all and
        # silently becomes the static arm - a different benchmark wearing
        # this one's label. A quantity order carries its own deadband (the
        # one above) instead of inheriting a leverage-scaled one.
        desired = self.c * equity / price
        delta = desired - abs(ctx.position)
        if delta > 0:
            ctx.buy(delta)
        else:
            ctx.sell(-delta)


def mean_notional(result) -> float:
    """v4's mean notional fraction over a run, as the market actually allows.

    Read off the ``target`` column clipped at the market's leverage cap,
    because a target of 1.4 on spot is a request for 1.0. This is the
    quantity the "it just holds less" critique is about, so it is measured
    rather than assumed.
    """
    if "target" not in result.df:
        return float("nan")
    tgt = np.abs(result.df["target"].to_numpy(dtype=float))
    return float(np.mean(np.clip(tgt, 0.0, result.market.leverage)))
