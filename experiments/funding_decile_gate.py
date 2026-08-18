"""Funding-decile gate on kelly_regime_v4 (backlog B-05, "funding as a gate").

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea
--------
R-14 measured that real Binance BTCUSDT perpetual funding is positive at
86.5% of settlements, costs a constant long ~15%/yr, and — worse — runs
at roughly +20%/yr *while ``kelly_regime_v4`` holds* versus +2.8%/yr flat,
because the crowd-regime vote that puts the strategy long is exactly the
condition that pushes funding up. R-16 found the trailing-funding-level
signal itself has forward-return content (Q1-Q5 spread +3.57pp over 14
days), but with a non-monotone middle that is a warning about how much of
that is noise.

This experiment is the *conservative* use of R-16: not a standalone
reversal signal (the higher-turnover route R-16/backlog explicitly warns
is "where strategies go to die", citing R-12's 28-of-32 in-sample /
0-of-28 out-of-sample result), but a low-turnover hard cutout layered on
top of the unmodified incumbent mechanism. When trailing funding is in
its own top decile — the crowded-long tail R-14 shows is the expensive
one — the gate forces flat for that bar; every other bar is
``kelly_regime_v4`` unchanged.

Mechanism
---------
``FundingDecileGate`` subclasses ``KellyRegimeV4`` unchanged. ``prepare()``
calls the base class first (so the anchor vote / vol targeting / latching
logic is untouched), then adds a causal, ROLLING (not expanding, not
whole-series) percentile-rank of the trailing funding series, and
overrides ``target`` to 0 wherever that rank exceeds ``decile``.
``on_bar`` is inherited from ``KellyRegime`` unmodified: it just reads
``ctx.bar["target"]`` and calls ``ctx.order_notional``.

Causal funding merge
---------------------
Funding settles at discrete instants (8-hourly). To avoid any
same-instant ambiguity between "settled at this bar" and "settled after
this bar", the raw settlement series is shifted by one settlement
*before* forward-filling onto the bar grid: a bar can only ever see a
rate that settled strictly before its own timestamp, with one full
settlement of margin. Where funding does not cover a bar at all (before
2020-01-01, after 2023-12-31, or ``funding=None``), the causal series is
NaN there — and NaN compares False against the decile threshold, so the
gate does not fire and the bar passes ``kelly_regime_v4``'s target
through unmodified. It is never treated as "not crowded, go flat" and
never silently forced flat either.

The percentile-rank itself is a ROLLING rank (``Series.rolling(window,
min_periods=window).rank(pct=True)``) over the trailing ``lookback_days``
of the causal series, so bar ``i``'s percentile is relative only to its
own trailing history, never to the full series. A full-series quantile
computed once and applied to every row is exactly the lookahead class
``test_causality_strict.py`` is built to catch (see the comment in
``run_eprocess.py``'s ``causality()``) — it would let an early-2020 bar's
gate depend on a 2023 funding spike it could not yet have observed.
``min_periods=window`` (not a fraction of it) means the gate cannot fire
at all until a bar has a full trailing window of *observed* funding
history behind it: with the default 180-day lookback and funding starting
exactly on 2020-01-01, that is the first ~180 days of 2020 with the gate
inactive (pass-through) even though funding technically covers those
bars — a real ramp-up effect, not a bug, and it is reported as such in
``run_funding_decile_gate.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4


def causal_funding_percentile(
    index: pd.DatetimeIndex,
    funding: pd.Series | None,
    lookback_days: int,
) -> pd.Series:
    """Rolling, causal percentile-rank of funding, aligned to ``index``.

    Returns a float Series aligned to ``index``: NaN wherever there is not
    a full trailing window of *causally observable* funding history (no
    funding data at all, before funding coverage starts, or still inside
    the first ``lookback_days`` of coverage). NaN is the "not crowded /
    unknown" value; callers must treat it as pass-through, not as "flat".

    Causal by construction: the shift-by-one-settlement merge means bar
    ``i`` sees only settlements strictly before its own timestamp, and the
    rolling (not expanding) window means bar ``i``'s rank depends only on
    bars ``<= i`` within the trailing ``lookback_days`` — never on rows
    after it and never on the whole series. Both properties hold
    regardless of how long ``index`` is, so truncating the frame to any
    prefix cannot change the value at any bar inside that prefix (verified
    by ``run_funding_decile_gate.py``'s ``causality()``).
    """
    if funding is None or len(funding) == 0:
        return pd.Series(np.nan, index=index, dtype=float)

    # Shift by one settlement before forward-filling: bar i must never see
    # a rate that settles at-or-after its own timestamp. Without the
    # shift, a bar whose timestamp exactly equals a settlement time would
    # be ambiguous about whether that settlement has "already happened".
    shifted = funding.sort_index().shift(1)
    per_bar = shifted.reindex(index, method="ffill")

    window = int(round(lookback_days * BARS_PER_DAY))
    window = max(window, 1)
    return per_bar.rolling(window, min_periods=window).rank(pct=True)


class FundingDecileGate(KellyRegimeV4):
    """kelly_regime_v4, forced flat whenever trailing funding hits its own top decile.

    Conservative gate for backlog B-05: the incumbent mechanism (anchor
    vote, fractional-Kelly vol targeting, latching, deadband) is entirely
    unmodified. The only change is an override: whenever the causal,
    rolling percentile-rank of the trailing funding series exceeds
    ``decile``, that bar's target is forced to 0; every other bar keeps
    ``kelly_regime_v4``'s own target. With ``funding=None`` this is
    byte-identical to plain ``KellyRegimeV4`` (the gate simply never
    fires), so it is safe to run outside the funding-covered window
    (2020-01-01 .. 2023-12-31).

    See the module docstring for the funding-merge causality argument and
    ``run_funding_decile_gate.py`` for the sweep, selection, and the
    by-hand causality self-check (this class is intentionally NOT
    ``@register``-ed; ``test_causality_strict.py`` only parametrizes over
    registered strategies).
    """

    name = "funding_decile_gate"

    def __init__(
        self,
        funding: pd.Series | None = None,
        lookback_days: int = 180,
        decile: float = 0.90,
        horizons: tuple[int, ...] = (20, 40, 80),
        **kwargs,
    ) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.funding = funding
        self.lookback_days = lookback_days
        self.decile = decile

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # kelly_regime_v4's target column, untouched

        pct = causal_funding_percentile(df.index, self.funding, self.lookback_days)
        # NaN > decile is False in pandas/numpy, so "no observable funding
        # history yet" already reads as "not crowded" (pass-through) with
        # no extra handling needed - but spell it out for the reader (and
        # so a future refactor that swaps the comparison op cannot
        # silently flip this).
        crowded = (pct > self.decile).to_numpy(dtype=bool, copy=True)
        crowded &= pct.notna().to_numpy()

        target = df["target"].to_numpy(dtype=float).copy()
        target[crowded] = 0.0

        df["target"] = target
        df["funding_pct_rank"] = pct.to_numpy()
        df["funding_crowded"] = crowded
        return df

    # on_bar is inherited unchanged from KellyRegime: it reads
    # ctx.bar["target"] and calls ctx.order_notional(t) only when the
    # target moved from the previous bar's target.
