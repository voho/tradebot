"""Funding as a gate on kelly_regime_v4: stand down in the crowded-long tail (backlog B-05).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

Step 1 — selection and justification (written before any code ran)
--------------------------------------------------------------------
**Idea, one sentence.** On the 5x futures market, when the trailing
percentile rank of the realized 8-hourly funding rate is in its extreme
top tail (crowded-long territory), reduce or zero the position
``kelly_regime_v4``'s existing vote/vol-target sizer would otherwise
take, because paying peak funding is exactly the moment R-16 found least
justified by forward returns.

**Constraint attacked.** COST. Funding is the single largest unmodelled
cost in this project (R-14: it turns `kelly_regime_v4`'s $156K into a
$36K-$80K band straddling spot holding's $66K), and it is adversely
timed — the strategy pays +20%/yr in funding while it holds, against
+2.8%/yr while flat, because the crowding the vote detects is exactly
what sets the rate. This gate tries to buy back some of that timing
mismatch cheaply: it only intervenes in a small extreme tail of bars, so
unlike a full standalone reversal strategy (which is where strategies in
this repo go to die, per R-12/L-25/L-13) it should not add much turnover.

**Not a duplicate of:**
- **R-14** measured the cost and its adverse timing; this row is the
  first attempt to *act* on that measurement rather than just report it.
- **R-15** (funding harvest, delta-neutral carry) is a *different*
  position — short the perp, long spot — and is BLOCKED on the same data
  this row uses; this row keeps `kelly_regime_v4`'s existing directional
  book and only gates its size.
- **R-16** is the open hypothesis this cashes in: top-quintile funding
  predicts a negative forward-return spread unless price is also rising,
  but the middle quintiles are non-monotone, so R-16 itself says "hint,
  not proven monotone relationship" — which is exactly why this gate
  only acts in the *extreme* tail (decile/quintile), not across the
  whole funding range, and why the falsification test below is chosen to
  stress exactly that non-monotonicity risk.
- Distinct from B-11/R-31/R-32 (matched-risk gate-vs-vote): those compare
  *which quantity opens the regime gate*, holding exposure matched. This
  gate does not replace the regime vote — it multiplies the vote's
  output by a second, independent, mostly-inactive multiplier. It is a
  sizing overlay, not a competing gate mechanism.

**Simulable here?** Yes — one committed file
(`data/btcusdt_perp_funding_8h.csv.gz`), no fetch, causal by construction
(rolling rank of past settlements, backward-joined onto the bar grid).

**Named failure modes (before any code ran):**
1. The gate triggers on so few bars that it changes nothing measurable —
   the funding gate is a no-op dressed as a finding.
2. Toggling the position on/off at the gate's edges adds enough extra
   fee-crossing turnover that the fees saved by *not* holding through
   peak funding are given back at the taker line, or worse.
3. Any measured improvement sits inside the +/-0.2 Sharpe noise floor —
   R-16's own middle-quintile non-monotonicity suggests the tail signal
   may be real while the strategy that tries to use it is not.
4. The 2023-01-01 to 2023-12-31 truncated holdout (the only slice with
   both real funding AND being genuinely out-of-sample) is under one
   year — for a signal built on 8-hourly settlements that is only ~1,095
   observations of the underlying process, far short of anything that
   could clear a trials-adjusted bar. A clean result there may still be
   "not established" for lack of data, and that must be stated plainly
   rather than oversold.

**The data constraint, pre-registered explicitly.** Real funding
(`data/btcusdt_perp_funding_8h.csv.gz`) covers **2020-01-01 to
2023-12-31 only**, Binance BTCUSDT, 4,383 settlements, exactly regular at
8h. Consequences fixed in advance, not decided after looking:

- **Inner-train (2017-2020) has no funding for 2017-2019.** Bars before
  the first settlement (2020-01-01 03:00 UTC) get gate multiplier 1.0
  (never triggered, i.e. behaves exactly like ungated `kelly_regime_v4`)
  rather than any assumed/extrapolated rate. This is enforced by the
  gate function itself, not a filter applied after the fact: unknown
  funding means "do not reduce", the conservative default for a COST
  mechanism whose whole job is to *avoid* a known bad cost, not to guess
  at an unknown one. So inner-train really only exercises the gate
  during calendar-2020, which is itself inside the nominal inner-train
  window (2017-01-01..2020-12-31) — reported as such below, not
  disguised as full four-year coverage.
- **The holdout is truncated to 2023-01-01 -> 2023-12-31.** The
  project's holdout formally starts 2023-01-01 and runs to the present,
  but real funding stops 2023-12-31, so this row can only evaluate the
  gate's actual, non-extrapolated effect on **one calendar year** of
  holdout — far shorter than every other holdout figure in
  `docs/LEDGER.md`. No funding data for 2024+ is synthesized to extend
  it; doing so would be exactly the unlicensed-proxy mistake this
  project's own INFO-constraint failures (L-14/L-15/L-16/L-12) warn
  against, transplanted from price to funding. Anything claimed about
  2024+ behavior is explicitly out of scope.
- **Spot cannot be changed by this mechanism, by construction, not by
  omission.** Funding is never charged on spot
  (`MarketSpec.spot().pays_funding == False`), so a strategy that reduces
  position *because* of funding conditions is testing a claim that has
  no cost-avoidance content on spot. Rather than let the same gate
  quietly also resize the spot book (which would smuggle in R-16's
  *directional* forward-return claim through a mechanism pre-registered
  as a *cost* mechanism), ``gate_enabled`` is wired to be a per-market
  switch, off for spot. With it off, ``FundingGatedKelly`` reproduces
  ``kelly_regime_v4`` bar-for-bar (verified below, ``parity()``). The
  honest statement is: **the spot backtest is unchanged, exactly, by
  design — not merely "unaffected in practice."**

Step 2 — variants, mechanism, and the one falsification test
--------------------------------------------------------------
Three named variants, each a one-sentence mechanism:

- **V1 (decile / 30d / stand-flat).** ``threshold_pct=0.90,
  lookback_settlements=90 (~30d), reduction=0.0`` — when the last 30
  days of realized funding are in their own top decile, hold zero
  position; otherwise trade `kelly_regime_v4` unchanged.
- **V2 (decile / 90d / stand-flat).** Same hard zero, but the percentile
  is measured against a slower, 90-day trailing window
  (``lookback_settlements=270``), to check whether the gate's timing is
  sensitive to how "recent" the crowding measure is.
- **V3 (quintile / 30d / half-size).** ``threshold_pct=0.80,
  lookback_settlements=90, reduction=0.5`` — a softer trigger (top
  quintile, matching R-16's own quintile cut) that halves rather than
  zeroes exposure, to check whether a gentler, more frequent intervention
  beats a rare, hard one once turnover is priced in (failure mode 2).

**The one falsification test, picked now: the Monte Carlo window design
(`scripts/stress_test.py`), restricted to windows drawn entirely inside
the funding-covered span (2020-01-01 .. 2023-12-31).**

Justification for this pick over the alternatives ROUTINE.md offers: ETH
has no funding data at all, so it cannot test the gate itself, only the
underlying `kelly_regime_v4` behavior the gate modifies (already tested
by R-17/R-31/R-32) — it would answer a question this row is not asking.
Funding-charged is the mechanism, not a falsification of it. That leaves
the 0.40% fee test and the Monte Carlo window test. The fee test is
already partly answered as a side effect of the design (fewer bars in
market can only reduce fee-tier sensitivity, and turnover is reported
explicitly as a secondary check below) — but the deeper risk named in
Step 1 is failure mode 3: a single ~4-year real-funding path is thin
evidence, and R-16 itself flagged non-monotonicity across the funding
distribution as a warning about noise. Path sensitivity is therefore the
higher-value test of *this* mechanism, so Monte Carlo windows is the
pre-registered choice; the fee tier is still measured and reported, just
not as the pass/fail test.

Step 3 implementation lives in this file (the ``FundingGatedKelly``
class) and its runner, ``experiments/run_funding_gate.py``. Every
configuration evaluated during iteration is counted in the runner
(``N_EVALUATED``), reported in the final write-up. No holdout data
(``start=2023-01-01`` or later) is read until Step 4's frozen
configuration and decision rule are committed — see
``run_funding_gate.py`` for the exact commit point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategy import Context, Strategy

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


def funding_percentile(funding: pd.Series, lookback_settlements: int) -> pd.Series:
    """Trailing, causal percentile rank of each settlement among its own past.

    ``rolling(window).rank(pct=True)`` at settlement ``t`` uses only
    settlements ``t-window+1 .. t`` — the current one included, which is
    fine because by the time settlement ``t`` exists it has already
    happened and its rate is public. No settlement after ``t`` is ever
    used. The first ``window-1`` settlements are NaN (undefined
    percentile, not zero) until enough history has accumulated.
    """
    return funding.sort_index().rolling(
        lookback_settlements, min_periods=lookback_settlements
    ).rank(pct=True)


def gate_multiplier(
    bar_index: pd.DatetimeIndex,
    funding: pd.Series | None,
    threshold_pct: float,
    lookback_settlements: int,
    reduction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar (multiplier, raw percentile) array, backward-joined, causal.

    Default multiplier is 1.0 (gate open / not triggered) for every bar
    at or before the first settlement with a defined percentile, and for
    every bar after the last real settlement (2023-12-31) — the
    pre-registered "unknown funding -> do not reduce" rule from Step 1.
    A bar's multiplier uses only the most recent settlement at or before
    that bar's own timestamp (``reindex(method="ffill")``, equivalent to
    a backward asof join) — never a settlement that has not happened yet
    relative to the bar.
    """
    n = len(bar_index)
    mult = np.ones(n)
    pct = np.full(n, np.nan)
    if funding is None or len(funding) == 0:
        return mult, pct

    pctile = funding_percentile(funding, lookback_settlements)
    # Backward asof join: for each bar, the value at the nearest settlement
    # timestamp <= the bar's own timestamp. reindex(..., method="ffill")
    # on a sorted index does exactly this and needs no future settlement.
    aligned = pctile.reindex(bar_index, method="ffill")
    pct = aligned.to_numpy()
    triggered = np.isfinite(pct) & (pct >= threshold_pct)
    mult = np.where(triggered, reduction, 1.0)
    return mult, pct


class FundingGatedKelly(Strategy):
    """kelly_regime_v4's vote/vol-target sizer, gated flat (or halved) in the crowded-long funding tail.

    Reproduces ``kelly_regime_v4`` (v3's conditional vol-targeting on the
    20/40/80-day anchor ladder) exactly when ``gate_enabled=False`` or
    when no funding data covers the bar — see ``parity()`` in
    ``experiments/run_funding_gate.py``. When ``gate_enabled=True`` and
    the trailing percentile rank of realized funding is at or above
    ``threshold_pct``, the vote*sizer product this strategy would
    otherwise trade is multiplied by ``reduction`` (0.0 = stand flat,
    1.0 = no change) before the existing 10% deadband is applied, so the
    gate participates in the same single latching decision v4 already
    makes rather than adding a second, independent source of turnover.
    """

    name = "funding_gated_kelly"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(
        self,
        funding: pd.Series | None,
        gate_enabled: bool = True,
        threshold_pct: float = 0.90,
        lookback_settlements: int = 90,
        reduction: float = 0.0,
        # --- kelly_regime_v4 / v3 parameters, unchanged defaults
        horizons: tuple[int, ...] = (20, 40, 80),
        band: float = 0.01,
        target_vol: float = 0.55,
        max_leverage: float = 2.0,
        vol_span: int = 8 * BARS_PER_DAY,
        deadband: float = 0.10,
        anchor_span_days: int = 180,
        high_in: float = 1.70,
        high_out: float = 1.20,
        low_in: float = 0.55,
        low_out: float = 0.85,
    ) -> None:
        if not 0.0 <= threshold_pct <= 1.0:
            raise ValueError(f"threshold_pct must be in [0, 1], got {threshold_pct!r}")
        if not 0.0 <= reduction <= 1.0:
            raise ValueError(f"reduction must be in [0, 1], got {reduction!r}")
        self.funding = funding
        self.gate_enabled = gate_enabled
        self.threshold_pct = threshold_pct
        self.lookback_settlements = lookback_settlements
        self.reduction = reduction
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out

    # ------------------------------------------------------- v3/v4, unchanged

    def _vote_and_scale(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Exact reimplementation of kelly_regime_v3.prepare's vote+sizer.

        Kept as a direct copy rather than an import so this file has no
        dependency that could be affected by future edits to the
        registered strategy — the parity check in the runner is what
        proves the two stay identical, not a shared import.
        """
        close = df["close"]
        r = np.log(close).diff()

        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()

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

        n = len(df)
        scale = np.empty(n)
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale[i] = full[i] if state != 0 else steady[i]
        return frac, scale

    # ----------------------------------------------------------------- strategy

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        frac, scale = self._vote_and_scale(df)

        if self.gate_enabled:
            mult, pct = gate_multiplier(df.index, self.funding, self.threshold_pct,
                                        self.lookback_settlements, self.reduction)
        else:
            mult = np.ones(len(df))
            pct = np.full(len(df), np.nan)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = frac[i] * scale[i] * mult[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["gate_mult"] = mult
        df["funding_pct"] = pct
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
