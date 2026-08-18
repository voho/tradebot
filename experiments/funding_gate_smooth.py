#!/usr/bin/env python
"""Smooth funding-crowding gate on top of ``kelly_regime_v4`` (backlog B-05, novel branch).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Run as a script (see the command
table at the bottom) or import ``FundingGateSmooth`` / the grid helpers
directly.

This is one of two independent, parallel branches on backlog item B-05
("funding as a gate on the existing strategy"). A separate agent ran the
*conservative* branch (a literal binary top-decile stand-flat gate) on a
different file at the same time; per ROUTINE.md's parallel-round rules
neither branch reads or writes the other's file, and the trials counted
here are only this branch's own.

The question, and why a binary decile gate is not the whole story
--------------------------------------------------------------------
R-16 (docs/VALIDATION.md, "Funding as a positioning signal") found rich
funding predicts negative forward returns *unless price is also rising
strongly*, with a monotone-ish but noisy relationship across quintiles —
not a cliff that only exists in the rarest 10%. A hard 0/1 decile gate
throws away that gradient: it does nothing from the 0th to the 89th
percentile and then fully stands flat, no matter how close to the cliff
funding already is. This file asks whether a smooth, monotone-nonincreasing
penalty that starts de-risking around the 50th-70th percentile and eases
toward a floor as funding gets richer captures more of R-16's signal than
an all-or-nothing cutoff, while still respecting the COST constraint (the
standing diagnosis: "costs scale WITH the signal — funding runs +20%/yr
while the strategy holds vs +2.8% flat", R-14) by cutting exposure
*before* the crowding is total, not just at the extreme.

Mechanism (pre-registered, not to be revised after seeing results)
--------------------------------------------------------------------
1. Take ``kelly_regime_v4``'s existing ``target`` exposure unmodified —
   same three-anchor vote, same conditional vol-targeted Kelly sizer,
   same internal hysteresis. Nothing about v4's own logic changes.
2. Compute, **causally**, the trailing percentile rank ``p`` of the
   current funding rate: a rolling window of the last ``window_days``'
   worth of 8-hourly settlements (via ``pandas.Series.rolling(...).rank
   (pct=True)``, which by construction only ever looks backward — the
   percentile at settlement ``i`` uses settlements ``<= i`` and nothing
   after it). See ``funding_percentile`` below and the causality self-audit
   in this file's ``causality()`` command.
3. Map ``p`` through a smooth, monotone-nonincreasing "comfort" function
   into ``[floor, 1.0]``:　``comfort = 1 - clip((p - midpoint) / (1 -
   midpoint), 0, 1) * (1 - floor)``. Equal to 1.0 for any ``p <=
   midpoint``; falls linearly (a ramp, not a step) to ``floor`` as ``p``
   approaches 1.0. ``midpoint`` and ``floor`` are free parameters, chosen
   on inner-validation.
4. Multiply v4's target by ``comfort`` **only when ``ctx.market.
   pays_funding`` is True** — spot is unaffected, since funding is not a
   cost there and R-16's signal was measured against a cost that does not
   exist on spot. This is verified explicitly by ``causality()``'s spot
   parity check below (spot must be bit-identical to plain
   ``kelly_regime_v4``, at every fee tier, because the gated code path is
   never entered).
5. Because the comfort factor drifts continuously bar-to-bar (funding
   ticks in a fixed rate for 8h, but the trailing percentile can still
   move as the window ages), an outer deadband is applied to the *gated*
   series in ``prepare()`` — the same latching idiom v4 already uses
   internally and that ``matched_risk.py`` uses for its gates — so the
   funding signal changes exposure a few times, not every bar. Without
   this the strategy would re-latch on every 5-minute wiggle in the
   percentile and manufacture turnover the mechanism was never meant to
   add (this is exactly the COST-constraint trap R-12 fell into).

Pre-registered decision rule — written and frozen BEFORE the 2023
holdout is read
--------------------------------------------------------------------
Promote only if, on the 2023 holdout (2023-01-01 -> 2023-12-31, the only
calendar year both inside this project's OOS_START=2023-01-01 split and
inside the committed 2020-2023 funding data), on futures 5x with real
funding charged, ALL of:

  P1  final balance (equivalently log growth) beats funding-charged
      ``kelly_regime_v4`` over the identical window.
  P2  the improvement exceeds the project's +/-0.2 Sharpe noise floor
      (R-20), OR is a max-drawdown improvement of >= 10 percentage
      points, AND is not fully explained by the matched-risk check (if
      realized_vol() drops by roughly the same proportion as drawdown or
      return improves, that is reported as a caveat even where P2 is met
      on the raw numbers — a lower-vol book almost always draws down less
      and returns less, which is R-31/R-32's lesson, not evidence of
      timing skill by itself).
  P3  falsification test: the ranking in P1 still holds when spot/futures
      taker fees are raised to the 0.40% Bitstamp entry tier (following
      ``scripts/fee_study.py``'s pattern). Targets R-12's lesson:
      turnover-driven improvements that only exist at an unrealistic
      0.10% fee tier are dead on arrival.
  P4  the (midpoint, floor) grid is a plateau on inner-validation, not a
      single lucky peak — neighbours are reported, not just the winner.

If any of P1-P4 fails, the result is NEGATIVE. If this rule would need to
change after seeing the 2023 numbers to produce a promotion, it will NOT
be changed — the result will be reported as NEGATIVE with the rule stated
as written here, per ROUTINE.md step 4 ("going back to find a threshold
that turns a rejection into a promotion is the thing that produced
28-of-32 in-sample winners and 0-of-28 out-of-sample", R-12).

Split (a modification of the standard ROUTINE.md split, forced by
funding data covering only 2020-01-01 -> 2023-12-31)
--------------------------------------------------------------------
    inner-train        2020-01-01 -> 2020-12-31   fit / iterate freely
    inner-validation    2021-01-01 -> 2022-12-31   select (midpoint, floor)
    holdout (frozen)    2023-01-01 -> 2023-12-31   step 4 ONLY, one year

A one-year holdout is low power compared to this project's usual 3.6-year
OOS window; that is stated plainly in the report and no claim here should
be read as more decisive than a single year supports.

Usage
-----
    python experiments/funding_gate_smooth.py sweep      # step 3, inner split
    python experiments/funding_gate_smooth.py causality   # lookahead self-audit
    python experiments/funding_gate_smooth.py holdout     # step 4, frozen config
    python experiments/funding_gate_smooth.py fees        # P3 falsification
    python experiments/funding_gate_smooth.py all
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY
SETTLEMENTS_PER_DAY = 3  # perpetual funding settles every 8h


# ---------------------------------------------------------------------------
# the matching axis (copied, not imported, per the task's file isolation —
# same ~10-line function experiments/matched_risk.py uses)
# ---------------------------------------------------------------------------


def realized_vol(equity: pd.Series | np.ndarray) -> float:
    """Annualized standard deviation of per-bar equity returns."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 3:
        return 0.0
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
    return float(np.std(rets, ddof=1) * np.sqrt(BARS_PER_YEAR))


# ---------------------------------------------------------------------------
# the funding signal
# ---------------------------------------------------------------------------


def funding_percentile(funding: pd.Series | None, index: pd.DatetimeIndex,
                        window_days: float, min_settlements: int = 10) -> np.ndarray:
    """Causal trailing percentile rank of funding, aligned onto ``index``.

    Two steps, both strictly backward-looking:

    1. At the SETTLEMENT level (8-hourly, ~3/day — computing the rank
       there rather than on the forward-filled bar series is both cheaper
       and the more natural definition of "trailing percentile of the
       funding rate", since funding only actually changes at a
       settlement): ``rolling(window_settlements).rank(pct=True)``. This
       is the percentile of the *most recent* settlement in a window
       ending at and including it — never a settlement that has not
       happened yet, and never a quantile fit over the whole series
       (which is exactly the lookahead class ROUTINE.md warns the
       truncation test will not catch on its own).
    2. Reindexed onto the bar index with ``method="ffill"``: a bar sees
       the percentile computed from the most recent settlement AT OR
       BEFORE its own timestamp, and nothing from after it. Bars before
       the first settlement (or before ``min_settlements`` have
       accumulated) get NaN, which the comfort function below reads as
       "no funding information yet" and maps to full comfort (1.0) — the
       gate only ever turns exposure DOWN from a fully-open default, so
       missing history cannot manufacture an advantage.
    """
    n = len(index)
    if funding is None or len(funding) == 0:
        return np.full(n, np.nan)

    f = funding.sort_index()
    window_settlements = max(min_settlements, int(round(window_days * SETTLEMENTS_PER_DAY)))
    pct_at_settlement = f.rolling(window_settlements, min_periods=min_settlements).rank(pct=True)

    bar_index = pd.DatetimeIndex(index)
    if bar_index.tz is None and pct_at_settlement.index.tz is not None:
        bar_index = bar_index.tz_localize(pct_at_settlement.index.tz)
    aligned = pct_at_settlement.reindex(bar_index, method="ffill")
    return aligned.to_numpy(dtype=float)


def comfort_from_percentile(p: np.ndarray, midpoint: float, floor: float) -> np.ndarray:
    """Smooth, monotone-nonincreasing map from percentile to exposure comfort.

    ``comfort(p) = 1`` for ``p <= midpoint``; ramps linearly down to
    ``floor`` as ``p -> 1``. A ramp, not a step — the point of this branch
    versus the conservative decile-cutoff branch running in parallel.
    """
    if not (0.0 <= floor <= 1.0):
        raise ValueError(f"floor must be in [0, 1], got {floor!r}")
    if not (0.0 <= midpoint < 1.0):
        raise ValueError(f"midpoint must be in [0, 1), got {midpoint!r}")
    clipped = np.clip(p, 0.0, 1.0)
    ramp = np.clip((clipped - midpoint) / (1.0 - midpoint), 0.0, 1.0)
    comfort = 1.0 - ramp * (1.0 - floor)
    # NaN (no funding history yet) -> full comfort, never a penalty.
    return np.where(np.isfinite(p), comfort, 1.0)


# ---------------------------------------------------------------------------
# the strategy
# ---------------------------------------------------------------------------


class FundingGateSmooth(KellyRegimeV4):
    """``kelly_regime_v4`` with exposure smoothly de-risked by funding crowding.

    See the module docstring for the mechanism and the pre-registered
    decision rule. Everything about v4 (anchors, sizer, deadband, warmup)
    is unchanged; this class only adds the funding comfort multiplier,
    applied on futures only.
    """

    name = "funding_gate_smooth"

    def __init__(self, funding: pd.Series | None = None, midpoint: float = 0.6,
                 floor: float = 0.5, window_days: float = 30.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.funding = funding
        self.midpoint = midpoint
        self.floor = floor
        self.window_days = window_days

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # sets df["target"] exactly as kelly_regime_v4 does
        raw = df["target"].to_numpy(dtype=float)

        p = funding_percentile(self.funding, df.index, self.window_days)
        comfort = comfort_from_percentile(p, self.midpoint, self.floor)

        # Outer deadband on the GATED series only (v4's own target already
        # carries its own internal hysteresis on the ungated path used for
        # spot). Without this, a continuously drifting comfort factor would
        # re-latch every bar the percentile ticks, adding turnover the
        # mechanism never intended - see the module docstring, step 5.
        n = len(df)
        gated = np.empty(n)
        pos = 0.0
        for i in range(n):
            desired = raw[i] * comfort[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            gated[i] = pos

        df["funding_pctl"] = p
        df["comfort"] = comfort
        df["target_gated"] = gated
        return df

    def on_bar(self, ctx: Context) -> None:
        # Spot never pays funding -> always the plain v4 target, so this
        # strategy is bit-identical to kelly_regime_v4 on spot at every fee
        # tier. Only futures (pays_funding=True) ever reads the gated path.
        key = "target_gated" if ctx.market.pays_funding else "target"
        t = float(ctx.bar[key])
        prev = float(ctx.prev[key]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# ---------------------------------------------------------------------------
# harness plumbing
# ---------------------------------------------------------------------------

DF, LABEL = load_dataset(ROOT / "data", "spot")
FUNDING = load_funding(ROOT / "data")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures_5x", FUTURES))

TRAIN = ("2020-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
HOLDOUT = ("2023-01-01", "2023-12-31")  # ONE year only - see module docstring

OUT = ROOT / "reports" / "funding_gate_smooth"

N_EVALUATED = 0          # distinct (midpoint, floor) configs scored in step 3
HOLDOUT_READS = 0        # every run that touches 2023 data, counted honestly


def run_period_funded(strategy, df, start, end, *, market, funding=None,
                      start_balance: float = 1_000.0):
    """``tradebot.window.run_period``, but able to pass ``funding`` through.

    ``run_period`` does not accept a ``funding`` kwarg (checked directly
    against ``src/tradebot/window.py``), so charging funding on a
    sub-period backtest means reproducing its warmup-prefix trimming by
    hand - the same pattern ``experiments/run_matched_risk.py``'s
    ``costs()`` uses for the identical reason.
    """
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    if hi <= lo:
        raise ValueError(f"empty period: {start!r} -> {end!r}")
    prefix = int(min(lo, max(strategy.warmup, 0)))
    frame = df.iloc[lo - prefix: hi]
    raw = run_backtest(strategy, frame, market, start_balance,
                       trade_start=prefix, funding=funding, data_label=LABEL)
    if prefix == 0:
        return raw
    return replace(raw, equity=raw.equity.iloc[prefix:], df=raw.df.iloc[prefix:])


def measure(strategy, start, end, *, market, funding=None, balance=1_000.0):
    """One funded backtest -> (Metrics, realized_vol, BacktestResult)."""
    result = run_period_funded(strategy, DF, start, end, market=market,
                               funding=funding, start_balance=balance)
    m = compute_metrics(result)
    return m, realized_vol(result.equity), result


def line(tag, m, vol, result):
    print(f"  {tag:34s} final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
          f"vol={vol:5.3f} trades={m.num_trades:>4d} "
          f"funding=${result.funding_paid:>8,.0f} fees=${m.fees_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


MIDPOINTS = (0.5, 0.6, 0.7)
FLOORS = (0.3, 0.5, 0.7)
WINDOW_DAYS = 30.0  # fixed, not swept - see module docstring ("free parameters")


# --------------------------------------------------------------------- sweep


def sweep() -> pd.DataFrame:
    """Step 3: score every (midpoint, floor) config on inner-train and
    inner-validation, futures 5x with real funding charged. Selection
    happens on inner-validation only; inner-train is reported for
    reference and to catch anything degenerate before it reaches
    inner-validation."""
    global N_EVALUATED
    rows = []
    for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        print(f"\n{split_name} ({start} -> {end}), futures 5x, real funding charged:")
        base_m, base_vol, base_res = measure(get_strategy("kelly_regime_v4"), start, end,
                                             market=FUTURES, funding=FUNDING)
        line("kelly_regime_v4 (baseline)", base_m, base_vol, base_res)
        hold_m, hold_vol, hold_res = measure(get_strategy("buy_and_hold"), start, end,
                                             market=SPOT, funding=None)
        line("buy_and_hold (spot benchmark)", hold_m, hold_vol, hold_res)
        rows.append({"split": split_name, "midpoint": np.nan, "floor": np.nan,
                     "strategy": "kelly_regime_v4", "final": base_m.final_balance,
                     "log_growth": float(np.log(base_m.final_balance / 1000.0)),
                     "max_dd": base_m.max_drawdown_pct, "sharpe": base_m.sharpe,
                     "vol": base_vol, "trades": base_m.num_trades,
                     "funding_paid": base_res.funding_paid})
        rows.append({"split": split_name, "midpoint": np.nan, "floor": np.nan,
                     "strategy": "buy_and_hold", "final": hold_m.final_balance,
                     "log_growth": float(np.log(hold_m.final_balance / 1000.0)),
                     "max_dd": hold_m.max_drawdown_pct, "sharpe": hold_m.sharpe,
                     "vol": hold_vol, "trades": hold_m.num_trades, "funding_paid": 0.0})

        for midpoint in MIDPOINTS:
            for floor in FLOORS:
                s = FundingGateSmooth(funding=FUNDING, midpoint=midpoint, floor=floor,
                                      window_days=WINDOW_DAYS)
                m, vol, res = measure(s, start, end, market=FUTURES, funding=FUNDING)
                if split_name == "inner-train":
                    N_EVALUATED += 1  # count each distinct config once, not per split
                line(f"midpoint={midpoint} floor={floor}", m, vol, res)
                rows.append({"split": split_name, "midpoint": midpoint, "floor": floor,
                             "strategy": "funding_gate_smooth", "final": m.final_balance,
                             "log_growth": float(np.log(m.final_balance / 1000.0)),
                             "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                             "vol": vol, "trades": m.num_trades,
                             "funding_paid": res.funding_paid})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "sweep.csv", index=False)
    print(f"\nconfigurations evaluated (distinct midpoint x floor, step 3): {N_EVALUATED}")
    print(f"written: {OUT / 'sweep.csv'}")

    val = out[(out.split == "inner-validation") & (out.strategy == "funding_gate_smooth")]
    best = val.sort_values("final", ascending=False).iloc[0]
    print(f"\nbest on inner-validation (final balance): "
          f"midpoint={best.midpoint} floor={best.floor} "
          f"final=${best.final:,.0f} sharpe={best.sharpe:.2f} DD={best.max_dd:.1f}%")
    print("\nplateau check - full inner-validation grid, final balance ($):")
    pivot = val.pivot(index="midpoint", columns="floor", values="final")
    print(pivot.round(0).to_string())
    return out


# --------------------------------------------------------------------- causality


def causality() -> None:
    """By-hand lookahead probe - experiments get no CI protection.

    Two checks:
    1. Tamper bars after a cut (multiply vs divide price by a constant)
       and confirm every order decision at or before the cut is identical
       - the standard truncation probe (R-21's method).
    2. The check truncation alone WOULD NOT catch (ROUTINE.md's explicit
       warning): confirm ``funding_pctl`` computed with the funding series
       truncated at some future date is identical, before that date, to
       ``funding_pctl`` computed with the full 2020-2023 series. A
       whole-series quantile would fail this; a rolling trailing one
       passes by construction.
    3. Spot parity: on spot (``pays_funding=False``), the strategy must be
       bit-identical to ``kelly_regime_v4`` regardless of the funding
       series or gate parameters, since the gated column is never read.
    """
    from tradebot.broker import PaperBroker

    df = DF.iloc[-260_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        s = FundingGateSmooth(funding=FUNDING, midpoint=0.6, floor=0.5)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down)) if oa != ob]
    pa = FundingGateSmooth(funding=FUNDING, midpoint=0.6, floor=0.5).prepare(up.copy())
    pb = FundingGateSmooth(funding=FUNDING, midpoint=0.6, floor=0.5).prepare(down.copy())
    worst = max(float(np.nanmax(np.abs(pa[c].to_numpy()[:cut] - pb[c].to_numpy()[:cut])))
                for c in ("target", "target_gated", "comfort", "funding_pctl"))
    good1 = not bad and worst < 1e-9
    print(f"1. price-tamper truncation probe: orders {'match' if not bad else f'DIFFER at {bad}'}, "
          f"max |column diff| before cut = {worst:.3e}   {'PASS' if good1 else 'FAIL'}")

    truncated_funding = FUNDING[FUNDING.index < df.index[cut]]
    p_full = funding_percentile(FUNDING, df.index, WINDOW_DAYS)
    p_trunc = funding_percentile(truncated_funding, df.index, WINDOW_DAYS)
    before = np.abs(p_full[:cut] - p_trunc[:cut])
    before = before[np.isfinite(before)]
    worst2 = float(before.max()) if len(before) else 0.0
    good2 = worst2 < 1e-9
    print(f"2. funding-series truncation probe (the one a price-only truncation test "
          f"would NOT catch): max |percentile diff| before cut = {worst2:.3e}   "
          f"{'PASS' if good2 else 'FAIL'}")

    v4 = get_strategy("kelly_regime_v4")
    fgs = FundingGateSmooth(funding=FUNDING, midpoint=0.2, floor=0.1)  # aggressive params on purpose
    ra = run_backtest(v4, df, SPOT, 10_000.0, data_label=LABEL)
    rb = run_backtest(fgs, df, SPOT, 10_000.0, data_label=LABEL)
    worst3 = float(np.max(np.abs(ra.equity.to_numpy() - rb.equity.to_numpy())))
    good3 = worst3 < 1e-6
    print(f"3. spot parity (funding never applies): max |equity diff| vs plain "
          f"kelly_regime_v4 over the full window = {worst3:.3e}   {'PASS' if good3 else 'FAIL'}")

    print(f"\noverall: {'PASS' if (good1 and good2 and good3) else 'FAIL'}")


# --------------------------------------------------------------------- holdout


# Frozen after `sweep()` was run and inspected on inner-train/inner-validation
# ONLY (9 configs on the pre-registered 3x3 grid, plus a 12-config
# supplementary robustness probe at floor in {0.0, 0.1, 0.2, 0.4} to check
# whether the 3x3 grid's winner sat on a real plateau or a noisy edge - see
# the report this script's author returned for the full reasoning and the
# 21-config total). midpoint=0.5, floor=0.3 was the pre-registered grid's
# best inner-validation final balance ($1,162 vs kelly_regime_v4's $887 and
# buy_and_hold's $574); the supplementary probe found floor=0.0 nominally
# higher ($1,264) but bouncing non-monotonically through floor in
# {0.1, 0.2, 0.3, 0.4} ($1,172 / $1,079 / $1,162 / $1,120) rather than
# settling smoothly, which reads as noise around a shallow, wide optimum
# rather than a sharp peak worth chasing to a grid boundary. The
# pre-registered grid's own winner is kept as the frozen point rather than
# moving to the single best number the supplementary probe happened to
# turn up - see ROUTINE.md step 4 on not moving the goalposts after
# looking, applied here to config selection rather than to the holdout
# itself, out of the same caution. Do not tune these by looking at
# holdout() output.
FROZEN_MIDPOINT = 0.5
FROZEN_FLOOR = 0.3
FROZEN_WINDOW_DAYS = WINDOW_DAYS


def holdout() -> pd.DataFrame:
    """Step 4. The ONE 2023 evaluation the decision rule is pre-registered
    against. Every call to this function (or to fees()) is a holdout read;
    count them in the report."""
    global HOLDOUT_READS
    HOLDOUT_READS += 1
    rows = []
    print(f"\nHOLDOUT {HOLDOUT[0]} -> {HOLDOUT[1]} (frozen config: "
          f"midpoint={FROZEN_MIDPOINT} floor={FROZEN_FLOOR} "
          f"window_days={FROZEN_WINDOW_DAYS})\n")
    for mname, market in MARKETS:
        funding_for_market = FUNDING if market.pays_funding else None
        print(f"{mname}:")
        contenders = [
            ("funding_gate_smooth", FundingGateSmooth(funding=FUNDING, midpoint=FROZEN_MIDPOINT,
                                                       floor=FROZEN_FLOOR,
                                                       window_days=FROZEN_WINDOW_DAYS)),
            ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
        ]
        if mname == "spot":
            contenders.append(("buy_and_hold", get_strategy("buy_and_hold")))
        else:
            contenders.append(("buy_and_hold (spot benchmark)", None))
        for tag, strat in contenders:
            if strat is None:
                m, vol, res = measure(get_strategy("buy_and_hold"), *HOLDOUT, market=SPOT,
                                      funding=None)
            else:
                m, vol, res = measure(strat, *HOLDOUT, market=market,
                                      funding=funding_for_market)
            line(f"  {tag}", m, vol, res)
            rows.append({"market": mname, "strategy": tag, "final": m.final_balance,
                         "log_growth": float(np.log(m.final_balance / 1000.0)),
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe, "vol": vol,
                         "trades": m.num_trades, "funding_paid": res.funding_paid,
                         "fees_paid": m.fees_paid, "liquidated": m.liquidated})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "holdout.csv", index=False)
    print(f"\nwritten: {OUT / 'holdout.csv'}")
    print(f"holdout reads so far (this process): {HOLDOUT_READS}")
    return out


# ------------------------------------------------------------------------ fees


def fees() -> pd.DataFrame:
    """P3, the pre-registered falsification test: does P1's ranking survive
    the 0.40% Bitstamp entry taker tier? Futures only, real funding charged
    (funding is independent of the fee tier); spot is skipped because the
    gate never activates there (see causality() check 3), so a fee-tier
    change cannot move the P1 ranking on spot - variant == baseline exactly."""
    global HOLDOUT_READS
    HOLDOUT_READS += 1
    rows = []
    print(f"\nP3 falsification: HOLDOUT {HOLDOUT[0]} -> {HOLDOUT[1]}, futures 5x, "
          f"real funding charged, at two fee tiers\n")
    for tier, label in ((0.0005, "0.05% (table default)"), (0.004, "0.40% (Bitstamp entry)")):
        market = MarketSpec.futures(leverage=5.0, fee_rate=tier)
        print(f"{label}:")
        variant = FundingGateSmooth(funding=FUNDING, midpoint=FROZEN_MIDPOINT,
                                    floor=FROZEN_FLOOR, window_days=FROZEN_WINDOW_DAYS)
        base = get_strategy("kelly_regime_v4")
        mv, volv, resv = measure(variant, *HOLDOUT, market=market, funding=FUNDING)
        mb, volb, resb = measure(base, *HOLDOUT, market=market, funding=FUNDING)
        line("  funding_gate_smooth", mv, volv, resv)
        line("  kelly_regime_v4", mb, volb, resb)
        wins = mv.final_balance > mb.final_balance
        print(f"  variant beats baseline (P1 ranking): {wins}")
        rows.append({"fee_tier": tier, "label": label, "strategy": "funding_gate_smooth",
                     "final": mv.final_balance, "max_dd": mv.max_drawdown_pct,
                     "sharpe": mv.sharpe, "vol": volv})
        rows.append({"fee_tier": tier, "label": label, "strategy": "kelly_regime_v4",
                     "final": mb.final_balance, "max_dd": mb.max_drawdown_pct,
                     "sharpe": mb.sharpe, "vol": volb})
        rows.append({"fee_tier": tier, "label": label, "strategy": "P1_ranking_holds",
                     "final": float(wins), "max_dd": np.nan, "sharpe": np.nan, "vol": np.nan})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "fees.csv", index=False)
    print(f"\nwritten: {OUT / 'fees.csv'}")
    print(f"holdout reads so far (this process): {HOLDOUT_READS}")
    return out


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    print(f"funding: {FUNDING.index[0]:%Y-%m-%d} -> {FUNDING.index[-1]:%Y-%m-%d}  "
          f"({len(FUNDING)} settlements)" if FUNDING is not None else "funding: MISSING",
          file=sys.stderr)
    cmds = {"sweep": sweep, "causality": causality, "holdout": holdout, "fees": fees}

    def all_() -> None:
        sweep()
        causality()
        holdout()
        fees()

    cmds["all"] = all_
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/funding_gate_smooth.py [{'|'.join(cmds)}]")
