#!/usr/bin/env python
"""Carry-adjusted Kelly dampening (R-33 pre-registration, variant B).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

The idea, one sentence
-----------------------
``kelly_regime_v4`` sizes exposure from price alone; the futures engine
charges real perpetual funding as an unmodeled cost. This variant feeds
the strategy's own trailing realized funding rate back into its sizing,
continuously and with no threshold, following the textbook Kelly-under-
financing-cost identity ``f* = (mu - cost) / sigma^2`` (QuantStart,
"Money Management via the Kelly Criterion"; Atlas Peak Research, "The
Kelly Criterion in Financial Markets"). ``kelly_regime_v4`` already
implements the ``mu/sigma^2`` shape as ``vote_fraction * min(target_vol
/ vol, max_leverage)``; this subclass nets a trailing, EWM-smoothed,
annualized funding rate out of that fraction as a smooth multiplicative
drag, expressed relative to the strategy's own ``target_vol`` (its units
for "how much variance a full-Kelly bet is allowed to carry").

This is variant **B** of R-33; variant **A** (``funding_veto``, a
conservative hard threshold) is being built independently in
``experiments/funding_veto.py`` and is deliberately not read here — the
two variants are compared only through their reported inner-validation
numbers, per the pre-registration.

Only real funding is used (``data/btcusdt_perp_funding_8h.csv.gz``,
2020-01-01..2023-12-31, 4,384 settlements). Every command in this file
stays inside 2020-01-01..2022-12-31 (inner-train / inner-validation) or,
for the spot-identity check, 2020-01-01..2022-12-31 as well — the
2023-01-01+ holdout is never read or reported from this file.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Context  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

# Sweep grid, and the two inner windows (holdout is 2023-01-01+, never
# touched here).
HALFLIVES = (1.0, 3.0, 7.0)
DRAG_CAPS = (0.5, 1.0, 2.0)
INNER_TRAIN = ("2020-01-01", "2021-12-31")
INNER_VAL = ("2022-01-01", "2022-12-31")
PERIODS = (("inner-train", INNER_TRAIN), ("inner-validation", INNER_VAL))


# --------------------------------------------------------------------- strategy


class CarryKelly(KellyRegimeV4):
    """``kelly_regime_v4`` with a continuous carry-adjusted Kelly dampening.

    Mechanism: ``f* = (mu - cost) / sigma^2`` under a financing cost is
    the growth-optimal fraction net of carry. ``v4`` already prices
    ``mu`` via its regime vote and ``sigma`` via realized vol; this
    class subtracts the strategy's own trailing realized funding cost
    from the numerator, expressed in the same ``target_vol`` units the
    sizer already uses, as a smooth multiplier — never a hard threshold
    (that is variant A, ``funding_veto``, built independently).

    Only the funding-paying market (futures) sees the multiplier. Spot
    is bit-identical to plain ``kelly_regime_v4`` by construction: this
    class never writes to ``df["target"]`` and ``on_bar`` reads
    ``df["target"]`` unmodified whenever ``ctx.market.pays_funding`` is
    False.
    """

    name = "carry_kelly"

    def __init__(self, funding_series: pd.Series, funding_halflife_days: float = 3.0,
                 drag_cap: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        if funding_series is None or len(funding_series) == 0:
            raise ValueError("carry_kelly requires a non-empty funding series")
        self.funding_series = funding_series.sort_index()
        self.funding_halflife_days = funding_halflife_days
        self.drag_cap = drag_cap

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4_target -> df["target"], untouched from here on

        # 1. annualize (3 settlements/day)
        funding_ann = self.funding_series * 3 * 365.25

        # 2. reindex onto the bar index causally: ffill, then zero before
        #    the first real settlement.
        funding_on_bars = funding_ann.reindex(df.index, method="ffill").fillna(0.0)

        # 3. extra causal safety margin against a settlement landing on a
        #    bar boundary.
        funding_on_bars = funding_on_bars.shift(1).fillna(0.0)

        # 4. causal EWM smoothing (span in bars; "half-life" here names
        #    the constructor knob H, wired directly as `span`, matching
        #    the pre-registration's `trailing = funding_ann.ewm(span=H).mean()`)
        span = self.funding_halflife_days * BARS_PER_DAY
        trailing = funding_on_bars.ewm(span=span, min_periods=1).mean()

        # 5. drag, expressed in target_vol units, capped by drag_cap; the
        #    applied multiplier is separately floored/capped at [0, 1] so
        #    a drag_cap > 1.0 only changes how fast the floor is reached,
        #    never lets the multiplier go negative or above 1.
        drag = np.clip((trailing / self.target_vol).to_numpy(dtype=float), 0.0, self.drag_cap)
        effective_multiplier = np.clip(1.0 - drag, 0.0, 1.0)
        assert effective_multiplier.min() >= 0.0 and effective_multiplier.max() <= 1.0, (
            f"effective_multiplier left [0,1]: min={effective_multiplier.min()} "
            f"max={effective_multiplier.max()}")

        # 6. store; df["target"] (spot) is left exactly as v4 computed it.
        df["trailing_funding_ann"] = trailing.to_numpy()
        df["drag"] = drag
        df["effective_multiplier"] = effective_multiplier
        df["carry_target"] = df["target"].to_numpy() * effective_multiplier
        return df

    def on_bar(self, ctx: Context) -> None:
        if ctx.market.pays_funding:
            t = float(ctx.bar["carry_target"])
            prev = float(ctx.prev["carry_target"]) if ctx.prev is not None else 0.0
        else:
            # bit-identical to KellyRegime.on_bar / v4 on spot
            t = float(ctx.bar["target"])
            prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# ------------------------------------------------------------------------ runs


def _period(strategy, market, start=None, end=None, funding=None):
    """Backtest over a date range, warmed on the bars before it (funding_study.py pattern)."""
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                        trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw.funding_paid, trimmed.df


def _row(label, m, funding_paid, avg_mult=None):
    log_growth = float(np.log(m.final_balance / m.start_balance))
    mult_s = f"{avg_mult:6.3f}" if avg_mult is not None else "   n/a"
    print(f"  {label:34s} final=${m.final_balance:>10,.0f}  logg={log_growth:+.4f}  "
          f"maxDD={m.max_drawdown_pct:6.2f}%  sharpe={m.sharpe:6.2f}  "
          f"trades={m.num_trades:4d}  avg_mult={mult_s}  funding=${funding_paid:>9,.0f}")
    return dict(final_balance=m.final_balance, log_growth=log_growth,
                max_dd=m.max_drawdown_pct, sharpe=m.sharpe, trades=m.num_trades,
                avg_mult=avg_mult, funding_paid=funding_paid)


def sweep() -> tuple[dict, dict]:
    """9 configs x 2 inner windows, plus the kelly_regime_v4 baseline on the same windows."""
    if REAL is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")

    print(f"{'baseline: kelly_regime_v4, 5x futures, real funding charged':74s}\n")
    baseline = {}
    for label, (start, end) in PERIODS:
        m, funding_paid, _ = _period(get_strategy("kelly_regime_v4"), FUTURES, start, end, funding=REAL)
        baseline[label] = _row(f"v4 baseline  [{label}]", m, funding_paid)

    configs = [(hl, dc) for hl in HALFLIVES for dc in DRAG_CAPS]
    n_backtests = len(configs) * len(PERIODS)
    print(f"\ncarry_kelly sweep: {len(configs)} configs x {len(PERIODS)} periods = "
          f"{n_backtests} backtests\n")
    results = {}
    for hl, dc in configs:
        for label, (start, end) in PERIODS:
            strat = CarryKelly(REAL, funding_halflife_days=hl, drag_cap=dc)
            m, funding_paid, df = _period(strat, FUTURES, start, end, funding=REAL)
            avg_mult = float(df["effective_multiplier"].mean())
            results[(hl, dc, label)] = _row(
                f"H={hl:>3.0f}d cap={dc:>3.1f}  [{label}]", m, funding_paid, avg_mult)
    return baseline, results


def best_config(results: dict) -> tuple[float, float]:
    """Best (halflife, drag_cap) by inner-validation log growth."""
    val = {(hl, dc): v for (hl, dc, label), v in results.items() if label == "inner-validation"}
    return max(val, key=lambda k: val[k]["log_growth"])


def best_by_drawdown(results: dict) -> tuple[float, float]:
    """Best (halflife, drag_cap) by inner-validation max drawdown (lower is better)."""
    val = {(hl, dc): v for (hl, dc, label), v in results.items() if label == "inner-validation"}
    return min(val, key=lambda k: val[k]["max_dd"])


# --------------------------------------------------------------------- causality


def causality() -> float:
    """Hand causality probe (experiments get none of test_causality_strict.py's coverage).

    Two *opposite* tampers (R-28/R-31 pattern, see experiments/run_matched_risk.py
    and experiments/run_gate_control.py): bars after a cut are multiplied by 3
    in one copy and divided by 3 in the other; every decision at or before the
    cut must be identical. Restricted to the 2020-2022 inner window so the
    probe never touches 2023+ data, even synthetically.
    """
    start, end = "2020-01-01", "2022-12-31"
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    df = DF.iloc[lo:hi].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        strat = CarryKelly(REAL, funding_halflife_days=3.0, drag_cap=1.0)
        prepared = strat.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            strat.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out, prepared

    dec_up, prep_up = decisions(up)
    dec_down, prep_down = decisions(down)

    bad = [b for b, oa, ob in zip(bars, dec_up, dec_down) if oa != ob]
    cols = ("target", "carry_target", "effective_multiplier")
    worst = max(
        float(np.nanmax(np.abs(prep_up[c].to_numpy()[:cut] - prep_down[c].to_numpy()[:cut])))
        for c in cols)
    mult = prep_up["effective_multiplier"].to_numpy()
    ok = not bad and worst == 0.0
    print(f"probe window {start}..{end}; tampered from bar {cut:,} of {len(df):,} "
          f"(x3 vs /3 on OHLC, x7 vs /7 on volume, from the cut onward)")
    print(f"  orders at {bars} {'match' if not bad else f'DIFFER at {bad}'}")
    print(f"  max |{'/'.join(cols)} diff| before the cut = {worst:.3e}")
    print(f"  effective_multiplier range over the whole probe frame: "
          f"[{mult.min():.6f}, {mult.max():.6f}]  (must stay within [0, 1])")
    print(f"  {'PASS' if ok else 'FAIL'} — no decision at or before the cut moved")
    return worst


# ------------------------------------------------------------------ spot check


def spotcheck(halflife_days: float, drag_cap: float) -> float:
    """Confirm carry_kelly's SPOT backtest is bit-identical to plain kelly_regime_v4's.

    Window is 2020-01-01..2022-12-31 (inner-train + inner-validation), not
    2020-01-01..2023-12-31 as the mechanical spec literally says — the
    overriding instruction not to touch 2023+ data takes precedence, and
    since this checks a structural code invariant (spot ignores the
    funding multiplier entirely), not a performance read, 2020-2022 fully
    establishes it without reading the holdout.
    """
    start, end = "2020-01-01", "2022-12-31"
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    carry = CarryKelly(REAL, funding_halflife_days=halflife_days, drag_cap=drag_cap)
    v4 = get_strategy("kelly_regime_v4")
    pre = min(lo, max(carry.warmup, v4.warmup))
    window = DF.iloc[lo - pre: hi]

    r_carry = run_backtest(carry, window, SPOT, 1_000.0, trade_start=pre, data_label=LABEL)
    r_v4 = run_backtest(v4, window, SPOT, 1_000.0, trade_start=pre, data_label=LABEL)
    eq_carry = r_carry.equity.iloc[pre:].to_numpy()
    eq_v4 = r_v4.equity.iloc[pre:].to_numpy()
    diff = float(np.max(np.abs(eq_carry - eq_v4)))
    print(f"spot identity check (H={halflife_days:g}d, drag_cap={drag_cap:g}), "
          f"window {start}..{end}, {len(eq_carry):,} bars compared")
    print(f"  max |equity diff| = {diff:.3e}   {'PASS' if diff == 0.0 else 'FAIL'}")
    return diff


# --------------------------------------------------------------------------- CLI


COMMANDS = ("causality", "sweep", "spotcheck", "all")


def main() -> None:
    if REAL is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    if choice not in COMMANDS:
        print(f"usage: python experiments/carry_kelly.py [{'|'.join(COMMANDS)}]")
        return

    if choice in ("causality", "all"):
        print(f"\n{'=' * 74}\ncausality\n{'=' * 74}")
        causality()

    hl, dc = 3.0, 1.0  # fallback config if spotcheck is run standalone
    if choice in ("sweep", "all"):
        print(f"\n{'=' * 74}\nsweep\n{'=' * 74}")
        baseline, results = sweep()
        hl_g, dc_g = best_config(results)
        hl_d, dc_d = best_by_drawdown(results)
        print(f"\nbest by inner-validation log growth: H={hl_g:g}d cap={dc_g:g} "
              f"(logg={results[(hl_g, dc_g, 'inner-validation')]['log_growth']:+.4f})")
        print(f"best by inner-validation max drawdown: H={hl_d:g}d cap={dc_d:g} "
              f"(maxDD={results[(hl_d, dc_d, 'inner-validation')]['max_dd']:.2f}%)")
        if (hl_g, dc_g) != (hl_d, dc_d):
            print("these DISAGREE: the log-growth winner is not the drawdown winner.")
        hl, dc = hl_g, dc_g

    if choice in ("spotcheck", "all"):
        print(f"\n{'=' * 74}\nspotcheck (best config by inner-validation log growth)\n{'=' * 74}")
        spotcheck(hl, dc)


if __name__ == "__main__":
    main()
