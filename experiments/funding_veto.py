#!/usr/bin/env python
"""Funding veto: stand flat on futures while trailing funding is expensive.

Backlog B-05, ledger R-33 **variant A** (the conservative one; the novel
carry-adjusted continuous dampening, `carry_kelly`, is variant B, an
independent file per the pre-registration in ``docs/LEDGER.md``).

Not registered: this lives under ``experiments/``, so it is not
auto-discovered by ``tradebot.registry`` per ``docs/ROUTINE.md`` step 5,
and it gets none of ``tests/test_causality_strict.py``'s automatic
coverage — see ``causality()`` below for the by-hand probe that stands in
for it (same two-opposite-tampers procedure as R-28/R-31/R-33's other
experiments).

**Verdict: NEGATIVE (see docs/LEDGER.md, R-33 results).** Selected on
2022 inner-validation, where every swept config beat the ``kelly_regime_v4``
baseline on both return and drawdown. Frozen at H=3, W=60 and read once on
the 2023 holdout: it loses to ``kelly_regime_v4`` on every promotion axis
(ΔmaxDD 0.0pp, ΔSharpe -0.39) and loses outright to ``buy_and_hold``. A
rule that stands down while funding is expensive stands down during parts
of a rally too - crowded longs and a genuine bull can look identical from
the funding rate alone.

Mechanism, exactly as pre-registered (do not re-derive; the design was
frozen in the ledger before this file was written)
--------------------------------------------------------------------
``FundingVeto`` subclasses ``kelly_regime_v4`` unchanged — same
20/40/80-day anchor vote, same conditional-vol sizer, same 10% deadband —
and adds one thing: on the funding-paying market only, stand flat whenever
the trailing annualized funding rate exceeds its own **rolling** (never
expanding-over-the-whole-series) historical 90th percentile.

    funding_ann      = funding * 3 * 365.25                    (annualize)
    funding_on_bars  = funding_ann.reindex(df.index, ffill).fillna(0.0)
    funding_on_bars  = funding_on_bars.shift(1).fillna(0.0)     (causal margin)
    trailing         = funding_on_bars.ewm(span=H, min_periods=1).mean()
    thresh           = trailing.rolling(W, min_periods=30d).quantile(0.90)
    veto             = (trailing > thresh).fillna(False)
    futures_target   = v4_target * (0.0 if veto else 1.0)
    spot_target      = v4_target                                (unmodified)

``H`` (``funding_halflife_days``) in {1, 3, 7} and ``W``
(``quantile_window_days``) in {60, 120, 180} are swept; the quantile
itself (0.90) is fixed, per the pre-registration.

Why a rolling, not an expanding or full-series, quantile: an
expanding-from-the-start or full-series statistic applied to early rows
is the project's most repeated lookahead bug (``docs/RESEARCH.md``
"Methodology findings" #1 and #5). ``rolling(W)`` at row ``i`` only ever
touches rows ``[i-W+1, i]``, so it is causal by construction.

Usage::

    python experiments/funding_veto.py causality    # by-hand lookahead probe
    python experiments/funding_veto.py sweep         # step 3: the 3x3 H x W grid
    python experiments/funding_veto.py spot_check     # spot bit-identity invariant
    python experiments/funding_veto.py all
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import BacktestResult, run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Context  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

# Split, per the R-33 pre-registration (funding data only covers
# 2020-01-01 -> 2023-12-31, so the usual 2017-2020/2021-2022 inner split
# is shifted forward to sit inside the funding-covered range).
TRAIN = ("2020-01-01", "2021-12-31")
VALID = ("2022-01-01", "2022-12-31")

H_GRID = (1, 3, 7)      # funding_halflife_days
W_GRID = (60, 120, 180)  # quantile_window_days
N_EVALUATED = len(H_GRID) * len(W_GRID)  # 9; the v4 baseline is not a swept config


class FundingVeto(KellyRegimeV4):
    """``kelly_regime_v4``, standing flat on futures while funding is expensive.

    Adds one input (a funding-rate series) and one behavior: on the
    funding-paying market, exposure is zeroed for any bar whose trailing
    annualized funding rate sits above its own rolling 90th-percentile
    threshold. ``target`` itself (v4's sizing decision) is left untouched
    — the veto is applied only at order time, and only on futures — so
    this strategy is bit-identical to ``kelly_regime_v4`` on spot: see
    ``spot_check()`` below, a design invariant rather than a finding.
    """

    name = "funding_veto"

    def __init__(
        self,
        funding: pd.Series | None = None,
        funding_halflife_days: float = 3.0,
        quantile_window_days: float = 120.0,
        funding_quantile: float = 0.90,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.funding = funding
        self.funding_halflife_days = funding_halflife_days
        self.quantile_window_days = quantile_window_days
        self.funding_quantile = funding_quantile

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4's target column, unmodified from here on

        if self.funding is None or len(self.funding) == 0:
            df["veto"] = False
            return df

        funding_ann = self.funding * 3 * 365.25  # 3 settlements/day -> annualized
        funding_on_bars = funding_ann.reindex(df.index, method="ffill")
        funding_on_bars = funding_on_bars.fillna(0.0)  # before the first real settlement
        funding_on_bars = funding_on_bars.shift(1).fillna(0.0)  # extra causal margin

        span = self.funding_halflife_days * BARS_PER_DAY
        trailing = funding_on_bars.ewm(span=span, min_periods=1).mean()

        window = int(round(self.quantile_window_days * BARS_PER_DAY))
        min_periods = 30 * BARS_PER_DAY
        thresh = trailing.rolling(window, min_periods=min_periods).quantile(
            self.funding_quantile
        )

        veto = (trailing > thresh).fillna(False)
        df["veto"] = veto.to_numpy()
        return df

    def on_bar(self, ctx: Context) -> None:
        def effective(row) -> float:
            """v4's target, zeroed by the veto on futures only."""
            raw = float(row["target"])
            if ctx.market.pays_funding and bool(row["veto"]):
                return 0.0
            return raw

        t = effective(ctx.bar)
        prev_t = effective(ctx.prev) if ctx.prev is not None else 0.0
        if abs(t - prev_t) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures


# --------------------------------------------------------------------- helpers


def _period(strategy, df: pd.DataFrame, market: MarketSpec, start: str, end: str,
            funding: pd.Series | None = None, start_balance: float = 1_000.0) -> BacktestResult:
    """Backtest over ``df[start:end]``, warmed on the bars before it.

    Same pattern as ``scripts/funding_study.py``'s ``_period``: bars before
    the window feed the strategy's indicators (``on_bar`` runs, so state
    warms normally) but the account stays flat at ``start_balance`` until
    the window's first bar, via ``trade_start`` — this avoids the
    warmup-prefix bias ``tradebot.window.run_period`` fixes generally
    (``docs/RESEARCH.md`` finding 8), reimplemented by hand here only
    because ``run_period`` does not accept a ``funding`` argument.
    """
    lo = int(df.index.searchsorted(start))
    hi = int(df.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    frame = df.iloc[lo - pre: hi]
    raw = run_backtest(strategy, frame, market, start_balance, trade_start=pre,
                       funding=funding, data_label=LABEL)
    if pre == 0:
        return raw
    return replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])


def _row(config: str, split: str, result: BacktestResult) -> dict:
    m = compute_metrics(result)
    veto_pct = (100.0 * float(result.df["veto"].mean())
                if "veto" in result.df.columns else float("nan"))
    return {
        "config": config,
        "split": split,
        "final_balance": m.final_balance,
        "log_growth": float(np.log(m.final_balance / result.start_balance)),
        "max_dd_pct": m.max_drawdown_pct,
        "sharpe": m.sharpe,
        "trades": m.num_trades,
        "veto_pct": veto_pct,
        "funding_paid": result.funding_paid,
    }


def _print_table(rows: list[dict]) -> None:
    hdr = (f"{'config':16s} {'split':17s} {'final $':>10s} {'log growth':>10s} "
           f"{'max DD%':>8s} {'sharpe':>7s} {'trades':>7s} {'veto %':>7s} {'funding $':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        veto = f"{r['veto_pct']:6.1f}%" if r["veto_pct"] == r["veto_pct"] else "   n/a"
        print(f"{r['config']:16s} {r['split']:17s} {r['final_balance']:>10,.0f} "
              f"{r['log_growth']:>10.4f} {r['max_dd_pct']:>7.1f}% {r['sharpe']:>7.2f} "
              f"{r['trades']:>7d} {veto:>7s} {r['funding_paid']:>10,.0f}")


# ------------------------------------------------------------------- causality


def causality() -> None:
    """The strict lookahead probe, by hand — this experiment gets no CI protection.

    Two-opposite-tampers, same procedure as R-28/R-31: bars after a cut
    are multiplied by 3 in one copy of the price data and divided by 3 in
    the other; every decision at or before the cut must be identical.
    Run once against price tampering (checks the inherited v4 ``target``
    path and, incidentally, ``veto`` since it shares ``prepare()``), and
    once against funding tampering (checks the veto's actual input
    directly — ``veto`` does not read ``df`` at all, so this is the more
    informative of the two for this strategy specifically).
    """
    df = DF.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]
    funding_slice = REAL[(REAL.index >= df.index[0]) & (REAL.index <= df.index[-1])]

    def decisions(frame, funding):
        s = FundingVeto(funding=funding, funding_halflife_days=3, quantile_window_days=120)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return prepared, out

    def probe(label, up, down, funding_up, funding_down):
        prep_up, dec_up = decisions(up, funding_up)
        prep_down, dec_down = decisions(down, funding_down)
        bad = [b for b, oa, ob in zip(bars, dec_up, dec_down) if oa != ob]
        worst = max(
            float(np.nanmax(np.abs(
                prep_up[c].to_numpy()[:cut].astype(float)
                - prep_down[c].to_numpy()[:cut].astype(float))))
            for c in ("target", "veto")
        )
        good = not bad and worst < 1e-12
        print(f"  [{label:8s}] orders {'match' if not bad else f'DIFFER at {bad}'}   "
              f"max |target/veto diff| before the cut = {worst:.3e}   "
              f"{'PASS' if good else 'FAIL'}")
        return good

    print(f"tampered from bar {cut:,} of {len(df):,} "
          f"({df.index[cut]:%Y-%m-%d %H:%M} UTC)\n")

    up_px, down_px = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up_px.iloc[cut:, up_px.columns.get_loc(col)] *= 3.0
        down_px.iloc[cut:, down_px.columns.get_loc(col)] /= 3.0
    up_px.iloc[cut:, up_px.columns.get_loc("volume")] *= 7.0
    down_px.iloc[cut:, down_px.columns.get_loc("volume")] /= 7.0
    ok_price = probe("price", up_px, down_px, funding_slice, funding_slice)

    ts_cut = df.index[cut]
    up_f = funding_slice.copy()
    down_f = funding_slice.copy()
    mask = up_f.index >= ts_cut
    up_f.loc[mask] = up_f.loc[mask] * 3.0 + 0.01
    down_f.loc[mask] = down_f.loc[mask] / 3.0 - 0.01
    ok_funding = probe("funding", df, df, up_f, down_f)

    ok = ok_price and ok_funding
    print(f"\n{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# ------------------------------------------------------------------------ step 3


def sweep() -> list[dict]:
    """Step 3: the 3x3 H x W grid on inner-train and inner-validation, futures + real funding."""
    if REAL is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")

    rows: list[dict] = []
    for split, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        base = get_strategy("kelly_regime_v4")
        r = _period(base, DF, FUTURES, start, end, funding=REAL)
        rows.append(_row("v4 baseline", split, r))

        for h in H_GRID:
            for w in W_GRID:
                s = FundingVeto(funding=REAL, funding_halflife_days=h, quantile_window_days=w)
                r = _period(s, DF, FUTURES, start, end, funding=REAL)
                rows.append(_row(f"H={h} W={w}", split, r))

    _print_table(rows)
    print(f"\n{N_EVALUATED} configurations evaluated in the sweep "
          f"({len(H_GRID)} halflives x {len(W_GRID)} windows), "
          f"each on {len(('inner-train', 'inner-validation'))} splits, "
          f"plus the v4 baseline on the same splits for comparison.")
    return rows


# -------------------------------------------------------------------- spot check


def spot_check(funding_halflife_days: float, quantile_window_days: float) -> float:
    """Design invariant: FundingVeto must be bit-identical to v4 on spot.

    Restricted to 2020-01-01 -> 2022-12-31 (inner-train + inner-validation)
    rather than the full 2020-2023 funding-covered range the mechanism
    note sketches, specifically so this file never evaluates a backtest
    window touching 2023 — the equality check itself carries no
    performance information (it is 0.0 by construction, regardless of
    which years it spans), but the window it runs over does not need to
    include the holdout to prove the invariant, so it doesn't.
    """
    veto_strat = FundingVeto(funding=REAL, funding_halflife_days=funding_halflife_days,
                             quantile_window_days=quantile_window_days)
    v4_strat = get_strategy("kelly_regime_v4")

    start, end = "2020-01-01", "2022-12-31"
    r_veto = _period(veto_strat, DF, SPOT, start, end, funding=REAL)
    r_v4 = _period(v4_strat, DF, SPOT, start, end)

    a = r_veto.equity.to_numpy(dtype=float)
    b = r_v4.equity.to_numpy(dtype=float)
    assert len(a) == len(b) and r_veto.equity.index.equals(r_v4.equity.index)
    max_diff = float(np.max(np.abs(a - b)))
    print(f"spot equity curves, {start} -> {end}, {len(a):,} bars: "
          f"max |funding_veto - kelly_regime_v4| = {max_diff:.10e}   "
          f"{'PASS (bit-identical)' if max_diff == 0.0 else 'FAIL'}")
    return max_diff


COMMANDS = {
    "causality": causality,
    "sweep": sweep,
    # H=3, W=60: best inner-validation log growth in the sweep (see sweep()).
    "spot_check": lambda: spot_check(3, 60),
}


def main() -> None:
    if REAL is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "all":
        for name, fn in COMMANDS.items():
            print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
            fn()
    elif choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python experiments/funding_veto.py [{'|'.join(COMMANDS)}|all]")


if __name__ == "__main__":
    main()
