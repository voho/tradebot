"""Delta-neutral crypto cash-and-carry: long spot BTC / short perp, harvest funding.

R-39 **novel** branch, backlog item **B-03**, 2026-08-19. Unregistered
experiment: it lives under ``experiments/`` so it is NOT auto-discovered
(docs/ROUTINE.md step 5 / ``tradebot.registry``'s package scan). Nothing
here is decorated with ``@register``, and this session commits nothing.

Pre-registration: ``docs/LEDGER.md``, "### R-39 pre-registration", the
bullet headed *Novel — ``experiments/funding_harvest_carry.py``*. Read
that before changing anything in this file.

Idea, in one sentence
---------------------
Hold spot BTC, short an equal-notional BTC perpetual against it, stay
delta-neutral, and collect the funding stream that every other strategy
in this repository *pays* — the other side of the trade R-14/L-05 found
costs ``kelly_regime_v4`` ~20%/yr while it holds.

Why now, and what is actually new
---------------------------------
``docs/VALIDATION.md`` ("The other side of the trade: harvesting the
premium (measured, 2020-2023)") already reports +82.0% gross over 4.0
years = +16.2%/yr, +14.6%/yr after a 0.10% taker on both legs with a
quarterly rebalance, +9.8%/yr at 0.40%, 13.5% of settlements negative,
worst 30-day run -1.31%. That was a **one-off compounding calculation**
over the committed Binance series, not a backtest, and no code for it
existed in this repository. Two things are new here:

1. it is real, re-runnable code with an explicit position/fee/margin
   ledger, so the fee and rebalance conventions behind those numbers are
   visible rather than implicit (and the VALIDATION cells are reproduced
   exactly as a self-check — see ``check_validation_parity``); and
2. it is extended through **2024-2026** with ``load_funding_extended``,
   i.e. the exact window the literature (He, Manela, Ross & von Wachter
   2024, SSRN 4301150; BitMEX 2025Q3/2026Q2 derivatives reports) says the
   carry trade crowded and its premium compressed. The pre-registered
   decision rule is stated against **2024-2026 specifically**: re-quoting
   2020-2023 is not a holdout test of anything.

What the model does, precisely
------------------------------
The account holds ``qty`` BTC of spot long and ``qty`` BTC of perp short
(equal notional at every rebalance, hence delta-neutral in BTC terms at
all times, not merely on average). State is carried in BTC units, not in
a return series, so the notional-vs-equity drift between rebalances is a
real modelled quantity rather than an assumption:

* at each 8h settlement ``t``: ``equity += rate[t] * qty * price[t]`` —
  the short leg *receives* funding when the rate is positive (longs pay
  shorts) and *pays* when it flips negative. Funding accrues on the
  **marked** notional ``qty * price[t]``, which is what an exchange
  actually charges, not on a fixed notional;
* at each rebalance date: reset ``qty`` so that ``qty * price = gearing *
  equity`` again, paying taker fees (below). Between rebalances the
  notional floats with price, so a bull quarter silently levers the
  position up and a bear quarter de-levers it. That drift is reported
  (``max_notional_over_equity``) because it is the margin risk, and it is
  the thing a pure "compound the rate series" calculation cannot see;
* price legs cancel **exactly**: spot P&L ``qty*(P-P0)`` and short-perp
  P&L ``-qty*(P-P0)`` sum to zero. Equity therefore moves only via
  funding and fees. That is not a modelling shortcut — it is forced by
  the data limitation named below, and it is the single most important
  caveat on every number this file prints.

Fee conventions (two, both reported)
------------------------------------
``roundtrip`` — at every rebalance both legs are fully closed and fully
reopened: ``fee * (notional_old * 2 legs + notional_new * 2 legs)``, plus
an initial entry (2 legs) and a final exit (2 legs). This is the
convention that reproduces VALIDATION.md's own arithmetic (quarterly ->
4/yr x 2 legs x 2 sides x 0.10% = 1.6%/yr, and 16.2 - 1.6 = 14.6). It is
deliberately pessimistic: nobody actually round-trips the whole book to
re-gear it.

``drift`` — only the *change* in notional is traded at a rebalance:
``fee * |notional_new - notional_old| * 2 legs``, still with a full entry
and exit. This is what a desk would really do. Both are reported; the
``roundtrip`` figure is the headline, for comparability.

THE LIMITATION THAT MATTERS MOST: there is no perp price series
---------------------------------------------------------------
This repository has **no separate perpetual price series** — the futures
market runs on the spot series and every report is labeled
``spot (perp proxy)`` (``tradebot.data.load_dataset``). For a
delta-neutral carry trade that is not a cosmetic caveat, it is a hole in
the middle of the trade: with spot and perp modelled by the *same* price
series, **the basis is identically zero by construction**, so this
backtest structurally CANNOT measure basis risk at entry or exit — one of
the two or three risks that actually decide whether this trade makes
money. The standing rule ("never proxy unavailable data out of price",
docs/ROUTINE.md) forbids manufacturing a basis out of the price series,
so none is manufactured. Every Sharpe below is therefore an **upper
bound** on a quantity whose main risk term is missing. See the
"unmodelled risks" section of the report for the rest.

Cross-venue caveat on the 2024-2026 half
----------------------------------------
``load_funding_extended`` concatenates real Binance funding (2020-2023)
with Deribit funding for the post-2023 gap only, tagged per settlement.
Deribit charges funding *continuously* (hourly ``interest_1h``, summed
here into 8h buckets) and is a different instrument from Binance's
discrete 8-hourly settlement; on the 2020-2023 overlap the two correlate
at r=0.69 but their level ratio is unstable year to year (0.21x-1.24x),
which is exactly why they are not rescaled onto a common level. Any
Binance-2020-23 vs Deribit-2024-26 comparison therefore mixes a real
change in the market with a venue change. This file answers that by
running the **same analysis a second time on the Deribit series alone**
across 2020-2026 (``deribit-only`` rows), which is venue-consistent and
is the comparison the decline claim should actually rest on.

Usage
-----
    python experiments/funding_harvest_carry.py            # everything
    python experiments/funding_harvest_carry.py parity     # VALIDATION check only
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import (  # noqa: E402
    load_dataset,
    load_funding,
    load_funding_deribit,
    load_funding_extended,
)
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import prefix_bars  # noqa: E402

DATA = ROOT / "data"
SETTLEMENTS_PER_YEAR = 3 * 365.25
DAYS_PER_YEAR = 365.25

# Sub-periods. The split is the venue boundary, not a chosen date: the
# committed Binance series ends 2023-12-31, everything after is Deribit.
BINANCE_ERA = ("2020-01-01", "2023-12-31")
DERIBIT_ERA = ("2024-01-01", "2026-12-31")

# every configuration evaluated is appended here, for the deflated-Sharpe
# trials count the routine requires (docs/ROUTINE.md step 3)
CONFIGS: list[str] = []


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_price_at_settlements(index: pd.DatetimeIndex) -> pd.Series:
    """Spot close as of each settlement, causally (last bar at or before it).

    Uses ``searchsorted``-style ffill reindexing: a settlement at 03:00
    sees the 02:55 bar close, never a later one. Settlements after the
    end of the price series are dropped rather than forward-filled, so
    the last real close can never be stretched into an uncovered future
    (the same cutoff discipline ``funding_gate_decile.py`` applies to the
    funding series).
    """
    df, label = load_dataset(DATA, "spot")
    close = df["close"]
    aligned = close.reindex(close.index.union(index)).sort_index().ffill().reindex(index)
    aligned = aligned.where(index <= close.index.max())
    return aligned.dropna(), label


@dataclass
class Series:
    """One funding series to run the whole analysis over."""
    name: str
    rate: pd.Series
    source: pd.Series
    price: pd.Series = field(default=None)
    price_label: str = ""


def build_series() -> dict[str, Series]:
    ext, src = load_funding_extended(DATA)
    der = load_funding_deribit(DATA)
    if ext is None or der is None:
        raise SystemExit("funding data missing; see docs/VALIDATION.md")
    out = {
        # primary: real Binance 2020-2023 + Deribit 2024-2026, source-tagged
        "extended": Series("extended", ext, src),
        # venue-consistent control: Deribit alone, 2020-2026
        "deribit-only": Series("deribit-only", der, pd.Series("deribit", index=der.index)),
    }
    for s in out.values():
        price, label = load_price_at_settlements(s.rate.index)
        s.rate = s.rate.reindex(price.index)
        s.source = s.source.reindex(price.index)
        s.price = price
        s.price_label = label
    return out


# --------------------------------------------------------------------------
# the carry backtest
# --------------------------------------------------------------------------

@dataclass
class CarryResult:
    equity: pd.Series          # account equity at each settlement
    funding: pd.Series         # funding received (+) / paid (-) per settlement, USD
    notional: pd.Series        # marked notional of ONE leg at each settlement
    fees_paid: float
    n_rebalances: int
    max_notional_over_equity: float
    max_short_leg_loss_over_equity: float   # peak margin need on the short leg
    start_equity: float
    config: str


def _rebalance_flags(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """True at the first settlement of each new period. ``freq='S'`` = every
    settlement (the continuously-rebalanced textbook case)."""
    if freq == "S":
        return np.ones(len(index), dtype=bool)
    naive = index.tz_localize(None)
    key = {"M": naive.to_period("M"), "Q": naive.to_period("Q"),
           "A": naive.to_period("Y")}[freq]
    flags = np.zeros(len(index), dtype=bool)
    prev = None
    for i, k in enumerate(key):
        if k != prev:
            flags[i] = True
            prev = k
    flags[0] = False          # the first settlement is the ENTRY, not a rebalance
    return flags


def carry_backtest(rate: pd.Series, price: pd.Series, *, fee: float = 0.0,
                   rebalance: str = "Q", fee_model: str = "roundtrip",
                   gearing: float = 1.0, start_equity: float = 1_000.0,
                   count: bool = True) -> CarryResult:
    """Long ``qty`` spot BTC / short ``qty`` BTC perp, delta-neutral, harvest funding.

    ``fee`` is the taker rate as a decimal (0.001 = 0.10%), charged on the
    traded notional of EACH leg. ``rebalance`` in {'S','M','Q','A'}.
    ``fee_model`` in {'roundtrip','drift'} — see the module docstring.
    """
    # A "configuration" is the trade specification, not the window it is
    # read over: the same spec evaluated on 2020-23 and on 2024-26 is one
    # config seen twice, not two configs. Both counts are reported.
    cfg = (f"fee={fee:.4f} rebalance={rebalance} "
           f"fee_model={fee_model} gearing={gearing:g}")
    if count:
        CONFIGS.append(cfg)

    idx = rate.index
    r = rate.to_numpy(dtype=float)
    p = price.to_numpy(dtype=float)
    reb = _rebalance_flags(idx, rebalance)

    equity = np.empty(len(idx))
    fundings = np.zeros(len(idx))
    notionals = np.empty(len(idx))

    eq = start_equity
    # ENTRY at the first settlement: open both legs.
    notional = gearing * eq
    qty = notional / p[0]
    entry_fee = fee * notional * 2.0            # 2 legs
    eq -= entry_fee
    fees = entry_fee
    n_reb = 0
    ref_price = p[0]                            # price at the last (re)opening
    max_lev = 0.0
    max_margin = 0.0

    for i in range(len(idx)):
        if i > 0 and reb[i]:
            old_notional = qty * p[i]
            new_notional = gearing * eq
            if fee_model == "roundtrip":
                traded = (old_notional + new_notional)
            else:                                # 'drift': trade only the delta
                traded = abs(new_notional - old_notional)
            f = fee * traded * 2.0               # 2 legs
            eq -= f
            fees += f
            qty = (gearing * eq) / p[i]
            ref_price = p[i]
            n_reb += 1

        marked = qty * p[i]
        notionals[i] = marked
        # the short perp's unrealized loss since the last (re)opening is the
        # margin the account must be able to post; the spot leg's matching
        # gain does not help unless the venue cross-margins it.
        margin_need = qty * (p[i] - ref_price)
        max_margin = max(max_margin, margin_need / max(eq, 1e-9))
        max_lev = max(max_lev, marked / max(eq, 1e-9))

        pay = r[i] * marked                      # short receives when rate > 0
        eq += pay
        fundings[i] = pay
        equity[i] = eq

    # EXIT: close both legs at the last price.
    exit_fee = fee * qty * p[-1] * 2.0
    eq -= exit_fee
    fees += exit_fee
    equity[-1] = eq

    return CarryResult(
        equity=pd.Series(equity, index=idx),
        funding=pd.Series(fundings, index=idx),
        notional=pd.Series(notionals, index=idx),
        fees_paid=fees,
        n_rebalances=n_reb,
        max_notional_over_equity=max_lev,
        max_short_leg_loss_over_equity=max_margin,
        start_equity=start_equity,
        config=cfg,
    )


# --------------------------------------------------------------------------
# metrics on a carry equity curve
# --------------------------------------------------------------------------

@dataclass
class CarryMetrics:
    years: float
    total_pct: float
    annualized_pct: float
    sharpe_daily: float
    worst_30d_pct: float
    max_drawdown_pct: float
    neg_settlement_frac: float
    n_settlements: int


def _daily_equity(equity: pd.Series) -> pd.Series:
    return equity.resample("1D").last().dropna()


def carry_metrics(res: CarryResult, rate: pd.Series) -> CarryMetrics:
    eq = res.equity
    years = (eq.index[-1] - eq.index[0]).days / DAYS_PER_YEAR
    # eq.iloc[0] already has the entry fee inside it; measure against the
    # true starting capital instead so the entry cost is not free.
    total = eq.iloc[-1] / res.start_equity - 1.0
    ann = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else float("nan")

    daily = _daily_equity(eq)
    dret = daily.pct_change().dropna()
    sd = dret.std(ddof=1)
    sharpe = float(dret.mean() / sd * np.sqrt(DAYS_PER_YEAR)) if sd > 0 else float("nan")

    roll = daily.rolling("30D").apply(lambda x: x[-1] / x[0] - 1.0, raw=True)
    worst30 = float(roll.min())

    peak = daily.cummax()
    dd = float(((daily - peak) / peak).min())

    return CarryMetrics(
        years=years,
        total_pct=100 * total,
        annualized_pct=100 * ann,
        sharpe_daily=sharpe,
        worst_30d_pct=100 * worst30,
        max_drawdown_pct=100 * dd,
        neg_settlement_frac=float((rate < 0).mean()),
        n_settlements=len(rate),
    )


def slice_period(s: Series, lo: str, hi: str) -> Series:
    m = (s.rate.index >= lo) & (s.rate.index < pd.Timestamp(hi, tz="UTC") + pd.Timedelta(days=1))
    out = Series(s.name, s.rate[m], s.source[m])
    out.price = s.price[m]
    out.price_label = s.price_label
    return out


def run_cell(s: Series, *, fee: float, rebalance: str, fee_model: str = "roundtrip",
             count: bool = True) -> tuple[CarryMetrics, CarryResult]:
    res = carry_backtest(s.rate, s.price, fee=fee, rebalance=rebalance,
                         fee_model=fee_model, count=count)
    return carry_metrics(res, s.rate), res


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------

def _hdr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def check_validation_parity(series: dict[str, Series]) -> None:
    """Reproduce docs/VALIDATION.md's five 2020-2023 cells from this code."""
    _hdr("0. parity check against docs/VALIDATION.md (2020-2023, real Binance)")
    binance = load_funding(DATA)
    gross = float(np.prod(1.0 + binance.to_numpy()) - 1.0)
    yrs = (binance.index[-1] - binance.index[0]).days / DAYS_PER_YEAR
    print(f"  raw compounding of the committed Binance series ({len(binance)} settlements):")
    print(f"    gross               {100 * gross:+.1f}% over {yrs:.1f} years "
          f"= {100 * ((1 + gross) ** (1 / yrs) - 1):+.2f}%/yr   "
          f"[VALIDATION.md: +82.0% / +16.2%/yr]")
    print(f"    negative settlements {(binance < 0).mean():.1%}"
          f"                                    [VALIDATION.md: 13.5%]")
    eq = (1 + binance).cumprod()
    roll = eq.rolling("30D").apply(lambda x: x[-1] / x[0] - 1.0, raw=True)
    print(f"    worst 30-day run    {100 * roll.min():+.2f}%"
          f"                                     [VALIDATION.md: -1.31%]")
    print("\n  VALIDATION.md's fee arithmetic, made explicit (it is a LINEAR")
    print("  subtraction from the gross annualized figure, not a compounded one):")
    for tier, label in ((0.001, "0.10%"), (0.004, "0.40%")):
        cost = 4 * 2 * 2 * tier      # 4 rebalances/yr x 2 legs x 2 sides
        print(f"    {label} taker: 4/yr x 2 legs x 2 sides = {100 * cost:.1f}%/yr  ->  "
              f"16.2 - {100 * cost:.1f} = {16.15 - 100 * cost:+.2f}%/yr   "
              f"[VALIDATION.md: {'+14.6' if tier == 0.001 else '+9.8'}%/yr]")
    print("\n  This file's own position-level backtest of the same window:")
    ext = slice_period(series["extended"], *BINANCE_ERA)
    m, res = run_cell(ext, fee=0.0, rebalance="S")
    print(f"    continuous rebalance, no fees   {m.annualized_pct:+7.2f}%/yr   "
          f"total {m.total_pct:+7.1f}%   maxLev {res.max_notional_over_equity:.2f}x")
    print("      ^ notional is reset to equity at EVERY settlement, so this is the")
    print("        exact like-for-like of VALIDATION.md's compounding calculation.")
    m, res = run_cell(ext, fee=0.0, rebalance="Q")
    print(f"    quarterly rebalance,  no fees   {m.annualized_pct:+7.2f}%/yr   "
          f"total {m.total_pct:+7.1f}%   maxLev {res.max_notional_over_equity:.2f}x")
    print("      ^ HIGHER, and not because the trade is better: between quarterly")
    print("        rebalances the notional floats with price, so the 2020-21 bull")
    print("        run levered the position up and it collected funding on a bigger")
    print("        book. That is an exposure difference, not an edge (docs/ROUTINE.md,")
    print("        'match risk before comparing anything'), and it is why the")
    print("        continuous row above is the one that is comparable to")
    print("        VALIDATION.md and the quarterly rows below carry a maxLev column.")


ARMS = (
    ("gross (cont.)", 0.0, "S", "roundtrip"),
    ("gross (qtrly)", 0.0, "Q", "roundtrip"),
    ("0.10% qtrly", 0.001, "Q", "roundtrip"),
    ("0.40% qtrly", 0.004, "Q", "roundtrip"),
)


def _arm_table(rows: list[tuple[str, Series]]) -> None:
    print(f"\n  {'period':26s} {'arm':14s} {'ann':>9s} {'total':>9s} {'sharpe':>7s} "
          f"{'worst30d':>9s} {'maxDD':>7s} {'neg':>6s} {'maxLev':>7s} {'n':>6s}")
    for label, s in rows:
        for alabel, tier, freq, fm in ARMS:
            m, res = run_cell(s, fee=tier, rebalance=freq, fee_model=fm)
            print(f"  {label:26s} {alabel:14s} {m.annualized_pct:>+8.2f}% "
                  f"{m.total_pct:>+8.1f}% {m.sharpe_daily:>7.2f} "
                  f"{m.worst_30d_pct:>+8.2f}% {m.max_drawdown_pct:>+6.2f}% "
                  f"{m.neg_settlement_frac:>5.1%} "
                  f"{res.max_notional_over_equity:>6.2f}x {m.n_settlements:>6d}")
        print()


def headline_table(series: dict[str, Series]) -> None:
    _hdr("1-5. the pre-registered cells: gross / 0.10% / 0.40%, by sub-period")
    ext = series["extended"]
    _arm_table([
        ("full 2020-2026", ext),
        ("2020-2023 (real Binance)", slice_period(ext, *BINANCE_ERA)),
        ("2024-2026 (Deribit ext.)", slice_period(ext, *DERIBIT_ERA)),
    ])
    print("  'gross (cont.)' resets notional to equity at every settlement and is")
    print("  the risk-matched read; 'gross (qtrly)' lets notional float between")
    print("  quarterly rebalances, so its extra return is leverage drift (maxLev),")
    print("  not edge. The two fee rows share the quarterly convention.")

    _hdr("1-5b. venue-consistent control: the SAME cells on Deribit alone, 2020-2026")
    der = series["deribit-only"]
    _arm_table([
        ("deribit 2020-2023", slice_period(der, *BINANCE_ERA)),
        ("deribit 2024-2026", slice_period(der, *DERIBIT_ERA)),
    ])
    print("  This is the comparison the 'the premium declined' claim should rest on:")
    print("  one venue, one settlement convention, both halves of the timeline.")


def per_year(series: dict[str, Series]) -> None:
    _hdr("6. annualized carry per calendar year - the shape of the decline")
    for name in ("extended", "deribit-only"):
        s = series[name]
        print(f"\n  {name}:")
        print(f"    {'year':6s} {'src':8s} {'gross(cont)':>12s} {'gross(qtr)':>11s} "
              f"{'0.10% qtr':>10s} {'0.40% qtr':>10s} {'sharpe':>7s} {'neg':>6s} "
              f"{'worst30d':>9s} {'n':>5s}")
        for year, grp in s.rate.groupby(s.rate.index.year):
            sub = slice_period(s, f"{year}-01-01", f"{year}-12-31")
            src = "/".join(sorted(set(sub.source.astype(str))))
            row = []
            for tier, freq in ((0.0, "S"), (0.0, "Q"), (0.001, "Q"), (0.004, "Q")):
                m, _ = run_cell(sub, fee=tier, rebalance=freq, count=False)
                row.append(m)
            print(f"    {year:<6d} {src:8s} {row[0].annualized_pct:>+11.2f}% "
                  f"{row[1].annualized_pct:>+10.2f}% {row[2].annualized_pct:>+9.2f}% "
                  f"{row[3].annualized_pct:>+9.2f}% "
                  f"{row[0].sharpe_daily:>7.2f} {row[0].neg_settlement_frac:>5.1%} "
                  f"{row[0].worst_30d_pct:>+8.2f}% {row[0].n_settlements:>5d}")
    print("\n  (per-year cells are DESCRIPTIVE slices of configurations already")
    print("   counted above, not new configurations - see the configs count.)")


def sharpe_vs_literature(series: dict[str, Series]) -> None:
    _hdr("7. carry Sharpe vs the literature's 6.45 -> 4.06 -> negative")
    ext = series["extended"]
    print("\n  literature (empirical work cited in docs/VALIDATION.md):")
    print("    ~6.45 over the 2020-2023 era, 4.06 from 2024, negative in 2025.\n")
    print(f"  {'window':28s} {'gross Sharpe':>13s} {'0.10% Sharpe':>13s} "
          f"{'0.40% Sharpe':>13s}")
    windows = [
        ("2020-2023 (Binance)", slice_period(ext, *BINANCE_ERA)),
        ("2024-2026 (Deribit)", slice_period(ext, *DERIBIT_ERA)),
        ("2024 only", slice_period(ext, "2024-01-01", "2024-12-31")),
        ("2025 only", slice_period(ext, "2025-01-01", "2025-12-31")),
        ("2026 YTD", slice_period(ext, "2026-01-01", "2026-12-31")),
    ]
    for label, s in windows:
        vals = []
        for tier, freq in ((0.0, "S"), (0.001, "Q"), (0.004, "Q")):
            m, _ = run_cell(s, fee=tier, rebalance=freq, count=False)
            vals.append(m.sharpe_daily)
        print(f"  {label:28s} {vals[0]:>13.2f} {vals[1]:>13.2f} {vals[2]:>13.2f}")
    print("\n  NOTE: a Sharpe on a stream with no basis risk in it is an upper")
    print("  bound, not a measurement. See the unmodelled-risk section.")


def rebalance_sensitivity(series: dict[str, Series]) -> None:
    _hdr("8. rebalance-frequency sensitivity (ROBUSTNESS, not a knob to tune)")
    ext = series["extended"]
    for label, s in (("2020-2023 (Binance)", slice_period(ext, *BINANCE_ERA)),
                     ("2024-2026 (Deribit)", slice_period(ext, *DERIBIT_ERA))):
        print(f"\n  {label}:")
        print(f"    {'rebalance':12s} {'fee model':10s} {'gross':>9s} {'0.10%':>9s} "
              f"{'0.40%':>9s} {'n_reb':>6s} {'maxLev':>7s} {'maxMargin':>10s}")
        for freq, fname in (("M", "monthly"), ("Q", "quarterly"), ("A", "annual")):
            for fm in ("roundtrip", "drift"):
                row, res = [], None
                for tier in (0.0, 0.001, 0.004):
                    m, res = run_cell(s, fee=tier, rebalance=freq, fee_model=fm)
                    row.append(m.annualized_pct)
                print(f"    {fname:12s} {fm:10s} {row[0]:>+8.2f}% {row[1]:>+8.2f}% "
                      f"{row[2]:>+8.2f}% {res.n_rebalances:>6d} "
                      f"{res.max_notional_over_equity:>6.2f}x "
                      f"{res.max_short_leg_loss_over_equity:>9.2f}x")
    print("\n  maxLev  = peak (one-leg notional / equity) reached between rebalances.")
    print("  maxMargin = peak unrealized loss on the SHORT leg since its last")
    print("  (re)opening, as a multiple of account equity: the margin the account")
    print("  must be able to post, and the number that decides whether it survives.")


def benchmarks() -> None:
    _hdr("9. risk profile vs buy_and_hold and kelly_regime_v4, 2024-01-01 -> 2026")
    df, label = load_dataset(DATA, "spot")
    rate, _ = load_funding_extended(DATA)
    spot, futures = MarketSpec.spot(), MarketSpec.futures(leverage=5.0)
    print(f"\n  {'arm':38s} {'final':>11s} {'total':>9s} {'maxDD':>8s} {'sharpe':>7s}")

    def run(name: str, market: MarketSpec, funding=None, tag=""):
        strat = get_strategy(name)
        lo = int(df.index.searchsorted("2024-01-01"))
        hi = len(df)
        pre = prefix_bars(df, lo, strat.warmup)
        res = run_backtest(strat, df.iloc[lo - pre:hi], market, 1_000.0,
                           trade_start=pre, funding=funding, data_label=label)
        from dataclasses import replace as _replace
        res = res if pre == 0 else _replace(res, equity=res.equity.iloc[pre:],
                                            df=res.df.iloc[pre:])
        m = compute_metrics(res)
        print(f"  {tag or name:38s} ${m.final_balance:>10,.0f} "
              f"{m.profit_pct:>+8.1f}% {m.max_drawdown_pct:>7.1f}% {m.sharpe:>7.2f}")
        return m

    run("buy_and_hold", spot, tag="buy_and_hold (spot 1x)")
    run("kelly_regime_v4", futures, tag="kelly_regime_v4 (5x, NO funding)")
    run("kelly_regime_v4", futures, funding=rate, tag="kelly_regime_v4 (5x, funding charged)")

    ext = build_series()["extended"]
    s = slice_period(ext, *DERIBIT_ERA)
    for tier, tlabel in ((0.0, "gross"), (0.001, "0.10%"), (0.004, "0.40%")):
        m, res = run_cell(s, fee=tier, rebalance="Q", count=False)
        eq = res.equity
        daily = _daily_equity(eq)
        print(f"  {'carry (quarterly, ' + tlabel + ')':38s} "
              f"${eq.iloc[-1]:>10,.0f} {m.total_pct:>+8.1f}% "
              f"{m.max_drawdown_pct:>7.2f}% {m.sharpe_daily:>7.2f}")
    print("\n  NOTE the arms carry very different risk. The carry arm's drawdown is")
    print("  tiny BECAUSE its only modelled risk is the funding rate flipping sign;")
    print("  the two risks that actually kill this trade (basis at entry/exit,")
    print("  liquidation on the short leg) are not in the number. This table is a")
    print("  return comparison, NOT a matched-risk comparison, and the standing")
    print("  rule (docs/ROUTINE.md, 'match risk before comparing anything') says a")
    print("  comparison whose arms carry different realized volatility is a")
    print("  statement about the exposures. It is printed for the pre-registered")
    print("  return bar only.")


def cash_benchmark(series: dict[str, Series]) -> None:
    _hdr("10. the comparison that actually decides it: carry vs holding cash")
    print("""
  The carry trade's capital sits as collateral. This backtest credits it
  ZERO interest, which was harmless when the premium was 16%/yr and is
  decisive when it is not: US T-bills paid roughly 5.3% (2024), 4.3%
  (2025) and ~3.7% (2026 YTD). Those are stated here as context, NOT
  fetched or modelled - this repository has no rates data, and per the
  standing rule none is proxied out of price. Read the comparison as
  indicative.
""")
    ext = series["extended"]
    approx_rf = {2024: 5.3, 2025: 4.3, 2026: 3.7}
    print(f"  {'year':6s} {'carry @0.10%':>13s} {'~T-bill':>9s} {'excess':>9s}")
    for year, rf in approx_rf.items():
        sub = slice_period(ext, f"{year}-01-01", f"{year}-12-31")
        m, _ = run_cell(sub, fee=0.001, rebalance="Q", count=False)
        print(f"  {year:<6d} {m.annualized_pct:>+12.2f}% {rf:>8.1f}% "
              f"{m.annualized_pct - rf:>+8.2f}%")


def configs_report() -> None:
    _hdr("configurations evaluated (deflated-Sharpe trials count)")
    uniq = sorted(set(CONFIGS))
    print(f"\n  distinct configurations evaluated: {len(uniq)}")
    print(f"  total configuration-evaluations (incl. re-runs on sub-periods): {len(CONFIGS)}")
    print("\n  the distinct set:")
    for c in uniq:
        print(f"    {c}")
    print("\n  NOTHING was selected on any of them: the primary specification")
    print("  (quarterly rebalance, roundtrip fees, gearing 1.0, tiers 0/0.10/0.40)")
    print("  was fixed by the R-39 pre-registration before any 2024-2026 number")
    print("  was read. The monthly/annual/drift rows are the reported")
    print("  neighbourhood, not candidates.")


def main() -> None:
    series = build_series()
    print(f"price series: {series['extended'].price_label}, "
          f"{series['extended'].price.index[0]:%Y-%m-%d} -> "
          f"{series['extended'].price.index[-1]:%Y-%m-%d}")
    print(f"extended funding: {len(series['extended'].rate)} settlements, "
          f"{(series['extended'].source == 'binance').sum()} binance / "
          f"{(series['extended'].source == 'deribit').sum()} deribit")
    print(f"deribit-only funding: {len(series['deribit-only'].rate)} settlements")

    if len(sys.argv) > 1 and sys.argv[1] == "parity":
        check_validation_parity(series)
        return
    check_validation_parity(series)
    headline_table(series)
    per_year(series)
    sharpe_vs_literature(series)
    rebalance_sensitivity(series)
    cash_benchmark(series)
    benchmarks()
    configs_report()


if __name__ == "__main__":
    main()
