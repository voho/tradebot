#!/usr/bin/env python
"""Backlog B-05 — funding as a gate on the incumbent (`kelly_regime_v4`).

The idea, one sentence: stand flat (or haircut exposure) when the real
Binance BTCUSDT funding rate is in its own recent top decile, because
R-14 showed funding is charged 7x richer exactly while the strategy
holds (+20%/yr vs +2.8%/yr flat) and R-16 showed forward spot returns
are lower after high funding (14d Q1-Q5 spread +3.57pp), independent of
a momentum proxy (corr with trailing return only 0.39).

This file is the ONLY file this branch may create or edit (see the task
brief). It follows the pattern of experiments/run_eprocess.py and
experiments/matched_risk.py: sys.path setup, an `ev()` helper built on
tradebot.window.run_period that counts every backtest into a
module-level counter, and TRAIN/VALID/OOS splits identical to the rest
of the repo.

Usage::

    python experiments/funding_gate.py rates       # what the gate sees, before any backtest
    python experiments/funding_gate.py sweep        # inner-train + inner-validation grid
    python experiments/funding_gate.py neighbours   # plateau check around the frozen config
    python experiments/funding_gate.py causality    # manual lookahead self-check (step 3)
    python experiments/funding_gate.py falsify      # pre-registered ETH falsification test
    python experiments/funding_gate.py holdout      # step 4 — DO NOT RUN before pre-registering
    python experiments/funding_gate.py all          # everything except holdout
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
FUNDING = load_funding(ROOT / "data")  # real Binance BTCUSDT, 2020-01-01 -> 2023-12-31
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS_START = "2023-01-01"

N_EVALUATED = 0     # configs evaluated in step 3, inner-train/inner-validation only
HOLDOUT_CALLS = 0   # every backtest call touching a date >= OOS_START (step 4 only)


# --------------------------------------------------------------------- strategy


class FundingGatedKellyV4(KellyRegimeV4):
    """kelly_regime_v4 with exposure gated off when funding is in its own recent top decile.

    Not `@register`-ed: this is a step-3/4 experiment, per ROUTINE.md,
    not a candidate for the comparison table until (if) it is promoted.

    Mechanism: `prepare()` calls the parent unchanged to get the causal
    `target` column, then multiplies it by a gate computed ENTIRELY from
    the funding series and bar timestamps (never from price), so the
    gate cannot introduce a price-side lookahead by construction — the
    causality self-check below verifies that empirically rather than
    just by this argument.

    The gate:
      1. On the funding *settlement* series (8-hourly, ~3/day), compute a
         causal reference threshold — a rolling time-window quantile
         (`lookback_days`) or an expanding (all-history-to-date) quantile
         if `lookback_days` is None. Both use only settlements at or
         before the current one: no future settlement ever informs a
         past threshold.
      2. Latch a boolean gate state per settlement with hysteresis:
         enter when the rate exceeds the `percentile` quantile, exit
         only once it drops back below the (lower) `exit_percentile`
         quantile. `exit_percentile == percentile` disables hysteresis.
         Before `min_settlements` observations exist, the threshold is
         undefined and the gate defaults OFF (ungated) — the honest
         behaviour when there isn't enough history to rank against.
      3. Forward-fill that per-settlement state onto every 5m bar using
         the LAST settlement at or before the bar's own timestamp
         (`Series.reindex(..., method="ffill")` on two sorted
         DatetimeIndexes is a backward as-of join — never uses a future
         settlement). Bars before the first settlement, or on a venue/
         window with no funding series at all (`funding=None`, e.g. the
         ETH falsification test), get gate=OFF everywhere: the strategy
         degenerates exactly to the ungated parent. Real funding data
         only exists 2020-01-01 to 2023-12-31 (confirmed by inspection,
         not assumed); outside that window this class is IDENTICAL to
         plain `kelly_regime_v4`, never a synthesized guess.
      4. When gated, `target` is multiplied by `haircut` (0.0 = hard
         flat, otherwise a partial exposure cut).
    """

    name = "funding_gated_kelly_v4"

    def __init__(
        self,
        funding: "pd.Series | None" = FUNDING,
        percentile: float = 0.90,
        exit_percentile: "float | None" = None,
        lookback_days: "float | None" = 365.0,
        min_settlements: int = 90,
        haircut: float = 0.0,
        horizons: tuple[int, ...] = (20, 40, 80),
        **kwargs,
    ) -> None:
        super().__init__(horizons=horizons, **kwargs)
        if not 0.0 < percentile < 1.0:
            raise ValueError(f"percentile must be in (0,1), got {percentile!r}")
        self.funding = funding
        self.percentile = percentile
        self.exit_percentile = percentile if exit_percentile is None else exit_percentile
        if not 0.0 < self.exit_percentile <= self.percentile:
            raise ValueError("exit_percentile must be in (0, percentile]")
        self.lookback_days = lookback_days
        self.min_settlements = min_settlements
        if not 0.0 <= haircut <= 1.0:
            raise ValueError(f"haircut must be in [0,1], got {haircut!r}")
        self.haircut = haircut

    def _gate_multiplier(self, index: pd.DatetimeIndex) -> np.ndarray:
        n = len(index)
        if self.funding is None or len(self.funding) == 0:
            return np.ones(n)

        f = self.funding.sort_index()
        if self.lookback_days is None:
            thr_in = f.expanding(min_periods=self.min_settlements).quantile(self.percentile)
            thr_out = f.expanding(min_periods=self.min_settlements).quantile(self.exit_percentile)
        else:
            window = f"{float(self.lookback_days):g}D"
            thr_in = f.rolling(window, min_periods=self.min_settlements).quantile(self.percentile)
            thr_out = f.rolling(window, min_periods=self.min_settlements).quantile(self.exit_percentile)

        f_vals = f.to_numpy(dtype=float)
        thr_in_v = thr_in.to_numpy(dtype=float)
        thr_out_v = thr_out.to_numpy(dtype=float)

        state = np.zeros(len(f), dtype=bool)
        on = False
        for i in range(len(f)):
            if not np.isfinite(thr_in_v[i]):
                on = False  # not enough history to rank against -> ungated
            elif on:
                if f_vals[i] < thr_out_v[i]:
                    on = False
            else:
                if f_vals[i] > thr_in_v[i]:
                    on = True
            state[i] = on

        gate_settled = pd.Series(state, index=f.index)
        # Backward as-of join onto the bar grid: each bar sees the last
        # settlement AT OR BEFORE its own timestamp, never a later one.
        gate_on_bars = gate_settled.reindex(index, method="ffill").fillna(False).to_numpy()
        return np.where(gate_on_bars, self.haircut, 1.0)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)
        mult = self._gate_multiplier(df.index)
        df["funding_gate_on"] = mult < 1.0 - 1e-12
        df["target"] = df["target"].to_numpy() * mult
        return df


# --------------------------------------------------------------------- helpers


def ev(strategy, start, end, df=None, market=SPOT, tag="", balance=1_000.0,
       count=True, holdout=False):
    """One backtest, one line, counted into the right counter."""
    global N_EVALUATED, HOLDOUT_CALLS
    if holdout:
        HOLDOUT_CALLS += 1
    elif count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    print(f"  {tag or strategy.name:38s} {market.name:11s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"fills={len(result.fills):>5d} fees=${m.fees_paid:>8,.0f} "
          f"DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f}{'  LIQUIDATED' if m.liquidated else ''}")
    return m


def ev_funding(strategy, start, end, market=FUTURES, tag="", balance=1_000.0,
               count=True, holdout=False, funding=FUNDING):
    """One backtest WITH funding charged, following funding_study.py's `_period()` pattern.

    `run_period` has no `funding=` parameter, so this calls
    `run_backtest` directly with the same warmup-prefix / trade_start /
    trim discipline `run_period` uses internally.
    """
    global N_EVALUATED, HOLDOUT_CALLS
    if holdout:
        HOLDOUT_CALLS += 1
    elif count:
        N_EVALUATED += 1
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre:hi], market, balance,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:],
                                           df=raw.df.iloc[pre:])
    m = compute_metrics(trimmed)
    print(f"  {tag or strategy.name:38s} {market.name:11s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
          f"funding paid=${raw.funding_paid:>8,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")
    return m


# --------------------------------------------------------------------------- rates


def rates() -> None:
    """What the gate actually sees, before any backtest is run."""
    f = FUNDING.sort_index()
    print(f"funding series: {len(f)} settlements, {f.index.min()} -> {f.index.max()}")
    print(f"  overlap with TRAIN {TRAIN}:  "
          f"{f[(f.index >= TRAIN[0]) & (f.index <= TRAIN[1] + ' 23:59:59')].shape[0]} settlements")
    print(f"  overlap with VALID {VALID}:  "
          f"{f[(f.index >= VALID[0]) & (f.index <= VALID[1] + ' 23:59:59')].shape[0]} settlements")
    print(f"  overlap with OOS >= {OOS_START}: "
          f"{f[f.index >= OOS_START].shape[0]} settlements "
          f"(last real settlement {f.index.max()})")

    for pct in (0.90, 0.95):
        thr_expanding = f.expanding(min_periods=90).quantile(pct)
        thr_365 = f.rolling("365D", min_periods=90).quantile(pct)
        on_expanding = (f >= thr_expanding).mean()
        on_365 = (f >= thr_365).mean()
        print(f"  p{int(pct*100)}: fraction of settlements gated, "
              f"expanding threshold={on_expanding:.1%}, 365D rolling threshold={on_365:.1%}")


# --------------------------------------------------------------------------- sweep


def _variants():
    """Small, targeted grid: percentile x lookback x haircut x hysteresis. Every entry is one trial."""
    out = []
    for pct in (0.90, 0.95):
        for lookback in (365.0, None):
            lb_tag = f"{int(lookback)}d" if lookback is not None else "expanding"
            for haircut in (0.0, 0.5):
                hc_tag = "hard" if haircut == 0.0 else f"haircut{haircut:g}"
                # with hysteresis (10pp lower exit) and without
                for exit_pct in (pct - 0.10, pct):
                    hy_tag = "hyst" if exit_pct < pct else "nohyst"
                    tag = f"p{int(pct*100)} lb={lb_tag} {hc_tag} {hy_tag}"
                    out.append((tag, dict(percentile=pct, lookback_days=lookback,
                                          haircut=haircut, exit_percentile=exit_pct)))
    return out


def _benchmarks(start, end, market, label, holdout=False):
    print(f"\n{label} benchmarks:")
    for name in ("buy_and_hold", "kelly_regime_v4"):
        ev(get_strategy(name), start, end, market=market, tag=f"  {name}",
           count=False, holdout=holdout)


def sweep() -> None:
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN, "INNER-TRAIN"), (VALID, "INNER-VALIDATION")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            print(f"{split} / {mname} variants (funding-free engine, cost isolated separately):")
            for tag, kw in _variants():
                ev(FundingGatedKellyV4(**kw), start, end, market=market, tag=tag)
    print(f"\nconfigurations evaluated so far (step 3, inner splits only): {N_EVALUATED}")


def sweep_funding_cost() -> None:
    """The COST channel in isolation: futures WITH funding charged, inner-validation only

    (the only inner split fully covered by the real 2020-2023 funding
    series -- see `rates()`). This is the number the gate is actually
    meant to move.
    """
    print("\nINNER-VALIDATION / futures 5x, FUNDING CHARGED (real series):")
    for name in ("buy_and_hold", "kelly_regime_v4"):
        ev_funding(get_strategy(name), *VALID, tag=f"  {name}", count=False)
    for tag, kw in _variants():
        ev_funding(FundingGatedKellyV4(**kw), *VALID, tag=tag)
    print(f"\nconfigurations evaluated so far (step 3, inner splits only): {N_EVALUATED}")


# ----------------------------------------------------------------------- neighbours


# Selected on INNER-VALIDATION, funding-charged futures (the split fully
# covered by real funding and the criterion the gate exists to move; see
# sweep_funding_cost()). p90/expanding/hard/hysteresis was both the most
# literal reading of B-05 ("top decile", "stand flat", low-turnover via
# latching) AND the best-performing hard-gate cell on that criterion:
# kelly_regime_v4 funding-charged INNER-VALIDATION futures = final $887
# (-11.3%), DD 34.7%, sharpe -0.06, funding paid $184; this config = final
# $980 (-2.0%), DD 31.6%, sharpe +0.08, funding paid $91 -- better on
# return, drawdown AND cost simultaneously, not a one-axis cherry pick.
FROZEN = dict(percentile=0.90, lookback_days=None, exit_percentile=0.80,
             min_settlements=90, haircut=0.0)


def neighbours() -> None:
    """Plateau, not peak: vary one knob at a time around the frozen config."""
    grid = [("FROZEN p90 lb365 hyst80 hard", {})]
    grid += [(f"percentile={p}", dict(percentile=p, exit_percentile=p - 0.10))
             for p in (0.85, 0.95)]
    grid += [(f"lookback={lb}", dict(lookback_days=lb))
             for lb in (180.0, None)]
    grid += [(f"exit_percentile={e}", dict(exit_percentile=e))
             for e in (0.90, 0.70)]
    grid += [(f"haircut={h}", dict(haircut=h)) for h in (0.25, 0.5, 0.75)]
    grid += [(f"min_settlements={m}", dict(min_settlements=m)) for m in (30, 180)]
    print("\nINNER-VALIDATION / futures 5x, FUNDING CHARGED, neighbourhood:")
    for tag, kw in grid:
        ev_funding(FundingGatedKellyV4(**{**FROZEN, **kw}), *VALID, tag=tag)
    print("\nINNER-TRAIN / futures 5x, FUNDING CHARGED (mostly no-op: real funding "
          "only overlaps the last ~3 months of this split), neighbourhood:")
    for tag, kw in grid:
        ev_funding(FundingGatedKellyV4(**{**FROZEN, **kw}), *TRAIN, tag=tag, count=False)
    print(f"\nconfigurations evaluated so far (step 3, inner splits only): {N_EVALUATED}")


# --------------------------------------------------------------------------- causality


def causality() -> None:
    """Manual lookahead self-check (no automatic protection for unregistered strategies).

    Two adversarial tampers of the OHLCV frame after a cut index (x3 and
    /3 on price and volume); every decision AT OR BEFORE the cut must be
    identical across the untouched frame and both tampers. This is the
    check that catches a whole-series stat computed once and applied to
    early rows, which plain truncation misses (ROUTINE.md's R-21).

    The funding gate itself never reads price, only the (untouched)
    funding series and bar timestamps, so it is causal by construction;
    this check verifies the COMBINED class (parent v4 logic + gate) has
    no bug that breaks that.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    df = DF.iloc[-300_000:].copy()  # covers real funding's tail, 2023, plus margin
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    base, up, down = df.copy(), df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        s = FundingGatedKellyV4(**FROZEN)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    a, b, c = decisions(base), decisions(up), decisions(down)
    bad = [bar for bar, oa, ob, oc in zip(bars, a, b, c) if oa != ob or oa != oc]
    print(f"tampered from bar {cut:,} of {len(df):,}; checked bars {bars}")
    print("FAIL - reads the future at bars " + str(bad) if bad
          else "PASS - every decision at or before the cut is bit-identical across "
               "base/x3/div3")

    pa = FundingGatedKellyV4(**FROZEN).prepare(base.copy())
    pb = FundingGatedKellyV4(**FROZEN).prepare(up.copy())
    pc = FundingGatedKellyV4(**FROZEN).prepare(down.copy())
    for col in ("target", "funding_gate_on"):
        diff_ab = np.abs(pa[col].to_numpy()[:cut].astype(float)
                         - pb[col].to_numpy()[:cut].astype(float))
        diff_ac = np.abs(pa[col].to_numpy()[:cut].astype(float)
                         - pc[col].to_numpy()[:cut].astype(float))
        worst = float(max(np.nanmax(diff_ab), np.nanmax(diff_ac)))
        print(f"  column {col:18s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if worst < 1e-12 else 'FAIL'}")


# --------------------------------------------------------------------------- falsification


def falsify() -> None:
    """Pre-registered falsification test: survive on ETH (chosen in step 2).

    No real ETH funding data exists anywhere in this repo, so `funding=None`
    for the ETH run. "Survive" here means: the GATED class produces output
    numerically IDENTICAL to the ungated `kelly_regime_v4` on both BTC and
    ETH Bitfinex data, because with no funding series the gate multiplier
    is 1.0 everywhere by construction (see class docstring, point 3). This
    tests only that wiring the (inactive) gate in does not corrupt the
    underlying v4 component's already-known ETH behaviour (R-17) -- it
    cannot test the funding mechanism itself, since ETH has no funding.
    """
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for market in (SPOT, FUTURES):
            m_v4 = ev(get_strategy("kelly_regime_v4"), None, None, df=df, market=market,
                     tag="  kelly_regime_v4 (ungated)", count=False)
            m_gate = ev(FundingGatedKellyV4(funding=None, **FROZEN), None, None, df=df,
                       market=market, tag="  funding_gated (funding=None)", count=False)
            same = (abs(m_v4.final_balance - m_gate.final_balance) < 1e-6
                   and abs(m_v4.max_drawdown_pct - m_gate.max_drawdown_pct) < 1e-9
                   and m_v4.num_trades == m_gate.num_trades)
            print(f"    identical to ungated v4: {'PASS' if same else 'FAIL'}")


# --------------------------------------------------------------------------- holdout

# ############################################################################
# DO NOT CALL holdout() UNTIL THE PRE-REGISTRATION TEXT IS WRITTEN INTO THE
# REPORT. See experiments/funding_gate_report.md, section "frozen
# configuration and decision rule", which must exist BEFORE this runs.
# ############################################################################


def holdout() -> None:
    """Step 4. Frozen config (`FROZEN` above); decision rule in the report, word for word."""
    print(f"\n{'='*70}\nHOLDOUT >= {OOS_START} (real funding covers up to "
          f"{FUNDING.index.max():%Y-%m-%d}; only that slice can test the gate's "
          f"mechanism -- the rest of the holdout is a null/regression control)\n{'='*70}")

    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x, funding-free")):
        print(f"\nHOLDOUT {OOS_START} -> end / {mname}:")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            ev(get_strategy(name), OOS_START, None, market=market,
               tag=f"  {name}", count=False, holdout=True)
        ev(FundingGatedKellyV4(**FROZEN), OOS_START, None, market=market,
           tag="  funding_gated_kelly_v4 (FROZEN)", count=False, holdout=True)

    fmax = FUNDING.index.max()
    print(f"\nHOLDOUT {OOS_START} -> {fmax:%Y-%m-%d} / futures 5x, FUNDING CHARGED "
          f"(real series only -- no extrapolation past {fmax:%Y-%m-%d}):")
    for name, strat in (("buy_and_hold", get_strategy("buy_and_hold")),
                        ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                        ("funding_gated_kelly_v4 (FROZEN)", FundingGatedKellyV4(**FROZEN))):
        ev_funding(strat, OOS_START, fmax.strftime("%Y-%m-%d"), market=FUTURES,
                  tag=f"  {name}", count=False, holdout=True)

    print(f"\nHOLDOUT calls made that touch a date >= {OOS_START}: {HOLDOUT_CALLS}")


# --------------------------------------------------------------------------- main


def main() -> None:
    cmds = {
        "rates": rates,
        "sweep": lambda: (sweep(), sweep_funding_cost()),
        "neighbours": neighbours,
        "causality": causality,
        "falsify": falsify,
        "holdout": holdout,
        "all": lambda: (rates(), sweep(), sweep_funding_cost(), neighbours(),
                        causality(), falsify()),
    }
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which not in cmds:
        print(f"unknown command {which!r}; choose from {sorted(cmds)}")
        sys.exit(1)
    cmds[which]()


if __name__ == "__main__":
    main()
