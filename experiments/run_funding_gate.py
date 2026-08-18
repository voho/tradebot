#!/usr/bin/env python
"""Driver for backlog B-05 — funding gate on kelly_regime_v4's exposure.

Splits follow ROUTINE.md step 3::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
                                                  (funding itself only starts
                                                  2020-01-01, so the gate only
                                                  ever exercises in the last
                                                  year of this slice)
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
                                                  (full funding coverage)
    holdout           2023-01-01 ->              step 4 only, pre-registered

Usage::

    python experiments/run_funding_gate.py sweep       # step 3, 12 configs
    python experiments/run_funding_gate.py causality    # by-hand lookahead probe
    python experiments/run_funding_gate.py holdout      # step 4, frozen, point estimates
    python experiments/run_funding_gate.py interval     # paired bootstrap on the holdout
    python experiments/run_funding_gate.py fees         # 0.40% Bitstamp tier stress check
    python experiments/run_funding_gate.py funding_cost # futures with funding CHARGED
    python experiments/run_funding_gate.py eth          # falsification / mechanics control

======================================================================
PRE-REGISTRATION — written after ``sweep`` (step 3, 12 configs, inner-train
+ inner-validation, spot) but BEFORE the holdout (OOS_START) was touched.
======================================================================

The sweep result (spot, inner-validation 2021-01-01 -> 2022-12-31;
kelly_regime_v4 baseline: final $998 (-0.2%), DD 33.2%, sharpe 0.14):

    w=30d q=0.90 mode=zero    final $1,110 (+11.0%)  DD 27.6%  sharpe 0.33
    w=90d q=0.90 mode=zero    final $1,101 (+10.1%)  DD 30.0%  sharpe 0.32
    w=60d q=0.90 mode=zero    final $  992 ( -0.8%)  DD 29.4%  sharpe 0.12
    (q=0.95 and mode="scale" variants all sat closer to the v4 baseline;
    full 12-row table reproduced by ``sweep``)

Frozen configuration (selected on inner-validation, spot, ONLY):

    FundingGateKelly(window_days=30, quantile=0.90, mode="zero", scale=0.3)

i.e. stand fully flat (multiply exposure by 0.0) whenever the most
recently settled 8h funding print exceeds the 90th percentile of its own
trailing 30-day history. ``scale`` is inert for this frozen config (mode
is "zero") and is reported only because it is swept for the other 6
"mode=scale" configurations in the neighbourhood table.

Neighbourhood note (P3, read honestly before the holdout): q=0.90+zero
improves max-drawdown over v4 at ALL three swept windows (27.6% / 29.4% /
30.0% vs the baseline's 33.2%) — that is the plateau. Return/Sharpe is
NOT a plateau at this granularity: w=30 and w=90 both beat v4 on Sharpe,
but w=60 sits slightly below it. This is disclosed now, not discovered
after seeing the holdout — a report that only cites w=30 and hides w=60
would be exactly the shopping ROUTINE.md warns against.

Decision rule (see PRE-REGISTERED PROMOTION RULE below, copied verbatim
from the task brief) is evaluated once, after this file is frozen, against:

  P1: beats buy_and_hold on the full 2023+ holdout, spot, after real costs
      (0.10% primary, 0.40% stress).
  P2: beats kelly_regime_v4 by > +/-0.2 Sharpe noise floor on log growth,
      OR matches its growth while cutting max DD by >= 10pp, OR (if the
      full-holdout effect is diluted by the ~1/3.6-year funding coverage)
      shows a clear non-noise improvement over v4 restricted to the
      funding-covered 2023 sub-window while being no worse than v4 on the
      full holdout.
  P3: the swept neighbourhood (12 configs) is a plateau, not a knife-edge.
  P4: the ETH control run (no ETH funding file exists, so the gate never
      fires) shows FundingGateKelly behaves identically to registered
      kelly_regime_v4 -- a sanity check on inherited mechanics, not a test
      of the funding channel.

Anything else is NEGATIVE. The rule is not to be changed after looking at
the holdout; a change would be flagged explicitly rather than silently
re-run.
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

from experiments.funding_gate import FundingGateKelly  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL_FUNDING = load_funding(ROOT / "data")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS_START = "2023-01-01"
OOS = (OOS_START, None)
OOS_COVERED = ("2023-01-01", "2023-12-31")  # funding-covered sub-window

N_EVALUATED = 0  # incremented once per distinct (window, quantile, mode) config

# The sweep grid: 3 windows x 2 quantiles x 2 modes = 12 configurations.
# Covers the task brief's window examples (30/60/90d) in full and its
# quantile examples at the extreme end (90th/95th) -- "top decile" is the
# stated hypothesis, so 85th was dropped in favour of spending the budget
# on 90th/95th x more windows. "scale" uses a single fixed fraction (0.3)
# rather than sweeping a fourth axis, to keep the grid inside 8-16 configs.
WINDOWS = (30.0, 60.0, 90.0)
QUANTILES = (0.90, 0.95)
MODES = ("zero", "scale")
FIXED_SCALE = 0.3

FROZEN = dict(window_days=30.0, quantile=0.90, mode="zero", scale=FIXED_SCALE)


# ---------------------------------------------------------------- utilities


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0):
    """One backtest over [start, end] -> (Metrics, BacktestResult)."""
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    return compute_metrics(result), result


def line(tag, m, result=None):
    extra = ""
    if result is not None and "funding_gate_active" in result.df:
        active = result.df["funding_gate_active"].to_numpy()
        covered = result.df["funding_covered"].to_numpy().astype(bool)
        if covered.any():
            extra = (f" covered={covered.mean():>5.1%} "
                     f"gate_active_of_covered={active[covered].mean():>5.1%}")
        else:
            extra = " covered=0.0%"
    print(f"  {tag:38s} final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
          f"trades={m.num_trades:>5d}{extra}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


def gk(**over):
    """A FundingGateKelly using the real committed funding file."""
    params = dict(funding=REAL_FUNDING)
    params.update(over)
    return FundingGateKelly(**params)


# -------------------------------------------------------------------- sweep


def sweep() -> None:
    """Step 3. 12 configs, inner-train and inner-validation, spot market.

    Spot is the market the pre-registered decision rule is defined on
    (P1/P2), so tuning happens there. Every row also reports
    ``gate_active`` -- the share of bars the gate fired on -- which is the
    diagnostic for pre-registered failure mode (b): a gate that is a
    generic "hold less" device would show a large, config-insensitive
    active fraction; a genuine top-decile gate should show a small one
    that scales roughly with ``1 - quantile``.
    """
    global N_EVALUATED
    print("kelly_regime_v4 baseline (spot):")
    for split_name, (s, e) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        m, res = measure(get_strategy("kelly_regime_v4"), s, e)
        line(f"{split_name:17s} v4", m, res)

    for window in WINDOWS:
        for q in QUANTILES:
            for mode in MODES:
                N_EVALUATED += 1
                tag = f"w={window:g}d q={q:.2f} mode={mode}"
                print(f"\n[{N_EVALUATED}/{len(WINDOWS) * len(QUANTILES) * len(MODES)}] {tag}")
                strat_kwargs = dict(window_days=window, quantile=q, mode=mode,
                                    scale=FIXED_SCALE)
                for split_name, (s, e) in (("inner-train", TRAIN),
                                           ("inner-validation", VALID)):
                    m, res = measure(gk(**strat_kwargs), s, e)
                    line(f"  {split_name}", m, res)
    print(f"\nconfigurations evaluated (step 3, distinct, counted once): {N_EVALUATED}")


# ------------------------------------------------------------------- causality


def causality() -> None:
    """The strict lookahead probe, by hand -- experiments get no CI protection.

    Same two-opposite-tampers procedure as R-28/B-11: bars after a cut are
    multiplied by 3 (price) / 7 (volume) in one copy and divided by the
    same in the other, and every decision at or before the cut must be
    identical. Run against the FROZEN config with the real funding series
    attached, so both the price-driven v4 mechanics and the funding-gate
    arithmetic are exercised together.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    df = DF.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        s = gk(**FROZEN)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    a, b = decisions(up), decisions(down)
    bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
    print(f"orders: tampered from bar {cut:,} of {len(df):,}; checked bars {bars}")
    print("  FAIL - reads the future at bars " + str(bad) if bad
          else "  PASS - every decision at or before the cut is unchanged")

    pa = gk(**FROZEN).prepare(up.copy())
    pb = gk(**FROZEN).prepare(down.copy())
    ok = not bad
    for col in ("target", "v4_target", "funding_gate_active"):
        diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
        worst = float(np.nanmax(diff))
        good = worst < 1e-12
        ok &= good
        print(f"  column {col:22s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if good else 'FAIL'}")
    print(f"\noverall: {'PASS' if ok else 'FAIL'}")


# --------------------------------------------------------------------- holdout


def holdout() -> None:
    """Step 4. Frozen config vs buy_and_hold and kelly_regime_v4.

    Both markets, full 2023+ holdout AND the funding-covered 2023-only
    sub-window (pre-registered failure mode (a) diagnostic).
    """
    for mname, market in MARKETS:
        for label, (s, e) in (("FULL HOLDOUT 2023-01-01 ->", OOS),
                              ("2023-ONLY (funding-covered)", OOS_COVERED)):
            print(f"\n{label} / {mname}:")
            for name in ("buy_and_hold", "kelly_regime_v4"):
                m, res = measure(get_strategy(name), s, e, market=market)
                line(name, m, res)
            m, res = measure(gk(**FROZEN), s, e, market=market)
            line("funding_gate_kelly (frozen)", m, res)

    # Gate activity / exposure-delta diagnostics for failure mode (b),
    # measured once on the funding-covered sub-window, spot.
    print("\ngate activity / exposure diagnostics (2023-only, spot):")
    _, res_fgk = measure(gk(**FROZEN), *OOS_COVERED, market=SPOT)
    _, res_v4 = measure(get_strategy("kelly_regime_v4"), *OOS_COVERED, market=SPOT)
    active = res_fgk.df["funding_gate_active"].to_numpy()
    covered_mask = res_fgk.df["funding_covered"].to_numpy().astype(bool)
    frac_covered = float(covered_mask.mean())
    frac_active_of_covered = float(active[covered_mask].mean()) if covered_mask.any() else float("nan")
    v4_tgt = np.abs(res_v4.df["target"].to_numpy())
    fgk_tgt = np.abs(res_fgk.df["target"].to_numpy())
    print(f"  bars with funding coverage: {frac_covered:.1%}")
    print(f"  gate active, of covered bars: {frac_active_of_covered:.1%}")
    print(f"  mean |exposure| v4:              {v4_tgt.mean():.3f}")
    print(f"  mean |exposure| funding_gate:    {fgk_tgt.mean():.3f}")
    print(f"  time-in-market v4:               {(v4_tgt > 1e-9).mean():.1%}")
    print(f"  time-in-market funding_gate:      {(fgk_tgt > 1e-9).mean():.1%}")

    # And the same coverage/activity fraction over the FULL holdout, to
    # report what share of it is even funding-covered at all.
    _, res_fgk_full = measure(gk(**FROZEN), *OOS, market=SPOT)
    active_full = res_fgk_full.df["funding_gate_active"].to_numpy()
    covered_full = res_fgk_full.df["funding_covered"].to_numpy().astype(bool)
    print(f"\n  full holdout: funding-covered bars = {covered_full.mean():.1%} "
          f"({int(covered_full.sum()):,} of {len(covered_full):,})")
    if covered_full.any():
        print(f"  full holdout: gate active, of covered bars = "
              f"{active_full[covered_full].mean():.1%}")


# ------------------------------------------------------------------- interval


def interval() -> None:
    """Paired stationary block-bootstrap (30-day mean block, 2000 resamples)
    on daily returns, exactly the R-29/R-30/R-31 method: identical resamples
    for both arms of a pair so the market's own variance cancels.

    Two comparisons (vs buy_and_hold, vs kelly_regime_v4), two periods (full
    holdout, 2023-only), both markets.
    """
    from tradebot.inference import (daily_returns, max_drawdown_from_returns,
                                    paired_bootstrap, stationary_bootstrap_indices,
                                    total_log_return)

    for mname, market in MARKETS:
        for label, (s, e) in (("FULL HOLDOUT", OOS), ("2023-ONLY", OOS_COVERED)):
            curves = {}
            for name, strat in (("buy_and_hold", get_strategy("buy_and_hold")),
                                ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                                ("funding_gate_kelly", gk(**FROZEN))):
                res = run_period(strat, DF, s, e, market=market,
                                 start_balance=1_000.0, data_label=LABEL)
                curves[name] = daily_returns(res.equity).to_numpy()
            n = len(curves["buy_and_hold"])
            idx = stationary_bootstrap_indices(n, 30.0, 2_000, np.random.default_rng(7))
            print(f"\n{label} / {mname}  ({n} daily observations)")
            for opp in ("buy_and_hold", "kelly_regime_v4"):
                a, b = curves["funding_gate_kelly"], curves[opp]
                for stat_name, stat in (("Delta log growth", total_log_return),
                                        ("Delta max DD (pp)", max_drawdown_from_returns)):
                    r = paired_bootstrap(a, b, stat, indices=idx)
                    mark = ("BETTER" if r.diff.lo > 0 else
                            ("WORSE" if r.diff.hi < 0 else "~"))
                    print(f"  fgk - {opp:16s} {stat_name:20s} {mark:6s} "
                          f"{r.diff.point:>+8.4f} [{r.diff.lo:>+8.4f}, {r.diff.hi:>+8.4f}]  "
                          f"P(fgk>opp)={r.p_positive:.2f}")


# ---------------------------------------------------------------------- fees


def fees() -> None:
    """Real fee tier check: spot at 0.10% (table assumption) and 0.40%
    (Bitstamp entry tier), full holdout.
    """
    for tier, label in ((0.001, "0.10% (table assumption)"),
                        (0.004, "0.40% (Bitstamp entry tier)")):
        market = MarketSpec.spot(fee_rate=tier)
        print(f"\nFULL HOLDOUT 2023-01-01 -> / spot @ {label}:")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            m, res = measure(get_strategy(name), *OOS, market=market)
            line(name, m, res)
        m, res = measure(gk(**FROZEN), *OOS, market=market)
        line("funding_gate_kelly (frozen)", m, res)


# ------------------------------------------------------------------ funding_cost


def funding_cost() -> None:
    """Futures holdout with real funding CHARGED (mirrors funding_study.py).

    Real funding is observed only through 2023-12-31; the mean rate fills
    the rest, exactly as ``scripts/funding_study.py`` does, so this is a
    band on the true number outside 2023, not a measurement.
    """
    real = REAL_FUNDING
    grid = pd.date_range(DF.index[0].ceil("8h"), DF.index[-1], freq="8h", tz="UTC")
    filler = pd.Series(float(real.mean()), index=grid)
    blended = pd.concat([filler[~filler.index.isin(real.index)], real]).sort_index()

    print(f"FULL HOLDOUT 2023-01-01 -> / futures 5x, funding CHARGED "
          f"(real through {real.index[-1]:%Y-%m}, mean "
          f"{real.mean() * 3 * 365.25:+.1%}/yr after):")
    lo = int(DF.index.searchsorted(OOS_START))
    contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                  ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                  ("funding_gate_kelly (frozen)", gk(**FROZEN))]
    for name, strat in contenders:
        pre = min(lo, strat.warmup)
        raw = run_backtest(strat, DF.iloc[lo - pre:], FUTURES, 1_000.0,
                           trade_start=pre, funding=blended, data_label=LABEL)
        trimmed = replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
        m = compute_metrics(trimmed)
        print(f"  {name:30s} final=${m.final_balance:>10,.0f} "
              f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
              f"funding paid=${raw.funding_paid:>8,.0f}"
              f"{'  LIQUIDATED' if m.liquidated else ''}")


# -------------------------------------------------------------------------- eth


def eth() -> None:
    """Pre-registered falsification: does the inherited v4 mechanics survive on ETH?

    No ETH funding file exists, so ``funding=None`` and the gate is a
    strict no-op by construction (``_funding_factor`` returns all-ones).
    This checks that ``FundingGateKelly`` degenerates exactly to
    ``kelly_regime_v4`` when there is nothing to gate on -- a control on
    the inherited mechanics, not a test of the funding channel, which has
    no ETH data to act on at all. Same venue and window as R-17/R-28/R-31.
    """
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        window = df.loc["2016-03-01":"2019-12-31"]
        print(f"\n{asset}  {len(window):,} bars  "
              f"{window.index[0]:%Y-%m-%d} -> {window.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            m_v4, res_v4 = measure(get_strategy("kelly_regime_v4"), None, None,
                                   df=window, market=market)
            m_fgk, res_fgk = measure(FundingGateKelly(funding=None, **FROZEN), None, None,
                                     df=window, market=market)
            line(f"  {mname} v4               ", m_v4, res_v4)
            line(f"  {mname} funding_gate(None)", m_fgk, res_fgk)
            worst = float(np.max(np.abs(res_v4.equity.to_numpy()
                                        - res_fgk.equity.to_numpy())))
            print(f"    max |equity difference| v4 vs funding_gate(no data): "
                  f"{worst:.3e}  {'IDENTICAL' if worst < 1e-6 else 'DIFFERS'}")


COMMANDS = {"sweep": sweep, "causality": causality, "holdout": holdout,
            "interval": interval, "fees": fees, "funding_cost": funding_cost,
            "eth": eth}


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate.py [{'|'.join(COMMANDS)}]")
