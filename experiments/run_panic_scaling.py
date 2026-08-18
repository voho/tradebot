#!/usr/bin/env python
"""Driver for the panic-state exposure shrink experiment (unregistered idea).

Splits follow ROUTINE.md step 3::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              step 4 only, pre-registered

Usage::

    python experiments/run_panic_scaling.py sweep       # step 3: 16 configs
    python experiments/run_panic_scaling.py ablation     # failure mode (a)
    python experiments/run_panic_scaling.py causality    # by-hand lookahead probe
    python experiments/run_panic_scaling.py holdout      # step 4, frozen config
    python experiments/run_panic_scaling.py interval     # paired bootstrap
    python experiments/run_panic_scaling.py costs        # 0.40% fee tier + funding
    python experiments/run_panic_scaling.py windows      # path sensitivity
    python experiments/run_panic_scaling.py eth          # falsification test
    python experiments/run_panic_scaling.py neighbours   # P5 plateau check
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

from experiments.panic_scaling import PanicScaledKelly  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics, max_drawdown_pct  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS = ("2023-01-01", None)

OUT = ROOT / "reports" / "panic_scaling"

N_EVALUATED = 0  # distinct PanicScaledKelly configurations searched in step 3

# ---------------------------------------------------------------------------
# Step 3 sweep grid. 16 configurations, counted once each (inner-train, spot).
#
#   9  joint panic (use_decline=True): decline_window in {60, 90, 120} days
#      x shrink in {0.25, 0.5, 0.75}, at vol_window=20d, tercile_lookback=365d
#   2  vol_window sensitivity at the center point (decline=90, shrink=0.5):
#      vol_window in {10, 30}
#   2  tercile_lookback sensitivity at the center point: lookback in {180, 730}
#   3  vol-only ablation (use_decline=False, failure mode (a)'s control arm):
#      shrink in {0.25, 0.5, 0.75}, at vol_window=20d, lookback=365d
# ---------------------------------------------------------------------------

_CENTER = dict(vol_window_days=20.0, tercile_lookback_days=365.0)


def _joint_configs():
    for dw in (60.0, 90.0, 120.0):
        for s in (0.25, 0.5, 0.75):
            yield dict(decline_window_days=dw, shrink=s, use_decline=True, **_CENTER)


def _vol_window_configs():
    for vw in (10.0, 30.0):
        yield dict(decline_window_days=90.0, shrink=0.5, use_decline=True,
                   vol_window_days=vw, tercile_lookback_days=365.0)


def _lookback_configs():
    for lb in (180.0, 730.0):
        yield dict(decline_window_days=90.0, shrink=0.5, use_decline=True,
                   vol_window_days=20.0, tercile_lookback_days=lb)


def _ablation_configs():
    for s in (0.25, 0.5, 0.75):
        yield dict(decline_window_days=90.0, shrink=s, use_decline=False, **_CENTER)


def all_configs() -> list[dict]:
    cfgs = (list(_joint_configs()) + list(_vol_window_configs())
            + list(_lookback_configs()) + list(_ablation_configs()))
    assert len(cfgs) == 16, len(cfgs)
    return cfgs


def _tag(cfg: dict) -> str:
    kind = "joint" if cfg["use_decline"] else "vol-only"
    return (f"{kind} N={cfg['decline_window_days']:.0f}d M={cfg['vol_window_days']:.0f}d "
            f"L={cfg['tercile_lookback_days']:.0f}d s={cfg['shrink']:.2f}")


# --------------------------------------------------------------------- helpers


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0):
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    return m, result


def line(tag, m, result):
    print(f"  {tag:52s} final=${m.final_balance:>11,.0f} "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
          f"fills={len(result.fills):>5d} fees=${m.fees_paid:>8,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# ----------------------------------------------------------------------- sweep


def sweep() -> None:
    """Step 3. Every configuration on inner-train and inner-validation, spot."""
    global N_EVALUATED
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cfgs = all_configs()
    for cfg in cfgs:
        N_EVALUATED += 1
        for (start, end), split in ((TRAIN, "inner-train"), (VALID, "inner-validation")):
            m, res = measure(PanicScaledKelly(**cfg), start, end, market=SPOT)
            rows.append({"split": split, **cfg, "final": m.final_balance,
                        "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                        "fills": len(res.fills), "fees": m.fees_paid,
                        "liquidated": m.liquidated,
                        "panic_frac": float(np.mean(res.df["panic"])) if "panic" in res.df else float("nan")})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sweep.csv", index=False)

    # v4 reference, both splits, not counted (it is not a swept configuration).
    for (start, end), split in ((TRAIN, "inner-train"), (VALID, "inner-validation")):
        m, res = measure(get_strategy("kelly_regime_v4"), start, end, market=SPOT)
        rows.append({"split": split, "decline_window_days": float("nan"),
                    "vol_window_days": float("nan"), "tercile_lookback_days": float("nan"),
                    "shrink": 1.0, "use_decline": None, "final": m.final_balance,
                    "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                    "fills": len(res.fills), "fees": m.fees_paid,
                    "liquidated": m.liquidated, "panic_frac": float("nan")})
        print(f"\n{split}: kelly_regime_v4 (reference)  final=${m.final_balance:,.0f} "
              f"DD={m.max_drawdown_pct:.1f}% sharpe={m.sharpe:.2f}")

    for split in ("inner-train", "inner-validation"):
        sub = df[df.split == split]
        print(f"\n{split} (spot):")
        for _, r in sub.iterrows():
            cfg = {k: r[k] for k in ("decline_window_days", "vol_window_days",
                                     "tercile_lookback_days", "shrink", "use_decline")}
            print(f"  {_tag(cfg):58s} final=${r['final']:>11,.0f} DD={r['max_dd']:>5.1f}% "
                  f"sharpe={r['sharpe']:>5.2f} panic%={r['panic_frac']:>5.1%} "
                  f"fills={r['fills']:>4.0f}")
    print(f"\nconfigurations evaluated (distinct, counted once): {N_EVALUATED}")
    print(f"written: {OUT / 'sweep.csv'}")


# -------------------------------------------------------------------- ablation


def ablation() -> None:
    """Failure mode (a): does the joint (decline x vol) condition earn its keep
    over vol-only, on inner-validation? And how does vol-only itself compare
    to plain v4 (R-10's prediction: vol-only should be no better, maybe worse).
    """
    OUT.mkdir(parents=True, exist_ok=True)
    print("inner-validation (spot), matched vol_window=20d, lookback=365d:\n")
    m4, r4 = measure(get_strategy("kelly_regime_v4"), *VALID, market=SPOT)
    line("kelly_regime_v4 (reference, no panic gate)", m4, r4)
    print()
    for s in (0.25, 0.5, 0.75):
        mj, rj = measure(PanicScaledKelly(decline_window_days=90.0, shrink=s,
                                          use_decline=True, **_CENTER), *VALID, market=SPOT)
        line(f"joint    (decline AND high_vol) shrink={s:.2f}", mj, rj)
        mv, rv = measure(PanicScaledKelly(decline_window_days=90.0, shrink=s,
                                          use_decline=False, **_CENTER), *VALID, market=SPOT)
        line(f"vol-only (high_vol alone)       shrink={s:.2f}", mv, rv)
        print(f"    -> joint vs vol-only:  Dfinal={mj.final_balance - mv.final_balance:+,.0f}  "
              f"DDelta={mj.max_drawdown_pct - mv.max_drawdown_pct:+.1f}pp  "
              f"Dsharpe={mj.sharpe - mv.sharpe:+.2f}")
        print(f"    -> vol-only vs v4:     Dfinal={mv.final_balance - m4.final_balance:+,.0f}  "
              f"DDelta={mv.max_drawdown_pct - m4.max_drawdown_pct:+.1f}pp  "
              f"Dsharpe={mv.sharpe - m4.sharpe:+.2f}\n")


# ------------------------------------------------------------------- causality


def causality() -> None:
    """The strict lookahead probe, by hand — experiments get no CI protection.

    Two-opposite-tampers procedure (same as R-28 / matched_risk): bars
    after a cut are multiplied by 3 in one copy and divided by 3 in the
    other, and every decision at or before the cut must be identical. The
    column comparison is what would catch a full-series quantile fit —
    the exact bug class the rolling-tercile threshold has to avoid.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    df = DF.iloc[-400_000:].copy()  # long enough to fill a 730-day tercile lookback
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    configs = [
        ("frozen", FROZEN),
        ("vol-only ablation", {**FROZEN, "use_decline": False}),
    ]
    ok = True
    for tag, cfg in configs:
        def decisions(frame):
            s = PanicScaledKelly(**cfg)
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

        pa = PanicScaledKelly(**cfg).prepare(up.copy())
        pb = PanicScaledKelly(**cfg).prepare(down.copy())
        worst = max(
            float(np.nanmax(np.abs(pa[c].to_numpy(dtype=float)[:cut]
                                   - pb[c].to_numpy(dtype=float)[:cut])))
            for c in ("target", "panic")
        )
        good = not bad and worst < 1e-9
        ok &= good
        print(f"  {tag:20s} orders {'match' if not bad else f'DIFFER at {bad}'}   "
              f"max |column difference| before the cut = {worst:.3e}   "
              f"{'PASS' if good else 'FAIL'}")
    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# --------------------------------------------------------------------- freeze
#
# Pre-registered BEFORE the holdout was read (ROUTINE.md step 4).
#
# Frozen configuration, selected on inner-validation only:
#
#   PanicScaledKelly(decline_window_days=90, vol_window_days=20,
#                     tercile_lookback_days=365, shrink=0.5,
#                     use_decline=True, horizons=(20, 40, 80))
#
# Selection rationale (see `sweep` output): among the 9 joint-panic
# configurations, decline_window=90/shrink=0.5 sits in the middle of a
# flat neighbourhood on inner-validation (2021-2022) rather than at an
# edge — shrink=0.25 (more aggressive) does not further improve drawdown
# over shrink=0.5 by more than path noise, and shrink=0.75 (weaker) gives
# back most of the drawdown benefit. decline_window=60 and 120 either
# side of 90 move final balance/DD by a similar small amount in both
# directions (a plateau, not a peak — see the `neighbours` command,
# P5). vol_window=20 and tercile_lookback=365 are the a-priori "middle of
# the swept range" defaults, not separately tuned peaks: the two
# sensitivity configs at 10/30 and 180/730 days did not produce a
# materially better drawdown/return trade-off than the center point.
#
# Decision rule, fixed now, not to be revisited after the holdout is read
# (mirrors ROUTINE.md's promotion bar plus the panic-specific P3/P4/P5
# added by the task):
#
#   P1  beats buy_and_hold out-of-sample on log growth after real costs
#       (0.10% tier primary, 0.40% tier reported as a stress check).
#   P2  beats kelly_regime_v4 by more than the +/-0.2 Sharpe noise floor
#       on log growth, OR matches its growth while cutting max drawdown
#       by >=10pp.
#   P3  the joint (decline x vol) indicator earns its keep over the
#       vol-only ablation on inner-validation (the `ablation` command) —
#       if it does not, that is reported plainly regardless of what the
#       holdout later shows.
#   P4  replicates on the ETH falsification test: the panic-scaled vs v4
#       ordering (which wins on return, which wins on drawdown) carries
#       the same sign on ETH as on the BTC holdout, and ETH drawdown is
#       no worse than v4's ETH drawdown + 5pp.
#   P5  the parameter neighbourhood around the frozen config is a
#       plateau, not a knife-edge (the `neighbours` command).
#
# PROMOTE-CANDIDATE only if all five hold. Anything else is NEGATIVE,
# written up with the same rigor as a win — including if failure mode
# (a) turns out to dominate (R-10's inverse-leverage effect sinking the
# joint condition down to, or below, vol-only, and vol-only in turn
# losing to plain v4).

FROZEN = dict(decline_window_days=90.0, vol_window_days=20.0,
              tercile_lookback_days=365.0, shrink=0.5, use_decline=True)


# --------------------------------------------------------------------- holdout


def holdout() -> None:
    """Step 4. Evaluate the frozen config once on 2023-01-01 ->, both markets."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for mname, market in MARKETS:
        print(f"\nHOLDOUT 2023-01-01 -> / {mname}")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            m, res = measure(get_strategy(name), *OOS, market=market)
            line(f"  {name}", m, res)
            rows.append({"market": mname, "arm": name, "final": m.final_balance,
                        "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                        "fills": len(res.fills), "fees": m.fees_paid,
                        "liquidated": m.liquidated})
        m, res = measure(PanicScaledKelly(**FROZEN), *OOS, market=market)
        line("  panic_scaled_kelly (frozen)", m, res)
        rows.append({"market": mname, "arm": "panic_scaled_kelly", "final": m.final_balance,
                    "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                    "fills": len(res.fills), "fees": m.fees_paid,
                    "liquidated": m.liquidated})
    pd.DataFrame(rows).to_csv(OUT / "holdout.csv", index=False)
    print(f"\nwritten: {OUT / 'holdout.csv'}")


# -------------------------------------------------------------------- interval


def interval() -> None:
    """Paired stationary block bootstrap on holdout daily returns (R-29/R-30/R-31 method).

    Two comparisons per market: frozen vs buy_and_hold, frozen vs
    kelly_regime_v4.
    """
    from tradebot.inference import (daily_returns, max_drawdown_from_returns,
                                    paired_bootstrap, stationary_bootstrap_indices,
                                    total_log_return)

    rows = []
    for mname, market in MARKETS:
        curves = {}
        for label, strat in (("buy_and_hold", get_strategy("buy_and_hold")),
                             ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                             ("panic_scaled_kelly", PanicScaledKelly(**FROZEN))):
            res = run_period(strat, DF, *OOS, market=market, start_balance=1_000.0,
                             data_label=LABEL)
            curves[label] = daily_returns(res.equity).to_numpy()
        n = len(curves["buy_and_hold"])
        idx = stationary_bootstrap_indices(n, 30.0, 2_000, np.random.default_rng(7))
        print(f"\n{mname}: paired panic_scaled_kelly vs reference on the 2023+ "
              f"holdout ({n} daily observations)")
        for ref in ("buy_and_hold", "kelly_regime_v4"):
            a, b = curves["panic_scaled_kelly"], curves[ref]
            for stat_name, stat in (("Delta log growth", total_log_return),
                                    ("Delta max drawdown (pp)", max_drawdown_from_returns)):
                r = paired_bootstrap(a, b, stat, indices=idx)
                mark = "beats" if r.diff.lo > 0 else ("worse" if r.diff.hi < 0 else "~noise~")
                print(f"  vs {ref:18s} {stat_name:26s} {mark:8s} {r.diff.point:>+8.4f} "
                      f"[{r.diff.lo:>+8.4f}, {r.diff.hi:>+8.4f}]  P(>0)={r.p_positive:.2f}")
                rows.append({"market": mname, "ref": ref, "stat": stat_name,
                            "panic": r.stat_a, "other": r.stat_b, "diff": r.diff.point,
                            "lo": r.diff.lo, "hi": r.diff.hi, "p_positive": r.p_positive,
                            "significant": r.significant})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "intervals.csv", index=False)
    print(f"\nwritten: {OUT / 'intervals.csv'}")


# ------------------------------------------------------------------------ costs


def costs() -> None:
    """Real fee tier check (0.40% Bitstamp entry tier) and funding-charged futures."""
    from tradebot.data import load_funding

    print("HOLDOUT 2023+ spot, at both taker tiers:")
    for tier, label in ((0.001, "0.10% (table assumption)"),
                        (0.004, "0.40% (Bitstamp entry tier)")):
        market = MarketSpec.spot(fee_rate=tier)
        print(f"  {label}")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            line(f"    {name}", *measure(get_strategy(name), *OOS, market=market))
        line("    panic_scaled_kelly (frozen)",
             *measure(PanicScaledKelly(**FROZEN), *OOS, market=market))

    real = load_funding(ROOT / "data")
    if real is None:
        print("\nno funding data present; skipping funding-charged futures check")
        return
    grid = pd.date_range(DF.index[0].ceil("8h"), DF.index[-1], freq="8h", tz="UTC")
    filler = pd.Series(float(real.mean()), index=grid)
    blended = pd.concat([filler[~filler.index.isin(real.index)], real]).sort_index()
    print(f"\nHOLDOUT 2023+ futures 5x with funding CHARGED (real through "
          f"{real.index[-1]:%Y-%m}, mean {real.mean() * 3 * 365.25:+.1%}/yr after):")
    contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                  ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                  ("panic_scaled_kelly", PanicScaledKelly(**FROZEN))]
    lo = int(DF.index.searchsorted(OOS[0]))
    for name, strat in contenders:
        pre = min(lo, strat.warmup)
        raw = run_backtest(strat, DF.iloc[lo - pre:], FUTURES, 1_000.0,
                           trade_start=pre, funding=blended, data_label=LABEL)
        trimmed = replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
        m = compute_metrics(trimmed)
        print(f"  {name:22s} final=${m.final_balance:>9,.0f} "
              f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
              f"funding paid=${raw.funding_paid:>7,.0f}"
              f"{'  LIQUIDATED' if m.liquidated else ''}")


# --------------------------------------------------------------------- windows


def windows(trials: int = 10, seed: int = 42) -> None:
    """Path sensitivity: identical random windows, frozen strategy vs v4 (paired)."""
    contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                  ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                  ("panic_scaled_kelly", PanicScaledKelly(**FROZEN))]
    warmup = max(s.warmup for _, s in contenders) + 10
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(trials):
        length = int(rng.integers(90, 731) * 288)
        specs.append((int(rng.integers(warmup, len(DF) - length)), length))

    rows = []
    for k, (start, length) in enumerate(specs, 1):
        window = DF.iloc[start - warmup: start + length]
        for mname, market in MARKETS:
            for name, strat in contenders:
                res = run_backtest(strat, window, market, 1_000.0,
                                   trade_start=warmup, data_label=LABEL)
                eq = res.equity.to_numpy(dtype=float)
                base, seg = eq[warmup], eq[warmup:]
                ok = np.isfinite(base) and base > 0
                rows.append({"trial": k, "market": mname, "strategy": name,
                            "return_pct": 100.0 * (seg[-1] / base - 1.0) if ok else -100.0,
                            "max_dd_pct": max_drawdown_pct(seg) if ok else 100.0,
                            "liquidated": res.liquidated})
        print(f"[{k}/{trials}]", end=" ", flush=True, file=sys.stderr)
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "windows.csv", index=False)

    print(f"\n{trials} random windows (90-730 days), identical across strategies:\n")
    for mname, _ in MARKETS:
        print(f"  {mname}:")
        sub = res[res.market == mname]
        bench = sub[sub.strategy == "buy_and_hold"].set_index("trial")["return_pct"]
        for name, _ in contenders:
            g = sub[sub.strategy == name].set_index("trial")
            print(f"    {name:22s} median return {g.return_pct.median():>+8.1f}%  "
                  f"median DD {g.max_dd_pct.median():>5.1f}%  "
                  f"worst DD {g.max_dd_pct.max():>5.1f}%  "
                  f"beat hold {(g['return_pct'] > bench).mean():>5.0%}  "
                  f"liq {g.liquidated.mean():>4.0%}")
        a = sub[sub.strategy == "panic_scaled_kelly"].set_index("trial")
        b = sub[sub.strategy == "kelly_regime_v4"].set_index("trial")
        d_ret = (a.return_pct - b.return_pct).dropna()
        d_dd = (a.max_dd_pct - b.max_dd_pct).dropna()
        print(f"    paired panic - v4: return median {d_ret.median():+.1f}pp, panic higher in "
              f"{(d_ret > 0).mean():.0%};  DD median {d_dd.median():+.1f}pp, "
              f"panic deeper in {(d_dd > 0).mean():.0%}")
        print()


# ------------------------------------------------------------------------- eth


def eth() -> None:
    """Pre-registered falsification: does the BTC-holdout ordering replicate on ETH?

    Same venue (Bitfinex), same window as R-17/R-28/R-31 (2016-03 -> 2019-12).
    The panic mechanism needs no funding or external data, so it runs
    unmodified on both assets.
    """
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            for name, strat in (("buy_and_hold", get_strategy("buy_and_hold")),
                                ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                                ("panic_scaled_kelly", PanicScaledKelly(**FROZEN))):
                m, res = measure(strat, None, None, df=df, market=market)
                line(f"    {name}", m, res)


# -------------------------------------------------------------------- neighbours


def neighbours() -> None:
    """P5: is the frozen config a plateau on inner-validation, not a knife-edge?"""
    print("inner-validation (spot), one knob moved at a time around the frozen config:\n")
    base = FROZEN
    m0, r0 = measure(PanicScaledKelly(**base), *VALID, market=SPOT)
    line("frozen", m0, r0)
    for key, values in (
        ("decline_window_days", (60.0, 90.0, 120.0)),
        ("vol_window_days", (10.0, 20.0, 30.0)),
        ("tercile_lookback_days", (180.0, 365.0, 730.0)),
        ("shrink", (0.25, 0.5, 0.75)),
    ):
        print(f"\n  varying {key}:")
        for v in values:
            cfg = {**base, key: v}
            m, r = measure(PanicScaledKelly(**cfg), *VALID, market=SPOT)
            mark = " <- frozen" if cfg == base else ""
            line(f"    {key}={v}", m, r)


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"sweep": sweep, "ablation": ablation, "causality": causality,
            "holdout": holdout, "interval": interval, "costs": costs,
            "windows": windows, "eth": eth, "neighbours": neighbours}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_panic_scaling.py [{'|'.join(cmds)}]")
