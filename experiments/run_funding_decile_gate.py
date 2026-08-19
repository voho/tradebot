#!/usr/bin/env python
"""Driver for backlog B-05 - funding as a gate on kelly_regime_v4.

Data constraint (read this before the numbers): the real Binance BTCUSDT
funding series (data/btcusdt_perp_funding_8h.csv.gz) covers only
2020-01-01 -> 2023-12-31 19:00 UTC (4,383 settlements). That is much
shorter than the project's usual inner-train/inner-validation/holdout
split, so the splits used here are:

    inner-train (partial)  2020-01-01 -> 2020-12-31   fit, sweep, iterate
    inner-validation        2021-01-01 -> 2022-12-31   select between variants
    holdout SLICE           2023-01-01 -> 2023-12-31   step 4 only, pre-registered

This is NOT the project's usual 2023-2026 holdout - funding data does not
extend past 2023, so "holdout" here means an eleven-and-a-half-month
slice of 2023 only. Treat every holdout number in this file as resting on
that much data and no more.

Every comparison charges the SAME real funding series to both the gated
variant and the kelly_regime_v4 baseline (``funding=FUNDING`` on both), and
reports spot buy_and_hold (which pays no funding) as the ultimate
benchmark, per docs/ROUTINE.md step 4 and the task brief.

Usage::

    python experiments/run_funding_decile_gate.py inspect       # what the gate does
    python experiments/run_funding_decile_gate.py sweep         # inner-train + inner-val, 3 variants
    python experiments/run_funding_decile_gate.py neighbours    # plateau check
    python experiments/run_funding_decile_gate.py causality     # by-hand lookahead check
    python experiments/run_funding_decile_gate.py falsification # pre-registered MC-window test
    python experiments/run_funding_decile_gate.py holdout       # step 4, frozen config, run ONCE
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

from experiments.funding_decile_gate import FundingDecileGate  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
FUNDING = load_funding(ROOT / "data")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

FUNDING_START = FUNDING.index[0]
FUNDING_END = FUNDING.index[-1]

TRAIN = ("2020-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS = ("2023-01-01", "2023-12-31")

N_EVALUATED = 0  # every distinct configuration scored, for the trials count


def make(kwargs, funding=None):
    return FundingDecileGate(funding=FUNDING if funding is None else funding, **kwargs)


def _period(strategy, market, start=None, end=None, funding=None, df=None,
            balance=1_000.0):
    """funding_study.py's ``_period`` pattern: warm on the bars before the
    window, trade only inside it, so a fresh $1,000 account measures the
    window and not a corpse or a handicapped-by-warmup strategy (R-22)."""
    frame = DF if df is None else df
    lo = 0 if start is None else int(frame.index.searchsorted(start))
    hi = len(frame) if end is None else int(frame.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, frame.iloc[lo - pre: hi], market, balance,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw.funding_paid, raw


def ev(strategy, start, end, market=FUTURES, funding=FUNDING, tag="", count=True):
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    m, funding_paid, raw = _period(strategy, market, start, end, funding=funding)
    print(f"  {tag or strategy.name:32s} {market.name:11s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"fills={len(raw.fills):>5d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} funding=${funding_paid:>8,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")
    return m, funding_paid, raw


# ------------------------------------------------------------------- variants
#
# Three pre-designed variants (step 2). Mechanism and falsification test are
# written down here, before any inner-train/inner-validation number below was
# read for the first time in this session.
#
# V1 HARD30    lookback=90 settlements (30d), threshold=0.90 (top decile),
#              haircut=0.0 (fully flat). Mechanism: rich funding IS the
#              adverse-timing cost R-14 measured; stand flat exactly there.
# V2 HAIRCUT30 same lookback/threshold, haircut=0.35 instead of 0.0.
#              Mechanism: a hard flip to flat may be a costly overreaction
#              if the top-decile signal is noisy (R-16's own text calls the
#              middle quintiles non-monotone); a partial cut trades away
#              some protection for less whipsaw and some upside capture.
# V3 HARD90    lookback=270 settlements (90d), threshold=0.90, haircut=0.0.
#              Mechanism: same hard gate as V1 but reacting to a slower
#              notion of "rich" - tests whether V1's result (if any) is a
#              property of "funding in the top decile" or an artifact of a
#              fast 30-day window that happens to fire often near 2021's
#              blow-off top.
#
# Pre-registered falsification test (ONE, chosen now, applied to all three):
# 20 Monte Carlo windows (90-300 days, uniform random start), drawn ONLY from
# within the funding-covered span (2020-04-01, after enough lookback to warm
# the gate, through 2023-12-23), identical windows for the variant and the
# kelly_regime_v4 baseline, BOTH charged the same real funding. FALSIFIED if
# the variant does not beat the v4 baseline on max drawdown in more than 50%
# of the paired windows - i.e. the single-path holdout result (if favourable)
# does not generalize even within the narrow span where funding is real.
VARIANTS = {
    "V1 HARD30":    dict(lookback_settlements=90,  threshold=0.90, haircut=0.00),
    "V2 HAIRCUT30": dict(lookback_settlements=90,  threshold=0.90, haircut=0.35),
    "V3 HARD90":    dict(lookback_settlements=270, threshold=0.90, haircut=0.00),
}


# ------------------------------------------------------------------- inspect


def inspect() -> None:
    """What the gate actually does, before any backtest is trusted."""
    print(f"real funding: {len(FUNDING):,} settlements  "
          f"{FUNDING_START:%Y-%m-%d %H:%M} -> {FUNDING_END:%Y-%m-%d %H:%M} UTC\n")

    for tag, kw in VARIANTS.items():
        s = make(kw)
        prepared = s.prepare(DF.loc["2019-10-01":"2023-12-31"].copy())
        gate = prepared["funding_gate"]
        in_window = prepared.loc[FUNDING_START:FUNDING_END, "funding_gate"]
        print(f"{tag}  lookback={kw['lookback_settlements']} settlements "
              f"threshold={kw['threshold']} haircut={kw['haircut']}")
        print(f"  fraction of bars gated (within funding span): "
              f"{(in_window < 1.0).mean():.1%}")
        print(f"  fraction of bars fully open (==1.0):           "
              f"{(in_window >= 1.0).mean():.1%}")
        print(f"  gate value distribution: min={gate.min():.2f} "
              f"mean={gate.mean():.3f} max={gate.max():.2f}\n")

    # Overlap between "v4 wants to be long" and "gate says stand down" - the
    # whole premise of B-05 is that these two conditions coincide.
    v4 = get_strategy("kelly_regime_v4")
    v4_prepared = v4.prepare(DF.loc["2019-10-01":"2023-12-31"].copy())
    gated = make(VARIANTS["V1 HARD30"]).prepare(DF.loc["2019-10-01":"2023-12-31"].copy())
    both = pd.DataFrame({
        "v4_target": v4_prepared.loc[FUNDING_START:FUNDING_END, "target"],
        "gate": gated.loc[FUNDING_START:FUNDING_END, "funding_gate"],
    }).dropna()
    v4_long = both["v4_target"] > 1e-9
    gate_shut = both["gate"] < 1.0
    print("V1 HARD30 gate vs v4's own (price-only) exposure, within the funding span:")
    print(f"  v4 wants to be long:              {v4_long.mean():.1%} of bars")
    print(f"  gate shut (>=90th pctile funding): {gate_shut.mean():.1%} of bars")
    print(f"  BOTH (gate binds on a live position): {(v4_long & gate_shut).mean():.1%} of bars")
    print(f"  gate shut while v4 already flat (no-op): "
          f"{(gate_shut & ~v4_long).mean():.1%} of bars")
    print("\nThe last line is the pre-registered failure mode: if the gate mostly")
    print("fires while v4 is already flat, it cannot change exposure and the")
    print("idea is a no-op regardless of what the signal says.")


# --------------------------------------------------------------------- sweep


def _benchmarks(start, end, label):
    print(f"\n{label} benchmarks:")
    ev(get_strategy("buy_and_hold"), start, end, market=SPOT, funding=None,
       tag="buy_and_hold (spot, no funding)", count=False)
    ev(get_strategy("kelly_regime_v4"), start, end, market=FUTURES, funding=None,
       tag="kelly_regime_v4 (funding-FREE, upper bound)", count=False)
    ev(get_strategy("kelly_regime_v4"), start, end, market=FUTURES, funding=FUNDING,
       tag="kelly_regime_v4 (real funding charged)", count=False)


def sweep() -> None:
    for (start, end), split in ((TRAIN, "INNER-TRAIN (2020, partial)"),
                                (VALID, "INNER-VALIDATION (2021-2022)")):
        _benchmarks(start, end, split)
        print(f"{split} variants (real funding charged to both arms):")
        for tag, kw in VARIANTS.items():
            ev(make(kw), start, end, market=FUTURES, funding=FUNDING, tag=tag)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


# ---------------------------------------------------------------- neighbours


def neighbours() -> None:
    """Plateau, not peak: vary threshold and lookback around V1's selection."""
    base = dict(lookback_settlements=90, threshold=0.90, haircut=0.00)
    grid = [("base V1 (30d, p90, flat)", {})]
    grid += [(f"threshold={t}", dict(threshold=t)) for t in (0.80, 0.85, 0.95)]
    grid += [(f"lookback={lb}stl(~{lb/3:.0f}d)", dict(lookback_settlements=lb))
             for lb in (21, 45, 135, 270)]  # ~7d, ~15d, ~45d, ~90d (3 settlements/day)
    grid += [(f"haircut={h}", dict(haircut=h)) for h in (0.20, 0.50)]
    print("INNER-VALIDATION neighbourhood (2021-2022):")
    for tag, kw in grid:
        ev(make({**base, **kw}), *VALID, market=FUTURES, funding=FUNDING, tag=tag)
    print("\nINNER-TRAIN neighbourhood (2020, partial), not counted twice:")
    for tag, kw in grid:
        ev(make({**base, **kw}), *TRAIN, market=FUTURES, funding=FUNDING, tag=tag,
           count=False)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


def neighbours_v3() -> None:
    """Plateau check around the ACTUAL frozen candidate, V3 HARD90 (lb=270stl~90d)."""
    base = dict(lookback_settlements=270, threshold=0.90, haircut=0.00)
    grid = [("base V3 (90d, p90, flat)", {})]
    grid += [(f"threshold={t}", dict(threshold=t)) for t in (0.80, 0.85, 0.95)]
    grid += [(f"lookback={lb}stl(~{lb/3:.0f}d)", dict(lookback_settlements=lb))
             for lb in (180, 225, 315, 360)]  # ~60d, ~75d, ~105d, ~120d
    grid += [(f"haircut={h}", dict(haircut=h)) for h in (0.20, 0.50)]
    print("INNER-VALIDATION neighbourhood around V3 (2021-2022):")
    for tag, kw in grid:
        ev(make({**base, **kw}), *VALID, market=FUTURES, funding=FUNDING, tag=tag)
    print("\nINNER-TRAIN neighbourhood around V3 (2020, partial), not counted twice:")
    for tag, kw in grid:
        ev(make({**base, **kw}), *TRAIN, market=FUTURES, funding=FUNDING, tag=tag,
           count=False)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


# ----------------------------------------------------------------- causality


def causality() -> None:
    """The strict on_bar peek check, run by hand (R-28's precedent).

    Bars after a cut point are multiplied by 3 in one copy and divided by 3
    in the other; every decision at or before the cut must be identical.
    The funding series itself is untouched by this test (it is not part of
    the OHLCV frame at all), so this specifically checks that v4's inherited
    price logic AND the gate's bar alignment do not leak the future.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    df = DF.loc["2019-10-01":"2023-12-31"].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    kw = VARIANTS["V1 HARD30"]

    def decisions(frame):
        s = make(kw)
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
    print(f"tampered from bar {cut:,} of {len(df):,}; checked bars {bars}")
    print("FAIL - reads the future at bars " + str(bad) if bad
          else "PASS - every decision at or before the cut is unchanged")

    pa = make(kw).prepare(up.copy())
    pb = make(kw).prepare(down.copy())
    for col in ("target", "funding_gate"):
        diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
        worst = float(np.nanmax(diff))
        print(f"  column {col:14s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if worst < 1e-12 else 'FAIL'}")

    print("\n(the funding_gate column depends only on the fixed, external real")
    print(" funding series and each bar's own timestamp - not on price at all -")
    print(" so it is expected to be bit-identical everywhere, including after")
    print(" the cut; the informative part of this check is the `target` column,")
    print(" which also depends on v4's inherited price-based anchors/vol.)")


# -------------------------------------------------------------- falsification


def falsification(trials: int = 20, seed: int = 20260819) -> None:
    """Pre-registered test: MC windows within the funding-covered span only."""
    from tradebot.metrics import max_drawdown_pct

    lo = int(DF.index.searchsorted("2020-04-01"))  # let the 30/90d lookback warm
    hi = int(DF.index.searchsorted(FUNDING_END, side="right"))
    warmup = get_strategy("kelly_regime_v4").warmup + 10

    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(trials):
        length = int(rng.integers(90, 301) * 288)
        start = int(rng.integers(lo + warmup, max(lo + warmup + 1, hi - length)))
        specs.append((start, length))

    rows = []
    for k, (start, length) in enumerate(specs, 1):
        window = DF.iloc[start - warmup: start + length]
        for tag, strat in (("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                           *[(vtag, make(kw)) for vtag, kw in VARIANTS.items()]):
            res = run_backtest(strat, window, FUTURES, 1_000.0, trade_start=warmup,
                               funding=FUNDING, data_label=LABEL)
            eq = res.equity.to_numpy(dtype=float)
            base, seg = eq[warmup], eq[warmup:]
            ok = np.isfinite(base) and base > 0
            rows.append({"trial": k, "strategy": tag,
                         "return_pct": 100.0 * (seg[-1] / base - 1.0) if ok else -100.0,
                         "max_dd_pct": max_drawdown_pct(seg) if ok else 100.0,
                         "liquidated": res.liquidated})
        print(f"[{k}/{trials}]", end=" ", flush=True, file=sys.stderr)
    res = pd.DataFrame(rows)
    print()

    print(f"\n{trials} random windows (90-300 days) within the funding span "
          f"(2020-04-01 .. {FUNDING_END:%Y-%m-%d}), real funding charged to all arms:\n")
    base_v4 = res[res.strategy == "kelly_regime_v4"].set_index("trial")
    for tag in list(VARIANTS) + ["kelly_regime_v4"]:
        g = res[res.strategy == tag].set_index("trial")
        print(f"  {tag:16s} median return {g.return_pct.median():>+8.1f}%  "
              f"median DD {g.max_dd_pct.median():>5.1f}%  "
              f"worst DD {g.max_dd_pct.max():>5.1f}%  liq {g.liquidated.mean():>4.0%}")

    print("\npaired vs kelly_regime_v4 (same windows, both funding-charged):")
    for tag in VARIANTS:
        g = res[res.strategy == tag].set_index("trial")
        dd = (g["max_dd_pct"] - base_v4["max_dd_pct"]).dropna()
        ret = (g["return_pct"] - base_v4["return_pct"]).dropna()
        shallower = (dd < 0).mean()
        print(f"  {tag:16s} DD shallower than v4 in {shallower:>5.0%} of windows "
              f"(median Delta DD {dd.median():+.1f}pp); "
              f"return higher in {(ret > 0).mean():>5.0%} of windows "
              f"(median Delta return {ret.median():+.1f}pp)  "
              f"-> {'PASS' if shallower > 0.50 else 'FALSIFIED'} "
              f"(pre-registered bar: shallower DD in >50% of windows)")


# ------------------------------------------------------------------- holdout
#
# ############################################################################
# STEP 4 PRE-REGISTRATION - written and frozen BEFORE this function was run
# against the 2023 holdout slice. See the final report for the verbatim text
# committed here; this comment is that same text, kept next to the code it
# governs.
#
# Selection (made on inner-train 2020 + inner-validation 2021-2022, BEFORE
# any holdout bar was read): among the three pre-registered variants, V3
# HARD90 dominates on every metric in BOTH inner splits - inner-train final
# $2,387 (best of the three) / DD 17.0% (best) / Sharpe 2.72 (best);
# inner-validation final $1,358 (best) / DD 23.6% (worst of the three by
# 2.5pp) / Sharpe 0.75 (best). It is not a clean sweep (its inner-validation
# drawdown is the one metric V1 wins), but it wins 5 of 6 cells and is the
# only variant that is never the worst performer in either split. V3 HARD90
# is therefore the frozen candidate; V1 and V2 are carried to the holdout
# for full disclosure (ROUTINE.md: every branch reports) but are not the
# promotion candidate.
#
# Frozen configuration: V3 HARD90 - lookback_settlements=270 (~90 real
# days), threshold=0.90 (top decile), haircut=0.00 (fully flat), all other
# kwargs at kelly_regime_v4's shipped defaults (horizons=(20,40,80),
# target_vol=0.55, max_leverage=2.0, deadband=0.10, vol_span=8d,
# anchor_span_days=180, high_in/out=1.70/1.20, low_in/out=0.55/0.85).
#
# Decision rule (default REJECT unless ALL hold), evaluated on 2023-01-01 ..
# 2023-12-31 (bounded by funding data availability, NOT the project's usual
# 2023-2026 holdout):
#   P1 - the FUTURES final balance of V3 HARD90 (real funding charged) beats
#        kelly_regime_v4's FUTURES final balance (real funding charged, same
#        series) AND beats spot buy_and_hold (which pays no funding).
#   P2 - the improvement over kelly_regime_v4 exceeds the +/-0.2 Sharpe noise
#        floor (R-20), OR is a drawdown improvement of >= 10 percentage
#        points relative to kelly_regime_v4 (both funding-charged).
#   P3 - survives the pre-registered falsification test above (shallower DD
#        than v4 in >50% of the 20 MC windows drawn from the funding span).
#   P4 - the neighbourhood (already swept above) is a plateau, not a single
#        lucky spike.
# If P1 fails, the round is NEGATIVE regardless of P2-P4.
# ############################################################################


def holdout() -> None:
    """Step 4. Run exactly once. Config is frozen above; do not edit after reading."""
    print(f"HOLDOUT SLICE 2023-01-01 -> 2023-12-31 "
          f"(bounded by real funding data through {FUNDING_END:%Y-%m-%d})\n")
    ev(get_strategy("buy_and_hold"), *OOS, market=SPOT, funding=None,
       tag="buy_and_hold (spot, pays no funding)", count=False)
    ev(get_strategy("kelly_regime_v4"), *OOS, market=FUTURES, funding=None,
       tag="kelly_regime_v4 (funding-FREE, upper bound)", count=False)
    ev(get_strategy("kelly_regime_v4"), *OOS, market=FUTURES, funding=FUNDING,
       tag="kelly_regime_v4 (real funding, baseline)", count=False)
    for tag, kw in VARIANTS.items():
        ev(make(kw), *OOS, market=FUTURES, funding=FUNDING,
           tag=f"{tag} (real funding, frozen)", count=False)


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    cmds = {"inspect": inspect, "sweep": sweep, "neighbours": neighbours,
            "neighbours_v3": neighbours_v3, "causality": causality,
            "falsification": falsification, "holdout": holdout}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_funding_decile_gate.py [{'|'.join(cmds)}]")
