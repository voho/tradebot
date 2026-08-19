#!/usr/bin/env python
"""Driver for backlog B-05 -- funding gate on kelly_regime_v4.

Splits follow ROUTINE.md step 3::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              NOT TOUCHED by this file

This driver deliberately contains no holdout command: step 4 is the
operator's job, after the decision rule below is pre-registered.

Usage::

    python experiments/run_funding_gate.py causality   # by-hand lookahead probe
    python experiments/run_funding_gate.py sweep        # step 3, inner-train search
    python experiments/run_funding_gate.py validate      # inner-validation selection
    python experiments/run_funding_gate.py neighbours    # plateau check around the pick
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from experiments.funding_gate import FUNDING, FundingGateV4  # noqa: E402
from scripts.experiment import FUTURES, SPOT, ev  # noqa: E402

TRAIN_END = "2020-12-31"
VALID = ("2021-01-01", "2022-12-31")

N_EVALUATED = 0  # distinct (settlement_span, funding_quantile, haircut) configs


def _ev(strategy, market, tag, **kwargs):
    return ev(strategy, market=market, tag=tag, **kwargs)


def baselines() -> None:
    """kelly_regime_v4 and buy_and_hold, same windows, for comparison."""
    from tradebot.registry import get_strategy

    print("baselines, funding-free (the ev() harness never charges funding):")
    for name in ("kelly_regime_v4", "buy_and_hold"):
        for market, mtag in ((SPOT, "spot"), (FUTURES, "fut5x")):
            _ev(get_strategy(name), market, f"{name:16s} {mtag:6s} train", end=TRAIN_END)
            _ev(get_strategy(name), market, f"{name:16s} {mtag:6s} valid",
                start=VALID[0], end=VALID[1])


def causality() -> None:
    """By-hand lookahead probe -- experiments/ gets no CI protection.

    Two checks, both against the FULL committed spot frame:

    1. Truncation: prepare() on df.iloc[:N] must match prepare() on the
       full frame for every row < N - warmup. Catches an i+1 peek.
    2. Two-opposite-tampers: multiply/divide all bars strictly AFTER a cut
       by 3 (price) and 7 (volume) in two copies; every column value at or
       before the cut must be identical between the two copies. Catches a
       full-series statistic (a scaler/quantile/mean/std fit once over the
       whole series and applied to early rows) that the truncation test
       alone cannot see, because that class of bug does not care how long
       the frame is -- only that later rows were read at all.
    """
    from tradebot.data import load_dataset

    DF, _ = load_dataset(ROOT / "data", "spot")
    strategy = FundingGateV4()

    # --- 1. truncation test
    N = 900_000
    warmup = strategy.warmup
    full = strategy.prepare(DF.copy())
    trunc = FundingGateV4().prepare(DF.iloc[:N].copy())
    cols = ["target", "funding_gate_active", "funding_haircut"]
    ok = True
    for c in cols:
        a = full[c].to_numpy()[: N - warmup]
        b = trunc[c].to_numpy()[: N - warmup]
        if c == "funding_gate_active":
            same = np.array_equal(a, b)
        else:
            same = np.allclose(np.nan_to_num(a.astype(float)),
                               np.nan_to_num(b.astype(float)), atol=1e-12)
        ok &= same
        print(f"  truncation[{c:22s}] rows<{N - warmup:,} identical: {same}")
    print(f"  truncation test: {'PASS' if ok else 'FAIL'}\n")

    # --- 2. two-opposite-tampers
    tail = DF.iloc[-300_000:].copy()
    cut = len(tail) - 5_000
    up, down = tail.copy(), tail.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    pa = FundingGateV4().prepare(up.copy())
    pb = FundingGateV4().prepare(down.copy())
    ok2 = True
    for c in cols:
        a = pa[c].to_numpy()[:cut].astype(float)
        b = pb[c].to_numpy()[:cut].astype(float)
        worst = float(np.nanmax(np.abs(np.nan_to_num(a) - np.nan_to_num(b))))
        good = worst < 1e-9
        ok2 &= good
        print(f"  tamper[{c:22s}] max |diff| before cut {cut:,}: {worst:.3e}  "
              f"{'PASS' if good else 'FAIL'}")
    print(f"  tamper test: {'PASS' if ok2 else 'FAIL'}\n")

    print(f"OVERALL CAUSALITY CHECK: {'PASS' if ok and ok2 else 'FAIL'}")


# ------------------------------------------------------------------- sweep


GRID_SPAN = (3.0, 5.0, 10.0)
GRID_QUANTILE = (0.80, 0.85, 0.90, 0.95)
GRID_HAIRCUT = (0.0, 0.3, 0.5)


def _configs() -> list[dict]:
    """The 8-point search: span sweep, then quantile sweep, then haircut sweep,
    each stage fixing the winner of the previous one at haircut=0.0 (hard gate)
    -- one-knob-at-a-time, matching the R-28/R-31 convention in this repo,
    not a full 3x4x3 factorial.
    """
    configs = []
    seen = set()

    def add(span, q, h):
        key = (span, q, h)
        if key not in seen:
            seen.add(key)
            configs.append({"settlement_span": span, "funding_quantile": q, "haircut": h})

    for span in GRID_SPAN:
        add(span, 0.90, 0.0)
    for q in GRID_QUANTILE:
        add(5.0, q, 0.0)
    for h in GRID_HAIRCUT:
        add(5.0, 0.90, h)
    return configs


def sweep() -> None:
    """Step 3. Inner-train only, both markets, one-knob-at-a-time search."""
    global N_EVALUATED
    print(f"funding data: {len(FUNDING):,} settlements "
          f"{FUNDING.index.min()} -> {FUNDING.index.max()}\n")
    configs = _configs()
    N_EVALUATED = len(configs)
    print(f"{len(configs)} distinct configurations, inner-train "
          f"(end={TRAIN_END}), both markets:\n")
    for cfg in configs:
        tag = (f"span={cfg['settlement_span']:g} q={cfg['funding_quantile']:.2f} "
               f"hc={cfg['haircut']:.1f}")
        _ev(FundingGateV4(**cfg), SPOT, f"{tag:30s} spot ", end=TRAIN_END)
        _ev(FundingGateV4(**cfg), FUTURES, f"{tag:30s} fut5x", end=TRAIN_END)
    print(f"\nconfigurations evaluated this phase: {N_EVALUATED}")


def validate() -> None:
    """Step 3. Same configs, inner-validation, both markets -- for selection."""
    configs = _configs()
    print(f"{len(configs)} configurations, inner-validation "
          f"({VALID[0]} -> {VALID[1]}), both markets:\n")
    for cfg in configs:
        tag = (f"span={cfg['settlement_span']:g} q={cfg['funding_quantile']:.2f} "
               f"hc={cfg['haircut']:.1f}")
        _ev(FundingGateV4(**cfg), SPOT, f"{tag:30s} spot ", start=VALID[0], end=VALID[1])
        _ev(FundingGateV4(**cfg), FUTURES, f"{tag:30s} fut5x", start=VALID[0], end=VALID[1])


NEIGHBOUR_CONFIGS = [
    {"settlement_span": 3.0, "funding_quantile": 0.80, "haircut": 0.0},
    {"settlement_span": 10.0, "funding_quantile": 0.80, "haircut": 0.0},
    {"settlement_span": 5.0, "funding_quantile": 0.75, "haircut": 0.0},
    {"settlement_span": 5.0, "funding_quantile": 0.80, "haircut": 0.3},
    {"settlement_span": 5.0, "funding_quantile": 0.80, "haircut": 0.5},
]


def neighbours() -> None:
    """Plateau check around the span=5, q=0.80, hc=0.0 pick -- both windows."""
    for start, end, label in ((None, TRAIN_END, "inner-train"),
                               (VALID[0], VALID[1], "inner-validation")):
        print(f"\n{label}:")
        for cfg in NEIGHBOUR_CONFIGS:
            tag = (f"span={cfg['settlement_span']:g} q={cfg['funding_quantile']:.2f} "
                   f"hc={cfg['haircut']:.1f}")
            _ev(FundingGateV4(**cfg), SPOT, f"{tag:30s} spot ", end=end, start=start)
            _ev(FundingGateV4(**cfg), FUTURES, f"{tag:30s} fut5x", end=end, start=start)
    print(f"\nconfigurations evaluated this phase: {len(NEIGHBOUR_CONFIGS)}")


COMMANDS = {"baselines": baselines, "causality": causality,
            "sweep": sweep, "validate": validate, "neighbours": neighbours}


if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate.py [{'|'.join(COMMANDS)}]")
