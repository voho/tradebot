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


# ---------------------------------------------------------------- step 4: holdout


def holdout() -> None:
    """Step 4. Frozen config only. R-33 decision rule (docs/LEDGER.md), checks P1-P5."""
    from dataclasses import replace

    import numpy as np
    import pandas as pd

    from tradebot.data import load_dataset, load_funding
    from tradebot.engine import run_backtest
    from tradebot.inference import (daily_returns, deflation_breakeven_sd,
                                     max_drawdown_from_returns, paired_bootstrap,
                                     stationary_bootstrap_indices, total_log_return)
    from tradebot.metrics import compute_metrics
    from tradebot.registry import get_strategy

    OOS_START = "2023-01-01"
    FROZEN = dict(settlement_span=5.0, funding_quantile=0.80, haircut=0.3)

    DF, LABEL = load_dataset(ROOT / "data", "spot")
    REAL = load_funding(ROOT / "data")

    def blended_from(real: "pd.Series") -> "pd.Series":
        """Real settlements where observed, the empirical mean everywhere else
        (matches scripts/funding_study.py's fullperiod() convention)."""
        mean = float(real.mean())
        grid = pd.date_range(DF.index[0].ceil("8h"), DF.index[-1], freq="8h", tz="UTC")
        filler = pd.Series(mean, index=grid)
        filler = filler[~filler.index.isin(real.index)]
        return pd.concat([filler, real]).sort_index()

    BLENDED = blended_from(REAL)

    def period(strategy, market, start=None, end=None, funding=None):
        lo = 0 if start is None else int(DF.index.searchsorted(start))
        hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
        pre = min(lo, strategy.warmup)
        raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                            trade_start=pre, funding=funding, data_label=LABEL)
        trimmed = (raw if pre == 0 else
                   replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
        return compute_metrics(trimmed), raw

    strategies = {
        "buy_and_hold": lambda: get_strategy("buy_and_hold"),
        "kelly_regime_v4": lambda: get_strategy("kelly_regime_v4"),
        "funding_gate_v4 (frozen)": lambda: FundingGateV4(**FROZEN),
    }

    print("=" * 78)
    print("P4 / base comparison -- funding-free, 0.10% spot / 0.05% futures, holdout")
    print("=" * 78)
    funding_free = {}
    for label, make in strategies.items():
        for mname, market in (("spot", SPOT), ("fut5x", FUTURES)):
            m, raw = period(make(), market, start=OOS_START)
            funding_free[(label, mname)] = (m, raw)
            print(f"  {label:26s} {mname:6s} final=${m.final_balance:>10,.0f} "
                  f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
                  f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>4d} "
                  f"{'LIQ' if m.liquidated else ''}")

    print("\n" + "=" * 78)
    print("P1 / cost -- futures, funding CHARGED (real through 2023-12, blended after)")
    print("=" * 78)
    funding_charged = {}
    for label, make in strategies.items():
        if label == "buy_and_hold":
            continue
        m, raw = period(make(), FUTURES, start=OOS_START, funding=BLENDED)
        funding_charged[label] = (m, raw)
        print(f"  {label:26s} fut5x  final=${m.final_balance:>10,.0f} "
              f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
              f"sharpe={m.sharpe:>5.2f} funding_paid=${raw.funding_paid:>8,.0f} "
              f"{'LIQ' if m.liquidated else ''}")

    print("\n" + "=" * 78)
    print("P5 / falsification -- 0.40% taker fee tier, spot, funding-free, holdout")
    print("=" * 78)
    fee40 = replace(SPOT, fee_rate=0.004)
    high_fee = {}
    for label, make in strategies.items():
        m, raw = period(make(), fee40, start=OOS_START)
        high_fee[label] = (m, raw)
        print(f"  {label:26s} spot@0.40% final=${m.final_balance:>10,.0f} "
              f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
              f"sharpe={m.sharpe:>5.2f}")

    print("\n" + "=" * 78)
    print("Paired block bootstrap: funding_gate_v4 (frozen) - kelly_regime_v4, holdout")
    print("30-day mean block, 2000 resamples, identical resample both arms")
    print("=" * 78)
    N_TRIALS_ROUND = 37       # R-33 (13) + R-34 (24), this parallel round
    N_TRIALS_PROJECT = 172 + N_TRIALS_ROUND   # R-32's running total + this round
    results = {}
    for mname, market in (("spot", SPOT), ("fut5x", FUTURES)):
        res_a = period(FundingGateV4(**FROZEN), market, start=OOS_START)[1]
        res_b = period(get_strategy("kelly_regime_v4"), market, start=OOS_START)[1]
        a = daily_returns(res_a.equity).to_numpy()
        b = daily_returns(res_b.equity).to_numpy()
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        idx = stationary_bootstrap_indices(n, 30.0, 2_000, np.random.default_rng(33))
        print(f"\n{mname} ({n} daily observations):")
        for stat_name, stat in (("Δ log growth", total_log_return),
                                 ("Δ max drawdown (pp)", max_drawdown_from_returns)):
            r = paired_bootstrap(a, b, stat, indices=idx)
            mark = "beats" if r.diff.lo > 0 else ("worse" if r.diff.hi < 0 else "~"
                   if stat_name.startswith("Δ log") else
                   ("cuts DD" if r.diff.lo > 0 else ("worse DD" if r.diff.hi < 0 else "~")))
            print(f"  {stat_name:22s} gate={r.stat_a:+.4f} v4={r.stat_b:+.4f} "
                  f"diff={r.diff.point:>+7.4f} [{r.diff.lo:>+7.4f}, {r.diff.hi:>+7.4f}] "
                  f"P(gate>v4)={r.p_positive:.2f}  {mark}")
            results[(mname, stat_name)] = r
        m_a = compute_metrics(res_a)
        skew = float(pd.Series(a).skew())
        kurt = float(pd.Series(a).kurtosis())
        breakeven = deflation_breakeven_sd(m_a.sharpe, n, skew, kurt, N_TRIALS_PROJECT)
        print(f"  gate Sharpe={m_a.sharpe:.2f}  deflation breakeven sd "
              f"(n_trials={N_TRIALS_PROJECT}): {breakeven:.2f}")

    print(f"\ntrials this round: R-33=13, R-34=24, total={N_TRIALS_ROUND}. "
          f"project cumulative: {N_TRIALS_PROJECT}.")
    print("\nHoldout consultations this call: 3 strategies x 2 markets funding-free "
          "(6) + 2 strategies funding-charged futures (2) + 3 strategies @0.40% fee "
          "spot (3) + 2 markets bootstrap re-reads of gate & v4 (4, reuses funding-free "
          "equity curves computed above) = 11 new backtests of the 2023+ holdout.")


COMMANDS["holdout"] = holdout


if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate.py [{'|'.join(COMMANDS)}]")
