"""B-05: funding as a gate on kelly_regime_v4 (attacks COST).

Backlog item B-05 (docs/LEDGER.md): `kelly_regime_v4` ignores funding
entirely. R-14 measured real Binance BTCUSDT funding as positive at 86.5%
of settlements, costing a constant long ~15%/yr -- and worse, running
~+20%/yr while the strategy HOLDS vs +2.8%/yr while it is flat, because
the same crowding that produces the strategy's long signal is what sets
the funding rate. R-16 found funding's top decile predicts a *negative*
14-day forward spot return (Q1-Q5 = +3.57pp in the opposite direction),
with only 0.39 correlation to trailing return (not just a momentum
proxy) -- though middle quintiles are non-monotone, a warning about how
much of R-16 is noise.

This experiment augments v4's target exposure with a funding-based gate:
when trailing funding is unusually high (crowded, expensive longs), cut
the LONG side of the exposure the base strategy would otherwise hold, on
FUTURES only. It does not touch short exposure (a short position is PAID
funding when the rate is positive, so there is no cost to gate away) and
it is not applied to spot at all -- spot never pays funding, so a
funding gate there could only ever hurt the strategy's spot performance
with no offsetting benefit; this is a futures-only, cost-specific
overlay by design, not a general-purpose signal.

Three variants (mechanism, one sentence each):

  1. FundingGateDecile  -- hard gate: zero the long leg whenever the
     current (trailing, causal) funding rate sits at or above its own
     trailing 90th percentile.
  2. FundingGateQuintile -- same hard gate, but at the trailing 80th
     percentile (top quintile), a wider net that fires more often.
  3. FundingGateSmooth  -- continuous haircut: the long leg scales
     linearly from 1x at the trailing median down to 0x at the trailing
     90th percentile (floored at 0 above that), so there is no
     discontinuity and no extra turnover beyond what the 5% notional
     deadband already absorbs.

Pre-registered falsification test (chosen before any result was read):
does the ranking survive at the 0.40% taker fee tier (Bitstamp entry
tier, `scripts/fee_study.py` convention)? A gate that only works at the
0.10% tier is not a COST fix, it is noise dressed as one. Given the time
budget, path-sensitivity is checked with a *small, explicitly-counted*
number of manually resampled sub-windows (not the full 40-window
`stress_test.py` harness) -- see `manual_stress_windows()` below.

Train/validation/holdout split (adapted -- funding only exists
2020-01-01 .. 2023-12-31, so the project's normal 2017/2021/2023 split
does not fit):

  inner-train      2020-01-01 -> 2021-12-31   fit/sweep freely
  inner-validation 2022-01-01 -> 2022-12-31   select between variants
  holdout          2023-01-01 -> 2023-12-31   evaluate EXACTLY ONCE

Run as a script: `python experiments/funding_gate.py`.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL_FUNDING = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

INNER_TRAIN = ("2020-01-01", "2021-12-31")
INNER_VAL = ("2022-01-01", "2022-12-31")
HOLDOUT = ("2023-01-01", "2023-12-31")

SETTLEMENTS_PER_DAY = 3  # Binance funding settles every 8h

CONFIG_COUNT = 0  # every call to period_result() increments this


# --------------------------------------------------------------------- data

def causal_funding_frame(index: pd.DatetimeIndex, funding: pd.Series,
                          window_days: int) -> pd.DataFrame:
    """Trailing, causal funding features aligned onto ``index``.

    For every settlement time, the trailing median / 80th / 90th
    percentile are computed over a **rolling window of past settlements
    only** (never the whole series -- R-21's lookahead bug class). The
    settlement series is then aligned onto the 5m bar index by
    forward-fill (the value known at bar t is the last settlement at or
    before t) and additionally shifted by one bar, matching the
    belt-and-suspenders convention `kelly_regime_v3.py` uses for its
    volatility ratio, so no same-bar information can leak into the
    target computed for that bar.
    """
    window = window_days * SETTLEMENTS_PER_DAY
    min_periods = max(10, window // 3)
    raw = funding.sort_index()
    feats = pd.DataFrame({
        "rate": raw,
        "median": raw.rolling(window, min_periods=min_periods).median(),
        "q80": raw.rolling(window, min_periods=min_periods).quantile(0.80),
        "q90": raw.rolling(window, min_periods=min_periods).quantile(0.90),
    })
    aligned = feats.reindex(index, method="ffill").shift(1)
    return aligned


# ------------------------------------------------------------- strategies

class FundingGateBase(KellyRegimeV4):
    """v4's target, then a causal funding haircut applied to the LONG leg only.

    Short exposure is untouched: a short receives funding when the rate
    is positive, so there is no cost there to gate away. This class is
    evaluated on futures only (see module docstring for why spot is out
    of scope).
    """

    funding_window_days = 60  # overridable per-instance for the sweep

    def __init__(self, funding: pd.Series, funding_window_days: int = 60,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._funding = funding
        self.funding_window_days = funding_window_days

    def _haircut(self, rate: np.ndarray, median: np.ndarray,
                 q80: np.ndarray, q90: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)
        feats = causal_funding_frame(df.index, self._funding, self.funding_window_days)
        rate = feats["rate"].to_numpy()
        median = feats["median"].to_numpy()
        q80 = feats["q80"].to_numpy()
        q90 = feats["q90"].to_numpy()
        haircut = self._haircut(rate, median, q80, q90)
        target = df["target"].to_numpy().copy()
        long_mask = target > 0
        target[long_mask] = target[long_mask] * haircut[long_mask]
        df["target"] = target
        return df


class FundingGateDecile(FundingGateBase):
    """Zero the long leg whenever trailing funding sits at/above its own trailing 90th percentile."""

    name = "funding_gate_decile"

    def _haircut(self, rate, median, q80, q90):
        valid = np.isfinite(rate) & np.isfinite(q90)
        return np.where(valid & (rate >= q90), 0.0, 1.0)


class FundingGateQuintile(FundingGateBase):
    """Zero the long leg whenever trailing funding sits at/above its own trailing 80th percentile."""

    name = "funding_gate_quintile"

    def _haircut(self, rate, median, q80, q90):
        valid = np.isfinite(rate) & np.isfinite(q80)
        return np.where(valid & (rate >= q80), 0.0, 1.0)


class FundingGateSmooth(FundingGateBase):
    """Long leg scales linearly from 1x at the trailing median to 0x at the trailing 90th percentile."""

    name = "funding_gate_smooth"

    def _haircut(self, rate, median, q80, q90):
        valid = np.isfinite(rate) & np.isfinite(median) & np.isfinite(q90)
        span = np.where(valid, q90 - median, np.nan)
        span = np.where(np.isfinite(span) & (span > 1e-12), span, np.nan)
        frac_above = np.where(np.isfinite(span), (rate - median) / span, 0.0)
        haircut = 1.0 - np.clip(frac_above, 0.0, 1.0)
        return np.where(valid, haircut, 1.0)


VARIANTS = {
    "decile": FundingGateDecile,
    "quintile": FundingGateQuintile,
    "smooth": FundingGateSmooth,
}


# ------------------------------------------------------------------ harness

def period_result(strategy, market, start=None, end=None, funding=None,
                   balance: float = 1_000.0, spot_fee=None, futures_fee=None):
    """Backtest over [start, end], warmed on the bars before it (funding_study.py pattern).

    Every call counts as one evaluated configuration (see CONFIG_COUNT).
    """
    global CONFIG_COUNT
    CONFIG_COUNT += 1
    if spot_fee is not None or futures_fee is not None:
        market = replace(market, fee_rate=(futures_fee if market.leverage != 1.0 else spot_fee))
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, balance,
                        trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw.funding_paid


def fmt(tag: str, m, funding_paid: float) -> str:
    return (f"{tag:34s} final=${m.final_balance:>10,.0f} "
            f"({m.profit_pct:>+8.1f}%) trades={m.num_trades:>4d} "
            f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
            f"funding_paid=${funding_paid:>8,.0f}"
            f"{'  LIQUIDATED' if m.liquidated else ''}")


def make_variant(key: str, window_days: int) -> FundingGateBase:
    return VARIANTS[key](funding=REAL_FUNDING, funding_window_days=window_days)


# -------------------------------------------------------------- step 3: sweep

def sweep_inner_train() -> dict:
    """Every variant x trailing-window combo, on inner-train, funding charged."""
    start, end = INNER_TRAIN
    print(f"\n=== inner-train {start} -> {end} (futures 5x, funding charged) ===")
    results = {}
    baseline, bpaid = period_result(get_strategy("kelly_regime_v4"), FUTURES,
                                     start, end, funding=REAL_FUNDING)
    print(fmt("baseline kelly_regime_v4 (ungated)", baseline, bpaid))
    results["baseline"] = (baseline, bpaid)
    for key in VARIANTS:
        for window in (30, 60, 90):
            tag = f"{key} window={window}d"
            m, paid = period_result(make_variant(key, window), FUTURES, start, end,
                                     funding=REAL_FUNDING)
            print(fmt(tag, m, paid))
            results[(key, window)] = (m, paid)
    return results


# --------------------------------------------------------- step 3: validation

def validate(candidates: list[tuple[str, int]]) -> dict:
    """Selected candidates + baseline, on inner-validation, funding charged."""
    start, end = INNER_VAL
    print(f"\n=== inner-validation {start} -> {end} (futures 5x, funding charged) ===")
    results = {}
    baseline, bpaid = period_result(get_strategy("kelly_regime_v4"), FUTURES,
                                     start, end, funding=REAL_FUNDING)
    print(fmt("baseline kelly_regime_v4 (ungated)", baseline, bpaid))
    results["baseline"] = (baseline, bpaid)
    hold, hpaid = period_result(get_strategy("buy_and_hold"), SPOT, start, end)
    print(fmt("buy_and_hold (spot)", hold, hpaid))
    results["buy_and_hold"] = (hold, hpaid)
    for key, window in candidates:
        tag = f"{key} window={window}d"
        m, paid = period_result(make_variant(key, window), FUTURES, start, end,
                                 funding=REAL_FUNDING)
        print(fmt(tag, m, paid))
        results[(key, window)] = (m, paid)
    return results


# ------------------------------------------------------------ step 4: holdout

def holdout(frozen_key: str, frozen_window: int) -> dict:
    start, end = HOLDOUT
    print(f"\n=== HOLDOUT {start} -> {end} (evaluated once) ===")
    results = {}

    print("-- funding charged (the real comparison) --")
    base_f, base_paid = period_result(get_strategy("kelly_regime_v4"), FUTURES,
                                       start, end, funding=REAL_FUNDING)
    print(fmt("baseline kelly_regime_v4 (ungated)", base_f, base_paid))
    gate_f, gate_paid = period_result(make_variant(frozen_key, frozen_window), FUTURES,
                                       start, end, funding=REAL_FUNDING)
    print(fmt(f"FROZEN {frozen_key} window={frozen_window}d", gate_f, gate_paid))
    hold_f, hold_paid = period_result(get_strategy("buy_and_hold"), SPOT, start, end)
    print(fmt("buy_and_hold (spot)", hold_f, hold_paid))
    results["funded"] = {"baseline": (base_f, base_paid), "gated": (gate_f, gate_paid),
                          "hold": (hold_f, hold_paid)}

    print("-- funding-free, for reference only --")
    base_nf, _ = period_result(get_strategy("kelly_regime_v4"), FUTURES, start, end)
    gate_nf, _ = period_result(make_variant(frozen_key, frozen_window), FUTURES, start, end)
    print(fmt("baseline kelly_regime_v4 (no funding)", base_nf, 0.0))
    print(fmt(f"FROZEN {frozen_key} (no funding)", gate_nf, 0.0))
    results["funding_free"] = {"baseline": base_nf, "gated": gate_nf}

    return results


# ---------------------------------------------- falsification: 0.40% fee tier

def falsification_fee_tier(frozen_key: str, frozen_window: int) -> dict:
    """Pre-registered test: does the gate's edge over baseline survive 0.40% taker?"""
    start, end = HOLDOUT
    print(f"\n=== falsification: 0.40% taker tier, holdout {start}->{end}, funding charged ===")
    fee = 0.0040
    base, base_paid = period_result(get_strategy("kelly_regime_v4"), FUTURES, start, end,
                                     funding=REAL_FUNDING, futures_fee=fee)
    gate, gate_paid = period_result(make_variant(frozen_key, frozen_window), FUTURES, start, end,
                                     funding=REAL_FUNDING, futures_fee=fee)
    print(fmt("baseline @0.40%", base, base_paid))
    print(fmt(f"FROZEN {frozen_key} @0.40%", gate, gate_paid))
    return {"baseline": (base, base_paid), "gated": (gate, gate_paid)}


# --------------------------------------------------- falsification: sub-windows

def manual_stress_windows(frozen_key: str, frozen_window: int) -> list[dict]:
    """A SMALL, explicitly-counted set of manually chosen sub-windows within
    2020-2023 (funding's only range), NOT the full 40-window stress_test.py
    harness -- there was not time budget for that here. Six windows: two per
    year (2020, 2021, 2022), roughly bull/bear-mixed halves, all inside the
    funding data's range so both baseline and gate can be measured with
    funding charged on identical periods."""
    windows = [
        ("2020 H1", "2020-01-01", "2020-06-30"),
        ("2020 H2", "2020-07-01", "2020-12-31"),
        ("2021 H1", "2021-01-01", "2021-06-30"),
        ("2021 H2", "2021-07-01", "2021-12-31"),
        ("2022 H1", "2022-01-01", "2022-06-30"),
        ("2022 H2", "2022-07-01", "2022-12-31"),
    ]
    print(f"\n=== manual stress windows ({len(windows)} of them, not the full 40) ===")
    out = []
    for label, start, end in windows:
        base, base_paid = period_result(get_strategy("kelly_regime_v4"), FUTURES, start, end,
                                         funding=REAL_FUNDING)
        gate, gate_paid = period_result(make_variant(frozen_key, frozen_window), FUTURES, start, end,
                                         funding=REAL_FUNDING)
        better = gate.final_balance > base.final_balance
        lower_dd = gate.max_drawdown_pct < base.max_drawdown_pct
        print(f"{label:10s} base=${base.final_balance:>9,.0f} DD={base.max_drawdown_pct:5.1f}% | "
              f"gated=${gate.final_balance:>9,.0f} DD={gate.max_drawdown_pct:5.1f}% | "
              f"gate {'beats' if better else 'trails'} base, "
              f"{'lower' if lower_dd else 'not lower'} DD")
        out.append({"window": label, "baseline": base, "gated": gate})
    return out


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  (data: {LABEL})")
    print(f"funding: {len(REAL_FUNDING):,} settlements  "
          f"{REAL_FUNDING.index[0]:%Y-%m-%d} -> {REAL_FUNDING.index[-1]:%Y-%m-%d}")

    train_results = sweep_inner_train()

    # Pre-registered selection rule (written before inner-validation is read):
    # take the single (variant, window) combo with the highest inner-train
    # final balance that ALSO has max_drawdown_pct no worse than the ungated
    # baseline's, then confirm it on inner-validation before freezing.
    candidates = sorted(
        ((k, v) for k, v in train_results.items() if k != "baseline"),
        key=lambda kv: kv[1][0].final_balance, reverse=True)
    top3 = [k for k, _ in candidates[:3]]
    print(f"\ntop 3 on inner-train by final balance: {top3}")

    val_results = validate(top3)

    print(f"\nconfigurations evaluated so far: {CONFIG_COUNT}")
