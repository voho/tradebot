"""Variant A of B-05: conservative funding-decile flat gate on kelly_regime_v4.

Pre-registration: experiments/funding_gate_preregistration.md

Mechanism: rank each bar's trailing-90-day annualized funding rate against
its own trailing history (a causal, expanding-then-rolling percentile, never
against the full series); force the v4 position flat once that percentile
enters a "crowded" state (>= enter_pct), release only once it drops back
below exit_pct (latching hysteresis, the same pattern kelly_regime_v3's
volatility-breakout state and the anchor vote already use).

NOT @register-ed: this is a research experiment file, not a promoted
strategy, and must not enter `tradebot run`'s matrix.

Usage (from repo root, venv activated):

    python experiments/funding_gate_conservative.py
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
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

# ---------------------------------------------------------------------------
# Splits (funding data covers 2020-01-01 .. 2023-12-31 only; see prereg doc).
# HOLDOUT_START is defined only so we can assert we never touch it -- this
# file must never backtest, print, or inspect anything at or after it.
# ---------------------------------------------------------------------------
INNER_TRAIN = ("2020-01-01", "2021-12-31")
INNER_VALID = ("2022-01-01", "2022-12-31")
HOLDOUT_START = pd.Timestamp("2023-01-01", tz="UTC")


class FundingGateConservative(KellyRegimeV4):
    """kelly_regime_v4 forced flat while trailing funding sits in a crowded percentile.

    Adds one causal input (perp funding) on top of v4's price-only regime
    vote: a trailing rolling percentile rank of the annualized funding rate
    latches the position flat above ``enter_pct`` and releases it only below
    ``exit_pct`` (hysteresis), exactly the pattern the anchor vote and v3's
    volatility-breakout state already use. Degrades to plain v4 wherever
    funding data is absent (percentile undefined -> gate never fires).
    """

    name = "funding_gate_conservative"  # not registered; name kept for logging only

    def __init__(
        self,
        funding: pd.Series,
        enter_pct: float = 0.90,
        exit_pct: float = 0.70,
        window_days: int = 90,
        min_periods_days: int = 30,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not (0.0 < exit_pct < enter_pct <= 1.0):
            raise ValueError(f"require 0 < exit_pct < enter_pct <= 1, got {exit_pct=} {enter_pct=}")
        self.funding = funding
        self.enter_pct = enter_pct
        self.exit_pct = exit_pct
        self.window_days = window_days
        self.min_periods_days = min_periods_days

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4's regime/vol target, in df["target"]
        base_target = df["target"].to_numpy().copy()

        # --- causal alignment: only settlements at-or-before each bar ------
        funding_aligned = self.funding.reindex(df.index, method="ffill").fillna(0.0)
        annualized = funding_aligned * 3 * 365.25  # 3 settlements/day, 8h perp

        window = self.window_days * BARS_PER_DAY
        min_periods = self.min_periods_days * BARS_PER_DAY
        # Trailing percentile rank of the CURRENT bar within its own trailing
        # window only -- never against the full series. pandas' Rolling.rank
        # (confirmed available: pandas 3.0.5 in this venv) computes exactly
        # that: for window ending at i, rank of x[i] among x[i-window+1..i].
        pct = annualized.rolling(window, min_periods=min_periods).rank(pct=True)
        # Insufficient trailing history (e.g. the first ~30 days funding
        # data exists at all) -> treat as "not crowded" (0th pct), which is
        # below every swept enter_pct, so the gate simply cannot fire yet.
        # This is the same "degrades to v4" default the prereg specifies.
        pct = pct.fillna(0.0).to_numpy()

        n = len(df)
        final_target = np.empty(n)
        gate_active = np.zeros(n, dtype=np.int8)
        state = 0  # 0 = open, 1 = latched flat (crowded)
        for i in range(n):
            p = pct[i]
            if state == 0 and p >= self.enter_pct:
                state = 1
            elif state == 1 and p <= self.exit_pct:
                state = 0
            gate_active[i] = state
            final_target[i] = 0.0 if state == 1 else base_target[i]

        df["target_pre_gate"] = base_target
        df["funding_pct"] = pct
        df["gate_active"] = gate_active
        df["target"] = final_target
        return df


# ---------------------------------------------------------------------------
# Backtest-over-a-window helper, replicated from scripts/funding_study.py's
# _period() because run_period() does not support funding. Identical logic.
# ---------------------------------------------------------------------------

def period(strategy, df, market, start, end, funding, balance=1000.0, data_label=""):
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, df.iloc[lo - pre: hi], market, balance,
                        trade_start=pre, funding=funding, data_label=data_label)
    trimmed = raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
    return compute_metrics(trimmed), raw.funding_paid, trimmed


def _assert_no_holdout(end: str | None) -> None:
    if end is None:
        raise ValueError("refusing an open-ended period end date -- holdout guard")
    if pd.Timestamp(end, tz="UTC") >= HOLDOUT_START:
        raise ValueError(f"refusing to backtest through {end}: touches funding-holdout (>= {HOLDOUT_START})")


def run_window(strategy, df, market, funding, window, label):
    start, end = window
    _assert_no_holdout(end)
    m, funding_paid, raw = period(strategy, df, market, start, end, funding, data_label=label)
    log_growth = float(np.log(max(m.final_balance, 1e-9) / m.start_balance))
    return {
        "final_balance": m.final_balance,
        "log_growth": log_growth,
        "max_drawdown_pct": m.max_drawdown_pct,
        "sharpe": m.sharpe,
        "num_trades": m.num_trades,
        "funding_paid": funding_paid,
    }, raw


CONFIGS = [
    (enter, exit_)
    for enter in (0.85, 0.90, 0.95)
    for exit_ in (0.60, 0.70, 0.75)
    if exit_ < enter
]


def fmt_row(name, r):
    return (f"{name:28s} ${r['final_balance']:>11,.0f}  "
            f"logG {r['log_growth']:>6.3f}  DD {r['max_drawdown_pct']:>5.1f}%  "
            f"Sharpe {r['sharpe']:>5.2f}  trades {r['num_trades']:>4d}  "
            f"funding ${r['funding_paid']:>9,.0f}")


def main() -> None:
    DF, LABEL = load_dataset(ROOT / "data", "spot")
    funding = load_funding(ROOT / "data")
    if funding is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    FUTURES = MarketSpec.futures(leverage=5.0)

    print(f"data: {LABEL}, {len(DF):,} bars, {DF.index[0]} .. {DF.index[-1]}")
    print(f"funding: {len(funding):,} settlements, {funding.index[0]} .. {funding.index[-1]}")
    print(f"configs swept: {len(CONFIGS)} -> {CONFIGS}")
    print()

    print(f"{'=' * 90}\nINNER-TRAIN {INNER_TRAIN[0]} .. {INNER_TRAIN[1]}\n{'=' * 90}")
    baseline_train, _ = run_window(KellyRegimeV4(), DF, FUTURES, funding, INNER_TRAIN, LABEL)
    print(fmt_row("kelly_regime_v4 (baseline)", baseline_train))
    train_results = {}
    for enter, exit_ in CONFIGS:
        strat = FundingGateConservative(funding=funding, enter_pct=enter, exit_pct=exit_)
        r, _ = run_window(strat, DF, FUTURES, funding, INNER_TRAIN, LABEL)
        train_results[(enter, exit_)] = r
        print(fmt_row(f"enter={enter:.2f} exit={exit_:.2f}", r))

    print(f"\n{'=' * 90}\nINNER-VALIDATION {INNER_VALID[0]} .. {INNER_VALID[1]}\n{'=' * 90}")
    baseline_valid, _ = run_window(KellyRegimeV4(), DF, FUTURES, funding, INNER_VALID, LABEL)
    print(fmt_row("kelly_regime_v4 (baseline)", baseline_valid))
    valid_results = {}
    valid_raw = {}
    for enter, exit_ in CONFIGS:
        strat = FundingGateConservative(funding=funding, enter_pct=enter, exit_pct=exit_)
        r, raw = run_window(strat, DF, FUTURES, funding, INNER_VALID, LABEL)
        valid_results[(enter, exit_)] = r
        valid_raw[(enter, exit_)] = raw
        print(fmt_row(f"enter={enter:.2f} exit={exit_:.2f}", r))

    # Select the frozen config on inner-validation log growth (primary
    # promotion criterion per the prereg's decision rule item 1).
    best_key = max(valid_results, key=lambda k: valid_results[k]["log_growth"])
    print(f"\nbest on inner-validation (log growth): enter={best_key[0]:.2f} exit={best_key[1]:.2f}")

    # --- (a) redundancy check: does the gate mostly fire when v4 was
    # already flat (pre-gate target ~ 0)? Computed for the frozen config,
    # on inner-validation (post-warmup rows only, i.e. raw.df after trim).
    best_strat = FundingGateConservative(funding=funding, enter_pct=best_key[0], exit_pct=best_key[1])
    _, best_raw = run_window(best_strat, DF, FUTURES, funding, INNER_VALID, LABEL)
    gated = best_raw.df["gate_active"].to_numpy() == 1
    pre_gate_target = best_raw.df["target_pre_gate"].to_numpy()
    n_gated = int(gated.sum())
    if n_gated > 0:
        already_flat = np.abs(pre_gate_target[gated]) < 1e-9
        redundancy_frac = float(already_flat.mean())
    else:
        redundancy_frac = float("nan")
    print(f"\nredundancy check (inner-validation, frozen config): "
          f"gate active on {n_gated}/{len(gated)} bars ({100 * n_gated / len(gated):.1f}%); "
          f"of those, {100 * redundancy_frac:.1f}% already had a flat pre-gate v4 target")

    # --- (b) regime-overfit check: does the edge over baseline (log growth
    # delta) hold in both 2020-2021 and 2022 separately, for the frozen cfg?
    train_edge = train_results[best_key]["log_growth"] - baseline_train["log_growth"]
    valid_edge = valid_results[best_key]["log_growth"] - baseline_valid["log_growth"]
    print(f"\nregime-split check (frozen config, log-growth edge over v4 baseline):")
    print(f"  inner-train (2020-2021): {train_edge:+.4f}")
    print(f"  inner-validation (2022): {valid_edge:+.4f}")

    print(f"\nFrozen config: enter={best_key[0]:.2f} exit={best_key[1]:.2f}")
    print(f"Configs evaluated: {len(CONFIGS)}")


if __name__ == "__main__":
    main()
