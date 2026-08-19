#!/usr/bin/env python
"""Funding-decile gate on kelly_regime_v4 (backlog B-05, ledger rows R-14/R-16).

**Mechanism, in one sentence.** Rich funding is priced evidence of a
crowded long trade, and the strategy already pays 7x more for funding
while it holds than while it is flat (R-14: +20.05%/yr held vs +2.78%/yr
flat) - refusing the richest decile of that cost directly attacks the
COST constraint without touching the regime-detection or sizing logic at
all.

**Pre-registered falsification test.** Does the gate's improvement over
the un-gated, funding-charged `kelly_regime_v4` baseline survive Bitstamp's
stress fee tier (0.40% taker), not just the default 0.05% futures taker?
A gate that only helps at the optimistic tier and reverses at the
realistic one is the R-12 failure mode (28-of-32 in-sample winners,
0-of-28 out-of-sample) repeating itself.

This is a private experiment (NOT `@register`-ed, NOT under
`src/tradebot/strategies/`) per ROUTINE.md step 5's rule for a NEGATIVE
or unresolved result: keep it in `experiments/`, out of the comparison
table.

Commands::

    python experiments/funding_decile_gate.py sweep       # steps 1-4
    python experiments/funding_decile_gate.py falsify      # step 5
    python experiments/funding_decile_gate.py causality    # causality proof
    python experiments/funding_decile_gate.py all
"""

from __future__ import annotations

import math
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
FUNDING = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)                       # default 0.05% taker
FUTURES_STRESS = MarketSpec.futures(leverage=5.0, fee_rate=0.0040)  # Bitstamp stress tier
SPOT = MarketSpec.spot()

# ROUTINE.md's inner split. Never read past OOS_START in this file.
OOS_START = "2023-01-01"
INNER_TRAIN = ("2017-01-01", "2020-12-31")
INNER_VAL = ("2021-01-01", "2022-12-31")
FUND_COVERED = ("2020-01-01", "2022-12-31")  # stops strictly before OOS_START


# --------------------------------------------------------------------------- strategy


class FundingDecileGate(KellyRegimeV4):
    """kelly_regime_v4, forced flat when trailing funding is in its top decile.

    Mechanism (B-05, grounded in R-14/R-16): rich funding is priced
    evidence of a crowded long trade, and the strategy already pays 7x
    more for funding while it holds than while it is flat (R-14:
    +20.05%/yr held vs +2.78%/yr flat, Cardaliaguet & Lehalle 2018's
    mean-field game of trade crowding) - refusing the richest decile of
    that cost directly attacks the COST constraint without touching the
    regime-detection or sizing logic at all.

    Implementation note, deliberately different from a naive "zero
    `target` unconditionally in `prepare()`": `prepare()` computes the
    causal funding percentile and a boolean `_funding_gate` diagnostic
    column, but leaves the inherited v4 `target` column untouched so a
    caller can inspect exactly what v4 would have done. The veto itself
    is applied in `on_bar()`, gated on `ctx.market.pays_funding`. This
    matters for regression safety: funding is never charged on spot
    (`MarketSpec.spot().pays_funding is False`), so a gate whose whole
    justification is "avoid a cost that does not exist on this market"
    has no business firing there - applying it unconditionally would make
    spot NOT match plain `kelly_regime_v4`, which is exactly the
    regression the task's own spot column is designed to catch. Tested
    directly by `_period` runs on SPOT below (must equal `kelly_regime_v4`
    bit-for-bit) and by the causality perturbation check.

    Pre-registered falsification test: does the improvement over the
    funding-charged `kelly_regime_v4` baseline survive Bitstamp's 0.40%
    taker stress tier? See `falsify()` below.
    """

    # Not @register-ed: private experiment, kept out of the comparison table
    # per ROUTINE.md step 5 (NEGATIVE / unresolved results live here).
    name = "_funding_decile_gate"

    #: settlements are 8-hourly; used to convert a day-denominated lookback
    #: into a settlement count for the rolling percentile window.
    SETTLEMENTS_PER_DAY = 3
    #: settlements needed before the trailing percentile is trusted at all
    #: (~10 days) - below this the gate is a no-op, same as missing data.
    MIN_SETTLEMENTS = 30
    #: how stale the most recent settlement may be before a bar is treated
    #: as having NO funding data (rather than silently carrying forward a
    #: value from days ago) - 2x the 8h cadence, tolerating one missed
    #: settlement.
    STALE_TOLERANCE = pd.Timedelta(hours=16)

    def __init__(self, funding: pd.Series | None = None, threshold: float = 0.90,
                 lookback_days: float | None = 90.0,
                 horizons: tuple[int, ...] = (20, 40, 80), **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.funding = funding
        self.threshold = threshold
        self.lookback_days = lookback_days
        # diagnostics, filled by prepare(); not used for any decision
        self.gated_bars_ = 0
        self.covered_bars_ = 0

    def _trailing_percentile(self) -> pd.Series:
        """Causal trailing percentile rank of each funding settlement.

        At settlement t, the rank uses only settlements <= t (a rolling
        or expanding window ending at t, inclusive) - causal by
        construction, since nothing after t is ever in the window. This
        operates on the 8-hourly settlement series itself, before any
        merge onto 5m bars.
        """
        fund = self.funding.sort_index()

        def rank_last(arr: np.ndarray) -> float:
            return float((arr[-1] >= arr).mean())

        if self.lookback_days is None:
            pct = fund.expanding(min_periods=self.MIN_SETTLEMENTS).apply(rank_last, raw=True)
        else:
            window = max(int(self.lookback_days * self.SETTLEMENTS_PER_DAY), self.MIN_SETTLEMENTS)
            pct = fund.rolling(window, min_periods=self.MIN_SETTLEMENTS).apply(rank_last, raw=True)
        return pct

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # inherits the exact v4 `target` column, untouched
        self.gated_bars_ = 0
        self.covered_bars_ = 0
        df["_funding_gate"] = False
        df["_funding_gate_pct"] = np.nan
        if self.funding is None or len(self.funding) == 0:
            return df

        pct_df = self._trailing_percentile().rename("funding_pct").to_frame()
        pct_df.index = pct_df.index.as_unit(df.index.unit)
        left = pd.DataFrame(index=df.index)
        # Bar i sees only settlements with timestamp <= bar i's timestamp
        # (direction="backward"); `tolerance` makes a bar with no recent
        # settlement (before 2020-01-01, or more than ~2 settlement
        # intervals after 2023-12-31) come back NaN instead of silently
        # carrying forward a stale value - "percentile unknown", not
        # "assume worst case".
        merged = pd.merge_asof(left, pct_df.sort_index(), left_index=True,
                                right_index=True, direction="backward",
                                tolerance=self.STALE_TOLERANCE)
        # Extra shift(1) safety margin on top of the asof-backward merge,
        # matching this repo's universal convention (kelly_regime.py's
        # `vol = (...).shift(1)`): bar i's decision uses only what was
        # already known as of bar i-1, never information dated exactly at
        # bar i.
        gate_input = merged["funding_pct"].shift(1)

        covered = gate_input.notna()
        gated = covered & (gate_input >= self.threshold)
        self.covered_bars_ = int(covered.sum())
        self.gated_bars_ = int(gated.sum())

        df["_funding_gate"] = gated.to_numpy()
        df["_funding_gate_pct"] = gate_input.to_numpy()
        return df

    def on_bar(self, ctx) -> None:
        t = float(ctx.bar["target"])
        if ctx.market.pays_funding and bool(ctx.bar["_funding_gate"]):
            t = 0.0
        if ctx.prev is not None:
            prev = float(ctx.prev["target"])
            if ctx.market.pays_funding and bool(ctx.prev["_funding_gate"]):
                prev = 0.0
        else:
            prev = 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# --------------------------------------------------------------------------- harness


def _period(make, market, start=None, end=None, funding=None):
    """Backtest over a date range, warmed on the bars before it.

    Copied/adapted from scripts/funding_study.py's `_period` (per
    instructions, not imported - that script is owned by a different part
    of the routine and is not to be edited here). `run_period` from
    tradebot.window does not forward a `funding=` kwarg, so this calls
    `run_backtest` directly with an explicit warmup prefix, exactly as
    funding_study.py does.
    """
    strategy = make()
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw, strategy


def make_gate(cfg: dict):
    return lambda cfg=cfg: FundingDecileGate(funding=FUNDING, **cfg)


def baseline_v4():
    return KellyRegimeV4()


def baseline_hold():
    return get_strategy("buy_and_hold")


CONFIGS = [
    dict(threshold=0.85, lookback_days=60),
    dict(threshold=0.85, lookback_days=180),
    dict(threshold=0.85, lookback_days=None),
    dict(threshold=0.90, lookback_days=60),
    dict(threshold=0.90, lookback_days=180),
    dict(threshold=0.90, lookback_days=None),
    dict(threshold=0.95, lookback_days=60),
    dict(threshold=0.95, lookback_days=180),
    dict(threshold=0.95, lookback_days=None),
]


def _fmt_cfg(cfg: dict) -> str:
    lb = "exp" if cfg["lookback_days"] is None else f"{cfg['lookback_days']:.0f}d"
    return f"thr={cfg['threshold']:.2f} lb={lb:>4s}"


def _row(tag: str, m, raw=None) -> None:
    extra = ""
    if raw is not None:
        gated = int(raw.df["_funding_gate"].to_numpy().sum()) if "_funding_gate" in raw.df else 0
        extra = f" funding_paid=${raw.funding_paid:>8,.0f} gated_bars={gated:>6d}"
    print(f"{tag:26s} final=${m.final_balance:>11,.0f} DD={m.max_drawdown_pct:5.1f}% "
          f"sharpe={m.sharpe:5.2f} trades={m.num_trades:5d}{extra}")


def run_sweep(period_name: str, start: str, end: str) -> tuple[list, object, object]:
    """One period: three baselines + every swept config, on futures w/ funding."""
    print(f"\n--- {period_name}  ({start} .. {end}) ---")
    v4_funded, v4_raw, _ = _period(baseline_v4, FUTURES, start, end, funding=FUNDING)
    v4_free, v4_free_raw, _ = _period(baseline_v4, FUTURES, start, end, funding=None)
    hold, hold_raw, _ = _period(baseline_hold, SPOT, start, end, funding=None)
    print("baselines:")
    _row("(a) v4 futures + funding", v4_funded, v4_raw)
    _row("(b) v4 futures fund-free", v4_free, v4_free_raw)
    _row("(c) buy_and_hold SPOT", hold, hold_raw)

    print("\nswept configs (futures + funding):")
    rows = []
    for cfg in CONFIGS:
        m, raw, strat = _period(make_gate(cfg), FUTURES, start, end, funding=FUNDING)
        _row(_fmt_cfg(cfg), m, raw)
        rows.append((cfg, m, raw))

    print("\nregression check (spot, must equal plain kelly_regime_v4):")
    v4_spot, v4_spot_raw, _ = _period(baseline_v4, SPOT, start, end, funding=None)
    gate_spot, gate_spot_raw, _ = _period(make_gate(CONFIGS[4]), SPOT, start, end, funding=None)
    identical = np.array_equal(v4_spot_raw.equity.to_numpy(), gate_spot_raw.equity.to_numpy())
    _row("v4 SPOT", v4_spot)
    _row("gate(thr=.90,lb=180) SPOT", gate_spot)
    print(f"spot equity curves identical: {identical}")

    return rows, v4_funded, v4_raw


def select_config(val_rows: list, val_v4) -> dict:
    """Select on inner-validation ONLY, never on inner-train. Rule fixed in
    advance: rank by final balance; the config must beat the funding-charged
    v4 baseline to be worth reporting further, but selection itself is just
    "best of the swept configs on inner-validation"."""
    best_cfg, best_m, best_raw = max(val_rows, key=lambda r: r[1].final_balance)
    print(f"\nSELECTED (on inner-validation only): {_fmt_cfg(best_cfg)}")
    print(f"  inner-validation final=${best_m.final_balance:,.0f} DD={best_m.max_drawdown_pct:.1f}% "
          f"sharpe={best_m.sharpe:.2f}  vs v4-funded final=${val_v4.final_balance:,.0f} "
          f"DD={val_v4.max_drawdown_pct:.1f}% sharpe={val_v4.sharpe:.2f}")
    return best_cfg


def causality_check(cfg: dict) -> bool:
    """Perturbation proof: multiply funding strictly AFTER a cutoff by 50x,
    recompute, assert `target` and `_funding_gate_pct` are byte-identical
    before the cutoff. Entirely below OOS_START; no automated CI test
    covers this file since it parametrizes only over the registry."""
    print("\n--- causality perturbation check ---")
    lo = int(DF.index.searchsorted("2019-06-01"))
    hi = int(DF.index.searchsorted("2022-12-31", side="right"))
    df = DF.iloc[lo:hi].copy()
    cutoff_ts = pd.Timestamp("2021-06-01", tz="UTC")
    cutoff_i = int(df.index.searchsorted(cutoff_ts))

    base = FundingDecileGate(funding=FUNDING, **cfg).prepare(df.copy())

    tampered_funding = FUNDING.copy()
    tampered_funding.loc[tampered_funding.index > cutoff_ts] *= 50.0
    tampered = FundingDecileGate(funding=tampered_funding, **cfg).prepare(df.copy())

    ok = True
    for col in ("target", "_funding_gate", "_funding_gate_pct"):
        a = base[col].to_numpy()[:cutoff_i]
        b = tampered[col].to_numpy()[:cutoff_i]
        mismatch = ~(pd.isna(a.astype(float)) & pd.isna(b.astype(float))) & (a != b) \
            if col == "_funding_gate_pct" else (a != b)
        n_bad = int(np.asarray(mismatch).sum())
        if n_bad:
            ok = False
            print(f"  CAUSALITY FAIL: column {col!r} differs at {n_bad} of {cutoff_i} "
                  f"bars before the cutoff")
        else:
            print(f"  {col!r}: identical for all {cutoff_i:,} bars before "
                  f"{cutoff_ts} ({0} mismatches)")
    print("CAUSALITY CHECK: " + ("PASSED" if ok else "FAILED") +
          f" - config {_fmt_cfg(cfg)}, funding after {cutoff_ts} multiplied by 50x")
    return ok


def falsify(cfg: dict) -> None:
    """Pre-registered falsification test: does the gate's improvement over
    the funding-charged v4 baseline survive Bitstamp's 0.40% stress tier?
    Run on the funding-covered window only (2020-2022), selected config
    only - not a fresh sweep."""
    print(f"\n--- falsification test: {_fmt_cfg(cfg)}, funding-covered window ---")
    start, end = FUND_COVERED

    m_sel_def, _, _ = _period(make_gate(cfg), FUTURES, start, end, funding=FUNDING)
    m_v4_def, _, _ = _period(baseline_v4, FUTURES, start, end, funding=FUNDING)
    m_sel_str, _, _ = _period(make_gate(cfg), FUTURES_STRESS, start, end, funding=FUNDING)
    m_v4_str, _, _ = _period(baseline_v4, FUTURES_STRESS, start, end, funding=FUNDING)

    def lg(m):
        return math.log(max(m.final_balance, 1e-9) / 1000.0)

    d_def = lg(m_sel_def) - lg(m_v4_def)
    d_str = lg(m_sel_str) - lg(m_v4_str)
    print(f"default fee (0.05% taker):  gate ${m_sel_def.final_balance:>10,.0f}  "
          f"v4 ${m_v4_def.final_balance:>10,.0f}   dlog_growth={d_def:+.4f}")
    print(f"stress fee  (0.40% taker):  gate ${m_sel_str.final_balance:>10,.0f}  "
          f"v4 ${m_v4_str.final_balance:>10,.0f}   dlog_growth={d_str:+.4f}")
    survived = (d_def > 0) == (d_str > 0)
    verdict = "SURVIVED (sign preserved)" if survived else "FAILED (sign reversed - R-12 pattern)"
    print(f"\nFALSIFICATION TEST: {verdict}")


def sweep() -> dict:
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  (data: {LABEL})")
    print(f"{len(FUNDING):,} funding settlements  {FUNDING.index[0]:%Y-%m-%d} -> "
          f"{FUNDING.index[-1]:%Y-%m-%d}\n")
    print(f"configurations swept: {len(CONFIGS)}")

    print("\n" + "=" * 78)
    print("STEP 1a: inner-train (2017-01-01 .. 2020-12-31)")
    print("funding data only exists from 2020-01-01, so the gate is a no-op for")
    print("~3 of these 4 years - expected, not a bug.")
    print("=" * 78)
    train_rows, train_v4, _ = run_sweep("inner-train", *INNER_TRAIN)

    print("\n" + "=" * 78)
    print("STEP 1b: inner-validation (2021-01-01 .. 2022-12-31)")
    print("=" * 78)
    val_rows, val_v4, _ = run_sweep("inner-validation", *INNER_VAL)

    selected = select_config(val_rows, val_v4)

    print("\n" + "=" * 78)
    print("STEP 2: funding-covered window (2020-01-01 .. 2022-12-31)")
    print("the fairer comparison for this idea - the gate can act throughout")
    print("=" * 78)
    run_sweep("funding-covered", *FUND_COVERED)

    return selected


def main() -> None:
    if FUNDING is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in ("sweep", "all"):
        selected = sweep()
    else:
        selected = dict(threshold=0.90, lookback_days=180)  # a sane default if run standalone
    if choice in ("falsify", "all"):
        falsify(selected)
    if choice in ("causality", "all"):
        causality_check(selected)
    if choice not in ("sweep", "falsify", "causality", "all"):
        print("usage: python experiments/funding_decile_gate.py [sweep|falsify|causality|all]")


if __name__ == "__main__":
    main()
