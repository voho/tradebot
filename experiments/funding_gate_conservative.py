"""Funding as a gate on kelly_regime_v4: stand flat in the top decile (backlog B-05).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

The idea, in one sentence
-------------------------
``kelly_regime_v4`` decides *how much* to hold from price alone; funding
(R-14) is a real, adversely-timed cost that scales with exactly the
crowding the strategy's vote is chasing (COST constraint), and R-16 found
that the top decile of funding predicts negative 14-day forward returns.
So: leave the vote and the sizer untouched, and force the position flat
whenever *currently known* funding sits at or above its own trailing
percentile — a pure cost gate, not a new direction signal, which is the
low-turnover use of R-16 the ledger asks for (B-05) rather than the
higher-turnover reversal use that has repeatedly lost (R-12).

Mechanism
---------
``FundingDecileGate`` calls ``KellyRegimeV4.prepare()`` unchanged to get
the parent's ``target`` column, then computes two causal, bar-aligned
series from the real Binance BTCUSDT funding history
(``data/btcusdt_perp_funding_8h.csv.gz``, 2020-01-01..2023-12-31):

1. ``funding_bar`` — the most recently *settled* funding rate at or
   before each 5-minute bar (``REAL.reindex(df.index, method="ffill")``).
   Settlements are public the instant they happen, so this is not a
   lookahead; it is real, already-known exchange data.
2. ``threshold_bar`` — a trailing rolling quantile of funding, computed on
   the 8-hourly *settlement* series (not resampled onto 5m bars, which
   would run a meaningless rolling window over piecewise-constant data),
   using only settlements strictly before the one in question
   (``.rolling(window).quantile(q).shift(1)``), then aligned onto the bar
   grid the same causal way.

Wherever either series is unknown -- before 2020-01-01, before the
trailing window has filled, or after the last committed settlement
(2023-12-31) -- the gate is provably disabled: ``target`` is bit-identical
to the parent's. Where both are known and ``funding_bar >= threshold_bar``,
``target`` is forced to 0.0; otherwise the parent's target passes through
unchanged.

Falsification test (chosen before any sweep ran)
--------------------------------------------------
Does it survive funding (``scripts/funding_study.py`` design): score with
real funding charged on 5x futures, inner-train and inner-validation, and
compare final balance / drawdown / Sharpe / funding paid against the
un-gated ``kelly_regime_v4`` and against ``buy_and_hold``. The outcome
that kills it: the vote gate already dodges most of the top-decile
funding on its own (R-14's ``timing()`` finding — the strategy is flat on
a large fraction of bars already), so forcing flat again on the bars where
it's already flat is a no-op, and forcing flat on bars where it *is*
holding costs return without a matching drawdown or cost benefit large
enough to clear it — and forcing flat on SPOT, where funding is never
charged, can only ever cost return for nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.data import load_funding
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

ROOT = Path(__file__).resolve().parents[1]


class FundingDecileGate(KellyRegimeV4):
    """kelly_regime_v4, forced flat whenever known funding is at/above its own trailing percentile.

    Attacks the COST constraint (R-14: funding is adversely timed against
    this strategy's vote, running ~+20%/yr while it holds vs +2.8%/yr
    flat). Pure cost gate: the vote and sizer are untouched, and outside
    the 2020-2023 funding-covered/warmed window the gate is provably
    disabled (target is bit-identical to kelly_regime_v4's). See
    docs/LEDGER.md backlog B-05 and R-16.
    """

    name = "funding_gate_conservative"

    # Class-level cache: the funding file is loaded once, even if this
    # strategy is instantiated many times inside a sweep. `None` here means
    # "not yet attempted"; load_funding() itself may legitimately return
    # None if the file is absent, in which case every prepare() call just
    # re-checks (cheap) and the gate is a no-op throughout.
    _real_funding_cache: pd.Series | None = None
    _real_funding_loaded: bool = False

    def __init__(self, funding_percentile_threshold: float = 0.90,
                 funding_lookback_days: int = 90, **kwargs) -> None:
        super().__init__(**kwargs)
        self.funding_percentile_threshold = funding_percentile_threshold
        self.funding_lookback_days = funding_lookback_days

    @classmethod
    def _load_real_funding(cls) -> pd.Series | None:
        if not cls._real_funding_loaded:
            cls._real_funding_cache = load_funding(ROOT / "data")
            cls._real_funding_loaded = True
        return cls._real_funding_cache

    def _causal_funding_and_threshold(
        self, index: pd.DatetimeIndex, real: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Bar-aligned, causal funding rate and trailing threshold.

        Both are built by reindexing an 8-hourly settlement-level series
        onto the 5-minute bar grid with ``method="ffill"``: for bar
        timestamp ``t``, that takes the value at the last settlement
        timestamp <= t (exact ties included, since settlements land on
        the 5-minute grid) -- i.e. "the most recently known rate" -- and
        leaves bars before the first settlement as NaN (nothing to fill
        forward from). Bars strictly after the last committed settlement
        are explicitly masked back to NaN even though ffill would
        otherwise carry the last value forward forever: the data run
        stops in 2023, and treating a four-year-stale rate as "currently
        known" would be a silent extrapolation, not real data.
        """
        window = self.funding_lookback_days * 3  # 3 settlements/day
        threshold_settlement = (
            real.rolling(window=window, min_periods=window)
            .quantile(self.funding_percentile_threshold)
            .shift(1)  # exclude the settlement itself from its own quantile
        )

        funding_bar = real.reindex(index, method="ffill")
        threshold_bar = threshold_settlement.reindex(index, method="ffill")

        out_of_coverage = index > real.index[-1]
        funding_bar = funding_bar.where(~out_of_coverage)
        threshold_bar = threshold_bar.where(~out_of_coverage)
        return funding_bar, threshold_bar

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # parent's target, computed exactly as it always is

        real = self._load_real_funding()
        if real is None or len(real) == 0:
            # No funding data committed at all: gate is a global no-op.
            df["funding_gate_active"] = False
            return df

        funding_bar, threshold_bar = self._causal_funding_and_threshold(df.index, real)
        known = funding_bar.notna().to_numpy() & threshold_bar.notna().to_numpy()
        gate_shut = known & (funding_bar.to_numpy() >= threshold_bar.to_numpy())

        parent_target = df["target"].to_numpy(copy=True)
        df["target"] = np.where(gate_shut, 0.0, parent_target)

        # Diagnostic columns only -- read by the manual checks below, not
        # used by on_bar (inherited unchanged from KellyRegime), so they
        # cannot themselves introduce lookahead into a decision.
        df["funding_rate_known"] = funding_bar
        df["funding_threshold"] = threshold_bar
        df["funding_gate_active"] = gate_shut
        return df


# --------------------------------------------------------------------------
# Everything below is the ad-hoc evaluation harness for this session, kept
# in the same file per the task instructions (only one lasting artifact).
# Run as `python experiments/funding_gate_conservative.py <command>`.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

    from dataclasses import replace

    from tradebot.broker import MarketSpec
    from tradebot.data import load_dataset
    from tradebot.engine import run_backtest
    from tradebot.metrics import compute_metrics
    from tradebot.registry import get_strategy

    DF, LABEL = load_dataset(ROOT / "data", "spot")
    REAL = load_funding(ROOT / "data")
    FUTURES = MarketSpec.futures(leverage=5.0)
    SPOT = MarketSpec.spot()

    TRAIN = ("2017-01-01", "2020-12-31")
    VALID = ("2021-01-01", "2022-12-31")

    N_EVALUATED = 0

    def period(strategy, market, start=None, end=None, funding=None, count=False,
               label=""):
        global N_EVALUATED
        if count:
            N_EVALUATED += 1
        lo = 0 if start is None else int(DF.index.searchsorted(start))
        hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
        pre = min(lo, strategy.warmup)
        raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                            trade_start=pre, funding=funding, data_label=LABEL)
        trimmed = raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:],
                                                df=raw.df.iloc[pre:])
        m = compute_metrics(trimmed)
        if label:
            print(f"  {label:34s} final=${m.final_balance:>11,.0f} "
                  f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
                  f"sharpe={m.sharpe:>5.2f} fees=${m.fees_paid:>7,.0f} "
                  f"funding=${raw.funding_paid:>7,.0f}"
                  f"{'  LIQ' if m.liquidated else ''}")
        return m, raw.funding_paid

    # ---------------------------------------------------------- ffill check

    def ffill_check() -> None:
        """Manual confirmation that reindex(method='ffill') is causal."""
        real = REAL
        settle = real.index[100]  # some interior settlement
        prev_settle = real.index[99]
        window = pd.date_range(settle - pd.Timedelta(minutes=30),
                                settle + pd.Timedelta(minutes=30), freq="5min", tz="UTC")
        aligned = real.reindex(window, method="ffill")
        print(f"settlement at {settle}  prev={prev_settle}\n")
        print(aligned.to_string())
        before = aligned[aligned.index < settle]
        at_after = aligned[aligned.index >= settle]
        ok = bool((before == real.loc[prev_settle]).all()) and \
            bool((at_after == real.loc[settle]).all())
        print(f"\nvalue only changes at/after the settlement timestamp: "
              f"{'PASS' if ok else 'FAIL'}")

    # -------------------------------------------------- bit-identical check

    def identity_check() -> None:
        """target must equal kelly_regime_v4's target exactly outside the covered/warmed window."""
        v4 = get_strategy("kelly_regime_v4")
        gate = FundingDecileGate()
        p_v4 = v4.prepare(DF.copy())
        p_gate = gate.prepare(DF.copy())
        outside = ~p_gate["funding_gate_active"].to_numpy(dtype=bool)
        # "outside the covered/warmed window" as the task defines it: gate
        # inactive because funding/threshold unknown (NOT because funding
        # was known but below threshold -- that's the mechanism working).
        unknown = p_gate["funding_rate_known"].isna().to_numpy() | \
            p_gate["funding_threshold"].isna().to_numpy()
        diff = np.abs(p_v4["target"].to_numpy() - p_gate["target"].to_numpy())
        diff_unknown = diff[unknown]
        print(f"bars total: {len(diff):,}")
        print(f"bars with funding/threshold UNKNOWN (gate must be a no-op): "
              f"{unknown.sum():,} ({unknown.mean():.1%})")
        print(f"  max |target diff| there: {diff_unknown.max():.3e}  "
              f"{'PASS' if diff_unknown.max() == 0.0 else 'FAIL'}")
        print(f"bars with funding/threshold KNOWN: {(~unknown).sum():,} "
              f"({(~unknown).mean():.1%})")
        known_active = p_gate["funding_gate_active"].to_numpy() & ~unknown
        print(f"  of those, gate ACTIVE (forced flat): {int(known_active.sum()):,} "
              f"({known_active.sum() / max((~unknown).sum(), 1):.1%} of known bars)")
        # ranges
        idx = DF.index
        print(f"\ndataset range: {idx[0]} -> {idx[-1]}")
        print(f"funding coverage: {REAL.index[0]} -> {REAL.index[-1]}")
        first_known = idx[~unknown][0] if (~unknown).any() else None
        last_known = idx[~unknown][-1] if (~unknown).any() else None
        print(f"first/last bar with gate decidable: {first_known} -> {last_known}")

    # ------------------------------------------------------------ causality

    def causality_check() -> None:
        """Truncation test: target at bar i unchanged whether fed df.iloc[:i+50] or the full df."""
        idx = DF.index
        checkpoints = []
        for label, ts in (("pre-2020", "2019-06-01"), ("in-2021", "2021-06-01"),
                           ("in-2023", "2023-06-01"), ("post-2023", "2024-06-01"),
                           ("post-2023-late", "2026-01-01")):
            pos = int(idx.searchsorted(ts))
            if 0 < pos < len(idx):
                checkpoints.append((label, pos))

        full = FundingDecileGate().prepare(DF.copy())
        print(f"{'checkpoint':16s} {'index':>10s} {'full target':>14s} "
              f"{'truncated target':>18s} {'match':>7s}")
        all_ok = True
        for label, i in checkpoints:
            trunc_df = DF.iloc[: i + 50].copy()
            trunc = FundingDecileGate().prepare(trunc_df)
            tv, fv = trunc["target"].to_numpy()[i], full["target"].to_numpy()[i]
            ok = abs(tv - fv) < 1e-12
            all_ok &= ok
            print(f"{label:16s} {i:>10d} {fv:>14.6f} {tv:>18.6f} "
                  f"{'PASS' if ok else 'FAIL'}")
        print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")

    # ---------------------------------------------------------------- sweep

    THRESHOLDS = (0.80, 0.85, 0.90, 0.95)
    LOOKBACKS = (30, 90, 180)

    def sweep() -> None:
        print(f"configs: {len(THRESHOLDS)} thresholds x {len(LOOKBACKS)} lookbacks "
              f"= {len(THRESHOLDS) * len(LOOKBACKS)} distinct FundingDecileGate configs\n")
        for market, mname, funding in ((SPOT, "spot", None), (FUTURES, "futures5x", REAL)):
            for split_name, (start, end) in (("INNER-TRAIN", TRAIN), ("INNER-VALID", VALID)):
                print(f"\n=== {split_name} / {mname} ===")
                period(get_strategy("buy_and_hold"), market, start, end, funding=funding,
                       label="buy_and_hold")
                period(get_strategy("kelly_regime_v4"), market, start, end, funding=funding,
                       label="kelly_regime_v4 (baseline)")
                for th in THRESHOLDS:
                    for lb in LOOKBACKS:
                        strat = FundingDecileGate(funding_percentile_threshold=th,
                                                   funding_lookback_days=lb)
                        period(strat, market, start, end, funding=funding, count=True,
                               label=f"gate th={th} lb={lb}d")
        print(f"\nconfigurations evaluated (distinct FundingDecileGate params): "
              f"{len(THRESHOLDS) * len(LOOKBACKS)}")
        print(f"total backtests this run (count=True calls): {N_EVALUATED}")

    # ------------------------------------------------------- plateau check

    def neighbours() -> None:
        """Finer grid around the inner-validation optimum (th=0.90, lb=90d):
        plateau or peak? INNER-VALIDATION only, since that is the selection
        criterion; each point here is a further distinct configuration and
        is counted."""
        grid = [(th, lb) for th in (0.88, 0.90, 0.92) for lb in (60, 90, 120)]
        print(f"neighbourhood configs: {len(grid)}\n")
        for market, mname, funding in ((SPOT, "spot", None), (FUTURES, "futures5x", REAL)):
            print(f"\n=== INNER-VALID / {mname} neighbourhood ===")
            for th, lb in grid:
                strat = FundingDecileGate(funding_percentile_threshold=th,
                                           funding_lookback_days=lb)
                period(strat, market, *VALID, funding=funding, count=True,
                       label=f"gate th={th} lb={lb}d")
        print(f"\nconfigurations evaluated this run: {len(grid)}")

    # -------------------------------------------------- gate-dodges-funding

    def already_dodges() -> None:
        """Does the vote gate already dodge most of the top-decile funding on its own?

        (One of the pre-registered concerns: is this gate redundant with the
        existing regime vote?) Mirrors funding_study.py's `timing()`.
        """
        v4 = get_strategy("kelly_regime_v4")
        prepared = v4.prepare(DF.copy())
        target = prepared["target"].to_numpy()
        flat = np.abs(target) < 1e-12
        in_market = pd.Series(~flat, index=DF.index)
        aligned = in_market.reindex(REAL.index, method="ffill").fillna(False)

        threshold = REAL.quantile(0.90)  # unconditional top decile, for context only
        top_decile = REAL >= threshold
        print(f"v4 vote is FLAT on {flat.mean():.1%} of all bars (full 2017-2026 history)")
        print(f"unconditional top-decile funding rate threshold: {threshold:+.6f} "
              f"({threshold * 3 * 365.25:+.1%}/yr)")
        print(f"\nof the {top_decile.sum():,} top-decile settlements (2020-2023 coverage):")
        print(f"  v4 is ALREADY flat during {(~aligned[top_decile.to_numpy()]).mean():.1%} "
              f"of them")
        print(f"  v4 is HOLDING during        {aligned[top_decile.to_numpy()].mean():.1%} "
              f"of them  <- this is the fraction the gate can still act on")

    CMDS = {"ffill": ffill_check, "identity": identity_check,
            "causality": causality_check, "sweep": sweep,
            "neighbours": neighbours, "already_dodges": already_dodges}

    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in CMDS:
        CMDS[choice]()
    else:
        print(f"usage: python experiments/funding_gate_conservative.py "
              f"[{'|'.join(CMDS)}]")
