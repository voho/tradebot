"""Continuous funding x momentum haircut on kelly_regime_v4 (backlog B-05, extended).

Not registered: lives under ``experiments/`` so it is not auto-discovered,
per ROUTINE.md step 5. Promote into ``src/tradebot/strategies/`` only if it
clears the promotion bar.

The idea
--------
R-14 found the strategy's futures leg is structurally short the funding
premium: funding runs +20%/yr while the strategy holds vs +2.8%/yr while
flat, because the crowding ``kelly_regime_v4`` detects is exactly what
sets the rate. R-16 found funding is also informative about *forward
returns*, but only jointly with momentum (docs/ALTERNATIVES.md, "Funding
as a positioning signal, not just a cost"): mean forward 7-day return by
funding tercile x trailing-7-day-return tercile is

|              | past low | past mid | past high |
|--------------|----------|----------|-----------|
| funding low  |  +2.83%  |  +1.74%  |  +2.16%   |
| funding mid  |  +0.55%  |  +1.36%  |  +3.24%   |
| funding high |  -1.68%  |  -1.54%  |  +1.22%   |

High funding is only bad news when momentum is NOT confirming. Correlation
between funding and trailing return is just 0.39, so funding is not simply
a momentum proxy — it carries information momentum does not.

``kelly_regime_v4``'s documented weakness (docs/STRATEGIES.md #1) is that
"it lags badly in steady bulls." A hard funding gate (stand flat above
some funding percentile) would cut exposure during exactly those steady
bulls too, since funding runs richest precisely when price is grinding up
with confirmation — the (high funding, high momentum) cell above, which is
the *good* cell in R-16's table. A momentum-conditioned, continuous
haircut is designed to only bite the bad cells (high funding, low/mid
momentum) and leave the confirmed-bull cell alone.

Mechanism
---------
1. ``v4_target`` = ``KellyRegimeV4.prepare(df)["target"]`` — the existing,
   already causally-deadbanded v4 signal. Unchanged.
2. Funding is reindexed onto the OHLCV index with ``ffill`` and explicitly
   masked to "no info" (multiplier 1, i.e. pass-through v4 behaviour)
   outside ``[funding.index[0], funding.index[-1]]``. The committed series
   covers 2020-01-01 -> 2023-12-31 only; without the mask, ``ffill`` would
   silently repeat the last 2023 rate through the rest of the 2023+
   holdout and all of 2024-2026 — more than 80% of the out-of-sample
   period — which would be fabricated information. See "Data coverage
   limitation" below; this is the single most important correctness
   requirement in this file.
3. ``frank_t`` = causal percentile rank of the current (masked/ffilled)
   funding rate within its own trailing window, computed on the funding
   series' *native* 8-hourly settlement cadence (window ~180 days ~ 540
   settlements, min_periods ~60 days ~ 180 settlements), via
   ``Series.rolling(window, min_periods).rank(pct=True)`` — pandas computes
   this as the rank of the LAST value in each trailing window relative to
   the other values in that same window (verified by hand below), so it
   is exactly the trailing, causal quantity the spec asks for and never
   touches a future settlement.
4. ``mrank_t`` = causal percentile rank of ``mom_t = log(close_t) -
   log(close_{t-7d})`` within its own trailing 180-day window (~51,840
   five-minute bars, min_periods ~60d ~17,280 bars), same
   ``rolling(...).rank(pct=True)`` construction, computed directly on the
   OHLCV frame's own close column (always available, no coverage gap).
5. ``danger_t = sigmoid((frank_t - r0)/w1) * sigmoid((m0 - mrank_t)/w2)``,
   high only when funding is crowded (frank near/above r0) AND momentum is
   NOT confirming (mrank below m0). ``multiplier_t = 1 - haircut_max *
   danger_t``. Wherever frank or mrank is not yet defined (insufficient
   trailing history) OR funding has no coverage at that bar, multiplier
   defaults to 1 (no haircut, i.e. unchanged v4 behaviour) rather than 0 or
   NaN — an unknown danger is not evidence of danger.
6. ``raw_t = v4_target_t * multiplier_t``. This project's own deadband
   (``self.deadband``, default 0.10, inherited from ``KellyRegime``) is
   then applied ONCE, in a fresh position loop against the previous FINAL
   position — not against v4's already-deadbanded target a second time —
   the same "state variable carried across a single scan" idiom
   ``kelly_regime_v3.py`` uses for its own vol-regime loop. This keeps the
   haircut from adding its own turnover on every bar it nudges the
   multiplier.

Grid (pre-registered, at most 6 configurations)
------------------------------------------------
``r0 in {0.75, 0.85}``, ``m0 = 0.5`` (fixed), ``haircut_max in {0.5, 1.0}``,
``w1 = w2 = 0.08`` (fixed) -> a 2x2 grid, 4 configurations total. See
``sweep()`` below; the exact count is printed and also stated in the
session report.

Data coverage limitation
-------------------------
``data/btcusdt_perp_funding_8h.csv.gz`` covers 2020-01-01 -> 2023-12-31.
Roughly 80% of the post-2023 holdout, and all of the 2017-2019 portion of
the inner-train split, have NO funding observation at all. On those bars
this strategy is mechanically identical to plain ``kelly_regime_v4``
(multiplier == 1 everywhere out of range) — it can only differ from v4
inside the observed window. This is a real, stated limitation, not a bug
being routed around: the honest reading of any result here is "what this
haircut does during 2020-2023," not "what it would do full-period."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class KellyRegimeFundingMomentum(KellyRegimeV4):
    """kelly_regime_v4 with a continuous funding x momentum exposure haircut.

    NOT registered (experiments/ only, per ROUTINE.md step 5). See module
    docstring for the mechanism, the pre-registered grid, and the funding
    data's coverage limitation.
    """

    name = "kelly_regime_funding_momentum"
    # IMPORTANT: kept EQUAL to KellyRegimeV4.warmup (80 days), not raised to
    # the 180-day mrank/frank window. tradebot.engine.run_backtest gates
    # on_bar itself on `i >= strategy.warmup`, independent of trade_start -
    # a longer warmup here would silently delay this strategy's first
    # possible trade relative to v4's by ~100 days on any period backtested
    # from the very start of the dataset (inner-train has no pre-2017 bars
    # to draw a warmup PREFIX from, so window.run_period's prefix mechanism
    # cannot compensate; a first attempt at this file set warmup=180d and
    # it silently cost inner-train ~40% of final balance for a reason that
    # had nothing to do with the funding/momentum mechanism - see the
    # report). Instead, the 180-day rolling windows fall back to their
    # documented "insufficient history -> multiplier 1 (pass-through v4)"
    # behaviour for their first ~100 days on any cold-started period; that
    # is the correct causal behaviour, not a bug to warm away.
    warmup = KellyRegimeV4.warmup

    def __init__(
        self,
        funding: pd.Series | None = None,
        r0: float = 0.85,
        m0: float = 0.5,
        haircut_max: float = 0.5,
        w1: float = 0.08,
        w2: float = 0.08,
        frank_window_days: float = 180.0,
        frank_min_days: float = 60.0,
        mom_lookback_days: float = 7.0,
        mrank_window_days: float = 180.0,
        mrank_min_days: float = 60.0,
        settlements_per_day: float = 3.0,  # Binance 8h cadence; a fixed constant,
        # not a data-derived statistic - used only to size the rolling window.
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.funding = funding
        self.r0 = r0
        self.m0 = m0
        self.haircut_max = haircut_max
        self.w1 = w1
        self.w2 = w2
        self.frank_window_days = frank_window_days
        self.frank_min_days = frank_min_days
        self.mom_lookback_days = mom_lookback_days
        self.mrank_window_days = mrank_window_days
        self.mrank_min_days = mrank_min_days
        self.settlements_per_day = settlements_per_day

    # ------------------------------------------------------------------ pieces

    def _multiplier(self, df: pd.DataFrame) -> np.ndarray:
        """Continuous funding x momentum haircut multiplier, causal throughout.

        Returns an all-ones array (pass-through v4 behaviour) when no
        funding series was supplied, or wherever funding has no coverage
        or the trailing windows have insufficient history.
        """
        n = len(df)
        if self.funding is None or len(self.funding) == 0:
            return np.ones(n)

        funding = self.funding.sort_index()

        # --- frank_t: causal percentile rank of funding, on its own native
        # settlement cadence (a trailing window ENDING at each settlement,
        # never touching a later one).
        win_f = max(int(round(self.frank_window_days * self.settlements_per_day)), 2)
        minp_f = max(int(round(self.frank_min_days * self.settlements_per_day)), 2)
        frank_native = funding.rolling(win_f, min_periods=minp_f).rank(pct=True)

        # Align onto the OHLCV index by forward-fill (last KNOWN settlement
        # as of each bar), then explicitly mask outside funding's observed
        # range. Without this mask, ffill would silently repeat the last
        # 2023-12-31 rate through the rest of the 2023+ holdout and all of
        # 2024-2026 - fabricated information for most of the OOS period.
        frank = frank_native.reindex(df.index, method="ffill")
        in_range = (df.index >= funding.index[0]) & (df.index <= funding.index[-1])
        frank = frank.where(in_range)

        # --- mrank_t: causal percentile rank of trailing 7-day momentum,
        # computed directly on this frame's own close column (always
        # available - no coverage gap here).
        close = df["close"]
        lb = max(int(round(self.mom_lookback_days * BARS_PER_DAY)), 1)
        mom = np.log(close) - np.log(close.shift(lb))
        win_m = max(int(round(self.mrank_window_days * BARS_PER_DAY)), 2)
        minp_m = max(int(round(self.mrank_min_days * BARS_PER_DAY)), 2)
        mrank = mom.rolling(win_m, min_periods=minp_m).rank(pct=True)

        valid = frank.notna().to_numpy() & mrank.notna().to_numpy()
        fr = frank.to_numpy(dtype=float)
        mr = mrank.to_numpy(dtype=float)

        with np.errstate(over="ignore", invalid="ignore"):
            danger = _sigmoid((fr - self.r0) / self.w1) * _sigmoid((self.m0 - mr) / self.w2)
        haircut_mult = 1.0 - self.haircut_max * danger

        return np.where(valid, haircut_mult, 1.0)

    # ----------------------------------------------------------------- strategy

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4's own causal, internally-deadbanded target
        v4_target = df["target"].to_numpy(dtype=float)

        multiplier = self._multiplier(df)
        raw = v4_target * multiplier

        n = len(df)
        final = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = raw[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            final[i] = pos

        df["v4_target"] = v4_target
        df["fm_multiplier"] = multiplier
        df["target"] = final
        return df

    # on_bar is inherited unchanged from KellyRegime: reads df["target"],
    # orders ctx.order_notional(t) on the notional fraction of equity.


# =========================================================================
# Driver: ROUTINE.md step 3 (inner-train / inner-validation only)
# =========================================================================

if __name__ == "__main__":
    import sys
    import time
    from dataclasses import replace
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

    from tradebot.broker import MarketSpec
    from tradebot.data import load_dataset, load_funding
    from tradebot.engine import run_backtest
    from tradebot.metrics import compute_metrics
    from tradebot.registry import get_strategy
    from tradebot.window import run_period

    DF, LABEL = load_dataset(ROOT / "data", "spot")
    REAL_FUNDING = load_funding(ROOT / "data")
    SPOT = MarketSpec.spot()
    FUTURES = MarketSpec.futures(leverage=5.0)

    TRAIN = ("2017-01-01", "2020-12-31")
    VALID = ("2021-01-01", "2022-12-31")

    def _period(strategy, start, end, market, funding=None):
        """One backtest over [start, end], warmed on the bars before it,
        with optional funding charged. Mirrors scripts/funding_study.py's
        _period and experiments/run_eprocess.py's costs() helper, since
        run_period() does not expose a funding kwarg."""
        lo = 0 if start is None else int(DF.index.searchsorted(start))
        hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
        pre = min(lo, strategy.warmup)
        raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                           trade_start=pre, funding=funding, data_label=LABEL)
        trimmed = (raw if pre == 0 else
                   replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
        return compute_metrics(trimmed), raw.funding_paid

    def ev(strategy, start, end, market, tag="", funding=None):
        t0 = time.time()
        m, funding_paid = _period(strategy, start, end, market, funding=funding)
        print(f"  {tag or strategy.name:36s} {market.name:11s} "
              f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
              f"trades={m.num_trades:>5d} DD={m.max_drawdown_pct:>5.1f}% "
              f"sharpe={m.sharpe:>5.2f} funding=${funding_paid:>8,.0f}"
              f"{'  LIQUIDATED' if m.liquidated else ''} [{time.time() - t0:.0f}s]")
        return m, funding_paid

    def _grid():
        """The pre-registered 2x2 grid: at most 6 configurations, here 4."""
        out = []
        for r0 in (0.75, 0.85):
            for hmax in (0.5, 1.0):
                tag = f"r0={r0} hmax={hmax}"
                out.append((tag, dict(r0=r0, m0=0.5, haircut_max=hmax, w1=0.08, w2=0.08)))
        return out

    def sweep() -> None:
        """Inner-train + inner-validation, both markets, funding on and off."""
        grid = _grid()
        for (start, end), split in ((TRAIN, "INNER-TRAIN"), (VALID, "INNER-VALIDATION")):
            for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
                for fund_label, fund in (("no funding", None), ("funding charged", REAL_FUNDING)):
                    print(f"\n{split} / {mname} / {fund_label}:")
                    ev(get_strategy("buy_and_hold"), start, end, market,
                       tag="buy_and_hold", funding=fund)
                    ev(get_strategy("kelly_regime_v4"), start, end, market,
                       tag="kelly_regime_v4 (baseline)", funding=fund)
                    for tag, kw in grid:
                        s = KellyRegimeFundingMomentum(funding=REAL_FUNDING, **kw)
                        ev(s, start, end, market, tag=f"fm {tag}", funding=fund)
        print(f"\nconfigurations evaluated (distinct KellyRegimeFundingMomentum "
              f"parameterizations searched, the pre-registered grid): {len(grid)}")

    def inspect(r0=0.85, m0=0.5, haircut_max=0.5, w1=0.08, w2=0.08) -> None:
        """Per-year multiplier behaviour, and a hard-gate comparison for 2021.

        Answers the report's key question: does the haircut avoid cutting
        exposure during the CONFIRMED 2021 bull continuation (funding rich,
        momentum also high) the way a hard funding-only gate would not?
        """
        s = KellyRegimeFundingMomentum(funding=REAL_FUNDING, r0=r0, m0=m0,
                                       haircut_max=haircut_max, w1=w1, w2=w2)
        prepared = s.prepare(DF.loc[:"2022-12-31"].copy())
        mult = prepared["fm_multiplier"]
        v4t = prepared["v4_target"]
        in_mkt = v4t.abs() > 1e-9

        print("by year (bars where v4_target != 0 only):")
        for year, idx in mult.groupby(mult.index.year):
            mask = in_mkt.loc[idx.index]
            sub = idx[mask]
            if len(sub) == 0:
                print(f"  {year}  (v4 flat all year)")
                continue
            haircut_frac = (sub < 0.999).mean()
            print(f"  {year}  mean multiplier {sub.mean():.3f}  "
                  f"haircut active {haircut_frac:.1%} of in-market bars  "
                  f"min {sub.min():.3f}")

        print("\n2021 bull continuation check (funding rank vs a hypothetical "
              "HARD gate that zeros exposure whenever frank >= r0, ignoring "
              f"momentum entirely; r0={r0}):")
        y21 = prepared.loc["2021-01-01":"2021-12-31"]
        funding_native = REAL_FUNDING.sort_index()
        win_f = max(int(round(180 * 3)), 2)
        minp_f = max(int(round(60 * 3)), 2)
        frank_native = funding_native.rolling(win_f, min_periods=minp_f).rank(pct=True)
        frank_21 = frank_native.reindex(y21.index, method="ffill")
        hard_gate_mult = np.where(frank_21 >= r0, 0.0, 1.0)
        our_mult = y21["fm_multiplier"].to_numpy()
        crowded = frank_21.to_numpy() >= r0
        print(f"  bars in 2021 with frank >= r0 (crowded funding): "
              f"{crowded.mean():.1%} of the year")
        if crowded.any():
            print(f"  this experiment's mean multiplier on those bars: "
                  f"{our_mult[crowded].mean():.3f}  "
                  f"(1.0 = fully pass-through, matches a hard gate's 0.0 only "
                  f"when momentum also fails to confirm)")
            print(f"  a HARD funding-only gate would set multiplier=0.0 on "
                  f"ALL of those bars, cutting {crowded.mean():.1%} of the "
                  f"year's exposure regardless of momentum")
        close = prepared["close"]
        lb = 7 * BARS_PER_DAY
        mom = np.log(close) - np.log(close.shift(lb))
        mrank_full = mom.rolling(180 * BARS_PER_DAY, min_periods=60 * BARS_PER_DAY).rank(pct=True)
        mrank_21 = mrank_full.reindex(y21.index).to_numpy()
        confirming = crowded & (mrank_21 >= m0)
        not_confirming = crowded & (mrank_21 < m0)
        if confirming.any():
            print(f"  of the crowded bars, momentum CONFIRMING (mrank>=m0): "
                  f"{confirming.sum():,} bars, our mean multiplier "
                  f"{our_mult[confirming].mean():.3f} "
                  f"(should be close to 1.0 - this is the bull-continuation case)")
        if not_confirming.any():
            print(f"  of the crowded bars, momentum NOT confirming (mrank<m0): "
                  f"{not_confirming.sum():,} bars, our mean multiplier "
                  f"{our_mult[not_confirming].mean():.3f} "
                  f"(should be well below 1.0 - this is the crowded-and-fragile case)")

    def causality() -> None:
        """Hand-run causality check (mirrors run_eprocess.py's causality()).

        This experiment is not covered by tests/test_causality_strict.py,
        which parametrizes over the registry only. Same two-opposite-
        tampers procedure: bars strictly after a cut are multiplied by 3 in
        one copy and divided by 3 in the other; every decision at or before
        the cut, and the prepared target/multiplier columns before the cut,
        must be identical.
        """
        from tradebot.broker import PaperBroker
        from tradebot.strategy import Context

        df = DF.iloc[-300_000:].copy()
        cut = len(df) - 5_000
        bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

        up, down = df.copy(), df.copy()
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

        kw = dict(funding=REAL_FUNDING, r0=0.85, m0=0.5, haircut_max=0.5, w1=0.08, w2=0.08)

        def decisions(frame):
            s = KellyRegimeFundingMomentum(**kw)
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

        pa = KellyRegimeFundingMomentum(**kw).prepare(up.copy())
        pb = KellyRegimeFundingMomentum(**kw).prepare(down.copy())
        for col in ("target", "v4_target", "fm_multiplier"):
            diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
            worst = float(np.nanmax(diff))
            print(f"  column {col:15s} max |difference| before the cut = {worst:.3e}"
                  f"  {'PASS' if worst < 1e-9 else 'FAIL'}")

    def rank_semantics() -> None:
        """Hand-verify rolling(...).rank(pct=True) is the trailing, causal
        quantity the mechanism assumes: rank of the LAST value in each
        window relative to the other values in that same window."""
        s = pd.Series([5, 1, 2, 9, 3, 3, 8, 0, 7, 6], dtype=float)
        r = s.rolling(4, min_periods=2).rank(pct=True)
        manual = [(s.iloc[max(0, i - 3):i + 1] <= s.iloc[i]).mean()
                  if i >= 1 else np.nan for i in range(len(s))]
        print("value:  ", s.tolist())
        print("rolling:", r.round(3).tolist())
        print("manual: ", [round(x, 3) if x == x else x for x in manual])

    COMMANDS = {"sweep": sweep, "causality": causality, "rank_semantics": rank_semantics,
                "inspect": inspect}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in COMMANDS:
        print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
              f"(data: {LABEL})", file=sys.stderr)
        if REAL_FUNDING is not None:
            print(f"funding: {len(REAL_FUNDING):,} settlements  "
                  f"{REAL_FUNDING.index[0]:%Y-%m-%d} -> {REAL_FUNDING.index[-1]:%Y-%m-%d}",
                  file=sys.stderr)
        COMMANDS[choice]()
    else:
        print(f"usage: python experiments/funding_momentum.py [{'|'.join(COMMANDS)}]")
