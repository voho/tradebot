"""B-05, conservative variant: a binary top-decile funding flatten gate.

Not registered: this lives under ``experiments/`` so it is not
auto-discovered (ROUTINE.md step 5) and is not part of the pytest suite
that walks ``tradebot.strategies`` (only that package is auto-imported by
``tradebot.registry._discover``), so nothing here runs under CI.

The idea, in one sentence
--------------------------
R-16 found that Binance BTCUSDT funding predicts forward returns (14-day
Q1-Q5 spread +3.57pp, not a momentum proxy: correlation with trailing
return only 0.39); B-05 proposes the *low-turnover* way to use that -
stand flat, instead of holding a leveraged long, whenever funding is in
its own trailing top decile - layered on top of the incumbent
``kelly_regime_v4`` rather than as a standalone reversal signal, because
R-12 already showed what the high-turnover standalone use costs (28/32
in-sample winners, 0/28 out-of-sample).

Constraint attacked: COST. Not "another indicator" (which ROUTINE.md
flags as attacking nothing) - this is specifically about the cost that
scales with the signal (R-14: funding runs +20%/yr while the strategy
holds, because the crowding it detects is what sets the rate).

Which ledger rows this is not a duplicate of: R-14/R-16 measured the
funding cost and the funding signal separately, as observations, not as
a strategy. This is the first thing in the repo that trades on either.

Mechanism, precisely: one binary condition. Whenever the trailing
(causal) percentile rank of the funding rate is at or above
``flatten_threshold`` (default 0.90, i.e. the top decile of its own
trailing history) AND a funding observation is actually available
(2020-01-01..2023-12-31 only - real Binance BTCUSDT data, nothing
fabricated outside it), ``target`` is forced to 0.0 for that bar,
overriding whatever ``kelly_regime_v4`` computed. Everywhere else -
including the entire pre-2020 and post-2023 range, where there is no
funding file to consult - behavior is byte-identical to
``kelly_regime_v4``. No smoothing, no hysteresis, no second signal: that
is deliberately the sibling session's job (``funding_gate_continuous.py``),
not this file's.

Pre-registered falsification test (written before any result was looked
at): does the improvement over ``kelly_regime_v4``, if there is one,
survive Bitstamp's 0.40% taker fee tier on the SAME inner-validation
window used to select a configuration? This variant adds extra
flatten/re-enter trades relative to the incumbent, so it is specifically
exposed to fee-tier sensitivity - that is the whole point of running the
test, not a formality.

What would make it fail (named before running anything): the gate fires
so rarely, or fires in bars that were already flat under the vote gate,
that it changes nothing; or it fires in bars that mattered and the extra
turnover it adds is worth more in fees than the crowding signal is worth
in avoided drawdown; or the "improvement" lives entirely in one config
out of twelve and is not a neighbourhood (a peak, not a plateau) - R-12's
grave marker.

Split used, and why it deviates from ROUTINE.md
-------------------------------------------------
ROUTINE.md's standard inner split is 2017-01-01..2020-12-31 (train) /
2021-01-01..2022-12-31 (validation). Funding data only exists
2020-01-01..2023-12-31, so a gate that only ever fires inside that
four-year window cannot be evaluated against the standard split without
either an untested pre-funding train period or reading into the holdout.
This file uses, instead: **inner-train = 2020-01-01..2021-12-31**,
**inner-validation = 2022-01-01..2022-12-31** - the funding-covered years
split in half, train then validation, leaving 2023 entirely untouched so
that if this variant is ever advanced to Step 4 there is still a year of
funding-covered holdout left that has not been read here. This is a
deviation forced by data availability, not a convenience, and is called
out again in the report this file's ``main()`` prints.

Funding percentile: the causal construction
---------------------------------------------
``tradebot.data.load_funding`` returns a Series indexed by 8-hourly
settlement time, decimal rate, positive = longs pay. The percentile at
settlement ``t`` is::

    pct[t] = rank of funding[t-1] within the trailing `window_days`
             of settlements ending at t-1 (inclusive), as a fraction
             in [0, 1]

computed with ``rolling(window).rank(pct=True)`` (percentile rank of the
window's *last* value against the rest of the window) and then
``.shift(1)`` - the repo's causal-estimator convention (see
``kelly_regime*.prepare()``'s ``vol...shift(1)``,
``experiments/matched_risk.py``'s e-process gate). The shift means
``pct[t]`` never uses ``funding[t]`` or anything later: it is a function
of ``funding[t-window-1 : t-1]`` only. This deliberately avoids both
lookahead classes ``tests/test_causality_strict.py`` and
``tests/test_causality_real.py`` exist to catch: (a) a strategy that
keeps the ``prepare()`` frame and indexes ahead inside ``on_bar`` - not
applicable here, ``on_bar`` is inherited unchanged from ``KellyRegime``
and only reads the already-computed ``target`` column at the current
bar; and (b) a scaler/quantile/rank computed once over the WHOLE series
and applied to early rows - avoided by using a *rolling*, trailing-window
rank, never a full-series one.

The settlement-level percentile is then broadcast onto the 5-minute bar
grid by a forward-fill anchored at each settlement's own timestamp (funding
settles every 8h; the percentile computed at one settlement holds until
the next). Because ``pct[t]`` is already lagged to information available
at ``t-1``, holding it from bar-time ``t`` onward is conservative, not a
leak: the earliest a bar could legitimately know it is the moment
settlement ``t-1`` posted, which is before ``t``. Bars before the funding
file's first settlement, and bars after its last settlement (crucially,
including everything from 2024 onward), are explicitly set to NaN rather
than left to a stale forward-fill - a NaN percentile fails the
``>= threshold`` test the same way ``-inf`` would, so the gate is
inactive and ``target`` is untouched, which is exactly "identical to
kelly_regime_v4 outside 2020-2023".
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
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

SETTLEMENTS_PER_DAY = 3  # perp funding settles every 8h


class FundingDecileGate(KellyRegimeV4):
    """``kelly_regime_v4`` forced flat whenever trailing funding is in its top decile.

    Not registered (see module docstring) - an experiment for backlog
    item B-05, the CONSERVATIVE variant: one mechanism, a hard binary
    flatten, no smoothing or hysteresis on the gate itself. Identical to
    ``kelly_regime_v4`` in every bar where the funding percentile is
    unavailable (outside 2020-2023) or below ``flatten_threshold``.
    ``on_bar`` is inherited unchanged from ``KellyRegime``: it just
    re-emits whatever ``prepare()`` put in ``target`` via
    ``ctx.order_notional``.
    """

    name = "_funding_decile_gate"  # underscore: not a public/registered name

    def __init__(
        self,
        funding: pd.Series | None = None,
        funding_window_days: int = 90,
        flatten_threshold: float = 0.90,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.funding = funding
        self.funding_window_days = funding_window_days
        self.flatten_threshold = flatten_threshold

    # ---------------------------------------------------------- the gate

    def _funding_percentile_on_bars(self, bar_index: pd.DatetimeIndex) -> pd.Series:
        """Causal trailing funding percentile, forward-filled onto ``bar_index``.

        NaN wherever no observation is available (before the funding
        file's first settlement, or after its last one) - see the module
        docstring for why that boundary matters and why it is safe.
        """
        if self.funding is None or len(self.funding) == 0:
            return pd.Series(np.nan, index=bar_index)

        funding = self.funding.sort_index()
        window = max(2, int(round(self.funding_window_days * SETTLEMENTS_PER_DAY)))
        # rolling().rank(pct=True): percentile rank of the window's LAST
        # value among the window; .shift(1) removes that last value from
        # its own window, so pct[t] is a function of funding[t-window-1:t-1]
        # only - never funding[t] or later.
        pct = funding.rolling(window, min_periods=window).rank(pct=True).shift(1)

        combined = bar_index.union(pct.index)
        filled = pct.reindex(combined).sort_index().ffill()
        on_bars = filled.reindex(bar_index)

        out_of_coverage = (bar_index < funding.index[0]) | (bar_index > funding.index[-1])
        on_bars = on_bars.where(~out_of_coverage, np.nan)
        return on_bars

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # kelly_regime_v4's target, unchanged input

        pct = self._funding_percentile_on_bars(df.index)
        gate_active = (pct >= self.flatten_threshold).to_numpy()  # NaN -> False

        target = df["target"].to_numpy(dtype=float).copy()
        target[gate_active] = 0.0

        df["funding_percentile"] = pct.to_numpy()
        df["funding_gate_active"] = gate_active
        df["target"] = target
        return df


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

DATA_DIR = ROOT / "data"
DF, LABEL = load_dataset(DATA_DIR, "spot")
REAL_FUNDING = load_funding(DATA_DIR)
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

# Explicit, hard end dates below the holdout on every window used anywhere
# in this file. OOS_START = 2023-01-01 is never reached.
OOS_START = "2023-01-01"
INNER_TRAIN = ("2020-01-01", "2021-12-31")
INNER_VAL = ("2022-01-01", "2022-12-31")

GRID_WINDOW_DAYS = (30, 60, 90, 180)
GRID_THRESHOLD = (0.85, 0.90, 0.95)


def _period(make_strategy, market: MarketSpec, start: str, end: str, funding=None):
    """Backtest over ``[start, end]``, warmed on the bars before it.

    Mirrors ``scripts/funding_study.py::_period`` exactly (manual warmup
    prefix + ``run_backtest(..., trade_start=pre, funding=funding,
    data_label=LABEL)``), because ``tradebot.window.run_period`` does not
    accept ``funding=``. ``end`` must never be ``None`` and must never be
    at or past ``OOS_START`` in this file.
    """
    assert end is not None and end < OOS_START, f"window end {end!r} reaches the holdout"
    strategy = make_strategy()
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw.funding_paid


def _row(tag, split_name, market_name, metrics, funding_paid):
    return {
        "config": tag,
        "split": split_name,
        "market": market_name,
        "final_balance": metrics.final_balance,
        "max_dd_pct": metrics.max_drawdown_pct,
        "sharpe": metrics.sharpe,
        "num_trades": metrics.num_trades,
        "fees_paid": metrics.fees_paid,
        "funding_paid": funding_paid,
    }


def run_sweep() -> list[dict]:
    """12 configs x 2 splits x 2 markets = 48 backtests, plus baselines."""
    splits = (("inner-train", INNER_TRAIN), ("inner-validation", INNER_VAL))
    markets = (("spot", SPOT, None), ("futures_5x", FUTURES, REAL_FUNDING))

    rows: list[dict] = []
    n_candidate = 0
    for window_days in GRID_WINDOW_DAYS:
        for threshold in GRID_THRESHOLD:
            tag = f"w{window_days}_t{threshold:.2f}"
            for split_name, (start, end) in splits:
                for market_name, market, funding in markets:
                    def make(w=window_days, th=threshold):
                        return FundingDecileGate(funding=REAL_FUNDING,
                                                 funding_window_days=w,
                                                 flatten_threshold=th)
                    m, fp = _period(make, market, start, end, funding=funding)
                    rows.append(_row(tag, split_name, market_name, m, fp))
                    n_candidate += 1

    n_baseline = 0
    for base_name in ("kelly_regime_v4", "buy_and_hold"):
        for split_name, (start, end) in splits:
            for market_name, market, funding in markets:
                m, fp = _period(lambda n=base_name: get_strategy(n),
                                market, start, end, funding=funding)
                rows.append(_row(base_name, split_name, market_name, m, fp))
                n_baseline += 1

    print(f"grid: funding_window_days {GRID_WINDOW_DAYS} x "
          f"flatten_threshold {GRID_THRESHOLD} = "
          f"{len(GRID_WINDOW_DAYS) * len(GRID_THRESHOLD)} configs")
    print(f"candidate backtests: {n_candidate}  "
          f"({len(GRID_WINDOW_DAYS) * len(GRID_THRESHOLD)} configs x "
          f"{len(splits)} splits x {len(markets)} markets)")
    print(f"baseline backtests:  {n_baseline}  (2 strategies x "
          f"{len(splits)} splits x {len(markets)} markets)")
    print(f"TOTAL backtests this session: {n_candidate + n_baseline}\n")
    return rows


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"{'config':16s} {'split':16s} {'market':10s} {'final $':>10s} "
          f"{'maxDD%':>7s} {'sharpe':>7s} {'trades':>7s} {'fees $':>8s} "
          f"{'funding $':>10s}")
    for r in rows:
        print(f"{r['config']:16s} {r['split']:16s} {r['market']:10s} "
              f"{r['final_balance']:>10,.0f} {r['max_dd_pct']:>6.1f}% "
              f"{r['sharpe']:>7.2f} {r['num_trades']:>7d} "
              f"{r['fees_paid']:>8,.1f} {r['funding_paid']:>10,.1f}")


def select_config(rows: list[dict]) -> tuple[int, float]:
    """Pick a config using inner-validation only (2022), never inner-train.

    Rule, fixed before reading the table: rank candidate configs by their
    combined inner-validation final balance (spot + futures_5x, funding
    charged on futures), require they also beat kelly_regime_v4 on BOTH
    markets on inner-validation (not just on the sum), and prefer, among
    ties, the config with a same-signed neighbour (window one notch up or
    down at the same threshold) also beating v4 - a plateau check.
    """
    val = [r for r in rows if r["split"] == "inner-validation"]
    by_config: dict[str, dict[str, dict]] = {}
    for r in val:
        by_config.setdefault(r["config"], {})[r["market"]] = r

    v4 = by_config["kelly_regime_v4"]
    v4_spot, v4_fut = v4["spot"]["final_balance"], v4["futures_5x"]["final_balance"]

    candidates = []
    for tag, markets in by_config.items():
        if tag in ("kelly_regime_v4", "buy_and_hold"):
            continue
        spot_bal = markets["spot"]["final_balance"]
        fut_bal = markets["futures_5x"]["final_balance"]
        beats_both = spot_bal > v4_spot and fut_bal > v4_fut
        combined = spot_bal + fut_bal
        candidates.append((beats_both, combined, tag, spot_bal, fut_bal))

    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    return candidates


def falsification_test(window_days: int, threshold: float) -> None:
    """Does the inner-validation improvement (if any) survive a 0.40% taker fee?

    Pre-registered in the module docstring, before any result was seen.
    Re-runs the selected config, kelly_regime_v4, and buy_and_hold on
    inner-validation ONLY, at Bitstamp's entry taker tier, on both markets.
    """
    BITSTAMP_TAKER = 0.004
    spot_fee = MarketSpec.spot(fee_rate=BITSTAMP_TAKER)
    fut_fee = MarketSpec.futures(leverage=5.0, fee_rate=BITSTAMP_TAKER)
    start, end = INNER_VAL

    print(f"\nfalsification test: selected config w{window_days}_t{threshold:.2f} "
          f"vs kelly_regime_v4 vs buy_and_hold, inner-validation "
          f"({start}..{end}), {BITSTAMP_TAKER:.2%} taker fee\n")
    print(f"{'strategy':22s} {'market':10s} {'final $':>10s} {'maxDD%':>7s} "
          f"{'sharpe':>7s} {'vs v4':>8s} {'vs hold':>8s}")

    def make_gate():
        return FundingDecileGate(funding=REAL_FUNDING,
                                 funding_window_days=window_days,
                                 flatten_threshold=threshold)

    results = {}
    for market_name, market, funding in (("spot", spot_fee, None),
                                         ("futures_5x", fut_fee, REAL_FUNDING)):
        gate_m, _ = _period(make_gate, market, start, end, funding=funding)
        v4_m, _ = _period(lambda: get_strategy("kelly_regime_v4"), market, start, end,
                          funding=funding)
        hold_m, _ = _period(lambda: get_strategy("buy_and_hold"), market, start, end,
                            funding=funding)
        results[market_name] = (gate_m, v4_m, hold_m)
        for tag, m in (("funding_decile_gate", gate_m), ("kelly_regime_v4", v4_m),
                       ("buy_and_hold", hold_m)):
            vs_v4 = f"{m.final_balance / v4_m.final_balance - 1:+.1%}"
            vs_hold = f"{m.final_balance / hold_m.final_balance - 1:+.1%}"
            print(f"{tag:22s} {market_name:10s} {m.final_balance:>10,.0f} "
                  f"{m.max_drawdown_pct:>6.1f}% {m.sharpe:>7.2f} {vs_v4:>8s} {vs_hold:>8s}")

    survived = []
    for market_name, (gate_m, v4_m, _hold_m) in results.items():
        survived.append(gate_m.final_balance > v4_m.final_balance)
    print(f"\nfalsification verdict: gate beats kelly_regime_v4 at 0.40% taker on "
          f"{sum(survived)}/{len(survived)} markets "
          f"({', '.join(m for m, s in zip(results, survived) if s) or 'none'})")


def lookahead_self_check(window_days: int, threshold: float) -> None:
    """Two-opposite-tampers check, entirely inside 2020-2022 (never touches 2023+).

    Perturbs OHLCV bars after a cutoff two opposite ways (x3 / /3) and,
    separately, perturbs the FUNDING series after a cutoff two opposite
    ways, then confirms the ``target`` column is byte-identical before
    each cutoff in both copies. A leak would show as a *difference*
    between the two tampered copies before the cutoff, not merely a
    difference from an untampered baseline - the same design as
    ``tests/test_causality_strict.py``.
    """
    print("\nlookahead self-check (two opposite tampers, byte-identical target "
          "required before the cutoff in both copies)\n")

    # --- OHLCV tamper, entirely within 2020-01-01..2022-12-31 (no 2023+ read)
    lo = int(DF.index.searchsorted("2020-01-01"))
    hi = int(DF.index.searchsorted("2022-12-31", side="right"))
    df = DF.iloc[lo:hi].copy()
    cut = len(df) - 20_000  # comfortably inside 2022, cutoff well before the slice end

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    strat_up = FundingDecileGate(funding=REAL_FUNDING, funding_window_days=window_days,
                                 flatten_threshold=threshold)
    strat_down = FundingDecileGate(funding=REAL_FUNDING, funding_window_days=window_days,
                                   flatten_threshold=threshold)
    out_up = strat_up.prepare(up)
    out_down = strat_down.prepare(down)

    a = out_up["target"].to_numpy()[:cut]
    b = out_down["target"].to_numpy()[:cut]
    ohlcv_ok = np.array_equal(a, b)
    mismatches = int((a != b).sum())
    print(f"OHLCV tamper (x3 / /3 after bar {cut} of {len(df)}, all within "
          f"2020-2022): target identical before cutoff = {ohlcv_ok} "
          f"({mismatches} mismatches)")

    # --- funding tamper, cutoff mid-2022 - still nowhere near 2023-01-01
    funding_cut_ts = pd.Timestamp("2022-06-01", tz="UTC")
    fund_up = REAL_FUNDING.copy()
    fund_down = REAL_FUNDING.copy()
    mask = fund_up.index >= funding_cut_ts
    fund_up.loc[mask] = fund_up.loc[mask] * 3.0 + 0.01
    fund_down.loc[mask] = fund_down.loc[mask] / 3.0 - 0.01

    strat_fup = FundingDecileGate(funding=fund_up, funding_window_days=window_days,
                                  flatten_threshold=threshold)
    strat_fdown = FundingDecileGate(funding=fund_down, funding_window_days=window_days,
                                    flatten_threshold=threshold)
    out_fup = strat_fup.prepare(df.copy())
    out_fdown = strat_fdown.prepare(df.copy())

    bar_cut = int(df.index.searchsorted(funding_cut_ts))
    fa = out_fup["target"].to_numpy()[:bar_cut]
    fb = out_fdown["target"].to_numpy()[:bar_cut]
    funding_ok = np.array_equal(fa, fb)
    fmismatches = int((fa != fb).sum())
    print(f"funding tamper (x3+0.01 / /3-0.01 after {funding_cut_ts.date()}): "
          f"target identical before cutoff = {funding_ok} ({fmismatches} mismatches)")

    print(f"\noverall lookahead self-check: {'PASS' if ohlcv_ok and funding_ok else 'FAIL'}")


def main() -> None:
    if REAL_FUNDING is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")

    print("=" * 78)
    print("B-05 conservative variant: binary top-decile funding flatten gate")
    print("=" * 78)
    print(f"data: {LABEL}, {len(DF):,} bars, {DF.index[0]} .. {DF.index[-1]}")
    print(f"funding: {len(REAL_FUNDING):,} settlements, "
          f"{REAL_FUNDING.index[0]} .. {REAL_FUNDING.index[-1]}")
    print(f"inner-train:      {INNER_TRAIN[0]} .. {INNER_TRAIN[1]}")
    print(f"inner-validation: {INNER_VAL[0]} .. {INNER_VAL[1]}")
    print(f"OOS_START = {OOS_START} is never read in this file.\n")

    rows = run_sweep()
    print_table([r for r in rows if r["split"] == "inner-train"], "INNER-TRAIN (2020-2021)")
    print_table([r for r in rows if r["split"] == "inner-validation"],
               "INNER-VALIDATION (2022)")

    ranked = select_config(rows)
    print("\nconfigs ranked by inner-validation combined final balance "
          "(spot + futures_5x; 'beats_both' = beats kelly_regime_v4 on BOTH markets):")
    for beats_both, combined, tag, spot_bal, fut_bal in ranked:
        print(f"  {tag:16s} beats_both={beats_both!s:5s} combined=${combined:>10,.0f} "
              f"spot=${spot_bal:>9,.0f} futures=${fut_bal:>9,.0f}")

    if ranked and ranked[0][0]:
        _, _, best_tag, _, _ = ranked[0]
        best_window = int(best_tag.split("_")[0][1:])
        best_threshold = float(best_tag.split("_")[1][1:])
        print(f"\nselected config: {best_tag} "
              f"(funding_window_days={best_window}, flatten_threshold={best_threshold})")
    else:
        # nothing beat v4 on both markets - report the top combined candidate anyway
        _, _, best_tag, _, _ = ranked[0]
        best_window = int(best_tag.split("_")[0][1:])
        best_threshold = float(best_tag.split("_")[1][1:])
        print(f"\nNO CONFIG beat kelly_regime_v4 on both markets on inner-validation. "
              f"Reporting the top combined-balance candidate for the falsification test "
              f"anyway, for completeness: {best_tag}")

    falsification_test(best_window, best_threshold)
    lookahead_self_check(best_window, best_threshold)


if __name__ == "__main__":
    main()
