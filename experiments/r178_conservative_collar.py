#!/usr/bin/env python
"""R-178 conservative branch: a literal rolling protective COLLAR on kelly_regime_v4.

Not registered: lives under ``experiments/`` (ROUTINE.md step 5). Promote
into ``src/tradebot/strategies/`` only if it clears the promotion bar in
``experiments/r178_direction.md``.

Mechanism, one sentence (frozen, see ``r178_direction.md``'s "Conservative
branch" and ``r178_shared.py``'s module docstring -- neither is edited by
this file): every 7 days, buy one 10%-OTM put and sell one 10%-OTM call
against ``overlay_frac`` of the account's combined (v4 + overlay) equity,
unconditionally -- ``put_stance ≡ +1``, ``call_stance ≡ -1`` at every bar,
fed once into the shared, unmodified ``r178_shared.simulate_overlay`` --
the standard Israelov & Klein (2016) rules-based equity-index collar,
applied here to a synthetic, causally DVOL-priced BTC/ETH options
structure instead of equity-index options. ``kelly_regime_v4``'s own
vote, sizing and trades are byte-identical to the registered strategy;
only an additive overlay differs.

Citation: Israelov, R. & Klein, M. (2016), "Risk and Return of Equity
Index Collar Strategies," JAI 19(1):41-54 -- rules-based collars
systematically underperform because they give up upside AND pay the
volatility risk premium, a structural drag, not noise. That is exactly
the named failure mode this branch's falsification test checks for.

Window discipline (frozen in r178_direction.md): DVOL coverage starts
2021-03-24, inside ROUTINE.md's own inner-validation window and after
inner-train ends (2020-12-31) -- an inner-train-only sweep would see 0%
DVOL coverage. Both branches therefore iterate/select on bars from
DVOL's first covered day (2021-03-24) through inner-validation's own end
(2022-12-31), NOT inner-train alone. The pre-registered holdout stays
2023-01-01 onward, UNTOUCHED here: every DataFrame this file loads is
truncated to ``.loc[:"2022-12-31"]`` before any backtest or overlay call
runs, so 2023+ bars are never read by this file, not even to size the
overlay's running notional (which is inherently sequential/causal from
each series' own start, so truncating the tail changes nothing about the
pre-truncation numbers).

Configs evaluated by this file: 13 (the frozen BTC futures_5x sweep:
overlay_frac in {0.25,0.50,1.00} x moneyness in {(0.90,1.10),(0.95,1.05)}
x cost_bps in {10,30} = 12, plus the frozen primary overlay_frac=0.50,
moneyness=(0.90,1.10), cost_bps=20) + 3 (frozen primary on BTC spot, ETH
futures_5x, ETH spot) = 16 pre-registered configs, plus 1 extra
(frozen primary, BTC futures_5x, cost_bps=100 cost-sensitivity stress,
explicitly NOT counted against the 16) + falsification-test resample
trials (own bounded window battery, see ``falsification_stress()``).

Usage::

    python experiments/r178_conservative_collar.py sweep   # 13 BTC futures_5x configs
    python experiments/r178_conservative_collar.py other    # frozen primary, 3 other markets
    python experiments/r178_conservative_collar.py cost     # cost_bps=100 stress extra
    python experiments/r178_conservative_collar.py stress   # falsification test
    python experiments/r178_conservative_collar.py all      # everything, in order
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments import r178_shared as shared  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

BARS_PER_DAY = 288
ROLL_BARS = 7 * BARS_PER_DAY

WINDOW_START = "2021-03-24"  # DVOL's own first covered day
WINDOW_END = "2022-12-31"    # inner-validation's own end (frozen holdout starts 2023-01-01)

FROZEN = dict(overlay_frac=0.50, put_moneyness=0.90, call_moneyness=1.10, cost_bps=20.0)

SWEEP_FRACS = (0.25, 0.50, 1.00)
SWEEP_MONEY = ((0.90, 1.10), (0.95, 1.05))
SWEEP_COSTS = (10.0, 30.0)

N_CONFIGS = 0  # distinct configurations evaluated, for the report's own count
CONFIG_LABELS: list[str] = []


def count(label: str) -> None:
    global N_CONFIGS
    if label not in CONFIG_LABELS:
        CONFIG_LABELS.append(label)
        N_CONFIGS += 1


# =========================================================================
# Data -- BTC and ETH, both truncated to end at inner-validation's own end
# (2022-12-31) BEFORE any backtest or overlay call runs, so no 2023+ bar
# is ever read by this file, per the pre-registration's holdout discipline.
# =========================================================================

_BTC_FULL, BTC_LABEL = load_dataset(ROOT / "data", "spot")
BTC_DF = _BTC_FULL.loc[:WINDOW_END].copy()

_ETH_FULL = load_ohlcv_csv(ROOT / "data" / "ethusd_coinbase_spot_5m.csv.gz")
ETH_DF = _ETH_FULL.loc[:WINDOW_END].copy()

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

BTC_CLOSE = BTC_DF["close"].to_numpy(dtype=float)
BTC_SIGMA = shared.load_dvol_sigma(ROOT / "data", shared.BTC_DVOL_FILE, BTC_DF)
ETH_CLOSE = ETH_DF["close"].to_numpy(dtype=float)
ETH_SIGMA = shared.load_dvol_sigma(ROOT / "data", shared.ETH_DVOL_FILE, ETH_DF)

INCUMBENT = "kelly_regime_v4"


# =========================================================================
# Helpers
# =========================================================================


def base_equity_for(df: pd.DataFrame, market: MarketSpec, balance: float = 1_000.0) -> np.ndarray:
    """v4's own unmodified equity curve over `df` (byte-identical strategy,
    no changes). `df` is already truncated to end at 2022-12-31 above."""
    result = run_backtest(get_strategy(INCUMBENT), df, market, balance)
    return result.equity.to_numpy(dtype=float)


def run_overlay(close: np.ndarray, sigma: np.ndarray, base_equity: np.ndarray,
                 overlay_frac: float, put_moneyness: float, call_moneyness: float,
                 cost_bps: float, roll_bars: int = ROLL_BARS) -> dict:
    """One collar config: put_stance ≡ +1 (long put), call_stance ≡ -1
    (short call), at every bar -- unconditional, no stance-switching."""
    n = len(close)
    put_stance = np.ones(n)
    call_stance = -np.ones(n)
    return shared.simulate_overlay(
        close, sigma, base_equity, put_stance, call_stance,
        overlay_frac=overlay_frac, put_moneyness=put_moneyness,
        call_moneyness=call_moneyness, roll_bars=roll_bars, cost_bps=cost_bps,
    )


def window_stats(equity: np.ndarray, index: pd.DatetimeIndex,
                  start: str = WINDOW_START, end: str = WINDOW_END) -> dict:
    """Metrics read ONLY on the [start, end] slice -- window-relative,
    since the strategy has already been compounding since 2017."""
    lo = int(index.searchsorted(pd.Timestamp(start, tz="UTC")))
    hi = int(index.searchsorted(pd.Timestamp(end, tz="UTC"), side="right")) - 1
    seg = equity[lo:hi + 1]
    start_val, end_val = seg[0], seg[-1]
    window_return_pct = (end_val / start_val - 1.0) * 100.0 if start_val > 0 else float("nan")
    return {
        "window_return_pct": window_return_pct,
        "sharpe": shared.sharpe_ratio(seg),
        "max_dd_pct": shared.max_drawdown_pct(seg),
        "lo": lo, "hi": hi,
    }


def config_row(label: str, close: np.ndarray, sigma: np.ndarray, base_equity: np.ndarray,
               index: pd.DatetimeIndex, overlay_frac: float, moneyness: tuple[float, float],
               cost_bps: float) -> dict:
    count(label)
    put_m, call_m = moneyness
    ov = run_overlay(close, sigma, base_equity, overlay_frac, put_m, call_m, cost_bps)
    combined_w = window_stats(ov["combined_equity"], index)
    alone_w = window_stats(base_equity, index)
    return {
        "label": label, "overlay_frac": overlay_frac,
        "put_m": put_m, "call_m": call_m, "cost_bps": cost_bps,
        "combined_return_%": combined_w["window_return_pct"],
        "combined_sharpe": combined_w["sharpe"],
        "combined_maxDD_%": combined_w["max_dd_pct"],
        "v4_return_%": alone_w["window_return_pct"],
        "v4_sharpe": alone_w["sharpe"],
        "v4_maxDD_%": alone_w["max_dd_pct"],
        "num_rolls": ov["num_rolls"], "total_cost": ov["total_cost"],
        "liquidated": ov["liquidated"],
    }


def print_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(df.round(3).to_string(index=False))
    return df


# =========================================================================
# Step: 13-config sweep, BTC futures_5x, [2021-03-24, 2022-12-31]
# =========================================================================


def sweep_btc_futures() -> pd.DataFrame:
    print("=" * 100)
    print("SWEEP -- BTC futures_5x, window 2021-03-24 -> 2022-12-31 (13 configs)")
    print("=" * 100)
    base_eq = base_equity_for(BTC_DF, FUTURES)
    rows = []
    for f in SWEEP_FRACS:
        for m in SWEEP_MONEY:
            for c in SWEEP_COSTS:
                label = f"btc_futures frac={f:.2f} m={m} cost={c:.0f}bps"
                rows.append(config_row(label, BTC_CLOSE, BTC_SIGMA, base_eq, BTC_DF.index, f, m, c))
    label = "btc_futures FROZEN PRIMARY frac=0.50 m=(0.90,1.10) cost=20bps"
    rows.append(config_row(label, BTC_CLOSE, BTC_SIGMA, base_eq, BTC_DF.index,
                            FROZEN["overlay_frac"], (FROZEN["put_moneyness"], FROZEN["call_moneyness"]),
                            FROZEN["cost_bps"]))
    print_table(rows)
    print(f"\nconfigurations evaluated so far: {N_CONFIGS}")
    return pd.DataFrame(rows)


# =========================================================================
# Step: frozen primary on 3 more market/asset combos (BTC spot, ETH
# futures_5x, ETH spot)
# =========================================================================


def other_markets() -> pd.DataFrame:
    print("=" * 100)
    print("FROZEN PRIMARY on 3 other markets: BTC spot, ETH futures_5x, ETH spot")
    print("=" * 100)
    rows = []

    base_eq = base_equity_for(BTC_DF, SPOT)
    rows.append(config_row("btc_spot FROZEN PRIMARY", BTC_CLOSE, BTC_SIGMA, base_eq, BTC_DF.index,
                            FROZEN["overlay_frac"], (FROZEN["put_moneyness"], FROZEN["call_moneyness"]),
                            FROZEN["cost_bps"]))

    base_eq = base_equity_for(ETH_DF, FUTURES)
    rows.append(config_row("eth_futures FROZEN PRIMARY", ETH_CLOSE, ETH_SIGMA, base_eq, ETH_DF.index,
                            FROZEN["overlay_frac"], (FROZEN["put_moneyness"], FROZEN["call_moneyness"]),
                            FROZEN["cost_bps"]))

    base_eq = base_equity_for(ETH_DF, SPOT)
    rows.append(config_row("eth_spot FROZEN PRIMARY", ETH_CLOSE, ETH_SIGMA, base_eq, ETH_DF.index,
                            FROZEN["overlay_frac"], (FROZEN["put_moneyness"], FROZEN["call_moneyness"]),
                            FROZEN["cost_bps"]))

    print_table(rows)
    print(f"\nconfigurations evaluated so far: {N_CONFIGS}")
    return pd.DataFrame(rows)


# =========================================================================
# Extra (not part of the 16, explicitly labeled as such): a 100bps
# cost-sensitivity stress on the frozen primary, since this round has no
# real Deribit order-book data to calibrate cost_bps against.
# =========================================================================


def cost_stress() -> pd.DataFrame:
    print("=" * 100)
    print("EXTRA (not part of the 16) -- frozen primary, BTC futures_5x, cost_bps=100 stress")
    print("=" * 100)
    base_eq = base_equity_for(BTC_DF, FUTURES)
    row = config_row("btc_futures FROZEN PRIMARY cost=100bps (stress, not counted)",
                      BTC_CLOSE, BTC_SIGMA, base_eq, BTC_DF.index,
                      FROZEN["overlay_frac"], (FROZEN["put_moneyness"], FROZEN["call_moneyness"]), 100.0)
    print_table([row])
    return pd.DataFrame([row])


# =========================================================================
# Falsification test, frozen: does the collar measurably reduce max
# drawdown vs. v4 alone over stress-relevant sub-windows within the
# DVOL-covered span?
#
# scripts/stress_test.py's own window battery samples random (start,
# length) pairs ANYWHERE across the full 2017-2026 dataset (warmup ..
# len(df)-length) -- NOT restricted to [2021-03-24, 2022-12-31]. Its
# windows therefore do not fit inside the DVOL-covered span (most would
# include pre-DVOL bars, where the overlay falls back to intrinsic-only
# pricing, or would reach into the 2023+ holdout this file must not
# touch). Per the frozen falsification rule's own fallback clause, this
# resamples an EQUIVALENT bounded battery: same warmup/trade_start
# discipline as stress_test.py's run()/evaluate(), but every window's
# OWN traded region (not its warmup prefix, which may reach back into
# untouched pre-DVOL 2017-2020 data -- that is not holdout, just
# DVOL-less, and is already used freely by inner-train) is constrained
# to fall entirely within [2021-03-24, 2022-12-31].
# =========================================================================


def run_stress_trial(close: np.ndarray, sigma: np.ndarray, df: pd.DataFrame,
                      market: MarketSpec, start_idx: int, length: int, warmup: int) -> dict:
    lo = start_idx - warmup
    hi = start_idx + length
    window = df.iloc[lo:hi]
    close_w = close[lo:hi]
    sigma_w = sigma[lo:hi]
    result = run_backtest(get_strategy(INCUMBENT), window, market, 1_000.0, trade_start=warmup)
    base_eq = result.equity.to_numpy(dtype=float)
    ov = run_overlay(close_w, sigma_w, base_eq, FROZEN["overlay_frac"],
                      FROZEN["put_moneyness"], FROZEN["call_moneyness"], FROZEN["cost_bps"])
    combined = ov["combined_equity"]
    seg_c = combined[warmup:]
    seg_v = base_eq[warmup:]
    if seg_v[0] <= 0 or not np.isfinite(seg_v[0]):
        return {"combined_maxDD": 100.0, "v4_maxDD": 100.0, "combined_sharpe": 0.0,
                "v4_sharpe": 0.0, "liquidated": True}
    return {
        "combined_maxDD": shared.max_drawdown_pct(seg_c),
        "v4_maxDD": shared.max_drawdown_pct(seg_v),
        "combined_sharpe": shared.sharpe_ratio(seg_c),
        "v4_sharpe": shared.sharpe_ratio(seg_v),
        "liquidated": bool(ov["liquidated"]) or bool(result.liquidated),
    }


def falsification_stress(trials: int = 30, min_days: int = 45, max_days: int = 350,
                          seed: int = 178) -> pd.DataFrame:
    print("=" * 100)
    print("FALSIFICATION TEST -- bounded resample within DVOL-covered span "
          f"[{WINDOW_START}, {WINDOW_END}], BTC futures_5x, frozen primary, {trials} trials")
    print("=" * 100)
    strat = get_strategy(INCUMBENT)
    warmup = strat.warmup + 10

    idx = BTC_DF.index
    span_lo = int(idx.searchsorted(pd.Timestamp(WINDOW_START, tz="UTC")))
    span_hi = int(idx.searchsorted(pd.Timestamp(WINDOW_END, tz="UTC"), side="right"))
    span_bars = span_hi - span_lo
    print(f"  DVOL-covered span: {span_bars:,} bars ({span_bars / BARS_PER_DAY:.0f} days) "
          f"from index {span_lo} to {span_hi}; warmup={warmup:,} bars drawn from BEFORE "
          f"{WINDOW_START} (pre-DVOL but not holdout, same as inner-train's own use elsewhere)")

    rng = np.random.default_rng(seed)
    rows = []
    for k in range(trials):
        length = int(rng.integers(min_days, max_days + 1) * BARS_PER_DAY)
        length = min(length, span_bars - BARS_PER_DAY)  # keep the window itself inside the span
        start_idx = int(rng.integers(span_lo, span_hi - length))
        # warmup prefix is allowed to dip before span_lo (pre-DVOL, not holdout);
        # only clamp so it never goes negative.
        w = min(warmup, start_idx)
        stats = run_stress_trial(BTC_CLOSE, BTC_SIGMA, BTC_DF, FUTURES, start_idx, length, w)
        rows.append({"trial": k, "start": idx[start_idx], "days": length // BARS_PER_DAY, **stats})
        print(f"  [{k + 1}/{trials}] {idx[start_idx]:%Y-%m-%d} +{length // BARS_PER_DAY}d  "
              f"combinedDD={stats['combined_maxDD']:5.1f}%  v4DD={stats['v4_maxDD']:5.1f}%  "
              f"{'LIQUIDATED' if stats['liquidated'] else ''}", file=sys.stderr)

    res = pd.DataFrame(rows)
    med_combined_dd = res["combined_maxDD"].median()
    med_v4_dd = res["v4_maxDD"].median()
    dd_reduction = med_v4_dd - med_combined_dd
    med_sharpe_delta = (res["combined_sharpe"] - res["v4_sharpe"]).median()
    liq_rate = res["liquidated"].mean() * 100.0

    print(f"\n  median maxDD  v4 alone = {med_v4_dd:.2f}%   collar+v4 = {med_combined_dd:.2f}%   "
          f"reduction = {dd_reduction:+.2f} pct points")
    print(f"  median Sharpe delta (combined - v4 alone) = {med_sharpe_delta:+.3f}")
    print(f"  liquidation rate across trials = {liq_rate:.1f}%")

    NOISE_FLOOR = 0.20  # R-20's ±0.2 Sharpe-equivalent noise floor
    passes = (dd_reduction > 0) and (abs(med_sharpe_delta) > NOISE_FLOOR) and (liq_rate == 0.0)
    # A drawdown reduction with a Sharpe delta indistinguishable from noise
    # (|delta| <= 0.2) is, per the frozen rule, "statistically
    # indistinguishable" -- the paid VRP bought nothing measurable.
    verdict = "PASS" if passes else "FAIL"
    print(f"\n  FALSIFICATION VERDICT: {verdict}")
    print(f"  (rule: median maxDD must be reduced AND the Sharpe delta must clear the "
          f"±{NOISE_FLOOR} noise floor R-20 established, AND zero liquidations across trials)")
    return res


# =========================================================================
# Sanity check: does the collar's overlay P&L direction make sense given
# BTC's known realized moves in this window (2021 top, 2022 bear)?
# =========================================================================


def sanity_check() -> None:
    print("=" * 100)
    print("SANITY CHECK -- collar overlay P&L direction vs. known BTC 2021-2022 moves")
    print("=" * 100)
    base_eq = base_equity_for(BTC_DF, FUTURES)
    ov = run_overlay(BTC_CLOSE, BTC_SIGMA, base_eq, FROZEN["overlay_frac"],
                      FROZEN["put_moneyness"], FROZEN["call_moneyness"], FROZEN["cost_bps"])
    idx = BTC_DF.index
    lo = int(idx.searchsorted(pd.Timestamp(WINDOW_START, tz="UTC")))
    hi = int(idx.searchsorted(pd.Timestamp(WINDOW_END, tz="UTC"), side="right")) - 1
    overlay_only_pnl = ov["overlay_pnl"][lo:hi + 1].sum()
    price_start, price_end = BTC_CLOSE[lo], BTC_CLOSE[hi]
    price_change_pct = (price_end / price_start - 1.0) * 100.0
    print(f"  BTC close {WINDOW_START} -> {WINDOW_END}: ${price_start:,.0f} -> ${price_end:,.0f} "
          f"({price_change_pct:+.1f}%)")
    print(f"  cumulative overlay-only P&L (window, excludes v4's own position): "
          f"${overlay_only_pnl:+,.0f}  (rolls={ov['num_rolls']}, total_cost=${ov['total_cost']:,.0f})")
    print("  expectation: net premium (short call gives up upside, long put costs premium)\n"
          "  should show a persistent drag most weeks -- Israelov & Klein's structural VRP\n"
          "  cost -- partially offset by the put paying off hard during acute crash weeks\n"
          "  (e.g. the May 2021 -50% crash, the mid-2022 Terra/3AC/FTX-adjacent legs of the\n"
          "  bear). A net-negative cumulative overlay P&L with visible partial offsets during\n"
          "  the sharpest down-weeks would be consistent with the literature; a large positive\n"
          "  overlay P&L over a window BTC fell hard would be the surprising result worth\n"
          "  double-checking, not reporting at face value.")


# =========================================================================
# main
# =========================================================================


if __name__ == "__main__":
    print(f"BTC: {len(BTC_DF):,} bars {BTC_DF.index[0]:%Y-%m-%d} -> {BTC_DF.index[-1]:%Y-%m-%d} "
          f"(data: {BTC_LABEL}, truncated to end {WINDOW_END})", file=sys.stderr)
    print(f"ETH: {len(ETH_DF):,} bars {ETH_DF.index[0]:%Y-%m-%d} -> {ETH_DF.index[-1]:%Y-%m-%d} "
          f"(coinbase spot, truncated to end {WINDOW_END})", file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    t0 = time.time()

    if choice in ("sweep", "all"):
        sweep_btc_futures()
    if choice in ("other", "all"):
        other_markets()
    if choice in ("cost", "all"):
        cost_stress()
    if choice in ("stress", "all"):
        falsification_stress()
    if choice in ("sanity", "all"):
        sanity_check()
    if choice not in ("sweep", "other", "cost", "stress", "sanity", "all"):
        print("usage: python experiments/r178_conservative_collar.py "
              "[sweep|other|cost|stress|sanity|all]")

    print(f"\n[{time.time() - t0:.0f}s]  total configurations evaluated: {N_CONFIGS}")
