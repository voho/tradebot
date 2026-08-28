#!/usr/bin/env python
"""R-178 NOVEL branch (08-28): fitness-switched VRP harvest/hedge, layered
additively on `kelly_regime_v4`'s own unmodified position via the shared
`experiments/r178_shared.py` engine (`vote_frac`, `simulate_overlay`).

Mechanism, one sentence (frozen, unchanged from `r178_direction.md`): every
7 days, read v4's OWN vote `frac` (`r178_shared.vote_frac`, values in
{0, 1/3, 2/3, 1}) at the roll's opening bar -- if `frac <= 1/3` (v4 itself
reads bearish/uncertain) open a LONG strangle (`put_stance=call_stance=+1`,
pay for convexity exactly when the account's own risk read says it is least
confident); if `frac >= 2/3` (v4 confidently bullish) open a SHORT strangle
(`put_stance=call_stance=-1`, harvest the volatility risk premium exactly
when v4's own vote is not hedging anything already). Citation: Gârleanu,
Pedersen & Poteshman (2009), "Demand-Based Option Pricing" -- who needs
insurance, and when, changes the sign of the priced premium; Bakshi &
Kapadia (2003) is the economic basis for the harvest leg specifically
(sellers are compensated richest in high-realized-vol regimes); Alexander &
Imeraj (2021) is the magnitude check (BVRP ~= 0.14) this round's own
honesty check (failure mode (c) in `r178_direction.md`) is watching for.

v4's sizing/vote/trades are BYTE-IDENTICAL to the registered strategy in
every config here -- this file never edits v4's own target, only sums an
independently-sized options sleeve on top of its equity curve, via the
shared, unmodified `r178_shared.simulate_overlay`.

Configs evaluated by this file: 13 (sweep: overlay_frac x moneyness x
cost_bps, BTC futures_5x) + 3 (frozen primary on BTC spot, ETH futures_5x,
ETH spot) = 16, plus one extra stress data point (frozen primary on BTC
futures_5x at cost_bps=100) = 17 backtested configurations, each requiring
one `run_backtest` (v4 alone) and up to two `simulate_overlay` calls (actual
cost_bps, and a cost_bps=0 companion run used only to isolate the in-window
cost contribution by differencing -- see `_cost_in_window` below). BTC's own
`run_backtest`/`simulate_overlay` pair is cached and reused across all 14
BTC configs; same for ETH's 2.

Every pre-holdout number in this file's output is computed on
`[WINDOW_START, OOS_START)` = [2021-03-24, 2023-01-01) ONLY (DVOL's own
coverage starts 2021-03-24; `r178_direction.md`'s Step 1 explains why
inner-train alone would see 0% DVOL coverage and never test the mechanism).
`simulate_overlay` and `run_backtest` themselves still run over each
asset's FULL causal history (2017/2019 -> 2026-08), because the overlay's
notional resizes off a running equity total and needs the full causal
history to price correctly -- but this file never reads or prints a value
derived from a bar at or after OOS_START=2023-01-01 (the pre-registered
holdout); `_assert_no_holdout_read` below is a mechanical guard on every
reported window slice.

Usage
-----
    python experiments/r178_novel_vrp_switch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import paired_bootstrap, total_log_return  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

from experiments.r178_shared import (  # noqa: E402
    BTC_DVOL_FILE,
    DEFAULT_ROLL_BARS,
    ETH_DVOL_FILE,
    load_dvol_sigma,
    log_growth,
    max_drawdown_pct,
    sharpe_ratio,
    simulate_overlay,
    vote_frac,
)

BARS_PER_DAY = 288

WINDOW_START = pd.Timestamp("2021-03-24", tz="UTC")
OOS_START = pd.Timestamp("2023-01-01", tz="UTC")  # pre-registered holdout boundary; never read

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

INCUMBENT = "kelly_regime_v4"
START_BALANCE = 1_000.0

# Frozen primary (identical to the conservative branch's, per r178_direction.md):
PRIMARY = dict(overlay_frac=0.5, put_moneyness=0.90, call_moneyness=1.10, cost_bps=20.0)

# Sweep grid (BTC futures_5x only, per r178_direction.md):
FRAC_GRID = (0.25, 0.50, 1.00)
MONEYNESS_GRID = ((0.90, 1.10), (0.95, 1.05))
COST_GRID = (10.0, 30.0)

# Falsification test's block-bootstrap settings, frozen here before any number is read.
# Block length: 3 days of 5m bars (864 bars). Chosen because the overlay's own cash
# flows realize discretely at each 2,016-bar (7-day) roll with within-cycle
# Black-Scholes path dependence, and DVOL itself only updates once/day (see
# r178_shared.align_dvol_causal) -- a few days is long enough to swallow that
# short-horizon serial correlation without shrinking to so few blocks (the window is
# ~648 days -> ~216 blocks of this length) that the percentile CI itself becomes noisy.
BOOT_BLOCK_BARS = 3 * BARS_PER_DAY
BOOT_N = 2_000
BOOT_SEED = 178


# --------------------------------------------------------------------------- guards


def _assert_no_holdout_read(index: pd.DatetimeIndex, lo: int, hi: int, label: str) -> None:
    """Fail loudly if the reported window slice reaches OOS_START."""
    if hi > lo and index[hi - 1] >= OOS_START:
        raise AssertionError(f"{label}: window slice reaches {index[hi - 1]}, at/after {OOS_START}")


def _window_bounds(index: pd.DatetimeIndex) -> tuple[int, int]:
    lo = int(index.searchsorted(WINDOW_START))
    hi = int(index.searchsorted(OOS_START))  # exclusive upper bound
    _assert_no_holdout_read(index, lo, hi, "window_bounds")
    return lo, hi


def _roll_open_bars(n: int, roll_bars: int) -> np.ndarray:
    """Deterministic roll-opening bar indices, matching `simulate_overlay`'s own
    `i0 = 0; while i0 < n - 1: ...; i0 = i1` schedule exactly (pure index
    arithmetic -- no data dependence, so this cannot itself read any bar's value)."""
    outs = []
    i0 = 0
    while i0 < n - 1:
        outs.append(i0)
        i1 = min(i0 + roll_bars, n - 1)
        i0 = i1
    return np.asarray(outs, dtype=int)


def _bar_returns(equity: np.ndarray) -> np.ndarray:
    prev = equity[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(prev > 0, np.diff(equity) / prev, 0.0)


# --------------------------------------------------------------------------- data


def load_asset(price_df: pd.DataFrame, dvol_file: str, label: str) -> dict:
    """One asset's full causal panel: close, DVOL sigma, v4's own vote/frac,
    stance, and v4's own unmodified equity curves on both markets."""
    close = price_df["close"].to_numpy(dtype=float)
    sigma = load_dvol_sigma(ROOT / "data", dvol_file, price_df)
    frac = vote_frac(price_df["close"])
    stance = np.where(frac <= 1.0 / 3.0 + 1e-9, 1.0, -1.0)

    base_fut = run_backtest(get_strategy(INCUMBENT), price_df, FUTURES, START_BALANCE,
                             data_label=label).equity.to_numpy(dtype=float)
    base_spot = run_backtest(get_strategy(INCUMBENT), price_df, SPOT, START_BALANCE,
                              data_label=label).equity.to_numpy(dtype=float)

    return dict(index=price_df.index, close=close, sigma=sigma, frac=frac, stance=stance,
                base_fut=base_fut, base_spot=base_spot)


# --------------------------------------------------------------------------- one config


def run_config(asset: dict, market_name: str, *, overlay_frac: float,
                put_moneyness: float, call_moneyness: float, cost_bps: float) -> dict:
    """Run one (asset, market, params) config; simulate_overlay sees the FULL causal
    history (required for correct causal pricing/notional-resizing), but every value
    returned here is read off the [WINDOW_START, OOS_START) slice only."""
    index = asset["index"]
    close, sigma, stance = asset["close"], asset["sigma"], asset["stance"]
    base_equity = asset["base_fut"] if market_name == "futures_5x" else asset["base_spot"]

    res = simulate_overlay(close, sigma, base_equity, stance, stance, overlay_frac,
                            put_moneyness=put_moneyness, call_moneyness=call_moneyness,
                            roll_bars=DEFAULT_ROLL_BARS, cost_bps=cost_bps)
    # Companion zero-cost run, used ONLY to isolate the in-window cost contribution
    # by differencing (see module docstring) -- simulate_overlay exposes only a
    # full-9-year `total_cost` aggregate, which would itself read holdout bars.
    res0 = simulate_overlay(close, sigma, base_equity, stance, stance, overlay_frac,
                             put_moneyness=put_moneyness, call_moneyness=call_moneyness,
                             roll_bars=DEFAULT_ROLL_BARS, cost_bps=0.0)

    lo, hi = _window_bounds(index)
    combined = res["combined_equity"]
    combined_nc = res0["combined_equity"]

    combined_w = combined[lo:hi]
    base_w = base_equity[lo:hi]
    _assert_no_holdout_read(index, lo, hi, "run_config")

    opens = _roll_open_bars(len(close), DEFAULT_ROLL_BARS)
    opens_w = opens[(opens >= lo) & (opens < hi)]
    num_rolls_w = int(len(opens_w))
    stance_w = stance[opens_w] if num_rolls_w else np.array([])
    frac_seller = float(np.mean(stance_w < 0)) if num_rolls_w else float("nan")
    frac_buyer = float(np.mean(stance_w > 0)) if num_rolls_w else float("nan")

    # cost realized within the window: (no-cost combined) minus (costed combined) is a
    # running cumulative total (both share the same base_equity, so the difference is
    # exactly cumsum(overlay_pnl_nocost - overlay_pnl_costed)); the window's own share
    # is that cumulative total's own increment across [lo, hi) -- see the module
    # docstring's note on the small second-order compounding residual.
    diff_cum = combined_nc - combined
    start_val = float(diff_cum[lo - 1]) if lo > 0 else 0.0
    cost_w = float(diff_cum[hi - 1] - start_val) if hi > lo else 0.0

    liquidated_w = bool(np.any(combined_w <= 0))

    combined_final = 1000.0 * (combined_w[-1] / combined_w[0]) if combined_w[0] > 0 else float("nan")
    base_final = 1000.0 * (base_w[-1] / base_w[0]) if base_w[0] > 0 else float("nan")

    return dict(
        combined_equity_window=combined_w, base_equity_window=base_w,
        combined_final_1000=combined_final, base_final_1000=base_final,
        combined_sharpe=sharpe_ratio(combined_w), base_sharpe=sharpe_ratio(base_w),
        combined_maxdd=max_drawdown_pct(combined_w), base_maxdd=max_drawdown_pct(base_w),
        num_rolls_window=num_rolls_w, cost_window=cost_w,
        liquidated_window=liquidated_w,
        frac_buyer=frac_buyer, frac_seller=frac_seller,
    )


# --------------------------------------------------------------------------- falsification


def falsification(asset: dict, market_name: str = "futures_5x") -> tuple:
    """Paired stationary-block-bootstrap 95% CI on Delta-log-growth (combined overlay
    vs. v4 alone), frozen primary config, window-restricted. Returns
    (PairedResult, lo, hi) for one asset."""
    out = run_config(asset, market_name, **PRIMARY)
    combined_w = out["combined_equity_window"]
    base_w = out["base_equity_window"]
    rc = _bar_returns(combined_w)
    rb = _bar_returns(base_w)
    result = paired_bootstrap(rc, rb, total_log_return, mean_block=BOOT_BLOCK_BARS,
                               n_boot=BOOT_N, seed=BOOT_SEED)
    return result, out


# --------------------------------------------------------------------------------- main


def sweep_table(btc: dict) -> pd.DataFrame:
    rows = []
    for frac in FRAC_GRID:
        for (pm, cm) in MONEYNESS_GRID:
            for cb in COST_GRID:
                out = run_config(btc, "futures_5x", overlay_frac=frac, put_moneyness=pm,
                                  call_moneyness=cm, cost_bps=cb)
                rows.append(dict(asset="BTC", market="futures_5x", overlay_frac=frac,
                                  put_m=pm, call_m=cm, cost_bps=cb, primary=False, **out))
    out = run_config(btc, "futures_5x", **PRIMARY)
    rows.append(dict(asset="BTC", market="futures_5x", overlay_frac=PRIMARY["overlay_frac"],
                      put_m=PRIMARY["put_moneyness"], call_m=PRIMARY["call_moneyness"],
                      cost_bps=PRIMARY["cost_bps"], primary=True, **out))
    return pd.DataFrame(rows)


def main() -> None:
    btc_df, btc_label = load_dataset(ROOT / "data", "spot")
    eth_df = load_ohlcv_csv(ROOT / "data" / "ethusd_coinbase_spot_5m.csv.gz")

    print(f"BTC: {len(btc_df):,} bars {btc_df.index[0]:%Y-%m-%d} -> {btc_df.index[-1]:%Y-%m-%d} "
          f"(data: {btc_label})", file=sys.stderr)
    print(f"ETH: {len(eth_df):,} bars {eth_df.index[0]:%Y-%m-%d} -> {eth_df.index[-1]:%Y-%m-%d}",
          file=sys.stderr)

    btc = load_asset(btc_df, BTC_DVOL_FILE, btc_label)
    eth = load_asset(eth_df, ETH_DVOL_FILE, "real")

    print("\n=== BTC futures_5x sweep (13 configs) ===")
    sweep = sweep_table(btc)
    for _, r in sweep.iterrows():
        tag = "PRIMARY" if r["primary"] else "       "
        print(f"  {tag} frac={r['overlay_frac']:.2f} m=({r['put_m']:.2f},{r['call_m']:.2f}) "
              f"cost={r['cost_bps']:>3.0f}bps  "
              f"combined=${r['combined_final_1000']:>8,.0f} (v4 alone=${r['base_final_1000']:>8,.0f})  "
              f"sharpe={r['combined_sharpe']:>6.2f} (v4={r['base_sharpe']:>6.2f})  "
              f"DD={r['combined_maxdd']:>5.1f}% (v4={r['base_maxdd']:>5.1f}%)  "
              f"rolls={r['num_rolls_window']:>4d} cost=${r['cost_window']:>8,.0f}  "
              f"buyer={r['frac_buyer']:.2f} seller={r['frac_seller']:.2f}"
              f"{'  LIQUIDATED' if r['liquidated_window'] else ''}")

    print("\n=== frozen primary, other markets/assets (3 configs) ===")
    extra_rows = []
    for asset_name, asset, market_name in (("BTC", btc, "spot"), ("ETH", eth, "futures_5x"),
                                            ("ETH", eth, "spot")):
        out = run_config(asset, market_name, **PRIMARY)
        extra_rows.append(dict(asset=asset_name, market=market_name, **PRIMARY, primary=True, **out))
        print(f"  {asset_name} {market_name:12s}  combined=${out['combined_final_1000']:>8,.0f} "
              f"(v4 alone=${out['base_final_1000']:>8,.0f})  "
              f"sharpe={out['combined_sharpe']:>6.2f} (v4={out['base_sharpe']:>6.2f})  "
              f"DD={out['combined_maxdd']:>5.1f}% (v4={out['base_maxdd']:>5.1f}%)  "
              f"rolls={out['num_rolls_window']:>4d} cost=${out['cost_window']:>8,.0f}  "
              f"buyer={out['frac_buyer']:.2f} seller={out['frac_seller']:.2f}"
              f"{'  LIQUIDATED' if out['liquidated_window'] else ''}")

    print("\n=== stress data point: frozen primary, BTC futures_5x, cost_bps=100 ===")
    stress_cfg = dict(PRIMARY)
    stress_cfg["cost_bps"] = 100.0
    out = run_config(btc, "futures_5x", **stress_cfg)
    print(f"  combined=${out['combined_final_1000']:>8,.0f} (v4 alone=${out['base_final_1000']:>8,.0f})  "
          f"sharpe={out['combined_sharpe']:>6.2f} (v4={out['base_sharpe']:>6.2f})  "
          f"DD={out['combined_maxdd']:>5.1f}% (v4={out['base_maxdd']:>5.1f}%)  "
          f"rolls={out['num_rolls_window']:>4d} cost=${out['cost_window']:>8,.0f}  "
          f"buyer={out['frac_buyer']:.2f} seller={out['frac_seller']:.2f}"
          f"{'  LIQUIDATED' if out['liquidated_window'] else ''}")

    print("\n=== falsification: paired stationary-block-bootstrap 95% CI on "
          "Delta-log-growth, frozen primary, futures_5x ===")
    print(f"  block length={BOOT_BLOCK_BARS} bars (3d), n_boot={BOOT_N}, seed={BOOT_SEED}")
    verdicts = {}
    for name, asset in (("BTC", btc), ("ETH", eth)):
        result, out = falsification(asset, "futures_5x")
        verdicts[name] = result
        print(f"  {name}: stat(combined)={result.stat_a:+.4f} stat(v4 alone)={result.stat_b:+.4f}  "
              f"diff={result.diff}  p(diff>0)={result.p_positive:.3f}  "
              f"significant={result.significant}")

    both_sig = all(v.significant for v in verdicts.values())
    same_sign = len({np.sign(v.diff.point) for v in verdicts.values()}) == 1
    passed = both_sig and same_sign
    print(f"\n  FROZEN FALSIFICATION VERDICT: {'PASS' if passed else 'FAIL'} "
          f"(both markets significant: {both_sig}; same sign: {same_sign})")

    OUT = ROOT / "reports" / "r178_novel_vrp_switch"
    OUT.mkdir(parents=True, exist_ok=True)
    sweep.drop(columns=["combined_equity_window", "base_equity_window"]).to_csv(
        OUT / "sweep_btc_futures.csv", index=False)
    print(f"\nwritten: {OUT / 'sweep_btc_futures.csv'}")


if __name__ == "__main__":
    main()
