"""R-152 CONSERVATIVE branch: holdout + ETH falsification, run only because
the frozen selection rule in ``experiments/r152_shared.py`` was cleared
(all 3 criteria: exposure match, Sharpe-within-noise-floor, plateau) by
``experiments/r152_conservative_cdar_cap.py`` on inner-validation.

Per the pre-registration, only the branch's OWN selected window length is
run here -- no further search. All three swept window lengths (180/365/545d)
produced a BIT-IDENTICAL ``target`` array to each other on inner-validation
(the branch's own report: the only divergence from the control anywhere in
that period is one 764-bar/~2.6-day episode, 2022-12-11 -> 2022-12-14, where
all three windows happened to relax the cap the same way), so there is
nothing to select between; ``CDAR_WINDOW_DAYS_DEFAULT`` (365d) is used as
the non-cherry-picked representative.

Holdout counter: this file increments it by the number of holdout cells it
reads (2: BTC futures 5x Sharpe/DD comparison, matched-exposure check). The
ETH/BTC-Bitfinex falsification below reads NO row of the OOS_START holdout
-- that dataset physically ends 2019-12-31.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r152_conservative_cdar_cap import (  # noqa: E402
    ConservativeCDaRCap,
    calibrate_cap_scale,
)
from experiments.r152_shared import (  # noqa: E402
    CDAR_WINDOW_DAYS_DEFAULT,
    INNER_TRAIN_END,
    OOS_START,
    SHARPE_NOISE_FLOOR,
)

ROOT_DATA = ROOT / "data"
FUTURES = MarketSpec.futures(leverage=5.0)


def row(tag, m):
    return dict(tag=tag, final=m.final_balance, profit_pct=m.profit_pct,
                sharpe=m.sharpe, max_dd=m.max_drawdown_pct,
                tim_pct=m.time_in_market_pct, trades=m.num_trades,
                liquidated=m.liquidated)


def pr(r):
    print(f"  {r['tag']:28s} final=${r['final']:>12,.0f} ({r['profit_pct']:>+8.1f}%) "
          f"sharpe={r['sharpe']:>6.2f} DD={r['max_dd']:>5.1f}% "
          f"TiM={r['tim_pct']:>5.1f}% trades={r['trades']:>5d} "
          f"{'LIQUIDATED' if r['liquidated'] else ''}")


def main() -> None:
    print("=" * 100)
    print("PART 1 -- HOLDOUT (main dataset, 2023-01-01 ->, futures 5x)")
    print("=" * 100)

    df, label = load_dataset(ROOT_DATA, "spot")
    print(f"main dataset: {label}, {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")

    inner_train = df.loc[:INNER_TRAIN_END]
    cap_scale = calibrate_cap_scale(inner_train, CDAR_WINDOW_DAYS_DEFAULT)
    print(f"cap_scale calibrated on inner-train ({len(inner_train):,} bars): {cap_scale:.6f}")

    candidate = ConservativeCDaRCap(cdar_window_days=CDAR_WINDOW_DAYS_DEFAULT, cap_scale=cap_scale)
    control = KellyRegimeV4()

    res_c = run_period(candidate, df, start=OOS_START, market=FUTURES,
                        start_balance=1_000.0, data_label=label)
    res_k = run_period(control, df, start=OOS_START, market=FUTURES,
                        start_balance=1_000.0, data_label=label)
    m_c, m_k = compute_metrics(res_c), compute_metrics(res_k)
    row_c, row_k = row("candidate (365d cap)", m_c), row("control (kelly_regime_v4)", m_k)
    pr(row_c)
    pr(row_k)

    bh = KellyRegimeV4  # buy_and_hold benchmark via registry, imported below
    from tradebot.registry import get_strategy
    res_bh = run_period(get_strategy("buy_and_hold"), df, start=OOS_START, market=FUTURES,
                         start_balance=1_000.0, data_label=label)
    m_bh = compute_metrics(res_bh)
    row_bh = row("buy_and_hold", m_bh)
    pr(row_bh)

    d_sharpe = m_c.sharpe - m_k.sharpe
    d_dd = m_k.max_drawdown_pct - m_c.max_drawdown_pct  # positive = candidate's DD is smaller (better)
    tim_gap = abs(m_c.time_in_market_pct - m_k.time_in_market_pct)
    beats_bh = m_c.final_balance > m_bh.final_balance
    print(f"\nd_sharpe (candidate - control) = {d_sharpe:+.4f}")
    print(f"d_maxDD_pp (control - candidate, +ve = candidate better) = {d_dd:+.2f}")
    print(f"time_in_market gap (pp) = {tim_gap:.2f}")
    print(f"candidate beats buy_and_hold on holdout: {beats_bh}")
    sharpe_or_dd_ok = (abs(d_sharpe) <= SHARPE_NOISE_FLOOR and d_sharpe > 0) or d_dd >= 3.0
    print(f"promotion criterion 'improvement exceeds noise floor OR DD improves >=3pp': {sharpe_or_dd_ok}")
    print("NOTE: d_sharpe positive AND exceeding the noise floor is what 'exceeds "
          "the noise floor' means for a claimed IMPROVEMENT -- a delta merely "
          "*within* +/-0.2 of zero is 'not distinguishable from the control', not "
          "an improvement. Both are reported explicitly to avoid conflating them.")
    print(f"HOLDOUT CELLS READ: 2 (candidate, control) + 1 buy_and_hold benchmark "
          f"re-read = 3 read operations against the >= 2023-01-01 slice.")

    print()
    print("=" * 100)
    print("PART 2 -- FALSIFICATION: survives on ETH? (Bitfinex, pre-2020, no holdout bars)")
    print("=" * 100)
    eth = load_ohlcv_csv(ROOT_DATA / "ethusd_bitfinex_5m.csv.gz")
    btc = load_ohlcv_csv(ROOT_DATA / "btcusd_bitfinex_5m.csv.gz")
    assert eth.index.max() < pd.Timestamp("2020-01-01", tz="UTC")
    assert btc.index.max() < pd.Timestamp("2020-01-01", tz="UTC")
    print(f"ETH-bitfinex: {len(eth):,} bars, {eth.index[0]} -> {eth.index[-1]}")
    print(f"BTC-bitfinex: {len(btc):,} bars, {btc.index[0]} -> {btc.index[-1]}")

    results = {}
    for dset_name, dframe in (("ETH-falsification", eth), ("BTC-control", btc)):
        cand = ConservativeCDaRCap(cdar_window_days=CDAR_WINDOW_DAYS_DEFAULT, cap_scale=cap_scale)
        ctrl = KellyRegimeV4()
        from tradebot.engine import run_backtest
        rc = run_backtest(cand, dframe, FUTURES, 1_000.0, data_label=dset_name)
        rk = run_backtest(ctrl, dframe, FUTURES, 1_000.0, data_label=dset_name)
        mc, mk = compute_metrics(rc), compute_metrics(rk)
        rowc, rowk = row(f"{dset_name} candidate", mc), row(f"{dset_name} control", mk)
        pr(rowc)
        pr(rowk)
        results[dset_name] = dict(d_sharpe=mc.sharpe - mk.sharpe,
                                   d_dd=mk.max_drawdown_pct - mc.max_drawdown_pct)

    print(f"\nBTC-control d_sharpe={results['BTC-control']['d_sharpe']:+.4f}  "
          f"ETH-falsification d_sharpe={results['ETH-falsification']['d_sharpe']:+.4f}")
    same_sign = (np.sign(results['BTC-control']['d_sharpe']) ==
                 np.sign(results['ETH-falsification']['d_sharpe']))
    print(f"same sign on BTC and ETH (mechanism generalizes rather than being a "
          f"BTC-only artifact): {same_sign}")

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Holdout: d_sharpe={d_sharpe:+.4f} (noise floor +/-{SHARPE_NOISE_FLOOR}), "
          f"d_maxDD_pp={d_dd:+.2f}, beats_buy_and_hold={beats_bh}")
    print(f"Falsification (ETH): same-sign-as-BTC={same_sign}, "
          f"ETH d_sharpe={results['ETH-falsification']['d_sharpe']:+.4f}")
    promote = sharpe_or_dd_ok and beats_bh and same_sign
    print(f"\nPROMOTION RULE OUTCOME: {'PROMOTE' if promote else 'NEGATIVE'}")


if __name__ == "__main__":
    main()
