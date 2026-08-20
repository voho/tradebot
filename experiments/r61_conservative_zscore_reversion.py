"""R-61 conservative branch: multi-horizon z-score mean-reversion vote,
reusing `kelly_regime`'s fractional-Kelly vol-targeted sizing byte-identical.

See `experiments/r61_shared.py` for the full pre-registration (mechanism,
literature, windows, decision rules D1-D5, promotion bar). This file only
implements the candidate strategy and runs the frozen protocol against it.

Not registered (no ``@register``): this is an experiment, not a shipped
strategy, per ROUTINE.md step 5. `src/tradebot/strategies/` is untouched.

Usage::

    python experiments/r61_conservative_zscore_reversion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments import r61_shared as shared  # noqa: E402
from experiments.r57_cross_asset_panel import measure  # noqa: E402
from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

DATA_DIR = ROOT / "data"


class KellyRegimeV61ZscoreReversion(Strategy):
    """Fractional-Kelly exposure gated by a multi-horizon z-score reversion vote.

    Same sizing machinery as `KellyRegime` (fractional-Kelly, vol-targeted,
    10% deadband) — copied byte-identical from `KellyRegime.prepare`/
    `on_bar` below. The ONLY change from `KellyRegime` is how `frac` (the
    0..1 latched vote) is computed: instead of three slow trend anchors
    (price vs. 30/50/100-day mean, vote bullish above), this votes bullish
    ("buy the dip") when price sits more than `z_thresh` standard
    deviations BELOW a short rolling mean, at three short horizons matched
    to Zaremba et al. (2021)'s daily reversal timescale
    (`r61_shared.Z_HORIZONS_DAYS = (1, 3, 7)` days), not v4's 20/40/80-day
    trend ladder. See `experiments/r61_shared.py` for the full citation and
    the pre-registered non-duplicate reasoning against this project's prior
    mean-reversion strategies (`overshoot_fade`, `attrition_reversion`,
    `rsi_reversion`) and against the twenty-one prior TREND-vote retunes of
    `kelly_regime_v4` (R-34..R-46, R-53..R-56, R-59, R-60) — none of those
    twenty-one touched the vote's sign rule, and none of the three prior
    reversion strategies used this fractional-Kelly vol-targeted sizing
    with a latching multi-horizon vote.
    """

    name = "kelly_regime_v61_zscore_reversion"
    # Longest z-score window is 7 days; generous margin, still far under
    # v4's own 100-day warmup since this signal's own lookback is short.
    warmup = 7 * BARS_PER_DAY + 10

    def __init__(self, z_thresh: float = 1.5,
                 target_vol: float = shared.TARGET_VOL,
                 max_leverage: float = shared.MAX_LEVERAGE,
                 vol_span: int = shared.VOL_SPAN_DAYS * BARS_PER_DAY,
                 deadband: float = shared.DEADBAND) -> None:
        self.z_thresh = z_thresh
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # Reversion vote: latch bullish (buy the dip) when price is more
        # than z_thresh std devs BELOW its short rolling mean, bearish
        # (stand aside) when more than z_thresh ABOVE, hold the previous
        # verdict in between (hysteresis, identical pattern to KellyRegime's
        # trend vote — only the underlying statistic and its sign differ).
        # Purely causal: rolling().mean()/.std() at row i use only rows
        # <= i, no .shift(-1) or forward index math anywhere in this method.
        votes = []
        for days in shared.Z_HORIZONS_DAYS:
            window = int(days * BARS_PER_DAY)
            roll_mean = close.rolling(window).mean()
            roll_std = close.rolling(window).std()
            z = (close - roll_mean) / roll_std
            v = pd.Series(
                np.where(z < -self.z_thresh, 1.0,
                         np.where(z > self.z_thresh, 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()

        # Fractional-Kelly sizing: exposure ~ target_vol / realized_vol.
        # Copied byte-identical from KellyRegime.prepare.
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            v = vol[i]
            scale = min(self.target_vol / v, self.max_leverage) if np.isfinite(v) and v > 0 else 0.0
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures


# ------------------------------------------------------------------ causality


def causality_check() -> bool:
    """Truncation test: target[i] for an early row must be identical whether
    or not later rows exist in the frame, since prepare() only ever reads
    close[0..i] through .rolling()/.mean()/.std() (no .shift(-1), no
    forward index math). Spot-checked directly, plus the project's
    standard tamper probe (corrupt the tail, confirm earlier decisions
    are unaffected).
    """
    print("=" * 100)
    print("CAUSALITY CHECK — truncation test + tamper probe")
    print("=" * 100)
    btc, _ = load_dataset(DATA_DIR, "spot")
    df = btc.iloc[-60_000:].copy()

    strat = KellyRegimeV61ZscoreReversion(z_thresh=1.5)
    full = strat.prepare(df.copy())

    ok_trunc = True
    check_points = [len(df) - 30_000, len(df) - 10_000, len(df) - 1]
    for cut in check_points:
        truncated = df.iloc[: cut + 1].copy()
        trunc_prepared = KellyRegimeV61ZscoreReversion(z_thresh=1.5).prepare(truncated)
        early = min(cut - 5000, cut)  # compare a row well before the truncation edge
        a = full["target"].iloc[early]
        b = trunc_prepared["target"].iloc[early]
        same = np.isclose(a, b, atol=1e-12)
        ok_trunc = ok_trunc and same
        print(f"  truncate at {cut}: target[{early}] full={a:.10f} "
              f"truncated={b:.10f} {'MATCH' if same else 'MISMATCH'}")

    # Tamper probe, project convention (tests/test_causality_strict.py):
    # corrupt only the tail, confirm decisions at/before the cut are identical
    # under two OPPOSITE tampers.
    from tradebot.broker import MarketSpec, PaperBroker

    tail = df.copy()
    cut = len(tail) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20)]
    up, down = tail.copy(), tail.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        s = KellyRegimeV61ZscoreReversion(z_thresh=1.5)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=MarketSpec.futures(leverage=5.0), start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    ok_tamper = all(x == y for x, y in zip(decisions(up), decisions(down)))
    print(f"  tamper probe (opposite tail corruptions, decisions at/before cut "
          f"identical): {'PASS' if ok_tamper else 'FAIL'}")

    ok = ok_trunc and ok_tamper
    print(f"\nCAUSALITY CHECK: {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------------ D1/D2


def run_d1_d2_sweep(panel):
    print("\n" + "=" * 100)
    print("D1/D2 SWEEP — Z_THRESH_GRID x 6 panel assets, PANEL_TRAIN, spot @0.10%")
    print("=" * 100)
    start, end = shared.PANEL_TRAIN
    market = shared.SPOT_BASE

    hold_by_asset = {}
    for a in panel:
        _, hold = measure(get_strategy("buy_and_hold"), a.df, start, end, market)
        hold_by_asset[a.ticker] = hold

    grid_rows = []
    for z_thresh in shared.Z_THRESH_GRID:
        d1 = 0
        d2 = 0
        dd_deltas = []
        for a in panel:
            strat = KellyRegimeV61ZscoreReversion(z_thresh=z_thresh)
            _, cand = measure(strat, a.df, start, end, market)
            hold = hold_by_asset[a.ticker]
            beats_final = cand.final_balance > hold.final_balance
            beats_dd = cand.max_drawdown_pct < hold.max_drawdown_pct
            d1 += int(beats_final)
            d2 += int(beats_dd)
            dd_delta = cand.max_drawdown_pct - hold.max_drawdown_pct
            dd_deltas.append(dd_delta)
            row = {
                "z_thresh": z_thresh, "asset": a.ticker,
                "cand_final": cand.final_balance, "cand_dd": cand.max_drawdown_pct,
                "cand_sharpe": cand.sharpe,
                "hold_final": hold.final_balance, "hold_dd": hold.max_drawdown_pct,
                "beats_final": beats_final, "beats_dd": beats_dd,
                "dd_delta": dd_delta,
            }
            grid_rows.append(row)
            print(f"  z={z_thresh:.1f} {a.ticker:5s} cand ${cand.final_balance:>10,.0f} "
                  f"DD {cand.max_drawdown_pct:5.1f}% Sharpe {cand.sharpe:5.2f} | "
                  f"hold ${hold.final_balance:>10,.0f} DD {hold.max_drawdown_pct:5.1f}% | "
                  f"beats_final={beats_final} beats_dd={beats_dd}")
        mean_dd_delta = float(np.mean(dd_deltas))
        print(f"  -> z={z_thresh:.1f}: D1={d1}/6  D2={d2}/6  mean(dDD)={mean_dd_delta:+.2f}pp\n")

    return pd.DataFrame(grid_rows), hold_by_asset


def select_z_thresh(grid_df: pd.DataFrame) -> tuple[float, int, int]:
    """Highest D1 wins; ties broken by mean Delta max-drawdown across the 6
    assets (more negative = candidate draws down less relative to hold =
    better), per the pre-registration."""
    summary = []
    for z_thresh in shared.Z_THRESH_GRID:
        sub = grid_df[grid_df.z_thresh == z_thresh]
        d1 = int(sub.beats_final.sum())
        d2 = int(sub.beats_dd.sum())
        mean_dd_delta = float(sub.dd_delta.mean())
        summary.append((z_thresh, d1, d2, mean_dd_delta))
    # sort by (-D1, mean_dd_delta): highest D1 first, then most negative dd delta
    summary.sort(key=lambda t: (-t[1], t[3]))
    best = summary[0]
    return best[0], best[1], best[2]


# ------------------------------------------------------------------------ D3


def run_d3(z_thresh: float):
    print("\n" + "=" * 100)
    print(f"D3 — BTC/ETH FALSIFICATION, candidate(z={z_thresh}) vs buy_and_hold vs "
          f"kelly_regime_v4, spot @0.10%")
    print("=" * 100)
    btc, _ = load_dataset(DATA_DIR, "spot")
    eth = load_coinbase_eth_spot(DATA_DIR)

    windows = [
        ("BTC_INNER_TRAIN", btc, shared.BTC_INNER_TRAIN),
        ("BTC_INNER_VALID", btc, shared.BTC_INNER_VALID),
        ("ETH_FULL", eth, (None, None)),
    ]
    market = shared.SPOT_BASE
    rows = []
    for label, df, (start, end) in windows:
        if df is None:
            print(f"  {label}: data missing, skipped")
            continue
        cand = KellyRegimeV61ZscoreReversion(z_thresh=z_thresh)
        _, cand_m = measure(cand, df, start, end, market)
        _, hold_m = measure(get_strategy("buy_and_hold"), df, start, end, market)
        _, v4_m = measure(get_strategy("kelly_regime_v4"), df, start, end, market)
        row = {
            "window": label,
            "cand_final": cand_m.final_balance, "cand_dd": cand_m.max_drawdown_pct,
            "cand_sharpe": cand_m.sharpe,
            "hold_final": hold_m.final_balance, "hold_dd": hold_m.max_drawdown_pct,
            "hold_sharpe": hold_m.sharpe,
            "v4_final": v4_m.final_balance, "v4_dd": v4_m.max_drawdown_pct,
            "v4_sharpe": v4_m.sharpe,
        }
        rows.append(row)
        print(f"  {label:16s} cand ${cand_m.final_balance:>10,.0f} DD "
              f"{cand_m.max_drawdown_pct:5.1f}% Sharpe {cand_m.sharpe:5.2f} | "
              f"hold ${hold_m.final_balance:>10,.0f} DD {hold_m.max_drawdown_pct:5.1f}% "
              f"Sharpe {hold_m.sharpe:5.2f} | "
              f"v4 ${v4_m.final_balance:>10,.0f} DD {v4_m.max_drawdown_pct:5.1f}% "
              f"Sharpe {v4_m.sharpe:5.2f}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ D4


def run_d4(panel, z_thresh: float, hold_by_asset_train):
    print("\n" + "=" * 100)
    print(f"D4 — 0.40% FEE TIER, candidate(z={z_thresh}) vs buy_and_hold, "
          f"PANEL_TRAIN, all 6 assets")
    print("=" * 100)
    start, end = shared.PANEL_TRAIN
    market = shared.SPOT_REAL
    d4 = 0
    rows = []
    for a in panel:
        cand = KellyRegimeV61ZscoreReversion(z_thresh=z_thresh)
        _, cand_m = measure(cand, a.df, start, end, market)
        _, hold_m = measure(get_strategy("buy_and_hold"), a.df, start, end, market)
        beats = cand_m.final_balance > hold_m.final_balance
        d4 += int(beats)
        rows.append({"asset": a.ticker, "cand_final": cand_m.final_balance,
                     "hold_final": hold_m.final_balance, "beats": beats})
        print(f"  {a.ticker:5s} cand ${cand_m.final_balance:>10,.0f} | "
              f"hold ${hold_m.final_balance:>10,.0f} | beats={beats}")
    print(f"  -> D4 = {d4}/6")
    return d4, pd.DataFrame(rows)


# ------------------------------------------------------------------------ D5


def run_d5(panel, z_thresh: float):
    print("\n" + "=" * 100)
    print(f"D5 — PANEL_TEST generalization, candidate(z={z_thresh}) vs buy_and_hold, "
          f"spot @0.10% (descriptive, not a gate)")
    print("=" * 100)
    start, end = shared.PANEL_TEST
    market = shared.SPOT_BASE
    d1 = 0
    d2 = 0
    rows = []
    for a in panel:
        cand = KellyRegimeV61ZscoreReversion(z_thresh=z_thresh)
        _, cand_m = measure(cand, a.df, start, end, market)
        _, hold_m = measure(get_strategy("buy_and_hold"), a.df, start, end, market)
        beats_final = cand_m.final_balance > hold_m.final_balance
        beats_dd = cand_m.max_drawdown_pct < hold_m.max_drawdown_pct
        d1 += int(beats_final)
        d2 += int(beats_dd)
        rows.append({
            "asset": a.ticker, "cand_final": cand_m.final_balance,
            "cand_dd": cand_m.max_drawdown_pct, "hold_final": hold_m.final_balance,
            "hold_dd": hold_m.max_drawdown_pct,
            "beats_final": beats_final, "beats_dd": beats_dd,
        })
        print(f"  {a.ticker:5s} cand ${cand_m.final_balance:>10,.0f} DD "
              f"{cand_m.max_drawdown_pct:5.1f}% | hold ${hold_m.final_balance:>10,.0f} "
              f"DD {hold_m.max_drawdown_pct:5.1f}% | beats_final={beats_final} "
              f"beats_dd={beats_dd}")
    print(f"  -> D1(test)={d1}/6  D2(test)={d2}/6")
    return d1, d2, pd.DataFrame(rows)


# ------------------------------------------------------------------------ main


def main() -> None:
    import experiments.r57_cross_asset_panel as panel_mod

    ok = causality_check()
    if not ok:
        raise SystemExit("CAUSALITY CHECK FAILED — refusing to run backtests")

    panel = shared.load_panel()
    print(f"\nPanel: {', '.join(a.ticker for a in panel)}")

    grid_df, hold_by_asset_train = run_d1_d2_sweep(panel)
    z_thresh, d1, d2 = select_z_thresh(grid_df)
    print(f"\nSELECTED z_thresh = {z_thresh}  (D1={d1}/6, D2={d2}/6)")

    d3_df = run_d3(z_thresh)
    d4, d4_df = run_d4(panel, z_thresh, hold_by_asset_train)
    d1_test, d2_test, d5_df = run_d5(panel, z_thresh)

    plateau_ok = True
    for z in shared.Z_THRESH_GRID:
        sub = grid_df[grid_df.z_thresh == z]
        if int(sub.beats_final.sum()) == 0:
            plateau_ok = False

    print("\n" + "=" * 100)
    print("PROMOTION BAR")
    print("=" * 100)
    promo = shared.promoted(d1, d2, d4, plateau_ok)
    print(f"  D1 >= 5/6: {d1}/6 -> {'PASS' if d1 >= 5 else 'FAIL'}")
    print(f"  D2 >= 4/6: {d2}/6 -> {'PASS' if d2 >= 4 else 'FAIL'}")
    print(f"  D4 >= 4/6: {d4}/6 -> {'PASS' if d4 >= 4 else 'FAIL'}")
    print(f"  plateau (no adjacent grid point collapses to 0/6): "
          f"{'PASS' if plateau_ok else 'FAIL'}")
    print(f"  -> OVERALL: {'PROMOTED' if promo else 'NOT PROMOTED'}")

    print(f"\nTotal backtest configurations evaluated (this branch): "
          f"{panel_mod.CONFIG_COUNT}")
    print("Holdout consultations added by this branch: 0 "
          "(no BTC/ETH bar dated 2023-01-01 or later is read anywhere)")


if __name__ == "__main__":
    main()
