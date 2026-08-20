#!/usr/bin/env python
"""R-61 novel branch: short-horizon z-score reversion, gated by a rolling causal Hurst exponent.

Not registered: lives under ``experiments/`` per ROUTINE.md step 5;
``src/tradebot/strategies/`` is untouched by this round. See
``experiments/r61_shared.py`` for the full pre-registration (mechanism,
literature, windows, decision rules D1-D5, promotion bar) — this file is
consistent with it and does not repeat it in full.

Mechanism, one sentence: build the conservative branch's identical
short-horizon (1/3/7-day) z-score mean-reversion vote, but only let it act
while this project's own ``rolling_causal_hurst`` (R-46,
``kelly_regime_v12_cppi_hurst.py``) measures the asset as CURRENTLY
anti-persistent (``H < 0.5``); stand flat through measured trending regimes
rather than buying every dip regardless of context. Sizing is
``kelly_regime``'s unmodified fractional-Kelly vol-targeting loop.

``z_thresh`` is frozen at 1.5 — the pre-registration's designated
middle-of-grid fallback for this branch (see r61_shared.py "MECHANISM":
the novel branch tests one additional mechanism on top of the
conservative branch's OWN PANEL_TRAIN-selected Z_THRESH, not a second
free hyperparameter; this branch cannot see the conservative branch's
live selection, so it uses the grid midpoint and flags the fallback
explicitly in its report for the operator to reconcile).

Data-read constraint (explicit for this session, stricter than
r61_shared's own ETH_FULL convention): no BTC or ETH bar dated
2023-01-01 or later is read anywhere in this file. ETH is therefore
truncated to <=2022-12-31 for D3 here, NOT run to "present" as
r61_shared's ETH_FULL literally describes — a deliberate, disclosed
deviation, flagged in the report for the operator to reconcile against
the conservative branch and the pre-registration text.

Usage::

    python experiments/r61_novel_hurst_gated_reversion.py causality
    python experiments/r61_novel_hurst_gated_reversion.py hurst_stats
    python experiments/r61_novel_hurst_gated_reversion.py d1d2
    python experiments/r61_novel_hurst_gated_reversion.py ablation
    python experiments/r61_novel_hurst_gated_reversion.py d3
    python experiments/r61_novel_hurst_gated_reversion.py d4
    python experiments/r61_novel_hurst_gated_reversion.py d5
    python experiments/r61_novel_hurst_gated_reversion.py all
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

from experiments import r61_shared as shared  # noqa: E402
from experiments.kelly_regime_v12_cppi_hurst import rolling_causal_hurst  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

Z_THRESH_FALLBACK = 1.5  # grid midpoint; see module docstring


# ------------------------------------------------------------------ strategy


class KellyRegimeV61HurstGatedReversion(Strategy):
    """Short-horizon z-score reversion, gated to fire only in anti-persistent (H<thresh) regimes.

    ``prepare()`` builds the z-score reversion vote exactly as the
    conservative branch does (three horizons, latched hysteresis,
    averaged), multiplies it by a 0/1 gate from this project's own
    ``rolling_causal_hurst``, then feeds the gated fraction through
    ``kelly_regime``'s unmodified fractional-Kelly vol-targeting sizing
    loop (copied verbatim below, not rederived).

    Pass ``hurst_thresh=1.0`` to defeat the gate (``H < 1.0`` is true for
    every realistic classical-R/S Hurst estimate), turning this into a
    plain, ungated version of the conservative branch's own mechanism —
    the ablation control used in this branch's own report.
    """

    name = "kelly_regime_v61_hurst_gated_reversion"

    def __init__(self, z_thresh: float = Z_THRESH_FALLBACK,
                 hurst_window_days: int = 60, hurst_thresh: float = 0.5,
                 target_vol: float = shared.TARGET_VOL,
                 max_leverage: float = shared.MAX_LEVERAGE,
                 vol_span: int = shared.VOL_SPAN_DAYS * BARS_PER_DAY,
                 deadband: float = shared.DEADBAND) -> None:
        self.z_thresh = z_thresh
        self.hurst_window_days = hurst_window_days
        self.hurst_thresh = hurst_thresh
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        # Covers both the longest z-score window (7 days) and the Hurst
        # window (60 days, +1 day for rolling_causal_hurst's own internal
        # shift(1)); at least 60 * BARS_PER_DAY + 10 per the pre-registration.
        self.warmup = int(max(max(shared.Z_HORIZONS_DAYS), hurst_window_days + 1)
                           * BARS_PER_DAY) + 10

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        # --- short-horizon z-score reversion vote (conservative branch's mechanism) ---
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

        # --- rolling causal Hurst gate (this branch's own addition) ---
        h = rolling_causal_hurst(close, self.hurst_window_days).reindex(df.index).to_numpy(dtype=float)
        gate = np.where(np.isfinite(h) & (h < self.hurst_thresh), 1.0, 0.0)
        frac_gated = frac * gate
        df["frac_raw"] = frac
        df["H"] = h
        df["gate"] = gate

        # --- kelly_regime's fractional-Kelly vol-targeting sizing loop, copied
        # verbatim (see src/tradebot/strategies/kelly_regime.py KellyRegime.prepare) ---
        r = np.log(close).diff()
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            v = vol[i]
            scale = min(self.target_vol / v, self.max_leverage) if np.isfinite(v) and v > 0 else 0.0
            desired = frac_gated[i] * scale
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


def make_gated(z_thresh: float = Z_THRESH_FALLBACK) -> KellyRegimeV61HurstGatedReversion:
    return KellyRegimeV61HurstGatedReversion(z_thresh=z_thresh, hurst_window_days=shared.HURST_WINDOW_DAYS,
                                              hurst_thresh=shared.HURST_THRESH)


def make_ungated(z_thresh: float = Z_THRESH_FALLBACK) -> KellyRegimeV61HurstGatedReversion:
    # hurst_thresh=1.0 defeats the gate (H < 1.0 true for every realistic
    # classical R/S Hurst estimate) -> gate === 1.0 -> frac_gated === frac.
    return KellyRegimeV61HurstGatedReversion(z_thresh=z_thresh, hurst_window_days=shared.HURST_WINDOW_DAYS,
                                              hurst_thresh=1.0)


# ------------------------------------------------------------------ harness

CONFIG_COUNT = 0  # every measure() call is counted, mirroring r57/r61_shared's own counter


def measure(strategy, df, start, end, market):
    global CONFIG_COUNT
    CONFIG_COUNT += 1
    return shared.measure(strategy, df, start, end, market)


def line(tag, m) -> None:
    print(f"  {tag:38s} final=${m.final_balance:>11,.0f} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} time_in_mkt={m.time_in_market_pct:>5.1f}% "
          f"trades={m.num_trades:>5d} fees=${m.fees_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# --------------------------------------------------------------------------- causality


def causality() -> None:
    """Two-opposite-tampers lookahead probe on the precomputed columns (frac_raw, H, gate, target).

    Since this strategy's ``target`` is fully precomputed in ``prepare()``
    (no path-dependent state carried in ``on_bar``, unlike the CPPI R-46
    branch), a column-level probe is sufficient: early rows of every
    ``prepare()``-built column must be bit-identical whether or not later
    rows in the frame are tampered with.
    """
    df, label = load_dataset(ROOT / "data", "spot")
    pre_2023 = df.loc[:"2022-12-31"]
    tail = pre_2023.iloc[-300_000:].copy()
    cut = len(tail) - 5_000

    up, down = tail.copy(), tail.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    pa = make_gated().prepare(up.copy())
    pb = make_gated().prepare(down.copy())

    ok = True
    for col in ("frac_raw", "H", "gate", "target"):
        a = pa[col].to_numpy(dtype=float)[:cut]
        b = pb[col].to_numpy(dtype=float)[:cut]
        finite = np.isfinite(a) & np.isfinite(b)
        worst = float(np.nanmax(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
        good = worst < 1e-9
        ok &= good
        print(f"  column={col:10s}  max |difference| before the cut = {worst:.3e}  "
              f"{'PASS' if good else 'FAIL'}")

    # Also a full sequential replay (target is a pure column lookup, but the
    # engine-level check is cheap and catches any accidental state leak in
    # on_bar()).
    strat_up = make_gated()
    strat_down = make_gated()
    a = run_backtest(strat_up, up.iloc[:cut + 1], shared.SPOT_BASE, 1_000.0, data_label=label)
    b = run_backtest(strat_down, down.iloc[:cut + 1], shared.SPOT_BASE, 1_000.0, data_label=label)
    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-6
    print(f"  {'equity replay':10s}  max |difference| before the cut = {worst_eq:.3e}  "
          f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")

    print(f"\ntampered from bar {cut:,} of {len(tail):,}; "
          f"{'PASS - no output at or before the cut moves' if ok else 'FAIL'}")


# --------------------------------------------------------------------------- Hurst stats (data property)


def hurst_stats() -> None:
    """Empirical rolling-causal-Hurst distribution: panel (6 assets, PANEL_TRAIN) vs BTC.

    A property of the DATA, not of any strategy (mirrors R-46's own
    ``hurst_stats`` command) -- does not call ``measure()`` / run any
    backtest, so it is not counted in CONFIG_COUNT.
    """
    print("=" * 100)
    print("EMPIRICAL ROLLING CAUSAL HURST DISTRIBUTION (data property, not a backtest)")
    print("=" * 100)

    def report(label, close, lo, hi):
        h = rolling_causal_hurst(close, shared.HURST_WINDOW_DAYS)
        h = h.loc[lo:hi].dropna()
        vals = h.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        pct_below = 100.0 * np.mean(vals < 0.5)
        print(f"  {label:12s} n={len(vals):>8,d}  mean={vals.mean():.3f}  median={np.median(vals):.3f}  "
              f"std={vals.std():.3f}  %H<0.5={pct_below:5.1f}%")
        return vals.mean(), pct_below

    panel = shared.load_panel()
    lo, hi = shared.PANEL_TRAIN
    print(f"\nPanel assets, PANEL_TRAIN {lo} -> {hi}:")
    panel_means = []
    for a in panel:
        m, p = report(a.ticker, a.df["close"], lo, hi)
        panel_means.append(m)
    print(f"  panel mean-of-means H = {np.mean(panel_means):.3f}")

    df, _ = load_dataset(ROOT / "data", "spot")
    btc_lo, btc_hi = shared.BTC_INNER_TRAIN[0], shared.BTC_INNER_VALID[1]
    print(f"\nBTC, BTC_INNER_TRAIN+BTC_INNER_VALID {btc_lo} -> {btc_hi} (comparison only):")
    btc_mean, btc_pct = report("BTC", df["close"], btc_lo, btc_hi)

    print(f"\nR-46 measured BTC's own rolling Hurst as persistently high (mean ~0.62). "
          f"This run's BTC mean = {btc_mean:.3f} -- "
          f"{'consistent with' if abs(btc_mean - 0.62) < 0.05 else 'DIVERGES from'} R-46's number "
          f"(expected: identical function, same estimator, should match closely).")
    print(f"Panel mean-of-means H = {np.mean(panel_means):.3f} vs BTC H = {btc_mean:.3f}: "
          f"panel is {'LOWER (supports the round hypothesis)' if np.mean(panel_means) < btc_mean else 'NOT lower than BTC'}.")


# --------------------------------------------------------------------------- D1/D2


def d1d2() -> None:
    print("=" * 100)
    print("D1/D2 — PANEL_TRAIN, spot @0.10%, gated candidate vs buy_and_hold")
    print("=" * 100)
    panel = shared.load_panel()
    rows = []
    for a in panel:
        cand = make_gated()
        res_c, m_c = measure(cand, a.df, *shared.PANEL_TRAIN, shared.SPOT_BASE)
        res_h, m_h = measure(BuyAndHold(), a.df, *shared.PANEL_TRAIN, shared.SPOT_BASE)
        beat_final = m_c.final_balance > m_h.final_balance
        beat_dd = m_c.max_drawdown_pct < m_h.max_drawdown_pct
        rows.append((a.ticker, m_c, m_h, beat_final, beat_dd))
        print(f"\n{a.ticker}:")
        line("candidate (hurst-gated)", m_c)
        line("buy_and_hold", m_h)
        print(f"    D1 beats hold final balance: {beat_final}   D2 lower max DD: {beat_dd}")

    d1 = sum(r[3] for r in rows)
    d2 = sum(r[4] for r in rows)
    print(f"\nD1 = {d1}/6  ({shared.d1_verdict(d1)})")
    print(f"D2 = {d2}/6")
    return rows


# --------------------------------------------------------------------------- ablation


def ablation() -> None:
    print("=" * 100)
    print("ABLATION — PANEL_TRAIN, spot @0.10%, gated vs ungated (hurst_thresh=1.0, gate defeated)")
    print("=" * 100)
    panel = shared.load_panel()
    rows = []
    for a in panel:
        res_g, m_g = measure(make_gated(), a.df, *shared.PANEL_TRAIN, shared.SPOT_BASE)
        res_u, m_u = measure(make_ungated(), a.df, *shared.PANEL_TRAIN, shared.SPOT_BASE)
        rows.append((a.ticker, m_g, m_u))
        print(f"\n{a.ticker}:")
        line("gated (H<0.5 only)", m_g)
        line("ungated (plain reversion)", m_u)
        print(f"    d(final)=${m_g.final_balance - m_u.final_balance:>+11,.0f}   "
              f"d(maxDD)={m_g.max_drawdown_pct - m_u.max_drawdown_pct:>+6.1f}pp   "
              f"d(sharpe)={m_g.sharpe - m_u.sharpe:>+5.2f}")
    return rows


# --------------------------------------------------------------------------- D3


def d3() -> None:
    print("=" * 100)
    print("D3 — BTC_INNER_TRAIN, BTC_INNER_VALID, ETH (<=2022-12-31 only, see module docstring), "
          "spot @0.10%")
    print("gated candidate vs buy_and_hold vs kelly_regime_v4")
    print("=" * 100)
    btc, label = load_dataset(ROOT / "data", "spot")
    eth_full = load_coinbase_eth_spot(ROOT / "data")
    eth = eth_full.loc[:"2022-12-31"]  # explicit truncation, see module docstring

    windows = [
        ("BTC_INNER_TRAIN", btc, shared.BTC_INNER_TRAIN),
        ("BTC_INNER_VALID", btc, shared.BTC_INNER_VALID),
        ("ETH (<=2022-12-31)", eth, (None, "2022-12-31")),
    ]
    rows = []
    for wname, df, (start, end) in windows:
        print(f"\n--- {wname}  ({start} -> {end}) ---")
        res_c, m_c = measure(make_gated(), df, start, end, shared.SPOT_BASE)
        res_h, m_h = measure(BuyAndHold(), df, start, end, shared.SPOT_BASE)
        res_v4, m_v4 = measure(KellyRegimeV4(), df, start, end, shared.SPOT_BASE)
        line("candidate (hurst-gated)", m_c)
        line("buy_and_hold", m_h)
        line("kelly_regime_v4", m_v4)
        rows.append((wname, m_c, m_h, m_v4))
    return rows


# --------------------------------------------------------------------------- D4


def d4() -> None:
    print("=" * 100)
    print("D4 — PANEL_TRAIN, spot @0.40% (real Bitstamp entry tier), gated candidate vs buy_and_hold")
    print("=" * 100)
    panel = shared.load_panel()
    beat = 0
    for a in panel:
        res_c, m_c = measure(make_gated(), a.df, *shared.PANEL_TRAIN, shared.SPOT_REAL)
        res_h, m_h = measure(BuyAndHold(), a.df, *shared.PANEL_TRAIN, shared.SPOT_REAL)
        b = m_c.final_balance > m_h.final_balance
        beat += b
        print(f"\n{a.ticker}:")
        line("candidate (hurst-gated) @0.40%", m_c)
        line("buy_and_hold @0.40%", m_h)
        print(f"    beats hold: {b}")
    print(f"\nD4 = {beat}/6")
    return beat


# --------------------------------------------------------------------------- D5


def d5() -> None:
    print("=" * 100)
    print("D5 — PANEL_TEST, spot @0.10%, gated candidate vs buy_and_hold (descriptive, not a gate)")
    print("=" * 100)
    panel = shared.load_panel()
    rows = []
    for a in panel:
        res_c, m_c = measure(make_gated(), a.df, *shared.PANEL_TEST, shared.SPOT_BASE)
        res_h, m_h = measure(BuyAndHold(), a.df, *shared.PANEL_TEST, shared.SPOT_BASE)
        beat_final = m_c.final_balance > m_h.final_balance
        beat_dd = m_c.max_drawdown_pct < m_h.max_drawdown_pct
        rows.append((a.ticker, m_c, m_h, beat_final, beat_dd))
        print(f"\n{a.ticker}:")
        line("candidate (hurst-gated)", m_c)
        line("buy_and_hold", m_h)
        print(f"    beats hold final balance: {beat_final}   lower max DD: {beat_dd}")
    d1t = sum(r[3] for r in rows)
    d2t = sum(r[4] for r in rows)
    print(f"\nPANEL_TEST: {d1t}/6 beat hold, {d2t}/6 lower max DD (descriptive)")
    return rows


# --------------------------------------------------------------------------- main


if __name__ == "__main__":
    cmds = {"causality": causality, "hurst_stats": hurst_stats, "d1d2": d1d2,
            "ablation": ablation, "d3": d3, "d4": d4, "d5": d5}

    def all_() -> None:
        causality()
        print()
        hurst_stats()
        print()
        d1d2()
        print()
        ablation()
        print()
        d3()
        print()
        d4()
        print()
        d5()
        print(f"\nTotal measure() (backtest) calls this run: {CONFIG_COUNT}")

    cmds["all"] = all_
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python {sys.argv[0]} [{'|'.join(cmds)}]")
