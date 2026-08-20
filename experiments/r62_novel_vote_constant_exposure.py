#!/usr/bin/env python
"""R-62 novel branch: kelly_regime_v4's latched multi-anchor vote, alone, at
a constant full-notional multiplier -- no volatility scaling of any kind.

Not registered: lives under ``experiments/`` per ROUTINE.md step 5;
``src/tradebot/strategies/`` is untouched by this round. See
``experiments/r62_shared.py`` for the full pre-registration (mechanism,
literature, windows, decision rules D1-D4, further-work bar) -- this file
is consistent with it and does not repeat it in full, and does not edit it.

Mechanism, one sentence: ``desired[i] = frac[i] * c_const`` where ``frac``
is `kelly_regime`'s exact latched 20/40/80-day multi-anchor vote (copied
byte-for-byte from ``KellyRegime.prepare`` / v4's inherited horizons -- same
defaults, ``horizons=(20,40,80), band=0.01``) and ``c_const = 1.0`` is a
FIXED constant, not swept -- full notional whenever the trend vote is "in",
the simplest possible binary trend rule, and not selected against any data
(there is no second free parameter to protect here). The same 10% deadband
and latching mechanics as v4 gate updates to the held position, and orders
are placed the same way v4 does (``ctx.order_notional(t)`` when the
deadband-filtered target changes).

This is the complement of the conservative branch (``frac`` forced to 1.0,
v4's own conditional vol-target ``scale`` kept): here the vote survives
and the risk-scaling machinery is deleted instead of the signal.

Usage::

    python experiments/r62_novel_vote_constant_exposure.py causality
    python experiments/r62_novel_vote_constant_exposure.py run
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

import experiments.r57_cross_asset_panel as r57  # noqa: E402
import experiments.r62_shared as shared  # noqa: E402
from experiments.r57_cross_asset_panel import (  # noqa: E402
    BEAR22,
    FULL,
    FUT_BASE,
    SPOT_BASE,
    SPOT_REAL,
)

OUT_DIR = ROOT / "reports" / "cross_asset_panel"


# ------------------------------------------------------------------ strategy


class VoteConstantExposure(Strategy):
    """kelly_regime_v4's latched multi-anchor vote alone, at a constant full-notional multiplier.

    ``prepare()`` is `KellyRegime`'s exact vote-computation loop (three
    anchors, 1% band, latched hysteresis: latch bullish above the anchor,
    bearish below, hold the previous verdict inside the band), copied
    byte-for-byte, with v4's conditional volatility-target ``scale``
    deleted entirely and replaced by a fixed constant ``c_const``. No
    volatility measurement, EWM vol span, or breakout-state machinery of
    any kind appears anywhere in this class.
    """

    name = "r62_novel_vote_constant_exposure"
    # Same warmup as kelly_regime_v4: the slowest anchor is 80 days.
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 c_const: float = 1.0, deadband: float = 0.10) -> None:
        self.horizons = horizons
        self.band = band
        self.c_const = c_const
        self.deadband = deadband

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        # Crowd-regime vote: latch bullish above the anchor, bearish below,
        # hold the previous verdict inside the band (hysteresis, not chop).
        # Byte-for-byte the same loop as KellyRegime.prepare / KellyRegimeV3
        # / KellyRegimeV4's inherited version, horizons=(20,40,80), band=0.01.
        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()

        # No volatility target, no EWM vol, no breakout state: constant
        # exposure whenever the vote is "in". Same deadband/latch mechanics
        # as v4 -- only the scale factor changed, to a bare constant.
        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = frac[i] * self.c_const
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


def cmd_causality(panel) -> bool:
    """Tamper probe, `experiments/r57_cross_asset_panel.py`'s `cmd_causality`
    methodology, against `VoteConstantExposure` on 2 panel assets."""
    print("=" * 100)
    print("CAUSALITY TAMPER PROBE -- VoteConstantExposure on 2 panel assets")
    print("=" * 100)
    market = MarketSpec.futures(leverage=5.0)
    all_ok = True
    for a in panel[:2]:
        tail = a.df.iloc[-60_000:].copy()
        cut = len(tail) - 5_000
        bars = [cut - k for k in (1, 2, 3, 5, 10, 20)]
        up, down = tail.copy(), tail.copy()
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

        def decisions(frame):
            s = VoteConstantExposure()
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=market, start_balance=10_000.0)
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        ok = all(x == y for x, y in zip(decisions(up), decisions(down)))
        all_ok = all_ok and ok
        print(f"  {a.ticker:5s} decisions identical under opposite post-cut "
              f"tampers: {'PASS' if ok else 'FAIL'}")
    return all_ok


# ------------------------------------------------------------------------ run


def cmd_run(panel) -> None:
    rows: list[dict] = []

    print("=" * 100)
    print("FULL WINDOW 2020-04-01 -> end -- spot @0.10% (D1 primary)")
    print("=" * 100)
    for a in panel:
        shared.cell(VoteConstantExposure(), "novel", a, FULL, SPOT_BASE, rows)

    print("\n" + "=" * 100)
    print("FULL WINDOW -- spot @0.40% Bitstamp entry tier (D4 fee context)")
    print("=" * 100)
    for a in panel:
        shared.cell(VoteConstantExposure(), "novel", a, FULL, SPOT_REAL, rows)

    print("\n" + "=" * 100)
    print("BEAR22 2022-05-01..2022-11-30 -- spot @0.10% (descriptive, R-57's D4 style)")
    print("=" * 100)
    for a in panel:
        shared.cell(VoteConstantExposure(), "novel", a, BEAR22, SPOT_BASE, rows)

    print("\n" + "=" * 100)
    print("D3 CONTROL -- BTC/ETH, CONTROL_WINDOW (2020-04-01..2022-12-31), spot @0.10%, +0 holdout")
    print("=" * 100)
    shared.run_control(VoteConstantExposure(), "novel", rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "r62_novel_cells.csv", index=False)

    print("\n" + "=" * 100)
    print("PRE-REGISTERED DECISION RULES")
    print("=" * 100)

    # IMPORTANT: shared.d1_from_rows/d4_from_rows filter only on
    # (arm, market, fee), not window. BEAR22 and the D3 control window both
    # also run at market="spot", fee=0.001 -- the same (market, fee) pair as
    # the D1 FULL-window slice -- so calling them against the full,
    # multi-window `rows` list silently pools cells from different windows
    # into one count. Discovered while building this runner; not a defect
    # introduced here but a latent trap in the shared helper's signature
    # (no `window` parameter to disambiguate). Fixed at the call site by
    # pre-filtering to each decision rule's own window before calling the
    # frozen shared functions -- the functions themselves are used exactly
    # as pre-registered, not reimplemented; only the input slice changes.
    full_label = f"{FULL[0]}:{FULL[1]}"
    ctrl_label = f"{shared.CONTROL_WINDOW[0]}:{shared.CONTROL_WINDOW[1]}"
    full_rows = [r for r in rows if r["window"] == full_label]
    ctrl_rows = [r for r in rows if r["window"] == ctrl_label]

    d1_k, d1_df = shared.d1_from_rows(full_rows, "novel", "spot", 0.001)
    print(f"D1 (primary, matched-exposure drawdown, spot @0.10%, FULL): "
          f"{d1_k}/6 assets -> {shared.d1_verdict(d1_k)}")

    d3_k, d3_df = shared.d1_from_rows(ctrl_rows, "novel", "spot", 0.001, n=2)
    assert len(d3_df) == 2, f"expected 2 D3 control rows (BTC, ETH), got {len(d3_df)}"
    print(f"D3 (BTC/ETH control, identical D1 methodology via shared.d1_from_rows, "
          f"{shared.CONTROL_WINDOW}): {d3_k}/2 -> {shared.d1_verdict(d3_k, n=2)}")

    d4_k = shared.d4_from_rows(full_rows, "novel", "spot", 0.004)
    print(f"D4 (0.40% fee context, beats buy_and_hold final balance): {d4_k}/6")

    fw = shared.further_work(d1_k, d3_k, d4_k)
    print(f"\nFurther-work bar (D1>=5/6 AND D3>=1/2 AND D4>=4/6): "
          f"{'MET' if fw else 'NOT MET'}")

    print(f"\nrows saved: {OUT_DIR / 'r62_novel_cells.csv'}")
    return rows, d1_k, d3_k, d4_k, fw


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    panel = shared.load_panel()
    print(f"Panel ({len(panel)}): {', '.join(a.ticker for a in panel)}\n")

    if cmd == "causality":
        cmd_causality(panel)
        return

    if cmd == "run":
        causality_ok = cmd_causality(panel)
        print()
        cmd_run(panel)
        total_configs = r57.CONFIG_COUNT + shared.extra_config_count()
        print(f"\nCausality probe: {'PASS' if causality_ok else 'FAIL'}")
        print(f"Total backtest configurations evaluated (this branch): {total_configs}")
        print("Holdout consultations added by this branch: 0 "
              "(panel reads never touch a reserved holdout; CONTROL_WINDOW ends 2022-12-31)")
        return

    raise SystemExit(f"unknown command {cmd!r} (causality | run)")


if __name__ == "__main__":
    main()
