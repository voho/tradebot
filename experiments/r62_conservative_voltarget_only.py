"""R-62 (conservative branch): does `kelly_regime_v4`'s conditional
volatility-target `scale[i]` factor, ALONE (the directional vote `frac[i]`
forced to 1.0 always, never gated), reproduce the matched-exposure drawdown
advantage R-57 found 6/6 on BTC/ETH and 0/6 on the six-asset panel?

Pre-registration (shared, frozen, both branches): `experiments/r62_shared.py`.
That file is not edited here. B-27 (docs/LEDGER.md) is the literal question
this branch answers: "does the SAME matched-exposure advantage appear for a
strategy that holds a constant vol-targeted exposure with no directional
vote at all ... isolating the SIZE machinery's own turnover/rebalancing
behavior from any signal, trend or reversion?"

`VolTargetOnly.prepare()` below is `KellyRegimeV3.prepare()` copied
byte-for-byte with exactly one change: the vote block is deleted and
`frac[i]` is a constant array of 1.0. Same defaults as `kelly_regime_v4`
uses (`target_vol=0.55, max_leverage=2.0, vol_span=8*BARS_PER_DAY,
anchor_span_days=180, high_in=1.70, high_out=1.20, low_in=0.55,
low_out=0.85, deadband=0.10`). Zero new parameters; nothing tuned, a
component removed. Not registered under `src/tradebot/strategies/` per
ROUTINE.md step 5 (an experimental, unpromoted variant lives under
``experiments/``).

Usage::

    uv run python experiments/r62_conservative_voltarget_only.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiments.r57_cross_asset_panel as r57  # noqa: E402
import experiments.r62_shared as r62  # noqa: E402
from experiments.r57_cross_asset_panel import (  # noqa: E402
    BEAR22,
    FULL,
    FUT_BASE,  # noqa: F401  (re-exported per task spec, unused directly here)
    SPOT_BASE,
    SPOT_REAL,
)
from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

OUT_DIR = ROOT / "reports" / "cross_asset_panel"


class VolTargetOnly(Strategy):
    """`kelly_regime_v3`'s conditional/extreme-only volatility-target `scale`
    alone, `frac` forced to 1.0 for every bar -- never stand aside, sized
    purely by the volatility target. B-27's literal request: isolate the
    SIZE machinery's own turnover/rebalancing behaviour from any directional
    signal at all, trend or reversion.

    Mechanism (copied byte-for-byte from `KellyRegimeV3.prepare`, minus the
    vote): hold a CONSTANT notional (`target_vol / EWM-slow-vol`) through
    "normal" realized volatility; switch to full inverse-vol sizing
    (`target_vol / EWM-fast-vol`, capped at `max_leverage`) only once
    volatility breaks out past `high_in`/`low_in`, latching until it
    retraces past `high_out`/`low_out` (the same hysteresis idea v3/v4 apply
    to the vote, applied here to the risk axis alone). Zero new parameters;
    every constant below is v4's own shipped default.
    """

    name = "r62_conservative_voltarget_only"
    warmup = 80 * BARS_PER_DAY + 10  # matches v4's warmup (180d slow anchor dominates in practice)

    def __init__(self, target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, anchor_span_days: int = 180,
                 high_in: float = 1.70, high_out: float = 1.20,
                 low_in: float = 0.55, low_out: float = 0.85,
                 deadband: float = 0.10) -> None:
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        self.deadband = deadband

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # frac[i] forced to 1.0 for every bar -- the vote is deleted entirely.
        frac = np.ones(len(df))

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
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
            ctx.order_notional(t)


# ------------------------------------------------------------- causality probe


def cmd_causality(panel) -> bool:
    """Same tamper-probe methodology as r57's `cmd_causality`, run against
    `VolTargetOnly` on at least 2 panel assets. Opposite up/down tampers of
    the tail of the series must produce identical queued orders at/before
    the cut -- proof the strategy never reads future bars."""
    print("=" * 100)
    print("CAUSALITY TAMPER PROBE -- VolTargetOnly (conservative) on panel assets")
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
            s = VolTargetOnly()
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


# ----------------------------------------------------------------- the run


def main() -> None:
    print("=" * 100)
    print("R-62 CONSERVATIVE: VolTargetOnly (v4's scale[i], frac[i]=1.0 always)")
    print("=" * 100)

    panel = r62.load_panel()
    print(f"Panel ({len(panel)}): {', '.join(a.ticker for a in panel)}\n")

    causality_ok = cmd_causality(panel)
    print()

    strat = VolTargetOnly()

    # NOTE ON A SHARED-HARNESS BUG (r62_shared.py, not edited -- see the
    # docstring above and the final report): `d1_from_rows`/`d4_from_rows`
    # filter rows only by (arm, market, fee), NOT by window. FULL, BEAR22
    # and CONTROL_WINDOW all share market="spot", fee=0.001, so calling
    # either helper on one `rows` list accumulated across all of this
    # round's cells (exactly what the task's own step-2/step-3 sequence
    # builds) silently pools rows from three different windows -- an
    # earlier run of this script did exactly that and printed the
    # impossible "D1 7/6". Fixed here on the caller's side, without
    # touching the frozen shared file, by keeping the FULL-window panel
    # cells in their own list (`full_rows`) and passing THAT to
    # `d1_from_rows`/`d4_from_rows`, never the grand `rows` list. D3
    # already used this pattern correctly (`ctrl_rows`, kept below).
    full_rows: list[dict] = []
    bear_rows: list[dict] = []
    ctrl_rows: list[dict] = []

    print("=" * 100)
    print("FULL window, spot @0.10% (D1 primary, D2 context)")
    print("=" * 100)
    for a in panel:
        r62.cell(strat, "conservative", a, FULL, SPOT_BASE, full_rows)

    print("\n" + "=" * 100)
    print("FULL window, spot @0.40% (D4 context -- Bitstamp entry-tier fee)")
    print("=" * 100)
    for a in panel:
        r62.cell(strat, "conservative", a, FULL, SPOT_REAL, full_rows)

    print("\n" + "=" * 100)
    print("BEAR22 2022-05-01..2022-11-30, spot @0.10% (descriptive, R-57 D4 style)")
    print("=" * 100)
    for a in panel:
        r62.cell(strat, "conservative", a, BEAR22, SPOT_BASE, bear_rows)

    print("\n" + "=" * 100)
    print("D3 control: BTC/ETH, CONTROL_WINDOW 2020-04-01..2022-12-31, spot @0.10%")
    print("=" * 100)
    r62.run_control(strat, "conservative", ctrl_rows)

    rows: list[dict] = full_rows + bear_rows + ctrl_rows
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "r62_conservative_cells.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved rows -> {out_path}")

    # ------------------------------------------------------- verdicts
    print("\n" + "=" * 100)
    print("PRE-REGISTERED DECISION RULES (r62_shared.py)")
    print("=" * 100)

    d1_k, d1_df = r62.d1_from_rows(full_rows, "conservative", "spot", 0.001)
    d1_p = r57.binomial_tail(d1_k, 6)
    d1_v = r62.d1_verdict(d1_k, 6)
    print(f"D1 (primary, matched-exposure DD, FULL, spot@0.10%): {d1_k}/6, "
          f"exact binomial p={d1_p:.4f} -> {d1_v}")

    d3_k, d3_df = r62.d1_from_rows(ctrl_rows, "conservative", "spot", 0.001, n=2)
    d3_p = r57.binomial_tail(d3_k, 2)
    d3_v = r62.d1_verdict(d3_k, 2)
    print(f"D3 (BTC/ETH control, matched-exposure DD, CONTROL_WINDOW, "
          f"spot@0.10%): {d3_k}/2, exact binomial p={d3_p:.4f} -> {d3_v}")

    d4_k = r62.d4_from_rows(full_rows, "conservative", "spot", 0.004)
    print(f"D4 (0.40% fee, FULL, beats buy_and_hold final balance): {d4_k}/6")

    fw = r62.further_work(d1_k, d3_k, d4_k)
    print(f"Further-work bar (D1>=5/6 AND D3>=1/2 AND D4>=4/6): "
          f"{'MET' if fw else 'NOT MET'}")

    print(f"\nCausality tamper probe: {'PASS' if causality_ok else 'FAIL'}")

    total_configs = r57.CONFIG_COUNT + r62.extra_config_count()
    print(f"Total backtest configurations evaluated "
          f"(r57.CONFIG_COUNT + r62.extra_config_count()): {total_configs}")
    print("Holdout consultations added by this round: 0 "
          "(CONTROL_WINDOW ends 2022-12-31; panel reads never touch a holdout)")


if __name__ == "__main__":
    main()
