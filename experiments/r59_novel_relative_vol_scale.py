"""R-59 NOVEL BRANCH (backlog B-25): self-normalizing, scale-invariant
exposure sizing for the `kelly_regime` family.

Not registered: lives under ``experiments/`` per ROUTINE.md step 5. This is
an experiment, not a strategy the runner should discover, so
``KellyRegimeRelativeVol`` below is a plain ``Strategy`` subclass with no
``@register`` decorator.

Pre-registration: ``experiments/r59_shared.py`` (read it first — this file
imports its windows, costs, panel loader and decision-rule helpers rather
than restating them). This module implements only the NOVEL branch's
candidate mechanism and the frozen D1-D4 measurement matrix pointed at it.

=====================================================================
THE MECHANISM
=====================================================================

``KellyRegimeV3.prepare()`` (inherited unchanged by ``KellyRegimeV4``,
which only changes the default anchor ladder to 20/40/80 days) computes two
exposure-scale terms from ABSOLUTE realized volatility:

    full   = min(target_vol / vol,  max_leverage)   # during a breakout
    steady = min(target_vol / slow, max_leverage)   # inside the normal band

``target_vol=0.55`` and ``max_leverage=2.0`` were tuned on BTC. R-57 found
that on six higher-volatility Coinbase instruments the strategy's one
surviving property (a matched-exposure drawdown advantage) inverts on 6 of
6, and named the likely cause: those two constants are BTC's ABSOLUTE
volatility scale, so on instruments whose realized volatility structurally
sits above BTC's, ``target_vol / vol`` is structurally smaller and
persistently binding — mean notional 0.18-0.26 on the panel vs. 0.38 BTC /
0.34 ETH over the same window (R-57's own measurement).

This branch's fix, and why it needs ZERO new fitted parameters (as distinct
from the conservative branch's per-asset calibration): normalize each
instrument's current volatility against its OWN long-run trailing average
before applying the single global target, rather than fitting a new
constant per instrument. This is the mechanism Barroso & Santa-Clara (2015,
J. Financial Economics 116(1), 111-120) use — scale a strategy by the
inverse of its OWN trailing realized volatility to hit a constant target,
because risk is predictable from an asset's own recent history and the
normalization is RELATIVE to the asset's own distribution, not an absolute
level — and the same point Moskowitz, Ooi & Pedersen (2012, J. Financial
Economics, "Time Series Momentum") and Baltas & Kosowski (2013/2017, J.
Investment Management) make for cross-instrument trend portfolios:
instruments with structurally different volatility levels need their bet
sized relative to their OWN volatility scale, not a shared absolute
constant. (Full citations and the round's pre-registration:
``experiments/r59_shared.py``.)

Concretely, one new causal series is added:

    long_run_vol = vol.ewm(span=LONG_RUN_SPAN_DAYS * BARS_PER_DAY,
                            min_periods=BARS_PER_DAY).mean()

``LONG_RUN_SPAN_DAYS = 720`` (~2 years) is a STRUCTURAL choice, fixed here
before any result on this branch was read, and is NOT swept or fit. It is
set deliberately longer than v3's existing 180-day ``anchor_span_days`` (the
span behind ``slow``, which already exists to drive the high/low breakout
hysteresis) so that ``long_run_vol`` captures a genuinely long-run reference
level distinct from the anchor already in the mechanism, rather than
duplicating it. Two years is also long enough to span both a full bull and
bear phase on every one of this round's eight assets' training windows, so
"long-run" means "long relative to a regime", not "long relative to a few
weeks".

The scale terms become dimensionless and self-calibrating:

    vol_rel   = vol / long_run_vol            # dimensionless, mean ~1 by
                                               # construction over any
                                               # instrument's own history
    full      = min(target_vol / vol_rel,               max_leverage)
    steady    = min(target_vol / (slow / long_run_vol),  max_leverage)

``target_vol=0.55`` and ``max_leverage=2.0`` are UNCHANGED GLOBAL CONSTANTS
— identical for BTC, ETH and all six panel assets. No per-asset number is
fit anywhere in this branch; that is the entire point of it, as distinct
from the conservative branch's per-asset calibration.

The hysteresis state machine (``ratio = vol / slow`` driving the +1/-1
breakout latch) is UNCHANGED — it is already a relative measure and is not
the diagnosed problem. Only the two scale terms change. Everything else in
``prepare()`` — the vote, the anchors, the deadband, the horizons default
of (20, 40, 80) inherited from ``KellyRegimeV4`` — is byte-identical to
``KellyRegimeV3.prepare()``.

A diagnostic-only side channel (``self._last_vol_rel`` / ``self._last_index``)
is stashed at the end of ``prepare()`` so the self-consistency check below
can read the exact series the strategy used, without a second, potentially
divergent, reimplementation. It is write-only bookkeeping: it is never read
inside ``prepare()`` or ``on_bar()``, so it cannot affect a single trading
decision or the causality probe.

Usage::

    uv run python experiments/r59_novel_relative_vol_scale.py selfcheck
    uv run python experiments/r59_novel_relative_vol_scale.py causality
    uv run python experiments/r59_novel_relative_vol_scale.py run
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.matched_hold import ConstantExposureHold, mean_notional  # noqa: E402
from experiments.r57_cross_asset_panel import binomial_tail  # noqa: E402
from experiments.r59_shared import (  # noqa: E402
    CONTROL,
    D2_REGRESSION_TOLERANCE_PP,
    PANEL_TEST,
    PANEL_TRAIN,
    R57_CONTROL_DD_ADVANTAGE,
    SPOT_BASE,
    SPOT_REAL,
    Asset,
    d1_verdict,
    d2_passes,
    load_panel,
    promoted,
)
from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.data import load_coinbase_spot, load_dataset  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns,
    max_drawdown_from_returns,
    paired_bootstrap,
    total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

OUT_DIR = ROOT / "reports" / "r59_novel"
BOOT_KW = dict(mean_block=30.0, n_boot=2_000, seed=7)

LONG_RUN_SPAN_DAYS = 720  # structural, ~2 years, fixed before any result read

CONFIG_COUNT = 0


# ================================================================== strategy


class KellyRegimeRelativeVol(KellyRegimeV3):
    """v3/v4's conditional-targeting mechanism with a self-normalizing scale.

    See module docstring for the full derivation. NOT registered — this is
    an experiment, constructed directly, never through
    ``tradebot.registry.get_strategy``.
    """

    name = "kelly_regime_v58_relvol"  # attribute only; no @register applied
    warmup = 80 * BARS_PER_DAY + 10  # identical to KellyRegimeV4 (horizons-driven)

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self._last_vol_rel: pd.Series | None = None  # diagnostic only

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

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
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())
        # --- the one change: a long-run, causal, per-instrument reference
        # level that turns the absolute scale terms below into dimensionless,
        # self-calibrating ones. Same EWM construction as `slow` above, just
        # a longer, structurally-fixed span (see module docstring).
        long_run_vol = (pd.Series(vol).ewm(
            span=LONG_RUN_SPAN_DAYS * BARS_PER_DAY,
            min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            vol_rel = np.where(long_run_vol > 0, vol / long_run_vol, np.nan)
            slow_rel = np.where(long_run_vol > 0, slow / long_run_vol, np.nan)
            full = np.minimum(self.target_vol / vol_rel, self.max_leverage)
            steady = np.minimum(self.target_vol / slow_rel, self.max_leverage)
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
        # Diagnostic-only: never read by prepare()/on_bar(), cannot affect
        # a trading decision or the causality probe. Used solely by
        # cmd_selfcheck() below to report vol_rel's own time-average.
        self._last_vol_rel = pd.Series(vol_rel, index=df.index)
        return df


def make_candidate() -> KellyRegimeRelativeVol:
    """Fresh instance per backtest, same convention as get_strategy()."""
    return KellyRegimeRelativeVol()


# =================================================================== helpers


def measure(strategy, df, start, end, market):
    global CONFIG_COUNT
    CONFIG_COUNT += 1
    result = run_period(strategy, df, start, end, market=market, start_balance=1_000.0)
    return result, compute_metrics(result)


def cell(ticker: str, df: pd.DataFrame, window, market, label: str, rows: list) -> dict:
    """One asset x window x market cell: candidate, hold, matched hold, intervals."""
    start, end = window
    cand_res, cand = measure(make_candidate(), df, start, end, market)
    hold_res, hold = measure(get_strategy("buy_and_hold"), df, start, end, market)

    c_mean = mean_notional(cand_res)
    mh_res, mh = measure(ConstantExposureHold(c_mean), df, start, end, market)

    cand_ret = daily_returns(cand_res.equity).to_numpy(dtype=float)
    mh_ret = daily_returns(mh_res.equity).to_numpy(dtype=float)
    hold_ret = daily_returns(hold_res.equity).to_numpy(dtype=float)
    n = min(len(cand_ret), len(mh_ret), len(hold_ret))
    dd_matched = paired_bootstrap(cand_ret[:n], mh_ret[:n],
                                  max_drawdown_from_returns, **BOOT_KW)
    growth_matched = paired_bootstrap(cand_ret[:n], mh_ret[:n],
                                      total_log_return, **BOOT_KW)

    row = {
        "asset": ticker, "window": label, "market": market.name,
        "fee": market.fee_rate,
        "cand_final": cand.final_balance, "cand_dd": cand.max_drawdown_pct,
        "cand_sharpe": cand.sharpe, "cand_trades": cand.num_trades,
        "cand_liq": cand.liquidated,
        "hold_final": hold.final_balance, "hold_dd": hold.max_drawdown_pct,
        "hold_sharpe": hold.sharpe, "hold_liq": hold.liquidated,
        "c_mean_notional": c_mean,
        "mh_final": mh.final_balance, "mh_dd": mh.max_drawdown_pct,
        "mh_sharpe": mh.sharpe,
        "dd_matched_diff": dd_matched.diff.point,
        "dd_matched_lo": dd_matched.diff.lo, "dd_matched_hi": dd_matched.diff.hi,
        "growth_matched_diff": growth_matched.diff.point,
        "growth_matched_lo": growth_matched.diff.lo,
        "growth_matched_hi": growth_matched.diff.hi,
    }
    rows.append(row)
    print(f"  {ticker:5s} {label:9s} {market.name:11s} fee={market.fee_rate:.2%}  "
          f"cand ${cand.final_balance:>10,.0f} DD {cand.max_drawdown_pct:5.1f}% | "
          f"hold ${hold.final_balance:>10,.0f} DD {hold.max_drawdown_pct:5.1f}% | "
          f"matched(c={c_mean:.2f}) ${mh.final_balance:>10,.0f} "
          f"DD {mh.max_drawdown_pct:5.1f}% | "
          f"dDD_matched {dd_matched.diff.point:+6.1f}pp "
          f"[{dd_matched.diff.lo:+6.1f},{dd_matched.diff.hi:+6.1f}]")
    return row


# ============================================================== self-check


def cmd_selfcheck(panel: list[Asset]) -> dict[str, float]:
    """vol_rel's own time-average on all 8 assets over their frozen window.

    Mechanism's own self-consistency check: vol_rel = vol/long_run_vol
    should average close to 1.0 over any instrument's own history by
    construction. Reads self._last_vol_rel, stashed as a side effect of
    the SAME prepare() call the backtest itself makes — no second,
    potentially-divergent computation.
    """
    print("=" * 100)
    print("SELF-CONSISTENCY CHECK — mean(vol_rel) over PANEL_TRAIN/CONTROL "
          f"(2020-04-01..2022-12-31), LONG_RUN_SPAN_DAYS={LONG_RUN_SPAN_DAYS}")
    print("=" * 100)
    start, end = PANEL_TRAIN  # identical window to CONTROL
    means: dict[str, float] = {}

    btc_df, _ = load_dataset(ROOT / "data", "spot")
    btc_df = btc_df.loc[:"2022-12-31"]
    eth_df = load_coinbase_spot(ROOT / "data", "ETH")
    eth_df = eth_df.loc[:"2022-12-31"]
    frames = [("BTC", btc_df), ("ETH", eth_df)] + [(a.ticker, a.df) for a in panel]

    for ticker, df in frames:
        cand = make_candidate()
        cand.prepare(df.copy())
        window_mask = (cand._last_vol_rel.index >= pd.Timestamp(start, tz="UTC")) & \
                      (cand._last_vol_rel.index <= pd.Timestamp(end, tz="UTC"))
        vr = cand._last_vol_rel[window_mask].to_numpy(dtype=float)
        vr = vr[np.isfinite(vr) & (vr > 0)]
        mean_vr = float(np.mean(vr)) if len(vr) else float("nan")
        means[ticker] = mean_vr
        flag = "" if 0.7 <= mean_vr <= 1.4 else "  <-- FAR FROM 1.0"
        print(f"  {ticker:5s} mean(vol_rel)={mean_vr:6.3f}  n={len(vr):>7,d}{flag}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"asset": k, "mean_vol_rel": v} for k, v in means.items()]) \
        .to_csv(OUT_DIR / "vol_rel_selfcheck.csv", index=False)
    return means


# ================================================================ causality


def cmd_causality(panel: list[Asset]) -> bool:
    """The test_causality_strict.py tamper methodology, adapted from R-57's
    cmd_causality, constructing KellyRegimeRelativeVol directly (unregistered)
    instead of through the registry. Run on 2 panel assets + BTC (pre-2023
    only, per this round's holdout restriction)."""
    print("=" * 100)
    print("CAUSALITY TAMPER PROBE — KellyRegimeRelativeVol (novel branch)")
    print("=" * 100)
    market = MarketSpec.futures(leverage=5.0)

    btc_df, _ = load_dataset(ROOT / "data", "spot")
    btc_df = btc_df.loc[:"2022-12-31"]
    probe_assets = [("BTC", btc_df)] + [(a.ticker, a.df) for a in panel[:2]]

    all_ok = True
    for ticker, df in probe_assets:
        tail = df.iloc[-60_000:].copy()
        cut = len(tail) - 5_000
        bars = [cut - k for k in (1, 2, 3, 5, 10, 20)]
        up, down = tail.copy(), tail.copy()
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

        def decisions(frame):
            s = KellyRegimeRelativeVol()
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
        print(f"  {ticker:5s} decisions identical under opposite post-cut "
              f"tampers: {'PASS' if ok else 'FAIL'}")
    return all_ok


# ====================================================================== D1-D4


def cmd_d1(panel: list[Asset]) -> tuple[int, pd.DataFrame]:
    print("\n" + "=" * 100)
    print("D1 (PRIMARY) — PANEL_TRAIN 2020-04-01..2022-12-31, spot @0.10%, "
          "matched-exposure drawdown")
    print("=" * 100)
    rows: list[dict] = []
    for a in panel:
        cell(a.ticker, a.df, PANEL_TRAIN, SPOT_BASE, "TRAIN", rows)
    df = pd.DataFrame(rows)
    k1 = int((df.cand_dd < df.mh_dd).sum())
    excl = int(((df.dd_matched_lo > 0) | (df.dd_matched_hi < 0)).sum())
    better_excl = int((df.dd_matched_hi < 0).sum())
    p1 = binomial_tail(k1, 6)
    print(f"\nD1: {k1}/6 assets, exact binomial p={p1:.4f} -> {d1_verdict(k1)}")
    print(f"    paired bootstrap: {excl}/6 intervals exclude zero "
          f"({better_excl}/6 in candidate's favour)")
    return k1, df


def cmd_d2(panel: list[Asset]) -> tuple[dict[str, float], pd.DataFrame]:
    print("\n" + "=" * 100)
    print("D2 (FALSIFICATION, control) — CONTROL 2020-04-01..2022-12-31, "
          "BTC/ETH, spot @0.10%, matched-exposure drawdown")
    print("=" * 100)
    btc_df, _ = load_dataset(ROOT / "data", "spot")
    btc_df = btc_df.loc[:"2022-12-31"]
    eth_df = load_coinbase_spot(ROOT / "data", "ETH")
    eth_df = eth_df.loc[:"2022-12-31"]

    rows: list[dict] = []
    cell("BTC", btc_df, CONTROL, SPOT_BASE, "CONTROL", rows)
    cell("ETH", eth_df, CONTROL, SPOT_BASE, "CONTROL", rows)
    df = pd.DataFrame(rows)
    dd_advantage = {r["asset"]: r["dd_matched_diff"] for r in rows}

    passed = d2_passes(dd_advantage)
    print(f"\nD2 candidate dDD: BTC {dd_advantage['BTC']:+.1f}pp "
          f"(R-57 v4 control: {R57_CONTROL_DD_ADVANTAGE['BTC']:+.1f}pp), "
          f"ETH {dd_advantage['ETH']:+.1f}pp "
          f"(R-57 v4 control: {R57_CONTROL_DD_ADVANTAGE['ETH']:+.1f}pp), "
          f"tolerance +{D2_REGRESSION_TOLERANCE_PP}pp -> "
          f"{'PASSES' if passed else 'FAILS'}")
    return dd_advantage, df


def cmd_d3(panel: list[Asset]) -> tuple[int, pd.DataFrame]:
    print("\n" + "=" * 100)
    print("D3 (GENERALIZATION, descriptive) — PANEL_TEST 2023-01-01..2026-08-20, "
          "spot @0.10%")
    print("=" * 100)
    rows: list[dict] = []
    for a in panel:
        cell(a.ticker, a.df, PANEL_TEST, SPOT_BASE, "TEST", rows)
    df = pd.DataFrame(rows)
    k3 = int((df.cand_dd < df.mh_dd).sum())
    print(f"\nD3: {k3}/6 assets favour the candidate on the matched-exposure "
          f"drawdown axis (descriptive, not a gate)")
    return k3, df


def cmd_d4(panel: list[Asset]) -> tuple[int, pd.DataFrame]:
    print("\n" + "=" * 100)
    print("D4 (0.40% FALSIFICATION) — PANEL_TRAIN, spot @0.40%, beats "
          "buy_and_hold's final balance")
    print("=" * 100)
    rows: list[dict] = []
    for a in panel:
        cell(a.ticker, a.df, PANEL_TRAIN, SPOT_REAL, "TRAIN", rows)
    df = pd.DataFrame(rows)
    k4 = int((df.cand_final > df.hold_final).sum())
    verdict = "SURVIVES" if k4 >= 5 else "FAILS (as predicted)"
    print(f"\nD4: {k4}/6 -> {verdict}")
    return k4, df


# ========================================================================= main


def cmd_run() -> None:
    panel = load_panel()
    print(f"Panel ({len(panel)}): {', '.join(a.ticker for a in panel)}\n")

    ok = cmd_causality(panel)
    if not ok:
        raise SystemExit("CAUSALITY PROBE FAILED — refusing to report results "
                         "until the strategy is causal.")
    print()

    vol_rel_means = cmd_selfcheck(panel)

    k1, d1_df = cmd_d1(panel)
    dd_advantage, d2_df = cmd_d2(panel)
    k3, d3_df = cmd_d3(panel)
    k4, d4_df = cmd_d4(panel)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d1_df.to_csv(OUT_DIR / "d1_panel_train.csv", index=False)
    d2_df.to_csv(OUT_DIR / "d2_control.csv", index=False)
    d3_df.to_csv(OUT_DIR / "d3_panel_test.csv", index=False)
    d4_df.to_csv(OUT_DIR / "d4_panel_train_040.csv", index=False)

    verdict = promoted(k1, dd_advantage)
    print("\n" + "=" * 100)
    print("VERDICT (mechanical application of experiments.r59_shared.promoted)")
    print("=" * 100)
    print(f"D1: {k1}/6 -> {d1_verdict(k1)}")
    print(f"D2: {'PASSES' if d2_passes(dd_advantage) else 'FAILS'} "
          f"(BTC {dd_advantage['BTC']:+.1f}pp, ETH {dd_advantage['ETH']:+.1f}pp)")
    print(f"D3 (descriptive): {k3}/6")
    print(f"D4 (0.40% fee, beats buy_and_hold): {k4}/6")
    print(f"vol_rel self-consistency (mean, should be ~1.0): "
          f"{ {k: round(v, 3) for k, v in vol_rel_means.items()} }")
    print(f"\n-> {'PROMOTE-CANDIDATE' if verdict else 'NEGATIVE'}")
    print(f"\nTotal backtest configurations evaluated: {CONFIG_COUNT}")
    print("Holdout consultations added by this round: 0 "
          "(no BTC/ETH bar past 2022-12-31 is read anywhere in this module)")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        cmd_run()
        return
    panel = load_panel()
    if cmd == "selfcheck":
        cmd_selfcheck(panel)
    elif cmd == "causality":
        cmd_causality(panel)
    else:
        raise SystemExit(f"unknown command {cmd!r} (selfcheck | causality | run)")
    print(f"\nTotal backtest configurations evaluated: {CONFIG_COUNT}")


if __name__ == "__main__":
    main()
