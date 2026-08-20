"""R-60 NOVEL BRANCH (backlog B-26): regime-switching-frequency-adaptive
hysteresis for the `kelly_regime` family's vote/gate.

Not registered: lives under ``experiments/`` per ROUTINE.md step 5. This is
an experiment, not a strategy the runner should discover, so
``KellyRegimeFlipAdaptiveHysteresis`` below is a plain ``Strategy`` subclass
with no ``@register`` decorator.

Pre-registration: ``experiments/r60_shared.py`` (read it first — this file
imports its windows, costs, panel loader and decision-rule helpers rather
than restating them). This module implements only the NOVEL branch's
candidate mechanism and the frozen D1-D4 measurement matrix pointed at it.

=====================================================================
THE MECHANISM
=====================================================================

``KellyRegime.prepare()`` (inherited unchanged through ``KellyRegimeV3`` and
``KellyRegimeV4``) computes three anchor votes, each latched inside a fixed
``band`` around its own rolling mean:

    v = 1  if close > anchor * (1 + band)
    v = 0  if close < anchor * (1 - band)
    v = previous verdict, otherwise (the hysteresis / latch)

``band = 0.01`` (1%) is a single global constant, shared by BTC and by six
further instruments R-57 found to be measurably more mean-reverting. Dai,
Zhang & Zhu (2010, SIAM J. Financial Mathematics 1(1)) and Dai, Yang, Zhang
& Zhu (2016, Math. of Operations Research 41(2)) show that in an optimal
two-state regime-switching trend-following model, the buy/sell trigger is a
function of the regime's OWN transition intensity: an instrument whose
belief state flips more often needs a WIDER no-trade band (acting on every
flip is ruinous when flips are partly noise); a more persistent-regime
instrument can use a narrower one.

This branch's fix touches ONLY that one constant, and only via a frozen,
pre-computed, per-asset value — never a per-bar recomputation:

  1. On PANEL_TRAIN (2020-04-01..2022-12-31) ONLY, for each asset, measure
     the RAW (unbanded, unlatched) crossing signal ``sign(close - anchor)``
     for each of the three fixed 20/40/80-day anchors (unchanged — this
     branch does not touch how the anchors are computed) and count sign
     changes per year. This is the discrete-time analogue of a transition
     intensity. The three anchors' flip rates are combined into ONE
     asset-level scalar by a plain mean, deliberately kept to one number
     per asset rather than one per anchor (``measure_flip_rate`` below).

  2. ``band_asset = clip(BASE_BAND * f(asset_flip_rate / btc_flip_rate),
     BAND_CLIP_LO, BAND_CLIP_HI)``, ``BASE_BAND = 0.01`` (v4's own constant).
     Two functional forms for ``f`` are tried (``derive_band`` below):
     "linear" (``f(x) = x``) and a damped "sqrt" variant (``f(x) =
     sqrt(x)``), each clipped to [0.5%, 3.0%]. By construction
     ``asset_flip_rate / btc_flip_rate = 1`` for BTC itself, so BTC's own
     band comes out at exactly 1.0% under either variant — the fix is a
     structural no-op on the asset the mechanism was tuned on, which is
     what makes the D2 control close to unaffected by design rather than by
     luck.

     A note on ratio orientation, stated explicitly because it is easy to
     get backwards: the ratio is ``asset_flip_rate / btc_flip_rate`` (higher
     ratio for a busier regime -> wider band), matching the direction the
     Dai/Zhang/Zhu literature actually specifies ("switches more often ->
     wider band") and giving ``f(1) = 1`` for BTC by construction. The
     inverse ratio (``btc / asset``) would satisfy the same BTC=1.0%
     self-consistency check but would narrow the band for a busier regime
     — the opposite of the cited mechanism — so it is not used here.

  3. This is a FROZEN, pre-computed scalar per asset per variant, computed
     ONCE from PANEL_TRAIN by ``measure_flip_rate`` + ``derive_band``
     (both entirely separate from the strategy class), then passed as a
     plain constructor argument (``band=...``) to
     ``KellyRegimeFlipAdaptiveHysteresis`` — which is otherwise IDENTICAL
     to ``KellyRegimeV4``: same ``__init__`` signature (band was already a
     constructor parameter of the whole family, just never varied
     per-asset), no ``prepare()`` or ``on_bar()`` override at all. The
     anchor computation, vote-fraction averaging, the latching logic and
     the entire exposure-scale mechanism (target_vol/max_leverage/breakout
     hysteresis) are inherited byte-for-byte from ``KellyRegimeV3`` — this
     branch's only change, anywhere, is which float is passed as ``band``.

Because the fitted value is a plain constructor argument used unchanged
across PANEL_TRAIN (D1), PANEL_TEST (D3) and CONTROL (D2), and
``measure_flip_rate``/``derive_band`` are never called from inside
``prepare()``/``on_bar()``, there is no new lookahead surface to probe: the
causality tamper probe below exercises the same ``prepare()``/``on_bar()``
code path v4 already has, just constructed with a pre-baked ``band``.

Literature basis, full citations: ``experiments/r60_shared.py``.

Usage::

    uv run python experiments/r60_novel_flip_adaptive_hysteresis.py bands
    uv run python experiments/r60_novel_flip_adaptive_hysteresis.py causality
    uv run python experiments/r60_novel_flip_adaptive_hysteresis.py run
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.matched_hold import ConstantExposureHold, mean_notional  # noqa: E402
from experiments.r57_cross_asset_panel import binomial_tail  # noqa: E402
from experiments.r60_shared import (  # noqa: E402
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
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

OUT_DIR = ROOT / "reports" / "r60_novel"
BOOT_KW = dict(mean_block=30.0, n_boot=2_000, seed=7)

FLIP_HORIZONS = (20, 40, 80)  # identical to KellyRegimeV4's anchor ladder
BASE_BAND = 0.01              # v4's own constant, unchanged
BAND_CLIP_LO = 0.005          # 0.5%
BAND_CLIP_HI = 0.03           # 3.0%

VARIANTS = {
    "linear": lambda ratio: ratio,
    "sqrt": lambda ratio: math.sqrt(ratio),
}

CONFIG_COUNT = 0
FLIP_MEASURE_COUNT = 0  # separate counter: not a backtest, a data pass


# ============================================================ the strategy


class KellyRegimeFlipAdaptiveHysteresis(KellyRegimeV4):
    """v4 with ONE change: a per-asset, PANEL_TRAIN-frozen hysteresis band.

    No ``prepare()``/``on_bar()`` override — ``band`` was already a
    constructor parameter of the whole ``kelly_regime`` family
    (``KellyRegime.__init__``); this class exists only to (a) make ``band``
    a required, explicit argument, so a caller cannot forget to pass the
    fitted value and silently fall back to v4's 1%, and (b) carry a distinct
    ``name`` for reporting. Everything else — anchors, vote, latch, sizing —
    is byte-identical to ``KellyRegimeV3.prepare()``/``KellyRegimeV4``
    because it is the literal same inherited code, not a copy.
    """

    name = "kelly_regime_v60_flipband"
    warmup = 80 * BARS_PER_DAY + 10  # identical to KellyRegimeV4 (horizons-driven)

    def __init__(self, band: float, horizons: tuple[int, ...] = (20, 40, 80),
                 **kwargs) -> None:
        super().__init__(horizons=horizons, band=band, **kwargs)


# ==================================================== offline fitting step
#
# Entirely separate from the strategy class above. Run ONCE per asset on
# PANEL_TRAIN, before any backtest of KellyRegimeFlipAdaptiveHysteresis is
# executed, and never called from prepare()/on_bar().


def measure_flip_rate(df: pd.DataFrame, start: str, end: str,
                       horizons: tuple[int, ...] = FLIP_HORIZONS,
                       ) -> tuple[float, dict[int, float]]:
    """Mean annualized sign-flip rate of sign(close - anchor) over [start,end].

    RAW crossing signal — no band, no latch — for each of the fixed anchor
    horizons (unchanged from v4). Counts a "flip" as any bar where the sign
    differs from the previous bar's, restricted to the [start, end] window
    (the anchors themselves are computed on the full series so the window's
    own bars are never cold-started). Combined into one asset-level scalar
    by a plain mean across the three anchors.
    """
    global FLIP_MEASURE_COUNT
    FLIP_MEASURE_COUNT += 1
    close = df["close"]
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    per_anchor: dict[int, float] = {}
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        sign = np.sign(close - anchor)
        window = sign.loc[start_ts:end_ts].dropna()
        if len(window) < 2:
            per_anchor[days] = float("nan")
            continue
        vals = window.to_numpy()
        flips = int((vals[1:] != vals[:-1]).sum())
        span_years = (window.index[-1] - window.index[0]) / pd.Timedelta(days=365.25)
        per_anchor[days] = flips / span_years if span_years > 0 else float("nan")
    combined = float(np.mean(list(per_anchor.values())))
    return combined, per_anchor


def derive_band(asset_flip_rate: float, btc_flip_rate: float, variant: str) -> float:
    """band_asset = clip(BASE_BAND * f(asset_flip_rate / btc_flip_rate)).

    ratio > 1 for an asset that flips MORE often than BTC -> wider band
    (per Dai/Zhang/Zhu: busier regime -> wider no-trade band). ratio = 1
    for BTC itself by construction, so BTC's derived band is exactly
    BASE_BAND under either variant. See module docstring for why this
    orientation, not its inverse, was chosen.
    """
    ratio = asset_flip_rate / btc_flip_rate
    f = VARIANTS[variant]
    band = BASE_BAND * f(ratio)
    return float(np.clip(band, BAND_CLIP_LO, BAND_CLIP_HI))


def load_frames(panel: list[Asset]) -> list[tuple[str, pd.DataFrame]]:
    """BTC (pre-2023 only) + ETH (pre-2023 only) + the six panel assets."""
    btc_df, _ = load_dataset(ROOT / "data", "spot")
    btc_df = btc_df.loc[:"2022-12-31"]
    eth_df = load_coinbase_spot(ROOT / "data", "ETH")
    eth_df = eth_df.loc[:"2022-12-31"]
    return [("BTC", btc_df), ("ETH", eth_df)] + [(a.ticker, a.df) for a in panel]


def cmd_bands(panel: list[Asset]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Measure PANEL_TRAIN flip rates for all 8 assets and derive bands.

    Returns (flip_rates, bands) where bands[ticker][variant] is the frozen,
    clipped hysteresis width for that asset under that functional form.
    """
    print("=" * 100)
    print("FLIP-RATE MEASUREMENT — PANEL_TRAIN 2020-04-01..2022-12-31, raw "
          "sign(close-anchor), horizons=(20,40,80)d")
    print("=" * 100)
    frames = load_frames(panel)
    flip_rates: dict[str, float] = {}
    per_anchor_rows: list[dict] = []
    for ticker, df in frames:
        combined, per_anchor = measure_flip_rate(df, *PANEL_TRAIN)
        flip_rates[ticker] = combined
        per_anchor_rows.append({"asset": ticker, **{f"flip_rate_{d}d": v
                                                     for d, v in per_anchor.items()},
                                "flip_rate_mean": combined})
        anchors_str = ", ".join(f"{d}d={v:6.2f}/yr" for d, v in per_anchor.items())
        print(f"  {ticker:5s} {anchors_str}  ->  mean {combined:6.2f} flips/yr")

    btc_rate = flip_rates["BTC"]
    bands: dict[str, dict[str, float]] = {}
    band_rows: list[dict] = []
    print(f"\nBTC flip rate (normalization base): {btc_rate:.3f} flips/yr")
    print(f"{'asset':6s} {'flip/yr':>9s} {'ratio(a/btc)':>13s} "
          f"{'band_linear':>12s} {'band_sqrt':>10s}")
    for ticker, rate in flip_rates.items():
        ratio = rate / btc_rate
        b_lin = derive_band(rate, btc_rate, "linear")
        b_sqrt = derive_band(rate, btc_rate, "sqrt")
        bands[ticker] = {"linear": b_lin, "sqrt": b_sqrt}
        band_rows.append({"asset": ticker, "flip_rate": rate, "ratio_vs_btc": ratio,
                          "band_linear": b_lin, "band_sqrt": b_sqrt})
        print(f"{ticker:6s} {rate:9.2f} {ratio:13.3f} {b_lin:12.4f} {b_sqrt:10.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_anchor_rows).to_csv(OUT_DIR / "flip_rates_per_anchor.csv", index=False)
    pd.DataFrame(band_rows).to_csv(OUT_DIR / "flip_rates_bands.csv", index=False)
    return flip_rates, bands


# ================================================================ causality


def cmd_causality(panel: list[Asset], bands: dict[str, dict[str, float]],
                   variant: str) -> bool:
    """Opposite-tamper probe (R-57's cmd_causality pattern), constructing
    KellyRegimeFlipAdaptiveHysteresis directly with each asset's frozen band,
    on BTC (pre-2023 only) plus two panel assets."""
    print("=" * 100)
    print(f"CAUSALITY TAMPER PROBE — KellyRegimeFlipAdaptiveHysteresis "
          f"(novel branch, variant={variant!r})")
    print("=" * 100)
    market = MarketSpec.futures(leverage=5.0)

    btc_df, _ = load_dataset(ROOT / "data", "spot")
    btc_df = btc_df.loc[:"2022-12-31"]
    probe_assets = [("BTC", btc_df)] + [(a.ticker, a.df) for a in panel[:2]]

    all_ok = True
    for ticker, df in probe_assets:
        band = bands[ticker][variant]
        tail = df.iloc[-60_000:].copy()
        cut = len(tail) - 5_000
        bars = [cut - k for k in (1, 2, 3, 5, 10, 20)]
        up, down = tail.copy(), tail.copy()
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

        def decisions(frame, band=band):
            s = KellyRegimeFlipAdaptiveHysteresis(band=band)
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
        print(f"  {ticker:5s} band={band:.4f} decisions identical under "
              f"opposite post-cut tampers: {'PASS' if ok else 'FAIL'}")
    return all_ok


# ==================================================================== D1-D4


def measure(strategy, df, start, end, market):
    global CONFIG_COUNT
    CONFIG_COUNT += 1
    result = run_period(strategy, df, start, end, market=market, start_balance=1_000.0)
    return result, compute_metrics(result)


def cell(ticker: str, df: pd.DataFrame, band: float, window, market, label: str,
         rows: list) -> dict:
    """One asset x window x market cell: candidate, hold, matched hold, intervals."""
    start, end = window
    cand_res, cand = measure(KellyRegimeFlipAdaptiveHysteresis(band=band),
                             df, start, end, market)
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
        "fee": market.fee_rate, "band": band,
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
    print(f"  {ticker:5s} {label:9s} {market.name:11s} fee={market.fee_rate:.2%} "
          f"band={band:.4f}  "
          f"cand ${cand.final_balance:>10,.0f} DD {cand.max_drawdown_pct:5.1f}% | "
          f"hold ${hold.final_balance:>10,.0f} DD {hold.max_drawdown_pct:5.1f}% | "
          f"matched(c={c_mean:.2f}) ${mh.final_balance:>10,.0f} "
          f"DD {mh.max_drawdown_pct:5.1f}% | "
          f"dDD_matched {dd_matched.diff.point:+6.1f}pp "
          f"[{dd_matched.diff.lo:+6.1f},{dd_matched.diff.hi:+6.1f}]")
    return row


def cmd_d1_variant(panel: list[Asset], bands: dict[str, dict[str, float]],
                   variant: str) -> tuple[int, pd.DataFrame]:
    print("\n" + "=" * 100)
    print(f"D1 (PRIMARY, variant={variant!r}) — PANEL_TRAIN 2020-04-01..2022-12-31, "
          "spot @0.10%, matched-exposure drawdown")
    print("=" * 100)
    rows: list[dict] = []
    for a in panel:
        cell(a.ticker, a.df, bands[a.ticker][variant], PANEL_TRAIN, SPOT_BASE, "TRAIN", rows)
    df = pd.DataFrame(rows)
    k1 = int((df.cand_dd < df.mh_dd).sum())
    excl = int(((df.dd_matched_lo > 0) | (df.dd_matched_hi < 0)).sum())
    better_excl = int((df.dd_matched_hi < 0).sum())
    p1 = binomial_tail(k1, 6)
    print(f"\nD1 [{variant}]: {k1}/6 assets, exact binomial p={p1:.4f} -> {d1_verdict(k1)}")
    print(f"    paired bootstrap: {excl}/6 intervals exclude zero "
          f"({better_excl}/6 in candidate's favour)")
    return k1, df


def cmd_d2(panel: list[Asset], bands: dict[str, dict[str, float]],
          variant: str) -> tuple[dict[str, float], pd.DataFrame]:
    print("\n" + "=" * 100)
    print(f"D2 (FALSIFICATION, control, variant={variant!r}) — CONTROL "
          "2020-04-01..2022-12-31, BTC/ETH, spot @0.10%, matched-exposure drawdown")
    print("=" * 100)
    btc_df, _ = load_dataset(ROOT / "data", "spot")
    btc_df = btc_df.loc[:"2022-12-31"]
    eth_df = load_coinbase_spot(ROOT / "data", "ETH")
    eth_df = eth_df.loc[:"2022-12-31"]

    rows: list[dict] = []
    cell("BTC", btc_df, bands["BTC"][variant], CONTROL, SPOT_BASE, "CONTROL", rows)
    cell("ETH", eth_df, bands["ETH"][variant], CONTROL, SPOT_BASE, "CONTROL", rows)
    df = pd.DataFrame(rows)
    dd_advantage = {r["asset"]: r["dd_matched_diff"] for r in rows}

    passed = d2_passes(dd_advantage)
    print(f"\nD2 [{variant}] candidate dDD: BTC {dd_advantage['BTC']:+.1f}pp "
          f"(R-57 v4 control: {R57_CONTROL_DD_ADVANTAGE['BTC']:+.1f}pp), "
          f"ETH {dd_advantage['ETH']:+.1f}pp "
          f"(R-57 v4 control: {R57_CONTROL_DD_ADVANTAGE['ETH']:+.1f}pp), "
          f"tolerance +{D2_REGRESSION_TOLERANCE_PP}pp -> "
          f"{'PASSES' if passed else 'FAILS'}")
    return dd_advantage, df


def cmd_d3(panel: list[Asset], bands: dict[str, dict[str, float]],
          variant: str) -> tuple[int, pd.DataFrame]:
    print("\n" + "=" * 100)
    print(f"D3 (GENERALIZATION, descriptive, variant={variant!r}) — PANEL_TEST "
          "2023-01-01..2026-08-20, spot @0.10%")
    print("=" * 100)
    rows: list[dict] = []
    for a in panel:
        cell(a.ticker, a.df, bands[a.ticker][variant], PANEL_TEST, SPOT_BASE, "TEST", rows)
    df = pd.DataFrame(rows)
    k3 = int((df.cand_dd < df.mh_dd).sum())
    print(f"\nD3 [{variant}]: {k3}/6 assets favour the candidate on the "
          "matched-exposure drawdown axis (descriptive, not a gate)")
    return k3, df


def cmd_d4(panel: list[Asset], bands: dict[str, dict[str, float]],
          variant: str) -> tuple[int, pd.DataFrame]:
    print("\n" + "=" * 100)
    print(f"D4 (0.40% FALSIFICATION, variant={variant!r}) — PANEL_TRAIN, spot @0.40%, "
          "beats buy_and_hold's final balance")
    print("=" * 100)
    rows: list[dict] = []
    for a in panel:
        cell(a.ticker, a.df, bands[a.ticker][variant], PANEL_TRAIN, SPOT_REAL, "TRAIN", rows)
    df = pd.DataFrame(rows)
    k4 = int((df.cand_final > df.hold_final).sum())
    verdict = "SURVIVES" if k4 >= 5 else "FAILS (as predicted)"
    print(f"\nD4 [{variant}]: {k4}/6 -> {verdict}")
    return k4, df


# ========================================================================= main


def cmd_run() -> None:
    panel = load_panel()
    print(f"Panel ({len(panel)}): {', '.join(a.ticker for a in panel)}\n")

    flip_rates, bands = cmd_bands(panel)
    print()

    # Causality probed with the "linear" variant's bands first (any variant
    # exercises the identical prepare()/on_bar() code path — the strategy
    # class does not branch on the functional form at all, only on the
    # float that was passed in).
    ok = cmd_causality(panel, bands, "linear")
    if not ok:
        raise SystemExit("CAUSALITY PROBE FAILED — refusing to report results "
                         "until the strategy is causal.")
    print()

    # --- D1 run for BOTH functional-form variants (the small, pre-specified
    # grid), on PANEL_TRAIN only — this is the model-selection step the
    # round's pre-registration authorizes ("try at least a plain linear
    # ratio and one damped variant"). Whichever variant is frozen for
    # D2/D3/D4 is decided from THIS D1 result and reported honestly either
    # way; both D1 results are reported below regardless of which is frozen.
    d1_results: dict[str, tuple[int, pd.DataFrame]] = {}
    for variant in VARIANTS:
        d1_results[variant] = cmd_d1_variant(panel, bands, variant)

    k1_linear, _ = d1_results["linear"]
    k1_sqrt, _ = d1_results["sqrt"]
    if k1_linear != k1_sqrt:
        frozen_variant = "linear" if k1_linear > k1_sqrt else "sqrt"
        reason = f"D1 {frozen_variant} scored {max(k1_linear, k1_sqrt)}/6 vs {min(k1_linear, k1_sqrt)}/6"
    else:
        # Tie on D1's count: prefer the damped variant, per the cited
        # literature's own caution that a linear response over-reacts to a
        # noisily-estimated flip rate (a single extra/missing flip near the
        # PANEL_TRAIN boundary moves a linear band twice as far as a sqrt
        # one). Decided as a rule BEFORE looking at which variant this tie
        # would favor beyond the D1 count itself.
        frozen_variant = "sqrt"
        reason = f"D1 tied at {k1_linear}/6 for both variants; damped (sqrt) preferred by design rationale"

    k1, d1_df = d1_results[frozen_variant]
    print(f"\nFrozen functional-form variant: {frozen_variant!r} ({reason})")

    dd_advantage, d2_df = cmd_d2(panel, bands, frozen_variant)
    k3, d3_df = cmd_d3(panel, bands, frozen_variant)
    k4, d4_df = cmd_d4(panel, bands, frozen_variant)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d1_results["linear"][1].to_csv(OUT_DIR / "d1_panel_train_linear.csv", index=False)
    d1_results["sqrt"][1].to_csv(OUT_DIR / "d1_panel_train_sqrt.csv", index=False)
    d2_df.to_csv(OUT_DIR / "d2_control.csv", index=False)
    d3_df.to_csv(OUT_DIR / "d3_panel_test.csv", index=False)
    d4_df.to_csv(OUT_DIR / "d4_panel_train_040.csv", index=False)

    verdict = promoted(k1, dd_advantage)
    print("\n" + "=" * 100)
    print("VERDICT (mechanical application of experiments.r60_shared.promoted)")
    print("=" * 100)
    print(f"D1 [linear]: {k1_linear}/6 -> {d1_verdict(k1_linear)}")
    print(f"D1 [sqrt]:   {k1_sqrt}/6 -> {d1_verdict(k1_sqrt)}")
    print(f"D1 [frozen={frozen_variant}]: {k1}/6 -> {d1_verdict(k1)}")
    print(f"D2: {'PASSES' if d2_passes(dd_advantage) else 'FAILS'} "
          f"(BTC {dd_advantage['BTC']:+.1f}pp, ETH {dd_advantage['ETH']:+.1f}pp)")
    print(f"D3 (descriptive): {k3}/6")
    print(f"D4 (0.40% fee, beats buy_and_hold): {k4}/6")
    print(f"BTC flip rate: {flip_rates['BTC']:.2f}/yr (normalization base)")
    print(f"\n-> {'PROMOTE-CANDIDATE' if verdict else 'NEGATIVE'}")
    print(f"\nTotal backtest configurations evaluated: {CONFIG_COUNT}")
    print(f"Flip-rate measurement passes (not backtests): {FLIP_MEASURE_COUNT} "
          f"(1 per asset x 8 assets, each covering 3 anchors = 24 anchor-level "
          "flip-rate computations)")
    print("Holdout consultations added by this round: 0 "
          "(no BTC/ETH bar past 2022-12-31 is read anywhere in this module)")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        cmd_run()
        return
    panel = load_panel()
    if cmd == "bands":
        cmd_bands(panel)
    elif cmd == "causality":
        _, bands = cmd_bands(panel)
        cmd_causality(panel, bands, "linear")
    else:
        raise SystemExit(f"unknown command {cmd!r} (bands | causality | run)")
    print(f"\nTotal backtest configurations evaluated: {CONFIG_COUNT}")


if __name__ == "__main__":
    main()
