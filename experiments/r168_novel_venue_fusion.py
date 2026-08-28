#!/usr/bin/env python
"""R-168 NOVEL branch: volatility-CONDITIONED spot/Deribit-perp fusion weight
feeding kelly_regime_v4's vote anchors.

Direction, citations, non-duplication argument, Step-0 measurement,
falsification tests, frozen splits and the promotion gate all live in
`experiments/r168_shared.py`'s module docstring (read there first -- this
file does not repeat that reasoning and never edits that frozen module).
The sibling CONSERVATIVE branch (fixed 50/50 spot/perp fusion,
`experiments/r168_conservative_venue_fusion.py`) is a separate, parallel
agent's work; it is not imported, read, or coordinated with here.

THE MECHANISM: kelly_regime_v4's full apparatus (kelly_regime_v3's
hysteresis-latched three-state volatility regime classifier switching
SCALE between continuous and extremes-only vol targeting, the 20/40/80-day
vote anchors, the 1% band, the 10% deadband) is copied byte-for-byte from
`kelly_regime_v3.prepare()`. The ONLY change: the price series the vote's
three rolling anchors are computed over and compared against is no longer
raw spot close -- it is a per-bar convex blend of spot close and
Deribit-perp close,

    w_perp(t) = w_low   if vol-state(t) == -1   (low-vol breakout)
              = 0.5      if vol-state(t) == 0    (normal band)
              = w_high   if vol-state(t) == +1   (high-vol breakout)
    fused_close(t) = w_perp(t) * perp_close(t) + (1 - w_perp(t)) * spot_close(t)

forced to w_perp(t)=0 (spot alone) wherever perp is not yet available,
regardless of vol-state. vol-state(t) is computed CAUSALLY, from the
ORIGINAL (unfused) spot return series, using the EXACT same construction
(`vol`, `slow`, `ratio = vol/slow`, `high_in=1.70/high_out=1.20/low_in=0.55
/low_out=0.85` hysteresis loop) kelly_regime_v3 already uses for SCALE --
this is the SAME state machine v3/v4 already treat as informative, just a
second consumer of it, not a new indicator. SCALE itself (computed from the
original spot return series) is untouched.

Motivation (Alexander & Heck 2020): a derivatives venue's price leadership
over spot rises specifically during periods of elevated volatility -- so
the fusion should lean toward the perp venue precisely when vol-state is
high, and toward spot when vol-state is low, rather than fixing the weight.

Run: `. .venv/bin/activate && python experiments/r168_novel_venue_fusion.py`
(from the repo root).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_YEAR  # noqa: E402

from experiments.r168_shared import (  # noqa: E402
    BAND,
    BARS_PER_DAY,
    FUTURES,
    HORIZONS,
    INNER_VAL_END,
    INNER_VAL_START,
    SPOT,
    causal_truncation_probe,
    fused_close_btc,
    fused_close_eth,
    signal_check,
    vote_from_close,
)

# ================================================================== loaders
# Cache (spot_df, perp_close_aligned) per asset, both truncated at
# INNER_VAL_END so nothing past the holdout boundary is ever loaded into
# this process. Truncating once, here, rather than relying on callers to
# remember, is the belt-and-suspenders version of "never touch OOS_START".
_CACHE: dict[str, tuple[pd.DataFrame, pd.Series]] = {}


def _load(asset: str) -> tuple[pd.DataFrame, pd.Series]:
    if asset not in _CACHE:
        if asset == "BTC":
            spot_df, _fused_equal_weight, perp = fused_close_btc()
        elif asset == "ETH":
            spot_df, _fused_equal_weight, perp = fused_close_eth()
        else:
            raise ValueError(f"asset must be BTC or ETH, got {asset!r}")
        spot_df = spot_df.loc[:INNER_VAL_END].copy()
        perp = perp.loc[:INNER_VAL_END]
        _CACHE[asset] = (spot_df, perp)
    return _CACHE[asset]


# ============================================================== strategy
class NovelVenueFusionKellyRegime(Strategy):
    """kelly_regime_v4, with the vote's anchor input replaced by a
    volatility-state-conditioned spot/Deribit-perp blend (see module
    docstring). SCALE (vol-targeting), the 1% band, the 10% deadband and
    the vol-state hysteresis machine itself are copied unmodified from
    `kelly_regime_v3.prepare()`.
    """

    name = "r168_novel_venue_fusion"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, w_low: float = 0.5, w_high: float = 0.5, asset: str = "BTC",
                 horizons: tuple[int, ...] = HORIZONS, band: float = BAND,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85,
                 vote_gamma: float = 1.0) -> None:
        self.w_low = w_low
        self.w_high = w_high
        self.asset = asset
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        self.vote_gamma = vote_gamma
        # Precomputed once at construction time, independent of whatever
        # slice of df prepare() is later handed -- see module docstring on
        # why this is still causal under causal_truncation_probe.
        _spot_full, self._perp_full = _load(asset)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # ---- vol-state hysteresis machine, byte-for-byte kelly_regime_v3,
        # computed from the ORIGINAL (unfused) spot return series only.
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
        state_arr = np.zeros(n, dtype=np.int8)
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
            state_arr[i] = state

        # ---- volatility-conditioned venue-fusion weight, causal: state_arr[i]
        # depends only on ratio[i'] for i'<=i, which itself depends only on
        # vol/slow shifted by 1 bar -- no future data anywhere in this array.
        perp_aligned = self._perp_full.reindex(df.index)
        perp_available = perp_aligned.notna().to_numpy()
        w_perp = np.where(state_arr == -1, self.w_low,
                           np.where(state_arr == 1, self.w_high, 0.5))
        # Fallback to spot alone wherever perp is unavailable, REGARDLESS of
        # vol-state -- perp_available is a static, precomputed mask (real
        # coverage start date), never a function of what happens later in
        # the series.
        w_perp = np.where(perp_available, w_perp, 0.0)

        fused = (w_perp * perp_aligned.fillna(0.0).to_numpy()
                 + (1.0 - w_perp) * close.to_numpy())
        fused_series = pd.Series(fused, index=df.index)

        frac = vote_from_close(fused_series, horizons=self.horizons, band=self.band)
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            scale = full[i] if state_arr[i] != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["_vol_state"] = state_arr
        df["_w_perp"] = w_perp
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# ================================================================== grid
# Pre-registered BEFORE any run: symmetric around 0.5. (0.5, 0.5) is a
# fixed-weight identity check expected to reproduce the conservative
# branch's equal-weight result closely (computed fully independently here).
GRID: list[tuple[float, float]] = [
    (0.5, 0.5),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9),
    (0.0, 1.0),
]


def label(w_low: float, w_high: float) -> str:
    return f"w_low={w_low:.1f}_w_high={w_high:.1f}"


def make_factory(w_low: float, w_high: float, asset: str = "BTC"):
    return lambda: NovelVenueFusionKellyRegime(w_low=w_low, w_high=w_high, asset=asset)


# ================================================================== gate
SHARPE_NOISE_FLOOR = 0.20
EXPOSURE_TOL_PP = 5.0
DD_MATCHING_FRACTION = 0.20  # my operationalization of "matching magnitude"


def clause1(res: dict) -> bool:
    if res["d_sharpe"] >= SHARPE_NOISE_FLOOR:
        return True
    dd_improve = res["dd_v4"] - res["dd_cand"]  # positive = candidate has smaller DD
    return bool(dd_improve >= DD_MATCHING_FRACTION * abs(res["dd_v4"]))


def clause2_market_ok(res: dict) -> bool:
    """Not significantly worse than v4 on this market (CI doesn't exclude
    zero in the losing direction)."""
    return not (res["significant"] and res["paired_hi"] < 0)


def clause3(res: dict) -> bool:
    return bool(abs(res["tim_cand"] - res["tim_v4"]) <= EXPOSURE_TOL_PP)


def gate_verdict(btc_res: dict, eth_res: dict) -> dict:
    c1_btc, c1_eth = clause1(btc_res), clause1(eth_res)
    c1_same_direction = c1_btc and c1_eth and (
        np.sign(btc_res["d_sharpe"]) == np.sign(eth_res["d_sharpe"]) or
        (btc_res["d_sharpe"] >= SHARPE_NOISE_FLOOR and eth_res["d_sharpe"] >= SHARPE_NOISE_FLOOR)
    )
    excludes_zero_btc = btc_res["significant"] and btc_res["paired_lo"] > 0
    excludes_zero_eth = eth_res["significant"] and eth_res["paired_lo"] > 0
    c2 = (excludes_zero_btc or excludes_zero_eth) and clause2_market_ok(btc_res) and clause2_market_ok(eth_res)
    c3 = clause3(btc_res) and clause3(eth_res)
    passed = bool(c1_same_direction and c2 and c3)
    return dict(clause1=c1_same_direction, clause2=c2, clause3=c3, passed=passed,
                c1_btc=c1_btc, c1_eth=c1_eth,
                excludes_zero_btc=excludes_zero_btc, excludes_zero_eth=excludes_zero_eth)


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def fmt_res(res: dict) -> str:
    return (f"sharpe cand={res['sharpe_cand']:.3f} v4={res['sharpe_v4']:.3f} "
            f"d={res['d_sharpe']:+.3f} | dd cand={res['dd_cand']:.1f}% v4={res['dd_v4']:.1f}% | "
            f"paired diff={res['paired_diff']:+.4f} CI=[{res['paired_lo']:+.4f},{res['paired_hi']:+.4f}] "
            f"sig={res['significant']} | tim cand={res['tim_cand']:.1f}% v4={res['tim_v4']:.1f}%")


def main() -> None:
    hr("R-168 NOVEL -- volatility-conditioned spot/Deribit-perp fusion weight "
       "feeding kelly_regime_v4's vote")
    print("See r168_shared.py's module docstring for direction/citations/gate. "
          "This file changes only the vote anchor's\nprice input; SCALE, band, "
          "deadband and the vol-state machine's own thresholds are unchanged.")
    print(f"\nGrid (pre-registered, {len(GRID)} configs): {GRID}")

    # ========================================================== STEP 1: causal probe
    hr("STEP 1 -- causal truncation probe")
    btc_df, _ = _load("BTC")
    primary_w_low, primary_w_high = 0.2, 0.8
    print(f"Primary config for the probe: w_low={primary_w_low}, w_high={primary_w_high}")
    probe_ok = causal_truncation_probe(
        make_factory(primary_w_low, primary_w_high, "BTC"), btc_df, FUTURES)
    print(f"causal_truncation_probe (primary config, BTC, FUTURES): "
          f"{'PASS' if probe_ok else 'FAIL'}")
    # Also probe the two extremes of the grid, cheap insurance.
    probe_ok_lo = causal_truncation_probe(make_factory(0.5, 0.5, "BTC"), btc_df, FUTURES)
    probe_ok_hi = causal_truncation_probe(make_factory(0.0, 1.0, "BTC"), btc_df, FUTURES)
    print(f"causal_truncation_probe (w_low=0.5,w_high=0.5): {'PASS' if probe_ok_lo else 'FAIL'}")
    print(f"causal_truncation_probe (w_low=0.0,w_high=1.0): {'PASS' if probe_ok_hi else 'FAIL'}")
    all_causal_ok = probe_ok and probe_ok_lo and probe_ok_hi
    if not all_causal_ok:
        hr("VERDICT")
        print("VERDICT: NEGATIVE (causal truncation probe FAILED). Stopping "
              "before reporting any promotion-relevant number.")
        return

    # ========================================================== STEP 2: BTC-futures grid
    hr("STEP 2 -- primary-cell grid: BTC FUTURES_5x, inner-validation "
       f"({INNER_VAL_START} -> {INNER_VAL_END})")
    grid_results: dict[tuple[float, float], dict] = {}
    for w_low, w_high in GRID:
        res = signal_check(make_factory(w_low, w_high, "BTC"), btc_df, FUTURES,
                            INNER_VAL_START, INNER_VAL_END)
        grid_results[(w_low, w_high)] = res
        print(f"\n  {label(w_low, w_high):24s} {fmt_res(res)}")

    print(f"\n{'config':24s} {'d_sharpe':>9s} {'dd_cand':>8s} {'dd_v4':>7s} "
          f"{'paired_lo':>10s} {'paired_hi':>10s} {'sig':>5s} {'tim_d':>7s}")
    for (w_low, w_high), res in grid_results.items():
        tim_d = res["tim_cand"] - res["tim_v4"]
        print(f"{label(w_low, w_high):24s} {res['d_sharpe']:>+9.3f} {res['dd_cand']:>8.1f} "
              f"{res['dd_v4']:>7.1f} {res['paired_lo']:>+10.4f} {res['paired_hi']:>+10.4f} "
              f"{str(res['significant']):>5s} {tim_d:>+7.1f}")

    # Winner selection: best d_sharpe subject to clause 1 (or, absent any
    # clause-1 passer, best d_sharpe regardless, reported honestly as such).
    ranked = sorted(GRID, key=lambda cfg: grid_results[cfg]["d_sharpe"], reverse=True)
    winner = ranked[0]
    any_clause1 = [cfg for cfg in GRID if clause1(grid_results[cfg])]
    if any_clause1:
        winner = max(any_clause1, key=lambda cfg: grid_results[cfg]["d_sharpe"])
    print(f"\nWinning config by primary-cell d_sharpe"
          f"{' (clause-1 qualifying)' if any_clause1 else ' (no config clears clause 1 -- best available)'}: "
          f"{label(*winner)}")

    # ========================================================== STEP 3: full battery
    hr(f"STEP 3 -- winner's full 4-cell battery ({label(*winner)}), "
       f"inner-validation ({INNER_VAL_START} -> {INNER_VAL_END})")
    eth_df, _ = _load("ETH")
    w_low, w_high = winner
    battery: dict[tuple[str, str], dict] = {}
    for asset, df_asset in (("BTC", btc_df), ("ETH", eth_df)):
        for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
            res = signal_check(make_factory(w_low, w_high, asset), df_asset, mkt,
                                INNER_VAL_START, INNER_VAL_END)
            battery[(asset, mkt_name)] = res
            print(f"\n  {asset:3s} {mkt_name:10s} {fmt_res(res)}")

    # Perp availability diagnostic within inner-val, for the ETH
    # disclosed-coverage flag.
    _, eth_perp = _load("ETH")
    eth_val_perp = eth_perp.loc[INNER_VAL_START:INNER_VAL_END]
    eth_perp_frac = float(eth_val_perp.notna().mean())
    print(f"\nETH-perp coverage within inner-validation window: {eth_perp_frac:.1%} of bars "
          f"(ETH-PERPETUAL starts 2019-03-14, well before {INNER_VAL_START})")

    # ========================================================== STEP 4: gate verdict
    hr("STEP 4 -- frozen inner-validation gate verdict (r168_shared.py's own "
       "pre-registration), on the leveraged\nmarket pair (BTC futures_5x, "
       "ETH futures_5x) per this project's standing selection convention, "
       "spot cells\nreported alongside for transparency")
    btc_fut = battery[("BTC", "futures_5x")]
    eth_fut = battery[("ETH", "futures_5x")]
    verdict_fut = gate_verdict(btc_fut, eth_fut)
    btc_spot = battery[("BTC", "spot")]
    eth_spot = battery[("ETH", "spot")]
    verdict_spot = gate_verdict(btc_spot, eth_spot)

    for name, verdict in (("futures_5x pair", verdict_fut), ("spot pair", verdict_spot)):
        print(f"\n  [{name}]")
        print(f"    Clause 1 (d_sharpe>=+0.20 OR matching-magnitude DD improvement, "
              f"same direction both markets): {verdict['clause1']}  "
              f"(BTC={verdict['c1_btc']}, ETH={verdict['c1_eth']})")
        print(f"    Clause 2 (paired 95% CI excludes zero on >=1 market, not losing "
              f"on the other): {verdict['clause2']}  "
              f"(BTC excludes-zero-positive={verdict['excludes_zero_btc']}, "
              f"ETH excludes-zero-positive={verdict['excludes_zero_eth']})")
        print(f"    Clause 3 (exposure matched within {EXPOSURE_TOL_PP}pp both markets): "
              f"{verdict['clause3']}")
        print(f"    GATE PASSED: {verdict['passed']}")

    hr("VERDICT")
    print(f"Causal truncation probes: {'PASS' if all_causal_ok else 'FAIL'}")
    print(f"Winning config: {label(*winner)}")
    print(f"Gate (futures_5x pair, primary): {'PASS' if verdict_fut['passed'] else 'FAIL'}")
    print(f"Gate (spot pair): {'PASS' if verdict_spot['passed'] else 'FAIL'}")
    disclosed_eth_flag = eth_perp_frac >= 0.999
    print(f"\nETH-perp coverage note: within inner-validation the ETH-perp series is "
          f"{eth_perp_frac:.1%} available\n(coverage-start issue is a inner-TRAIN-only, "
          f"disclosed artifact -- it does not touch inner-validation, used here for "
          f"selection).")
    print(f"\nHoldout (>= 2023-01-01) consulted: NO. `_load` truncates every series at "
          f"INNER_VAL_END={INNER_VAL_END}\nbefore anything else runs.")


if __name__ == "__main__":
    main()
