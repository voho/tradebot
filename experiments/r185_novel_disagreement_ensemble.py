#!/usr/bin/env python
"""R-185 NOVEL branch: causal, dynamically-reweighted VOTE-LEVEL ensemble of
kelly_regime_v4's spot vote and its Deribit-perp twin.

Direction, citations, non-duplication argument, Step-0 measurement,
falsification tests, frozen splits and the standard promotion gate all live
in `experiments/r185_shared.py`'s module docstring (read there first -- this
file does not repeat that reasoning and never edits that frozen module). The
sibling CONSERVATIVE branch (fixed on/off disagreement haircut,
`experiments/r185_conservative_disagreement_veto.py`) is a separate,
parallel agent's work; it is not imported, read, or coordinated with here.

THE MECHANISM, in one sentence: replace kelly_regime_v4's single vote
`frac_spot(t)` with a convex blend `frac(t) = (1-w_perp(t))*frac_spot(t) +
w_perp(t)*frac_perp(t)`, where `w_perp(t)` is set causally, per bar, by each
source's own trailing REALIZED quality (Bates & Granger 1969) -- not by a
fixed weight (that is the equal-weight control below) and not by a
disagreement-triggered haircut (that is the conservative branch). This is a
VOTE-LEVEL ensemble: the two inputs being blended are the already-computed
discrete vote fractions `frac_spot`, `frac_perp` in {0, 1/3, 2/3, 1}
(`shared.spot_and_perp_votes`), never raw close prices -- R-168 blended
PRICE LEVELS before computing anchors on the blend; this round never forms
a blended price series at all.

WEIGHTING FORMULA, exactly as implemented (see `_ensemble_series` below):

    r(t)          = log(close(t)) - log(close(t-1))          [causal]
    pnl_spot(t)   = frac_spot(t-1) * r(t)
    pnl_perp(t)   = frac_perp(t-1) * r(t)
    q_spot(t)     = EWMA(span=QUALITY_SPAN_BARS)(pnl_spot)[t-1]   (shifted once
                    more so q_spot(t) only sees pnl_spot(j) for j <= t-1)
    q_perp(t)     = EWMA(span=QUALITY_SPAN_BARS)(pnl_perp)[t-1]   (same)
    denom(t)      = q_spot(t) + q_perp(t)
    raw_w(t)      = q_perp(t) / denom(t)   if denom(t) > EPS_DENOM
                   = FALLBACK_W            otherwise (degenerate: both
                     sources' trailing quality sums to ~zero or negative --
                     no reliable read on RELATIVE reliability, so neither
                     source is trusted more than the other)
    w_perp(t)     = clip(raw_w(t), W_CLIP_LO, W_CLIP_HI)
    frac(t)       = (1 - w_perp(t)) * frac_spot(t) + w_perp(t) * frac_perp(t)

Causality: q_spot(t)/q_perp(t) depend only on frac_{spot,perp}(j) for
j <= t-2 and r(j) for j <= t-1 -- both strictly before bar t's own close.
w_perp(t) is a pure function of q_spot(t), q_perp(t), so it never touches
bar t's own return or close. `frac_spot(t)`/`frac_perp(t)` themselves use
bar t's own close, exactly as v4's existing vote already does (unchanged,
pre-existing, non-lookahead convention) -- the causality requirement in the
pre-registration is about the WEIGHT, not the vote inputs it blends.
Verified below with `shared.causal_truncation_probe`.

Named tunable parameters (the whole set, swept in `GRID` below):
  - `quality_span_days`: the trailing EWMA half-life-ish span (in days,
    converted to bars via BARS_PER_DAY) over which each source's realized
    quality is estimated. Grid: {45, 90, 180}.
  - `w_clip = (W_CLIP_LO, W_CLIP_HI)`: symmetric bounds the dynamic weight
    is clipped to, so a source is never trusted at 0% or (usually) 100%.
    Grid: {(0.1,0.9), (0.2,0.8), (0.0,1.0)}.
  - `EPS_DENOM = 1e-9` (fixed, not swept): degenerate-denominator threshold.
  - `FALLBACK_W = 0.5` (fixed, not swept): the exact fallback rule for the
    degenerate case -- default to an equal-weight blend, i.e. fall through
    to exactly what the control arm does everywhere, all the time.

CONTROL ARM (mandatory falsification control, same file, same machinery):
`frac(t) = 0.5*frac_spot(t) + 0.5*frac_perp(t)`, constant weight, zero new
tunable parameters (`mode="equal"` below -- literally `FALLBACK_W`
everywhere, unconditionally, rather than only on the degenerate branch).
The frozen gate requires the dynamic arm to beat this control by
`d_sharpe >= +0.20` with CI excluding zero on BTC inner-validation, IN
ADDITION to the standard BTC+ETH-vs-v4 gate -- both are evaluated below and
reported honestly, including a NEGATIVE if the dynamic weighting does not
clear the control (see `docs/RESEARCH.md`'s anchor-count finding: more
inputs alone does not reliably help in this project).

Both strategy variants otherwise copy kelly_regime_v3's SCALE / hysteresis /
deadband / latch loop byte-for-byte (same pattern R-168's novel branch
used) -- only the `frac` array feeding the loop differs.

Run: `. .venv/bin/activate && python experiments/r185_novel_disagreement_ensemble.py`
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

from tradebot.inference import daily_returns, paired_bootstrap, total_log_return  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402

from experiments import r185_shared as shared  # noqa: E402
from experiments.r185_shared import (  # noqa: E402
    FUTURES,
    INNER_VAL_END,
    INNER_VAL_START,
    SPOT,
    causal_truncation_probe,
    run_candidate,
    signal_check,
)

# ================================================================== loaders
# Cache (spot_df, frac_spot, frac_perp, mask) per asset, HARD-truncated at
# INNER_VAL_END before anything downstream can touch it -- belt-and-
# suspenders against ever reading 2023-01-01+ data, same discipline R-168's
# novel branch used.
_CACHE: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]] = {}


def _load(asset: str):
    if asset not in _CACHE:
        spot_df, frac_spot, frac_perp, mask = shared.spot_and_perp_votes(asset)
        pos = int(spot_df.index.searchsorted(INNER_VAL_END, side="right"))
        _CACHE[asset] = (
            spot_df.iloc[:pos].copy(),
            frac_spot[:pos],
            frac_perp[:pos],
            mask[:pos],
        )
    return _CACHE[asset]


# ============================================================ the ensemble
EPS_DENOM = 1e-9   # degenerate-denominator threshold, named and fixed (not swept)
FALLBACK_W = 0.5   # exact fallback rule for the degenerate case

_ENSEMBLE_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _ensemble_series(asset: str, mode: str, quality_span_days: float,
                      w_clip: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (frac, w_perp) arrays, full precomputed history (<=INNER_VAL_END),
    aligned to `_load(asset)[0].index`. Precomputed ONCE over the whole
    available history (not re-cold-started per backtest window) so the EWMA
    quality estimate is warm identically regardless of which sub-window
    (inner-train / inner-validation / a single market cell) is later
    backtested -- exactly analogous to how v4's own rolling anchors are
    unaffected by which slice of history they are later queried over,
    given adequate warmup.
    """
    key = (asset, mode) if mode == "equal" else (asset, mode, quality_span_days, w_clip)
    if key in _ENSEMBLE_CACHE:
        return _ENSEMBLE_CACHE[key]

    spot_df, frac_spot, frac_perp, _mask = _load(asset)
    idx = spot_df.index
    n = len(idx)

    if mode == "equal":
        w_perp = np.full(n, FALLBACK_W)
    elif mode == "dynamic":
        close = spot_df["close"]
        r = np.log(close).diff()
        fs = pd.Series(frac_spot, index=idx)
        fp = pd.Series(frac_perp, index=idx)
        pnl_spot = fs.shift(1) * r
        pnl_perp = fp.shift(1) * r
        span_bars = max(1, int(round(quality_span_days * BARS_PER_DAY)))
        # .shift(1) after the ewm mean: q_*(t) sees only pnl_*(j) for j<=t-1,
        # itself built from frac_*(j-1) and r(j) -- nothing at or after t.
        q_spot = pnl_spot.ewm(span=span_bars, min_periods=span_bars).mean().shift(1).to_numpy()
        q_perp = pnl_perp.ewm(span=span_bars, min_periods=span_bars).mean().shift(1).to_numpy()
        denom = q_spot + q_perp
        with np.errstate(invalid="ignore"):
            usable = denom > EPS_DENOM  # NaN (pre-warmup) compares False -> fallback
            raw_w = np.where(usable, np.divide(q_perp, denom, out=np.full(n, FALLBACK_W),
                                                where=usable), FALLBACK_W)
        lo, hi = w_clip
        w_perp = np.clip(raw_w, lo, hi)
    else:
        raise ValueError(f"mode must be 'dynamic' or 'equal', got {mode!r}")

    frac = (1.0 - w_perp) * frac_spot + w_perp * frac_perp
    _ENSEMBLE_CACHE[key] = (frac, w_perp)
    return frac, w_perp


# ================================================================ strategy
class DisagreementEnsembleKellyRegime(Strategy):
    """kelly_regime_v4 with its single spot vote replaced by a causal
    spot/Deribit-perp VOTE-level ensemble (see module docstring).

    `mode="dynamic"`: Bates & Granger (1969) trailing-quality reweighting.
    `mode="equal"`: the mandatory falsification control, a constant 50/50
    blend with zero new tunable parameters. SCALE (vol-targeting), the 1%
    vote band, the 10% deadband and the vol-state hysteresis machine are
    copied byte-for-byte from `kelly_regime_v3.prepare()` in both modes.
    """

    name = "r185_novel_disagreement_ensemble"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, asset: str = "BTC", mode: str = "dynamic",
                 quality_span_days: float = 90.0, w_clip: tuple[float, float] = (0.1, 0.9),
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85) -> None:
        self.asset = asset
        self.mode = mode
        self.quality_span_days = quality_span_days
        self.w_clip = w_clip
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        # Precomputed once at construction time, over the full precomputed
        # (<=INNER_VAL_END) history -- see `_ensemble_series` docstring on
        # why this stays causal under causal_truncation_probe regardless of
        # what slice of df `prepare()` is later handed.
        self._frac_full, self._w_perp_full = _ensemble_series(
            asset, mode, quality_span_days, w_clip)
        self._index_full = _load(asset)[0].index

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        frac = pd.Series(self._frac_full, index=self._index_full).reindex(df.index).to_numpy()
        w_perp = pd.Series(self._w_perp_full, index=self._index_full).reindex(df.index).to_numpy()

        # ---- vol-state hysteresis + conditional vol-targeting, byte-for-
        # byte kelly_regime_v3.prepare() (only the `frac` source above differs).
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
        df["_w_perp"] = w_perp
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# ============================================================ vs-control check
def signal_check_vs_control(cand_factory, control_factory, df: pd.DataFrame, market,
                             start: str, end: str, seed: int = 185) -> dict:
    """Same construction as `shared.signal_check`, but paired against the
    equal-weight control arm instead of v4 -- required by the gate's clause
    4. Reuses the frozen `run_candidate` utility for both runs."""
    m_cand, res_cand = run_candidate(cand_factory, df, market, start, end)
    m_ctrl, res_ctrl = run_candidate(control_factory, df, market, start, end)
    r_cand = daily_returns(res_cand.equity)
    r_ctrl = daily_returns(res_ctrl.equity)
    n = min(len(r_cand), len(r_ctrl))
    paired = paired_bootstrap(r_cand.to_numpy()[:n], r_ctrl.to_numpy()[:n],
                               stat=total_log_return, seed=seed)
    return {
        "sharpe_cand": m_cand.sharpe, "sharpe_ctrl": m_ctrl.sharpe,
        "d_sharpe": m_cand.sharpe - m_ctrl.sharpe,
        "dd_cand": m_cand.max_drawdown_pct, "dd_ctrl": m_ctrl.max_drawdown_pct,
        "paired_diff": paired.diff.point, "paired_lo": paired.diff.lo,
        "paired_hi": paired.diff.hi, "significant": paired.significant,
        "tim_cand": m_cand.time_in_market_pct, "tim_ctrl": m_ctrl.time_in_market_pct,
    }


# ================================================================== grid
# Pre-registered BEFORE any run: 3 quality spans x 3 clip-bound pairs.
SPAN_GRID_DAYS: list[float] = [45.0, 90.0, 180.0]
CLIP_GRID: list[tuple[float, float]] = [(0.1, 0.9), (0.2, 0.8), (0.0, 1.0)]
GRID: list[tuple[float, tuple[float, float]]] = [
    (span, clip) for span in SPAN_GRID_DAYS for clip in CLIP_GRID
]


def label(span: float, clip: tuple[float, float]) -> str:
    return f"span={span:.0f}d_clip=[{clip[0]:.1f},{clip[1]:.1f}]"


def make_dynamic_factory(span: float, clip: tuple[float, float], asset: str = "BTC"):
    return lambda: DisagreementEnsembleKellyRegime(
        asset=asset, mode="dynamic", quality_span_days=span, w_clip=clip)


def make_control_factory(asset: str = "BTC"):
    return lambda: DisagreementEnsembleKellyRegime(asset=asset, mode="equal")


# ================================================================== gate
SHARPE_NOISE_FLOOR = 0.20
EXPOSURE_TOL_PP = 5.0
DD_MATCHING_FRACTION = 0.20  # same operationalization R-168 used, for direct comparability


def clause1(res: dict) -> bool:
    if res["d_sharpe"] >= SHARPE_NOISE_FLOOR:
        return True
    dd_improve = res["dd_v4"] - res["dd_cand"]
    return bool(dd_improve >= DD_MATCHING_FRACTION * abs(res["dd_v4"]))


def clause2_market_ok(res: dict) -> bool:
    return not (res["significant"] and res["paired_hi"] < 0)


def clause3(res: dict) -> bool:
    return bool(abs(res["tim_cand"] - res["tim_v4"]) <= EXPOSURE_TOL_PP)


def gate_verdict_vs_v4(btc_res: dict, eth_res: dict) -> dict:
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


def clause4_vs_control(res_vs_control_btc: dict) -> dict:
    d_ok = res_vs_control_btc["d_sharpe"] >= SHARPE_NOISE_FLOOR
    ci_excludes_zero_positive = res_vs_control_btc["significant"] and res_vs_control_btc["paired_lo"] > 0
    return dict(d_ok=d_ok, ci_excludes_zero_positive=ci_excludes_zero_positive,
                passed=bool(d_ok and ci_excludes_zero_positive))


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def fmt_vs_v4(res: dict) -> str:
    return (f"sharpe cand={res['sharpe_cand']:.3f} v4={res['sharpe_v4']:.3f} "
            f"d={res['d_sharpe']:+.3f} | dd cand={res['dd_cand']:.1f}% v4={res['dd_v4']:.1f}% | "
            f"paired diff={res['paired_diff']:+.4f} CI=[{res['paired_lo']:+.4f},{res['paired_hi']:+.4f}] "
            f"sig={res['significant']} | tim cand={res['tim_cand']:.1f}% v4={res['tim_v4']:.1f}%")


def fmt_vs_ctrl(res: dict) -> str:
    return (f"sharpe cand={res['sharpe_cand']:.3f} ctrl={res['sharpe_ctrl']:.3f} "
            f"d={res['d_sharpe']:+.3f} | dd cand={res['dd_cand']:.1f}% ctrl={res['dd_ctrl']:.1f}% | "
            f"paired diff={res['paired_diff']:+.4f} CI=[{res['paired_lo']:+.4f},{res['paired_hi']:+.4f}] "
            f"sig={res['significant']} | tim cand={res['tim_cand']:.1f}% ctrl={res['tim_ctrl']:.1f}%")


def main() -> None:
    n_configs_evaluated = 0

    hr("R-185 NOVEL -- causal, dynamically-reweighted spot/Deribit-perp VOTE-level "
       "ensemble feeding kelly_regime_v4")
    print("See r185_shared.py's module docstring for direction/citations/gate. "
          "This file blends two already-computed\ndiscrete VOTES (frac_spot, "
          "frac_perp), never raw close prices -- see the module docstring's "
          "exact formula.")
    print(f"\nDynamic-arm grid (pre-registered, {len(GRID)} configs): "
          f"spans={SPAN_GRID_DAYS}, clips={CLIP_GRID}")
    print("Control arm: fixed 50/50, zero new tunable parameters, 1 config.")

    # ========================================================== STEP 1: causal probe
    hr("STEP 1 -- causal truncation probe")
    btc_df, _fs, _fp, _mask = _load("BTC")
    primary_span, primary_clip = 90.0, (0.2, 0.8)
    print(f"Primary config for the probe: span={primary_span}d, clip={primary_clip}")
    probe_dynamic = causal_truncation_probe(
        make_dynamic_factory(primary_span, primary_clip, "BTC"), btc_df)
    probe_control = causal_truncation_probe(make_control_factory("BTC"), btc_df)
    print(f"causal_truncation_probe (dynamic, primary config, BTC): "
          f"{'PASS' if probe_dynamic else 'FAIL'}")
    print(f"causal_truncation_probe (equal-weight control, BTC): "
          f"{'PASS' if probe_control else 'FAIL'}")
    all_causal_ok = probe_dynamic and probe_control
    if not all_causal_ok:
        hr("VERDICT")
        print("VERDICT: NEGATIVE (causal truncation probe FAILED). Stopping "
              "before reporting any promotion-relevant number.")
        return

    # ==================================================== STEP 2: dynamic-arm grid
    hr("STEP 2 -- dynamic-arm grid: BTC FUTURES_5x, inner-validation "
       f"({INNER_VAL_START} -> {INNER_VAL_END}), vs v4")
    grid_results: dict[tuple, dict] = {}
    for span, clip in GRID:
        res = signal_check(make_dynamic_factory(span, clip, "BTC"), btc_df, FUTURES,
                            INNER_VAL_START, INNER_VAL_END)
        n_configs_evaluated += 1
        grid_results[(span, clip)] = res
        print(f"\n  {label(span, clip):28s} {fmt_vs_v4(res)}")

    print(f"\n{'config':28s} {'d_sharpe':>9s} {'dd_cand':>8s} {'dd_v4':>7s} "
          f"{'paired_lo':>10s} {'paired_hi':>10s} {'sig':>5s} {'tim_d':>7s}")
    for (span, clip), res in grid_results.items():
        tim_d = res["tim_cand"] - res["tim_v4"]
        print(f"{label(span, clip):28s} {res['d_sharpe']:>+9.3f} {res['dd_cand']:>8.1f} "
              f"{res['dd_v4']:>7.1f} {res['paired_lo']:>+10.4f} {res['paired_hi']:>+10.4f} "
              f"{str(res['significant']):>5s} {tim_d:>+7.1f}")

    ranked = sorted(GRID, key=lambda cfg: grid_results[cfg]["d_sharpe"], reverse=True)
    winner = ranked[0]
    any_clause1 = [cfg for cfg in GRID if clause1(grid_results[cfg])]
    if any_clause1:
        winner = max(any_clause1, key=lambda cfg: grid_results[cfg]["d_sharpe"])
    win_span, win_clip = winner
    print(f"\nWinning config by primary-cell d_sharpe"
          f"{' (clause-1 qualifying)' if any_clause1 else ' (no config clears clause 1 -- best available)'}: "
          f"{label(*winner)}")

    # ============================================== STEP 3: winner's full battery vs v4
    hr(f"STEP 3 -- winner's full 4-cell battery ({label(*winner)}) vs v4, "
       f"inner-validation ({INNER_VAL_START} -> {INNER_VAL_END})")
    eth_df, _, _, _ = _load("ETH")
    dynamic_battery: dict[tuple[str, str], dict] = {("BTC", "futures_5x"): grid_results[winner]}
    for asset, df_asset in (("BTC", btc_df), ("ETH", eth_df)):
        for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
            if (asset, mkt_name) in dynamic_battery:
                continue  # BTC futures_5x already evaluated in the STEP 2 grid -- reused, not re-counted
            res = signal_check(make_dynamic_factory(win_span, win_clip, asset), df_asset, mkt,
                                INNER_VAL_START, INNER_VAL_END)
            n_configs_evaluated += 1
            dynamic_battery[(asset, mkt_name)] = res
            print(f"\n  [dynamic] {asset:3s} {mkt_name:10s} {fmt_vs_v4(res)}")
    print(f"\n  [dynamic] BTC futures_5x (reused from STEP 2 grid): "
          f"{fmt_vs_v4(dynamic_battery[('BTC', 'futures_5x')])}")

    # ============================================== STEP 4: control's full battery vs v4
    hr(f"STEP 4 -- equal-weight control's full 4-cell battery vs v4, "
       f"inner-validation ({INNER_VAL_START} -> {INNER_VAL_END})")
    control_battery: dict[tuple[str, str], dict] = {}
    for asset, df_asset in (("BTC", btc_df), ("ETH", eth_df)):
        for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
            res = signal_check(make_control_factory(asset), df_asset, mkt,
                                INNER_VAL_START, INNER_VAL_END)
            n_configs_evaluated += 1
            control_battery[(asset, mkt_name)] = res
            print(f"\n  [control] {asset:3s} {mkt_name:10s} {fmt_vs_v4(res)}")

    # ============================================== STEP 5: winner vs control, all 4 cells
    hr("STEP 5 -- winner vs equal-weight control (clause 4 requirement), "
       "all 4 cells")
    vs_control_battery: dict[tuple[str, str], dict] = {}
    for asset, df_asset in (("BTC", btc_df), ("ETH", eth_df)):
        for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
            res = signal_check_vs_control(
                make_dynamic_factory(win_span, win_clip, asset), make_control_factory(asset),
                df_asset, mkt, INNER_VAL_START, INNER_VAL_END)
            vs_control_battery[(asset, mkt_name)] = res
            print(f"\n  [dynamic vs control] {asset:3s} {mkt_name:10s} {fmt_vs_ctrl(res)}")
    print("\n(These 4 comparisons re-run the already-defined winner and control "
          "configs for a fresh paired statistic;\nno new hyperparameter setting "
          "is introduced, so they are NOT added to the configuration count.)")

    # ========================================================== STEP 6: side-by-side table
    hr("STEP 6 -- both arms, side by side, all cells (Sharpe, vs v4, vs control)")
    print(f"{'asset':5s} {'mkt':10s} {'sharpe_dyn':>10s} {'sharpe_ctrl':>11s} {'sharpe_v4':>9s} "
          f"{'d_dyn_v4':>9s} {'d_ctrl_v4':>9s} {'d_dyn_ctrl':>10s} {'dd_dyn':>7s} {'dd_ctrl':>7s} "
          f"{'tim_dyn':>8s} {'tim_ctrl':>8s} {'tim_v4':>7s}")
    for asset in ("BTC", "ETH"):
        for mkt_name in ("spot", "futures_5x"):
            dyn = dynamic_battery[(asset, mkt_name)]
            ctrl = control_battery[(asset, mkt_name)]
            vsc = vs_control_battery[(asset, mkt_name)]
            print(f"{asset:5s} {mkt_name:10s} {dyn['sharpe_cand']:>10.3f} {ctrl['sharpe_cand']:>11.3f} "
                  f"{dyn['sharpe_v4']:>9.3f} {dyn['d_sharpe']:>+9.3f} {ctrl['d_sharpe']:>+9.3f} "
                  f"{vsc['d_sharpe']:>+10.3f} {dyn['dd_cand']:>7.1f} {ctrl['dd_cand']:>7.1f} "
                  f"{dyn['tim_cand']:>8.1f} {ctrl['tim_cand']:>8.1f} {dyn['tim_v4']:>7.1f}")

    # ========================================================== STEP 7: gate verdict
    hr("STEP 7 -- frozen inner-validation gate verdict")
    btc_fut_v4 = dynamic_battery[("BTC", "futures_5x")]
    eth_fut_v4 = dynamic_battery[("ETH", "futures_5x")]
    verdict_v4_fut = gate_verdict_vs_v4(btc_fut_v4, eth_fut_v4)
    btc_spot_v4 = dynamic_battery[("BTC", "spot")]
    eth_spot_v4 = dynamic_battery[("ETH", "spot")]
    verdict_v4_spot = gate_verdict_vs_v4(btc_spot_v4, eth_spot_v4)

    for name, verdict in (("futures_5x pair", verdict_v4_fut), ("spot pair", verdict_v4_spot)):
        print(f"\n  [vs v4, {name}]")
        print(f"    Clause 1 (d_sharpe>=+0.20 OR matching-magnitude DD improvement, "
              f"same direction both markets): {verdict['clause1']}  "
              f"(BTC={verdict['c1_btc']}, ETH={verdict['c1_eth']})")
        print(f"    Clause 2 (paired 95% CI excludes zero on >=1 market, not losing "
              f"on the other): {verdict['clause2']}  "
              f"(BTC excludes-zero-positive={verdict['excludes_zero_btc']}, "
              f"ETH excludes-zero-positive={verdict['excludes_zero_eth']})")
        print(f"    Clause 3 (exposure matched within {EXPOSURE_TOL_PP}pp both markets): "
              f"{verdict['clause3']}")
        print(f"    Standard gate (clauses 1-3) PASSED: {verdict['passed']}")

    vsc_btc_fut = vs_control_battery[("BTC", "futures_5x")]
    vsc_btc_spot = vs_control_battery[("BTC", "spot")]
    verdict_c4_fut = clause4_vs_control(vsc_btc_fut)
    verdict_c4_spot = clause4_vs_control(vsc_btc_spot)
    print("\n  [clause 4, dynamic vs equal-weight control, BTC]")
    print(f"    futures_5x: d_sharpe={vsc_btc_fut['d_sharpe']:+.3f} (>= +0.20: "
          f"{verdict_c4_fut['d_ok']}), CI=[{vsc_btc_fut['paired_lo']:+.4f},"
          f"{vsc_btc_fut['paired_hi']:+.4f}] excludes-zero-positive="
          f"{verdict_c4_fut['ci_excludes_zero_positive']}  -> PASSED={verdict_c4_fut['passed']}")
    print(f"    spot (reported alongside, not gating): d_sharpe={vsc_btc_spot['d_sharpe']:+.3f}, "
          f"CI=[{vsc_btc_spot['paired_lo']:+.4f},{vsc_btc_spot['paired_hi']:+.4f}] "
          f"excludes-zero-positive={verdict_c4_spot['ci_excludes_zero_positive']}")

    overall_passed = verdict_v4_fut["passed"] and verdict_c4_fut["passed"]

    hr("VERDICT")
    print(f"Causal truncation probes: {'PASS' if all_causal_ok else 'FAIL'}")
    print(f"Winning config: {label(*winner)}")
    print(f"Gate vs v4 (futures_5x pair, primary): {'PASS' if verdict_v4_fut['passed'] else 'FAIL'}")
    print(f"Gate vs v4 (spot pair): {'PASS' if verdict_v4_spot['passed'] else 'FAIL'}")
    print(f"Gate clause 4 (vs equal-weight control, BTC futures_5x): "
          f"{'PASS' if verdict_c4_fut['passed'] else 'FAIL'}")
    print(f"\nOVERALL (standard gate AND clause 4, futures_5x pair): "
          f"{'PASS' if overall_passed else 'NEGATIVE'}")
    print(f"\nTotal configuration evaluations counted: {n_configs_evaluated} "
          f"(grid={len(GRID)} on BTC futures_5x + dynamic-arm remaining battery cells=3 "
          f"+ control-arm full battery=4)")
    print(f"\nHoldout (>= 2023-01-01) consulted: NO. `_load` hard-truncates every series at "
          f"INNER_VAL_END={INNER_VAL_END}\nbefore anything else runs.")


if __name__ == "__main__":
    main()
