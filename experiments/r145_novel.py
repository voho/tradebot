"""R-145 NOVEL branch: funding-adaptive spot/futures migration threshold for
`kelly_regime_v4`'s own, unmodified `target`.

**Relationship to the shared pre-registration.** `experiments/r145_shared.py`
(frozen, not modified here) freezes the harness (`HybridBroker`,
`run_hybrid_backtest`), the conservative mechanism (`route_fixed_threshold`,
a FIXED spot/futures split point at `threshold=1.0`), and both branches'
decision rules. This file is the novel branch's *own* contribution: instead
of a fixed threshold, the migration point itself is a causal function of
trailing BTC funding, built the same way `route_fixed_threshold` is (a
precomputed `RouteFn` derived only from `target` and
`r145_shared.trailing_funding_ewm`), so it inherits that helper's causality
(shift-by-1-bar) and its ETH degeneracy (funding=None -> all-zero EWM ->
threshold==1.0 everywhere -> identical to `route_fixed_threshold(1.0)`)
without any special-casing.

**Mechanism (one sentence).** When BTC's trailing funding is negative
(shorts are paying longs to hold futures), lower the spot/futures migration
threshold continuously from 1.0 toward 0.0 as trailing funding gets more
negative -- shifting progressively more of `target` (the SAME causal
directional exposure conservative already computes; total notional and beta
are unchanged, only the financing venue moves) out of the funding-free spot
leg and into the funding-EARNING futures leg -- and revert the threshold to
1.0 the instant trailing funding is non-negative, so the mechanism is inert
except during an actual rebate window.

**Why this should make money (mechanism, expanded).** Ackerer, Hugonnier &
Jermann (2024/2025 working paper) treat venue choice for a fixed directional
target as a pure financing decision -- an arbitrage-free perpetual is a
continuously-refinanced spot replication, with funding as the financing
leg. Conservative's fixed threshold=1.0 spends that financing choice only on
the `>1.0x` excess (rare for v4: mean historical notional 0.18-0.38x,
R-57/R-62). During a NEGATIVE-funding window the "cheap financing" venue is
futures, not spot, for the WHOLE position, not just the excess -- so a
route that only ever reroutes the >1.0x excess is leaving most of the
available rebate on the table exactly when it is most available. Lowering
the threshold captures more of it, still without changing `target` itself,
still without taking any new directional or basis bet (see guardrail
below).

**Guardrail, taken seriously.** Schmeling, Schrimpf & Todorov (2023, BIS
Working Paper 1087, "Crypto Carry" -- https://www.bis.org/publ/work1087.pdf)
measure the crypto futures carry/funding premium directly and find it large
and volatile, decaying and sometimes negative in parts of 2024-2025. Per
this round's shared pre-registration, that citation is a guardrail, not a
license: this design may only ever AVOID financing cost already being paid
on `target`'s unchanged, already-existing directional exposure, never take
on a NEW, zero-net-exposure carry bet -- that would duplicate B-03
(delta-neutral spot-long/futures-short harvest, NEGATIVE, R-39/R-144-era
ledger). This design satisfies that constraint by construction: `spot_frac
+ fut_frac == target` at every bar, identically to conservative's route --
only the SPLIT point moves, never the sum. Total directional BTC exposure
(hence total price risk) is bar-for-bar identical to conservative's; only
which venue finances it changes.

**Literature search on funding persistence/episode structure (done before
choosing the EWM span, per ROUTINE.md step 2) -- reported honestly,
including its limits:**

- Direct academic evidence on the AR/half-life of the funding rate itself
  was not found. Multiple 2025-2026 papers turned up in search (Zhang,
  SSRN 6185958, "Funding Rate Mechanism in Perpetual Futures"; the
  "Designing funding rates..." paper, arXiv:2506.08573; an MDPI
  "Two-Tiered Structure" paper, 2227-7390/14/2/346) discuss funding-rate
  *design* (how exchanges compute it) and market-microstructure dynamics,
  not a measured autocorrelation/half-life number for the realized rate.
  Beyond the Schmeling-Schrimpf-Todorov (2023) citation r145_shared.py
  already carries, no paper found here adds a quantified persistence
  statistic -- stated plainly rather than padded, per this round's
  instructions.
- What IS available is qualitative/market-record evidence that negative
  BTC funding comes in multi-week regimes, not scattered single-settlement
  noise: reporting on the Nov-2022 FTX collapse describes BTC funding
  running negative for roughly 50 consecutive days before shorts
  capitulated (bottoming near $15.5k, recovering by late January 2023);
  the same source and CoinDesk (2026-04-16 "Bitcoin funding rates turn
  most negative since 2023") both describe deeply negative funding as a
  regime-scale phenomenon that has historically coincided with cyclical
  bottoms (Mar 2020, mid-2021, Nov 2022), not a single-bar event. This
  motivates using a multi-DAY EWM span (this file's `SPAN_DAYS`), not an
  intra-day one, and choosing the primary span (7d) to sit inside that
  observed multi-week regime length rather than reacting to individual
  8h settlements.
- This file's OWN measurement (below, done on the causal `trailing_funding_
  ewm` helper, inner-train BTC 2020 -- the only inner-train year funding
  data actually covers, since `load_funding_extended` starts 2020-01-01)
  independently confirms the regime-not-noise picture: `neg_frac` (share
  of bars with negative trailing EWM) is 0.13-0.15 across every span tried,
  changing only slowly with span, which is what a slowly-decaying process
  looks like, not white noise (white noise would show `neg_frac` closer to
  0.5 and a min/p1 gap collapsing much faster as span grows). See
  `_calibration_report()` below for the exact numbers.

**Calibration discipline (fit on inner-train, evaluate on
inner-validation, never the reverse).** The threshold ramp needs a scale --
"how negative is negative" -- and that scale is fit ONLY on inner-train
data (2020-01-01 -> 2020-12-31, the overlap of `INNER_TRAIN_END` and BTC's
funding history) and then frozen; inner-validation (2021-2022) is read only
to SELECT between the resulting configs, never to re-tune the scale itself.
`REF_NEG_ABS[span]` below is the 1st percentile magnitude of
`trailing_funding_ewm` on that inner-train year, per span -- the ramp
reaches its floor (threshold=0, fully into futures) at that empirically
extreme-but-observed level, and only there.

**Functional form.**
    threshold(ewm) = clip(1.0 + ewm / REF_NEG_ABS[span], 0.0, 1.0)
`ewm >= 0` (funding non-negative) always gives `threshold == 1.0`,
identical to conservative. `ewm < 0` ramps threshold linearly down to 0.0
(fully financed through futures) once `ewm <= -REF_NEG_ABS[span]`. The ramp
is continuous (no on/off switch) so it cannot manufacture large extra
turnover from small funding wiggles near zero -- a discrete threshold flip
would.

**3 configs (own contribution, declared before running):**
1. primary: `span_days=7` (weekly -- literature-grounded middle of the
   multi-week regime length above).
2. robustness A: `span_days=3` (faster/noisier -- reacts within days).
3. robustness B: `span_days=14` (slower/smoother -- reacts over ~2 weeks).
Each span's own `REF_NEG_ABS` is calibrated separately on the same
inner-train year (see table below), so span is the only thing that varies
between configs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from tradebot import inference as inf
from tradebot.broker import MarketSpec

from r145_shared import (
    FUT_FEE,
    INNER_VAL_END,
    INNER_VAL_START,
    NOVEL_D_SHARPE_FLOOR,
    NOVEL_RATIO_KILL,
    D_SHARPE_FLOOR,
    TURNOVER_SAVINGS_KILL,
    EXPOSURE_MATCH_TOL_PCT,
    RouteFn,
    compute_target,
    fut_market,
    load_btc,
    load_eth,
    route_fixed_threshold,
    run_hybrid_backtest,
    spot_market,
    trailing_funding_ewm,
)

# ----------------------------------------------------------- frozen design

# Calibrated on BTC inner-train 2020-01-01 -> 2020-12-31 ONLY (the overlap
# of INNER_TRAIN_END and real funding history -- load_funding_extended
# starts 2020-01-01). 1st-percentile magnitude of trailing_funding_ewm(...),
# per span. See `_calibration_report()` to reproduce.
REF_NEG_ABS: dict[float, float] = {
    3.0: 4.616290161271371e-06,
    7.0: 3.0573200378560226e-06,
    14.0: 1.6638436658537818e-06,
}

SPAN_PRIMARY = 7.0
SPAN_ROBUST_A = 3.0
SPAN_ROBUST_B = 14.0

CONFIGS = (SPAN_PRIMARY, SPAN_ROBUST_A, SPAN_ROBUST_B)   # 3 total, per pre-registration


def route_funding_adaptive(funding: pd.Series | None, span_days: float,
                           ref_neg_abs: float) -> Callable[[pd.DataFrame], RouteFn]:
    """The novel mechanism: same shape as `route_fixed_threshold`, but the
    split point is a per-bar array, not a scalar.

    ``threshold[i] = clip(1.0 + ewm[i] / ref_neg_abs, 0.0, 1.0)``
    ``spot_frac[i] = min(target[i], threshold[i])``
    ``fut_frac[i]  = target[i] - spot_frac[i]``  (always >= 0)

    Built only from `target` (v4's own causal signal, reused unmodified via
    `compute_target`) and `trailing_funding_ewm` (already causal -- shift-
    by-1-bar built into that helper) -- no other runtime state, matching
    `route_fixed_threshold`'s own construction and inheriting its ETH
    degeneracy: `funding=None` -> `trailing_funding_ewm` returns an
    all-zero array -> `threshold` is 1.0 at every bar -> this reduces to
    `route_fixed_threshold(1.0)` exactly, verified numerically below.
    """
    def build(frame: pd.DataFrame) -> RouteFn:
        target = compute_target(frame)
        ewm = trailing_funding_ewm(funding, frame.index, span_days)
        threshold = np.clip(1.0 + ewm / ref_neg_abs, 0.0, 1.0)
        spot_frac = np.minimum(target, threshold)
        spot_frac = np.clip(spot_frac, 0.0, None)
        fut_frac = target - spot_frac
        return RouteFn(spot_frac, fut_frac)
    return build


# ------------------------------------------------------------ calibration

def _calibration_report(df: pd.DataFrame, funding: pd.Series) -> None:
    """Reproduces REF_NEG_ABS and prints the inner-train persistence
    diagnostics this file's docstring cites, on the calibration slice only
    (2020, the sole inner-train year with real funding data).
    """
    lo = int(df.index.searchsorted("2020-01-01"))
    hi = int(df.index.searchsorted("2020-12-31", side="right"))
    frame = df.iloc[lo:hi]
    print(f"[calibration] inner-train funding-calibration slice: {frame.index[0]} -> "
          f"{frame.index[-1]} ({len(frame):,} bars)")
    for span in (SPAN_ROBUST_A, SPAN_PRIMARY, SPAN_ROBUST_B):
        ewm = trailing_funding_ewm(funding, frame.index, span)
        p1 = float(np.percentile(ewm, 1))
        neg_frac = float((ewm < 0).mean())
        print(f"  span={span:>5.1f}d: min={ewm.min(): .3e}  p1={p1: .3e}  "
              f"neg_frac={neg_frac:.3f}  (frozen REF_NEG_ABS={REF_NEG_ABS[span]:.3e})")


# --------------------------------------------------------------- ETH check

def eth_degeneracy_check() -> None:
    """Primary-config novel route on ETH (funding=None) must equal
    `route_fixed_threshold(1.0)`'s route to floating tolerance -- this
    branch's version of conservative's own mechanical ETH check.
    """
    eth_df, eth_funding, eth_label = load_eth()
    lo = int(eth_df.index.searchsorted(INNER_VAL_START))
    hi = int(eth_df.index.searchsorted(INNER_VAL_END, side="right"))
    frame = eth_df.iloc[lo:hi]

    novel_route = route_funding_adaptive(eth_funding, SPAN_PRIMARY,
                                          REF_NEG_ABS[SPAN_PRIMARY])(frame)
    fixed_route = route_fixed_threshold(1.0)(compute_target(frame))

    spot_diff = float(np.abs(novel_route.spot_frac - fixed_route.spot_frac).max())
    fut_diff = float(np.abs(novel_route.fut_frac - fixed_route.fut_frac).max())
    print(f"[ETH degeneracy] max|spot_frac diff|={spot_diff:.3e}  "
          f"max|fut_frac diff|={fut_diff:.3e}  ({eth_label})")
    assert spot_diff < 1e-12, f"novel route diverges from fixed(1.0) on ETH: {spot_diff:.3e}"
    assert fut_diff < 1e-12, f"novel route diverges from fixed(1.0) on ETH: {fut_diff:.3e}"


# --------------------------------------------------------------- reporting

def _exposure_stats(equity: pd.Series, fills: int) -> dict:
    rets = equity.pct_change().dropna()
    return {
        "time_in_market": float((equity.diff().fillna(0.0) != 0.0).mean()),
        "ann_vol": float(rets.std() * np.sqrt(365.0 * 288.0)) if len(rets) else float("nan"),
        "fills": fills,
    }


def run_all() -> None:
    btc, btc_funding, btc_label = load_btc()
    print(f"BTC: {len(btc):,} bars ({btc_label}), funding present: {btc_funding is not None}\n")

    print("=== calibration (inner-train 2020, funding-covered slice) ===")
    _calibration_report(btc, btc_funding)

    print("\n=== ETH degeneracy check (primary config, funding=None) ===")
    eth_degeneracy_check()

    fee_tiers = {"base (0.10% spot)": 0.001, "real (0.40% spot)": 0.004}

    print("\n=== conservative reproduction + novel primary/robustness, BTC inner-validation ===")
    results: dict[str, dict[str, object]] = {}
    for tier_name, spot_fee in fee_tiers.items():
        spot_mkt = spot_market(spot_fee)
        fut_mkt = fut_market(FUT_FEE)

        def conservative_builder(frame: pd.DataFrame) -> RouteFn:
            return route_fixed_threshold(1.0)(compute_target(frame))

        conservative = run_hybrid_backtest(
            btc, conservative_builder, spot_mkt, fut_mkt,
            funding=btc_funding, start=INNER_VAL_START, end=INNER_VAL_END)

        results[tier_name] = {"conservative": conservative}

        for span in CONFIGS:
            novel = run_hybrid_backtest(
                btc, route_funding_adaptive(btc_funding, span, REF_NEG_ABS[span]),
                spot_mkt, fut_mkt, funding=btc_funding,
                start=INNER_VAL_START, end=INNER_VAL_END)
            results[tier_name][f"novel_span{span:g}"] = novel

    # ---- print raw numbers
    for tier_name, arms in results.items():
        print(f"\n--- fee tier: {tier_name} ---")
        cons = arms["conservative"]
        cons_stats = _exposure_stats(cons.equity, cons.fills_spot + cons.fills_fut)
        print(f"  conservative: final=${cons.final_balance:,.2f} fees=${cons.fees_paid:,.2f} "
              f"funding_paid=${cons.funding_paid:,.2f} fills(spot/fut)={cons.fills_spot}/"
              f"{cons.fills_fut} time_in_mkt={cons_stats['time_in_market']:.4f} "
              f"ann_vol={cons_stats['ann_vol']:.4f} liq={cons.liquidated}")
        for span in CONFIGS:
            nov = arms[f"novel_span{span:g}"]
            nov_stats = _exposure_stats(nov.equity, nov.fills_spot + nov.fills_fut)
            rebate = cons.funding_paid - nov.funding_paid
            extra_fees = nov.fees_paid - cons.fees_paid
            ratio = rebate / extra_fees if extra_fees > 0 else float("inf") if rebate > 0 else float("nan")
            print(f"  novel(span={span:g}d): final=${nov.final_balance:,.2f} "
                  f"fees=${nov.fees_paid:,.2f} funding_paid=${nov.funding_paid:,.2f} "
                  f"fills(spot/fut)={nov.fills_spot}/{nov.fills_fut} "
                  f"time_in_mkt={nov_stats['time_in_market']:.4f} ann_vol={nov_stats['ann_vol']:.4f} "
                  f"liq={nov.liquidated} | rebate=${rebate:,.4f} extra_fees=${extra_fees:,.4f} "
                  f"ratio={ratio:.3f}")

    # ---- d_sharpe: novel primary vs conservative, both fee tiers
    print("\n=== d_sharpe (novel primary span=7d vs conservative), paired bootstrap ===")
    for tier_name, arms in results.items():
        cons = arms["conservative"]
        nov = arms[f"novel_span{SPAN_PRIMARY:g}"]
        cons_daily = inf.daily_returns(cons.equity).to_numpy()
        nov_daily = inf.daily_returns(nov.equity).to_numpy()
        n = min(len(cons_daily), len(nov_daily))
        res = inf.paired_bootstrap(nov_daily[:n], cons_daily[:n], inf.annualized_sharpe)
        print(f"  {tier_name}: d_sharpe point={res.diff.point:+.4f} "
              f"[{res.diff.lo:+.4f}, {res.diff.hi:+.4f}] p_positive={res.p_positive:.3f} "
              f"significant={res.significant} (n_days matched={n})")

    # ---- d_sharpe for robustness spans too, base fee tier only
    print("\n=== d_sharpe, robustness spans, base fee tier only ===")
    base = results["base (0.10% spot)"]
    cons_daily = inf.daily_returns(base["conservative"].equity).to_numpy()
    for span in (SPAN_ROBUST_A, SPAN_ROBUST_B):
        nov = base[f"novel_span{span:g}"]
        nov_daily = inf.daily_returns(nov.equity).to_numpy()
        n = min(len(cons_daily), len(nov_daily))
        res = inf.paired_bootstrap(nov_daily[:n], cons_daily[:n], inf.annualized_sharpe)
        print(f"  span={span:g}d: d_sharpe point={res.diff.point:+.4f} "
              f"[{res.diff.lo:+.4f}, {res.diff.hi:+.4f}] p_positive={res.p_positive:.3f} "
              f"significant={res.significant}")

    # ---- negative-funding-time diagnostic + mechanism-triggered bars
    print("\n=== negative-funding-time diagnostic, BTC inner-validation ===")
    lo = int(btc.index.searchsorted(INNER_VAL_START))
    hi = int(btc.index.searchsorted(INNER_VAL_END, side="right"))
    frame = btc.iloc[lo:hi]
    fixed_route = route_fixed_threshold(1.0)(compute_target(frame))
    for span in CONFIGS:
        ewm = trailing_funding_ewm(btc_funding, frame.index, span)
        neg_bars = int((ewm < 0).sum())
        neg_days = neg_bars / 288.0
        novel_route = route_funding_adaptive(btc_funding, span, REF_NEG_ABS[span])(frame)
        deviated = int((np.abs(novel_route.fut_frac - fixed_route.fut_frac) > 1e-9).sum())
        print(f"  span={span:g}d: bars with ewm<0 = {neg_bars:,} ({neg_days:.1f} days of "
              f"{len(frame)/288.0:.0f}); bars where novel route deviates from fixed(1.0) = "
              f"{deviated:,} ({deviated/288.0:.1f} days)")

    print("\nDone.")


if __name__ == "__main__":
    run_all()
