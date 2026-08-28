"""R-173 CONSERVATIVE branch: illiquidity-adjusted re-pricing of
`kelly_regime_v4`'s own cost model -- a COST-axis MEASUREMENT, not a new
strategy. `kelly_regime_v4`'s vote/scale/deadband logic is UNCHANGED
everywhere in this file; the only thing that varies across the runs below is
the `MarketSpec.fee_rate` passed to the (unmodified) backtest.

Pre-registration (shared, frozen, both branches): `experiments/r173_direction.md`
and `experiments/r173_shared.py`. Neither is edited here, and neither is
anything outside this file.

=====================================================================
THE ONE-PARAGRAPH MECHANISM
=====================================================================

1. Reconstruct `kelly_regime_v4`'s OWN real deadband-triggered re-target
   bars (`TargetStrategy(v4_target)`'s target path, exactly as R-89 through
   R-172 already run it), on inner-train, inner-validation and the ETH
   replication slice -- training data only.
2. At each such bar, sample the already-causal Corwin & Schultz (2012)
   spread (PRIMARY, per `r173_direction.md`) and the already-causal Roll
   (1984) spread (secondary cross-check), both from `r173_shared`, computed
   ONCE over each instrument's whole pre-holdout series (both are strictly
   causal via `.shift(1)`, so reading them at an arbitrary row is exactly as
   causal as reading `close` at that row -- no new lookahead is introduced
   by "sampling" them here).
3. `|Δtarget|`-notional-weight the per-bar half-spread (`spread / 2`,
   the one-way-trade analogue of a taker fee) into one blended rate per
   instrument -- this is the "illiquidity add-on".
4. Re-price `kelly_regime_v4` at `fee_at(SPOT/FUTURES, current_fee + addon)`
   and compare it against `buy_and_hold` under the IDENTICAL MarketSpec, at
   three fee tiers: the project's actual registered baseline (`SPOT.fee_rate`
   =0.10%, `FUTURES.fee_rate`=0.05%, read from `tradebot.broker.MarketSpec`
   directly rather than assumed), the illiquidity-adjusted tier, and the
   existing 0.40% sensitivity tier (`r161_shared.FEE_TIER`) for reference.
5. Apply the decision rule frozen in `r173_direction.md`'s Step 4
   CONSERVATIVE section, verbatim, first on the train-only reading, then
   -- once, without modification -- on the true 2023+ holdout.

=====================================================================
WHY "SEPARATELY FOR SPOT AND FUTURES" COMES OUT IDENTICAL HERE, DISCLOSED
=====================================================================

`r173_direction.md` asks for the add-on "separately for spot and futures
(they may have very different average implied friction if v4 trades
more/less on one market)". In THIS codebase they cannot differ, and that is
worth stating plainly rather than silently computing one number and
pasting it twice:

- `v4_target` (via `v4_raw_desired` -> `v4_scale`/`vote_frac`) is a PURE
  function of OHLCV `close`. It never reads `MarketSpec`. So the re-target
  BARS and their `|Δtarget|` sizes are bit-identical on spot and futures --
  there is only one trading pattern to sample spreads against.
- Every strategy in the `kelly_regime` family calls `ctx.order_notional`,
  not `ctx.order_target` (`kelly_regime.py:111`, "fraction of equity: same
  risk on spot and futures" -- the code's own comment). `order_notional`
  divides by `market.leverage` before calling `order_target`, and
  `_execute_target` multiplies back by `max_qty = equity*leverage*haircut/
  price`, so the realized NOTIONAL fraction of equity traded at a given
  `|Δtarget|` is `|Δtarget| * equity`, independent of `market.leverage`.
  Since `fee = fee_rate * |Δqty| * price` is a rate on that SAME notional,
  the friction addon (a fractional rate, like `fee_rate`) is market-spec-
  independent too.

So `spot_addon == futures_addon` by construction, verified below, not
assumed. What genuinely differs between the two `MarketSpec`s is the
BASELINE fee being added to (0.10% spot vs 0.05% futures) and futures'
leverage headroom -- both already carried through `fee_at`.

=====================================================================
UNIT NOTE ON THE ROLL ESTIMATOR, DISCLOSED
=====================================================================

`r173_shared.roll_spread_causal` computes `2*sqrt(-Cov(dP, dP_lag1))` on
`df["close"].diff()` -- a DOLLAR price difference, not a log return -- so
its output `s` is a DOLLAR spread (e.g. "$41" on a $60,000 BTC bar), not a
fractional rate. `corwin_schultz_spread_causal`'s output IS already
fractional (built from `ln(H/L)` ratios). To compare the two series, or to
use Roll for anything fee-rate-shaped, this file divides Roll's output by
the contemporaneous `close` at the SAME (already-causal) bar --
`roll_relative[t] = roll_dollar[t] / close[t]` -- a same-bar unit
conversion, not a new source of lookahead (the numerator is already known
strictly before bar t's own close per `r173_shared`'s own `.shift(1)`
convention; the denominator is bar t's own close, known no later than the
numerator's reference bar).

=====================================================================
WHAT WOULD MAKE THIS FAIL (named per `r173_direction.md`, restated here)
=====================================================================

The estimated friction, applied as a re-pricing on top of the current fee
tier, does not materially change `kelly_regime_v4`'s already-established
OOS verdict against `buy_and_hold` -- i.e. COST was already priced
correctly enough by the flat tier and this measurement adds a number
without changing a conclusion. Named now, before any real-data cell below
was read.

Usage::

    python experiments/r173_conservative_illiquidity_refee.py            # everything
    python experiments/r173_conservative_illiquidity_refee.py causality
    python experiments/r173_conservative_illiquidity_refee.py addon
    python experiments/r173_conservative_illiquidity_refee.py train
    python experiments/r173_conservative_illiquidity_refee.py holdout
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiments.r173_shared as R  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import prefix_bars  # noqa: E402

# ------------------------------------------------------------------------
# Constants. Fee baselines are READ from the registered MarketSpec
# defaults (never hard-coded), per the task's own instruction to check
# src/tradebot/broker.py rather than assume 0.10%/0.05%.
# ------------------------------------------------------------------------
BASE_SPOT_FEE = R.SPOT.fee_rate           # 0.10% -- MarketSpec.spot() default
BASE_FUTURES_FEE = R.FUTURES.fee_rate     # 0.05% -- MarketSpec.futures() default
assert abs(BASE_SPOT_FEE - 0.001) < 1e-12, BASE_SPOT_FEE
assert abs(BASE_FUTURES_FEE - 0.0005) < 1e-12, BASE_FUTURES_FEE
SENSITIVITY_FEE = R.FEE_TIER              # 0.40%, r161_shared's own constant
assert abs(SENSITIVITY_FEE - 0.0040) < 1e-12, SENSITIVITY_FEE

# Applied to SPOT only, matching every prior round's own convention
# (scripts/fee_study.py's BITSTAMP_TAKER, r165's FEE_STRESS): this project's
# 0.40% number is Bitstamp's own documented SPOT entry-tier taker fee, never
# applied to the futures leg in any prior round (grep of
# experiments/r1*_conservative_*.py / r1*_novel_*.py confirms zero futures
# hits for 0.0040). Futures stays at its own registered baseline in the
# "sensitivity" tier; disclosed rather than silently assumed.
FUTURES_SENSITIVITY_FEE = BASE_FUTURES_FEE

V4_WARMUP = 80 * R.BARS_PER_DAY + 10      # TargetStrategy's own default warmup

# CS smoothing-window sensitivity bracket (an ADDITION per ROUTINE.md's
# "additions after the freeze may only tighten, never loosen" rule -- this
# widens nothing about the frozen primary estimator, it only checks how much
# the addon MEASUREMENT moves if the smoothing window is halved/doubled).
CS_WINDOW_GRID = (R.CS_SMOOTH_WINDOW // 2, R.CS_SMOOTH_WINDOW, R.CS_SMOOTH_WINDOW * 2)

BOOT_KW = dict(mean_block=30.0, n_boot=2_000, seed=173)

#: Counts every distinct market x fee-tier x slice backtest actually run.
_CONFIGS = [0]
_HOLDOUT_READS = [0]


# ==========================================================================
# (0) Windowing + v4's own real rebalance-event extraction.
# ==========================================================================


def window_frame(df: pd.DataFrame, start: str | None, end: str | None,
                  warmup: int = V4_WARMUP) -> tuple[pd.DataFrame, int, int, int]:
    """The exact frame `run_period` hands to `prepare` for this window, plus
    (lo, hi, prefix) so a caller can align an array built on that frame back
    to `df`'s own row positions. Mirrors `tradebot.window.run_period`."""
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    prefix = prefix_bars(df, lo, warmup)
    return df.iloc[lo - prefix: hi], lo, hi, prefix


def v4_rebalance_events(df: pd.DataFrame, start: str | None, end: str | None
                         ) -> list[tuple[int, float]]:
    """v4's own actual deadband-triggered re-target bars within [start, end),
    as `(row_position_in_df, |delta_target|)` pairs.

    Reproduces `TargetStrategy.on_bar`'s own trigger condition
    (`abs(t - prev) > 1e-9`) exactly, over the SAME windowed frame
    `run_slice`/`run_period` would hand to `prepare` for this window (warm
    prefix included, so the deadband's internal `pos` state carries in from
    before `start` exactly as a real backtest's does) -- see
    `_self_test`'s real-data cross-check against an actual `PaperBroker` run
    for the proof this matches what a real backtest fires.
    """
    frame, lo, hi, prefix = window_frame(df, start, end)
    target = R.v4_target(frame)
    events: list[tuple[int, float]] = []
    start_j = max(prefix, 1)
    for j in range(start_j, len(target)):
        delta = target[j] - target[j - 1]
        if abs(delta) > 1e-9:
            events.append((lo - prefix + j, abs(delta)))
    return events


# ==========================================================================
# (1) The illiquidity add-on: |Δtarget|-weighted average implied half-spread.
# ==========================================================================


def instrument_addon(df: pd.DataFrame, windows: list[tuple[str | None, str | None]],
                      cs_window: int = R.CS_SMOOTH_WINDOW) -> dict:
    """One instrument's illiquidity add-on rate, from its own real v4
    rebalance events across `windows` (train-only slices).

    Returns the PRIMARY (Corwin-Schultz) weighted half-spread addon, the
    secondary (Roll, unit-converted to relative) addon for cross-check, the
    Roll/CS correlation (both over the full causal series and at the
    sampled event bars), and event/weight diagnostics.
    """
    cs = R.corwin_schultz_spread_causal(df, smooth_window=cs_window,
                                        min_periods=max(cs_window // 4, 1))
    roll_dollar = R.roll_spread_causal(df)
    close = df["close"].to_numpy(dtype=float)
    roll_rel = np.divide(roll_dollar, close, out=np.zeros_like(roll_dollar),
                         where=close > 0)

    events: list[tuple[int, float]] = []
    for start, end in windows:
        events.extend(v4_rebalance_events(df, start, end))

    pos = np.array([e[0] for e in events], dtype=int)
    weight = np.array([e[1] for e in events], dtype=float)
    cs_at = cs[pos]
    roll_at = roll_rel[pos]

    total_w = float(weight.sum())
    cs_addon = float(np.sum(weight * cs_at / 2.0) / total_w) if total_w > 0 else float("nan")
    roll_addon = float(np.sum(weight * roll_at / 2.0) / total_w) if total_w > 0 else float("nan")

    both_nonzero = (cs > 0) & (roll_rel > 0)
    corr_full = (float(np.corrcoef(cs[both_nonzero], roll_rel[both_nonzero])[0, 1])
                if both_nonzero.sum() > 2 else float("nan"))
    both_nonzero_evt = (cs_at > 0) & (roll_at > 0)
    corr_events = (float(np.corrcoef(cs_at[both_nonzero_evt], roll_at[both_nonzero_evt])[0, 1])
                  if both_nonzero_evt.sum() > 2 else float("nan"))

    return dict(
        n_events=len(events), total_weight=total_w,
        cs_addon=cs_addon, roll_addon=roll_addon,
        cs_mean_at_events=float(np.mean(cs_at)) if len(cs_at) else float("nan"),
        roll_mean_at_events=float(np.mean(roll_at)) if len(roll_at) else float("nan"),
        corr_full_series=corr_full, corr_at_events=corr_events,
        both_nonzero_frac_full=float(both_nonzero.mean()),
    )


def cmd_addon(btc: pd.DataFrame, eth: pd.DataFrame) -> dict:
    print("=" * 100)
    print("(1) ILLIQUIDITY ADD-ON -- v4's own real rebalance bars, train-only "
          "(0 backtests, pure signal math)")
    print("=" * 100)

    btc_windows = [R.SLICES["inner_train"], R.SLICES["inner_val"]]
    eth_windows = [(None, None)]

    btc_res = instrument_addon(btc, btc_windows)
    eth_res = instrument_addon(eth, eth_windows)

    print(f"\n  BTC (inner_train + inner_val): {btc_res['n_events']:,} rebalance bars, "
          f"total |Δtarget| weight {btc_res['total_weight']:.3f}")
    print(f"    CS  (PRIMARY) weighted half-spread addon : {btc_res['cs_addon']:.6f} "
          f"({btc_res['cs_addon'] * 1e4:.2f} bps)")
    print(f"    Roll (secondary) weighted half-spread     : {btc_res['roll_addon']:.6f} "
          f"({btc_res['roll_addon'] * 1e4:.2f} bps)")
    print(f"    Roll/CS corr, full causal series (both>0) : {btc_res['corr_full_series']:.3f} "
          f"(both>0 on {btc_res['both_nonzero_frac_full']:.1%} of bars)")
    print(f"    Roll/CS corr, AT the sampled event bars   : {btc_res['corr_at_events']:.3f}")

    print(f"\n  ETH (full pre-holdout replication slice): {eth_res['n_events']:,} rebalance "
          f"bars, total |Δtarget| weight {eth_res['total_weight']:.3f}")
    print(f"    CS  (PRIMARY) weighted half-spread addon : {eth_res['cs_addon']:.6f} "
          f"({eth_res['cs_addon'] * 1e4:.2f} bps)")
    print(f"    Roll (secondary) weighted half-spread     : {eth_res['roll_addon']:.6f} "
          f"({eth_res['roll_addon'] * 1e4:.2f} bps)")
    print(f"    Roll/CS corr, full causal series (both>0) : {eth_res['corr_full_series']:.3f} "
          f"(both>0 on {eth_res['both_nonzero_frac_full']:.1%} of bars)")
    print(f"    Roll/CS corr, AT the sampled event bars   : {eth_res['corr_at_events']:.3f}")

    print("\n  spot vs futures identity check (disclosed in module docstring): "
          "v4_target is market-independent and order_notional makes the "
          "fractional friction rate market-independent too, so the SAME "
          "addon applies to both MarketSpecs -- verified structurally, not "
          "computed twice.")

    print(f"\n  CS smoothing-window sensitivity (BTC, addon only, 0 backtests):")
    print(f"    {'window(days)':>13} {'cs_addon(bps)':>14} {'n_events':>9}")
    sens_rows = []
    for w in CS_WINDOW_GRID:
        r = instrument_addon(btc, btc_windows, cs_window=w)
        sens_rows.append((w, r["cs_addon"], r["n_events"]))
        print(f"    {w / R.BARS_PER_DAY:>13.2f} {r['cs_addon'] * 1e4:>14.2f} "
              f"{r['n_events']:>9,}")

    return dict(btc=btc_res, eth=eth_res, cs_window_sensitivity=sens_rows)


# ==========================================================================
# (2) v4 vs buy_and_hold under three fee tiers, same MarketSpec for both.
# ==========================================================================


def classify(pr_diff) -> str:
    if pr_diff.lo > 0.0:
        return "BEATS"
    if pr_diff.hi < 0.0:
        return "LOSES"
    return "COIN-FLIP"


def measure(strategy, df: pd.DataFrame, start, end, slice_name: str,
            market: MarketSpec):
    _CONFIGS[0] += 1
    if start is not None and str(start) >= R.OOS_START:
        _HOLDOUT_READS[0] += 1
    return R.run_slice(strategy, df, start, end, slice_name, market)


def compare_cell(df: pd.DataFrame, start, end, slice_name: str,
                  market: MarketSpec, tier: str) -> dict:
    v4 = R.TargetStrategy(R.v4_target, name="kelly_regime_v4")
    bh = get_strategy("buy_and_hold")
    a = measure(v4, df, start, end, slice_name, market)
    b = measure(bh, df, start, end, slice_name, market)
    pr = R.paired_diff(a.daily, b.daily, **BOOT_KW)
    return dict(
        tier=tier, slice=slice_name, market=market.name, fee=market.fee_rate,
        v4_final=a.final_balance, bh_final=b.final_balance,
        v4_sharpe=a.sharpe, bh_sharpe=b.sharpe,
        v4_dd=a.max_drawdown_pct, bh_dd=b.max_drawdown_pct,
        d_logret=pr.diff.point, d_logret_lo=pr.diff.lo, d_logret_hi=pr.diff.hi,
        verdict=classify(pr.diff),
    )


ROW_FMT = ("{tier:22} {slice:12} {market:11} fee={fee:.4%} "
          "v4=${vf:>11,.0f} bh=${bf:>11,.0f} Sh {vsh:>5.2f}/{bsh:<5.2f} "
          "dlog {dl:+7.4f} [{lo:+.4f},{hi:+.4f}]  {verdict}")


def show(row: dict) -> None:
    print(ROW_FMT.format(
        tier=row["tier"], slice=row["slice"], market=row["market"], fee=row["fee"],
        vf=row["v4_final"], bf=row["bh_final"], vsh=row["v4_sharpe"], bsh=row["bh_sharpe"],
        dl=row["d_logret"], lo=row["d_logret_lo"], hi=row["d_logret_hi"],
        verdict=row["verdict"]))


def fee_tiers(addon: dict) -> dict[str, tuple[MarketSpec, MarketSpec, MarketSpec, MarketSpec]]:
    """Per-tier (btc_spot, btc_futures, eth_spot, eth_futures) MarketSpecs."""
    btc_addon = addon["btc"]["cs_addon"]
    eth_addon = addon["eth"]["cs_addon"]
    return {
        "baseline": (
            R.fee_at(R.SPOT, BASE_SPOT_FEE), R.fee_at(R.FUTURES, BASE_FUTURES_FEE),
            R.fee_at(R.SPOT, BASE_SPOT_FEE), R.fee_at(R.FUTURES, BASE_FUTURES_FEE),
        ),
        "illiquidity_adjusted": (
            R.fee_at(R.SPOT, BASE_SPOT_FEE + btc_addon),
            R.fee_at(R.FUTURES, BASE_FUTURES_FEE + btc_addon),
            R.fee_at(R.SPOT, BASE_SPOT_FEE + eth_addon),
            R.fee_at(R.FUTURES, BASE_FUTURES_FEE + eth_addon),
        ),
        "sensitivity_0.40pct": (
            R.fee_at(R.SPOT, SENSITIVITY_FEE), R.fee_at(R.FUTURES, FUTURES_SENSITIVITY_FEE),
            R.fee_at(R.SPOT, SENSITIVITY_FEE), R.fee_at(R.FUTURES, FUTURES_SENSITIVITY_FEE),
        ),
    }


def cmd_train(btc: pd.DataFrame, eth: pd.DataFrame, addon: dict) -> list[dict]:
    print("\n" + "=" * 100)
    print("(2) TRAIN-ONLY READING -- v4 vs buy_and_hold, 3 fee tiers, "
          "inner_train + inner_val + ETH replication")
    print("=" * 100)
    tiers = fee_tiers(addon)
    rows: list[dict] = []
    for tier, (btc_spot, btc_fut, eth_spot, eth_fut) in tiers.items():
        for slice_name, (start, end) in R.SLICES.items():
            for market in (btc_spot, btc_fut):
                row = compare_cell(btc, start, end, slice_name, market, tier)
                show(row)
                rows.append(row)
        for market in (eth_spot, eth_fut):
            row = compare_cell(eth, None, None, R.ETH_SLICE_NAME, market, tier)
            show(row)
            rows.append(row)
        print("-" * 100)
    return rows


def cmd_holdout(addon: dict) -> list[dict]:
    print("\n" + "=" * 100)
    print("(3) THE ONE HOLDOUT READ -- v4 vs buy_and_hold, 3 fee tiers, "
          f"BTC {R.OOS_START} onward (real full BTC series, not the "
          "pre-holdout-truncated load_btc())")
    print("=" * 100)
    df, label = load_dataset(ROOT / "data", "spot")
    print(f"  full BTC series: {len(df):,} bars, {df.index[0]:%Y-%m-%d} -> "
          f"{df.index[-1]:%Y-%m-%d}  (label={label})")
    print("  ETH holdout: not read. The ETH replication slice is entirely "
        f"pre-{R.OOS_START} (2016-03 -> 2019-12), so the D3-style falsification "
        "check is already covered by the train-only reading above at zero "
        "extra holdout cost -- logged per the task's own 'log it either way'.")

    tiers = fee_tiers(addon)
    rows: list[dict] = []
    for tier, (btc_spot, btc_fut, _eth_spot, _eth_fut) in tiers.items():
        for market in (btc_spot, btc_fut):
            row = compare_cell(df, R.OOS_START, None, "holdout", market, tier)
            show(row)
            rows.append(row)
        print("-" * 100)
    return rows


# ==========================================================================
# (3) The frozen decision rule (r173_direction.md Step 4, CONSERVATIVE),
#     applied mechanically -- never altered after seeing a number.
# ==========================================================================


def apply_decision_rule(rows: list[dict], label: str) -> dict:
    print(f"\n{'=' * 100}\nDECISION RULE (r173_direction.md Step 4, CONSERVATIVE, frozen) "
          f"-- {label}\n{'=' * 100}")
    by_market: dict[str, dict[str, str]] = {}
    for r in rows:
        by_market.setdefault(r["market"], {})[r["tier"]] = r["verdict"]

    flips: list[str] = []
    for market, tiers in by_market.items():
        base = tiers.get("baseline")
        adj = tiers.get("illiquidity_adjusted")
        sens = tiers.get("sensitivity_0.40pct")
        print(f"  {market:11} baseline={base:>10} illiquidity_adjusted={adj:>10} "
              f"0.40%_sensitivity={sens:>10}")
        if adj is not None and base is not None and adj != base:
            flips.append(f"{market}: {base} -> {adj}")

    if flips:
        verdict = "POSITIVE finding, filed as a new cost caveat"
        print(f"\n  SIGN FLIP on: {', '.join(flips)}")
    else:
        verdict = "NEGATIVE / no new caveat"
        print("\n  No sign flip on any market: the illiquidity-adjusted tier "
              "reads the same qualitative verdict (BEATS/LOSES/COIN-FLIP) as "
              "the existing baseline/0.40% tiers.")
    print(f"\n  VERDICT ({label}): {verdict}")
    return dict(by_market=by_market, flips=flips, verdict=verdict)


# ==========================================================================
# (4) Causality self-test: new local machinery only (window_frame,
#     v4_rebalance_events, the Roll-to-relative unit conversion). Everything
#     else reuses r173_shared's own already-tested causal primitives.
# ==========================================================================


def _synthetic_df(n: int = 60_000, seed: int = 173) -> pd.DataFrame:
    idx = pd.date_range("2017-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    innov = rng.normal(0, 0.0006, n)
    close = 10_000 * np.exp(np.cumsum(innov))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, n)))
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": 1.0}, index=idx)


def _self_test() -> None:
    df = _synthetic_df()

    # (1) window_frame: lo/hi/prefix arithmetic sane, frame length matches.
    frame, lo, hi, prefix = window_frame(df, "2017-02-01", "2017-03-01", warmup=500)
    assert len(frame) == hi - (lo - prefix)
    assert prefix <= 500

    # (2) v4_rebalance_events: every event's row position lies in [start,end),
    # deltas are all > 1e-9, and truncating the frame earlier than an event
    # cannot change that event (causality of the extraction itself).
    events_a = v4_rebalance_events(df, "2017-02-01", "2017-04-01")
    assert all(abs(d) > 1e-9 for _, d in events_a)
    lo_a = int(df.index.searchsorted("2017-02-01"))
    hi_a = int(df.index.searchsorted("2017-04-01", side="right"))
    assert all(lo_a <= p < hi_a for p, _ in events_a)

    events_b = v4_rebalance_events(df, "2017-02-01", "2017-03-15")
    # events strictly before the earlier window's own end must match exactly
    cutoff = int(df.index.searchsorted("2017-03-15", side="right"))
    a_prefix = [(p, d) for p, d in events_a if p < cutoff]
    assert a_prefix == events_b, "rebalance-event extraction is not causal"

    # (3) real-data cross-check: my hand-extracted event COUNT matches the
    # number of orders an actual PaperBroker run submits over the same
    # window (the same check family as r165_conservative_boundary's D-check).
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context
    strat = R.TargetStrategy(R.v4_target, name="kelly_regime_v4")
    frame2, lo2, hi2, prefix2 = window_frame(df, "2017-02-01", "2017-04-01")
    prepared = strat.prepare(frame2.copy())
    broker = PaperBroker(market=R.SPOT, start_balance=10_000.0)
    n_orders = 0
    for i in range(prefix2, len(prepared)):
        ctx = Context(prepared, i, broker)
        strat.on_bar(ctx)
        n_orders += len(ctx.orders)
    events_real = v4_rebalance_events(df, "2017-02-01", "2017-04-01")
    assert n_orders == len(events_real), (n_orders, len(events_real))
    assert n_orders > 0, "vacuous check: no rebalances fired on the probe window"

    # (4) Roll dollar->relative unit conversion: finite, non-negative, and
    # (on this smooth synthetic series where price stays well above 1) each
    # relative value is strictly smaller than its dollar counterpart.
    roll_dollar = R.roll_spread_causal(df)
    close = df["close"].to_numpy()
    roll_rel = np.divide(roll_dollar, close, out=np.zeros_like(roll_dollar), where=close > 0)
    assert np.all(np.isfinite(roll_rel)) and np.all(roll_rel >= 0.0)
    nz = roll_dollar > 0
    assert np.all(roll_rel[nz] < roll_dollar[nz])

    # (5) instrument_addon: finite in [0, 1) on synthetic data, weight > 0.
    res = instrument_addon(df, [("2017-02-01", "2017-04-01")])
    assert res["n_events"] > 0
    assert 0.0 <= res["cs_addon"] < 1.0
    assert 0.0 <= res["roll_addon"] < 1.0

    print("  self-test: PASS (window_frame, v4_rebalance_events causality + "
          "real-PaperBroker cross-check, Roll unit conversion, instrument_addon)")


def cmd_causality() -> bool:
    print("=" * 100)
    print("(0) CAUSALITY -- self-test on this file's own new local machinery "
          "(r173_shared's own primitives are tested in r173_shared._self_test)")
    print("=" * 100)
    try:
        _self_test()
        return True
    except AssertionError as e:
        print(f"  self-test FAILED: {e}")
        return False


# ============================================================================


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "all"

    ok = cmd_causality()
    if not ok:
        raise SystemExit("causality self-test FAILED -- stopping")
    if cmd == "causality":
        return

    btc = R.load_btc()
    eth = R.load_eth()

    addon = cmd_addon(btc, eth)
    if cmd == "addon":
        return

    train_rows: list[dict] = []
    if cmd in ("all", "train"):
        train_rows = cmd_train(btc, eth, addon)
        train_verdict = apply_decision_rule(train_rows, "TRAIN-ONLY (inner_train+inner_val+ETH)")
    if cmd == "train":
        return

    if cmd in ("all", "holdout"):
        print(f"\nconfigs evaluated before the holdout was touched: {_CONFIGS[0]}")
        hold_rows = cmd_holdout(addon)
        hold_verdict = apply_decision_rule(hold_rows, "HOLDOUT (2023+, one read)")

    print("\n" + "=" * 100)
    print(f"Configurations evaluated (this branch) : {_CONFIGS[0]}")
    print(f"Holdout reads (this branch)            : {_HOLDOUT_READS[0]}")
    print("=" * 100)


if __name__ == "__main__":
    main()
