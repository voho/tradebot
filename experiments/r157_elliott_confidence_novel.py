#!/usr/bin/env python
"""R-157 NOVEL branch (B-10): Elliott-Wave STRUCTURE as a SIZE-axis confidence
gate on ``kelly_regime_v4``, not as a directional predictor.

Not registered: lives under ``experiments/`` per ROUTINE.md step 5. No file
under ``src/tradebot/strategies/`` is modified.

The idea, precisely
--------------------
This project's most robust finding across 25 registered strategies: every
strategy that decides HOW MUCH to hold (the SIZE axis) makes money; every
strategy that tries to predict WHAT HAPPENS NEXT (the DIRECTION axis) loses
to fees. The parallel CONSERVATIVE branch of this same round builds a
literal, deterministic Elliott Wave counter that reads a directional
buy/sell signal off wave completion -- expected, per R-18's prior literature
review, to fail, because it is a direction predictor.

This branch is different. It builds the same causal ZigZag + Elliott-Wave
structural-validity machinery, but never reads a direction off it. Instead
it produces a continuous, bar-by-bar, causal STRUCTURAL CLARITY score
``c[i] in [0,1]`` -- how cleanly the recent price path conforms to a
coherent Elliott-Wave count (5-wave impulse OR 3-wave A-B-C correction) AT
ALL, regardless of which way that structure points. ``c`` says nothing
about whether to be long or short; it says whether the crowd's recent
behaviour is legible (a well-formed wave count exists) or noise (no
coherent count fits). That score is then used to DAMPEN
``kelly_regime_v4``'s existing exposure in low-clarity periods -- a
confidence gate on the SIZE axis, not a new signal on the DIRECTION axis.

Mechanism
---------
1. Causal ZigZag pivots on a daily-ATR-scaled reversal threshold (never
   repainted -- a pivot, once confirmed, is never revisited).
2. At every bar, take the most recent (up to 5) confirmed pivots plus the
   currently-forming (unconfirmed) leg as a 6-point candidate impulse
   (waves 1-5) and, separately, the most recent (up to 3) confirmed pivots
   plus the forming leg as a 4-point candidate correction (A-B-C). Score
   each against the textbook rules (wave 2 retraces 38.2-100% of wave 1
   without exceeding wave 0's start; wave 3 is never the shortest of
   1/3/5; wave 4 does not overlap wave 1; corrective B retraces 38.2-78.6%
   of A) with smooth (not hard pass/fail) bands, so the score is
   continuous and updates every bar as the forming leg's retracement
   develops. ``c[i] = max(impulse_score[i], corrective_score[i])``.
3. Subclass ``KellyRegimeV4``, copy its ``prepare()`` loop body verbatim
   (v3/v4's own conditional-vol-target state machine), and change exactly
   one line: ``desired = frac[i] * scale * blend(c[i])`` where
   ``blend(c) = confidence_floor + (1 - confidence_floor) * c``, inserted
   BEFORE the deadband check (inserting after would defeat the deadband,
   since ``c`` changes every bar -- turnover would explode). Nothing else
   about v4 (anchors, vol target, cap, deadband amount) is touched, per
   the project's own "test the factors before retuning either" rule
   (R-62).

Not a duplicate of anything in docs/LEDGER.md section C: the only
Elliott-Wave-related ruled-out entry is R-18's desk rejection of Elliott
Wave as a DIRECTIONAL predictor, which does not cover a structural-
confidence SIZING gate. A literature search (completed before this round
was dispatched) found no prior published or in-repo use of Elliott Wave
structure as a continuous confidence/regime-clarity multiplier on a
sizing rule.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # so `from scripts.experiment import ...` resolves regardless of cwd

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

CONFIG_COUNT = 0  # every prepare()/backtest configuration evaluated, counted honestly


# =====================================================================
# Step 1 -- causal ZigZag + Elliott-Wave structural-clarity engine
# =====================================================================

def daily_atr_pct_causal(df: pd.DataFrame, atr_days: int = 14) -> np.ndarray:
    """Real (Wilder-style) daily True Range, EWM-smoothed over ``atr_days``,
    as a fraction of price, and made causal exactly like every other daily
    external signal already in this project (``tradebot.data``'s
    ``align_*_causal`` family): a day's own True Range is only knowable
    once that day has closed, so it is shifted one full day later before
    being broadcast (ffill) back onto the 5-minute grid. A bar on day D
    therefore only ever sees ATR information computed from days strictly
    before D -- never from D's own still-forming high/low.

    Built from the 5m closes' own daily resample (open=first, high=max,
    low=min, close=last close) rather than the true intraday high/low --
    this project's OHLCV bars are 5-minute, so a "daily ATR" computed this
    way is a slightly tighter proxy for the day's true range than one
    built from intraday extremes, but it is causal, self-contained (no new
    data file) and only used as a threshold SCALE for the free constant
    ``k`` below -- exactly calibrated, empirically, by the k-sweep.
    """
    daily = df["close"].resample("1D").agg(["first", "max", "min", "last"])
    daily.columns = ["open", "high", "low", "close"]
    prev_close = daily["close"].shift(1)
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_days, min_periods=atr_days).mean()
    atr_pct = atr / daily["close"]
    atr_pct.index = atr_pct.index + pd.Timedelta(days=1)  # visible only from next day
    full = (atr_pct.reindex(atr_pct.index.union(df.index))
            .sort_index().ffill().reindex(df.index))
    return full.to_numpy()


def _band_score(x: float, lo: float, hi: float, soft: float) -> float:
    """1.0 inside [lo, hi]; decays linearly to 0 outside over ``soft`` * width."""
    width = hi - lo
    if width <= 0:
        return 0.0
    if lo <= x <= hi:
        return 1.0
    d = (lo - x) if x < lo else (x - hi)
    return max(0.0, 1.0 - d / (soft * width))


def _ramp_score(margin: float, scale: float) -> float:
    """1.0 for margin >= 0 (rule satisfied); decays linearly to 0 over ``scale``
    below zero (rule violated by that much)."""
    if scale <= 1e-12:
        return 1.0 if margin >= 0 else 0.0
    if margin >= 0:
        return 1.0
    return max(0.0, 1.0 + margin / scale)


def score_impulse(points: list[float]) -> float:
    """Score up to a 6-point (5-leg) candidate 5-wave impulse count.

    ``points`` is [p0, p1, ..., pn] in chronological order -- the earliest
    of the last (up to 5) confirmed pivots through the CURRENT price
    (the still-forming, unconfirmed leg). Waves i = leg (points[i-1] ->
    points[i]). Scores only the rules checkable with however many legs are
    present (2 legs minimum: wave 1 + wave 2), so a clean, still-forming
    early structure is not penalized merely for being early -- and a
    complete 5-leg structure that violates nothing scores 1.0.
    """
    n = len(points) - 1
    if n < 2:
        return 0.0
    legs = [points[i + 1] - points[i] for i in range(n)]
    amps = [max(abs(leg), 1e-12) for leg in legs]
    s = 1.0 if legs[0] >= 0 else -1.0

    scores = []

    # Wave 2 retraces 38.2-100% of wave 1, without fully retracing past
    # wave 0's start (points[0]).
    retr2 = amps[1] / amps[0]
    sc2 = _band_score(retr2, 0.382, 1.0, 0.25)
    cross = s * (points[0] - points[2])  # > 0 means wave 2 crossed back past p0
    if cross > 0:
        sc2 *= max(0.0, 1.0 - (cross / amps[0]) / 0.15)
    scores.append(sc2)

    # Wave 3 is never the shortest of waves 1/3/5 (checked against whatever
    # of 1 and 5 is already available).
    if n >= 3:
        others = [amps[0]] + ([amps[4]] if n >= 5 else [])
        margin3 = amps[2] - min(others)
        scores.append(_ramp_score(margin3, amps[0] * 0.5))

    # Wave 4 does not overlap wave 1's price territory (non-diagonal rule).
    if n >= 4:
        margin4 = s * (points[4] - points[1])
        scores.append(_ramp_score(margin4, amps[0] * 0.20))

    return float(np.mean(scores))


def score_corrective(points: list[float]) -> float:
    """Score up to a 4-point (3-leg) candidate A-B-C correction.

    B retraces 38.2-78.6% of A; a forming C that runs opposite to A's
    direction (i.e. not yet resuming the correction's direction) is
    penalized smoothly, not zeroed, since C is still in progress.
    """
    n = len(points) - 1
    if n < 2:
        return 0.0
    legs = [points[i + 1] - points[i] for i in range(n)]
    amps = [max(abs(leg), 1e-12) for leg in legs]
    retr_b = amps[1] / amps[0]
    score = _band_score(retr_b, 0.382, 0.786, 0.25)
    if n >= 3:
        s_a = 1.0 if legs[0] >= 0 else -1.0
        s_c = 1.0 if legs[2] >= 0 else -1.0
        if s_c != s_a:
            frac = amps[2] / amps[0]
            score *= max(0.3, 1.0 - min(frac, 1.0))
    return float(score)


def elliott_wave_confidence(df: pd.DataFrame, k: float = 2.5,
                             atr_days: int = 14) -> np.ndarray:
    """Causal, bar-by-bar structural-clarity score ``c[i] in [0,1]``.

    ZigZag: a new pivot confirms only once price reverses by >= ``k`` *
    the daily-ATR-scaled threshold from the running extreme since the
    last CONFIRMED pivot. Confirmed pivots are appended to a list and
    never modified afterward -- no repainting. At every bar, the most
    recent confirmed pivots plus the current (still-forming) extreme are
    scored as a candidate impulse and a candidate correction; ``c[i]`` is
    the better of the two. Direction is never read off this -- only how
    cleanly the recent path fits SOME coherent wave structure.
    """
    close = df["close"].to_numpy()
    thresh_pct = k * daily_atr_pct_causal(df, atr_days=atr_days)
    n = len(df)
    c = np.zeros(n)

    pivots: list[float] = []
    direction = 1  # 1 = searching for the next HIGH, -1 = searching for the next LOW
    extreme_price = close[0]

    for i in range(n):
        px = close[i]
        thresh = thresh_pct[i]
        if direction == 1:
            if px > extreme_price:
                extreme_price = px
            elif np.isfinite(thresh) and (extreme_price - px) / extreme_price >= thresh:
                pivots.append(extreme_price)
                direction = -1
                extreme_price = px
        else:
            if px < extreme_price:
                extreme_price = px
            elif np.isfinite(thresh) and (px - extreme_price) / extreme_price >= thresh:
                pivots.append(extreme_price)
                direction = 1
                extreme_price = px

        pts5 = (pivots[-5:] if len(pivots) >= 5 else pivots[:]) + [px]
        pts3 = (pivots[-3:] if len(pivots) >= 3 else pivots[:]) + [px]
        si = score_impulse(pts5) if len(pts5) >= 3 else 0.0
        sc = score_corrective(pts3) if len(pts3) >= 3 else 0.0
        c[i] = si if si >= sc else sc

    return c


def blend(c: float, confidence_floor: float) -> float:
    """confidence_floor + (1 - confidence_floor) * c -- never zeroes v4's
    base signal, only dampens it in low-clarity periods."""
    return confidence_floor + (1.0 - confidence_floor) * c


# =====================================================================
# Step 2 -- fold c[i] into kelly_regime_v4's sizing as a dampener
# =====================================================================

class ElliottConfidenceKellyV4(KellyRegimeV4):
    """``kelly_regime_v4`` with exposure dampened, before the deadband, by
    a continuous Elliott-Wave structural-clarity score. See module
    docstring for the full mechanism. Not registered (experiment only).
    """

    name = "elliott_confidence_kelly_v4"

    def __init__(self, confidence_floor: float = 0.5, ew_k: float = 2.5,
                 ew_atr_days: int = 14, **kwargs) -> None:
        super().__init__(**kwargs)
        self.confidence_floor = confidence_floor
        self.ew_k = ew_k
        self.ew_atr_days = ew_atr_days

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # ---- identical to KellyRegimeV3.prepare() up through computing
        # `frac`, `full`, `steady` -- copied, not inherited, because the
        # confidence dampener must be applied to `desired` BEFORE the
        # deadband check inside the per-bar loop.
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
               * np.sqrt(365.25 * BARS_PER_DAY)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        # ---- the one new ingredient
        c = elliott_wave_confidence(df, k=self.ew_k, atr_days=self.ew_atr_days)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0
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
            # ---- the ONLY changed line relative to v3/v4's loop body:
            desired = frac[i] * scale * blend(c[i], self.confidence_floor)
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df


# =====================================================================
# Causality self-check (manual -- this class is not registered, so the
# framework's own per-strategy causality suite does not cover it)
# =====================================================================

def causality_truncation_check(df: pd.DataFrame, strategy_factory, pairs) -> bool:
    """prepare() on df.iloc[:N] vs df.iloc[:M] (M > N) must produce
    identical `target` values for the first N rows, for each (N, M) pair.
    """
    ok_all = True
    for label, n, m in pairs:
        s_n = strategy_factory()
        s_m = strategy_factory()
        out_n = s_n.prepare(df.iloc[:n].copy())["target"].to_numpy()
        out_m = s_m.prepare(df.iloc[:m].copy())["target"].to_numpy()[:n]
        same = np.array_equal(out_n, out_m)
        ok_all &= same
        print(f"  truncation[{label}] N={n:,} M={m:,}: "
              f"{'PASS (identical)' if same else 'FAIL (mismatch!)'}")
        if not same:
            diff = np.flatnonzero(out_n != out_m)
            print(f"    first mismatch at row {diff[0]} of {n:,}: "
                  f"{out_n[diff[0]]!r} vs {out_m[diff[0]]!r}")
    return ok_all


# =====================================================================
# Evaluation harness
# =====================================================================

def ev(strategy, df, market, start=None, end=None, tag="", balance=1_000.0):
    global CONFIG_COUNT
    CONFIG_COUNT += 1
    t0 = time.time()
    if start is None and end is None:
        result = run_backtest(strategy, df, market, balance)
    else:
        result = run_period(strategy, df, start, end, market=market, start_balance=balance)
    m = compute_metrics(result)
    equity = result.equity.to_numpy()
    rets = np.diff(np.log(np.clip(equity, 1e-9, None)))
    realized_vol = (float(np.std(rets) * np.sqrt(365.25 * BARS_PER_DAY))
                    if len(rets) > 1 else float("nan"))
    print(f"{tag or strategy.name:34s} {market.name:11s} "
          f"final=${m.final_balance:>13,.0f} ({m.profit_pct:>+9.1f}%) "
          f"trades={m.num_trades:>5d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} tim={m.time_in_market_pct:>5.1f}% "
          f"rvol={realized_vol:>5.2f} {'LIQUIDATED' if m.liquidated else ''} "
          f"[{time.time() - t0:.0f}s]")
    return m


def main() -> None:
    from scripts.experiment import DF, FUTURES, OOS_START, SPOT

    INNER_TRAIN = ("2017-01-01", "2020-12-31")
    INNER_VAL = ("2021-01-01", "2022-12-31")

    print("=" * 100)
    print("CAUSALITY SELF-CHECK (manual truncation -- class not registered)")
    print("=" * 100)
    causality_ok = causality_truncation_check(
        DF, lambda: ElliottConfidenceKellyV4(confidence_floor=0.5, ew_k=2.5),
        pairs=[("small", 50_000, 90_000), ("medium", 150_000, 260_000),
               ("large", 400_000, 700_000)],
    )
    print(f"\nOVERALL causality truncation check: {'PASS' if causality_ok else 'FAIL'}\n")

    print("=" * 100)
    print("INNER-TRAIN (2017-01-01 -> 2020-12-31), futures 5x")
    print("=" * 100)
    ev(KellyRegimeV4(), DF, FUTURES, *INNER_TRAIN, tag="baseline v4")
    train_results = {}
    for k in (2.5, 4.0):
        for cf in (0.3, 0.5, 0.7):
            strat = ElliottConfidenceKellyV4(confidence_floor=cf, ew_k=k)
            m = ev(strat, DF, FUTURES, *INNER_TRAIN, tag=f"k={k} floor={cf}")
            train_results[(k, cf)] = m

    print("\n" + "=" * 100)
    print("INNER-VALIDATION (2021-01-01 -> 2022-12-31), futures 5x")
    print("=" * 100)
    base_val = ev(KellyRegimeV4(), DF, FUTURES, *INNER_VAL, tag="baseline v4")
    val_results = {}
    for k in (2.5, 4.0):
        for cf in (0.3, 0.5, 0.7):
            strat = ElliottConfidenceKellyV4(confidence_floor=cf, ew_k=k)
            m = ev(strat, DF, FUTURES, *INNER_VAL, tag=f"k={k} floor={cf}")
            val_results[(k, cf)] = m

    print("\nSelection criterion (stated before looking at holdout): best Sharpe on "
          "inner-validation, breaking ties by lower max drawdown; must also beat "
          "unmodified v4's inner-validation Sharpe to be worth freezing at all.")
    best_key = max(val_results, key=lambda key: (val_results[key].sharpe,
                                                   -val_results[key].max_drawdown_pct))
    best_k, best_cf = best_key
    print(f"Baseline v4 inner-val: sharpe={base_val.sharpe:.2f} DD={base_val.max_drawdown_pct:.1f}%")
    for key, m in sorted(val_results.items()):
        marker = "  <== FROZEN" if key == best_key else ""
        print(f"  k={key[0]} floor={key[1]}: sharpe={m.sharpe:.2f} "
              f"DD={m.max_drawdown_pct:.1f}% final=${m.final_balance:,.0f}{marker}")
    print(f"\nFROZEN CONFIGURATION: ew_k={best_k}, confidence_floor={best_cf}")

    frozen = lambda: ElliottConfidenceKellyV4(confidence_floor=best_cf, ew_k=best_k)  # noqa: E731

    print("\n" + "=" * 100)
    print(f"HOLDOUT (start={OOS_START}), spot and futures_5x")
    print("=" * 100)
    holdout = {}
    for mname, market in (("spot", SPOT), ("futures_5x", FUTURES)):
        for sname, strat in (("frozen (modified v4)", frozen()),
                              ("unmodified kelly_regime_v4", KellyRegimeV4()),
                              ("buy_and_hold", BuyAndHold())):
            tag = f"{mname:11s}| {sname}"
            m = ev(strat, DF, market, start=OOS_START, tag=tag)
            holdout[(mname, sname)] = m

    print("\n" + "=" * 100)
    print("ETH FALSIFICATION (pre-registered, frozen -- ethusd_bitfinex_5m.csv.gz, "
          "R-17 window convention: full file, 2016-03 -> 2019-12, the only window "
          "this file covers)")
    print("=" * 100)
    eth = load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")
    print(f"ETH Bitfinex: {len(eth):,} bars, {eth.index[0]} -> {eth.index[-1]}\n")
    eth_results = {}
    for mname, market in (("spot", SPOT), ("futures_5x", FUTURES)):
        for sname, strat in (("frozen (modified v4)", frozen()),
                              ("unmodified kelly_regime_v4", KellyRegimeV4()),
                              ("buy_and_hold", BuyAndHold())):
            tag = f"{mname:11s}| {sname}"
            m = ev(strat, eth, market, tag=tag)  # full file, no start/end slicing needed
            eth_results[(mname, sname)] = m

    print("\n" + "=" * 100)
    print(f"Total configurations evaluated: {CONFIG_COUNT}")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("DECISION RULE CHECK (frozen before any holdout number was seen)")
    print("=" * 100)
    for mname in ("spot", "futures_5x"):
        frozen_m = holdout[(mname, "frozen (modified v4)")]
        v4_m = holdout[(mname, "unmodified kelly_regime_v4")]
        hold_m = holdout[(mname, "buy_and_hold")]
        a = frozen_m.final_balance > hold_m.final_balance
        dsharpe = frozen_m.sharpe - v4_m.sharpe
        ddd = v4_m.max_drawdown_pct - frozen_m.max_drawdown_pct
        b = (dsharpe > 0.2) or (ddd > 0 and frozen_m.final_balance >= v4_m.final_balance * 0.98)
        print(f"{mname}: (a) beats buy&hold = {a}  "
              f"(b) dSharpe={dsharpe:+.2f} dDD(pp)={ddd:+.1f} -> clears/DD-improves = {b}")

    print("\n(c) ETH falsification -- does frozen preserve v4's own property "
          "(non-degradation), cell by cell:")
    for mname in ("spot", "futures_5x"):
        frozen_m = eth_results[(mname, "frozen (modified v4)")]
        v4_m = eth_results[(mname, "unmodified kelly_regime_v4")]
        hold_m = eth_results[(mname, "buy_and_hold")]
        dsharpe = frozen_m.sharpe - v4_m.sharpe
        ddd = v4_m.max_drawdown_pct - frozen_m.max_drawdown_pct
        beats_hold_frozen = frozen_m.final_balance > hold_m.final_balance
        beats_hold_v4 = v4_m.final_balance > hold_m.final_balance
        non_degraded = dsharpe > -0.2 and ddd > -5.0  # frozen not meaningfully worse than v4
        print(f"  {mname}: v4_beats_hold={beats_hold_v4} frozen_beats_hold={beats_hold_frozen} "
              f"dSharpe(frozen-v4)={dsharpe:+.2f} dDD(pp)={ddd:+.1f} "
              f"-> non_degraded_vs_v4={non_degraded}")


if __name__ == "__main__":
    main()
