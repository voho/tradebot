#!/usr/bin/env python
"""R-77 NOVEL branch: cross-venue (Bitstamp-vs-Coinbase) lead-lag price
discovery as an early, precision-weighted confirming vote on
``kelly_regime_v4``'s own 3-anchor gate.

Idea, one sentence
------------------
Every INFO-axis round before this one (R-44 on-chain, R-53 macro, R-54/
R-55/R-58 stablecoin, R-73 DVOL, R-74 MVRV, R-75 calendar, R-76 cross-
instrument pairs) fed the strategy either an *external* data source about
a different quantity, or a relationship between two *different*
instruments on the *same* venue; none has used a second, independent read
of the SAME instrument's price from a *different* venue. Hasbrouck (1995,
J. Finance 50(4)) and Gonzalo & Granger (1995, J. Business & Econ. Stat.
13(1)) are the classic information-share / component-share framework for
which of several markets trading one asset moves first; Putninš (2013, J.
Financial Markets) is the correction for unequal microstructure noise
between venues (the Information Leadership Share); Alexander & Heck
(2020, J. Financial Stability 50:100776) study exactly Bitstamp vs.
Coinbase among their regulated-spot panel and find price leadership
between spot venues is not fixed or symmetric. R-73 (DVOL) and R-74 (MVRV
rate-of-change) both failed because the external signal *lagged* the
price-anchor gate rather than leading it (diagnosed explicitly in both
ledger entries); a genuine cross-venue lead-lag effect, being a different
market's price at the SAME frequency rather than a slower external
series, is a mechanism that could plausibly avoid that specific failure
mode -- IF it exists and is stable. This file's whole job is to measure
that *before* writing a single line of strategy-sizing code (`descriptive()`,
below, which is the falsification gate, not a formality).

Constraint attacked: INFO. Not a duplicate of R-41 (Deribit perp-vs-spot
basis -- a futures/spot spread on ONE venue), R-76 (cross-instrument
pairs, same-venue legs), R-73/R-74 (both external signals diagnosed as
*lagging* -- exactly the failure mode this mechanism is tested against),
or this round's own disjoint CONSERVATIVE sibling (same-bar Bitstamp/
Coinbase divergence as a mean-reversion/arbitrage-friction vote -- a
different economic effect, same-bar rather than lagged, not read or
coordinated with, per docs/ROUTINE.md's parallelism rule).

Shared infra
------------
``experiments/r77_shared.py`` (frozen, not edited here) supplies
``load_crossvenue_bars()`` -- the Bitstamp OHLCV frame (the tradable
series) restricted to timestamps both venues have a bar, with one causal
column, ``divergence = log(bitstamp_close / coinbase_close)``. That
column is same-bar and NOT used here (per the shared module's own
docstring and this round's brief) -- ``coinbase_close`` is instead
reconstructed as ``bitstamp_close / exp(divergence)`` (algebraically
identical to the real Coinbase close on the same intersection index,
verified byte-for-byte against ``tradebot.data.load_coinbase_spot``
before this file was written: max|diff| ~1.5e-11), so this file's own
join is the SAME intersection the shared loader already computed -- no
new lookahead surface.

Critical causality requirement
-------------------------------
A bar timestamped T on both venues nominally closes at the same
wall-clock instant, so ``coinbase_return[T]`` and ``bitstamp_return[T]``
are contemporaneous -- using ``coinbase_return[T]`` to size the Bitstamp
bar T itself is lookahead, full stop, even though both numbers carry the
same index label. Every column this file computes that touches Coinbase
data uses ONLY ``coinbase_return.shift(k)`` for ``k >= 1``. This is
enforced three ways: (1) the feature-construction code never references
an unshifted Coinbase column, (2) ``causality()`` runs a two-pathway
tamper probe (Bitstamp OHLCV, reconstructed-Coinbase-via-divergence) plus
a truncation-invariance check, and (3) a synthetic series with a KNOWN
lag structure verifies the statistic actually recovers a real lead when
one is deliberately injected (a green suite alone does not prove the
statistic is sensitive to anything; the R-21 $3.7e23-with-a-green-suite
history is exactly why this is a dedicated test, not an inference from
"causality() passed").

Pre-registered falsification test (frozen before any number below was read)
-----------------------------------------------------------------------------
Measure the lead-lag statistic's sign and strength SEPARATELY on
inner-train (<=2020-12-31) and inner-validation (2021-01-01->2022-12-31).
Primary statistic: the ROLLING (causal, 3-day window) partial correlation
of ``bitstamp_return[t]`` on ``coinbase_return[t-1]``, controlling for
``bitstamp_return[t-1]`` (does Coinbase's prior bar predict Bitstamp's
current bar beyond Bitstamp's own momentum) -- reported as its mean/
median/frac-time-positive on each split. Supporting robustness check: the
STATIC (whole-split, non-rolling) version of the same partial correlation
at lag in {1,2,3,5,10,20} bars, each split, against its own standard
error (~1/sqrt(n)).

GATE RULE, frozen now: call the effect STABLE only if, at the primary
lag=1 statistic, (a) sign(train) == sign(valid) at both the rolling
mean and the static estimate, AND (b) |static partial r| clears 3x its
own standard error in BOTH splits (distinguishable from zero, not just a
label), AND (c) a majority (>=4/6) of the robustness lags {1,2,3,5,10,20}
agree in sign with the primary lag within each split. Failing (a), (b) or
(c) means the direction of leadership is NOT established -- report
NEGATIVE at this gate, per this round's brief, and do NOT build a sized
strategy around it. ``sweep()``/``select()``/``artifact()``/``holdout()``
all check this gate first and refuse to run a real sweep if it fails
(mirroring ``experiments/r76_novel_pairs_distance_kelly.py``'s
``cmd_all()`` stop-at-the-gate pattern and R-75's "neither branch built a
line of strategy code past the gate" convention) -- a `KellyRegimeV20`-
style Strategy class is still written below (so the mechanism is
concrete, reusable, and the causality tamper probe has something real to
tamper with), but it is never sized against a real backtest with a swept
configuration.

Code reuse decision
--------------------
``_anchor_votes`` is duplicated (not imported) from
``kelly_regime_v14_macro_lead.py`` / ``kelly_regime_v16_stablecoin_confirm.py``
-- the same precedent those files themselves set duplicating it from each
other, since neither is shared project infrastructure. The confirming-
vote combination shape, ``frac = (anchor_sum + w*vote)/(3+w)``, is R-53's
own architecture, reused verbatim (structurally, not by import) with a
per-bar ``w`` (0 while leadership is inactive, a fixed weight while
latched active) rather than a constant one, the natural generalization
for a signal that is not always "on" the way the stablecoin vote is.
``experiments/r77_shared.py`` is imported unmodified.

Usage
-----
    python experiments/r77_novel_crossvenue_leadlag.py descriptive   # THE gate
    python experiments/r77_novel_crossvenue_leadlag.py sweep         # inner-train, only if gate passes
    python experiments/r77_novel_crossvenue_leadlag.py select        # inner-validation, only if gate passes
    python experiments/r77_novel_crossvenue_leadlag.py artifact      # exposure-artifact R^2 check
    python experiments/r77_novel_crossvenue_leadlag.py causality     # lookahead probe (mandatory, always run)
    python experiments/r77_novel_crossvenue_leadlag.py holdout       # only if both gates clear
    python experiments/r77_novel_crossvenue_leadlag.py all           # everything, in order
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r77_shared import (  # noqa: E402
    INNER_TRAIN_END, INNER_VALID, OOS_START, load_crossvenue_bars,
)

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures 5x", FUTURES))

TRAIN = ("2017-01-01", INNER_TRAIN_END)
VALID = INNER_VALID
INCUMBENT = "kelly_regime_v4"

# Primary, structural (not swept) construction choices, fixed before any
# strategy number was read:
LAG = 1                     # the shortest possible lead: the very next Coinbase bar
ROLL_WINDOW = 3 * BARS_PER_DAY   # 3 days (~864 bars): long enough for a
                                  # stable rolling partial-correlation estimate at
                                  # 5-minute frequency, short enough to track the
                                  # non-stationary, time-varying leadership
                                  # Alexander & Heck (2020) document -- a fixed
                                  # window over the WHOLE series would contradict
                                  # that finding by construction.
ROBUSTNESS_LAGS = (1, 2, 3, 5, 10, 20)

N_EVALUATED = 0  # distinct configurations evaluated, project-trials count
_SEEN_CONFIGS: set[str] = set()


# --------------------------------------------------------------------- data


def assert_no_holdout(df: pd.DataFrame) -> None:
    """Every frame this file touches must have max timestamp < OOS_START."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


def load_bars(allow_holdout: bool = False) -> pd.DataFrame:
    """The shared cross-venue frame, truncated before OOS_START unless
    explicitly told not to (only ``holdout()`` ever passes True, and only
    once both pre-registered gates have cleared)."""
    bars = load_crossvenue_bars()
    if not allow_holdout:
        cutoff = pd.Timestamp(OOS_START, tz=bars.index.tz)
        bars = bars.loc[bars.index < cutoff].copy()
        assert_no_holdout(bars)
    return bars


def add_return_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct Coinbase's own close and both venues' own 5-minute log
    returns. ``coinbase_close`` uses ONLY ``close`` and ``divergence``,
    both already on df -- no new file read, no new join surface beyond
    what ``r77_shared.load_crossvenue_bars`` already computed."""
    df = df.copy()
    df["coinbase_close"] = df["close"] / np.exp(df["divergence"])
    df["bt_ret"] = np.log(df["close"]).diff()
    df["cb_ret"] = np.log(df["coinbase_close"]).diff()
    return df


DF = add_return_columns(load_bars())
print(f"{len(DF):,} aligned bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d} "
      f"(pre-holdout only, enforced by load_bars/assert_no_holdout)", file=sys.stderr)


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0,
            config_key: str | None = None):
    global N_EVALUATED
    if config_key is not None and config_key not in _SEEN_CONFIGS:
        _SEEN_CONFIGS.add(config_key)
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market, start_balance=balance)
    return compute_metrics(result), result


def line(tag, m, market_name=""):
    print(f"  {tag:42s} {market_name:10s} final=${m.final_balance:>11,.0f} "
          f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>7,.0f}{'  LIQUIDATED' if m.liquidated else ''}")


# ------------------------------------------------------- the lead-lag statistic


def _partial_corr_from_series(y: pd.Series, x: pd.Series, w: pd.Series) -> tuple[float, int]:
    """Static (whole-sample) partial correlation of y on x, controlling for
    w: r(x,y|w) = (rxy - rxw*rwy) / sqrt((1-rxw^2)(1-rwy^2)). Rows with any
    NaN (from the leading shift) are dropped first."""
    frame = pd.DataFrame({"x": x, "y": y, "w": w}).dropna()
    n = len(frame)
    if n < 10:
        return float("nan"), n
    rxy = frame["x"].corr(frame["y"])
    rxw = frame["x"].corr(frame["w"])
    rwy = frame["w"].corr(frame["y"])
    denom = np.sqrt((1.0 - rxw ** 2) * (1.0 - rwy ** 2))
    return float((rxy - rxw * rwy) / denom) if denom > 0 else float("nan"), n


def static_leadlag_partial_corr(df: pd.DataFrame, lag: int) -> tuple[float, int]:
    """The gate's supporting statistic at one lag: bitstamp_return[t] on
    coinbase_return[t-lag], controlling for bitstamp_return[t-1]. ``x`` is
    built with ``.shift(lag)`` -- coinbase_return at index i-lag never
    index >= i, satisfying this file's causality requirement by
    construction."""
    y = df["bt_ret"]
    x = df["cb_ret"].shift(lag)
    w = df["bt_ret"].shift(1)
    return _partial_corr_from_series(y, x, w)


def rolling_leadlag_partial_corr(bt_ret: pd.Series, cb_ret: pd.Series,
                                  window: int = ROLL_WINDOW, lag: int = LAG) -> pd.Series:
    """THE causal, rolling lead-lag statistic (design point 1): rolling
    partial correlation of bitstamp_return[t] on coinbase_return[t-lag],
    controlling for bitstamp_return[t-1], over a trailing ``window``-bar
    window. Computed via closed-form rolling moments (all ``.rolling(...).sum()``
    calls, each using only rows <= i at row i) rather than a per-bar
    regression loop -- exactly as causal, an order of magnitude cheaper,
    and easy to audit: every input to every sum at row i is index <= i,
    and ``x``/``w`` are pre-shifted so the coinbase leg never touches
    index i or later.
    """
    y = bt_ret
    x = cb_ret.shift(lag)
    w = bt_ret.shift(1)
    n = float(window)
    r = window
    sx, sy, sw = (s.rolling(r, min_periods=r).sum() for s in (x, y, w))
    sxx, syy, sww = ((s * s).rolling(r, min_periods=r).sum() for s in (x, y, w))
    sxy = (x * y).rolling(r, min_periods=r).sum()
    sxw = (x * w).rolling(r, min_periods=r).sum()
    swy = (w * y).rolling(r, min_periods=r).sum()
    mx, my, mw = sx / n, sy / n, sw / n
    Sxx, Syy, Sww = sxx - n * mx * mx, syy - n * my * my, sww - n * mw * mw
    Sxy, Sxw, Swy = sxy - n * mx * my, sxw - n * mx * mw, swy - n * mw * my
    with np.errstate(divide="ignore", invalid="ignore"):
        rxy = Sxy / np.sqrt(Sxx * Syy)
        rxw = Sxw / np.sqrt(Sxx * Sww)
        rwy = Swy / np.sqrt(Sww * Syy)
        denom = np.sqrt((1.0 - rxw ** 2) * (1.0 - rwy ** 2))
        partial = (rxy - rxw * rwy) / denom
    return partial


def _split_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    train_end = pd.Timestamp(TRAIN[1], tz=df.index.tz)
    valid_lo = pd.Timestamp(VALID[0], tz=df.index.tz)
    valid_hi = pd.Timestamp(VALID[1], tz=df.index.tz)
    return {
        "train": df.index <= train_end,
        "valid": (df.index >= valid_lo) & (df.index <= valid_hi),
    }


# ------------------------------------------------------- step 2b: descriptive


def descriptive() -> dict:
    """Step 2b -- THE falsification gate, run and reported before any
    strategy sweep. Measures the lead-lag statistic's sign/strength
    separately on inner-train and inner-validation and applies the frozen
    GATE RULE from this module's docstring."""
    masks = _split_masks(DF)

    print("=" * 88)
    print("PRIMARY: rolling (causal, 3-day window) partial correlation, "
          "coinbase_return[t-1] -> bitstamp_return[t] | bitstamp_return[t-1]")
    print("=" * 88)
    rolling = rolling_leadlag_partial_corr(DF["bt_ret"], DF["cb_ret"])
    rolling_stats = {}
    for split, mask in masks.items():
        s = rolling[mask].dropna()
        rolling_stats[split] = {
            "mean": float(s.mean()), "median": float(s.median()),
            "frac_pos": float((s > 0).mean()), "std": float(s.std()), "n": int(len(s)),
        }
        st = rolling_stats[split]
        print(f"  {split:5s}: mean={st['mean']:+.5f}  median={st['median']:+.5f}  "
              f"frac-time-positive={st['frac_pos']:.3f}  std={st['std']:.5f}  n={st['n']:,}")

    print("\n" + "=" * 88)
    print("SUPPORTING: static (whole-split) partial correlation, per lag, per split "
          "(vs its own ~1/sqrt(n) standard error)")
    print("=" * 88)
    static_stats = {}
    for lag in ROBUSTNESS_LAGS:
        static_stats[lag] = {}
        for split, mask in masks.items():
            r, n = static_leadlag_partial_corr(DF[mask], lag)
            se = 1.0 / np.sqrt(n) if n > 0 else float("nan")
            static_stats[lag][split] = {"r": r, "n": n, "se": se}
            print(f"  lag={lag:2d}  {split:5s}: partial_r={r:+.5f}  SE~{se:.5f}  "
                  f"|r|/SE={abs(r) / se if se else float('nan'):.2f}  n={n:,}")

    # ---- GATE RULE, applied exactly as frozen in this module's docstring ----
    primary = static_stats[LAG]
    r_train, se_train = primary["train"]["r"], primary["train"]["se"]
    r_valid, se_valid = primary["valid"]["r"], primary["valid"]["se"]
    roll_train_sign = np.sign(rolling_stats["train"]["mean"])
    roll_valid_sign = np.sign(rolling_stats["valid"]["mean"])
    static_sign_stable = np.sign(r_train) == np.sign(r_valid) != 0
    rolling_sign_stable = roll_train_sign == roll_valid_sign != 0
    sign_stable = bool(static_sign_stable and rolling_sign_stable)

    clears_train = abs(r_train) > 3.0 * se_train
    clears_valid = abs(r_valid) > 3.0 * se_valid
    distinguishable_from_zero = bool(clears_train and clears_valid)

    primary_sign = np.sign(r_train) if r_train else 0
    agree = sum(1 for lag in ROBUSTNESS_LAGS
                if np.sign(static_stats[lag]["train"]["r"]) == primary_sign and primary_sign != 0)
    robust_majority = bool(agree >= 4)

    gate_pass = bool(sign_stable and distinguishable_from_zero and robust_majority)

    print("\n" + "=" * 88)
    print("PRE-REGISTERED GATE RULE (frozen before this data was read):")
    print("  (a) sign(train) == sign(valid), primary rolling AND static statistic")
    print("  (b) |static partial r| > 3x its own SE, in BOTH splits (not just labeled a sign)")
    print("  (c) >=4/6 robustness lags {1,2,3,5,10,20} agree in sign with the primary lag (train)")
    print("=" * 88)
    print(f"  (a) sign-stable:              static {'PASS' if static_sign_stable else 'FAIL'} "
          f"(train sign={np.sign(r_train):+.0f}, valid sign={np.sign(r_valid):+.0f})  "
          f"rolling {'PASS' if rolling_sign_stable else 'FAIL'} "
          f"(train sign={roll_train_sign:+.0f}, valid sign={roll_valid_sign:+.0f})")
    print(f"  (b) distinguishable-from-zero: {'PASS' if distinguishable_from_zero else 'FAIL'}  "
          f"train |r|/SE={abs(r_train) / se_train:.2f} (need >3)  "
          f"valid |r|/SE={abs(r_valid) / se_valid:.2f} (need >3)")
    print(f"  (c) robustness majority:      {'PASS' if robust_majority else 'FAIL'}  "
          f"{agree}/6 lags agree with primary train sign")
    print(f"\n  GATE: {'PASS -> may proceed to a sweep' if gate_pass else 'FAIL -> STOP, per the pre-registered rule'}")
    print("=" * 88)

    return {
        "rolling_stats": rolling_stats, "static_stats": static_stats,
        "gate_pass": gate_pass, "sign_stable": sign_stable,
        "distinguishable_from_zero": distinguishable_from_zero,
        "robust_majority": robust_majority,
    }


# ------------------------------------------------------------------ the vote


def _anchor_votes(close: pd.Series, horizons=(20, 40, 80), band: float = 0.01) -> dict:
    """Exactly kelly_regime_v4's own per-anchor latched vote. Duplicated
    (not imported) from kelly_regime_v14_macro_lead.py / v16 -- see this
    module's "Code reuse decision"."""
    votes = {}
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes[days] = v.ffill().fillna(0.0)
    return votes


def _lead_vote(bt_ret: pd.Series, cb_ret: pd.Series, thresh_hi: float, gap: float,
               window: int = ROLL_WINDOW, lag: int = LAG) -> tuple[pd.Series, pd.Series]:
    """Latched ACTIVE flag (hysteresis on |rolling partial corr|) and a
    directional vote (sign of coinbase's own lagged return) -- both fully
    causal by construction (the rolling statistic and ``cb_ret.shift(lag)``
    never read a coinbase value at or after the row being decided).
    ``active`` defaults to 0 (inactive) wherever the statistic is NaN
    (insufficient warmup) -- absence never manufactures a vote, matching
    every prior confirming-vote signal's convention."""
    stat = rolling_leadlag_partial_corr(bt_ret, cb_ret, window=window, lag=lag).abs()
    thresh_lo = thresh_hi - gap
    raw = np.where(stat > thresh_hi, 1.0, np.where(stat < thresh_lo, 0.0, np.nan))
    active = pd.Series(raw, index=stat.index).ffill().fillna(0.0)
    cb_lag = cb_ret.shift(lag)
    direction = pd.Series(np.where(cb_lag > 0, 1.0, np.where(cb_lag < 0, 0.0, 0.5)),
                           index=cb_ret.index)
    return active, direction


class KellyRegimeV20CrossVenueLeadLag(KellyRegimeV3):
    """v4's 3-anchor vote plus a 4th, precision-weighted confirming vote
    from cross-venue (Coinbase-leads-Bitstamp) lead-lag structure:
    ``frac = (anchor_sum + eff_weight*direction)/(3+eff_weight)`` where
    ``eff_weight = weight * active`` -- ``active`` is a latched 0/1 flag on
    whether the rolling partial correlation of ``coinbase_return[t-1]``
    against ``bitstamp_return[t]`` (controlling for
    ``bitstamp_return[t-1]``) currently clears a threshold, and
    ``direction`` is the sign of Coinbase's own most recent LAGGED return.
    ``eff_weight=0`` (either ``weight=0`` or the statistic never crosses
    ``thresh_hi``) recovers v4 exactly -- the identity check verified in
    ``causality()``. NOTE: per this round's pre-registered gate (see
    ``descriptive()``), this class is written for completeness and for the
    causality tamper probe to have something concrete to test -- it is
    never swept against a real backtest with a frozen configuration,
    because the gate the mechanism depends on did not clear.
    """

    name = "_r77_kelly_regime_v20_crossvenue_leadlag"  # NOT @register-ed, experiment-local
    warmup = 80 * BARS_PER_DAY + ROLL_WINDOW + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 thresh_hi: float = 0.02, gap: float = 0.01, weight: float = 0.33,
                 window: int = ROLL_WINDOW, lag: int = LAG, **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.thresh_hi = thresh_hi
        self.gap = gap
        self.weight = weight
        self.window = window
        self.lag = lag

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = _anchor_votes(close, self.horizons, self.band)
        anchor_sum = sum(votes.values())
        n_anchors = float(len(votes))

        if "cb_ret" in df.columns and "bt_ret" in df.columns:
            bt_ret, cb_ret = df["bt_ret"], df["cb_ret"]
        else:
            coinbase_close = df["close"] / np.exp(df["divergence"])
            bt_ret = np.log(df["close"]).diff()
            cb_ret = np.log(coinbase_close).diff()

        active, direction = _lead_vote(bt_ret, cb_ret, self.thresh_hi, self.gap,
                                        window=self.window, lag=self.lag)
        eff_weight = self.weight * active

        combined = (anchor_sum + eff_weight * direction) / (n_anchors + eff_weight)
        frac = combined.to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

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
        vstate = 0
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if vstate == 0:
                    vstate = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif vstate == 1 and x < self.high_out:
                    vstate = 0
                elif vstate == -1 and x > self.low_out:
                    vstate = 0
            scale = full[i] if vstate != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["v20_frac"] = frac
        df["v20_active"] = active.to_numpy()
        df["v20_direction"] = direction.to_numpy()
        df["v20_anchor_sum"] = anchor_sum.to_numpy()
        return df


# --------------------------------------------------------------- gated steps


PRIMARY_KW = dict(thresh_hi=0.02, gap=0.01, weight=0.33)
THRESH_GAP = (
    ("tight", 0.015, 0.010),
    ("primary", 0.020, 0.010),
    ("loose", 0.030, 0.015),
)
WEIGHTS = (0.15, 0.33, 0.5, 1.0)

_GATE_CACHE: dict | None = None


def _gate() -> dict:
    global _GATE_CACHE
    if _GATE_CACHE is None:
        _GATE_CACHE = descriptive()
    return _GATE_CACHE


def _grid():
    out = [("identity (weight=0)", dict(thresh_hi=0.02, gap=0.01, weight=0.0))]
    for tg_label, thresh_hi, gap in THRESH_GAP:
        for weight in WEIGHTS:
            out.append((f"{tg_label} (thresh={thresh_hi:.3f} gap={gap:.3f}) w={weight:.2f}",
                        dict(thresh_hi=thresh_hi, gap=gap, weight=weight)))
    return out


def _config_key(kw: dict) -> str:
    return f"leadlag|thresh={kw['thresh_hi']:.4f}|gap={kw['gap']:.4f}|w={kw['weight']:.3f}"


def sweep() -> None:
    """Step 3: inner-train, only if the pre-registered gate passes."""
    gate = _gate()
    if not gate["gate_pass"]:
        print("\nGATE FAILS (see descriptive() above) -- per the pre-registered rule, "
              "NO strategy configuration is swept. 0 configurations evaluated in this step.")
        return
    print(f"\nINNER-TRAIN {TRAIN} / spot -- benchmarks:")
    for name in ("buy_and_hold", INCUMBENT):
        m, _ = measure(get_strategy(name), *TRAIN, market=SPOT)
        line(name, m, "spot")
    print(f"\nINNER-TRAIN {TRAIN} / spot -- candidate configurations:")
    for label, kw in _grid():
        m, _ = measure(KellyRegimeV20CrossVenueLeadLag(**kw), *TRAIN, market=SPOT,
                        config_key=_config_key(kw))
        line(label, m, "spot")
    print(f"\nconfigurations evaluated (step 3): {N_EVALUATED}")


def select():
    """Inner-validation, only if the pre-registered gate passes."""
    gate = _gate()
    if not gate["gate_pass"]:
        print("\nGATE FAILS (see descriptive() above) -- per the pre-registered rule, "
              "NO strategy configuration is selected. 0 configurations evaluated in this step.")
        return None, None
    ctl = {}
    for mname, market in MARKETS:
        m, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        ctl[mname] = m
    best_label, best_kw, best_score = None, None, -1e9
    for label, kw in _grid():
        cell = {}
        for mname, market in MARKETS:
            m, _ = measure(KellyRegimeV20CrossVenueLeadLag(**kw), *VALID, market=market,
                            config_key=_config_key(kw))
            cell[mname] = m
        m_train, _ = measure(KellyRegimeV20CrossVenueLeadLag(**kw), *TRAIN, market=SPOT)
        score = min(m_train.sharpe, cell["spot"].sharpe)
        if score > best_score:
            best_label, best_kw, best_score = label, kw, score
    print(f"best by min(train,valid) spot Sharpe: {best_label} (score={best_score:.2f})")
    print(f"v4 control spot Sharpe: {ctl['spot'].sharpe:.2f}  futures Sharpe: {ctl['futures 5x'].sharpe:.2f}")
    return best_label, best_kw


def artifact(kw: dict | None = None) -> None:
    """Exposure-artifact R^2 check -- only meaningful if the gate passed
    and a configuration was actually selected."""
    gate = _gate()
    if not gate["gate_pass"]:
        print("\nGATE FAILS -- no configuration to check for an exposure artifact "
              "(the round never reaches a sized strategy). Skipped.")
        return
    kw = kw or PRIMARY_KW
    v4 = get_strategy(INCUMBENT)
    for mname, market in MARKETS:
        cand = KellyRegimeV20CrossVenueLeadLag(**kw)
        lo = int(DF.index.searchsorted(VALID[0]))
        hi = int(DF.index.searchsorted(VALID[1], side="right"))
        prefix = min(lo, max(cand.warmup, v4.warmup))
        frame = DF.iloc[lo - prefix:hi]
        v4_t = v4.prepare(frame.copy())["target"].to_numpy()[prefix:]
        cand_t = cand.prepare(frame.copy())["target"].to_numpy()[prefix:]
        mean_abs_v4 = float(np.mean(np.abs(v4_t)))
        mean_abs_cand = float(np.mean(np.abs(cand_t)))
        alpha = mean_abs_cand / mean_abs_v4 if mean_abs_v4 > 0 else 0.0
        rescaled = alpha * v4_t
        ss_res = float(np.sum((cand_t - rescaled) ** 2))
        ss_tot = float(np.sum((cand_t - cand_t.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  {mname:10s} R^2(cand vs alpha*v4)={r2:.4f}  "
              f"{'EXPOSURE-LEVEL ARTIFACT' if r2 > 0.95 else 'genuinely different exposure shape'}")


def holdout() -> None:
    """Only if BOTH pre-registered gates clear. They do not -- see
    descriptive(). This function refuses to read OOS_START or later."""
    gate = _gate()
    if not gate["gate_pass"]:
        print("\nGATE FAILS at descriptive() -- per the pre-registered decision rule "
              "(step 4 of this round's brief), the 2023+ holdout is NEVER read. "
              "Nothing further to report; this is the round's stopping point.")
        return
    print("\n(unreachable in this round: the gate never passed)")


# ------------------------------------------------------------------ causality


def causality() -> None:
    """Dedicated lookahead probe, three parts:

    (1) Tamper probe on the two independent pathways (Bitstamp OHLCV, and
        the reconstructed-Coinbase-via-divergence pathway) -- every
        decision at or before a cut bar must be unchanged when only bars
        from the cut onward are tampered, checked both via the strategy's
        QUEUED ORDERS (the R-21-grade check: on_bar, not just prepare's
        columns) and via the prepared feature columns directly.
    (2) Truncation invariance: the rolling lead-lag statistic computed on
        a frame truncated at row i+1 must exactly equal the same
        statistic's value at row i computed on the full frame -- a pure
        function of bars <= i, by construction of the rolling-sum formula,
        verified numerically rather than merely argued.
    (3) A synthetic series with a KNOWN lag structure (Coinbase genuinely
        leads Bitstamp by exactly LAG bars, by construction) -- verifies
        the statistic actually recovers a strong, correctly-signed partial
        correlation when a real lead-lag relationship is deliberately
        injected, so a green suite is evidence the statistic can detect
        something, not merely evidence nothing was checked.
    """
    print("=" * 88)
    print("(1) TAMPER PROBE -- Bitstamp OHLCV and reconstructed-Coinbase pathway")
    print("=" * 88)
    pre = DF.iloc[-300_000:].copy()
    cut = len(pre) - 5_000
    bars_to_check = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    def decisions(frame, kw):
        s = KellyRegimeV20CrossVenueLeadLag(**kw)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars_to_check:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    def run_probe(name, tamper_fn):
        up, down = pre.copy(), pre.copy()
        tamper_fn(up, down)
        a = decisions(up, PRIMARY_KW)
        b = decisions(down, PRIMARY_KW)
        bad = [bar for bar, oa, ob in zip(bars_to_check, a, b) if oa != ob]
        print(f"[{name}] tampered from bar {cut:,} of {len(pre):,}; checked bars {bars_to_check}")
        print("  FAIL - reads the future at bars " + str(bad) if bad
              else "  PASS - every decision at or before the cut is unchanged")

        pa = KellyRegimeV20CrossVenueLeadLag(**PRIMARY_KW).prepare(up.copy())
        pb = KellyRegimeV20CrossVenueLeadLag(**PRIMARY_KW).prepare(down.copy())
        for col in ("target", "v20_frac", "v20_active", "v20_direction"):
            diff = np.abs(pa[col].to_numpy()[:cut].astype(float)
                          - pb[col].to_numpy()[:cut].astype(float))
            worst = float(np.nanmax(diff))
            print(f"    column {col:14s} max |difference| before the cut = {worst:.3e}"
                  f"  {'PASS' if worst < 1e-9 else 'FAIL'}")

    def tamper_ohlcv(up, down):
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0
        # divergence (and therefore the reconstructed Coinbase pathway)
        # tampered independently of the OHLCV tamper above
        up.iloc[cut:, up.columns.get_loc("divergence")] += 5.0
        down.iloc[cut:, down.columns.get_loc("divergence")] -= 5.0
        for f in (up, down):
            if "cb_ret" in f.columns:
                f.drop(columns=["cb_ret", "bt_ret", "coinbase_close"], inplace=True, errors="ignore")

    def tamper_divergence_only(up, down):
        up.iloc[cut:, up.columns.get_loc("divergence")] += 8.0
        down.iloc[cut:, down.columns.get_loc("divergence")] -= 8.0
        for f in (up, down):
            if "cb_ret" in f.columns:
                f.drop(columns=["cb_ret", "bt_ret", "coinbase_close"], inplace=True, errors="ignore")

    run_probe("Bitstamp OHLCV + reconstructed Coinbase (both tampered)", tamper_ohlcv)
    run_probe("divergence-only (isolates the Coinbase-reconstruction pathway)",
              tamper_divergence_only)

    print("\n" + "=" * 88)
    print("(2) TRUNCATION INVARIANCE -- rolling statistic at row i must be a pure "
          "function of rows <= i")
    print("=" * 88)
    sub = DF.iloc[-50_000:].copy()
    full_stat = rolling_leadlag_partial_corr(sub["bt_ret"], sub["cb_ret"])
    check_positions = [10_000, 20_000, 30_000, 40_000, 49_000]
    all_pass = True
    for pos in check_positions:
        truncated = sub.iloc[: pos + 1]
        trunc_stat = rolling_leadlag_partial_corr(truncated["bt_ret"], truncated["cb_ret"])
        a = full_stat.iloc[pos]
        b = trunc_stat.iloc[-1]
        ok = (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9
        all_pass = all_pass and ok
        print(f"  row {pos:6d}: full-series value={a!r}  truncated-at-row value={b!r}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"  truncation invariance: {'PASS' if all_pass else 'FAIL'}")

    print("\n" + "=" * 88)
    print("(3) SYNTHETIC KNOWN-LAG-STRUCTURE SANITY CHECK -- does the statistic detect "
          "a real, injected lead when one exists?")
    print("=" * 88)
    rng = np.random.default_rng(77)
    n = 50_000
    idx = pd.date_range("2020-01-01", periods=n, freq="5min", tz="UTC")
    cb_shock = rng.normal(0, 1.0, n)
    noise_bt = rng.normal(0, 1.0, n)
    beta_true = 0.6
    # Bitstamp's return at t is driven partly by Coinbase's shock at t-LAG
    # (a genuine, deliberately injected lead) plus its own independent noise.
    bt_synth = np.zeros(n)
    bt_synth[LAG:] = beta_true * cb_shock[:-LAG] + noise_bt[LAG:]
    bt_synth[:LAG] = noise_bt[:LAG]
    synth = pd.DataFrame({"bt_ret": bt_synth, "cb_ret": cb_shock}, index=idx)
    stat = rolling_leadlag_partial_corr(synth["bt_ret"], synth["cb_ret"], window=2_000, lag=LAG)
    mean_r = float(stat.dropna().mean())
    detects = mean_r > 0.15  # should be large and unambiguously positive given beta_true=0.6
    print(f"  injected beta={beta_true}, recovered mean rolling partial r = {mean_r:+.4f}  "
          f"{'PASS - statistic detects the injected lead' if detects else 'FAIL - statistic is not sensitive'}")

    # identity check: weight=0 recovers v4 exactly
    v4 = get_strategy(INCUMBENT)
    ident = KellyRegimeV20CrossVenueLeadLag(thresh_hi=0.02, gap=0.01, weight=0.0)
    frame = DF.iloc[-20_000:].copy()
    t_v4 = v4.prepare(frame.copy())["target"].to_numpy()
    t_ident = ident.prepare(frame.copy())["target"].to_numpy()
    worst = float(np.nanmax(np.abs(t_v4 - t_ident)))
    print(f"\nidentity check (weight=0 recovers v4 exactly): max|diff|={worst:.3e}  "
          f"{'PASS' if worst < 1e-9 else 'FAIL'}")


# --------------------------------------------------------------------- driver


def all_checks() -> None:
    print("=" * 88)
    print("STEP 2b -- descriptive: THE pre-registered falsification gate")
    print("=" * 88)
    gate = _gate()
    print("\n" + "=" * 88)
    print("STEP 3 -- sweep (inner-train, gated)")
    print("=" * 88)
    sweep()
    print("\n" + "=" * 88)
    print("STEP 3 -- select (inner-validation, gated)")
    print("=" * 88)
    select()
    print("\n" + "=" * 88)
    print("EXPOSURE-ARTIFACT CHECK (gated)")
    print("=" * 88)
    artifact()
    print("\n" + "=" * 88)
    print("CAUSALITY / NO-LOOKAHEAD PROBE (unconditional)")
    print("=" * 88)
    causality()
    print("\n" + "=" * 88)
    print("HOLDOUT (gated)")
    print("=" * 88)
    holdout()
    print(f"\ntotal distinct configurations evaluated (N_EVALUATED): {N_EVALUATED}")
    print(f"GATE RESULT: {'PASS' if gate['gate_pass'] else 'FAIL'}")


if __name__ == "__main__":
    cmds = {"descriptive": descriptive, "sweep": sweep, "select": select,
            "artifact": artifact, "causality": causality, "holdout": holdout,
            "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/r77_novel_crossvenue_leadlag.py [{'|'.join(cmds)}]")
