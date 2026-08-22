#!/usr/bin/env python
"""R-93 NOVEL branch: HEDGE-BLENDED GROSSMAN-ZHOU POPULATION, replacing
`kelly_regime_v4`'s conditional-vol-target `scale` with an ONLINE
multiplicative-weights (Hedge) blend of five Grossman & Zhou (1993)
drawdown-constrained sizing curves, run at different drawdown tolerances
alpha, while leaving v4's validated 3-anchor `frac` vote untouched.

CITATIONS. Grossman, S. J., & Zhou, Z. (1993), "Optimal Investment
Strategies for Controlling Drawdowns," Mathematical Finance, 3(3), 241-276
(the single-alpha sizing curve, `r93_shared.scale_gz`). Klass, M. J., &
Nowicki, K. (2005), "The Grossman-Zhou investment strategy outperforms
its offspring," Statistics and Probability Letters, 74(3), 245-256 (shows
the GZ rule is provably NOT growth-optimal in DISCRETE time -- only in the
continuous-time limit the original paper assumes -- so any one fixed alpha
chosen up front is a discrete-time approximation of unknown quality, not a
theorem). Freund, Y., & Schapire, R. E. (1997), "A Decision-Theoretic
Generalization of On-Line Learning and an Application to Boosting," Journal
of Computer and System Sciences, 55(1), 119-139 (the Hedge / exponential-
weights algorithm used below to blend the population online).

MECHANISM, in one sentence. Because Klass & Nowicki show no single alpha is
provably best in discrete time, maintain a small population of Grossman-Zhou
curves at different alpha (drawdown-tolerance) values, each bookkept on its
OWN causal, shadow (never-traded) equity path, and blend their CURRENT scale
outputs online via a Hedge/multiplicative-weights recursion driven by each
member's own realized shadow log-return -- trading only the single blended
result through ONE shared order stream, so turnover never scales with the
number of population members (the specific failure this design avoids: an
earlier blended council of independent-trading experts, `hedge_experts` /
`game_council` -- ledger L-09/L-17 -- had turnover scale with expert count
and the fee bill ate the edge).

WHICH CONSTRAINT IT ATTACKS: SIZE (same axis as the conservative sibling
and `r93_shared`'s module docstring; see that file for the full non-
duplication argument against R-38/R-46/R-59/R-60/R-62/R-87/R-45). This
branch's OWN distinguishing claim, not shared with the conservative
sibling's single fixed-alpha candidate: committing to one alpha ahead of
time is exactly the choice Klass & Nowicki (2005) show has no discrete-time
optimality guarantee, so an adaptive population blend is the more general
test of the same underlying mechanism, and stands or falls on whether that
adaptivity is worth its own switching cost -- a genuinely different
empirical question than "does one particular alpha work," not a re-run of
it.

--------------------------------------------------------------------------
DESIGN, DISCLOSED BEFORE ANY NUMBER BELOW IS READ
--------------------------------------------------------------------------
(1) POPULATION. alpha in {0.15, 0.20, 0.30, 0.40, 0.50} (K=5), the same
    grid as the conservative branch, for direct comparability. Each member
    uses v4's own max_leverage (2.0) and v4's own vote `frac` -- members
    differ ONLY in alpha.

(2) SHADOW BOOKKEEPING, THE SIMPLIFICATION NAMED UP FRONT. Each member's
    "own equity path" is a purely causal, hypothetical bookkeeping series,
    NEVER an order stream: it does not touch the real broker account and
    generates no real fills. Per member k, per bar t (t=1..n-1, bar 0
    starts at a fresh peak with the shadow balance):

        bar_ret_t      = close_t / close_(t-1) - 1
        fee_t          = SHADOW_FEE_RATE * |target_k[t-1] - target_k[t-2]|
                         * shadow_equity_k[t-1]
        shadow_equity_k[t] = shadow_equity_k[t-1] * (1 + target_k[t-1] * bar_ret_t) - fee_t
        scale_k[t]     = scale_gz(shadow_equity_k[0..t], alpha_k, max_leverage)   [r93_shared, applied to THIS member's own equity]
        target_k[t]    = v4_vote_frac[t] * scale_k[t]

    i.e. exactly the "member_target_(t-1) * bar_return_t - fee_on_change"
    practitioner shortcut the task names, made concrete: the fee is
    notional turnover cost on the position CHANGE made when target_k[t-1]
    was established, charged against the bar it earns its return on. This
    is a defensible approximation of "this member's own equity path"
    because it captures the two things that matter for the Hedge
    recursion -- the member's own realized P&L path (drift + vol as their
    own drawdown-constrained sizing would have produced it) and its own
    turnover cost -- without needing five independent broker accounts.
    SHADOW_FEE_RATE is fixed at SPOT's own 0.10% taker (`r93_shared.SPOT.
    fee_rate`) for ALL shadow bookkeeping regardless of which real market
    the blended output is later traded on; this is a second disclosed
    simplification (the weighting dynamics only need a REASONABLE,
    consistent turnover cost across members to differentiate low-alpha
    (frequent de-lever/re-lever, more fee drag) from high-alpha (steadier,
    less fee drag) members -- they do not need to match the real market's
    own fee exactly, since no shadow path is ever actually paying that
    fee).

    Critically, and unlike `r93_shared.GZScaledKellyV4`'s CONTROL wiring
    (which needs the REAL account's `ctx.equity`, live, because it cannot
    be vectorized ahead of the backtest that produces it -- see that
    module's docstring), every shadow path here is a pure, deterministic
    function of PRICE and the vote alone: it does not depend on the real
    account, so the entire population -- and everything built from it,
    including the Hedge weights and the final traded target -- CAN be
    vectorized once in `prepare()`. This is a real structural difference
    from the reference wiring, not an oversight, and is why this file's
    `HedgeBlendedGZKellyV4` is architecturally a `TargetStrategy` subclass
    (precomputed `target` column) rather than a live `on_bar` `ctx.equity`
    consumer.

(3) HEDGE WEIGHTS, DAILY CADENCE. `eta` is swept in {0.01, 0.05, 0.1}
    (log-return units). The multiplicative-weights update is done once per
    CALENDAR DAY (not per 5-minute bar) for numerical stability -- five-
    minute log-returns are dominated by noise relative to any of these
    alpha curves' actual signal, and updating 288x/day would mean `eta` in
    per-bar units is a different, much smaller number with no natural
    scale; daily is this project's own standard aggregation cadence
    (`BARS_PER_DAY = 288`, used throughout `r93_shared`). Recursion, in
    log-weight space for numerical stability at large eta:

        logw_k(0)  = 0                                  (day 0: uniform)
        logw_k(d)  = logw_k(d-1) + eta * r_k(d-1)        (d = 1, 2, ...)
        w_k(d)     = softmax_k(logw_k(d))

    where `r_k(d-1)` is member k's own shadow equity's REALIZED daily log
    return over calendar day d-1 (`log(1 + pct_change of that day's last
    shadow-equity value)`), so `w(d)` -- valid for every bar inside
    calendar day d -- depends only on information available at day d-1's
    close: causal by construction, verified below with a truncation probe
    on real BTC data.

    "Risk-adjusted P&L," named in the task's mechanism sentence: this
    branch does NOT apply a second, separate Sharpe-like normalization on
    top of `r_k`. Each member's raw shadow log-return already reflects
    ITS OWN drawdown-constrained (alpha-scaled) exposure -- a low-alpha
    member is already smaller and steadier by construction -- so the risk
    adjustment is structural (built into what is being measured), not a
    second normalization layered on the Hedge input. Disclosed as the
    interpretation used, not a hidden design choice.

(4) TRADED SCALE. `scale_blend(t) = sum_k w_k(t) * scale_k(t)` -- each
    member's OWN current-bar GZ scale (from ITS OWN shadow equity),
    blended by weight; NOT a blended alpha fed through one shared
    `scale_gz` call. `desired(t) = v4_vote_frac(t) * scale_blend(t)`,
    then v4's own 10% deadband (`r93_shared.apply_deadband`) -- identical
    order mechanics to v4 and to the conservative sibling's
    `GZScaledKellyV4`. Only THIS single blended path is ever traded; the
    five members never place an order.

(5) TWO STRUCTURAL LIMIT CHECKS, sanity checks on this file's OWN
    bookkeeping code, NOT scored candidates, NOT run through `compare()`:
      - eta -> 0: the softmax is flat regardless of `r_k` (exp(0)=1 for
        every member, every day), so `scale_blend` must equal the plain
        arithmetic mean of the five members' own `scale_k` series. Checked
        by literally calling the same `hedge_weights_over_bars` machinery
        with eta=0.0 and comparing to a directly-computed uniform mean --
        proves the general-case code collapses correctly at the boundary
        rather than hiding behind a special-cased branch.
      - eta -> infinity: the softmax collapses to a one-hot vector on
        whichever member had the single best PRIOR-day realized return
        (maximally reactive, deliberately degenerate). Checked by running
        the same machinery at a large finite eta (1e4, in log-weight
        space with max-subtraction for numerical stability, so this does
        not overflow) and comparing the resulting daily weight vectors to
        a directly-computed daily argmax.

--------------------------------------------------------------------------
FROZEN DECISION RULE (written before any real number below was read)
--------------------------------------------------------------------------
Step A (mechanism gate, before any performance number is read):
  A1 causality  -- a custom `causal_truncation_probe_hedge` (this file,
      not `r93_shared`'s, since the shadow/Hedge machinery is this
      branch's own construction): truncate real BTC data at cuts (0.35,
      0.55, 0.80), rebuild `build_target`, the shared prefix of
      `scale_mat`, `equity_mat`, daily weights and the final target array
      must match bit-for-bit; perturbing the tail must never move the
      prefix. Also exercised through one real, truncated-vs-full
      `run_backtest` (not only the synthetic/direct-array check), per the
      task's explicit instruction to run it "against a real equity/
      backtest, not just synthetic data."
  A2 sanity checks -- the eta->0 and eta->infinity limit checks above both
      pass numerically.

Step B (sweep): eta in {0.01, 0.05, 0.1}, each run through `r93_shared.
  compare()` -- 3 slices (inner_train, inner_val, eth_replication) x 2
  markets (spot, futures_5x) = 6 cells/eta x 3 eta = 18 cells minimum,
  every cell reported.

Step C (finalist selection, inner-train + inner-validation ONLY, holdout
  untouched): among the 3 eta values, select the one clearing, on
  inner-validation on BOTH markets:
      Delta Sharpe > +0.2   OR   a risk-matched drawdown improvement
      (exposure_ratio and vol_ratio both in [0.9, 1.1], per `r93_shared.
      compare()`'s own `risk_matched` flag).
  If more than one eta clears this bar, the finalist is the one with the
  larger mean inner-validation Delta Sharpe across the two markets. If
  NONE clears it, the eta with the best (least-bad) mean inner-validation
  Delta Sharpe is still carried forward as "finalist" ONLY so the
  pre-registered falsification/fee checks below can be run and reported in
  full (per the routine's "every branch reports, including the dead ones"
  rule) -- it is explicitly flagged as not having cleared Step C, and the
  branch cannot be a PROMOTE-CANDIDATE regardless of what follows.

Step D (PRE-REGISTERED FALSIFICATION TEST -- the one that matters most for
  this branch; not substituted after seeing results). The whole point of
  adaptive blending is that it should be MORE robust across regimes than a
  single fixed alpha. Because true Monte Carlo windows drawn from the
  README's published stress test span 2017-2026 and this branch may not
  touch any bar at or after OOS_START, this is a PRE-HOLDOUT-RESTRICTED
  adaptation of `scripts/stress_test.py`'s own methodology (same random
  window / warmup-prefix / trade_start construction), reimplemented in
  this file against `r93_shared.load_btc()` (already truncated <
  2023-01-01) rather than calling that script directly, which pulls the
  full untruncated dataset. Disclosed, not silently substituted.
  PRE-REGISTERED KILL CONDITION: if the finalist's survival / profitable-
  window behaviour on this pre-holdout stress test is WORSE than (a) v4's
  OWN behaviour measured identically, on the SAME window set (the fair,
  same-basis comparison this branch can actually run), the mechanism is
  DISQUALIFIED as a PROMOTE-CANDIDATE regardless of its inner-validation
  point estimate. The README's already-published full-history v4 number
  (survives all 40 windows, profitable in 85-88%) is ALSO reported for
  context, explicitly flagged as a different, larger window universe
  (spans through 2026) that this branch cannot reproduce without touching
  the holdout -- informative, not the primary kill trigger.

Step E (cost check): finalist re-run through `compare()` at a 0.40% taker
  (`r93_shared.fee_at(SPOT, 0.004)`), inner-validation, both markets --
  does the SIGN of the paired difference survive.

Promotion bar (PROMOTE-CANDIDATE vs NEGATIVE; this round cannot issue a
true PROMOTED verdict without a holdout run, which is out of scope here by
the task's own rule against touching OOS_START -- "PROMOTE-CANDIDATE"
means "clears every check this round could run and is worth a holdout
evaluation in a future round"): Step A passes, Step C's finalist actually
cleared Step C's own bar (not just "least-bad"), Step D's kill condition
does NOT fire, Step E's sign survives. Any failure -> NEGATIVE, written up
with the same care as a win, per the routine's own rule.

This file imports `r93_shared` (READ-ONLY, not edited) for `vote_frac`,
`v4_vote_frac`, `v4_target`, `scale_gz`, `running_drawdown`, `apply_
deadband`, `compare`, `print_rows`, `load_btc`, `load_eth`, `fee_at`,
`SPOT`, `FUTURES`, `V4_MAX_LEVERAGE`, `OOS_START`, and reuses rather than
duplicates its vote/drawdown/comparison machinery. Nothing in this file
reads a bar at or after OOS_START (2023-01-01); every load goes through
`r93_shared`'s truncating loaders and the maximum timestamp actually
touched anywhere is tracked and printed at the end of `main()`.
--------------------------------------------------------------------------
RESULTS -- filled in after the run below (this section is the actual
findings write-up; everything above was written before any number in this
section was read)
--------------------------------------------------------------------------
[[FILLED_AFTER_RUN]]
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import max_drawdown_pct  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
import tradebot.strategies  # noqa: E402,F401  (registers buy_and_hold / kelly_regime_v4)

from experiments.r93_shared import (  # noqa: E402
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SPOT,
    TargetStrategy,
    V4_MAX_LEVERAGE,
    apply_deadband,
    compare,
    fee_at,
    load_btc,
    load_eth,
    print_rows,
    scale_gz,
    v4_target,
    v4_vote_frac,
)

BARS_PER_DAY = 288
POPULATION_ALPHAS: tuple[float, ...] = (0.15, 0.20, 0.30, 0.40, 0.50)
K = len(POPULATION_ALPHAS)
ETA_GRID: tuple[float, ...] = (0.01, 0.05, 0.1)
ETA_INF_PROXY = 1.0e4  # large-but-finite stand-in for eta -> infinity
SHADOW_FEE_RATE = SPOT.fee_rate  # 0.10%, fixed for all shadow bookkeeping (disclosed above)
SHADOW_BALANCE = 1_000.0
HEDGE_WARMUP_DAYS = 80 + 30  # v4's own 80-day longest anchor, + 30 days for weights to warm
HEDGE_WARMUP_BARS = HEDGE_WARMUP_DAYS * BARS_PER_DAY + 10
TAKER_040 = 0.0040


# =============================================================== (1)
# population_scale_matrix: the K shadow paths, independent of eta. This is
# the expensive O(n) part; cached by frame identity so the eta sweep and
# the two sanity checks reuse it instead of recomputing.
# ===============================================================
_POP_CACHE: dict = {}
_TARGET_CACHE: dict = {}


def _key(df: pd.DataFrame) -> tuple:
    """Cache key for a frame's identity. Endpoints alone are NOT enough --
    the causal truncation probe deliberately builds several perturbed frames
    that share length/first-close/last-close (the tail multiplier lands on
    the same final value regardless of where the perturbation starts) but
    differ in the interior, so a content hash of the actual close array is
    required or the cache silently returns a stale result for a different
    frame (caught by the probe itself during development; fixed here, not
    worked around)."""
    close_bytes = np.ascontiguousarray(df["close"].to_numpy(dtype=float)).tobytes()
    digest = hashlib.blake2b(close_bytes, digest_size=16).hexdigest()
    return (len(df), int(df.index[0].value), int(df.index[-1].value), digest)


def population_scale_matrix(df: pd.DataFrame, alphas: tuple[float, ...] = POPULATION_ALPHAS,
                            max_leverage: float = V4_MAX_LEVERAGE,
                            fee_rate: float = SHADOW_FEE_RATE,
                            shadow_balance: float = SHADOW_BALANCE
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The K population members' own shadow equity/scale paths (design (2)).

    Returns (frac (n,), scale_mat (n,K), equity_mat (n,K)). Pure function of
    df["close"] and v4_vote_frac(df) -- no real account, no eta. Cached.
    """
    key = ("pop", _key(df), alphas, max_leverage, fee_rate, shadow_balance)
    if key in _POP_CACHE:
        return _POP_CACHE[key]

    frac = v4_vote_frac(df).to_numpy()
    close = df["close"].to_numpy(dtype=float)
    n = len(df)
    alphas_arr = np.asarray(alphas, dtype=float)

    equity_mat = np.empty((n, K), dtype=float)
    scale_mat = np.empty((n, K), dtype=float)

    equity = np.full(K, shadow_balance, dtype=float)
    peak = np.full(K, shadow_balance, dtype=float)
    d0 = np.zeros(K)
    scale0 = max_leverage * np.clip(1.0 - d0 / alphas_arr, 0.0, 1.0)
    target0 = frac[0] * scale0
    equity_mat[0] = equity
    scale_mat[0] = scale0
    target_prev = target0.copy()
    target_prevprev = np.zeros(K)

    for t in range(1, n):
        bar_ret = close[t] / close[t - 1] - 1.0
        fee = fee_rate * np.abs(target_prev - target_prevprev) * equity
        equity = equity * (1.0 + target_prev * bar_ret) - fee
        equity = np.maximum(equity, 1e-8)  # numerical guard, never binds at these alphas/leverage
        peak = np.maximum(peak, equity)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(peak > 0, 1.0 - equity / peak, 0.0)
        scale = max_leverage * np.clip(1.0 - d / alphas_arr, 0.0, 1.0)
        target = frac[t] * scale

        equity_mat[t] = equity
        scale_mat[t] = scale
        target_prevprev = target_prev
        target_prev = target

    _POP_CACHE[key] = (frac, scale_mat, equity_mat)
    return frac, scale_mat, equity_mat


# =============================================================== (2)
# Hedge weight recursion, daily cadence (design (3)).
# ===============================================================

def _daily_last_and_logret(df: pd.DataFrame, equity_mat: np.ndarray) -> tuple[pd.DatetimeIndex, np.ndarray]:
    dates = df.index.normalize()
    eq_df = pd.DataFrame(equity_mat, index=dates)
    daily_last = eq_df.groupby(level=0).last()
    simple = daily_last.pct_change()
    simple.iloc[0] = 0.0
    logret = np.log1p(simple.to_numpy())
    return daily_last.index, logret


def selftest_inline_formula_matches_shared(df: pd.DataFrame) -> tuple[bool, str]:
    """population_scale_matrix reimplements scale_gz's formula INLINE, for the
    same reason r93_shared.GZScaledKellyV4's own on_bar does (it needs a
    live, incremental peak/scale as the equity series grows, and calling the
    vectorized scale_gz() fresh at every bar would be O(n^2)) -- not a
    divergent reimplementation. Proved here, not just asserted: run the
    population loop once, then hand each member's OWN already-realized
    equity path to the ACTUAL r93_shared.scale_gz()/running_drawdown()
    functions and check they reproduce this file's inline scale bit-for-bit."""
    _, scale_mat, equity_mat = population_scale_matrix(df)
    ok = True
    worst = 0.0
    for k, alpha in enumerate(POPULATION_ALPHAS):
        eq = pd.Series(equity_mat[:, k], index=df.index)
        shared_scale = scale_gz(eq, alpha, V4_MAX_LEVERAGE).to_numpy()
        diff = float(np.max(np.abs(shared_scale - scale_mat[:, k])))
        worst = max(worst, diff)
        if diff > 1e-9:
            ok = False
    return ok, f"max|inline_scale - r93_shared.scale_gz(same realized equity path)| across all {K} members = {worst:.3e}"


def hedge_weights_over_bars(df: pd.DataFrame, equity_mat: np.ndarray, eta: float) -> np.ndarray:
    """Design (3): daily multiplicative-weights recursion, broadcast to every
    bar of its own calendar day. Causal: day d's weight uses only day d-1's
    (and earlier) realized shadow return."""
    dates, logret = _daily_last_and_logret(df, equity_mat)
    d_n = len(dates)
    daily_w = np.empty((d_n, K), dtype=float)
    logw = np.zeros(K)
    daily_w[0] = 1.0 / K
    for i in range(1, d_n):
        logw = logw + eta * logret[i - 1]
        logw = logw - logw.max()
        w = np.exp(logw)
        w = w / w.sum()
        daily_w[i] = w
        logw = np.log(np.clip(w, 1e-300, None))

    day_idx = pd.Series(np.arange(d_n), index=dates)
    bar_day_idx = day_idx.reindex(df.index.normalize()).to_numpy()
    return daily_w[bar_day_idx]


def uniform_weights_over_bars(n: int) -> np.ndarray:
    """eta -> 0 limit, computed directly (not via the recursion) for the sanity check."""
    return np.full((n, K), 1.0 / K)


def bestfollow_weights_over_bars(df: pd.DataFrame, equity_mat: np.ndarray) -> np.ndarray:
    """eta -> infinity limit, computed directly (argmax of the prior day's
    realized return) for the sanity check."""
    dates, logret = _daily_last_and_logret(df, equity_mat)
    d_n = len(dates)
    daily_w = np.zeros((d_n, K), dtype=float)
    daily_w[0] = 1.0 / K
    for i in range(1, d_n):
        best = int(np.argmax(logret[i - 1]))
        daily_w[i, best] = 1.0
    day_idx = pd.Series(np.arange(d_n), index=dates)
    bar_day_idx = day_idx.reindex(df.index.normalize()).to_numpy()
    return daily_w[bar_day_idx]


# =============================================================== (3)
# build_target: the assembled, traded path (design (4)).
# ===============================================================

def build_target(df: pd.DataFrame, eta: float, alphas: tuple[float, ...] = POPULATION_ALPHAS,
                 max_leverage: float = V4_MAX_LEVERAGE, fee_rate: float = SHADOW_FEE_RATE,
                 shadow_balance: float = SHADOW_BALANCE) -> np.ndarray:
    key = ("target", _key(df), eta, alphas, max_leverage, fee_rate, shadow_balance)
    if key in _TARGET_CACHE:
        return _TARGET_CACHE[key]
    frac, scale_mat, equity_mat = population_scale_matrix(df, alphas, max_leverage, fee_rate, shadow_balance)
    weights = hedge_weights_over_bars(df, equity_mat, eta)
    scale_blend = np.sum(weights * scale_mat, axis=1)
    desired = frac * scale_blend
    target = apply_deadband(desired)
    _TARGET_CACHE[key] = target
    return target


class HedgeBlendedGZKellyV4(TargetStrategy):
    """kelly_regime_v4 with `scale` replaced by an online Hedge blend of five
    Grossman & Zhou (1993) drawdown-constrained curves (alpha in
    {0.15,0.20,0.30,0.40,0.50}), weighted by each curve's own realized causal
    shadow log-return via a Freund & Schapire (1997) multiplicative-weights
    recursion at daily cadence. Only the blended output ever trades.
    """

    def __init__(self, eta: float, alphas: tuple[float, ...] = POPULATION_ALPHAS,
                max_leverage: float = V4_MAX_LEVERAGE, fee_rate: float = SHADOW_FEE_RATE,
                shadow_balance: float = SHADOW_BALANCE) -> None:
        self.eta = eta
        super().__init__(
            build_target=lambda df: build_target(df, eta, alphas, max_leverage, fee_rate, shadow_balance),
            name=f"r93_novel_hedge_gz_blend_eta{eta:g}",
            warmup=HEDGE_WARMUP_BARS,
        )


# =============================================================== (4)
# A1: causal truncation probe, custom to this branch's own machinery, run
# on REAL BTC data (not only synthetic), plus one real truncated-vs-full
# run_backtest check.
# ===============================================================

def causal_truncation_probe_hedge(df: pd.DataFrame, eta: float,
                                  cuts: tuple[float, ...] = (0.35, 0.55, 0.80)) -> tuple[bool, list[str]]:
    msgs = []
    ok = True
    full_target = build_target(df, eta)
    for cut in cuts:
        k = int(len(df) * cut)
        if k < 2:
            continue
        part_df = df.iloc[:k]
        part_target = build_target(part_df, eta)
        if not np.allclose(full_target[:k], part_target, atol=1e-9, rtol=0.0):
            bad = int(np.sum(~np.isclose(full_target[:k], part_target, atol=1e-9, rtol=0.0)))
            ok = False
            msgs.append(f"cut={cut}: FAIL, {bad}/{k} target bars differ")
            continue
        # tail perturbation: must not move the shared prefix
        perturbed = df.copy()
        tail_close = perturbed["close"].to_numpy().copy()
        tail_close[k:] = tail_close[k:] * 1e6 + 1e9
        perturbed["close"] = tail_close
        perturbed["open"] = tail_close
        perturbed["high"] = tail_close * 1.0001
        perturbed["low"] = tail_close * 0.9999
        pert_target = build_target(perturbed, eta)
        if not np.allclose(pert_target[:k], full_target[:k], atol=1e-9, rtol=0.0):
            ok = False
            msgs.append(f"cut={cut}: FAIL, perturbing bars>={k} moved the prefix")
            continue
        msgs.append(f"cut={cut}: PASS (prefix match, tail-perturbation invariant)")
    return ok, msgs


def causal_truncation_probe_realbacktest(df: pd.DataFrame, eta: float,
                                         cut: float = 0.6) -> tuple[bool, str]:
    """A1, second leg: run the ACTUAL Strategy through run_backtest on the
    full frame and on a truncated frame, and verify the shared prefix of the
    resulting `target` column (what really drove orders) matches exactly --
    exercised against a real equity/backtest, not just the direct-array check
    above."""
    strat = HedgeBlendedGZKellyV4(eta=eta)
    n = len(df)
    k = int(n * cut)
    full_res = run_backtest(HedgeBlendedGZKellyV4(eta=eta), df, SPOT, 1_000.0)
    part_res = run_backtest(strat, df.iloc[:k], SPOT, 1_000.0)
    full_t = full_res.df["target"].to_numpy()[:k]
    part_t = part_res.df["target"].to_numpy()
    ok = bool(np.allclose(full_t, part_t, atol=1e-9, rtol=0.0)) and not full_res.liquidated \
        and not part_res.liquidated
    msg = (f"cut={cut}: full-vs-truncated real run_backtest target prefix match="
          f"{np.allclose(full_t, part_t, atol=1e-9, rtol=0.0)}, "
          f"liquidated(full)={full_res.liquidated}, liquidated(part)={part_res.liquidated}")
    return ok, msg


# =============================================================== (5)
# A2: eta->0 / eta->infinity sanity checks (design (5)).
# ===============================================================

def sanity_eta_zero(df: pd.DataFrame) -> tuple[bool, str]:
    _, scale_mat, equity_mat = population_scale_matrix(df)
    w = hedge_weights_over_bars(df, equity_mat, eta=0.0)
    w_direct = uniform_weights_over_bars(len(df))
    blend_hedge = np.sum(w * scale_mat, axis=1)
    blend_direct = np.sum(w_direct * scale_mat, axis=1)
    ok = bool(np.allclose(blend_hedge, blend_direct, atol=1e-10, rtol=0.0))
    ok = ok and bool(np.allclose(w, 1.0 / K, atol=1e-12, rtol=0.0))
    max_abs_diff = float(np.max(np.abs(blend_hedge - blend_direct)))
    return ok, f"eta=0 blend vs directly-computed uniform mean: max|diff|={max_abs_diff:.3e}"


def sanity_eta_inf(df: pd.DataFrame) -> tuple[bool, str]:
    _, scale_mat, equity_mat = population_scale_matrix(df)
    w_large = hedge_weights_over_bars(df, equity_mat, eta=ETA_INF_PROXY)
    w_direct = bestfollow_weights_over_bars(df, equity_mat)
    # compare only the argmax member each day (softmax at eta=1e4 is one-hot
    # to far better than float precision whenever the daily returns differ
    # at all, but ties/exact-zero days are legitimately ambiguous)
    match = np.argmax(w_large, axis=1) == np.argmax(w_direct, axis=1)
    frac_match = float(match.mean())
    ok = frac_match > 0.999
    return ok, f"eta={ETA_INF_PROXY:g} argmax-member agreement with direct best-follow: {frac_match:.4%}"


# =============================================================== (6)
# Step D: pre-holdout-restricted Monte Carlo stress test, this branch's own
# reimplementation of scripts/stress_test.py's methodology (random window +
# warmup prefix + trade_start), scoped to r93_shared.load_btc() (already
# truncated < OOS_START) since the script itself pulls the full untruncated
# dataset. Compares the finalist against v4 and buy_and_hold on the SAME
# window set.
# ===============================================================

def evaluate_window(strategy, window: pd.DataFrame, eval_start: int, market: MarketSpec,
                    balance: float = 1_000.0) -> dict:
    result = run_backtest(strategy, window, market, balance, trade_start=eval_start)
    equity = result.equity.to_numpy(dtype=float)
    base = equity[eval_start]
    if not np.isfinite(base) or base <= 0:
        return {"return_pct": -100.0, "max_dd_pct": 100.0, "liquidated": True}
    seg = equity[eval_start:]
    return {
        "return_pct": 100.0 * (float(seg[-1]) / base - 1.0),
        "max_dd_pct": max_drawdown_pct(seg),
        "liquidated": bool(result.liquidated),
    }


def run_stress_test(finalist_eta: float, trials: int = 24, min_days: int = 60, max_days: int = 500,
                    markets: tuple[MarketSpec, ...] = (SPOT, FUTURES), seed: int = 93) -> pd.DataFrame:
    btc = load_btc()
    strategies = {
        "r93_novel_hedge_gz_blend": lambda: HedgeBlendedGZKellyV4(eta=finalist_eta),
        "kelly_regime_v4": lambda: TargetStrategy(v4_target, name="kelly_regime_v4"),
        "buy_and_hold": lambda: get_strategy("buy_and_hold"),
    }
    warmup = max(HEDGE_WARMUP_BARS, 23050) + 10
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(trials):
        length = int(rng.integers(min_days, max_days + 1) * BARS_PER_DAY)
        start = int(rng.integers(warmup, len(btc) - length))
        specs.append((start, length))

    rows = []
    t0 = time.time()
    for i, (start, length) in enumerate(specs, 1):
        window = btc.iloc[start - warmup: start + length]
        eval_start = warmup
        for market in markets:
            for name, factory in strategies.items():
                stats = evaluate_window(factory(), window, eval_start, market)
                rows.append({"trial": i, "market": market.name, "strategy": name,
                            "start": window.index[eval_start], "days": length // BARS_PER_DAY, **stats})
        print(f"  [stress {i}/{trials}] {window.index[eval_start]:%Y-%m-%d} +{length // BARS_PER_DAY}d "
              f"({time.time() - t0:.0f}s elapsed)", file=sys.stderr)
    return pd.DataFrame(rows)


def summarize_stress(res: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (market, name), grp in res.groupby(["market", "strategy"], sort=False):
        out.append({
            "market": market, "strategy": name,
            "median_return_pct": grp["return_pct"].median(),
            "profitable_pct": float((grp["return_pct"] > 0).mean() * 100.0),
            "worst_pct": grp["return_pct"].min(),
            "median_maxdd_pct": grp["max_dd_pct"].median(),
            "worst_maxdd_pct": grp["max_dd_pct"].max(),
            "liquidated_pct": float(grp["liquidated"].mean() * 100.0),
            "n_windows": len(grp),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------- misc

def hdr(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:
    max_ts = []
    hdr("R-93 NOVEL BRANCH -- HEDGE-BLENDED GROSSMAN-ZHOU POPULATION")
    print("mechanism: K=5 Grossman-Zhou (1993) drawdown-constrained sizing curves at alpha in")
    print(f"{POPULATION_ALPHAS}, each bookkept on its own causal shadow equity path, blended online via a")
    print("Freund & Schapire (1997) Hedge/multiplicative-weights recursion (daily cadence) driven by")
    print("each curve's own realized shadow log-return. Only the blended scale ever trades.")

    btc = load_btc()
    max_ts.append(btc.index.max())
    print(f"\nBTC: {len(btc):,} bars  {btc.index[0]} -> {btc.index[-1]}  (< {OOS_START})")
    eth = load_eth()
    max_ts.append(eth.index.max())
    print(f"ETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}")

    # ---- inline-formula consistency (proves the incremental loop reuses
    # r93_shared's own GZ formula rather than a divergent one) ------------
    hdr("INLINE-FORMULA CONSISTENCY CHECK (population_scale_matrix vs r93_shared.scale_gz)")
    consist_ok, consist_msg = selftest_inline_formula_matches_shared(btc.loc[:INNER_TRAIN_END])
    print(f"  {consist_msg}  -> {'PASS' if consist_ok else 'FAIL'}")

    # ---- Step A1: causality -------------------------------------------
    hdr("STEP A1 -- CAUSAL TRUNCATION PROBE (this branch's own machinery, real BTC data)")
    probe_eta = ETA_GRID[1]  # 0.05, representative -- the machinery is eta-agnostic by construction
    a1_direct_ok, a1_msgs = causal_truncation_probe_hedge(btc, probe_eta)
    for m in a1_msgs:
        print(f"  direct array probe -- {m}")
    print(f"  direct-array A1 leg: {'PASS' if a1_direct_ok else 'FAIL'}")

    a1_real_ok, a1_real_msg = causal_truncation_probe_realbacktest(btc, probe_eta)
    print(f"  real-backtest leg  -- {a1_real_msg}")
    print(f"  real-backtest A1 leg: {'PASS' if a1_real_ok else 'FAIL'}")
    a1_pass = a1_direct_ok and a1_real_ok
    print(f"A1 = {'PASS' if a1_pass else 'FAIL'}")

    # ---- Step A2: eta->0 / eta->infinity sanity checks -----------------
    hdr("STEP A2 -- SANITY CHECKS (eta->0 uniform blend, eta->infinity best-follow) -- NOT scored candidates")
    train = btc.loc[:INNER_TRAIN_END]
    zero_ok, zero_msg = sanity_eta_zero(train)
    print(f"  eta->0:        {zero_msg}  -> {'PASS' if zero_ok else 'FAIL'}")
    inf_ok, inf_msg = sanity_eta_inf(train)
    print(f"  eta->infinity: {inf_msg}  -> {'PASS' if inf_ok else 'FAIL'}")
    a2_pass = zero_ok and inf_ok
    print(f"A2 = {'PASS' if a2_pass else 'FAIL'}")

    gate_pass = consist_ok and a1_pass and a2_pass
    if not gate_pass:
        hdr("VERDICT")
        print("Step A (causality / sanity gate) FAILED -- reported NEGATIVE. No Step B sweep is run;")
        print("forcing a backtest sweep past a causality failure would not add information (the R-21 lesson).")
        print(f"\nConfigurations evaluated (backtested): 0.")
        print(f"\nmax timestamp read anywhere: {max(max_ts)}  (< {OOS_START})")
        return
    print("\nStep A passes -- proceeding to Step B (eta sweep).")

    # ---- Step B: eta sweep through compare() ----------------------------
    hdr("STEP B -- ETA SWEEP, 3 real configs x compare() (3 slices x 2 markets = 6 cells/eta)")
    all_rows: dict[float, list[dict]] = {}
    for eta in ETA_GRID:
        print(f"\n-- eta={eta} --")
        rows = compare(HedgeBlendedGZKellyV4(eta=eta), label=f"r93_novel_hedge_eta{eta:g}")
        print_rows(rows)
        all_rows[eta] = rows

    # ---- Step C: finalist selection (inner-train + inner-validation only) --
    hdr("STEP C -- FINALIST SELECTION (inner-validation only, holdout untouched)")
    selection = {}
    for eta, rows in all_rows.items():
        val = [r for r in rows if r["slice"] == "inner_val"]
        dsh = {r["market"]: r["d_sharpe"] for r in val}
        rm = {r["market"]: r["risk_matched"] for r in val}
        ddd = {r["market"]: r["d_dd"] for r in val}
        b2_sharpe = all(v > 0.2 for v in dsh.values())
        b2_dd = all(v < 0.0 for v in ddd.values()) and all(rm.values())
        clears = b2_sharpe or b2_dd
        mean_dsh = float(np.mean(list(dsh.values())))
        via = "Sharpe" if b2_sharpe else ("matched drawdown" if b2_dd else "neither")
        selection[eta] = dict(clears=clears, mean_dsh=mean_dsh, dsh=dsh, ddd=ddd, rm=rm, via=via)
        print(f"eta={eta:<5} dSharpe={dsh}  dMaxDD={ {k: round(v,2) for k,v in ddd.items()} }  "
              f"risk_matched={rm}  clears_step_C={clears} (via {selection[eta]['via']})")

    clearing = {e: s for e, s in selection.items() if s["clears"]}
    if clearing:
        finalist_eta = max(clearing, key=lambda e: clearing[e]["mean_dsh"])
        finalist_cleared = True
    else:
        finalist_eta = max(selection, key=lambda e: selection[e]["mean_dsh"])
        finalist_cleared = False
    print(f"\nFinalist: eta={finalist_eta}  (cleared Step C bar: {finalist_cleared})")

    # ---- Step D: pre-registered falsification (stress test) -------------
    hdr("STEP D -- PRE-REGISTERED FALSIFICATION: PRE-HOLDOUT MONTE CARLO STRESS WINDOWS")
    print(f"Running r93 novel's own scripts/stress_test.py-style reimplementation, scoped to load_btc()")
    print(f"(< {OOS_START}), against the finalist (eta={finalist_eta}), kelly_regime_v4 and buy_and_hold,")
    print(f"on the SAME window set (fair, same-basis comparison).")
    stress_res = run_stress_test(finalist_eta)
    stress_summary = summarize_stress(stress_res)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(stress_summary.round(1).to_string(index=False))

    def _stat(strategy, market, col):
        row = stress_summary[(stress_summary.strategy == strategy) & (stress_summary.market == market)]
        return float(row[col].iloc[0]) if len(row) else float("nan")

    kill_fired = False
    kill_details = []
    for market in ("spot", "futures_5x"):
        cand_prof = _stat("r93_novel_hedge_gz_blend", market, "profitable_pct")
        ctrl_prof = _stat("kelly_regime_v4", market, "profitable_pct")
        cand_liq = _stat("r93_novel_hedge_gz_blend", market, "liquidated_pct")
        ctrl_liq = _stat("kelly_regime_v4", market, "liquidated_pct")
        worse = (cand_prof < ctrl_prof) or (cand_liq > ctrl_liq)
        kill_details.append((market, cand_prof, ctrl_prof, cand_liq, ctrl_liq, worse))
        kill_fired = kill_fired or worse
        print(f"  {market:12s} candidate profitable%={cand_prof:.1f} vs v4 profitable%={ctrl_prof:.1f}  "
              f"| candidate liquidated%={cand_liq:.1f} vs v4 liquidated%={ctrl_liq:.1f}  "
              f"-> {'WORSE (kill fires)' if worse else 'not worse'}")
    print(f"\nREADME published (full 2017-2026 history, different/larger window universe, context only):")
    print(f"  kelly_regime_v4 survives all 40 stress windows, profitable in 85-88%.")
    print(f"\nSTEP D KILL CONDITION: {'FIRED -- DISQUALIFIED' if kill_fired else 'did not fire'}")

    # ---- Step E: cost check ----------------------------------------------
    hdr("STEP E -- COST CHECK: 0.40% TAKER, inner-validation")
    spot_040 = fee_at(SPOT, TAKER_040)
    fut_040 = fee_at(FUTURES, TAKER_040)
    fee_rows = compare(HedgeBlendedGZKellyV4(eta=finalist_eta), label=f"r93_novel_hedge_eta{finalist_eta:g}@40bp",
                       markets=(spot_040, fut_040), include_eth=False)
    fee_rows = [r for r in fee_rows if r["slice"] == "inner_val"]
    print_rows(fee_rows)
    base_val = {r["market"]: r["d_log_growth"]
               for r in all_rows[finalist_eta] if r["slice"] == "inner_val"}
    fee_ok = []
    for r in fee_rows:
        base_sign = np.sign(base_val.get(r["market"], 0.0))
        cand_sign = np.sign(r["d_log_growth"])
        same = bool(base_sign == cand_sign) if base_sign != 0 else True
        fee_ok.append(same)
        print(f"   {r['market']:14s} d_log_growth={r['d_log_growth']:+.4f}  base(0.10%) sign={base_sign:+.0f}  "
              f"same sign at 0.40%: {same}")
    fee_sign_survives = all(fee_ok)
    print(f"Step E = {'PASS (sign survives)' if fee_sign_survives else 'FAIL (sign reverses)'}")

    # ---- verdict ------------------------------------------------------
    hdr("VERDICT")
    print(f"Inline-formula consistency:         {'PASS' if consist_ok else 'FAIL'}")
    print(f"Step A (causality + sanity):        {'PASS' if gate_pass else 'FAIL'}")
    print(f"Step C (finalist cleared its bar):  {'PASS' if finalist_cleared else 'FAIL'} (eta={finalist_eta})")
    print(f"Step D (falsification kill cond.):  {'DID NOT FIRE (pass)' if not kill_fired else 'FIRED (fail)'}")
    print(f"Step E (0.40% taker sign):          {'PASS' if fee_sign_survives else 'FAIL'}")
    promote = gate_pass and finalist_cleared and (not kill_fired) and fee_sign_survives
    print(f"\nVERDICT: {'PROMOTE-CANDIDATE' if promote else 'NEGATIVE'}")

    hdr("CONFIGURATIONS EVALUATED")
    n_eta_cells = len(ETA_GRID) * 6  # 3 slices x 2 markets
    n_stress = len(stress_res["trial"].unique()) * 2  # windows x markets, finalist + v4 + b&h all run per cell
    n_fee = len(fee_rows)
    print(f"  Step B eta sweep:            {len(ETA_GRID)} eta values x 6 cells = {n_eta_cells}")
    print(f"  Step D stress test:          {n_stress} window-market cells (finalist eta={finalist_eta} only)")
    print(f"  Step E fee check:            {n_fee} cells")
    print(f"  Sanity checks (NOT scored):  eta->0, eta->infinity (2, direct numerical verification only)")
    print(f"  => total scored configuration-cells: {n_eta_cells + n_stress + n_fee}")
    print(f"     distinct eta values swept: {len(ETA_GRID)} (real) + 2 (sanity limits, not backtested via compare())")

    print(f"\nmax timestamp read anywhere in this branch (BTC and ETH): {max(max_ts)}  "
          f"(< {OOS_START}) -- no holdout bar was read.")


if __name__ == "__main__":
    main()
