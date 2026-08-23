#!/usr/bin/env python
"""NOVEL branch, R-96: a contemporaneous, CONVICTION-weighted BTC+ETH
allocator for kelly_regime_v4 -- allocate more of a shared risk budget to
whichever asset's own regime vote is currently more UNANIMOUS, rather than
to a fixed split (B-16/B-19/B-20, five prior REJECTED attempts) or to
whichever leg recently paid off (the parallel conservative branch's
performance-chasing Hedge/multiplicative-weights allocator).

Constraint attacked: SIZE (docs/LEDGER.md standing diagnosis) -- decide how
much to hold, not what happens next -- via a genuinely different capital-
allocation mechanism between two already-promoted single-asset books.

Not a duplicate of:
- B-16 (R-43): covariance/mean (Sigma^-1 mu) allocator on RAW daily
  returns -- a return/risk estimator, not a state read off the strategy's
  own vote.
- B-19 (R-51): one-time fixed split and inverse-realized-volatility split
  -- both static or backward-looking risk measures, still external to the
  vote.
- B-20 (R-52): calendar-cadence and drift-band-triggered fixed 50/50 --
  the weight is a FIXED CONSTANT (0.5/0.5) that never varies with either
  asset's own state; this round's weight moves every rebalance with a
  quantity v4 itself already computes.
- The parallel conservative branch's Hedge/multiplicative-weights
  reallocator: reacts to each leg's own TRAILING REALIZED PnL, a lagging
  signal computed after the fact. This branch reacts to each leg's own
  CURRENT vote-confidence, a signal available at the SAME bar the
  reallocation decision is made -- zero lag by construction.

Mechanism, one sentence
------------------------
R-62 (docs/LEDGER.md) localized kelly_regime_v4's SIZE signature to
`frac` alone -- the fraction of three latched moving-average anchors
(20/40/80-day, see kelly_regime_v3.py::prepare / kelly_regime.py) that
currently agree on direction, taking only the values {0, 1/3, 2/3, 1}.
That `frac` is v4's own per-bar CONFIDENCE in its regime call. This round
allocates pooled capital between independently-run BTC and ETH
kelly_regime_v4 books in proportion to each asset's own current `frac`,
`w_btc = conf_btc / (conf_btc + conf_eth)` (50/50 fallback when both are
flat), at a fixed DAILY cadence -- a contemporaneous, information-
coefficient-style conviction tilt, not a performance-chasing or
fixed-formula one.

Citations (best effort; flagged where verification was not possible
inside this sandbox)
-----------------------------------------------------------------------
- Grinold, R.C. & Kahn, R.N. (2000), "Active Portfolio Management" (2nd
  ed.), McGraw-Hill -- the "Fundamental Law of Active Management",
  IR = IC * sqrt(BR), and the general prescription of scaling a bet's
  size by the CONTEMPORANEOUS confidence/information-coefficient of the
  signal generating it, is the closest formal analogue to "size by
  current vote confidence" this project has cited (R-63 already used
  this same law on this same codebase for a different purpose -- cross-
  sectional breadth). Not independently re-derived from a primary
  source inside this sandbox; cited as the standard reference for
  confidence-proportional position sizing.
- Black, F. & Litterman, R. (1992), "Global Portfolio Optimization",
  Financial Analysts Journal 48(5) -- the general idea of tilting a
  portfolio's weights toward views held with higher confidence
  (formalized there via view uncertainty covariance Omega) is the
  standard citation for "size a tilt by conviction" in the asset-
  allocation literature. This round's mechanism is a much cruder,
  un-Bayesian analogue (no covariance-shrinkage machinery, no market-
  equilibrium prior) -- cited as the closest named precedent for the
  QUALITATIVE idea, not as a claim that Black-Litterman's actual
  machinery is used here. Flagged: not independently re-verified
  against the primary paper in this sandbox; page/section details are
  from memory of the standard textbook treatment (e.g. Idzorek 2005's
  exposition), not re-derived.
- This repo's own R-62 (docs/LEDGER.md, "Localized by R-62") -- the
  finding that `frac` alone reproduces v4's whole behavioral signature,
  which is the entire empirical warrant for using `frac` as a state
  variable here rather than inventing a new indicator (an INFO-axis
  move this branch deliberately avoids -- see docs/ROUTINE.md step 1).

Falsification test, fixed BEFORE any result was read (restated verbatim
from the task brief; the numeric decision rule below was authored before
`main()` was ever executed)
-----------------------------------------------------------------------
On inner-validation (2021-01-01 -> 2022-12-31), compute Sharpe, final
balance and a max-drawdown-adjusted growth statistic
(`calmar = profit_pct / max_dd_pct`, profit_pct rebased to the period's
own start) for:
  (a) BTC-solo kelly_regime_v4
  (b) ETH-solo kelly_regime_v4
  (c) periodically-rebalanced fixed 50/50 (SAME continuous engine,
      SAME cadence, w_btc=w_eth=0.5 constant)
  (d) the confidence-weighted construction (daily cadence, primary)

PASS only if (d) beats BOTH (a) and (b) AND (c) on BOTH Sharpe AND
`calmar`, simultaneously, on inner-validation. Anything else is FAIL
(report NEGATIVE) -- this includes the named failure mode: if BTC's and
ETH's vote-confidence series are highly correlated, w_btc/w_eth should
sit close to 0.5/0.5 almost always and (d) should be statistically
indistinguishable from (c), which the pre-registered rule above already
treats as a FAIL (not beating (c) is sufficient by itself to fail).
The BTC/ETH vote-confidence correlation is measured and reported
explicitly (see `corr` command / `confidence_correlation()`), exactly
because it is named in advance as the quantity that determines whether
this mechanism has any room to matter.

Hard rules honored
--------------------
- Only this NEW file is written. `kelly_regime_covkelly.py` and
  `kelly_regime_covkelly_v3_continuous.py` are imported from, UNCHANGED,
  reusing their `load_assets` (hard cutoff <= 2022-12-31 applied there,
  once), `_segment_bounds`, `weight_at`, `portfolio_metrics`,
  `r_squared` and continuous-engine primitives
  (`continuous_leg_equity`, `_segment_returns`, `period_metrics`,
  `run_continuous_full` for the fixed-50/50 reference specifically).
- No 2023+ literal appears anywhere in this file outside this sentence
  and comments referencing it as forbidden -- grep confirms this. Every
  frame used here is derived from `load_assets()`'s hard-sliced return
  value.
- The per-bar vote CONFIDENCE (`frac`) used for allocation is a fresh,
  from-scratch reimplementation of kelly_regime_v3.py::prepare's vote
  block (that block computes `frac` locally but the registered strategy
  does NOT expose it as a DataFrame column -- only the post-deadband
  `target` column is returned -- confirmed by reading the file; grepped
  `git show` history and current source for any exposed vote/frac/conf
  column: none exists). The reimplementation is validated by asserting
  it reproduces `KellyRegimeV4().prepare(df)["target"]` to numerical
  tolerance on BOTH assets' full pre-holdout frames (see
  `validate_vote_reimplementation`) -- so the confidence series used for
  allocation is proven, not assumed, to be the same quantity v4's own
  sizing already consumes.
- No lookahead in the allocation weight itself: the per-bar `frac` is
  resampled to one value per calendar day (`.resample("1D").last()`,
  itself only ever reading bars <= that day's close) and then
  `.shift(1)`'d before any rebalance decision reads it -- so day D's
  capital split is decided using only information available as of day
  D-1's close, the same causal convention `build_weight_series` in the
  reused conservative-branch file already uses for its own mu/Sigma
  inputs. See `causality_check_confidence` for the mechanical tamper
  probe.

Usage::

    python experiments/r96_novel_confidence_allocator.py validate    # frac reimplementation check
    python experiments/r96_novel_confidence_allocator.py corr        # BTC/ETH vote-confidence correlation
    python experiments/r96_novel_confidence_allocator.py headline    # the pre-registered falsification test
    python experiments/r96_novel_confidence_allocator.py causality   # mandatory no-lookahead check
    python experiments/r96_novel_confidence_allocator.py artifact    # mandatory R^2 exposure-artifact diagnostic
    python experiments/r96_novel_confidence_allocator.py all         # everything above, in order
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

from experiments.kelly_regime_covkelly import (  # noqa: E402
    SPOT,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END,
    load_assets,
    _segment_bounds,
    weight_at,
    portfolio_metrics,
    r_squared,
)
from experiments.kelly_regime_covkelly_v3_continuous import (  # noqa: E402
    FULL_START, FULL_END,
    continuous_leg_equity,
    _segment_returns,
    period_metrics,
    run_continuous_full,
)

N_EVALUATED = 0  # confidence-weighted allocator configurations only, same
                 # convention as both reused files: fixed-50/50 and solo-v4
                 # baselines are NOT counted.


# =================================================== vote-confidence reimpl.

def _vote_and_target(df: pd.DataFrame, strat: KellyRegimeV4) -> tuple[np.ndarray, np.ndarray]:
    """Byte-for-byte reimplementation of kelly_regime_v3.py::prepare.

    Returns (frac, target) as plain numpy arrays aligned to df.index.
    `frac` is the fraction of the three latched anchors currently voting
    bullish -- v4's own per-bar confidence, R-62's isolated factor -- and
    is NOT a column the registered strategy exposes (only the post-
    deadband `target` is). `target` is recomputed here purely so it can
    be checked, bar-for-bar, against the registered strategy's own
    `prepare()` output (see `validate_vote_reimplementation`): if this
    function's `target` does not match v4's real `target`, `frac` cannot
    be trusted either, since both come from the same block of code.

    Every operation below is the same rolling/ewm/ffill/shift(1) causal
    vocabulary `kelly_regime.py`/`kelly_regime_v3.py` already use and
    the project's own causality suite already accepts for the registered
    strategy -- nothing new is invented here, only recomputed.
    """
    close = df["close"]
    r = np.log(close).diff()

    votes = []
    for days in strat.horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + strat.band), 1.0,
                     np.where(close < anchor * (1.0 - strat.band), 0.0, np.nan)),
            index=df.index,
        )
        votes.append(v.ffill().fillna(0.0))
    frac = (sum(votes) / len(votes)).to_numpy()
    if strat.vote_gamma != 1.0:
        frac = frac ** strat.vote_gamma

    vol = (r.ewm(span=strat.vol_span, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    slow = (pd.Series(vol).ewm(span=strat.anchor_span_days * BARS_PER_DAY,
                                min_periods=BARS_PER_DAY).mean().to_numpy())

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(strat.target_vol / vol, strat.max_leverage)
        steady = np.minimum(strat.target_vol / slow, strat.max_leverage)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    n = len(df)
    target = np.zeros(n)
    pos = 0.0
    state = 0
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > strat.high_in else (-1 if x < strat.low_in else 0)
            elif state == 1 and x < strat.high_out:
                state = 0
            elif state == -1 and x > strat.low_out:
                state = 0
        scale = full[i] if state != 0 else steady[i]
        desired = frac[i] * scale
        if abs(desired - pos) > strat.deadband:
            pos = desired
        target[i] = pos

    return frac, target


def validate_vote_reimplementation(df: pd.DataFrame, label: str, atol: float = 1e-9) -> bool:
    """Assert `_vote_and_target`'s `target` matches the REGISTERED
    KellyRegimeV4's own `prepare()` output, bit-for-bit (to float
    tolerance). This is the proof that the `frac` this file allocates on
    is the same confidence value v4's own sizing already consumes, not a
    lookalike reimplementation that happens to differ in some corner.
    """
    strat = KellyRegimeV4()
    real_target = strat.prepare(df.copy())["target"].to_numpy()
    _, mine_target = _vote_and_target(df, strat)
    max_diff = float(np.nanmax(np.abs(real_target - mine_target)))
    ok = max_diff < atol
    print(f"[validate:{label}] max|reimplemented target - registered v4 target| = "
          f"{max_diff:.3e}  PASS={ok}")
    return ok


# ======================================================= confidence weights

def daily_confidence(df: pd.DataFrame, strat: KellyRegimeV4 | None = None) -> pd.Series:
    """v4's own per-bar vote confidence, resampled to one value per
    calendar day and shift(1)'d -- day D's value reflects only bars
    strictly before day D (the same causal convention
    `kelly_regime_covkelly.py::daily_log_returns`/`build_weight_series`
    already use for their own mu/Sigma inputs).
    """
    strat = strat or KellyRegimeV4()
    frac, _ = _vote_and_target(df, strat)
    frac_s = pd.Series(frac, index=df.index)
    daily = frac_s.resample("1D").last().ffill()
    return daily.shift(1)


def build_confidence_weights(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> pd.DataFrame:
    """w_btc = conf_btc / (conf_btc + conf_eth), 50/50 fallback when both
    are flat (conf_btc + conf_eth == 0). Columns match `weight_at`'s
    expected shape exactly (`w_btc`, `w_eth`, `fallback`) so that function
    -- imported UNCHANGED from `kelly_regime_covkelly.py` -- can be reused
    verbatim for the asof lookup at each rebalance boundary.
    """
    conf_b = daily_confidence(btc_df)
    conf_e = daily_confidence(eth_df)
    idx = conf_b.index.union(conf_e.index)
    cb = conf_b.reindex(idx).ffill()
    ce = conf_e.reindex(idx).ffill()
    cb_arr, ce_arr = cb.to_numpy(), ce.to_numpy()
    finite = np.isfinite(cb_arr) & np.isfinite(ce_arr)
    total = np.where(finite, cb_arr + ce_arr, np.nan)
    # fallback (50/50) whenever either leg's confidence is not yet defined
    # (the single NaN day at each asset's own shift(1)-induced start-of-
    # history, or before either asset's frac warmup has produced a real
    # vote) OR both legs are simultaneously flat (total confidence == 0) --
    # never propagate a NaN into pooled capital.
    fallback = (~finite) | (total <= 1e-12)
    safe_total = np.where(fallback, 1.0, total)  # avoid 0-div; result discarded by np.where below
    w_b = np.where(fallback, 0.5, np.where(finite, cb_arr, 0.0) / safe_total)
    w_e = np.where(fallback, 0.5, np.where(finite, ce_arr, 0.0) / safe_total)
    return pd.DataFrame({"w_btc": w_b, "w_eth": w_e, "fallback": fallback,
                         "conf_btc": cb_arr, "conf_eth": ce_arr}, index=idx)


def confidence_correlation(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> dict:
    """The pre-registered failure-mode measurement: Pearson correlation of
    the two assets' own vote-confidence series. High correlation means
    the two are almost always both fully confident or both flat together
    -- in which case w_btc/w_eth degenerates to ~50/50 and this mechanism
    adds nothing over the fixed-50/50 reference, exactly the failure mode
    named in the pre-registration.
    """
    conf_b = daily_confidence(btc_df)
    conf_e = daily_confidence(eth_df)
    idx = conf_b.index.union(conf_e.index)
    cb = conf_b.reindex(idx).ffill()
    ce = conf_e.reindex(idx).ffill()
    out = {}
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)),
                          ("valid", (VALID_START, VALID_END)),
                          ("full", (FULL_START, FULL_END))):
        cbs = cb.loc[s:e].dropna()
        ces = ce.loc[s:e].dropna()
        common = cbs.index.intersection(ces.index)
        if len(common) < 5:
            out[label] = float("nan")
            continue
        x, y = cbs.loc[common].to_numpy(), ces.loc[common].to_numpy()
        out[label] = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    return out


# ======================================================== continuous engine

def run_confidence_alloc(
    btc_df: pd.DataFrame, eth_df: pd.DataFrame,
    rebalance_freq: str, market: MarketSpec = SPOT, start_balance: float = 1000.0,
) -> dict:
    """The confidence-weighted allocator, built with the EXACT continuous-
    engine pattern `kelly_regime_covkelly_v3_continuous.py` established:
    each leg's kelly_regime_v4 runs ONCE, continuously, over the whole
    window (never reset), and cross-asset reallocation is a separate,
    fully causal, post-hoc step that reads only each leg's own vote
    confidence, reindexes it against a rebalance schedule, and rescales
    the (already-computed) continuous leg curves -- never re-running the
    strategy. `weight_at` is imported UNCHANGED from the conservative
    branch's file; only the WEIGHTS fed into it are new.
    """
    btc_full = continuous_leg_equity(btc_df, market, start_balance=start_balance)
    eth_full = continuous_leg_equity(eth_df, market, start_balance=start_balance)
    weights_df = build_confidence_weights(btc_df, eth_df)

    bounds = _segment_bounds(FULL_START, FULL_END, rebalance_freq)
    btc_segs = _segment_returns(btc_full, bounds)
    eth_segs = _segment_returns(eth_full, bounds)
    n = min(len(btc_segs), len(eth_segs))

    pooled = start_balance
    pieces = []
    log_rows = []
    for i in range(n):
        sb, se = btc_segs[i], eth_segs[i]
        seg_start, seg_end = sb["seg_start"], sb["seg_end"]
        w_b, w_e, fb = weight_at(weights_df, seg_start)

        dollars_b = pooled * w_b
        dollars_e = pooled * w_e

        btc_sub = btc_full.loc[seg_start:seg_end]
        eth_sub = eth_full.loc[seg_start:seg_end]
        scale_b = (dollars_b / sb["base_val"]) if sb["base_val"] > 0 else 0.0
        scale_e = (dollars_e / se["base_val"]) if se["base_val"] > 0 else 0.0
        btc_leg = btc_sub * scale_b
        eth_leg = eth_sub * scale_e

        idx = btc_leg.index.union(eth_leg.index)
        combined = btc_leg.reindex(idx).ffill().bfill().fillna(0.0) \
                 + eth_leg.reindex(idx).ffill().bfill().fillna(0.0)
        if len(combined) == 0:
            continue
        pieces.append(combined)
        pooled = float(combined.iloc[-1])
        log_rows.append({"date": seg_start, "w_btc": w_b, "w_eth": w_e, "fallback": fb,
                         "pooled_end": pooled})

    equity = pd.concat(pieces).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    weights_log = pd.DataFrame(log_rows)
    return {"equity": equity, "weights_log": weights_log,
            "final_balance": float(equity.iloc[-1]) if len(equity) else start_balance,
            "btc_full": btc_full, "eth_full": eth_full}


def allocation_turnover(weights_log: pd.DataFrame) -> dict:
    """Turnover of the ALLOCATION itself (mean/sum |delta w_btc| across
    rebalances). Each leg's OWN internal trading turnover/fees are
    identical across every candidate compared here (same kelly_regime_v4
    code, same price data -- only the DOLLARS allocated to it differ, and
    fees are charged as a fraction of that leg's own notional, so the
    per-leg trading turnover fraction is invariant to how much capital
    the leg is holding). Caveat, stated honestly: the reallocation of
    capital BETWEEN legs is a pure accounting rescale of each leg's
    already-computed continuous equity curve (the "rescale, don't
    replay" trick this whole engine family relies on -- see both reused
    files' docstrings), so this simulation charges NO transaction cost
    for moving dollars from the BTC book to the ETH book or back. That
    is an inherited simplification of the promoted continuous-engine
    pattern (identical in the fixed-50/50 reference computed the same
    way), not something unique to the confidence-weighted construction.
    """
    if len(weights_log) < 2:
        return {"n_rebalances": len(weights_log), "mean_abs_dw": 0.0, "sum_abs_dw": 0.0}
    dw = weights_log["w_btc"].diff().abs().dropna()
    return {"n_rebalances": len(weights_log), "mean_abs_dw": float(dw.mean()),
            "sum_abs_dw": float(dw.sum())}


# =============================================================== headline

def _solo_metrics(df: pd.DataFrame, market: MarketSpec = SPOT) -> dict:
    full = continuous_leg_equity(df, market)
    return {"train": period_metrics(full, TRAIN_START, TRAIN_END),
           "valid": period_metrics(full, VALID_START, VALID_END), "_full": full}


def _calmar(m: dict) -> float:
    profit_pct = (m["final_balance"] / 1000.0 - 1.0) * 100.0
    return profit_pct / m["max_dd_pct"] if m["max_dd_pct"] > 1e-9 else float("nan")


def run_headline(data_dir: str = "data") -> dict:
    global N_EVALUATED
    btc_df, eth_df = load_assets(data_dir)

    print("=== step 0: validate the vote-confidence reimplementation ===")
    ok_b = validate_vote_reimplementation(btc_df, "BTC")
    ok_e = validate_vote_reimplementation(eth_df, "ETH")
    if not (ok_b and ok_e):
        raise RuntimeError("vote reimplementation does not match registered v4 -- fix before trusting anything below")

    print("\n=== step 1: BTC/ETH vote-confidence correlation (pre-registered failure-mode check) ===")
    corr = confidence_correlation(btc_df, eth_df)
    for label, v in corr.items():
        print(f"  {label:6s}: corr(conf_btc, conf_eth) = {v:+.3f}")

    print("\n=== step 2: (a)/(b) solo legs, continuous engine ===")
    solo_btc = _solo_metrics(btc_df)
    solo_eth = _solo_metrics(eth_df)

    cadences = {"daily (D, primary)": "D", "weekly (W-MON, robustness)": "W-MON"}
    out = {"corr": corr, "solo_btc": solo_btc, "solo_eth": solo_eth, "cells": {}}

    for cad_label, freq in cadences.items():
        print(f"\n=== step 3: cadence = {cad_label} ===")
        N_EVALUATED += 1  # confidence-weighted allocator config

        conf_res = run_confidence_alloc(btc_df, eth_df, freq)
        conf_eq = conf_res["equity"]
        conf = {"train": period_metrics(conf_eq, TRAIN_START, TRAIN_END),
               "valid": period_metrics(conf_eq, VALID_START, VALID_END)}

        # (c) fixed 50/50 reference -- SAME continuous engine, SAME cadence,
        # imported UNCHANGED from the conservative branch's continuous file.
        fixed_res = run_continuous_full(btc_df, eth_df, freq, "fixed5050")
        fixed_eq = fixed_res["equity"]
        fixed = {"train": period_metrics(fixed_eq, TRAIN_START, TRAIN_END),
                "valid": period_metrics(fixed_eq, VALID_START, VALID_END)}

        turnover = allocation_turnover(conf_res["weights_log"])

        out["cells"][cad_label] = {"confidence": conf, "fixed5050": fixed,
                                   "turnover": turnover, "_conf_eq": conf_eq,
                                   "_fixed_eq": fixed_eq, "_btc_full": conf_res["btc_full"],
                                   "_eth_full": conf_res["eth_full"]}

    _print_headline(out)
    _decision_rule(out)
    print(f"\ntotal N_EVALUATED (confidence-weighted allocator configurations): {N_EVALUATED}")
    return out


def _print_headline(out: dict) -> None:
    print("\n=== HEADLINE (spot) ===")
    header = f"{'candidate':<32} {'cadence':<26} {'period':<6} {'final':>10} {'sharpe':>8} {'maxDD%':>8} {'calmar':>8}"
    print(header)
    for label, m in (("v4 BTC-solo (a)", out["solo_btc"]), ("v4 ETH-solo (b)", out["solo_eth"])):
        for period in ("train", "valid"):
            row = m[period]
            print(f"{label:<32} {'(cadence n/a)':<26} {period:<6} {row['final_balance']:>10.1f} "
                  f"{row['sharpe']:>8.2f} {row['max_dd_pct']:>8.1f} {_calmar(row):>8.3f}")
    for cad_label, cell in out["cells"].items():
        for name, key in (("fixed 50/50 (c)", "fixed5050"), ("confidence-weighted (d)", "confidence")):
            table = cell[key]
            for period in ("train", "valid"):
                row = table[period]
                print(f"{name:<32} {cad_label:<26} {period:<6} {row['final_balance']:>10.1f} "
                      f"{row['sharpe']:>8.2f} {row['max_dd_pct']:>8.1f} {_calmar(row):>8.3f}")
        t = cell["turnover"]
        print(f"  [{cad_label}] allocation turnover: n_rebalances={t['n_rebalances']} "
              f"mean|dw|={t['mean_abs_dw']:.4f} sum|dw|={t['sum_abs_dw']:.2f}")


def _decision_rule(out: dict) -> None:
    """The pre-registered decision rule, applied mechanically, on the
    PRIMARY (daily) cadence's inner-validation numbers only -- exactly as
    frozen in this file's module docstring before any result was read.
    """
    primary = out["cells"]["daily (D, primary)"]
    d = primary["confidence"]["valid"]
    a = out["solo_btc"]["valid"]
    b = out["solo_eth"]["valid"]
    c = primary["fixed5050"]["valid"]

    d_sharpe, d_calmar = d["sharpe"], _calmar(d)
    checks = {
        "beats (a) BTC-solo on Sharpe": d_sharpe > a["sharpe"],
        "beats (a) BTC-solo on calmar": d_calmar > _calmar(a),
        "beats (b) ETH-solo on Sharpe": d_sharpe > b["sharpe"],
        "beats (b) ETH-solo on calmar": d_calmar > _calmar(b),
        "beats (c) fixed 50/50 on Sharpe": d_sharpe > c["sharpe"],
        "beats (c) fixed 50/50 on calmar": d_calmar > _calmar(c),
    }
    print("\n=== PRE-REGISTERED DECISION RULE (inner-validation, daily cadence) ===")
    for label, result in checks.items():
        print(f"  {label:38s}: {'PASS' if result else 'FAIL'}")
    overall = all(checks.values())
    print(f"\n  OVERALL: {'PASS -- construction clears the pre-registered bar' if overall else 'FAIL -- NEGATIVE result'}")


# =============================================================== diagnostics

def causality_check_confidence(data_dir: str = "data") -> bool:
    """Truncation tamper probe, modeled on both reused files' own
    `causality_check`/`causality_check_continuous`: perturb every bar
    strictly after a cut date by a large multiplicative factor (up and
    down), recompute the confidence-weighted pooled equity, and confirm
    the pooled equity BEFORE the cut is unchanged in both tampered runs.
    """
    btc_df, eth_df = load_assets(data_dir)
    cut = pd.Timestamp("2021-06-30", tz="UTC")
    K = 137.0

    def tamper(df: pd.DataFrame, factor: float) -> pd.DataFrame:
        out = df.copy()
        mask = out.index > cut
        for col in ("open", "high", "low", "close"):
            out.loc[mask, col] = out.loc[mask, col] * factor
        return out

    base = run_confidence_alloc(btc_df, eth_df, "D")
    up = run_confidence_alloc(tamper(btc_df, K), tamper(eth_df, K), "D")
    down = run_confidence_alloc(tamper(btc_df, 1.0 / K), tamper(eth_df, 1.0 / K), "D")

    pre = base["equity"].index <= cut
    b = base["equity"][pre].to_numpy()
    u = up["equity"].reindex(base["equity"].index)[pre].to_numpy()
    d = down["equity"].reindex(base["equity"].index)[pre].to_numpy()
    max_diff_up = float(np.nanmax(np.abs(b - u)))
    max_diff_down = float(np.nanmax(np.abs(b - d)))
    ok = max_diff_up < 1e-6 and max_diff_down < 1e-6
    print(f"causality check (confidence allocator): cut={cut.date()}, K={K}")
    print(f"  max |base - up-tampered| pooled equity before cut: {max_diff_up:.3e}")
    print(f"  max |base - down-tampered| pooled equity before cut: {max_diff_down:.3e}")
    print(f"  PASS (pooled equity before cut unchanged): {ok}")
    return ok


def artifact_diagnostics(data_dir: str = "data") -> None:
    """Mandatory R^2 exposure-artifact diagnostic (same convention as
    both reused files' `artifact_diagnostics`/`artifact_diagnostics_full`):
    confidence-weighted vs a flat rescale of BTC-solo, and confidence-
    weighted vs the fixed-50/50 reference, both on the CONTINUOUS engine's
    own objects so the check isolates the allocator's contribution from
    the engine choice.
    """
    btc_df, eth_df = load_assets(data_dir)
    print("\n=== exposure-artifact diagnostics (confidence allocator) ===")
    for cad_label, freq in (("daily (D)", "D"), ("weekly (W-MON)", "W-MON")):
        conf_res = run_confidence_alloc(btc_df, eth_df, freq)
        fixed_res = run_continuous_full(btc_df, eth_df, freq, "fixed5050")
        btc_full = conf_res["btc_full"]
        for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)),
                              ("valid", (VALID_START, VALID_END))):
            conf_sub = conf_res["equity"].loc[s:e]
            fixed_sub = fixed_res["equity"].loc[s:e]
            solo_sub = btc_full.loc[s:e]
            r2_solo = r_squared(conf_sub, solo_sub)
            r2_fixed = r_squared(conf_sub, fixed_sub)
            flag_solo = "FLAT-RESCALE ARTIFACT" if r2_solo > 0.95 else "ok"
            flag_fixed = "SAME AS FIXED SPLIT" if r2_fixed > 0.95 else "ok"
            print(f"[{cad_label} / {label}] confidence vs flat-rescaled v4-BTC-solo: "
                  f"R^2 = {r2_solo:.4f} -> {flag_solo}")
            print(f"[{cad_label} / {label}] confidence vs fixed 50/50 (continuous control): "
                  f"R^2 = {r2_fixed:.4f} -> {flag_fixed}")


# ===================================================================== CLI

def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "validate":
        btc_df, eth_df = load_assets()
        validate_vote_reimplementation(btc_df, "BTC")
        validate_vote_reimplementation(eth_df, "ETH")
    elif cmd == "corr":
        btc_df, eth_df = load_assets()
        for label, v in confidence_correlation(btc_df, eth_df).items():
            print(f"{label:6s}: corr(conf_btc, conf_eth) = {v:+.3f}")
    elif cmd == "headline":
        run_headline()
    elif cmd == "causality":
        causality_check_confidence()
    elif cmd == "artifact":
        artifact_diagnostics()
    elif cmd == "all":
        run_headline()
        causality_check_confidence()
        artifact_diagnostics()
        print(f"\ntotal N_EVALUATED (confidence-weighted allocator configurations): {N_EVALUATED}")
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
