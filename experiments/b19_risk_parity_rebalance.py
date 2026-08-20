#!/usr/bin/env python
"""B-19, NOVEL branch: does an inverse-volatility-weighted BTC+ETH portfolio
of `kelly_regime_v4`, rebalanced at cadences R-50 never tried (quarterly,
semiannual) and run through R-50's own continuous (non-restarting) engine
PLUS an explicit rebalance-turnover cost this project's rebalancing work
has never charged, survive pre-registration, the fee-tier falsification
test and this project's holdout process?

Idea, one sentence
-------------------
Weight the BTC and ETH legs of a periodically-rebalanced `kelly_regime_v4`
portfolio inversely to each leg's OWN trailing realized volatility (no
covariance matrix, no expected-return estimate at all -- a simplified
special case of risk parity / equal risk contribution), and rebalance at a
genuinely new, lower cadence (quarterly / semiannual) that this project's
own COST diagnosis and the rebalancing-frequency literature both predict
should matter once turnover is priced honestly.

Constraint attacked
--------------------
SIZE (how much to hold in each leg, not what happens next -- exactly the
axis this project's one-line summary says is the only one that has ever
worked) and N (a second asset raises the effective number of regime
observations, the same N-axis R-42/R-43/R-50 already opened) and COST
(this file is the first round on this backlog item to charge an explicit
transaction cost for the portfolio-level rebalance itself, not just for
each leg's own v4-driven trades -- see "A gap in R-50's engine" below).

Not a duplicate of
--------------------
- `kelly_regime_covkelly[.py|_v2.py]` (R-42/R-43, REJECTED): that
  allocator solves Sigma^-1*mu, a closed-form mean-variance weight that
  needs a trailing EWM *mean-return* estimate per leg -- exactly the
  estimation-risk this project has repeatedly found too noisy at 5m-bar
  cadence to be worth it (R-37/R-38/R-45, and the fixed-mean regression
  named explicitly in R-42/R-43's own novel branch). This file's weights
  use ONLY each leg's own trailing volatility -- no mean, no covariance
  matrix, no cross-asset term at all -- per Maillard, Roncalli & Teiletche
  (2010), "The Properties of Equally Weighted Risk Contribution
  Portfolios", *Journal of Portfolio Management* 36(4), 60-70 (also SSRN
  1271972): the Equal Risk Contribution portfolio's own paper notes that
  when pairwise correlations are equal (the 2-asset case makes this
  automatic -- there is only one pairwise correlation), ERC reduces
  exactly to inverse-volatility weighting. That is the "well-established
  middle ground" this file implements: strictly less information than
  Sigma^-1*mu (no mean, no cross term), strictly more than R-50's static
  50/50 (which uses no information about either leg at all).
- R-50's own static-50/50-via-continuous-engine finding (the B-19 lead
  itself, ΔSharpe +0.79/+0.80 vs v4-solo, monthly/weekly, inner-validation
  only): that portfolio's weights are FIXED at (0.5, 0.5) forever; this
  file's weights move every rebalance with each leg's own realized
  volatility. It is re-derived here (not copied from R-50's printed
  number) as the reference this file's candidate must beat.
- R-50 itself only ever tested monthly and weekly cadence. This file's
  cadence axis (monthly / quarterly / semiannual) is chosen from Dichtl,
  Drobetz & Wambach (2016 -- originally circulated 2012-2014), "Testing
  Rebalancing Strategies for Stock-Bond Portfolios Across Different Asset
  Allocations", *Applied Economics* 48(9), 772-788 (SSRN 1927764/2479384):
  a double block bootstrap on US/UK/German stock-bond portfolios found the
  realistic-cost-optimal rebalancing frequency sits between quarterly and
  yearly, not weekly/monthly -- exactly the frequencies this backlog item
  never tried.

A gap in R-50's engine, found and fixed here (not a criticism of R-50,
whose brief was the segment-restart artifact only)
------------------------------------------------------------------------
R-50's `run_continuous_full` (the file this project's brief told this
branch to REUSE, and does reuse unchanged for both legs' continuous
curves) reallocates pooled capital between legs by an ALGEBRAIC RESCALE
of each leg's own independently-computed continuous equity curve -- by
design, so the strategy is never re-invoked and its latch state is never
disturbed. That is exactly right for removing the restart artifact. But
it has a side effect nobody in R-50 needed to think about because both of
R-50's arms (fixed 50/50, and the Sigma^-1*mu dynamic allocator) were
being compared to EACH OTHER on the SAME zero-rebalance-cost basis: the
rescale-don't-replay trick means moving dollars from the BTC sub-account
into the ETH sub-account at a rebalance boundary costs NOTHING in R-50's
engine -- there is no `fee_rate * traded_notional` charged for the
portfolio-level reallocation itself, only for each leg's own v4-driven
trades (which the rescale correctly preserves, since those are already
baked into the continuous curve). For a study whose entire second axis is
"does rebalancing less often help once realistic costs bite", reusing
R-50's engine unmodified would make cadence a free variable with no cost
to trade off against -- the COST axis this file exists to test would be
silently absent from the very engine measuring it.

Fix, kept as close as possible to R-50's own mechanism: `run_portfolio_
continuous_costed` below is R-50's `run_continuous_full`, generalized to
accept either weight mode (`fixed5050` or `invvol`) and extended with ONE
new step per rebalance boundary -- before computing the new segment's
target dollars, the dollar amount that must be SOLD from the overweight
leg and BOUGHT into the underweight leg to reach the new target weight is
computed (`shift`), and `2 * rebalance_fee_rate * shift` (one taker fee on
the sell side, one on the buy side -- a full round trip) is deducted from
pooled capital before the new segment starts. `rebalance_fee_rate`
defaults to the SAME taker tier as `market.fee_rate`, i.e. the portfolio
rebalance is assumed to pay the same fee schedule as each leg's own v4
trades -- stated explicitly because it is an assumption, not a measured
fact (this project has no combined-order venue data). The underlying
per-leg continuous curves (`continuous_leg_equity`, `_segment_returns`,
`_segment_bounds`, `weight_at`) are imported UNCHANGED from R-50's and
R-42's files -- only the capital-combination step gains a cost, matching
this project's own hard rule ("only this file is touched").

This cost is applied EQUALLY to the fixed-50/50 reference (re-derived
here on the same costed engine, not copied from R-50's zero-cost number)
so the candidate-vs-reference comparison stays apples-to-apples: a fixed
split still needs occasional rebalancing back to 50/50 as prices diverge,
and it should pay for that exactly as the inverse-vol candidate pays for
its own reallocations.

Pre-registered falsification test (chosen now, before any result exists)
--------------------------------------------------------------------------
The candidate is REJECTED (no holdout read) if EITHER of the following
holds on the inner splits:
  (F1) Exposure-artifact check. Regress the candidate's return series
       against a flat-rescaled BTC-solo `kelly_regime_v4` benchmark
       (`r_squared`, imported unchanged from `kelly_regime_covkelly.py`,
       the same >0.95 threshold used throughout R-34/R-42/R-43/R-50). If
       R^2 > 0.95 in EITHER inner split, the "diversification" result is
       relabeled leverage/exposure, not a real effect.
  (F2) Fee-tier survival. Re-run the selected candidate AND the
       fixed-50/50 reference (same cadence, same costed engine) at the
       realistic 0.40% Bitstamp taker tier (`scripts/fee_study.py`'s own
       BITSTAMP_TAKER constant, applied to both `market.fee_rate` and
       `rebalance_fee_rate`). FAIL if the candidate's inner-validation
       Sharpe advantage over the fixed-50/50 reference at 0.40% is
       negative (turnover cost has flipped the sign of the finding) OR if
       the candidate no longer beats `buy_and_hold` on inner-validation
       at 0.40%.
If either fires, STOP -- do not read the 2023+ holdout.

Pre-registered promotion decision rule (mirrors ROUTINE.md's bar; written
before any result exists)
--------------------------------------------------------------------------
Only if F1 and F2 both PASS does this branch read the holdout once, on
the frozen winning (lookback, cadence) configuration selected below.
PROMOTE (as a documented-negative-eligible candidate for `kelly_
regime_covkelly`-style unregistered write-up -- true registration would
still need B-17's "wire multi-asset into run.py" half, per the backlog
row) iff ALL of:
  (P1) beats `buy_and_hold` OOS at the 0.10% baseline AND the 0.40% real
       taker tier;
  (P2) beats the re-derived static-50/50-continuous-engine reference by
       more than the +/-0.2 Sharpe noise floor (R-20), OR shows a
       drawdown/tail improvement over it (this project's repeatedly-
       replicating property, per ROUTINE.md's own promotion bar);
  (P3) survives F1 and F2 (already required to reach this point, checked
       again on the holdout itself);
  (P4) the (lookback, cadence) neighbourhood is a PLATEAU, not a peak --
       report neighbours, not just the winner.
Anything else is NEGATIVE. If the decision rule is changed after seeing
the holdout, that will be stated explicitly and the result downgraded to
in-sample, per ROUTINE.md.

Selection rule inside the sweep (fixed before the sweep runs, mirrors
`kelly_regime_covkelly.py::select_best`)
--------------------------------------------------------------------------
Rank candidates by `min(train_sharpe, valid_sharpe)` -- guards against the
train-loses/validation-wins overfit signature that sank R-37/R-38/R-40 --
tie-break on `-valid_max_dd_pct`.

Grid (12 configurations, matching this project's established scale)
--------------------------------------------------------------------------
lookback_days in {30, 60, 90, 180} (trailing realized-vol estimation
window for the inverse-vol weights -- no single value is prescribed by
Maillard/Roncalli/Teiletche, so a spread from ~1 to ~6 months is swept) x
rebalance_freq in {monthly (MS), quarterly (QS), semiannual (2QS)}.

Hard rules honored
--------------------
- Only this NEW file is touched. `kelly_regime_covkelly.py`, `_v2.py`,
  `_v3_continuous.py`, `kelly_regime_dual_fixed.py`, `kelly_regime_v4.py`
  and everything under `src/tradebot/` are imported from, unmodified.
- Inner-only by default: `load_assets` (imported unchanged from
  `kelly_regime_covkelly.py`) hard-slices to <= 2022-12-31. The ONLY
  function in this file that reads 2023+ data is `holdout()`, which
  imports the full, uncut `BTC`/`ETH` frames from `kelly_regime_dual_
  fixed.py` (itself unmodified) and slices them to >= OOS_START itself --
  the exact gating precedent `kelly_regime_dual_bootstrap.py` (R-43) set.
  Grep this file for "2023-01-01" to confirm the only occurrences are the
  `OOS_START` constant and this sentence.
- No lookahead: `causality_check` runs the same multiply/divide
  truncation tamper probe R-50/R-42 used, extended to this file's new
  rebalance-cost code path.
- `N_EVALUATED` counts every distinct inverse-vol-candidate configuration
  backtested (matching `kelly_regime_covkelly.py`'s own convention: fixed-
  50/50 and solo-v4 baselines are NOT counted there either). A separate
  `N_BACKTESTS_TOTAL` counts every distinct backtest of any kind (including
  baselines, fee-tier re-runs, causality/tamper runs, futures) for this
  project's "count every distinct configuration" bookkeeping instruction --
  reported at the end of every CLI command, both numbers, labelled.

Sources (author, year, venue)
--------------------------------
- Maillard, S., Roncalli, T., Teiletche, J. (2010), "The Properties of
  Equally Weighted Risk Contribution Portfolios", Journal of Portfolio
  Management 36(4), 60-70 (SSRN 1271972). Primary source for inverse-
  volatility weighting as the correlation-agnostic special case of Equal
  Risk Contribution / risk parity.
- Qian, E. (2005), "Risk Parity Portfolios: Efficient Portfolios Through
  True Diversification", PanAgora Asset Management working paper --
  earlier practitioner statement of the risk-parity / inverse-vol
  weighting idea this file implements.
- Dichtl, H., Drobetz, W., Wambach, M. (2016), "Testing Rebalancing
  Strategies for Stock-Bond Portfolios Across Different Asset
  Allocations", Applied Economics 48(9), 772-788 (working paper
  circulated 2012-2014, SSRN 1927764 / 2479384). Double block bootstrap
  finding that realistic-cost-optimal rebalancing frequency sits between
  quarterly and yearly for stock-bond portfolios -- motivates this file's
  quarterly/semiannual cadence axis, genuinely untried by R-50.
- This project's own R-33 ("Holding less draws down less; that is
  arithmetic, not evidence") -- the exposure-artifact check below applies
  that lesson directly.
- R-42, R-43, R-50 (this ledger) -- the mechanism this file reuses
  (continuous engine) and the mechanism it deliberately does NOT reuse
  (Sigma^-1*mu).

Usage::

    python experiments/b19_risk_parity_rebalance.py sweep       # 12 configs, spot, inner splits
    python experiments/b19_risk_parity_rebalance.py select      # pick winner, headline table vs references
    python experiments/b19_risk_parity_rebalance.py causality   # mandatory no-lookahead check
    python experiments/b19_risk_parity_rebalance.py artifact    # mandatory R^2 exposure-artifact diagnostic (F1)
    python experiments/b19_risk_parity_rebalance.py feetier     # 0.40% taker stress test (F2)
    python experiments/b19_risk_parity_rebalance.py futures     # winner + references on 5x futures (secondary)
    python experiments/b19_risk_parity_rebalance.py holdout     # PRE-REGISTERED, run ONLY if F1+F2 both pass
    python experiments/b19_risk_parity_rebalance.py all         # sweep+select+causality+artifact+feetier+futures
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
from tradebot.metrics import max_drawdown_pct, sharpe_ratio  # noqa: E402
from tradebot.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.kelly_regime_covkelly import (  # noqa: E402
    SPOT,
    FUTURES5X,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END,
    load_assets,
    daily_log_returns,
    weight_at,
    run_v4_solo,
    portfolio_metrics,
    r_squared,
)
from experiments.kelly_regime_covkelly_v3_continuous import (  # noqa: E402
    continuous_leg_equity,
    _segment_returns,
    period_metrics,
    FULL_START, FULL_END,
)
from experiments.kelly_regime_covkelly import _segment_bounds  # noqa: E402

# the ONE place OOS_START is spelled out as a literal in this file, per
# the gating convention set by kelly_regime_dual_bootstrap.py (R-43)
OOS_START = "2023-01-01"

N_EVALUATED = 0        # inverse-vol candidate configurations only (project convention)
N_BACKTESTS_TOTAL = 0  # every distinct backtest of any kind, this file's own full count


def _count(n: int = 1) -> None:
    global N_BACKTESTS_TOTAL
    N_BACKTESTS_TOTAL += n


# ============================================================ vol weights

def trailing_vol_series(df: pd.DataFrame, lookback_days: int) -> pd.Series:
    """Causal trailing realized volatility of daily log returns.

    vol[T] is computed from daily returns strictly BEFORE T: a rolling std
    over `lookback_days` daily observations, then `.shift(1)` -- the same
    causal convention `kelly_regime_covkelly.py::build_weight_series` uses
    for its own EWM statistics. No expanding/rolling window here ever
    includes its own row's future.
    """
    r = daily_log_returns(df)
    vol = r.rolling(lookback_days, min_periods=max(5, lookback_days // 2)).std()
    return vol.shift(1)


def build_invvol_weight_series(btc_df: pd.DataFrame, eth_df: pd.DataFrame,
                               lookback_days: int) -> pd.DataFrame:
    """Inverse-volatility weights: w_i = (1/vol_i) / sum_j(1/vol_j).

    Uses ONLY each leg's own trailing volatility -- no covariance matrix,
    no cross-asset term, no expected-return estimate anywhere. During
    warmup (either vol not yet finite, or non-positive) falls back to
    0.5/0.5, matching `build_weight_series`'s own fallback convention so
    the two allocators' warmup behavior is comparable.
    """
    vol_b = trailing_vol_series(btc_df, lookback_days).rename("vol_btc")
    vol_e = trailing_vol_series(eth_df, lookback_days).rename("vol_eth")
    out = pd.concat([vol_b, vol_e], axis=1).dropna(how="all").ffill()

    wb = np.full(len(out), np.nan)
    we = np.full(len(out), np.nan)
    fallback = np.zeros(len(out), dtype=bool)
    vb = out["vol_btc"].to_numpy()
    ve = out["vol_eth"].to_numpy()
    for i in range(len(out)):
        if not (np.isfinite(vb[i]) and np.isfinite(ve[i])) or vb[i] <= 0 or ve[i] <= 0:
            fallback[i] = True
            wb[i], we[i] = 0.5, 0.5
            continue
        inv_b, inv_e = 1.0 / vb[i], 1.0 / ve[i]
        s = inv_b + inv_e
        wb[i], we[i] = inv_b / s, inv_e / s
    out["w_btc"] = wb
    out["w_eth"] = we
    out["fallback"] = fallback
    return out


# ======================================================== costed engine

def run_portfolio_continuous_costed(
    btc_df: pd.DataFrame, eth_df: pd.DataFrame,
    rebalance_freq: str, weight_mode: str, lookback_days: int | None = None,
    market: MarketSpec = SPOT, start_balance: float = 1000.0,
    rebalance_fee_rate: float | None = None,
    v4_kwargs: dict | None = None,
    full_start: str = FULL_START, full_end: str = FULL_END,
) -> dict:
    """R-50's continuous engine (`run_continuous_full`, reused via its own
    imported building blocks: `continuous_leg_equity`, `_segment_returns`,
    `_segment_bounds`, `weight_at`), generalized to weight_mode in
    {"fixed5050", "invvol"} and extended with an explicit rebalance-
    turnover cost -- see the module docstring's "A gap in R-50's engine"
    section for why this is necessary for a cost-sensitivity study and
    exactly what changed. Applies identically to both weight modes so the
    candidate-vs-reference comparison pays the same cost convention.
    """
    global N_BACKTESTS_TOTAL
    N_BACKTESTS_TOTAL += 1
    if rebalance_fee_rate is None:
        rebalance_fee_rate = market.fee_rate

    btc_full = continuous_leg_equity(btc_df, market, v4_kwargs,
                                     start=full_start, end=full_end,
                                     start_balance=start_balance)
    eth_full = continuous_leg_equity(eth_df, market, v4_kwargs,
                                     start=full_start, end=full_end,
                                     start_balance=start_balance)

    if weight_mode == "invvol":
        weights_df = build_invvol_weight_series(btc_df, eth_df, lookback_days)
    elif weight_mode == "fixed5050":
        weights_df = None
    else:
        raise ValueError(weight_mode)

    bounds = _segment_bounds(full_start, full_end, rebalance_freq)
    btc_segs = _segment_returns(btc_full, bounds)
    eth_segs = _segment_returns(eth_full, bounds)
    n = min(len(btc_segs), len(eth_segs))

    pooled = start_balance
    pieces: list[pd.Series] = []
    log_rows = []
    fees_rebalance_total = 0.0
    prev_b = prev_e = None

    for i in range(n):
        sb, se = btc_segs[i], eth_segs[i]
        seg_start, seg_end = sb["seg_start"], sb["seg_end"]
        if weight_mode == "invvol":
            w_b, w_e, fb = weight_at(weights_df, seg_start)
        else:
            w_b, w_e, fb = 0.5, 0.5, False

        if i == 0 or prev_b is None:
            pooled_pre = pooled
            fee_this = 0.0
        else:
            # dollars each leg drifted to by the end of the PRIOR segment,
            # under the PRIOR segment's weights -- this is what a real
            # broker would show as the account's actual split right before
            # today's rebalance decision.
            pooled_pre = prev_b + prev_e
            target_b = pooled_pre * w_b
            shift = abs(target_b - prev_b)
            # one taker fee to sell the overweight leg, one to buy the
            # underweight leg -- a full round trip on the shifted notional
            fee_this = 2.0 * rebalance_fee_rate * shift
            fees_rebalance_total += fee_this

        pooled_after = max(0.0, pooled_pre - fee_this)
        dollars_b = pooled_after * w_b
        dollars_e = pooled_after * w_e
        cash = pooled_after * max(0.0, 1.0 - w_b - w_e)

        btc_sub = btc_full.loc[seg_start:seg_end]
        eth_sub = eth_full.loc[seg_start:seg_end]
        scale_b = (dollars_b / sb["base_val"]) if sb["base_val"] > 0 else 0.0
        scale_e = (dollars_e / se["base_val"]) if se["base_val"] > 0 else 0.0
        btc_leg = btc_sub * scale_b
        eth_leg = eth_sub * scale_e

        idx = btc_leg.index.union(eth_leg.index)
        combined = (btc_leg.reindex(idx).ffill().bfill().fillna(0.0)
                   + eth_leg.reindex(idx).ffill().bfill().fillna(0.0) + cash)
        if len(combined) == 0:
            continue
        pieces.append(combined)

        prev_b = float(btc_leg.iloc[-1]) if len(btc_leg) else dollars_b
        prev_e = float(eth_leg.iloc[-1]) if len(eth_leg) else dollars_e
        pooled = float(combined.iloc[-1])
        log_rows.append({"date": seg_start, "w_btc": w_b, "w_eth": w_e,
                         "fallback": fb, "fee_this": fee_this, "pooled_end": pooled})

    equity = pd.concat(pieces).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    return {"equity": equity, "weights_log": pd.DataFrame(log_rows),
           "fees_rebalance": fees_rebalance_total,
           "final_balance": float(equity.iloc[-1]) if len(equity) else start_balance,
           "btc_full": btc_full, "eth_full": eth_full}


# =================================================================== sweep

LOOKBACKS = (30, 60, 90, 180)
CADENCES = {"monthly": "MS", "quarterly": "QS", "semiannual": "2QS"}

SWEEP_GRID = [{"lookback_days": lb, "rebalance_freq": freq}
             for lb in LOOKBACKS for freq in CADENCES.values()]


def eval_config(btc_df, eth_df, lookback_days: int, rebalance_freq: str,
                market: MarketSpec = SPOT, rebalance_fee_rate: float | None = None) -> dict:
    global N_EVALUATED
    N_EVALUATED += 1
    res = run_portfolio_continuous_costed(btc_df, eth_df, rebalance_freq, "invvol",
                                          lookback_days, market=market,
                                          rebalance_fee_rate=rebalance_fee_rate)
    eq = res["equity"]
    return {"train": period_metrics(eq, TRAIN_START, TRAIN_END),
           "valid": period_metrics(eq, VALID_START, VALID_END),
           "fees_rebalance": res["fees_rebalance"]}


def run_sweep(data_dir: str = "data") -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    btc_df, eth_df = load_assets(data_dir)
    rows = []
    for cfg in SWEEP_GRID:
        r = eval_config(btc_df, eth_df, cfg["lookback_days"], cfg["rebalance_freq"])
        rows.append({"config": cfg, "train": r["train"], "valid": r["valid"],
                    "fees_rebalance": r["fees_rebalance"]})
        print(f"lookback={cfg['lookback_days']:>3}d freq={cfg['rebalance_freq']:<5} | "
              f"train final={r['train']['final_balance']:>9.1f} Sharpe={r['train']['sharpe']:>6.2f} "
              f"DD={r['train']['max_dd_pct']:>5.1f}% || "
              f"valid final={r['valid']['final_balance']:>9.1f} Sharpe={r['valid']['sharpe']:>6.2f} "
              f"DD={r['valid']['max_dd_pct']:>5.1f}% | rebal_fees=${r['fees_rebalance']:.2f}")
    print(f"\nconfigs evaluated this call: {len(SWEEP_GRID)} (N_EVALUATED so far: {N_EVALUATED})")
    return rows, btc_df, eth_df


def select_best(rows: list[dict]) -> dict:
    """Pre-registered selection rule (written before the sweep ran): rank
    by min(train_sharpe, valid_sharpe) -- guards against the train-loses/
    validation-wins overfit signature that sank R-37/R-38/R-40 -- tie-
    break on -valid_max_dd_pct."""
    def score(r):
        return (min(r["train"]["sharpe"], r["valid"]["sharpe"]), -r["valid"]["max_dd_pct"])
    return max(rows, key=score)


def neighbourhood_report(rows: list[dict], best: dict) -> None:
    """P4: is the winner a plateau or a peak? Print every config's valid
    Sharpe sorted, with the winner marked."""
    print("\n=== parameter neighbourhood (inner-validation Sharpe, sorted) ===")
    ordered = sorted(rows, key=lambda r: -r["valid"]["sharpe"])
    for r in ordered:
        mark = "  <== WINNER" if r["config"] == best["config"] else ""
        print(f"lookback={r['config']['lookback_days']:>3}d "
              f"freq={r['config']['rebalance_freq']:<5} "
              f"valid Sharpe={r['valid']['sharpe']:>6.2f} "
              f"train Sharpe={r['train']['sharpe']:>6.2f}{mark}")


# ================================================================ headline

def _bh_metrics(df: pd.DataFrame, start: str, end: str, market: MarketSpec,
                start_balance: float = 1000.0) -> dict:
    global N_BACKTESTS_TOTAL
    N_BACKTESTS_TOTAL += 1
    res = run_period(BuyAndHold(), df, start=start, end=end, market=market,
                     start_balance=start_balance)
    eq = res.equity
    return {"final_balance": float(eq.iloc[-1]), "sharpe": sharpe_ratio(eq.to_numpy()),
           "max_dd_pct": max_drawdown_pct(eq.to_numpy())}


def _solo_metrics(btc_df: pd.DataFrame, market: MarketSpec = SPOT) -> dict:
    global N_BACKTESTS_TOTAL
    N_BACKTESTS_TOTAL += 1
    eq = continuous_leg_equity(btc_df, market, None, start=FULL_START, end=FULL_END)
    return {"train": period_metrics(eq, TRAIN_START, TRAIN_END),
           "valid": period_metrics(eq, VALID_START, VALID_END), "_equity": eq}


def run_headline(data_dir: str = "data", market: MarketSpec = SPOT) -> dict:
    btc_df, eth_df = load_assets(data_dir)
    rows, _, _ = run_sweep(data_dir)
    best = select_best(rows)
    print(f"\nselected best config: {best['config']}")
    neighbourhood_report(rows, best)
    freq = best["config"]["rebalance_freq"]
    lb = best["config"]["lookback_days"]

    invvol_res = run_portfolio_continuous_costed(btc_df, eth_df, freq, "invvol", lb, market=market)
    fixed_res = run_portfolio_continuous_costed(btc_df, eth_df, freq, "fixed5050", market=market)
    solo_eq = continuous_leg_equity(btc_df, market, None, start=FULL_START, end=FULL_END)

    invvol = {"train": period_metrics(invvol_res["equity"], TRAIN_START, TRAIN_END),
             "valid": period_metrics(invvol_res["equity"], VALID_START, VALID_END)}
    fixed = {"train": period_metrics(fixed_res["equity"], TRAIN_START, TRAIN_END),
            "valid": period_metrics(fixed_res["equity"], VALID_START, VALID_END)}
    solo = {"train": period_metrics(solo_eq, TRAIN_START, TRAIN_END),
           "valid": period_metrics(solo_eq, VALID_START, VALID_END)}
    bh = {"train": _bh_metrics(btc_df, TRAIN_START, TRAIN_END, market),
         "valid": _bh_metrics(btc_df, VALID_START, VALID_END, market)}

    print(f"\n=== HEADLINE (spot, {[k for k, v in CADENCES.items() if v == freq][0]} "
          f"rebalance, lookback={lb}d, costed engine) ===")
    header = f"{'candidate':<32} {'period':<6} {'final':>10} {'sharpe':>8} {'maxDD%':>8}"
    print(header)
    for name, table in (("inverse-vol (candidate)", invvol),
                        ("fixed 50/50 (re-derived ref)", fixed),
                        ("v4 BTC-solo (reference)", solo),
                        ("buy_and_hold BTC", bh)):
        for label in ("train", "valid"):
            m = table[label]
            print(f"{name:<32} {label:<6} {m['final_balance']:>10.1f} "
                  f"{m['sharpe']:>8.2f} {m['max_dd_pct']:>8.1f}")

    d_sharpe_solo = invvol["valid"]["sharpe"] - solo["valid"]["sharpe"]
    d_sharpe_fixed = invvol["valid"]["sharpe"] - fixed["valid"]["sharpe"]
    d_dd_solo = invvol["valid"]["max_dd_pct"] - solo["valid"]["max_dd_pct"]
    d_dd_fixed = invvol["valid"]["max_dd_pct"] - fixed["valid"]["max_dd_pct"]
    print(f"\nvalid ΔSharpe vs v4-solo:      {d_sharpe_solo:+.2f}")
    print(f"valid ΔSharpe vs fixed-50/50:  {d_sharpe_fixed:+.2f}")
    print(f"valid ΔmaxDD vs v4-solo:       {d_dd_solo:+.1f}pp")
    print(f"valid ΔmaxDD vs fixed-50/50:   {d_dd_fixed:+.1f}pp")
    print(f"\ntotal N_EVALUATED (invvol candidate configs): {N_EVALUATED}")
    print(f"total N_BACKTESTS_TOTAL (all distinct backtests so far): {N_BACKTESTS_TOTAL}")
    return {"best": best, "rows": rows, "invvol": invvol, "fixed": fixed,
           "solo": solo, "bh": bh, "freq": freq, "lb": lb, "market": market,
           "btc_df": btc_df, "eth_df": eth_df}


# =============================================================== diagnostics

def causality_check(data_dir: str = "data", lookback_days: int = 60,
                    rebalance_freq: str = "QS") -> bool:
    """Truncation tamper probe on this file's new code path (inverse-vol
    weight computation + rebalance-cost-bearing capital walk), modeled on
    R-50/R-42's own `causality_check[_continuous]`."""
    btc_df, eth_df = load_assets(data_dir)
    cut = pd.Timestamp("2021-06-30", tz="UTC")
    K = 137.0

    def tamper(df: pd.DataFrame, factor: float) -> pd.DataFrame:
        out = df.copy()
        mask = out.index > cut
        for col in ("open", "high", "low", "close"):
            out.loc[mask, col] = out.loc[mask, col] * factor
        return out

    base = run_portfolio_continuous_costed(btc_df, eth_df, rebalance_freq, "invvol", lookback_days)
    up = run_portfolio_continuous_costed(tamper(btc_df, K), tamper(eth_df, K),
                                         rebalance_freq, "invvol", lookback_days)
    down = run_portfolio_continuous_costed(tamper(btc_df, 1.0 / K), tamper(eth_df, 1.0 / K),
                                           rebalance_freq, "invvol", lookback_days)

    pre = base["equity"].index <= cut
    b = base["equity"][pre].to_numpy()
    u = up["equity"].reindex(base["equity"].index)[pre].to_numpy()
    d = down["equity"].reindex(base["equity"].index)[pre].to_numpy()
    max_diff_up = float(np.nanmax(np.abs(b - u)))
    max_diff_down = float(np.nanmax(np.abs(b - d)))
    ok = max_diff_up < 1e-6 and max_diff_down < 1e-6
    print(f"causality check (b19 costed engine): cut={cut.date()}, K={K}, "
          f"lookback={lookback_days}d, freq={rebalance_freq}")
    print(f"  max |base - up-tampered| pooled equity before cut: {max_diff_up:.3e}")
    print(f"  max |base - down-tampered| pooled equity before cut: {max_diff_down:.3e}")
    print(f"  PASS (pooled equity before cut unchanged): {ok}")
    return ok


def artifact_check(btc_df, eth_df, lookback_days: int, rebalance_freq: str,
                   market: MarketSpec = SPOT) -> dict:
    """F1: R^2 exposure-artifact diagnostic -- candidate vs flat-rescaled
    v4-BTC-solo (continuous), and vs the re-derived fixed-50/50 reference
    (same cadence, same costed engine), on both inner splits."""
    print("\n=== F1: exposure-artifact diagnostics ===")
    invvol_res = run_portfolio_continuous_costed(btc_df, eth_df, rebalance_freq, "invvol",
                                                 lookback_days, market=market)
    fixed_res = run_portfolio_continuous_costed(btc_df, eth_df, rebalance_freq, "fixed5050",
                                                market=market)
    solo_eq = continuous_leg_equity(btc_df, market, None, start=FULL_START, end=FULL_END)

    out = {}
    fail = False
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)), ("valid", (VALID_START, VALID_END))):
        cand_sub = invvol_res["equity"].loc[s:e]
        fixed_sub = fixed_res["equity"].loc[s:e]
        solo_sub = solo_eq.loc[s:e]
        r2_solo = r_squared(cand_sub, solo_sub)
        r2_fixed = r_squared(cand_sub, fixed_sub)
        flag_solo = "ARTIFACT (R^2>0.95)" if r2_solo > 0.95 else "ok"
        if r2_solo > 0.95:
            fail = True
        print(f"[{label}] candidate vs flat-rescaled v4-BTC-solo: R^2={r2_solo:.4f} -> {flag_solo}")
        print(f"[{label}] candidate vs fixed-50/50 reference:     R^2={r2_fixed:.4f}")
        out[label] = {"r2_solo": r2_solo, "r2_fixed": r2_fixed}
    out["F1_pass"] = not fail
    print(f"\nF1 (exposure-artifact falsification test): {'PASS' if out['F1_pass'] else 'FAIL'}")
    return out


def feetier_check(btc_df, eth_df, lookback_days: int, rebalance_freq: str,
                  market_kind: str = "spot") -> dict:
    """F2: 0.40% Bitstamp taker tier, applied to BOTH each leg's own v4
    trading and the portfolio-level rebalance cost, reusing
    `scripts/fee_study.py`'s own BITSTAMP_TAKER convention."""
    BITSTAMP_TAKER = 0.004
    market_04 = MarketSpec.spot(fee_rate=BITSTAMP_TAKER) if market_kind == "spot" \
        else MarketSpec.futures(leverage=5.0, fee_rate=BITSTAMP_TAKER)

    print(f"\n=== F2: 0.40% taker fee-tier stress test ({market_kind}) ===")
    invvol_res = run_portfolio_continuous_costed(btc_df, eth_df, rebalance_freq, "invvol",
                                                 lookback_days, market=market_04)
    fixed_res = run_portfolio_continuous_costed(btc_df, eth_df, rebalance_freq, "fixed5050",
                                                market=market_04)
    bh = _bh_metrics(btc_df, VALID_START, VALID_END, market_04)

    invvol_valid = period_metrics(invvol_res["equity"], VALID_START, VALID_END)
    fixed_valid = period_metrics(fixed_res["equity"], VALID_START, VALID_END)

    d_sharpe = invvol_valid["sharpe"] - fixed_valid["sharpe"]
    beats_bh = invvol_valid["final_balance"] > bh["final_balance"]
    print(f"inner-valid @0.40%: candidate final=${invvol_valid['final_balance']:.0f} "
          f"Sharpe={invvol_valid['sharpe']:.2f}")
    print(f"inner-valid @0.40%: fixed-50/50 final=${fixed_valid['final_balance']:.0f} "
          f"Sharpe={fixed_valid['sharpe']:.2f}")
    print(f"inner-valid @0.40%: buy_and_hold final=${bh['final_balance']:.0f} "
          f"Sharpe={bh['sharpe']:.2f}")
    print(f"ΔSharpe (candidate - fixed50/50) @0.40%: {d_sharpe:+.2f}")
    print(f"candidate beats buy_and_hold @0.40%: {beats_bh}")
    f2_pass = (d_sharpe >= 0.0) and beats_bh
    print(f"\nF2 (fee-tier falsification test): {'PASS' if f2_pass else 'FAIL'}")
    return {"invvol_valid": invvol_valid, "fixed_valid": fixed_valid, "bh": bh,
           "d_sharpe": d_sharpe, "beats_bh": beats_bh, "F2_pass": f2_pass}


# =================================================================== futures

def futures_check(lookback_days: int, rebalance_freq: str, data_dir: str = "data") -> dict:
    btc_df, eth_df = load_assets(data_dir)
    return run_headline_for_market(btc_df, eth_df, lookback_days, rebalance_freq, FUTURES5X)


def run_headline_for_market(btc_df, eth_df, lookback_days: int, rebalance_freq: str,
                            market: MarketSpec) -> dict:
    invvol_res = run_portfolio_continuous_costed(btc_df, eth_df, rebalance_freq, "invvol",
                                                 lookback_days, market=market)
    fixed_res = run_portfolio_continuous_costed(btc_df, eth_df, rebalance_freq, "fixed5050",
                                                market=market)
    solo_eq = continuous_leg_equity(btc_df, market, None, start=FULL_START, end=FULL_END)
    bh = {"train": _bh_metrics(btc_df, TRAIN_START, TRAIN_END, market),
         "valid": _bh_metrics(btc_df, VALID_START, VALID_END, market)}

    invvol = {"train": period_metrics(invvol_res["equity"], TRAIN_START, TRAIN_END),
             "valid": period_metrics(invvol_res["equity"], VALID_START, VALID_END)}
    fixed = {"train": period_metrics(fixed_res["equity"], TRAIN_START, TRAIN_END),
            "valid": period_metrics(fixed_res["equity"], VALID_START, VALID_END)}
    solo = {"train": period_metrics(solo_eq, TRAIN_START, TRAIN_END),
           "valid": period_metrics(solo_eq, VALID_START, VALID_END)}

    print(f"\n=== {market.name} headline (lookback={lookback_days}d, freq={rebalance_freq}) ===")
    for name, table in (("inverse-vol (candidate)", invvol), ("fixed 50/50", fixed),
                        ("v4 BTC-solo", solo), ("buy_and_hold", bh)):
        for label in ("train", "valid"):
            m = table[label]
            print(f"{name:<28} {label:<6} final={m['final_balance']:>10.1f} "
                  f"Sharpe={m['sharpe']:>6.2f} DD={m['max_dd_pct']:>5.1f}%")
    return {"invvol": invvol, "fixed": fixed, "solo": solo, "bh": bh}


# ===================================================================== holdout

def holdout(lookback_days: int, rebalance_freq: str) -> dict:
    """Step 4: the ONE pre-registered holdout read for this claim. Run
    ONLY if F1 and F2 both PASS on the inner splits. Uses the full, uncut
    BTC/ETH frames imported from `kelly_regime_dual_fixed.py` (unmodified)
    -- the same gating precedent `kelly_regime_dual_bootstrap.py` (R-43)
    set -- sliced to >= OOS_START ("2023-01-01") here, and ONLY here.
    """
    from experiments.kelly_regime_dual_fixed import BTC, ETH  # noqa: E402  (full, uncut)

    print(f"=== PRE-REGISTERED HOLDOUT READ ({OOS_START} onward) === "
          f"lookback={lookback_days}d freq={rebalance_freq}")
    holdout_end = str(min(BTC.index[-1], ETH.index[-1]).date())

    btc_h = BTC.loc[OOS_START:]
    eth_h = ETH.loc[OOS_START:]

    out = {}
    for tag, market, fee in (("0.10% baseline", SPOT, None), ("0.40% real taker",
                              MarketSpec.spot(fee_rate=0.004), None)):
        invvol_res = run_portfolio_continuous_costed(
            btc_h, eth_h, rebalance_freq, "invvol", lookback_days, market=market,
            full_start=OOS_START, full_end=holdout_end)
        fixed_res = run_portfolio_continuous_costed(
            btc_h, eth_h, rebalance_freq, "fixed5050", market=market,
            full_start=OOS_START, full_end=holdout_end)
        solo_eq = continuous_leg_equity(btc_h, market, None, start=OOS_START, end=holdout_end)
        bh = _bh_metrics(btc_h, OOS_START, holdout_end, market)

        cand = period_metrics(invvol_res["equity"], OOS_START, holdout_end)
        fixed = period_metrics(fixed_res["equity"], OOS_START, holdout_end)
        solo = period_metrics(solo_eq, OOS_START, holdout_end)

        print(f"\n--- {tag} ---")
        print(f"candidate:    final=${cand['final_balance']:.0f} Sharpe={cand['sharpe']:.2f} "
              f"DD={cand['max_dd_pct']:.1f}%")
        print(f"fixed 50/50:  final=${fixed['final_balance']:.0f} Sharpe={fixed['sharpe']:.2f} "
              f"DD={fixed['max_dd_pct']:.1f}%")
        print(f"v4 BTC-solo:  final=${solo['final_balance']:.0f} Sharpe={solo['sharpe']:.2f} "
              f"DD={solo['max_dd_pct']:.1f}%")
        print(f"buy_and_hold: final=${bh['final_balance']:.0f} Sharpe={bh['sharpe']:.2f} "
              f"DD={bh['max_dd_pct']:.1f}%")
        out[tag] = {"candidate": cand, "fixed5050": fixed, "v4_solo": solo, "buy_and_hold": bh}

    return out


# ===================================================================== CLI

def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "sweep":
        run_sweep()
    elif cmd == "select":
        run_headline()
    elif cmd == "causality":
        causality_check()
    elif cmd == "artifact":
        btc_df, eth_df = load_assets()
        rows, _, _ = run_sweep()
        best = select_best(rows)
        artifact_check(btc_df, eth_df, best["config"]["lookback_days"],
                       best["config"]["rebalance_freq"])
    elif cmd == "feetier":
        btc_df, eth_df = load_assets()
        rows, _, _ = run_sweep()
        best = select_best(rows)
        feetier_check(btc_df, eth_df, best["config"]["lookback_days"],
                      best["config"]["rebalance_freq"])
    elif cmd == "futures":
        rows, btc_df, eth_df = run_sweep()
        best = select_best(rows)
        futures_check(best["config"]["lookback_days"], best["config"]["rebalance_freq"])
    elif cmd == "holdout":
        rows, _, _ = run_sweep()
        best = select_best(rows)
        holdout(best["config"]["lookback_days"], best["config"]["rebalance_freq"])
    elif cmd == "all":
        out = run_headline()
        best = out["best"]
        causality_check(lookback_days=best["config"]["lookback_days"],
                        rebalance_freq=best["config"]["rebalance_freq"])
        artifact_check(out["btc_df"], out["eth_df"], best["config"]["lookback_days"],
                       best["config"]["rebalance_freq"])
        feetier_check(out["btc_df"], out["eth_df"], best["config"]["lookback_days"],
                      best["config"]["rebalance_freq"])
        futures_check(best["config"]["lookback_days"], best["config"]["rebalance_freq"])
        print(f"\ntotal N_EVALUATED (invvol candidate configs): {N_EVALUATED}")
        print(f"total N_BACKTESTS_TOTAL (all distinct backtests): {N_BACKTESTS_TOTAL}")
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
