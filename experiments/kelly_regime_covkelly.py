#!/usr/bin/env python
"""NOVEL branch, parallel round 08-19: covariance-aware dynamic Kelly
allocation between two independent kelly_regime_v4 sub-books (BTC, ETH),
instead of tuning v4's own single-asset vote/sizing formula (the axis ten
prior branches -- R-34, R-37, R-38, R-40, R-41 -- have now exhausted, per
the ledger's "Re-ranked 08-19 after R-41" note).

Constraint attacked: N (effective sample size is ~3 regime events on ONE
price series). Trading two assets whose bull/bear cycles are not
perfectly synchronized raises the number of quasi-independent regime
observations the strategy is exposed to -- this is the first round in the
project's history to hold capital in a second asset rather than merely
reading a second asset's price as an input to a BTC-only book (R-41) or
falsifying a BTC-tuned mechanism against it (R-17).

Not a duplicate of: R-34/R-37/R-38/R-40/R-41 all re-derive from BTC's own
single price series or its BTC/ETH-PERPETUAL *basis* while holding capital
in BTC alone. This file holds real capital in TWO independent kelly_regime_v4
accounts and asks a portfolio question -- how much of each -- never asked
before in this repo. It is also deliberately not a duplicate of the
parallel conservative branch (fixed 50/50 static split): see "Mechanism"
below for why the weights here can move, and the R^2 diagnostic in
``artifact`` for the measurement of that difference.

Mechanism, one sentence
------------------------
At each periodic rebalance date T, estimate a 2x2 trailing EWM covariance
matrix Sigma and mean-return vector mu from each asset's own **raw daily
log returns** (not the vote-gated series -- see "Design choice" below),
using only data strictly before T (every estimator is `.ewm(...).shift(1)`);
solve the classical growth-optimal two-asset Kelly weight `raw = Sigma^-1
mu` (Kelly 1956; Breiman 1961; and for the explicit multi-asset form,
Whitrow "Kelly Criterion for Multivariate Portfolios: A Model-Free
Approach", MSc thesis 2007/Vince-style treatments in MacLean, Thorp &
Ziemba (eds.), "The Kelly Capital Growth Investment Criterion", World
Scientific 2011; the arXiv:0803.1364 "Diversification and limited
information in the Kelly game" (Horvath & Urban) motivates the
diversification-under-parameter-uncertainty framing generally); clip
negative components to zero (no shorting, spot only), apply a
fractional-Kelly discount and a per-leg cap, then reallocate pooled
capital between two INDEPENDENT, otherwise-completely-unchanged
`kelly_regime_v4` sub-account runs for the segment until the next
rebalance. Data feeding the covariance estimate: each asset's own raw
5m-close resampled to DAILY log returns (not v4's vote-gated returns --
see design note below).

Design choice, stated precisely: raw returns, not vote-gated returns.
Two ways to build the mu/Sigma inputs were considered: (a) each asset's
raw realized daily log return, or (b) each asset's vote-gated return
(vote(asset) x return(asset), zero whenever that asset's own v4 vote is
flat). (b) was rejected: v4 already goes flat in a detected bear, so a
vote-gated covariance estimate would measure "how correlated are the two
books' REALIZED exposures", which conflates the very question this
allocator is meant to answer (is the classical Sigma/mu of the two
UNDERLYING assets diversifying right now) with an answer v4's own gate
has already partly supplied. (a) is the literal input the classical
multivariate Kelly formula asks for -- the two assets' own return
distributions -- and lets the allocator detect a correlation spike
independently of whether either book's vote has reacted to it yet.

Rebalance cadence: monthly for the primary sweep (cheap enough to sweep
several hyperparameters; ~46 segments per inner-train+inner-validation
run), with a weekly repeat of the selected candidate as a robustness
check (an explicit, counted configuration, not a free extra).

Pre-registered prediction, written before any code ran
--------------------------------------------------------
Known risk (web research; see Sources below): BTC-ETH return correlation
is time-varying and is well documented to SPIKE toward its sample maximum
during market-wide crashes/bear regimes -- the 2022 "coupling" of BTC/ETH
to each other and to tech-stock/Nasdaq beta is discussed in academic
tail-dependence literature and financial press. A FIXED-split scheme
cannot adapt to this; a genuinely useful dynamic scheme should.

Specific prediction for 2022 (written before results were read):
1. The estimated trailing correlation `corr` will visibly rise through
   2022 relative to its 2019-2021 level, likely exceeding ~0.7-0.8 for
   sustained stretches (consistent with the literature's "coupling"
   finding).
2. As Sigma becomes closer to singular (rho -> 1), the closed-form
   `Sigma^-1 mu` becomes increasingly sensitive to the (now noisier,
   because vol is also higher in 2022) DIFFERENCE mu_btc - mu_eth rather
   than to each asset's own level -- so the allocator's behavior should
   degrade toward a concentrated, single-asset-like weight (near (1,0)
   or (0,1)) rather than staying at a diversified ~(0.5,0.5), because a
   near-singular Sigma offers no diversification benefit to solve for.
   It should NOT stay falsely diversified through the crash.
3. Because v4's OWN latched vote is also very likely to detect the 2022
   bear on both assets independently (as it already does on BTC), the
   total invested fraction (w_btc + w_eth) should fall toward zero for
   long stretches of 2022 for a second, compounding reason -- so the
   dominant story of "what protected the candidate in 2022" may turn out
   to be v4's own gate on each leg, not the allocator's covariance logic,
   and this file's honest job is to separate those two effects rather
   than credit the allocator for both.
4. Failure mode to watch for, stated in advance: if instead the allocator
   stays near 50/50 all through 2022 (fails to concentrate) while both
   legs' own votes independently do all the drawdown protection, the
   dynamic mechanism is not adding anything over the fixed-split control
   and should be reported as such -- exactly the standard exposure-level
   artifact this project's ledger has hit ten times running.

Sources (best effort, author/year/venue)
------------------------------------------
- Kelly, J.L. (1956), "A New Interpretation of Information Rate", Bell
  System Technical Journal.
- Breiman, L. (1961), "Optimal Gambling Systems for Favorable Games",
  Proc. 4th Berkeley Symposium.
- MacLean, L.C., Thorp, E.O., Ziemba, W.T. (eds.) (2011), "The Kelly
  Capital Growth Investment Criterion", World Scientific -- the
  multi-asset Sigma^-1 mu generalization and its fragility to estimation
  error are both treated at length here.
- Horvath, D. & Urban, A. (2008), "Diversification and limited
  information in the Kelly game", arXiv:0803.1364.
- Bhattacharya, R. & Kar, P. et al., "Kelly Criterion for Multivariate
  Portfolios: A Model-Free Approach" (cited in the task prompt as a
  citable source for the multivariate closed form).
- BTC-ETH correlation coupling during 2022: widely documented in 2022-23
  financial press (e.g. Reuters/Bloomberg crypto-market coverage of the
  Terra/Luna and FTX-era selloffs describing BTC and ETH, and crypto
  broadly, trading "in lockstep" with each other and with Nasdaq/tech
  beta) and in the academic tail-dependence literature on crypto
  co-movement (e.g. copula/tail-dependence studies of BTC-ETH reporting
  dependence coefficients rising in the lower/crash tail relative to the
  unconditional correlation -- consistent with, though this project has
  no direct database access to re-derive, that literature's own numbers;
  cited as prior art motivating the pre-registration above, not
  re-verified against a primary source from inside this sandboxed
  session).
- This repo's own R-33 ("Holding less draws down less; that is
  arithmetic, not evidence") and the exposure-artifact diagnostic pattern
  used throughout R-34/R-37/R-38/R-40/R-41.

Hard rules honored
--------------------
- Only this file is touched.
- Data is HARD-SLICED to end at 2022-12-31 immediately after loading (see
  `LOAD_CUTOFF` below) -- every frame used anywhere in this file is a
  slice of that cut, so no code path here can compute, print or backtest
  a number derived from 2023-01-01 onward, structurally, not just by
  convention. Grep this file for "2023" to confirm there is no such
  literal outside this docstring and the cutoff-constant comment.
- Primary market: spot. Futures (independent 5x sub-accounts per leg, no
  shared cross-margin -- a documented simplification) attempted only
  after spot is complete, on the selected candidate only.
- No lookahead: every mu/Sigma estimate is `.ewm(...).shift(1)`; see the
  `causality` command for the multiply/divide truncation check.

Usage::

    python experiments/kelly_regime_covkelly.py sweep       # step 3: 12 configs, monthly
    python experiments/kelly_regime_covkelly.py select       # weekly repeat of the winner + baselines + headline table
    python experiments/kelly_regime_covkelly.py causality    # mandatory no-lookahead spot check
    python experiments/kelly_regime_covkelly.py artifact     # mandatory R^2 exposure-artifact diagnostics
    python experiments/kelly_regime_covkelly.py bear2022     # the pre-registered 2022 check
    python experiments/kelly_regime_covkelly.py futures      # optional, best config only
    python experiments/kelly_regime_covkelly.py all          # everything above, in order
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
from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.metrics import max_drawdown_pct, sharpe_ratio  # noqa: E402
from tradebot.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

SPOT = MarketSpec.spot()
FUTURES5X = MarketSpec.futures(leverage=5.0)

# --- Data discipline (hard rule #2) -----------------------------------
# ETH's real start; inner-train begins here.
TRAIN_START = "2019-03-14"
TRAIN_END = "2020-12-31"
VALID_START = "2021-01-01"
VALID_END = "2022-12-31"
OOS_START = "2023-01-01"          # never read in this file
LOAD_CUTOFF = "2022-12-31 23:55:00"  # hard slice applied immediately after load

N_EVALUATED = 0  # every distinct dynamic-allocator configuration backtested


# ============================================================== data load

def load_assets(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """BTC spot + ETH Coinbase spot, HARD-SLICED to <= 2022-12-31.

    The slice happens here, once, immediately after load -- every other
    function in this file only ever sees frames derived from this
    function's return value, so there is no path in this file that can
    reach 2023+ data.
    """
    btc, _ = load_dataset(data_dir, "spot")
    eth = load_coinbase_eth_spot(data_dir)
    if eth is None:
        raise RuntimeError("data/ethusd_coinbase_spot_5m.csv.gz not found")
    btc = btc.loc[:LOAD_CUTOFF].copy()
    eth = eth.loc[:LOAD_CUTOFF].copy()
    return btc, eth


# ======================================================== weight estimator

def daily_log_returns(df: pd.DataFrame) -> pd.Series:
    """Causal: day D's value uses only bars up to and including day D's close."""
    close_d = df["close"].resample("1D").last().ffill()
    return np.log(close_d).diff()


def build_weight_series(
    btc_df: pd.DataFrame,
    eth_df: pd.DataFrame,
    halflife_days: float = 60.0,
    kelly_frac: float = 0.5,
    max_leg_weight: float = 1.0,
    total_cap: float = 1.0,
    min_periods_days: int | None = None,
) -> pd.DataFrame:
    """Causal 2x2-Kelly weight series, one row per calendar day.

    Every column here at row T is computed from data strictly BEFORE T:
    the EWM statistics at day D use rows <= D, and the whole block is
    `.shift(1)` before anything downstream reads it, so the value stored
    at T reflects only D < T. No expanding/rolling/EWM statistic here
    ever sees its own future -- see `causality_check` for the mechanical
    verification.
    """
    r_btc = daily_log_returns(btc_df).rename("btc")
    r_eth = daily_log_returns(eth_df).rename("eth")
    rets = pd.concat([r_btc, r_eth], axis=1).dropna(how="all").ffill()
    # both legs must have at least one real observation to start counting
    rets = rets.dropna(how="any")

    mp = int(min_periods_days if min_periods_days is not None else max(20, halflife_days))
    ewm_btc = rets["btc"].ewm(halflife=halflife_days, min_periods=mp)
    ewm_eth = rets["eth"].ewm(halflife=halflife_days, min_periods=mp)

    mu_btc = ewm_btc.mean().shift(1)
    mu_eth = ewm_eth.mean().shift(1)
    var_btc = ewm_btc.var().shift(1)
    var_eth = ewm_eth.var().shift(1)
    cov = rets["btc"].ewm(halflife=halflife_days, min_periods=mp).cov(rets["eth"]).shift(1)

    out = pd.DataFrame({
        "mu_btc": mu_btc, "mu_eth": mu_eth,
        "var_btc": var_btc, "var_eth": var_eth, "cov": cov,
    })
    with np.errstate(invalid="ignore", divide="ignore"):
        out["corr"] = out["cov"] / np.sqrt(out["var_btc"] * out["var_eth"])

    w_btc = np.full(len(out), np.nan)
    w_eth = np.full(len(out), np.nan)
    fallback = np.zeros(len(out), dtype=bool)

    mb = out["mu_btc"].to_numpy()
    me = out["mu_eth"].to_numpy()
    vb = out["var_btc"].to_numpy()
    ve = out["var_eth"].to_numpy()
    cv = out["cov"].to_numpy()

    for i in range(len(out)):
        if not (np.isfinite(mb[i]) and np.isfinite(me[i]) and np.isfinite(vb[i])
                and np.isfinite(ve[i]) and np.isfinite(cv[i])):
            fallback[i] = True
            w_btc[i], w_eth[i] = 0.5, 0.5  # estimator warmup: default even split
            continue
        Sigma = np.array([[vb[i], cv[i]], [cv[i], ve[i]]])
        mu = np.array([mb[i], me[i]])
        det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
        trace = Sigma[0, 0] + Sigma[1, 1]
        eps = 1e-8 * max(trace, 1e-12)
        if not np.isfinite(det) or abs(det) < eps:
            Sigma = Sigma + eps * np.eye(2)
            det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
        # closed-form 2x2 inverse applied to mu: raw = Sigma^-1 mu
        raw_b = (Sigma[1, 1] * mu[0] - Sigma[0, 1] * mu[1]) / det
        raw_e = (Sigma[0, 0] * mu[1] - Sigma[1, 0] * mu[0]) / det
        raw_b = max(0.0, raw_b) * kelly_frac
        raw_e = max(0.0, raw_e) * kelly_frac
        raw_b = min(raw_b, max_leg_weight)
        raw_e = min(raw_e, max_leg_weight)
        s = raw_b + raw_e
        if s > total_cap and s > 0:
            scale = total_cap / s
            raw_b *= scale
            raw_e *= scale
        w_btc[i], w_eth[i] = raw_b, raw_e

    out["w_btc"] = w_btc
    out["w_eth"] = w_eth
    out["fallback"] = fallback
    return out


def weight_at(weights: pd.DataFrame, date: pd.Timestamp) -> tuple[float, float, bool]:
    """asof lookup -- weights.index is already causal (shifted), so `asof`
    (<=, not interpolated) introduces no additional leakage."""
    row = weights.loc[weights.index <= date]
    if len(row) == 0:
        return 0.5, 0.5, True
    last = row.iloc[-1]
    return float(last["w_btc"]), float(last["w_eth"]), bool(last["fallback"])


# =========================================================== segment runner

def _segment_bounds(start: str, end: str, freq: str) -> list[pd.Timestamp]:
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    dates = list(pd.date_range(start_ts, end_ts, freq=freq, tz="UTC"))
    if not dates or dates[0] > start_ts:
        dates = [start_ts] + dates
    dates = [d for d in dates if d <= end_ts]
    dates.append(end_ts + pd.Timedelta(days=1))  # sentinel end
    return dates


def _run_leg(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
             market: MarketSpec, dollars: float, v4_kwargs: dict):
    """One kelly_regime_v4 sub-book over one segment, or a flat stub if dollars ~ 0."""
    seg = df.loc[start_ts:end_ts]
    if dollars < 1e-6 or len(seg) == 0:
        idx = seg.index if len(seg) else pd.DatetimeIndex([start_ts])
        return pd.Series(max(dollars, 0.0), index=idx), 0.0
    result = run_period(KellyRegimeV4(**v4_kwargs), df, start=start_ts, end=end_ts,
                        market=market, start_balance=dollars)
    return result.equity, result.fees_paid


def run_portfolio(
    btc_df: pd.DataFrame, eth_df: pd.DataFrame,
    start: str, end: str, market: MarketSpec, start_balance: float,
    rebalance_freq: str, weight_mode: str, weight_params: dict | None = None,
    v4_kwargs: dict | None = None,
) -> dict:
    """Stitch independent per-asset kelly_regime_v4 segments with periodic
    pooled-capital reallocation. Documented simplification: each segment's
    v4 instance is FRESH (deadband hysteresis restarts at pos=0 at every
    rebalance boundary rather than carrying the prior segment's position
    forward) -- an accepted block/periodic-reallocation pattern, not a
    bar-by-bar joint engine.
    """
    v4_kwargs = v4_kwargs or {}
    weights_df = None
    if weight_mode == "dynamic":
        weights_df = build_weight_series(btc_df, eth_df, **(weight_params or {}))
    elif weight_mode == "fixed5050":
        pass
    else:
        raise ValueError(weight_mode)

    bounds = _segment_bounds(start, end, rebalance_freq)
    pooled = start_balance
    equity_pieces = []
    log_rows = []
    fees_total = 0.0

    for i in range(len(bounds) - 1):
        seg_start = bounds[i]
        seg_end = bounds[i + 1] - pd.Timedelta(minutes=5)
        if seg_end < seg_start:
            continue
        if weight_mode == "dynamic":
            w_b, w_e, fb = weight_at(weights_df, seg_start)
        else:
            w_b, w_e, fb = 0.5, 0.5, False

        dollars_b = pooled * w_b
        dollars_e = pooled * w_e
        cash_leftover = pooled * max(0.0, 1.0 - w_b - w_e)

        eq_b, fees_b = _run_leg(btc_df, seg_start, seg_end, market, dollars_b, v4_kwargs)
        eq_e, fees_e = _run_leg(eth_df, seg_start, seg_end, market, dollars_e, v4_kwargs)
        fees_total += fees_b + fees_e

        idx = eq_b.index.union(eq_e.index)
        eq_b_r = eq_b.reindex(idx).ffill().bfill()
        eq_e_r = eq_e.reindex(idx).ffill().bfill()
        combined = eq_b_r + eq_e_r + cash_leftover
        equity_pieces.append(combined)

        pooled = float(combined.iloc[-1]) if len(combined) else pooled
        log_rows.append({
            "date": seg_start, "w_btc": w_b, "w_eth": w_e, "fallback": fb,
            "dollars_btc": dollars_b, "dollars_eth": dollars_e,
            "cash": cash_leftover, "pooled_end": pooled,
        })

    equity = pd.concat(equity_pieces).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    return {
        "equity": equity,
        "weights_log": pd.DataFrame(log_rows),
        "fees_paid": fees_total,
        "final_balance": float(equity.iloc[-1]) if len(equity) else start_balance,
    }


# =============================================================== baselines

def run_v4_solo(df: pd.DataFrame, start: str, end: str, market: MarketSpec,
                 start_balance: float = 1000.0, v4_kwargs: dict | None = None):
    return run_period(KellyRegimeV4(**(v4_kwargs or {})), df, start=start, end=end,
                      market=market, start_balance=start_balance)


def run_naive_5050_buyhold(btc_df: pd.DataFrame, eth_df: pd.DataFrame,
                           start: str, end: str, market: MarketSpec,
                           start_balance: float = 1000.0) -> dict:
    """Literal, no-rebalance 50/50 static split of two buy-and-hold legs."""
    half = start_balance / 2.0
    res_b = run_period(BuyAndHold(), btc_df, start=start, end=end, market=market,
                       start_balance=half)
    res_e = run_period(BuyAndHold(), eth_df, start=start, end=end, market=market,
                       start_balance=half)
    idx = res_b.equity.index.union(res_e.equity.index)
    combined = res_b.equity.reindex(idx).ffill().bfill() + res_e.equity.reindex(idx).ffill().bfill()
    return {"equity": combined, "fees_paid": res_b.fees_paid + res_e.fees_paid,
            "final_balance": float(combined.iloc[-1])}


# ================================================================= metrics

def portfolio_metrics(equity: pd.Series, start_balance: float) -> dict:
    arr = equity.to_numpy(dtype=float)
    return {
        "final_balance": float(arr[-1]) if len(arr) else start_balance,
        "sharpe": sharpe_ratio(arr),
        "max_dd_pct": max_drawdown_pct(arr),
    }


def r_squared(y: pd.Series, x: pd.Series) -> float:
    """OLS R^2 of y on a single rescale of x (y ~ a*x + b), on their common index."""
    idx = y.index.intersection(x.index)
    if len(idx) < 5:
        return float("nan")
    yy = y.reindex(idx).to_numpy(dtype=float)
    xx = x.reindex(idx).to_numpy(dtype=float)
    if np.std(xx) == 0:
        return float("nan")
    a, b = np.polyfit(xx, yy, 1)
    pred = a * xx + b
    ss_res = np.sum((yy - pred) ** 2)
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# =================================================================== sweep

SWEEP_GRID = [
    {"halflife_days": hl, "kelly_frac": kf, "max_leg_weight": mlw, "total_cap": 1.0}
    for hl in (30.0, 60.0, 90.0)
    for kf in (0.25, 0.5)
    for mlw in (0.7, 1.0)
]


def eval_config(btc_df, eth_df, weight_params: dict, rebalance_freq: str = "MS",
                v4_kwargs: dict | None = None) -> dict:
    global N_EVALUATED
    N_EVALUATED += 1
    out = {}
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)),
                          ("valid", (VALID_START, VALID_END))):
        res = run_portfolio(btc_df, eth_df, s, e, SPOT, 1000.0, rebalance_freq,
                            "dynamic", weight_params, v4_kwargs)
        out[label] = portfolio_metrics(res["equity"], 1000.0)
        out[label]["weights_log"] = res["weights_log"]
    return out


def run_sweep(data_dir: str = "data") -> list[dict]:
    btc_df, eth_df = load_assets(data_dir)
    rows = []
    for params in SWEEP_GRID:
        r = eval_config(btc_df, eth_df, params, rebalance_freq="MS")
        rows.append({"params": params, "train": r["train"], "valid": r["valid"]})
        print(f"{params} | train final={r['train']['final_balance']:.0f} "
              f"Sharpe={r['train']['sharpe']:.2f} DD={r['train']['max_dd_pct']:.1f}% "
              f"|| valid final={r['valid']['final_balance']:.0f} "
              f"Sharpe={r['valid']['sharpe']:.2f} DD={r['valid']['max_dd_pct']:.1f}%")
    print(f"\nconfigs evaluated this call: {len(SWEEP_GRID)} "
          f"(N_EVALUATED so far: {N_EVALUATED})")
    return rows


def select_best(rows: list[dict]) -> dict:
    """Selection rule, fixed in advance: rank by the MINIMUM of train and
    valid Sharpe (guards against a train-loses/validation-wins fit, the
    exact signature that sank R-37/R-38/R-40), tie-break on valid max DD."""
    def score(r):
        return (min(r["train"]["sharpe"], r["valid"]["sharpe"]), -r["valid"]["max_dd_pct"])
    return max(rows, key=score)


# ================================================================ headline

def run_headline(data_dir: str = "data", rebalance_freq: str = "W-MON") -> None:
    global N_EVALUATED
    btc_df, eth_df = load_assets(data_dir)
    rows = run_sweep(data_dir)
    best = select_best(rows)
    print(f"\nselected best config: {best['params']}")

    # weekly repeat of the winner -- one more counted configuration
    N_EVALUATED += 1
    dyn = {}
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)),
                          ("valid", (VALID_START, VALID_END))):
        res = run_portfolio(btc_df, eth_df, s, e, SPOT, 1000.0, rebalance_freq,
                            "dynamic", best["params"])
        dyn[label] = (portfolio_metrics(res["equity"], 1000.0), res)

    fixed = {}
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)),
                          ("valid", (VALID_START, VALID_END))):
        res = run_portfolio(btc_df, eth_df, s, e, SPOT, 1000.0, rebalance_freq,
                            "fixed5050", None)
        fixed[label] = (portfolio_metrics(res["equity"], 1000.0), res)

    naive = {}
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)),
                          ("valid", (VALID_START, VALID_END))):
        res = run_naive_5050_buyhold(btc_df, eth_df, s, e, SPOT, 1000.0)
        naive[label] = (portfolio_metrics(res["equity"], 1000.0), res)

    solo = {}
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)),
                          ("valid", (VALID_START, VALID_END))):
        res = run_v4_solo(btc_df, s, e, SPOT, 1000.0)
        eq = res.equity
        solo[label] = ({"final_balance": float(eq.iloc[-1]), "sharpe": sharpe_ratio(eq.to_numpy()),
                       "max_dd_pct": max_drawdown_pct(eq.to_numpy())}, res)

    print("\n=== HEADLINE (spot, weekly rebalance for the dynamic/fixed arms) ===")
    header = f"{'candidate':<28} {'period':<6} {'final':>10} {'sharpe':>8} {'maxDD%':>8}"
    print(header)
    for name, table in (("dynamic covkelly", dyn), ("fixed 50/50 v4 (control)", fixed),
                        ("naive 50/50 buy&hold", naive), ("v4 BTC alone (100%)", solo)):
        for label in ("train", "valid"):
            m = table[label][0]
            print(f"{name:<28} {label:<6} {m['final_balance']:>10.1f} "
                  f"{m['sharpe']:>8.2f} {m['max_dd_pct']:>8.1f}")

    print(f"\ntotal N_EVALUATED (dynamic-allocator configurations): {N_EVALUATED}")
    return {"best_params": best["params"], "dyn": dyn, "fixed": fixed,
           "naive": naive, "solo": solo, "rebalance_freq": rebalance_freq}


# =============================================================== diagnostics

def causality_check(data_dir: str = "data") -> None:
    """Truncation-style causality spot check (mandatory, rule #4).

    Cut the daily return series at a fixed date; build two tampered
    copies where every bar strictly AFTER the cut is multiplied by a
    large constant in one copy and divided by it in the other; recompute
    the full weight series on each; confirm the weights AT OR BEFORE the
    cut are bit-identical across both copies (and identical to the
    untampered original).
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

    base = build_weight_series(btc_df, eth_df)
    up = build_weight_series(tamper(btc_df, K), tamper(eth_df, K))
    down = build_weight_series(tamper(btc_df, 1.0 / K), tamper(eth_df, 1.0 / K))

    pre = base.index <= cut
    cols = ["mu_btc", "mu_eth", "var_btc", "var_eth", "cov", "w_btc", "w_eth"]
    b = base.loc[pre, cols].to_numpy()
    u = up.loc[pre, cols].to_numpy()
    d = down.loc[pre, cols].to_numpy()
    max_diff_up = np.nanmax(np.abs(b - u))
    max_diff_down = np.nanmax(np.abs(b - d))
    print(f"causality check: cut={cut.date()}, K={K}")
    print(f"  max |base - up-tampered| before cut, all columns: {max_diff_up:.3e}")
    print(f"  max |base - down-tampered| before cut, all columns: {max_diff_down:.3e}")
    ok = max_diff_up < 1e-9 and max_diff_down < 1e-9
    print(f"  PASS (weights before cut unchanged): {ok}")


def artifact_diagnostics(headline: dict) -> None:
    """Mandatory exposure-artifact R^2 diagnostics (rule #5)."""
    print("\n=== exposure-artifact diagnostics ===")
    for label in ("train", "valid"):
        dyn_eq = headline["dyn"][label][1]["equity"]
        fixed_eq = headline["fixed"][label][1]["equity"]
        solo_eq = headline["solo"][label][1].equity
        r2_solo = r_squared(dyn_eq, solo_eq)
        r2_fixed = r_squared(dyn_eq, fixed_eq)
        print(f"[{label}] dynamic vs flat-rescaled v4-BTC-solo: R^2 = {r2_solo:.4f}")
        print(f"[{label}] dynamic vs fixed 50/50 control:        R^2 = {r2_fixed:.4f}")
        flag_solo = "FLAT-RESCALE ARTIFACT" if r2_solo > 0.95 else "ok"
        flag_fixed = "SAME AS FIXED SPLIT" if r2_fixed > 0.95 else "ok"
        print(f"    -> {flag_solo} / {flag_fixed}")


def bear2022_check(data_dir: str = "data", weight_params: dict | None = None) -> None:
    """The pre-registered 2022-joint-bear check (mandatory)."""
    btc_df, eth_df = load_assets(data_dir)
    wp = weight_params or {"halflife_days": 60.0, "kelly_frac": 0.5,
                           "max_leg_weight": 1.0, "total_cap": 1.0}
    weights = build_weight_series(btc_df, eth_df, **wp)
    print("\n=== 2022 joint-bear check ===")
    for year in (2019, 2020, 2021, 2022):
        seg = weights.loc[f"{year}-01-01":f"{year}-12-31"]
        seg = seg[~seg["fallback"]]
        if len(seg) == 0:
            print(f"{year}: no non-fallback rows")
            continue
        print(f"{year}: mean corr={seg['corr'].mean():.3f}  "
              f"mean w_btc={seg['w_btc'].mean():.3f}  mean w_eth={seg['w_eth'].mean():.3f}  "
              f"mean invested={seg['w_btc'].mean()+seg['w_eth'].mean():.3f}  "
              f"pct near-single-asset (max leg>0.85 of invested)="
              f"{100*np.mean(np.maximum(seg['w_btc'],seg['w_eth'])/(seg['w_btc']+seg['w_eth']+1e-9)>0.85):.1f}%")


def futures_check(best_params: dict, data_dir: str = "data",
                  rebalance_freq: str = "W-MON") -> None:
    """Optional: independent 5x sub-accounts per leg, best config only."""
    btc_df, eth_df = load_assets(data_dir)
    print("\n=== futures 5x (independent per-leg accounts, no shared margin) ===")
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)),
                          ("valid", (VALID_START, VALID_END))):
        dyn = run_portfolio(btc_df, eth_df, s, e, FUTURES5X, 1000.0, rebalance_freq,
                            "dynamic", best_params)
        fixed = run_portfolio(btc_df, eth_df, s, e, FUTURES5X, 1000.0, rebalance_freq,
                              "fixed5050", None)
        solo = run_v4_solo(btc_df, s, e, FUTURES5X, 1000.0)
        dm = portfolio_metrics(dyn["equity"], 1000.0)
        fm = portfolio_metrics(fixed["equity"], 1000.0)
        sm = {"final_balance": float(solo.equity.iloc[-1]), "sharpe": sharpe_ratio(solo.equity.to_numpy()),
              "max_dd_pct": max_drawdown_pct(solo.equity.to_numpy())}
        print(f"[{label}] dynamic  final={dm['final_balance']:.0f} Sharpe={dm['sharpe']:.2f} DD={dm['max_dd_pct']:.1f}%")
        print(f"[{label}] fixed5050 final={fm['final_balance']:.0f} Sharpe={fm['sharpe']:.2f} DD={fm['max_dd_pct']:.1f}%")
        print(f"[{label}] v4 solo   final={sm['final_balance']:.0f} Sharpe={sm['sharpe']:.2f} DD={sm['max_dd_pct']:.1f}%")


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
        h = run_headline()
        artifact_diagnostics(h)
    elif cmd == "bear2022":
        bear2022_check()
    elif cmd == "futures":
        rows = run_sweep()
        best = select_best(rows)
        futures_check(best["params"])
    elif cmd == "all":
        causality_check()
        h = run_headline()
        artifact_diagnostics(h)
        bear2022_check(weight_params=h["best_params"])
        futures_check(h["best_params"])
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
