#!/usr/bin/env python
"""kelly_regime_v4 as two independent, fixed-split sub-books (BTC + ETH), summed (CONSERVATIVE branch).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. ``kelly_regime_v4`` itself is
NOT modified, forked, or subclassed anywhere in this file -- it is called
unchanged, twice, via ``get_strategy("kelly_regime_v4")``.

The idea
--------
Every prior attempt to improve ``kelly_regime_v4`` (R-34, R-37, R-38,
R-40, R-41 -- ten branches across five parallel rounds) modified the
strategy's OWN single-asset BTC vote or sizing formula and failed, either
as a disguised flat-exposure rescale (R^2 > 0.95 against a flat rescale
of v4) or by losing to v4 on the pre-2020 BTC control before an ETH
falsification test is even read (the R-37/38/40 "train-loses,
validation-wins" signature). The standing ledger note after R-41 reads
the single-asset SIZE axis as likely exhausted for this strategy family.

This branch attacks a DIFFERENT constraint: **N ~ 3**, effective sample
size. A single BTC price series has been exposed to roughly three
independent regime events (2017-18, 2020-21, 2021-22) in this project's
whole history -- that is the number, not the 1.01M five-minute bars, that
governs how much can be learned. Trading a SECOND asset whose bull/bear
cycles are not perfectly synchronized with BTC's does not manufacture new
information about BTC's regimes, but it does raise the number of
quasi-independent regime *exposures* the strategy's capital sees, in the
same sense that adding a second, imperfectly-correlated bet to a Kelly
book raises the number of quasi-independent trials the bettor is exposed
to without touching the edge on either bet alone.

Mechanism (one sentence, written before any code ran)
-------------------------------------------------------
Kelly (1956) / Breiman (1961) growth-optimal betting extends to multiple
SIMULTANEOUS bets: for bets that are genuinely independent (or only
partially correlated), fractional allocation across them is safe and the
combined portfolio's long-run growth rate approaches the SUM of the
individual bets' own optimal growth rates as their pairwise correlation
falls toward zero (see the classical multivariate-Kelly result via the
covariance matrix, alpha = C^-1 (mu - r), collected in MacLean, Thorp &
Ziemba, "The Kelly Capital Growth Investment Criterion", World Scientific
2011; and the correlated-bets analysis in "Diversification and limited
information in the Kelly game", arXiv:0803.1364) -- so running
``kelly_regime_v4`` completely unchanged, independently, on BTC and on
ETH, each funded with a fixed fraction of total capital, should produce a
SUMMED equity curve whose drawdown is shallower than either single-asset
book alone, in proportion to how imperfectly the two assets' bad regimes
overlap.

Pre-registered failure mode (written before any code ran) -- THE central
falsification test of this branch
------------------------------------------------------------------------
BTC-ETH return correlation is NOT constant: it is well documented to be
time-varying and to SPIKE toward its highest level specifically DURING
market-wide crashes and bear regimes, not during calm periods -- the
opposite of when diversification is needed least. Academic tail-dependence
work on exactly this pair finds the same pattern: "Crashing Together,
Rallying Apart: Dynamic Conditional Tail Dependence in Cryptocurrency
Markets" (arXiv:2606.16840) measures BTC-ETH lower-tail dependence at an
edge density near 0.94 through the 2022 bear market -- i.e. almost every
severe joint move was a JOINT crash, not an idiosyncratic one -- and
reports lower-tail dominance (joint crash risk exceeding joint boom risk)
as the rule rather than the exception across crypto pairs generally.
Kettani et al., "Cryptocurrency Market Maturation and Evolving Risk
Profiles: A Comparative Analysis of Bitcoin and Ethereum Tail Risk
Dynamics" (FinTech 5(2), 2025, doi:10.3390/fintech5020028) and a
nonparametric-dependence study in Statistical Methods & Applications
(Springer, 2026) report the same asymmetric, downturn-amplified BTC-ETH
dependence structure. ``kelly_regime_v4``'s entire value proposition in
this project (L-01, R-19, R-33, R-36) is drawdown protection specifically
AROUND a crash. If BTC's and ETH's bad regimes coincide -- which the
literature above says they do, especially in 2022 -- then a fixed-split
dual book gains almost nothing in drawdown terms over a single-asset book
during precisely the regime the whole mechanism is supposed to help with,
even though it may still look fine in calm periods. **This is checked
explicitly below** (``bearcheck``), on the mandated
2021-01-01..2022-12-31 inner-validation window, which contains the 2022
bear by construction.

Constraint attacked
--------------------
N~3 -- a second, imperfectly-synchronized regime-cycle exposure -- NOT
the SIZE axis (v4's own vote/sizing formula is untouched) and NOT INFO in
the R-41 sense (this uses ETH's own real price, not a derived signal fed
into v4's BTC decision).

Not a duplicate of
-------------------
- R-34/R-37/R-38/R-40/R-41 (ten branches, five rounds): every one of them
  reworked v4's OWN vote or sizing formula on the SAME BTC price series.
  This file changes neither; it runs the unmodified strategy on a second,
  independently-transacted asset and only decides how much STARTING
  CAPITAL each copy gets. There is no new signal fed into v4's decision
  process at all.
- R-17/R-28's ETH usage: both used ETH only as a FALSIFICATION check --
  replay a BTC-tuned mechanism on ETH and see if the property survives.
  Capital never sat in the ETH book; it was a read-only control. Here ETH
  is a live, capital-holding leg of the portfolio for the first time.
- R-41 (Deribit basis): used a second, real, independently-transacted
  price series (BTC perp vs BTC spot) as an INPUT into v4's single BTC
  decision (INFO axis). This file uses a second, real, independently-
  transacted ASSET (ETH) as a second capital-holding BOOK, running v4
  twice rather than feeding it a new input once.

Simulable here?
----------------
Yes. Two independent, unmodified backtests (``tradebot.window.run_period``,
the same call every other experiment file in this repo uses) on committed
real OHLCV, summed bar-by-bar. No engine change.

What would make it fail (falsification, restated as a decision question)
--------------------------------------------------------------------------
Does the dual book's max drawdown over 2021-01-01..2022-12-31 (the
mandated inner-validation window, which contains the 2022 BTC/ETH joint
bear) actually improve versus ``kelly_regime_v4`` run on BTC alone with
full capital? If the correlation-spike literature above is right, the
answer should be "little or not at all", even if the SAME comparison
looks better in the calmer inner-train window (2019-03-14..2020-12-31).
That asymmetry -- if it appears -- is the honest result, not a bug.

Usage
-----
    python experiments/kelly_regime_dual_fixed.py sweep       # step 3, inner-train, splits x spot
    python experiments/kelly_regime_dual_fixed.py select      # step 5, inner-validation, splits x both markets
    python experiments/kelly_regime_dual_fixed.py baselines   # the three required baselines, both windows
    python experiments/kelly_regime_dual_fixed.py bearcheck   # the pre-registered 2021-22 joint-bear check
    python experiments/kelly_regime_dual_fixed.py artifact    # exposure-artifact R^2 vs a flat rescale of BTC-only v4
    python experiments/kelly_regime_dual_fixed.py causality   # two-opposite-tampers probe, both legs + split ratio
    python experiments/kelly_regime_dual_fixed.py all
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.engine import BacktestResult, run_backtest  # noqa: E402
from tradebot.metrics import max_drawdown_pct, sharpe_ratio  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

INCUMBENT = "kelly_regime_v4"
HOLD = "buy_and_hold"

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

# ---------------------------------------------------------------- data rule
# Inner-train = 2019-03-14 (ETH's real start on the committed Coinbase
# file) -> 2020-12-31. Inner-validation = 2021-01-01 -> 2022-12-31 (the
# 2022 BTC/ETH joint bear). NOTHING in this file reads, backtests on, or
# prints a number derived from 2023-01-01 onward. Every date-bounded slice
# in this file is either within [TRAIN[0], VALID[1]] or explicitly capped
# at ":2022-12-31" (the causality probes' data window) -- grep this file
# for "2023"/"2024"/"2025"/"2026" to confirm: only variable names/comments
# describing that boundary, and two literature citation years, appear.
TRAIN = ("2019-03-14", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")

BTC, BTC_LABEL = load_dataset(ROOT / "data", "spot")
ETH = load_coinbase_eth_spot(ROOT / "data")
if ETH is None:
    raise RuntimeError("data/ethusd_coinbase_spot_5m.csv.gz not found -- cannot run this experiment")
ETH_LABEL = "real (Coinbase spot)"

N_EVALUATED = 0  # distinct split configurations searched (routine's trials count)
_SEEN_CONFIGS: set[str] = set()

OUT = ROOT / "reports" / "kelly_regime_dual_fixed"


# --------------------------------------------------------------------- split weights


def _annualized_vol(df: pd.DataFrame, start: str, end: str) -> float:
    """Realized annualized vol of log returns, restricted STRICTLY to [start, end].

    Used only to build the ``vol_weighted`` split candidate from
    inner-train data. Reads no row outside ``[start, end]`` by
    construction -- ``.loc[start:end]`` raises/slices, it does not widen.
    """
    window = df.loc[start:end, "close"]
    r = np.log(window).diff().dropna()
    return float(r.std(ddof=1) * np.sqrt(BARS_PER_YEAR))


# vol-weighted split computed ONCE, from TRAIN data only (never VALID),
# inverse-vol (risk-parity style): the higher-vol asset gets less capital.
_BTC_TRAIN_VOL = _annualized_vol(BTC, *TRAIN)
_ETH_TRAIN_VOL = _annualized_vol(ETH, *TRAIN)
_VOL_W_BTC = (1.0 / _BTC_TRAIN_VOL) / (1.0 / _BTC_TRAIN_VOL + 1.0 / _ETH_TRAIN_VOL)

SPLITS: dict[str, tuple[float, float]] = {
    "50_50": (0.50, 0.50),
    "60_40_btc": (0.60, 0.40),
    "70_30_btc": (0.70, 0.30),
    "40_60_eth": (0.40, 0.60),
    "30_70_eth": (0.30, 0.70),
    "vol_weighted": (round(_VOL_W_BTC, 4), round(1.0 - _VOL_W_BTC, 4)),
}


# --------------------------------------------------------------------- legs and combination


def run_leg(strategy_name: str, df: pd.DataFrame, label: str, start: str, end: str,
            market: MarketSpec, balance: float) -> BacktestResult:
    """One independent, unmodified strategy run on one asset over one period."""
    strat = get_strategy(strategy_name)
    return run_period(strat, df, start, end, market=market, start_balance=balance, data_label=label)


def combine_equity(res_a: BacktestResult, res_b: BacktestResult) -> pd.Series:
    """Sum two independent equity curves onto their union bar grid.

    Forward-fill each leg between its own bar closes (its equity is
    piecewise-constant there by construction); any leg with no
    observation yet at the very start of the union grid is filled with
    its own ``start_balance`` (flat, not yet resolved), never with a
    forward-looking value.
    """
    idx = res_a.equity.index.union(res_b.equity.index)
    ea = res_a.equity.reindex(idx).ffill().fillna(res_a.start_balance)
    eb = res_b.equity.reindex(idx).ffill().fillna(res_b.start_balance)
    return (ea + eb).rename("equity")


def portfolio_metrics(equity: pd.Series) -> dict:
    eq = equity.to_numpy(dtype=float)
    return {
        "final": float(eq[-1]),
        "sharpe": sharpe_ratio(eq),
        "max_dd": max_drawdown_pct(eq),
    }


def run_dual(split_name: str, start: str, end: str, market: MarketSpec,
             total: float = 1_000.0, count: bool = False) -> dict:
    """Run kelly_regime_v4 independently on BTC and ETH with a fixed capital split, summed."""
    global N_EVALUATED
    if count and split_name not in _SEEN_CONFIGS:
        _SEEN_CONFIGS.add(split_name)
        N_EVALUATED += 1
    w_btc, w_eth = SPLITS[split_name]
    res_btc = run_leg(INCUMBENT, BTC, BTC_LABEL, start, end, market, total * w_btc)
    res_eth = run_leg(INCUMBENT, ETH, ETH_LABEL, start, end, market, total * w_eth)
    equity = combine_equity(res_btc, res_eth)
    m = portfolio_metrics(equity)
    return {"split": split_name, "w_btc": w_btc, "w_eth": w_eth, "equity": equity,
            "res_btc": res_btc, "res_eth": res_eth, **m}


def run_baseline_v4_btc(start: str, end: str, market: MarketSpec, total: float = 1_000.0) -> dict:
    res = run_leg(INCUMBENT, BTC, BTC_LABEL, start, end, market, total)
    return {"label": "kelly_regime_v4 BTC-only", "equity": res.equity, "res": res,
            **portfolio_metrics(res.equity)}


def run_baseline_hold_dual(start: str, end: str, market: MarketSpec, total: float = 1_000.0) -> dict:
    res_btc = run_leg(HOLD, BTC, BTC_LABEL, start, end, market, total * 0.5)
    res_eth = run_leg(HOLD, ETH, ETH_LABEL, start, end, market, total * 0.5)
    equity = combine_equity(res_btc, res_eth)
    return {"label": "naive 50/50 buy&hold BTC+ETH", "equity": equity,
            "res_btc": res_btc, "res_eth": res_eth, **portfolio_metrics(equity)}


def run_baseline_hold_btc(start: str, end: str, market: MarketSpec, total: float = 1_000.0) -> dict:
    res = run_leg(HOLD, BTC, BTC_LABEL, start, end, market, total)
    return {"label": "buy_and_hold BTC-only", "equity": res.equity, "res": res,
            **portfolio_metrics(res.equity)}


def line(tag: str, m: dict) -> str:
    return (f"  {tag:38s} final=${m['final']:>10,.0f} "
            f"sharpe={m['sharpe']:>6.2f} maxDD={m['max_dd']:>5.1f}%")


# --------------------------------------------------------------------------- step 3


def sweep() -> pd.DataFrame:
    """Step 3: every split, inner-train, spot only (the required minimum)."""
    print(f"vol-weighted split, computed from TRAIN ({TRAIN[0]}..{TRAIN[1]}) ONLY: "
          f"BTC train vol={_BTC_TRAIN_VOL:.3f}/yr, ETH train vol={_ETH_TRAIN_VOL:.3f}/yr "
          f"-> w_btc={_VOL_W_BTC:.4f}, w_eth={1 - _VOL_W_BTC:.4f}")
    rows = []
    t0 = time.time()
    for name in SPLITS:
        d = run_dual(name, *TRAIN, market=SPOT, count=True)
        rows.append({"split": name, "w_btc": d["w_btc"], "w_eth": d["w_eth"], "market": "spot",
                     "window": "train", "final": d["final"], "sharpe": d["sharpe"], "max_dd": d["max_dd"]})
        print(f"[{N_EVALUATED}] " + line(f"{name} ({d['w_btc']:.2f}/{d['w_eth']:.2f})", d) +
              f"  [{time.time() - t0:.0f}s]")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "sweep_inner_train.csv", index=False)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    print(f"written: {OUT / 'sweep_inner_train.csv'}")
    return out


# --------------------------------------------------------------------------- step 5


def select() -> pd.DataFrame:
    """Step 5: every split, inner-validation, BOTH markets."""
    rows = []
    for name in SPLITS:
        for mname, market in MARKETS:
            d = run_dual(name, *VALID, market=market, count=True)
            rows.append({"split": name, "w_btc": d["w_btc"], "w_eth": d["w_eth"], "market": mname,
                         "window": "validation", "final": d["final"], "sharpe": d["sharpe"],
                         "max_dd": d["max_dd"]})
        s, f = rows[-2], rows[-1]
        print(f"{name:14s} ({s['w_btc']:.2f}/{s['w_eth']:.2f})  "
              f"spot: ${s['final']:>9,.0f} sh{s['sharpe']:>6.2f} DD{s['max_dd']:>5.1f}%   "
              f"fut: ${f['final']:>9,.0f} sh{f['sharpe']:>6.2f} DD{f['max_dd']:>5.1f}%")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "select_inner_validation.csv", index=False)
    print(f"\nconfigurations evaluated total: {N_EVALUATED}")
    print(f"written: {OUT / 'select_inner_validation.csv'}")
    return out


# ------------------------------------------------------------------------ baselines


def baselines() -> pd.DataFrame:
    """The three mandated baselines, both windows, both markets."""
    rows = []
    for wname, (start, end) in (("train", TRAIN), ("validation", VALID)):
        for mname, market in MARKETS:
            a = run_baseline_v4_btc(start, end, market)
            b = run_baseline_hold_dual(start, end, market)
            c = run_baseline_hold_btc(start, end, market)
            for lbl, d in ((a["label"], a), (b["label"], b), (c["label"], c)):
                rows.append({"baseline": lbl, "window": wname, "market": mname,
                             "final": d["final"], "sharpe": d["sharpe"], "max_dd": d["max_dd"]})
            print(f"\n-- {wname} / {mname} --")
            print(line(a["label"], a))
            print(line(b["label"], b))
            print(line(c["label"], c))
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "baselines.csv", index=False)
    print(f"\nwritten: {OUT / 'baselines.csv'}")
    return out


# ------------------------------------------------------------------------ bear check


def bearcheck(split_name: str = "50_50") -> None:
    """The pre-registered 2021-01-01..2022-12-31 joint-bear check.

    Does the dual book's drawdown over the window containing the 2022
    BTC/ETH joint bear actually improve versus kelly_regime_v4 BTC-only,
    or does the predicted correlation-spike failure mode show up? Run for
    every split so the answer isn't cherry-picked to one candidate.
    """
    print(f"\n=== 2022-joint-bear check, inner-validation {VALID[0]}..{VALID[1]} (spot) ===")
    ctl = run_baseline_v4_btc(*VALID, SPOT)
    print(line("kelly_regime_v4 BTC-only (control)", ctl))
    for name in SPLITS:
        d = run_dual(name, *VALID, market=SPOT)
        delta_dd = d["max_dd"] - ctl["max_dd"]
        verdict = "IMPROVED" if delta_dd < -1.0 else ("WORSE" if delta_dd > 1.0 else "NO CHANGE (~noise)")
        print(f"  {name:14s} ({d['w_btc']:.2f}/{d['w_eth']:.2f})  DD={d['max_dd']:5.1f}% "
              f"vs control {ctl['max_dd']:5.1f}%  delta={delta_dd:+.1f}pp  [{verdict}]")

    print(f"\n=== same check, inner-train {TRAIN[0]}..{TRAIN[1]} (spot), for the train-vs-validation signature ===")
    ctl_t = run_baseline_v4_btc(*TRAIN, SPOT)
    print(line("kelly_regime_v4 BTC-only (control)", ctl_t))
    for name in SPLITS:
        d = run_dual(name, *TRAIN, market=SPOT)
        delta_dd = d["max_dd"] - ctl_t["max_dd"]
        print(f"  {name:14s} ({d['w_btc']:.2f}/{d['w_eth']:.2f})  DD={d['max_dd']:5.1f}% "
              f"vs control {ctl_t['max_dd']:5.1f}%  delta={delta_dd:+.1f}pp  "
              f"final=${d['final']:,.0f} vs ${ctl_t['final']:,.0f}  "
              f"[{'beats v4' if d['final'] > ctl_t['final'] else 'LOSES to v4'} on return]")


# ------------------------------------------------------------------------ exposure-artifact diagnostic


def exposure_artifact_check(split_name: str = "50_50") -> None:
    """Mandatory diagnostic: is the dual book just relabeled leverage on v4-BTC-alone?

    Portfolio-level "aggregate exposure" = sum over legs of
    (leg's own target fraction x leg's own equity) / total portfolio
    equity -- the leveraged fraction of TOTAL capital the book is
    carrying, bar by bar. Compared, via R^2, against a mean-matched flat
    rescale of BTC-only v4's own target series (v4's target IS its
    exposure fraction of ITS OWN equity; since BTC-only v4 holds all the
    capital, its target series already lives on the same "fraction of
    total capital" scale). R^2 > 0.95 is this project's standing
    signature of a fake win (R-33/R-34/R-37/R-38/R-40/R-41).
    """
    print(f"\nexposure-artifact check ({split_name}, inner-validation, mean-matched flat rescale of v4 BTC-only):")
    for mname, market in MARKETS:
        d = run_dual(split_name, *VALID, market=market)
        ctl = run_baseline_v4_btc(*VALID, market)

        rb, re = d["res_btc"], d["res_eth"]
        idx = rb.df.index.union(re.df.index)
        tgt_b = rb.df["target"].reindex(idx).ffill().fillna(0.0)
        eq_b = rb.equity.reindex(idx).ffill().fillna(rb.start_balance)
        tgt_e = re.df["target"].reindex(idx).ffill().fillna(0.0)
        eq_e = re.equity.reindex(idx).ffill().fillna(re.start_balance)
        total_eq = eq_b + eq_e
        exposure = (tgt_b * eq_b + tgt_e * eq_e) / total_eq

        v4_tgt = ctl["res"].df["target"].reindex(idx).ffill().fillna(0.0)

        mask = np.isfinite(exposure.to_numpy()) & np.isfinite(v4_tgt.to_numpy())
        y = exposure.to_numpy()[mask]
        x_raw = v4_tgt.to_numpy()[mask]
        c = float(np.mean(y)) / float(np.mean(x_raw)) if np.mean(x_raw) != 0 else float("nan")
        x = c * x_raw

        ss_res = float(np.sum((y - x) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
        verdict = ("EXPOSURE-LEVEL ARTIFACT (R^2 > 0.95)" if np.isfinite(r2) and r2 > 0.95
                    else "not a flat rescale by this test")
        print(f"  {mname}: rescale c={c:.3f}  corr={corr:.4f}  R^2={r2:.4f}  {verdict}")


# ------------------------------------------------------------------------ causality


def _price_tamper_probe(df: pd.DataFrame, label: str, cut: int, tag: str) -> bool:
    """Standard two-opposite-tampers probe for kelly_regime_v4 on one asset."""
    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    pa = get_strategy(INCUMBENT).prepare(up.copy())
    pb = get_strategy(INCUMBENT).prepare(down.copy())
    a = pa["target"].to_numpy(dtype=float)[:cut]
    b = pb["target"].to_numpy(dtype=float)[:cut]
    worst = float(np.nanmax(np.abs(a - b)))
    ok = worst < 1e-9
    print(f"  [{tag}] max|target diff| before cut = {worst:.3e}  {'PASS' if ok else 'FAIL'}")

    ra = run_backtest(get_strategy(INCUMBENT), up.iloc[:cut + 1], SPOT, 1_000.0, data_label=label)
    rb = run_backtest(get_strategy(INCUMBENT), down.iloc[:cut + 1], SPOT, 1_000.0, data_label=label)
    worst_eq = float(np.max(np.abs(ra.equity.to_numpy()[:cut] - rb.equity.to_numpy()[:cut])))
    ok2 = worst_eq < 1e-6
    print(f"  [{tag}] max|equity diff| before cut = {worst_eq:.3e}  {'PASS' if ok2 else 'FAIL'}")
    return ok and ok2


def causality() -> None:
    """Truncation-style causality checks, both legs plus the split ratio and the combine step.

    1. The standard two-opposite-tampers probe against kelly_regime_v4
       run on BTC (this is re-verifying what test_causality_strict.py
       already covers for a registered strategy -- cheap, and the task
       asks for it explicitly).
    2. The same probe against kelly_regime_v4 run on ETH -- the strategy
       is unmodified, but this is the first time in the project it holds
       capital on ETH rather than only being read as a falsification
       control, so it is worth checking directly rather than assuming.
    3. The vol_weighted split ratio: proof by construction that it reads
       ONLY the TRAIN window (`.loc[TRAIN[0]:TRAIN[1]]`), demonstrated by
       recomputing it after corrupting every bar strictly after TRAIN's
       end and confirming the ratio is bit-identical.
    4. The combine step: confirm the SUMMED portfolio equity curve before
       a cut date is unchanged when bars after that cut (on both legs) are
       tampered in opposite directions -- the wrapper this file adds on
       top of two already-causal backtests introduces no lookahead of its
       own.
    """
    pre_2023_btc = BTC.loc[:"2022-12-31"]
    df_btc = pre_2023_btc.iloc[-300_000:].copy()
    cut_btc = len(df_btc) - 5_000

    pre_2023_eth = ETH.loc[:"2022-12-31"]
    df_eth = pre_2023_eth.iloc[-300_000:].copy()
    cut_eth = len(df_eth) - 5_000

    print("=== probe 1: kelly_regime_v4 on BTC ===")
    ok1 = _price_tamper_probe(df_btc, BTC_LABEL, cut_btc, "BTC")

    print("=== probe 2: kelly_regime_v4 on ETH ===")
    ok2 = _price_tamper_probe(df_eth, ETH_LABEL, cut_eth, "ETH")

    print("=== probe 3: vol_weighted split ratio reads TRAIN only ===")
    # NOTE: TRAIN[1] = "2020-12-31" is read INCLUSIVELY by pandas label
    # slicing (.loc[...] covers the whole day 2020-12-31 00:00..23:55), so
    # the corruption boundary must be the START of the NEXT day, not
    # `index > Timestamp(TRAIN[1])` (which is midnight and would corrupt
    # 2020-12-31's own bars -- bars TRAIN itself reads -- a test-harness
    # bug caught by this very probe on first run, not a strategy bug).
    after_train_ts = pd.Timestamp(TRAIN[1], tz="UTC") + pd.Timedelta(days=1)
    btc_corrupted = BTC.copy()
    after_train = btc_corrupted.index >= after_train_ts
    for col in ("open", "high", "low", "close"):
        btc_corrupted.loc[after_train, col] *= 1000.0
    eth_corrupted = ETH.copy()
    after_train_e = eth_corrupted.index >= after_train_ts
    for col in ("open", "high", "low", "close"):
        eth_corrupted.loc[after_train_e, col] *= 1000.0
    vol_btc_c = _annualized_vol(btc_corrupted, *TRAIN)
    vol_eth_c = _annualized_vol(eth_corrupted, *TRAIN)
    w_btc_c = (1.0 / vol_btc_c) / (1.0 / vol_btc_c + 1.0 / vol_eth_c)
    diff = abs(w_btc_c - _VOL_W_BTC)
    ok3 = diff < 1e-12
    print(f"  original w_btc={_VOL_W_BTC:.10f}  after corrupting all post-TRAIN bars x1000: "
          f"w_btc={w_btc_c:.10f}  diff={diff:.3e}  {'PASS' if ok3 else 'FAIL'}")

    print("=== probe 4: combined portfolio equity before cut, both legs tampered ===")
    # A shared calendar window (ETH's own last-300k-bar span before 2023,
    # since ETH's grid is the shorter/sparser of the two) sliced onto BOTH
    # assets, so "before the cut" means the same calendar instant on both
    # legs rather than the same row position on two different grids.
    shared_start, shared_end = df_eth.index[0], df_eth.index[-1]
    df_btc_shared = BTC.loc[shared_start:shared_end].copy()
    df_eth_shared = df_eth.copy()
    cut_date = df_eth_shared.index[-5_000]  # same convention as probes 1-2: 5,000 bars before the end
    cut_b = int(df_btc_shared.index.searchsorted(cut_date))
    cut_e = int(df_eth_shared.index.searchsorted(cut_date))

    up_b, down_b = df_btc_shared.copy(), df_btc_shared.copy()
    up_e, down_e = df_eth_shared.copy(), df_eth_shared.copy()
    for col in ("open", "high", "low", "close"):
        up_b.iloc[cut_b:, up_b.columns.get_loc(col)] *= 3.0
        down_b.iloc[cut_b:, down_b.columns.get_loc(col)] /= 3.0
        up_e.iloc[cut_e:, up_e.columns.get_loc(col)] *= 3.0
        down_e.iloc[cut_e:, down_e.columns.get_loc(col)] /= 3.0

    # Full-length runs (no truncation to the cut), exactly like probes 1-2 -
    # only bars AT OR AFTER the cut are tampered, and only bars STRICTLY
    # BEFORE the cut are compared, so the tampered bar at the cut itself
    # (included in both tampered copies, by construction of `iloc[cut:]`)
    # never enters the "before cut" comparison window.
    ra_b = run_backtest(get_strategy(INCUMBENT), up_b, SPOT, 500.0, data_label=BTC_LABEL)
    ra_e = run_backtest(get_strategy(INCUMBENT), up_e, SPOT, 500.0, data_label=ETH_LABEL)
    rb_b = run_backtest(get_strategy(INCUMBENT), down_b, SPOT, 500.0, data_label=BTC_LABEL)
    rb_e = run_backtest(get_strategy(INCUMBENT), down_e, SPOT, 500.0, data_label=ETH_LABEL)

    eq_up = combine_equity(ra_b, ra_e)
    eq_down = combine_equity(rb_b, rb_e)
    common_idx = eq_up.index.intersection(eq_down.index)
    common_idx = common_idx[common_idx < cut_date]
    worst = float(np.max(np.abs(eq_up.reindex(common_idx).to_numpy() -
                                 eq_down.reindex(common_idx).to_numpy())))
    ok4 = worst < 1e-6
    print(f"  max|combined equity diff| strictly before cut ({cut_date}) = {worst:.3e}  "
          f"{'PASS' if ok4 else 'FAIL'}")

    ok = ok1 and ok2 and ok3 and ok4
    print(f"\noverall (probes 1-4): {'PASS' if ok else 'FAIL'} -- "
          "no decision at or before any cut moves when only post-cut data is tampered")


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"BTC: {len(BTC):,} bars {BTC.index[0]:%Y-%m-%d} -> {BTC.index[-1]:%Y-%m-%d} (data: {BTC_LABEL})",
          file=sys.stderr)
    print(f"ETH: {len(ETH):,} bars {ETH.index[0]:%Y-%m-%d} -> {ETH.index[-1]:%Y-%m-%d} (data: {ETH_LABEL})",
          file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "sweep":
        sweep()
    elif choice == "select":
        select()
    elif choice == "baselines":
        baselines()
    elif choice == "bearcheck":
        bearcheck()
    elif choice == "artifact":
        exposure_artifact_check()
    elif choice == "causality":
        causality()
    elif choice == "all":
        sweep()
        select()
        baselines()
        bearcheck()
        exposure_artifact_check()
        causality()
    else:
        print("usage: python experiments/kelly_regime_dual_fixed.py "
              "[sweep|select|baselines|bearcheck|artifact|causality|all]")
