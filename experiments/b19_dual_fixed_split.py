#!/usr/bin/env python
"""B-19, CONSERVATIVE branch: is R-50's periodic-rebalance edge really a
rebalancing effect, or does a NEVER-rebalanced static split capture it?

Backlog item attacked: **B-19** -- "Does a periodically-rebalanced,
EQUAL-WEIGHT (static 50/50) BTC+ETH portfolio of ``kelly_regime_v4``, run
through R-50's continuous (non-restarting) engine, survive pre-registration
and this project's falsification/cost/holdout process?" B-19's own note
names the cheapest first check: "re-express the candidate as a one-time-
split adapter strategy using the existing ``multiasset.py`` (no periodic
rebalancing at all) rather than building periodic-rebalance-via-return-
splicing into ``multiasset.py``." This file IS that check.

Mechanism, one sentence
------------------------
Split capital 50/50 (and, as a neighbourhood check, 60/40 and 40/60)
between two fresh, UNMODIFIED ``KellyRegimeV4`` instances -- one on BTC,
one on ETH -- ONCE, at the start, via ``tradebot.multiasset.run_multi_backtest``,
and never touch the split again; if imperfectly-correlated bad regimes
(N~3) let a fixed-but-never-rebalanced blend still draw down less than
either single-asset book, that is real diversification benefit obtained
with zero rebalancing turnover, and it settles whether R-50's
periodically-rebalanced number needed the rebalancing at all.

Why this is a genuine, non-duplicate test of R-50's finding
-------------------------------------------------------------
Two academic camps disagree about where a "rebalancing premium" comes
from. Booth & Fama, "Diversification Returns and Asset Contributions"
(Financial Analysts Journal 48(3), 1992) and Willenbrock, "Diversification
Return, Portfolio Rebalancing, and the Commodity Return Puzzle" (Financial
Analysts Journal 67(4), 2011; arXiv:1109.1256) define the diversification
return as an incremental return that specifically requires periodic
rebalancing back to fixed target weights -- Willenbrock's own framing is
that rebalancing "forces the investor to sell assets that have appreciated
... and buy assets that have declined," in contrast to a buy-and-hold
portfolio whose winners simply grow to dominate it. Chambers & Zdanowicz,
"The Limitations of Diversification Return" (Journal of Portfolio
Management 40(4), 2014) counter that diversification return is not itself
a source of added expected value, and that whatever excess return
rebalancing produces comes from mean-reversion (selling relative winners,
buying relative losers) rather than from variance reduction or
diversification per se -- meaning the *risk*-side benefit (the drawdown
improvement R-50 actually measured, not the return improvement, which
sits inside this project's own noise floor per L-01/R-33) should be
present even in a portfolio that never rebalances at all, because it is
just the ordinary correlation/volatility-blending benefit of holding two
imperfectly-correlated books, full stop. This file is the direct empirical
test of that disagreement on THIS project's own asset pair and strategy:
if the NEVER-rebalanced static split recovers most of R-50's Sharpe/DD
improvement, that favours Chambers' reading (rebalancing was not the
active ingredient, blending was); if it recovers little or none of it,
that favours Booth & Fama / Willenbrock (the periodic sell-winners/buy-
losers act is doing real work).

Constraint attacked (docs/LEDGER.md standing diagnosis)
-----------------------------------------------------------
SIZE and N~3 -- identical framing to B-19's own row and to
``kelly_regime_dual_fixed.py`` (R-42/R-43, B-16): a second, imperfectly-
synchronized regime-cycle exposure, not a change to v4's own vote or
sizing formula (untouched, imported unchanged).

Not a duplicate of
--------------------
- R-50 / ``kelly_regime_covkelly_v3_continuous.py``: that file's
  "fixed5050_continuous" arm IS periodically rebalanced -- every segment
  (monthly or weekly) it rescales each leg's already-continuous equity
  curve back to exactly 50/50 of pooled capital (see that file's
  ``run_continuous_full``: ``dollars_b = pooled * w_b`` recomputed every
  segment boundary, ``w_b=0.5`` fixed but re-APPLIED each period). That is
  the periodic-rebalance-to-fixed-weights design the Booth & Fama /
  Willenbrock definition requires. This file's candidate splits capital
  ONCE and never touches it again -- true buy-and-hold of the initial
  split, weights left to drift with each leg's own performance -- which is
  the only lever this task is scoped to pull (multi-asset periodic
  rebalancing is explicitly out of scope for this branch; a separate,
  disjoint session is attacking a periodically-rebalanced novel variant).
- ``kelly_regime_dual_fixed.py`` (R-42/R-43, B-16): also a one-time,
  never-rebalanced split, and this file's SPLITS grid intentionally
  mirrors its weight choices (50/50, 60/40, 40/60 vs. its 50/50, 60/40,
  70/30, 40/60, 30/70, vol_weighted) so the two are comparable. The
  difference this file exists to make is the ONE the task specifies: this
  file composes the two legs through ``tradebot.multiasset.run_multi_backtest``
  / ``MultiAssetSpec`` -- the tested, promoted, general composition
  primitive (R-49) -- rather than that file's own ad hoc
  ``combine_equity`` helper (which R-49's own module docstring says
  ``multiasset.py`` was "generalized from"). It also runs a fresh,
  targeted falsification pair this file's predecessor did not: the
  mandatory exposure-artifact R^2 check framed explicitly against the
  standing "match risk before comparing anything" rule (R-33/R-46), and a
  0.40% taker fee-tier re-run, both pre-registered below BEFORE any result
  was read. B-16's own holdout read (bear-quartile drawdown-delta,
  ``vol_weighted``) answered a related but different question (a
  resampled-window quantity, not this file's point-comparison promotion
  rule) and is cited, not repeated.

Simulable here?
-----------------
Yes. Two independent, unmodified ``kelly_regime_v4`` backtests on
committed real OHLCV (BTC Bitstamp spot, ETH Coinbase spot), composed by
``tradebot.multiasset.run_multi_backtest`` -- no engine change, no new
data, no proxying.

Pre-registered falsification test (chosen before any result was read)
--------------------------------------------------------------------------
Two checks, BOTH must pass on the inner splits before the holdout is ever
touched:
  (1) exposure-artifact check: the candidate's aggregate exposure series
      must NOT be an R^2 > 0.95 flat rescale of BTC-solo ``kelly_regime_v4``'s
      own exposure (this project's standing "match risk before comparing
      anything" rule -- R-33/R-46/L-04 all died of exactly this).
  (2) must survive the realistic 0.40% Bitstamp taker tier: the 50/50
      candidate's advantage over BTC-solo v4 (Sharpe / drawdown) must not
      flip sign or vanish at 0.40% relative to what it shows at the
      project's usual 0.10% tier.
If EITHER fails, STOP -- do not read the 2023+ holdout. Report NEGATIVE.

Pre-registered promotion decision rule (written before the holdout is read)
--------------------------------------------------------------------------
Promote (PROMOTED-CANDIDATE) only if, on the 2023+ holdout, using the
FROZEN 50/50 configuration and no other:
  (a) beats ``buy_and_hold`` OOS after real costs (0.10% spot as the
      table convention, reported alongside 0.40%);
  (b) the improvement over BTC-solo ``kelly_regime_v4`` exceeds the
      +/-0.2 Sharpe noise floor (R-20) OR is a drawdown/tail improvement;
  (c) survives both falsification checks above;
  (d) the 50/50 -> 60/40 -> 40/60 neighbourhood is a plateau, not a
      knife-edge (no metric flips sign or changes by more than the noise
      floor between adjacent splits).
Anything else is NEGATIVE. If all of the inner-split/falsification gates
pass but the holdout is never reached because a prior branch already spent
it this round, the row is PARKED, not NEGATIVE -- "not tested" is not "a
negative result" (ROUTINE.md, "Running directions in parallel").

Data-window rule
-------------------
Inner-train = 2019-03-14 (ETH's real start on the committed Coinbase
file) -> 2020-12-31. Inner-validation = 2021-01-01 -> 2022-12-31 (the 2022
BTC/ETH joint bear -- this project's real constraint, per the task brief,
not ROUTINE.md's generic 2017 example, which predates ETH's committed
series). Holdout = 2023-01-01 onward, read AT MOST ONCE, only if every
prior gate above passes, using the frozen config named there.

Usage
-------
    python experiments/b19_dual_fixed_split.py sweep      # step 3: inner-train, spot, all splits
    python experiments/b19_dual_fixed_split.py select     # step 3: inner-validation, both markets, all splits
    python experiments/b19_dual_fixed_split.py baselines  # required baselines, both windows, both markets
    python experiments/b19_dual_fixed_split.py artifact   # pre-registered falsification test 1: R^2 exposure check
    python experiments/b19_dual_fixed_split.py feetier    # pre-registered falsification test 2: 0.40% taker
    python experiments/b19_dual_fixed_split.py causality  # no-lookahead sanity check on THIS file's composition
    python experiments/b19_dual_fixed_split.py holdout    # step 4: ONE read, frozen 50/50 config, only if gates pass
    python experiments/b19_dual_fixed_split.py all        # sweep+select+baselines+artifact+feetier+causality (no holdout)
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
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.multiasset import MultiAssetSpec, run_multi_backtest  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

INCUMBENT = "kelly_regime_v4"
HOLD = "buy_and_hold"

SPOT = MarketSpec.spot()
SPOT_04 = MarketSpec.spot(fee_rate=0.004)          # Bitstamp entry taker tier
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures5x", FUTURES))

# ---------------------------------------------------------------- data rule
# See module docstring. Grepped before every run below: no "2023"/"2024"/
# "2025"/"2026" date literal used to slice data appears anywhere in this
# file outside comments/docstrings and the dedicated `holdout()` function,
# which is the only place OOS_START is ever passed as a `start=`.
TRAIN = ("2019-03-14", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS_START = "2023-01-01"

BTC, BTC_LABEL = load_dataset(ROOT / "data", "spot")
ETH = load_coinbase_eth_spot(ROOT / "data")
if ETH is None:
    raise RuntimeError("data/ethusd_coinbase_spot_5m.csv.gz not found -- cannot run this experiment")
ETH_LABEL = "real (Coinbase spot)"

SPLITS: dict[str, tuple[float, float]] = {
    "50_50": (0.50, 0.50),
    "60_40_btc": (0.60, 0.40),
    "40_60_eth": (0.40, 0.60),
}
FROZEN_CANDIDATE = "50_50"  # the pre-registered holdout config -- see docstring

N_EVALUATED = 0            # every distinct DUAL-BOOK backtest configuration actually run
_SEEN: set[tuple] = set()  # dedup key: (split, window, market, fee_tag)
HOLDOUT_READS = 0          # increments only inside holdout()

_CACHE: dict = {}          # (split, start, end, market.name, fee) -> MultiBacktestResult


# --------------------------------------------------------------------- core


def run_dual(split_name: str, start: str, end: str, market: MarketSpec,
             total: float = 1_000.0, count: bool = True, fee_tag: str = "std") -> dict:
    """The candidate: ONE call to the tested composition primitive, no rebalancing.

    ``multiasset.run_multi_backtest`` runs each leg via ``run_period`` (the
    fair-warmup wrapper) ONCE over the whole ``[start, end]`` window at a
    fixed split decided before the call -- structurally incapable of
    rebalancing mid-window, which is exactly this branch's scope.
    """
    global N_EVALUATED
    key = (split_name, start, end, market.name, fee_tag)
    if key in _CACHE:
        return _CACHE[key]
    if count:
        dedup = (split_name, start, end, market.name, fee_tag)
        if dedup not in _SEEN:
            _SEEN.add(dedup)
            N_EVALUATED += 1

    w_btc, w_eth = SPLITS[split_name]
    specs = [
        MultiAssetSpec(label="BTC", strategy=KellyRegimeV4(), df=BTC, market=market),
        MultiAssetSpec(label="ETH", strategy=KellyRegimeV4(), df=ETH, market=market),
    ]
    res = run_multi_backtest(specs, [w_btc, w_eth], total, start=start, end=end)
    m = res.metrics
    out = {"split": split_name, "w_btc": w_btc, "w_eth": w_eth, "result": res,
           "final": m.final_balance, "sharpe": m.sharpe, "max_dd": m.max_drawdown_pct,
           "num_trades": m.num_trades}
    _CACHE[key] = out
    return out


def run_baseline_v4_btc(start: str, end: str, market: MarketSpec, total: float = 1_000.0) -> dict:
    res = run_period(get_strategy(INCUMBENT), BTC, start, end, market=market,
                     start_balance=total, data_label=BTC_LABEL)
    m = compute_metrics(res)
    return {"label": "kelly_regime_v4 BTC-only", "result": res,
            "final": m.final_balance, "sharpe": m.sharpe, "max_dd": m.max_drawdown_pct}


def run_baseline_hold_btc(start: str, end: str, market: MarketSpec, total: float = 1_000.0) -> dict:
    res = run_period(get_strategy(HOLD), BTC, start, end, market=market,
                     start_balance=total, data_label=BTC_LABEL)
    m = compute_metrics(res)
    return {"label": "buy_and_hold BTC-only", "result": res,
            "final": m.final_balance, "sharpe": m.sharpe, "max_dd": m.max_drawdown_pct}


def run_baseline_hold_dual(start: str, end: str, market: MarketSpec, total: float = 1_000.0) -> dict:
    specs = [
        MultiAssetSpec(label="BTC", strategy=get_strategy(HOLD), df=BTC, market=market),
        MultiAssetSpec(label="ETH", strategy=get_strategy(HOLD), df=ETH, market=market),
    ]
    res = run_multi_backtest(specs, [0.5, 0.5], total, start=start, end=end)
    m = res.metrics
    return {"label": "naive 50/50 buy&hold BTC+ETH", "result": res,
            "final": m.final_balance, "sharpe": m.sharpe, "max_dd": m.max_drawdown_pct}


def line(tag: str, d: dict) -> str:
    return (f"  {tag:40s} final=${d['final']:>10,.0f} sharpe={d['sharpe']:>6.2f} "
            f"maxDD={d['max_dd']:>5.1f}%")


# --------------------------------------------------------------------------- step 3


def sweep() -> pd.DataFrame:
    """Inner-train, spot, every split -- the required minimum grid."""
    rows = []
    t0 = time.time()
    for name in SPLITS:
        d = run_dual(name, *TRAIN, SPOT)
        rows.append({"split": name, "market": "spot", "window": "train", **_row(d)})
        print(f"[{N_EVALUATED}] " + line(f"{name} ({d['w_btc']:.2f}/{d['w_eth']:.2f})", d) +
              f"  [{time.time() - t0:.0f}s]")
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    return pd.DataFrame(rows)


def _row(d: dict) -> dict:
    return {"final": d["final"], "sharpe": d["sharpe"], "max_dd": d["max_dd"]}


def select() -> pd.DataFrame:
    """Inner-validation, both markets, every split."""
    rows = []
    for name in SPLITS:
        cells = []
        for mname, market in MARKETS:
            d = run_dual(name, *VALID, market)
            rows.append({"split": name, "market": mname, "window": "validation", **_row(d)})
            cells.append((mname, d))
        s = dict(cells)["spot"]
        f = dict(cells)["futures5x"]
        print(f"{name:12s} ({s['w_btc']:.2f}/{s['w_eth']:.2f})  "
              f"spot: ${s['final']:>9,.0f} sh{s['sharpe']:>6.2f} DD{s['max_dd']:>5.1f}%   "
              f"fut: ${f['final']:>9,.0f} sh{f['sharpe']:>6.2f} DD{f['max_dd']:>5.1f}%")
    print(f"\nconfigurations evaluated total: {N_EVALUATED}")
    return pd.DataFrame(rows)


def baselines() -> pd.DataFrame:
    """Required baselines: v4 BTC-solo, naive dual hold, BTC hold -- both windows, both markets."""
    rows = []
    for wname, (start, end) in (("train", TRAIN), ("validation", VALID)):
        for mname, market in MARKETS:
            a = run_baseline_v4_btc(start, end, market)
            b = run_baseline_hold_dual(start, end, market)
            c = run_baseline_hold_btc(start, end, market)
            for d in (a, b, c):
                rows.append({"baseline": d["label"], "window": wname, "market": mname, **_row(d)})
            print(f"\n-- {wname} / {mname} --")
            for d in (a, b, c):
                print(line(d["label"], d))
    print(f"\n(baselines are reference points, not counted in N_EVALUATED, "
          f"matching kelly_regime_dual_fixed.py's own convention)")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- falsification test 1: R^2


def artifact() -> pd.DataFrame:
    """Falsification test 1 (pre-registered): is the dual book just relabeled
    leverage on v4-BTC-alone? Standing rule (R-33/R-46): R^2 > 0.95 of the
    candidate's aggregate exposure against a mean-matched flat rescale of
    BTC-solo v4's own exposure means "this is not diversification, it's
    relabeled leverage" -- treat that as FAILING the check, not a footnote.

    Aggregate exposure = sum over legs of (leg's own target fraction x
    leg's own equity) / total portfolio equity -- the fraction of TOTAL
    capital the book is levered to, bar by bar, on the same scale as
    BTC-solo v4's own ``target`` column (which already is v4's exposure
    fraction of its own -- 100% of -- equity).
    """
    rows = []
    print("exposure-artifact check (50_50, inner-validation, mean-matched flat rescale of v4 BTC-only):")
    for mname, market in MARKETS:
        d = run_dual(FROZEN_CANDIDATE, *VALID, market, count=False)
        ctl = run_baseline_v4_btc(*VALID, market)

        res = d["result"]
        rb, re = res.leg_results[0], res.leg_results[1]
        idx = rb.df.index.union(re.df.index)
        tgt_b = rb.df["target"].reindex(idx).ffill().fillna(0.0)
        eq_b = rb.equity.reindex(idx).ffill().fillna(rb.start_balance)
        tgt_e = re.df["target"].reindex(idx).ffill().fillna(0.0)
        eq_e = re.equity.reindex(idx).ffill().fillna(re.start_balance)
        total_eq = eq_b + eq_e
        exposure = (tgt_b * eq_b + tgt_e * eq_e) / total_eq

        v4_tgt = ctl["result"].df["target"].reindex(idx).ffill().fillna(0.0)

        mask = np.isfinite(exposure.to_numpy()) & np.isfinite(v4_tgt.to_numpy())
        y = exposure.to_numpy()[mask]
        x_raw = v4_tgt.to_numpy()[mask]
        c = float(np.mean(y)) / float(np.mean(x_raw)) if np.mean(x_raw) != 0 else float("nan")
        x = c * x_raw

        ss_res = float(np.sum((y - x) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
        verdict = ("EXPOSURE-LEVEL ARTIFACT (R^2 > 0.95) -- FAILS falsification test"
                   if np.isfinite(r2) and r2 > 0.95 else "not a flat rescale -- PASSES this check")
        print(f"  {mname}: rescale c={c:.3f}  corr={corr:.4f}  R^2={r2:.4f}  {verdict}")
        rows.append({"market": mname, "rescale_c": c, "corr": corr, "r2": r2,
                     "artifact": bool(np.isfinite(r2) and r2 > 0.95)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- falsification test 2: fee tier


def feetier() -> pd.DataFrame:
    """Falsification test 2 (pre-registered): does the 50/50 candidate's
    advantage over v4-BTC-solo survive Bitstamp's real 0.40% taker tier,
    or is it a 0.10%-tier artifact of turnover this project has been
    burned by before?
    """
    rows = []
    print("0.40% taker fee-tier check (50_50 vs v4 BTC-solo, both windows, spot):\n")
    for wname, (start, end) in (("train", TRAIN), ("validation", VALID)):
        for fee_tag, market in (("0.10%", SPOT), ("0.40%", SPOT_04)):
            d = run_dual(FROZEN_CANDIDATE, start, end, market, fee_tag=fee_tag)
            ctl = run_baseline_v4_btc(start, end, market)
            hold = run_baseline_hold_btc(start, end, market)
            delta_sharpe = d["sharpe"] - ctl["sharpe"]
            delta_dd = d["max_dd"] - ctl["max_dd"]
            print(f"  {wname:11s} @ {fee_tag}:  dual50/50 final=${d['final']:>9,.0f} "
                  f"sh={d['sharpe']:>6.2f} DD={d['max_dd']:>5.1f}%   |  "
                  f"v4-solo final=${ctl['final']:>9,.0f} sh={ctl['sharpe']:>6.2f} "
                  f"DD={ctl['max_dd']:>5.1f}%   |  dSharpe={delta_sharpe:+.2f} dDD={delta_dd:+.1f}pp   |  "
                  f"hold final=${hold['final']:>9,.0f}")
            rows.append({"window": wname, "fee": fee_tag, "dual_final": d["final"],
                        "dual_sharpe": d["sharpe"], "dual_dd": d["max_dd"],
                        "v4_final": ctl["final"], "v4_sharpe": ctl["sharpe"], "v4_dd": ctl["max_dd"],
                        "hold_final": hold["final"], "delta_sharpe": delta_sharpe, "delta_dd": delta_dd})
    df = pd.DataFrame(rows)
    for wname in ("train", "validation"):
        sub = df[df["window"] == wname]
        d10 = sub[sub["fee"] == "0.10%"].iloc[0]
        d40 = sub[sub["fee"] == "0.40%"].iloc[0]
        flip = (np.sign(d10["delta_dd"]) != np.sign(d40["delta_dd"])) if d10["delta_dd"] != 0 else False
        print(f"\n  {wname}: dDD sign at 0.10%={'neg (better)' if d10['delta_dd']<0 else 'pos (worse)'} "
              f"vs at 0.40%={'neg (better)' if d40['delta_dd']<0 else 'pos (worse)'}  "
              f"{'FLIPPED -- FAILS' if flip else 'stable -- PASSES this check'}")
    return df


# ------------------------------------------------------------------------ causality


def causality() -> bool:
    """No-lookahead sanity check on THIS file's own composition code.

    ``multiasset.py`` is already causality-tested in isolation
    (``tests/test_multiasset.py``) and ``kelly_regime_v4`` is already
    causality-tested by CI (``test_causality_strict.py``). What is NEW here
    is the specific call this file makes -- two ``KellyRegimeV4`` legs on
    real BTC/ETH data through ``run_multi_backtest`` with a fixed split --
    so this re-verifies end to end rather than assuming the pieces compose
    safely. Standard two-opposite-tampers probe: multiply bars after a cut
    by K in one copy, divide by K in the other, and confirm the combined
    PORTFOLIO equity curve strictly before the cut is bit-identical.
    """
    cut_date = pd.Timestamp("2020-06-30", tz="UTC")  # inside TRAIN, nowhere near the holdout
    K = 137.0

    def tamper(df: pd.DataFrame, factor: float) -> pd.DataFrame:
        out = df.copy()
        mask = out.index > cut_date
        for col in ("open", "high", "low", "close"):
            out.loc[mask, col] = out.loc[mask, col] * factor
        out.loc[mask, "volume"] = out.loc[mask, "volume"] * (factor if factor > 1 else 1.0 / factor)
        return out

    def dual_equity(btc_df, eth_df, market=SPOT, total=1_000.0):
        specs = [
            MultiAssetSpec(label="BTC", strategy=KellyRegimeV4(), df=btc_df, market=market),
            MultiAssetSpec(label="ETH", strategy=KellyRegimeV4(), df=eth_df, market=market),
        ]
        res = run_multi_backtest(specs, [0.5, 0.5], total, start=TRAIN[0], end=TRAIN[1])
        return res.portfolio.equity

    base = dual_equity(BTC, ETH)
    up = dual_equity(tamper(BTC, K), tamper(ETH, K))
    down = dual_equity(tamper(BTC, 1.0 / K), tamper(ETH, 1.0 / K))

    pre = base.index[base.index <= cut_date]
    b = base.reindex(pre).to_numpy()
    u = up.reindex(pre).to_numpy()
    dn = down.reindex(pre).to_numpy()
    max_diff_up = float(np.nanmax(np.abs(b - u)))
    max_diff_down = float(np.nanmax(np.abs(b - dn)))
    ok = max_diff_up < 1e-6 and max_diff_down < 1e-6
    print(f"causality probe on this file's composition (cut={cut_date.date()}, K={K}):")
    print(f"  max|base - up-tampered| portfolio equity before cut:   {max_diff_up:.3e}")
    print(f"  max|base - down-tampered| portfolio equity before cut: {max_diff_down:.3e}")
    print(f"  {'PASS' if ok else 'FAIL'}: portfolio equity strictly before the cut is unchanged "
          "when only post-cut bars, on either leg, are tampered.")
    return ok


# ------------------------------------------------------------------------------- step 4: holdout


def holdout() -> pd.DataFrame:
    """ONE read of the 2023+ holdout, frozen 50/50 config, ONLY if every
    prior gate (inner-validation improvement, exposure-artifact check,
    0.40% fee-tier check, neighbourhood plateau) already passed. The
    decision rule is the one written in this file's module docstring,
    fixed BEFORE this function is ever called. Do not edit this function's
    thresholds after reading its output.
    """
    global HOLDOUT_READS
    rows = []
    print(f"=== HOLDOUT READ ({OOS_START} onward), frozen config: {FROZEN_CANDIDATE} ===\n")
    for mname, market in (("spot", SPOT), ("spot@0.40%", SPOT_04)):
        real_market = SPOT_04 if mname == "spot@0.40%" else market
        d = run_dual(FROZEN_CANDIDATE, OOS_START, None, real_market, fee_tag=mname)
        ctl = run_baseline_v4_btc(OOS_START, None, real_market)
        hold = run_baseline_hold_btc(OOS_START, None, real_market)
        HOLDOUT_READS += 1
        rows.append({"market": mname, "dual_final": d["final"], "dual_sharpe": d["sharpe"],
                    "dual_dd": d["max_dd"], "v4_final": ctl["final"], "v4_sharpe": ctl["sharpe"],
                    "v4_dd": ctl["max_dd"], "hold_final": hold["final"], "hold_sharpe": hold["sharpe"]})
        print(f"  {mname:12s}  dual50/50: ${d['final']:>10,.0f} sh={d['sharpe']:>6.2f} DD={d['max_dd']:>5.1f}%   "
              f"v4-solo: ${ctl['final']:>10,.0f} sh={ctl['sharpe']:>6.2f} DD={ctl['max_dd']:>5.1f}%   "
              f"hold: ${hold['final']:>10,.0f} sh={hold['sharpe']:>6.2f}")
    print(f"\nholdout reads this call: {HOLDOUT_READS}")
    return pd.DataFrame(rows)


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
    elif choice == "artifact":
        artifact()
    elif choice == "feetier":
        feetier()
    elif choice == "causality":
        causality()
    elif choice == "holdout":
        holdout()
    elif choice == "all":
        sweep()
        select()
        baselines()
        artifact()
        feetier()
        causality()
    else:
        print("usage: python experiments/b19_dual_fixed_split.py "
              "[sweep|select|baselines|artifact|feetier|causality|holdout|all]")
