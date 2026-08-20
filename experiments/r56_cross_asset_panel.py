"""R-56: does `kelly_regime_v4`'s one surviving property replicate across a
PANEL of instruments it has never seen, or is it a BTC-and-ETH coincidence?

Not registered: lives under ``experiments/`` per ROUTINE.md step 5. This is a
pure replication exercise. `kelly_regime_v4` is run byte-identical — no
parameter is touched, nothing is swept, nothing is selected — so there is no
tuning step and no in-sample/out-of-sample split to protect on these assets:
none of them was ever used to fit anything in this project.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **N≈3** (the standing diagnosis' effective-sample-size
constraint — "~3 regime events, not 1.01M bars").

The honest form of the attack, stated up front rather than claimed: adding
instruments does **not** add independent regime events. BTC, ETH and the
large alts share the same 2021 top and the same 2022 bear; their daily
returns correlate ~0.7-0.9. What a panel does add is independent *price
paths* and independent *microstructure* — a decade of trading behaviour that
`kelly_regime_v4`'s three anchors, its vol target and its 5%-deadband
rebalancing have never been exposed to. That is the axis this project has
never tested: every "does it replicate" check in this ledger is n=1 asset.

  - R-17 (ETH Bitfinex 2016-2019): one asset, and it shares the 2018 BTC
    bear with the main dataset — not an independent window either.
  - R-47 / B-08 (ETH Coinbase 2020-2026): one asset, genuinely independent
    window (ETH's own 2022 bear), and the finding was: the drawdown/tail
    property replicates, the return edge dies at the 0.40% tier.

So the project's single surviving positive claim — "the risk property
transfers, the return property does not" (R-17's line, L-01's lesson,
confirmed once on ETH by R-47) — currently rests on **two correlated
assets**, one of which is the asset the strategy was fitted on. Six more
instruments turn one anecdote into a countable rate, and the count is
pre-registered below. This is not a seventeenth variation on v4's own
vote-and-scale mechanism (R-34, R-37, R-38, R-40, R-41, R-42, R-43, R-44,
R-45, R-46, R-53, R-54, R-55 — every one NEGATIVE): nothing about the
strategy changes here at all.

**It also costs zero holdout consultations.** The BTC 2023+ file is never
read by this module; the panel assets never fitted anything. Same convention
as R-47/B-08, which logged +0 for reading ETH 2020-2026.

=====================================================================
PRE-REGISTRATION — written and committed BEFORE any panel result was read
=====================================================================

Nothing below was chosen after seeing a strategy number on any panel asset.
The asset-selection rule was executed against Coinbase's public product list
and three fixed 2020 probe days (a liquidity measurement, not a performance
one) before this file was written; the fetched panel was NOT run through any
backtest before this docstring was committed.

--- 1. Asset selection rule (mechanical, liquidity-based, no discretion) ---

  (a) Universe: Coinbase Exchange USD spot products, ``status == online``
      and trading enabled on 2026-08-20, excluding BTC-USD and ETH-USD
      (already in this project) and excluding stablecoins / wrapped /
      staked tokens (fixed base-ticker exclusion list in ``EXCLUDED_BASES``).
  (b) Alive in 2020: at least 750 of the 864 possible 5-minute candles
      across three fixed probe days — 2020-01-02, 2020-01-15, 2020-01-28.
      Ten products qualified.
  (c) Rank those ten by mean daily USD volume over the same three probe
      days, descending. The measured ranking was:
        BCH 24.7M, LTC 10.4M, ETC 9.3M, XRP 8.4M, DASH 8.3M, LINK 3.8M,
        XTZ 2.6M, OXT 2.1M, XLM 1.7M, ZRX 0.8M.
  (d) Continuity: after fetching 2020-01-01 -> 2026-08-20 at 5m, an asset
      qualifies only if its largest single gap is <= 7 days (no listing or
      suspension hole) AND bar coverage is >= 80% of the expected 5-minute
      grid.
  (e) **The panel is the six highest-ranked assets that pass (d).**

  AMENDMENT, 2026-08-20, recorded in full because amending a
  pre-registration is exactly the move this project's own routine warns
  about. Rule (d) originally read "coverage >= 95% AND largest gap <= 7
  days" as a single continuity gate. Run against the fetched panel it
  excluded FOUR of seven candidates — XRP (62.6%, the 905-day listing
  hole it was written for) but also ETC (91.5%), DASH (82.1%) and XTZ
  (91.0%), none of which has a gap over 6h40m. The gate was conflating two
  different things: a *listing hole* (what it was meant to catch) and
  *thin trading* (a 5-minute interval with no print produces no candle on
  Coinbase). It left n=3, at which the pre-registered 6/6 threshold cannot
  be reached at all, so the round could not have returned a verdict.
  The rule is therefore split into its two intended parts, and the
  liquidity floor is derived rather than picked: a coverage fraction f
  stretches v4's 20-day anchor to 20/f calendar days, and R-07 measured
  the anchor plateau as the 18-28 day region, so f >= 0.80 keeps the
  shortest anchor at <= 25 days, inside that validated plateau (ETC 21.9d,
  XTZ 22.0d, DASH 24.4d). **No backtest had been run on any panel asset
  when this amendment was written** — the only numbers read were bar
  counts, coverage fractions and gap lengths, all properties of the data
  files rather than of any strategy — and the panel it produces
  (BCH, LTC, ETC, DASH, LINK, XTZ) is the six the liquidity ranking named
  in the first place, minus XRP, plus the pre-authorized XTZ substitute.
  The decision rules in section 4 are untouched.

  Named in advance so the substitution cannot be read as a post-hoc choice:
  XRP-USD is expected to FAIL rule (d) — Coinbase suspended XRP-USD trading
  on 2021-01-19 (SEC complaint) and relisted it in July 2023, a ~2.5-year
  hole. If it fails, XTZ-USD (rank 7) takes its place. No other substitution
  is authorized; if a second asset fails (d), the panel is five and the
  binomial thresholds in section 4 are recomputed for n=5 (6/6 -> 5/5).

--- 2. Windows (fixed before running) ---

  FULL   2020-04-01 -> last bar in the file (2026-08-19/20).
         Starts three months after the data does so that v4 enters the
         measured period WARM: its warmup is 80 days x 288 bars, and
         ``run_period`` takes that prefix from bars before the window
         (R-22's warmup-prefix bias, which cost this project ~75% of a
         number once already).
  BEAR22 2022-05-01 -> 2022-11-30 (Terra/Luna through FTX). The identical
         window B-08 pre-registered for ETH, reused deliberately so the
         panel result is comparable to the one ETH result that exists.

--- 3. Arms and costs ---

  Arms:
    * ``kelly_regime_v4`` — frozen, zero parameters changed.
    * ``buy_and_hold`` — the project's benchmark (README: the bar every
      strategy must clear is buy_and_hold on spot).
    * ``ConstantExposureHold(c = v4's own mean clipped notional over the
      SAME window and market, rebalanced, deadband 0.10)`` — the R-33
      matched-risk arm on the **mean-notional** axis. It has no gate, no
      forecast and no anchors; it can only hold. This is the arm that
      matters, because the standing rule of this repo is "match risk
      before comparing anything": three of this project's findings died
      of comparing a half-invested strategy against a fully-invested
      benchmark (R-28/R-31, R-32, L-04/R-33).
    * Robustness only: the same arm matched on the **equal-realized-
      volatility** axis, c solved per asset per market by the proportional
      iteration R-33 used (tolerance 2%, cap at the market's leverage).
      Reported, not used for the primary decision, because a solver
      introduces a per-window fit the mean-notional axis does not need.

  Costs:
    * spot base    0.10% taker (this project's standard comparison cost)
    * spot real    0.40% taker (Bitstamp's entry tier — the falsification
                   test; R-13 measured this project's break-even at 0.104%)
    * futures 5x   0.05% taker, **funding NOT charged** — no altcoin
                   funding series exists in this repo and none is proxied
                   from price ("never proxy unavailable data out of
                   price"). Every futures number here is therefore an
                   upper bound, exactly as the README's standing warning
                   says, and is reported as secondary evidence only.

--- 4. Decision rules, frozen (default is REJECT) ---

  **D1 — PRIMARY (the risk claim, matched).** FULL window, spot @0.10%:
  count the assets where v4's max drawdown is strictly lower than the
  MEAN-NOTIONAL-MATCHED hold's.
      6 of 6  -> REPLICATES (exact one-sided binomial p = 0.0156 under a
                 50% null — the honest null here, since a coin flip is
                 what "no gating effect at matched exposure" predicts)
      5 of 6  -> SUGGESTIVE, explicitly NOT established (p = 0.109)
      <= 4/6  -> FAILS TO REPLICATE
  Reported alongside, per asset: the paired stationary-block-bootstrap
  interval on the drawdown difference (daily returns, 30-day mean block,
  2,000 resamples, seed 7, identical resamples for both arms) and how many
  of the six exclude zero.

  **D2 — the pre-registered FALSIFICATION test (ROUTINE step 2's menu:
  "does it survive a 0.40% taker").** FULL window, spot @0.40%: v4's final
  balance beats ``buy_and_hold``'s in >= 5 of 6 assets.
      PREDICTION, recorded now: **D2 FAILS.** R-13's fee study and R-47's
      ETH replication both say the return edge does not survive the real
      entry tier. If D2 passes, that is a genuine surprise and must be
      treated as a new hypothesis, not a promotion — one panel is not a
      promotion bar.

  **D3 — context, NOT evidence.** The same drawdown count against the
  UNMATCHED, fully-invested ``buy_and_hold``, both markets. Expected 6/6.
  It is reported to show the size of the exposure artifact next to D1, and
  is explicitly not counted as support for anything (R-33: 88-92% of this
  project's headline drawdown gap was the exposure level).

  **D4 — the bear window.** D1 and D3 recomputed on BEAR22. Descriptive:
  n=1 window per asset, so no significance is claimed from it.

--- 5. What would make this fail (named before any code ran) ---

  v4's drawdown advantage over a **matched-exposure** hold is absent or
  sign-unstable outside BTC/ETH (D1 <= 4/6). That outcome would mean the
  one positive claim this project still leans on is a two-asset
  coincidence on two correlated assets, one of which is the asset the
  strategy was fitted on — and it would be the most important negative
  result in the ledger since R-33.

  The mirror-image failure is also possible and is not a win: D1 passing
  while D2 fails reproduces R-47's ETH finding on six more instruments,
  which strengthens the risk claim and leaves the strategy still
  unprofitable at real costs.

--- 6. Configurations evaluated ---

  Counted by ``CONFIG_COUNT`` and printed at the end. There is no sweep:
  the only "search" in this round is the volatility-matching solver, whose
  iterations are counted honestly even though they fit a benchmark arm
  rather than the strategy.

Usage::

    python experiments/r56_cross_asset_panel.py panel      # selection + integrity
    python experiments/r56_cross_asset_panel.py causality  # tamper probe, per asset
    python experiments/r56_cross_asset_panel.py run        # the frozen matrix
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.matched_hold import ConstantExposureHold, mean_notional  # noqa: E402
from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.data import load_ohlcv_csv  # noqa: E402
from tradebot.engine import validate_ohlcv  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns,
    max_drawdown_from_returns,
    paired_bootstrap,
    total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "reports" / "cross_asset_panel"

# Rule 1(c)'s measured ranking, frozen. The panel is the first six of these
# that pass the continuity rule 1(d); XRP is expected to fail it.
RANKED_CANDIDATES = [
    ("BCH", "bchusd_coinbase_spot_5m.csv.gz", 24_701_781),
    ("LTC", "ltcusd_coinbase_spot_5m.csv.gz", 10_379_663),
    ("ETC", "etcusd_coinbase_spot_5m.csv.gz", 9_319_843),
    ("XRP", "xrpusd_coinbase_spot_5m.csv.gz", 8_425_496),
    ("DASH", "dashusd_coinbase_spot_5m.csv.gz", 8_288_839),
    ("LINK", "linkusd_coinbase_spot_5m.csv.gz", 3_838_850),
    ("XTZ", "xtzusd_coinbase_spot_5m.csv.gz", 2_605_351),
]

FETCH_START = pd.Timestamp("2020-01-01", tz="UTC")
FETCH_END = pd.Timestamp("2026-08-20", tz="UTC")
MIN_COVERAGE = 0.80  # see the rule-1(d) amendment in the docstring
MAX_GAP = pd.Timedelta(days=7)
PANEL_SIZE = 6

FULL = ("2020-04-01", None)
BEAR22 = ("2022-05-01", "2022-11-30")

SPOT_BASE = MarketSpec.spot()                  # 0.10% taker
SPOT_REAL = MarketSpec.spot(fee_rate=0.004)    # 0.40% Bitstamp entry tier
FUT_BASE = MarketSpec.futures(leverage=5.0)    # 0.05% taker, funding NOT charged

INCUMBENT = "kelly_regime_v4"
BOOT_KW = dict(mean_block=30.0, n_boot=2_000, seed=7)

CONFIG_COUNT = 0


# ------------------------------------------------------------------ helpers


def measure(strategy, df, start, end, market):
    """One backtest. Every call is counted — there is no free evaluation."""
    global CONFIG_COUNT
    CONFIG_COUNT += 1
    result = run_period(strategy, df, start, end, market=market, start_balance=1_000.0)
    return result, compute_metrics(result)


def realized_vol(equity: pd.Series) -> float:
    """Annualized volatility of the daily equity curve (the R-31 convention)."""
    rets = daily_returns(equity).to_numpy(dtype=float)
    if len(rets) < 2:
        return float("nan")
    return float(np.std(rets, ddof=1) * math.sqrt(365.25))


def solve_c(target_vol, df, start, end, market, tol=0.02, max_iter=8):
    """Constant exposure whose realized vol matches ``target_vol`` (R-33's
    proportional iteration). Iterations are counted as configurations."""
    c_max = float(market.leverage)
    c = min(0.5, c_max)
    res, _ = measure(ConstantExposureHold(c), df, start, end, market)
    vol = realized_vol(res.equity)
    for _ in range(max_iter):
        if not np.isfinite(vol) or vol <= 0:
            return float("nan"), vol
        if abs(vol - target_vol) <= tol * target_vol:
            return c, vol
        c = float(np.clip(c * (target_vol / vol), 1e-3, c_max))
        res, _ = measure(ConstantExposureHold(c), df, start, end, market)
        vol = realized_vol(res.equity)
        if c >= c_max and vol < target_vol:  # cap binds, no match exists
            return c, vol
    return c, vol


def binomial_tail(k: int, n: int, p: float = 0.5) -> float:
    """One-sided exact binomial P(X >= k) — the honest null for a sign count."""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


# ------------------------------------------------------- panel + integrity


@dataclass
class Asset:
    ticker: str
    df: pd.DataFrame
    coverage: float
    max_gap: pd.Timedelta
    qualifies: bool


def load_candidates() -> list[Asset]:
    out = []
    expected = int((FETCH_END - FETCH_START) / pd.Timedelta(minutes=5))
    for ticker, fname, _vol in RANKED_CANDIDATES:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  {ticker:5s} MISSING ({fname})")
            continue
        df = load_ohlcv_csv(path)
        validate_ohlcv(df)
        gaps = df.index.to_series().diff().dropna()
        max_gap = gaps.max() if len(gaps) else pd.Timedelta(0)
        coverage = len(df) / expected
        ok = coverage >= MIN_COVERAGE and max_gap <= MAX_GAP
        out.append(Asset(ticker, df, coverage, max_gap, ok))
    return out


def select_panel(candidates: list[Asset]) -> list[Asset]:
    qualifying = [a for a in candidates if a.qualifies]
    return qualifying[:PANEL_SIZE]


def cmd_panel() -> list[Asset]:
    print("=" * 100)
    print("PANEL SELECTION — rule 1(d) as amended: largest gap <= 7 days "
          "(no listing hole) and coverage >= 80% of the 5m grid")
    print("=" * 100)
    candidates = load_candidates()
    for a in candidates:
        print(f"  {a.ticker:5s} bars={len(a.df):>8,d} "
              f"{a.df.index[0]:%Y-%m-%d}->{a.df.index[-1]:%Y-%m-%d} "
              f"coverage={a.coverage:6.1%} max_gap={str(a.max_gap):>20s} "
              f"20d-anchor spans {20 / a.coverage:5.1f}d "
              f"{'QUALIFIES' if a.qualifies else 'EXCLUDED'}")
    panel = select_panel(candidates)
    print(f"\nPanel ({len(panel)}): {', '.join(a.ticker for a in panel)}")
    if len(panel) < PANEL_SIZE:
        print(f"NOTE: fewer than {PANEL_SIZE} assets qualified — binomial "
              f"thresholds recomputed for n={len(panel)} per the pre-registration.")
    return panel


def cmd_causality(panel: list[Asset]) -> None:
    """The tests/test_causality_strict.py tamper methodology, run against each
    new loading path (that module hard-codes the BTC spot loader, so these
    files are not covered by it automatically)."""
    print("=" * 100)
    print(f"CAUSALITY TAMPER PROBE — {INCUMBENT} on each panel asset")
    print("=" * 100)
    market = MarketSpec.futures(leverage=5.0)
    for a in panel:
        tail = a.df.iloc[-60_000:].copy()
        cut = len(tail) - 5_000
        bars = [cut - k for k in (1, 2, 3, 5, 10, 20)]
        up, down = tail.copy(), tail.copy()
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

        def decisions(frame):
            s = get_strategy(INCUMBENT)
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=market, start_balance=10_000.0)
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        ok = all(x == y for x, y in zip(decisions(up), decisions(down)))
        print(f"  {a.ticker:5s} decisions identical under opposite post-cut "
              f"tampers: {'PASS' if ok else 'FAIL'}")


# ----------------------------------------------------------------- the run


def cell(a: Asset, window, market, label: str, rows: list) -> dict:
    """One asset x window x market cell: v4, hold, matched hold, intervals."""
    start, end = window
    v4_res, v4 = measure(get_strategy(INCUMBENT), a.df, start, end, market)
    hold_res, hold = measure(get_strategy("buy_and_hold"), a.df, start, end, market)

    c_mean = mean_notional(v4_res)
    mh_res, mh = measure(ConstantExposureHold(c_mean), a.df, start, end, market)

    v4_ret = daily_returns(v4_res.equity).to_numpy(dtype=float)
    mh_ret = daily_returns(mh_res.equity).to_numpy(dtype=float)
    hold_ret = daily_returns(hold_res.equity).to_numpy(dtype=float)
    n = min(len(v4_ret), len(mh_ret), len(hold_ret))
    dd_matched = paired_bootstrap(v4_ret[:n], mh_ret[:n],
                                  max_drawdown_from_returns, **BOOT_KW)
    growth_matched = paired_bootstrap(v4_ret[:n], mh_ret[:n],
                                      total_log_return, **BOOT_KW)

    row = {
        "asset": a.ticker, "window": label, "market": market.name,
        "fee": market.fee_rate,
        "v4_final": v4.final_balance, "v4_dd": v4.max_drawdown_pct,
        "v4_sharpe": v4.sharpe, "v4_trades": v4.num_trades,
        "v4_liq": v4.liquidated,
        "hold_final": hold.final_balance, "hold_dd": hold.max_drawdown_pct,
        "hold_sharpe": hold.sharpe, "hold_liq": hold.liquidated,
        "c_mean_notional": c_mean,
        "mh_final": mh.final_balance, "mh_dd": mh.max_drawdown_pct,
        "mh_sharpe": mh.sharpe,
        "v4_vol": realized_vol(v4_res.equity), "mh_vol": realized_vol(mh_res.equity),
        "dd_matched_diff": dd_matched.diff.point,
        "dd_matched_lo": dd_matched.diff.lo, "dd_matched_hi": dd_matched.diff.hi,
        "growth_matched_diff": growth_matched.diff.point,
        "growth_matched_lo": growth_matched.diff.lo,
        "growth_matched_hi": growth_matched.diff.hi,
    }
    rows.append(row)
    print(f"  {a.ticker:5s} {label:7s} {market.name:11s} fee={market.fee_rate:.2%}  "
          f"v4 ${v4.final_balance:>10,.0f} DD {v4.max_drawdown_pct:5.1f}% | "
          f"hold ${hold.final_balance:>10,.0f} DD {hold.max_drawdown_pct:5.1f}% | "
          f"matched(c={c_mean:.2f}) ${mh.final_balance:>10,.0f} "
          f"DD {mh.max_drawdown_pct:5.1f}% | "
          f"dDD_matched {dd_matched.diff.point:+6.1f}pp "
          f"[{dd_matched.diff.lo:+6.1f},{dd_matched.diff.hi:+6.1f}]")
    return row


def cmd_run(panel: list[Asset]) -> None:
    rows: list[dict] = []

    print("=" * 100)
    print("FULL WINDOW 2020-04-01 -> end — spot @0.10% (D1 primary, D3 context)")
    print("=" * 100)
    for a in panel:
        cell(a, FULL, SPOT_BASE, "FULL", rows)

    print("\n" + "=" * 100)
    print("FULL WINDOW — futures 5x @0.05%, funding NOT charged (upper bound)")
    print("=" * 100)
    for a in panel:
        cell(a, FULL, FUT_BASE, "FULL", rows)

    print("\n" + "=" * 100)
    print("FULL WINDOW — spot @0.40% Bitstamp entry tier (D2 falsification test)")
    print("=" * 100)
    for a in panel:
        cell(a, FULL, SPOT_REAL, "FULL", rows)

    print("\n" + "=" * 100)
    print("BEAR22 2022-05-01..2022-11-30 — spot @0.10% and futures 5x (D4)")
    print("=" * 100)
    for a in panel:
        cell(a, BEAR22, SPOT_BASE, "BEAR22", rows)
    for a in panel:
        cell(a, BEAR22, FUT_BASE, "BEAR22", rows)

    print("\n" + "=" * 100)
    print("ROBUSTNESS — equal-realized-volatility matching, FULL window, spot")
    print("=" * 100)
    vol_rows = []
    for a in panel:
        v4_res, v4 = measure(get_strategy(INCUMBENT), a.df, *FULL, SPOT_BASE)
        target = realized_vol(v4_res.equity)
        c, achieved = solve_c(target, a.df, *FULL, SPOT_BASE)
        vh_res, vh = measure(ConstantExposureHold(c), a.df, *FULL, SPOT_BASE)
        resid = abs(achieved - target) / target if target else float("nan")
        valid = resid <= 0.05
        vol_rows.append({"asset": a.ticker, "c_vol": c, "target_vol": target,
                         "achieved_vol": achieved, "resid": resid, "valid": valid,
                         "v4_dd": v4.max_drawdown_pct, "vh_dd": vh.max_drawdown_pct,
                         "v4_final": v4.final_balance, "vh_final": vh.final_balance})
        print(f"  {a.ticker:5s} c={c:5.3f} target_vol={target:5.3f} "
              f"achieved={achieved:5.3f} resid={resid:5.1%} "
              f"{'VALID' if valid else 'MATCH FAILED — cell void'} | "
              f"v4 DD {v4.max_drawdown_pct:5.1f}% vs vol-matched hold DD "
              f"{vh.max_drawdown_pct:5.1f}% | v4 ${v4.final_balance:>10,.0f} vs "
              f"${vh.final_balance:>10,.0f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "cells.csv", index=False)
    pd.DataFrame(vol_rows).to_csv(OUT_DIR / "vol_matched.csv", index=False)

    verdicts(rows, vol_rows, len(panel))
    print(f"\nTotal backtest configurations evaluated: {CONFIG_COUNT}")
    print("Holdout consultations added by this round: 0 "
          "(the BTC 2023+ file is never read here; no panel asset fitted anything)")


def verdicts(rows: list[dict], vol_rows: list[dict], n: int) -> None:
    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("PRE-REGISTERED DECISION RULES")
    print("=" * 100)

    d1 = df[(df.window == "FULL") & (df.market == "spot") & (df.fee == 0.001)]
    k1 = int((d1.v4_dd < d1.mh_dd).sum())
    excl = int(((d1.dd_matched_lo > 0) | (d1.dd_matched_hi < 0)).sum())
    better_excl = int((d1.dd_matched_hi < 0).sum())
    p1 = binomial_tail(k1, n)
    if k1 == n:
        v1 = "REPLICATES"
    elif k1 == n - 1:
        v1 = "SUGGESTIVE (not established)"
    else:
        v1 = "FAILS TO REPLICATE"
    print(f"D1 (primary, matched-exposure drawdown, spot @0.10%, FULL): "
          f"{k1}/{n} assets, exact binomial p={p1:.4f} -> {v1}")
    print(f"    paired bootstrap: {excl}/{n} intervals exclude zero "
          f"({better_excl}/{n} of them in v4's favour)")

    d2 = df[(df.window == "FULL") & (df.market == "spot") & (df.fee == 0.004)]
    k2 = int((d2.v4_final > d2.hold_final).sum())
    v2 = "SURVIVES" if k2 >= n - 1 else "FAILS (as predicted)"
    print(f"D2 (falsification test, 0.40% taker, beats buy_and_hold): "
          f"{k2}/{n} -> {v2}")

    for mkt, fee, tag in (("spot", 0.001, "spot @0.10%"),
                          ("futures_5x", 0.0005, "futures 5x")):
        d3 = df[(df.window == "FULL") & (df.market == mkt) & (df.fee == fee)]
        k3 = int((d3.v4_dd < d3.hold_dd).sum())
        print(f"D3 (context only — UNMATCHED drawdown vs fully-invested hold, "
              f"{tag}): {k3}/{n}")

    for mkt, fee, tag in (("spot", 0.001, "spot @0.10%"),
                          ("futures_5x", 0.0005, "futures 5x")):
        d4 = df[(df.window == "BEAR22") & (df.market == mkt) & (df.fee == fee)]
        km = int((d4.v4_dd < d4.mh_dd).sum())
        ku = int((d4.v4_dd < d4.hold_dd).sum())
        print(f"D4 (BEAR22, {tag}): matched {km}/{n}, unmatched {ku}/{n} "
              f"(descriptive — one window per asset)")

    vdf = pd.DataFrame(vol_rows)
    valid = vdf[vdf.valid]
    kv = int((valid.v4_dd < valid.vh_dd).sum())
    print(f"Robustness (equal-realized-vol matching, spot, FULL): "
          f"{kv}/{len(valid)} of the {len(valid)} valid cells favour v4 "
          f"({len(vdf) - len(valid)} cells void — match residual > 5%)")


def cmd_control(panel: list[Asset]) -> None:
    """POST-HOC control, run AFTER D1 returned 0/6 and clearly labelled as such.

    It cannot change D1's verdict — that is already recorded — and it is not a
    second decision rule. It answers the one question D1's result raises and
    cannot itself settle: is the matched-exposure drawdown advantage *absent
    outside BTC/ETH*, or was it absent *in this period* everywhere? So the
    identical comparison is run on BTC (the fitted asset) and ETH (R-47's
    asset) over a window every panel asset shares, truncated at 2022-12-31 so
    **no 2023+ BTC bar is read and the holdout counter stays at +0**.
    """
    from tradebot.data import load_coinbase_spot, load_dataset

    print("=" * 100)
    print("POST-HOC CONTROL (not a decision rule) — same comparison on BTC and "
          "ETH, 2020-04-01..2022-12-31, spot @0.10%")
    print("=" * 100)
    window = ("2020-04-01", "2022-12-31")
    btc, _ = load_dataset(DATA_DIR, "spot")
    frames = [("BTC", btc), ("ETH", load_coinbase_spot(DATA_DIR, "ETH"))]
    frames += [(a.ticker, a.df) for a in panel]
    rows = []
    for ticker, df in frames:
        if df is None:
            continue
        v4_res, v4 = measure(get_strategy(INCUMBENT), df, *window, SPOT_BASE)
        c = mean_notional(v4_res)
        mh_res, mh = measure(ConstantExposureHold(c), df, *window, SPOT_BASE)
        v4_ret = daily_returns(v4_res.equity).to_numpy(dtype=float)
        mh_ret = daily_returns(mh_res.equity).to_numpy(dtype=float)
        n = min(len(v4_ret), len(mh_ret))
        dd = paired_bootstrap(v4_ret[:n], mh_ret[:n],
                              max_drawdown_from_returns, **BOOT_KW)
        rows.append({"asset": ticker, "c_mean_notional": c,
                     "v4_dd": v4.max_drawdown_pct, "mh_dd": mh.max_drawdown_pct,
                     "v4_final": v4.final_balance, "mh_final": mh.final_balance,
                     "dd_diff": dd.diff.point, "dd_lo": dd.diff.lo,
                     "dd_hi": dd.diff.hi})
        print(f"  {ticker:5s} c={c:4.2f} v4 DD {v4.max_drawdown_pct:5.1f}% vs "
              f"matched hold DD {mh.max_drawdown_pct:5.1f}% "
              f"(dDD {dd.diff.point:+6.1f}pp [{dd.diff.lo:+6.1f},{dd.diff.hi:+6.1f}], "
              f"negative = v4 better) | v4 ${v4.final_balance:>9,.0f} vs "
              f"${mh.final_balance:>9,.0f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "control_pre2023.csv", index=False)
    better = [r["asset"] for r in rows if r["v4_dd"] < r["mh_dd"]]
    print(f"\n  v4's drawdown lower than the matched hold's in "
          f"{len(better)}/{len(rows)} cells: {', '.join(better) if better else 'none'}")
    print(f"  Configurations added by this control: counted in the total below. "
          f"Holdout consultations added: 0 (window ends 2022-12-31).")


def cmd_chart() -> None:
    """The picture D1 and D3 disagree about, drawn in the project's forest style.

    Left: each panel asset's drawdown difference against the MATCHED hold on
    the FULL window, with its paired-bootstrap interval. Right: the same
    difference against the fully-invested ``buy_and_hold`` — the comparison
    the README table makes. The two panels are the whole result.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tradebot.report import (BASELINE, CRITICAL, GOOD, GRID, INK, MUTED,
                                 PAGE, SURFACE)

    cells = pd.read_csv(OUT_DIR / "cells.csv")
    ctrl = pd.read_csv(OUT_DIR / "control_pre2023.csv")
    full = cells[(cells.window == "FULL") & (cells.market == "spot")
                 & (cells.fee == 0.001)].set_index("asset")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(PAGE)

    # left: matched, with intervals (the pre-registered comparison)
    ax = axes[0]
    order = list(full.index)
    y = np.arange(len(order))
    point = full.loc[order, "dd_matched_diff"].to_numpy()
    lo = full.loc[order, "dd_matched_lo"].to_numpy()
    hi = full.loc[order, "dd_matched_hi"].to_numpy()
    excl = (lo > 0) | (hi < 0)
    colors = [GOOD if p < 0 and e else CRITICAL if p > 0 and e else MUTED
              for p, e in zip(point, excl)]
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(True, axis="x", color=GRID, linewidth=1.0)
    ax.hlines(y, lo, hi, color=colors, linewidth=2.4, alpha=0.55)
    ax.scatter(point, y, s=40, color=colors, zorder=3,
               edgecolors=SURFACE, linewidths=1.2)
    ax.axvline(0.0, color=INK, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_title("vs a hold carrying v4's OWN mean exposure  ·  "
                 "left of zero = v4 draws down less",
                 color=INK, fontsize=10, loc="left")

    # right: the same six against the fully-invested benchmark
    ax = axes[1]
    gap = (full.loc[order, "v4_dd"] - full.loc[order, "hold_dd"]).to_numpy()
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(True, axis="x", color=GRID, linewidth=1.0)
    # deliberately NOT the "good" colour: this panel is the artifact,
    # and colouring it green would be the picture arguing for itself.
    ax.barh(y, gap, color=MUTED, alpha=0.65, height=0.55)
    ax.axvline(0.0, color=INK, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_title("vs the fully-invested buy_and_hold  ·  the comparison the "
                 "README table makes  ·  every bar is exposure, not gating",
                 color=INK, fontsize=10, loc="left")

    ctrl_line = ", ".join(
        f"{r.asset} {r.dd_diff:+.1f}pp" for r in ctrl.itertuples()
        if r.asset in ("BTC", "ETH"))
    fig.suptitle(
        "R-56 · kelly_regime_v4's drawdown advantage on six instruments it "
        "was never fitted on\n"
        "spot, 0.10% taker, 2020-04-01 -> 2026-08-20 · Δ max drawdown (pp) · "
        f"same comparison 2020-04..2022-12 on the fitted assets: {ctrl_line}",
        color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = OUT_DIR / "panel_drawdown.png"
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)
    print(f"chart: {path}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "panel"
    panel = cmd_panel()
    if not panel:
        raise SystemExit("no qualifying assets — fetch the panel first "
                         "(scripts/fetch_coinbase_panel.py)")
    if cmd == "panel":
        return
    print()
    if cmd == "causality":
        cmd_causality(panel)
        return
    if cmd == "run":
        cmd_causality(panel)
        print()
        cmd_run(panel)
        return
    if cmd == "chart":
        cmd_chart()
        return
    if cmd == "control":
        cmd_control(panel)
        print(f"\nTotal backtest configurations evaluated by this command: "
              f"{CONFIG_COUNT}")
        return
    raise SystemExit(f"unknown command {cmd!r} "
                     f"(panel | causality | run | control | chart)")


if __name__ == "__main__":
    main()
