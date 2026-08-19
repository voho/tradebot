#!/usr/bin/env python
"""Conservative branch, this session's SIZE-axis round (see docs/LEDGER.md R-33/R-36).

R-36 confirmed that `kelly_regime_v4` out-returns a passive hold matched to
its own realized volatility, and that the edge survives outside the
2017-2020 bull (post-2021-start windows: win-rate 68.2%/81.8%, median
+5.0pp/+7.4pp spot/futures) though it thins about 10x relative to the
pooled, bull-heavy number. v4's sizing knobs (`target_vol`, `max_leverage`,
`anchor_span_days`) were never retuned against that evidence - they date
to L-01/R-07, an anchor-*horizon* sweep on a different axis entirely.

This file changes NO mechanism: no new signal, no new detection logic, no
new sizing formula. It only asks whether a different point on the SAME
`min(target_vol / vol, max_leverage)` frontier `kelly_regime_v4` already
implements captures more of the confirmed post-2021 edge than the shipped
defaults (`target_vol=0.55, max_leverage=2.0`).

Not a duplicate of:
  - R-06/R-07: those swept `horizons`, the vote anchors that decide
    *whether* the regime is bullish. This sweeps `target_vol`/
    `max_leverage`/`anchor_span_days`, which decide *how much* to hold
    once the vote is bullish - a disjoint set of constructor arguments.
  - L-02/`kelly_regime_v2`: that changed `vote_gamma`, a convex response
    curve on the DISCRETE VOTE. Nothing here touches `vote_gamma` or the
    vote at all.
  - R-34/R-35: those added new SIZE *inputs* (a Bayesian posterior, a
    funding decile) multiplying the existing sizer. This adds no new
    input signal of any kind - it only re-solves the constant knobs of
    the sizer that is already registered.

Pre-registered failure modes (named before any sweep ran):
  (a) any improvement over the shipped defaults sits inside the project's
      +/-0.2 Sharpe noise floor (R-20) - i.e. it is noise;
  (b) the "improvement" is a plateau-free spike, not a region;
  (c) the winner just carries more raw exposure than the defaults at the
      same period - the L-04/R-28/R-31/R-32/R-33 exposure-level artifact,
      this project's most repeated failure mode. Checked explicitly below
      by comparing realized volatility and mean notional fraction against
      the shipped defaults on the SAME inner-validation period.

Usage::

    python experiments/kelly_regime_v6_retune.py sweep       # step 3: inner-train + inner-validation grid
    python experiments/kelly_regime_v6_retune.py select      # selection + plateau neighbourhood
    python experiments/kelly_regime_v6_retune.py causality   # by-hand lookahead probe
    python experiments/kelly_regime_v6_retune.py eth         # pre-registered falsification
    python experiments/kelly_regime_v6_retune.py all         # everything, in order
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")   # inner-train: fit, sweep, iterate freely
VALID = ("2021-01-01", "2022-12-31")   # inner-validation: select between variants
# OOS_START = "2023-01-01"  -- NEVER read in this file, by construction.

OUT = ROOT / "reports" / "kelly_regime_v6_retune"

INCUMBENT_KW = dict(target_vol=0.55, max_leverage=2.0)  # shipped v4 defaults

BARS_PER_YEAR = 365.25 * 24 * 12  # 5m bars, matches tradebot.metrics

# ---- the sweep grid ---------------------------------------------------
# Primary axis: target_vol x max_leverage around the shipped 0.55 / 2.0.
TARGET_VOL_GRID = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90)
MAX_LEV_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
# Secondary axis, applied only around the inner-validation winner: the
# hysteresis anchor span that decides how quickly the vol-targeting
# switches from "steady" to "full inverse-vol" sizing. v3/v4 already
# expose it; R-06/R-07 never touched it (that sweep was over `horizons`,
# the regime vote anchors, a different constructor argument entirely).
ANCHOR_SPAN_GRID = (90, 180, 270, 365)  # days; 180 is the v3/v4 default

_SEEN: set[tuple] = set()  # distinct configurations evaluated, for deflated Sharpe


def _config_key(**kw) -> tuple:
    return tuple(sorted(kw.items()))


def cand(**kw) -> KellyRegimeV4:
    """One kelly_regime_v4 instance at a given point on its own sizing frontier.

    No subclass, no new code path: this is exactly the registered
    strategy's __init__ with different keyword arguments, so its
    prepare()/on_bar() are byte-identical to the promoted kelly_regime_v4.
    """
    _SEEN.add(_config_key(**kw))
    return KellyRegimeV4(**kw)


def realized_vol(equity) -> float:
    """Annualized std of per-bar equity returns (matches experiments/matched_risk.py)."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 3:
        return 0.0
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
    sd = np.std(rets, ddof=1)
    return float(sd * np.sqrt(BARS_PER_YEAR)) if np.isfinite(sd) else 0.0


def mean_notional(result) -> float:
    """Mean |target| fraction actually allowed by the market (matches matched_hold.py)."""
    if "target" not in result.df:
        return float("nan")
    tgt = np.abs(result.df["target"].to_numpy(dtype=float))
    return float(np.mean(np.clip(tgt, 0.0, result.market.leverage)))


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0):
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    return m, realized_vol(result.equity), mean_notional(result), result


def row(tag, m, vol, notional, result) -> None:
    print(f"  {tag:26s} final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"vol={vol:5.3f} notional={notional:5.3f} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>6.2f} trades={m.num_trades:>4d}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# ------------------------------------------------------------------- sweep


def _grid_rows(start, end, split) -> list[dict]:
    rows = []
    for tv in TARGET_VOL_GRID:
        for ml in MAX_LEV_GRID:
            strat = cand(target_vol=tv, max_leverage=ml)
            for mname, market in MARKETS:
                m, vol, notional, res = measure(strat, start, end, market=market)
                rows.append({"split": split, "market": mname,
                             "target_vol": tv, "max_leverage": ml,
                             "anchor_span_days": 180,
                             "final": m.final_balance, "profit_pct": m.profit_pct,
                             "vol": vol, "mean_notional": notional,
                             "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                             "trades": m.num_trades, "liquidated": m.liquidated})
    return rows


def sweep() -> pd.DataFrame:
    """Step 3: the target_vol x max_leverage grid, inner-train and inner-validation."""
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    for (start, end), split in ((TRAIN, "inner-train"), (VALID, "inner-validation")):
        print(f"\n--- {split} ({start} -> {end}) ---")
        rows += _grid_rows(start, end, split)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sweep.csv", index=False)
    print(f"\nwrote {OUT / 'sweep.csv'}  ({time.time() - t0:.0f}s)")
    print(f"distinct configurations evaluated so far: {len(_SEEN)}")
    return df


# ------------------------------------------------------------------ select


def select() -> None:
    """Step 3->4: rank on inner-validation only, show the plateau, check exposure."""
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "sweep.csv"
    if csv.exists():
        df = pd.read_csv(csv)
    else:
        df = sweep()

    valid_spot = df[(df.split == "inner-validation") & (df.market == "spot")].copy()
    valid_fut = df[(df.split == "inner-validation") & (df.market == "futures")].copy()

    print("\n=== inner-validation ranking, primary metric = spot Sharpe ===")
    ranked = valid_spot.sort_values("sharpe", ascending=False)
    print(ranked[["target_vol", "max_leverage", "sharpe", "profit_pct", "max_dd",
                  "vol"]].head(10).to_string(index=False))

    best = ranked.iloc[0]
    tv, ml = float(best.target_vol), float(best.max_leverage)
    print(f"\nWinner on inner-validation spot Sharpe: target_vol={tv}, "
          f"max_leverage={ml}  (Sharpe={best.sharpe:.3f})")

    fut_row = valid_fut[(valid_fut.target_vol == tv) & (valid_fut.max_leverage == ml)]
    if len(fut_row):
        fr = fut_row.iloc[0]
        print(f"  same config, futures: Sharpe={fr.sharpe:.3f}, "
              f"DD={fr.max_dd:.1f}%, profit={fr.profit_pct:+.1f}%")

    # --- plateau neighbourhood: every grid point within one step of the winner
    tv_idx = TARGET_VOL_GRID.index(tv)
    ml_idx = MAX_LEV_GRID.index(ml)
    tv_nb = TARGET_VOL_GRID[max(0, tv_idx - 1): tv_idx + 2]
    ml_nb = MAX_LEV_GRID[max(0, ml_idx - 1): ml_idx + 2]
    nb = valid_spot[valid_spot.target_vol.isin(tv_nb) & valid_spot.max_leverage.isin(ml_nb)]
    print(f"\n=== plateau neighbourhood (target_vol in {tv_nb}, "
          f"max_leverage in {ml_nb}), inner-validation spot ===")
    pivot = nb.pivot(index="target_vol", columns="max_leverage", values="sharpe")
    print(pivot.to_string())
    print(f"Sharpe range across neighbourhood: "
          f"[{nb.sharpe.min():.3f}, {nb.sharpe.max():.3f}]  "
          f"(project noise floor is +/-0.2 Sharpe, R-20)")

    # --- shipped defaults, same period, for direct comparison
    print("\n=== shipped v4 defaults (target_vol=0.55, max_leverage=2.0), "
          "inner-validation ===")
    for mname, market in MARKETS:
        m, vol, notional, res = measure(cand(**INCUMBENT_KW), *VALID, market=market)
        row(f"  defaults / {mname}", m, vol, notional, res)
    def_spot = valid_spot[(valid_spot.target_vol == 0.55) & (valid_spot.max_leverage == 2.0)]
    if len(def_spot):
        dr = def_spot.iloc[0]
        print(f"  (from grid) defaults spot Sharpe={dr.sharpe:.3f}, "
              f"vol={dr.vol:.3f}, notional={dr.mean_notional:.3f}")

    print(f"\n=== candidate vs defaults ===")
    cand_row = valid_spot[(valid_spot.target_vol == tv) & (valid_spot.max_leverage == ml)].iloc[0]
    dvol = def_spot.iloc[0].vol if len(def_spot) else float("nan")
    dnotional = def_spot.iloc[0].mean_notional if len(def_spot) else float("nan")
    vol_gap = abs(cand_row.vol - dvol) / dvol if dvol else float("nan")
    not_gap = abs(cand_row.mean_notional - dnotional) / dnotional if dnotional else float("nan")
    print(f"  candidate spot Sharpe {cand_row.sharpe:.3f} vs defaults {dr.sharpe:.3f} "
          f"(delta {cand_row.sharpe - dr.sharpe:+.3f})")
    print(f"  realized vol: candidate {cand_row.vol:.3f} vs defaults {dvol:.3f} "
          f"-> relative gap {vol_gap:.1%}")
    print(f"  mean notional: candidate {cand_row.mean_notional:.3f} vs defaults "
          f"{dnotional:.3f} -> relative gap {not_gap:.1%}")
    if vol_gap > 0.10 or not_gap > 0.10:
        print("  >>> EXPOSURE-ARTIFACT WARNING: candidate differs from the shipped "
              "defaults by more than ~10% relative on realized vol or mean notional "
              "on the SAME period. Any Sharpe/return improvement may just be "
              "running hotter, not a better risk/return trade (L-04/R-28/R-31/"
              "R-32/R-33 pattern).")
    else:
        print("  candidate's exposure is within ~10% of the shipped defaults on "
              "this period: an improvement here is NOT explained by simply "
              "carrying more risk.")

    # --- secondary axis: anchor_span_days around the winner
    print(f"\n=== secondary sweep: anchor_span_days at target_vol={tv}, "
          f"max_leverage={ml}, inner-validation spot ===")
    span_rows = []
    for span in ANCHOR_SPAN_GRID:
        strat = cand(target_vol=tv, max_leverage=ml, anchor_span_days=span)
        m, vol, notional, res = measure(strat, *VALID, market=SPOT)
        span_rows.append({"anchor_span_days": span, "sharpe": m.sharpe,
                          "profit_pct": m.profit_pct, "max_dd": m.max_drawdown_pct,
                          "vol": vol})
        row(f"  anchor_span={span}d", m, vol, notional, res)
    pd.DataFrame(span_rows).to_csv(OUT / "anchor_span_neighbourhood.csv", index=False)

    print(f"\ndistinct configurations evaluated in total: {len(_SEEN)}")
    (OUT / "selected_config.txt").write_text(
        f"target_vol={tv}\nmax_leverage={ml}\nanchor_span_days=180\n"
        f"n_configs_evaluated={len(_SEEN)}\n")
    print(f"wrote {OUT / 'selected_config.txt'}")


# --------------------------------------------------------------- causality


def causality() -> None:
    """By-hand lookahead probe - experiments get no CI protection.

    Same two-opposite-tampers procedure as R-28/R-31/R-33: bars after a
    cut are multiplied by 3 in one copy and divided by 3 in the other,
    and every decision at or before the cut must be bit-identical. The
    candidate here reuses kelly_regime_v4's prepare()/on_bar() verbatim
    (only constructor scalars differ), so this mainly re-confirms that
    passing different target_vol/max_leverage/anchor_span_days values
    cannot introduce a leak the registered defaults don't already have -
    but it is run explicitly rather than assumed.
    """
    from tradebot.broker import PaperBroker
    from tradebot.orders import Order
    from tradebot.strategy import Context

    cfg = OUT / "selected_config.txt"
    if cfg.exists():
        kv = dict(line.split("=") for line in cfg.read_text().splitlines() if "=" in line)
        tv, ml = float(kv["target_vol"]), float(kv["max_leverage"])
    else:
        tv, ml = 0.55, 2.0
        print("(no selected_config.txt yet - probing the shipped defaults instead)")

    df = DF.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    strat = KellyRegimeV4(target_vol=tv, max_leverage=ml)

    def decisions(frame):
        s = KellyRegimeV4(target_vol=tv, max_leverage=ml)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        broker.execute(Order(target=0.1), prepared.index[0], float(prepared["open"].iloc[0]))
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down)) if oa != ob]
    pa = strat.prepare(up.copy())
    pb = KellyRegimeV4(target_vol=tv, max_leverage=ml).prepare(down.copy())
    worst_col = float(np.nanmax(np.abs(pa["target"].to_numpy()[:cut]
                                       - pb["target"].to_numpy()[:cut])))
    ok = not bad and worst_col < 1e-12
    print(f"config target_vol={tv} max_leverage={ml}")
    print(f"  orders {'match' if not bad else f'DIFFER at {bad}'}   "
          f"max |target column difference| before the cut = {worst_col:.3e}   "
          f"{'PASS' if ok else 'FAIL'}")

    a = run_backtest(KellyRegimeV4(target_vol=tv, max_leverage=ml), up.iloc[:cut + 1],
                     FUTURES, 1_000.0, data_label=LABEL)
    b = run_backtest(KellyRegimeV4(target_vol=tv, max_leverage=ml), down.iloc[:cut + 1],
                     FUTURES, 1_000.0, data_label=LABEL)
    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-9
    print(f"  max |equity difference| before the cut = {worst_eq:.3e}   "
          f"{'PASS' if worst_eq < 1e-9 else 'FAIL'}")

    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# --------------------------------------------------------------------- eth


def eth() -> None:
    """Pre-registered falsification: candidate vs shipped defaults on ETH.

    Same venue (Bitfinex), same window R-17/R-28/R-31/R-33 used, only the
    asset varies; BTC on this file is the control. Rule fixed BEFORE
    running: if the candidate is not at least as good as the shipped
    defaults on ETH by more than a token margin, or is visibly worse on
    ETH while better on BTC, this whole direction fails.
    """
    cfg = OUT / "selected_config.txt"
    if cfg.exists():
        kv = dict(line.split("=") for line in cfg.read_text().splitlines() if "=" in line)
        tv, ml = float(kv["target_vol"]), float(kv["max_leverage"])
    else:
        tv, ml = 0.55, 2.0
        print("(no selected_config.txt yet - comparing defaults against themselves)")

    rows = []
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            for label, strat in (("candidate", KellyRegimeV4(target_vol=tv, max_leverage=ml)),
                                 ("shipped defaults", KellyRegimeV4(**INCUMBENT_KW)),
                                 ("buy_and_hold", get_strategy("buy_and_hold"))):
                m, vol, notional, res = measure(strat, None, None, df=df, market=market)
                row(f"    {label}", m, vol, notional, res)
                rows.append({"asset": asset, "market": mname, "arm": label,
                             "final": m.final_balance, "profit_pct": m.profit_pct,
                             "sharpe": m.sharpe, "max_dd": m.max_drawdown_pct,
                             "vol": vol, "liquidated": m.liquidated})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "eth_falsification.csv", index=False)

    print("\n=== falsification verdict, candidate vs shipped defaults ===")
    verdict_ok = True
    for asset in ("BTC (control)", "ETH (test)"):
        for mname, _ in MARKETS:
            c = out[(out.asset == asset) & (out.market == mname) & (out.arm == "candidate")].iloc[0]
            d = out[(out.asset == asset) & (out.market == mname) & (out.arm == "shipped defaults")].iloc[0]
            d_sharpe = c.sharpe - d.sharpe
            d_profit = c.profit_pct - d.profit_pct
            d_dd = c.max_dd - d.max_dd  # negative = candidate draws down less
            ok = d_sharpe > -0.05 and d_profit > -2.0  # "at least as good, non-token margin"
            verdict_ok &= ok if "ETH" in asset else True
            print(f"  {asset:16s} {mname:8s} d(Sharpe)={d_sharpe:+.3f} "
                  f"d(profit)={d_profit:+.1f}pp d(maxDD)={d_dd:+.1f}pp  "
                  f"{'OK' if ok else 'WORSE'}")
    print(f"\nETH falsification: {'PASS' if verdict_ok else 'FAIL'}")
    print(f"wrote {OUT / 'eth_falsification.csv'}")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    cmds = {"sweep": sweep, "select": select, "causality": causality, "eth": eth}

    def all_() -> None:
        sweep()
        select()
        causality()
        eth()

    cmds["all"] = all_
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python {sys.argv[0]} [{'|'.join(cmds)}]")
