#!/usr/bin/env python
"""R-188 — ten new candidates in one round: five game-theoretic / state-of-the-art
sizing-and-learning rules and five intraday (1-10 trades a day) rules.

This file is the round's harness AND its pre-registration. Everything below
the "Frozen rules" heading was written before any holdout bar was read and
is the only place the selection and keep rules live; the ledger entry
(docs/LEDGER.md, R-188) quotes them from here.

Candidates (each is a registered module under ``tradebot/strategies`` while the
round runs; the ones the keep rule drops move to ``experiments/r188_<name>.py``):

    game theory / SOTA           intraday
    ------------------------     ---------------------------------------
    robust_kelly   (DRO Kelly)   noise_area_breakout (Zarattini et al. 2024)
    coin_betting   (KT / O&P16)  intraday_momentum   (Gao et al. 2018)
    level_k        (CH / Nagel)  session_drift       (hour-of-day drift)
    focal_levels   (Schelling)   vwap_reversion      (price pressure)
    mfg_crowding   (Casgrain-J)  jump_momentum       (Lee-Mykland 2008)

Slices (ROUTINE.md Step 3): inner-train 2017-01-01 -> 2020-12-31 (fit and
sweep), inner-validation 2021-01-01 -> 2022-12-31 (select between configs),
holdout 2023-01-01 -> end (read once, after ``freeze()``). Every slice is run
through ``tradebot.window.run_period`` so a strategy enters it warm, flat and
with the full $1,000 — the same protocol ``scripts/inference.py`` uses for its
holdout column.

Usage::

    python experiments/r188_shared.py sweep      # inner-train + inner-validation grid
    python experiments/r188_shared.py select     # apply the frozen selection rule
    python experiments/r188_shared.py hourly     # Step-A hour-of-day dispersion gate
    python experiments/r188_shared.py holdout    # ONE pass, frozen configs, 2023+
    python experiments/r188_shared.py decide     # apply the frozen keep rule

Outputs land in ``reports/r188_candidates/``.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.session import BARS_PER_DAY  # noqa: E402
from tradebot.window import run_period  # noqa: E402

OUT = ROOT / "reports" / "r188_candidates"

SLICES = {
    "inner_train": ("2017-01-01", "2020-12-31"),
    "inner_val": ("2021-01-01", "2022-12-31"),
    "holdout": ("2023-01-01", None),
}
MARKETS = {"spot": MarketSpec.spot(), "futures": MarketSpec.futures(leverage=5.0)}
BENCHMARK = "buy_and_hold"
NOISE_FLOOR = 0.20  # R-20's ±0.2 Sharpe noise floor

GAME_THEORY = ("robust_kelly", "coin_betting", "level_k", "focal_levels", "mfg_crowding")
INTRADAY = ("noise_area_breakout", "intraday_momentum", "session_drift",
            "vwap_reversion", "jump_momentum")
CANDIDATES = GAME_THEORY + INTRADAY


def _cls(name: str):
    """The candidate's class, wherever it currently lives (registered or dropped)."""
    try:
        from tradebot.registry import available_strategies
        reg = available_strategies()
        if name in reg:
            return reg[name]
    except Exception:  # pragma: no cover - registry import problems surface below
        pass
    import importlib

    mod = importlib.import_module(f"r188_{name}")
    for obj in vars(mod).values():
        if isinstance(obj, type) and getattr(obj, "name", None) == name:
            return obj
    raise KeyError(name)


# ----------------------------------------------------------------- the grid
#
# Small on purpose: ROUTINE.md counts every configuration evaluated toward
# the project's deflated-Sharpe trials, across all ten candidates at once.
# 31 configurations x 2 markets x 2 training slices = 124 evaluations.

GRID: dict[str, list[tuple[str, dict]]] = {
    "robust_kelly": [
        ("kappa0.25", dict(kappa=0.25)),
        ("kappa0.5", dict(kappa=0.5)),
        ("kappa0.75", dict(kappa=0.75)),
    ],
    "coin_betting": [
        ("day_d0.99_s1", dict(round_bars=BARS_PER_DAY, discount=0.99, scale=1.0)),
        ("day_d0.99_s3", dict(round_bars=BARS_PER_DAY, discount=0.99, scale=3.0)),
        ("4h_d0.99_s3", dict(round_bars=48, discount=0.99, scale=3.0)),
    ],
    "level_k": [
        ("anticipate_f12", dict(anticipate=True, fast_bars=12)),
        ("anticipate_f36", dict(anticipate=True, fast_bars=36)),
        ("follow_f12", dict(anticipate=False, fast_bars=12)),
        ("follow_f36", dict(anticipate=False, fast_bars=36)),
    ],
    "focal_levels": [
        ("breakout_h24", dict(mode="breakout", hold_bars=24)),
        ("breakout_h24_s5", dict(mode="breakout", hold_bars=24, spacing_frac=0.05)),
        ("bounce_h24", dict(mode="bounce", hold_bars=24)),
    ],
    "mfg_crowding": [
        ("gamma0", dict(gamma=0.0)),
        ("gamma0.5", dict(gamma=0.5)),
        ("gamma1.0", dict(gamma=1.0)),
    ],
    "noise_area_breakout": [
        ("band1.0", dict(band_mult=1.0)),
        ("band1.5", dict(band_mult=1.5)),
        ("band1.0_hourly", dict(band_mult=1.0, check_every=12)),
    ],
    "intraday_momentum": [
        ("h1", dict(first_hours=1, last_hours=1)),
        ("h2", dict(first_hours=2, last_hours=2)),
        ("h4", dict(first_hours=4, last_hours=4)),
    ],
    "session_drift": [
        ("lb90_t1.5", dict(lookback_days=90, t_min=1.5)),
        ("lb90_t2.5", dict(lookback_days=90, t_min=2.5)),
        ("lb180_t2.0", dict(lookback_days=180, t_min=2.0)),
    ],
    "vwap_reversion": [
        ("z2.0", dict(entry_z=2.0)),
        ("z3.0", dict(entry_z=3.0)),
        ("z2.0_edge3", dict(entry_z=2.0, min_edge_mult=3.0)),
    ],
    "jump_momentum": [
        ("a0.01_h12", dict(alpha=0.01, hold_bars=12)),
        ("a0.01_h48", dict(alpha=0.01, hold_bars=48)),
        ("a0.05_h24", dict(alpha=0.05, hold_bars=24)),
    ],
}


# ------------------------------------------------------------ Frozen rules
#
# Written 2026-09-02 before ``sweep`` produced a number and before any bar
# dated 2023-01-01 or later was read by this round.
#
# SELECTION (inner-validation only): for each candidate, the configuration
# with the highest mean of its spot and futures Sharpe on 2021-2022. Ties go
# to the configuration with fewer round trips. The winner's parameters become
# the registered defaults; the count of configurations evaluated (all of them,
# both slices, both markets) goes to the ledger.
#
# KEEP / PROMOTE / DROP (holdout, spot, 0.10% taker, fresh $1,000 from
# 2023-01-01, one pass per frozen candidate):
#
#   PROMOTE  the frozen config beats buy_and_hold's holdout spot Sharpe by more
#            than NOISE_FLOOR, did the same on inner-validation, and has a
#            lower holdout max drawdown than buy_and_hold. (ROUTINE.md's
#            promotion bar; nothing is expected to clear it.)
#   KEEP     not PROMOTE, but (a) the holdout spot balance ends above $1,000,
#            (b) the holdout spot Sharpe is not below buy_and_hold's by more
#            than NOISE_FLOOR, and (c) the inner-validation spot balance also
#            ended above $1,000. A KEEP is registered in the comparison table
#            as a REGISTERED strategy with an interval, not a promotion.
#   DROP     everything else. Code moves to experiments/, the ledger records
#            it, section C gets a do-not-retry ruling.
#
# The rule partitions the outcome space: PROMOTE is tested first, then KEEP,
# then DROP is the complement. Nothing about it may change after ``holdout``
# runs; if it did, the ledger entry must say so and the result is in-sample.
#
# FALSIFICATION (named now, run only for a KEEP or PROMOTE): the kept config
# must remain profitable on the holdout at Binance's regular 0.10% spot tier
# with 1bp slippage AND at a 0.20% taker; a strategy whose profit disappears
# between 0.10% and 0.20% is a fee artefact and is downgraded to DROP.


def _one(args):
    name, label, params, slice_name, market_name, fee, slippage = args
    df = _DF
    start, end = SLICES[slice_name]
    cls = _cls(name)
    strat = cls(**params) if params is not None else cls()
    market = MARKETS[market_name]
    if fee is not None:
        market = MarketSpec.spot(fee_rate=fee) if market_name == "spot" else \
            MarketSpec.futures(leverage=5.0, fee_rate=fee)
    t0 = time.time()
    res = run_period(strat, df, start, end, market=market, start_balance=1_000.0,
                     slippage_bps=slippage, data_label=_LABEL)
    m = compute_metrics(res)
    days = len(res.equity) / BARS_PER_DAY
    return {
        "strategy": name, "config": label, "slice": slice_name, "market": market_name,
        "fee": market.fee_rate, "slippage_bps": slippage,
        "final_balance": m.final_balance, "profit_pct": m.profit_pct,
        "sharpe": m.sharpe, "max_dd_pct": m.max_drawdown_pct,
        "trades": m.num_trades, "fills": len(res.fills), "trades_per_day": m.num_trades / days,
        "fills_per_day": len(res.fills) / days, "time_in_market_pct": m.time_in_market_pct,
        "fees_paid": m.fees_paid, "liquidated": m.liquidated, "days": days,
        "seconds": time.time() - t0,
    }


_DF, _LABEL = None, None


def _load():
    global _DF, _LABEL
    if _DF is None:
        _DF, _LABEL = load_dataset(ROOT / "data", "spot")
    return _DF


def _run_jobs(jobs, procs: int = 4) -> pd.DataFrame:
    _load()
    rows = []
    with ProcessPoolExecutor(max_workers=procs) as ex:  # fork: workers inherit _DF
        for row in ex.map(_one, jobs):
            print(f"  {row['strategy']:20s} {row['config']:16s} {row['slice']:11s} "
                  f"{row['market']:7s} ${row['final_balance']:>10,.0f} sharpe {row['sharpe']:>5.2f} "
                  f"DD {row['max_dd_pct']:>5.1f}% {row['trades_per_day']:>5.2f} trades/day "
                  f"[{row['seconds']:.0f}s]", file=sys.stderr)
            rows.append(row)
    return pd.DataFrame(rows)


def sweep(procs: int = 4) -> pd.DataFrame:
    """Every configuration on inner-train and inner-validation, both markets."""
    jobs = [(name, label, params, s, mk, None, 0.0)
            for name, cfgs in GRID.items() for label, params in cfgs
            for s in ("inner_train", "inner_val") for mk in MARKETS]
    jobs += [(BENCHMARK, "default", {}, s, mk, None, 0.0)
             for s in ("inner_train", "inner_val") for mk in MARKETS]
    frame = _run_jobs(jobs, procs)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "sweep.csv", index=False)
    n_cfg = sum(len(c) for c in GRID.values())
    print(f"\n{n_cfg} configurations x 2 slices x 2 markets = "
          f"{len(frame) - 4} candidate evaluations (+4 benchmark) -> {OUT / 'sweep.csv'}")
    return frame


def select(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The frozen selection rule, applied to inner-validation."""
    if frame is None:
        frame = pd.read_csv(OUT / "sweep.csv")
    val = frame[(frame["slice"] == "inner_val") & (frame.strategy != BENCHMARK)]
    agg = val.groupby(["strategy", "config"]).agg(
        mean_sharpe=("sharpe", "mean"), trades=("trades", "sum")).reset_index()
    agg = agg.sort_values(["strategy", "mean_sharpe", "trades"], ascending=[True, False, True])
    chosen = agg.groupby("strategy").head(1).reset_index(drop=True)
    chosen["params"] = [dict(GRID[s])[c] for s, c in zip(chosen.strategy, chosen.config)]
    OUT.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(OUT / "selected.csv", index=False)
    bench = frame[(frame.strategy == BENCHMARK) & (frame["slice"] == "inner_val")]
    print("\nselected on inner-validation (mean spot/futures Sharpe):")
    for _, r in chosen.iterrows():
        print(f"  {r.strategy:20s} {r.config:16s} mean Sharpe {r.mean_sharpe:>5.2f}  {r.params}")
    print(f"  {BENCHMARK:20s} {'':16s} mean Sharpe {bench.sharpe.mean():>5.2f}")
    return chosen


def frozen_params() -> dict[str, dict]:
    sel = pd.read_csv(OUT / "selected.csv")
    return {r.strategy: dict(GRID[r.strategy])[r.config] for _, r in sel.iterrows()}


def holdout(procs: int = 4) -> pd.DataFrame:
    """ONE pass: the frozen configuration of every candidate on 2023+, both markets.

    Also runs the two fee/slippage falsification cells named above for every
    candidate in the same pass, so the holdout is consulted once, not twice.
    """
    params = frozen_params()
    jobs = []
    for name, p in params.items():
        for mk in MARKETS:
            jobs.append((name, "frozen", p, "holdout", mk, None, 0.0))
        jobs.append((name, "frozen_slip1bp", p, "holdout", "spot", None, 1.0))
        jobs.append((name, "frozen_fee0.20", p, "holdout", "spot", 0.002, 0.0))
    for mk in MARKETS:
        jobs.append((BENCHMARK, "default", {}, "holdout", mk, None, 0.0))
    jobs.append((BENCHMARK, "default_fee0.20", {}, "holdout", "spot", 0.002, 0.0))
    frame = _run_jobs(jobs, procs)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "holdout.csv", index=False)
    return frame


def decide() -> pd.DataFrame:
    """The frozen KEEP / PROMOTE / DROP rule."""
    sweep_df = pd.read_csv(OUT / "sweep.csv")
    hold = pd.read_csv(OUT / "holdout.csv")
    sel = pd.read_csv(OUT / "selected.csv")

    def cell(frame, strategy, config, slice_name, market):
        sub = frame[(frame.strategy == strategy) & (frame.config == config)
                    & (frame["slice"] == slice_name) & (frame.market == market)]
        assert len(sub) == 1, (strategy, config, slice_name, market, len(sub))
        return sub.iloc[0]

    bench_h = cell(hold, BENCHMARK, "default", "holdout", "spot")
    bench_v = cell(sweep_df, BENCHMARK, "default", "inner_val", "spot")
    bench_h20 = cell(hold, BENCHMARK, "default_fee0.20", "holdout", "spot")

    rows = []
    for _, s in sel.iterrows():
        h = cell(hold, s.strategy, "frozen", "holdout", "spot")
        hf = cell(hold, s.strategy, "frozen", "holdout", "futures")
        v = cell(sweep_df, s.strategy, s.config, "inner_val", "spot")
        slip = cell(hold, s.strategy, "frozen_slip1bp", "holdout", "spot")
        fee20 = cell(hold, s.strategy, "frozen_fee0.20", "holdout", "spot")

        promote = (h.sharpe > bench_h.sharpe + NOISE_FLOOR
                   and v.sharpe > bench_v.sharpe + NOISE_FLOOR
                   and h.max_dd_pct < bench_h.max_dd_pct)
        keep = (h.final_balance > 1_000.0
                and h.sharpe >= bench_h.sharpe - NOISE_FLOOR
                and v.final_balance > 1_000.0)
        verdict = "PROMOTE" if promote else ("KEEP" if keep else "DROP")
        falsified = False
        if verdict != "DROP":
            falsified = not (slip.final_balance > 1_000.0 and fee20.final_balance > 1_000.0)
            if falsified:
                verdict = "DROP (falsified by fees)"
        rows.append({
            "strategy": s.strategy, "config": s.config, "verdict": verdict,
            "holdout_spot_final": h.final_balance, "holdout_spot_sharpe": h.sharpe,
            "holdout_spot_dd": h.max_dd_pct, "holdout_spot_trades_per_day": h.trades_per_day,
            "holdout_fut_final": hf.final_balance, "holdout_fut_sharpe": hf.sharpe,
            "holdout_fut_dd": hf.max_dd_pct, "holdout_fut_liquidated": hf.liquidated,
            "innerval_spot_final": v.final_balance, "innerval_spot_sharpe": v.sharpe,
            "holdout_spot_slip1bp_final": slip.final_balance,
            "holdout_spot_fee0.20_final": fee20.final_balance,
            "bench_holdout_sharpe": bench_h.sharpe, "bench_holdout_dd": bench_h.max_dd_pct,
            "bench_holdout_final": bench_h.final_balance,
            "bench_holdout_fee0.20_final": bench_h20.final_balance,
            "bench_innerval_sharpe": bench_v.sharpe,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "decision.csv", index=False)
    print("\nFROZEN RULE, APPLIED ONCE:")
    for _, r in out.iterrows():
        print(f"  {r.strategy:20s} {r.verdict:26s} holdout spot ${r.holdout_spot_final:>9,.0f} "
              f"sharpe {r.holdout_spot_sharpe:>5.2f} (hold {r.bench_holdout_sharpe:.2f}) "
              f"DD {r.holdout_spot_dd:>4.1f}% (hold {r.bench_holdout_dd:.1f}%)  "
              f"inner-val ${r.innerval_spot_final:>9,.0f}  {r.holdout_spot_trades_per_day:.2f} trades/day")
    return out


def hourly_gate(n_boot: int = 2_000, seed: int = 0) -> dict:
    """Step-A measurement for session_drift: is BTC's hour-of-day mean-return
    dispersion on inner-train distinguishable from a block-bootstrap null?

    Mirrors R-75's day-of-week gate (which failed). Statistic: the standard
    deviation across the 24 UTC hours of the mean 5-minute return. Null:
    circularly shift the return series by a random number of bars (multiples
    of 5 minutes, not of a day), which keeps every autocorrelation and
    destroys only the clock alignment.
    """
    df = _load()
    start, end = SLICES["inner_train"]
    sub = df.loc[start:end]
    r = np.log(sub["close"]).diff().fillna(0.0).to_numpy()
    hour = sub.index.hour.to_numpy()

    def stat(x):
        means = np.bincount(hour, weights=x, minlength=24) / np.bincount(hour, minlength=24)
        return float(np.std(means))

    observed = stat(r)
    rng = np.random.default_rng(seed)
    null = np.array([stat(np.roll(r, int(rng.integers(1, len(r))))) for _ in range(n_boot)])
    p = float((null >= observed).mean())
    means = np.bincount(hour, weights=r, minlength=24) / np.bincount(hour, minlength=24)
    out = {"observed_sd_across_hours": observed, "null_mean": float(null.mean()),
           "null_p95": float(np.quantile(null, 0.95)), "empirical_p": p,
           "best_hour_utc": int(np.argmax(means)), "worst_hour_utc": int(np.argmin(means))}
    print("hour-of-day gate (inner-train 2017-2020):")
    for k, v in out.items():
        print(f"  {k:26s} {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.Series(out).to_csv(OUT / "hourly_gate.csv")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "sweep":
        sweep()
        select()
    elif cmd == "select":
        select()
    elif cmd == "hourly":
        hourly_gate()
    elif cmd == "holdout":
        holdout()
        decide()
    elif cmd == "decide":
        decide()
    else:
        print(__doc__)
