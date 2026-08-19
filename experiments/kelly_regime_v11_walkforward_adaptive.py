#!/usr/bin/env python
"""Walk-forward re-estimation of v4's sizing frontier (N=3 axis, ARCHITECTURE not signal).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea, in one sentence
--------------------------
Replace ``kelly_regime_v4``'s two frozen sizing constants (``target_vol``,
``max_leverage``) with values that are periodically RE-ESTIMATED from a
trailing rolling lookback window on a fixed schedule, using ONLY data
strictly before each refit point -- everything else (the vote, the anchor
ladder, the vol-breakout hysteresis, the deadband) is copied verbatim from
``kelly_regime_v3``/``v4``, unchanged.

Twelve prior branches on this axis (R-34 through R-44, see docs/LEDGER.md
section C) all added a NEW SIGNAL or gate to v4's fixed architecture and
all failed the same way: beat v4 on 2021-2022 inner-validation, lost on the
pre-2020 BTC control or ETH. This branch adds no new signal, no new data
source. It changes the ARCHITECTURE instead: v4's constants were "frozen
once from a single full-history backtest" (see this file's own commission).
Adaptive/walk-forward regime-based BTC trading literature (2025) argues
that re-fitting a model's own parameters on a trailing window, rather than
freezing them from one historical fit, is what a real deployment would
actually do, and is more robust to the regime that produced the frozen fit
having ended. The N≈3 problem (effective sample size ~3 regime events) is
attacked by turning "fit once, hope it generalizes to events never in the
fitting sample" into "re-fit sequentially as each event becomes historical
data" -- closer to what live re-tuning looks like, and a genuinely
different mechanism than any of the twelve signal-adding branches.

Pre-registered BEFORE any code ran (see module-level constants below):
  - refit schedule: PRIMARY = refit every 365 days, using the trailing 730
    days (2x the refit interval, so every refit's fit uses two full cycles
    of the anchor ladder's slowest anchor (80d) and material history of at
    least one full year). Two NEIGHBOURS (180d/365d "faster", 730d/1460d
    "slower") are also run, decided at the same time, purely to check the
    schedule is a plateau and not a peak -- exactly as ROUTINE.md's
    promotion bar requires ("the parameter neighbourhood must be a
    plateau, not a peak"). The schedule set does not change after seeing
    any result.
  - re-estimation rule: at each refit point, grid-search
    ``target_vol in {0.35,0.45,0.55,0.65,0.75} x max_leverage in
    {1.0,1.5,2.0,2.5,3.0}`` (25 combos, symmetric around v4's shipped
    0.55/2.0), scoring each combo by a FAST, fully-causal proxy Sharpe of
    ``lag(exposure) * r`` over the trailing lookback window only (no fees,
    no deadband, no hysteresis switch in the proxy -- a deliberate
    simplification of the scoring rule only; every REPORTED number below
    still comes from the real engine with real fees). Ties broken toward
    the point closest to v4's own defaults, for determinism.
  - first-refit fallback: until ``lookback_days`` of trailing history
    exists (measured in bar POSITIONS from wherever the frame handed to
    ``prepare()`` starts, not calendar dates -- this project's engine
    calls ``prepare()`` fresh per backtest, so "trailing history" means
    "history in the frame", exactly like every other constructor scalar
    here), every bar uses v4's own shipped defaults
    (``target_vol=0.55, max_leverage=2.0``) verbatim. Stated up front, not
    discovered after the fact.

Falsification test (pre-registered, the project's standard one, chosen
before running anything): does the candidate match-or-beat
``kelly_regime_v4`` on ETH data (Bitfinex, 2016-2019), and not visibly
underperform it on the pre-2020 BTC control run of the identical pipeline?
See ``eth()`` below.

Not a duplicate of
-------------------
- R-37 (conservative): retunes ``target_vol``/``max_leverage`` ONCE,
  globally, using knowledge of the confirmed post-2021 edge (R-36) to pick
  a single new frozen pair -- a point estimate, exactly like v4's own
  original fit. This file never picks one pair and freezes it; it produces
  a genuinely time-varying, piecewise-constant series re-estimated
  SEQUENTIALLY through time, each estimate blind to everything after its
  own refit point, which R-37's single global retune structurally cannot
  be (it used the whole pre-2023 evidence, inner-validation included, to
  justify the one number it shipped).
- R-37 (novel, per-vote-state Kelly fraction), R-38 (risk-constrained Kelly
  cap, CRRA fraction): both REPLACE v4's ``min(target_vol/vol,
  max_leverage)`` formula with a different formula driven by a new
  continuous estimate (mu/sigma^2, a drawdown-probability cap). This file
  keeps that exact formula, and v3's exact hysteresis switch, unchanged --
  the only new arithmetic is that the two SCALARS plugged into it are
  looked up from a piecewise-constant refit schedule instead of being
  literal constructor constants. No new signal source, no new formula.
- R-40 (ladder bagging / cross-ladder shrinkage): re-derives the VOTE from
  multiple anchor ladders. This file's vote is v4's single (20,40,80)
  ladder, byte-for-byte unchanged; nothing here touches the anchor ladder.
- R-34/R-35/R-38 basis/funding/on-chain branches: all add a new external
  data source as a SIZE input. This file reads no data this project has
  not already used for v4 itself (OHLCV only).

Pre-registered failure modes (named before any code ran)
------------------------------------------------------------
(a) The refit schedule barely moves target_vol/max_leverage away from
    v4's own 0.55/2.0 (the grid's argmax keeps landing near the shipped
    defaults) -- "no effect", a legitimate negative, not a bug.
(b) Any improvement sits inside the +/-0.2 Sharpe noise floor.
(c) Exposure-level artifact: R^2 > 0.95 against a mean-notional-matched
    flat rescale of v4's own target series (this project's standard test,
    R-33/R-34's diagnostic), checked explicitly below.
(d) Fails the ETH falsification test, or is visibly worse on ETH than on
    the BTC control through the identical pipeline -- the exact signature
    of all twelve prior SIZE-axis branches (fitted to the 2021-2022
    window, not generalizable).
(e) A lookahead bug in the refit loop itself makes everything above moot;
    checked explicitly and FIRST, per this session's brief, via ``causality``.

Usage
-----
    python experiments/kelly_regime_v11_walkforward_adaptive.py sweep       # step 3
    python experiments/kelly_regime_v11_walkforward_adaptive.py select      # step 4/5
    python experiments/kelly_regime_v11_walkforward_adaptive.py artifact    # failure mode (c)
    python experiments/kelly_regime_v11_walkforward_adaptive.py causality   # step 6 (do this first)
    python experiments/kelly_regime_v11_walkforward_adaptive.py eth         # step 7 / failure mode (d)
"""

from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# v4's shipped defaults -- the fallback used until a lookback window exists,
# and the centre of the re-estimation grid.
FALLBACK_TARGET_VOL = 0.55
FALLBACK_MAX_LEVERAGE = 2.0

# Re-estimation grid, fixed in advance -- symmetric around the fallback.
TV_GRID = (0.35, 0.45, 0.55, 0.65, 0.75)
ML_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)


# --------------------------------------------------------------------- strategy


class KellyRegimeV11WalkforwardAdaptive(Strategy):
    """v3/v4's unchanged vote+hysteresis-scale formula, with target_vol/max_leverage refit on a schedule.

    Everything about the vote (latched anchor ladder), the vol-breakout
    hysteresis state machine, and the deadband is copied verbatim from
    ``kelly_regime_v3.py``/``kelly_regime_v4.py``. The only change: instead
    of two constructor scalars, ``target_vol``/``max_leverage`` are two
    per-bar arrays, piecewise-constant between refit points, each segment
    chosen by a grid search over trailing history STRICTLY BEFORE that
    segment's own start -- never using a bar at or after the point the
    segment starts from.
    """

    name = "kelly_regime_v11_walkforward_adaptive"

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70, high_out: float = 1.20,
                 low_in: float = 0.55, low_out: float = 0.85,
                 refit_days: int = 365, lookback_days: int = 730,
                 tv_grid: tuple[float, ...] = TV_GRID, ml_grid: tuple[float, ...] = ML_GRID) -> None:
        self.horizons = horizons
        self.band = band
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        self.refit_days = refit_days
        self.lookback_days = lookback_days
        self.tv_grid = tv_grid
        self.ml_grid = ml_grid
        max_anchor_days = max(horizons)
        # warmup: enough for the vote's own slowest anchor AND one full
        # lookback window, so the first refit inside the tested period (not
        # just the fallback) is exercised whenever the data allows it.
        self.warmup = int(max(max_anchor_days, lookback_days) * BARS_PER_DAY) + 10
        self.n_refits_ = 0  # filled in by prepare(), read by callers for reporting

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        idx = df.index
        n = len(df)
        r = np.log(close).diff().to_numpy()

        # ---- vote: byte-for-byte identical to kelly_regime.py / v4 ----
        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=idx,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()

        # ---- realized vol + slow vol anchor: byte-for-byte identical to v3 ----
        rs = pd.Series(r, index=idx)
        vol = (rs.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                    min_periods=BARS_PER_DAY).mean().to_numpy())

        # ---- walk-forward re-estimation of (target_vol, max_leverage) ----
        # Causal by construction: refit at bar j uses only frac/vol/r at
        # indices < j (see _score_window). Piecewise-constant between
        # refit points; fallback to v4's shipped defaults before the first
        # refit has enough trailing history.
        lookback_bars = int(self.lookback_days * BARS_PER_DAY)
        refit_bars = int(self.refit_days * BARS_PER_DAY)

        tv_active = np.full(n, FALLBACK_TARGET_VOL)
        ml_active = np.full(n, FALLBACK_MAX_LEVERAGE)

        n_refits = 0
        if lookback_bars < n:
            j = lookback_bars
            while j < n:
                a = j - lookback_bars  # window is frac/vol/r[a:j) -- strictly < j
                tv, ml = self._grid_search(frac, vol, r, a, j)
                nxt = min(j + refit_bars, n)
                tv_active[j:nxt] = tv
                ml_active[j:nxt] = ml
                n_refits += 1
                j = nxt
        self.n_refits_ = n_refits

        # ---- v3's exact hysteresis scale, but with per-bar tv/ml arrays ----
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(tv_active / vol, ml_active)
            steady = np.minimum(tv_active / slow, ml_active)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        scale = np.zeros(n)
        state = 0
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale[i] = full[i] if state != 0 else steady[i]

        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = frac[i] * scale[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["_target_vol_active"] = tv_active
        df["_max_leverage_active"] = ml_active
        df["_frac"] = frac
        return df

    def _grid_search(self, frac: np.ndarray, vol: np.ndarray, r: np.ndarray,
                      a: int, j: int) -> tuple[float, float]:
        """Score every (target_vol, max_leverage) combo on window [a, j) only.

        Proxy score: annualized mean/std of ``lag(exposure_raw) * r`` inside
        the window -- exposure_raw uses the plain vol-target formula
        (``frac * min(tv/vol, ml)``, no hysteresis switch, no fees, no
        deadband: a deliberate simplification of the SCORING rule only).
        Every index touched is in ``[a, j)``, strictly before ``j`` -- the
        refit point itself never reads its own future.
        """
        fw = frac[a:j]
        vw = vol[a:j]
        rw = r[a:j]
        best = None
        best_key = None
        for tv, ml in product(self.tv_grid, self.ml_grid):
            with np.errstate(divide="ignore", invalid="ignore"):
                exp_raw = fw * np.minimum(tv / vw, ml)
            exp_raw = np.where(np.isfinite(exp_raw), exp_raw, 0.0)
            # lag exposure by one bar so the score at position k uses
            # exp_raw[k-1] (decided using data < k) times rw[k] (the
            # realized return over [k-1, k]) -- both strictly inside [a, j).
            pr = exp_raw[:-1] * rw[1:]
            pr = pr[np.isfinite(pr)]
            if len(pr) < 2:
                score = -np.inf
            else:
                mu, sd = float(np.mean(pr)), float(np.std(pr, ddof=1))
                score = (mu / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else mu * BARS_PER_YEAR
            dist = abs(tv - FALLBACK_TARGET_VOL) + abs(ml - FALLBACK_MAX_LEVERAGE)
            key = (score, -dist)  # tie-break toward v4's own defaults
            if best_key is None or key > best_key:
                best_key = key
                best = (tv, ml)
        return best

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# ------------------------------------------------------------------------ harness

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
# OOS_START = "2023-01-01"  -- NEVER read in this file, by construction.

INCUMBENT = "kelly_regime_v4"

# Schedules -- fixed in advance, not fitted; three total: primary + two
# neighbours for the plateau check. Never extended after seeing a result.
SCHEDULES = {
    "primary_365d_730d": dict(refit_days=365, lookback_days=730),
    "faster_180d_365d": dict(refit_days=180, lookback_days=365),
    "slower_730d_1460d": dict(refit_days=730, lookback_days=1460),
}
PRIMARY = "primary_365d_730d"

N_EVALUATED = 0  # distinct schedule configurations searched in step 3

OUT = ROOT / "reports" / "kelly_regime_v11_walkforward_adaptive"


def mean_notional(result) -> float:
    if "target" not in result.df:
        return float("nan")
    tgt = np.abs(result.df["target"].to_numpy(dtype=float))
    return float(np.mean(np.clip(tgt, 0.0, result.market.leverage)))


def realized_vol(equity) -> float:
    eq = equity.to_numpy(dtype=float) if hasattr(equity, "to_numpy") else np.asarray(equity)
    if len(eq) < 3:
        return float("nan")
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
    return float(rets.std(ddof=1) * np.sqrt(BARS_PER_YEAR))


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0, count=False):
    """One backtest -> (metrics, realized vol, mean notional, result, n_refits)."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                         start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    n_refits = getattr(strategy, "n_refits_", None)
    return m, realized_vol(result.equity), mean_notional(result), result, n_refits


def line(tag, m, vol, notional, n_refits):
    refits_s = f"refits={n_refits:>2d}" if n_refits is not None else "refits= -"
    print(f"  {tag:44s} final=${m.final_balance:>11,.0f} "
          f"vol={vol:5.3f} notional={notional:5.3f} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} {refits_s}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# --------------------------------------------------------------------------- step 3


def sweep() -> pd.DataFrame:
    """Step 3: measure every pre-registered schedule on inner-train, both markets, vs v4 control."""
    rows = []
    t0 = time.time()
    for label, kw in SCHEDULES.items():
        for mi, (mname, market) in enumerate(MARKETS):
            strat = KellyRegimeV11WalkforwardAdaptive(**kw)
            m, vol, notional, res, nref = measure(strat, *TRAIN, market=market, count=(mi == 0))
            rows.append({"label": label, "refit_days": kw["refit_days"],
                         "lookback_days": kw["lookback_days"], "market": mname,
                         "final": m.final_balance, "vol": vol, "notional": notional,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "trades": m.num_trades, "fees": m.fees_paid,
                         "n_refits": nref, "liquidated": m.liquidated})
            print(f"[{N_EVALUATED:>2d}] {label:20s} {mname:8s} "
                  f"final=${m.final_balance:>10,.0f} DD={m.max_drawdown_pct:>5.1f}% "
                  f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} refits={nref} "
                  f"[{time.time() - t0:.0f}s]")
    for mname, market in MARKETS:
        m, vol, notional, res, _ = measure(get_strategy(INCUMBENT), *TRAIN, market=market)
        rows.append({"label": "kelly_regime_v4_control", "refit_days": None,
                     "lookback_days": None, "market": mname, "final": m.final_balance,
                     "vol": vol, "notional": notional, "max_dd": m.max_drawdown_pct,
                     "sharpe": m.sharpe, "trades": m.num_trades, "fees": m.fees_paid,
                     "n_refits": None, "liquidated": m.liquidated})
        print(f"[ctl] {'kelly_regime_v4_control':20s} {mname:8s} "
              f"final=${m.final_balance:>10,.0f} DD={m.max_drawdown_pct:>5.1f}% "
              f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d}")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "sweep_inner_train.csv", index=False)
    print(f"\nconfigurations evaluated (step 3): {N_EVALUATED}")
    print(f"written: {OUT / 'sweep_inner_train.csv'}")
    return out


# --------------------------------------------------------------------------- step 5


def select() -> pd.DataFrame:
    """Step 5: score every schedule on inner-validation, both markets, vs v4 control."""
    rows = []
    for label, kw in SCHEDULES.items():
        for mname, market in MARKETS:
            strat = KellyRegimeV11WalkforwardAdaptive(**kw)
            m, vol, notional, res, nref = measure(strat, *VALID, market=market)
            rows.append({"label": label, "refit_days": kw["refit_days"],
                         "lookback_days": kw["lookback_days"], "market": mname,
                         "final": m.final_balance, "vol": vol, "notional": notional,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "trades": m.num_trades, "fees": m.fees_paid,
                         "n_refits": nref, "liquidated": m.liquidated})
        s = [x for x in rows if x["label"] == label and x["market"] == "spot"][-1]
        f = [x for x in rows if x["label"] == label and x["market"] == "futures"][-1]
        print(f"{label:20s} spot: ${s['final']:>9,.0f} DD{s['max_dd']:>5.1f}% "
              f"sh{s['sharpe']:>5.2f} tr{s['trades']:>4d} refits={s['n_refits']}   "
              f"fut: ${f['final']:>9,.0f} DD{f['max_dd']:>5.1f}% "
              f"sh{f['sharpe']:>5.2f} tr{f['trades']:>4d} refits={f['n_refits']}")
    for mname, market in MARKETS:
        m, vol, notional, res, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        rows.append({"label": "kelly_regime_v4_control", "refit_days": None,
                     "lookback_days": None, "market": mname, "final": m.final_balance,
                     "vol": vol, "notional": notional, "max_dd": m.max_drawdown_pct,
                     "sharpe": m.sharpe, "trades": m.num_trades, "fees": m.fees_paid,
                     "n_refits": None, "liquidated": m.liquidated})
    ctl_s = [x for x in rows if x["label"] == "kelly_regime_v4_control" and x["market"] == "spot"][-1]
    ctl_f = [x for x in rows if x["label"] == "kelly_regime_v4_control" and x["market"] == "futures"][-1]
    print(f"{'kelly_regime_v4 (control)':20s} spot: ${ctl_s['final']:>9,.0f} "
          f"DD{ctl_s['max_dd']:>5.1f}% sh{ctl_s['sharpe']:>5.2f} tr{ctl_s['trades']:>4d}   "
          f"fut: ${ctl_f['final']:>9,.0f} DD{ctl_f['max_dd']:>5.1f}% "
          f"sh{ctl_f['sharpe']:>5.2f} tr{ctl_f['trades']:>4d}")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "select_inner_validation.csv", index=False)
    print(f"\nwritten: {OUT / 'select_inner_validation.csv'}")

    print("\n=== plateau check: primary vs neighbours, inner-validation spot Sharpe ===")
    for label in SCHEDULES:
        rowspot = [x for x in rows if x["label"] == label and x["market"] == "spot"][-1]
        print(f"  {label:20s} sharpe={rowspot['sharpe']:.3f}")
    return out


# --------------------------------------------------------------------------- failure mode (c)


def exposure_artifact_check() -> None:
    """Mandatory exposure-artifact check (ROUTINE.md standing rule, sharpened by R-33).

    Build a "flat-rescaled v4" comparator: v4's own unchanged target,
    multiplied by a single constant c chosen so its mean notional matches
    the primary candidate's mean notional over the SAME period. Report R^2
    of the candidate's target series against that flat rescale, on inner-
    validation, both markets. R^2 > 0.95 means "this is the standard
    exposure-level artifact".
    """
    print("\nexposure-artifact check (inner-validation, mean-notional-matched flat rescale of v4):")
    kw = SCHEDULES[PRIMARY]
    for mname, market in MARKETS:
        cand = KellyRegimeV11WalkforwardAdaptive(**kw)
        m_c, vol_c, not_c, res_c, nref = measure(cand, *VALID, market=market)
        v4 = get_strategy(INCUMBENT)
        m_v4, vol_v4, not_v4, res_v4, _ = measure(v4, *VALID, market=market)

        cand_t = res_c.df["target"].to_numpy(dtype=float)
        v4_t = res_v4.df["target"].reindex(res_c.df.index).to_numpy(dtype=float)
        c = not_c / not_v4 if not_v4 > 0 else float("nan")
        flat = c * v4_t

        mask = np.isfinite(cand_t) & np.isfinite(flat)
        x = flat[mask]
        y = cand_t[mask]
        ss_res = float(np.sum((y - x) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")

        verdict = ("EXPOSURE-LEVEL ARTIFACT (R^2 > 0.95)" if np.isfinite(r2) and r2 > 0.95
                    else "not a flat rescale by this test")
        print(f"  {mname}: cand notional={not_c:.3f} v4 notional={not_v4:.3f} c={c:.3f}  "
              f"vol cand={vol_c:.3f} vol v4={vol_v4:.3f}  "
              f"corr={corr:.4f}  R^2={r2:.4f}  refits={nref}  {verdict}")


# ------------------------------------------------------------------------ causality


def causality() -> None:
    """Step 6: by-hand two-opposite-tampers lookahead probe, at TWO truncation points after different refits.

    Same procedure as R-28/R-31/R-33/R-37/R-38/R-40: bars strictly after a
    cut are multiplied by 3 in one copy, divided by 3 in another; every
    decision at or before the cut must be bit-identical. This is the
    critical check for this branch specifically -- the refit loop reads a
    trailing window at each schedule point, and a subtle bug there (e.g.
    slicing to ``j+1`` instead of ``j``, or reusing a full-series rolling
    statistic) is exactly the "i+1 peek" failure mode ROUTINE.md warns
    about. Run at TWO different truncation points, each chosen to fall
    strictly after at least one refit has already happened for the primary
    schedule (refit_bars = 365*288 = 105,120), so the probe actually
    exercises the refit arithmetic, not just the unchanged vote/hysteresis
    code paths already probed by prior rounds.
    """
    kw = SCHEDULES[PRIMARY]
    refit_bars = int(kw["refit_days"] * BARS_PER_DAY)
    lookback_bars = int(kw["lookback_days"] * BARS_PER_DAY)
    first_refit_at = lookback_bars  # position of the first refit, within the frame

    pre_2023 = DF.loc[:"2022-12-31"]
    df = pre_2023.copy()
    n = len(df)

    # Two truncation points, both strictly after >=1 refit has happened and
    # falling after a DIFFERENT number of refits from each other: one
    # shortly after the second refit event, one after the fourth.
    cuts = [first_refit_at + 1 * refit_bars + 5_000, first_refit_at + 3 * refit_bars + 5_000]
    cuts = [c for c in cuts if c < n - 60_000]
    assert len(cuts) == 2, f"pre-2023 series too short for two post-refit truncation points: {n:,} bars"

    overall_ok = True
    for cut in cuts:
        bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]
        up, down = df.iloc[:cut + 50_000].copy(), df.iloc[:cut + 50_000].copy()
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

        def prepared(frame):
            return KellyRegimeV11WalkforwardAdaptive(**kw).prepare(frame.copy())

        pa = prepared(up)
        pb = prepared(down)
        ok = True
        n_refits_before_cut = int(np.sum(
            np.arange(lookback_bars, cut, refit_bars) < cut)) if lookback_bars < cut else 0
        print(f"\n--- truncation at bar {cut:,} of {len(df):,} "
              f"({n_refits_before_cut} refit(s) of the primary schedule fall before this cut) ---")
        for col in ("target", "_target_vol_active", "_max_leverage_active", "_frac"):
            a = pa[col].to_numpy(dtype=float)[:cut]
            b = pb[col].to_numpy(dtype=float)[:cut]
            worst = float(np.nanmax(np.abs(a - b)))
            good = worst < 1e-9
            ok &= good
            print(f"  column={col:24s} max |difference| before the cut = {worst:.3e}  "
                  f"{'PASS' if good else 'FAIL'}")

        from tradebot.broker import PaperBroker
        from tradebot.orders import Order

        def decisions(frame):
            s = KellyRegimeV11WalkforwardAdaptive(**kw)
            prep = s.prepare(frame.copy())
            broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
            broker.execute(Order(target=0.1), prep.index[0], float(prep["open"].iloc[0]))
            out = []
            for i in bars:
                ctx = Context(prep, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down)) if oa != ob]
        ok &= not bad
        print(f"  orders {'match' if not bad else f'DIFFER at bars {bad}'} at the probe bars")

        a = run_backtest(KellyRegimeV11WalkforwardAdaptive(**kw), up.iloc[:cut + 1], FUTURES,
                          1_000.0, data_label=LABEL)
        b = run_backtest(KellyRegimeV11WalkforwardAdaptive(**kw), down.iloc[:cut + 1], FUTURES,
                          1_000.0, data_label=LABEL)
        worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
        ok &= worst_eq < 1e-6
        print(f"  max |equity difference| before the cut = {worst_eq:.3e}  "
              f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")

        overall_ok &= ok
        print(f"  truncation at {cut:,}: {'PASS' if ok else 'FAIL'}")

    print(f"\ntwo truncation points tested: {cuts}")
    print(f"overall causality probe: {'PASS - no decision at or before either cut moves' if overall_ok else 'FAIL'}")


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Step 7: pre-registered falsification -- does every schedule hold on ETH?

    Same venue (Bitfinex), same window as R-17/R-28/R-31/R-33/R-37/R-38/R-40,
    both spot and 5x futures, every schedule vs shipped v4 defaults as the
    control, on both the BTC control run and the ETH test run of the
    identical pipeline -- this is whole-file, pre-2020 data, safe under
    this session's rule. Falsification rule (fixed before running): if a
    candidate is not at least comparable to v4 on ETH, or is visibly worse
    on ETH than on the BTC control run through the identical code, this
    direction fails. Reports the actual refit count achieved on each short
    falsification file explicitly, since these files are far shorter than
    the main dataset.
    """
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            m_v4, vol_v4, not_v4, res_v4, _ = measure(get_strategy(INCUMBENT), None, None,
                                                       df=df, market=market)
            line(f"    {INCUMBENT} (control)", m_v4, vol_v4, not_v4, None)
            for label, kw in SCHEDULES.items():
                cand = KellyRegimeV11WalkforwardAdaptive(**kw)
                m_c, vol_c, not_c, res_c, nref = measure(cand, None, None, df=df, market=market)
                line(f"    v11_walkforward[{label}]", m_c, vol_c, not_c, nref)


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "sweep":
        sweep()
    elif choice == "select":
        select()
    elif choice == "artifact":
        exposure_artifact_check()
    elif choice == "causality":
        causality()
    elif choice == "eth":
        eth()
    else:
        print("usage: python experiments/kelly_regime_v11_walkforward_adaptive.py "
              "[sweep|select|artifact|causality|eth]")
