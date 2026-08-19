#!/usr/bin/env python
"""Ensemble-disagreement uncertainty shrink on top of a bagged anchor-ladder vote (SIZE/ERR axis).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea
--------
R-07 swept 9 anchor-ladder base periods in the 18-28 day range and found
that whole region is a validated PLATEAU (Sharpe spread 1.52-1.60 sits
inside the +/-0.2 noise floor). ``kelly_regime_v4`` ships with exactly one
point on that plateau (20/40/80) and treats it as certain -- there is no
mechanism anywhere in the signal path that ever registers when the regime
call itself is uncertain. Separately, the parameter-uncertainty literature
(Baker & McHale 2013, Decision Analysis 10(3), "Optimal Betting Under
Parameter Uncertainty: Improving the Kelly Criterion"; Sukhov 2025, SSRN,
"Bayesian Kelly Criterion with Parameter Uncertainty") shows the Kelly bet
should be shrunk continuously as the estimate of the edge becomes less
certain, rather than trusted at face value. This project's own ledger
already cites the same family (MacLean, Thorp & Ziemba 2010) for why
exposure should be a FRACTION of full Kelly at all.

Mechanism, one sentence: use real-time DISAGREEMENT across a small, fixed
(not fitted) ensemble of anchor-ladders spanning R-07's validated 18-28 day
plateau as a causal, continuously-updating estimate of how uncertain the
current regime call is, and shrink v4's exposure by that uncertainty
(Baker & McHale style), instead of trusting one frozen ladder's verdict
unconditionally.

Constraint attacked: ERR (no error control anywhere in the signal path).
This is the first attempt in this project's history to turn "the anchor
choice sits on a plateau, not a peak" (R-07) into a RUNTIME signal, rather
than leaving it as a static design choice made once and never revisited.

Not a duplicate of
-------------------
- R-06/R-07 (single-ladder sweep only, no disagreement signal -- this file
  is the first to read the plateau's WIDTH as a live signal rather than
  just picking one point on it).
- ``kelly_regime_v2`` (its ``vote_gamma`` convexity shrinks based on
  disagreement among the 3 anchors WITHIN one fixed ladder -- a fixed,
  non-adaptive functional form, always the same convex curve regardless of
  what is actually going on. This experiment computes disagreement ACROSS
  several different ladders/timescale choices, a different axis, and
  derives the shrink strength from an explicit parameter-uncertainty
  formula rather than an arbitrary exponent).
- R-34 (``harsanyi_crowd``'s Bayesian bull/bear/chop TYPE posterior -- a
  completely different signal source, a discrete hidden-state belief, not
  ensemble disagreement across timescale choices).
- R-37 (per-vote-state Kelly fraction replacing v4's global ``target_vol``)
  and R-38 (risk-constrained drawdown-probability cap, CRRA fraction) --
  both modify the sizing/vol-target formula using return moments (mu,
  sigma^2). This file leaves v4's vol-targeting ``scale[t]`` completely
  untouched; it only touches the vote/frac axis, using ensemble
  disagreement rather than return moments as its uncertainty signal.
- The companion "conservative" branch of this same round,
  ``kelly_regime_v8_ladder_bag.py`` (not read, not modified here): that
  branch computes only the PLAIN AVERAGE of the same ladder ensemble's
  votes, with no disagreement-based shrinkage at all. This branch is a
  strict superset of that mechanism: start from the same bagged/averaged
  vote, then ADD the uncertainty-shrink term on top, so that
  ``kappa=0`` (no shrink) collapses exactly to the conservative branch's
  mechanism -- verified numerically below.

Mechanism, precisely
---------------------
1. Same fixed-in-advance ensemble as the conservative branch: doubling
   ladders with base days {18, 20, 22, 24, 26, 28} (6 ladders), each
   ``(h, 2h, 4h)``, each with its own latched vote ``frac_k[t]`` computed
   exactly as ``kelly_regime.py`` does (copied verbatim per ladder).
2. ``frac_bagged[t] = mean_k(frac_k[t])``, exactly as the conservative
   branch.
3. NEW: a causal disagreement measure. Each ladder's OWN combined vote
   (mean of its 3 latched anchor votes) is itself in {0, 1/3, 2/3, 1};
   round it to the ladder's binary bull/bear call
   ``vote_k[t] = 1.0 if frac_k[t] >= 0.5 else 0.0`` and define
   ``disagree[t] = std_k(vote_k[t])`` -- the population std across the 6
   ladders' binary votes at time t. This is exactly 0 when all 6 ladders
   agree and near its maximum (~0.5) when the ensemble is split. Causal by
   construction: each ``vote_k[t]`` only uses information at or before t
   (it is built from the same latched, ``ffill``-only construction as
   every vote in this file).
4. Baker-McHale-style shrinkage:
   ``shrink[t] = 1 / (1 + kappa * disagree[t]**2)``, bounded to
   ``(shrink_floor, 1]`` via ``max(shrink[t], shrink_floor)`` so the shrink
   can attenuate but never fully zero out exposure from disagreement
   alone -- mirroring the parameter-uncertainty literature's result that
   the Kelly fraction shrinks continuously with estimation variance rather
   than switching off. ``kappa`` is the one new free hyperparameter (shrink
   strength); ``kappa=0`` reduces exactly to the conservative branch
   (verified numerically in ``kappa_zero_check()`` below).
5. ``frac_final[t] = frac_bagged[t] * shrink[t]``.
6. Feed ``frac_final[t]`` into v3/v4's UNCHANGED conditional-volatility-
   target scale (copied verbatim from ``kelly_regime_v3.py``).
7. ``target[t] = frac_final[t] * scale[t]``, debounced by the same 10%
   deadband v3/v4 use, applied once to the final combined signal.

Causal construction -- read this before the code
---------------------------------------------------
Follows the same discipline as ``kelly_regime_v7_ddcap.py``: no expanding
statistic may see its own future.

1. Each ladder's 3 anchors (h, 2h, 4h) are simple rolling means
   (``close.rolling(window).mean()``) -- backward-looking windows only.
2. Each anchor's vote is the latched ``np.where(...).ffill().fillna(0.0)``
   idiom used verbatim from ``kelly_regime.py``: bar t's vote depends only
   on ``close[<=t]`` and the vote's own past (the ffill), never on any bar
   after t.
3. ``frac_k[t]`` (per-ladder combined vote) and ``vote_k[t]`` (its binary
   round) are therefore both causal at bar t by construction, being pure
   pointwise functions of already-causal per-anchor votes.
4. ``disagree[t] = std_k(vote_k[t])`` is a cross-sectional (across the 6
   ladders, not across time) standard deviation computed independently at
   each bar t from the 6 already-causal ``vote_k[t]`` values at that same
   bar -- it is NOT an expanding or rolling statistic over time, so there
   is no window to leak across, and no full-series fit (no global mean,
   std or quantile taken over the whole series and applied to early rows).
5. ``shrink[t]`` and ``frac_final[t]`` are pointwise functions of
   ``disagree[t]`` and ``frac_bagged[t]``, both already causal at t.
6. v3/v4's conditional-vol-target ``scale[t]`` is copied verbatim,
   unchanged, from ``kelly_regime_v3.py`` (its own EWM/shift(1) causal
   construction, unmodified here).
7. The final deadband loop is a strictly causal state machine identical to
   v3/v4's own (``pos`` only updates from ``target[i]`` computed at or
   before bar i).

Pre-registered failure modes (named before any code ran)
-----------------------------------------------------------
(a) ``disagree[t]`` is nonzero only for a handful of bars around every
    latch-flip (the 6 ensemble ladders flip at similar but not identical
    times, since 18-28 days is a narrow band) and is otherwise near-zero --
    meaning the shrink term is economically negligible, adding turnover
    without changing exposure meaningfully.
(b) Whatever improvement appears is an exposure-level artifact: regress
    the candidate's ``target`` series against a mean-notional-matched flat
    rescale of v4's own ``target`` series and report R^2 (methodology
    copied from ``kelly_regime_v7_ddcap.py`` exactly) -- R^2 > 0.95 is this
    project's standing threshold for "just a rescale".
(c) The shrink term just recreates ``kelly_regime_v2``'s already-tested
    (and not-promoted) convex vote response through a different formula --
    checked explicitly by reporting the correlation between this
    candidate's ``target`` series and ``kelly_regime_v2``'s (shipped
    defaults) on the same period; a correlation near 1.0 means this is a
    re-derivation of an already-negative result.
(d) Fails the ETH falsification test, the same failure mode every
    SIZE-axis attempt since R-34 has hit.

Report which happened, precisely.

Usage
-----
    python experiments/kelly_regime_v8_uncertainty_shrink.py sweep       # step 3
    python experiments/kelly_regime_v8_uncertainty_shrink.py select      # step 5
    python experiments/kelly_regime_v8_uncertainty_shrink.py artifact    # exposure-artifact check
    python experiments/kelly_regime_v8_uncertainty_shrink.py v2corr      # correlation vs kelly_regime_v2
    python experiments/kelly_regime_v8_uncertainty_shrink.py kappazero   # kappa=0 numerical-equivalence check
    python experiments/kelly_regime_v8_uncertainty_shrink.py causality   # step 6
    python experiments/kelly_regime_v8_uncertainty_shrink.py eth         # step 7
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
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

LADDER_BASE_DAYS = (18, 20, 22, 24, 26, 28)  # fixed in advance, R-07's validated plateau


# --------------------------------------------------------------------- strategy


def _ladder_votes(close: pd.Series, base_days: int, band: float) -> tuple[np.ndarray, np.ndarray]:
    """One doubling ladder (h, 2h, 4h): returns (frac_k[t], vote_k[t] binary round)."""
    horizons = (base_days, base_days * 2, base_days * 4)
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes.append(v.ffill().fillna(0.0))
    frac_k = (sum(votes) / len(votes)).to_numpy()
    vote_k = (frac_k >= 0.5).astype(float)
    return frac_k, vote_k


class KellyRegimeV8UncertaintyShrink(Strategy):
    """Bagged 6-ladder vote (R-07 plateau), shrunk by causal cross-ladder disagreement.

    ``frac_bagged = mean_k(frac_k)`` over 6 fixed doubling ladders spanning
    R-07's 18-28 day validated plateau; ``disagree = std_k(vote_k)`` across
    the same 6 ladders' binary latched votes; ``shrink = max(shrink_floor,
    1/(1+kappa*disagree**2))``; ``frac_final = frac_bagged * shrink`` feeds
    v3/v4's unchanged conditional-vol-target scale. ``kappa=0`` collapses
    this exactly to the plain bagged-vote (conservative-branch) mechanism.
    """

    name = "kelly_regime_v8_uncertainty_shrink"

    def __init__(self, base_days: tuple[int, ...] = LADDER_BASE_DAYS, band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85,
                 kappa: float = 8.0, shrink_floor: float = 0.3) -> None:
        self.base_days = base_days
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        self.kappa = float(kappa)
        self.shrink_floor = float(shrink_floor)
        # 28-day base ladder needs 4*28=112 days of anchor history; keep the
        # same generous margin convention v4 uses for its own 80-day anchor.
        self.warmup = max(base_days) * 4 * BARS_PER_DAY + 10

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        idx = df.index
        n = len(df)
        r = np.log(close).diff()

        # ---- 6-ladder ensemble, each computed exactly as kelly_regime.py does ----
        frac_ks = []
        vote_ks = []
        for base in self.base_days:
            frac_k, vote_k = _ladder_votes(close, base, self.band)
            frac_ks.append(frac_k)
            vote_ks.append(vote_k)
        frac_mat = np.vstack(frac_ks)   # (6, n)
        vote_mat = np.vstack(vote_ks)   # (6, n)

        frac_bagged = frac_mat.mean(axis=0)
        disagree = vote_mat.std(axis=0)  # population std (ddof=0) across the 6 ladders, per bar

        # ---- Baker-McHale-style shrink, causal & pointwise in disagree[t] ----
        shrink = 1.0 / (1.0 + self.kappa * disagree ** 2)
        shrink = np.maximum(shrink, self.shrink_floor)
        frac_final = frac_bagged * shrink

        # ---- v3/v4's own conditional-vol-target scale, verbatim (kelly_regime_v3.py) ----
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                    min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        scale = np.zeros(n)
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
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
            desired = frac_final[i] * scale[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["_frac_bagged"] = frac_bagged
        df["_disagree"] = disagree
        df["_shrink"] = shrink
        df["_frac_final"] = frac_final
        df["_scale"] = scale
        _ = idx  # kept for symmetry with the v7 template; unused directly
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures


# ------------------------------------------------------------------------ harness

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")

INCUMBENT = "kelly_regime_v4"
V2 = "kelly_regime_v2"

N_EVALUATED = 0  # distinct configurations searched in step 3, for deflated Sharpe

OUT = ROOT / "reports" / "kelly_regime_v8_uncertainty_shrink"


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
    """One backtest -> (metrics, realized vol, mean notional, result)."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                         start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    return m, realized_vol(result.equity), mean_notional(result), result


def line(tag, m, vol, notional, result):
    print(f"  {tag:44s} final=${m.final_balance:>11,.0f} "
          f"vol={vol:5.3f} notional={notional:5.3f} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# --------------------------------------------------------------------------- step 3

KAPPA_GRID = (0.0, 2.0, 8.0, 20.0)
FLOOR_GRID = (0.2, 0.4)


def grid_configs():
    for kappa in KAPPA_GRID:
        for floor in FLOOR_GRID:
            yield dict(kappa=kappa, shrink_floor=floor)


def sweep() -> pd.DataFrame:
    """Step 3: sweep the (kappa, shrink_floor) grid on inner-train only, spot market."""
    rows = []
    t0 = time.time()
    for cfg in grid_configs():
        strat = KellyRegimeV8UncertaintyShrink(**cfg)
        m, vol, notional, res = measure(strat, *TRAIN, market=SPOT, count=True)
        mean_disagree = float(res.df["_disagree"].mean()) if "_disagree" in res.df else float("nan")
        frac_disagree_nonzero = (float((res.df["_disagree"] > 1e-9).mean())
                                  if "_disagree" in res.df else float("nan"))
        mean_shrink = float(res.df["_shrink"].mean()) if "_shrink" in res.df else float("nan")
        rows.append({**cfg, "final": m.final_balance, "vol": vol,
                     "notional": notional, "max_dd": m.max_drawdown_pct,
                     "sharpe": m.sharpe, "trades": m.num_trades,
                     "fees": m.fees_paid, "liquidated": m.liquidated,
                     "mean_disagree": mean_disagree,
                     "frac_bars_disagree_nonzero": frac_disagree_nonzero,
                     "mean_shrink": mean_shrink})
        print(f"[{N_EVALUATED:>3d}] kappa={cfg['kappa']:6.2f} floor={cfg['shrink_floor']:.2f}  "
              f"final=${m.final_balance:>10,.0f} DD={m.max_drawdown_pct:>5.1f}% "
              f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
              f"mean_disagree={mean_disagree:.4f} nonzero%={frac_disagree_nonzero:5.1%} "
              f"mean_shrink={mean_shrink:.4f} [{time.time() - t0:.0f}s]")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "sweep_inner_train.csv", index=False)
    print(f"\nconfigurations evaluated (step 3): {N_EVALUATED}")
    print(f"written: {OUT / 'sweep_inner_train.csv'}")
    return out


# --------------------------------------------------------------------------- step 5


def select(candidates: list[dict] | None = None) -> None:
    """Step 5: score candidates on inner-validation, both markets, plateau view.

    Controls reported alongside each candidate: kelly_regime_v4 (shipped
    defaults) and this file's own kappa=0 reduction (the conservative
    branch's mechanism, computed here from this file's own code, not by
    reading the companion branch's file).
    """
    if candidates is None:
        candidates = list(grid_configs())
    rows = []

    # controls, once per market
    ctrl = {}
    for mname, market in MARKETS:
        m_v4, vol_v4, not_v4, res_v4 = measure(get_strategy(INCUMBENT), *VALID, market=market)
        kz = KellyRegimeV8UncertaintyShrink(kappa=0.0, shrink_floor=0.2)
        m_kz, vol_kz, not_kz, res_kz = measure(kz, *VALID, market=market)
        ctrl[mname] = dict(v4=(m_v4, vol_v4, not_v4), kappa0=(m_kz, vol_kz, not_kz))
        print(f"[control {mname}] {INCUMBENT}: final=${m_v4.final_balance:,.0f} "
              f"DD={m_v4.max_drawdown_pct:.1f}% sharpe={m_v4.sharpe:.2f}   "
              f"kappa=0 reduction: final=${m_kz.final_balance:,.0f} "
              f"DD={m_kz.max_drawdown_pct:.1f}% sharpe={m_kz.sharpe:.2f}")

    for cfg in candidates:
        strat_spot = KellyRegimeV8UncertaintyShrink(**cfg)
        strat_fut = KellyRegimeV8UncertaintyShrink(**cfg)
        m_s, vol_s, not_s, res_s = measure(strat_spot, *VALID, market=SPOT)
        m_f, vol_f, not_f, res_f = measure(strat_fut, *VALID, market=FUTURES)
        rows.append({**cfg,
                     "spot_final": m_s.final_balance, "spot_dd": m_s.max_drawdown_pct,
                     "spot_sharpe": m_s.sharpe, "spot_trades": m_s.num_trades,
                     "spot_vol": vol_s, "spot_notional": not_s,
                     "fut_final": m_f.final_balance, "fut_dd": m_f.max_drawdown_pct,
                     "fut_sharpe": m_f.sharpe, "fut_trades": m_f.num_trades,
                     "fut_vol": vol_f, "fut_notional": not_f,
                     "v4_spot_final": ctrl["spot"]["v4"][0].final_balance,
                     "v4_spot_sharpe": ctrl["spot"]["v4"][0].sharpe,
                     "v4_fut_final": ctrl["futures"]["v4"][0].final_balance,
                     "v4_fut_sharpe": ctrl["futures"]["v4"][0].sharpe,
                     "kappa0_spot_final": ctrl["spot"]["kappa0"][0].final_balance,
                     "kappa0_spot_sharpe": ctrl["spot"]["kappa0"][0].sharpe,
                     "kappa0_fut_final": ctrl["futures"]["kappa0"][0].final_balance,
                     "kappa0_fut_sharpe": ctrl["futures"]["kappa0"][0].sharpe})
        print(f"kappa={cfg['kappa']:6.2f} floor={cfg['shrink_floor']:.2f}  "
              f"spot: ${m_s.final_balance:>9,.0f} DD{m_s.max_drawdown_pct:>5.1f}% "
              f"sh{m_s.sharpe:>5.2f}  fut: ${m_f.final_balance:>9,.0f} "
              f"DD{m_f.max_drawdown_pct:>5.1f}% sh{m_f.sharpe:>5.2f}")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "select_inner_validation.csv", index=False)
    print(f"\nwritten: {OUT / 'select_inner_validation.csv'}")


# --------------------------------------------------------------------------- kappa=0 check


def kappa_zero_check() -> None:
    """Verify kappa=0 collapses exactly to the plain bagged-vote (conservative-branch) mechanism.

    With kappa=0, shrink[t] = 1/(1+0*disagree**2) = 1.0 everywhere, before
    the floor clamp (1.0 > any shrink_floor <= 1, so the floor never binds
    either) -- so frac_final == frac_bagged identically. Checked two ways:
    directly on the prepared columns, and by confirming the result is
    identical for both tested shrink_floor values (0.2 and 0.4), since the
    floor must be irrelevant when shrink is already 1.0 everywhere.
    """
    strat_02 = KellyRegimeV8UncertaintyShrink(kappa=0.0, shrink_floor=0.2)
    strat_04 = KellyRegimeV8UncertaintyShrink(kappa=0.0, shrink_floor=0.4)
    frame = DF.loc[:"2020-12-31"].iloc[-200_000:].copy()
    p02 = strat_02.prepare(frame.copy())
    p04 = strat_04.prepare(frame.copy())

    diff_frac = float(np.max(np.abs(p02["_frac_final"].to_numpy() - p02["_frac_bagged"].to_numpy())))
    diff_shrink = float(np.max(np.abs(p02["_shrink"].to_numpy() - 1.0)))
    diff_across_floor = float(np.max(np.abs(p02["target"].to_numpy() - p04["target"].to_numpy())))

    print(f"max |frac_final - frac_bagged| at kappa=0, floor=0.2: {diff_frac:.3e}")
    print(f"max |shrink - 1.0| at kappa=0, floor=0.2: {diff_shrink:.3e}")
    print(f"max |target(floor=0.2) - target(floor=0.4)| at kappa=0: {diff_across_floor:.3e}")
    ok = diff_frac < 1e-12 and diff_shrink < 1e-12 and diff_across_floor < 1e-12
    print(f"kappa=0 numerical-equivalence check: {'PASS' if ok else 'FAIL'}")


# --------------------------------------------------------------------------- artifact


def exposure_artifact_check(candidate_kwargs: dict) -> None:
    """Mandatory exposure-artifact check (ROUTINE.md standing rule, sharpened by R-33).

    Build a "flat-rescaled v4" comparator: v4's own unchanged target,
    multiplied by a single constant c chosen so its mean notional matches
    the candidate's mean notional over the SAME period. Report R^2 of the
    candidate's target series against that flat rescale, on inner-
    validation, both markets. R^2 > 0.95 (R-34's own 0.997 threshold) means
    "this is the standard exposure-level artifact", reported honestly as
    such, not as a win. Methodology copied verbatim from
    kelly_regime_v7_ddcap.py.
    """
    print("\nexposure-artifact check (inner-validation, mean-notional-matched flat rescale of v4):")
    for mname, market in MARKETS:
        cand = KellyRegimeV8UncertaintyShrink(**candidate_kwargs)
        m_c, vol_c, not_c, res_c = measure(cand, *VALID, market=market)
        v4 = get_strategy(INCUMBENT)
        m_v4, vol_v4, not_v4, res_v4 = measure(v4, *VALID, market=market)

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
        mean_abs_diff = float(np.mean(np.abs(y - x)))
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")

        print(f"  {mname}: candidate mean notional={not_c:.3f}  v4 mean notional={not_v4:.3f}  "
              f"c={c:.3f}")
        print(f"    corr(candidate, c*v4) = {corr:.4f}")
        print(f"    R^2 of candidate ~ c*v4 = {r2:.4f}")
        print(f"    mean|candidate - c*v4| = {mean_abs_diff:.4f}")
        verdict = ("EXPOSURE-LEVEL ARTIFACT (R^2 > 0.95)" if np.isfinite(r2) and r2 > 0.95
                    else "not a flat rescale by this test")
        print(f"    verdict: {verdict}")


# --------------------------------------------------------------------------- v2 correlation


def v2_correlation_check(candidate_kwargs: dict) -> None:
    """Pre-registered failure mode (c): does this just re-derive kelly_regime_v2?

    Report corr(candidate target, kelly_regime_v2 shipped-default target) on
    inner-validation, both markets. Near 1.0 means the disagreement-shrink
    term is a re-derivation of v2's already-tested convex vote response.
    """
    print("\ncorrelation vs kelly_regime_v2 (shipped defaults), inner-validation:")
    for mname, market in MARKETS:
        cand = KellyRegimeV8UncertaintyShrink(**candidate_kwargs)
        m_c, vol_c, not_c, res_c = measure(cand, *VALID, market=market)
        v2 = get_strategy(V2)
        m_v2, vol_v2, not_v2, res_v2 = measure(v2, *VALID, market=market)

        cand_t = res_c.df["target"].to_numpy(dtype=float)
        v2_t = res_v2.df["target"].reindex(res_c.df.index).to_numpy(dtype=float)
        mask = np.isfinite(cand_t) & np.isfinite(v2_t)
        corr = float(np.corrcoef(cand_t[mask], v2_t[mask])[0, 1]) if mask.sum() > 1 else float("nan")
        print(f"  {mname}: corr(candidate, kelly_regime_v2) = {corr:.4f}  "
              f"(candidate mean notional={not_c:.3f}, v2 mean notional={not_v2:.3f})")


# ------------------------------------------------------------------------ causality


def causality() -> None:
    """Step 6: by-hand two-opposite-tampers lookahead probe.

    Experiments get no CI protection (test_causality_strict.py parametrizes
    over the registry only). Same procedure as R-28/R-31/R-33/R-37/R-38:
    bars after a cut are multiplied by 3 in one copy, divided by 3 in
    another; every decision at or before the cut must be bit-identical.
    Particularly important here since ``_disagree`` is a NEW cross-sectional
    signal built from 6 independently-latched ladders -- exactly the kind of
    place a subtle indexing bug could leak a later ladder flip into an
    earlier bar's disagreement reading. The check below compares the
    prepared columns directly, not just orders, for that reason.
    """
    # Restricted to strictly pre-2023 bars: this session's brief forbids
    # reading any bar dated 2023-01-01 or later for ANY purpose, so the
    # probe is confined to the inner splits rather than the dataset tail.
    pre_2023 = DF.loc[:"2022-12-31"]
    df = pre_2023.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    strat_kwargs = dict(kappa=8.0, shrink_floor=0.2)

    def prepared(frame):
        return KellyRegimeV8UncertaintyShrink(**strat_kwargs).prepare(frame.copy())

    pa = prepared(up)
    pb = prepared(down)
    ok = True
    for col in ("target", "_frac_bagged", "_disagree", "_shrink", "_frac_final", "_scale"):
        a = pa[col].to_numpy(dtype=float)[:cut]
        b = pb[col].to_numpy(dtype=float)[:cut]
        worst = float(np.nanmax(np.abs(a - b)))
        good = worst < 1e-9
        ok &= good
        print(f"  column={col:16s} max |difference| before the cut = {worst:.3e}  "
              f"{'PASS' if good else 'FAIL'}")

    from tradebot.broker import PaperBroker
    from tradebot.orders import Order

    def decisions(frame):
        s = KellyRegimeV8UncertaintyShrink(**strat_kwargs)
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

    a = run_backtest(KellyRegimeV8UncertaintyShrink(**strat_kwargs), up.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    b = run_backtest(KellyRegimeV8UncertaintyShrink(**strat_kwargs), down.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-6
    print(f"  max |equity difference| before the cut = {worst_eq:.3e}  "
          f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")

    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# ------------------------------------------------------------------------------ eth


def eth(candidate_kwargs: dict) -> None:
    """Step 7: pre-registered falsification -- does the candidate hold on ETH?

    Same venue (Bitfinex), same window as R-17/R-28/R-31/R-33/R-37/R-38,
    both spot and 5x futures, candidate vs shipped v4 defaults as the
    control. Falsification rule (fixed before running): if the candidate is
    not at least comparable to v4 on ETH, or is visibly worse on ETH than
    the BTC control run through the identical code, this direction fails.
    """
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            m_v4, vol_v4, not_v4, res_v4 = measure(get_strategy(INCUMBENT), None, None,
                                                    df=df, market=market)
            line(f"    {INCUMBENT} (control)", m_v4, vol_v4, not_v4, res_v4)
            cand = KellyRegimeV8UncertaintyShrink(**candidate_kwargs)
            m_c, vol_c, not_c, res_c = measure(cand, None, None, df=df, market=market)
            line("    kelly_regime_v8_uncertainty_shrink (candidate)", m_c, vol_c, not_c, res_c)


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    default_cand = dict(kappa=8.0, shrink_floor=0.2)  # mid-grid, most representative of a moderate shrink
    if choice == "sweep":
        sweep()
    elif choice == "select":
        select()
    elif choice == "artifact":
        exposure_artifact_check(default_cand)
    elif choice == "v2corr":
        v2_correlation_check(default_cand)
    elif choice == "kappazero":
        kappa_zero_check()
    elif choice == "causality":
        causality()
    elif choice == "eth":
        eth(default_cand)
    else:
        print("usage: python experiments/kelly_regime_v8_uncertainty_shrink.py "
              "[sweep|select|artifact|v2corr|kappazero|causality|eth]")
