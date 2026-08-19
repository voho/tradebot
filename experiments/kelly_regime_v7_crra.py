#!/usr/bin/env python
"""CRRA/Merton Kelly sizing under an explicit drawdown-risk tolerance (SIZE axis, v7 novel branch).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea
--------
``kelly_regime_v4`` answers "how much should I hold?" with
``scale = min(target_vol / realized_vol, max_leverage)`` — two hand-picked
constants (``target_vol=0.55``, ``max_leverage=2.0``) with no probabilistic
interpretation. Busseti, Ryu & Boyd (2016, "Risk-Constrained Kelly
Gambling", Journal of Investing 25(3) / arXiv:1603.06183) show that the
Kelly gambling problem with an explicit probabilistic drawdown constraint
``Prob(min future wealth < alpha) < beta`` reduces, via a convex bound, to
maximizing a CRRA/isoelastic-utility criterion with risk-aversion parameter
``lambda = ln(beta) / ln(alpha)`` — a max-acceptable-drawdown ``1-alpha`` at
confidence ``1-beta`` maps directly onto a single risk-aversion number.
Combined with the classical Merton (1969, 1971) result that the CRRA-optimal
bet fraction under a drift-``mu``/variance-``sigma**2`` return process is
``f* = mu / (lambda * sigma**2)`` (the plain Kelly fraction ``mu/sigma**2``
is the ``lambda=1`` case), this gives a formal, probability-calibrated
sizing FORMULA rather than a hand-tuned cap.

Mechanism, one sentence: replace v4's entire sizing engine (conditional
vol-targeting, ``target_vol``, ``max_leverage``) with
``f*[t] = mu[t] / (lambda * sigma[t]**2)``, where ``mu[t]``/``sigma[t]**2``
are causally estimated per-bar from an EWM of log returns and ``lambda`` is
FIXED IN ADVANCE from a chosen ``(alpha, beta)`` drawdown-risk tolerance,
keeping v4's bull/bear/chop vote (``frac``) unchanged as a hard
multiplicative gate: ``target[t] = frac[t] * clip(f*[t], 0, safety_cap)``.

Constraint attacked: SIZE (v4's own axis — how much to hold given a fixed
regime call, not a better regime call).

Not a duplicate of
-------------------
- L-01..L-04: ad hoc sizing constants (``target_vol``/``max_leverage``,
  vote-response exponents) vs. this, a formula DERIVED from an explicit
  probabilistic drawdown target with no free tuning knob beyond the target
  itself.
- R-28/R-31 (e-process anytime-valid hypothesis testing) gates the
  regime/direction signal — a different question. This file never touches
  the vote; it only replaces the size given a vote that already fired.
- R-34 (Bayesian posterior replacing the vote, both branches): also a
  DIRECTION-axis change dressed as sizing. This file keeps v4's vote
  verbatim.
- R-37 novel (``kelly_regime_v6_state_kelly.py``): estimates a SEPARATE
  mu_state/sigma_state**2 per one of v4's 4 discrete vote states, scaled by
  an arbitrary SEARCHED ``kelly_mult`` constant with no formal
  drawdown-probability calibration. This file estimates ONE continuous,
  non-state-conditional mu[t]/sigma[t]**2 pair updated every bar (the vote
  only gates the final position, it never selects which moment estimate to
  use), and its scaling constant ``lambda`` is DERIVED from a stated
  ``(alpha, beta)`` drawdown tolerance rather than searched for best
  backtest score. That derivation, not the state-conditioning, is the crux
  of the "formal error control" claim that makes this genuinely new rather
  than a retune of R-37's idea.

Causal construction for mu[t] / sigma[t]**2 — read this before the code
-------------------------------------------------------------------------
This is the single easiest place to leak the future into the past (R-21: a
one-day lookahead broadcast onto 5m bars was worth +2.1 Sharpe and passed a
truncation test; an ``i+1`` peek returned $3.7e23 with a green suite). The
construction here, precisely, so it can be checked without reading code:

1. ``r[t] = log(close[t] / close[t-1])``, the bar's own realized log
   return — known entirely at the close of bar t.
2. Build a TIME-HALFLIFE (not occurrence-count) exponentially weighted mean
   of ``r`` over the WHOLE series using pandas'
   ``ewm(halflife=<days>, times=df.index, min_periods=...)`` — every bar
   contributes, there is no per-state bucketing here (unlike R-37 novel):
   this is the one continuous mu[t]/sigma[t]**2 pair the module docstring
   above describes. ``mu_raw[t]`` at this point uses only ``r[<=t]``, so it
   is already causal for a decision made at bar t.
3. Variance the same way pandas' own ``var`` does NOT support the ``times``
   argument (checked empirically against this repo's pandas 3.0.5:
   ``.ewm(..., times=...).var()`` raises "var is not implemented with
   times") — so it is built by hand from two EWM means, mirroring R-37's
   own workaround: ``mu2_raw[t] = ewm(r**2, halflife, times=idx).mean()``,
   ``var_raw[t] = clip(mu2_raw[t] - mu_raw[t]**2, lower=0)``.
4. Both ``mu_raw`` and ``var_raw`` are shifted ONE MORE bar
   (``.shift(1)``), exactly mirroring ``kelly_regime.py``'s own
   ``vol = ewm(...).shift(1)`` convention — a deliberate one-bar safety
   margin on top of step 2's already-causal construction. Bar t's sizing
   decision (computed at the close of bar t, filled at the open of bar
   t+1) reads mu/var as of bar t-1, never bar t itself.
5. ``f*[t] = mu[t] / (lambda * var[t])`` where ``var[t] > 0``, else 0;
   floored at 0 (never short — mirrors v4's own logic, which also never
   shorts) and capped at ``safety_cap`` (a generous constant, default 5.0,
   used ONLY to prevent numerical blowup when ``var[t]`` is tiny — not a
   real constraint, unlike v4's ``max_leverage``, which is the whole
   sizing cap there).
6. ``target[t] = frac[t] * f*[t]`` (``frac[t]`` is v4's own vote, unchanged
   and already causal), then the same 10% deadband v4 uses.

Pre-registered failure modes (named before any code ran)
----------------------------------------------------------
(a) A drift-over-variance formula is much more sensitive to the (very
    noisy) mu estimate at 5-minute cadence than v4's volatility-only
    sizing is: expect wild position swings, high turnover, and possibly
    degenerate behavior (near-zero or pinned-at-safety-cap exposure) if
    mu's estimation noise dominates its signal. This is the failure mode
    R-34's novel branch found for a different, real, independent signal
    ("too noisy at its native cadence to pay 5-minute-bar trading costs").
    Checked explicitly below via trade counts, turnover, and the fraction
    of bars pinned at the safety cap or at zero.
(b) Whatever improvement appears is, once again, an exposure-level
    artifact (L-04/R-28/R-31/R-32/R-33/R-34 conservative): the winning
    config simply asks for more raw notional than v4's default. Checked
    explicitly below with the project's own R-34-style flat-rescale R**2
    diagnostic (R**2 > 0.95 against a flat-rescaled v4 series is reported
    honestly as the standard artifact, not as a win).
(c) R-37's novel branch found state-conditional mu workable in aggregate
    on inner-validation but overfit to one BTC window on ETH falsification
    — a shorter, more responsive halflife (this branch's grid is 7-60
    days vs R-37's 30-365) makes this MORE likely, not less, since it is
    itself the manipulation under test (a drift estimate needs a shorter
    window than a pure volatility estimate to be usable at all).

Usage
-----
    python experiments/kelly_regime_v7_crra.py sweep       # step 3
    python experiments/kelly_regime_v7_crra.py select      # step 5
    python experiments/kelly_regime_v7_crra.py artifact    # exposure-artifact check
    python experiments/kelly_regime_v7_crra.py causality   # step 6
    python experiments/kelly_regime_v7_crra.py eth         # step 7
"""

from __future__ import annotations

import math
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


# --------------------------------------------------------------------- strategy


class KellyRegimeV7Crra(Strategy):
    """v4's exact vote, sized by a CRRA/Merton fraction under a stated drawdown tolerance.

    Everything about the vote (horizons, band, latching hysteresis) is
    copied verbatim from ``kelly_regime_v4`` / ``kelly_regime_v3`` /
    ``kelly_regime.py``. Everything about the sizer is replaced: instead of
    ``scale = min(target_vol / realized_vol, max_leverage)``, the position
    is ``f* = mu / (lambda * sigma**2)`` — the classical Merton CRRA
    fraction — where ``lambda = ln(beta) / ln(alpha)`` is fixed from a
    stated max-acceptable-drawdown ``1-alpha`` at confidence ``1-beta``
    (Busseti, Ryu & Boyd 2016), and ``mu``/``sigma**2`` are one continuous,
    causally estimated pair (see module docstring), not per-state.
    """

    name = "kelly_regime_v7_crra"

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 deadband: float = 0.10, halflife_days: float = 15.0,
                 alpha: float = 0.6, beta: float = 0.05,
                 safety_cap: float = 5.0, min_periods: int | None = None) -> None:
        if not (0.0 < alpha < 1.0) or not (0.0 < beta < 1.0):
            raise ValueError("alpha and beta must be in (0, 1)")
        self.horizons = horizons
        self.band = band
        self.deadband = deadband
        self.halflife_days = float(halflife_days)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lam = math.log(self.beta) / math.log(self.alpha)  # Busseti-Ryu-Boyd (2016)
        self.safety_cap = float(safety_cap)
        self.min_periods = int(min_periods) if min_periods is not None else BARS_PER_DAY
        # Warmup scales with the half-life being tested, mirroring
        # kelly_regime_v6_state_kelly.py's reasoning: inner-validation has
        # pre-period history to draw on, inner-train does not (the dataset
        # starts exactly on 2017-01-01, so inner-train is unavoidably cold
        # regardless of this number — a limitation v4 itself shares, R-22).
        self.warmup = max(80 * BARS_PER_DAY + 10, int(3 * self.halflife_days * BARS_PER_DAY))

    def _vote(self, df: pd.DataFrame) -> np.ndarray:
        """v4's vote, verbatim (see kelly_regime.py / kelly_regime_v4.py)."""
        close = df["close"]
        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        return (sum(votes) / len(votes)).to_numpy()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        idx = df.index
        n = len(df)

        frac = self._vote(df)

        r = np.log(close).diff()
        halflife = pd.Timedelta(days=self.halflife_days)
        mu_raw = r.ewm(halflife=halflife, times=idx, min_periods=self.min_periods).mean()
        mu2_raw = (r ** 2).ewm(halflife=halflife, times=idx,
                                min_periods=self.min_periods).mean()
        var_raw = (mu2_raw - mu_raw ** 2).clip(lower=0.0)

        # One extra bar of lag on top of the already-causal EWM (see module
        # docstring step 4) — the same safety margin v4's own
        # `vol = ewm(...).shift(1)` uses.
        mu = mu_raw.shift(1).to_numpy()
        var = var_raw.shift(1).to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            f_star = np.where(var > 0, mu / (self.lam * var), np.nan)
        f_star = np.where(np.isfinite(f_star), f_star, 0.0)
        f_star = np.clip(f_star, 0.0, self.safety_cap)  # never short; cap only for blowup safety

        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = frac[i] * f_star[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["_frac"] = frac
        df["_mu"] = mu
        df["_var"] = var
        df["_f_star"] = f_star
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

N_EVALUATED = 0  # distinct configurations searched in step 3, for deflated Sharpe


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

ALPHA_GRID = (0.5, 0.6, 0.7, 0.8)
BETA_GRID = (0.05, 0.10)
HALFLIFE_GRID = (7.0, 15.0, 30.0, 60.0)
SAFETY_CAP_DEFAULT = 5.0


def grid_configs():
    for a in ALPHA_GRID:
        for b in BETA_GRID:
            for hl in HALFLIFE_GRID:
                yield dict(alpha=a, beta=b, halflife_days=hl, safety_cap=SAFETY_CAP_DEFAULT)


def sweep() -> pd.DataFrame:
    """Step 3: sweep the (alpha, beta, halflife) grid on inner-train only, spot market."""
    rows = []
    t0 = time.time()
    for cfg in grid_configs():
        strat = KellyRegimeV7Crra(**cfg)
        m, vol, notional, res = measure(strat, *TRAIN, market=SPOT, count=True)
        pinned_cap = float(np.mean(np.isclose(res.df["_f_star"].to_numpy(dtype=float),
                                               cfg["safety_cap"])))
        rows.append({**cfg, "lam": strat.lam, "final": m.final_balance, "vol": vol,
                     "notional": notional, "max_dd": m.max_drawdown_pct,
                     "sharpe": m.sharpe, "trades": m.num_trades,
                     "fees": m.fees_paid, "liquidated": m.liquidated,
                     "pinned_at_cap_frac": pinned_cap})
        print(f"[{N_EVALUATED:>3d}] alpha={cfg['alpha']:.2f} beta={cfg['beta']:.2f} "
              f"hl={cfg['halflife_days']:>4.0f}d lam={strat.lam:6.3f}  "
              f"final=${m.final_balance:>10,.0f} DD={m.max_drawdown_pct:>5.1f}% "
              f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
              f"pinned={pinned_cap:5.1%} [{time.time() - t0:.0f}s]")
    df = pd.DataFrame(rows)
    OUT = ROOT / "reports" / "kelly_regime_v7_crra"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "sweep_inner_train.csv", index=False)
    print(f"\nconfigurations evaluated (step 3): {N_EVALUATED}")
    print(f"written: {OUT / 'sweep_inner_train.csv'}")
    return df


# --------------------------------------------------------------------------- step 5


def select(candidates: list[dict] | None = None) -> None:
    """Step 5: score candidates on inner-validation, both markets, plateau view."""
    if candidates is None:
        candidates = list(grid_configs())
    rows = []
    for cfg in candidates:
        strat_spot = KellyRegimeV7Crra(**cfg)
        strat_fut = KellyRegimeV7Crra(**cfg)
        m_s, vol_s, not_s, res_s = measure(strat_spot, *VALID, market=SPOT)
        m_f, vol_f, not_f, res_f = measure(strat_fut, *VALID, market=FUTURES)
        rows.append({**cfg, "lam": strat_spot.lam,
                     "spot_final": m_s.final_balance, "spot_dd": m_s.max_drawdown_pct,
                     "spot_sharpe": m_s.sharpe, "spot_trades": m_s.num_trades,
                     "spot_vol": vol_s, "spot_notional": not_s,
                     "fut_final": m_f.final_balance, "fut_dd": m_f.max_drawdown_pct,
                     "fut_sharpe": m_f.sharpe, "fut_trades": m_f.num_trades,
                     "fut_vol": vol_f, "fut_notional": not_f})
        print(f"alpha={cfg['alpha']:.2f} beta={cfg['beta']:.2f} "
              f"hl={cfg['halflife_days']:>4.0f}d lam={strat_spot.lam:6.3f}  "
              f"spot: ${m_s.final_balance:>9,.0f} DD{m_s.max_drawdown_pct:>5.1f}% "
              f"sh{m_s.sharpe:>5.2f} tr{m_s.num_trades:>5d}  "
              f"fut: ${m_f.final_balance:>9,.0f} DD{m_f.max_drawdown_pct:>5.1f}% "
              f"sh{m_f.sharpe:>5.2f} tr{m_f.num_trades:>5d}")
    out = pd.DataFrame(rows)
    OUT = ROOT / "reports" / "kelly_regime_v7_crra"
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "select_inner_validation.csv", index=False)
    print(f"\nwritten: {OUT / 'select_inner_validation.csv'}")


def v4_baseline(start, end, market) -> None:
    m, vol, notional, res = measure(get_strategy(INCUMBENT), start, end, market=market)
    line(f"{INCUMBENT} (shipped defaults)", m, vol, notional, res)


# ------------------------------------------------------------------- exposure artifact


def build_target_series(strategy, start, end) -> pd.Series:
    """The strategy's own `target` column, causal, over [start, end], warmed from before it."""
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    prefix = min(lo, strategy.warmup)
    frame = strategy.prepare(DF.iloc[lo - prefix: hi].copy())
    return frame["target"].iloc[prefix:]


def artifact_check(candidate_kwargs: dict, start=VALID[0], end=VALID[1]) -> None:
    """Mandatory exposure-artifact check (this project's standing rule, R-33/R-34/R-37).

    Builds a "flat-rescaled v4": v4's own unchanged target series multiplied
    by a single constant chosen so its mean notional matches the
    candidate's mean notional over the same period, then reports the R**2
    between the candidate's exposure series and that flat-rescaled v4
    series. R**2 > 0.95 (R-34's own threshold, itself set from R-33's
    0.997) means: report this honestly as the standard exposure-level
    artifact, not as a win.
    """
    cand = KellyRegimeV7Crra(**candidate_kwargs)
    v4 = get_strategy(INCUMBENT)
    cand_series = build_target_series(cand, start, end)
    v4_series = build_target_series(v4, start, end)
    idx = cand_series.index.intersection(v4_series.index)
    c = cand_series.loc[idx].to_numpy(dtype=float)
    v = v4_series.loc[idx].to_numpy(dtype=float)

    mean_c = float(np.mean(np.abs(c)))
    mean_v = float(np.mean(np.abs(v)))
    mult = mean_c / mean_v if mean_v > 0 else float("nan")
    flat_rescaled = mult * v

    if np.std(c) > 0 and np.std(flat_rescaled) > 0:
        r = float(np.corrcoef(c, flat_rescaled)[0, 1])
        r2 = r ** 2
    else:
        r2 = float("nan")

    vol_c = float(np.std(np.diff(c)) * np.sqrt(BARS_PER_YEAR)) if len(c) > 2 else float("nan")

    print(f"  period {start} -> {end}, n={len(idx):,} bars")
    print(f"  candidate mean |notional| = {mean_c:.4f}   v4 mean |notional| = {mean_v:.4f}")
    print(f"  flat-rescale multiplier (matches mean notional) = {mult:.4f}")
    print(f"  R**2(candidate exposure, {mult:.4f} * v4 exposure) = {r2:.4f}")
    print(f"  candidate exposure-series realized vol proxy = {vol_c:.4f}")
    verdict = ("STANDARD EXPOSURE-LEVEL ARTIFACT (R**2 > 0.95)" if r2 > 0.95
               else "not the flat-rescale artifact by this test (R**2 <= 0.95)")
    print(f"  verdict: {verdict}")


# ------------------------------------------------------------------------ causality


def causality() -> None:
    """Step 6: by-hand two-opposite-tampers lookahead probe.

    Experiments get no CI protection (test_causality_strict.py parametrizes
    over the registry only). Same procedure as R-28/R-31/R-33/R-37: bars
    after a cut are multiplied by 3 in one copy, divided by 3 in another;
    every decision at or before the cut must be bit-identical. Particularly
    important here because of the custom EWM/shift construction for
    mu/var in ``prepare`` — exactly the kind of full-series statistic
    ROUTINE.md warns a truncation test alone will not catch if it is
    computed over the WHOLE series and applied to early rows. The check
    below compares the prepared columns directly, not just order objects,
    for that reason.
    """
    # Restricted to strictly pre-2023 bars: this session's brief is
    # explicit ("do not read or use any bar dated 2023-01-01 or later, for
    # ANY purpose"), so the probe is confined to the inner splits rather
    # than relying on the project's usual "tail probes don't count against
    # the holdout" convention.
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

    strat_kwargs = dict(halflife_days=15.0, alpha=0.6, beta=0.05, safety_cap=5.0)

    def prepared(frame):
        return KellyRegimeV7Crra(**strat_kwargs).prepare(frame.copy())

    pa = prepared(up)
    pb = prepared(down)
    ok = True
    for col in ("target", "_frac", "_mu", "_var", "_f_star"):
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
        s = KellyRegimeV7Crra(**strat_kwargs)
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

    a = run_backtest(KellyRegimeV7Crra(**strat_kwargs), up.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    b = run_backtest(KellyRegimeV7Crra(**strat_kwargs), down.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-6
    print(f"  max |equity difference| before the cut = {worst_eq:.3e}  "
          f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")

    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# ------------------------------------------------------------------------------ eth


def eth(candidate_kwargs: dict) -> None:
    """Step 7: pre-registered falsification — does the candidate hold on ETH?

    Same venue (Bitfinex), same window as R-17/R-28/R-31/R-33/R-37, both
    spot and 5x futures, candidate vs shipped v4 defaults as the control.
    Falsification rule (fixed before running): if the candidate's
    Sharpe/drawdown/return on ETH is not at least as good as v4's by more
    than a token margin, or the candidate is visibly worse on ETH than the
    BTC control run through this identical pipeline (curve-fitting to
    BTC-specific history), this direction fails, full stop.
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
            cand = KellyRegimeV7Crra(**candidate_kwargs)
            m_c, vol_c, not_c, res_c = measure(cand, None, None, df=df, market=market)
            line("    kelly_regime_v7_crra (candidate)", m_c, vol_c, not_c, res_c)


# ------------------------------------------------------------------------------- main


# The inner-validation-selected candidate (see select_inner_validation.csv).
# halflife=60d dominates its own (alpha, beta) block in 6 of 8 cases; among
# the halflife=60d configs, alpha=0.50/beta=0.05 tops both markets
# (spot Sharpe 0.53, futures Sharpe 0.47) with alpha=0.50/beta=0.10 and
# alpha=0.60/beta=0.10 close behind (Sharpe 0.43-0.51) -- see the session
# report for the honest plateau/noise discussion (n=4-10 trades per config).
DEFAULT_CANDIDATE = dict(halflife_days=60.0, alpha=0.50, beta=0.05, safety_cap=5.0)


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "sweep":
        sweep()
    elif choice == "select":
        select()
    elif choice == "artifact":
        artifact_check(DEFAULT_CANDIDATE)
    elif choice == "causality":
        causality()
    elif choice == "eth":
        eth(DEFAULT_CANDIDATE)
    else:
        print("usage: python experiments/kelly_regime_v7_crra.py "
              "[sweep|select|artifact|causality|eth]")
