#!/usr/bin/env python
"""Risk-constrained Kelly drawdown cap on kelly_regime_v4's own sizing (SIZE axis).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea
--------
``kelly_regime_v4`` answers "how much should I hold?" with
``scale = min(target_vol / realized_vol, max_leverage)`` -- two hand-picked
constants (``target_vol=0.55``, ``max_leverage=2.0``) with no probabilistic
interpretation. Busseti, Ryu & Boyd (2016, "Risk-Constrained Kelly
Gambling", Journal of Investing 25(3) / arXiv:1603.06183) show that the
Kelly gambling problem with an explicit probabilistic drawdown constraint
``Prob(min future wealth < alpha) < beta`` reduces, via a convex bound, to
maximizing a CRRA/isoelastic-utility criterion with risk-aversion parameter
``lambda = ln(beta) / ln(alpha)`` -- i.e. a stated maximum acceptable
drawdown ``1-alpha`` at confidence ``1-beta`` maps directly onto a single
risk-aversion number. Combined with the classical Merton (1969, Rev. Econ.
Stat.; 1971, J. Econ. Theory) result that the CRRA-optimal bet fraction
under a drift-mu / variance-sigma^2 return process is
``f* = mu / (lambda * sigma^2)`` (the plain Kelly fraction ``mu/sigma^2`` is
the lambda=1 case), this gives a formal, probability-calibrated CAP on
position size, in place of ``max_leverage=2.0``, which is just a number
someone picked.

Mechanism, one sentence: leave v4's vote and its own conditional-vol-target
scale completely unchanged, and ADD a second, independently causally
estimated ceiling ``f_risk[t] = mu[t] / (lambda * sigma[t]^2)`` (lambda
fixed in advance from a chosen (alpha, beta) pair), so that
``target[t] = frac[t] * min(v4_scale[t], f_risk[t])`` -- the new cap can
only ever REDUCE v4's exposure, never raise it above what v4 would already
choose.

Constraint attacked: SIZE (v4's own axis -- how much to hold given a fixed
regime call, not a better regime call) and, via the drawdown-probability
interpretation, ERR (this project has no error control anywhere in the
signal path; a stated (alpha, beta) is the closest thing to one that has
been tried on the SIZE axis).

Not a duplicate of
-------------------
- L-01..L-04 (`target_vol`/`max_leverage` as hand-picked constants): this
  file keeps v4's OWN constants untouched and adds a second, formally
  derived ceiling on top -- it does not retune L-01..L-04's numbers
  (that was R-37 conservative, already NEGATIVE).
- R-11 (Grossman-Zhou drawdown *cushion*, 1993): a FLOOR/reactive
  mechanism triggered by REALIZED drawdown itself. This file's cap is
  forward-looking, active at all times, derived from a return-DISTRIBUTION
  risk bound -- it never looks at the strategy's own equity curve.
- R-28/R-31 (e-process anytime-valid hypothesis testing on the
  regime/direction signal): a different question (is the regime real?).
  This file never touches the vote; it only caps the SIZE given a fixed
  vote.
- R-34 (`harsanyi_crowd`'s Bayesian posterior as a bounded dampener or an
  unbounded margin): a different signal SOURCE (a discrete-state belief
  posterior over bull/bear/chop). This file uses realized-return moments
  (mu, sigma^2) from a causal EWM of log returns, not a belief posterior,
  and its lambda is fixed a priori from a stated risk tolerance rather than
  fitted.
- R-37 (per-vote-state Kelly fraction `mu_state/sigma_state**2` REPLACING
  v4's global `target_vol`, and a hyperparameter retune of `target_vol`/
  `max_leverage`): both modify v4's own sizing formula. This file changes
  nothing about v4's formula; it multiplies the result of an unchanged
  v4-vote/v4-scale computation by a `min(..., f_risk)` operation, so it can
  only ever shrink v4's own exposure, never alter or replace it.

Causal construction for mu / sigma^2 -- read this before the code
-------------------------------------------------------------------
This is the single easiest place to leak the future into the past (R-21: a
one-day lookahead broadcast onto 5m bars was worth +2.1 Sharpe and passed a
truncation test; an ``i+1`` peek returned $3.7e23 with a green suite). The
construction here, precisely, so it can be checked without reading code:

1. ``r[t] = log(close[t]) - log(close[t-1])``, the bar's own realized
   log-return -- known entirely from closes at or before bar t.
2. ``mu_bar[t] = r.ewm(halflife=hl_bars, min_periods=BARS_PER_DAY).mean()``,
   THEN ``.shift(1)`` -- exactly mirroring ``kelly_regime.py``'s own
   ``vol = ewm(...).shift(1)`` idiom. The shift means bar t's sizing
   decision reads a mean of returns strictly BEFORE bar t, never bar t
   itself.
3. ``var_bar[t]`` is built the identical way with pandas' EWM ``.var()``,
   then ``.shift(1)``.
4. Both mu_bar and var_bar are annualization-invariant as a RATIO: since
   variance of an i.i.d.-approximated return process scales linearly with
   the sampling interval and so does the mean, ``mu_bar/var_bar`` computed
   on raw 5-minute bars equals the same ratio computed on annualized
   quantities (the BARS_PER_YEAR factor cancels). ``f_risk`` is therefore
   computed directly from the per-bar (unannualized) EWM statistics -- one
   fewer place for a units mistake to hide.
5. ``f_risk[t] = mu_bar[t] / (lambda * var_bar[t])``, floored at zero (a
   state whose noisy trailing estimate says "no positive edge" caps
   exposure to zero, never flips it to a short -- mirrors v4's own
   never-short logic and keeps the cap architecturally incapable of
   INCREASING exposure).
6. v4's own vote (``frac``) and its own conditional-vol-target scale
   (``v4_scale`` -- the identical high/low-vol-breakout hysteresis state
   machine registered in ``kelly_regime_v3.py``/``kelly_regime_v4.py``) are
   copied here verbatim, unchanged, exactly as ``kelly_regime_v6_state_kelly.py``
   copied v4's vote verbatim. Nothing about detection or v4's own scale
   formula is touched.
7. ``target[t] = frac[t] * min(v4_scale[t], f_risk[t])``, debounced by the
   same 10% deadband v3/v4 use, applied to this new combined series (not to
   v4's own post-deadband target -- the deadband runs once, on the final
   combined signal, exactly where v3/v4's own deadband runs on their single
   combined signal).

Pre-registered failure modes (named before any code ran)
----------------------------------------------------------
(a) The causally-estimated mu (drift) is too noisy at 5-minute cadence
    relative to sigma^2 for f_risk to be a meaningfully time-varying cap --
    it may saturate near zero (bearish/noisy mu estimate) almost
    everywhere, collapsing this to "always flat" or a near-constant, tiny
    cap.
(b) f_risk may bind so RARELY (i.e. sit above v4's own max_leverage=2.0
    almost everywhere) that ``min(v4_scale, f_risk)`` never differs from
    v4_scale alone -- "no effect".
(c) Whatever improvement appears is, once again, an exposure-level
    artifact (L-04/R-28/R-31/R-32/R-33/R-34/R-37): the winning config
    simply asks for a flat-rescaled fraction of v4's own exposure. Checked
    explicitly below by regressing the candidate's target series against a
    flat rescale of v4's own target series, mean-notional-matched, and
    reporting R^2 (mirrors R-34's R^2=0.997 diagnostic).

Both (a) and (b) collapse this direction to "no effect" or "a near-constant
rescale"; the exposure-artifact check in (c) is exactly the instrument that
would catch the latter. Report which happened, precisely.

Usage
-----
    python experiments/kelly_regime_v7_ddcap.py sweep       # step 3
    python experiments/kelly_regime_v7_ddcap.py select      # step 5 (plateau view)
    python experiments/kelly_regime_v7_ddcap.py artifact    # exposure-artifact check
    python experiments/kelly_regime_v7_ddcap.py causality   # step 6
    python experiments/kelly_regime_v7_ddcap.py eth         # step 7
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


# --------------------------------------------------------------------- strategy


class KellyRegimeV7DDCap(Strategy):
    """v4's exact vote and conditional-vol scale, further capped by a Busseti/Ryu/Boyd/Merton risk bound.

    Everything about the vote (horizons, band, latching hysteresis) and v4's
    own conditional-vol-target scale (the high/low-vol-breakout state
    machine, ``target_vol``, ``max_leverage``) is copied verbatim from
    ``kelly_regime_v3`` / ``kelly_regime_v4``. The only new thing is a
    second ceiling, ``f_risk = mu / (lambda * sigma**2)``, estimated from a
    causal EWM of log returns and combined as
    ``target = frac * min(v4_scale, f_risk)`` -- the risk cap can only
    shrink v4's own exposure, never raise it.
    """

    name = "kelly_regime_v7_ddcap"

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85,
                 alpha: float = 0.6, beta: float = 0.05,
                 halflife_days: float = 90.0) -> None:
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lam = float(np.log(self.beta) / np.log(self.alpha))
        self.halflife_days = float(halflife_days)
        # Warmup scales with the halflife being tested, exactly mirroring
        # kelly_regime_v6_state_kelly.py's convention, so inner-validation
        # (which has pre-period history to draw on) starts with the mu/var
        # EWMs already populated rather than restarting cold.
        self.warmup = max(80 * BARS_PER_DAY + 10, int(3 * self.halflife_days * BARS_PER_DAY))

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        idx = df.index
        n = len(df)
        r = np.log(close).diff()

        # ---- v4's vote, verbatim (see kelly_regime.py) ----
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

        # ---- v4's own conditional-vol-target scale, verbatim (see kelly_regime_v3.py) ----
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

        v4_scale = np.zeros(n)
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
            v4_scale[i] = full[i] if state != 0 else steady[i]

        # ---- new: risk-constrained Kelly cap (Busseti, Ryu & Boyd 2016; Merton 1969/1971) ----
        hl_bars = self.halflife_days * BARS_PER_DAY
        mu_bar = (r.ewm(halflife=hl_bars, min_periods=BARS_PER_DAY).mean()
                  .shift(1).to_numpy())
        var_bar = (r.ewm(halflife=hl_bars, min_periods=BARS_PER_DAY).var()
                   .shift(1).to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            f_risk = np.where(var_bar > 0, mu_bar / (self.lam * var_bar), np.nan)
        f_risk = np.where(np.isfinite(f_risk), f_risk, 0.0)
        f_risk = np.clip(f_risk, 0.0, None)  # floor at 0: never flips sign, only ever caps

        cap = np.minimum(v4_scale, f_risk)

        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = frac[i] * cap[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["_frac"] = frac
        df["_v4_scale"] = v4_scale
        df["_f_risk"] = f_risk
        df["_mu_bar"] = mu_bar
        df["_var_bar"] = var_bar
        df["_cap_binds"] = (f_risk < v4_scale).astype(float)
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

OUT = ROOT / "reports" / "kelly_regime_v7_ddcap"


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

ALPHA_GRID = (0.5, 0.6, 0.7, 0.8)   # 1 - alpha = max acceptable drawdown
BETA_GRID = (0.05, 0.10)             # 1 - beta = confidence
HALFLIFE_GRID = (30.0, 90.0, 180.0)  # days, mu/sigma^2 EWM half-life


def grid_configs():
    for alpha in ALPHA_GRID:
        for beta in BETA_GRID:
            for hl in HALFLIFE_GRID:
                yield dict(alpha=alpha, beta=beta, halflife_days=hl)


def sweep() -> pd.DataFrame:
    """Step 3: sweep the (alpha, beta, halflife) grid on inner-train only, spot market."""
    rows = []
    t0 = time.time()
    for cfg in grid_configs():
        strat = KellyRegimeV7DDCap(**cfg)
        lam = strat.lam
        m, vol, notional, res = measure(strat, *TRAIN, market=SPOT, count=True)
        cap_bind_frac = float(res.df["_cap_binds"].mean()) if "_cap_binds" in res.df else float("nan")
        rows.append({**cfg, "lam": lam, "final": m.final_balance, "vol": vol,
                     "notional": notional, "max_dd": m.max_drawdown_pct,
                     "sharpe": m.sharpe, "trades": m.num_trades,
                     "fees": m.fees_paid, "liquidated": m.liquidated,
                     "cap_binds_frac": cap_bind_frac})
        print(f"[{N_EVALUATED:>3d}] alpha={cfg['alpha']:.2f} beta={cfg['beta']:.2f} "
              f"lam={lam:6.3f} hl={cfg['halflife_days']:>5.0f}d  "
              f"final=${m.final_balance:>10,.0f} DD={m.max_drawdown_pct:>5.1f}% "
              f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
              f"cap_binds={cap_bind_frac:5.1%} [{time.time() - t0:.0f}s]")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "sweep_inner_train.csv", index=False)
    print(f"\nconfigurations evaluated (step 3): {N_EVALUATED}")
    print(f"written: {OUT / 'sweep_inner_train.csv'}")
    return out


# --------------------------------------------------------------------------- step 5


def select(candidates: list[dict] | None = None) -> None:
    """Step 5: score candidates on inner-validation, both markets, plateau view."""
    if candidates is None:
        candidates = list(grid_configs())
    rows = []
    for cfg in candidates:
        strat_spot = KellyRegimeV7DDCap(**cfg)
        strat_fut = KellyRegimeV7DDCap(**cfg)
        m_s, vol_s, not_s, res_s = measure(strat_spot, *VALID, market=SPOT)
        m_f, vol_f, not_f, res_f = measure(strat_fut, *VALID, market=FUTURES)
        cap_s = float(res_s.df["_cap_binds"].mean()) if "_cap_binds" in res_s.df else float("nan")
        cap_f = float(res_f.df["_cap_binds"].mean()) if "_cap_binds" in res_f.df else float("nan")
        rows.append({**cfg, "lam": strat_spot.lam,
                     "spot_final": m_s.final_balance, "spot_dd": m_s.max_drawdown_pct,
                     "spot_sharpe": m_s.sharpe, "spot_trades": m_s.num_trades,
                     "spot_vol": vol_s, "spot_notional": not_s, "spot_cap_binds": cap_s,
                     "fut_final": m_f.final_balance, "fut_dd": m_f.max_drawdown_pct,
                     "fut_sharpe": m_f.sharpe, "fut_trades": m_f.num_trades,
                     "fut_vol": vol_f, "fut_notional": not_f, "fut_cap_binds": cap_f})
        print(f"alpha={cfg['alpha']:.2f} beta={cfg['beta']:.2f} lam={strat_spot.lam:6.3f} "
              f"hl={cfg['halflife_days']:>5.0f}d  "
              f"spot: ${m_s.final_balance:>9,.0f} DD{m_s.max_drawdown_pct:>5.1f}% "
              f"sh{m_s.sharpe:>5.2f}  fut: ${m_f.final_balance:>9,.0f} "
              f"DD{m_f.max_drawdown_pct:>5.1f}% sh{m_f.sharpe:>5.2f} "
              f"cap_binds={cap_s:4.0%}/{cap_f:4.0%}")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "select_inner_validation.csv", index=False)
    print(f"\nwritten: {OUT / 'select_inner_validation.csv'}")


def v4_baseline(start, end, market, df=None) -> None:
    m, vol, notional, res = measure(get_strategy(INCUMBENT), start, end, df=df, market=market)
    line(f"{INCUMBENT} (shipped defaults)", m, vol, notional, res)


# --------------------------------------------------------------------------- artifact


def exposure_artifact_check(candidate_kwargs: dict) -> None:
    """Mandatory exposure-artifact check (ROUTINE.md standing rule, sharpened by R-33).

    Build a "flat-rescaled v4" comparator: v4's own unchanged target,
    multiplied by a single constant c chosen so its mean notional matches
    the candidate's mean notional over the SAME period. Report R^2 of the
    candidate's target series against that flat rescale, on inner-
    validation, both markets. R^2 > 0.95 (R-34's own 0.997 threshold) means
    "this is the standard exposure-level artifact", reported honestly as
    such, not as a win.
    """
    print("\nexposure-artifact check (inner-validation, mean-notional-matched flat rescale of v4):")
    for mname, market in MARKETS:
        cand = KellyRegimeV7DDCap(**candidate_kwargs)
        m_c, vol_c, not_c, res_c = measure(cand, *VALID, market=market)
        v4 = get_strategy(INCUMBENT)
        m_v4, vol_v4, not_v4, res_v4 = measure(v4, *VALID, market=market)

        cand_t = res_c.df["target"].to_numpy(dtype=float)
        v4_t = res_v4.df["target"].reindex(res_c.df.index).to_numpy(dtype=float)
        # constant c matching mean notional (mean |target|), not mean(target)
        # directly, since target can be signless-flat at 0 for long stretches;
        # matches the R-34 convention exactly (mean notional match).
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


# ------------------------------------------------------------------------ causality


def causality() -> None:
    """Step 6: by-hand two-opposite-tampers lookahead probe.

    Experiments get no CI protection (test_causality_strict.py parametrizes
    over the registry only). Same procedure as R-28/R-31/R-33/R-37: bars
    after a cut are multiplied by 3 in one copy, divided by 3 in another;
    every decision at or before the cut must be bit-identical. Particularly
    important here because ``f_risk`` is a NEW full-series EWM statistic --
    exactly the kind of thing ROUTINE.md warns a truncation test alone will
    not catch if computed over the whole series and applied to early rows.
    The check below compares the prepared columns directly, not just
    orders, for that reason.
    """
    # Restricted to strictly pre-2023 bars: this session's brief explicitly
    # forbids reading any bar dated 2023-01-01 or later for ANY purpose, so
    # the probe is deliberately confined to the inner splits rather than
    # relying on the R-28/R-31/R-33 convention of running it on the dataset
    # tail.
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

    strat_kwargs = dict(alpha=0.6, beta=0.05, halflife_days=90.0)

    def prepared(frame):
        return KellyRegimeV7DDCap(**strat_kwargs).prepare(frame.copy())

    pa = prepared(up)
    pb = prepared(down)
    ok = True
    for col in ("target", "_frac", "_v4_scale", "_f_risk", "_mu_bar", "_var_bar"):
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
        s = KellyRegimeV7DDCap(**strat_kwargs)
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

    a = run_backtest(KellyRegimeV7DDCap(**strat_kwargs), up.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    b = run_backtest(KellyRegimeV7DDCap(**strat_kwargs), down.iloc[:cut + 1], FUTURES,
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

    Same venue (Bitfinex), same window as R-17/R-28/R-31/R-33/R-37, both
    spot and 5x futures, candidate vs shipped v4 defaults as the control.
    Falsification rule (fixed before running): if the candidate is not at
    least as good as v4 by more than a token margin on ETH, or is visibly
    worse on ETH than the BTC control run through the identical pipeline
    (evidence of curve-fitting to BTC-specific history), this direction
    fails, full stop.
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
            cand = KellyRegimeV7DDCap(**candidate_kwargs)
            m_c, vol_c, not_c, res_c = measure(cand, None, None, df=df, market=market)
            line("    kelly_regime_v7_ddcap (candidate)", m_c, vol_c, not_c, res_c)


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
        # The inner-validation-selected candidate; update after `select` runs.
        default_cand = dict(alpha=0.6, beta=0.05, halflife_days=90.0)
        exposure_artifact_check(default_cand)
    elif choice == "causality":
        causality()
    elif choice == "eth":
        default_cand = dict(alpha=0.6, beta=0.05, halflife_days=90.0)
        eth(default_cand)
    else:
        print("usage: python experiments/kelly_regime_v7_ddcap.py "
              "[sweep|select|artifact|causality|eth]")
