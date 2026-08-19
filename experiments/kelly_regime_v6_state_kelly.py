#!/usr/bin/env python
"""State-conditional Kelly sizing on kelly_regime_v4's own vote (SIZE axis, R-37 novel branch).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea
--------
``kelly_regime_v4`` answers "how bullish is the crowd?" with a three-anchor
latched vote that lands on one of four discrete states,
``frac in {0, 1/3, 2/3, 1}``, and answers "how much should I hold?" with one
GLOBAL number: ``scale = min(target_vol / realized_vol, max_leverage)``,
identical across all four states. ``target = frac * scale``. So "just
barely bullish" (frac=1/3) and "unanimously bullish" (frac=1) differ only
by a multiplicative vote fraction applied to the SAME volatility-implied
Sharpe estimate — the sizing formula never asks whether the two states
actually carry a different expected return per unit of risk.

The literal Kelly criterion for a single risky bet under log utility
(Kelly 1956; Thorp) is ``f* = mu / sigma**2`` — continuous-time / per-period
drift over variance, not variance alone. If the four vote states carry
genuinely different forward drift-to-variance ratios, sizing all of them
off one global ``target_vol`` leaves a measurable, structural
inefficiency on the table. This file tests that, directly, on this
project's own data.

Mechanism, one sentence: reuse v4's exact 3-anchor latched vote unchanged,
but replace its single global ``scale`` with FOUR separately, causally
estimated Kelly scales — one per vote state — each built from that state's
own trailing mu_state / sigma_state**2, with a conservative fractional-Kelly
multiplier and the same max_leverage cap v4 already uses.

Constraint attacked: SIZE (v4's own axis — how much to hold given a fixed
regime call, not a better regime call).

Not a duplicate of
-------------------
- R-01 (HMM), R-02 (jump models), R-03 (BOCPD), R-28 (e-process gate),
  R-34 (Bayesian posterior margin, both branches): all of these REPLACE or
  AUGMENT the regime-DETECTION mechanism — a different way to decide
  bull/bear/chop. This file changes nothing about detection: the vote here
  is v4's vote, verbatim, three anchors, same band, same latching
  hysteresis. Only the response to a *given* vote state changes.
- L-03 / ``kelly_regime_v2`` (R-06's convex vote response, ``frac**gamma``):
  reshapes how PARTIAL votes are treated but still multiplies a single
  global ``target_vol`` — it never estimates a different expected return
  per state. This file estimates four different (mu, sigma**2) pairs, one
  per state, and never touches the exponent on ``frac``.
- R-33 / R-36 (matched-risk, B-13/B-14): those measure v4 AS REGISTERED
  against a passive benchmark at equal risk; they do not modify v4's
  sizing formula. This file is the "can a SIZE-axis modification capture
  more of the post-2021 edge" follow-up R-36 explicitly named as open.

Causal construction for mu_state / sigma_state**2 — read this before the code
-------------------------------------------------------------------------
This is the single easiest place to leak the future into the past (R-21:
a one-day lookahead broadcast onto 5m bars was worth +2.1 Sharpe and passed
a truncation test; an ``i+1`` peek returned $3.7e23 with a green suite).
The construction here, precisely, so it can be checked without reading code:

1. ``frac[t]`` is v4's own vote at bar t (unchanged, causal by construction
   — it already ships as a registered strategy).
2. ``r[t] = log(close[t] / close[t-1])``, the bar's own realized return —
   known at the close of bar t, the same bar whose close prints ``r[t]``.
3. Attribute ``r[t]`` to the vote state that was active over the period
   that produced it: ``bucket_state[t] = frac[t-1]``, ``bucket_ret[t] =
   r[t]``. This pair becomes fully known at the close of bar t.
4. For each of the four states k, build a TIME-HALFLIFE (not
   occurrence-count) exponentially weighted mean and variance of
   ``bucket_ret`` restricted to bars where ``bucket_state == k``, using
   pandas' ``ewm(halflife=<days>, times=<those bars' own timestamps>)``.
   Time-based decay (not occurrence-based) means an old bull episode's
   evidence decays in calendar time even if bull bars are sparse over a
   long calendar span, and dense clusters of the same state do not
   silently inflate its half-life in wall-clock terms. Variance is
   ``E[r**2] - E[r]**2`` via two such EWMs (pandas' ``var`` does not
   support ``times``), clipped at zero.
5. This EWM lives only at the occurrence timestamps (the state-k bars).
   It is reindexed onto the FULL bar index and forward-filled, so between
   occurrences of state k the estimate stays flat at its last known value
   — nothing here ever reads a bar that has not happened yet, it just
   holds an old, decaying belief until state k recurs.
6. The reindexed, forward-filled, per-state series is then shifted ONE
   MORE bar (``.shift(1)``), exactly mirroring ``kelly_regime.py``'s own
   ``vol = ewm(...).shift(1)`` convention. This is a deliberate one-bar
   safety margin on top of step 3's already-causal pairing: bar t's sizing
   decision (computed at the close of bar t, filled at the open of bar
   t+1) reads mu_state / var_state as of bar t-1, never bar t itself, even
   though step 3's pairing would already have been legal without it.
7. At bar t, the strategy looks up ``mu_state[t]`` / ``var_state[t]`` for
   k = frac[t] (the CURRENT vote, already causal) from the four series
   built in steps 4-6, forms ``kelly_f = clip(mu_state / var_state, 0,
   None)`` (floored at zero — a state whose noisy trailing estimate says
   "negative expected return" sizes to flat, exactly v4's own logic for
   its bear state, never to a short), and sets
   ``scale_state = min(kelly_mult * kelly_f, max_leverage)``.
8. ``target = frac[t] * scale_state[t]``, then the same 10% deadband v4
   uses. Note step 8 keeps v4's multiplicative structure: even if a
   noisy trailing estimate for the all-bearish state (frac=0) came out
   positive, ``frac=0`` still forces the position flat — the vote keeps
   its role as a hard gate, and only the SIZE given to a non-zero vote
   state is now state-conditional.
9. Below ``min_obs`` occurrences of a state (default 2,000, roughly a
   week of that state's own bars, checked on the SAME lagged basis as
   step 6), the strategy stays flat in that state rather than size off an
   unstable estimate — this is the direct, pre-registered mitigation for
   failure mode (a) below.

Pre-registered failure modes (named before any code ran)
----------------------------------------------------------
(a) State-conditional drift estimates are too noisy at 5-minute cadence to
    be usable — this project's own R-34 novel branch found exactly this
    failure mode for a different, real, independent continuous signal.
    Mitigated by (not eliminated by) the ``min_obs`` floor and by sweeping
    the estimation half-life; if it still fails, that is the honest
    result, not a bug.
(b) Whatever improvement appears is, once again, an exposure-level
    artifact (L-04/R-28/R-31/R-32/R-33): the winning config simply asks
    for more raw notional than v4's default. Checked explicitly below by
    comparing realized volatility and mean notional fraction against
    shipped v4 on the same period.
(c) Turnover increases materially because the state-conditional scale
    changes value more often than v3/v4's hysteresis-latched vol regime
    did. Checked explicitly below via trade counts.

Usage
-----
    python experiments/kelly_regime_v6_state_kelly.py sweep       # step 3
    python experiments/kelly_regime_v6_state_kelly.py select      # step 5
    python experiments/kelly_regime_v6_state_kelly.py causality   # step 6
    python experiments/kelly_regime_v6_state_kelly.py eth         # step 7
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
STATES = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
STATE_LABELS = {0.0: "0/3", 1.0 / 3.0: "1/3", 2.0 / 3.0: "2/3", 1.0: "3/3"}


# --------------------------------------------------------------------- strategy


class KellyRegimeV6StateKelly(Strategy):
    """v4's exact vote, sized by a per-state causal Kelly ratio instead of one global target_vol.

    Everything about the vote (horizons, band, latching hysteresis) is
    copied verbatim from ``kelly_regime_v4`` / ``kelly_regime_v3`` /
    ``kelly_regime.py``. The only thing that changes is what happens after
    the vote: instead of ``scale = min(target_vol / realized_vol,
    max_leverage)`` shared by all four states, each state k gets its own
    ``scale_state_k = min(kelly_mult * mu_k / sigma_k**2, max_leverage)``,
    both moments estimated causally (see module docstring for the exact
    construction). ``target = frac * scale_state``, debounced by the same
    10% deadband.
    """

    name = "kelly_regime_v6_state_kelly"

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 deadband: float = 0.10, halflife_days: float = 90.0,
                 kelly_mult: float = 0.5, max_leverage: float = 2.0,
                 min_obs: int = 2000, stat_horizon_bars: int = 1) -> None:
        self.horizons = horizons
        self.band = band
        self.deadband = deadband
        self.halflife_days = float(halflife_days)
        self.kelly_mult = float(kelly_mult)
        self.max_leverage = float(max_leverage)
        self.min_obs = int(min_obs)
        self.stat_horizon_bars = int(stat_horizon_bars)
        # Warmup scales with the half-life being tested so inner-validation
        # (which does have pre-period history to draw on) starts with the
        # per-state EWMs already populated, rather than restarting cold at
        # 2021-01-01. Inner-train has no pre-2017 data at all (the dataset
        # starts exactly on 2017-01-01), so this only helps later splits —
        # a limitation this strategy shares with kelly_regime_v4 itself
        # (R-22).
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
        h = self.stat_horizon_bars

        frac = self._vote(df)
        frac_s = pd.Series(frac, index=idx)

        # r[t] = log(close[t]/close[t-h]): the h-bar TRAILING (never
        # forward-looking) return ending at bar t, known entirely from
        # closes at or before t.
        r = np.log(close).diff(h)

        # bucket_state[t] = frac[t-h]: the vote that was active over the
        # h-bar window whose realized return prints at t. Both halves of
        # this pair are known no later than the close of bar t.
        bucket_state = frac_s.shift(h)
        bucket_ret = r

        halflife = pd.Timedelta(days=self.halflife_days)
        mu_arr = np.full(n, np.nan)
        var_arr = np.full(n, np.nan)
        cnt_arr = np.zeros(n)
        bstate_np = bucket_state.to_numpy(dtype=float)

        for k in STATES:
            mask = np.isclose(bstate_np, k)
            occ = bucket_ret[mask].dropna()
            if len(occ) == 0:
                continue
            ew = occ.ewm(halflife=halflife, times=occ.index, min_periods=self.min_obs)
            mu_occ = ew.mean()
            mu2_occ = occ.pow(2).ewm(halflife=halflife, times=occ.index,
                                      min_periods=self.min_obs).mean()
            var_occ = (mu2_occ - mu_occ ** 2).clip(lower=0.0)
            cnt_occ = pd.Series(np.arange(1, len(occ) + 1), index=occ.index)

            # Reindex the sparse (occurrence-only) estimate onto the full
            # bar index, forward-filled so it holds its last value between
            # occurrences, THEN shift one more bar — the extra safety
            # margin described in the module docstring, mirroring
            # kelly_regime.py's `vol = ewm(...).shift(1)`.
            mu_full = mu_occ.reindex(idx).ffill().shift(1)
            var_full = var_occ.reindex(idx).ffill().shift(1)
            cnt_full = cnt_occ.reindex(idx).ffill().shift(1).fillna(0.0)

            sel = np.isclose(frac, k)  # bars where THIS state is the current vote
            mu_arr[sel] = mu_full.to_numpy()[sel]
            var_arr[sel] = var_full.to_numpy()[sel]
            cnt_arr[sel] = cnt_full.to_numpy()[sel]

        with np.errstate(divide="ignore", invalid="ignore"):
            kelly_f = np.where(var_arr > 0, mu_arr / var_arr, np.nan)
        kelly_f = np.where(np.isfinite(kelly_f), kelly_f, 0.0)
        kelly_f = np.clip(kelly_f, 0.0, None)  # never short a state; mirrors v4

        scale_state = np.minimum(self.kelly_mult * kelly_f, self.max_leverage)
        enough = cnt_arr >= self.min_obs
        scale_state = np.where(enough, scale_state, 0.0)

        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = frac[i] * scale_state[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["_frac"] = frac
        df["_mu_state"] = mu_arr
        df["_var_state"] = var_arr
        df["_kelly_f_state"] = kelly_f
        df["_enough"] = enough
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


HALFLIFE_GRID = (30.0, 90.0, 180.0, 365.0)
KELLY_MULT_GRID = (0.25, 0.5, 0.75, 1.0)
STAT_HORIZON_GRID = (1, 288)  # 5-minute bars vs 1-day bars
MAX_LEVERAGE_DEFAULT = 2.0


def grid_configs():
    for hl in HALFLIFE_GRID:
        for km in KELLY_MULT_GRID:
            for sh in STAT_HORIZON_GRID:
                yield dict(halflife_days=hl, kelly_mult=km,
                           max_leverage=MAX_LEVERAGE_DEFAULT, stat_horizon_bars=sh)


def sweep() -> pd.DataFrame:
    """Step 3: sweep the grid on inner-train only, spot market (counted once)."""
    rows = []
    t0 = time.time()
    for cfg in grid_configs():
        strat = KellyRegimeV6StateKelly(**cfg)
        m, vol, notional, res = measure(strat, *TRAIN, market=SPOT, count=True)
        rows.append({**cfg, "final": m.final_balance, "vol": vol,
                     "notional": notional, "max_dd": m.max_drawdown_pct,
                     "sharpe": m.sharpe, "trades": m.num_trades,
                     "fees": m.fees_paid, "liquidated": m.liquidated})
        print(f"[{N_EVALUATED:>3d}] hl={cfg['halflife_days']:>5.0f}d "
              f"km={cfg['kelly_mult']:.2f} sh={cfg['stat_horizon_bars']:>3d}bars  "
              f"final=${m.final_balance:>10,.0f} DD={m.max_drawdown_pct:>5.1f}% "
              f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
              f"[{time.time() - t0:.0f}s]")
    df = pd.DataFrame(rows)
    OUT = ROOT / "reports" / "kelly_regime_v6"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "sweep_inner_train.csv", index=False)
    print(f"\nconfigurations evaluated (step 3): {N_EVALUATED}")
    print(f"written: {OUT / 'sweep_inner_train.csv'}")
    return df


# --------------------------------------------------------------------------- step 5


def state_stats_snapshot(strategy: KellyRegimeV6StateKelly, start, end) -> None:
    """Print the actual measured mu_state / sigma_state**2 by vote state.

    Read straight off the strategy's own prepared columns over the given
    period — the empirical claim under test ("do the states actually
    differ") stated in numbers, not just inferred from the backtest
    outcome.
    """
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    prefix = min(lo, strategy.warmup)
    frame = strategy.prepare(DF.iloc[lo - prefix: hi].copy())
    frame = frame.iloc[prefix:]
    print(f"  state-conditional stats, {start} -> {end} "
          f"(halflife={strategy.halflife_days:.0f}d, "
          f"stat_horizon={strategy.stat_horizon_bars}bars):")
    for k in STATES:
        sub = frame[np.isclose(frame["_frac"].to_numpy(), k) & frame["_enough"]]
        if len(sub) == 0:
            print(f"    frac={STATE_LABELS[k]}  no bars with enough history yet")
            continue
        mu = sub["_mu_state"].mean()
        var = sub["_var_state"].mean()
        kf = sub["_kelly_f_state"].mean()
        print(f"    frac={STATE_LABELS[k]}  bars={len(sub):>7d}  "
              f"mean mu_state={mu:+.3e}/bar  mean var_state={var:.3e}  "
              f"mean kelly_f={kf:+.3f}  "
              f"(annualized mu~{mu * BARS_PER_YEAR:+.2%}/yr)")


def select(candidates: list[dict] | None = None) -> None:
    """Step 5: score candidates on inner-validation, both markets, plateau view."""
    if candidates is None:
        # A neighbourhood around a plausible half-Kelly default, chosen
        # before reading inner-validation results: this is the grid this
        # function is FIRST called with, before any narrowing.
        candidates = list(grid_configs())
    rows = []
    for cfg in candidates:
        strat_spot = KellyRegimeV6StateKelly(**cfg)
        strat_fut = KellyRegimeV6StateKelly(**cfg)
        m_s, vol_s, not_s, res_s = measure(strat_spot, *VALID, market=SPOT)
        m_f, vol_f, not_f, res_f = measure(strat_fut, *VALID, market=FUTURES)
        rows.append({**cfg, "spot_final": m_s.final_balance, "spot_dd": m_s.max_drawdown_pct,
                     "spot_sharpe": m_s.sharpe, "spot_trades": m_s.num_trades,
                     "spot_vol": vol_s, "spot_notional": not_s,
                     "fut_final": m_f.final_balance, "fut_dd": m_f.max_drawdown_pct,
                     "fut_sharpe": m_f.sharpe, "fut_trades": m_f.num_trades,
                     "fut_vol": vol_f, "fut_notional": not_f})
        print(f"hl={cfg['halflife_days']:>5.0f}d km={cfg['kelly_mult']:.2f} "
              f"sh={cfg['stat_horizon_bars']:>3d}  "
              f"spot: ${m_s.final_balance:>9,.0f} DD{m_s.max_drawdown_pct:>5.1f}% "
              f"sh{m_s.sharpe:>5.2f}  fut: ${m_f.final_balance:>9,.0f} "
              f"DD{m_f.max_drawdown_pct:>5.1f}% sh{m_f.sharpe:>5.2f}")
    out = pd.DataFrame(rows)
    OUT = ROOT / "reports" / "kelly_regime_v6"
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "select_inner_validation.csv", index=False)
    print(f"\nwritten: {OUT / 'select_inner_validation.csv'}")


def v4_baseline(start, end, market) -> None:
    m, vol, notional, res = measure(get_strategy(INCUMBENT), start, end, market=market)
    line(f"{INCUMBENT} (shipped defaults)", m, vol, notional, res)


# ------------------------------------------------------------------------ causality


def causality() -> None:
    """Step 6: by-hand two-opposite-tampers lookahead probe.

    Experiments get no CI protection (test_causality_strict.py parametrizes
    over the registry only). Same procedure as R-28/R-31/R-33: bars after a
    cut are multiplied by 3 in one copy, divided by 3 in another; every
    decision at or before the cut must be bit-identical. This is
    particularly important here because of the custom EWM/reindex/ffill/
    shift construction in ``prepare`` — exactly the kind of full-series
    statistic ROUTINE.md warns a truncation test alone will not catch if
    it is computed over the WHOLE series and applied to early rows. The
    check below compares the prepared columns directly, not just orders,
    for that reason.
    """
    # Restricted to strictly pre-2023 bars: this project's own convention
    # (R-28/R-31/R-33) runs this structural probe on the dataset tail
    # without treating it as a holdout read, but this session's brief is
    # explicit ("do not read or use any bar dated 2023-01-01 or later, for
    # ANY purpose"), so the probe is deliberately confined to the inner
    # splits rather than relying on that convention.
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

    strat_kwargs = dict(halflife_days=90.0, kelly_mult=0.5, max_leverage=2.0)

    def prepared(frame):
        return KellyRegimeV6StateKelly(**strat_kwargs).prepare(frame.copy())

    pa = prepared(up)
    pb = prepared(down)
    ok = True
    for col in ("target", "_frac", "_mu_state", "_var_state", "_kelly_f_state"):
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
        s = KellyRegimeV6StateKelly(**strat_kwargs)
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

    a = run_backtest(KellyRegimeV6StateKelly(**strat_kwargs), up.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    b = run_backtest(KellyRegimeV6StateKelly(**strat_kwargs), down.iloc[:cut + 1], FUTURES,
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

    Same venue (Bitfinex), same window as R-17/R-28/R-31/R-33, both spot and
    5x futures, candidate vs shipped v4 defaults as the control. Falsification
    rule (fixed before running): if the candidate's Sharpe/drawdown/return on
    ETH is not at least as good as v4's by a margin bigger than a token
    amount, or the candidate is visibly re-fit to BTC-specific history (much
    worse on ETH than on the BTC control run through this identical
    pipeline), this whole direction is treated as a failure, not just this
    check.
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
            cand = KellyRegimeV6StateKelly(**candidate_kwargs)
            m_c, vol_c, not_c, res_c = measure(cand, None, None, df=df, market=market)
            line("    kelly_regime_v6 (candidate)", m_c, vol_c, not_c, res_c)


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "sweep":
        sweep()
    elif choice == "select":
        select()
    elif choice == "causality":
        causality()
    elif choice == "eth":
        # The inner-validation-selected candidate (see select_inner_validation.csv):
        # halflife=365d, kelly_mult=0.25, max_leverage=2.0, stat_horizon=1 bar.
        default_cand = dict(halflife_days=365.0, kelly_mult=0.25, max_leverage=2.0,
                             stat_horizon_bars=1)
        eth(default_cand)
    else:
        print("usage: python experiments/kelly_regime_v6_state_kelly.py "
              "[sweep|select|causality|eth]")
