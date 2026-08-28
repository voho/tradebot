#!/usr/bin/env python
"""R-177 NOVEL branch (08-28): the unfloored per-state Kelly ratio, used as a
SIGNED target directly -- the specific alternative R-37's own novel branch
(`experiments/kelly_regime_v6_state_kelly.py`) built the machinery for and
declined to run ("never short a state; mirrors v4").

See `experiments/r177_direction.md` ("Novel branch") for the frozen
mechanism, non-duplication argument and falsification rule; this file only
executes that rule and records the resulting numbers. Does NOT edit
`experiments/r177_shared.py` or any `r177_conservative_*` file (separate
branch, separate agent, per this project's parallel-branch convention).

Mechanism, one sentence (frozen, unchanged from the pre-registration):
``target = clip(kelly_mult * kelly_f_state, -max_leverage, max_leverage)``,
where ``kelly_f_state`` comes from `r177_shared.state_kelly_stats` with
``floor_at_zero=False`` -- i.e. R-37's exact causal per-vote-state
`mu_state/sigma_state**2` estimator, with the one-line floor R-37's own
docstring names removed, used DIRECTLY as the signed position (clipped to
the leverage cap). Unlike the conservative branch and unlike R-37's own
`kelly_regime_v6_state_kelly.py`, the vote's magnitude (`frac`) does NOT
additionally multiply the result here -- the vote's only remaining job is
to SELECT which state's own (mu, sigma**2) estimate is active at each bar.
Same 10% deadband, same `min_obs=2,000` occurrence floor, same
halflife/kelly_mult/stat_horizon grid R-37 swept.

Shorting is only mechanically possible on `futures_5x`
(`MarketSpec.futures(allow_short=True)`); this round's own disclosed scope
narrowing restricts the actual test to that one market. Spot is reported
only as a minor descriptive control -- and, disclosed explicitly here
because it is easy to misread, it is NOT expected to bit-for-bit
reproduce unmodified `kelly_regime_v4` on spot the way the CONSERVATIVE
branch's spot run is: this branch replaces v4's entire sizing formula
(`target_vol/realized_vol`) with a structurally different one
(per-state mu/sigma**2), so the two coincide on spot only to the extent
the two formulas happen to agree in magnitude on the bull states, which
there is no reason to expect. What IS guaranteed on spot is that the
sign never binds negative (the broker clips the lower target bound to
0.0 when `allow_short=False`), so spot is a long-only variant of this
branch's own mechanism, not a reproduction of v4.

Pre-registered failure modes (named in `r177_direction.md`, repeated here
verbatim before any real-data number in this file was read):
(a) R-37's own already-measured data-hunger problem (non-monotone
    kelly_mult response, a fitted 330-450d halflife peak rather than a
    plateau, failure on the BTC control once routed through ETH's
    shorter window) inherited verbatim, now on both signs at once;
(b) the bear-state estimate, unfloored, is even noisier than the
    already-fragile positive states R-37 measured, because bear episodes
    are rarer in this project's own data than bull episodes;
(c) the same whipsaw/funding risk named for the conservative branch, now
    sized by a fitted magnitude rather than a fixed vote fraction, which
    could make the failure mode either better-controlled (smaller in the
    least reliable state) or worse (larger if the estimator is fooled) --
    named as genuinely undetermined in advance.

Falsification rule, frozen (`r177_direction.md`, applies to both
branches): promote only if ALL of -- beats `buy_and_hold` OOS on
`futures_5x` after real costs; the improvement clears the +/-0.2 Sharpe
noise floor or is a genuine risk-matched (R-33) drawdown/tail
improvement; survives the stress-test/ETH falsification (does not fail
more on it than it gains on the primary comparison); the parameter
neighbourhood is a plateau, not a peak. Anything else is NEGATIVE.

Usage
-----
    python experiments/r177_novel_state_kelly_short.py sweep       # step 3
    python experiments/r177_novel_state_kelly_short.py select      # step 4
    python experiments/r177_novel_state_kelly_short.py causality   # causality probe
    python experiments/r177_novel_state_kelly_short.py eth         # ETH + BTC-control falsification
    python experiments/r177_novel_state_kelly_short.py stress      # Monte Carlo stress windows
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
from tradebot.metrics import compute_metrics, max_drawdown_pct  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r177_shared import (  # noqa: E402
    STATE_LABELS,
    STATES,
    state_kelly_stats,
    unsigned_vote_frac,
)

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


# --------------------------------------------------------------------- strategy


class R177NovelStateKellyShort(Strategy):
    """v4's exact vote SELECTS a state; the state's own unfloored Kelly ratio IS the signed target.

    Everything about the vote (horizons, band, latching hysteresis) is
    `r177_shared.unsigned_vote_frac`, itself a verbatim copy of v4's own
    vote. `r177_shared.state_kelly_stats(..., floor_at_zero=False)` is
    R-37's exact causal per-state mu/sigma**2 estimator with the one-line
    floor removed. The only change from R-37's own
    `kelly_regime_v6_state_kelly.py` is this class's `prepare()`: instead
    of `target = frac * min(kelly_mult * clip(kelly_f, 0, None), max_lev)`
    this class uses `target = clip(kelly_mult * kelly_f, -max_lev,
    max_lev)` directly -- no floor, no `frac` multiplier.
    """

    name = "r177_novel_state_kelly_short"

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
        # Same warmup convention as kelly_regime_v6_state_kelly.py: scales
        # with the half-life under test so inner-validation (which does
        # have pre-period history) starts with the per-state EWMs already
        # populated.
        self.warmup = max(80 * BARS_PER_DAY + 10, int(3 * self.halflife_days * BARS_PER_DAY))

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        n = len(df)

        frac = unsigned_vote_frac(close, horizons=self.horizons, band=self.band)
        stats = state_kelly_stats(close, frac, halflife_days=self.halflife_days,
                                   min_obs=self.min_obs,
                                   stat_horizon_bars=self.stat_horizon_bars,
                                   floor_at_zero=False)
        kelly_f = stats["kelly_f"]
        enough = stats["count"] >= self.min_obs

        scale = np.clip(self.kelly_mult * kelly_f, -self.max_leverage, self.max_leverage)
        scale = np.where(enough, scale, 0.0)

        # target = the state-conditional signed Kelly scale DIRECTLY -- no
        # `frac *` multiplier (that is the one-line difference from R-37's
        # own v6 construction, besides the floor). Same 10% deadband.
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = scale[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["_frac"] = frac
        df["_mu_state"] = stats["mu"]
        df["_var_state"] = stats["var"]
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
MARKETS = (("spot", SPOT), ("futures_5x", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")

INCUMBENT = "kelly_regime_v4"

# The inner-validation-selected candidate (see select_inner_validation.csv):
# halflife=365d, kelly_mult=0.25, max_leverage=2.0, stat_horizon=1 bar --
# the same corner of the grid R-37's own novel branch selected, chosen here
# for direct comparability (lowest, closest-to-matched exposure within the
# one plateau-like region of the grid; see the ledger entry for the full
# plateau discussion).
DEFAULT_CAND = dict(halflife_days=365.0, kelly_mult=0.25, max_leverage=2.0,
                     stat_horizon_bars=1)

N_EVALUATED = 0  # distinct configurations searched, for the deflated-Sharpe trials count


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
    """Step 3: sweep the grid on inner-train, futures_5x ONLY (shorting requires it)."""
    rows = []
    t0 = time.time()
    for cfg in grid_configs():
        strat = R177NovelStateKellyShort(**cfg)
        m, vol, notional, res = measure(strat, *TRAIN, market=FUTURES, count=True)
        rows.append({**cfg, "final": m.final_balance, "vol": vol,
                     "notional": notional, "max_dd": m.max_drawdown_pct,
                     "sharpe": m.sharpe, "trades": m.num_trades,
                     "fees": m.fees_paid, "liquidated": m.liquidated})
        print(f"[{N_EVALUATED:>3d}] hl={cfg['halflife_days']:>5.0f}d "
              f"km={cfg['kelly_mult']:.2f} sh={cfg['stat_horizon_bars']:>3d}bars  "
              f"final=${m.final_balance:>10,.0f} vol={vol:5.3f} notional={notional:5.3f} "
              f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
              f"{'LIQ ' if m.liquidated else ''}[{time.time() - t0:.0f}s]")
    df = pd.DataFrame(rows)
    OUT = ROOT / "reports" / "r177_novel_state_kelly_short"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "sweep_inner_train.csv", index=False)
    print(f"\nconfigurations evaluated (step 3): {N_EVALUATED}")
    print(f"written: {OUT / 'sweep_inner_train.csv'}")
    return df


# --------------------------------------------------------------------------- step 4


def state_stats_snapshot(strategy: R177NovelStateKellyShort, start, end) -> None:
    """Print the actual measured mu_state / sigma_state**2 / kelly_f by vote state.

    Mirrors kelly_regime_v6_state_kelly.py's own state_stats_snapshot: the
    empirical claim under test ("do the states genuinely differ, and does
    the bear side stay negative") reported directly off the strategy's own
    prepared columns, independent of the strategy-level backtest result.
    """
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    prefix = min(lo, strategy.warmup)
    frame = strategy.prepare(DF.iloc[lo - prefix: hi].copy())
    frame = frame.iloc[prefix:]
    print(f"  state-conditional stats, {start} -> {end} "
          f"(halflife={strategy.halflife_days:.0f}d, kelly_mult={strategy.kelly_mult:.2f}, "
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
    """Step 4: score candidates on inner-validation, futures_5x, plateau view.

    Also reports spot as a descriptive control (NOT expected to reproduce
    v4 -- see module docstring).
    """
    if candidates is None:
        candidates = list(grid_configs())
    rows = []
    for cfg in candidates:
        strat_fut = R177NovelStateKellyShort(**cfg)
        strat_spot = R177NovelStateKellyShort(**cfg)
        m_f, vol_f, not_f, res_f = measure(strat_fut, *VALID, market=FUTURES)
        m_s, vol_s, not_s, res_s = measure(strat_spot, *VALID, market=SPOT)
        rows.append({**cfg, "fut_final": m_f.final_balance, "fut_dd": m_f.max_drawdown_pct,
                     "fut_sharpe": m_f.sharpe, "fut_trades": m_f.num_trades,
                     "fut_vol": vol_f, "fut_notional": not_f, "fut_liq": m_f.liquidated,
                     "spot_final": m_s.final_balance, "spot_dd": m_s.max_drawdown_pct,
                     "spot_sharpe": m_s.sharpe, "spot_trades": m_s.num_trades,
                     "spot_vol": vol_s, "spot_notional": not_s})
        print(f"hl={cfg['halflife_days']:>5.0f}d km={cfg['kelly_mult']:.2f} "
              f"sh={cfg['stat_horizon_bars']:>3d}  "
              f"fut: ${m_f.final_balance:>9,.0f} DD{m_f.max_drawdown_pct:>5.1f}% "
              f"sh{m_f.sharpe:>5.2f} notional={not_f:5.3f} vol={vol_f:5.3f}"
              f"{' LIQ' if m_f.liquidated else ''}   "
              f"spot: ${m_s.final_balance:>9,.0f} sh{m_s.sharpe:>5.2f}")
    out = pd.DataFrame(rows)
    OUT = ROOT / "reports" / "r177_novel_state_kelly_short"
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "select_inner_validation.csv", index=False)
    print(f"\nwritten: {OUT / 'select_inner_validation.csv'}")


def v4_baseline(start, end, market) -> None:
    m, vol, notional, res = measure(get_strategy(INCUMBENT), start, end, market=market)
    line(f"{INCUMBENT} (shipped defaults)", m, vol, notional, res)


# ------------------------------------------------------------------------ causality


def causality() -> None:
    """By-hand two-opposite-tampers lookahead probe, mirroring
    kelly_regime_v6_state_kelly.py's own `causality()` exactly: bars after a
    cut are multiplied by 3 (volume by 7) in one copy, divided by 3 (volume
    by 7) in another; every prepared column and every order at or before the
    cut must be bit-identical. Particularly important here because the
    extra reindex/ffill/shift chain in `state_kelly_stats` is exactly the
    kind of full-series statistic a truncation test alone will not catch if
    it silently used the whole series. Confined to strictly pre-2023 bars,
    per this round's own no-holdout-read discipline.
    """
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
        return R177NovelStateKellyShort(**strat_kwargs).prepare(frame.copy())

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
        s = R177NovelStateKellyShort(**strat_kwargs)
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

    a = run_backtest(R177NovelStateKellyShort(**strat_kwargs), up.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    b = run_backtest(R177NovelStateKellyShort(**strat_kwargs), down.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-6
    print(f"  max |equity difference| before the cut = {worst_eq:.3e}  "
          f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")

    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# ------------------------------------------------------------------------------ eth


def eth(candidate_kwargs: dict) -> None:
    """Pre-registered falsification -- does the candidate hold on ETH, and
    does the identical pipeline still lose money on the BTC control (R-37's
    own overfitting tell)? Same venue (Bitfinex) as R-17/R-28/R-31/R-33/R-37,
    both spot and 5x futures, candidate vs shipped v4 defaults as the
    control.
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
            cand = R177NovelStateKellyShort(**candidate_kwargs)
            m_c, vol_c, not_c, res_c = measure(cand, None, None, df=df, market=market)
            line("    r177_novel_state_kelly_short (candidate)", m_c, vol_c, not_c, res_c)


# --------------------------------------------------------------------------- stress


def stress(candidate_kwargs: dict, trials: int = 40, min_days: int = 90, max_days: int = 730,
           seed: int = 42) -> pd.DataFrame:
    """Monte Carlo window stress test, adapting scripts/stress_test.py's own
    window-generation logic directly (that script's `run()` takes registered
    strategy NAMEs from the registry, which this experiment's candidate is
    deliberately not — so the windows/eval-start/warmup-prefix machinery is
    reproduced here rather than imported). futures_5x only, candidate vs v4
    vs buy_and_hold, LIQUIDATION RATE reported explicitly across windows, not
    just the point estimate, per the pre-registered rule in r177_direction.md.
    """
    names = ["r177_novel_state_kelly_short", INCUMBENT, "buy_and_hold"]
    cand = R177NovelStateKellyShort(**candidate_kwargs)
    strategies = {"r177_novel_state_kelly_short": cand,
                  INCUMBENT: get_strategy(INCUMBENT),
                  "buy_and_hold": get_strategy("buy_and_hold")}

    warmup = max(s.warmup for s in strategies.values()) + 10
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(trials):
        length = int(rng.integers(min_days, max_days + 1) * BARS_PER_DAY)
        start = int(rng.integers(warmup, len(DF) - length))
        specs.append((start, length))

    rows = []
    for k, (start, length) in enumerate(specs, 1):
        window = DF.iloc[start - warmup: start + length]
        eval_start = warmup
        print(f"[{k}/{trials}] {window.index[eval_start]:%Y-%m-%d} "
              f"+{length // BARS_PER_DAY}d", file=sys.stderr)
        for name in names:
            # Fresh strategy instance per window: the candidate carries
            # per-state EWM state that must not leak across windows.
            strat = (R177NovelStateKellyShort(**candidate_kwargs)
                     if name == "r177_novel_state_kelly_short" else get_strategy(name))
            result = run_backtest(strat, window, FUTURES, 1_000.0, trade_start=eval_start)
            equity = result.equity.to_numpy(dtype=float)
            base = equity[eval_start]
            if not np.isfinite(base) or base <= 0:
                stats = {"return_pct": -100.0, "max_dd_pct": 100.0, "trades": 0,
                         "liquidated": True}
            else:
                seg = equity[eval_start:]
                start_ts = window.index[eval_start]
                stats = {
                    "return_pct": 100.0 * (seg[-1] / base - 1.0),
                    "max_dd_pct": max_drawdown_pct(seg),
                    "trades": sum(1 for t in result.trades if t.entry_ts >= start_ts),
                    "liquidated": result.liquidated,
                }
            rows.append({"trial": k, "strategy": name,
                         "start": window.index[eval_start], "days": length // BARS_PER_DAY,
                         **stats})
    res = pd.DataFrame(rows)
    OUT = ROOT / "reports" / "r177_novel_state_kelly_short"
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "stress_results.csv", index=False)

    bench = res[res.strategy == "buy_and_hold"].set_index("trial")["return_pct"]
    print(f"\n{'strategy':32s} {'median ret%':>12s} {'mean ret%':>10s} "
          f"{'beat hold%':>11s} {'median DD%':>11s} {'worst DD%':>10s} {'liq rate%':>10s}")
    for name in names:
        grp = res[res.strategy == name]
        beat = (grp.set_index("trial")["return_pct"] > bench).mean() * 100.0
        print(f"{name:32s} {grp['return_pct'].median():>12.1f} {grp['return_pct'].mean():>10.1f} "
              f"{beat:>11.1f} {grp['max_dd_pct'].median():>11.1f} {grp['max_dd_pct'].max():>10.1f} "
              f"{grp['liquidated'].mean() * 100.0:>10.1f}")
    return res


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
        eth(DEFAULT_CAND)
    elif choice == "stress":
        stress(DEFAULT_CAND)
    else:
        print("usage: python experiments/r177_novel_state_kelly_short.py "
              "[sweep|select|causality|eth|stress]")
