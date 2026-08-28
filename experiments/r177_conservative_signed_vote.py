#!/usr/bin/env python
"""R-177 conservative branch: sign-symmetric vote on kelly_regime_v4 (SIZE axis).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote into
``src/tradebot/strategies/`` only if it clears the promotion bar in
``experiments/r177_direction.md``.

Mechanism, one sentence (frozen, see ``experiments/r177_direction.md``'s
"Conservative branch" and ``r177_shared.py``'s module docstring --
neither may be edited by this file): ``target = signed_vote_frac(close) *
scale``, where ``signed_vote_frac`` is v4's own 3-anchor latched vote
(20/40/80-day anchors, 1% band, latching hysteresis), remapped from
``[0, 1]`` to ``[-1, 1]`` via ``2*frac - 1`` -- so a bear vote now shorts
instead of flattening -- and ``scale`` is v4's own continuous
volatility-targeting formula, ``min(target_vol / realized_vol,
max_leverage)`` with ``target_vol=0.55, max_leverage=2.0`` (the
``KellyRegime``/``kelly_regime.py`` base-class formula ``kelly_regime_v4``
inherits its constants from). Zero new parameters beyond what v4 already
exposes. Same 10% deadband, same latching hysteresis.

One disclosed implementation note, not a deviation: ``kelly_regime_v4``
is registered as a subclass of ``kelly_regime_v3``, which additionally
switches between this continuous formula ("full") and a slow-vol-anchored
variant ("steady") outside/inside a volatility-regime band. Both
``r177_direction.md`` and ``r177_shared.py`` describe "v4's own scale"
using the plain continuous formula above (matching ``kelly_regime.py``'s
literal ``KellyRegime.prepare()`` loop, not the extra full/steady state
machine ``kelly_regime_v3.prepare()`` layers on top) -- this is the same
simplification R-37's own novel branch (``kelly_regime_v6_state_kelly.py``,
whose docstring states plainly "answers 'how much should I hold' with one
GLOBAL number: scale = min(target_vol/realized_vol, max_leverage)") already
made when describing v4's scale for a SIZE-axis experiment. This file
follows that established, named precedent rather than re-deriving a new
convention: the anchors (20/40/80) and constants (target_vol, max_leverage,
vol_span, deadband) are v4's; the full/steady vol-extremity switch is v3's
own additional layer, not "v4's scale" as either frozen document describes
it, so it is not reproduced here. Flagged explicitly rather than silently
picked.

Usage::

    python experiments/r177_conservative_signed_vote.py train       # step 3
    python experiments/r177_conservative_signed_vote.py deadband    # step 3b
    python experiments/r177_conservative_signed_vote.py validate    # step 4
    python experiments/r177_conservative_signed_vote.py causality   # causality probe
    python experiments/r177_conservative_signed_vote.py stress      # falsification: MC stress
    python experiments/r177_conservative_signed_vote.py eth         # falsification: ETH
    python experiments/r177_conservative_signed_vote.py funding     # bonus: real funding, 2020-23
    python experiments/r177_conservative_signed_vote.py all         # everything, in order
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

from experiments.r177_shared import signed_vote_frac  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics, max_drawdown_pct  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


# =========================================================================
# The strategy
# =========================================================================


class R177ConservativeSignedVote(Strategy):
    """kelly_regime_v4's own vote, signed, times v4's own scale (R-177 conservative branch).

    ``frac`` in ``kelly_regime_v4`` lands on one of four discrete states,
    ``{0, 1/3, 2/3, 1}``, and only ever produces a long-or-flat position
    (``target = frac * scale >= 0``). This strategy replaces ``frac`` with
    ``r177_shared.signed_vote_frac`` -- the SAME three-anchor vote,
    remapped to ``{-1, -1/3, 1/3, 1}`` -- so the two vote states below the
    50% line (unanimous bear, and 1-of-3-anchors bullish) now carry
    negative sign instead of a small-to-zero long. Everything else
    (anchors, band, hysteresis, scale formula, deadband) is byte-identical
    to v4's own construction.
    """

    name = "r177_conservative_signed_vote"

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10) -> None:
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.warmup = max(horizons) * BARS_PER_DAY + 10

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # The ONLY change from kelly_regime_v4: signed instead of unsigned.
        frac = signed_vote_frac(close, horizons=self.horizons, band=self.band)

        # v4's own continuous vol-targeting scale, unmodified (see module
        # docstring for why this is the "full"/base formula, not v3's
        # extra full/steady switch).
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            v = vol[i]
            scale = min(self.target_vol / v, self.max_leverage) if np.isfinite(v) and v > 0 else 0.0
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["_frac_signed"] = frac
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures


# =========================================================================
# Harness (style follows experiments/kelly_regime_v6_state_kelly.py)
# =========================================================================

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")

INCUMBENT = "kelly_regime_v4"

N_EVALUATED = 0  # distinct configurations evaluated, for the deflated-Sharpe count
EVALUATED_CONFIGS: list[str] = []  # human-readable labels, deduplicated


def realized_exposure(result) -> np.ndarray:
    """Actual |notional|/equity held at each bar, reconstructed from FILLS.

    This is deliberately NOT the strategy's own ``target`` column: that
    column is the strategy's internally-desired (pre-broker-clamp) target,
    which for this candidate can be negative on spot even though the
    broker (``allow_short=False``) always clamps the executed position to
    >= 0 there. Reading exposure off ``target`` directly would silently
    count a clamped, never-executed short as real notional. Reconstructing
    from fills (the same pattern ``scripts/funding_study.py::timing`` uses)
    reports what was actually held.
    """
    price = result.df["close"].to_numpy(dtype=float)
    equity = result.equity.to_numpy(dtype=float)
    n = len(price)
    pos = np.zeros(n)
    offset = {ts: i for i, ts in enumerate(result.df.index)}
    running, last = 0.0, 0
    for f in result.fills:
        i = offset.get(f.ts)
        if i is None:
            continue
        pos[last:i] = running
        running += f.qty if f.side.name == "BUY" else -f.qty
        last = i
    pos[last:] = running
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(equity > 0, np.abs(pos) * price / np.maximum(equity, 1e-9), 0.0)


def mean_notional(result) -> float:
    if len(result.df) == 0:
        return float("nan")
    return float(np.mean(realized_exposure(result)))


def realized_vol(equity) -> float:
    eq = equity.to_numpy(dtype=float) if hasattr(equity, "to_numpy") else np.asarray(equity)
    if len(eq) < 3:
        return float("nan")
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
    return float(rets.std(ddof=1) * np.sqrt(BARS_PER_YEAR))


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0,
            count_label: str | None = None):
    """One backtest -> (metrics, realized vol, mean notional, result).

    ``count_label`` increments the trials counter exactly once per
    distinct label (a parameter configuration), no matter how many
    markets/splits it is subsequently scored on -- the same convention
    R-33/R-37 use ("configurations" vs raw "backtests").
    """
    global N_EVALUATED
    if count_label is not None and count_label not in EVALUATED_CONFIGS:
        EVALUATED_CONFIGS.append(count_label)
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                         start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    return m, realized_vol(result.equity), mean_notional(result), result


def line(tag, m, vol, notional, result):
    fills = len(result.fills)
    print(f"  {tag:42s} final=${m.final_balance:>11,.0f} "
          f"vol={vol:5.3f} notional={notional:5.3f} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>4d} fills={fills:>5d} "
          f"fees=${m.fees_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# =========================================================================
# Step 3 -- train-only sanity check (the frozen candidate, deadband=0.10)
# =========================================================================


def step3_train() -> None:
    print("=" * 78)
    print("STEP 3 -- inner-train (2017-01-01 -> 2020-12-31), frozen defaults")
    print("=" * 78)
    cand = R177ConservativeSignedVote()
    for mname, market in MARKETS:
        m_c, vol_c, not_c, res_c = measure(cand, *TRAIN, market=market,
                                            count_label="deadband=0.10")
        line(f"[{mname}] r177_conservative (deadband=0.10)", m_c, vol_c, not_c, res_c)
        m_v, vol_v, not_v, res_v = measure(get_strategy(INCUMBENT), *TRAIN, market=market)
        line(f"[{mname}] {INCUMBENT} (control)", m_v, vol_v, not_v, res_v)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


# =========================================================================
# Step 3b -- does the deadband itself need widening now the vote can cross
# zero? Sweep on inner-train ONLY, futures_5x ONLY.
# =========================================================================

DEADBAND_GRID = (0.10, 0.15, 0.20, 0.25, 0.30)


def step3b_deadband_sweep() -> pd.DataFrame:
    print("=" * 78)
    print("STEP 3b -- deadband sweep, inner-train, futures_5x only")
    print("=" * 78)
    rows = []
    for db in DEADBAND_GRID:
        cand = R177ConservativeSignedVote(deadband=db)
        m, vol, notional, res = measure(cand, *TRAIN, market=FUTURES,
                                         count_label=f"deadband={db:.2f}")
        rows.append({"deadband": db, "final": m.final_balance, "vol": vol,
                     "notional": notional, "max_dd": m.max_drawdown_pct,
                     "sharpe": m.sharpe, "trades": m.num_trades,
                     "fills": len(res.fills), "fees": m.fees_paid,
                     "liquidated": m.liquidated})
        print(f"  deadband={db:.2f}  final=${m.final_balance:>10,.0f} "
              f"vol={vol:5.3f} notional={notional:5.3f} DD={m.max_drawdown_pct:>5.1f}% "
              f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>4d} fills={len(res.fills):>5d}"
              f"{'  LIQUIDATED' if m.liquidated else ''}")
    out = pd.DataFrame(rows)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    return out


def pick_deadband_alternate(sweep_df: pd.DataFrame) -> float:
    """Report-only heuristic, applied AFTER the sweep is printed: the
    widest deadband whose Sharpe does not fall more than 0.05 below the
    default's while cutting fill-count turnover by at least 15%. If none
    qualifies, the default itself is returned (no alternate is worth
    carrying forward) and this is stated plainly, not hidden.
    """
    base = sweep_df[np.isclose(sweep_df["deadband"], 0.10)].iloc[0]
    best = 0.10
    for _, row in sweep_df.iterrows():
        if row["deadband"] <= 0.10:
            continue
        sharpe_ok = row["sharpe"] >= base["sharpe"] - 0.05
        turnover_cut = row["fills"] <= base["fills"] * 0.85
        if sharpe_ok and turnover_cut:
            best = float(row["deadband"])
    return best


# =========================================================================
# Step 4 -- inner-validation: frozen candidate(s) vs v4, futures_5x + spot
# identity check
# =========================================================================


def step4_validate(alt_deadband: float | None) -> None:
    print("=" * 78)
    print("STEP 4 -- inner-validation (2021-01-01 -> 2022-12-31)")
    print("=" * 78)

    deadbands = [0.10] if alt_deadband is None or np.isclose(alt_deadband, 0.10) \
        else [0.10, alt_deadband]

    for mname, market in MARKETS:
        print(f"\n-- {mname} --")
        m_v, vol_v, not_v, res_v = measure(get_strategy(INCUMBENT), *VALID, market=market)
        line(f"{INCUMBENT} (control)", m_v, vol_v, not_v, res_v)
        for db in deadbands:
            cand = R177ConservativeSignedVote(deadband=db)
            m_c, vol_c, not_c, res_c = measure(cand, *VALID, market=market)
            line(f"r177_conservative (deadband={db:.2f})", m_c, vol_c, not_c, res_c)
            if mname == "futures":
                d_sharpe = m_c.sharpe - m_v.sharpe
                d_vol = vol_c / vol_v - 1.0 if vol_v else float("nan")
                d_notional = not_c / not_v - 1.0 if not_v else float("nan")
                print(f"    ΔSharpe={d_sharpe:+.3f}  Δvol/v4={d_vol:+.1%}  "
                      f"Δnotional/v4={d_notional:+.1%}")

    # Identity check: on spot, does the signed vote ever actually reproduce
    # v4's own trajectory? Checked directly on the underlying arrays, not
    # asserted -- report whatever is actually true.
    print("\n-- spot identity check (byte-level) --")
    cand = R177ConservativeSignedVote()
    lo = int(DF.index.searchsorted(VALID[0]))
    hi = int(DF.index.searchsorted(VALID[1], side="right"))
    prefix = min(lo, cand.warmup)
    frame_c = cand.prepare(DF.iloc[lo - prefix: hi].copy())["target"].to_numpy()
    v4 = get_strategy(INCUMBENT)
    prefix_v4 = min(lo, v4.warmup)
    frame_v4 = v4.prepare(DF.iloc[lo - prefix_v4: hi].copy())["target"].to_numpy()
    # Align both series to the validation window itself.
    frame_c = frame_c[prefix:]
    frame_v4 = frame_v4[prefix_v4:]
    n = min(len(frame_c), len(frame_v4))
    frame_c, frame_v4 = frame_c[:n], frame_v4[:n]
    clipped_c = np.clip(frame_c, 0.0, None)  # what spot's broker would clamp *this bar's* target to
    frac_diff = float(np.mean(~np.isclose(clipped_c, frame_v4, atol=1e-9)))
    max_diff = float(np.max(np.abs(clipped_c - frame_v4)))
    print(f"  fraction of bars where clip(candidate_target,0,None) != v4_target: "
          f"{frac_diff:.4%}  (max abs diff where they differ: {max_diff:.4f})")
    print("  (a nonzero fraction here means the internal hysteresis states diverge "
          "whenever the signed vote goes negative in the '1-of-3 anchors bullish' "
          "state, where v4 itself still holds a small LONG rather than flat -- "
          "see the full report for what this does to the final backtest numbers.)")


# =========================================================================
# Causality probe (adapted from kelly_regime_v6_state_kelly.py::causality)
# =========================================================================


def causality() -> None:
    print("=" * 78)
    print("CAUSALITY PROBE -- two opposite tampers of the post-cut future")
    print("=" * 78)
    pre_holdout = DF.loc[:"2022-12-31"]
    df = pre_holdout.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def prepared(frame):
        return R177ConservativeSignedVote().prepare(frame.copy())

    pa = prepared(up)
    pb = prepared(down)
    ok = True
    for col in ("target", "_frac_signed"):
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
        s = R177ConservativeSignedVote()
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

    a = run_backtest(R177ConservativeSignedVote(), up.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    b = run_backtest(R177ConservativeSignedVote(), down.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-6
    print(f"  max |equity difference| before the cut = {worst_eq:.3e}  "
          f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")

    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# =========================================================================
# Falsification test 1 -- Monte Carlo stress windows (scripts/stress_test.py)
#
# scripts/stress_test.py is registry-name-based (get_strategy(name));
# this strategy is deliberately unregistered, so its window-generation
# formula and per-window evaluation logic are reproduced verbatim,
# parameterized on a strategy INSTANCE -- same convention as
# experiments/r153_novel_cdar_budget.py's evaluate_instance/
# run_stress_battery.
# =========================================================================


def evaluate_instance(strategy, window: pd.DataFrame, eval_start: int,
                       market: MarketSpec, balance: float = 1_000.0) -> dict:
    """Identical body to scripts/stress_test.py's own ``evaluate()``."""
    result = run_backtest(strategy, window, market, balance, trade_start=eval_start)
    equity = result.equity.to_numpy(dtype=float)
    base = equity[eval_start]
    if not np.isfinite(base) or base <= 0:
        return {"return_pct": -100.0, "max_dd_pct": 100.0, "trades": 0, "liquidated": True}
    seg = equity[eval_start:]
    start_ts = window.index[eval_start]
    return {
        "return_pct": 100.0 * (seg[-1] / base - 1.0),
        "max_dd_pct": max_drawdown_pct(seg),
        "trades": sum(1 for t in result.trades if t.entry_ts >= start_ts),
        "liquidated": result.liquidated,
    }


def run_stress_battery(trials: int = 40, min_days: int = 90, max_days: int = 730,
                        seed: int = 42, market: MarketSpec = FUTURES) -> pd.DataFrame:
    """Identical window-generation formula to scripts/stress_test.py's own
    ``run()`` (same rng calls, same warmup/trade_start discipline)."""
    factories = {
        "r177_conservative": lambda: R177ConservativeSignedVote(),
        "kelly_regime_v4": lambda: get_strategy("kelly_regime_v4"),
        "buy_and_hold": lambda: get_strategy("buy_and_hold"),
    }
    warmup = max(f().warmup for f in factories.values()) + 10
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(trials):
        length = int(rng.integers(min_days, max_days + 1) * BARS_PER_DAY)
        start = int(rng.integers(warmup, len(DF) - length))
        window = DF.iloc[start - warmup: start + length]
        eval_start = warmup
        print(f"[{k + 1}/{trials}] {window.index[eval_start]:%Y-%m-%d} "
              f"+{length // BARS_PER_DAY}d", file=sys.stderr)
        for name, factory in factories.items():
            stats = evaluate_instance(factory(), window, eval_start, market)
            rows.append({"trial": k, "days": length // BARS_PER_DAY, "strategy": name, **stats})
    return pd.DataFrame(rows)


def summarize_stress(res: pd.DataFrame) -> pd.DataFrame:
    out = []
    bench = res[res.strategy == "buy_and_hold"].set_index("trial")["return_pct"]
    for name, grp in res.groupby("strategy", sort=False):
        g = grp.set_index("trial")
        beat = (g["return_pct"] > bench.loc[g.index]).mean() * 100.0
        out.append({
            "strategy": name,
            "median return %": g["return_pct"].median(),
            "mean return %": g["return_pct"].mean(),
            "profitable %": (g["return_pct"] > 0).mean() * 100.0,
            "beat hold %": beat,
            "worst %": g["return_pct"].min(),
            "median maxDD %": g["max_dd_pct"].median(),
            "worst maxDD %": g["max_dd_pct"].max(),
            "liquidated %": g["liquidated"].mean() * 100.0,
        })
    return pd.DataFrame(out)


def falsification_stress() -> None:
    print("=" * 78)
    print("FALSIFICATION TEST 1 -- Monte Carlo stress windows, futures_5x, 40 trials")
    print("=" * 78)
    res = run_stress_battery()
    OUT = ROOT / "reports" / "r177_conservative"
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "stress_results.csv", index=False)
    summary = summarize_stress(res)
    summary.to_csv(OUT / "stress_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(2).to_string(index=False))
    print(f"\nwritten: {OUT / 'stress_summary.csv'}")


# =========================================================================
# Falsification test 2 -- ETH (Bitfinex), same construction as
# kelly_regime_v6_state_kelly.py::eth
# =========================================================================


def falsification_eth() -> None:
    print("=" * 78)
    print("FALSIFICATION TEST 2 -- ETH Bitfinex vs BTC control")
    print("=" * 78)
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
            cand = R177ConservativeSignedVote()
            m_c, vol_c, not_c, res_c = measure(cand, None, None, df=df, market=market)
            line("    r177_conservative (candidate)", m_c, vol_c, not_c, res_c)


# =========================================================================
# Bonus (not required by the falsification rule, but directly relevant to
# the pre-registered funding risk): real BTC perp funding, 2020-2023.
# =========================================================================


def funding_check() -> None:
    print("=" * 78)
    print("BONUS -- real funding charged, 2020-01-01 -> 2023-12-31, futures_5x")
    print("=" * 78)
    real = load_funding(ROOT / "data")
    if real is None:
        print("  no funding data committed; skipping")
        return
    start, end = "2020-01-01", "2023-12-31"
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    for name, strat_factory in (
        (INCUMBENT, lambda: get_strategy(INCUMBENT)),
        ("r177_conservative", lambda: R177ConservativeSignedVote()),
    ):
        strat = strat_factory()
        pre = min(lo, strat.warmup)
        raw_free = run_backtest(strat, DF.iloc[lo - pre: hi], FUTURES, 1_000.0,
                                 trade_start=pre, data_label=LABEL)
        strat2 = strat_factory()
        raw_paid = run_backtest(strat2, DF.iloc[lo - pre: hi], FUTURES, 1_000.0,
                                 trade_start=pre, funding=real, data_label=LABEL)
        free = compute_metrics(raw_free)
        paid = compute_metrics(raw_paid)
        print(f"  {name:24s} funding-free=${free.final_balance:>10,.0f}  "
              f"with-funding=${paid.final_balance:>10,.0f}  "
              f"cost={paid.final_balance / free.final_balance - 1:>+6.1%}  "
              f"funding_paid=${raw_paid.funding_paid:>+9,.0f}")


# =========================================================================
# main
# =========================================================================


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    t0 = time.time()

    sweep_df = None
    alt = None

    if choice in ("train", "all"):
        step3_train()
    if choice in ("deadband", "all"):
        sweep_df = step3b_deadband_sweep()
        alt = pick_deadband_alternate(sweep_df)
        print(f"\nplateau-check alternate carried into step 4: deadband={alt:.2f}"
              f"{' (== default, no alternate clears the bar)' if np.isclose(alt, 0.10) else ''}")
    if choice in ("validate", "all"):
        if sweep_df is None:
            sweep_df = step3b_deadband_sweep()
            alt = pick_deadband_alternate(sweep_df)
        step4_validate(alt)
    if choice in ("causality", "all"):
        causality()
    if choice in ("stress", "all"):
        falsification_stress()
    if choice in ("eth", "all"):
        falsification_eth()
    if choice in ("funding", "all"):
        funding_check()
    if choice not in ("train", "deadband", "validate", "causality", "stress", "eth",
                       "funding", "all"):
        print("usage: python experiments/r177_conservative_signed_vote.py "
              "[train|deadband|validate|causality|stress|eth|funding|all]")

    print(f"\n[{time.time() - t0:.0f}s]  total configurations evaluated: {N_EVALUATED}")
