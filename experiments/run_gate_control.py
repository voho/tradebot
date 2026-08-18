#!/usr/bin/env python
"""Driver for the ungated control and the gate replication (ledger R-32).

Splits follow ROUTINE.md step 3::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              step 4 only, pre-registered

Usage::

    python experiments/run_gate_control.py frontier    # the sweep, inner split
    python experiments/run_gate_control.py match       # freeze the multipliers
    python experiments/run_gate_control.py causality   # by-hand lookahead check
    python experiments/run_gate_control.py holdout     # step 4, frozen configs
    python experiments/run_gate_control.py inference   # paired intervals
    python experiments/run_gate_control.py eth         # falsification test
    python experiments/run_gate_control.py costs       # 0.40% tier, funding
    python experiments/run_gate_control.py windows     # 40-window resample
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.gate_control import GatedKelly  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import (DAYS_PER_YEAR, daily_returns,  # noqa: E402
                                max_drawdown_from_returns, paired_bootstrap,
                                stationary_bootstrap_indices, total_log_return)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS = ("2023-01-01", None)

# The multiplier grid swept on the inner split. Every (gate, multiplier)
# pair is one configuration, counted for the deflated Sharpe.
GRID = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)

N_EVALUATED = 0


# ------------------------------------------------------------------- reporting

def realized_vol(equity: pd.Series) -> float:
    """Annualized volatility of the equity curve, from daily returns."""
    r = daily_returns(equity).to_numpy()
    return float(np.std(r, ddof=1) * np.sqrt(DAYS_PER_YEAR)) if len(r) > 2 else float("nan")


def describe(result, tag: str) -> dict:
    m = compute_metrics(result)
    vol = realized_vol(result.equity)
    exposure = float(np.abs(result.df["target"].to_numpy()).mean()) \
        if "target" in result.df.columns else float("nan")
    row = {"tag": tag, "market": result.market.name, "final": m.final_balance,
           "log_growth": float(np.log(max(m.final_balance, 1e-9) / result.start_balance)),
           "vol": vol, "dd": m.max_drawdown_pct, "sharpe": m.sharpe,
           "fills": len(result.fills), "fees": m.fees_paid,
           "exposure": exposure, "liquidated": m.liquidated}
    print(f"  {tag:30s} {result.market.name:9s} final=${m.final_balance:>11,.0f} "
          f"vol={vol:>5.2f} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} ret/vol={row['log_growth'] / vol if vol > 0 else 0:>5.2f} "
          f"exp={exposure:>4.2f} fills={len(result.fills):>5d} "
          f"fees=${m.fees_paid:>7,.0f}{'  LIQUIDATED' if m.liquidated else ''}")
    return row


def ev(strategy, start, end, *, df=None, market=SPOT, tag="", balance=1_000.0,
       count=True) -> dict:
    """One backtest, one line, counted."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    return describe(result, tag or strategy.name)


def _benchmarks(start, end, market, label) -> None:
    print(f"\n{label} benchmarks:")
    for name in ("buy_and_hold", "kelly_regime_v4"):
        ev(get_strategy(name), start, end, market=market, tag=f"  {name}",
           count=False)


# ---------------------------------------------------------------- the frontier

def _sweep(start, end, market) -> pd.DataFrame:
    rows = []
    for gate in ("none", "vote", "evidence"):
        for m in GRID:
            row = ev(GatedKelly(gate=gate, multiplier=m), start, end,
                     market=market, tag=f"{gate:8s} x{m:g}")
            rows.append({**row, "gate": gate, "mult": m})
    return pd.DataFrame(rows)


def _interp_at(frame: pd.DataFrame, x: str, y: str, at: float) -> float:
    """Linear interpolation of ``y`` against a monotone-sorted ``x``."""
    sub = frame.dropna(subset=[x, y]).sort_values(x)
    if len(sub) < 2 or at < sub[x].iloc[0] or at > sub[x].iloc[-1]:
        return float("nan")
    return float(np.interp(at, sub[x].to_numpy(), sub[y].to_numpy()))


def _compare_at_matched_risk(frame: pd.DataFrame, risk: str = "vol") -> None:
    """Read each gate's frontier at the risk levels all of them cover."""
    arms = {g: frame[frame.gate == g] for g in ("none", "vote", "evidence")}
    lo = max(a[risk].min() for a in arms.values())
    hi = min(a[risk].max() for a in arms.values())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        print(f"  no overlapping {risk} range across the three gates")
        return
    print(f"\n  log growth at matched {risk} (overlap {lo:.2f}..{hi:.2f}):")
    print(f"    {risk:>8s} " + "".join(f"{g:>12s}" for g in arms))
    for level in np.linspace(lo, hi, 6):
        cells = "".join(f"{_interp_at(a, risk, 'log_growth', level):>12.2f}"
                        for a in arms.values())
        print(f"    {level:>8.2f} {cells}")


def frontier() -> None:
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN, "INNER-TRAIN"),
                                    (VALID, "INNER-VALIDATION")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            print(f"{split} / {mname} frontier:")
            frame = _sweep(start, end, market)
            _compare_at_matched_risk(frame, "vol")
            _compare_at_matched_risk(frame, "dd")
    print(f"\nconfigurations evaluated in this run: {N_EVALUATED} backtests, "
          f"{len(GRID) * 3} distinct configurations")


def match() -> None:
    """Freeze one multiplier per gate: equal realized vol on inner-validation.

    The matching target is the incumbent-shaped arm at its own scale
    (``vote`` at multiplier 1), so the frozen set is "the incumbent, and
    the other two gates run at the incumbent's risk". The multiplier is
    chosen on inner-validation only; the holdout is not read here.
    """
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"\nINNER-VALIDATION / {mname}: multiplier that matches the "
              f"vote arm's realized vol")
        frame = _sweep(*VALID, market)
        target = float(frame[(frame.gate == "vote") & (frame.mult == 1.0)]["vol"].iloc[0])
        print(f"  target realized vol = {target:.3f}")
        for gate in ("none", "vote", "evidence"):
            arm = frame[frame.gate == gate]
            m = _interp_at(arm, "vol", "mult", target)
            print(f"    {gate:9s} multiplier = {m:.2f}  "
                  f"(vol range {arm.vol.min():.2f}..{arm.vol.max():.2f})")
    print(f"\nconfigurations evaluated in this run: {N_EVALUATED}")


# --------------------------------------------------------------------- frozen

# Frozen on inner-validation before the holdout was read: per market, the
# multiplier that puts each gate at the *same* realized volatility as the
# incumbent-shaped `vote` arm at its own scale (spot 0.327, futures 0.322
# annualized). Read off the 33-configuration sweep by linear interpolation;
# `python experiments/run_gate_control.py match` reproduces them. The
# decision rule that uses them is in docs/LEDGER.md, row R-32.
FROZEN: dict[str, dict[str, float]] = {
    "spot": {"none": 0.61, "vote": 1.00, "evidence": 7.83},
    "futures_5x": {"none": 0.48, "vote": 1.00, "evidence": 3.25},
}


def _frozen_arms(market: MarketSpec):
    return [(f"{gate} x{mult:g}", GatedKelly(gate=gate, multiplier=mult))
            for gate, mult in FROZEN[market.name].items()]


def holdout() -> None:
    """Step 4. Configurations frozen; the decision rule is in the ledger."""
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"\nHOLDOUT 2023-01-01 -> / {mname}:")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            ev(get_strategy(name), *OOS, market=market, tag=f"  {name}",
               count=False)
        for tag, strat in _frozen_arms(market):
            ev(strat, *OOS, market=market, tag=f"  {tag} (FROZEN)", count=False)


def inference() -> None:
    """Paired block-bootstrap intervals between the arms, on the holdout.

    The point estimates above are one path. These are the same paired
    stationary-bootstrap comparisons the comparison table carries (R-29 /
    R-30): 30-day mean block, 2,000 resamples, identical resample indices
    applied to both arms so the market's own variance cancels.
    """
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        curves = {}
        for name in ("buy_and_hold", "kelly_regime_v4"):
            r = run_period(get_strategy(name), DF, *OOS, market=market,
                           start_balance=1_000.0, data_label=LABEL)
            curves[name] = daily_returns(r.equity)
        for tag, strat in _frozen_arms(market):
            r = run_period(strat, DF, *OOS, market=market,
                           start_balance=1_000.0, data_label=LABEL)
            curves[tag] = daily_returns(r.equity)

        n = min(len(v) for v in curves.values())
        idx = stationary_bootstrap_indices(n, 30.0, 2_000,
                                           np.random.default_rng(7))
        print(f"\nHOLDOUT / {mname}: paired differences, 95% block-bootstrap")
        names = list(curves)
        ev_arm = next(n for n in names if n.startswith("evidence"))
        vote_arm = next(n for n in names if n.startswith("vote"))
        none_arm = next(n for n in names if n.startswith("none"))
        pairs = [(ev_arm, vote_arm),          # Q1: the B-11 question
                 (ev_arm, none_arm),          # Q2: does the e-process gate pay?
                 (vote_arm, none_arm),        # Q2: does the latched vote pay?
                 (ev_arm, "buy_and_hold"), (vote_arm, "buy_and_hold"),
                 (none_arm, "buy_and_hold"),  # the promotion bar
                 (ev_arm, "kelly_regime_v4"), (vote_arm, "kelly_regime_v4")]
        for a, b in pairs:
            for stat, label in ((total_log_return, "Δ log growth"),
                                (max_drawdown_from_returns, "Δ max DD  ")):
                res = paired_bootstrap(curves[a].to_numpy()[:n],
                                       curves[b].to_numpy()[:n], stat,
                                       indices=idx)
                mark = "*" if res.significant else " "
                print(f"  {label} {a:14s} vs {b:16s} "
                      f"{res.diff.point:>+7.2f} [{res.diff.lo:>+6.2f}, "
                      f"{res.diff.hi:>+6.2f}]{mark}  P(>0)={res.p_positive:.2f}")


# ------------------------------------------------------------------ the checks

def causality() -> None:
    """The strict on_bar peek check, run by hand for an unregistered strategy.

    ``tests/test_causality_strict.py`` parametrizes over the *registry*, so
    an experiment gets none of that protection. Same two-opposite-tampers
    procedure as R-28: bars after a cut are multiplied by 3 in one copy and
    divided by 3 in the other, and every decision at or before the cut must
    be identical. The prepared columns are compared directly as well, which
    is what catches a full-series statistic applied to early rows.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    df = DF.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    for gate in ("none", "vote", "evidence"):
        def decisions(frame):
            s = GatedKelly(gate=gate, multiplier=2.0)
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out, prepared

        a, pa = decisions(up)
        b, pb = decisions(down)
        bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
        status = ("FAIL - reads the future at bars " + str(bad) if bad
                  else "PASS - decisions unchanged at and before the cut")
        worst = max(float(np.nanmax(np.abs(pa[c].to_numpy()[:cut]
                                           - pb[c].to_numpy()[:cut])))
                    for c in ("target", "gate"))
        print(f"  gate={gate:9s} {status}; max |Δcolumn| before the cut = "
              f"{worst:.3e} {'PASS' if worst < 1e-12 else 'FAIL'}")


def eth() -> None:
    """Pre-registered falsification: does the frontier ordering survive on ETH?

    Same venue (Bitfinex), same window, only the asset varies — the R-17
    design, so the incumbent's numbers there are the comparison.
    """
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for market in (SPOT, FUTURES):
            for name in ("buy_and_hold", "kelly_regime_v4"):
                ev(get_strategy(name), None, None, df=df, market=market,
                   tag=f"  {name}", count=False)
            for tag, strat in _frozen_arms(market):
                ev(strat, None, None, df=df, market=market, tag=f"  {tag}",
                   count=False)


def costs() -> None:
    """Step-4 cost checks: the real fee tier, and funding on the futures side."""
    from tradebot.data import load_funding

    print("HOLDOUT 2023+ at Bitstamp's 0.40% entry taker tier (spot):")
    for tier, label in ((0.001, "0.10% (table assumption)"),
                        (0.004, "0.40% (entry tier)")):
        market = MarketSpec.spot(fee_rate=tier)
        print(f"  {label}")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            ev(get_strategy(name), *OOS, market=market, tag=f"    {name}",
               count=False)
        for tag, strat in _frozen_arms(market):
            ev(strat, *OOS, market=market, tag=f"    {tag}", count=False)

    real = load_funding(ROOT / "data")
    grid = pd.date_range(DF.index[0].ceil("8h"), DF.index[-1], freq="8h", tz="UTC")
    filler = pd.Series(float(real.mean()), index=grid)
    blended = pd.concat([filler[~filler.index.isin(real.index)], real]).sort_index()
    print(f"\nHOLDOUT 2023+ with funding CHARGED (real through "
          f"{real.index[-1]:%Y-%m}, mean {real.mean() * 3 * 365.25:+.1%}/yr after):")
    arms = [("buy_and_hold", get_strategy("buy_and_hold")),
            ("kelly_regime_v4", get_strategy("kelly_regime_v4"))] + _frozen_arms(FUTURES)
    for name, strat in arms:
        lo = int(DF.index.searchsorted(OOS[0]))
        pre = min(lo, strat.warmup)
        raw = run_backtest(strat, DF.iloc[lo - pre:], FUTURES, 1_000.0,
                           trade_start=pre, funding=blended, data_label=LABEL)
        trimmed = replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
        m = compute_metrics(trimmed)
        print(f"  {name:24s} final=${m.final_balance:>9,.0f} "
              f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
              f"funding paid=${raw.funding_paid:>7,.0f}")


def chart() -> None:
    """Draw the four frontier panels into reports/gate_control/frontier.png.

    Inner split only — the holdout is read for the frozen points alone, and
    a 33-point sweep of it would be 33 consultations for a picture.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    from tradebot.report import (BASELINE, GRID as GRIDC, INK, INK_2, MUTED,
                                 PAGE, SERIES, SURFACE, _style_axes)

    colors = {"none": MUTED, "vote": SERIES[0], "evidence": SERIES[1]}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=PAGE)
    panels = [((TRAIN, "inner-train 2017-2020"), (SPOT, "spot")),
              ((VALID, "inner-validation 2021-2022"), (SPOT, "spot")),
              ((TRAIN, "inner-train 2017-2020"), (FUTURES, "futures 5x")),
              ((VALID, "inner-validation 2021-2022"), (FUTURES, "futures 5x"))]
    for ax, ((period, pname), (market, mname)) in zip(axes.ravel(), panels):
        frame = _sweep(*period, market)
        _style_axes(ax)
        # _style_axes formats the y axis as whole currency units; these
        # panels plot log growth, where 0.3 and -0.09 both render as "0".
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1f}"))
        ax.set_facecolor(SURFACE)
        for gate in ("none", "vote", "evidence"):
            arm = frame[frame.gate == gate].sort_values("vol")
            ax.plot(arm["vol"], arm["log_growth"], marker="o", markersize=3.5,
                    linewidth=1.8, color=colors[gate], label=gate, zorder=3)
            frozen = FROZEN[market.name][gate]
            near = arm.iloc[(arm["mult"] - frozen).abs().argsort()[:1]]
            ax.scatter(near["vol"], near["log_growth"], s=70, zorder=4,
                       facecolor="none", edgecolor=colors[gate], linewidth=1.6)
        ax.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=1)
        ax.set_title(f"{pname} · {mname}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("realized volatility (annualized)", color=MUTED, fontsize=8)
        ax.set_ylabel("log growth", color=MUTED, fontsize=8)
        ax.grid(True, color=GRIDC, linewidth=0.8)
        ax.legend(loc="best", fontsize=8, labelcolor=INK_2, frameon=True,
                  facecolor=SURFACE, edgecolor="none", framealpha=0.85)
    fig.suptitle("Same sizer, three gates: log growth against realized risk",
                 color=INK, fontsize=12, x=0.09, ha="left")
    fig.text(0.09, 0.945, "each curve is one gate swept over 11 exposure "
             "multipliers; rings mark the frozen matched-risk point",
             color=MUTED, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "reports" / "gate_control"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "frontier.png", dpi=130, facecolor=PAGE)
    plt.close(fig)
    print(f"wrote {out / 'frontier.png'}")


def deflated() -> None:
    """Deflated Sharpe for this round's two headline arms (Bailey & LdP 2014).

    The trial set is this session's 33 configurations, scored on the split
    the selection was made on (inner-validation, spot). The project-level
    count is R-29's floor of 103 plus this round's 33.
    """
    from tradebot.inference import (annualized_sharpe, deflated_sharpe_ratio,
                                    deflation_breakeven_sd, expected_max_sharpe,
                                    min_track_record_length, moments)

    sharpes = []
    for gate in ("none", "vote", "evidence"):
        for m in GRID:
            r = run_period(GatedKelly(gate=gate, multiplier=m), DF, *VALID,
                           market=SPOT, start_balance=1_000.0, data_label=LABEL)
            sharpes.append(annualized_sharpe(daily_returns(r.equity).to_numpy()))
    sharpes = np.asarray(sharpes, dtype=float)
    sd_trials = float(np.nanstd(sharpes, ddof=1))
    print(f"trials this session: {len(sharpes)} configurations, "
          f"inner-validation daily Sharpe sd={sd_trials:.3f}, "
          f"range {np.nanmin(sharpes):.2f}..{np.nanmax(sharpes):.2f}")

    for tag, strat in _frozen_arms(SPOT):
        if tag.startswith("none"):
            continue
        r = run_period(strat, DF, *OOS, market=SPOT, start_balance=1_000.0,
                       data_label=LABEL)
        rets = daily_returns(r.equity).to_numpy()
        sr = annualized_sharpe(rets)
        skew, kurt = moments(rets)
        print(f"\n  {tag} — holdout spot, daily Sharpe {sr:.2f}, n={len(rets)}")
        for n_trials, label in ((len(sharpes), "this session"),
                                (103 + len(sharpes), "project-level")):
            dsr = deflated_sharpe_ratio(sr, len(rets), skew, kurt, n_trials,
                                        sd_trials)
            star = expected_max_sharpe(n_trials, sd_trials)
            print(f"    {label:14s} N={n_trials:>3d}  SR*={star:.2f}  DSR={dsr:.3f}")
        be = deflation_breakeven_sd(sr, len(rets), skew, kurt, 103 + len(sharpes))
        mtrl = min_track_record_length(
            sr, skew, kurt, benchmark=expected_max_sharpe(136, sd_trials))
        print(f"    break-even trial sd {be:.2f}; "
              f"min track record {mtrl / DAYS_PER_YEAR:.1f}y "
              f"(holdout is {len(rets) / DAYS_PER_YEAR:.1f}y)")


def windows(trials: int = 40, seed: int = 42) -> None:
    """Path sensitivity: the R-19 design, identical windows across arms."""
    from tradebot.metrics import max_drawdown_pct

    def contenders_for(market):
        return ([("buy_and_hold", get_strategy("buy_and_hold")),
                 ("kelly_regime_v4", get_strategy("kelly_regime_v4"))]
                + _frozen_arms(market))

    contenders = contenders_for(SPOT)
    warmup = max(s.warmup for _, s in contenders) + 10
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(trials):
        length = int(rng.integers(90, 731) * 288)
        specs.append((int(rng.integers(warmup, len(DF) - length)), length))

    rows = []
    for k, (start, length) in enumerate(specs, 1):
        window = DF.iloc[start - warmup: start + length]
        for mname, market in (("spot", SPOT), ("futures", FUTURES)):
            # Each market gets its own frozen multipliers, so the arms are
            # matched on risk *within* the market they are compared in.
            for name, strat in contenders_for(market):
                res = run_backtest(strat, window, market, 1_000.0,
                                   trade_start=warmup, data_label=LABEL)
                eq = res.equity.to_numpy(dtype=float)
                base, seg = eq[warmup], eq[warmup:]
                ok = np.isfinite(base) and base > 0
                rows.append({"trial": k, "market": mname, "strategy": name,
                             "return_pct": 100.0 * (seg[-1] / base - 1.0) if ok else -100.0,
                             "max_dd_pct": max_drawdown_pct(seg) if ok else 100.0,
                             "liquidated": res.liquidated})
        print(f"[{k}/{trials}]", end=" ", flush=True, file=sys.stderr)
    res = pd.DataFrame(rows)
    out = ROOT / "reports" / "gate_control"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "windows.csv", index=False)

    print(f"\n{trials} random windows (90-730 days), identical across arms:\n")
    for mname, market in (("spot", SPOT), ("futures", FUTURES)):
        names = [n for n, _ in contenders_for(market)]
        print(f"  {mname}:")
        sub = res[res.market == mname]
        bench = sub[sub.strategy == "buy_and_hold"].set_index("trial")["return_pct"]
        for name in names:
            g = sub[sub.strategy == name].set_index("trial")
            print(f"    {name:16s} median return {g.return_pct.median():>+8.1f}%  "
                  f"median DD {g.max_dd_pct.median():>5.1f}%  "
                  f"worst DD {g.max_dd_pct.max():>5.1f}%  "
                  f"P(DD>50%) {(g.max_dd_pct > 50).mean():>5.0%}  "
                  f"beat hold {(g['return_pct'] > bench).mean():>5.0%}  "
                  f"liq {g.liquidated.mean():>4.0%}")
        ev_name = next(n for n in names if n.startswith("evidence"))
        vote_name = next(n for n in names if n.startswith("vote"))
        none_name = next(n for n in names if n.startswith("none"))
        for x, y in ((ev_name, vote_name), (vote_name, none_name),
                     (ev_name, none_name)):
            a = sub[sub.strategy == x].set_index("trial")
            b = sub[sub.strategy == y].set_index("trial")
            dd = (a["max_dd_pct"] - b["max_dd_pct"]).dropna()
            rt = (a["return_pct"] - b["return_pct"]).dropna()
            print(f"    paired ({x.split()[0]} - {y.split()[0]}): "
                  f"DD median {dd.median():+.1f}pp, deeper in {(dd > 0).mean():.0%} "
                  f"of windows; return median {rt.median():+.1f}pp, "
                  f"higher in {(rt > 0).mean():.0%}")
        print()


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"frontier": frontier, "match": match, "causality": causality,
            "holdout": holdout, "inference": inference, "eth": eth,
            "costs": costs, "windows": windows, "deflated": deflated, "chart": chart}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_gate_control.py [{'|'.join(cmds)}]")
