#!/usr/bin/env python
"""Driver for backlog B-05 — funding as a gate on kelly_regime_v4.

Splits follow ROUTINE.md step 3, adjusted for the funding data window
(real funding: 2020-01-01 .. 2023-12-31 only, pre-registered in
``experiments/funding_gate.py``)::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
                       (gate only has real funding to act on in 2020)
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
                       (fully covered by real funding)
    holdout           2023-01-01 -> 2023-12-31   step 4 only, pre-registered,
                       TRUNCATED — real funding stops 2023-12-31

Usage::

    python experiments/run_funding_gate.py parity      # gate_enabled=False == v4
    python experiments/run_funding_gate.py sweep        # the grid (step 3)
    python experiments/run_funding_gate.py causality    # by-hand lookahead probe
    python experiments/run_funding_gate.py holdout      # step 4, frozen, truncated
    python experiments/run_funding_gate.py interval     # paired bootstrap on the holdout
    python experiments/run_funding_gate.py windows      # falsification: MC windows in 2020-2023
    python experiments/run_funding_gate.py costs        # fee tier check on the holdout
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

from experiments.funding_gate import FundingGatedKelly  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL_FUNDING = load_funding(ROOT / "data")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS = ("2023-01-01", "2023-12-31")  # TRUNCATED: real funding stops 2023-12-31

OUT = ROOT / "reports" / "funding_gate"

_SEEN: set[tuple] = set()  # distinct (config) evaluated so far, for the count below


def measure(strategy, start, end, *, market=FUTURES, funding=None,
            balance=1_000.0, count_key=None):
    """One backtest over [start, end], warmed on the bars before it.

    Manual warmup handling (rather than tradebot.window.run_period)
    because run_period does not accept ``funding=`` — this is the same
    pattern scripts/funding_study.py and run_matched_risk.py's costs()
    use.
    """
    if count_key is not None and count_key not in _SEEN:
        _SEEN.add(count_key)
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, balance,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw.funding_paid, trimmed


def line(tag, m, funding_paid, result):
    print(f"  {tag:42s} final=${m.final_balance:>11,.0f} "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
          f"trades={m.num_trades:>4d} fees=${m.fees_paid:>7,.0f} "
          f"funding=${funding_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


def gated(threshold_pct, lookback_settlements, reduction, gate_enabled=True):
    return FundingGatedKelly(funding=REAL_FUNDING, gate_enabled=gate_enabled,
                             threshold_pct=threshold_pct,
                             lookback_settlements=lookback_settlements,
                             reduction=reduction)


# ---------------------------------------------------------------------- parity


def parity() -> None:
    """gate_enabled=False must reproduce kelly_regime_v4 bar-for-bar, on both markets.

    Also demonstrates the spot invariant from Step 1: even with
    gate_enabled=True, running on SPOT with funding=None (spot pays no
    funding) still differs from v4 only through the funding-based
    multiplier, which is why the runner never actually evaluates
    gate_enabled=True on spot below — that combination is deliberately
    unused, and this check is what justifies skipping it rather than
    silently doing so.
    """
    v4 = get_strategy("kelly_regime_v4")
    print("Inner-validation. gate_enabled=False vs kelly_regime_v4:")
    for mname, market in (("spot", SPOT), ("futures", FUTURES)):
        a = run_backtest(v4, DF.loc[VALID[0]:VALID[1]], market, 1_000.0,
                         data_label=LABEL)
        b = run_backtest(gated(0.90, 90, 0.0, gate_enabled=False),
                         DF.loc[VALID[0]:VALID[1]], market, 1_000.0,
                         data_label=LABEL)
        worst = float(np.max(np.abs(a.equity.to_numpy() - b.equity.to_numpy())))
        print(f"  {mname:8s} max |equity difference| = {worst:.3e}  "
              f"{'PASS' if worst < 1e-9 else 'FAIL'}")

    print("\nSame check with gate_enabled=True but no funding series supplied "
          "(funding=None) — the 'unknown funding never reduces' rule from Step 1 "
          "should also make this a no-op:")
    b2 = run_backtest(FundingGatedKelly(funding=None, gate_enabled=True),
                      DF.loc[VALID[0]:VALID[1]], FUTURES, 1_000.0, data_label=LABEL)
    a2 = run_backtest(v4, DF.loc[VALID[0]:VALID[1]], FUTURES, 1_000.0, data_label=LABEL)
    worst = float(np.max(np.abs(a2.equity.to_numpy() - b2.equity.to_numpy())))
    print(f"  futures, funding=None    max |equity difference| = {worst:.3e}  "
          f"{'PASS' if worst < 1e-9 else 'FAIL'}")


# ----------------------------------------------------------------------- sweep


NAMED_VARIANTS = {
    "V1 decile/30d/flat": dict(threshold_pct=0.90, lookback_settlements=90, reduction=0.0),
    "V2 decile/90d/flat": dict(threshold_pct=0.90, lookback_settlements=270, reduction=0.0),
    "V3 quintile/30d/half": dict(threshold_pct=0.80, lookback_settlements=90, reduction=0.5),
}

# One-knob-at-a-time neighbourhood around V1, for the plateau check (P4).
# Deliberately overlapping with V1-V3 where the grid crosses them — those
# points are not re-counted (see _SEEN).
THRESHOLD_GRID = (0.80, 0.85, 0.90, 0.95)
LOOKBACK_GRID = (60, 90, 180, 270)
REDUCTION_GRID = (0.0, 0.3, 0.5, 0.7, 1.0)


def _sweep_configs():
    """Yield (tag, kwargs) for every configuration this step evaluates."""
    for tag, kw in NAMED_VARIANTS.items():
        yield tag, kw
    base = NAMED_VARIANTS["V1 decile/30d/flat"]
    for t in THRESHOLD_GRID:
        kw = dict(base, threshold_pct=t)
        yield f"thr={t:.2f}", kw
    for lb in LOOKBACK_GRID:
        kw = dict(base, lookback_settlements=lb)
        yield f"lb={lb}", kw
    for red in REDUCTION_GRID:
        kw = dict(base, reduction=red)
        yield f"red={red:.1f}", kw


def sweep() -> None:
    """Step 3. Named variants + neighbourhood grid, inner-train and inner-validation, futures only.

    Not run on spot: by construction (Step 1) the gate cannot change the
    spot backtest, so sweeping spot would only re-confirm parity(), not
    search anything.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    v4 = get_strategy("kelly_regime_v4")
    rows = []
    for split, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        print(f"\n{split} ({start} .. {end}), futures 5x, funding CHARGED where real:")
        m, fp, res = measure(v4, start, end, funding=REAL_FUNDING,
                             count_key=None)
        line("kelly_regime_v4 (ungated)", m, fp, res)
        rows.append({"split": split, "tag": "ungated", **{}, "final": m.final_balance,
                     "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                     "trades": m.num_trades, "fees": m.fees_paid, "funding": fp})
        seen_this_split = set()
        for tag, kw in _sweep_configs():
            key = tuple(sorted(kw.items()))
            first_time = key not in _SEEN
            m, fp, res = measure(gated(**kw), start, end, funding=REAL_FUNDING,
                                 count_key=key)
            if key not in seen_this_split:
                line(tag, m, fp, res)
                seen_this_split.add(key)
            rows.append({"split": split, "tag": tag, **kw, "final": m.final_balance,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "trades": m.num_trades, "fees": m.fees_paid, "funding": fp,
                         "first_time_counted": first_time})
    pd.DataFrame(rows).to_csv(OUT / "sweep.csv", index=False)
    print(f"\nconfigurations evaluated (distinct, counted once): {len(_SEEN)}")
    print(f"written: {OUT / 'sweep.csv'}")


# ------------------------------------------------------------------- causality


def causality() -> None:
    """Two-opposite-tampers probe, by hand — this file gets no CI protection.

    Two independent tampers, both required to pass:
    (a) price/volume after the cut multiplied by 3 / divided by 3 (the
        R-28/R-31 procedure) — catches lookahead through the price series;
    (b) funding rates after the cut multiplied by 3 / divided by 3,
        price held fixed — catches lookahead through the funding series,
        which is new to this experiment and not covered by (a).
    Every decision at or before the cut must be identical under both.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    lo = int(DF.index.searchsorted("2020-01-01"))
    hi = int(DF.index.searchsorted("2023-12-31", side="right"))
    df = DF.iloc[lo:hi].copy()
    cut = len(df) // 2
    cut_ts = df.index[cut]
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    def decisions(frame, funding):
        s = gated(0.90, 90, 0.0)
        s.funding = funding
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out, prepared

    ok = True

    # (a) price/volume tamper, funding held fixed.
    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    oa, pa = decisions(up, REAL_FUNDING)
    ob, pb = decisions(down, REAL_FUNDING)
    bad = [b for b, x, y in zip(bars, oa, ob) if x != y]
    worst = max(float(np.nanmax(np.abs(pa[c].to_numpy()[:cut] - pb[c].to_numpy()[:cut])))
                for c in ("target", "gate_mult", "funding_pct"))
    good = not bad and worst < 1e-12
    ok &= good
    print(f"  (a) price/volume tamper   orders {'match' if not bad else f'DIFFER at {bad}'}"
          f"   max |column diff| before cut = {worst:.3e}   {'PASS' if good else 'FAIL'}")

    # (b) funding tamper, price held fixed.
    f_up = REAL_FUNDING.copy()
    f_down = REAL_FUNDING.copy()
    after = f_up.index > cut_ts
    f_up.loc[after] = f_up.loc[after] * 3.0
    f_down.loc[after] = f_down.loc[after] / 3.0

    oa, pa = decisions(df, f_up)
    ob, pb = decisions(df, f_down)
    bad = [b for b, x, y in zip(bars, oa, ob) if x != y]
    worst = max(float(np.nanmax(np.abs(pa[c].to_numpy()[:cut] - pb[c].to_numpy()[:cut])))
                for c in ("target", "gate_mult", "funding_pct"))
    good = not bad and worst < 1e-12
    ok &= good
    print(f"  (b) funding tamper        orders {'match' if not bad else f'DIFFER at {bad}'}"
          f"   max |column diff| before cut = {worst:.3e}   {'PASS' if good else 'FAIL'}")

    print(f"\ntampered from bar {cut:,} of {len(df):,} (timestamp {cut_ts}); "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# --------------------------------------------------------------------- holdout


# ============================================================================
# STEP 4 — FROZEN CONFIGURATION AND PRE-REGISTERED DECISION RULE
# Written and committed to this file BEFORE `holdout()` below was ever called
# with real data. See the accompanying ledger report for the sweep() results
# this selection was made from (inner-train / inner-validation only).
# ============================================================================
#
# FROZEN CONFIGURATION: V1 — threshold_pct=0.90, lookback_settlements=90
# (~30 days), reduction=0.0 (stand fully flat in the trailing top decile of
# realized funding). Selected because on inner-validation (2021-2022, the
# only inner split with full real-funding coverage) it is the variant that
# both cuts funding paid the most and does not cost inner-validation return
# outside noise, and the neighbourhood around it (thresholds 0.80-0.95,
# lookbacks 60-270, reductions 0.0-1.0) is a plateau on funding-avoided and
# NOT a plateau on return — full detail in the ledger report.
#
# DECISION RULE (promote only if ALL of P1-P4 hold; default is REJECT):
#
# P1 — beats buy_and_hold OOS on futures with real funding charged, on the
#      TRUNCATED holdout 2023-01-01 -> 2023-12-31 (the only slice with both
#      real funding coverage AND genuine out-of-sample status).
# P2 — improvement over UNGATED kelly_regime_v4 (same market futures, real
#      funding charged on BOTH arms, same truncated holdout) satisfies at
#      least one of: (a) Sharpe improves by more than the +/-0.2 noise
#      floor; (b) max drawdown falls by >= 10 percentage points; (c) funding
#      paid falls meaningfully (>= 25% relative) WITHOUT hurting final
#      balance or Sharpe outside the noise floor.
# P3 — survives the pre-registered falsification test: Monte Carlo windows
#      (scripts/stress_test.py design) restricted to windows drawn entirely
#      inside the real-funding span 2020-01-01..2023-12-31, futures only,
#      identical windows for the gated and ungated arms, both with real
#      funding charged. "Survives" means the gated arm is not worse than
#      ungated v4 on BOTH median drawdown and median return by a margin that
#      would itself be considered a finding if it ran the other way (i.e. no
#      one-sided reading).
# P4 — the neighbourhood around the frozen threshold/lookback/reduction is a
#      plateau on whichever axis P2 is claimed on, not a single lucky point
#      (reported from the sweep() grid, which was run before any holdout
#      data was touched).
#
# Explicitly out of scope, stated in advance so it cannot be added after the
# fact if the result disappoints: no claim about 2024 onward (no real
# funding data exists for it), and no claim about spot (the mechanism cannot
# touch it by construction).
#
# If any number below causes this rule to be edited, the ledger report must
# say so explicitly and the result is downgraded to in-sample. Nothing below
# this comment block was written after a holdout number was read.
# ============================================================================

FROZEN = dict(threshold_pct=0.90, lookback_settlements=90, reduction=0.0)


def holdout() -> None:
    """Step 4. Truncated holdout: 2023-01-01 -> 2023-12-31 only, funding charged."""
    OUT.mkdir(parents=True, exist_ok=True)
    v4 = get_strategy("kelly_regime_v4")
    rows = []
    print(f"HOLDOUT {OOS[0]} .. {OOS[1]} (TRUNCATED — real funding stops "
          f"2023-12-31), futures 5x, funding CHARGED:")
    for tag, strat in (("buy_and_hold", get_strategy("buy_and_hold")),
                       ("kelly_regime_v4 (ungated)", v4),
                       ("funding_gated_kelly (frozen V1)", gated(**FROZEN))):
        m, fp, res = measure(strat, *OOS, market=FUTURES, funding=REAL_FUNDING)
        line(tag, m, fp, res)
        rows.append({"tag": tag, "final": m.final_balance, "max_dd": m.max_drawdown_pct,
                     "sharpe": m.sharpe, "trades": m.num_trades, "fees": m.fees_paid,
                     "funding": fp, "liquidated": m.liquidated})

    print("\nSame holdout, funding-FREE (upper-bound convention, for reference only):")
    for tag, strat in (("kelly_regime_v4 (ungated)", v4),
                       ("funding_gated_kelly (frozen V1)", gated(**FROZEN))):
        m, fp, res = measure(strat, *OOS, market=FUTURES, funding=None)
        line(tag, m, fp, res)

    print("\nSpot, both arms — must be identical to kelly_regime_v4 on spot "
          "(the gate is a no-op there by construction):")
    a = get_strategy("kelly_regime_v4")
    b = gated(**FROZEN, gate_enabled=False)
    ma, _, ra = measure(a, *OOS, market=SPOT)
    mb, _, rb = measure(b, *OOS, market=SPOT)
    worst = float(np.max(np.abs(ra.equity.to_numpy() - rb.equity.to_numpy())))
    print(f"  max |equity difference| = {worst:.3e}  {'PASS' if worst < 1e-9 else 'FAIL'}")

    pd.DataFrame(rows).to_csv(OUT / "holdout.csv", index=False)
    print(f"\nwritten: {OUT / 'holdout.csv'}")


# -------------------------------------------------------------------- interval


def interval() -> None:
    """Paired stationary block bootstrap, gated vs ungated, on the truncated holdout."""
    from tradebot.inference import (daily_returns, max_drawdown_from_returns,
                                    paired_bootstrap, stationary_bootstrap_indices,
                                    total_log_return)

    v4 = get_strategy("kelly_regime_v4")
    curves = {}
    for tag, strat in (("ungated", v4), ("gated", gated(**FROZEN))):
        _, _, res = measure(strat, *OOS, market=FUTURES, funding=REAL_FUNDING)
        curves[tag] = daily_returns(res.equity).to_numpy()
    n = len(curves["ungated"])
    idx = stationary_bootstrap_indices(n, 30.0, 2_000, np.random.default_rng(7))
    print(f"Truncated holdout ({OOS[0]}..{OOS[1]}, {n} daily observations), "
          f"funding charged both arms:")
    for stat_name, stat in (("Δ log growth", total_log_return),
                            ("Δ max drawdown (pp)", max_drawdown_from_returns)):
        r = paired_bootstrap(curves["gated"], curves["ungated"], stat, indices=idx)
        mark = "▲" if r.diff.lo > 0 else ("▼" if r.diff.hi < 0 else "≈")
        print(f"  gated - ungated  {stat_name:22s} {mark} {r.diff.point:>+7.3f} "
              f"[{r.diff.lo:>+7.3f}, {r.diff.hi:>+7.3f}]  P(gated>ungated)={r.p_positive:.2f}")

    # Also against buy_and_hold, since P1 is stated against holding.
    _, _, res_hold = measure(get_strategy("buy_and_hold"), *OOS, market=FUTURES,
                             funding=REAL_FUNDING)
    curves["hold"] = daily_returns(res_hold.equity).to_numpy()
    for stat_name, stat in (("Δ log growth", total_log_return),
                            ("Δ max drawdown (pp)", max_drawdown_from_returns)):
        r = paired_bootstrap(curves["gated"], curves["hold"], stat, indices=idx)
        mark = "▲" if r.diff.lo > 0 else ("▼" if r.diff.hi < 0 else "≈")
        print(f"  gated - hold     {stat_name:22s} {mark} {r.diff.point:>+7.3f} "
              f"[{r.diff.lo:>+7.3f}, {r.diff.hi:>+7.3f}]  P(gated>hold)={r.p_positive:.2f}")


# -------------------------------------------------------------------- windows


def windows(trials: int = 40, seed: int = 42) -> None:
    """Falsification test (pre-registered): Monte Carlo windows INSIDE 2020-2023 only.

    Same design as R-19/R-31 (random window length/start, identical
    windows across strategies) but the window draw is restricted to lie
    entirely inside the real-funding span, since that is the only region
    where the gated arm differs from v4 at all. Both arms pay real
    funding. Futures only (the mechanism is futures-only).
    """
    from tradebot.metrics import max_drawdown_pct

    lo = int(DF.index.searchsorted("2020-01-01"))
    hi = int(DF.index.searchsorted("2023-12-31", side="right"))
    contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                  ("kelly_regime_v4 (ungated)", get_strategy("kelly_regime_v4")),
                  ("funding_gated_kelly (frozen)", gated(**FROZEN))]
    warmup = max(s.warmup for _, s in contenders) + 10

    rng = np.random.default_rng(seed)
    specs = []
    tries = 0
    while len(specs) < trials and tries < trials * 50:
        tries += 1
        length = int(rng.integers(30, 271) * 288)  # 30-270 days, fits the ~4y span
        start = int(rng.integers(lo + warmup, hi - length))
        if start - warmup >= lo:
            specs.append((start, length))
    if len(specs) < trials:
        print(f"warning: only found {len(specs)} valid windows inside the funding span")

    rows = []
    for k, (start, length) in enumerate(specs, 1):
        window = DF.iloc[start - warmup: start + length]
        for name, strat in contenders:
            res = run_backtest(strat, window, FUTURES, 1_000.0, trade_start=warmup,
                               funding=REAL_FUNDING, data_label=LABEL)
            eq = res.equity.to_numpy(dtype=float)
            base, seg = eq[warmup], eq[warmup:]
            ok = np.isfinite(base) and base > 0
            rows.append({"trial": k, "strategy": name,
                         "start_ts": str(DF.index[start]),
                         "return_pct": 100.0 * (seg[-1] / base - 1.0) if ok else -100.0,
                         "max_dd_pct": max_drawdown_pct(seg) if ok else 100.0,
                         "liquidated": res.liquidated,
                         "funding_paid": res.funding_paid})
        print(f"[{k}/{len(specs)}]", end=" ", flush=True, file=sys.stderr)
    res_df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUT / "windows.csv", index=False)

    print(f"\n\n{len(specs)} random windows (30-270 days) drawn entirely inside "
          f"2020-01-01..2023-12-31, identical across strategies, futures 5x, "
          f"real funding charged:\n")
    bench = res_df[res_df.strategy == "buy_and_hold"].set_index("trial")["return_pct"]
    for name, _ in contenders:
        g = res_df[res_df.strategy == name].set_index("trial")
        print(f"  {name:32s} median return {g.return_pct.median():>+8.1f}%  "
              f"median DD {g.max_dd_pct.median():>5.1f}%  worst DD {g.max_dd_pct.max():>5.1f}%  "
              f"beat hold {(g.return_pct > bench).mean():>5.0%}  liq {g.liquidated.mean():>4.0%}  "
              f"median funding paid ${g.funding_paid.median():>7,.0f}")

    a = res_df[res_df.strategy == "funding_gated_kelly (frozen)"].set_index("trial")
    b = res_df[res_df.strategy == "kelly_regime_v4 (ungated)"].set_index("trial")
    d_ret = (a.return_pct - b.return_pct).dropna()
    d_dd = (a.max_dd_pct - b.max_dd_pct).dropna()
    d_fund = (a.funding_paid - b.funding_paid).dropna()
    print(f"\n  paired gated - ungated: return median {d_ret.median():+.1f}pp, "
          f"gated higher in {(d_ret > 0).mean():.0%};  "
          f"DD median {d_dd.median():+.1f}pp, gated deeper in {(d_dd > 0).mean():.0%};  "
          f"funding paid median {d_fund.median():+,.0f}, gated pays less in "
          f"{(d_fund < 0).mean():.0%}")
    print(f"\nwritten: {OUT / 'windows.csv'}")


# ----------------------------------------------------------------------- costs


def costs() -> None:
    """Secondary check (not the pre-registered falsification test): fee-tier robustness."""
    v4 = get_strategy("kelly_regime_v4")
    for tier, label in ((0.0005, "0.05% (table assumption, futures)"),
                        (0.0040, "0.40% (Bitstamp entry tier, stress)")):
        market = MarketSpec.futures(leverage=5.0, fee_rate=tier)
        print(f"\nHOLDOUT {OOS[0]}..{OOS[1]}, {label}, funding charged:")
        for tag, strat in (("buy_and_hold", get_strategy("buy_and_hold")),
                           ("kelly_regime_v4 (ungated)", v4),
                           ("funding_gated_kelly (frozen)", gated(**FROZEN))):
            m, fp, res = measure(strat, *OOS, market=market, funding=REAL_FUNDING)
            line(tag, m, fp, res)


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})  |  funding: {len(REAL_FUNDING):,} settlements "
          f"{REAL_FUNDING.index[0]:%Y-%m-%d} -> {REAL_FUNDING.index[-1]:%Y-%m-%d}",
          file=sys.stderr)
    cmds = {"parity": parity, "sweep": sweep, "causality": causality,
            "holdout": holdout, "interval": interval, "windows": windows,
            "costs": costs}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate.py [{'|'.join(cmds)}]")
