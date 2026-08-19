#!/usr/bin/env python
"""R-35's funding-decile gate, re-read against the EXTENDED funding series (R-39 conservative).

Unregistered experiment: lives under ``experiments/`` so it is NOT
auto-discovered (docs/ROUTINE.md step 5 / the registry's package-scan in
``tradebot.registry``). Do not decorate with ``@register``. Nothing here
is committed by this session; the operator merges.

Pre-registration: ``docs/LEDGER.md``, "R-39 pre-registration -- the
network re-check R-38 asked for, done properly, and what it unblocks."
This file is exactly the "Conservative" variant named there:

    Literally ``experiments/funding_gate_decile.py``'s decision logic
    (v4's vote/scale untouched, ``target = 0`` where the trailing
    rolling-window funding percentile >= the fixed 0.90 decile, same
    swept lookback set {90, 180, 365, expanding}, same 3-settlement
    causal EWM smoothing) repointed at ``load_funding_extended()`` in
    place of ``load_funding()``. No new parameter, no new mechanism --
    only the funding series' length changes.

How "byte-for-byte identical decision logic" is guaranteed here
---------------------------------------------------------------
Not by copying the R-35 file's code into this one (a copy can drift, and
a reader then has to diff two 300-line modules to believe the claim), but
by **importing R-35's class unmodified and injecting the longer funding
series through the constructor argument it already has**:

    FundingGateDecile(funding_window_days=w, decile=0.90, funding=<extended>)

``experiments/funding_gate_decile.py``'s ``__init__`` already accepts
``funding: pd.Series | None`` and only falls back to
``load_funding(ROOT/"data")`` when it is ``None`` -- that hook exists in
the R-35 file itself (it was added there for its own causality probe).
So the swap is a one-argument change and *nothing else can possibly
differ*: ``prepare`` and ``on_bar`` are the same function objects, not
copies of them. :func:`identity_check` asserts exactly that
(``FundingGateDecileExtended.prepare is FundingGateDecile.prepare`` and
the same for ``on_bar``), and it runs as the first thing every command
below does.

The thin subclass :class:`FundingGateDecileExtended` exists only to give
the arm a distinct ``name`` in report tables and to default ``funding``
to the extended series. It overrides no method that touches a decision.

What the extended series is, and the one seam a reader must know about
----------------------------------------------------------------------
``tradebot.data.load_funding_extended`` returns real Binance BTCUSDT
funding for 2020-01-01..2023-12-31 (4,383 settlements, settled 03/11/19
UTC) concatenated with Deribit BTC-PERPETUAL funding for the genuine
post-2023 gap only (2,884 8h buckets, closed 00/08/16 UTC), tagged per
settlement with its source. The two are **not** rescaled onto a common
level (the cross-venue ratio is unstable year to year: 0.64x/1.24x/
0.21x/0.34x for 2020-2023 -- see ``load_funding_deribit``'s docstring).

That has a consequence this branch must state rather than hide: the
gate's percentile rank is computed over a *trailing window of
settlements*, so for roughly one window-length after the splice
(2024-01-01 for w=180: until ~2024-06-28) the rank of a Deribit rate is
taken against a history that is partly Binance. This is not a bug in the
loader and it is not something this branch is allowed to "fix" -- the
pre-registration fixes the mechanism and permits only the funding
series' length to change -- but it is a real seam, and §3/§8 of the
report quantify it (gate-fire rate by year) instead of asserting it is
harmless.

Commands
--------
::

    python experiments/funding_gate_decile_extended.py build      # 1. decile=1.1 == v4?
    python experiments/funding_gate_decile_extended.py causality  # 2. two-opposite-tampers
    python experiments/funding_gate_decile_extended.py coverage   # 3. funding coverage / fire rate
    python experiments/funding_gate_decile_extended.py validation # 4. reproduce R-35 inner-val
    python experiments/funding_gate_decile_extended.py holdout    # 5. THE pre-registered read
    python experiments/funding_gate_decile_extended.py neighbours # 6. w=90/365/expanding
    python experiments/funding_gate_decile_extended.py costs      # 7. 0.40% tier + exposure
    python experiments/funding_gate_decile_extended.py subperiod  # 8. 2023 vs 2024-2026
    python experiments/funding_gate_decile_extended.py all
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

from experiments.funding_gate_decile import (  # noqa: E402
    FundingGateDecile, _funding_percentile)
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding, load_funding_extended  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import (annualized_sharpe, daily_returns,  # noqa: E402
                                max_drawdown_from_returns, paired_bootstrap,
                                stationary_bootstrap_indices, total_log_return)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import prefix_bars  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

# The extended series (Binance 2020-2023 + Deribit 2024-2026) and the
# R-35 series (Binance only), kept side by side so every table can say
# which one produced it.
FUNDING_EXT, FUNDING_SRC = load_funding_extended(ROOT / "data")
FUNDING_R35 = load_funding(ROOT / "data")

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS = ("2023-01-01", None)          # the FULL extended holdout, 2023-01-01 -> end of data
OOS_2023 = ("2023-01-01", "2023-12-31")   # Binance-sourced funding: what R-35 already saw
OOS_2024P = ("2024-01-01", None)          # Deribit-sourced funding: genuinely new

# Pre-registered sweep set. w=180 is THE decision config (R-35's own
# recommendation, chosen as the middle of the four pre-registered sweep
# points and matching v3/v4's 180-day anchor convention -- not by
# searching for the best Sharpe). The others are neighbourhood context.
DECISION_W = 180
SWEEP = (90, 180, 365, None)
DECILE = 0.90                        # FIXED. Never tuned. Never swept.

MEAN_BLOCK, N_BOOT, SEED = 30.0, 2_000, 7

N_EVALUATED = 0                      # every backtest that produces a number


# --------------------------------------------------------------- the arm

class FundingGateDecileExtended(FundingGateDecile):
    """R-35's gate with the extended funding series injected. No logic of its own.

    Overrides ``__init__`` (to default ``funding`` to the extended series)
    and ``name`` (so tables can tell the arms apart). It does NOT override
    ``prepare`` or ``on_bar`` -- see :func:`identity_check`.
    """

    name = "funding_gate_decile_ext"

    def __init__(self, *args, funding: pd.Series | None = None, **kwargs) -> None:
        super().__init__(*args,
                         funding=FUNDING_EXT if funding is None else funding,
                         **kwargs)


def identity_check() -> None:
    """Assert the decision logic is literally R-35's, not a copy of it."""
    assert FundingGateDecileExtended.prepare is FundingGateDecile.prepare
    assert FundingGateDecileExtended.on_bar is FundingGateDecile.on_bar
    assert FundingGateDecileExtended.warmup == FundingGateDecile.warmup
    s = FundingGateDecileExtended(funding_window_days=DECISION_W)
    r35 = FundingGateDecile(funding_window_days=DECISION_W)
    shared = {k: v for k, v in vars(s).items() if k != "funding"}
    assert shared == {k: v for k, v in vars(r35).items() if k != "funding"}, \
        "a non-funding parameter differs between the two arms"
    assert s.funding is FUNDING_EXT and s.decile == DECILE == r35.decile


def gate(w: float | None) -> FundingGateDecileExtended:
    return FundingGateDecileExtended(funding_window_days=w, decile=DECILE)


def gate_r35(w: float | None) -> FundingGateDecile:
    """The same class on R-35's own (Binance-only) series, for the §4 check."""
    return FundingGateDecile(funding_window_days=w, decile=DECILE,
                             funding=FUNDING_R35)


class ScaledV4(KellyRegimeV4):
    """kelly_regime_v4 with its target flat-multiplied by a constant.

    The exposure-artifact control (R-35 §5, the standing L-04/R-33 rule):
    holding less draws down less, so before any gate claim is believed,
    v4 itself is rescaled to the gate's own mean exposure and re-run. At
    ``scale=1.0`` this is v4 exactly.
    """

    name = "kelly_regime_v4_scaled"

    def __init__(self, scale: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scale = float(scale)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = super().prepare(df)
        out["target"] = out["target"] * self.scale
        return out


# ------------------------------------------------------------- harness

def run_period_funding(strategy, df, start, end, *, market, funding=None,
                       start_balance: float = 1_000.0, data_label: str = LABEL):
    """``tradebot.window.run_period``, plus the engine's ``funding=`` argument.

    Identical warm-then-trade semantics (``trade_start=prefix``): the
    prefix warms indicator state but its orders are dropped, so the
    account is flat at ``start_balance`` when the measured period opens
    and ``funding_paid`` is in-period by construction (a flat account
    pays no funding -- ``PaperBroker.apply_funding`` returns early when
    ``pos == 0``).
    """
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    if hi <= lo:
        raise ValueError(f"empty period: {start!r} -> {end!r}")
    prefix = prefix_bars(df, lo, strategy.warmup)
    frame = df.iloc[lo - prefix: hi]
    result = run_backtest(strategy, frame, market, start_balance,
                          data_label=data_label, trade_start=prefix,
                          funding=funding)
    if prefix == 0:
        return result
    return replace(result, equity=result.equity.iloc[prefix:],
                   df=result.df.iloc[prefix:])


def _slice(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Label-based inclusive slice that tolerates ``None`` on either side."""
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    return df.iloc[lo:hi]


def describe(result, tag: str) -> dict:
    m = compute_metrics(result)
    exposure = (float(np.abs(result.df["target"].to_numpy()).mean())
                if "target" in result.df.columns else float("nan"))
    fired = (float(result.df["gated"].to_numpy().mean())
             if "gated" in result.df.columns else float("nan"))
    row = {"tag": tag, "market": result.market.name,
           "final": m.final_balance, "dd": m.max_drawdown_pct,
           "sharpe": m.sharpe, "trades": m.num_trades,
           "fees": m.fees_paid, "funding": result.funding_paid,
           "exposure": exposure, "fire": fired,
           "liquidated": m.liquidated, "equity": result.equity}
    print(f"  {tag:34s} {result.market.name:10s} final=${m.final_balance:>10,.0f} "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
          f"trades={m.num_trades:>4d} exp={exposure:>5.3f} "
          f"fund=${result.funding_paid:>7,.0f} fees=${m.fees_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")
    return row


def ev(strategy, start, end, *, market=FUTURES, funding=None, tag="",
       count: bool = True, df=None) -> dict:
    """One backtest, one line, counted for the deflated-Sharpe trials tally."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    result = run_period_funding(strategy, DF if df is None else df, start, end,
                                market=market, funding=funding)
    return describe(result, tag or strategy.name)


def v4() -> KellyRegimeV4:
    return get_strategy("kelly_regime_v4")


# -------------------------------------------------- 1. build / correctness

def build() -> None:
    """``decile=1.1`` (structurally cannot fire) must reproduce v4 bit-identically.

    ``pctl`` lives in [0, 1] so ``pctl >= 1.1`` is never true; the gate is
    then a no-op and ``target`` must equal ``kelly_regime_v4``'s own
    prepared target to the last bit. R-35 ran this against the Binance
    series; this runs it against the EXTENDED series, on a slice that
    spans the Binance->Deribit splice, so a loader bug that fabricated a
    value (or an out-of-range percentile) after 2023-12-31 would show up.
    """
    identity_check()
    print("build check: decile=1.1 must be bit-identical to kelly_regime_v4\n")
    for label, lo, hi in (
            ("2020-06-01..2022-12-31 (R-35's own slice)", "2020-06-01", "2022-12-31"),
            ("2023-01-01..end (extended holdout, spans the splice)", "2023-01-01", None),
            ("full series 2017-2026", None, None)):
        frame = _slice(DF, lo, hi).copy()
        base = v4().prepare(frame.copy())["target"].to_numpy()
        for w in SWEEP:
            got = FundingGateDecileExtended(funding_window_days=w,
                                            decile=1.1).prepare(frame.copy())
            diff = float(np.nanmax(np.abs(got["target"].to_numpy() - base)))
            pctl = got["funding_pctl"].to_numpy()
            finite = pctl[np.isfinite(pctl)]
            rng = (f"[{finite.min():.3f}, {finite.max():.3f}]" if len(finite)
                   else "(none finite)")
            print(f"  {label:52s} w={str(w):9s} {len(frame):>9,d} bars  "
                  f"max|Δtarget|={diff:.3e}  pctl range {rng}  "
                  f"{'PASS' if diff == 0.0 else 'FAIL'}")


# ---------------------------------------------------- 2. causality probe

def causality() -> None:
    """Two-opposite-tampers, by hand (an unregistered strategy gets no CI).

    ``tests/test_causality_strict.py`` parametrizes over the *registry*,
    so nothing here is covered by it. Procedure (R-28/R-32/R-35's):
    multiply OHLC by 3 from a cut point onward in one copy and divide by
    3 in the other, then require every prepared column and every order
    decision at or before the cut to be bit-identical. Comparing the
    prepared columns directly -- not just the orders -- is what catches
    the failure mode this project has been bitten by specifically: a
    global mean/std/quantile computed over the whole series and applied
    to early rows.

    Both ``_funding_percentile`` code paths (``.rolling`` and
    ``.expanding``) are checked, and the probe is run on a slice that
    ends inside the Deribit extension so the splice logic is exercised.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    identity_check()
    probes = (("Deribit region", "2024-03-01", None),
              ("spans the Binance->Deribit splice", "2022-06-01", "2024-12-31"),
              ("R-35's own region", "2020-06-01", "2022-12-31"))
    for label, lo, hi in probes:
        frame = _slice(DF, lo, hi).copy()
        cut = len(frame) - 20_000
        bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000, 5_000)]
        print(f"\ntamper probe [{label}]: {len(frame):,} bars "
              f"{frame.index[0]:%Y-%m-%d} -> {frame.index[-1]:%Y-%m-%d}, "
              f"cut at bar {cut:,} ({frame.index[cut]:%Y-%m-%d %H:%M})")

        up, down = frame.copy(), frame.copy()
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

        for w in SWEEP:
            def decisions(f):
                s = gate(w)
                prepared = s.prepare(f.copy())
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
            worst = {}
            for c in ("target", "v4_target", "funding_pctl", "gated"):
                x = pa[c].to_numpy()[:cut].astype(float)
                y = pb[c].to_numpy()[:cut].astype(float)
                d = np.abs(np.nan_to_num(x, nan=-1.0) - np.nan_to_num(y, nan=-1.0))
                worst[c] = float(np.max(d)) if len(d) else 0.0
            ok = not bad and max(worst.values()) == 0.0
            print(f"  w={str(w):9s} orders "
                  f"{'PASS' if not bad else 'FAIL at ' + str(bad)}   "
                  + "  ".join(f"{c}={worst[c]:.3e}" for c in worst)
                  + f"   {'PASS' if ok else 'FAIL'}")

    # Second, orthogonal leak check, aimed at the specific failure mode
    # this project has been bitten by: a full-series statistic (global
    # mean/std/quantile) applied to early rows. If one existed, dropping
    # later settlements would move an earlier row's rank. It must not.
    print("\ntruncation probe: rank(t) must not depend on settlements after t")
    for measured_lo, measured_hi, drop_from in (
            ("2021-01-01", "2022-12-31", "2023-01-01"),
            ("2023-01-01", "2023-12-31", "2024-01-01"),
            ("2024-01-01", "2024-12-31", "2025-01-01"),
            ("2025-01-01", "2025-12-31", "2026-01-01")):
        idx = _slice(DF, measured_lo, measured_hi).index
        cutoff = pd.Timestamp(drop_from, tz="UTC")
        short = FUNDING_EXT[FUNDING_EXT.index < cutoff]
        worst = {}
        for w in SWEEP:
            full = _funding_percentile(FUNDING_EXT, idx, w, 3.0, 30)
            trunc = _funding_percentile(short, idx, w, 3.0, 30)
            d = np.abs(np.nan_to_num(full, nan=-1.0)
                       - np.nan_to_num(trunc, nan=-1.0))
            worst[str(w)] = float(d.max())
        ok = max(worst.values()) == 0.0
        print(f"  measured {measured_lo}..{measured_hi}, all settlements "
              f">= {drop_from} dropped: max|Δpctl| "
              + "  ".join(f"w={k}:{v:.1e}" for k, v in worst.items())
              + f"   {'PASS' if ok else 'FAIL'}")

    # Third: the funding series itself must not be reachable from price.
    # Perturbing every price bar leaves funding_pctl untouched everywhere,
    # which is the whole point of using a non-price signal (R-16/R-35).
    print("\nindependence probe: funding_pctl must not move when ALL prices move")
    frame = _slice(DF, "2023-01-01", None).copy()
    scaled = frame.copy()
    for col in ("open", "high", "low", "close"):
        scaled[col] *= 5.0
    a = gate(DECISION_W).prepare(frame.copy())["funding_pctl"].to_numpy()
    b = gate(DECISION_W).prepare(scaled)["funding_pctl"].to_numpy()
    d = np.abs(np.nan_to_num(a, nan=-1.0) - np.nan_to_num(b, nan=-1.0))
    print(f"  w={DECISION_W} max|Δfunding_pctl| over the whole holdout = "
          f"{float(d.max()):.3e} {'PASS' if d.max() == 0.0 else 'FAIL'}")


# --------------------------------------------------------- 3. coverage

def coverage() -> None:
    """Funding coverage, gate-fire rate, and the Binance/Deribit split."""
    identity_check()
    print("funding series:")
    for name, s in (("R-35 (Binance only)", FUNDING_R35),
                    ("R-39 extended", FUNDING_EXT)):
        print(f"  {name:22s} {len(s):>6,d} settlements  "
              f"{s.index[0]:%Y-%m-%d} -> {s.index[-1]:%Y-%m-%d}")
    print(f"  source split: " + ", ".join(
        f"{k}={v:,}" for k, v in FUNDING_SRC.value_counts().items()))
    print("\nannualized cost to a constant long, by year (extended series):")
    for year, grp in FUNDING_EXT.groupby(FUNDING_EXT.index.year):
        src = FUNDING_SRC[grp.index].unique()
        print(f"  {year}  {grp.mean() * 3 * 365.25:+7.2%}  "
              f"positive {(grp > 0).mean():>4.0%}  n={len(grp):>5,d}  "
              f"source={'/'.join(src)}")

    print("\nper-slice coverage and gate-fire rate "
          f"(w={DECISION_W}, decile={DECILE}):")
    print(f"  {'slice':34s} {'bars':>10s} {'covered':>9s} {'fire(all)':>10s} "
          f"{'fire(covered)':>14s} {'binance':>8s} {'deribit':>8s}")
    slices = (("inner-train 2017-2020", *TRAIN),
              ("inner-validation 2021-2022", *VALID),
              ("holdout 2023-01-01 -> end", *OOS),
              ("  holdout 2023 only", *OOS_2023),
              ("  holdout 2024-2026", *OOS_2024P))
    for label, lo, hi in slices:
        idx = _slice(DF, lo, hi).index
        for tag, series in (("ext", FUNDING_EXT),):
            pctl = _funding_percentile(series, idx, DECISION_W, 3.0, 30)
            covered = np.isfinite(pctl)
            fired = covered & (pctl >= DECILE)
            src = FUNDING_SRC.reindex(
                idx.union(FUNDING_SRC.index)).sort_index().ffill().reindex(idx)
            src = src.where(idx <= series.index.max())
            nb = float((src == "binance").mean())
            nd = float((src == "deribit").mean())
            print(f"  {label:34s} {len(idx):>10,d} {covered.mean():>8.1%} "
                  f"{fired.mean():>9.1%} "
                  f"{(fired.sum() / max(covered.sum(), 1)):>13.1%} "
                  f"{nb:>7.1%} {nd:>7.1%}")

    print("\ngate-fire rate by calendar year (w=180), the splice seam check:")
    idx = _slice(DF, "2020-01-01", None).index
    pctl = pd.Series(_funding_percentile(FUNDING_EXT, idx, DECISION_W, 3.0, 30),
                     index=idx)
    fired = pctl >= DECILE
    for year, grp in fired.groupby(fired.index.year):
        cov = pctl[pctl.index.year == year].notna().mean()
        print(f"  {year}  fire={grp.mean():>6.1%}  covered={cov:>6.1%}")


# ------------------------------------------- 4. reproduce R-35 inner-val

def validation() -> None:
    """R-35's inner-validation table, recomputed with the extended loader.

    2021-2022 funding is Binance-sourced in BOTH loaders (the extension
    only adds settlements *after* 2023-12-31), so every number here must
    match R-35's report exactly. A difference means the loader changed
    something it should not have, and the pre-registration says
    investigate rather than proceed.
    """
    identity_check()
    for label, period in (("inner-train", TRAIN), ("inner-validation", VALID)):
        for market, mname in ((FUTURES, "futures5x"), (SPOT, "spot")):
            print(f"\n{label} / {mname}:")
            ev(v4(), *period, market=market, tag="kelly_regime_v4")
            for w in SWEEP:
                a = ev(gate(w), *period, market=market, tag=f"gate(ext) w={w}")
                b = ev(gate_r35(w), *period, market=market,
                       tag=f"gate(R-35) w={w}")
                same = (abs(a["final"] - b["final"]) < 1e-9
                        and a["trades"] == b["trades"])
                print(f"    -> extended vs R-35 series identical: "
                      f"{'YES' if same else 'NO  <-- INVESTIGATE'}")


# --------------------------------------------- 5. THE pre-registered read

def _curves(arms, period, market, funding):
    out = {}
    for tag, strat in arms:
        r = run_period_funding(strat, DF, *period, market=market, funding=funding)
        describe(r, tag)
        out[tag] = daily_returns(r.equity)
        global N_EVALUATED
        N_EVALUATED += 1
    return out


def _paired(curves, base_tag, period_label):
    n = min(len(v) for v in curves.values())
    idx = stationary_bootstrap_indices(n, MEAN_BLOCK, N_BOOT,
                                       np.random.default_rng(SEED))
    print(f"\n  {period_label}: paired stationary block bootstrap "
          f"({MEAN_BLOCK:.0f}-day mean block, {N_BOOT:,} resamples, "
          f"identical indices for both arms, n={n} days)")
    base = curves[base_tag].to_numpy()[:n]
    for tag, series in curves.items():
        if tag == base_tag:
            continue
        for stat, label in ((total_log_return, "Δ log growth"),
                            (annualized_sharpe, "Δ Sharpe    "),
                            (max_drawdown_from_returns, "Δ max DD %  ")):
            res = paired_bootstrap(series.to_numpy()[:n], base, stat,
                                   indices=idx)
            mark = "*" if res.significant else " "
            print(f"    {label} {tag:26s} vs {base_tag:16s} "
                  f"{res.diff.point:>+8.3f} [{res.diff.lo:>+8.3f}, "
                  f"{res.diff.hi:>+8.3f}]{mark} P(>0)={res.p_positive:.3f}")


def holdout() -> None:
    """Step 4. THE pre-registered decision: w=180, full extended holdout, futures.

    Funding charged as a first-class cost (the ``funding_study.py``
    convention) is the primary cell -- this is a COST-axis mechanism and
    the pre-registration's falsification test is that it must survive the
    bill it was built to reduce. The funding-free cell is reported beside
    it because every other futures figure in this repo is funding-free
    and a reader needs the comparison; it is not the decision cell.
    """
    identity_check()
    arms = [("kelly_regime_v4", v4()),
            (f"gate w={DECISION_W} (FROZEN)", gate(DECISION_W))]
    print(f"HOLDOUT {OOS[0]} -> {DF.index[-1]:%Y-%m-%d}, futures5x, "
          "FUNDING CHARGED (decision cell):")
    c = _curves(arms, OOS, FUTURES, FUNDING_EXT)
    _paired(c, "kelly_regime_v4", "futures5x / funding charged")

    print(f"\nHOLDOUT {OOS[0]} -> {DF.index[-1]:%Y-%m-%d}, futures5x, "
          "funding-free (context, not the decision cell):")
    c = _curves(arms, OOS, FUTURES, None)
    _paired(c, "kelly_regime_v4", "futures5x / funding free")

    print(f"\nHOLDOUT {OOS[0]} -> {DF.index[-1]:%Y-%m-%d}, spot "
          "(diagnostic: isolates R-16's return-forecast channel):")
    c = _curves(arms, OOS, SPOT, None)
    _paired(c, "kelly_regime_v4", "spot")

    print("\ncontext (not part of the decision rule):")
    ev(get_strategy("buy_and_hold"), *OOS, market=SPOT, tag="buy_and_hold spot")


# ------------------------------------------------------- 6. neighbourhood

def neighbours() -> None:
    """w=90/365/expanding on the same holdout -- PLATEAU CONTEXT, NOT A MENU.

    w=180 is the pre-registered decision config. These are reported so a
    reader can see whether the neighbourhood is flat or spiky; picking
    the best of them after the fact would be exactly the goalpost move
    ROUTINE.md step 4 forbids, and R-12 is what that produces.
    """
    identity_check()
    arms = [("kelly_regime_v4", v4())] + \
           [(f"gate w={w}", gate(w)) for w in SWEEP]
    print(f"HOLDOUT {OOS[0]} -> end, futures5x, FUNDING CHARGED, all four windows:")
    c = _curves(arms, OOS, FUTURES, FUNDING_EXT)
    _paired(c, "kelly_regime_v4", "futures5x / funding charged")


# ------------------------------------------------------------- 7. costs

def costs() -> None:
    """The 0.40% taker tier, and the exposure-artifact control."""
    identity_check()
    arms = [("kelly_regime_v4", v4()),
            (f"gate w={DECISION_W}", gate(DECISION_W))]

    for tier in (0.0005, 0.004):
        print(f"\nHOLDOUT futures5x @ taker {tier:.2%}, funding charged:")
        m = MarketSpec.futures(leverage=5.0, fee_rate=tier)
        for tag, s in arms:
            ev(s, *OOS, market=m, funding=FUNDING_EXT, tag=tag)
    for tier in (0.001, 0.004):
        print(f"\nHOLDOUT spot @ taker {tier:.2%}:")
        m = MarketSpec.spot(fee_rate=tier)
        for tag, s in arms:
            ev(s, *OOS, market=m, tag=tag)

    print("\nexposure-artifact check (the standing L-04/R-33 rule): "
          "mean|target| and the flat-rescale control")
    for period, plabel in ((OOS, "holdout 2023+"), (VALID, "inner-val 2021-22")):
        g = ev(gate(DECISION_W), *period, market=FUTURES, funding=FUNDING_EXT,
               tag=f"gate w={DECISION_W} [{plabel}]")
        b = ev(v4(), *period, market=FUTURES, funding=FUNDING_EXT,
               tag=f"kelly_regime_v4 [{plabel}]")
        ratio = g["exposure"] / b["exposure"]
        print(f"    mean|target| gate={g['exposure']:.4f} v4={b['exposure']:.4f} "
              f"ratio={ratio:.3f}   gate fires on {g['fire']:.1%} of bars")
        r = ev(ScaledV4(scale=ratio), *period, market=FUTURES,
               funding=FUNDING_EXT, tag=f"v4 x{ratio:.3f} (rescale control)")
        print(f"    -> matched-exposure control reaches sharpe={r['sharpe']:.2f} "
              f"DD={r['dd']:.1f}% final=${r['final']:,.0f}; "
              f"gate reaches sharpe={g['sharpe']:.2f} DD={g['dd']:.1f}% "
              f"final=${g['final']:,.0f}")
        # Paired interval on the artifact question itself, not just points.
        cur = {}
        for tag, s in ((f"gate w={DECISION_W}", gate(DECISION_W)),
                       (f"v4 x{ratio:.3f}", ScaledV4(scale=ratio))):
            res = run_period_funding(s, DF, *period, market=FUTURES,
                                     funding=FUNDING_EXT)
            cur[tag] = daily_returns(res.equity)
        _paired(cur, f"v4 x{ratio:.3f}", f"{plabel}: gate vs matched-exposure v4")


# ---------------------------------------------------------- 8. sub-periods

def subperiod() -> None:
    """2023 (Binance funding, already seen by R-35) vs 2024-2026 (Deribit, new).

    If the edge lives only in 2023 it is R-35's already-read result and
    not new evidence; if it lives in 2024-2026 it is the first genuinely
    unseen funding data this mechanism has met.
    """
    identity_check()
    arms = [("kelly_regime_v4", v4()),
            (f"gate w={DECISION_W}", gate(DECISION_W))]
    for period, label in ((OOS_2023, "2023 only (Binance-sourced funding)"),
                          (OOS_2024P, "2024-01-01 -> end (Deribit-sourced)")):
        print(f"\nHOLDOUT sub-period {label}, futures5x, funding charged:")
        c = _curves(arms, period, FUTURES, FUNDING_EXT)
        _paired(c, "kelly_regime_v4", label)


# ------------------------------------------------- 9. seam robustness check

def seam() -> None:
    """Post-hoc: is the 2024-2026 negative an artifact of the venue splice?

    NOT part of the pre-registered decision and it cannot rescue it --
    the pre-registration fixes ``load_funding_extended()`` as the input.
    It is here because a reader is entitled to know whether the result
    survives removing the one structural oddity this branch inherited:
    for ~one window-length after 2024-01-01 the trailing rank of a
    Deribit rate is taken against a partly-Binance history, and the two
    venues sit at different levels.

    The check re-runs the identical frozen configuration on 2024-2026
    with a *pure Deribit* funding series (``load_funding_deribit``, no
    splice at all, so every rank is Deribit-vs-Deribit) and on the
    2025-01-01+ slice, which is more than a full 365-day window past the
    splice for every swept lookback. If the negative persists in both,
    the seam is not the explanation.
    """
    from tradebot.data import load_funding_deribit

    identity_check()
    deribit = load_funding_deribit(ROOT / "data")
    for label, period, series in (
            ("2024+ / spliced series (as pre-registered)", OOS_2024P, FUNDING_EXT),
            ("2024+ / pure Deribit series (no splice)", OOS_2024P, deribit),
            ("2025+ / spliced, > 1 full 365d window past the splice",
             ("2025-01-01", None), FUNDING_EXT)):
        print(f"\n{label}, futures5x, funding charged "
              f"(gate ranks against {'Deribit only' if series is deribit else 'the spliced series'}):")
        arms = [("kelly_regime_v4", v4()),
                (f"gate w={DECISION_W}",
                 FundingGateDecileExtended(funding_window_days=DECISION_W,
                                           decile=DECILE, funding=series))]
        c = _curves(arms, period, FUTURES, series)
        _paired(c, "kelly_regime_v4", label)


# ------------------------------------------- 10. descriptive: why it failed

def why() -> None:
    """Descriptive only, no backtests: what the gate was standing aside for.

    R-16's premise is that top-decile funding forecasts weak forward
    returns. This measures that premise directly, per slice, with the
    same causal percentile column the gate itself uses: mean forward
    14-day log return on gated bars vs ungated bars. It generates no
    Sharpe and evaluates no configuration, so it is not in the trials
    tally -- it is here to give the ledger row a mechanism rather than
    just a sign.
    """
    identity_check()
    horizon = 14 * 288
    close = DF["close"]
    fwd = np.log(close.shift(-horizon) / close)
    pctl = pd.Series(_funding_percentile(FUNDING_EXT, DF.index, DECISION_W,
                                         3.0, 30), index=DF.index)
    fired = pctl >= DECILE
    print(f"mean forward {horizon // 288}-day log return, gated vs ungated bars "
          f"(w={DECISION_W}, decile={DECILE}):")
    print(f"  {'slice':40s} {'gated':>9s} {'ungated':>9s} {'spread':>9s} {'n_gated':>9s}")
    for label, lo, hi in (("inner-validation 2021-2022", *VALID),
                          ("holdout 2023 (Binance funding)", *OOS_2023),
                          ("holdout 2024-2026 (Deribit funding)", *OOS_2024P),
                          ("holdout 2023-01-01 -> end (all)", *OOS)):
        m = _slice(DF, lo, hi).index
        g, f = fired.reindex(m), fwd.reindex(m)
        a = float(f[g & f.notna()].mean())
        b = float(f[~g & f.notna()].mean())
        print(f"  {label:40s} {a:>+9.4f} {b:>+9.4f} {a - b:>+9.4f} "
              f"{int(g.sum()):>9,d}")
    print("\nR-16's premise is that the top decile forecasts a WEAK forward\n"
          "return, i.e. a negative spread. A positive spread means the gate\n"
          "stood aside during the better half of the tape.")


COMMANDS = {"build": build, "causality": causality, "coverage": coverage,
            "validation": validation, "holdout": holdout,
            "neighbours": neighbours, "costs": costs, "subperiod": subperiod,
            "seam": seam, "why": why}


def main() -> None:
    if FUNDING_EXT is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d} "
          f"(data: {LABEL})", file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "all":
        for name, fn in COMMANDS.items():
            print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
            fn()
    elif choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python {Path(__file__).name} "
              f"[{'|'.join(COMMANDS)}|all]")
        return
    print(f"\nconfigurations evaluated this invocation: {N_EVALUATED}")


if __name__ == "__main__":
    main()
