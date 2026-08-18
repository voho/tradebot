#!/usr/bin/env python
"""B-05: funding as a gate on kelly_regime_v4 (stand flat in the top decile).

Not registered: lives under ``experiments/`` so it is not auto-discovered,
per ROUTINE.md step 5. Promote into ``src/tradebot/strategies/`` only if it
clears the promotion bar (it does not, see the frozen-config verdict at the
bottom of this file's driver).

The idea
--------
``kelly_regime_v4`` decides "how much to hold" from price alone. R-14 found
funding is not just a cost but an ADVERSELY TIMED one: while the strategy
holds, BTC perp funding runs about +20%/yr; while it is flat, about +2.8%/yr
- because the crowding that produces the strategy's own bullish vote is
exactly what sets the funding rate. R-16, independently, found that a high
trailing-funding state predicts NEGATIVE 14-day forward returns unless price
is also rising, and that this is not simply a momentum proxy (correlation
with trailing return only 0.39) - though the middle quintiles are
non-monotone, a real noise warning.

Mechanism (one sentence): when trailing realized funding is unusually rich
- in the top decile of its own recent, CAUSAL history - the strategy stands
flat (or scales down), regardless of what the regime vote says, because
that state is both the most expensive state to be long in (R-14) and not
reliably still a good state to be long in (R-16).

This is a COST-constraint move primarily: the mechanism as implemented is a
funding-cost avoidance filter layered on an unchanged sizing/vote engine.
It also leans on INFO in the sense that the *causal quantile threshold*
generalizes R-16's descriptive, non-causal quintile sort into a forward
usable trading rule - if that generalization is where the edge lives (not
just cost-dodging), the honest label is INFO too. Read it as attacking both,
primarily COST.

Not a duplicate of R-14 (which measured the cost of funding on the
UNCHANGED v4 rule and made no trading-rule change - a pure accounting
exercise) or of R-16 (a return-prediction study on spot forward returns,
with no simulated strategy, no costs, no position sizing, no turnover). This
file is the first place funding is used as an INPUT to a trading decision
rather than only as an input to the P&L.

Data constraint, stated up front and not hidden: real committed Binance
BTCUSDT funding (``data/btcusdt_perp_funding_8h.csv.gz``) covers only
2020-01-01 03:00 UTC -> 2023-12-31 19:00 UTC. Outside that window the gate
has nothing to act on and MUST fall back to plain ``kelly_regime_v4``
behavior (never error, never impute a rate) - see ``_gate_series`` below.
This confines the whole study, train through holdout, to inside that
window; there is no "extend to 2026" available here (B-02 is blocked on
network access) and no synthesized filler is used anywhere in this file.

What would make it fail (named before any code ran): the top-decile funding
state is short-lived and so tightly coupled to the strategy's own
already-bullish state (both are downstream of the same crowding) that
flattening out of it mostly adds turnover on top of an already-low-turnover
strategy, and the extra fees from the flatten/re-entry cycles exceed the
funding actually saved - particularly once the taker fee is raised from the
table's assumed 0.10%/0.05% to Bitstamp's real 0.40% entry tier. That is
this file's one pre-registered falsification test (see ``falsify()``).

Sources consulted (2025-2026 web search on funding-crowding as a sizing /
gating signal, not just carry harvest): the "crowded trade" framing of perp
funding used in industry carry-harvest write-ups generally treats a funding
extreme as a DIRECTIONAL fade signal (go short into rich funding). This
file deliberately does NOT do that - R-12 in this repo's ledger is the
standing record of what high-turnover standalone reversal use of a
crowding signal costs here. The design below only ever REMOVES or REDUCES
an existing long exposure; it never adds a new short. No single paper
sharpened the design enough to change a threshold or window choice, so
none is cited as load-bearing; the mechanism is grounded entirely in this
repo's own R-14/R-16 findings as instructed.

Variants (5, all layered on kelly_regime_v3/v4's existing 20/40/80d vote +
conditional-vol target; only the FINAL exposure is touched)
-----------------------------------------------------------
V1 FLAT-EXP-D9   hard flatten, top DECILE (q=0.90), EXPANDING causal
                 quantile, level = trailing 9-settlement (3-day) mean rate.
V2 FLAT-ROLL-D9  same as V1 but a ROLLING 1-year (1095-settlement) causal
                 quantile window instead of expanding.
V3 SCALE-EXP-D9  same as V1 but SCALE exposure to 0.3x instead of hard
                 flatten (partial de-risk rather than full exit).
V4 FLAT-EXP-Q80  same as V1 but top QUINTILE (q=0.80) instead of decile -
                 a looser trigger, closer to R-16's coarser quintile sort.
V5 FLAT-EXP-ROC  gate on the RATE-OF-CHANGE of trailing funding (3-day
                 change) rather than its level, top decile, expanding
                 quantile - tests "is it richness or the SPEED of
                 richening that matters".

Causality of the gate itself (read this before trusting any number below)
---------------------------------------------------------------------------
``_gate_series`` computes, on the funding-settlement time axis (NOT the 5m
bar axis):

    level[t]  = rolling_mean(funding, K)[t]            # ends AT t, so known at t
    metric[t] = level[t]              (or level.diff(L)[t] for the RoC variant)
    thresh[t] = expanding_or_rolling_quantile(metric.shift(1), q)[t]
    active[t] = metric[t] >= thresh[t]

``metric.shift(1)`` means the value the quantile machinery sees for
settlement ``t`` is ``metric[t-1]``, one settlement EARLIER. Because
``.expanding()``/``.rolling()`` at position ``t`` only ever look at rows
``<= t`` of the series they are called on, ``thresh[t]`` is therefore a
function of ``{metric[s] : s < t}`` only - strictly prior settlements,
never ``metric[t]`` itself or anything later. ``active[t]`` then compares
the (t-known) ``metric[t]`` against that (strictly-prior) threshold. This
is an expanding/rolling quantile of the funding series' OWN PAST, never a
quantile of the whole series - the full-series-fit lookahead class R-28 and
ROUTINE.md warn a truncation test alone will not catch. Verified two ways
in ``causality()`` below: (1) by direct inspection of the shift, printed at
a handful of settlements; (2) by the standard two-opposite-tampers
truncation check, run on price, exactly as R-28 ran it by hand for its own
unregistered strategy.

The settlement-time gate is then aligned onto the 5m bar index by
forward-fill, and explicitly zeroed outside ``[funding.index.min(),
funding.index.max()]`` so a bar after the data ends does not silently
inherit the last known gate state forever - that would be exactly the
"proxy unavailable data" failure this project's INFO rows already paid
for.
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

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL_FUNDING = load_funding(ROOT / "data")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

FUND_START = REAL_FUNDING.index.min()
FUND_END = REAL_FUNDING.index.max()

# THIS study's split (not the repo's usual 2017/2021/2023 one): real funding
# only covers 2020-01 -> 2023-12, so both inner slices must fit inside it.
TRAIN = ("2020-01-01", "2021-12-31")
VALID = ("2022-01-01", "2022-12-31")
# Holdout: 2023-01-01 through the LAST date load_funding actually returns
# data for (computed from the data, not assumed - see main() banner).
HOLDOUT = ("2023-01-01", FUND_END.strftime("%Y-%m-%d"))

N_EVALUATED = 0  # every distinct configuration this file scores, for deflated Sharpe


# --------------------------------------------------------------------- strategy


class FundingGatedV4(KellyRegimeV3):
    """kelly_regime_v4 with an added funding gate: flat/scaled in the top
    decile of trailing, causally-quantiled funding richness (backlog B-05).
    """

    name = "b05_funding_gated_v4"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(
        self,
        funding: pd.Series | None = None,
        gate_quantile: float = 0.90,
        gate_mode: str = "flatten",       # "flatten" | "scale"
        scale_factor: float = 0.3,
        level_settlements: int = 9,        # 3 days @ 8h settlements
        window_mode: str = "expanding",    # "expanding" | "rolling"
        rolling_settlements: int = 1095,   # 365d * 3 settlements/day
        min_settlements: int = 270,        # 90 days before the gate can fire
        use_roc: bool = False,
        roc_lag_settlements: int = 9,
        horizons: tuple[int, ...] = (20, 40, 80),
        **kwargs,
    ) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.funding = funding
        self.gate_quantile = gate_quantile
        self.gate_mode = gate_mode
        self.scale_factor = scale_factor
        self.level_settlements = level_settlements
        self.window_mode = window_mode
        self.rolling_settlements = rolling_settlements
        self.min_settlements = min_settlements
        self.use_roc = use_roc
        self.roc_lag_settlements = roc_lag_settlements

    # ---- the causal gate ---------------------------------------------------

    def _gate_series(self, index: pd.DatetimeIndex) -> np.ndarray:
        """1.0 where the gate is ACTIVE (top-decile funding state), else 0.0.

        0.0 (never active) wherever funding data does not exist - before
        the committed history starts, after it ends, or inside the
        ``min_settlements`` causal warmup - so the strategy falls back
        cleanly to plain v3/v4 behavior there.
        """
        if self.funding is None or len(self.funding) == 0:
            return np.zeros(len(index))

        f = self.funding.sort_index()
        level = f.rolling(self.level_settlements,
                          min_periods=self.level_settlements).mean()
        metric = level.diff(self.roc_lag_settlements) if self.use_roc else level

        shifted = metric.shift(1)  # strictly-before values only, see module docstring
        if self.window_mode == "expanding":
            thresh = shifted.expanding(min_periods=self.min_settlements).quantile(
                self.gate_quantile)
        else:
            thresh = shifted.rolling(self.rolling_settlements,
                                     min_periods=self.min_settlements).quantile(
                self.gate_quantile)

        active = (metric >= thresh).astype(float)
        active = active.where(thresh.notna() & metric.notna(), 0.0)

        aligned = active.reindex(index, method="ffill")
        in_range = (index >= f.index.min()) & (index <= f.index.max())
        return np.where(in_range, aligned.fillna(0.0).to_numpy(), 0.0)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v3's latched-vote / conditional-vol target, untouched
        gate = self._gate_series(df.index)
        base = df["target"].to_numpy()
        gated = (np.where(gate > 0.5, 0.0, base) if self.gate_mode == "flatten"
                 else np.where(gate > 0.5, base * self.scale_factor, base))
        df["target"] = gated
        df["funding_gate_active"] = gate
        return df


# ----------------------------------------------------------------------- runner


def ev(strategy, start, end, df=None, market=SPOT, tag="", balance=1_000.0,
       count=True, funding=None):
    """One backtest, one line, counted. Mirrors experiments/run_eprocess.py's ev()."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    if funding is None:
        result = run_period(strategy, frame, start, end, market=market,
                            start_balance=balance, data_label=LABEL)
    else:
        # run_period has no funding parameter; replicate its warmup-prefix
        # trimming by hand, the same pattern scripts/funding_study.py uses.
        lo = 0 if start is None else int(frame.index.searchsorted(start))
        hi = len(frame) if end is None else int(frame.index.searchsorted(end, side="right"))
        pre = min(lo, strategy.warmup)
        raw = run_backtest(strategy, frame.iloc[lo - pre: hi], market, balance,
                           trade_start=pre, funding=funding, data_label=LABEL)
        result = raw if pre == 0 else replace(
            raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
    m = compute_metrics(result)
    print(f"  {tag or strategy.name:32s} {market.name:11s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"trades={m.num_trades:>4d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} funding=${result.funding_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")
    return m, result


def _variants():
    """The 5 pre-registered gate variants. Each entry is one distinct configuration."""
    return [
        ("V1 FLAT-EXP-D9  ", dict(gate_quantile=0.90, gate_mode="flatten",
                                  window_mode="expanding", level_settlements=9)),
        ("V2 FLAT-ROLL-D9 ", dict(gate_quantile=0.90, gate_mode="flatten",
                                  window_mode="rolling", level_settlements=9)),
        ("V3 SCALE-EXP-D9 ", dict(gate_quantile=0.90, gate_mode="scale",
                                  scale_factor=0.3, window_mode="expanding",
                                  level_settlements=9)),
        ("V4 FLAT-EXP-Q80 ", dict(gate_quantile=0.80, gate_mode="flatten",
                                  window_mode="expanding", level_settlements=9)),
        ("V5 FLAT-EXP-ROC ", dict(gate_quantile=0.90, gate_mode="flatten",
                                  window_mode="expanding", use_roc=True,
                                  level_settlements=9, roc_lag_settlements=9)),
    ]


def _benchmarks(start, end, market, label):
    print(f"\n{label} benchmarks:")
    for name in ("buy_and_hold", "kelly_regime_v4"):
        # Charge real funding here too (harmless on spot - market.pays_funding
        # gates it inside the engine) so the comparison against the gated
        # variants below is apples-to-apples, not gate-pays / plain-v4-free.
        ev(get_strategy(name), start, end, market=market, tag=f"  {name}",
           count=False, funding=REAL_FUNDING)


def sweep() -> None:
    """Step 3: inner-train + inner-validation, both markets, all 5 variants."""
    for market, mname in ((FUTURES, "futures 5x"), (SPOT, "spot")):
        for (start, end), split in ((TRAIN, "INNER-TRAIN"), (VALID, "INNER-VALIDATION")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            print(f"{split} / {mname} variants (funding gate, real funding applied):")
            for tag, kw in _variants():
                s = FundingGatedV4(funding=REAL_FUNDING, **kw)
                ev(s, start, end, market=market, tag=tag, funding=REAL_FUNDING)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


def neighbours() -> None:
    """Plateau check around the selected winner (V1), one knob at a time."""
    base = dict(gate_quantile=0.90, gate_mode="flatten", window_mode="expanding",
               level_settlements=9)
    grid = [("base V1 q=.90 K=9        ", {})]
    grid += [(f"q={q:.2f}                   ", dict(gate_quantile=q))
             for q in (0.85, 0.95)]
    grid += [(f"K={k}d settlements         ", dict(level_settlements=k))
             for k in (3, 27)]  # 1 day, 9 days
    grid += [("min_settlements=90        ", dict(min_settlements=90)),
             ("min_settlements=540       ", dict(min_settlements=540))]
    for market, mname in ((FUTURES, "futures 5x"), (SPOT, "spot")):
        print(f"\nINNER-VALIDATION neighbourhood / {mname}:")
        for tag, kw in grid:
            s = FundingGatedV4(funding=REAL_FUNDING, **{**base, **kw})
            ev(s, *VALID, market=market, tag=tag, funding=REAL_FUNDING)
        print(f"INNER-TRAIN neighbourhood / {mname}:")
        for tag, kw in grid:
            s = FundingGatedV4(funding=REAL_FUNDING, **{**base, **kw})
            ev(s, *TRAIN, market=market, tag=tag, funding=REAL_FUNDING, count=False)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


# ------------------------------------------------------------------- causality


def causality() -> None:
    """Both mandatory causality checks for the funding gate.

    (1) direct inspection: print the shift/expanding-quantile relationship
        at a handful of settlements, so the "strictly before" claim in the
        module docstring can be read off actual numbers, not just code.
    (2) the two-opposite-tampers truncation check R-28 used on price, run
        by hand exactly as experiments/run_eprocess.py does it (this file
        gets none of test_causality_strict.py's protection - it only
        parametrizes over the registry).
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    s = FundingGatedV4(funding=REAL_FUNDING)
    f = REAL_FUNDING.sort_index()
    level = f.rolling(9, min_periods=9).mean()
    shifted = level.shift(1)
    thresh = shifted.expanding(min_periods=270).quantile(0.90)

    print("(1) direct inspection - threshold at t uses only settlements < t:\n")
    print(f"{'settlement t':<22s} {'level[t]':>11s} {'thresh[t]':>11s}  "
          f"{'= q90 of level[:t-1]?':>22s}")
    check_at = [500, 1500, 2500, 3500, 4300]
    ok = True
    for pos in check_at:
        t = level.index[pos]
        manual = float(level.iloc[:pos].quantile(0.90))  # level[0 .. pos-1], i.e. strictly < t
        engine = float(thresh.iloc[pos])
        match = abs(manual - engine) < 1e-12
        ok &= match
        print(f"{str(t):<22s} {level.iloc[pos]:>11.6f} {engine:>11.6f}  "
              f"{'PASS' if match else 'FAIL'} (manual={manual:.6f})")
    print(f"\n  {'PASS' if ok else 'FAIL'}: thresh[t] recomputed directly from "
          "level[0:t] (excluding t) matches the shift(1)+expanding().quantile()\n"
          "  pipeline bit-for-bit at every sampled t. No row's threshold uses its\n"
          "  own or any later settlement.")

    print("\n(2) two-opposite-tampers truncation check, on PRICE (R-28's design):\n")
    df = DF.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        strat = FundingGatedV4(funding=REAL_FUNDING)
        prepared = strat.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            strat.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    a, b = decisions(up), decisions(down)
    bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
    print(f"  tampered from bar {cut:,} of {len(df):,}; checked bars {bars}")
    print("  FAIL - reads the future at bars " + str(bad) if bad
          else "  PASS - every decision at or before the cut is unchanged")

    pa = FundingGatedV4(funding=REAL_FUNDING).prepare(up.copy())
    pb = FundingGatedV4(funding=REAL_FUNDING).prepare(down.copy())
    for col in ("target", "funding_gate_active"):
        diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
        worst = float(np.nanmax(diff))
        print(f"  column {col:22s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if worst < 1e-12 else 'FAIL'}")
    print("\n  (funding_gate_active is expected to be identical regardless of the\n"
          "  price tamper - it depends only on the funding series - so this is\n"
          "  really re-confirming the inherited v3 price-causality plus checking\n"
          "  the override logic does not accidentally read price.)")


# --------------------------------------------------------------------- holdout

# Frozen BEFORE the holdout (2023-01-01 -> real-funding-end) was read.
#
# Selection basis, stated honestly: inner-validation (2022) turned out to be
# a DEGENERATE test for this idea - 2022's trailing funding never reaches
# the expanding top-decile-of-2020-2021 threshold for ANY of the 5 variants
# or the 6 neighbour perturbations (measured gate-active fraction in 2022:
# 0.0-0.1% of bars for every configuration), so every configuration scores
# IDENTICALLY to plain kelly_regime_v4 on inner-validation. Inner-validation
# therefore confirms only "the gate does no harm when it does not fire", not
# which variant is best - the selection below is made on INNER-TRAIN alone,
# which is weaker evidence than the routine's usual two-split design assumes
# and is flagged as such in the report rather than hidden.
#
# On inner-train (2020-2021, where 2020 and especially 2021's funding spike
# do trip the gate), V1 (hard flatten, top decile q=0.90, EXPANDING causal
# quantile, 3-day trailing level) beats plain v4 on futures_5x on ALL THREE
# axes at once: final balance $4,239 vs $4,050, max drawdown 18.3% vs 24.4%,
# funding paid $526 vs $894. It is also the best point in its own 6-point
# neighbourhood on return and drawdown - which is the OPPOSITE of a plateau
# on inner-train (q=0.85 scores $3,224, a 24% drop) and is reported as such:
# this is a peak on the very split it was picked on, not a validated
# plateau, because inner-validation could not supply an independent check.
FROZEN = dict(gate_quantile=0.90, gate_mode="flatten", window_mode="expanding",
             level_settlements=9, min_settlements=270)
DECISION_RULE = (
    "Promote V1 (frozen params above) over plain kelly_regime_v4 IFF, on the "
    "holdout, futures_5x final balance with real funding charged to BOTH "
    "beats buy_and_hold AND V1 beats plain v4 by more than the +/-0.2 Sharpe "
    "noise floor OR by a materially smaller max drawdown, AND V1 survives "
    "the falsification test (0.40% taker, see falsify()). Anything else is "
    "NOT PROMOTED / NEGATIVE, per ROUTINE.md's default-reject bar."
)
# Reported too, per "Running directions in parallel": a branch that stays
# silent about its non-winning configurations is selection by the operator.
ALSO = [(tag, kw) for tag, kw in _variants() if not tag.startswith("V1")]


def holdout() -> None:
    """Step 4. Frozen config, evaluated once, both markets, with funding."""
    print(f"HOLDOUT {HOLDOUT[0]} -> {HOLDOUT[1]}  "
          f"(real funding data ends {FUND_END}; this IS the whole holdout, "
          f"not a truncated view of a longer one)\n")
    for market, mname in ((FUTURES, "futures 5x"), (SPOT, "spot")):
        print(f"\n{mname}, WITHOUT funding charged (upper-bound view):")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            ev(get_strategy(name), *HOLDOUT, market=market, tag=f"  {name}", count=False)
        ev(FundingGatedV4(funding=REAL_FUNDING, **FROZEN), *HOLDOUT, market=market,
           tag="  V1 funding_gated_v4 (FROZEN)", count=False)

        print(f"\n{mname}, WITH real funding charged (the number that matters):")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            ev(get_strategy(name), *HOLDOUT, market=market, tag=f"  {name}",
               count=False, funding=REAL_FUNDING)
        ev(FundingGatedV4(funding=REAL_FUNDING, **FROZEN), *HOLDOUT, market=market,
           tag="  V1 funding_gated_v4 (FROZEN)", count=False, funding=REAL_FUNDING)
        for tag, kw in ALSO:
            ev(FundingGatedV4(funding=REAL_FUNDING, **kw), *HOLDOUT, market=market,
               tag=f"  {tag} (also-ran)", count=False, funding=REAL_FUNDING)


def falsify() -> None:
    """Pre-registered falsification test: does V1 survive Bitstamp's 0.40% taker?

    Kill condition, written down before running: if, on the holdout,
    futures_5x, with real funding charged to both, V1's final balance is
    WORSE than plain kelly_regime_v4's at the SAME 0.40% fee, the gate's
    extra flatten/re-entry turnover costs more than the funding it saves,
    and the mechanism is falsified.
    """
    print(f"HOLDOUT {HOLDOUT[0]} -> {HOLDOUT[1]}, futures_5x, real funding "
          "charged, 0.10% vs Bitstamp's 0.40% taker tier:\n")
    results = {}
    for fee, label in ((0.0005, "0.05% (table default, futures)"),
                       (0.0040, "0.40% (Bitstamp entry taker)")):
        venue = MarketSpec.futures(leverage=5.0, fee_rate=fee)
        print(f"  {label}")
        v4, _ = ev(get_strategy("kelly_regime_v4"), *HOLDOUT, market=venue,
                  tag="    kelly_regime_v4", count=False, funding=REAL_FUNDING)
        v1, _ = ev(FundingGatedV4(funding=REAL_FUNDING, **FROZEN), *HOLDOUT,
                  market=venue, tag="    V1 funding_gated_v4", count=False,
                  funding=REAL_FUNDING)
        results[fee] = (v4.final_balance, v1.final_balance)

    lo_v4, lo_v1 = results[0.0005]
    hi_v4, hi_v1 = results[0.0040]
    print(f"\n  V1 - v4 at 0.05%: {lo_v1 - lo_v4:+,.0f}")
    print(f"  V1 - v4 at 0.40%: {hi_v1 - hi_v4:+,.0f}")
    survives = hi_v1 >= hi_v4
    print(f"\n  FALSIFICATION TEST: {'PASS (survives 0.40%)' if survives else 'FAIL (falsified at 0.40%)'}")


COMMANDS = {"sweep": sweep, "neighbours": neighbours, "causality": causality,
           "holdout": holdout, "falsify": falsify}


def main() -> None:
    print(f"spot: {len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    print(f"funding: {len(REAL_FUNDING):,} settlements  {FUND_START} -> {FUND_END}",
          file=sys.stderr)
    print(f"holdout window used by this file: {HOLDOUT[0]} -> {HOLDOUT[1]}",
          file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python experiments/b05_funding_gate.py [{'|'.join(COMMANDS)}]")


if __name__ == "__main__":
    main()
