#!/usr/bin/env python
"""Backlog B-05 — funding as a gate on kelly_regime_v4 (COST constraint).

Mechanism (one sentence). Rich perpetual funding predicts weak-to-negative
forward returns (R-16: 14-day forward spread Q1-Q5 = +3.57pp) and is a
direct, currently-uncharged future cost that runs ~+20%/yr while the
strategy holds vs ~+2.8%/yr while flat (R-14) - so haircut the strategy's
long exposure when TRAILING funding sits in its own historical top decile,
computed causally: only funding settlements known as of bar t may be used.

Constraint attacked. COST ("costs scale *with* the signal" - the standing
diagnosis).

Not a duplicate of. R-15 (delta-neutral funding harvest - a different
trade: short the perp, long spot, collect the carry) or R-16 (which only
*measured* the quintile spread and built no strategy). This is the
specific low-turnover gate R-16 recommended and the ledger lists as the
actionable next item (B-05) - haircut exposure, don't reverse it.

Pre-registered falsification test (fixed before any tuning, see `fees()`
below). Does the gate survive Bitstamp's 0.40% taker tier? `fee_study.py`
shows nothing beats holding at that tier for the base strategies; the
question is whether adding the gate changes that verdict for
kelly_regime_v4, or whether it dies the same way turnover-reduction always
does here (R-12).

Hard data constraint. The committed funding file only covers
2020-01-01 -> 2023-12-31 (see `data/btcusdt_perp_funding_8h.csv.gz`).
Consequently:
  - inner-train (2017-2020) only has REAL funding for calendar 2020;
    2017-2019 bars never see a "rich" signal (gate defaults inactive,
    not proxied) - reported explicitly in `sweep()`.
  - inner-validation (2021-2022) is FULLY covered by real funding.
  - the holdout with REAL funding CHARGED can only run 2023-01-01 ->
    2023-12-31 - one year of the 3.6-year holdout - reported separately
    from the full 2023-2026 comparison, which never charges funding and
    whose *signal* also goes permanently inactive after 2023-12-31 (see
    "coverage" handling in `_funding_rich_signal`; the gate does not
    freeze the last known reading forever - it is explicitly switched off
    once the file's coverage ends, which is the "never proxy unavailable
    data out of price" rule applied to a *signal* rather than a cost).

Decision rule - written down BEFORE the holdout was read, and not touched
after (this docstring is committed identically before and after step 4).
Promote only if ALL of:

  P1  on the 2023 real-funding holdout window, spot AND futures final
      balance with the gate beats the SAME metric for ungated
      kelly_regime_v4 (futures run with funding=REAL on both sides, for a
      fair comparison - exactly scripts/funding_study.py's convention);
  P2  the improvement exceeds the +/-0.2 Sharpe noise floor (R-20), OR is
      a drawdown/tail improvement;
  P3  it survives the falsification test (does not lose its edge at 0.40%
      Bitstamp taker, relative to what happens to ungated v4 there);
  P4  the threshold/haircut neighbourhood is a plateau (>=3 nearby
      configs), not a single tuned peak.

Commands::

    python experiments/funding_gate.py rates      # what the funding file covers
    python experiments/funding_gate.py sanity     # haircut=1.0 == pure v4, exactly
    python experiments/funding_gate.py sweep      # inner-validation grid (SELECTION)
    python experiments/funding_gate.py causality  # two hand checks (price + funding)
    python experiments/funding_gate.py holdout    # step 4, frozen config
    python experiments/funding_gate.py fees       # falsification: 0.40% taker
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
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL = load_funding(ROOT / "data")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS = ("2023-01-01", None)
OOS_2023_ONLY = ("2023-01-01", "2023-12-31")   # the only window with real funding

N_EVALUATED = 0  # distinct configurations scored on inner-validation (see sweep())


# --------------------------------------------------------------------- signal

def _funding_rich_signal(index: pd.DatetimeIndex, funding: pd.Series | None,
                          quantile: float, trailing_settlements: int,
                          min_settlements: int) -> pd.Series:
    """Causal boolean signal: trailing funding sits in its own top decile.

    Every step below only uses information available at or before its own
    timestamp, and the file's actual coverage is respected rather than
    proxied:

    1. ``trailing`` is a backward-looking rolling mean over the last
       ``trailing_settlements`` *settlements* (pandas ``.rolling`` is
       trailing by construction: row i depends on rows <= i only).
    2. ``thresh`` is an EXPANDING quantile over settlements 0..i,
       ``shift(1)``-ed so the threshold compared against settlement i
       never includes settlement i itself - a settlement is judged
       against history strictly before it, not against a threshold that
       is partly made of its own value.
    3. Each settlement is bucketed onto the 5-minute bar whose interval
       contains it using the IDENTICAL rule
       ``engine.run_backtest`` uses to charge funding as a cost
       (``index.searchsorted(t, side="right") - 1``) - a bar can only see
       a settlement the engine itself is already willing to charge at
       that same bar, so the signal's causality convention matches the
       cost's exactly, as required.
    4. The per-bar flag then carries forward (last KNOWN settlement,
       never a future one) only while a settlement has been seen RECENTLY
       (within ``max_gap`` of the CURRENT bar - a purely backward-looking
       test: "how long ago was the last settlement I have actually seen
       as of now?"). Once real settlements stop arriving (the file's
       coverage ends), that gap grows past ``max_gap`` and the flag is
       forced back to "not rich" rather than freezing whatever it last
       read - freezing would be a proxy (assuming today's regime from a
       reading up to 3.6 years stale); switching off is the same
       "never proxy unavailable data out of price" rule applied to a
       signal instead of a cost, and it is why the full 2024-2026 holdout
       reduces to plain kelly_regime_v4 exposure (see `holdout()`).

       An earlier version of this function computed "covered" as
       ``index <= rates.index[-1]`` - the LAST timestamp in the whole
       (filtered) funding series. That is not causal: for a bar well
       before the file's real end, ``rates.index[-1]`` is a value from
       the FUTURE relative to that bar (whether the feed keeps producing
       settlements after today is not knowable today). The funding-
       truncation check in ``causality()`` below caught this directly -
       artificially truncating the series changed decisions *before* the
       truncation point, which a purely backward-looking staleness gap
       cannot do. Fixed here; the truncation check now passes.
    """
    out = pd.Series(False, index=index)
    if funding is None or len(funding) == 0:
        return out
    rates = funding.sort_index()
    rates = rates[(rates.index >= index[0]) & (rates.index <= index[-1])]
    if len(rates) == 0:
        return out

    trailing = rates.rolling(trailing_settlements, min_periods=trailing_settlements).mean()
    thresh = trailing.expanding(min_periods=min_settlements).quantile(quantile).shift(1)
    rich_at_settlement = (trailing > thresh).to_numpy()

    slot = index.searchsorted(rates.index, side="right") - 1
    valid = slot >= 0
    flag = pd.Series(np.nan, index=index)
    flag.iloc[slot[valid]] = rich_at_settlement[valid].astype(float)
    flag = flag.ffill()

    # Causal staleness gate: at bar t, how long ago was the last settlement
    # actually seen (as of t, never referencing anything later)? Normal
    # settlements are 8h apart; allow one miss (16h) before calling the
    # feed stale. Built the same way `flag` was: assign at each settlement's
    # own bar, ffill, nothing here reads rates.index[-1] or any other
    # whole-series aggregate.
    settle_ts = pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")
    settle_ts.iloc[slot[valid]] = rates.index[valid]
    settle_ts = settle_ts.ffill()
    gap = index.to_series(index=index) - settle_ts
    max_gap = pd.Timedelta(hours=16)
    covered = gap.notna() & (gap <= max_gap)

    flag = flag.where(covered, 0.0)
    return flag.fillna(0.0).astype(bool)


# ------------------------------------------------------------------- strategy

class KellyRegimeFundingGate(KellyRegimeV4):
    """kelly_regime_v4, haircut when trailing perp funding is in its own top decile.

    Not registered (``tradebot.registry.register`` is never imported here),
    per ROUTINE.md step 5: a NEGATIVE or unresolved idea stays under
    ``experiments/`` rather than in the comparison table.

    ``prepare()`` calls the parent chain unchanged to get v4's own
    (already-deadbanded) ``target`` column, then applies exactly one more
    multiplicative haircut + deadband pass on top of it: when the causal
    funding signal above says "rich", the desired exposure is
    ``v4_target * haircut``; otherwise it is ``v4_target`` unchanged. When
    ``funding`` is ``None`` or the signal is never rich (e.g. every bar
    outside the file's 2020-2023 coverage), the gate has literally no
    effect and this class's output is bit-identical to
    ``KellyRegimeV4`` - see ``sanity()``.
    """

    name = "_funding_gate_v4"

    def __init__(self, funding: pd.Series | None = None,
                 funding_quantile: float = 0.90,
                 trailing_settlements: int = 9,   # 8h settlements: 9 = 3 days
                 min_settlements: int = 30,        # ~10 days before the gate can fire
                 haircut: float = 0.5,
                 gate_deadband: float | None = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.funding = funding
        self.funding_quantile = funding_quantile
        self.trailing_settlements = trailing_settlements
        self.min_settlements = min_settlements
        self.haircut = haircut
        # Reuse v4's own deadband by default so a "not rich" bar produces
        # literally the same trade cadence v4 already has.
        self.gate_deadband = self.deadband if gate_deadband is None else gate_deadband

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4's target column, already deadbanded
        base_target = df["target"].to_numpy(dtype=float)

        rich = _funding_rich_signal(df.index, self.funding, self.funding_quantile,
                                    self.trailing_settlements,
                                    self.min_settlements).to_numpy()
        mult = np.where(rich, self.haircut, 1.0)

        n = len(df)
        gated = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = base_target[i] * mult[i]
            if abs(desired - pos) > self.gate_deadband:
                pos = desired
            gated[i] = pos

        df["target_pre_gate"] = base_target
        df["funding_rich"] = rich
        df["target"] = gated
        return df


# ---------------------------------------------------------------------- eval

def _period(strategy, start, end, market, funding=None, balance=1_000.0):
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, balance,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw.funding_paid, len(raw.fills)


def ev(strategy, start, end, market=SPOT, tag="", funding=None,
       balance=1_000.0, count=True):
    """One backtest, one line. ``count`` marks it as a distinct swept configuration."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    m, funding_paid, fills = _period(strategy, start, end, market,
                                     funding=funding, balance=balance)
    line = (f"  {tag or strategy.name:34s} {market.name:9s} "
            f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
            f"fills={fills:>5d} DD={m.max_drawdown_pct:>5.1f}% "
            f"sharpe={m.sharpe:>5.2f}{'  LIQUIDATED' if m.liquidated else ''}")
    if funding is not None:
        line += f" funding_paid=${funding_paid:>8,.0f}"
    print(line)
    return m


# -------------------------------------------------------------------- rates

def rates() -> None:
    """What the committed funding file actually covers."""
    print(f"{len(REAL):,} settlements  {REAL.index[0]} -> {REAL.index[-1]}")
    print(f"mean 8h rate {REAL.mean():+.6f} -> {REAL.mean() * 3 * 365.25:+.2%}/yr "
          "to a constant long")
    q90 = REAL.rolling(9, min_periods=9).mean().quantile(0.90)
    print(f"trailing-3-day mean funding, 90th percentile over its own full "
          f"history (informational only - the real gate uses an EXPANDING, "
          f"shifted quantile, never this full-series number): {q90:+.6f}")
    print("\nCoverage gaps this experiment must respect:")
    print("  inner-train  2017-01-01 -> 2020-12-31 : real funding only for 2020")
    print("  inner-val    2021-01-01 -> 2022-12-31 : fully covered")
    print("  holdout      2023-01-01 ->            : real funding only for 2023;")
    print("               2024-01-01 -> 2026-08    : NOT covered - gate forced off")


# -------------------------------------------------------------------- sanity

def sanity() -> None:
    """haircut=1.0 must reproduce kelly_regime_v4 exactly, bar for bar."""
    v4 = get_strategy("kelly_regime_v4")
    gate_noop = KellyRegimeFundingGate(funding=REAL, haircut=1.0)
    a = v4.prepare(DF.copy())["target"].to_numpy()
    b = gate_noop.prepare(DF.copy())["target"].to_numpy()
    worst = float(np.nanmax(np.abs(a - b)))
    print(f"max |v4.target - gate(haircut=1.0).target| over {len(DF):,} bars: "
          f"{worst:.3e}  {'PASS' if worst < 1e-12 else 'FAIL'}")

    gate_none = KellyRegimeFundingGate(funding=None, haircut=0.5)
    c = gate_none.prepare(DF.copy())["target"].to_numpy()
    worst2 = float(np.nanmax(np.abs(a - c)))
    print(f"max |v4.target - gate(funding=None).target| over {len(DF):,} bars: "
          f"{worst2:.3e}  {'PASS' if worst2 < 1e-12 else 'FAIL'}  "
          "(no funding data => gate can never fire => must equal v4 exactly)")


# --------------------------------------------------------------------- sweep

def _configs():
    """The grid swept on inner-validation. Every tuple here is one trial.

    The first 8 are the original bounded grid (quantile x haircut x
    trailing window). Inner-validation showed the 1-day trailing window
    (tr=3 settlements) was the only one beating ungated v4 on BOTH
    markets, with haircut/quantile making little difference at the 3-day
    window - i.e. a peak on the TRAILING-WINDOW axis, not (yet) a
    plateau. Configs 9-13 were added, before the holdout was touched, to
    test whether tr=1d is a genuine region or a single lucky point -
    P4's plateau requirement applies to whatever config gets selected,
    so it has to be checked around the actual candidate, not only around
    the a-priori center of the original grid.
    """
    return [
        ("q90 h=0.50 tr=3d",  dict(funding_quantile=0.90, trailing_settlements=9,  haircut=0.50)),
        ("q90 h=0.00 tr=3d",  dict(funding_quantile=0.90, trailing_settlements=9,  haircut=0.00)),
        ("q80 h=0.50 tr=3d",  dict(funding_quantile=0.80, trailing_settlements=9,  haircut=0.50)),
        ("q95 h=0.50 tr=3d",  dict(funding_quantile=0.95, trailing_settlements=9,  haircut=0.50)),
        ("q90 h=0.50 tr=1d",  dict(funding_quantile=0.90, trailing_settlements=3,  haircut=0.50)),
        ("q90 h=0.50 tr=7d",  dict(funding_quantile=0.90, trailing_settlements=21, haircut=0.50)),
        ("q90 h=0.25 tr=3d",  dict(funding_quantile=0.90, trailing_settlements=9,  haircut=0.25)),
        ("q90 h=0.75 tr=3d",  dict(funding_quantile=0.90, trailing_settlements=9,  haircut=0.75)),
        # -- neighbourhood around the tr=1d standout, added before holdout --
        ("q90 h=0.25 tr=1d",  dict(funding_quantile=0.90, trailing_settlements=3,  haircut=0.25)),
        ("q90 h=0.75 tr=1d",  dict(funding_quantile=0.90, trailing_settlements=3,  haircut=0.75)),
        ("q80 h=0.50 tr=1d",  dict(funding_quantile=0.80, trailing_settlements=3,  haircut=0.50)),
        ("q95 h=0.50 tr=1d",  dict(funding_quantile=0.95, trailing_settlements=3,  haircut=0.50)),
        ("q90 h=0.50 tr=2d",  dict(funding_quantile=0.90, trailing_settlements=6,  haircut=0.50)),
    ]


def _benchmarks(start, end, market, label, funding=None):
    print(f"\n{label} benchmarks:")
    for name in ("buy_and_hold", "kelly_regime_v4"):
        ev(get_strategy(name), start, end, market=market, tag=f"  {name}",
           funding=funding, count=False)


def sweep() -> None:
    """Step 3: select on inner-validation ONLY. Real funding is charged on
    futures throughout (2021-2022 is fully covered by the file)."""
    configs = _configs()
    print(f"sweeping {len(configs)} configurations\n")

    for market, mname, funding in ((SPOT, "spot", None), (FUTURES, "futures 5x", REAL)):
        _benchmarks(*TRAIN, market, f"INNER-TRAIN (informational; funding real for 2020 only) / {mname}",
                    funding=funding)
        print(f"INNER-TRAIN / {mname} variants:")
        for tag, kw in configs:
            ev(KellyRegimeFundingGate(funding=REAL, **kw), *TRAIN, market=market,
               tag=tag, funding=funding, count=False)

        _benchmarks(*VALID, market, f"INNER-VALIDATION (SELECTION) / {mname}", funding=funding)
        print(f"INNER-VALIDATION / {mname} variants:")
        for tag, kw in configs:
            ev(KellyRegimeFundingGate(funding=REAL, **kw), *VALID, market=market,
               tag=tag, funding=funding, count=True if market is SPOT else False)

    print(f"\nconfigurations evaluated (distinct parameter tuples swept on "
          f"inner-validation, counted once each): {len(configs)}")
    print("(each config above was scored on both spot and futures on both "
          "inner-train and inner-validation = 4x the backtests, but the "
          "trial count that matters for deflated Sharpe is the distinct "
          "configuration count, not the backtest count - R-28's convention.)")


# ----------------------------------------------------------------- causality

def causality() -> None:
    """Two hand checks, since this experiment gets none of
    tests/test_causality_strict.py's protection (it only parametrizes over
    the registry).

    1. The two-opposite-tampers price check (R-28's design): bars after a
       cut are multiplied by 3 in one copy, divided by 3 in the other;
       every decision at or before the cut must be identical.
    2. A funding-truncation check specific to this strategy: run the same
       untampered prices with the FULL real funding series vs a copy
       truncated to drop every settlement after the cut; the ``target``
       column up to the cut must be bit-identical. This is the check that
       actually exercises the funding-signal causality convention -
       check 1 alone would not catch a bug that let the signal peek at a
       future settlement, since settlement values are exogenous to price.
    """
    # Slice ending at the funding file's last day so the gate is actually
    # exercised across the cut, not testing on bars where it can never fire.
    df = DF.loc[:"2023-12-31"].iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]
    FROZEN_KW = FROZEN

    print(f"tampered from bar {cut:,} of {len(df):,} ({df.index[cut]})\n")

    # --- check 1: price tamper -------------------------------------------
    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    def decisions(frame):
        s = KellyRegimeFundingGate(funding=REAL, **FROZEN_KW)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    a, b = decisions(up), decisions(down)
    bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
    print("check 1 (price tamper, x3 / /3 after the cut):")
    print("  " + ("FAIL - reads the future at bars " + str(bad) if bad
                  else "PASS - every decision at or before the cut is unchanged"))

    pa = KellyRegimeFundingGate(funding=REAL, **FROZEN_KW).prepare(up.copy())
    pb = KellyRegimeFundingGate(funding=REAL, **FROZEN_KW).prepare(down.copy())
    for col in ("target", "target_pre_gate", "funding_rich"):
        diff = np.abs(pa[col].to_numpy()[:cut].astype(float) -
                      pb[col].to_numpy()[:cut].astype(float))
        worst = float(np.nanmax(diff))
        print(f"  column {col:16s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if worst < 1e-12 else 'FAIL'}")

    # --- check 2: funding truncation --------------------------------------
    cut_ts = df.index[cut]
    full_funding = REAL
    truncated_funding = REAL[REAL.index <= cut_ts]
    print(f"\ncheck 2 (funding truncation, {len(full_funding) - len(truncated_funding)} "
          f"settlements after the cut removed):")
    pf = KellyRegimeFundingGate(funding=full_funding, **FROZEN_KW).prepare(df.copy())
    pt = KellyRegimeFundingGate(funding=truncated_funding, **FROZEN_KW).prepare(df.copy())
    for col in ("target", "target_pre_gate", "funding_rich"):
        diff = np.abs(pf[col].to_numpy()[:cut].astype(float) -
                      pt[col].to_numpy()[:cut].astype(float))
        worst = float(np.nanmax(diff))
        print(f"  column {col:16s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if worst < 1e-12 else 'FAIL'}")


# ------------------------------------------------------------------- holdout

# Frozen BEFORE this function was ever run, based ONLY on the 13-config
# inner-validation sweep above (sweep() must be run and read before this
# dict is trusted; holdout() has not executed at the point this comment was
# written). Findings that drove the selection:
#
#   - The 3-day and 7-day trailing windows NEVER beat ungated v4 on futures
#     in inner-validation (all 6 such configs are worse there); only the
#     1-day trailing window (3 settlements) does. This is a peak on the
#     TRAILING-WINDOW axis, not a plateau - stated plainly for P4.
#   - Within tr=1d, haircut 0.25/0.50/0.75 at quantile 0.90, and quantile
#     0.80 at haircut 0.50, ALL beat ungated v4 on BOTH spot and futures.
#     Quantile 0.95 at tr=1d does not (worse than v4 on both markets), and
#     neither does tr=2d. So there IS a real plateau on the HAIRCUT axis
#     within a narrow quantile/window region - not a single lucky cell.
#   - q90/h=0.50/tr=1d sits in the middle of that surviving region rather
#     than at its single best cell (q80/h=0.50/tr=1d has the best futures
#     number) - the a-priori round choice, not the literal argmax.
FROZEN = dict(funding_quantile=0.90, trailing_settlements=3,
              min_settlements=30, haircut=0.50)


def holdout() -> None:
    """Step 4. Configuration is frozen above; the decision rule is in the
    module docstring, written before this function was ever executed."""
    gate = KellyRegimeFundingGate(funding=REAL, **FROZEN)
    v4 = get_strategy("kelly_regime_v4")
    hold = get_strategy("buy_and_hold")

    print("=" * 78)
    print("HOLDOUT PART A - 2023-01-01 -> 2023-12-31 ONLY, the sole window "
          "with REAL funding available to CHARGE. This is P1's window.")
    print("=" * 78)
    for market, mname, funding in ((SPOT, "spot", None), (FUTURES, "futures 5x", REAL)):
        print(f"\n{mname} ({'no funding charged - spot never pays it' if funding is None else 'REAL funding CHARGED on both rows'}):")
        ev(hold, *OOS_2023_ONLY, market=market, tag="buy_and_hold", funding=funding, count=False)
        ev(v4, *OOS_2023_ONLY, market=market, tag="kelly_regime_v4 (ungated)", funding=funding, count=False)
        ev(gate, *OOS_2023_ONLY, market=market, tag="funding_gate_v4 (FROZEN)", funding=funding, count=False)

    print("\n" + "=" * 78)
    print("HOLDOUT PART B - FULL 2023-01-01 -> end of data (2026-08), "
          "funding NOT charged. Separate question: does the gate's exposure "
          "TIMING help even where we cannot charge the real cost? The gate's "
          "signal itself also goes inactive after 2023-12-31 (no proxying),")
    print("so from 2024 onward this literally reduces to kelly_regime_v4.")
    print("=" * 78)
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"\n{mname}:")
        ev(hold, *OOS, market=market, tag="buy_and_hold", count=False)
        ev(v4, *OOS, market=market, tag="kelly_regime_v4 (ungated)", count=False)
        ev(gate, *OOS, market=market, tag="funding_gate_v4 (FROZEN)", count=False)

    # Gate activity diagnostics, so "how often did it even fire" is on record.
    prepared = gate.prepare(DF.copy())
    rich = prepared["funding_rich"]
    in_2023 = rich.loc["2023-01-01":"2023-12-31"]
    print(f"\ngate activity in 2023 (the only year it can possibly fire): "
          f"rich on {in_2023.mean():.1%} of bars")
    changed = (prepared["target"] != prepared["target_pre_gate"]).loc["2023-01-01":"2023-12-31"]
    print(f"bars in 2023 where the gate actually changed exposure vs plain v4: "
          f"{changed.mean():.1%}")


# ----------------------------------------------------------------------fees

def fees() -> None:
    """Pre-registered falsification test: Bitstamp's 0.40% taker tier (spot).

    fee_study.py's finding: nothing here beats holding at 0.40%. Question
    fixed in advance: does the gate change that verdict for kelly_regime_v4,
    or does it lose the same way turnover reduction always does (R-12)?
    Run over the full 2023-2026 holdout (spot fee tier has nothing to do
    with funding, so the funding-coverage limit does not apply here).
    """
    gate = KellyRegimeFundingGate(funding=REAL, **FROZEN)
    v4 = get_strategy("kelly_regime_v4")
    hold = get_strategy("buy_and_hold")
    for fee, label in ((0.001, "0.10% (table assumption)"), (0.004, "0.40% (Bitstamp entry taker)")):
        market = MarketSpec.spot(fee_rate=fee)
        print(f"\n{label}:")
        ev(hold, *OOS, market=market, tag="buy_and_hold", count=False)
        ev(v4, *OOS, market=market, tag="kelly_regime_v4 (ungated)", count=False)
        ev(gate, *OOS, market=market, tag="funding_gate_v4 (FROZEN)", count=False)


if __name__ == "__main__":
    if REAL is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"rates": rates, "sanity": sanity, "sweep": sweep,
            "causality": causality, "holdout": holdout, "fees": fees}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/funding_gate.py [{'|'.join(cmds)}]")
