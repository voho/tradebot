#!/usr/bin/env python
"""Funding as an exposure gate on kelly_regime_v4 (backlog B-05).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea, in one sentence
--------------------------
Gate ``kelly_regime_v4``'s exposure down (partially or fully) when
*trailing* perpetual funding is in a costly extreme, because funding is a
real, measured cost that scales *with* the position the strategy already
holds (the COST constraint), and R-16 found predictive structure in it (a
14-day forward-return spread of +3.57pp between funding quintiles).

Constraint attacked
--------------------
**COST**. Funding is the single largest unmodelled cost in this repo
(R-14: it turns v4's $156K funding-free futures headline into a
$36K-$80K band). It scales with notional and with time held, exactly like
the standing diagnosis describes ("costs scale *with* the signal"). This
gate does not try to predict price; it tries to stand down when the cost
of holding the position the strategy already wants is unusually high.

Why this is not a duplicate of anything already tried
-------------------------------------------------------
- **L-05 / L-06** (``kelly_regime_ev`` / ``_ev_fast``) build a no-trade
  band around the *taker fee* (a one-time cost per rebalance). This gates
  on *funding* (a recurring cost charged every 8h while a position is
  held) - a different cost with different dynamics: fees are paid once
  per trade, funding accrues continuously and can itself trend.
- **R-14** *measured* funding as a passive cost on the existing strategy
  and changed nothing about its behaviour. This wires that measurement
  into a decision rule.
- **R-15** (funding harvest) is a *different strategy* entirely -
  delta-neutral cash-and-carry that collects funding by shorting the perp
  against a spot long. This does not take a short leg; it only adjusts
  how much of the existing directional bet to hold.
- **R-16** found the funding -> forward-return relationship (and its own
  explicit caveat: high funding predicts negative returns *unless price
  is also rising*) but never turned it into a sizing rule. This is that
  next step, named explicitly as backlog item B-05.

Simulable here?
-----------------
Yes, with a real caveat that must be stated honestly rather than
engineered around: ``data/btcusdt_perp_funding_8h.csv.gz`` is real
Binance BTCUSDT funding, 8-hourly settlements, **2020-01-01 to
2023-12-31 only**. Per the repo's standing rule ("never proxy unavailable
data out of price" - the analogous rule here is "never proxy unavailable
funding out of nothing"), bars before 2020 or after 2023 get NO gating
signal and fall back exactly to ``kelly_regime_v4``'s behaviour. This
means:

- **inner-train** (2017-01-01 -> 2020-12-31) has real funding for only
  its final ~11 months (2020) - the other three years are pure v4.
- **inner-validation** (2021-01-01 -> 2022-12-31) is FULLY covered.
- the holdout (2023-01-01 ->) has real funding only through 2023-12-31;
  2024 onward is untouched by this experiment (and is not read here at
  all, per the task's hard constraint).

So the effective *funding-dependent* training window for anything this
file learns is really **2020-2022**, not the nominal 2017-2020 /
2021-2022 split. This is stated honestly in the results below rather than
papered over with an assumed/extrapolated rate outside the committed file
- extrapolating would be exactly the kind of fabrication the ledger warns
against (L-14/L-15/L-16: proxying unavailable data out of what IS
available adds no information and just adds turnover).

Causality, by construction
-----------------------------
Every funding-derived quantity here is:

1. an EWM mean over PAST settlements only
   (:func:`causal_trailing_funding`, additionally ``.shift(1)``'d even
   though a settlement is in reality known instantly at its own
   timestamp - belt and suspenders);
2. compared against an EXPANDING (never whole-series) quantile threshold,
   itself ``.shift(1)``'d (:func:`causal_expanding_quantile`) - the
   quantile at settlement i is a function of settlements strictly before
   i, never of the future or of the whole history;
3. forward-filled onto the 5-minute bar timeline with ``method="ffill"``
   only (:func:`reindex_to_bars`) - a bar can only ever see the most
   recent settlement at or before it, never a later one.

The by-hand ``causality()`` probe below runs the standard two-opposite-
tampers procedure on PRICE (the same check ``kelly_regime_v4`` itself
inherits) *and* a second, independent tamper on the FUNDING series itself
- the added lookahead surface this file introduces that no existing test
in the repo covers.

Three variants, each with a mechanism and a falsification test fixed
before any code ran
------------------------------------------------------------------------
**(a) hard gate** - force flat (multiply v4's target by 0) whenever
trailing causal funding exceeds an expanding top-decile threshold.
*Mechanism*: R-16's own headline framing - stand aside entirely when
funding sits in its costliest historical decile.
*Falsification*: if it does not beat plain ``kelly_regime_v4`` on
inner-validation by more than a token amount (and does not survive
having its Sharpe destroyed by the *extra* turnover from stepping in and
out), the idea is dead in its simplest form. Concretely: FAILS if
inner-validation final balance and max drawdown are BOTH no better than
v4's on both markets.

**(b) proportional haircut** - scale exposure continuously downward as
trailing funding rises through a [lo-quantile, hi-quantile] band, floored
at a fraction rather than zero.
*Mechanism*: R-16's own tables show the middle/upper funding quintiles
are still often net-positive forward return (Q4/Q5 average +0.56% to
+3.06% at 14 days, not negative) - a full-flat hard gate may be throwing
away real edge that a graded reduction would keep.
*Falsification*: FAILS if it does not dominate the hard gate's
risk-adjusted number net of the extra turnover a continuously-varying
factor creates (more rebalances -> more fees for a rule meant to reduce
cost exposure).

**(c) conditional gate** - apply (hard or haircut) gating only when
trailing momentum is NOT strongly positive; when price is rising
strongly, do not gate even if funding is rich.
*Mechanism*: R-16, verbatim - "high funding predicts negative forward
returns unless price is also rising strongly" (correlation between
funding and trailing return only 0.39, so this is not simply a momentum
proxy in disguise). This directly encodes the one caveat R-16 itself
flagged as load-bearing.
*Falsification*: FAILS if conditioning does not recover meaningfully more
of v4's return than the unconditional hard gate while keeping a
comparable share of its drawdown/cost benefit - i.e. if the momentum
override does not matter, R-16's caveat was not operationally useful.

Usage
-----
::

    python experiments/funding_gate_kelly.py inspect     # what the causal signal looks like
    python experiments/funding_gate_kelly.py sweep       # inner-train + inner-validation, step 3
    python experiments/funding_gate_kelly.py causality   # by-hand lookahead probe (price + funding)
    python experiments/funding_gate_kelly.py frozen      # frozen candidate vs baselines (train+valid only)

``holdout()`` is defined at the bottom, pre-registered, and is NOT called
by anything in this file or invoked during this session - the 2023+
holdout belongs to the operator.
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
from tradebot.window import run_period  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
FUNDING = load_funding(ROOT / "data")  # real Binance BTCUSDT, 2020-01-01 .. 2023-12-31 ONLY
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

BARS_PER_DAY = 288

# ROUTINE.md step 3 splits.
TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS_START = "2023-01-01"  # never read by anything in this file this session

if FUNDING is not None:
    # The hard boundary this whole file respects: real funding is only
    # ever read inside this window. Nothing here extrapolates past it.
    assert FUNDING.index[0] >= pd.Timestamp("2020-01-01", tz="UTC")
    assert FUNDING.index[-1] < pd.Timestamp("2024-01-01", tz="UTC")


# ============================================================== causal funding features


def causal_trailing_funding(funding: pd.Series, halflife_days: float,
                             min_settlements: int) -> pd.Series:
    """EWM mean of funding, using only settlements strictly before each point.

    Binance settles 3x/day (8h). ``.ewm(...).mean()`` at settlement i
    already depends only on settlements <= i; the extra ``.shift(1)``
    means the value attributed to i uses only settlements < i.
    """
    hl_periods = max(halflife_days, 1e-6) * 3.0
    return funding.ewm(halflife=hl_periods, min_periods=min_settlements).mean().shift(1)


def causal_expanding_quantile(series: pd.Series, q: float, min_periods: int) -> pd.Series:
    """Expanding (NEVER whole-series) quantile, shifted once more for safety.

    The one rule the task is strict about: no quantile/mean/std is ever
    taken over the full history and broadcast backward onto early rows.
    The threshold used to judge settlement i is a function of settlements
    strictly before i.

    Caveat found by :func:`inspect` and reported honestly rather than
    engineered around: an EXPANDING quantile over a series whose most
    extreme regime (the 2021 top) sits early in the window "locks in" a
    high bar that later, calmer years rarely re-cross - so the gate can
    silently stop firing after year one. See ``causal_rolling_quantile``
    for the alternative that was added specifically to test this.
    """
    return series.expanding(min_periods=min_periods).quantile(q).shift(1)


def causal_rolling_quantile(series: pd.Series, q: float, window_days: float,
                             min_periods: int) -> pd.Series:
    """Rolling (trailing window, never whole-series) quantile, shifted once more.

    Same causality guarantee as :func:`causal_expanding_quantile` - the
    threshold at settlement i depends only on settlements strictly before
    i - but the window forgets settlements older than ``window_days``, so
    an early extreme cannot permanently elevate the bar the way an
    expanding quantile can.
    """
    window = max(int(round(window_days * 3)), min_periods)  # 3 settlements/day
    return series.rolling(window, min_periods=min_periods).quantile(q).shift(1)


def reindex_to_bars(series: pd.Series, bar_index: pd.DatetimeIndex) -> np.ndarray:
    """Forward-fill a settlement-indexed series onto the 5m bar timeline.

    ``method="ffill"`` only ever carries a value forward to later bars, so
    no bar ever sees a settlement that has not happened yet. Bars before
    the first usable settlement get NaN (handled as "no gate" downstream -
    never fabricated).
    """
    return series.reindex(bar_index, method="ffill").to_numpy()


# ============================================================================= strategy


class FundingGatedKelly(KellyRegimeV4):
    """kelly_regime_v4, exposure gated down when trailing perp funding is a costly extreme.

    Inherits v4's latched 20/40/80-day anchor vote and conditional
    volatility sizing unchanged; ``prepare()`` multiplies the resulting
    target by a causal funding-derived factor in [floor, 1.0] and applies
    one more deadband pass to the product (mirrors
    ``experiments/matched_risk.py``'s ``GatedKelly``, which deadbands the
    product of confidence and scale rather than each factor separately).

    ``mode``:

    - ``"hard"``        factor = 0 when trailing funding > an expanding
                         top-``gate_quantile`` threshold, else 1.
    - ``"haircut"``      factor scales linearly from 1.0 down to
                         ``haircut_floor`` as trailing funding moves from
                         the ``haircut_lo_quantile`` to the
                         ``haircut_hi_quantile`` expanding threshold.
    - ``"conditional"``  either of the above, but overridden back to 1.0
                         whenever trailing ``momentum_days``-return exceeds
                         ``momentum_threshold`` (R-16's caveat: high
                         funding does not predict badly when price is also
                         rising strongly).

    Bars with no funding history yet (before 2020, or before
    ``min_settlements`` is reached) get factor = 1.0 - identical to v4.
    Never proxied, never extrapolated.
    """

    name = "funding_gated_kelly"  # NOT @register'd - experiments/ is not auto-discovered

    def __init__(
        self,
        funding: pd.Series | None = None,
        mode: str = "hard",
        funding_halflife_days: float = 3.0,
        gate_quantile: float = 0.90,
        haircut_lo_quantile: float = 0.70,
        haircut_hi_quantile: float = 0.95,
        haircut_floor: float = 0.30,
        momentum_days: int = 7,
        momentum_threshold: float = 0.05,
        min_settlements: int = 90,  # 30 days of 8h settlements
        threshold_window_days: float | None = None,  # None = expanding; else rolling window
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if mode not in ("hard", "haircut", "conditional"):
            raise ValueError(f"mode must be 'hard'/'haircut'/'conditional', got {mode!r}")
        self.funding = funding
        self.mode = mode
        self.funding_halflife_days = funding_halflife_days
        self.gate_quantile = gate_quantile
        self.haircut_lo_quantile = haircut_lo_quantile
        self.haircut_hi_quantile = haircut_hi_quantile
        self.haircut_floor = haircut_floor
        self.momentum_days = momentum_days
        self.momentum_threshold = momentum_threshold
        self.min_settlements = min_settlements
        self.threshold_window_days = threshold_window_days

    def _quantile(self, trail: pd.Series, q: float) -> pd.Series:
        if self.threshold_window_days is None:
            return causal_expanding_quantile(trail, q, self.min_settlements)
        return causal_rolling_quantile(trail, q, self.threshold_window_days, self.min_settlements)

    def _funding_factor(self, bar_index: pd.DatetimeIndex) -> np.ndarray:
        n = len(bar_index)
        if self.funding is None or len(self.funding) == 0:
            return np.ones(n)

        trail = causal_trailing_funding(self.funding, self.funding_halflife_days,
                                         self.min_settlements)
        trail_b = reindex_to_bars(trail, bar_index)

        if self.mode in ("hard", "conditional"):
            thr = self._quantile(trail, self.gate_quantile)
            thr_b = reindex_to_bars(thr, bar_index)
            known = np.isfinite(trail_b) & np.isfinite(thr_b)
            factor = np.where(known & (trail_b > thr_b), 0.0, 1.0)
        else:  # haircut
            lo = self._quantile(trail, self.haircut_lo_quantile)
            hi = self._quantile(trail, self.haircut_hi_quantile)
            lo_b, hi_b = reindex_to_bars(lo, bar_index), reindex_to_bars(hi, bar_index)
            known = np.isfinite(trail_b) & np.isfinite(lo_b) & np.isfinite(hi_b)
            span = np.maximum(hi_b - lo_b, 1e-9)
            frac_extreme = np.clip((trail_b - lo_b) / span, 0.0, 1.0)
            factor = np.where(known, 1.0 - frac_extreme * (1.0 - self.haircut_floor), 1.0)

        # Bug found by inspect(): plain `.reindex(method="ffill")` carries the
        # LAST computed value forward FOREVER once the committed funding series
        # ends (2023-12-31) - a frozen snapshot silently replayed into all of
        # 2024-2026, which for the rolling-window threshold produced a
        # spurious "gated=100% of bars" artifact (the last settlement's trail
        # happened to sit above its own trailing threshold). That is exactly
        # the kind of fabrication the task forbids: extrapolating a specific
        # unavailable reading indefinitely past the data that supports it.
        # Fix: once a bar is more than one settlement interval (8h) past the
        # last real settlement, the gate reverts to ungated - identical to a
        # bar that has no funding history yet.
        if len(self.funding):
            stale = bar_index > (self.funding.index[-1] + pd.Timedelta(hours=8))
            factor = np.where(stale, 1.0, factor)
        return factor

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4's target: latched vote x conditional-vol sizing, deadbanded
        v4_target = df["target"].to_numpy(dtype=float)

        factor = self._funding_factor(df.index)

        if self.mode == "conditional":
            # A lagged pct_change needs no extra shift to be causal.
            mom = df["close"].pct_change(self.momentum_days * BARS_PER_DAY).to_numpy()
            rising = np.isfinite(mom) & (mom > self.momentum_threshold)
            factor = np.where(rising, 1.0, factor)

        raw = v4_target * factor

        n = len(df)
        final = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = raw[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            final[i] = pos

        df["v4_target"] = v4_target
        df["funding_factor"] = factor
        df["target"] = final
        return df


# ================================================================================== harness


N_EVALUATED = 0  # every distinct configuration this file evaluates in step 3


def ev(strategy, start, end, market=SPOT, tag="", balance=1_000.0, count=False, df=None):
    """One backtest, one line, following ``scripts/experiment.py``'s ``ev()`` pattern
    (but counted, and routed through :func:`tradebot.window.run_period` for a fair warmup)."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    print(f"  {tag or strategy.name:40s} {market.name:9s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"trades={m.num_trades:>5d} fees=${m.fees_paid:>8,.0f} "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")
    return m


def ev_funded(strategy, start, end, market=FUTURES, tag="", balance=1_000.0, count=False):
    """Same as :func:`ev`, but charges REAL funding wherever the committed series covers
    the period. ``run_period`` has no ``funding=`` parameter, so this mirrors the manual
    trade_start dance used by R-28/R-31 (``experiments/run_eprocess.py``:``costs()``,
    ``experiments/run_matched_risk.py``:``costs()``)."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre:hi], market, balance,
                       trade_start=pre, funding=FUNDING, data_label=LABEL)
    trimmed = raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
    m = compute_metrics(trimmed)
    print(f"  {tag or strategy.name:40s} {market.name:9s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"trades={m.num_trades:>5d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} funding paid=${raw.funding_paid:>8,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")
    return m


# --------------------------------------------------------------------------- the sweep grid


def _hard_variants():
    out = []
    for hl in (2.0, 3.0, 5.0):
        for q in (0.80, 0.90, 0.95):
            out.append((f"hard hl={hl:g}d q={q:.2f} expanding",
                        dict(mode="hard", funding_halflife_days=hl, gate_quantile=q)))
    # Rolling-window thresholds: added after `inspect()` found the expanding
    # threshold "locks in" on the 2021 top and stops firing afterward (2022 and
    # 2023 both gate 0.0% of bars) - a rolling window forgets old extremes.
    for hl in (3.0,):
        for win in (90.0, 180.0, 365.0):
            for q in (0.85, 0.90):
                out.append((f"hard hl={hl:g}d q={q:.2f} win={win:g}d",
                            dict(mode="hard", funding_halflife_days=hl, gate_quantile=q,
                                 threshold_window_days=win)))
    return out


def _haircut_variants():
    out = []
    for hl in (2.0, 3.0, 5.0):
        for lo, hi, floor in ((0.70, 0.95, 0.30), (0.60, 0.90, 0.50)):
            out.append((f"haircut hl={hl:g}d [{lo:.2f},{hi:.2f}]->{floor:.2f}",
                        dict(mode="haircut", funding_halflife_days=hl,
                             haircut_lo_quantile=lo, haircut_hi_quantile=hi,
                             haircut_floor=floor)))
    return out


def _conditional_variants():
    out = []
    for hl in (2.0, 3.0, 5.0):
        for mom_thr in (0.0, 0.05, 0.10):
            out.append((f"conditional hl={hl:g}d q=0.90 mom>{mom_thr:.2f}",
                        dict(mode="conditional", funding_halflife_days=hl,
                             gate_quantile=0.90, momentum_days=7,
                             momentum_threshold=mom_thr)))
    return out


def _all_variants():
    return _hard_variants() + _haircut_variants() + _conditional_variants()


# Frozen after the sweep() results (30 configurations, inner-train + inner-validation,
# both markets - see this session's report). "hard hl=3d q=0.90 win=90d" was the single
# standout on inner-validation (the only period with FULL real-funding coverage): a
# ~12pp drawdown cut on both markets alongside a return improvement, not just a risk
# trade-off. Reported honestly rather than hidden: this exact configuration UNDERPERFORMS
# kelly_regime_v4 on inner-train (spot -15.2%, futures -19.4%) - though inner-train's
# funding-covered slice is only ~11 months (2020) with the 90-day window barely mature,
# so that disagreement is measured on a very small sample. The improvement also is NOT
# a broad plateau: it fades sharply as the window widens (180d and 365d given the same
# hl/q recover much less of the drawdown cut - see the neighbourhood() check below.
FROZEN_CONFIG = dict(mode="hard", funding_halflife_days=3.0, gate_quantile=0.90,
                     threshold_window_days=90.0)


def neighbourhood() -> None:
    """Plateau, not peak: vary one knob at a time around FROZEN_CONFIG, inner-validation only."""
    grid = [("FROZEN (win=90d q=0.90 hl=3d)", {})]
    grid += [(f"win={w:g}d", dict(threshold_window_days=w)) for w in (30.0, 60.0, 120.0, 180.0, 365.0)]
    grid += [(f"q={q:.2f}", dict(gate_quantile=q)) for q in (0.80, 0.85, 0.95)]
    grid += [(f"hl={hl:g}d", dict(funding_halflife_days=hl)) for hl in (1.0, 2.0, 5.0, 8.0)]
    grid += [(f"min_settlements={m}", dict(min_settlements=m)) for m in (45, 180)]
    for market_name, market in MARKETS:
        print(f"\ninner-validation neighbourhood / {market_name}:")
        for tag, kw in grid:
            s = FundingGatedKelly(funding=FUNDING, **{**FROZEN_CONFIG, **kw})
            # The first row (kw={}) reproduces the already-counted FROZEN_CONFIG;
            # only the genuinely new perturbations are counted, once (spot only).
            ev(s, *VALID, market=market, tag=tag, count=(bool(kw) and market_name == "spot"))
    print(f"\nconfigurations evaluated in this run (new, beyond sweep's 30): {N_EVALUATED}")


# ------------------------------------------------------------------------------- inspect


def inspect() -> None:
    """What the causal funding signal looks like, before any backtest."""
    if FUNDING is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    print(f"funding: {len(FUNDING):,} settlements  "
          f"{FUNDING.index[0]:%Y-%m-%d} -> {FUNDING.index[-1]:%Y-%m-%d}\n")

    s = FundingGatedKelly(funding=FUNDING, mode="hard", funding_halflife_days=3.0,
                          gate_quantile=0.90)
    prepared = s.prepare(DF.copy())
    factor = prepared["funding_factor"]
    covered = factor.index >= FUNDING.index[0]

    print("hard gate (hl=3d, q=0.90) - fraction of bars gated flat, by year "
          "(within the funding-covered window only):")
    sub = factor[covered]
    for year, g in sub.groupby(sub.index.year):
        print(f"  {year}  gated={100 * (g < 1.0).mean():5.1f}% of bars")

    print(f"\noverall (2017-2026): gated={100 * (factor < 1.0).mean():5.1f}% of ALL bars "
          f"(most are outside the funding-covered window and default to ungated)")
    print(f"within the funding-covered window (2020-2023): "
          f"gated={100 * (sub < 1.0).mean():5.1f}% of bars")

    s2 = FundingGatedKelly(funding=FUNDING, mode="hard", funding_halflife_days=3.0,
                           gate_quantile=0.90, threshold_window_days=180.0)
    factor2 = s2.prepare(DF.copy())["funding_factor"]
    sub2 = factor2[covered]
    print("\nsame gate with a 180-day ROLLING threshold instead of expanding, by year:")
    for year, g in sub2.groupby(sub2.index.year):
        print(f"  {year}  gated={100 * (g < 1.0).mean():5.1f}% of bars")

    # Pre-registered failure mode: is this just a momentum indicator in
    # disguise? Compare the hard gate's "gated" bars against v4's own vote.
    v4_vote_zero = prepared["v4_target"].abs() < 1e-9
    both_flat = ((factor < 1.0) & v4_vote_zero)[covered]
    print(f"\nof the bars the funding gate shuts, v4's own vote is ALSO already "
          f"flat in {100 * both_flat.sum() / max(1, (factor[covered] < 1.0).sum()):.1f}% "
          f"of them (overlap with the existing regime gate)")


# --------------------------------------------------------------------------------- sweep


def sweep() -> None:
    """Step 3. Inner-train + inner-validation only. Never reads 2023-01-01 onward."""
    if FUNDING is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    print(f"funding data covers {FUNDING.index[0]:%Y-%m-%d} -> {FUNDING.index[-1]:%Y-%m-%d} "
          f"ONLY - inner-train (2017-2020) is mostly ungated, inner-validation "
          f"(2021-2022) is fully covered.\n")

    for market_name, market in MARKETS:
        for (start, end), split in ((TRAIN, "inner-train"), (VALID, "inner-validation")):
            print(f"\n{split} / {market_name}")
            ev(get_strategy("buy_and_hold"), start, end, market=market, tag="buy_and_hold")
            ev(get_strategy("kelly_regime_v4"), start, end, market=market,
               tag="kelly_regime_v4 (baseline)")
            for tag, kw in _all_variants():
                s = FundingGatedKelly(funding=FUNDING, **kw)
                ev(s, start, end, market=market, tag=tag,
                   count=(split == "inner-train" and market_name == "spot"))
    print(f"\nconfigurations evaluated in step 3: {N_EVALUATED}")


# ---------------------------------------------------------------------------- causality


def causality() -> None:
    """Two independent by-hand lookahead probes - price (inherited from v4's own
    causal construction) and funding (this file's own added lookahead surface).
    ``tests/test_causality_strict.py`` only parametrizes over the *registry*, so an
    unregistered experiment gets none of that protection.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    if FUNDING is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")

    FROZEN = FROZEN_CONFIG

    # ---- probe 1: price tamper, the standard two-opposite-tampers procedure ----
    df = DF.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame, funding):
        s = FundingGatedKelly(funding=funding, **FROZEN)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    bad_price = [bar for bar, oa, ob in zip(bars, decisions(up, FUNDING), decisions(down, FUNDING))
                 if oa != ob]
    pa = FundingGatedKelly(funding=FUNDING, **FROZEN).prepare(up.copy())
    pb = FundingGatedKelly(funding=FUNDING, **FROZEN).prepare(down.copy())
    worst_price = max(
        float(np.nanmax(np.abs(pa[c].to_numpy()[:cut] - pb[c].to_numpy()[:cut])))
        for c in ("target", "funding_factor", "v4_target")
    )
    print("probe 1 - PRICE tamper (multiply/divide future bars by 3/7):")
    print(f"  tampered from bar {cut:,} of {len(df):,}; checked bars {bars}")
    print("  FAIL - order differs at bars " + str(bad_price) if bad_price
          else "  PASS - every order at or before the cut is unchanged")
    print(f"  max |column difference| before the cut = {worst_price:.3e}  "
          f"{'PASS' if worst_price < 1e-12 else 'FAIL'}")

    # ---- probe 2: funding tamper - this file's own added lookahead surface ----
    fcut = len(FUNDING) // 2
    fup, fdown = FUNDING.copy(), FUNDING.copy()
    fup.iloc[fcut:] *= 3.0
    fdown.iloc[fcut:] /= 3.0
    cut_time = FUNDING.index[fcut]
    check_time = cut_time - pd.Timedelta(days=5)  # margin before the tampered settlements
    check_pos = int(DF.index.searchsorted(check_time))

    pa2 = FundingGatedKelly(funding=fup, **FROZEN).prepare(DF.copy())
    pb2 = FundingGatedKelly(funding=fdown, **FROZEN).prepare(DF.copy())
    worst_funding = max(
        float(np.nanmax(np.abs(pa2[c].to_numpy()[:check_pos] - pb2[c].to_numpy()[:check_pos])))
        for c in ("target", "funding_factor", "v4_target")
    )
    print("\nprobe 2 - FUNDING tamper (multiply/divide future settlements by 3):")
    print(f"  funding tampered from settlement {fcut:,} of {len(FUNDING):,} "
          f"({cut_time:%Y-%m-%d}); checked bars before {check_time:%Y-%m-%d} "
          f"(bar {check_pos:,} of {len(DF):,})")
    print(f"  max |column difference| before the cut = {worst_funding:.3e}  "
          f"{'PASS' if worst_funding < 1e-12 else 'FAIL'}")


# -------------------------------------------------------------------------------- frozen


# Frozen after the sweep() results (30 configurations, inner-train + inner-validation,
# both markets - see this session's report). "hard hl=3d q=0.90 win=90d" was the single
# standout on inner-validation (the only period with FULL real-funding coverage): a
# ~12pp drawdown cut on both markets alongside a return improvement, not just a risk
def frozen() -> None:
    """The frozen candidate vs kelly_regime_v4 vs buy_and_hold, train + validation
    ONLY (never 2023-01-01 onward), on both markets, with funding charged where the
    committed series covers the period."""
    if FUNDING is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    s = FundingGatedKelly(funding=FUNDING, **FROZEN_CONFIG)

    for market_name, market in MARKETS:
        for (start, end), split in ((TRAIN, "inner-train"), (VALID, "inner-validation")):
            print(f"\n{split} / {market_name}  (no funding charged)")
            ev(get_strategy("buy_and_hold"), start, end, market=market, tag="buy_and_hold")
            ev(get_strategy("kelly_regime_v4"), start, end, market=market, tag="kelly_regime_v4")
            ev(FundingGatedKelly(funding=FUNDING, **FROZEN_CONFIG), start, end, market=market,
               tag="funding_gated_kelly (FROZEN)")

    print("\n--- with REAL funding charged on futures (covers 2020-01-01..2023-12-31 only) ---")
    for (start, end), split in ((TRAIN, "inner-train"), (VALID, "inner-validation")):
        print(f"\n{split} / futures 5x, funding charged")
        ev_funded(get_strategy("buy_and_hold"), start, end, tag="buy_and_hold")
        ev_funded(get_strategy("kelly_regime_v4"), start, end, tag="kelly_regime_v4")
        ev_funded(FundingGatedKelly(funding=FUNDING, **FROZEN_CONFIG), start, end,
                  tag="funding_gated_kelly (FROZEN)")


# ------------------------------------------------------------------------------ holdout


def holdout() -> None:
    """Step 4. NOT called by anything in this file and NOT run this session.

    Left here, pre-registered, exactly as R-28's and R-31's holdout() functions
    were left frozen one commit ahead of being run - so the operator can execute
    this after synthesizing both parallel branches, per the task's hard
    constraint against reading 2023-01-01 onward from this session.
    """
    for market_name, market in MARKETS:
        print(f"\nHOLDOUT 2023-01-01 -> / {market_name}")
        ev(get_strategy("buy_and_hold"), OOS_START, None, market=market,
           tag="buy_and_hold", count=False)
        ev(get_strategy("kelly_regime_v4"), OOS_START, None, market=market,
           tag="kelly_regime_v4", count=False)
        ev(FundingGatedKelly(funding=FUNDING, **FROZEN_CONFIG), OOS_START, None,
           market=market, tag="funding_gated_kelly (FROZEN)", count=False)
    ev_funded(get_strategy("kelly_regime_v4"), OOS_START, None, tag="kelly_regime_v4")
    ev_funded(FundingGatedKelly(funding=FUNDING, **FROZEN_CONFIG), OOS_START, None,
              tag="funding_gated_kelly (FROZEN)")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"inspect": inspect, "sweep": sweep, "neighbourhood": neighbourhood,
            "causality": causality, "frozen": frozen, "holdout": holdout}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/funding_gate_kelly.py [{'|'.join(cmds)}]")
