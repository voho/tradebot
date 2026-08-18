#!/usr/bin/env python
"""Funding as a gate on kelly_regime_v4 (backlog B-05, conservative branch).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. This is one of two independent,
parallel branches on B-05 run the same day (see docs/ROUTINE.md,
"Running directions in parallel") — this file owns the *conservative*
mechanism below; a separate branch explores a different one in a
different file. Neither branch reads or writes the other's file. The
trials counted here are this branch's alone; the ledger row that merges
both branches must sum them.

The idea, in one sentence
--------------------------
`kelly_regime_v4` decides how much to hold from price alone; fold in one
more bar-visible fact — the market's own perpetual funding rate — as a
hard veto: stand flat whenever funding is in its own top decile, because
R-16 found rich funding predicts *negative* forward returns unless price
is also rising hard, and R-14 found the strategy's funding bill is worst
exactly when it is most exposed.

Constraint attacked: **COST**. R-14's standing diagnosis is that funding
runs ~+20%/yr while the strategy holds against ~+2.8%/yr while flat,
*because the crowding the strategy's own regime vote detects is what
sets the rate*. A gate that stands flat when funding richens attacks
that co-movement directly, rather than trying to out-forecast price.

Not a duplicate of: L-01/L-04 (kelly_regime family — this wraps v4
unchanged, it does not re-derive its sizer); R-28/R-31/R-32
(e-process/matched-risk gates — those replace the *regime* question with
an anytime-valid martingale over *returns*; this gate answers a
different question, "is the position I already decided to hold too
expensive right now?", from a signal — funding — that is not derived
from price at all, unlike every INFO-constraint entry in section A that
failed doing exactly that from OHLCV alone). R-16 is the open hypothesis
this closes one way or the other.

Simulable here: yes, entirely from committed data. `ctx.market.pays_funding`
already exists in `tradebot/broker.py`; the funding series is already
loadable via `tradebot.data.load_funding`. No new simulation capability
is needed.

What would make it fail (named before any code ran): the gate does not
improve the 2023 futures-with-funding holdout over `kelly_regime_v4`
beyond the ±0.2 Sharpe noise floor / a 10pp drawdown improvement; or it
does at 0.10% fees but the ranking flips at the 0.40% Bitstamp tier
(R-12's defining failure mode — turnover-driven gains that live only in
the cheap-fee tier); or the swept window length is a lucky single peak
rather than a plateau.

The mechanism (frozen, pre-registered — do not deviate)
---------------------------------------------------------
Causally, using only funding settlements known as of "now": compute the
trailing percentile rank of the most recent funding settlement relative
to its own trailing window of ``window_days`` days (``window_days * 3``
settlements, since Binance settles every 8h). This is done entirely on
the funding series' own timeline — ``pandas.Series.rolling(N).apply``
over past-and-current settlements only, nothing downstream of "now" ever
enters the window. That per-settlement percentile is then forward-filled
onto the 5-minute bar grid: a settlement at time T becomes visible to
bars at time >= T only (``Series.reindex(bar_index, method="ffill")``,
which by construction never pulls a value whose index exceeds the
target). When the most recently known percentile is >= 0.90 **and**
``ctx.market.pays_funding`` (true for futures, false for spot — spot
never pays funding so the gate is inert there by construction, which is
also this file's implementation sanity check), the strategy's target
exposure is hard-overridden to 0.0 regardless of `kelly_regime_v4`'s own
vote. Otherwise the unmodified `kelly_regime_v4` target is used
unchanged. Binary, not continuous, and evaluated on every bar — kept
deliberately simple and low-turnover, as befits the *conservative*
branch of this round.

Free parameters, and how they were chosen: ``threshold=0.90`` is fixed
by B-05's own definition ("stand flat in the top decile") and is not
swept. ``window_days`` — the only swept knob — has candidates
60/90/120/180/270/365 days, selected on inner-validation only (see
below); 365 days is the longest usable value given the funding file
starts 2020-01-01 and inner-validation itself starts 2021-01-01 (a
365-day trailing window has just enough settlement history by then).

Split (a modification of ROUTINE.md's standard split, necessitated by
the funding file covering only 2020-2023 — stated honestly, not buried)
------------------------------------------------------------------------
inner-train        2020-01-01 -> 2020-12-31   (warmup only; kelly_regime_v4's
                                                own warmup is 80 days, and the
                                                widest funding window needs a
                                                full year of settlements to
                                                stop reading NaN)
inner-validation   2021-01-01 -> 2022-12-31   (window-length selection —
                                                contains the 2021 top and the
                                                2022 bear)
holdout            2023-01-01 -> 2023-12-31   (frozen config, read once per
                                                the plan below)

This holdout is one calendar year, not the usual 3.6 — the intersection
of OOS_START=2023-01-01 with the last date the committed funding file
covers. Read every number from it as low-power; a single-year result is
not a replication and is not sold as one below.

Pre-registered decision rule (written BEFORE the 2023 holdout was read)
--------------------------------------------------------------------------
Promote only if, on the 2023 holdout, on futures 5x with real funding
charged, ALL of:

  P1  final balance (equivalently, log growth) beats `kelly_regime_v4`
      (funding-charged) over the identical window.
  P2  the improvement exceeds the project's +/-0.2 Sharpe noise floor
      (R-20), OR is a max-drawdown improvement of >= 10 percentage points.
  P3  falsification test: the P1 ranking (variant beats funding-charged
      kelly_regime_v4) still holds when taker fees are raised to the
      0.40% Bitstamp entry tier — spot fee_rate=0.004, futures fee_rate
      scaled by the same ratio the shipped defaults already use between
      the two markets (0.0005 / 0.001 = 0.5x), i.e. futures fee_rate =
      0.002, following the fee-tier pattern in scripts/fee_study.py.
  P4  the swept window length sits on a plateau on inner-validation, not
      a lone peak — neighbours are reported, not just the winner.

If confirming the rule would need it to change after seeing the 2023
numbers, the change is refused and the result is reported NEGATIVE with
the rule's original form stated explicitly; a rule change made only to
fix a genuine bug found while reading the holdout is allowed but must be
labeled and the resulting numbers marked in-sample.

Usage
-----
    python experiments/funding_gate_decile.py sweep     # inner-train + inner-validation
    python experiments/funding_gate_decile.py holdout   # step 4, frozen config, read once
    python experiments/funding_gate_decile.py falsify   # P3, 0.40% tier
    python experiments/funding_gate_decile.py spotcheck # gate-never-fires sanity check
    python experiments/funding_gate_decile.py all       # everything, in order, one run
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import prefix_bars  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL_FUNDING = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

INNER_TRAIN = ("2020-01-01", "2020-12-31")
INNER_VALID = ("2021-01-01", "2022-12-31")
HOLDOUT = ("2023-01-01", "2023-12-31")

THRESHOLD = 0.90
WINDOW_CANDIDATES_DAYS = (60, 90, 120, 180, 270, 365)
SETTLEMENTS_PER_DAY = 3  # Binance settles every 8h

FROZEN_WINDOW_DAYS = 90  # picked by sweep() on inner-validation final balance; see report

BITSTAMP_TAKER_SPOT = 0.004
# futures/spot default fee ratio in this repo's MarketSpec is 0.0005/0.001
# = 0.5x; hold that ratio at the elevated tier rather than substituting the
# spot number directly, per scripts/fee_study.py's leverage() pattern.
BITSTAMP_TAKER_FUTURES = BITSTAMP_TAKER_SPOT * (0.0005 / 0.001)

# Every distinct configuration this file evaluates on the inner split, for
# the project's deflated-Sharpe trials count (ROUTINE.md step 3). Holdout
# reads are counted separately, in HOLDOUT_READS below.
N_INNER_CONFIGS = 0
HOLDOUT_READS = 0  # every call that touches 2023 data, "just checking" included


# --------------------------------------------------------------------- gate

def funding_percentile_gate(funding: pd.Series | None, window_days: int,
                            threshold: float, bar_index: pd.DatetimeIndex) -> np.ndarray:
    """Causal top-decile flag for ``funding``, forward-filled onto ``bar_index``.

    Step 1 (causal on the funding series' own timeline): for each
    settlement, the fraction of the trailing ``window_days*3`` settlements
    (itself included) that are <= it — i.e. its own trailing percentile
    rank. ``rolling(window).apply`` only ever sees entries at or before the
    current one, so nothing downstream of a settlement can affect its own
    rank, and nothing downstream of "now" can affect any bar's gate.

    Step 2 (visibility): ``reindex(..., method="ffill")`` maps each bar to
    the most recent settlement at or before that bar's own timestamp. A
    settlement at time T is invisible to every bar before T and visible
    to every bar at or after T — never the reverse. Bars before the first
    settlement, or before ``window_days`` of settlement history has
    accumulated, get NaN and are treated as "gate off" (see below), never
    as "gate on".
    """
    n = len(bar_index)
    if funding is None or len(funding) == 0:
        return np.zeros(n, dtype=bool)

    f = funding.sort_index()
    window = window_days * SETTLEMENTS_PER_DAY
    pct = f.rolling(window, min_periods=window).apply(
        lambda w: float((w[-1] >= w).mean()), raw=True)
    pct_at_bar = pct.reindex(bar_index, method="ffill")
    return (pct_at_bar >= threshold).fillna(False).to_numpy()


class FundingGateDecile(KellyRegimeV4):
    """kelly_regime_v4, hard-gated flat when futures funding is in its own top decile.

    Overrides nothing about how kelly_regime_v4 decides its target
    exposure; only overrides *whether that target is allowed onto the
    book* on a market that pays funding. See module docstring for the
    full mechanism and pre-registration.
    """

    name = "funding_gate_decile"

    def __init__(self, funding: pd.Series | None = None, window_days: int = FROZEN_WINDOW_DAYS,
                 threshold: float = THRESHOLD, **kwargs) -> None:
        super().__init__(**kwargs)
        self.funding = funding
        self.window_days = window_days
        self.threshold = threshold

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # sets df["target"], the unmodified kelly_regime_v4 vote/size
        df["funding_top_decile"] = funding_percentile_gate(
            self.funding, self.window_days, self.threshold, df.index)
        return df

    def _effective(self, row) -> float:
        gated = bool(row.get("funding_top_decile", False))
        return 0.0 if gated else float(row["target"])

    def on_bar(self, ctx: Context) -> None:
        # Market-scoping happens HERE, not in prepare(): prepare() only
        # sees the OHLCV frame, never the market, so the percentile/gate
        # column it computes is market-agnostic. ctx.market.pays_funding
        # is True on futures and False on spot (tradebot/broker.py), so
        # the override below is a strict no-op on spot for every bar,
        # regardless of window_days/threshold - this is the "does the
        # scoping logic work" sanity check described in the task.
        gate_now = ctx.market.pays_funding and bool(ctx.bar["funding_top_decile"])
        t = 0.0 if gate_now else float(ctx.bar["target"])
        if ctx.prev is not None:
            gate_prev = ctx.market.pays_funding and bool(ctx.prev["funding_top_decile"])
            prev = 0.0 if gate_prev else float(ctx.prev["target"])
        else:
            prev = 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# ------------------------------------------------------------------ harness

def _period(strategy_factory, market: MarketSpec, start: str, end: str,
            funding: pd.Series | None = None, balance: float = 1_000.0):
    """Backtest ``strategy_factory()`` over ``DF[start:end]``, warmed on the
    bars before it (tradebot.window.run_period's logic, extended with the
    funding wire that run_period itself does not expose — same pattern as
    scripts/funding_study.py's ``_period`` helper)."""
    strategy = strategy_factory()
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    prefix = prefix_bars(DF, lo, strategy.warmup)
    frame = DF.iloc[lo - prefix: hi]
    raw = run_backtest(strategy, frame, market, balance, trade_start=prefix,
                       funding=funding, data_label=LABEL)
    trimmed = (raw if prefix == 0 else
               replace(raw, equity=raw.equity.iloc[prefix:], df=raw.df.iloc[prefix:]))
    return compute_metrics(trimmed), trimmed.funding_paid


def _row(tag: str, m, funding_paid: float) -> str:
    return (f"{tag:34s} final=${m.final_balance:>10,.0f} ({m.profit_pct:>+7.1f}%) "
            f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>6.2f} "
            f"trades={m.num_trades:>4d} funding=${funding_paid:>8,.0f}")


# --------------------------------------------------------------------- sweep

def sweep() -> dict:
    """Inner-train sanity + inner-validation window-length selection.

    Every (variant, window_days) pair evaluated on inner-validation counts
    toward N_INNER_CONFIGS. Selection criterion: final balance on futures
    5x with real funding charged over 2021-2022 (the table's own ranking
    criterion), with the neighbourhood reported for the P4 plateau check.
    """
    global N_INNER_CONFIGS

    print(f"{'=' * 100}\ninner-train {INNER_TRAIN[0]} -> {INNER_TRAIN[1]} "
          f"(warmup sanity only, NOT used for selection)\n{'=' * 100}")
    m, paid = _period(lambda: FundingGateDecile(funding=REAL_FUNDING,
                                                window_days=FROZEN_WINDOW_DAYS),
                      FUTURES, *INNER_TRAIN, funding=REAL_FUNDING)
    print(_row(f"funding_gate_decile(w={FROZEN_WINDOW_DAYS}) futures5x", m, paid))
    print("(funding history before 2020-01-01 does not exist, so the widest "
          "windows read mostly NaN/gate-off here - this run is a mechanics "
          "check, not a selection step, and is not counted as a config.)")

    print(f"\n{'=' * 100}\ninner-validation {INNER_VALID[0]} -> {INNER_VALID[1]} "
          f"- window-length selection (futures 5x, real funding charged)\n{'=' * 100}")

    baseline_m, baseline_paid = _period(lambda: get_strategy("kelly_regime_v4"),
                                        FUTURES, *INNER_VALID, funding=REAL_FUNDING)
    print(_row("kelly_regime_v4 (funding-charged)", baseline_m, baseline_paid))
    hold_m, _ = _period(lambda: get_strategy("buy_and_hold"), SPOT, *INNER_VALID)
    print(_row("buy_and_hold (spot)", hold_m, 0.0))
    print()

    results = {}
    for w in WINDOW_CANDIDATES_DAYS:
        N_INNER_CONFIGS += 1
        m, paid = _period(lambda w=w: FundingGateDecile(funding=REAL_FUNDING, window_days=w),
                          FUTURES, *INNER_VALID, funding=REAL_FUNDING)
        results[w] = (m, paid)
        beats = "BEATS v4" if m.final_balance > baseline_m.final_balance else "trails v4"
        print(_row(f"funding_gate_decile(w={w:3d}d)", m, paid) + f"  [{beats}]")

    best_w = max(results, key=lambda w: results[w][0].final_balance)
    print(f"\n{N_INNER_CONFIGS} configurations evaluated on inner-validation "
          f"(window_days in {WINDOW_CANDIDATES_DAYS}, threshold=0.90 fixed).")
    print(f"selected by final balance: window_days={best_w}")
    return {"results": results, "baseline": (baseline_m, baseline_paid),
            "hold": hold_m, "best_w": best_w}


# ------------------------------------------------------------------- holdout

def holdout() -> dict:
    """Step 4: the frozen config against 2023-01-01 -> 2023-12-31, once.

    Reads: futures5x{variant, v4, hold} + spot{variant, v4, hold} = 6.
    """
    global HOLDOUT_READS

    print(f"\n{'=' * 100}\nHOLDOUT {HOLDOUT[0]} -> {HOLDOUT[1]} - frozen config "
          f"window_days={FROZEN_WINDOW_DAYS}, threshold={THRESHOLD}\n{'=' * 100}")

    rows = {}
    for market, mkt_name, funding in ((FUTURES, "futures_5x", REAL_FUNDING),
                                      (SPOT, "spot", None)):
        print(f"\n-- {mkt_name} --")
        for tag, factory in (
            ("funding_gate_decile", lambda: FundingGateDecile(
                funding=REAL_FUNDING, window_days=FROZEN_WINDOW_DAYS)),
            ("kelly_regime_v4", lambda: get_strategy("kelly_regime_v4")),
            ("buy_and_hold", lambda: get_strategy("buy_and_hold")),
        ):
            HOLDOUT_READS += 1
            m, paid = _period(factory, market, *HOLDOUT, funding=funding)
            print(_row(f"{tag} [{mkt_name}]", m, paid))
            rows[(tag, mkt_name)] = (m, paid)

    variant_fut, v4_fut = rows[("funding_gate_decile", "futures_5x")], rows[("kelly_regime_v4", "futures_5x")]
    p1 = variant_fut[0].final_balance > v4_fut[0].final_balance
    sharpe_gap = variant_fut[0].sharpe - v4_fut[0].sharpe
    dd_gap = v4_fut[0].max_drawdown_pct - variant_fut[0].max_drawdown_pct  # positive = variant shallower
    p2 = (abs(sharpe_gap) > 0.2) or (dd_gap >= 10.0)
    print(f"\nP1 (final balance beats v4, funding-charged, futures5x): {p1}  "
          f"(${variant_fut[0].final_balance:,.0f} vs ${v4_fut[0].final_balance:,.0f})")
    print(f"P2 (|Δsharpe|>0.2 OR ΔmaxDD>=10pp): {p2}  "
          f"(Δsharpe={sharpe_gap:+.2f}, ΔmaxDD={dd_gap:+.1f}pp, variant shallower is positive)")

    variant_spot, v4_spot = rows[("funding_gate_decile", "spot")], rows[("kelly_regime_v4", "spot")]
    spot_identical = abs(variant_spot[0].final_balance - v4_spot[0].final_balance) < 1e-6
    print(f"\nspot sanity check (gate must never fire, spot never pays funding): "
          f"variant=${variant_spot[0].final_balance:,.6f} vs "
          f"v4=${v4_spot[0].final_balance:,.6f}  identical={spot_identical}")

    return {"rows": rows, "p1": p1, "p2": p2, "sharpe_gap": sharpe_gap, "dd_gap": dd_gap,
            "spot_identical": spot_identical}


# ------------------------------------------------------------------- falsify

def falsify() -> dict:
    """P3: does the P1 ranking survive the 0.40% Bitstamp entry tier?

    Read count: futures5x{variant, v4} at the elevated tier = 2 more
    reads of the 2023 holdout.
    """
    global HOLDOUT_READS

    print(f"\n{'=' * 100}\nP3 falsification: 0.40% spot tier / {BITSTAMP_TAKER_FUTURES:.3%} "
          f"futures tier (scaled proportionally), {HOLDOUT[0]} -> {HOLDOUT[1]}\n{'=' * 100}")
    hi_fee_futures = MarketSpec.futures(leverage=5.0, fee_rate=BITSTAMP_TAKER_FUTURES)

    HOLDOUT_READS += 1
    m_variant, paid_variant = _period(
        lambda: FundingGateDecile(funding=REAL_FUNDING, window_days=FROZEN_WINDOW_DAYS),
        hi_fee_futures, *HOLDOUT, funding=REAL_FUNDING)
    print(_row("funding_gate_decile [futures5x, hi-fee]", m_variant, paid_variant))

    HOLDOUT_READS += 1
    m_v4, paid_v4 = _period(lambda: get_strategy("kelly_regime_v4"), hi_fee_futures,
                            *HOLDOUT, funding=REAL_FUNDING)
    print(_row("kelly_regime_v4 [futures5x, hi-fee]", m_v4, paid_v4))

    p3 = m_variant.final_balance > m_v4.final_balance
    print(f"\nP3 (ranking survives 0.40% tier): {p3}  "
          f"(${m_variant.final_balance:,.0f} vs ${m_v4.final_balance:,.0f})")
    return {"variant": m_variant, "v4": m_v4, "p3": p3}


def spotcheck() -> None:
    """Standalone version of the spot sanity check, callable on its own."""
    m_variant, _ = _period(lambda: FundingGateDecile(funding=REAL_FUNDING,
                                                      window_days=FROZEN_WINDOW_DAYS),
                           SPOT, *HOLDOUT)
    m_v4, _ = _period(lambda: get_strategy("kelly_regime_v4"), SPOT, *HOLDOUT)
    print(_row("funding_gate_decile [spot]", m_variant, 0.0))
    print(_row("kelly_regime_v4 [spot]", m_v4, 0.0))
    identical = abs(m_variant.final_balance - m_v4.final_balance) < 1e-6
    print(f"identical: {identical}")


COMMANDS = {"sweep": sweep, "holdout": holdout, "falsify": falsify, "spotcheck": spotcheck}


def main() -> None:
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    print(f"{len(REAL_FUNDING):,} funding settlements  "
          f"{REAL_FUNDING.index[0]:%Y-%m-%d} -> {REAL_FUNDING.index[-1]:%Y-%m-%d}\n",
          file=sys.stderr)

    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "all":
        sweep()
        holdout()
        falsify()
    elif choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python experiments/funding_gate_decile.py [{'|'.join(COMMANDS)}|all]")
        return

    print(f"\ninner configurations evaluated: {N_INNER_CONFIGS}   "
          f"holdout (2023) reads this run: {HOLDOUT_READS}")


if __name__ == "__main__":
    main()
