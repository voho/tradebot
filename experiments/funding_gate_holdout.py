"""B-05 holdout evaluation, pre-registered before the 2023 holdout was read.

This file is committed BEFORE its own output is read (the git log entry
for this commit is the pre-registration timestamp, per the convention
R-28/R-31/R-32 established: "the freezing commit is one commit ahead of
the results"). Do not edit the frozen config or the decision rule below
after seeing output from this script.

Context
-------
Two candidate variants for backlog item B-05 (funding as a gate on
kelly_regime_v4, COST constraint) were built and tuned in parallel, each
on its own file, per ROUTINE.md's parallelism rules:

- ``experiments/funding_gate_decile.py`` -- FundingDecileGate. Conservative:
  a single binary mechanism, force flat when trailing funding is in its
  own top decile. On inner-validation (2022) it beat kelly_regime_v4 on
  BOTH markets across 9 of 12 swept configs (a plateau, not a peak), and
  its improvement survived a 0.40% taker fee tier on that same window.
  Gate active ~10-14% of bars -- a targeted, occasional intervention.

- ``experiments/funding_gate_continuous.py`` -- FundingAwareKelly. Novel:
  a continuous funding-aware Kelly haircut extending kelly_regime_ev's
  analytic fee-deadband derivation to a continuously-accruing cost. Also
  beat kelly_regime_v4 on every one of its 9 swept inner-validation
  configs, and won 10/12 paired Monte Carlo windows drawn from
  2020-2022. But its own report flags a serious doubt, in its own words:
  mean exposure drops to 40-75% of kelly_regime_v4's, it de-levers on
  SPOT too (where there is no funding bill to avoid at all), and "the
  clean win on every validation cell is suspicious for the same reason
  R-28's win was -- it may just be de-levering helped in a year v4 lost
  money" (2022 was a losing year for v4: Sharpe -1.44). That is exactly
  the mechanical-delever artifact R-31/R-32 already caught once in this
  project (an e-process gate's "0-of-40-windows" drawdown win dissolved,
  and partly REVERSED, once measured at matched risk against the vote
  gate it was compared to).

Selection between the two candidates (made on training/inner-validation
evidence only, per ROUTINE.md step 3 -- "select on it as often as you
like"; this is NOT the pre-registered holdout step):

**FundingDecileGate is carried forward to the holdout. FundingAwareKelly
is not.** Reasons, weighed before looking at 2023: (1) it is a much
smaller, more targeted intervention (~10-14% of bars) rather than a
persistent ~25-60% de-lever of the whole book, so it is mechanically
less likely to be "just a de-lever that wins because 2022 was a loss" --
the project's own prior finding says a persistent exposure cut is
exactly the shape that manufactures a false risk finding; an occasional,
threshold-triggered one is a smaller target for that critique, though
not immune to it (see the mechanism check below, which is run on the
holdout regardless of outcome). (2) It only fires on the funding
percentile signal itself, never on spot where there is no cost basis to
avoid, whereas the continuous variant's own report flags firing on spot
too as a sign it may be trading R-16's noisy return-predictive signal
rather than "cost avoidance" specifically. (3) It is simpler and closer
to B-05's literal framing ("stand flat in the top decile"). (4) Its own
falsification test (0.40% fee) was run on the SAME window used for
config selection and passed; the continuous variant's falsification
(paired MC windows) is stronger evidence in isolation, but its own
report concludes "not something I'd promote on this evidence alone,"
while the decile gate's does not carry an equivalent self-flagged doubt
of that severity.

This is a documented judgement call, not a coin flip -- and it is
recorded here, before the holdout, specifically so it cannot be quietly
revised if the holdout had gone the other way. FundingAwareKelly's
inner-validation numbers stand as reported by its own session and are
recorded in the ledger as a parallel branch, per "every branch reports,
including the dead ones" -- it simply does not spend a holdout
consultation, because only one candidate is being promoted through this
round.

Frozen configuration
---------------------
``FundingDecileGate(funding_window_days=30, flatten_threshold=0.85)`` --
the top-ranked config on inner-validation that beat kelly_regime_v4 on
BOTH markets, per funding_gate_decile.py's own ``select_config`` rule,
sitting in a neighbourhood (w60/w90/w180 at t=0.85) that also beats v4,
i.e. a plateau along the window-length axis at the selected threshold.

Holdout window
---------------
Funding data is committed 2020-01-01..2023-12-31 only. The project's
real OOS_START is 2023-01-01 and its holdout otherwise runs to the
present (2026-08); nothing here can read funding past 2023-12-31, so the
holdout for THIS strategy specifically is **2023-01-01..2023-12-31 only
-- a single year**, far short of the project's usual 3.6-year holdout.
That is a real limitation of this finding's statistical power, stated
here before any number is read, not discovered afterward.

Pre-registered decision rule (promote only if ALL of P1-P4 hold; default
is REJECT)
-----------------------------------------------------------------------
- **P1 (return).** On 2023-01-01..2023-12-31, spot final balance beats
  ``buy_and_hold`` (the project's binding benchmark).
- **P2 (materiality).** The improvement over ``kelly_regime_v4``
  (funding-charged, same window, futures 5x -- the market where the
  mechanism actually operates) exceeds the +/-0.2 Sharpe noise floor
  (R-20), OR is a drawdown improvement of >= 10 percentage points.
- **P3 (falsification, chosen now).** The improvement over
  ``kelly_regime_v4`` on futures survives Bitstamp's 0.40% taker fee
  tier, re-run on THIS holdout window (not just on the inner-validation
  window where it was already checked once) -- confirming the fee
  result was not an artifact of the specific training window.
- **P4 (plateau, not peak).** The immediate neighbours already
  characterized as a plateau on inner-validation (w60/w90/w180 at
  t=0.85) are re-run on the holdout and checked for the same ordering
  (not required to all individually beat v4, since one year is a small
  sample -- but no wild reversal that would suggest a peak rather than a
  region).

**Recorded regardless of P1-P4, as a mechanism check, not a promotion
criterion (out of scope for a full matched-risk redo -- that is backlog
item B-13, not this round):** kelly_regime_v4's and the candidate's mean
exposure on the holdout window, side by side. If the entire improvement
traces to a lower mean exposure with no funding-specific timing content,
that is flagged explicitly and the result is downgraded in the written
lesson even if P1-P4 formally pass, per this project's standing rule
that a comparison against a fully-invested benchmark is not the same
claim as a comparison against a risk-matched one (R-31, R-32).

**Stated prediction, before looking:** P1 and P3 likely hold (the
0.40% fee result was robust across both inner splits and the gate is a
COST-avoidance mechanism as much as a return one). P2 is the most
uncertain -- one year is a small sample and 2023 is not obviously a year
funding runs unusually rich, unlike the 2020-2021 configs it was tuned
on. The mechanism check will likely show SOME of the gap is attributable
to lower mean exposure, since regime-gated sizing being flat some of the
time always has that side effect (L-04's own robust finding) -- the
question is whether it is ALL of the gap or only part of it.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

from funding_gate_decile import FundingDecileGate  # noqa: E402

DATA_DIR = ROOT / "data"
DF, LABEL = load_dataset(DATA_DIR, "spot")
REAL_FUNDING = load_funding(DATA_DIR)
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

HOLDOUT = ("2023-01-01", "2023-12-31")
NOISE_FLOOR_SHARPE = 0.2
WINDOW, THRESHOLD = 30, 0.85
NEIGHBOURS = ((60, 0.85), (90, 0.85), (180, 0.85))


def _period(make_strategy, market: MarketSpec, start: str, end: str, funding=None):
    strategy = make_strategy()
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw.funding_paid, trimmed


def make_gate(window=WINDOW, threshold=THRESHOLD):
    return lambda: FundingDecileGate(funding=REAL_FUNDING, funding_window_days=window,
                                     flatten_threshold=threshold)


def mean_exposure(trimmed) -> float:
    """Mean |notional/equity| over bars where the strategy is in the market."""
    df = trimmed.df
    if "target" not in df.columns:
        return float("nan")
    t = df["target"].to_numpy(dtype=float)
    active = np.abs(t) > 1e-9
    return float(np.abs(t[active]).mean()) if active.any() else 0.0


def p1_p2() -> dict:
    print("=" * 78)
    print("P1 (return) / P2 (materiality) -- holdout 2023-01-01..2023-12-31")
    print("=" * 78)
    out = {}
    for market_name, market, funding in (("spot", SPOT, None),
                                         ("futures_5x", FUTURES, REAL_FUNDING)):
        gate_m, gate_fp, gate_raw = _period(make_gate(), market, *HOLDOUT, funding=funding)
        v4_m, v4_fp, v4_raw = _period(lambda: get_strategy("kelly_regime_v4"), market,
                                      *HOLDOUT, funding=funding)
        hold_m, hold_fp, _ = _period(lambda: get_strategy("buy_and_hold"), market,
                                     *HOLDOUT, funding=funding)
        out[market_name] = dict(gate=gate_m, v4=v4_m, hold=hold_m,
                                gate_raw=gate_raw, v4_raw=v4_raw,
                                gate_funding=gate_fp, v4_funding=v4_fp)
        print(f"\n-- {market_name} --")
        for tag, m, fp in (("FundingDecileGate", gate_m, gate_fp),
                           ("kelly_regime_v4", v4_m, v4_fp),
                           ("buy_and_hold", hold_m, hold_fp)):
            print(f"  {tag:20s} final=${m.final_balance:>10,.0f} "
                  f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>6.2f} "
                  f"trades={m.num_trades:>4d} fees=${m.fees_paid:>7,.1f} "
                  f"funding=${fp:>8,.1f}")
    return out


def p3_falsification() -> None:
    print("\n" + "=" * 78)
    print("P3 (falsification, pre-registered): 0.40% taker fee, holdout window")
    print("=" * 78)
    BITSTAMP_TAKER = 0.004
    spot_fee = MarketSpec.spot(fee_rate=BITSTAMP_TAKER)
    fut_fee = MarketSpec.futures(leverage=5.0, fee_rate=BITSTAMP_TAKER)
    for market_name, market, funding in (("spot", spot_fee, None),
                                         ("futures_5x", fut_fee, REAL_FUNDING)):
        gate_m, _, _ = _period(make_gate(), market, *HOLDOUT, funding=funding)
        v4_m, _, _ = _period(lambda: get_strategy("kelly_regime_v4"), market,
                             *HOLDOUT, funding=funding)
        hold_m, _, _ = _period(lambda: get_strategy("buy_and_hold"), market,
                               *HOLDOUT, funding=funding)
        print(f"  {market_name:10s} gate=${gate_m.final_balance:>9,.0f} "
              f"v4=${v4_m.final_balance:>9,.0f} hold=${hold_m.final_balance:>9,.0f}  "
              f"gate_beats_v4={gate_m.final_balance > v4_m.final_balance}")


def p4_plateau() -> None:
    print("\n" + "=" * 78)
    print("P4 (plateau, not peak): neighbours on the holdout, futures 5x")
    print("=" * 78)
    v4_m, _, _ = _period(lambda: get_strategy("kelly_regime_v4"), FUTURES, *HOLDOUT,
                         funding=REAL_FUNDING)
    print(f"  kelly_regime_v4          final=${v4_m.final_balance:>9,.0f} "
          f"DD={v4_m.max_drawdown_pct:>5.1f}%")
    for w, th in ((WINDOW, THRESHOLD),) + NEIGHBOURS:
        m, _, _ = _period(make_gate(w, th), FUTURES, *HOLDOUT, funding=REAL_FUNDING)
        print(f"  w{w}_t{th:.2f}                final=${m.final_balance:>9,.0f} "
              f"DD={m.max_drawdown_pct:>5.1f}%  beats_v4={m.final_balance > v4_m.final_balance}")


def mechanism_check(results: dict) -> None:
    print("\n" + "=" * 78)
    print("Mechanism check (not a promotion criterion): mean exposure, holdout futures")
    print("=" * 78)
    gate_exp = mean_exposure(results["futures_5x"]["gate_raw"])
    v4_exp = mean_exposure(results["futures_5x"]["v4_raw"])
    print(f"  FundingDecileGate mean |exposure| while active: {gate_exp:.3f}")
    print(f"  kelly_regime_v4   mean |exposure| while active: {v4_exp:.3f}")
    print(f"  ratio: {gate_exp / v4_exp:.2f}x" if v4_exp else "  ratio: n/a")
    gate = results["futures_5x"]["gate"]
    v4 = results["futures_5x"]["v4"]
    print(f"  time in market -- gate vs v4 not directly comparable via Metrics; "
          f"see num_trades above as a proxy for how often the gate actually fired.")


def main() -> None:
    print(f"data: {LABEL}, {DF.index[0]} .. {DF.index[-1]}")
    print(f"funding: {REAL_FUNDING.index[0]} .. {REAL_FUNDING.index[-1]}")
    print(f"HOLDOUT (frozen, pre-registered): {HOLDOUT[0]} .. {HOLDOUT[1]}\n")

    results = p1_p2()
    p3_falsification()
    p4_plateau()
    mechanism_check(results)

    print("\n" + "=" * 78)
    print("DECISION")
    print("=" * 78)
    spot = results["spot"]
    fut = results["futures_5x"]
    p1 = spot["gate"].final_balance > spot["hold"].final_balance
    dsharpe = fut["gate"].sharpe - fut["v4"].sharpe
    ddd = fut["v4"].max_drawdown_pct - fut["gate"].max_drawdown_pct
    p2 = abs(dsharpe) > NOISE_FLOOR_SHARPE or ddd >= 10.0
    print(f"P1 (spot beats buy_and_hold): {p1}  "
          f"(${spot['gate'].final_balance:,.0f} vs ${spot['hold'].final_balance:,.0f})")
    print(f"P2 (materiality vs v4, futures): dSharpe={dsharpe:+.2f} dDD={ddd:+.1f}pp "
          f"-> {p2}")


if __name__ == "__main__":
    main()
