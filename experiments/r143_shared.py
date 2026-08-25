"""Shared, read-only utilities and FROZEN pre-registration for the R-143
round (08-25): the backward holdout -- BTC 2014-2016, history that no
strategy or study in this project's 142 prior rounds has ever used.

Idea in one sentence
---------------------
``scripts/build_bitstamp_dataset.py`` defaults its own ``--start`` to
2017-01-01 "so the span... keep[s] the file a reasonable size for git" --
a git-hygiene reason, not a research reason. Bitstamp's public
``/api/v2/ohlc/`` endpoint serves the identical venue, identical 5m
schema, back to 2013 (verified live this round: 97.3% non-empty-bar
coverage in 2013, 99.4% in 2014, 96.8% in 2015, 97.6% in 2016; 2012 is
7.4% and excluded as unusable). Every one of this project's 142 rounds --
every "does it replicate" check, every regime-timing gate, every N~3
significance procedure -- has held the *era* fixed at 2017-> and varied
only the asset or the statistic. This round varies the era instead: does
``kelly_regime_v4``, run FROZEN with zero refit, still show its one
surviving property (the matched-exposure drawdown reduction; R-20/R-33)
on three years of BTC history it was never fitted, tuned, or selected
against?

Citations
---------
- Linnainmaa & Roberts (2018), "The History of the Cross-Section of Stock
  Returns," Review of Financial Studies 31(7), 2606-2649 -- hand-collect
  PRE-SAMPLE data and re-test anomalies discovered after the original
  sample began; most vanish. The cautionary reading of this round.
- Baltussen, Swinkels & van Vliet (2021), "Global Factor Premiums,"
  Journal of Financial Economics 142(3), 1128-1154 -- the same move over
  217 years and 24 premiums; the majority SURVIVE. The optimistic reading.
  Cited together deliberately: this is a genuinely two-sided test, not one
  pre-registered to confirm.
- Arnott, Harvey & Markowitz (2019), "A Backtesting Protocol in the Era of
  Machine Learning," Journal of Financial Data Science 1(1), 64-74 --
  true out-of-sample means data never touched during development, which a
  backward extension satisfies in a way no forward holdout consulted
  ~700 times (see docs/LEDGER.md's holdout-consultation list) still can.
- Gandal, Hamrick, Moore & Oberman (2018), "Price manipulation in the
  Bitcoin ecosystem," Journal of Monetary Economics 95, 86-96 -- the
  Markus/Willy Mt. Gox trading bots inflated volume and price through
  late 2013; this is why 2013 is reported only as a disclosed sensitivity
  window, never the primary claim, and 2012 is excluded outright.

Constraint attacked: N~3, directly, for the first time. Four prior N~3
rounds (R-101 jackknife, R-104 bootstrap/PSR, R-138 permutation, R-140
Synthetic Control Method) are all PROCEDURES applied to the same fixed
~3-6 dated episodes -- no procedure can manufacture sample size. This is
the first round that adds events instead of re-analyzing the ones already
on hand.

Not a duplicate of
-------------------
- R-57 (six-instrument cross-asset panel, matched-exposure drawdown
  INVERTS 6/6): asset-space external validity. This is time-space,
  never tested before.
- R-17/R-47 (ETH replication): a second asset, same era (2017-).
- R-127/R-137 (ETH-idiosyncratic-event excision): this round avoids ETH
  entirely, side-stepping that confound rather than re-litigating it.
- R-45/R-118/R-119 (parameter reselection / robustness on the SAME data):
  this round refits nothing -- v4 runs with its shipped, already-frozen
  parameters, exactly as registered.
- R-138/R-140 (better statistics on the SAME ~6 events): this round does
  not re-analyze the six-episode set at all; it is a disjoint calendar.
- Section C ("Ruled out") has no row about data extent, and a full-ledger
  grep (``grep -oE "201[2-6]-[0-9]{2}-[0-9]{2}"``) finds zero dated
  windows before 2017 anywhere in 142 rounds.

Data provenance and quality (verified this round, before any strategy ran)
----------------------------------------------------------------------------
``scripts/fetch_bitstamp_early.py`` hits Bitstamp's public OHLC endpoint
directly (the GitHub mirror ``build_bitstamp_dataset.py`` normally clones
is proxy-blocked in this session; the API is not) and writes
``data/btcusd_spot_5m_pre2017.csv.gz`` in the canonical schema
(``timestamp,open,high,low,close,volume``, ms epoch), 2013-01-01 up to
(not including) 2017-01-01, the exact first timestamp of the committed
``btcusd_spot_5m.csv.gz`` (1483228800000 ms = 2017-01-01T00:00:00Z) -- so
the two files concatenate into one continuous series with no gap and no
overlap. :func:`load_extended_btc_spot` below does exactly that and
asserts monotonicity/uniqueness/no-duplicate-timestamp on the join.

Guardrails frozen before any backtest ran (this file, this commit --
before either branch has been dispatched or has seen a single result
number from it)
-----------------------------------------------------------------------
1. **PRIMARY window is 2014-01-01 -> 2016-12-31.** 2012 is excluded
   entirely (7.4% coverage, unusable). 2013 is reported ONLY as a
   disclosed sensitivity window (Mt. Gox-era manipulation, Gandal et al.
   2018) -- if the primary result and the 2013-inclusive sensitivity
   disagree, the primary window's verdict stands and the disagreement is
   reported, not averaged away or used to pick the more favorable number.
2. **Spot is the PRIMARY market.** No BTC perpetual futures existed
   before 2017 (Deribit's own BTC-PERPETUAL chart data starts 2018-08-14,
   R-15/B-15), so ``futures_5x`` on this window is a leverage-only
   overlay on the identical spot price path with no real funding history
   -- it may be reported as a secondary/illustrative number but funding
   cannot be charged on it and it must not be used as the primary claim.
3. **Zero refit.** ``kelly_regime_v4`` runs with its shipped, already
   registered parameters (``horizons=(20, 40, 80)``, everything else
   inherited from v3/v1 defaults). No parameter in this round is chosen,
   swept, or tuned against 2014-2016 data at any point -- that is the
   entire point of a backward holdout, and doing so would silently turn
   it into another in-sample fit.
4. **Primary statistic is the matched-exposure DRAWDOWN property, not
   Sharpe.** Per R-20/R-33, drawdown reduction is the one property that
   has actually replicated in this project; Sharpe differences under
   ±0.2 are noise-floor. This window is ~1,096 days; at R-78's own
   measured 3.0%/day paired-return noise, no correctly-sized test run
   ONLY on this window can resolve a Sharpe difference smaller than that
   noise floor, so Sharpe here is reported as a descriptive number, never
   as a promotion criterion on its own.
5. **Exposure matching is done INSIDE each sub-window** (R-33's own
   preferred construction, not one exposure frozen on the whole period),
   using :class:`experiments.matched_hold.ConstantExposureHold` on the
   MEAN-NOTIONAL axis (the literal "it just holds less" reading -- needs
   no solver, unlike the equal-realized-vol axis which is unstable
   period-to-period per R-33's own finding).
6. **Time-in-market and realized volatility are reported for every arm,
   including controls** (R-131's lesson: an ablation/control that changes
   the exposure has not isolated the mechanism, it has replaced it).
7. **New episodes in the extended calendar (novel branch) must be
   independently, publicly documented market events dated by external
   reporting** (a news event, an exchange collapse, a regulatory
   announcement) -- never a date selected by inspecting where a detector
   or the price series itself has an extremum. Picking dates from the
   series a detection-lag gate then scores would make the gate's own
   pass/fail circular. This mirrors exactly how the existing six-episode
   calendar in ``experiments/r100_shared.py`` was built (news-dated
   onsets: 2018-01-17 post-top, 2018-12-15 capitulation, 2020-03-12
   COVID, 2021-11-10 top, 2022-05-09 Terra/Luna, 2022-11-08 FTX).

Pre-registered falsification test (frozen now, before any run)
----------------------------------------------------------------
CONSERVATIVE branch, decisive check: compute the matched-exposure max
drawdown gap (v4 minus ``ConstantExposureHold`` at v4's own mean
notional, matched inside each sub-window) in each of the PRIMARY_SUBWINDOWS
below, on BTC spot. **Kill condition:** if v4's drawdown is >= the matched
hold's drawdown (i.e. the gap is >= 0, no advantage) in >= 50% of the
sub-windows (i.e. >= 3 of 6), the matched-exposure drawdown-reduction
property is judged a 2017-2022 calibration artifact that does not survive
the earliest, cleanest out-of-sample era this project can construct --
the R-57 inversion arriving from the time axis instead of the asset axis.
Anything better than that kill condition is reported as-is, including a
partial/mixed result; this is not an evaluate-once gate, it is a
pre-registered threshold exactly like every other round's Step-A/Step-4
bar.

NOVEL branch, decisive check: extend ``r100_shared.STRESS_EPISODES`` with
independently-documented pre-2017 episodes (guardrail 7 above) and re-run
the standard Step-A six-episode detection-lag gate machinery
(``anchor_majority`` below, copied verbatim from r100_shared.py, is the
v4 baseline the closed detectors are compared against) on the extended
calendar. This is not a ninth regime-timing detector -- it re-uses two
ALREADY-CLOSED detectors (BOCPD, best prior score 2/6; CUSUM, R-139,
0/6-3/6 across its own 36-cell sweep) on new episodes the gate has never
seen, auditing whether the six-episode INSTRUMENT ITSELF -- which has
now closed eight detectors and twenty INFO signals -- generalizes past
the six dates it was built from, or whether its own composition
(R-85: "dominated by sudden shocks... exactly one slow build-up") is why
nothing has ever cleared it.

Guardrail for the novel branch's own falsification: if the extended gate
produces a materially different pass/fail verdict for v4's own anchor
vote (the incumbent baseline every detector is compared against) than the
original six-episode gate did, that is reported as a finding about the
GATE, not folded silently into a detector's score.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics, max_drawdown_pct  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.matched_hold import ConstantExposureHold, mean_notional  # noqa: E402
from experiments.r100_shared import (  # noqa: E402
    BARS_PER_DAY,
    STRESS_EPISODES,
    anchor_majority,
    episode_window,
)

DATA_DIR = ROOT / "data"
PRE2017_FILE = DATA_DIR / "btcusd_spot_5m_pre2017.csv.gz"
CANONICAL_FILE = DATA_DIR / "btcusd_spot_5m.csv.gz"

# Windows, per guardrail 1.
SENSITIVITY_START = "2013-01-01"   # disclosed, secondary; Mt. Gox-era manipulation risk
PRIMARY_START = "2014-01-01"
PRIMARY_END = "2016-12-31"
JOIN_TS = pd.Timestamp("2017-01-01", tz="UTC")  # exact first bar of the canonical file

SPOT = MarketSpec.spot()

# Six ~180-day sub-windows tiling the primary 3-year period, for the
# INSIDE-window exposure matching guardrail 5 requires. Chosen by simple
# calendar tiling, fixed before any backtest ran -- not selected to land on
# favorable sub-periods.
PRIMARY_SUBWINDOWS = [
    ("2014-01-01", "2014-06-30"),
    ("2014-07-01", "2014-12-31"),
    ("2015-01-01", "2015-06-30"),
    ("2015-07-01", "2015-12-31"),
    ("2016-01-01", "2016-06-30"),
    ("2016-07-01", "2016-12-31"),
]


def load_extended_btc_spot() -> pd.DataFrame:
    """Concatenate the pre-2017 extension with the canonical 2017-> file.

    Asserts a clean, gap-free, duplicate-free join at exactly ``JOIN_TS``.
    Raises if ``PRE2017_FILE`` has not been fetched yet
    (``python scripts/fetch_bitstamp_early.py``).
    """
    if not PRE2017_FILE.exists():
        raise FileNotFoundError(
            f"{PRE2017_FILE} not found -- run "
            "`python scripts/fetch_bitstamp_early.py --start 2013-01-01 --end 2017-01-01`"
        )
    early = load_ohlcv_csv(PRE2017_FILE)
    late = load_ohlcv_csv(CANONICAL_FILE)
    assert early.index.max() < JOIN_TS <= late.index.min(), (
        f"join is not clean: early ends {early.index.max()}, "
        f"late starts {late.index.min()}, expected join at {JOIN_TS}"
    )
    combined = pd.concat([early, late])
    assert combined.index.is_unique and combined.index.is_monotonic_increasing
    gaps = combined.index.to_series().diff().dropna()
    expected = pd.Timedelta(minutes=5)
    n_gaps = int((gaps > expected).sum())
    if n_gaps:
        print(f"note: {n_gaps} bar gaps > 5min in the combined series "
              f"(max {gaps.max()})", file=sys.stderr)
    return combined


def matched_drawdown_gap(df: pd.DataFrame, strategy_factory, start: str, end: str,
                          market: MarketSpec = SPOT) -> dict:
    """v4 (or any strategy) vs its own mean-notional-matched constant hold,
    on one sub-window, warmed on bars before ``start``.

    Returns dict with both arms' drawdown, time-in-market, realized vol
    (annualized) and the gap (v4_dd - hold_dd; negative means v4 draws
    down LESS, i.e. the property holds).
    """
    strat_result = run_period(strategy_factory(), df, start, end, market=market,
                              start_balance=1_000.0)
    strat_m = compute_metrics(strat_result)
    c = mean_notional(strat_result)
    hold_result = run_period(ConstantExposureHold(c=max(c, 1e-6)), df, start, end,
                             market=market, start_balance=1_000.0)
    hold_m = compute_metrics(hold_result)

    def _vol_and_tim(result):
        eq = result.equity.to_numpy(dtype=float)
        r = np.diff(np.log(eq[eq > 0]))
        vol = float(np.std(r) * np.sqrt(BARS_PER_DAY * 365.25)) if len(r) > 1 else float("nan")
        tim = float(np.mean(result.df["target"].abs() > 1e-9)) if "target" in result.df else float("nan")
        return vol, tim

    strat_vol, strat_tim = _vol_and_tim(strat_result)
    hold_vol, hold_tim = _vol_and_tim(hold_result)

    return {
        "start": start, "end": end,
        "strat_dd": strat_m.max_drawdown_pct, "hold_dd": hold_m.max_drawdown_pct,
        "gap": strat_m.max_drawdown_pct - hold_m.max_drawdown_pct,
        "strat_final": strat_m.final_balance, "hold_final": hold_m.final_balance,
        "strat_sharpe": strat_m.sharpe, "hold_sharpe": hold_m.sharpe,
        "mean_notional_c": c,
        "strat_vol": strat_vol, "hold_vol": hold_vol,
        "strat_tim": strat_tim, "hold_tim": hold_tim,
    }
