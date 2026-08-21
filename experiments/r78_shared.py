"""R-78: shared, frozen infrastructure for the B-06 viability round.

Committed BEFORE either branch was written, per ROUTINE.md's parallelism
rules: both branches import this file, NEITHER EDITS IT, and it computes no
verdict of its own. Its only job is to (a) carry this round's
pre-registration verbatim, (b) build the paired daily-return series both
branches read, and (c) provide the one piece of new machinery the novel
branch needs (a decision-cadence wrapper), so that neither branch can
quietly redefine the measurement after seeing a number.

=====================================================================
WHY THIS ROUND
=====================================================================

Constraints attacked: **N≈3** (effective sample size — this project's only
route to more than ~3 independent regime events is more time) and **ERR**
(no error control in the signal path).

Backlog item: **B-06**, forward paper trading. It is the only genuinely
open, unblocked item on the ranked list other than B-32 (an infrastructure
gap explicitly filed as "not urgent while every candidate fails its
interval anyway"), and every re-ranking since R-29 has called it "the
highest-value item on merit" and "the standing zero-cost recommendation".
Nine consecutive rounds (R-63, R-65, R-67, R-68, R-69, R-70, R-73, R-74,
R-75, R-76, R-77) have now closed with some form of R-67's line: *no
mechanism can narrow an interval — only more data, more breadth, or
forward evidence can.*

The whole project has therefore been leaning on B-06 for nine rounds, and
**nobody has asked it the two questions that decide whether it can carry
that weight**:

1. **How long must it run?** R-71's own next step is "let the record
   accumulate; a future round should not re-read this entry's tools until
   B-06 has enough rows for `anytime_valid_first_exclusion` to have a real
   chance of firing." *Enough rows* was never quantified. If the answer is
   months, B-06 is on track and the standing recommendation is sound. If
   the answer is centuries, the standing recommendation is void and the
   ledger has been recommending a placebo since R-29.
2. **Is it recording what it thinks it is recording?** The record has been
   live since 2026-08-19 and has never been audited against the backtest
   whose claim it exists to test.

This is not a duplicate of:

- **R-48** (built the recorder; never asked how long it needs).
- **R-71** (scheduled it, extended it to seven strategies, and built the
  anytime-valid reading tool; explicitly deferred "how many rows is
  enough" to a future round — this one).
- **R-29 / R-30 / R-70** (fixed-`n` inference *on the backtest dataset*;
  none of them touches the forward record or its horizon).
- Any SIZE/COST/INFO mechanism round: nothing here sweeps a strategy
  parameter or proposes a new mechanism.

=====================================================================
PRE-REGISTRATION — frozen before any number in either branch was read
=====================================================================

Both branches classify B-06's status. Nothing is promoted or rejected, so
ROUTINE.md's promotion bar does not apply; what is pre-registered instead
is the *classification rule*, so that a bad answer cannot be re-labelled a
good one after the fact.

---------------------------------------------------------------------
CONSERVATIVE BRANCH — `r78_conservative_b06_horizon.py`
---------------------------------------------------------------------

**Question.** At what forward sample size `n` (days of recorded paper
trading) does the Waudby-Smith & Ramdas (2024) anytime-valid confidence
sequence — the tool R-71 built for exactly this purpose,
`tradebot.inference.empirical_bernstein_confidence_sequence` — first
exclude zero on the paired daily-return difference
`kelly_regime_v4 − buy_and_hold` on spot, if the forward effect size
equals the one measured on 2017–2022?

**Method, frozen.** Estimate the paired daily-return difference series on
pre-2023 data only (inner-train 2017-01-01→2020-12-31 and
inner-validation 2021-01-01→2022-12-31, reported separately, never
pooled-then-cherry-picked). Resample forward paths from it with this
project's own `stationary_bootstrap_indices` at its established 30-day
mean block, run the real CS on each path, and read off
`anytime_valid_first_exclusion`.

**Both fee tiers are reported, and the live one is the decisive one.**
The comparison table assumes 0.10% taker; `scripts/paper_trade.py` charges
Bitstamp's real **0.40%** entry tier. B-06's forward record is therefore a
0.40% experiment, and the fee study (R-13, README) already says no
strategy here beats buy-and-hold at 0.40%. The horizon is computed at both
so that the difference between "the question the table asks" and "the
question B-06 is actually running" is visible rather than assumed away.

**Classification rule (pre-registered).** Let `n50` be the median first-
exclusion horizon in trading days across bootstrap paths, at the 0.40%
tier B-06 actually runs at, on the more recent (inner-validation) estimate:

- **ON TRACK** — `n50 <= 1095` (3 years) AND ≥50% of paths exclude zero
  within 5 years.
- **SLOW BUT VIABLE** — `n50 <= 9125` (25 years).
- **NOT VIABLE AS SPECIFIED** — `n50 > 9125`, or fewer than 50% of paths
  exclude zero within 25 years. In this case B-06's standing description
  in the ledger ("the only source of uncontaminated evidence", "the
  standing zero-cost recommendation") must be **qualified in place with
  the measured horizon**, and the backlog row updated. Nothing is deleted.

**The sign is reported, never suppressed.** If the sequence excludes zero
*against* the strategy, that is the finding.

**Pre-registered falsification test (chosen now, before any run).** The
machinery must pass both of:

- **F1 (null calibration on real-shaped data).** Recentre the same
  empirical difference series to exactly zero mean and rerun. Across
  bootstrap paths, the CS must exclude zero in **≤5%** of them at
  α=0.05. This is R-71's synthetic calibration check repeated on real
  autocorrelated, heavy-tailed inputs. If it fires more often than 5%,
  the horizon numbers are inflated and the round reports the machinery
  failure instead of a horizon.
- **F2 (power on a known effect).** Shift the same series so its daily
  mean is +0.001 (≈ +36.5%/yr simple, an effect far larger than anything
  this project has ever measured). The CS must exclude zero in **≥90%** of
  paths within 5 years. If it does not, the tool cannot detect even an
  implausibly large effect and no horizon it reports means anything.

---------------------------------------------------------------------
NOVEL BRANCH — `r78_novel_record_fidelity.py`
---------------------------------------------------------------------

**Question.** Is B-06's live record actually running the strategy that was
backtested?

**Three measurements, frozen.**

- **M1 — realized cadence.** Inter-row spacing of the committed
  `reports/paper_trading/*.csv` against the `*/15 * * * *` design and
  against the 5-minute bar the strategies decide on.
- **M2 — rebalance capture.** 18 of this project's registered strategies,
  the whole `kelly_regime` family among them, gate their order on
  `abs(target[i] - target[i-1]) > 1e-9` — an **edge-triggered** comparison
  against the immediately-preceding 5-minute bar, not a level comparison
  against the account's actual position. `inception_catchup_target()`'s
  own docstring names this and patches it *at inception only*. Measure, on
  pre-2023 data, what fraction of `kelly_regime_v4`'s target changes
  survive a 1-in-`k` decision grid for `k` ∈ {1, 3, 24, 288} — i.e. the
  designed 15-minute cadence, the realized cadence M1 measures, and a
  daily one.
- **M3 — what that costs.** Run `kelly_regime_v4` through the backtest
  engine on the full 5-minute frame, but with `on_bar` called only on the
  same 1-in-`k` grid, so indicators are identical and only the decision
  cadence changes. Report final balance, Sharpe, max drawdown and trade
  count against the `k=1` baseline, on inner-train and inner-validation.

**Classification rule (pre-registered).** The realized cadence is:

- **BOUNDED AND ACCEPTABLE** — as R-71 judged it — if, at the realized
  `k` from M1, v4's Sharpe falls by **≤0.2** (this project's own noise
  floor, R-20) on **both** windows and its final balance falls by ≤20% on
  both.
- **MATERIALLY COSTLY** otherwise, in which case the recorder is not
  measuring the strategy the ledger thinks it is, and the round must say
  so and propose the fix.

**Pre-registered falsification test (chosen now).** If M2 finds that
**≥90%** of v4's target changes survive the realized decision grid, the
edge-trigger concern is refuted, M3's result is attributable to something
else, and the novel branch has found nothing. Named now so it cannot be
quietly dropped if it fires.

=====================================================================
HOLDOUT DISCIPLINE
=====================================================================

Every frame this module hands either branch is truncated at
2022-12-31 23:59:59 at load time, with a runtime assertion. `OOS_START` is
imported for the assertion only and is never used as a slice bound for a
measurement. Expected holdout increment for this round: **+0**.

The live `reports/paper_trading/*.csv` files the novel branch reads are
dated 2026-08 but are explicitly outside the committed backtest dataset
and outside the holdout counter, by the same convention R-71 established
and stated: they only ever come from the live Bitstamp public feed.

Note the standing instruction from B-33/R-72: any round descended from
`r63_shared.py`'s pattern must retire or explicitly count its `BTC_HOLD`
cell. This round is **not** descended from that pattern — it builds no
`BTC_HOLD` cell and never calls `load_dataset()` without truncation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.inference import daily_returns  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# ------------------------------------------------------------------ windows

OOS_START = "2023-01-01"          # assertion sentinel ONLY; never a slice bound
LAST_ALLOWED = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")

W_TRAIN = ("2017-01-01", "2020-12-31")     # inner-train
W_VAL = ("2021-01-01", "2022-12-31")       # inner-validation

# The two fee tiers that matter. 0.10% is the comparison table's assumption;
# 0.40% is what scripts/paper_trade.py actually charges the live record.
FEE_TABLE = 0.0010
FEE_LIVE = 0.0040

TRADING_DAYS = {"1y": 365, "3y": 1095, "5y": 1825, "10y": 3650, "25y": 9125}


def load_truncated() -> tuple[pd.DataFrame, str]:
    """The committed spot frame, hard-truncated before the holdout."""
    df, label = load_dataset(ROOT / "data", "spot")
    df = df.loc[:LAST_ALLOWED]
    assert_no_holdout(df)
    return df, label


def assert_no_holdout(df: pd.DataFrame) -> None:
    """Refuse to hand a branch anything dated 2023-01-01 or later."""
    if len(df) and pd.Timestamp(df.index[-1]) > LAST_ALLOWED:
        raise AssertionError(
            f"holdout leak: frame reaches {df.index[-1]}, past {LAST_ALLOWED} "
            f"(OOS_START={OOS_START})")


# ------------------------------------------------- decision-cadence wrapper

class SparseDecision(Strategy):
    """Wrap a strategy so ``on_bar`` fires only on a 1-in-``k`` bar grid.

    This is the exact semantics of ``scripts/paper_trade.py`` under a cron
    slower than the bar interval: ``prepare()`` still sees the complete
    5-minute history (the recorder fetches the full window from Bitstamp
    every run, so every indicator is computed on every bar), but the
    strategy is only *asked for a decision* on the candles a run happens to
    land on. Candles between two runs are not delayed — they are never
    presented to ``on_bar`` at all.

    Isolating the cadence this way, rather than by decimating the frame to
    ``k``-bar candles, is deliberate: decimation would also change every
    indicator's horizon in wall-clock time and confound the measurement.
    Here ``k=1`` reproduces the unwrapped strategy bit-for-bit, which the
    novel branch checks explicitly before reading anything else.

    **Disclosed difference from the live recorder**, named rather than
    hidden: ``paper_trade.py`` constructs a *fresh* strategy instance on
    every invocation, so any state a strategy accumulates in ``self``
    during ``on_bar`` is discarded between decisions. This wrapper keeps
    one instance. For the ``kelly_regime`` family the two are identical —
    ``on_bar`` reads only the precomputed ``target`` column and holds no
    ``self`` state (see ``src/tradebot/strategies/kelly_regime.py``) — but
    the equivalence is family-specific, not general, and is not claimed for
    any other strategy.
    """

    def __init__(self, inner: Strategy, k: int, phase: int = 0) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.inner = inner
        self.k = int(k)
        self.phase = int(phase) % int(k)
        self.name = f"{inner.name}__k{k}"
        self.warmup = inner.warmup
        self.decisions_offered = 0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.inner.prepare(df)

    def on_bar(self, ctx: Context) -> None:
        if ctx.i % self.k != self.phase:
            return
        self.decisions_offered += 1
        self.inner.on_bar(ctx)


# -------------------------------------------------------- paired difference

def paired_daily_diff(df: pd.DataFrame, label: str, window: tuple[str, str],
                      fee: float, a: str = "kelly_regime_v4",
                      b: str = "buy_and_hold") -> pd.Series:
    """``a``'s daily return minus ``b``'s, over ``window``, on spot at ``fee``.

    Both arms are run through ``run_period`` so each enters the window warm
    and flat with the full starting balance (R-22's warmup-prefix bias).
    """
    market = MarketSpec.spot(fee_rate=fee)
    out = {}
    for name in (a, b):
        result = run_period(get_strategy(name), df, window[0], window[1],
                            market=market, data_label=label)
        out[name] = daily_returns(result.equity)
    joined = pd.concat([out[a], out[b]], axis=1, join="inner")
    joined.columns = ["a", "b"]
    return (joined["a"] - joined["b"]).dropna()


def bootstrap_paths(diffs: np.ndarray, n_days: int, n_paths: int,
                    mean_block: float = 30.0, seed: int = 78) -> np.ndarray:
    """``n_paths`` forward paths of length ``n_days`` resampled from ``diffs``.

    Uses this project's own stationary bootstrap at its established 30-day
    mean block, so the resampled paths carry the difference series' own
    autocorrelation rather than an i.i.d. idealization of it. When
    ``n_days`` exceeds ``len(diffs)`` the blocks wrap, which is the
    standard circular-block convention and the only way to project a
    horizon longer than the sample that estimates it — stated here because
    it is an assumption, not a free lunch: it presumes the forward regime
    resembles the estimation window.
    """
    from tradebot.inference import stationary_bootstrap_indices

    rng = np.random.default_rng(seed)
    n = len(diffs)
    reps = int(np.ceil(n_days / n))
    idx = stationary_bootstrap_indices(n, mean_block, n_paths * reps, rng)
    idx = idx.reshape(n_paths, reps * n)[:, :n_days]
    return diffs[idx]
