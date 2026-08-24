"""Shared, read-only utilities and pre-registration for the R-119 round (08-24).

DIRECTION, in one sentence: does selecting `kelly_regime_v4`'s own already-
swept free parameters (anchor-ladder base, `target_vol`, `max_leverage`) by
R-118's own robust CVaR criterion, but scored against synthetic paths whose
crash/jump component is calibrated from EXTERNALLY PUBLISHED estimates
(literature figures, never touching this project's own price file) instead
of parameters fit to this project's own 2017-2020 training window, escape
the ceiling R-118 found?

**Direct precedent, and the reason this round exists.** R-118 (08-24, same
day) tried two resampling/generative mechanisms for the SAME selection
problem -- a stationary block bootstrap of the realized window's own bars,
and a 3-state Markov-switching jump-diffusion model FIT BY METHOD-OF-MOMENTS
to that same window -- and both selected (a near-copy of) `kelly_regime_v4`'s
own shipped point estimate rather than a materially different configuration.
R-118's own closing diagnosis, quoted directly from `docs/LEDGER.md`: "no
resampling or low-order generative method fit *to that one window* has
anywhere to move the selection TO -- the plateau itself is the ceiling... the
only two remaining escape routes this ledger has ever named for N approx 3
are read from a genuinely different window (ETH, the six-asset panel -- both
already closed per the standing diagnosis) or forward evidence (B-06,
already running unattended)." That sentence is read literally here for a
THIRD option neither it nor R-45 named: a genuinely different window is not
the only way to inject information the training window itself never
contained -- a genuinely different *source*, published independently of this
project's own data, can too. This round is the first in this ledger to
calibrate a synthetic stress generator from numbers that were never computed
from this project's own price file at all.

**Which constraint this round attacks: N approx 3** (effective sample size
is approx 3 regime events, not 1.01M bars) -- directly, via the CALIBRATION
SOURCE of the selection procedure, a dimension R-45 (calendar folds of the
realized window) and R-118 (block-bootstrap / MOM-fit of the realized
window) both left untouched: every synthetic scenario either of those rounds
could construct was, by construction, bounded by what 2017-2020 BTC actually
did or by that window's own estimated moments. Both branches below leave
`kelly_regime_v4`'s `frac * scale` mechanism (R-62's factorization)
completely unchanged -- only which POINT on the already-validated (ladder,
target_vol, max_leverage) grid the CVaR criterion selects can differ.

**Literature grounding, fetched via WebSearch this round (2026-08-24):**

- Scaillet, O., Treccani, A., & Trevisan, C. (2020), "High-Frequency Jump
  Analysis of the Bitcoin Market," *Journal of Financial Econometrics*
  18(2), 209-232 (working paper: https://www.scaillet.ch/pdfs/bitcoin.pdf).
  Finds jumps are frequent at high frequency, averaging roughly ONE JUMP DAY
  PER WEEK across the sample studied. Used below as an external, non-fitted
  jump-arrival rate: `EXT_JUMP_DAYS_PER_WEEK = 1.0` -> daily jump
  probability `EXT_JUMP_PROB_PER_DAY = 1.0 / 7.0`.
- MDPI *Mathematics* 9(20), 2567 (2021), "Detecting Jump Risk and
  Jump-Diffusion Model for Bitcoin Options Pricing and Hedging"
  (https://www.mdpi.com/2227-7390/9/20/2567). Reports jump intensity
  9.89%-16.85% across 2015-2018 and, in a bull/bear split, 16.8%/29.2%;
  reports jump sizes economically significant with mean +4.65% for
  positive jumps and -4.14% for negative jumps. Used below as the external,
  non-fitted per-jump SIZE distribution: a signed jump drawn with equal
  probability from a positive component (mean +4.65%, disclosed
  simplification: std set to half the mean magnitude, 2.3%, since the
  source does not report jump-size dispersion) or a negative component
  (mean -4.14%, std 2.1%, same convention).
- Aggregate, cross-source crash catalogue (CCN "10 Biggest Bitcoin Crashes
  in History"; NYDIG Research "Charting Drawdowns During Up Cycles"; Live
  Volatile "Bitcoin Drawdown & Recovery Analysis"; CNBC 2026-02-12 "Bitcoin's
  drawdown hit 50%. History shows it may have further to go" -- four
  independent sources cross-checked this round, none affiliated with this
  project or fit to its data). Convergent figures used below, all external:
  since 2014, four drawdowns have exceeded 50%, the three largest averaging
  approximately an 80% peak-to-trough decline; since 2012, the average
  peak-to-trough decline across each ~4-year cycle's bear phase is
  approximately 75%; average time-to-recovery to the prior all-time high is
  approximately 643 days (~1.76 years). Used below as the external,
  non-fitted BEAR-STATE severity/duration pair for the novel branch's
  regime structure: `EXT_BEAR_DRAWDOWN = 0.775` (mean of 75% and 80%,
  disclosed as a simple average of the two headline figures rather than a
  cherry-pick of whichever is larger) and `EXT_BEAR_DURATION_DAYS = 365`
  (a bear PHASE inside an approximately 4-year cycle whose recovery leg
  alone averages ~643 days, so the bear phase itself -- top to trough, not
  trough to recovery -- is set to one year, the commonly cited "crypto
  winter" duration in the same sources, disclosed as coarser than the
  recovery figure because none of the four sources decomposes a full cycle
  into named phase lengths the way they do for drawdown depth).

None of these three figures was computed from `data/btcusd_*.csv.gz` or any
other file this project ships. They were read off independent, published
sources this round located by web search and are frozen here, before either
branch runs, exactly as `docs/ROUTINE.md` step 2 requires.

**Not a duplicate of:**

- R-118 (the direct precedent, see above): both R-118 branches calibrate
  their synthetic generator EXCLUSIVELY from `load_inner_train_btc()` --
  the stationary bootstrap resamples that data's own blocks, and the
  regime-switching model estimates drift/vol/transitions/jumps from that
  data's own moments via method-of-moments. Neither branch below fits
  anything to that file; both branches' crash/jump parameters come only
  from the three literature sources frozen above. (The novel branch below
  DOES still estimate one quantity from the training window: ordinary,
  non-crash-day diffusion volatility, exactly as R-118's own module
  docstring flags for its own AR(1) component -- disclosed in
  `experiments/r119_novel_regimeswitch_external.py`'s own docstring, and
  the round's falsification test is designed to be informative either way.)
- R-45 (three real calendar folds of the one realized window, minimax
  selection): neither branch below scores any config on a REAL sub-interval
  during selection; both select entirely on synthetic paths, exactly like
  R-118, differing from R-118 only in the calibration SOURCE.
- R-97 (Wasserstein-DRO Kelly sizing keyed to the causal regime-cycle
  count) and R-98/R-99 (GPD tail-shape / BNS jump-diffusion estimated from
  this project's OWN data, used as live regime-timing ALARMS): both fit
  their tail/jump parameters to this project's own bars and use them to
  time trades in real time, a structurally different role from this
  round's use of literature parameters purely to build OFFLINE synthetic
  scenarios for one-shot parameter SELECTION, never touched again once the
  grid point is frozen.
- R-40/R-45/R-06/R-07 (real-data point-estimate and calendar-fold
  searches): the baselines every N approx 3 attempt including this one is
  trying to beat.

**Is it simulable here?** Yes. Both branches consume zero new data files --
the three literature numbers above are Python float constants, and every
synthetic path is a plain OHLCV frame fed through the unchanged real engine
(`tradebot.engine.run_backtest`), exactly like R-118. The decision-bearing
numbers (Step 4, `evaluate_candidate`) are measured on 100% real market data,
reusing R-118's own frozen falsification pipeline UNCHANGED (imported, not
copied, so the promotion bar is provably identical across both rounds).

**What would make each branch fail, named now, before any code ran:**

- Conservative (GBM diffusion + externally-calibrated compound-Poisson
  jump, no regime-switching structure at all): this is the simplest
  possible reading of the three literature numbers, with no additional
  machinery to inject a genuinely different regime TYPE beyond "jumps
  happen at the literature's rate and size." The pre-registered
  expectation is that it behaves like a jump-augmented but otherwise
  memoryless single-regime path -- likely to select a MORE conservative
  (lower max_leverage and/or lower target_vol) grid point than v4's own
  default, because unconditional jump risk penalizes leverage everywhere
  rather than concentrating penalty in a detectable bear state, and to
  still fail the falsification bar for the same reason R-118's branches
  did (a materially different selection does not by itself imply it
  generalizes better to the BTC pre-2020 control).
- Novel (3-state regime structure matching R-118's novel branch's SHAPE,
  bear-state severity/duration set from the external crash catalogue
  instead of MOM-fit): if the externally-sourced bear state (77.5% decline,
  ~1-year phase) is not materially more severe or differently shaped than
  what R-118's OWN fitted bear state already produced from 2017-2020's real
  2018 collapse, this branch will reproduce R-118's finding for a new
  reason -- not because internal-window fitting is inherently bounded, but
  because the external, independently-published consensus on "how bad does
  a BTC bear get" turns out to be QUANTITATIVELY SIMILAR to what the one
  training window already contained, which would be the more informative
  result: the ceiling is not an artifact of fitting-to-one-window after
  all, it is that 2017-2020 (which contains the whole 2018 bear) was
  already a fair sample of BTC's own long-run crash behavior. If instead
  the externally-calibrated bear state is quantitatively harsher (deeper
  and/or more prolonged) than 2018 was, and selection moves to a materially
  different, more defensive grid point, the falsification step is where
  this round's real news would be: does that more-defensive point still
  underperform v4 on the BTC pre-2020 control (repeating R-118's finding
  for a THIRD independent reason) or does it, for the first time in this
  N approx 3 line of attack, do no worse?

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither may edit it. It imports R-118's frozen
generic machinery (`GRID`, `build_kelly`, `robust_score`, `select_config`,
`evaluate_candidate`, data loaders, `V4_DEFAULT`) UNCHANGED, so the grid
searched and the promotion bar applied are provably identical to R-118's own
-- only the synthetic path GENERATOR differs, isolating the one variable
this round tests. Nothing here, and nothing either branch may add, reads a
bar at or after OOS_START (2023-01-01).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from experiments.r118_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    CVAR_FRACTION,
    FEE_TIER,
    FUTURES,
    GRID,
    LADDER_BASES,
    MARKETS,
    MAX_LEV_GRID,
    N_DRAWS,
    SHARPE_NOISE_FLOOR,
    SPOT,
    TARGET_VOL_GRID,
    V4_DEFAULT,
    assert_no_holdout,
    build_kelly,
    evaluate_candidate,
    load_bitfinex_pair,
    load_btc_full,
    load_inner_train_btc,
    print_report,
    robust_score,
    score_on_path,
    select_config,
)

# ------------------------------------------------------------------------
# External literature constants -- FROZEN before either branch runs.
# Every branch's path_generator MUST import these, never redefine them.
# ------------------------------------------------------------------------

# Scaillet, Treccani & Trevisan (2020): ~1 jump day per week.
EXT_JUMP_DAYS_PER_WEEK = 1.0
EXT_JUMP_PROB_PER_DAY = EXT_JUMP_DAYS_PER_WEEK / 7.0

# MDPI Mathematics 9(20) 2567 (2021): mean jump sizes +4.65% / -4.14%,
# disclosed-simplification std = half the mean magnitude (source does not
# report dispersion).
EXT_JUMP_UP_MEAN = 0.0465
EXT_JUMP_UP_STD = 0.0465 / 2.0
EXT_JUMP_DOWN_MEAN = -0.0414
EXT_JUMP_DOWN_STD = 0.0414 / 2.0

# Cross-source crash catalogue (CCN, NYDIG, Live Volatile, CNBC 2026-02-12),
# averaged across sources, not cherry-picked.
EXT_BEAR_DRAWDOWN = 0.775          # mean of "75%" and "80%" headline figures
EXT_BEAR_DURATION_DAYS = 365.0     # ~1yr crash phase inside a ~4yr cycle


def _self_test() -> None:
    assert GRID and len(GRID) == 12
    assert V4_DEFAULT == (20, 0.55, 2.0)
    assert abs(EXT_JUMP_PROB_PER_DAY - 1.0 / 7.0) < 1e-12
    assert EXT_JUMP_UP_MEAN > 0 > EXT_JUMP_DOWN_MEAN
    assert 0.0 < EXT_BEAR_DRAWDOWN < 1.0
    # Sanity: R-118's own machinery still reproduces v4 bit for bit.
    from tradebot.registry import get_strategy
    v4 = get_strategy("kelly_regime_v4")
    cand = build_kelly(*V4_DEFAULT)
    idx = np.arange(150_000)
    import pandas as pd
    dts = pd.date_range("2017-01-01", periods=len(idx), freq="5min", tz="UTC")
    rng = np.random.default_rng(119)
    close = 10_000 * np.exp(np.cumsum(rng.normal(0, 0.0005, len(idx))))
    df = pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995,
                       "close": close, "volume": 1.0}, index=dts)
    df_v4 = v4.prepare(df.copy())
    df_cand = cand.prepare(df.copy())
    assert np.allclose(df_v4["target"].to_numpy(), df_cand["target"].to_numpy(),
                       equal_nan=True), "r119_shared: build_kelly(V4_DEFAULT) drifted from v4"


_self_test()
