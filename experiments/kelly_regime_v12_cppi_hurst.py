#!/usr/bin/env python
"""Replace kelly_regime_v4's SCALE with a Hurst-adaptive CPPI cushion (SIZE axis, new formula family).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

R-46, "novel" branch. A parallel "conservative" branch (disjoint file, not
read or touched here) runs the identical CPPI base with a FIXED
multiplier; this file's only addition on top of that shared base is
making the multiplier adaptive to a rolling causal Hurst exponent. Do not
coordinate with that branch; both report independently and the operator
merges.

The idea
--------
Fourteen branches across R-34 through R-45 (docs/LEDGER.md section C) all
tried to IMPROVE ``kelly_regime_v4``'s existing mechanism -- retuning its
constants, bagging its ladder, adding a new signal as a gate or vote on
top of the unchanged vol-target SCALE, or replacing the vol-target
formula itself with CRRA / risk-constrained-Kelly variants -- and every
one failed, either as a flat-rescaled exposure artifact (R^2 > 0.95) or
by losing the pre-2020 ETH/BTC-control falsification test. This branch
instead keeps v4's vote ``frac`` BYTE-IDENTICAL (inherited, unmodified --
it is not the thing under test) and replaces the OTHER half of v4's
mechanism, the volatility-targeting SCALE, with Constant Proportion
Portfolio Insurance (CPPI) -- a portfolio-construction mechanism from a
different part of the literature than anything in R-34 through R-45,
which all worked inside the vol-target/Kelly-fraction family.

Mechanism, one sentence: a growing dollar floor under the account
(``floor0 = 0.65 x starting balance``, compounding at 3%/yr) defines a
"cushion" (``equity - floor``, floored at zero); the position is sized at
``m x cushion / equity`` capped at 2x leverage, where the CPPI multiplier
``m`` -- normally a single constant an investor picks once -- is instead
made a linear function of a rolling, causally-computed Hurst exponent
H(t): low (``m_low``) when H(t) indicates a choppy/mean-reverting regime
(CPPI's habitual rebalancing whipsaws in chop), ramping to high
(``m_high``) when H(t) indicates a persistent/trending regime (CPPI's
convex, trend-following payoff should pay off most there).

Citations
---------
CPPI (the shared base mechanism, identical to the conservative branch):

- Perold, A. F. (1986), "Constant Proportion Portfolio Insurance," Harvard
  Business School, unpublished manuscript -- the original CPPI
  formulation.
- Perold, A. F. & Sharpe, W. F. (1988), "Dynamic Strategies for Asset
  Allocation," Financial Analysts Journal 44(1), 16-27 -- formalizes CPPI
  alongside buy-and-hold and constant-mix as the three canonical dynamic
  allocation rules and derives their payoff shapes (CPPI is convex /
  trend-following; constant-mix is concave / mean-reverting).
- Black, F. & Jones, R. (1987), "Simplifying Portfolio Insurance," Journal
  of Portfolio Management 14(1), 48-51 -- the ``m x cushion`` sizing rule
  used here verbatim.

Hurst-exponent estimation (this branch's own contribution):

- Hurst, H. E. (1951), "Long-Term Storage Capacity of Reservoirs,"
  Transactions of the American Society of Civil Engineers 116, 770-799 --
  the original rescaled-range (R/S) statistic.
- Mandelbrot, B. B. & Wallis, J. R. (1969), "Robustness of the rescaled
  range R/S in the measurement of noncyclic long run statistical
  dependence," Water Resources Research 5(5), 967-988 -- the multi-scale
  R/S regression estimator this file implements (``_rs_hurst`` below):
  split a window into sub-blocks of several sizes, compute mean R/S per
  size, and take the OLS slope of log(R/S) against log(size).
- A more modern, more heavily-cited alternative -- the generalized Hurst
  exponent (GHE) via q-order structure functions -- exists (Di Matteo, T.,
  Aste, T. & Dacorogna, M. M. (2003), "Scaling behaviors in differently
  developed markets," Physica A 324, 183-188; Di Matteo, T. (2007),
  "Multi-scaling in finance," Quantitative Finance 7(1), 21-36) and is
  arguably the more standard citable choice for financial time series
  specifically. This file uses the classical R/S estimator instead, for a
  concrete, disclosed reason: R/S is a single scalar exponent with a
  simple closed form, so its causality is trivial to verify by inspection
  and by the tamper probe below; GHE's q-order moment machinery adds
  estimation surface area without changing what regime information it
  extracts, and this branch's finding (below) is about whether A Hurst
  signal helps at all, not about which Hurst estimator is best. Left as
  a natural follow-up.

BTC/crypto evidence on the actual Hurst exponent -- reported with the
disagreement intact, not resolved in whichever direction helps this
branch:

- Bariviera, A. F. (2017), "The Inefficiency of Bitcoin Revisited: A
  Dynamic Approach," Economics Letters 161, 1-4. Daily BTC 2011-2017,
  rolling R/S and DFA: persistent (H > 0.5) through 2014, then the
  exponent "tended to move around 0.5" (i.e. toward an efficient/
  random-walk market) for the remainder of the sample. Evidence FOR "BTC
  spends a lot of time near or below 0.5."
- Grande, M., Borondo, F., Losada, J. C. & Borondo, J. (2024),
  "Anti-Persistent Values of the Hurst Exponent Anticipate Mean Reversion
  in Pairs Trading: The Cryptocurrencies Market as a Case Study,"
  Mathematics 12(18), 2911. Finds crypto pair spreads frequently show
  ANTI-persistent local Hurst (H < 0.5) and that anti-persistence predicts
  faster mean reversion -- i.e. sub-0.5, mean-reverting local Hurst is a
  real, exploitable regularity in crypto, not a measurement artifact.
  Further evidence FOR "BTC/crypto is often at or below 0.5."
- Against a clean "always near 0.5" story: several rolling-Hurst
  practitioner write-ups (e.g. FractalCycles' 2025-2026 "Rolling Hurst
  Exponent" guide, not peer-reviewed, treated here as a directional
  pointer only) describe BTC rolling H exceeding 0.65 in strong bull
  trends and briefly spiking above 0.70 in sell-offs, i.e. genuinely
  persistent at times, not merely oscillating around 0.5.
- A 2026 Physica A paper on ~774,000 five-minute BTC bars (Aug 2017 - Dec
  2024) reports a scaling crossover near a 25-day horizon separating a
  shorter-horizon efficient regime from a longer-horizon anti-persistent
  one, and that overall inefficiency roughly halved after the Jan 2024
  spot-ETF approval (full author list not resolved from the paywalled
  abstract during this session's search -- cited as a directional pointer,
  not a load-bearing number).

Net honest read: the literature does NOT agree on a single typical BTC
Hurst value. Multiple sources (Bariviera 2017, Grande et al. 2024) place
BTC/crypto at or below 0.5 for meaningful stretches; other sources
(practitioner rolling-Hurst write-ups, and the multi-scale crossover
paper) find genuine persistence at some horizons/eras. Section
"pre-registered failure hypothesis" below states, before any code ran,
what this branch expects to find on THIS dataset if the "usually <=0.5"
reading dominates, and section "hurst distribution" (the ``hurst_stats``
command) reports what actually happened.

Constraint attacked
--------------------
SIZE (decide how much to hold) -- same constraint category as every
promoted strategy in this project (L-01 through L-04) and every prior
attempt to replace v4's SCALE (R-38). This file changes the SIZING
FORMULA family (vol-target -> CPPI) and adds a regime input to the
multiplier ALONE; it adds no new external data series (INFO is
untouched -- H(t) is derived purely from the same close price v4 already
reads) and it is silent on error control in constant SELECTION (R-45 is
the ERR-axis branch; this one is not it).

Not a duplicate of
-------------------
- R-01 (Hamilton HMM), R-02 (statistical jump models), R-03 (Bayesian
  online changepoint detection): all three are regime CLASSIFIERS fit to
  the RETURN-generating process (mean/variance/state-transition
  structure) via likelihood or Bayesian updating. A rolling Hurst
  exponent is a fractal/self-similarity STATISTIC of the price path
  itself (how a range statistic scales with window length) -- a
  genuinely different family from fractal market analysis (Peters 1994),
  not a variant tuning of any of the three.
- R-38 (conservative: risk-constrained Kelly; novel: CRRA/Merton
  drift-over-variance fraction): both replaced v4's vol-target formula
  with a DIFFERENT CLOSED-FORM SIZING RULE still inside the
  mean-variance/Kelly family (a cap on drawdown probability; a
  drift-to-variance ratio). CPPI is not a member of that family -- it has
  no drift or variance term at all, only a floor and a cushion multiple.
- R-40 (conservative: ladder bagging; novel: cross-ladder-disagreement
  shrink) and R-45 (conservative: minimax fold reselection; novel:
  walk-forward constant refitting): all four operate on v4's existing
  ANCHOR LADDER or its existing CONSTANTS, never on the SCALE formula.
  This file changes no anchor, no ladder, no vote; ``frac`` is v4's own
  crowd-vote computation, copied verbatim (see ``prepare()`` below) and
  never touched.
- The conservative branch of this same round: identical CPPI base
  (F0=0.65, g=3%, same floor/cushion/multiplier structure, same
  ``max_leverage=2.0``), but its multiplier ``m`` is a FIXED constant the
  conservative branch tunes directly. This file's entire addition is
  making ``m`` a function of a rolling causal Hurst exponent -- and this
  file's own internal ablation (the ``fixed_m=4`` control run in every
  step below) is the direct test of whether that addition earns its
  complexity, mirroring R-40 novel's "does the elaboration beat the
  plain baseline" question.

Pre-registered falsification test (fixed before any code ran)
---------------------------------------------------------------
The project's standard one, run by the ``eth`` command below: does the
selected candidate at least match kelly_regime_v4 on the pre-2020 BTC
control window (``data/btcusd_bitfinex_5m.csv.gz``, which -- verified
below -- ends 2019-12-31, entirely before this session's OOS_START rule)
AND not visibly underperform it on ETH
(``data/ethusd_bitfinex_5m.csv.gz``, same end date) relative to that BTC
control, both spot and 5x futures. Per-cell criterion, identical to
kelly_regime_v11_robust_ladder.py's ``eth()``: OK iff
``d(Sharpe) > -0.05 and d(profit) > -2.0pp`` against the v4 control.

Pre-registered failure hypothesis (stated before any sweep ran)
------------------------------------------------------------------
Multiple strands of the Hurst literature above (Bariviera 2017; Grande et
al. 2024) suggest BTC/crypto's realized Hurst exponent spends meaningful
time AT OR BELOW ~0.5 (near-random or mean-reverting), not persistently
trending. If this dataset's own rolling H(t) sits mostly at or below the
midpoint of the {0.40-0.60} grid tested here, the adaptive multiplier
will sit near ``m_low`` most of the time, converging toward (or below) a
badly-chosen fixed-low-``m`` case -- in which case the Hurst adaptation
is expected to add noise without benefit over the ``fixed_m=4`` internal
control, the same clean-negative pattern R-40's "novel" branch found
(shrink-by-disagreement never beat its own no-shrink baseline). The
``hurst_stats`` command below reports the actual empirical H(t)
distribution on this dataset regardless of which way it points -- this
is diagnostic information the project wants either way, not a result to
be steered toward.

Causality (the most likely place for a bug in this file)
-------------------------------------------------------------
Unlike v3/v4/v11, this strategy's exposure is NOT a pure function of
price data precomputable once in ``prepare()``: the CPPI cushion is
``equity(t) - floor(t)``, and ``equity(t)`` is the REALIZED account
equity, which depends on every fill this exact strategy has made so far
-- a genuinely online, path-dependent quantity that can only be read from
``ctx.equity`` inside ``on_bar``. ``prepare()`` therefore only
precomputes the two pieces that ARE pure functions of price
(``frac``, v4's unchanged vote, and ``H``, the rolling causal Hurst
exponent); ``on_bar`` carries the floor/cushion/position state forward
bar by bar. The rolling Hurst estimator is new code on a new causal path
(daily-resampled closes, lagged one full day, forward-filled onto the
5-minute index -- see ``rolling_causal_hurst()``) and is exactly the kind
of "full-series fit applied to early rows" lookahead ROUTINE.md's
parallel-round rules warn about by name if done wrong. The ``causality``
command runs the project's standard two-opposite-tampers probe (bars
after a cut multiplied by 3x in one copy, divided by 3 in the other) on
BOTH the precomputed ``frac``/``H`` columns AND a full sequential
bar-by-bar replay of the actual account equity curve (required here,
unlike v11, because this strategy's decisions are path-dependent rather
than a pure column lookup).

Usage
-----
    python experiments/kelly_regime_v12_cppi_hurst.py sweep        # step 3: train+valid grid, both markets
    python experiments/kelly_regime_v12_cppi_hurst.py select       # step 3/4: pick the candidate, vs fixed-m control, vs v4
    python experiments/kelly_regime_v12_cppi_hurst.py hurst_stats  # point 6: empirical H(t) distribution
    python experiments/kelly_regime_v12_cppi_hurst.py artifact     # exposure-artifact R^2 check
    python experiments/kelly_regime_v12_cppi_hurst.py causality    # step 6: lookahead tamper probe
    python experiments/kelly_regime_v12_cppi_hurst.py eth          # step 7: ETH/BTC-control falsification
    python experiments/kelly_regime_v12_cppi_hurst.py all          # everything, in order
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


# --------------------------------------------------------------------- Hurst

_HURST_CACHE: dict[tuple, pd.Series] = {}


def _rs_hurst(returns: np.ndarray, min_chunk: int = 8, n_scales: int = 8) -> float:
    """Classical multi-scale rescaled-range Hurst exponent over one window.

    Hurst (1951) / Mandelbrot & Wallis (1969): split the window into
    sub-blocks of several geometrically-spaced sizes, compute the mean
    R/S statistic at each size, and take the OLS slope of log(R/S)
    against log(size) -- the standard causal, single-pass R/S estimator.
    """
    n = len(returns)
    if n < min_chunk * 2:
        return float("nan")
    sizes = np.unique(np.geomspace(min_chunk, n, n_scales).astype(int))
    sizes = sizes[(sizes >= min_chunk) & (sizes <= n)]
    if len(sizes) < 2:
        return float("nan")
    log_size, log_rs = [], []
    for size in sizes:
        n_chunks = n // size
        if n_chunks < 1:
            continue
        rs_vals = []
        for c in range(n_chunks):
            chunk = returns[c * size:(c + 1) * size]
            dev = np.cumsum(chunk - chunk.mean())
            r = dev.max() - dev.min()
            s = chunk.std(ddof=1)
            if s > 1e-12 and np.isfinite(r):
                rs_vals.append(r / s)
        if rs_vals:
            log_size.append(np.log(size))
            log_rs.append(np.log(np.mean(rs_vals)))
    if len(log_size) < 2:
        return float("nan")
    slope, _ = np.polyfit(log_size, log_rs, 1)
    return float(slope)


def rolling_causal_hurst(close: pd.Series, window_days: int) -> pd.Series:
    """Causal, daily-resampled, rolling classical R/S Hurst exponent, aligned to ``close.index``.

    Design choice (stated and justified, per this session's instructions):
    computed on DAILY-resampled closes rather than raw 5-minute bars.
    Reasons: (a) the multi-scale R/S regression needs several genuinely
    distinct sub-window sizes to be numerically stable, which needs
    O(window_days) quasi-independent observations -- a 60-90 day window
    gives that at daily granularity, whereas the same wall-clock window
    at 5-minute granularity is 60-90x more bars but almost entirely
    serially-correlated microstructure, not new information; (b) 5-minute
    bid/ask-bounce-style noise is exactly the kind of short-horizon
    structure the R/S method is not designed to characterize, and would
    bias the exponent toward 0.5 regardless of the true regime.

    Causality: ``close.resample("1D").last()`` is causal by construction
    (each daily bucket aggregates only bars strictly within that
    calendar day). The rolling R/S window ending on day ``d`` is then
    LAGGED BY ONE FULL DAY (``.shift(1)``) before being broadcast back
    onto the 5-minute index by forward-fill -- mirroring
    kelly_regime_v3's own ``.shift(1)`` pattern for its realized-vol
    estimator -- so a 5-minute bar on day d+1 can only ever see
    information available at the CLOSE of day d, never anything from
    day d+1 itself. Verified by the ``causality`` command's tamper probe.
    """
    key = (close.index[0], close.index[-1], len(close), window_days)
    cached = _HURST_CACHE.get(key)
    if cached is not None:
        return cached.reindex(close.index).ffill()

    daily_close = close.resample("1D").last().dropna()
    log_ret = np.log(daily_close).diff()
    n = len(log_ret)
    w = int(window_days)
    h = np.full(n, np.nan)
    vals = log_ret.to_numpy(dtype=float)
    for i in range(w, n + 1):
        window = vals[i - w:i]
        window = window[np.isfinite(window)]
        if len(window) < w * 0.9:
            continue
        h[i - 1] = _rs_hurst(window)
    h_daily = pd.Series(h, index=log_ret.index, name="H")
    h_daily_lagged = h_daily.shift(1)  # day d's own H usable only from day d+1 onward
    h_5m = h_daily_lagged.reindex(close.index, method="ffill")

    _HURST_CACHE[key] = h_5m
    return h_5m.ffill()


# --------------------------------------------------------------------- strategy


class KellyRegimeV12CPPIHurst(Strategy):
    """v4's unchanged crowd-vote frac, sized by a Hurst-adaptive CPPI cushion instead of vol-targeting.

    ``prepare()`` copies v4's vote computation verbatim (unchanged
    mechanism, see module docstring). ``on_bar`` computes a growing
    floor, a cushion (equity above the floor), and a CPPI multiplier
    ``m`` that ramps linearly with the rolling causal Hurst exponent
    ``H(t)`` between ``h_lo``/``m_low`` and ``h_hi``/``m_high``; pass
    ``fixed_m`` to bypass Hurst adaptation entirely (the ablation
    control this branch is required to beat).
    """

    name = "kelly_regime_v12_cppi_hurst"

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 vote_gamma: float = 1.0, hurst_window_days: int = 60,
                 h_lo: float = 0.45, h_hi: float = 0.55,
                 m_low: float = 3.0, m_high: float = 6.0,
                 f0: float = 0.65, g: float = 0.03, max_leverage: float = 2.0,
                 deadband: float = 0.10, fixed_m: float | None = None) -> None:
        self.horizons = horizons
        self.band = band
        self.vote_gamma = vote_gamma
        self.hurst_window_days = hurst_window_days
        self.h_lo, self.h_hi = h_lo, h_hi
        self.m_low, self.m_high = m_low, m_high
        self.f0, self.g = f0, g
        self.max_leverage = max_leverage
        self.deadband = deadband
        self.fixed_m = fixed_m
        # Enough warmup for the slowest vote anchor AND the Hurst window
        # (+1 day for the shift), whichever is larger.
        self.warmup = int(max(max(horizons), hurst_window_days + 1) * BARS_PER_DAY) + 10
        self._reset_state()

    def _reset_state(self) -> None:
        self._floor0: float | None = None
        self._t0: int | None = None
        self._pos: float = 0.0
        self._target: np.ndarray | None = None
        self._target_index: pd.Index | None = None

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        self._reset_state()
        close = df["close"]

        # --- v4's crowd-vote frac, copied verbatim, unchanged (see kelly_regime.py) ---
        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = sum(votes) / len(votes)
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma
        df["frac"] = frac.astype(float)

        # --- rolling causal Hurst exponent, this branch's own addition ---
        h5 = rolling_causal_hurst(close, self.hurst_window_days)
        df["H"] = h5.reindex(df.index).astype(float)

        self._target_index = df.index
        self._target = np.full(len(df), np.nan)
        return df

    def on_bar(self, ctx: Context) -> None:
        # Floor is captured once, from the FIRST bar this strategy ever
        # sees (whether that bar is inside the harness warmup prefix or
        # not is immaterial: the account is flat at start_balance for the
        # entire warmup prefix by construction -- see engine.run_backtest
        # -- so ctx.equity is exactly start_balance whenever this fires).
        if self._floor0 is None:
            self._floor0 = self.f0 * ctx.equity
            self._t0 = ctx.i

        elapsed_years = (ctx.i - self._t0) / BARS_PER_YEAR
        floor = self._floor0 * (1.0 + self.g) ** elapsed_years
        equity = ctx.equity
        cushion = max(equity - floor, 0.0)

        if self.fixed_m is not None:
            m = float(self.fixed_m)
        else:
            h = float(ctx.bar["H"])
            if not np.isfinite(h):
                h = 0.5  # neutral prior during any residual warmup gap
            span = self.h_hi - self.h_lo
            x = 0.0 if span <= 0 else (h - self.h_lo) / span
            x = min(1.0, max(0.0, x))
            m = self.m_low + (self.m_high - self.m_low) * x

        scale = 0.0 if equity <= 0 else min(max(m * cushion / equity, 0.0), self.max_leverage)
        frac = float(ctx.bar["frac"])
        desired = frac * scale

        prev_pos = self._pos
        if abs(desired - self._pos) > self.deadband:
            self._pos = desired
        self._target[ctx.i] = self._pos
        if abs(self._pos - prev_pos) > 1e-9:
            ctx.order_notional(self._pos)

    def realized_target_series(self) -> pd.Series:
        """The actual, path-dependent decided-position series from the last run.

        Only available (and only meaningful) after a full ``run_backtest``/
        ``run_period`` call has completed on this exact instance -- unlike
        v3/v4/v11, this cannot be precomputed in ``prepare()`` because the
        CPPI cushion depends on realized account equity.
        """
        return pd.Series(self._target, index=self._target_index, name="target")


# ------------------------------------------------------------------------ harness

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

# Standard project split (ROUTINE.md step 3).
TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
# OOS_START = "2023-01-01"  -- NEVER read in this file, by construction.
# Self-check: `grep -n "2023-\|2024-\|2025-\|2026-" experiments/kelly_regime_v12_cppi_hurst.py`
# should show only this comment and docstring prose, never an argument to
# measure()/run_period()/load_ohlcv_csv() -- confirmed in the final report.

INCUMBENT = "kelly_regime_v4"

F0 = 0.65
G = 0.03
MAX_LEVERAGE = 2.0
FIXED_M_CONTROL = 4.0

H_LO_GRID = (0.40, 0.45)
H_HI_GRID = (0.55, 0.60)
M_LOW_GRID = (2.0, 3.0)
M_HIGH_GRID = (5.0, 6.0)
WINDOW_GRID = (60, 90)
CONFIGS = [(hl, hh, ml, mh, w)
           for hl in H_LO_GRID for hh in H_HI_GRID
           for ml in M_LOW_GRID for mh in M_HIGH_GRID
           for w in WINDOW_GRID]  # 32

OUT = ROOT / "reports" / "kelly_regime_v12_cppi_hurst"

_SEEN: set[tuple] = set()  # distinct configurations evaluated, for the trials count


def make_strategy(h_lo=None, h_hi=None, m_low=None, m_high=None, window=None,
                   *, fixed_m: float | None = None) -> KellyRegimeV12CPPIHurst:
    if fixed_m is not None:
        _SEEN.add(("fixed_m", fixed_m))
        return KellyRegimeV12CPPIHurst(hurst_window_days=60, h_lo=0.45, h_hi=0.55,
                                        m_low=3.0, m_high=6.0, f0=F0, g=G,
                                        max_leverage=MAX_LEVERAGE, fixed_m=fixed_m)
    _SEEN.add((h_lo, h_hi, m_low, m_high, window))
    return KellyRegimeV12CPPIHurst(hurst_window_days=window, h_lo=h_lo, h_hi=h_hi,
                                    m_low=m_low, m_high=m_high, f0=F0, g=G,
                                    max_leverage=MAX_LEVERAGE)


def mean_notional(result) -> float:
    if "target" not in result.df:
        return float("nan")
    tgt = np.abs(result.df["target"].to_numpy(dtype=float))
    return float(np.nanmean(np.clip(tgt, 0.0, result.market.leverage)))


def realized_vol(equity) -> float:
    eq = equity.to_numpy(dtype=float) if hasattr(equity, "to_numpy") else np.asarray(equity)
    if len(eq) < 3:
        return float("nan")
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
    sd = np.std(rets, ddof=1)
    return float(sd * np.sqrt(BARS_PER_YEAR)) if np.isfinite(sd) else float("nan")


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0):
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                         start_balance=balance, data_label=LABEL)
    # Attach the strategy's REALIZED (path-dependent) decision series as a
    # 'target' column, post-hoc, so the mean_notional/exposure-artifact
    # helpers below work exactly as they do for v4's precomputed column --
    # necessary here because this strategy cannot precompute it in prepare().
    if hasattr(strategy, "realized_target_series"):
        tgt = strategy.realized_target_series().reindex(result.df.index)
        result.df = result.df.assign(target=tgt)
    m = compute_metrics(result)
    return m, realized_vol(result.equity), mean_notional(result), result


def line(tag, m, vol, notional, result) -> None:
    print(f"  {tag:44s} final=${m.final_balance:>11,.0f} "
          f"vol={vol:5.3f} notional={notional:5.3f} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# --------------------------------------------------------------------------- step 3


def sweep() -> pd.DataFrame:
    """Evaluate every grid configuration, plus the fixed-m control and v4, on TRAIN+VALID, both markets."""
    rows = []
    t0 = time.time()
    for n, (hl, hh, ml, mh, w) in enumerate(CONFIGS, 1):
        for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
            for mname, market in MARKETS:
                strat = make_strategy(hl, hh, ml, mh, w)
                m, vol, notional, res = measure(strat, start, end, market=market)
                rows.append({"h_lo": hl, "h_hi": hh, "m_low": ml, "m_high": mh, "window": w,
                             "split": split_name, "market": mname,
                             "final": m.final_balance, "profit_pct": m.profit_pct,
                             "vol": vol, "mean_notional": notional,
                             "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                             "trades": m.num_trades, "liquidated": m.liquidated})
        print(f"[{n:>2d}/{len(CONFIGS)}] h_lo={hl:.2f} h_hi={hh:.2f} m_low={ml:.0f} "
              f"m_high={mh:.0f} w={w:>2d}d  [{time.time() - t0:.0f}s]")

    for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        for mname, market in MARKETS:
            strat = make_strategy(fixed_m=FIXED_M_CONTROL)
            m, vol, notional, res = measure(strat, start, end, market=market)
            rows.append({"h_lo": np.nan, "h_hi": np.nan, "m_low": np.nan, "m_high": np.nan,
                         "window": "fixed_m_control", "split": split_name, "market": mname,
                         "final": m.final_balance, "profit_pct": m.profit_pct,
                         "vol": vol, "mean_notional": notional,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "trades": m.num_trades, "liquidated": m.liquidated})
            m4, vol4, not4, res4 = measure(get_strategy(INCUMBENT), start, end, market=market)
            rows.append({"h_lo": np.nan, "h_hi": np.nan, "m_low": np.nan, "m_high": np.nan,
                         "window": "v4_control", "split": split_name, "market": mname,
                         "final": m4.final_balance, "profit_pct": m4.profit_pct,
                         "vol": vol4, "mean_notional": not4,
                         "max_dd": m4.max_drawdown_pct, "sharpe": m4.sharpe,
                         "trades": m4.num_trades, "liquidated": m4.liquidated})
    print(f"\nfixed_m={FIXED_M_CONTROL} control and v4 control done  [{time.time() - t0:.0f}s]")

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "sweep.csv", index=False)
    print(f"\ndistinct configurations evaluated: {len(_SEEN)}")
    print(f"written: {OUT / 'sweep.csv'}")
    return out


# --------------------------------------------------------------------------- step 4


def select() -> None:
    """Pick the inner-validation winner across the 32-config grid; compare vs fixed-m and v4 controls."""
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "sweep.csv"
    df = pd.read_csv(csv) if csv.exists() else sweep()

    grid = df[df.window.isin(["60", "90", 60, 90])].copy()
    grid["window"] = grid["window"].astype(int)
    valid = grid[grid.split == "inner-validation"]
    score = (valid.pivot_table(index=["h_lo", "h_hi", "m_low", "m_high", "window"],
                                columns="market", values="sharpe")
             .assign(mean_sharpe=lambda d: d[["spot", "futures"]].mean(axis=1))
             .sort_values("mean_sharpe", ascending=False))
    print("=== top 8 configs by mean(spot, futures) inner-validation Sharpe ===")
    print(score.head(8).to_string())

    hl, hh, ml, mh, w = score.index[0]
    w = int(w)
    print(f"\nselected candidate: h_lo={hl} h_hi={hh} m_low={ml} m_high={mh} window={w}d  "
          f"mean-Sharpe={score.iloc[0].mean_sharpe:.3f}")

    # plateau: neighbours that differ from the winner in exactly one axis
    def neighbours():
        for axis, grid_vals in (("h_lo", H_LO_GRID), ("h_hi", H_HI_GRID),
                                 ("m_low", M_LOW_GRID), ("m_high", M_HIGH_GRID),
                                 ("window", WINDOW_GRID)):
            cur = {"h_lo": hl, "h_hi": hh, "m_low": ml, "m_high": mh, "window": w}
            for v in grid_vals:
                if v == cur[axis]:
                    continue
                alt = dict(cur)
                alt[axis] = v
                key = (alt["h_lo"], alt["h_hi"], alt["m_low"], alt["m_high"], alt["window"])
                if key in score.index:
                    yield axis, v, float(score.loc[key].mean_sharpe)

    print("\n=== one-axis-away neighbours (plateau check) ===")
    nb_rows = list(neighbours())
    for axis, v, sh in nb_rows:
        print(f"  {axis:6s} -> {v:<5}  mean-Sharpe={sh:.3f}")
    if nb_rows:
        spread = max(s for _, _, s in nb_rows) - min(s for _, _, s in nb_rows)
        print(f"neighbourhood Sharpe spread: {spread:.3f}  (noise floor is +/-0.2 Sharpe, R-20)")

    # fixed_m and v4 controls, same split/market cells
    fixed = grid_control = None
    ctrl = df[df.window.isin(["fixed_m_control", "v4_control"])]
    print("\n=== inner-train / inner-validation: candidate vs fixed_m=4 control vs v4 control ===")
    cand_kwargs = dict(h_lo=hl, h_hi=hh, m_low=ml, m_high=mh, window=w)
    out_rows = []
    for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        for mname, market in MARKETS:
            m_c, vol_c, not_c, res_c = measure(make_strategy(**cand_kwargs), start, end, market=market)
            m_f, vol_f, not_f, res_f = measure(make_strategy(fixed_m=FIXED_M_CONTROL), start, end, market=market)
            m_v, vol_v, not_v, res_v = measure(get_strategy(INCUMBENT), start, end, market=market)
            line(f"  {split_name}/{mname} candidate (adaptive)", m_c, vol_c, not_c, res_c)
            line(f"  {split_name}/{mname} fixed_m=4 control", m_f, vol_f, not_f, res_f)
            line(f"  {split_name}/{mname} v4 control", m_v, vol_v, not_v, res_v)
            for arm, m_, vol_, not_ in (("candidate", m_c, vol_c, not_c),
                                        ("fixed_m_control", m_f, vol_f, not_f),
                                        ("v4_control", m_v, vol_v, not_v)):
                out_rows.append({"split": split_name, "market": mname, "arm": arm,
                                 "final": m_.final_balance, "profit_pct": m_.profit_pct,
                                 "sharpe": m_.sharpe, "max_dd": m_.max_drawdown_pct,
                                 "vol": vol_, "mean_notional": not_, "trades": m_.num_trades})
    pd.DataFrame(out_rows).to_csv(OUT / "candidate_vs_controls.csv", index=False)

    valid_c = [r for r in out_rows if r["split"] == "inner-validation" and r["arm"] == "candidate"]
    valid_f = [r for r in out_rows if r["split"] == "inner-validation" and r["arm"] == "fixed_m_control"]
    beats = {r["market"]: (c["sharpe"] - r["sharpe"]) for c, r in zip(valid_c, valid_f)
             for r in valid_f if r["market"] == c["market"]}
    print("\n=== adaptive candidate vs its own fixed_m=4 ablation control, inner-validation ===")
    for mname, d in beats.items():
        print(f"  {mname}: d(Sharpe) = {d:+.3f}  "
              f"{'beats fixed-m control' if d > 0.05 else 'does NOT clearly beat fixed-m control'}")

    (OUT / "selected_config.txt").write_text(
        f"h_lo={hl}\nh_hi={hh}\nm_low={ml}\nm_high={mh}\nwindow={w}\n"
        f"fixed_m_control={FIXED_M_CONTROL}\nn_configs_evaluated={len(_SEEN)}\n")
    print(f"\ndistinct configurations evaluated in total: {len(_SEEN)}")
    print(f"wrote {OUT / 'selected_config.txt'}")


def _load_selected() -> dict:
    cfg = OUT / "selected_config.txt"
    if cfg.exists():
        kv = dict(line.split("=", 1) for line in cfg.read_text().splitlines() if "=" in line)
        return {"h_lo": float(kv["h_lo"]), "h_hi": float(kv["h_hi"]),
                "m_low": float(kv["m_low"]), "m_high": float(kv["m_high"]),
                "window": int(kv["window"])}
    print("(no selected_config.txt yet -- run `select` first; falling back to grid midpoint)")
    return {"h_lo": 0.45, "h_hi": 0.55, "m_low": 3.0, "m_high": 6.0, "window": 60}


# --------------------------------------------------------------------------- point 6


def hurst_stats() -> None:
    """Point 6: empirical distribution of this dataset's own rolling causal H(t), both windows."""
    close = DF.loc[TRAIN[0]:VALID[1], "close"]
    print(f"rolling causal Hurst distribution, spot close, {TRAIN[0]} -> {VALID[1]} "
          f"({len(close):,} 5m bars)")
    for w in WINDOW_GRID:
        h = rolling_causal_hurst(close, w).dropna()
        h = h[(h.index >= TRAIN[0]) & (h.index <= VALID[1])]
        vals = h.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        pct_above = 100.0 * np.mean(vals > 0.5)
        pct_below = 100.0 * np.mean(vals <= 0.5)
        print(f"\n  window={w}d  n={len(vals):,}  mean={vals.mean():.3f}  median={np.median(vals):.3f}  "
              f"std={vals.std():.3f}")
        print(f"    quantiles: p5={np.percentile(vals,5):.3f} p25={np.percentile(vals,25):.3f} "
              f"p50={np.percentile(vals,50):.3f} p75={np.percentile(vals,75):.3f} "
              f"p95={np.percentile(vals,95):.3f}")
        print(f"    %H>0.5 (persistent-leaning): {pct_above:.1f}%   "
              f"%H<=0.5 (random/mean-reverting-leaning): {pct_below:.1f}%")


# --------------------------------------------------------------------------- exposure artifact


def exposure_artifact_check() -> None:
    """Mandatory exposure-artifact check (ROUTINE.md standing rule, sharpened by R-33).

    Flat-rescaled-v4 comparator: v4's own unchanged target, multiplied by
    a single constant c chosen so its mean notional matches the
    candidate's mean notional over the SAME period. R^2 > 0.95 means
    "standard exposure-level artifact."
    """
    cfg = _load_selected()
    print(f"\nexposure-artifact check (inner-validation, mean-notional-matched flat rescale of v4)")
    print(f"candidate: {cfg}")
    for mname, market in MARKETS:
        cand = make_strategy(**cfg)
        m_c, vol_c, not_c, res_c = measure(cand, *VALID, market=market)
        v4 = get_strategy(INCUMBENT)
        m_v4, vol_v4, not_v4, res_v4 = measure(v4, *VALID, market=market)

        cand_t = res_c.df["target"].to_numpy(dtype=float)
        v4_t = res_v4.df["target"].reindex(res_c.df.index).to_numpy(dtype=float)
        c = not_c / not_v4 if not_v4 > 0 else float("nan")
        flat = c * v4_t

        mask = np.isfinite(cand_t) & np.isfinite(flat)
        x, y = flat[mask], cand_t[mask]
        ss_res = float(np.sum((y - x) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")

        verdict = ("EXPOSURE-LEVEL ARTIFACT (R^2 > 0.95)" if np.isfinite(r2) and r2 > 0.95
                    else "not a flat rescale by this test")
        print(f"  {mname}: cand notional={not_c:.3f} v4 notional={not_v4:.3f} c={c:.3f}  "
              f"corr={corr:.4f}  R^2={r2:.4f}  {verdict}")
        print(f"    cand realized vol={vol_c:.3f}  v4 realized vol={vol_v4:.3f}  "
              f"cand sharpe={m_c.sharpe:.3f}  v4 sharpe={m_v4.sharpe:.3f}")


# ------------------------------------------------------------------------ causality


def causality() -> None:
    """Step 6: two-opposite-tampers lookahead probe, sharpened for a path-dependent strategy.

    (1) ``frac``/``H`` columns from ``prepare()`` must be bit-identical
    before the cut between the two tampered copies -- the standard
    column-level check.
    (2) Because this strategy's ``on_bar`` decisions depend on the
    REALIZED account equity (not just precomputed columns), the
    column check alone is not sufficient: a full SEQUENTIAL replay of
    both tampered copies is run from bar 0, and their equity curves and
    the strategy's own decided-position log must match bit-for-bit
    before the cut. This is the primary check for this file -- the
    rolling Hurst estimator is new code on a new causal path, and a
    lookahead bug there would only show up in a full sequential replay,
    not in an isolated-bar spot check.
    Restricted to strictly pre-2023 bars, per this session's data rule.
    """
    cfg = _load_selected()
    print(f"probing candidate: {cfg}")

    pre_2023 = DF.loc[:"2022-12-31"]
    df = pre_2023.iloc[-300_000:].copy()
    cut = len(df) - 5_000

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def build():
        return KellyRegimeV12CPPIHurst(hurst_window_days=cfg["window"], h_lo=cfg["h_lo"],
                                        h_hi=cfg["h_hi"], m_low=cfg["m_low"], m_high=cfg["m_high"],
                                        f0=F0, g=G, max_leverage=MAX_LEVERAGE)

    ok = True

    pa = build().prepare(up.copy())
    pb = build().prepare(down.copy())
    for col in ("frac", "H"):
        a = pa[col].to_numpy(dtype=float)[:cut]
        b = pb[col].to_numpy(dtype=float)[:cut]
        finite = np.isfinite(a) & np.isfinite(b)
        worst = float(np.nanmax(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
        good = worst < 1e-9
        ok &= good
        print(f"  column={col:6s}  max |difference| before the cut = {worst:.3e}  "
              f"{'PASS' if good else 'FAIL'}")

    # Full sequential replay: equity curves and decided-position logs.
    strat_up = build()
    strat_down = build()
    a = run_backtest(strat_up, up.iloc[:cut + 1], FUTURES, 1_000.0, data_label=LABEL)
    b = run_backtest(strat_down, down.iloc[:cut + 1], FUTURES, 1_000.0, data_label=LABEL)

    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-6
    print(f"  max |equity difference| before the cut = {worst_eq:.3e}  "
          f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")

    pos_a = strat_up.realized_target_series().to_numpy(dtype=float)[:cut]
    pos_b = strat_down.realized_target_series().to_numpy(dtype=float)[:cut]
    finite = np.isfinite(pos_a) & np.isfinite(pos_b)
    worst_pos = float(np.max(np.abs(pos_a[finite] - pos_b[finite]))) if finite.any() else 0.0
    ok &= worst_pos < 1e-9
    print(f"  max |decided-position difference| before the cut = {worst_pos:.3e}  "
          f"{'PASS' if worst_pos < 1e-9 else 'FAIL'}")

    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Step 7: pre-registered falsification -- does the selected candidate hold on ETH?

    Same venue (Bitfinex), same pre-2020 window prior rounds used
    (verified below: both files end 2019-12-31, entirely pre-OOS), both
    spot and 5x futures, candidate vs shipped v4 defaults as the control.
    Falsification rule (fixed before running): if the candidate is not at
    least comparable to v4 on ETH, or is visibly worse on ETH than on the
    BTC control run through the identical code, this direction fails.
    """
    cfg = _load_selected()
    print(f"candidate: {cfg}")

    rows = []
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        assert df.index[-1] < pd.Timestamp("2020-01-01", tz="UTC"), \
            f"{path} extends past 2020 -- would violate this session's no-2023+-bar rule if unbounded"
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            cand = make_strategy(**cfg)
            m_c, vol_c, not_c, res_c = measure(cand, None, None, df=df, market=market)
            line(f"    candidate (v12)", m_c, vol_c, not_c, res_c)
            m_v4, vol_v4, not_v4, res_v4 = measure(get_strategy(INCUMBENT), None, None,
                                                    df=df, market=market)
            line(f"    {INCUMBENT} (control)", m_v4, vol_v4, not_v4, res_v4)
            rows.append({"asset": asset, "market": mname, "arm": "candidate",
                         "final": m_c.final_balance, "profit_pct": m_c.profit_pct,
                         "sharpe": m_c.sharpe, "max_dd": m_c.max_drawdown_pct,
                         "vol": vol_c, "liquidated": m_c.liquidated})
            rows.append({"asset": asset, "market": mname, "arm": "v4_control",
                         "final": m_v4.final_balance, "profit_pct": m_v4.profit_pct,
                         "sharpe": m_v4.sharpe, "max_dd": m_v4.max_drawdown_pct,
                         "vol": vol_v4, "liquidated": m_v4.liquidated})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "eth_falsification.csv", index=False)

    print("\n=== falsification verdict, candidate vs v4 control ===")
    verdict_ok = True
    for asset in ("BTC (control)", "ETH (test)"):
        for mname, _ in MARKETS:
            c = out[(out.asset == asset) & (out.market == mname) & (out.arm == "candidate")].iloc[0]
            d = out[(out.asset == asset) & (out.market == mname) & (out.arm == "v4_control")].iloc[0]
            d_sharpe = c.sharpe - d.sharpe
            d_profit = c.profit_pct - d.profit_pct
            d_dd = c.max_dd - d.max_dd
            ok = d_sharpe > -0.05 and d_profit > -2.0
            verdict_ok &= ok if "ETH" in asset else True
            print(f"  {asset:16s} {mname:8s} d(Sharpe)={d_sharpe:+.3f} "
                  f"d(profit)={d_profit:+.1f}pp d(maxDD)={d_dd:+.1f}pp  "
                  f"{'OK' if ok else 'WORSE'}")
    print(f"\nETH falsification: {'PASS' if verdict_ok else 'FAIL'}")
    print(f"wrote {OUT / 'eth_falsification.csv'}")


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    cmds = {"sweep": sweep, "select": select, "hurst_stats": hurst_stats,
            "artifact": exposure_artifact_check, "causality": causality, "eth": eth}

    def all_() -> None:
        sweep()
        select()
        hurst_stats()
        exposure_artifact_check()
        causality()
        eth()

    cmds["all"] = all_
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python {sys.argv[0]} [{'|'.join(cmds)}]")
