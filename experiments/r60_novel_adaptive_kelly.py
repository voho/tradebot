"""R-60 NOVEL BRANCH: a from-scratch, variance-ratio-blended trend/reversion
Kelly sizer, replacing `kelly_regime_v4`'s discrete 3-anchor trend vote
entirely, tested on the mandate laid out for this round.

Not registered: lives under ``experiments/`` per ROUTINE.md step 5.
``AdaptiveKellyVR`` below inherits directly from ``tradebot.strategy.Strategy``
(NOT from any ``kelly_regime*`` class) and is never passed through
``tradebot.registry`` -- it is constructed directly in this module's own
test/measurement functions, exactly like R-59's novel branch constructed
``KellyRegimeRelativeVol`` directly.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **SIZE** (the sizing/directional-signal mechanism
itself) and, more fundamentally than any prior SIZE-axis round, **INFO** --
every trend-vote-tuning branch (R-34, R-37, R-38, R-40, R-41, R-42, R-43,
R-45, R-46, R-53-R-56, R-59; nineteen branches, twelve rounds, LEDGER.md
section A/C) kept v4's binary/latched trend assumption fixed and retuned a
parameter around it. R-59 (closing B-25) found that neither a per-asset
scale calibration nor a self-normalizing relative-vol scale restores the
panel drawdown property, and concluded the panel's advantage for a
matched-exposure hold "looks like a buy-the-dip / mean-reversion effect
these instruments reward, not a sizing-constant problem" -- and explicitly
recommended trying a genuinely different strategy family, not a twentieth
tweak to v4's vote mechanism.

This round is that mandate: build one continuous directional signal that
BLENDS trend-following and mean-reversion by a data-driven confidence
weight (the variance-ratio test), rather than assuming trend-following is
always the right regime and only gating its strength. Grounding:

  - Lo & MacKinlay (1988, Rev. Fin. Studies, "Stock Market Prices Do Not
    Follow Random Walks") -- VR(q) = Var(r^(q)) / (q . Var(r^(1))). VR>1:
    positive serial correlation (trend-following regime). VR<1: negative
    serial correlation (mean-reverting regime). VR=1: random walk.
  - Lo (2004, J. Portfolio Management, "The Adaptive Markets Hypothesis")
    -- markets cycle between these regimes rather than sitting permanently
    in one; a sizer that assumes one regime forever is mis-specified by
    construction, which is the honest diagnosis of every prior branch in
    this ledger's SIZE-axis history.
  - Beluska & Vojtko (2024, Quantpedia/SSRN, "Revisiting Trend-following
    and Mean-reversion Strategies in Bitcoin") -- a strategy COMBINING
    trend and mean-reversion signals on BTC outperforms either alone,
    2015-2024. This round tests whether that combination, expressed as a
    continuous VR-weighted blend rather than a discrete switch, also fixes
    the one property this project has actually established (matched-
    exposure drawdown) on the panel R-57 showed it fails on.

**Not a duplicate of anything in LEDGER.md section C.** Every prior
SIZE-axis entry retuned v4's existing vote/scale mechanism; this strategy
shares no code with `kelly_regime*` and computes its directional signal
from scratch. Not a duplicate of R-59's own novel branch (self-normalizing
relative-vol SCALE, vote unchanged) -- this round changes the vote itself,
not the scale term multiplying it.

**Simulable here**: yes. 5m OHLCV bars, bar-close signals (`prepare()`
columns), next-open fills (unchanged engine), no order book, no queue
model. The variance-ratio aggregation from 5m to daily is itself new
machinery this round adds (nothing before this has aggregated to daily
bars inside a strategy), built causally with an explicit shift, not
proxied from anything unavailable in OHLCV.

**What would make this fail, named now, before any strategy number was
read:** the exact pre-registered decision rule in section 4 below. If F1
fails (the strategy loses to v4 on the pre-2020 BTC control by more than
the +/-0.2 Sharpe noise floor -- exactly the signature that killed R-40,
R-41, R-42, R-43, R-45, R-46 before ETH was even read), or F2 regresses
v4's established ETH matched-exposure property, or F3 scores <5/6 on the
panel, this is NEGATIVE and the BTC holdout is never read.

=====================================================================
STEP 2 -- SOURCES READ, WHAT THEY CLAIM, ON WHAT DATA, AT WHAT COST
=====================================================================

- Lo & MacKinlay (1988): VR test on weekly NYSE/AMEX equal- and
  value-weighted index returns, 1962-1985; no trading cost or strategy is
  proposed, it is a statistical rejection of the random walk. Used here
  only for its test statistic's definition and regime interpretation, not
  for any claimed tradeable edge.
- Lo (2004): a framework paper (Adaptive Markets Hypothesis), no backtest,
  no cost assumption, no instrument count -- cited for the "regime cycling"
  interpretation of VR departures from 1, not as an empirical result to
  replicate.
- Beluska & Vojtko (2024): BTC only, 2015-2024, a combined trend+
  mean-reversion long/short(?)/long-flat overlay; exact cost assumption
  and instrument count not re-derived here (SSRN preprint, not re-run) --
  cited for the qualitative claim that combining the two families beats
  either alone on BTC, which is the mechanism this round tests on this
  project's own data/cost/instrument set rather than trusting the
  citation's own reported numbers.

=====================================================================
THE MECHANISM -- frozen BEFORE any strategy number was read
=====================================================================

Five pieces, in the order the mandate specifies:

1. VARIANCE RATIO. 5m closes are aggregated to daily via a causal
   ``resample("1D").last()`` (a day's own close is only known once its
   last 5m bar has closed). VR(q) uses the SIMPLE ratio form given in the
   mandate, not Lo-MacKinlay's bias-corrected overlapping-sum estimator:

       VR(q) = Var(rq) / (q * Var(r1))

   where ``r1`` is the 1-day log return and ``rq`` is the (overlapping)
   q-day log return, both rolling over a window of ``VR_WINDOW_DAYS``
   calendar days. Frozen constants:

     VR_WINDOW_DAYS = 90   -- one quarter: enough independent q-blocks
                              (90/5 = 18) for a stable ratio while still
                              adaptive within a single quarter; the same
                              order of magnitude as v4's own longest
                              anchor (80 days), so the VR estimate updates
                              on a comparable timescale to the trend
                              signal it is meant to arbitrate between.
     Q_DAYS           = 5  -- a trading-week-scale horizon, the classic
                              choice in the VR literature (Lo-MacKinlay's
                              own weekly q), and deliberately the SAME
                              horizon as the reversion component's own
                              short EMA below, so "is q-day serial
                              correlation positive or negative" and "how
                              extended is price over q days" are measured
                              on one consistent clock.

   The resulting daily VR series is reindexed onto the 5m bar grid with an
   explicit ONE-CALENDAR-DAY SHIFT (``vr_daily.index += Timedelta(days=1)``
   before an as-of/ffill reindex): a day's VR value, which needs that
   day's own close to compute, is not available until the day is over, so
   it first applies to bars starting at the NEXT day's first bar, never
   during the day whose data it was computed from. This is the "no
   partial-day leak" guarantee the mandate requires.

2. TREND COMPONENT. Reuses v4's own anchor ladder (20/40/80 days) but
   replaces the discrete >/< band-and-latch VOTE with a continuous,
   bounded signal: for each anchor horizon h,

       dev_h = close / anchor_h - 1
       z_h   = dev_h / rolling_std(dev_h, window=h)     (self-normalizing:
                                                          no new free
                                                          parameter, the
                                                          same horizon
                                                          normalizes its
                                                          own deviation)
       trend_h = tanh(z_h)                              (bounded to [-1,1])

   trend_component = mean(trend_h for h in (20, 40, 80))

3. REVERSION COMPONENT. The negative of a normalized short-horizon price
   extension:

       short_ema  = EMA(close, span = SHORT_HORIZON_DAYS days)
       short_std  = rolling_std(close, window = SHORT_HORIZON_DAYS days)
       ext        = (close - short_ema) / short_std
       reversion_component = -tanh(ext)

   SHORT_HORIZON_DAYS = 5 -- frozen equal to Q_DAYS (see above: one
   consistent short clock for "is price overextended" and "is the q-day
   VR horizon trending or reverting").

4. BLEND. w_trend = clip((VR - VR_LO) / (VR_HI - VR_LO), 0, 1):

       VR_LO = 0.8, VR_HI = 1.2  -- a symmetric +/-20% band around the
       random-walk null VR=1 (the Lo-MacKinlay boundary itself), chosen
       as a round, pre-registered number before any result was read. Below
       0.8: VR is unambiguously in mean-reversion territory (fully
       reversion, w_trend=0). Above 1.2: unambiguously trending (fully
       trend, w_trend=1). Exactly at the null (VR=1): a 50/50 blend, which
       is the honest "no information from this test" case.

       signal = w_trend * trend_component + (1 - w_trend) * reversion_component

   ``signal`` is a convex combination of two already-bounded [-1,1] terms,
   so it is itself in [-1,1] by construction; clipped defensively anyway.

5. SIZING. Unchanged fractional-Kelly, vol-targeted machinery, IDENTICAL
   constants to v4's own shipped default (not re-tuned -- this round tests
   the directional signal, not a new scale; re-tuning both at once would
   confound which change did what):

       target_vol = 0.55, max_leverage = 2.0, vol_span = 8 days,
       deadband = 0.10

       desired_exposure = signal * min(target_vol / realized_vol, max_leverage)

   Unlike v4, ``signal`` (and therefore desired_exposure) can go negative.
   ``ctx.order_notional`` is used exactly as `kelly_regime.on_bar` uses it;
   `PaperBroker._execute_target` already clamps the resulting fraction to
   ``[0, 1]`` on spot (``allow_short=False``) and to ``[-1, 1]`` on futures
   (``allow_short=True``) -- so a negative signal is a genuine short on
   futures and is clamped flat (never a naked short) on spot, with no
   strategy-side special-casing required; this is the same clamp every
   other strategy in this repo already relies on.

Total new fitted parameters vs. the mandate's own suggested defaults: ZERO
beyond the frozen constants named above (VR_WINDOW_DAYS, Q_DAYS, VR_LO,
VR_HI, SHORT_HORIZON_DAYS); target_vol/max_leverage/vol_span/deadband are
v4's own unchanged values, not swept.

=====================================================================
DATA AND HOLDOUT DISCIPLINE
=====================================================================

BTC is loaded ONLY through ``load_btc_pre2023()`` below, which truncates
to ``<= 2022-12-31`` at the moment of loading -- every function in this
module that touches BTC receives an already-truncated frame, so there is
no code path anywhere in this file, gated or otherwise, that can read a
2023+ BTC bar. ETH (``load_coinbase_spot``) and the 6-asset panel
(``experiments.r57_cross_asset_panel``'s loaders) are read at full range,
per this project's established convention (R-47, R-57, R-59: nothing was
ever fitted on them, so reading them costs 0 holdout consultations).

=====================================================================
PRE-REGISTERED FALSIFICATION BATTERY -- frozen before any strategy number
was read
=====================================================================

F1 -- BTC PRE-2020 CONTROL. 2017-01-01 -> 2019-12-31 (inside inner-train,
  no 2023+ bar). AdaptiveKellyVR's Sharpe must not underperform frozen
  `kelly_regime_v4`'s by more than the project's +/-0.2 Sharpe noise floor,
  on BOTH spot (0.10%) and futures 5x (0.05%, no funding). This is the
  exact failure signature that killed six prior branches before ETH was
  even read (R-40, R-41, R-42, R-43, R-45, R-46) -- checked first.

F2 -- ETH REPLICATION. Coinbase ETH, full window (2020-01-01 -> last bar,
  R-47's own convention), spot @0.10%. AdaptiveKellyVR and frozen v4 are
  each matched, on this SAME window, against a `ConstantExposureHold` at
  THEIR OWN mean notional (the R-33/R-57 matched-exposure convention).
  AdaptiveKellyVR's matched-exposure drawdown ADVANTAGE (candidate dd -
  matched-hold dd; negative = candidate draws down less, i.e. better) must
  not be worse than v4's own matched-exposure advantage, measured on the
  identical window, by more than F2_REGRESSION_TOLERANCE_PP = 5.0
  percentage points -- the same tolerance R-59 pre-registered for its own
  analogous regression check.

F3 -- PRIMARY: 6-ASSET PANEL D1. BCH/LTC/ETC/DASH/LINK/XTZ, FULL window
  (2020-04-01 -> last bar, R-57's own panel/window convention), spot
  @0.10% (`MarketSpec.spot()`). Count assets where AdaptiveKellyVR's OWN
  max drawdown is strictly lower than a `ConstantExposureHold` matched to
  AdaptiveKellyVR's OWN mean notional over the same window. v4 alone
  scores 0/6 here (R-57); R-59's two SIZE-axis fixes both also scored 0/6.
  Paired-bootstrap interval per asset: 30-day mean block, 2000 resamples,
  seed 7 (`BOOT_KW`, identical to `r57_cross_asset_panel.py`).
  Gate: >= 5/6 required to proceed (F3_GATE_K = 5).

F4 -- CONTEXT (not a gate). Same panel, 0.40% Bitstamp entry-tier fee:
  does AdaptiveKellyVR beat `buy_and_hold` in >= 5/6? PREDICTION, recorded
  now: FAILS, per this project's unbroken record on this exact check
  (R-13, R-47, R-57's D2, R-59's D4) -- no strategy in this project's
  history has ever cleared this bar, and nothing about this round targets
  the return-vs-cost problem, only the sizing/directional signal.

=====================================================================
PRE-REGISTERED DECISION RULE (frozen here, BEFORE F1-F4 are run; not moved
after seeing any number)
=====================================================================

READY FOR HOLDOUT CONSULTATION only if ALL of:
    F1 passes (Sharpe gap <= 0.2 in AdaptiveKellyVR's favour-or-better, OR
               a drawdown/tail improvement per the project's standing
               "risk property, not return property" convention)
    AND F2 does not regress (advantage no worse than v4's own by > 5.0pp)
    AND F3 >= 5/6

Otherwise: NEGATIVE. Report the specific failing test and numbers, and DO
NOT read any BTC bar dated 2023-01-01 or later, under any circumstance --
enforced structurally by ``load_btc_pre2023()``, not merely by convention.

Every backtest is counted in ``CONFIG_COUNT`` (module-level global,
incremented once per ``measure()`` call) and printed at the end.

Usage::

    python experiments/r60_novel_adaptive_kelly.py causality
    python experiments/r60_novel_adaptive_kelly.py selfcheck
    python experiments/r60_novel_adaptive_kelly.py run
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.matched_hold import ConstantExposureHold, mean_notional  # noqa: E402
from experiments.r57_cross_asset_panel import (  # noqa: E402
    binomial_tail,
    load_candidates,
    select_panel,
)
from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.data import load_coinbase_spot, load_dataset  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns,
    max_drawdown_from_returns,
    paired_bootstrap,
    total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "reports" / "r60_novel"

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# ---- frozen mechanism constants (see module docstring section "THE MECHANISM") ----
VR_WINDOW_DAYS = 90
Q_DAYS = 5
VR_LO, VR_HI = 0.8, 1.2
ANCHOR_HORIZONS_DAYS = (20, 40, 80)   # reused from kelly_regime_v4, unchanged
SHORT_HORIZON_DAYS = 5                # == Q_DAYS, deliberately (see docstring)

# ---- reused, unchanged v4 sizing constants (not re-tuned this round) ----
TARGET_VOL = 0.55
MAX_LEVERAGE = 2.0
VOL_SPAN_DAYS = 8
DEADBAND = 0.10

INCUMBENT = "kelly_regime_v4"
BOOT_KW = dict(mean_block=30.0, n_boot=2_000, seed=7)

# ---- pre-registered decision thresholds ----
F1_NOISE_FLOOR = 0.2           # Sharpe
F2_REGRESSION_TOLERANCE_PP = 5.0
F3_GATE_K = 5                  # of 6

BTC_CONTROL_END = "2022-12-31"  # BTC is NEVER loaded past this in this file

CONFIG_COUNT = 0


# ============================================================== the strategy


class AdaptiveKellyVR(Strategy):
    """Unified adaptive-Kelly sizer: VR-blended trend/reversion signal.

    Fractional-Kelly, vol-targeted sizing (v4's own unchanged mechanism),
    but the directional input is a single continuous signal in [-1, 1]
    that blends a continuous trend-following term with a continuous
    mean-reversion term, weighted by a rolling variance-ratio confidence
    (Lo & MacKinlay 1988; Lo 2004 Adaptive Markets Hypothesis; Beluska &
    Vojtko 2024 on combining both families on BTC). See the module
    docstring for the full derivation and every frozen constant.

    Inherits directly from ``tradebot.strategy.Strategy`` -- NOT from any
    ``kelly_regime*`` class. Not registered; constructed directly.
    """

    name = "adaptive_kelly_vr"
    # VR needs >= VR_WINDOW_DAYS + Q_DAYS days of daily history; the
    # 80-day trend anchor and 8-day vol span are both shorter. Generous
    # margin, same convention as every kelly_regime* warmup.
    warmup = (VR_WINDOW_DAYS + Q_DAYS + 20) * BARS_PER_DAY + 10

    def __init__(self,
                 vr_window_days: int = VR_WINDOW_DAYS,
                 q_days: int = Q_DAYS,
                 vr_lo: float = VR_LO,
                 vr_hi: float = VR_HI,
                 horizons: tuple[int, ...] = ANCHOR_HORIZONS_DAYS,
                 short_horizon_days: int = SHORT_HORIZON_DAYS,
                 target_vol: float = TARGET_VOL,
                 max_leverage: float = MAX_LEVERAGE,
                 vol_span_days: int = VOL_SPAN_DAYS,
                 deadband: float = DEADBAND) -> None:
        self.vr_window_days = vr_window_days
        self.q_days = q_days
        self.vr_lo = vr_lo
        self.vr_hi = vr_hi
        self.horizons = horizons
        self.short_horizon_days = short_horizon_days
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span_days * BARS_PER_DAY
        self.deadband = deadband
        # Diagnostic-only side channels (never read by prepare()/on_bar()):
        # populated at the end of prepare() so selfcheck()/tests can
        # inspect the exact series the strategy used, without a second,
        # potentially-divergent reimplementation.
        self._last_vr: pd.Series | None = None
        self._last_w_trend: pd.Series | None = None
        self._last_signal: pd.Series | None = None

    # -------------------------------------------------------------- prepare

    def _variance_ratio_5m(self, df: pd.DataFrame) -> np.ndarray:
        """Causal rolling VR(q), computed on daily-aggregated closes,
        reindexed onto the 5m grid with a one-day shift (no partial-day
        leak: a day's own VR value is unusable until that day is over)."""
        daily_close = df["close"].resample("1D").last()
        r1 = np.log(daily_close).diff()
        rq = np.log(daily_close).diff(self.q_days)
        var1 = r1.rolling(self.vr_window_days).var()
        varq = rq.rolling(self.vr_window_days).var()
        with np.errstate(divide="ignore", invalid="ignore"):
            vr_daily = varq / (self.q_days * var1)
        # The value computed "as of" day D's close is not knowable until
        # day D has ended -- shift the label forward one full day so it
        # first applies at day D+1's first bar, never during day D itself.
        vr_daily = vr_daily.copy()
        vr_daily.index = vr_daily.index + pd.Timedelta(days=1)
        vr_5m = vr_daily.reindex(df.index, method="ffill")
        return vr_5m.to_numpy(dtype=float)

    def _trend_component(self, close: pd.Series) -> np.ndarray:
        terms = []
        for h in self.horizons:
            window = h * BARS_PER_DAY
            anchor = close.rolling(window).mean()
            dev = close / anchor - 1.0
            dev_std = dev.rolling(window).std()
            with np.errstate(divide="ignore", invalid="ignore"):
                z = dev / dev_std
            terms.append(np.tanh(z))
        return pd.concat(terms, axis=1).mean(axis=1).to_numpy(dtype=float)

    def _reversion_component(self, close: pd.Series) -> np.ndarray:
        span = self.short_horizon_days * BARS_PER_DAY
        short_ema = close.ewm(span=span, min_periods=BARS_PER_DAY).mean()
        short_std = close.rolling(span).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            ext = (close - short_ema) / short_std
        return (-np.tanh(ext)).to_numpy(dtype=float)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        vr = self._variance_ratio_5m(df)
        with np.errstate(invalid="ignore"):
            w_trend = np.clip((vr - self.vr_lo) / (self.vr_hi - self.vr_lo), 0.0, 1.0)
        w_trend = np.where(np.isfinite(vr), w_trend, 0.5)  # neutral pre-warmup default

        trend = np.nan_to_num(self._trend_component(close), nan=0.0)
        reversion = np.nan_to_num(self._reversion_component(close), nan=0.0)

        signal = w_trend * trend + (1.0 - w_trend) * reversion
        signal = np.clip(signal, -1.0, 1.0)

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            v = vol[i]
            scale = min(self.target_vol / v, self.max_leverage) if np.isfinite(v) and v > 0 else 0.0
            desired = signal[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        # diagnostic-only, never read inside prepare()/on_bar()
        self._last_vr = pd.Series(vr, index=df.index)
        self._last_w_trend = pd.Series(w_trend, index=df.index)
        self._last_signal = pd.Series(signal, index=df.index)
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # same clamp convention as kelly_regime.on_bar


def make_candidate() -> AdaptiveKellyVR:
    """Fresh instance per backtest -- no shared mutable state across runs."""
    return AdaptiveKellyVR()


# ==================================================================== helpers


def load_btc_pre2023():
    """The ONLY way BTC is ever loaded in this module. Truncates at load
    time, before any other code runs, so no function below -- gated or
    not -- can ever see a 2023+ BTC bar."""
    df, label = load_dataset(DATA_DIR, "spot")
    return df.loc[:BTC_CONTROL_END], label


def measure(strategy, df, start, end, market):
    """One backtest. Every call is counted."""
    global CONFIG_COUNT
    CONFIG_COUNT += 1
    result = run_period(strategy, df, start, end, market=market, start_balance=1_000.0)
    return result, compute_metrics(result)


SPOT_BASE = MarketSpec.spot()                  # 0.10% taker
SPOT_REAL = MarketSpec.spot(fee_rate=0.004)    # 0.40% Bitstamp entry tier
FUT_BASE = MarketSpec.futures(leverage=5.0)    # 0.05% taker, no funding


# =================================================================== causality


def _tamper_probe(make_strategy, df: pd.DataFrame, market: MarketSpec) -> bool:
    """The test_causality_strict.py tamper methodology, adapted: perturb
    every bar after a cut point in two opposite directions and confirm
    every decision AT OR BEFORE the cut is identical either way."""
    tail = df.iloc[-60_000:].copy() if len(df) > 60_000 else df.copy()
    cut = max(len(tail) - 5_000, len(tail) // 2)
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20) if cut - k >= 0]
    up, down = tail.copy(), tail.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        s = make_strategy()
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=market, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    return all(x == y for x, y in zip(decisions(up), decisions(down)))


def cmd_causality() -> bool:
    print("=" * 100)
    print("CAUSALITY TAMPER PROBE — AdaptiveKellyVR")
    print("=" * 100)
    market = MarketSpec.futures(leverage=5.0)
    btc_df, _ = load_btc_pre2023()
    eth_df = load_coinbase_spot(DATA_DIR, "ETH")
    panel = select_panel(load_candidates())
    probes = [("BTC(pre-2023)", btc_df), ("ETH", eth_df)] + [(a.ticker, a.df) for a in panel[:2]]

    all_ok = True
    for ticker, df in probes:
        if df is None:
            continue
        ok = _tamper_probe(make_candidate, df, market)
        all_ok = all_ok and ok
        print(f"  {ticker:14s} decisions identical under opposite post-cut tampers: "
              f"{'PASS' if ok else 'FAIL'}")
    return all_ok


# ============================================================== self-check


def cmd_selfcheck() -> None:
    """Sanity numbers on the mechanism's own intermediate series: does VR
    actually move around 1.0 and take both regimes, is w_trend using its
    full [0,1] range, is the blended signal actually bounded."""
    print("=" * 100)
    print("SELF-CONSISTENCY CHECK — VR / w_trend / signal on BTC pre-2020")
    print("=" * 100)
    btc_df, _ = load_btc_pre2023()
    cand = make_candidate()
    cand.prepare(btc_df.copy())
    window = ("2017-06-01", "2019-12-31")  # skip the cold-start warmup region
    mask = (cand._last_vr.index >= pd.Timestamp(window[0], tz="UTC")) & \
           (cand._last_vr.index <= pd.Timestamp(window[1], tz="UTC"))
    vr = cand._last_vr[mask].dropna()
    wt = cand._last_w_trend[mask].dropna()
    sig = cand._last_signal[mask].dropna()
    print(f"  VR:       mean={vr.mean():.3f} std={vr.std():.3f} "
          f"min={vr.min():.3f} max={vr.max():.3f} "
          f"frac<0.8={float((vr < 0.8).mean()):.1%} frac>1.2={float((vr > 1.2).mean()):.1%}")
    print(f"  w_trend:  mean={wt.mean():.3f} frac==0={float((wt == 0).mean()):.1%} "
          f"frac==1={float((wt == 1).mean()):.1%}")
    print(f"  signal:   mean={sig.mean():.3f} std={sig.std():.3f} "
          f"min={sig.min():.3f} max={sig.max():.3f}")
    assert sig.min() >= -1.0 - 1e-9 and sig.max() <= 1.0 + 1e-9, \
        "signal escaped its pre-registered [-1,1] bound"


# ======================================================================= F1


def cmd_f1() -> dict:
    print("\n" + "=" * 100)
    print("F1 — BTC PRE-2020 CONTROL, 2017-01-01..2019-12-31, spot @0.10% + futures 5x")
    print("=" * 100)
    btc_df, _ = load_btc_pre2023()
    window = ("2017-01-01", "2019-12-31")
    out = {}
    for market, tag in ((SPOT_BASE, "spot"), (FUT_BASE, "futures_5x")):
        cand_res, cand = measure(make_candidate(), btc_df, *window, market)
        v4_res, v4 = measure(get_strategy(INCUMBENT), btc_df, *window, market)
        gap = cand.sharpe - v4.sharpe
        dd_better = cand.max_drawdown_pct < v4.max_drawdown_pct
        passes = (gap >= -F1_NOISE_FLOOR) or dd_better
        out[tag] = dict(cand_sharpe=cand.sharpe, v4_sharpe=v4.sharpe, gap=gap,
                        cand_dd=cand.max_drawdown_pct, v4_dd=v4.max_drawdown_pct,
                        cand_final=cand.final_balance, v4_final=v4.final_balance,
                        passes=passes)
        print(f"  {tag:11s} candidate sharpe={cand.sharpe:6.2f} DD={cand.max_drawdown_pct:5.1f}% "
              f"${cand.final_balance:>9,.0f}  |  v4 sharpe={v4.sharpe:6.2f} "
              f"DD={v4.max_drawdown_pct:5.1f}% ${v4.final_balance:>9,.0f}  |  "
              f"Δsharpe={gap:+.3f}  {'PASS' if passes else 'FAIL'}")
    return out


# ======================================================================= F2


def cmd_f2() -> dict:
    print("\n" + "=" * 100)
    print("F2 — ETH REPLICATION, full window 2020-01-01..last bar, spot @0.10%, "
          "matched-exposure drawdown")
    print("=" * 100)
    eth_df = load_coinbase_spot(DATA_DIR, "ETH")
    window = ("2020-01-01", None)

    def matched_dd_advantage(strategy_factory):
        res, m = measure(strategy_factory(), eth_df, *window, SPOT_BASE)
        c = mean_notional(res)
        mh_res, mh = measure(ConstantExposureHold(c), eth_df, *window, SPOT_BASE)
        ret = daily_returns(res.equity).to_numpy(dtype=float)
        mh_ret = daily_returns(mh_res.equity).to_numpy(dtype=float)
        n = min(len(ret), len(mh_ret))
        dd = paired_bootstrap(ret[:n], mh_ret[:n], max_drawdown_from_returns, **BOOT_KW)
        return dict(final=m.final_balance, dd=m.max_drawdown_pct, c_mean=c,
                    mh_dd=mh.max_drawdown_pct, mh_final=mh.final_balance,
                    dd_advantage=dd.diff.point, dd_lo=dd.diff.lo, dd_hi=dd.diff.hi)

    cand = matched_dd_advantage(make_candidate)
    v4 = matched_dd_advantage(lambda: get_strategy(INCUMBENT))

    regression = cand["dd_advantage"] - v4["dd_advantage"]
    regresses = regression > F2_REGRESSION_TOLERANCE_PP
    print(f"  candidate: c={cand['c_mean']:.3f} DD={cand['dd']:5.1f}% vs matched-hold "
          f"DD={cand['mh_dd']:5.1f}%  dDD={cand['dd_advantage']:+6.1f}pp "
          f"[{cand['dd_lo']:+6.1f},{cand['dd_hi']:+6.1f}]  final=${cand['final']:>9,.0f}")
    print(f"  v4:        c={v4['c_mean']:.3f} DD={v4['dd']:5.1f}% vs matched-hold "
          f"DD={v4['mh_dd']:5.1f}%  dDD={v4['dd_advantage']:+6.1f}pp "
          f"[{v4['dd_lo']:+6.1f},{v4['dd_hi']:+6.1f}]  final=${v4['final']:>9,.0f}")
    print(f"  regression (candidate - v4, positive=worse): {regression:+.1f}pp "
          f"(tolerance {F2_REGRESSION_TOLERANCE_PP}pp) -> "
          f"{'REGRESSES' if regresses else 'does not regress'}")
    return dict(candidate=cand, v4=v4, regression=regression, regresses=regresses)


# ======================================================================= F3/F4


def cmd_panel_cell(ticker, df, window, market, label, rows) -> dict:
    start, end = window
    cand_res, cand = measure(make_candidate(), df, start, end, market)
    hold_res, hold = measure(get_strategy("buy_and_hold"), df, start, end, market)

    c_mean = mean_notional(cand_res)
    mh_res, mh = measure(ConstantExposureHold(c_mean), df, start, end, market)

    cand_ret = daily_returns(cand_res.equity).to_numpy(dtype=float)
    mh_ret = daily_returns(mh_res.equity).to_numpy(dtype=float)
    n = min(len(cand_ret), len(mh_ret))
    dd = paired_bootstrap(cand_ret[:n], mh_ret[:n], max_drawdown_from_returns, **BOOT_KW)
    growth = paired_bootstrap(cand_ret[:n], mh_ret[:n], total_log_return, **BOOT_KW)

    row = dict(asset=ticker, window=label, market=market.name, fee=market.fee_rate,
               cand_final=cand.final_balance, cand_dd=cand.max_drawdown_pct,
               cand_sharpe=cand.sharpe, cand_trades=cand.num_trades,
               hold_final=hold.final_balance, hold_dd=hold.max_drawdown_pct,
               c_mean_notional=c_mean, mh_final=mh.final_balance, mh_dd=mh.max_drawdown_pct,
               dd_diff=dd.diff.point, dd_lo=dd.diff.lo, dd_hi=dd.diff.hi,
               growth_diff=growth.diff.point, growth_lo=growth.diff.lo, growth_hi=growth.diff.hi)
    rows.append(row)
    print(f"  {ticker:5s} {label:5s} {market.name:11s} fee={market.fee_rate:.2%}  "
          f"cand ${cand.final_balance:>10,.0f} DD {cand.max_drawdown_pct:5.1f}% | "
          f"hold ${hold.final_balance:>10,.0f} DD {hold.max_drawdown_pct:5.1f}% | "
          f"matched(c={c_mean:.2f}) ${mh.final_balance:>10,.0f} DD {mh.max_drawdown_pct:5.1f}% | "
          f"dDD {dd.diff.point:+6.1f}pp [{dd.diff.lo:+6.1f},{dd.diff.hi:+6.1f}]")
    return row


def cmd_f3(panel) -> tuple[int, pd.DataFrame]:
    print("\n" + "=" * 100)
    print("F3 (PRIMARY) — 6-asset panel D1, FULL 2020-04-01..last bar, spot @0.10%, "
          "matched-exposure drawdown")
    print("=" * 100)
    rows = []
    for a in panel:
        cmd_panel_cell(a.ticker, a.df, ("2020-04-01", None), SPOT_BASE, "FULL", rows)
    df = pd.DataFrame(rows)
    k = int((df.cand_dd < df.mh_dd).sum())
    excl = int(((df.dd_lo > 0) | (df.dd_hi < 0)).sum())
    better_excl = int((df.dd_hi < 0).sum())
    p = binomial_tail(k, len(panel))
    print(f"\nF3: {k}/{len(panel)} assets, exact binomial p={p:.4f}  "
          f"({excl}/{len(panel)} intervals exclude zero, {better_excl}/{len(panel)} in "
          f"candidate's favour) -> {'GATE PASSES' if k >= F3_GATE_K else 'GATE FAILS'}")
    return k, df


def cmd_f4(panel) -> tuple[int, pd.DataFrame]:
    print("\n" + "=" * 100)
    print("F4 (CONTEXT, not a gate) — panel, FULL window, spot @0.40% Bitstamp tier, "
          "beats buy_and_hold")
    print("=" * 100)
    rows = []
    for a in panel:
        cmd_panel_cell(a.ticker, a.df, ("2020-04-01", None), SPOT_REAL, "FULL", rows)
    df = pd.DataFrame(rows)
    k = int((df.cand_final > df.hold_final).sum())
    print(f"\nF4: {k}/{len(panel)} beats buy_and_hold at 0.40% (context only, "
          f"prediction was FAILS, no strategy in this project's history has cleared this)")
    return k, df


# ========================================================================= main


def cmd_run() -> None:
    ok = cmd_causality()
    if not ok:
        raise SystemExit("CAUSALITY PROBE FAILED — refusing to report results "
                         "until the strategy is causal.")
    print()
    cmd_selfcheck()

    f1 = cmd_f1()
    f2 = cmd_f2()

    panel = select_panel(load_candidates())
    print(f"\nPanel ({len(panel)}): {', '.join(a.ticker for a in panel)}")
    k3, f3_df = cmd_f3(panel)
    k4, f4_df = cmd_f4(panel)

    f1_passes = all(v["passes"] for v in f1.values())
    f2_ok = not f2["regresses"]
    f3_ok = k3 >= F3_GATE_K

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f3_df.to_csv(OUT_DIR / "f3_panel.csv", index=False)
    f4_df.to_csv(OUT_DIR / "f4_panel_040.csv", index=False)

    print("\n" + "=" * 100)
    print("PRE-REGISTERED DECISION RULE (frozen before any of F1-F4 was run)")
    print("=" * 100)
    print(f"F1 (BTC pre-2020 control): {'PASS' if f1_passes else 'FAIL'} "
          f"(spot Δsharpe={f1['spot']['gap']:+.3f}, futures Δsharpe={f1['futures_5x']['gap']:+.3f})")
    print(f"F2 (ETH regression check): {'PASS (no regression)' if f2_ok else 'FAIL (regresses)'} "
          f"(Δ={f2['regression']:+.1f}pp, tolerance {F2_REGRESSION_TOLERANCE_PP}pp)")
    print(f"F3 (panel D1, primary):    {'PASS' if f3_ok else 'FAIL'} ({k3}/{len(panel)}, "
          f"gate is >= {F3_GATE_K}/{len(panel)})")
    print(f"F4 (context, 0.40% tier):  {k4}/{len(panel)} (not a gate)")

    ready = f1_passes and f2_ok and f3_ok
    print(f"\n-> {'READY FOR HOLDOUT CONSULTATION' if ready else 'NEGATIVE'}")
    if not ready:
        print("   Per the pre-registered rule: STOP. No BTC bar dated 2023-01-01 or "
              "later is read anywhere in this module (see load_btc_pre2023()).")
    print(f"\nTotal backtest configurations evaluated: {CONFIG_COUNT}")
    print("Holdout consultations added by this round: 0 "
          "(load_btc_pre2023() truncates every BTC read to <= 2022-12-31; "
          "ETH and the panel are not the project's holdout, per R-47/R-57/R-59 convention)")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        cmd_run()
        return
    if cmd == "causality":
        cmd_causality()
    elif cmd == "selfcheck":
        cmd_selfcheck()
    elif cmd == "f1":
        cmd_f1()
    elif cmd == "f2":
        cmd_f2()
    elif cmd == "f3":
        cmd_f3(select_panel(load_candidates()))
    elif cmd == "f4":
        cmd_f4(select_panel(load_candidates()))
    else:
        raise SystemExit(f"unknown command {cmd!r} "
                         f"(causality | selfcheck | f1 | f2 | f3 | f4 | run)")
    print(f"\nTotal backtest configurations evaluated: {CONFIG_COUNT}")


if __name__ == "__main__":
    main()
