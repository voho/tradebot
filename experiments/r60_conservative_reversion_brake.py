"""R-60 conservative branch: a bounded "reversion brake" overlay on top of
frozen `kelly_regime_v4` (off-backlog, per this round's own mandate — the
backlog was empty of anything OPEN besides B-06 after R-59 closed B-25 and
B-23; the operator's own instruction for this round is to try a genuinely
different strategy family on the R-57 panel rather than a 20th SIZE-axis
tweak, and this branch is the "minimal deviation from the working
incumbent" reading of that instruction).

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Nineteen independent SIZE-axis branches (R-34, R-37, R-38, R-40, R-41,
R-42, R-43, R-45, R-46, R-53 -> R-56, R-59 x2) have tuned
`kelly_regime_v4`'s own vote/scale mechanism and lost every time. R-57
then showed the one thing that survived all nineteen attempts — v4's
matched-exposure drawdown advantage — is present on BTC/ETH and INVERTS
6-of-6 on the R-57 panel (BCH, LTC, ETC, DASH, LINK, XTZ), instruments v4
was never fitted on. R-59's own diagnosis, independently reached by both
of its branches: a rebalanced constant-exposure hold behaves like a
buy-the-dip / mean-reversion rule on these higher-volatility instruments,
and that is most of what the matched hold is winning by — not the sizing
constant's magnitude (R-59-conservative) or its dimensional form
(R-59-novel). v4's trend-vote NEVER changes its own timing to exploit
that; it only ever changes its own SCALE.

This round attacks that gap directly, on a genuinely different axis than
SIZE: **regime-conditional TIMING**, not scale. Constraint attacked:
**SIZE** (still, but the vote's *behaviour* rather than its *magnitude* —
an axis none of the nineteen prior branches touched) and, if it clears
the panel gate, potentially **N≈3** (a mechanism that travels to six
independent instruments rather than two correlated ones).

Grounding, cited before any BTC number is read:
  - Lo & MacKinlay (1988, Review of Financial Studies 1(1), "Stock Market
    Prices Do Not Follow Random Walks") — the variance-ratio statistic
    VR(q) = Var(r^(q)) / (q * Var(r^(1))). VR > 1: positive serial
    correlation (trend-following regime). VR < 1: negative serial
    correlation (mean-reverting regime). VR = 1: random walk. This is the
    regime detector this round uses; it is a STATE variable, not a
    forecast of direction.
  - Lo (2004, J. Portfolio Management 30(5), "The Adaptive Markets
    Hypothesis") — markets cycle between trending and mean-reverting
    regimes rather than being permanently one or the other, which is why
    a fixed-mechanism trend vote can be right in one regime and wrong in
    another, and why a REGIME-GATED overlay (active only when the state
    variable says "mean-reverting", inert otherwise) is the right shape
    for a fix, rather than an always-on combination.
  - Beluska & Vojtko (2024, Quantpedia/SSRN, "Revisiting Trend-following
    and Mean-reversion Strategies in Bitcoin") — a strategy COMBINING
    trend and mean-reversion signals on BTC outperforms either alone,
    2015-2024. This round is the minimal, bounded, fallback-safe version
    of that combination: v4's own trend vote stays exactly as is, and a
    small mean-reversion brake activates ONLY in the regime VR already
    identifies as mean-reverting, and only counters the vote (never
    amplifies it).

**Not a duplicate of anything in section C.** Every prior SIZE-axis
branch (R-34 -> R-56, R-59 x2) changed how big v4's exposure is as a
function of vote/vol; none of them added a state-dependent TIMING
override that can move exposure in the direction OPPOSITE the vote. Not a
duplicate of the five INFO-axis branches (R-44, R-53, R-54, R-55, R-58 x2)
either — those all tried to recover a second information channel
(stablecoin supply, on-chain activity); this round uses no information
beyond the same OHLCV close series v4 already reads, and its axis is
mean-reversion vs. trend-following REGIME, not a new data source.

**Simulable here**: yes — a rolling variance-ratio and an EMA-extension
z-score are both `.rolling()`/`.ewm()`/`.shift()` computations over the
already-loaded close series, exactly like every anchor and vol term v4
already computes. No order book, no new data file.

**What would make this fail, named now, before any BTC number is read**:
if the brake fails the F1 pre-2020 BTC control (the R-40/R-41/R-42/R-43/
R-45/R-46 failure signature — losing to v4 on a window inside inner-train
before the panel is even read); OR if it regresses v4's own already-
established ETH matched-exposure drawdown property (F2); OR if it scores
< 5/6 on the R-57 panel's matched-exposure drawdown count (F3) — i.e. it
does not meaningfully outperform v4's own measured 0/6 there. Any of
those is NEGATIVE and this branch does not touch the BTC holdout under
any circumstance.

=====================================================================
MECHANISM (frozen here, BEFORE any strategy result is read anywhere in
this file — the constants below are chosen by reasoning/citation alone)
=====================================================================

1. **Rolling variance ratio VR(q)`, daily, causal.** 5m closes are
   resampled to daily (the last close of each UTC day — pandas
   `resample("1D").last()`, left-labelled, so the label for day D holds
   the last print BEFORE day D+1 starts). VR is computed on this daily
   series with a `VR_WINDOW_DAYS = 90` trailing window and
   `Q_DAYS = 5` (a business-week-scale horizon, matched to the 5-day EMA
   the extension signal below also uses, so both halves of the mechanism
   share one clock). 90 days is a quarter's worth of daily observations —
   long enough that VR's sampling noise (nontrivial at daily frequency
   under the null) does not flip the state every week, short enough to
   track Lo's regime-cycling claim rather than measuring one full-sample
   constant. These are round numbers chosen for their scale, not fit to
   any BTC result — see the discipline note below.

2. **Reversion extension z-score.** `ext = log(close) - EMA_5day(log
   close)` on the same daily series; `z = ext / rolling_std(ext,
   DISP_WINDOW_DAYS=30)`. A short EMA (5 days) so the "extension" reads
   as a genuinely short-run deviation, z-scored against its own trailing
   30-day dispersion so the threshold is dimensionless and adapts to each
   instrument's own volatility (no per-instrument constant to fit, unlike
   the SIZE-axis branches R-59 already showed do not work).

3. **No partial-day leak.** Both series are computed on the daily frame,
   then their index is shifted forward by exactly one calendar day before
   being reindexed onto the 5m bar grid with `ffill`. The daily value
   labelled day D is not actually complete until day D's last 5m print
   (23:55 UTC); shifting its index to D+1 is what makes it available
   starting at D+1's first bar, not before — the literal implementation
   of "bar i only uses fully-completed daily data." Verified below with a
   tamper probe (Section: causality), not just asserted.

4. **The brake.** Trigger: `VR < VR_THRESHOLD=0.85` (an honest buffer
   below the neutral 1.0 — VR's null-sampling noise, not a genuinely
   mean-reverting state, is what a threshold placed AT 1.0 would catch)
   AND `|z| > Z_THRESHOLD=1.5`. Both thresholds are the exact values this
   round's own mandate suggested, adopted rather than swept, precisely so
   they cannot be read as fit to any BTC or panel result. Strength:
       vr_strength = clip((VR_THRESHOLD - VR) / VR_THRESHOLD, 0, 1)
       z_strength  = clip((|z| - Z_THRESHOLD) / Z_THRESHOLD, 0, 1)
       strength    = vr_strength * z_strength          (0 outside the gate)
       fade        = MAX_FADE * strength               (MAX_FADE = 1.0)
   New target, applied to v4's OWN (unmodified) target column:
       new_target = target * (1 - fade * (1 + COUNTER_FRAC * sign(z)))
   with `COUNTER_FRAC = 0.30`. This is the ONLY new arithmetic in the
   file; everything upstream of `target` (the three-anchor vote, the
   vol-target scale, v4's own deadband) is `super().prepare()`,
   byte-for-byte.

   **Why this exact formula, and the safety property it has by
   construction** (not just empirically checked): v4's `target` is always
   >= 0 for this whole strategy family (a [0,1] vote fraction times a
   non-negative vol-scale — it never shorts). Write the bracket
   `B = 1 - fade*(1 + COUNTER_FRAC*sign(z))`. Since `fade in [0,1]` and
   `COUNTER_FRAC in [0,1]`:
     - `sign(z) = -1` (price BELOW its short EMA — the reversion signal
       is BULLISH, i.e. it AGREES with v4's long-only bet): B in
       [COUNTER_FRAC, 1] — the brake can fade toward a floor
       (`COUNTER_FRAC * target`, never below it) but can never flip sign,
       because amplifying an already-agreeing long bet is exactly the
       amplification this brake is forbidden from doing.
     - `sign(z) = +1` (price ABOVE its short EMA — the reversion signal
       is BEARISH, i.e. it DISAGREES with v4's long-only bet): B in
       [-COUNTER_FRAC, 1] — the brake may fade all the way through zero
       into a small countertrend short, capped at `COUNTER_FRAC *
       target` in magnitude.
   In both cases `|new_target| <= target = |v4's own target|` for EVERY
   value of `fade` and `COUNTER_FRAC` in `[0,1]` — an algebraic invariant,
   not a per-run measurement, so this overlay cannot amplify v4's
   exposure and cannot exceed v4's own `max_leverage` (v4 already caps
   `target` there) under any parameter choice in that range. Outside the
   trigger, `strength = 0` so `fade = 0` so `new_target = target` EXACTLY
   (floating-point identical) — the fallback-safety property that makes
   this "conservative" in the `kelly_regime_v2`/`v3`/`v4` sense: v4 is
   reproduced byte-for-byte whenever the brake's own trigger is absent.

**Discipline**: `VR_WINDOW_DAYS`, `Q_DAYS`, `VR_THRESHOLD`,
`EMA_SPAN_DAYS`, `DISP_WINDOW_DAYS`, `Z_THRESHOLD` are the round's own
suggested values, adopted as-is. `MAX_FADE` and `COUNTER_FRAC` are this
branch's two free choices; they are fixed BEFORE any strategy backtest
runs anywhere in this file (this docstring is written and the constants
below are set in code before `cmd_causality`, `cmd_inner`, or any F-test
is invoked) and are never adjusted afterward. A small neighbourhood
robustness check (Section: neighbourhood) is run AFTER the frozen
F1-F4 battery, on non-holdout data only, purely to characterize whether
the frozen choice sits on a plateau — it cannot and does not change the
frozen configuration or the decision rule below.

=====================================================================
PRE-REGISTERED FALSIFICATION BATTERY (frozen before any strategy number
in this file is read — F1 runs first, in the order below)
=====================================================================

F1 — BTC PRE-2020 CONTROL (2017-01-01 -> 2019-12-31, inside inner-train,
     never touches 2023+). Candidate's Sharpe must not underperform
     frozen `kelly_regime_v4`'s Sharpe on the identical window by more
     than the project's +/-0.2 Sharpe noise floor (R-20). This is the
     exact failure signature that killed R-40, R-41, R-42, R-43, R-45,
     R-46 before ETH was even read — checked FIRST.

F2 — ETH REPLICATION (Coinbase ETH, full window, spot @0.10%,
     `ConstantExposureHold` matched to the candidate's OWN mean notional,
     R-57/R-47's `cell()` methodology). Must not regress the matched-
     exposure drawdown advantage v4 already has there, measured on the
     SAME window in this same file (tolerance: D2_REGRESSION_TOLERANCE_PP
     = 5.0 percentage points, the same tolerance R-59's own pre-
     registration used for an identical purpose).

F3 — PRIMARY: 6-ASSET PANEL D1 (BCH, LTC, ETC, DASH, LINK, XTZ, FULL
     window 2020-04-01 -> last bar, spot @0.10%, `MarketSpec.spot()`).
     Count assets where the candidate's own max drawdown is strictly
     below a `ConstantExposureHold` matched to the candidate's OWN mean
     notional over the same window. v4 alone scores 0/6 (R-57). Reported
     with each asset's paired stationary-block-bootstrap interval (daily
     returns, 30-day mean block, 2,000 resamples, seed 7 — `BOOT_KW`,
     identical to R-57/R-59).

F4 — CONTEXT (report only, not a gate). Same panel at the 0.40% Bitstamp
     entry-tier fee: does the candidate beat `buy_and_hold`'s final
     balance in >= 5/6? No strategy in this project's history has ever
     cleared this; recorded for completeness only.

**PRE-REGISTERED DECISION RULE, written here before F1-F4 run and never
moved afterward:** this branch is "ready for a holdout consultation" only
if F1 PASSES AND F2 does not regress AND F3 >= 5/6. Anything else is
NEGATIVE. **If the rule is not cleared, this file's own logic never
reads a BTC bar dated 2023-01-01 or later, under any circumstance** — the
data loaders below truncate BTC reads at the F1/F2/F3 windows explicitly,
not just by which window string is later passed to `run_period`.

Usage::

    python experiments/r60_conservative_reversion_brake.py causality
    python experiments/r60_conservative_reversion_brake.py inner     # descriptive, non-gating
    python experiments/r60_conservative_reversion_brake.py f1
    python experiments/r60_conservative_reversion_brake.py f2
    python experiments/r60_conservative_reversion_brake.py f3
    python experiments/r60_conservative_reversion_brake.py f4
    python experiments/r60_conservative_reversion_brake.py neighborhood
    python experiments/r60_conservative_reversion_brake.py run        # everything, writes CSVs
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
    BOOT_KW,
    SPOT_BASE,
    SPOT_REAL,
    Asset,
    binomial_tail,
)
from experiments.r59_shared import load_panel  # noqa: E402
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
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "reports" / "r60_conservative"
REPORT_PATH = ROOT / "experiments" / "reports" / "r60_conservative_report.md"

# ---------------------------------------------------------- frozen mechanism
# Every constant below is fixed BEFORE any backtest in this file runs (see
# the docstring's MECHANISM section for the reasoning behind each one) and
# is never adjusted after seeing a result.

VR_WINDOW_DAYS = 90
Q_DAYS = 5
VR_THRESHOLD = 0.85
EMA_SPAN_DAYS = 5
DISP_WINDOW_DAYS = 30
Z_THRESHOLD = 1.5
MAX_FADE = 1.0
COUNTER_FRAC = 0.30

# ------------------------------------------------------------------ windows
# F1 sits inside inner-train (2017-2020); nothing here reads a BTC bar
# dated 2023-01-01 or later — enforced below by truncating the loaded BTC
# frame immediately after reading it, before any other line touches it.
F1_WINDOW = ("2017-01-01", "2019-12-31")
INNER_TRAIN = ("2017-01-01", "2020-12-31")
INNER_VALID = ("2021-01-01", "2022-12-31")
BTC_TRUNCATE_AT = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")

# F2/F3/F4 windows. ETH and the panel carry no holdout restriction in this
# project (R-47/B-08, R-57) — read freely, full range.
ETH_FULL = ("2020-04-01", None)   # matches R-57's own FULL definition
PANEL_FULL = ("2020-04-01", None)

D2_REGRESSION_TOLERANCE_PP = 5.0  # same convention R-59's own D2 used

CONFIG_COUNT = 0


# ------------------------------------------------------------------ helpers


def measure(strategy, df, start, end, market):
    """One backtest. Every call is counted — there is no free evaluation."""
    global CONFIG_COUNT
    CONFIG_COUNT += 1
    result = run_period(strategy, df, start, end, market=market, start_balance=1_000.0)
    return result, compute_metrics(result)


def load_btc_no_holdout() -> pd.DataFrame:
    """BTC spot, truncated at 2022-12-31 BEFORE any other line touches it.

    This is the round's holdout guarantee enforced in code, not just by
    which window string is later passed to `run_period` — the exact
    pattern R-59's own branches used (`load_control_assets`).
    """
    df, _label = load_dataset(DATA_DIR, "spot")
    return df[df.index <= BTC_TRUNCATE_AT]


def load_eth() -> pd.DataFrame:
    """ETH Coinbase spot, full range. No holdout restriction (R-47/B-08)."""
    df = load_coinbase_spot(DATA_DIR, "ETH")
    if df is None:
        raise SystemExit("ETH data not found — expected data/ethusd_coinbase_spot_5m.csv.gz")
    return df


# --------------------------------------------------------------- mechanism


def daily_vr_and_zscore(df: pd.DataFrame, vr_window_days: int, q_days: int,
                         ema_span_days: int, disp_window_days: int
                         ) -> tuple[pd.Series, pd.Series]:
    """Rolling variance ratio and reversion-extension z-score, daily, causal.

    Both series are computed on the DAILY frame (so each is causal at
    daily granularity by construction — rolling/ewm over daily closes),
    then their index is shifted forward by exactly one calendar day: the
    row labelled day D is not actually complete until day D's own last 5m
    print, so shifting it to D+1 is what makes it available starting at
    D+1, not before. Reindexing onto the 5m grid with `ffill` (done by the
    caller) is then a genuinely causal operation — see the docstring's
    "No partial-day leak" section and the causality tamper probe below.
    """
    close = df["close"]
    daily = close.resample("1D").last().dropna()
    logp = np.log(daily)

    r1 = logp.diff()
    var1 = r1.rolling(vr_window_days, min_periods=vr_window_days).var()
    rq = logp.diff(q_days)
    varq = rq.rolling(vr_window_days, min_periods=vr_window_days).var()
    with np.errstate(divide="ignore", invalid="ignore"):
        vr_vals = np.where((var1.to_numpy() > 0) & np.isfinite(var1.to_numpy()),
                           varq.to_numpy() / (q_days * var1.to_numpy()), np.nan)
    vr = pd.Series(vr_vals, index=daily.index)

    ema = logp.ewm(span=ema_span_days, min_periods=ema_span_days).mean()
    ext = logp - ema
    disp = ext.rolling(disp_window_days, min_periods=disp_window_days).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z_vals = np.where((disp.to_numpy() > 0) & np.isfinite(disp.to_numpy()),
                          ext.to_numpy() / disp.to_numpy(), np.nan)
    z = pd.Series(z_vals, index=daily.index)

    vr_avail = vr.copy()
    vr_avail.index = vr_avail.index + pd.Timedelta(days=1)
    z_avail = z.copy()
    z_avail.index = z_avail.index + pd.Timedelta(days=1)
    return vr_avail, z_avail


class ReversionBrakeV4(KellyRegimeV4):
    """`kelly_regime_v4` plus a bounded, additive reversion-brake overlay.

    Subclasses `KellyRegimeV4` and overrides only `prepare()`: calls
    `super().prepare()` for v4's own unmodified target, then applies the
    VR/z brake described in this module's docstring. `on_bar` is
    inherited unchanged from `KellyRegime` (reads the `target` column,
    orders on a change bigger than 1e-9). Not `@register`ed and not under
    `src/tradebot/strategies/` — an experiment, instantiated directly.
    """

    name = "r60_reversion_brake"

    def __init__(self, vr_window_days: int = VR_WINDOW_DAYS, q_days: int = Q_DAYS,
                 vr_threshold: float = VR_THRESHOLD, ema_span_days: int = EMA_SPAN_DAYS,
                 disp_window_days: int = DISP_WINDOW_DAYS, z_threshold: float = Z_THRESHOLD,
                 max_fade: float = MAX_FADE, counter_frac: float = COUNTER_FRAC,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if not (0.0 <= max_fade <= 1.0):
            raise ValueError(f"max_fade must be in [0,1], got {max_fade!r}")
        if not (0.0 <= counter_frac <= 1.0):
            raise ValueError(f"counter_frac must be in [0,1], got {counter_frac!r}")
        self.vr_window_days = vr_window_days
        self.q_days = q_days
        self.vr_threshold = vr_threshold
        self.ema_span_days = ema_span_days
        self.disp_window_days = disp_window_days
        self.z_threshold = z_threshold
        self.max_fade = max_fade
        self.counter_frac = counter_frac
        # Extra warmup so the overlay itself is warm at the first measured
        # bar: max(v4's own warmup, the overlay's own daily lookback + a
        # small buffer), in days converted to bars.
        overlay_days = vr_window_days + q_days + max(ema_span_days, disp_window_days) + 5
        self.warmup = max(KellyRegimeV4.warmup, overlay_days * BARS_PER_DAY)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4's own target column, byte-for-byte
        vr_avail, z_avail = daily_vr_and_zscore(
            df, self.vr_window_days, self.q_days, self.ema_span_days, self.disp_window_days)
        vr_5m = vr_avail.reindex(df.index, method="ffill").to_numpy(dtype=float)
        z_5m = z_avail.reindex(df.index, method="ffill").to_numpy(dtype=float)

        target = df["target"].to_numpy(dtype=float)
        assert np.all(target >= -1e-9), (
            "overlay's boundedness proof assumes v4's own target is non-negative")

        trigger = (np.isfinite(vr_5m) & np.isfinite(z_5m)
                  & (vr_5m < self.vr_threshold) & (np.abs(z_5m) > self.z_threshold))
        vr_strength = np.clip((self.vr_threshold - vr_5m) / self.vr_threshold, 0.0, 1.0)
        z_strength = np.clip((np.abs(z_5m) - self.z_threshold) / self.z_threshold, 0.0, 1.0)
        strength = np.where(trigger, vr_strength * z_strength, 0.0)
        strength = np.clip(np.nan_to_num(strength, nan=0.0), 0.0, 1.0)

        fade = self.max_fade * strength
        sign_z = np.nan_to_num(np.sign(z_5m), nan=0.0)
        bracket = 1.0 - fade * (1.0 + self.counter_frac * sign_z)
        new_target = target * bracket
        # Defensive cap: the bracket's algebraic range is [-counter_frac, 1]
        # for fade,counter_frac in [0,1], so this should already hold; keep
        # the explicit clip so a float edge case can never silently violate
        # the "never amplify v4's own target" guarantee.
        new_target = np.clip(new_target, -np.abs(target), np.abs(target))

        df["vr"] = vr_5m
        df["reversion_z"] = z_5m
        df["brake_strength"] = strength
        df["target"] = new_target
        return df


# --------------------------------------------------------------- causality


def cmd_causality() -> bool:
    """R-57/R-59's tamper-probe methodology: perturb bars after a cut point
    in opposite directions and confirm every decision at or before the cut
    is identical. Constructs the candidate directly (never registered)."""
    print("=" * 100)
    print("CAUSALITY TAMPER PROBE — ReversionBrakeV4(frozen)")
    print("=" * 100)
    btc = load_btc_no_holdout()
    eth = load_eth()
    market = MarketSpec.futures(leverage=5.0)
    all_ok = True
    for label, frame in (("BTC", btc), ("ETH", eth)):
        tail = frame.iloc[-90_000:].copy()  # long enough to warm the overlay
        cut = len(tail) - 5_000
        bars = [cut - k for k in (1, 2, 3, 5, 10, 20)]
        up, down = tail.copy(), tail.copy()
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

        def decisions(frame_):
            s = ReversionBrakeV4()
            prepared = s.prepare(frame_.copy())
            broker = PaperBroker(market=market, start_balance=10_000.0)
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        ok = all(x == y for x, y in zip(decisions(up), decisions(down)))
        all_ok = all_ok and ok
        print(f"  {label:5s} decisions identical under opposite post-cut tampers: "
              f"{'PASS' if ok else 'FAIL'}")

    # Boundedness invariant, checked numerically over a real series (not
    # just algebraically argued in the docstring): |new_target| <= |v4's
    # own target| at every bar.
    v4_only = KellyRegimeV4().prepare(btc.copy())["target"].to_numpy(dtype=float)
    brake = ReversionBrakeV4().prepare(btc.copy())["target"].to_numpy(dtype=float)
    bounded = bool(np.all(np.abs(brake) <= np.abs(v4_only) + 1e-9))
    print(f"  boundedness invariant (|brake target| <= |v4 target|, every BTC bar "
          f"pre-2023): {'PASS' if bounded else 'FAIL'}")
    fallback_bars = int(np.sum(np.isclose(brake, v4_only)))
    print(f"  exact fallback-to-v4 bars: {fallback_bars}/{len(v4_only)} "
          f"({fallback_bars / len(v4_only):.1%}) — brake active the rest of the time")
    all_ok = all_ok and bounded
    return all_ok


# ----------------------------------------------------------------- F1-F4


def cmd_f1() -> dict:
    print("=" * 100)
    print("F1 — BTC PRE-2020 CONTROL (2017-01-01..2019-12-31, inside inner-train)")
    print("=" * 100)
    btc = load_btc_no_holdout()
    _, cand = measure(ReversionBrakeV4(), btc, *F1_WINDOW, SPOT_BASE)
    _, v4 = measure(KellyRegimeV4(), btc, *F1_WINDOW, SPOT_BASE)
    delta = cand.sharpe - v4.sharpe
    passed = delta >= -0.2
    print(f"  candidate sharpe={cand.sharpe:.3f} DD={cand.max_drawdown_pct:.1f}% "
          f"final=${cand.final_balance:,.0f}")
    print(f"  v4        sharpe={v4.sharpe:.3f} DD={v4.max_drawdown_pct:.1f}% "
          f"final=${v4.final_balance:,.0f}")
    print(f"  delta sharpe = {delta:+.3f}  (noise floor -0.2)  -> "
          f"{'PASS' if passed else 'FAIL'}")
    return {"cand_sharpe": cand.sharpe, "v4_sharpe": v4.sharpe, "delta": delta,
            "passed": passed}


def cmd_inner() -> list[dict]:
    """Descriptive only — ROUTINE.md step 3's inner-train/inner-validation
    split. Reported for the record; changes nothing (the frozen mechanism
    constants were fixed in this file's docstring/module constants BEFORE
    this or any other command ran)."""
    print("=" * 100)
    print("INNER-TRAIN / INNER-VALIDATION — descriptive, non-gating, BTC")
    print("=" * 100)
    btc = load_btc_no_holdout()
    rows = []
    for label, window in (("inner-train 2017-2020", INNER_TRAIN),
                          ("inner-validation 2021-2022", INNER_VALID)):
        _, cand = measure(ReversionBrakeV4(), btc, *window, SPOT_BASE)
        _, v4 = measure(KellyRegimeV4(), btc, *window, SPOT_BASE)
        rows.append({"window": label, "cand_sharpe": cand.sharpe, "v4_sharpe": v4.sharpe,
                    "cand_dd": cand.max_drawdown_pct, "v4_dd": v4.max_drawdown_pct,
                    "cand_final": cand.final_balance, "v4_final": v4.final_balance})
        print(f"  {label:28s} candidate sharpe={cand.sharpe:5.2f} DD={cand.max_drawdown_pct:5.1f}% "
              f"final=${cand.final_balance:>10,.0f} | v4 sharpe={v4.sharpe:5.2f} "
              f"DD={v4.max_drawdown_pct:5.1f}% final=${v4.final_balance:>10,.0f}")
    return rows


def cell(a_ticker: str, df: pd.DataFrame, strategy, window, market, label: str,
        rows: list) -> dict:
    """One asset x window x market cell: candidate, buy_and_hold, matched
    hold, paired-bootstrap intervals. Same structure as R-57/R-59's cell()."""
    start, end = window
    cand_res, cand = measure(strategy, df, start, end, market)
    hold_res, hold = measure(get_strategy("buy_and_hold"), df, start, end, market)

    c_mean = mean_notional(cand_res)
    mh_res, mh = measure(ConstantExposureHold(c_mean), df, start, end, market)

    cand_ret = daily_returns(cand_res.equity).to_numpy(dtype=float)
    mh_ret = daily_returns(mh_res.equity).to_numpy(dtype=float)
    hold_ret = daily_returns(hold_res.equity).to_numpy(dtype=float)
    n = min(len(cand_ret), len(mh_ret), len(hold_ret))
    dd_matched = paired_bootstrap(cand_ret[:n], mh_ret[:n], max_drawdown_from_returns, **BOOT_KW)
    growth_matched = paired_bootstrap(cand_ret[:n], mh_ret[:n], total_log_return, **BOOT_KW)

    row = {
        "asset": a_ticker, "window": label, "market": market.name, "fee": market.fee_rate,
        "cand_final": cand.final_balance, "cand_dd": cand.max_drawdown_pct,
        "cand_sharpe": cand.sharpe, "cand_trades": cand.num_trades, "cand_liq": cand.liquidated,
        "hold_final": hold.final_balance, "hold_dd": hold.max_drawdown_pct,
        "hold_sharpe": hold.sharpe, "hold_liq": hold.liquidated,
        "c_mean_notional": c_mean,
        "mh_final": mh.final_balance, "mh_dd": mh.max_drawdown_pct, "mh_sharpe": mh.sharpe,
        "dd_matched_diff": dd_matched.diff.point,
        "dd_matched_lo": dd_matched.diff.lo, "dd_matched_hi": dd_matched.diff.hi,
        "growth_matched_diff": growth_matched.diff.point,
        "growth_matched_lo": growth_matched.diff.lo, "growth_matched_hi": growth_matched.diff.hi,
    }
    rows.append(row)
    print(f"  {a_ticker:5s} {label:9s} {market.name:11s} fee={market.fee_rate:.2%}  "
          f"cand ${cand.final_balance:>10,.0f} DD {cand.max_drawdown_pct:5.1f}% | "
          f"hold ${hold.final_balance:>10,.0f} DD {hold.max_drawdown_pct:5.1f}% | "
          f"matched(c={c_mean:.2f}) ${mh.final_balance:>10,.0f} DD {mh.max_drawdown_pct:5.1f}% | "
          f"dDD_matched {dd_matched.diff.point:+6.1f}pp "
          f"[{dd_matched.diff.lo:+6.1f},{dd_matched.diff.hi:+6.1f}]")
    return row


def cmd_f2() -> dict:
    print("=" * 100)
    print("F2 — ETH REPLICATION (full window, spot @0.10%, matched-exposure drawdown)")
    print("=" * 100)
    eth = load_eth()
    rows: list[dict] = []
    cand_row = cell("ETH", eth, ReversionBrakeV4(), ETH_FULL, SPOT_BASE, "FULL", rows)
    v4_row = cell("ETH", eth, KellyRegimeV4(), ETH_FULL, SPOT_BASE, "FULL", rows)
    cand_adv = cand_row["dd_matched_diff"]  # negative = candidate draws down less (better)
    v4_adv = v4_row["dd_matched_diff"]
    regressed = cand_adv > v4_adv + D2_REGRESSION_TOLERANCE_PP
    print(f"  candidate matched-exposure dDD = {cand_adv:+.1f}pp   "
          f"v4 matched-exposure dDD = {v4_adv:+.1f}pp   "
          f"tolerance = {D2_REGRESSION_TOLERANCE_PP:+.1f}pp -> "
          f"{'REGRESSES' if regressed else 'DOES NOT REGRESS'}")
    return {"cand_adv": cand_adv, "v4_adv": v4_adv, "regressed": regressed, "rows": rows}


def cmd_f3(panel: list[Asset]) -> dict:
    print("=" * 100)
    print("F3 (PRIMARY) — 6-ASSET PANEL D1, FULL window, spot @0.10%")
    print("=" * 100)
    rows: list[dict] = []
    for a in panel:
        cell(a.ticker, a.df, ReversionBrakeV4(), PANEL_FULL, SPOT_BASE, "FULL", rows)
    d1 = pd.DataFrame(rows)
    n = len(panel)
    k = int((d1.cand_dd < d1.mh_dd).sum())
    excl = int(((d1.dd_matched_lo > 0) | (d1.dd_matched_hi < 0)).sum())
    better_excl = int((d1.dd_matched_hi < 0).sum())
    p = binomial_tail(k, n)
    print(f"\n  F3: {k}/{n} assets (exact binomial p={p:.4f}); "
          f"{excl}/{n} bootstrap intervals exclude zero ({better_excl}/{n} favour candidate)")
    return {"k": k, "n": n, "p": p, "rows": rows}


def cmd_f4(panel: list[Asset]) -> dict:
    print("=" * 100)
    print("F4 (CONTEXT ONLY) — 6-ASSET PANEL, FULL window, spot @0.40% Bitstamp tier")
    print("=" * 100)
    rows = []
    for a in panel:
        cand_res, cand = measure(ReversionBrakeV4(), a.df, *PANEL_FULL, SPOT_REAL)
        hold_res, hold = measure(get_strategy("buy_and_hold"), a.df, *PANEL_FULL, SPOT_REAL)
        rows.append({"asset": a.ticker, "cand_final": cand.final_balance,
                    "hold_final": hold.final_balance})
        print(f"  {a.ticker:5s} candidate ${cand.final_balance:>10,.0f} vs "
              f"buy_and_hold ${hold.final_balance:>10,.0f} -> "
              f"{'beats' if cand.final_balance > hold.final_balance else 'loses to'} hold")
    n = len(panel)
    k = int(sum(1 for r in rows if r["cand_final"] > r["hold_final"]))
    print(f"\n  F4: {k}/{n} beat buy_and_hold at 0.40% -> "
          f"{'SURVIVES' if k >= n - 1 else 'FAILS (context only, not a gate)'}")
    return {"k": k, "n": n, "rows": rows}


# --------------------------------------------------------- neighbourhood


def cmd_neighborhood() -> list[dict]:
    """Post-hoc robustness only, run AFTER the frozen F1-F4 battery, on
    non-holdout BTC data (inner-train + inner-validation) only. Cannot and
    does not change the frozen candidate above or the decision rule — it
    exists purely to show whether MAX_FADE/COUNTER_FRAC (this branch's two
    free choices) sit on a plateau, per ROUTINE.md's promotion-bar
    convention ("report the neighbours, not just the winner")."""
    print("=" * 100)
    print("NEIGHBOURHOOD (post-hoc, non-gating) — MAX_FADE x COUNTER_FRAC, "
          "BTC inner-train + inner-validation")
    print("=" * 100)
    btc = load_btc_no_holdout()
    rows = []
    for max_fade in (0.5, 1.0):
        for counter_frac in (0.0, 0.3, 0.6):
            strat_kwargs = dict(max_fade=max_fade, counter_frac=counter_frac)
            for label, window in (("inner-train", INNER_TRAIN), ("inner-validation", INNER_VALID)):
                _, cand = measure(ReversionBrakeV4(**strat_kwargs), btc, *window, SPOT_BASE)
                rows.append({"max_fade": max_fade, "counter_frac": counter_frac,
                            "window": label, "sharpe": cand.sharpe,
                            "dd": cand.max_drawdown_pct, "final": cand.final_balance})
                print(f"  max_fade={max_fade:.1f} counter_frac={counter_frac:.1f} "
                      f"{label:17s} sharpe={cand.sharpe:5.2f} DD={cand.max_drawdown_pct:5.1f}% "
                      f"final=${cand.final_balance:>10,.0f}")
    return rows


# ----------------------------------------------------------------- verdict


def verdict(f1: dict, f2: dict, f3: dict) -> str:
    f1_ok = f1["passed"]
    f2_ok = not f2["regressed"]
    f3_ok = f3["k"] >= f3["n"] - 1  # >= 5/6
    ready = f1_ok and f2_ok and f3_ok
    print("\n" + "=" * 100)
    print("PRE-REGISTERED DECISION RULE (frozen before F1-F4 ran, applied mechanically)")
    print("=" * 100)
    print(f"  F1 (BTC pre-2020 control)         : {'PASS' if f1_ok else 'FAIL'} "
          f"(delta sharpe {f1['delta']:+.3f}, floor -0.2)")
    print(f"  F2 (ETH matched-exposure, no regress): {'PASS' if f2_ok else 'FAIL'} "
          f"(cand {f2['cand_adv']:+.1f}pp vs v4 {f2['v4_adv']:+.1f}pp, "
          f"tolerance {D2_REGRESSION_TOLERANCE_PP:+.1f}pp)")
    print(f"  F3 (panel D1 >= 5/6)              : {'PASS' if f3_ok else 'FAIL'} "
          f"({f3['k']}/{f3['n']})")
    print(f"\n  OVERALL: {'READY FOR HOLDOUT CONSULTATION' if ready else 'NEGATIVE'}")
    return "READY" if ready else "NEGATIVE"


# ---------------------------------------------------------------------- run


def cmd_run() -> None:
    print(f"Frozen mechanism: VR_WINDOW_DAYS={VR_WINDOW_DAYS} Q_DAYS={Q_DAYS} "
          f"VR_THRESHOLD={VR_THRESHOLD} EMA_SPAN_DAYS={EMA_SPAN_DAYS} "
          f"DISP_WINDOW_DAYS={DISP_WINDOW_DAYS} Z_THRESHOLD={Z_THRESHOLD} "
          f"MAX_FADE={MAX_FADE} COUNTER_FRAC={COUNTER_FRAC}\n")

    ok = cmd_causality()
    print()
    if not ok:
        raise SystemExit("CAUSALITY PROBE FAILED — refusing to report F1-F4 "
                         "results until the lookahead bug is fixed.")

    inner_rows = cmd_inner()
    print()
    f1 = cmd_f1()
    print()
    f2 = cmd_f2()
    print()
    panel = load_panel()
    f3 = cmd_f3(panel)
    print()
    f4 = cmd_f4(panel)
    print()
    nb_rows = cmd_neighborhood()

    v = verdict(f1, f2, f3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inner_rows).to_csv(OUT_DIR / "inner.csv", index=False)
    pd.DataFrame(f2["rows"]).to_csv(OUT_DIR / "f2_eth.csv", index=False)
    pd.DataFrame(f3["rows"]).to_csv(OUT_DIR / "f3_panel.csv", index=False)
    pd.DataFrame(f4["rows"]).to_csv(OUT_DIR / "f4_panel_040.csv", index=False)
    pd.DataFrame(nb_rows).to_csv(OUT_DIR / "neighborhood.csv", index=False)

    print(f"\nTotal backtest configurations evaluated: {CONFIG_COUNT}")
    print(f"Verdict: {v}")
    print("Holdout consultations added by this round: 0 "
          "(BTC truncated at 2022-12-31 before any other line touches it; "
          "ETH and panel reads cost +0 per the R-47/R-57 convention)")
    print(f"No BTC bar dated 2023-01-01 or later was read anywhere in this file "
          f"(load_btc_no_holdout() truncates at {BTC_TRUNCATE_AT}).")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    dispatch = {
        "causality": lambda: print(f"\nOverall: {'PASS' if cmd_causality() else 'FAIL'}"),
        "inner": cmd_inner,
        "f1": cmd_f1,
        "f2": cmd_f2,
        "f3": lambda: cmd_f3(load_panel()),
        "f4": lambda: cmd_f4(load_panel()),
        "neighborhood": cmd_neighborhood,
        "run": cmd_run,
    }
    if cmd not in dispatch:
        raise SystemExit(f"unknown command {cmd!r} ({' | '.join(dispatch)})")
    dispatch[cmd]()
    if cmd != "run":
        print(f"\nConfigurations evaluated by this command: {CONFIG_COUNT}")


if __name__ == "__main__":
    main()
