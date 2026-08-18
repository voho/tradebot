"""Skewness-gated exposure: a third-moment haircut on top of kelly_regime_v4's
volatility target (a novel direction, not yet a ledger row — see the session
report for the ID this becomes).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The literature, and what it actually says
------------------------------------------
Daniel & Moskowitz (2016, JFE 122, "Momentum Crashes"): momentum returns
are strongly negatively skewed, with rare but severe crashes concentrated
in specific conditions — "panic states" that follow a market decline
(their example: the loser decile was down 84% from its peak by March
2009) *and* coincide with high ex-ante market volatility, and the crash
itself fires exactly as the market rebounds. Their mechanism is dynamic
beta: past losers carry an embedded, option-like convexity (a levered
call on the market), so when a depressed market snaps back hard, losers
outperform winners by even more, and a momentum book (long winners,
underweight/short losers) is hurt on both legs at once. Their fix — an
implementable dynamic momentum strategy that scales exposure down using
forecasts of momentum's own mean and variance — roughly doubles the
Sharpe ratio and alpha of static momentum.

Bianchi, De Polis & Petrella ("Time-Varying Skewness and Momentum
Crashes", SSRN 4182040 / CEPR DP19030 / whitesphd.com/workingpapers/wp8,
"Taming Momentum Crashes"): momentum's return skewness is itself
time-varying and deepens exactly during crashes. Their applied
contribution is a crash indicator built from the **interaction between
conditional volatility and skewness** — not skewness alone, and not
volatility alone — which "significantly predicts left-tail realizations
of momentum returns, ESPECIALLY AT DAILY FREQUENCY, capturing information
about crash risk beyond volatility alone." A skewness-based dynamic
allocation built on that indicator "improves daily downside risk
management and earns significant alphas over existing momentum-timing
approaches" (i.e. beats volatility-only timing of the Barroso &
Santa-Clara / Moreira & Muir kind). They also report momentum's skewness
"cannot be fully reconciled with asymmetric market exposure" — i.e. it is
not just a restatement of Daniel & Moskowitz's dynamic-beta story, there
is genuine information in the skewness moment itself.

Amaya, Christoffersen, Jacobs & Vasquez (2015, JFE 118, "Does Realized
Skewness Predict the Cross-Section of Equity Returns?") supply the
estimator used here: realized skewness from N intraday returns over a
window,

    RSkew = sqrt(N) * sum(r_i^3) / RV^1.5,   RV = sum(r_i^2)

which is exactly the (unadjusted) sample skewness of the intraday returns
inside the window — the standard construction for turning high-frequency
bars into a moment estimate for a lower-frequency decision.

A caution found while searching, reported rather than dropped: a 2025
Finance Research Letters paper ("Cryptocurrency Market Risk-Managed
Momentum Strategies") reports that crypto momentum, unlike equities,
LACKS extended momentum crashes, and that risk-managed crypto momentum
there improves performance via augmented returns rather than downside
mitigation. That is exactly the same shape of literature-inversion this
project already found once (R-10: BTC's inverse leverage effect means
high volatility forecasts the HIGHEST forward Sharpe, the opposite of
equities, which is why v3/v4 target volatility only in the extremes
rather than continuously). So the equity-momentum-crash mechanism is not
assumed to transfer; it is tested, and the possibility that it does not
is named up front rather than discovered only after a negative result.

Mechanism, one sentence
------------------------
Hostile (very negative) realized skewness of recent 5-minute returns —
Bianchi/De Polis/Petrella's crash indicator, interacted with volatility
per their actual finding rather than used standalone — should predict the
kind of sharp reversal that inflates drawdown beyond what
``kelly_regime_v4``'s existing volatility target alone catches, so
cutting exposure when skewness turns hostile (conditional on volatility
also being elevated) should improve drawdown further.

Constraint attacked, and why "another indicator" does not apply
-----------------------------------------------------------------
SIZE, via ERR-adjacent reasoning (a crash-risk-aware haircut is a form of
tail-risk control, the same framing R-28 used for the e-process gate).
This is not "another indicator" in the sense ROUTINE.md warns against:
"another indicator" means a new predictor of *direction* — should we be
long or flat — feeding the vote, and every entry that did that (L-12,
L-14..L-23, the losing baselines) lost to fees. This proposal never
touches the vote (``frac``, the 20/40/80-day anchor gate) at all. It only
multiplies the existing exposure scale — the ``target_vol/vol`` sizer
that already answers "how much" — by a haircut in [min_mult, 1]. The vote
still decides whether to be long or flat; skewness only decides how much
of that position to actually hold. That is SIZE by the project's own
definition, not INFO.

Not a duplicate of
-------------------
- **R-08** (better volatility *forecasting*, a timescale-blended
  estimator) and **R-09** (alternative volatility *estimators*: Parkinson,
  Garman-Klass, Rogers-Satchell, Yang-Zhang): both changed the SECOND
  moment feeding the existing sizer, and both lost — R-08 sign-invertingly
  ($52K vs $115K), because a genuinely better volatility read de-levers
  BTC exactly into its highest-forward-Sharpe states (R-10's inverse
  leverage effect). That is the mistake this experiment must not repeat:
  the haircut below is built to fire on skewness, not to smuggle in a
  second, better volatility estimate — the interaction variant (mode
  "interaction") uses the EXISTING vol/slow-vol ratio only as a gate
  condition, unchanged from v3, never as a replacement input to sizing.
  This experiment introduces a THIRD moment nobody here has used; R-08/R-09
  are both still purely about the second.
- **R-28 / R-31** (e-process evidence gate on drift, unified Kelly sizer):
  a completely different mechanism — anytime-valid sequential testing of
  the null "drift is zero" via a wealth/e-process, gating on accumulated
  evidence about the FIRST moment (mean). Nothing here tests a null or
  accumulates evidence; the skew haircut is a direct, memoryless function
  of an estimated third moment. R-31's finding (that R-28's apparent risk
  edge was entirely an artifact of holding less exposure, which vanished
  once risk was matched) is a methodological warning this experiment
  takes seriously — see the causality/exposure discipline below and the
  explicit comparison against `kelly_regime_v4` at its OWN unmodified
  exposure level (the haircut only ever multiplies exposure down from the
  baseline, by construction, so there is no matched-risk ambiguity to
  create in the first place: the haircut variants can only draw down
  LESS than v4 given the same vote and vol, never hold more).
- **L-02 (v2, convex vote)** and **L-03 (v3, conditional vol targeting)**:
  both modify how the EXISTING inputs (the vote fraction, the volatility
  ratio) are used — v2 raises the vote to a power, v3 switches between
  continuous and extremes-only vol targeting. Neither introduces a new
  statistic computed from the price series. This experiment computes a
  third moment (skewness) that does not exist anywhere else in the
  codebase and multiplies it in as an independent factor.

Simulable here
---------------
Yes. Rolling realized skewness of 5-minute log returns, OHLCV close only,
no new data source, ``.shift(1)``'d so bar i's skewness uses only returns
through bar i-1 (the same convention the incumbent's volatility estimator
already uses).

Pre-registered falsification test (chosen now, before any tuning)
--------------------------------------------------------------------
ETH, Bitfinex, the R-17/R-28/R-31 window (``data/{btcusd,ethusd}
_bitfinex_5m.csv.gz``, 2016-03 to 2019-12). The frozen candidate's max
drawdown, on both spot and 5x futures, must be no worse than
`kelly_regime_v4`'s own drawdown on that same window PLUS 5 percentage
points (the identical tolerance R-28/R-31 pre-registered for their ETH
checks, kept unchanged for comparability). If the skew haircut makes
ETH's drawdown WORSE than the v4 baseline by more than 5pp on either
market, the idea is falsified: a mechanism this project would call
"real" should not need a different asset's regime-specific skewness
distribution to keep working, since the whole claim is about a
market-structural property of crash dynamics, not a BTC-only pattern.

Variants
--------
``mode="off"``          v4 exactly, haircut multiplier pinned at 1.0.
                         Used only as an internal parity check.
``mode="hard"``         binary cutoff: haircut to ``hard_mult`` when
                         skewness crosses below ``skew_in``, latched
                         until it recovers above ``skew_out``
                         (hysteresis, the same pattern v3/v4 already use
                         on the volatility axis).
``mode="continuous"``   multiplicative haircut linear in how hostile
                         skewness is: 1.0 at/above ``skew_ceiling``,
                         floored at ``min_mult`` at/below ``skew_floor``.
                         Analogous to how volatility already scales
                         exposure continuously in the base `kelly_regime`.
``mode="interaction"``  Bianchi/De Polis/Petrella's actual finding: the
                         hard cutoff only fires when skewness is hostile
                         AND the existing vol/slow-vol ratio is ALSO
                         elevated (the v3 "high" breakout state) — the
                         interaction term, not either moment alone.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tradebot.strategy import Context, Strategy  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


def realized_skew(log_ret: pd.Series, window_bars: int) -> pd.Series:
    """Amaya et al. (2015) realized skewness over a trailing window, shift(1)'d.

    ``sqrt(n) * sum(r^3) / (sum(r^2))^1.5`` — the sample skewness of the
    ``window_bars`` most recent log returns. The ``.shift(1)`` means the
    value attached to bar ``i`` is computed from returns strictly BEFORE
    bar ``i`` closes: it never includes bar i's own return, matching the
    incumbent's ``vol...shift(1)`` convention exactly.
    """
    r2 = log_ret.pow(2).rolling(window_bars).sum()
    r3 = log_ret.pow(3).rolling(window_bars).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.sqrt(window_bars) * r3 / r2.pow(1.5)
    return skew.shift(1)


class SkewGatedKelly(Strategy):
    """kelly_regime_v4 with an additional exposure haircut driven by realized skewness.

    Self-contained (does not import or subclass the registered v4 class,
    matching the convention `experiments/matched_risk.py` already set) so
    parity with v4 can be checked directly: with ``mode="off"`` this
    strategy's `target` column must be bit-identical to
    ``kelly_regime_v4``'s.
    """

    name = "skew_gated_kelly"
    warmup = 100 * BARS_PER_DAY + 10  # matches v4: covers the 80d anchor + skew window headroom

    def __init__(
        self,
        mode: str = "continuous",
        # --- the vote gate (v4's, unchanged)
        horizons: tuple[int, ...] = (20, 40, 80),
        band: float = 0.01,
        # --- the shared sizer (v4's conditional targeting, unchanged)
        target_vol: float = 0.55,
        max_leverage: float = 2.0,
        vol_span: int = 8 * BARS_PER_DAY,
        deadband: float = 0.10,
        anchor_span_days: int = 180,
        high_in: float = 1.70,
        high_out: float = 1.20,
        low_in: float = 0.55,
        low_out: float = 0.85,
        # --- the skewness haircut (new)
        skew_window_days: float = 10.0,
        skew_in: float = -3.0,
        skew_out: float = -1.5,
        skew_floor: float = -6.0,
        skew_ceiling: float = 0.0,
        min_mult: float = 0.3,
        hard_mult: float = 0.3,
    ) -> None:
        if mode not in ("off", "hard", "continuous", "interaction"):
            raise ValueError(f"mode must be off/hard/continuous/interaction, got {mode!r}")
        self.mode = mode
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        self.skew_window_days = skew_window_days
        self.skew_in, self.skew_out = skew_in, skew_out
        self.skew_floor, self.skew_ceiling = skew_floor, skew_ceiling
        self.min_mult = min_mult
        self.hard_mult = hard_mult

    # --------------------------------------------------------------- the vote

    def _vote(self, close: pd.Series) -> np.ndarray:
        """Latched multi-anchor vote — v4's gate, unchanged."""
        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=close.index,
            )
            votes.append(v.ffill().fillna(0.0))
        return (sum(votes) / len(votes)).to_numpy()

    # -------------------------------------------------------------- the sizer

    def _scale(self, vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """v3/v4's conditional (extremes-only) volatility target. Returns (scale, ratio)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            full = np.minimum(self.target_vol / vol, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)

        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                    min_periods=BARS_PER_DAY).mean().to_numpy())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        out = np.empty(len(vol))
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        for i in range(len(vol)):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            out[i] = full[i] if state != 0 else steady[i]
        return out, ratio

    # ------------------------------------------------------------- the haircut

    def _skew_mult(self, skew: np.ndarray, vol_ratio: np.ndarray) -> np.ndarray:
        n = len(skew)
        if self.mode == "off":
            return np.ones(n)

        if self.mode == "continuous":
            span = self.skew_ceiling - self.skew_floor
            with np.errstate(invalid="ignore"):
                x = (skew - self.skew_floor) / span
            x = np.clip(np.nan_to_num(x, nan=1.0), 0.0, 1.0)
            mult = self.min_mult + (1.0 - self.min_mult) * x
            # No skew estimate yet (warmup remainder) -> no haircut, not max haircut.
            return np.where(np.isfinite(skew), mult, 1.0)

        # "hard" and "interaction" share the same latch; interaction adds a
        # volatility-breakout condition to the trigger, per Bianchi/De
        # Polis/Petrella's actual finding that the INTERACTION predicts
        # crash risk, not skewness alone.
        mult = np.ones(n)
        state = 0
        for i in range(n):
            x = skew[i]
            if np.isfinite(x):
                hostile = x < self.skew_in
                if self.mode == "interaction":
                    r = vol_ratio[i]
                    hostile = hostile and np.isfinite(r) and r > self.high_in
                if state == 0 and hostile:
                    state = 1
                elif state == 1 and x > self.skew_out:
                    state = 0
            mult[i] = self.hard_mult if state == 1 else 1.0
        return mult

    # ----------------------------------------------------------------- strategy

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        frac = self._vote(close)
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        scale, ratio = self._scale(vol)

        skew_window_bars = int(self.skew_window_days * BARS_PER_DAY)
        skew = realized_skew(r, skew_window_bars).to_numpy()
        mult = self._skew_mult(skew, ratio)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = frac[i] * scale[i] * mult[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["skew"] = skew
        df["skew_mult"] = mult
        df["vol_ratio"] = ratio
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# ============================================================================
# Runner. Iterate only on inner-train (<=2020-12-31) and inner-validation
# (2021-01-01..2022-12-31), per ROUTINE.md step 3. Nothing below reads
# OOS_START = "2023-01-01" or later, and the ``holdout()`` function at the
# bottom is written for the operator's later use and is NOT called by this
# session — see the assignment: "stop before step 4".
# ============================================================================

from scripts.experiment import ev  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"


def v4_baseline(**kw):
    """kelly_regime_v4 reproduced via mode="off" for an apples-to-apples baseline."""
    return SkewGatedKelly(mode="off", **kw)


def sweep_step3() -> None:
    """The step-3 configuration sweep. Every call here is one counted configuration."""
    from tradebot.strategies.buy_and_hold import BuyAndHold

    configs = []
    configs.append(("baseline v4 (mode=off)", dict(mode="off")))
    for w in (3, 5, 10, 20):
        configs.append((f"hard w={w}d", dict(mode="hard", skew_window_days=w)))
    for w in (3, 5, 10, 20):
        configs.append((f"continuous w={w}d", dict(mode="continuous", skew_window_days=w)))
    for w in (3, 5, 10, 20):
        configs.append((f"interaction w={w}d", dict(mode="interaction", skew_window_days=w)))
    # Threshold neighbourhood around the w=10d default, hard mode.
    for skew_in, skew_out in ((-2.0, -1.0), (-3.0, -1.5), (-4.0, -2.0)):
        configs.append((f"hard thr=({skew_in},{skew_out})",
                         dict(mode="hard", skew_window_days=10, skew_in=skew_in, skew_out=skew_out)))
    # hard_mult / min_mult neighbourhood.
    for m in (0.15, 0.3, 0.5):
        configs.append((f"hard mult={m}", dict(mode="hard", skew_window_days=10, hard_mult=m)))
        configs.append((f"continuous min_mult={m}", dict(mode="continuous", skew_window_days=10, min_mult=m)))

    print(f"{len(configs)} configurations x 2 markets x 2 splits "
          f"= {len(configs) * 4} backtests\n")

    print("=== inner-train (<= 2020-12-31), spot ===")
    for tag, kw in configs:
        ev(SkewGatedKelly(**kw), df=DF, market=SPOT, tag=tag, end=INNER_TRAIN_END)
    print("\n=== inner-train (<= 2020-12-31), futures 5x ===")
    for tag, kw in configs:
        ev(SkewGatedKelly(**kw), df=DF, market=FUTURES, tag=tag, end=INNER_TRAIN_END)
    print("\n=== inner-validation (2021-01-01..2022-12-31), spot ===")
    for tag, kw in configs:
        ev(SkewGatedKelly(**kw), df=DF, market=SPOT, tag=tag, start=INNER_VAL_START, end=INNER_VAL_END)
    print("\n=== inner-validation (2021-01-01..2022-12-31), futures 5x ===")
    for tag, kw in configs:
        ev(SkewGatedKelly(**kw), df=DF, market=FUTURES, tag=tag, start=INNER_VAL_START, end=INNER_VAL_END)

    print("\n=== buy_and_hold, both splits, spot ===")
    ev(BuyAndHold(), df=DF, market=SPOT, tag="hold IT", end=INNER_TRAIN_END)
    ev(BuyAndHold(), df=DF, market=SPOT, tag="hold IV", start=INNER_VAL_START, end=INNER_VAL_END)


def parity_check() -> None:
    """mode="off" must reproduce kelly_regime_v4 exactly (bit-identical target column)."""
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

    df = DF.iloc[-300_000:].copy()
    a = KellyRegimeV4().prepare(df.copy())
    b = SkewGatedKelly(mode="off").prepare(df.copy())
    diff = np.abs(a["target"].to_numpy() - b["target"].to_numpy())
    worst = float(np.nanmax(diff))
    print(f"parity (mode=off vs kelly_regime_v4): max |target difference| = {worst:.3e} "
          f"{'PASS' if worst < 1e-12 else 'FAIL'}")


def causality() -> None:
    """By-hand two-opposite-tampers probe (matches run_eprocess.py's causality()).

    Unregistered strategies get none of test_causality_strict.py's
    protection (it only parametrizes over the registry), so this repeats
    the procedure by hand: bars after a cut are multiplied by 3 in one
    copy and divided by 3 in the other; every decision at or before the
    cut must be identical, and the prepared columns must match bit-for-bit
    before the cut too (the check a plain truncation test cannot catch: a
    full-series mean/std/quantile leaking into early rows).
    """
    from tradebot.broker import PaperBroker

    df = DF.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    for mode in ("hard", "continuous", "interaction"):
        def decisions(frame, mode=mode):
            s = SkewGatedKelly(mode=mode, **FROZEN_KW_NO_MODE)
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
        print(f"[{mode}] tampered from bar {cut:,} of {len(df):,}; checked bars {bars}")
        print(("  FAIL - reads the future at bars " + str(bad)) if bad
              else "  PASS - every decision at or before the cut is unchanged")

        pa = SkewGatedKelly(mode=mode, **FROZEN_KW_NO_MODE).prepare(up.copy())
        pb = SkewGatedKelly(mode=mode, **FROZEN_KW_NO_MODE).prepare(down.copy())
        for col in ("target", "skew", "skew_mult", "vol_ratio"):
            diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
            worst = float(np.nanmax(diff))
            print(f"    column {col:10s} max |difference| before the cut = {worst:.3e}"
                  f"  {'PASS' if worst < 1e-12 else 'FAIL'}")


def falsification_eth() -> None:
    """Pre-registered falsification test: does the frozen candidate's drawdown
    on ETH (Bitfinex, R-17 window) stay within v4's own drawdown + 5pp?

    This reads data/{btcusd,ethusd}_bitfinex_5m.csv.gz, a separate,
    pre-2020 window (2016-03..2019-12) that R-17/R-28/R-31 already
    established is fine to use for falsification without touching the
    2023+ BTC holdout.
    """
    from tradebot.data import load_ohlcv_csv
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

    btc = load_ohlcv_csv(ROOT / "data" / "btcusd_bitfinex_5m.csv.gz")
    eth = load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")

    print(f"BTC control: {len(btc):,} bars {btc.index[0]} -> {btc.index[-1]}")
    print(f"ETH control: {len(eth):,} bars {eth.index[0]} -> {eth.index[-1]}")

    for label, df in (("BTC", btc), ("ETH", eth)):
        for market_name, market in (("spot", SPOT), ("futures5x", FUTURES)):
            m_v4 = ev(KellyRegimeV4(), df=df, market=market, tag=f"{label} {market_name} v4")
            m_sk = ev(SkewGatedKelly(**FROZEN), df=df, market=market, tag=f"{label} {market_name} skew")
            allowance = m_v4.max_drawdown_pct + 5.0
            verdict = "PASS" if m_sk.max_drawdown_pct <= allowance else "FAIL"
            print(f"  -> v4 DD={m_v4.max_drawdown_pct:.1f}% allowance<= {allowance:.1f}% "
                  f"skew DD={m_sk.max_drawdown_pct:.1f}%  {verdict}")


# ---------------------------------------------------------------------------
# Frozen configuration (step 4 pre-registration). See the session report for
# the exact selection reasoning; do NOT modify these values to chase a
# holdout result — that is the "moving the goalposts" failure ROUTINE.md
# names explicitly.
# ---------------------------------------------------------------------------
FROZEN_KW_NO_MODE = dict(
    horizons=(20, 40, 80), band=0.01,
    target_vol=0.55, max_leverage=2.0, vol_span=8 * BARS_PER_DAY, deadband=0.10,
    anchor_span_days=180, high_in=1.70, high_out=1.20, low_in=0.55, low_out=0.85,
    skew_window_days=20.0, skew_in=-3.0, skew_out=-1.5,
    skew_floor=-6.0, skew_ceiling=0.0, min_mult=0.3, hard_mult=0.3,
)
FROZEN = dict(mode="hard", **FROZEN_KW_NO_MODE)


def holdout() -> None:  # pragma: no cover - reserved for the operator, NOT run this session
    """Step 4. NOT executed in this session (assignment: "stop before step 4").

    Left here, frozen, for the operator to run once after synthesizing
    both parallel branches:

        ev(SkewGatedKelly(**FROZEN), market=SPOT, start=OOS_START)
        ev(SkewGatedKelly(**FROZEN), market=FUTURES, start=OOS_START)
        ev(v4_baseline(), market=SPOT, start=OOS_START)   # kelly_regime_v4 comparison
        ev(v4_baseline(), market=FUTURES, start=OOS_START)

    Pre-registered decision rule (written before any 2023+ data was read;
    do not revise after looking — a revision downgrades the result to
    in-sample, per ROUTINE.md):

    - **P1** spot final balance beats `buy_and_hold`.
    - **P2** the improvement over `kelly_regime_v4` (not just buy_and_hold)
      is either > +0.2 Sharpe, or a max-drawdown improvement of >= 10
      percentage points on at least one market. Set at 10pp rather than
      ROUTINE's generic bar because the inner-validation win that
      justified freezing this config was itself only 3.7-5.0pp (spot
      28.2% vs 33.2%; futures 28.6% vs 32.3%) and the ETH falsification
      showed 0.0-0.9pp — a rule that would call anything under ~5pp a
      win on the BTC holdout would not be asking more of the holdout than
      the training evidence already gave it.
    - **P3 (falsification, already run in step 2/3 on ETH, NOT the BTC
      holdout)**: PASSED literally (drawdown within v4 + 5pp on both
      markets), but substantively weak — ETH drawdown was UNCHANGED
      (36.5% spot, 35.1% futures, both markets, to one decimal) while
      return fell 17-33%. Read this falsification result as "did not
      falsify the mechanism, but did not confirm a transferable drawdown
      benefit either" rather than as support.
    - **P4** the parameter neighbourhood must be closer to a plateau than
      the peak this session found: hard mode's DD improvement on
      inner-validation was non-monotonic in window length (w=5d tied or
      lost to baseline; w=20d was the standout), which is a warning sign
      already recorded honestly in the session report, not resolved here.

    **Stated prediction, before the holdout is read:** P1 likely fails or
    is a coin flip. 2023+ is characterized elsewhere in this project
    (R-31) as a bull holdout, and this session's own inner-train result
    (2017-2020, also bull-plus-2018-bear) showed the same shape: hard
    mode w=20d costs 14-20% of final balance for a <1pp drawdown change.
    If 2023+ behaves like inner-train rather than like inner-validation's
    2021-top/2022-bear regime, P1 fails for the same structural reason
    R-31 found for the e-process gate: a haircut calibrated on a
    regime-change window does not have a regime change to protect
    against in a trending holdout, and only pays its cost.
    """
    raise RuntimeError(
        "holdout() is reserved for the operator; this session does not call it "
        "(see the assignment's hard constraint: do not read 2023-01-01 onward)."
    )


if __name__ == "__main__":
    cmds = {"sweep": sweep_step3, "parity": parity_check, "causality": causality,
            "falsify": falsification_eth}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        t0 = time.time()
        cmds[choice]()
        print(f"\n[{time.time() - t0:.0f}s]", file=sys.stderr)
    else:
        print(f"usage: python experiments/skew_gate_kelly.py [{'|'.join(cmds)}]")
