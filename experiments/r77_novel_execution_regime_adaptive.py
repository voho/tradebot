#!/usr/bin/env python
"""R-77 (novel branch): regime-adaptive execution urgency for kelly_regime_v4.

Not registered: lives under ``experiments/`` so it is not auto-discovered,
per ROUTINE.md step 5. Does not modify ``kelly_regime_v4.py``,
``kelly_regime_v3.py``, ``kelly_regime.py``, ``engine.py`` or ``broker.py``
-- all reused read-only as libraries. Does not modify either of R-56's own
files (``experiments/kelly_regime_exec_limit_conservative.py`` /
``_novel.py``), which this module only *reads* for their causal-reuse
pattern. Zero overlap with the sibling parallel branch's file
(``experiments/r77_conservative_narrowcap_execution.py``, B-24's narrower
fixed-N re-pre-registration) -- not read, not touched.

THE IDEA, ONE SENTENCE
-----------------------
R-56 found that a resting limit order with ANY fixed patience N >= 3 bars
delays v4's flip-to-flat de-risking events during crash transitions (124/128
events beyond a 1-2 bar tolerance for large N) even though its fee-saving
benefit is real and monotonic in N -- so instead of picking a smaller fixed
N and hoping (B-24's re-pre-registration, being tested in parallel this
round), this branch makes the patience ITSELF a causal function of the
market's own regime/stress state, collapsing to near-immediate execution
specifically when the pending order is a de-risking move during elevated
volatility -- exactly the situation R-56 diagnosed as dangerous -- while
staying patient (capturing the maker-fee saving) the rest of the time.

CONSTRAINT ATTACKED: COST (costs that scale with the signal), same
constraint as R-56, via the same "change HOW an already-decided trade
fills, not WHEN or how MUCH" mechanism. Not a duplicate of L-05/L-06 (WHEN
to trade), R-12/R-13 (fee TIER sweeps), R-40 (the SIZE/vote signal itself),
or R-56 (which tested exactly two FIXED-patience fill models, conservative
100%-fill-on-touch and novel penetration-probability). This round's
difference from R-56, stated precisely: patience N is no longer a single
frozen number chosen before the backtest -- it is `N(i)`, a causal function
of bar i's own regime state, chosen so the danger R-56 quantified
(de-risking lag during crash transitions) is targeted directly by the
mechanism rather than avoided statistically by capping N (B-24's approach,
running in parallel this round on the sibling file).

THE STRESS PROXY -- reused, not invented
------------------------------------------
`kelly_regime_v3.prepare()` (inherited unchanged by v4) already computes,
every bar, entirely causally:

    vol[i]  = EWM(log-return, span=vol_span).std() * sqrt(BARS_PER_YEAR), shift(1)
    slow[i] = EWM(vol, span=anchor_span_days * BARS_PER_DAY).mean()
    ratio[i] = vol[i] / slow[i]

`ratio[i]` is v4's own realized-volatility-over-its-own-trend ratio -- the
exact quantity v4's regime-latch state machine thresholds against
(`high_in=1.70`, `high_out=1.20`, `low_in=0.55`, `low_out=0.85`) to decide
whether it is in a volatility breakout. It is not exposed as a column by
`prepare()`, so it is reproduced here read-only from the published formula,
the identical pattern R-56's own novel branch used for `scale[i]`
(`compute_v4_signal` in `kelly_regime_exec_limit_novel.py`). Nothing new is
computed from the data that v4 did not already compute from it; this round
reuses v4's own stress measurement rather than inventing a second one,
which is also why v4's own thresholds (1.20, 1.70) are used below as
GROUNDED, not arbitrary, sweep points for this round's own trigger.

THE MECHANISM, EXACT FORMULA (pre-registered before any run)
----------------------------------------------------------------
For each rebalance decision at bar i (`target[i] != target[i-1]`, v4's own
already-causal, already-hysteresis-gated signal, untouched here):

    is_derisking(i) = target[i] < target[i-1] - eps      # any exposure cut
                                                            # (v4's vote frac
                                                            # in [0,1] means
                                                            # target is never
                                                            # negative, so
                                                            # "de-risking" here
                                                            # is always
                                                            # flip-toward-flat,
                                                            # never
                                                            # flip-to-short --
                                                            # noted honestly,
                                                            # see "deviation"
                                                            # section below)

    if is_derisking(i) and ratio[i] >= s_override:
        N(i) = n_min                                       # crash-transition
                                                            # override: collapse
                                                            # to near-immediate
                                                            # taker
    else:
        urgency_mult(i) = 1 + kappa * max(0, ratio[i] - 1)
        N(i) = max(n_min, round(n_base / urgency_mult(i)))  # Almgren-Chriss-
                                                            # style: patience
                                                            # shrinks smoothly
                                                            # as realized vol
                                                            # runs hot over its
                                                            # own trend

Citations for the functional form: Almgren & Chriss (2000, "Optimal
Execution of Portfolio Transactions", J. Risk 3(2); 2001, "Optimal
Liquidation of a Stock Portfolio", J. Portfolio Management) formalize
execution urgency as a trade-off between market-impact cost (favors
patience) and timing/volatility risk of resting (favors speed) -- the
optimal trading horizon shrinks as volatility risk rises relative to
impact cost, which is exactly what `urgency_mult` implements: patience
divided by a factor increasing in excess realized vol. Cartea, Jaimungal &
Penalva (2015, *Algorithmic and High-Frequency Trading*, Cambridge
University Press, ch. 6) generalize this to inventory-risk-averse
execution where posting aggressiveness scales with the trader's own
volatility/inventory-risk estimate -- the discrete "collapse near-
immediately when a risk-reducing trade meets elevated volatility" override
above is this round's application of that idea to the ONE case R-56 showed
is actually dangerous to delay (a de-risking flip during a vol spike),
rather than scaling every order uniformly.

Fill mechanism (the R-56 conservative vs. novel choice, stated and
justified): this round reuses R-56 CONSERVATIVE branch's 100%-fill-on-touch
model (post a resting limit at the decision bar's close; the first later
bar whose [low, high] touches it fills in full at the maker fee; forced
taker fallback at the deadline bar's open) rather than R-56 NOVEL's
Cont & Kukanov (2017) penetration-probability model. Justification: R-56
novel's own ablations already showed (a) the literature-grounded
probability-of-fill discount underperforms a flat, arbitrary discount in
9/9 head-to-head comparisons, and (b) BOTH of R-56's fixed-N variants lose
to the always-taker baseline in every slice once fill uncertainty is
priced in -- so stacking a second uncertain-fill layer on top of an
execution-URGENCY mechanism this round is actually trying to isolate would
make it impossible to tell whether a negative result comes from the timing
idea or from fill-probability pessimism. Holding the fill model at R-56
conservative's (optimistic, but the SAME optimism this project has already
measured and can compare against directly) isolates the one new variable:
adaptive vs. fixed patience.

WHAT THIS ROUND DELIBERATELY DOES NOT MODEL: same as R-56 -- no queue
position, no order-book depth, no informed-flow avoidance beyond the
touch-based fill rule inherited from the conservative branch. See that
file's own "WHAT IS NOT MODELLED" section; unchanged here.

DEVIATION FROM THE BRIEF, STATED PLAINLY: the brief's candidate mechanism
list mentions "the pending order is itself a flip toward flat/short in the
bearish direction". For `kelly_regime_v4` specifically this collapses to
flip-toward-flat only: `frac[i]` (the crowd-regime vote) is constructed in
`kelly_regime_v3.prepare()`/`kelly_regime.py` as an average of {0.0, 1.0}
votes, always in [0, 1], and `scale[i] >= 0`, so `target[i] = frac[i] *
scale[i]` can never be negative -- v4 never shorts, on spot OR futures (its
own docstring: "the strategy stands flat rather than shorting a
historically upward-drifting asset"). This is verified, not assumed: see
`_assert_v4_never_shorts` in the causality-probe section below. The
mechanism above is written for "de-risking = any exposure reduction",
which is the only form this can take for this specific strategy.

PRE-REGISTERED FALSIFICATION TESTS (before any run)
------------------------------------------------------
1. ETH falsification (Bitfinex, entirely pre-2020,
   ``data/ethusd_bitfinex_5m.csv.gz``) -- does the fee-saving / Sharpe
   pattern replicate directionally (same sign as inner-train)?
2. BTC pre-2020 control window (Bitfinex,
   ``data/btcusd_bitfinex_5m.csv.gz``, entirely pre-2020) -- same
   directional-replication bar.
3. Crash-transition-lag test, same construction as R-56's: for every
   flip-to-flat event (``target[i-1] > 0.05`` and ``target[i] < 1e-9``) in
   inner-train + inner-validation, how many bars until the position is
   fully flat, vs. the always-taker baseline's fixed 1 bar. PASS = mean/max
   lag inside the project's 1-2 bar tolerance, no systematic blowup.
4. **This round's own thesis test.** At a fee-savings level comparable to
   a fixed-N conservative baseline in N ~ 6-12 territory (the range B-24's
   parallel branch is re-testing this same round), does the ADAPTIVE rule
   show a MATERIALLY LOWER crash-transition-lag violation rate than the
   matched FIXED-N baseline at the SAME fee-savings level? Exact
   comparison, pre-registered: run the fixed-N ablation of this SAME
   engine (`kappa=0, s_override=inf` -- i.e. `N(i) == n_base` always, which
   makes the fixed-N arm and the adaptive arm share every other assumption
   byte-for-byte, isolating adaptivity as the only difference) over
   N in {3, 6, 9, 12, 18, 24} on the same combined inner-train +
   inner-validation window; pick the N whose total fee-dollars-saved is
   closest to the chosen adaptive configuration's; compare, on the SAME
   set of flip-to-flat events, (i) the count of events lagging > 2 bars
   ("violations") and (ii) the mean lag across all events. **Bar for
   "materially lower", fixed now:** the adaptive rule passes test 4 only if
   its violation count is at least 50% lower than the matched fixed-N
   baseline's AND its mean lag is not worse (<=) than the matched fixed-N
   baseline's mean lag. Anything else (including "both zero", which proves
   nothing about the mechanism) is reported as FAIL / INCONCLUSIVE, not
   quietly counted as a pass.

PRE-REGISTERED PROMOTION DECISION RULE (before any run, applied mechanically)
--------------------------------------------------------------------------------
PROMOTE only if ALL of:
  (a) beats kelly_regime_v4 on inner-validation Sharpe by > +-0.2, or is a
      clear drawdown/tail win, on BOTH spot and futures;
  (b) ETH falsification passes directionally;
  (c) BTC pre-2020 control does not decisively fail;
  (d) crash-transition-lag test passes at the 1-2 bar threshold;
  (e) test 4 (adaptive-vs-matched-fixed-N lag comparison) confirms the
      thesis per the bar fixed above;
  (f) the swept parameter neighbourhood (n_base x kappa x s_override x
      n_min) is a plateau, not a lucky single point.
Anything else -> NEGATIVE. Any deviation from this plan after seeing
results is stated explicitly and downgrades the result to in-sample, per
ROUTINE.md step 4 -- not applied quietly.

DATA DISCIPLINE
----------------
Every backtest in this file is restricted to:
  - inner-train:       2017-01-01 -> 2020-12-31 (BTC, committed spot file)
  - inner-validation:  2021-01-01 -> 2022-12-31 (BTC, committed spot file)
  - combined window (crash-lag / test-4 only): 2017-01-01 -> 2022-12-31
  - ETH falsification: data/ethusd_bitfinex_5m.csv.gz (2016-03 -> 2019-12,
    the whole file -- physically cannot contain a holdout bar)
  - BTC control:       data/btcusd_bitfinex_5m.csv.gz (2016-01 -> 2019-12,
    same property, same file R-56 used)
No code path in this file ever loads, slices past, prints, or otherwise
touches a bar dated 2023-01-01 or later; the working BTC frame is cut to
inner-validation's end immediately on load (see ``load_working_frame``),
so nothing downstream can reach a holdout bar even by accident. Grepped by
the author before finishing; the operator should re-grep independently, as
this project's own practice requires.

USAGE
-----
    python experiments/r77_novel_execution_regime_adaptive.py causality
    python experiments/r77_novel_execution_regime_adaptive.py phase1
    python experiments/r77_novel_execution_regime_adaptive.py validate
    python experiments/r77_novel_execution_regime_adaptive.py falsify
    python experiments/r77_novel_execution_regime_adaptive.py crashlag
    python experiments/r77_novel_execution_regime_adaptive.py test4
    python experiments/r77_novel_execution_regime_adaptive.py all      # everything, in order
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.data import load_ohlcv_csv, load_dataset  # noqa: E402
from tradebot.engine import BacktestResult, build_trades, run_backtest, validate_ohlcv  # noqa: E402
from tradebot.metrics import Metrics, compute_metrics  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import prefix_bars, run_period  # noqa: E402

DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------- data guard
OOS_START = "2023-01-01"           # exclusive-lower-bound sentinel only, never a data read
INNER_TRAIN = ("2017-01-01", "2020-12-31")
INNER_VAL = ("2021-01-01", "2022-12-31")
COMBINED_WINDOW = ("2017-01-01", "2022-12-31")     # inner-train U inner-validation, for crash-lag/test-4
BTC_CONTROL = ("2017-01-01", "2019-12-31")          # pre-2020 control window (on the Bitfinex control file)

# Bitstamp fee schedule, per R-56 (verified via web search 2026-08-20; unchanged
# since, reused rather than re-fetched).
TAKER_ENTRY = 0.0040
MAKER_ENTRY = 0.0030
TAKER_TOP = 0.0003
MAKER_TOP = 0.0000
FEE_TIERS = {"entry": (TAKER_ENTRY, MAKER_ENTRY), "top": (TAKER_TOP, MAKER_TOP)}

V4_WARMUP = KellyRegimeV4().warmup  # 80d anchor + 10 = 23,050 bars

SPOT = MarketSpec.spot()
FUT = MarketSpec.futures(leverage=5.0)
MARKETS = {"spot": SPOT, "futures_5x": FUT}

CONFIG_COUNTER = {"core": 0, "diagnostic": 0}


def _count(kind: str = "core", k: int = 1) -> None:
    CONFIG_COUNTER[kind] += k


# ============================================================ the v4 signal + stress proxy
def compute_v4_target_and_ratio(df: pd.DataFrame, strategy: KellyRegimeV4 | None = None,
                                 _peek_bug: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """v4's real, causal ``target[i]`` AND its own realized-vol/slow-vol ``ratio[i]``.

    ``target`` comes from the REAL registered ``KellyRegimeV4().prepare()``
    (byte-for-byte parity with the strategy this project tests and trusts --
    the SIZE/vote signal is completely untouched by this module).

    ``ratio[i] = vol[i] / slow[i]`` is NOT exposed as a column by
    ``kelly_regime_v3.prepare`` (v4's parent), so it is reproduced here from
    the published formula (kelly_regime_v3.py lines ~68-78) -- the same
    read-only-duplication pattern R-56's own novel branch used for
    ``scale[i]`` in ``compute_v4_signal``. Both ``vol`` and ``slow`` are
    already causal in the source: ``vol`` is built from an EWM of past
    returns and explicitly ``.shift(1)``'d before use, so ``ratio[i]`` uses
    no information from bar i's own return, let alone any later bar.

    ``_peek_bug=True`` deliberately breaks that shift (drops the
    ``.shift(1)``, so ``ratio[i]`` would see bar i's own return) -- used
    ONLY by the causality probe's guard-the-guard check, never in any
    reported result.
    """
    strategy = strategy or KellyRegimeV4()
    prepared = strategy.prepare(df.copy())
    target = prepared["target"].to_numpy(dtype=float)

    close = df["close"]
    r = np.log(close).diff()
    raw_vol = r.ewm(span=strategy.vol_span, min_periods=BARS_PER_DAY).std() * np.sqrt(BARS_PER_YEAR)
    vol = (raw_vol if _peek_bug else raw_vol.shift(1)).to_numpy()
    slow = (pd.Series(vol).ewm(span=strategy.anchor_span_days * BARS_PER_DAY,
                                min_periods=BARS_PER_DAY).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)

    assert len(target) == len(ratio) == len(df)
    return target, ratio


def _assert_v4_never_shorts(target: np.ndarray) -> None:
    """Verifies the docstring's "de-risking is always flip-to-flat, never
    flip-to-short" claim, rather than just asserting it in prose."""
    assert np.all(target >= -1e-9), (
        "v4's target went negative -- the 'v4 never shorts' assumption this "
        "module's de-risking definition relies on is false; investigate "
        "before trusting any de-risking-override result below")


# ============================================================ adaptive config
@dataclass(frozen=True)
class AdaptiveConfig:
    """One configuration of the regime-adaptive-patience execution model.

    ``kappa=0.0, s_override=inf`` collapses the formula to ``N(i) ==
    n_base`` for every order -- i.e. this dataclass ALSO IS the fixed-N
    ablation used for test 4's matched-fee-savings comparison, sharing
    every other assumption (fill model, fee tiers, broker) byte-for-byte
    with the adaptive arm so adaptivity is the only isolated variable.
    """

    n_base: int = 12       # patience (bars) when ratio[i] <= 1 (vol at or below its own trend)
    kappa: float = 2.0     # Almgren-Chriss urgency scaling on excess vol ratio
    s_override: float = 1.20  # ratio threshold above which a DE-RISKING order collapses to n_min
    n_min: int = 1          # floor patience (1 bar = the empty-window case -> immediate taker, R-56's N=1 identity)

    def tag(self) -> str:
        sov = "inf" if math.isinf(self.s_override) else f"{self.s_override:g}"
        return f"nbase{self.n_base}_kappa{self.kappa:g}_sov{sov}_nmin{self.n_min}"

    def n_eff(self, ratio_i: float, is_derisking: bool) -> tuple[int, bool]:
        """Returns (N(i), override_fired) -- pure function of this bar's own
        (already-causal) ratio and the direction of the pending order."""
        if is_derisking and math.isfinite(ratio_i) and ratio_i >= self.s_override:
            return self.n_min, True
        urgency_mult = 1.0 + self.kappa * max(0.0, ratio_i - 1.0) if math.isfinite(ratio_i) else 1.0
        n = int(round(self.n_base / urgency_mult))
        return max(self.n_min, n), False


def fixed_n_config(n: int) -> AdaptiveConfig:
    """The fixed-N ablation: N(i) == n for every order, same engine otherwise."""
    return AdaptiveConfig(n_base=n, kappa=0.0, s_override=math.inf, n_min=1)


EPS_TARGET = 1e-9


# ============================================================ the simulator
@dataclass
class Pending:
    target_frac: float
    limit_price: float
    placed_at: int
    deadline: int
    is_buy: bool
    is_derisking: bool
    override_fired: bool
    n_eff: int


def run_adaptive_backtest(
    df: pd.DataFrame,
    base_market: MarketSpec,
    taker_fee: float,
    maker_fee: float,
    config: AdaptiveConfig,
    start_balance: float,
    trade_start: int = 0,
    data_label: str = "",
    strategy: KellyRegimeV4 | None = None,
    target: np.ndarray | None = None,
    ratio: np.ndarray | None = None,
) -> tuple[BacktestResult, dict]:
    """Re-simulate v4's own causal target series with regime-adaptive-patience fills.

    Structurally identical to R-56 conservative branch's ``run_backtest_limit``
    (post a resting limit at the decision bar's close, 100%-fill-on-touch
    over the causal window, forced taker fallback at the deadline bar's
    open, cancel-and-replace on a fresh decision) -- reused as the fill
    mechanism per this module's own "fill mechanism" justification above.
    The one change: the patience window length is ``config.n_eff(ratio[i],
    is_derisking)`` per order, not a single frozen constant.

    ON THE (i+N) FALLBACK PRICE: same causal reading as R-56 conservative
    (see that file's own docstring section of the same name) -- the
    touch-check window is bars i+1..i+N-1 (full high/low ranges, fully known
    by the time they are used); the forced fallback fires at bar i+N's OPEN,
    using no information from that bar's own high/low. At N=1 the window is
    empty and every order falls straight to the forced-taker fallback at
    bar i+1's open -- bit-for-bit identical to the as-shipped baseline (a
    correctness check, verified below in the causality probe).
    """
    strategy = strategy or KellyRegimeV4()
    validate_ohlcv(df)
    if target is None or ratio is None:
        target, ratio = compute_v4_target_and_ratio(df, strategy)
    _assert_v4_never_shorts(target)

    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    index = df.index
    n = len(df)
    leverage = max(base_market.leverage, 1e-9)

    taker_market = replace(base_market, fee_rate=taker_fee, name=f"{base_market.name}_adaptive")
    maker_market = replace(base_market, fee_rate=maker_fee, name=f"{base_market.name}_adaptive")
    broker = PaperBroker(market=taker_market, start_balance=start_balance)

    equity = [0.0] * n
    fills = []
    pending: Pending | None = None
    events: list[dict] = []
    cancels = 0
    maker_fills = 0
    taker_fallback_fills = 0
    override_fires = 0
    n_eff_values: list[int] = []

    for i in range(n):
        ts = index[i]

        liq = broker.check_liquidation(ts, opens[i], opens[i], opens[i])
        if liq is not None:
            fills.append(liq)
            pending = None

        if pending is not None and not broker.dead:
            if i == pending.deadline:
                broker.market = taker_market
                out = broker._execute_target(pending.target_frac, ts, opens[i])  # noqa: SLF001
                fills.extend(out)
                real = bool(out)
                if real:
                    taker_fallback_fills += 1
                events.append({"placed_at": pending.placed_at, "resolved_at": i,
                                "kind": "taker_fallback" if real else "taker_fallback_dust",
                                "bars": i - pending.placed_at, "target": pending.target_frac,
                                "is_derisking": pending.is_derisking, "override_fired": pending.override_fired,
                                "n_eff": pending.n_eff, "real_fill": real})
                pending = None
            elif pending.placed_at < i < pending.deadline:
                hi, lo = highs[i], lows[i]
                touched = (lo <= pending.limit_price) if pending.is_buy else (hi >= pending.limit_price)
                if touched:
                    broker.market = maker_market
                    out = broker._execute_target(pending.target_frac, ts, pending.limit_price)  # noqa: SLF001
                    fills.extend(out)
                    real = bool(out)
                    if real:
                        maker_fills += 1
                    events.append({"placed_at": pending.placed_at, "resolved_at": i,
                                    "kind": "maker_touch" if real else "maker_touch_dust",
                                    "bars": i - pending.placed_at, "target": pending.target_frac,
                                    "is_derisking": pending.is_derisking, "override_fired": pending.override_fired,
                                    "n_eff": pending.n_eff, "real_fill": real})
                    pending = None

        liq = broker.check_liquidation(ts, opens[i], highs[i], lows[i])
        if liq is not None:
            fills.append(liq)
            pending = None

        equity[i] = broker.equity(closes[i])
        if not math.isfinite(equity[i]):
            raise ValueError(f"non-finite equity at bar {i} ({index[i]})")

        last_bar = i == n - 1
        tradable = (not broker.dead) and (not last_bar) and (i >= strategy.warmup) and (i >= trade_start)
        if tradable:
            prev_t = target[i - 1] if i > 0 else 0.0
            if abs(target[i] - prev_t) > EPS_TARGET:
                new_frac = target[i] / leverage
                old_frac = prev_t / leverage
                is_derisking = new_frac < old_frac - 1e-9
                if pending is not None:
                    cancels += 1
                    events.append({"placed_at": pending.placed_at, "resolved_at": i,
                                    "kind": "cancelled", "bars": i - pending.placed_at,
                                    "target": pending.target_frac})
                    pending = None
                n_eff, override_fired = config.n_eff(ratio[i], is_derisking)
                n_eff_values.append(n_eff)
                if override_fired:
                    override_fires += 1
                deadline = min(i + n_eff, n - 1)
                if deadline > i:
                    pending = Pending(target_frac=new_frac, limit_price=closes[i], placed_at=i,
                                       deadline=deadline, is_buy=(new_frac > old_frac),
                                       is_derisking=is_derisking, override_fired=override_fired, n_eff=n_eff)

    trades = build_trades(fills, end_price=closes[-1] if n else None, broker=broker)
    result = BacktestResult(
        strategy_name=f"{strategy.name}_adaptive[{config.tag()}]",
        market=taker_market, start_balance=start_balance, data_label=data_label,
        equity=pd.Series(equity, index=index, name="equity"),
        fills=fills, trades=trades, df=df, liquidated=broker.dead,
        fees_paid=broker.fees_paid, funding_paid=0.0,
    )
    diag = {
        "cancels": cancels, "maker_fills": maker_fills, "taker_fallback_fills": taker_fallback_fills,
        "override_fires": override_fires, "decisions": len(n_eff_values),
        "mean_n_eff": float(np.mean(n_eff_values)) if n_eff_values else float("nan"),
        "events": events, "n_bars": n,
    }
    return result, diag


def run_adaptive_period(df: pd.DataFrame, start, end, base_market: MarketSpec,
                         taker_fee: float, maker_fee: float, config: AdaptiveConfig,
                         start_balance: float = 1_000.0, data_label: str = "",
                         strategy: KellyRegimeV4 | None = None) -> tuple[BacktestResult, dict]:
    """``run_period`` analogue for the adaptive simulator -- fair warmup prefix."""
    strategy = strategy or KellyRegimeV4()
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    if hi <= lo:
        raise ValueError(f"empty period: {start!r} -> {end!r}")
    prefix = prefix_bars(df, lo, strategy.warmup)
    frame = df.iloc[lo - prefix: hi]
    result, diag = run_adaptive_backtest(frame, base_market, taker_fee, maker_fee, config, start_balance,
                                          trade_start=prefix, data_label=data_label, strategy=strategy)
    if prefix == 0:
        return result, diag
    trimmed = replace(result, equity=result.equity.iloc[prefix:], df=result.df.iloc[prefix:])
    for e in diag["events"]:
        e["placed_at_trimmed"] = e.get("placed_at", 0) - prefix
        e["resolved_at_trimmed"] = e.get("resolved_at", 0) - prefix
    return trimmed, diag


def run_taker_baseline(df: pd.DataFrame, start, end, market: MarketSpec, fee: float,
                        start_balance: float = 1_000.0, data_label: str = "") -> BacktestResult:
    """The real, unmodified kelly_regime_v4 on the real, unmodified engine."""
    m = replace(market, fee_rate=fee)
    return run_period(KellyRegimeV4(), df, start, end, market=m,
                       start_balance=start_balance, data_label=data_label)


# ============================================================ reporting helpers
def _row(tag: str, m: Metrics, diag: dict | None = None, base_fees: float | None = None) -> dict:
    out = {"tag": tag, "final_balance": m.final_balance, "profit_pct": m.profit_pct,
           "num_trades": m.num_trades, "max_dd_pct": m.max_drawdown_pct,
           "sharpe": m.sharpe, "fees_paid": m.fees_paid, "liquidated": m.liquidated}
    if diag is not None:
        total = diag["maker_fills"] + diag["taker_fallback_fills"]
        out["maker_fill_rate_pct"] = 100.0 * diag["maker_fills"] / total if total else float("nan")
        out["cancels"] = diag["cancels"]
        out["override_fires"] = diag["override_fires"]
        out["decisions"] = diag["decisions"]
        out["mean_n_eff"] = diag["mean_n_eff"]
    if base_fees is not None:
        out["fees_saved"] = base_fees - m.fees_paid
        out["fees_saved_pct"] = 100.0 * (base_fees - m.fees_paid) / base_fees if base_fees else float("nan")
    return out


def _print_row(r: dict) -> None:
    extra = ""
    if "maker_fill_rate_pct" in r:
        extra = (f" maker%={r['maker_fill_rate_pct']:>5.1f} cancel={r['cancels']:>2d} "
                  f"override={r['override_fires']:>3d} decisions={r['decisions']:>3d} "
                  f"meanN={r['mean_n_eff']:>5.1f}")
    fs = f" fee$saved={r['fees_saved']:>+8.2f}({r['fees_saved_pct']:>+5.1f}%)" if "fees_saved" in r else ""
    print(f"  {r['tag']:44s} final=${r['final_balance']:>11,.1f} ({r['profit_pct']:>+8.1f}%) "
          f"trades={r['num_trades']:>4d} DD={r['max_dd_pct']:>5.1f}% sharpe={r['sharpe']:>5.2f} "
          f"fees=${r['fees_paid']:>8.2f}{fs}{extra}{' LIQUIDATED' if r['liquidated'] else ''}")


# ============================================================ data loading (discipline-checked)
def load_working_frame() -> pd.DataFrame:
    """Full committed BTC CSV, immediately cut to inner-validation's end.

    Every function below is handed this frame (or a sub-slice of it) --
    never the raw load -- so no bar dated OOS_START or later can reach any
    computation, print, or report in this module.
    """
    df, label = load_dataset(DATA_DIR, "spot")
    cut = df.loc[:pd.Timestamp(INNER_VAL[1], tz="UTC")]
    del df
    assert cut.index[-1] < pd.Timestamp(OOS_START, tz="UTC"), (
        "data discipline violated: a holdout bar leaked into the working frame")
    return cut, label


# ============================================================ causality probe
def _synthetic_frame(n: int, base_price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": base_price, "high": base_price, "low": base_price,
                          "close": base_price, "volume": 1.0}, index=idx)


def _synthetic_ratio_peek_check() -> bool:
    """Guard-the-guard for the STRESS PROXY specifically (not the fill sim,
    which R-56 conservative's own probe already guards and this module's
    fill loop is a direct copy of): a deterministic scenario where dropping
    ``vol``'s ``.shift(1)`` (the ``_peek_bug=True`` path in
    ``compute_v4_target_and_ratio``) lets ``ratio[i]`` see bar i's own
    return, changing whether the de-risking override fires at a
    hand-constructed bar where it should not (correct) vs. does (buggy).
    """
    # Both vol's and slow's EWMs use a hardcoded min_periods=BARS_PER_DAY
    # (288, from kelly_regime_v3.prepare's own formula) regardless of span,
    # so slow's own warmup alone needs ~288+288=576 bars before ratio is
    # non-NaN at all -- n must clear that with real margin around `spike`.
    n = 1200
    idx = pd.date_range("2018-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    # Quiet random walk everywhere except one deliberate, large return at
    # bar `spike`, engineered so only bar i == spike's OWN return crosses
    # the override threshold if peeked -- the causal read must not see it
    # until the NEXT bar.
    rets = rng.normal(0.0, 0.0003, size=n)
    spike = 900
    rets[spike] = 0.08  # one huge 5-minute return
    price = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"open": price, "high": price * 1.0001, "low": price * 0.9999,
                        "close": price, "volume": 1.0}, index=idx)

    strat = KellyRegimeV4()
    strat.vol_span = 12   # a tiny span so the spike dominates raw_vol immediately
    _, ratio_ok = compute_v4_target_and_ratio(df, strat, _peek_bug=False)
    _, ratio_bug = compute_v4_target_and_ratio(df, strat, _peek_bug=True)
    # The correct (shifted) series must not react to bar `spike`'s own
    # return until bar spike+1; the buggy (unshifted) series reacts at
    # bar `spike` itself.
    ok_reacts_early = math.isfinite(ratio_ok[spike]) and ratio_ok[spike] > 3.0
    bug_reacts_early = math.isfinite(ratio_bug[spike]) and ratio_bug[spike] > 3.0
    caught = (not ok_reacts_early) and bug_reacts_early
    print(f"  synthetic ratio-peek: correct ratio[spike]={ratio_ok[spike]!s} "
          f"buggy ratio[spike]={ratio_bug[spike]!s}")
    print(f"  guard-the-guard (stress proxy, synthetic, deterministic): {'PASS' if caught else 'FAIL'}")
    return caught


def causality_probe(df: pd.DataFrame, config: AdaptiveConfig, market: MarketSpec) -> bool:
    """Truncation/tamper probe, R-56 conservative's own pattern: two OPPOSITE
    tampers (x3 / /3 on OHLC, x7 / /7 on volume) from a cut bar onward. Every
    order whose deadline is at or before the cut must fill identically under
    both tampers -- nothing before the cut differs between them, and
    ``ratio[i]`` for i < cut is unaffected (it depends only on returns up to
    and including bar i-1, per ``.shift(1)``). Orders whose deadline lands
    AFTER the cut may (and do) diverge -- that is the real alternative
    outcome a resting order would see, not a leak.
    """
    cut = len(df) - 5_000
    print(f"\ncausality probe: frame={len(df)} bars, cut at bar {cut} ({df.index[cut]})")

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    ok = True
    res_up, diag_up = run_adaptive_backtest(up, market, TAKER_ENTRY, MAKER_ENTRY, config, 1_000.0)
    res_down, diag_down = run_adaptive_backtest(down, market, TAKER_ENTRY, MAKER_ENTRY, config, 1_000.0)
    pre_cut_up = [{k: v for k, v in e.items() if k != "resolved_at"} for e in diag_up["events"]
                  if e["resolved_at"] < cut]
    pre_cut_down = [{k: v for k, v in e.items() if k != "resolved_at"} for e in diag_down["events"]
                    if e["resolved_at"] < cut]
    match_events = pre_cut_up == pre_cut_down
    fills_up = [(f.ts, f.side, round(f.qty, 8), round(f.price, 6), round(f.fee, 8))
                for f in res_up.fills if f.ts < df.index[cut]]
    fills_down = [(f.ts, f.side, round(f.qty, 8), round(f.price, 6), round(f.fee, 8))
                  for f in res_down.fills if f.ts < df.index[cut]]
    match_fills = fills_up == fills_down
    print(f"  pre-cut order events identical under up/down tamper: {match_events} "
          f"({len(pre_cut_up)} events)")
    print(f"  pre-cut fills identical under up/down tamper: {match_fills} ({len(fills_up)} fills)")
    ok = ok and match_events and match_fills

    diverges_after = round(res_up.equity.iloc[-1], 2) != round(res_down.equity.iloc[-1], 2)
    print(f"  post-cut final equity differs (proves the probe isn't vacuous): {diverges_after} "
          f"(up=${res_up.equity.iloc[-1]:,.2f} down=${res_down.equity.iloc[-1]:,.2f})")
    ok = ok and diverges_after

    ratio_caught = _synthetic_ratio_peek_check()
    ok = ok and ratio_caught

    # N=1 sanity: n_base=1, kappa=0, s_override=inf, n_min=1 must reduce
    # bit-for-bit to the as-shipped, always-taker baseline (same identity
    # R-56 conservative's own N=1 case establishes).
    identity_cfg = fixed_n_config(1)
    base = run_taker_baseline(df, None, None, market, TAKER_ENTRY, data_label="probe")
    lim1, _ = run_adaptive_backtest(df, market, TAKER_ENTRY, MAKER_ENTRY, identity_cfg, 1_000.0)
    n1_match = round(base.equity.iloc[-1], 6) == round(lim1.equity.iloc[-1], 6)
    print(f"  identity (n_base=1,kappa=0,s_override=inf) reduces exactly to the taker baseline: "
          f"{n1_match} (baseline=${base.equity.iloc[-1]:,.6f} adaptive=${lim1.equity.iloc[-1]:,.6f})")
    ok = ok and n1_match

    print(f"\nCAUSALITY PROBE: {'PASS' if ok else 'FAIL'}")
    _count("diagnostic", 4)   # up-tamper, down-tamper, identity, (synthetic check counted separately)
    return ok


# ============================================================ Phase 1 -- parameter search
def phase1_grid(df: pd.DataFrame) -> tuple[list[dict], AdaptiveConfig]:
    """Inner-train, spot, entry-tier fees. Stage A (n_base x kappa, s_override
    fixed a priori at v4's own high_out=1.20) -> Stage B (s_override
    sensitivity at the Stage-A winner) -> Stage C (n_min sensitivity at the
    Stage-A/B winner). 6 + 2 + 1 = 9 configurations, chosen BEFORE running
    anything (this function's grid literals are the pre-registration).
    """
    print("=" * 100)
    print("PHASE 1 -- parameter search, inner-train, spot, entry tier (9 configs + 1 baseline)")
    print("=" * 100)
    start, end = INNER_TRAIN
    base = run_taker_baseline(df, start, end, SPOT, TAKER_ENTRY, data_label="real")
    base_m = compute_metrics(base)
    base_row = _row("BASELINE taker-only", base_m)
    _print_row(base_row)
    _count("core")
    rows = [dict(base_row, stage="baseline", n_base=None, kappa=None, s_override=None, n_min=None)]

    # Stage A: n_base x kappa, s_override=1.20 (v4's own high_out), n_min=1
    stage_a: list[tuple[AdaptiveConfig, dict]] = []
    for n_base in (12, 24):
        for kappa in (1.0, 2.0, 4.0):
            cfg = AdaptiveConfig(n_base=n_base, kappa=kappa, s_override=1.20, n_min=1)
            res, diag = run_adaptive_period(df, start, end, SPOT, TAKER_ENTRY, MAKER_ENTRY, cfg,
                                             data_label="real")
            m = compute_metrics(res)
            row = _row(f"A {cfg.tag()}", m, diag, base_m.fees_paid)
            _print_row(row)
            _count("core")
            row.update(stage="A", n_base=n_base, kappa=kappa, s_override=1.20, n_min=1)
            rows.append(row)
            stage_a.append((cfg, row))

    winner_cfg, winner_row = max(stage_a, key=lambda cr: cr[1]["sharpe"])
    print(f"\nStage-A winner: {winner_cfg.tag()}  sharpe={winner_row['sharpe']:.3f}")

    # Stage B: s_override sensitivity at the Stage-A winner (1.20 already run)
    stage_b: list[tuple[AdaptiveConfig, dict]] = [(winner_cfg, winner_row)]
    for sov in (1.0, 1.70):
        cfg = replace(winner_cfg, s_override=sov)
        res, diag = run_adaptive_period(df, start, end, SPOT, TAKER_ENTRY, MAKER_ENTRY, cfg, data_label="real")
        m = compute_metrics(res)
        row = _row(f"B {cfg.tag()}", m, diag, base_m.fees_paid)
        _print_row(row)
        _count("core")
        row.update(stage="B", n_base=cfg.n_base, kappa=cfg.kappa, s_override=sov, n_min=cfg.n_min)
        rows.append(row)
        stage_b.append((cfg, row))

    winner_cfg, winner_row = max(stage_b, key=lambda cr: cr[1]["sharpe"])
    print(f"\nStage-B winner: {winner_cfg.tag()}  sharpe={winner_row['sharpe']:.3f}")

    # Stage C: n_min sensitivity at the Stage-A/B winner (n_min=1 already run)
    cfg = replace(winner_cfg, n_min=2)
    res, diag = run_adaptive_period(df, start, end, SPOT, TAKER_ENTRY, MAKER_ENTRY, cfg, data_label="real")
    m = compute_metrics(res)
    row = _row(f"C {cfg.tag()}", m, diag, base_m.fees_paid)
    _print_row(row)
    _count("core")
    row.update(stage="C", n_base=cfg.n_base, kappa=cfg.kappa, s_override=cfg.s_override, n_min=2)
    rows.append(row)

    final_candidates = stage_b + [(cfg, row)]
    winner_cfg, winner_row = max(final_candidates, key=lambda cr: cr[1]["sharpe"])
    print(f"\nPHASE 1 WINNER: {winner_cfg.tag()}  sharpe={winner_row['sharpe']:.3f} "
          f"(baseline sharpe={base_m.sharpe:.3f})")
    return rows, winner_cfg


# ============================================================ full-matrix validation
def validate_final(df: pd.DataFrame, winner: AdaptiveConfig) -> list[dict]:
    """Winner config x {spot, futures_5x} x {inner-train, inner-val} x {entry, top} = 8, + 8 baselines."""
    print("\n" + "=" * 100)
    print(f"FULL-MATRIX VALIDATION -- {winner.tag()} x 2 markets x 2 periods x 2 fee tiers")
    print("=" * 100)
    rows = []
    for mname, market in MARKETS.items():
        for pname, (start, end) in (("inner-train", INNER_TRAIN), ("inner-val", INNER_VAL)):
            for tname, (taker, maker) in FEE_TIERS.items():
                base = run_taker_baseline(df, start, end, market, taker, data_label="real")
                base_m = compute_metrics(base)
                print(f"\n-- {mname} / {pname} / {tname} tier --")
                base_row = _row("BASELINE taker-only", base_m)
                _print_row(base_row)
                _count("core")
                rows.append(dict(base_row, market=mname, period=pname, tier=tname, kind="baseline"))

                res, diag = run_adaptive_period(df, start, end, market, taker, maker, winner, data_label="real")
                m = compute_metrics(res)
                row = _row(winner.tag(), m, diag, base_m.fees_paid)
                _print_row(row)
                _count("core")
                row.update(market=mname, period=pname, tier=tname, kind="adaptive",
                           sharpe_delta=m.sharpe - base_m.sharpe)
                print(f"    sharpe_delta = {row['sharpe_delta']:+.3f}")
                rows.append(row)
    return rows


# ============================================================ falsification
def falsification(winner: AdaptiveConfig) -> list[dict]:
    """Winner config x {ETH-falsification, BTC-control} x {spot, futures_5x} x entry tier = 4 + 4 baselines."""
    print("\n" + "=" * 100)
    print("FALSIFICATION -- ETH (Bitfinex, pre-2020) + BTC control (Bitfinex, pre-2020), entry tier")
    print("=" * 100)
    eth = load_ohlcv_csv(DATA_DIR / "ethusd_bitfinex_5m.csv.gz")
    btc = load_ohlcv_csv(DATA_DIR / "btcusd_bitfinex_5m.csv.gz")
    assert eth.index.max() < pd.Timestamp("2020-01-01", tz="UTC")
    assert btc.index.max() < pd.Timestamp("2020-01-01", tz="UTC")

    rows = []
    for dname, dset in (("ETH-falsification", eth), ("BTC-control", btc)):
        for mname, market in MARKETS.items():
            base_res = run_backtest(KellyRegimeV4(), dset, replace(market, fee_rate=TAKER_ENTRY),
                                     1_000.0, data_label=dname)
            base_m = compute_metrics(base_res)
            print(f"\n-- {dname} / {mname} --")
            base_row = _row("BASELINE taker-only", base_m)
            _print_row(base_row)
            _count("core")
            rows.append(dict(base_row, dataset=dname, market=mname, kind="baseline"))

            res, diag = run_adaptive_period(dset, None, None, market, TAKER_ENTRY, MAKER_ENTRY, winner,
                                             data_label=dname)
            m = compute_metrics(res)
            row = _row(winner.tag(), m, diag, base_m.fees_paid)
            _print_row(row)
            _count("core")
            row.update(dataset=dname, market=mname, kind="adaptive", sharpe_delta=m.sharpe - base_m.sharpe)
            print(f"    sharpe_delta = {row['sharpe_delta']:+.3f}")
            rows.append(row)
    return rows


# ============================================================ crash-transition-lag (falsification test 3)
def find_flip_to_flat_events(target: np.ndarray) -> list[int]:
    """Bars where v4's own target drops from meaningfully-invested to (near) zero."""
    out = []
    for i in range(1, len(target)):
        if target[i - 1] > 0.05 and target[i] < 1e-9:
            out.append(i)
    return out


def crash_lag_for_config(df: pd.DataFrame, config: AdaptiveConfig, market: MarketSpec,
                          start=None, end=None) -> tuple[list[dict], dict]:
    """For every flip-to-flat event in [start, end], bars until resolved
    (from the run's own event log), vs. the always-taker baseline's fixed 1
    bar. Returns (per-event rows, run diag)."""
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    prefix = prefix_bars(df, lo, V4_WARMUP)
    frame = df.iloc[lo - prefix: hi]
    target, ratio = compute_v4_target_and_ratio(frame)
    events = find_flip_to_flat_events(target)
    events = [e for e in events if e >= prefix]

    result, diag = run_adaptive_backtest(frame, market, TAKER_ENTRY, MAKER_ENTRY, config, 1_000.0,
                                          trade_start=prefix, target=target, ratio=ratio)
    by_placed = {e["placed_at"]: e for e in diag["events"]}
    rows = []
    for fb in events:
        e = by_placed.get(fb)
        if e is None:
            continue
        rows.append({"event_bar": fb, "event_ts": str(frame.index[fb]), "lag_bars": e["bars"],
                      "kind": e["kind"], "n_eff": e.get("n_eff"), "override_fired": e.get("override_fired")})
    return rows, diag


def falsification_crash_lag(df: pd.DataFrame, winner: AdaptiveConfig) -> list[dict]:
    print("\n" + "=" * 100)
    print(f"FALSIFICATION (3) -- crash-transition-lag, {winner.tag()}, combined window, spot, entry tier")
    print("=" * 100)
    rows, diag = crash_lag_for_config(df, winner, SPOT, start=COMBINED_WINDOW[0], end=COMBINED_WINDOW[1])
    _count("core")
    if not rows:
        print("no flip-to-flat events in this window")
        return rows
    for r in rows:
        print(f"  {r['event_ts']}  lag={r['lag_bars']:>3} bars  kind={r['kind']:16s} "
              f"n_eff={r['n_eff']}  override={r['override_fired']}")
    lags = [r["lag_bars"] for r in rows]
    violations = [r for r in rows if r["lag_bars"] > 2]
    print(f"\n{len(rows)} flip-to-flat events; mean lag={np.mean(lags):.2f} bars "
          f"max lag={max(lags)} bars; violations(>2 bars)={len(violations)}/{len(rows)} "
          f"(baseline is always 1 bar / 5 minutes)")
    return rows


# ============================================================ test 4 -- the thesis test
def thesis_test4(df: pd.DataFrame, winner: AdaptiveConfig,
                  fixed_ns: tuple[int, ...] = (3, 6, 9, 12, 18, 24)) -> dict:
    """Adaptive vs. matched fixed-N crash-transition-lag comparison, combined
    window, spot, entry tier. Pre-registered bar (restated from the module
    docstring): adaptive PASSES only if its violation count (lag > 2 bars)
    is >= 50% lower than the fee-savings-matched fixed-N baseline's AND its
    mean lag is <= the matched baseline's mean lag."""
    print("\n" + "=" * 100)
    print(f"TEST 4 (THESIS) -- {winner.tag()} vs. fee-savings-matched fixed-N, combined window, "
          "spot, entry tier")
    print("=" * 100)
    start, end = COMBINED_WINDOW
    base = run_taker_baseline(df, start, end, SPOT, TAKER_ENTRY, data_label="real")
    base_m = compute_metrics(base)
    _count("core")

    winner_rows, winner_diag = crash_lag_for_config(df, winner, SPOT, start=start, end=end)
    winner_res, _ = run_adaptive_period(df, start, end, SPOT, TAKER_ENTRY, MAKER_ENTRY, winner, data_label="real")
    winner_m = compute_metrics(winner_res)
    winner_fee_saved_pct = 100.0 * (base_m.fees_paid - winner_m.fees_paid) / base_m.fees_paid
    print(f"\nADAPTIVE {winner.tag()}: fee_saved={winner_fee_saved_pct:+.1f}%  "
          f"events={len(winner_rows)}  mean_lag={np.mean([r['lag_bars'] for r in winner_rows]):.2f}  "
          f"violations(>2)={sum(1 for r in winner_rows if r['lag_bars'] > 2)}")

    sweep_rows = []
    for nfix in fixed_ns:
        cfg = fixed_n_config(nfix)
        rows, diag = crash_lag_for_config(df, cfg, SPOT, start=start, end=end)
        res, _ = run_adaptive_period(df, start, end, SPOT, TAKER_ENTRY, MAKER_ENTRY, cfg, data_label="real")
        m = compute_metrics(res)
        fee_saved_pct = 100.0 * (base_m.fees_paid - m.fees_paid) / base_m.fees_paid
        lags = [r["lag_bars"] for r in rows]
        mean_lag = float(np.mean(lags)) if lags else 0.0
        violations = sum(1 for lag in lags if lag > 2)
        sweep_rows.append({"n": nfix, "fee_saved_pct": fee_saved_pct, "events": len(rows),
                            "mean_lag": mean_lag, "violations": violations, "sharpe": m.sharpe})
        _count("core")
        print(f"  fixed N={nfix:>3d}: fee_saved={fee_saved_pct:+6.1f}%  events={len(rows):>3d}  "
              f"mean_lag={mean_lag:>5.2f}  violations(>2)={violations:>3d}  sharpe={m.sharpe:.3f}")

    matched = min(sweep_rows, key=lambda r: abs(r["fee_saved_pct"] - winner_fee_saved_pct))
    winner_mean_lag = float(np.mean([r["lag_bars"] for r in winner_rows])) if winner_rows else 0.0
    winner_violations = sum(1 for r in winner_rows if r["lag_bars"] > 2)

    print(f"\nMatched fixed-N: N={matched['n']} (fee_saved={matched['fee_saved_pct']:+.1f}% vs. "
          f"adaptive's {winner_fee_saved_pct:+.1f}%)")
    print(f"  adaptive:  violations={winner_violations}  mean_lag={winner_mean_lag:.2f}")
    print(f"  fixed N={matched['n']}: violations={matched['violations']}  mean_lag={matched['mean_lag']:.2f}")

    violation_drop_ok = (matched["violations"] == 0 and winner_violations == 0) is False and (
        winner_violations <= 0.5 * matched["violations"] if matched["violations"] else winner_violations == 0)
    mean_lag_ok = winner_mean_lag <= matched["mean_lag"] + 1e-9
    both_zero = matched["violations"] == 0 and winner_violations == 0
    verdict = "INCONCLUSIVE (both zero violations)" if both_zero else (
        "PASS" if (violation_drop_ok and mean_lag_ok) else "FAIL")
    print(f"\nTEST 4 VERDICT: {verdict}")
    return {"winner_fee_saved_pct": winner_fee_saved_pct, "winner_mean_lag": winner_mean_lag,
            "winner_violations": winner_violations, "matched_n": matched["n"],
            "matched_fee_saved_pct": matched["fee_saved_pct"], "matched_mean_lag": matched["mean_lag"],
            "matched_violations": matched["violations"], "sweep": sweep_rows, "verdict": verdict}


# ============================================================ main
def main() -> None:
    df, label = load_working_frame()
    print(f"{len(df):,} bars  {df.index[0]} -> {df.index[-1]}  (data: {label})")
    print(f"restricted to inner-train {INNER_TRAIN} and inner-validation {INNER_VAL}\n")

    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    default_cfg = AdaptiveConfig()

    if choice in ("causality", "all"):
        probe_df = df.iloc[-160_000:]
        causality_probe(probe_df, default_cfg, SPOT)

    winner = default_cfg
    if choice in ("phase1", "all"):
        _, winner = phase1_grid(df)

    if choice in ("validate", "all"):
        validate_final(df, winner)

    if choice in ("falsify", "all"):
        falsification(winner)

    if choice in ("crashlag", "all"):
        falsification_crash_lag(df, winner)

    if choice in ("test4", "all"):
        thesis_test4(df, winner)

    print(f"\nCONFIGS EVALUATED THIS RUN: core={CONFIG_COUNTER['core']} "
          f"diagnostic={CONFIG_COUNTER['diagnostic']} "
          f"total={CONFIG_COUNTER['core'] + CONFIG_COUNTER['diagnostic']}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[{time.time() - t0:.0f}s]")
