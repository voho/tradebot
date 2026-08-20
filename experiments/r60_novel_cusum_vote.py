"""R-60 NOVEL BRANCH (backlog B-26): replace kelly_regime_v4's moving-
average-crossing-with-latch vote with a CUSUM sequential change-point
detector on log-returns, and test whether faster/genuinely-different vote
TIMING (not exposure SCALE, which R-59 tested twice and closed) restores
the matched-exposure drawdown property on R-57's six-asset panel.

Not registered: lives under ``experiments/`` per ROUTINE.md step 5. This
is an experiment, not a strategy the runner should discover, so
``KellyRegimeCusumVote`` below is a plain ``Strategy`` subclass with no
``@register`` decorator, constructed directly, never through
``tradebot.registry.get_strategy``.

Pre-registration: ``experiments/r60_shared.py`` (read it first — this file
imports its windows, costs, panel loader and decision-rule helpers rather
than restating them). This module writes its OWN pre-registration below,
for the parts r60_shared.py deliberately leaves to each branch: the exact
CUSUM formula, the design choice between a single detector and an
ensemble, and the (k, h) sensitivity grid. Everything below this line was
written and committed BEFORE a single backtest in this file was run —
before any drawdown or performance number, on any asset, was read.

=====================================================================
1. THE CUSUM FORMULA (Page, 1954, "Continuous Inspection Schemes",
   Biometrika 41(1/2), 100-115), exactly as specified in this round's
   task brief
=====================================================================

Let r_t = log(close_t) - log(close_{t-1}) (one bar's log-return). Two
one-sided running sums, a reference drift k > 0 and a threshold h > 0:

    S+_t = max(0, S+_{t-1} + (r_t - k))     # accumulates evidence of an
                                             # upward shift
    S-_t = max(0, S-_{t-1} - (r_t + k))     # accumulates evidence of a
                                             # downward shift

A regime flip signals the first bar where S+_t >= h (bullish) or
S-_t >= h (bearish); the triggering side resets to 0 on signal (the
other side is left untouched, exactly as the task brief specifies). The
vote LATCHES between signals — holds the previous verdict — the same
hysteresis semantics as the moving-average anchors it replaces
(``KellyRegimeV3.prepare()``'s ``v.ffill().fillna(0.0)``), so the initial
state before any signal has fired is bearish/flat (0.0), matching that
same default. This is causal and sequential by construction: each bar's
update reads only ``r_t`` and the previous bar's running-sum state, never
a batch statistic over the whole series and never a future bar — see
section 4's causality tamper probe.

=====================================================================
2. DESIGN CHOICE: (b), an ENSEMBLE of 3 CUSUM detectors on a doubling
   threshold ladder — not (a), a single detector
=====================================================================

`` kelly_regime_v4``'s vote is not one signal — it is three anchors
(20/40/80 calendar days, "a clean doubling ladder", per v4's own
docstring) averaged into a continuous ``frac`` in [0, 1], which then
scales exposure. Replacing that with a SINGLE CUSUM detector (design a)
would change two things about the vote at once: its timing mechanism
AND its granularity (continuous multi-voter average -> a single binary
latch jumping straight between 0.0 and 1.0). That confounds the
question this round asks. Design (b) instead keeps the vote's
architecture as parallel as possible to v4's own — three detectors,
averaged, same as the three anchors — and changes ONLY the timing
mechanism each one uses to decide bullish/bearish. This isolates the
axis this round is actually testing.

The three detectors share one reference drift ``k`` and a threshold
ladder ``h, 2h, 4h`` — a doubling ladder on the THRESHOLD, deliberately
built to mirror v4's own doubling ladder on the ANCHOR CALENDAR LENGTH
(20/40/80 = doubling twice), but expressed in the natural unit for a
CUSUM (how much accumulated evidence is required) rather than a
calendar length, since a CUSUM has no fixed "window" to double. Doubling
h roughly doubles the average evidence (and hence, in expectation, the
average bars) needed to trigger for a given excess drift, giving the
same fast/medium/slow structure the three anchors provide, built from
one shared, structural multiplier ladder `` CUSUM_LADDER_MULTIPLIERS =
(1.0, 2.0, 4.0)`` that is FIXED, not swept — chosen for this structural
reason before any grid result was read, exactly as v4's own 20/40/80 was
motivated by "each anchor covers twice the horizon of the last" rather
than fit.

=====================================================================
3. PRE-REGISTERED (k, h) SENSITIVITY GRID — narrow, fixed before any
   backtest in this file ran
=====================================================================

Grid values are derived from one back-of-envelope calculation, not from
any performance number:

    BARS_PER_YEAR = 365.25 * 288 ~= 105,192 five-minute bars/year
    BTC's own typical annualized realized vol over this project's data
    is ~50-80% (kelly_regime's own docstring; R-57 measured the panel's
    vol as structurally HIGHER than BTC's). A typical PER-BAR log-return
    standard deviation is therefore roughly
        sigma_bar ~= annualized_vol / sqrt(BARS_PER_YEAR)
                   ~= 0.55 / 324 ~= 0.0017  (BTC)
    and higher (up to roughly 2-3x) on the panel's higher-vol
    instruments.

``CUSUM_K_GRID = (0.0005, 0.0010, 0.0020)`` — per-bar reference drift, in
log-return units, spanning roughly 0.3x to over 1x BTC's own per-bar
sigma_bar. k must be strictly positive (a k=0 CUSUM on a driftless
random walk accumulates purely from noise and never mean-reverts to 0,
which would make it fire on chop alone); these three values keep k
comfortably below a full sigma_bar so the detector is genuinely
sensitive to sub-large moves, rather than requiring an extreme one.

``CUSUM_H_GRID = (0.0100, 0.0200, 0.0400)`` — the BASE (fastest-rung)
cumulative threshold, i.e. 1%, 2% or 4% of accumulated directional
log-return before the fastest detector in the ensemble flips. This is
the SAME order of magnitude as v4's own 1% price band, just accumulated
sequentially rather than read off an instantaneous ratio.

3 x 3 = **9 grid points**. Every grid point is evaluated identically on
PANEL_TRAIN (D1-analog) and CONTROL (D2-analog) — see section 5 — and
NOTHING else (no D3/D4/D5 number) is read until one grid point is
selected. This is deliberately narrow: R-59's own two branches already
established this strategy family's SIZE axis is exhausted, so this round
spends its trials budget confirming a genuinely different axis rather
than re-running a broad search on a mechanism whose sensitivity to
sizing is already known not to be the fix.

k and h are GLOBAL constants, identical across all 8 assets (BTC, ETH,
6 panel) exactly as v4's own 1% band and 20/40/80-day horizons are.
No per-asset number is fit anywhere in this branch — that keeps this
round's own question (does TIMING, not SCALE, fix it) uncontaminated by
a per-asset-calibration confound, which is what R-59 already tested
(twice) and closed.

=====================================================================
4. SELECTION RULE — fixed before any grid result was read
=====================================================================

For each of the 9 (k, h) grid points, compute (on PANEL_TRAIN and
CONTROL only, spot @0.10%, never touching PANEL_TEST or the 0.40% tier):

  - ``k1``  = count of the 6 PANEL_TRAIN assets where the candidate's max
              drawdown is strictly below the mean-notional-matched hold's
              (identical methodology to r60_shared's D1).
  - ``margin`` = the D2 "slack": min over {BTC, ETH} of
              ``(R57_CONTROL_DD_ADVANTAGE[ticker] + D2_REGRESSION_TOLERANCE_PP)
              - candidate's own matched-exposure dd advantage on that
              ticker`` — i.e. how much room remains before the candidate
              would fail D2's regression-tolerance gate. Larger margin =
              safer.

**Selection: the grid point with the highest ``k1``; ties broken by the
largest ``margin``; any further tie broken by the lowest (k, h) index in
a fixed ascending grid-iteration order (k ascending, then h ascending),
for determinism.** The selected (k, h) is then FROZEN — used unchanged
for the causality probe, D3, D4 and D5 below. This is the literal
selection rule the round's own brief offered as an example
("best D1 count, ties broken by D2 margin").

=====================================================================
5. WHAT WOULD MAKE THIS FAIL (named before any code ran, restating
   r60_shared's own named failure modes, specific to this branch)
=====================================================================

  - No grid point reaches D1 >= 5/6 on PANEL_TRAIN.
  - The selected grid point's own D2 regresses v4's BTC/ETH control by
    more than 5pp on either asset (the sweep's own margin computation is
    designed to select AGAINST this, but the sweep's margin is a
    point-estimate heuristic, not the frozen bootstrap-interval D2 gate
    itself, so this is checked again, formally, after selection).
  - D3 (crash-transition lag): a CUSUM tuned to react fast in quiet chop
    can be SLOW to re-trigger immediately after a large move exhausts its
    running sum — precisely the risk this round's brief names as CUSUM's
    known failure mode, and precisely a crash. Checked explicitly below,
    per r60_shared's frozen D3 rule (candidate's mean flip-to-flat lag
    across the three CRASH_WINDOWS must not exceed v4's own baseline lag
    by more than 2 bars).
  - Whipsaw/churn: CUSUM without hysteresis (beyond the shared latch) can
    flip far more often than v4's slow, latched vote. D5 reports the
    candidate's trade count on the panel against v4's own, on the same
    assets and window, as a named diagnostic (not a gate, since it is not
    part of r60_shared's frozen promotion bar) — but if the candidate's
    turnover is dramatically higher, that is reported plainly as a cost
    this round's D1-D3 gates do not themselves price in.

Usage::

    uv run python experiments/r60_novel_cusum_vote.py sweep      # select (k,h)
    uv run python experiments/r60_novel_cusum_vote.py causality  # tamper probe
    uv run python experiments/r60_novel_cusum_vote.py run        # full D1-D5
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r60_shared import (  # noqa: E402
    CONTROL,
    CRASH_WINDOWS,
    D2_REGRESSION_TOLERANCE_PP,
    D3_MAX_EXTRA_LAG_BARS,
    PANEL_TEST,
    PANEL_TRAIN,
    R57_CONTROL_DD_ADVANTAGE,
    SPOT_BASE,
    SPOT_REAL,
    Asset,
    ConstantExposureHold,
    binomial_tail,
    d1_verdict,
    d2_passes,
    d3_passes,
    load_panel,
    mean_notional,
    promoted,
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
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import prefix_bars  # noqa: E402
from tradebot.window import run_period  # noqa: E402

OUT_DIR = ROOT / "reports" / "r60_novel"
BOOT_KW = dict(mean_block=30.0, n_boot=2_000, seed=7)

# Pre-registered, section 3 above. Fixed before any backtest ran.
CUSUM_K_GRID = (0.0005, 0.0010, 0.0020)
CUSUM_H_GRID = (0.0100, 0.0200, 0.0400)
CUSUM_LADDER_MULTIPLIERS = (1.0, 2.0, 4.0)  # structural, fixed, not swept

CONFIG_COUNT = 0
FLAT_EPS = 1e-9  # "flat" = target exactly 0.0 (frac == 0 on every detector),
                 # same convention R-56's crash_transition_lag_check used.


# ================================================================== strategy


def _cusum_vote(r: np.ndarray, k: float, h: float) -> np.ndarray:
    """One CUSUM detector's latched bullish(1.0)/bearish(0.0) vote series.

    A genuine bar-by-bar Python loop, deliberately NOT a vectorized numpy
    shortcut: a running-sum state with a reset-on-signal is exactly the
    kind of recursive computation a "clever" vectorized rewrite could
    leak lookahead into (e.g. any formulation that peeks at where a
    threshold crossing will occur before reaching that bar). See
    ``cmd_causality`` below for the tamper probe this is built to pass.
    """
    n = len(r)
    vote = np.empty(n, dtype=float)
    s_pos = 0.0
    s_neg = 0.0
    state = 0.0  # bearish/flat until the first signal - same default as
                 # v3/v4's ffill().fillna(0.0) anchor vote
    for i in range(n):
        ri = r[i]
        if ri == ri and np.isfinite(ri):  # NaN-safe without an extra call
            s_pos = max(0.0, s_pos + (ri - k))
            s_neg = max(0.0, s_neg - (ri + k))
            if s_pos >= h:
                state = 1.0
                s_pos = 0.0
            elif s_neg >= h:
                state = 0.0
                s_neg = 0.0
        vote[i] = state
    return vote


class KellyRegimeCusumVote(KellyRegimeV3):
    """v3/v4's fractional-Kelly vol-targeting mechanism, with the
    moving-average-crossing-with-latch vote replaced by a 3-detector CUSUM
    ensemble (design (b), see module docstring section 2). NOT registered
    — an experiment, constructed directly, never through
    ``tradebot.registry.get_strategy``.

    Everything else is byte-identical to ``KellyRegimeV3.prepare()``
    (inherited unchanged by ``KellyRegimeV4``): the conditional
    vol-targeting scale terms, the high/low breakout hysteresis state
    machine, the 10% deadband, ``target_vol=0.55``, ``max_leverage=2.0``.
    Only the vote-computation block changes.
    """

    name = "kelly_regime_r60_cusum_vote"  # attribute only; no @register
    warmup = 80 * BARS_PER_DAY + 10  # identical to KellyRegimeV4, for a
                                     # fair apples-to-apples comparison

    def __init__(self, cusum_k: float = 0.0010, cusum_h: float = 0.0200,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.cusum_k = float(cusum_k)
        self.cusum_h = float(cusum_h)  # base (fastest-rung) threshold;
                                       # ensemble ladder is h, 2h, 4h

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r_series = np.log(close).diff()
        r = r_series.to_numpy(dtype=float)

        # --- the ONE change: a 3-detector CUSUM ensemble replaces the
        # 3-anchor moving-average-crossing vote. Same "vote and average"
        # role: sum/len, then vote_gamma, exactly as KellyRegimeV3.
        votes = [_cusum_vote(r, self.cusum_k, self.cusum_h * mult)
                 for mult in CUSUM_LADDER_MULTIPLIERS]
        frac = sum(votes) / len(votes)
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        # --- everything below is byte-identical to
        # KellyRegimeV3.prepare() (see src/tradebot/strategies/kelly_regime_v3.py)
        vol = (r_series.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df


def make_candidate(k: float, h: float) -> KellyRegimeCusumVote:
    """Fresh instance per backtest, same convention as get_strategy()."""
    return KellyRegimeCusumVote(cusum_k=k, cusum_h=h)


# =================================================================== helpers


def measure(strategy, df, start, end, market):
    """One backtest. Every call is counted."""
    global CONFIG_COUNT
    CONFIG_COUNT += 1
    result = run_period(strategy, df, start, end, market=market, start_balance=1_000.0)
    return result, compute_metrics(result)


def _btc_df():
    df, _ = load_dataset(ROOT / "data", "spot")
    return df.loc[:"2022-12-31"]


def _eth_df():
    return load_coinbase_spot(ROOT / "data", "ETH").loc[:"2022-12-31"]


# =========================================================== sweep/selection


def sweep_cell(ticker: str, df: pd.DataFrame, window, k: float, h: float,
                rows: list) -> dict:
    """Candidate + mean-notional-matched hold only (buy_and_hold is not
    needed for selection). Both are counted."""
    start, end = window
    cand_res, cand = measure(make_candidate(k, h), df, start, end, SPOT_BASE)
    c_mean = mean_notional(cand_res)
    mh_res, mh = measure(ConstantExposureHold(c_mean), df, start, end, SPOT_BASE)

    cand_ret = daily_returns(cand_res.equity).to_numpy(dtype=float)
    mh_ret = daily_returns(mh_res.equity).to_numpy(dtype=float)
    n = min(len(cand_ret), len(mh_ret))
    dd = paired_bootstrap(cand_ret[:n], mh_ret[:n], max_drawdown_from_returns, **BOOT_KW)

    row = {
        "k": k, "h": h, "ticker": ticker,
        "cand_dd": cand.max_drawdown_pct, "cand_trades": cand.num_trades,
        "c_mean_notional": c_mean, "mh_dd": mh.max_drawdown_pct,
        "dd_diff": dd.diff.point, "dd_lo": dd.diff.lo, "dd_hi": dd.diff.hi,
    }
    rows.append(row)
    return row


def cmd_sweep() -> tuple[float, float, pd.DataFrame, pd.DataFrame]:
    print("=" * 100)
    print(f"SWEEP — {len(CUSUM_K_GRID)}x{len(CUSUM_H_GRID)}="
          f"{len(CUSUM_K_GRID) * len(CUSUM_H_GRID)} (k,h) grid points, "
          "PANEL_TRAIN (6 assets) + CONTROL (BTC, ETH), spot @0.10%")
    print("=" * 100)
    panel = load_panel()
    btc_df, eth_df = _btc_df(), _eth_df()
    train_frames = [(a.ticker, a.df) for a in panel]
    control_frames = [("BTC", btc_df), ("ETH", eth_df)]

    rows: list[dict] = []
    for k in CUSUM_K_GRID:
        for h in CUSUM_H_GRID:
            for ticker, df in train_frames:
                sweep_cell(ticker, df, PANEL_TRAIN, k, h, rows)
            for ticker, df in control_frames:
                sweep_cell(ticker, df, CONTROL, k, h, rows)
            g = pd.DataFrame(rows)
            g = g[(g.k == k) & (g.h == h)]
            tr = g[g.ticker.isin([a.ticker for a in panel])]
            k1 = int((tr.cand_dd < tr.mh_dd).sum())
            ctl = g[g.ticker.isin(["BTC", "ETH"])].set_index("ticker")
            print(f"  k={k:.4f} h={h:.4f}  TRAIN k1={k1}/6  "
                  f"CONTROL BTC dDD={ctl.loc['BTC','dd_diff']:+6.1f}pp "
                  f"ETH dDD={ctl.loc['ETH','dd_diff']:+6.1f}pp")

    df = pd.DataFrame(rows)
    panel_tickers = [a.ticker for a in panel]
    summary = []
    for (k, h), g in df.groupby(["k", "h"], sort=False):
        tr = g[g.ticker.isin(panel_tickers)]
        k1 = int((tr.cand_dd < tr.mh_dd).sum())
        ctl = g[g.ticker.isin(["BTC", "ETH"])].set_index("ticker")
        btc_adv, eth_adv = ctl.loc["BTC", "dd_diff"], ctl.loc["ETH", "dd_diff"]
        margin = min(
            (R57_CONTROL_DD_ADVANTAGE["BTC"] + D2_REGRESSION_TOLERANCE_PP) - btc_adv,
            (R57_CONTROL_DD_ADVANTAGE["ETH"] + D2_REGRESSION_TOLERANCE_PP) - eth_adv,
        )
        summary.append({"k": k, "h": h, "k1": k1, "btc_adv": btc_adv,
                        "eth_adv": eth_adv, "margin": margin})
    summary_df = pd.DataFrame(summary)
    # Deterministic tie-break: ascending (k, h) grid-iteration order is
    # already the DataFrame's row order (groupby(sort=False) over the
    # nested loop above), so a stable sort on (k1 desc, margin desc)
    # keeps the earliest grid point among ties - exactly the pre-
    # registered "lowest (k,h) index" rule in section 4.
    summary_df = summary_df.sort_values(["k1", "margin"], ascending=[False, False],
                                        kind="stable")
    sel = summary_df.iloc[0]
    print("\n" + "-" * 100)
    print("SELECTION (highest D1 k1, ties broken by D2 margin, "
          "further ties by ascending (k,h) grid order):")
    print(summary_df.to_string(index=False))
    print(f"\nSELECTED: k={sel.k:.4f} h={sel.h:.4f}  "
          f"(k1={int(sel.k1)}/6, margin={sel.margin:+.1f}pp)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "sweep_cells.csv", index=False)
    summary_df.to_csv(OUT_DIR / "sweep_summary.csv", index=False)
    return float(sel.k), float(sel.h), df, summary_df


# ================================================================ causality


def cmd_causality(k: float, h: float, label: str = "") -> bool:
    """test_causality_strict.py's tamper methodology (R-57/R-59's adapted
    form): opposite 3x/÷3 price and 7x/÷7 volume tampers after a cut,
    decisions compared at 1/2/3/5/10/20 bars before the cut. Run on BTC
    (pre-2023 only, per this round's holdout restriction) plus 2 panel
    assets, constructing KellyRegimeCusumVote directly. Counts 0
    backtests (prepare()+on_bar() only, no broker/measure() run)."""
    tag = f" ({label})" if label else ""
    print("=" * 100)
    print(f"CAUSALITY TAMPER PROBE{tag} — KellyRegimeCusumVote(k={k}, h={h})")
    print("=" * 100)
    market = MarketSpec.futures(leverage=5.0)
    panel = load_panel()
    probe_assets = [("BTC", _btc_df())] + [(a.ticker, a.df) for a in panel[:2]]

    all_ok = True
    for ticker, df in probe_assets:
        tail = df.iloc[-60_000:].copy()
        cut = len(tail) - 5_000
        bars = [cut - x for x in (1, 2, 3, 5, 10, 20)]
        up, down = tail.copy(), tail.copy()
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

        def decisions(frame):
            s = make_candidate(k, h)
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=market, start_balance=10_000.0)
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        ok = all(x == y for x, y in zip(decisions(up), decisions(down)))
        all_ok = all_ok and ok
        print(f"  {ticker:5s} decisions identical under opposite post-cut "
              f"tampers: {'PASS' if ok else 'FAIL'}")
    return all_ok


# ==================================================================== D1-D5


def cell(ticker: str, df: pd.DataFrame, window, market, label: str,
         k: float, h: float, rows: list, v4_baseline: bool = False) -> dict:
    """Candidate + buy_and_hold + mean-notional-matched hold + (optionally)
    v4's own unmodified baseline on the same asset/window, for a turnover
    comparison. Every measure() call is counted."""
    start, end = window
    cand_res, cand = measure(make_candidate(k, h), df, start, end, market)
    hold_res, hold = measure(get_strategy("buy_and_hold"), df, start, end, market)

    c_mean = mean_notional(cand_res)
    mh_res, mh = measure(ConstantExposureHold(c_mean), df, start, end, market)

    cand_ret = daily_returns(cand_res.equity).to_numpy(dtype=float)
    mh_ret = daily_returns(mh_res.equity).to_numpy(dtype=float)
    n = min(len(cand_ret), len(mh_ret))
    dd_matched = paired_bootstrap(cand_ret[:n], mh_ret[:n],
                                  max_drawdown_from_returns, **BOOT_KW)
    growth_matched = paired_bootstrap(cand_ret[:n], mh_ret[:n],
                                      total_log_return, **BOOT_KW)

    row = {
        "asset": ticker, "window": label, "market": market.name,
        "fee": market.fee_rate,
        "cand_final": cand.final_balance, "cand_dd": cand.max_drawdown_pct,
        "cand_sharpe": cand.sharpe, "cand_trades": cand.num_trades,
        "hold_final": hold.final_balance, "hold_dd": hold.max_drawdown_pct,
        "c_mean_notional": c_mean,
        "mh_final": mh.final_balance, "mh_dd": mh.max_drawdown_pct,
        "dd_matched_diff": dd_matched.diff.point,
        "dd_matched_lo": dd_matched.diff.lo, "dd_matched_hi": dd_matched.diff.hi,
        "growth_matched_diff": growth_matched.diff.point,
        "growth_matched_lo": growth_matched.diff.lo,
        "growth_matched_hi": growth_matched.diff.hi,
    }
    if v4_baseline:
        v4_res, v4 = measure(get_strategy("kelly_regime_v4"), df, start, end, market)
        row["v4_trades"] = v4.num_trades
        row["v4_dd"] = v4.max_drawdown_pct
        row["v4_final"] = v4.final_balance
    rows.append(row)
    print(f"  {ticker:5s} {label:7s} {market.name:11s} fee={market.fee_rate:.2%}  "
          f"cand ${cand.final_balance:>10,.0f} DD {cand.max_drawdown_pct:5.1f}% "
          f"trades={cand.num_trades:>4d} | "
          f"hold ${hold.final_balance:>10,.0f} DD {hold.max_drawdown_pct:5.1f}% | "
          f"matched(c={c_mean:.2f}) DD {mh.max_drawdown_pct:5.1f}% | "
          f"dDD_matched {dd_matched.diff.point:+6.1f}pp "
          f"[{dd_matched.diff.lo:+6.1f},{dd_matched.diff.hi:+6.1f}]"
          + (f" | v4 trades={row['v4_trades']}" if v4_baseline else ""))
    return row


def cmd_d1(k: float, h: float) -> tuple[int, pd.DataFrame]:
    print("\n" + "=" * 100)
    print("D1 (PRIMARY) — PANEL_TRAIN 2020-04-01..2022-12-31, spot @0.10%, "
          "matched-exposure drawdown (with v4 baseline turnover)")
    print("=" * 100)
    panel = load_panel()
    rows: list[dict] = []
    for a in panel:
        cell(a.ticker, a.df, PANEL_TRAIN, SPOT_BASE, "TRAIN", k, h, rows,
             v4_baseline=True)
    df = pd.DataFrame(rows)
    k1 = int((df.cand_dd < df.mh_dd).sum())
    excl = int(((df.dd_matched_lo > 0) | (df.dd_matched_hi < 0)).sum())
    better_excl = int((df.dd_matched_hi < 0).sum())
    p1 = binomial_tail(k1, 6)
    print(f"\nD1: {k1}/6 assets, exact binomial p={p1:.4f} -> {d1_verdict(k1)}")
    print(f"    paired bootstrap: {excl}/6 intervals exclude zero "
          f"({better_excl}/6 in candidate's favour)")
    print(f"    turnover: candidate trades {list(df.cand_trades)} vs "
          f"v4 baseline trades {list(df.v4_trades)}")
    return k1, df


def cmd_d2(k: float, h: float) -> tuple[dict[str, float], pd.DataFrame]:
    print("\n" + "=" * 100)
    print("D2 (FALSIFICATION) — CONTROL 2020-04-01..2022-12-31, BTC/ETH, "
          "spot @0.10%, matched-exposure drawdown")
    print("=" * 100)
    rows: list[dict] = []
    cell("BTC", _btc_df(), CONTROL, SPOT_BASE, "CONTROL", k, h, rows, v4_baseline=True)
    cell("ETH", _eth_df(), CONTROL, SPOT_BASE, "CONTROL", k, h, rows, v4_baseline=True)
    df = pd.DataFrame(rows)
    dd_advantage = {r["asset"]: r["dd_matched_diff"] for r in rows}
    passed = d2_passes(dd_advantage)
    print(f"\nD2 candidate dDD: BTC {dd_advantage['BTC']:+.1f}pp "
          f"(R-57 v4 control: {R57_CONTROL_DD_ADVANTAGE['BTC']:+.1f}pp), "
          f"ETH {dd_advantage['ETH']:+.1f}pp "
          f"(R-57 v4 control: {R57_CONTROL_DD_ADVANTAGE['ETH']:+.1f}pp), "
          f"tolerance +{D2_REGRESSION_TOLERANCE_PP}pp -> "
          f"{'PASSES' if passed else 'FAILS'}")
    return dd_advantage, df


def _flip_to_flat_lag(strategy, df: pd.DataFrame, window: tuple[str, str]) -> float:
    """Bars from the crash window's own price peak to the strategy's own
    target first reaching flat (target < FLAT_EPS) after that peak.

    Not a backtest: calls prepare() directly on a frame carrying the
    strategy's own warmup prefix (mirrors run_period's own prefix logic
    via tradebot.window.prefix_bars, without invoking run_period/measure
    since only the target/close series are needed, not a broker run).
    Costs 0 towards CONFIG_COUNT, same convention as the causality probe.
    """
    start, end = window
    lo = int(df.index.searchsorted(pd.Timestamp(start, tz="UTC")))
    hi = int(df.index.searchsorted(pd.Timestamp(end, tz="UTC"), side="right"))
    prefix = prefix_bars(df, lo, strategy.warmup)
    frame = df.iloc[lo - prefix: hi].copy()
    prepared = strategy.prepare(frame)
    target = prepared["target"].to_numpy(dtype=float)
    close = prepared["close"].to_numpy(dtype=float)

    measured_start = prefix  # position, within `frame`, where the window begins
    peak_rel = int(np.argmax(close[measured_start:])) + measured_start
    for i in range(peak_rel, len(target)):
        if target[i] < FLAT_EPS:
            return float(i - peak_rel)
    return float("nan")  # never flattened within the window


def cmd_d3(k: float, h: float) -> tuple[float, float, pd.DataFrame]:
    print("\n" + "=" * 100)
    print("D3 (CRASH-TRANSITION-LAG) — BTC, the three CRASH_WINDOWS, spot, "
          "candidate vs v4's unmodified baseline signal")
    print("=" * 100)
    df = _btc_full_for_crash()
    cand = make_candidate(k, h)
    baseline = get_strategy("kelly_regime_v4")
    rows = []
    for name, window in CRASH_WINDOWS.items():
        cand_lag = _flip_to_flat_lag(cand, df, window)
        base_lag = _flip_to_flat_lag(baseline, df, window)
        rows.append({"window": name, "cand_lag_bars": cand_lag,
                     "baseline_lag_bars": base_lag})
        print(f"  {name:14s} candidate lag={cand_lag!s:>6s} bars   "
              f"v4 baseline lag={base_lag!s:>6s} bars")
    dfw = pd.DataFrame(rows)
    cand_mean = float(np.nanmean(dfw.cand_lag_bars))
    base_mean = float(np.nanmean(dfw.baseline_lag_bars))
    passed = d3_passes(cand_mean, base_mean)
    print(f"\nD3: candidate mean lag={cand_mean:.2f} bars, "
          f"v4 baseline mean lag={base_mean:.2f} bars, "
          f"tolerance +{D3_MAX_EXTRA_LAG_BARS} bars -> "
          f"{'PASSES' if passed else 'FAILS'}")
    return cand_mean, base_mean, dfw


def _btc_full_for_crash():
    """BTC's full series (not truncated to CONTROL) — the three
    CRASH_WINDOWS include Nov 2018, which predates PANEL_TRAIN/CONTROL's
    2020-04-01 start. All three windows are pre-2023 (see r60_shared),
    so this stays inside the +0 holdout-cost convention."""
    df, _ = load_dataset(ROOT / "data", "spot")
    return df.loc[:"2022-12-31"]


def cmd_d4(k: float, h: float) -> tuple[int, pd.DataFrame]:
    print("\n" + "=" * 100)
    print("D4 (GENERALIZATION, descriptive) — PANEL_TEST 2023-01-01..2026-08-20, "
          "spot @0.10%")
    print("=" * 100)
    panel = load_panel()
    rows: list[dict] = []
    for a in panel:
        cell(a.ticker, a.df, PANEL_TEST, SPOT_BASE, "TEST", k, h, rows)
    df = pd.DataFrame(rows)
    k4 = int((df.cand_dd < df.mh_dd).sum())
    print(f"\nD4: {k4}/6 assets favour the candidate on the matched-exposure "
          f"drawdown axis (descriptive, not a gate)")
    return k4, df


def cmd_d5(k: float, h: float) -> tuple[int, pd.DataFrame]:
    print("\n" + "=" * 100)
    print("D5 (0.40% FEE FALSIFICATION) — PANEL_TRAIN, spot @0.40%, beats "
          "buy_and_hold's final balance (predicted: FAILS)")
    print("=" * 100)
    panel = load_panel()
    rows: list[dict] = []
    for a in panel:
        cell(a.ticker, a.df, PANEL_TRAIN, SPOT_REAL, "TRAIN", k, h, rows,
             v4_baseline=True)
    df = pd.DataFrame(rows)
    k5 = int((df.cand_final > df.hold_final).sum())
    verdict = "SURVIVES" if k5 >= 5 else "FAILS (as predicted)"
    print(f"\nD5: {k5}/6 -> {verdict}")
    print(f"    turnover @0.40%: candidate trades {list(df.cand_trades)} vs "
          f"v4 baseline trades {list(df.v4_trades)}")
    return k5, df


# ========================================================================= main


def cmd_run() -> None:
    k0, h0 = CUSUM_K_GRID[len(CUSUM_K_GRID) // 2], CUSUM_H_GRID[len(CUSUM_H_GRID) // 2]
    ok = cmd_causality(k0, h0, label="pre-sweep sanity gate, mid-grid config")
    if not ok:
        raise SystemExit("CAUSALITY PROBE FAILED (pre-sweep) — refusing to run "
                         "the sweep until the strategy is causal.")
    print()

    k, h, sweep_df, summary_df = cmd_sweep()
    print()

    ok = cmd_causality(k, h, label="post-selection, frozen config")
    if not ok:
        raise SystemExit("CAUSALITY PROBE FAILED (selected config) — refusing "
                         "to report results until the strategy is causal.")

    k1, d1_df = cmd_d1(k, h)
    dd_advantage, d2_df = cmd_d2(k, h)
    cand_lag, base_lag, d3_df = cmd_d3(k, h)
    k4, d4_df = cmd_d4(k, h)
    k5, d5_df = cmd_d5(k, h)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d1_df.to_csv(OUT_DIR / "d1_panel_train.csv", index=False)
    d2_df.to_csv(OUT_DIR / "d2_control.csv", index=False)
    d3_df.to_csv(OUT_DIR / "d3_crash_lag.csv", index=False)
    d4_df.to_csv(OUT_DIR / "d4_panel_test.csv", index=False)
    d5_df.to_csv(OUT_DIR / "d5_panel_train_040.csv", index=False)

    verdict = promoted(k1, dd_advantage, cand_lag, base_lag)
    print("\n" + "=" * 100)
    print("VERDICT (mechanical application of experiments.r60_shared.promoted)")
    print("=" * 100)
    print(f"Selected (k,h) = ({k}, {h})")
    print(f"D1: {k1}/6 -> {d1_verdict(k1)}")
    print(f"D2: {'PASSES' if d2_passes(dd_advantage) else 'FAILS'} "
          f"(BTC {dd_advantage['BTC']:+.1f}pp, ETH {dd_advantage['ETH']:+.1f}pp)")
    print(f"D3: {'PASSES' if d3_passes(cand_lag, base_lag) else 'FAILS'} "
          f"(candidate {cand_lag:.2f} bars vs v4 baseline {base_lag:.2f} bars)")
    print(f"D4 (descriptive): {k4}/6")
    print(f"D5 (0.40% fee, beats buy_and_hold): {k5}/6")
    print(f"\n-> {'PROMOTE-CANDIDATE' if verdict else 'NEGATIVE'}")
    print(f"\nTotal backtest configurations evaluated: {CONFIG_COUNT}")
    print("Holdout consultations added by this round: 0 "
          "(no BTC/ETH bar past 2022-12-31 is read anywhere in this module)")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        cmd_run()
        return
    if cmd == "sweep":
        cmd_sweep()
    elif cmd == "causality":
        k0 = CUSUM_K_GRID[len(CUSUM_K_GRID) // 2]
        h0 = CUSUM_H_GRID[len(CUSUM_H_GRID) // 2]
        cmd_causality(k0, h0, label="mid-grid config")
    else:
        raise SystemExit(f"unknown command {cmd!r} (sweep | causality | run)")
    print(f"\nTotal backtest configurations evaluated: {CONFIG_COUNT}")


if __name__ == "__main__":
    main()
