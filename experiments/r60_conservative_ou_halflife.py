"""R-60 conservative branch: per-asset OU half-life rescaling of `kelly_regime_v4`'s
anchor ladder (backlog B-26). Pre-registration: `experiments/r60_shared.py` — read
it first; this file implements its "conservative branch" (the vote-TIMING axis,
as opposed to R-59's SCALE axis) and nothing else about the strategy changes.

=====================================================================
PRE-REGISTRATION — written and frozen BEFORE any drawdown/performance number
was read. Price-only exploration (the half-life numbers themselves, which are
a property of the data, not of any strategy's performance) was run before this
docstring was committed, per the operator's own instruction that this is
permitted; no backtest, no D1-D5 cell, and no config-count number existed when
this section was written.
=====================================================================

Mechanism, one sentence: replace `kelly_regime_v4`'s fixed (20,40,80)-calendar-
day anchor ladder with a ladder scaled, per asset, by that asset's own
structurally-estimated Ornstein-Uhlenbeck mean-reversion half-life relative to
BTC's — everything else (the moving-average-crossing vote with its 1% band and
latching hysteresis, the conditional fractional-Kelly vol targeting, the 10%
deadband, `target_vol=0.55`, `max_leverage=2.0`) is untouched, and
`KellyRegimeV4`/`KellyRegimeV3`/`KellyRegime` are used exactly as shipped —
`horizons` is a constructor argument they already accept, so nothing under
`src/tradebot/strategies/` is edited.

--- 1. The half-life formula (exact, frozen) ---

For asset i, on its own PANEL_TRAIN/CONTROL-window (2020-04-01 -> 2022-12-31,
the same calendar range under two names, per `r60_shared.PANEL_TRAIN` /
`r60_shared.CONTROL`) daily-resampled close series (`close.resample("1D").last()
.dropna()`, i.e. one observation per calendar day — not the native 5-minute
bar, so theta and the resulting half-life are denominated in DAYS, directly
comparable to the anchor ladder's own calendar-day units):

    p_t       = log(close_t)                          (t = 0 .. T, daily)
    mu_i      = mean(p_t)  over the whole window       (long-run mean, log-price)
    x_t       = p_{t-1} - mu_i        for t = 1 .. T
    y_t       = p_t - p_{t-1}         for t = 1 .. T
    beta_i    = sum(x_t * y_t) / sum(x_t * x_t)        (OLS slope, THROUGH THE
                                                         ORIGIN — x is already
                                                         demeaned by construction,
                                                         so no separate intercept
                                                         term is fit; verified
                                                         against an OLS-with-
                                                         intercept variant before
                                                         freezing this choice —
                                                         intercepts differ from
                                                         zero by <=0.002 on log-
                                                         price units, i.e. the
                                                         no-intercept and with-
                                                         intercept slopes agree
                                                         to 4 significant figures
                                                         on every one of the 8
                                                         assets, so the simpler,
                                                         literal reading of "regress
                                                         y against x" is used and
                                                         nothing was chosen because
                                                         one variant scored better
                                                         on this — both were checked
                                                         BEFORE this docstring was
                                                         written, on price data only)
    theta_i   = -beta_i                                (mean-reversion speed,
                                                         per calendar day; theta_i
                                                         > 0 for genuine mean
                                                         reversion)
    halflife_i = ln(2) / theta_i                       (calendar days)

This is the standard OU/AR(1) discretization (Chan 2013, ch. 2; see
`r60_shared.py`'s literature section for the full citation) applied to LOG price
rather than raw price (crypto price levels differ by orders of magnitude across
the panel, so log-price is the scale-invariant choice — raw-price OU on BCH's
~$200-$600 range and BTC's ~$5k-$69k range would not be comparable at all).

--- 2. The anchor-scaling formula (exact, frozen) ---

BTC is the reference asset (it is `kelly_regime_v4`'s own fitting asset, per
`kelly_regime_v4`'s own docstring and R-07's own anchor sweep, which was run on
BTC alone). For every asset i (including BTC itself, trivially):

    ratio_i        = halflife_i / halflife_BTC
    ratio_clamped_i = clip(ratio_i, CLAMP_LO, CLAMP_HI)     (see section 3)
    horizons_i      = (20 * ratio_clamped_i,
                       40 * ratio_clamped_i,
                       80 * ratio_clamped_i)               (calendar days,
                                                             floats — passed
                                                             directly to
                                                             `KellyRegimeV4
                                                             (horizons=horizons_i)`,
                                                             which already
                                                             accepts floats:
                                                             `int(days *
                                                             BARS_PER_DAY)` is
                                                             applied downstream,
                                                             unchanged, inside
                                                             `KellyRegime.prepare`)

By construction, ratio_BTC = halflife_BTC / halflife_BTC = 1.0 EXACTLY, for
every possible pair of numbers (a number divided by itself), so BTC's own
candidate horizons are byte-identical to `kelly_regime_v4`'s shipped default
(20, 40, 80) and its D3 crash-lag cell (section 6) is therefore a trivial
identity check by construction, not a result — this is stated now, before any
half-life number was computed for BTC specifically, because it is a
consequence of the formula's algebra, not of the data.

The ladder's own internal doubling structure (each anchor exactly 2x and 4x the
shortest) is preserved for every asset — only the common scale factor changes,
which is the literal "vote arrives faster/slower, not differently shaped"
reading of B-26's own question.

--- 3. The clamping rule (exact, frozen, and why this bound specifically) ---

    CLAMP_LO = 18 / 20 = 0.90
    CLAMP_HI = 28 / 20 = 1.40

This is the R-07-validated 18-28 calendar-day plateau for the SHORTEST anchor
(`kelly_regime_v4`'s own docstring: "across nine anchor sets in the 18-28 day
range, EVERY variant cut max drawdown... below ~18 days the plateau breaks
sharply"), expressed as a multiplicative bound on the ratio so that
`20 * ratio_clamped_i` never leaves the one region this project has actually
validated by direct sweep. This is the identical derivation logic R-57's own
panel-selection amendment used (LEDGER.md R-57: "a coverage fraction f
stretches the 20-day anchor to 20/f calendar days ... f >= 0.80 keeps it <= 25
days, inside that validated plateau") — a project-validated empirical bound
converted into a clamp on a different quantity via the same algebra, not a
threshold invented for this round. The clamp exists to prevent the OU estimate
(a noisy statistic on ~1,000 daily observations, with no guarantee of staying
inside any particular range) from pushing an asset's ladder into anchor lengths
this project has never swept and has no evidence about.

--- 4. What is NOT being tuned ---

The half-life formula, the reference asset, and the clamp bound are all fixed
by the algebra above and by a PRIOR, already-published project finding (R-07's
plateau), not by anything measured in this round. No half-life number, no
resulting horizon, and no clamp-bound choice was adjusted after seeing a single
drawdown, Sharpe, or final-balance figure for any candidate. The one number
this round is free to determine from price data before pre-registration
(permitted explicitly by the operator's brief) is the half-life estimates
themselves and the resulting per-asset horizons table — reproduced in section 4
of the results below, computed by running exactly the formula above and nothing
else.

--- 5. Known limitation, named in advance (a data-availability fact, not a
performance number) ---

Every candidate's instance `.warmup` is overridden (see `make_candidate` below)
to `int(round(max(horizons_i) * BARS_PER_DAY)) + 10`, matching `kelly_regime_v4`'s
own `80 * BARS_PER_DAY + 10` formula generalized to whatever the scaled longest
anchor actually is — this is required for correctness (an R-22-class warmup-
prefix bias: a strategy whose longest anchor is scaled UP needs a longer
warm prefix before it enters the measured window fully warm, or its own rolling
mean is silently NaN-filled to a bearish-leaning default for the difference).
`experiments/matched_hold.py` and `KellyRegimeV4` both already accept this kind
of instance-level attribute; nothing under `src/` changes. One asset's clamp
resolves at the CEILING (ratio 1.40, longest anchor 112 days) with a
PANEL_TRAIN window starting only 91 days after that asset's own data begins
(2020-01-01 -> 2020-04-01) — its rolling anchor is not fully warm for
approximately the first three weeks of the measured PANEL_TRAIN window. This is
named now, before the affected asset's identity or D1 cell was read, as a
known, small, cold-start bias inherent to the panel's fetch range
(`r57_cross_asset_panel.FETCH_START = 2020-01-01`), not a bug in this file.

=====================================================================
DECISION RULES: identical to `r60_shared.py`'s D1-D5 and promotion bar,
reproduced there and not restated here. This branch adds nothing to them and
does not relax them.
=====================================================================

Configurations evaluated: counted honestly in three buckets, per the operator's
instruction to count every backtest AND every OU-fit — (a) `measure()`-style
`run_period` backtests (D1, D2, D4, D5), (b) the 8 OU half-life closed-form
regressions (one per asset, no solver, no iteration — a single OLS-through-
-origin call each), (c) prepare()-only signal evaluations for the D3
crash-transition-lag check and its ETH supplementary robustness cousin, which
call `strategy.prepare()` directly rather than `run_period` and are therefore
NOT counted in bucket (a), per the identical convention R-57's own causality
probe and this round's own causality probe use ("calls prepare()/on_bar()
directly, not run_period, so it is not counted as a backtest configuration").
All three buckets are reported honestly in the final total regardless.

Usage::

    uv run python experiments/r60_conservative_ou_halflife.py halflife    # OU calibration table only
    uv run python experiments/r60_conservative_ou_halflife.py causality   # tamper probe
    uv run python experiments/r60_conservative_ou_halflife.py run         # everything, writes CSVs
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
from experiments.r57_cross_asset_panel import Asset, binomial_tail  # noqa: E402
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
    d1_verdict,
    d2_passes,
    d3_passes,
    load_panel,
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
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "reports" / "r60_conservative"
REPORT_PATH = ROOT / "experiments" / "reports" / "r60_conservative_report.md"

REF_ASSET = "BTC"
BASE_HORIZONS = (20.0, 40.0, 80.0)
CLAMP_LO = 18.0 / 20.0
CLAMP_HI = 28.0 / 20.0
BOOT_KW = dict(mean_block=30.0, n_boot=2_000, seed=7)

CONFIG_COUNT = 0        # bucket (a): run_period backtests
OU_FIT_COUNT = 0        # bucket (b): OU regressions
SIGNAL_ONLY_COUNT = 0   # bucket (c): prepare()-only D3 evaluations


# ------------------------------------------------------------------ helpers


def measure(strategy, df, start, end, market):
    """One backtest. Every call is counted -- there is no free evaluation."""
    global CONFIG_COUNT
    CONFIG_COUNT += 1
    result = run_period(strategy, df, start, end, market=market, start_balance=1_000.0)
    return result, compute_metrics(result)


def load_control_assets() -> tuple[Asset, Asset]:
    """BTC and ETH, truncated at CONTROL's end BEFORE any other line touches
    them (R-59's own precedent), so no 2023+ bar of either is ever loaded into
    a variable this module can backtest."""
    end_ts = pd.Timestamp(CONTROL[1], tz="UTC")
    btc_df, _label = load_dataset(DATA_DIR, "spot")
    btc_df = btc_df[btc_df.index <= end_ts]
    eth_df = load_coinbase_spot(DATA_DIR, "ETH")
    eth_df = eth_df[eth_df.index <= end_ts]
    btc = Asset("BTC", btc_df, coverage=1.0, max_gap=pd.Timedelta(0), qualifies=True)
    eth = Asset("ETH", eth_df, coverage=1.0, max_gap=pd.Timedelta(0), qualifies=True)
    return btc, eth


# --------------------------------------------------------- OU half-life fit


def ou_halflife(df: pd.DataFrame, start: str, end: str) -> tuple[float, float, int]:
    """Section 1's formula, exact. Returns (theta_per_day, halflife_days, n_obs).

    halflife is NaN if theta <= 0 (no mean reversion detected at the daily
    scale over this window) -- named as a possible outcome in the
    pre-registration above, though it did not occur on any of the 8 assets.
    """
    global OU_FIT_COUNT
    OU_FIT_COUNT += 1
    lo = int(df.index.searchsorted(start))
    hi = int(df.index.searchsorted(end, side="right"))
    window = df.iloc[lo:hi]
    daily = window["close"].resample("1D").last().dropna()
    logp = np.log(daily.to_numpy(dtype=float))
    mu = float(logp.mean())
    x = logp[:-1] - mu
    y = logp[1:] - logp[:-1]
    beta = float(np.sum(x * y) / np.sum(x * x))
    theta = -beta
    halflife = float(np.log(2.0) / theta) if theta > 0 else float("nan")
    return theta, halflife, len(daily)


def compute_horizons(all_assets: dict[str, pd.DataFrame]) -> tuple[dict, list[dict]]:
    """Section 1+2+3's formula applied to all 8 assets. `all_assets` maps
    ticker -> the asset's own dataframe (already truncated for BTC/ETH)."""
    rows = []
    theta_hl = {}
    for ticker, df in all_assets.items():
        window = CONTROL if ticker in ("BTC", "ETH") else PANEL_TRAIN
        theta, hl, n = ou_halflife(df, *window)
        theta_hl[ticker] = (theta, hl, n)

    hl_btc = theta_hl[REF_ASSET][1]
    horizons: dict[str, tuple[float, float, float]] = {}
    for ticker, (theta, hl, n) in theta_hl.items():
        ratio_raw = hl / hl_btc
        ratio_clamped = float(np.clip(ratio_raw, CLAMP_LO, CLAMP_HI))
        clamped_flag = "CEILING" if ratio_raw > CLAMP_HI else ("FLOOR" if ratio_raw < CLAMP_LO else "-")
        h = tuple(round(b * ratio_clamped, 3) for b in BASE_HORIZONS)
        horizons[ticker] = h
        rows.append({
            "asset": ticker, "theta_per_day": theta, "halflife_days": hl,
            "n_daily_obs": n, "ratio_raw": ratio_raw, "ratio_clamped": ratio_clamped,
            "clamp": clamped_flag, "horizon_short": h[0], "horizon_mid": h[1],
            "horizon_long": h[2],
        })
        print(f"  {ticker:5s} theta={theta:.5f}/day  halflife={hl:7.2f}d  n={n:4d}  "
              f"ratio={ratio_raw:6.3f} -> clamped={ratio_clamped:5.3f} [{clamped_flag:7s}]  "
              f"horizons=({h[0]:.1f},{h[1]:.1f},{h[2]:.1f})d")
    return horizons, rows


def make_candidate(horizons: tuple[float, ...]) -> KellyRegimeV4:
    """`kelly_regime_v4` with a rescaled anchor ladder and a correctly-extended
    warmup (section 5's fix) -- everything else is the shipped default."""
    strat = KellyRegimeV4(horizons=horizons)
    strat.warmup = int(round(max(horizons) * BARS_PER_DAY)) + 10
    return strat


# ------------------------------------------------------------------ causality


def cmd_causality(horizons_map: dict, probe_assets: list[Asset]) -> bool:
    """R-57/R-59's tamper-probe methodology, adapted: constructs
    `KellyRegimeV4(horizons=horizons_i)` directly (this branch's candidate
    never goes through the registry for non-BTC assets)."""
    print("=" * 100)
    print("CAUSALITY TAMPER PROBE -- calibrated kelly_regime_v4(horizons=horizons_i)")
    print("=" * 100)
    market = MarketSpec.futures(leverage=5.0)
    all_ok = True
    for a in probe_assets:
        h = horizons_map[a.ticker]
        tail = a.df.iloc[-60_000:].copy()
        cut = len(tail) - 5_000
        bars = [cut - k for k in (1, 2, 3, 5, 10, 20)]
        up, down = tail.copy(), tail.copy()
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

        def decisions(frame):
            s = make_candidate(h)
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
        print(f"  {a.ticker:5s} horizons=({h[0]:.1f},{h[1]:.1f},{h[2]:.1f}) decisions identical "
              f"under opposite post-cut tampers: {'PASS' if ok else 'FAIL'}")
    return all_ok


# ------------------------------------------------------------------ D1/D2/D4/D5 cell


def cell(a: Asset, strategy, window, market, label: str, rows: list) -> dict:
    """One asset x window x market cell: candidate, buy_and_hold, matched hold,
    paired-bootstrap intervals. Identical structure to R-57/R-59's `cell()`."""
    start, end = window
    cand_res, cand = measure(strategy, a.df, start, end, market)
    hold_res, hold = measure(get_strategy("buy_and_hold"), a.df, start, end, market)

    c_mean = mean_notional(cand_res)
    mh_res, mh = measure(ConstantExposureHold(c_mean), a.df, start, end, market)

    cand_ret = daily_returns(cand_res.equity).to_numpy(dtype=float)
    mh_ret = daily_returns(mh_res.equity).to_numpy(dtype=float)
    hold_ret = daily_returns(hold_res.equity).to_numpy(dtype=float)
    n = min(len(cand_ret), len(mh_ret), len(hold_ret))
    dd_matched = paired_bootstrap(cand_ret[:n], mh_ret[:n], max_drawdown_from_returns, **BOOT_KW)
    growth_matched = paired_bootstrap(cand_ret[:n], mh_ret[:n], total_log_return, **BOOT_KW)

    row = {
        "asset": a.ticker, "window": label, "market": market.name,
        "fee": market.fee_rate,
        "cand_final": cand.final_balance, "cand_dd": cand.max_drawdown_pct,
        "cand_sharpe": cand.sharpe, "cand_trades": cand.num_trades,
        "cand_liq": cand.liquidated,
        "hold_final": hold.final_balance, "hold_dd": hold.max_drawdown_pct,
        "hold_sharpe": hold.sharpe, "hold_liq": hold.liquidated,
        "c_mean_notional": c_mean,
        "mh_final": mh.final_balance, "mh_dd": mh.max_drawdown_pct,
        "mh_sharpe": mh.sharpe,
        "dd_matched_diff": dd_matched.diff.point,
        "dd_matched_lo": dd_matched.diff.lo, "dd_matched_hi": dd_matched.diff.hi,
        "growth_matched_diff": growth_matched.diff.point,
        "growth_matched_lo": growth_matched.diff.lo,
        "growth_matched_hi": growth_matched.diff.hi,
    }
    rows.append(row)
    print(f"  {a.ticker:5s} {label:11s} {market.name:11s} fee={market.fee_rate:.2%}  "
          f"cand ${cand.final_balance:>10,.0f} DD {cand.max_drawdown_pct:5.1f}% | "
          f"hold ${hold.final_balance:>10,.0f} DD {hold.max_drawdown_pct:5.1f}% | "
          f"matched(c={c_mean:.2f}) ${mh.final_balance:>10,.0f} "
          f"DD {mh.max_drawdown_pct:5.1f}% | "
          f"dDD_matched {dd_matched.diff.point:+6.1f}pp "
          f"[{dd_matched.diff.lo:+6.1f},{dd_matched.diff.hi:+6.1f}]")
    return row


# --------------------------------------------------------- D3 crash-transition-lag


def flip_to_flat_events(target: np.ndarray) -> list[int]:
    """Bars where the target signal drops to (near) zero from meaningfully
    nonzero -- de-risking events. Identical definition to R-56/R-59's own."""
    out = []
    for i in range(1, len(target)):
        if abs(target[i - 1]) > 0.05 and abs(target[i]) < 1e-9:
            out.append(i)
    return out


def bars_to_flatten(df: pd.DataFrame, strategy, window: tuple[str, str],
                     prefix_days: int = 200) -> float | None:
    """Bars from the window's own start to the strategy's own first
    flip-to-flat event inside the window. `prefix_days` is a fixed, generous
    warm prefix (>= the largest possible clamped horizon, 112 days) so the
    anchor ladder is fully valid by the time the window starts, independent of
    which of the two signals (candidate/baseline) is being measured -- this
    keeps both signals on identical footing rather than each getting its own
    (possibly different) warmup-derived prefix length."""
    start, end = window
    lo = int(df.index.searchsorted(start))
    hi = int(df.index.searchsorted(end, side="right"))
    prefix = min(lo, prefix_days * BARS_PER_DAY)
    frame = df.iloc[lo - prefix: hi].copy()
    prepared = strategy.prepare(frame)
    target = prepared["target"].to_numpy(dtype=float)
    window_start_idx = prefix
    events = [i for i in flip_to_flat_events(target) if i >= window_start_idx]
    if not events:
        return None
    event_ts = frame.index[events[0]]
    window_start_ts = frame.index[window_start_idx]
    return float((event_ts - window_start_ts) / pd.Timedelta(minutes=5))


def cmd_crash_lag(btc: Asset, eth: Asset, horizons_map: dict) -> dict:
    """D3, the frozen gate: BTC only, the three CRASH_WINDOWS, candidate vs
    v4's unmodified baseline. BTC's candidate horizons equal (20,40,80) EXACTLY
    (ratio_BTC = 1.0 by construction, see pre-registration section 2), so this
    cell is a trivial identity check -- reported in full below and named as
    such, not silently glossed over. A non-gating ETH supplementary check
    (2 of the 3 windows -- ETH has no 2018 data) is reported alongside for
    context only, since ETH's ratio is NOT 1.0 and is therefore a genuine,
    if secondary, test of the mechanism."""
    global SIGNAL_ONLY_COUNT
    print("=" * 100)
    print("D3 (CRASH-TRANSITION-LAG, GATING) -- BTC, three CRASH_WINDOWS, spot")
    print("=" * 100)
    baseline = make_candidate(BASE_HORIZONS)
    cand = make_candidate(horizons_map["BTC"])
    btc_rows = []
    for label, window in CRASH_WINDOWS.items():
        b_lag = bars_to_flatten(btc.df, baseline, window)
        c_lag = bars_to_flatten(btc.df, cand, window)
        SIGNAL_ONLY_COUNT += 2
        btc_rows.append({"window": label, "baseline_lag_bars": b_lag, "candidate_lag_bars": c_lag})
        print(f"  BTC {label:15s} baseline lag={b_lag} bars   candidate lag={c_lag} bars")

    valid = [r for r in btc_rows if r["baseline_lag_bars"] is not None and r["candidate_lag_bars"] is not None]
    baseline_mean = float(np.mean([r["baseline_lag_bars"] for r in valid])) if valid else float("nan")
    candidate_mean = float(np.mean([r["candidate_lag_bars"] for r in valid])) if valid else float("nan")
    passes = d3_passes(candidate_mean, baseline_mean) if valid else False
    print(f"\n  BTC mean lag: baseline={baseline_mean:.2f} bars, candidate={candidate_mean:.2f} bars "
          f"-> {'PASS' if passes else 'FAIL'} (tolerance +{D3_MAX_EXTRA_LAG_BARS} bars)")
    print("  (trivial by construction: BTC's candidate horizons == (20,40,80) exactly, "
          "since ratio_BTC = halflife_BTC / halflife_BTC = 1.0)")

    print("\n" + "-" * 100)
    print("D3 SUPPLEMENTARY (non-gating, robustness only) -- ETH, 2 of 3 CRASH_WINDOWS "
          "(no 2018 ETH data)")
    print("-" * 100)
    eth_cand = make_candidate(horizons_map["ETH"])
    eth_rows = []
    for label, window in CRASH_WINDOWS.items():
        if label == "2018-11":
            print(f"  ETH {label:15s} skipped -- no ETH data before 2019-03-14")
            continue
        b_lag = bars_to_flatten(eth.df, baseline, window)
        c_lag = bars_to_flatten(eth.df, eth_cand, window)
        SIGNAL_ONLY_COUNT += 2
        eth_rows.append({"window": label, "baseline_lag_bars": b_lag, "candidate_lag_bars": c_lag})
        print(f"  ETH {label:15s} baseline lag={b_lag} bars   candidate lag={c_lag} bars")

    return {"btc_rows": btc_rows, "eth_rows": eth_rows, "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean, "passes": passes}


# ---------------------------------------------------------------------- run


def cmd_run() -> None:
    panel = load_panel()
    btc, eth = load_control_assets()
    all_assets_df = {"BTC": btc.df, "ETH": eth.df}
    all_assets_df.update({a.ticker: a.df for a in panel})

    print()
    print("=" * 100)
    print("OU HALF-LIFE CALIBRATION -- PANEL_TRAIN/CONTROL (2020-04-01 -> 2022-12-31), "
          "structural, price-only")
    print("=" * 100)
    horizons_map, calib_rows = compute_horizons(all_assets_df)

    print()
    causality_ok = cmd_causality(horizons_map, [btc] + panel[:3])
    if not causality_ok:
        raise SystemExit("CAUSALITY PROBE FAILED -- refusing to report D1-D5 "
                         "results until the lookahead bug is fixed.")

    print("\n" + "=" * 100)
    print("D1 (PRIMARY) -- PANEL_TRAIN, spot @0.10%, OU-rescaled horizons per asset")
    print("=" * 100)
    d1_rows: list[dict] = []
    for a in panel:
        strat = make_candidate(horizons_map[a.ticker])
        cell(a, strat, PANEL_TRAIN, SPOT_BASE, "PANEL_TRAIN", d1_rows)

    print("\n" + "=" * 100)
    print("D2 (FALSIFICATION CONTROL) -- CONTROL window, BTC and ETH, OU-rescaled horizons")
    print("=" * 100)
    d2_rows: list[dict] = []
    for a in (btc, eth):
        strat = make_candidate(horizons_map[a.ticker])
        cell(a, strat, CONTROL, SPOT_BASE, "CONTROL", d2_rows)

    print()
    d3_result = cmd_crash_lag(btc, eth, horizons_map)

    print("\n" + "=" * 100)
    print("D4 (GENERALIZATION, reported not gating) -- PANEL_TEST, spot @0.10%, frozen horizons")
    print("=" * 100)
    d4_rows: list[dict] = []
    for a in panel:
        strat = make_candidate(horizons_map[a.ticker])
        cell(a, strat, PANEL_TEST, SPOT_BASE, "PANEL_TEST", d4_rows)

    print("\n" + "=" * 100)
    print("D5 (0.40% FEE FALSIFICATION) -- PANEL_TRAIN, spot @0.40%, frozen horizons")
    print("=" * 100)
    d5_rows: list[dict] = []
    for a in panel:
        strat = make_candidate(horizons_map[a.ticker])
        cell(a, strat, PANEL_TRAIN, SPOT_REAL, "PANEL_TRAIN", d5_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(calib_rows).to_csv(OUT_DIR / "calibration.csv", index=False)
    pd.DataFrame(d1_rows).to_csv(OUT_DIR / "d1_panel_train.csv", index=False)
    pd.DataFrame(d2_rows).to_csv(OUT_DIR / "d2_control.csv", index=False)
    pd.DataFrame(d3_result["btc_rows"]).to_csv(OUT_DIR / "d3_btc_crash_lag.csv", index=False)
    pd.DataFrame(d3_result["eth_rows"]).to_csv(OUT_DIR / "d3_eth_supplementary.csv", index=False)
    pd.DataFrame(d4_rows).to_csv(OUT_DIR / "d4_panel_test.csv", index=False)
    pd.DataFrame(d5_rows).to_csv(OUT_DIR / "d5_fee_falsification.csv", index=False)

    verdicts(horizons_map, d1_rows, d2_rows, d3_result, d4_rows, d5_rows, len(panel))
    print(f"\nBacktest configurations (bucket a, run_period/measure()): {CONFIG_COUNT}")
    print(f"OU half-life regressions (bucket b, closed-form, no iteration): {OU_FIT_COUNT}")
    print(f"prepare()-only D3 signal evaluations (bucket c, not backtests): {SIGNAL_ONLY_COUNT}")
    print(f"Total evaluations, all buckets: {CONFIG_COUNT + OU_FIT_COUNT + SIGNAL_ONLY_COUNT}")
    print("Holdout consultations added by this round: 0 "
          "(BTC/ETH truncated at 2022-12-31 before any other line touches them; "
          "panel-asset reads cost +0 per the pre-registration; the D3 crash windows "
          "are all pre-2023 and already read by every registered strategy's own backtest)")


def verdicts(horizons_map: dict, d1_rows: list[dict], d2_rows: list[dict],
             d3_result: dict, d4_rows: list[dict], d5_rows: list[dict], n: int) -> None:
    print("\n" + "=" * 100)
    print("PRE-REGISTERED DECISION RULES (experiments/r60_shared.py)")
    print("=" * 100)

    d1 = pd.DataFrame(d1_rows)
    k1 = int((d1.cand_dd < d1.mh_dd).sum())
    excl = int(((d1.dd_matched_lo > 0) | (d1.dd_matched_hi < 0)).sum())
    better_excl = int((d1.dd_matched_hi < 0).sum())
    p1 = binomial_tail(k1, n)
    print(f"D1 (primary, matched-exposure drawdown, PANEL_TRAIN, spot @0.10%): "
          f"{k1}/{n} -> {d1_verdict(k1, n)} (exact binomial p={p1:.4f})")
    print(f"    paired bootstrap: {excl}/{n} intervals exclude zero "
          f"({better_excl}/{n} of them in the candidate's favour)")

    d2 = pd.DataFrame(d2_rows).set_index("asset")
    dd_advantage = {t: float(d2.loc[t, "dd_matched_diff"]) for t in ("BTC", "ETH")}
    print(f"D2 (falsification control, CONTROL window): "
          f"BTC {dd_advantage['BTC']:+.1f}pp (R-57: {R57_CONTROL_DD_ADVANTAGE['BTC']:+.1f}pp), "
          f"ETH {dd_advantage['ETH']:+.1f}pp (R-57: {R57_CONTROL_DD_ADVANTAGE['ETH']:+.1f}pp), "
          f"tolerance {D2_REGRESSION_TOLERANCE_PP:+.1f}pp -> "
          f"{'PASSES' if d2_passes(dd_advantage) else 'FAILS'}")

    print(f"D3 (crash-transition-lag, BTC, gating): baseline={d3_result['baseline_mean']:.2f} bars, "
          f"candidate={d3_result['candidate_mean']:.2f} bars -> "
          f"{'PASSES' if d3_result['passes'] else 'FAILS'}")

    d4 = pd.DataFrame(d4_rows)
    k4 = int((d4.cand_dd < d4.mh_dd).sum())
    print(f"D4 (generalization, PANEL_TEST, descriptive): {k4}/{n} -> {d1_verdict(k4, n)}")

    d5 = pd.DataFrame(d5_rows)
    k5 = int((d5.cand_final > d5.hold_final).sum())
    print(f"D5 (0.40% fee falsification, beats buy_and_hold final balance): "
          f"{k5}/{n} -> {'SURVIVES' if k5 >= n - 1 else 'FAILS (as predicted)'}")

    verdict = "PROMOTE-CANDIDATE" if promoted(k1, dd_advantage, d3_result["candidate_mean"],
                                              d3_result["baseline_mean"]) else "NEGATIVE"
    print(f"\nOVERALL (promoted(k1, dd_advantage, candidate_lag, baseline_lag) "
          f"mechanically applied): {verdict}")
    print("horizons_i: " + ", ".join(
        f"{k}=({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})" for k, v in horizons_map.items()))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "halflife":
        panel = load_panel()
        btc, eth = load_control_assets()
        all_assets_df = {"BTC": btc.df, "ETH": eth.df}
        all_assets_df.update({a.ticker: a.df for a in panel})
        compute_horizons(all_assets_df)
        print(f"\nOU half-life regressions evaluated: {OU_FIT_COUNT}")
        return
    if cmd == "causality":
        panel = load_panel()
        btc, eth = load_control_assets()
        all_assets_df = {"BTC": btc.df, "ETH": eth.df}
        all_assets_df.update({a.ticker: a.df for a in panel})
        horizons_map, _ = compute_horizons(all_assets_df)
        print()
        ok = cmd_causality(horizons_map, [btc] + panel[:3])
        print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
        return
    if cmd == "run":
        cmd_run()
        return
    raise SystemExit(f"unknown command {cmd!r} (halflife | causality | run)")


if __name__ == "__main__":
    main()
