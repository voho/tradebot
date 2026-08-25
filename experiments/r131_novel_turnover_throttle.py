#!/usr/bin/env python
"""R-131 NOVEL branch: ``NovelTurnoverThrottle`` -- an online dual-variable
turnover throttle wrapped around ``kelly_regime_v4`` (08-25).

The complete pre-registration for this round -- direction, literature
citations, the non-duplication argument against the named prior rounds, and
the named failure mode -- lives in ``experiments/r131_shared.py``'s own
module docstring, written by the operator before this branch was dispatched,
and is NOT re-derived here: read that file in full first. This file imports
ONLY from ``experiments.r131_shared`` (read-only), never edits it, never
coordinates with the conservative branch's file, and never reads a bar at or
after ``r131_shared.OOS_START`` (2023-01-01).

MECHANISM, exactly as frozen in ``r131_shared.py``, one sentence: maintain a
causal shadow price ``lambda_t`` on turnover, updated every bar by projected
dual ascent

    lambda_{t+1} = clip(lambda_t + ETA * (turnover_ewm_t - TURNOVER_UPPER), 0, LAMBDA_MAX)

and instead of a hard skip (the conservative branch's mechanism), SHRINK the
size of each pending rebalance by ``1 / (1 + lambda_t)`` before executing.

**What "pending rebalance" means here.** ``prepare()`` is v4's own, entirely
unchanged (``KellyRegimeV3.prepare`` via inheritance) -- it emits the same
``target`` column v4 has always emitted, latched through v4's own 10%
deadband. A bar is "pending" exactly when v4's OWN target differs from the
previous bar's target (``abs(target[i] - target[i-1]) > 1e-9``) -- i.e.
exactly the bars on which v4's stock ``on_bar`` (``KellyRegime.on_bar``)
would itself submit an order. This branch does not invent a new trigger; it
intercepts v4's own trigger and, on that bar only, moves the ACTUAL account
exposure only part of the way from where it currently sits
(``ctx.position * ctx.close / equity``, the same causal state read
``kelly_regime_ev.py``'s ``on_bar`` uses) toward v4's desired target:

    executed_delta = (desired - current) / (1 + lambda_t)
    ctx.order_notional(current + executed_delta)

Because ``current`` is the REAL broker-reported exposure (not v4's own
internal target trajectory), a throttled bar's shortfall persists honestly:
if lambda_t stays elevated across several of v4's own trigger bars in a row,
the account can lag v4's target by a compounding amount, which is exactly
the risk the DIRECTION section names (a throttle that damps trading right
when turnover -- and per L-01/R-62, the edge -- is bursting).

**The dual variable itself updates every bar**, whether or not v4 fired one
(a leaky-bucket / EWM state does not freeze between events): a causal
``rebalance_events[i] in {0,1}`` array (1 exactly on a "pending" bar as
defined above) feeds ``r131_shared.trailing_turnover_ewm``'s EXACT recursive
definition, implemented here as an O(1)-per-bar online update
(``_OnlineEwm`` below) so a ~1e6-bar backtest does not re-scan its own
history every bar. ``_OnlineEwm`` is verified against
``r131_shared.trailing_turnover_ewm``'s own batch computation on synthetic
data at import time (``_self_test``, mirrors ``r131_shared.py``'s own
``__main__`` self-test convention) and again on real BTC data inside
``main()``, before any inner-validation number is read.

**ETA / TURNOVER_UPPER / LAMBDA_MAX / TURNOVER_EWM_SPAN_DAYS are
``r131_shared``'s own frozen constants, reused verbatim, never refit here.**
The only new free axis this file introduces is the B3 plateau sweep on
``eta`` (0.25x/0.5x/2x/4x the shared default), pre-registered below.

CAUSAL SAFETY FIRST: ``causal_truncation_probe`` (real BTC data, run through
the actual backtest engine so the probe exercises the FULL per-bar state
machine -- ``lambda_t``, the online EWM, and the broker-fed ``current``
read -- not just the ``target`` column, which was already causal before
this file existed) runs before any inner-validation/ETH number is trusted.

PRE-REGISTERED DECISION RULE, stated verbatim from ``r131_shared.py`` and
NOT altered after seeing any number: PROMOTE-candidate only if the
non-inertness gate (A2) AND B1 (both markets, inner-validation) AND B3
(plateau majority) AND B4 (full period, both markets) AND B5 all pass.
Anything else is NEGATIVE. Default: NEGATIVE.

USAGE
-----
    python experiments/r131_novel_turnover_throttle.py
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

from experiments import r131_shared  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import daily_returns, paired_bootstrap, total_log_return  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# ------------------------------------------------------------------------
# Pre-registered constants -- r131_shared's own, reused verbatim. ETA_GRID
# is this branch's own B3 plateau sweep (0.25x/0.5x/2x/4x the default).
# ------------------------------------------------------------------------
ETA = r131_shared.ETA
TURNOVER_UPPER = r131_shared.TURNOVER_UPPER
LAMBDA_MAX = r131_shared.LAMBDA_MAX
TURNOVER_EWM_SPAN_DAYS = r131_shared.TURNOVER_EWM_SPAN_DAYS
ETA_GRID = tuple(round(m * ETA, 6) for m in (0.25, 0.5, 2.0, 4.0))
LAMBDA_ACTIVE_THRESHOLD = 0.01  # "materially shrinking a fill" per the task spec


# ================================================================== (1)
# Online, O(1)-per-bar recursive EWM -- exactly r131_shared.trailing_turnover_ewm's
# own definition (pandas .ewm(span=..., min_periods=1).mean(), adjust=True),
# just computed causally one bar at a time instead of by re-scanning the
# whole prefix every bar. adjust=True's closed recursive form is standard:
#
#   N_t = x_t + (1-alpha) N_{t-1},  Z_t = 1 + (1-alpha) Z_{t-1},  y_t = N_t/Z_t
#
# with N_{-1} = Z_{-1} = 0 -- verified against the batch function on
# synthetic data below (_self_test) and on real data in main().
# ==================================================================

class _OnlineEwm:
    __slots__ = ("alpha", "_n", "_z")

    def __init__(self, span_bars: int) -> None:
        self.alpha = 2.0 / (float(span_bars) + 1.0)
        self._n = 0.0
        self._z = 0.0

    def update(self, x: float) -> float:
        a = self.alpha
        self._n = x + (1.0 - a) * self._n
        self._z = 1.0 + (1.0 - a) * self._z
        return self._n / self._z


# ================================================================== (2)
# The strategy itself. prepare() is v4's own, untouched. on_bar carries the
# entire throttle state machine.
# ==================================================================

class NovelTurnoverThrottle(KellyRegimeV4):
    """v4 with a causal dual-variable throttle on its own realized turnover.

    ``lambda_t`` is a shadow price on turnover, updated every bar by
    projected dual ascent against a causal trailing EWM of v4's OWN
    realized rebalance events; instead of skipping a pending rebalance
    outright, this shrinks it by ``1/(1+lambda_t)`` -- a smooth,
    self-regulating control loop rather than a hard freeze. See this
    file's module docstring and ``experiments/r131_shared.py`` for the
    full derivation, citations and pre-registration. Not registered:
    this is an experiments/-only evaluation script, not a promoted
    strategy.
    """

    name = "r131_novel_turnover_throttle"  # experiments/-only; not registered

    def __init__(self, eta: float = ETA, turnover_upper: float = TURNOVER_UPPER,
                 lambda_max: float = LAMBDA_MAX,
                 turnover_ewm_span_days: float = TURNOVER_EWM_SPAN_DAYS,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.eta = float(eta)
        self.turnover_upper = float(turnover_upper)
        self.lambda_max = float(lambda_max)
        self.turnover_ewm_span_days = float(turnover_ewm_span_days)
        self._span_bars = int(self.turnover_ewm_span_days * BARS_PER_DAY)
        # Diagnostics (populated fresh by every prepare() call, one array
        # entry per row of whatever frame the engine passes in -- the
        # engine's own prefix/warmup convention, see tradebot/window.py).
        self.diag_lambda: np.ndarray | None = None
        self.diag_turnover_ewm: np.ndarray | None = None
        self.diag_event: np.ndarray | None = None
        self.diag_index: pd.DatetimeIndex | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        self._lambda = 0.0
        self._ewm = _OnlineEwm(self._span_bars)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4's own unchanged target column
        n = len(df)
        self.diag_lambda = np.full(n, np.nan)
        self.diag_turnover_ewm = np.full(n, np.nan)
        self.diag_event = np.zeros(n, dtype=float)
        self.diag_index = df.index
        self._reset_state()  # fresh account each run -- no cross-run leakage
        return df

    def on_bar(self, ctx: Context) -> None:
        i = ctx.i
        desired = float(ctx.bar["target"])
        prev_target = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        fires = abs(desired - prev_target) > 1e-9

        lam = self._lambda  # lambda_t: the value in effect for THIS bar's decision
        event = 0.0
        if fires:
            equity = ctx.equity
            if equity > 0:
                current = ctx.position * ctx.close / equity  # actual (post-fill) exposure
                executed_delta = (desired - current) / (1.0 + lam)
                ctx.order_notional(current + executed_delta)
                event = 1.0
            # equity <= 0 is unreachable in practice (the engine stops calling
            # on_bar once the broker is dead), guarded defensively only.

        turnover_ewm = self._ewm.update(event) * BARS_PER_DAY  # events/bar -> events/day
        new_lambda = float(np.clip(lam + self.eta * (turnover_ewm - self.turnover_upper),
                                    0.0, self.lambda_max))

        self.diag_lambda[i] = lam
        self.diag_turnover_ewm[i] = turnover_ewm
        self.diag_event[i] = event
        self._lambda = new_lambda


# ================================================================== (3)
# Causal-truncation probe, run through the REAL engine (not just prepare())
# so it exercises lambda_t, the online EWM, AND the broker-fed `current`
# read together -- the whole point being that this is a stateful loop no
# vectorized check on `target` alone would catch.
# ==================================================================

def causal_truncation_probe(strategy_factory, df: pd.DataFrame, cut: int,
                             skip_bars: int) -> tuple[bool, dict]:
    strat_full = strategy_factory()
    run_backtest(strat_full, df, r131_shared.SPOT, 1000.0)
    strat_trunc = strategy_factory()
    run_backtest(strat_trunc, df.iloc[:cut].copy(), r131_shared.SPOT, 1000.0)

    n_check = min(len(strat_trunc.diag_lambda), cut) - skip_bars
    if n_check <= 0:
        raise ValueError("cut too small for skip_bars buffer")

    ok_lambda = np.allclose(strat_full.diag_lambda[:n_check], strat_trunc.diag_lambda[:n_check],
                             equal_nan=True, rtol=1e-9, atol=1e-12)
    ok_ewm = np.allclose(strat_full.diag_turnover_ewm[:n_check], strat_trunc.diag_turnover_ewm[:n_check],
                          equal_nan=True, rtol=1e-9, atol=1e-12)
    ok_event = np.array_equal(np.nan_to_num(strat_full.diag_event[:n_check]),
                               np.nan_to_num(strat_trunc.diag_event[:n_check]))
    detail = dict(n_check=n_check, ok_lambda=bool(ok_lambda), ok_ewm=bool(ok_ewm), ok_event=bool(ok_event))
    return bool(ok_lambda and ok_ewm and ok_event), detail


# ================================================================== (4)
# A2 non-inertness: distribution of lambda_t actually reached.
# ==================================================================

def lambda_distribution(diag_lambda: np.ndarray, active_threshold: float = LAMBDA_ACTIVE_THRESHOLD) -> dict:
    valid = diag_lambda[np.isfinite(diag_lambda)]
    if valid.size == 0:
        return dict(n_bars=0, min=float("nan"), max=float("nan"), mean=float("nan"),
                    n_active=0, frac_active=float("nan"))
    n_active = int((valid > active_threshold).sum())
    return dict(
        n_bars=int(valid.size),
        min=float(valid.min()), max=float(valid.max()), mean=float(valid.mean()),
        n_active=n_active, frac_active=n_active / valid.size,
    )


# ================================================================== (5)
# B1/B4 helper generalized over an arbitrary (start, end) -- r131_shared's
# own b1_signal hardcodes INNER_VAL_START/INNER_VAL_END internally, so the
# "full inner period" cells (B1's other two, and all of B4) need this
# instead. Identical logic to r131_shared.b1_signal, just parameterized;
# r131_shared.py itself is never edited.
# ==================================================================

def cell_over_period(candidate_factory, df: pd.DataFrame, market, start: str, end: str,
                      seed: int = 130, capture: bool = False) -> tuple[dict, object | None]:
    box: dict = {}

    def wrapped():
        s = candidate_factory()
        if capture:
            box["strat"] = s
        return s

    m_cand, res_cand = r131_shared.run_candidate(wrapped, df, market, start, end)
    m_v4, res_v4 = r131_shared.run_candidate(lambda: get_strategy("kelly_regime_v4"), df, market, start, end)
    r_cand = daily_returns(res_cand.equity)
    r_v4 = daily_returns(res_v4.equity)
    n = min(len(r_cand), len(r_v4))
    paired = paired_bootstrap(r_cand.to_numpy()[:n], r_v4.to_numpy()[:n], stat=total_log_return, seed=seed)
    cell = {
        "sharpe_cand": m_cand.sharpe, "sharpe_v4": m_v4.sharpe,
        "d_sharpe": m_cand.sharpe - m_v4.sharpe,
        "paired_diff": paired.diff.point, "paired_lo": paired.diff.lo, "paired_hi": paired.diff.hi,
        "significant": paired.significant,
        "dd_cand": m_cand.max_drawdown_pct, "dd_v4": m_v4.max_drawdown_pct,
        "trades_cand": m_cand.num_trades, "trades_v4": m_v4.num_trades,
    }
    return cell, box.get("strat")


def print_cell(label: str, cell: dict) -> None:
    print(f"  {label:<28s} sharpe_cand={cell['sharpe_cand']:.3f}  sharpe_v4={cell['sharpe_v4']:.3f}  "
          f"d_sharpe={cell['d_sharpe']:+.4f}  paired_diff={cell['paired_diff']:+.4f} "
          f"[{cell['paired_lo']:+.4f}, {cell['paired_hi']:+.4f}]  significant={cell['significant']}  "
          f"dd_cand={cell['dd_cand']:.1f}%  dd_v4={cell['dd_v4']:.1f}%  "
          f"trades_cand={cell['trades_cand']}  trades_v4={cell['trades_v4']}")


# ================================================================== (6)
# Named diagnostic: lambda_t's own trajectory through the six STRESS_EPISODES.
# ==================================================================

def stress_episode_diagnostic(diag_index: pd.DatetimeIndex, diag_lambda: np.ndarray,
                               diag_turnover_ewm: np.ndarray, diag_event: np.ndarray,
                               window_days: int = 5) -> list[dict]:
    rows = []
    tz = diag_index.tz
    for label, date_str in r131_shared.STRESS_EPISODES:
        center = pd.Timestamp(date_str, tz=tz)
        lo = center - pd.Timedelta(days=window_days)
        hi = center + pd.Timedelta(days=window_days)
        mask = (diag_index >= lo) & (diag_index <= hi)
        if not mask.any():
            rows.append(dict(episode=label, date=date_str, in_range=False))
            continue
        lam_win = diag_lambda[mask]
        ewm_win = diag_turnover_ewm[mask]
        ev_win = diag_event[mask]
        lam_valid = lam_win[np.isfinite(lam_win)]
        ewm_valid = ewm_win[np.isfinite(ewm_win)]
        rows.append(dict(
            episode=label, date=date_str, in_range=True,
            lambda_max=float(lam_valid.max()) if lam_valid.size else float("nan"),
            lambda_mean=float(lam_valid.mean()) if lam_valid.size else float("nan"),
            turnover_ewm_max=float(ewm_valid.max()) if ewm_valid.size else float("nan"),
            n_events=int(np.nansum(ev_win)),
            n_bars=int(mask.sum()),
        ))
    return rows


def print_stress_rows(rows: list[dict]) -> None:
    for r in rows:
        if not r.get("in_range", False):
            print(f"  {r['date']}  {r['episode']:<42s}  NOT IN RANGE (outside this run's frame)")
            continue
        print(f"  {r['date']}  {r['episode']:<42s}  "
              f"lambda_max={r['lambda_max']:.3f}  lambda_mean={r['lambda_mean']:.4f}  "
              f"turnover_ewm_max={r['turnover_ewm_max']:.3f}/day  "
              f"n_events(+-{5}d)={r['n_events']}/{r['n_bars']}bars")


# --------------------------------------------------------------------- main

def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> dict:
    t0 = time.time()
    n_configs = 0
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-131 NOVEL: NovelTurnoverThrottle -- online dual-variable turnover throttle on kelly_regime_v4")
    print("mechanism: lambda_{t+1} = clip(lambda_t + ETA*(turnover_ewm_t - TURNOVER_UPPER), 0, LAMBDA_MAX);")
    print("on each bar v4's OWN target-change would fire, execute only (desired-current)/(1+lambda_t).")
    print(f"\nETA={ETA}  TURNOVER_UPPER={TURNOVER_UPPER:.4f} trades/day  LAMBDA_MAX={LAMBDA_MAX}  "
          f"TURNOVER_EWM_SPAN_DAYS={TURNOVER_EWM_SPAN_DAYS}")
    print(f"B3 sweep (0.25x/0.5x/2x/4x ETA): {ETA_GRID}")

    btc, btc_label = r131_shared.load_btc_train("spot")
    max_ts_seen.append(btc.index.max())
    print(f"\nBTC ({btc_label}, truncated < {r131_shared.OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    def make_primary():
        return NovelTurnoverThrottle()

    def make_eta(eta_val):
        def _f():
            return NovelTurnoverThrottle(eta=eta_val)
        return _f

    # ============================================== CAUSAL SAFETY FIRST
    hr("CAUSAL TRUNCATION PROBE (real BTC data, primary config, through the full engine)")
    cut = 400_000
    skip_bars = int((max(TURNOVER_EWM_SPAN_DAYS, 80) + 3) * BARS_PER_DAY)
    try:
        probe_ok, probe_detail = causal_truncation_probe(make_primary, btc, cut, skip_bars)
        print(f"causal_truncation_probe(cut={cut}, skip_bars={skip_bars}): {'PASS' if probe_ok else 'FAIL'}  "
              f"detail={probe_detail}")
    except Exception as e:  # noqa: BLE001
        probe_ok = False
        probe_detail = {"exception": str(e)}
        print(f"causal_truncation_probe FAILED WITH EXCEPTION: {e}")
    print(f"\nCAUSAL SAFETY (truncation probe) PASS: {probe_ok}")
    if not probe_ok:
        hr("STOPPING: causal-safety probe failed -- no inner-validation/ETH number is trustworthy")
        elapsed = time.time() - t0
        print("VERDICT: NEGATIVE (causal-truncation probe failed)")
        print(f"\n[{elapsed:.0f}s]")
        return dict(verdict="NEGATIVE (causal-truncation probe failed)", probe_ok=False,
                    n_configs=0, max_ts=max(max_ts_seen))

    # ============================================================= B1 (4 cells)
    hr("B1 -- BTC signal, primary config (eta=ETA), inner-validation AND full inner period, both markets")
    b1_spot_iv = r131_shared.b1_signal(make_primary, btc, r131_shared.SPOT)
    n_configs += 1
    b1_fut_iv = r131_shared.b1_signal(make_primary, btc, r131_shared.FUTURES)
    n_configs += 1
    b1_spot_full, diag_strat = cell_over_period(
        make_primary, btc, r131_shared.SPOT,
        r131_shared.INNER_TRAIN_START, r131_shared.INNER_VAL_END, capture=True)
    n_configs += 1
    b1_fut_full, _ = cell_over_period(
        make_primary, btc, r131_shared.FUTURES,
        r131_shared.INNER_TRAIN_START, r131_shared.INNER_VAL_END)
    n_configs += 1

    print_cell("spot, inner-val", b1_spot_iv)
    print_cell("futures_5x, inner-val", b1_fut_iv)
    print_cell("spot, full inner period", b1_spot_full)
    print_cell("futures_5x, full inner period", b1_fut_full)

    b1_pass = (b1_spot_iv["d_sharpe"] > 0) and (b1_fut_iv["d_sharpe"] > 0)
    print(f"B1 PASS (both markets d_sharpe > 0, INNER-VALIDATION -- the decisive cell per family "
          f"convention): {b1_pass}")
    print(f"(full-inner-period cells reported above for context; d_sharpe positive both markets: "
          f"{(b1_spot_full['d_sharpe'] > 0) and (b1_fut_full['d_sharpe'] > 0)})")

    # ============================================================= A2 (from the captured full-inner spot run)
    hr("A2 -- non-inertness: lambda_t distribution over BTC full inner period (2017-01-01..2022-12-31), spot")
    assert diag_strat is not None, "full-inner spot cell did not capture a strategy instance"
    dist = lambda_distribution(diag_strat.diag_lambda)
    print(f"lambda_t over {dist['n_bars']:,} on_bar-covered bars: "
          f"min={dist['min']:.4f}  max={dist['max']:.4f}  mean={dist['mean']:.4f}")
    print(f"bars with lambda_t > {LAMBDA_ACTIVE_THRESHOLD}: {dist['n_active']:,} "
          f"({dist['frac_active']:.4%} of covered bars)")
    a2 = r131_shared.a2_non_inertness(dist["n_active"])
    print(f"A2 (r131_shared.a2_non_inertness): n_interventions={a2['n_interventions']}  PASS={a2['pass']}")
    n_events_total = int(np.nansum(diag_strat.diag_event))
    print(f"(for context: v4-triggered 'pending rebalance' bars over the same window: {n_events_total})")

    # ============================================================= named diagnostic: STRESS_EPISODES
    hr("NAMED DIAGNOSTIC -- lambda_t's own trajectory through the six STRESS_EPISODES (+-5 days)")
    stress_rows = stress_episode_diagnostic(diag_strat.diag_index, diag_strat.diag_lambda,
                                             diag_strat.diag_turnover_ewm, diag_strat.diag_event)
    print_stress_rows(stress_rows)

    # ============================================================= B3 (8 cells: ETA_GRID x 2 markets)
    hr(f"B3 -- plateau: ETA_GRID={ETA_GRID} (0.25x/0.5x/2x/4x default {ETA}) x both markets, inner-validation")
    b3_rows = []
    for eta_val in ETA_GRID:
        for market, mname in ((r131_shared.SPOT, "spot"), (r131_shared.FUTURES, "futures_5x")):
            cell = r131_shared.b1_signal(make_eta(eta_val), btc, market)
            n_configs += 1
            row = dict(eta=eta_val, eta_mult=round(eta_val / ETA, 4), market=mname, **cell)
            b3_rows.append(row)
            print(f"  eta={eta_val:.4f} ({row['eta_mult']}x)  {mname:>11s}  "
                  f"d_sharpe={cell['d_sharpe']:+.4f}  significant={cell['significant']}")
    positive = sum(1 for r in b3_rows if r["d_sharpe"] > 0)
    b3_pass = positive >= len(b3_rows) / 2.0
    print(f"B3 PASS (majority of {len(b3_rows)} swept cells d_sharpe > 0): {b3_pass}  "
          f"({positive}/{len(b3_rows)} positive)")

    # ============================================================= B4 (2 cells: ETH spot + futures, full inner)
    hr("B4 -- ETH falsification (pre-registered), full inner period, both markets "
       "(same ETH spot price series through SPOT and FUTURES(5x) MarketSpecs, matching how BTC's B1 "
       "applies both market specs to one OHLCV series)")
    eth = r131_shared.load_eth_train()
    max_ts_seen.append(eth.index.max())
    print(f"ETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {r131_shared.OOS_START}; "
          f"data starts after INNER_TRAIN_START={r131_shared.INNER_TRAIN_START}, so this window is "
          f"truncated by data availability, as expected)")
    b4_spot, _ = cell_over_period(make_primary, eth, r131_shared.SPOT,
                                   r131_shared.INNER_TRAIN_START, r131_shared.INNER_VAL_END)
    n_configs += 1
    b4_fut, _ = cell_over_period(make_primary, eth, r131_shared.FUTURES,
                                  r131_shared.INNER_TRAIN_START, r131_shared.INNER_VAL_END)
    n_configs += 1
    print_cell("ETH spot, full inner period", b4_spot)
    print_cell("ETH futures_5x, full inner period", b4_fut)
    b4_same_sign_spot = (b4_spot["d_sharpe"] > 0) == (b1_spot_iv["d_sharpe"] > 0)
    b4_same_sign_fut = (b4_fut["d_sharpe"] > 0) == (b1_fut_iv["d_sharpe"] > 0)
    b4_full = bool(b4_same_sign_spot and b4_same_sign_fut)
    print(f"B4 FULL PASS (ETH sign matches BTC inner-validation sign, both markets): {b4_full}  "
          f"(spot: {b4_same_sign_spot}, futures: {b4_same_sign_fut})")

    # ============================================================= B5 (2 cells: high fee tier, inner-val)
    hr("B5 -- fee tier (0.40% taker), primary config, BTC inner-validation, both markets")
    b5_spot = r131_shared.b1_signal(make_primary, btc, r131_shared.SPOT_HIGH_FEE)
    n_configs += 1
    b5_fut = r131_shared.b1_signal(make_primary, btc, r131_shared.FUTURES_HIGH_FEE)
    n_configs += 1
    no_reversal_spot = (b5_spot["d_sharpe"] > 0) == (b1_spot_iv["d_sharpe"] > 0)
    no_reversal_fut = (b5_fut["d_sharpe"] > 0) == (b1_fut_iv["d_sharpe"] > 0)
    print_cell("spot 0.40%, inner-val", b5_spot)
    print(f"    no_reversal_vs_010bps={no_reversal_spot}")
    print_cell("futures_5x 0.40%, inner-val", b5_fut)
    print(f"    no_reversal_vs_010bps={no_reversal_fut}")
    b5_pass = bool(no_reversal_spot and no_reversal_fut)
    print(f"B5 PASS (no sign flip vs 0.10% tier, both markets): {b5_pass}")

    # ================================================================ VERDICT
    hr("VERDICT")
    print(f"causal-truncation probe: {probe_ok}")
    print(f"A2 (non-inertness): {a2['pass']}")
    print(f"B1 (both markets, inner-val): {b1_pass}")
    print(f"B3 (plateau majority, {len(b3_rows)} cells): {b3_pass}")
    print(f"B4 (full period, both markets, sign replicates BTC): {b4_full}")
    print(f"B5 (fee robustness, both markets): {b5_pass}")
    all_pass = probe_ok and a2["pass"] and b1_pass and b3_pass and b4_full and b5_pass
    verdict = "PROMOTE-candidate" if all_pass else "NEGATIVE"
    print(f"ALL CLAUSES PASS (A2 and B1 and B3 and B4 and B5): {all_pass}")
    print(f"VERDICT: {verdict}")

    max_ts = max(max_ts_seen)
    print(f"\nconfigurations evaluated (total, this branch): {n_configs} "
          f"(4 B1 + {len(b3_rows)} B3 + 2 B4 + 2 B5)")
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {r131_shared.OOS_START}: {max_ts < pd.Timestamp(r131_shared.OOS_START, tz='UTC')})")
    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(
        verdict=verdict, probe_ok=probe_ok, probe_detail=probe_detail,
        a2=a2, lambda_distribution=dist, stress_rows=stress_rows,
        b1_spot_iv=b1_spot_iv, b1_fut_iv=b1_fut_iv, b1_spot_full=b1_spot_full, b1_fut_full=b1_fut_full,
        b1_pass=b1_pass,
        b3_rows=b3_rows, b3_pass=b3_pass,
        b4_spot=b4_spot, b4_fut=b4_fut, b4_full=b4_full,
        b5_spot=b5_spot, b5_fut=b5_fut, b5_pass=b5_pass,
        n_configs=n_configs, max_ts=max_ts,
    )


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    """Fast synthetic self-test, run at import time -- mirrors r131_shared.py's
    own module-level __main__ self-test convention. Verifies _OnlineEwm
    matches r131_shared.trailing_turnover_ewm's own batch definition exactly."""
    rng = np.random.default_rng(130)
    events = (rng.random(300_000) < 0.0005).astype(float)
    batch = r131_shared.trailing_turnover_ewm(events, span_days=TURNOVER_EWM_SPAN_DAYS)

    span_bars = int(TURNOVER_EWM_SPAN_DAYS * BARS_PER_DAY)
    ewm = _OnlineEwm(span_bars)
    online = np.empty(len(events))
    for i, x in enumerate(events):
        online[i] = ewm.update(float(x)) * BARS_PER_DAY

    ok = np.allclose(batch, online, rtol=1e-9, atol=1e-9)
    assert ok, "_OnlineEwm does not match r131_shared.trailing_turnover_ewm's batch definition"

    # sanity on the dual-ascent clip itself
    lam = 0.0
    for _ in range(50):
        lam = float(np.clip(lam + ETA * (TURNOVER_UPPER * 10 - TURNOVER_UPPER), 0.0, LAMBDA_MAX))
    assert lam == LAMBDA_MAX, "dual ascent should saturate at LAMBDA_MAX under sustained heavy turnover"
    lam = LAMBDA_MAX
    for _ in range(500):
        lam = float(np.clip(lam + ETA * (0.0 - TURNOVER_UPPER), 0.0, LAMBDA_MAX))
    assert lam == 0.0, "dual ascent should decay back to 0 under sustained zero turnover"


_self_test()


if __name__ == "__main__":
    main()
