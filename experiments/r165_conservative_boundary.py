"""R-165 (conservative branch): trade-to-the-boundary applied to `scale`
ALONE, leaving `frac`'s vote and hysteresis untouched.

Pre-registration (shared, frozen, both branches): ``experiments/r165_shared.py``.
That file is not edited here, and neither is anything else outside this file
and ``experiments/reports/r165_conservative_report.md``.

=====================================================================
THE ONE-LINE MECHANISM
=====================================================================

``kelly_regime_v4``'s position loop (in ``KellyRegimeV3.prepare``) is::

    scale   = full[i] if state != 0 else steady[i]     # the vol-target ratio
    desired = frac[i] * scale                          # vote x scale
    if abs(desired - pos) > self.deadband: pos = desired

This branch inserts ONE state variable, ``eff_scale``, between the ``scale``
formula and the product, and lets it track ``scale`` by a proportional-cost
(Constantinides 1986, JPE 94(4); Davis & Norman 1990, Math. OR 15(4))
**no-trade-region-with-boundary-destination** rule instead of copying it::

    gap = scale - eff
    if   gap >  sub: eff = scale - sub      # move only to the near boundary
    elif gap < -sub: eff = scale + sub
    desired = frac[i] * eff                 # <- the only edited line after
    if abs(desired - pos) > self.deadband: pos = desired   # v4's, untouched

``frac`` (the 3-anchor vote), the ``full``/``steady`` arrays, the high/low
breakout hysteresis ``state`` machine and the final ``deadband`` position
update are copied byte-for-byte from ``KellyRegimeV3.prepare``.

=====================================================================
THE k PARAMETERIZATION, AND WHY k=1 MUST BE v4
=====================================================================

The sub-band half-width is::

    sub = (1 - k) * w         w in {0.05, 0.10} = {half, all} of v4's deadband

so that

    k = 1.0  ->  sub = 0      eff == scale at every bar  == kelly_regime_v4
                             EXACTLY (the regression check; ``cmd_causality``
                             asserts bit-for-bit identity of the target path)
    k = 0.75 ->  sub = 0.25 w
    k = 0.50 ->  sub = 0.50 w
    k = 0.25 ->  sub = 0.75 w
    k = 0.0  ->  sub = w      the widest boundary policy in the family: the
                             effective scale is only ever dragged along, never
                             pushed, and moves the minimum distance that keeps
                             it within ``w`` of the formula's own output.

A DEVIATION FROM THE TASK'S WORDING, DECLARED HERE RATHER THAN BURIED: the
brief describes ``k=0`` as "never move / frozen". A literally frozen
``eff_scale`` is a constant, which is not a member of the trade-to-boundary
family at all (it is "no policy"), and it would make ``k`` index two
different objects at its two ends. ``k`` therefore indexes the sub-band
width as a fraction of ``w``, monotone from "no band at all" (k=1, v4) to
"the full band" (k=0), which is the same family R-64's conservative branch
swept with the opposite orientation. Both endpoints of the task's
description are honoured in the sense that matters: k=1 IS v4 bit-for-bit,
and k=0 IS the maximally-lagged / minimally-trading member.

Note the grid contains three exact aliases by construction, which are used
as an internal consistency check rather than pretended to be extra evidence:
``(k=0.5, w=0.10)`` == ``(k=0.0, w=0.05)`` == sub 0.050, ``(k=0.75, w=0.10)``
== ``(k=0.5, w=0.05)`` == sub 0.025, and ``k=1.0`` at either ``w`` is v4.
Distinct mechanisms in the 10-cell grid: **7** (sub in {0.100, 0.075, 0.050,
0.0375, 0.025, 0.0125, 0.000}).

=====================================================================
WHY THIS COULD BE A NULL (NAMED BEFORE RUNNING, NOT AFTER)
=====================================================================

1. **The outer deadband eats it.** ``desired = frac * eff`` and ``frac`` is
   at most 1, so a sub-band of ``w`` on ``eff`` perturbs ``desired`` by at
   most ``w`` -- exactly the size of v4's own position deadband. The
   mechanism can then only change *which side* of the deadband a bar falls
   on, not whether the strategy trades at all. ``cmd_diagnostic`` measures
   how often ``eff`` actually differs from ``scale``, and by how much.
2. **Lag, not saving.** ``eff`` is a laggard version of ``scale``: in a
   volatility spike it de-levers late. That is a risk change, not a cost
   change, and D0 (risk match) plus D2 (advantage must GROW with the fee)
   are what separate the two readings.
3. **R-64's law transfers.** R-64 found this exact destination policy on the
   whole product failed because a no-trade region never returns to flat.
   Here the region is on ``scale`` only and ``frac`` still multiplies it, so
   ``desired`` reaches exactly 0 whenever the vote does -- the specific
   killer of R-64's conservative arm is structurally absent. If the arm
   still fails, it fails for a different reason and that is worth recording.

=====================================================================
SELECTION RULE, FROZEN BEFORE ANY NUMBER WAS READ
=====================================================================

(Written into this file before the first sweep run; git history and the
report both record it. ``k=1`` is excluded from candidacy -- it IS the
incumbent.)

  S1 (mechanism operates): total FILL COUNT strictly below v4's on BOTH
     inner splits, on spot.
  S2 (risk match, ROUTINE's first standing rule): |c_arm - c_v4| / c_v4
     <= 0.10 on BOTH inner splits and BOTH markets, where c is mean
     notional (``experiments/matched_hold.mean_notional``), AND
     time-in-market within 10% of v4's on the same cells.
  SELECT: among cells passing S1 and S2, the one with the highest
     inner-VALIDATION mean of (arm - v4) total log growth across
     {spot@0.10%, futures_5x}. Ties -> the LARGER sub-band (more literal
     Constantinides/Davis-Norman).
  FALLBACK: if no cell passes S1 and S2, freeze the a-priori theory-literal
     cell (k=0.0, w=0.10, i.e. sub-band = v4's own shipped deadband, zero
     new fitted constants), report the screen failure plainly, and run the
     holdout once anyway so that the round has a measurement rather than an
     argument.

The holdout decision rule is r165_shared's D0-D6 verbatim; nothing here
adds to or loosens it.

=====================================================================
DATA DISCIPLINE
=====================================================================

``sweep``/``diagnostic``/``causality``/``select`` load BTC **truncated at
2022-12-31** (``load_btc_inner``), so no holdout bar can be touched during
iteration. Only ``holdout`` loads the full frame, and it runs exactly once,
after the config is frozen by the rule above. ETH-A (Bitfinex
2016-03 -> 2019-12, built by ``scripts/build_bitfinex_dataset.py``) is
entirely pre-2020 and costs zero holdout consultations.

The class is deliberately NOT ``@register``-ed (ROUTINE.md step 5), so
``tests/test_causality_strict.py`` does not auto-test it; ``cmd_causality``
carries the same truncation + tamper probe in-file.

Usage::

    python experiments/r165_conservative_boundary.py            # everything
    python experiments/r165_conservative_boundary.py causality
    python experiments/r165_conservative_boundary.py diagnostic
    python experiments/r165_conservative_boundary.py sweep
    python experiments/r165_conservative_boundary.py holdout
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiments.r165_shared as R  # noqa: E402
from experiments.matched_hold import mean_notional  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_funding, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns,
    annualized_sharpe,
    max_drawdown_from_returns,
    paired_bootstrap,
    total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import prefix_bars, run_period  # noqa: E402

DATA = ROOT / "data"
OUT_DIR = ROOT / "experiments" / "reports"

# ------------------------------------------------------------------ splits
INNER_TRAIN = (None, R.INNER_TRAIN_END)
INNER_VAL = (R.INNER_VAL_START, R.INNER_VAL_END)
HOLDOUT = (R.OOS_START, None)
SPLITS = (("inner-train", INNER_TRAIN), ("inner-val", INNER_VAL))

BOOT_KW = dict(mean_block=30.0, n_boot=2_000, seed=7)
FEE_BASE = 0.0010          # the comparison table's tier
FEE_STRESS = 0.0040        # scripts/fee_study.py's BITSTAMP_TAKER, D2's tier
RISK_MATCH_TOL = 0.10      # D0
SHARPE_NOISE_FLOOR = 0.2   # R-20

#: The mechanism's index. k=1.0 IS kelly_regime_v4 (regression check).
K_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
#: Band-width choices: half of v4's shipped deadband, and all of it.
W_GRID = (0.05, 0.10)

#: Counts every backtest actually run, so the trials count is observed.
_CONFIGS = [0]
_HOLDOUT_READS = [0]


# ============================================================== the strategy


class BoundaryScale(KellyRegimeV4):
    """v4 with a no-trade region, boundary destination, on ``scale`` alone.

    ``KellyRegimeV3.prepare``'s body copied faithfully; the only edit is the
    three lines that let an internal ``eff_scale`` track ``scale`` instead of
    equalling it. ``frac``, the hysteresis state machine and the final
    ``deadband`` position update are untouched.

    ``sub = (1 - k) * band_w``; ``k=1`` gives ``sub=0`` and therefore
    ``eff == scale`` at every bar, i.e. kelly_regime_v4 bit-for-bit.
    """

    name = "r165_conservative_boundary_scale"
    warmup = 80 * BARS_PER_DAY + 10  # v4's

    def __init__(self, k: float = 1.0, band_w: float = 0.10, **kwargs) -> None:
        super().__init__(**kwargs)
        if not np.isfinite(k) or k < 0.0 or k > 1.0:
            raise ValueError(f"k must be in [0, 1], got {k!r}")
        if not np.isfinite(band_w) or band_w < 0.0:
            raise ValueError(f"band_w must be >= 0, got {band_w!r}")
        self.k = float(k)
        self.band_w = float(band_w)

    @property
    def sub(self) -> float:
        """The sub-band half-width the boundary rule actually uses."""
        return (1.0 - self.k) * self.band_w

    # ---- v3's prepare body, with `eff_scale` inserted ---------------------

    def _targets(self, df: pd.DataFrame, record: list | None = None) -> np.ndarray:
        close = df["close"]
        r = np.log(close).diff()

        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
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
        eff = 0.0  # the effective scale: the internal state this branch adds
        sub = self.sub
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
            # ---- THE ONLY CHANGE vs KellyRegimeV3.prepare ------------------
            gap = scale - eff
            if gap > sub:
                if record is not None:
                    record.append((i, gap, eff, scale, +1))
                eff = scale - sub
            elif gap < -sub:
                if record is not None:
                    record.append((i, gap, eff, scale, -1))
                eff = scale + sub
            desired = frac[i] * eff          # v4: frac[i] * scale
            # ----------------------------------------------------------------
            if abs(desired - pos) > self.deadband:   # v4's own update, intact
                pos = desired
            target[i] = pos

        return target

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["target"] = self._targets(df)
        return df


def arm(k: float, w: float) -> BoundaryScale:
    """A fresh arm; every other constant is kelly_regime_v4's shipped one."""
    return BoundaryScale(k=k, band_w=w)


# ================================================================== plumbing


def spot(fee: float = FEE_BASE) -> MarketSpec:
    return MarketSpec.spot(fee_rate=fee)


def futures(fee: float = 0.0005, leverage: float = 5.0) -> MarketSpec:
    return MarketSpec.futures(leverage=leverage, fee_rate=fee)


def load_btc_full() -> pd.DataFrame:
    return load_ohlcv_csv(DATA / "btcusd_spot_5m.csv.gz")


def load_btc_inner() -> pd.DataFrame:
    """BTC truncated at 2022-12-31: the holdout cannot be read by accident."""
    return load_btc_full().loc[:R.INNER_VAL_END]


def load_eth_a() -> pd.DataFrame:
    """ETH-A: Bitfinex 2016-03 -> 2019-12 (scripts/build_bitfinex_dataset.py).

    Entirely pre-2020, so the D3 falsification costs zero holdout reads.
    """
    return load_ohlcv_csv(DATA / "ethusd_bitfinex_5m.csv.gz")


def v4():
    """A fresh, unmodified incumbent. Never mutate its defaults."""
    return get_strategy("kelly_regime_v4")


def measure(strategy, df: pd.DataFrame, window, market: MarketSpec,
            balance: float = 1_000.0):
    """One backtest over a window, warmed on the bars before it. Counted."""
    start, end = window
    _CONFIGS[0] += 1
    if start is not None and str(start) >= R.OOS_START:
        _HOLDOUT_READS[0] += 1
    res = run_period(strategy, df, start, end, market=market,
                     start_balance=balance)
    return res, compute_metrics(res)


def compare(a_strategy, df: pd.DataFrame, window, market: MarketSpec,
            label: str = "", baseline=None) -> dict:
    """Arm vs baseline (default kelly_regime_v4) on one window/market.

    Everything D0-D5 needs, in one flat dict: paired stationary block
    bootstrap on total log growth AND on Sharpe (r165_shared D1 asks for
    both), mean notional and time-in-market for the risk gate, fees, round
    trips (``num_trades``) AND fill count (``len(fills)``) -- the two
    turnover units ROUTINE's standing rule insists on carrying separately.
    """
    base = v4() if baseline is None else baseline
    arm_res, arm_m = measure(a_strategy, df, window, market)
    b_res, b_m = measure(base, df, window, market)

    a = daily_returns(arm_res.equity).to_numpy(dtype=float)
    b = daily_returns(b_res.equity).to_numpy(dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    growth = paired_bootstrap(a, b, total_log_return, **BOOT_KW)
    sharpe = paired_bootstrap(a, b, annualized_sharpe, **BOOT_KW)
    dd = paired_bootstrap(a, b, max_drawdown_from_returns, **BOOT_KW)

    c_arm, c_base = mean_notional(arm_res), mean_notional(b_res)
    mismatch = abs(c_arm - c_base) / c_base if c_base > 0 else float("nan")
    tim_a, tim_b = arm_m.time_in_market_pct, b_m.time_in_market_pct
    tim_mismatch = abs(tim_a - tim_b) / tim_b if tim_b > 0 else float("nan")
    vol_a = float(np.std(a, ddof=1) * np.sqrt(365.25)) if n > 2 else float("nan")
    vol_b = float(np.std(b, ddof=1) * np.sqrt(365.25)) if n > 2 else float("nan")
    vol_mismatch = abs(vol_a - vol_b) / vol_b if vol_b > 0 else float("nan")

    return dict(
        label=label, market=market.name, fee=market.fee_rate, n_days=n,
        arm_final=arm_m.final_balance, base_final=b_m.final_balance,
        arm_sharpe=arm_m.sharpe, base_sharpe=b_m.sharpe,
        arm_dd=arm_m.max_drawdown_pct, base_dd=b_m.max_drawdown_pct,
        arm_trades=arm_m.num_trades, base_trades=b_m.num_trades,
        arm_fills=len(arm_res.fills), base_fills=len(b_res.fills),
        arm_fees=arm_m.fees_paid, base_fees=b_m.fees_paid,
        arm_tim=tim_a, base_tim=tim_b, tim_mismatch=tim_mismatch,
        arm_vol=vol_a, base_vol=vol_b, vol_mismatch=vol_mismatch,
        d_logret=growth.diff.point, d_logret_lo=growth.diff.lo,
        d_logret_hi=growth.diff.hi,
        d_logret_excl0=(growth.diff.lo > 0.0 or growth.diff.hi < 0.0),
        d_sharpe=sharpe.diff.point, d_sharpe_lo=sharpe.diff.lo,
        d_sharpe_hi=sharpe.diff.hi,
        d_sharpe_excl0=(sharpe.diff.lo > 0.0 or sharpe.diff.hi < 0.0),
        d_dd=dd.diff.point, d_dd_lo=dd.diff.lo, d_dd_hi=dd.diff.hi,
        c_arm=c_arm, c_base=c_base, risk_mismatch=mismatch,
        d0_void=bool((np.isfinite(mismatch) and mismatch > RISK_MATCH_TOL)
                     or (np.isfinite(tim_mismatch) and tim_mismatch > RISK_MATCH_TOL)),
    )


def _print_header(text: str) -> None:
    print("\n" + "=" * 118)
    print(text)
    print("=" * 118)


CELL_FMT = ("{k:>5} w={w:<5} sub={sub:<6} {split:<11} {mkt:<11} fee={fee:.4f} "
            "arm=${af:>11,.0f} v4=${vf:>11,.0f} Sh {ash:>5.2f}/{vsh:<5.2f} "
            "DD {add:>5.1f}/{vdd:<5.1f} fills {afl:>5}/{vfl:<5} "
            "fees ${afe:>8,.0f}/${vfe:<8,.0f} c {ac:.3f}/{vc:.3f} "
            "dlog {dl:+7.4f} [{lo:+.3f},{hi:+.3f}]{void}")


def show(row: dict) -> None:
    print(CELL_FMT.format(
        k=f"{row['k']:.2f}", w=f"{row['w']:.2f}", sub=f"{row['sub']:.4f}",
        split=row["split"], mkt=row["market"], fee=row["fee"],
        af=row["arm_final"], vf=row["base_final"],
        ash=row["arm_sharpe"], vsh=row["base_sharpe"],
        add=row["arm_dd"], vdd=row["base_dd"],
        afl=row["arm_fills"], vfl=row["base_fills"],
        afe=row["arm_fees"], vfe=row["base_fees"],
        ac=row["c_arm"], vc=row["c_base"],
        dl=row["d_logret"], lo=row["d_logret_lo"], hi=row["d_logret_hi"],
        void="  D0-VOID" if row["d0_void"] else ""))


def _row(res: dict, k: float, w: float, split: str) -> dict:
    out = dict(res)
    out.update(k=k, w=w, sub=(1.0 - k) * w, split=split)
    return out


def window_frame(df: pd.DataFrame, window, warmup: int) -> tuple[pd.DataFrame, int]:
    """The exact frame ``run_period`` hands to ``prepare``, plus the index at
    which trading starts (same three lines as ``window.run_period``)."""
    start, end = window
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    prefix = prefix_bars(df, lo, warmup)
    return df.iloc[lo - prefix: hi], prefix


# =============================================================== subcommands


def cmd_causality(df: pd.DataFrame) -> bool:
    """Truncation + tamper probe, plus the k=1 == v4 regression check.

    A. ``BoundaryScale(k=1, w=any)``'s target path is bit-identical to
       ``KellyRegimeV4``'s -- the check that the copied v3 loop did not drift.
    B. Truncation: ``prepare`` on a frame cut at ``m`` reproduces the
       full-frame targets over ``[:m]`` exactly. A full-series statistic (a
       scaler, quantile, mean or std over the whole frame) breaks this.
    C. Tamper: corrupt every bar after ``m`` (x3 and /3); no target before
       ``m`` may change, and the two tampers must agree.
    """
    _print_header("(0) CAUSALITY -- truncation / tamper probe + k=1 identity "
                  "(class is unregistered)")
    frame = df.iloc[-400_000:].copy()
    ok = True

    b = KellyRegimeV4().prepare(frame.copy())["target"].to_numpy(dtype=float)
    for w in W_GRID:
        a = BoundaryScale(k=1.0, band_w=w)._targets(frame)
        ident = bool(np.array_equal(a, b))
        ok &= ident
        print(f"  A. k=1.00 w={w:.2f}: target path == kelly_regime_v4 "
              f"bit-for-bit : {'PASS' if ident else 'FAIL'}")

    s = BoundaryScale(k=0.0, band_w=0.10)
    full = s._targets(frame)
    for m in (len(frame) - 5_000, len(frame) - 50_000):
        trunc = s._targets(frame.iloc[:m].copy())
        same = bool(np.array_equal(full[:m], trunc))
        ok &= same
        print(f"  B. truncate at m={m:>7}: targets[:m] identical           : "
              f"{'PASS' if same else 'FAIL'}")

    m = len(frame) - 5_000
    up, down = frame.copy(), frame.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[m:, up.columns.get_loc(col)] *= 3.0
        down.iloc[m:, down.columns.get_loc(col)] /= 3.0
    up.iloc[m:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[m:, down.columns.get_loc("volume")] /= 7.0
    tu, td = s._targets(up), s._targets(down)
    c1 = bool(np.array_equal(tu[:m], full[:m]))
    c2 = bool(np.array_equal(td[:m], full[:m]))
    c3 = bool(np.array_equal(tu[:m], td[:m]))
    ok &= c1 and c2 and c3
    print(f"  C. tamper x3 after m: targets[:m] unchanged               : "
          f"{'PASS' if c1 else 'FAIL'}")
    print(f"  C. tamper /3 after m: targets[:m] unchanged               : "
          f"{'PASS' if c2 else 'FAIL'}")
    print(f"  C. opposite tampers agree on targets[:m]                  : "
          f"{'PASS' if c3 else 'FAIL'}")

    # D. tests/test_causality_strict.py's own check, run by hand because the
    # class is unregistered: compare the ORDERS queued at bars where the arm
    # actually trades (non-vacuous by construction) under opposite tampers of
    # every later bar. This is the check that caught R-21's `i + 1` peek.
    from tradebot.broker import PaperBroker  # local: test-only plumbing
    from tradebot.strategy import Context
    small = df.iloc[-40_000:].copy()
    cut = len(small) - 5_000
    probe = BoundaryScale(k=0.25, band_w=0.05)
    path = probe._targets(small)
    bars = [i for i in range(probe.warmup, cut)
            if abs(path[i] - path[i - 1]) > 1e-12][-8:]
    tam_up, tam_dn = small.copy(), small.copy()
    for col in ("open", "high", "low", "close"):
        tam_up.iloc[cut:, tam_up.columns.get_loc(col)] *= 3.0
        tam_dn.iloc[cut:, tam_dn.columns.get_loc(col)] /= 3.0

    def decisions(frame):
        s2 = BoundaryScale(k=0.25, band_w=0.05)
        prepared = s2.prepare(frame.copy())
        broker = PaperBroker(market=spot(), start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s2.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    du, dd = decisions(tam_up), decisions(tam_dn)
    n_orders = sum(len(x) for x in du)
    d_ok = bool(du == dd) and n_orders > 0
    ok &= d_ok
    print(f"  D. orders at {len(bars)} trading bars identical under opposite "
          f"future tampers ({n_orders} orders, non-vacuous): "
          f"{'PASS' if d_ok else 'FAIL'}")

    print(f"\n  CAUSALITY: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def cmd_decay(df: pd.DataFrame) -> dict:
    """r165_shared's own measurement, re-run here: how persistent is `scale`'s
    input? Free (no backtest, 0 configs)."""
    _print_header("(0b) `scale`'s feeding series: causal half-life on inner-train "
                  "(r165_shared helper, re-run as a cross-check)")
    v4s = KellyRegimeV4()
    vol = R.realized_vol_series(df["close"], v4s.vol_span)
    out = R.causal_autocorr_halflife_days(vol)
    gap = R.order_of_magnitude_gap(out["halflife_days"])
    print(f"  vol span              : {v4s.vol_span} bars "
          f"({v4s.vol_span / BARS_PER_DAY:.0f} days)")
    print(f"  causal ACF(1 day)     : {out['acf_lag1']:.4f}  "
          f"(n={out['n_days']} daily obs, fit end {R.INNER_TRAIN_END})")
    print(f"  implied half-life     : {out['halflife_days']:.1f} days")
    print(f"  v4 anchor half-lives  : {R.V4_ANCHOR_HALFLIVES_DAYS} days (R-64)")
    print(f"  ratio / same band?    : {gap['ratio']:.1f}x  "
          f"in_same_band={gap['in_same_band']}")
    print(f"  r165_shared falsification test fires (1-15d band): "
          f"{gap['falsification_test_fires']}")
    return dict(**out, **{f"gap_{k}": v for k, v in gap.items()})


def cmd_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """Does the mechanism even operate? How far does `eff` sit from `scale`?

    Free: no backtest, 0 configs. For each (k, w) it recomputes the target
    path and reports how many bars the boundary rule fired on, the mean
    |scale - eff| gap actually carried, and the resulting target-path
    turnover (sum |delta pos|) and step count -- the honest turnover unit,
    since ``num_trades`` counts round trips.
    """
    _print_header("(1) MECHANISM DIAGNOSTIC -- does `eff_scale` ever differ "
                  "from `scale`? (free, 0 configs)")
    print(f"  {'split':<12} {'k':>5} {'w':>5} {'sub':>7} {'boundary hits':>14} "
          f"{'mean|gap|':>10} {'steps':>7} {'turnover':>9} {'mean step':>10}")
    rows = []
    for split, window in SPLITS:
        for w in W_GRID:
            for k in K_GRID:
                frame, prefix = window_frame(df, window, BoundaryScale.warmup)
                s = BoundaryScale(k=k, band_w=w)
                rec: list = []
                tgt = s._targets(frame, record=rec)
                lo = max(prefix, BoundaryScale.warmup)
                hits = sum(1 for e in rec if e[0] >= lo)
                gaps = np.array([abs(e[1]) for e in rec if e[0] >= lo])
                d = np.abs(np.diff(tgt))[max(lo - 1, 0):]
                steps = d[d > 1e-9]
                row = dict(split=split, k=k, w=w, sub=s.sub, hits=hits,
                           mean_gap=float(gaps.mean()) if len(gaps) else 0.0,
                           steps=int(len(steps)),
                           turnover=float(steps.sum()),
                           mean_step=float(steps.mean()) if len(steps) else 0.0)
                rows.append(row)
                print(f"  {split:<12} {k:>5.2f} {w:>5.2f} {s.sub:>7.4f} "
                      f"{hits:>14,} {row['mean_gap']:>10.4f} "
                      f"{row['steps']:>7} {row['turnover']:>9.2f} "
                      f"{row['mean_step']:>10.4f}")
        print()
    frame_out = pd.DataFrame(rows)
    frame_out.to_csv(OUT_DIR / "r165_conservative_diagnostic.csv", index=False)
    return frame_out


def cmd_sweep(df: pd.DataFrame) -> list[dict]:
    """The 10-cell grid on both inner splits, spot @0.10% and futures 5x."""
    _print_header("(2) GRID SWEEP -- inner-train + inner-validation, "
                  "arm vs kelly_regime_v4 (holdout NOT read)")
    rows: list[dict] = []
    for market in (spot(FEE_BASE), futures()):
        for split, window in SPLITS:
            for w in W_GRID:
                for k in K_GRID:
                    row = _row(compare(arm(k, w), df, window, market,
                                       label=f"boundary k={k:.2f} w={w:.2f}"),
                               k, w, split)
                    show(row)
                    rows.append(row)
            print("-" * 118)
    pd.DataFrame(rows).to_csv(OUT_DIR / "r165_conservative_sweep.csv", index=False)
    return rows


# ================================================ the frozen selection rule


def select_frozen(rows: list[dict]) -> dict:
    """Apply the selection rule frozen in this file's docstring. Verbatim."""
    _print_header("(3) SELECTION -- the rule frozen in this file's docstring, "
                  "applied mechanically")
    cands = sorted({(r["k"], r["w"]) for r in rows if r["k"] < 1.0})
    verdicts = []
    for k, w in cands:
        cells = [r for r in rows if r["k"] == k and r["w"] == w]
        spot_cells = [r for r in cells if r["market"] == "spot"]
        s1 = all(r["arm_fills"] < r["base_fills"] for r in spot_cells) and bool(spot_cells)
        s2 = all(not r["d0_void"] for r in cells)
        iv = [r for r in cells if r["split"] == "inner-val"]
        score = float(np.mean([r["d_logret"] for r in iv])) if iv else float("nan")
        verdicts.append(dict(k=k, w=w, sub=(1 - k) * w, s1=s1, s2=s2,
                             score=score,
                             fills=[(r["market"], r["split"], r["arm_fills"],
                                     r["base_fills"]) for r in spot_cells]))
    print(f"  {'k':>5} {'w':>5} {'sub':>7} {'S1 fills fall':>14} "
          f"{'S2 risk match':>14} {'inner-val mean dlog':>20}")
    for v in verdicts:
        print(f"  {v['k']:>5.2f} {v['w']:>5.2f} {v['sub']:>7.4f} "
              f"{('PASS' if v['s1'] else 'fail'):>14} "
              f"{('PASS' if v['s2'] else 'fail'):>14} {v['score']:>+20.4f}")

    passing = [v for v in verdicts if v["s1"] and v["s2"]]
    if passing:
        best = max(passing, key=lambda v: (v["score"], v["sub"]))
        why = ("highest inner-validation mean log-growth difference among the "
               "cells passing S1 (fills fall) and S2 (risk match)")
    else:
        best = next(v for v in verdicts if v["k"] == 0.0 and v["w"] == 0.10)
        why = ("FALLBACK: no cell passed S1 and S2, so the a-priori "
               "theory-literal cell (k=0, w=v4's own deadband) is frozen and "
               "the screen failure is reported")
    print(f"\n  FROZEN: k={best['k']:.2f}, w={best['w']:.2f} "
          f"(sub-band {best['sub']:.4f})")
    print(f"  WHY   : {why}")
    return dict(best, why=why, n_candidates=len(cands))


# ==================================================== the holdout, run once


def cmd_holdout(frozen: dict) -> dict:
    """Run the frozen config on the holdout ONCE, then apply D0-D6 verbatim."""
    k, w = frozen["k"], frozen["w"]
    df = load_btc_full()
    _print_header(f"(4) HOLDOUT -- frozen config k={k:.2f} w={w:.2f} "
                  f"(sub={frozen['sub']:.4f}), {R.OOS_START} -> "
                  f"{df.index[-1]:%Y-%m-%d}")
    rows: list[dict] = []

    # D1: both markets, base fee tier, arm vs v4.
    for market in (spot(FEE_BASE), futures()):
        row = _row(compare(arm(k, w), df, HOLDOUT, market,
                           label=f"FROZEN k={k:.2f} w={w:.2f}"), k, w, "holdout")
        show(row)
        rows.append(row)

    # D2: the cost-mechanism test -- the advantage must GROW at 0.40%.
    _print_header("(4b) D2 -- the same holdout comparison at the 0.40% taker "
                  "tier (scripts/fee_study.py's BITSTAMP_TAKER)")
    for market in (spot(FEE_STRESS),):
        row = _row(compare(arm(k, w), df, HOLDOUT, market,
                           label=f"FROZEN k={k:.2f} w={w:.2f} @0.40%"),
                   k, w, "holdout-stress")
        show(row)
        rows.append(row)

    # D5: the immediate neighbours of the frozen cell, on the holdout.
    _print_header("(4c) D5 -- immediate neighbours of the frozen cell "
                  "(one k step either way), holdout spot @0.10%")
    ks = sorted(K_GRID)
    i = ks.index(k)
    for kn in [x for x in (ks[max(i - 1, 0)], ks[min(i + 1, len(ks) - 1)]) if x != k]:
        row = _row(compare(arm(kn, w), df, HOLDOUT, spot(FEE_BASE),
                           label=f"neighbour k={kn:.2f} w={w:.2f}"),
                   kn, w, "holdout-neighbour")
        show(row)
        rows.append(row)

    # The standing bar: does it still beat buy_and_hold out-of-sample?
    _print_header("(4d) STANDING BAR -- frozen arm and v4 vs buy_and_hold, "
                  "holdout spot @0.10%")
    bh = compare(arm(k, w), df, HOLDOUT, spot(FEE_BASE),
                 label="FROZEN vs buy_and_hold", baseline=get_strategy("buy_and_hold"))
    bh = _row(bh, k, w, "holdout-vs-bh")
    show(bh)
    rows.append(bh)

    pd.DataFrame(rows).to_csv(OUT_DIR / "r165_conservative_holdout.csv", index=False)
    return dict(rows=rows)


def cmd_eth(frozen: dict) -> list[dict]:
    """D3 -- ETH-A falsification on Bitfinex 2016-03 -> 2019-12 (+0 holdout)."""
    k, w = frozen["k"], frozen["w"]
    eth = load_eth_a()
    _print_header(f"(5) D3 FALSIFICATION -- ETH-A Bitfinex "
                  f"{eth.index[0]:%Y-%m} -> {eth.index[-1]:%Y-%m}, "
                  f"{len(eth):,} bars, spot @0.10% and @0.40%")
    rows = []
    for market in (spot(FEE_BASE), spot(FEE_STRESS)):
        row = _row(compare(arm(k, w), eth, (None, None), market,
                           label=f"FROZEN k={k:.2f} w={w:.2f}"), k, w, "eth-a")
        show(row)
        rows.append(row)
    return rows


def cmd_funding(frozen: dict) -> list[dict]:
    """D6 -- futures with real funding charged (run only if D1-D5 all pass)."""
    k, w = frozen["k"], frozen["w"]
    df = load_btc_full()
    real = load_funding(DATA)
    _print_header("(6) D6 FUNDING -- futures 5x with real Binance funding charged")
    if real is None:
        print("  no funding file; skipped")
        return []
    market = futures()
    out = []
    for label, s in ((f"FROZEN k={k:.2f} w={w:.2f}", arm(k, w)),
                     ("kelly_regime_v4", v4())):
        lo = int(df.index.searchsorted(R.OOS_START))
        pre = min(lo, s.warmup)
        _CONFIGS[0] += 1
        _HOLDOUT_READS[0] += 1
        raw = run_backtest(s, df.iloc[lo - pre:], market, 1_000.0, trade_start=pre)
        from dataclasses import replace as _replace
        res = raw if pre == 0 else _replace(raw, equity=raw.equity.iloc[pre:],
                                            df=raw.df.iloc[pre:])
        m = compute_metrics(res)
        print(f"  {label:<28} final=${m.final_balance:>12,.0f} "
              f"sharpe={m.sharpe:>5.2f} DD={m.max_drawdown_pct:>5.1f}% "
              f"funding=${raw.funding_paid:>10,.0f}")
        out.append(dict(label=label, final=m.final_balance, sharpe=m.sharpe,
                        dd=m.max_drawdown_pct, funding=raw.funding_paid))
    return out


# ====================================================== the D0-D6 verdict


def verdict(frozen: dict, hold_rows: list[dict], eth_rows: list[dict]) -> dict:
    """r165_shared's D0-D6, applied mechanically to the frozen cell."""
    _print_header("(7) DECISION RULE D0-D6 (r165_shared, frozen before any "
                  "number was read)")
    d1_cells = [r for r in hold_rows if r["split"] == "holdout"]
    stress = [r for r in hold_rows if r["split"] == "holdout-stress"]
    nbrs = [r for r in hold_rows if r["split"] == "holdout-neighbour"]

    # D0
    d0_void = [r for r in d1_cells if r["d0_void"]]
    d0 = not d0_void
    print(f"  D0 risk-match gate  : {'PASS' if d0 else 'VOID'}")
    for r in d1_cells:
        print(f"       {r['market']:<11} mean notional {r['c_arm']:.4f} vs "
              f"{r['c_base']:.4f} ({r['risk_mismatch']:+.2%}), "
              f"time-in-market {r['arm_tim']:.1f}% vs {r['base_tim']:.1f}% "
              f"({r['tim_mismatch']:+.2%}), realized vol "
              f"{r['arm_vol']:.3f} vs {r['base_vol']:.3f} "
              f"({r['vol_mismatch']:+.2%})")

    # D1: favourable exclusion of zero on >=1 of {growth, Sharpe} on BOTH
    # markets, with the other metric not significantly negative on either.
    def fav(r, m):
        return r[f"d_{m}_excl0"] and r[f"d_{m}"] > 0
    def adverse(r, m):
        return r[f"d_{m}_excl0"] and r[f"d_{m}"] < 0
    d1_ok = (bool(d1_cells)
             and all(fav(r, "logret") or fav(r, "sharpe") for r in d1_cells)
             and not any(adverse(r, "logret") or adverse(r, "sharpe")
                         for r in d1_cells))
    print(f"  D1 holdout vs v4    : {'PASS' if d1_ok else 'FAIL'}")
    for r in d1_cells:
        print(f"       {r['market']:<11} dlog {r['d_logret']:+.4f} "
              f"[{r['d_logret_lo']:+.4f}, {r['d_logret_hi']:+.4f}]   "
              f"dSharpe {r['d_sharpe']:+.4f} "
              f"[{r['d_sharpe_lo']:+.4f}, {r['d_sharpe_hi']:+.4f}]")

    # D2: the advantage must GROW when the fee quadruples.
    base_spot = next((r for r in d1_cells if r["market"] == "spot"), None)
    d2 = (bool(stress) and base_spot is not None
          and stress[0]["d_logret"] > base_spot["d_logret"])
    if base_spot is not None and stress:
        print(f"  D2 cost mechanism   : {'PASS' if d2 else 'FAIL'}  "
              f"dlog@0.10% {base_spot['d_logret']:+.4f} -> "
              f"dlog@0.40% {stress[0]['d_logret']:+.4f}")

    # D3: ETH-A sign must not reverse.
    eth_base = next((r for r in eth_rows if abs(r["fee"] - FEE_BASE) < 1e-12), None)
    d3 = eth_base is not None and eth_base["d_logret"] >= 0.0
    if eth_base is not None:
        print(f"  D3 ETH-A falsif.    : {'PASS' if d3 else 'FAIL'}  "
              f"dlog {eth_base['d_logret']:+.4f} "
              f"[{eth_base['d_logret_lo']:+.4f}, {eth_base['d_logret_hi']:+.4f}]")

    # D4: total fill count must fall.
    d4 = all(r["arm_fills"] < r["base_fills"] for r in d1_cells)
    print(f"  D4 turnover falls   : {'PASS' if d4 else 'FAIL'}")
    for r in d1_cells:
        print(f"       {r['market']:<11} fills {r['arm_fills']} vs "
              f"{r['base_fills']}  |  round trips {r['arm_trades']} vs "
              f"{r['base_trades']}  |  fees ${r['arm_fees']:,.0f} vs "
              f"${r['base_fees']:,.0f}")

    # D5: neighbour Sharpe within the +/-0.2 noise floor of the frozen cell.
    spot_frozen = base_spot
    d5 = True
    if spot_frozen is not None and nbrs:
        for r in nbrs:
            gapv = abs(r["arm_sharpe"] - spot_frozen["arm_sharpe"])
            print(f"  D5 neighbour k={r['k']:.2f}   : arm Sharpe "
                  f"{r['arm_sharpe']:.3f} vs frozen "
                  f"{spot_frozen['arm_sharpe']:.3f}  (|gap| {gapv:.3f} vs the "
                  f"{SHARPE_NOISE_FLOOR} floor) "
                  f"{'plateau' if gapv <= SHARPE_NOISE_FLOOR else 'CLIFF'}")
            d5 &= gapv <= SHARPE_NOISE_FLOOR
    print(f"  D5 plateau not peak : {'PASS' if d5 else 'FAIL (cliff)'}")

    if not d0:
        v = "REJECT (NEGATIVE): D0 void -- the comparison is an exposure statement"
    elif not (d1_ok and d2 and d3 and d4):
        fails = [n for n, ok in (("D1", d1_ok), ("D2", d2), ("D3", d3),
                                 ("D4", d4)) if not ok]
        v = f"REJECT (NEGATIVE): {', '.join(fails)} not satisfied"
    elif not d5:
        v = "PARTIAL: D1-D4 hold but D5 shows a cliff, not a plateau"
    else:
        v = "PROMOTE"
    print(f"\n  VERDICT: {v}")
    return dict(d0=d0, d1=d1_ok, d2=d2, d3=d3, d4=d4, d5=d5, verdict=v)


# ====================================================================== main


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "all"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("R-165 CONSERVATIVE: trade-to-the-boundary on `scale` ALONE "
          "(Constantinides 1986 / Davis & Norman 1990)")
    print("=" * 118)
    print("arm        : BoundaryScale(k, band_w); sub-band = (1-k)*band_w")
    print(f"k grid     : {K_GRID}   (k=1.0 IS kelly_regime_v4, bit-for-bit)")
    print(f"w grid     : {W_GRID}   (half of v4's deadband, and all of it)")
    print(f"splits     : inner-train (-> {R.INNER_TRAIN_END}), inner-val "
          f"({R.INNER_VAL_START} -> {R.INNER_VAL_END}), holdout "
          f"({R.OOS_START} ->)")

    df = load_btc_inner()
    print(f"BTC (inner): {len(df):,} bars, {df.index[0]:%Y-%m-%d} -> "
          f"{df.index[-1]:%Y-%m-%d}   [truncated at {R.INNER_VAL_END} on load]")

    causal = None
    rows: list[dict] = []
    frozen = None

    if cmd in ("all", "causality"):
        causal = cmd_causality(df)
        if not causal:
            raise SystemExit("causality probe FAILED -- stopping")
    if cmd in ("all", "decay"):
        cmd_decay(df)
    if cmd in ("all", "diagnostic"):
        cmd_diagnostic(df)
    if cmd in ("all", "sweep", "select", "holdout"):
        if cmd == "holdout" and len(argv) >= 3:
            # `holdout K W`: re-use the config the (deterministic) sweep
            # already froze, instead of paying for the sweep twice. The
            # numbers are identical either way; this only keeps the
            # configuration count honest rather than double-counting.
            k_arg, w_arg = float(argv[1]), float(argv[2])
            frozen = dict(k=k_arg, w=w_arg, sub=(1.0 - k_arg) * w_arg,
                          why="frozen by the earlier `sweep` run's selection "
                              "rule; sweep not repeated (deterministic)",
                          s1=True, s2=True, score=float("nan"),
                          n_candidates=len(K_GRID[:-1]) * len(W_GRID))
            print(f"\n  FROZEN (from the recorded sweep): k={k_arg:.2f} "
                  f"w={w_arg:.2f} (sub-band {frozen['sub']:.4f})")
        else:
            rows = cmd_sweep(df)
            frozen = select_frozen(rows)
    if cmd in ("all", "holdout") and frozen is not None:
        print(f"\n  configs evaluated before the holdout was touched: "
              f"{_CONFIGS[0]}")
        hold = cmd_holdout(frozen)
        eth = cmd_eth(frozen)
        vd = verdict(frozen, hold["rows"], eth)
        if all((vd["d0"], vd["d1"], vd["d2"], vd["d3"], vd["d4"], vd["d5"])):
            cmd_funding(frozen)
        else:
            print("\n  D6 funding: not run -- r165_shared runs it only if "
                  "D1-D5 all pass.")

    print()
    print("=" * 118)
    if causal is not None:
        print(f"Causality probe            : {'PASS' if causal else 'FAIL'}")
    print(f"Configurations evaluated   : {_CONFIGS[0]}  (backtests run by this "
          f"branch)")
    print(f"Holdout reads (this branch): {_HOLDOUT_READS[0]}")
    print("=" * 118)


if __name__ == "__main__":
    main()
