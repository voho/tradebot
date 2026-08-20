"""R-64 (conservative branch): trade to the **boundary** of the no-trade
region instead of to the target.

Pre-registration (shared, frozen, both branches): ``experiments/r64_shared.py``.
That file is not edited here, and neither is anything else outside this file.

=====================================================================
THE ONE-LINE MECHANISM
=====================================================================

``kelly_regime_v4``'s position-update loop -- inherited unchanged from
``KellyRegime.prepare`` via ``KellyRegimeV3.prepare``, untouched since L-04 --
is::

    if abs(desired - pos) > self.deadband:
        pos = desired

That is a no-trade region of half-width ``deadband`` around the *current*
position, with a **trade-to-target** destination: on breaching the band the
position jumps the whole distance.

Under **purely proportional** transaction costs -- which is exactly what this
simulator charges (``MarketSpec`` levies ``fee_rate x |traded notional|``;
there is no impact term, no spread, no queue, see ``src/tradebot/broker.py``)
-- the optimal policy is a no-trade region plus, on exit, a trade only **to
the nearest boundary of that region**. Constantinides (1986, JPE 94(4)),
Davis & Norman (1990, Math. OR 15(4)), Shreve & Soner (1994, Ann. Appl.
Prob. 4(3)), Liu (2004, J. Finance 59(1)). Moving past the boundary to the
target buys nothing and pays for the whole distance. This branch is that
one-line correction::

    if desired - pos > band:   pos = desired - k * band
    elif pos - desired > band: pos = desired + k * band

**Zero new fitted parameters.** ``band`` *is* v4's own shipped
``deadband=0.10``; it is not re-tuned, not swept for selection, and not
re-derived. Everything else -- the 20/40/80 latched anchors, the 1% vote
band, the conditional volatility target with its ``high_in/high_out/
low_in/low_out`` hysteresis, ``target_vol=0.55``, ``max_leverage=2.0``,
``vol_span=8*BARS_PER_DAY``, ``anchor_span_days=180`` -- is copied
byte-for-byte from ``KellyRegimeV3.prepare`` and must not change.

=====================================================================
WHY THERE IS A ``k``, AND WHAT IT IS NOT
=====================================================================

"Trade to the boundary" is ambiguous in a discrete-time, discrete-signal
setting, and the ambiguity is real rather than cosmetic: the continuous-time
results describe a reflecting barrier at the boundary of a *continuously*
monitored region, while this rule fires at most once per 5m bar on a target
that itself jumps discontinuously (the vote is a step function of three
latched anchors). ``k`` indexes the family between the two readings:

    k = 0.0   ->  pos = desired          == v4 exactly, trade to target
    k = 0.5   ->  half-way back          the discrete-time compromise
    k = 1.0   ->  pos = desired -/+ band the literal continuous-time policy

``k`` is therefore **not a fitted parameter to be selected on**; it is the
mechanism's own index, and the shape of the response in ``k`` is the
mechanism's signature. If the effect is real it must be **monotone in k**:
each increment of ``k`` shortens every step by the same fraction of the
band, so it must move fees monotonically and (if fee-saving is what is
happening) must move growth monotonically too. A non-monotone response in
``k`` would mean the numbers are being driven by something other than the
step-length change, and this file says so loudly if that is what it finds.
The frozen proposal is ``k = 1.0`` -- the literal theory -- chosen a priori,
not by peak-picking; the sweep is D5 plateau evidence and diagnostic, not
selection.

=====================================================================
THE THREE THINGS THAT CAN MAKE THIS A NULL, MEASURED NOT ASSUMED
=====================================================================

1. **The deadband is already doing the work** (failure mode 2 of the
   pre-registration). If exits from the band are rare and the *overshoot*
   ``o = |desired - pos|`` at those exits is large relative to ``band``,
   then "to the boundary" and "to the target" differ by ``k*band`` on a
   handful of rebalances and the arm is a rounding error. ``cmd_diagnostic``
   measures exactly this: the exit count per split and the full distribution
   of ``o/band``. The saved turnover per exit is ``k*band``, so the
   fractional turnover saving is ``k*band/o`` -- if the median ``o/band``
   is large, the ceiling on this mechanism is small, and no amount of
   parameter choice can raise it.

2. **The broker's own deadband eats the smaller steps.** ``broker.py``
   defines ``REBALANCE_DEADBAND = 0.05``: same-sign target adjustments
   smaller than 5% of *max* notional (equity x leverage) are silently
   dropped. On spot (leverage 1) that is 5% of equity; on 5x futures it is
   **25% of equity**, which is larger than most steps this arm asks for.
   ``experiments/matched_hold.py`` hit the same trap from the other side and
   documents it in ``ConstantExposureHold.on_bar``. This arm cannot route
   around it (``order_notional`` is v4's own execution path and changing it
   would change more than the destination rule), so ``cmd_futures``
   *measures* it: intended target changes vs fills actually executed. A
   swallowed step means the strategy's internal ``pos`` -- which is
   open-loop, computed in ``prepare`` -- has moved while the account has
   not, so the arm silently degrades into a different policy. This is
   reported plainly rather than worked around; ``broker.py`` is not edited.

3. **Lower turnover, worse tracking: a lag, not a saving.** Both this arm
   and the round's novel arm hold a position closer to the *previous* one
   than v4 does. On a trend rule that is a lag, and the pre-registered D2
   test is what separates the two readings: a genuine cost mechanism's
   advantage must **grow** when the taker fee quadruples from 0.10% to
   0.40%. This file runs the D2 shape on the *inner* split as a rehearsal.
   If the advantage does not grow there, that is a finding and it is
   reported as one.

A fourth, specific to this rule and not in the shared pre-registration
because it is a consequence rather than a risk: at ``k > 0`` the position
**never reaches zero**. When the vote turns bearish ``desired`` goes to 0
and the arm steps to ``k*band`` and then stops, because it is now inside its
own no-trade region. So the arm carries a residual long of up to 10% of
equity through every bear regime that v4 sits out flat. That is not a bug --
it is what a no-trade region *means* -- but it makes the D0 risk-match gate
live, so ``mean_notional`` is reported on every single cell.

=====================================================================
DATA DISCIPLINE
=====================================================================

**No bar dated 2023-01-01 or later is read anywhere in this file.** The BTC
frame is truncated to ``2022-12-31`` immediately on load (``load_btc_inner``
below), so the holdout cannot be touched even by accident; ``run_period``
would have truncated at the window end anyway, and the truncation is
belt-and-braces. Splits are ``INNER_TRAIN`` (... -> 2020-12-31) and
``INNER_VAL`` (2021-01-01 -> 2022-12-31) from ``r64_shared``. The
falsification instrument is **ETH-A** (``load_eth_a()``, Bitfinex
2016-03 -> 2019-12), which is entirely pre-2020 and costs zero holdout
consultations (the R-19/R-28 convention). Holdout consultations added by
this branch: **0**.

The class is deliberately **not** ``@register``-ed (ROUTINE.md step 5: an
experimental, unpromoted variant lives under ``experiments/``), which means
``tests/test_causality_strict.py`` does not auto-test it -- so this file
carries its own truncation/tamper causality probe (``cmd_causality``), run
the way prior rounds ran theirs.

=====================================================================
A NOTE ON THE COPIED LOOP
=====================================================================

``KellyRegimeV3.prepare`` is one monolithic method; the position-update loop
is not factored out, so overriding the destination rule requires copying the
body. The copy in ``_targets`` below is faithful: the vote block, the vol /
slow-vol block, the ``full``/``steady`` arrays and the state machine are
character-for-character v3's, and the *only* edited lines are the three that
replace ``pos = desired``. ``k=0.0`` therefore reproduces v4 bit-for-bit,
and ``cmd_causality`` asserts that identity as a regression check on the
faithfulness of the copy.

Usage::

    python experiments/r64_conservative_trade_to_boundary.py            # all
    python experiments/r64_conservative_trade_to_boundary.py sweep
    python experiments/r64_conservative_trade_to_boundary.py stress
    python experiments/r64_conservative_trade_to_boundary.py futures
    python experiments/r64_conservative_trade_to_boundary.py eth
    python experiments/r64_conservative_trade_to_boundary.py diagnostic
    python experiments/r64_conservative_trade_to_boundary.py causality
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiments.r64_shared as R  # noqa: E402
from tradebot.broker import REBALANCE_DEADBAND, MarketSpec  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import prefix_bars  # noqa: E402

OUT_DIR = ROOT / "experiments" / "reports"

#: The mechanism's own index, swept for D5 / monotonicity evidence. k=0 IS
#: v4 (trade to target); k=1 is the literal Constantinides/Davis-Norman
#: policy. NOT a selection grid -- the frozen proposal is k=1.0 a priori.
K_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

#: The frozen proposal for the holdout.
K_FROZEN = 1.0

SHARPE_NOISE_FLOOR = 0.2  # R-20


# ============================================================== the strategy


class TradeToBoundary(KellyRegimeV4):
    """v4 with one line changed: on leaving the no-trade band, move only to
    the band's nearest **boundary** (``k=1``), not all the way to the target.

    ``KellyRegimeV3.prepare``'s body copied faithfully (see the module
    docstring); the only edit is the destination of the position update::

        v4:   if abs(desired - pos) > band: pos = desired
        here: if desired - pos > band:      pos = desired - k * band
              elif pos - desired > band:    pos = desired + k * band

    ``band`` is v4's own shipped ``deadband=0.10``, re-used, not re-fitted.
    ``k=0.0`` reproduces v4 exactly; ``k=1.0`` is the literal
    proportional-cost optimum. Zero new fitted parameters.
    """

    name = "r64_conservative_trade_to_boundary"
    warmup = 80 * BARS_PER_DAY + 10  # v4's

    def __init__(self, k: float = K_FROZEN, **kwargs) -> None:
        super().__init__(**kwargs)
        if not np.isfinite(k) or k < 0.0 or k > 1.0:
            raise ValueError(f"k must be in [0, 1], got {k!r}")
        self.k = float(k)

    # ---- v3's prepare body, with the destination rule replaced -------------

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
        band = self.deadband
        kb = self.k * band
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
            # ---- THE ONE CHANGE vs KellyRegimeV3.prepare --------------------
            if desired - pos > band:
                if record is not None:
                    record.append((i, desired - pos, pos, desired, +1))
                pos = desired - kb
            elif pos - desired > band:
                if record is not None:
                    record.append((i, pos - desired, pos, desired, -1))
                pos = desired + kb
            # ----------------------------------------------------------------
            target[i] = pos

        return target

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["target"] = self._targets(df)
        return df


def arm(k: float = K_FROZEN) -> TradeToBoundary:
    """A fresh arm at index ``k``. Every other constant is v4's shipped one."""
    return TradeToBoundary(k=k)


# ================================================================== plumbing


def load_btc_inner() -> pd.DataFrame:
    """BTC truncated at 2022-12-31 so no holdout bar can be read at all."""
    df = R.load_btc()
    return df.loc[:"2022-12-31"]


def window_frame(df: pd.DataFrame, window, warmup: int) -> tuple[pd.DataFrame, int]:
    """Exactly the frame ``run_period`` hands to ``prepare``, plus the index
    at which trading starts. Re-derived here (same three lines as
    ``window.run_period``) so the diagnostics see what the backtest saw."""
    start, end = window
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    prefix = prefix_bars(df, lo, warmup)
    return df.iloc[lo - prefix: hi], prefix


def _row(res: dict, k: float, split: str) -> dict:
    out = dict(res)
    out["k"] = k
    out["split"] = split
    out["arm_over_v4_fees"] = (res["arm_fees"] / res["v4_fees"]
                               if res["v4_fees"] > 0 else float("nan"))
    return out


def _print_header(text: str) -> None:
    print("\n" + "=" * 104)
    print(text)
    print("=" * 104)


CELL_FMT = ("{k:>5} {split:<12} {mkt:<11} fee={fee:.4f} "
            "arm=${af:>11,.0f} v4=${vf:>11,.0f} "
            "Sh {ash:>5.2f}/{vsh:<5.2f} DD {add:>5.1f}/{vdd:<5.1f} "
            "tr {atr:>4}/{vtr:<4} fees ${afe:>8,.0f}/${vfe:<8,.0f} "
            "c {ac:.3f}/{vc:.3f} dlog {dl:+7.4f} [{lo:+.3f},{hi:+.3f}]{void}")


def show(row: dict) -> None:
    print(CELL_FMT.format(
        k=f"{row['k']:.2f}", split=row["split"], mkt=row["market"],
        fee=row["fee"], af=row["arm_final"], vf=row["v4_final"],
        ash=row["arm_sharpe"], vsh=row["v4_sharpe"],
        add=row["arm_dd"], vdd=row["v4_dd"],
        atr=row["arm_trades"], vtr=row["v4_trades"],
        afe=row["arm_fees"], vfe=row["v4_fees"],
        ac=row["c_arm"], vc=row["c_v4"],
        dl=row["d_logret"], lo=row["d_logret_lo"], hi=row["d_logret_hi"],
        void="  D0-VOID" if row["d0_void"] else ""))


SPLITS = (("inner-train", R.INNER_TRAIN), ("inner-val", R.INNER_VAL))


# =============================================================== subcommands


def cmd_sweep(df: pd.DataFrame, rows: list) -> list[dict]:
    """(1) The k-sweep on both inner splits, spot @0.10%, arm vs v4."""
    _print_header("(1) k-SWEEP -- spot @0.10%, arm vs kelly_regime_v4, inner splits")
    out = []
    for split, window in SPLITS:
        for k in K_GRID:
            row = _row(R.compare(arm(k), df, window, R.spot(R.FEE_BASE),
                                 label=f"boundary k={k:.2f}"), k, split)
            show(row)
            out.append(row)
            rows.append(row)
        print("-" * 104)
    return out


def cmd_stress(df: pd.DataFrame, rows: list) -> list[dict]:
    """(2) The same at 0.40% -- the D2 rehearsal on the inner split."""
    _print_header("(2) k-SWEEP -- spot @0.40% (FEE_STRESS): the D2 shape, rehearsed inner")
    out = []
    for split, window in SPLITS:
        for k in K_GRID:
            row = _row(R.compare(arm(k), df, window, R.spot(R.FEE_STRESS),
                                 label=f"boundary k={k:.2f}"), k, split)
            show(row)
            out.append(row)
            rows.append(row)
        print("-" * 104)
    return out


def fill_through(strategy, df: pd.DataFrame, window, market: MarketSpec) -> dict:
    """How many of the strategy's intended position changes became trades?

    The strategy's ``pos`` is open-loop: ``prepare`` decides the whole target
    path, ``on_bar`` emits an order whenever the path moves, and the broker
    then drops any same-sign adjustment worth less than ``REBALANCE_DEADBAND``
    (5%) of max notional -- equity on spot, ``equity x leverage`` on futures.
    So on 5x futures a 10%-of-equity step is 2% of max notional and vanishes.
    This counts intended changes (from the target path) against fills actually
    executed (from the backtest), which is the size of the gap.
    """
    frame, prefix = window_frame(df, window, strategy.warmup)
    tgt = strategy._targets(frame) if isinstance(strategy, TradeToBoundary) \
        else strategy.prepare(frame.copy())["target"].to_numpy(dtype=float)
    n = len(tgt)
    # on_bar runs for i >= warmup, orders survive for i >= prefix, and the
    # engine never calls on_bar on the last bar.
    lo = max(prefix, strategy.warmup)
    steps = np.abs(np.diff(tgt))[max(lo - 1, 0): n - 2] if n > 2 else np.array([])
    intended = steps > 1e-9
    res, _ = R.measure(strategy, df, window, market)
    n_fills = len(res.fills)
    lev = market.leverage
    # A step of size s (fraction of EQUITY) is routed as order_target(s/lev);
    # the broker drops it when s < REBALANCE_DEADBAND * lev. Closes to zero
    # and sign flips always execute.
    threshold = REBALANCE_DEADBAND * lev
    small = intended & (steps < threshold)
    return dict(
        intended=int(intended.sum()),
        fills=n_fills,
        below_broker_band=int(small.sum()),
        broker_step_threshold=float(threshold),
        median_step=float(np.median(steps[intended])) if intended.any() else float("nan"),
        fill_ratio=float(n_fills / intended.sum()) if intended.sum() else float("nan"),
    )


def cmd_futures(df: pd.DataFrame, rows: list) -> list[dict]:
    """(3) Futures 5x, and the REBALANCE_DEADBAND trap, measured."""
    _print_header("(3) FUTURES 5x -- and the broker's own REBALANCE_DEADBAND trap")
    print(f"broker.REBALANCE_DEADBAND = {REBALANCE_DEADBAND}; on 5x futures a "
          f"same-sign step below {REBALANCE_DEADBAND * 5:.0%} of EQUITY is "
          f"silently dropped.\n(matched_hold.py documents the same trap from "
          f"the other side -- ConstantExposureHold.on_bar.)\n")
    out = []
    fut = R.futures()
    for split, window in SPLITS:
        for k in (0.5, K_FROZEN):
            row = _row(R.compare(arm(k), df, window, fut,
                                 label=f"boundary k={k:.2f}"), k, split)
            show(row)
            out.append(row)
            rows.append(row)
        print("-" * 104)

    _print_header("(3b) FILL-THROUGH: intended target changes vs fills executed")
    print(f"{'strategy':<28} {'split':<12} {'market':<11} {'intended':>9} "
          f"{'fills':>7} {'ratio':>7} {'<band':>7} {'medstep':>8}")
    ft_rows = []
    for market in (R.spot(R.FEE_BASE), fut):
        for split, window in SPLITS:
            for label, s in (("kelly_regime_v4", R.v4()),
                             (f"boundary k={K_FROZEN:.2f}", arm(K_FROZEN))):
                d = fill_through(s, df, window, market)
                print(f"{label:<28} {split:<12} {market.name:<11} "
                      f"{d['intended']:>9} {d['fills']:>7} {d['fill_ratio']:>7.2f} "
                      f"{d['below_broker_band']:>7} {d['median_step']:>8.3f}")
                ft_rows.append(dict(strategy=label, split=split,
                                    market=market.name, **d))
    pd.DataFrame(ft_rows).to_csv(OUT_DIR / "r64_conservative_fillthrough.csv",
                                 index=False)
    return out


def cmd_eth(rows: list) -> list[dict]:
    """(4) D3 falsification on ETH-A (Bitfinex 2016-03 -> 2019-12, +0 holdout)."""
    _print_header("(4) D3 FALSIFICATION -- ETH-A, Bitfinex 2016-03 -> 2019-12, spot @0.10%")
    eth = R.load_eth_a()
    print(f"ETH-A: {len(eth):,} bars, {eth.index[0]} -> {eth.index[-1]}\n")
    out = []
    for k in (0.5, K_FROZEN):
        row = _row(R.compare(arm(k), eth, (None, None), R.spot(R.FEE_BASE),
                             label=f"boundary k={k:.2f}"), k, "eth-a")
        show(row)
        out.append(row)
        rows.append(row)
    # D2 shape on ETH too -- costs nothing and is the same mechanism claim.
    for k in (K_FROZEN,):
        row = _row(R.compare(arm(k), eth, (None, None), R.spot(R.FEE_STRESS),
                             label=f"boundary k={k:.2f}"), k, "eth-a")
        show(row)
        out.append(row)
        rows.append(row)
    return out


def cmd_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """(6) Does the mechanism even operate? Band exits and their overshoot.

    Runs v4's own rule (k=0, i.e. the incumbent bit-for-bit) and records
    every bar at which ``|desired - pos| > band``. The step v4 takes is the
    overshoot ``o``; the step the arm takes is ``o - k*band``. The whole
    mechanism is worth ``k*band`` of turnover per exit, so ``band/o`` is the
    fractional saving and ``o/band`` is the ceiling's reciprocal.
    """
    _print_header("(6) BAND-EXIT DIAGNOSTIC -- how much room is there, actually?")
    recs = []
    for split, window in SPLITS:
        frame, prefix = window_frame(df, window, KellyRegimeV4.warmup)
        rec: list = []
        v4like = TradeToBoundary(k=0.0)  # == kelly_regime_v4's rule exactly
        v4like._targets(frame, record=rec)
        n_period = len(frame) - prefix
        o = np.array([e[1] for e in rec if e[0] >= prefix])
        band = v4like.deadband
        if len(o) == 0:
            print(f"{split}: no band exits at all")
            continue
        ratio = o / band
        days = n_period / BARS_PER_DAY
        row = dict(
            split=split, bars=n_period, days=round(days, 1), exits=len(o),
            exits_per_year=round(len(o) / (days / 365.25), 1),
            o_mean=o.mean(), o_median=float(np.median(o)),
            o_p10=float(np.percentile(o, 10)), o_p25=float(np.percentile(o, 25)),
            o_p75=float(np.percentile(o, 75)), o_p90=float(np.percentile(o, 90)),
            o_max=o.max(),
            ratio_median=float(np.median(ratio)),
            frac_exits_o_lt_2band=float((ratio < 2.0).mean()),
            turnover_v4=float(o.sum()),
            turnover_saved_k1=float(band * len(o)),
            saved_frac=float(band * len(o) / o.sum()),
        )
        recs.append(row)
        print(f"{split}: {n_period:,} bars ({days:.0f} d), band={band:.2f}")
        print(f"    band exits                        : {len(o)}  "
              f"({row['exits_per_year']}/yr)")
        print(f"    overshoot o=|desired-pos| at exit : mean {o.mean():.3f}  "
              f"median {np.median(o):.3f}  p10 {row['o_p10']:.3f}  "
              f"p25 {row['o_p25']:.3f}  p75 {row['o_p75']:.3f}  "
              f"p90 {row['o_p90']:.3f}  max {o.max():.3f}")
        print(f"    o / band                          : median "
              f"{row['ratio_median']:.2f}   (frac of exits with o < 2*band: "
              f"{row['frac_exits_o_lt_2band']:.1%})")
        print(f"    total turnover v4 (sum o)         : {o.sum():.2f} "
              f"equity-units")
        print(f"    turnover the k=1 arm removes      : {band * len(o):.2f} "
              f"= {row['saved_frac']:.1%} of it")
        print(f"    => fee saving ceiling @0.10%      : "
              f"{band * len(o) * R.FEE_BASE:.4f} equity-units "
              f"({band * len(o) * R.FEE_BASE * 100:.2f}% of equity over the split)")
        print(f"    => fee saving ceiling @0.40%      : "
              f"{band * len(o) * R.FEE_STRESS:.4f} equity-units "
              f"({band * len(o) * R.FEE_STRESS * 100:.2f}% of equity over the split)")
        print()
    out = pd.DataFrame(recs)
    out.to_csv(OUT_DIR / "r64_conservative_bandexits.csv", index=False)
    return out


def cmd_causality(df: pd.DataFrame) -> bool:
    """(5) Truncation + tamper probe, plus the k=0 == v4 identity check.

    The class is not registered, so ``tests/test_causality_strict.py`` does
    not reach it. Three assertions:

    A. **k=0 identity.** ``TradeToBoundary(k=0)``'s target path is
       bit-identical to ``KellyRegimeV4``'s on the same frame. This is a
       regression check on the faithfulness of the copied loop -- if the
       copy drifted from v3's body, this fails.
    B. **Truncation.** Prepare on a frame truncated at ``m``; the target
       series over the shared prefix must be bit-identical to the
       full-frame run's. A statistic computed over the whole series (a
       scaler, quantile, mean, std) would break this.
    C. **Tamper.** Corrupt every bar after ``m`` (x3 up and /3 down,
       independently); no target at index < m may change, under either
       tamper, and the two tampers must agree with each other.
    """
    _print_header("(5) CAUSALITY -- truncation / tamper probe (class is unregistered)")
    frame = df.iloc[-400_000:].copy()
    ok = True

    a = TradeToBoundary(k=0.0)._targets(frame)
    b = KellyRegimeV4().prepare(frame.copy())["target"].to_numpy(dtype=float)
    ident = bool(np.array_equal(a, b))
    ok &= ident
    print(f"  A. k=0 target path == kelly_regime_v4 bit-for-bit : "
          f"{'PASS' if ident else 'FAIL'}")

    s = arm(K_FROZEN)
    full = s._targets(frame)
    for m in (len(frame) - 5_000, len(frame) - 50_000):
        trunc = s._targets(frame.iloc[:m].copy())
        same = bool(np.array_equal(full[:m], trunc))
        ok &= same
        print(f"  B. truncate at m={m:>7}: targets[:m] identical        : "
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
    print(f"  C. tamper x3 after m: targets[:m] unchanged        : "
          f"{'PASS' if c1 else 'FAIL'}")
    print(f"  C. tamper /3 after m: targets[:m] unchanged        : "
          f"{'PASS' if c2 else 'FAIL'}")
    print(f"  C. opposite tampers agree on targets[:m]           : "
          f"{'PASS' if c3 else 'FAIL'}")
    print(f"\n  CAUSALITY: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


# ================================================================= verdicts


def monotonicity(rows: list[dict], split: str, fee: float, market: str = "spot") -> dict:
    """Is the response monotone in k -- the mechanism's own signature?"""
    sel = sorted([r for r in rows if r["split"] == split and r["market"] == market
                  and abs(r["fee"] - fee) < 1e-12], key=lambda r: r["k"])
    ks = [r["k"] for r in sel]
    fees = [r["arm_fees"] for r in sel]
    dlog = [r["d_logret"] for r in sel]
    sharpe = [r["arm_sharpe"] for r in sel]

    def mono(v):
        d = np.diff(v)
        return ("non-decreasing" if np.all(d >= -1e-12) else
                "non-increasing" if np.all(d <= 1e-12) else "NON-MONOTONE")

    return dict(split=split, fee=fee, ks=ks, fees=fees, dlog=dlog,
                sharpe=sharpe, fees_mono=mono(fees), dlog_mono=mono(dlog),
                sharpe_mono=mono(sharpe),
                sharpe_spread=(max(sharpe) - min(sharpe)) if sharpe else float("nan"))


def report_sweep_shape(rows: list[dict]) -> None:
    _print_header("k-SWEEP SHAPE: monotonicity (the mechanism's signature) and D5 plateau")
    for split, _ in SPLITS:
        for fee in (R.FEE_BASE, R.FEE_STRESS):
            m = monotonicity(rows, split, fee)
            if not m["ks"]:
                continue
            print(f"\n  {split}  spot @{fee:.2%}")
            print(f"    k          : " + "  ".join(f"{k:>8.2f}" for k in m["ks"]))
            print(f"    arm fees $ : " + "  ".join(f"{v:>8,.0f}" for v in m["fees"]))
            print(f"    d_logret   : " + "  ".join(f"{v:>+8.4f}" for v in m["dlog"]))
            print(f"    arm Sharpe : " + "  ".join(f"{v:>8.3f}" for v in m["sharpe"]))
            print(f"    monotone?  : fees {m['fees_mono']}, "
                  f"d_logret {m['dlog_mono']}, Sharpe {m['sharpe_mono']}")
            print(f"    Sharpe spread over the k plateau: {m['sharpe_spread']:.3f} "
                  f"vs the +/-{SHARPE_NOISE_FLOOR} noise floor -> "
                  f"{'INSIDE the floor (a plateau, and an indistinguishable one)' if m['sharpe_spread'] <= 2 * SHARPE_NOISE_FLOOR else 'OUTSIDE the floor'}")


def report_d2(rows: list[dict]) -> None:
    _print_header("D2 REHEARSAL (inner): does the advantage GROW when the fee quadruples?")
    print("  REQUIRE d_logret(0.40%) > d_logret(0.10%). Inner-split rehearsal only;")
    print("  the pre-registered D2 is decided on the holdout by the operator.\n")
    print(f"  {'split':<12} {'k':>5} {'dlog@0.10%':>12} {'dlog@0.40%':>12} "
          f"{'grows?':>8}")
    for split, _ in SPLITS:
        for k in K_GRID:
            base = [r for r in rows if r["split"] == split and r["market"] == "spot"
                    and r["k"] == k and abs(r["fee"] - R.FEE_BASE) < 1e-12]
            stress = [r for r in rows if r["split"] == split and r["market"] == "spot"
                      and r["k"] == k and abs(r["fee"] - R.FEE_STRESS) < 1e-12]
            if not base or not stress:
                continue
            b, s = base[0]["d_logret"], stress[0]["d_logret"]
            print(f"  {split:<12} {k:>5.2f} {b:>+12.4f} {s:>+12.4f} "
                  f"{'YES' if R.d2_satisfied(b, s) else 'no':>8}")


def report_d4(rows: list[dict]) -> None:
    _print_header("D4 TURNOVER SANITY: did the mechanism operate? (fees must FALL)")
    print(f"  {'split':<12} {'market':<11} {'fee':>7} {'k':>5} "
          f"{'arm trades':>11} {'v4 trades':>10} {'arm fees':>11} "
          f"{'v4 fees':>11} {'fees fell?':>11}")
    for r in rows:
        print(f"  {r['split']:<12} {r['market']:<11} {r['fee']:>7.4f} "
              f"{r['k']:>5.2f} {r['arm_trades']:>11} {r['v4_trades']:>10} "
              f"{r['arm_fees']:>11,.0f} {r['v4_fees']:>11,.0f} "
              f"{'YES' if r['arm_fees'] < r['v4_fees'] else 'NO':>11}")


def report_d0(rows: list[dict]) -> None:
    _print_header("D0 RISK-MATCH GATE (mean notional): the residual-long consequence")
    print("  At k>0 the arm never returns to exactly zero -- it stops k*band away")
    print("  from a desired of 0 -- so it carries a residual long through bear")
    print("  regimes v4 sits out flat. That is what a no-trade region means, and")
    print("  it is exactly what D0 exists to catch.\n")
    print(f"  {'split':<12} {'market':<11} {'fee':>7} {'k':>5} {'c_arm':>8} "
          f"{'c_v4':>8} {'mismatch':>9} {'D0':>7}")
    for r in rows:
        print(f"  {r['split']:<12} {r['market']:<11} {r['fee']:>7.4f} "
              f"{r['k']:>5.2f} {r['c_arm']:>8.4f} {r['c_v4']:>8.4f} "
              f"{r['risk_mismatch']:>8.1%} "
              f"{'VOID' if r['d0_void'] else 'ok':>7}")


# ====================================================================== main


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "all"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 104)
    print("R-64 CONSERVATIVE: trade to the BOUNDARY of the no-trade region "
          "(Constantinides 1986 / Davis-Norman 1990)")
    print("=" * 104)
    print(f"arm         : TradeToBoundary(k=...), band = v4's shipped "
          f"deadband = {TradeToBoundary().deadband}")
    print(f"frozen k    : {K_FROZEN} (a priori: the literal continuous-time "
          f"policy, NOT peak-picked)")
    print(f"k grid      : {K_GRID}   (k=0.0 IS kelly_regime_v4)")
    print("splits      : INNER_TRAIN (-> 2020-12-31), INNER_VAL "
          "(2021-01-01 -> 2022-12-31)")
    print("holdout     : NOT READ. BTC frame truncated at 2022-12-31 on load.")

    df = load_btc_inner()
    print(f"BTC (inner) : {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")

    rows: list[dict] = []
    causal = None

    if cmd in ("all", "causality"):
        causal = cmd_causality(df)
    if cmd in ("all", "diagnostic"):
        cmd_diagnostic(df)
    if cmd in ("all", "sweep"):
        cmd_sweep(df, rows)
    if cmd in ("all", "stress"):
        cmd_stress(df, rows)
    if cmd in ("all", "futures"):
        cmd_futures(df, rows)
    if cmd in ("all", "eth"):
        cmd_eth(rows)

    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "r64_conservative_cells.csv",
                                  index=False)
        print(f"\nSaved cells -> {OUT_DIR / 'r64_conservative_cells.csv'}")

    if cmd == "all":
        spot_rows = [r for r in rows if r["market"] == "spot"
                     and r["split"] in ("inner-train", "inner-val")]
        report_sweep_shape(spot_rows)
        report_d2(spot_rows)
        report_d4(rows)
        report_d0(rows)

        _print_header("D3 (ETH-A falsification): sign of arm-minus-v4 log growth")
        for r in rows:
            if r["split"] == "eth-a":
                print(f"  k={r['k']:.2f} fee={r['fee']:.4f}  d_logret="
                      f"{r['d_logret']:+.4f} "
                      f"[{r['d_logret_lo']:+.3f}, {r['d_logret_hi']:+.3f}]  "
                      f"sign {'POSITIVE' if r['d_logret'] > 0 else 'NEGATIVE'}")

        _print_header("FROZEN CONFIGURATION PROPOSED FOR THE HOLDOUT")
        print("  experiments.r64_conservative_trade_to_boundary.TradeToBoundary(k=1.0)")
        print("  i.e. TradeToBoundary(k=1.0, horizons=(20, 40, 80), band=0.01,")
        print("       target_vol=0.55, max_leverage=2.0, vol_span=2304,")
        print("       deadband=0.10, vote_gamma=1.0, anchor_span_days=180,")
        print("       high_in=1.70, high_out=1.20, low_in=0.55, low_out=0.85)")
        print("  -- every argument except k is kelly_regime_v4's shipped default.")

    print()
    print("=" * 104)
    if causal is not None:
        print(f"Causality probe            : {'PASS' if causal else 'FAIL'}")
    print(f"Configurations evaluated   : {R.configs_evaluated()}")
    print("Holdout consultations added: 0 (BTC truncated at 2022-12-31; "
          "ETH-A ends 2019-12)")
    print("=" * 104)


if __name__ == "__main__":
    main()
