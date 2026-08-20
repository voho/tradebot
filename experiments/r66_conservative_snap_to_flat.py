"""R-66 (conservative branch, B-29 literal): R-64's trade-to-the-boundary rule
plus **one conditional** -- when the target is exactly flat, go exactly flat.

Pre-registration (shared, frozen, both branches): ``experiments/r66_shared.py``.
That file is not edited here, and neither is anything else outside this file.
Nothing is written to disk by this module: every number below is printed.

(Round renumbered R-65 -> R-66 by the operator: a concurrent session claimed
R-65 on ``main`` for a different round -- holding period / rank buffer -- while
this pre-registration was being written. R-31/R-32 same-day-parallel
precedent. Nothing else about the round changed.)

=====================================================================
THE ONE-LINE MECHANISM
=====================================================================

R-64's conservative arm replaced ``kelly_regime_v4``'s **trade-to-target**
destination with the **trade-to-the-boundary** destination that the
proportional-cost literature derives (Constantinides 1986; Davis & Norman
1990; Shreve & Soner 1994; Liu 2004)::

    v4:    if abs(desired - pos) > band:  pos = desired
    R-64:  if desired - pos > band:       pos = desired - k * band
           elif pos - desired > band:     pos = desired + k * band

It removed a real 43% of turnover, its advantage **grew** with the fee (D2
passed cleanly on both inner splits), and it still lost -- on ETH-A by
**-0.430 [-0.791, -0.101]** log-growth at k=1, the only interval in the whole
round that excluded zero, and on the wrong side. R-64 named the cause: a
no-trade region never lets the position reach *exactly* flat, so the arm
carried a residual long of up to ``k*band`` = 10% of equity through every bear
regime v4 sits out. This arm is that diagnosis made falsifiable in one line::

    if desired == 0.0:
        pos = 0.0                          # <-- the whole of B-29
    elif desired - pos > band:
        pos = desired - k * band
    elif pos - desired > band:
        pos = desired + k * band

That unconditional form is the **frozen primary**. It is B-29's literal
wording ("...that still snaps to exactly flat when ``desired == 0``") and it
matches the precedent in ``kelly_regime_ev`` (L-05), quoted below.

**A gated variant is carried as a labelled ablation, and it is instructive.**
The first draft of this arm guarded the snap with the band breach::

    if desired == 0.0 and pos != 0.0 and abs(desired - pos) > band:   # ABLATION

The reasoning was that a *destination* change must not introduce a new
*trigger* v4 lacks. The reasoning is wrong, and the operator's own diagnostic
caught it before any performance number was recorded: the boundary rule parks
the position at exactly ``k*band``, and ``|0 - 0.10|`` is **not** greater than
``0.10``, so the guarded snap never fires in exactly the state it exists for.
Measured below on both instruments, the gated form still holds a nonzero
position on a large minority of flat bars. The two forms are therefore both
reported: the difference between them isolates how much of any effect is the
residual long **specifically**, which is the round's actual question.

=====================================================================
WHY ``desired == 0.0`` EXACTLY, AND NOT A THRESHOLD
=====================================================================

``desired = frac[i] * scale``. ``frac`` is the mean of three **latched**
step-function votes, each exactly 0.0 or exactly 1.0 after
``ffill().fillna(0.0)``, so ``frac`` takes exactly the values {0, 1/3, 2/3, 1}
and ``desired`` is exactly ``0.0`` iff the regime gate is fully bearish (or the
vol estimate is not yet finite, where ``scale`` is set to 0.0 and ``pos`` is
already 0). There is no floating-point ambiguity to threshold around: the test
``desired == 0.0`` is a test of a discrete state, not of a small number.

**This is the project's own precedent, not a new idea.** ``kelly_regime_ev``
(L-05) already carries the identical exception, in ``on_bar``, and says why --
``src/tradebot/strategies/kelly_regime_ev.py`` lines 95-101::

    # Always allow a full exit: standing flat is the one move whose
    # benefit is not captured by the quadratic (it removes the whole
    # position's risk, and the regime gate asked for it).
    if desired == 0.0 and abs(current) > 1e-9:
        ctx.order_notional(0.0)
        return
    if abs(desired - current) > band:
        ctx.order_notional(desired)

L-05 tests ``desired == 0.0`` exactly, unconditionally on its band, for exactly
this reason -- and its band is not v4's fixed 0.10 but its own fee/vol/horizon
derived width. The frozen primary here is the same rule.

=====================================================================
A CONSEQUENCE WORTH STATING PLAINLY: k=0 IS NO LONGER v4
=====================================================================

Under R-64's rule, and under the gated ablation, ``k = 0`` reproduces
``kelly_regime_v4`` bit-for-bit, because at k=0 the boundary destination *is*
the target. Under the **unconditional** primary it does not, and the reason is
a genuinely new fact about the incumbent that this round had to measure:

**``kelly_regime_v4`` itself carries a residual long on flat bars.** Its rule
``if abs(desired - pos) > deadband: pos = desired`` does not fire when the
target falls from, say, 0.09 to 0.0, because that step is inside its own band.
So v4 too can sit long through a fully bearish regime. Measured below: on
BTC-inner this essentially never happens; on ETH-A it happens on a few percent
of v4's own flat bars. The disease R-64 diagnosed in the boundary arm is
present in the incumbent as well -- rarely, and asset-dependently.

Therefore the unconditional arm at ``k = 0`` is **v4 plus a full-exit rule**,
not v4; it is strictly cleaner than v4 on this axis. That is reported, not
hidden: ``cmd_causality`` asserts the k=0 identity for the *gated* form (which
is the regression check on the faithfulness of the copied loop) and separately
measures exactly where and how often the unconditional form departs from v4.

=====================================================================
ZERO NEW FITTED PARAMETERS
=====================================================================

``band`` **is** v4's own shipped ``deadband = 0.10``: re-used, never re-tuned,
never re-derived. ``k`` keeps R-64's grid {0, 0.25, 0.5, 0.75, 1.0} with
``k = 1.0`` frozen **a priori** as the proposal (the literal continuous-time
policy), not peak-picked. Everything else -- the 20/40/80 latched anchors, the
1% vote band, the conditional volatility target with its
``high_in/high_out/low_in/low_out`` hysteresis, ``target_vol=0.55``,
``max_leverage=2.0``, ``vol_span=8*BARS_PER_DAY``, ``anchor_span_days=180`` --
is copied character-for-character from ``KellyRegimeV3.prepare``.

=====================================================================
WHY THIS IS NOT A DUPLICATE
=====================================================================

- **R-64 conservative** (``experiments/r64_conservative_trade_to_boundary.py``)
  is this arm minus the conditional. That is the point: B-29 exists because
  R-64's own write-up says the two consequences of its one line -- a
  *destination* change (worth a real 43% of turnover, cost slope in its
  favour) and a *never-goes-flat* consequence (fatal) -- are separable in
  exactly one conditional, and asks for this experiment. The battery is
  inherited verbatim so the two rounds' numbers are directly comparable.
  R-64's arm is **imported** here, not re-implemented, so every comparison is
  against the literal published object.
- **R-64 novel (Garleanu-Pedersen partial adjustment)** replaced the band with
  a smooth *rate*. Nothing here re-tries that.
- **L-05 / L-06 (``kelly_regime_ev`` / ``_fast``)** derive the band's *width*
  from fee, vol and horizon and leave the destination at "jump to target".
  This arm leaves the width at v4's 0.10 and changes the *destination*. The
  full-exit exception is borrowed from L-05 and cited as such; the rest of L-05
  is not in play.
- **R-34 -> R-63**: twenty-two variants of ``frac`` or ``scale`` -- *what* to
  hold. This touches neither factor, only how the position moves.
- **R-12 / R-13 (section C, "tuning turnover to fit a fee tier")**: there is no
  free parameter to fit. D2 requires the advantage to **grow** as the fee
  quadruples, which a tier-fitted rule fails by construction.

=====================================================================
WHAT WOULD MAKE THIS FAIL -- NAMED BEFORE ANYTHING WAS RUN
=====================================================================

1. **The snap gives back exactly what the boundary saved.** B-29's own named
   failure mode, and the one to watch. The largest steps this strategy ever
   takes are regime exits -- vote to zero -- and the conditional forces
   precisely those to complete in full. If most of R-64's 43% turnover saving
   lived in those steps, D4 shows turnover flat or rising and the 43% was
   never bankable. ``cmd_turnover`` measures the snap events' share of total
   turnover directly rather than inferring it.
2. **The residual long was never the cause.** If ETH-A stays near -0.430 with
   flat reachable again, R-64's published diagnosis is **REFUTED** and the
   boundary destination is simply worse on this data for a reason not yet
   identified. That is a reportable answer, not a null, and section C's R-64
   row would need annotating in place.
3. **Lag.** Even with flat reachable, at k>0 the arm still sits ``k*band``
   short of every non-zero target: systematically under-exposed in a bull
   regime, over-exposed on the way down from a partial vote. Lag on a trend
   rule is the one thing this project has repeatedly measured as expensive
   (R-53's macro veto, R-60's CUSUM timing).
4. **k=1 remains a singular point.** R-64 derived it and it is unchanged here:
   after a step the arm sits at ``desired -/+ k*band``, so the *re-trigger*
   band is ``(1-k)*band``, which collapses to zero at k=1 and turns the
   position into a reflecting barrier that re-fires on any same-direction
   drift. Intended position changes therefore explode at k=1 (R-64 measured
   117,243 against v4's 543), while most fall below
   ``broker.REBALANCE_DEADBAND`` and are silently dropped, so **backtest trade
   counts are far lower than intended-change counts**. Both are reported below
   and they are never conflated.

=====================================================================
DATA DISCIPLINE
=====================================================================

**No bar dated 2023-01-01 or later is read anywhere in this file.** BTC is
loaded through ``R.load_btc_inner()``, which hard-truncates at 2022-12-31, so
the holdout cannot be touched even by accident. Splits are ``INNER_TRAIN``
(... -> 2020-12-31) and ``INNER_VAL`` (2021-01-01 -> 2022-12-31). The
falsification instrument is **ETH-A** (``R.load_eth_a()``, Bitfinex 2016-03 ->
2019-12), entirely pre-2020, which costs zero holdout consultations under the
R-19/R-28 convention. **Holdout consultations added by this branch: 0.** Even
if every inner gate clears, this branch STOPS and reports rather than reading
the holdout itself.

The class is deliberately **not** ``@register``-ed (ROUTINE.md step 5: an
unpromoted experiment lives under ``experiments/``), so
``tests/test_causality_strict.py`` does not reach it -- hence the truncation /
tamper probe carried here in ``cmd_causality``, as R-64 carried its own.

=====================================================================
A NOTE ON THE COPIED LOOP
=====================================================================

``KellyRegimeV3.prepare`` is one monolithic method and the position-update loop
is not factored out, so overriding the destination rule requires copying the
body. The copy in ``_targets`` below is faithful: the vote block, the vol /
slow-vol block, the ``full``/``steady`` arrays and the hysteresis state machine
are character-for-character v3's. The **only** edits are (a) the destination
block, and (b) two guarded recorder lines (``record``, ``desired_out``) that
write to caller-supplied containers and cannot affect the path.
``cmd_causality``'s check A asserts that the gated arm at k=0 is bit-identical
to ``KellyRegimeV4``, which is the regression test on that copy.

Usage::

    .venv/bin/python experiments/r66_conservative_snap_to_flat.py          # all
    .venv/bin/python experiments/r66_conservative_snap_to_flat.py sweep
    .venv/bin/python experiments/r66_conservative_snap_to_flat.py stress
    .venv/bin/python experiments/r66_conservative_snap_to_flat.py eth
    .venv/bin/python experiments/r66_conservative_snap_to_flat.py ablation
    .venv/bin/python experiments/r66_conservative_snap_to_flat.py diagnostic
    .venv/bin/python experiments/r66_conservative_snap_to_flat.py residual
    .venv/bin/python experiments/r66_conservative_snap_to_flat.py turnover
    .venv/bin/python experiments/r66_conservative_snap_to_flat.py futures
    .venv/bin/python experiments/r66_conservative_snap_to_flat.py causality
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiments.r66_shared as R  # noqa: E402
from experiments.r64_conservative_trade_to_boundary import (  # noqa: E402
    TradeToBoundary,  # R-64's arm, imported rather than re-implemented
)
from tradebot.broker import REBALANCE_DEADBAND, MarketSpec  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import prefix_bars  # noqa: E402

#: The mechanism's own index, inherited from R-64 unchanged. k=1 is the literal
#: Constantinides/Davis-Norman policy. NOT a selection grid -- the frozen
#: proposal is k=1.0, a priori.
K_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

#: The frozen proposal.
K_FROZEN = 1.0

#: The two cells R-64 published on ETH-A, for the D3b comparison.
R64_ETH_A = {1.0: -0.430, 0.5: -0.309}

SPLITS = (("inner-train", R.INNER_TRAIN), ("inner-val", R.INNER_VAL))


# ============================================================== the strategy


class SnapToFlat(KellyRegimeV4):
    """R-64's trade-to-the-boundary rule plus one conditional: when the target
    is exactly flat, go exactly flat.

    ``KellyRegimeV3.prepare``'s body copied faithfully (see the module
    docstring); the only edit to the path is the destination block::

        v4:      if abs(desired - pos) > band:  pos = desired
        R-64:    if desired - pos > band:       pos = desired - k*band
                 elif pos - desired > band:     pos = desired + k*band
        primary: if desired == 0.0:             pos = 0.0
                 elif desired - pos > band:     pos = desired - k*band
                 elif pos - desired > band:     pos = desired + k*band

    ``gated=True`` selects the labelled ablation, in which the snap is
    additionally guarded by ``abs(desired - pos) > band``. That guard defeats
    the purpose -- the boundary rule parks at exactly ``k*band`` and
    ``|0-0.10|`` is not ``> 0.10`` -- so the ablation still leaks a residual
    long; it is kept only because the primary-minus-ablation difference is
    exactly the residual long's contribution.

    ``band`` is v4's own shipped ``deadband = 0.10``, re-used, not re-fitted.
    Zero new fitted parameters.
    """

    name = "r66_conservative_snap_to_flat"
    warmup = 80 * BARS_PER_DAY + 10  # v4's

    def __init__(self, k: float = K_FROZEN, gated: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        if not np.isfinite(k) or k < 0.0 or k > 1.0:
            raise ValueError(f"k must be in [0, 1], got {k!r}")
        self.k = float(k)
        self.gated = bool(gated)

    # ---- v3's prepare body, with the destination rule replaced -------------

    def _targets(self, df: pd.DataFrame, record: list | None = None,
                 desired_out: np.ndarray | None = None) -> np.ndarray:
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
        gated = self.gated
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
            if desired_out is not None:
                desired_out[i] = desired
            # ---- THE CHANGE vs KellyRegimeV3.prepare ------------------------
            # B-29: a full exit is always completed in full (L-05's precedent).
            # `pos != 0.0` only suppresses no-op events; it is not a condition.
            if desired == 0.0 and pos != 0.0 and (
                    not gated or abs(desired - pos) > band):
                if record is not None:
                    record.append((i, "snap", pos, desired, abs(pos)))
                pos = 0.0
            elif desired - pos > band:
                if record is not None:
                    record.append((i, "up", pos, desired, abs(desired - kb - pos)))
                pos = desired - kb
            elif pos - desired > band:
                if record is not None:
                    record.append((i, "down", pos, desired, abs(desired + kb - pos)))
                pos = desired + kb
            # -----------------------------------------------------------------
            target[i] = pos

        return target

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["target"] = self._targets(df)
        return df


def arm(k: float = K_FROZEN, gated: bool = False) -> SnapToFlat:
    """A fresh arm at index ``k``. Every other constant is v4's shipped one."""
    return SnapToFlat(k=k, gated=gated)


# ================================================================== plumbing


def window_frame(df: pd.DataFrame, window, warmup: int) -> tuple[pd.DataFrame, int]:
    """Exactly the frame ``run_period`` hands to ``prepare``, plus the index at
    which trading starts. Re-derived here (the same three lines as
    ``window.run_period``) so the diagnostics see what the backtest saw."""
    start, end = window
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    prefix = prefix_bars(df, lo, warmup)
    return df.iloc[lo - prefix: hi], prefix


def _row(res: dict, k: float, split: str, variant: str = "primary") -> dict:
    out = dict(res)
    out["k"] = k
    out["split"] = split
    out["variant"] = variant
    return out


def _print_header(text: str) -> None:
    print("\n" + "=" * 122)
    print(text)
    print("=" * 122)


CELL_FMT = ("{var:<9} {k:>5} {split:<12} {mkt:<11} fee={fee:.4f} "
            "arm=${af:>11,.0f} v4=${vf:>11,.0f} "
            "Sh {ash:>5.2f}/{vsh:<5.2f} DD {add:>5.1f}/{vdd:<5.1f} "
            "tr {atr:>4}/{vtr:<4} fees ${afe:>8,.0f}/${vfe:<8,.0f} "
            "c {ac:.3f}/{vc:.3f} ({mm:>5.1%}) dlog {dl:+7.4f} "
            "[{lo:+.3f},{hi:+.3f}]{void}")


def show(row: dict) -> None:
    print(CELL_FMT.format(
        var=row["variant"], k=f"{row['k']:.2f}", split=row["split"],
        mkt=row["market"], fee=row["fee"], af=row["arm_final"], vf=row["v4_final"],
        ash=row["arm_sharpe"], vsh=row["v4_sharpe"],
        add=row["arm_dd"], vdd=row["v4_dd"],
        atr=row["arm_trades"], vtr=row["v4_trades"],
        afe=row["arm_fees"], vfe=row["v4_fees"],
        ac=row["c_arm"], vc=row["c_v4"], mm=row["risk_mismatch"],
        dl=row["d_logret"], lo=row["d_logret_lo"], hi=row["d_logret_hi"],
        void="  D0-VOID" if row["d0_void"] else ""))


# =============================================================== subcommands


def cmd_sweep(df: pd.DataFrame, rows: list) -> list[dict]:
    """(1) The k-sweep on both inner splits, spot @0.10%, arm vs v4."""
    _print_header("(1) k-SWEEP -- spot @0.10%, arm vs kelly_regime_v4, inner splits")
    out = []
    for split, window in SPLITS:
        for k in K_GRID:
            row = _row(R.compare(arm(k), df, window, R.spot(R.FEE_BASE),
                                 label=f"snap k={k:.2f}"), k, split)
            show(row)
            out.append(row)
            rows.append(row)
        print("-" * 122)
    return out


def cmd_stress(df: pd.DataFrame, rows: list) -> list[dict]:
    """(2) The same at 0.40% -- the D2 cost-mechanism shape, inner rehearsal."""
    _print_header("(2) k-SWEEP -- spot @0.40% (FEE_STRESS): the D2 shape, rehearsed inner")
    out = []
    for split, window in SPLITS:
        for k in K_GRID:
            row = _row(R.compare(arm(k), df, window, R.spot(R.FEE_STRESS),
                                 label=f"snap k={k:.2f}"), k, split)
            show(row)
            out.append(row)
            rows.append(row)
        print("-" * 122)
    return out


def cmd_eth(rows: list) -> list[dict]:
    """(3) D3 + D3b on ETH-A (Bitfinex 2016-03 -> 2019-12, +0 holdout)."""
    _print_header("(3) D3 / D3b -- ETH-A, Bitfinex 2016-03 -> 2019-12, spot @0.10%")
    eth = R.load_eth_a()
    print(f"ETH-A: {len(eth):,} bars, {eth.index[0]} -> {eth.index[-1]}")
    print(f"R-64's published cells on this identical window: "
          f"k=1.00 {R64_ETH_A[1.0]:+.3f} [-0.791, -0.101], "
          f"k=0.50 {R64_ETH_A[0.5]:+.3f} [-0.598, -0.043]")
    print(f"D3b bar (half the k=1 gap closed): d_logret > {R.D3B_BAR:+.3f}\n")
    out = []
    for k in (0.5, K_FROZEN):
        row = _row(R.compare(arm(k), eth, (None, None), R.spot(R.FEE_BASE),
                             label=f"snap k={k:.2f}"), k, "eth-a")
        show(row)
        print("      " + R.fmt(row))
        out.append(row)
        rows.append(row)
    # D2 shape on ETH too -- the same mechanism claim, and it costs nothing new.
    for k in (K_FROZEN,):
        row = _row(R.compare(arm(k), eth, (None, None), R.spot(R.FEE_STRESS),
                             label=f"snap k={k:.2f}"), k, "eth-a")
        show(row)
        out.append(row)
        rows.append(row)
    return out


def cmd_ablation(df: pd.DataFrame, rows: list) -> list[dict]:
    """(3b) The GATED ablation -- the same arm with the snap guarded by the
    band breach, which (as the operator's diagnostic showed) means it barely
    fires. primary minus ablation isolates the residual long's contribution."""
    _print_header("(3b) ABLATION -- gated snap (guarded by abs(desired-pos) > band), "
                  "spot @0.10%")
    print("  The guard defeats the snap: the boundary rule parks at exactly k*band and")
    print("  |0 - 0.10| is NOT > 0.10. This row is the residual-long control, not a")
    print("  candidate. primary - ablation = the residual long's contribution.\n")
    out = []
    eth = R.load_eth_a()
    cells = [(s, w, df) for s, w in SPLITS] + [("eth-a", (None, None), eth)]
    for split, window, frame in cells:
        ks = (0.5, K_FROZEN) if split == "eth-a" else (K_FROZEN,)
        for k in ks:
            row = _row(R.compare(arm(k, gated=True), frame, window,
                                 R.spot(R.FEE_BASE), label=f"gated k={k:.2f}"),
                       k, split, variant="ablation")
            show(row)
            out.append(row)
            rows.append(row)
    return out


def _path_stats(tgt: np.ndarray, lo: int) -> dict:
    """Total variation, intended-change count and mean notional of a target
    path over the tradable region. Free: no backtest, 0 configs counted."""
    d = np.abs(np.diff(tgt))[max(lo - 1, 0):]
    ch = d[d > 1e-9]
    seg = tgt[lo:]
    return dict(tv=float(ch.sum()), changes=int(len(ch)),
                mean_notional=float(np.abs(seg).mean()) if len(seg) else float("nan"),
                mean_step=float(ch.mean()) if len(ch) else 0.0)


def cmd_turnover(df: pd.DataFrame) -> list[dict]:
    """(4) D4 -- intended turnover off the target path, per arm and per k.

    These are **intended** position changes read straight off the target
    column, ignoring ``broker.REBALANCE_DEADBAND``. Backtest trade counts (in
    the sweep tables) are far lower because the broker silently drops small
    same-sign adjustments; the two are never conflated.

    B-29's named failure mode is that the largest steps this strategy takes are
    regime exits, and the conditional forces exactly those to complete in full
    -- so the fraction of turnover carried by snap events is the give-back.
    """
    _print_header("(4) D4 TURNOVER -- INTENDED changes off the target path (not fills)")
    eth = R.load_eth_a()
    cells = [("inner-train", R.INNER_TRAIN, df),
             ("inner-val", R.INNER_VAL, df),
             ("inner-full", (None, R.INNER_CUTOFF), df),
             ("eth-a", (None, None), eth)]
    out = []
    for split, window, src in cells:
        frame, prefix = window_frame(src, window, SnapToFlat.warmup)
        lo = max(prefix, SnapToFlat.warmup)
        desired = np.empty(len(frame))
        SnapToFlat(k=0.0)._targets(frame, desired_out=desired)
        flat_share = float((desired[lo:] == 0.0).mean())
        v4t = KellyRegimeV4().prepare(frame.copy())["target"].to_numpy(dtype=float)
        v4s = _path_stats(v4t, lo)
        print(f"\n  {split}: {len(frame) - lo:,} tradable bars, "
              f"{flat_share:.1%} of them have desired == 0.0")
        print(f"    {'arm':<24} {'k':>5} {'TV':>9} {'vs v4':>7} {'changes':>9} "
              f"{'meanstep':>9} {'mean notl':>10} {'snaps':>7} {'snap notl':>10} "
              f"{'snap %TV':>9}")
        print(f"    {'kelly_regime_v4':<24} {'--':>5} {v4s['tv']:>9.2f} "
              f"{1.0:>6.0%} {v4s['changes']:>9,} {v4s['mean_step']:>9.4f} "
              f"{v4s['mean_notional']:>10.4f} {'--':>7} {'--':>10} {'--':>9}")
        out.append(dict(split=split, arm="kelly_regime_v4", k=float("nan"), **v4s))
        for name, factory, ks in (
                ("R-64 boundary", lambda k: TradeToBoundary(k=k), (0.5, K_FROZEN)),
                ("snap gated (ablation)", lambda k: SnapToFlat(k=k, gated=True),
                 (0.5, K_FROZEN)),
                ("snap primary", lambda k: SnapToFlat(k=k), K_GRID)):
            for k in ks:
                s = factory(k)
                rec: list = []
                tgt = (s._targets(frame, record=rec)
                       if isinstance(s, SnapToFlat) else s._targets(frame, record=rec))
                st = _path_stats(tgt, lo)
                snaps = [e for e in rec if e[1] == "snap" and e[0] >= lo]
                snap_notl = float(sum(e[4] for e in snaps))
                print(f"    {name:<24} {k:>5.2f} {st['tv']:>9.2f} "
                      f"{st['tv'] / v4s['tv']:>6.0%} {st['changes']:>9,} "
                      f"{st['mean_step']:>9.4f} {st['mean_notional']:>10.4f} "
                      f"{len(snaps):>7,} {snap_notl:>10.2f} "
                      f"{snap_notl / st['tv'] if st['tv'] else float('nan'):>8.1%}")
                out.append(dict(split=split, arm=name, k=k, snaps=len(snaps),
                                snap_notional=snap_notl, **st))
    print("\n  'snaps' counts events of the conditional, so R-64's arm reports 0 by")
    print("  construction -- it has no such event; its exits to a flat target stop at")
    print("  k*band and leave the residual long behind (see the residual diagnostic).")
    print("  'changes' are INTENDED position changes, not fills: at k=1 the re-trigger")
    print("  band collapses to zero and the rule re-trades on any drift, so the count")
    print("  explodes while most of it is below broker.REBALANCE_DEADBAND and never")
    print("  executes. Backtest trade counts are in the sweep tables; do not conflate.")
    return out


def cmd_residual(df: pd.DataFrame) -> list[dict]:
    """(5) The residual-long diagnostic: the quantity that is supposed to
    explain the ETH-A gap, measured rather than assumed.

    On every bar where ``desired`` (v4's own raw target, before the band) is
    exactly 0.0 -- the bars the regime gate asks to sit out -- report the
    position actually held by v4, by R-64's arm, by the gated ablation and by
    the primary. Free: no backtest, 0 configs counted.
    """
    _print_header("(5) RESIDUAL-LONG DIAGNOSTIC -- position held on bars where "
                  "desired == 0.0 exactly")
    eth = R.load_eth_a()
    cells = [("inner-train", R.INNER_TRAIN, df),
             ("inner-val", R.INNER_VAL, df),
             ("inner-full", (None, R.INNER_CUTOFF), df),
             ("eth-a", (None, None), eth)]
    out = []
    for split, window, src in cells:
        frame, prefix = window_frame(src, window, SnapToFlat.warmup)
        lo = max(prefix, SnapToFlat.warmup)
        desired = np.empty(len(frame))
        SnapToFlat(k=0.0)._targets(frame, desired_out=desired)
        mask = (desired[lo:] == 0.0)
        n_bars = len(desired) - lo
        print(f"\n  {split}: {n_bars:,} tradable bars, {int(mask.sum()):,} "
              f"({mask.mean():.1%}) with desired == 0.0")
        if mask.sum() == 0:
            continue
        print(f"    {'arm':<24} {'k':>5} {'nonzero bars':>13} {'share':>7} "
              f"{'mean pos':>9} {'max pos':>8}")
        paths = [("kelly_regime_v4", float("nan"),
                  KellyRegimeV4().prepare(frame.copy())["target"].to_numpy(dtype=float))]
        for k in (0.5, K_FROZEN):
            paths.append((f"R-64 boundary", k, TradeToBoundary(k=k)._targets(frame)))
        for k in (0.5, K_FROZEN):
            paths.append(("snap gated (ablation)", k,
                          SnapToFlat(k=k, gated=True)._targets(frame)))
        for k in (0.5, K_FROZEN):
            paths.append(("snap primary", k, SnapToFlat(k=k)._targets(frame)))
        for name, k, path in paths:
            p = path[lo:][mask]
            nz = int((p != 0.0).sum())
            row = dict(split=split, arm=name, k=k, flat_bars=int(mask.sum()),
                       nonzero_bars=nz, nonzero_share=nz / mask.sum(),
                       mean_pos=float(p.mean()), max_pos=float(p.max()))
            out.append(row)
            print(f"    {name:<24} {'--' if not np.isfinite(k) else f'{k:.2f}':>5} "
                  f"{nz:>13,} {nz / mask.sum():>6.1%} {p.mean():>9.4f} "
                  f"{p.max():>8.4f}")
    print("\n  kelly_regime_v4's own row is the new fact: its rule "
          "`if abs(desired-pos) > deadband`")
    print("  does not fire when the target falls from inside the band to zero, so the")
    print("  incumbent can itself sit long through a fully bearish regime. The primary")
    print("  arm is therefore strictly cleaner than v4 on this axis, not merely equal.")
    return out


def fill_through(strategy, df: pd.DataFrame, window, market: MarketSpec) -> dict:
    """B-30: how many of the strategy's intended position changes became fills?

    ``prepare`` decides the whole target path open-loop; ``on_bar`` emits an
    order whenever the path moves; the broker then drops any same-sign
    adjustment worth less than ``REBALANCE_DEADBAND`` (5%) of *max* notional --
    equity on spot, ``equity x leverage`` on futures. On 5x futures that is 25%
    of equity, larger than most steps either arm asks for, so about half of even
    v4's own intended futures rebalances are silently discarded. Approach
    copied from ``experiments/r64_conservative_trade_to_boundary.fill_through``.
    """
    frame, prefix = window_frame(df, window, strategy.warmup)
    if isinstance(strategy, (SnapToFlat, TradeToBoundary)):
        tgt = strategy._targets(frame)
    else:
        tgt = strategy.prepare(frame.copy())["target"].to_numpy(dtype=float)
    n = len(tgt)
    lo = max(prefix, strategy.warmup)
    steps = np.abs(np.diff(tgt))[max(lo - 1, 0): n - 2] if n > 2 else np.array([])
    intended = steps > 1e-9
    res, _ = R.measure(strategy, df, window, market)
    n_fills = len(res.fills)
    threshold = REBALANCE_DEADBAND * market.leverage
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
    """(6) Futures 5x -- optional context only; spot is the decision market.
    Reported ONLY together with the B-30 fill-through table, per the frozen
    pre-registration's D6 note."""
    _print_header("(6) FUTURES 5x -- CONTEXT ONLY (spot is the decision market), with "
                  "B-30 fill-through")
    print(f"  broker.REBALANCE_DEADBAND = {REBALANCE_DEADBAND}: on 5x futures a "
          f"same-sign step below {REBALANCE_DEADBAND * 5:.0%} of EQUITY is silently")
    print("  dropped. No futures number below may be quoted without the fill-through")
    print("  table that follows it.\n")
    out = []
    fut = R.futures()
    for split, window in SPLITS:
        row = _row(R.compare(arm(K_FROZEN), df, window, fut,
                             label=f"snap k={K_FROZEN:.2f}"), K_FROZEN, split)
        show(row)
        out.append(row)
        rows.append(row)

    print(f"\n  {'strategy':<28} {'split':<12} {'market':<11} {'intended':>9} "
          f"{'fills':>7} {'ratio':>7} {'<band':>7} {'medstep':>8}")
    for market in (R.spot(R.FEE_BASE), fut):
        for split, window in SPLITS:
            for label, s in (("kelly_regime_v4", R.v4()),
                             (f"snap k={K_FROZEN:.2f}", arm(K_FROZEN))):
                d = fill_through(s, df, window, market)
                print(f"  {label:<28} {split:<12} {market.name:<11} "
                      f"{d['intended']:>9,} {d['fills']:>7,} {d['fill_ratio']:>7.3f} "
                      f"{d['below_broker_band']:>7,} {d['median_step']:>8.4f}")
    return out


def cmd_diagnostic(df: pd.DataFrame) -> None:
    """(7) Band-exit accounting: how much room the destination change has, and
    how much of it the snap conditional hands back. Free, 0 configs."""
    _print_header("(7) BAND-EXIT DIAGNOSTIC -- exits, overshoot, and the exits that "
                  "are exits-to-flat")
    for split, window in SPLITS:
        frame, prefix = window_frame(df, window, KellyRegimeV4.warmup)
        rec: list = []
        v4like = SnapToFlat(k=0.0, gated=True)  # == kelly_regime_v4's rule exactly
        v4like._targets(frame, record=rec)
        lo = max(prefix, KellyRegimeV4.warmup)
        ev = [e for e in rec if e[0] >= lo]
        if not ev:
            print(f"  {split}: no band exits at all")
            continue
        band = v4like.deadband
        o = np.array([abs(e[3] - e[2]) for e in ev])  # |desired - pos| at exit
        is_snap = np.array([e[1] == "snap" for e in ev])
        n_period = len(frame) - lo
        days = n_period / BARS_PER_DAY
        print(f"  {split}: {n_period:,} bars ({days:.0f} d), band={band:.2f}")
        print(f"    band exits (v4's own rule)        : {len(o)} "
              f"({len(o) / (days / 365.25):.1f}/yr)")
        print(f"    of which exits to a flat target   : {int(is_snap.sum())} "
              f"({is_snap.mean():.1%} of exits)")
        print(f"    overshoot o=|desired-pos| at exit : mean {o.mean():.3f}  "
              f"median {np.median(o):.3f}  max {o.max():.3f}")
        if is_snap.any():
            print(f"    o at exits-to-flat                : mean "
                  f"{o[is_snap].mean():.3f}  median {np.median(o[is_snap]):.3f}")
        if (~is_snap).any():
            print(f"    o at other exits                  : mean "
                  f"{o[~is_snap].mean():.3f}  median {np.median(o[~is_snap]):.3f}")
        print(f"    total turnover v4 (sum o)         : {o.sum():.2f} equity-units")
        print(f"    R-64 would remove (band per exit) : {band * len(o):.2f} "
              f"= {band * len(o) / o.sum():.1%} of it")
        print(f"    this arm removes (non-flat exits) : "
              f"{band * (~is_snap).sum():.2f} = "
              f"{band * (~is_snap).sum() / o.sum():.1%} of it")
        print(f"    => the conditional gives back     : "
              f"{band * is_snap.sum():.2f} equity-units "
              f"({band * is_snap.sum() / (band * len(o)):.1%} of R-64's saving)\n")
    print("  NOTE: these are exits of v4's own rule, so 'R-64 would remove' is the")
    print("  first-order accounting only. At k=1 the re-trigger band collapses and")
    print("  R-64's arm then re-trades continuously; cmd_turnover has the real path.")


def cmd_causality(df: pd.DataFrame) -> bool:
    """(8) Truncation + tamper probe, plus the k=0 identity checks.

    The class is not registered, so ``tests/test_causality_strict.py`` does not
    reach it. The assertions:

    A. **k=0 identity (gated form).** ``SnapToFlat(k=0, gated=True)``'s target
       path is bit-identical to ``KellyRegimeV4``'s on the same frame -- the
       regression check on the faithfulness of the copied loop.
    A2. **k=0, unconditional form, vs v4.** NOT asserted equal: the primary
       snap also completes exits v4 leaves inside its own band. Every bar at
       which they differ must have ``desired == 0.0``, and that is asserted --
       it is the precise statement that the only edit is the full-exit rule.
    A3. **Snap is the only difference from R-64 at k=1.** Every differing bar
       must be at or after the first snap event.
    B. **Truncation.** Prepare on a frame truncated at m; the target series over
       the shared prefix must be bit-identical to the full-frame run's. A
       statistic computed over the whole series would break this.
    C. **Tamper.** Corrupt every bar after m (x3 up and /3 down); no target at
       index < m may change under either tamper, and the two must agree.
    """
    _print_header("(8) CAUSALITY -- truncation / tamper probe (class is unregistered)")
    frame = df.iloc[-400_000:].copy()
    ok = True

    desired = np.empty(len(frame))
    a = SnapToFlat(k=0.0, gated=True)._targets(frame, desired_out=desired)
    b = KellyRegimeV4().prepare(frame.copy())["target"].to_numpy(dtype=float)
    ident = bool(np.array_equal(a, b))
    ok &= ident
    print(f"  A.  gated k=0 target path == kelly_regime_v4 bit-for-bit : "
          f"{'PASS' if ident else 'FAIL'}")

    a0 = SnapToFlat(k=0.0)._targets(frame)
    diff0 = np.flatnonzero(a0 != b)
    a2 = bool(len(diff0) == 0 or np.all(desired[diff0] == 0.0))
    ok &= a2
    print(f"  A2. primary k=0 differs from v4 ONLY where desired==0   : "
          f"{'PASS' if a2 else 'FAIL'}  ({len(diff0):,} bars differ; "
          f"all have desired==0: {a2})")

    rec: list = []
    s1 = SnapToFlat(k=K_FROZEN)
    full = s1._targets(frame, record=rec)
    r64_full = TradeToBoundary(k=K_FROZEN)._targets(frame)
    snap_idx = [e[0] for e in rec if e[1] == "snap"]
    diff_idx = np.flatnonzero(full != r64_full)
    if len(diff_idx) == 0:
        a3 = len(snap_idx) == 0
        note = "identical paths (no snap fired on this frame)"
    else:
        a3 = bool(snap_idx and diff_idx[0] >= min(snap_idx))
        note = (f"{len(diff_idx):,} bars differ, first at {int(diff_idx[0])}, "
                f"first snap at {min(snap_idx) if snap_idx else None}")
    ok &= a3
    print(f"  A3. differs from R-64 only from the first snap on       : "
          f"{'PASS' if a3 else 'FAIL'}  ({note})")

    for m in (len(frame) - 5_000, len(frame) - 50_000):
        trunc = s1._targets(frame.iloc[:m].copy())
        same = bool(np.array_equal(full[:m], trunc))
        ok &= same
        print(f"  B.  truncate at m={m:>7}: targets[:m] identical       : "
              f"{'PASS' if same else 'FAIL'}")

    m = len(frame) - 5_000
    up, down = frame.copy(), frame.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[m:, up.columns.get_loc(col)] *= 3.0
        down.iloc[m:, down.columns.get_loc(col)] /= 3.0
    up.iloc[m:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[m:, down.columns.get_loc("volume")] /= 7.0
    tu, td = s1._targets(up), s1._targets(down)
    c1 = bool(np.array_equal(tu[:m], full[:m]))
    c2 = bool(np.array_equal(td[:m], full[:m]))
    c3 = bool(np.array_equal(tu[:m], td[:m]))
    ok &= c1 and c2 and c3
    print(f"  C.  tamper x3 after m: targets[:m] unchanged             : "
          f"{'PASS' if c1 else 'FAIL'}")
    print(f"  C.  tamper /3 after m: targets[:m] unchanged             : "
          f"{'PASS' if c2 else 'FAIL'}")
    print(f"  C.  opposite tampers agree on targets[:m]                : "
          f"{'PASS' if c3 else 'FAIL'}")
    print(f"\n  CAUSALITY: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


# ================================================================= reporting


def _spot(rows: list[dict], split: str, fee: float,
          variant: str = "primary") -> list[dict]:
    return sorted([r for r in rows if r["split"] == split and r["market"] == "spot"
                   and r.get("variant", "primary") == variant
                   and abs(r["fee"] - fee) < 1e-12], key=lambda r: r["k"])


def report_d5(rows: list[dict]) -> dict:
    """D5: the response must be MONOTONE in the mechanism's own index."""
    _print_header("D5 -- MONOTONICITY IN k (the mechanism's signature) and the "
                  "Sharpe plateau")
    print("  R-64 found NON-MONOTONE responses on 3 of 4 split x tier cells and read")
    print("  that as the signature that something other than the step-length change")
    print("  was moving the numbers. R.monotone allows ties; a sign change fails.\n")
    verdict = {}
    for split, _ in SPLITS:
        for fee in (R.FEE_BASE, R.FEE_STRESS):
            sel = _spot(rows, split, fee)
            if not sel:
                continue
            ks = [r["k"] for r in sel]
            fees = [r["arm_fees"] for r in sel]
            dlog = [r["d_logret"] for r in sel]
            sharpe = [r["arm_sharpe"] for r in sel]
            dd = [r["arm_dd"] for r in sel]
            m_dlog = R.monotone(dlog)
            verdict[(split, fee)] = m_dlog
            print(f"  {split}  spot @{fee:.2%}")
            print("    k          : " + "  ".join(f"{k:>8.2f}" for k in ks))
            print("    arm fees $ : " + "  ".join(f"{v:>8,.0f}" for v in fees))
            print("    d_logret   : " + "  ".join(f"{v:>+8.4f}" for v in dlog))
            print("    arm Sharpe : " + "  ".join(f"{v:>8.3f}" for v in sharpe))
            print("    arm maxDD% : " + "  ".join(f"{v:>8.2f}" for v in dd))
            print(f"    R.monotone : fees {R.monotone(fees)}, d_logret {m_dlog}, "
                  f"Sharpe {R.monotone(sharpe)}, maxDD {R.monotone(dd)}")
            spread = max(sharpe) - min(sharpe)
            print(f"    Sharpe spread over the k plateau: {spread:.3f} vs the "
                  f"+/-{R.SHARPE_NOISE_FLOOR} noise floor -> "
                  f"{'INSIDE (indistinguishable)' if spread <= 2 * R.SHARPE_NOISE_FLOOR else 'OUTSIDE'}\n")
    return verdict


def report_d2(rows: list[dict]) -> dict:
    _print_header("D2 -- COST-MECHANISM TEST: does the advantage GROW when the fee "
                  "quadruples?")
    print("  REQUIRE d_logret(0.40%) > d_logret(0.10%), per k. Inner-split rehearsal;")
    print("  the pre-registered D2 is decided on the holdout, which this branch does")
    print("  not read. R-64's arm passed this cleanly at every k>0 on both splits.\n")
    print(f"  {'split':<12} {'k':>5} {'dlog@0.10%':>12} {'dlog@0.40%':>12} "
          f"{'delta':>9} {'grows?':>8}")
    out = {}
    for split, _ in SPLITS:
        base = {r["k"]: r["d_logret"] for r in _spot(rows, split, R.FEE_BASE)}
        stress = {r["k"]: r["d_logret"] for r in _spot(rows, split, R.FEE_STRESS)}
        for k in K_GRID:
            if k not in base or k not in stress:
                continue
            ok = R.d2_satisfied(base[k], stress[k])
            out[(split, k)] = ok
            print(f"  {split:<12} {k:>5.2f} {base[k]:>+12.4f} {stress[k]:>+12.4f} "
                  f"{stress[k] - base[k]:>+9.4f} {'YES' if ok else 'no':>8}")
    ea = {r["fee"]: r["d_logret"] for r in rows
          if r["split"] == "eth-a" and r["k"] == K_FROZEN
          and r.get("variant", "primary") == "primary"}
    if R.FEE_BASE in ea and R.FEE_STRESS in ea:
        ok = R.d2_satisfied(ea[R.FEE_BASE], ea[R.FEE_STRESS])
        out[("eth-a", K_FROZEN)] = ok
        print(f"  {'eth-a':<12} {K_FROZEN:>5.2f} {ea[R.FEE_BASE]:>+12.4f} "
              f"{ea[R.FEE_STRESS]:>+12.4f} {ea[R.FEE_STRESS] - ea[R.FEE_BASE]:>+9.4f} "
              f"{'YES' if ok else 'no':>8}")
    return out


def report_d0_d4(rows: list[dict]) -> None:
    _print_header("D0 (risk match) and D4 (fees/trades), every measured cell")
    print(f"  {'variant':<10} {'split':<12} {'market':<11} {'fee':>7} {'k':>5} "
          f"{'c_arm':>8} {'c_v4':>8} {'mismatch':>9} {'D0':>6} {'arm fees':>10} "
          f"{'v4 fees':>10} {'fell?':>6} {'arm tr':>7} {'v4 tr':>6}")
    for r in rows:
        print(f"  {r.get('variant', 'primary'):<10} {r['split']:<12} "
              f"{r['market']:<11} {r['fee']:>7.4f} "
              f"{r['k']:>5.2f} {r['c_arm']:>8.4f} {r['c_v4']:>8.4f} "
              f"{r['risk_mismatch']:>8.2%} {'VOID' if r['d0_void'] else 'ok':>6} "
              f"{r['arm_fees']:>10,.0f} {r['v4_fees']:>10,.0f} "
              f"{'YES' if r['arm_fees'] < r['v4_fees'] else 'NO':>6} "
              f"{r['arm_trades']:>7} {r['v4_trades']:>6}")


def report_d3(rows: list[dict]) -> None:
    _print_header("D3 (sign on ETH-A) and D3b (is R-64's residual-long diagnosis "
                  "CONFIRMED or REFUTED?)")
    for r in rows:
        if r["split"] != "eth-a":
            continue
        print(f"  {r.get('variant', 'primary'):<10} k={r['k']:.2f} "
              f"fee={r['fee']:.4f}  d_logret={r['d_logret']:+.4f} "
              f"[{r['d_logret_lo']:+.3f}, {r['d_logret_hi']:+.3f}]  "
              f"sign {'POSITIVE' if r['d_logret'] > 0 else 'NEGATIVE'}  "
              f"(D3 {'pass' if r['d_logret'] > 0 else 'FAIL'})")
    print()
    for k in (K_FROZEN, 0.5):
        for variant in ("primary", "ablation"):
            sel = [r for r in rows if r["split"] == "eth-a" and r["k"] == k
                   and r.get("variant", "primary") == variant
                   and abs(r["fee"] - R.FEE_BASE) < 1e-12]
            if not sel:
                continue
            d = sel[0]["d_logret"]
            r64 = R64_ETH_A[k]
            closed = (d - r64) / (0.0 - r64)
            print(f"  k={k:.2f} {variant:<9}:  R-64 {r64:+.3f}  ->  {d:+.4f}   "
                  f"gap closed: {closed:.1%}")
    sel = [r for r in rows if r["split"] == "eth-a" and r["k"] == K_FROZEN
           and r.get("variant", "primary") == "primary"
           and abs(r["fee"] - R.FEE_BASE) < 1e-12]
    if sel:
        d = sel[0]["d_logret"]
        ok = R.d3b_satisfied(d)
        print(f"\n  D3b bar is d_logret > {R.D3B_BAR:+.3f} (half the k=1 gap closed).")
        print(f"  primary k=1.00 on ETH-A = {d:+.4f}  ->  R.d3b_satisfied = {ok}")
        print(f"  => R-64's residual-long diagnosis is "
              f"{'CONFIRMED' if ok else 'REFUTED'} by this measurement.")


# ====================================================================== main


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "all"

    print("=" * 122)
    print("R-66 CONSERVATIVE (B-29): trade to the BOUNDARY, but when the target is "
          "exactly flat, go exactly flat")
    print("=" * 122)
    print(f"arm         : SnapToFlat(k=...), band = v4's shipped deadband = "
          f"{SnapToFlat().deadband}")
    print("primary     : if desired == 0.0: pos = 0.0   (UNCONDITIONAL; B-29's literal "
          "wording, L-05's precedent)")
    print("ablation    : the same guarded by abs(desired-pos) > band -- reported as a "
          "labelled control only")
    print(f"frozen k    : {K_FROZEN} (a priori: the literal continuous-time policy, "
          f"NOT peak-picked)")
    print(f"k grid      : {K_GRID}")
    print("baseline    : kelly_regime_v4, unmodified, paired on identical days")
    print("splits      : INNER_TRAIN (-> 2020-12-31), INNER_VAL (2021-01-01 -> "
          "2022-12-31)")
    print("holdout     : NOT READ. BTC loaded via R.load_btc_inner() "
          "(hard-truncated 2022-12-31).")

    df = R.load_btc_inner()
    print(f"BTC (inner) : {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")
    assert str(df.index[-1])[:10] <= "2022-12-31", "holdout leaked into the frame"

    rows: list[dict] = []
    causal = None
    d5 = d2 = None

    if cmd in ("all", "causality"):
        causal = cmd_causality(df)
    if cmd in ("all", "diagnostic"):
        cmd_diagnostic(df)
    if cmd in ("all", "residual"):
        cmd_residual(df)
    if cmd in ("all", "turnover", "diagnostic"):
        cmd_turnover(df)
    if cmd in ("all", "sweep"):
        cmd_sweep(df, rows)
    if cmd in ("all", "stress"):
        cmd_stress(df, rows)
    if cmd in ("all", "eth"):
        cmd_eth(rows)
    if cmd in ("all", "ablation"):
        cmd_ablation(df, rows)
    if cmd in ("all", "futures"):
        cmd_futures(df, rows)

    if cmd == "all":
        spot_rows = [r for r in rows if r["market"] == "spot"]
        d2 = report_d2(spot_rows)
        d5 = report_d5(spot_rows)
        report_d0_d4(rows)
        report_d3(rows)

        # ---------------------------------------------------------- verdict
        _print_header("VERDICT (R.promotion), against the frozen bar in r66_shared.py")
        val = _spot(rows, "inner-val", R.FEE_BASE)
        sel = [r for r in val if r["k"] == K_FROZEN]
        eth = [r for r in rows if r["split"] == "eth-a" and r["k"] == K_FROZEN
               and r.get("variant", "primary") == "primary"
               and abs(r["fee"] - R.FEE_BASE) < 1e-12]
        if sel and eth:
            v = sel[0]
            turn_fell = all(r["arm_fees"] < r["v4_fees"] for r in rows
                            if r["k"] == K_FROZEN and r["market"] == "spot"
                            and r.get("variant", "primary") == "primary")
            plateau = all((d5 or {}).values())
            d2_ok = all(ok for (s, k), ok in (d2 or {}).items() if k == K_FROZEN)
            print("  D1 is a HOLDOUT gate and this branch does not read the holdout.")
            print("  The call below substitutes the inner-validation cell for D1 and is")
            print("  therefore a REHEARSAL of the bar, not the bar itself. Every other")
            print("  gate is decided on data this branch is allowed to read.\n")
            print(f"    D0 void (inner-val, k={K_FROZEN})      : {v['d0_void']} "
                  f"(mismatch {v['risk_mismatch']:.2%})")
            print(f"    D1-rehearsal point / excl. zero      : {v['d_logret']:+.4f} / "
                  f"{v['d_logret_excludes_zero']}")
            print(f"    D2 (advantage grows with the fee)    : {d2_ok}")
            print(f"    D3 (ETH-A sign positive)             : "
                  f"{eth[0]['d_logret'] > 0} ({eth[0]['d_logret']:+.4f})")
            print(f"    D3b (R-64 diagnosis, > {R.D3B_BAR:+.3f})      : "
                  f"{R.d3b_satisfied(eth[0]['d_logret'])}")
            print(f"    D4 (fees fell, all spot k=1 primary) : {turn_fell}")
            print(f"    D5 (monotone in k, all 4 cells)      : {plateau}")
            print("    beats buy_and_hold OOS               : NOT EVALUATED "
                  "(holdout not read)")
            verdict = R.promotion(
                d0_void=v["d0_void"], d1_point=v["d_logret"],
                d1_excludes_zero=v["d_logret_excludes_zero"], d2_ok=d2_ok,
                d3_eth_a=eth[0]["d_logret"], turnover_fell=turn_fell,
                plateau=plateau, beats_hold_oos=False)
            print(f"\n  R.promotion(...) = {verdict}")

    print()
    print("=" * 122)
    if causal is not None:
        print(f"Causality probe            : {'PASS' if causal else 'FAIL'}")
    print(f"Configurations evaluated   : {R.configs_evaluated()}")
    print("Holdout consultations added: 0 (BTC hard-truncated at 2022-12-31 by "
          "R.load_btc_inner; ETH-A ends 2019-12)")
    print("=" * 122)


if __name__ == "__main__":
    main()
