"""R-66 (novel branch): the no-trade band's **width profile in the exposure
level** -- ``band(f) = deadband * f**p`` instead of a constant ``deadband``.

Pre-registration (shared, frozen, both branches): ``experiments/r66_shared.py``.
That file is not edited here, and neither is anything else outside this file.
The conservative branch (``r66_conservative_snap_to_flat.py``) is owned by
another session and is not read, imported or touched.

=====================================================================
THE ONE-LINE MECHANISM
=====================================================================

``kelly_regime_v4``'s position-update loop -- inherited unchanged from
``KellyRegime.prepare`` via ``KellyRegimeV3.prepare``, untouched since L-04 --
is a no-trade band of **constant** half-width around the current position::

    if abs(desired - pos) > self.deadband:      # deadband = 0.10, always
        pos = desired

Every prior attempt on this object changed either the width **as a scalar**
(L-05/L-06 ``kelly_regime_ev``: ``|df| > 2*fee/(H*sigma^2)``, still one number
applied at every exposure level) or the **destination** (R-64: trade to the
band's boundary rather than to the target). Nobody has changed *how the width
varies with the exposure level it is a band around*. This branch does exactly
that, and nothing else::

    band(f) = deadband * (f / f_ref) ** p ,   f_ref = 1.0
    if abs(desired - pos) > band(desired):   pos = desired

The destination is still ``desired`` -- v4's own trade-to-target -- so this is
strictly orthogonal to R-64's change.

=====================================================================
THE DERIVATION, AND WHAT WAS ACTUALLY VERIFIED
=====================================================================

For a small **proportional** cost the optimal no-trade region is a band around
the *frictionless* target whose half-width is O(cost^(1/3)) and, crucially,
whose size depends on the target level itself. The verified statement, quoted
from the source rather than paraphrased:

    pi_+/- = pi_* +/- ( 3/(4 gamma) * pi_*^2 (1 - pi_*)^2 )^(1/3) * eps^(1/3)
             + O(eps)

  -- Gerhold, S., Guasoni, P., Muhle-Karbe, J. & Schachermayer, W. (2014).
     "Transaction costs, trading volume, and the liquidity premium." Finance
     and Stochastics 18(1), 1-37 (arXiv:1108.1167). ``eps`` is the bid-ask
     spread, ``pi_*`` = mu/(gamma sigma^2) the frictionless Merton weight.
     VERIFIED: the formula above was read off the paper's own text.

  -- Muhle-Karbe, J., Reppen, M. & Soner, H. M. (2017). "A primer on portfolio
     choice with small transaction costs." Annual Review of Financial
     Economics 9, 301-331 (arXiv:1612.01302). States the CRRA/Black-Scholes
     half-width as ``Delta pi_BS = ( 3/(2 gamma) pi_BS^2 (1 - pi_BS)^2 )^(1/3)``
     (times lambda^(1/3)), and the O(lambda^(1/3)) width / O(lambda^(2/3))
     welfare-loss pair. VERIFIED by direct read.

     Note the constant differs (3/(4 gamma) vs 3/(2 gamma)): that is the
     spread-vs-half-spread convention, not a disagreement. **Only the shape
     matters here**, and both sources agree on it exactly:
     half-width ~ ( f^2 (1-f)^2 * cost )^(1/3).

  -- Rogers, L. C. G. (2004). "Why is the effect of proportional transaction
     costs O(delta^(2/3))?" In Mathematics of Finance, Contemporary
     Mathematics 351, 303-308. PARTIALLY VERIFIED: the paper exists with that
     title and venue and the O(delta^(2/3)) welfare / O(delta^(1/3)) width
     pair is what it is cited for throughout the surveys above; the primary
     text itself was not reachable, so it is cited here only for the
     exponent's *intuition*, which is elementary and reproduced below.

  -- Janecek, K. & Shreve, S. E. (2004). "Asymptotic analysis for optimal
     investment and consumption with transaction costs." Finance and
     Stochastics 8(2), 181-206. PARTIALLY VERIFIED: abstract confirmed
     ("asymptotic expansion of the value function in powers of lambda^(1/3)"
     and "asymptotic results on the boundary of the no-trade region"); the
     explicit boundary formula was behind a paywall and was NOT read here.
     The shape claim above therefore rests on Gerhold et al. and the primer,
     both of which were read.

The exponent is elementary and does not need trust: staying inside a band of
half-width D costs a displacement penalty quadratic in D (the value function
is smooth and at an interior maximum), while the trading needed to stay inside
it scales as 1/D. Minimising ``c1*D^2 + c2*eps/D`` gives ``D ~ eps^(1/3)``.
The ``f^2(1-f)^2`` factor is the local curvature of that same value function,
which vanishes where the frictionless target vanishes.

**The structurally important consequence, and this branch's entire point.**
Near f = 0 the half-width scales as **f^(2/3)**: it *vanishes* as the target
goes to zero. Applied here, ``kelly_regime_v4``'s bear-regime exit -- the vote
goes to exactly 0.0 -- lands the position on **exactly flat, with no residual
long**, because the band it would have to escape has zero width there. That is
the same fix the conservative branch of this round adds by hand as a special
case; here it is a *derived consequence* of the width profile, with no
conditional written for zero anywhere in the loop.

=====================================================================
WHAT WAS DELIBERATELY NOT IMPLEMENTED, AND WHY
=====================================================================

The verified shape is ``(f^2 (1-f)^2)^(1/3)``; this branch implements only the
``f^(2/3)`` half of it and drops ``(1-f)^(2/3)``. Stated plainly because it is
a real deviation from the theory:

- ``(1-f)^2`` is the curvature term for a weight in [0,1]. Here ``desired``
  ranges over [0, max_leverage] = [0, 2] (``frac`` in [0,1] times ``scale``
  capped at 2.0), so ``(1-f)^2`` is zero at f=1 -- the middle of the operating
  range, and precisely the point where ``f_ref`` anchors the band to v4's own
  shipped 0.10. A profile that collapses the band to zero in the middle of the
  range is not the theory transplanted, it is a different rule.
- The claim under test is about the **f -> 0** end -- the snap-to-flat
  consequence. That end is governed by ``f^(2/3)`` alone.

So the arm is the theory's low-exposure asymptotics, honestly labelled, not
the full formula.

=====================================================================
f_ref, p, AND WHY NEITHER IS A FITTED PARAMETER
=====================================================================

``f_ref = 1.0`` (full notional) is a **structural reference, not a fitted
constant**: at f = 1 the band equals v4's own shipped ``deadband = 0.10``
exactly, so the arm coincides with the incumbent at full exposure and departs
from it only in *how the width changes away from there*. It is not tuned. An
``f_ref`` ablation is reported at the end purely as sensitivity **context**,
clearly labelled as such and counted in the configuration total.

``p`` is the mechanism's own index, not a selection grid:

    p = 0      ->  band(f) = 0.10 for all f  ==  kelly_regime_v4, bit-for-bit
    p = 2/3    ->  the literal theoretical value, FROZEN A PRIORI
    p = 1      ->  band proportional to f (a "percentage band"), the natural
                   over-shoot of the same idea

``cmd_causality`` asserts the p=0 identity against ``KellyRegimeV4`` as a
regression check on the faithfulness of the copied loop. D5 requires the
response to be **monotone in p**; a non-monotone response is the signature
that something other than the width profile is moving the numbers.

=====================================================================
WHERE THE BAND IS EVALUATED -- DECIDED A PRIORI, BEFORE ANY RUN
=====================================================================

``band(f)`` needs an ``f``, and there are three candidates: ``desired``,
``pos``, or their midpoint. **The primary is ``desired``.** The reason is the
theory's own: the asymptotic band is a region around the *frictionless
target* ``pi_*``, and ``desired`` is this strategy's frictionless target --
``pos`` is the friction-distorted state the band exists to tolerate. Using
``pos`` would make the width a function of the very quantity the rule is
supposed to be indifferent to, and would destroy the snap-to-flat property
(at ``desired = 0`` with ``pos = 0.4`` the band would be 0.10*0.4^p > 0, so
the position would stall short of flat -- exactly R-64's failure).

``pos`` and midpoint are reported as an **ablation only**, and counted.

=====================================================================
THE CONFOUND THIS BRANCH MUST MEASURE, NOT ARGUE ABOUT
=====================================================================

``band(f) = 0.10 * f^p`` with p > 0 makes the band **narrower everywhere below
f = 1**, and v4's exposure sits below 1 most of the time. A narrower band
trades more. So this arm may simply track the target better and earn more by
paying more -- which is **not a cost mechanism at all**. Two things settle it,
and both are in the output below:

1. **D2**: re-run every decision cell at the 0.40% tier. A cost mechanism's
   advantage must GROW as the fee quadruples (``R.d2_satisfied``). If it
   shrinks or inverts, this arm is NEGATIVE regardless of anything else.
2. **The mean realized band width and the trade count**, per split and per p,
   printed next to v4's constant 0.10 and v4's own trade count, plus a clean
   decomposition on v4's *own* path of the trades this rule would add (band
   narrower at low f) against the trades it would remove (band wider above
   f = 1, where ``scale`` runs to the 2.0 leverage cap).

=====================================================================
WHAT WOULD MAKE THIS FAIL -- NAMED NOW, BEFORE ANY CODE RAN
=====================================================================

1. **It is a turnover confound, not a cost mechanism** (failure mode 3 of the
   shared pre-registration). The most likely outcome: fees rise, growth
   maybe rises with them at 0.10%, and D2 kills it at 0.40%.
2. **The snap-to-flat property is worth nothing.** R-64's diagnosis says the
   residual long is what sank the boundary policy. But v4 *already* reaches
   exactly flat -- when ``desired`` steps from 0.8 to 0.0 the gap is 0.8 > 0.10
   so v4 jumps the whole way. v4 has no residual-long problem to fix. This
   arm's snap-to-flat only differs from v4 in the narrow case where
   ``|desired - pos| <= 0.10`` while ``desired == 0``, i.e. positions already
   under 10% of equity. If that case is rare, the entire "derived
   snap-to-flat" story is a rounding error on BTC and the arm reduces to its
   turnover confound. **This is measured directly** (``cmd_signature``), and
   it is the single most likely way this branch is a null.
3. **Chatter at low exposure.** As f -> 0 the band -> 0, so a target hovering
   near a small positive value re-triggers on arbitrarily small moves. That is
   the same degeneracy R-64 hit at k=1 (a reflecting barrier emitting an order
   every bar), reached from the other side. If the step-count explodes, the
   broker's own ``REBALANCE_DEADBAND = 0.05`` masks it rather than fixing it,
   and the arm is measuring the broker.
4. **Non-monotone in p.** D5. If growth is not monotone in p the named
   mechanism is not what is moving the numbers.

=====================================================================
DATA DISCIPLINE
=====================================================================

**No bar dated 2023-01-01 or later is read anywhere in this file.** BTC comes
from ``R.load_btc_inner()``, which hard-truncates at 2022-12-31. Splits are
``INNER_TRAIN`` (... -> 2020-12-31) and ``INNER_VAL`` (2021-01-01 ->
2022-12-31). The falsification instrument is **ETH-A** (``R.load_eth_a()``,
Bitfinex 2016-03 -> 2019-12), entirely pre-2020, costing zero holdout
consultations (the R-19/R-28 convention). **Holdout consultations added by
this branch: 0.**

The class is deliberately **not** ``@register``-ed (ROUTINE.md step 5), so
``tests/test_causality_strict.py`` does not reach it -- this file therefore
carries its own truncation + tamper causality probe, as the R-64 files do.

=====================================================================
A NOTE ON THE COPIED LOOP
=====================================================================

``KellyRegimeV3.prepare`` is one monolithic method; the position-update loop
is not factored out, so changing the trigger width requires copying the body.
The copy in ``_targets`` below is faithful: the vote block, the vol / slow-vol
block, the ``full``/``steady`` arrays and the hysteresis state machine are
character-for-character v3's (20/40/80 anchors via v4, 1% vote band,
``target_vol=0.55``, ``max_leverage=2.0``, ``vol_span=8*BARS_PER_DAY``,
``anchor_span_days=180``, ``high_in/high_out/low_in/low_out`` =
1.70/1.20/0.55/0.85). The **only** edited line is the trigger width:
``self.deadband`` becomes ``self.deadband * (f/f_ref)**p``. ``p = 0``
therefore reproduces v4 bit-for-bit, and ``cmd_causality`` asserts it.

Usage::

    python experiments/r66_novel_scaled_band.py            # everything
    python experiments/r66_novel_scaled_band.py sweep
    python experiments/r66_novel_scaled_band.py stress
    python experiments/r66_novel_scaled_band.py eth
    python experiments/r66_novel_scaled_band.py diagnostic
    python experiments/r66_novel_scaled_band.py signature
    python experiments/r66_novel_scaled_band.py ablation
    python experiments/r66_novel_scaled_band.py futures
    python experiments/r66_novel_scaled_band.py causality
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
from tradebot.broker import REBALANCE_DEADBAND, MarketSpec  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import prefix_bars  # noqa: E402

OUT_DIR = ROOT / "experiments" / "reports"

#: The mechanism's own index. p=0 IS kelly_regime_v4 (constant band); p=2/3 is
#: the literal theoretical value. NOT a selection grid.
P_GRID = (0.0, 1.0 / 6.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0)

#: The frozen a-priori proposal: the literal exponent from the asymptotics.
P_FROZEN = 2.0 / 3.0

#: Structural, not fitted: at f = 1 the band equals v4's shipped 0.10 exactly.
F_REF = 1.0

#: Where the band is evaluated. Decided a priori (see the module docstring):
#: the theory's band is around the frictionless target, which is `desired`.
AT_PRIMARY = "desired"

SHARPE_NOISE_FLOOR = R.SHARPE_NOISE_FLOOR  # R-20


# ============================================================== the strategy


class ScaledBand(KellyRegimeV4):
    """v4 with one line changed: the no-trade band's half-width becomes a
    function of the exposure level it is a band around.

    ``KellyRegimeV3.prepare``'s body copied faithfully (see the module
    docstring); the only edit is the trigger width::

        v4:   if abs(desired - pos) > self.deadband:        pos = desired
        here: if abs(desired - pos) > band(desired):        pos = desired
              with band(f) = self.deadband * (f / f_ref) ** p

    ``deadband`` is v4's own shipped 0.10, re-used and not re-fitted;
    ``f_ref = 1.0`` is structural (the band equals v4's at full notional).
    ``p = 0`` reproduces v4 exactly; ``p = 2/3`` is the literal
    proportional-cost asymptotic exponent.
    """

    name = "r66_novel_scaled_band"
    warmup = 80 * BARS_PER_DAY + 10  # v4's

    def __init__(self, p: float = P_FROZEN, f_ref: float = F_REF,
                 at: str = AT_PRIMARY, **kwargs) -> None:
        super().__init__(**kwargs)
        if not np.isfinite(p) or p < 0.0:
            raise ValueError(f"p must be finite and >= 0, got {p!r}")
        if not np.isfinite(f_ref) or f_ref <= 0.0:
            raise ValueError(f"f_ref must be > 0, got {f_ref!r}")
        if at not in ("desired", "pos", "mid"):
            raise ValueError(f"at must be desired|pos|mid, got {at!r}")
        self.p = float(p)
        self.f_ref = float(f_ref)
        self.at = at

    # ---- v3's prepare body, with the TRIGGER WIDTH replaced ----------------

    def _targets(self, df: pd.DataFrame, trace: dict | None = None) -> np.ndarray:
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
        p, f_ref, at, deadband = self.p, self.f_ref, self.at, self.deadband
        if trace is not None:
            tr_desired = np.zeros(n)
            tr_band = np.zeros(n)
            tr_fired = np.zeros(n, dtype=bool)
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
            f = desired if at == "desired" else (pos if at == "pos"
                                                 else 0.5 * (desired + pos))
            band = deadband * (f / f_ref) ** p     # p == 0 -> exactly deadband
            fired = abs(desired - pos) > band
            if fired:
                pos = desired
            # ----------------------------------------------------------------
            if trace is not None:
                tr_desired[i] = desired
                tr_band[i] = band
                tr_fired[i] = fired
            target[i] = pos

        if trace is not None:
            trace["desired"] = tr_desired
            trace["band"] = tr_band
            trace["fired"] = tr_fired
            trace["target"] = target
        return target

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["target"] = self._targets(df)
        return df


def arm(p: float = P_FROZEN, f_ref: float = F_REF,
        at: str = AT_PRIMARY) -> ScaledBand:
    """A fresh arm at index ``p``. Every other constant is v4's shipped one."""
    return ScaledBand(p=p, f_ref=f_ref, at=at)


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


def _row(res: dict, p: float, split: str, tag: str = "") -> dict:
    out = dict(res)
    out["p"] = p
    out["split"] = split
    out["tag"] = tag
    out["arm_over_v4_fees"] = (res["arm_fees"] / res["v4_fees"]
                               if res["v4_fees"] > 0 else float("nan"))
    return out


def _print_header(text: str) -> None:
    print("\n" + "=" * 118)
    print(text)
    print("=" * 118)


CELL_FMT = ("{p:>6} {split:<12} {mkt:<11} fee={fee:.4f} "
            "arm=${af:>11,.0f} v4=${vf:>11,.0f} "
            "Sh {ash:>5.2f}/{vsh:<5.2f} DD {add:>5.1f}/{vdd:<5.1f} "
            "tr {atr:>5}/{vtr:<5} fees ${afe:>8,.0f}/${vfe:<8,.0f} "
            "c {ac:.3f}/{vc:.3f} dlog {dl:+7.4f} [{lo:+.3f},{hi:+.3f}]{void}")


def show(row: dict) -> None:
    print(CELL_FMT.format(
        p=f"{row['p']:.3f}", split=row["split"], mkt=row["market"],
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
    """(1) The p-sweep on both inner splits, spot @0.10%, arm vs v4."""
    _print_header("(1) p-SWEEP -- spot @0.10%, arm vs kelly_regime_v4, inner splits")
    out = []
    for split, window in SPLITS:
        for p in P_GRID:
            row = _row(R.compare(arm(p), df, window, R.spot(R.FEE_BASE),
                                 label=f"scaled band p={p:.3f}"), p, split)
            show(row)
            out.append(row)
            rows.append(row)
        print("-" * 118)
    return out


def cmd_stress(df: pd.DataFrame, rows: list) -> list[dict]:
    """(2) The same at 0.40% -- D2, the discriminating cost-mechanism test."""
    _print_header("(2) p-SWEEP -- spot @0.40% (FEE_STRESS): D2, the discriminating test")
    out = []
    for split, window in SPLITS:
        for p in P_GRID:
            row = _row(R.compare(arm(p), df, window, R.spot(R.FEE_STRESS),
                                 label=f"scaled band p={p:.3f}"), p, split)
            show(row)
            out.append(row)
            rows.append(row)
        print("-" * 118)
    return out


def cmd_eth(rows: list) -> list[dict]:
    """(3) D3 falsification on ETH-A (Bitfinex 2016-03 -> 2019-12, +0 holdout)."""
    _print_header("(3) D3 FALSIFICATION -- ETH-A, Bitfinex 2016-03 -> 2019-12, spot")
    print("  R-64's *destination*-change arm scored -0.430 [-0.791, -0.101] on this")
    print("  same window at its own frozen setting. This is a width change, not a")
    print("  destination change, so that number is context, not a bar.\n")
    eth = R.load_eth_a()
    print(f"ETH-A: {len(eth):,} bars, {eth.index[0]} -> {eth.index[-1]}\n")
    out = []
    for p in P_GRID:
        row = _row(R.compare(arm(p), eth, (None, None), R.spot(R.FEE_BASE),
                             label=f"scaled band p={p:.3f}"), p, "eth-a")
        show(row)
        out.append(row)
        rows.append(row)
    print("-" * 118)
    for p in (P_FROZEN,):
        row = _row(R.compare(arm(p), eth, (None, None), R.spot(R.FEE_STRESS),
                             label=f"scaled band p={p:.3f}"), p, "eth-a")
        show(row)
        out.append(row)
        rows.append(row)
    return out


def cmd_ablation(df: pd.DataFrame, rows: list) -> list[dict]:
    """(4) CONTEXT ONLY: where the band is evaluated, and f_ref sensitivity.

    Both are ablations, both are counted in the configuration total, and
    NEITHER can change the frozen proposal (``at='desired'``, ``f_ref=1.0``,
    ``p=2/3``), which was fixed before any number below was produced.
    """
    _print_header("(4) ABLATIONS (CONTEXT ONLY -- the frozen config is at=desired, f_ref=1.0)")
    out = []
    print("  (4a) band evaluated at pos / midpoint instead of desired, p=2/3\n")
    for split, window in SPLITS:
        for at in ("pos", "mid"):
            row = _row(R.compare(arm(P_FROZEN, at=at), df, window,
                                 R.spot(R.FEE_BASE),
                                 label=f"p={P_FROZEN:.3f} at={at}"),
                       P_FROZEN, split, tag=f"at={at}")
            show(row)
            out.append(row)
            rows.append(row)
    print("\n  (4b) f_ref sensitivity at p=2/3, at=desired "
          "(f_ref=1.0 is the structural choice and is NOT tuned)\n")
    for split, window in SPLITS:
        for f_ref in (0.5, 2.0):
            row = _row(R.compare(arm(P_FROZEN, f_ref=f_ref), df, window,
                                 R.spot(R.FEE_BASE),
                                 label=f"p={P_FROZEN:.3f} f_ref={f_ref}"),
                       P_FROZEN, split, tag=f"f_ref={f_ref}")
            show(row)
            out.append(row)
            rows.append(row)
    return out


def fill_through(strategy, df: pd.DataFrame, window, market: MarketSpec) -> dict:
    """How many of the strategy's intended position changes became fills?

    ``prepare`` decides the whole target path open-loop; ``on_bar`` emits an
    order whenever the path moves; the broker then drops any same-sign
    adjustment worth less than ``REBALANCE_DEADBAND`` (5%) of MAX notional --
    equity on spot, ``equity x leverage`` on futures. On 5x futures that is
    25% of equity, larger than most steps either arm asks for (backlog B-30).
    """
    frame, prefix = window_frame(df, window, strategy.warmup)
    tgt = (strategy._targets(frame) if isinstance(strategy, ScaledBand)
           else strategy.prepare(frame.copy())["target"].to_numpy(dtype=float))
    n = len(tgt)
    lo = max(prefix, strategy.warmup)
    steps = np.abs(np.diff(tgt))[max(lo - 1, 0): n - 2] if n > 2 else np.array([])
    intended = steps > 1e-9
    res, _ = R.measure(strategy, df, window, market)
    threshold = REBALANCE_DEADBAND * market.leverage
    small = intended & (steps < threshold)
    return dict(
        intended=int(intended.sum()),
        fills=len(res.fills),
        below_broker_band=int(small.sum()),
        broker_step_threshold=float(threshold),
        median_step=float(np.median(steps[intended])) if intended.any() else float("nan"),
        fill_ratio=float(len(res.fills) / intended.sum()) if intended.sum() else float("nan"),
    )


def cmd_futures(df: pd.DataFrame, rows: list) -> list[dict]:
    """(5) OPTIONAL CONTEXT: futures 5x, always with fill_through (B-30).

    Spot is the decision market. No futures number here is quoted without the
    fill-through table below it.
    """
    _print_header("(5) FUTURES 5x -- OPTIONAL CONTEXT ONLY, never without fill_through")
    print(f"broker.REBALANCE_DEADBAND = {REBALANCE_DEADBAND}; on 5x futures a "
          f"same-sign step below {REBALANCE_DEADBAND * 5:.0%} of EQUITY is "
          f"silently dropped (B-30).\n")
    out = []
    fut = R.futures()
    for split, window in SPLITS:
        for p in (P_FROZEN,):
            row = _row(R.compare(arm(p), df, window, fut,
                                 label=f"scaled band p={p:.3f}"), p, split)
            show(row)
            out.append(row)
            rows.append(row)

    print("\n  fill-through: intended target changes vs fills actually executed\n")
    print(f"  {'strategy':<26} {'split':<12} {'market':<11} {'intended':>9} "
          f"{'fills':>7} {'ratio':>7} {'<band':>8} {'medstep':>8}")
    ft_rows = []
    for market in (R.spot(R.FEE_BASE), fut):
        for split, window in SPLITS:
            for label, s in (("kelly_regime_v4", R.v4()),
                             (f"scaled p={P_FROZEN:.3f}", arm(P_FROZEN))):
                d = fill_through(s, df, window, market)
                print(f"  {label:<26} {split:<12} {market.name:<11} "
                      f"{d['intended']:>9} {d['fills']:>7} {d['fill_ratio']:>7.2f} "
                      f"{d['below_broker_band']:>8} {d['median_step']:>8.3f}")
                ft_rows.append(dict(strategy=label, split=split,
                                    market=market.name, **d))
    pd.DataFrame(ft_rows).to_csv(OUT_DIR / "r66_novel_fillthrough.csv", index=False)
    return out


# --------------------------------------------------- the mechanism's signature


def _signature_one(frame: pd.DataFrame, prefix: int, label: str) -> list[dict]:
    """Free (no backtest, 0 configs) measurement of what the rule actually did.

    Everything reported here is read off the target path, not assumed:

    - how often v4's ``desired`` is EXACTLY 0.0, and what the arm's position
      is on those bars (must be exactly 0 for any p > 0);
    - the mean band width the rule actually used, vs v4's constant 0.10;
    - the distribution of ``desired`` -- where on the f^p curve this strategy
      actually lives;
    - the trade decomposition, computed on v4's OWN path so it is a clean
      counterfactual: bars where the arm's rule would fire and v4's would not
      (trades ADDED by the narrower band at low f), and the reverse (trades
      REMOVED by the wider band above f = 1, where ``scale`` runs to the 2.0
      leverage cap).
    """
    lo = max(prefix, ScaledBand.warmup)
    # v4's own path (p=0 IS v4), traced once.
    v4_tr: dict = {}
    ScaledBand(p=0.0)._targets(frame, trace=v4_tr)
    v4_des = v4_tr["desired"][lo:]
    v4_tgt = v4_tr["target"][lo:]
    v4_fired = v4_tr["fired"][lo:]
    # |desired - pos| on v4's path, needed for the counterfactual.
    v4_pos_prev = np.concatenate(([v4_tr["target"][lo - 1] if lo else 0.0],
                                  v4_tr["target"][lo:-1]))
    gap = np.abs(v4_des - v4_pos_prev)

    zero_bars = v4_des == 0.0
    out = []
    print(f"\n  {label}:  {len(v4_des):,} traded bars  "
          f"(v4 desired == 0.0 exactly on {int(zero_bars.sum()):,} = "
          f"{zero_bars.mean():.1%} of them)")
    # THE CONTROL. "The vanishing band delivers exact flat" is only a claim if
    # the incumbent sometimes does not. v4's own residual on a zero target is
    # the baseline the arm's zero has to be measured against.
    v4_resid = v4_tgt[zero_bars]
    v4_nonflat = int((v4_resid > 0.0).sum())
    print(f"    CONTROL -- kelly_regime_v4's OWN residual on those bars: "
          f"non-flat on {v4_nonflat:,} "
          f"({(v4_nonflat / max(int(zero_bars.sum()), 1)):.1%} of zero-target bars), "
          f"mean {v4_resid[v4_resid > 0].mean() if v4_nonflat else 0.0:.4f}, "
          f"max {v4_resid.max() if len(v4_resid) else 0.0:.4f}")
    q = np.percentile(v4_des, [10, 25, 50, 75, 90, 99])
    print(f"    distribution of desired (v4 path): mean {v4_des.mean():.3f}  "
          f"p10 {q[0]:.3f}  p25 {q[1]:.3f}  p50 {q[2]:.3f}  p75 {q[3]:.3f}  "
          f"p90 {q[4]:.3f}  p99 {q[5]:.3f}  max {v4_des.max():.3f}")
    print(f"    fraction of bars with desired < 1.0 (band NARROWER than v4's): "
          f"{(v4_des < 1.0).mean():.1%};  desired > 1.0 (band WIDER): "
          f"{(v4_des > 1.0).mean():.1%}")
    print()
    print(f"    {'p':>6} {'mean band':>10} {'band@fire':>10} {'steps':>7} "
            f"{'turnover':>9} {'pos>0 on desired==0':>21} "
            f"{'cf +trades':>11} {'cf -trades':>11}")
    for p in P_GRID:
        tr: dict = {}
        s = ScaledBand(p=p)
        s._targets(frame, trace=tr)
        band = tr["band"][lo:]
        fired = tr["fired"][lo:]
        tgt = tr["target"][lo:]
        # the snap-to-flat check: arm's position on bars where the target is 0
        resid = tgt[zero_bars]
        n_nonflat = int((np.abs(resid) > 0.0).sum())
        steps = np.abs(np.diff(tr["target"][max(lo - 1, 0):]))
        steps = steps[steps > 1e-9]
        # counterfactual on v4's own path: would this width fire where v4's
        # 0.10 did not, and vice versa?
        cf_band = 0.10 * (v4_des / F_REF) ** p
        cf_fire = gap > cf_band
        added = int((cf_fire & ~v4_fired).sum())
        removed = int((~cf_fire & v4_fired).sum())
        row = dict(label=label, p=p, mean_band=float(band.mean()),
                   band_at_fire=float(band[fired].mean()) if fired.any() else float("nan"),
                   steps=int(len(steps)), turnover=float(steps.sum()),
                   nonflat_on_zero=n_nonflat,
                   max_resid_on_zero=float(np.abs(resid).max()) if len(resid) else 0.0,
                   cf_trades_added=added, cf_trades_removed=removed,
                   frac_desired_below_1=float((v4_des < 1.0).mean()),
                   mean_desired=float(v4_des.mean()),
                   bars=int(len(v4_des)),
                   v4_zero_bars=int(zero_bars.sum()))
        out.append(row)
        print(f"    {p:>6.3f} {row['mean_band']:>10.4f} "
              f"{row['band_at_fire']:>10.4f} {row['steps']:>7} "
              f"{row['turnover']:>9.2f} {n_nonflat:>21,} "
              f"{added:>11,} {removed:>11,}")
    print(f"    (v4's band is the constant 0.1000 by construction; 'cf +/-trades' "
          f"is the counterfactual on v4's OWN path)")
    return out


def cmd_signature(df: pd.DataFrame) -> pd.DataFrame:
    """(6) The mechanism's own signature, measured rather than assumed."""
    _print_header("(6) MECHANISM SIGNATURE -- band widths, snap-to-flat, trade decomposition")
    recs: list[dict] = []
    for split, window in SPLITS:
        frame, prefix = window_frame(df, window, ScaledBand.warmup)
        recs += _signature_one(frame, prefix, split)
    eth = R.load_eth_a()
    frame, prefix = window_frame(eth, (None, None), ScaledBand.warmup)
    recs += _signature_one(frame, prefix, "eth-a")
    out = pd.DataFrame(recs)
    out.to_csv(OUT_DIR / "r66_novel_signature.csv", index=False)
    return out


def cmd_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """(7) Failure mode 2, measured: is there any snap-to-flat work to do?

    v4 ALREADY reaches exactly flat whenever ``|desired - pos| > 0.10`` and
    ``desired == 0`` -- which is most bear-regime exits, since the position it
    is leaving is usually well above 10% of equity. The arm's derived
    snap-to-flat therefore only bites in the narrow case ``desired == 0`` and
    ``0 < |pos| <= 0.10``. If that case is rare, the "derived snap-to-flat"
    story is a rounding error on this data and the arm reduces to its turnover
    confound. This counts it directly, on v4's own path.
    """
    _print_header("(7) IS THERE ANY SNAP-TO-FLAT WORK TO DO? (failure mode 2, on v4's path)")
    recs = []
    frames = [(s, *window_frame(df, w, ScaledBand.warmup)) for s, w in SPLITS]
    eth = R.load_eth_a()
    frames.append(("eth-a", *window_frame(eth, (None, None), ScaledBand.warmup)))
    for label, frame, prefix in frames:
        lo = max(prefix, ScaledBand.warmup)
        tr: dict = {}
        ScaledBand(p=0.0)._targets(frame, trace=tr)
        des = tr["desired"][lo:]
        tgt = tr["target"][lo:]
        zero = des == 0.0
        stranded = zero & (tgt > 0.0)
        n_z = int(zero.sum())
        n_s = int(stranded.sum())
        row = dict(split=label, bars=int(len(des)), zero_target_bars=n_z,
                   v4_stranded_bars=n_s,
                   v4_stranded_frac_of_zero=(n_s / n_z if n_z else float("nan")),
                   v4_max_stranded=float(tgt[stranded].max()) if n_s else 0.0,
                   v4_mean_stranded=float(tgt[stranded].mean()) if n_s else 0.0)
        recs.append(row)
        print(f"  {label:<12} bars {len(des):>9,}   desired==0 on {n_z:>9,} "
              f"({zero.mean():>6.1%})   of which v4 still holds pos>0 on "
              f"{n_s:>8,} ({row['v4_stranded_frac_of_zero'] if n_z else float('nan'):>6.1%})"
              f"   mean residual {row['v4_mean_stranded']:.4f} "
              f"max {row['v4_max_stranded']:.4f}")
    out = pd.DataFrame(recs)
    out.to_csv(OUT_DIR / "r66_novel_snapwork.csv", index=False)
    print("\n  Reading: v4's residual on a zero target is bounded by its own 0.10 band.")
    print("  Whatever the arm's snap-to-flat is worth, it is worth AT MOST this much")
    print("  exposure over this many bars -- and if 'v4 stranded bars' is small, the")
    print("  derived snap-to-flat is a rounding error and this arm is its confound.")
    return out


def cmd_causality(df: pd.DataFrame) -> bool:
    """(8) Truncation + tamper probe, plus the p=0 == v4 identity check.

    A. **p=0 identity.** ``ScaledBand(p=0)``'s target path must be
       bit-identical to ``KellyRegimeV4``'s on the same frame -- a regression
       check on the faithfulness of the copied loop.
    B. **Truncation.** Prepare on a frame truncated at ``m``; the targets over
       the shared prefix must be bit-identical to the full-frame run's. A
       statistic computed over the whole series (scaler, quantile, mean, std)
       would break this.
    C. **Tamper.** Corrupt every bar after ``m`` (x3 up and /3 down); no target
       at index < m may change under either tamper, and the two must agree.
    D. **Snap-to-flat is structural.** On the same frame, at every p > 0, the
       target must be exactly 0.0 on every bar where ``desired`` is exactly
       0.0 -- the derived consequence, asserted rather than hoped for.
    """
    _print_header("(8) CAUSALITY -- truncation / tamper probe (the class is unregistered)")
    frame = df.iloc[-400_000:].copy()
    ok = True

    a = ScaledBand(p=0.0)._targets(frame)
    b = KellyRegimeV4().prepare(frame.copy())["target"].to_numpy(dtype=float)
    ident = bool(np.array_equal(a, b))
    ok &= ident
    print(f"  A. p=0 target path == kelly_regime_v4 bit-for-bit : "
          f"{'PASS' if ident else 'FAIL'}")

    s = arm(P_FROZEN)
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

    for p in (1.0 / 3.0, P_FROZEN, 1.0):
        tr: dict = {}
        ScaledBand(p=p)._targets(frame, trace=tr)
        z = tr["desired"] == 0.0
        flat = bool(np.all(tr["target"][z] == 0.0)) if z.any() else True
        ok &= flat
        print(f"  D. p={p:.3f}: target exactly 0 on all {int(z.sum()):>7,} "
              f"zero-desired bars : {'PASS' if flat else 'FAIL'}")

    print(f"\n  CAUSALITY: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


# ================================================================= verdicts


def report_shape(rows: list[dict]) -> dict:
    """D5: is the response monotone in p -- the mechanism's own signature?"""
    _print_header("D5 -- SHAPE IN p (the mechanism's index): monotone, or something else?")
    verdicts = {}
    for split, _ in SPLITS + (("eth-a", None),):
        for fee in (R.FEE_BASE, R.FEE_STRESS):
            sel = sorted([r for r in rows if r["split"] == split
                          and r["market"] == "spot" and not r["tag"]
                          and abs(r["fee"] - fee) < 1e-12],
                         key=lambda r: r["p"])
            if len(sel) < 3:
                continue
            ps = [r["p"] for r in sel]
            dlog = [r["d_logret"] for r in sel]
            fees = [r["arm_fees"] for r in sel]
            sharpe = [r["arm_sharpe"] for r in sel]
            trades = [r["arm_trades"] for r in sel]
            print(f"\n  {split}  spot @{fee:.2%}")
            print("    p          : " + "  ".join(f"{v:>8.3f}" for v in ps))
            print("    d_logret   : " + "  ".join(f"{v:>+8.4f}" for v in dlog))
            print("    arm fees $ : " + "  ".join(f"{v:>8,.0f}" for v in fees))
            print("    arm trades : " + "  ".join(f"{v:>8d}" for v in trades))
            print("    arm Sharpe : " + "  ".join(f"{v:>8.3f}" for v in sharpe))
            mono_d = R.monotone(dlog)
            print(f"    monotone in p?  d_logret {'YES' if mono_d else 'NO'}   "
                  f"fees {'YES' if R.monotone(fees) else 'NO'}   "
                  f"Sharpe {'YES' if R.monotone(sharpe) else 'NO'}")
            spread = max(sharpe) - min(sharpe)
            print(f"    Sharpe spread over the p grid: {spread:.3f} vs the "
                  f"+/-{SHARPE_NOISE_FLOOR} noise floor -> "
                  f"{'INSIDE (indistinguishable)' if spread <= 2 * SHARPE_NOISE_FLOOR else 'OUTSIDE'}")
            verdicts[(split, fee)] = mono_d
    return verdicts


def report_d2(rows: list[dict]) -> dict:
    """D2: the advantage must GROW when the fee quadruples."""
    _print_header("D2 -- THE DISCRIMINATING TEST: does the advantage GROW at 0.40%?")
    print("  REQUIRE d_logret(0.40%) > d_logret(0.10%). A cost mechanism's edge grows")
    print("  with the fee; a tracking-quality change's edge shrinks or inverts.\n")
    print(f"  {'split':<12} {'p':>6} {'dlog@0.10%':>12} {'dlog@0.40%':>12} "
          f"{'delta':>10} {'grows?':>8}")
    out = {}
    for split in ("inner-train", "inner-val", "eth-a"):
        for p in P_GRID:
            base = [r for r in rows if r["split"] == split and r["market"] == "spot"
                    and not r["tag"] and r["p"] == p
                    and abs(r["fee"] - R.FEE_BASE) < 1e-12]
            stress = [r for r in rows if r["split"] == split and r["market"] == "spot"
                      and not r["tag"] and r["p"] == p
                      and abs(r["fee"] - R.FEE_STRESS) < 1e-12]
            if not base or not stress:
                continue
            b, s = base[0]["d_logret"], stress[0]["d_logret"]
            ok = R.d2_satisfied(b, s)
            out[(split, p)] = ok
            print(f"  {split:<12} {p:>6.3f} {b:>+12.4f} {s:>+12.4f} "
                  f"{s - b:>+10.4f} {'YES' if ok else 'no':>8}")
    return out


def report_d4(rows: list[dict]) -> None:
    _print_header("D4 -- TURNOVER SANITY (a no-trade-region device should trade LESS)")
    print(f"  {'split':<12} {'market':<11} {'fee':>7} {'p':>6} {'tag':<12} "
          f"{'arm trades':>11} {'v4 trades':>10} {'arm fees':>11} "
          f"{'v4 fees':>11} {'fell?':>7}")
    for r in rows:
        print(f"  {r['split']:<12} {r['market']:<11} {r['fee']:>7.4f} "
              f"{r['p']:>6.3f} {r['tag']:<12} {r['arm_trades']:>11} "
              f"{r['v4_trades']:>10} {r['arm_fees']:>11,.0f} "
              f"{r['v4_fees']:>11,.0f} "
              f"{'YES' if r['arm_fees'] < r['v4_fees'] else 'NO':>7}")


def report_d0(rows: list[dict], df: pd.DataFrame) -> list[dict]:
    _print_header("D0 -- RISK-MATCH GATE (mean notional). A void cell is not a growth claim.")
    print(f"  {'split':<12} {'market':<11} {'fee':>7} {'p':>6} {'tag':<12} "
          f"{'c_arm':>8} {'c_v4':>8} {'mismatch':>9} {'D0':>7}")
    for r in rows:
        print(f"  {r['split']:<12} {r['market']:<11} {r['fee']:>7.4f} "
              f"{r['p']:>6.3f} {r['tag']:<12} {r['c_arm']:>8.4f} "
              f"{r['c_v4']:>8.4f} {r['risk_mismatch']:>8.1%} "
              f"{'VOID' if r['d0_void'] else 'ok':>7}")
    voids = [r for r in rows if r["d0_void"] and r["p"] == P_FROZEN
             and r["market"] == "spot" and not r["tag"]
             and abs(r["fee"] - R.FEE_BASE) < 1e-12
             and r["split"] in ("inner-train", "inner-val")]
    out = []
    if voids:
        print("\n  Frozen-p cells are D0-VOID: re-reporting them against "
              "ConstantExposureHold(c_arm)\n")
        for r in voids:
            window = dict(SPLITS)[r["split"]]
            cell = R.matched_hold_cell(arm(P_FROZEN), df, window,
                                       R.spot(R.FEE_BASE),
                                       label=f"{r['split']} matched-hold")
            print(f"  {cell['label']:<28} c={cell['c']:.4f}  "
                  f"arm=${cell['arm_final']:>11,.0f} hold=${cell['hold_final']:>11,.0f} "
                  f"dlog={cell['d_logret']:+.4f} "
                  f"[{cell['d_logret_lo']:+.3f}, {cell['d_logret_hi']:+.3f}]")
            out.append(cell)
    else:
        print("\n  No frozen-p spot decision cell is D0-void.")
    return out


# ====================================================================== main


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "all"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("R-66 NOVEL: the no-trade band's WIDTH PROFILE -- band(f) = deadband * "
          "(f/f_ref)**p  (Gerhold et al. 2014)")
    print("=" * 118)
    print(f"arm         : ScaledBand(p=...), deadband = v4's shipped "
          f"{ScaledBand().deadband}, f_ref = {F_REF} (structural, not tuned)")
    print(f"frozen p    : {P_FROZEN:.6f} = 2/3, the literal asymptotic exponent "
          f"(a priori, NOT peak-picked)")
    print(f"p grid      : {tuple(round(p, 4) for p in P_GRID)}   "
          f"(p=0.0 IS kelly_regime_v4)")
    print(f"band at     : {AT_PRIMARY} (the frictionless target -- decided a "
          f"priori; pos/mid are ablations)")
    print("splits      : INNER_TRAIN (-> 2020-12-31), INNER_VAL "
          "(2021-01-01 -> 2022-12-31)")
    print("holdout     : NOT READ. BTC frame truncated at 2022-12-31 on load.")

    df = R.load_btc_inner()
    print(f"BTC (inner) : {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")

    rows: list[dict] = []
    causal = None

    if cmd in ("all", "causality"):
        causal = cmd_causality(df)
    if cmd in ("all", "diagnostic"):
        cmd_diagnostic(df)
    if cmd in ("all", "signature", "diagnostic"):
        cmd_signature(df)
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

    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "r66_novel_cells.csv", index=False)
        print(f"\nSaved cells -> {OUT_DIR / 'r66_novel_cells.csv'}")

    if cmd == "all":
        report_shape(rows)
        d2 = report_d2(rows)
        report_d4(rows)
        report_d0(rows, df)

        _print_header("D3 (ETH-A falsification): sign of arm-minus-v4 log growth")
        for r in rows:
            if r["split"] == "eth-a":
                print(f"  p={r['p']:.3f} fee={r['fee']:.4f}  d_logret="
                      f"{r['d_logret']:+.4f} "
                      f"[{r['d_logret_lo']:+.3f}, {r['d_logret_hi']:+.3f}]  "
                      f"sign {'POSITIVE' if r['d_logret'] > 0 else 'NEGATIVE'}")

        # ------------------------------------------------ the frozen verdict
        _print_header("R.promotion(...) ON THE FROZEN CONFIG (p=2/3, at=desired, f_ref=1.0)")
        print("  NOTE: D1 is defined on the HOLDOUT, which this branch does not read.")
        print("  The verdict below is evaluated with inner-validation standing in for")
        print("  D1, which can only ever REJECT -- it cannot promote. An arm that")
        print("  fails a gate here is NEGATIVE and never reads the holdout at all.\n")

        def pick(split, fee, p=P_FROZEN):
            m = [r for r in rows if r["split"] == split and r["market"] == "spot"
                 and not r["tag"] and r["p"] == p and abs(r["fee"] - fee) < 1e-12]
            return m[0] if m else None

        iv = pick("inner-val", R.FEE_BASE)
        ivs = pick("inner-val", R.FEE_STRESS)
        ea = pick("eth-a", R.FEE_BASE)
        it = pick("inner-train", R.FEE_BASE)
        if iv and ivs and ea and it:
            mono_ok = R.monotone(
                [r["d_logret"] for r in sorted(
                    [x for x in rows if x["split"] == "inner-val"
                     and x["market"] == "spot" and not x["tag"]
                     and abs(x["fee"] - R.FEE_BASE) < 1e-12],
                    key=lambda x: x["p"])])
            turnover_fell = iv["arm_fees"] < iv["v4_fees"]
            verdict = R.promotion(
                d0_void=iv["d0_void"],
                d1_point=iv["d_logret"],
                d1_excludes_zero=iv["d_logret_excludes_zero"],
                d2_ok=R.d2_satisfied(iv["d_logret"], ivs["d_logret"]),
                d3_eth_a=ea["d_logret"],
                turnover_fell=turnover_fell,
                plateau=mono_ok,
                beats_hold_oos=False,  # not evaluable: the holdout is not read
            )
            print(f"  D0 risk mismatch (inner-val, p=2/3) : "
                  f"{iv['risk_mismatch']:.1%} -> "
                  f"{'VOID' if iv['d0_void'] else 'ok'}")
            print(f"  D1 stand-in (inner-val d_logret)    : {iv['d_logret']:+.4f} "
                  f"[{iv['d_logret_lo']:+.3f}, {iv['d_logret_hi']:+.3f}] -> "
                  f"{'excludes zero' if iv['d_logret_excludes_zero'] else 'INCLUDES ZERO'}")
            print(f"  D2 advantage grows @0.40%           : "
                  f"{iv['d_logret']:+.4f} -> {ivs['d_logret']:+.4f} -> "
                  f"{'YES' if R.d2_satisfied(iv['d_logret'], ivs['d_logret']) else 'NO'}")
            print(f"  D3 ETH-A sign                       : {ea['d_logret']:+.4f} "
                  f"[{ea['d_logret_lo']:+.3f}, {ea['d_logret_hi']:+.3f}] -> "
                  f"{'POSITIVE' if ea['d_logret'] > 0 else 'NEGATIVE'}")
            print(f"  D4 turnover fell (inner-val fees)   : "
                  f"${iv['arm_fees']:,.0f} vs ${iv['v4_fees']:,.0f} -> "
                  f"{'YES' if turnover_fell else 'NO'}")
            print(f"  D5 monotone in p (inner-val @0.10%) : "
                  f"{'YES' if mono_ok else 'NO'}")
            print(f"  buy_and_hold OOS                    : NOT EVALUATED "
                  f"(holdout not read; passed as False)")
            print(f"\n  R.promotion(...) = {verdict}")

        _print_header("CONFIGURATION PROPOSED (frozen a priori, unchanged by anything above)")
        print("  experiments.r66_novel_scaled_band.ScaledBand(p=2/3, f_ref=1.0, "
              "at='desired')")
        print("  i.e. ScaledBand(p=0.6667, f_ref=1.0, at='desired', "
              "horizons=(20, 40, 80),")
        print("       band=0.01, target_vol=0.55, max_leverage=2.0, vol_span=2304,")
        print("       deadband=0.10, vote_gamma=1.0, anchor_span_days=180,")
        print("       high_in=1.70, high_out=1.20, low_in=0.55, low_out=0.85)")
        print("  -- every argument except p/f_ref/at is kelly_regime_v4's shipped "
              "default.")

    print()
    print("=" * 118)
    if causal is not None:
        print(f"Causality probe            : {'PASS' if causal else 'FAIL'}")
    print(f"Configurations evaluated   : {R.configs_evaluated()}")
    print("Holdout consultations added: 0 (BTC truncated at 2022-12-31; "
          "ETH-A ends 2019-12)")
    print("=" * 118)


if __name__ == "__main__":
    main()
