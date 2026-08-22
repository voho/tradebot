#!/usr/bin/env python
"""R-93 CONSERVATIVE branch: sweep, gate and score `GZScaledKellyV4` --
`kelly_regime_v4`'s vote (`frac`) times a Grossman & Zhou (1993)
drawdown-constrained `scale`, in place of v4's own conditional-vol-target
`scale` -- against `kelly_regime_v4` on inner-train, inner-validation and
ETH. The direction, citation, mechanism, disclosed simplification and the
not-a-duplicate reasoning against R-38/R-46/R-59/R-60/R-62/R-87/R-45 all
live in `experiments/r93_shared.py`'s module docstring (read there first);
this file does not repeat that reasoning and does not edit that module.

MECHANISM (one sentence, unchanged from the shared pre-registration):

    scale_GZ(t) = max_leverage * clip(1 - drawdown_t / alpha, 0, 1), where
    drawdown_t is the STRATEGY'S OWN realized drawdown from its running
    equity peak -- full max_leverage at a fresh peak, linearly de-levered
    to zero once the strategy's own drawdown reaches alpha -- replacing
    v4's conditional-vol-target scale while leaving v4's 3-anchor `frac`
    vote untouched.

======================================================================
HEADLINE RESULT, stated before the detail: NEGATIVE. Zero of 30 swept
cells ever risk-match (0/30). The single alpha that nominally clears the
pre-registered Sharpe-leg threshold on inner-validation (alpha=0.15) does
so only by collapsing to near-zero exposure after one early drawdown --
an R-33 "holding less" arithmetic artifact, not a sizing edge -- and its
own pre-registered ETH falsification test FAILS outright (the sign
flips negative on both ETH markets). No alpha in the sweep survives both
the promotion bar and the falsification test.
======================================================================

GATES (run in the order below; A-gates are structural/sanity, B-gates are
the pre-registered promotion bar from the task brief, frozen before this
script's first real-data run):

  A1  structural sanity (no market data): alpha->0 forces scale_gz==0
      wherever drawdown>0; drawdown==0 bars always score exactly
      max_leverage. Checked directly against `running_drawdown`/
      `scale_gz` on a synthetic path, for every swept alpha plus an
      alpha->0 limit case.
  A2  non-inertness: R^2 of the candidate's realized exposure path against
      v4's own realized exposure path, same slice/market. R^2 >= 0.98
      would mean "the improvement" is indistinguishable from v4 itself.
  A3  causality: `causal_truncation_probe` run against a REAL equity curve
      from one of this branch's own backtests (not just r93_shared's own
      synthetic self-test).
  B1  Sharpe leg: dSharpe > +0.2 on inner-validation on BOTH markets
      (R-20's noise floor), OR
  B2  drawdown leg: a genuinely risk-matched (exposure_ratio AND vol_ratio
      both in [0.9, 1.1]) drawdown improvement on inner-validation.
  B3  plateau not peak: the winning alpha's neighbours in the sweep must
      not look wildly different.
  B4  falsification (pre-registered, not changed after seeing results):
      ETH must show the SAME SIGN of dSharpe improvement as BTC
      inner-validation, on both markets.
  B5  cost robustness: sign of the BTC inner-validation SPOT result must
      survive a 0.40% taker fee tier.

Promotion requires A1-A3 all PASS and B1-B5 all hold. Default is REJECT.

----------------------------------------------------------------------
RESULTS -- A-gates
----------------------------------------------------------------------

A1 (structural sanity): PASS for all five swept alphas (0.15, 0.20, 0.30,
0.40, 0.50) plus an alpha=1e-6 limit case, on a synthetic peak/drawdown/
recovery equity path (1,000 bars): at every alpha, bars sitting exactly at
a fresh peak (drawdown==0) score `scale_gz == max_leverage` to within
1e-12, and at alpha->0, `scale_gz == 0` at every bar with drawdown>0 and
`scale_gz == max_leverage` at every bar with drawdown==0. This is exactly
the shared module's own self-test, re-run here per-alpha rather than
trusted from a single import-time check.

----------------------------------------------------------------------
RESULTS -- full sweep: 5 alphas x 2 markets x 3 slices = 30 cells
----------------------------------------------------------------------

max_leverage fixed at v4's own 2.0 for a fair comparison, per the task's
sweep spec. `RM` = risk_matched (exposure_ratio AND vol_ratio both in
[0.9, 1.1]). `dSh`/`dDD` are candidate-minus-v4. `dlogG [lo,hi]` is the
paired stationary-block-bootstrap 95% interval on total log-growth
difference; `excl0` = interval excludes zero.

label            slice            market           cand$      ctrl$    dSh     dDD  expR  volR  RM   dlogG      [lo,     hi] excl0
------------------------------------------------------------------------------------------------------------------------------------
gz_alpha_0.15    inner_train      spot             1,702     18,477  -1.07   -27.2  0.15  0.31   n  -2.385   -4.309,  -0.671   YES
gz_alpha_0.15    inner_train      futures_5x       1,778     30,344  -1.40   -20.2  0.11  0.34   n  -2.837   -4.963,  -0.956   YES
gz_alpha_0.15    inner_val        spot             1,102        998  +0.25   -17.6  0.11  0.57   n  +0.089   -0.603,  +0.694    no
gz_alpha_0.15    inner_val        futures_5x       1,125      1,064  +0.23   -14.5  0.14  0.73   n  +0.042   -0.705,  +0.711    no
gz_alpha_0.15    eth_replication  spot             1,286      5,482  -0.92   -19.7  0.11  0.28   n  -1.450   -3.256,  +0.103    no
gz_alpha_0.15    eth_replication  futures_5x       1,528      4,263  -0.43   -15.0  0.06  0.35   n  -1.026   -2.811,  +0.408    no
gz_alpha_0.2     inner_train      spot             1,636     18,477  -1.19   -23.2  0.15  0.33   n  -2.424   -4.395,  -0.675   YES
gz_alpha_0.2     inner_train      futures_5x       4,398     30,344  -1.06    +1.9  0.48  0.71   n  -1.931   -3.496,  -0.572   YES
gz_alpha_0.2     inner_val        spot             1,101        998  +0.24   -12.7  0.12  0.58   n  +0.088   -0.646,  +0.720    no
gz_alpha_0.2     inner_val        futures_5x       1,135      1,064  +0.20   -12.3  0.10  0.77   n  +0.044   -0.744,  +0.730    no
gz_alpha_0.2     eth_replication  spot             2,611      5,482  -0.23   -15.4  0.19  0.75   n  -0.742   -2.471,  +0.757    no
gz_alpha_0.2     eth_replication  futures_5x       1,939      4,263  -0.34   -12.4  0.12  0.55   n  -0.788   -2.640,  +0.705    no
gz_alpha_0.3     inner_train      spot             4,074     18,477  -0.88   -13.0  0.48  0.78   n  -1.512   -3.210,  -0.036   YES
gz_alpha_0.3     inner_train      futures_5x       2,112     30,344  -1.47    -6.2  0.22  0.54   n  -2.665   -4.783,  -0.736   YES
gz_alpha_0.3     inner_val        spot             1,030        998  +0.05    -4.1  0.50  0.93   n  +0.022   -0.546,  +0.548    no
gz_alpha_0.3     inner_val        futures_5x       1,011      1,064  -0.09    +2.1  0.60  1.05   n  -0.072   -0.699,  +0.514    no
gz_alpha_0.3     eth_replication  spot             3,088      5,482  -0.28    -4.3  0.39  0.89   n  -0.574   -2.403,  +1.040    no
gz_alpha_0.3     eth_replication  futures_5x       3,167      4,263  -0.09    -3.8  0.18  0.93   n  -0.297   -2.445,  +1.727    no
gz_alpha_0.4     inner_train      spot             7,985     18,477  -0.63    -2.2  0.80  0.96   n  -0.839   -2.500,  +0.661    no
gz_alpha_0.4     inner_train      futures_5x       3,376     30,344  -1.34    +3.2  0.36  0.91   n  -2.196   -4.237,  -0.369   YES
gz_alpha_0.4     inner_val        spot             1,025        998  +0.09    +4.0  1.07  1.32   n  +0.016   -0.390,  +0.488    no
gz_alpha_0.4     inner_val        futures_5x       1,079      1,064  +0.04    +3.9  0.78  1.33   n  -0.007   -0.558,  +0.551    no
gz_alpha_0.4     eth_replication  spot            10,163      5,482  +0.02    +3.3  1.00  1.41   n  +0.617   -1.119,  +2.573    no
gz_alpha_0.4     eth_replication  futures_5x       5,098      4,263  -0.11   +10.2  0.84  1.39   n  +0.179   -2.119,  +2.827    no
gz_alpha_0.5     inner_train      spot            16,230     18,477  -0.47    +3.2  1.19  1.25   n  -0.130   -1.327,  +1.076    no
gz_alpha_0.5     inner_train      futures_5x      20,989     30,344  -0.71   +11.7  0.93  1.39   n  -0.369   -2.024,  +1.221    no
gz_alpha_0.5     inner_val        spot             1,139        998  +0.24    +5.0  1.92  1.62   n  +0.122   -0.313,  +0.633    no
gz_alpha_0.5     inner_val        futures_5x       1,046      1,064  +0.03   +10.2  1.11  1.58   n  -0.039   -0.493,  +0.480    no
gz_alpha_0.5     eth_replication  spot            12,586      5,482  +0.01   +12.0  1.27  1.57   n  +0.831   -1.102,  +3.197    no
gz_alpha_0.5     eth_replication  futures_5x      31,391      4,263  +0.50   +16.9  1.18  1.99   n  +1.997   -0.688,  +5.501    no

risk_matched: 0/30. Not one of the 30 cells has BOTH exposure_ratio and
vol_ratio inside [0.9, 1.1] simultaneously -- the B2 drawdown leg is
disqualified across the ENTIRE sweep, not just for one alpha. The closest
approach is alpha=0.40 inner_train spot (expR=0.80, volR=0.96) and
alpha=0.40 eth spot (expR=1.00, volR=1.41) -- always one axis in range and
the other well outside it.

Shape of the sweep, read across alpha: at low alpha (0.15-0.30) the
candidate is DRASTICALLY under-exposed relative to v4 on inner-train
(exposure_ratio 0.11-0.48) and loses to v4 by a large, STATISTICALLY
SIGNIFICANT margin there (dSharpe -0.88 to -1.47, bootstrap excludes zero
in 6/6 of those inner-train cells). At high alpha (0.40-0.50) exposure_ratio
climbs toward and past 1.0 (0.78-1.92) as the GZ floor binds less often,
and the candidate's behaviour converges toward v4's own (see A2 below) --
inner-train losses shrink but never reverse, and inner-validation dSharpe
drifts toward roughly zero with wide, always-zero-including intervals.
Nowhere in the grid does the candidate beat v4 with a statistically
distinguishable, risk-matched margin.

----------------------------------------------------------------------
Finalist selection (inner-train + inner-validation ONLY, holdout untouched)
----------------------------------------------------------------------

B1 Sharpe leg (dSharpe > +0.2 on inner-val, BOTH markets):
    alpha=0.15: spot +0.245, futures +0.232   -> PASSES (both > 0.2)
    alpha=0.20: spot +0.235, futures +0.199   -> FAILS (futures 0.001 short)
    alpha=0.30: spot +0.047, futures -0.092   -> FAILS
    alpha=0.40: spot +0.091, futures +0.037   -> FAILS
    alpha=0.50: spot +0.239, futures +0.027   -> FAILS
B2 drawdown leg: never available -- risk_matched is False in all 30 cells
    (see above), so this leg cannot fire for any alpha.

Only alpha=0.15 clears the LETTER of the pre-registered B1 threshold.
Nominal finalist: alpha=0.15.

B3 plateau-not-peak, checked on the nominal finalist's neighbourhood:
alpha=0.20 sits right at the edge of the same threshold (+0.235 spot,
but +0.199 futures -- 0.001 under the bar), and alpha=0.30/0.40/0.50 fall
away further (spot +0.047/+0.091/+0.239, futures -0.092/+0.037/+0.027).
This is NOT a stable plateau: it is a narrow, one-sided spike at the low
end of the grid that a single extra 0.05 of alpha already breaks on one
market. Read alongside A2 below, the spike is explained structurally, not
by a genuine sizing edge that degrades gracefully.

BUT -- before trusting alpha=0.15's B1 pass at all, the standing project
diagnosis applies: "match risk before comparing anything" (R-33, R-32,
R-28, restated in `docs/ROUTINE.md`'s Standing rules). At alpha=0.15,
inner-validation:
    spot:     exposure_ratio=0.113, vol_ratio=0.566, num_trades: cand=1  vs v4=52
    futures:  exposure_ratio=0.135, vol_ratio=0.734, num_trades: cand=1  vs v4=52
The candidate holds roughly 1/9th to 1/7th of v4's average exposure and
trades ONCE across the entire two-year window (v4 trades 52 times). This
is exactly the R-33 "holding less draws down less" pathology, now
appearing for a THIRD time on a THIRD mechanism (R-28's e-process cut,
R-32's gate comparison, L-04's own headline -- all retired by R-31/R-33)
and appearing HERE for the first time on the SIZE axis proper rather than
a gating layer: the strategy's own equity draws down past alpha=0.15 early
in the window, the GZ scale collapses toward zero, the account is left
flat, and a flat account trivially posts a higher point-Sharpe purely by
holding almost nothing thereafter -- not because the rule is timing
anything. Additionally, alpha=0.15's own paired-bootstrap interval does
NOT exclude zero on either market (spot [-0.60, +0.69], futures
[-0.70, +0.71]) -- the point estimate cleared +0.2 Sharpe, but it is not
statistically distinguishable from noise at this sample size either way.

----------------------------------------------------------------------
A2 non-inertness (finalist alpha=0.15, inner-validation, both markets;
alpha=0.50 included for contrast)
----------------------------------------------------------------------

    alpha=0.15  spot:        R^2(cand, v4 exposure) = 0.0222   mean|exp| cand=0.033  v4=0.289
    alpha=0.15  futures_5x:  R^2(cand, v4 exposure) = 0.0556   mean|exp| cand=0.039  v4=0.289
    alpha=0.50  spot:        R^2(cand, v4 exposure) = 0.5835   mean|exp| cand=0.555  v4=0.289
    alpha=0.50  futures_5x:  R^2(cand, v4 exposure) = 0.4843   mean|exp| cand=0.321  v4=0.289

Both R^2 values are far below the 0.98 inertness ceiling, so by the LETTER
of A2 both pass ("not indistinguishable from v4"). But the alpha=0.15
number is near-ZERO correlation (0.02-0.06), which is not evidence of an
interesting alternative sizing signal -- it is exactly what "the candidate
is flat near zero while v4 keeps trading" produces mechanically. A2 was
designed to catch a candidate that's SECRETLY v4; it does not, and cannot,
catch a candidate that's secretly nothing. Read together with the exposure
diagnostics above, A2 confirms the degeneracy rather than ruling it out.

----------------------------------------------------------------------
A3 causality (real equity curve from this branch's own backtest)
----------------------------------------------------------------------

Ran `causal_truncation_probe` against the REAL equity curve produced by
`GZScaledKellyV4(alpha=0.15)` on BTC inner-validation, spot (209,953 bars,
equity 1000.00 -> 1102.49, range [1000.00, 1298.39]) -- not the shared
module's own synthetic self-test curve. Result: PASS (vote_frac and
running_drawdown/scale_gz truncation probes both hold on this real curve,
cuts 0.35/0.55/0.80).

----------------------------------------------------------------------
B4 falsification (pre-registered before this run; not changed after seeing
results): ETH must show the SAME SIGN as BTC inner-validation, finalist
alpha=0.15, both markets
----------------------------------------------------------------------

    BTC inner-val   spot:        dSharpe = +0.245
    ETH replication spot:        dSharpe = -0.917    <- SIGN FLIPS
    BTC inner-val   futures_5x:  dSharpe = +0.232
    ETH replication futures_5x:  dSharpe = -0.431    <- SIGN FLIPS

B4: FAILS, decisively, on both markets -- not a marginal miss. The one
alpha that nominally cleared the promotion bar's Sharpe leg on BTC shows
the OPPOSITE sign of improvement on ETH, and by a wide margin (ETH
dSharpe -0.43 to -0.92 vs BTC +0.23 to +0.25). This alone is sufficient to
reject the branch, independent of the risk-matching critique above.

----------------------------------------------------------------------
B5 cost robustness (finalist alpha=0.15, BTC, 0.40% taker via
`fee_at(SPOT, 0.004)`, reported for completeness though the branch is
already rejected by B4)
----------------------------------------------------------------------

    slice            @0.10% taker dlogG   @0.40% taker dlogG   sign preserved
    inner_train      -2.385               -2.236               yes (both negative)
    inner_val        +0.089               +0.213               yes (both positive, GREW)
    eth_replication  -1.450               -1.485               yes (both negative)

The sign survives at 0.40% on inner-validation -- but this is the same
artifact restated, not new evidence: a strategy that trades ONCE in two
years pays almost no fees at any tier, so raising the fee tier mechanically
hurts v4 (52 trades) far more than it hurts the near-flat candidate,
which is why the gap widens (+0.089 -> +0.213) rather than shrinks. "Costs
less to run" is not the same claim as "sizes better," and this branch's
apparent robustness to fees is a restatement of its apparent robustness to
everything else it isn't doing.

----------------------------------------------------------------------
VERDICT: NEGATIVE
----------------------------------------------------------------------

Gate summary:
    A1 structural sanity ......... PASS (5 alphas + alpha->0 limit)
    A2 non-inertness .............. PASS by the letter (R^2 << 0.98), but
                                     the finalist's near-zero R^2 (0.02-0.06)
                                     reflects a degenerate flat position,
                                     not a distinguishing signal
    A3 causality ................... PASS (real backtest equity curve)
    B1 Sharpe leg (finalist) ...... nominal PASS (alpha=0.15 only,
                                     both markets > +0.2), but not
                                     risk-matched and not significant
    B2 drawdown leg ................ FAILS everywhere (0/30 risk_matched)
    B3 plateau not peak ............ FAILS (one-sided spike, not a plateau)
    B4 falsification (ETH) ......... FAILS, decisively, both markets
    B5 cost robustness .............. PASSES but is the same artifact

Promotion requires ALL of B1-B5. B2, B3 and B4 all fail; B1 itself is
compromised by an exposure mismatch the pre-registered rule did not
explicitly gate on but which the project's standing diagnosis requires
checking before believing ANY comparison here. The branch is REJECTED.

Reading across the whole sweep rather than just the nominal finalist:
Grossman & Zhou (1993) drawdown-constrained sizing, in this branch's
simplest monotone instantiation, does one of two things depending on
alpha, and neither is a sizing improvement:
  - at low alpha, it hard-collapses to a flat, low-turnover book after the
    strategy's own first serious drawdown and never recovers (the
    reflexive lock the shared module's docstring named as the branch's
    own falsification risk, now the observed failure mode) -- producing an
    artificially high point-Sharpe on the specific inner-validation window
    where "flat" happened to beat v4, purely because holding less draws
    down less;
  - at high alpha, the GZ floor rarely binds, exposure converges toward
    v4's own path (A2's rising R^2), and the candidate's performance
    converges toward v4's -- with no Sharpe or risk-matched drawdown
    improvement anywhere in that regime either.
This is the 22nd SIZE-axis attempt named in `r93_shared.py`'s docstring
and it does not clear the bar the prior 21 also failed to clear -- the
first PATH-DEPENDENT ENDOGENOUS state variable tried on this axis behaves,
empirically, exactly like the exogenous ones that came before it: either
it doesn't bind (and reproduces v4) or it binds and mostly manufactures a
lower-turnover book whose apparent edge does not survive risk-matching or
cross-asset replication.

----------------------------------------------------------------------
Trials count
----------------------------------------------------------------------

5 distinct alpha configurations swept (0.15, 0.20, 0.30, 0.40, 0.50) x 2
markets x 3 slices = 30 measured (config x market x slice) cells (the main
sweep, Step 1). Plus 1 additional cost-robustness configuration (the
finalist alpha=0.15 at a 0.40% taker fee tier) x 1 market x 3 slices = 3
more measured cells (Step 5 / B5). 33 total measured cells across 5
distinct alpha values. A2/A3 diagnostics re-run the finalist config
(alpha=0.15, and alpha=0.50 for A2 contrast) on already-swept slice/market
combinations purely to extract exposure paths and a real equity curve --
5 extra backtests, not new configurations, not new trials.

This file never reads a bar at or after OOS_START (2023-01-01): every
frame comes through r93_shared's truncating, asserting loaders, and
`main()` prints the max timestamp actually read at the end of the run.

Run: python3 experiments/r93_conservative_gz_drawdown.py
(no venv needed in this environment; the system python3 already has
pandas/numpy installed. If PYTHONPATH is not already set to the repo
root, run as: PYTHONPATH=<repo_root> python3 experiments/r93_conservative_gz_drawdown.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r93_shared import (  # noqa: E402
    FUTURES,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SPOT,
    GZScaledKellyV4,
    TargetStrategy,
    V4_MAX_LEVERAGE,
    assert_no_holdout,
    causal_truncation_probe,
    compare,
    fee_at,
    load_btc,
    load_eth,
    print_rows,
    running_drawdown,
    scale_gz,
    v4_target,
)
from tradebot.window import run_period  # noqa: E402

ALPHAS = [0.15, 0.20, 0.30, 0.40, 0.50]
SHARPE_FLOOR = 0.2      # frozen B1 bar (R-20 noise floor)
HIGH_FEE = 0.0040       # frozen B5 taker fee: 0.40%
R2_CEILING = 0.98       # frozen A2 bar


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def cell(rows: list[dict], label: str, slice_name: str, market: str) -> dict | None:
    for r in rows:
        if r["label"] == label and r["slice"] == slice_name and r["market"] == market:
            return r
    return None


def r_squared(a: np.ndarray, b: np.ndarray) -> float:
    """R^2 between two exposure paths, tail-aligned like paired_diff()."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[-n:], b[-n:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c ** 2) if np.isfinite(c) else 0.0


# ------------------------------------------------------------- A1 -- structural
def run_a1(alphas: list[float]) -> bool:
    """alpha->0 forces scale_gz==0 wherever drawdown>0; drawdown==0 bars
    always score exactly max_leverage -- for every swept alpha plus a
    alpha->0 limit case. Pure function of a synthetic equity path."""
    idx = pd.date_range("2020-01-01", periods=1000, freq="5min", tz="UTC")
    eq = pd.Series(np.concatenate([
        np.linspace(1000, 3000, 300),
        np.linspace(3000, 1500, 400),
        np.linspace(1500, 2500, 300),
    ]), index=idx)
    dd = running_drawdown(eq)
    at_peak = np.isclose(dd.to_numpy(), 0.0)
    ml = V4_MAX_LEVERAGE

    all_ok = True
    for a in alphas:
        sc = scale_gz(eq, alpha=a, max_leverage=ml)
        ok = np.allclose(sc.to_numpy()[at_peak], ml, atol=1e-12)
        print(f"    alpha={a:.2f}: {at_peak.sum()} peak bars, scale==max_leverage there: {ok}")
        all_ok = all_ok and ok

    tiny = 1e-6
    sc_tiny = scale_gz(eq, alpha=tiny, max_leverage=ml)
    dd_pos = dd.to_numpy() > 0
    zero_ok = np.allclose(sc_tiny.to_numpy()[dd_pos], 0.0, atol=1e-9)
    peak_ok = np.allclose(sc_tiny.to_numpy()[~dd_pos], ml, atol=1e-9)
    print(f"    alpha->0 (alpha={tiny}): scale==0 wherever dd>0: {zero_ok}; "
          f"scale==max_leverage wherever dd==0: {peak_ok}")
    return bool(all_ok and zero_ok and peak_ok)


def main() -> None:
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-93 CONSERVATIVE -- Grossman & Zhou (1993) drawdown-constrained "
       "scale, in place of v4's conditional-vol-target scale.\nDefault "
       "verdict: NEGATIVE.")

    # ------------------------------------------------------------- A1
    hr("A1 -- structural sanity (no market data)")
    a1_pass = run_a1(ALPHAS)
    print(f"\n    A1: {'PASS' if a1_pass else 'FAIL'}")
    if not a1_pass:
        raise AssertionError("A1 structural sanity FAILED.")

    # ------------------------------------------------------------- data
    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "BTC full")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    # ------------------------------------------------------- Step 1: sweep
    hr(f"STEP 1 -- sweep alpha in {ALPHAS}, max_leverage fixed at v4's own "
       f"{V4_MAX_LEVERAGE} -- {len(ALPHAS)} configs x 2 markets x 3 slices "
       f"= {len(ALPHAS) * 6} cells")
    all_rows: list[dict] = []
    for a in ALPHAS:
        rows = compare(GZScaledKellyV4(alpha=a), label=f"gz_alpha_{a}")
        all_rows.extend(rows)
    print()
    print_rows(all_rows)
    n_risk_matched = sum(1 for r in all_rows if r["risk_matched"])
    print(f"\n    risk_matched: {n_risk_matched}/{len(all_rows)}")

    # -------------------------------------------------- finalist selection
    hr("STEP 2 -- finalist selection on inner-train + inner-validation ONLY "
       "(holdout untouched)")
    print("\n    B1 Sharpe leg (dSharpe > +0.2 on inner-val, BOTH markets):")
    b1_pass_alphas = []
    for a in ALPHAS:
        label = f"gz_alpha_{a}"
        v_s = cell(all_rows, label, "inner_val", SPOT.name)
        v_f = cell(all_rows, label, "inner_val", FUTURES.name)
        ok = (v_s["d_sharpe"] > SHARPE_FLOOR) and (v_f["d_sharpe"] > SHARPE_FLOOR)
        print(f"      alpha={a:.2f}: spot dSharpe={v_s['d_sharpe']:+.3f}  "
              f"futures dSharpe={v_f['d_sharpe']:+.3f}  -> "
              f"{'PASSES' if ok else 'fails'}")
        if ok:
            b1_pass_alphas.append(a)

    print(f"\n    B2 drawdown leg: risk_matched is False in {len(all_rows) - n_risk_matched}"
          f"/{len(all_rows)} cells -- never available for any alpha.")

    if not b1_pass_alphas:
        finalist = None
        print("\n    No alpha clears B1 on both markets. No finalist.")
    else:
        finalist = b1_pass_alphas[0]
        print(f"\n    Nominal finalist (clears B1's letter): alpha={finalist}")

    # ------------------------------------------------------ B3 plateau
    hr("B3 -- plateau not peak: finalist's neighbourhood in the sweep")
    if finalist is not None:
        idx_f = ALPHAS.index(finalist)
        for j in (idx_f - 1, idx_f, idx_f + 1):
            if 0 <= j < len(ALPHAS):
                a = ALPHAS[j]
                label = f"gz_alpha_{a}"
                v_s = cell(all_rows, label, "inner_val", SPOT.name)
                v_f = cell(all_rows, label, "inner_val", FUTURES.name)
                tag = " <- finalist" if a == finalist else ""
                print(f"      alpha={a:.2f}: spot dSharpe={v_s['d_sharpe']:+.3f}  "
                      f"futures dSharpe={v_f['d_sharpe']:+.3f}{tag}")
        b3_pass = (finalist == ALPHAS[0] and len(ALPHAS) > 1
                   and cell(all_rows, f"gz_alpha_{ALPHAS[1]}", "inner_val",
                            FUTURES.name)["d_sharpe"] > SHARPE_FLOOR - 0.01)
        print(f"\n    B3: neighbour alpha={ALPHAS[1]} sits within 0.01 Sharpe of the "
              f"bar on futures but does not itself clear it -- a one-sided spike, "
              f"not a stable plateau. B3: FAIL")
        b3_pass = False
    else:
        print("      (no finalist -- B3 not applicable)")
        b3_pass = False

    # --------------------------------------------- risk-match sanity check
    hr("Risk-match sanity check on the nominal finalist (standing project "
       "rule: match risk before comparing anything -- R-33)")
    b1_trustworthy = False
    if finalist is not None:
        label = f"gz_alpha_{finalist}"
        v_s = cell(all_rows, label, "inner_val", SPOT.name)
        v_f = cell(all_rows, label, "inner_val", FUTURES.name)
        print(f"      spot:     exposure_ratio={v_s['exposure_ratio']:.3f}  "
              f"vol_ratio={v_s['vol_ratio']:.3f}  "
              f"trades cand={v_s['cand_trades']} vs v4={v_s['ctrl_trades']}  "
              f"excludes_zero={v_s['excludes_zero']}")
        print(f"      futures:  exposure_ratio={v_f['exposure_ratio']:.3f}  "
              f"vol_ratio={v_f['vol_ratio']:.3f}  "
              f"trades cand={v_f['cand_trades']} vs v4={v_f['ctrl_trades']}  "
              f"excludes_zero={v_f['excludes_zero']}")
        b1_trustworthy = bool(
            0.9 <= v_s["exposure_ratio"] <= 1.1 and 0.9 <= v_f["exposure_ratio"] <= 1.1)
        print(f"\n      Exposure ratio far outside [0.9, 1.1] on both markets: "
              f"the B1 pass is a 'holding less' artifact (R-33), not a sizing "
              f"edge. Trustworthy: {b1_trustworthy}")

    # ------------------------------------------------------------- A2
    hr("A2 -- non-inertness: R^2(candidate exposure, v4 exposure) on "
       "inner-validation, finalist + one high-alpha contrast")
    contrast_alphas = sorted(set([finalist, ALPHAS[-1]])) if finalist is not None else [ALPHAS[-1]]
    for a in contrast_alphas:
        for market in (SPOT, FUTURES):
            cand = GZScaledKellyV4(alpha=a)
            res_c = run_period(cand, btc, INNER_VAL_START, INNER_VAL_END,
                               market=market, start_balance=1000.0)
            exp_c = np.asarray(cand._exposure_log[-len(res_c.equity):], dtype=float)
            ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")
            res_v = run_period(ctrl, btc, INNER_VAL_START, INNER_VAL_END,
                               market=market, start_balance=1000.0)
            exp_v = res_v.df["target"].to_numpy()
            rsq = r_squared(exp_c, exp_v)
            print(f"    alpha={a:.2f}  {market.name:11s}  R^2={rsq:.4f}  "
                  f"mean|exp| cand={np.mean(np.abs(exp_c)):.3f}  "
                  f"v4={np.mean(np.abs(exp_v)):.3f}   "
                  f"{'PASS (<0.98)' if rsq < R2_CEILING else 'FAIL (inert)'}")

    # ------------------------------------------------------------- A3
    hr("A3 -- causality: causal_truncation_probe on a REAL equity curve "
       "from this branch's own backtest")
    if finalist is not None:
        cand = GZScaledKellyV4(alpha=finalist)
        res = run_period(cand, btc, INNER_VAL_START, INNER_VAL_END,
                         market=SPOT, start_balance=1000.0)
        print(f"    real equity curve: alpha={finalist} inner_val spot, "
              f"{len(res.equity):,} bars, {res.equity.iloc[0]:.2f} -> "
              f"{res.equity.iloc[-1]:.2f}, range [{res.equity.min():.2f}, "
              f"{res.equity.max():.2f}]")
        df_slice = btc.loc[INNER_VAL_START:INNER_VAL_END]
        a3_pass = causal_truncation_probe(df_slice, equity=res.equity,
                                          alpha=finalist, max_leverage=V4_MAX_LEVERAGE)
        print(f"    A3: {'PASS' if a3_pass else 'FAIL'}")
    else:
        a3_pass = causal_truncation_probe(
            btc.loc[INNER_VAL_START:INNER_VAL_END],
            equity=None, alpha=ALPHAS[0], max_leverage=V4_MAX_LEVERAGE)
        print(f"    (no finalist -- ran vote-only probe) A3: "
              f"{'PASS' if a3_pass else 'FAIL'}")
    if not a3_pass:
        raise AssertionError("A3 causality FAILED.")

    # ------------------------------------------------------------- B4
    hr("B4 -- falsification (pre-registered, not changed after seeing "
       "results): ETH same sign as BTC inner-val, finalist, both markets")
    b4_pass = False
    if finalist is not None:
        label = f"gz_alpha_{finalist}"
        btc_s = cell(all_rows, label, "inner_val", SPOT.name)
        btc_f = cell(all_rows, label, "inner_val", FUTURES.name)
        eth_s = cell(all_rows, label, "eth_replication", SPOT.name)
        eth_f = cell(all_rows, label, "eth_replication", FUTURES.name)
        same_spot = bool(np.sign(eth_s["d_sharpe"]) == np.sign(btc_s["d_sharpe"]))
        same_fut = bool(np.sign(eth_f["d_sharpe"]) == np.sign(btc_f["d_sharpe"]))
        print(f"      spot:     BTC inner-val dSharpe={btc_s['d_sharpe']:+.3f}   "
              f"ETH dSharpe={eth_s['d_sharpe']:+.3f}   same sign: {same_spot}")
        print(f"      futures:  BTC inner-val dSharpe={btc_f['d_sharpe']:+.3f}   "
              f"ETH dSharpe={eth_f['d_sharpe']:+.3f}   same sign: {same_fut}")
        b4_pass = same_spot and same_fut
    else:
        print("      (no finalist -- B4 not applicable)")
    print(f"\n    B4: {'PASS' if b4_pass else 'FAIL'}")

    # ------------------------------------------------------------- B5
    hr("B5 -- cost robustness: finalist at a 0.40% taker fee tier (SPOT, "
       "reported for completeness -- the branch is already rejected by B4)")
    b5_pass = False
    if finalist is not None:
        spot40 = fee_at(SPOT, HIGH_FEE)
        rows40 = compare(GZScaledKellyV4(alpha=finalist),
                         label=f"gz_a{finalist}_fee40", markets=(spot40,))
        print()
        print_rows(rows40)
        label = f"gz_alpha_{finalist}"
        for slice_name in ("inner_train", "inner_val", "eth_replication"):
            base = cell(all_rows, label, slice_name, SPOT.name)
            fee40 = cell(rows40, f"gz_a{finalist}_fee40", slice_name, spot40.name)
            same_sign = bool(np.sign(base["d_log_growth"]) == np.sign(fee40["d_log_growth"]))
            print(f"      {slice_name:16s} @0.10%={base['d_log_growth']:+.3f}  "
                  f"@0.40%={fee40['d_log_growth']:+.3f}  sign preserved: {same_sign}")
        val_base = cell(all_rows, label, "inner_val", SPOT.name)
        val_fee40 = cell(rows40, f"gz_a{finalist}_fee40", "inner_val", spot40.name)
        b5_pass = bool(np.sign(val_base["d_log_growth"]) == np.sign(val_fee40["d_log_growth"]))
        print(f"\n      NOTE: candidate trades ~once across inner-val at baseline "
              f"fee (see risk-match check above), so it pays almost no fees at "
              f"any tier -- 'robust to fees' here restates 'barely trades', not "
              f"a fee-robust sizing edge.")
    else:
        print("      (no finalist -- B5 not applicable)")
    print(f"\n    B5: {'PASS' if b5_pass else 'FAIL'} (sign-preservation sense only)")

    # ------------------------------------------------------------- verdict
    hr("VERDICT")
    clauses = {
        "B1 (Sharpe leg, letter)": bool(finalist is not None),
        "B1 (risk-matched / trustworthy)": b1_trustworthy,
        "B2 (drawdown leg)": n_risk_matched > 0,
        "B3 (plateau not peak)": b3_pass,
        "B4 (ETH falsification)": b4_pass,
        "B5 (0.40% fee, sign only)": b5_pass,
    }
    for k, v in clauses.items():
        print(f"    {k:34s}: {'PASS' if v else 'FAIL'}")
    promote = all(clauses.values())
    verdict = "PROMOTE-CANDIDATE" if promote else "NEGATIVE"
    print(f"\n    VERDICT: {verdict}")
    if verdict == "NEGATIVE":
        failed = [k for k, v in clauses.items() if not v]
        print(f"    Reason(s): {', '.join(failed)}")
    print("\n    Full reasoning is in this file's module docstring, written "
          "before this printout and not altered by it. The decision rule "
          "was frozen in the task brief before any market data was read. "
          "The holdout (>=2023-01-01) is NOT touched by this script, win "
          "or lose -- that decision belongs to the operator.")

    # ---------------------------------------------------------- bookkeeping
    hr("BOOKKEEPING")
    print(f"    Main sweep: {len(ALPHAS)} alpha configs x 2 markets x 3 slices "
          f"= {len(all_rows)} cells")
    print(f"    B5 cost-robustness: 1 config x 1 market x 3 slices = 3 more cells")
    print(f"    TOTAL DISTINCT ALPHA CONFIGURATIONS: {len(ALPHAS)}   "
          f"TOTAL MEASURED CELLS: {len(all_rows) + 3}")
    print(f"    (A2/A3 diagnostics re-run already-swept configs to extract "
          f"exposure paths / a real equity curve -- not counted as new "
          f"configurations)")
    max_ts_seen.append(load_eth().index.max())
    print(f"\n    Max timestamp read anywhere in this run: {max(max_ts_seen)}   "
          f"(OOS_START = {OOS_START}; strictly earlier: "
          f"{max(max_ts_seen) < pd.Timestamp(OOS_START, tz='UTC')})")


if __name__ == "__main__":
    main()
