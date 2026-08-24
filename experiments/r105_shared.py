"""Shared, read-only utilities and pre-registration for the R-105 round (08-24).

DIRECTION, in one sentence: give `kelly_regime_v4`'s 3-anchor directional VOTE
a live, causal measure of its own SPECIFICATION uncertainty -- how much
alternative, equally-defensible anchor-ladder choices disagree with the
shipped 20/40/80 vote right now -- and discount exposure multiplicatively
when they disagree, as a fourth ERR-axis construction and the first to use
MODEL/SPECIFICATION uncertainty (disagreement across alternative
parameterizations of the SAME mechanism) rather than SAMPLING uncertainty of
one fixed parameterization's realized edge (R-28/retracted, R-87, R-104 --
three attempts, all on sampling uncertainty) or resampling over historical
STRESS EPISODES (R-101, six discrete events).

**Literature grounding, fetched and read via WebSearch this round:**

- Chatfield, C. (1995), "Model Uncertainty, Data Mining and Statistical
  Inference", *Journal of the Royal Statistical Society, Series A* 158(3),
  419-466. The general statistical distinction this round rests on: MODEL
  (specification) uncertainty -- not knowing which of several defensible
  model forms is correct -- is a source of error additional to, and
  conceptually distinct from, SAMPLING uncertainty of one fixed model's
  estimated parameters. Every prior ERR-axis round in this ledger (R-28,
  R-87, R-104) measured sampling uncertainty of a single fixed vote's
  historical edge or confidence; none measured disagreement ACROSS
  alternative model specifications. This round is the first to do so.
- Raftery, A. E., Gneiting, T., Balabdaoui, F., & Polakowski, M. (2005),
  "Using Bayesian Model Averaging to Calibrate Forecast Ensembles",
  *Monthly Weather Review* 133(5), 1155-1174. Establishes the operational
  principle both branches below implement in a simplified, non-Bayesian
  form: an ensemble's own SPREAD (disagreement among members of a
  multi-model or multi-parameterization ensemble) is a genuine, useful
  proxy for forecast uncertainty, distinct from any single member's own
  confidence -- the spread-skill relationship. Neither branch below fits a
  full BMA posterior (no weights are learned; every ensemble member here is
  either a pre-registered, a-priori parameterization or literally a
  component of the shipped vote itself) -- only the ensemble-SPREAD-as-
  uncertainty principle is imported, not the estimation machinery.
- Baltas, N., & Kosowski, R. (2013), "Momentum Strategies in Futures Markets
  and Trend-Following Funds", working paper (Imperial College Business
  School; later published in *Journal of Banking & Finance*, 2017 variant);
  see also the closely related time-scale-diversification literature
  (Levine & Pedersen 2016, already this project's own kelly_regime_v2/v3/v4
  citation chain; Baltas & Kosowski's own finding that trend signals at
  different lookback horizons carry low pairwise correlation and combining
  several improves risk-adjusted performance and cuts drawdowns). Cited as
  the economic reason a multi-horizon anchor-ladder ENSEMBLE (this round's
  novel branch) is a reasonable, literature-grounded uncertainty probe
  rather than an arbitrary one: if different-speed trend anchors normally
  disagree by only a little and diverge sharply exactly around regime
  transitions, ensemble spread across anchor ladders is plausibly informative
  about exactly the moments this project's whole `kelly_regime` family
  exists to protect against.
- Quenouille, M. H. (1949), "Approximate Tests of Correlation in
  Time-Series", *Journal of the Royal Statistical Society, Series B* 11(1),
  68-84; Tukey, J. W. (1958), "Bias and Confidence in Not-Quite Large
  Samples", *Annals of Mathematical Statistics* 29(2), 614; Efron, B.
  (1979), "Bootstrap Methods: Another Look at the Jackknife", *Annals of
  Statistics* 7(1), 1-26. The same jackknife citations R-101 used --
  reused here for the CONSERVATIVE branch, but applied to a structurally
  different sampling unit: R-101 jackknifed the vote's REALIZED EDGE across
  six discrete historical STRESS EPISODES (a resampling over TIME/history,
  N=6). This round's conservative branch jackknifes the vote's own THREE
  ANCHOR COMPONENTS at a single bar (a resampling over the MODEL'S OWN
  CONSTRUCTION, contemporaneous, N=3, needs no history or burn-in at all)
  -- the "genuinely different sampling unit" R-101's own closing line named
  as untried ("neither an exogenous market statistic... nor a
  resampling-based empirical uncertainty estimate over the same six sparse
  episodes").

**Which constraint this attacks: ERR** (no error control anywhere in the
signal path). Fourth attempt on this axis, after R-28 (e-process drawdown
cut, RETRACTED -- R-33 showed the whole effect was an unmatched
exposure-level artifact), R-87 (Adaptive Conformal Inference on the vote's
confidence and the Kelly scale's dispersion, both NEGATIVE -- the
vote-confidence wrapper never escaped Step-A inertness because "BTC's real
vote-lean hit rate (~55.1%) sits persistently above the 50% coin-flip target
this instance tracked"), and R-104 (periodic Monte Carlo bootstrap /
continuous HAC-PSR significance discount of the vote's own historical edge,
both NEGATIVE -- the novel PSR branch was inert for the identical reason
R-87 found, "BTC's vote-only edge is significant almost everywhere", by a
structurally different non-conformal estimator; the conservative bootstrap
branch bound non-trivially and passed four of five clauses but failed B1
asymmetrically between markets). R-104's own closing line named the live,
untried candidate this round tries: "model/specification uncertainty across
the vote's own construction... is one live, untried candidate."

**Not a duplicate of:**
- R-104 (bootstrap/HAC significance of the vote's own FIXED historical
  edge): a SAMPLING-uncertainty construction -- it asks "is the realized
  P&L of the one shipped vote distinguishable from zero", using a bootstrap
  or HAC standard error of a TIME SERIES of daily P&L. Neither branch below
  ever computes a standard error, a t-statistic, a p-value, or resamples a
  P&L series at all -- both measure, at each bar independently, how much a
  SMALL SET OF ALTERNATIVE MODEL SPECIFICATIONS (either alternative anchor
  ladders, or the shipped ladder's own components with one removed) would
  have voted differently than the vote actually shipped. No notion of
  statistical significance appears anywhere in this round.
- R-87 (Adaptive Conformal Inference): an ONLINE COVERAGE CALIBRATION
  scheme with a miscoverage feedback loop against a nominal target. Neither
  branch below tracks coverage, adjusts a quantile, or maintains any
  feedback state across bars -- the conservative branch's jackknife
  statistic is fully determined, at every bar, by that bar's own three
  anchor votes (no memory of any earlier bar at all); the novel branch's
  ensemble-spread statistic only carries an expanding NORMALIZATION scale
  (see below), never a coverage target or calibration loop.
- R-101 (delete-one-episode jackknife of the vote's REALIZED EDGE across six
  historical stress episodes): the closest methodological relative, and the
  reason the conservative branch below is careful to state the distinction
  explicitly. R-101's jackknife resamples HISTORY (N=6 discrete episodes,
  each an interval of many days, requiring the whole pre-holdout record to
  exist before a single jackknife estimate can be formed) to ask "how
  sensitive is the vote's own average historical P&L to leaving out any one
  stress event". This round's conservative branch jackknifes the vote's
  OWN THREE ANCHOR COMPONENTS (N=3, a structural feature of the model
  itself, not of history) to ask "how sensitive is TODAY'S vote to leaving
  out any one anchor" -- computable from a single bar's own anchor values
  alone, no historical accumulation, no stress-episode calendar, and no
  P&L anywhere in the statistic. R-101's own closing line named this
  distinction as the live, untried candidate on the SIZE axis; here it is
  tried on the ERR axis instead, since it measures disagreement/uncertainty
  about the SIGNAL rather than an empirical confidence multiplier fit to
  realized returns.
- R-28 / R-31 (e-process drawdown cut, retracted): a game-theoretic
  testing-by-betting martingale used to trigger a discrete drawdown cut,
  retracted because the whole effect was an exposure-collapse artifact.
  Neither branch below uses a martingale, a betting scheme, or a discrete
  cut-trigger; both apply continuous multiplicative shrinkage via a
  disagreement/dispersion statistic that carries no betting-process
  interpretation at all.
- R-06 / R-07 (anchor-ladder empirical sweeps establishing the 18-28 day
  plateau) and R-40 (bagging the ladder plateau) and R-45 (robust parameter
  selection / walk-forward re-estimation of the ladder): all SEARCH for, or
  average the SIGNAL of, alternative ladders in backtest space, in order to
  pick or blend a BETTER-PERFORMING ladder. Both branches below hold the
  shipped 20/40/80 ladder as the TRADED vote completely UNCHANGED at every
  bar (verified by the Step-0 kill switch below) -- alternative ladders (or
  the shipped ladder's own leave-one-out sub-votes) are used ONLY to build a
  disagreement statistic that discounts the SHIPPED vote's exposure; no
  alternative ladder's own signal or P&L is ever traded, blended into the
  traded signal, or selected between.
- R-97 (Wasserstein-DRO ambiguity-ball sizing keyed on regime-cycle count):
  a distributional-ROBUSTNESS optimization framework (worst-case return
  over an ambiguity ball), not an ensemble-disagreement or jackknife
  statistic; carries no notion of "how much do alternative specifications
  disagree" anywhere.
- Every SIZE-axis round (R-34...R-103): all retune `scale`'s magnitude,
  supply an exogenous or endogenous market-state variable, or decompose
  v4's own realized-variance object; none measures disagreement across
  alternative anchor-ladder or anchor-component specifications of the VOTE.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither may edit it. Nothing here reads a bar
at or after OOS_START (2023-01-01); every function that walks a data frame
is either called through `assert_no_holdout`-guarded slices (`compare()`,
`run_slice()`, inherited unmodified from r102_shared through r104_shared)
or, for the Step-0 grid below, is explicitly restricted to
`INNER_TRAIN_START..INNER_TRAIN_END`.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) Baltas & Kosowski's own finding is that different-horizon trend signals
NORMALLY carry low-to-moderate correlation -- if that is true here too, the
novel branch's ensemble disagreement may simply be a common, unremarkable
state (bind_frac high) that is not concentrated around genuinely risky
transitions, reproducing the R-87/R-104 "real but inert" pattern by a third
estimator. (2) `kelly_regime_v4`'s own vote is ALREADY latched (each anchor
holds its previous verdict inside the 1% band rather than sitting neutral),
so anchor DISAGREEMENT (the conservative branch's mixed 1-of-3/2-of-3
states) may be concentrated in exactly the bars where the vote is mid-
transition and about to resolve favourably -- discounting exposure there
could remove part of the edge rather than protect against risk, the same
"discounting the informative part" failure several SIZE-axis rounds
(R-59, R-60) found when they touched the vote's timing rather than its
scale. (3) The conservative branch's jackknife statistic takes only two
values by construction (0 when the three anchors are unanimous, a fixed
positive constant when exactly one dissents) -- if this project's own
noise floor (+/-0.2 Sharpe) cannot be moved by a purely BINARY discount
regardless of floor, that is itself the informative result, distinct from
either prior ERR round's failure shape (both of which had continuous,
graded discounts).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Re-exported verbatim from r104_shared (itself re-exporting r103_shared,
# r102_shared): identical control machinery, so every number in this round
# is directly comparable to R-101/R-102/R-103/R-104's own.
from experiments.r104_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    ETH_SLICE_NAME,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SLICES,
    SPOT,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_series,
    causal_truncation_probe_vote,
    compare,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_target,
    v4_vote_frac,
)

# The generic vote-construction primitive (not re-exported past r102_shared,
# imported directly from there): v4's own vote is `vote_frac(df, (20,40,80),
# band=0.01)`. Both branches below build alternative/partial specifications
# of the SAME primitive rather than re-implementing the anchor-vote logic.
from experiments.r102_shared import (  # noqa: E402,F401
    V4_BAND,
    V4_HORIZONS,
    vote_frac,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert abs(V4_BAND - 0.01) < 1e-12, V4_BAND

# ------------------------------------------------------------------------
# Pre-registered constants shared by BOTH branches' Step-0 gate and B1-B5
# promotion bar -- FIXED before either branch was dispatched, identical to
# R-104's own convention so every number this round produces is directly
# comparable to R-104's (and, transitively, R-101/R-102/R-103's).
# ------------------------------------------------------------------------
STEP0_FLOOR_GRID = (0.3, 0.5, 0.7)      # discount clip floor, Step-0 selection grid
SELECTION_ORDER = (0.5, 0.3, 0.7)        # pre-registered primary-cell preference order
BIND_FRAC_THRESH = 0.01                  # Step-0 kill switch A: must bind >1% of inner-train
R2_THRESH = 0.98                         # Step-0 kill switch B: must not be a near-exact rescale
B3_MIN_DAYS_GRID = (60, 120, 250)        # burn-in plateau grid (conservative branch only --
                                          # see r105_conservative's own docstring for why the
                                          # novel branch's B3 grid differs)
FEE_TIER = 0.0040                        # 0.40% taker, B5
SHARPE_NOISE_FLOOR = 0.2                 # ROUTINE.md's own promotion bar


# ------------------------------------------------------------------------
# Generic B1/B2/B4/B5 promotion-bar machinery, factored out so BOTH
# branches call the IDENTICAL gate code rather than each re-implementing
# it (R-104's own two branches each wrote their own copy; centralizing it
# here removes any chance of the two branches' gates silently diverging).
# B3 (plateau) is intentionally NOT centralized: the two branches sweep
# structurally different nuisance parameters (burn-in days vs. the
# ensemble's own membership grid), so each branch's own file defines its
# own B3 given this module's `inner_val_rows` helper.
# ------------------------------------------------------------------------

def inner_val_rows(build_fn, label: str, btc: pd.DataFrame,
                   markets: tuple = (SPOT, FUTURES)) -> list[dict]:
    """Lightweight inner-validation-only comparison (both markets), reused
    verbatim from R-104's own idiom for B3/B5 cells that don't need the
    full compare() overhead (inner_train + ETH)."""
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")
    cand = TargetStrategy(build_fn, name=f"r105_{label}")
    rows = []
    for market in markets:
        a = run_slice(cand, btc, INNER_VAL_START, INNER_VAL_END, "inner_val", market)
        b = run_slice(ctrl, btc, INNER_VAL_START, INNER_VAL_END, "inner_val", market)
        pr = paired_diff(a.daily, b.daily)
        exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure
                    if b.mean_abs_exposure else float("nan"))
        vol_ratio = (a.realized_vol / b.realized_vol
                    if b.realized_vol else float("nan"))
        risk_matched = (bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1)
                       if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False)
        rows.append(dict(
            label=label, market=market.name,
            d_sharpe=a.sharpe - b.sharpe, d_dd=a.max_drawdown_pct - b.max_drawdown_pct,
            exposure_ratio=exp_ratio, vol_ratio=vol_ratio, risk_matched=risk_matched,
            boot_d_loggrowth=pr.diff.point, boot_lo=pr.diff.lo, boot_hi=pr.diff.hi,
            excludes_zero=bool(pr.diff.lo > 0 or pr.diff.hi < 0),
        ))
    return rows


def b1_from_inner_val(inner_val_primary: list[dict]) -> tuple[bool, list[dict]]:
    """B1 (gating): on inner_val, BOTH markets -- d_sharpe > +0.2 OR the
    bootstrap interval's lower bound excludes zero on the favourable side."""
    cells = []
    for r in inner_val_primary:
        passes = (r["d_sharpe"] > SHARPE_NOISE_FLOOR) or (r["boot_lo"] > 0)
        cells.append(dict(market=r["market"], passes=passes, boot_lo=r["boot_lo"],
                          boot_hi=r["boot_hi"], d_sharpe=r["d_sharpe"]))
    return all(c["passes"] for c in cells), cells


def b2_diagnostic(inner_val_primary: list[dict]) -> tuple[bool, list[dict]]:
    """B2 (diagnostic ONLY, never itself gates promotion): drawdown
    improvement counts only where risk_matched (R-33's own standing rule)."""
    cells = [dict(market=r["market"], risk_matched=r["risk_matched"], d_dd=r["d_dd"],
                  voided=not r["risk_matched"]) for r in inner_val_primary]
    return True, cells


def b4_eth_falsification(eth_primary: list[dict],
                         inner_val_primary: list[dict]) -> tuple[bool, bool, list[dict]]:
    """B4 (ETH falsification, PRE-REGISTERED): same sign as BTC inner_val,
    per market. Returns (partial_pass [>=1 market], full_pass [all markets], cells)."""
    cells = []
    for r in eth_primary:
        btc_match = next((c for c in inner_val_primary if c["market"] == r["market"]), None)
        same_sign = (btc_match is not None and
                    np.sign(r["d_sharpe"]) == np.sign(btc_match["d_sharpe"]) and
                    r["d_sharpe"] != 0)
        cells.append(dict(market=r["market"], d_sharpe=r["d_sharpe"],
                          excludes_zero=r["excludes_zero"], boot_lo=r["boot_lo"],
                          boot_hi=r["boot_hi"], same_sign_as_btc=same_sign))
    return any(c["same_sign_as_btc"] for c in cells), all(c["same_sign_as_btc"] for c in cells), cells


def b5_fee_tier(build_primary, label: str, btc: pd.DataFrame,
                inner_val_primary: list[dict]) -> tuple[bool, list[dict]]:
    """B5 (cost robustness, gating): at the selected primary cell, 0.40%
    taker on both markets, BTC inner-validation only -- no sign reversal
    on either d_sharpe or the bootstrap log-growth point estimate."""
    fee_markets = (fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER))
    fee_rows = inner_val_rows(build_primary, label, btc, markets=fee_markets)
    cells = []
    for r in fee_rows:
        base = next((c for c in inner_val_primary if c["market"] == r["market"]), None)
        d_sharpe_no_reversal = (base is not None and
                               not (np.sign(r["d_sharpe"]) != np.sign(base["d_sharpe"])
                                    and r["d_sharpe"] != 0 and base["d_sharpe"] != 0))
        dlog_no_reversal = (base is not None and
                          not (np.sign(r["boot_d_loggrowth"]) != np.sign(base["boot_d_loggrowth"])
                               and r["boot_d_loggrowth"] != 0 and base["boot_d_loggrowth"] != 0))
        cells.append(dict(market=r["market"], d_sharpe=r["d_sharpe"],
                          base_d_sharpe=base["d_sharpe"] if base else float("nan"),
                          boot_d_loggrowth=r["boot_d_loggrowth"],
                          base_boot_d_loggrowth=base["boot_d_loggrowth"] if base else float("nan"),
                          d_sharpe_no_reversal=d_sharpe_no_reversal,
                          dlog_no_reversal=dlog_no_reversal,
                          no_reversal=d_sharpe_no_reversal and dlog_no_reversal))
    return all(c["no_reversal"] for c in cells), cells


def print_plateau_table(all_rows: dict) -> None:
    hdr = (f"{'grid_key':>10s} {'market':>9s} {'dSh':>7s} {'dDD':>7s} "
          f"{'expR':>5s} {'volR':>5s} {'RM':>3s} {'dlogG':>7s} "
          f"{'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for key, rows in all_rows.items():
        for r in rows:
            print(f"{key!s:>10s} {r['market']:>9s} {r['d_sharpe']:+7.2f} "
                  f"{r['d_dd']:+7.1f} {r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
                  f"{'Y' if r['risk_matched'] else 'n':>3s} {r['boot_d_loggrowth']:+7.3f} "
                  f"{r['boot_lo']:+8.3f},{r['boot_hi']:+8.3f} "
                  f"{'YES' if r['excludes_zero'] else 'no':>5s}")


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    # V4_HORIZONS / V4_BAND / vote_frac reproduce v4_vote_frac exactly.
    idx = pd.date_range("2017-01-01", periods=200_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(105)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    df = pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995,
                       "close": close, "volume": 1.0}, index=idx)
    a = vote_frac(df, V4_HORIZONS, V4_BAND).to_numpy()
    b = v4_vote_frac(df).to_numpy()
    assert np.allclose(a, b, equal_nan=True), "vote_frac(V4_HORIZONS) must equal v4_vote_frac"

    # b1_from_inner_val / b4_eth_falsification: boundary sanity.
    fake_inner = [dict(market="spot", d_sharpe=0.3, boot_lo=-0.1, boot_hi=0.5),
                 dict(market="futures_5x", d_sharpe=0.05, boot_lo=0.01, boot_hi=0.2)]
    ok, cells = b1_from_inner_val(fake_inner)
    assert ok and all(c["passes"] for c in cells)
    fake_inner_fail = [dict(market="spot", d_sharpe=0.05, boot_lo=-0.1, boot_hi=0.1),
                       dict(market="futures_5x", d_sharpe=0.05, boot_lo=0.01, boot_hi=0.2)]
    ok2, _ = b1_from_inner_val(fake_inner_fail)
    assert not ok2


_self_test()
