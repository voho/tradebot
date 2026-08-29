"""Shared, read-only utilities and frozen pre-registration for the R-186 round (08-29).

DIRECTION, in one sentence: `kelly_regime_v4`'s realized position is a
PATH-DEPENDENT quantization of its desired exposure -- `if |desired - pos| >
deadband: pos = desired`, so the grid the strategy snaps to depends on where
it happened to have been, not on the signal alone -- and since that 10%
deadband is 26-55% of v4's own mean notional (R-57/R-62 as quoted by R-145),
v4's realized equity curve is one arbitrary member of a wide family of paths
its own rule admits. This round measures the size of that arbitrariness
(REBALANCE TIMING LUCK; Hoffstein, Sibears & Faber 2019, J. Index Investing
10(1)) and asks whether removing it improves the strategy, at unchanged mean
exposure and (for the conservative branch) unchanged aggregate traded
notional.

**Which constraint this attacks: N=3, primarily.** R-20's +/-0.2 Sharpe path-
noise floor has never been decomposed into "market noise" (real, irreducible)
and "implementation-phase noise" (an artifact of where in its own deadband
cycle the strategy happened to start). Every other N~3 lever in this project
is closed or costed: more instruments (R-63, breadth 1.47 of 8; B-28 blocked
on data), more BTC episodes (R-143, R-144), a different unit of observation
(R-178), forward evidence (B-06, costed at 18.9 years / never by R-78).
Secondarily COST, but as a constraint the design must NOT violate (aggregate
turnover is checked, not spent) -- that is what distinguishes this from every
closed "trade less/more" family (R-12, R-64 novel/R-130/R-165, R-131/R-133).

**Literature grounding (fetched via WebSearch before either branch is
dispatched):**

- Hoffstein, C., Sibears, J. & Faber, N. (2019), "Rebalance Timing Luck: The
  Difference Between Hired and Fired", The Journal of Index Investing 10(1),
  27-. Defines rebalance timing luck as the standard deviation of returns
  between identically managed portfolios rebalanced on different dates --
  exactly the statistic the Step-0 gate below measures on v4.
- Hoffstein, C., Faber, N. & Braun, S. (2020), "Rebalance Timing Luck: The
  (Dumb) Luck of Smart Beta", SSRN 3673910. Five long-only US equity factors,
  monthly rebalance, ~1960s-2019, no explicit cost model: timing luck often
  exceeds 100bps annualized; the prescribed fix is overlapping ("tranched")
  portfolios. Their monthly rebalance grid is COARSER than v4's continuous
  5-minute-bar deadband, so their magnitude is read here as an upper bound,
  not a forecast.
- Zarattini, C. & Pagani, A. (2025), "The Tranching Dilemma: A Cost-Aware
  Approach to Mitigate Rebalance Timing Luck in Factor Portfolios", SSRN
  5747964. US equity momentum, 1991-2024, explicit transaction costs: the
  CAGR gap between best/worst rebalance schedule reaches ~350bps; variance
  falls roughly as 1/N tranches with mean return UNCHANGED; but tranching
  normally multiplies fill count, and the paper's own caveat is that the net
  benefit is confined to investors who are not fee-sensitive at the margin.
  This is the central risk the conservative branch's turnover kill switch
  below exists to catch (single-book averaging, not N separate books).
- Jegadeesh, N. & Titman, S. (1993), "Returns to Buying Winners and Selling
  Losers", Journal of Finance 48(1), 65-91. The canonical overlapping-cohort
  construction, adopted to raise test power; no significant return
  difference between overlapping and non-overlapping portfolios was found --
  mean unchanged, noise reduced. The academic warrant for "mean-preserving,
  variance-reducing".

**Not a duplicate of** (full detail in the R-186 section-B entry):
R-40 (bagged VOTE spans -- K different signals, not K phase-offsets of one
identical signal); R-64 conservative / R-66 both branches / R-89 / R-173 /
L-05,L-06 (all change WHERE the position goes or HOW WIDE the band is, never
whether the destination depends on history); R-64 novel / R-130 / R-165
(Gaerleanu-Pedersen SMOOTH partial adjustment, ruled out for proportional
fees; this construction is a MEAN OF K DISCRETE JUMPS, each member still
snaps its full amount -- total variation is checked, not assumed, by the
turnover kill switch); R-131/R-133 (turnover corridor changes the ORDER that
follows a decision, never the decision -- here every member takes its full
decision immediately); R-93 novel (Hedge blend of K DIFFERENT mechanisms
with LEARNED weights; here weights are fixed 1/K and every member is the
byte-identical rule); R-147 (reweights the VOTE; the vote is untouched);
R-184 (audits which hyperparameter was SELECTED; this round reselects
nothing -- v4's shipped (20,40,80)/0.10/0.55/2.0 are kept exactly); R-72/B-30
and R-134/B-43 (the BROKER's separate 5%-of-max-notional deadband; avoided by
construction since the ensemble is executed as one book presenting one
full-size order stream, never K separate 1/K-sized order streams that the
broker's own deadband could absorb).

**Simulable here:** yes, with ZERO external data -- data/btcusd_spot_5m.csv.gz
only for construction and Step-0/inner splits; data/ethusd_bitfinex_5m.csv.gz
(scripts/build_bitfinex_dataset.py's frame, loaded here via `load_eth()`) for
the conservative branch's falsification test; the 0.40% fee tier
(scripts/fee_study.py's `fee_at()`, reused from r102_shared) for the novel
branch's falsification test.

**What would make it fail, named now, in likelihood order:**
(a) INERT -- members re-synchronize after most moves that exceed the deadband
    from both positions; if mean pairwise correlation rho_bar -> 1 there is
    nothing to average away. Most likely outcome; this is the Step-0 stop.
(b) REAL BUT SUB-FLOOR -- timing luck exists but is worth < +0.20 Sharpe
    (R-20's own noise floor) -- NEGATIVE regardless of sign.
(c) TURNOVER CONFOUND -- the averaged path has MORE total variation than
    v4's; any gain is bought with fees. Kill switch, not a promotion.
(d) BTC-ONLY -- present on BTC, absent/inverted on ETH -- the BTC-pass/
    ETH-invert signature this ledger has hit repeatedly (R-126, R-149, R-150,
    R-168, R-185). This round's own falsification test for exactly this.
(e) RESPONSIVENESS LOSS -- at the transition bars R-62 says carry the edge,
    the ensemble de-risks more slowly, so stress-episode drawdown worsens
    even as average volatility falls.

**Frozen splits** (identical to every prior round in this lineage):
inner-train 2017-01-01 -> 2020-12-31 (fit/debug + the Step-0 gate; not a
promotion-relevant number); inner-validation 2021-01-01 -> 2022-12-31 (all
selection, both branches); holdout 2023-01-01 -> untouched by both branches.

**Step-0 gate** (computed ONCE, on BTC inner-train ONLY, before either branch
is implemented in full -- see `step0_gate()` below): build K=8 single-phase
copies of v4's own raw-desired -> deadband quantizer (S = 144 bars = 1/2 day,
about 1/6 of v4's own measured 3.3-day fill spacing, L-05); measure S_bar =
mean Sharpe across the eight and rho_bar = mean pairwise correlation of their
daily returns; the implied ensemble volatility factor is
`v = sqrt((1 + (K-1)*rho_bar) / K)` and the implied Sharpe gain is
`dS_implied = S_bar * (1/v - 1)`. **Proceed past Step 0 only if dS_implied >=
+0.20** -- R-20's own measured noise floor, the exact noise of the comparison
the promotion bar below will run. This measurement (S_bar, rho_bar,
dS_implied) is the round's product whatever happens next: it decomposes R-20's
noise floor into market noise and implementation-phase noise for the first
time in this ledger, and is reported in section B regardless of verdict.

**Kill switches (both branches, checked on every reported comparison, via
`compare()`'s own `exposure_ratio` and this module's `turnover_ratio`):**
(i) mean |exposure| within 1% of v4's (else this has become a tracking-
    tightness change and duplicates R-64/R-66's family, not this round's);
(ii) aggregate traded notional (`turnover()`) within 5% of v4's for the
    conservative branch, within 25% for the novel branch (else any gain is
    bought with fees, duplicating the R-12 "trade less/more" family);
(iii) R^2 of the candidate's target path against v4's own target < 0.98
    (else this is a relabeling of v4 itself, not a distinct construction).

**Promotion threshold (inner-validation, both markets, both branches):** all
of (a) paired-bootstrap Delta-Sharpe vs unmodified `kelly_regime_v4` >= +0.20
with a 95% CI excluding zero, OR a >=5pp max-drawdown reduction at matched
mean exposure; (b) the branch's own pre-registered falsification test passes;
(c) the frozen K/S (conservative) or g (novel) bracket is a plateau, not a
single lucky point; (d) no sign reversal at the 0.40% fee tier.

**One deliberate, disclosed deviation from the standard `compare()` harness,
frozen now:** this mechanism is DESIGNED to lower `vol_ratio` (realized
volatility) at unchanged `exposure_ratio` -- that is the entire claim for the
conservative branch. Risk-matching is therefore judged on `exposure_ratio`
and mean time-in-market only (kill switch (i)); the realized-volatility
reduction is reported as the RESULT, never smuggled in as evidence of
matching. Stated before any number exists, per R-181's closing line about
building risk-matching into the gate's own construction.

Both branches import this module and MUST NOT modify it or commit any
changes to it -- it is the frozen reference, exactly as prior rounds'
`r102_shared.py` has served for lineages since R-102.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r102_shared import (  # noqa: E402,F401
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
    V4_ANCHOR_SPAN_DAYS,
    V4_BAND,
    V4_DEADBAND,
    V4_HORIZONS,
    V4_MAX_LEVERAGE,
    V4_TARGET_VOL,
    V4_VOL_SPAN,
    SliceResult,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_series,
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

# ---------------------------------------------------------------- constants
K_DEFAULT = 8
S_DEFAULT = BARS_PER_DAY // 2  # 144 bars = 1/2 day
G_DEFAULT = V4_DEADBAND  # primary grid = v4's own deadband


# ============================================================ (A) conservative
# K-phase overlapping ensemble of the byte-identical v4 quantizer.

def single_phase_target(df: pd.DataFrame, k: int, K: int = K_DEFAULT,
                         S: int = S_DEFAULT, deadband: float = V4_DEADBAND,
                         desired: np.ndarray | None = None) -> np.ndarray:
    """One phase-offset copy of v4's own deadband quantizer: identical to
    `apply_deadband`, except the `if |desired-pos|>deadband: pos=desired`
    check is only EVALUATED on bars where `(i // S) % K == k`; on every other
    bar the copy simply holds its previous position. Causal: `phase` depends
    only on the bar index `i`, never on future data."""
    if desired is None:
        desired = v4_raw_desired(df)
    desired = np.asarray(desired, dtype=float)
    n = len(desired)
    phase = (np.arange(n) // S) % K
    pos = 0.0
    out = np.zeros(n)
    for i in range(n):
        if phase[i] == k and abs(desired[i] - pos) > deadband:
            pos = float(desired[i])
        out[i] = pos
    return out


def phase_ensemble_target(df: pd.DataFrame, K: int = K_DEFAULT,
                           S: int = S_DEFAULT, deadband: float = V4_DEADBAND) -> np.ndarray:
    """CONSERVATIVE (Variant A): the mean of K byte-identical, phase-offset
    copies of v4's own quantizer, held as ONE book. K=1 must reproduce
    `v4_target(df)` exactly -- see `_self_test` below."""
    desired = v4_raw_desired(df)
    members = np.stack([
        single_phase_target(df, k, K=K, S=S, deadband=deadband, desired=desired)
        for k in range(K)
    ])
    return members.mean(axis=0)


# ================================================================ (B) novel
# Memoryless absolute-grid quantizer: zero timing luck BY CONSTRUCTION.

def absolute_grid_target(df: pd.DataFrame, g: float = G_DEFAULT) -> np.ndarray:
    """NOVEL (Variant B): `pos = g * round(desired / g)` -- a pure function of
    the CURRENT desired exposure, with no reference to history at all. The
    no-trade region is preserved (the position only moves when `desired`
    crosses a grid line spaced `g` apart); this is NOT the smooth trading
    rate R-64 novel/R-130/R-165 ruled out -- every move is a full snap to the
    nearest grid point, never a fractional/partial step."""
    desired = np.asarray(v4_raw_desired(df), dtype=float)
    return g * np.round(desired / g)


# ---------------------------------------------------------------- shared checks

def turnover(target: np.ndarray) -> float:
    """Sum of |delta target| across a full path -- proportional fees bill on this."""
    t = np.asarray(target, dtype=float)
    return float(np.abs(np.diff(t)).sum())


def turnover_ratio(candidate: np.ndarray, control: np.ndarray | None = None,
                    df: pd.DataFrame | None = None) -> float:
    """candidate turnover / v4 turnover, aligned on `candidate`'s own length."""
    if control is None:
        assert df is not None, "turnover_ratio needs either `control` or `df`"
        control = v4_target(df)
    n = min(len(candidate), len(control))
    c_t = turnover(np.asarray(candidate)[-n:])
    v_t = turnover(np.asarray(control)[-n:])
    return c_t / v_t if v_t else float("nan")


# ---------------------------------------------------------------- Step-0 gate

def step0_gate(K: int = K_DEFAULT, S: int = S_DEFAULT, deadband: float = V4_DEADBAND,
               btc: pd.DataFrame | None = None,
               market=None) -> dict:
    """Computed ONCE, on BTC inner-train only, before either branch is
    implemented in full. Returns S_bar, rho_bar, the implied ensemble
    volatility factor `v`, and the implied Sharpe gain `dS_implied`. The round
    proceeds past Step 0 only if `dS_implied >= 0.20`."""
    if btc is None:
        btc = load_btc()
    if market is None:
        market = SPOT
    assert_no_holdout(btc, "step0_gate")

    desired = v4_raw_desired(btc)
    results = []
    for k in range(K):
        strat = TargetStrategy(
            lambda d, _k=k: single_phase_target(d, _k, K=K, S=S, deadband=deadband),
            name=f"r186_phase{k}",
        )
        results.append(
            run_slice(strat, btc, INNER_TRAIN_START, INNER_TRAIN_END,
                      f"inner_train_phase{k}", market)
        )

    sharpes = np.array([r.sharpe for r in results])
    n = min(len(r.daily) for r in results)
    daily_matrix = np.stack([r.daily[-n:] for r in results])
    corr = np.corrcoef(daily_matrix)
    off_diag = corr[~np.eye(K, dtype=bool)]
    rho_bar = float(np.mean(off_diag))
    s_bar = float(np.mean(sharpes))
    v = float(np.sqrt((1 + (K - 1) * rho_bar) / K))
    ds_implied = float(s_bar * (1.0 / v - 1.0)) if v > 0 else float("nan")

    return {
        "K": K, "S": S, "sharpes": sharpes.tolist(), "S_bar": s_bar,
        "rho_bar": rho_bar, "v": v, "dS_implied": ds_implied,
        "gate_pass": bool(ds_implied >= 0.20),
        "_desired_len": len(desired),
    }


def _self_test() -> None:
    """K=1 identity check + causality probe, run on import."""
    btc = load_btc()
    small = btc.iloc[: 120 * BARS_PER_DAY]  # ~120 days, plenty of warmup + margin
    v4 = v4_target(small)
    ens1 = phase_ensemble_target(small, K=1, S=S_DEFAULT)
    assert np.allclose(v4, ens1), "K=1 phase ensemble must reproduce v4_target exactly"

    grid_at_deadband = absolute_grid_target(small, g=V4_DEADBAND)
    assert grid_at_deadband.shape == v4.shape

    def _phase_builder(d):
        return phase_ensemble_target(d, K=4, S=S_DEFAULT)

    def _grid_builder(d):
        return absolute_grid_target(d, g=V4_DEADBAND)

    for name, builder in (("phase_ensemble(K=4)", _phase_builder),
                           ("absolute_grid(g=0.10)", _grid_builder)):
        causal_truncation_probe_series(builder, small)  # raises AssertionError on failure

    print("r186_shared self-test: OK "
          f"(K=1 identity match, causality probes clean, "
          f"grid shape={grid_at_deadband.shape})")


if __name__ == "__main__":
    _self_test()
