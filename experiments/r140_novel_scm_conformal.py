#!/usr/bin/env python
"""R-140 NOVEL branch: Chernozhukov, Wuthrich & Zhu (2021, JASA 116(536);
arXiv:1712.09089) conformal inference for the SAME SCM point estimate the
conservative branch also uses, applied to `kelly_regime_v4`'s own edge-
concentration claim.

The complete pre-registration -- mechanism, literature, non-duplication
argument, named failure modes, and the frozen three-way decision rule --
lives in `experiments/r140_shared.py`'s module docstring, written before
either branch ran. Read that file in full first. This file imports ONLY
from `experiments.r140_shared` (read-only, never edited here), never
imports from the conservative branch's file, and never reads a bar at or
after `r140_shared.OOS_START` (2023-01-01) -- checked with an explicit
assertion on every window built below, not merely asserted in prose.

=====================================================================
STRUCTURAL DIFFERENCE FROM THE CONSERVATIVE BRANCH (Abadie in-space
placebo), stated once, precisely
=====================================================================
The conservative branch permutes WHICH UNIT is treated: for each episode
it re-fits a *fresh* SCM against each of the 6 donors in turn (each donor
relabeled "treated", the other 5 as its own donor pool), producing a
cross-sectional placebo distribution of at most 6 draws per episode.

This branch fits the target's own SCM weights EXACTLY ONCE per (market,
episode) cell -- on the pre-fit window only, via `fit_scm_weights`, never
touching the post-event window during fitting -- and then asks a
different question of the resulting RESIDUAL PATH: under an
exchangeability/stationarity assumption on the fitted residuals
`u[t] = target[t] - synthetic[t]` over the full extended window (pre-fit
period followed immediately by the event window, no gap, no refit), is
the REAL post-event-window average residual unusual relative to what a
same-length trailing window would look like if the residual path's own
block structure were rotated to other positions? This is a permutation
of TIME POSITIONS within one fixed fit, not a permutation of which
CROSS-SECTIONAL UNIT is "treated" -- structurally distinct, and, per
Cattaneo/Feng/Titiunik (2025), complementary rather than redundant.

=====================================================================
THE CWZ (2021) CONSTRUCTION, AND WHERE THIS IMPLEMENTATION DEVIATES FROM IT
=====================================================================
CWZ's own Algorithm (their Sec. 3): given a residual vector
`u = (u_1, ..., u_T)` covering both the pre- and post-treatment periods
under a hypothesized null effect, and a chosen family of permutations
`{pi_0=id, pi_1, ..., pi_{K-1}}` that leaves the joint law of `u`
invariant under the null (exact for i.i.d. residuals using ALL T cyclic
shifts by 1; for serially-dependent residuals they note a MOVING-BLOCK
version that permutes CONTIGUOUS BLOCKS to preserve within-block
dependence, at the cost of using fewer, coarser permutations), the
p-value is `p = (1/K) * sum_k 1{S(pi_k(u)) >= S(u)}`, where `S` is a
test statistic evaluated on the *last* `T1` (post-period-length) entries
of the permuted vector. Because `pi_0 = id` is always included in the
sum, `p >= 1/K` always holds and the test is exact (not merely
asymptotic) under the stated exchangeability assumption -- this is the
entire point of the construction, and why NO add-one smoothing is used
below (unlike R-138's conservative-cross-sectional convention, which
draws random pseudo-samples rather than enumerating a permutation group
and so needs add-one smoothing for a proper i.e. never-zero p-value;
this test enumerates the full permutation group by construction, so
smoothing would only make the reported p-value more conservative than
the exact quantity it computes -- disclosed, not applied).

Implementation choices made here, each disclosed because the
pre-registration does not fix it:

1. **Block length.** Chosen PER CELL (per market x episode), from that
   cell's OWN fitted residual path, via its empirical autocorrelation:
   the smallest lag at which |ACF| first drops below the
   white-noise 95% band `1.96/sqrt(T)`, floored at 2 (a deliberately
   conservative floor: an insignificant lag-1 ACF alone does not rule
   out weak real dependence in a short series, and floor=1 would reduce
   to the i.i.d. case CWZ flag as the SPECIAL case, not the general
   one) and capped at `T // 8` (so at least 8 rotations remain, however
   short the series). Reported per cell below, not chosen once and
   hidden.
2. **Rotation family.** Moving-BLOCK circular rotations: shift the whole
   `T`-length residual vector by `k * block_length` positions
   (`k = 0 .. K-1`, `K = T // block_length`), treating the extended
   window as a ring. This is the block generalization of CWZ's own
   Algorithm 1 for dependent residuals (their Remark on serial
   correlation), not an ad hoc choice.
3. **Test statistic.** `S(u) = mean(u over the trailing WINDOW_PRE_DAYS +
   WINDOW_POST_DAYS + 1 = 26 entries)` -- literally "the post-event-
   window average residual" the task specifies, matching
   `r140_shared.event_gap`'s window convention exactly (same [event -
   WINDOW_PRE_DAYS, event + WINDOW_POST_DAYS] span) but reporting a MEAN
   rather than `event_gap`'s SUM, because the null distribution here is
   built from trailing windows of donors/rotations with the identical
   fixed length, so mean and sum rank identically -- mean is reported
   because it is on the same per-day scale as the residuals themselves.
4. **Two-sided vs one-sided p-value.** BOTH are computed and reported
   per cell. The gate check against `BTC_P_GATE`/`ETH_P_GATE` uses the
   TWO-SIDED p-value (`|S(pi_k(u))| >= |S(u)|`), for direct comparability
   with R-138's own convention (`r138_shared.permutation_test`, also
   two-sided on an absolute-value statistic) -- disclosed as a deliberate
   choice for cross-tool comparability, not a default.
5. **Pooling across the 4 episodes.** Fisher's (1932) combined-probability
   test on the 4 episodes' two-sided p-values:
   `chi2 = -2 * sum(ln p_i)`, `df = 2*4 = 8`, evaluated with the exact
   closed-form survival function for even-df chi-square (no scipy
   dependency, matching `r140_shared.py`'s own no-new-dependency
   convention: `S(x; 2k) = exp(-x/2) * sum_{j=0}^{k-1} (x/2)^j / j!`,
   the standard Gamma/Erlang identity, cross-checked against a 2M-draw
   Monte Carlo in scratch work before being trusted here). Fisher's
   method is chosen explicitly OVER Stouffer's Z (the other standard
   choice) because Stouffer's requires committing to a single common
   sign across the 4 episodes, and the operator's own smoke-test note
   in the dispatch ("event gaps ... varied in sign across episodes")
   already rules that out before any number in this file was computed --
   using Stouffer's after seeing that note would be picking the pooling
   rule to fit the data. Fisher's omnibus test needs no such assumption.
6. **Placebo sanity check (#7 in the task).** One donor ticker, chosen
   by an arbitrary PRE-STATED rule (the first column of the panel as
   returned by `load_donor_daily_returns`, i.e. whichever ticker sorts
   first in the already-committed R-57 panel order -- not selected by
   looking at any result), is treated as a fake "target" against the
   remaining 5 donors, run through the IDENTICAL fit + conformal
   procedure across the 4 BTC-window episodes. This checks the
   MACHINERY, not the strategy: a block-permutation test that is
   trivially always-significant or always-inert on an arbitrary donor's
   own return path would invalidate the real reads regardless of what
   they show.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments import r140_shared as shared  # noqa: E402

OOS_START_TS = pd.Timestamp(shared.OOS_START, tz="UTC")

N_CONFIGS = 0  # incremented as each (target, episode) cell is evaluated


# ------------------------------------------------------- block length


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    denom = float(np.sum(x**2))
    out = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        num = float(np.sum(x[: n - lag] * x[lag:]))
        out[lag - 1] = num / denom if denom > 0 else 0.0
    return out


def choose_block_length(u: np.ndarray) -> tuple[int, np.ndarray, float]:
    """Smallest lag at which |ACF| first drops below the white-noise 95%
    band `1.96/sqrt(T)`, floored at 2, capped at `T // 8`. Returns
    (block_length, acf_values, threshold) -- all three reported per cell,
    not hidden inside a single number."""
    n = len(u)
    max_lag = max(1, min(30, n // 4))
    a = _acf(u, max_lag)
    thresh = 1.96 / np.sqrt(n)
    lag0 = None
    for lag, val in enumerate(a, start=1):
        if abs(val) < thresh:
            lag0 = lag
            break
    if lag0 is None:
        lag0 = max_lag
    cap = n // 8
    block_len = max(2, lag0)
    if cap >= 2:
        block_len = min(block_len, cap)
    return int(block_len), a, float(thresh)


# ------------------------------------------------------- conformal test


def conformal_block_permutation(u: np.ndarray, n_post: int, block_length: int) -> dict:
    """CWZ (2021)-style moving-block circular-rotation conformal test.
    `u` is the FULL extended-window residual vector (pre-fit followed
    immediately by the event window, chronological order); the real
    post-event-window statistic is the mean of the trailing `n_post`
    entries. Returns the observed statistic, two-sided and one-sided
    (direction-matched) p-values, K (number of rotations), and the null
    array itself for inspection."""
    T = len(u)
    K = T // block_length
    assert K >= 2, f"degenerate rotation family: T={T} block_length={block_length}"
    stats = np.empty(K)
    for k in range(K):
        shifted = np.roll(u, k * block_length)
        stats[k] = float(np.mean(shifted[-n_post:]))
    observed = stats[0]
    two_sided_p = float(np.mean(np.abs(stats) >= abs(observed) - 1e-15))
    if observed > 0:
        one_sided_p = float(np.mean(stats >= observed - 1e-15))
    elif observed < 0:
        one_sided_p = float(np.mean(stats <= observed + 1e-15))
    else:
        one_sided_p = 1.0
    return {
        "observed": observed,
        "K": K,
        "two_sided_p": two_sided_p,
        "one_sided_p": one_sided_p,
        "null_stats": stats,
    }


def fisher_combined_pvalue(pvalues: list) -> tuple[float, float]:
    """Fisher's (1932) combined-probability test, closed-form even-df
    chi-square survival function (no scipy dependency). Returns
    (chi2_statistic, pooled_pvalue)."""
    k = len(pvalues)
    x = -2.0 * sum(math.log(p) for p in pvalues)
    s = sum((x / 2.0) ** j / math.factorial(j) for j in range(k))
    p_pooled = math.exp(-x / 2.0) * s
    return x, min(1.0, max(0.0, p_pooled))


# ------------------------------------------------------- per-cell pipeline


def episode_windows(event_date_str: str) -> tuple[pd.Timestamp, ...]:
    """(fit_start, fit_end, ew_start, ew_end) for one episode, exactly the
    convention `PRE_FIT_DAYS`'s own docstring describes: the fit window is
    the `PRE_FIT_DAYS` immediately preceding the event window's own start,
    with no overlap. Reproduces the operator's own smoke-test RMSPE ratios
    bit-for-bit (verified in scratch work before writing this file)."""
    event_date = pd.Timestamp(event_date_str, tz="UTC")
    ew_start = event_date - pd.Timedelta(days=shared.WINDOW_PRE_DAYS)
    ew_end = event_date + pd.Timedelta(days=shared.WINDOW_POST_DAYS)
    fit_end = ew_start - pd.Timedelta(days=1)
    fit_start = fit_end - pd.Timedelta(days=shared.PRE_FIT_DAYS - 1)
    return fit_start, fit_end, ew_start, ew_end


def run_cell(target: pd.Series, donors: pd.DataFrame, episode_name: str,
             event_date_str: str) -> dict:
    """One (target, episode) cell: Step-A validity diagnostic, single SCM
    fit on the pre-fit window, residual path over the extended window,
    block-permutation conformal test. Never reads a bar >= OOS_START."""
    global N_CONFIGS
    fit_start, fit_end, ew_start, ew_end = episode_windows(event_date_str)
    assert ew_end < OOS_START_TS, f"{episode_name}: event window reaches OOS_START"

    # --- Step A: pre-fit validity diagnostic (shared, unmodified) -----
    ratio, target_rmspe, median_donor_rmspe = shared.pre_fit_rmspe_ratio(
        donors, target, fit_start, fit_end
    )

    # --- single fixed fit, pre-fit window only -------------------------
    weights = shared.fit_scm_weights(donors, target, fit_start, fit_end)

    # --- residual path over the FULL extended window (pre-fit + event) -
    idx = donors.index.intersection(target.index)
    idx = idx[(idx >= fit_start) & (idx <= ew_end)]
    assert idx.max() < OOS_START_TS, f"{episode_name}: extended window reaches OOS_START"
    synth = shared.synthetic_path(donors, weights, idx)
    u = (target.loc[idx] - synth).to_numpy()
    post_mask = np.asarray(idx >= ew_start)
    n_pre = int((~post_mask).sum())
    n_post = int(post_mask.sum())

    # `event_gap` reused verbatim for the standard cumulative-gap artifact
    # (a SUM over the event window), reported alongside but not used by
    # the conformal test itself (which uses a MEAN, see module docstring).
    gap_cumulative = shared.event_gap(target, synth, pd.Timestamp(event_date_str, tz="UTC"))

    block_length, acf_vals, acf_thresh = choose_block_length(u)
    conformal = conformal_block_permutation(u, n_post, block_length)
    N_CONFIGS += 1

    return {
        "episode": episode_name,
        "event_date": event_date_str,
        "fit_start": fit_start, "fit_end": fit_end,
        "ew_start": ew_start, "ew_end": ew_end,
        "ratio": ratio, "target_rmspe": target_rmspe,
        "median_donor_rmspe": median_donor_rmspe,
        "T": len(u), "n_pre": n_pre, "n_post": n_post,
        "block_length": block_length, "acf_lag1": float(acf_vals[0]) if len(acf_vals) else float("nan"),
        "acf_thresh": acf_thresh,
        "gap_cumulative": gap_cumulative,
        "conformal": conformal,
    }


def run_market_or_placebo(target: pd.Series, donors: pd.DataFrame, label: str) -> list:
    print(f"\n{'=' * 78}\n  {label}\n{'=' * 78}")
    results = []
    for episode_name, event_date_str in shared.DONOR_COVERED_EPISODES:
        r = run_cell(target, donors, episode_name, event_date_str)
        c = r["conformal"]
        print(
            f"{episode_name:38s} ratio={r['ratio']:.3f} "
            f"[T={r['T']:3d} n_pre={r['n_pre']:3d} n_post={r['n_post']:2d} "
            f"block_len={r['block_length']} K={c['K']:3d}]  "
            f"gap_cum={r['gap_cumulative']:+.5f} "
            f"S_obs={c['observed']:+.6f}  "
            f"p_two_sided={c['two_sided_p']:.4f} p_one_sided={c['one_sided_p']:.4f}"
        )
        results.append(r)
    return results


# ------------------------------------------------------- main


def main() -> None:
    print("R-140 NOVEL: CWZ (2021) conformal / block-permutation SCM inference")
    print(f"OOS_START = {shared.OOS_START} -- verified never read below (explicit "
          f"assertions on every window; see per-cell prints, all event windows end "
          f"well before this date since all 4 DONOR_COVERED_EPISODES sit in "
          f"2020-03..2022-11-28 and PRIMARY_MARKET/target series are pre-truncated "
          f"to INNER_VAL_END={shared.INNER_VAL_END} by "
          f"`load_v4_and_extended_donor_returns`).")

    # ------------------------------------------------------------ BTC ---
    btc_excess, donors = shared.load_v4_and_extended_donor_returns("btc", shared.PRIMARY_MARKET)
    btc_results = run_market_or_placebo(btc_excess, donors, "BTC (primary)")

    # --- Step-A gate: >= 3/4 BTC episodes failing RMSPE_GATE -> STOP ----
    btc_fail_count = sum(1 for r in btc_results if r["ratio"] > shared.RMSPE_GATE)
    print(f"\nStep-A gate (BTC): {btc_fail_count}/4 episodes exceed "
          f"RMSPE_GATE={shared.RMSPE_GATE}")

    if btc_fail_count >= 3:
        print("\n" + "=" * 78)
        print("VERDICT: INVALID (Step-A stop) -- majority of BTC episodes fail the "
              "pre-fit validity gate. No p-value computed or reported past this "
              "point per the frozen decision rule.")
        print("=" * 78)
        print(f"\nConfigurations evaluated: {N_CONFIGS}")
        return

    # --- pooled BTC significance: Fisher's combined-probability test ---
    btc_two_sided_ps = [r["conformal"]["two_sided_p"] for r in btc_results]
    btc_chi2, btc_pooled_p = fisher_combined_pvalue(btc_two_sided_ps)
    print(f"\nBTC pooled (Fisher, df=8): per-episode two-sided p = "
          f"{[f'{p:.4f}' for p in btc_two_sided_ps]}, "
          f"chi2={btc_chi2:.4f}, pooled p={btc_pooled_p:.5f} "
          f"(gate: < {shared.BTC_P_GATE})")
    btc_significant = btc_pooled_p < shared.BTC_P_GATE

    # ------------------------------------------------------------ ETH ---
    eth_excess, eth_donors = shared.load_v4_and_extended_donor_returns("eth", shared.PRIMARY_MARKET)
    eth_results = run_market_or_placebo(eth_excess, eth_donors, "ETH (falsification/replication)")

    eth_fail_count = sum(1 for r in eth_results if r["ratio"] > shared.RMSPE_GATE)
    print(f"\nStep-A gate (ETH, reported for completeness -- the frozen rule only "
          f"stops on BTC failing): {eth_fail_count}/4 episodes exceed "
          f"RMSPE_GATE={shared.RMSPE_GATE}")

    eth_two_sided_ps = [r["conformal"]["two_sided_p"] for r in eth_results]
    eth_chi2, eth_pooled_p = fisher_combined_pvalue(eth_two_sided_ps)
    print(f"\nETH pooled (Fisher, df=8): per-episode two-sided p = "
          f"{[f'{p:.4f}' for p in eth_two_sided_ps]}, "
          f"chi2={eth_chi2:.4f}, pooled p={eth_pooled_p:.5f} "
          f"(gate: < {shared.ETH_P_GATE})")

    # same-sign check (BTC pooled observed sign vs ETH pooled observed sign),
    # via the sum of each market's observed per-episode statistics (sign of
    # the pooled directional effect, since Fisher's test itself is sign-blind)
    btc_sign = np.sign(sum(r["conformal"]["observed"] for r in btc_results))
    eth_sign = np.sign(sum(r["conformal"]["observed"] for r in eth_results))
    same_sign = bool(btc_sign != 0 and btc_sign == eth_sign)
    print(f"\nSign check: BTC pooled-direction sign={btc_sign:+.0f}, "
          f"ETH pooled-direction sign={eth_sign:+.0f}, same_sign={same_sign}")

    eth_replicates = eth_pooled_p < shared.ETH_P_GATE and same_sign

    # ------------------------------------------------------- decision ---
    print(f"\n{'=' * 78}")
    if btc_significant and eth_replicates:
        verdict = "VALID & CONFIRMS"
    else:
        verdict = "VALID & DOES NOT CONFIRM"
    print(f"VERDICT: {verdict}")
    print(f"  BTC pooled p={btc_pooled_p:.5f} {'<' if btc_significant else '>='} "
          f"{shared.BTC_P_GATE} -> {'PASS' if btc_significant else 'FAIL'}")
    print(f"  ETH pooled p={eth_pooled_p:.5f} {'<' if eth_pooled_p < shared.ETH_P_GATE else '>='} "
          f"{shared.ETH_P_GATE} AND same_sign={same_sign} -> "
          f"{'PASS' if eth_replicates else 'FAIL'}")
    print("=" * 78)

    # -------------------------------------------- placebo sanity check --
    print(f"\n{'-' * 78}")
    print("PLACEBO SANITY CHECK (task item #7): one donor treated as a fake "
          "'target' against the remaining 5 donors, identical procedure, BTC "
          "episode windows. Tests the MACHINERY, not the strategy.")
    print(f"{'-' * 78}")
    placebo_ticker = sorted(donors.columns)[0]
    print(f"Placebo target ticker (pre-stated rule: first column, sorted, of the "
          f"already-committed R-57 panel): {placebo_ticker}")
    placebo_target = donors[placebo_ticker]
    placebo_donors = donors.drop(columns=[placebo_ticker])
    placebo_results = run_market_or_placebo(placebo_target, placebo_donors,
                                             f"PLACEBO target={placebo_ticker}")
    placebo_two_sided_ps = [r["conformal"]["two_sided_p"] for r in placebo_results]
    placebo_reject_count = sum(1 for p in placebo_two_sided_ps if p < shared.BTC_P_GATE)
    print(f"\nPlacebo two-sided p-values: {[f'{p:.4f}' for p in placebo_two_sided_ps]}")
    print(f"Placebo episodes rejecting at BTC_P_GATE={shared.BTC_P_GATE}: "
          f"{placebo_reject_count}/4")
    if placebo_reject_count >= 3:
        print("CAUTION: machinery rejects on an arbitrary donor's own return path in "
              "a majority of episodes -- suggests the test may be miscalibrated "
              "(too liberal) and the real reads above should be treated with more "
              "skepticism than the raw p-values suggest.")
    elif placebo_reject_count == 0:
        print("Machinery does not reject on an arbitrary donor's own return path in "
              "any episode -- consistent with (not proof of) reasonable calibration; "
              "does not by itself rule out the test being too conservative (inert).")
    else:
        print("Machinery rejects on 1-2/4 placebo episodes -- within the range "
              "expected for a two-sided test at p<0.10 run on 4 non-independent "
              "windows of one donor's own noisy return path; not evidence of "
              "gross miscalibration in either direction.")

    print(f"\nConfigurations evaluated: {N_CONFIGS} "
          f"(4 BTC episodes + 4 ETH episodes + 4 placebo episodes = 12, one frozen "
          f"procedure -- block-length rule, rotation family, statistic, and pooling "
          f"method were all fixed before any p-value was computed; no sweep, no "
          f"alternative statistic tried and discarded)")


if __name__ == "__main__":
    main()
