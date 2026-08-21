"""R-79 (novel branch): halving-cycle phase as a phase-informed volatility
ANCHOR for kelly_regime_v3/v4's conditional-volatility-target `scale`
factor. Attacks INFO on the SIZE axis (not the directional vote).

=====================================================================
WHAT THIS FILE IS
=====================================================================

Direction: R-79's novel branch. Six external/derived INFO signals have
already failed on this strategy family (on-chain B-07/R-44, macro
VIX/DXY R-53/R-54, stablecoin flow R-54/R-55/R-58, DVOL/VRP R-73, MVRV
R-74, day-of-week/session timing R-75 -- docs/LEDGER.md section C).
None used Bitcoin's block-reward halving schedule (grepped: zero hits
for "halving"/"stock-to-flow"/"cycle phase" in the ledger before this
round -- confirmed by this session, see report). Halving dates are
public, deterministic, known years in advance: zero external fetch,
zero coverage/staleness risk, unlike DVOL (2021+ only) or MVRV.

This branch tests whether REALIZED VOLATILITY (not returns -- that is
the conservative branch's disjoint axis, not read or coordinated with
here) is systematically phase-dependent, per the "post-halving
compression, pre-top expansion, bear-phase compression" pattern in
industry commentary and partially evidenced in:
  - Gatsios et al. (2025, JRFM 18(5):242, "Is Bitcoin's Market
    Maturing?") -- documents VOLATILITY (not just returns) varying
    across the 2012/2016/2020/2024 halving events, shrinking in
    amplitude across cycles (a market-maturation reading). The most
    direct citation for this round's specific claim.
  - Lim, B.C. (2026, SSRN 6589402) -- documents a structural break in
    cycle four's RETURN pattern (sign flip vs cycle three). Cited as a
    caution that ANY halving-cycle-conditioned claim (this one
    included) risks not generalizing across cycles, even though the
    target variable here (volatility) differs from his (returns).
  - Bongaerts, Kang & van Dijk (2020, FAJ 76(4)) -- the "conditional
    volatility targeting beats continuous targeting" literature
    kelly_regime_v3 already rests on (see its own docstring); this
    round asks whether the *anchor* that conditional targeting
    compares "current" volatility against can be improved with a
    phase-informed expectation rather than a pure trailing window.

Mechanism (see kelly_regime_v3.py): the strategy already computes a
FAST realized-vol estimate (`vol`, ~8-day EWM, annualized) and compares
it to a SLOW trailing "normal volatility" anchor (`slow`, a 180-day EWM
of that same fast vol) via `ratio = vol/slow` to decide whether it is
in a normal band (size off `slow`) or a breakout (size off `vol`
directly). That `slow` anchor is purely backward-looking and has no
way to anticipate a cycle phase the strategy has not yet lived through.
If halving phase predicts *where realized vol is likely to sit*, a
phase-informed correction to the anchor could be added -- IF Step A
below finds the underlying pattern is real and not a generic
few-year-trend artifact.

A NAMED REASON THIS BRANCH MIGHT FAIL EVEN IF STEP A FINDS A REAL
PATTERN: R-62 (docs/LEDGER.md, grep "### R-62") isolated
kelly_regime_v4's conditional-volatility-target `scale` factor ALONE
(vote deleted, frac==1.0 always) across the 8-instrument panel and
found it reproduces NEITHER of the strategy's two headline properties
(matched-exposure drawdown advantage: 0/6 panel, inverts on the BTC/ETH
control; return timing: also absent) -- the vote carries essentially
the entire signature, and the SIZE-axis factor alone carries almost
nothing measurable. This is prior, on-topic, real evidence that a
SIZE-axis-only intervention on this family is a structurally weak
lever REGARDLESS of whether the underlying cycle-volatility pattern is
real. If Step B is reached and returns a null result on the strategy
composition, that is CONSISTENT WITH R-62's prior finding, not new
evidence against the volatility-phase hypothesis itself -- which is
exactly why this file's falsification battery (if Step B is reached)
includes a feature-level diagnostic independent of composing into v4,
to tell the two explanations apart.

OTHER NAMED FAILURE MODES (before any number is computed):
  (b) EMH pre-pricing -- weaker here than for the conservative branch's
      RETURN claim, since EMH says less about volatility forecastability
      than about return forecastability, but still a live prior against
      this branch, stated honestly rather than dismissed.
  (c) generic multi-year-trend confound -- ANY partition of a period
      containing a bull-then-bear cycle shows SOME volatility dispersion
      by pure construction (bear markets are more volatile than steady
      bulls on average, independent of any halving-specific mechanism).
      This is why Step A uses a placebo-offset null (arbitrary ~4-year
      partitions with the SAME inter-event spacing statistics as the
      real halving schedule) rather than a naive block-bootstrap, which
      would not control for this confound.

Data discipline: imports experiments/r79_shared.py READ-ONLY (per the
brief, not modified, not forked). OOS_START = 2023-01-01 is never
approached; assert_no_holdout() is checked at every load site, mirroring
experiments/r75_novel_session_vol_gate.py's pattern. TRAIN =
(2017-01-01, 2020-12-31) for Step A and, if reached, Step B's sweep.
VALID = (2021-01-01, 2022-12-31) for cross-cycle replication, feature
diagnostics, and candidate selection only.

POWER LIMITATION (stated per r79_shared.py's own docstring, before any
number is computed): pre-holdout coverage is AT MOST 1 fully-observed
inter-halving cycle (the tail of 2016-07-09->2020-05-11, months ~6-45,
observed via inner-train) plus a badly-truncated partial second cycle
(2020-05-11->2024-04-20, months ~8-31 only, observed via
inner-validation) -- worse than DVOL's n=3 stress episodes (R-73).
Any cross-cycle replication claim here is built on effectively 2
partially-observed cycles and should be read as low-power evidence,
not a confident generalization, regardless of which way it comes out.

Step A procedure (frozen before any number was computed -- this exact
ordering is what runs in main()):
  1. BTC inner-train (2017-01-01 -> 2020-12-31) realized vol (same
     rolling construction kelly_regime_v3/v4 use internally: EWM
     span=8*BARS_PER_DAY, annualized by sqrt(BARS_PER_YEAR), shift(1)
     for causality) conditioned on phase_bucket(index, n_buckets=4).
     Dispersion statistic = population std of the 4 conditional
     mean-volatility values.
  2. Placebo-offset null: 500 placebo reference dates from
     r79_shared.placebo_reference_dates(n=500, seed=179) (seed 179,
     deliberately different from the conservative branch's seed 79, so
     the two branches' placebo draws are independent; fixed once,
     never redrawn). For each, bucket inner-train bars by
     r79_shared.fake_cycle_phase() bracketed off that reference date,
     compute the identical dispersion statistic. True-halving
     dispersion must clear the 95th percentile of this null.
  3. Cross-cycle replication: rank the phase buckets by mean vol using
     ONLY the inner-train-observable portion of cycle 1 (BTC bars
     2017-01-01 -> 2020-05-10, i.e. before the 2020-05-11 halving),
     independently using ONLY the inner-validation-observable portion
     of cycle 2 (BTC bars 2021-01-01 -> 2022-12-31, entirely inside
     the 2020-05-11 -> 2024-04-20 cycle). Report which buckets each
     window actually observes and the Spearman rank correlation
     restricted to buckets both windows observe. ETH is explicitly NOT
     a valid replication check here (no comparable halving schedule;
     ETH moved to proof-of-stake Sept 2022).

Pre-registered stop rule (fixed now, before any number is computed):
proceed to Step B ONLY IF BOTH (a) true-halving dispersion clears the
placebo null's 95th percentile, AND (b) cross-cycle Spearman rank
correlation on shared buckets is >= 0.5. If either fails, STOP -- report
as a clean negative with the gate as the product. This bar is not
relaxed after seeing the numbers.

Run: ``python experiments/r79_novel_halving_phase_volatility.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_dataset  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
import r79_shared as shared  # noqa: E402

OOS_START = "2023-01-01"
TRAIN = ("2017-01-01", "2020-12-31 23:55:00")
VALID = ("2021-01-01", "2022-12-31 23:55:00")
H2 = pd.Timestamp("2020-05-11", tz="UTC")  # third halving: cycle-1/cycle-2 boundary

SEED_PLACEBO = 179  # deliberately different from the conservative branch's 79
N_PLACEBO = 500

# same default as KellyRegime/KellyRegimeV3 -- the EXACT construction the
# strategy consumes internally, not a reinvented estimator.
VOL_SPAN = 8 * BARS_PER_DAY


def assert_no_holdout(df: pd.DataFrame) -> None:
    """Mirrors experiments/r75_novel_session_vol_gate.py's guard."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


# ============================================================ realized vol

def realized_vol_v3(df: pd.DataFrame, vol_span: int = VOL_SPAN) -> pd.Series:
    """Byte-for-byte the `vol` line inside KellyRegime.prepare() /
    KellyRegimeV3.prepare(): EWM std of log returns, span=vol_span,
    min_periods=BARS_PER_DAY, annualized by sqrt(BARS_PER_YEAR), shifted
    by 1 bar so row i never uses its own bar's return (causal)."""
    r = np.log(df["close"]).diff()
    vol = (r.ewm(span=vol_span, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1)
    return vol


def dispersion_stat(bucket_means: np.ndarray) -> float:
    """Population std (ddof=0) of the per-bucket mean-volatility values."""
    return float(np.std(bucket_means, ddof=0))


def bucket_means(vol: pd.Series, bucket: pd.Series, n_buckets: int = 4) -> pd.Series:
    """Mean vol per bucket 0..n_buckets-1 (NaN for an unobserved bucket)."""
    df = pd.DataFrame({"vol": vol.to_numpy(), "bucket": bucket.to_numpy()})
    df = df.dropna()
    means = df.groupby("bucket")["vol"].mean()
    return means.reindex(range(n_buckets))


def fast_fake_bucket(index: pd.DatetimeIndex, anchor: pd.Timestamp,
                      gap_days: float = 1406.0, n_buckets: int = 4) -> np.ndarray:
    """Vectorized reimplementation of r79_shared.fake_cycle_phase + local
    bucketing, numerically identical to calling
    ``local_bucket_from_phase(shared.fake_cycle_phase(index, anchor, gap_days))``
    but O(n) in numpy rather than a per-timestamp Python loop -- needed
    because the placebo null calls this 500 times over ~420k bars each
    (210M row-iterations), which is intractable at Python-loop speed.
    r79_shared.py itself is not modified; this is a new, independently
    written function local to this file per the brief's "add a new
    function here rather than forking or editing existing functions"
    guidance, verified bit-identical against the reference implementation
    below via the __main__ self-check.

    Math: fake_bracket_fn's bracket(ts) = (anchor + n*gap, anchor + (n+1)*gap)
    where n = floor((ts-anchor)/gap); fake_cycle_phase = (ts-lo)/gap, which
    is exactly the fractional part of (ts-anchor)/gap (always in [0,1) for
    the floor-based n above, matching shared.fake_cycle_phase exactly).
    """
    idx = pd.DatetimeIndex(index)
    idx_naive = idx.tz_convert(None) if idx.tz is not None else idx
    anchor_naive = anchor.tz_convert(None) if anchor.tz is not None else anchor
    gap_ns = gap_days * 86400 * 1e9
    delta_ns = (idx_naive.values - anchor_naive.to_datetime64()) / np.timedelta64(1, "ns")
    ratio = delta_ns / gap_ns
    phase = ratio - np.floor(ratio)  # fractional part, matches fake_cycle_phase
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    return np.clip(np.digitize(phase, edges[1:-1]), 0, n_buckets - 1)


def local_bucket_from_phase(phase: pd.Series, n_buckets: int = 4) -> pd.Series:
    """Same bucketing rule as r79_shared.phase_bucket, applied to a
    pre-computed phase series (needed for the placebo null, since
    r79_shared has no fake_phase_bucket helper and must not be forked
    or edited -- this local reimplementation of the *bucketing math
    only* keeps the shared file untouched)."""
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    b = np.clip(np.digitize(phase.to_numpy(), edges[1:-1]), 0, n_buckets - 1)
    return pd.Series(b, index=phase.index, name="bucket")


def spearman_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation via Pearson-on-ranks, same construction R-75's
    novel branch used (np.corrcoef of .rank())."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if len(ra) < 2 or np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    print("=" * 78)
    print("R-79 novel branch: Step A measurement gate")
    print("(halving-cycle phase vs. realized volatility)")
    print("=" * 78)

    # ---- 0. load, restrict to pre-holdout, guard ----
    btc_full, btc_label = load_dataset(ROOT / "data", "spot")
    btc_full = btc_full.loc[:VALID[1]]  # never touch anything >= OOS_START
    assert_no_holdout(btc_full)
    print(f"\nBTC loaded (label={btc_label}): {btc_full.index.min()} -> "
          f"{btc_full.index.max()} ({len(btc_full):,} bars)")

    vol_full = realized_vol_v3(btc_full)

    btc_train = btc_full.loc[TRAIN[0]:TRAIN[1]]
    assert_no_holdout(btc_train)
    vol_train = vol_full.loc[btc_train.index]
    print(f"Inner-train: {btc_train.index.min()} -> {btc_train.index.max()} "
          f"({len(btc_train):,} bars)")

    # ================================================== Step A.1: true stat
    phase_bucket_train = shared.phase_bucket(btc_train.index, n_buckets=4)
    true_means = bucket_means(vol_train, phase_bucket_train, n_buckets=4)
    true_disp = dispersion_stat(true_means.to_numpy())
    print("\n--- Step A.1: true halving-phase conditional mean volatility ---")
    print(true_means.to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"Bucket bar counts: "
          f"{phase_bucket_train.value_counts().sort_index().to_dict()}")
    print(f"True dispersion statistic (population std of 4 bucket means): "
          f"{true_disp:.6f}")

    # ================================================== Step A.2: placebo null
    print(f"\n--- Step A.2: placebo-offset null "
          f"(n={N_PLACEBO}, seed={SEED_PLACEBO}) ---")
    anchors = shared.placebo_reference_dates(n=N_PLACEBO, seed=SEED_PLACEBO)

    # verify the fast vectorized bucketer against the reference (slow,
    # per-shared.py) implementation on a small sample before trusting it
    # for all 500 draws -- guards against a silent math error in the
    # vectorization above.
    _check_idx = btc_train.index[::5000]  # ~85 timestamps, cheap
    _ref_phase = shared.fake_cycle_phase(_check_idx, anchors[0])
    _ref_bucket = local_bucket_from_phase(_ref_phase, n_buckets=4).to_numpy()
    _fast_bucket = fast_fake_bucket(_check_idx, anchors[0])
    assert np.array_equal(_ref_bucket, _fast_bucket), (
        "fast_fake_bucket disagrees with the reference r79_shared.fake_cycle_phase "
        "+ local_bucket_from_phase path -- vectorization bug, not safe to use")
    print(f"Vectorized fast_fake_bucket verified bit-identical to the reference "
          f"r79_shared.fake_cycle_phase path on {len(_check_idx)} sample timestamps.")

    vol_train_np = vol_train.to_numpy()
    null_stats = np.empty(N_PLACEBO)
    for i, anchor in enumerate(anchors):
        fake_bucket = fast_fake_bucket(btc_train.index, anchor)
        df_tmp = pd.DataFrame({"vol": vol_train_np, "bucket": fake_bucket}).dropna()
        m = df_tmp.groupby("bucket")["vol"].mean().reindex(range(4)).dropna()
        null_stats[i] = dispersion_stat(m.to_numpy())
    p95 = float(np.percentile(null_stats, 95))
    p99 = float(np.percentile(null_stats, 99))
    pval = float((null_stats >= true_disp).mean())
    print(f"Null dispersion: mean={null_stats.mean():.6f} std={null_stats.std():.6f} "
          f"p95={p95:.6f} p99={p99:.6f}")
    print(f"True dispersion {true_disp:.6f} -> "
          f"{'EXCEEDS' if true_disp > p95 else 'does NOT exceed'} p95, "
          f"empirical p-value={pval:.4f}")
    criterion_a = true_disp > p95

    # ================================================== Step A.3: cross-cycle
    print("\n--- Step A.3: cross-cycle replication ---")
    cycle1 = btc_train.loc[btc_train.index < H2]
    print(f"Cycle-1-observed subset (inner-train, before {H2.date()}): "
          f"{cycle1.index.min()} -> {cycle1.index.max()} ({len(cycle1):,} bars)")
    vol_cycle1 = vol_full.loc[cycle1.index]
    bucket_cycle1 = shared.phase_bucket(cycle1.index, n_buckets=4)
    means_cycle1 = bucket_means(vol_cycle1, bucket_cycle1, n_buckets=4)
    print("Cycle-1 bucket means:")
    print(means_cycle1.to_string(float_format=lambda x: f"{x:.6f}"))

    btc_valid = btc_full.loc[VALID[0]:VALID[1]]
    assert_no_holdout(btc_valid)
    assert btc_valid.index.min() >= H2, (
        "inner-validation window must sit entirely inside cycle 2 for this "
        "replication design to be valid")
    print(f"\nCycle-2-observed subset (inner-validation): "
          f"{btc_valid.index.min()} -> {btc_valid.index.max()} "
          f"({len(btc_valid):,} bars)")
    vol_cycle2 = vol_full.loc[btc_valid.index]
    bucket_cycle2 = shared.phase_bucket(btc_valid.index, n_buckets=4)
    means_cycle2 = bucket_means(vol_cycle2, bucket_cycle2, n_buckets=4)
    print("Cycle-2 bucket means:")
    print(means_cycle2.to_string(float_format=lambda x: f"{x:.6f}"))

    shared_buckets = sorted(
        set(means_cycle1.dropna().index) & set(means_cycle2.dropna().index))
    print(f"\nBuckets observed by BOTH windows: {shared_buckets}")
    print(f"Cycle-1-only buckets (unobserved by inner-validation): "
          f"{sorted(set(means_cycle1.dropna().index) - set(shared_buckets))}")
    print(f"Cycle-2-only buckets (unobserved by inner-train): "
          f"{sorted(set(means_cycle2.dropna().index) - set(shared_buckets))}")

    if len(shared_buckets) >= 2:
        rank_corr = spearman_rank_corr(
            means_cycle1.loc[shared_buckets].to_numpy(),
            means_cycle2.loc[shared_buckets].to_numpy(),
        )
    else:
        rank_corr = float("nan")
    n_shared = len(shared_buckets)
    print(f"Spearman rank correlation on {n_shared} shared bucket(s): "
          f"{rank_corr:.4f}" if np.isfinite(rank_corr) else
          f"Spearman rank correlation: undefined ({n_shared} shared bucket(s) "
          "-- need >=2 for a rank correlation)")
    print("POWER NOTE: with only "
          f"{n_shared} shared bucket(s) out of 4, an n={n_shared} Spearman "
          "correlation takes values from a small discrete set and should be "
          "read as weak evidence in either direction, consistent with "
          "r79_shared.py's own documented thin-coverage warning.")
    criterion_b = np.isfinite(rank_corr) and rank_corr >= 0.5

    # ================================================== stop rule
    gate_pass = criterion_a and criterion_b
    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE (frozen before either number above was "
          "computed):")
    print("  proceed to Step B only if (a) true-halving dispersion > placebo ")
    print("  null p95 AND (b) cross-cycle Spearman rank correlation on shared ")
    print("  buckets >= 0.5.")
    print(f"  (a) placebo-offset dispersion test: {'PASS' if criterion_a else 'FAIL'}")
    print(f"  (b) cross-cycle replication:        {'PASS' if criterion_b else 'FAIL'}")
    print(f"  GATE: {'PASS -> proceed to Step B' if gate_pass else 'FAIL -> STOP, report negative'}")
    print("=" * 78)

    summary = {
        "true_dispersion": true_disp,
        "null_mean": float(null_stats.mean()),
        "null_std": float(null_stats.std()),
        "null_p95": p95,
        "null_p99": p99,
        "empirical_pvalue": pval,
        "criterion_a_pass": bool(criterion_a),
        "n_shared_buckets": n_shared,
        "shared_buckets": str(shared_buckets),
        "rank_corr": rank_corr,
        "criterion_b_pass": bool(criterion_b),
        "gate_pass": bool(gate_pass),
    }
    out_path = ROOT / "experiments" / "reports" / "r79_novel_gate_summary.csv"
    out_path.parent.mkdir(exist_ok=True)
    pd.Series(summary).to_csv(out_path, header=["value"])
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
