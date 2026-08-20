"""R-70 novel branch: a bootstrap-studentized Sharpe-difference test for
B-36, estimating the long-run covariance nonparametrically via this
project's own stationary-bootstrap convention rather than a kernel.

=====================================================================
PRE-REGISTRATION (written before any real-data number in this file was
computed -- see the synthetic validation in `_synthetic_coverage_check`,
which was run and passed BEFORE `main()` touched `r70_shared.build_all_cells()`)
=====================================================================

**Mechanism, one sentence.** Ledoit & Wolf (2008, J. Empirical Finance
15(5), "Robust performance hypothesis testing with the Sharpe ratio")
studentize the Sharpe-ratio-difference statistic by an estimate of its own
asymptotic standard error before doing inference, and the specific
estimator of that standard error is a free methodological choice -- the
paper's own realization uses a Parzen-kernel HAC estimate of the long-run
covariance of the four return moments, but nothing in the studentization
argument requires a kernel; substituting this project's own stationary
bootstrap (Politis & Romano 1994, 30-day mean block -- the convention
`paired_bootstrap`, R-20's noise floor and every other interval in this
repo already use) for that one piece should give materially the same
inference on this specific autocorrelated, heavy-tailed crypto series,
because both are consistent nonparametric-vs-semiparametric estimators of
the same underlying quantity.

**Falsification tests, named in advance, two of them:**

(F1) *Synthetic calibration (gate before real data).* On repeated
synthetic draws from a known-parameter, AR(1)-correlated pair of return
series with a KNOWN true Sharpe gap, the empirical coverage of both the
normal-approximation CI and the single-bootstrap studentized-t CI (see
"the simplification" below) must land within a generous band around the
nominal 95% -- 88%-99%, generous because the check runs a few hundred
synthetic replications, not tens of thousands. FAILURE OUTCOME: either
interval's empirical coverage falls outside that band on either the iid
or the AR(1)-persistent synthetic setting. A failure here means the
single-bootstrap studentized-t shortcut is miscalibrated on data shaped
like this project's and the whole construction should not be trusted on
the real cells without the full (expensive) double bootstrap. This gate
is checked BEFORE any real cell is touched, per ROUTINE.md step 2.

(F2) *Real-data agreement with the existing plain-percentile-bootstrap
difference test.* This branch runs in parallel with a conservative branch
computing the same studentized statistic via a kernel-HAC estimator, so
this file cannot compare against that branch's numbers directly -- it was
not run yet when this was written. What CAN be pre-registered is
agreement with the number this repo already has: R-68's own
`paired_bootstrap`-on-`total_log_return` difference test (the six
reference cells quoted in this round's task, reproduced by
`experiments/r70_shared.py`). This file reports `se_boot` and `tstat`
plainly for every one of the six real cells -- none omitted, none
selected -- and flags explicitly any cell where `se_boot` (in Sharpe
units, not growth units, so the comparison is of RELATIVE spread, not
absolute level) implies a materially different picture than the existing
interval's own implied half-width scaled the same way. Concretely: flag
any cell where this test's significance verdict (does the 95% interval
exclude zero) DISAGREES with the existing plain-percentile-bootstrap
verdict on the same cell. FAILURE OUTCOME (of the round's actual research
question, not of this construction): if studentization flips a verdict
that the plain-percentile test got right, or fails to flip one it got
wrong on grounds that later prove correct, that disagreement is itself
the informative result B-36 asked this round to produce -- it does not by
itself indict either method, but it means the two must be reconciled
(against the conservative branch's kernel-HAC number) before a claim on
this axis can be defended.

=====================================================================
THE STATISTIC
=====================================================================

For two aligned, equal-length daily-return series ``x1``, ``x2``:

    SR_i = mean(x_i) / std(x_i, ddof=1)      -- SAMPLE std, matching
                                                 `tradebot.inference
                                                 .annualized_sharpe`'s own
                                                 convention (ddof=1), so a
                                                 number here is comparable
                                                 to every other Sharpe this
                                                 repo already reports.
    T(x1, x2) = SR1 - SR2

Not annualized: annualizing multiplies both `T` and its bootstrap
replicates by the SAME constant (sqrt(365.25)), so the studentized
statistic `T / se_boot` is scale-invariant and the significance verdict
is identical either way. Daily units are reported because they are the
finer-grained, more literal reading of the underlying return series.

=====================================================================
THE STANDARD ERROR -- THIS BRANCH'S FORK FROM THE CONSERVATIVE ONE
=====================================================================

Draw `B` resample index vectors with
`tradebot.inference.stationary_bootstrap_indices(n, mean_block=30.0,
n_boot=B, rng)` and apply the SAME index vector to both `x1` and `x2`
(paired, exactly as `paired_bootstrap` already does in this repo) --
pairing is what makes the two series' shared variance cancel in the
difference, which is the entire reason a difference test resolves better
than either level (R-68's own finding). For each replicate b, compute
`T*_b = SR1*_b - SR2*_b`. Then:

    se_boot = std({T*_b}, ddof=1)
    tstat   = T(x1, x2) / se_boot

**The simplification, named explicitly.** A fully general studentized
(bootstrap-t) interval needs EACH replicate's own standard error, which
in turn needs a bootstrap WITHIN the bootstrap -- a nested/double
bootstrap, `O(B_outer * B_inner)` resamples. This file uses the standard
practical shortcut instead (see e.g. Davison & Hinkley 1997, *Bootstrap
Methods and their Application*, ch.5.4 -- "the single bootstrap"): every
replicate is studentized by the SAME `se_boot` computed once on the
original sample, i.e.

    studentized_b = (T*_b - T) / se_boot
    q_lo, q_hi    = alpha/2, 1-alpha/2 quantiles of {studentized_b}
    CI            = [T - se_boot * q_hi,  T - se_boot * q_lo]

This is NOT the fully general double-bootstrap studentized interval and
is not claimed to be; it is a documented compromise that this file's own
synthetic gate (F1) checks is adequately calibrated for THIS use before
it is trusted on real data.

=====================================================================
WHAT THIS FILE DOES NOT DO
=====================================================================

- It does not edit `experiments/r70_shared.py`, `tradebot/inference.py`,
  or any file outside itself. `tradebot.inference.stationary_bootstrap_indices`
  is called directly, never reimplemented.
- It builds no strategy and evaluates no new portfolio configuration --
  `config_count()` is reported at the end and should equal exactly what
  `r70_shared.build_all_cells()` itself already spent.
- It selects no winner among the three arms and omits no cell.
- Holdout: +0 (inherits `r70_shared`'s W_TRAIN/W_VAL restriction).

    .venv/bin/python experiments/r70_novel_bootstrap_studentized.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tradebot.inference import (  # noqa: E402
    Interval,
    norm_ppf,
    stationary_bootstrap_indices,
)

BOOT_BLOCK = 30.0  # days -- this project's established mean block length
BOOT_N = 2_000
BOOT_SEED = 7


# --------------------------------------------------------------- the stat


def sharpe_ratio(x: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Non-annualized daily Sharpe: mean / sample std (ddof=1, matching
    `tradebot.inference.annualized_sharpe`'s own convention).

    Axis-aware like the rest of `tradebot.inference`: accepts a 1-D series
    or an ``(n_boot, n)`` stack and reduces along the last axis, so the
    bootstrap loop below applies this to a whole resample matrix at once.
    """
    x = np.asarray(x, dtype=float)
    sd = x.std(axis=-1, ddof=ddof)
    mean = x.mean(axis=-1)
    ok = (sd > 0) & np.isfinite(sd)
    safe = np.where(ok, sd, 1.0)
    out = np.where(ok, mean / safe, 0.0)
    return out if x.ndim > 1 else float(out)


@dataclass
class StudentizedSharpeDiff:
    """Bootstrap-studentized Sharpe-ratio-difference test.

    Same shape as `tradebot.inference.PairedResult`, extended with the
    studentized machinery: `se_boot` and `tstat` are the studentized
    statistic itself, `normal_ci` is `diff +/- z*se_boot`, and
    `studentized_ci` is the single-bootstrap studentized-t interval (see
    module docstring for why it is not the fully general double
    bootstrap).
    """

    sr_a: float
    sr_b: float
    diff: float
    se_boot: float
    tstat: float
    normal_ci: Interval
    studentized_ci: Interval
    n_boot: int
    mean_block: float
    n: int

    @property
    def significant_normal(self) -> bool:
        return self.normal_ci.lo > 0.0 or self.normal_ci.hi < 0.0

    @property
    def significant_studentized(self) -> bool:
        return self.studentized_ci.lo > 0.0 or self.studentized_ci.hi < 0.0


def bootstrap_studentized_sharpe_diff(
    a: np.ndarray, b: np.ndarray, *,
    mean_block: float = BOOT_BLOCK, n_boot: int = BOOT_N,
    level: float = 0.95, seed: int = BOOT_SEED,
    indices: np.ndarray | None = None, ddof: int = 1,
) -> StudentizedSharpeDiff:
    """Ledoit-Wolf (2008)-style studentized Sharpe-difference test, with the
    standard error estimated via this project's OWN stationary-bootstrap
    convention (Politis & Romano 1994) instead of a kernel HAC estimate.

    ``a`` and ``b`` must be aligned, equal-length daily-return series from
    the same period -- exactly `paired_bootstrap`'s own precondition, for
    the same reason: the resample has to draw the same days from both
    series so their shared variance cancels in the difference.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"unaligned series: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 3:
        raise ValueError("need at least 3 paired observations")

    if indices is None:
        idx = stationary_bootstrap_indices(
            n, mean_block, n_boot, np.random.default_rng(seed))
    else:
        idx = indices
        if idx.shape[1] != n:
            raise ValueError(f"indices are for n={idx.shape[1]}, series has {n}")

    sr_a = sharpe_ratio(a, ddof)
    sr_b = sharpe_ratio(b, ddof)
    diff = sr_a - sr_b

    boot_diff = sharpe_ratio(a[idx], ddof) - sharpe_ratio(b[idx], ddof)  # (n_boot,)
    se_boot = float(boot_diff.std(ddof=1))

    tail = (1.0 - level) / 2.0
    if se_boot <= 0.0 or not np.isfinite(se_boot):
        tstat = float("nan")
        normal_ci = Interval(diff, float("nan"), float("nan"), level)
        studentized_ci = Interval(diff, float("nan"), float("nan"), level)
    else:
        tstat = diff / se_boot
        z = norm_ppf(1.0 - tail)
        normal_ci = Interval(diff, diff - z * se_boot, diff + z * se_boot, level)

        studentized = (boot_diff - diff) / se_boot
        q_lo, q_hi = np.percentile(studentized, [100 * tail, 100 * (1 - tail)])
        studentized_ci = Interval(
            diff, diff - se_boot * q_hi, diff - se_boot * q_lo, level)

    return StudentizedSharpeDiff(
        sr_a=float(sr_a), sr_b=float(sr_b), diff=float(diff),
        se_boot=se_boot, tstat=float(tstat),
        normal_ci=normal_ci, studentized_ci=studentized_ci,
        n_boot=int(idx.shape[0]), mean_block=float(mean_block), n=n)


# ------------------------------------------------------- (F1) synthetic gate


def _ar1_pair(n: int, mu_a: float, sd_a: float, mu_b: float, sd_b: float,
             phi: float, corr: float, rng: np.random.Generator):
    """Two AR(1) series with stationary mean/std (mu_a, sd_a), (mu_b, sd_b)
    and innovation correlation ``corr`` -- a paired-arm-like synthetic
    analogue of two trend variants trading the same signal at different
    speeds. Burn-in of 200 bars discarded so the returned series starts at
    stationarity, not at the deterministic initial condition."""
    burn = 200
    cov = np.array([[1.0, corr], [corr, 1.0]])
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n + burn, 2)) @ L.T
    eps_a = z[:, 0] * sd_a * np.sqrt(max(1.0 - phi ** 2, 1e-9))
    eps_b = z[:, 1] * sd_b * np.sqrt(max(1.0 - phi ** 2, 1e-9))
    xa = np.empty(n + burn)
    xb = np.empty(n + burn)
    xa[0], xb[0] = mu_a, mu_b
    for t in range(1, n + burn):
        xa[t] = mu_a + phi * (xa[t - 1] - mu_a) + eps_a[t]
        xb[t] = mu_b + phi * (xb[t - 1] - mu_b) + eps_b[t]
    return xa[burn:], xb[burn:]


def _synthetic_coverage_check(n: int, phi: float, m_reps: int = 250,
                              n_boot: int = 600, seed: int = 12345) -> dict:
    """Empirical coverage of both intervals over ``m_reps`` synthetic draws
    with a KNOWN true Sharpe gap. Ground truth uses the AR(1)'s own
    stationary mean/std (mu, sd) -- unaffected by phi, since an AR(1) with
    mean-reverting target mu has the same unconditional mean and, by
    construction of the innovation scale above, the same stationary std
    regardless of phi."""
    rng = np.random.default_rng(seed)
    mu_a, sd_a = 0.0008, 0.020
    mu_b, sd_b = 0.0003, 0.020
    true_diff = mu_a / sd_a - mu_b / sd_b
    corr = 0.85  # paired arms in this repo are highly correlated by construction

    hits_normal = 0
    hits_stud = 0
    widths_normal = []
    widths_stud = []
    for _ in range(m_reps):
        xa, xb = _ar1_pair(n, mu_a, sd_a, mu_b, sd_b, phi, corr, rng)
        res = bootstrap_studentized_sharpe_diff(
            xa, xb, mean_block=BOOT_BLOCK, n_boot=n_boot,
            seed=int(rng.integers(0, 2**31 - 1)))
        if res.normal_ci.lo <= true_diff <= res.normal_ci.hi:
            hits_normal += 1
        if res.studentized_ci.lo <= true_diff <= res.studentized_ci.hi:
            hits_stud += 1
        widths_normal.append(res.normal_ci.hi - res.normal_ci.lo)
        widths_stud.append(res.studentized_ci.hi - res.studentized_ci.lo)

    return {
        "n": n, "phi": phi, "m_reps": m_reps, "true_diff": true_diff,
        "coverage_normal": hits_normal / m_reps,
        "coverage_studentized": hits_stud / m_reps,
        "mean_width_normal": float(np.mean(widths_normal)),
        "mean_width_studentized": float(np.mean(widths_stud)),
    }


def run_falsification_gate() -> bool:
    """(F1). Runs the synthetic coverage check on settings that bracket the
    real cells: n matching W_VAL (364) and W_TRAIN (639), phi=0 (iid, a
    lower bound on how hard this should be) and phi=0.2 (mildly persistent,
    in the range real daily crypto returns actually show -- this repo's
    30-day mean block is a deliberately generous choice, not a fitted one,
    so this checks the shortcut still works when true memory is much
    shorter than the block length assumes).

    Returns True (gate PASSED) iff every setting's coverage for BOTH
    intervals lands in [0.88, 0.99]. Printed either way -- a failure here
    stops `main()` before it touches real data.
    """
    print("=== (F1) synthetic calibration gate ===")
    ok = True
    for n in (364, 639):
        for phi in (0.0, 0.2):
            r = _synthetic_coverage_check(n, phi)
            band_ok = (0.88 <= r["coverage_normal"] <= 0.99
                      and 0.88 <= r["coverage_studentized"] <= 0.99)
            ok = ok and band_ok
            print(f"  n={n:4d} phi={phi:.1f}  true_diff={r['true_diff']:+.4f}  "
                  f"coverage: normal={r['coverage_normal']:.3f}  "
                  f"studentized={r['coverage_studentized']:.3f}  "
                  f"mean_width: normal={r['mean_width_normal']:.4f}  "
                  f"studentized={r['mean_width_studentized']:.4f}  "
                  f"{'OK' if band_ok else 'FAIL'}")
    print(f"  GATE {'PASSED' if ok else 'FAILED'} "
          f"(both intervals within [0.88, 0.99] on every setting)"
          if ok else
          f"  GATE FAILED -- see module docstring (F1): the construction "
          f"is not trusted on real data until this is fixed.")
    return ok


# --------------------------------------------------------------- real cells


# Existing plain-percentile-bootstrap growth-difference reference numbers
# (already computed by R-68/this round's task description, NOT reproduced
# here -- reported only so this file's own printout can flag agreement or
# disagreement in SIGNIFICANCE, per (F2)). Units: log-growth, not Sharpe --
# not comparable in LEVEL to this file's Sharpe-difference numbers, only in
# whether the interval excludes zero.
REFERENCE_GROWTH_DIFF = {
    ("r67_hysteresis_0.080", "W_TRAIN"): (+0.45, -0.07, 1.10),
    ("r68_entry_only_0.080", "W_TRAIN"): (+0.51, -0.06, 1.29),
    ("r68_novel_derived_mult1.0", "W_TRAIN"): (+0.71, 0.05, 1.47),
    ("r67_hysteresis_0.080", "W_VAL"): (+0.43, -0.11, 0.93),
    ("r68_entry_only_0.080", "W_VAL"): (+0.78, 0.28, 1.29),
    ("r68_novel_derived_mult1.0", "W_VAL"): (+0.46, -0.25, 1.06),
}


def main():
    gate_ok = run_falsification_gate()
    if not gate_ok:
        print("\nSTOPPING before real data per pre-registered (F1): gate failed.")
        return

    from experiments.r70_shared import build_all_cells
    from experiments.r68_shared import config_count

    print("\n=== (F2) real cells: bootstrap-studentized Sharpe-difference ===")
    header = (f"{'arm':28s} {'window':8s} {'n':>4s}  {'SR_cand':>8s} "
              f"{'SR_base':>8s} {'diff':>8s} {'se_boot':>8s} {'tstat':>7s}  "
              f"{'normal_CI':>22s}  {'studentized_CI':>22s}  sig(N/S)  "
              f"ref_sig  agree")
    print(header)

    cells = build_all_cells()
    rows = []
    for (arm, window), (cand, base, matched, n) in cells.items():
        res = bootstrap_studentized_sharpe_diff(cand, base)
        ref_point, ref_lo, ref_hi = REFERENCE_GROWTH_DIFF[(arm, window)]
        ref_sig = ref_lo > 0.0 or ref_hi < 0.0
        this_sig = res.significant_studentized
        agree = "yes" if this_sig == ref_sig else "DISAGREE"
        print(f"{arm:28s} {window:8s} {n:4d}  {res.sr_a:+8.4f} {res.sr_b:+8.4f} "
              f"{res.diff:+8.4f} {res.se_boot:8.4f} {res.tstat:+7.2f}  "
              f"[{res.normal_ci.lo:+8.4f},{res.normal_ci.hi:+8.4f}]  "
              f"[{res.studentized_ci.lo:+8.4f},{res.studentized_ci.hi:+8.4f}]  "
              f"{'Y' if res.significant_normal else 'n'}/"
              f"{'Y' if this_sig else 'n'}      "
              f"{'Y' if ref_sig else 'n'}       {agree}"
              f"{'  (matched=False)' if not matched else ''}")
        rows.append((arm, window, res, ref_sig, agree))

    disagreements = [r for r in rows if r[4] == "DISAGREE"]
    print(f"\n{len(disagreements)}/6 cells disagree with the plain-percentile "
          f"reference on significance (F2 flag threshold: any disagreement "
          f"is flagged, not a pass/fail gate -- see module docstring).")
    for arm, window, res, ref_sig, agree in disagreements:
        print(f"  FLAGGED: {arm} {window}: studentized tstat={res.tstat:+.2f} "
              f"(sig={res.significant_studentized}) vs reference sig={ref_sig}")

    print(f"\nconfig_count() = {config_count()} "
          f"(entirely from build_all_cells(); this file evaluates no new "
          f"portfolio configuration)")
    print("Holdout consultations: +0 (inherits r70_shared's W_TRAIN/W_VAL "
          "restriction; see B-33).")


if __name__ == "__main__":
    main()
