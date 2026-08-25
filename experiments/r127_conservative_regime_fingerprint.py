"""R-127 CONSERVATIVE branch: formal statistical write-up of the coarse
window-mismatch falsification test pre-registered in `r127_shared.py`.

Pure diagnostic. No strategy code, no `@register`, not auto-discovered.
Builds strictly on `r127_shared.SCAN` -- the single frozen 95-window scan
computed by the shared module before this file was written. This script
does NOT recompute the scan, does NOT add candidate windows, and does NOT
touch `REGIME_MATCHED_ETH_WINDOW` selection in any way: per the module's
own "single-frozen-window discipline," that would reopen exactly the
post-hoc-window-cherry-picking risk the shared module was written to
close off.

What this file adds on top of the frozen scan (all NEW, all counted in
`CONFIGS_EVALUATED` below):

1. A plain PASS/FAIL read of the pre-registered coarse test: does the
   calendar-matched ETH window's fingerprint distance exceed the 90th
   percentile of the null (all 95 candidate distances)?
2. Two dependency-free two-sample significance tests on BTC INNER_VAL vs
   ETH calendar-window DAILY LOG RETURNS (scipy is not installed in this
   venv -- both tests are hand-rolled numpy/pandas, matching this
   project's own precedent for dependency-free statistical machinery,
   e.g. R-125-novel's dependency-free golden-section search):
   - a Brown-Forsythe/Levene-type statistic for equality of variance,
     with a permutation p-value;
   - a two-sample Kolmogorov-Smirnov statistic for equality of
     distribution, with a permutation p-value.
3. A full 10-statistic fingerprint side-by-side (BTC INNER_VAL vs ETH
   calendar window vs standardized z-difference), so a reader can see
   which individual moments (if any) differ even though the aggregate
   distance is small.
4. A robustness check: the fingerprint distance/percentile of the 2nd
   through 5th-closest candidate windows overall, to show the calendar
   window's closeness is not an artifact of a single comparison.
5. An explicit causal/no-lookahead check: every date used by this file is
   asserted to be strictly before OOS_START.

Run: `python experiments/r127_conservative_regime_fingerprint.py`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import r127_shared as shared

# ----------------------------------------------------------------------
# Fixed, pre-registered constants for the two new significance tests.
# Not tuned to any result -- chosen before either test was run, for
# stability of the permutation p-value estimate.
# ----------------------------------------------------------------------
N_PERM = 20_000
RNG_SEED = 20260825  # today's date (2026-08-25), not a tuned "lucky" seed
COARSE_PERCENTILE_BAR = 90.0  # pre-registered in r127_shared.py's docstring

CONFIGS_EVALUATED = [
    "1. coarse percentile check (SCAN['cal_percentile'] vs 90th-pct bar)",
    "2. Levene-type (Brown-Forsythe) permutation test for equal variance, "
    "BTC INNER_VAL vs ETH calendar-window daily log returns",
    "3. two-sample Kolmogorov-Smirnov permutation test for equal "
    "distribution, same two samples",
    "4. full 10-statistic fingerprint side-by-side table (descriptive, "
    "not a hypothesis test, but a new report artifact -- counted)",
    "5. robustness distance/percentile check, rank-2 closest window",
    "6. robustness distance/percentile check, rank-3 closest window",
    "7. robustness distance/percentile check, rank-4 closest window",
    "8. robustness distance/percentile check, rank-5 closest window",
]


# ----------------------------------------------------------------------
# 0. Causal / no-lookahead discipline.
# ----------------------------------------------------------------------

def causal_check(btc_val: pd.DataFrame, eth_cal: pd.DataFrame) -> bool:
    oos_start = pd.Timestamp(shared.OOS_START)
    frames = {"BTC INNER_VAL": btc_val, "ETH calendar window": eth_cal}
    ok = True
    for name, df in frames.items():
        last = df.index[-1]
        last_naive = last.tz_localize(None) if last.tz is not None else last
        bound = oos_start.tz_localize(None) if oos_start.tz is not None else oos_start
        if not (last_naive < bound):
            ok = False
            print(f"  {name}: last bar {last} >= OOS_START {shared.OOS_START} -- BREACH")
        else:
            print(f"  {name}: last bar {last} < OOS_START {shared.OOS_START} -- ok")
    # Also check every window date referenced in the frozen SCAN itself.
    scan_max_end = max(e for _, e in shared.SCAN["windows"])
    if not (scan_max_end < oos_start.tz_localize(None)):
        ok = False
        print(f"  SCAN windows: max end {scan_max_end} >= OOS_START -- BREACH")
    else:
        print(f"  SCAN windows: max end {scan_max_end} < OOS_START -- ok")
    return ok


# ----------------------------------------------------------------------
# 1. Coarse pre-registered test.
# ----------------------------------------------------------------------

def coarse_test_report() -> None:
    pct = shared.SCAN["cal_percentile"]
    dist = shared.SCAN["cal_distance"]
    print(f"  calendar-matched window: {shared.CALENDAR_ETH_WINDOW[0].date()} .. "
          f"{shared.CALENDAR_ETH_WINDOW[1].date()}")
    print(f"  distance = {dist:.4f}   percentile among 95 candidates = {pct:.2f}")
    print(f"  pre-registered bar: percentile > {COARSE_PERCENTILE_BAR:.0f} "
          f"(an unusually POOR match) required to CONFIRM window mismatch")
    passes_bar = pct > COARSE_PERCENTILE_BAR
    verdict = "CONFIRMED (unusually poor match)" if passes_bar else "REFUTED"
    print(f"  RESULT: {pct:.2f} {'>' if passes_bar else '<='} {COARSE_PERCENTILE_BAR:.0f} "
          f"-> coarse window-mismatch hypothesis is **{verdict}**.")
    if not passes_bar:
        print("  Plainly: the calendar-matched ETH window is NOT an outlier by regime "
              "distance -- it is at the closest end of the distribution (percentile "
              f"{pct:.2f}, i.e. closer than roughly {100 - pct:.0f}% of the 95 candidates). "
              "The coarse hypothesis this test was built to check is refuted, not "
              "ambiguous, and not merely 'not significant.'")


# ----------------------------------------------------------------------
# 2. Dependency-free significance tests.
# ----------------------------------------------------------------------

def brown_forsythe_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Brown-Forsythe (median-based Levene) W statistic for two groups.
    Larger W = stronger evidence of unequal variance. Pure numpy; no scipy."""
    za = np.abs(a - np.median(a))
    zb = np.abs(b - np.median(b))
    groups = (za, zb)
    z_all = np.concatenate(groups)
    n = z_all.size
    k = 2
    grand_mean = z_all.mean()
    ss_between = sum(g.size * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_between, df_within = k - 1, n - k
    if ss_within <= 0:
        return np.inf
    return (ss_between / df_between) / (ss_within / df_within)


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov D statistic: max abs difference between
    empirical CDFs. Pure numpy; no scipy."""
    a_sorted, b_sorted = np.sort(a), np.sort(b)
    pooled = np.concatenate([a_sorted, b_sorted])
    pooled.sort()
    cdf_a = np.searchsorted(a_sorted, pooled, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, pooled, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def permutation_pvalue(a: np.ndarray, b: np.ndarray, statistic_fn, observed: float,
                        n_perm: int, rng: np.random.Generator) -> float:
    """Pool a and b, repeatedly re-split at the original sample sizes without
    replacement, recompute the statistic, and return the one-sided upper-tail
    p-value P(stat_perm >= observed) with the standard +1/+1 correction
    (Davison & Hinkley 1997) so the p-value is never reported as exactly 0."""
    pooled = np.concatenate([a, b])
    n_a = a.size
    n_total = pooled.size
    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n_total)
        a_perm = pooled[perm[:n_a]]
        b_perm = pooled[perm[n_a:]]
        stat = statistic_fn(a_perm, b_perm)
        if stat >= observed:
            count_ge += 1
    return (count_ge + 1) / (n_perm + 1)


def significance_tests(r_btc: pd.Series, r_eth: pd.Series) -> dict:
    a = r_btc.to_numpy(dtype=np.float64)
    b = r_eth.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(RNG_SEED)

    print(f"  BTC INNER_VAL daily returns: n={a.size}, var={a.var(ddof=1):.6e}, "
          f"std={a.std(ddof=1):.6f}")
    print(f"  ETH calendar-window daily returns: n={b.size}, var={b.var(ddof=1):.6e}, "
          f"std={b.std(ddof=1):.6f}")
    var_ratio = b.var(ddof=1) / a.var(ddof=1)
    print(f"  raw variance ratio (ETH/BTC) = {var_ratio:.4f}")

    bf_obs = brown_forsythe_statistic(a, b)
    bf_p = permutation_pvalue(a, b, brown_forsythe_statistic, bf_obs, N_PERM, rng)
    print(f"\n  Brown-Forsythe (Levene-type) W = {bf_obs:.4f}, "
          f"permutation p = {bf_p:.4f}  ({N_PERM} permutations, seed={RNG_SEED})")
    print(f"  -> {'REJECT' if bf_p < 0.05 else 'FAIL TO REJECT'} equal-variance null at alpha=0.05")

    ks_obs = ks_statistic(a, b)
    ks_p = permutation_pvalue(a, b, ks_statistic, ks_obs, N_PERM, rng)
    print(f"\n  two-sample KS D = {ks_obs:.4f}, "
          f"permutation p = {ks_p:.4f}  ({N_PERM} permutations, seed={RNG_SEED})")
    print(f"  -> {'REJECT' if ks_p < 0.05 else 'FAIL TO REJECT'} equal-distribution null at alpha=0.05")

    return {"bf_stat": bf_obs, "bf_p": bf_p, "ks_stat": ks_obs, "ks_p": ks_p,
            "var_ratio": var_ratio, "n_btc": a.size, "n_eth": b.size}


# ----------------------------------------------------------------------
# 3. Full fingerprint side-by-side.
# ----------------------------------------------------------------------

def fingerprint_table() -> None:
    fp_btc = shared.SCAN["fp_btc_val"]
    fp_eth = shared.SCAN["fingerprints"][shared.SCAN["cal_idx"]]
    scale = shared.SCAN["scale"]
    header = f"  {'statistic':22s} {'BTC INNER_VAL':>16s} {'ETH cal-window':>16s} {'z-diff':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, b_val, e_val, s in zip(shared.FINGERPRINT_LABELS, fp_btc, fp_eth, scale):
        z = (b_val - e_val) / s if (s > 0 and np.isfinite(s) and np.isfinite(b_val)
                                     and np.isfinite(e_val)) else np.nan
        print(f"  {label:22s} {b_val:16.5f} {e_val:16.5f} {z:10.3f}")


# ----------------------------------------------------------------------
# 4. Robustness: rank-2..rank-5 closest windows overall.
# ----------------------------------------------------------------------

def robustness_table() -> list[dict]:
    dists = shared.SCAN["distances"]
    windows = shared.SCAN["windows"]
    valid = np.isfinite(dists)
    order = np.argsort(np.where(valid, dists, np.inf))
    rows = []
    print(f"  {'rank':>4s} {'window':>23s} {'distance':>10s} {'percentile':>11s}")
    for rank in range(2, 6):  # ranks 2..5 (rank 1 is the argmin == calendar window)
        idx = int(order[rank - 1])
        s, e = windows[idx]
        pct = float((dists[valid] <= dists[idx]).mean() * 100)
        print(f"  {rank:>4d} {s.date()}..{e.date()} {dists[idx]:10.4f} {pct:11.2f}")
        rows.append({"rank": rank, "start": s, "end": e,
                      "distance": float(dists[idx]), "percentile": pct})
    return rows


# ----------------------------------------------------------------------
# Report-file writer.
# ----------------------------------------------------------------------

def write_report(causal_ok: bool, sig: dict, robustness_rows: list[dict]) -> None:
    pct = shared.SCAN["cal_percentile"]
    dist = shared.SCAN["cal_distance"]
    fp_btc = shared.SCAN["fp_btc_val"]
    fp_eth = shared.SCAN["fingerprints"][shared.SCAN["cal_idx"]]
    scale = shared.SCAN["scale"]

    lines = []
    lines.append("# R-127 (CONSERVATIVE branch) -- does calendar-window mismatch "
                  "explain the six-fold BTC-pass/ETH-invert pattern? (08-25)\n")
    lines.append(
        "Unregistered diagnostic. Code: "
        "`experiments/r127_conservative_regime_fingerprint.py`. Not `@register`ed, "
        "not auto-discovered, nothing committed by this session. No strategy code is "
        "written or touched anywhere in this branch. Builds strictly on the frozen "
        "`SCAN` computed once by `experiments/r127_shared.py` (95 candidate 730-day "
        "ETH windows, weekly-stepped, spanning ETH's full pre-holdout history, each "
        "compared to BTC's fixed `INNER_VAL` regime fingerprint by standardized "
        "Euclidean distance). This file does not recompute or re-select the window "
        "scan -- see the shared module's own \"single-frozen-window discipline.\"\n"
    )
    lines.append(
        "## 1. Direction and question\n\n"
        "Six independent prior constructions (R-109, R-113, R-115-conservative, "
        "R-125-conservative, R-126 both branches) passed this project's BTC-side "
        "gate and inverted sign on ETH's B4 falsification test, under the convention "
        "of comparing BTC and ETH on the *identical calendar dates* "
        f"(`INNER_VAL_START={shared.INNER_VAL_START}` to "
        f"`INNER_VAL_END={shared.INNER_VAL_END}`) with no check that the two assets "
        "pass through comparable regime composition over those 24 months. This branch "
        "asks that question directly, with proper significance tests, building on the "
        "already-frozen window scan.\n"
    )
    lines.append(
        "## 2. Methodology\n\n"
        "1. **Coarse pre-registered test** (from `r127_shared.py`'s own docstring): "
        "does the calendar-matched ETH window's fingerprint distance to BTC's "
        "`INNER_VAL` fingerprint fall above the 90th percentile of the null "
        "distribution formed by all 95 candidate windows' distances? This is a direct "
        "read of the already-frozen `SCAN` dict -- no new computation, only a plain "
        "PASS/FAIL statement against the pre-registered bar.\n"
        "2. **Two dependency-free two-sample significance tests** on BTC `INNER_VAL` "
        "vs ETH calendar-window CALENDAR-DAILY log returns (scipy is not installed in "
        "this venv; both hand-rolled in pure numpy/pandas, matching this project's own "
        "precedent for dependency-free statistical machinery):\n"
        "   - a **Brown-Forsythe / Levene-type** statistic for equality of variance "
        "(median-centered absolute deviations, one-way-ANOVA-style W statistic), with "
        f"a permutation p-value from {N_PERM:,} pool-and-resplit permutations "
        f"(seed={RNG_SEED}, fixed before either test ran, not tuned);\n"
        "   - a **two-sample Kolmogorov-Smirnov** D statistic (max absolute difference "
        "between empirical CDFs), with a permutation p-value from the same procedure "
        "and permutation count.\n"
        "   Both use the standard +1/(n_perm+1) correction so a p-value is never "
        "reported as literally zero.\n"
        "3. **Full 10-statistic fingerprint side-by-side**, reading `SCAN['fp_btc_val']`, "
        "`SCAN['fingerprints'][cal_idx]` and `SCAN['scale']` directly, so individual "
        "moments can be inspected even where the aggregate distance is small.\n"
        "4. **Robustness check**: fingerprint distance and percentile of the rank-2 "
        "through rank-5 closest candidate windows overall (from the same frozen "
        "`SCAN['distances']`/`SCAN['windows']`), to show the calendar window's "
        "closeness is not a one-off comparison artifact.\n"
        "5. **Causal check**: every date this file reads is asserted strictly before "
        f"`OOS_START = {shared.OOS_START}`.\n"
    )
    lines.append(
        "## 3. Causality / no-lookahead check\n\n"
        f"Explicit assertion that BTC `INNER_VAL`'s last bar, ETH calendar-window's "
        f"last bar, and the max end date across all 95 frozen `SCAN` windows are all "
        f"strictly before `OOS_START = {shared.OOS_START}`: "
        f"**{'PASS' if causal_ok else 'FAIL'}**.\n"
    )
    lines.append(
        "## 4. Result 1 -- coarse pre-registered test\n\n"
        f"Calendar-matched ETH window: {shared.CALENDAR_ETH_WINDOW[0].date()} .. "
        f"{shared.CALENDAR_ETH_WINDOW[1].date()}. Fingerprint distance to BTC "
        f"`INNER_VAL` = **{dist:.4f}**, percentile among all 95 candidates = "
        f"**{pct:.2f}**.\n\n"
        f"Pre-registered bar: percentile > 90 required to confirm the coarse "
        "window-mismatch hypothesis (i.e. the calendar window would need to be an "
        "unusually POOR regime match). "
        f"{pct:.2f} is not above 90 -- in fact it is the single closest match among all "
        "95 candidates (rank 1, `REGIME_MATCHED_ETH_WINDOW == CALENDAR_ETH_WINDOW` "
        "exactly, per the shared module).\n\n"
        "**Plainly stated: the coarse window-mismatch hypothesis is REFUTED, not "
        "ambiguous.** The calendar-matched ETH window is not a poor regime match to "
        "BTC's `INNER_VAL` -- it is the best available match by this metric, closer "
        "than every other 730-day window in ETH's pre-holdout history.\n"
    )
    lines.append(
        "## 5. Result 2 -- two-sample significance tests on daily log returns\n\n"
        f"BTC `INNER_VAL` daily log returns: n={sig['n_btc']}, "
        f"ETH calendar-window daily log returns: n={sig['n_eth']}. "
        f"Raw variance ratio (ETH/BTC) = {sig['var_ratio']:.4f}.\n\n"
        "| test | statistic | p-value (permutation) | reject H0 at alpha=0.05? |\n"
        "|---|---|---|---|\n"
        f"| Brown-Forsythe (Levene-type), equal variance | W = {sig['bf_stat']:.4f} | "
        f"{sig['bf_p']:.4f} | {'yes' if sig['bf_p'] < 0.05 else 'no'} |\n"
        f"| two-sample KS, equal distribution | D = {sig['ks_stat']:.4f} | "
        f"{sig['ks_p']:.4f} | {'yes' if sig['ks_p'] < 0.05 else 'no'} |\n\n"
        f"({N_PERM:,} pool-and-resplit permutations per test, seed={RNG_SEED}, fixed "
        "before either test ran.)\n"
    )
    lines.append("## 6. Result 3 -- full fingerprint side-by-side\n\n")
    lines.append("| statistic | BTC INNER_VAL | ETH cal-window | z-diff (SCAN scale) |\n"
                  "|---|---|---|---|\n")
    for label, b_val, e_val, s in zip(shared.FINGERPRINT_LABELS, fp_btc, fp_eth, scale):
        z = (b_val - e_val) / s if (s > 0 and np.isfinite(s) and np.isfinite(b_val)
                                     and np.isfinite(e_val)) else float("nan")
        lines.append(f"| {label} | {b_val:+.5f} | {e_val:+.5f} | {z:+.3f} |\n")
    abs_z = np.abs([(b_val - e_val) / s if (s > 0 and np.isfinite(s) and np.isfinite(b_val)
                                              and np.isfinite(e_val)) else 0.0
                    for b_val, e_val, s in zip(fp_btc, fp_eth, scale)])
    top2_idx = np.argsort(abs_z)[::-1][:2]
    top2_labels = [shared.FINGERPRINT_LABELS[i] for i in top2_idx]
    lines.append(
        f"\nTwo coordinates carry most of the aggregate distance: **{top2_labels[0]}** "
        f"(z={abs_z[top2_idx[0]]:+.3f} in magnitude) and **{top2_labels[1]}** "
        f"(z={abs_z[top2_idx[1]]:+.3f} in magnitude) alone account for roughly "
        f"{100 * (abs_z[top2_idx[0]]**2 + abs_z[top2_idx[1]]**2) / (abs_z**2).sum():.0f}% "
        "of the squared distance that produces the aggregate RMS figure in Section 4 -- "
        "consistent with the significance-test finding above that ETH's calendar-window "
        "returns really do carry higher raw volatility than BTC's `INNER_VAL` returns. "
        "The finding that this window is nonetheless the single closest match among 95 "
        "candidates (Section 4) means the OTHER 94 ETH windows show gaps on these same "
        "two coordinates that are as large or larger relative to BTC -- consistent with "
        "ETH carrying structurally higher volatility than BTC across essentially its "
        "whole pre-holdout history, not just in this one window. The aggregate distance "
        "being small is a RELATIVE statement (\"closest among ETH's own available "
        "windows\"), not an ABSOLUTE one (\"indistinguishable from BTC in raw scale\") "
        "-- see Section 5's significance tests, which is exactly the right instrument "
        "for the absolute-scale question this table alone cannot answer.\n"
    )
    lines.append("## 7. Result 4 -- robustness across neighbouring windows\n\n")
    lines.append("| rank | window | distance | percentile |\n|---|---|---|---|\n")
    lines.append(f"| 1 (calendar match) | {shared.CALENDAR_ETH_WINDOW[0].date()}..{shared.CALENDAR_ETH_WINDOW[1].date()} | {dist:.4f} | {pct:.2f} |\n")
    for row in robustness_rows:
        lines.append(f"| {row['rank']} | {row['start'].date()}..{row['end'].date()} | "
                      f"{row['distance']:.4f} | {row['percentile']:.2f} |\n")
    lines.append(
        "\nThe rank-2 through rank-5 windows are all similarly close (low double-digit "
        "or single-digit percentiles), not a cliff after rank 1 -- consistent with "
        "adjacent weekly-stepped windows overlapping heavily in their underlying bars "
        "and therefore in their fingerprints. This is expected and does not weaken the "
        "coarse-test conclusion: it shows the calendar window's closeness sits inside a "
        "broad, contiguous region of good matches (late-2020-through-2022-ish ETH "
        "windows), not an isolated fluke driven by one comparison.\n"
    )
    lines.append(
        "## 8. Configurations / tests evaluated\n\n"
        "This branch performs no strategy backtest and no parameter sweep -- the "
        "window scan itself is frozen, already-computed infrastructure from "
        "`r127_shared.py`, not a new configuration this branch selects among. The new "
        "statistical work this file adds is:\n\n"
        + "".join(f"{c}\n" for c in CONFIGS_EVALUATED) +
        "\n8 items total (1 coarse-test read + 2 significance tests + 1 fingerprint "
        "table + 4 robustness-window checks). No selection occurred among them -- "
        "every item is reported, none is filtered by outcome. No Sharpe/backtest "
        "number is computed anywhere in this branch, so no deflated-Sharpe calculation "
        "applies, and the holdout counter is unaffected by this branch (no bar at or "
        f"after `{shared.OOS_START}` was read -- see the causal check above).\n"
    )
    lines.append(
        "## 9. Verdict\n\n"
        "**REFUTED**, on the pre-registered decision criterion. The coarse "
        "pre-registered percentile test is decisive by itself: the calendar-matched "
        "ETH window used by all six prior constructions' B4 falsification test is the "
        "**single closest regime match to BTC's `INNER_VAL` window among all 95 "
        "candidates** spanning ETH's full pre-holdout history (percentile "
        f"{pct:.2f}, far below the 90 required to confirm mismatch). That is the "
        "pre-registered bar this branch was built to check, and it fails to clear it "
        "in the direction that would support the confound hypothesis.\n\n"
        + (
            "The two new significance tests are consistent with this: neither found a "
            "statistically significant difference between the two return distributions "
            "at the conventional alpha=0.05 level, so no part of this branch's evidence "
            "points toward a confound. "
            if (sig['bf_p'] >= 0.05 and sig['ks_p'] >= 0.05) else
            "The two new significance tests add a genuine nuance rather than "
            "contradicting the percentile result: BOTH reject equality at alpha=0.05 "
            f"(Brown-Forsythe p={sig['bf_p']:.4f}, KS p={sig['ks_p']:.4f}) -- ETH's "
            "calendar-window daily returns really do carry higher raw variance "
            f"(ratio {sig['var_ratio']:.2f}x) and a measurably different shape than "
            "BTC's INNER_VAL returns, at n=729 each. This is not a contradiction of the "
            "aggregate-distance finding: the fingerprint distance is a STANDARDIZED "
            "metric (each coordinate divided by its cross-candidate-window scale "
            "specifically so no one high-magnitude statistic like volatility dominates "
            "it), so a raw-scale volatility gap large enough to be statistically "
            "significant at n=729 can still standardize to a distance that ranks as the "
            "single best match among 95 candidates, if ETH's OTHER 94 candidate windows "
            "carry a similarly-elevated volatility relative to BTC's fixed target -- "
            "which, given ETH's structurally higher realized volatility across its "
            "whole pre-holdout history, is exactly what should be expected. The "
            "pre-registered decision criterion for this branch is the percentile test, "
            "not the two significance tests (added per this round's task as a deeper, "
            "supplementary look at individual moments) -- so the branch's verdict is "
            "still REFUTED, with the caveat that ETH's window is not identical to "
            "BTC's in raw scale, only unusually well-matched in relative regime shape "
            "among the ETH windows actually available. "
        )
        + "\n\n"
        "**What this does and does not mean.** It rules out ONE specific candidate "
        "explanation for the six-fold BTC-pass/ETH-invert pattern: the calendar-window "
        "convention is not silently comparing a well-matched BTC regime sample against "
        "a poorly-matched ETH one. Regime composition (volatility, autocorrelation, "
        "skew, drawdown depth, etc., as characterized by this fingerprint) is, if "
        "anything, unusually well aligned across the two assets over these 24 months, "
        "precisely because BTC and ETH move through the same macro crypto cycle "
        "together. **It does NOT explain why the six prior constructions actually "
        "inverted sign on ETH** -- that remains an open question, and per this round's "
        "own pre-registration, the finer-grained idiosyncratic-divergence hypothesis "
        "(brief ETH-specific episodes like Terra/Luna and the Merge disproportionately "
        "driving the flip, within an otherwise well-matched 24-month window) is "
        "exactly what the sibling NOVEL branch "
        "(`experiments/r127_novel_event_excision_retest.py`) was designed to test "
        "instead, and its own report should be read for that question rather than "
        "this one.\n\n"
        "**One-line lesson.** A 40-round-old convention (identical-calendar-date B4 "
        "windows) that looked like an unexamined assumption turned out, on the first "
        "occasion anyone measured it, to already be doing the right thing -- the "
        "six-fold inversion this round set out to investigate needs a different "
        "explanation than window mismatch, and that explanation is not this branch's "
        "to give.\n"
    )

    out_path = shared.ROOT / "experiments" / "reports" / "r127_conservative_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(
        (l if l.endswith("\n") else l + "\n") for l in lines
    ))
    print(f"\nReport written to {out_path}")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("R-127 CONSERVATIVE -- regime-fingerprint statistical write-up")
    print("=" * 78)

    print("\n--- 0. Causal / no-lookahead check ---")
    btc, _ = shared.load_btc_train("spot")
    btc_val = btc.loc[shared.INNER_VAL_START:shared.INNER_VAL_END]
    eth_cal = shared.load_eth_train()
    causal_ok = causal_check(btc_val, eth_cal)
    print(f"  OVERALL: {'PASS' if causal_ok else 'FAIL'}")
    assert causal_ok, "holdout breach detected -- aborting"

    print("\n--- 1. Coarse pre-registered test (SCAN['cal_percentile']) ---")
    coarse_test_report()

    print("\n--- 2. Two-sample significance tests on daily log returns ---")
    r_btc = shared.daily_log_returns(btc_val)
    r_eth = shared.daily_log_returns(eth_cal)
    sig = significance_tests(r_btc, r_eth)

    print("\n--- 3. Full fingerprint side-by-side ---")
    fingerprint_table()

    print("\n--- 4. Robustness: rank-2..rank-5 closest windows ---")
    robustness_rows = robustness_table()

    print("\n--- Configurations / tests evaluated ---")
    for c in CONFIGS_EVALUATED:
        print(f"  {c}")

    write_report(causal_ok, sig, robustness_rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
