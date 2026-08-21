"""R-78 conservative branch — how long must B-06 run before it can answer?

Nine rounds have closed by recommending forward paper trading (B-06) as the
only thing that can still move an interval in this project. R-71 built the
tool to read it — Waudby-Smith & Ramdas (2024, JRSSB 86(1); arXiv:2010.09686)
Theorem 2, the closed-form predictable-plug-in empirical-Bernstein anytime-
valid confidence sequence, shipped as
``tradebot.inference.empirical_bernstein_confidence_sequence`` — and closed
with "a future round should not re-read this entry's tools until B-06 has
enough rows for ``anytime_valid_first_exclusion`` to have a real chance of
firing."

*Enough rows* was never quantified. This branch quantifies it.

The pre-registration, the classification rule and the two falsification
tests (F1 null calibration, F2 power on a known effect) are frozen in
``experiments/r78_shared.py`` and are not restated here so they cannot
drift. Run::

    python experiments/r78_conservative_b06_horizon.py

Holdout: **+0**. Every frame comes from ``r78_shared.load_truncated()``,
which truncates at 2022-12-31 and asserts it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.r78_shared import (  # noqa: E402
    FEE_LIVE,
    FEE_TABLE,
    TRADING_DAYS,
    W_TRAIN,
    W_VAL,
    bootstrap_paths,
    load_truncated,
    paired_daily_diff,
)
from tradebot.inference import (  # noqa: E402
    anytime_valid_first_exclusion,
    empirical_bernstein_confidence_sequence,
)

N_PATHS = 400
HORIZON_DAYS = TRADING_DAYS["25y"]
ALPHA = 0.05
# R-71's documented engineering choice for B-06's arms, reused verbatim
# rather than re-tuned here: two unleveraged no-short spot accounts on the
# same instrument, so a paired daily-return difference is bounded by
# roughly the day's own BTCUSD move.
BOUND = 0.5

CONFIGS = 0     # real-data backtest runs; incremented as they happen


def _first_exclusions(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First-exclusion index and sign for each bootstrap path.

    ``np.nan`` where the sequence never excludes zero over the horizon.
    Sign is +1 if the exclusion is above zero (for the strategy), -1 if
    below (against it), 0 where there is none.
    """
    firsts = np.full(len(paths), np.nan)
    signs = np.zeros(len(paths))
    for p, path in enumerate(paths):
        cs = empirical_bernstein_confidence_sequence(path, bound=BOUND, alpha=ALPHA)
        n = anytime_valid_first_exclusion(cs)
        if n is not None:
            firsts[p] = n
            signs[p] = 1.0 if float(cs["lower"].to_numpy()[n - 1]) > 0 else -1.0
    return firsts, signs


def _summarize(tag: str, firsts: np.ndarray, signs: np.ndarray) -> dict:
    fired = np.isfinite(firsts)
    row = {
        "tag": tag,
        "fired_pct": 100.0 * fired.mean(),
        "n50": float(np.median(firsts[fired])) if fired.any() else float("inf"),
        "against_pct": 100.0 * (signs < 0).mean(),
        "for_pct": 100.0 * (signs > 0).mean(),
    }
    for name, days in TRADING_DAYS.items():
        row[f"by_{name}"] = 100.0 * np.mean(fired & (firsts <= days))
    return row


def main() -> None:
    global CONFIGS
    df, label = load_truncated()
    print(f"data: {label}, {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")

    rows = []
    diff_store = {}
    for wname, window in (("inner-train", W_TRAIN), ("inner-val", W_VAL)):
        for fname, fee in (("0.10% (table)", FEE_TABLE), ("0.40% (live)", FEE_LIVE)):
            d = paired_daily_diff(df, label, window, fee)
            CONFIGS += 2      # two real-data backtest runs per cell
            diff_store[(wname, fname)] = d
            mu, sd = float(d.mean()), float(d.std(ddof=1))
            ann = mu * 365
            print(f"\n[{wname} | {fname}] n={len(d)} days  "
                  f"mean={mu:+.6f}/day ({ann:+.1%}/yr)  sd={sd:.6f}  "
                  f"t={mu / (sd / np.sqrt(len(d))):+.2f}  "
                  f"zero-days={100.0 * (d == 0).mean():.1f}%")

            paths = bootstrap_paths(d.to_numpy(), HORIZON_DAYS, N_PATHS)
            firsts, signs = _first_exclusions(paths)
            row = _summarize(f"{wname} | {fname}", firsts, signs)
            rows.append(row)
            print(f"    first exclusion: fired in {row['fired_pct']:.1f}% of "
                  f"{N_PATHS} paths within 25y; median n50="
                  f"{row['n50']:,.0f} days"
                  + (f" ({row['n50'] / 365:.1f} years)"
                     if np.isfinite(row["n50"]) else "")
                  + f"; direction: {row['for_pct']:.1f}% for / "
                    f"{row['against_pct']:.1f}% against the strategy")
            print("    cumulative: " + "  ".join(
                f"{k}={row[f'by_{k}']:.1f}%" for k in TRADING_DAYS))

    # ---------------------------------------------------- falsification tests
    print("\n" + "=" * 70)
    print("PRE-REGISTERED FALSIFICATION TESTS")
    print("=" * 70)

    base = diff_store[("inner-val", "0.40% (live)")].to_numpy()

    null = base - base.mean()
    f1_firsts, _ = _first_exclusions(bootstrap_paths(null, HORIZON_DAYS, N_PATHS,
                                                     seed=781))
    f1_rate = 100.0 * np.mean(np.isfinite(f1_firsts))
    f1_pass = f1_rate <= 5.0
    print(f"F1 null calibration (recentred inner-val diffs, true mean 0): "
          f"CS excluded zero on {f1_rate:.2f}% of paths over 25y "
          f"(bar: <= 5.00%) -> {'PASS' if f1_pass else 'FAIL'}")

    shifted = null + 0.001
    f2_firsts, _ = _first_exclusions(bootstrap_paths(shifted, TRADING_DAYS["5y"],
                                                     N_PATHS, seed=782))
    f2_rate = 100.0 * np.mean(np.isfinite(f2_firsts))
    f2_pass = f2_rate >= 90.0
    print(f"F2 power on a known +0.001/day (+36.5%/yr simple) effect: "
          f"CS excluded zero on {f2_rate:.2f}% of paths within 5y "
          f"(bar: >= 90.00%) -> {'PASS' if f2_pass else 'FAIL'}")

    # ------------------------------------------------------- classification
    print("\n" + "=" * 70)
    print("PRE-REGISTERED CLASSIFICATION (decided on inner-val @ 0.40% live tier)")
    print("=" * 70)
    decisive = next(r for r in rows if r["tag"] == "inner-val | 0.40% (live)")
    n50, by5 = decisive["n50"], decisive["by_5y"]
    if n50 <= TRADING_DAYS["3y"] and by5 >= 50.0:
        verdict = "ON TRACK"
    elif n50 <= TRADING_DAYS["25y"]:
        verdict = "SLOW BUT VIABLE"
    else:
        verdict = "NOT VIABLE AS SPECIFIED"
    print(f"n50 = {n50:,.0f} days, 5-year firing rate = {by5:.1f}%  ->  {verdict}")
    if not (f1_pass and f2_pass):
        print("NOTE: a falsification test FAILED - the horizon above is not "
              "to be believed and this round reports the machinery failure.")

    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False))

    _post_hoc_diagnostics(diff_store, f2_pass)

    print(f"\nconfigs evaluated (real-data backtest runs): {CONFIGS}")
    print(f"bootstrap paths simulated (not real-data configs): "
          f"{N_PATHS * (len(rows) + 3):,}")


def _post_hoc_diagnostics(diff_store: dict, f2_pass: bool) -> None:
    """Everything below was written AFTER the pre-registered run, to explain
    F2's failure. It changes no threshold and reverses no classification.

    Declared post-hoc explicitly, per ROUTINE.md: going back to fix a
    misunderstanding is allowed, going back to move a bar is not. The
    classification above stands exactly as pre-registered; these numbers
    only say *why* F2 could never have passed and whether the headline
    conclusion survives that.
    """
    print("\n" + "=" * 70)
    print("POST-HOC DIAGNOSTICS (written after the run; no threshold moved)")
    print("=" * 70)

    print(
        "\nWhy F2 failed. The bar was set on the effect's ANNUALIZED size\n"
        "(+0.001/day ~ +36.5%/yr, larger than anything this project has\n"
        "measured) without checking it against the difference series' own\n"
        "daily NOISE. Two paper accounts on the same instrument still differ\n"
        "by ~3% on a typical day, so +0.001/day is a t-statistic of ~1.4 at\n"
        "five years - below significance for ANY valid test, sequential or\n"
        "not. F2 was therefore a test of the effect size, not of the\n"
        "machinery, and no correctly-implemented tool could have passed it.\n"
        "It stays on the record as FAILED."
    )

    # A conservative tool can only make a horizon LONGER than the truth. The
    # fixed-n t-test is the optimistic bound no valid sequential test can
    # beat - if even IT needs centuries, the conclusion is not an artifact
    # of the confidence sequence's conservatism.
    print("\nFixed-n reference (look-once, invalid for a growing record, and\n"
          "therefore a LOWER BOUND on any honest sequential horizon):")
    ref = []
    for (wname, fname), d in diff_store.items():
        mu, sd = float(d.mean()), float(d.std(ddof=1))
        n_needed = (1.96 * sd / abs(mu)) ** 2 if mu != 0 else float("inf")
        ref.append({"cell": f"{wname} | {fname}",
                    "mean_per_day": mu, "sd_per_day": sd,
                    "fixed_n_days": n_needed, "fixed_n_years": n_needed / 365.0})
    print(pd.DataFrame(ref).to_string(index=False,
                                      float_format=lambda v: f"{v:,.6g}"))

    # And the same power check at an effect the data could actually carry.
    base = diff_store[("inner-val", "0.40% (live)")].to_numpy()
    null = base - base.mean()
    sd = float(np.std(null, ddof=1))
    big = 3.0 * sd / np.sqrt(TRADING_DAYS["5y"]) * 1.96   # ~t=5.9 at 5 years
    f2b = _first_exclusions(bootstrap_paths(null + big, TRADING_DAYS["5y"],
                                            N_PATHS, seed=783))[0]
    print(f"\nF2' (post-hoc, NOT the pre-registered bar): at a genuinely\n"
          f"detectable +{big:.5f}/day the same tool excluded zero on "
          f"{100.0 * np.mean(np.isfinite(f2b)):.1f}% of paths within 5y - so\n"
          f"the machinery has power when the effect has any, and F1's clean\n"
          f"calibration plus this says the long horizons above are a property\n"
          f"of the data, not of the tool. (f2_pass as pre-registered: "
          f"{f2_pass}.)")


if __name__ == "__main__":
    main()
