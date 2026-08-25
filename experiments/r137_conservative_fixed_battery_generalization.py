"""R-137 CONSERVATIVE branch: does R-127's frozen fixed excision battery
(named-event windows, a data-driven low-BTC/ETH-correlation-day filter, and
their union) generalize past the ONE construction R-127 tested, and does the
real excision beat a random-day placebo control R-127 itself never ran?

Full pre-registration lives in `r137_shared.py`'s own module docstring
(frozen, read-only here, including its ADDENDUM narrowing `IN_SCOPE` to four
constructions). One-paragraph restatement of just this branch's design, for
a reader of this file alone:

For each of `r137_shared.IN_SCOPE` (`R-113`, `R-115-conservative`,
`R-125-conservative`, `R-126-conservative`), reproduce the construction's own
candidate/baseline daily return series via `r137_loaders.CONSTRUCTIONS`
(already-built, already self-checked against each construction's own
published number -- not reimplemented here), compute `gap_before =
r137_shared.gap_sharpe(...)`, then apply THREE excision variants using
excluded-day sets built EXACTLY as R-127 built them (`r127_shared`'s own
`TERRA_LUNA_WINDOW` / `THE_MERGE_WINDOW` / `low_correlation_days`, imported
verbatim, no new constant chosen after seeing any R-137 number): (a) named
events, (b) low-BTC/ETH-correlation days, (c) their union. For each variant,
re-score with `r137_shared.excise_and_regap`, run the random-day placebo
control at the SAME excised-day count (`r137_shared.random_day_placebo` +
`placebo_pvalue`), and classify with `r137_shared.classify_movement`. Round-
level verdicts (`r137_shared.round_verdict`) are computed separately per
excision variant.

**R-113 caveat, carried forward from `r137_shared`/`r137_loaders`:** R-113's
own cell is BASKET-level (8-asset panel equity curve), not an isolated ETH
comparison -- reported in the same table, flagged, never averaged in with
the three isolated-ETH cells un-flagged. Its own correlation filter is
computed on plain BTC/ETH spot price frames, independent of the panel's own
8-asset composition, per `r137_shared`'s own R-113 caveat.

**R-125-conservative caveat, carried forward from `r137_shared`'s own
(pre-addendum) Decision rule section:** B1-level, not B4-level -- it never
reached a clean BTC/ETH inversion to narrow in the first place. Reported in
the same 4-way table for completeness, but explicitly EXCLUDED from the
round-level majority count for every excision variant (named here, before
any number below was read, so it cannot be quietly included or excluded
after the fact depending on which reading favors the round's own outcome --
this is the module docstring's own instruction, unaffected by the later
addendum which only changed `IN_SCOPE` membership and confirmed `MAJORITY_K`
needs no recalibration for n=4).

Which ETH raw price frame feeds each construction's own low-correlation
filter (per this round's own instruction: reuse whichever raw ETH frame the
construction's own loader already loads internally, not a fresh reload of
some other window):
  - R-113:              `r127_shared.load_eth_train()` (plain ETH spot,
                         INNER_VAL-restricted, independent of the panel's own
                         8-asset composition -- same BTC/ETH pairing R-127's
                         own novel branch used for its low-corr filter).
  - R-115-conservative:  `r115_conservative_shared.load_eth_coinbase()`
                         (same function `load_r115_conservative()` calls).
  - R-125-conservative:  `r125_shared.load_eth_train()` (same function
                         `load_r125_conservative()` calls).
  - R-126-conservative:  `r126_shared.load_eth_train()` (same function
                         `load_r126_conservative()` calls).
BTC's leg is `r127_shared.load_btc_train("spot")` throughout, for every
construction -- one shared BTC daily log-return series, matching R-127's own
novel branch's own low-corr calculation.

No bar dated `OOS_START = 2023-01-01` or later is read anywhere in this file
-- every loader above (both `r137_loaders.CONSTRUCTIONS` and every raw ETH/
BTC reload here) truncates and asserts internally; nothing here re-derives
or retunes any excision constant.

Run: `python experiments/r137_conservative_fixed_battery_generalization.py`
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments import r137_shared as r137  # noqa: E402  frozen, read-only
from experiments import r127_shared as r127  # noqa: E402  frozen, read-only
from experiments.r137_loaders import CONSTRUCTIONS  # noqa: E402  already-built, already-tested

VARIANTS = ("named", "low_corr", "union")
VARIANT_LABEL = {
    "named": "(a) named-event excision (Terra/Luna + The Merge)",
    "low_corr": "(b) low-BTC/ETH-correlation-day excision",
    "union": "(c) union of (a) and (b)",
}

# R-125-conservative is reported in every table below but explicitly
# excluded from the round-level majority count, per r137_shared's own
# (pre-addendum) Decision rule section -- named here, before any number is
# read, so it cannot be quietly included/excluded after the fact.
MAJORITY_SCOPE = [c for c in r137.IN_SCOPE if c != "R-125-conservative"]


# ----------------------------------------------------------------------
# Excluded-day sets, built EXACTLY as R-127 built them
# (experiments/r127_novel_event_excision_retest.py's own
# named_event_days() / data_driven_low_corr_days()) -- no new constant, no
# re-derivation, imported verbatim from r127_shared.
# ----------------------------------------------------------------------

def named_event_days() -> pd.DatetimeIndex:
    """Calendar days covered by TERRA_LUNA_WINDOW + THE_MERGE_WINDOW,
    inclusive, byte-identical construction to R-127 novel's own
    `named_event_days()`."""
    parts = []
    for start, end in (r127.TERRA_LUNA_WINDOW, r127.THE_MERGE_WINDOW):
        parts.append(pd.date_range(start, end, freq="1D"))
    return pd.DatetimeIndex(np.concatenate([p.values for p in parts])).unique()


NAMED_DAYS = named_event_days()


def _assert_no_holdout(df: pd.DataFrame, label: str) -> None:
    last = df.index[-1]
    ok = last < pd.Timestamp(r127.OOS_START, tz=last.tz)
    assert ok, f"holdout breach in {label}: last bar {last}"


def _eth_raw_frame(name: str) -> pd.DataFrame:
    """The same raw ETH OHLCV frame each construction's own loader already
    loads internally (re-loaded here a second time, cheaply, only to get a
    PRICE-level series for the correlation filter -- candidate_daily/
    baseline_daily are strategy RETURN series, not price series)."""
    if name == "R-113":
        # Plain ETH spot, independent of the panel's own 8-asset
        # composition, per r137_shared's own R-113 caveat.
        eth = r127.load_eth_train()
    elif name == "R-115-conservative":
        from experiments.r115_conservative_shared import load_eth_coinbase
        eth = load_eth_coinbase()
    elif name == "R-125-conservative":
        from experiments import r125_shared
        eth = r125_shared.load_eth_train()
    elif name == "R-126-conservative":
        from experiments import r126_shared
        eth = r126_shared.load_eth_train()
    else:
        raise ValueError(f"no ETH raw-frame rule for {name!r}")
    _assert_no_holdout(eth, f"{name} ETH raw price frame (correlation filter)")
    return eth


def low_corr_days_for(name: str) -> pd.DatetimeIndex:
    btc_df, _ = r127.load_btc_train("spot")
    _assert_no_holdout(btc_df, f"{name} BTC raw price frame (correlation filter)")
    eth_df = _eth_raw_frame(name)

    btc_daily = r127.daily_log_returns(btc_df)
    eth_daily = r127.daily_log_returns(eth_df)
    return r137.low_correlation_days(btc_daily, eth_daily, r137.CORR_WINDOW_DAYS,
                                      r137.LOW_CORR_THRESHOLD)


# ----------------------------------------------------------------------
# R-113 sanity check: do the named-event windows actually fall inside
# R-113's own W_VAL (2022-01-01..2022-12-31)? Both events are dated in
# 2022, and W_VAL is calendar year 2022, so they should -- checked, not
# assumed, before any excision runs.
# ----------------------------------------------------------------------

def sanity_check_r113_window() -> None:
    from experiments.r63_shared import W_VAL

    w_start, w_end = pd.Timestamp(W_VAL[0]), pd.Timestamp(W_VAL[1])
    print("R-113 sanity check: do the named-event windows fall inside R-113's own W_VAL?")
    print(f"  W_VAL = {w_start.date()} .. {w_end.date()}")
    for label, (start, end) in (("TERRA_LUNA_WINDOW", r127.TERRA_LUNA_WINDOW),
                                 ("THE_MERGE_WINDOW", r127.THE_MERGE_WINDOW)):
        inside = (start >= w_start) and (end <= w_end)
        print(f"  {label:20s} {start.date()} .. {end.date()}  inside W_VAL: {inside}")
        assert inside, (
            f"{label} ({start.date()}..{end.date()}) does NOT fall inside R-113's own "
            f"W_VAL ({w_start.date()}..{w_end.date()}) -- stop, this invalidates the "
            "named-event excision for R-113's basket-level cell")
    print("  PASS: both named-event windows fall inside R-113's own W_VAL (2022).\n")


# ----------------------------------------------------------------------
# Per-construction, per-variant evaluation.
# ----------------------------------------------------------------------

def evaluate_construction(name: str) -> dict:
    print("=" * 100)
    print(f"{name}")
    print("=" * 100)

    load = CONSTRUCTIONS[name]()
    candidate_daily, baseline_daily = load["candidate_daily"], load["baseline_daily"]
    published_gap = load["published_gap"]
    gap_before = r137.gap_sharpe(candidate_daily, baseline_daily)

    print(f"  note: {load['note']}")
    print(f"  published_gap (construction's own, see note above) = {published_gap:+.4f}")
    print(f"  gap_before (this round's own daily-resampled reference point) = {gap_before:+.4f}")

    low_corr_days = low_corr_days_for(name)
    union_days = pd.DatetimeIndex(NAMED_DAYS).union(pd.DatetimeIndex(low_corr_days))
    overlap = len(NAMED_DAYS) + len(low_corr_days) - len(union_days)
    print(f"  named-event days (calendar, Terra/Luna + The Merge): {len(NAMED_DAYS)}")
    print(f"  low-correlation days (this construction's own BTC/ETH pairing): {len(low_corr_days)}")
    print(f"  union: {len(union_days)} distinct days (overlap: {overlap})")

    day_sets = {"named": NAMED_DAYS, "low_corr": low_corr_days, "union": union_days}

    rows = {}
    for variant in VARIANTS:
        days = day_sets[variant]
        result = r137.excise_and_regap(candidate_daily, baseline_daily, days)
        gap_after = result["gap_after"]
        n_excised = result["n_excised"]

        placebo_draws = r137.random_day_placebo(candidate_daily, baseline_daily,
                                                  n_exclude=n_excised)
        placebo_p = r137.placebo_pvalue(gap_before, gap_after, placebo_draws)
        classification = r137.classify_movement(gap_before, gap_after, placebo_p)

        rows[variant] = dict(
            n_before=result["n_before"], n_after=result["n_after"], n_excised=n_excised,
            gap_before=gap_before, gap_after=gap_after,
            boot_lo=result["boot_lo"], boot_hi=result["boot_hi"],
            placebo_p=placebo_p, classification=classification,
        )
        print(f"\n  -- {VARIANT_LABEL[variant]} --")
        print(f"     n_excised={n_excised}  (n_before={result['n_before']} -> "
              f"n_after={result['n_after']})")
        print(f"     gap_before={gap_before:+.4f}  gap_after={gap_after:+.4f}  "
              f"boot_CI=[{result['boot_lo']:+.4f},{result['boot_hi']:+.4f}]")
        print(f"     placebo_p={placebo_p:.4f} (n_draws={r137.N_PLACEBO_DRAWS})  "
              f"-> classification={classification}")

    print()
    return dict(name=name, published_gap=published_gap, gap_before=gap_before,
                n_named=len(NAMED_DAYS), n_low_corr=len(low_corr_days), n_union=len(union_days),
                rows=rows)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def print_table(results: dict) -> None:
    print("=" * 100)
    print("4x3 CLASSIFICATION TABLE (construction x excision variant)")
    print("=" * 100)
    hdr = (f"{'construction':22s} {'variant':10s} {'n_exc':>6s} {'gap_before':>11s} "
           f"{'gap_after':>10s} {'boot_lo':>9s} {'boot_hi':>9s} {'placebo_p':>10s}  classification")
    print(hdr)
    print("-" * len(hdr))
    for name in r137.IN_SCOPE:
        r = results[name]
        for variant in VARIANTS:
            row = r["rows"][variant]
            flag = "  [R-113: BASKET-level, not isolated ETH]" if name == "R-113" else \
                   ("  [R-125-conservative: B1-level, excluded from majority count]"
                    if name == "R-125-conservative" and variant == "named" else "")
            print(f"{name:22s} {variant:10s} {row['n_excised']:>6d} {row['gap_before']:>+11.4f} "
                  f"{row['gap_after']:>+10.4f} {row['boot_lo']:>+9.4f} {row['boot_hi']:>+9.4f} "
                  f"{row['placebo_p']:>10.4f}  {row['classification']}{flag}")
        print()


def round_verdicts(results: dict) -> dict:
    print("=" * 100)
    print("ROUND-LEVEL VERDICTS (one per excision variant)")
    print("=" * 100)
    print(f"  Majority-count scope (R-125-conservative excluded per r137_shared's own "
          f"Decision rule -- B1-level, never reached a clean inversion to narrow): "
          f"{MAJORITY_SCOPE}")
    print(f"  MAJORITY_K={r137.MAJORITY_K}  (of {len(MAJORITY_SCOPE)} counted constructions)\n")

    verdicts = {}
    for variant in VARIANTS:
        per_construction = {name: results[name]["rows"][variant]["classification"]
                             for name in r137.IN_SCOPE}
        verdict = r137.round_verdict(per_construction, MAJORITY_SCOPE)
        verdicts[variant] = verdict
        hits = [name for name in MAJORITY_SCOPE
                if per_construction[name] in ("GENERALIZES", "SIGN_FLIP")]
        print(f"  {VARIANT_LABEL[variant]}")
        print(f"    classifications (all 4, incl. R-125-conservative for completeness): "
              f"{per_construction}")
        print(f"    hits within majority scope: {hits}  ({len(hits)}/{len(MAJORITY_SCOPE)})")
        print(f"    -> VERDICT: {verdict}\n")
    return verdicts


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print("R-137 CONSERVATIVE branch: fixed-battery excision generalization test")
    print("=" * 100)
    print(f"IN_SCOPE = {r137.IN_SCOPE}")
    print(f"EXCLUDED = {r137.EXCLUDED}\n")

    sanity_check_r113_window()

    results = {}
    for name in r137.IN_SCOPE:
        results[name] = evaluate_construction(name)

    print_table(results)
    verdicts = round_verdicts(results)

    n_constructions = len(r137.IN_SCOPE)
    n_variants = len(VARIANTS)
    n_configs = n_constructions * n_variants
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  Constructions evaluated: {n_constructions} ({r137.IN_SCOPE})")
    print(f"  Excision variants per construction: {n_variants} {VARIANTS}")
    print(f"  Total (construction x variant) configs evaluated: {n_configs}")
    print(f"  Random-day placebo draws per config: {r137.N_PLACEBO_DRAWS} "
          f"(seed={r137.PLACEBO_SEED}) -> {n_configs * r137.N_PLACEBO_DRAWS} total placebo draws")
    print(f"  Round-level verdicts: named={verdicts['named']}  "
          f"low_corr={verdicts['low_corr']}  union={verdicts['union']}")

    print("\n  One-paragraph summary:")
    all_class = {(name, v): results[name]["rows"][v]["classification"]
                 for name in r137.IN_SCOPE for v in VARIANTS}
    n_generalize = sum(1 for c in all_class.values() if c == "GENERALIZES")
    n_flip = sum(1 for c in all_class.values() if c == "SIGN_FLIP")
    n_not = sum(1 for c in all_class.values() if c == "NOT_GENERALIZE")
    print(f"  Across the {n_configs} (construction x variant) cells, {n_generalize} classified "
          f"GENERALIZES, {n_flip} SIGN_FLIP, and {n_not} NOT_GENERALIZE. R-127's own single-"
          f"construction finding (60-78% narrowing) is reported here as a claim about ONE "
          f"construction only; this round's own three round-level verdicts above state "
          f"whether that finding generalizes across the four in-scope constructions and "
          f"survives the random-day placebo control R-127 never ran, per r137_shared's own "
          f"pre-registered decision rule and its R-113/R-125-conservative reporting caveats.")

    print(f"\n  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
