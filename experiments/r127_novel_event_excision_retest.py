"""R-127 NOVEL branch: does excising brief ETH-idiosyncratic divergence
episodes from R-126 novel's own already-recorded B4 comparison flip or
materially narrow the BTC/ETH sign gap?

Full pre-registration lives in `r127_shared.py`'s own module docstring
(frozen, read-only here). One-paragraph restatement of just this branch's
design, for a reader of this file alone:

R-127's window scan (`r127_shared.SCAN`, computed before either branch
script existed) already found the calendar-matched ETH `INNER_VAL` window
(2021-01-01..2022-12-31) is the SINGLE BEST regime-fingerprint match to
BTC's own `INNER_VAL` among 95 candidates -- refuting the coarse
"wrong window" hypothesis before any strategy number was touched. This
branch tests the finer hypothesis instead: *within* that (correctly
chosen) window, do a handful of brief, ETH-idiosyncratic, structurally
dated divergence episodes -- Terra/Luna's collapse and The Merge, plus a
data-driven low-BTC/ETH-correlation-day filter -- disproportionately
drive the sign flip that R-126 novel's CVaR-budgeted `champions_council`
reallocation showed (BTC spot `d_sharpe=+0.388`, ETH spot
`d_sharpe=-0.530`, both already clearing the +/-0.2 noise floor)? No
refit, no new mechanism: R-126 novel's own frozen fit is re-evaluated
under a POST-HOC calendar-day filter of its already-realized DAILY return
series (`r127_shared.excise_days` -- never a mutation of the 5m OHLCV bars
fed to the backtest engine, per that function's own docstring).

Falsification test, pre-registered (r127_shared.py): does excising (a)
`TERRA_LUNA_WINDOW` + `THE_MERGE_WINDOW`, (b) data-driven low-correlation
days, or (c) their union, flip ETH's `d_sharpe` sign to match BTC's, or
materially narrow the gap? A gap that is unchanged or widens refutes the
idiosyncratic-divergence hypothesis for this construction.

Reused, read-only, never modified:
  - `r126_shared.py`     -- member signal matrix, payoff construction,
                            `b1_signal`/`run_target_series`/
                            `run_candidate_council` primitives, splits.
  - `r126_novel_cvar_council.py` -- `fit_novel_council` (the frozen CVaR-
                            budgeted weight schedule + target series R-126
                            novel already validated on BTC and ETH).
  - `r127_shared.py`     -- `TERRA_LUNA_WINDOW`, `THE_MERGE_WINDOW`,
                            `low_correlation_days`, `excise_days`,
                            `daily_log_returns`, `load_btc_train`,
                            `load_eth_train` (this module's own, which
                            returns ETH restricted to the calendar-matched
                            INNER_VAL window only -- distinct from
                            `r126_shared.load_eth_train`, which returns
                            ETH's full pre-holdout history; each is used
                            exactly where the task calls for it, see
                            inline comments below).

No bar dated `OOS_START = 2023-01-01` or later is read anywhere in this
file -- both `r126_shared` and `r127_shared`'s own loaders assert this
internally, and this file additionally prints and checks the last-bar
timestamp of every frame it loads (see `_check_no_holdout` below).

This script does NOT use `r126_novel_cvar_council.cached()` -- it calls
`fit_novel_council` directly on ETH's frame every run, so a stale on-disk
cache from a prior `r126_novel_cvar_council.py` invocation (keyed under
`/tmp/.../r126_novel_cache` by default) cannot silently feed this branch
a number computed by different code. Slower, but unambiguous.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import r126_shared as r126               # noqa: E402  frozen, read-only
import r126_novel_cvar_council as novel  # noqa: E402  frozen, read-only
import r127_shared as r127               # noqa: E402  frozen, read-only

from tradebot.inference import (  # noqa: E402
    annualized_sharpe, daily_returns, paired_bootstrap, total_log_return,
)

NOISE_FLOOR = 0.2  # this project's standard +/-0.2 Sharpe noise floor (R-20)


# ----------------------------------------------------------------------
# Holdout discipline: every frame this file reads is checked explicitly,
# on top of the assertions already inside r126_shared/r127_shared's own
# loaders.
# ----------------------------------------------------------------------

def _check_no_holdout(df: pd.DataFrame, label: str) -> None:
    last = df.index[-1]
    ok = last < pd.Timestamp(r127.OOS_START, tz=last.tz)
    print(f"  [holdout check] {label}: last bar {last}  "
          f"(< {r127.OOS_START} = {ok})", flush=True)
    assert ok, f"holdout breach in {label}: last bar {last}"


# ----------------------------------------------------------------------
# Step 1: reproduce R-126 novel's own recorded ETH B4 number, unmodified.
#
# main() in r126_novel_cvar_council.py does exactly this:
#   df_eth = sh.load_eth_train()                       # r126_shared's own,
#                                                        # ETH's FULL pre-
#                                                        # holdout history
#                                                        # (2019-03-14 ->
#                                                        # 2022-12-31), not
#                                                        # just INNER_VAL --
#                                                        # so the CVaR
#                                                        # weight schedule
#                                                        # has real trailing
#                                                        # history rather
#                                                        # than the
#                                                        # equal-weight
#                                                        # fallback through
#                                                        # most of INNER_VAL.
#   fit_eth = fit_novel_council(df_eth)                 # primary config:
#                                                        # alpha=0.05,
#                                                        # lookback=90
#   b1_eth = sh.b1_signal(fit_eth["target"], df_eth, sh.SPOT)
# which internally backtests the candidate and champions_council over
# INNER_VAL_START..INNER_VAL_END only. Replicated here verbatim.
# ----------------------------------------------------------------------

def reproduce_r126_baseline() -> dict:
    print("\n" + "=" * 70, flush=True)
    print("STEP 1: reproduce R-126 novel's own ETH B4 number (self-check)", flush=True)
    print("=" * 70, flush=True)

    df_eth = r126.load_eth_train()  # r126_shared's own -- FULL pre-holdout ETH
    _check_no_holdout(df_eth, "r126_shared.load_eth_train() (full pre-holdout ETH)")

    print(f"\n[fit] CVaR-budgeted council on ETH's full pre-holdout history "
          f"({len(df_eth)} bars)...", flush=True)
    t0 = time.time()
    fit_eth = novel.fit_novel_council(df_eth)  # primary config, alpha=0.05/lookback=90
    print(f"  fit complete in {time.time() - t0:.1f}s", flush=True)

    diag = novel.summarize_diagnostics(fit_eth["diagnostics"])
    print(f"  solver diagnostics: {diag}", flush=True)

    print("\n[B1] candidate (CVaR council) vs champions_council, ETH spot, "
          f"{r126.INNER_VAL_START}..{r126.INNER_VAL_END}", flush=True)
    b1_eth = r126.b1_signal(fit_eth["target"], df_eth, r126.SPOT)
    print(f"  {b1_eth}", flush=True)

    published_d_sharpe = -0.530
    diff = b1_eth["d_sharpe"] - published_d_sharpe
    print(f"\n  R-126 published ETH spot d_sharpe: {published_d_sharpe:+.4f}")
    print(f"  this reproduction's d_sharpe:       {b1_eth['d_sharpe']:+.4f}")
    print(f"  difference:                          {diff:+.4f}")
    same_sign = np.sign(b1_eth["d_sharpe"]) == np.sign(published_d_sharpe)
    close = abs(diff) < 0.05
    print(f"  same sign as published: {same_sign}   |diff| < 0.05: {close}")
    assert same_sign, (
        "reproduction SIGN mismatch vs R-126's published ETH d_sharpe -- "
        "stop and re-check the construction before proceeding")

    return {"df_eth": df_eth, "fit_eth": fit_eth, "b1_eth": b1_eth,
            "published_d_sharpe": published_d_sharpe, "reproduction_diff": diff}


# ----------------------------------------------------------------------
# Step 2: raw daily candidate/reference return series over INNER_VAL, via
# the exact same primitives r126_shared.b1_signal calls internally
# (run_target_series / run_candidate_council / tradebot.inference.
# daily_returns), so the underlying data is identical to Step 1's -- only
# now the daily series themselves are kept for excision rather than being
# reduced immediately to a paired-bootstrap summary.
# ----------------------------------------------------------------------

def get_eth_daily_series(target: np.ndarray, df_eth: pd.DataFrame) -> dict:
    print("\n" + "=" * 70, flush=True)
    print("STEP 2: raw daily candidate/reference series, ETH spot, INNER_VAL", flush=True)
    print("=" * 70, flush=True)

    m_cand, res_cand = r126.run_target_series(
        target, df_eth, r126.SPOT, r126.INNER_VAL_START, r126.INNER_VAL_END)
    m_council, res_council = r126.run_candidate_council(df_eth, r126.SPOT)

    r_cand = daily_returns(res_cand.equity)
    r_council = daily_returns(res_council.equity)
    n = min(len(r_cand), len(r_council))
    r_cand, r_council = r_cand.iloc[:n], r_council.iloc[:n]

    print(f"  candidate daily series:  {len(r_cand)} days, "
          f"{r_cand.index.min().date()}..{r_cand.index.max().date()}", flush=True)
    print(f"  reference daily series:  {len(r_council)} days, "
          f"{r_council.index.min().date()}..{r_council.index.max().date()}", flush=True)
    print(f"  bar-level Sharpe (compute_metrics): cand={m_cand.sharpe:.4f} "
          f"council={m_council.sharpe:.4f}  d_sharpe={m_cand.sharpe - m_council.sharpe:+.4f}",
          flush=True)

    return {"r_cand": r_cand, "r_council": r_council,
            "m_cand": m_cand, "m_council": m_council}


# ----------------------------------------------------------------------
# Excision sets
# ----------------------------------------------------------------------

def named_event_days() -> pd.DatetimeIndex:
    """Calendar days covered by TERRA_LUNA_WINDOW + THE_MERGE_WINDOW,
    inclusive, structural dates from r127_shared (never fit)."""
    parts = []
    for start, end in (r127.TERRA_LUNA_WINDOW, r127.THE_MERGE_WINDOW):
        parts.append(pd.date_range(start, end, freq="1D"))
    return pd.DatetimeIndex(np.concatenate([p.values for p in parts])).unique()


def data_driven_low_corr_days() -> tuple[pd.DatetimeIndex, pd.Series]:
    """Calendar days where trailing 14-day BTC/ETH daily-return correlation
    falls below the structural LOW_CORR_THRESHOLD, computed exactly per the
    task's instruction: BTC's daily log returns over its own pre-holdout
    history (r127_shared.load_btc_train, joined inner so only overlapping
    days matter), ETH's daily log returns restricted to the
    calendar-matched INNER_VAL window (r127_shared.load_eth_train -- NOTE:
    this is r127_shared's own version, INNER_VAL-only, deliberately
    different from r126_shared.load_eth_train used in Step 1/2 above)."""
    btc_df, _ = r127.load_btc_train("spot")
    _check_no_holdout(btc_df, "r127_shared.load_btc_train() (BTC, for correlation calc)")
    eth_val = r127.load_eth_train()  # r127_shared's own: INNER_VAL-only
    _check_no_holdout(eth_val, "r127_shared.load_eth_train() (ETH INNER_VAL-only, for correlation calc)")

    btc_daily = r127.daily_log_returns(btc_df)
    eth_daily = r127.daily_log_returns(eth_val)
    corr = r127.rolling_btc_eth_daily_corr(btc_daily, eth_daily)
    low_days = r127.low_correlation_days(btc_daily, eth_daily)
    return low_days, corr


# ----------------------------------------------------------------------
# Step 3: paired significance three ways, plus the un-excised baseline at
# the same seed for a like-for-like comparison basis.
# ----------------------------------------------------------------------

def evaluate_variant(r_cand: pd.Series, r_council: pd.Series,
                      excluded_days: pd.DatetimeIndex | None, label: str,
                      seed: int = 127) -> dict:
    n_before = len(r_cand)
    if excluded_days is not None and len(excluded_days) > 0:
        r_c = r127.excise_days(r_cand, excluded_days)
        r_r = r127.excise_days(r_council, excluded_days)
    else:
        r_c, r_r = r_cand, r_council

    idx = r_c.index.intersection(r_r.index)
    a = r_c.loc[idx].to_numpy(dtype=float)
    b = r_r.loc[idx].to_numpy(dtype=float)
    n_excised = n_before - len(idx)

    paired = paired_bootstrap(a, b, stat=total_log_return, seed=seed)
    d_sharpe_daily = annualized_sharpe(a) - annualized_sharpe(b)

    row = {
        "label": label, "n_days": len(idx), "n_excised": n_excised,
        "sharpe_daily_cand": annualized_sharpe(a), "sharpe_daily_council": annualized_sharpe(b),
        "d_sharpe_daily": d_sharpe_daily,
        "paired_diff": paired.diff.point, "paired_lo": paired.diff.lo, "paired_hi": paired.diff.hi,
        "p_positive": paired.p_positive, "significant": paired.significant,
    }
    print(f"  [{label}] n_days={row['n_days']} (excised {row['n_excised']})  "
          f"d_sharpe(daily)={row['d_sharpe_daily']:+.4f}  "
          f"paired_diff={row['paired_diff']:+.5f} "
          f"[{row['paired_lo']:+.5f},{row['paired_hi']:+.5f}]  "
          f"significant={row['significant']}", flush=True)
    return row


def run_excision_battery(r_cand: pd.Series, r_council: pd.Series) -> dict:
    print("\n" + "=" * 70, flush=True)
    print("STEP 3: paired significance, three excisions vs un-excised baseline "
          "(all seed=127)", flush=True)
    print("=" * 70, flush=True)

    named_days = named_event_days()
    print(f"\n  named-event days (Terra/Luna + The Merge): {len(named_days)} calendar days", flush=True)

    low_corr_days, corr_series = data_driven_low_corr_days()
    n_valid_corr = int(corr_series.notna().sum())
    print(f"  data-driven low-correlation days: {len(low_corr_days)} of "
          f"{n_valid_corr} days with a defined trailing-14d correlation "
          f"(threshold={r127.LOW_CORR_THRESHOLD})", flush=True)

    union_days = pd.DatetimeIndex(named_days).union(pd.DatetimeIndex(low_corr_days))
    overlap = len(named_days) + len(low_corr_days) - len(union_days)
    print(f"  union: {len(union_days)} distinct days "
          f"(overlap between the two sets: {overlap} days)", flush=True)

    print()
    baseline = evaluate_variant(r_cand, r_council, None, "baseline (unexcised)")
    variant_a = evaluate_variant(r_cand, r_council, named_days, "(a) named-event excision")
    variant_b = evaluate_variant(r_cand, r_council, low_corr_days, "(b) low-correlation-day excision")
    variant_c = evaluate_variant(r_cand, r_council, union_days, "(c) union excision")

    return {"baseline": baseline, "a_named": variant_a, "b_lowcorr": variant_b,
            "c_union": variant_c, "named_days": named_days,
            "low_corr_days": low_corr_days, "union_days": union_days,
            "corr_series": corr_series}


# ----------------------------------------------------------------------
# Step 4: verdict against the pre-registered falsification test.
# ----------------------------------------------------------------------

def classify_movement(baseline_d: float, variant_d: float) -> str:
    """Classify a variant's daily-Sharpe gap relative to baseline, using
    this project's own +/-0.2 Sharpe noise floor as the "material" bar."""
    if np.sign(variant_d) != np.sign(baseline_d) and variant_d != 0:
        return "SIGN FLIP"
    narrowed = abs(variant_d) - abs(baseline_d)  # negative = narrower
    if narrowed <= -NOISE_FLOOR:
        return "materially narrowed"
    if narrowed >= NOISE_FLOOR:
        return "widened"
    return "unchanged (within noise floor)"


def render_verdict(battery: dict, published_bar_level_d_sharpe: float) -> str:
    print("\n" + "=" * 70, flush=True)
    print("STEP 4: verdict against the pre-registered falsification test", flush=True)
    print("=" * 70, flush=True)

    baseline_d = battery["baseline"]["d_sharpe_daily"]
    print(f"\n  baseline (unexcised) daily-return d_sharpe: {baseline_d:+.4f}  "
          f"(bar-level d_sharpe from Step 1/2: {published_bar_level_d_sharpe:+.4f})", flush=True)

    movements = {}
    for key, name in (("a_named", "(a) named-event"), ("b_lowcorr", "(b) low-correlation"),
                       ("c_union", "(c) union")):
        row = battery[key]
        move = classify_movement(baseline_d, row["d_sharpe_daily"])
        movements[key] = move
        print(f"  {name:22s}: d_sharpe(daily)={row['d_sharpe_daily']:+.4f}  "
              f"delta_vs_baseline={row['d_sharpe_daily'] - baseline_d:+.4f}  -> {move}", flush=True)

    any_flip_or_narrow = any(m in ("SIGN FLIP", "materially narrowed") for m in movements.values())

    print()
    if any_flip_or_narrow:
        verdict = ("CONFIRMED (partial/full): at least one excision flips the sign or "
                   "materially narrows the BTC/ETH d_sharpe gap -- supports the "
                   "idiosyncratic-divergence hypothesis for this construction.")
    else:
        verdict = ("REFUTED for this construction: no excision flips the sign or "
                   "materially narrows the gap (every variant is unchanged-within-noise-floor "
                   "or wider) -- the inversion is not explained by brief ETH-idiosyncratic "
                   "divergence episodes for R-126 novel's CVaR-budgeted champions_council "
                   "reallocation.")
    print(f"  VERDICT: {verdict}", flush=True)
    return verdict


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    n_configs = 0

    print("=" * 70, flush=True)
    print("R-127 NOVEL branch: event-excision retest of R-126 novel's ETH B4", flush=True)
    print("=" * 70, flush=True)

    step1 = reproduce_r126_baseline()
    n_configs += 1  # (1) baseline reproduction

    step2 = get_eth_daily_series(step1["fit_eth"]["target"], step1["df_eth"])

    battery = run_excision_battery(step2["r_cand"], step2["r_council"])
    n_configs += 3  # (2)(3)(4) named-event / low-corr / union excision paired tests
    # note: "baseline (unexcised)" inside run_excision_battery reuses the
    # same daily series as the Step-1 reproduction (just re-evaluated at
    # seed=127 rather than seed=126, for a like-for-like basis against the
    # three excisions) -- not counted again as a distinct configuration.

    published_bar_level_d_sharpe = step1["b1_eth"]["d_sharpe"]
    verdict = render_verdict(battery, published_bar_level_d_sharpe)

    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  Step 1 reproduction: this run d_sharpe={step1['b1_eth']['d_sharpe']:+.4f}  "
          f"vs R-126 published {step1['published_d_sharpe']:+.4f}  "
          f"(diff={step1['reproduction_diff']:+.4f})", flush=True)
    for key, name in (("baseline", "baseline (unexcised, seed=127)"),
                       ("a_named", "(a) named-event excision"),
                       ("b_lowcorr", "(b) low-correlation excision"),
                       ("c_union", "(c) union excision")):
        row = battery[key]
        print(f"  {name:32s}: d_sharpe(daily)={row['d_sharpe_daily']:+.4f}  "
              f"paired_diff={row['paired_diff']:+.5f} significant={row['significant']}",
              flush=True)
    print(f"\n  Configurations evaluated this run: {n_configs}", flush=True)
    print("    1. baseline reproduction (Step 1, R-126 novel's own primary "
          "config, ETH spot, unexcised, seed=126)", flush=True)
    print("    2. (a) named-event excision paired test (seed=127)", flush=True)
    print("    3. (b) data-driven low-correlation-day excision paired test (seed=127)", flush=True)
    print("    4. (c) union-of-(a)-and-(b) excision paired test (seed=127)", flush=True)
    print(f"  Elapsed: {time.time() - t0:.1f}s", flush=True)
    print(f"\n  VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
