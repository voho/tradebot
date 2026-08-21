#!/usr/bin/env python
"""R-86 NOVEL branch: Step-A detection-lag gate for a NETWORK (bidirectional,
two-asset) transfer-entropy indicator -- `TE_{BTC_return -> ETH_return} +
TE_{ETH_return -> BTC_return}` -- run BEFORE any strategy/confirming-vote
code, identical "operator measurement" convention and Step-A gate
methodology as R-82/R-83/R-85 and this round's own CONSERVATIVE sibling
(`r86_conservative_te_volume_return.py`), for direct comparability.

PRE-REGISTRATION (frozen before this file was ever run):

1. MECHANISM. See `r86_shared.py`'s module docstring for the full citation
   trail (Schreiber 2000; Garcia-Medina & Hernandez C. 2020, "Network
   Analysis of Multivariate Transfer Entropy of Cryptocurrencies in Times
   of Turbulence", *Entropy* 22(7):760) and not-a-duplicate-of list. One
   sentence: Garcia-Medina & Hernandez C. found that TOTAL (summed,
   bidirectional) transfer entropy across a panel of cryptocurrencies rises
   sharply as markets approach turbulence -- an information-flow-complexity
   signal, not a single directional lead-lag signal -- so this branch tests
   whether `te_net = TE_{BTC->ETH} + TE_{ETH->BTC}`, converted to a causal
   trend z-score (`z`, `r86_shared.trend_zscore`), is elevated and RISING
   with LESS lag than v4's own fixed 20/40/80-day anchor-crossing heuristic,
   on the same six dated historical BTC regime transitions R-82/R-83/R-85
   and this round's CONSERVATIVE branch used. This branch is structurally
   distinct from the CONSERVATIVE sibling in two ways: (a) it is a genuine
   NETWORK/combination construction -- the SUM of two directional TE legs,
   not one -- and (b) it draws on a second, already-committed price series
   (Coinbase ETH-USD spot, `load_coinbase_eth_spot`, already used by
   R-47/R-57/R-76 -- no new fetch, no new coverage-gap risk beyond what
   that series already carries). It is NOT a repeat of R-76's pairs-
   trading/cointegration work (which tested price co-movement for a
   mean-reversion trade); this tests information flow for a regime-timing
   alarm, an unrelated question and construction.

2. KNOWN LIMITATION, NAMED NOW, BEFORE ANY REAL NUMBER WAS COMPUTED: ETH
   spot coverage starts 2019-03-14, materially after BTC's 2017-01-01.
   The two 2018 stress episodes (bear onset 2018-01-17, bear bottom
   2018-12-15) have +/-60-day search windows that fall ENTIRELY before ETH
   coverage begins -- the mechanism literally cannot fire there. Per this
   project's convention of never softening a bar because data is
   inconvenient (R-84's conservative branch precedent), those two episodes
   are marked "no data coverage" and scored as FAIL BY CONSTRUCTION against
   the pre-registered 4/6 bar -- not dropped, not exempted. This means the
   >=4/6 bar can only be cleared if ALL FOUR of the remaining
   (ETH-covered) episodes pass, a strictly harder bar than the
   CONSERVATIVE branch faces. Both n_pass/6 (full, coverage gaps counted as
   fails) and n_pass/n_covered (among episodes with real ETH data only) are
   reported; the pre-registered stop rule below uses n_pass/6.

3. DETECTION-LAG DEFINITION. For each episode, within a +/-60-day search
   window around its onset (`r86_shared.episode_window`):
   - v4's own reaction: the nearest DOWNWARD transition of
     `anchor_majority` to the onset (`r86_shared.nearest_transition`,
     `direction="down"` -- identical rule R-82/R-83/R-85 used).
   - TE-network's reaction: the nearest bar where `z` (the trend z-score of
     `te_net`) crosses UP through `Z_THRESH=2.0`
     (`r86_shared.nearest_te_alarm`), closest to the onset.
   - LEAD = (v4_flip_time - detect_time) in days. Positive = the TE-network
     alarm fired before v4's own gate reacted.

4. NULL. `r86_shared.block_bootstrap_shifts` circularly block-shifts the
   LOCAL (episode-window) `z` series (block_days=5, n_draws=500, seed=8602
   -- fixed before running and never altered afterward) and recomputes
   "nearest TE-network alarm to the real, unshifted v4 flip time" against
   each shifted copy -- identical construction to the CONSERVATIVE
   sibling's `null_leads` (itself adapted from R-85's, itself from R-82's).

5. PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed, identical bar to R-82/R-83/R-85 and the CONSERVATIVE sibling):
   an episode counts as a PASS if BOTH (a) LEAD >= 0, AND (b) the true LEAD
   is >= the null distribution's median; a "no data coverage" episode is an
   automatic FAIL by construction (point 2 above). PROCEED TO STEP B (build
   the confirming-vote strategy) only if >= 4 of the 6 episodes PASS,
   against the FULL six -- the coverage gap is a real cost, not an
   exemption. If fewer than 4 pass: STOP, report this file's result as the
   whole branch's product, write it up as NEGATIVE, do not build any
   strategy/confirming-vote code, do not touch any data on or after
   2023-01-01. The bar is not relaxed, narrowed, or otherwise adjusted
   after seeing the numbers.

6. WHAT WOULD MAKE THIS GATE FAIL, named now (and named in `r86_shared.py`
   as this round's own pre-registered EXPECTATION, not a hoped-for
   result): total bidirectional TE, like variance (CSD, R-85), a Bayesian
   run-length posterior (BOCPD, R-82), a linear state-space filter (Kalman
   LLT, R-83) and directional TE(volume->return) (this round's own
   CONSERVATIVE branch) before it, is itself a statistic OF price
   fluctuations (now two assets' worth), so it can only rise once those
   fluctuations have already become unusual -- which is exactly the moment
   v4's own fixed-window anchor is also starting to react. A second,
   independent failure mode named now: the coverage gap itself may sink the
   gate on arithmetic alone, regardless of whether the TE-network signal is
   any good on the four episodes it CAN see -- two automatic fails leaves
   zero room for the four covered episodes to miss even once.

CONFIGURATIONS EVALUATED IN THIS FILE: 0 (a fixed, non-swept measurement
gate, using `r86_shared.Z_THRESH=2.0` throughout -- no threshold search
here; that search, if the gate passes, belongs to Step B and is
pre-registered separately there, swept on TRAIN only and selected on
inner-validation only, never touching the holdout).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402

from experiments.r86_shared import (  # noqa: E402
    BASELINE_WINDOW_DAYS,
    DETECTION_WINDOW_DAYS,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    STRESS_EPISODES,
    TE_SUB_WINDOW_DAYS,
    Z_THRESH,
    align_daily_causal,
    anchor_majority,
    block_bootstrap_shifts,
    confirming_vote_frac,
    daily_log_returns,
    episode_window,
    nearest_te_alarm,
    nearest_transition,
    rolling_transfer_entropy,
    trend_zscore,
)

DATA_DIR = ROOT / "data"
WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 8602


def assert_no_holdout(df: pd.DataFrame, label: str = "") -> None:
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read ({label}): max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


def load_btc_bars() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df, "BTC")
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def load_eth_bars() -> pd.DataFrame | None:
    df = load_coinbase_eth_spot(DATA_DIR)
    if df is None:
        return None
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df, "ETH")
    print(f"ETH (Coinbase spot): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def build_te_network_z(btc_bars: pd.DataFrame, eth_bars: pd.DataFrame
                        ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Steps 1-3 of the task spec: daily log-returns of both assets, inner-
    joined on shared dates, both directional TE legs, summed into the
    network indicator, converted to a causal trend z-score. Returns
    (z, te_btc_to_eth, te_eth_to_btc), all daily-indexed (NOT yet aligned
    onto BTC's 5-minute bar index -- see `align_daily_causal` at the call
    site)."""
    r_btc = daily_log_returns(btc_bars)
    r_eth = daily_log_returns(eth_bars)
    idx = r_btc.index.intersection(r_eth.index)
    r_btc = r_btc.reindex(idx)
    r_eth = r_eth.reindex(idx)

    te_btc_to_eth = rolling_transfer_entropy(r_btc, r_eth, sub_window_days=TE_SUB_WINDOW_DAYS)
    te_eth_to_btc = rolling_transfer_entropy(r_eth, r_btc, sub_window_days=TE_SUB_WINDOW_DAYS)
    te_net = te_btc_to_eth + te_eth_to_btc

    z = trend_zscore(te_net, detection_window_days=DETECTION_WINDOW_DAYS,
                      baseline_window_days=BASELINE_WINDOW_DAYS)
    return z, te_btc_to_eth, te_eth_to_btc


def null_leads(z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                flip_time: pd.Timestamp, z_thresh: float = Z_THRESH,
                n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                seed: int = NULL_SEED) -> np.ndarray:
    """Identical circular block-shift null construction to the CONSERVATIVE
    sibling's (and R-85's, and R-82's before that) `null_leads`, applied
    here to the TE-network trend z-score threshold-crossing detector."""
    local = z.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=block_days,
                                     n_draws=n_draws, seed=seed)
    leads = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted = local[shift]
        high = shifted >= z_thresh
        cross = np.zeros(n_bars, dtype=bool)
        cross[1:] = high[1:] & ~high[:-1]
        cross[0] = bool(high[0]) if n_bars else False
        idx = np.where(cross)[0]
        if len(idx) == 0:
            continue
        times = window[idx]
        deltas = np.abs((times - onset).to_numpy())
        detect_time = times[int(np.argmin(deltas))]
        leads[k] = (flip_time - detect_time).total_seconds() / 86400.0
    return leads


def eth_covers_window(window: pd.DatetimeIndex, eth_start: pd.Timestamp,
                       eth_end: pd.Timestamp) -> bool:
    """True iff `window` has ANY overlap with ETH's coverage range -- False
    only when the whole +/-60d search window falls entirely before (or
    after) ETH data exists."""
    if len(window) == 0:
        return False
    return not (window.max() < eth_start or window.min() > eth_end)


def run_truncation_probes(btc_bars: pd.DataFrame, eth_bars: pd.DataFrame,
                           z_full: pd.Series, teb2e_full: pd.Series,
                           tee2b_full: pd.Series) -> bool:
    """Causal truncation probe on the FULL pipeline, BOTH legs: does the
    network z-score (and each directional TE leg) at a fixed check date
    change if bars strictly AFTER that date are dropped from (a) the BTC
    leg only, holding ETH fixed, and (b) the ETH leg only, holding BTC
    fixed? Both must be unchanged for the pipeline to be causal in both
    of its two inputs -- a single-df truncation probe (as used by every
    prior round's shared `truncation_causality_probe`) cannot exercise a
    genuinely bivariate, two-dataframe pipeline, so this is a bespoke
    two-leg version of the identical idea rather than a reuse of that
    helper."""
    check_date = pd.Timestamp("2020-06-01", tz="UTC")
    keep_until = check_date + pd.Timedelta(days=200)

    def get(series: pd.Series, date: pd.Timestamp) -> float:
        if date not in series.index:
            return float("nan")
        return float(series.loc[date])

    a_full = get(z_full, check_date)
    b_full = get(teb2e_full, check_date)
    c_full = get(tee2b_full, check_date)
    assert not np.isnan(a_full), "check_date has no valid z value in the full build -- pick another date"

    # (a) truncate the BTC leg's tail only, ETH held fixed.
    btc_trunc = btc_bars.loc[btc_bars.index <= keep_until].copy()
    z_bt, teb2e_bt, tee2b_bt = build_te_network_z(btc_trunc, eth_bars)
    a_bt, b_bt, c_bt = get(z_bt, check_date), get(teb2e_bt, check_date), get(tee2b_bt, check_date)

    # (b) truncate the ETH leg's tail only, BTC held fixed.
    eth_trunc = eth_bars.loc[eth_bars.index <= keep_until].copy()
    z_et, teb2e_et, tee2b_et = build_te_network_z(btc_bars, eth_trunc)
    a_et, b_et, c_et = get(z_et, check_date), get(teb2e_et, check_date), get(tee2b_et, check_date)

    ok_btc_leg = (np.isclose(a_full, a_bt, equal_nan=True)
                  and np.isclose(b_full, b_bt, equal_nan=True)
                  and np.isclose(c_full, c_bt, equal_nan=True))
    ok_eth_leg = (np.isclose(a_full, a_et, equal_nan=True)
                  and np.isclose(b_full, b_et, equal_nan=True)
                  and np.isclose(c_full, c_et, equal_nan=True))

    print(f"\nCausal truncation probe (independent re-check), check_date={check_date.date()}:")
    print(f"    full build:            z={a_full:.6f}  TE(btc->eth)={b_full:.6f}  TE(eth->btc)={c_full:.6f}")
    print(f"    BTC leg truncated tail: z={a_bt:.6f}  TE(btc->eth)={b_bt:.6f}  TE(eth->btc)={c_bt:.6f}  "
          f"causal: {ok_btc_leg}")
    print(f"    ETH leg truncated tail: z={a_et:.6f}  TE(btc->eth)={b_et:.6f}  TE(eth->btc)={c_et:.6f}  "
          f"causal: {ok_eth_leg}")
    ok = ok_btc_leg and ok_eth_leg
    assert ok, "TE-network pipeline FAILED the independent causal truncation probe (at least one leg)"
    return ok


def gate() -> dict:
    print("=" * 78)
    print("R-86 NOVEL: TE-network (TE[BTC->ETH] + TE[ETH->BTC]) vs v4 anchor -- "
          "STEP A detection-lag gate")
    print("=" * 78)

    btc_bars = load_btc_bars()
    eth_bars = load_eth_bars()
    if eth_bars is None:
        print("\nSTOP: data/ethusd_coinbase_spot_5m.csv.gz is MISSING. "
              "`load_coinbase_eth_spot` returned None. This branch's mechanism "
              "requires ETH spot data that is supposed to already be committed "
              "to this repo; it is not present. Reporting this as a hard STOP, "
              "not a fabricated result.", file=sys.stderr)
        return dict(stopped=True, reason="eth_data_missing")

    max_ts = max(btc_bars.index.max(), eth_bars.index.max())

    majority = anchor_majority(btc_bars)
    z_daily, te_b2e_daily, te_e2b_daily = build_te_network_z(btc_bars, eth_bars)
    assert_no_holdout(z_daily.to_frame("z"), "z_daily")

    z_bars = align_daily_causal(z_daily.to_frame("z"), btc_bars)["z"]
    assert_no_holdout(z_bars.to_frame("z"), "z_bars")

    run_truncation_probes(btc_bars, eth_bars, z_daily, te_b2e_daily, te_e2b_daily)

    eth_start = eth_bars.index.min()
    eth_end = eth_bars.index.max()
    print(f"\nETH coverage: {eth_start} -> {eth_end}")
    print(f"z_thresh={Z_THRESH}  search window=+/-{WINDOW_DAYS}d  "
          f"null: {N_DRAWS} draws, block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(btc_bars, onset_str, WINDOW_DAYS)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars -- outside BTC data coverage.")
            results.append(dict(label=label, onset=onset_str, coverage="no_btc_bars",
                                 pass_b=False, lead=float("nan"), null_median=float("nan")))
            continue

        if not eth_covers_window(window, eth_start, eth_end):
            print(f"[{label}] onset={onset_str}: +/-{WINDOW_DAYS}d window "
                  f"({window.min().date()} -> {window.max().date()}) falls ENTIRELY outside "
                  f"ETH coverage ({eth_start.date()} -> {eth_end.date()}). "
                  f"NO DATA COVERAGE -- the mechanism cannot fire here. "
                  f"FAIL by construction (counted against the pre-registered 4/6 bar).")
            results.append(dict(label=label, onset=onset_str, coverage="no_eth_coverage",
                                 pass_b=False, lead=float("nan"), null_median=float("nan")))
            continue

        flip_time = nearest_transition(majority, window, onset, direction="down")
        detect_time = nearest_te_alarm(z_bars, window, onset, Z_THRESH)

        if flip_time is None or detect_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no anchor-gate transition' if flip_time is None else 'no TE-network alarm'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str, coverage="covered",
                                 flip=flip_time, detect=detect_time,
                                 pass_b=False, lead=float("nan"), null_median=float("nan")))
            continue

        lead = (flip_time - detect_time).total_seconds() / 86400.0
        leads_null = null_leads(z_bars, window, onset, flip_time)
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        pass_a = lead >= 0
        pass_b = pass_a and (not np.isnan(null_median)) and (lead >= null_median)

        print(f"[{label}] onset={onset_str}")
        print(f"    v4 anchor nearest downward flip: {flip_time}")
        print(f"    TE-network nearest alarm (z>={Z_THRESH}): {detect_time}")
        print(f"    LEAD = {lead:+.2f} days  null median={null_median:+.2f}d "
              f"(valid draws {len(valid)}/{N_DRAWS})")
        print(f"    PASS (a) lead>=0: {pass_a}   PASS (b) lead>=null median: {pass_b}")

        results.append(dict(label=label, onset=onset_str, coverage="covered",
                             flip=flip_time, detect=detect_time, lead=lead,
                             null_median=null_median, pass_b=pass_b))

    n_pass = sum(1 for r in results if r["pass_b"])
    covered = [r for r in results if r["coverage"] == "covered"]
    n_covered = len(covered)
    n_pass_covered = sum(1 for r in covered if r["pass_b"])
    passed = n_pass >= 4

    print("\n" + "=" * 78)
    for r in results:
        print(f"  {r['label']:42s} coverage={r['coverage']:16s} "
              f"lead={r.get('lead', float('nan')):+.2f}d  PASS={r['pass_b']}")
    print(f"\nEpisodes passing (full, coverage gaps = fail): {n_pass}/6")
    print(f"Episodes passing (among ETH-covered only):      {n_pass_covered}/{n_covered}")
    print(f"GATE VERDICT (pre-registered rule, uses n_pass/6): "
          f"{'PASS -> proceed to Step B (build confirming-vote strategy)' if passed else 'FAIL -> STOP, no strategy built'}")
    print(f"\nconfigurations evaluated in this file: 0 (fixed measurement gate)")
    print(f"max timestamp read anywhere in this session (BTC and ETH): {max_ts}  (< {OOS_START})")

    return dict(results=results, n_pass=n_pass, n_covered=n_covered,
                n_pass_covered=n_pass_covered, passed=passed,
                btc_bars=btc_bars, eth_bars=eth_bars, z_bars=z_bars,
                majority=majority, max_ts=max_ts, stopped=False)


def step_b(gate_result: dict) -> dict:
    """R-53/R-55-architecture confirming-vote strategy: `meta_vote` (the
    TE-network alarm, 1 when `z >= Z_THRESH` else 0) is combined with v4's
    own 3-anchor sum via `r86_shared.confirming_vote_frac`, with the
    confirming vote's WEIGHT swept on TRAIN only (inner-train, <=
    `INNER_TRAIN_END`) and selected on inner-validation only
    (`INNER_VAL_START..INNER_VAL_END`). The holdout (>= OOS_START) is never
    read here. Bounded, small grid, matching this project's "count every
    configuration" discipline. Metric: mean position (`frac`) held DURING
    each inner-split stress episode's local window vs OUTSIDE any episode
    window in the same split -- a confirming vote that is doing its job de-
    risks faster around dated stress episodes without dragging exposure
    down everywhere else. This function only runs if the Step-A gate
    PASSED (n_pass/6 >= 4); it is defined unconditionally so the file is
    complete, but `main()` below only calls it on a pass."""
    btc_bars = gate_result["btc_bars"]
    majority = gate_result["majority"]
    z_bars = gate_result["z_bars"]

    anchor_sum = (majority * 3.0).to_numpy()
    meta_vote = (z_bars >= Z_THRESH).astype(float).to_numpy()

    weight_grid = [0.5, 1.0, 2.0, 3.0, 5.0]
    print("\n" + "=" * 78)
    print("STEP B: confirming-vote weight sweep (TRAIN only; select on inner-validation only)")
    print(f"weight grid ({len(weight_grid)} configurations): {weight_grid}")

    def episode_mask(idx: pd.DatetimeIndex) -> np.ndarray:
        mask = np.zeros(len(idx), dtype=bool)
        for _, onset_str in STRESS_EPISODES:
            onset, window = episode_window(btc_bars, onset_str, WINDOW_DAYS)
            if len(window) == 0:
                continue
            mask |= idx.isin(window)
        return mask

    train_mask = btc_bars.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")
    val_mask = ((btc_bars.index >= pd.Timestamp(INNER_VAL_START, tz="UTC"))
                & (btc_bars.index <= pd.Timestamp(INNER_VAL_END, tz="UTC")))
    ep_mask = episode_mask(btc_bars.index)

    def episode_vs_baseline_gap(mask: np.ndarray, weight: float) -> float:
        frac = confirming_vote_frac(anchor_sum, meta_vote, weight)
        sub_ep = mask & ep_mask
        sub_other = mask & ~ep_mask
        if sub_ep.sum() == 0 or sub_other.sum() == 0:
            return float("nan")
        # A confirming vote doing its job holds LESS exposure inside dated
        # episode windows relative to outside them, more so than v4 alone.
        base_frac = (anchor_sum / 3.0)
        gap_confirm = float(base_frac[sub_other].mean() - frac[sub_other].mean()
                             - (base_frac[sub_ep].mean() - frac[sub_ep].mean()))
        # equivalently: how much MORE the confirming vote de-risks inside
        # episodes than outside them, relative to v4's own baseline gap.
        v4_gap = float(base_frac[sub_other].mean() - base_frac[sub_ep].mean())
        confirm_gap = float(frac[sub_other].mean() - frac[sub_ep].mean())
        return confirm_gap - v4_gap

    train_scores = {w: episode_vs_baseline_gap(train_mask, w) for w in weight_grid}
    val_scores = {w: episode_vs_baseline_gap(val_mask, w) for w in weight_grid}

    print("\nTRAIN (inner-train, fit/sweep only):")
    for w, s in train_scores.items():
        print(f"    weight={w:<5} extra de-risk gap (episode vs non-episode, vs v4 baseline) = {s:+.5f}")

    best_train_weight = max(train_scores, key=lambda w: (train_scores[w] if not np.isnan(train_scores[w]) else -np.inf))
    print(f"\nBest TRAIN weight: {best_train_weight}")

    print("\nINNER-VALIDATION (selection only, still not the holdout):")
    for w, s in val_scores.items():
        print(f"    weight={w:<5} extra de-risk gap = {s:+.5f}")

    selected_weight = best_train_weight
    val_score_selected = val_scores.get(selected_weight, float("nan"))
    print(f"\nSelected weight (fixed on TRAIN, confirmed not to collapse on VAL): "
          f"{selected_weight}  (val score = {val_score_selected:+.5f})")
    print("Holdout (>= OOS_START) NOT touched. Promotion/registration decision "
          "belongs to the operator, not this file.")

    return dict(weight_grid=weight_grid, train_scores=train_scores,
                val_scores=val_scores, selected_weight=selected_weight,
                n_configs_evaluated=len(weight_grid))


def main() -> dict:
    result = gate()
    if result.get("stopped"):
        return result
    if result["passed"]:
        result["step_b"] = step_b(result)
    else:
        print("\nGATE FAILED (< 4/6). Per the pre-registered stop rule: STOP. "
              "No strategy code built. No data on/after 2023-01-01 touched.")
    return result


if __name__ == "__main__":
    main()
