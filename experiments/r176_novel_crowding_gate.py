"""R-176 NOVEL branch: dollar-bar arrival-rate "crowding" gate.

Frozen pre-registration: `experiments/r176_direction.md` (Novel variant,
Step 1 Q4(b)). Shared, read-only engine: `experiments/r176_shared.py`
(imported only, never edited).

Mechanism, one sentence (verbatim from the pre-registration): multiply
`kelly_regime_v4`'s own unmodified `frac * scale` by a latching crowding
haircut (0.5 while dollar-bar arrival intensity sits above its own causal
90th-percentile trailing threshold, 1.0 otherwise, releasing below the 60th
percentile), leaving the vote and scale byte-identical to `kelly_regime_v4`
today. `crowding_target()` in r176_shared.py is this candidate's complete,
already-implemented, causally-verified target-exposure path at the default
(frozen) parameters; this file (a) runs the pre-registered two-part
mechanism/falsification check FIRST, decisive on its own regardless of any
performance number, and (b) only if not falsified, sweeps the two
pre-registered free parameters (`CROWDING_HAIRCUT`,
`(high_in_q, high_out_q)`) on inner-train/inner-validation.

Configs evaluated by this file: reported explicitly at the bottom of
execution (mechanism check is itself 1 evaluated configuration; the sweep,
if reached, adds up to 12 more, per the pre-registration's own count).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r176_shared import (  # noqa: E402
    BARS_PER_DAY,
    CROWDING_HAIRCUT,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    INTENSITY_HIGH_IN_Q,
    INTENSITY_HIGH_OUT_Q,
    apply_deadband,
    causal_truncation_probe_series,
    compare,
    crowding_haircut,
    crowding_target,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    v4_raw_desired,
    v4_symmetric_vol,
)
# Additional r102_shared constants r176_shared.py does not itself re-export,
# needed only to duplicate (never to alter) v4's own internal breakout-state
# machine for the mechanism check below. r176_shared.py DOES re-export
# `conditional_target_scale` and `v4_symmetric_vol` themselves; those are
# used here only to cross-check the local duplicate reproduces them exactly
# (see `_verify_breakout_state_reconstruction` in the self-test block).
from experiments.r102_shared import (  # noqa: E402
    V4_ANCHOR_SPAN_DAYS,
    V4_HIGH_IN,
    V4_HIGH_OUT,
    V4_LOW_IN,
    V4_LOW_OUT,
    V4_MAX_LEVERAGE,
    V4_TARGET_VOL,
    conditional_target_scale,
)

CONFIGS_EVALUATED = 0  # incremented as each real-data configuration is run


# ======================================================================
# v4's own volatility-breakout state indicator, reconstructed locally.
#
# conditional_target_scale() (r102_shared.py, re-exported by r176_shared.py)
# returns only the BLENDED scale value (`full[i] if state!=0 else steady[i]`)
# -- it does not expose the latching state itself. This duplicates that
# state machine's LOOP verbatim (same thresholds, same latch semantics) to
# recover `state != 0` as a boolean array, which is exactly
# `out[i] != steady[i]` in the pre-registration's own terms. It is a
# reconstruction, not an alteration: `_verify_breakout_state_reconstruction`
# below re-derives `full`/`steady` independently and asserts
# `conditional_target_scale(vol)` equals
# `np.where(breakout_state, full, steady)` bar-for-bar before this is
# trusted for the mechanism check.
# ======================================================================

def v4_breakout_state(df: pd.DataFrame) -> np.ndarray:
    """Boolean array: True where v4's own conditional-vol-target scale is in
    its 'breakout' (full inverse-vol) state rather than its steady
    (anchor-relative) state, using v4's own OWN unmodified volatility input
    (`v4_symmetric_vol`) and OWN unmodified thresholds/anchor span."""
    vol = np.asarray(v4_symmetric_vol(df), dtype=float)
    slow = (pd.Series(vol).ewm(span=V4_ANCHOR_SPAN_DAYS * BARS_PER_DAY,
                                min_periods=BARS_PER_DAY).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)

    n = len(vol)
    state = np.zeros(n, dtype=int)
    s = 0
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if s == 0:
                s = 1 if x > V4_HIGH_IN else (-1 if x < V4_LOW_IN else 0)
            elif s == 1 and x < V4_HIGH_OUT:
                s = 0
            elif s == -1 and x > V4_LOW_OUT:
                s = 0
        state[i] = s
    return state != 0


def _verify_breakout_state_reconstruction(df: pd.DataFrame) -> None:
    """Proves `v4_breakout_state` above reproduces `conditional_target_scale`'s
    real internal state exactly, by independently rebuilding `full`/`steady`
    and checking `conditional_target_scale(vol) == where(state!=0, full, steady)`
    bar-for-bar on real data."""
    vol = np.asarray(v4_symmetric_vol(df), dtype=float)
    slow = (pd.Series(vol).ewm(span=V4_ANCHOR_SPAN_DAYS * BARS_PER_DAY,
                                min_periods=BARS_PER_DAY).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        full = np.minimum(V4_TARGET_VOL / vol, V4_MAX_LEVERAGE)
        steady = np.minimum(V4_TARGET_VOL / slow, V4_MAX_LEVERAGE)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    breakout = v4_breakout_state(df)
    reconstructed = np.where(breakout, full, steady)
    real = conditional_target_scale(vol)
    if not np.allclose(reconstructed, real, atol=1e-10, rtol=1e-9):
        bad = int(np.sum(~np.isclose(reconstructed, real, atol=1e-10, rtol=1e-9)))
        raise AssertionError(
            f"v4_breakout_state reconstruction mismatch: {bad}/{len(real)} bars differ "
            "from conditional_target_scale's real output")


# ======================================================================
# Step 1 Q4(b) — two-part mechanism/falsification check. Decisive on its
# own, run BEFORE any performance number, per the frozen pre-registration.
# ======================================================================

def mechanism_check(btc: pd.DataFrame) -> dict:
    _verify_breakout_state_reconstruction(btc)

    haircut, crowded = crowding_haircut(btc)  # frozen defaults: 0.90/0.60/0.5
    breakout = v4_breakout_state(btc)

    crowded_s = pd.Series(crowded, index=btc.index)
    breakout_s = pd.Series(breakout, index=btc.index)
    it_crowded = crowded_s.loc[INNER_TRAIN_START:INNER_TRAIN_END].to_numpy().astype(float)
    it_breakout = breakout_s.loc[INNER_TRAIN_START:INNER_TRAIN_END].to_numpy().astype(float)

    n = len(it_crowded)
    frac_crowded = float(np.mean(it_crowded))
    r2_cb = r_squared(it_crowded, it_breakout)   # crowded regressed on breakout's mean
    r2_bc = r_squared(it_breakout, it_crowded)   # breakout regressed on crowded's mean

    pass_i = 0.02 < frac_crowded < 0.40
    # report the WORSE (higher) of the two R^2 orderings against the 0.5 bar,
    # so the pass/fail cannot depend on an arbitrary a/b choice
    r2_worst = max(v for v in (r2_cb, r2_bc) if np.isfinite(v))
    pass_ii = r2_worst < 0.5

    falsified = not (pass_i and pass_ii)
    return dict(n=n, frac_crowded=frac_crowded, r2_cb=r2_cb, r2_bc=r2_bc,
                r2_worst=r2_worst, pass_i=pass_i, pass_ii=pass_ii,
                falsified=falsified)


# ======================================================================
# Step 3 — training-only sweep of the two pre-registered free parameters.
# ======================================================================

def make_crowding_target(haircut: float, high_in_q: float, high_out_q: float):
    """Thin wrapper around `crowding_haircut`/`v4_raw_desired` (both imported,
    unedited, from r176_shared.py), parameterized by the two swept knobs.
    Duplicates only the 3-line composition `crowding_target()` itself does,
    not any of the underlying primitives."""

    def build(df: pd.DataFrame) -> np.ndarray:
        hc, _ = crowding_haircut(df, high_in_q=high_in_q, high_out_q=high_out_q,
                                  haircut=haircut)
        return apply_deadband(v4_raw_desired(df) * hc)

    build.__name__ = f"crowd_h{haircut:.2f}_qi{high_in_q:.2f}_qo{high_out_q:.2f}"
    return build


HAIRCUT_GRID = (0.25, 0.4, 0.5, 0.7)
QUANTILE_GRID = ((0.85, 0.55), (0.90, 0.60), (0.95, 0.70))


def run_sweep(btc: pd.DataFrame) -> list[dict]:
    global CONFIGS_EVALUATED
    all_rows: list[dict] = []
    n_configs = len(HAIRCUT_GRID) * len(QUANTILE_GRID)
    print(f"\n=== Sweep: {n_configs} configurations "
          f"({len(HAIRCUT_GRID)} haircuts x {len(QUANTILE_GRID)} quantile pairs), "
          "inner-train + inner-val, BTC only (no pruning) ===")
    for haircut in HAIRCUT_GRID:
        for (qi, qo) in QUANTILE_GRID:
            label = f"h{haircut}_qi{qi}_qo{qo}"
            build = make_crowding_target(haircut, qi, qo)
            rows = compare(build, label=label, btc=btc, include_eth=False)
            CONFIGS_EVALUATED += 1
            all_rows.extend(rows)
            print_rows(rows)
    return all_rows


def summarize_by_config(rows: list[dict]) -> list[dict]:
    """One row per (haircut, qi, qo) config: mean d_sharpe across inner_train
    + inner_val, both markets, plus the inner_val-only figure used for
    selection (per Step 3: select on inner-validation)."""
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    out = []
    for label, rs in by_label.items():
        val_rows = [r for r in rs if r["slice"] == "inner_val"]
        train_rows = [r for r in rs if r["slice"] == "inner_train"]
        out.append({
            "label": label,
            "val_d_sharpe_mean": float(np.mean([r["d_sharpe"] for r in val_rows])),
            "val_d_logg_mean": float(np.mean([r["d_log_growth"] for r in val_rows])),
            "val_excl0_any": any(r["excludes_zero"] for r in val_rows),
            "val_excl0_positive": any(r["excludes_zero"] and r["boot_lo"] > 0 for r in val_rows),
            "train_d_sharpe_mean": float(np.mean([r["d_sharpe"] for r in train_rows])),
            "train_dd_mean": float(np.mean([r["d_dd"] for r in train_rows])),
            "val_dd_mean": float(np.mean([r["d_dd"] for r in val_rows])),
        })
    out.sort(key=lambda d: d["val_d_sharpe_mean"], reverse=True)
    return out


def main() -> None:
    global CONFIGS_EVALUATED
    print("=" * 100)
    print("R-176 NOVEL BRANCH: dollar-bar arrival-rate crowding gate")
    print("=" * 100)

    btc = load_btc()

    # ------------------------------------------------------------------
    # Step 1 Q4(b): mechanism check FIRST, decisive independent of any
    # performance number.
    # ------------------------------------------------------------------
    mc = mechanism_check(btc)
    CONFIGS_EVALUATED += 1  # the mechanism check's own (default-parameter) configuration
    print("\n--- Mechanism / falsification check (BTC inner-train, frozen "
          "default parameters: high_in_q=0.90, high_out_q=0.60, haircut=0.5) ---")
    print(f"n bars evaluated (inner-train): {mc['n']}")
    print(f"(i)  fraction of bars 'crowded': {mc['frac_crowded']:.4f} "
          f"({'PASS' if mc['pass_i'] else 'FAIL'}: needs 0.02 < x < 0.40)")
    print(f"(ii) R^2(crowded, breakout)  = {mc['r2_cb']:+.4f}")
    print(f"     R^2(breakout, crowded)  = {mc['r2_bc']:+.4f}")
    print(f"     worst-case R^2 used for the gate = {mc['r2_worst']:+.4f} "
          f"({'PASS' if mc['pass_ii'] else 'FAIL'}: needs < 0.5)")
    verdict = "FALSIFIED" if mc["falsified"] else "NOT FALSIFIED"
    print(f"\n>>> Mechanism check verdict: {verdict} <<<")

    if mc["falsified"]:
        print("\nBranch is FALSIFIED on the mechanism check alone, per the frozen "
              "pre-registration (Step 1 Q4(b)). Stopping before any performance "
              "comparison, as pre-registered.")
        print(f"\nTotal configurations evaluated by this file: {CONFIGS_EVALUATED} "
              "(the mechanism check itself, at the frozen default parameters).")
        return

    print("\nMechanism check passed both parts. Proceeding to the pre-registered "
          "training-only sweep.")

    # ------------------------------------------------------------------
    # Step 3: sweep on inner-train / inner-validation only.
    # ------------------------------------------------------------------
    sweep_rows = run_sweep(btc)
    summary = summarize_by_config(sweep_rows)

    print("\n=== Per-config summary (sorted by inner-val mean d_sharpe, best first) ===")
    print(f"{'label':30s} {'val_dSh':>9s} {'val_dlogG':>10s} {'val_excl0+':>10s} "
          f"{'train_dSh':>10s} {'train_dDD':>10s} {'val_dDD':>9s}")
    for s in summary:
        print(f"{s['label']:30s} {s['val_d_sharpe_mean']:+9.3f} "
              f"{s['val_d_logg_mean']:+10.3f} {str(s['val_excl0_positive']):>10s} "
              f"{s['train_d_sharpe_mean']:+10.3f} {s['train_dd_mean']:+10.2f} "
              f"{s['val_dd_mean']:+9.2f}")

    winner = summary[0]
    print(f"\n>>> Winning configuration (by inner-val mean d_sharpe): {winner['label']} <<<")

    # Noise-floor / plateau diagnostics.
    print("\n--- Promotion-adjacent diagnostics (inner-validation) ---")
    print(f"Winner val d_sharpe mean: {winner['val_d_sharpe_mean']:+.3f} "
          f"(needs to clear +/-0.20 noise floor, R-20)")
    clears_noise_floor = winner["val_d_sharpe_mean"] > 0.20
    print(f"Clears +0.20 noise floor: {clears_noise_floor}")
    print(f"Winner val excludes-zero on the WINNING side (any market): "
          f"{winner['val_excl0_positive']}")
    print(f"Winner val drawdown delta (mean, cand - ctrl): {winner['val_dd_mean']:+.2f} pp")

    others = [s["val_d_sharpe_mean"] for s in summary[1:]]
    if others:
        spread = winner["val_d_sharpe_mean"] - max(others)
        print(f"\nPlateau check: winner vs. runner-up val d_sharpe gap = {spread:+.3f} "
              f"(runner-up = {max(others):+.3f}). All {len(summary)} configs' val d_sharpe: "
              f"{[round(s['val_d_sharpe_mean'], 3) for s in summary]}")

    # ------------------------------------------------------------------
    # Winning config: ETH replication + risk-matching diagnostics.
    # ------------------------------------------------------------------
    parts = winner["label"].replace("h", "").replace("qi", "").replace("qo", "").split("_")
    win_haircut, win_qi, win_qo = (float(p) for p in parts)
    print(f"\n--- Winning config re-run WITH include_eth=True: "
          f"haircut={win_haircut}, high_in_q={win_qi}, high_out_q={win_qo} ---")
    eth = load_eth()
    win_build = make_crowding_target(win_haircut, win_qi, win_qo)
    win_rows_eth = compare(win_build, label="winner_eth", btc=btc, eth=eth, include_eth=True)
    CONFIGS_EVALUATED += 1  # the winner's ETH-inclusive re-run
    print_rows(win_rows_eth)
    print("\nRisk-matching diagnostics for the winner (all slices/markets):")
    for r in win_rows_eth:
        print(f"  {r['slice']:16s} {r['market']:12s} exposure_ratio={r['exposure_ratio']:.3f} "
              f"vol_ratio={r['vol_ratio']:.3f} risk_matched={r['risk_matched']}")

    print(f"\nTotal configurations evaluated by this file: {CONFIGS_EVALUATED} "
          "(1 mechanism check + sweep + 1 winner ETH-inclusive re-run).")


# ======================================================================
# Self-test / sanity block (module-level, mirrors r176_shared.py's own
# `_self_test()` pattern): confirms causality via
# `causal_truncation_probe_series` for at least 2 swept configurations.
# ======================================================================

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=400_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(1760)
    sigma = np.full(len(idx), 0.0006)
    # a genuine, temporary volatility regime change (not just a volume burst),
    # so v4's OWN breakout-state machine is non-degenerate on this synthetic
    # series too -- crowding's own gate is driven by volume/dollar activity,
    # but the mechanism check needs the *volatility* breakout state to fire.
    burst_lo, burst_hi = len(idx) // 3, len(idx) // 3 + 25_000
    sigma[burst_lo:burst_hi] *= 6.0
    innov = rng.normal(0, sigma, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    volume = np.abs(rng.normal(20, 5, len(idx)))
    volume[len(idx) // 2:] *= 3.0  # a genuine activity burst, so crowding fires
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                        "close": close, "volume": volume}, index=idx)

    # (1) local breakout-state reconstruction matches conditional_target_scale
    #     exactly, on synthetic data too.
    _verify_breakout_state_reconstruction(df)

    # (2) causal truncation probes on >= 2 swept parameterizations of the
    #     candidate builder, per the mandated self-test coverage.
    swept = [
        make_crowding_target(0.25, 0.85, 0.55),
        make_crowding_target(0.5, 0.90, 0.60),
        make_crowding_target(0.7, 0.95, 0.70),
    ]
    for build in swept:
        assert causal_truncation_probe_series(build, df), build.__name__
    assert causal_truncation_probe_series(crowding_target, df)
    assert causal_truncation_probe_series(v4_breakout_state, df)

    # (3) v4_breakout_state is a genuine boolean 0/1 array with both states
    #     represented on data with a real regime change.
    bstate = v4_breakout_state(df)
    assert bstate.dtype == bool
    assert 0.0 < np.mean(bstate) < 1.0, "breakout state degenerate on synthetic burst data"

    print(f"r176_novel_crowding_gate self-test OK "
          f"(breakout-state reconstruction verified, {len(swept) + 2} causal "
          f"truncation probes passed, synthetic breakout fraction={np.mean(bstate):.3f})")


if __name__ == "__main__":
    _self_test()
    main()
else:
    _self_test()
