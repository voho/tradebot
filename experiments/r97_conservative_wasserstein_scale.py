"""R-97 CONSERVATIVE branch: independent Step-0 gate re-derivation.

Round R-97's idea (see `experiments/r97_shared.py` for the full literature
grounding and duplicate-check, written by the operator, read but never
imported into the gate computation below): replace `kelly_regime_v4`'s
`scale = min(target_vol/vol, max_leverage)` factor outright with a
Wasserstein-distributionally-robust Kelly fraction, shrunk by a DRO
discount set by the causal count of completed regime cycles observed so
far --

    scale_robust = scale * discount(N)

where `discount` in (0, 1] comes from the Mohajerin Esfahani & Kuhn (2018,
Math. Program. 171(1-2), 115-166) finite-sample Wasserstein-ball radius
around the empirical regime-conditional return distribution, and the
robust-Kelly-under-a-ball reformulation is Li (2023, arXiv:2302.13979,
"Wasserstein-Kelly Portfolios"). This is the literal, closest-to-the-paper
construction: a full replacement of v4's existing scale factor with the
DRO-discounted one -- hence "conservative" relative to any softer blend.

THIS FILE's job, per docs/ROUTINE.md step 2 and the round instructions, is
to INDEPENDENTLY re-derive and verify the two pre-registered Step-0 kill
switches that `r97_shared.step0_gate` computed, from scratch, without
calling `r97_shared.step0_gate` -- so a bug shared between "the gate" and
"the check on the gate" cannot hide. Independence taken here:

- The regime-vote reconstruction below is written fresh from
  `src/tradebot/strategies/kelly_regime.py` /
  `kelly_regime_v3.py` / `kelly_regime_v4.py` (confirmed bar-for-bar:
  v4 = v3 with horizons=(20,40,80), band=0.01 inherited from the
  `KellyRegime` base `__init__` default -- v3 does not touch `band`),
  not copy-pasted from `r97_shared.anchor_votes`.
- The cycle count is cross-checked TWO ways in this file: a vectorized
  numpy pass (matching the mechanism described in the docstring of
  `r97_shared.regime_cycle_count`) and an independent O(n) Python loop
  over lean-sign flips, asserted equal at every one of the six episode
  probe points before the gate numbers are trusted.
- The DRO radius/discount formula itself (`eps(N) = kappa*sqrt(log(1/beta)/N)`,
  `discount(N) = 1/(1+eps(N)/eps(N_ref))`) is a PARAMETER CHOICE fixed a
  priori by the round (BETA_CONF=0.10, KAPPA=1.0, N_REF=3, all pre-registered
  in `r97_shared.py`'s docstring before any real-data number was computed)
  -- there is nothing to "independently derive" about an arbitrary formula
  choice, so this file reuses the identical formula, but recomputes it from
  first principles (not by importing `r97_shared.wasserstein_radius` /
  `dro_discount`) and cross-checks the closed form against a direct
  numeric evaluation.

RESULT: independent numbers MATCH the operator's exactly (see run log
appended after the first execution, and the report given to the caller).
N(episode) = [30, 48, 68, 97, 103, 109], discount(episode) = [0.7597,
0.8000, 0.8264, 0.8504, 0.8542, 0.8577].

  Kill switch A (spread, >=4 distinct N across 6 episodes): PASS (6 distinct).
  Kill switch B (magnitude, max/min discount ratio >=1.3x): FAIL (ratio ~=1.129).

Per the pre-registered rule in `r97_shared.py`'s docstring ("Both switches
... checked BEFORE either branch writes a single line of strategy or
backtest code" / kill switch B: "there is no reason to build one"), this
branch STOPS HERE. No strategy code, no `scripts/experiment.py` call, no
backtest, zero configurations evaluated. The `scale_robust = scale *
discount(N)` construction described above is recorded but was never
implemented past this gate script -- that is the correct, honest outcome
for a round whose own pre-registered falsification test fired at Step 0,
matching R-91's "A0 kill switch" and R-94/R-95's precedent of stopping at
a self-imposed measurement gate before any backtest code ran.

Data discipline: this file reads ONLY `df.loc[:INNER_VAL_END]`
(2022-12-31 and earlier) via `tradebot.data.load_dataset`; it never
constructs a timestamp on or after `OOS_START = 2023-01-01`, enforced by
an explicit assertion before the gate is computed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --------------------------------------------------------------- constants
# kelly_regime_v4's actual, verified-by-reading-the-source parameters:
#   src/tradebot/strategies/kelly_regime_v4.py:  horizons=(20, 40, 80)
#   src/tradebot/strategies/kelly_regime.py:     band=0.01 (base __init__
#     default; kelly_regime_v3 does not override it, kelly_regime_v4 does
#     not override it either -- confirmed by reading both files in full).
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01
BARS_PER_DAY = 288

INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# DRO formula parameters, pre-registered fixed constants (r97_shared.py
# docstring), reused verbatim as a PARAMETER CHOICE -- not re-derived,
# since there is nothing data-dependent about them.
BETA_CONF = 0.10
KAPPA = 1.0
N_REF = 3.0

# Same six dated BTC stress episodes R-82 through R-96 all use, copied
# from r97_shared.py's STRESS_EPISODES (a fixed input table, not a
# computed quantity -- reusing it is not the thing being independently
# verified; the N(t) and discount(t) *computations* are).
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]


# ------------------------------------------------------- independent vote
def compute_vote_fraction(df: pd.DataFrame) -> np.ndarray:
    """Independent reconstruction of v4's 3-anchor majority vote.

    Bar-for-bar equivalent to `KellyRegime.prepare`'s vote block: rolling
    mean anchor at each horizon (in days -> bars), latched bullish (1.0)
    above anchor*(1+band), latched bearish (0.0) below anchor*(1-band),
    holding the previous verdict inside the band via forward-fill, with
    an initial fill of 0.0 (bearish) before the first anchor value exists
    -- exactly what `.ffill().fillna(0.0)` does in the strategy source.
    """
    close = df["close"]
    per_horizon = []
    for days in V4_HORIZONS:
        window = days * BARS_PER_DAY
        anchor = close.rolling(window).mean()
        upper = anchor * (1.0 + V4_BAND)
        lower = anchor * (1.0 - V4_BAND)
        raw = pd.Series(np.nan, index=df.index)
        raw[close > upper] = 1.0
        raw[close < lower] = 0.0
        latched = raw.ffill().fillna(0.0)
        per_horizon.append(latched.to_numpy())
    frac = np.mean(np.vstack(per_horizon), axis=0)
    return frac


def cycle_count_vectorized(frac: np.ndarray) -> np.ndarray:
    """Vectorized causal completed-cycle count from the vote fraction.

    `frac` in {0, 1/3, 2/3, 1} (never exactly 0.5), so lean = sign(frac -
    0.5) in {-1, +1} throughout. A regime "transition" is a lean sign
    flip; a completed "cycle" is every second flip (there-and-back).
    """
    lean = np.where(frac > 0.5, 1, -1)
    flips = np.zeros(len(lean), dtype=np.int64)
    flips[1:] = (lean[1:] != lean[:-1]).astype(np.int64)
    cum_flips = np.cumsum(flips)
    cycles = cum_flips // 2
    return cycles.astype(float)


def cycle_count_loop(frac: np.ndarray, probe_idx: list[int]) -> dict[int, float]:
    """Independent O(n) loop reconstruction, evaluated only at the probe
    indices (the bars immediately preceding each episode onset) for
    tractability -- cross-checks the vectorized path with a structurally
    different implementation (running flip counter, not cumsum/diff).
    """
    probe_set = set(probe_idx)
    out: dict[int, float] = {}
    lean_prev = None
    n_flips = 0
    for i, f in enumerate(frac):
        lean = 1 if f > 0.5 else -1
        if lean_prev is not None and lean != lean_prev:
            n_flips += 1
        lean_prev = lean
        if i in probe_set:
            out[i] = float(n_flips // 2)
    return out


# --------------------------------------------------------------- DRO math
def wasserstein_radius(n: float, kappa: float = KAPPA, beta: float = BETA_CONF) -> float:
    """eps(N) = kappa * sqrt(log(1/beta) / N); eps(0) = +inf (no completed
    cycle observed yet -> maximal distrust of the empirical distribution).
    """
    if n <= 0:
        return float("inf")
    return kappa * np.sqrt(np.log(1.0 / beta) / n)


def dro_discount(n: float, n_ref: float = N_REF) -> float:
    """discount(N) = 1 / (1 + eps(N)/eps(N_ref)), bounded in (0, 1],
    discount(N_ref) = 0.5 by construction, discount -> 1 as N -> inf,
    discount -> 0 as N -> 0.

    Closed-form simplification cross-checked below (in main()) against
    plugging the sqrt expressions in directly: eps(N)/eps(N_ref) =
    sqrt(n_ref/N) when N>0, so discount(N) = 1/(1+sqrt(n_ref/N)) -- a
    second, independent path to the same number that does not go through
    `wasserstein_radius` at all.
    """
    eps_n = wasserstein_radius(n)
    eps_ref = wasserstein_radius(n_ref)
    if np.isinf(eps_n):
        return 0.0
    return 1.0 / (1.0 + eps_n / eps_ref)


def dro_discount_closed_form(n: float, n_ref: float = N_REF) -> float:
    """discount(N) = 1/(1+sqrt(n_ref/N)) for N>0, 0 for N<=0 -- algebraic
    simplification of dro_discount that never calls wasserstein_radius.
    """
    if n <= 0:
        return 0.0
    return 1.0 / (1.0 + np.sqrt(n_ref / n))


# ------------------------------------------------------------------- main
def main() -> None:
    from tradebot.data import load_dataset

    df_full, label = load_dataset(ROOT / "data", "spot")

    # --- holdout guard: restrict BEFORE any computation, then assert. ---
    df = df_full.loc[:INNER_VAL_END]
    cutoff = pd.Timestamp(OOS_START, tz="UTC")
    assert len(df) > 0 and pd.Timestamp(df.index.max()) < cutoff, (
        f"holdout leak: max timestamp read = {df.index.max()}, must be < {OOS_START}")

    print(f"data: {label}, inner (<= {INNER_VAL_END}) bars: {len(df):,}")
    print(f"max timestamp read: {df.index.max()}  (< {OOS_START}: "
          f"{pd.Timestamp(df.index.max()) < cutoff})\n")

    frac = compute_vote_fraction(df)
    cycles_vec = cycle_count_vectorized(frac)

    # Probe indices: last bar strictly before each episode onset.
    probe_idx = []
    probe_labels = []
    for name, onset_str in STRESS_EPISODES:
        onset = pd.Timestamp(onset_str, tz="UTC")
        pos = df.index.searchsorted(onset, side="left") - 1
        if pos < 0:
            probe_idx.append(None)
        else:
            probe_idx.append(int(pos))
        probe_labels.append((name, onset_str))

    valid_positions = [p for p in probe_idx if p is not None]
    cycles_loop = cycle_count_loop(frac, valid_positions)

    print("Cross-check: vectorized vs independent-loop cycle count at each probe:")
    for (name, onset_str), pos in zip(probe_labels, probe_idx):
        if pos is None:
            print(f"  {name:42s} onset {onset_str}: no pre-onset data")
            continue
        v_vec = cycles_vec[pos]
        v_loop = cycles_loop[pos]
        match = "OK" if v_vec == v_loop else "MISMATCH"
        print(f"  {name:42s} onset {onset_str}: vectorized N={v_vec:.0f}  "
              f"loop N={v_loop:.0f}  [{match}]")
        assert v_vec == v_loop, "vectorized and loop cycle counts disagree -- bug"

    print()

    rows = []
    for (name, onset_str), pos in zip(probe_labels, probe_idx):
        if pos is None:
            rows.append((name, onset_str, None, None))
            continue
        n = float(cycles_vec[pos])
        ts = df.index[pos]
        d_a = dro_discount(n)
        d_b = dro_discount_closed_form(n)
        assert np.isclose(d_a, d_b, rtol=1e-12), (
            f"discount formula mismatch at N={n}: {d_a} vs {d_b}")
        rows.append((name, onset_str, n, d_a, ts))

    print("episode                                    onset        pre-onset-ts               N     discount")
    for r in rows:
        if r[2] is None:
            name, onset_str = r[0], r[1]
            print(f"{name:42s} {onset_str}  (no pre-onset data)")
        else:
            name, onset_str, n, d, ts = r
            print(f"{name:42s} {onset_str}  {str(ts):26s} N={n:5.0f}  discount={d:.4f}")

    valid = [(r[2], r[3]) for r in rows if r[2] is not None]
    n_values_sorted = sorted({n for n, _ in valid})
    d_values = [d for _, d in valid]

    spread_pass = len(n_values_sorted) >= 4
    ratio = (max(d_values) / min(d_values)) if d_values and min(d_values) > 0 else float("inf")
    magnitude_pass = ratio >= 1.3

    print(f"\ndistinct N values across episodes: {n_values_sorted}  "
          f"({len(n_values_sorted)} distinct)")
    print(f"KILL SWITCH A (spread, need >=4 distinct N): "
          f"{'PASS' if spread_pass else 'FAIL'}")
    print(f"discount values: {[round(d, 4) for d in d_values]}")
    print(f"discount ratio max/min: {ratio:.4f}")
    print(f"KILL SWITCH B (magnitude, need ratio >=1.3x): "
          f"{'PASS' if magnitude_pass else 'FAIL'}")

    overall = spread_pass and magnitude_pass
    print(f"\nSTEP-0 GATE (independent re-derivation): "
          f"{'PASS -- would proceed to build/backtest' if overall else 'FAIL -- STOP here, no strategy/backtest code'}")

    # Comparison against the operator's reported numbers (r97_shared.py
    # run output quoted in the task), stated explicitly rather than
    # silently trusted.
    operator_n = [30, 48, 68, 97, 103, 109]
    operator_d = [0.7597, 0.8000, 0.8264, 0.8504, 0.8542, 0.8577]
    my_n = [r[2] for r in rows if r[2] is not None]
    my_d = [round(r[3], 4) for r in rows if r[2] is not None]
    print(f"\noperator N:      {operator_n}")
    print(f"independent N:   {[int(x) for x in my_n]}")
    print(f"operator disc:   {operator_d}")
    print(f"independent disc:{my_d}")
    n_match = [int(x) for x in my_n] == operator_n
    d_match = all(abs(a - b) < 1e-3 for a, b in zip(my_d, operator_d))
    print(f"N values match operator: {n_match}")
    print(f"discount values match operator (tol 1e-3): {d_match}")

    if overall:
        print("\n*** GATE PASSED -- this contradicts the operator's reported FAIL. ***")
        print("Do not proceed further without first resolving this discrepancy.")
    else:
        print("\nGate result confirms operator: FAIL on kill switch B (magnitude).")
        print("Per the pre-registered rule, this branch stops here. No strategy")
        print("code, no scripts/experiment.py call, no backtest was run. Zero")
        print("configurations evaluated -- the correct, honest outcome for a")
        print("round whose Step-0 falsification test fired before Step 3.")


if __name__ == "__main__":
    main()
