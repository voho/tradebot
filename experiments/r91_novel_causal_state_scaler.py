#!/usr/bin/env python
"""R-91 NOVEL branch: a CAUSAL, EXPANDING-WINDOW STATE-CONDITIONAL SCALER
applied to `kelly_regime_v4`'s vote, in the Correction/Rebound (turning-
point) states only.

MECHANISM (one sentence). Goulding, Harvey & Mazzoleni (2023)'s own
methodological contribution is not the fixed discount this round's
CONSERVATIVE sibling implements, but to ESTIMATE state-conditional
risk/return from data and blend dynamically; this branch is the direct
operationalisation of that idea: a `CausalStateScaler` (expanding-window,
strictly one-bar-lagged) tracks each state's own running Sharpe-like
statistic and squashes it into a [0, 1] multiplier applied to v4's vote
ONLY when the bar is classified Correction or Rebound (Bull/Bear bars are
untouched -- same scope restriction as the conservative branch, made in
`r91_shared` so the two branches are comparable and the identity point is
trivial).

THIS ROUND'S OWN PRE-MEASURED KILL SWITCH (A0, see `r91_shared` docstring).
Causal (here: descriptive, full-inner-train, one-shot, per the module's own
`causal_state_stats` docstring) state-conditional Sharpe-like on BTC
inner-train: Bull +0.190, Bear -0.103, Correction -0.066, Rebound +0.185.
The pre-registered rule requires Correction AND Rebound to both rank BELOW
Bull AND Bear. Rebound (+0.185) ranks ABOVE Bear (-0.103) -- **the kill
switch FIRES**. Per pre-registration this branch is NEGATIVE regardless of
downstream numbers. Per this project's R-89/R-90 convention (a fired Step-0
kill switch does not exempt a branch from a full, honest sweep) this file
still implements, runs, and fully measures the branch end-to-end, and
re-derives the A0 numbers independently below as a sanity check.

WHY THIS IS A DISTINCT QUESTION FROM THE CONSERVATIVE BRANCH, even though
both share the same fired A0 kill switch: the conservative branch reads a
FIXED, literature-set discount, so its A0 relevance is a single yes/no
question -- does the paper's ranking replicate at all. This branch's
scaler is DYNAMIC: it does not know, on any given bar, that full-sample
Rebound is (mildly) favourable -- it only ever sees its OWN running,
one-bar-lagged estimate, built from an expanding window that starts empty.
Two distinct ways this can diverge from the full-sample A0 number are
measured explicitly below (Step A0b): (a) during the `min_obs` burn-in the
scaler is pinned to 1.0 regardless of the true sign, so it cannot act on
this round's kill switch finding either way until enough Correction/Rebound
observations have accumulated, and (b) even after burn-in, an EXPANDING
(not rolling) causal estimate mixes in whatever the *early* Correction/
Rebound sample looked like, which need not match the full-sample number
quoted above. Both are reported as this branch's own distinct finding,
not a restatement of the conservative branch's A0 result.

--------------------------------------------------------------------------
THE FROZEN MECHANISM (not redesigned here; implemented exactly as spec'd
by the operator, semantics preserved bit-for-bit)
--------------------------------------------------------------------------
For a given (k, min_obs):
    state[i]  = r91_shared.state_labels(df)[i]           (Bull/Bear/Corr/Reb)
    bar_ret[i] = log(close[i]) - log(close[i-1])          (bar_ret[0] := 0)
    tracker = CausalStateScaler(min_obs=min_obs, k=k)
    for i in range(len(df)):
        if is_turning_point(state)[i]:
            scaler[i] = min(1.0, tracker.scaler_for(state[i]))   # READ first
        else:
            scaler[i] = 1.0
        tracker.update(state[i], bar_ret[i])                      # THEN update
    raw = v4_vote_frac(df) * scaler * v4_scale(df)
    target = apply_deadband(raw)

`CausalStateScaler` is driven bar-by-bar, in a single forward pass, over
WHATEVER frame it is given, starting fresh at row 0 of that frame every
call -- never pre-computed over a longer frame and then sliced. When this
function is called directly on the full pre-holdout frame (as it is for
the A0b/A2/A3 diagnostics below and for `compare()`'s own r2 computation),
the tracker's running stats accumulate continuously from 2017 through
2022 in one pass, exactly as v4's own EWM vol estimate does. When it is
called by `compare()`'s internal backtest machinery on a
warmup-then-slice frame (`tradebot.window.run_period`'s own fairness
device, shared identically by the control), it starts fresh at row 0 of
THAT frame -- the same convention every one of the prior 90 rounds'
constructs (including the v4 control itself) has been measured under, not
a new relaxation introduced by this branch.

--------------------------------------------------------------------------
THE FROZEN GRID -- exactly 7 configurations, none added or dropped after
results
--------------------------------------------------------------------------
  A1 identity (NOT counted among the 6 swept): min_obs=10**9 (never fires
    within the whole ~6-year pre-holdout series -> scaler == 1.0 always).
    Must reproduce r91_shared.v4_target(df) bit-for-bit.

  swept: k in {1.0, 2.0, 4.0} x min_obs_days in {180, 365}
         (min_obs = min_obs_days * BARS_PER_DAY)                    = 6

--------------------------------------------------------------------------
THE FROZEN DECISION RULE (written before any config past A1 is run)
--------------------------------------------------------------------------
Step A0 (this round's own kill switch, re-derived independently below,
checked BEFORE Step A on any config): causal, inner-train-only,
state-conditional Sharpe-like must rank Correction and Rebound both below
Bull and Bear. FIRED (see above) -- reported, branch proceeds anyway per
R-89/R-90 convention.

Step A, per configuration, before any performance number is read:
  A1 identity  -- min_obs=10**9 config == v4_target(df), exactly.
  A2 non-inert -- R^2 of the candidate's exposure path against v4's own,
                  on inner-train, must be < 0.98 for each of the 6 swept
                  configs.
  A3 causality -- causal_truncation_probe passes (cuts 0.55, 0.80) for
                  each of the 6 swept configs' build functions, plus the
                  identity config, on real BTC inner-train.

Step B selection: `compare()` over slice_names=("inner_train","inner_val"),
markets=(SPOT, FUTURES), for all 7 configs (identity + 6 swept), on BTC.
Selection statistic = inner-validation paired log-growth difference vs v4
on futures_5x, among the Step-A survivors (the 6 swept configs only -- the
identity is control-equivalent by construction and is run through
`compare()` only as a harness sanity check, not eligible for selection).

Promotion bar (default REJECT). ALL FIVE must hold for "CANDIDATE FOR
HOLDOUT"; otherwise NEGATIVE, with the failing clause(s) named exactly:
  B1 -- paired bootstrap excludes zero in >=1 of 4 cells AND point
        estimate positive in all 4.
  B2 -- EITHER dSharpe > +0.2 on inner-val on BOTH markets, OR a max-DD
        improvement on inner-val on BOTH markets WHERE risk_matched is
        True for both -- an unmatched drawdown improvement is not
        evidence (standing rule, R-28/R-32/R-33).
  B3 -- plateau not peak: report the finalist's immediate grid neighbours
        on BOTH swept axes (k and min_obs_days) and state whether they
        move with the finalist or reverse sharply.
  B4 -- falsification, ETH replication (Bitfinex ETH ends 2019-12-31,
        inner_train only): same SIGN of d_loggrowth as BTC inner-train on
        BOTH markets, else FAILS.
  B5 -- cost robustness: re-run BTC inner-validation at 0.40% taker on
        both markets; report whether sign reverses vs the 0.10% baseline.

This branch does NOT read the holdout and does NOT decide promotion to it
-- it reports CANDIDATE FOR HOLDOUT or NEGATIVE; the holdout read, if any,
is the operator's job. The final verdict here is NEGATIVE by
pre-registration (A0 fired) regardless of the B1-B5 outcome, per this
round's own rule -- but every clause is still measured and reported.

This file never reads a bar at or after OOS_START (2023-01-01): every
load goes through r91_shared's truncating loaders, and the max timestamp
actually touched is tracked and printed at the end of main().
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r91_shared import (  # noqa: E402
    BARS_PER_DAY,
    CausalStateScaler,
    FUTURES,
    INNER_TRAIN_END,
    INNER_VAL_START,
    OOS_START,
    SPOT,
    STATE_NAMES,
    apply_deadband,
    causal_state_stats,
    causal_truncation_probe,
    compare,
    fee_at,
    is_turning_point,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    state_labels,
    v4_scale,
    v4_target,
    v4_vote_frac,
)

# ---------------------------------------------------------------------------
# FROZEN GRID
# ---------------------------------------------------------------------------
K_GRID = (1.0, 2.0, 4.0)
MIN_OBS_DAYS_GRID = (180, 365)
IDENTITY_MIN_OBS = 10 ** 9

TAKER_040 = 0.0040  # B5 cost-robustness fee


# ---------------------------------------------------------------------------
# THE FROZEN MECHANISM, exactly as specified.
# ---------------------------------------------------------------------------
def build_target(df: pd.DataFrame, *, min_obs: int, k: float) -> np.ndarray:
    """One forward pass over ``df`` from row 0: a fresh `CausalStateScaler`
    every call, `scaler_for` read BEFORE `update` on every bar, Bull/Bear
    bars left at scaler==1.0, clipped to [0, 1] before multiplying in."""
    state = state_labels(df)
    tp = is_turning_point(state)
    close = df["close"].to_numpy(dtype=float)
    log_close = np.log(close)
    bar_ret = np.diff(log_close, prepend=log_close[0]) if len(log_close) else np.array([])

    tracker = CausalStateScaler(min_obs=min_obs, k=k)
    scaler = np.ones(len(df))
    for i in range(len(df)):
        s = int(state[i])
        if tp[i]:
            scaler[i] = min(1.0, tracker.scaler_for(s))
        tracker.update(s, float(bar_ret[i]))

    raw = v4_vote_frac(df) * scaler * v4_scale(df)
    return apply_deadband(raw)


# ---------------------------------------------------------------------------
# Config wrapper + a small cache (same style as r90_novel_adaptive_ratchet):
# `compare()` calls the build function once directly (for r2_vs_control) and
# again per (slice, market) cell via `TargetStrategy.prepare`; several of
# these calls land on the identical frame (both markets share one slice's
# frame), so caching by frame identity avoids redundant O(n) passes.
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _key(df: pd.DataFrame) -> tuple:
    return (len(df), int(df.index[0].value), int(df.index[-1].value),
            float(df["close"].iloc[0]), float(df["close"].iloc[-1]))


class Config:
    def __init__(self, label: str, *, k: float, min_obs: int, identity: bool = False,
                 min_obs_days: int | None = None):
        self.label = label
        self.k = k
        self.min_obs = min_obs
        self.min_obs_days = min_obs_days
        self.identity = identity

    def build(self, df: pd.DataFrame) -> np.ndarray:
        key = ("build", self.label) + _key(df)
        if key in _CACHE:
            return _CACHE[key]
        out = build_target(df, min_obs=self.min_obs, k=self.k)
        _CACHE[key] = out
        return out


def frozen_grid() -> list[Config]:
    cfgs = []
    for k in K_GRID:
        for md in MIN_OBS_DAYS_GRID:
            cfgs.append(Config(f"k={k:.1f},minobs={md}d", k=k,
                               min_obs=md * BARS_PER_DAY, min_obs_days=md))
    return cfgs


IDENTITY = Config("identity(A1)", k=2.0, min_obs=IDENTITY_MIN_OBS, identity=True)


def hdr(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:
    max_ts = []

    hdr("R-91 NOVEL BRANCH -- CAUSAL EXPANDING-WINDOW STATE-CONDITIONAL SCALER (B-40, GHM 2023)")
    print("mechanism: scaler[i] = clip(tracker.scaler_for(state[i]), 0, 1) at Correction/Rebound bars,")
    print("1.0 at Bull/Bear bars; tracker.update(state[i], bar_ret[i]) called AFTER scaler_for on")
    print("every bar (strict one-bar lag, expanding window, never reset mid-frame).")
    print(f"\nfrozen grid: 1 identity (A1, min_obs={IDENTITY_MIN_OBS:.0e}) + "
          f"k in {K_GRID} x min_obs_days in {MIN_OBS_DAYS_GRID} = 6 swept = 7 total configurations")

    btc = load_btc()
    max_ts.append(btc.index.max())
    print(f"\nBTC: {len(btc):,} bars  {btc.index[0]} -> {btc.index[-1]}  (< {OOS_START})")
    train = btc.loc[:INNER_TRAIN_END]
    print(f"inner-train frame: {len(train):,} bars  {train.index[0]} -> {train.index[-1]}")

    # ================================================================ A0
    hdr("STEP A0 -- KILL SWITCH RE-DERIVATION (independent, before Step A)")
    print("Descriptive (full-inner-train, one-shot) state-conditional Sharpe-like of bar-level")
    print("log returns, contemporaneous state (causal_state_stats), on BTC inner-train:\n")
    state_train = state_labels(train)
    close_train = train["close"].to_numpy(dtype=float)
    bar_ret_train = np.diff(np.log(close_train), prepend=np.log(close_train[0]))
    a0_stats = causal_state_stats(bar_ret_train, state_train)
    print(f"{'state':12s} {'n':>9s} {'mean':>12s} {'vol':>10s} {'sharpe-like':>12s}")
    for k in (0, 1, 2, 3):
        v = a0_stats[k]
        print(f"{STATE_NAMES[k]:12s} {v['n']:9d} {v['mean']:+12.6e} {v['vol']:10.6f} {v['sharpe']:+12.4f}")

    operator_quoted = {0: 0.190, 1: -0.103, 2: -0.066, 3: 0.185}
    print("\nOperator-quoted values: Bull +0.190, Bear -0.103, Correction -0.066, Rebound +0.185")
    max_diff = max(abs(a0_stats[k]["sharpe"] - operator_quoted[k]) for k in operator_quoted)
    print(f"max abs diff vs operator-quoted (rounded to 3dp): {max_diff:.4f}  "
          f"-> {'MATCHES (independently re-derived)' if max_diff < 0.001 else 'DIFFERS -- investigate'}")

    bull_s, bear_s, corr_s, reb_s = (a0_stats[0]["sharpe"], a0_stats[1]["sharpe"],
                                     a0_stats[2]["sharpe"], a0_stats[3]["sharpe"])
    a0_pass = (corr_s < min(bull_s, bear_s)) and (reb_s < min(bull_s, bear_s))
    print(f"\nA0 rule: Correction AND Rebound must both rank below Bull AND Bear.")
    print(f"  Correction ({corr_s:+.4f}) < min(Bull,Bear) ({min(bull_s, bear_s):+.4f})? "
          f"{corr_s < min(bull_s, bear_s)}")
    print(f"  Rebound    ({reb_s:+.4f}) < min(Bull,Bear) ({min(bull_s, bear_s):+.4f})? "
          f"{reb_s < min(bull_s, bear_s)}")
    print(f"  => A0 = {'PASS (mechanism replicates)' if a0_pass else 'FAIL -- KILL SWITCH FIRES'}")
    print("\nPer R-89/R-90 convention, a fired kill switch does not exempt this branch from a full,")
    print("honest sweep; every step below is still run and reported.")

    hdr("STEP A0b -- WHY THE DYNAMIC ESTIMATE IS A DISTINCT QUESTION FROM THE FIXED-DISCOUNT BRANCH")
    print("The tracker's running Sharpe-like estimate for Correction/Rebound, sampled at several")
    print("points along inner-train, shows what the CAUSAL, one-bar-lagged estimate actually 'knew'")
    print("at that point in time (vs the full-sample A0 number above, which is never available to")
    print("a causal estimator in real time):\n")
    probe_tracker = CausalStateScaler(min_obs=180 * BARS_PER_DAY, k=2.0)
    sample_points = [int(len(train) * f) for f in (0.15, 0.30, 0.50, 0.70, 0.90, 1.0)]
    sample_points[-1] = len(train) - 1
    print(f"{'bar#':>9s} {'date':>20s} {'n_corr':>8s} {'n_reb':>8s} "
          f"{'corr_sharpe':>12s} {'reb_sharpe':>12s} {'scaler_corr':>12s} {'scaler_reb':>12s}")
    j = 0
    for i in range(len(train)):
        s = int(state_train[i])
        if j < len(sample_points) and i == sample_points[j]:
            n_corr, n_reb = probe_tracker.n[2], probe_tracker.n[3]
            if n_corr >= probe_tracker.min_obs:
                mean_c = probe_tracker.sum[2] / n_corr
                var_c = max(probe_tracker.sumsq[2] / n_corr - mean_c * mean_c, 1e-12)
                sh_c = mean_c / var_c ** 0.5 * np.sqrt(365.25 * BARS_PER_DAY)
            else:
                sh_c = float("nan")
            if n_reb >= probe_tracker.min_obs:
                mean_r = probe_tracker.sum[3] / n_reb
                var_r = max(probe_tracker.sumsq[3] / n_reb - mean_r * mean_r, 1e-12)
                sh_r = mean_r / var_r ** 0.5 * np.sqrt(365.25 * BARS_PER_DAY)
            else:
                sh_r = float("nan")
            sc_c = min(1.0, probe_tracker.scaler_for(2))
            sc_r = min(1.0, probe_tracker.scaler_for(3))
            print(f"{i:9d} {str(train.index[i].date()):>20s} {n_corr:8d} {n_reb:8d} "
                  f"{sh_c:+12.4f} {sh_r:+12.4f} {sc_c:12.4f} {sc_r:12.4f}")
            j += 1
        probe_tracker.update(s, float(bar_ret_train[i]))
    print(f"\n(min_obs=180d shown; burn-in means scaler_corr/scaler_reb are pinned to 1.0 until each")
    print("state individually accumulates 180*288 observations -- Correction/Rebound bars are rarer")
    print("than Bull/Bear, so this burn-in can span a materially longer WALL-CLOCK stretch of the")
    print("series than 180 calendar days. Distinct finding from the conservative branch: this")
    print("branch's shrinkage in Rebound, where it occurs, is a burn-in / early-sample artifact of a")
    print("causal expanding-window estimator, not a restatement of the fixed A0 ranking.")

    # ================================================================ A1/A2/A3
    hdr("STEP A -- MECHANISM GATE (before any performance number)")

    ident_path_train = IDENTITY.build(train)
    v4_path_train = v4_target(train)
    a1_train_max = float(np.max(np.abs(ident_path_train - v4_path_train)))
    ident_path_full = IDENTITY.build(btc)
    v4_path_full = v4_target(btc)
    a1_full_max = float(np.max(np.abs(ident_path_full - v4_path_full)))
    a1 = (a1_train_max == 0.0) and (a1_full_max == 0.0)
    print(f"A1 identity (min_obs={IDENTITY_MIN_OBS:.0e}, never fires within the pre-holdout series):")
    print(f"  max|identity - v4_target| on inner-train : {a1_train_max:.3e}")
    print(f"  max|identity - v4_target| on full pre-holdout frame: {a1_full_max:.3e}")
    print(f"  A1 = {'PASS' if a1 else 'FAIL'}")

    cfgs = frozen_grid()
    assert len(cfgs) == 6, len(cfgs)

    print("\nA2 non-inertness (inner-train): R^2 of candidate exposure path vs v4_target must be < 0.98")
    print(f"{'config':22s} {'k':>5s} {'min_obs_d':>9s} {'R^2 vs v4':>10s} {'status':>8s}")
    print("-" * 60)
    a2 = {}
    for cfg in cfgs:
        cand_path = cfg.build(train)
        rsq = r_squared(cand_path, v4_path_train)
        ok = rsq < 0.98
        a2[cfg.label] = dict(r2=rsq, ok=ok)
        print(f"{cfg.label:22s} {cfg.k:5.1f} {cfg.min_obs_days:9d} {rsq:10.5f} "
              f"{'ok' if ok else 'INERT':>8s}")
    n_inert = sum(1 for v in a2.values() if not v["ok"])
    print(f"\n{n_inert} of 6 swept configurations are non-inert-FAIL (R^2 >= 0.98) and excluded from "
          f"selection.")

    hdr("STEP A3 -- CAUSAL TRUNCATION PROBE")
    print("Rebuild the target on 55% and 80% truncations of inner-train; the surviving prefix must")
    print("match bit-for-bit. This is the check most likely to catch a stateful tracker being driven")
    print("over the full series and then sliced, instead of being re-driven fresh from bar 0 of")
    print("whatever frame is passed in.\n")
    probe_cfgs = [IDENTITY] + cfgs
    a3 = {}
    all_a3_pass = True
    for cfg in probe_cfgs:
        try:
            ok = causal_truncation_probe(cfg.build, train, cuts=(0.55, 0.80))
            msg = "PASS (cuts 0.55, 0.80)"
        except AssertionError as exc:
            ok = False
            msg = f"FAIL -- {exc}"
        a3[cfg.label] = ok
        all_a3_pass = all_a3_pass and ok
        print(f"  {cfg.label:22s} {msg}")
    print(f"\nA3 = {'PASS' if all_a3_pass else 'FAIL'} (checked on all 7 configurations, not just the "
          f"eventual finalist)")

    survivors = [c for c in cfgs if a2[c.label]["ok"]]
    print(f"\nStep A survivors eligible for selection: {len(survivors)} of 6 "
          f"({', '.join(c.label for c in survivors)})")

    # ================================================================ STEP B
    hdr("STEP B -- FULL GRID, 7 configurations (1 identity + 6 swept) x 4 (slice x market) cells")
    print("candidate vs kelly_regime_v4 control; d_loggrowth is the paired block-bootstrap difference")
    print("(30-day blocks, 2000 resamples). All 7 configs run and reported regardless of A2 status.\n")

    all_rows: dict[str, list[dict]] = {}
    for cfg in [IDENTITY] + cfgs:
        rows = compare(cfg.build, btc, label=cfg.label)
        all_rows[cfg.label] = rows
        print_rows(rows)
        print()

    def sel_stat(label: str) -> float:
        for r in all_rows[label]:
            if r["slice"] == "inner_val" and r["market"] == "futures_5x":
                return r["d_loggrowth"]
        return float("nan")

    hdr("SELECTION -- inner-validation paired log-growth difference vs v4, futures_5x")
    print(f"{'config':22s} {'eligible':>9s} {'selstat':>9s} {'[lo':>9s},{'hi]':>9s} {'why not':<20s}")
    print("-" * 82)
    ident_row = [r for r in all_rows[IDENTITY.label]
                 if r["slice"] == "inner_val" and r["market"] == "futures_5x"][0]
    print(f"{IDENTITY.label:22s} {'n/a':>9s} {ident_row['d_loggrowth']:+9.3f} "
          f"{ident_row['d_lo']:+9.3f},{ident_row['d_hi']:+9.3f} "
          f"{'not swept (control-equiv, harness check only)':<20s}")
    for cfg in cfgs:
        reasons = []
        if not a2[cfg.label]["ok"]:
            reasons.append("R^2>=0.98 (inert)")
        ok = not reasons
        row = [r for r in all_rows[cfg.label]
               if r["slice"] == "inner_val" and r["market"] == "futures_5x"][0]
        print(f"{cfg.label:22s} {'YES' if ok else 'no':>9s} "
              f"{row['d_loggrowth']:+9.3f} {row['d_lo']:+9.3f},{row['d_hi']:+9.3f} "
              f"{'; '.join(reasons):<20s}")

    if not survivors:
        hdr("VERDICT")
        print("No Step-A-eligible swept configuration. Combined with the A0 kill switch already")
        print("fired above, VERDICT: NEGATIVE.")
        print(f"\nConfigurations evaluated: 1 identity + 6 swept = 7 distinct configurations.")
        print(f"BTC cells (7 configs x 2 markets x 2 slices): 28")
        print(f"max timestamp read anywhere: {max(max_ts)}  (< {OOS_START})")
        return

    finalist = max(survivors, key=lambda c: sel_stat(c.label))
    print(f"\nFINALIST: {finalist.label}  (k={finalist.k}, min_obs_days={finalist.min_obs_days})   "
          f"selection statistic {sel_stat(finalist.label):+.4f} log units (inner-val, futures_5x)")

    frows = all_rows[finalist.label]
    print()
    print_rows(frows)

    # ------------------------------------------------------------ B1-B5
    hdr("THE FROZEN DECISION RULE -- clause by clause (B1-B5)")

    pts = [r["d_loggrowth"] for r in frows]
    excl = [r["excludes_zero"] for r in frows]
    b1 = any(excl) and all(p > 0 for p in pts)
    print(f"B1 paired bootstrap: excludes zero in {sum(excl)}/4 cells; point estimate positive in "
          f"{sum(1 for p in pts if p > 0)}/4 cells")
    for r in frows:
        print(f"   {r['slice']:11s} {r['market']:11s} d_loggrowth={r['d_loggrowth']:+.4f} "
              f"[{r['d_lo']:+.4f},{r['d_hi']:+.4f}]  excludes_zero={r['excludes_zero']}")
    print(f"   B1 = {'PASS' if b1 else 'FAIL'}")

    val = [r for r in frows if r["slice"] == "inner_val"]
    dsh = {r["market"]: r["d_sharpe"] for r in val}
    ddd = {r["market"]: r["d_dd"] for r in val}
    rm = {r["market"]: r["risk_matched"] for r in val}
    b2_sharpe = all(v > 0.2 for v in dsh.values())
    b2_dd = all(v < 0.0 for v in ddd.values()) and all(rm.values())
    b2 = b2_sharpe or b2_dd
    print(f"\nB2 noise floor (inner-validation):")
    print(f"   dSharpe:  " + ", ".join(f"{k}={v:+.3f}" for k, v in dsh.items())
          + f"   -> both > +0.2: {b2_sharpe}")
    print(f"   dMaxDD:   " + ", ".join(f"{k}={v:+.2f}pp" for k, v in ddd.items())
          + f"   -> both improved: {all(v < 0.0 for v in ddd.values())}")
    print(f"   risk_matched: " + ", ".join(f"{k}={v}" for k, v in rm.items())
          + f"   -> both matched: {all(rm.values())}")
    print(f"   exposure_ratio / vol_ratio (cand/v4), inner-val: " + ", ".join(
        f"{r['market']}=exp{r['exposure_ratio']:.3f}/vol{r['vol_ratio']:.3f}" for r in val))
    print(f"   drawdown leg counts as evidence only if BOTH improved AND BOTH risk-matched: {b2_dd}")
    print(f"   B2 = {'PASS' if b2 else 'FAIL'} (via "
          f"{'Sharpe' if b2_sharpe else ('matched drawdown' if b2_dd else 'neither')})")

    print("\nB3 plateau not peak: the finalist's immediate grid neighbours (k axis and min_obs axis)")
    print(f"{'config':22s} {'k':>5s} {'minobs_d':>8s} {'selstat':>9s} {'note':<14s}")
    print("-" * 65)
    fk, fmd = finalist.k, finalist.min_obs_days
    neigh_labels = []
    ik = K_GRID.index(fk)
    for di in (-1, 1):
        j = ik + di
        if 0 <= j < len(K_GRID):
            neigh_labels.append(f"k={K_GRID[j]:.1f},minobs={fmd}d")
    other_md = [m for m in MIN_OBS_DAYS_GRID if m != fmd]
    for md in other_md:
        neigh_labels.append(f"k={fk:.1f},minobs={md}d")
    b3_vals = []
    for lb in [finalist.label] + neigh_labels:
        s = sel_stat(lb)
        note = "<-- FINALIST" if lb == finalist.label else ""
        cfgobj = next(c for c in cfgs if c.label == lb)
        print(f"{lb:22s} {cfgobj.k:5.1f} {cfgobj.min_obs_days:8d} {s:+9.4f} {note:<14s}")
        if lb != finalist.label:
            b3_vals.append(s)
    fin_stat = sel_stat(finalist.label)
    same_direction = all(np.sign(v) == np.sign(fin_stat) for v in b3_vals) if b3_vals else False
    print(f"\n   finalist selstat sign: {'+' if fin_stat > 0 else '-'}; neighbours all same sign: "
          f"{same_direction}")
    b3 = bool(b3_vals) and same_direction
    print(f"   B3 = {'PASS' if b3 else 'FAIL'} ('plateau' reading: neighbours move with the finalist "
          f"rather than reversing sharply, per the values above)")

    hdr("B4 -- FALSIFICATION: ETH REPLICATION (inner_train only)")
    eth = load_eth()
    max_ts.append(eth.index.max())
    print(f"ETH (Bitfinex): {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}")
    print(f"ETH ends {eth.index.max().date()}, before inner-validation begins ({INNER_VAL_START}) -- "
          f"only inner_train is run on ETH; reported plainly, not worked around.")
    eth_cfg = Config(finalist.label, k=finalist.k, min_obs=finalist.min_obs,
                     min_obs_days=finalist.min_obs_days)
    eth_rows = compare(eth_cfg.build, eth, label=finalist.label, slice_names=("inner_train",))
    print()
    print_rows(eth_rows)
    btc_train_sign_by_mkt = {r["market"]: (1.0 if r["d_loggrowth"] > 0 else -1.0)
                             for r in frows if r["slice"] == "inner_train"}
    eth_ok = []
    for r in eth_rows:
        btc_sign = btc_train_sign_by_mkt.get(r["market"])
        same = btc_sign is not None and np.sign(r["d_loggrowth"]) == btc_sign
        eth_ok.append(same)
        print(f"   {r['market']:11s}: BTC inner-train sign={'+' if btc_sign and btc_sign > 0 else '-'}  "
              f"ETH d_loggrowth={r['d_loggrowth']:+.4f}  same sign: {same}")
    b4 = bool(eth_rows) and all(eth_ok)
    print(f"   B4 = {'PASS' if b4 else 'FAIL'}")

    hdr("B5 -- COST ROBUSTNESS: 0.40% TAKER, inner-validation")
    spot_040 = fee_at(SPOT, TAKER_040)
    fut_040 = fee_at(FUTURES, TAKER_040)
    fee_cfg = Config(finalist.label + "@40bp", k=finalist.k, min_obs=finalist.min_obs,
                     min_obs_days=finalist.min_obs_days)
    fee_rows = compare(fee_cfg.build, btc, label=fee_cfg.label,
                       markets=(spot_040, fut_040), slice_names=("inner_val",))
    print()
    print_rows(fee_rows)
    val_sign_by_mkt = {r["market"]: (1.0 if r["d_loggrowth"] > 0 else -1.0) for r in val}
    fee_sign_ok = []
    for r in fee_rows:
        base_mkt = "spot" if "spot" in r["market"] else "futures_5x"
        base_sign = val_sign_by_mkt.get(base_mkt)
        same = base_sign is not None and np.sign(r["d_loggrowth"]) == base_sign
        fee_sign_ok.append(same)
        print(f"   {r['market']:14s} d_loggrowth={r['d_loggrowth']:+.4f}  "
              f"(base-fee inner-val {base_mkt} sign={'+' if base_sign and base_sign > 0 else '-'})  "
              f"same sign: {same}")
    b5 = all(fee_sign_ok)
    print(f"   B5 = {'PASS' if b5 else 'FAIL'} (sign reversal vs 0.10% baseline: {not b5})")

    # ------------------------------------------------------------ verdict
    hdr("VERDICT")
    print(f"A0 (this round's own kill switch): {'PASS' if a0_pass else 'FIRED (FAIL)'} -- "
          f"pre-registration says NEGATIVE regardless of B1-B5, unless the operator sets A0 aside")
    print("per convention (not this branch's call to make).")
    clauses = {"B1": b1, "B2": b2, "B3": b3, "B4 ETH": b4, "B5 0.40% taker": b5}
    for k, v in clauses.items():
        print(f"  {k:16s} {'PASS' if v else 'FAIL'}")
    b_all_pass = all(clauses.values())
    promote = a0_pass and b_all_pass
    print(f"\nVERDICT: {'CANDIDATE FOR HOLDOUT' if promote else 'NEGATIVE'}")
    if not promote:
        failed = ([] if a0_pass else ["A0 (kill switch fired)"]) + \
                 [k for k, v in clauses.items() if not v]
        print(f"Failing clause(s): {', '.join(failed)}")

    print(f"\nFinalist config: k={finalist.k}, min_obs_days={finalist.min_obs_days} "
          f"(min_obs={finalist.min_obs} bars)")

    print(f"\nConfigurations evaluated in this file:")
    print(f"  distinct configurations: 7 (1 identity + 6 swept)")
    print(f"  BTC cells via compare() (7 configs x 2 markets x 2 slices): 28")
    print(f"  finalist ETH cells (1 config x 2 markets x 1 slice, inner_train only): 2")
    print(f"  finalist fee-robustness cells (1 config x 2 markets x 1 slice, inner_val @0.40%): 2")
    print(f"  => total configuration-cells evaluated: 32")
    print(f"\nmax timestamp read anywhere in this branch (BTC and ETH): {max(max_ts)}  "
          f"(< {OOS_START}) -- no holdout bar was read.")


if __name__ == "__main__":
    main()
