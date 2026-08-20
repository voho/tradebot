"""R-68 operator measurement: attack the INTERVAL, not the mechanism.

Prices, R-67's own committed rule, and three inference procedures this
project has never run. No new strategy, no new parameter, no holdout read.

=====================================================================
WHY THIS FILE EXISTS
=====================================================================

R-67's one-line lesson is the ledger's live problem:

    three rounds (R-63 -> R-65 -> R-67) have improved this signal's
    economics by 10-80x and every one died on the same interval, which
    is no longer a fact about any mechanism but a fact about how much
    this dataset can resolve.

R-68 dispatched two branches at the mechanism anyway, because B-34's two
questions are answerable independently of the interval. This file attacks
the interval itself, and it exists because the round's commissioned
literature survey said, in its own summary, that the yield is here and not
in a fourth mechanism -- and named three specific procedures nobody in this
repo has run. All three are run below.

**(1) Test the DIFFERENCE SERIES, not two overlapping level intervals.**
Every D1 cell in R-63/R-65/R-67 compares one arm against a volatility-matched
hold and reports an interval containing zero. Not one round has asked the
question the mechanism is actually about: *does the banded arm beat the
UNBANDED arm?* Those two P&L series are enormously correlated -- they are the
same signal traded at different speeds -- so their difference can be far
better resolved than either level. This is precisely the construction behind
Novy-Marx & Velikov's `alpha^{FF4+}_net` (a net alpha measured against the
corresponding un-banded strategy), which is why their incremental t-statistics
reach 2.1-3.5 where level tests would show nothing. The formal tool is
Ledoit, O., & Wolf, M. (2008), "Robust performance hypothesis testing with
the Sharpe ratio," *Journal of Empirical Finance* 15(5), 850-859: a
studentized time-series bootstrap interval for the DIFFERENCE between two
strategies. `compare(cand_banded, cand_unbanded)` is that test in this repo's
existing machinery, and no round has called it with two candidates.

**(2) Test the WHOLE SWEEP for monotone trend, not one selected cell.**
Patton, A. J., & Timmermann, A. (2010), "Monotonicity in asset returns: New
tests with applications to the term structure, the CAPM, and portfolio
sorts," *Journal of Financial Economics* 98(3), 605-625. With cells ordered
by delta and mu_i their risk-matched mean daily edge, Delta_i = mu_i -
mu_{i-1}, their MR test is

    H0: Delta <= 0        H1: min_i Delta_i > 0        J_T = min_i Delta_i

with critical values from the stationary bootstrap of Politis & Romano
(1994), the null imposed at the least-favourable point Delta = 0 (following
White 2000), and -- the property that makes it the right tool here -- **a
randomized time index COMMON ACROSS CELLS, so cross-cell dependence is
preserved rather than assumed away**. A parameter sweep evaluated on one
price history is the maximally dependent case, and this test is built for it.
Their Up/Down statistics are computed too, because they diagnose exactly the
situation this project is in: whether a non-rejection is a shape problem or a
power problem.

Recorded against the test, not glossed: Romano, J. P., & Wolf, M. (2013),
"Testing for monotonicity in expected asset returns," *Journal of Empirical
Finance* 23, 93-116, show the PT test can break down when the relation is
non-monotonic or only weakly increasing, and can then falsely establish a
strictly increasing relation. NBIM's own no-trade-band study (Discussion Note
01/2018) found a **hump-shaped** alpha profile in band width -- rising, then
insignificant beyond 3-4pp, then negative -- which is exactly that case. So
the MR test here is reported as ONE piece of evidence with a named failure
mode, the Up/Down decomposition is reported beside it precisely because it
reveals a hump, and a rejection would NOT be treated as establishing
monotonicity on its own.

**(3) Report BETC-5%, not just break-even.**
Fieberg, Liedtke, Poddig, Walker & Zaremba (*JFQA* 60(7), 2025, 3116-3153)
report, for exactly this kind of crypto cross-sectional strategy, both the
break-even transaction cost (the cost at which net return hits zero) and
**BETC-5%** (the cost at which the net return stops being significant at
5%). This project has quoted a 0.104% break-even since R-63 and has never
computed the second number, which is always the smaller one and is the
honest one. It is computed below over a fee grid, for the point estimate and
for the interval's lower bound separately.

=====================================================================
WHAT THIS FILE DOES NOT DO
=====================================================================

- It does not select anything, and it changes no verdict. It is a
  measurement of how much the data can resolve, run on a rule that was
  frozen and published by a previous round.
- It uses **R-67's own committed implementation** (`build_hysteresis_targets`
  from `experiments/r67_conservative_hysteresis.py`) deliberately, so that
  what is under test is the inference and not a re-implementation. That file
  was independently reproduced by the operator this session: re-running its
  `identity` and `frontier` commands returned its committed CSVs unchanged,
  and its published W_VAL ordering to four decimals.
- **Holdout: +0.** Everything below runs on W_TRAIN and W_VAL only. W_FULL6
  is deliberately NOT used, even though R-63/R-65/R-67 all score their
  headline cells on it, because that window runs past `OOS_START` and B-33 --
  whether such a read is genuinely free -- is still unresolved and now
  load-bearing on two rounds. Declining to add a third is free here.
- The MR test's power is limited by the number of cells (11) and by their
  dependence. A non-rejection is the expected outcome and is not evidence
  of a flat frontier; that is what the Up/Down statistics are for.

    .venv/bin/python experiments/r68_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import csv  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns,
    max_drawdown_from_returns,
    paired_bootstrap,
    stationary_bootstrap_indices,
    total_log_return,
)

from experiments.r68_shared import (  # noqa: E402
    DELTA_GRID_EXT,
    OUT_DIR,
    SPOT_BASE,
    SPOT_FREE,
    UNIVERSE_8,
    W_TRAIN,
    W_VAL,
    align_frames,
    compare,
    config_count,
    cross_sectional_score,
    load_universe,
    volmatched_hold_equity,
    warm_window,
)
from experiments.r67_conservative_hysteresis import (  # noqa: E402
    BUFFER_FIXED,
    HOLD_FIXED,
    K_FIXED,
    build_hysteresis_targets,
)

BOOT_N = 2_000
BOOT_BLOCK = 30.0  # days; the block length every other cell in this axis uses
BOOT_SEED = 7

# The fee grid for BETC / BETC-5%. Spans free to 1.0%, bracketing the 0.10%
# base tier, the 0.40% real tier and R-63's published 0.104% break-even.
FEE_GRID = (0.0, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003, 0.004,
            0.006, 0.008, 0.010)

# The two cells the difference test compares. Both are published:
# delta=0.000 is R-65's frozen winner, delta=0.080 is R-67's.
DELTA_UNBANDED = 0.000
DELTA_BANDED = 0.080


# ------------------------------------------------------------------ cells


def build_cell(frames, window, delta):
    """Aligned prices + targets, sliced to the evaluation window, with the
    strict right edge R-63's conservative branch added after finding the
    shared helper admitted one bar of the reserved holdout."""
    sub = {t: frames[t] for t in UNIVERSE_8}
    warm = align_frames(sub, warm_window(window))
    targets = build_hysteresis_targets(warm, K_FIXED, BUFFER_FIXED,
                                       HOLD_FIXED, delta)
    idx = warm[UNIVERSE_8[0]].index
    idx = idx[idx >= pd.Timestamp(window[0], tz="UTC")]
    idx = idx[idx < pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)]
    assert not (idx >= pd.Timestamp("2023-01-01", tz="UTC")).any(), \
        "holdout hygiene: this file evaluates W_TRAIN and W_VAL only"
    return {t: df.loc[idx] for t, df in warm.items()}, targets.loc[idx]


def cell_series(frames, window, delta, market=SPOT_BASE):
    """(candidate daily returns, volatility-matched-hold daily returns, matched)."""
    from experiments.r68_shared import simulate_portfolio
    aligned, targets = build_cell(frames, window, delta)
    eq = simulate_portfolio(targets, aligned, market)
    bench, c, vol, matched = volmatched_hold_equity(eq, aligned, UNIVERSE_8, market)
    if bench is None:
        return None, None, False, targets
    a = daily_returns(eq).to_numpy(dtype=float)
    b = daily_returns(bench).to_numpy(dtype=float)
    n = min(len(a), len(b))
    return a[:n], b[:n], matched, targets


def compare_daily(a: np.ndarray, b: np.ndarray) -> dict:
    """`r63_shared.compare`, but taking DAILY RETURNS directly.

    The shared helper takes equity series and re-derives daily returns by
    resampling a DatetimeIndex; everything here already holds daily returns,
    and rebuilding a fake equity index to hand back would be a second place
    for a resampling convention to drift. Same statistics, same paired
    stationary bootstrap, same block length and seed.
    """
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    kw = dict(mean_block=BOOT_BLOCK, n_boot=BOOT_N, seed=BOOT_SEED)
    growth = paired_bootstrap(a, b, total_log_return, **kw)
    ddown = paired_bootstrap(a, b, max_drawdown_from_returns, **kw)
    return {
        "growth_diff": growth.diff.point,
        "growth_lo": growth.diff.lo,
        "growth_hi": growth.diff.hi,
        "dd_diff": ddown.diff.point,
        "dd_lo": ddown.diff.lo,
        "dd_hi": ddown.diff.hi,
        "n_days": n,
    }


# ------------------------------------------------- (2) monotonicity test


def mr_test(mu_matrix: np.ndarray, seed: int = BOOT_SEED, n_boot: int = BOOT_N):
    """Patton & Timmermann (2010) MR test with a COMMON randomized time index.

    ``mu_matrix`` is ``(n_days, n_cells)``: column i is cell i's per-day
    risk-matched edge (candidate minus its own volatility-matched hold),
    cells ordered by increasing delta. One stationary-bootstrap index vector
    is drawn per replication and applied to EVERY column, which is what
    preserves the cross-cell dependence a one-dataset sweep has.

    Returns the observed statistics, their studentized forms, and
    null-imposed bootstrap p-values. Studentization follows PT's own
    recommendation (citing Hansen 2005, Romano & Wolf 2005).
    """
    n_days, n_cells = mu_matrix.shape
    mu = mu_matrix.mean(axis=0)
    delta = np.diff(mu)                      # Delta_i, i = 1..n_cells-1

    idx = stationary_bootstrap_indices(n_days, BOOT_BLOCK, n_boot,
                                       np.random.default_rng(seed))
    boot_mu = mu_matrix[idx].mean(axis=1)    # (n_boot, n_cells)
    boot_delta = np.diff(boot_mu, axis=1)    # (n_boot, n_cells-1)

    sd = boot_delta.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)

    t_obs = delta / sd
    j_obs = float(np.nanmin(t_obs))
    j_boot = np.nanmin((boot_delta - delta) / sd, axis=1)
    p_mr = float(np.mean(j_boot > j_obs))

    up = float(np.sum(delta[delta > 0]))
    down = float(np.sum(np.abs(delta[delta < 0])))
    boot_up = np.array([float(np.sum(d[d > 0])) for d in (boot_delta - delta)])
    boot_down = np.array([float(np.sum(np.abs(d[d < 0])))
                          for d in (boot_delta - delta)])
    return {
        "n_cells": n_cells,
        "n_days": n_days,
        "mu": mu,
        "delta": delta,
        "delta_sd": sd,
        "delta_t": t_obs,
        "J_T": j_obs,
        "p_mr": p_mr,
        "up": up,
        "down": down,
        "p_up": float(np.mean(boot_up > up)),
        "p_down": float(np.mean(boot_down > down)),
        "n_delta_positive": int((delta > 0).sum()),
        "n_delta_negative": int((delta < 0).sum()),
    }


# --------------------------------------------------- (3) BETC and BETC-5%


def betc_curve(frames, window, delta):
    """Net growth difference vs the volatility-matched hold, as a function of
    the taker fee. Returns one row per fee, with the interval."""
    rows = []
    for fee in FEE_GRID:
        market = SPOT_FREE if fee == 0.0 else MarketSpec.spot(fee_rate=fee)
        a, b, matched, _ = cell_series(frames, window, delta, market)
        if a is None:
            continue
        cmp_ = compare_daily(a, b)
        rows.append({
            "window": window[0] + ".." + str(window[1]),
            "delta": delta,
            "fee": fee,
            "risk_matched": matched,
            "growth_diff": cmp_["growth_diff"],
            "growth_lo": cmp_["growth_lo"],
            "growth_hi": cmp_["growth_hi"],
        })
    return rows


def _cross(xs, ys):
    """First x at which y crosses from >0 to <=0, by linear interpolation.
    None if y never starts positive; the left edge if it is never positive."""
    for i in range(1, len(xs)):
        if ys[i - 1] > 0.0 >= ys[i]:
            span = ys[i - 1] - ys[i]
            frac = ys[i - 1] / span if span else 0.0
            return xs[i - 1] + frac * (xs[i] - xs[i - 1])
    return None


# ------------------------------------------------------------------ driver


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def main():
    frames = load_universe(UNIVERSE_8)
    mr_rows, diff_rows, betc_rows = [], [], []

    for window, name in ((W_TRAIN, "W_TRAIN"), (W_VAL, "W_VAL")):
        print(f"\n=== {name} {window} ===")

        # ---- build every cell once, keeping the per-day risk-matched edge
        cols, kept, voided = [], [], []
        series = {}
        for delta in DELTA_GRID_EXT:
            a, b, matched, _ = cell_series(frames, window, delta)
            if a is None or not matched:
                voided.append(delta)
                continue
            cols.append(a - b)
            kept.append(delta)
            series[delta] = a
        if voided:
            print(f"  VOIDED (risk match failed, per ROUTINE's standing rule): "
                  f"{voided}")
        n = min(len(c) for c in cols)
        mu_matrix = np.column_stack([c[:n] for c in cols])

        # ---- (2) monotone trend across the sweep
        res = mr_test(mu_matrix)
        print(f"  -- (2) Patton-Timmermann MR test over {res['n_cells']} cells, "
              f"{res['n_days']} days --")
        for i, d in enumerate(kept):
            print(f"     delta={d:.3f}  mu={res['mu'][i]:+.6f}/day"
                  + (f"   Delta={res['delta'][i-1]:+.6f} "
                     f"(t={res['delta_t'][i-1]:+.2f})" if i else ""))
        print(f"     J_T = {res['J_T']:+.3f}   p(MR) = {res['p_mr']:.3f}   "
              f"[{res['n_delta_positive']} up, {res['n_delta_negative']} down]")
        print(f"     Up = {res['up']:+.6f} (p={res['p_up']:.3f})   "
              f"Down = {res['down']:+.6f} (p={res['p_down']:.3f})")
        mr_rows.append({
            "window": name, "n_cells": res["n_cells"], "n_days": res["n_days"],
            "deltas": " ".join(f"{d:.3f}" for d in kept),
            "J_T": res["J_T"], "p_mr": res["p_mr"],
            "up": res["up"], "p_up": res["p_up"],
            "down": res["down"], "p_down": res["p_down"],
            "n_delta_positive": res["n_delta_positive"],
            "n_delta_negative": res["n_delta_negative"],
            "voided": " ".join(f"{d:.3f}" for d in voided),
        })

        # ---- (1) the difference test nobody has run: banded vs unbanded
        if DELTA_BANDED in series and DELTA_UNBANDED in series:
            a = series[DELTA_BANDED]
            b = series[DELTA_UNBANDED]
            cmp_ = compare_daily(a, b)
            print(f"  -- (1) difference test: delta={DELTA_BANDED} vs "
                  f"delta={DELTA_UNBANDED} (R-67's arm vs R-65's) --")
            print(f"     growth diff {cmp_['growth_diff']:+.4f} "
                  f"[{cmp_['growth_lo']:+.4f}, {cmp_['growth_hi']:+.4f}]   "
                  f"dd diff {cmp_['dd_diff']:+.2f}pp "
                  f"[{cmp_['dd_lo']:+.2f}, {cmp_['dd_hi']:+.2f}]")
            diff_rows.append({"window": name, "banded": DELTA_BANDED,
                              "unbanded": DELTA_UNBANDED, **cmp_})

        # ---- (3) BETC and BETC-5% for R-67's own selected cell
        rows = betc_curve(frames, window, DELTA_BANDED)
        betc_rows += rows
        fees = [r["fee"] for r in rows]
        betc = _cross(fees, [r["growth_diff"] for r in rows])
        betc5 = _cross(fees, [r["growth_lo"] for r in rows])
        print(f"  -- (3) BETC for delta={DELTA_BANDED} --")
        for r in rows:
            print(f"     fee {r['fee']*100:>5.2f}%  growth {r['growth_diff']:+.4f} "
                  f"[{r['growth_lo']:+.4f}, {r['growth_hi']:+.4f}]"
                  + ("" if r["risk_matched"] else "   VOID (risk match failed)"))
        print(f"     BETC     = "
              + (f"{betc*100:.4f}%" if betc is not None
                 else "n/a (never positive on this window)"))
        print(f"     BETC-5%  = "
              + (f"{betc5*100:.4f}%" if betc5 is not None
                 else "n/a -- the lower bound is <= 0 at ZERO fee, so there is "
                      "no cost at which this edge is significant"))

    write_csv(OUT_DIR / "r68_monotonicity.csv", mr_rows)
    write_csv(OUT_DIR / "r68_difference_test.csv", diff_rows)
    write_csv(OUT_DIR / "r68_betc.csv", betc_rows)
    print(f"\nconfig_count() = {config_count()}")
    print("Holdout consultations: +0 (W_TRAIN and W_VAL only; W_FULL6 is "
          "deliberately not used -- see B-33).")


if __name__ == "__main__":
    main()
