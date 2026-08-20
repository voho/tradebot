"""R-64 NOVEL branch -- partial adjustment toward a persistence-weighted AIM
portfolio (Garleanu & Pedersen).

The pre-registration for this round is the module docstring of
`experiments/r64_shared.py`, which is FROZEN and is not edited here. This
file implements one candidate and measures it; it does not define or relax
a decision rule.

=====================================================================
THE THEORY, AND WHICH TWO PARTS OF IT ARE BEING TESTED
=====================================================================

Garleanu, N., & Pedersen, L. H. (2013), "Dynamic Trading with Predictable
Returns and Transaction Costs", *Journal of Finance* 68(6), 2309-2340; and
its continuous-time companion Garleanu & Pedersen (2016), "Dynamic
portfolio choice with frictions", *Journal of Economic Theory* 165,
487-516. Their result has two halves and BOTH are the point of this branch:

**(a) Trade partially toward an aim, never to the target.** With quadratic
transaction costs and mean-reverting return predictors the optimal policy
is `x_t = x_{t-1} + a (aim_t - x_{t-1})` -- move a constant fraction
`a in (0,1]` of the way toward an *aim* portfolio each period. `a` rises
with signal persistence and falls with transaction cost.

**(b) The aim OVERWEIGHTS PERSISTENT SIGNALS.** When several predictors
with different decay rates are combined, the aim does NOT weight them by
unconditional predictive power. In GP (2016)'s continuous-time form a
factor decaying at rate `phi_j` enters the aim shrunk by
`a / (a + phi_j) = 1 / (1 + phi_j/a)`. Fast-decaying factors are
discounted because you cannot trade fast enough to capture them before
they die, and paying to chase them is negative-value.

R-63's score is `mean over h in {20,40,80} days of (close/anchor_h - 1)` --
an EQUAL weight over three horizons whose measured decay rates differ by
a factor of ~2.8 (below). GP says that is wrong on both counts: it
over-weights the fast horizon *and* it rebalances all the way to the
target. This branch fixes both, and the GP(a)-vs-GP(b) decomposition below
isolates how much each half is worth.

=====================================================================
THE SUBSTRATE IS R-63'S, PROVABLY
=====================================================================

`HORIZONS`, `BARS_PER_DAY`, `WARM_DAYS`, `warm_window`, `DEADBAND`,
`conditional_vol_scale`, `basket_log_returns` and `r63_baseline_targets`
are all imported from `r64_shared`, which re-exports them from R-63's own
novel arm. Nothing is copied. The horizon components below are literally
the three terms R-63 averages, taken separately instead of averaged.

Construction, in order:

  1. COMPONENTS. `c_{i,h}(t) = close_i(t) / anchor_{i,h}(t) - 1` with
     `anchor_{i,h} = close_i.rolling(h*288).mean()`, for h in (20,40,80).
     R-63's `cross_sectional_score` is exactly `mean_h c_{i,h}`.

  2. PER-HORIZON TARGET (Markowitz) PORTFOLIO. For each h, R-63's OWN
     frozen selection rule applied to that single component: hold the
     single highest-scoring asset that has a POSITIVE component, else
     flat. R-63 froze k=1, so its target portfolio is the top-1 portfolio;
     applying the identical rule per component is the minimal-change
     decomposition and keeps the arm on R-63's substrate. This is a
     CROSS-SECTION: an argmax across columns within one row, never a
     time-series quantile.

  3. AIM. `aim_t = sum_h  W_h * m_h(t)`, with the GP persistence weights
     `W_h propto a/(a + phi_h)`, normalized to sum to 1 so the aim's total
     notional stays in [0,1] and so the equal-weight comparison
     (`W_h = 1/3`, R-63's weighting) is like-for-like.

  4. PARTIAL ADJUSTMENT ON THE WEIGHT VECTOR.
     `x_t = x_{t-1} + a (aim_t - x_{t-1})`, applied elementwise to the
     asset weight vector, so the arm holds continuous fractional weights
     across several assets and never makes a discrete swap. `x` is
     initialized to `aim` at the first bar on which the aim is live (deep
     inside the 91-day warm-up, ~11 days before any evaluation window
     starts), so a small `a` does not enter the evaluation window still
     converging from zero.

     WEIGHT FLOOR. `x_i` decays geometrically and never reaches exactly
     zero, so without a floor the "set of held assets" would be
     "everything ever held" and `holding_period_days` would be
     meaningless. Weights below `W_FLOOR = 1e-3` of the pre-scale shape
     (~$0.5 on a $1,000 book after the vol scale, below the engine's own
     $5 minimum notional) are set to zero. This is a measurement
     hygiene constant, not a tuned parameter, and it is not swept. It is
     applied to the OUTPUT of the recursion, never fed back into the
     state -- an in-state floor deadlocks slow arms, because an entering
     asset grows by only `a*W_h` per bar and would be zeroed before it
     could accumulate past the floor.

  5. SIZING. R-63's `conditional_vol_scale`, unchanged, driven by the
     EQUAL-WEIGHT ALL-N basket's log returns -- NOT by this arm's own
     holdings. That circularity was explicitly ruled out in R-63 and it
     stays ruled out. Desired TOTAL notional is `scale(t) * sum_i x_i(t)`,
     latched through R-63's 0.10 `DEADBAND` exactly as R-63 latches
     `scale * m/k`, clipped to 1.0 (long-only, unlevered), then split
     across assets in proportion to `x`. With `sum_i x_i = 1` and the
     latch not binding this reduces to R-63's `w_i = scale * x_i`.

=====================================================================
THE MEASURED DECAY RATES  (W_TRAIN ONLY -- `python ... fit` reproduces)
=====================================================================

An AR(1) through the origin, `y_{t+1} = rho y_t`, fitted by OLS on the
CROSS-SECTIONALLY DEMEANED component `y_{i,h}(t) = c_{i,h}(t) -
mean_j c_{j,h}(t)`, pooled across the 8 assets of U8, on W_TRAIN
(2020-04-01 -> 2021-12-31, 184,320 bars) and NOTHING ELSE. Demeaned
because a long-only cross-sectional portfolio only ever consumes the
component's cross-sectional dispersion; the common level is not tradeable
by a rank rule. The continuous-time decay rate is `phi = -ln(rho)` per
5-minute bar; half-life is `ln 2 / phi`.

    h = 20d:  rho = 0.99927955   phi = 7.2071e-04 /bar   half-life  3.34 d
    h = 40d:  rho = 0.99958701   phi = 4.1308e-04 /bar   half-life  5.83 d
    h = 80d:  rho = 0.99974712   phi = 2.5291e-04 /bar   half-life  9.52 d

(The raw, non-demeaned components are slower -- 5.55 / 9.57 / 15.49 days --
and are printed by `fit` for the record but are not used.)

The half-lives are monotone in the horizon and roughly proportional to it,
which is what a rolling-mean anchor should produce, so the fit is not
obviously an artifact.

**IN-SAMPLE STATUS, STATED PLAINLY.** These three numbers are fitted on
W_TRAIN and then used on W_TRAIN, W_VAL and W_FULL6 (which CONTAINS
W_TRAIN). That is in-sample by construction. It is accepted here because
`phi_h` is a SHAPE parameter of the signal, not a performance-selected
one: no candidate return was consulted in fitting it, and the fit is a
single unregularized OLS with no free choices. It is NOT a full-series
fit: it never sees W_VAL or W_HOLD, and it is frozen as a module constant
so that `build_targets` is a pure causal function of the price prefix.
This paragraph is the disclosure the round asked for; it belongs in the
ledger.

=====================================================================
THE DERIVATION OF THE TRADING RATE `a`  (derived, NOT fitted)
=====================================================================

Assumptions, each stated so a skeptic can redo or reject it:

  (A1) GP's continuous-time closed form. With risk penalty
       `(gamma/2) x' Sigma x` per unit time and quadratic trading cost
       `(1/2) dx' (lambda_S Sigma) dx`, GP (2016) give the trading rate
       `a = sqrt(gamma / lambda_S)` and the aim shrinkage
       `a/(a + phi_j)` used above. Working in the scalar (one-risk-unit)
       normalization, `lambda_S * sigma^2 = lambda` where `lambda` is the
       cost coefficient in RETURN units per squared weight per bar, so
       `a = sigma * sqrt(gamma / lambda)`.

  (A2) OUR COSTS ARE LINEAR, NOT QUADRATIC. This is the honest wrinkle
       and it is stated rather than hidden: a 10bps proportional taker fee
       is `f * |Delta|`, not `(lambda/2) Delta^2`. GP's closed form does
       not apply to linear costs (the linear-cost problem has a no-trade
       *region*, which is what the CONSERVATIVE branch implements). The
       quadratic surrogate is calibrated to charge the SAME TOTAL COST as
       the linear fee at a reference per-bar trading amplitude
       `Delta*`: `(lambda/2) Delta*^2 = f Delta*`, hence
       `lambda = 2 f / Delta*`.

  (A3) `Delta*` = R-63's own measured per-bar turnover, 3.44 round-trip
       per day / 288 bars = 0.0119444 of equity per bar. Anchoring on the
       incumbent policy's turnover is circular in principle -- the optimal
       policy trades less, which would raise `lambda` and lower `a`. It is
       used anyway because it is the only turnover number this project has
       measured for this signal, the circularity is disclosed, and
       `a propto 1/sqrt(lambda)` so the sensitivity band x{0.25..4} spans a
       16-to-256-fold range in the implied `Delta*`. The marginal-cost
       calibration `lambda = f/Delta*` (rather than total-cost) would move
       `a` by exactly sqrt(2), well inside the band.

  (A4) `gamma = 1`. Log utility / full Kelly, whose per-period
       certainty-equivalent penalty is exactly `(1/2) x' Sigma x`. Chosen
       because this project's incumbent family is `kelly_regime_*` and
       because it requires no fitting. Any other `gamma` moves `a` by
       `sqrt(gamma)` and is inside the band for gamma in [1/16, 16].

  (A5) `sigma` = per-bar standard deviation of the EQUAL-WEIGHT ALL-N
       basket's log return on W_TRAIN, U8 = **0.00299076** (97.0%
       annualized). Measured on W_TRAIN only, same window as `phi`.

Plugging in:

    Delta* = 3.44 / 288                       = 0.01194444
    lambda = 2 * 0.001 / 0.01194444           = 0.16744186
    a      = 0.00299076 * sqrt(1 / 0.16744186) = 0.00730886 per bar

    => adjustment half-life ln2/a = 94.8 bars = 0.329 days
    => EWMA-equivalent span 2/a - 1 = 272 bars = 0.95 days

    GP aim weights at that `a`:  W_20 = 0.32240
                                 W_40 = 0.33525
                                 W_80 = 0.34235
    (equal weights would be 0.33333 each)

**A NOTE ON WHAT THAT LAST BLOCK MEANS, RECORDED BEFORE ANY RESULT.** The
derived `a` is FAST relative to all three decay rates (`a` is 10x to 29x
larger than `phi_h`), so the GP shrinkage factors `a/(a+phi_h)` are all
within 6% of 1 and the persistence weighting is very nearly a no-op at the
derived rate. GP part (b) can only bite when `a` is small relative to
`phi`, i.e. at the slow end of the band. This is a prediction of the
derivation, not a finding, and it is written down here so the
decomposition result below reads as arithmetic rather than as a surprise.

SENSITIVITY BAND, a PLATEAU CHECK and NOT a selection. `a_derived x
{0.25, 0.5, 1, 2, 4}` -- pre-specified. Two further frontier-only points
(`x 1/64`, `x 1/16`) and the no-smoothing endpoint (`a = 1`, trade fully
to the aim every bar) are added so the frontier actually reaches the
low-turnover end of the axis this round exists to measure. Every one of
them is reported. The FROZEN configuration is and stays `a = a_derived`,
`mode = "gp"`.

Run as:
    python3 experiments/r64_novel_aim_portfolio.py fit
    python3 experiments/r64_novel_aim_portfolio.py checks
    python3 experiments/r64_novel_aim_portfolio.py frontier
    python3 experiments/r64_novel_aim_portfolio.py run
    python3 experiments/r64_novel_aim_portfolio.py scramble
    python3 experiments/r64_novel_aim_portfolio.py all
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.r63_shared import check_causality  # noqa: E402
from experiments.r64_shared import (  # noqa: E402
    BARS_PER_DAY,
    D5_BAR,
    DEADBAND,
    HORIZONS,
    OUT_DIR,
    R63_NET_D1,
    R63_TURNOVER_PER_DAY,
    SCRAMBLE_SEEDS,
    SPOT_BASE,
    SPOT_FREE,
    SPOT_REAL,
    UNIVERSE_6,
    UNIVERSE_8,
    W_FULL6,
    W_TRAIN,
    W_VAL,
    align_frames,
    basket_log_returns,
    compare,
    conditional_vol_scale,
    config_count,
    cross_sectional_score,
    d1_pass,
    d2_pass,
    d3_pass,
    d5_pass,
    frontier_row,
    further_work,
    holding_period_days,
    load_universe,
    matched_hold_targets,
    mean_total_notional,
    r63_baseline_targets,
    realized_vol,
    scramble_targets,
    simulate_portfolio,
    static_hold_equity,
    turnover_stats,
    volmatched_hold_equity,
    warm_window,
)

# ------------------------------------------------------------------ frozen
# Measured on W_TRAIN ONLY (U8, 184,320 bars). `fit` recomputes them and
# asserts these constants still match, so they cannot silently drift.

PHI_PER_BAR = {20: 7.20706e-04, 40: 4.13077e-04, 80: 2.52912e-04}
HALF_LIFE_DAYS = {20: 3.3387, 40: 5.8255, 80: 9.5159}
RHO_PER_BAR = {20: 0.99927955, 40: 0.99958701, 80: 0.99974712}
SIGMA_BAR_TRAIN = 0.00299076  # per-bar sd of the EW U8 basket log return

FEE = 0.001
GAMMA = 1.0
DELTA_STAR = R63_TURNOVER_PER_DAY / BARS_PER_DAY  # 0.01194444 per bar
LAMBDA_QUAD = 2.0 * FEE / DELTA_STAR              # 0.16744186
A_DERIVED = SIGMA_BAR_TRAIN * math.sqrt(GAMMA / LAMBDA_QUAD)  # 0.00730886

# Pre-specified plateau band (a multiplier grid), and the frontier-only
# extension toward zero turnover. Both are reported in full.
BAND = (0.25, 0.5, 1.0, 2.0, 4.0)
FRONTIER_EXTRA = (1.0 / 64.0, 1.0 / 16.0)

W_FLOOR = 1e-3  # measurement hygiene; see the docstring. Not swept.

FROZEN_A = A_DERIVED
FROZEN_MODE = "gp"


def aim_weights(a: float, mode: str = "gp") -> dict[int, float]:
    """GP's persistence shrinkage `a/(a+phi_h)`, normalized to sum to 1.

    ``mode="equal"`` returns R-63's own equal weighting, which is what the
    GP(a)-vs-GP(b) decomposition holds `a` fixed against.
    """
    if mode == "equal":
        return {h: 1.0 / len(HORIZONS) for h in HORIZONS}
    if mode != "gp":
        raise ValueError(f"unknown weighting mode {mode!r}")
    raw = {h: a / (a + PHI_PER_BAR[h]) for h in HORIZONS}
    tot = sum(raw.values())
    return {h: v / tot for h, v in raw.items()}


# ------------------------------------------------------------------ signal


def horizon_components(aligned: dict[str, pd.DataFrame]) -> dict[int, np.ndarray]:
    """The three terms R-63 averages, kept SEPARATE.

    `c_{i,h}(t) = close_i(t)/anchor_{i,h}(t) - 1`, rolling means only, so
    row t uses rows <= t and nothing else.
    """
    assets = list(aligned.keys())
    out: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        cols = []
        for t in assets:
            close = aligned[t]["close"]
            anchor = close.rolling(int(h * BARS_PER_DAY)).mean()
            cols.append((close / anchor - 1.0).to_numpy(dtype=float))
        out[h] = np.column_stack(cols)
    return out


def _top1_onehot(c: np.ndarray) -> np.ndarray:
    """R-63's own selection rule on ONE component: the single highest-scoring
    asset that has a positive score; flat if none.

    A CROSS-SECTION -- argmax across columns within a row. Never a
    time-series quantile.
    """
    valid = np.isfinite(c)
    masked = np.where(valid, c, -np.inf)
    best = np.argmax(masked, axis=1)
    rows = np.arange(c.shape[0])
    out = np.zeros_like(c)
    live = valid[rows, best] & (masked[rows, best] > 0.0)
    out[rows[live], best[live]] = 1.0
    return out


def build_targets(aligned: dict[str, pd.DataFrame], a: float,
                  mode: str = "gp") -> pd.DataFrame:
    """Target weight matrix: partial adjustment toward the aim portfolio.

    Pure causal function of the price prefix and the frozen constants: no
    statistic of any kind is computed over the whole series here.
    """
    if not (0.0 < a <= 1.0):
        raise ValueError(f"trading rate a must be in (0, 1]; got {a}")
    assets = list(aligned.keys())
    idx = aligned[assets[0]].index
    comps = horizon_components(aligned)
    W = aim_weights(a, mode)

    aim = np.zeros((len(idx), len(assets)))
    for h in HORIZONS:
        aim += W[h] * _top1_onehot(comps[h])

    n, k = aim.shape
    x = np.zeros(k)
    X = np.zeros((n, k))
    started = False
    for i in range(n):
        row = aim[i]
        if not started:
            if row.any():
                x = row.copy()
                started = True
        else:
            x = x + a * (row - x)
        X[i] = x

    # The floor is applied to the OUTPUT, never fed back into the state. An
    # in-state floor deadlocks a slow arm: an entering asset grows by
    # a*W_h per bar, which for small `a` is below the floor, so it would be
    # zeroed every bar and could never accumulate. (First version of this
    # file did exactly that and produced a permanently flat arm at a/64 --
    # recorded here rather than silently fixed.)
    X[X < W_FLOOR] = 0.0
    s = X.sum(axis=1)
    scale = conditional_vol_scale(basket_log_returns(aligned))

    desired = scale * s
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        d = desired[i]
        if abs(d - cur) > DEADBAND:
            cur = d
        pos[i] = cur

    total = np.minimum(np.nan_to_num(pos, nan=0.0), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shape = np.where(s[:, None] > 1e-12, X / np.maximum(s, 1e-12)[:, None], 0.0)
    w = np.clip(total[:, None] * shape, 0.0, 1.0)
    return pd.DataFrame(w, index=idx, columns=assets)


# ------------------------------------------------------------------ cells


def _slice_index(warm: dict[str, pd.DataFrame], window):
    idx = next(iter(warm.values())).index
    idx = idx[idx >= pd.Timestamp(window[0], tz="UTC")]
    if window[1] is not None:
        # STRICT right-exclusive: r63_shared._hi's `end + 1 day` convention
        # would admit the following day's 00:00 bar (documented amendment in
        # that file). This branch never relies on it.
        idx = idx[idx < pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)]
    return idx


_WARM_CACHE: dict = {}


def _warm_frames(frames, universe, window):
    """`align_frames` is deterministic and expensive; cache it per cell.

    Pure memoization of a pure function -- it changes no number, it only
    stops the frontier realigning the same panel 34 times.
    """
    key = (tuple(universe), window)
    if key not in _WARM_CACHE:
        _WARM_CACHE[key] = align_frames({t: frames[t] for t in universe},
                                        warm_window(window))
    return _WARM_CACHE[key]


def build_cell(frames, universe, window, a, mode="gp", baseline=False):
    """Aligned prices + targets, both sliced to the evaluation window.

    ``baseline=True`` builds R-63's own frozen k=1 arm instead, through the
    identical warm-up/slice path, so the frontier's reference point is
    measured on exactly the same grid as every candidate row.
    """
    warm = _warm_frames(frames, universe, window)
    targets = r63_baseline_targets(warm, 1) if baseline else build_targets(warm, a, mode)
    idx = _slice_index(warm, window)
    score = cross_sectional_score(warm).loc[idx]
    first_warm = bool(np.isfinite(score.iloc[0].to_numpy()).all())
    aligned_eval = {t: df.loc[idx] for t, df in warm.items()}
    return aligned_eval, targets.loc[idx], first_warm


def evaluate(targets, aligned, assets, window_name, universe_name, arm, params):
    """One frontier row: 0.10% and 0 bps, both against VOLMATCH_HOLD."""
    out = {}
    cmps = {}
    for tag, market in (("net", SPOT_BASE), ("gross", SPOT_FREE)):
        cand = simulate_portfolio(targets, aligned, market)
        bench, c, vol, matched = volmatched_hold_equity(cand, aligned, assets, market)
        if bench is None:
            raise RuntimeError(f"{arm} {window_name}: volmatch produced no benchmark")
        cmps[tag] = compare(cand, bench)
        out[f"{tag}_volmatch_c"] = c
        out[f"{tag}_volmatch_vol"] = vol
        out[f"{tag}_volmatch_matched"] = matched
        out[f"{tag}_cand_vol"] = realized_vol(cand)
    row = frontier_row(arm, params, targets, cmps["net"], cmps["gross"],
                       "VOLMATCH_HOLD", window_name, universe_name, **out)
    return row


def write_csv(path, rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for key in r:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"  wrote {path}")
    return path


def fmt_front(r):
    return (f"    hold {r['hold_days']:7.3f}d | turn {r['turnover_per_day']:7.4f}/d"
            f" | mtn {r['mean_notional']:.3f}"
            f" | GROSS {r['gross_growth_diff']:+8.3f}"
            f" [{r['gross_growth_lo']:+7.3f},{r['gross_growth_hi']:+7.3f}]"
            f" | NET {r['net_growth_diff']:+8.3f}"
            f" [{r['net_growth_lo']:+7.3f},{r['net_growth_hi']:+7.3f}]")


# ------------------------------------------------------------------ fit


def cmd_fit(frames):
    """Recompute the frozen decay constants on W_TRAIN and assert they match."""
    print("== decay fit: W_TRAIN ONLY, U8, AR(1) through the origin ==")
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    idx = _slice_index(warm, W_TRAIN)
    print(f"  fit index: {len(idx):,} bars  {idx[0]} -> {idx[-1]}")
    assert idx[-1] < pd.Timestamp("2022-01-01", tz="UTC"), "fit window leaked past W_TRAIN"

    comps = horizon_components(warm)
    full_idx = next(iter(warm.values())).index
    lo = int(full_idx.searchsorted(idx[0], side="left"))
    hi = int(full_idx.searchsorted(idx[-1], side="right"))
    assert hi - lo == len(idx), "fit slice is not contiguous with the warm index"

    rows = []
    ok = True
    for h in HORIZONS:
        c = comps[h][lo:hi]
        y = c - np.nanmean(c, axis=1, keepdims=True)     # CROSS-SECTIONAL demean
        a_, b_ = y[:-1].ravel(), y[1:].ravel()
        m = np.isfinite(a_) & np.isfinite(b_)
        rho = float((a_[m] * b_[m]).sum() / (a_[m] * a_[m]).sum())
        phi = -math.log(rho)
        hl = math.log(2) / phi / BARS_PER_DAY
        ar, br = c[:-1].ravel(), c[1:].ravel()
        mr = np.isfinite(ar) & np.isfinite(br)
        rho_raw = float((ar[mr] * br[mr]).sum() / (ar[mr] * ar[mr]).sum())
        hl_raw = math.log(2) / -math.log(rho_raw) / BARS_PER_DAY
        drift = abs(phi - PHI_PER_BAR[h]) / PHI_PER_BAR[h]
        ok &= drift < 1e-3
        print(f"  h={h:3d}d  rho={rho:.8f}  phi={phi:.5e}/bar  half-life={hl:7.4f}d"
              f"   (raw, unused: half-life {hl_raw:6.3f}d)   frozen-drift {drift:.2e}")
        rows.append({"horizon_days": h, "rho_per_bar": rho, "phi_per_bar": phi,
                     "half_life_days": hl, "rho_raw": rho_raw,
                     "half_life_days_raw": hl_raw, "n_obs": int(m.sum()),
                     "frozen_phi": PHI_PER_BAR[h], "frozen_drift": drift})

    r = basket_log_returns(warm).loc[idx].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    sigma = float(np.std(r, ddof=1))
    sdrift = abs(sigma - SIGMA_BAR_TRAIN) / SIGMA_BAR_TRAIN
    ok &= sdrift < 1e-3
    print(f"  sigma (per-bar EW U8 basket sd) = {sigma:.8f}  "
          f"({sigma * math.sqrt(365.25 * BARS_PER_DAY) * 100:.1f}% ann)  "
          f"frozen-drift {sdrift:.2e}")

    print("\n  -- derivation --")
    print(f"  Delta* = R63 turnover/day / 288 = {R63_TURNOVER_PER_DAY}/288 = {DELTA_STAR:.8f}")
    print(f"  lambda = 2f/Delta*             = 2*{FEE}/{DELTA_STAR:.8f} = {LAMBDA_QUAD:.8f}")
    print(f"  a      = sigma*sqrt(gamma/lambda) = {SIGMA_BAR_TRAIN:.8f}"
          f"*sqrt({GAMMA}/{LAMBDA_QUAD:.6f}) = {A_DERIVED:.8f} /bar")
    print(f"  adjustment half-life = ln2/a = {math.log(2)/A_DERIVED:.1f} bars = "
          f"{math.log(2)/A_DERIVED/BARS_PER_DAY:.3f} days")
    for mult in FRONTIER_EXTRA + BAND + (1.0 / A_DERIVED,):
        a = min(A_DERIVED * mult, 1.0)
        W = aim_weights(a, "gp")
        tag = "FROZEN" if abs(mult - 1.0) < 1e-12 else ""
        print(f"  a={a:.6f} (x{mult:>8.5f})  W20={W[20]:.5f} W40={W[40]:.5f} "
              f"W80={W[80]:.5f}  {tag}")

    rows.append({"horizon_days": -1, "sigma_bar_train": sigma,
                 "delta_star": DELTA_STAR, "lambda_quad": LAMBDA_QUAD,
                 "gamma": GAMMA, "fee": FEE, "a_derived": A_DERIVED,
                 "adjust_half_life_days": math.log(2) / A_DERIVED / BARS_PER_DAY})
    write_csv(OUT_DIR / "novel_decay_fit.csv", rows)
    print(f"  frozen constants reproduce: {ok}")
    return ok


# ------------------------------------------------------------------ checks


def cmd_checks(frames):
    print("== correctness gates ==")
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    n = len(next(iter(warm.values())))

    # 1. truncation test at 60% of bars (the round's explicit instruction)
    cut = int(n * 0.6)
    full = build_targets(warm, FROZEN_A, FROZEN_MODE)
    trunc = build_targets({t: df.iloc[:cut] for t, df in warm.items()},
                          FROZEN_A, FROZEN_MODE)
    a60 = np.nan_to_num(full.iloc[:cut].to_numpy(), nan=0.0)
    b60 = np.nan_to_num(trunc.to_numpy(), nan=0.0)
    exact60 = bool(np.array_equal(a60, b60))
    print(f"  truncation @60% ({cut:,}/{n:,} bars) EXACT: {exact60}"
          f"  max|diff|={float(np.abs(a60 - b60).max()):.3e}")

    # 2. r63_shared's own truncation probe, at the derived rate and at the
    #    slowest frontier rate (the recursion is stateful; a slow rate has a
    #    longer memory and is the harder case).
    c1 = check_causality(lambda al: build_targets(al, FROZEN_A, FROZEN_MODE), warm)
    c2 = check_causality(
        lambda al: build_targets(al, A_DERIVED / 64.0, FROZEN_MODE), warm)
    print(f"  check_causality(a=derived): {c1}")
    print(f"  check_causality(a=derived/64): {c2}")

    # 3. perturbation probe: corrupt the TAIL x10, early rows must not move.
    cutp = int(n * 0.6)
    bad = {}
    for t, df in warm.items():
        d = df.copy()
        for col in ("open", "high", "low", "close"):
            v = d[col].to_numpy(dtype=float).copy()
            v[cutp:] *= 10.0
            d[col] = v
        bad[t] = d
    pa = np.nan_to_num(build_targets(warm, FROZEN_A).to_numpy()[:cutp], nan=0.0)
    pb = np.nan_to_num(build_targets(bad, FROZEN_A).to_numpy()[:cutp], nan=0.0)
    probe = bool(np.allclose(pa, pb, atol=1e-12, rtol=0.0))
    print(f"  perturbation probe (tail x10, early rows unchanged): {probe}")

    # 4. does `a` actually control holding period / turnover?
    print("  turnover response to `a` (W_TRAIN, U8):")
    idx = _slice_index(warm, W_TRAIN)
    resp = []
    for mult in FRONTIER_EXTRA + BAND + (1.0 / A_DERIVED,):
        a = min(A_DERIVED * mult, 1.0)
        tg = build_targets(warm, a, FROZEN_MODE).loc[idx]
        ts = turnover_stats(tg)
        hd = holding_period_days(tg)
        resp.append(ts["turnover_per_day"])
        print(f"    a={a:.6f} (x{mult:>8.5f})  turnover {ts['turnover_per_day']:8.4f}/d"
              f"  rebalances {ts['rebalances_per_day']:8.3f}/d  hold {hd:8.3f}d"
              f"  mtn {mean_total_notional(tg):.3f}")
    monotone = (all(resp[i] <= resp[i + 1] for i in range(len(resp) - 1))
                and resp[-1] > resp[0])
    print(f"  turnover non-decreasing in `a` and strictly higher end-to-end: "
          f"{monotone}")

    # 5. cross-section identity: our components average to R-63's score.
    comps = horizon_components(warm)
    avg = sum(comps[h] for h in HORIZONS) / len(HORIZONS)
    r63 = cross_sectional_score(warm).to_numpy(dtype=float)
    same = bool(np.allclose(np.nan_to_num(avg), np.nan_to_num(r63),
                            atol=1e-12, rtol=0.0))
    print(f"  mean of our 3 components == R-63's cross_sectional_score: {same}")

    return bool(exact60 and c1 and c2 and probe and monotone and same)


# ------------------------------------------------------------------ frontier


def cmd_frontier(frames):
    """The round's deliverable: value and cost on the same points.

    W_TRAIN and W_VAL only (U8). The decision window is NOT touched here.
    """
    print("== FRONTIER: W_TRAIN and W_VAL, U8, vs VOLMATCH_HOLD at 0.10% and 0bps ==")
    mults = tuple(sorted(set(FRONTIER_EXTRA + BAND)))
    rows = []
    for wname, window in (("W_TRAIN", W_TRAIN), ("W_VAL", W_VAL)):
        print(f"  -- {wname} --")
        # R-63's reference point, through this branch's own reporting.
        aligned, tg, warm_ok = build_cell(frames, UNIVERSE_8, window, 1.0,
                                          baseline=True)
        r = evaluate(tg, aligned, UNIVERSE_8, wname, "U8", "R63_BASELINE_k1",
                     {"a": float("nan"), "mult": float("nan"), "mode": "r63_equal_rank"})
        r["kind"] = "reference"
        r["first_bar_warm"] = warm_ok
        rows.append(r)
        print("  R63_BASELINE_k1")
        print(fmt_front(r))

        for mode in ("gp", "equal"):
            for mult in mults + (1.0 / A_DERIVED,):
                a = min(A_DERIVED * mult, 1.0)
                aligned, tg, warm_ok = build_cell(frames, UNIVERSE_8, window, a, mode)
                kind = ("frozen" if (mode == "gp" and abs(mult - 1.0) < 1e-12)
                        else ("band" if (mode == "gp" and mult in BAND)
                              else ("decomposition" if mode == "equal"
                                    else "frontier_extension")))
                if a >= 1.0:
                    kind = "no_smoothing_endpoint" if mode == "gp" else "decomposition"
                r = evaluate(tg, aligned, UNIVERSE_8, wname, "U8",
                             f"aim_{mode}", {"a": a, "mult": mult, "mode": mode})
                r["kind"] = kind
                r["first_bar_warm"] = warm_ok
                rows.append(r)
                print(f"  aim_{mode} a={a:.6f} (x{mult:.5f}) [{kind}]")
                print(fmt_front(r))

    write_csv(OUT_DIR / "novel_frontier.csv", rows)
    return rows


# ------------------------------------------------------------------ run


def cmd_run(frames, a=None, mode=None):
    a = FROZEN_A if a is None else a
    mode = FROZEN_MODE if mode is None else mode
    print(f"== DECISION CELLS: frozen a={a:.8f}, mode={mode} ==")
    rows = []

    aligned, targets, warm_ok = build_cell(frames, UNIVERSE_6, W_FULL6, a, mode)
    print(f"  W_FULL6 bars {len(targets):,}  {targets.index[0]} -> {targets.index[-1]}")
    print(f"  first evaluated bar warm for every asset: {warm_ok}")
    if not warm_ok:
        raise RuntimeError("W_FULL6 first evaluated bar not warm")

    # ---- SUBSTRATE REPRODUCTION: R-63's own D1 cell, its own benchmark ----
    # R-63 published turnover 3.44/day and net growth -7.537 vs MATCHED_HOLD
    # on this exact cell. If these do not come back, the substrate drifted
    # and nothing else in this file extends the number it claims to extend.
    _, rt, rwarm = build_cell(frames, UNIVERSE_6, W_FULL6, 1.0, baseline=True)
    rts = turnover_stats(rt)
    rc = mean_total_notional(rt)
    r_cand = simulate_portfolio(rt, aligned, SPOT_BASE)
    r_mh = simulate_portfolio(matched_hold_targets(rt.index, UNIVERSE_6, rc),
                              aligned, SPOT_BASE)
    r_cmp = compare(r_cand, r_mh)
    r_free = simulate_portfolio(rt, aligned, SPOT_FREE)
    r_mh_free = simulate_portfolio(matched_hold_targets(rt.index, UNIVERSE_6, rc),
                                   aligned, SPOT_FREE)
    r_gross = compare(r_free, r_mh_free)["growth_diff"]
    print("  [R-63 REFERENCE REPRODUCTION, W_FULL6 U6 vs MATCHED_HOLD]")
    print(f"    turnover {rts['turnover_per_day']:.4f}/d (R-63 published "
          f"{R63_TURNOVER_PER_DAY}; deadband-aware measure here)")
    print(f"    net growth_diff {r_cmp['growth_diff']:+.4f} (R-63 published "
          f"{R63_NET_D1:+.3f})   gross {r_gross:+.4f} (R-63 published +0.480)")
    print(f"    cand_final {r_cmp['cand_final']:,.4f} (R-63 published 1.4419)  "
          f"mtn {rc:.4f} (R-63 published 0.5249)  bars {len(rt):,} (R-63 671,271)")
    rows.append({"arm": "R63_BASELINE_k1", "window": "W_FULL6", "universe": "U6",
                 "bench": "MATCHED_HOLD", "kind": "substrate reproduction",
                 "turnover_per_day": rts["turnover_per_day"],
                 "hold_days": holding_period_days(rt), "mean_notional": rc,
                 "net_growth_diff": r_cmp["growth_diff"],
                 "net_growth_lo": r_cmp["growth_lo"],
                 "net_growth_hi": r_cmp["growth_hi"],
                 "net_dd_diff": r_cmp["dd_diff"], "net_dd_lo": r_cmp["dd_lo"],
                 "net_dd_hi": r_cmp["dd_hi"], "gross_growth_diff": r_gross,
                 "cand_final": r_cmp["cand_final"],
                 "bench_final": r_cmp["bench_final"], "cand_dd": r_cmp["cand_dd"],
                 "bench_dd": r_cmp["bench_dd"], "n_days": r_cmp["n_days"],
                 "first_bar_warm": rwarm,
                 "r63_published_turnover": R63_TURNOVER_PER_DAY,
                 "r63_published_net": R63_NET_D1, "r63_published_gross": 0.480})

    d12 = evaluate(targets, aligned, UNIVERSE_6, "W_FULL6", "U6",
                   f"aim_{mode}", {"a": a, "mode": mode})
    d12["kind"] = "D1/D2/D5 primary"
    d12["first_bar_warm"] = warm_ok
    print(fmt_front(d12))
    print(f"    volmatch matched: net={d12['net_volmatch_matched']} "
          f"(c={d12['net_volmatch_c']:.3f}, bench vol {d12['net_volmatch_vol']:.3f} "
          f"vs cand {d12['net_cand_vol']:.3f}) | "
          f"gross={d12['gross_volmatch_matched']} (c={d12['gross_volmatch_c']:.3f})")

    if d12["net_volmatch_matched"]:
        d1, d2 = d1_pass(d12), d2_pass(d12)
    else:
        d1 = d2 = False
        print("    !! VOLMATCH did not match at 0.10% -- D1/D2 are VOIDED, not scored")
    d5 = d5_pass(d12) if d12["gross_volmatch_matched"] else False
    if not d12["gross_volmatch_matched"]:
        print("    !! VOLMATCH did not match at 0 bps -- D5 is VOIDED, not scored")
    d12["d1_pass"] = d1
    d12["d2_pass"] = d2
    d12["d5_pass"] = d5
    rows.append(d12)
    print(f"    D1 PASS={d1}  D2 PASS={d2}  D5 PASS={d5} "
          f"(gross {d12['gross_growth_diff']:+.3f} vs bar {D5_BAR:+.3f})")

    # Continuity / context benchmarks on the same cell.
    cand = simulate_portfolio(targets, aligned, SPOT_BASE)
    c = mean_total_notional(targets)
    mh = simulate_portfolio(matched_hold_targets(targets.index, UNIVERSE_6, c),
                            aligned, SPOT_BASE)
    ew = static_hold_equity(aligned, UNIVERSE_6, SPOT_BASE)
    btc = frames["BTC"]
    btc_on = btc.reindex(btc.index.union(targets.index)).ffill().reindex(targets.index)
    btc_eq = static_hold_equity({"BTC": btc_on}, ["BTC"], SPOT_BASE)
    for label, bench in (("MATCHED_HOLD", mh), ("EW_HOLD", ew), ("BTC_HOLD", btc_eq)):
        cm = compare(cand, bench)
        row = {"arm": f"aim_{mode}", "window": "W_FULL6", "universe": "U6",
               "bench": label, "kind": "context", "p_a": a, "p_mode": mode,
               "mean_notional": c, "hold_days": holding_period_days(targets),
               "turnover_per_day": turnover_stats(targets)["turnover_per_day"],
               "net_growth_diff": cm["growth_diff"], "net_growth_lo": cm["growth_lo"],
               "net_growth_hi": cm["growth_hi"], "net_dd_diff": cm["dd_diff"],
               "net_dd_lo": cm["dd_lo"], "net_dd_hi": cm["dd_hi"],
               "cand_final": cm["cand_final"], "bench_final": cm["bench_final"],
               "cand_dd": cm["cand_dd"], "bench_dd": cm["bench_dd"],
               "n_days": cm["n_days"]}
        rows.append(row)
        print(f"  [context vs {label}] cand {cm['cand_final']:,.1f} vs "
              f"{cm['bench_final']:,.1f} | growth {cm['growth_diff']:+.3f} "
              f"[{cm['growth_lo']:+.3f},{cm['growth_hi']:+.3f}] | dd "
              f"{cm['cand_dd']:.1f}% vs {cm['bench_dd']:.1f}%")

    # D4: W_FULL6 at 0.40%, beats EW_HOLD outright.
    cand40 = simulate_portfolio(targets, aligned, SPOT_REAL)
    ew40 = static_hold_equity(aligned, UNIVERSE_6, SPOT_REAL)
    d4 = bool(cand40.iloc[-1] > ew40.iloc[-1])
    rows.append({"arm": f"aim_{mode}", "window": "W_FULL6", "universe": "U6",
                 "bench": "EW_HOLD", "kind": "D4 @0.40%", "p_a": a, "p_mode": mode,
                 "cand_final": float(cand40.iloc[-1]),
                 "bench_final": float(ew40.iloc[-1]), "d4_pass": d4})
    print(f"  [D4 @0.40%] cand {cand40.iloc[-1]:,.1f} vs EW_HOLD "
          f"{ew40.iloc[-1]:,.1f} -> D4 PASS={d4}")

    # D3: W_VAL on U8.
    aligned3, targets3, warm3 = build_cell(frames, UNIVERSE_8, W_VAL, a, mode)
    if not warm3:
        raise RuntimeError("W_VAL first evaluated bar not warm")
    d3row = evaluate(targets3, aligned3, UNIVERSE_8, "W_VAL", "U8",
                     f"aim_{mode}", {"a": a, "mode": mode})
    d3row["kind"] = "D3 inner-validation"
    d3row["first_bar_warm"] = warm3
    print("  [D3] W_VAL U8")
    print(fmt_front(d3row))
    print(f"    volmatch matched: {d3row['net_volmatch_matched']}")
    d3 = d3_pass(d3row) if d3row["net_volmatch_matched"] else False
    if not d3row["net_volmatch_matched"]:
        print("    !! VOLMATCH did not match on W_VAL -- D3 is VOIDED, not scored")
    d3row["d3_pass"] = d3
    rows.append(d3row)
    print(f"    D3 PASS={d3}  (growth {d3row['net_growth_diff']:+.3f}, "
          f"dd {d3row['net_dd_diff']:+.3f})")

    write_csv(OUT_DIR / "novel_cells.csv", rows)
    return {"d1": d1, "d2": d2, "d3": d3, "d5": d5, "d4": d4,
            "d1_row": d12, "targets": targets, "aligned": aligned, "a": a,
            "mode": mode}


# ------------------------------------------------------------------ scramble


def cmd_scramble(frames, state=None, a=None, mode=None):
    a = FROZEN_A if a is None else a
    mode = FROZEN_MODE if mode is None else mode
    print(f"== FALSIFICATION: cross-section scramble, seeds 0..9, D1 cell, "
          f"a={a:.8f} ==")
    if state is None:
        aligned, targets, _ = build_cell(frames, UNIVERSE_6, W_FULL6, a, mode)
        cand = simulate_portfolio(targets, aligned, SPOT_BASE)
        bench, c, vol, matched = volmatched_hold_equity(cand, aligned, UNIVERSE_6,
                                                        SPOT_BASE)
        real = compare(cand, bench)["growth_diff"]
    else:
        aligned, targets = state["aligned"], state["targets"]
        real = state["d1_row"]["net_growth_diff"]
        cand = simulate_portfolio(targets, aligned, SPOT_BASE)
        bench, c, vol, matched = volmatched_hold_equity(cand, aligned, UNIVERSE_6,
                                                        SPOT_BASE)

    rows, diffs = [], []
    for seed in SCRAMBLE_SEEDS:
        st = scramble_targets(targets, seed)
        eq = simulate_portfolio(st, aligned, SPOT_BASE)
        r = compare(eq, bench)
        diffs.append(r["growth_diff"])
        rows.append({"arm": f"aim_{mode}_scrambled", "seed": seed, "p_a": a,
                     "window": "W_FULL6", "universe": "U6", "fee": 0.001,
                     "bench": "VOLMATCH_HOLD",
                     "mean_notional": mean_total_notional(st),
                     "turnover_per_day": turnover_stats(st)["turnover_per_day"],
                     **{k: r[k] for k in ("cand_final", "bench_final", "cand_dd",
                                          "bench_dd", "growth_diff", "growth_lo",
                                          "growth_hi", "dd_diff", "dd_lo", "dd_hi",
                                          "n_days")}})
        print(f"  seed {seed}: growth_diff {r['growth_diff']:+.4f}  "
              f"final {r['cand_final']:>12,.2f}  dd {r['cand_dd']:5.1f}%")

    p90 = float(np.percentile(diffs, 90))
    survived = bool(real > p90)
    rows.append({"arm": f"aim_{mode}", "seed": -1, "p_a": a, "window": "W_FULL6",
                 "universe": "U6", "fee": 0.001, "bench": "VOLMATCH_HOLD",
                 "growth_diff": real, "scramble_p90": p90,
                 "scramble_survived": survived,
                 "mean_notional": mean_total_notional(targets),
                 "turnover_per_day": turnover_stats(targets)["turnover_per_day"]})
    print(f"  real growth_diff {real:+.4f} vs scramble p90 {p90:+.4f} -> "
          f"SURVIVED={survived}")
    print(f"  candidate turnover {turnover_stats(targets)['turnover_per_day']:.4f}/d "
          f"vs scramble mean {np.mean([r['turnover_per_day'] for r in rows[:-1]]):.4f}/d")

    # ---- ADDITIONAL DIAGNOSTIC CONTROL (not the pre-registered one) ----
    # `scramble_targets` redraws a permutation on every bar whose target
    # vector changed. R-63's arm changed its vector rarely, so the control
    # preserved its turnover. THIS arm changes its vector on almost every
    # bar by construction, so the pre-registered control redraws almost
    # every bar and trades far more than the candidate -- it is charged
    # fees the candidate never pays, which biases the control IN THE
    # CANDIDATE'S FAVOUR. Reported, not fixed in the frozen file.
    #
    # The fair analogue for a continuous arm is a SINGLE fixed relabeling
    # of assets held for the whole run: it destroys the asset->signal
    # assignment exactly as thoroughly and preserves turnover and total
    # notional bar-for-bar (a column permutation is an L1 isometry).
    print("  -- diagnostic: fixed-permutation scramble (turnover-preserving) --")
    fdiffs = []
    cols = list(targets.columns)
    for seed in SCRAMBLE_SEEDS:
        perm = np.random.default_rng(1000 + seed).permutation(len(cols))
        st = pd.DataFrame(targets.to_numpy()[:, perm], index=targets.index,
                          columns=cols)
        eq = simulate_portfolio(st, aligned, SPOT_BASE)
        r = compare(eq, bench)
        fdiffs.append(r["growth_diff"])
        ident = bool(np.array_equal(perm, np.arange(len(cols))))
        rows.append({"arm": f"aim_{mode}_fixedperm", "seed": seed, "p_a": a,
                     "window": "W_FULL6", "universe": "U6", "fee": 0.001,
                     "bench": "VOLMATCH_HOLD", "perm": ";".join(map(str, perm)),
                     "identity_perm": ident,
                     "mean_notional": mean_total_notional(st),
                     "turnover_per_day": turnover_stats(st)["turnover_per_day"],
                     **{k: r[k] for k in ("cand_final", "bench_final", "cand_dd",
                                          "bench_dd", "growth_diff", "growth_lo",
                                          "growth_hi", "dd_diff", "dd_lo", "dd_hi",
                                          "n_days")}})
        print(f"    seed {seed} perm {perm}: growth_diff {r['growth_diff']:+.4f}"
              f"  turnover {rows[-1]['turnover_per_day']:.4f}/d"
              f"{'  (IDENTITY)' if ident else ''}")
    fp90 = float(np.percentile(fdiffs, 90))
    fsurv = bool(real > fp90)
    rows.append({"arm": f"aim_{mode}", "seed": -2, "p_a": a, "window": "W_FULL6",
                 "universe": "U6", "fee": 0.001, "bench": "VOLMATCH_HOLD",
                 "growth_diff": real, "scramble_p90": fp90,
                 "scramble_survived": fsurv,
                 "perm": "fixed-permutation diagnostic control"})
    print(f"    real {real:+.4f} vs fixed-perm p90 {fp90:+.4f} -> "
          f"survived={fsurv}  (DIAGNOSTIC, not the pre-registered cell)")

    write_csv(OUT_DIR / "novel_scramble.csv", rows)
    return survived


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "checks", "frontier", "run",
                                    "scramble", "cells", "all"])
    ap.add_argument("--a", type=float, default=None)
    ap.add_argument("--mode", default=None)
    args = ap.parse_args()

    frames = load_universe(UNIVERSE_8)

    if args.cmd == "fit":
        cmd_fit(frames)
    elif args.cmd == "checks":
        cmd_checks(frames)
    elif args.cmd == "frontier":
        cmd_frontier(frames)
    elif args.cmd == "run":
        cmd_run(frames, args.a, args.mode)
    elif args.cmd == "scramble":
        cmd_scramble(frames, None, args.a, args.mode)
    elif args.cmd == "cells":
        st = cmd_run(frames, args.a, args.mode)
        surv = cmd_scramble(frames, st)
        fw = further_work(st["d1"], st["d2"], st["d3"], st["d5"], surv)
        print(f"\n== further_work(d1={st['d1']}, d2={st['d2']}, d3={st['d3']}, "
              f"d5={st['d5']}, scramble={surv}) = {fw} ==")
        print(f"   (D4 @0.40% = {st['d4']}, reported but not part of the bar)")
        print("  -> W_HOLD is NOT read by this branch under any outcome.")
    else:
        cmd_fit(frames)
        gates = cmd_checks(frames)
        print(f"\n  correctness gates all pass: {gates}")
        if not gates:
            raise SystemExit("correctness gates FAILED -- no number below is believable")
        cmd_frontier(frames)
        st = cmd_run(frames, args.a, args.mode)
        surv = cmd_scramble(frames, st)
        fw = further_work(st["d1"], st["d2"], st["d3"], st["d5"], surv)
        print(f"\n== further_work(d1={st['d1']}, d2={st['d2']}, d3={st['d3']}, "
              f"d5={st['d5']}, scramble={surv}) = {fw} ==")
        print(f"   (D4 @0.40% = {st['d4']}, reported but not part of the bar)")
        if fw:
            print("  -> STOP. Report to the operator; the holdout read is theirs.")
        else:
            print("  -> DONE. W_HOLD is NOT read.")

    print(f"\nconfig_count() = {config_count()}")


if __name__ == "__main__":
    main()
