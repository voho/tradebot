"""R-119 NOVEL branch: robust selection via Monte Carlo draws from a 3-state
regime-switching jump-diffusion model calibrated from EXTERNAL LITERATURE,
not from this project's own price file.

See `experiments/r119_shared.py`'s module docstring for the full
pre-registration (direction, literature citations, "not a duplicate of"
list, and the pre-registered expected outcome for this specific branch).
This file implements only the novel branch's one novel piece: a
`path_generator(seed)` that draws synthetic OHLCV paths from a 3-state
(bull/chop/bear) Markov-switching jump-diffusion construction whose
parameters come from `r119_shared.py`'s frozen external constants (jump
rate/size, bear drawdown/duration) plus disclosed, round-number choices for
everything the frozen constants don't cover -- mirroring the SHAPE of
`experiments/r118_novel_regimeswitch_calibration.py`'s fitted model, but
never touching `load_inner_train_btc()` for drift, vol-per-regime, or
transition structure the way R-118 did.

CALIBRATION SOURCE OF EVERY PARAMETER (disclosed in full before any
simulation ran or any grid point was scored):

1. BEAR STATE drift -- 100% externally sourced, frozen in `r119_shared.py`.
   `EXT_BEAR_DRAWDOWN = 0.775` (mean of two crash-catalogue headline
   figures) and `EXT_BEAR_DURATION_DAYS = 365` (the commonly-cited
   "crypto winter" phase length) translate to a per-day bear log-drift by
   requiring the cumulative price fall by `EXT_BEAR_DRAWDOWN` on average
   over the phase:
       bear_drift_per_day = ln(1 - EXT_BEAR_DRAWDOWN) / EXT_BEAR_DURATION_DAYS
                           = ln(0.225) / 365 = -0.0040867 (log-return/day)
   Annualized for reporting: -0.0040867 * 365.25 = -1.4927/yr (log), i.e. a
   simple-return equivalent of exp(-1.4927) - 1 = -77.5%/yr -- consistent
   by construction with the 365-day, 77.5%-decline inputs.

2. BULL and CHOP state drift -- round-number, disclosed BEFORE looking at
   any simulation output, since no single figure in `r119_shared.py` is as
   authoritative for the "up" and "sideways" legs as the crash catalogue is
   for the bear leg:
     - CHOP drift = 0.0 exactly (round number: the "no informative timing
       edge" convention this project already uses in
       `tradebot.data.generate_synthetic_pair`, whose own hand-set chop
       state also carries 0.0 annualized drift).
     - BULL drift is set by a single disclosed RULE, not a free number:
       given round-number state DURATIONS (below) and the frozen bear
       drift, bull drift is whatever value makes one full cycle's net log
       return approximately zero (chop already contributes ~0, so bull
       must offset bear alone) -- the same "no expected edge from timing
       alone" convention `generate_synthetic_pair` uses for its own
       roughly-offsetting +2.2/-2.0 annualized bull/bear drift pair, made
       exact here rather than eyeballed:
           bull_drift_per_day = -bear_drift_per_day * BEAR_DAYS / BULL_DAYS
                               = 0.0020434 (log-return/day)
       Annualized: +0.7463/yr log (simple-return equivalent +110.9%/yr).
       This is a strong bull leg, but it is the mechanical consequence of
       one disclosed rule (cycle-neutrality) plus round-number durations,
       not a number tuned to produce any particular selection outcome.

3. STATE DURATIONS / cycle structure -- external anchor is crypto's
   commonly-cited ~4-year halving-driven cycle; round numbers, chosen
   before any simulation output was seen:
     - BEAR phase: 365 days (`EXT_BEAR_DURATION_DAYS`, frozen).
     - BULL phase: 730 days (2 years -- round number: "the bull leg runs
       about twice as long as the crash phase within a ~4-year cycle," a
       standard halving-cycle narrative -- post-halving rally, distribution/
       chop, crash, repeat).
     - CHOP phase: 366 days (the remainder needed to land the full cycle
       on almost exactly 4 calendar years: 365 + 730 + 366 = 1461 days =
       4 x 365.25).
   Transitions are a simple, disclosed DETERMINISTIC CYCLE (bull -> chop ->
   bear -> bull -> ...), not a transition matrix counted from real data:
   each day, state k is left with probability 1/duration_k (a geometric
   sojourn time with the stated mean) and always moves to the NEXT state in
   the fixed cycle order on leaving -- the simplest Markov structure that
   reproduces "mean state durations sized so a full cycle averages ~4
   years, with a 365-day bear phase inside it" without counting anything
   from real data. The initial day's state is drawn with probability
   proportional to each state's mean duration (the natural stationary-like
   weighting for a cyclical semi-Markov chain). NOTE: this project's own
   `load_inner_train_btc()` window is exactly 1,461 days (420,768 bars /
   288 bars-per-day) -- i.e. exactly one full external ~4-year cycle as
   constructed here, an intentional match to keep every synthetic path the
   same duration as R-118's, not a coincidence discovered after the fact
   (durations were fixed by the formula above, in round numbers, before any
   path was drawn).

4. VOLATILITY -- externally-anchored for bull/chop via the ONE quantity
   this branch estimates from this project's own training data (disclosed
   here, exactly as `r119_shared.py`'s docstring anticipates): pooled,
   UNCONDITIONAL daily realized vol from `load_inner_train_btc()` (a single
   scalar -- not per-regime, not a drift or transition estimate):
       RV_d = sqrt(sum of that day's squared 5-min log returns)
       ORDINARY_VOL_ANNUAL = mean_d(RV_d) * sqrt(365.25)
   Computed once at import time: **ORDINARY_VOL_ANNUAL ~= 0.772 (77.2%/yr)**
   -- squarely inside the commonly-cited 60-100%/yr range for BTC's
   steady-state volatility, used as an internal, more-defensible floor
   than guessing a round external figure for "ordinary" (non-crash) vol.
   This ONE internally-sourced scalar is applied to BOTH the bull and chop
   states identically (disclosed simplification: this branch does not
   claim bull markets are quieter than chop markets, only that both are
   "ordinary," non-crash conditions). BEAR-state vol is then set via a
   disclosed, round-number, commonly-cited multiplier -- crash/bear-market
   periods for BTC are widely reported at roughly double steady-state
   vol (e.g., 2018's crash-year realized vol vs. 2019's range-bound vol) --
   `BEAR_VOL_MULT = 2.0`, giving BEAR_VOL_ANNUAL ~= 1.545 (154.5%/yr).

5. JUMP component -- 100% externally sourced, frozen in `r119_shared.py`,
   applied on top of whichever state (bull/chop/bear) is active that day,
   the same mechanism R-118's own jump overlay used (Bernoulli arrival +
   Normal-sized additive log-return shock), with literature-sourced
   parameters instead of fitted ones. `EXT_JUMP_PROB_PER_DAY` (~1/7, "one
   jump day per week," Scaillet et al. 2020) is applied at DAY granularity
   -- matching the literature figure's own unit -- rather than converted to
   a bar-level rate the way R-118's fitted `jump_rate` (already a per-bar
   quantity) was: each day is independently flagged with probability
   `EXT_JUMP_PROB_PER_DAY`; a flagged day gets exactly one jump, placed at
   one uniformly-random bar within that day, signed +/- with equal
   probability and sized from `EXT_JUMP_UP_MEAN/STD` or
   `EXT_JUMP_DOWN_MEAN/STD` (MDPI Mathematics 9(20) 2567, 2021).

WITHIN-BAR MICRO-NOISE: this construction does not add a separate noise
term on top of the regime structure -- the per-bar GBM innovation IS the
micro-noise: every 5-minute bar draws its own iid `Normal(0, 1)` shock,
scaled by that day's active-state annualized vol converted to bar-scale
(`sigma_bar * sqrt(dt_bar_years)`, `dt_bar_years = 300s / seconds_per_year`
via `Normal(0, 1)`), so consecutive bars within the same day are not
identical even though they share the same drift/vol regime. No additional
noise term is layered on top; disclosed here rather than adding an
unnecessary second free parameter.

QUANTITATIVE COMPARISON TO R-118's OWN FITTED BEAR STATE (computed once,
disclosed in the round-4 report, not used to re-tune anything here): R-118's
ledger entry describes its fitted high-vol/jump regime as "-65%/yr
annualised drift, 143%/yr vol" with a jump component "4-sigma threshold,
0.53% of bars." This branch's externally-anchored bear state is drift
-149.3%/yr (about 2.3x steeper -- HARSHER), vol 154.5%/yr (about 1.08x --
roughly COMPARABLE, marginally higher), and an implied jump-flagged-bar
fraction of EXT_JUMP_PROB_PER_DAY / 288 ~= 0.0496% of bars (about 1/11th
R-118's fitted rate -- MILDER on jump frequency, though each jump here can
still be large: mean magnitude 4.14-4.65%). Net: harsher sustained bear
drift, similar bear vol, materially rarer (but not smaller) jumps than
R-118's own fitted model.

None of items 1, 3 (the bear-phase duration and the deterministic-cycle
SHAPE anchor), or 5 was estimated from this project's own data. Item 4 uses
exactly one internally-sourced scalar, disclosed above and again in the
final report, exactly as `r119_shared.py`'s pre-registration anticipates.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.r119_shared import (
    BARS_PER_DAY,
    GRID,
    N_DRAWS,
    SPOT,
    V4_DEFAULT,
    EXT_BEAR_DRAWDOWN,
    EXT_BEAR_DURATION_DAYS,
    EXT_JUMP_PROB_PER_DAY,
    EXT_JUMP_UP_MEAN,
    EXT_JUMP_UP_STD,
    EXT_JUMP_DOWN_MEAN,
    EXT_JUMP_DOWN_STD,
    evaluate_candidate,
    load_inner_train_btc,
    print_report,
    score_on_path,
    select_config,
)

ROOT = Path(__file__).resolve().parents[1]


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ------------------------------------------------------------------------
# State definitions, durations, drift, vol -- see module docstring for the
# full calibration-source disclosure for every number below.
# ------------------------------------------------------------------------

BULL, CHOP, BEAR = 0, 1, 2
STATE_NAMES = {BULL: "bull", CHOP: "chop", BEAR: "bear"}
CYCLE_ORDER = (BULL, CHOP, BEAR)   # deterministic cyclical transition order

DAYS_PER_YEAR = 365.25
BARS_PER_YEAR = DAYS_PER_YEAR * BARS_PER_DAY
SECONDS_PER_YEAR = DAYS_PER_YEAR * 24 * 3600
START_PRICE = 10_000.0            # arbitrary; strategy scoring is scale-invariant (log returns)
START = "2017-01-01"              # matches load_inner_train_btc()'s own start, round's convention

# --- durations (days) ---
BEAR_DURATION_DAYS = EXT_BEAR_DURATION_DAYS          # 365, frozen external
BULL_DURATION_DAYS = 2.0 * BEAR_DURATION_DAYS        # 730, round number (see docstring item 3)
CHOP_DURATION_DAYS = 4.0 * DAYS_PER_YEAR - BEAR_DURATION_DAYS - BULL_DURATION_DAYS  # 366
STATE_DURATION_DAYS = {BULL: BULL_DURATION_DAYS, CHOP: CHOP_DURATION_DAYS, BEAR: BEAR_DURATION_DAYS}
CYCLE_DAYS = BULL_DURATION_DAYS + CHOP_DURATION_DAYS + BEAR_DURATION_DAYS
assert abs(CYCLE_DAYS - 4.0 * DAYS_PER_YEAR) < 1e-9

# --- drift (log-return per day), see docstring items 1-2 ---
BEAR_DRIFT_PER_DAY = np.log(1.0 - EXT_BEAR_DRAWDOWN) / EXT_BEAR_DURATION_DAYS
BULL_DRIFT_PER_DAY = -BEAR_DRIFT_PER_DAY * BEAR_DURATION_DAYS / BULL_DURATION_DAYS
CHOP_DRIFT_PER_DAY = 0.0
STATE_DRIFT_PER_DAY = {BULL: BULL_DRIFT_PER_DAY, CHOP: CHOP_DRIFT_PER_DAY, BEAR: BEAR_DRIFT_PER_DAY}
STATE_DRIFT_ANNUAL = {k: v * DAYS_PER_YEAR for k, v in STATE_DRIFT_PER_DAY.items()}


def _pooled_ordinary_vol_annual() -> float:
    """The ONE internally-sourced scalar this branch uses: pooled,
    unconditional daily realized vol from `load_inner_train_btc()`
    (2017-01-01..2020-12-31). Not a per-regime estimate, not a drift or
    transition estimate -- disclosed prominently in the module docstring."""
    df = load_inner_train_btc()
    close = df["close"].to_numpy(dtype=float)
    n_bars = len(close)
    assert n_bars % BARS_PER_DAY == 0, "expected a whole number of days"
    n_days = n_bars // BARS_PER_DAY
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    logret[0] = 0.0
    r_by_day = logret.reshape(n_days, BARS_PER_DAY)
    rv = np.sqrt(np.sum(r_by_day ** 2, axis=1))
    daily_vol = float(np.mean(rv))
    return daily_vol * np.sqrt(DAYS_PER_YEAR)


ORDINARY_VOL_ANNUAL = _pooled_ordinary_vol_annual()   # ~= 0.772 (the one internal scalar)
BEAR_VOL_MULT = 2.0                                    # round, commonly-cited crash-vol multiple
STATE_VOL_ANNUAL = {
    BULL: ORDINARY_VOL_ANNUAL,
    CHOP: ORDINARY_VOL_ANNUAL,
    BEAR: ORDINARY_VOL_ANNUAL * BEAR_VOL_MULT,
}

N_BARS_TARGET = len(load_inner_train_btc())
assert N_BARS_TARGET == 420_768, f"expected 420768 bars, got {N_BARS_TARGET}"
N_DAYS_TARGET = N_BARS_TARGET // BARS_PER_DAY
assert N_DAYS_TARGET == 1461


# ------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------

def _simulate_regime_days(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Day-level state sequence: deterministic cyclical order (bull -> chop
    -> bear -> bull -> ...), geometric sojourn per state with mean
    `STATE_DURATION_DAYS[state]` (leave-probability 1/duration each day),
    matching "mean state durations sized so a full cycle averages ~4 years,
    with a 365-day bear phase inside it" -- no transition matrix counted
    from real data anywhere in this function."""
    p_stay = {s: 1.0 - 1.0 / STATE_DURATION_DAYS[s] for s in CYCLE_ORDER}
    durations = np.array([STATE_DURATION_DAYS[s] for s in CYCLE_ORDER])
    init_p = durations / durations.sum()
    cycle_idx = int(rng.choice(len(CYCLE_ORDER), p=init_p))

    regime_day = np.empty(n_days, dtype=np.int64)
    u = rng.random(n_days)
    for t in range(n_days):
        state = CYCLE_ORDER[cycle_idx]
        regime_day[t] = state
        if u[t] > p_stay[state]:
            cycle_idx = (cycle_idx + 1) % len(CYCLE_ORDER)
    return regime_day


def simulate_path(n_bars: int, seed: int, start_price: float = START_PRICE,
                  start: str = START) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assert n_bars % BARS_PER_DAY == 0
    n_days = n_bars // BARS_PER_DAY

    regime_day = _simulate_regime_days(n_days, rng)
    mu_annual_lut = np.array([STATE_DRIFT_ANNUAL[s] for s in range(3)])
    sigma_annual_lut = np.array([STATE_VOL_ANNUAL[s] for s in range(3)])
    mu_day = mu_annual_lut[regime_day]
    sigma_day = sigma_annual_lut[regime_day]

    mu_bar = np.repeat(mu_day, BARS_PER_DAY)          # (n_bars,)
    sigma_bar = np.repeat(sigma_day, BARS_PER_DAY)    # (n_bars,)

    dt_bar_years = (5 * 60) / SECONDS_PER_YEAR
    z = rng.normal(size=n_bars)
    diffusion = (mu_bar - 0.5 * sigma_bar ** 2) * dt_bar_years + sigma_bar * np.sqrt(dt_bar_years) * z

    # --- jump component: day-level Bernoulli(EXT_JUMP_PROB_PER_DAY), one
    # jump per flagged day at a uniformly-random bar within that day.
    jump_flag_day = rng.random(n_days) < EXT_JUMP_PROB_PER_DAY
    n_flagged = int(jump_flag_day.sum())
    if n_flagged:
        flagged_days = np.nonzero(jump_flag_day)[0]
        bar_within_day = rng.integers(0, BARS_PER_DAY, size=n_flagged)
        jump_bar_idx = flagged_days * BARS_PER_DAY + bar_within_day
        is_up = rng.random(n_flagged) < 0.5
        jump_vals = np.where(
            is_up,
            rng.normal(EXT_JUMP_UP_MEAN, EXT_JUMP_UP_STD, size=n_flagged),
            rng.normal(EXT_JUMP_DOWN_MEAN, EXT_JUMP_DOWN_STD, size=n_flagged),
        )
        diffusion[jump_bar_idx] += jump_vals

    log_price = np.log(start_price) + np.cumsum(diffusion)
    close = np.exp(log_price)
    open_ = np.concatenate([[start_price], close[:-1]])

    hi_lo_mult = 0.0005
    high = np.maximum(open_, close) * (1.0 + hi_lo_mult)
    low = np.minimum(open_, close) * (1.0 - hi_lo_mult)

    idx = pd.date_range(start=start, periods=n_bars, freq="5min", tz="UTC")
    ohlc = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )

    # sanity checks -- own synthetic output.
    assert len(ohlc) == n_bars
    assert np.isfinite(ohlc.to_numpy()).all(), "NaN/Inf in synthetic path"
    assert (ohlc[["open", "high", "low", "close"]] > 0).to_numpy().all(), "non-positive price"
    assert (ohlc["high"] >= ohlc[["open", "close", "low"]].max(axis=1) - 1e-9).all()
    assert (ohlc["low"] <= ohlc[["open", "close", "high"]].min(axis=1) + 1e-9).all()

    return ohlc


def path_generator(seed: int) -> pd.DataFrame:
    return simulate_path(N_BARS_TARGET, seed)


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

def main() -> None:
    hr("R-119 NOVEL: externally-calibrated regime-switching Monte Carlo selection")
    print("Calibration source summary (see module docstring for full disclosure):")
    print(f"  EXT_BEAR_DRAWDOWN={EXT_BEAR_DRAWDOWN}  EXT_BEAR_DURATION_DAYS={EXT_BEAR_DURATION_DAYS}"
          f"  EXT_JUMP_PROB_PER_DAY={EXT_JUMP_PROB_PER_DAY:.4f}")
    print(f"  state durations (days): bull={BULL_DURATION_DAYS} chop={CHOP_DURATION_DAYS} "
          f"bear={BEAR_DURATION_DAYS}  (cycle={CYCLE_DAYS} = {CYCLE_DAYS / DAYS_PER_YEAR:.2f} yr)")
    for s in CYCLE_ORDER:
        print(f"  state={STATE_NAMES[s]:5s} drift_annual={STATE_DRIFT_ANNUAL[s]:+.4f} "
              f"(log/yr)  vol_annual={STATE_VOL_ANNUAL[s]:.4f}")
    print(f"  ORDINARY_VOL_ANNUAL (the one internally-sourced scalar, pooled daily RV from "
          f"load_inner_train_btc()) = {ORDINARY_VOL_ANNUAL:.4f}")
    print(f"  vs R-118's own fitted bear state: drift -0.65/yr, vol 1.43/yr, jump 0.53% of bars "
          f"(quoted from docs/LEDGER.md via r119_shared.py's docstring)")
    print(f"  ours: drift {STATE_DRIFT_ANNUAL[BEAR]:+.4f}/yr "
          f"(~{STATE_DRIFT_ANNUAL[BEAR] / -0.65:.2f}x), vol {STATE_VOL_ANNUAL[BEAR]:.4f}/yr "
          f"(~{STATE_VOL_ANNUAL[BEAR] / 1.43:.2f}x), implied jump-flagged bars "
          f"~{EXT_JUMP_PROB_PER_DAY / BARS_PER_DAY:.5f} (~{(EXT_JUMP_PROB_PER_DAY / BARS_PER_DAY) / 0.0053:.2f}x)")

    # --- time a single path draw + sanity-check a single backtest ---
    t0 = time.time()
    p0 = path_generator(0)
    t_sim = time.time() - t0
    t1 = time.time()
    s0 = score_on_path(V4_DEFAULT, p0, SPOT)
    t_bt = time.time() - t1
    print(f"\n[timing] one simulate_path: {t_sim:.2f}s, one score_on_path (V4_DEFAULT): {t_bt:.2f}s")
    print(f"[sanity] V4_DEFAULT sharpe on path_generator(0): {s0:+.3f}  "
          f"(finite={np.isfinite(s0)}, n_bars={len(p0)}, "
          f"close range=[{p0['close'].min():.1f}, {p0['close'].max():.1f}])")
    per_draw = t_sim + t_bt * len(GRID)
    est_total_min = per_draw * N_DRAWS / 60.0
    print(f"[timing] estimated full sweep ({N_DRAWS} draws x {len(GRID)} configs "
          f"= {N_DRAWS * len(GRID)} backtests + {N_DRAWS} sims): ~{est_total_min:.1f} min")

    hr(f"Selection sweep: n_draws={N_DRAWS}, grid={len(GRID)} points, "
       f"{N_DRAWS * len(GRID)} backtests")
    t_sweep0 = time.time()
    best_config, table = select_config(path_generator, n_draws=N_DRAWS, grid=GRID, market=SPOT)
    sweep_time = time.time() - t_sweep0
    print(f"[timing] actual sweep wall time: {sweep_time / 60.0:.1f} min")

    print("\nSelection table (config -> mean, std, robust/CVaR Sharpe over synthetic draws):")
    for cfg in GRID:
        row = table[cfg]
        marker = "  <== SELECTED" if cfg == best_config else ""
        print(f"  base={cfg[0]:3d} tv={cfg[1]:.2f} ml={cfg[2]:.1f}  "
              f"mean={row['mean']:+.3f} std={row['std']:.3f} robust={row['robust']:+.3f}{marker}")
    print(f"\nBest config (max robust/CVaR score): {best_config}")
    print(f"v4's own default: {V4_DEFAULT}  {'(SAME)' if best_config == V4_DEFAULT else '(DIFFERENT)'}")

    hr("Step 4: frozen real-data evaluate_candidate (called exactly once)")
    result = evaluate_candidate(best_config, "R119_novel")
    print_report(result)

    out = dict(
        branch="novel_regimeswitch_external",
        calibration=dict(
            ext_bear_drawdown=EXT_BEAR_DRAWDOWN,
            ext_bear_duration_days=EXT_BEAR_DURATION_DAYS,
            ext_jump_prob_per_day=EXT_JUMP_PROB_PER_DAY,
            state_duration_days={STATE_NAMES[s]: STATE_DURATION_DAYS[s] for s in CYCLE_ORDER},
            state_drift_annual_log={STATE_NAMES[s]: STATE_DRIFT_ANNUAL[s] for s in CYCLE_ORDER},
            state_vol_annual={STATE_NAMES[s]: STATE_VOL_ANNUAL[s] for s in CYCLE_ORDER},
            ordinary_vol_annual_internal_scalar=ORDINARY_VOL_ANNUAL,
            bear_vol_mult=BEAR_VOL_MULT,
            cycle_days=CYCLE_DAYS,
        ),
        n_draws=N_DRAWS,
        total_backtests_selection_sweep=N_DRAWS * len(GRID),
        timing=dict(sim_seconds=t_sim, backtest_seconds=t_bt, sweep_wall_seconds=sweep_time),
        selection_table={f"{c[0]}_{c[1]}_{c[2]}": table[c] for c in GRID},
        best_config=list(best_config),
        v4_default=list(V4_DEFAULT),
        selected_equals_v4_default=bool(best_config == V4_DEFAULT),
        evaluate_candidate_result=result,
    )
    out_path = ROOT / "experiments" / "r119_novel_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved calibration + selection table + evaluate_candidate result -> {out_path}")


if __name__ == "__main__":
    main()
