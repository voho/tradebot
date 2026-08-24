#!/usr/bin/env python
"""R-106 NOVEL branch: ``EntropyDisagreementKellyV4`` -- the SHANNON ENTROPY of
the four cross-model regime-detector alarm states (``r106_shared.
build_normalized_states``: BOCPD, Kalman-LLT, CSD, Hawkes), read as a live,
causal, bar-level multiplicative discount on ``kelly_regime_v4``'s own
``frac * scale`` product, exactly as the sibling CONSERVATIVE branch's
``cross_sectional_std`` statistic is used -- but built from a STRUCTURALLY
DIFFERENT combination rule.

MECHANISM, one sentence: treat the four models' normalized [0,1] alarm
readings at each bar as an unnormalized weight vector, map it onto the
4-outcome probability simplex, and take the SHANNON ENTROPY of that
distribution (Shannon 1948) -- high entropy means the four models are
roughly EQUALLY alarmed (genuine ambiguity about which regime signal to
trust right now), low entropy means one or two models DOMINATE the alarm
reading (the models agree on which signal matters, even if the overall
alarm LEVEL is high) -- a construction that is maximized by an EVEN SPREAD
of relative mass regardless of the states' absolute magnitude, structurally
distinct from cross-sectional standard deviation, which is maximized by
EXTREME spread in magnitude and is blind to how that spread is distributed
across the four members. Literature grounding (Zarnowitz & Lambros 1987;
Bomberger 1996) and the full non-duplication argument against every other
prior ERR-axis round and every SIZE-axis round live in
``experiments/r106_shared.py``'s own module docstring (read in full before
this file was written); not re-derived here.

SIMPLEX CONSTRUCTION, disclosed choice: softmax(states) vs. plain
sum-normalization. Softmax adds a temperature hyperparameter with no
principled value pre-registered anywhere in this project's own ledger --
sweeping it would silently inflate this branch's trial count with an
UNJUSTIFIED free knob purely to make the entropy statistic "sharper" or
"flatter" after the fact. Plain sum-normalization (``p_i = state_i /
sum(states)``) needs no such knob: it is well-defined here because
``r106_shared.causal_rolling_percentile_rank`` computes each row as
``mean(window <= x[t])`` with ``x[t]`` ITSELF included in the window, so
every normalized state is provably in ``(0, 1]`` (never exactly 0; verified
by ``_self_test`` below), guaranteeing ``sum(states) > 0`` and a
well-defined simplex everywhere all four states are non-NaN. This module
therefore uses plain sum-normalization, not softmax.

EXACT CONSTRUCTION. At every bar ``t`` with all four states non-NaN:
  1. ``p_i[t] = state_i[t] / sum_j state_j[t]``  (4-outcome simplex).
  2. ``H[t] = -sum_i p_i[t] * log(p_i[t])``, ``H_norm[t] = H[t] / log(4)``
     -- bounded in ``[0, 1]`` (0 = one model totally dominates the alarm
     reading; 1 = all four equally alarmed).
  3. ``discount[t] = clip(1 - H_norm[t], floor, 1.0)`` -- BOUNDED,
     MONOTONIC NON-INCREASING in ``H_norm`` (never increases exposure
     relative to v4, exactly the sibling branch's own brake shape), no
     expanding-quantile reference scale needed (unlike R-105's ensemble
     branch) because ``H_norm`` is ALREADY bounded ``[0,1]`` by
     construction -- one fewer nuisance parameter than R-105's own novel
     branch needed.
  4. Wherever any of the four states is NaN (warmup), ``discount[t] = 1.0``
     (parity with v4 -- the standard convention every prior SIZE/ERR-axis
     round in this project's ledger uses).
  5. ``build_target(df, floor) = apply_deadband(v4_raw_desired(df) *
     discount_array(df, floor))`` -- ``vote`` and ``scale`` themselves are
     completely UNTOUCHED, identical SLOT convention to every recent
     SIZE/ERR-axis round.

DISCLOSED WARMUP OVERRIDE (measured empirically on real BTC data, not
assumed): ``r106_shared.build_normalized_states`` needs ~298 calendar days
of trailing history before ALL FOUR normalized states are simultaneously
non-NaN on this project's real BTC series (verified directly: feeding
``build_normalized_states`` frames starting at three different calendar
dates each produced the first fully-valid row at day 298 of that frame,
regardless of where the frame itself started -- see this file's own
``_measure_required_warmup_days`` note below). ``TargetStrategy``'s SHARED
default ``warmup`` (80 calendar days, imported unchanged from
``r105_shared``) is far short of that. Per the documented R-103/R-104/R-105
trap ("inflating warmup globally silences on_bar entirely on any frame
shorter than the sentinel"), this file does NOT mutate
``TargetStrategy.warmup`` globally -- instead every ``TargetStrategy``
instance THIS FILE constructs (candidate AND control, for fairness) is
given an EXPLICIT ``warmup=CUSTOM_WARMUP_BARS`` override
(``CUSTOM_WARMUP_DAYS = 340`` calendar days, chosen to leave a ~42-day
margin past the measured 298-day requirement). Consequence, disclosed: the
generic ``r105_shared.compare()`` / ``inner_val_rows()`` / ``b5_fee_tier()``
helpers all construct ``TargetStrategy`` WITHOUT this override, so they
cannot be reused as-is for this branch's own promotion-bar cells -- this
file defines its own ``compare_custom`` / ``inner_val_rows_custom`` /
``b5_fee_tier_custom`` (identical logic, ``warmup`` threaded through) and
reuses the PURE post-processing helpers (``b1_from_inner_val``,
``b2_diagnostic``, ``b4_eth_falsification``, none of which construct a
``TargetStrategy`` at all) directly from ``r105_shared`` unchanged, so the
promotion-bar ARITHMETIC stays numerically identical to R-104/R-105's own
even though the backtest construction underneath it does not.
``eth_replication`` (frame starts at its own true beginning, ``lo=0`` so
``prefix=0`` regardless of ``warmup``) is UNAFFECTED by the override and
therefore measures the first ~298 days of ETH with ``discount=1.0``
(parity with v4) throughout -- understating rather than overstating any
genuine effect there, disclosed, not patched around.

=====================================================================
PRE-REGISTRATION (frozen before any real-data entropy, discount, R^2, or
backtest number in this file was computed -- docs/ROUTINE.md steps 1-2).
Anything below later contradicted by what actually happened is stated in
the results section, not edited back into this banner.
=====================================================================

STEP-0 GRID: ``floor in STEP0_FLOOR_GRID = (0.3, 0.5, 0.7)`` (identical grid
values to the CONSERVATIVE branch, reused as plain constants for direct
comparability -- this file does not import the sibling's own module).
Computed on BTC's inner-train window only. KILL (STOP, this file's entire
product, NEGATIVE) if ``R^2(candidate_target, v4_target) > 0.98`` on ALL
THREE floor-grid cells -- the single, pre-registered, decision-bearing
Step-0 test named in this round's dispatch. ``bind_frac`` (fraction of
inner-train bars where ``discount < 1``) is additionally computed and
reported as corroborating diagnostic context, NOT part of the pass/fail
decision. If Step-0 does not kill, PRIMARY CELL SELECTION: the same
``qualifies = bind_frac > BIND_FRAC_THRESH(0.01) AND r_sq < R2_THRESH(0.98)``
rule and ``SELECTION_ORDER = (0.5, 0.3, 0.7)`` preference order R-105's own
novel branch used (imported as plain constants from ``r105_shared``, for
comparability); STOP (NEGATIVE) if no cell qualifies.

TWO PRE-REGISTERED FALSIFICATION TESTS (named now, before any real-data
number exists, both DISCLOSED regardless of outcome, neither alone kills
the round -- both branches still get measured and reported):
  (F1) REDUNDANCY: Pearson correlation between this file's ``H_norm`` and
       the CONSERVATIVE branch's ``cross_sectional_std`` statistic (the
       latter recomputed INLINE from ``r106_shared.build_normalized_states``
       alone, as ``states.std(axis=1)`` -- this file does not read or
       import the sibling's own module), over inner-train. If
       ``corr > 0.9``, that is disclosed as evidence this is a redundant
       reparameterization of the conservative branch's statistic rather
       than a structurally different construction -- reported plainly
       either way.
  (F2) DEGENERACY: does ``H_norm`` visibly rise during a MAJORITY (>=4 of
       6) of ``r106_shared.STRESS_EPISODES``, relative to its own
       unconditional (whole pre-holdout, non-NaN) mean? Computed on the
       WIDER pre-holdout range (matching ``r106_shared.step0_gate``'s own
       convention: three of six episode onsets fall after
       ``INNER_TRAIN_END``, inside ``INNER_VAL_START..INNER_VAL_END``,
       still non-holdout). If entropy does NOT concentrate around a
       majority of the six episodes, that is disclosed as evidence of a
       degenerate, uninformative statistic (passing the shared Step-0
       gate, which only checked the RAW normalized states' own properties,
       says nothing about THIS combination rule's own behaviour).

PROMOTION BAR (gating unless noted), via this file's own
``compare_custom()``/``inner_val_rows_custom()``:
  B1 (gating): ``r105_shared.b1_from_inner_val`` on the primary cell's
     inner_val rows, both markets -- ``d_sharpe > +0.2`` OR bootstrap
     interval excludes zero favourably.
  B2 (diagnostic ONLY, never gates): ``r105_shared.b2_diagnostic`` --
     drawdown improvement counted only where risk-matched (R-33's rule).
  B3 (plateau, gating): sweep the SAME ``STEP0_FLOOR_GRID`` on
     inner-validation (not a separate nuisance parameter -- this
     construction has none, having no expanding-quantile reference scale
     to sweep); PASS requires a directionally-consistent (same-sign
     majority) region across the 6 cells (3 floors x 2 markets), not an
     isolated peak.
  B4 (ETH falsification, gating, pre-registered): ``r105_shared.
     b4_eth_falsification`` -- FULL pass required (both markets same-sign
     as BTC inner_val).
  B5 (cost robustness, gating): this file's own ``b5_fee_tier_custom`` at
     0.40% taker (``r105_shared.FEE_TIER``), primary cell, BTC inner_val,
     both markets -- no sign reversal.
PROMOTE-candidate only if causal-truncation probe passes AND Step-0 AND B1
AND B3 AND B4 (full) AND B5 all hold (B2 diagnostic-only). Default:
NEGATIVE. No threshold or decision rule is changed after seeing a number.

HOLDOUT (run once a primary cell is selected, REGARDLESS of the inner-val
promotion-bar outcome, per this round's own dispatch instructions -- report
all of it): OOS (``start=OOS_START``) vs. spot ``buy_and_hold``, both
markets, at the standard fee tier AND the 0.40% taker tier, futures BOTH
funding-free (this project's published upper-bound convention) AND with
REAL funding charged (``tradebot.data.load_funding_extended``, Binance
2020-2023 + Deribit extension where the Binance series ends) -- mirroring
``scripts/funding_study.py``'s own ``_period`` helper, generalized here to
accept any strategy object rather than only registry-looked-up ones (the
registry has no entry for this file's own composed target).

CAUSAL SAFETY: ``r105_shared.causal_truncation_probe_series`` (generic,
works on any ``df -> np.ndarray`` builder) on the composed ``build_target``
at the primary floor, on real BTC data.

WHAT WOULD MAKE THIS FAIL, named now: (1) F1 above -- entropy could turn
out to be a near-total redirection of the conservative branch's own
statistic, adding no NEW information despite the different construction.
(2) F2 above -- entropy could fail to concentrate around genuine historical
stress at all, the same "real but inert"/degenerate-statistic shape
R-87/R-104's own novel branches found by other estimators. (3) Because
``discount[t] < 1`` iff ``H_norm[t] > 0``, and every non-NaN state is
STRICTLY positive by the percentile-rank construction (see above), an
exactly-uniform four-way tie is a MEASURE-ZERO event on real, continuous
price-derived alarm scores -- ``bind_frac`` is expected to come out close
to 100% (not near-degenerate the way R-105's ensemble discount was),
meaning ``floor`` alone controls how DEEP the discount goes almost
everywhere it is non-NaN, not WHETHER it binds at all; a materially
different disclosed property from R-105's own novel branch, verified
numerically below, not assumed.

CONFIGURATIONS EVALUATED IN THIS FILE: 3 (Step-0 floor grid) + 6 (primary
cell's ``compare_custom()``: inner_train x2 + inner_val x2 +
eth_replication x2) + 6 (B3's floor-grid plateau, 3 floors x 2 markets, 2
reused directly from the primary cell's own inner_val rows) + 2 (B5's
0.40% fee tier, 2 markets) + up to 10 (holdout: candidate x {spot std,
spot fee-tier, futures std no-funding, futures std w/funding, futures
fee-tier no-funding} + the SAME 5 for the v4 control, run for a matched
comparison) + 2 (spot ``buy_and_hold`` benchmark, std fee + fee-tier) = 29
total IF Step-0 selects a primary cell; 3 total (Step-0 grid only) if it
does not.

USAGE
-----
    python experiments/r106_novel_entropy_disagreement_brake.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding_extended  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import daily_returns as inference_daily_returns  # noqa: E402
from tradebot.inference import total_log_return  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import prefix_bars, run_period  # noqa: E402

from experiments.r106_shared import (  # noqa: E402
    BARS_PER_DAY,
    ETH_SLICE_NAME,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    MODEL_NAMES,
    OOS_START,
    SPOT,
    STRESS_EPISODES,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    build_normalized_states,
    episode_disagreement_summary,
    episode_window,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    v4_raw_desired,
    v4_target,
)
from experiments.r105_shared import (  # noqa: E402
    BIND_FRAC_THRESH,
    FEE_TIER,
    R2_THRESH,
    SELECTION_ORDER,
    SHARPE_NOISE_FLOOR,
    STEP0_FLOOR_GRID,
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    causal_truncation_probe_series,
    hr,
    print_plateau_table,
)

DATA_DIR = ROOT / "data"

# ---------------------------------------------------------- pre-registered
CUSTOM_WARMUP_DAYS = 340   # measured requirement ~298d + ~42d margin (see docstring)
CUSTOM_WARMUP_BARS = CUSTOM_WARMUP_DAYS * BARS_PER_DAY + 10
FLOOR_DEFAULT = 0.5


# ================================================================== (1)
# The mechanism: 4-state simplex, Shannon entropy, bounded clip-discount.
# ==================================================================

def simplex_probs(states: pd.DataFrame) -> pd.DataFrame:
    """Plain sum-to-1 normalization of the four (0,1]-valued states onto a
    4-outcome probability simplex (see module docstring for why softmax's
    temperature knob is deliberately NOT used)."""
    total = states.sum(axis=1)
    return states.div(total, axis=0)


def shannon_entropy_norm(states: pd.DataFrame) -> pd.Series:
    """H_norm[t] = -sum_i p_i log(p_i) / log(4), in [0, 1]. NaN wherever any
    of the four states is NaN (states.sum/.div already propagate NaN)."""
    p = simplex_probs(states)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p * np.log(p)
    # p_i is provably > 0 wherever non-NaN (see docstring), so no 0*log(0)
    # case can arise; terms is only ever NaN where p itself is NaN.
    h = -terms.sum(axis=1, skipna=False)
    return (h / np.log(4)).rename("entropy_norm")


def cross_sectional_std_inline(states: pd.DataFrame) -> pd.Series:
    """The CONSERVATIVE branch's own statistic, recomputed INLINE (per this
    round's own pre-registration) rather than imported from a sibling
    branch's file -- identical one-line reduction to
    ``r106_shared.cross_sectional_std``."""
    return states.std(axis=1, skipna=False).rename("cross_sectional_std_inline")


def discount_from_entropy(h_norm: pd.Series, floor: float) -> np.ndarray:
    vals = h_norm.to_numpy()
    raw = np.clip(1.0 - vals, floor, 1.0)
    return np.where(np.isfinite(vals), raw, 1.0)


def build_target(df: pd.DataFrame, floor: float = FLOOR_DEFAULT,
                 states: pd.DataFrame | None = None) -> np.ndarray:
    """``build_target(df, floor) = apply_deadband(v4_raw_desired(df) *
    discount_array(df, floor))``. ``states`` may be supplied precomputed
    (Step-0 grid reuses one ``build_normalized_states`` call across all
    three floors)."""
    if states is None:
        states = build_normalized_states(df)
    h_norm = shannon_entropy_norm(states)
    discount = discount_from_entropy(h_norm, floor)
    raw = v4_raw_desired(df) * discount
    return apply_deadband(raw)


def make_build_target(floor: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, floor=floor)
    _build.__name__ = f"entropy_disagreement_floor{floor:g}"
    return _build


# ================================================================== (2)
# Step-0 grid + both pre-registered falsification tests.
# ==================================================================

def step0_grid(btc: pd.DataFrame) -> dict:
    mask = np.asarray((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                      (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    n_bars = int(mask.sum())

    states = build_normalized_states(btc)
    h_norm = shannon_entropy_norm(states)
    cross_std = cross_sectional_std_inline(states)

    # Every non-NaN state is strictly > 0 by construction -- verify on real
    # data, not just the synthetic self-test below.
    valid_states = states.dropna()
    min_state = float(valid_states.to_numpy().min()) if len(valid_states) else float("nan")

    raw_base = v4_raw_desired(btc)
    ctrl_target = v4_target(btc)

    rows = []
    for floor in STEP0_FLOOR_GRID:
        discount = discount_from_entropy(h_norm, floor)
        target = apply_deadband(raw_base * discount)
        bind_frac = float(np.mean(discount[mask] < 1.0 - 1e-9))
        r_sq = r_squared(target[mask], ctrl_target[mask])
        qualifies = (bind_frac > BIND_FRAC_THRESH) and (r_sq < R2_THRESH)
        rows.append(dict(floor=floor, bind_frac=bind_frac, r_sq=r_sq, qualifies=qualifies))

    step0_kill = all(r["r_sq"] > R2_THRESH for r in rows)

    # ---------------------------------------------------------------- F1
    h_train = h_norm.to_numpy()[mask]
    cs_train = cross_std.to_numpy()[mask]
    finite = np.isfinite(h_train) & np.isfinite(cs_train)
    if finite.sum() >= 2 and np.std(h_train[finite]) > 0 and np.std(cs_train[finite]) > 0:
        f1_corr = float(np.corrcoef(h_train[finite], cs_train[finite])[0, 1])
    else:
        f1_corr = float("nan")
    f1_redundant = bool(np.isfinite(f1_corr) and abs(f1_corr) > 0.9)

    # ---------------------------------------------------------------- F2
    ep_summary = episode_disagreement_summary(btc, h_norm, window_days=60)
    unconditional_mean = float(h_norm.dropna().mean())
    ep_summary = ep_summary.copy()
    ep_summary["rises_vs_unconditional"] = ep_summary["mean"] > unconditional_mean
    n_rising = int(ep_summary["rises_vs_unconditional"].sum())
    f2_majority_rises = bool(n_rising >= 4)  # majority of 6

    return dict(
        rows=rows, n_bars=n_bars, step0_kill=step0_kill,
        min_state=min_state,
        f1_corr=f1_corr, f1_redundant=f1_redundant,
        f2_episode_summary=ep_summary, f2_unconditional_mean=unconditional_mean,
        f2_n_rising=n_rising, f2_majority_rises=f2_majority_rises,
        states=states, h_norm=h_norm, cross_std=cross_std,
    )


def select_primary(rows: list[dict]) -> dict | None:
    by_floor = {r["floor"]: r for r in rows}
    for f in SELECTION_ORDER:
        r = by_floor.get(f)
        if r is not None and r["qualifies"]:
            return r
    return None


def print_step0_table(step0: dict) -> None:
    rows, n_bars = step0["rows"], step0["n_bars"]
    print(f"\nSTEP-0 GRID (inner-train slice, {INNER_TRAIN_START} -> {INNER_TRAIN_END}, "
          f"{n_bars:,} bars)")
    print("KILL rule (decision-bearing): R^2(candidate, v4) > 0.98 on ALL 3 cells")
    print(f"(diagnostic-only) qualify = bind_frac > {BIND_FRAC_THRESH:.0%} AND r_sq < {R2_THRESH}")
    hdr = f"{'floor':>6s} {'bind_frac':>10s} {'r_sq':>8s} {'qualifies':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tag = "  <- grid centre" if r["floor"] == 0.5 else ""
        print(f"{r['floor']:6.2f} {r['bind_frac']:10.4f} {r['r_sq']:8.4f} "
              f"{'YES' if r['qualifies'] else 'no':>10s}{tag}")
    print(f"\nSTEP-0 KILL (R^2>0.98 on ALL cells): {step0['step0_kill']}")
    print(f"min observed non-NaN state (real data, sanity check it is > 0): {step0['min_state']:.6f}")


def print_falsification_tests(step0: dict) -> None:
    hr("PRE-REGISTERED FALSIFICATION TEST F1 -- redundancy vs. cross_sectional_std")
    print(f"Pearson corr(H_norm, cross_sectional_std_inline), inner-train: "
          f"{step0['f1_corr']:+.4f}")
    print(f"redundant reparameterization (|corr| > 0.9): {step0['f1_redundant']}")
    print("(disclosed either way; does not by itself kill the round)")

    hr("PRE-REGISTERED FALSIFICATION TEST F2 -- episode concentration")
    print(step0["f2_episode_summary"].to_string(index=False))
    print(f"\nunconditional H_norm mean (whole pre-holdout, non-NaN): "
          f"{step0['f2_unconditional_mean']:.4f}")
    print(f"episodes where mean H_norm > unconditional mean: {step0['f2_n_rising']} / 6")
    print(f"MAJORITY (>=4/6) rises during stress: {step0['f2_majority_rises']}")


# ================================================================== (3)
# Local compare()/inner_val_rows()/b5 equivalents, warmup-overridden.
# ==================================================================

def daily_simple_returns(equity: pd.Series) -> np.ndarray:
    return inference_daily_returns(equity).to_numpy()


@dataclass
class SliceResultLocal:
    name: str
    market: str
    sharpe: float
    max_drawdown_pct: float
    log_growth: float
    daily: np.ndarray
    mean_abs_exposure: float
    realized_vol: float


def run_slice_custom(build_fn, name: str, df: pd.DataFrame, start: str | None, end: str | None,
                     slice_name: str, market: MarketSpec, warmup: int,
                     balance: float = 1_000.0) -> SliceResultLocal:
    """``r102_shared.run_slice``, but the wrapped ``TargetStrategy`` gets an
    EXPLICIT ``warmup`` override rather than the shared 80-day default."""
    if end is not None:
        assert pd.Timestamp(end) < pd.Timestamp(OOS_START), (
            f"run_slice_custom({slice_name!r}): end={end} not before OOS_START")
    assert_no_holdout(df, slice_name)
    strategy = TargetStrategy(build_fn, name=name, warmup=warmup)
    res = run_period(strategy, df, start, end, market=market, start_balance=balance)
    assert_no_holdout(res.equity.to_frame(), f"{slice_name} result")
    m = compute_metrics(res)
    d = daily_simple_returns(res.equity)
    exposure = res.df["target"].to_numpy() if "target" in res.df.columns else np.array([np.nan])
    return SliceResultLocal(
        name=slice_name, market=market.name, sharpe=m.sharpe,
        max_drawdown_pct=m.max_drawdown_pct, log_growth=float(total_log_return(d)), daily=d,
        mean_abs_exposure=float(np.nanmean(np.abs(exposure))),
        realized_vol=float(np.nanstd(d) * np.sqrt(365.25)) if len(d) > 1 else float("nan"),
    )


SLICES_LOCAL = {"inner_train": (INNER_TRAIN_START, INNER_TRAIN_END),
                "inner_val": (INNER_VAL_START, INNER_VAL_END)}


def compare_custom(candidate_build, *, label: str, btc: pd.DataFrame, eth: pd.DataFrame,
                   markets: tuple[MarketSpec, ...] = (SPOT, FUTURES),
                   warmup: int = CUSTOM_WARMUP_BARS, seed: int = 0) -> list[dict]:
    """``r102_shared.compare()``, warmup-overridden for BOTH candidate and
    control (fair comparison -- same prefix length for both)."""
    assert_no_holdout(btc, "compare_custom(): btc")
    assert_no_holdout(eth, "compare_custom(): eth")

    jobs = [(name, start, end, btc) for name, (start, end) in SLICES_LOCAL.items()]
    jobs.append((ETH_SLICE_NAME, None, None, eth))

    rows = []
    for slice_name, start, end, df in jobs:
        for market in markets:
            a = run_slice_custom(candidate_build, f"r106_{label}", df, start, end,
                                 slice_name, market, warmup)
            b = run_slice_custom(v4_target, "kelly_regime_v4", df, start, end,
                                 slice_name, market, warmup)
            pr = paired_diff(a.daily, b.daily, seed=seed)
            exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure
                        if b.mean_abs_exposure else float("nan"))
            vol_ratio = (a.realized_vol / b.realized_vol
                        if b.realized_vol else float("nan"))
            risk_matched = (bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1)
                           if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False)
            rows.append(dict(
                slice=slice_name, market=market.name,
                candidate_sharpe=a.sharpe, control_sharpe=b.sharpe,
                d_sharpe=a.sharpe - b.sharpe, d_dd=a.max_drawdown_pct - b.max_drawdown_pct,
                exposure_ratio=exp_ratio, vol_ratio=vol_ratio, risk_matched=risk_matched,
                boot_d_loggrowth=pr.diff.point, boot_lo=pr.diff.lo, boot_hi=pr.diff.hi,
                excludes_zero=bool(pr.diff.lo > 0 or pr.diff.hi < 0),
            ))
    return rows


def inner_val_rows_custom(build_fn, label: str, btc: pd.DataFrame,
                          markets: tuple = (SPOT, FUTURES),
                          warmup: int = CUSTOM_WARMUP_BARS) -> list[dict]:
    rows = []
    for market in markets:
        a = run_slice_custom(build_fn, f"r106_{label}", btc, INNER_VAL_START, INNER_VAL_END,
                             "inner_val", market, warmup)
        b = run_slice_custom(v4_target, "kelly_regime_v4", btc, INNER_VAL_START, INNER_VAL_END,
                             "inner_val", market, warmup)
        pr = paired_diff(a.daily, b.daily)
        exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure
                    if b.mean_abs_exposure else float("nan"))
        vol_ratio = (a.realized_vol / b.realized_vol
                    if b.realized_vol else float("nan"))
        risk_matched = (bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1)
                       if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False)
        rows.append(dict(
            label=label, market=market.name,
            d_sharpe=a.sharpe - b.sharpe, d_dd=a.max_drawdown_pct - b.max_drawdown_pct,
            exposure_ratio=exp_ratio, vol_ratio=vol_ratio, risk_matched=risk_matched,
            boot_d_loggrowth=pr.diff.point, boot_lo=pr.diff.lo, boot_hi=pr.diff.hi,
            excludes_zero=bool(pr.diff.lo > 0 or pr.diff.hi < 0),
        ))
    return rows


def b5_fee_tier_custom(build_primary, label: str, btc: pd.DataFrame,
                       inner_val_primary: list[dict]) -> tuple[bool, list[dict]]:
    fee_markets = (fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER))
    fee_rows = inner_val_rows_custom(build_primary, label, btc, markets=fee_markets)
    cells = []
    for r in fee_rows:
        base = next((c for c in inner_val_primary if c["market"] == r["market"]), None)
        d_sharpe_no_reversal = (base is not None and
                               not (np.sign(r["d_sharpe"]) != np.sign(base["d_sharpe"])
                                    and r["d_sharpe"] != 0 and base["d_sharpe"] != 0))
        dlog_no_reversal = (base is not None and
                          not (np.sign(r["boot_d_loggrowth"]) != np.sign(base["boot_d_loggrowth"])
                               and r["boot_d_loggrowth"] != 0 and base["boot_d_loggrowth"] != 0))
        cells.append(dict(market=r["market"], d_sharpe=r["d_sharpe"],
                          base_d_sharpe=base["d_sharpe"] if base else float("nan"),
                          boot_d_loggrowth=r["boot_d_loggrowth"],
                          base_boot_d_loggrowth=base["boot_d_loggrowth"] if base else float("nan"),
                          d_sharpe_no_reversal=d_sharpe_no_reversal,
                          dlog_no_reversal=dlog_no_reversal,
                          no_reversal=d_sharpe_no_reversal and dlog_no_reversal))
    return all(c["no_reversal"] for c in cells), cells


def run_b3(primary_floor: float, inner_val_primary_rows: list[dict],
          btc: pd.DataFrame) -> tuple[dict[float, list[dict]], bool]:
    plateau_rows: dict[float, list[dict]] = {}
    for floor in STEP0_FLOOR_GRID:
        if floor == primary_floor:
            plateau_rows[floor] = [dict(label=f"entropy_disagreement_floor{floor:g}",
                                        market=r["market"], d_sharpe=r["d_sharpe"], d_dd=r["d_dd"],
                                        exposure_ratio=r["exposure_ratio"], vol_ratio=r["vol_ratio"],
                                        risk_matched=r["risk_matched"],
                                        boot_d_loggrowth=r["boot_d_loggrowth"], boot_lo=r["boot_lo"],
                                        boot_hi=r["boot_hi"], excludes_zero=r["excludes_zero"])
                                   for r in inner_val_primary_rows]
        else:
            bf = make_build_target(floor)
            label = f"entropy_disagreement_floor{floor:g}"
            plateau_rows[floor] = inner_val_rows_custom(bf, label, btc)

    same_sign_flags = [r["d_sharpe"] > 0 for rows in plateau_rows.values() for r in rows]
    b3_pass = (sum(same_sign_flags) >= len(same_sign_flags) / 2.0) if same_sign_flags else False
    return plateau_rows, b3_pass


# ================================================================== (4)
# HOLDOUT evaluation.
# ==================================================================

def load_full_btc() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    print(f"full BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}",
          file=sys.stderr)
    return df


@dataclass
class HoldoutResult:
    tag: str
    market: str
    final_balance: float
    sharpe: float
    max_drawdown_pct: float
    log_growth: float
    daily: np.ndarray
    funding_paid: float


def run_holdout_period(build_fn, name: str, df_full: pd.DataFrame, market: MarketSpec,
                       warmup: int, funding: pd.Series | None = None,
                       start: str = OOS_START, balance: float = 1_000.0) -> HoldoutResult:
    """``scripts/funding_study.py``'s own ``_period`` helper, generalized to
    accept any pure ``build_target`` function via ``TargetStrategy`` rather
    than only registry-looked-up strategies. Deliberately bypasses
    ``run_slice``'s ``assert_no_holdout`` guard on the INPUT frame (the
    whole point of a holdout run) -- the guard is still applied to every
    OTHER frame this file touches before this point."""
    strategy = TargetStrategy(build_fn, name=name, warmup=warmup)
    lo = int(df_full.index.searchsorted(pd.Timestamp(start, tz="UTC")))
    hi = len(df_full)
    pre = prefix_bars(df_full, lo, strategy.warmup)
    raw = run_backtest(strategy, df_full.iloc[lo - pre: hi], market, balance,
                       trade_start=pre, funding=funding)
    from dataclasses import replace
    trimmed = raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:],
                                           df=raw.df.iloc[pre:])
    m = compute_metrics(trimmed)
    d = daily_simple_returns(trimmed.equity)
    return HoldoutResult(tag=name, market=market.name, final_balance=m.final_balance,
                        sharpe=m.sharpe, max_drawdown_pct=m.max_drawdown_pct,
                        log_growth=float(total_log_return(d)), daily=d,
                        funding_paid=float(raw.funding_paid))


def run_holdout_suite(build_primary, primary_floor: float, df_full: pd.DataFrame,
                      warmup: int) -> dict:
    hr(f"HOLDOUT (start={OOS_START}) -- primary floor={primary_floor:g}")
    real_funding, funding_source = load_funding_extended(DATA_DIR)
    print(f"real funding series: {'available, ' + str(len(real_funding)) + ' settlements, ' + str(real_funding.index[0].date()) + ' -> ' + str(real_funding.index[-1].date()) if real_funding is not None else 'NOT AVAILABLE'}")

    fee_spot, fee_fut = fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER)
    cells = []
    configs = [
        ("spot_std", SPOT, None),
        ("spot_fee0.40%", fee_spot, None),
        ("futures_std_nofunding", FUTURES, None),
        ("futures_std_realfunding", FUTURES, real_funding),
        ("futures_fee0.40%_nofunding", fee_fut, None),
    ]
    bh_strategy = get_strategy("buy_and_hold")
    bh_rows = {}
    for tag, market, funding in (("spot_std", SPOT, None), ("spot_fee0.40%", fee_spot, None)):
        # buy_and_hold isn't a pure build_target -- run it via the real registered
        # strategy object directly rather than through TargetStrategy.
        lo = int(df_full.index.searchsorted(pd.Timestamp(OOS_START, tz="UTC")))
        hi = len(df_full)
        pre = prefix_bars(df_full, lo, bh_strategy.warmup)
        raw = run_backtest(bh_strategy, df_full.iloc[lo - pre: hi], market, 1_000.0,
                           trade_start=pre, funding=funding)
        from dataclasses import replace
        trimmed = raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:],
                                               df=raw.df.iloc[pre:])
        m = compute_metrics(trimmed)
        d = daily_simple_returns(trimmed.equity)
        bh_rows[tag] = HoldoutResult(tag="buy_and_hold", market=market.name,
                                    final_balance=m.final_balance, sharpe=m.sharpe,
                                    max_drawdown_pct=m.max_drawdown_pct,
                                    log_growth=float(total_log_return(d)), daily=d,
                                    funding_paid=0.0)

    for tag, market, funding in configs:
        cand = run_holdout_period(build_primary, f"r106_entropy_floor{primary_floor:g}",
                                  df_full, market, warmup, funding)
        ctrl = run_holdout_period(v4_target, "kelly_regime_v4", df_full, market, warmup, funding)
        pr = paired_diff(cand.daily, ctrl.daily)
        bh_tag = "spot_fee0.40%" if "fee0.40%" in tag else "spot_std"
        bh = bh_rows[bh_tag]
        cells.append(dict(
            tag=tag, market=market.name,
            candidate_sharpe=cand.sharpe, control_sharpe=ctrl.sharpe,
            d_sharpe=cand.sharpe - ctrl.sharpe,
            candidate_final=cand.final_balance, control_final=ctrl.final_balance,
            candidate_dd=cand.max_drawdown_pct, control_dd=ctrl.max_drawdown_pct,
            candidate_funding_paid=cand.funding_paid, control_funding_paid=ctrl.funding_paid,
            boot_d_loggrowth=pr.diff.point, boot_lo=pr.diff.lo, boot_hi=pr.diff.hi,
            excludes_zero=bool(pr.diff.lo > 0 or pr.diff.hi < 0),
            bh_final=bh.final_balance, bh_sharpe=bh.sharpe,
            candidate_beats_bh=cand.final_balance > bh.final_balance,
            control_beats_bh=ctrl.final_balance > bh.final_balance,
        ))
    return dict(cells=cells, funding_source=funding_source)


def print_holdout(res: dict) -> None:
    hdr = (f"{'config':>26s} {'market':>9s} {'cand_Sh':>8s} {'ctrl_Sh':>8s} {'dSh':>7s} "
          f"{'cand_$':>10s} {'ctrl_$':>10s} {'bh_$':>10s} {'cand>bh':>8s} {'dlogG':>7s} "
          f"{'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for c in res["cells"]:
        print(f"{c['tag']:>26s} {c['market']:>9s} {c['candidate_sharpe']:8.2f} "
              f"{c['control_sharpe']:8.2f} {c['d_sharpe']:+7.2f} "
              f"{c['candidate_final']:10,.0f} {c['control_final']:10,.0f} "
              f"{c['bh_final']:10,.0f} {'YES' if c['candidate_beats_bh'] else 'no':>8s} "
              f"{c['boot_d_loggrowth']:+7.3f} {c['boot_lo']:+8.3f},{c['boot_hi']:+8.3f} "
              f"{'YES' if c['excludes_zero'] else 'no':>5s}")


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-106 NOVEL: EntropyDisagreementKellyV4 -- Shannon-entropy disagreement "
       "discount on v4's own frac*scale")
    print("mechanism: 4-model normalized alarm states -> sum-normalized simplex -> Shannon")
    print("entropy (normalized by log(4)) -> bounded, monotonic, floor-clipped multiplicative")
    print("discount on v4's UNCHANGED frac*scale, before v4's own deadband.")
    print(f"CUSTOM_WARMUP_DAYS={CUSTOM_WARMUP_DAYS} ({CUSTOM_WARMUP_BARS:,} bars) -- see module")
    print("docstring for the measured 298-day requirement this overrides the shared 80-day")
    print("TargetStrategy default for (applied to BOTH candidate and control, for fairness).")

    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    # ============================================================= STEP 0
    hr("STEP 0 -- non-degeneracy kill switch + BOTH pre-registered falsification tests")
    step0 = step0_grid(btc)
    print_step0_table(step0)
    print_falsification_tests(step0)

    if step0["step0_kill"]:
        hr("STEP-0 GATE: R^2 > 0.98 ON ALL 3 CELLS -- STOPPING HERE")
        print("The entropy-based brake does not measurably diverge from v4's own path on any")
        print("floor-grid cell over inner-train. Per this file's own pre-registration, this")
        print("Step-0 table (plus both falsification tests) is the branch's ENTIRE product,")
        print("reported NEGATIVE / stopped-at-Step-0. No promotion-bar or holdout code runs.")
        n_configs = len(STEP0_FLOOR_GRID)
        verdict = "NEGATIVE (Step-0 kill switch: R^2>0.98 on all cells)"
        hr("VERDICT")
        print(f"VERDICT: {verdict}")
        print(f"\nconfigurations evaluated (total): {n_configs} (3 Step-0 grid only)")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, step0=step0, primary=None, verdict=verdict, n_configs=n_configs)

    primary = select_primary(step0["rows"])
    if primary is None:
        hr("STEP-0: NO CELL QUALIFIES (bind_frac>1% AND r_sq<0.98) -- STOPPING HERE")
        n_configs = len(STEP0_FLOOR_GRID)
        verdict = "NEGATIVE (no Step-0 cell qualifies)"
        hr("VERDICT")
        print(f"VERDICT: {verdict}")
        print(f"\nconfigurations evaluated (total): {n_configs}")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, step0=step0, primary=None, verdict=verdict, n_configs=n_configs)

    primary_floor = primary["floor"]
    print(f"\nPRIMARY CELL SELECTED: floor={primary_floor:g}  "
          f"(bind_frac={primary['bind_frac']:.4f}, r_sq={primary['r_sq']:.4f})")
    build_primary = make_build_target(primary_floor)

    # ==================================================== CAUSAL PROBE
    hr("CAUSAL TRUNCATION PROBE (composed build_target, real BTC data)")
    try:
        probe_ok = causal_truncation_probe_series(build_primary, btc)
        print("  PASS")
    except AssertionError as e:
        probe_ok = False
        print(f"  FAIL: {e}")

    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "main(): eth")
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")

    # ============================================================ B1-B5
    hr(f"PROMOTION BAR -- PRIMARY CELL floor={primary_floor:g}")
    label = f"entropy_disagreement_floor{primary_floor:g}"
    rows = compare_custom(build_primary, label=label, btc=btc, eth=eth)
    print_rows_local(rows)

    inner_val_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_primary = [r for r in rows if r["slice"] == "eth_replication"]

    b1_pass, b1_cells = b1_from_inner_val(inner_val_primary)
    b2_pass, b2_cells = b2_diagnostic(inner_val_primary)
    b3_rows, b3_pass = run_b3(primary_floor, inner_val_primary, btc)
    b4_partial, b4_full, b4_cells = b4_eth_falsification(eth_primary, inner_val_primary)
    b5_pass, b5_cells = b5_fee_tier_custom(build_primary, label, btc, inner_val_primary)

    hr("B1 -- inner-validation (dSharpe > +0.2 OR bootstrap excludes zero positively)")
    for c in b1_cells:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  PASS={c['passes']}")
    print(f"B1 PASS (both markets): {b1_pass}")

    hr("B2 -- diagnostic only")
    for c in b2_cells:
        status = "VALID" if c["risk_matched"] else "VOID (not risk-matched)"
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.2f}pp  risk_matched={c['risk_matched']}  [{status}]")

    hr("B3 -- plateau: floor grid {0.3,0.5,0.7}, inner-validation")
    print_plateau_table(b3_rows)
    print(f"\nB3 (directionally consistent majority): {b3_pass}")

    hr("B4 -- ETH falsification")
    for c in b4_cells:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"same_sign_as_btc_inner_val={c['same_sign_as_btc']}")
    print(f"B4 FULL PASS: {b4_full}")

    hr("B5 -- fee-tier robustness (0.40% taker)")
    for c in b5_cells:
        print(f"  {c['market']:>9s}  fee-tier d_sharpe={c['d_sharpe']:+.4f}  "
              f"standard-fee d_sharpe={c['base_d_sharpe']:+.4f}  no_reversal={c['no_reversal']}")
    print(f"B5 PASS: {b5_pass}")

    inner_val_all_pass = probe_ok and b1_pass and b3_pass and b4_full and b5_pass
    print(f"\nINNER-VAL PROMOTION-BAR ALL-PASS (causal-safety AND B1 AND B3 AND B4_full AND B5): "
          f"{inner_val_all_pass}")

    # ============================================================ HOLDOUT
    df_full = load_full_btc()
    max_ts_seen.append(df_full.index.max())
    holdout = run_holdout_suite(build_primary, primary_floor, df_full, CUSTOM_WARMUP_BARS)
    print_holdout(holdout)

    n_configs = (3 + 6 + 6 + 2 + 10 + 2)
    max_ts = max(max_ts_seen)

    hr("VERDICT")
    verdict = "PROMOTE-candidate" if inner_val_all_pass else "NEGATIVE"
    print(f"causal safety: {probe_ok}")
    print(f"Step-0: PASS (not killed)   B1={b1_pass}   B2=diagnostic-only   B3={b3_pass}   "
          f"B4_full={b4_full}   B5={b5_pass}")
    print(f"F1 (redundancy vs cross_sectional_std): corr={step0['f1_corr']:+.4f}  "
          f"flagged_redundant={step0['f1_redundant']}")
    print(f"F2 (episode concentration): {step0['f2_n_rising']}/6 episodes rise, "
          f"majority={step0['f2_majority_rises']}")
    print(f"VERDICT (per pre-registered inner-val promotion bar): {verdict}")
    print("Holdout numbers above are reported per this round's own dispatch instructions,")
    print("REGARDLESS of the inner-val verdict -- they do not alter the frozen decision rule.")
    print(f"\nconfigurations evaluated (total): {n_configs}")
    print(f"max timestamp read anywhere in this branch: {max_ts}")
    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0=step0, primary=primary, probe_ok=probe_ok,
               compare_rows=rows, b1_cells=b1_cells, b2_cells=b2_cells, b3_rows=b3_rows,
               b4_cells=b4_cells, b5_cells=b5_cells,
               b1_pass=b1_pass, b3_pass=b3_pass, b4_full=b4_full, b5_pass=b5_pass,
               inner_val_all_pass=inner_val_all_pass, holdout=holdout,
               verdict=verdict, n_configs=n_configs, max_ts=max_ts)


def print_rows_local(rows: list[dict]) -> None:
    hdr = (f"{'slice':>16s} {'market':>9s} {'cand_Sh':>8s} {'ctrl_Sh':>8s} {'dSh':>7s} "
          f"{'dDD':>7s} {'expR':>5s} {'volR':>5s} {'RM':>3s} {'dlogG':>7s} "
          f"{'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['slice']:>16s} {r['market']:>9s} {r['candidate_sharpe']:8.2f} "
              f"{r['control_sharpe']:8.2f} {r['d_sharpe']:+7.2f} {r['d_dd']:+7.1f} "
              f"{r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
              f"{'Y' if r['risk_matched'] else 'n':>3s} {r['boot_d_loggrowth']:+7.3f} "
              f"{r['boot_lo']:+8.3f},{r['boot_hi']:+8.3f} "
              f"{'YES' if r['excludes_zero'] else 'no':>5s}")


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2018-01-01", periods=2000, freq="1D", tz="UTC")
    rng = np.random.default_rng(1060)

    # simplex_probs / shannon_entropy_norm: known-input sanity.
    equal = pd.DataFrame({m: np.full(10, 0.25) for m in MODEL_NAMES},
                         index=pd.date_range("2020-01-01", periods=10, freq="1D", tz="UTC"))
    h_equal = shannon_entropy_norm(equal)
    assert np.allclose(h_equal.to_numpy(), 1.0, atol=1e-9), h_equal.to_numpy()

    dominant = pd.DataFrame({
        "bocpd": np.full(10, 0.97), "kalman": np.full(10, 0.01),
        "csd": np.full(10, 0.01), "hawkes": np.full(10, 0.01),
    }, index=equal.index)
    h_dom = shannon_entropy_norm(dominant)
    assert (h_dom.to_numpy() < 0.3).all(), h_dom.to_numpy()
    assert (h_dom.to_numpy() >= 0.0).all()

    # discount monotonicity: higher entropy -> lower (or equal) discount.
    h = pd.Series(np.linspace(0, 1, 50))
    for floor in (0.3, 0.5, 0.7):
        d = discount_from_entropy(h, floor)
        assert np.all(np.diff(d) <= 1e-12), "discount must be non-increasing in entropy"
        assert (d >= floor - 1e-9).all() and (d <= 1.0 + 1e-9).all()

    # NaN state -> discount == 1.0 exactly.
    h_nan = pd.Series([np.nan, 0.5, np.nan])
    d_nan = discount_from_entropy(h_nan, 0.3)
    assert d_nan[0] == 1.0 and d_nan[2] == 1.0

    # cross_sectional_std_inline of identical columns is exactly zero.
    same = pd.DataFrame({m: rng.normal(size=len(idx)) for m in ["x"]}, index=idx)
    same = pd.DataFrame({m: same["x"] for m in MODEL_NAMES}, index=idx)
    cs_same = cross_sectional_std_inline(same)
    assert np.allclose(cs_same.to_numpy(), 0.0)

    print("R-106 novel self-test: OK", file=sys.stderr)


_self_test()


if __name__ == "__main__":
    main()
