#!/usr/bin/env python
"""R-116 CONSERVATIVE branch: ``BreadthConfirmVoteKellyV4`` -- does the panel
breadth of agreement between the six OTHER Coinbase-listed instruments this
project already has 5m spot data for (BCH, LTC, ETC, DASH, LINK, XTZ), each
run through `kelly_regime_v4`'s own, byte-identical, unmodified 20/40/80-day
anchor-vote construction, and BTC's own vote-side, carry information that
should DISCOUNT (never boost) BTC's exposure when that breadth falls below a
majority? Full citations, the constraint attacked (INFO primary, ERR
secondary), and the complete non-duplication argument against R-76, the
xsmom lineage (R-63/65/67/68/107/110/111/113), R-100, B-05/R-35/R-39, R-41
and R-105/R-106 live in ``experiments/r116_shared.py``'s own module
docstring (read in full before this file was written) and are not
re-derived here beyond the one-paragraph summary above.

This file owns itself only: it reads r116_shared.py (frozen, neutral,
shared ground for a two-branch parallel round) and never edits it, never
touches the NOVEL branch's own files, and never touches src/ or docs/.

=====================================================================
PRE-REGISTRATION -- 2026-08-24, before any economic backtest number in this
file was computed. Anything below later contradicted by what actually
happened is stated in the results, never edited back into this banner.
=====================================================================

1. MECHANISM (one sentence): compute `agree_frac` (r116_shared, already
   built/tested) -- the fraction of the six alts whose OWN v4 vote-side
   agrees with BTC's OWN v4 vote-side, at every bar -- and multiply v4's
   unchanged `frac * scale` product by a bounded, monotonic,
   NEVER-INCREASE-ONLY step discount: full trust (1.0x) while the panel's
   agreement is at or above a majority `threshold`, discounted to a fixed
   `floor` otherwise. This is architecturally closer to R-105/R-106's
   `discount_from_disagreement`-style multiplicative brake on the FINAL
   scale than to `r82_shared.confirming_vote_frac` (R-53/R-55's rule),
   which blends an external discrete vote directly into the 3-anchor
   numerator (`(anchor_sum + weight*meta_vote) / (3+weight)`) -- a
   construction that can move the vote EITHER direction depending on
   `meta_vote`'s value, which is the wrong shape for "discount, never
   boost" (this project's own standing R-106 lesson). The discrete,
   step-function CONFIRMING-VOTE character the task asks for (full trust
   vs. discounted, gated by a majority threshold on a discrete 6-way
   count) is preserved; only the COMBINATION rule is the R-105/R-106
   multiplicative-brake shape rather than R-53/R-55's numerator-blend
   shape, and that choice is deliberate, not an oversight.

2. CONSTRUCTION (exact):

       votes[t, k]     = anchor_majority(alt_k)[t]   for k in the 6-alt panel,
                         PLUS BTC's own anchor_majority[t]  (r116_shared.build_panel_votes,
                         v4's byte-identical 20/40/80-day/1%-band construction, independently
                         per instrument, on the aligned 5m panel grid)
       agree[t]        = r116_shared.agree_frac(votes, home="BTC")[t]   in [0, 1]
       discount[t]      = 1.0                    if agree[t] >= threshold
                          = floor                 otherwise
       target[t]        = v4_targets(btc_df)[t] * discount[t]     (attached onto BTC's own
                                                                    native bar grid, causal
                                                                    reindex+ffill)

   NEUTRAL-FILL DEVIATION FROM `attach_to_btc`, DISCLOSED: `attach_to_btc`'s
   own convention fills bars before the panel's first aligned timestamp
   with 0.0, correct for a VOTE (0.0 = its own latched bearish default) or
   for `panel_disagreement` (0.0 = no disagreement, neutral). For an
   AGREEMENT FRACTION, 0.0 means "zero of six alts agree" -- maximal
   disagreement, NOT neutral. Using it here would silently apply the
   FLOOR discount for any bar before panel data exists, the opposite of
   "discount only on positive evidence of disagreement." This file
   therefore writes its own two-line reindex+ffill helper
   (`attach_agree`, identical mechanics, `fillna(1.0)`: neutral = full
   trust in v4 = no discount, until the panel actually says otherwise).
   In practice this edge case never binds on any window this file reads:
   W_TRAIN, W_VAL and the holdout are all >= the panel's own 2020-04-01
   coverage start.

3. STEP-0 GATE (frozen ahead of any economic number, r116_shared.r_squared,
   R-73's own standing artifact signature): `r_sq(agree, BTC's own v4
   vote)` and `r_sq(agree, BTC's own v4 EXPOSURE)`, both on W_TRAIN. KILL
   (stop before any economic backtest) if BOTH exceed 0.95 -- a flat
   rescale, not new information. Reported honestly either way.

4. CAUSALITY: `r116_shared.check_causality` (== `r63_shared.check_causality`,
   already causality-verified) applied to the PANEL half of the pipeline
   (`agree_frac` built from a truncated `aligned` dict vs. the full one);
   a second, independent truncation probe (`truncation_probe`, this file)
   applied to the FULL composed `build_target(df)` on real BTC data. Both
   must pass before any economic number is trusted.

5. CONFIG GRID, iterated ONLY on `r116_shared.W_TRAIN` (2020-04-01 ->
   2021-12-31), SPOT market: THRESHOLDS = {0.5 (bare majority, 3-of-6),
   4/6 (~0.667, clear majority), 5/6 (~0.833, near-unanimous)} x FLOORS =
   {0.3, 0.5, 0.7} -- 9 diagnostic cells, reported for bind_frac /
   non-degeneracy only (point estimates, NOT used to pick a winner -- a
   grid searched on train and then the best CELL carried to validation is
   exactly the goalpost-moving ROUTINE.md step 4 forbids). THREE DESIGNED
   CONFIGS are what is actually carried to W_VAL:
       C_A: threshold=4/6, floor=0.5   (the "moderate default")
       C_B: threshold=4/6, floor=0.3   (more aggressive discount)
       C_C: threshold=5/6, floor=0.5   (stricter trigger, same discount)

6. SELECTION RULE ON `r116_shared.W_VAL` (2022-01-01 -> 2022-12-31, SPOT),
   frozen before any W_VAL number was read: for each of C_A/C_B/C_C,
   compute `compare(candidate_equity, v4_targets_equity)` (paired
   stationary-block-bootstrap, r116_shared/r63_shared, 2,000 resamples).
   PRIMARY = the config with the HIGHEST `growth_diff` point estimate
   (ties broken toward C_A). This is a single deterministic rule with no
   discretion exercised after the numbers are seen.

7. FALSIFICATION TEST (pre-registered NOW, before it is run): does
   `agree`'s own regime transitions (a bar where agreement DECREASES,
   `r82_shared.nearest_transition(..., direction="down")`) lead BTC's own
   v4 vote's transitions (same test on `panel_votes["BTC"]`) on a MAJORITY
   of `r116_shared.STRESS_EPISODES`, beating a block-bootstrap null
   (`r82_shared.block_bootstrap_shifts`, 30-day blocks, 200 draws, seed 7)?
   DISCLOSED, PRE-REGISTERED CAVEAT: the 6-alt panel's own committed data
   starts 2020-04-01 (R-63); THREE of the six named episodes (2018-01-17,
   2018-12-15, 2020-03-12) predate that coverage entirely or are
   left-censored inside their +/-60-day window, so a lead is structurally
   impossible to observe for them regardless of the signal's true
   quality. The nominal ">=4/6" gate is therefore evaluated here, stated
   NOW rather than after seeing the count, as ">= majority of the
   episodes with actual panel coverage" (at most 3: 2021-11-10, 2022-05-09,
   2022-11-08), reported alongside the raw 6-episode table so the
   coverage gap is never hidden inside a passing number.

8. HOLDOUT DECISION RULE (frozen NOW, before `OOS_START=2023-01-01` is
   read anywhere in this file): run the frozen PRIMARY config exactly
   once, BTC spot (`MarketSpec.spot()`, 0.10%) and futures_5x
   (`MarketSpec.futures(leverage=5.0)`, 0.05%), `compare()` against both
   `v4_targets` and `buy_and_hold`. PROMOTE only if ALL of:
     - D1 (`d1_pass`: growth_diff > 0, 95% CI excludes zero) OR D2
       (`d2_pass`: dd_diff < 0, 95% CI excludes zero) vs `v4_targets`,
       on AT LEAST ONE market;
     - beats `buy_and_hold` out-of-sample after real costs on that same
       market (final balance and Sharpe);
     - the improvement exceeds the +/-0.2 Sharpe noise floor OR is a
       genuine drawdown/tail improvement;
     - survives the falsification test (majority of AVAILABLE episodes,
       beats the block-bootstrap null);
     - survives the real 0.40% spot taker tier and real futures funding
       charged over their observed overlap with the holdout
       (2023-01-01 -> 2023-12-31, the committed Binance funding file's own
       coverage) without a sign reversal;
     - the {threshold, floor} neighbourhood on the W_TRAIN diagnostic grid
       is a plateau, not an isolated peak.
   Anything else is NEGATIVE. Default is REJECT. If any number below
   causes this rule to be edited after being read, that is disclosed
   explicitly and the result is downgraded to in-sample -- per
   docs/ROUTINE.md step 4, this has NOT happened in this file.

----------------------------------------------------------------------
Run: python experiments/r116_conservative_breadth_vote.py
(from the repo root, with the project venv active)
----------------------------------------------------------------------
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r116_shared import (  # noqa: E402
    UNIVERSE_6,
    W_TRAIN,
    W_VAL,
    OOS_START,
    build_panel_votes,
    agree_frac,
    r_squared,
    v4_targets,
    check_causality,
    compare,
    d1_pass,
    d2_pass,
    d3_pass,
    note_config,
    config_count,
)
from experiments.r63_shared import align_frames, load_universe  # noqa: E402
from experiments.r82_shared import (  # noqa: E402
    STRESS_EPISODES,
    anchor_majority,
    nearest_transition,
    episode_window,
    block_bootstrap_shifts,
)

BARS_PER_DAY = 288
WARMUP = 80 * BARS_PER_DAY + 10   # v4's own anchor requirement, unmodified

UNIVERSE_7 = ("BTC",) + UNIVERSE_6      # the panel actually read: BTC + the six alts
PANEL_START = W_TRAIN[0]                 # 2020-04-01, the panel's own real coverage start

SPOT = MarketSpec.spot()                 # 0.10% taker
SPOT_REAL = MarketSpec.spot(fee_rate=0.004)   # 0.40% taker
FUTURES = MarketSpec.futures(leverage=5.0)    # 0.05% taker, funding-capable
FUNDING_HOLDOUT_END = "2023-12-31"       # committed Binance funding file's own coverage

THRESHOLDS = {"bare_majority(3/6)": 0.5, "majority(4/6)": 4.0 / 6.0, "near_unanimous(5/6)": 5.0 / 6.0}
FLOORS = (0.3, 0.5, 0.7)
DESIGNED_CONFIGS = {
    "C_A": dict(threshold_name="majority(4/6)", floor=0.5),
    "C_B": dict(threshold_name="majority(4/6)", floor=0.3),
    "C_C": dict(threshold_name="near_unanimous(5/6)", floor=0.5),
}


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ================================================================== (1)
# The panel signal: agree_frac, computed ONCE over its own full available
# history and cached, since it does not depend on which BTC window a given
# backtest reads (see docstring item 2's causality note).
# ==================================================================

_CACHE: dict[str, object] = {}


def panel_signal() -> tuple[pd.Series, pd.DataFrame]:
    if "agree" not in _CACHE:
        votes = build_panel_votes(tickers=UNIVERSE_7, window=(PANEL_START, None))
        _CACHE["votes"] = votes
        _CACHE["agree"] = agree_frac(votes, home="BTC")
    return _CACHE["agree"], _CACHE["votes"]


def attach_agree(btc_df: pd.DataFrame, agree: pd.Series) -> pd.Series:
    """Reindex `agree` onto `btc_df`'s own native index, causal
    reindex+ffill (identical mechanics to `r116_shared.attach_to_btc`),
    but with fillna(1.0) -- neutral = full trust in v4, not 0.0 = maximal
    disagreement. See docstring item 2 for why this deliberately deviates
    from `attach_to_btc`'s own fill convention."""
    return agree.reindex(btc_df.index, method="ffill").fillna(1.0)


def discount_from_agree(agree_on_btc: np.ndarray, threshold: float, floor: float) -> np.ndarray:
    """Bounded {floor, 1.0} step discount. Monotonic non-increasing in
    disagreement by construction -- never raises exposure above 1.0x."""
    return np.where(agree_on_btc >= threshold, 1.0, floor)


def make_build_target(threshold: float, floor: float):
    def build_target(df: pd.DataFrame) -> np.ndarray:
        base = v4_targets(df).to_numpy(dtype=float)
        agree, _ = panel_signal()
        a_on_btc = attach_agree(df, agree).to_numpy(dtype=float)
        disc = discount_from_agree(a_on_btc, threshold, floor)
        return base * disc
    build_target.__name__ = f"breadth_confirm_t{threshold:.3f}_f{floor:g}"
    return build_target


class TargetStrategy(Strategy):
    """Wrap a pure ``build_target(df) -> np.ndarray`` as a runnable
    strategy -- the pattern at experiments/r102_shared.py:510, duplicated
    here per this project's own per-round-file convention (not imported
    across rounds)."""

    name = "r116_conservative_control"
    warmup = WARMUP

    def __init__(self, build_target, name: str = "r116_conservative_control",
                warmup: int | None = None) -> None:
        self._build = build_target
        self.name = name
        if warmup is not None:
            self.warmup = warmup

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["target"] = np.asarray(self._build(df), dtype=float)
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


def truncation_probe(build_target, df: pd.DataFrame, cut_from_end: int = 20_000) -> bool:
    """Truncation probe on the FULL composed `build_target(df)`: does the
    target change, for rows before the cut, if bars after it are dropped?"""
    idx = df.index
    cut = len(idx) - cut_from_end
    if cut <= 1:
        raise ValueError("frame too short for the truncation probe")
    full = np.asarray(build_target(df), dtype=float)
    trunc = np.asarray(build_target(df.iloc[:cut].copy()), dtype=float)
    m = min(cut, len(trunc))
    a = np.nan_to_num(full[:m], nan=0.0)
    b = np.nan_to_num(trunc[:m], nan=0.0)
    return bool(np.allclose(a, b, atol=1e-9, rtol=0.0))


# ================================================================== (2)
# Backtest runners.
# ==================================================================

def run_target(build_target, df: pd.DataFrame, start, end, market: MarketSpec,
               name: str, warmup: int = WARMUP):
    note_config()
    strat = TargetStrategy(build_target, name=name, warmup=warmup)
    return run_period(strat, df, start, end, market=market, start_balance=1_000.0)


def run_target_funding(build_target, df: pd.DataFrame, start, end, market: MarketSpec,
                       name: str, funding: pd.Series, warmup: int = WARMUP):
    """Like run_target, but with real funding charged -- run_period has no
    `funding` kwarg, so this reproduces its warmup-prefix logic by hand,
    the same pattern scripts/funding_study.py's own `_period` uses."""
    note_config()
    strat = TargetStrategy(build_target, name=name, warmup=warmup)
    lo = 0 if start is None else int(df.index.searchsorted(pd.Timestamp(start, tz="UTC")))
    hi = (len(df) if end is None
         else int(df.index.searchsorted(pd.Timestamp(end, tz="UTC"), side="right")))
    prefix = min(lo, strat.warmup)
    frame = df.iloc[lo - prefix: hi]
    raw = run_backtest(strat, frame, market, 1_000.0, trade_start=prefix, funding=funding)
    return raw if prefix == 0 else replace(raw, equity=raw.equity.iloc[prefix:],
                                           df=raw.df.iloc[prefix:])


def summarize(res) -> dict:
    m = compute_metrics(res)
    return dict(final=m.final_balance, sharpe=m.sharpe, dd=m.max_drawdown_pct,
               trades=m.num_trades, equity=res.equity)


# ================================================================== (3)
# STEP 0 -- collinearity gate.
# ==================================================================

def step0(df_btc: pd.DataFrame) -> dict:
    agree, votes = panel_signal()
    idx = agree.loc[W_TRAIN[0]:W_TRAIN[1]].index
    agree_train = agree.loc[idx]
    btc_vote_train = votes["BTC"].loc[idx]

    r2_vs_vote = r_squared(agree_train, btc_vote_train)

    v4_full = v4_targets(df_btc)
    v4_on_panel = v4_full.reindex(idx, method="ffill")
    r2_vs_exposure = r_squared(agree_train, v4_on_panel)

    killed = r2_vs_vote > 0.95 and r2_vs_exposure > 0.95
    return dict(r2_vs_vote=r2_vs_vote, r2_vs_exposure=r2_vs_exposure, killed=killed,
               n_bars=len(idx))


# ================================================================== (4)
# Causality checks.
# ==================================================================

def run_causality_checks(df_btc: pd.DataFrame) -> dict:
    frames = load_universe(UNIVERSE_7)
    aligned = align_frames(frames, (PANEL_START, None))

    def _panel_agree_from_aligned(aligned_dict):
        votes = {t: anchor_majority(aligned_dict[t]) for t in UNIVERSE_7}
        votes_df = pd.DataFrame(votes)
        return pd.DataFrame({"agree": agree_frac(votes_df, home="BTC")})

    panel_ok = check_causality(_panel_agree_from_aligned, aligned, cut_from_end=20_000)

    primary_build = make_build_target(THRESHOLDS["majority(4/6)"], 0.5)
    full_ok = truncation_probe(primary_build, df_btc, cut_from_end=20_000)

    return dict(panel_causal_ok=panel_ok, full_composed_causal_ok=full_ok)


# ================================================================== (5)
# Train-only diagnostic grid (bind_frac / non-degeneracy; NOT used to pick
# a winner -- see pre-registration item 5).
# ==================================================================

def diagnostic_grid(df_btc: pd.DataFrame) -> list[dict]:
    rows = []
    bench = run_target(v4_targets, df_btc, W_TRAIN[0], W_TRAIN[1], SPOT, "v4_bench")
    bench_m = summarize(bench)
    agree, _ = panel_signal()
    a_on_btc = attach_agree(df_btc, agree)   # same for every cell; computed once

    for tname, thresh in THRESHOLDS.items():
        for floor in FLOORS:
            build = make_build_target(thresh, floor)
            res = run_target(build, df_btc, W_TRAIN[0], W_TRAIN[1], SPOT,
                             f"diag_{tname}_{floor:g}")
            m = summarize(res)
            disc_train = discount_from_agree(
                a_on_btc.reindex(res.equity.index).to_numpy(), thresh, floor)
            bind_frac = float(np.mean(disc_train < 1.0))
            rows.append(dict(threshold_name=tname, threshold=thresh, floor=floor,
                             bind_frac=bind_frac, cand_final=m["final"], cand_sharpe=m["sharpe"],
                             cand_dd=m["dd"], bench_final=bench_m["final"],
                             bench_sharpe=bench_m["sharpe"], bench_dd=bench_m["dd"]))
    return rows


def print_diagnostic_grid(rows: list[dict]) -> None:
    hdr = (f"{'threshold':>18s} {'floor':>6s} {'bind_frac':>10s} {'cand_final':>12s} "
          f"{'cand_Sh':>8s} {'cand_DD%':>9s} {'bench_final':>12s} {'bench_Sh':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['threshold_name']:>18s} {r['floor']:6.2f} {r['bind_frac']:10.3f} "
              f"${r['cand_final']:>10,.0f} {r['cand_sharpe']:8.2f} {r['cand_dd']:9.1f} "
              f"${r['bench_final']:>10,.0f} {r['bench_sharpe']:9.2f}")


# ================================================================== (6)
# W_VAL selection among the three designed configs.
# ==================================================================

def select_primary(df_btc: pd.DataFrame) -> dict:
    bench = run_target(v4_targets, df_btc, W_VAL[0], W_VAL[1], SPOT, "v4_bench_val")
    rows = []
    for cname, cfg in DESIGNED_CONFIGS.items():
        thresh = THRESHOLDS[cfg["threshold_name"]]
        floor = cfg["floor"]
        build = make_build_target(thresh, floor)
        cand = run_target(build, df_btc, W_VAL[0], W_VAL[1], SPOT, f"val_{cname}")
        row = compare(cand.equity, bench.equity)
        row.update(config=cname, threshold_name=cfg["threshold_name"], threshold=thresh,
                   floor=floor, d1=d1_pass(row), d2=d2_pass(row), d3=d3_pass(row))
        rows.append(row)

    primary = max(rows, key=lambda r: (r["growth_diff"], r["config"] == "C_A"))
    return dict(rows=rows, primary=primary)


def print_val_rows(rows: list[dict]) -> None:
    hdr = (f"{'config':>6s} {'threshold':>18s} {'floor':>6s} {'growth_diff':>12s} "
          f"{'[lo,hi]':>22s} {'dd_diff':>9s} {'[lo,hi]':>20s} {'D1':>5s} {'D2':>5s} {'D3':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['config']:>6s} {r['threshold_name']:>18s} {r['floor']:6.2f} "
              f"{r['growth_diff']:+12.4f} [{r['growth_lo']:+.3f},{r['growth_hi']:+.3f}]  "
              f"{r['dd_diff']:+9.4f} [{r['dd_lo']:+.3f},{r['dd_hi']:+.3f}]  "
              f"{'PASS' if r['d1'] else 'no':>5s} {'PASS' if r['d2'] else 'no':>5s} "
              f"{'PASS' if r['d3'] else 'no':>5s}")


# ================================================================== (7)
# Falsification test: six-episode lead-time gate + block-bootstrap null.
# ==================================================================

def _lead_count(agree_series: pd.Series, btc_vote: pd.Series, episodes) -> tuple[int, int, list[dict]]:
    n_lead, n_used = 0, 0
    details = []
    for label, onset_str in episodes:
        onset, window = episode_window(agree_series.to_frame(), onset_str, window_days=60)
        if len(window) < 10:
            details.append(dict(label=label, onset=onset_str, coverage=False, led=None))
            continue
        t_a = nearest_transition(agree_series, window, onset, direction="down")
        t_b = nearest_transition(btc_vote, window, onset, direction="down")
        if t_a is None or t_b is None:
            details.append(dict(label=label, onset=onset_str, coverage=True, found=False, led=None))
            continue
        led = bool(t_a <= t_b)
        n_used += 1
        n_lead += int(led)
        details.append(dict(label=label, onset=onset_str, coverage=True, found=True,
                            t_agree=t_a, t_vote=t_b, lead_bars=(t_b - t_a) / pd.Timedelta(minutes=5),
                            led=led))
    return n_lead, n_used, details


def falsification_test() -> dict:
    agree, votes = panel_signal()
    btc_vote = votes["BTC"]

    n_lead, n_used, details = _lead_count(agree, btc_vote, STRESS_EPISODES)

    shifts = block_bootstrap_shifts(len(agree), block_days=30, n_draws=200, seed=7)
    null_leads = []
    for shift in shifts:
        shifted = pd.Series(agree.to_numpy()[shift], index=agree.index)
        nl, nu, _ = _lead_count(shifted, btc_vote, STRESS_EPISODES)
        if nu > 0:
            null_leads.append(nl)
    null_leads = np.asarray(null_leads, dtype=float)

    majority_available = n_used > 0 and n_lead >= int(np.ceil(n_used / 2.0))
    beats_null = len(null_leads) > 0 and n_lead > float(np.percentile(null_leads, 90))
    passed = majority_available and beats_null

    return dict(n_lead=n_lead, n_used=n_used, n_total_episodes=len(STRESS_EPISODES),
               details=details, null_mean=float(np.mean(null_leads)) if len(null_leads) else float("nan"),
               null_p90=float(np.percentile(null_leads, 90)) if len(null_leads) else float("nan"),
               n_null_draws=len(null_leads), majority_available=majority_available,
               beats_null=beats_null, passed=passed)


# ================================================================== (8)
# Holdout.
# ==================================================================

def load_btc_full() -> pd.DataFrame:
    df, _label = load_dataset(ROOT / "data", "spot")
    return df


def run_holdout(primary: dict, df_btc_full: pd.DataFrame) -> dict:
    thresh, floor = primary["threshold"], primary["floor"]
    build = make_build_target(thresh, floor)
    result = {"market": {}}

    for market, mname in ((SPOT, "spot"), (FUTURES, "futures_5x")):
        cand = run_target(build, df_btc_full, OOS_START, None, market, f"holdout_cand_{mname}")
        bench = run_target(v4_targets, df_btc_full, OOS_START, None, market, f"holdout_v4_{mname}")
        note_config()
        hold = run_period(get_strategy("buy_and_hold"), df_btc_full, OOS_START, None,
                          market=market, start_balance=1_000.0)
        vs_v4 = compare(cand.equity, bench.equity)
        vs_hold = compare(cand.equity, hold.equity)
        cand_m, bench_m, hold_m = summarize(cand), summarize(bench), summarize(hold)
        result["market"][mname] = dict(
            cand=cand_m, bench=bench_m, hold=hold_m,
            vs_v4=vs_v4, vs_hold=vs_hold,
            d1=d1_pass(vs_v4), d2=d2_pass(vs_v4),
            beats_hold_final=cand_m["final"] > hold_m["final"],
            beats_hold_sharpe=cand_m["sharpe"] > hold_m["sharpe"],
            sharpe_edge_vs_v4=cand_m["sharpe"] - bench_m["sharpe"],
        )

    # -- real fee tier (0.40% spot taker)
    real_fee = {}
    for mkt, mname in ((SPOT_REAL, "spot@0.40%"),):
        cand = run_target(build, df_btc_full, OOS_START, None, mkt, f"holdout_cand_{mname}")
        bench = run_target(v4_targets, df_btc_full, OOS_START, None, mkt, f"holdout_v4_{mname}")
        cand_m, bench_m = summarize(cand), summarize(bench)
        real_fee[mname] = dict(cand=cand_m, bench=bench_m,
                               sharpe_edge=cand_m["sharpe"] - bench_m["sharpe"])

    # -- real funding on futures, restricted to the observed overlap
    funding = load_funding(ROOT / "data")
    funding_result = None
    if funding is not None:
        cand_free = run_target(build, df_btc_full, OOS_START, FUNDING_HOLDOUT_END, FUTURES,
                               "holdout_cand_fut_free")
        cand_paid = run_target_funding(build, df_btc_full, OOS_START, FUNDING_HOLDOUT_END, FUTURES,
                                       "holdout_cand_fut_paid", funding)
        bench_free = run_target(v4_targets, df_btc_full, OOS_START, FUNDING_HOLDOUT_END, FUTURES,
                                "holdout_v4_fut_free")
        bench_paid = run_target_funding(v4_targets, df_btc_full, OOS_START, FUNDING_HOLDOUT_END,
                                        FUTURES, "holdout_v4_fut_paid", funding)
        cf, cp, bf, bp = (summarize(cand_free), summarize(cand_paid),
                          summarize(bench_free), summarize(bench_paid))
        d_sharpe_free = cf["sharpe"] - bf["sharpe"]
        d_sharpe_paid = cp["sharpe"] - bp["sharpe"]
        funding_result = dict(cand_free=cf, cand_paid=cp, bench_free=bf, bench_paid=bp,
                              d_sharpe_free=d_sharpe_free, d_sharpe_paid=d_sharpe_paid,
                              no_reversal=bool(np.sign(d_sharpe_free) == np.sign(d_sharpe_paid)
                                              or d_sharpe_free == 0 or d_sharpe_paid == 0))

    return dict(market=result["market"], real_fee=real_fee, funding=funding_result)


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()

    hr("R-116 CONSERVATIVE: BreadthConfirmVoteKellyV4 -- 6-alt panel breadth of "
      "agreement with BTC's own v4 vote, as a discrete, never-boost, majority-gated "
      "discount on v4's unchanged frac*scale")

    df_btc = load_btc_full()
    df_btc_train_only = df_btc[df_btc.index < pd.Timestamp(OOS_START, tz="UTC")]
    print(f"BTC spot: {len(df_btc):,} bars, {df_btc.index[0]} -> {df_btc.index[-1]}")

    hr("STEP 0 -- COLLINEARITY GATE (r_sq of agree_frac vs BTC's own v4 vote/exposure, W_TRAIN)")
    s0 = step0(df_btc_train_only)
    print(f"  r_sq(agree, BTC v4 vote)     = {s0['r2_vs_vote']:.4f}")
    print(f"  r_sq(agree, BTC v4 exposure) = {s0['r2_vs_exposure']:.4f}   (n={s0['n_bars']:,} bars)")
    print(f"  KILL (both > 0.95, R-73's own flat-rescale signature): {s0['killed']}")

    hr("CAUSALITY CHECKS (before any economic backtest is trusted)")
    causal = run_causality_checks(df_btc_train_only)
    print(f"  panel pipeline truncation probe (check_causality, r63_shared):  "
          f"{'PASS' if causal['panel_causal_ok'] else 'FAIL'}")
    print(f"  full composed build_target truncation probe (this file's own): "
          f"{'PASS' if causal['full_composed_causal_ok'] else 'FAIL'}")
    causal_ok = causal["panel_causal_ok"] and causal["full_composed_causal_ok"]

    if s0["killed"] or not causal_ok:
        reason = "Step-0 kill (near-total collinearity)" if s0["killed"] else "causality FAIL"
        hr(f"STOPPING HERE: {reason}")
        print(f"n_configs so far: {config_count()}")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(verdict=f"NEGATIVE ({reason})", step0=s0, causal=causal,
                   n_configs=config_count())

    hr("TRAIN-ONLY DIAGNOSTIC GRID (bind_frac / non-degeneracy; NOT used to pick a winner -- "
      "3 thresholds x 3 floors, W_TRAIN, spot)")
    diag_rows = diagnostic_grid(df_btc_train_only)
    print_diagnostic_grid(diag_rows)

    hr("W_VAL SELECTION -- three DESIGNED configs (C_A/C_B/C_C), compare() vs v4_targets")
    sel = select_primary(df_btc_train_only)
    print_val_rows(sel["rows"])
    primary = sel["primary"]
    print(f"\nPRIMARY SELECTED (highest growth_diff point estimate, ties to C_A): "
          f"{primary['config']} (threshold={primary['threshold_name']}={primary['threshold']:.4f}, "
          f"floor={primary['floor']:g})")
    print(f"  D1={primary['d1']}  D2={primary['d2']}  D3={primary['d3']}")

    hr("PLATEAU CHECK -- primary's neighbourhood on the W_TRAIN diagnostic grid")
    by_key = {(r["threshold_name"], r["floor"]): r for r in diag_rows}
    prim_key = (primary["threshold_name"], primary["floor"])
    prim_row = by_key[prim_key]
    prim_edge = prim_row["cand_sharpe"] - prim_row["bench_sharpe"]
    same_sign = []
    for floor in FLOORS:
        if floor == primary["floor"]:
            continue
        r = by_key[(primary["threshold_name"], floor)]
        edge = r["cand_sharpe"] - r["bench_sharpe"]
        same_sign.append(np.sign(edge) == np.sign(prim_edge))
        print(f"  floor={floor:g} (same threshold): sharpe_edge={edge:+.3f}  "
              f"(primary={prim_edge:+.3f})  same_sign={np.sign(edge) == np.sign(prim_edge)}")
    plateau_ok = all(same_sign) if same_sign else False
    print(f"  plateau (all floor-neighbours share primary's sign): {plateau_ok}")

    hr("FALSIFICATION TEST -- does agree's own transition lead BTC's own v4 vote's "
      "transition on a majority of the STRESS_EPISODES with real panel coverage?")
    fals = falsification_test()
    for d in fals["details"]:
        if not d["coverage"]:
            print(f"  {d['label']:>42s} ({d['onset']}): NO PANEL COVERAGE (pre-2020-04-01) -- excluded")
        elif not d.get("found", False):
            print(f"  {d['label']:>42s} ({d['onset']}): covered, no transition found in +/-60d window")
        else:
            print(f"  {d['label']:>42s} ({d['onset']}): agree@{d['t_agree']}  vote@{d['t_vote']}  "
                  f"lead_bars={d['lead_bars']:+.0f}  LED={d['led']}")
    print(f"\n  real: {fals['n_lead']}/{fals['n_used']} available episodes led "
          f"(of {fals['n_total_episodes']} named, {fals['n_used']} had panel coverage)")
    print(f"  null (block-bootstrap, {fals['n_null_draws']} draws): mean={fals['null_mean']:.2f}  "
          f"p90={fals['null_p90']:.2f}")
    print(f"  majority-of-available: {fals['majority_available']}   beats null p90: {fals['beats_null']}")
    print(f"  FALSIFICATION TEST PASSED (both clauses): {fals['passed']}")

    hr(f"HOLDOUT (frozen primary: threshold={primary['threshold_name']}, floor={primary['floor']:g}) "
      f"-- run exactly once, per the frozen decision rule in this file's own pre-registration")
    holdout = run_holdout(primary, df_btc)

    for mname, r in holdout["market"].items():
        print(f"\n  -- {mname} --")
        print(f"  candidate: final=${r['cand']['final']:>10,.0f} Sharpe={r['cand']['sharpe']:+.2f} "
              f"DD={r['cand']['dd']:5.1f}%  trades={r['cand']['trades']}")
        print(f"  v4 bench:  final=${r['bench']['final']:>10,.0f} Sharpe={r['bench']['sharpe']:+.2f} "
              f"DD={r['bench']['dd']:5.1f}%")
        print(f"  buy&hold:  final=${r['hold']['final']:>10,.0f} Sharpe={r['hold']['sharpe']:+.2f} "
              f"DD={r['hold']['dd']:5.1f}%")
        v = r["vs_v4"]
        h = r["vs_hold"]
        print(f"  vs v4:    growth_diff={v['growth_diff']:+.4f} [{v['growth_lo']:+.3f},{v['growth_hi']:+.3f}]  "
              f"dd_diff={v['dd_diff']:+.4f} [{v['dd_lo']:+.3f},{v['dd_hi']:+.3f}]  "
              f"D1={r['d1']}  D2={r['d2']}")
        print(f"  vs hold:  growth_diff={h['growth_diff']:+.4f} [{h['growth_lo']:+.3f},{h['growth_hi']:+.3f}]  "
              f"beats_hold_final={r['beats_hold_final']}  beats_hold_sharpe={r['beats_hold_sharpe']}")
        print(f"  sharpe_edge_vs_v4={r['sharpe_edge_vs_v4']:+.3f}  "
              f"clears +/-0.2 noise floor: {abs(r['sharpe_edge_vs_v4']) > 0.2}")

    hr("REAL COSTS -- 0.40% spot taker tier")
    for mname, r in holdout["real_fee"].items():
        print(f"  {mname:>12s}  candidate=${r['cand']['final']:>10,.0f} (Sh {r['cand']['sharpe']:+.2f})  "
              f"v4=${r['bench']['final']:>10,.0f} (Sh {r['bench']['sharpe']:+.2f})  "
              f"sharpe_edge={r['sharpe_edge']:+.3f}")
    base_edge = holdout["market"]["spot"]["sharpe_edge_vs_v4"]
    real_edge = holdout["real_fee"]["spot@0.40%"]["sharpe_edge"]
    print(f"  sign preserved (0.10% -> 0.40%): {np.sign(base_edge) == np.sign(real_edge)}")

    hr(f"REAL COSTS -- futures funding charged ({OOS_START} -> {FUNDING_HOLDOUT_END}, "
      f"committed Binance coverage)")
    if holdout["funding"] is None:
        print("  no funding data committed -- skipped")
        funding_no_reversal = None
    else:
        f = holdout["funding"]
        print(f"  candidate: funding-free ${f['cand_free']['final']:>10,.0f} (Sh {f['cand_free']['sharpe']:+.2f})  "
              f"funding-charged ${f['cand_paid']['final']:>10,.0f} (Sh {f['cand_paid']['sharpe']:+.2f})")
        print(f"  v4 bench:  funding-free ${f['bench_free']['final']:>10,.0f} (Sh {f['bench_free']['sharpe']:+.2f})  "
              f"funding-charged ${f['bench_paid']['final']:>10,.0f} (Sh {f['bench_paid']['sharpe']:+.2f})")
        print(f"  sharpe_edge: funding-free={f['d_sharpe_free']:+.3f}  funding-charged={f['d_sharpe_paid']:+.3f}  "
              f"sign preserved={f['no_reversal']}")
        funding_no_reversal = f["no_reversal"]

    hr("VERDICT")
    any_market_d1_or_d2 = any(r["d1"] or r["d2"] for r in holdout["market"].values())
    any_beats_hold = any(r["beats_hold_final"] and r["beats_hold_sharpe"]
                         for r in holdout["market"].values())
    any_noise_floor = any(abs(r["sharpe_edge_vs_v4"]) > 0.2 for r in holdout["market"].values())
    fee_ok = np.sign(base_edge) == np.sign(real_edge)
    funding_ok = funding_no_reversal if funding_no_reversal is not None else True

    promote = (any_market_d1_or_d2 and any_beats_hold and any_noise_floor
              and fals["passed"] and fee_ok and funding_ok and plateau_ok)
    verdict = "PROMOTE" if promote else "NEGATIVE"

    print(f"  D1 or D2 (>=1 market) vs v4:            {any_market_d1_or_d2}")
    print(f"  beats buy_and_hold (>=1 market):        {any_beats_hold}")
    print(f"  clears +/-0.2 Sharpe noise floor:       {any_noise_floor}")
    print(f"  falsification test passed:              {fals['passed']}")
    print(f"  survives 0.40% fee tier (no reversal):  {fee_ok}")
    print(f"  survives real funding (no reversal):    {funding_ok}")
    print(f"  plateau, not peak:                      {plateau_ok}")
    print(f"\n  VERDICT: {verdict}  (default is REJECT; ALL clauses required to PROMOTE)")

    n_configs = config_count()
    print(f"\nconfigurations evaluated (this branch, this process): {n_configs}")
    print(f"[{time.time() - t0:.0f}s]")

    return dict(step0=s0, causal=causal, diag_rows=diag_rows, val_rows=sel["rows"],
               primary=primary, plateau_ok=plateau_ok, falsification=fals, holdout=holdout,
               verdict=verdict, n_configs=n_configs)


if __name__ == "__main__":
    main()
