"""R-76 (novel branch): distance-method pairs trading with Kelly/vol-target sizing.

=====================================================================
WHAT THIS FILE IS
=====================================================================

Direction (see the R-76 prompt / docs/LEDGER.md's standing diagnosis):
statistical-arbitrage / pairs trading between two of this project's own
committed price series -- the first round to make a CROSS-INSTRUMENT
RELATIONSHIP the tradeable signal itself, rather than feeding an external
series into kelly_regime_v4's single-instrument gate (six prior INFO-axis
attempts: on-chain B-07/R-44, macro VIX/DXY R-53/R-54, stablecoin flow
R-54/R-55/R-58, DVOL/VRP R-73, MVRV R-74, calendar/session R-75) or
trading each instrument off its own single-series signal at a fixed
capital split (four prior multi-asset rounds: B-16/B-19/B-20, R-63, R-61,
R-57). Nothing in this project's history has ever computed or traded the
SPREAD between two instruments before.

Citations: Gatev, Goetzmann & Rouwenhorst (2006, Review of Financial
Studies 19(3):797-827) -- the distance method: match pairs by minimum
sum-of-squared-deviations between normalized price paths in a formation
period, deliberately without cointegration. Fil & Kristoufek (2020, IEEE
Access 8:172644-172651) -- crypto pairs application; distance beats
cointegration for crypto, and documents 5-minute mean-reversion "missing
in daily data" -- why this screen runs at native 5-minute resolution.

Mechanism, novel to this branch relative to R-76's disjoint conservative
sibling (not read, not coordinated with, per docs/ROUTINE.md): pair
SELECTION by the distance method (no regression, no fitted hedge ratio),
combined with this project's own validated fractional-Kelly / vol-target
sizing SHAPE, reused mechanically from kelly_regime_v3.prepare()
(``scale = min(target_vol / realized_vol, max_leverage)``) -- applied
here to the SPREAD's own realized volatility instead of a single asset's.
Continuous, |z|-proportional sizing (not a binary in/out gate) is this
branch's point of departure from the conservative sibling.

Universe (identical 8 instruments to the conservative branch, stated
redundantly per the disjoint-branch rule -- no coordination, no read of
its file): BTC (Bitstamp spot, ``tradebot.data.load_dataset(..,"spot")``),
ETH/BCH/DASH/ETC/LINK/LTC/XTZ (Coinbase spot,
``tradebot.data.load_coinbase_spot``). VENUE CAVEAT, named explicitly:
BTC is Bitstamp-sourced, the other 7 are Coinbase-sourced -- a genuine
cross-venue basis could add noise to any BTC-leg pair's spread. This is
not corrected for; it is a real property of the data this file uses.

OOS_START = "2023-01-01". This file never reads, prints, or holds in
memory any bar timestamped on or after that date, from any of the 8
files: every loader truncates immediately after ``load_*`` returns and
before any other operation touches the frame (mirroring
experiments/r75_novel_session_vol_gate.py's own pattern of loading the
full file and slicing before further use), ``assert_no_holdout`` is
applied to every truncated frame, and the max timestamp actually read
from each of the 8 files is printed by the ``report`` command.

Step A (mandatory, pre-registered, run before any strategy code): for
every one of the 28 unique pairs with >=250 overlapping formation-period
(2017-01-01 -> 2020-12-31, restricted per-pair to that pair's own data
overlap) daily-equivalent 5-minute observations, normalize each leg to
1.0 at the pair's own overlap start, compute the sum of squared
deviations between the normalized paths, rank ascending. Null test on the
single minimum-distance pair only: block-bootstrap (1-day / 288-bar
contiguous blocks, with replacement, seed=760, 1000 reps) one leg's own
log-return series into a synthetic price path, recompute its distance
against the OTHER, real, unshuffled leg; p = fraction of null distances
<= the observed minimum. PRE-REGISTERED STOP RULE, fixed now: proceed to
Step B only if p < 0.05. If it fails, STOP -- the ranked table is the
round's entire result. The screen itself (28 pairs, 1 bootstrap test) is
a fixed, non-swept measurement: 0 configurations, this project's standing
convention for this exact kind of gate study (R-53/R-54/R-73/R-74/R-75's
own Step-A studies all contribute 0 the same way).

Step B (only if the gate passes): the literal distance-method spread
(simple difference of the two legs' normalized price paths, no OLS, no
fitted hedge ratio), dollar-neutral (equal notional both legs, 50/50
capital split via tradebot.multiasset.run_multi_backtest), z-scored
against a rolling 30-day window (structural, not swept -- matches the
conservative branch's own structural pick), sized by
``scale = min(target_vol / spread_vol, max_leverage)`` where
``spread_vol`` is an EWM-vol of the spread's own first difference using
kelly_regime_v3's own ``vol_span`` convention (8 * BARS_PER_DAY), and
``target_vol``/``max_leverage`` are kelly_regime_v4's own shipped
defaults (0.55 / 2.0), reused VERBATIM AND UNSWEPT -- a structural reuse,
not a refit, exactly as R-62's isolation studies reused v4's shipped
defaults unchanged. Continuous sizing: exposure ramps linearly from 0 at
z=0 to full ``scale`` at a pre-registered max-|z| cap, one of exactly
{2, 4, 6} -- the branch's entire trials count if Step B runs (3
configurations). Two Strategy instances of one parametrized class,
opposite sign per leg, each reading a precomputed target column joined
CAUSALLY onto that leg's own native bar index (reindex/union/sort/ffill,
the same construction as ``align_onchain_causal`` etc. minus their +1-day
publication-lag shift, since both legs already share one 5-minute UTC
clock) -- see ``align_pair_signal_causal``. ``MarketSpec.futures()`` for
both legs (needed for the short leg); benchmark is a 50/50
``buy_and_hold`` of the same two legs on spot, NOT single-asset
buy_and_hold (the risk-matched comparison for a market-neutral book).

IMPLEMENTATION NOTE, stated plainly rather than silently: the spec calls
for ``ctx.order_target(sign * scale * ramp(|z|))``, not
``ctx.order_notional(...)`` (kelly_regime's own convention). ``order_target``
interprets its argument as a fraction of MAX NOTIONAL (equity x
market.leverage), clamped hard to [-1, 1] inside the broker, not as a
fraction of equity independent of leverage. With ``MarketSpec.futures()``
defaulting to leverage=5x and ``scale`` able to reach ``max_leverage=2.0``,
a ramped-up signal very often saturates the clamp at |target|=1.0, i.e.
the leg sits at its full 5x-equity notional cap rather than the 2x-equity
notional a ``order_notional`` reading of the same ``scale`` would produce.
This is a real, structural consequence of following the spec literally
(not a bug), and it means the realized notional leverage per leg here is
materially higher than kelly_regime_v4's own single-asset convention --
worth weighing when reading the growth/drawdown numbers below.

Run: ``python experiments/r76_novel_pairs_distance_kelly.py <cmd>``
Commands: screen | stepb | falsify | report | all (default)
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_spot, load_dataset  # noqa: E402
from tradebot.multiasset import MultiAssetSpec, run_multi_backtest  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

# ============================================================== constants

OOS_START = "2023-01-01"
FORMATION_START = "2017-01-01"
FORMATION_END = "2020-12-31 23:55:00"
TRAIN_END = "2020-12-31 23:55:00"
VALID_START = "2021-01-01"
VALID_END = "2022-12-31 23:55:00"

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

MIN_FORMATION_DAYS = 250          # "~1 year" of overlapping formation bars
ROLL_WINDOW = 30 * BARS_PER_DAY   # 30-day z-score window (structural, unswept)
VOL_SPAN = 8 * BARS_PER_DAY       # kelly_regime_v3's own vol_span convention
TARGET_VOL = 0.55                 # kelly_regime_v4 shipped default, verbatim
MAX_LEVERAGE = 2.0                # kelly_regime_v4 shipped default, verbatim
RAMP_CAPS = (2.0, 4.0, 6.0)       # pre-registered, fixed -- the branch's entire trials count

BOOT_SEED = 760
N_BOOT = 1000

START_BALANCE = 1_000.0

# Fixed BEFORE the screen runs, so the bootstrapped leg of the eventual
# winning pair cannot be picked after the fact: itertools.combinations
# over this list yields (a, b) with `a` always the earlier-listed
# instrument, and `a` is always the leg whose return series is
# block-bootstrapped in the Step-A null test.
UNIVERSE = ["BTC", "ETH", "BCH", "DASH", "ETC", "LINK", "LTC", "XTZ"]

REPORTS_DIR = ROOT / "experiments" / "reports"

_MAX_TS_READ: dict[str, pd.Timestamp] = {}


# ================================================================== loading

def assert_no_holdout(df: pd.DataFrame) -> None:
    """Every frame this file touches must have max timestamp < OOS_START."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


_LEG_CACHE: dict[str, pd.DataFrame] = {}


def load_leg(name: str) -> pd.DataFrame:
    """Load one instrument's full committed history, then IMMEDIATELY
    truncate to strictly before OOS_START -- before any other operation
    touches the frame, mirroring r75_novel_session_vol_gate.py's own
    load-then-slice pattern. Cached per name (each of the 28 pairs reuses
    the same 8 loaded legs rather than re-reading the CSV 7 times each).
    """
    if name in _LEG_CACHE:
        return _LEG_CACHE[name]
    if name == "BTC":
        full, _label = load_dataset(ROOT / "data", "spot")
    else:
        full = load_coinbase_spot(ROOT / "data", name)
        if full is None:
            raise RuntimeError(f"data file missing for {name} -- cannot run this experiment")
    cutoff = pd.Timestamp(OOS_START, tz=full.index.tz)
    df = full.loc[full.index < cutoff].copy()
    del full
    assert_no_holdout(df)
    _MAX_TS_READ[name] = df.index.max()
    _LEG_CACHE[name] = df
    return df


def load_all_legs() -> dict[str, pd.DataFrame]:
    return {n: load_leg(n) for n in UNIVERSE}


# ============================================================ step A: screen

def pair_overlap_formation(a_df: pd.DataFrame, b_df: pd.DataFrame) -> pd.DatetimeIndex:
    """Overlapping formation-period (2017-2020, per-pair data availability)
    5-minute bars: the intersection of both legs' own timestamps within
    the formation window."""
    a_f = a_df.loc[FORMATION_START:FORMATION_END]
    b_f = b_df.loc[FORMATION_START:FORMATION_END]
    return a_f.index.intersection(b_f.index)


def normalized_distance(a_close: pd.Series, b_close: pd.Series,
                         idx: pd.DatetimeIndex) -> tuple[float, pd.Series, pd.Series]:
    """Sum of squared deviations between the two legs' normalized (start
    at 1.0 at the pair's own overlap start) price paths over `idx`."""
    a = a_close.loc[idx]
    b = b_close.loc[idx]
    an = (a / a.iloc[0]).rename("norm_a")
    bn = (b / b.iloc[0]).rename("norm_b")
    dist = float(((an - bn) ** 2).sum())
    return dist, an, bn


def run_screen() -> tuple[pd.DataFrame, list[dict]]:
    """Step A: rank all 28 pairs by formation-period distance; report which
    qualify on the >=250-daily-equivalent-observation data filter."""
    dfs = load_all_legs()
    rows = []
    for a, b in itertools.combinations(UNIVERSE, 2):
        idx = pair_overlap_formation(dfs[a], dfs[b])
        n_bars = len(idx)
        daily_equiv = n_bars / BARS_PER_DAY
        qualifies = daily_equiv >= MIN_FORMATION_DAYS
        row = {"pair": f"{a}-{b}", "a": a, "b": b, "overlap_bars": n_bars,
               "daily_equiv_days": daily_equiv, "qualifies": qualifies,
               "distance": np.nan, "overlap_start": None, "overlap_end": None}
        if qualifies:
            dist, _, _ = normalized_distance(dfs[a]["close"], dfs[b]["close"], idx)
            row["distance"] = dist
            row["overlap_start"] = idx.min()
            row["overlap_end"] = idx.max()
        rows.append(row)
    table = pd.DataFrame(rows)
    table = table.sort_values(["qualifies", "distance"], ascending=[False, True],
                              na_position="last").reset_index(drop=True)
    qualifying = table[table["qualifies"]].to_dict("records")
    return table, qualifying


def block_bootstrap_null(a_close: pd.Series, b_close: pd.Series, idx: pd.DatetimeIndex,
                          n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                          block: int = BARS_PER_DAY) -> np.ndarray:
    """Null distribution for the winning pair's distance statistic.

    Resamples leg `a`'s own log-return series (over the formation overlap)
    in contiguous 1-day (288-bar) blocks, with replacement, to the same
    total length; reconstructs a synthetic normalized price path from the
    resampled returns (starting at 1.0); recomputes the sum-of-squared-
    deviations against leg `b`'s REAL, unshuffled normalized path. `a` is
    always the earlier-listed instrument of the pair in UNIVERSE order
    (fixed before the screen ran, see UNIVERSE's own docstring note) --
    the bootstrapped leg is not chosen after seeing which pair wins.
    """
    a = a_close.loc[idx].to_numpy(dtype=float)
    b = b_close.loc[idx].to_numpy(dtype=float)
    bn = b / b[0]

    log_r_a = np.diff(np.log(a))
    n = len(log_r_a)
    n_blocks_avail = n // block
    if n_blocks_avail < 2:
        raise RuntimeError("not enough formation bars for the block bootstrap")
    blocks = [log_r_a[i * block:(i + 1) * block] for i in range(n_blocks_avail)]
    needed = int(np.ceil(n / block))

    rng = np.random.default_rng(seed)
    null_dist = np.empty(n_boot)
    for k in range(n_boot):
        chosen = rng.integers(0, n_blocks_avail, size=needed)
        resampled = np.concatenate([blocks[i] for i in chosen])[:n]
        log_path = np.concatenate([[0.0], np.cumsum(resampled)])
        synth = np.exp(log_path)
        null_dist[k] = float(((synth - bn) ** 2).sum())
    return null_dist


def cmd_screen(verbose: bool = True) -> dict:
    dfs = load_all_legs()
    table, qualifying = run_screen()
    if verbose:
        print("=" * 88)
        print("R-76 novel branch: Step A -- distance-method pair screen (formation 2017-2020)")
        print("=" * 88)
        with pd.option_context("display.max_rows", 40, "display.width", 140):
            print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        n_qual = len(qualifying)
        print(f"\n{n_qual} of 28 pairs qualify (>= {MIN_FORMATION_DAYS} overlapping "
              f"formation-period daily-equivalent 5-minute observations).")

    if not qualifying:
        raise RuntimeError("no pair qualifies on the data-availability filter -- cannot proceed")

    winner = min(qualifying, key=lambda r: r["distance"])
    loser = max(qualifying, key=lambda r: r["distance"])
    a, b = winner["a"], winner["b"]
    idx = pair_overlap_formation(dfs[a], dfs[b])

    if verbose:
        print(f"\nMinimum-distance pair: {winner['pair']}  distance={winner['distance']:.6f}  "
              f"overlap={winner['overlap_start']} -> {winner['overlap_end']} "
              f"({winner['overlap_bars']:,} bars, {winner['daily_equiv_days']:.1f} days)")
        print(f"Bootstrapping leg '{a}' (earlier-listed in UNIVERSE order, fixed before the screen ran) "
              f"against leg '{b}''s real path, {N_BOOT} reps, seed={BOOT_SEED}...")

    null_dist = block_bootstrap_null(dfs[a]["close"], dfs[b]["close"], idx)
    pval = float((null_dist <= winner["distance"]).mean())
    gate_pass = pval < 0.05

    if verbose:
        print(f"Null distance: mean={null_dist.mean():.6f} std={null_dist.std():.6f} "
              f"p05={np.percentile(null_dist, 5):.6f} p50={np.percentile(null_dist, 50):.6f}")
        print(f"Observed minimum distance: {winner['distance']:.6f}  p-value={pval:.4f}")
        print("\n" + "=" * 88)
        print("PRE-REGISTERED STOP RULE (frozen before any number above was computed):")
        print("  proceed to Step B only if the minimum-distance pair's bootstrap p < 0.05.")
        print(f"  GATE: {'PASS -> proceed to Step B' if gate_pass else 'FAIL -> STOP, report the ranked table as the result'}")
        print("=" * 88)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "r76_novel_pairs_screen.csv", index=False)

    return {
        "table": table, "qualifying": qualifying, "n_qualifying": len(qualifying),
        "winner": winner, "loser": loser, "null_dist": null_dist, "pval": pval,
        "gate_pass": gate_pass,
    }


# ======================================================= step B: pairs + kelly

def build_pair_signal(a_name: str, b_name: str) -> dict:
    """The distance-method spread, z-score and Kelly/vol-target scale for
    one pair, computed on the two legs' full pre-holdout overlap (not just
    the formation window -- this feeds inner-train AND inner-validation).
    """
    a_df, b_df = load_leg(a_name), load_leg(b_name)
    pair_start = max(a_df.index.min(), b_df.index.min())
    idx = a_df.index.intersection(b_df.index)
    idx = idx[idx >= pair_start]

    a_close = a_df["close"].loc[idx]
    b_close = b_df["close"].loc[idx]
    an = a_close / a_close.iloc[0]
    bn = b_close / b_close.iloc[0]
    spread = (an - bn).rename("spread")

    roll_mean = spread.rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).mean()
    roll_std = spread.rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).std()
    z = ((spread - roll_mean) / roll_std).replace([np.inf, -np.inf], np.nan)

    # spread_vol: EWM vol of the spread's own first difference (a spread
    # can be negative or cross zero, so its own "return" is a simple diff,
    # not a log-return) -- same EWM span / annualization / one-bar-shift
    # SHAPE as kelly_regime_v3's own realized-vol estimate.
    spread_diff = spread.diff()
    vol = (spread_diff.ewm(span=VOL_SPAN, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_scale = np.minimum(TARGET_VOL / vol, MAX_LEVERAGE)
    scale = pd.Series(
        np.where(np.isfinite(raw_scale) & (vol > 0), raw_scale, 0.0), index=idx)

    return {"index": idx, "pair_start": pair_start, "z": z, "scale": scale,
            "a_df": a_df, "b_df": b_df}


def leg_targets(sig: dict, cap: float) -> tuple[pd.Series, pd.Series]:
    """Continuous, |z|-ramped, sign-opposed target fraction per leg.

    ramp(|z|) linearly interpolates 0 at z=0 to 1 at |z|=cap (clamped
    beyond). Long the leg below the pair's shared normalized level (spread
    below its own rolling mean, z<0) and short the leg above it.
    """
    z, scale = sig["z"], sig["scale"]
    ramp = (z.abs() / cap).clip(lower=0.0, upper=1.0)
    signed = (-np.sign(z) * scale * ramp).fillna(0.0)
    pos_a = signed.rename("target")
    pos_b = (-signed).rename("target")
    return pos_a, pos_b


def align_pair_signal_causal(sig_series: pd.Series, leg_df: pd.DataFrame) -> pd.Series:
    """Join the pair-level target signal (computed on the joint overlap
    grid) onto one leg's own native bar index, causally: a bar at time T
    sees only the most recently computed signal value at or before T,
    never a future one. Same reindex/union/sort/ffill construction as
    align_onchain_causal / align_macro_causal / align_stablecoin_causal /
    align_dvol_causal / align_mvrv_causal in tradebot/data.py -- minus
    their +1-day publication-lag shift, because both legs here already
    share one native 5-minute UTC clock (no reporting lag to model).
    """
    idx_union = sig_series.index.union(leg_df.index)
    joined = sig_series.reindex(idx_union).sort_index().ffill().reindex(leg_df.index)
    return joined.fillna(0.0)


class PairLegStrategy(Strategy):
    """One leg of an R-76 distance-method pairs book.

    Reads a precomputed, causally-joined target column (already |z|-ramped
    and Kelly/vol-target scaled, opposite sign per leg) and orders to it
    every bar. Not @register-ed: an experiment-local, one-off construction
    per (pair, cap) combination, following this repo's own convention for
    unregistered ad-hoc Strategy subclasses inside experiments/ (see e.g.
    tests/test_multiasset.py's toy strategies, or the 17 other experiment
    files that locally subclass Strategy without registering it).
    """

    name = "_r76_pair_leg"
    warmup = ROLL_WINDOW + VOL_SPAN + 10

    def __init__(self, target: pd.Series) -> None:
        self._target = target

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["target"] = self._target.reindex(df.index).fillna(0.0).to_numpy()
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)


def run_pair_config(a_name: str, b_name: str, cap: float, start, end,
                     fee_rate: float = 0.0005, start_balance: float = START_BALANCE):
    sig = build_pair_signal(a_name, b_name)
    pos_a_raw, pos_b_raw = leg_targets(sig, cap)
    a_df, b_df = sig["a_df"], sig["b_df"]
    a_target = align_pair_signal_causal(pos_a_raw, a_df)
    b_target = align_pair_signal_causal(pos_b_raw, b_df)
    market = MarketSpec.futures(fee_rate=fee_rate)
    specs = [
        MultiAssetSpec(a_name, PairLegStrategy(a_target), a_df, market),
        MultiAssetSpec(b_name, PairLegStrategy(b_target), b_df, market),
    ]
    return run_multi_backtest(specs, [0.5, 0.5], start_balance, start=start, end=end)


def run_pair_benchmark(a_name: str, b_name: str, start, end, start_balance: float = START_BALANCE):
    """50/50 buy_and_hold of the same two legs, spot -- the risk-matched
    benchmark for a market-neutral book, NOT single-asset buy_and_hold."""
    a_df, b_df = load_leg(a_name), load_leg(b_name)
    market = MarketSpec.spot()
    specs = [
        MultiAssetSpec(a_name, get_strategy("buy_and_hold"), a_df, market),
        MultiAssetSpec(b_name, get_strategy("buy_and_hold"), b_df, market),
    ]
    return run_multi_backtest(specs, [0.5, 0.5], start_balance, start=start, end=end)


def _fmt_result(label: str, result) -> str:
    m = result.metrics
    return (f"  {label:34s} final=${m.final_balance:>10,.2f} "
            f"growth={m.final_balance / result.portfolio.start_balance:>6.3f}x "
            f"sharpe={m.sharpe:>6.2f} maxDD={m.max_drawdown_pct:>5.1f}%")


def cmd_stepb(screen_result: dict | None = None, verbose: bool = True) -> dict:
    if screen_result is None:
        screen_result = cmd_screen(verbose=verbose)
    winner = screen_result["winner"]
    a, b = winner["a"], winner["b"]
    sig = build_pair_signal(a, b)
    pair_start = sig["pair_start"]

    if verbose:
        print("\n" + "=" * 88)
        print(f"R-76 novel branch: Step B -- {a}-{b} distance-method spread, Kelly/vol-target sizing")
        print(f"pair_start={pair_start}  (later of the two legs' own first available bar)")
        print("=" * 88)

    windows = [("inner-train", pair_start, TRAIN_END), ("inner-validation", VALID_START, VALID_END)]
    rows = []
    for wname, start, end in windows:
        bench = run_pair_benchmark(a, b, start, end)
        if verbose:
            print(f"\n-- {wname} ({start} -> {end}) --")
            print(_fmt_result("50/50 buy&hold benchmark", bench))
        rows.append({"window": wname, "cap": None, "kind": "benchmark",
                     "final": bench.metrics.final_balance,
                     "growth": bench.metrics.final_balance / bench.portfolio.start_balance,
                     "sharpe": bench.metrics.sharpe, "max_dd": bench.metrics.max_drawdown_pct})
        for cap in RAMP_CAPS:
            res = run_pair_config(a, b, cap, start, end)
            if verbose:
                print(_fmt_result(f"cap={cap:g}", res))
            rows.append({"window": wname, "cap": cap, "kind": "strategy",
                         "final": res.metrics.final_balance,
                         "growth": res.metrics.final_balance / res.portfolio.start_balance,
                         "sharpe": res.metrics.sharpe, "max_dd": res.metrics.max_drawdown_pct})

    table = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "r76_novel_pairs_stepB.csv", index=False)
    if verbose:
        print(f"\nStep B configurations evaluated: {len(RAMP_CAPS)} "
              f"(max-|z| ramp cap in {RAMP_CAPS}, reported on both windows)")
        print(f"written: {REPORTS_DIR / 'r76_novel_pairs_stepB.csv'}")
    return {"pair": (a, b), "pair_start": pair_start, "table": table}


# ================================================================ falsification

def cmd_falsify(screen_result: dict | None = None, verbose: bool = True) -> dict:
    """Both pre-registered falsification tests: (1) scramble control on the
    LARGEST-distance qualifying pair, inner-validation only; (2) cost
    sensitivity at fee_rate=0.004 on the winning pair, inner-validation
    only. Both against the 50/50-hold benchmark of their own two legs.
    """
    if screen_result is None:
        screen_result = cmd_screen(verbose=verbose)
    winner, loser = screen_result["winner"], screen_result["loser"]
    wa, wb = winner["a"], winner["b"]
    la, lb = loser["a"], loser["b"]

    if verbose:
        print("\n" + "=" * 88)
        print("R-76 novel branch: falsification tests (inner-validation only)")
        print("=" * 88)
        print(f"\n[1] Scramble control -- largest-distance qualifying pair: {loser['pair']} "
              f"(distance={loser['distance']:.6f}, vs winner's {winner['distance']:.6f})")

    rows = []
    bench_loser = run_pair_benchmark(la, lb, VALID_START, VALID_END)
    rows.append({"test": "scramble_control", "pair": loser["pair"], "cap": None, "kind": "benchmark",
                 "growth": bench_loser.metrics.final_balance / bench_loser.portfolio.start_balance,
                 "sharpe": bench_loser.metrics.sharpe, "max_dd": bench_loser.metrics.max_drawdown_pct})
    if verbose:
        print(_fmt_result("50/50 buy&hold benchmark", bench_loser))
    for cap in RAMP_CAPS:
        res = run_pair_config(la, lb, cap, VALID_START, VALID_END)
        rows.append({"test": "scramble_control", "pair": loser["pair"], "cap": cap, "kind": "strategy",
                     "growth": res.metrics.final_balance / res.portfolio.start_balance,
                     "sharpe": res.metrics.sharpe, "max_dd": res.metrics.max_drawdown_pct})
        if verbose:
            print(_fmt_result(f"cap={cap:g}", res))

    if verbose:
        print(f"\n[2] Cost sensitivity -- winning pair {winner['pair']}, fee_rate=0.004 (0.40% tier)")
    bench_winner = run_pair_benchmark(wa, wb, VALID_START, VALID_END)
    rows.append({"test": "cost_sensitivity", "pair": winner["pair"], "cap": None, "kind": "benchmark",
                 "growth": bench_winner.metrics.final_balance / bench_winner.portfolio.start_balance,
                 "sharpe": bench_winner.metrics.sharpe, "max_dd": bench_winner.metrics.max_drawdown_pct})
    if verbose:
        print(_fmt_result("50/50 buy&hold benchmark", bench_winner))
    for cap in RAMP_CAPS:
        res = run_pair_config(wa, wb, cap, VALID_START, VALID_END, fee_rate=0.004)
        rows.append({"test": "cost_sensitivity", "pair": winner["pair"], "cap": cap, "kind": "strategy",
                     "growth": res.metrics.final_balance / res.portfolio.start_balance,
                     "sharpe": res.metrics.sharpe, "max_dd": res.metrics.max_drawdown_pct})
        if verbose:
            print(_fmt_result(f"cap={cap:g} fee=0.004", res))

    table = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "r76_novel_pairs_falsify.csv", index=False)
    if verbose:
        print(f"\nwritten: {REPORTS_DIR / 'r76_novel_pairs_falsify.csv'}")
        print("(Both falsification tests replay the pre-registered {2,4,6} ramp-cap grid; "
              "they contribute 0 additional configurations to the trials count -- falsification "
              "replays, not a new sweep, this project's standing convention.)")
    return {"table": table}


# ======================================================================= report

def cmd_report() -> None:
    print("=" * 88)
    print("R-76 novel branch: max timestamp actually read from each of the 8 data files")
    print("=" * 88)
    load_all_legs()
    for name in UNIVERSE:
        print(f"  {name:6s} max timestamp read: {_MAX_TS_READ[name]}")


# =========================================================================== CLI

def cmd_all() -> None:
    screen_result = cmd_screen()
    if not screen_result["gate_pass"]:
        print("\nGATE FAILS -- per the pre-registered stop rule, STOPPING here. "
              "No strategy is built. The ranked screen table above is the round's entire result.")
        cmd_report()
        return
    cmd_stepb(screen_result)
    cmd_falsify(screen_result)
    cmd_report()


COMMANDS = {
    "screen": lambda: cmd_screen(),
    "stepb": lambda: cmd_stepb(),
    "falsify": lambda: cmd_falsify(),
    "report": cmd_report,
    "all": cmd_all,
}


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}; choose one of {sorted(COMMANDS)}")
        sys.exit(1)
    COMMANDS[cmd]()


if __name__ == "__main__":
    main()
