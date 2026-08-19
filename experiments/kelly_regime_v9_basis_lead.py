#!/usr/bin/env python
"""NOVEL branch, parallel round 08-19: does real-time perp basis LEAD the
vote's own flip dates, and if so, can that be used to let kelly_regime_v4's
slow 3-anchor vote finish latching sooner (the TIMING axis, never touched
by R-34/R-35/R-37/R-38/R-40 -- all five worked the SIZE/magnitude axis)?

Idea, one sentence: price sitting above its slow anchors is a LAGGING
signature of crowd accumulation (Cardaliaguet & Lehalle 2018); leveraged
perp demand -- visible in real-time basis, now that a genuine second price
series exists (Deribit BTC-PERPETUAL) -- is a plausible LEADING signature
of the same accumulation, since speculative capital often expresses a view
through cheap leverage before it fully shows up in spot's slow moving
averages.

Constraint attacked: INFO (one price series) -- this is the first session
with a real, independently-transacted second series to test that against.
Not a duplicate of R-34 (Bayesian posterior as a SIZE input), R-35
(funding as a SIZE input), R-37/R-38 (retuned/formalized sizing constants),
R-40 (bagging the anchor ladder) -- all six of those touch the magnitude
axis; this touches only WHEN the vote finishes latching.

Pre-registered failure mode, named before any code ran: basis moves
together with price rather than ahead of it (both are the same leveraged
crowd, expressing the same view through two venues at once), so the
"lead" is zero or is autocorrelation of basis with itself, not a real
timing edge -- and even if a small lead exists, five-minute-bar trading
costs eat it, or the effect is the same fitted-to-one-window artifact that
sank R-37/R-38/R-40.

Usage::

    python experiments/kelly_regime_v9_basis_lead.py leadlag     # step 2
    python experiments/kelly_regime_v9_basis_lead.py sweep       # step 3
    python experiments/kelly_regime_v9_basis_lead.py exposure    # diag 1
    python experiments/kelly_regime_v9_basis_lead.py funding     # diag 3
    python experiments/kelly_regime_v9_basis_lead.py causality   # diag 4
    python experiments/kelly_regime_v9_basis_lead.py all         # everything
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import (  # noqa: E402
    compute_basis, load_dataset, load_deribit_perp_price, load_funding,
)
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR, KellyRegime
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

# ROUTINE.md's inner-train is 2017-01-01 -> 2020-12-31, but Deribit
# BTC-PERPETUAL chart data starts 2018-08-14 (probed empirically, see
# tradebot.data.load_deribit_perp_price) -- so the *basis-covered* part of
# inner-train is shorter. This window plays two roles at once: it is both
# the fitting window (step 3) and the earlier control the R-37/R-38/R-40
# overfitting signature check needs (diagnostic 2).
TRAIN_BASIS = ("2018-08-14", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS_START = "2023-01-01"  # never read in this file

N_EVALUATED = 0  # every distinct configuration backtested, for deflated Sharpe


# --------------------------------------------------------------------- data


def build_basis_dataframe() -> tuple[pd.DataFrame, str]:
    """Spot OHLCV with a causal ``basis`` column merged on, NaN before coverage.

    ``basis`` is ``log(perp_close / spot_close)`` on the Deribit perp's own
    grid (``compute_basis``, already causal internally -- as-of join, never
    interpolated), then merged onto the spot bar index with one more
    as-of/ffill step so a strategy reading ``df["basis"]`` at bar *i* only
    ever sees perp information timestamped at or before bar *i*. Nothing is
    back-filled: rows before 2018-08-14 are NaN, and the strategy below
    falls back to kelly_regime_v4's exact unmodified behaviour there.
    """
    spot, label = load_dataset(ROOT / "data", "spot")
    perp = load_deribit_perp_price(ROOT / "data", "BTC")
    if perp is None:
        raise FileNotFoundError("data/btcusdt_deribit_perp_5m.csv.gz not found")
    basis_on_perp = compute_basis(spot, perp)
    combined_idx = spot.index.union(basis_on_perp.index)
    basis_on_spot = (
        basis_on_perp.reindex(combined_idx).sort_index().ffill().reindex(spot.index)
    )
    basis_on_spot = basis_on_spot.where(spot.index >= basis_on_perp.index.min())
    out = spot.copy()
    out["basis"] = basis_on_spot
    return out, label


DF, LABEL = build_basis_dataframe()
print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
      f"(data: {LABEL}); basis coverage {DF['basis'].notna().sum():,} bars from "
      f"{DF['basis'].dropna().index[0]:%Y-%m-%d}", file=sys.stderr)


def ev(strategy, start, end, df=None, market=SPOT, tag="", balance=1_000.0,
       count=True):
    """One backtest, one line, counted."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    print(f"  {tag or strategy.name:34s} {market.name:9s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"fills={len(result.fills):>5d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f}{'  LIQUIDATED' if m.liquidated else ''}")
    return m


# ---------------------------------------------------------------- the vote


def _anchor_votes(close: pd.Series, horizons=(20, 40, 80), band: float = 0.01):
    """Exactly kelly_regime_v4's per-anchor latched vote. Returns a dict."""
    votes = {}
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes[days] = v.ffill().fillna(0.0)
    return votes


def _basis_state_and_runlen(basis: pd.Series, smooth_days: float, threshold: float):
    """Hysteresis-latched ternary basis vote (+1/-1/0) and its run length.

    Same single-band hysteresis style kelly_regime already uses for its
    price anchors (in and out at the same level), applied to a causally
    smoothed, one-bar-lagged basis instead of price-vs-anchor. ``run_len``
    is how many consecutive bars the current state has held -- the vector
    the "sustained for at least N days" confirmation test reads.

    NaN basis (before 2018-08-14 coverage) always latches to neutral 0.0,
    which is what makes the whole mechanism fall back to v4 automatically
    wherever basis is unavailable, with no special-casing needed downstream.
    """
    span = max(1, int(smooth_days * BARS_PER_DAY))
    smooth = basis.ewm(span=span, min_periods=max(1, span // 2)).mean().shift(1)
    raw = np.where(smooth > threshold, 1.0,
                    np.where(smooth < -threshold, -1.0, np.nan))
    latched = pd.Series(raw, index=basis.index).ffill().fillna(0.0)
    change = latched.ne(latched.shift())
    change.iloc[0] = True
    grp = change.cumsum()
    run_len = (latched.groupby(grp).cumcount() + 1).to_numpy()
    return latched.to_numpy(), run_len


class KellyRegimeV9BasisLead(KellyRegimeV3):
    """Basis as a 4th, faster confirming vote -- accelerates the vote's
    OWN latch timing, never its sizing.

    Mechanism, one sentence: when 2 of the 3 slow price anchors already
    agree on a direction (majority formed, one laggard anchor still
    pending) and the real-time Deribit perp basis -- a faster,
    leverage-based proxy for the same crowd accumulation the slow anchors
    detect only after it has already moved price -- has stayed
    causally-lagged-and-smoothed confirmed in that SAME direction for at
    least ``min_confirm_days``, let the vote finish latching to full
    agreement immediately, instead of waiting for the slowest (80-day)
    anchor to also cross; it can never flip the regime against the anchor
    majority, because the promotion condition only fires at 2/3 or 1/3 (2
    of 3 anchors already agreeing), never at 0/3 or 3/3, and never at the
    minority-leaning values.

    Sizing (conditional vol targeting, deadband, cap) is v3/v4's, entirely
    unchanged -- only ``prepare()``'s vote-construction differs, to keep
    this a clean test of the TIMING axis and not the magnitude axis every
    prior round (R-34/R-35/R-37/R-38/R-40) already tried.
    """

    name = "kelly_regime_v9_basis_lead"
    warmup = 80 * BARS_PER_DAY + 10  # same as v4; dominates the basis warmup

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 basis_smooth_days: float = 1.0, basis_threshold: float = 0.0020,
                 min_confirm_days: float = 2.0, **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.basis_smooth_days = basis_smooth_days
        self.basis_threshold = basis_threshold
        self.min_confirm_days = min_confirm_days

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = _anchor_votes(close, self.horizons, self.band)
        anchor_frac = (sum(votes.values()) / len(votes)).to_numpy()

        if "basis" in df.columns:
            state, run_len = _basis_state_and_runlen(
                df["basis"], self.basis_smooth_days, self.basis_threshold)
            min_bars = int(self.min_confirm_days * BARS_PER_DAY)
            confirmed_bull = (state == 1.0) & (run_len >= min_bars)
            confirmed_bear = (state == -1.0) & (run_len >= min_bars)
        else:
            confirmed_bull = np.zeros(len(df), dtype=bool)
            confirmed_bear = np.zeros(len(df), dtype=bool)

        frac = anchor_frac.copy()
        promote_bull = np.isclose(anchor_frac, 2.0 / 3.0) & confirmed_bull
        promote_bear = np.isclose(anchor_frac, 1.0 / 3.0) & confirmed_bear
        frac[promote_bull] = 1.0
        frac[promote_bear] = 0.0
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        # From here down: byte-for-byte kelly_regime_v3's conditional
        # vol-targeting sizer, unchanged, just fed `frac` instead of v3's own.
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        vstate = 0
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if vstate == 0:
                    vstate = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif vstate == 1 and x < self.high_out:
                    vstate = 0
                elif vstate == -1 and x > self.low_out:
                    vstate = 0
            scale = full[i] if vstate != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["v9_frac"] = frac
        df["v9_anchor_frac"] = anchor_frac
        df["v9_promoted"] = promote_bull | promote_bear
        return df


# ------------------------------------------------------------- step 2: leadlag


def _episodes(anchor_frac: np.ndarray, level: float):
    """Contiguous runs where anchor_frac == level. Returns (start_i, end_i, next_level)."""
    is_level = np.isclose(anchor_frac, level)
    out = []
    i = 0
    n = len(anchor_frac)
    while i < n:
        if is_level[i]:
            j = i
            while j < n and is_level[j]:
                j += 1
            next_level = anchor_frac[j] if j < n else np.nan
            out.append((i, j - 1, next_level))
            i = j
        else:
            i += 1
    return out


def _block_bootstrap(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap: preserves x's own short-range autocorrelation,
    destroys its alignment with anything else (here, the anchor-vote episodes)."""
    n = len(x)
    out = np.empty(n, dtype=x.dtype)
    pos = 0
    while pos < n:
        start = rng.integers(0, n)
        take = min(block, n - pos)
        idx = (start + np.arange(take)) % n
        out[pos:pos + take] = x[idx]
        pos += take
    return out


def _leadlag_one_config(anchor_frac: np.ndarray, basis: pd.Series,
                        smooth_days: float, threshold: float,
                        min_confirm_days: float, label: str) -> dict:
    state, run_len = _basis_state_and_runlen(basis, smooth_days, threshold)
    min_bars = int(min_confirm_days * BARS_PER_DAY)
    confirmed_bull = (state == 1.0) & (run_len >= min_bars)
    confirmed_bear = (state == -1.0) & (run_len >= min_bars)

    def score(confirmed, level, up_level, resolves_correct_fn, already_confirmed_lead=True):
        eps = _episodes(anchor_frac, level)
        n_up, n_confirmed_before_resolve, n_confirmed_up = 0, 0, 0
        n_already_at_start = 0
        lead_days = []
        for start, end, nxt in eps:
            if not np.isfinite(nxt):
                continue  # runs off the end of the window, no resolution observed
            resolved_up = resolves_correct_fn(nxt)
            n_up += int(resolved_up)
            seg = confirmed[start:end + 1]
            if seg[0]:
                n_already_at_start += 1
            if seg.any():
                n_confirmed_before_resolve += 1
                n_confirmed_up += int(resolved_up)
                first_true = start + int(np.argmax(seg))
                lead_days.append((first_true - start) / BARS_PER_DAY)
        return dict(n_episodes=len(eps), n_resolved=n_up + (0), n_up=n_up,
                    n_confirmed=n_confirmed_before_resolve,
                    n_confirmed_and_up=n_confirmed_up,
                    n_already_confirmed_at_start=n_already_at_start,
                    lead_days=lead_days)

    bull = score(confirmed_bull, 2.0 / 3.0, 1.0, lambda nxt: np.isclose(nxt, 1.0))
    bear = score(confirmed_bear, 1.0 / 3.0, 0.0, lambda nxt: np.isclose(nxt, 0.0))
    return dict(label=label, smooth_days=smooth_days, threshold=threshold,
                min_confirm_days=min_confirm_days, bull=bull, bear=bear)


def _summarize_leadlag(res: dict) -> None:
    for side in ("bull", "bear"):
        d = res[side]
        base_rate = d["n_up"] / d["n_episodes"] if d["n_episodes"] else float("nan")
        conf_rate = (d["n_confirmed_and_up"] / d["n_confirmed"]
                     if d["n_confirmed"] else float("nan"))
        med_lead = float(np.median(d["lead_days"])) if d["lead_days"] else float("nan")
        print(f"    {side:4s}: episodes={d['n_episodes']:>3d}  "
              f"base P(resolve correct)={base_rate:>5.1%}  "
              f"basis-confirmed-before-resolve: n={d['n_confirmed']:>3d} "
              f"P(correct|confirmed)={conf_rate:>5.1%}  "
              f"already-confirmed-at-episode-start={d['n_already_confirmed_at_start']:>3d}  "
              f"median lead (days, among confirmed)={med_lead:>5.2f}")


def leadlag(n_boot: int = 200, seed: int = 0) -> None:
    """Step 2 -- honest, purely descriptive: does basis lead the vote's own
    flip dates, more than chance/autocorrelation explains? Run BEFORE
    committing to any trading formula.

    Target event: a "2/3 majority" episode (2 of 3 anchors already agree on
    a direction, the mechanism's exact use case) either resolves to full
    agreement (the direction the majority already leaned -- "correct") or
    reverts (a false start). Does a sustained, causally-lagged basis
    confirmation in the majority's direction predict which happens, beyond
    the base rate -- and does it typically arrive before the resolution
    (a genuine lead) rather than after?

    Null: a moving-block bootstrap of the basis series (30-day blocks)
    preserves basis's own autocorrelation but destroys its alignment with
    the (fixed) anchor-vote episodes -- repeated `n_boot` times to see
    whether the observed P(correct | confirmed) could arise from an
    unrelated, similarly-autocorrelated series.
    """
    lo, hi = TRAIN_BASIS
    seg = DF.loc[lo:hi]
    close = seg["close"]
    votes = _anchor_votes(close)
    anchor_frac = (sum(votes.values()) / len(votes)).to_numpy()
    basis = seg["basis"]

    print(f"leadlag study window: {lo} -> {hi}  ({len(seg):,} bars)")
    print(f"anchor_frac state counts: "
          + ", ".join(f"{v:.3f}={int(np.isclose(anchor_frac, v).sum()):,}"
                       for v in (0.0, 1.0 / 3, 2.0 / 3, 1.0)))

    configs = [
        ("smooth=0.5d thr=0.15% confirm=1d", 0.5, 0.0015, 1.0),
        ("smooth=1d   thr=0.20% confirm=2d", 1.0, 0.0020, 2.0),
        ("smooth=1d   thr=0.30% confirm=2d", 1.0, 0.0030, 2.0),
        ("smooth=2d   thr=0.30% confirm=4d", 2.0, 0.0030, 4.0),
        ("smooth=2d   thr=0.20% confirm=1d", 2.0, 0.0020, 1.0),
        ("smooth=4d   thr=0.30% confirm=4d", 4.0, 0.0030, 4.0),
    ]
    print(f"\n{len(configs)} lead-lag study configurations "
          "(descriptive only, not backtests -- tracked separately from step 3):\n")

    observed = []
    for label, sd, thr, mc in configs:
        res = _leadlag_one_config(anchor_frac, basis, sd, thr, mc, label)
        print(f"  [{label}]")
        _summarize_leadlag(res)
        observed.append(res)

    # Null: block-bootstrap the basis series, re-score the SAME fixed
    # anchor-vote episodes against each resampled basis path.
    print(f"\nnull test ({n_boot} moving-block bootstrap resamples, "
          "30-day blocks, basis series only -- episodes held fixed):")
    rng = np.random.default_rng(seed)
    block = 30 * BARS_PER_DAY
    basis_arr = basis.to_numpy()
    basis_index = basis.index
    for label, sd, thr, mc in configs:
        obs = next(r for r in observed if r["label"] == label)
        obs_rate_bull = (obs["bull"]["n_confirmed_and_up"] / obs["bull"]["n_confirmed"]
                          if obs["bull"]["n_confirmed"] else float("nan"))
        obs_rate_bear = (obs["bear"]["n_confirmed_and_up"] / obs["bear"]["n_confirmed"]
                          if obs["bear"]["n_confirmed"] else float("nan"))
        null_bull, null_bear = [], []
        for _ in range(n_boot):
            shuffled = pd.Series(_block_bootstrap(basis_arr, block, rng), index=basis_index)
            res_b = _leadlag_one_config(anchor_frac, shuffled, sd, thr, mc, "null")
            if res_b["bull"]["n_confirmed"]:
                null_bull.append(res_b["bull"]["n_confirmed_and_up"] / res_b["bull"]["n_confirmed"])
            if res_b["bear"]["n_confirmed"]:
                null_bear.append(res_b["bear"]["n_confirmed_and_up"] / res_b["bear"]["n_confirmed"])
        null_bull = np.array(null_bull) if null_bull else np.array([np.nan])
        null_bear = np.array(null_bear) if null_bear else np.array([np.nan])
        pctl_bull = float((null_bull < obs_rate_bull).mean()) if np.isfinite(obs_rate_bull) else float("nan")
        pctl_bear = float((null_bear < obs_rate_bear).mean()) if np.isfinite(obs_rate_bear) else float("nan")
        print(f"  [{label}]")
        print(f"    bull: observed P(correct|confirmed)={obs_rate_bull:>5.1%}  "
              f"null mean={np.nanmean(null_bull):>5.1%} sd={np.nanstd(null_bull):.3f}  "
              f"observed exceeds null in {pctl_bull:>5.1%} of resamples")
        print(f"    bear: observed P(correct|confirmed)={obs_rate_bear:>5.1%}  "
              f"null mean={np.nanmean(null_bear):>5.1%} sd={np.nanstd(null_bear):.3f}  "
              f"observed exceeds null in {pctl_bear:>5.1%} of resamples")

    print(f"\nlead-lag configurations evaluated (descriptive, not counted toward "
          f"step-3 trials): {len(configs)}")


# --------------------------------------------------------------- step 3: sweep


def _grid():
    out = []
    for sd in (0.5, 1.0, 2.0):
        for thr in (0.0015, 0.0030):
            for mc in (1.0, 3.0):
                out.append((f"sd={sd:g}d thr={thr:.4f} mc={mc:g}d",
                            dict(basis_smooth_days=sd, basis_threshold=thr,
                                 min_confirm_days=mc)))
    return out


def _benchmarks(start, end, market, label):
    print(f"\n{label} benchmarks:")
    for name in ("buy_and_hold", "kelly_regime_v4"):
        ev(get_strategy(name), start, end, market=market, tag=f"  {name}", count=False)


def sweep() -> None:
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN_BASIS, "INNER-TRAIN-WITH-BASIS"),
                                     (VALID, "INNER-VALIDATION")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            print(f"{split} / {mname} candidate configurations:")
            for tag, kw in _grid():
                ev(KellyRegimeV9BasisLead(**kw), start, end, market=market, tag=tag)
    print(f"\nconfigurations evaluated in this run: {N_EVALUATED}")


def select() -> None:
    """Best inner-validation candidate, printed with its full 4-cell table."""
    best_tag, best_kw, best_sharpe = None, None, -1e9
    for tag, kw in _grid():
        m = ev(KellyRegimeV9BasisLead(**kw), *VALID, market=SPOT, tag=f"(scan) {tag}")
        if m.sharpe > best_sharpe:
            best_tag, best_kw, best_sharpe = tag, kw, m.sharpe
    print(f"\nbest inner-validation spot Sharpe: {best_tag}  ({best_sharpe:.2f})\n")
    print("full 4-cell table for the selected candidate:")
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN_BASIS, "TRAIN-WITH-BASIS"), (VALID, "VALID")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            ev(KellyRegimeV9BasisLead(**best_kw), start, end, market=market,
               tag=f"  candidate ({best_tag})", count=False)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    return best_tag, best_kw


# ------------------------------------------------------------------ diagnostics


def exposure_artifact_check(kw: dict | None = None) -> None:
    """Diagnostic 1: regress the candidate's target series against a
    mean-notional-matched flat rescale of v4's own target series on
    inner-validation, both markets. R^2 > 0.95 -> "just a rescale".
    """
    kw = kw or dict(basis_smooth_days=1.0, basis_threshold=0.0020, min_confirm_days=2.0)
    v4 = get_strategy("kelly_regime_v4")
    cand = KellyRegimeV9BasisLead(**kw)
    print(f"exposure-artifact check, candidate={kw}")
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        lo = int(DF.index.searchsorted(VALID[0]))
        hi = int(DF.index.searchsorted(VALID[1], side="right"))
        prefix = min(lo, max(cand.warmup, v4.warmup))
        frame = DF.iloc[lo - prefix:hi]

        v4_prepared = v4.prepare(frame.copy())
        cand_prepared = cand.prepare(frame.copy())
        v4_t = v4_prepared["target"].to_numpy()[prefix:]
        cand_t = cand_prepared["target"].to_numpy()[prefix:]

        mean_abs_v4 = np.mean(np.abs(v4_t))
        mean_abs_cand = np.mean(np.abs(cand_t))
        alpha = mean_abs_cand / mean_abs_v4 if mean_abs_v4 > 0 else 0.0
        rescaled = alpha * v4_t
        ss_res = np.sum((cand_t - rescaled) ** 2)
        ss_tot = np.sum((cand_t - cand_t.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        corr = np.corrcoef(cand_t, v4_t)[0, 1]
        print(f"  {mname:9s} mean|v4|={mean_abs_v4:.3f} mean|cand|={mean_abs_cand:.3f} "
              f"alpha={alpha:.3f}  R^2(cand vs alpha*v4)={r2:.3f}  raw corr={corr:.3f}  "
              f"{'JUST A RESCALE' if r2 > 0.95 else 'genuinely different exposure shape'}")


def funding_corr() -> None:
    """Diagnostic 3: basis vs the repo's committed funding series, on the
    actual working window (re-verified, not just trusted)."""
    funding = load_funding(ROOT / "data")
    if funding is None:
        print("no funding file found")
        return
    daily_basis = DF["basis"].resample("1D").mean()
    daily_funding = funding.resample("1D").mean()
    both = pd.concat([daily_basis.rename("basis"), daily_funding.rename("funding")],
                      axis=1).dropna()
    print(f"daily-resampled overlap: {len(both):,} days, "
          f"{both.index[0]:%Y-%m-%d} -> {both.index[-1]:%Y-%m-%d}")
    print(f"  full overlap correlation:                  {both['basis'].corr(both['funding']):.3f}")
    tb = both.loc[TRAIN_BASIS[0]:TRAIN_BASIS[1]]
    print(f"  inner-train-with-basis correlation (n={len(tb)}):  "
          f"{tb['basis'].corr(tb['funding']):.3f}")
    vl = both.loc[VALID[0]:VALID[1]]
    print(f"  inner-validation correlation (n={len(vl)}):        "
          f"{vl['basis'].corr(vl['funding']):.3f}")


def causality() -> None:
    """Diagnostic 4: two-opposite-tampers probe, restricted to strictly
    pre-2023 data, PLUS an explicit basis-tamper (the pathway a
    lookahead bug would most plausibly hide in, since it is new this
    session -- R-21's warning taken seriously)."""
    from tradebot.broker import PaperBroker

    kw = dict(basis_smooth_days=1.0, basis_threshold=0.0020, min_confirm_days=2.0)

    pre2023 = DF[DF.index < OOS_START]
    df = pre2023.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    def run_probe(name, tamper_fn):
        up, down = df.copy(), df.copy()
        tamper_fn(up, down)

        def decisions(frame):
            s = KellyRegimeV9BasisLead(**kw)
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        a, b = decisions(up), decisions(down)
        bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
        print(f"[{name}] tampered from bar {cut:,} of {len(df):,}; checked bars {bars}")
        print("  FAIL - reads the future at bars " + str(bad) if bad
              else "  PASS - every decision at or before the cut is unchanged")

        pa = KellyRegimeV9BasisLead(**kw).prepare(up.copy())
        pb = KellyRegimeV9BasisLead(**kw).prepare(down.copy())
        for col in ("target", "v9_frac", "v9_anchor_frac", "v9_promoted"):
            diff = np.abs(pa[col].to_numpy()[:cut].astype(float)
                          - pb[col].to_numpy()[:cut].astype(float))
            worst = float(np.nanmax(diff))
            print(f"    column {col:16s} max |difference| before the cut = {worst:.3e}"
                  f"  {'PASS' if worst < 1e-12 else 'FAIL'}")

    def tamper_ohlcv(up, down):
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def tamper_basis(up, down):
        bcol = up.columns.get_loc("basis")
        up.iloc[cut:, bcol] = 0.05    # +5% basis: an extreme, sustained "bull" reading
        down.iloc[cut:, bcol] = -0.05  # -5% basis: an extreme, sustained "bear" reading

    def tamper_both(up, down):
        tamper_ohlcv(up, down)
        tamper_basis(up, down)

    run_probe("OHLCV tamper (standard)", tamper_ohlcv)
    run_probe("BASIS tamper (the new pathway)", tamper_basis)
    run_probe("both at once", tamper_both)


def overfit_signature_check(kw: dict | None = None) -> None:
    """Diagnostic 2: candidate vs v4 on BOTH inner-train-with-basis AND
    inner-validation, both markets -- the exact R-37/R-38/R-40 signature
    check (win only on 2021-22, lose on the earlier control -> suspect)."""
    kw = kw or dict(basis_smooth_days=1.0, basis_threshold=0.0020, min_confirm_days=2.0)
    print(f"overfitting-signature check, candidate={kw}")
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN_BASIS, "TRAIN-WITH-BASIS (control)"),
                                     (VALID, "VALIDATION")):
            v4m = ev(get_strategy("kelly_regime_v4"), start, end, market=market,
                     tag=f"  v4  [{split}/{mname}]", count=False)
            cm = ev(KellyRegimeV9BasisLead(**kw), start, end, market=market,
                    tag=f"  cand[{split}/{mname}]", count=False)
            ratio = cm.final_balance / v4m.final_balance if v4m.final_balance else float("nan")
            print(f"    -> candidate/v4 final-balance ratio: {ratio:.3f}x")


# --------------------------------------------------------------------- driver


def all_checks() -> None:
    print("=" * 78)
    print("STEP 2 -- lead-lag study (descriptive, run before any strategy code)")
    print("=" * 78)
    leadlag()
    print("\n" + "=" * 78)
    print("STEP 3 -- sweep")
    print("=" * 78)
    sweep()
    print("\n" + "=" * 78)
    print("DIAGNOSTIC 1 -- exposure-artifact check")
    print("=" * 78)
    exposure_artifact_check()
    print("\n" + "=" * 78)
    print("DIAGNOSTIC 2 -- R-37/R-38/R-40 overfitting-signature check")
    print("=" * 78)
    overfit_signature_check()
    print("\n" + "=" * 78)
    print("DIAGNOSTIC 3 -- basis vs funding correlation")
    print("=" * 78)
    funding_corr()
    print("\n" + "=" * 78)
    print("DIAGNOSTIC 4 -- causality / no-lookahead")
    print("=" * 78)
    causality()
    print(f"\ntotal configurations evaluated (backtests, step 3 + diagnostics): "
          f"{N_EVALUATED}")


if __name__ == "__main__":
    cmds = {"leadlag": leadlag, "sweep": sweep, "select": select,
            "exposure": exposure_artifact_check, "overfit": overfit_signature_check,
            "funding": funding_corr, "causality": causality, "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/kelly_regime_v9_basis_lead.py [{'|'.join(cmds)}]")
