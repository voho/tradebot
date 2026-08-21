#!/usr/bin/env python
"""R-74 CONSERVATIVE branch: MVRV-ratio LEVEL as a confirming vote on
kelly_regime_v4's own 3-anchor regime gate.

Full pre-registration is in ``experiments/r74_conservative_mvrv_signal.py``
(mechanism, citation, directionality decision, and the z-score-window
decision -- all made before any number below was produced). Restated
briefly here for a reader of this file alone:

Idea, one sentence
-------------------
Historically extreme MVRV Z-scores (aggregate holder unrealized
profit/loss, far above/below its own since-inception norm) have marked
Bitcoin's market-cycle tops (Mahmudov & Puell 2018, building on Carter &
Le Calvez's realized-cap concept, Honeybadger Capital 2018), so the
EUPHORIA extreme alone -- never the capitulation extreme -- is used as a
one-directional, DILUTING confirming vote alongside kelly_regime_v4's
three unchanged price anchors, exactly R-55's validated architecture:

    frac = (anchor_sum + mvrv_weight * mvrv_vote) / (3 + mvrv_weight)

``anchor_sum`` is v4's own three UNCHANGED 0/1 latched price-anchor votes.
``mvrv_vote`` is a latched 0/1 vote on the causal, EXPANDING-window MVRV
Z-score (``r74_conservative_mvrv_signal.compute_mvrv_z``, unmodified,
imported read-only): 0 ("euphoric", dilutive) while the Z-score is latched
above ``thresh_hi``, back to 1 ("not euphoric", no dilution) once it falls
below ``thresh_hi - gap``, defaulting to 1 wherever data is absent or still
warming up. ``mvrv_weight=0`` recovers v4 exactly (identity check, verified
in ``causality()``).

Constraint attacked: INFO (one price series). Sixth structurally distinct
INFO-axis signal tried in this project (on-chain activity B-07/R-44; macro
VIX/DXY R-53; stablecoin-supply deceleration R-54/R-55; options-implied
vol/VRP R-73; this round's MVRV). Not a duplicate of any of those: MVRV is
a valuation ratio (aggregate holder cost basis vs. today's price), not a
flow, not a priced vol expectation, not a macro spillover, and not the
traded asset's own on-chain activity count.

Not a duplicate of this same round's parallel NOVEL branch (MVRV rate of
change) -- disjoint files, not read, not coordinated with, per
ROUTINE.md's parallelism rules.

Pre-registered falsification battery (the standard battery this project's
INFO-axis rounds use, chosen BEFORE any number below was produced)
--------------------------------------------------------------------
(a) Beats kelly_regime_v4 on inner-validation Sharpe by more than the
    +/-0.2 noise floor (R-20), OR shows a clear drawdown/tail improvement,
    on BOTH markets (spot and futures 5x).
(b) Exposure-artifact check: R^2 of the candidate's realized ``target``
    path against a mean-notional-matched flat rescale of v4's own
    unmodified ``target`` path, same window. R^2 > ~0.95 means "flat
    rescale, not a mechanism" and is an automatic FAIL regardless of any
    Sharpe number.
(c) ETH falsification: identical construction (same mechanism, same
    architecture, ETH's own MVRV series via ``asset="ETH"``) must not fail
    decisively or oppositely-signed vs. the BTC control.
(d) BTC pre-2020 control (2017-01-01 -> 2019-12-31): reported for context,
    "does it also work / not hurt outside the fitted window" per
    ROUTINE.md's standard practice; a decisive negative result here counts
    against promotion even though it is not one of the four primary
    must-pass gates below.
(e) Plateau, not peak: the (thresh_hi x mvrv_weight) neighbourhood is
    reported in full, not just the best cell.

**Pre-registered decision rule, fixed before any result is read:** CLEARS
PRE-REGISTRATION (ready for holdout) only if ALL of (a), (b), (c), (e)
pass AND (d) does not decisively fail. Any single failure among (a)/(b)/
(c)/(e) is an automatic NEGATIVE and the 2023+ holdout is never touched.

Code reuse decision, stated plainly
-------------------------------------
``_anchor_votes`` is DUPLICATED (not imported) from
``experiments/kelly_regime_v14_macro_lead.py`` / ``_v16_stablecoin_confirm.py``,
the same precedent every prior INFO-axis round in this project has used
for that helper (it is v4's own vote computation, private experiment code,
not shared infrastructure). ``compute_mvrv_z`` IS imported unchanged from
this round's own ``r74_conservative_mvrv_signal.py``. Neither
``kelly_regime.py``, ``kelly_regime_v3.py``, ``kelly_regime_v4.py``, nor
``src/tradebot/data.py`` is edited anywhere in this branch.

Holdout discipline
-------------------
``assert_no_holdout()`` is called after every data load in this file
(BTC bars, ETH bars, and inside ``compute_mvrv_z`` itself via its own
truncation). BTC/ETH bar frames are truncated to strictly before
``OOS_START`` at load time -- no 2023+ bar or MVRV value is ever held in
memory by this file, let alone read or printed.

Usage
-----
    python experiments/r74_conservative_mvrv_level.py descriptive
    python experiments/r74_conservative_mvrv_level.py sweep        # step 3 (inner-train)
    python experiments/r74_conservative_mvrv_level.py select       # step 3 (inner-validation)
    python experiments/r74_conservative_mvrv_level.py artifact     # falsification (b)
    python experiments/r74_conservative_mvrv_level.py causality    # lookahead probe (price + MVRV pathway)
    python experiments/r74_conservative_mvrv_level.py eth          # falsification (c)
    python experiments/r74_conservative_mvrv_level.py pre2020      # falsification (d)
    python experiments/r74_conservative_mvrv_level.py all          # everything, in order
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import MVRV_FILES, load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r74_conservative_mvrv_signal import compute_mvrv_z  # noqa: E402

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures 5x", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")     # inner-train
VALID = ("2021-01-01", "2022-12-31")     # inner-validation
PRE2020 = ("2017-01-01", "2019-12-31")   # falsification (d)
OOS_START = "2023-01-01"                 # NEVER read in this file

INCUMBENT = "kelly_regime_v4"
DATA_DIR = ROOT / "data"

N_EVALUATED = 0  # distinct configurations evaluated, project-trials count
_SEEN_CONFIGS: set[str] = set()


# ---------------------------------------------------------------- holdout guard


def assert_no_holdout(df: pd.DataFrame) -> None:
    """Hard guard: the max timestamp in any frame this file touches must be
    strictly before OOS_START. Raises, never silently truncates further --
    every load site below already truncates explicitly; this is the
    independent second check."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


# --------------------------------------------------------------------- data


def build_mvrv_dataframe(kind: str = "spot") -> tuple[pd.DataFrame, str]:
    """Canonical BTC OHLCV, truncated before OOS_START, with a causal
    ``mvrv_z_visible`` column merged on."""
    df, label = load_dataset(DATA_DIR, kind)
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    z = compute_mvrv_z(df, DATA_DIR, asset="BTC")
    df["mvrv_z_visible"] = z
    assert_no_holdout(df)
    return df, label


DF, LABEL = build_mvrv_dataframe("spot")
print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
      f"(data: {LABEL}); MVRV-z coverage {DF['mvrv_z_visible'].notna().sum():,} bars "
      f"from {DF['mvrv_z_visible'].dropna().index[0]:%Y-%m-%d}"
      if DF["mvrv_z_visible"].notna().any() else "MVRV-z: NO COVERAGE", file=sys.stderr)


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0,
            config_key: str | None = None):
    """One backtest -> Metrics. ``config_key`` counts a DISTINCT configuration
    exactly once across the whole session, however many market/period cells
    it is subsequently re-scored on (v4 control and diagnostic re-reads pass
    config_key=None and are never counted)."""
    global N_EVALUATED
    if config_key is not None and config_key not in _SEEN_CONFIGS:
        _SEEN_CONFIGS.add(config_key)
        N_EVALUATED += 1
    frame = DF if df is None else df
    assert_no_holdout(frame if end is None else frame.loc[:end])
    result = run_period(strategy, frame, start, end, market=market,
                         start_balance=balance, data_label=LABEL)
    return compute_metrics(result), result


def line(tag, m, market_name=""):
    print(f"  {tag:42s} {market_name:10s} final=${m.final_balance:>11,.0f} "
          f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>7,.0f}{'  LIQUIDATED' if m.liquidated else ''}")


# ---------------------------------------------------------------- the vote


def _anchor_votes(close: pd.Series, horizons=(20, 40, 80), band: float = 0.01) -> dict:
    """Exactly kelly_regime_v4's own per-anchor latched vote. Duplicated
    (not imported) from kelly_regime_v14_macro_lead.py / the prior INFO-axis
    rounds' own private copies -- see this module's docstring, "Code reuse
    decision"."""
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


def _mvrv_vote(z: pd.Series, thresh_hi: float, gap: float) -> pd.Series:
    """Latched MVRV-euphoria vote, 0/1, hysteresis on ``z``.

    0 ("euphoric", dilutive) once z is confirmed above thresh_hi.
    1 ("not euphoric", no dilution) once z is confirmed below thresh_hi-gap.
    Latched in between. Defaults to 1 (no dilution) wherever z is NaN
    (absent data or still inside the 365-day expanding-stat warmup) -- the
    same "absence -> no dilution, falls back to v4's own anchor-only vote"
    convention every prior INFO signal in this project uses.
    """
    thresh_lo = thresh_hi - gap
    raw = np.where(z > thresh_hi, 0.0, np.where(z < thresh_lo, 1.0, np.nan))
    vote = pd.Series(raw, index=z.index).ffill().fillna(1.0)
    return vote


def _size_from_frac(df: pd.DataFrame, frac: np.ndarray, strat: KellyRegimeV3) -> np.ndarray:
    """v3's own conditional vol-targeting sizer, byte-for-byte, factored out
    so both the candidate and the pure-anchor-only comparison arms share
    identical sizing logic -- only ``frac`` differs between them."""
    close = df["close"]
    r = np.log(close).diff()
    vol = (r.ewm(span=strat.vol_span, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    slow = (pd.Series(vol).ewm(span=strat.anchor_span_days * BARS_PER_DAY,
                                min_periods=BARS_PER_DAY).mean().to_numpy())

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(strat.target_vol / vol, strat.max_leverage)
        steady = np.minimum(strat.target_vol / slow, strat.max_leverage)
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
                vstate = 1 if x > strat.high_in else (-1 if x < strat.low_in else 0)
            elif vstate == 1 and x < strat.high_out:
                vstate = 0
            elif vstate == -1 and x > strat.low_out:
                vstate = 0
        scale = full[i] if vstate != 0 else steady[i]
        desired = frac[i] * scale
        if abs(desired - pos) > strat.deadband:
            pos = desired
        target[i] = pos
    return target


class KellyRegimeR74MvrvConfirm(KellyRegimeV3):
    """v4's 3-anchor vote plus a 4th, weighted CONFIRMING vote from MVRV
    euphoria-extreme detection. Mechanism:
    ``frac = (anchor_sum + mvrv_weight * mvrv_vote) / (3 + mvrv_weight)``,
    where ``anchor_sum`` is v4's own three 0/1 latched price-anchor votes
    (unchanged) and ``mvrv_vote`` is the latched MVRV-euphoria state above
    (0/1, only ever fires dilutive). R-55's validated confirming-vote
    architecture, fed a genuinely novel signal (MVRV level, not rate of
    change). ``mvrv_weight=0`` recovers v4 exactly (identity check,
    verified in ``causality()``).
    """

    name = "kelly_regime_r74_mvrv_confirm"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 thresh_hi: float = 2.5, gap: float = 0.5, mvrv_weight: float = 0.33,
                 asset: str = "BTC", **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.thresh_hi = thresh_hi
        self.gap = gap
        self.mvrv_weight = mvrv_weight
        self.asset = asset

    def _z_series(self, df: pd.DataFrame) -> pd.Series:
        if "mvrv_z_visible" in df.columns and self.asset == "BTC":
            return df["mvrv_z_visible"]
        return compute_mvrv_z(df, DATA_DIR, asset=self.asset)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        votes = _anchor_votes(close, self.horizons, self.band)
        anchor_sum = sum(votes.values())  # 0..3 per bar
        n_anchors = float(len(votes))

        z = self._z_series(df)
        if self.mvrv_weight > 0:
            mvrv_vote = _mvrv_vote(z, self.thresh_hi, self.gap)
        else:
            mvrv_vote = pd.Series(1.0, index=df.index)

        combined = (anchor_sum + self.mvrv_weight * mvrv_vote) / (n_anchors + self.mvrv_weight)
        frac = combined.to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        target = _size_from_frac(df, frac, self)

        df["target"] = target
        df["r74_frac"] = frac
        df["r74_mvrv_vote"] = mvrv_vote.to_numpy()
        df["r74_mvrv_z"] = z.to_numpy()
        df["r74_anchor_sum"] = anchor_sum.to_numpy()
        return df


# ------------------------------------------------------------- the grid

# THRESH_HI: conventional statistical-extreme cutoffs (2/2.5/3 standard
# deviations of the expanding Z-score), fixed a priori -- not fit to any
# observed lead/lag or Sharpe result. GAP: a single fixed hysteresis band
# (0.5, half the smallest inter-threshold spacing above), matching this
# project's existing hysteresis convention (R-53/R-54/R-55 all use a
# fixed gap alongside a swept threshold).
THRESH_HI = (2.0, 2.5, 3.0)
GAP = 0.5

# WEIGHTS: mirrors R-55's own validated confirming-vote grid (a fraction of
# ONE anchor vote, from light to moderate dilution) -- 0.0 is the explicit
# identity control (recovers v4 exactly, independent of threshold).
WEIGHTS = (0.15, 0.33, 0.5)

PRIMARY_KW = dict(thresh_hi=2.5, gap=GAP, mvrv_weight=0.33)


def _grid():
    """10 configs: identity control (1) + 3 thresh_hi x 3 weights (9)."""
    out = [("identity (mvrv_weight=0)", dict(thresh_hi=2.5, gap=GAP, mvrv_weight=0.0))]
    for thresh_hi in THRESH_HI:
        for weight in WEIGHTS:
            label = f"thresh={thresh_hi:.2f} gap={GAP:.2f} w={weight:.2f}"
            out.append((label, dict(thresh_hi=thresh_hi, gap=GAP, mvrv_weight=weight)))
    return out


def _config_key(kw: dict) -> str:
    return f"mvrv_confirm|thresh={kw['thresh_hi']:.3f}|gap={kw['gap']:.3f}|w={kw['mvrv_weight']:.3f}"


# ------------------------------------------------------- step 2b: descriptive


def descriptive() -> None:
    """Step 2b -- purely descriptive, no free parameter fitted: MVRV-vote
    transition frequency vs. the price anchors' own, over the combined
    inner-train + inner-validation window."""
    lo, hi = TRAIN[0], VALID[1]
    frame = DF.loc[lo:hi]
    close = frame["close"]

    print(f"descriptive window (inner-train + inner-validation): {lo} -> {hi}")
    print("\nprice-anchor vote transition counts over the SAME window (context):")
    for days in (20, 40, 80):
        anchor = DF["close"].rolling(int(days * BARS_PER_DAY)).mean().loc[lo:hi]
        v = pd.Series(np.where(close > anchor * 1.01, 1.0,
                                np.where(close < anchor * 0.99, 0.0, np.nan)),
                      index=close.index).ffill().fillna(0.0)
        print(f"  {days:>3d}d price anchor: {int(v.ne(v.shift()).sum()):>4d} vote flips")

    z = frame["mvrv_z_visible"]
    print(f"\nmvrv_z coverage in window: {z.notna().sum():,} / {len(z):,} bars "
          f"(min={z.min():.2f}, max={z.max():.2f}, mean={z.mean():.2f})")
    for thresh_hi in THRESH_HI:
        vote = _mvrv_vote(z, thresh_hi, GAP)
        flips_to_euphoric = int(((vote == 0.0) & (vote.shift() == 1.0)).sum())
        flips_to_calm = int(((vote == 1.0) & (vote.shift() == 0.0)).sum())
        frac_euphoric = float((vote == 0.0).mean())
        print(f"  thresh_hi={thresh_hi:.2f} gap={GAP:.2f}: {flips_to_euphoric} euphoria-onset "
              f"event(s), {flips_to_calm} calm-return event(s), "
              f"{frac_euphoric:.1%} of bars latched euphoric")


# --------------------------------------------------------------- step 3: sweep


def sweep() -> None:
    """Step 3: every config on inner-train ONLY, spot primary."""
    print(f"\nINNER-TRAIN {TRAIN} / spot -- benchmarks:")
    for name in ("buy_and_hold", INCUMBENT):
        m, _ = measure(get_strategy(name), *TRAIN, market=SPOT)
        line(name, m, "spot")

    print(f"\nINNER-TRAIN {TRAIN} / spot -- candidate configurations:")
    for label, kw in _grid():
        m, _ = measure(KellyRegimeR74MvrvConfirm(**kw), *TRAIN, market=SPOT,
                        config_key=_config_key(kw))
        line(label, m, "spot")

    print(f"\nconfigurations evaluated (step 3, distinct thresh/weight pairs): {N_EVALUATED}")


# -------------------------------------------------------------- step 3: select


def select():
    """Every config on inner-validation ONLY, BOTH markets, vs v4 control."""
    rows = []
    print(f"\nINNER-VALIDATION {VALID} -- v4 control:")
    ctl = {}
    for mname, market in MARKETS:
        m, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        ctl[mname] = m
        line(f"{INCUMBENT} (control)", m, mname)

    print(f"\nINNER-VALIDATION {VALID} -- candidate configurations:")
    best_label, best_kw, best_score = None, None, -1e9
    for label, kw in _grid():
        cell = {}
        for mname, market in MARKETS:
            m, _ = measure(KellyRegimeR74MvrvConfirm(**kw), *VALID, market=market,
                            config_key=_config_key(kw))
            cell[mname] = m
            line(label, m, mname)
            rows.append({"label": label, **kw, "market": mname,
                         "final": m.final_balance, "sharpe": m.sharpe,
                         "max_dd": m.max_drawdown_pct, "trades": m.num_trades})
        m_train, _ = measure(KellyRegimeR74MvrvConfirm(**kw), *TRAIN, market=SPOT)
        score = min(m_train.sharpe, cell["spot"].sharpe)
        if score > best_score:
            best_label, best_kw, best_score = label, kw, score

    print(f"\nbest by min(train,valid) spot Sharpe: {best_label}  (score={best_score:.2f})")
    print(f"v4 control spot Sharpe: {ctl['spot'].sharpe:.2f}   "
          f"v4 control futures Sharpe: {ctl['futures 5x'].sharpe:.2f}")

    print("\nside-by-side: best candidate's TRAIN window vs VALIDATION window (overfitting-signature check):")
    for mname, market in MARKETS:
        m_train, _ = measure(KellyRegimeR74MvrvConfirm(**best_kw), *TRAIN, market=market)
        m_valid, _ = measure(KellyRegimeR74MvrvConfirm(**best_kw), *VALID, market=market)
        m_train_v4, _ = measure(get_strategy(INCUMBENT), *TRAIN, market=market)
        m_valid_v4, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        print(f"  {mname}:")
        print(f"    candidate  TRAIN final=${m_train.final_balance:>11,.0f} sharpe={m_train.sharpe:.2f} DD={m_train.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid.final_balance:>11,.0f} sharpe={m_valid.sharpe:.2f} DD={m_valid.max_drawdown_pct:.1f}%")
        print(f"    v4         TRAIN final=${m_train_v4.final_balance:>11,.0f} sharpe={m_train_v4.sharpe:.2f} DD={m_train_v4.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid_v4.final_balance:>11,.0f} sharpe={m_valid_v4.sharpe:.2f} DD={m_valid_v4.max_drawdown_pct:.1f}%")
        print(f"    candidate - v4 (valid): Delta sharpe={m_valid.sharpe - m_valid_v4.sharpe:+.3f}  "
              f"Delta DD={m_valid.max_drawdown_pct - m_valid_v4.max_drawdown_pct:+.1f}pp")

    print("\nparameter-neighbourhood plateau check -- falsification (e) -- "
          "(spot, inner-validation Sharpe, all thresh_hi x weight cells):")
    grid_by_key = {(t, w): None for t in THRESH_HI for w in WEIGHTS}
    for thresh_hi in THRESH_HI:
        for weight in WEIGHTS:
            m, _ = measure(KellyRegimeR74MvrvConfirm(thresh_hi=thresh_hi, gap=GAP, mvrv_weight=weight),
                            *VALID, market=SPOT)
            grid_by_key[(thresh_hi, weight)] = m.sharpe
    for thresh_hi in THRESH_HI:
        row = "  ".join(f"w={w:.2f}:{grid_by_key[(thresh_hi, w)]:.2f}" for w in WEIGHTS)
        print(f"  thresh_hi={thresh_hi:.2f}  {row}")
    m_ident, _ = measure(KellyRegimeR74MvrvConfirm(mvrv_weight=0.0), *VALID, market=SPOT)
    print(f"  identity (mvrv_weight=0, should equal v4 spot Sharpe {ctl['spot'].sharpe:.2f}): {m_ident.sharpe:.2f}")

    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    return best_label, best_kw


# ------------------------------------------------------------------ artifact


def artifact(kw: dict | None = None) -> None:
    """Falsification (b), mandatory exposure-artifact check: R^2 of the
    candidate's target series against a mean-notional-matched flat rescale
    of v4's own target, on inner-validation, both markets. R^2 > 0.95 ->
    artifact, automatic FAIL regardless of any Sharpe number."""
    kw = kw or PRIMARY_KW
    print(f"exposure-artifact check, candidate={kw}")
    v4 = get_strategy(INCUMBENT)
    for mname, market in MARKETS:
        cand = KellyRegimeR74MvrvConfirm(**kw)
        lo = int(DF.index.searchsorted(VALID[0]))
        hi = int(DF.index.searchsorted(VALID[1], side="right"))
        prefix = min(lo, max(cand.warmup, v4.warmup))
        frame = DF.iloc[lo - prefix:hi]

        v4_prepared = v4.prepare(frame.copy())
        cand_prepared = cand.prepare(frame.copy())
        v4_t = v4_prepared["target"].to_numpy()[prefix:]
        cand_t = cand_prepared["target"].to_numpy()[prefix:]

        mean_abs_v4 = float(np.mean(np.abs(v4_t)))
        mean_abs_cand = float(np.mean(np.abs(cand_t)))
        alpha = mean_abs_cand / mean_abs_v4 if mean_abs_v4 > 0 else 0.0
        rescaled = alpha * v4_t
        ss_res = float(np.sum((cand_t - rescaled) ** 2))
        ss_tot = float(np.sum((cand_t - cand_t.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        corr = float(np.corrcoef(cand_t, v4_t)[0, 1])
        verdict = "EXPOSURE-LEVEL ARTIFACT" if r2 > 0.95 else "genuinely different exposure shape"
        print(f"  {mname:10s} mean|v4|={mean_abs_v4:.3f} mean|cand|={mean_abs_cand:.3f} "
              f"alpha={alpha:.3f}  R^2(cand vs alpha*v4)={r2:.4f}  raw corr={corr:.4f}  {verdict}")


# ------------------------------------------------------------------ causality


def _make_tampered_mvrv_dir(cut_day: pd.Timestamp, factor: float, tmp_root: Path,
                             asset: str = "BTC") -> Path:
    """Copy the real MVRV CSV into a fresh dir, multiplying every row dated
    on/after ``cut_day`` by ``factor``. Used only for the causality probe
    below -- never writes into the real ``data/`` dir."""
    fname = MVRV_FILES[asset]
    out_dir = tmp_root / f"mvrv_x{factor:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_DIR / fname)
    dates = pd.to_datetime(raw["timestamp"])
    mask = dates >= cut_day.tz_localize(None)
    raw.loc[mask, "mvrv"] = raw.loc[mask, "mvrv"] * factor
    raw.to_csv(out_dir / fname, index=False, compression="gzip")
    return out_dir


def causality(kw: dict | None = None) -> None:
    """Two-independent-pathway tamper probe: price OHLCV AND the MVRV input
    tampered independently after a cut. Every decision at or before the cut
    must be unchanged. Restricted to strictly pre-2023 bars."""
    kw = kw or PRIMARY_KW

    df = DF.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]
    cut_day = df.index[cut].normalize()

    def strategy_for(data_dir: Path | None):
        s = KellyRegimeR74MvrvConfirm(**kw)
        if data_dir is not None:
            def patched(frame, _dd=data_dir):
                if "mvrv_z_visible" in frame.columns:
                    # price-only tamper path: reuse the precomputed (real-MVRV) column
                    return frame["mvrv_z_visible"]
                return compute_mvrv_z(frame, _dd, asset="BTC")
            s._z_series = patched  # noqa: SLF001 (deliberate, test-only monkeypatch)
        return s

    def run_probe(name, tamper_ohlcv_fn=None, mvrv_dir_up=None, mvrv_dir_down=None):
        up, down = df.copy(), df.copy()
        if tamper_ohlcv_fn is not None:
            tamper_ohlcv_fn(up, down)
        if mvrv_dir_up is not None:
            up = up.drop(columns=["mvrv_z_visible"])
            down = down.drop(columns=["mvrv_z_visible"])

        def decisions(frame, data_dir):
            s = strategy_for(data_dir)
            prepared = s.prepare(frame.copy())
            broker = _fresh_broker()
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        a = decisions(up, mvrv_dir_up)
        b = decisions(down, mvrv_dir_down)
        bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
        print(f"[{name}] tampered from bar {cut:,} of {len(df):,} "
              f"(calendar day {cut_day.date()}); checked bars {bars}")
        print("  FAIL - reads the future at bars " + str(bad) if bad
              else "  PASS - every decision at or before the cut is unchanged")

        pa = strategy_for(mvrv_dir_up).prepare(up.copy())
        pb = strategy_for(mvrv_dir_down).prepare(down.copy())
        for col in ("target", "r74_frac", "r74_mvrv_vote", "r74_anchor_sum"):
            diff = np.abs(pa[col].to_numpy()[:cut].astype(float)
                          - pb[col].to_numpy()[:cut].astype(float))
            worst = float(np.nanmax(diff))
            print(f"    column {col:16s} max |difference| before the cut = {worst:.3e}"
                  f"  {'PASS' if worst < 1e-9 else 'FAIL'}")

    def tamper_ohlcv(up, down):
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    tmp_root = Path(tempfile.mkdtemp(prefix="r74_mvrv_confirm_causality_"))
    try:
        run_probe("PRICE tamper (standard)", tamper_ohlcv_fn=tamper_ohlcv)

        mvrv_dir_up = _make_tampered_mvrv_dir(cut_day, 50.0, tmp_root)
        mvrv_dir_down = _make_tampered_mvrv_dir(cut_day, 1.0 / 50.0, tmp_root)
        run_probe("MVRV tamper (the new pathway)",
                   mvrv_dir_up=mvrv_dir_up, mvrv_dir_down=mvrv_dir_down)

        run_probe("both at once", tamper_ohlcv_fn=tamper_ohlcv,
                   mvrv_dir_up=mvrv_dir_up, mvrv_dir_down=mvrv_dir_down)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    v4 = get_strategy(INCUMBENT)
    ident = KellyRegimeR74MvrvConfirm(thresh_hi=2.5, gap=GAP, mvrv_weight=0.0)
    frame = df.iloc[-20_000:].copy()
    t_v4 = v4.prepare(frame.copy())["target"].to_numpy()
    t_ident = ident.prepare(frame.copy())["target"].to_numpy()
    worst = float(np.nanmax(np.abs(t_v4 - t_ident)))
    print(f"\nidentity check (mvrv_weight=0 recovers v4 exactly): "
          f"max|diff|={worst:.3e}  {'PASS' if worst < 1e-9 else 'FAIL'}")


def _fresh_broker():
    from tradebot.broker import PaperBroker
    return PaperBroker(market=FUTURES, start_balance=10_000.0)


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Falsification (c), pre-registered rule below, fixed before running.

    ETH-USD Bitfinex spot (2016-03 -> 2019-12-31, well before OOS_START)
    against ETH's own MVRV series (2018-01-01 ->, per this round's brief).
    Identical construction (``asset="ETH"``): if the candidate is not at
    least comparable to v4 on ETH, or is visibly worse on ETH than on the
    BTC control through the identical code, this direction fails. An
    ETH-only failure must be reported, not hidden.
    """
    eth_spot = load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")
    eth_spot = eth_spot.loc[eth_spot.index < pd.Timestamp(OOS_START, tz=eth_spot.index.tz)].copy()
    assert_no_holdout(eth_spot)
    eth_z = compute_mvrv_z(eth_spot, DATA_DIR, asset="ETH")
    eth_df = eth_spot.copy()
    eth_df["mvrv_z_visible"] = eth_z
    assert_no_holdout(eth_df)

    overlap = eth_df["mvrv_z_visible"].dropna()
    print(f"ETH spot file: {len(eth_spot):,} bars  {eth_spot.index[0]:%Y-%m-%d} -> "
          f"{eth_spot.index[-1]:%Y-%m-%d}")
    if len(overlap):
        print(f"ETH MVRV-z coverage overlapping ETH spot: {len(overlap):,} bars "
              f"visible from {overlap.index[0]:%Y-%m-%d} to {overlap.index[-1]:%Y-%m-%d}")
    else:
        print("ETH MVRV-z coverage overlapping ETH spot: NONE")

    frames = {"BTC (control)": DF, "ETH (test)": eth_df}
    results = {}
    for asset_name, frame in frames.items():
        asset = "BTC" if "BTC" in asset_name else "ETH"
        print(f"\n{asset_name}  {len(frame):,} bars  {frame.index[0]:%Y-%m-%d} -> {frame.index[-1]:%Y-%m-%d}")
        results[asset_name] = {}
        for mname, market in MARKETS:
            m_v4, _ = measure(get_strategy(INCUMBENT), None, None, df=frame, market=market)
            line(f"{INCUMBENT} (control)", m_v4, mname)
            asset_results = {"v4": m_v4}
            for label, kw in _grid():
                kw_asset = dict(kw, asset=asset)
                cand = KellyRegimeR74MvrvConfirm(**kw_asset)
                m_c, _ = measure(cand, None, None, df=frame, market=market)
                line(f"mvrv[{label}]", m_c, mname)
                asset_results[label] = m_c
            results[asset_name][mname] = asset_results

    print("\nfalsification verdict per config (candidate/v4 final-balance ratio, BTC control vs ETH test):")
    any_fail = False
    for label, kw in _grid():
        for mname, market in MARKETS:
            btc_r = (results["BTC (control)"][mname][label].final_balance
                      / results["BTC (control)"][mname]["v4"].final_balance)
            eth_r = (results["ETH (test)"][mname][label].final_balance
                      / results["ETH (test)"][mname]["v4"].final_balance)
            worse_on_eth = eth_r < btc_r - 0.02
            eth_beats_v4 = eth_r >= 1.0
            flag = "FAIL" if (worse_on_eth and not eth_beats_v4) else ("caution" if worse_on_eth else "ok")
            any_fail = any_fail or flag == "FAIL"
            print(f"  {label:32s} {mname:10s} BTC ratio={btc_r:.3f}x  ETH ratio={eth_r:.3f}x  [{flag}]")
    print(f"\nprimary candidate ({PRIMARY_KW}) checked above; "
          f"overall falsification verdict: {'FAIL' if any_fail else 'no outright FAIL by the pre-registered rule'}")


# ------------------------------------------------------------------------------ pre2020


def pre2020_control() -> None:
    """Falsification (d): BTC pre-2020 control, 2017-01-01 -> 2019-12-31.
    Reported for context; a decisive negative here counts against
    promotion even though it is not one of the four primary must-pass
    gates."""
    print(f"BTC PRE-2020 CONTROL {PRE2020}:")
    for mname, market in MARKETS:
        m_v4, _ = measure(get_strategy(INCUMBENT), *PRE2020, market=market)
        line(f"{INCUMBENT} (control)", m_v4, mname)
        for label, kw in _grid():
            m_c, _ = measure(KellyRegimeR74MvrvConfirm(**kw), *PRE2020, market=market,
                              config_key=_config_key(kw))
            line(label, m_c, mname)
            print(f"    Delta sharpe={m_c.sharpe - m_v4.sharpe:+.3f}  "
                  f"Delta DD={m_c.max_drawdown_pct - m_v4.max_drawdown_pct:+.1f}pp")

    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


# --------------------------------------------------------------------- driver


def all_checks() -> None:
    print("=" * 78)
    print("STEP 2b -- descriptive: MVRV-euphoria vote frequency vs anchor votes")
    print("=" * 78)
    descriptive()
    print("\n" + "=" * 78)
    print("STEP 3 -- sweep (inner-train)")
    print("=" * 78)
    sweep()
    print("\n" + "=" * 78)
    print("STEP 3 -- select (inner-validation, both markets)")
    print("=" * 78)
    select()
    print("\n" + "=" * 78)
    print("EXPOSURE-ARTIFACT CHECK -- falsification (b)")
    print("=" * 78)
    artifact()
    print("\n" + "=" * 78)
    print("CAUSALITY / NO-LOOKAHEAD PROBE")
    print("=" * 78)
    causality()
    print("\n" + "=" * 78)
    print("ETH FALSIFICATION TEST -- falsification (c)")
    print("=" * 78)
    eth()
    print("\n" + "=" * 78)
    print("BTC PRE-2020 CONTROL -- falsification (d)")
    print("=" * 78)
    pre2020_control()
    print(f"\ntotal distinct configurations evaluated (N_EVALUATED): {N_EVALUATED}")
    print(f"max timestamp read anywhere in this session: {DF.index.max()} (< {OOS_START})")


if __name__ == "__main__":
    cmds = {"descriptive": descriptive, "sweep": sweep, "select": select,
            "artifact": artifact, "causality": causality, "eth": eth,
            "pre2020": pre2020_control, "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/r74_conservative_mvrv_level.py [{'|'.join(cmds)}]")
