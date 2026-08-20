#!/usr/bin/env python
"""NOVEL branch, parallel round R-53: VIX/DXY macro stress as a FOURTH,
FASTER vote inside kelly_regime_v4's own REGIME GATE -- not a size
multiplier on top of it.

Idea, one sentence
------------------
The literature argues equity-implied fear (VIX) and dollar-strength flows
(DXY) *lead* crypto risk-off moves rather than merely correlating with
them contemporaneously (Luo, Tsai & Yen 2024/2025, SSRN, on the VIX
term-structure slope as a primary determinant of Bitcoin returns; IMF WP
2023/213 on crypto-equity spillover intensifying specifically during
stress periods) -- so a macro-stress vote should be able to flip
`kelly_regime_v4`'s gate bearish FASTER than its three slow price anchors
(20/40/80-day SMAs) can on their own, precisely at the moment a laggy gate
costs the most: the start of a crash, before price has moved 1% below any
anchor. This file adds `stress_z` (imported unchanged from the shared,
operator-authored `experiments/_macro_signal.py`) as a fourth latched vote
input, alongside v4's own three anchor votes, using the SAME
np.where+ffill+fillna hysteresis discipline the anchors already use --
never as a continuous multiplicative haircut on the vol-targeted SIZE
formula, which is the disjoint, parallel CONSERVATIVE branch's job on this
same round (`kelly_regime_v14_macro_brake.py`, not read, not coordinated
with).

Constraint attacked: INFO (one price series). VIX and DXY are two
genuinely new, price-independent information channels -- the second
attempt at INFO this project has made, after on-chain metrics (B-07,
R-44, both branches NEGATIVE for reasons unrelated to data quality).

Not a duplicate of
-------------------
- R-44's `kelly_regime_v10_hashribbon_vote.py` (this file's own structural
  template, per the round's own brief): SAME architecture family (a
  latched 4th vote precision-weighted into the vote-generation mechanism,
  v3's SIZE formula untouched), but a DIFFERENT signal (hash-ribbon miner
  capitulation, once-a-cycle and price-independent via mining economics)
  vs. THIS file's macro stress (VIX/DXY, market-wide and independent via a
  different mechanism -- equity/FX risk appetite, not BTC's own supply
  side). R-44's own vote only ever pushes exposure UP (recovery=bullish);
  this file's vote is deliberately asymmetric the OTHER way -- it only
  ever pushes exposure DOWN (stress=bearish veto-leaning), per the
  mechanism section below.
- L-04/L-01 (`kelly_regime_v4`, the incumbent): this file's `macro_weight
  -> 0` recovers v4 EXACTLY (an identity check run in `causality()` and
  implicit in the grid) -- it is an additive vote input, not a
  replacement, exactly as R-44's hash-ribbon vote was.
- L-12 is not a strategy ID in this ledger as of R-52's rows; if it refers
  to a different round's conditional-vol finding, that SIZE-axis
  mechanism (v3's steady/full vol-target switch) is imported unchanged
  here and never modified.
- B-07/R-08/R-10: R-08 found a BETTER volatility forecast made this
  strategy family WORSE (de-levering more promptly into BTC's
  highest-forward-Sharpe high-vol states, Baur & Dimpfl 2018's inverse
  leverage effect). This file's mechanism is built to respect that sign
  discipline: the macro vote never forecasts volatility or touches the
  SIZE formula at all -- it only ever removes exposure through the GATE,
  the mechanism L-04/L-01 says is where the project's actual edge lives,
  and only in the risk-off direction the spillover literature's mechanism
  predicts, never the reverse.
- The disjoint CONSERVATIVE branch running in parallel this round
  (`kelly_regime_v14_macro_brake.py`, per the round brief): architecturally
  different by design -- that branch's job is a continuous multiplicative
  haircut on SIZE; this file never touches the vol-targeting scale at all,
  only the vote fraction that feeds it. Not read, not coordinated with.

Mechanism, precisely
---------------------
`stress_z = 0.5*vix_z + 0.5*dxy_mom_z` (imported unchanged from
`_macro_signal.py`; positive = elevated fear and/or dollar strengthening =
risk-off). A latched macro vote, `macro_vote in {0, 1}`:

    macro_vote -> 0 ("stress")   when stress_z crosses ABOVE thresh_hi
    macro_vote -> 1 ("calm")     when stress_z crosses BELOW thresh_lo
    macro_vote unchanged (latched) while thresh_lo <= stress_z <= thresh_hi
    macro_vote defaults to 1 ("calm") before the first crossing, or
        wherever macro data is unavailable (data absent -> no veto,
        falls back to v4's own anchor-only vote exactly)

`thresh_hi = 1.0` (one standard deviation of trailing stress; a fixed,
a-priori choice, not swept, the same discipline v4's own 1% anchor band
receives). `gap = thresh_hi - thresh_lo` IS swept (0.0/0.75/1.25):
`gap=0.0` collapses to a single memoryless threshold with no hysteresis
dead-zone -- the explicit negative control this round's brief calls for.

Combined vote: `frac = (anchor_sum + macro_weight*macro_vote) / (3 +
macro_weight)`, where `anchor_sum` is v4's own three UNCHANGED 0/1 latched
price-anchor votes. `macro_weight` is swept (0.15/0.33/0.5/1.0, the last
an explicit unweighted-4-way-average negative control, matching R-44's
convention) rather than fixed at 1.0 for the same precision argument R-44
used: a macro regime flip is a much rarer, lower-frequency event than a
20-day price-anchor flip (see `descriptive()`).

**This is a REGIME-DETECTION mechanism, not a scale multiplier, and the
distinction is load-bearing, not cosmetic.** Because `macro_vote` defaults
to 1 ("calm") and only ever moves to 0 ("stress") on a confirmed
above-threshold reading, the vote can ONLY ever pull `frac` DOWN relative
to what the three anchors alone would produce -- it never manufactures
bullish exposure the anchors themselves would not already grant, mirroring
R-44's own one-directional discipline in the opposite sign (R-44's vote
only pushes UP; this one only pushes DOWN, matching the literature's
risk-off-specific claim). And because it is combined with the anchor
votes BEFORE the vol-targeting SIZE formula runs, a macro flip changes
`frac` on the exact bar it latches -- not smoothed, not scaled by
realized volatility, not gated behind the SIZE formula's own separate
high/low-vol hysteresis (`vstate` below) -- which is precisely the "faster
than the slowest anchor" property the mechanism claims to test.

Pre-registered failure modes (named before any code ran, per the round brief)
-------------------------------------------------------------------------------
(1) The macro vote never actually LEADS the price anchors in the handful
    of stress episodes available (2018, 2020-03, 2022) -- it flips at the
    same time or later, adding noise without adding lead time. Checked
    directly in `leadtime()` by comparing flip *timestamps*, not
    aggregate Sharpe.
(2) It doesn't beat v4 on inner-validation Sharpe/drawdown by more than
    the +/-0.2 noise floor (R-20) -- or does, but the edge traces to the
    N~=3 stress episodes that built the composite in the first place
    (an artifact of a tiny event count, not a real edge).
(3) Fails the causality tamper probe (price OR macro pathway).
(4) Fails the ETH falsification test, OR the averaged-vote mechanism adds
    nothing over its own simplest ablation (a hard override instead of a
    weighted average) -- the R-40/R-46 "elaboration adds nothing over a
    simpler baseline" failure pattern, checked explicitly in `ablation()`.

Usage
-----
    python experiments/kelly_regime_v14_macro_lead.py descriptive  # step 2b, vote-frequency context
    python experiments/kelly_regime_v14_macro_lead.py leadtime     # failure mode (1)
    python experiments/kelly_regime_v14_macro_lead.py sweep        # step 3 (inner-train)
    python experiments/kelly_regime_v14_macro_lead.py select       # step 3 (inner-validation)
    python experiments/kelly_regime_v14_macro_lead.py ablation     # failure mode (4), weighted-vote vs hard-override
    python experiments/kelly_regime_v14_macro_lead.py artifact     # exposure-artifact check
    python experiments/kelly_regime_v14_macro_lead.py causality    # lookahead probe (price + macro pathway)
    python experiments/kelly_regime_v14_macro_lead.py eth          # failure mode (4), ETH falsification
    python experiments/kelly_regime_v14_macro_lead.py all          # everything, in order
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
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments._macro_signal import compute_macro_stress  # noqa: E402

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures 5x", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")     # inner-train
VALID = ("2021-01-01", "2022-12-31")     # inner-validation
OOS_START = "2023-01-01"                 # never read in this file

INCUMBENT = "kelly_regime_v4"
DATA_DIR = ROOT / "data"

N_EVALUATED = 0  # distinct configurations evaluated, project-trials count
_SEEN_CONFIGS: set[str] = set()


# --------------------------------------------------------------------- data


def build_macro_dataframe() -> tuple[pd.DataFrame, str]:
    """Canonical spot OHLCV with a causal ``stress_z_visible`` column merged on.

    ``stress_z_visible`` comes from ``compute_macro_stress`` (shared,
    unedited) -- already causal (trailing-window z-scores plus a FRED
    publication-lag shift). Computed ONCE here, exactly as R-44's
    ``hashrate_visible`` column was, so every subsequent backtest just
    reads a column rather than re-loading/re-aligning the macro CSVs.
    """
    spot, label = load_dataset(DATA_DIR, "spot")
    stress = compute_macro_stress(spot, DATA_DIR)
    out = spot.copy()
    out["stress_z_visible"] = stress
    return out, label


DF, LABEL = build_macro_dataframe()
print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
      f"(data: {LABEL}); macro-stress coverage {DF['stress_z_visible'].notna().sum():,} bars "
      f"from {DF['stress_z_visible'].dropna().index[0]:%Y-%m-%d}", file=sys.stderr)


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
    """Exactly kelly_regime_v4's own per-anchor latched vote."""
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


def _macro_vote(stress_z: pd.Series, thresh_hi: float, gap: float) -> pd.Series:
    """Latched macro-stress vote, 0/1, hysteresis on ``stress_z``.

    ``macro_vote -> 0`` ("stress", the only direction this vote ever
    fires) requires ``stress_z`` to cross ABOVE ``thresh_hi``.
    ``macro_vote -> 1`` ("calm", the default and the re-arming condition)
    requires it to fall back BELOW ``thresh_lo = thresh_hi - gap``.
    ``gap=0.0`` collapses ``thresh_lo`` onto ``thresh_hi``: a single
    memoryless threshold with no hysteresis dead-zone at all -- kept in
    the swept grid as the explicit negative control this round's brief
    asked for (mirrors R-44's ``capitulation_band=0.0``). Defaults to 1.0
    ("calm") wherever ``stress_z`` is NaN (before the macro series' own
    ~60-day warmup, or if the macro column is entirely absent) -- absence
    of macro information means no veto, not an assumed worst case, so a
    candidate with no macro data at all recovers v4's anchor-only vote
    exactly (the same fallback discipline R-44 used, applied with the
    opposite default value because this vote's neutral state is 1, not
    0)."""
    thresh_lo = thresh_hi - gap
    raw = np.where(stress_z > thresh_hi, 0.0,
                    np.where(stress_z < thresh_lo, 1.0, np.nan))
    vote = pd.Series(raw, index=stress_z.index).ffill().fillna(1.0)
    return vote


class KellyRegimeV14MacroLead(KellyRegimeV3):
    """v4's 3-anchor vote plus a 4th, precision-weighted macro-stress (VIX/DXY) vote.

    Mechanism: ``frac = (anchor_vote_sum + macro_weight * macro_vote) / (3 +
    macro_weight)``, where ``anchor_vote_sum`` is v4's own three 0/1 latched
    price-anchor votes (unchanged) and ``macro_vote`` is the latched
    macro-stress state above (0/1, only ever fires bearish). ``macro_weight``
    is a fixed, swept constant rather than a naive 1.0 (unweighted 4-way
    average), for the same precision/rarity argument R-44 used for its
    hash-ribbon vote. ``macro_weight=0`` recovers v4 exactly (identity
    check, verified in ``causality()``). Everything else -- v3/v4's
    conditional-vol-target scale, the 2x cap, the 10% deadband -- is copied
    verbatim, unchanged.
    """

    name = "kelly_regime_v14_macro_lead"
    warmup = 80 * BARS_PER_DAY + 10  # same as v4; macro's own ~60d z-score warmup is shorter

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 thresh_hi: float = 1.0, gap: float = 0.75, macro_weight: float = 0.33,
                 **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.thresh_hi = thresh_hi
        self.gap = gap
        self.macro_weight = macro_weight

    def _stress_series(self, df: pd.DataFrame) -> pd.Series:
        if "stress_z_visible" in df.columns:
            return df["stress_z_visible"]
        return compute_macro_stress(df, DATA_DIR)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = _anchor_votes(close, self.horizons, self.band)
        anchor_sum = sum(votes.values())  # 0..3 per bar
        n_anchors = float(len(votes))

        stress_z = self._stress_series(df)
        if self.macro_weight > 0:
            macro_vote = _macro_vote(stress_z, self.thresh_hi, self.gap)
        else:
            macro_vote = pd.Series(1.0, index=df.index)

        combined = (anchor_sum + self.macro_weight * macro_vote) / (n_anchors + self.macro_weight)
        frac = combined.to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        # From here down: byte-for-byte v3's conditional vol-targeting sizer,
        # unchanged -- only the vote fraction feeding it differs.
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
        df["v14_frac"] = frac
        df["v14_macro_vote"] = macro_vote.to_numpy()
        df["v14_stress_z"] = stress_z.to_numpy()
        df["v14_anchor_sum"] = anchor_sum.to_numpy()
        return df


class KellyRegimeV14MacroOverride(KellyRegimeV3):
    """Ablation: the SIMPLEST possible version of the same idea.

    No weighted average, no ``macro_weight`` free parameter at all: while
    ``macro_vote`` reads 0 ("stress", latched via the identical
    ``_macro_vote`` hysteresis as the primary candidate), force ``frac`` to
    0 -- a full stand-down, overriding the anchors completely. Otherwise
    ``frac`` is v4's own unmodified 3-anchor average. This is the direct
    test of the R-40/R-46 "elaboration adds nothing over a simpler
    baseline" failure pattern this round's brief pre-registered: if a hard
    veto does just as well (or better) than the precision-weighted average
    above, the averaging machinery is not earning its keep.
    """

    name = "kelly_regime_v14_macro_override"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 thresh_hi: float = 1.0, gap: float = 0.75, **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.thresh_hi = thresh_hi
        self.gap = gap

    def _stress_series(self, df: pd.DataFrame) -> pd.Series:
        if "stress_z_visible" in df.columns:
            return df["stress_z_visible"]
        return compute_macro_stress(df, DATA_DIR)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = _anchor_votes(close, self.horizons, self.band)
        anchor_sum = sum(votes.values())
        n_anchors = float(len(votes))
        anchor_frac = (anchor_sum / n_anchors).to_numpy()

        stress_z = self._stress_series(df)
        macro_vote = _macro_vote(stress_z, self.thresh_hi, self.gap).to_numpy()

        frac = np.where(macro_vote == 0.0, 0.0, anchor_frac)
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

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
        df["v14o_frac"] = frac
        df["v14o_macro_vote"] = macro_vote
        return df


# ------------------------------------------------------------- the grid

# gap=0.0 is the explicit negative control: no hysteresis dead zone at all,
# so the vote collapses to the naive, memoryless "stress_z > thresh_hi"
# reading. 0.75/1.25 require stress to genuinely subside (not merely dip
# under 1.0) before the next stress event counts as new.
GAPS = (0.0, 0.75, 1.25)
# macro_weight=1.0 is the explicit negative control: a literal unweighted
# 4-way average. 0.15/0.33/0.5 are the precision-weighted candidates.
WEIGHTS = (0.15, 0.33, 0.5, 1.0)
THRESH_HI = 1.0  # fixed a-priori (one trailing std of stress), never swept


def _grid():
    out = []
    for gap in GAPS:
        for weight in WEIGHTS:
            label = (f"gap={gap:.2f}"
                      f"{'(naive-nohys)' if gap == 0.0 else ''}"
                      f" w={weight:.2f}{'(unweighted)' if weight == 1.0 else ''}")
            out.append((label, dict(thresh_hi=THRESH_HI, gap=gap, macro_weight=weight)))
    return out


def _config_key(kw: dict) -> str:
    return f"lead|thresh={kw['thresh_hi']:.3f}|gap={kw['gap']:.3f}|w={kw['macro_weight']:.3f}"


def _override_grid():
    out = []
    for gap in GAPS:
        label = f"override gap={gap:.2f}{'(naive-nohys)' if gap == 0.0 else ''}"
        out.append((label, dict(thresh_hi=THRESH_HI, gap=gap)))
    return out


def _override_config_key(kw: dict) -> str:
    return f"override|thresh={kw['thresh_hi']:.3f}|gap={kw['gap']:.3f}"


PRIMARY_KW = dict(thresh_hi=1.0, gap=0.75, macro_weight=0.33)


# ------------------------------------------------------- step 2b: descriptive


def descriptive() -> None:
    """Step 2b -- purely descriptive, no free parameter fitted: how often has
    the latched macro-stress vote actually fired over inner-train+validation,
    and how does that compare to how often each of v4's own three price
    anchors flips over the identical window (the empirical basis for
    treating ``macro_weight`` as a fraction of one full vote)."""
    lo, hi = TRAIN[0], VALID[1]
    frame = DF.loc[lo:hi]
    close = frame["close"]

    print(f"descriptive window (inner-train + inner-validation): {lo} -> {hi}")
    print("\nprice-anchor vote transition counts over the SAME window (context for macro_weight):")
    for days in (20, 40, 80):
        anchor = DF["close"].rolling(int(days * BARS_PER_DAY)).mean().loc[lo:hi]
        v = pd.Series(np.where(close > anchor * 1.01, 1.0,
                                np.where(close < anchor * 0.99, 0.0, np.nan)),
                      index=close.index).ffill().fillna(0.0)
        print(f"  {days:>3d}d price anchor: {int(v.ne(v.shift()).sum()):>4d} vote flips")

    stress = frame["stress_z_visible"]
    print(f"\nstress_z coverage in window: {stress.notna().sum():,} / {len(stress):,} bars "
          f"(NaN before {stress.dropna().index[0]:%Y-%m-%d} if any)")
    print(f"stress_z summary: mean={stress.mean():.2f} std={stress.std():.2f} "
          f"min={stress.min():.2f} max={stress.max():.2f}")

    for gap in GAPS:
        vote = _macro_vote(stress, THRESH_HI, gap)
        flips_to_stress = int(((vote == 0.0) & (vote.shift() == 1.0)).sum())
        flips_to_calm = int(((vote == 1.0) & (vote.shift() == 0.0)).sum())
        label = f"gap={gap:.2f}" + (" (naive, no hysteresis)" if gap == 0.0 else "")
        print(f"  macro vote {label}: {flips_to_stress} stress-onset event(s), "
              f"{flips_to_calm} calm-return event(s)")


# ------------------------------------------------------- failure mode (1): lead time


def _daily_transitions(series: pd.Series, target_value: float, min_gap_days: int = 14) -> list:
    """Daily-resampled transition INTO ``target_value``, deduplicated so
    transitions within ``min_gap_days`` of a prior one (same direction)
    count as one episode's onset, not repeated bar-level noise."""
    daily = series.resample("1D").last().ffill()
    is_target = (daily == target_value)
    # NOTE: ``is_target.shift().fillna(False)`` upcasts to object dtype (the
    # leading NaN forces it), so ``~`` on the filled Python-bool objects does
    # BITWISE invert (``~True == -2``, truthy) rather than logical negation --
    # every day after the first "target" day would wrongly count as a fresh
    # onset. ``shift(fill_value=False)`` keeps the Series boolean-dtype, so
    # ``~`` is logical negation as intended. Caught by hand: the pre-fix
    # version returned 41 macro "onsets" including runs of consecutive
    # calendar days, versus 12 genuine latch transitions confirmed
    # independently by ``descriptive()``'s bar-level flip count.
    prev = is_target.shift(fill_value=False)
    onsets = daily.index[is_target & (~prev)]
    kept = []
    for d in onsets:
        if not kept or (d - kept[-1]).days >= min_gap_days:
            kept.append(d)
    return kept


def leadtime() -> None:
    """Pre-registered failure mode (1): does the macro vote's bear onset
    actually come BEFORE the price-anchor gate's own bear onset in the
    handful of stress episodes available (2018, 2020-03, 2022)? Compares
    flip TIMESTAMPS directly, not aggregate Sharpe -- the round brief's
    explicit instruction. Uses the primary config's threshold/gap only
    (thresh_hi=1.0, gap=0.75); this is a descriptive check, not a fit, and
    is not counted toward N_EVALUATED.
    """
    lo, hi = TRAIN[0], VALID[1]
    frame = DF.loc[lo:hi]
    close = frame["close"]

    votes = _anchor_votes(close, (20, 40, 80), 0.01)
    anchor20_bear = 1.0 - votes[20]           # fastest single anchor, 1=bearish
    anchor_sum = sum(votes.values())
    majority_bear = (anchor_sum < 1.5).astype(float)  # 1 when >=2 of 3 anchors bearish

    stress = frame["stress_z_visible"]
    macro_vote = _macro_vote(stress, PRIMARY_KW["thresh_hi"], PRIMARY_KW["gap"])
    macro_bear = 1.0 - macro_vote

    macro_onsets = _daily_transitions(macro_bear, 1.0)
    anchor20_onsets = _daily_transitions(anchor20_bear, 1.0)
    majority_onsets = _daily_transitions(majority_bear, 1.0)

    print(f"macro bear-onset episodes (gap={PRIMARY_KW['gap']}, thresh_hi={PRIMARY_KW['thresh_hi']}): "
          f"{len(macro_onsets)}")
    print(f"20d-anchor bear-onset episodes: {len(anchor20_onsets)}")
    print(f"3-anchor MAJORITY bear-onset episodes: {len(majority_onsets)}")

    def nearest(target_date, candidates, window_days=180):
        best, best_dist = None, None
        for c in candidates:
            dist = (c - target_date).days
            if abs(dist) <= window_days and (best_dist is None or abs(dist) < abs(best_dist)):
                best, best_dist = c, dist
        return best, best_dist

    print("\nmacro onset vs nearest 20d-anchor onset (positive lead_days = macro flips FIRST):")
    leads_vs_20d = []
    for d in macro_onsets:
        match, dist = nearest(d, anchor20_onsets)
        lead = -dist if dist is not None else None
        if lead is not None:
            leads_vs_20d.append(lead)
        print(f"  macro onset {d.date()}  ->  nearest 20d-anchor onset "
              f"{match.date() if match else 'none within 180d'}  "
              f"lead_days={lead if lead is not None else 'n/a'}")

    print("\nmacro onset vs nearest 3-anchor MAJORITY onset (positive lead_days = macro flips FIRST):")
    leads_vs_majority = []
    for d in macro_onsets:
        match, dist = nearest(d, majority_onsets)
        lead = -dist if dist is not None else None
        if lead is not None:
            leads_vs_majority.append(lead)
        print(f"  macro onset {d.date()}  ->  nearest majority-anchor onset "
              f"{match.date() if match else 'none within 180d'}  "
              f"lead_days={lead if lead is not None else 'n/a'}")

    def summarize(name, leads):
        if not leads:
            print(f"\n{name}: no matched pairs within window -- cannot assess lead/lag")
            return
        n_lead = sum(1 for x in leads if x > 0)
        print(f"\n{name}: {len(leads)} matched episode(s), "
              f"{n_lead}/{len(leads)} macro-leads, "
              f"median lead_days={float(np.median(leads)):.1f}, "
              f"individual leads={[round(x, 1) for x in leads]}")

    summarize("SUMMARY vs 20d anchor (fastest single anchor)", leads_vs_20d)
    summarize("SUMMARY vs 3-anchor majority (the actual gate-flip proxy)", leads_vs_majority)
    print("\nleadtime step: 0 configurations counted toward N_EVALUATED (descriptive, no free parameter fitted)")


# --------------------------------------------------------------- step 3: sweep


def sweep() -> None:
    """Step 3: every (gap, weight) config on inner-train ONLY, spot primary."""
    print(f"\nINNER-TRAIN {TRAIN} / spot -- benchmarks:")
    for name in ("buy_and_hold", INCUMBENT):
        m, _ = measure(get_strategy(name), *TRAIN, market=SPOT)
        line(name, m, "spot")

    print(f"\nINNER-TRAIN {TRAIN} / spot -- candidate configurations:")
    for label, kw in _grid():
        m, _ = measure(KellyRegimeV14MacroLead(**kw), *TRAIN, market=SPOT,
                        config_key=_config_key(kw))
        line(label, m, "spot")

    print(f"\nconfigurations evaluated (step 3, distinct gap/weight pairs): {N_EVALUATED}")


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
            m, _ = measure(KellyRegimeV14MacroLead(**kw), *VALID, market=market,
                            config_key=_config_key(kw))
            cell[mname] = m
            line(label, m, mname)
            rows.append({"label": label, **kw, "market": mname,
                         "final": m.final_balance, "sharpe": m.sharpe,
                         "max_dd": m.max_drawdown_pct, "trades": m.num_trades})
        # selection rule fixed in advance: min(train, valid) spot sharpe,
        # guards against a train-loses/validation-wins overfit signature
        m_train, _ = measure(KellyRegimeV14MacroLead(**kw), *TRAIN, market=SPOT)
        score = min(m_train.sharpe, cell["spot"].sharpe)
        if score > best_score:
            best_label, best_kw, best_score = label, kw, score

    print(f"\nbest by min(train,valid) spot Sharpe: {best_label}  (score={best_score:.2f})")
    print(f"v4 control spot Sharpe: {ctl['spot'].sharpe:.2f}   "
          f"v4 control futures Sharpe: {ctl['futures 5x'].sharpe:.2f}")

    print("\nside-by-side: best candidate's TRAIN window vs VALIDATION window (overfitting-signature check):")
    for mname, market in MARKETS:
        m_train, _ = measure(KellyRegimeV14MacroLead(**best_kw), *TRAIN, market=market)
        m_valid, _ = measure(KellyRegimeV14MacroLead(**best_kw), *VALID, market=market)
        m_train_v4, _ = measure(get_strategy(INCUMBENT), *TRAIN, market=market)
        m_valid_v4, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        print(f"  {mname}:")
        print(f"    candidate  TRAIN final=${m_train.final_balance:>11,.0f} sharpe={m_train.sharpe:.2f} DD={m_train.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid.final_balance:>11,.0f} sharpe={m_valid.sharpe:.2f} DD={m_valid.max_drawdown_pct:.1f}%")
        print(f"    v4         TRAIN final=${m_train_v4.final_balance:>11,.0f} sharpe={m_train_v4.sharpe:.2f} DD={m_train_v4.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid_v4.final_balance:>11,.0f} sharpe={m_valid_v4.sharpe:.2f} DD={m_valid_v4.max_drawdown_pct:.1f}%")
        print(f"    candidate - v4 (valid): Delta sharpe={m_valid.sharpe - m_valid_v4.sharpe:+.3f}  "
              f"Delta DD={m_valid.max_drawdown_pct - m_valid_v4.max_drawdown_pct:+.1f}pp")

    print("\nparameter-neighbourhood plateau check (spot, inner-validation Sharpe, all 12 cells):")
    grid_by_key = {(kw["gap"], kw["macro_weight"]): None for _, kw in _grid()}
    for label, kw in _grid():
        m, _ = measure(KellyRegimeV14MacroLead(**kw), *VALID, market=SPOT)
        grid_by_key[(kw["gap"], kw["macro_weight"])] = m.sharpe
    for gap in GAPS:
        row = "  ".join(f"w={w:.2f}:{grid_by_key[(gap, w)]:.2f}" for w in WEIGHTS)
        print(f"  gap={gap:.2f}  {row}")

    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    return best_label, best_kw


# -------------------------------------------------------- failure mode (4): ablation


def ablation() -> None:
    """Failure mode (4): the R-40/R-46 check. Compare the primary
    precision-weighted-average candidate against the simplest possible
    version of the same idea -- a hard override with no weight parameter
    at all -- on inner-train and inner-validation, both markets. If the
    override does as well or better, the averaging machinery is not
    earning its keep."""
    print("ABLATION: precision-weighted average (primary) vs hard override (simplest baseline)")
    for gap in GAPS:
        lead_kw = dict(thresh_hi=THRESH_HI, gap=gap, macro_weight=PRIMARY_KW["macro_weight"])
        over_kw = dict(thresh_hi=THRESH_HI, gap=gap)
        print(f"\n-- gap={gap:.2f} --")
        for split_name, split in (("TRAIN", TRAIN), ("VALID", VALID)):
            for mname, market in MARKETS:
                m_lead, _ = measure(KellyRegimeV14MacroLead(**lead_kw), *split, market=market,
                                     config_key=_config_key(lead_kw))
                m_over, _ = measure(KellyRegimeV14MacroOverride(**over_kw), *split, market=market,
                                     config_key=_override_config_key(over_kw))
                print(f"  {split_name:5s} {mname:10s}  "
                      f"weighted-avg final=${m_lead.final_balance:>11,.0f} sharpe={m_lead.sharpe:.2f} DD={m_lead.max_drawdown_pct:.1f}%   "
                      f"override final=${m_over.final_balance:>11,.0f} sharpe={m_over.sharpe:.2f} DD={m_over.max_drawdown_pct:.1f}%   "
                      f"Delta(avg-override) sharpe={m_lead.sharpe - m_over.sharpe:+.3f}")

    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


# ------------------------------------------------------------------ artifact


def artifact(kw: dict | None = None) -> None:
    """Mandatory exposure-artifact check: R^2 of the candidate's target
    series against a mean-notional-matched flat rescale of v4's own
    target, on inner-validation, both markets. R^2 > 0.95 -> artifact."""
    kw = kw or PRIMARY_KW
    print(f"exposure-artifact check, candidate={kw}")
    v4 = get_strategy(INCUMBENT)
    for mname, market in MARKETS:
        cand = KellyRegimeV14MacroLead(**kw)
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


def _make_tampered_macro_dir(cut_day: pd.Timestamp, factor: float, tmp_root: Path) -> Path:
    """Copy the three real macro CSVs into a fresh dir, multiplying every
    row dated on/after ``cut_day`` by ``factor``. Used only for the
    causality probe below -- never writes into the real ``data/`` dir."""
    out_dir = tmp_root / f"macro_x{factor:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"spx_daily.csv.gz": "sp500", "vix_daily.csv.gz": "vixcls", "dxy_daily.csv.gz": "dtwexbgs"}
    for fname, col in files.items():
        raw = pd.read_csv(DATA_DIR / fname)
        dates = pd.to_datetime(raw["date"])
        mask = dates >= cut_day.tz_localize(None)
        raw.loc[mask, col] = raw.loc[mask, col] * factor
        raw.to_csv(out_dir / fname, index=False, compression="gzip")
    return out_dir


def causality(kw: dict | None = None) -> None:
    """Two-independent-pathway tamper probe: price OHLCV AND the macro
    stress input tampered independently after a cut. Every decision at or
    before the cut must be unchanged. Restricted to strictly pre-2023
    bars."""
    kw = kw or PRIMARY_KW

    pre2023 = DF[DF.index < OOS_START]
    df = pre2023.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]
    cut_day = df.index[cut].normalize()

    def strategy_for(data_dir: Path | None):
        s = KellyRegimeV14MacroLead(**kw)
        if data_dir is not None:
            orig = s._stress_series

            def patched(frame, _dd=data_dir):
                if "stress_z_visible" in frame.columns:
                    # price-only tamper path: reuse the precomputed (real-macro) column
                    return frame["stress_z_visible"]
                return compute_macro_stress(frame, _dd)
            s._stress_series = patched  # noqa: SLF001 (deliberate, test-only monkeypatch)
        return s

    def run_probe(name, tamper_ohlcv_fn=None, macro_dir_up=None, macro_dir_down=None):
        up, down = df.copy(), df.copy()
        if tamper_ohlcv_fn is not None:
            tamper_ohlcv_fn(up, down)
        if macro_dir_up is not None:
            # drop the precomputed column so `_stress_series` recomputes
            # from the tampered on-disk CSVs via compute_macro_stress
            up = up.drop(columns=["stress_z_visible"])
            down = down.drop(columns=["stress_z_visible"])

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

        a = decisions(up, macro_dir_up)
        b = decisions(down, macro_dir_down)
        bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
        print(f"[{name}] tampered from bar {cut:,} of {len(df):,} "
              f"(calendar day {cut_day.date()}); checked bars {bars}")
        print("  FAIL - reads the future at bars " + str(bad) if bad
              else "  PASS - every decision at or before the cut is unchanged")

        pa = strategy_for(macro_dir_up).prepare(up.copy())
        pb = strategy_for(macro_dir_down).prepare(down.copy())
        for col in ("target", "v14_frac", "v14_macro_vote", "v14_anchor_sum"):
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

    tmp_root = Path(tempfile.mkdtemp(prefix="v14_macro_causality_"))
    try:
        run_probe("PRICE tamper (standard)", tamper_ohlcv_fn=tamper_ohlcv)

        macro_dir_up = _make_tampered_macro_dir(cut_day, 50.0, tmp_root)
        macro_dir_down = _make_tampered_macro_dir(cut_day, 1.0 / 50.0, tmp_root)
        run_probe("MACRO tamper (the new stress_z pathway)",
                   macro_dir_up=macro_dir_up, macro_dir_down=macro_dir_down)

        run_probe("both at once", tamper_ohlcv_fn=tamper_ohlcv,
                   macro_dir_up=macro_dir_up, macro_dir_down=macro_dir_down)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # identity check: macro_weight=0 recovers v4 exactly on a plain slice
    v4 = get_strategy(INCUMBENT)
    ident = KellyRegimeV14MacroLead(thresh_hi=1.0, gap=0.75, macro_weight=0.0)
    frame = df.iloc[-20_000:].copy()
    t_v4 = v4.prepare(frame.copy())["target"].to_numpy()
    t_ident = ident.prepare(frame.copy())["target"].to_numpy()
    worst = float(np.nanmax(np.abs(t_v4 - t_ident)))
    print(f"\nidentity check (macro_weight=0 recovers v4 exactly): "
          f"max|diff|={worst:.3e}  {'PASS' if worst < 1e-9 else 'FAIL'}")


def _fresh_broker():
    from tradebot.broker import PaperBroker
    return PaperBroker(market=FUTURES, start_balance=10_000.0)


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Falsification test (pre-registered rule below, fixed before running).

    ETH-USD Bitfinex spot (2016-03 -> 2019-12-31) overlaps macro data
    coverage (2016-06 ->) almost entirely -- the macro signal is
    market-wide (VIX/DXY), not BTC- or ETH-specific, so it is aligned onto
    the ETH bar grid exactly as it is onto BTC's, via the same
    ``compute_macro_stress`` call. Every candidate config vs v4 control, on
    BTC (control, identical pipeline/window) and ETH (test).

    PRE-REGISTERED RULE, fixed now, before results are read: if the
    candidate is not at least comparable to v4 on ETH, or is visibly worse
    on ETH than on the BTC control through the identical code, this
    direction fails. Since the macro signal is asset-agnostic by
    construction, an ETH-only failure is itself informative and must be
    reported, not hidden.
    """
    eth_spot = load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")
    eth_stress = compute_macro_stress(eth_spot, DATA_DIR)
    eth_df = eth_spot.copy()
    eth_df["stress_z_visible"] = eth_stress

    overlap = eth_df["stress_z_visible"].dropna()
    print(f"ETH spot file: {len(eth_spot):,} bars  {eth_spot.index[0]:%Y-%m-%d} -> "
          f"{eth_spot.index[-1]:%Y-%m-%d}")
    print(f"macro stress coverage overlapping ETH spot: {len(overlap):,} bars "
          f"visible from {overlap.index[0]:%Y-%m-%d} to {overlap.index[-1]:%Y-%m-%d}  "
          f"(NaN in this overlap: {eth_df.loc[overlap.index[0]:overlap.index[-1], 'stress_z_visible'].isna().sum()})")

    frames = {"BTC (control)": DF[DF.index < OOS_START], "ETH (test)": eth_df}
    results = {}
    for asset, frame in frames.items():
        print(f"\n{asset}  {len(frame):,} bars  {frame.index[0]:%Y-%m-%d} -> {frame.index[-1]:%Y-%m-%d}")
        results[asset] = {}
        for mname, market in MARKETS:
            m_v4, _ = measure(get_strategy(INCUMBENT), None, None, df=frame, market=market)
            line(f"{INCUMBENT} (control)", m_v4, mname)
            asset_results = {"v4": m_v4}
            for label, kw in _grid():
                cand = KellyRegimeV14MacroLead(**kw)
                m_c, _ = measure(cand, None, None, df=frame, market=market)
                line(f"v14[{label}]", m_c, mname)
                asset_results[label] = m_c
            results[asset][mname] = asset_results

    print("\nfalsification verdict per config (candidate/v4 final-balance ratio, BTC control vs ETH test):")
    any_fail = False
    for label, kw in _grid():
        for mname, market in MARKETS:
            btc_r = results["BTC (control)"][mname][label].final_balance / results["BTC (control)"][mname]["v4"].final_balance
            eth_r = results["ETH (test)"][mname][label].final_balance / results["ETH (test)"][mname]["v4"].final_balance
            worse_on_eth = eth_r < btc_r - 0.02  # small tolerance for noise
            eth_beats_v4 = eth_r >= 1.0
            flag = "FAIL" if (worse_on_eth and not eth_beats_v4) else ("caution" if worse_on_eth else "ok")
            any_fail = any_fail or flag == "FAIL"
            print(f"  {label:40s} {mname:10s} BTC ratio={btc_r:.3f}x  ETH ratio={eth_r:.3f}x  [{flag}]")
    print(f"\nprimary candidate ({PRIMARY_KW}) checked above; "
          f"overall falsification verdict: {'FAIL' if any_fail else 'no outright FAIL by the pre-registered rule'}")


# --------------------------------------------------------------------- driver


def all_checks() -> None:
    print("=" * 78)
    print("STEP 2b -- descriptive: macro-stress vote frequency vs anchor votes")
    print("=" * 78)
    descriptive()
    print("\n" + "=" * 78)
    print("FAILURE MODE (1) -- lead-time check: does macro flip before the price anchors?")
    print("=" * 78)
    leadtime()
    print("\n" + "=" * 78)
    print("STEP 3 -- sweep (inner-train)")
    print("=" * 78)
    sweep()
    print("\n" + "=" * 78)
    print("STEP 3 -- select (inner-validation, both markets)")
    print("=" * 78)
    select()
    print("\n" + "=" * 78)
    print("FAILURE MODE (4) -- ablation: weighted average vs hard override")
    print("=" * 78)
    ablation()
    print("\n" + "=" * 78)
    print("EXPOSURE-ARTIFACT CHECK")
    print("=" * 78)
    artifact()
    print("\n" + "=" * 78)
    print("CAUSALITY / NO-LOOKAHEAD PROBE")
    print("=" * 78)
    causality()
    print("\n" + "=" * 78)
    print("ETH FALSIFICATION TEST")
    print("=" * 78)
    eth()
    print(f"\ntotal distinct configurations evaluated (N_EVALUATED): {N_EVALUATED}")


if __name__ == "__main__":
    cmds = {"descriptive": descriptive, "leadtime": leadtime, "sweep": sweep, "select": select,
            "ablation": ablation, "artifact": artifact, "causality": causality, "eth": eth,
            "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/kelly_regime_v14_macro_lead.py [{'|'.join(cmds)}]")
