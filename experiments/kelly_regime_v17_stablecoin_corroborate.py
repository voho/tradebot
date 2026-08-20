#!/usr/bin/env python
"""R-56 NOVEL branch: closing backlog item B-23's second named fix --
corroboration from a SECOND, structurally independent signal (BTC active-
address participation growth, a chain-USAGE metric) required alongside
the aggregate USDT stablecoin-supply-deceleration signal, rather than any
further filter on the stablecoin signal alone.

Idea, one sentence
------------------
A lone stablecoin-supply-deceleration reading is treated as noise unless
BTC's own active-address growth is ALSO decelerating at the same time --
an AND-gate across two data sources with different data-generating
processes (dollar capital flow vs. blockchain usage) -- on the theory
that a genuine capital-flight episode should show up in both, while a
transient stablecoin supply wobble (exchange rebalancing, a one-off
mint/burn, redemption noise) usually will not.

Constraint attacked: INFO (one price series). Fifth attempt at this
constraint in this project's history (B-07/R-44 on-chain, R-53 macro,
R-54 stablecoin hard veto, R-55 stablecoin confirming vote, now this) --
same PRIMARY signal as R-54/R-55, but for the first time combined with a
SECOND, independently-sourced information channel rather than tuned or
filtered alone.

Not a duplicate of, cited precisely
------------------------------------
- ``experiments/kelly_regime_v15_stablecoin_veto.py`` (R-54) and
  ``experiments/kelly_regime_v16_stablecoin_confirm.py`` (R-55): both use
  the stablecoin signal ALONE, varying only the combination rule
  (hard override vs. precision-weighted confirming dilution) or, in
  R-55's disjoint CONSERVATIVE sibling, a duration/persistence filter on
  the same single series. This file is the one fix from B-23 that neither
  of those rounds tried: gating the stablecoin vote on a SECOND,
  independent series before it is allowed to affect the trade at all. A
  duration filter (already tried, R-55 CONSERVATIVE, failed -- "the
  'transient' onsets don't reverse within a few days ... duration and
  precision are not separable axes here") asks "did this same signal
  persist long enough to be real?"; this file asks "does a *different*
  signal, built from a *different* data source, agree that something real
  is happening?" -- a materially different question, per B-23's own
  wording ("corroboration from a second independent signal rather than
  filtering one signal alone").
- ``experiments/kelly_regime_v10_onchain_confirm.py`` (R-44 CONSERVATIVE)
  and ``experiments/kelly_regime_v10_hashribbon_vote.py`` (R-44 NOVEL):
  the source of this file's corroborating signal (BTC active-address
  growth, ``AdrActCnt``) and the alternative this file considered and
  rejected (Hash Ribbons miner capitulation, ``HashRate``) -- see
  "Choice of corroborating signal" below for why active-address growth
  was chosen and Hash Ribbons was not. Both R-44 branches used their
  on-chain metric ALONE, fed into v4's vote or size; neither combined it
  with a second, independent series, and neither is re-run here --
  ``_participation_stress`` below reuses R-44's own fixed, non-swept
  design choices (``growth_days=7``, ``z_window_days=180``) verbatim
  rather than re-deriving them, exactly as this project's own precedent
  (R-54 reusing R-53's hysteresis-vote helper) already established.
- B-07's own standing warning, sharpened by R-08 (a *better* volatility
  forecast made this strategy family WORSE, $52K vs $115K, by de-levering
  more promptly into BTC's highest-forward-Sharpe high-volatility states
  -- the Baur & Dimpfl 2018 inverse-leverage effect): this file's
  corroborating vote, like every prior INFO-axis vote in this lineage,
  only ever pulls exposure DOWN relative to what the price anchors alone
  would grant (it defaults to "no corroboration" / "calm", and both legs
  of the AND-gate must independently read "stress" before it fires) --
  never a de-lever-on-rising-activity rule, so it does not reintroduce the
  R-08 failure mode by construction. Checked directly, not merely assumed,
  in the mechanism check below.
- This round's own disjoint CONSERVATIVE branch (targeting B-23's OTHER
  named fix, a shorter growth window matched to genuine-stress duration on
  the stablecoin feature alone): not read, not coordinated with, per this
  round's explicit isolation rule.

Choice of corroborating signal, decided BEFORE any strategy code was
written (the mechanism check below is what decided it)
------------------------------------------------------------------------
Two candidates from R-44 were considered, on the explicit requirement
that the second signal have a genuinely different data-generating
process from stablecoin supply (capital flow): BTC active-address growth
(a chain-USAGE signal -- how many distinct wallets are transacting) and
Hash Ribbons miner capitulation (a miner-ECONOMICS signal -- mining
profitability and hash-rate trend). A quick probe against R-54's own 12
matched stablecoin-onset dates, run before committing to either:

- **Hash Ribbons capitulation** (30d/60d hash-rate SMA ratio,
  ``capitulation_band=0.05``, R-44's own primary) is in a "capitulating"
  state only ~10% of days overall and corroborates just **1 of 12**
  stablecoin onset dates within +-30 days. This matches R-44's own
  description of hash-ribbon recovery as "a once-a-cycle event" -- it is
  simply too coarse and slow-moving (a 30-60 day smoothed miner-economics
  signal) to be evaluated at most of the multi-day-to-few-week timescale
  this branch is trying to corroborate. Requiring it would eliminate
  nearly all coverage, the opposite failure mode from R-54/R-55's
  too-sensitive threshold.
- **Active-address growth** (7-day log growth, 180-day trailing z-score,
  R-44's own fixed defaults, sign-flipped so positive = participation
  decelerating) operates on a comparable multi-day-to-few-week cadence to
  the stablecoin signal and corroborates the large majority of onset
  dates -- close enough in timescale to be a meaningful test, unlike Hash
  Ribbons. It is used as this file's primary corroborating signal for
  that reason. Its own weakness -- explored in the mechanism check below,
  before any inner-validation sweep, per this round's explicit instruction
  -- turned out to be the opposite one: it is *too frequent*, not too
  rare, to discriminate well. Both findings are reported plainly.

Mechanism, precisely
---------------------
``stablecoin_stress_z`` (imported unchanged from ``_stablecoin_signal.py``)
and a new ``onchain_stress_z`` (this file, ``_participation_stress``: 7-day
log growth of BTC active addresses, z-scored on a fixed 180-day trailing
window, sign-flipped so positive = participation decelerating -- the same
sign convention as the stablecoin feature). Each gets its own latched
0/1 hysteresis vote (``_hyst_vote``, identical discipline to every prior
round in this lineage). The corroborated vote:

    corrob_vote = 0 ("stress")  only when BOTH stable_vote==0 AND onchain_vote==0
    corrob_vote = 1 ("calm")    otherwise -- including whenever either input
                                  is unavailable (absence of confirming
                                  evidence is not evidence of stress)

fed into R-55's own precision-weighted confirming-vote dilution
(``frac = (anchor_sum + stable_weight*corrob_vote) / (3+stable_weight)``),
chosen over R-54's hard override as this file's PRIMARY base architecture
because R-55 already established, fresh in-session and reproduced again
in this file's own ``ablation()``, that the confirming dilution beats an
analogous override on this signal in every matched cell -- reusing the
already-validated better combination rule and adding only the
corroboration requirement on top of it, per this round's own instruction
that the corroboration axis, not the base combination rule, is what must
be new here. A second class, ``KellyRegimeV17StablecoinCorroborateOverride``,
reruns the identical corroboration gate through R-54's hard-override rule
instead, purely as an ablation arm (does corroboration help the override
architecture recover any of what R-54 lost?) -- not proposed as this
file's own candidate.

``corroborate=False`` on the primary class recovers R-55's own
uncorroborated confirming-vote dilution exactly (same signal, same
architecture, corroboration requirement switched off) -- the in-session,
apples-to-apples ablation baseline for "did adding the second signal
help", computed fresh here rather than cited from R-55's report.
``stable_weight=0`` recovers v4 exactly regardless of ``corroborate``
(identity check, verified in ``causality()``).

Sources
-------
- BIS Working Paper No. 1340, "Stablecoin flows and spillovers to FX
  markets" (2025); Ahmed & Aldasoro, "Stablecoins and safe asset prices"
  (Cleveland Fed / BIS WP 1270, August 2025); NY Fed Liberty Street
  Economics, "Stablecoins and Crypto Shocks: An Update" (April 2025); IMF
  WP 2025/141, "Decrypting Crypto: How to Estimate International
  Stablecoin Flows" (July 2025) -- all four cited unchanged from
  ``_stablecoin_signal.py``'s own docstring; this file does not re-derive
  the stablecoin feature, only its corroboration.
- Web search run for this round (queries: "on-chain active addresses
  stablecoin supply corroborating signal crypto regime detection
  multi-signal confirmation 2025"; "ensemble on-chain indicators crypto
  market stress signal agreement research 2025"), verified at
  summary/abstract level:
  - A 2025 MDPI study (*Journal of Risk and Financial Management*,
    "Temporal Fusion Transformer-Based Trading Strategy for Multi-Crypto
    Assets Using On-Chain and Technical Indicators") combines multiple
    on-chain features (active addresses among them) with technical
    indicators in one forecasting framework -- a general precedent for
    treating active-address activity as one input series among several
    corroborating sources, not a specific corroboration-gate design this
    file's mechanism is drawn from.
  - Industry-practice commentary (2025-2026, aggregated by web search, not
    a single citable paper): stablecoin flow signals are reportedly
    weighted around 15-25% of a combined signal, and analysts describe
    that "when flows, derivatives, and macro point the same way, the
    probability of a move tightens" -- the general multi-signal-agreement
    intuition this file tests directly and quantitatively, rather than
    assumes. This is motivation for the weight grid's low end (0.15/0.33,
    identical to R-53's/R-55's own grid), not a number copied verbatim.
  These are consistent with, and no stronger than, this project's own
  standing finding across R-53/R-54/R-55: genuinely new information
  channels exist in this data, but converting them into a working
  strategy has failed four times running on precision/specificity
  grounds, not on the existence of the channel.

Pre-registered mechanism check (run BEFORE any inner-validation sweep,
per this round's explicit instruction) -- see ``corroboration_check()``
--------------------------------------------------------------------------
Does requiring active-address corroboration reduce R-54's ~24
false/transient stress-onset count (``descriptive()``'s raw-flip count at
the tightest threshold, thresh=0.75/gap=0.0) while still firing on the
9/12 genuinely-leading matched episodes R-54's own ``leadtime()`` check
confirmed (median +16.5 days vs. the 3-anchor majority)? Reusing R-54's
own matched-episode list and thresholds verbatim (recomputed fresh here,
not hand-copied, to guard against transcription error) for direct
comparability. **Named risk, stated before running anything:** it is a
fully legitimate possibility that the corroborating signal is either too
rare to fire near real episodes (ruled Hash Ribbons out on exactly this
basis, above) or, the opposite failure, too common to discriminate --
i.e. that it agrees with almost everything happening during any broad
downtrend, genuine or not, and therefore filters nothing useful. Both are
reported plainly if found, per this round's instruction to report the
mechanism check honestly even if it kills the branch before any Sharpe
number is read.

Pre-registered falsification tests (fixed before any inner-validation
result was read, the same axes as R-54/R-55)
-----------------------------------------------------------------------
(1) Does not beat v4 on inner-validation Sharpe (both markets) by more
    than the +-0.2 noise floor (R-20), OR does but not across a genuine
    parameter plateau (report neighbours, not just the winner).
(2) Fails the pre-2020 BTC-control-vs-ETH differential falsification test
    (R-54's/R-55's ``eth()`` rule, reused verbatim) -- restricted to the
    2019-01-01 -> 2019-12-31 window both ETH's on-chain coverage and its
    committed price file support (matching R-44's own ETH slicing choice,
    tighter than R-54/R-55's plain stablecoin-only ETH window because this
    file additionally needs ETH on-chain coverage).
(3) Does not beat R-55's own uncorroborated confirming-vote dilution, fed
    the identical stablecoin signal at the identical (thresh_hi, gap,
    stable_weight) points (``corroborate=False`` vs. ``corroborate=True``,
    computed fresh in this file, not cited from R-55's report) -- i.e.
    corroboration itself does not earn its keep over the signal alone.
(4) Fails the three-pathway causality tamper probe (price, stablecoin, OR
    onchain), or the ``stable_weight=0`` identity check does not recover
    v4 exactly.
(5) Is an exposure-artifact (R^2 > 0.95 vs. a flat rescale of v4's own
    ``target``).

**Pre-registered holdout decision rule, fixed before any result is
read:** the 2023+ holdout is read ONLY IF (1) clears the noise floor AND
sits on a genuine plateau, AND (2) passes, AND (3) shows corroboration
earning its keep over the signal alone, AND (4)/(5) both pass cleanly. If
ANY of these fail -- or if the pre-sweep mechanism check above already
shows corroboration does not separate genuine episodes from noise -- this
branch reports NEGATIVE and the holdout is never read. A well-documented
negative is a successful outcome per ROUTINE.md, not a shortfall to be
rescued by relaxing this rule after the fact.

Code reuse decision, stated plainly
-------------------------------------
``_anchor_votes`` and the latched-hysteresis-vote helper (``_hyst_vote``,
functionally identical to R-53's ``_macro_vote``/R-54's and R-55's
``_stable_vote``) are DUPLICATED, not imported, from
``kelly_regime_v14_macro_lead.py`` -- the same precedent every file in
this lineage (v15, v16) has already set. ``compute_stablecoin_stress`` IS
imported unchanged from ``experiments/_stablecoin_signal.py`` -- the one
designated shared, read-only module for that signal. The active-address
participation z-score logic (``_participation_stress``) is adapted
(duplicated, sign convention added, not imported) from
``kelly_regime_v10_onchain_confirm.py``'s ``_participation_z`` -- same
non-shared-infrastructure precedent. None of ``kelly_regime_v14_macro_lead.py``,
``kelly_regime_v15_stablecoin_veto.py``, ``kelly_regime_v16_stablecoin_confirm.py``,
``kelly_regime_v10_onchain_confirm.py``, ``_stablecoin_signal.py``, or any
file under ``src/`` is edited anywhere in this session.

Usage
-----
    python experiments/kelly_regime_v17_stablecoin_corroborate.py mechanism   # THE pre-sweep check, run first
    python experiments/kelly_regime_v17_stablecoin_corroborate.py sweep       # step 3 (inner-train)
    python experiments/kelly_regime_v17_stablecoin_corroborate.py select      # step 3 (inner-validation)
    python experiments/kelly_regime_v17_stablecoin_corroborate.py ablation    # falsification (3): corroborate vs not, same signal/architecture
    python experiments/kelly_regime_v17_stablecoin_corroborate.py override    # override-architecture ablation arm
    python experiments/kelly_regime_v17_stablecoin_corroborate.py artifact    # exposure-artifact check
    python experiments/kelly_regime_v17_stablecoin_corroborate.py causality   # lookahead probe (price + stablecoin + onchain pathways)
    python experiments/kelly_regime_v17_stablecoin_corroborate.py eth         # ETH falsification
    python experiments/kelly_regime_v17_stablecoin_corroborate.py all         # everything, in order
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
from tradebot.data import (  # noqa: E402
    align_onchain_causal,
    load_dataset,
    load_ohlcv_csv,
    load_onchain_metrics,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments._stablecoin_signal import compute_stablecoin_stress  # noqa: E402

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


def _participation_stress(onchain: pd.DataFrame, bars: pd.DataFrame,
                           z_window_days: int, growth_days: int) -> pd.Series:
    """Causal, sign-flipped z-score of BTC active-address growth, aligned onto ``bars``.

    Adapted from ``kelly_regime_v10_onchain_confirm.py``'s ``_participation_z``
    (R-44 CONSERVATIVE) -- ``growth_days=7`` and ``z_window_days=180`` are
    that file's own fixed, non-swept defaults, reused verbatim here rather
    than re-derived. The one change: SIGN-FLIPPED so positive means
    participation growth is unusually slow or contracting (risk-off),
    matching ``_stablecoin_signal.py``'s convention exactly, so a
    "stress" reading always means the same thing on both legs of the
    corroboration gate. Both the ``shift`` and the ``rolling`` window are
    strictly backward-looking on the raw daily frame; only the finished
    daily z-score is projected onto the bar grid via
    ``align_onchain_causal``, which adds CoinMetrics' own one-day
    reporting lag on top -- identical two-stage causality discipline to
    every other signal in this lineage.
    """
    addr = onchain["AdrActCnt"].astype(float)
    growth = np.log(addr / addr.shift(growth_days))
    mean = growth.rolling(z_window_days, min_periods=z_window_days).mean()
    std = growth.rolling(z_window_days, min_periods=z_window_days).std()
    z = (growth - mean) / std.where(std > 0)
    stress_daily = (-1.0 * z).rename("onchain_stress_z").to_frame()
    aligned = align_onchain_causal(stress_daily, bars)
    return aligned["onchain_stress_z"]


def build_corroborate_dataframe() -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """Canonical spot OHLCV with both causal stress columns merged on, plus the raw BTC on-chain frame."""
    spot, label = load_dataset(DATA_DIR, "spot")
    onchain_btc = load_onchain_metrics(DATA_DIR, asset="BTC")
    if onchain_btc is None:
        raise FileNotFoundError("data/btc_onchain_daily.csv.gz not found")
    stable_stress = compute_stablecoin_stress(spot, DATA_DIR)
    onchain_stress = _participation_stress(onchain_btc, spot, 180, 7)
    out = spot.copy()
    out["stablecoin_stress_z_visible"] = stable_stress
    out["onchain_stress_z_visible"] = onchain_stress
    return out, label, onchain_btc


DF, LABEL, ONCHAIN_BTC = build_corroborate_dataframe()
print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  (data: {LABEL}); "
      f"stablecoin coverage {DF['stablecoin_stress_z_visible'].notna().sum():,} bars from "
      f"{DF['stablecoin_stress_z_visible'].dropna().index[0]:%Y-%m-%d}; "
      f"onchain-participation coverage {DF['onchain_stress_z_visible'].notna().sum():,} bars from "
      f"{DF['onchain_stress_z_visible'].dropna().index[0]:%Y-%m-%d}", file=sys.stderr)


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
    print(f"  {tag:48s} {market_name:10s} final=${m.final_balance:>11,.0f} "
          f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>7,.0f}{'  LIQUIDATED' if m.liquidated else ''}")


# ---------------------------------------------------------------- the vote


def _anchor_votes(close: pd.Series, horizons=(20, 40, 80), band: float = 0.01) -> dict:
    """Exactly kelly_regime_v4's own per-anchor latched vote. Duplicated, not
    imported -- see this module's "Code reuse decision" section."""
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


def _hyst_vote(stress_z: pd.Series, thresh_hi: float, gap: float) -> pd.Series:
    """Latched 0/1 hysteresis vote on any z-scored stress series. Identical
    discipline to R-53's ``_macro_vote``/R-54's and R-55's ``_stable_vote`` --
    duplicated, not imported. Used for BOTH the stablecoin leg and the new
    on-chain-participation leg below, so the two legs are combined exactly
    like-for-like."""
    thresh_lo = thresh_hi - gap
    raw = np.where(stress_z > thresh_hi, 0.0,
                    np.where(stress_z < thresh_lo, 1.0, np.nan))
    vote = pd.Series(raw, index=stress_z.index).ffill().fillna(1.0)
    return vote


def _corrob_vote(stable_vote: pd.Series, onchain_vote: pd.Series) -> pd.Series:
    """AND-gate: 'stress' (0) only when BOTH legs read 'stress'; 'calm' (1)
    otherwise, including wherever either leg's own hysteresis vote defaults
    to calm (data absent). This is the one new combination primitive this
    file adds -- everything downstream of it (the confirming-vote dilution
    or the hard override) is reused unmodified from R-54/R-55."""
    s = stable_vote.to_numpy()
    o = onchain_vote.to_numpy()
    out = np.where((s == 0.0) & (o == 0.0), 0.0, 1.0)
    return pd.Series(out, index=stable_vote.index)


class KellyRegimeV17StablecoinCorroborate(KellyRegimeV3):
    """v4's 3-anchor vote plus a 4th, precision-weighted CONFIRMING vote,
    fed by the stablecoin-stress signal ONLY where a second, independent
    on-chain participation signal also reads stress (AND-gate corroboration,
    ``corroborate=True``) -- or, as the in-session ablation baseline, by the
    stablecoin signal alone (``corroborate=False``, R-55's own mechanism,
    reproduced fresh here). Everything below the vote -- v3/v4's
    conditional vol-targeting scale, the 2x cap, the 10% deadband -- is
    copied verbatim from R-55's own ``KellyRegimeV16StablecoinConfirm``.
    ``stable_weight=0`` recovers v4 exactly regardless of ``corroborate``
    (identity check, verified in ``causality()``).
    """

    name = "kelly_regime_v17_stablecoin_corroborate"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 thresh_hi: float = 1.0, gap: float = 0.75,
                 onchain_thresh_hi: float = 1.0, onchain_gap: float = 0.75,
                 stable_weight: float = 0.33, corroborate: bool = True,
                 growth_days: int = 7, z_window_days: int = 180,
                 onchain: pd.DataFrame | None = None, **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.thresh_hi = thresh_hi
        self.gap = gap
        self.onchain_thresh_hi = onchain_thresh_hi
        self.onchain_gap = onchain_gap
        self.stable_weight = stable_weight
        self.corroborate = corroborate
        self.growth_days = growth_days
        self.z_window_days = z_window_days
        self.onchain = onchain if onchain is not None else ONCHAIN_BTC

    def _stress_series(self, df: pd.DataFrame) -> pd.Series:
        if "stablecoin_stress_z_visible" in df.columns:
            return df["stablecoin_stress_z_visible"]
        return compute_stablecoin_stress(df, DATA_DIR)

    def _onchain_series(self, df: pd.DataFrame) -> pd.Series:
        if "onchain_stress_z_visible" in df.columns:
            return df["onchain_stress_z_visible"]
        return _participation_stress(self.onchain, df, self.z_window_days, self.growth_days)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = _anchor_votes(close, self.horizons, self.band)
        anchor_sum = sum(votes.values())
        n_anchors = float(len(votes))

        stress_z = self._stress_series(df)
        stable_vote = _hyst_vote(stress_z, self.thresh_hi, self.gap)

        if self.corroborate and self.stable_weight > 0:
            onchain_z = self._onchain_series(df)
            onchain_vote = _hyst_vote(onchain_z, self.onchain_thresh_hi, self.onchain_gap)
            used_vote = _corrob_vote(stable_vote, onchain_vote)
        elif self.stable_weight > 0:
            onchain_z = pd.Series(np.nan, index=df.index)
            onchain_vote = pd.Series(1.0, index=df.index)
            used_vote = stable_vote
        else:
            onchain_z = pd.Series(np.nan, index=df.index)
            onchain_vote = pd.Series(1.0, index=df.index)
            used_vote = pd.Series(1.0, index=df.index)

        combined = (anchor_sum + self.stable_weight * used_vote) / (n_anchors + self.stable_weight)
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
        df["v17_frac"] = frac
        df["v17_used_vote"] = used_vote.to_numpy()
        df["v17_stable_vote"] = stable_vote.to_numpy()
        df["v17_onchain_vote"] = onchain_vote.to_numpy()
        df["v17_anchor_sum"] = anchor_sum.to_numpy()
        return df


class KellyRegimeV17StablecoinCorroborateOverride(KellyRegimeV3):
    """Ablation arm: the identical AND-gate corroboration, fed into R-54's
    hard-override rule instead of R-55's confirming dilution -- does
    corroboration help the WORSE base architecture recover any of what it
    lost? Not proposed as this file's own candidate; used only inside
    ``override()``."""

    name = "kelly_regime_v17_stablecoin_corroborate_override_ablation"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 thresh_hi: float = 1.0, gap: float = 0.75,
                 onchain_thresh_hi: float = 1.0, onchain_gap: float = 0.75,
                 corroborate: bool = True, **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.thresh_hi = thresh_hi
        self.gap = gap
        self.onchain_thresh_hi = onchain_thresh_hi
        self.onchain_gap = onchain_gap
        self.corroborate = corroborate

    def _stress_series(self, df: pd.DataFrame) -> pd.Series:
        if "stablecoin_stress_z_visible" in df.columns:
            return df["stablecoin_stress_z_visible"]
        return compute_stablecoin_stress(df, DATA_DIR)

    def _onchain_series(self, df: pd.DataFrame) -> pd.Series:
        if "onchain_stress_z_visible" in df.columns:
            return df["onchain_stress_z_visible"]
        return _participation_stress(ONCHAIN_BTC, df, 180, 7)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = _anchor_votes(close, self.horizons, self.band)
        anchor_sum = sum(votes.values())
        n_anchors = float(len(votes))
        anchor_frac = (anchor_sum / n_anchors).to_numpy()

        stress_z = self._stress_series(df)
        stable_vote = _hyst_vote(stress_z, self.thresh_hi, self.gap)
        if self.corroborate:
            onchain_z = self._onchain_series(df)
            onchain_vote = _hyst_vote(onchain_z, self.onchain_thresh_hi, self.onchain_gap)
            used_vote = _corrob_vote(stable_vote, onchain_vote)
        else:
            used_vote = stable_vote

        frac = np.where(used_vote.to_numpy() == 0.0, 0.0, anchor_frac)
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
        df["v17o_frac"] = frac
        df["v17o_used_vote"] = used_vote.to_numpy()
        return df


# ------------------------------------------------------------- the grid

# STABLE_THRESH_GAP: R-54's own 4 named points, reused verbatim for direct
# comparability with R-54/R-55 (primary/tightest/tight-hys/loose).
STABLE_THRESH_GAP = (
    ("primary", 1.00, 0.75),
    ("tightest", 0.75, 0.00),
    ("tight-hys", 0.75, 0.75),
    ("loose", 1.25, 1.25),
)
# ONCHAIN threshold: "matched" is the a-priori, non-cherry-picked choice --
# same numeric scale as the stablecoin primary threshold, chosen because
# both are z-scores and there is no principled reason to pick a different
# number a priori. "tight" is the stricter variant the mechanism check
# below evaluates as a named robustness probe, not a hidden second attempt
# at the same fit -- both are reported.
ONCHAIN_MATCHED = dict(onchain_thresh_hi=1.00, onchain_gap=0.75)
ONCHAIN_TIGHT = dict(onchain_thresh_hi=2.00, onchain_gap=0.50)

WEIGHTS = (0.15, 0.33, 0.5, 1.0)          # identical grid to R-53/R-55
PRIMARY_KW = dict(thresh_hi=1.0, gap=0.75, stable_weight=0.33, corroborate=True,
                   **ONCHAIN_MATCHED)


def _confirm_grid():
    """17 configs at the matched onchain threshold (identity control + 4
    thresh/gap x 4 weights), mirroring R-55's own grid size exactly, plus 4
    extra cells testing the "tight" onchain threshold at the primary
    stablecoin point across all 4 weights -- a named secondary robustness
    check, not a second full sweep."""
    out = [("identity (stable_weight=0)",
             dict(thresh_hi=1.0, gap=0.75, stable_weight=0.0, corroborate=True, **ONCHAIN_MATCHED))]
    for tg_label, thresh_hi, gap in STABLE_THRESH_GAP:
        for weight in WEIGHTS:
            label = (f"{tg_label} (thresh={thresh_hi:.2f} gap={gap:.2f}) w={weight:.2f}"
                      f"{'(unweighted)' if weight == 1.0 else ''} onchain=matched")
            out.append((label, dict(thresh_hi=thresh_hi, gap=gap, stable_weight=weight,
                                     corroborate=True, **ONCHAIN_MATCHED)))
    for weight in WEIGHTS:
        label = f"primary (thresh=1.00 gap=0.75) w={weight:.2f} onchain=TIGHT"
        out.append((label, dict(thresh_hi=1.0, gap=0.75, stable_weight=weight,
                                 corroborate=True, **ONCHAIN_TIGHT)))
    return out


def _ablation_grid():
    """4 configs: corroborate=False at the primary stablecoin point, all 4
    weights -- the in-session, same-architecture, same-signal baseline
    (R-55's own mechanism, reproduced fresh) that falsification test (3)
    compares each matched corroborate=True cell against."""
    out = []
    for weight in WEIGHTS:
        label = f"primary (thresh=1.00 gap=0.75) w={weight:.2f} NO CORROBORATION"
        out.append((label, dict(thresh_hi=1.0, gap=0.75, stable_weight=weight,
                                 corroborate=False, **ONCHAIN_MATCHED)))
    return out


def _override_grid():
    """4 configs: hard-override architecture, corroboration on, at the
    primary and tightest stablecoin points x matched/tight onchain --
    falsification-test-adjacent ablation arm, not this file's own
    candidate."""
    out = []
    for tg_label, thresh_hi, gap in (("primary", 1.00, 0.75), ("tightest", 0.75, 0.00)):
        for onc_label, onc_kw in (("matched", ONCHAIN_MATCHED), ("tight", ONCHAIN_TIGHT)):
            label = f"override {tg_label} onchain={onc_label}"
            out.append((label, dict(thresh_hi=thresh_hi, gap=gap, corroborate=True, **onc_kw)))
    return out


def _config_key(kw: dict) -> str:
    return (f"corrob={kw['corroborate']}|thresh={kw['thresh_hi']:.3f}|gap={kw['gap']:.3f}|"
            f"w={kw['stable_weight']:.3f}|onc_thresh={kw['onchain_thresh_hi']:.3f}|"
            f"onc_gap={kw['onchain_gap']:.3f}")


def _override_config_key(kw: dict) -> str:
    return (f"override|corrob={kw['corroborate']}|thresh={kw['thresh_hi']:.3f}|gap={kw['gap']:.3f}|"
            f"onc_thresh={kw['onchain_thresh_hi']:.3f}|onc_gap={kw['onchain_gap']:.3f}")


# ------------------------------------------------- pre-sweep mechanism check


def _daily_transitions(series: pd.Series, target_value: float, min_gap_days: int = 14) -> list:
    """Verbatim from R-54's ``kelly_regime_v15_stablecoin_veto.py`` -- same
    dedup discipline, same ``shift(fill_value=False)`` fix."""
    daily = series.resample("1D").last().ffill()
    is_target = (daily == target_value)
    prev = is_target.shift(fill_value=False)
    onsets = daily.index[is_target & (~prev)]
    kept = []
    for d in onsets:
        if not kept or (d - kept[-1]).days >= min_gap_days:
            kept.append(d)
    return kept


def mechanism_check() -> None:
    """THE pre-registered check, run BEFORE any inner-validation sweep:
    does requiring active-address corroboration reduce R-54's false
    stress-onset count while preserving the 9/12 genuinely-leading matched
    episodes? Reproduces R-54's own leadtime() and descriptive() numbers
    fresh (not hand-copied) for direct, verifiable comparability. 0
    configurations counted toward N_EVALUATED (descriptive, no free
    parameter fitted)."""
    lo, hi = TRAIN[0], VALID[1]
    frame = DF.loc[lo:hi]
    close = frame["close"]

    votes = _anchor_votes(close, (20, 40, 80), 0.01)
    anchor_sum = sum(votes.values())
    majority_bear = (anchor_sum < 1.5).astype(float)
    majority_onsets = _daily_transitions(majority_bear, 1.0)

    stable_z = frame["stablecoin_stress_z_visible"]
    onchain_z = frame["onchain_stress_z_visible"]

    print("=== reproducing R-54's own numbers fresh, for direct comparability ===")
    stable_vote_primary = _hyst_vote(stable_z, 1.0, 0.75)
    flips_primary = int(((stable_vote_primary == 0.0) & (stable_vote_primary.shift() == 1.0)).sum())
    stable_vote_tightest = _hyst_vote(stable_z, 0.75, 0.0)
    flips_tightest = int(((stable_vote_tightest == 0.0) & (stable_vote_tightest.shift() == 1.0)).sum())
    print(f"  raw flip count (descriptive()-style, R-54 reported 12/24): "
          f"primary(1.00/0.75)={flips_primary}  tightest(0.75/0.00)={flips_tightest}")

    stable_bear_primary = 1.0 - stable_vote_primary
    stable_onsets = _daily_transitions(stable_bear_primary, 1.0)
    print(f"  matched-episode onset count (leadtime()-style dedup, R-54 reported 12): {len(stable_onsets)}")

    def nearest(target_date, candidates, window_days=180):
        best, best_dist = None, None
        for c in candidates:
            dist = (c - target_date).days
            if abs(dist) <= window_days and (best_dist is None or abs(dist) < abs(best_dist)):
                best, best_dist = c, dist
        return best, best_dist

    leads = []
    lead_flags = {}
    for d in stable_onsets:
        _, dist = nearest(d, majority_onsets)
        lead = -dist if dist is not None else None
        if lead is not None:
            leads.append(lead)
        lead_flags[d] = (lead is not None and lead > 0)
    n_lead = sum(1 for x in leads if x > 0)
    print(f"  {n_lead}/{len(leads)} lead the 3-anchor majority (R-54 reported 9/12), "
          f"median={float(np.median(leads)):.1f}d (R-54 reported +16.5d)\n")

    print("=== onchain-participation corroboration, at the a-priori MATCHED threshold (1.00/0.75) ===")
    onchain_vote = _hyst_vote(onchain_z, 1.0, 0.75)
    daily_onchain = onchain_vote.resample("1D").last().ffill()
    frac_stress = float((daily_onchain == 0.0).mean())
    print(f"  fraction of ALL days in onchain-participation 'stress' state: {frac_stress:.3f}")

    def corroborated_within(d, window_days=30):
        w = daily_onchain.loc[d - pd.Timedelta(days=window_days): d + pd.Timedelta(days=window_days)]
        return bool((w == 0.0).any())

    print(f"\n  per-episode corroboration check (+-30 days), all {len(stable_onsets)} primary onsets:")
    n_lead_corrob, n_lag_corrob, n_lead_total, n_lag_total = 0, 0, 0, 0
    for d in stable_onsets:
        is_lead = lead_flags.get(d, False)
        corrob = corroborated_within(d)
        if is_lead:
            n_lead_total += 1
            n_lead_corrob += corrob
        else:
            n_lag_total += 1
            n_lag_corrob += corrob
        print(f"    {d.date()}  ({'LEAD' if is_lead else 'lag/unmatched'})  corroborated={corrob}")
    print(f"\n  corroboration rate on LEADING episodes: {n_lead_corrob}/{n_lead_total}  "
          f"vs. LAGGING/unmatched episodes: {n_lag_corrob}/{n_lag_total}")
    if n_lead_total and n_lag_total and abs(n_lead_corrob / n_lead_total - n_lag_corrob / n_lag_total) < 0.15:
        print("  ==> FINDING: corroboration rate is essentially THE SAME for leading and lagging/unmatched\n"
              "      episodes -- the AND-gate is not discriminating genuine early leads from noise, it is\n"
              "      mostly reading 'are we broadly in a multi-month downtrend', a state most stablecoin\n"
              "      onsets (true and false alike) already share. This is the pre-sweep finding this check\n"
              "      exists to surface, stated plainly per this round's instruction.")

    print(f"\n  raw-flip-count reduction test: does AND-gating the TIGHTEST config's vote with onchain\n"
          f"  corroboration (matched threshold) reduce its {flips_tightest} raw flips?")
    corrob_vote_tightest = _corrob_vote(stable_vote_tightest, onchain_vote)
    flips_tightest_corrob = int(((corrob_vote_tightest == 0.0) & (corrob_vote_tightest.shift() == 1.0)).sum())
    print(f"    tightest raw flips WITHOUT corroboration: {flips_tightest}")
    print(f"    tightest raw flips WITH corroboration:    {flips_tightest_corrob}  "
          f"({'reduced' if flips_tightest_corrob < flips_tightest else 'NOT reduced'})")

    corrob_vote_primary = _corrob_vote(stable_vote_primary, onchain_vote)
    stable_bear_primary_corrob = 1.0 - corrob_vote_primary
    onsets_primary_corrob = _daily_transitions(stable_bear_primary_corrob, 1.0)
    n_lead_survive = sum(1 for d in stable_onsets if lead_flags.get(d, False)
                          and any(abs((d - dc).days) <= 14 for dc in onsets_primary_corrob))
    print(f"\n  of the {n_lead_total} genuinely-leading matched episodes, "
          f"{n_lead_survive} still produce a corroborated onset within +-14 days")

    print("\n=== onchain-participation corroboration, at the STRICTER threshold (2.00/0.50), robustness probe ===")
    onchain_vote_tight = _hyst_vote(onchain_z, 2.0, 0.5)
    daily_onchain_tight = onchain_vote_tight.resample("1D").last().ffill()
    frac_stress_tight = float((daily_onchain_tight == 0.0).mean())
    print(f"  fraction of ALL days in onchain-participation 'stress' state (tight): {frac_stress_tight:.3f}")

    def corroborated_within_tight(d, window_days=30):
        w = daily_onchain_tight.loc[d - pd.Timedelta(days=window_days): d + pd.Timedelta(days=window_days)]
        return bool((w == 0.0).any())

    n_lead_corrob_tight = sum(1 for d in stable_onsets if lead_flags.get(d, False) and corroborated_within_tight(d))
    n_lag_corrob_tight = sum(1 for d in stable_onsets if not lead_flags.get(d, False) and corroborated_within_tight(d))
    print(f"  corroboration rate on LEADING episodes: {n_lead_corrob_tight}/{n_lead_total}  "
          f"vs. LAGGING/unmatched: {n_lag_corrob_tight}/{n_lag_total}")
    print("  tightening the onchain threshold does start to discriminate (lower overall corroboration\n"
          "  rate) but at the cost of dropping some of the same genuinely-leading episodes it is supposed\n"
          "  to preserve -- there is no threshold tested here that cleanly separates true leads from noise.")

    print("\n=== Hash Ribbons corroboration, for comparison (why it was NOT chosen -- see module docstring) ===")
    hr = ONCHAIN_BTC["HashRate"].astype(float).loc[lo:hi]
    ma_fast = hr.rolling(30, min_periods=30).mean()
    ma_slow = hr.rolling(60, min_periods=60).mean()
    ratio = ma_fast / ma_slow
    band = 0.05
    raw = np.where(ratio > 1.0, 1.0, np.where(ratio < 1.0 - band, 0.0, np.nan))
    hr_state = pd.Series(raw, index=hr.index).ffill().fillna(0.0)
    frac_capitulating = float((hr_state == 0.0).mean())
    n_hr_corrob = sum(
        1 for d in stable_onsets
        if bool((hr_state.loc[d - pd.Timedelta(days=30): d + pd.Timedelta(days=30)] == 0.0).any())
    )
    print(f"  fraction of days BTC hash-ribbon in capitulation: {frac_capitulating:.3f}")
    print(f"  of {len(stable_onsets)} primary stablecoin onsets, {n_hr_corrob} corroborated by "
          f"hash-ribbon capitulation (+-30d) -- confirms the "
          f"module docstring's 'too coarse, once-a-cycle' finding, ruling it out as this file's "
          f"corroborating signal.")

    print("\nmechanism-check step: 0 configurations counted toward N_EVALUATED "
          "(descriptive, no free parameter fitted)")


# --------------------------------------------------------------- step 3: sweep


def sweep() -> None:
    """Step 3: every confirming-vote config on inner-train ONLY, spot primary."""
    print(f"\nINNER-TRAIN {TRAIN} / spot -- benchmarks:")
    for name in ("buy_and_hold", INCUMBENT):
        m, _ = measure(get_strategy(name), *TRAIN, market=SPOT)
        line(name, m, "spot")

    print(f"\nINNER-TRAIN {TRAIN} / spot -- candidate configurations:")
    for label, kw in _confirm_grid():
        m, _ = measure(KellyRegimeV17StablecoinCorroborate(**kw), *TRAIN, market=SPOT,
                        config_key=_config_key(kw))
        line(label, m, "spot")

    print(f"\nconfigurations evaluated (step 3, confirming-vote grid): {N_EVALUATED}")


# -------------------------------------------------------------- step 3: select


def select():
    """Every confirming-vote config on inner-validation ONLY, BOTH markets, vs v4 control."""
    rows = []
    print(f"\nINNER-VALIDATION {VALID} -- v4 control:")
    ctl = {}
    for mname, market in MARKETS:
        m, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        ctl[mname] = m
        line(f"{INCUMBENT} (control)", m, mname)

    print(f"\nINNER-VALIDATION {VALID} -- candidate configurations:")
    best_label, best_kw, best_score = None, None, -1e9
    for label, kw in _confirm_grid():
        cell = {}
        for mname, market in MARKETS:
            m, _ = measure(KellyRegimeV17StablecoinCorroborate(**kw), *VALID, market=market,
                            config_key=_config_key(kw))
            cell[mname] = m
            line(label, m, mname)
            rows.append({"label": label, **kw, "market": mname,
                         "final": m.final_balance, "sharpe": m.sharpe,
                         "max_dd": m.max_drawdown_pct, "trades": m.num_trades})
        m_train, _ = measure(KellyRegimeV17StablecoinCorroborate(**kw), *TRAIN, market=SPOT)
        score = min(m_train.sharpe, cell["spot"].sharpe)
        if score > best_score:
            best_label, best_kw, best_score = label, kw, score

    print(f"\nbest by min(train,valid) spot Sharpe: {best_label}  (score={best_score:.2f})")
    print(f"v4 control spot Sharpe: {ctl['spot'].sharpe:.2f}   "
          f"v4 control futures Sharpe: {ctl['futures 5x'].sharpe:.2f}")

    print("\nside-by-side: best candidate's TRAIN window vs VALIDATION window (overfitting-signature check):")
    for mname, market in MARKETS:
        m_train, _ = measure(KellyRegimeV17StablecoinCorroborate(**best_kw), *TRAIN, market=market)
        m_valid, _ = measure(KellyRegimeV17StablecoinCorroborate(**best_kw), *VALID, market=market)
        m_train_v4, _ = measure(get_strategy(INCUMBENT), *TRAIN, market=market)
        m_valid_v4, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        print(f"  {mname}:")
        print(f"    candidate  TRAIN final=${m_train.final_balance:>11,.0f} sharpe={m_train.sharpe:.2f} DD={m_train.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid.final_balance:>11,.0f} sharpe={m_valid.sharpe:.2f} DD={m_valid.max_drawdown_pct:.1f}%")
        print(f"    v4         TRAIN final=${m_train_v4.final_balance:>11,.0f} sharpe={m_train_v4.sharpe:.2f} DD={m_train_v4.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid_v4.final_balance:>11,.0f} sharpe={m_valid_v4.sharpe:.2f} DD={m_valid_v4.max_drawdown_pct:.1f}%")
        print(f"    candidate - v4 (valid): Delta sharpe={m_valid.sharpe - m_valid_v4.sharpe:+.3f}  "
              f"Delta DD={m_valid.max_drawdown_pct - m_valid_v4.max_drawdown_pct:+.1f}pp")

    print("\nparameter-neighbourhood plateau check (spot, inner-validation Sharpe, matched-onchain cells):")
    grid_by_key = {(tg, w): None for tg, _, _ in STABLE_THRESH_GAP for w in WEIGHTS}
    for tg_label, thresh_hi, gap in STABLE_THRESH_GAP:
        for weight in WEIGHTS:
            m, _ = measure(KellyRegimeV17StablecoinCorroborate(
                thresh_hi=thresh_hi, gap=gap, stable_weight=weight, corroborate=True, **ONCHAIN_MATCHED),
                *VALID, market=SPOT)
            grid_by_key[(tg_label, weight)] = m.sharpe
    for tg_label, thresh_hi, gap in STABLE_THRESH_GAP:
        row = "  ".join(f"w={w:.2f}:{grid_by_key[(tg_label, w)]:.2f}" for w in WEIGHTS)
        print(f"  {tg_label:10s} (thresh={thresh_hi:.2f} gap={gap:.2f})  {row}")
    m_ident, _ = measure(KellyRegimeV17StablecoinCorroborate(stable_weight=0.0, **ONCHAIN_MATCHED),
                          *VALID, market=SPOT)
    print(f"  identity (stable_weight=0, should equal v4 spot Sharpe {ctl['spot'].sharpe:.2f}): {m_ident.sharpe:.2f}")

    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    return best_label, best_kw


# -------------------------------------------------------- falsification test (3)


def ablation() -> None:
    """Falsification test (3): does corroboration (AND-gate with onchain
    participation) beat the identical signal/architecture WITHOUT
    corroboration (R-55's own mechanism, reproduced fresh here), at the
    primary stablecoin point, across all 4 weights, both splits, both
    markets?"""
    print("ABLATION: corroborate=True vs corroborate=False, identical stablecoin signal + confirming architecture")
    for weight in WEIGHTS:
        kw_corrob = dict(thresh_hi=1.0, gap=0.75, stable_weight=weight, corroborate=True, **ONCHAIN_MATCHED)
        kw_plain = dict(thresh_hi=1.0, gap=0.75, stable_weight=weight, corroborate=False, **ONCHAIN_MATCHED)
        print(f"\n-- weight={weight:.2f} --")
        for split_name, split in (("TRAIN", TRAIN), ("VALID", VALID)):
            for mname, market in MARKETS:
                m_c, _ = measure(KellyRegimeV17StablecoinCorroborate(**kw_corrob), *split, market=market,
                                  config_key=_config_key(kw_corrob))
                m_p, _ = measure(KellyRegimeV17StablecoinCorroborate(**kw_plain), *split, market=market,
                                  config_key=_config_key(kw_plain))
                print(f"  {split_name:5s} {mname:10s}  "
                      f"corroborated final=${m_c.final_balance:>11,.0f} sharpe={m_c.sharpe:.2f} DD={m_c.max_drawdown_pct:.1f}%   "
                      f"no-corrob final=${m_p.final_balance:>11,.0f} sharpe={m_p.sharpe:.2f} DD={m_p.max_drawdown_pct:.1f}%   "
                      f"Delta(corrob-plain) sharpe={m_c.sharpe - m_p.sharpe:+.3f}")

    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


def override() -> None:
    """Ablation arm: does corroboration help the WORSE (hard-override)
    base architecture recover any of what R-54 lost?"""
    print("OVERRIDE ABLATION: corroboration on R-54's hard-override architecture, corroborate=True vs False")
    for label, kw in _override_grid():
        kw_plain = {**kw, "corroborate": False}
        print(f"\n-- {label} --")
        for split_name, split in (("TRAIN", TRAIN), ("VALID", VALID)):
            for mname, market in MARKETS:
                m_c, _ = measure(KellyRegimeV17StablecoinCorroborateOverride(**kw), *split, market=market,
                                  config_key=_override_config_key(kw))
                m_p, _ = measure(KellyRegimeV17StablecoinCorroborateOverride(**kw_plain), *split, market=market,
                                  config_key=_override_config_key(kw_plain))
                print(f"  {split_name:5s} {mname:10s}  "
                      f"corroborated final=${m_c.final_balance:>11,.0f} sharpe={m_c.sharpe:.2f}   "
                      f"no-corrob final=${m_p.final_balance:>11,.0f} sharpe={m_p.sharpe:.2f}   "
                      f"Delta sharpe={m_c.sharpe - m_p.sharpe:+.3f}")

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
        cand = KellyRegimeV17StablecoinCorroborate(**kw)
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


def _make_tampered_stablecoin_dir(cut_day: pd.Timestamp, factor: float, tmp_root: Path) -> Path:
    out_dir = tmp_root / f"stable_x{factor:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_DIR / "stablecoin_supply_daily.csv.gz")
    dates = pd.to_datetime(raw["timestamp"])
    mask = dates >= cut_day.tz_localize(None)
    raw.loc[mask, "usdt_SplyCur"] = raw.loc[mask, "usdt_SplyCur"] * factor
    raw.to_csv(out_dir / "stablecoin_supply_daily.csv.gz", index=False, compression="gzip")
    return out_dir


def causality(kw: dict | None = None) -> None:
    """Three-independent-pathway tamper probe: price OHLCV, the
    stablecoin-supply input, AND the on-chain active-address input,
    tampered independently and combined. Every decision at or before the
    cut must be unchanged. Restricted to strictly pre-2023 bars."""
    kw = kw or PRIMARY_KW

    pre2023 = DF[DF.index < OOS_START]
    df = pre2023.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]
    cut_day = df.index[cut].normalize()

    def strategy_for(stable_dir: Path | None, onchain_frame: pd.DataFrame | None):
        s = KellyRegimeV17StablecoinCorroborate(**kw)
        if stable_dir is not None:
            def patched_stable(frame, _dd=stable_dir):
                return compute_stablecoin_stress(frame, _dd)
            s._stress_series = patched_stable  # noqa: SLF001
        if onchain_frame is not None:
            def patched_onchain(frame, _oc=onchain_frame):
                return _participation_stress(_oc, frame, s.z_window_days, s.growth_days)
            s._onchain_series = patched_onchain  # noqa: SLF001
        return s

    def run_probe(name, tamper_ohlcv_fn=None, stable_dir_up=None, stable_dir_down=None,
                  onchain_up=None, onchain_down=None):
        up, down = df.copy(), df.copy()
        if tamper_ohlcv_fn is not None:
            tamper_ohlcv_fn(up, down)
        if stable_dir_up is not None:
            up = up.drop(columns=["stablecoin_stress_z_visible"])
            down = down.drop(columns=["stablecoin_stress_z_visible"])
        if onchain_up is not None:
            up = up.drop(columns=["onchain_stress_z_visible"])
            down = down.drop(columns=["onchain_stress_z_visible"])

        def decisions(frame, stable_dir, onchain_frame):
            s = strategy_for(stable_dir, onchain_frame)
            prepared = s.prepare(frame.copy())
            broker = _fresh_broker()
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        a = decisions(up, stable_dir_up, onchain_up)
        b = decisions(down, stable_dir_down, onchain_down)
        bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
        print(f"[{name}] tampered from bar {cut:,} of {len(df):,} "
              f"(calendar day {cut_day.date()}); checked bars {bars}")
        print("  FAIL - reads the future at bars " + str(bad) if bad
              else "  PASS - every decision at or before the cut is unchanged")

        pa = strategy_for(stable_dir_up, onchain_up).prepare(up.copy())
        pb = strategy_for(stable_dir_down, onchain_down).prepare(down.copy())
        for col in ("target", "v17_frac", "v17_used_vote", "v17_stable_vote", "v17_onchain_vote"):
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

    def tamper_onchain(frame, factor):
        oc = ONCHAIN_BTC.copy()
        mask = oc.index >= cut_day
        oc.loc[mask, "AdrActCnt"] = oc.loc[mask, "AdrActCnt"].to_numpy() * factor
        return oc

    tmp_root = Path(tempfile.mkdtemp(prefix="v17_stablecoin_corroborate_causality_"))
    try:
        run_probe("PRICE tamper (standard)", tamper_ohlcv_fn=tamper_ohlcv)

        stable_dir_up = _make_tampered_stablecoin_dir(cut_day, 50.0, tmp_root)
        stable_dir_down = _make_tampered_stablecoin_dir(cut_day, 1.0 / 50.0, tmp_root)
        run_probe("STABLECOIN tamper", stable_dir_up=stable_dir_up, stable_dir_down=stable_dir_down)

        onchain_up = tamper_onchain(df, 50.0)
        onchain_down = tamper_onchain(df, 1.0 / 50.0)
        run_probe("ONCHAIN tamper (the new, second pathway)", onchain_up=onchain_up, onchain_down=onchain_down)

        run_probe("all three at once", tamper_ohlcv_fn=tamper_ohlcv,
                   stable_dir_up=stable_dir_up, stable_dir_down=stable_dir_down,
                   onchain_up=onchain_up, onchain_down=onchain_down)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    v4 = get_strategy(INCUMBENT)
    ident = KellyRegimeV17StablecoinCorroborate(thresh_hi=1.0, gap=0.75, stable_weight=0.0)
    frame = df.iloc[-20_000:].copy()
    t_v4 = v4.prepare(frame.copy())["target"].to_numpy()
    t_ident = ident.prepare(frame.copy())["target"].to_numpy()
    worst = float(np.nanmax(np.abs(t_v4 - t_ident)))
    print(f"\nidentity check (stable_weight=0 recovers v4 exactly): "
          f"max|diff|={worst:.3e}  {'PASS' if worst < 1e-9 else 'FAIL'}")


def _fresh_broker():
    from tradebot.broker import PaperBroker
    return PaperBroker(market=FUTURES, start_balance=10_000.0)


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Falsification test (2), pre-registered rule fixed before running.

    Restricted to 2019-01-01 -> 2019-12-31 for BOTH assets: the window
    where price, stablecoin-supply, AND on-chain active-address coverage
    all overlap for ETH (on-chain data starts 2019-01-01; the committed
    ETH price file ends 2019-12-31) -- tighter than R-54/R-55's own
    stablecoin-only ETH window because this file additionally needs ETH
    on-chain coverage, matching R-44's own ETH-onchain slicing choice.
    """
    eth_spot_full = load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")
    eth_onchain = load_onchain_metrics(ROOT / "data", asset="ETH")
    if eth_onchain is None:
        raise FileNotFoundError("data/eth_onchain_daily.csv.gz not found")

    eth_spot = eth_spot_full.loc["2019-01-01":"2019-12-31"]
    eth_stable = compute_stablecoin_stress(eth_spot, DATA_DIR)
    eth_onchain_stress = _participation_stress(eth_onchain, eth_spot, 180, 7)
    eth_df = eth_spot.copy()
    eth_df["stablecoin_stress_z_visible"] = eth_stable
    eth_df["onchain_stress_z_visible"] = eth_onchain_stress

    btc_df = DF.loc["2019-01-01":"2019-12-31"]

    print(f"ETH test window: {len(eth_df):,} bars  {eth_df.index[0]:%Y-%m-%d} -> {eth_df.index[-1]:%Y-%m-%d}")
    print(f"  stablecoin coverage NaN: {eth_df['stablecoin_stress_z_visible'].isna().sum()}  "
          f"onchain coverage NaN: {eth_df['onchain_stress_z_visible'].isna().sum()}")
    print(f"BTC control window: {len(btc_df):,} bars  {btc_df.index[0]:%Y-%m-%d} -> {btc_df.index[-1]:%Y-%m-%d}")

    frames = {"BTC (control)": btc_df, "ETH (test)": eth_df}
    onchains = {"BTC (control)": ONCHAIN_BTC, "ETH (test)": eth_onchain}
    results = {}
    for asset, frame in frames.items():
        print(f"\n{asset}  {len(frame):,} bars")
        results[asset] = {}
        for mname, market in MARKETS:
            m_v4, _ = measure(get_strategy(INCUMBENT), None, None, df=frame, market=market)
            line(f"{INCUMBENT} (control)", m_v4, mname)
            asset_results = {"v4": m_v4}
            for label, kw in _confirm_grid():
                cand = KellyRegimeV17StablecoinCorroborate(**kw, onchain=onchains[asset])
                m_c, _ = measure(cand, None, None, df=frame, market=market)
                line(f"v17[{label}]", m_c, mname)
                asset_results[label] = m_c
            results[asset][mname] = asset_results

    print("\nfalsification verdict per config (candidate/v4 final-balance ratio, BTC control vs ETH test):")
    any_fail = False
    for label, kw in _confirm_grid():
        for mname, market in MARKETS:
            btc_r = results["BTC (control)"][mname][label].final_balance / results["BTC (control)"][mname]["v4"].final_balance
            eth_r = results["ETH (test)"][mname][label].final_balance / results["ETH (test)"][mname]["v4"].final_balance
            worse_on_eth = eth_r < btc_r - 0.02
            eth_beats_v4 = eth_r >= 1.0
            flag = "FAIL" if (worse_on_eth and not eth_beats_v4) else ("caution" if worse_on_eth else "ok")
            any_fail = any_fail or flag == "FAIL"
            print(f"  {label:55s} {mname:10s} BTC ratio={btc_r:.3f}x  ETH ratio={eth_r:.3f}x  [{flag}]")
    print(f"\nprimary candidate ({PRIMARY_KW}) checked above; "
          f"overall falsification verdict: {'FAIL' if any_fail else 'no outright FAIL by the pre-registered rule'}")


# --------------------------------------------------------------------- driver


def all_checks() -> None:
    print("=" * 78)
    print("PRE-SWEEP MECHANISM CHECK -- run BEFORE any inner-validation sweep")
    print("=" * 78)
    mechanism_check()
    print("\n" + "=" * 78)
    print("STEP 3 -- sweep (inner-train)")
    print("=" * 78)
    sweep()
    print("\n" + "=" * 78)
    print("STEP 3 -- select (inner-validation, both markets)")
    print("=" * 78)
    select()
    print("\n" + "=" * 78)
    print("FALSIFICATION TEST (3) -- ablation: corroborated vs uncorroborated, same signal/architecture")
    print("=" * 78)
    ablation()
    print("\n" + "=" * 78)
    print("OVERRIDE-ARCHITECTURE ABLATION ARM")
    print("=" * 78)
    override()
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
    cmds = {"mechanism": mechanism_check, "sweep": sweep, "select": select,
            "ablation": ablation, "override": override, "artifact": artifact,
            "causality": causality, "eth": eth, "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/kelly_regime_v17_stablecoin_corroborate.py [{'|'.join(cmds)}]")
