#!/usr/bin/env python
"""R-54 NOVEL branch: aggregate USDT stablecoin-supply DECELERATION as a
hard, unweighted veto on kelly_regime_v4's own price-anchor regime gate.

Idea, one sentence
------------------
Stablecoin issuance is the on-ramp for new dollar capital entering crypto
trading and redemption is the off-ramp, so a sharp deceleration or
outright contraction in aggregate USDT supply (``stablecoin_stress_z``,
imported unchanged from the shared, operator-authored -- to THIS branch
only -- ``experiments/_stablecoin_signal.py``) plausibly reflects capital
already leaving the system before that shows up as price weakness, in a
way an external, indirectly-correlated index (VIX/DXY, R-53) structurally
cannot be: it is mechanically tied to capital actually moving in or out
of the crypto trading system, not an indirect spillover from equity/FX
markets.

Constraint attacked: INFO (one price series). Aggregate stablecoin supply
is a genuinely new, price-independent information channel -- the THIRD
attempt at INFO this project has made, after on-chain activity metrics
(B-07, R-44, active addresses/hash rate -- a supply-side/usage signal
about BTC's own network) and external macro (R-53, VIX/DXY -- indices
that describe the rest of the financial system). This one is neither of
those: it is crypto-native (unlike R-53) but is about capital FLOW/
LIQUIDITY moving through stablecoins, not chain ACTIVITY (unlike B-07).

Not a duplicate of, cited precisely
------------------------------------
- B-07/R-44's on-chain branches (``kelly_regime_v10_hashribbon_vote.py``
  and its sibling): BTC's own active-address/hash-rate activity -- a
  supply-side/usage signal about the chain BTC itself trades on, not a
  liquidity-flow signal about dollar capital entering/leaving the system.
  Stablecoin supply says nothing about how many addresses are active or
  how much hashpower is mining; it says how much fresh dollar liquidity
  the trading system currently holds.
- R-53's macro branches (``kelly_regime_v14_macro_brake.py``,
  ``kelly_regime_v14_macro_lead.py``): VIX/DXY are external market
  indices, correlated with crypto only indirectly through risk-appetite
  spillover -- R-53's own lead-time check found this indirection costs
  real time (median lag -5.5 days against the 3-anchor majority).
  Stablecoin supply is crypto-native: it is literally the dollar balance
  sitting inside the trading system, not a proxy observed from outside it.
- L-12/``harsanyi_crowd``: a price-DERIVED crowding signal (a Bayesian
  posterior fit to BTC's own return history). Stablecoin supply is an
  independent data channel entirely -- it does not touch BTC's OHLCV at
  all, causally or otherwise, until the shared alignment step merges it
  onto the bar grid.
- This round's own disjoint CONSERVATIVE branch
  (``experiments/kelly_regime_v15_macro_veto.py``): NOT read, not
  coordinated with, per this round's explicit isolation rule.
- **B-21** (this project's own backlog item, filed by R-53): the hard,
  unweighted macro veto that beat v4 outright on inner-validation but was
  never pre-registered or lead-time-tested. That item is VIX/DXY-fed and
  is this round's CONSERVATIVE branch's job, not this one's -- this file
  reuses B-21's ARCHITECTURE (a hard override: force ``frac=0`` while a
  latched vote reads "stress", v4's own unmodified 3-anchor average
  otherwise) deliberately, so that if this branch's outcome differs from
  R-53's, the difference is attributable to the SIGNAL (stablecoin supply
  vs VIX/DXY), not to the combination rule.

Mechanism, precisely -- the exact feature formula (pre-registered in
``experiments/_stablecoin_signal.py``, read that file's module docstring
for the full derivation and literature; summarized here)
------------------------------------------------------------------------
``growth_14d = log(supply_t) - log(supply_{t-14})`` (14-calendar-day log
growth of aggregate USDT circulating supply, USDT alone -- see the signal
module's docstring for why USDC is fetched-and-verified-reachable but
deliberately not blended in). ``stablecoin_stress_z = -1 * zscore(growth_14d,
trailing 365d, min_periods=60)`` -- positive means growth is unusually
slow or supply is contracting (risk-off), matching R-53's ``stress_z``
sign convention. BOTH windows (14-day growth, 365-day z-score) are FIXED
A-PRIORI and never swept anywhere in this file, the identical discipline
``_macro_signal.py`` used for VIX/DXY.

A latched veto vote, ``stable_vote in {0, 1}``:

    stable_vote -> 0 ("stress")   when stablecoin_stress_z crosses ABOVE thresh_hi
    stable_vote -> 1 ("calm")     when stablecoin_stress_z crosses BELOW thresh_lo
    stable_vote unchanged (latched) while thresh_lo <= stablecoin_stress_z <= thresh_hi
    stable_vote defaults to 1 ("calm") before the first crossing, or
        wherever stablecoin data is unavailable (data absent -> no veto,
        falls back to v4's own anchor-only vote exactly)

``thresh_lo = thresh_hi - gap``. Both ``thresh_hi`` and ``gap`` ARE swept
(this is the veto-sensitivity axis of the ARCHITECTURE, not the signal
formula -- same status as R-53's ``gap``/``macro_weight`` grid): the
combined vote is the SAME hard-override architecture as R-53's
``KellyRegimeV14MacroOverride`` --

    frac = 0.0                       while stable_vote == 0 ("stress")
    frac = anchor_sum / 3            otherwise (v4's own unmodified vote)

-- so ``thresh_hi`` and ``gap`` control only WHEN the veto fires, never
what it does once fired. This file does NOT re-test the precision-
weighted-averaging architecture R-53's novel branch already rejected
(losing to its own hard-override ablation in 10/12 matched cells) --
using the hard override as the PRIMARY candidate here, per this round's
explicit instruction, is the direct, honest continuation of that finding
rather than a re-derivation of the version R-53 already ruled out.

Code reuse decision, stated plainly (per this round's instruction to
document the choice): the anchor-vote and latched-hysteresis-vote helper
functions below are DUPLICATED from ``kelly_regime_v14_macro_lead.py``
(same ~15 lines each) rather than imported from it. That file is a
private, unregistered experiment from a PRIOR round, not shared
infrastructure -- importing from it would create an undocumented
coupling to a file this branch has no authority to keep stable, and
R-53's own two branches already established the norm of NOT sharing code
across branches except through an explicitly-designated shared module
(there, ``_macro_signal.py``; here, ``_stablecoin_signal.py``).
``kelly_regime_v14_macro_lead.py`` itself is not edited anywhere in this
session.

Pre-registered falsification test, centerpiece of this round (named
before any code ran)
---------------------------------------------------------------------
**Does the stablecoin-stress veto's flip actually LEAD
``kelly_regime_v4``'s own 3-anchor MAJORITY price-gate flip** in the
stress episodes available in inner-train+inner-validation (2018, 2020-03
BTC crash, and the 2022 bear more broadly), compared by flip TIMESTAMPS
-- not aggregate Sharpe -- exactly as R-53's ``leadtime()`` did. This is
the SAME test that killed R-53's averaged-vote candidate (median offset
-5.5 days, i.e. net LAG, against the majority-anchor flip) and it is run
here first, before any Sharpe number is read, because R-53 already
showed a signal can look promising on other axes while failing the one
that determines whether the mechanism can work at all. **Named risk,
stated before running anything:** it is a fully legitimate possibility
that this signal ALSO lags rather than leads -- daily on-chain supply
data feeding a 5-minute-bar strategy has a coarser native cadence than
what it is being asked to lead, and if the lead-time check finds a lag,
that is reported as a real negative result, exactly as honestly as R-53
did, not explained away.

**UST/Terra collapse (May 2022) scope decision, stated before running
anything:** IN SCOPE as a secondary, labeled check within the lead-time
window, not as a primary matched episode alongside 2018/2020-03/the 2022
bear broadly. Reasoning: UST/LUNA was an ALGORITHMIC stablecoin, and this
branch's signal is aggregate USDT supply -- a mechanically different
instrument. But the May 2022 panic produced real, measurable stress on
USDT itself (a brief secondary de-peg and elevated redemption pressure
as capital fled the wider stablecoin complex), so it is a genuine test of
whether USDT supply reacted with useful lead time to a stablecoin-
specific liquidity event, reported and flagged with this caveat rather
than silently pooled with the price-driven episodes.

Other checks run, per this round's brief
------------------------------------------
ETH falsification, pre-2020 BTC control (via the same TRAIN+VALID window
the lead-time check uses), causality tamper probe on BOTH the price
pathway and the new stablecoin-data pathway independently, exposure-
artifact R² check (>0.95 vs a flat rescale of v4 = instant reject),
parameter-neighbourhood plateau check, and an identity check that
disabling the veto (``enabled=False``) recovers v4 exactly.

Usage
-----
    python experiments/kelly_regime_v15_stablecoin_veto.py descriptive
    python experiments/kelly_regime_v15_stablecoin_veto.py leadtime     # THE centerpiece check
    python experiments/kelly_regime_v15_stablecoin_veto.py sweep        # step 3 (inner-train)
    python experiments/kelly_regime_v15_stablecoin_veto.py select       # step 3 (inner-validation)
    python experiments/kelly_regime_v15_stablecoin_veto.py artifact     # exposure-artifact check
    python experiments/kelly_regime_v15_stablecoin_veto.py causality    # lookahead probe (price + stablecoin pathway)
    python experiments/kelly_regime_v15_stablecoin_veto.py eth          # ETH falsification
    python experiments/kelly_regime_v15_stablecoin_veto.py all          # everything, in order
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


def build_stablecoin_dataframe() -> tuple[pd.DataFrame, str]:
    """Canonical spot OHLCV with a causal ``stablecoin_stress_z_visible`` column merged on."""
    spot, label = load_dataset(DATA_DIR, "spot")
    stress = compute_stablecoin_stress(spot, DATA_DIR)
    out = spot.copy()
    out["stablecoin_stress_z_visible"] = stress
    return out, label


DF, LABEL = build_stablecoin_dataframe()
print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
      f"(data: {LABEL}); stablecoin-stress coverage {DF['stablecoin_stress_z_visible'].notna().sum():,} bars "
      f"from {DF['stablecoin_stress_z_visible'].dropna().index[0]:%Y-%m-%d}", file=sys.stderr)


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
    """Exactly kelly_regime_v4's own per-anchor latched vote.
    Duplicated (not imported) from kelly_regime_v14_macro_lead.py -- see
    this module's docstring, "Code reuse decision", for why."""
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


def _stable_vote(stress_z: pd.Series, thresh_hi: float, gap: float) -> pd.Series:
    """Latched stablecoin-stress vote, 0/1, hysteresis on ``stress_z``.
    Same hysteresis discipline as R-53's ``_macro_vote`` -- duplicated,
    not imported, per this module's "Code reuse decision"."""
    thresh_lo = thresh_hi - gap
    raw = np.where(stress_z > thresh_hi, 0.0,
                    np.where(stress_z < thresh_lo, 1.0, np.nan))
    vote = pd.Series(raw, index=stress_z.index).ffill().fillna(1.0)
    return vote


class KellyRegimeV15StablecoinVeto(KellyRegimeV3):
    """v4's 3-anchor vote with a hard veto: force ``frac=0`` while the
    latched stablecoin-supply-deceleration vote reads "stress"; v4's own
    unmodified 3-anchor average otherwise. Identical combination rule to
    R-53's ``KellyRegimeV14MacroOverride`` (B-21's architecture), fed by
    ``stablecoin_stress_z`` instead of VIX/DXY ``stress_z``. ``enabled=False``
    recovers v4 exactly (identity check, verified in ``causality()``).
    """

    name = "kelly_regime_v15_stablecoin_veto"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 thresh_hi: float = 1.0, gap: float = 0.75, enabled: bool = True,
                 **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.thresh_hi = thresh_hi
        self.gap = gap
        self.enabled = enabled

    def _stress_series(self, df: pd.DataFrame) -> pd.Series:
        if "stablecoin_stress_z_visible" in df.columns:
            return df["stablecoin_stress_z_visible"]
        return compute_stablecoin_stress(df, DATA_DIR)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = _anchor_votes(close, self.horizons, self.band)
        anchor_sum = sum(votes.values())
        n_anchors = float(len(votes))
        anchor_frac = (anchor_sum / n_anchors).to_numpy()

        stress_z = self._stress_series(df)
        if self.enabled:
            stable_vote = _stable_vote(stress_z, self.thresh_hi, self.gap).to_numpy()
        else:
            stable_vote = np.ones(len(df))  # identity check: never vetoes

        frac = np.where(stable_vote == 0.0, 0.0, anchor_frac)
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
        df["v15_frac"] = frac
        df["v15_stable_vote"] = stable_vote
        df["v15_stress_z"] = stress_z.to_numpy()
        df["v15_anchor_sum"] = anchor_sum.to_numpy()
        return df


# ------------------------------------------------------------- the grid

# thresh_hi is swept here (unlike R-53, where it was fixed a-priori)
# because this file's ARCHITECTURE is fixed to the hard override, so
# thresh_hi/gap together are the veto-sensitivity axis, not the signal
# formula -- the signal formula itself (14d growth, 365d z-score) is what
# stays fixed a-priori per this file's pre-registration.
THRESH_HIS = (0.75, 1.0, 1.25)
GAPS = (0.0, 0.75, 1.25)
PRIMARY_KW = dict(thresh_hi=1.0, gap=0.75)


def _grid():
    out = []
    for thresh_hi in THRESH_HIS:
        for gap in GAPS:
            label = f"thresh={thresh_hi:.2f} gap={gap:.2f}{'(naive-nohys)' if gap == 0.0 else ''}"
            out.append((label, dict(thresh_hi=thresh_hi, gap=gap)))
    return out


def _config_key(kw: dict) -> str:
    return f"veto|thresh={kw['thresh_hi']:.3f}|gap={kw['gap']:.3f}"


# ------------------------------------------------------- step 2b: descriptive


def descriptive() -> None:
    """Step 2b -- purely descriptive, no free parameter fitted: how often
    has the latched stablecoin-stress vote actually fired over inner-train
    + inner-validation, and how does that compare to how often each of
    v4's own three price anchors flips over the identical window."""
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

    stress = frame["stablecoin_stress_z_visible"]
    print(f"\nstablecoin_stress_z coverage in window: {stress.notna().sum():,} / {len(stress):,} bars "
          f"(NaN before {stress.dropna().index[0]:%Y-%m-%d} if any)")
    print(f"stablecoin_stress_z summary: mean={stress.mean():.2f} std={stress.std():.2f} "
          f"min={stress.min():.2f} max={stress.max():.2f}")

    for thresh_hi in THRESH_HIS:
        for gap in GAPS:
            vote = _stable_vote(stress, thresh_hi, gap)
            flips_to_stress = int(((vote == 0.0) & (vote.shift() == 1.0)).sum())
            flips_to_calm = int(((vote == 1.0) & (vote.shift() == 0.0)).sum())
            print(f"  thresh={thresh_hi:.2f} gap={gap:.2f}: {flips_to_stress} stress-onset event(s), "
                  f"{flips_to_calm} calm-return event(s)")


# ------------------------------------------------------- failure mode: lead time


def _daily_transitions(series: pd.Series, target_value: float, min_gap_days: int = 14) -> list:
    """Daily-resampled transition INTO ``target_value``, deduplicated so
    transitions within ``min_gap_days`` of a prior one count as one
    episode's onset. Uses ``shift(fill_value=False)`` (NOT
    ``.shift().fillna(False)``, which R-53 found silently upcasts to
    object dtype and makes ``~`` do bitwise, not logical, negation on
    Python bool objects) -- fixed here from the start per the round's own
    explicit instruction, not rediscovered."""
    daily = series.resample("1D").last().ffill()
    is_target = (daily == target_value)
    prev = is_target.shift(fill_value=False)
    onsets = daily.index[is_target & (~prev)]
    kept = []
    for d in onsets:
        if not kept or (d - kept[-1]).days >= min_gap_days:
            kept.append(d)
    return kept


def leadtime() -> None:
    """THE CENTERPIECE CHECK, pre-registered: does the stablecoin-stress
    veto's bear onset come BEFORE the price-anchor gate's own bear onset
    in the handful of stress episodes available (2018, 2020-03, 2022)?
    Compares flip TIMESTAMPS directly, not aggregate Sharpe. Uses the
    primary config only (thresh_hi=1.0, gap=0.75); descriptive, not a fit,
    not counted toward N_EVALUATED.
    """
    lo, hi = TRAIN[0], VALID[1]
    frame = DF.loc[lo:hi]
    close = frame["close"]

    votes = _anchor_votes(close, (20, 40, 80), 0.01)
    anchor20_bear = 1.0 - votes[20]
    anchor_sum = sum(votes.values())
    majority_bear = (anchor_sum < 1.5).astype(float)

    stress = frame["stablecoin_stress_z_visible"]
    stable_vote = _stable_vote(stress, PRIMARY_KW["thresh_hi"], PRIMARY_KW["gap"])
    stable_bear = 1.0 - stable_vote

    stable_onsets = _daily_transitions(stable_bear, 1.0)
    anchor20_onsets = _daily_transitions(anchor20_bear, 1.0)
    majority_onsets = _daily_transitions(majority_bear, 1.0)

    print(f"stablecoin bear-onset episodes (thresh_hi={PRIMARY_KW['thresh_hi']}, gap={PRIMARY_KW['gap']}): "
          f"{len(stable_onsets)}")
    print(f"20d-anchor bear-onset episodes: {len(anchor20_onsets)}")
    print(f"3-anchor MAJORITY bear-onset episodes: {len(majority_onsets)}")
    print(f"all stablecoin onset dates: {[d.date().isoformat() for d in stable_onsets]}")

    def nearest(target_date, candidates, window_days=180):
        best, best_dist = None, None
        for c in candidates:
            dist = (c - target_date).days
            if abs(dist) <= window_days and (best_dist is None or abs(dist) < abs(best_dist)):
                best, best_dist = c, dist
        return best, best_dist

    print("\nstablecoin onset vs nearest 20d-anchor onset (positive lead_days = stablecoin flips FIRST):")
    leads_vs_20d = []
    for d in stable_onsets:
        match, dist = nearest(d, anchor20_onsets)
        lead = -dist if dist is not None else None
        if lead is not None:
            leads_vs_20d.append(lead)
        print(f"  stablecoin onset {d.date()}  ->  nearest 20d-anchor onset "
              f"{match.date() if match else 'none within 180d'}  "
              f"lead_days={lead if lead is not None else 'n/a'}")

    print("\nstablecoin onset vs nearest 3-anchor MAJORITY onset (positive lead_days = stablecoin flips FIRST):")
    leads_vs_majority = []
    for d in stable_onsets:
        match, dist = nearest(d, majority_onsets)
        lead = -dist if dist is not None else None
        if lead is not None:
            leads_vs_majority.append(lead)
        print(f"  stablecoin onset {d.date()}  ->  nearest majority-anchor onset "
              f"{match.date() if match else 'none within 180d'}  "
              f"lead_days={lead if lead is not None else 'n/a'}")

    def summarize(name, leads):
        if not leads:
            print(f"\n{name}: no matched pairs within window -- cannot assess lead/lag")
            return
        n_lead = sum(1 for x in leads if x > 0)
        print(f"\n{name}: {len(leads)} matched episode(s), "
              f"{n_lead}/{len(leads)} stablecoin-leads, "
              f"median lead_days={float(np.median(leads)):.1f}, "
              f"individual leads={[round(x, 1) for x in leads]}")

    summarize("SUMMARY vs 20d anchor (fastest single anchor)", leads_vs_20d)
    summarize("SUMMARY vs 3-anchor majority (the actual gate-flip proxy)", leads_vs_majority)

    print("\n--- UST/Terra collapse (May 2022), labeled secondary check per pre-registration ---")
    print("(algorithmic-stablecoin event, not USDT/USDC directly -- IN SCOPE as a labeled")
    print(" secondary check of whether USDT supply itself reacted with lead time, not pooled")
    print(" with the primary matched-episode set above)")
    ust_window = frame.loc["2022-04-15":"2022-06-15"]
    ust_stress = ust_window["stablecoin_stress_z_visible"]
    ust_close = ust_window["close"]
    print(f"  stablecoin_stress_z over the window: min={ust_stress.min():.2f} "
          f"max={ust_stress.max():.2f} on {ust_stress.idxmax().date() if ust_stress.notna().any() else 'n/a'}")
    ust_vote_flips = stable_bear.loc["2022-04-15":"2022-06-15"]
    ust_stress_onset = ust_vote_flips[ust_vote_flips.diff().fillna(0) > 0]
    print(f"  stable_vote flips to bear inside this window: "
          f"{[t.date().isoformat() for t in ust_stress_onset.index[:5]]}"
          f"{' ...' if len(ust_stress_onset) > 5 else ''} ({len(ust_stress_onset)} bar-level flips)")
    price_low_date = ust_close.idxmin()
    print(f"  BTC price low inside this window: {price_low_date.date()} (${ust_close.min():,.0f})")

    print("\nleadtime step: 0 configurations counted toward N_EVALUATED (descriptive, no free parameter fitted)")


# --------------------------------------------------------------- step 3: sweep


def sweep() -> None:
    """Step 3: every (thresh_hi, gap) config on inner-train ONLY, spot primary."""
    print(f"\nINNER-TRAIN {TRAIN} / spot -- benchmarks:")
    for name in ("buy_and_hold", INCUMBENT):
        m, _ = measure(get_strategy(name), *TRAIN, market=SPOT)
        line(name, m, "spot")

    print(f"\nINNER-TRAIN {TRAIN} / spot -- candidate configurations:")
    for label, kw in _grid():
        m, _ = measure(KellyRegimeV15StablecoinVeto(**kw), *TRAIN, market=SPOT,
                        config_key=_config_key(kw))
        line(label, m, "spot")

    print(f"\nconfigurations evaluated (step 3, distinct thresh/gap pairs): {N_EVALUATED}")


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
            m, _ = measure(KellyRegimeV15StablecoinVeto(**kw), *VALID, market=market,
                            config_key=_config_key(kw))
            cell[mname] = m
            line(label, m, mname)
            rows.append({"label": label, **kw, "market": mname,
                         "final": m.final_balance, "sharpe": m.sharpe,
                         "max_dd": m.max_drawdown_pct, "trades": m.num_trades})
        # selection rule fixed in advance: min(train, valid) spot sharpe,
        # guards against a train-loses/validation-wins overfit signature
        m_train, _ = measure(KellyRegimeV15StablecoinVeto(**kw), *TRAIN, market=SPOT)
        score = min(m_train.sharpe, cell["spot"].sharpe)
        if score > best_score:
            best_label, best_kw, best_score = label, kw, score

    print(f"\nbest by min(train,valid) spot Sharpe: {best_label}  (score={best_score:.2f})")
    print(f"v4 control spot Sharpe: {ctl['spot'].sharpe:.2f}   "
          f"v4 control futures Sharpe: {ctl['futures 5x'].sharpe:.2f}")

    print("\nside-by-side: best candidate's TRAIN window vs VALIDATION window (overfitting-signature check):")
    for mname, market in MARKETS:
        m_train, _ = measure(KellyRegimeV15StablecoinVeto(**best_kw), *TRAIN, market=market)
        m_valid, _ = measure(KellyRegimeV15StablecoinVeto(**best_kw), *VALID, market=market)
        m_train_v4, _ = measure(get_strategy(INCUMBENT), *TRAIN, market=market)
        m_valid_v4, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        print(f"  {mname}:")
        print(f"    candidate  TRAIN final=${m_train.final_balance:>11,.0f} sharpe={m_train.sharpe:.2f} DD={m_train.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid.final_balance:>11,.0f} sharpe={m_valid.sharpe:.2f} DD={m_valid.max_drawdown_pct:.1f}%")
        print(f"    v4         TRAIN final=${m_train_v4.final_balance:>11,.0f} sharpe={m_train_v4.sharpe:.2f} DD={m_train_v4.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid_v4.final_balance:>11,.0f} sharpe={m_valid_v4.sharpe:.2f} DD={m_valid_v4.max_drawdown_pct:.1f}%")
        print(f"    candidate - v4 (valid): Delta sharpe={m_valid.sharpe - m_valid_v4.sharpe:+.3f}  "
              f"Delta DD={m_valid.max_drawdown_pct - m_valid_v4.max_drawdown_pct:+.1f}pp")

    print("\nparameter-neighbourhood plateau check (spot, inner-validation Sharpe, all 9 cells):")
    grid_by_key = {(kw["thresh_hi"], kw["gap"]): None for _, kw in _grid()}
    for label, kw in _grid():
        m, _ = measure(KellyRegimeV15StablecoinVeto(**kw), *VALID, market=SPOT)
        grid_by_key[(kw["thresh_hi"], kw["gap"])] = m.sharpe
    for thresh_hi in THRESH_HIS:
        row = "  ".join(f"gap={g:.2f}:{grid_by_key[(thresh_hi, g)]:.2f}" for g in GAPS)
        print(f"  thresh={thresh_hi:.2f}  {row}")

    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    return best_label, best_kw


# ------------------------------------------------------------------ artifact


def artifact(kw: dict | None = None) -> None:
    """Mandatory exposure-artifact check: R^2 of the candidate's target
    series against a mean-notional-matched flat rescale of v4's own
    target, on inner-validation, both markets. R^2 > 0.95 -> artifact."""
    kw = kw or PRIMARY_KW
    print(f"exposure-artifact check, candidate={kw}")
    v4 = get_strategy(INCUMBENT)
    for mname, market in MARKETS:
        cand = KellyRegimeV15StablecoinVeto(**kw)
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
    """Copy the real stablecoin-supply CSV into a fresh dir, multiplying
    every row dated on/after ``cut_day`` by ``factor``. Used only for the
    causality probe below -- never writes into the real ``data/`` dir."""
    out_dir = tmp_root / f"stable_x{factor:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_DIR / "stablecoin_supply_daily.csv.gz")
    dates = pd.to_datetime(raw["timestamp"])
    mask = dates >= cut_day.tz_localize(None)
    raw.loc[mask, "usdt_SplyCur"] = raw.loc[mask, "usdt_SplyCur"] * factor
    raw.to_csv(out_dir / "stablecoin_supply_daily.csv.gz", index=False, compression="gzip")
    return out_dir


def causality(kw: dict | None = None) -> None:
    """Two-independent-pathway tamper probe: price OHLCV AND the
    stablecoin-supply input tampered independently after a cut. Every
    decision at or before the cut must be unchanged. Restricted to
    strictly pre-2023 bars."""
    kw = kw or PRIMARY_KW

    pre2023 = DF[DF.index < OOS_START]
    df = pre2023.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]
    cut_day = df.index[cut].normalize()

    def strategy_for(data_dir: Path | None):
        s = KellyRegimeV15StablecoinVeto(**kw)
        if data_dir is not None:
            def patched(frame, _dd=data_dir):
                if "stablecoin_stress_z_visible" in frame.columns:
                    # price-only tamper path: reuse the precomputed (real-stablecoin) column
                    return frame["stablecoin_stress_z_visible"]
                return compute_stablecoin_stress(frame, _dd)
            s._stress_series = patched  # noqa: SLF001 (deliberate, test-only monkeypatch)
        return s

    def run_probe(name, tamper_ohlcv_fn=None, stable_dir_up=None, stable_dir_down=None):
        up, down = df.copy(), df.copy()
        if tamper_ohlcv_fn is not None:
            tamper_ohlcv_fn(up, down)
        if stable_dir_up is not None:
            up = up.drop(columns=["stablecoin_stress_z_visible"])
            down = down.drop(columns=["stablecoin_stress_z_visible"])

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

        a = decisions(up, stable_dir_up)
        b = decisions(down, stable_dir_down)
        bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
        print(f"[{name}] tampered from bar {cut:,} of {len(df):,} "
              f"(calendar day {cut_day.date()}); checked bars {bars}")
        print("  FAIL - reads the future at bars " + str(bad) if bad
              else "  PASS - every decision at or before the cut is unchanged")

        pa = strategy_for(stable_dir_up).prepare(up.copy())
        pb = strategy_for(stable_dir_down).prepare(down.copy())
        for col in ("target", "v15_frac", "v15_stable_vote", "v15_anchor_sum"):
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

    tmp_root = Path(tempfile.mkdtemp(prefix="v15_stablecoin_causality_"))
    try:
        run_probe("PRICE tamper (standard)", tamper_ohlcv_fn=tamper_ohlcv)

        stable_dir_up = _make_tampered_stablecoin_dir(cut_day, 50.0, tmp_root)
        stable_dir_down = _make_tampered_stablecoin_dir(cut_day, 1.0 / 50.0, tmp_root)
        run_probe("STABLECOIN tamper (the new supply-data pathway)",
                   stable_dir_up=stable_dir_up, stable_dir_down=stable_dir_down)

        run_probe("both at once", tamper_ohlcv_fn=tamper_ohlcv,
                   stable_dir_up=stable_dir_up, stable_dir_down=stable_dir_down)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # identity check: enabled=False recovers v4 exactly on a plain slice
    v4 = get_strategy(INCUMBENT)
    ident = KellyRegimeV15StablecoinVeto(thresh_hi=1.0, gap=0.75, enabled=False)
    frame = df.iloc[-20_000:].copy()
    t_v4 = v4.prepare(frame.copy())["target"].to_numpy()
    t_ident = ident.prepare(frame.copy())["target"].to_numpy()
    worst = float(np.nanmax(np.abs(t_v4 - t_ident)))
    print(f"\nidentity check (enabled=False recovers v4 exactly): "
          f"max|diff|={worst:.3e}  {'PASS' if worst < 1e-9 else 'FAIL'}")


def _fresh_broker():
    from tradebot.broker import PaperBroker
    return PaperBroker(market=FUTURES, start_balance=10_000.0)


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Falsification test (pre-registered rule below, fixed before running).

    ETH-USD Bitfinex spot (2016-03 -> 2019-12-31) against USDT-supply
    coverage (2017-01-01 ->). Stablecoin supply is asset-agnostic by
    construction (it is not derived from either BTC's or ETH's own price)
    -- the pre-registered rule mirrors R-53's: if the candidate is not at
    least comparable to v4 on ETH, or is visibly worse on ETH than on the
    BTC control through the identical code, this direction fails. An
    ETH-only failure must be reported, not hidden.
    """
    eth_spot = load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")
    eth_stress = compute_stablecoin_stress(eth_spot, DATA_DIR)
    eth_df = eth_spot.copy()
    eth_df["stablecoin_stress_z_visible"] = eth_stress

    overlap = eth_df["stablecoin_stress_z_visible"].dropna()
    print(f"ETH spot file: {len(eth_spot):,} bars  {eth_spot.index[0]:%Y-%m-%d} -> "
          f"{eth_spot.index[-1]:%Y-%m-%d}")
    print(f"stablecoin stress coverage overlapping ETH spot: {len(overlap):,} bars "
          f"visible from {overlap.index[0]:%Y-%m-%d} to {overlap.index[-1]:%Y-%m-%d}  "
          f"(NaN in this overlap: {eth_df.loc[overlap.index[0]:overlap.index[-1], 'stablecoin_stress_z_visible'].isna().sum()})")

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
                cand = KellyRegimeV15StablecoinVeto(**kw)
                m_c, _ = measure(cand, None, None, df=frame, market=market)
                line(f"v15[{label}]", m_c, mname)
                asset_results[label] = m_c
            results[asset][mname] = asset_results

    print("\nfalsification verdict per config (candidate/v4 final-balance ratio, BTC control vs ETH test):")
    any_fail = False
    for label, kw in _grid():
        for mname, market in MARKETS:
            btc_r = results["BTC (control)"][mname][label].final_balance / results["BTC (control)"][mname]["v4"].final_balance
            eth_r = results["ETH (test)"][mname][label].final_balance / results["ETH (test)"][mname]["v4"].final_balance
            worse_on_eth = eth_r < btc_r - 0.02
            eth_beats_v4 = eth_r >= 1.0
            flag = "FAIL" if (worse_on_eth and not eth_beats_v4) else ("caution" if worse_on_eth else "ok")
            any_fail = any_fail or flag == "FAIL"
            print(f"  {label:30s} {mname:10s} BTC ratio={btc_r:.3f}x  ETH ratio={eth_r:.3f}x  [{flag}]")
    print(f"\nprimary candidate ({PRIMARY_KW}) checked above; "
          f"overall falsification verdict: {'FAIL' if any_fail else 'no outright FAIL by the pre-registered rule'}")


# --------------------------------------------------------------------- driver


def all_checks() -> None:
    print("=" * 78)
    print("STEP 2b -- descriptive: stablecoin-stress vote frequency vs anchor votes")
    print("=" * 78)
    descriptive()
    print("\n" + "=" * 78)
    print("CENTERPIECE CHECK -- lead-time: does stablecoin stress flip before the price anchors?")
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
            "artifact": artifact, "causality": causality, "eth": eth, "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/kelly_regime_v15_stablecoin_veto.py [{'|'.join(cmds)}]")
