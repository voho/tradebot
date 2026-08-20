#!/usr/bin/env python
"""R-55 NOVEL branch, closing backlog item B-22 (second of its two named,
untried fixes): aggregate USDT stablecoin-supply DECELERATION fed into a
precision-weighted CONFIRMING vote on kelly_regime_v4's own price-anchor
regime gate -- not a hard, unweighted override.

Idea, one sentence
------------------
R-54 confirmed that `stablecoin_stress_z` (`experiments/_stablecoin_signal.py`,
unedited, imported here) genuinely LEADS kelly_regime_v4's own 3-anchor
majority price-gate flip (9/12 matched stress episodes, median +16.5
days) -- the first INFO-axis signal in this project's three attempts to
do so -- but a hard veto built on it (`frac=0` while latched "stress",
`experiments/kelly_regime_v15_stablecoin_veto.py`) still lost to v4,
because the threshold tight enough to catch real stress early also fires
on ~24 transient supply wobbles that are not followed by price weakness,
and standing flat through every false alarm costs more than the genuine
early exits recover. This file tests B-22's second named fix directly:
feed the SAME confirmed-leading signal into R-53's own precision-weighted
CONFIRMING-vote architecture (`KellyRegimeV14MacroLead`,
`experiments/kelly_regime_v14_macro_lead.py`) instead of a unilateral
override, so a false stress-onset only dilutes the vote by a fraction of
one anchor (`stable_weight`, e.g. 0.15-0.5 of one full vote) instead of
zeroing it outright -- on the theory that this should be *more forgiving*
of a noisy threshold than the hard veto was, while a genuine, sustained
stress episode still pulls the vote down every bar it is latched.

Constraint attacked: INFO (one price series). Fourth attempt at this
constraint (B-07/R-44 on-chain, R-53 macro, R-54 stablecoin-hard-veto,
now this) -- same signal as R-54, different combination rule.

Not a duplicate of, cited precisely
------------------------------------
- `experiments/kelly_regime_v15_stablecoin_veto.py` (R-54 NOVEL, this
  round's own direct predecessor): SAME signal (`stablecoin_stress_z`,
  imported unchanged from the same `_stablecoin_signal.py`), DIFFERENT
  combination rule -- v15 forces `frac=0` outright while latched
  "stress" (a hard override, B-21's architecture); this file instead
  computes `frac = (anchor_sum + stable_weight*stable_vote)/(3+stable_weight)`,
  a precision-weighted average where a stress latch only ever costs a
  fraction of one vote. v15's own report named this exact fix (its
  "next step" #2) as untried; this file is that test, not a re-run of
  v15's own grid.
- `experiments/kelly_regime_v14_macro_lead.py` (R-53 NOVEL): this file's
  own architectural template -- `KellyRegimeV14MacroLead`'s combination
  rule is copied literally (duplicated, not imported, since that file is
  a private prior-round experiment, not shared infrastructure -- the
  same precedent R-54's `kelly_regime_v15_stablecoin_veto.py` itself
  used when it duplicated `_anchor_votes`/`_macro_vote` from that same
  file). What differs: the feeding signal. R-53 fed VIX/DXY `stress_z`,
  later shown (by R-53's own `leadtime()`) to LAG the 3-anchor majority
  (33% lead rate, median -5.5 days) -- and under that lagging signal, the
  precision-weighted average did WORSE than R-53's own simplest
  hard-override ablation in 10/12 matched cells (R-53's `ablation()`
  finding, cited in the ledger's R-54 row and this project's B-22 entry).
  R-53's own report explicitly named the open question this file answers:
  "might behave differently fed by a signal that actually leads." Testing
  the identical architecture with a genuinely-leading signal is not a
  duplicate of R-53's negative result -- it is the direct, pre-registered
  resolution of the confound R-53 itself could not separate (lagging
  signal vs. architecturally-flawed combination rule).
- B-07/R-44's on-chain branches, L-12/`harsanyi_crowd`: as in R-54's own
  "not a duplicate of" section -- stablecoin supply is neither chain
  activity nor a price-derived crowding signal.
- This round's own disjoint CONSERVATIVE branch (targeting B-22's OTHER
  named fix, a magnitude-and-duration persistence filter on the hard
  veto: `experiments/kelly_regime_v16_stablecoin_persist.py`): NOT read,
  not coordinated with, per this round's explicit isolation rule.

Mechanism, precisely
---------------------
`stablecoin_stress_z` computed exactly as in `_stablecoin_signal.py`
(14-day log growth of aggregate USDT supply, z-scored on a fixed 365-day
trailing window, sign-flipped so positive = risk-off) -- imported
unchanged via `compute_stablecoin_stress`, never recomputed or modified
here. A latched confirming vote, `stable_vote in {0, 1}`, using the SAME
hysteresis discipline R-54's `_stable_vote` and R-53's `_macro_vote` both
used (duplicated below, not imported, per the precedent named above):

    stable_vote -> 0 ("stress")   when stablecoin_stress_z crosses ABOVE thresh_hi
    stable_vote -> 1 ("calm")     when stablecoin_stress_z crosses BELOW thresh_lo
    stable_vote unchanged (latched) while thresh_lo <= stablecoin_stress_z <= thresh_hi
    stable_vote defaults to 1 ("calm"), or wherever stablecoin data is
        unavailable (absence -> no dilution, falls back to v4's own
        anchor-only vote exactly)

Combined vote -- R-53's `KellyRegimeV14MacroLead` rule, literally:

    frac = (anchor_sum + stable_weight * stable_vote) / (3 + stable_weight)

`anchor_sum` is v4's own three UNCHANGED 0/1 latched price-anchor votes.
`stable_weight` is swept over {0 (identity control), 0.15, 0.33, 0.5, 1.0
(unweighted-5-way-average negative control)} -- IDENTICAL grid to R-53's
`WEIGHTS`, per this round's instruction, on the same precision argument:
a rarer, sharper signal (R-54's `descriptive()` found the stablecoin vote
flips far less often than the 20d price anchor) should count as LESS than
a full vote, not more. `stable_weight=0` recovers v4 exactly (identity
check, verified in `causality()`).

Because `stable_vote` defaults to 1 ("calm") and only ever drops to 0
("stress") on a confirmed above-threshold reading, this vote -- like
R-53's -- can only ever pull `frac` DOWN relative to what the three
anchors alone would produce; it never manufactures bullish exposure the
anchors themselves would not already grant. Unlike v15's hard override,
a single false stress-onset costs at most `stable_weight/(3+stable_weight)`
of the vote (12-25% at the swept weights) instead of the entire vote --
the direct mechanism by which this architecture is hypothesized to be
more forgiving of the threshold's false-positive rate while retaining the
confirmed lead-time advantage on genuine episodes.

Pre-registered falsification tests (named before any code ran)
-----------------------------------------------------------------
(1) Does not beat v4 on inner-validation Sharpe (both markets) by more
    than the +/-0.2 noise floor (R-20), OR does but not across a genuine
    parameter plateau (report neighbours, not just the winner).
(2) Fails the pre-2020 BTC-control-vs-ETH differential falsification test
    (R-54's `eth()` rule, reused verbatim).
(3) Does NOT clear R-53's own negative finding: at matching (thresh_hi,
    gap) points, computed fresh in THIS file's `ablation()`, the
    precision-weighted average does not beat an analogous hard-override
    fed by the identical signal -- i.e. the architecture itself, not
    signal quality, is the defect, and feeding it a leading signal changes
    nothing.
(4) Fails the causality tamper probe (price OR stablecoin pathway), or
    the `stable_weight=0` identity check does not recover v4 exactly.
(5) Is an exposure-artifact (R^2 > 0.95 vs. a flat rescale of v4's own
    `target`).

**Pre-registered holdout decision rule, fixed before any result is
read:** the 2023+ holdout is read ONLY IF (1) clears the noise floor
AND sits on a genuine plateau, AND (2) passes, AND (3) shows the
confirming-vote architecture actually beats its own hard-override
ablation (earning its keep, unlike R-53), AND (4)/(5) both pass cleanly.
If ANY of these fail, this branch reports NEGATIVE and the holdout is
never read -- a well-documented negative is a successful outcome per
ROUTINE.md, not a shortfall to be rescued by relaxing this rule after
the fact.

Code reuse decision, stated plainly
-------------------------------------
`_anchor_votes` and the latched hysteresis-vote helper (`_stable_vote`)
are DUPLICATED (not imported) from `kelly_regime_v14_macro_lead.py`,
exactly the precedent `kelly_regime_v15_stablecoin_veto.py` itself set
when it duplicated the same two helpers from the same file (see that
file's own "Code reuse decision" section). `compute_stablecoin_stress`
IS imported unchanged from `experiments/_stablecoin_signal.py` -- the
one designated shared module for this signal, read-only reuse exactly as
R-54's own file used it. Neither `kelly_regime_v14_macro_lead.py` nor
`kelly_regime_v15_stablecoin_veto.py` nor `_stablecoin_signal.py` is
edited anywhere in this session.

Usage
-----
    python experiments/kelly_regime_v16_stablecoin_confirm.py descriptive
    python experiments/kelly_regime_v16_stablecoin_confirm.py sweep        # step 3 (inner-train)
    python experiments/kelly_regime_v16_stablecoin_confirm.py select       # step 3 (inner-validation)
    python experiments/kelly_regime_v16_stablecoin_confirm.py ablation     # falsification (3): confirm vs override, same signal
    python experiments/kelly_regime_v16_stablecoin_confirm.py artifact     # exposure-artifact check
    python experiments/kelly_regime_v16_stablecoin_confirm.py causality    # lookahead probe (price + stablecoin pathway)
    python experiments/kelly_regime_v16_stablecoin_confirm.py eth          # ETH falsification
    python experiments/kelly_regime_v16_stablecoin_confirm.py all          # everything, in order
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
    Duplicated (not imported) from kelly_regime_v14_macro_lead.py -- the
    same precedent kelly_regime_v15_stablecoin_veto.py itself used; see
    this module's docstring, "Code reuse decision"."""
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
    Identical hysteresis discipline to R-53's ``_macro_vote`` and R-54's
    ``_stable_vote`` -- duplicated, not imported, per this module's
    "Code reuse decision"."""
    thresh_lo = thresh_hi - gap
    raw = np.where(stress_z > thresh_hi, 0.0,
                    np.where(stress_z < thresh_lo, 1.0, np.nan))
    vote = pd.Series(raw, index=stress_z.index).ffill().fillna(1.0)
    return vote


class KellyRegimeV16StablecoinConfirm(KellyRegimeV3):
    """v4's 3-anchor vote plus a 4th, precision-weighted CONFIRMING vote
    from stablecoin-supply-deceleration stress. Mechanism:
    ``frac = (anchor_sum + stable_weight * stable_vote) / (3 + stable_weight)``,
    where ``anchor_sum`` is v4's own three 0/1 latched price-anchor votes
    (unchanged) and ``stable_vote`` is the latched stablecoin-stress state
    above (0/1, only ever fires bearish/dilutive). Literally
    ``KellyRegimeV14MacroLead``'s combination rule
    (``experiments/kelly_regime_v14_macro_lead.py``), duplicated with the
    feeding signal swapped from VIX/DXY to stablecoin supply.
    ``stable_weight=0`` recovers v4 exactly (identity check, verified in
    ``causality()``).
    """

    name = "kelly_regime_v16_stablecoin_confirm"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 thresh_hi: float = 1.0, gap: float = 0.75, stable_weight: float = 0.33,
                 **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.thresh_hi = thresh_hi
        self.gap = gap
        self.stable_weight = stable_weight

    def _stress_series(self, df: pd.DataFrame) -> pd.Series:
        if "stablecoin_stress_z_visible" in df.columns:
            return df["stablecoin_stress_z_visible"]
        return compute_stablecoin_stress(df, DATA_DIR)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = _anchor_votes(close, self.horizons, self.band)
        anchor_sum = sum(votes.values())  # 0..3 per bar
        n_anchors = float(len(votes))

        stress_z = self._stress_series(df)
        if self.stable_weight > 0:
            stable_vote = _stable_vote(stress_z, self.thresh_hi, self.gap)
        else:
            stable_vote = pd.Series(1.0, index=df.index)

        combined = (anchor_sum + self.stable_weight * stable_vote) / (n_anchors + self.stable_weight)
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
        df["v16_frac"] = frac
        df["v16_stable_vote"] = stable_vote.to_numpy()
        df["v16_stress_z"] = stress_z.to_numpy()
        df["v16_anchor_sum"] = anchor_sum.to_numpy()
        return df


class KellyRegimeV16StablecoinOverride(KellyRegimeV3):
    """Ablation, for falsification test (3): the hard-override version of
    the SAME signal, at the SAME (thresh_hi, gap) points, computed fresh
    in this session. ``frac = 0`` while ``stable_vote`` reads 0 ("stress"),
    v4's own unmodified 3-anchor average otherwise -- architecturally
    identical to ``kelly_regime_v15_stablecoin_veto.py``'s
    ``KellyRegimeV15StablecoinVeto`` (duplicated here, not imported, same
    precedent as above), used ONLY as the comparison arm inside
    ``ablation()`` to test whether the confirming-vote architecture earns
    its keep over the simplest baseline, the R-53/R-40/R-46 "elaboration
    adds nothing over a simpler baseline" check.
    """

    name = "kelly_regime_v16_stablecoin_override_ablation"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 thresh_hi: float = 1.0, gap: float = 0.75, **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.thresh_hi = thresh_hi
        self.gap = gap

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
        stable_vote = _stable_vote(stress_z, self.thresh_hi, self.gap).to_numpy()

        frac = np.where(stable_vote == 0.0, 0.0, anchor_frac)
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
        df["v16o_frac"] = frac
        df["v16o_stable_vote"] = stable_vote
        return df


# ------------------------------------------------------------- the grid

# WEIGHTS: identical grid to R-53's KellyRegimeV14MacroLead -- 0.0 is the
# explicit identity control (recovers v4 exactly, independent of thresh/gap),
# 1.0 is the explicit unweighted-4-way-average negative control, 0.15/0.33/0.5
# are the precision-weighted candidates per this round's brief.
WEIGHTS = (0.15, 0.33, 0.5, 1.0)

# THRESH_GAP: 4 points drawn from R-54's own 3x3 grid (thresh_hi in
# {0.75,1.0,1.25} x gap in {0.0,0.75,1.25}) -- the round brief's minimum of
# 2, extended to 4 for better plateau resolution. "primary" is R-54's own
# selected primary; "tightest" is R-54's single most false-positive-prone
# cell (thresh=0.75, gap=0.0, 24 stress-onsets, no hysteresis dead-zone --
# the setting a hard veto handled worst); "tight-hys" adds the hysteresis
# band back at the same tight threshold; "loose" is R-54's loosest cell,
# included as a low-false-positive contrast.
THRESH_GAP = (
    ("primary", 1.0, 0.75),
    ("tightest", 0.75, 0.0),
    ("tight-hys", 0.75, 0.75),
    ("loose", 1.25, 1.25),
)
PRIMARY_KW = dict(thresh_hi=1.0, gap=0.75, stable_weight=0.33)


def _grid():
    """17 configs: identity control (1) + 4 thresh/gap x 4 weights (16)."""
    out = [("identity (stable_weight=0)", dict(thresh_hi=1.0, gap=0.75, stable_weight=0.0))]
    for tg_label, thresh_hi, gap in THRESH_GAP:
        for weight in WEIGHTS:
            label = (f"{tg_label} (thresh={thresh_hi:.2f} gap={gap:.2f}) "
                      f"w={weight:.2f}{'(unweighted)' if weight == 1.0 else ''}")
            out.append((label, dict(thresh_hi=thresh_hi, gap=gap, stable_weight=weight)))
    return out


def _config_key(kw: dict) -> str:
    return f"confirm|thresh={kw['thresh_hi']:.3f}|gap={kw['gap']:.3f}|w={kw['stable_weight']:.3f}"


def _override_grid():
    out = []
    for tg_label, thresh_hi, gap in THRESH_GAP:
        label = f"override {tg_label} (thresh={thresh_hi:.2f} gap={gap:.2f})"
        out.append((label, dict(thresh_hi=thresh_hi, gap=gap)))
    return out


def _override_config_key(kw: dict) -> str:
    return f"override|thresh={kw['thresh_hi']:.3f}|gap={kw['gap']:.3f}"


# ------------------------------------------------------- step 2b: descriptive


def descriptive() -> None:
    """Step 2b -- purely descriptive, no free parameter fitted. The
    lead-time centerpiece check itself is NOT re-run here: it is a
    property of the raw signal + hysteresis vote alone, independent of how
    that vote is subsequently combined with the anchors, and was already
    established by R-54's `leadtime()` (9/12 episodes lead, median +16.5
    days) at these same thresh/gap settings -- re-deriving it would
    reproduce identical numbers, not add information. This function
    instead reports vote-transition frequency, the empirical basis for
    treating stable_weight as a fraction of one vote."""
    lo, hi = TRAIN[0], VALID[1]
    frame = DF.loc[lo:hi]
    close = frame["close"]

    print(f"descriptive window (inner-train + inner-validation): {lo} -> {hi}")
    print("(lead-time property established by R-54's leadtime(): 9/12 episodes lead "
          "the 3-anchor majority, median +16.5 days -- not re-derived here, see report)")
    print("\nprice-anchor vote transition counts over the SAME window (context):")
    for days in (20, 40, 80):
        anchor = DF["close"].rolling(int(days * BARS_PER_DAY)).mean().loc[lo:hi]
        v = pd.Series(np.where(close > anchor * 1.01, 1.0,
                                np.where(close < anchor * 0.99, 0.0, np.nan)),
                      index=close.index).ffill().fillna(0.0)
        print(f"  {days:>3d}d price anchor: {int(v.ne(v.shift()).sum()):>4d} vote flips")

    stress = frame["stablecoin_stress_z_visible"]
    print(f"\nstablecoin_stress_z coverage in window: {stress.notna().sum():,} / {len(stress):,} bars")
    for tg_label, thresh_hi, gap in THRESH_GAP:
        vote = _stable_vote(stress, thresh_hi, gap)
        flips_to_stress = int(((vote == 0.0) & (vote.shift() == 1.0)).sum())
        flips_to_calm = int(((vote == 1.0) & (vote.shift() == 0.0)).sum())
        print(f"  {tg_label:10s} thresh={thresh_hi:.2f} gap={gap:.2f}: "
              f"{flips_to_stress} stress-onset event(s), {flips_to_calm} calm-return event(s)")


# --------------------------------------------------------------- step 3: sweep


def sweep() -> None:
    """Step 3: every config on inner-train ONLY, spot primary."""
    print(f"\nINNER-TRAIN {TRAIN} / spot -- benchmarks:")
    for name in ("buy_and_hold", INCUMBENT):
        m, _ = measure(get_strategy(name), *TRAIN, market=SPOT)
        line(name, m, "spot")

    print(f"\nINNER-TRAIN {TRAIN} / spot -- candidate configurations:")
    for label, kw in _grid():
        m, _ = measure(KellyRegimeV16StablecoinConfirm(**kw), *TRAIN, market=SPOT,
                        config_key=_config_key(kw))
        line(label, m, "spot")

    print(f"\nconfigurations evaluated (step 3, distinct thresh/gap/weight triples): {N_EVALUATED}")


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
            m, _ = measure(KellyRegimeV16StablecoinConfirm(**kw), *VALID, market=market,
                            config_key=_config_key(kw))
            cell[mname] = m
            line(label, m, mname)
            rows.append({"label": label, **kw, "market": mname,
                         "final": m.final_balance, "sharpe": m.sharpe,
                         "max_dd": m.max_drawdown_pct, "trades": m.num_trades})
        # selection rule fixed in advance: min(train, valid) spot sharpe,
        # guards against a train-loses/validation-wins overfit signature
        m_train, _ = measure(KellyRegimeV16StablecoinConfirm(**kw), *TRAIN, market=SPOT)
        score = min(m_train.sharpe, cell["spot"].sharpe)
        if score > best_score:
            best_label, best_kw, best_score = label, kw, score

    print(f"\nbest by min(train,valid) spot Sharpe: {best_label}  (score={best_score:.2f})")
    print(f"v4 control spot Sharpe: {ctl['spot'].sharpe:.2f}   "
          f"v4 control futures Sharpe: {ctl['futures 5x'].sharpe:.2f}")

    print("\nside-by-side: best candidate's TRAIN window vs VALIDATION window (overfitting-signature check):")
    for mname, market in MARKETS:
        m_train, _ = measure(KellyRegimeV16StablecoinConfirm(**best_kw), *TRAIN, market=market)
        m_valid, _ = measure(KellyRegimeV16StablecoinConfirm(**best_kw), *VALID, market=market)
        m_train_v4, _ = measure(get_strategy(INCUMBENT), *TRAIN, market=market)
        m_valid_v4, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        print(f"  {mname}:")
        print(f"    candidate  TRAIN final=${m_train.final_balance:>11,.0f} sharpe={m_train.sharpe:.2f} DD={m_train.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid.final_balance:>11,.0f} sharpe={m_valid.sharpe:.2f} DD={m_valid.max_drawdown_pct:.1f}%")
        print(f"    v4         TRAIN final=${m_train_v4.final_balance:>11,.0f} sharpe={m_train_v4.sharpe:.2f} DD={m_train_v4.max_drawdown_pct:.1f}%  "
              f"VALID final=${m_valid_v4.final_balance:>11,.0f} sharpe={m_valid_v4.sharpe:.2f} DD={m_valid_v4.max_drawdown_pct:.1f}%")
        print(f"    candidate - v4 (valid): Delta sharpe={m_valid.sharpe - m_valid_v4.sharpe:+.3f}  "
              f"Delta DD={m_valid.max_drawdown_pct - m_valid_v4.max_drawdown_pct:+.1f}pp")

    print("\nparameter-neighbourhood plateau check (spot, inner-validation Sharpe, all thresh/gap x weight cells):")
    grid_by_key = {(tg, w): None for tg, _, _ in THRESH_GAP for w in WEIGHTS}
    for tg_label, thresh_hi, gap in THRESH_GAP:
        for weight in WEIGHTS:
            m, _ = measure(KellyRegimeV16StablecoinConfirm(thresh_hi=thresh_hi, gap=gap, stable_weight=weight),
                            *VALID, market=SPOT)
            grid_by_key[(tg_label, weight)] = m.sharpe
    for tg_label, thresh_hi, gap in THRESH_GAP:
        row = "  ".join(f"w={w:.2f}:{grid_by_key[(tg_label, w)]:.2f}" for w in WEIGHTS)
        print(f"  {tg_label:10s} (thresh={thresh_hi:.2f} gap={gap:.2f})  {row}")
    m_ident, _ = measure(KellyRegimeV16StablecoinConfirm(stable_weight=0.0), *VALID, market=SPOT)
    print(f"  identity (stable_weight=0, should equal v4 spot Sharpe {ctl['spot'].sharpe:.2f}): {m_ident.sharpe:.2f}")

    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    return best_label, best_kw


# -------------------------------------------------------- falsification test (3): ablation


def ablation() -> None:
    """Falsification test (3), pre-registered centerpiece of this round:
    does the precision-weighted CONFIRMING vote beat an analogous hard
    OVERRIDE, both fed by the identical stablecoin signal, at matching
    (thresh_hi, gap) points -- computed fresh here, not cited from R-54's
    report? This is the direct test of whether R-53's own negative finding
    (averaged vote loses to hard override in 10/12 cells, when fed a
    LAGGING signal) still holds once the feeding signal is one that
    genuinely LEADS (R-54's confirmed result). The confirming-vote
    candidate uses its PRIMARY weight (0.33, matching R-53's own primary)
    at each thresh/gap point."""
    print("ABLATION: precision-weighted CONFIRMING vote (primary weight=0.33) vs hard OVERRIDE, same signal")
    for tg_label, thresh_hi, gap in THRESH_GAP:
        confirm_kw = dict(thresh_hi=thresh_hi, gap=gap, stable_weight=PRIMARY_KW["stable_weight"])
        over_kw = dict(thresh_hi=thresh_hi, gap=gap)
        print(f"\n-- {tg_label} (thresh={thresh_hi:.2f} gap={gap:.2f}) --")
        for split_name, split in (("TRAIN", TRAIN), ("VALID", VALID)):
            for mname, market in MARKETS:
                m_conf, _ = measure(KellyRegimeV16StablecoinConfirm(**confirm_kw), *split, market=market,
                                     config_key=_config_key(confirm_kw))
                m_over, _ = measure(KellyRegimeV16StablecoinOverride(**over_kw), *split, market=market,
                                     config_key=_override_config_key(over_kw))
                print(f"  {split_name:5s} {mname:10s}  "
                      f"confirm final=${m_conf.final_balance:>11,.0f} sharpe={m_conf.sharpe:.2f} DD={m_conf.max_drawdown_pct:.1f}%   "
                      f"override final=${m_over.final_balance:>11,.0f} sharpe={m_over.sharpe:.2f} DD={m_over.max_drawdown_pct:.1f}%   "
                      f"Delta(confirm-override) sharpe={m_conf.sharpe - m_over.sharpe:+.3f}")

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
        cand = KellyRegimeV16StablecoinConfirm(**kw)
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
        s = KellyRegimeV16StablecoinConfirm(**kw)
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
        for col in ("target", "v16_frac", "v16_stable_vote", "v16_anchor_sum"):
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

    tmp_root = Path(tempfile.mkdtemp(prefix="v16_stablecoin_confirm_causality_"))
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

    # identity check: stable_weight=0 recovers v4 exactly on a plain slice
    v4 = get_strategy(INCUMBENT)
    ident = KellyRegimeV16StablecoinConfirm(thresh_hi=1.0, gap=0.75, stable_weight=0.0)
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
    """Falsification test (2), pre-registered rule below, fixed before
    running.

    ETH-USD Bitfinex spot (2016-03 -> 2019-12-31) against USDT-supply
    coverage (2017-01-01 ->). Stablecoin supply is asset-agnostic by
    construction -- the pre-registered rule mirrors R-54's/R-53's own
    eth(): if the candidate is not at least comparable to v4 on ETH, or is
    visibly worse on ETH than on the BTC control through the identical
    code, this direction fails. An ETH-only failure must be reported, not
    hidden.
    """
    eth_spot = load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")
    eth_stress = compute_stablecoin_stress(eth_spot, DATA_DIR)
    eth_df = eth_spot.copy()
    eth_df["stablecoin_stress_z_visible"] = eth_stress

    overlap = eth_df["stablecoin_stress_z_visible"].dropna()
    print(f"ETH spot file: {len(eth_spot):,} bars  {eth_spot.index[0]:%Y-%m-%d} -> "
          f"{eth_spot.index[-1]:%Y-%m-%d}")
    print(f"stablecoin stress coverage overlapping ETH spot: {len(overlap):,} bars "
          f"visible from {overlap.index[0]:%Y-%m-%d} to {overlap.index[-1]:%Y-%m-%d}")

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
                cand = KellyRegimeV16StablecoinConfirm(**kw)
                m_c, _ = measure(cand, None, None, df=frame, market=market)
                line(f"v16[{label}]", m_c, mname)
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
            print(f"  {label:45s} {mname:10s} BTC ratio={btc_r:.3f}x  ETH ratio={eth_r:.3f}x  [{flag}]")
    print(f"\nprimary candidate ({PRIMARY_KW}) checked above; "
          f"overall falsification verdict: {'FAIL' if any_fail else 'no outright FAIL by the pre-registered rule'}")


# --------------------------------------------------------------------- driver


def all_checks() -> None:
    print("=" * 78)
    print("STEP 2b -- descriptive: stablecoin-stress vote frequency vs anchor votes")
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
    print("FALSIFICATION TEST (3) -- ablation: confirming vote vs hard override, same signal")
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
    cmds = {"descriptive": descriptive, "sweep": sweep, "select": select,
            "ablation": ablation, "artifact": artifact, "causality": causality, "eth": eth,
            "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/kelly_regime_v16_stablecoin_confirm.py [{'|'.join(cmds)}]")
