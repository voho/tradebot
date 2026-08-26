"""Shared, read-only utilities and pre-registration for the R-150 round (08-26).

DIRECTION, in one sentence: `champions_council` (L-08, 08-14) has never had
its own top-level position SCALE touched since registration -- it multiplies
its Hedge-blended member vote by a plain, continuous, always-on
`min(target_vol / vol, max_leverage)` ratio (no regime hysteresis of any
kind), one day *before* R-33/L-02 discovered on 08-15 that a continuous
target underperforms a conditional (extremes-only, hysteresis-gated) one --
and this round adds one improvement at each of two structurally different
loci (the portfolio-level scale; the per-member signal magnitude feeding the
blend), while leaving the six-member Hedge weight update, `eta`, `fixed_share`
and the member set completely untouched.

**Which constraint this attacks: SIZE**, the one constraint this project's
own standing diagnosis credits as "what actually worked" -- applied here for
the first time to `champions_council`'s own SCALE (as opposed to its
cross-strategy Hedge-weight ALLOCATION, R-125/R-126, a different sub-axis
already closed).

**Why this object, why this axis, not a 3rd allocation-weight construction
or a 30th `kelly_regime_v4` variant.** `docs/LEDGER.md`'s standing diagnosis
and R-149's own re-ranking (docs/LEDGER.md ~line 15854) confirm the ranked
backlog holds only B-06 (forward paper trading, already running unattended,
18.9-year median horizon per R-78) and B-44 (LOW-priority methodology note),
and that every "profitable, registered, multi-signal object this project has
now given a dedicated sizing round" -- `kelly_regime_v4` (28+ SIZE-axis
attempts), `champions_council`'s own ALLOCATION (R-125/R-126),
`hedge_experts`'s composition (R-128-R-136), `replicator_book`'s blend
(R-148), `universal_kelly`'s exposure mixture (R-149) -- has failed to
improve on its own incumbent. R-125/R-126's own text is explicit that what
was varied was the Hedge weight vector ACROSS `champions_council`'s six
members (Equal Risk Contribution / CVaR budgeting substituted for
exponential weights) -- grep-confirmed against `docs/LEDGER.md`; R-126's own
"What was done" section states its shared harness reuses
"`champions_council`'s own vol-target/deadband tail UNCHANGED -- so only the
weight-vector construction differs." The portfolio-level SCALE this round
touches is that unchanged tail. Read directly from
`src/tradebot/strategies/champions_council.py` (lines 94-98) before writing
this file: `desired = blend * min(self.target_vol / v, self.max_leverage)`
-- no `high_in`/`high_out`/`low_in`/`low_out` hysteresis state machine of any
kind, unlike `kelly_regime_v3`/`v4`'s scale. `champions_council` was
registered 08-14 (L-08), one calendar day *before* L-02 (08-15) discovered
that a conditional/extremes-only target beats a continuous one -- the
upgrade was chronologically impossible at registration time and has never
been retrofitted since, even though `champions_council` already borrows v4's
own `target_vol=0.55`/`max_leverage=2.0`/`vol_span=8*BARS_PER_DAY`
*parameters* verbatim (confirmed by direct comparison of the two files'
`__init__` defaults) -- it copied v3/v4's numbers, never its mechanism. This
is the same "pick a genuinely different, never-improved locus on an
already-profitable object" move R-148 made for `replicator_book`'s SIZE axis
and R-149 made for `universal_kelly`'s exposure mixture, applied here to the
one locus on `champions_council` specifically that both of its own dedicated
rounds (R-125, R-126) left untouched by their own admission.

**Not a duplicate of:**
- R-125/R-126 (`champions_council`'s cross-strategy Hedge weights -> Equal
  Risk Contribution / CVaR-budgeted reallocation ACROSS its six members,
  both NEGATIVE): both rounds' own text discloses the vol-target/deadband
  tail was reused unchanged as their shared control machinery. This round
  changes nothing about how the six members are weighted against each other
  -- `eta`, `fixed_share` and the softmax/Hedge update are byte-identical to
  the control in both branches here -- it changes only what multiplies the
  already-blended vote (conservative) or what each member's own signal looks
  like before blending (novel).
- R-46 (Grossman-Zhou drawdown-based CPPI cushion replacing `kelly_regime_v4`'s
  scale) and its own R-46 predecessor (textbook fixed-multiplier/Hurst-adaptive
  CPPI, also on `kelly_regime_v4`): both are path-dependent, account-equity-floor
  constructions on a DIFFERENT object's ONE directional vote. Neither branch
  here uses a floor, a cushion, or `champions_council`'s own realized equity
  path at all -- both stay strictly volatility-driven (conservative) or
  per-member-payoff-driven (novel), the same family of mechanism
  `champions_council` already partially uses, not the CPPI family this
  project tried and closed twice on a different object.
- R-73 (Deribit DVOL-derived variance risk premium as a bounded,
  never-increase-only multiplicative brake on `kelly_regime_v4`, NEGATIVE,
  4-for-4 failure of that specific architecture): a different object, a
  different (options-implied) data channel, and a structurally different
  mechanism (a one-sided haircut multiplier vs. this round's two-sided
  regime-switching scale / per-member Kelly pre-scaling). Neither branch
  here reads DVOL or any implied-volatility series.
- R-148 (Bongaerts-Kang-van Dijk conditional vol-target bolted onto
  `replicator_book`'s fixed `scale=0.75` constant, conservative; per-species
  fractional-Kelly pre-scaling before `replicator_book`'s replicator blend,
  novel, both NEGATIVE) and R-149 (the same conditional-vol-target machinery
  onto `universal_kelly`'s fixed `kappa=0.5`, conservative; fixed-share
  re-injection into its wealth posterior, novel, both NEGATIVE): this
  round's CONSERVATIVE branch is the THIRD application of the identical,
  unretuned Bongaerts-Kang-van Dijk primitive, each time to a different
  object that had no regime-switching scale of its own before its round --
  disclosed as reuse, not claimed as a fresh mechanism, exactly as R-148 and
  R-149 disclosed their own first and second applications. This round's
  NOVEL branch is the SECOND application of per-member/per-species
  fractional-Kelly pre-scaling (Whitrow 2007) before a Hedge-style blend,
  after R-148's first application to `replicator_book`'s five REPLICATOR-
  weighted species -- applied here to `champions_council`'s six HEDGE-
  weighted (exponential-weights, not replicator-dynamics) members, a
  structurally different combination rule feeding on the same sizing
  primitive. R-148's own novel branch failed decisively (turnover ~3x
  control, ΔSharpe -0.90/-1.00 at inner-validation) from 5-minute-bar
  estimation noise in the per-member mu/var Kelly estimate (grep-confirmed,
  docs/LEDGER.md R-148 section) -- this round's own falsification test
  (below) is pre-registered explicitly as a test of whether that failure
  mode generalizes to a second, differently-weighted object, or is specific
  to replicator dynamics' own slower reallocation. This is disclosed
  plainly, not papered over: the novel branch is a genuine but modest-odds
  bet by the operator's own reading of the prior evidence, run for its
  information value, not because it is expected to win.

**Literature grounding, fetched and read/re-verified via WebSearch this
session before either branch was dispatched:**
- Bongaerts, D., Kang, X., & van Dijk, M. (2020), "Conditional Volatility
  Targeting," *Financial Analysts Journal* 76(4) -- conditional/extremes-only
  vol targeting beats continuous targeting on Sharpe and tail risk at low
  turnover. Already this project's own citation (`kelly_regime_v3`/`v4`,
  PROMOTED; reused by R-148, R-149); reused a THIRD time here, unretuned, as
  the conservative branch's entire scaling mechanism.
- Baur, D. G., & Dimpfl, T. (2018), "Asymmetric volatility in
  cryptocurrencies," *Economics Letters* 173 -- BTC's inverse leverage
  effect, the asset-class fact that makes extremes-only targeting bite here;
  already this project's v3/v4/R-148/R-149 citation, reused for the same
  reason.
- Whitrow, C. (2007), "Algorithms for optimal allocation of bets on many
  simultaneous events," *Journal of the Royal Statistical Society: Series C*
  56(5), 607-623 -- the multi-simultaneous-bet Kelly problem: several
  correlated bets, each with its own estimable edge/variance, sized
  independently before combination. Already this project's own citation
  (R-148's novel branch, `replicator_book`'s five species); reused a SECOND
  time here on `champions_council`'s six Hedge-weighted members, a
  differently-constructed simultaneous-bet set (softmax/exponential-weights
  membership probabilities, not replicator-dynamics fitness shares).
- Han, Y., Yu, J., & Mathew, T. (2025), "Optimal Betting: Beyond the
  Long-Term Growth," arXiv:2503.17927 -- the plug-in Kelly estimator is
  biased upward under realistic (jointly Gaussian, finite-sample) return
  models, which is why a per-signal fractional-Kelly estimate computed at
  high (5-minute) frequency over-sizes relative to its own true edge.
  Grounds this round's pre-registered failure risk for the novel branch
  (below), the same citation R-148's own write-up used to explain its
  observed failure mode after the fact -- used here BEFORE any number
  exists, as a stated prior risk, not a post-hoc explanation. Verified real
  via WebSearch this session; not previously cited anywhere else in this
  project (grep-confirmed against docs/LEDGER.md and docs/RESEARCH.md).

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
- Conservative: `champions_council`'s existing continuous scale and the
  conditional scale differ only in HOW they behave once volatility leaves
  its "steady" band -- if BTC/ETH realized vol rarely enters the "elevated"
  or "extreme" hysteresis states over inner-train/inner-validation (the same
  risk R-149 named for `universal_kelly`), the conditional scale collapses
  to the same `target_vol/vol` ratio already in use, the candidate is
  statistically indistinguishable from the control, and the Step-0 A2 kill
  switch (R^2 > 0.98 against the unmodified control) fires before any Sharpe
  number is read.
- Novel: this is a named, high-prior-probability failure, not a rhetorical
  one -- R-148's own structurally analogous construction (per-signal
  fractional-Kelly pre-scaling before a Hedge-style blend of several
  correlated members) already failed on `replicator_book` from exactly the
  mechanism Han-Yu-Mathew (2025) predicts: causal mu/var estimates at
  5-minute-bar granularity are noisy and biased, so the per-member Kelly
  factor whipsaws between near-zero and its cap, degrading the blend's own
  input quality and inflating turnover well beyond any edge it could add.
  Kill condition, named now: turnover ratio (fills) > 2x the control AND
  ΔSharpe negative on both BTC markets at inner-validation. If this
  reproduces, it is the SECOND structurally distinct object on which
  5-minute-bar per-signal fractional-Kelly pre-scaling has failed the same
  way, closing the mechanism generally rather than per-object.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither may edit it, so both are measured by
identical machinery, the r89-r149 convention. Nothing here reads a bar at or
after OOS_START (2023-01-01); `compare()` asserts this explicitly for every
slice it runs.
"""

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns as inference_daily_returns,
    paired_bootstrap,
    total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# ---------------------------------------------------------------- splits
INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

# champions_council's own shipped constants (do not change: the control must
# be champions_council, not a re-parameterisation of it). Verified against
# src/tradebot/strategies/champions_council.py.
CC_ETA = 0.06
CC_FIXED_SHARE = 1e-4
CC_TARGET_VOL = 0.55
CC_MAX_LEVERAGE = 2.0
CC_VOL_SPAN = 8 * BARS_PER_DAY
CC_DEADBAND = 0.10
CC_WARMUP = 100 * BARS_PER_DAY + 10
CC_NUM_MEMBERS = 6  # KellyRegime, HedgeExperts, ReplicatorBook, UniversalKelly, buy&hold, flat

# kelly_regime_v3/v4's own shipped conditional-vol-target constants,
# reproduced byte-for-byte from src/tradebot/strategies/kelly_regime_v3.py.
# UNTOUCHED by the conservative branch -- reused verbatim, not retuned.
V4_TARGET_VOL = 0.55
V4_MAX_LEVERAGE = 2.0
V4_VOL_SPAN = 8 * BARS_PER_DAY
V4_ANCHOR_SPAN_DAYS = 180
V4_HIGH_IN, V4_HIGH_OUT = 1.70, 1.20
V4_LOW_IN, V4_LOW_OUT = 0.55, 0.85

# Novel branch's own new locus: per-member causal Kelly-fraction estimation.
# The memory timescale reuses champions_council's OWN existing vol span
# (its only pre-existing memory constant) rather than inventing a new free
# parameter -- disclosed explicitly, not hidden.
KELLY_HALFLIFE_BARS = CC_VOL_SPAN  # 2304 bars = 8 days
KELLY_MIN_VAR = 1e-10


# ------------------------------------------------------------------ data

def assert_no_holdout(df: pd.DataFrame, label: str = "") -> None:
    """Fail loudly if any bar at or after the holdout boundary is present."""
    if len(df) and df.index[-1] >= pd.Timestamp(OOS_START, tz="UTC"):
        raise AssertionError(
            f"{label}: frame reaches {df.index[-1]}, at/after OOS_START={OOS_START}")


def _truncate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df[df.index < pd.Timestamp(OOS_START, tz="UTC")]
    assert_no_holdout(out, label)
    return out


def load_btc() -> pd.DataFrame:
    """The committed BTC spot series, truncated before the holdout."""
    df, _label = load_dataset(ROOT / "data", "spot")
    return _truncate(df, "BTC")


def load_eth() -> pd.DataFrame:
    """Bitfinex ETH (this project's standing cross-asset replication series)."""
    return _truncate(load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz"), "ETH")


# ================================================================== (1)
# champions_council's own construction, reproduced EXACTLY where needed for
# the candidates. The CONTROL itself is never reproduced by hand -- it is
# always the literal registered ChampionsCouncil class (see
# champions_council_target below) so the true control can never silently
# diverge from what ships.
# ==================================================================

def champions_council_target(df: pd.DataFrame) -> np.ndarray:
    """The registered ChampionsCouncil's own target path -- the control."""
    from tradebot.strategies.champions_council import ChampionsCouncil
    prepared = ChampionsCouncil().prepare(df.copy())
    return prepared["target"].to_numpy(dtype=float)


def champions_council_target_deadband(df: pd.DataFrame, deadband: float) -> np.ndarray:
    """champions_council's own registered class with ONLY `deadband` widened
    -- a zero-information, turnover-only control for the mandatory B6 gate
    (R-132/R-135's own lesson: a turnover-reduction-in-a-losing-regime
    artifact can clear a falsification battery that never isolates
    turnover). No sizing mechanism differs from the control at all."""
    from tradebot.strategies.champions_council import ChampionsCouncil
    prepared = ChampionsCouncil(deadband=deadband).prepare(df.copy())
    return prepared["target"].to_numpy(dtype=float)


def champions_signals(df: pd.DataFrame, vol_span: int = CC_VOL_SPAN
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduces ChampionsCouncil.prepare()'s member-signal matrix `a`
    (num_members columns: KellyRegime, HedgeExperts, ReplicatorBook,
    UniversalKelly, buy&hold, flat) and its `vol` array, byte-for-byte."""
    from tradebot.strategies.hedge_experts import HedgeExperts
    from tradebot.strategies.kelly_regime import KellyRegime
    from tradebot.strategies.replicator_book import ReplicatorBook
    from tradebot.strategies.universal_kelly import UniversalKelly

    base = df[["open", "high", "low", "close", "volume"]]
    signals = []
    for member in [KellyRegime(), HedgeExperts(), ReplicatorBook(), UniversalKelly()]:
        prepared = member.prepare(base.copy())
        signals.append(np.clip(np.nan_to_num(
            prepared["target"].to_numpy(dtype=np.float64)), -1.0, 1.0))
    n = len(df)
    signals.append(np.ones(n))
    signals.append(np.zeros(n))
    a = np.column_stack(signals)

    r = np.log(df["close"]).diff()
    r_a = r.to_numpy()
    vol = (r.ewm(span=vol_span, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    return a, vol, r_a


def champions_blend(a: np.ndarray, vol: np.ndarray, r_a: np.ndarray, *,
                     eta: float = CC_ETA, fixed_share: float = CC_FIXED_SHARE,
                     member_scale: np.ndarray | None = None) -> np.ndarray:
    """Reproduces the Hedge weight update loop exactly (same `logw`
    accumulation, same softmax, same fixed-share re-injection) and returns
    the pre-scale, pre-deadband blended vote `blend[i] = p[i] @ sig_i`.

    `sig_i` defaults to `a[i]` (control-identical, produces a `blend` that,
    fed through champions_council's OWN unmodified
    `min(target_vol/v, max_leverage)` scale and deadband, reproduces the
    registered class exactly -- checked in `_self_test`). If `member_scale`
    (an (n, num_members) array, already causal/shifted) is supplied, `sig_i
    = a[i] * member_scale[i]` instead -- the novel branch's ONLY structural
    change, confined to this one line; the Hedge weight update itself
    (`g`, `logw`) still uses the UNSCALED `a[i-1]`, exactly as the control
    computes fitness/regret from realized, unscaled member signals.

    On bars where `vol` is not yet valid (the ~1-day EWM warmup at the very
    start of the series, entirely inside champions_council's own
    `warmup=100*BARS_PER_DAY+10` and therefore never read by any measured
    slice), `blend` simply holds its previous value -- a harmless
    placeholder, disclosed here rather than silently reproducing the
    control's own pos-holding behaviour bar for bar (which would require
    also threading the deadband/pos state through this function, entangling
    two mechanisms this round keeps deliberately separate).
    """
    n, num = a.shape
    blend = np.zeros(n)
    logw = np.zeros(num)
    for i in range(1, n):
        v = vol[i]
        if not np.isfinite(v) or v <= 0 or not np.isfinite(r_a[i]):
            blend[i] = blend[i - 1]
            continue
        bar_vol = v / np.sqrt(BARS_PER_YEAR)
        g = np.clip(a[i - 1] * r_a[i] / (3.0 * bar_vol), -1.0, 1.0)
        logw = logw + eta * g
        logw -= logw.max()
        p = np.exp(logw)
        p /= p.sum()
        p = (1.0 - fixed_share) * p + fixed_share / num
        logw = np.log(p)
        sig_i = a[i] if member_scale is None else a[i] * member_scale[i]
        blend[i] = float(p @ sig_i)
    return blend


def apply_deadband(scale_times_blend: np.ndarray, deadband: float = CC_DEADBAND) -> np.ndarray:
    """champions_council's own re-target rule, reproduced verbatim: move
    only if the change in the already-scaled desired position exceeds
    `deadband`."""
    n = len(scale_times_blend)
    target = np.zeros(n)
    pos = 0.0
    for i in range(n):
        desired = float(scale_times_blend[i])
        if abs(desired - pos) > deadband:
            pos = desired
        target[i] = pos
    return target


# ================================================================== (2)
# kelly_regime_v3/v4's conditional volatility-target machinery, reproduced
# byte-for-byte from experiments/r148_shared.py / r149_shared.py /
# kelly_regime_v3.py. UNTOUCHED by this round -- the conservative branch
# feeds it the SAME BTC/ETH symmetric realized volatility v4 itself uses,
# not retuned.
# ==================================================================

def v4_symmetric_vol(df: pd.DataFrame, span: int = V4_VOL_SPAN) -> np.ndarray:
    r = np.log(df["close"]).diff()
    return (r.ewm(span=span, min_periods=BARS_PER_DAY).std()
            * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()


def conditional_target_scale(vol: np.ndarray, anchor_span_days: int = V4_ANCHOR_SPAN_DAYS,
                              high_in: float = V4_HIGH_IN, high_out: float = V4_HIGH_OUT,
                              low_in: float = V4_LOW_IN, low_out: float = V4_LOW_OUT,
                              target_vol: float = V4_TARGET_VOL,
                              max_leverage: float = V4_MAX_LEVERAGE) -> np.ndarray:
    vol = np.asarray(vol, dtype=float)
    slow = (pd.Series(vol).ewm(span=anchor_span_days * BARS_PER_DAY,
                                min_periods=BARS_PER_DAY).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(target_vol / vol, max_leverage)
        steady = np.minimum(target_vol / slow, max_leverage)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    n = len(vol)
    out = np.zeros(n)
    state = 0
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > high_in else (-1 if x < low_in else 0)
            elif state == 1 and x < high_out:
                state = 0
            elif state == -1 and x > low_out:
                state = 0
        out[i] = full[i] if state != 0 else steady[i]
    return out


# ================================================================== (3)
# CONSERVATIVE candidate: replace champions_council's plain, continuous
# `min(target_vol/v, max_leverage)` scale with v4's own (unretuned)
# conditional-volatility-target scale. The member set, the Hedge weight
# update and the deadband rule are byte-identical to the control; only the
# scalar that multiplies the blended vote differs.
# ==================================================================

def conservative_target(df: pd.DataFrame, target_vol: float = V4_TARGET_VOL,
                         max_leverage: float = V4_MAX_LEVERAGE,
                         deadband: float = CC_DEADBAND) -> np.ndarray:
    a, vol, r_a = champions_signals(df)
    blend = champions_blend(a, vol, r_a)
    scale = conditional_target_scale(v4_symmetric_vol(df), target_vol=target_vol,
                                      max_leverage=max_leverage)
    return apply_deadband(blend * scale, deadband)


# ================================================================== (4)
# NOVEL candidate: per-member fractional-Kelly pre-scaling (Whitrow 2007)
# of each of champions_council's six members' own signal, from that
# member's own causal, EWM-estimated Kelly fraction (mean/variance of its
# OWN vol-normalized per-bar payoff `g`, the same quantity already driving
# the Hedge weight update -- no new signal is introduced, only a new use of
# one already computed), BEFORE the Hedge blend. The Hedge weight update
# itself, the portfolio-level scale (still champions_council's own plain
# continuous ratio, UNCHANGED) and the deadband are all identical to the
# control -- only the per-member signal magnitude feeding the blend differs.
# ==================================================================

def per_member_kelly_fraction(a: np.ndarray, vol: np.ndarray, r_a: np.ndarray, *,
                               halflife_bars: int = KELLY_HALFLIFE_BARS,
                               kelly_cap: float = 1.5,
                               min_var: float = KELLY_MIN_VAR) -> np.ndarray:
    """Causal, EWM-estimated per-member Kelly fraction f_k = mu_k / var_k
    (Kelly 1956 single-bet form, applied independently per simultaneous bet
    per Whitrow 2007), fractionalized by clipping to [0, kelly_cap] rather
    than the raw (unbounded, estimation-error-fragile) full-Kelly value
    (MacLean, Thorp & Ziemba 2010). `mu_k`/`var_k` are the EWM mean/variance
    of member k's OWN vol-normalized per-bar payoff `g` (the same quantity
    champions_council's own Hedge update already computes internally),
    shifted by one bar so bar i's fraction uses only payoffs strictly
    before i -- fully vectorized and causal by construction (no sequential
    state needed: pandas `ewm(adjust=False)` is itself a causal running
    accumulator, and `.shift(1)` removes bar i's own contribution)."""
    n, num = a.shape
    bar_vol = vol / np.sqrt(BARS_PER_YEAR)
    g = np.zeros((n, num))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = a[:-1] * (r_a[1:] / (3.0 * bar_vol[1:]))[:, None]
    valid = np.isfinite(vol) & (vol > 0) & np.isfinite(r_a)
    g[1:] = np.where(valid[1:, None], np.clip(np.nan_to_num(ratio), -1.0, 1.0), 0.0)

    alpha = 1.0 / float(halflife_bars)
    g_df = pd.DataFrame(g)
    mu = g_df.ewm(alpha=alpha, adjust=False).mean().shift(1).to_numpy()
    var = g_df.ewm(alpha=alpha, adjust=False).var(bias=False).shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(var > min_var, mu / var, 0.0)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(f, 0.0, kelly_cap)


def novel_target(df: pd.DataFrame, kelly_cap: float = 1.5,
                  target_vol: float = CC_TARGET_VOL, max_leverage: float = CC_MAX_LEVERAGE,
                  deadband: float = CC_DEADBAND) -> np.ndarray:
    a, vol, r_a = champions_signals(df)
    kelly_f = per_member_kelly_fraction(a, vol, r_a, kelly_cap=kelly_cap)
    blend = champions_blend(a, vol, r_a, member_scale=kelly_f)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.minimum(target_vol / vol, max_leverage)
    scale = np.where(np.isfinite(scale), scale, 0.0)
    return apply_deadband(blend * scale, deadband)


# ------------------------------------------------------- causal truncation

def causal_truncation_probe_series(build_fn, df: pd.DataFrame,
                                    cuts: tuple[float, ...] = (0.35, 0.55, 0.80)) -> bool:
    full = np.asarray(build_fn(df), dtype=float)
    for cut in cuts:
        k = int(len(df) * cut)
        if k < BARS_PER_DAY * 2:
            continue
        part = np.asarray(build_fn(df.iloc[:k]), dtype=float)
        a, b = full[:k], part
        m = np.isfinite(a) & np.isfinite(b)
        if not np.allclose(a[m], b[m], atol=1e-8, rtol=1e-7):
            bad = int(np.sum(~np.isclose(a[m], b[m], atol=1e-8, rtol=1e-7)))
            raise AssertionError(f"{build_fn.__name__} causality FAIL at cut={cut}: {bad} bars differ")
        perturbed = df.copy()
        tail = perturbed.iloc[k:].copy()
        for col in ("open", "high", "low", "close"):
            if col in tail.columns:
                tail[col] = tail[col] * 3.7 + 1.0
        perturbed.iloc[k:] = tail
        pert = np.asarray(build_fn(perturbed), dtype=float)
        pm = np.isfinite(a) & np.isfinite(pert[:k])
        if not np.allclose(a[pm], pert[:k][pm], atol=1e-8, rtol=1e-7):
            raise AssertionError(f"{build_fn.__name__} peeks at bar>=k, cut={cut}")
    return True


# ================================================================== (5)
# compare(): run any pure `build_target(df) -> np.ndarray` candidate over
# inner-train, inner-validation and the ETH replication slice, vs
# champions_council's own control, on BOTH markets. Never touches
# OOS_START. Structurally identical to r148_shared.py / r149_shared.py's
# compare().
# ==================================================================

SLICES: dict[str, tuple[str | None, str | None]] = {
    "inner_train": (INNER_TRAIN_START, INNER_TRAIN_END),
    "inner_val": (INNER_VAL_START, INNER_VAL_END),
}
ETH_SLICE_NAME = "eth_replication"

for _name, (_s, _e) in SLICES.items():
    if _e is not None:
        assert pd.Timestamp(_e) < pd.Timestamp(OOS_START), (
            f"SLICES[{_name!r}] end={_e} is not before OOS_START={OOS_START}")


@dataclass
class SliceResult:
    name: str
    market: str
    final_balance: float
    sharpe: float
    max_drawdown_pct: float
    num_trades: int
    log_growth: float
    daily: np.ndarray
    mean_abs_exposure: float
    realized_vol: float


def daily_simple_returns(equity: pd.Series) -> np.ndarray:
    return inference_daily_returns(equity).to_numpy()


class TargetStrategy(Strategy):
    """Wrap a pure ``build_target(df) -> np.ndarray`` as a runnable strategy.

    champions_council's own registered convention is ``order_notional``
    (verified against ``src/tradebot/strategies/champions_council.py``'s
    ``on_bar``) -- fraction of EQUITY, independent of leverage, with the
    broker's own leverage cap clamping anything the market cannot support.
    Correct for control, conservative and novel here alike: no
    order_target/order_notional convention mismatch to guard against
    (unlike R-148's replicator_book), exactly as in R-149's universal_kelly.
    """

    name = "r150_control"
    warmup = CC_WARMUP

    def __init__(self, build_target, name: str = "r150_control",
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


def run_slice(strategy: Strategy, df: pd.DataFrame, start: str | None, end: str | None,
              slice_name: str, market: MarketSpec = SPOT,
              balance: float = 1_000.0) -> SliceResult:
    if end is not None:
        assert pd.Timestamp(end) < pd.Timestamp(OOS_START), (
            f"run_slice({slice_name!r}): end={end} is not before OOS_START={OOS_START}")
    assert_no_holdout(df, slice_name)

    res = run_period(strategy, df, start, end, market=market, start_balance=balance)
    assert_no_holdout(res.equity.to_frame(), f"{slice_name} result")
    m = compute_metrics(res)
    d = daily_simple_returns(res.equity)
    exposure = res.df["target"].to_numpy() if "target" in res.df.columns else np.array([np.nan])
    return SliceResult(
        name=slice_name, market=market.name, final_balance=m.final_balance,
        sharpe=m.sharpe, max_drawdown_pct=m.max_drawdown_pct,
        num_trades=m.num_trades, log_growth=float(total_log_return(d)), daily=d,
        mean_abs_exposure=float(np.nanmean(np.abs(exposure))),
        realized_vol=float(np.nanstd(d) * np.sqrt(365.25)) if len(d) > 1 else float("nan"),
    )


def paired_diff(candidate: np.ndarray, control: np.ndarray, *,
                 mean_block: float = 30.0, n_boot: int = 2_000, seed: int = 0):
    n = min(len(candidate), len(control))
    return paired_bootstrap(np.asarray(candidate[-n:], dtype=float),
                             np.asarray(control[-n:], dtype=float),
                             total_log_return, mean_block=mean_block,
                             n_boot=n_boot, seed=seed)


def compare(candidate_build, *, label: str, btc: pd.DataFrame | None = None,
            eth: pd.DataFrame | None = None, control_build=None,
            markets: tuple[MarketSpec, ...] = (SPOT, FUTURES),
            include_eth: bool = True, seed: int = 0) -> list[dict]:
    if control_build is None:
        control_build = champions_council_target
    if btc is None:
        btc = load_btc()
    assert_no_holdout(btc, "compare(): btc")
    if include_eth and eth is None:
        eth = load_eth()
    if include_eth:
        assert_no_holdout(eth, "compare(): eth")

    cand = TargetStrategy(candidate_build, name=f"r150_{label}")
    ctrl = TargetStrategy(control_build, name="champions_council")

    rows = []
    jobs = [(name, start, end, btc) for name, (start, end) in SLICES.items()]
    if include_eth:
        jobs.append((ETH_SLICE_NAME, None, None, eth))

    for slice_name, start, end, df in jobs:
        for market in markets:
            a = run_slice(cand, df, start, end, slice_name, market)
            b = run_slice(ctrl, df, start, end, slice_name, market)
            pr = paired_diff(a.daily, b.daily, seed=seed)
            exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure
                         if b.mean_abs_exposure else float("nan"))
            vol_ratio = (a.realized_vol / b.realized_vol
                         if b.realized_vol else float("nan"))
            rows.append({
                "label": label, "slice": slice_name, "market": market.name,
                "cand_final": a.final_balance, "ctrl_final": b.final_balance,
                "cand_log_growth": a.log_growth, "ctrl_log_growth": b.log_growth,
                "d_log_growth": a.log_growth - b.log_growth,
                "cand_sharpe": a.sharpe, "ctrl_sharpe": b.sharpe,
                "d_sharpe": a.sharpe - b.sharpe,
                "cand_dd": a.max_drawdown_pct, "ctrl_dd": b.max_drawdown_pct,
                "d_dd": a.max_drawdown_pct - b.max_drawdown_pct,
                "cand_trades": a.num_trades, "ctrl_trades": b.num_trades,
                "exposure_ratio": exp_ratio, "vol_ratio": vol_ratio,
                "risk_matched": bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1)
                                if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False,
                "boot_d_loggrowth": pr.diff.point,
                "boot_lo": pr.diff.lo, "boot_hi": pr.diff.hi,
                "excludes_zero": bool(pr.diff.lo > 0 or pr.diff.hi < 0),
            })
    return rows


def print_rows(rows: list[dict]) -> None:
    hdr = (f"{'label':26s} {'slice':16s} {'market':11s} {'cand$':>10s} {'ctrl$':>10s} "
           f"{'dSh':>6s} {'dDD':>7s} {'expR':>5s} {'volR':>5s} {'RM':>3s} "
           f"{'dlogG':>7s} {'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label'][:26]:26s} {r['slice']:16s} {r['market']:11s} "
              f"{r['cand_final']:10,.0f} {r['ctrl_final']:10,.0f} "
              f"{r['d_sharpe']:+6.2f} {r['d_dd']:+7.1f} "
              f"{r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
              f"{'Y' if r['risk_matched'] else 'n':>3s} "
              f"{r['boot_d_loggrowth']:+7.3f} {r['boot_lo']:+8.3f},{r['boot_hi']:+8.3f} "
              f"{'YES' if r['excludes_zero'] else 'no':>5s}")


def fee_at(market: MarketSpec, fee_rate: float) -> MarketSpec:
    return MarketSpec(name=market.name, leverage=market.leverage, fee_rate=fee_rate,
                       allow_short=market.allow_short,
                       maintenance_margin_rate=market.maintenance_margin_rate,
                       min_notional=market.min_notional, pays_funding=market.pays_funding)


def run_b6(*, primary_market: MarketSpec, cand_inner_val_trades: int,
           deadband_grid: tuple[float, ...] = (0.15, 0.20, 0.30, 0.50, 0.75, 1.00),
           btc: pd.DataFrame | None = None, seed: int = 0) -> dict:
    """B6 (mandatory if B1 passes): the zero-information, turnover-matched
    control R-132/R-135 found necessary. Finds the deadband in
    `deadband_grid` whose `champions_council_target_deadband` inner-
    validation trade count on `primary_market` is closest to
    `cand_inner_val_trades`, then measures THAT deadband-only (no sizing
    change at all) control's own paired difference against the TRUE
    (deadband=0.10) control. If it ALSO clears B1's own bar, the candidate's
    B1 pass is a turnover-reduction artifact, not its own mechanism."""
    if btc is None:
        btc = load_btc()
    true_ctrl_strategy = TargetStrategy(champions_council_target, name="r150_true_control")
    true_ctrl_slice = run_slice(true_ctrl_strategy, btc, INNER_VAL_START, INNER_VAL_END,
                                 "inner_val", primary_market)
    best = None
    trials = {}
    for db in deadband_grid:
        build = functools.partial(champions_council_target_deadband, deadband=db)
        strat = TargetStrategy(build, name=f"r150_db{db:.2f}")
        sl = run_slice(strat, btc, INNER_VAL_START, INNER_VAL_END, "inner_val", primary_market)
        trials[db] = sl.num_trades
        diff = abs(sl.num_trades - cand_inner_val_trades)
        if best is None or diff < best[0]:
            best = (diff, db, sl)
    _, chosen_db, chosen_slice = best
    pr = paired_diff(chosen_slice.daily, true_ctrl_slice.daily, seed=seed)
    d_sharpe = chosen_slice.sharpe - true_ctrl_slice.sharpe
    artifact_detected = bool((d_sharpe > 0.2) or (pr.diff.lo > 0 or pr.diff.hi < 0))
    return {
        "trials_trade_counts": trials, "chosen_deadband": chosen_db,
        "chosen_trades": chosen_slice.num_trades, "cand_trades": cand_inner_val_trades,
        "d_sharpe": d_sharpe, "boot_lo": pr.diff.lo, "boot_hi": pr.diff.hi,
        "artifact_detected": artifact_detected,
    }


def r_squared(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2 or np.std(b) == 0:
        return float("nan")
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((b - np.mean(b)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# ============================================================ pre-registration
#
# PROMOTION BAR (identical shape to R-89-R-149's standard bar, adapted to
# this round's control being `champions_council` rather than
# `kelly_regime_v4`, `replicator_book` or `universal_kelly`):
#
#  A2 (Step-0 non-inertness kill switch): R^2 of the candidate's own final
#     target path against champions_council's unmodified
#     `champions_council_target`, computed on inner-train. If R^2 > 0.98,
#     STOP -- the candidate is a disguised no-op, not a genuinely different
#     sizing mechanism, and no Sharpe number is read past this point.
#  B1: bootstrap paired difference in total log-growth, inner-validation,
#     >= 1 of 2 markets: ΔSharpe > +0.2 OR the 95% bootstrap interval
#     excludes zero.
#  B2 (diagnostic, not gating): exposure_ratio / vol_ratio reported for
#     every cell per R-33's standing rule, so a return improvement is never
#     silently an exposure-level artifact read as a mechanism.
#  B3: plateau -- conservative sweeps target_vol in {0.44, 0.55, 0.66}
#     (+/-20% of V4_TARGET_VOL, the same grid R-149's conservative branch
#     used); novel sweeps kelly_cap in {1.0, 1.5, 2.0} (the same grid
#     R-148's novel branch used, both project precedents reused unchanged
#     rather than re-derived). Sign of d_sharpe on the primary market must
#     hold across the full grid; a single winning cell with no support
#     around it does not clear this bar.
#  B4: ETH same-sign falsification -- the candidate's d_sharpe sign
#     (candidate vs champions_council, inner-validation, BTC) must agree
#     with the ETH replication slice's own sign on at least one market.
#  B5: 0.40% taker-fee-tier re-run on spot (fee_at(SPOT, 0.004)) -- the
#     edge, if any, must not require the 0.10% fee tier to exist. Gates
#     only if B1 passed.
#  B6 (mandatory per R-132/R-135's own lesson, only run if B1 passes): a
#     zero-information, turnover-matched control --
#     `champions_council_target_deadband(df, deadband)`, champions_council's
#     own unmodified scale with ONLY `deadband` widened until its
#     inner-validation trade count on the primary market roughly matches the
#     candidate's -- must NOT itself clear B1's own bar (ΔSharpe > +0.2 OR
#     95% CI excludes zero) against the true (deadband=0.10) control. If it
#     does, the candidate's own B1 pass is reclassified as a
#     turnover-reduction-in-a-losing-regime artifact, not a genuine
#     mechanism, and the round's verdict is NEGATIVE regardless of A2-B5.
#
# Promote-candidate only if A2 does not trip AND B1 passes (>=1 market) AND
# B4 passes AND B5 passes (or is moot because B1 failed) AND B6 does not
# reclassify the pass as an artifact. Anything else is NEGATIVE. This is the
# SAME bar both branches must clear; neither may weaken it after seeing a
# number.
# ============================================================


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=60_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(150)
    innov = rng.normal(0, 0.0006, len(idx))
    jump_idx = rng.choice(len(idx), size=15, replace=False)
    innov[jump_idx] += rng.choice([-1, 1], size=15) * rng.uniform(0.01, 0.03, size=15)
    innov[30_000:35_000] *= 5.0  # a genuine volatility-regime shift
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                        "close": close, "volume": 1.0}, index=idx)

    # (1) control self-consistency: champions_council_target reproduces the
    # registered ChampionsCouncil class exactly (trivially, by construction).
    ctrl = champions_council_target(df)
    assert len(ctrl) == len(df)

    # (1b) STRONGER check: champions_blend()+the control's own UNCHANGED
    # scale/deadband reproduces the registered class byte-for-byte -- this
    # verifies champions_blend()'s per-bar loop is not just well-formed but
    # an exact reproduction of ChampionsCouncil.prepare()'s own arithmetic,
    # the load-bearing assumption both candidate branches depend on.
    a, vol, r_a = champions_signals(df)
    blend = champions_blend(a, vol, r_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        own_scale = np.minimum(CC_TARGET_VOL / vol, CC_MAX_LEVERAGE)
    own_scale = np.where(np.isfinite(own_scale), own_scale, 0.0)
    reproduced = apply_deadband(blend * own_scale, CC_DEADBAND)
    assert np.allclose(reproduced, ctrl, atol=1e-9), \
        "champions_blend()+own scale/deadband diverges from the registered ChampionsCouncil class"

    # (2) conservative candidate is well-formed and genuinely different.
    cons = conservative_target(df)
    assert np.all(cons >= -V4_MAX_LEVERAGE - 1e-9) and np.all(cons <= V4_MAX_LEVERAGE + 1e-9)
    assert not np.allclose(cons, ctrl)
    scale_probe = conditional_target_scale(v4_symmetric_vol(df))
    assert np.nanmax(scale_probe) > 1.0 + 1e-6, \
        "conditional scale never exceeds 1x on synthetic data -- vol-regime shift is not exercising it"

    # (3) novel candidate is well-formed, bounded, and genuinely different.
    nov = novel_target(df)
    assert np.all(nov >= -CC_MAX_LEVERAGE - 1e-9) and np.all(nov <= CC_MAX_LEVERAGE + 1e-9)
    assert not np.allclose(nov, ctrl)
    assert not np.allclose(nov, cons)

    # (4) kelly_cap=0 degenerates the novel branch to an all-flat member
    # signal (sanity: the mechanism is wired correctly, not a no-op that
    # happens to look different from rounding).
    kf0 = per_member_kelly_fraction(a, vol, r_a, kelly_cap=0.0)
    assert np.allclose(kf0, 0.0)

    # (5) causal truncation probes -- no candidate may peek at future bars.
    assert causal_truncation_probe_series(champions_council_target, df)
    assert causal_truncation_probe_series(conservative_target, df)
    assert causal_truncation_probe_series(novel_target, df)
    assert causal_truncation_probe_series(v4_symmetric_vol, df)

    # (6) r_squared sanity.
    assert abs(r_squared(cons, cons) - 1.0) < 1e-9
    assert r_squared(cons, rng.normal(0, 1, len(cons))) < 0.5

    # (7) order_notional dispatch sanity on futures: a target of 2.0 (both
    # branches' own ceiling) should realize ~2x the notional of a target of
    # 1.0, confirming order_notional correctly scales past 1x on a leveraged
    # market -- champions_council's native convention, no dispatch flag
    # needed (unlike R-148's replicator_book).
    fut = MarketSpec.futures(leverage=5.0)

    class _Probe(Strategy):
        name = "r150_probe"
        warmup = 10

        def __init__(self, level: float) -> None:
            self.level = level

        def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
            return frame

        def on_bar(self, ctx: Context) -> None:
            if ctx.i == self.warmup:
                ctx.order_notional(self.level)

    small = df.iloc[:2_000]
    res_1x = run_period(_Probe(1.0), small, None, None, market=fut, start_balance=1_000.0)
    res_2x = run_period(_Probe(2.0), small, None, None, market=fut, start_balance=1_000.0)
    qty_1x = abs(res_1x.fills[0].qty) if res_1x.fills else 0.0
    qty_2x = abs(res_2x.fills[0].qty) if res_2x.fills else 0.0
    assert qty_1x > 0 and qty_2x > 0, "probe fired no fill -- test is not exercising the path"
    ratio = qty_2x / qty_1x
    assert 1.8 < ratio < 2.2, (
        f"order_notional(2.0) vs order_notional(1.0) on 5x futures should differ ~2x, got {ratio:.2f}x")


_self_test()
