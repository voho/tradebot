"""Shared, read-only utilities and pre-registration for the R-148 round (08-26).

DIRECTION, in one sentence: `replicator_book` (L-10, 08-12) has never had its
own SIZE mechanism touched since registration -- its final position is the
replicator-weighted blend of five species' signals multiplied by a single
FIXED constant (`scale=0.75`), with no volatility-based risk management of
any kind -- and this round adds one, using two structurally different loci
for it, while leaving the species definitions, the fitness measure and the
replicator/logit weight-update dynamics completely untouched.

**Which constraint this attacks: SIZE**, the one constraint this project's
own standing diagnosis credits as "what actually worked" -- but for the
first time applied to `replicator_book` rather than `kelly_regime_v4`.

**Why this object, not a 20th `kelly_regime_v4` variant.** `docs/LEDGER.md`'s
standing diagnosis and R-147's own re-ranking confirm the backlog holds only
B-06 (forward paper trading, already running unattended) and that
`kelly_regime_v4`'s own axes are exhausted: 19+ INFO-axis signals (on-chain
activity/hash rate, VIX/DXY macro, stablecoin supply, Deribit DVOL level and
momentum, Deribit term-structure level and slope, MVRV valuation, Bitcoin
dominance/Google Trends sentiment composite, transfer entropy), 28+
internal SIZE-axis retunes of the vote or the conditional-vol scale, ERR
closed across seven notions of uncertainty including this project's most
recent round (R-147, the vote's own combination weights via James-Stein
shrinkage and sequential Bayesian model averaging), eight structurally
distinct regime-timing mechanisms (HMM, BOCPD, Kalman LLT, critical slowing
down, transfer entropy, Hawkes, CUSUM, LPPLS), five COST-model families, and
four N~3 significance procedures. `champions_council`'s between-strategy
allocation (R-125/R-126) and `hedge_experts`'s expert composition
(R-128-R-136) have each also received dedicated rounds. `replicator_book`
has not: grep-confirmed against `docs/LEDGER.md`, its only two mentions in
the entire research log are its own L-10 registration row and a single
incidental listing as one of four `champions_council` members in R-125's
own text (`docs/LEDGER.md` line ~2639) -- neither touches its internal
mechanism. This is the same "pick a genuinely different OBJECT whose own
axis was never varied" move R-107 (multi-asset registration engine) and
R-125 (`champions_council`'s allocation) made when `kelly_regime_v4`'s own
axes were exhausted, applied to the one profitable, registered,
never-improved object left.

**Not a duplicate of:**
- L-10's own registration finding ("fitness measured on realized returns is
  a lagging estimator; the reallocation arrives after the regime"): that is
  a claim about the SPEED of the weight update, not about whether the
  strategy manages risk once a position is held. This round does not touch
  fitness, the replicator update, or its halflife -- it changes only how
  large a position the (unchanged, still-lagging) blended signal turns
  into. Both branches are pre-registered as blind to whether L-10's lag
  problem is real; if it is, a volatility-target overlay is not expected to
  fix it (named as a failure risk below, not papered over).
- R-125/R-126 (`champions_council`'s Hedge-weight allocation ACROSS six
  member strategies, replaced with Equal Risk Contribution / CVaR
  budgeting): a portfolio-of-STRATEGIES object. This round is entirely
  INSIDE `replicator_book`, one of those six members, and does not touch
  `champions_council`, Hedge, or any cross-strategy weight.
- R-62 (kelly_regime_v4's vote x scale factorization) and the 28+ SIZE-axis
  rounds on that strategy: those retune ONE anchor-based directional vote's
  own conditional volatility target. This round's conservative branch reuses
  that exact, already-validated MACHINERY (Bongaerts-Kang-van Dijk 2020,
  reproduced byte-for-byte from `kelly_regime_v3`/`v4`, not re-derived or
  retuned) but bolts it onto a *different* strategy's *different* underlying
  signal (`replicator_book`'s five-species blend, not v4's three-anchor
  vote); the novel branch's per-species Kelly-fraction sizing has no
  analogue anywhere in the `kelly_regime_v4` family, which has exactly one
  directional signal to size, not five simultaneous ones.
- R-38 (Busseti/Ryu/Boyd risk-constrained Kelly on `kelly_regime`'s own
  point-drift estimate) and R-125 (CVaR substituted into `kelly_regime_v4`'s
  own `scale`): both resize ONE already-existing directional bet. The novel
  branch here sizes FIVE simultaneous, correlated bets (the five species)
  independently before they are blended -- the multi-simultaneous-bet Kelly
  problem (Whitrow 2007, cited below), which this project has never posed
  because every other registered strategy that uses Kelly sizing
  (`kelly_regime*`, `universal_kelly`) trades one blended signal, not several
  named sub-strategies with individually estimable edge/variance.

**Literature grounding, fetched and read via WebSearch before either branch
was dispatched:**
- Bongaerts, D., Kang, X., & van Dijk, M. (2020), "Conditional Volatility
  Targeting," *Financial Analysts Journal* 76(4) -- conventional continuous
  volatility targeting can fail to improve, and can deepen drawdowns;
  re-sizing only in the volatility EXTREMES (high or low) and holding
  notional steady otherwise improves Sharpe and cuts tails at low turnover.
  Already this project's own validated mechanism (`kelly_regime_v3`/`v4`,
  promoted R-46/R-89); reused VERBATIM here, not retuned, as the
  conservative branch's entire mechanism.
- Baur, D. G., & Dimpfl, T. (2018), "Asymmetric volatility in
  cryptocurrencies," *Economics Letters* 173 -- BTC has an INVERSE leverage
  effect (positive shocks raise volatility more than negative ones), the
  asset-class fact that makes extremes-only targeting bite here rather than
  merely avoid continuous-targeting's textbook failure mode. Already this
  project's own v3/v4 citation; reused for the same reason it applies to any
  BTC-denominated position, including `replicator_book`'s.
- Kelly, J. L. (1956), "A New Interpretation of Information Rate," *Bell
  System Technical Journal* 35(4); Breiman, L. (1961), "Optimal Gambling
  Systems for Favorable Games," *Proc. 4th Berkeley Symposium* -- the
  log-optimal bet maximizes long-run growth. Already this project's
  foundational citation for the whole `kelly_regime` family and
  `universal_kelly`; reused here for the novel branch's per-species sizing.
- MacLean, L. C., Thorp, E. O., & Ziemba, W. T. (2010), "Long-Term Capital
  Growth: The Good and Bad Properties of the Kelly Criterion," *Quantitative
  Finance* 10(7) -- full Kelly is fragile to estimation error; a FRACTIONAL
  Kelly bet trades a controlled amount of growth rate for a large reduction
  in variance of terminal wealth. Already this project's citation for
  `kelly_regime`'s own fractional application; reused for the novel branch's
  per-species fraction, capped well below full Kelly for the same reason.
- Whitrow, C. (2007), "Algorithms for Optimal Allocation of Bets on Many
  Simultaneous Games," *Journal of the Royal Statistical Society: Series C
  (Applied Statistics)* 56(5), 607-623 -- extends Kelly staking from one bet
  to SEVERAL simultaneous, possibly correlated bets, each with its own
  estimated edge and variance, solved via the same growth-maximization
  logic applied jointly. `replicator_book`'s five species are exactly this
  setting (five simultaneous directional bets on the same underlying,
  blended by replicator weight rather than by a jointly-solved Kelly
  vector) and no round on this project has posed the sizing problem this
  way before -- the novel branch's whole mechanism, and the reason it is
  structurally distinct from every single-signal Kelly/CVaR resizing this
  project has already tried.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
- Conservative: `replicator_book`'s already-lagging fitness signal (L-10's
  own finding) means the blended position can still be pointed the wrong
  way when a regime turns. Bongaerts-Kang-van Dijk's own machinery can only
  ever change HOW MUCH is held, never the sign, and its high-volatility
  branch can request up to `max_leverage=2.0x` (double `replicator_book`'s
  own current 1.0x hard cap) -- if the blended signal is systematically
  late, a volatility-conditioned overlay that is willing to size UP in a
  calm-turned-volatile transition could amplify a stale, wrong-signed bet
  rather than protect it. This is a real, named risk, not a rhetorical one:
  it is exactly the failure mode that would produce a WORSE, not merely
  unchanged, drawdown.
- Novel: fitness and pnl at 5-minute granularity are dominated by fee drag
  and noise (the same reason this project's `hedge_experts`/`game_council`
  and 20+ other predictors lose to fees) -- if a species' trailing pnl
  variance is mostly noise rather than a stable edge/variance ratio, its
  estimated Kelly fraction will itself be noisy, causing the per-species
  weight fed into the replicator blend to whipsaw and raise turnover without
  improving the signal. The fractional cap and EWM smoothing below are
  named defenses against this, but the branch is pre-registered to report
  turnover honestly regardless of the Sharpe outcome.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither may edit it, so both are measured by
identical machinery, the r89-r147 convention. Nothing here reads a bar at
or after OOS_START (2023-01-01); `compare()` asserts this explicitly for
every slice it runs.
"""

from __future__ import annotations

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

# replicator_book's own shipped constants (do not change: the control must
# be replicator_book, not a re-parameterisation of it). Verified against
# src/tradebot/strategies/replicator_book.py.
R_BETA = 5.0
R_FITNESS_HALFLIFE = 1152
R_DEADBAND = 0.10
R_SHARE_CAP = 0.5
R_SHARE_FLOOR = 0.02
R_SCALE = 0.75
R_SPECIES_FEE = 0.0005

# kelly_regime_v3/v4's own shipped conditional-vol-target constants,
# reproduced byte-for-byte from src/tradebot/strategies/kelly_regime_v3.py.
# UNTOUCHED by the conservative branch -- reused verbatim, not retuned.
V4_TARGET_VOL = 0.55
V4_MAX_LEVERAGE = 2.0
V4_VOL_SPAN = 8 * BARS_PER_DAY
V4_ANCHOR_SPAN_DAYS = 180
V4_HIGH_IN, V4_HIGH_OUT = 1.70, 1.20
V4_LOW_IN, V4_LOW_OUT = 0.55, 0.85


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
# replicator_book's own construction, reproduced EXACTLY (species signals,
# fitness, replicator/logit weight update) so it can be called with a
# candidate's own species-signal matrix or scale path substituted in.
# ==================================================================

def species_signals(df: pd.DataFrame) -> np.ndarray:
    """The five species' raw signals in [-1, 1], reproduced verbatim from
    ReplicatorBook.prepare -- untouched by either branch."""
    close = df["close"]
    log_close = np.log(close)
    r = log_close.diff()
    sig1 = r.ewm(span=288, min_periods=250).std()

    s_tf_f = np.sign(close.ewm(span=48, adjust=False).mean()
                     - close.ewm(span=192, adjust=False).mean())
    s_tf_s = np.sign(close.ewm(span=288, adjust=False).mean()
                     - close.ewm(span=1152, adjust=False).mean())

    zf = (log_close - log_close.ewm(span=288, adjust=False).mean()) \
        / (sig1 * np.sqrt(288.0))
    zs = (log_close - log_close.ewm(span=2016, adjust=False).mean()) \
        / (sig1 * np.sqrt(2016.0))
    s_fun_f = -np.clip(zf / 2.0, -1.0, 1.0)
    s_fun_s = -np.clip(zs / 2.0, -1.0, 1.0)

    n = len(df)
    return np.column_stack([
        np.nan_to_num(s_tf_f.to_numpy(), nan=0.0),
        np.nan_to_num(s_tf_s.to_numpy(), nan=0.0),
        np.nan_to_num(s_fun_f.to_numpy(), nan=0.0),
        np.nan_to_num(s_fun_s.to_numpy(), nan=0.0),
        np.zeros(n),  # cash species
    ])


def species_fitness(df: pd.DataFrame, sig: np.ndarray,
                    species_fee: float = R_SPECIES_FEE,
                    fitness_halflife: int = R_FITNESS_HALFLIFE) -> np.ndarray:
    """Per-species net pnl and EWMA fitness, reproduced verbatim."""
    close = df["close"]
    r = np.log(close).diff()
    r_a = np.nan_to_num(r.to_numpy(), nan=0.0)
    sig_prev = np.empty_like(sig)
    sig_prev[0] = 0.0
    sig_prev[1:] = sig[:-1]
    pnl = sig_prev * r_a[:, None] - species_fee * np.abs(sig - sig_prev)
    alpha = 1.0 / float(fitness_halflife)
    fit = (pd.DataFrame(np.vstack([np.zeros((1, sig.shape[1])), pnl]))
           .ewm(alpha=alpha, adjust=False).mean()
           .to_numpy()[1:])
    return fit, pnl


def replicator_weights(fit: np.ndarray, beta: float = R_BETA,
                       cap: float = R_SHARE_CAP, floor: float = R_SHARE_FLOOR) -> np.ndarray:
    """The replicator/logit weight-update loop, reproduced verbatim.
    Returns the full (n, k) weight path (needed by the novel branch to
    blend a per-species-scaled signal instead of the raw one)."""
    n, k = fit.shape
    w = np.full(k, 1.0 / k)
    out = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        f_i = fit[i]
        w = w * np.exp(beta * (f_i - float(w @ f_i)))
        w /= w.sum()
        np.minimum(w, cap, out=w)
        w /= w.sum()
        np.maximum(w, floor, out=w)
        w /= w.sum()
        out[i] = w
    return out


def apply_deadband(desired: np.ndarray, deadband: float = R_DEADBAND) -> np.ndarray:
    """replicator_book's own re-target deadband, applied to a desired-exposure path.
    Clamps to +/-1x, exactly as the registered class does -- correct for the
    control and for the novel branch (whose signal magnitude is already
    bounded to +/-1 before the 0.75 scale, so this clamp never binds)."""
    target = np.zeros(len(desired))
    pos = 0.0
    for i, d in enumerate(desired):
        if abs(d - pos) > deadband:
            pos = min(1.0, max(-1.0, float(d)))
        target[i] = pos
    return target


def apply_deadband_leveraged(desired: np.ndarray, deadband: float = R_DEADBAND,
                             max_leverage: float = V4_MAX_LEVERAGE) -> np.ndarray:
    """Same re-target deadband, WITHOUT replicator_book's own +/-1x re-clamp.
    The conservative branch's scale is v4's conditional volatility target,
    which by design ranges up to `max_leverage` (2.0) in a low-volatility
    breakout -- clamping the result back to +/-1 would silently discard
    exactly the half of Bongaerts-Kang-van Dijk's mechanism that levers UP
    in calm regimes, leaving only the de-risking half in a high-vol
    breakout. Clamped instead to +/-max_leverage, matching kelly_regime_v4's
    own ceiling (never left unbounded)."""
    target = np.zeros(len(desired))
    pos = 0.0
    for i, d in enumerate(desired):
        if abs(d - pos) > deadband:
            pos = min(max_leverage, max(-max_leverage, float(d)))
        target[i] = pos
    return target


def replicator_target(df: pd.DataFrame, scale: float = R_SCALE,
                      deadband: float = R_DEADBAND, beta: float = R_BETA,
                      cap: float = R_SHARE_CAP, floor: float = R_SHARE_FLOOR,
                      fitness_halflife: int = R_FITNESS_HALFLIFE,
                      species_fee: float = R_SPECIES_FEE) -> np.ndarray:
    """replicator_book's complete, final target path -- the control,
    reproduced from the registered strategy's own defaults."""
    sig = species_signals(df)
    fit, _pnl = species_fitness(df, sig, species_fee, fitness_halflife)
    w = replicator_weights(fit, beta, cap, floor)
    raw = np.einsum("ij,ij->i", w, sig) * scale
    return apply_deadband(raw, deadband)


# ================================================================== (2)
# kelly_regime_v3/v4's conditional volatility-target machinery, reproduced
# byte-for-byte from experiments/r146_shared.py / kelly_regime_v3.py.
# UNTOUCHED by this round -- the conservative branch feeds it the SAME
# BTC/ETH symmetric realized volatility v4 itself uses, not retuned in any
# way.
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


def v4_scale(df: pd.DataFrame) -> np.ndarray:
    """kelly_regime_v3/v4's conditional volatility-target scale factor,
    reproduced exactly. Reused verbatim by the conservative branch below."""
    return conditional_target_scale(v4_symmetric_vol(df))


# ================================================================== (3)
# CONSERVATIVE candidate: replace replicator_book's fixed `scale=0.75`
# constant with v4's own (unretuned) conditional-volatility-target path.
# Species, fitness, replicator weights and the deadband are byte-identical
# to the control.
# ==================================================================

def conservative_target(df: pd.DataFrame, deadband: float = R_DEADBAND,
                        beta: float = R_BETA, cap: float = R_SHARE_CAP,
                        floor: float = R_SHARE_FLOOR,
                        fitness_halflife: int = R_FITNESS_HALFLIFE,
                        species_fee: float = R_SPECIES_FEE,
                        target_vol: float = V4_TARGET_VOL,
                        max_leverage: float = V4_MAX_LEVERAGE) -> np.ndarray:
    sig = species_signals(df)
    fit, _pnl = species_fitness(df, sig, species_fee, fitness_halflife)
    w = replicator_weights(fit, beta, cap, floor)
    blended = np.einsum("ij,ij->i", w, sig)
    scale = conditional_target_scale(v4_symmetric_vol(df), target_vol=target_vol,
                                     max_leverage=max_leverage)
    raw = blended * scale
    return apply_deadband_leveraged(raw, deadband, max_leverage)


# ================================================================== (4)
# NOVEL candidate: per-species fractional-Kelly sizing (Whitrow 2007) BEFORE
# the replicator blend. Each species' raw +/-1 signal is pre-multiplied by
# its own causally-estimated Kelly fraction (trailing mean/variance of its
# OWN fee-adjusted pnl, same EWM halflife as fitness), capped well below
# full Kelly. The replicator weights, fitness measure, deadband and the
# portfolio-level scale=0.75 constant are UNCHANGED -- only the per-species
# signal magnitude fed into the blend differs from the control.
# ==================================================================

def per_species_kelly_fraction(pnl: np.ndarray, fitness_halflife: int = R_FITNESS_HALFLIFE,
                               kelly_cap: float = 1.5, min_var: float = 1e-10) -> np.ndarray:
    """Causal, EWM-estimated per-species Kelly fraction f_k = mu_k / var_k
    (Kelly 1956 single-bet form, applied independently per simultaneous game
    per Whitrow 2007), fractionalized by clipping to [0, kelly_cap] rather
    than the raw (unbounded, estimation-error-fragile) full-Kelly value
    (MacLean, Thorp & Ziemba 2010). `mu_k`/`var_k` are the EWM mean/variance
    of species k's OWN net pnl (same halflife as the fitness measure, so no
    new free timescale is introduced), shifted by one bar so bar i's
    fraction uses only pnl strictly before i."""
    n, k = pnl.shape
    alpha = 1.0 / float(fitness_halflife)
    pnl_df = pd.DataFrame(pnl)
    mu = pnl_df.ewm(alpha=alpha, adjust=False).mean().shift(1).to_numpy()
    var = pnl_df.ewm(alpha=alpha, adjust=False).var(bias=False).shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(var > min_var, mu / var, 0.0)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(f, 0.0, kelly_cap)


def novel_target(df: pd.DataFrame, scale: float = R_SCALE, deadband: float = R_DEADBAND,
                 beta: float = R_BETA, cap: float = R_SHARE_CAP, floor: float = R_SHARE_FLOOR,
                 fitness_halflife: int = R_FITNESS_HALFLIFE, species_fee: float = R_SPECIES_FEE,
                 kelly_cap: float = 1.5) -> np.ndarray:
    sig = species_signals(df)
    fit, pnl = species_fitness(df, sig, species_fee, fitness_halflife)
    kelly_f = per_species_kelly_fraction(pnl, fitness_halflife, kelly_cap)
    sig_scaled = np.clip(sig * kelly_f, -1.0, 1.0)
    w = replicator_weights(fit, beta, cap, floor)
    raw = np.einsum("ij,ij->i", w, sig_scaled) * scale
    return apply_deadband(raw, deadband)


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
# replicator_book's own control, on BOTH markets. Never touches OOS_START.
# Structurally identical to r146_shared.py's compare().
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

    Two execution conventions, and getting this wrong silently changes what
    is actually measured:

    - ``use_notional=False`` (default): ``ctx.order_target(t)`` -- ``t`` is a
      fraction of the MARKET's own max leverage (the broker clamps it to
      [lo, 1.0]; 1.0 = fully using whatever leverage the market allows).
      This is `replicator_book`'s own registered convention (verified
      against ``src/tradebot/strategies/replicator_book.py``'s ``on_bar``)
      and is correct for the control and for the novel branch, whose
      signal magnitude stays bounded to +/-1 by construction exactly as
      the control's does.
    - ``use_notional=True``: ``ctx.order_notional(t)`` -- ``t`` is a
      fraction of EQUITY, independent of leverage (``kelly_regime_v4``'s
      own convention). Required for the conservative branch: its scale is
      v4's conditional volatility target, an ABSOLUTE leverage multiple
      (up to ``max_leverage=2.0``), not a fraction of whatever the market
      allows. Feeding a value like 2.0 into plain ``order_target`` on a 5x
      futures market would silently clamp to 1.0 = fully using the 5x
      leverage (10x the intended exposure) -- caught by this round's own
      Step-0 diligence before either branch was dispatched.
    """

    name = "r148_control"
    warmup = 8 * BARS_PER_DAY + V4_ANCHOR_SPAN_DAYS * BARS_PER_DAY

    def __init__(self, build_target, name: str = "r148_control",
                warmup: int | None = None, use_notional: bool = False) -> None:
        self._build = build_target
        self.name = name
        self.use_notional = use_notional
        if warmup is not None:
            self.warmup = warmup

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["target"] = np.asarray(self._build(df), dtype=float)
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            if self.use_notional:
                ctx.order_notional(t)
            else:
                ctx.order_target(t)


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
           include_eth: bool = True, seed: int = 0,
           use_notional: bool = False) -> list[dict]:
    """``use_notional=True`` for the conservative branch (its scale is an
    absolute leverage multiple, kelly_regime_v4's convention); leave False
    (the default) for the novel branch and any candidate whose signal
    magnitude stays bounded to +/-1 like the control's -- see
    ``TargetStrategy``'s own docstring for why this is not cosmetic."""
    if control_build is None:
        control_build = replicator_target
    if btc is None:
        btc = load_btc()
    assert_no_holdout(btc, "compare(): btc")
    if include_eth and eth is None:
        eth = load_eth()
    if include_eth:
        assert_no_holdout(eth, "compare(): eth")

    cand = TargetStrategy(candidate_build, name=f"r148_{label}", use_notional=use_notional)
    ctrl = TargetStrategy(control_build, name="replicator_book")

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
# PROMOTION BAR (identical shape to R-89-R-147's standard bar, adapted to
# this round's control being `replicator_book` rather than `kelly_regime_v4`):
#
#  A2 (Step-0 non-inertness kill switch): R^2 of the candidate's own final
#     target path against replicator_book's unmodified `replicator_target`,
#     computed on inner-train. If R^2 > 0.98, STOP -- the candidate is a
#     disguised no-op, not a genuinely different sizing mechanism, and no
#     Sharpe number is read past this point.
#  B1: bootstrap paired difference in total log-growth, inner-validation,
#     BOTH markets: ΔSharpe > +0.2 OR the 95% bootstrap interval excludes
#     zero.
#  B2 (diagnostic, not gating): exposure_ratio / vol_ratio reported for
#     every cell. NOT a gate here (the whole point of both branches is to
#     change how much risk is taken, so risk-matching the control is not
#     the right question) but must be reported per R-33's standing rule so
#     a return improvement is never silently an exposure-level artifact
#     read as a mechanism. `exposure_ratio` is NOT directly comparable for
#     the conservative branch (its `target` column is in absolute-leverage
#     units, `order_notional`-executed; the control's is in
#     fraction-of-max-leverage units, `order_target`-executed -- see
#     `TargetStrategy`'s own docstring) -- read `vol_ratio` (computed from
#     realized daily equity returns, convention-independent) as the
#     trustworthy risk comparison for that branch, and disclose the
#     `exposure_ratio` caveat explicitly rather than silently comparing
#     mismatched units.
#  B3: plateau -- conservative reuses v4's own constants completely
#     unretuned (no free parameter to sweep; B3 here is a +/-20% sensitivity
#     probe on target_vol alone, checking the sign of B1 does not flip);
#     novel sweeps kelly_cap in {1.0, 1.5, 2.0} and checks sign stability.
#     A single winning cell with no support around it does not clear this
#     bar.
#  B4: ETH same-sign falsification -- the candidate's ΔSharpe (or bootstrap
#     direction) on the ETH replication slice must agree in SIGN with the
#     BTC inner-validation result on at least one market.
#  B5: 0.40% taker-fee-tier re-run on spot (fee_at(SPOT, 0.004)) -- the
#     edge, if any, must not require the 0.10% fee tier to exist.
#
# Promote only if A2 does not trip AND B1 passes on >=1 market AND B4
# passes AND B5's edge (if B1 passed) survives in sign. Anything else is
# NEGATIVE. This is the SAME bar both branches must clear; neither may
# weaken it after seeing a number.
# ============================================================


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=60_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(148)
    innov = rng.normal(0, 0.0006, len(idx))
    jump_idx = rng.choice(len(idx), size=15, replace=False)
    innov[jump_idx] += rng.choice([-1, 1], size=15) * rng.uniform(0.01, 0.03, size=15)
    # A genuine volatility-regime shift (5x std for one stretch) so the
    # conditional-vol-target's high-volatility state is actually exercised
    # on synthetic data, rather than only ever sitting in the steady band.
    innov[30_000:35_000] *= 5.0
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": 1.0}, index=idx)

    # (1) control self-consistency: replicator_target reproduces the
    # registered ReplicatorBook class exactly.
    from tradebot.strategies.replicator_book import ReplicatorBook
    strat = ReplicatorBook()
    prepared = strat.prepare(df.copy())
    assert np.allclose(prepared["target"].to_numpy(), replicator_target(df), atol=1e-9), \
        "replicator_target() diverges from the registered ReplicatorBook class"

    # (2) conservative candidate is well-formed and genuinely different.
    # It is deliberately NOT clamped to +/-1x (see apply_deadband_leveraged's
    # own docstring): v4's conditional scale legitimately ranges up to
    # max_leverage=2.0 in a low-volatility breakout, and re-clamping to
    # +/-1 would silently discard that half of the mechanism.
    cons = conservative_target(df)
    assert np.all(cons >= -V4_MAX_LEVERAGE - 1e-9) and np.all(cons <= V4_MAX_LEVERAGE + 1e-9)
    assert not np.allclose(cons, replicator_target(df))
    assert np.nanmax(np.abs(cons)) > 1.0 + 1e-6, \
        "conservative branch never exceeds 1x -- conditional scale is not binding on synthetic data"

    # (3) novel candidate is well-formed, bounded, and genuinely different.
    nov = novel_target(df)
    assert np.all(nov >= -1.0 - 1e-9) and np.all(nov <= 1.0 + 1e-9)
    assert not np.allclose(nov, replicator_target(df))
    assert not np.allclose(nov, cons)

    # (4) per-species Kelly fraction is causal, bounded, and non-degenerate.
    sig = species_signals(df)
    fit, pnl = species_fitness(df, sig)
    kf = per_species_kelly_fraction(pnl)
    assert kf.shape == sig.shape
    assert np.all(kf >= 0.0) and np.all(kf <= 1.5 + 1e-9)
    assert np.nanstd(kf) > 0, "per-species Kelly fraction is degenerate (constant) on synthetic data"

    # (5) causal truncation probes -- no candidate may peek at future bars.
    assert causal_truncation_probe_series(replicator_target, df)
    assert causal_truncation_probe_series(conservative_target, df)
    assert causal_truncation_probe_series(novel_target, df)
    assert causal_truncation_probe_series(v4_scale, df)

    # (6) r_squared sanity.
    assert abs(r_squared(cons, cons) - 1.0) < 1e-9
    assert r_squared(cons, rng.normal(0, 1, len(cons))) < 0.5

    # (7) execution-convention check, through the REAL engine, on FUTURES
    # (leverage=5.0) -- this is what R-148's own Step-0 diligence caught:
    # feeding a >1 conservative target into plain order_target would
    # silently clamp to "fully using 5x leverage" (10x the intended
    # exposure). Force a config where the control sits pinned at target=1.0
    # (a plain always-long probe reproducing the CONTROL's own on_bar
    # convention) and confirm the realized position notional differs by
    # ~5x between order_target and order_notional execution, as the
    # broker's own documented semantics predict.
    fut = MarketSpec.futures(leverage=5.0)

    class _Probe(Strategy):
        name = "r148_probe"
        warmup = 10

        def __init__(self, use_notional: bool) -> None:
            self.use_notional = use_notional

        def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
            return frame

        def on_bar(self, ctx: Context) -> None:
            if ctx.i == self.warmup:
                if self.use_notional:
                    ctx.order_notional(1.0)
                else:
                    ctx.order_target(1.0)

    small = df.iloc[:2_000]
    res_target = run_period(_Probe(False), small, None, None, market=fut, start_balance=1_000.0)
    res_notional = run_period(_Probe(True), small, None, None, market=fut, start_balance=1_000.0)
    qty_target = abs(res_target.fills[0].qty) if res_target.fills else 0.0
    qty_notional = abs(res_notional.fills[0].qty) if res_notional.fills else 0.0
    assert qty_target > 0 and qty_notional > 0, "probe fired no fill -- test is not exercising the path"
    ratio = qty_target / qty_notional
    assert 4.5 < ratio < 5.5, (
        f"order_target(1.0) vs order_notional(1.0) on 5x futures should differ ~5x, got {ratio:.2f}x "
        "-- TargetStrategy's use_notional dispatch is not doing what this round's design assumes")


_self_test()
