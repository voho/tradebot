"""Shared, read-only utilities for the R-82 round (08-21).

Idea in one sentence: `kelly_regime_v4`'s regime vote is a fixed-window
EMA-crossing heuristic (three latched anchors) with no probabilistic
notion of "how confident are we that a regime break just happened" -- this
round replaces that heuristic, in whole or in part, with Bayesian Online
Changepoint Detection (BOCPD; Adams & MacKay 2007, arXiv:0710.3742), which
maintains a formal posterior over the run length (days since the last
regime break) using a Normal-Inverse-Gamma conjugate observation model and
a constant-hazard changepoint prior, updated online/causally by the
message-passing recursion in their Section 3. Financial application
precedent: Bianchi, Prata & Vecchio (2025, "Online Learning of Order Flow
and Market Impact with Bayesian Change-Point Detection Methods", Applied
Mathematical Finance / arXiv:2307.02375), who use the same machinery for
online regime identification in order-flow/market-impact models and report
it materially outperforming static/rolling-window alternatives.

Which constraint this attacks: **ERR** (no error control anywhere in this
project's signal path -- v4's vote is a deterministic 0/1 latch with no
notion of its own uncertainty) and **N-approx-3** (the standing diagnosis
is that this project's effective sample size is a handful of regime
events; BOCPD is a formal, principled ESTIMATOR of how many independent
regime segments actually exist and where they are, rather than a fixed
20/40/80-day heuristic asserted without any error model). This is
explicitly NOT a tenth INFO-axis signal: it consumes no data beyond the
committed OHLCV close series v4 itself already uses -- same information,
a formally different (and error-aware) way of extracting a regime
estimate from it. R-81's own backlog re-ranking named exactly this
combination as the remaining open door: "a genuinely new kind of
information or error-control mechanism this project has not yet checked
in any form."

Not a duplicate of:
- R-34 (`harsanyi_crowd` posterior as a SIZE input): a game-theoretic
  minority/majority-game posterior over OTHER TRADERS' behaviour, unrelated
  statistical machinery, no notion of a changepoint or run length.
- R-38 (Busseti/Ryu/Boyd 2016 risk-constrained Kelly, conservative cap +
  full sizing-formula replacement): a convex-optimisation BET-SIZING
  framework taking a return/vol estimate as given; it does not estimate
  regime structure at all, and both of its branches failed the identical
  ETH falsification test for a *drift-estimate* reason (systematically
  under-holding through a trend) that this round's mechanism does not
  share, since BOCPD's E[mu] is a POSTERIOR MEAN over discrete regime
  segments, not a continuously-updated point drift estimate.
- R-80 (causal meta-labeling logistic model, ERR axis): a discriminative
  classifier trained on hand-engineered features of v4's own trailing hit
  rate. BOCPD is a generative single-series segmentation model with no
  trained weights and no features beyond the log-return series itself.
  R-80's hard-won lesson is reused directly: any confirming vote fed into
  `confirming_vote_frac` must be DISCRETE (0/1), not continuous, so the
  formula retains its ability to reach exactly flat / exactly full (fixed
  by R-81 and preserved here -- see `confirming_vote_frac` below).
- R-53/R-55/R-73/R-74/R-79/R-81 (the nine prior INFO-axis rounds): every
  one of them fed in a NEW external or timestamp-derived data channel and
  found it lagged the anchor gate. This round introduces no new data
  channel at all -- it is a different ESTIMATOR of regime state from the
  same OHLCV close series v4 already reads, so the "does external info
  lead price" question those nine rounds asked does not apply here. The
  applicable question, and this round's Step-A gate, is instead: does a
  formal changepoint estimator detect KNOWN, DATED, historical regime
  breaks with shorter lag than v4's own fixed-window anchor heuristic?
- R-62 (factored v4 into vote x scale, found the vote carries the whole
  matched-exposure drawdown signature and the scale factor carries none of
  it): motivates keeping this round's SCALE / conditional-vol-targeting
  factor untouched in both branches and confining the change to the
  DIRECTION/vote side, which is where R-62 showed the signature lives.

This module is read-only utility, written by the operator before dispatch
(same convention as r79_shared.py/r80_shared.py/r81_shared.py). Neither
branch edits it. Contains: (1) a byte-for-byte duplicate of
`kelly_regime_v4`'s 3-anchor vote construction (duplicated, not imported --
R-54/R-55's convention); (2) the R-53/R-55 confirming-vote combination
rule; (3) the BOCPD engine itself, run on DAILY-resampled log returns (not
raw 5-minute bars) -- a deliberate, disclosed design choice: v4's own
anchors already operate on a 20-80 CALENDAR-DAY horizon, so segmenting at
daily rather than 5-minute granularity matches the horizon the mechanism
is meant to describe and cuts the recursion from ~1.01M steps to ~3,510,
which is the difference between seconds and a build that cannot iterate;
(4) the causal daily-to-bar alignment (reuses `tradebot.data.
align_onchain_causal` -- a full calendar-day shift before any bar may see
a day's posterior, the same causal contract this project already uses for
every other daily-cadence signal, not reinvented here); (5) the dated
stress-episode table both branches' Step-A detection-lag gates test
against; (6) a block-bootstrap null generator for that gate; (7) shared
date constants and the causality truncation probe.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import align_onchain_causal  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# kelly_regime_v4's own anchor ladder and band -- duplicated, not imported.
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

# Inner split per docs/ROUTINE.md step 3. Holdout (>= OOS_START) is never
# read by either branch in this step.
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# Dated, PUBLICLY KNOWN historical BTC regime transitions, restricted to
# before OOS_START (six, giving more power than R-81's three since this
# signal needs no external feed with its own later coverage start). Onset
# dates are the conventional dates cited for each event in the crypto
# market-history literature/press, fixed here before any BOCPD number was
# computed.
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

# ----------------------------------------------------------------- v4 vote


def anchor_votes(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                  band: float = V4_BAND) -> list[pd.Series]:
    """The three latched 0/1 anchor votes `kelly_regime`/`_v3`/`_v4` use internally.

    Causal: row i depends only on rows <= i (rolling mean + ffill/latch).
    """
    close = df["close"]
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=df.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return votes


def anchor_majority(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                     band: float = V4_BAND) -> pd.Series:
    """`frac` = mean of the three anchor votes, in {0, 1/3, 2/3, 1} -- v4's
    own gate, exactly, for use as the Step-A detection-lag comparison
    baseline."""
    votes = anchor_votes(df, horizons, band)
    return sum(votes) / len(votes)


def confirming_vote_frac(anchor_sum: np.ndarray, meta_vote: np.ndarray,
                          weight: float) -> np.ndarray:
    """R-53/R-55's combination rule.

    ``frac = (anchor_sum + weight * meta_vote) / (3 + weight)``

    ``anchor_sum`` in [0, 3], ``meta_vote`` in {0, 1} (per R-80/R-81's
    lesson: keep it DISCRETE so the formula can still reach exactly
    flat/exactly full). ``weight == 0`` recovers `kelly_regime_v4` exactly
    -- the required identity-recovery check every confirming-vote round
    has run.
    """
    anchor_sum = np.asarray(anchor_sum, dtype=float)
    meta_vote = np.asarray(meta_vote, dtype=float)
    return (anchor_sum + weight * meta_vote) / (3.0 + weight)


# ------------------------------------------------------------------ BOCPD
#
# Adams & MacKay (2007), Normal-Inverse-Gamma conjugate observation model
# (unknown mean AND variance per regime segment), constant hazard. Prior
# hyperparameters (mu0, kappa0, alpha0, beta0) are FIXED a priori --
# beta0/(alpha0-1) = 0.0032/1.0 implies a prior daily-vol belief of
# ~5.7%/day, a round, weakly-informative choice for BTC and not fit to any
# observed number. hazard_lambda (mean prior run length in days) defaults
# to 250 -- roughly 3x v4's own slowest 80-day anchor, i.e. "a regime
# typically persists the better part of a year", again fixed before any
# posterior was computed, not tuned to it.

MU0 = 0.0
KAPPA0 = 1.0
ALPHA0 = 2.0
BETA0 = 0.0032
DEFAULT_HAZARD_LAMBDA = 250.0
K_SHORT_DAYS = 5   # "a changepoint likely just happened" horizon, fixed a priori
PRUNE_THRESH = 1e-6

_lgamma = np.vectorize(math.lgamma)


def _student_t_pdf(x: float, df: np.ndarray, loc: np.ndarray, scale2: np.ndarray) -> np.ndarray:
    scale2 = np.maximum(scale2, 1e-12)
    z2 = (x - loc) ** 2 / scale2
    log_pdf = (_lgamma((df + 1.0) / 2.0) - _lgamma(df / 2.0)
               - 0.5 * np.log(df * np.pi * scale2)
               - (df + 1.0) / 2.0 * np.log1p(z2 / df))
    return np.exp(log_pdf)


def bocpd_daily(returns: np.ndarray, hazard_lambda: float = DEFAULT_HAZARD_LAMBDA,
                 mu0: float = MU0, kappa0: float = KAPPA0,
                 alpha0: float = ALPHA0, beta0: float = BETA0,
                 prune_thresh: float = PRUNE_THRESH) -> dict:
    """Online BOCPD over a 1-D array of DAILY log returns. Causal: day t's
    outputs use only ``returns[:t+1]``, enforced by construction (each
    iteration only ever reads ``returns[t]`` and the state carried from
    the previous iteration).

    Returns a dict of length-``len(returns)`` arrays:

    - ``map_run_length``: the MAP (most probable) run length in days,
      i.e. the modal estimate of "days since the last regime break".
    - ``e_mu``: model-averaged posterior mean of the CURRENT regime's
      mean daily log return, E[mu_t | x_1:t] = sum_r P(r_t=r|x_1:t) * mu_r
      -- a Bayesian-model-averaged drift estimate over the whole
      run-length posterior, not a single point estimate.
    - ``p_recent_cp``: P(run_length_t <= K_SHORT_DAYS | x_1:t) -- posterior
      probability that a changepoint happened within the last
      ``K_SHORT_DAYS`` days, i.e. "how likely is it we are in a freshly-
      started regime right now".
    """
    n = len(returns)
    h = 1.0 / hazard_lambda

    post = np.array([1.0])
    rl = np.array([0])
    mu_n = np.array([mu0])
    kappa_n = np.array([kappa0])
    alpha_n = np.array([alpha0])
    beta_n = np.array([beta0])

    map_run_length = np.zeros(n, dtype=int)
    e_mu = np.zeros(n)
    p_recent_cp = np.zeros(n)

    for t in range(n):
        x = float(returns[t])
        df = 2.0 * alpha_n
        scale2 = beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n)
        pred = _student_t_pdf(x, df, mu_n, scale2)
        pred = np.where(np.isfinite(pred), pred, 0.0)

        growth = post * pred * (1.0 - h)
        cp = float(np.sum(post * pred * h))

        new_post_raw = np.concatenate(([cp], growth))
        total = float(new_post_raw.sum())
        if not np.isfinite(total) or total <= 0.0:
            new_post_raw = np.where(np.isfinite(new_post_raw) & (new_post_raw > 0),
                                     new_post_raw, 0.0)
            total = float(new_post_raw.sum())
            if total <= 0.0:
                # Degenerate step (e.g. x is NaN): fall back to a full reset,
                # never silently propagate a NaN posterior forward.
                new_post_raw = np.zeros_like(new_post_raw)
                new_post_raw[0] = 1.0
                total = 1.0
        new_post = new_post_raw / total

        new_rl = np.concatenate(([0], rl + 1))
        new_mu = np.concatenate(([mu0], (kappa_n * mu_n + x) / (kappa_n + 1.0)))
        new_kappa = np.concatenate(([kappa0], kappa_n + 1.0))
        new_alpha = np.concatenate(([alpha0], alpha_n + 0.5))
        new_beta = np.concatenate(
            ([beta0], beta_n + kappa_n * (x - mu_n) ** 2 / (2.0 * (kappa_n + 1.0))))

        keep = new_post >= prune_thresh
        keep[0] = True  # always keep the freshest (run-length-0) hypothesis
        post = new_post[keep]
        post = post / post.sum()
        rl, mu_n, kappa_n, alpha_n, beta_n = (
            new_rl[keep], new_mu[keep], new_kappa[keep], new_alpha[keep], new_beta[keep])

        map_run_length[t] = int(rl[int(np.argmax(post))])
        e_mu[t] = float(np.sum(post * mu_n))
        p_recent_cp[t] = float(post[rl <= K_SHORT_DAYS].sum())

    return dict(map_run_length=map_run_length, e_mu=e_mu, p_recent_cp=p_recent_cp)


def bocpd_daily_causal_signals(df: pd.DataFrame, hazard_lambda: float = DEFAULT_HAZARD_LAMBDA
                                ) -> pd.DataFrame:
    """Resample ``df["close"]`` to daily, run `bocpd_daily`, and align the
    result onto ``df``'s own 5-minute index with a full-calendar-day causal
    shift (`tradebot.data.align_onchain_causal` -- day D's posterior, which
    depends on day D's own close, only becomes visible to bars starting
    2026-01-02T00:00 UTC given a day dated 2026-01-01, matching every other
    daily-cadence signal already in this project). Returns a DataFrame
    indexed like ``df`` with columns ``bocpd_map_run_length``,
    ``bocpd_e_mu``, ``bocpd_p_recent_cp``.
    """
    daily_close = df["close"].resample("1D").last().dropna()
    daily_ret = np.log(daily_close).diff().dropna()
    out = bocpd_daily(daily_ret.to_numpy(), hazard_lambda=hazard_lambda)
    daily = pd.DataFrame(
        {"bocpd_map_run_length": out["map_run_length"],
         "bocpd_e_mu": out["e_mu"],
         "bocpd_p_recent_cp": out["p_recent_cp"]},
        index=daily_ret.index,
    )
    return align_onchain_causal(daily, df)


# --------------------------------------------------------- Step-A gate infra


def nearest_transition(series: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, direction: str = "down") -> pd.Timestamp | None:
    """Timestamp, within `window`, of the anchor-gate transition closest to
    `onset`. Duplicated from R-81's own gate file (self-contained, not
    imported, per this project's per-round shared-module convention)."""
    vals = series.reindex(window).to_numpy()
    changed = np.zeros(len(vals), dtype=bool)
    if direction == "down":
        changed[1:] = vals[1:] < vals[:-1]
    elif direction == "any":
        changed[1:] = vals[1:] != vals[:-1]
    else:
        raise ValueError(f"unknown direction {direction!r}")
    idx = np.where(changed)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def nearest_bocpd_detection(map_run_length: pd.Series, window: pd.DatetimeIndex,
                             onset: pd.Timestamp, k_short: int = K_SHORT_DAYS
                             ) -> pd.Timestamp | None:
    """Timestamp, within `window`, of the first bar where the MAP run
    length crosses down to `<= k_short` (a likely-recent changepoint),
    closest to `onset` -- the BOCPD analogue of `nearest_transition`."""
    vals = map_run_length.reindex(window).to_numpy()
    short = vals <= k_short
    cross = np.zeros(len(vals), dtype=bool)
    cross[1:] = short[1:] & ~short[:-1]
    cross[0] = bool(short[0])
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def episode_window(bars: pd.DataFrame, onset_str: str, window_days: int = 60
                    ) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    onset = pd.Timestamp(onset_str, tz="UTC")
    lo = onset - pd.Timedelta(days=window_days)
    hi = onset + pd.Timedelta(days=window_days)
    window = bars.index[(bars.index >= lo) & (bars.index <= hi)]
    return onset, window


def block_bootstrap_shifts(n_bars: int, block_days: int, n_draws: int, seed: int) -> list[np.ndarray]:
    """`n_draws` causal-safe circular block-shifts of a length-`n_bars`
    index, for a block-bootstrap null on the Step-A detection-lag gate."""
    rng = np.random.default_rng(seed)
    block = int(block_days * BARS_PER_DAY)
    draws = []
    for _ in range(n_draws):
        shift = int(rng.integers(block, n_bars - block)) if n_bars > 2 * block else int(rng.integers(1, n_bars))
        draws.append((np.arange(n_bars) + shift) % n_bars)
    return draws


def truncation_causality_probe(build_target_fn, df: pd.DataFrame,
                                check_at: int, shorter_by: int = 20_000) -> bool:
    """Standard truncation probe: does `target[check_at]` change if bars
    after it are dropped? Returns True if causal (identical both ways)."""
    full = build_target_fn(df)
    short = build_target_fn(df.iloc[:check_at + shorter_by].copy())
    a, b = full[check_at], short[check_at]
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, equal_nan=True))
