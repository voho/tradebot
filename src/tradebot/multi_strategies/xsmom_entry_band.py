"""Cross-sectional trend allocator with an asymmetric entry band.

R-63's cross-sectional trend score, sized by `kelly_regime_v4`'s own
conditional-volatility target, held under R-65's rank-buffer/hold-days
selection loop generalized by R-68 into two independent eligibility
thresholds -- and frozen at R-68's own selected winner, `ENTRY_ONLY,
delta=0.080` (`delta_in=0.080, delta_out=0.0`): a new entrant must clear a
raised bar, but an incumbent is never forced out early.

This is backlog **B-32**'s first registered candidate: the multi-asset
family five research rounds (R-63, R-65, R-67, R-68) built and refined
entirely inside ``experiments/`` because no registration path existed for a
strategy that needs to see more than one instrument's state at once. R-68's
own verdict on this configuration was NEGATIVE (`further_work=False`;
D1/D2 both failed, `docs/LEDGER.md` `### R-68`) -- it is registered here as
the correctness check for the registration path itself (B-32), reproducing
R-68's already-published numbers through the new infrastructure, not as a
promoted strategy. Nothing about that verdict changes by being registered.

PORTED, not imported, from three frozen experiment files -- ``experiments/``
is never imported from ``src/tradebot/`` in this repo (see
``multi_engine.py``'s module docstring) -- faithfully, parameter for
parameter, with the exact source cited at each block:

- the score, the basket return and the conditional volatility scale:
  ``experiments/r63_novel_xsmom_rank.py``, `cross_sectional_score` (lines
  136-153), `basket_log_returns` (156-166), `conditional_vol_scale`
  (169-204) -- all copied verbatim, no constant changed;
- the eligibility/selection loop and the sizing block:
  ``experiments/r68_conservative_band_decomposition.py``, `band_selection`
  (315-415) and `_size` (418-438) -- copied verbatim; `band_selection`
  generalizes R-65's/R-67's single `delta` into the independent
  `(delta_in, delta_out)` pair frozen below;
- the frozen parameters themselves: `k=1`, `buffer=0.05`, `hold_days=1`
  (``experiments/r67_shared.py`` `R65_K`/`R65_BUFFER`/`R65_HOLD_DAYS`,
  R-65's selected winner, unchanged since), `delta_in=0.080, delta_out=0.0`
  (``experiments/r68_conservative_band_decomposition.py``'s
  `FROZEN_SUBARM = "ENTRY_ONLY"`, `FROZEN_D = 0.080`, selected on W_VAL
  before any D-cell was computed, see that file's `cmd_select` block and
  `docs/LEDGER.md` `### R-68`), and the universe, `UNIVERSE_6` (R-68's own
  W_FULL6/U6 decision cell -- `experiments/r68_conservative_band_decomposition.py::cmd_run`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.multi_engine import UNIVERSE_6
from tradebot.multi_strategy import MultiAssetStrategy, register_multi_asset

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# The score's three lookback horizons, in days (r63_novel_xsmom_rank.py:103).
HORIZONS = (20, 40, 80)
# > the 80-day anchor, so the first evaluated bar is warm
# (r63_novel_xsmom_rank.py:104, WARM_DAYS).
WARM_DAYS = 91

# kelly_regime_v4's shipped sizing constants, untouched
# (r63_novel_xsmom_rank.py:108-114).
TARGET_VOL = 0.55
MAX_LEVERAGE = 2.0
VOL_SPAN = 8 * BARS_PER_DAY
ANCHOR_SPAN_DAYS = 180
HIGH_IN, HIGH_OUT = 1.70, 1.20
LOW_IN, LOW_OUT = 0.55, 0.85
# The vol-scale latch's own deadband (on desired TOTAL notional), distinct
# from the engine's execution deadband on TRADED notional
# (r63_novel_xsmom_rank.py:114).
VOL_LATCH_DEADBAND = 0.10

# R-65's frozen winner, inherited unchanged by R-67 and R-68
# (r67_shared.py:278-280).
K_FIXED = 1
BUFFER_FIXED = 0.05
HOLD_FIXED = 1

# R-68's selected winner: ENTRY_ONLY at d=0.080, i.e. delta_in=d, delta_out=0
# (r68_conservative_band_decomposition.py:220-224, 303-304).
DELTA_IN = 0.080
DELTA_OUT = 0.0


# --------------------------------------------------------------- the score


def cross_sectional_score(aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """`score_i(t) = mean_h (close_i(t) / anchor_{i,h}(t) - 1)`.

    Rolling means only: row t uses rows <= t and nothing else. No
    standardization, no ranking over time, no whole-series statistic of any
    kind. Ported verbatim from `experiments/r63_novel_xsmom_rank.py:136-153`.
    """
    cols = {}
    for t, df in aligned.items():
        close = df["close"]
        acc = None
        for h in HORIZONS:
            anchor = close.rolling(int(h * BARS_PER_DAY)).mean()
            term = close / anchor - 1.0
            acc = term if acc is None else acc + term
        cols[t] = acc / len(HORIZONS)
    return pd.DataFrame(cols, index=next(iter(aligned.values())).index)


def basket_log_returns(aligned: dict[str, pd.DataFrame]) -> pd.Series:
    """Log return series of the EQUAL-WEIGHT ALL-N basket -- what the
    portfolio volatility target is driven from, not the top-k basket (that
    would make the scale a function of the weights it determines). Ported
    verbatim from `experiments/r63_novel_xsmom_rank.py:156-166`.
    """
    acc = None
    for df in aligned.values():
        r = np.log(df["close"]).diff()
        acc = r if acc is None else acc + r
    return acc / len(aligned)


def conditional_vol_scale(r: pd.Series) -> np.ndarray:
    """The scale half of `KellyRegimeV3.prepare()`, copied verbatim from
    `experiments/r63_novel_xsmom_rank.py:169-204`: the 8-day EWM realized
    vol (shifted one bar), its 180-day EWM anchor, the latched high/low
    breakout state machine, and `full` vs `steady` inverse-vol sizing, all
    at v4's shipped constants.
    """
    vol = (r.ewm(span=VOL_SPAN, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    slow = (pd.Series(vol).ewm(span=ANCHOR_SPAN_DAYS * BARS_PER_DAY,
                               min_periods=BARS_PER_DAY).mean().to_numpy())

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(TARGET_VOL / vol, MAX_LEVERAGE)
        steady = np.minimum(TARGET_VOL / slow, MAX_LEVERAGE)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    n = len(vol)
    scale = np.zeros(n)
    state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > HIGH_IN else (-1 if x < LOW_IN else 0)
            elif state == 1 and x < HIGH_OUT:
                state = 0
            elif state == -1 and x > LOW_OUT:
                state = 0
        scale[i] = full[i] if state != 0 else steady[i]
    return scale


# --------------------------------------------------------- the entry band


def band_selection(s: np.ndarray, k: int, buffer: float, hold_days: float,
                   delta_in: float, delta_out: float) -> np.ndarray:
    """R-65's rank-buffer/hold-days selection loop, generalized by R-68 to
    two independent eligibility thresholds. Ported verbatim (event ledger
    bookkeeping dropped -- it is a diagnostic, not part of the decision)
    from `experiments/r68_conservative_band_decomposition.py:315-415`.

        enter_eligible = isfinite(s) & (s >  +delta_in)   new entrants
        hold_eligible  = isfinite(s) & (s >  -delta_out)  incumbents

    At `delta_in == delta_out` this is R-67's rule; at
    `delta_in == delta_out == 0.0` it is R-65's. STRICTLY CAUSAL BY
    CONSTRUCTION: a forward loop whose state at bar `i` depends on rows
    <= i and nothing else. No mean, std, quantile or scaler is taken
    anywhere; both thresholds and `buffer` are raw score units.

    Returns the boolean selection matrix, `sel[i, a]` = asset `a` held at
    bar `i`.
    """
    n, n_assets = s.shape
    finite = np.isfinite(s)
    enter_eligible = finite & (s > float(delta_in))
    hold_eligible = finite & (s > -float(delta_out))
    hold_bars = int(round(float(hold_days) * BARS_PER_DAY))
    buf = float(buffer)

    sel = np.zeros((n, n_assets), dtype=bool)
    held: list[int] = []
    last_change = -(1 << 60)

    for i in range(n):
        row = s[i]
        elig_in = enter_eligible[i]
        elig_hold = hold_eligible[i]
        changed = False

        # (a) forced exits -- never blocked by the timer. An incumbent
        #     leaves only once its score is no longer above -delta_out.
        if held:
            keep = [a for a in held if elig_hold[a]]
            if len(keep) != len(held):
                held = keep
                changed = True

        # entries into empty slots (including refilling a slot a forced
        # exit just freed, and re-entering from flat). A new entrant must
        # clear +delta_in.
        if len(held) < k:
            free = [a for a in range(n_assets) if elig_in[a] and a not in held]
            if free:
                free.sort(key=lambda a: -row[a])
                for a in free[: k - len(held)]:
                    held.append(a)
                    changed = True

        # (b) voluntary swap -- buffered AND time-gated. The challenger is
        #     a new entrant and must clear +delta_in.
        elif not changed:
            worst = min(held, key=lambda a: row[a])
            out = [a for a in range(n_assets) if elig_in[a] and a not in held]
            if out:
                best = max(out, key=lambda a: row[a])
                if row[best] > row[worst] + buf and (i - last_change) >= hold_bars:
                    held.remove(worst)
                    held.append(best)
                    changed = True

        if changed:
            last_change = i
        if held:
            sel[i, held] = True

    return sel


def _size(sel: np.ndarray, aligned: dict[str, pd.DataFrame], k: int,
         index: pd.Index, assets: list[str]) -> pd.DataFrame:
    """R-63's sizing block. Ported verbatim from
    `experiments/r68_conservative_band_decomposition.py:418-438` (itself
    copied unmodified from R-65/R-67)."""
    n = sel.shape[0]
    m = sel.sum(axis=1)
    scale = conditional_vol_scale(basket_log_returns(aligned))

    desired = scale * (m / float(k))
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        d = desired[i]
        if abs(d - cur) > VOL_LATCH_DEADBAND:
            cur = d
        pos[i] = cur

    total = np.minimum(pos, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(m > 0, total / np.maximum(m, 1), 0.0)
    w = sel * per[:, None]
    return pd.DataFrame(w, index=index, columns=assets)


def build_targets(aligned: dict[str, pd.DataFrame], k: int = K_FIXED,
                  buffer: float = BUFFER_FIXED, hold_days: float = HOLD_FIXED,
                  delta_in: float = DELTA_IN, delta_out: float = DELTA_OUT) -> pd.DataFrame:
    """Target weight matrix for the frozen R-68 ENTRY_ONLY configuration."""
    score = cross_sectional_score(aligned)
    assets = list(score.columns)
    s = score.to_numpy(dtype=float)
    sel = band_selection(s, k, buffer, hold_days, delta_in, delta_out)
    return _size(sel, aligned, k, score.index, assets)


@register_multi_asset
class XsmomEntryBand(MultiAssetStrategy):
    """R-63's cross-sectional trend score under R-68's asymmetric entry band (delta_in=0.080); NEGATIVE (R-68, D1/D2 both failed), registered as B-32's infrastructure correctness check."""

    name = "xsmom_entry_band"
    instruments = UNIVERSE_6
    warmup_days = WARM_DAYS

    def build_targets(self, aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return build_targets(aligned)
