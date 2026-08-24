"""Shared, read-only utilities for the R-115 NOVEL branch (smart-money /
retail long-short DIVERGENCE, 08-24).

This module duplicates nothing that already exists correctly in
``r81_shared.py`` -- it *imports* that module's anchor-vote construction,
episode table, block-bootstrap null generator and causality probe (the
same reuse pattern ``r81_conservative_crowding_vote.py`` itself used
against ``r81_shared.py``) rather than recopying them, since ``r81_shared``
is itself already the frozen, operator-authored shared-utility file for
this exact price series / vote construction and re-typing it byte-for-byte
into a second file would only create a second place for the two copies to
drift apart. What IS new here: the retail-vs-smart-money DIVERGENCE
feature construction (``divergence_z``), which no prior round has built,
and a coverage-gap diagnostic for ``count_long_short_ratio`` (R-81 only
ever measured the gap on ``sum_toptrader_long_short_ratio``).

=============================================================================
CONSTRUCTION (frozen before any real-market gate number was computed)
=============================================================================

``count_long_short_ratio`` (Binance's ALL-ACCOUNT, count-weighted long/short
ratio -- i.e. every account trading the symbol, dominated by retail volume)
and ``sum_toptrader_long_short_ratio`` (Binance's TOP-TRADER, size-weighted
ratio -- the largest accounts by margin balance, R-81's own primary
signal) are each z-scored against their own trailing ``window_days``-day
mean/std, causally (rolling, backward-only -- see ``divergence_z`` below).
``window_days=14`` reuses ``r81_shared.crowding_z``'s own choice rather
than re-deriving a new one: it is a reasonable "roughly two weeks of
recent regime" baseline for a ratio series with slow-moving level drift,
and reusing it (instead of picking a fresh number that could be tuned
toward a good-looking result) keeps this round's one new design choice
confined to the divergence FORMULA itself, not the baseline window too.

    retail_z      = z(count_long_short_ratio)
    smart_z       = z(sum_toptrader_long_short_ratio)
    divergence_z  = smart_z - retail_z

Economic reading (stated now, not retrofitted after seeing a result):
``divergence_z > 0`` means top-traders are MORE net-long, relative to
their own recent baseline, than retail is relative to ITS OWN baseline --
smart money leaning bullish while retail lags/leans bearish.
``divergence_z < 0`` means the reverse: retail is more net-long relative
to its own baseline than top-traders are relative to theirs -- retail
crowded long while smart money is not following, the Barber & Odean
(2000, JF 55(2), "Trading Is Hazardous to Your Wealth") "retail crowds
the wrong way" setup, theorized here to precede a downside forced-
deleveraging reversal. The pre-registered Step-A gate below tests
``|divergence_z|`` (MAGNITUDE only, matching R-81's own ``|ls_z|``
convention and this round's brief) -- it asks only whether an extreme
divergence, in EITHER direction, arrives before v4's own reaction, not
which direction it points. Direction would only start to matter at a
Step B this round does not reach unless the gate passes.

Literature grounding (short web search run before writing any gate code,
disclosed honestly): no peer-reviewed 2023-2025 paper was found that
studies this EXACT construction (Binance's own top-trader-vs-all-account
long/short ratio divergence). What was found: (1) Dunbar & Owusu-Amoako
(2023, "Predictability of crypto returns: The impact of trading
behavior," Journal of Behavioral and Experimental Finance 39:100812) --
genuinely crypto-specific and peer-reviewed, showing CME Bitcoin-futures
speculative/retail (COT "non-commercial") net-short positioning is a
statistically strong return predictor even controlling for attention,
uncertainty and sentiment; this supports "retail crypto-futures
positioning carries return-predictive information" but is a DIFFERENT
venue and a different retail/institutional split (CFTC COT categories,
not Binance's top-trader-by-margin-balance definition) than this round's
signal, so it is suggestive support, not a direct precedent. (2) Several
2025-vintage practitioner/vendor sources (Sharpe Terminal, CoinGlass,
Hyblock Academy) describe the exact qualitative pattern this round
formalizes -- "when top traders diverge from the retail crowd... align
with top traders" -- as a widely-used discretionary heuristic, but these
are not peer-reviewed research and are reported here as exactly that:
a heuristic in wide practitioner use, not academic evidence. Absent a
crypto-specific academic study of this precise divergence, this round
otherwise leans on the general retail-sentiment-as-contrarian-indicator
literature Barber & Odean (2000) anchors, disclosed as such rather than
overclaimed as a crypto-derivatives-specific citation.

Not a duplicate of R-81 (`r81_shared.crowding_z`'s ``ls_z``): that used
``sum_toptrader_long_short_ratio`` ALONE (one trader class, "is the
top-trader ratio itself extreme"). This is a two-column DIVERGENCE
between two DIFFERENT trader classes' ratios -- an economically distinct
claim (crowding measured relative to who else is/isn't crowded the same
way, not crowding measured in isolation) built from a column
(``count_long_short_ratio``) R-81 never touched at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.data import align_metrics_causal, load_binance_metrics

# Re-exported, not recopied -- see module docstring for why.
from experiments.r81_shared import (  # noqa: F401
    BARS_PER_DAY,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    METRICS_END,
    METRICS_START,
    OOS_START,
    STRESS_EPISODES,
    anchor_majority,
    anchor_votes,
    block_bootstrap_lead_null,
    truncation_causality_probe,
)

Z_WINDOW_DAYS = 14  # reused from r81_shared.crowding_z, not re-derived


def load_metrics_truncated(data_dir, asset: str) -> pd.DataFrame | None:
    """Raw Binance metrics for ``asset``, truncated to ``METRICS_END``
    (== ``INNER_VAL_END``, strictly before ``OOS_START``), or ``None`` if
    the file is absent. Mirrors ``r81_shared.load_crowding_inputs`` exactly
    (kept as a local copy, one function, rather than importing that
    specific helper, so this module's own import list stays confined to
    genuinely shared machinery rather than one branch's own convenience
    wrapper)."""
    df = load_binance_metrics(data_dir, asset)
    if df is None:
        return None
    cutoff = pd.Timestamp(METRICS_END, tz="UTC") + pd.Timedelta(days=1)
    return df.loc[df.index < cutoff].copy()


def _causal_zscore(s: pd.Series, window_days: int) -> pd.Series:
    w = int(window_days * BARS_PER_DAY)
    mean = s.rolling(w, min_periods=w // 4).mean()
    std = s.rolling(w, min_periods=w // 4).std()
    return (s - mean) / std.replace(0.0, np.nan)


def divergence_z(metrics: pd.DataFrame, bars: pd.DataFrame,
                  window_days: int = Z_WINDOW_DAYS) -> pd.DataFrame:
    """The three causal, bar-aligned features this round's gate tests:
    ``retail_z``, ``smart_z``, ``divergence_z = smart_z - retail_z``. See
    the module docstring for the exact formula, sign convention and why
    ``window_days`` defaults to 14 (reused from R-81, not re-picked).

    Causal by construction: ``align_metrics_causal`` only ffills past
    values onto ``bars``' grid (never a future one -- see that function's
    own docstring in ``tradebot/data.py``), and ``rolling`` only looks
    backward. No full-series fit (mean/std over the WHOLE series) is
    taken anywhere -- both moments are themselves rolling, so a truncated
    frame cannot change an earlier bar's z-score.
    """
    aligned = align_metrics_causal(metrics, bars)
    retail_z = _causal_zscore(aligned["count_long_short_ratio"], window_days)
    smart_z = _causal_zscore(aligned["sum_toptrader_long_short_ratio"], window_days)
    return pd.DataFrame(
        {"retail_z": retail_z, "smart_z": smart_z, "divergence_z": smart_z - retail_z},
        index=bars.index,
    )


def coverage_gap_pct(metrics: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """NaN percentage of each column in ``cols`` over ``metrics``' own full
    (already-truncated-to-METRICS_END) index -- the same "whole committed
    window" accounting R-81's ledger entry reported for
    ``sum_toptrader_long_short_ratio``, applied here to whichever columns
    are asked for."""
    n = len(metrics)
    return {c: 100.0 * float(metrics[c].isna().sum()) / n for c in cols} if n else {}


def episode_window_gap_pct(metrics: pd.DataFrame, cols: list[str], onset: str,
                            window_days: int) -> dict[str, float] | None:
    """Same NaN-percentage diagnostic as ``coverage_gap_pct``, restricted
    to one episode's own ``[onset - window_days, onset + window_days]``
    search window -- the window the gate itself searches, so a column can
    look "mostly clean" over the whole file (``coverage_gap_pct``) while
    still being unusable for one specific episode, or vice versa. Returns
    ``None`` if the window has zero overlap with ``metrics``' own index
    (the episode entirely predates this asset's coverage)."""
    onset_ts = pd.Timestamp(onset, tz="UTC")
    lo = onset_ts - pd.Timedelta(days=window_days)
    hi = onset_ts + pd.Timedelta(days=window_days)
    seg = metrics.loc[(metrics.index >= lo) & (metrics.index <= hi)]
    if len(seg) == 0:
        return None
    return coverage_gap_pct(seg, cols)
