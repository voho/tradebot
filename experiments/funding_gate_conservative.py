"""Variant A (conservative) of B-05: a pure risk-override funding gate on
top of ``kelly_regime_v4``.

Pre-registered design (see the project pre-registration written before any
of this file's code ran): stand flat (target -> 0) whenever the *causally
known* funding rate at the current bar sits at or above its own trailing
decile threshold, AND ``kelly_regime_v4``'s own target says long (> 0).
Short targets and flat targets are never touched. The gate can only ever
*remove* exposure relative to v4 - it can never add any - so on any bar
where it doesn't fire, behaviour is byte-identical to v4.

Mechanism, restated: funding is the price of being on the crowded side of
a perp. R-16 (this repo) found real predictive content in the *rank* of
funding within its own trailing distribution, not in the raw rate, and
flagged the raw-rate middle quintiles as non-monotone / noisy - which is
why this variant thresholds on a rank statistic (a trailing quantile) and
only uses the extreme (top-decile-ish) region rather than trying to be
monotone across the whole range. R-14 (this repo) measured funding as
running about 20%/yr while the strategy holds a long, against 2.8% flat -
i.e. cost scales with exactly the state (long, funding rich) this gate
vetoes.

Longs only, deliberately. Positive funding is paid BY longs TO shorts; a
short position in that same state is being PAID, not charged, so vetoing
shorts on the same signal would remove a state that is a tailwind, not a
drag - the opposite of a "pure risk override". The mirror trade (dampen
shorts when funding is very *negative*, i.e. shorts are the crowded,
paying side) is a real, symmetric idea, but it is a second mechanism, not
implied by "never add exposure, only ever remove it" applied to the
in-sample-observed sign of funding (positive 86% of settlements per
``scripts/funding_study.py``). Left out of this conservative variant on
purpose; flagged in the report rather than silently added.

Causality: the trailing quantile of funding is computed on the funding
SETTLEMENT series (8-hourly), using ``.rolling(window, min_periods=...)
.quantile(q)`` - which is an expanding quantile for the first ``window``
settlements and a genuine trailing rolling quantile after that - then
SHIFTED by one settlement, so the threshold effective at settlement t is
computed from settlements strictly before t. The (rate, threshold) pair is
then merged onto the 5m bar grid with ``pd.merge_asof(..., direction=
"backward", tolerance=8h)``, so a bar only ever sees a settlement whose
timestamp is <= the bar's own timestamp, and never one so old (more than
one settlement interval) that it would silently keep "gating" long past
the end of the committed funding data (2023-12-31). Outside 2020-2023
this makes the funding/threshold columns NaN, which is treated as "no
gate" -> byte-identical v4 behaviour, never fabricated or extrapolated.

Not registered (@register is intentionally NOT applied) and does not
import or modify anything under ``src/tradebot/strategies/``. Pure
scratch space per project convention (``experiments/`` is not
auto-discovered by ``tradebot.registry``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.data import load_funding
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

# repo_root/data - experiments/funding_gate_conservative.py -> parents[0]
# is experiments/, parents[1] is the repo root. Verified against the
# actual file location, not assumed.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# One settlement every 8h (verified exactly regular, no gaps, over the
# whole committed 2020-01-01..2023-12-31 file: min==max==8h).
SETTLEMENTS_PER_DAY = 3


def _funding_threshold(funding: pd.Series, window_days: int, quantile: float,
                        min_settlements: int) -> pd.DataFrame:
    """Causal (rate, trailing-quantile-threshold) pair, settlement-indexed.

    ``rolling(window, min_periods=min_settlements)`` is an expanding
    quantile for the first ``window`` settlements (uses however many are
    available, once at least ``min_settlements`` exist) and a true
    trailing rolling quantile once the window is full - i.e. exactly the
    "expanding early, then rolling window" shape asked for, in one call.
    ``.shift(1)`` then excludes the current settlement from its own
    threshold, so the comparison is never self-referential.
    """
    window = window_days * SETTLEMENTS_PER_DAY
    thr = (funding.rolling(window, min_periods=min_settlements)
           .quantile(quantile).shift(1))
    return pd.DataFrame({"funding_rate": funding, "funding_thr": thr})


class FundingGateConservative(KellyRegimeV4):
    """kelly_regime_v4, standing flat when trailing funding hits its own top decile.

    Extra knobs beyond v4 (both frozen after the funding-train sweep - see
    the experiment report, not this docstring, for the swept values and
    the final choice):

    - ``funding_window_days``: length of the trailing rolling quantile
      window over funding settlements (8h each). Recommended: 180 days
      (540 settlements) - long enough to span multiple regime cycles
      (the underlying vote's own slowest anchor is 80 days) so the
      decile threshold isn't just re-fit to the last few weeks, short
      enough that it still reacts within the ~4-year funding sample
      instead of being dominated by the first year forever.
    - ``funding_quantile``: the decile threshold itself, e.g. 0.90 for
      "top decile". The one tunable knob, per the pre-registration.
    - ``funding_min_settlements``: how many past settlements must exist
      before the gate can fire at all (else NaN threshold -> no gate).
      Recommended: 90 (30 days) - enough points for a 90th-percentile
      estimate to mean anything.
    """

    name = "funding_gate_conservative"  # NOT @register'd - see module docstring

    def __init__(self, funding_window_days: int = 180, funding_quantile: float = 0.90,
                 funding_min_settlements: int = 90, **kwargs) -> None:
        super().__init__(**kwargs)
        self.funding_window_days = funding_window_days
        self.funding_quantile = funding_quantile
        self.funding_min_settlements = funding_min_settlements

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4's `target` column, unmodified computation

        funding = load_funding(DATA_DIR)
        if funding is None:
            # No committed funding file reachable - degrade to byte-identical v4.
            df["funding_rate"] = np.nan
            df["funding_thr"] = np.nan
            df["funding_gated"] = False
            return df

        feats = _funding_threshold(funding, self.funding_window_days,
                                    self.funding_quantile, self.funding_min_settlements)
        # merge_asof requires identical index dtypes; the bar index and the
        # funding index can carry different datetime64 resolutions (ms vs
        # us) even though both are UTC-aware timestamps, so normalize.
        feats = feats.copy()
        feats.index = feats.index.as_unit(df.index.unit) if hasattr(df.index, "unit") \
            else feats.index

        merged = pd.merge_asof(
            pd.DataFrame(index=df.index), feats,
            left_index=True, right_index=True,
            direction="backward", tolerance=pd.Timedelta(hours=8),
        )

        rate = merged["funding_rate"].to_numpy()
        thr = merged["funding_thr"].to_numpy()
        target = df["target"].to_numpy(copy=True)

        # NaN-safe: NaN >= anything is False in numpy, so bars with no
        # (or not-yet-warmed-up) funding data simply never gate - exactly
        # the "NaN -> no gate -> v4 behaviour" fallback required.
        gate = np.isfinite(rate) & np.isfinite(thr) & (rate >= thr) & (target > 0.0)

        target = np.where(gate, 0.0, target)

        df["funding_rate"] = rate
        df["funding_thr"] = thr
        df["funding_gated"] = gate
        df["target"] = target
        return df

    # on_bar intentionally NOT overridden: v4's on_bar just plays
    # ctx.bar["target"], which is exactly the (possibly gated) column
    # this prepare() produces.
