"""Aggregate stablecoin (USDT) supply-deceleration signal for the R-54
NOVEL branch. Private to this branch -- not shared with any parallel
branch this round (the parallel CONSERVATIVE branch has its own,
unrelated files: ``experiments/kelly_regime_v15_macro_veto.py`` and
``experiments/reports/v15_macro_veto_report.md``, neither read nor
touched here).

Data: ``data/stablecoin_supply_daily.csv.gz``, fetched by
``scripts/fetch_stablecoin_supply.py`` from CoinMetrics' free community
API -- ``SplyCur`` (current circulating supply) for USDT, 2017-01-01 to
2026-08-19, committed with 0 NaN and 0 calendar-day gaps over the full
range (verified at fetch time; see the fetch script's own docstring for
the raw numbers).

**Data-scope decision, stated plainly (per this round's brief and this
project's standing "never proxy unavailable data out of price" rule):**
USDC's ``SplyCur`` is also served by the same API, but its real (non-
placeholder) minted supply only begins 2018-09-25 -- roughly 21 months
after this project's inner-train window starts (2017-01-01), and USDT is
overwhelmingly the dominant stablecoin by supply for most of the
2017-2022 span this round's falsification episodes fall in. Rather than
combine the two series (which would either leave USDC's pre-launch
period as an implicit zero -- defensible as a literal fact, but easy to
mistake for backfilling a missing observation -- or introduce a
structural discontinuity in the aggregate the day USDC's real history
begins), this signal uses **USDT alone for the entire period**. This is
the plainer, harder-to-misread choice named in this round's brief as one
of the two explicitly sanctioned options. USDC's data was fetched and
inspected only to confirm it is reachable and clean (see the fetch
script's docstring); it is not used anywhere below.

Mechanism (why this should carry signal, stated in one sentence per
ROUTINE.md step 2): stablecoin issuance is the on-ramp for new dollar
capital entering crypto trading and redemption is the off-ramp, so a
sharp deceleration or outright contraction in aggregate stablecoin
supply plausibly reflects capital already leaving the system -- a
leading indicator of the liquidity withdrawal that later shows up as
price weakness -- in a way an external, indirectly-correlated index
(VIX/DXY, R-53) structurally cannot be, because it is mechanically tied
to capital actually moving in or out of the crypto trading system rather
than an indirect spillover from equity/FX markets.

Literature (found via web search for this round, verified at
abstract/working-paper level, not re-derived here -- the mechanism is
simple enough to state and test directly):
- BIS Working Paper No. 1340, "Stablecoin flows and spillovers to FX
  markets" (2025).
- Ahmed & Aldasoro, "Stablecoins and safe asset prices," Cleveland Fed
  financial-stability conference paper / BIS WP 1270 (August 2025) --
  stablecoin inflows/outflows move measurably and are tied to short-term
  dollar funding markets.
- Federal Reserve Bank of New York, Liberty Street Economics,
  "Stablecoins and Crypto Shocks: An Update" (April 2025) -- documents
  stablecoin market behavior around crypto stress events.
- IMF Working Paper 2025/141, "Decrypting Crypto: How to Estimate
  International Stablecoin Flows" (July 2025).

**Exactly one derived feature**, fixed a-priori and never swept (same
discipline as ``_macro_signal.py``'s fixed VIX/DXY windows), computed
entirely on the RAW DAILY series before any causal shift is applied
(rolling stats must never see a future day; ``align_stablecoin_causal``
only controls *when* an already-final daily number becomes visible on
the bar grid, not whether the number itself was computed causally):

- ``growth_14d``: 14-calendar-day LOG growth of aggregate USDT supply,
  ``log(supply_t) - log(supply_{t-14})``. Log rather than simple percent
  growth because supply itself spans four orders of magnitude over
  2017-2026 (from ~$10M to ~$188B) and log growth keeps the same
  deceleration in relative terms comparable across eras. 14 days rather
  than 20 (DXY's momentum window in R-53) because stablecoin issuance/
  redemption is lumpier and can react to acute stress within days
  (mint/burn transactions are on-chain and near-instant once a stress
  event triggers redemptions), so a shorter window is the a-priori
  choice for a signal whose whole value proposition is arriving faster
  than a monthly-cadence anchor -- not fit to any observed lead/lag
  result, chosen before ``leadtime()`` was run.
- ``stablecoin_stress_z``: ``growth_14d`` z-scored against its own
  trailing 365-day mean/std (``min_periods=60``, matching
  ``_macro_signal.py``'s VIX window exactly), then SIGN-FLIPPED:
  ``stablecoin_stress_z = -1 * zscore(growth_14d)``. Positive values
  mean growth is unusually SLOW or supply is CONTRACTING relative to its
  own trailing year -- the risk-off direction, matching the sign
  convention ``_macro_signal.py``'s ``stress_z`` already uses (positive
  = risk-off) so a reader familiar with the R-53 signal reads this one
  the same way.

**Sign hypothesis, stated before any code ran:** elevated
``stablecoin_stress_z`` (supply decelerating or contracting) should LEAD
-- flip to the risk-off state BEFORE -- ``kelly_regime_v4``'s own
3-anchor majority price-gate flip, on the same stress episodes R-53's
lead-time check used (2018, 2020-03, 2022), because capital leaving the
system is a cause of the subsequent price weakness the anchors react to,
not merely a correlate of it.

**Named risk, stated before any code ran (per this round's explicit
instruction):** it is a fully legitimate, real possibility that this
signal ALSO lags rather than leads, for a structural reason independent
of the mechanism's truth -- daily on-chain supply data feeding a
5-minute-bar strategy has a coarser native cadence than the bars it is
being asked to lead, and CoinMetrics' own reporting/indexing lag (this
signal is given the same 1-day publication-lag causal shift as the
on-chain and macro loaders, via ``align_stablecoin_causal``) could by
itself erase a genuine same-day lead. If the lead-time check below finds
a lag, that is reported plainly as a real negative result, not
explained away.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.data import align_stablecoin_causal, load_stablecoin_supply

GROWTH_WINDOW_DAYS = 14
ZSCORE_WINDOW_DAYS = 365


def compute_stablecoin_stress(df: pd.DataFrame, data_dir: str | Path) -> pd.Series:
    """Causal ``stablecoin_stress_z`` aligned to ``df``'s bar index, or all-NaN if data is absent.

    Every rolling statistic (the 14-day growth rate, the 365-day z-score
    mean/std) is computed on the raw daily USDT-supply frame (strictly
    backward-looking: values at day D use only days <= D), and only the
    finished daily ``stablecoin_stress_z`` series is projected onto the
    bar grid via ``align_stablecoin_causal`` -- which additionally shifts
    it one more day forward for CoinMetrics' own reporting lag, exactly
    as ``align_macro_causal``/``align_onchain_causal`` already do. A bar
    therefore only ever sees a value computed from a day's supply
    reported strictly before its own previous day.
    """
    supply = load_stablecoin_supply(data_dir)
    if supply is None:
        return pd.Series(index=df.index, dtype=float)

    s = supply["supply"]
    log_s = np.log(s)
    growth = log_s - log_s.shift(GROWTH_WINDOW_DAYS)
    z = (growth - growth.rolling(ZSCORE_WINDOW_DAYS, min_periods=60).mean()) / growth.rolling(
        ZSCORE_WINDOW_DAYS, min_periods=60
    ).std()
    stress_daily = (-1.0 * z).rename("stablecoin_stress_z").to_frame()
    return align_stablecoin_causal(stress_daily, df)["stablecoin_stress_z"]
