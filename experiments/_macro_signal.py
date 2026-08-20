"""Shared macro-stress signal for the R-53 macro-INFO round.

Operator-authored, shared infrastructure -- both parallel branches
(``kelly_regime_v14_macro_brake.py``, conservative, and
``kelly_regime_v14_macro_lead.py``, novel) import this unchanged, so the
underlying signal construction is identical between them and any
difference in results is attributable to *how the signal is used*
(a scale haircut vs. a regime-vote input), not to one branch quietly
fitting a better version of the signal itself. Neither branch may edit
this file; a branch that wants a different construction says so in its
own report rather than forking this one, per ROUTINE.md's parallelism
rules on disjoint files.

Data: ``data/{spx,vix,dxy}_daily.csv.gz``, fetched by
``scripts/fetch_macro_data.py`` from FRED (S&P 500 close, VIX close, Fed
trade-weighted broad dollar index). All three describe the rest of the
financial system, not BTC/ETH's own price or chain -- the first genuinely
new INFO channel this project has tried since on-chain metrics (B-07,
R-44, both branches NEGATIVE for unrelated reasons).

Mechanism (why this should carry signal): the VIX/DXY-Bitcoin spillover
literature argues equity-market fear (VIX) and dollar risk-off flows
(DXY strength) lead or coincide with crypto drawdowns rather than merely
correlating with them contemporaneously in tranquil periods -- Luo, Tsai
& Yen (2024/2025, SSRN) find the VIX term-structure slope is a primary
determinant of Bitcoin returns with a persistent negative sign across
maturities, unchanged by the Jan-2024 spot-ETF institutionalization even
as Bitcoin's sensitivity to its *own* implied vol fell; the IMF (WP
2023/213) finds crypto-equity spillovers intensify specifically during
stress periods rather than holding at a constant level; multiple
2024-2026 studies document a strengthening BTC/DXY inverse relationship
consistent with Bitcoin trading as a risk asset against dollar strength,
not as an uncorrelated or safe-haven asset (Klein, Thu & Walther 2018,
*Int. Rev. Financial Analysis*, made the same "risk asset, not safe
haven" case earlier from equity/gold correlations alone).

Only two derived features, both z-scored against a fixed trailing window
so the composite carries no unit-specific scale, and both computed on the
RAW DAILY series before any causal shift is applied (rolling stats must
never see a future day; ``align_macro_causal`` only controls *when* an
already-final daily number becomes visible on the bar grid, not whether
the number itself was computed causally):

- ``vix_z``:  VIX close, z-scored against its own trailing 365-day
  mean/std. Levels-based, not returns-based -- VIX is already
  mean-reverting and stationary in level, unlike price.
- ``dxy_mom_z``: 20-day change in the dollar index, z-scored against the
  trailing 365-day distribution of that same 20-day change. Momentum, not
  level, because DXY itself is a slow-moving unit-root-like series and the
  literature's mechanism is dollar *strengthening* (risk-off flows into
  the dollar), not any particular DXY level.

``stress_z = 0.5 * vix_z + 0.5 * dxy_mom_z`` -- an equal a-priori weight,
not fit to this data, so the composite itself is not a hidden free
parameter either branch could be accused of tuning. Positive values mean
elevated equity fear and/or dollar strengthening; both branches read that
as the risk-off direction the literature's mechanism predicts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tradebot.data import align_macro_causal, load_macro_metrics

VIX_WINDOW_DAYS = 365
DXY_MOM_DAYS = 20
DXY_WINDOW_DAYS = 365


def compute_macro_stress(df: pd.DataFrame, data_dir: str | Path) -> pd.Series:
    """Causal ``stress_z`` aligned to ``df``'s bar index, or all-NaN if data is absent.

    Every rolling statistic is computed on the raw daily macro frame
    (strictly backward-looking: ``.rolling(...).mean()`` / ``.std()`` at
    day D use only days <= D), and only the finished daily ``stress_z``
    series is projected onto the bar grid via ``align_macro_causal`` --
    which additionally shifts it one more day forward for FRED's own
    publication lag. A bar therefore only ever sees a ``stress_z`` value
    computed from macro data published on or before its own previous day.
    """
    macro = load_macro_metrics(data_dir)
    if macro is None:
        return pd.Series(index=df.index, dtype=float)

    vix = macro["vix"]
    vix_z = (vix - vix.rolling(VIX_WINDOW_DAYS, min_periods=60).mean()) / vix.rolling(
        VIX_WINDOW_DAYS, min_periods=60
    ).std()

    dxy = macro["dxy"]
    dxy_mom = dxy - dxy.shift(DXY_MOM_DAYS)
    dxy_mom_z = (dxy_mom - dxy_mom.rolling(DXY_WINDOW_DAYS, min_periods=60).mean()) / dxy_mom.rolling(
        DXY_WINDOW_DAYS, min_periods=60
    ).std()

    stress_daily = (0.5 * vix_z + 0.5 * dxy_mom_z).rename("stress_z").to_frame()
    return align_macro_causal(stress_daily, df)["stress_z"]
