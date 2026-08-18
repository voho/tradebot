"""kelly_regime_v4 with exposure reduced by a continuous funding-crowding cost."""

from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.data import load_funding
from tradebot.registry import register
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


@register
class KellyRegimeFunding(KellyRegimeV4):
    """v4's sizing minus a continuous funding-crowding cost term (backlog B-05).

    Mechanism: ``kelly_regime``'s sizing step is
    ``desired = frac * min(target_vol/vol, max_leverage)`` - a stand-in
    for the growth-optimal ``mu/vol**2``, calibrated through the constant
    ``target_vol`` instead of an estimated ``mu``. A continuously
    accruing cost ``phi`` (the EWM-smoothed annualized perp funding rate)
    subtracts linearly from that same numerator, the continuous analogue
    of how a one-time transaction fee subtracts from the no-trade band
    ``kelly_regime_ev`` already derives (Constantinides 1986):

        exposure = max(0, desired - k * phi / vol**2)

    floored at 0 since the anchor vote is always in [0, 1] (never
    short), so there is no negative exposure to protect. Everything else
    - anchors, conditional vol targeting, deadband, cap - is v4
    unchanged; this only touches the sizing step, after the regime vote
    has already run.

    Why funding, specifically: v4 infers crowding indirectly, from price
    sitting above or below slow anchors. Funding is a *direct, priced*
    measurement of the same thing - Schmeling, Schrimpf & Todorov (2023,
    BIS Working Paper 1087) trace crypto's large, volatile funding basis
    to leveraged trend-chasing demand meeting limited arbitrage capital,
    i.e. funding is the rent the crowd pays for being crowded. Angeris,
    Chitra & Evans (2022) and He, Manela, Ross & von Wachter (2024) give
    the no-arbitrage relation between perp price, spot price and funding
    that motivates treating it as a continuous carry cost inside the
    growth objective, rather than a one-time fee.

    **What the evidence supports, and what it does not.** Frozen config
    (k=1.5, 1-day funding EWM span) selected on funding-inner-validation
    (2022) against all 12 swept (k, span) configs - a plateau along k,
    a smaller but still-favourable-everywhere ridge along span - then
    read once on funding-holdout (2023-01-01..2023-12-31, futures 5x,
    **funding charged**, vs this same v4 baseline also funding-charged):
    max drawdown falls from 28.4% to 12.1%, a **statistically
    significant** paired block-bootstrap difference of -15.9pp
    [-24.0, -6.2] (30-day blocks, 2,000 resamples, R-29/R-30's
    protocol). That drawdown cut also beats a plain constant-scaled-down
    v4 matched to the *same realized volatility* in every measured
    period (funding-inner-train, funding-inner-validation, and this
    holdout) - so it is not merely "hold less", which is exactly the
    artifact that dissolved the e-process gate's drawdown claim at
    matched risk (R-31, R-32). **The return improvement is NOT
    established**: the holdout log-growth difference is a small,
    *negative*, non-significant point estimate (-0.28 [-0.89, +0.29]),
    the wrong sign against funding-inner-validation's positive one - read
    this exactly the way v4's own docstring reads its own return claim.
    Interesting side finding, not the promotion basis: on **spot**,
    which pays no funding at all, the same funding-derived exposure cut
    still reduces drawdown substantially (24%->7% in 2020-21, 28%->10%
    in 2022, 24%->12% in the 2023 holdout) and even improved 2022's spot
    return - consistent with funding acting as a genuine crowding/timing
    signal (R-16's finding that funding predicts forward returns) rather
    than only a cost hedge.

    **Read the default comparison table's futures_5x number for this
    strategy as an understatement of its purpose**, more so than for any
    other row: `tradebot run`'s standard matrix does not charge funding
    at all (see the README's funding warning), so it cannot show the
    cost this strategy exists to protect against - the number that
    matters is the funding-charged one above, reproducible with
    `scripts/funding_study.py`. See docs/VALIDATION.md for the full
    sweep, the matched-risk checks and the by-year regime split.
    """

    name = "kelly_regime_funding"

    def __init__(self, funding: pd.Series | None = None, k: float = 1.5,
                 funding_span_days: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.funding = funding if funding is not None else load_funding(_DATA_DIR)
        self.k = k
        self.funding_span_days = funding_span_days

    def _phi(self, df: pd.DataFrame) -> np.ndarray:
        """Causal EWM-smoothed annualized funding rate, aligned to df.index.

        Aligned with a 24h backward ``merge_asof`` tolerance (settlements
        are 8h apart, so a bar is normally within one settlement of a
        real observation) rather than an unbounded forward-fill. A plain
        ffill would carry the *last ever* committed settlement (funding
        data ends 2023-12-31) forward indefinitely into every later bar
        - years-stale and, worse, divided by a *shrunk* `vol**2` in later,
        calmer regimes, which pushed the funding term far out of scale
        with `desired` and zeroed the strategy out entirely on real data
        past 2023 (caught by `tests/test_causality_strict.py`'s
        real-slice trading check). Outside the tolerance this returns 0,
        i.e. degrades to plain v4 - the same "no data -> no adjustment"
        contract already promised for bars before the funding file's
        first settlement. Verified to reproduce the ffill version
        byte-for-bar within every window this strategy was actually
        selected or evaluated on (funding-inner-train/-validation/
        -holdout, all inside the file's real 2020-2023 coverage, where
        settlements are never more than 8h apart).
        """
        if self.funding is None or len(self.funding) == 0:
            return np.zeros(len(df))
        bars = pd.DataFrame({"timestamp": df.index})
        funding_df = self.funding.rename("rate").reset_index()
        funding_df.columns = ["timestamp", "rate"]
        funding_df["timestamp"] = funding_df["timestamp"].astype(bars["timestamp"].dtype)
        merged = pd.merge_asof(bars, funding_df, on="timestamp", direction="backward",
                                tolerance=pd.Timedelta(hours=24))
        aligned = merged["rate"].fillna(0.0).to_numpy()
        annualized = aligned * 3.0 * 365.25  # 3 settlements/day -> per-year rate
        span_bars = max(1, int(round(self.funding_span_days * BARS_PER_DAY)))
        phi = pd.Series(annualized, index=df.index).ewm(span=span_bars, min_periods=1).mean()
        return phi.to_numpy()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()
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

        phi = self._phi(df)
        with np.errstate(divide="ignore", invalid="ignore"):
            funding_term = self.k * phi / (vol ** 2)
        valid_vol = np.isfinite(vol) & (vol > 0)
        funding_term = np.where(valid_vol & np.isfinite(funding_term), funding_term, 0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = max(0.0, frac[i] * scale - funding_term[i])
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df
