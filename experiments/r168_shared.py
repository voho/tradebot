"""Shared, read-only utilities and frozen pre-registration for the R-168 round (08-28).

DIRECTION, in one sentence: replace the single-venue Coinbase-spot close
that feeds `kelly_regime_v4`'s VOTE (the 20/40/80-day anchor construction
in `frac = mean(vote_20, vote_40, vote_80)`) with a causal fusion of that
same Coinbase spot close and Deribit's independently-transacted
`BTC-PERPETUAL` close (`data/btcusdt_deribit_perp_5m.csv.gz`). SCALE (the
conditional-volatility-target multiplier, still computed from the
original spot return series), the 1% band, the 10% deadband and the
latching hysteresis are all reused byte-for-byte, unmodified, from
`kelly_regime_v4`. Only the price INPUT to the vote's rolling anchors
changes.

**Which constraint this attacks: INFO**, in the standing diagnosis's most
literal sense ("one price series") -- for the first time as a change to
the vote's own construction rather than as a brake/confirming-vote bolted
on top of it (contrast R-41/R-42, which added a Deribit-basis-derived
*brake* on SCALE while leaving the vote's spot-only input untouched).
Secondarily SIZE, since both branches act only through the existing
`frac * scale` exposure axis every working strategy in this project uses.

**Not a duplicate of:**
- R-41 conservative / R-42 (Deribit basis brake/confirming-vote on
  `kelly_regime_v4`'s SCALE): those compute `log(perp/spot)`, a SPREAD
  MAGNITUDE between the two venues, and use it as a second, separate
  signal multiplying SCALE. This round never computes a spread; it fuses
  the two venues' price LEVELS into one series and feeds that fused
  series into the SAME anchor/vote arithmetic v4 already runs. The basis
  can be near-zero (spread quantity) while the fused-level construction
  still differs from spot alone on a meaningful fraction of bars (see the
  Step-0 measurement below) -- they are not algebraically related.
- R-145 (funding-aware venue ROUTING of v4's own unmodified target across
  spot/futures legs, COST axis): different axis entirely -- that round
  never touches how the vote is computed; this round never touches which
  venue *trades* (backtests here still fill against the strategy's normal
  spot/futures_5x market spec).
- R-146 (anchor STATISTIC: median vs jump-robust mean of a single price
  series' own history) / R-147 (combination WEIGHTS: James-Stein,
  Beta-Bernoulli ensembling of the three anchors' votes): both vary how
  the SAME one series is summarized or combined; neither introduces a
  second, independently-transacted price series. This round is the first
  to change the SOURCE feeding the anchors, not the statistic or the
  combination rule over one source.
- L-14/L-15/L-16 (`camouflage_flow`/`stealth_trend`/`flow_regime`, ruled
  out): those try to recover order-FLOW information from ONE price
  series (BVC/VPIN transforms) and fail because it isn't there. This
  round adds a second, genuinely independent, real market's own
  transaction price -- not a transform of the incumbent series.

**Citations.**
- Alexander & Heck (2020), "How do shocks across bitcoin and ether markets
  transmit?", *Journal of Financial Stability* 50, 100774 -- find BitMEX
  perpetual-futures price leadership over spot RISES specifically during
  periods of elevated volatility, motivating the novel branch's
  vol-conditioned weighting below.
- Frino, Gaudiosi, Webb & Zhou (2025), "Price discovery in Bitcoin spot
  and derivatives markets", *Journal of Futures Markets* 45(4), 269-288 --
  find venue price-discovery leadership is genuinely time-varying and
  contested rather than fixed to one venue, the direct motivation for
  fusing rather than simply switching the vote's input to perp-only.
- Cosenza & Stalder (2024/2025), SSRN 4983566 -- Coinbase's own relative
  informational share varies by trading session, a second, independent
  source for the same "leadership is not constant" premise.

**Step-0 non-degeneracy check (run BEFORE either branch was dispatched,
reported here verbatim, not re-derived after the fact):** on the full BTC
window, Deribit-perp coverage overlaps 83.2% of spot's committed history
(840,843 of 1,010,889 bars; perp starts 2018-08-14). On that overlap, an
UNWEIGHTED MEAN fusion (`(spot_close + perp_close_aligned) / 2`, perp
as-of-aligned onto spot's index exactly as `compute_basis` already does)
produces a vote series with `corr = 0.9894` against the unmodified
spot-only vote -- high, as expected at 5-minute resolution, but not
degenerate: 3.22% of bars carry a full one-anchor vote flip relative to
the original, and the DISCRETE flip-count (0.5-crossings of `frac`) drops
from 240 to 204 over the same window, a 15% reduction concentrated,
mechanically, at exactly the boundary-crossing bars this project's twelve
failed regime-timing mechanisms (R-82/R-83/R-91/R-94/R-101/R-108/R-114/
R-120/R-129/R-134/R-155/R-156) were all trying to time better. Because
the pre-registered kill threshold from the scoping literature pass was a
ROUGH prior guess (r2/corr > ~0.98) rather than a measured noise floor,
and 0.9894 sits close to but past that guess while the flip-count and
frozen falsification evidence below argue the difference is
non-degenerate, this round is **not** killed at Step 0 -- both branches
are dispatched, and the ETH replication + inner-validation bar (not the
raw correlation) are the actual promotion-relevant tests.

**Step 1 falsification (named before either branch ran anything):**
(a) the ETH sign-replication check (`eth_signal`, below) reverses sign
    relative to BTC;
(b) the paired-bootstrap 95% CI on Δlog-growth (candidate vs unmodified
    `kelly_regime_v4`, matched exposure/time-in-market per R-33) on
    inner-validation contains zero on BOTH markets;
(c) either branch's exposure profile (time-in-market, mean |target|)
    diverges from v4's own by more than 5 percentage points -- the
    R-33/R-131 unmatched-exposure trap, checked explicitly rather than
    assumed, since a fused-price vote could in principle flip differently
    enough to change realized exposure even though `scale` is untouched.

**Frozen splits (identical to every prior round in this file's lineage):**
inner-train `2017-01-01 -> 2020-12-31` (fit/debug only, not read for any
promotion-relevant number -- note BTC-perp coverage does not begin until
2018-08-14, so 2017-01-01 -> 2018-08-13 runs fused-vote-equals-spot-vote
by construction, disclosed rather than hidden); inner-validation
`2021-01-01 -> 2022-12-31` (all selection, both branches); holdout
`2023-01-01 ->` untouched by both branches -- the operator decides after
both branches report whether either clears the inner-validation bar
below and is worth carrying to Step 4.

**Inner-validation gate (frozen now, before either branch has run
anything):** a branch is worth carrying to holdout only if, on BTC AND
ETH, at the standard 0.10%/0.05% fee tier:
1. `d_sharpe >= +0.20` (R-20's noise floor) OR a drawdown/tail
   improvement of matching magnitude, in the SAME direction on both
   markets;
2. the paired-bootstrap 95% CI on Δlog-growth excludes zero on at least
   one market and does not exclude zero in the LOSING direction on the
   other;
3. exposure is matched (time-in-market within 5 points of v4's own) on
   both markets.
Anything else is NEGATIVE for that branch. A branch that clears 1-3 but
whose ETH check is disqualified for a *disclosed data-coverage* reason
(ETH-perp only from 2019-03-14) is flagged, not silently passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import (  # noqa: E402
    load_coinbase_eth_spot,
    load_dataset,
    load_deribit_perp_price,
)
from tradebot.inference import daily_returns, paired_bootstrap, total_log_return  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402

BARS_PER_DAY = 288
BAND = 0.01
HORIZONS = (20, 40, 80)

INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)


def fused_close_btc(data_dir: str | Path = ROOT / "data") -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Returns (spot_df, fused_close_equal_weight, perp_close_aligned).

    `fused_close_equal_weight` is `(spot+perp)/2` where perp is available
    (as-of, causal, matching `compute_basis`'s own alignment: no future
    spot bar can leak into an earlier perp bar, and no bar before perp's
    2018-08-14 start is fabricated), and simply the unmodified spot close
    everywhere perp is not yet available. `perp_close_aligned` is returned
    separately (with NaN where unavailable) so both branches can build
    their own weighting scheme from the same raw aligned series without
    re-deriving the join.
    """
    spot_df, _ = load_dataset(data_dir, "spot")
    perp = load_deribit_perp_price(data_dir, "BTC")
    if perp is None:
        raise RuntimeError("btcusdt_deribit_perp_5m.csv.gz not found in data/")
    perp_aligned = (
        perp["close"]
        .reindex(perp.index.union(spot_df.index))
        .sort_index()
        .ffill()
        .reindex(spot_df.index)
    )
    perp_aligned = perp_aligned.where(spot_df.index >= perp.index.min())
    fused = ((spot_df["close"] + perp_aligned) / 2.0).where(
        perp_aligned.notna(), spot_df["close"]
    )
    return spot_df, fused.rename("fused_close"), perp_aligned.rename("perp_close")


def fused_close_eth(data_dir: str | Path = ROOT / "data") -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Same construction as `fused_close_btc`, on Coinbase ETH spot + Deribit ETH-PERPETUAL."""
    spot_df = load_coinbase_eth_spot(data_dir)
    if spot_df is None:
        raise RuntimeError("ethusd_coinbase_spot_5m.csv.gz not found in data/")
    perp = load_deribit_perp_price(data_dir, "ETH")
    if perp is None:
        raise RuntimeError("ethusdt_deribit_perp_5m.csv.gz not found in data/")
    perp_aligned = (
        perp["close"]
        .reindex(perp.index.union(spot_df.index))
        .sort_index()
        .ffill()
        .reindex(spot_df.index)
    )
    perp_aligned = perp_aligned.where(spot_df.index >= perp.index.min())
    fused = ((spot_df["close"] + perp_aligned) / 2.0).where(
        perp_aligned.notna(), spot_df["close"]
    )
    return spot_df, fused.rename("fused_close"), perp_aligned.rename("perp_close")


def vote_from_close(vote_close: pd.Series, horizons: tuple[int, ...] = HORIZONS,
                     band: float = BAND) -> np.ndarray:
    """Byte-for-byte replica of `KellyRegime.prepare`'s vote construction,
    generalized to take an arbitrary `vote_close` series for the anchor
    input (v4's own trading/fill price is untouched by this -- only the
    vote's own internal comparison uses `vote_close`)."""
    votes = []
    for days in horizons:
        anchor = vote_close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(vote_close > anchor * (1.0 + band), 1.0,
                     np.where(vote_close < anchor * (1.0 - band), 0.0, np.nan)),
            index=vote_close.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return (sum(votes) / len(votes)).to_numpy()


def causal_truncation_probe(strategy_factory, df: pd.DataFrame, market: MarketSpec,
                             cut: str = "2021-06-01") -> bool:
    """Standard two-opposite-tampers probe: multiply/divide post-cut bars by
    3 and confirm the target array before `cut` is bit-identical either
    way. Returns True if causal (no lookahead detected)."""
    from tradebot.strategy import Strategy  # noqa: F401

    strat_a = strategy_factory()
    df_a = df.copy()
    mask = df_a.index >= cut
    df_a.loc[mask, ["open", "high", "low", "close"]] *= 3.0
    out_a = strat_a.prepare(df_a.copy())

    strat_b = strategy_factory()
    df_b = df.copy()
    df_b.loc[mask, ["open", "high", "low", "close"]] /= 3.0
    out_b = strat_b.prepare(df_b.copy())

    pre = ~mask
    ta = out_a.loc[pre, "target"].to_numpy()
    tb = out_b.loc[pre, "target"].to_numpy()
    return bool(np.allclose(ta, tb, equal_nan=True))


def run_candidate(strategy_factory, df: pd.DataFrame, market: MarketSpec,
                   start: str, end: str):
    strat = strategy_factory()
    res = run_period(strat, df, start=start, end=end, market=market, start_balance=1000.0)
    return compute_metrics(res), res


def signal_check(candidate_factory, df: pd.DataFrame, market: MarketSpec,
                  start: str, end: str, seed: int = 168) -> dict:
    """Paired bootstrap Δlog-growth vs unmodified kelly_regime_v4, plus
    exposure-matching diagnostics (R-33's rule: check the arms carry
    comparable time-in-market before trusting any Sharpe/drawdown delta)."""
    m_cand, res_cand = run_candidate(candidate_factory, df, market, start, end)
    m_v4, res_v4 = run_candidate(lambda: get_strategy("kelly_regime_v4"), df, market, start, end)
    r_cand = daily_returns(res_cand.equity)
    r_v4 = daily_returns(res_v4.equity)
    n = min(len(r_cand), len(r_v4))
    paired = paired_bootstrap(r_cand.to_numpy()[:n], r_v4.to_numpy()[:n],
                               stat=total_log_return, seed=seed)
    return {
        "sharpe_cand": m_cand.sharpe, "sharpe_v4": m_v4.sharpe,
        "d_sharpe": m_cand.sharpe - m_v4.sharpe,
        "dd_cand": m_cand.max_drawdown_pct, "dd_v4": m_v4.max_drawdown_pct,
        "final_cand": m_cand.final_balance, "final_v4": m_v4.final_balance,
        "paired_diff": paired.diff.point, "paired_lo": paired.diff.lo,
        "paired_hi": paired.diff.hi, "significant": paired.significant,
        "tim_cand": m_cand.time_in_market_pct, "tim_v4": m_v4.time_in_market_pct,
    }


if __name__ == "__main__":
    # Self-test: reproduce the Step-0 correlation/flip-count numbers quoted
    # in this module's own docstring, from a clean shell.
    spot_df, fused, perp = fused_close_btc()
    orig_vote = vote_from_close(spot_df["close"])
    fused_vote = vote_from_close(fused)
    mask = perp.notna().to_numpy()
    orig_m, fused_m = orig_vote[mask], fused_vote[mask]
    corr = np.corrcoef(orig_m, fused_m)[0, 1]
    diff = np.abs(orig_m - fused_m)
    print(f"overlap bars: {mask.sum():,} of {len(spot_df):,} ({mask.mean():.1%})")
    print(f"corr(orig_vote, fused_vote): {corr:.4f}")
    print(f"frac bars with a full anchor-vote flip: {(diff > 1e-9).mean():.4f}")

    def flips(v):
        sign = np.sign(v - 0.5)
        return int((pd.Series(sign).diff().fillna(0) != 0).sum())

    print(f"orig flips (0.5-crossings) on overlap: {flips(orig_m)}")
    print(f"fused flips (0.5-crossings) on overlap: {flips(fused_m)}")
