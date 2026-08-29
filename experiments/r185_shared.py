"""Shared, read-only utilities and frozen pre-registration for the R-185 round (08-29).

DIRECTION, in one sentence: run `kelly_regime_v4`'s existing, unmodified
3-anchor VOTE construction TWICE in parallel -- once on Coinbase spot
close (as today, `frac_spot`), once on Deribit's independently-transacted
`BTC-PERPETUAL`/`ETH-PERPETUAL` close (`frac_perp`, same anchor/band/
horizon arithmetic, a different price series) -- and use DISAGREEMENT
between the two discrete vote outcomes as a measure of specification/
model uncertainty. SCALE (the conditional-volatility-target multiplier),
the 1% band, the 10% deadband and the latching hysteresis are reused
byte-for-byte from `kelly_regime_v4`/`v3`/base. Neither branch fuses the
two venues' price LEVELS into one series at any point -- that construction
(equal-weight and vol-conditioned fusion) is R-168, already closed.

**Which constraint this attacks: ERR** ("no error control anywhere in the
signal path") -- every one of this project's ~14 prior ERR-axis attempts
(ACI, meta-labeling x4, PBO/CSCV, SPRT/GROW, anytime-valid Hoeffding/
betting bounds, RCPS, online-FDR, bootstrap/HAC-PSR) is keyed on ONE
notion of uncertainty: is the vote's own historical edge distinguishable
from zero, or is a classifier's confidence in an already-computed bet
well-calibrated. This round is keyed on a genuinely different notion --
model/SPECIFICATION uncertainty, operationalized as disagreement between
two independently-transacted markets' own discrete regime reads -- which
is exactly the gap R-104's own closing line named as untried (grep
"model/specification uncertainty across the vote's own anchor choices"
in docs/LEDGER.md). Secondarily INFO, in the standing diagnosis's literal
"one price series" sense, since Deribit's perp is the one genuinely
second, independently-transacted price series this project has ever had
(R-41's framing) -- used here as a second MEASUREMENT INSTRUMENT to check
agreement, not as a second predictive signal or a price to fuse/trade on.

Grounding: Hasbrouck (1995), "One Security, Many Markets: Determining the
Contributions to Price Discovery", J. Finance 50(4), 1175-1199; Gonzalo &
Granger (1995), "Estimation of Common Long-Memory Components in
Cointegrated Systems", J. Business & Economic Statistics 13(1), 27-35; de
Jong (2002), "Measures of Contributions to Price Discovery", J. Financial
Markets 5(3) -- linked markets should impound the same information, so
disagreement is the textbook signature of a temporary breakdown in that
linkage (funding stress, leverage-driven basis dislocation), a period
when neither read should be trusted at full conviction. Bates & Granger
(1969), "The Combination of Forecasts", Operational Research Quarterly
20(4), 451-468 -- weighting independent estimators by recent MUTUAL
reliability outperforms picking one as primary, the novel branch's basis.

**Not a duplicate of:**
- R-168 (cross-venue price FUSION into the vote's anchor input, i.e.
  `fused = w*spot + (1-w)*perp` as a CONTINUOUS price-level blend, swept
  over `w`): that round found a large, clean, BTC-only effect that scaled
  SMOOTHLY and MONOTONICALLY with "how much perp" (spot-only Sharpe 0.251
  -> 50/50 fusion 1.091 -> perp-only 2.318 on BTC futures_5x inner-val),
  arguing the effect is a generic asset-specific rescale of the vote's
  dynamic range rather than a genuine agreement/disagreement signal, and
  it failed ETH replication in both its flat and vol-conditioned-weight
  branches. This round never blends price LEVELS -- it computes two
  complete, independent, DISCRETE vote outputs (`frac_spot`, `frac_perp`
  in {0, 1/3, 2/3, 1}) and acts only on their categorical (dis)agreement,
  either as a bounded never-increase haircut (conservative) or as an
  ensemble reweighting driven by trailing mutual disagreement, never by a
  fixed or vol-conditioned mixing weight (novel). Disclosed risk, named
  BEFORE either branch is dispatched: whatever makes BTC's Deribit-perp
  series special in R-168 (longer/cleaner history, deeper liquidity) may
  make this construction a BTC-only artifact too -- the frozen decision
  rule below requires BTC+ETH agreement for exactly this reason.
- R-41 conservative / R-42 (Deribit basis SPREAD MAGNITUDE, `log(perp/
  spot)`, as a continuous SCALE brake): a level-difference statistic, not
  a comparison of two independent discrete regime CALLS.
- R-100 (Binance-vs-Deribit FUNDING-RATE divergence as a 15th INFO signal
  / execution-timing brake): a different field (funding rate, not price),
  used as a lead-time signal, not a specification-uncertainty veto.
- R-146 (anchor STATISTIC: median/jump-robust mean over the ONE existing
  spot series) / R-147 (anchor COMBINATION WEIGHTS, James-Stein/Beta-
  Bernoulli, over the ONE existing spot series): both operate on a single
  price series; this round never changes the anchor statistic or the
  3-way average within a source, only which SERIES each of two parallel,
  otherwise-identical vote instances reads.
- R-40 (bagging/shrinking a vote across nearby ANCHOR-SPAN ladders on the
  SAME series, closed): a different axis of "specification" -- data-
  SOURCE disagreement between two independently-transacted markets, not
  lookback-SPAN disagreement within one market.
- R-97 (Wasserstein-DRO Kelly): a one-shot distributional-ambiguity
  radius derived from the historical regime-cycle count, not a real-time
  cross-source comparison; scoped by its own closing line to the
  "ambiguity/confidence state variable keyed on historical-edge
  dispersion" family, which this is not.

**Frozen splits** (identical to every prior round in this lineage):
inner-train `2017-01-01 -> 2020-12-31` (fit/debug only, not read for any
promotion-relevant number); inner-validation `2021-01-01 -> 2022-12-31`
(all selection, both branches); holdout `2023-01-01 ->` untouched by
both branches -- the operator decides after both branches report whether
either clears the inner-validation gate below and is worth carrying to
Step 4. Note BTC-perp coverage starts 2018-08-14 and ETH-perp 2019-03-14,
so pre-coverage bars run `frac_perp` undefined / degenerate-agreeing with
`frac_spot` by construction (disclosed, not hidden).

**Step-0 non-degeneracy check** (run BEFORE either branch was dispatched,
`python experiments/r185_shared.py`, reported here verbatim, over the
FULL each-asset overlap window, not just inner-train/val -- a non-
degeneracy check is not a promotion-relevant number): BTC, 840,843
overlap bars (83.2% of history) -- discrete-vote disagreement rate
**3.50%**, 602 disagreement episodes, mean length 48.9 bars (0.17 days),
max 587 bars (2.0 days); spot-vote and perp-vote both flip (0.5-crossing)
exactly 240 times over the overlap, so disagreement is concentrated in
NEAR-boundary anchor splits (one anchor differs) rather than wholesale
majority reversals. ETH, 781,350 overlap bars (100.0% of history, since
ETH-perp coverage starts almost exactly where ETH-spot's committed series
does) -- disagreement rate **0.24%**, an order of magnitude rarer than
BTC's, 132 episodes, mean length 14.0 bars (0.05 days). This is NOT a
Step-0 kill (BTC's rate and episode structure are clearly non-degenerate,
comparable in kind to R-168's 3.22% full-anchor-flip rate), but it is a
disclosed, named risk carried into the gate rather than found after the
fact: on ETH the mechanism will be near-inert most of the time by
construction, so a branch's ETH performance is more likely to land close
to a tie with v4 than to move decisively in either direction -- exactly
the "does not exclude zero in the losing direction" escape clause in the
gate below, not the "clears +0.20" one. A branch that passes ETH only
via this near-inert route should be reported as such, not read as a
positive ETH replication.

**Frozen inner-validation gate, per branch** (matches R-168's own gate
shape exactly, for direct comparability): worth carrying to holdout only
if, on BTC AND ETH, at the standard 0.10%/0.05% fee tier:
1. `d_sharpe >= +0.20` (R-20's noise floor) OR a drawdown/tail
   improvement of matching magnitude, in the SAME direction on both
   markets;
2. the paired-bootstrap 95% CI on delta-log-growth excludes zero on at
   least one market and does not exclude zero in the LOSING direction on
   the other;
3. exposure is matched (time-in-market within 5 points of v4's own) on
   both markets, per R-33/R-131's risk-matching discipline.
Anything else is NEGATIVE for that branch. The novel branch additionally
requires (4): it must beat a naive EQUAL-WEIGHT 6-anchor ensemble control
(fixed 50/50 spot/perp anchor blend at the vote level, zero new tunable
parameters) by the same `d_sharpe >= +0.20` / CI-excludes-zero bar on
BTC inner-validation -- isolating disagreement-WEIGHTING as the active
ingredient rather than "having two vote sources" in general (per
RESEARCH.md's own anchor-count finding: more anchors alone did not help
in the 7-48-anchor ladder sweep).

Power check: this reuses the exact statistic (paired delta-log-growth /
Sharpe), the exact bootstrap machinery, and the exact BTC-perp/ETH-perp
window R-168 already resolved cleanly on (BTC futures_5x CI
[+0.317, +0.715], excluding zero, on ~4.3 pre-2023 perp-covered years) --
so the +-0.2 Sharpe threshold's reachability at this data volume is
already demonstrated on this identical data, not merely asserted.
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


def load_spot_and_perp(asset: str = "BTC", data_dir: str | Path = ROOT / "data"):
    """Returns (spot_df, perp_close_aligned) for `asset` in {"BTC","ETH"}.

    `perp_close_aligned` is Deribit's own close, as-of/causally aligned
    onto spot's index (never a future spot bar leaking into an earlier
    perp bar), NaN before that instrument's perp coverage starts.
    """
    if asset == "BTC":
        spot_df, _ = load_dataset(data_dir, "spot")
    elif asset == "ETH":
        spot_df = load_coinbase_eth_spot(data_dir)
        if spot_df is None:
            raise RuntimeError("ethusd_coinbase_spot_5m.csv.gz not found in data/")
    else:
        raise ValueError(asset)

    perp = load_deribit_perp_price(data_dir, asset)
    if perp is None:
        raise RuntimeError(f"{asset.lower()}usdt_deribit_perp_5m.csv.gz not found in data/")
    perp_aligned = (
        perp["close"]
        .reindex(perp.index.union(spot_df.index))
        .sort_index()
        .ffill()
        .reindex(spot_df.index)
    )
    perp_aligned = perp_aligned.where(spot_df.index >= perp.index.min())
    return spot_df, perp_aligned.rename("perp_close")


def vote_from_close(vote_close: pd.Series, horizons: tuple[int, ...] = HORIZONS,
                     band: float = BAND) -> np.ndarray:
    """Byte-for-byte replica of `KellyRegime.prepare`'s vote construction,
    generalized to an arbitrary `vote_close` series (fill price for v4's
    own trading is untouched -- only the vote's internal comparison uses
    this)."""
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


def spot_and_perp_votes(asset: str = "BTC", data_dir: str | Path = ROOT / "data"):
    """Returns (spot_df, frac_spot, frac_perp, perp_available_mask).

    Where perp is unavailable (pre-coverage), `frac_perp` is defined as
    equal to `frac_spot` (agreement by construction, disclosed) so both
    branches can run over v4's full history without special-casing NaNs
    in their own strategy code; the mask lets both the Step-0 check and
    any branch measure/exclude pre-coverage bars explicitly instead.
    """
    spot_df, perp_close = load_spot_and_perp(asset, data_dir)
    frac_spot = vote_from_close(spot_df["close"])
    mask = perp_close.notna().to_numpy()
    perp_filled = perp_close.where(perp_close.notna(), spot_df["close"])
    frac_perp = vote_from_close(perp_filled)
    frac_perp = np.where(mask, frac_perp, frac_spot)
    return spot_df, frac_spot, frac_perp, mask


def causal_truncation_probe(strategy_factory, df: pd.DataFrame, cut: str = "2021-06-01") -> bool:
    """Standard two-opposite-tampers probe: multiply/divide post-cut bars by
    3 and confirm the target array before `cut` is bit-identical either
    way. Returns True if causal (no lookahead detected)."""
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


def run_candidate(strategy_factory, df: pd.DataFrame, market: MarketSpec, start: str, end: str):
    strat = strategy_factory()
    res = run_period(strat, df, start=start, end=end, market=market, start_balance=1000.0)
    return compute_metrics(res), res


def signal_check(candidate_factory, df: pd.DataFrame, market: MarketSpec,
                  start: str, end: str, seed: int = 185) -> dict:
    """Paired bootstrap delta-log-growth vs unmodified kelly_regime_v4,
    plus exposure-matching diagnostics (R-33's rule)."""
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
    # Step-0 non-degeneracy check: run BEFORE either branch is dispatched.
    for asset in ("BTC", "ETH"):
        spot_df, frac_spot, frac_perp, mask = spot_and_perp_votes(asset)
        fs, fp = frac_spot[mask], frac_perp[mask]
        agree = fs == fp
        print(f"--- {asset} ---")
        print(f"overlap bars: {mask.sum():,} of {len(spot_df):,} ({mask.mean():.1%})")
        print(f"agreement rate on overlap: {agree.mean():.4f}")
        print(f"disagreement rate on overlap: {(~agree).mean():.4f}")

        def flips(v):
            sign = np.sign(v - 0.5)
            return int((pd.Series(sign).diff().fillna(0) != 0).sum())

        print(f"spot-vote flips (0.5-crossings) on overlap: {flips(fs)}")
        print(f"perp-vote flips (0.5-crossings) on overlap: {flips(fp)}")
        # Disagreement-episode structure: mean run length in bars.
        dis = (~agree).astype(int)
        runs = []
        cur = 0
        for x in dis:
            if x:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        if runs:
            print(f"disagreement episodes: {len(runs)}, mean length "
                  f"{np.mean(runs):.1f} bars ({np.mean(runs) / BARS_PER_DAY:.2f} days), "
                  f"max {max(runs)} bars ({max(runs) / BARS_PER_DAY:.1f} days)")
        else:
            print("disagreement episodes: 0")
