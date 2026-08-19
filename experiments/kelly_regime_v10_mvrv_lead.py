#!/usr/bin/env python
"""NOVEL branch, parallel round 08-19: does real on-chain MVRV (market
value / realized value) genuinely LEAD kelly_regime_v4's own 3-anchor
vote at cycle-turn timescales (weeks-to-months, the literature's actual
claim -- Mahmudov & Puell 2018; Grobys 2026, Int. Rev. Financial Analysis,
"Using on-chain data to predict Bitcoin cycles"), or does it carry
information the vote structurally cannot see even at zero lag (a VALUE
statistic vs the vote's TREND statistic -- Asness, Moskowitz & Pedersen
2013, J. Finance, "Value and Momentum Everywhere")?

Mechanism (final, after step 2 below): an agreement/disagreement
VALUATION GATE. When the price vote is bull-leaning (>=2/3 anchors) AND
MVRV is NOT overvalued (agreement / still-room-to-run), amplify v4's
computed exposure above 1x; when the vote is bull-leaning but MVRV IS
overvalued (disagreement -- late-cycle euphoria, textbook top-formation
per Grobys 2026), dampen it. On the bear-leaning side, MVRV can only ever
dampen further (never amplify against a negative-drift vote), preserving
kelly_regime's own "stand flat rather than short a negative-drift bet"
logic (Bell & Cover 1980).

Architecturally distinct from experiments/kelly_regime_v10_mvrv_brake.py
(the parallel conservative branch, a bounded NEVER-INCREASE-ONLY dampener
mult in [1-lambda, 1] on v4's unchanged vote): this gate can move the
multiplier ABOVE 1.0 on agreement, is keyed off a genuinely different
input (an on-chain valuation LEVEL crossed against the vote's own
STATE, i.e. an interaction term, not a standalone brake), and is
symmetric in structure (though asymmetric in effect, by design, on the
bear side) rather than a pure one-directional risk cap.

Constraint attacked: INFO (one price series) -- MVRV requires blockchain
transaction history (when each coin last moved) that price alone cannot
recover. Step 2 below empirically interrogates HOW independent it really
is at the timescale this design needs, honestly, before committing.

Not a duplicate of: L-12 (Bayesian posterior used as a DIRECTION input,
which lost -- this uses MVRV purely as a SIZE modulator, the axis that
has actually worked in this project); L-14/L-15/L-16 (recovered
"information" that was provably a price transform -- step 2 below runs
the equivalent check on MVRV specifically, and takes the answer
seriously even though it is uncomfortable); R-34 (Bayesian posterior
SIZE input, real signal but too noisy at its native cadence to pay 5m
costs -- MVRV here operates at a much slower, cycle-length cadence by
construction); R-41/kelly_regime_v9_basis_lead (this file's own
template: same "ask lead-lag before writing strategy code" discipline,
different data source, different native timescale -- v9 tested days,
this tests weeks-to-months per the literature's actual claim); the
conservative brake template (see above -- never-increase-only vs this
file's two-sided agreement gate).

Usage::

    python experiments/kelly_regime_v10_mvrv_lead.py verify      # data checks
    python experiments/kelly_regime_v10_mvrv_lead.py orthogonal  # redundancy vs price
    python experiments/kelly_regime_v10_mvrv_lead.py leadlag     # step 2
    python experiments/kelly_regime_v10_mvrv_lead.py sweep       # step 3
    python experiments/kelly_regime_v10_mvrv_lead.py select      # step 3, pick + 4-cell table
    python experiments/kelly_regime_v10_mvrv_lead.py exposure    # diag (a)
    python experiments/kelly_regime_v10_mvrv_lead.py overfit     # diag (b)
    python experiments/kelly_regime_v10_mvrv_lead.py duplicate   # pre-registered decisive check
    python experiments/kelly_regime_v10_mvrv_lead.py causality   # diag (c)
    python experiments/kelly_regime_v10_mvrv_lead.py volcorr     # diag (d), R-08 trap
    python experiments/kelly_regime_v10_mvrv_lead.py eth         # falsification
    python experiments/kelly_regime_v10_mvrv_lead.py all         # everything except eth
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import (  # noqa: E402
    load_coinbase_eth_spot, load_dataset, load_onchain,
)
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

TRAIN = ("2017-01-01", "2020-12-31")     # inner-train (ROUTINE.md default)
VALID = ("2021-01-01", "2022-12-31")     # inner-validation
OOS_START = "2023-01-01"                 # never read in this file

N_EVALUATED = 0  # every distinct configuration backtested, for deflated Sharpe


# --------------------------------------------------------------------- data


def build_mvrv_dataframe(asset: str = "BTC") -> tuple[pd.DataFrame, str]:
    """Spot OHLCV (BTC canonical, or ETH Coinbase) with a causal ``mvrv``
    column ffilled on. NaN before onchain coverage; nothing back-filled."""
    if asset == "BTC":
        spot, label = load_dataset(ROOT / "data", "spot")
    elif asset == "ETH":
        spot = load_coinbase_eth_spot(ROOT / "data")
        if spot is None:
            raise FileNotFoundError("data/ethusd_coinbase_spot_5m.csv.gz not found")
        label = "real"
    else:
        raise ValueError(asset)

    onchain = load_onchain(ROOT / "data", asset)
    if onchain is None:
        raise FileNotFoundError(f"onchain data for {asset} not found")

    combined_idx = spot.index.union(onchain.index)
    mvrv_on_spot = (
        onchain["mvrv"].reindex(combined_idx).sort_index().ffill().reindex(spot.index)
    )
    mvrv_on_spot = mvrv_on_spot.where(spot.index >= onchain.index.min())
    out = spot.copy()
    out["mvrv"] = mvrv_on_spot
    return out, label


DF, LABEL = build_mvrv_dataframe("BTC")
print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
      f"(data: {LABEL}); mvrv coverage {DF['mvrv'].notna().sum():,} bars from "
      f"{DF['mvrv'].dropna().index[0]:%Y-%m-%d}", file=sys.stderr)


def ev(strategy, start, end, df=None, market=SPOT, tag="", balance=1_000.0,
       count=True):
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    print(f"  {tag or strategy.name:34s} {market.name:9s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"fills={len(result.fills):>5d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f}{'  LIQUIDATED' if m.liquidated else ''}")
    return m


# ------------------------------------------------------------ data verification


def verify() -> None:
    """Verify the loader's causal-shift claim and basic data sanity ourselves."""
    onchain_raw = pd.read_csv(ROOT / "data" / "btcusd_onchain_daily.csv.gz",
                               parse_dates=["timestamp"])
    onchain_loaded = load_onchain(ROOT / "data", "BTC")
    probe_row = onchain_raw["mvrv"].first_valid_index() + 50
    raw_first = pd.Timestamp(onchain_raw["timestamp"].iloc[probe_row])
    if raw_first.tz is None:
        raw_first = raw_first.tz_localize("UTC")
    raw_val = onchain_raw["mvrv"].iloc[probe_row]
    loaded_val = onchain_loaded["mvrv"].loc[raw_first + pd.Timedelta(days=1)]
    print(f"causal-shift check: raw row timestamped {raw_first} (mvrv={raw_val:.4f}) "
          f"appears in the loaded frame at {raw_first + pd.Timedelta(days=1)} "
          f"(mvrv={loaded_val:.4f})  "
          f"{'PASS' if abs(raw_val - loaded_val) < 1e-9 else 'FAIL'}")
    print(f"loaded index tz: {onchain_loaded.index.tz}, "
          f"monotonic: {onchain_loaded.index.is_monotonic_increasing}")

    print(f"\nBTC onchain: {len(onchain_loaded):,} rows, "
          f"{onchain_loaded.index[0]:%Y-%m-%d} -> {onchain_loaded.index[-1]:%Y-%m-%d}, "
          f"mvrv NaN={onchain_loaded['mvrv'].isna().sum()}")
    s = onchain_loaded["mvrv"].dropna()
    print(f"mvrv full-history describe:\n{s.describe()}")
    print(f"mvrv max is at {s.idxmax():%Y-%m-%d} ({s.max():.2f}) -- "
          f"the metric's own inception transient (realized cap near zero in the "
          f"first days after 2010-07-18); it is >1400 bars before our study "
          f"windows start and falls out of any rolling window <=1460d well "
          f"before 2017.")

    eth_onchain = load_onchain(ROOT / "data", "ETH")
    eth_spot = load_coinbase_eth_spot(ROOT / "data")
    print(f"\nETH onchain: {eth_onchain.index[0]:%Y-%m-%d} -> "
          f"{eth_onchain.index[-1]:%Y-%m-%d}")
    print(f"ETH coinbase spot: {eth_spot.index[0]:%Y-%m-%d} -> "
          f"{eth_spot.index[-1]:%Y-%m-%d}")
    overlap_start = max(eth_onchain.index[0], eth_spot.index[0])
    print(f"ETH overlap window (falsification test range, bounded above by "
          f"OOS_START): {overlap_start:%Y-%m-%d} -> {OOS_START}")

    # ffill-onto-5m-grid sanity: a spot-grid mvrv value must equal the most
    # recent onchain value at or before that bar, never a future one.
    spot, _ = load_dataset(ROOT / "data", "spot")
    combined_idx = spot.index.union(onchain_loaded.index)
    m5 = (onchain_loaded["mvrv"].reindex(combined_idx).sort_index().ffill()
          .reindex(spot.index))
    probe_day = pd.Timestamp("2021-06-15", tz="UTC")
    bar_before_midnight = probe_day - pd.Timedelta(minutes=5)
    bar_after_midnight = probe_day
    print(f"\n5m-grid ffill probe around {probe_day.date()} 00:00 UTC:")
    print(f"  bar {bar_before_midnight} mvrv={m5.loc[bar_before_midnight]:.4f} "
          f"(should equal {probe_day.date()-pd.Timedelta(days=1)}'s onchain row, "
          f"i.e. the shifted value from {probe_day.date()-pd.Timedelta(days=2)})")
    print(f"  bar {bar_after_midnight}     mvrv={m5.loc[bar_after_midnight]:.4f} "
          f"(should equal the row shifted onto {probe_day.date()})")
    expect_before = onchain_loaded["mvrv"].loc[probe_day - pd.Timedelta(days=1)]
    expect_after = onchain_loaded["mvrv"].loc[probe_day]
    ok = (abs(m5.loc[bar_before_midnight] - expect_before) < 1e-9
          and abs(m5.loc[bar_after_midnight] - expect_after) < 1e-9)
    print(f"  {'PASS' if ok else 'FAIL'}")


# ------------------------------------------------------- orthogonality study


def orthogonal() -> None:
    """Is MVRV's LEVEL, at the multi-month timescale this design needs,
    actually recoverable from price alone (the L-14/L-15/L-16 trap), or is
    it genuinely carrying information price doesn't have? Both a level
    check and a returns check, honestly, before any candidate is built."""
    close = DF["close"]
    mvrv = DF["mvrv"]
    daily_close = close.resample("1D").last()
    daily_mvrv = mvrv.resample("1D").last()
    df = pd.concat([daily_close.rename("close"), daily_mvrv.rename("mvrv")],
                    axis=1).dropna()
    df = df.loc[TRAIN[0]:VALID[1]]

    print("LEVEL correlation: mvrv vs price/MA(d) ratio (both causal, daily):")
    for d in (200, 365, 500, 730):
        ma = df["close"].rolling(d).mean()
        ratio = df["close"] / ma
        sub = pd.concat([ratio.rename("ratio"), df["mvrv"]], axis=1).dropna()
        r = sub["ratio"].corr(sub["mvrv"])
        print(f"  price/MA{d:>3d}d  corr(level, mvrv) = {r:>6.3f}  (n={len(sub)})"
              f"  {'-- mvrv level looks nearly redundant with a pure price ratio' if abs(r) > 0.9 else ''}")

    print("\nRETURNS correlation (day-to-day change, the genuinely-new-info axis):")
    dr = np.log(df["close"]).diff()
    dmvrv = np.log(df["mvrv"]).diff()
    sub = pd.concat([dr.rename("dr"), dmvrv.rename("dmvrv")], axis=1).dropna()
    r = sub["dr"].corr(sub["dmvrv"])
    print(f"  corr(d log price, d log mvrv) = {r:.3f}  (n={len(sub)})  -- "
          f"low here means day-to-day mvrv moves are NOT just price moves; "
          f"realized-cap only updates when dormant coins actually move on-chain")

    print("\nvote frac vs mvrv, both raw and via a 730d Z-score:")
    votes = _anchor_votes(close)
    anchor_frac = (sum(votes.values()) / len(votes)).resample("1D").last()
    roll_mean = df["mvrv"].rolling(730, min_periods=365).mean()
    roll_std = df["mvrv"].rolling(730, min_periods=365).std()
    z = (df["mvrv"] - roll_mean) / roll_std
    sub = pd.concat([anchor_frac.rename("frac"), df["mvrv"].rename("mvrv"), z.rename("z")],
                     axis=1).dropna().loc[TRAIN[0]:VALID[1]]
    print(f"  corr(vote frac, mvrv raw)      = {sub['frac'].corr(sub['mvrv']):.3f}")
    print(f"  corr(vote frac, mvrv Z(730d))  = {sub['frac'].corr(sub['z']):.3f}")
    print(f"\nHonest read: MVRV's LEVEL over the multi-month windows this design "
          f"needs correlates strongly (>0.8) with a price/MA(500-730d) ratio -- "
          f"a quantity derivable from price alone. Day-to-day CHANGES correlate "
          f"much less (~0.1-0.2). This does not settle non-duplication by itself "
          f"-- the decisive test is whether the actual candidate MECHANISM's "
          f"output (not the raw signal) is collinear with the same mechanism fed "
          f"a price-only proxy instead; see the `duplicate` command, "
          f"pre-registered below as a falsification criterion.")


# ---------------------------------------------------------------- the vote


def _anchor_votes(close: pd.Series, horizons=(20, 40, 80), band: float = 0.01):
    votes = {}
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes[days] = v.ffill().fillna(0.0)
    return votes


def _confirmed_regime_causal(side_raw: np.ndarray, min_dwell_bars: int) -> np.ndarray:
    """Debounce the raw bull/bear side (frac>=0.5 vs <0.5) so a "flip" means a
    side that actually held for >= min_dwell_bars, not 5m chop across the
    2/3-1/3 boundary. Causal: a candidate side only confirms once it has
    ALREADY held for the dwell requirement -- no lookahead into the future
    length of the run."""
    n = len(side_raw)
    out = np.empty(n, dtype=int)
    confirmed = side_raw[0]
    candidate = side_raw[0]
    cand_start = 0
    for i in range(n):
        if side_raw[i] != candidate:
            candidate = side_raw[i]
            cand_start = i
        if candidate != confirmed and (i - cand_start + 1) >= min_dwell_bars:
            confirmed = candidate
        out[i] = confirmed
    return out


def _mvrv_state_causal(mvrv: pd.Series, window_bars: int, z_in: float, z_out: float):
    """Hysteresis-latched ternary MVRV valuation state (+1 overvalued /
    -1 undervalued / 0 neutral) from a causal ROLLING (not expanding -- so
    the metric's own 2010 inception transient cannot contaminate it
    forever) Z-score. One extra bar of shift beyond the loader's own
    1-day causal shift, matching this project's belt-and-suspenders
    convention (kelly_regime_v3's vol calc, v9's basis smoothing)."""
    roll_mean = mvrv.rolling(window_bars, min_periods=window_bars // 2).mean()
    roll_std = mvrv.rolling(window_bars, min_periods=window_bars // 2).std()
    z = ((mvrv - roll_mean) / roll_std).shift(1)
    x = z.to_numpy()
    n = len(x)
    out = np.zeros(n)
    s = 0
    for i in range(n):
        xi = x[i]
        if np.isfinite(xi):
            if s == 0:
                s = 1 if xi > z_in else (-1 if xi < -z_in else 0)
            elif s == 1 and xi < z_out:
                s = 0
            elif s == -1 and xi > -z_out:
                s = 0
        out[i] = s
    return out, z


# ------------------------------------------------------------- step 2: leadlag


def _block_bootstrap(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    out = np.empty(n, dtype=x.dtype)
    pos = 0
    while pos < n:
        start = rng.integers(0, n)
        take = min(block, n - pos)
        idx = (start + np.arange(take)) % n
        out[pos:pos + take] = x[idx]
        pos += take
    return out


def _confirmed_flips(close: pd.Series, start, end, min_dwell_days: float):
    """Cycle-scale regime flips: the vote's own side (bull/bear-leaning,
    frac>=0.5 vs <0.5), debounced to require min_dwell_days of persistence
    before counting as a real flip -- filters the intraday/daily chop out
    of the raw 0.5-crossing (225 raw crossings on 2017-2022; ~10-20 after
    debouncing, which is the right order of magnitude for "cycle regime
    changes", not "every anchor wiggle")."""
    votes = _anchor_votes(close)
    frac = (sum(votes.values()) / len(votes))
    side_raw = np.where(frac.to_numpy() >= 0.5, 1, -1)
    conf = _confirmed_regime_causal(side_raw, int(min_dwell_days * BARS_PER_DAY))
    conf_s = pd.Series(conf, index=close.index).loc[start:end]
    change = conf_s.ne(conf_s.shift())
    change.iloc[0] = True
    flips = [(t, int(conf_s.loc[t])) for t in conf_s.index[change]]
    return flips, frac


def _score_leadlag(flips, state_s: pd.Series, lookback_days: int = 400):
    """For each confirmed flip, look back up to lookback_days for the most
    recent onset of the MVRV state that WOULD have predicted it (overvalued
    before a bear flip, undervalued before a bull flip). Returns per-side
    hit counts and lead times (days from onset to flip)."""
    lead_bear, lead_bull = [], []
    n_bear = n_bull = 0
    for t, side in flips:
        seg = state_s.loc[t - pd.Timedelta(days=lookback_days):t]
        if side == -1:
            n_bear += 1
            onsets = seg.index[(seg == 1) & (seg.shift(1) != 1)]
            if len(onsets):
                lead_bear.append((t - onsets[-1]).total_seconds() / 86400)
        else:
            n_bull += 1
            onsets = seg.index[(seg == -1) & (seg.shift(1) != -1)]
            if len(onsets):
                lead_bull.append((t - onsets[-1]).total_seconds() / 86400)
    return dict(n_bear=n_bear, n_bull=n_bull, lead_bear=lead_bear, lead_bull=lead_bull)


def leadlag(n_boot: int = 200, seed: int = 0) -> None:
    """Step 2 -- does MVRV genuinely precede the vote's own cycle-scale
    regime flips, at the weeks-to-months timescale the literature actually
    claims (Grobys 2026), more than a block-bootstrap null explains?

    Study window: 2017-01-01 -> 2022-12-31 (inner-train + inner-validation
    -- the holdout is never read). Reference event: a CONFIRMED (debounced,
    >=20-day-dwell) vote flip between bear-leaning and bull-leaning.
    Candidate predictor: the most recent onset, within the prior 400 days,
    of the matching MVRV valuation extreme.

    Null: moving-block bootstrap of the MVRV Z-path (90-day blocks --
    long enough to preserve MVRV's own multi-month persistence, the
    property that makes the naive "confirmed rate" look inflated) against
    the SAME fixed flip dates.
    """
    close = DF["close"].loc[:VALID[1]]
    mvrv = DF["mvrv"].loc[:VALID[1]]

    configs = [
        ("win=365d z_in=1.0", 365, 1.0, 0.5),
        ("win=365d z_in=1.5", 365, 1.5, 0.75),
        ("win=730d z_in=1.0", 730, 1.0, 0.5),
        ("win=730d z_in=1.5", 730, 1.5, 0.75),
        ("win=1460d z_in=1.0", 1460, 1.0, 0.5),
        ("win=1460d z_in=1.5", 1460, 1.5, 0.75),
    ]
    dwell_options = (14.0, 20.0, 30.0)

    print(f"leadlag study window: {TRAIN[0]} -> {VALID[1]}")
    print(f"{len(configs)} MVRV-state configs x {len(dwell_options)} dwell "
          f"filters (descriptive only, not backtests -- tracked separately "
          f"from step 3):\n")

    rng = np.random.default_rng(seed)
    for min_dwell in dwell_options:
        flips, _ = _confirmed_flips(close, TRAIN[0], VALID[1], min_dwell)
        n_bear_flips = sum(1 for _, s in flips if s == -1)
        n_bull_flips = sum(1 for _, s in flips if s == 1)
        print(f"--- min_dwell={min_dwell:g}d: {len(flips)} confirmed flips "
              f"({n_bear_flips} to bear, {n_bull_flips} to bull) ---")
        for label, win, z_in, z_out in configs:
            state, _ = _mvrv_state_causal(mvrv, int(win * BARS_PER_DAY), z_in, z_out)
            state_s = pd.Series(state, index=close.index)
            obs = _score_leadlag(flips, state_s)
            rate_bear = len(obs["lead_bear"]) / obs["n_bear"] if obs["n_bear"] else float("nan")
            rate_bull = len(obs["lead_bull"]) / obs["n_bull"] if obs["n_bull"] else float("nan")
            med_bear = float(np.median(obs["lead_bear"])) if obs["lead_bear"] else float("nan")
            med_bull = float(np.median(obs["lead_bull"])) if obs["lead_bull"] else float("nan")
            print(f"  [{label}] bear: {len(obs['lead_bear'])}/{obs['n_bear']} confirmed "
                  f"({rate_bear:>5.1%}), median lead={med_bear:>6.1f}d   |   "
                  f"bull: {len(obs['lead_bull'])}/{obs['n_bull']} confirmed "
                  f"({rate_bull:>5.1%}), median lead={med_bull:>6.1f}d")

        # null test at ONE representative config per dwell level, since n is
        # already tiny (this is the config the sweep below actually seeds from)
        label, win, z_in, z_out = configs[2]  # win=730d z_in=1.0
        state, z = _mvrv_state_causal(mvrv, int(win * BARS_PER_DAY), z_in, z_out)
        state_s = pd.Series(state, index=close.index)
        obs = _score_leadlag(flips, state_s)
        obs_rate_bear = len(obs["lead_bear"]) / obs["n_bear"] if obs["n_bear"] else float("nan")
        obs_rate_bull = len(obs["lead_bull"]) / obs["n_bull"] if obs["n_bull"] else float("nan")
        z_arr = z.to_numpy()
        z_idx = z.index
        block = 90 * BARS_PER_DAY
        null_bear, null_bull = [], []
        for _ in range(n_boot):
            z_shuf = pd.Series(_block_bootstrap(np.nan_to_num(z_arr, nan=0.0), block, rng),
                                index=z_idx)
            x = z_shuf.to_numpy()
            n = len(x)
            out = np.zeros(n)
            s = 0
            for i in range(n):
                if s == 0:
                    s = 1 if x[i] > z_in else (-1 if x[i] < -z_in else 0)
                elif s == 1 and x[i] < z_out:
                    s = 0
                elif s == -1 and x[i] > -z_out:
                    s = 0
                out[i] = s
            state_shuf = pd.Series(out, index=z_idx)
            r = _score_leadlag(flips, state_shuf)
            if r["n_bear"]:
                null_bear.append(len(r["lead_bear"]) / r["n_bear"])
            if r["n_bull"]:
                null_bull.append(len(r["lead_bull"]) / r["n_bull"])
        null_bear = np.array(null_bear) if null_bear else np.array([np.nan])
        null_bull = np.array(null_bull) if null_bull else np.array([np.nan])
        pctl_bear = float((null_bear < obs_rate_bear).mean()) if np.isfinite(obs_rate_bear) else float("nan")
        pctl_bull = float((null_bull < obs_rate_bull).mean()) if np.isfinite(obs_rate_bull) else float("nan")
        print(f"  null test ({n_boot} block-bootstrap resamples of MVRV Z, "
              f"90-day blocks, config=[{label}]):")
        print(f"    bear: observed confirm-rate={obs_rate_bear:>5.1%}  "
              f"null mean={np.nanmean(null_bear):>5.1%} sd={np.nanstd(null_bear):.3f}  "
              f"observed exceeds null in {pctl_bear:>5.1%} of resamples "
              f"(n_bear_flips={obs['n_bear']})")
        print(f"    bull: observed confirm-rate={obs_rate_bull:>5.1%}  "
              f"null mean={np.nanmean(null_bull):>5.1%} sd={np.nanstd(null_bull):.3f}  "
              f"observed exceeds null in {pctl_bull:>5.1%} of resamples "
              f"(n_bull_flips={obs['n_bull']})")
        print()

    print(f"lead-lag configurations evaluated (descriptive, not counted toward "
          f"step-3 trials): {len(configs)} MVRV configs x {len(dwell_options)} "
          f"dwell filters = {len(configs) * len(dwell_options)}")


# --------------------------------------------------- the candidate strategy


class KellyRegimeV10MvrvLead(KellyRegimeV3):
    """Agreement/disagreement MVRV valuation gate on top of v4's unchanged
    vote and conditional-vol sizer.

    Mechanism, one sentence: when the price vote is bull-leaning and MVRV
    is NOT overvalued, amplify v4's computed exposure (agreement -- Kelly
    logic says the crowd is still accumulating AND valuation hasn't caught
    up, the strongest form of the accumulation story Bell & Cover 1980 /
    Cardaliaguet & Lehalle 2018 already use); when the vote is
    bull-leaning but MVRV IS overvalued, dampen (disagreement -- late-
    cycle euphoria, the textbook top-formation signature in Grobys 2026);
    on the bear-leaning side MVRV can only ever dampen further, never
    amplify against a negative-drift vote, preserving v4's refusal to
    short.

    Sizing (conditional vol targeting, deadband, cap) is v3/v4's,
    unchanged; only a post-hoc multiplier is applied to the resulting
    ``frac * scale`` product before the deadband/position-hold loop.
    """

    name = "kelly_regime_v10_mvrv_lead"

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 mvrv_window_days: float = 730.0, mvrv_z_in: float = 1.0,
                 mvrv_z_out: float = 0.5, boost: float = 0.20,
                 damp_bull: float = 0.40, damp_bear: float = 0.20,
                 use_proxy: bool = False, **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.mvrv_window_days = mvrv_window_days
        self.mvrv_z_in = mvrv_z_in
        self.mvrv_z_out = mvrv_z_out
        self.boost = boost
        self.damp_bull = damp_bull
        self.damp_bear = damp_bear
        self.use_proxy = use_proxy  # duplicate-check: feed a price/MA proxy instead
        self.warmup = max(80 * BARS_PER_DAY + 10,
                          int(mvrv_window_days * BARS_PER_DAY) + 10)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = _anchor_votes(close, self.horizons, self.band)
        frac = (sum(votes.values()) / len(votes)).to_numpy()

        if self.use_proxy:
            # pre-registered duplicate check: identical mechanism, MVRV
            # replaced by a causal price/MA(same window) ratio
            ma = close.rolling(int(self.mvrv_window_days * BARS_PER_DAY)).mean()
            source = (close / ma).rename("proxy")
        elif "mvrv" in df.columns:
            source = df["mvrv"]
        else:
            source = None

        if source is not None:
            state, _ = _mvrv_state_causal(
                source, int(self.mvrv_window_days * BARS_PER_DAY),
                self.mvrv_z_in, self.mvrv_z_out)
        else:
            state = np.zeros(len(df))

        bull_leaning = frac >= 0.5
        overvalued = state == 1.0
        undervalued = state == -1.0

        mult = np.ones(len(df))
        mult = np.where(bull_leaning & undervalued, 1.0 + self.boost, mult)
        mult = np.where(bull_leaning & overvalued, 1.0 - self.damp_bull, mult)
        mult = np.where(~bull_leaning & overvalued, 1.0 - self.damp_bear, mult)
        mult = np.clip(mult, 1.0 - max(self.damp_bull, self.damp_bear), 1.0 + self.boost)

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

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        vstate = 0
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if vstate == 0:
                    vstate = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif vstate == 1 and x < self.high_out:
                    vstate = 0
                elif vstate == -1 and x > self.low_out:
                    vstate = 0
            scale = full[i] if vstate != 0 else steady[i]
            desired = frac[i] * scale * mult[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["v10_mult"] = mult
        df["v10_state"] = state
        return df


# --------------------------------------------------------------- step 3: sweep


def _grid():
    out = []
    for win in (365.0, 730.0, 1460.0):
        for z_in in (1.0, 1.75):
            for boost in (0.15, 0.30):
                for damp_bull in (0.35, 0.55):
                    out.append((
                        f"win={win:g}d z_in={z_in:g} boost={boost:g} damp={damp_bull:g}",
                        dict(mvrv_window_days=win, mvrv_z_in=z_in, mvrv_z_out=z_in * 0.5,
                             boost=boost, damp_bull=damp_bull, damp_bear=damp_bull * 0.5)))
    return out


def _benchmarks(start, end, market, label):
    print(f"\n{label} benchmarks:")
    for name in ("buy_and_hold", "kelly_regime_v4"):
        ev(get_strategy(name), start, end, market=market, tag=f"  {name}", count=False)


def sweep() -> None:
    grid = _grid()
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN, "INNER-TRAIN"), (VALID, "INNER-VALIDATION")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            print(f"{split} / {mname} candidate configurations:")
            for tag, kw in grid:
                ev(KellyRegimeV10MvrvLead(**kw), start, end, market=market, tag=tag)
    print(f"\ndistinct configurations in grid: {len(grid)}")
    print(f"configurations evaluated (backtests) in this run: {N_EVALUATED}")


def select():
    grid = _grid()
    best_tag, best_kw, best_sharpe = None, None, -1e9
    for tag, kw in grid:
        m = ev(KellyRegimeV10MvrvLead(**kw), *VALID, market=SPOT, tag=f"(scan) {tag}")
        if m.sharpe > best_sharpe:
            best_tag, best_kw, best_sharpe = tag, kw, m.sharpe
    print(f"\nbest inner-validation spot Sharpe: {best_tag}  ({best_sharpe:.2f})\n")
    print("full 4-cell table for the selected candidate:")
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN, "TRAIN"), (VALID, "VALID")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            ev(KellyRegimeV10MvrvLead(**best_kw), start, end, market=market,
               tag=f"  candidate ({best_tag})", count=False)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    return best_tag, best_kw


DEFAULT_KW = dict(mvrv_window_days=730.0, mvrv_z_in=1.0, mvrv_z_out=0.5,
                   boost=0.20, damp_bull=0.40, damp_bear=0.20)


# ------------------------------------------------------------------ diagnostics


def exposure_artifact_check(kw: dict | None = None) -> None:
    """Diagnostic (a): candidate's target vs a mean-notional-matched flat
    rescale of v4's own target, both markets, R^2 > 0.95 -> "just a rescale"."""
    kw = kw or DEFAULT_KW
    v4 = get_strategy("kelly_regime_v4")
    cand = KellyRegimeV10MvrvLead(**kw)
    print(f"exposure-artifact check, candidate={kw}")
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        lo = int(DF.index.searchsorted(VALID[0]))
        hi = int(DF.index.searchsorted(VALID[1], side="right"))
        prefix = min(lo, max(cand.warmup, v4.warmup))
        frame = DF.iloc[lo - prefix:hi]

        v4_prepared = v4.prepare(frame.copy())
        cand_prepared = cand.prepare(frame.copy())
        v4_t = v4_prepared["target"].to_numpy()[prefix:]
        cand_t = cand_prepared["target"].to_numpy()[prefix:]

        mean_abs_v4 = np.mean(np.abs(v4_t))
        mean_abs_cand = np.mean(np.abs(cand_t))
        alpha = mean_abs_cand / mean_abs_v4 if mean_abs_v4 > 0 else 0.0
        rescaled = alpha * v4_t
        ss_res = np.sum((cand_t - rescaled) ** 2)
        ss_tot = np.sum((cand_t - cand_t.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        corr = np.corrcoef(cand_t, v4_t)[0, 1]
        print(f"  {mname:9s} mean|v4|={mean_abs_v4:.3f} mean|cand|={mean_abs_cand:.3f} "
              f"alpha={alpha:.3f}  R^2(cand vs alpha*v4)={r2:.3f}  raw corr={corr:.3f}  "
              f"{'JUST A RESCALE' if r2 > 0.95 else 'genuinely different exposure shape'}")


def duplicate_check(kw: dict | None = None) -> None:
    """PRE-REGISTERED decisive falsification test: is the candidate's
    target series, fed real MVRV, collinear (R^2>0.95) with the IDENTICAL
    mechanism fed a price/MA(same window) proxy instead of MVRV? This
    directly operationalizes the `orthogonal` command's finding that
    MVRV's LEVEL correlates heavily with a price-only ratio at these
    timescales -- if the resulting TRADING mechanism is collinear too,
    treat it as an L-14-style price-transform duplicate regardless of
    backtest performance."""
    kw = kw or DEFAULT_KW
    print(f"duplicate check (MVRV-driven vs price-proxy-driven target), "
          f"candidate={kw}")
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN, "TRAIN"), (VALID, "VALID")):
            lo = int(DF.index.searchsorted(start))
            hi = int(DF.index.searchsorted(end, side="right"))
            cand_real = KellyRegimeV10MvrvLead(**kw)
            cand_proxy = KellyRegimeV10MvrvLead(use_proxy=True, **kw)
            prefix = min(lo, max(cand_real.warmup, cand_proxy.warmup))
            frame = DF.iloc[lo - prefix:hi]

            real_t = cand_real.prepare(frame.copy())["target"].to_numpy()[prefix:]
            proxy_t = cand_proxy.prepare(frame.copy())["target"].to_numpy()[prefix:]

            ss_res = np.sum((real_t - proxy_t) ** 2)
            ss_tot = np.sum((real_t - real_t.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            corr = (np.corrcoef(real_t, proxy_t)[0, 1]
                    if np.std(real_t) > 0 and np.std(proxy_t) > 0 else float("nan"))
            print(f"  [{split}/{mname}] R^2(mvrv-target vs proxy-target)={r2:.3f}  "
                  f"corr={corr:.3f}  "
                  f"{'DUPLICATE -- price alone reproduces this mechanism' if r2 > 0.95 else 'genuinely different'}")


def overfit_signature_check(kw: dict | None = None) -> None:
    """Diagnostic (b): candidate vs v4 on BOTH train and validation, both
    markets -- the R-37/R-38/R-40 signature check (wins only on 2021-22,
    loses on the earlier control -> suspect)."""
    kw = kw or DEFAULT_KW
    print(f"overfitting-signature check, candidate={kw}")
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN, "TRAIN (control)"), (VALID, "VALIDATION")):
            v4m = ev(get_strategy("kelly_regime_v4"), start, end, market=market,
                     tag=f"  v4  [{split}/{mname}]", count=False)
            cm = ev(KellyRegimeV10MvrvLead(**kw), start, end, market=market,
                    tag=f"  cand[{split}/{mname}]", count=False)
            ratio = cm.final_balance / v4m.final_balance if v4m.final_balance else float("nan")
            print(f"    -> candidate/v4 final-balance ratio: {ratio:.3f}x")


def volcorr_check(kw: dict | None = None) -> None:
    """Diagnostic (d), the R-08 trap: is the multiplier secretly a
    volatility timer (correlated with realized vol) rather than a
    valuation signal? BTC has an inverse leverage effect -- a mechanism
    that de-levers into high-vol states would look good on paper here and
    be sign-inverted in reality."""
    kw = kw or DEFAULT_KW
    cand = KellyRegimeV10MvrvLead(**kw)
    lo = int(DF.index.searchsorted(TRAIN[0]))
    hi = int(DF.index.searchsorted(VALID[1], side="right"))
    prefix = min(lo, cand.warmup)
    frame = DF.iloc[lo - prefix:hi]
    prepared = cand.prepare(frame.copy())
    r = np.log(prepared["close"]).diff()
    realized_vol = (r.ewm(span=8 * BARS_PER_DAY, min_periods=BARS_PER_DAY).std()
                    * np.sqrt(BARS_PER_YEAR))
    mult = prepared["v10_mult"].to_numpy()[prefix:]
    vol = realized_vol.to_numpy()[prefix:]
    mask = np.isfinite(mult) & np.isfinite(vol)
    corr = np.corrcoef(mult[mask], vol[mask])[0, 1]
    print(f"corr(v10_mult, realized_vol) = {corr:.3f} over {TRAIN[0]}..{VALID[1]}  "
          f"{'WARNING: looks like a vol timer (R-08 trap)' if abs(corr) > 0.3 else 'not a vol timer by this measure'}")
    # also: mean forward vol conditional on mult>1 vs mult<1 (does the gate
    # systematically move exposure the "wrong" direction relative to R-10's
    # finding that high vol carries the HIGHEST forward Sharpe here?)
    fwd_ret = r.shift(-1).rolling(5 * BARS_PER_DAY).sum().shift(-5 * BARS_PER_DAY)
    fwd = fwd_ret.to_numpy()[prefix:]
    mask2 = np.isfinite(mult) & np.isfinite(fwd)
    boosted = mult[mask2] > 1.0
    damped = mult[mask2] < 1.0
    print(f"mean 5d-forward log return | mult>1 (boosted): "
          f"{np.nanmean(fwd[mask2][boosted]):+.5f}  n={boosted.sum()}")
    print(f"mean 5d-forward log return | mult<1 (damped):  "
          f"{np.nanmean(fwd[mask2][damped]):+.5f}  n={damped.sum()}")


def causality(kw: dict | None = None) -> None:
    """Diagnostic (c): two-opposite-tampers probe, strictly pre-2023,
    PLUS an explicit MVRV-tamper (the new pathway this session adds)."""
    from tradebot.broker import PaperBroker

    kw = kw or DEFAULT_KW

    pre2023 = DF[DF.index < OOS_START]
    df = pre2023.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    def run_probe(name, tamper_fn):
        up, down = df.copy(), df.copy()
        tamper_fn(up, down)

        def decisions(frame):
            s = KellyRegimeV10MvrvLead(**kw)
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        a, b = decisions(up), decisions(down)
        bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
        print(f"[{name}] tampered from bar {cut:,} of {len(df):,}; checked bars {bars}")
        print("  FAIL - reads the future at bars " + str(bad) if bad
              else "  PASS - every decision at or before the cut is unchanged")

        pa = KellyRegimeV10MvrvLead(**kw).prepare(up.copy())
        pb = KellyRegimeV10MvrvLead(**kw).prepare(down.copy())
        for col in ("target", "v10_mult", "v10_state"):
            diff = np.abs(pa[col].to_numpy()[:cut].astype(float)
                          - pb[col].to_numpy()[:cut].astype(float))
            worst = float(np.nanmax(diff))
            print(f"    column {col:16s} max |difference| before the cut = {worst:.3e}"
                  f"  {'PASS' if worst < 1e-12 else 'FAIL'}")

    def tamper_ohlcv(up, down):
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def tamper_mvrv(up, down):
        mcol = up.columns.get_loc("mvrv")
        up.iloc[cut:, mcol] = 20.0    # extreme, sustained "overvalued" reading
        down.iloc[cut:, mcol] = 0.1   # extreme, sustained "undervalued" reading

    def tamper_both(up, down):
        tamper_ohlcv(up, down)
        tamper_mvrv(up, down)

    run_probe("OHLCV tamper (standard)", tamper_ohlcv)
    run_probe("MVRV tamper (the new pathway)", tamper_mvrv)
    run_probe("both at once", tamper_both)


def eth(kw: dict | None = None) -> None:
    """Falsification test: does the candidate survive on ETH? Uses the
    real Coinbase ETH spot series and real ETH on-chain MVRV, confirming
    their overlap window first, restricted to pre-2023."""
    kw = kw or DEFAULT_KW
    eth_df, eth_label = build_mvrv_dataframe("ETH")
    onchain = load_onchain(ROOT / "data", "ETH")
    overlap_start = max(eth_df.index[0], onchain.index[0])
    print(f"ETH data: {eth_label}, {eth_df.index[0]:%Y-%m-%d} -> "
          f"{eth_df.index[-1]:%Y-%m-%d}; mvrv coverage from "
          f"{eth_df['mvrv'].dropna().index[0]:%Y-%m-%d}")
    print(f"overlap window: {overlap_start:%Y-%m-%d} -> {OOS_START} (pre-holdout only)\n")

    eth_start = str(eth_df["mvrv"].dropna().index[0].date())
    eth_end = "2022-12-31"

    global DF, LABEL
    saved_df, saved_label = DF, LABEL
    DF, LABEL = eth_df, eth_label
    try:
        for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
            _benchmarks(eth_start, eth_end, market, f"ETH {eth_start}..{eth_end} / {mname}")
            ev(KellyRegimeV10MvrvLead(**kw), eth_start, eth_end, market=market,
               tag=f"  candidate", count=False)
    finally:
        DF, LABEL = saved_df, saved_label


# --------------------------------------------------------------------- driver


def all_checks() -> None:
    print("=" * 78); print("DATA VERIFICATION"); print("=" * 78)
    verify()
    print("\n" + "=" * 78); print("ORTHOGONALITY / REDUNDANCY STUDY"); print("=" * 78)
    orthogonal()
    print("\n" + "=" * 78); print("STEP 2 -- lead-lag study"); print("=" * 78)
    leadlag()
    print("\n" + "=" * 78); print("STEP 3 -- sweep"); print("=" * 78)
    sweep()
    print("\n" + "=" * 78); print("DIAGNOSTIC (a) -- exposure-artifact check"); print("=" * 78)
    exposure_artifact_check()
    print("\n" + "=" * 78); print("PRE-REGISTERED DUPLICATE CHECK"); print("=" * 78)
    duplicate_check()
    print("\n" + "=" * 78); print("DIAGNOSTIC (b) -- overfitting-signature check"); print("=" * 78)
    overfit_signature_check()
    print("\n" + "=" * 78); print("DIAGNOSTIC (c) -- causality / no-lookahead"); print("=" * 78)
    causality()
    print("\n" + "=" * 78); print("DIAGNOSTIC (d) -- vol-correlation (R-08 trap)"); print("=" * 78)
    volcorr_check()
    print(f"\ntotal configurations evaluated (backtests, step 3 + diagnostics): "
          f"{N_EVALUATED}")


if __name__ == "__main__":
    cmds = {"verify": verify, "orthogonal": orthogonal, "leadlag": leadlag,
            "sweep": sweep, "select": select, "exposure": exposure_artifact_check,
            "duplicate": duplicate_check, "overfit": overfit_signature_check,
            "causality": causality, "volcorr": volcorr_check, "eth": eth,
            "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/kelly_regime_v10_mvrv_lead.py [{'|'.join(cmds)}]")
