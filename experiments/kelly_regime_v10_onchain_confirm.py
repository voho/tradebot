#!/usr/bin/env python
"""kelly_regime_v4 with a bounded, symmetric on-chain participation-confirmation multiplier (CONSERVATIVE branch, R-44).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

The idea
--------
Genuine on-chain (blockchain-level, not price-derived) daily metrics are
committed for the first time this round: BTC/ETH active addresses,
transaction count and hash rate from CoinMetrics' free community API
(``data/btc_onchain_daily.csv.gz``, ``data/eth_onchain_daily.csv.gz``,
2017-01-01 -> present, zero missing values). This is the first genuinely
price-independent information channel in the whole project -- every prior
feature, including R-41's real Deribit basis, is derived from a market
price series.

Mechanism, one sentence: v4's vote and conditional-vol-target scale are
reproduced byte-for-byte; on top, a bounded, SYMMETRIC multiplier
``mult = 1 + lam * tanh(z / z_scale) in [1-lam, 1+lam]`` raises exposure
when active-address growth is confirming the trend (participation
z-score positive -- broadening genuine usage) and lowers it when growth is
flat or negative (a thinner, less-confirmed rally), applied on top of v4's
unchanged target. ``z`` is a rolling z-score of 7-day log active-address
growth against its own trailing distribution, causally lagged one full day
past CoinMetrics' own reporting delay via ``tradebot.data.align_onchain_causal``.

Why this must be able to raise exposure, not just shrink it
-------------------------------------------------------------
R-08 (LEDGER.md) found a genuinely *better volatility forecast* fed into
this strategy family made results WORSE ($52K vs $115K) because it
de-levered more promptly into BTC's high-volatility, high-forward-Sharpe
states (Baur & Dimpfl 2018's inverse leverage effect). Backlog item B-07
names the exact trap this round is built to avoid: on-chain flows predict
*volatility*, not direction, so any modifier that reduces exposure when
activity rises will likely repeat R-08's sign-inversion failure. This
mechanism is deliberately NOT that: it reads on-chain activity as evidence
about whether the price vote's implied crowd participation is genuine or
thin, and moves exposure both ways around 1.0x symmetrically -- never a
"reduce when active" rule. R-34's conservative branch (``kelly_regime_v5_damp.py``,
``mult in [1-lam, 1]``) is the architectural cautionary tale here too: a
bounded, never-increase-only multiplier collapsed into a flat rescale of
v4 in disguise (R^2=0.997). Symmetry is what keeps this branch out of that
trap by construction, checked directly below rather than assumed.

Constraint attacked
--------------------
INFO -- one price series is the project's #1 standing-diagnosis
constraint, and this is the first attack on it that is not itself derived
from OHLCV (unlike L-12/L-14/L-15/L-16, all four INFO-labelled failures in
section A, which tried to recover missing information FROM PRICE) and not
a second, independently-transacted PRICE series either (R-41's basis
brake). Blockchain activity is orthogonal both to price and to the
exchange-observed basis/funding signals this project has already tried
and ruled out (R-35, R-39, R-41).

Not a duplicate of
-------------------
- R-34 conservative (``kelly_regime_v5_damp.py``): same ``prepare()``
  architecture (v4's vote/scale untouched, a single bounded multiplier on
  top) but that multiplier was ``mult in [1-lam, 1]`` (never-increase) fed
  by a smoothed transform of the SAME OHLCV close series the vote already
  reads -- and it collapsed to a flat rescale (R^2=0.997). This file's
  multiplier is symmetric (``[1-lam, 1+lam]``) and fed by a genuinely new,
  price-independent data source (blockchain address activity), checked for
  the identical failure mode below rather than assumed clear of it.
- R-41 conservative (``kelly_regime_v9_basis_brake.py``): same house style
  (bounded multiplier, byte-for-byte v3/v4 core, sweep/select/artifact/
  causality/eth harness) but a DIFFERENT constraint (a second PRICE series,
  not price-independent) and a DIFFERENT architecture (never-increase-only
  brake, not symmetric).
- R-42/R-43 (dual-asset diversification): those hold capital in a SECOND
  ASSET (ETH) to diversify; this file trades BTC alone and adds a new
  INFORMATION channel about the same asset. Orthogonal questions.
- A sibling agent runs a structurally different on-chain signal (Hash
  Ribbons miner-capitulation, combined via the vote rather than a
  multiplier) in parallel this round, on a disjoint file. Not read or
  coordinated with here, per ROUTINE.md's parallelism rules.

Causality
---------
``align_onchain_causal`` is already causal by construction (a metric dated
day D only becomes visible starting D+1 00:00 UTC -- see its own docstring
in ``tradebot/data.py``). This file computes the rolling growth/z-score on
the RAW daily on-chain frame first (``growth = log(AdrActCnt_t /
AdrActCnt_{t-7d})``, both the ``shift`` and the trailing ``rolling`` window
strictly backward-looking) and only THEN calls ``align_onchain_causal`` to
project the already-causal daily z-score onto the 5m bar grid -- so the
CoinMetrics reporting lag is never re-derived, only reused, per the task
brief. Two independent two-opposite-tampers probes are run: the standard
PRICE probe (bars after a cut multiplied/divided by 3, copied from
``kelly_regime_v9_basis_brake.py``'s own procedure) and a second, new-to-
this-file probe that tampers the raw ON-CHAIN metric itself after the same
cut -- because the price probe alone cannot exercise the one new
ingredient this file adds (the on-chain frame is loaded independently of
``df`` and injected via a constructor argument, exactly as R-41's ``basis``
argument was, precisely so this second probe is possible).

Usage
-----
    python experiments/kelly_regime_v10_onchain_confirm.py sweep       # step 3, inner-train
    python experiments/kelly_regime_v10_onchain_confirm.py select      # step 5, inner-validation, both markets
    python experiments/kelly_regime_v10_onchain_confirm.py artifact    # exposure-artifact check (R-33/R-34)
    python experiments/kelly_regime_v10_onchain_confirm.py causality   # two-opposite-tampers, price + on-chain
    python experiments/kelly_regime_v10_onchain_confirm.py eth         # pre-registered ETH falsification
    python experiments/kelly_regime_v10_onchain_confirm.py all         # everything above, in order
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import (  # noqa: E402
    align_onchain_causal,
    load_dataset,
    load_ohlcv_csv,
    load_onchain_metrics,
)
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


# --------------------------------------------------------------------- on-chain z-score

def _participation_z(onchain: pd.DataFrame, bars: pd.DataFrame,
                      z_window_days: int, growth_days: int) -> pd.Series:
    """Causal rolling z-score of active-address growth, aligned onto ``bars``' index.

    ``growth_t = log(AdrActCnt_t / AdrActCnt_{t-growth_days})`` -- a
    trailing multi-day growth rate rather than day-over-day, deliberately:
    on-chain address activity has a strong day-of-week seasonality
    (weekday >> weekend), and a 7-day (default) rate cancels that pattern
    instead of injecting it into a signal meant to move smoothly. ``z`` is
    ``growth`` standardized against its own trailing ``z_window_days``-day
    mean/std (rolling, ``min_periods=z_window_days`` -- no z-score is
    reported until a full window of history exists, so there is no
    partial-window lookahead-by-instability). Both the ``shift`` and the
    ``rolling`` window are strictly backward-looking: this function is
    causal on the DAILY frame before ``align_onchain_causal`` ever runs,
    which then adds CoinMetrics' own one-day reporting lag on top -- two
    independent causal safeguards, not one re-derived from the other.
    """
    addr = onchain["AdrActCnt"].astype(float)
    growth = np.log(addr / addr.shift(growth_days))
    mean = growth.rolling(z_window_days, min_periods=z_window_days).mean()
    std = growth.rolling(z_window_days, min_periods=z_window_days).std()
    z = (growth - mean) / std.where(std > 0)
    aligned = align_onchain_causal(pd.DataFrame({"z": z}), bars)
    return aligned["z"]


# --------------------------------------------------------------------- strategy


class KellyRegimeV10OnchainConfirm(Strategy):
    """v4's vote + conditional vol-targeting exposure, scaled by a bounded, symmetric on-chain confirmation multiplier.

    See module docstring for the full mechanism. Defaults for every
    v4-inherited parameter match ``kelly_regime_v4`` exactly; ``lam``,
    ``z_window_days``, ``growth_days`` and ``z_scale`` are the only new
    knobs, and only ``lam`` and ``z_window_days`` are swept (``growth_days``
    and ``z_scale`` are fixed, documented design choices -- see module and
    ``_participation_z`` docstrings -- to keep this branch's search grid
    literal and minimal, per its own brief). ``onchain`` is injected (the
    raw daily CoinMetrics frame) rather than hardwired to the BTC default,
    purely so the causality probe and the ETH falsification test can swap
    it for a tampered or a different-asset frame; every value used is
    still computed inside ``prepare`` from ``df.index``, nothing is
    precomputed and reused across instances.
    """

    name = "kelly_regime_v10_onchain_confirm"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85,
                 lam: float = 0.15, z_window_days: int = 180,
                 growth_days: int = 7, z_scale: float = 2.0,
                 onchain: pd.DataFrame | None = None) -> None:
        # ---- identical to kelly_regime / v3 / v4 -------------------------
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        # ---- new: the on-chain confirmation multiplier ---------------------
        self.lam = lam                        # mult in [1-lam, 1+lam]; 0 = exact v4
        self.z_window_days = z_window_days    # trailing days for the z-score baseline
        self.growth_days = growth_days        # fixed at 7 (weekly growth), not swept
        self.z_scale = z_scale                # fixed tanh scale, not swept
        self.onchain = onchain if onchain is not None else ONCHAIN_BTC

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # ---- byte-for-byte v3/v4: latched multi-anchor vote -> frac ------
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

        # ---- byte-for-byte v3/v4: conditional vol-targeting scale --------
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

        # ---- new: the on-chain confirmation multiplier ---------------------
        raw_z = _participation_z(self.onchain, df, self.z_window_days,
                                  self.growth_days).reindex(df.index).to_numpy(dtype=float)
        z_used = np.where(np.isfinite(raw_z), raw_z, 0.0)  # no data yet -> z=0 -> mult=1 (fallback)
        mult = 1.0 + self.lam * np.tanh(z_used / self.z_scale)  # in [1-lam, 1+lam], symmetric

        # ---- single causal forward pass: byte-for-byte v3/v4 breakout
        # hysteresis on the vol-targeting state, plus the new multiplier ---
        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal vol band, +1 high-vol breakout, -1 low-vol breakout
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
            desired = frac[i] * mult[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["_frac"] = frac
        df["_mult"] = mult
        df["_z_raw"] = raw_z
        return df

    def on_bar(self, ctx: Context) -> None:
        # Identical execution pattern to kelly_regime.KellyRegime.on_bar:
        # signal at bar close, fill at next open via order_notional.
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures


# ------------------------------------------------------------------------ harness

DF, LABEL = load_dataset(ROOT / "data", "spot")
ONCHAIN_BTC = load_onchain_metrics(ROOT / "data", asset="BTC")
if ONCHAIN_BTC is None:
    raise RuntimeError("data/btc_onchain_daily.csv.gz not found -- cannot run this experiment")

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

# BTC on-chain data starts exactly 2017-01-01, same day the committed
# canonical spot series (btcusd_spot_5m.csv.gz, Bitstamp) starts -- so,
# unlike R-41's basis brake (which needed a shifted inner-train start
# because Deribit coverage begins 2018-08-14), the standard ROUTINE.md
# inner-train/inner-validation split applies unmodified.
TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")

INCUMBENT = "kelly_regime_v4"

# ---- sweep grid: fixed a-priori choices, not fit to inner-validation ----
# z_window candidates span roughly a quarter, half-year and full year of
# trailing history for the growth baseline -- short enough to adapt to a
# changing on-chain regime, long enough that "full window" (the min_periods
# requirement) is reached well inside inner-train. lam candidates are the
# 0.10-0.20 range the task brief names, bracketing the R-34 conservative
# branch's own selected 0.3 from below (this multiplier is symmetric, so a
# given lam moves exposure both further and less far than an equivalent
# never-increase-only lam would).
Z_WINDOWS = (90, 180, 365)
LAM = (0.10, 0.15, 0.20)
PRIMARY = dict(z_window_days=180, lam=0.15)  # grid midpoint, used for causality/eth defaults

N_EVALUATED = 0  # distinct configurations searched (routine's trials count)
_SEEN_CONFIGS: set[tuple] = set()

OUT = ROOT / "reports" / "kelly_regime_v10_onchain_confirm"


def mean_notional(result) -> float:
    if "target" not in result.df:
        return float("nan")
    tgt = np.abs(result.df["target"].to_numpy(dtype=float))
    return float(np.mean(np.clip(tgt, 0.0, result.market.leverage)))


def realized_vol(equity) -> float:
    eq = equity.to_numpy(dtype=float) if hasattr(equity, "to_numpy") else np.asarray(equity)
    if len(eq) < 3:
        return float("nan")
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
    return float(rets.std(ddof=1) * np.sqrt(BARS_PER_YEAR))


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0,
            count_key: tuple | None = None):
    """One backtest -> (metrics, realized vol, mean notional, result).

    ``count_key`` is a hashable identity for the CONFIGURATION under test
    (not the market/window) -- N_EVALUATED increments once per distinct
    key ever passed, matching the convention in
    ``kelly_regime_v9_basis_brake.py`` (count once per config, not once
    per (config x market x window) backtest run).
    """
    global N_EVALUATED
    if count_key is not None and count_key not in _SEEN_CONFIGS:
        _SEEN_CONFIGS.add(count_key)
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                         start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    return m, realized_vol(result.equity), mean_notional(result), result


def line(tag, m, vol, notional, result):
    print(f"  {tag:44s} final=${m.final_balance:>11,.0f} "
          f"vol={vol:5.3f} notional={notional:5.3f} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


def all_configs():
    for zw in Z_WINDOWS:
        for lam in LAM:
            yield zw, lam


# --------------------------------------------------------------------------- step 3


def sweep() -> pd.DataFrame:
    """Step 3: every (z_window_days, lam) config on inner-train, spot primary."""
    rows = []
    t0 = time.time()
    for zw, lam in all_configs():
        key = (zw, lam)
        strat = KellyRegimeV10OnchainConfirm(z_window_days=zw, lam=lam)
        m, vol, notional, res = measure(strat, *TRAIN, market=SPOT, count_key=key)
        rows.append({"z_window": zw, "lam": lam, "market": "spot",
                     "final": m.final_balance, "vol": vol, "notional": notional,
                     "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                     "trades": m.num_trades, "fees": m.fees_paid, "liquidated": m.liquidated})
        print(f"[{N_EVALUATED:>2d}] z_window={zw:4d}d lam={lam:.2f}  "
              f"final=${m.final_balance:>9,.0f} DD={m.max_drawdown_pct:>5.1f}% "
              f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>4d} "
              f"notional={notional:.3f} [{time.time() - t0:.0f}s]")
    # lam=0 correctness check: must reduce to v4 bit-for-bit regardless of z_window
    zero = KellyRegimeV10OnchainConfirm(lam=0.0, z_window_days=180)
    m0, vol0, not0, res0 = measure(zero, *TRAIN, market=SPOT, count_key=("lam0-correctness",))
    v4 = get_strategy(INCUMBENT)
    m4, vol4, not4, res4 = measure(v4, *TRAIN, market=SPOT)
    diff = float(np.max(np.abs(res0.df["target"].to_numpy() - res4.df["target"].reindex(res0.df.index).to_numpy())))
    print(f"\nlam=0 correctness check (max|target diff| vs v4): {diff:.3e}  "
          f"{'PASS' if diff < 1e-9 else 'FAIL'}")
    print(f"v4 control (train):  final=${m4.final_balance:>9,.0f} DD={m4.max_drawdown_pct:>5.1f}% "
          f"sharpe={m4.sharpe:>5.2f} trades={m4.num_trades:>4d}")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "sweep_inner_train.csv", index=False)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    print(f"written: {OUT / 'sweep_inner_train.csv'}")
    return out


# --------------------------------------------------------------------------- step 5


def select() -> pd.DataFrame:
    """Step 5: every config on inner-validation, BOTH markets, vs v4 control -- the R-37/38/40/41 check."""
    rows = []
    for zw, lam in all_configs():
        for mname, market in MARKETS:
            strat = KellyRegimeV10OnchainConfirm(z_window_days=zw, lam=lam)
            m, vol, notional, res = measure(strat, *VALID, market=market)
            rows.append({"z_window": zw, "lam": lam, "market": mname,
                         "final": m.final_balance, "vol": vol, "notional": notional,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "trades": m.num_trades, "fees": m.fees_paid, "liquidated": m.liquidated})
        s = rows[-2]
        f = rows[-1]
        print(f"z_window={zw:4d}d lam={lam:.2f}  "
              f"spot: ${s['final']:>9,.0f} DD{s['max_dd']:>5.1f}% sh{s['sharpe']:>5.2f} tr{s['trades']:>4d}   "
              f"fut: ${f['final']:>9,.0f} DD{f['max_dd']:>5.1f}% sh{f['sharpe']:>5.2f} tr{f['trades']:>4d}")
    for mname, market in MARKETS:
        m, vol, notional, res = measure(get_strategy(INCUMBENT), *VALID, market=market)
        rows.append({"z_window": None, "lam": None, "market": mname,
                     "final": m.final_balance, "vol": vol, "notional": notional,
                     "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                     "trades": m.num_trades, "fees": m.fees_paid, "liquidated": m.liquidated,
                     "label": "kelly_regime_v4_control"})
    ctl_s = rows[-2]
    ctl_f = rows[-1]
    print(f"{'kelly_regime_v4 (control)':26s} spot: ${ctl_s['final']:>9,.0f} "
          f"DD{ctl_s['max_dd']:>5.1f}% sh{ctl_s['sharpe']:>5.2f} tr{ctl_s['trades']:>4d}   "
          f"fut: ${ctl_f['final']:>9,.0f} DD{ctl_f['max_dd']:>5.1f}% "
          f"sh{ctl_f['sharpe']:>5.2f} tr{ctl_f['trades']:>4d}")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "select_inner_validation.csv", index=False)
    print(f"\nwritten: {OUT / 'select_inner_validation.csv'}")
    return out


def train_vs_valid_signature(zw: int, lam: float) -> None:
    """The R-37/R-38/R-40/R-41 overfitting signature check, for one named candidate.

    Prints inner-train and inner-validation, both markets, candidate vs
    v4, side by side -- so a win-on-validation/lose-on-train pattern
    (every prior SIZE-axis round's shared failure mode) is visible
    directly rather than requiring the reader to cross-reference two CSVs.
    """
    print(f"\n=== overfitting-signature check: z_window={zw}d lam={lam} ===")
    for wname, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        for mname, market in MARKETS:
            cand = KellyRegimeV10OnchainConfirm(z_window_days=zw, lam=lam)
            m_c, vol_c, not_c, res_c = measure(cand, start, end, market=market)
            m_v4, vol_v4, not_v4, res_v4 = measure(get_strategy(INCUMBENT), start, end, market=market)
            beats = "beats v4" if m_c.final_balance > m_v4.final_balance else "LOSES to v4"
            print(f"  {wname:18s} {mname:8s} cand=${m_c.final_balance:>9,.0f} "
                  f"(DD{m_c.max_drawdown_pct:5.1f}% sh{m_c.sharpe:5.2f} tr{m_c.num_trades:4d})  "
                  f"v4=${m_v4.final_balance:>9,.0f} (DD{m_v4.max_drawdown_pct:5.1f}% "
                  f"sh{m_v4.sharpe:5.2f} tr{m_v4.num_trades:4d})   [{beats}]")


# ------------------------------------------------------------------------ diagnostic 1


def exposure_artifact_check() -> None:
    """Diagnostic (1): mandatory exposure-artifact check (R-33/R-34's standing threshold).

    Mean-notional-matched flat rescale of v4's own target, R^2 against
    the candidate's target, inner-validation, both markets. R^2 > 0.95
    means "this is a flat rescale, not a real mechanism" -- the exact
    pattern from ``kelly_regime_v9_basis_brake.py``. Because this
    multiplier is symmetric (can raise as well as lower exposure), it is
    architecturally harder to collapse into a flat rescale than R-34's
    never-increase-only design, but that is a prior, not a result --
    checked directly here, on every swept configuration.
    """
    print("\nexposure-artifact check (inner-validation, mean-notional-matched flat rescale of v4):")
    for zw, lam in all_configs():
        print(f" z_window={zw:4d}d lam={lam:.2f}:")
        for mname, market in MARKETS:
            cand = KellyRegimeV10OnchainConfirm(z_window_days=zw, lam=lam)
            m_c, vol_c, not_c, res_c = measure(cand, *VALID, market=market)
            v4 = get_strategy(INCUMBENT)
            m_v4, vol_v4, not_v4, res_v4 = measure(v4, *VALID, market=market)

            cand_t = res_c.df["target"].to_numpy(dtype=float)
            v4_t = res_v4.df["target"].reindex(res_c.df.index).to_numpy(dtype=float)
            c = not_c / not_v4 if not_v4 > 0 else float("nan")
            flat = c * v4_t

            mask = np.isfinite(cand_t) & np.isfinite(flat)
            x = flat[mask]
            y = cand_t[mask]
            ss_res = float(np.sum((y - x) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")

            verdict = ("EXPOSURE-LEVEL ARTIFACT (R^2 > 0.95)" if np.isfinite(r2) and r2 > 0.95
                        else "not a flat rescale by this test")
            print(f"    {mname}: cand notional={not_c:.3f} v4 notional={not_v4:.3f} c={c:.3f}  "
                  f"corr={corr:.4f}  R^2={r2:.4f}  {verdict}")


# ------------------------------------------------------------------------ diagnostic 2 / causality


def causality() -> None:
    """Diagnostic (2): two-opposite-tampers, on PRICE and, separately, on the raw ON-CHAIN metric.

    Restricted to strictly pre-2023 bars. The price probe is copied from
    ``kelly_regime_v9_basis_brake.py``'s own procedure. The on-chain probe
    is new to this file: it tampers the raw ``AdrActCnt`` column of the
    injected on-chain frame itself (not a derived series) after a cutoff
    DAY, because a price-only probe cannot exercise this file's one new
    ingredient -- the on-chain frame is loaded independently of ``df`` and
    injected via a constructor argument, exactly as R-41's ``basis``
    argument was.
    """
    pre_2023 = DF.loc[:"2022-12-31"]
    df = pre_2023.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    print("=== price tamper probe ===")
    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def prepared(frame):
        return KellyRegimeV10OnchainConfirm(**PRIMARY).prepare(frame.copy())

    pa = prepared(up)
    pb = prepared(down)
    ok = True
    for col in ("target", "_frac", "_mult"):
        a = pa[col].to_numpy(dtype=float)[:cut]
        b = pb[col].to_numpy(dtype=float)[:cut]
        worst = float(np.nanmax(np.abs(a - b)))
        good = worst < 1e-9
        ok &= good
        print(f"  column={col:16s} max |difference| before the cut = {worst:.3e}  "
              f"{'PASS' if good else 'FAIL'}")

    from tradebot.broker import PaperBroker
    from tradebot.orders import Order

    def decisions(frame):
        s = KellyRegimeV10OnchainConfirm(**PRIMARY)
        prep = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        broker.execute(Order(target=0.1), prep.index[0], float(prep["open"].iloc[0]))
        out = []
        for i in bars:
            ctx = Context(prep, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down)) if oa != ob]
    ok &= not bad
    print(f"  orders {'match' if not bad else f'DIFFER at bars {bad}'} at the probe bars")

    a = run_backtest(KellyRegimeV10OnchainConfirm(**PRIMARY), up.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    b = run_backtest(KellyRegimeV10OnchainConfirm(**PRIMARY), down.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-6
    print(f"  max |equity difference| before the cut = {worst_eq:.3e}  "
          f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")
    print(f"  tampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS' if ok else 'FAIL'} -- no price-dependent decision at or before the cut moves")

    print("\n=== on-chain tamper probe (new to this file) ===")
    cut_ts = df.index[cut]
    cut_day = pd.Timestamp(cut_ts.date(), tz="UTC")
    onchain_up = ONCHAIN_BTC.copy()
    onchain_down = ONCHAIN_BTC.copy()
    mask_after = onchain_up.index >= cut_day
    onchain_up.loc[mask_after, "AdrActCnt"] *= 3.0
    onchain_down.loc[mask_after, "AdrActCnt"] /= 3.0

    strat_up = KellyRegimeV10OnchainConfirm(**PRIMARY, onchain=onchain_up)
    strat_down = KellyRegimeV10OnchainConfirm(**PRIMARY, onchain=onchain_down)
    pu = strat_up.prepare(df.copy())
    pd_ = strat_down.prepare(df.copy())
    ok2 = True
    for col in ("target", "_mult", "_z_raw"):
        a2 = pu[col].to_numpy(dtype=float)[:cut]
        b2 = pd_[col].to_numpy(dtype=float)[:cut]
        worst2 = float(np.nanmax(np.abs(a2 - b2)))
        good2 = worst2 < 1e-9
        ok2 &= good2
        print(f"  column={col:16s} max |difference| before the cut = {worst2:.3e}  "
              f"{'PASS' if good2 else 'FAIL'}")
    print(f"  on-chain-tamper probe: {'PASS' if ok2 else 'FAIL'} -- "
          f"no on-chain-dependent decision at or before the cut moves when ONLY the "
          f"raw on-chain metric (not price) is tampered after the cut day")


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Step 7: pre-registered falsification -- does every candidate hold on ETH?

    Restricted to the overlap between each asset's price file and its own
    on-chain coverage: BTC control uses ``btcusd_bitfinex_5m.csv.gz``
    (R-17's pre-2020 falsification file, 2016-01-01 -> 2019-12-31) sliced
    to 2017-01-01 (BTC on-chain start) -> 2019-12-31; ETH test uses
    ``ethusd_bitfinex_5m.csv.gz`` (2016-03-09 -> 2019-12-31) sliced to
    2019-01-01 (ETH on-chain start) -> 2019-12-31. Neither slice touches
    the 2023+ holdout -- both source files stop in 2019-2020, well before
    it. Pre-registered rule (fixed before this was run): if the candidate
    is not at least comparable to v4 on ETH, or is visibly worse on ETH
    than on the BTC control through the identical code, this direction
    fails.
    """
    onchain_eth = load_onchain_metrics(ROOT / "data", asset="ETH")
    if onchain_eth is None:
        print("no ETH on-chain data found -- cannot run the ETH falsification test")
        return
    specs = (
        ("BTC (control)", "btcusd_bitfinex_5m.csv.gz", "2017-01-01", ONCHAIN_BTC),
        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz", "2019-01-01", onchain_eth),
    )
    for asset, path, onchain_start, onchain in specs:
        full = load_ohlcv_csv(ROOT / "data" / path)
        df = full.loc[onchain_start:]
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}  "
              f"(on-chain coverage from {onchain_start})")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            m_v4, vol_v4, not_v4, res_v4 = measure(get_strategy(INCUMBENT), None, None,
                                                    df=df, market=market)
            line(f"    {INCUMBENT} (control)", m_v4, vol_v4, not_v4, res_v4)
            for zw, lam in all_configs():
                cand = KellyRegimeV10OnchainConfirm(z_window_days=zw, lam=lam, onchain=onchain)
                m_c, vol_c, not_c, res_c = measure(cand, None, None, df=df, market=market)
                line(f"    v10[zw={zw}d,lam={lam:.2f}]", m_c, vol_c, not_c, res_c)


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"spot: {len(DF):,} bars {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d} (data: {LABEL})",
          file=sys.stderr)
    print(f"on-chain BTC: {len(ONCHAIN_BTC):,} days "
          f"{ONCHAIN_BTC.index[0]:%Y-%m-%d} -> {ONCHAIN_BTC.index[-1]:%Y-%m-%d}",
          file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "sweep":
        sweep()
    elif choice == "select":
        select()
    elif choice == "artifact":
        exposure_artifact_check()
    elif choice == "causality":
        causality()
    elif choice == "eth":
        eth()
    elif choice == "all":
        sweep()
        select()
        exposure_artifact_check()
        causality()
        eth()
    else:
        print("usage: python experiments/kelly_regime_v10_onchain_confirm.py "
              "[sweep|select|artifact|causality|eth|all]")
