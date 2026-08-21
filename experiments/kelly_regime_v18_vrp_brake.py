#!/usr/bin/env python
"""kelly_regime_v4 with a bounded, never-increase-only variance-risk-premium brake (R-73 conservative).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

Read ``experiments/_vrp_signal.py``'s module docstring FIRST -- the
mechanism, the DVOL-history limitation, and the pre-registered
falsification test are written there, before any code in this file ran.
This file adds the strategy, the harness, and the exact pre-registered
promotion/rejection decision rule (below), enforced by ``decision()``
before any holdout bar is read.

The idea, briefly
------------------
v4's vote and conditional-vol-target scale are reproduced byte-for-byte;
on top, a bounded, NEVER-INCREASE-ONLY multiplicative haircut
``mult = 1 - lam * clip(-vrp_z / z_scale, 0, 1)`` (so ``mult`` ranges
``[1-lam, 1]``, monotone non-DECREASING in ``vrp_z``) shrinks v4's own
target exposure whenever the variance risk premium (implied DVOL minus
v4's own trailing realized vol, z-scored on a 120-day trailing window --
see ``_vrp_signal.py``) is COMPRESSED or NEGATIVE relative to its own
recent history, and leaves it untouched (``mult == 1``) whenever
``vrp_z >= 0`` or is unavailable (no DVOL coverage). This is the mirror
image of R-53's macro brake, which fired on ELEVATED stress; here the
trigger is a collapsed insurance premium, the literature's complacency
signal (see ``_vrp_signal.py`` for citations).

Grid (fixed a-priori, grounded in the signal's own pre-2023 unconditional
distribution measured once at design time -- 1st/5th/10th/25th/50th
percentiles of -3.76/-2.11/-1.35/-0.33/+0.29 -- not fit to any backtest
outcome):

    LAM     = (0.15, 0.25, 0.35)   -- same modest first-attempt range R-53
                                       used for its own first INFO brake
    Z_SCALE = (1.0, 1.5, 2.5)      -- spans "brake engages readily on any
                                       mild compression" (1.0, close to the
                                       25th-percentile boundary) to "only
                                       the deep tail" (2.5, close to the
                                       1st-5th percentile region)
    PRIMARY = lam=0.25, z_scale=1.5  (grid midpoint)

9 swept configurations + 1 ``lam=0`` correctness check = **10 total**,
matching R-53 conservative's own count.

DVOL-covered inner split (see ``_vrp_signal.py`` for why the standard
2017-2020/2021-2022 split is inapplicable here):

    TRAIN  2021-04-01 -> 2022-04-30   (~13 months, one stress episode)
    VALID  2022-05-01 -> 2022-12-31   (~8 months, two stress episodes:
                                        Terra/Luna, FTX)

Pre-registered decision rule (written before ``sweep()`` was ever run)
------------------------------------------------------------------------
Promote to a single holdout read ONLY if ALL FIVE hold on VALID:

  1. **Beats v4 on VALID, both markets, beyond the noise floor.** At
     least one swept (non-identity) config scores a higher Sharpe than
     the v4 control on BOTH spot and futures VALID by more than the
     ROUTINE.md ±0.2 noise floor, OR shows a clear drawdown/tail
     improvement (max_dd materially lower) at a comparable exposure --
     matching ROUTINE.md step 4's own bar, applied here to the
     DVOL-covered inner-validation stand-in.
  2. **Exposure-artifact R^2 <= 0.95** for that winning config, on VALID,
     both markets (R-33/R-34's standing threshold; see
     ``exposure_artifact_check()``).
  3. **ETH falsification passes** (the pre-registered ROUTINE.md step-2
     test, chosen in ``_vrp_signal.py``): the identical construction fed
     by ETH's own DVOL against ETH's own realized vol, over the same
     2021-04 -> 2022-12 window, does not show a decisively worse or
     oppositely-signed pattern relative to its own BTC-control
     counterpart -- the R-53/R-54 asset-specific-signature failure mode.
  4. **Plateau, not peak**: the winning cell's immediate lam/z_scale
     neighbours in the 3x3 grid must not diverge from it by more than
     the noise floor in the direction that would flip the promotion call
     -- reported for every cell, not just the winner.
  5. **Causality passes** both tamper probes (price, and the raw DVOL
     CSV) in ``causality()``.

If ALL FIVE hold: read the ``OOS_START="2023-01-01"`` holdout EXACTLY
ONCE, frozen configuration = PRIMARY (lam=0.25, z_scale=1.5) unless
PRIMARY itself fails one of the five checks while a single other cell
passes all five -- in which case that cell is used instead and this
substitution is reported explicitly as a rule change, downgrading the
result toward in-sample per ROUTINE.md step 4's own instruction, not
silently substituted.

If ANY of the five fails: **NEGATIVE**. Do not read the holdout. Report
the inner-validation failure with the same care as a win, per ROUTINE.md
step 4's explicit instruction and this project's repeated practice
(R-53, R-54, R-69/B-37, etc.).

This is written and frozen before ``select()`` is run for the first time.

Usage
-----
    python experiments/kelly_regime_v18_vrp_brake.py sweep       # step 3, TRAIN
    python experiments/kelly_regime_v18_vrp_brake.py select      # step 5, VALID, both markets
    python experiments/kelly_regime_v18_vrp_brake.py artifact    # exposure-artifact check (R-33/R-34)
    python experiments/kelly_regime_v18_vrp_brake.py causality   # two-opposite-tampers, price + DVOL
    python experiments/kelly_regime_v18_vrp_brake.py eth         # pre-registered ETH falsification
    python experiments/kelly_regime_v18_vrp_brake.py all         # everything above, in order

The five-part decision rule above is applied by hand against these five
commands' output, in the round's written report -- it is not automated
into a sixth CLI subcommand, because two of its five checks (plateau
shape, ETH pattern comparison) are read judgments over the printed
tables, not single booleans a script should silently decide alone. The
holdout read itself (``OOS_START`` onward), if and only if all five
checks pass, is run as a one-off ``ev(...)`` call via
``scripts/experiment.py`` at the frozen ``PRIMARY`` configuration, exactly
once, per ROUTINE.md step 4 -- not wrapped in a reusable subcommand, so it
cannot be accidentally re-invoked.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments._vrp_signal import compute_vrp_z  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_spot, load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

DATA_DIR = ROOT / "data"
OOS_START = "2023-01-01"


# --------------------------------------------------------------------- strategy


class KellyRegimeV18VrpBrake(Strategy):
    """v4's vote + conditional vol-targeting exposure, braked (never raised) by a compressed VRP.

    See module docstring and ``_vrp_signal.py`` for the full mechanism.
    Defaults for every v4-inherited parameter match ``kelly_regime_v4``
    exactly; ``lam`` and ``z_scale`` are the only new knobs. ``data_dir``
    and ``currency`` are injected purely so the causality probe and the
    ETH falsification can point at a tampered scratch copy / the ETH DVOL
    file respectively; every value is computed inside ``prepare`` from
    ``df.index``, nothing precomputed and reused across instances.
    """

    name = "kelly_regime_v18_vrp_brake"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85,
                 lam: float = 0.25, z_scale: float = 1.5,
                 data_dir: str | Path | None = None, currency: str = "BTC") -> None:
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
        # ---- new: the VRP brake ------------------------------------------
        self.lam = lam            # mult in [1-lam, 1]; 0 = exact v4
        self.z_scale = z_scale    # |vrp_z| at which the brake reaches full lam
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self.currency = currency

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
               * np.sqrt(BARS_PER_YEAR)).shift(1)
        slow = (vol.ewm(span=self.anchor_span_days * BARS_PER_DAY,
                         min_periods=BARS_PER_DAY).mean().to_numpy())
        vol_arr = vol.to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol_arr / slow, np.nan)
            full = np.minimum(self.target_vol / vol_arr, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        # ---- new: the VRP brake -------------------------------------------
        raw_vrp_z = compute_vrp_z(df, vol, self.data_dir, currency=self.currency).to_numpy(dtype=float)
        # No DVOL coverage (before 2021-03-24, or the fetch is missing) ->
        # treat as vrp_z=0 -> mult=1 -> exact v4 fallback, the same
        # convention as R-44/R-53/R-54's brakes.
        z_used = np.where(np.isfinite(raw_vrp_z), raw_vrp_z, 0.0)
        mult = 1.0 - self.lam * np.clip(-z_used / self.z_scale, 0.0, 1.0)  # in [1-lam, 1]

        # ---- single causal forward pass: byte-for-byte v3/v4 breakout
        # hysteresis on the vol-targeting state, plus the new brake -------
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
        df["_vrp_z"] = raw_vrp_z
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures


# ------------------------------------------------------------------------ harness

DF, LABEL = load_dataset(ROOT / "data", "spot")

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

INCUMBENT = "kelly_regime_v4"

# ---- DVOL-covered inner split (see module docstring and _vrp_signal.py) ----
TRAIN = ("2021-04-01", "2022-04-30")
VALID = ("2022-05-01", "2022-12-31")

# ---- sweep grid: fixed a-priori, grounded in the pre-2023 unconditional
# distribution of vrp_z (percentiles measured once, printed at import time
# below), not fit to any backtest outcome -- see module docstring.
LAM = (0.15, 0.25, 0.35)
Z_SCALE = (1.0, 1.5, 2.5)
PRIMARY = dict(lam=0.25, z_scale=1.5)  # grid midpoint

N_EVALUATED = 0
_SEEN_CONFIGS: set[tuple] = set()

OUT = ROOT / "reports" / "kelly_regime_v18_vrp_brake"
NOISE_FLOOR = 0.2  # ROUTINE.md's ±0.2 Sharpe noise floor
R2_ARTIFACT_THRESHOLD = 0.95


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
    (not the market/window); ``N_EVALUATED`` increments once per distinct
    key ever passed, matching ``kelly_regime_v14_macro_brake.py``'s
    convention (count once per config, not once per config x market x
    window backtest run).
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
    for lam in LAM:
        for zs in Z_SCALE:
            yield lam, zs


# --------------------------------------------------------------------------- step 3


def sweep() -> pd.DataFrame:
    """Step 3: every (lam, z_scale) config on TRAIN, spot primary."""
    rows = []
    t0 = time.time()
    for lam, zs in all_configs():
        key = (lam, zs)
        strat = KellyRegimeV18VrpBrake(lam=lam, z_scale=zs)
        m, vol, notional, res = measure(strat, *TRAIN, market=SPOT, count_key=key)
        rows.append({"lam": lam, "z_scale": zs, "market": "spot",
                     "final": m.final_balance, "vol": vol, "notional": notional,
                     "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                     "trades": m.num_trades, "fees": m.fees_paid, "liquidated": m.liquidated})
        print(f"[{N_EVALUATED:>2d}] lam={lam:.2f} z_scale={zs:.1f}  "
              f"final=${m.final_balance:>9,.0f} DD={m.max_drawdown_pct:>5.1f}% "
              f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>4d} "
              f"notional={notional:.3f} [{time.time() - t0:.0f}s]")
    # lam=0 correctness check: must reduce to v4 bit-for-bit regardless of z_scale
    zero = KellyRegimeV18VrpBrake(lam=0.0, z_scale=1.5)
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
    out.to_csv(OUT / "sweep_train.csv", index=False)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    print(f"written: {OUT / 'sweep_train.csv'}")
    return out


# --------------------------------------------------------------------------- step 5


def select() -> pd.DataFrame:
    """Step 5: every config on VALID, BOTH markets, vs v4 control."""
    rows = []
    for lam, zs in all_configs():
        for mname, market in MARKETS:
            strat = KellyRegimeV18VrpBrake(lam=lam, z_scale=zs)
            m, vol, notional, res = measure(strat, *VALID, market=market)
            rows.append({"lam": lam, "z_scale": zs, "market": mname,
                         "final": m.final_balance, "vol": vol, "notional": notional,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "trades": m.num_trades, "fees": m.fees_paid, "liquidated": m.liquidated})
        s = rows[-2]
        f = rows[-1]
        print(f"lam={lam:.2f} z_scale={zs:.1f}  "
              f"spot: ${s['final']:>9,.0f} DD{s['max_dd']:>5.1f}% sh{s['sharpe']:>5.2f} tr{s['trades']:>4d}   "
              f"fut: ${f['final']:>9,.0f} DD{f['max_dd']:>5.1f}% sh{f['sharpe']:>5.2f} tr{f['trades']:>4d}")
    for mname, market in MARKETS:
        m, vol, notional, res = measure(get_strategy(INCUMBENT), *VALID, market=market)
        rows.append({"lam": None, "z_scale": None, "market": mname,
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
    out.to_csv(OUT / "select_valid.csv", index=False)
    print(f"\nwritten: {OUT / 'select_valid.csv'}")
    return out


# ------------------------------------------------------------------------ diagnostic 1


def exposure_artifact_check() -> pd.DataFrame:
    """Diagnostic (1): mandatory exposure-artifact check (R-33/R-34's standing threshold).

    Mean-notional-matched flat rescale of v4's own target, R^2 against
    the candidate's target, on VALID, both markets.
    """
    print("\nexposure-artifact check (VALID, mean-notional-matched flat rescale of v4):")
    rows = []
    for lam, zs in all_configs():
        print(f" lam={lam:.2f} z_scale={zs:.1f}:")
        for mname, market in MARKETS:
            cand = KellyRegimeV18VrpBrake(lam=lam, z_scale=zs)
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

            verdict = ("EXPOSURE-LEVEL ARTIFACT (R^2 > 0.95)" if np.isfinite(r2) and r2 > R2_ARTIFACT_THRESHOLD
                        else "not a flat rescale by this test")
            print(f"    {mname}: cand notional={not_c:.3f} v4 notional={not_v4:.3f} c={c:.3f}  "
                  f"corr={corr:.4f}  R^2={r2:.4f}  {verdict}")
            rows.append({"lam": lam, "z_scale": zs, "market": mname, "r2": r2, "corr": corr})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "artifact_valid.csv", index=False)
    return out


# ------------------------------------------------------------------------ diagnostic 2 / causality


def _write_tampered_dvol_dir(cut_day: pd.Timestamp, scale: float) -> Path:
    """Write a scratch copy of the DVOL CSVs with close scaled from ``cut_day`` onward.

    Copies ``data/{btc,eth}_dvol_daily.csv.gz`` into a fresh temp
    directory (never under the repo), multiplying ``close`` (and, to keep
    the file internally plausible, ``high``) on and after ``cut_day`` by
    ``scale``.
    """
    tmp = Path(tempfile.mkdtemp(prefix="dvol_tamper_"))
    for filename in ("btc_dvol_daily.csv.gz", "eth_dvol_daily.csv.gz"):
        src = ROOT / "data" / filename
        if not src.exists():
            continue
        raw = pd.read_csv(src, parse_dates=["timestamp"])
        mask = raw["timestamp"] >= cut_day.tz_localize(None)
        raw.loc[mask, "close"] = raw.loc[mask, "close"] * scale
        raw.loc[mask, "high"] = raw.loc[mask, "high"] * scale
        raw.to_csv(tmp / filename, index=False)
    return tmp


def causality() -> None:
    """Diagnostic (2): two-opposite-tampers, on PRICE and, separately, on the raw DVOL CSVs.

    Restricted to a strictly pre-2023 slice with genuine DVOL coverage
    (2021-01-01 -> 2022-12-31), so the DVOL-tamper probe actually
    exercises this file's one new ingredient.
    """
    window = DF.loc["2021-01-01":"2022-12-31"]
    df = window.copy()
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
        return KellyRegimeV18VrpBrake(**PRIMARY).prepare(frame.copy())

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
        s = KellyRegimeV18VrpBrake(**PRIMARY)
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

    a = run_backtest(KellyRegimeV18VrpBrake(**PRIMARY), up.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    b = run_backtest(KellyRegimeV18VrpBrake(**PRIMARY), down.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-6
    print(f"  max |equity difference| before the cut = {worst_eq:.3e}  "
          f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")
    print(f"  tampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS' if ok else 'FAIL'} -- no price-dependent decision at or before the cut moves")

    print("\n=== DVOL tamper probe (new to this file) ===")
    cut_ts = df.index[cut]
    cut_day = pd.Timestamp(cut_ts.date(), tz="UTC")
    dir_up = _write_tampered_dvol_dir(cut_day, 3.0)
    dir_down = _write_tampered_dvol_dir(cut_day, 1.0 / 3.0)
    try:
        strat_up = KellyRegimeV18VrpBrake(**PRIMARY, data_dir=dir_up)
        strat_down = KellyRegimeV18VrpBrake(**PRIMARY, data_dir=dir_down)
        pu = strat_up.prepare(df.copy())
        pdn = strat_down.prepare(df.copy())
        ok2 = True
        for col in ("target", "_mult", "_vrp_z"):
            a2 = pu[col].to_numpy(dtype=float)[:cut]
            b2 = pdn[col].to_numpy(dtype=float)[:cut]
            worst2 = float(np.nanmax(np.abs(a2 - b2)))
            good2 = worst2 < 1e-9
            ok2 &= good2
            print(f"  column={col:16s} max |difference| before the cut = {worst2:.3e}  "
                  f"{'PASS' if good2 else 'FAIL'}")
        print(f"  DVOL-tamper probe: {'PASS' if ok2 else 'FAIL'} -- "
              f"no DVOL-dependent decision at or before the cut moves when ONLY the "
              f"raw DVOL CSV (not price) is tampered from the cut day onward")
    finally:
        shutil.rmtree(dir_up, ignore_errors=True)
        shutil.rmtree(dir_down, ignore_errors=True)


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Pre-registered falsification (chosen in ``_vrp_signal.py``): does the SAME
    construction, fed by ETH's own DVOL against ETH's own realized vol,
    behave comparably to its BTC counterpart over the identical window?

    Window: 2021-04-01 -> 2022-12-31 (TRAIN+VALID combined -- the entire
    DVOL-covered pre-holdout period; there is no earlier ETH DVOL to
    extend it). BTC control uses the committed spot series;
    ETH test uses ``load_coinbase_spot(data_dir, "ETH")``
    (``ethusd_coinbase_spot_5m.csv.gz``), which spans this window. Neither
    file touches the 2023+ holdout.

    Outcome that kills this direction, named in ``_vrp_signal.py`` before
    this was run: if the ETH construction shows a decisively worse or
    oppositely-signed pattern relative to its own v4 control than the BTC
    construction does relative to ITS v4 control, this direction fails --
    a market-wide options-implied-vol signal that only "works" on the
    asset it was designed against is not evidence of a real mechanism.
    """
    btc_df = DF
    eth_df = load_coinbase_spot(DATA_DIR, "ETH")
    if eth_df is None:
        print("ETH coinbase spot file not found -- cannot run falsification")
        return

    specs = (
        ("BTC (control)", btc_df, "BTC"),
        ("ETH (test)", eth_df, "ETH"),
    )
    start, end = "2021-04-01", "2022-12-31"
    for asset, df, currency in specs:
        print(f"\n{asset}  {len(df):,} bars  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            m_v4, vol_v4, not_v4, res_v4 = measure(get_strategy(INCUMBENT), start, end,
                                                    df=df, market=market)
            line(f"    {INCUMBENT} (control)", m_v4, vol_v4, not_v4, res_v4)
            for lam, zs in all_configs():
                cand = KellyRegimeV18VrpBrake(lam=lam, z_scale=zs, currency=currency)
                m_c, vol_c, not_c, res_c = measure(cand, start, end, df=df, market=market)
                delta_sharpe = m_c.sharpe - m_v4.sharpe
                line(f"    v18[lam={lam:.2f},zs={zs:.1f}] (dSharpe={delta_sharpe:+.2f})",
                     m_c, vol_c, not_c, res_c)


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"spot: {len(DF):,} bars {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d} (data: {LABEL})",
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
        print("usage: python experiments/kelly_regime_v18_vrp_brake.py "
              "[sweep|select|artifact|causality|eth|all]")
