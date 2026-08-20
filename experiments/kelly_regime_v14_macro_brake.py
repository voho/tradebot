#!/usr/bin/env python
"""kelly_regime_v4 with a bounded, never-increase-only macro-stress brake (CONSERVATIVE branch, R-53).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

The idea
--------
This session commits the project's first genuinely price-independent,
market-wide data source: daily S&P 500 close, VIX close and the Fed's
trade-weighted broad dollar index (DXY proxy), 2016-06 -> present, from
FRED's free public CSV endpoint (``data/{spx,vix,dxy}_daily.csv.gz``,
``scripts/fetch_macro_data.py``). Unlike every prior INFO attack in this
project -- L-12/L-14/L-15/L-16 (all four tried to recover missing
information FROM PRICE and failed), R-41's Deribit basis (a SECOND price
series, still BTC-specific and traded), and R-44's on-chain confirmation
(price-independent, but specific to BTC's own network) -- VIX and DXY
describe the *rest of the financial system*: equity-implied fear and
dollar-strength flows that exist whether or not anyone is trading BTC.

The shared signal ``experiments/_macro_signal.py`` (operator-authored,
shared with the sibling novel branch, not edited by this file) exposes
``compute_macro_stress(df, data_dir) -> stress_z``, a causal composite:
``stress_z = 0.5 * vix_z + 0.5 * dxy_mom_z``, where ``vix_z`` is the VIX
level z-scored on its own trailing 365-day window and ``dxy_mom_z`` is
20-day DXY momentum z-scored on its own trailing 365-day window, aligned
onto any bar grid one full day forward for FRED's publication lag (same
convention as ``align_onchain_causal``, B-07). Positive ``stress_z`` means
elevated equity fear and/or dollar strengthening -- the literature's
risk-off direction. See that file's own docstring for full citations
(Luo, Tsai & Yen 2024/2025 SSRN on the VIX term structure leading Bitcoin
returns; IMF WP 2023/213 on crypto-equity spillover intensifying in
stress periods; the 2024-2026 BTC/DXY inverse-correlation literature;
Klein, Thu & Walther 2018 on Bitcoin behaving as a risk asset, not a safe
haven).

Mechanism, one sentence
------------------------
v4's vote and conditional-vol-target scale are reproduced byte-for-byte;
on top, a bounded, NEVER-INCREASE-ONLY multiplicative haircut
``mult = 1 - lam * clip(stress_z / z_scale, 0, 1)`` (so ``mult`` ranges
``[1-lam, 1]``, monotone non-increasing in ``stress_z``) shrinks v4's own
target exposure whenever macro stress is elevated, and leaves it
untouched (``mult == 1``) whenever ``stress_z <= 0`` or is unavailable.

Why never-increase-only, not symmetric (unlike R-44's on-chain confirmation)
-------------------------------------------------------------------------------
R-44's on-chain multiplier is deliberately symmetric because its own
governing risk (R-08/B-07) is a SIGN-INVERSION trap: a modifier that only
ever *reduces* exposure when a genuinely informative volume/activity
metric rises would repeat R-08's finding that BTC's forward Sharpe is
highest exactly when a naive "more activity/vol -> less exposure" rule
would de-lever (Baur & Dimpfl 2018's inverse leverage effect, R-10). That
trap does not obviously apply here in the same direction: the
macro-stress literature's claim is specifically that *elevated* VIX/DXY
stress leads *crypto risk-off* (drawdown), not that *calm* markets
(``stress_z`` very negative) forecast unusually strong forward Bitcoin
returns the way BTC's own historically elevated volatility does. Design
this brake as a pure one-sided risk reduction -- "never increase
exposure on the strength of a risk-off signal" -- is therefore the more
conservative reading of the same literature, and matches the task brief's
own framing (analogous to ``harsanyi_crowd``'s crowding haircut and
R-41's basis brake, both never-increase-only). The symmetric alternative
(raising exposure when ``stress_z`` is very negative) is the sibling
novel branch's structurally different question, not this one's.

Constraint attacked
--------------------
INFO -- the project's #1 standing-diagnosis constraint, and the third
attempt at it that is not a transform of the incumbent price series
(after R-41's basis and R-44's on-chain), the first from data describing
the broader financial system rather than BTC's own market or network.

Known failure mode, checked explicitly (not assumed absent)
---------------------------------------------------------------
R-34's conservative branch (``kelly_regime_v5_damp.py``) used an
identically-SHAPED never-increase-only multiplier (``mult in [1-lam,1]``)
fed by a smoothed transform of the SAME price series already driving the
vote, and it collapsed into a flat rescale of v4 in disguise
(R^2=0.997 against a mean-matched constant-exposure control). This
file's input (macro stress) is architecturally independent of price --
but that is a prior, not a result. The identical R-33/R-34 exposure-
artifact regression (candidate's own ``target`` vs a mean-notional-
matched flat rescale of v4's ``target``, R^2) is run on every swept
config below, and reported honestly whatever it says.

Not a duplicate of
-------------------
- **L-01** (``kelly_regime_v4``): the strategy this file wraps unchanged.
  Every v4-inherited parameter keeps its exact v4 default; this file adds
  exactly one new axis on top.
- **L-12** (``harsanyi_crowd``): a belief-margin *direction* signal built
  from bar-return likelihoods (price-derived, INFO unaddressed) that lost
  as a direction input. This file's own architectural cousin, R-34's
  ``kelly_regime_v5_damp.py``, tested L-12's own stated hypothesis (feed
  the SAME price-derived posterior as a SIZE dampener instead) and it
  collapsed into an exposure-level artifact -- the exact failure mode
  checked for above. This file repeats that architecture with a
  genuinely different, price-independent input.
- **R-08 / R-10** (better volatility forecasting / inverse leverage
  effect): R-08 found a strictly better *volatility* forecast made things
  WORSE ($52K vs $115K) by de-levering more promptly into BTC's own
  high-vol, high-forward-Sharpe states (R-10's inverse-leverage finding).
  This file's signal is not a volatility forecast of BTC itself and does
  not condition on BTC's own realized or implied vol at all -- it reads
  an independent, market-wide risk-off indicator. The "why never-increase,
  not symmetric" section above addresses directly why this file's design
  is not simply R-08's trap wearing a new signal.
- **B-07 / R-44** (``kelly_regime_v10_onchain_confirm.py``): same house
  style (bounded multiplier on top of byte-for-byte v4, sweep/select/
  artifact/causality/eth harness) and the same INFO constraint, but a
  DIFFERENT data source (blockchain activity, specific to the traded
  asset's own network) and a DIFFERENT architecture (symmetric, can raise
  exposure). This file's data (VIX/DXY) describes the rest of the
  financial system, and its multiplier is never-increase-only by design,
  per the section above.
- **R-41** (``kelly_regime_v9_basis_brake.py``): same never-increase-only
  architectural template and the identical mechanism shape this file
  copies, but a DIFFERENT data source -- basis is a second, independently
  TRADED price series (still fundamentally a market-price signal); VIX
  and DXY are not traded BTC/ETH instruments at all, and are not derived
  from any BTC/ETH price series either directly or via a spread.
- A sibling agent runs a structurally different exploitation of the SAME
  shared ``stress_z`` signal (feeding it into the regime vote itself,
  rather than a post-hoc multiplier) in parallel this round, on a disjoint
  file (``kelly_regime_v14_macro_lead.py``). Not read or coordinated with
  here, per ROUTINE.md's parallelism rules.

Causality
---------
``compute_macro_stress`` is already causal by construction (every rolling
statistic computed on the raw daily macro frame before any shift; the
final daily series is projected onto the bar grid one more day forward
for FRED's own publication lag -- see its own docstring). Two independent
two-opposite-tampers probes are run: the standard PRICE probe (bars after
a cut multiplied/divided by 3, copied from ``kelly_regime_v9_basis_brake.py``'s
own procedure) and a second, new-to-this-file probe that tampers the raw
macro CSVs themselves (VIX and DXY, scaled 3x / (1/3)x from a cut DAY
onward, written to a throwaway scratch directory -- never under the
repo -- and read back through ``compute_macro_stress`` unmodified) so this
file's one new ingredient (the injected ``data_dir``) is genuinely
exercised, not just the price path the standard probe already covers.

Usage
-----
    python experiments/kelly_regime_v14_macro_brake.py sweep       # step 3, inner-train
    python experiments/kelly_regime_v14_macro_brake.py select      # step 5, inner-validation, both markets
    python experiments/kelly_regime_v14_macro_brake.py artifact    # exposure-artifact check (R-33/R-34)
    python experiments/kelly_regime_v14_macro_brake.py causality   # two-opposite-tampers, price + macro
    python experiments/kelly_regime_v14_macro_brake.py eth         # pre-registered ETH falsification
    python experiments/kelly_regime_v14_macro_brake.py all         # everything above, in order
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

from experiments._macro_signal import compute_macro_stress  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

DATA_DIR = ROOT / "data"


# --------------------------------------------------------------------- strategy


class KellyRegimeV14MacroBrake(Strategy):
    """v4's vote + conditional vol-targeting exposure, braked (never raised) by macro stress.

    See module docstring for the full mechanism. Defaults for every
    v4-inherited parameter match ``kelly_regime_v4`` exactly; ``lam`` and
    ``z_scale`` are the only new knobs. ``data_dir`` is injected (defaults
    to the repo's real ``data/``) purely so the causality probe can point
    it at a tampered scratch copy; every value used is still computed
    inside ``prepare`` from ``df.index``, nothing is precomputed and
    reused across instances.
    """

    name = "kelly_regime_v14_macro_brake"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85,
                 lam: float = 0.25, z_scale: float = 2.0,
                 data_dir: str | Path | None = None) -> None:
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
        # ---- new: the macro-stress brake -----------------------------------
        self.lam = lam            # mult in [1-lam, 1]; 0 = exact v4
        self.z_scale = z_scale    # stress_z level at which the brake reaches full lam
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR

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

        # ---- new: the macro-stress brake ------------------------------------
        raw_stress = (compute_macro_stress(df, self.data_dir)
                      .reindex(df.index).to_numpy(dtype=float))
        # No data yet (before macro coverage begins, or the fetch is
        # missing) -> treat as zero stress -> mult=1 -> exact v4 fallback,
        # the same convention as R-44's on-chain confirmation.
        z_used = np.where(np.isfinite(raw_stress), raw_stress, 0.0)
        mult = 1.0 - self.lam * np.clip(z_used / self.z_scale, 0.0, 1.0)  # in [1-lam, 1]

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
        df["_stress_z"] = raw_stress
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

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

# Macro coverage (2016-06-01 onward) fully precedes the committed spot
# series' own start (2017-01-01) once the z-score's 60-day min_periods
# warmup is accounted for -- verified directly (0 NaNs in stress_z across
# the entire DF index, printed at import time below) -- so, unlike R-41's
# basis brake (Deribit coverage starts 2018-08-14, mid-way through a
# typical inner-train window), the standard ROUTINE.md inner-train /
# inner-validation split applies unmodified with no fallback-window caveat
# needed on the primary BTC series.
TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")

INCUMBENT = "kelly_regime_v4"

# ---- sweep grid: fixed a-priori choices, not fit to inner-validation ----
# z_scale candidates are chosen from stress_z's own UNCONDITIONAL
# distribution on the full committed spot index (std=0.91, p90=1.11,
# p95=1.63, p99=3.21, measured once at design time, not re-fit per
# window): 1.0 saturates the brake close to its ~p90 tail, 2.0 close to
# its ~p97-98 tail, 3.0 close to its ~p99 tail -- spanning "brake engages
# on any mildly elevated day" to "brake only engages in genuine tail
# stress episodes". lam candidates (0.15-0.35) bracket R-41's basis
# brake's selected 0.5 from below and R-44's on-chain lam range
# (0.10-0.20) from above -- a deliberately modest haircut range for a
# FIRST test of a brand-new information channel, not tuned to this data.
LAM = (0.15, 0.25, 0.35)
Z_SCALE = (1.0, 2.0, 3.0)
PRIMARY = dict(lam=0.25, z_scale=2.0)  # grid midpoint, used for causality/eth defaults

N_EVALUATED = 0  # distinct configurations searched (routine's trials count)
_SEEN_CONFIGS: set[tuple] = set()

OUT = ROOT / "reports" / "kelly_regime_v14_macro_brake"


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
    ``kelly_regime_v9_basis_brake.py`` / ``kelly_regime_v10_onchain_confirm.py``
    (count once per config, not once per (config x market x window)
    backtest run).
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
    """Step 3: every (lam, z_scale) config on inner-train, spot primary."""
    rows = []
    t0 = time.time()
    for lam, zs in all_configs():
        key = (lam, zs)
        strat = KellyRegimeV14MacroBrake(lam=lam, z_scale=zs)
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
    zero = KellyRegimeV14MacroBrake(lam=0.0, z_scale=2.0)
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
    """Step 5: every config on inner-validation, BOTH markets, vs v4 control -- the R-37/38/40/41/44 check."""
    rows = []
    for lam, zs in all_configs():
        for mname, market in MARKETS:
            strat = KellyRegimeV14MacroBrake(lam=lam, z_scale=zs)
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
    out.to_csv(OUT / "select_inner_validation.csv", index=False)
    print(f"\nwritten: {OUT / 'select_inner_validation.csv'}")
    return out


def train_vs_valid_signature(lam: float, zs: float) -> None:
    """The R-37/R-38/R-40/R-41/R-44 overfitting signature check, for one named candidate.

    Prints inner-train and inner-validation, both markets, candidate vs
    v4, side by side -- so a win-on-validation/lose-on-train pattern
    (every prior SIZE-axis round's shared failure mode) is visible
    directly rather than requiring the reader to cross-reference two CSVs.
    """
    print(f"\n=== overfitting-signature check: lam={lam} z_scale={zs} ===")
    for wname, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        for mname, market in MARKETS:
            cand = KellyRegimeV14MacroBrake(lam=lam, z_scale=zs)
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
    means "this is a flat rescale, not a real mechanism" -- R-34's own
    ``kelly_regime_v5_damp.py`` failure, checked directly here on every
    swept configuration rather than assumed absent because the input this
    time is price-independent.
    """
    print("\nexposure-artifact check (inner-validation, mean-notional-matched flat rescale of v4):")
    for lam, zs in all_configs():
        print(f" lam={lam:.2f} z_scale={zs:.1f}:")
        for mname, market in MARKETS:
            cand = KellyRegimeV14MacroBrake(lam=lam, z_scale=zs)
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


def _write_tampered_macro_dir(cut_day: pd.Timestamp, scale: float) -> Path:
    """Write a scratch copy of the macro CSVs with vix/dxy scaled from ``cut_day`` onward.

    Copies ``data/{spx,vix,dxy}_daily.csv.gz`` into a fresh temp directory
    (never under the repo), multiplying VIX and DXY's raw values on and
    after ``cut_day`` by ``scale``; ``spx`` is copied unmodified
    (``compute_macro_stress`` never reads it, but ``load_macro_metrics``
    requires all three files to be present or it returns ``None``). Used
    only by ``causality()``; the directory is removed by the caller.
    """
    tmp = Path(tempfile.mkdtemp(prefix="macro_tamper_"))
    for col, filename in (("spx", "spx_daily.csv.gz"), ("vix", "vix_daily.csv.gz"),
                          ("dxy", "dxy_daily.csv.gz")):
        raw = pd.read_csv(ROOT / "data" / filename, parse_dates=["date"])
        if col in ("vix", "dxy"):
            value_col = raw.columns[1]
            mask = raw["date"] >= cut_day.tz_localize(None)
            raw.loc[mask, value_col] = raw.loc[mask, value_col] * scale
        raw.to_csv(tmp / filename, index=False)
    return tmp


def causality() -> None:
    """Diagnostic (2): two-opposite-tampers, on PRICE and, separately, on the raw MACRO CSVs.

    Restricted to strictly pre-2023 bars. The price probe is copied from
    ``kelly_regime_v9_basis_brake.py``'s own procedure. The macro probe is
    new to this file: it tampers the raw VIX/DXY CSVs themselves (not the
    price frame) from a cut DAY onward, in a throwaway scratch directory,
    because a price-only probe cannot exercise this file's one new
    ingredient -- ``data_dir`` is injected via a constructor argument,
    exactly as R-41's ``basis`` argument and R-44's ``onchain`` argument
    were, precisely so this second probe is possible.
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
        return KellyRegimeV14MacroBrake(**PRIMARY).prepare(frame.copy())

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
        s = KellyRegimeV14MacroBrake(**PRIMARY)
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

    a = run_backtest(KellyRegimeV14MacroBrake(**PRIMARY), up.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    b = run_backtest(KellyRegimeV14MacroBrake(**PRIMARY), down.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-6
    print(f"  max |equity difference| before the cut = {worst_eq:.3e}  "
          f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")
    print(f"  tampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS' if ok else 'FAIL'} -- no price-dependent decision at or before the cut moves")

    print("\n=== macro tamper probe (new to this file) ===")
    cut_ts = df.index[cut]
    cut_day = pd.Timestamp(cut_ts.date(), tz="UTC")
    dir_up = _write_tampered_macro_dir(cut_day, 3.0)
    dir_down = _write_tampered_macro_dir(cut_day, 1.0 / 3.0)
    try:
        strat_up = KellyRegimeV14MacroBrake(**PRIMARY, data_dir=dir_up)
        strat_down = KellyRegimeV14MacroBrake(**PRIMARY, data_dir=dir_down)
        pu = strat_up.prepare(df.copy())
        pdn = strat_down.prepare(df.copy())
        ok2 = True
        for col in ("target", "_mult", "_stress_z"):
            a2 = pu[col].to_numpy(dtype=float)[:cut]
            b2 = pdn[col].to_numpy(dtype=float)[:cut]
            worst2 = float(np.nanmax(np.abs(a2 - b2)))
            good2 = worst2 < 1e-9
            ok2 &= good2
            print(f"  column={col:16s} max |difference| before the cut = {worst2:.3e}  "
                  f"{'PASS' if good2 else 'FAIL'}")
        print(f"  macro-tamper probe: {'PASS' if ok2 else 'FAIL'} -- "
              f"no macro-dependent decision at or before the cut moves when ONLY the "
              f"raw VIX/DXY CSVs (not price) are tampered from the cut day onward")
    finally:
        shutil.rmtree(dir_up, ignore_errors=True)
        shutil.rmtree(dir_down, ignore_errors=True)


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Step 7: pre-registered falsification -- does every candidate hold on ETH?

    Standard pre-2020 BTC-control/ETH falsification pair this project
    always uses for this kind of check (per ``kelly_regime_v10_onchain_confirm.py``'s
    ``eth()``): ``btcusd_bitfinex_5m.csv.gz`` (2016-01-01 -> 2019-12-31) as
    the control, ``ethusd_bitfinex_5m.csv.gz`` (2016-03-09 -> 2019-12-31)
    as the test. Neither file touches the 2023+ holdout. VIX/DXY coverage
    begins 2016-06-01, a few months into both files -- the strategy's own
    ``isfinite`` fallback (mult=1, exact v4) handles that gap the same way
    it handles any other missing-data period, verified directly by the
    ``_stress_z`` column rather than assumed. Pre-registered rule (fixed
    before this was run): if the candidate is not at least comparable to
    v4 on ETH, or is visibly worse on ETH than on the BTC control through
    the identical code, this direction fails -- the macro signal is
    market-wide, not BTC-specific, so an ETH-only failure is itself
    reportable evidence, not something to explain away.
    """
    specs = (
        ("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz"),
    )
    for asset, path in specs:
        df = load_ohlcv_csv(ROOT / "data" / path)
        n_nan = int(compute_macro_stress(df, DATA_DIR).isna().sum())
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}  "
              f"(stress_z NaN bars: {n_nan:,} of {len(df):,}, pre-macro-coverage fallback)")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            m_v4, vol_v4, not_v4, res_v4 = measure(get_strategy(INCUMBENT), None, None,
                                                    df=df, market=market)
            line(f"    {INCUMBENT} (control)", m_v4, vol_v4, not_v4, res_v4)
            for lam, zs in all_configs():
                cand = KellyRegimeV14MacroBrake(lam=lam, z_scale=zs)
                m_c, vol_c, not_c, res_c = measure(cand, None, None, df=df, market=market)
                line(f"    v14[lam={lam:.2f},zs={zs:.1f}]", m_c, vol_c, not_c, res_c)


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"spot: {len(DF):,} bars {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d} (data: {LABEL})",
          file=sys.stderr)
    _full_stress = compute_macro_stress(DF, DATA_DIR)
    print(f"stress_z on the full spot index: {_full_stress.isna().sum():,} NaN of {len(_full_stress):,} bars",
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
        print("usage: python experiments/kelly_regime_v14_macro_brake.py "
              "[sweep|select|artifact|causality|eth|all]")
