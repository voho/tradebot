#!/usr/bin/env python
"""R-76 CONSERVATIVE branch: statistical-arbitrage / pairs trading between two
of this project's own price series -- Step A cointegration screen, then (only
if it passes) the literal Gatev-Goetzmann-Rouwenhorst pairs trade at 5m.

=====================================================================
WHAT THIS FILE IS, AND WHY IT IS NEW
=====================================================================

Every one of this project's six prior INFO-axis attempts (on-chain B-07/R-44,
macro VIX/DXY R-53/R-54, stablecoin flow R-54/R-55/R-58, DVOL/VRP R-73, MVRV
R-74, calendar/session R-75) fed an *external* series into `kelly_regime_v4`'s
own single-instrument gate. Every prior multi-asset round (B-16/B-19/B-20's
BTC+ETH weighting, R-63's cross-sectional panel, R-61's per-asset vote,
R-57's byte-identical panel replication) traded each instrument off its OWN
single-series signal at a fixed capital split. None of them ever computed or
traded the SPREAD between two instruments -- the relationship itself as the
signal, rather than either instrument's own price history. Grepped
`docs/LEDGER.md` for "pairs"/"cointegrat"/"spread": zero hits before this
round.

**Citations.** Engle & Granger (1987), "Co-integration and Error
Correction," Econometrica 55(2) -- the two-step method this screen
hand-rolls (OLS the levels, then test the residual for a unit root).
Vidyamurthy (2004), *Pairs Trading: Quantitative Methods and Analysis*,
Wiley. Gatev, Goetzmann & Rouwenhorst (2006), "Pairs Trading: Performance of
a Relative-Value Arbitrage Rule," RFS 19(3):797-827 -- the entry/exit
z-score thresholds (|z|>2 in, |z|<0.5 out) reused verbatim in Step B, not
refit. Fil & Kristoufek (2020), "Pairs Trading in Cryptocurrency Markets,"
IEEE Access 8:172644-172651 -- found Johansen cointegration among large-cap
coins on 2017-2019 daily data, but also documents that crypto pairs are
frequently NOT cointegrated at conventional significance, so a null screen
result here would be consistent with prior literature.

**Mechanism, one sentence.** If two instruments' log prices share a common
stochastic trend (are cointegrated), the linear combination that cancels
that trend is a stationary, mean-reverting spread, and betting on its
reversion via a frozen hedge ratio is a genuinely different kind of edge
than anything this project has traded: not "what will this one series do
next" but "how far has a stable relationship between two series drifted".

Constraint attacked: **INFO** -- the first time this project's tradeable
signal is a cross-instrument RELATIONSHIP rather than any one instrument's
own price/derived series.

Not a duplicate of: R-44/R-53/R-54/R-55/R-58/R-73/R-74 (all external-data
INFO, single-instrument decision), B-16/B-19/B-20/R-57/R-61/R-63
(multi-asset, but every one sums independently-decided single-series bets
at a fixed split -- none computes a spread). This round's parallel NOVEL
branch is a disjoint file, not read, not coordinated with, per
`docs/ROUTINE.md`'s parallelism rules.

**Venue caveat, named explicitly per the task brief.** BTC is
Bitstamp-sourced (`btcusd_spot_5m.csv.gz`); the other 7 instruments are all
Coinbase-sourced. Any BTC-leg pair's spread can therefore carry a genuine
cross-venue basis wobble that a same-venue (all-Coinbase) pair would not --
this is a real confound, not a bookkeeping detail, and is called out again
wherever a BTC pair appears in the results below.

An informal single-pair orientation check by the orchestrator (BTC vs ETH,
Engle-Granger via `statsmodels.tsa.stattools.coint`, 2019-03-14->2020-12-31)
came back p=0.368, not significant. That was a quick peek on one pair with a
different (asymptotic, statsmodels-table) test; it is NOT this branch's
pre-registered gate. This file's own hand-rolled block-bootstrap screen,
covering all 28 pairs, is reported below on its own terms.

=====================================================================
STEP A -- THE PRE-REGISTERED SCREEN (mandatory before any strategy code)
=====================================================================

For every unique pair among the 8 instruments (28 pairs), restricted to
that pair's own overlapping formation-period daily observations, kept only
if >=250 overlapping obs (checked, never relaxed after the count is seen):

1. Resample each instrument's inner-train close to daily (last obs of each
   UTC day), log-transform.
2. OLS `log(A) = alpha + beta*log(B) + e` via plain `numpy.linalg.lstsq`
   (no `statsmodels` dependency added -- this project hand-rolls its own
   bootstrap-based tests rather than using asymptotic critical-value
   tables, and this screen follows that practice).
3. Dickey-Fuller-style regression on the residual: `delta(e_t) = phi*e_{t-1}
   + eps_t` (no drift/trend -- `e_t` is already a demeaned OLS residual),
   `phi` estimated by the identical `lstsq` machinery. A stationary/
   mean-reverting spread has `phi` significantly negative.
4. Null distribution of `phi` under H0 (true random walk): block bootstrap
   -- resample the *observed* `delta(e_t)` series in contiguous blocks of
   20 trading days, with replacement, to the same total length; cumulative-
   sum into a synthetic series `e*_t`; estimate `phi*` by the identical
   regression. 2,000 reps, seed=76. p-value = fraction of null `phi*` <=
   the observed `phi` (left-tail: more negative = more mean-reverting).
5. Both regression directions tried (A on B, B on A); the lower p-value
   direction is kept, per standard (asymmetric) Engle-Granger practice.
6. All 56 tests (28 pairs x 2 directions) ranked by p-value.

**Honest caveat about this specific null, stated before any result is
read and not glossed over in the writeup below:** step 4's bootstrap
resamples the ALREADY-FITTED residual's own differences; it does not
re-estimate alpha/beta on a fresh synthetic pair of random walks each
replicate. The classical Engle-Granger result (Phillips & Ouliaris 1990;
MacKinnon 1994/2010) is that critical values for a unit-root test on an
OLS-generated residual must be notably more negative than plain
Dickey-Fuller ones, precisely because two degrees of freedom were already
spent fitting alpha/beta on the same sample -- a bias this bootstrap does
not correct for, because the task's pre-registered procedure (reproduced
above) fixes alpha/beta once and only resamples the residual's
differences. That makes this specific test LIBERAL (more prone to false
positives from spurious regression, Granger & Newbold 1974) relative to
the standard EG tables. This is not a bug to be silently patched --the
procedure was pre-registered exactly as specified above, run faithfully,
and the caveat is reported alongside every number it could affect.

**Pre-registered stop rule (fixed before any p-value was computed):**
proceed to Step B only if the single best-ranked pair has bootstrap
p<0.05 AND OLS hedge ratio beta in [0.1, 10] AND overlapping formation
sample >=250 daily obs. Otherwise STOP, report the full table as the
round's entire result, build no strategy.

**Configuration count.** The screen itself (56 tests) is a fixed,
non-swept measurement: counts as 0 toward the trials budget, per this
project's established convention for this exact kind of gate study
(R-73/74/75's novel branches).

=====================================================================
STEP B (only if the gate passes) -- the literal pairs trade
=====================================================================

5-minute bars, `alpha`/`beta` frozen from the formation-period DAILY OLS
(never re-estimated on 5m data). Spread recomputed at 5m, z-scored against
a rolling 30-day window (one structural choice, fixed, not swept).
Gatev-Goetzmann-Rouwenhorst thresholds, not fitted: enter |z|>2 (long the
underpriced leg, short the overpriced leg), exit both legs flat at |z|<0.5.
Both legs trade `MarketSpec.futures()` (needed for the short leg -- spot is
long-only in this engine). Composed via `tradebot.multiasset.run_multi_backtest`
at a fixed 50/50 capital split (not swept -- the natural split for a
market-neutral book; only `f` is pre-registered as swept, below).

**Position-size fraction `f`: pre-registered now as exactly 3 structural
values {0.25, 0.5, 1.0}** of each leg's own max notional -- the branch's
entire trials count if Step B runs, else 0.

Evaluated on inner-train (<=2020-12-31, restricted to the qualifying
pair's own data availability) and inner-validation (2021-01-01 ->
2022-12-31), never the holdout. Benchmarks: (a) a fixed 50/50 buy-and-hold
of the same two legs (spot, the risk-matched benchmark for a market-neutral
book), and (b) `kelly_regime_v4` run independently on each leg, futures 5x,
for reference only (not a real benchmark -- it was never built as a
market-neutral book).

=====================================================================
FALSIFICATION TESTS (chosen now, before any Step B result is read)
=====================================================================

1. Formation-window robustness: the identical screen re-run on
   2018-01-01 -> 2021-12-31 (still entirely inside inner-train +
   inner-validation). Does the same pair, or any p<0.05 pair, still
   qualify?
2. Cost sensitivity: the qualifying config (f=1.0, the full-size arm) at
   `fee_rate=0.004` on both legs, inner-validation only, vs the 50/50-hold
   benchmark.

No promotion decision is made here -- measurements only; the orchestrator
decides.

=====================================================================
DATA DISCIPLINE
=====================================================================

`OOS_START = "2023-01-01"`. Every one of the 8 files is truncated to
strictly before that timestamp immediately on load, and re-checked by
`assert_no_holdout`, exactly like `experiments/r75_conservative_dow_signal.py`'s
guard. No bar dated 2023-01-01 or later is ever read, held or printed by
this file. The max timestamp actually read from each file is printed by
`load_all()`.

Usage::

    python experiments/r76_conservative_pairs_cointegration.py screen  # step A, main formation window + gate verdict
    python experiments/r76_conservative_pairs_cointegration.py shift   # falsification 1: shifted formation window
    python experiments/r76_conservative_pairs_cointegration.py stepb   # step B (only proceeds if the gate passed)
    python experiments/r76_conservative_pairs_cointegration.py cost    # falsification 2: 0.40% fee tier
    python experiments/r76_conservative_pairs_cointegration.py all
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_spot, load_dataset  # noqa: E402
from tradebot.multiasset import MultiAssetSpec, run_multi_backtest  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DATA_DIR = ROOT / "data"
OOS_START = "2023-01-01"          # NEVER read in this file

TICKERS = ["BTC", "ETH", "BCH", "DASH", "ETC", "LINK", "LTC", "XTZ"]
VENUE = {"BTC": "Bitstamp", "ETH": "Coinbase", "BCH": "Coinbase", "DASH": "Coinbase",
         "ETC": "Coinbase", "LINK": "Coinbase", "LTC": "Coinbase", "XTZ": "Coinbase"}

FORMATION = ("2017-01-01", "2020-12-31")        # inner-train, main screen
FORMATION_SHIFT = ("2018-01-01", "2021-12-31")  # falsification 1
INNER_TRAIN_END = "2020-12-31"
INNER_VALID = ("2021-01-01", "2022-12-31")

MIN_OBS = 250       # pre-registered floor, never relaxed after the count is seen
N_BOOT = 2000
BLOCK = 20
BOOT_SEED = 76       # this round's number, fixed once, never re-drawn
N_PAIRS = 28
N_TESTS = N_PAIRS * 2   # 56, reported for transparency, counted as 0 (fixed measurement)

ENTRY_Z = 2.0
EXIT_Z = 0.5
ZSCORE_WINDOW = "30D"
ZSCORE_MIN_PERIODS = 4320   # ~15 days of 5m bars: half the window, fixed, not swept
F_VALUES = (0.25, 0.5, 1.0)  # the branch's entire Step-B trials count

FUT_BASE = MarketSpec.futures(leverage=5.0)              # 0.05% taker, funding not charged
FUT_REAL = MarketSpec.futures(leverage=5.0, fee_rate=0.004)  # falsification 2
SPOT = MarketSpec.spot()

OUT_DIR = ROOT / "reports" / "r76_pairs_cointegration"


# ---------------------------------------------------------------- holdout guard


def assert_no_holdout(df: pd.DataFrame) -> None:
    """Hard guard: the max timestamp in any frame this file touches must be
    strictly before OOS_START. Independent of any truncation already done
    at load time."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


# --------------------------------------------------------------------- data


def _load_one(ticker: str) -> pd.DataFrame:
    if ticker == "BTC":
        df, _label = load_dataset(DATA_DIR, "spot")
    else:
        df = load_coinbase_spot(DATA_DIR, ticker)
        if df is None:
            raise RuntimeError(f"data/{ticker.lower()}usd_coinbase_spot_5m.csv.gz not found")
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    df = df.loc[df.index < cutoff].copy()
    assert_no_holdout(df)
    return df


_RAW_CACHE: dict[str, pd.DataFrame] | None = None


def load_all() -> dict[str, pd.DataFrame]:
    """Load all 8 instruments, truncated < OOS_START, guard-checked. Prints
    the max timestamp actually read from each file (the holdout proof)."""
    global _RAW_CACHE
    if _RAW_CACHE is not None:
        return _RAW_CACHE
    raw = {}
    print("Loading 8 instruments (truncated strictly before OOS_START "
          f"= {OOS_START}):", file=sys.stderr)
    for t in TICKERS:
        df = _load_one(t)
        raw[t] = df
        print(f"  {t:5s} ({VENUE[t]:9s}) {len(df):>8,d} bars  "
              f"{df.index.min()} -> {df.index.max()}  (max ts read)", file=sys.stderr)
    _RAW_CACHE = raw
    return raw


def daily_log_prices(raw: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """Daily (last-of-UTC-day) log close, per instrument, over its own
    full pre-holdout history. Formation-window overlap is taken per-pair
    at test time, not forced here."""
    return {t: np.log(df["close"].resample("1D").last().dropna()) for t, df in raw.items()}


# ------------------------------------------------------------- the EG screen


def _overlap(a: pd.Series, b: pd.Series, start: str, end: str) -> tuple[np.ndarray, np.ndarray, int]:
    common = a.index.intersection(b.index)
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    common = common[(common >= lo) & (common <= hi)]
    return a.loc[common].to_numpy(dtype=float), b.loc[common].to_numpy(dtype=float), len(common)


def _ols_lstsq(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """log(y) = alpha + beta*log(x) + e, via plain numpy.linalg.lstsq."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float(coef[0]), float(coef[1]), resid


def _phi_ols(e: np.ndarray) -> float:
    """delta(e_t) = phi*e_{t-1} + eps_t, no drift/trend, via lstsq."""
    lag = e[:-1]
    dy = np.diff(e)
    X = lag.reshape(-1, 1)
    coef, *_ = np.linalg.lstsq(X, dy, rcond=None)
    return float(coef[0])


def _block_bootstrap_null(delta: np.ndarray, n_boot: int = N_BOOT, block: int = BLOCK,
                           seed: int = BOOT_SEED) -> np.ndarray:
    """H0 null for phi: resample the observed delta(e_t) in contiguous
    20-day blocks, with replacement, to the same length; cumsum into a
    synthetic random-walk-like e*_t; re-estimate phi* by the identical
    regression. 2,000 reps."""
    n = len(delta)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        pieces = [delta[s:s + block] for s in starts]
        resampled = np.concatenate(pieces)[:n]
        e_star = np.concatenate(([0.0], np.cumsum(resampled)))
        out[b] = _phi_ols(e_star)
    return out


def _eg_test(y: np.ndarray, x: np.ndarray) -> dict:
    alpha, beta, e = _ols_lstsq(y, x)
    delta = np.diff(e)
    phi = _phi_ols(e)
    null = _block_bootstrap_null(delta)
    pval = float(np.mean(null <= phi))
    return dict(alpha=alpha, beta=beta, phi=phi, pval=pval, n=len(e))


def run_screen(daily: dict[str, pd.Series], formation: tuple[str, str]) -> pd.DataFrame:
    """The full 28-pair, 56-test screen over one formation window."""
    rows = []
    for a, b in itertools.combinations(TICKERS, 2):
        ya, yb, n = _overlap(daily[a], daily[b], *formation)
        if n < MIN_OBS:
            rows.append(dict(pair=f"{a}-{b}", direction="NA", y=None, x=None,
                              alpha=np.nan, beta=np.nan, phi=np.nan, pval=np.nan,
                              n_obs=n, qualifies=False))
            continue
        r1 = _eg_test(ya, yb)
        r1.update(y=a, x=b)
        r2 = _eg_test(yb, ya)
        r2.update(y=b, x=a)
        best = r1 if r1["pval"] <= r2["pval"] else r2
        rows.append(dict(pair=f"{a}-{b}", direction=f"{best['y']}~{best['x']}",
                          y=best["y"], x=best["x"], alpha=best["alpha"], beta=best["beta"],
                          phi=best["phi"], pval=best["pval"], n_obs=n, qualifies=True))
    df = pd.DataFrame(rows).sort_values("pval", na_position="last").reset_index(drop=True)
    return df


def evaluate_gate(df: pd.DataFrame) -> dict:
    """Pre-registered stop rule, applied to the top-ranked qualifying row."""
    q = df[df["qualifies"]].reset_index(drop=True)
    if len(q) == 0:
        return dict(passed=False, reason="no pair had >=250 overlapping formation obs")
    best = q.iloc[0]
    beta_ok = 0.1 <= best["beta"] <= 10.0
    p_ok = best["pval"] < 0.05
    n_ok = best["n_obs"] >= MIN_OBS
    passed = bool(beta_ok and p_ok and n_ok)
    return dict(passed=passed, best=best, beta_ok=beta_ok, p_ok=p_ok, n_ok=n_ok)


def cmd_screen() -> dict:
    raw = load_all()
    daily = daily_log_prices(raw)
    print("\n" + "=" * 100)
    print(f"STEP A -- cointegration screen, formation {FORMATION[0]} -> {FORMATION[1]}  "
          f"({N_PAIRS} pairs x 2 directions = {N_TESTS} tests, counted as 0 configurations)")
    print("=" * 100)
    t0 = time.time()
    df = run_screen(daily, FORMATION)
    print(df.to_string(index=False,
                        formatters={"alpha": "{:.4f}".format, "beta": "{:.4f}".format,
                                    "phi": "{:.5f}".format, "pval": "{:.4f}".format}))
    print(f"\n[{time.time() - t0:.1f}s]  qualifying pairs (n_obs >= {MIN_OBS}): "
          f"{int(df['qualifies'].sum())} / {N_PAIRS}")

    gate = evaluate_gate(df)
    print("\nPRE-REGISTERED STOP RULE: proceed to Step B only if the best-ranked pair has "
          f"p<0.05 AND beta in [0.1,10] AND n_obs>={MIN_OBS}.")
    if gate["passed"]:
        b = gate["best"]
        print(f"GATE: PASS -- {b['pair']} ({b['direction']}), beta={b['beta']:.4f}, "
              f"phi={b['phi']:.5f}, p={b['pval']:.4f}, n_obs={int(b['n_obs'])} "
              f"-> proceed to Step B on this pair.")
        if "BTC" in b["pair"]:
            print("  NOTE: this pair includes BTC (Bitstamp) against a Coinbase leg -- "
                  "the venue caveat applies directly to this result.")
    else:
        print(f"GATE: FAIL ({gate.get('reason', 'best pair does not clear the conjunction')}) "
              "-- STOP, report screen only, build no strategy.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "screen_main.csv", index=False)
    return dict(table=df, gate=gate)


def cmd_shift() -> dict:
    """Falsification 1: identical screen, formation window shifted."""
    raw = load_all()
    daily = daily_log_prices(raw)
    print("\n" + "=" * 100)
    print(f"FALSIFICATION 1 -- identical screen, formation SHIFTED to "
          f"{FORMATION_SHIFT[0]} -> {FORMATION_SHIFT[1]}")
    print("=" * 100)
    t0 = time.time()
    df = run_screen(daily, FORMATION_SHIFT)
    print(df.to_string(index=False,
                        formatters={"alpha": "{:.4f}".format, "beta": "{:.4f}".format,
                                    "phi": "{:.5f}".format, "pval": "{:.4f}".format}))
    print(f"\n[{time.time() - t0:.1f}s]  qualifying pairs (n_obs >= {MIN_OBS}): "
          f"{int(df['qualifies'].sum())} / {N_PAIRS}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "screen_shifted.csv", index=False)
    return dict(table=df)


# ------------------------------------------------------------------ step B


def build_zscore(raw: dict[str, pd.DataFrame], y_ticker: str, x_ticker: str,
                  alpha: float, beta: float) -> pd.Series:
    """5m spread and rolling z-score, frozen alpha/beta from the DAILY
    formation-period OLS (never re-estimated here). Both legs' close
    series are merged onto their union timestamp grid causally
    (forward-fill only ever pulls a PAST observation forward -- no
    lookahead), matching the causal-join pattern of every
    ``align_*_causal`` helper in ``tradebot/data.py``, generalized from a
    daily-into-intraday join to a same-frequency (5m-into-5m) one, so no
    extra day-boundary shift is needed here. The rolling mean/std use a
    calendar-time window (``30D``), which only ever looks backward from
    each row's own timestamp, so the z-score at time T depends on no bar
    timestamped after T."""
    y_close = np.log(raw[y_ticker]["close"])
    x_close = np.log(raw[x_ticker]["close"])
    idx = y_close.index.union(x_close.index)
    y_a = y_close.reindex(idx).ffill()
    x_a = x_close.reindex(idx).ffill()
    spread = (y_a - alpha - beta * x_a).dropna()
    assert_no_holdout(spread.to_frame("spread"))
    roll_mean = spread.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).mean()
    roll_std = spread.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).std()
    return ((spread - roll_mean) / roll_std).rename("zscore")


def _join_zscore_causal(zscore: pd.Series, bars: pd.DataFrame) -> pd.Series:
    """Reindex the (already causal) union-grid z-score onto one leg's own
    bar index. The leg's own timestamps are a subset of the union grid the
    z-score was built on, so this is an exact-match lookup for the normal
    case; ``ffill`` only guards a leg-specific missing bar, and can only
    ever pull a value from an earlier timestamp forward."""
    return zscore.reindex(zscore.index.union(bars.index)).sort_index().ffill().reindex(bars.index)


class PairsLegStrategy(Strategy):
    """One leg of a frozen-hedge-ratio pairs trade (R-76). Reads a
    precomputed z-score column joined causally onto this leg's own bars;
    enters when |z|>2 (GGR 2006 canonical threshold, not fitted), exits
    both legs flat when |z|<0.5. ``own_sign``=+1 for the OLS-dependent
    ("y") leg, -1 for the regressor ("x") leg: the y-leg goes long when z
    is very negative (spread below its trailing mean -> y underpriced
    relative to x) and short when z is very positive; the x-leg mirrors
    it."""

    warmup = 0  # the z-score already carries its own causal warmup (NaN early on)

    def __init__(self, zscore: pd.Series, own_sign: int, f: float,
                 entry_z: float = ENTRY_Z, exit_z: float = EXIT_Z, name: str = "pairs_leg"):
        self.zscore = zscore
        self.own_sign = own_sign
        self.f = f
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.name = name

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["zscore"] = _join_zscore_causal(self.zscore, df)
        return df

    def on_bar(self, ctx) -> None:
        z = ctx.bar["zscore"]
        if not np.isfinite(z):
            return
        if ctx.in_market:
            if abs(z) < self.exit_z:
                ctx.close_position()
            return
        if z > self.entry_z:
            ctx.order_target(-self.own_sign * self.f)
        elif z < -self.entry_z:
            ctx.order_target(self.own_sign * self.f)


def run_pairs(raw: dict[str, pd.DataFrame], y_ticker: str, x_ticker: str,
              zscore: pd.Series, f: float, market: MarketSpec,
              start: str | None, end: str | None, start_balance: float = 1_000.0):
    leg_y = PairsLegStrategy(zscore, own_sign=1, f=f, name=f"pairs_{y_ticker}")
    leg_x = PairsLegStrategy(zscore, own_sign=-1, f=f, name=f"pairs_{x_ticker}")
    specs = [MultiAssetSpec(y_ticker, leg_y, raw[y_ticker], market),
             MultiAssetSpec(x_ticker, leg_x, raw[x_ticker], market)]
    return run_multi_backtest(specs, weights=[0.5, 0.5], start_balance=start_balance,
                               start=start, end=end)


def run_hold_5050(raw: dict[str, pd.DataFrame], y_ticker: str, x_ticker: str,
                   start: str | None, end: str | None, start_balance: float = 1_000.0):
    """Benchmark (a): fixed 50/50 buy-and-hold of the same two legs, spot --
    the risk-matched benchmark for a market-neutral book (not the fully
    invested single-asset buy_and_hold the README table uses)."""
    hold_y = get_strategy("buy_and_hold")
    hold_x = get_strategy("buy_and_hold")
    specs = [MultiAssetSpec(y_ticker, hold_y, raw[y_ticker], SPOT),
             MultiAssetSpec(x_ticker, hold_x, raw[x_ticker], SPOT)]
    return run_multi_backtest(specs, weights=[0.5, 0.5], start_balance=start_balance,
                               start=start, end=end)


def _row(tag: str, window: str, m) -> dict:
    return dict(tag=tag, window=window, final=m.final_balance, sharpe=m.sharpe,
                max_dd=m.max_drawdown_pct, liquidated=m.liquidated, trades=m.num_trades)


def evaluate_stepb(y_ticker: str, x_ticker: str, alpha: float, beta: float) -> pd.DataFrame:
    raw = load_all()
    zscore = build_zscore(raw, y_ticker, x_ticker, alpha, beta)

    # per-pair overlap: neither leg has data before its own real start, so
    # the effective inner-train window is the intersection with FORMATION.
    both_start = max(raw[y_ticker].index.min(), raw[x_ticker].index.min())
    train_start = max(pd.Timestamp(FORMATION[0], tz="UTC"), both_start).strftime("%Y-%m-%d")
    print(f"\nSTEP B -- pair {y_ticker}~{x_ticker}, frozen alpha={alpha:.6f} beta={beta:.6f}")
    print(f"  effective inner-train window for this pair: {train_start} -> {INNER_TRAIN_END} "
          f"(both legs' own data starts {both_start.date()})")
    print(f"  inner-validation window: {INNER_VALID[0]} -> {INNER_VALID[1]}")

    rows = []
    windows = [("train", (train_start, INNER_TRAIN_END)), ("valid", INNER_VALID)]
    for wname, (s, e) in windows:
        hold = run_hold_5050(raw, y_ticker, x_ticker, s, e)
        rows.append(_row("50/50 buy&hold (spot)", wname, hold.metrics))
        for f in F_VALUES:
            res = run_pairs(raw, y_ticker, x_ticker, zscore, f, FUT_BASE, s, e)
            rows.append(_row(f"pairs f={f}", wname, res.metrics))
        # kelly_regime_v4 per leg, futures 5x, reference only -- not summed
        for ticker in (y_ticker, x_ticker):
            kv4 = run_period(get_strategy("kelly_regime_v4"), raw[ticker], s, e,
                              market=FUT_BASE, start_balance=1_000.0)
            from tradebot.metrics import compute_metrics
            rows.append(_row(f"kelly_regime_v4 {ticker}-only (context)", wname, compute_metrics(kv4)))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, formatters={
        "final": "{:,.1f}".format, "sharpe": "{:.3f}".format, "max_dd": "{:.2f}".format}))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "stepb.csv", index=False)
    return df


def cmd_stepb() -> pd.DataFrame | None:
    screen_out = cmd_screen()
    gate = screen_out["gate"]
    if not gate["passed"]:
        print("\nGate did not pass -- Step B is not run. This is the complete, "
              "successful result of this branch (0 Step-B configurations).")
        return None
    b = gate["best"]
    return evaluate_stepb(b["y"], b["x"], b["alpha"], b["beta"])


def cmd_cost() -> pd.DataFrame | None:
    """Falsification 2: the qualifying config (f=1.0) at the 0.40% fee
    tier on both legs, inner-validation only, vs the 50/50-hold benchmark."""
    screen_out = cmd_screen()
    gate = screen_out["gate"]
    if not gate["passed"]:
        print("\nGate did not pass -- there is no qualifying config to cost-test.")
        return None
    b = gate["best"]
    raw = load_all()
    zscore = build_zscore(raw, b["y"], b["x"], b["alpha"], b["beta"])
    print("\n" + "=" * 100)
    print(f"FALSIFICATION 2 -- {b['y']}~{b['x']}, fee_rate=0.004 (0.40% tier), "
          f"inner-validation only, f=1.0 (the qualifying config)")
    print("=" * 100)
    rows = []
    hold = run_hold_5050(raw, b["y"], b["x"], *INNER_VALID)
    rows.append(_row("50/50 buy&hold (spot)", "valid", hold.metrics))
    for f in F_VALUES:
        res = run_pairs(raw, b["y"], b["x"], zscore, f, FUT_REAL, *INNER_VALID)
        rows.append(_row(f"pairs f={f} @0.40%", "valid", res.metrics))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, formatters={
        "final": "{:,.1f}".format, "sharpe": "{:.3f}".format, "max_dd": "{:.2f}".format}))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "cost_sensitivity.csv", index=False)
    return df


# --------------------------------------------------------------------- main


def cmd_all() -> None:
    cmd_screen()
    cmd_shift()
    cmd_stepb()
    cmd_cost()


if __name__ == "__main__":
    cmds = {"screen": cmd_screen, "shift": cmd_shift, "stepb": cmd_stepb,
            "cost": cmd_cost, "all": cmd_all}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/r76_conservative_pairs_cointegration.py "
              f"[{'|'.join(cmds)}]")
