#!/usr/bin/env python
"""R-101 NOVEL branch: causal expanding-window jackknife confidence on kelly_regime_v4.

Not registered: lives under ``experiments/`` per ROUTINE.md step 5. Promote
into ``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea
--------
``kelly_regime_v4`` sizes exposure as ``desired = frac * scale``, where
``frac`` is the 3-anchor regime vote (0, 1/3, 2/3, 1) and ``scale`` is a
volatility target (inherited unchanged from v3's conditional targeting).
R-62 (ledger) established that ``frac`` carries the strategy's whole
risk/drawdown signature and ``scale`` contributes none of it. The
project's standing diagnosis names N~=3 (effective sample size ~=3 regime
events) as one of four binding constraints (FRONTIER.md / LEDGER.md
"standing diagnosis").

This round introduces a third multiplicative factor, ``conf``
(confidence), derived via a Quenouille-Tukey delete-one-group jackknife
(Quenouille 1949, Biometrika 36; Tukey 1958, Ann. Math. Statist. 29
abstract; Efron 1979, "Bootstrap methods: another look at the jackknife",
Ann. Statist. 7(1)) over the six historical stress episodes this project
uses as its standard regime-timing calendar (``STRESS_EPISODES`` below).
For each episode, the leave-one-out estimate is the vote's (``frac``'s)
realized log-growth edge recomputed with that episode's ~120-day window
(+/-60 days) excluded. The coefficient of variation across the leave-one-
out estimates (CV = std/mean, or a normalized range when mean is near
zero) is a literal, formal measurement of "how much does the vote's edge
depend on any single one of our ~3-6 effective regime events" -- an
actual number for the N~=3 diagnosis instead of a qualitative caveat::

    conf = clip(1 - k * CV, conf_floor, 1.0)
    desired = frac * scale * conf

This branch -- NOVEL / CAUSAL EXPANDING
----------------------------------------
Unlike a conservative branch that would freeze ``conf`` as one constant
computed once over the whole history, THIS branch recomputes the
jackknife CV causally, as an expanding-window statistic, using only
stress episodes that have FULLY RESOLVED (their +/-60-day window has
fully closed, by calendar date) as of the current bar. ``conf`` is
therefore a genuinely time-varying series a live deployment could compute
with zero lookahead: at bar i it uses only rows with ``date <= i`` and
only episodes whose window has closed by that date.

Before any episode has resolved, or while fewer than two have resolved,
the jackknife itself is undefined (a "leave-one-out" spread needs at
least two groups to have any dispersion at all), so the pre-registered
rule is::

    conf = conf_floor   (not an undefined jackknife CV)

This is the honest "we know nothing yet" prior: with 0 or 1 resolved
episodes there is no evidence yet about how fragile the vote's edge is to
any one event, so the strategy assumes the worst-permitted confidence
(the floor) rather than fabricating a defined-but-meaningless statistic
from an empty or singleton sample.

k=0 identity harness check
---------------------------
The grid includes k=0. Mathematically ``1 - 0*CV == 1`` for any CV, so a
k=0 run should reproduce ``kelly_regime_v4`` bit-for-bit. To make that
true even during the "fewer than two episodes resolved" period (where the
pre-registered rule above would otherwise substitute ``conf_floor``
regardless of k), the k=0 arm is special-cased in code to set
``conf = 1.0`` unconditionally, bypassing both the floor-default and the
jackknife machinery entirely. This is a deliberate carve-out that exists
ONLY to give the harness a bit-for-bit sanity check on the multiplication
wiring (frac * scale * conf); it is not a claim that k=0 is a sensible
"real" configuration of the confidence mechanism, and it is excluded from
the KS-A/KS-B kill-switch checks and from strategy selection.

Causal warmup, and why this file does not use ``tradebot.window.run_period``
------------------------------------------------------------------------------
The expanding jackknife's state is, by construction, "everything since
the start of the data the strategy has seen" -- not a fixed N-day lookback
like the vote's anchors. ``strategy.warmup`` in this codebase is
overloaded to mean two things at once: (1) how many prior bars
``tradebot.window.run_period`` hands a strategy as prefix before a
sub-period starts, and (2) the frame-relative bar index at which
``tradebot.engine.run_backtest`` starts calling ``on_bar`` at all
(``i >= strategy.warmup`` gates the call itself, not merely order
placement). An earlier version of this file set ``warmup`` to a huge
sentinel to get (1) -- full history -- and broke (2): with a
100-million-bar gate, ``on_bar`` was never called at all, 0 trades on
every configuration including the k=0 identity arm. That bug never
touched a single row of the holdout (it manifested entirely on inner-train
/ inner-validation), but it is recorded here because a "0 trades,
$1,000.00 unchanged" result looks like a valid flat outcome instead of a
broken harness, and the project's own words for exactly this failure mode
are R-24's "$3.7e23 with a green suite" and R-21's "an audit built exactly
that strategy."

The fix: leave ``warmup`` at v4's own value (80 days -- what the vote's
slowest anchor needs to be defined, unrelated to how much history the
jackknife wants), and give the jackknife its long history a different
way -- ``ev()`` below builds the evaluation frame itself as
*every bar of the dataset from its true start through the period's end*,
rather than going through ``run_period``'s ``warmup``-bars prefix. This
is what a live deployment starting at data inception would actually see,
which is the entire premise of "causal expanding" in this branch's name.

Kill switches (pre-registered, checked BEFORE any Sharpe/backtest sweep)
--------------------------------------------------------------------------
KS-A (real dispersion): CV of the ``conf`` series itself, over inner-train
u inner-validation (2017-01-01 -> 2022-12-31), at the a-priori grid cell
k=1.0, conf_floor=0.5, must be >= 0.05. If ``conf`` is nearly constant,
the "confidence, updated as evidence arrives" claim has already failed on
its own terms.

KS-B (not a flat rescale): R^2 of the resulting exposure path
(frac*scale*conf, a-priori cell) against v4's own unmodified exposure
path (frac*scale, i.e. v4's actual ``target`` column) over the same
window must be < 0.95. R^2 >= 0.95 means this is a flat rescale
reproducing the exposure-collapse artifact that has killed most prior
SIZE-axis attempts (R-33's diagnosis; see also R-93/R-97/R-73 in the
long-form ledger for the identical check applied to unrelated
mechanisms) -- not a real mechanism, regardless of what ``conf`` claims
to measure.

If either kill switch fails: STOP. No Sharpe sweep, no holdout read.

Usage
-----
    python experiments/r101_novel_jackknife_causal.py killswitch  # step 0, first
    python experiments/r101_novel_jackknife_causal.py timeline    # episode resolution timeline
    python experiments/r101_novel_jackknife_causal.py sweep       # only if killswitch passes
    python experiments/r101_novel_jackknife_causal.py eth         # falsification test
    python experiments/r101_novel_jackknife_causal.py causality   # strict on_bar peek check
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dataclasses import replace  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

# Inner split only, per ROUTINE.md step 3. The exclusive upper bounds below
# are the ONLY "202[3-9]"-shaped tokens permitted in this file (see the
# grep-audit note at the bottom) and are never passed to a data read that
# would touch the holdout -- they only ever appear as an `end=` argument.
TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
FULL_PRE_HOLDOUT = ("2017-01-01", "2022-12-31")  # inner-train U inner-validation

# The project's standard six-episode regime-timing calendar, copied as
# pre-registered from experiments/r99_shared.py's STRESS_EPISODES (that
# file is not importable from this isolated worktree, so the six dates are
# reproduced verbatim rather than imported -- see the r101 report for the
# note on why).
STRESS_EPISODES = (
    "2018-01-17",  # Jan 2018 top / crash
    "2018-12-15",  # 2018 bear bottom
    "2020-03-12",  # COVID crash
    "2021-11-10",  # Nov 2021 top
    "2022-05-09",  # Terra/Luna collapse
    "2022-11-08",  # FTX collapse
)
EPISODE_HALF_WINDOW_DAYS = 60

N_EVALUATED = 0  # every configuration this file backtests, for deflated Sharpe


# --------------------------------------------------------------------- strategy


class KellyRegimeV4JackknifeCausal(KellyRegimeV4):
    """v4 with desired = frac * scale * conf, conf a causal expanding jackknife CV.

    See module docstring for the full mechanism, the k=0 identity
    carve-out and the causal-warmup rationale.
    """

    name = "kelly_regime_v4_jackknife_causal"  # experimental; not registered

    def __init__(self, k: float = 1.0, conf_floor: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.k = float(k)
        self.conf_floor = float(conf_floor)
        # warmup is intentionally left at v4's own value (see module
        # docstring, "Causal warmup"): it governs the anchor vote's
        # burn-in, not the jackknife's history, which ev() below supplies
        # separately as a full-history evaluation frame.

    # ------------------------------------------------------------ jackknife conf

    def _causal_conf(self, df: pd.DataFrame, frac: np.ndarray, r: np.ndarray) -> np.ndarray:
        """conf[i], using only rows <= i and only episodes resolved by row i's date.

        x[i] = frac[i-1] * r[i]  -- the vote's bar-by-bar log-growth
        contribution, using the exposure decided at the PRIOR bar's close
        applied to the current bar's return (matches the one-bar order/fill
        lag the engine itself uses).

        For a bar-index window [a, b] (episode date +/- EPISODE_HALF_WINDOW_DAYS,
        clipped to the frame), the leave-one-out edge estimate at row i is
        the mean of x over all rows <= i EXCLUDING [a, b] -- computed by
        prefix sums, so it is O(1) per (episode, row) and by construction
        cannot see anything past i.
        """
        n = len(df)
        x = np.concatenate(([0.0], frac[:-1])) * r
        x = np.nan_to_num(x, nan=0.0)
        csum = np.cumsum(x)
        ccount = np.arange(1, n + 1, dtype=float)

        # Causal running std of the raw return series, used only to
        # normalize CV when the leave-one-out mean edge is near zero.
        r0 = np.nan_to_num(r, nan=0.0)
        csum_r = np.cumsum(r0)
        csum_r2 = np.cumsum(r0 * r0)
        run_mean_r = csum_r / ccount
        run_var_r = np.maximum(csum_r2 / ccount - run_mean_r ** 2, 0.0)
        run_std_r = np.sqrt(run_var_r)

        edge_loo = np.full((len(STRESS_EPISODES), n), np.nan)
        resolved = np.zeros((len(STRESS_EPISODES), n), dtype=bool)

        for j, date in enumerate(STRESS_EPISODES):
            center = pd.Timestamp(date, tz="UTC")
            w_start = center - pd.Timedelta(days=EPISODE_HALF_WINDOW_DAYS)
            w_end = center + pd.Timedelta(days=EPISODE_HALF_WINDOW_DAYS)

            # "Resolved as of row i" is a pure calendar comparison against
            # row i's own timestamp -- independent of how far the frame
            # extends, so a frame that stops before w_end (e.g. the
            # inner-validation frame relative to the 2022-11-08 FTX
            # episode's 2023-01-07 window close) correctly NEVER marks
            # that episode resolved, rather than clamping to the last row.
            # (Compared via the tz-aware pandas Index directly, not cast
            # through np.datetime64, which silently drops the tz.)
            res_j = np.asarray(df.index >= w_end)
            resolved[j] = res_j
            if not res_j.any():
                continue

            a = max(int(df.index.searchsorted(w_start)), 0)
            b = min(int(df.index.searchsorted(w_end, side="right")) - 1, n - 1)
            if b < a:
                continue
            wsum = csum[b] - (csum[a - 1] if a > 0 else 0.0)
            wcount = float(b - a + 1)
            denom = ccount - wcount
            with np.errstate(divide="ignore", invalid="ignore"):
                el = (csum - wsum) / denom
            edge_loo[j] = np.where(res_j & (denom > 0), el, np.nan)

        n_resolved = resolved.sum(axis=0)
        enough = n_resolved >= 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_e = np.nanmean(edge_loo, axis=0)
            std_e = np.nanstd(edge_loo, axis=0, ddof=1)
            rng_e = np.nanmax(edge_loo, axis=0) - np.nanmin(edge_loo, axis=0)

        tol = 1e-9
        near_zero = np.abs(mean_e) <= tol
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(near_zero, rng_e / np.maximum(run_std_r, 1e-12),
                          std_e / np.maximum(np.abs(mean_e), tol))
        cv = np.nan_to_num(cv, nan=0.0, posinf=0.0, neginf=0.0)

        if self.k == 0.0:
            # Harness sanity check only -- see module docstring.
            return np.ones(n, dtype=float)

        conf_calc = np.clip(1.0 - self.k * cv, self.conf_floor, 1.0)
        conf = np.where(enough, conf_calc, self.conf_floor)
        # Store the raw CV too, for KS-A/timeline diagnostics.
        self._last_cv = cv
        self._last_n_resolved = n_resolved
        return conf

    # ----------------------------------------------------------------- strategy

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r_series = np.log(close).diff()
        r = r_series.to_numpy()

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

        vol = (r_series.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                    min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        conf = self._causal_conf(df, frac, r)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0
        for i in range(n):
            x_ratio = ratio[i]
            if np.isfinite(x_ratio):
                if state == 0:
                    state = 1 if x_ratio > self.high_in else (-1 if x_ratio < self.low_in else 0)
                elif state == 1 and x_ratio < self.high_out:
                    state = 0
                elif state == -1 and x_ratio > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = frac[i] * scale * conf[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["conf"] = conf
        df["frac"] = frac
        return df


# --------------------------------------------------------------------------- ev


def ev(strategy, start, end, df=None, market=SPOT, tag="", balance=1_000.0, count=True):
    """One backtest over [start, end], printed and counted. end must be pre-holdout.

    Deliberately does NOT use ``tradebot.window.run_period``: that gives a
    strategy only ``strategy.warmup`` bars of prefix before ``start``,
    which is the right amount for the vote/scale anchors (80 days) but
    starves the causal jackknife of the history it is meant to have had
    since data inception (see the module docstring). Instead this frame
    is *every bar of the dataset from its true start through end* -- a
    strict superset of what run_period would give -- with the account
    trimmed to start trading only at ``start``, the same fairness
    run_period provides, just with a longer, more honest prefix. Applied
    uniformly to every strategy evaluated in this file (benchmarks
    included) so every comparison stays apples-to-apples.
    """
    global N_EVALUATED
    assert end is not None and str(end) < "2023-01-01", \
        f"refusing to evaluate past the holdout boundary: end={end!r}"
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    lo = int(frame.index.searchsorted(start))
    hi = int(frame.index.searchsorted(end, side="right"))
    full_prefix_frame = frame.iloc[0:hi]  # every bar from the dataset's true start
    raw = run_backtest(strategy, full_prefix_frame, market, balance,
                       data_label=LABEL, trade_start=lo)
    result = raw if lo == 0 else replace(raw, equity=raw.equity.iloc[lo:], df=raw.df.iloc[lo:])
    m = compute_metrics(result)
    print(f"  {tag or strategy.name:34s} {market.name:11s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"trades={m.num_trades:>5d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f}{'  LIQUIDATED' if m.liquidated else ''}")
    return m


# ------------------------------------------------------------------- killswitch


def killswitch() -> None:
    """Step 0: KS-A and KS-B, at the a-priori cell, BEFORE any Sharpe sweep."""
    strat = KellyRegimeV4JackknifeCausal(k=1.0, conf_floor=0.5)
    start, end = FULL_PRE_HOLDOUT
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    frame = DF.iloc[:hi]  # every bar through end -- causal history from dataset start
    prepared = strat.prepare(frame.copy())
    seg = prepared.iloc[lo:hi]  # inner-train U inner-validation window itself

    conf = seg["conf"].to_numpy()
    mine = seg["target"].to_numpy()

    v4 = KellyRegimeV4()
    v4_prepared = v4.prepare(frame.copy())
    v4_target = v4_prepared["target"].to_numpy()[lo:hi]

    conf_mean, conf_std = float(conf.mean()), float(conf.std(ddof=1))
    conf_cv = conf_std / abs(conf_mean) if abs(conf_mean) > 1e-12 else float("inf")

    ss_res = float(np.sum((mine - v4_target) ** 2))
    ss_tot = float(np.sum((v4_target - v4_target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")

    ks_a_threshold = 0.05
    ks_b_threshold = 0.95
    ks_a_pass = conf_cv >= ks_a_threshold
    ks_b_pass = r2 < ks_b_threshold

    print(f"a-priori cell: k=1.0, conf_floor=0.5, window {start} -> {end} "
          f"({len(seg):,} bars)")
    print(f"conf: mean={conf_mean:.4f} std={conf_std:.4f} "
          f"min={conf.min():.4f} max={conf.max():.4f} unique_values={len(np.unique(np.round(conf, 6)))}")
    print(f"\nKS-A (real dispersion): CV(conf) = {conf_cv:.4f}   "
          f"threshold >= {ks_a_threshold}   -> {'PASS' if ks_a_pass else 'FAIL'}")
    print(f"KS-B (not a flat rescale): R^2(target vs v4 target) = {r2:.4f}   "
          f"threshold < {ks_b_threshold}   -> {'PASS' if ks_b_pass else 'FAIL'}")

    if ks_a_pass and ks_b_pass:
        print("\nBoth kill switches PASS -- proceed to the sweep.")
    else:
        print("\nAt least one kill switch FAILS -- STOP per pre-registration. "
              "No Sharpe/backtest sweep, no holdout read.")
    return ks_a_pass, ks_b_pass, conf_cv, r2


def timeline() -> None:
    """How many episodes are resolved, and when, over the pre-holdout window."""
    print("Episode resolution timeline (date + 60d = window close):")
    resolved_by = []
    for date in STRESS_EPISODES:
        center = pd.Timestamp(date, tz="UTC")
        close_date = center + pd.Timedelta(days=EPISODE_HALF_WINDOW_DAYS)
        resolved_by.append((date, close_date))
        print(f"  {date}  window closes {close_date.date()}")

    for checkpoint in ("2018-06-30", "2019-06-30", "2020-12-31", "2021-12-31", "2022-12-31"):
        cp = pd.Timestamp(checkpoint, tz="UTC")
        n_resolved = sum(1 for _, close_date in resolved_by if close_date <= cp)
        print(f"  as of {checkpoint}: {n_resolved} of {len(STRESS_EPISODES)} episodes resolved")


# ----------------------------------------------------------------------- sweep


def grid_configs():
    """The exact, named-in-advance grid: k=0 identity + 3x3 = 10 configurations."""
    out = [("k=0.0 (identity harness check)", dict(k=0.0, conf_floor=1.0))]
    for k in (0.5, 1.0, 2.0):
        for cf in (0.3, 0.5, 0.7):
            out.append((f"k={k:g} floor={cf:g}", dict(k=k, conf_floor=cf)))
    return out


def _benchmarks(start, end, market, label):
    print(f"\n{label} benchmarks:")
    for name in ("buy_and_hold", "kelly_regime_v4"):
        ev(get_strategy(name), start, end, market=market, tag=f"  {name}", count=False)


def sweep() -> None:
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN, "INNER-TRAIN"), (VALID, "INNER-VALIDATION")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            print(f"{split} / {mname} variants:")
            for tag, kw in grid_configs():
                ev(KellyRegimeV4JackknifeCausal(**kw), start, end, market=market, tag=tag)
    print(f"\nconfigurations evaluated in this run: {N_EVALUATED}")


def eth() -> None:
    """Falsification test on ETH, following R-17's convention (same venue,
    same window, Bitfinex BTC + ETH -- the committed Coinbase ETH file this
    round's pre-registration names does not exist in this isolated
    worktree; the Bitfinex pair is the same substitution R-28's `eth()`
    used for the same reason).

    Only two of the six STRESS_EPISODES (2018-01-17, 2018-12-15) fall
    inside the Bitfinex files' 2016-03 -> 2019-12 range, so the causal
    jackknife only ever has >=2 resolved episodes -- and conf can differ
    from conf_floor -- from 2019-02-13 (episode 2's window close) to
    2019-12-31, about 10.5 months of the ~3.8-year file. That is stated
    here, not hidden.
    """
    for asset, path in (("BTC (control, Bitfinex)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test, Bitfinex)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        assert df.index[-1] < pd.Timestamp("2023-01-01", tz="UTC"), \
            "refusing to proceed: file extends into the holdout"
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for market in (SPOT, FUTURES):
            for name in ("buy_and_hold", "kelly_regime_v4"):
                res = run_backtest(get_strategy(name), df, market, 1_000.0, data_label="bitfinex")
                m = compute_metrics(res)
                print(f"  {name:34s} {market.name:11s} final=${m.final_balance:>11,.0f} "
                      f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
                      f"sharpe={m.sharpe:>5.2f}")
            strat = KellyRegimeV4JackknifeCausal(k=1.0, conf_floor=0.5)
            res = run_backtest(strat, df, market, 1_000.0, data_label="bitfinex")
            m = compute_metrics(res)
            print(f"  {'a-priori k=1.0 floor=0.5':34s} {market.name:11s} "
                  f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
                  f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f}")


# -------------------------------------------------------------------- causality


def causality() -> None:
    """Strict on_bar peek check, run by hand for an unregistered strategy.

    Same two-opposite-tampers procedure tests/test_causality_strict.py uses
    for registered strategies (it only parametrizes over those).
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    # DF extends to "today" (paper-trading updates it past OOS_START), so
    # the tail slice used elsewhere in this pattern (run_eprocess.py's
    # `DF.iloc[-200_000:]`) would land inside the 2023+ holdout here.
    # Bound explicitly to the pre-holdout region first.
    pre_holdout = DF.loc[:"2022-12-31"]
    assert pre_holdout.index[-1] < pd.Timestamp("2023-01-01", tz="UTC")
    df = pre_holdout.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        s = KellyRegimeV4JackknifeCausal(k=1.0, conf_floor=0.5)
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
    print(f"tampered from bar {cut:,} of {len(df):,}; checked bars {bars}")
    print("FAIL - reads the future at bars " + str(bad) if bad
          else "PASS - every decision at or before the cut is unchanged")

    pa = KellyRegimeV4JackknifeCausal(k=1.0, conf_floor=0.5).prepare(up.copy())
    pb = KellyRegimeV4JackknifeCausal(k=1.0, conf_floor=0.5).prepare(down.copy())
    for col in ("target", "conf", "frac"):
        diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
        worst = float(np.nanmax(diff))
        print(f"  column {col:9s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if worst < 1e-9 else 'FAIL'}")


if __name__ == "__main__":
    # Report only the pre-holdout portion's size/range -- DF itself extends
    # past 2023-01-01 (paper trading keeps appending to the committed file),
    # and this project's single strictest rule is to never read or print a
    # bar dated 2023-01-01 or later, so DF.index[-1] is never printed here.
    _pre = DF.loc[:"2022-12-31"]
    print(f"{len(DF):,} bars total in the committed file; using only the "
          f"{len(_pre):,} bars <= 2022-12-31 ({_pre.index[0]:%Y-%m-%d} -> "
          f"{_pre.index[-1]:%Y-%m-%d})  (data: {LABEL})", file=sys.stderr)
    cmds = {"killswitch": killswitch, "timeline": timeline, "sweep": sweep,
            "eth": eth, "causality": causality}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/r101_novel_jackknife_causal.py [{'|'.join(cmds)}]")
