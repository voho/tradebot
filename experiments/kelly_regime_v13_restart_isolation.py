#!/usr/bin/env python
"""Isolate kelly_regime_covkelly's rebalance-restart artifact on a SINGLE asset (B-18).

Not registered: lives under ``experiments/`` (ROUTINE.md step 5). Not a
strategy variant -- a diagnostic on the *engine*, not the *signal*.

The question
------------
R-42/R-43 found the same symptom in ``experiments/kelly_regime_covkelly.py``
/ ``_v2.py``'s two-asset (BTC+ETH) dynamic Kelly allocator: at MONTHLY
rebalance cadence, inner-validation (2021-2022) Sharpe goes negative and
loses to baselines; at WEEKLY cadence, identical hyperparameters, it flips
positive and beats them. R-43 assumed this was mean-estimation noise in the
mu/Sigma covariance allocator and built shrinkage fixes -- which helped but
did not eliminate the flip. Backlog item B-18 asks whether the flip is
instead simply an artifact of the rebalance ENGINE: both files' ``_run_leg``
re-instantiates a fresh ``KellyRegimeV4`` and calls
``tradebot.window.run_period`` separately for every segment, and because
``run_period`` only feeds ``prepare()`` an 80-day warmup prefix before each
segment (v4's own ``warmup`` attribute), the ``pos``/``state`` loop inside
``prepare()`` -- a 10%-deadband position latch and a 3-state volatility-
regime latch, both hard-coded to start at 0/flat at row 0 of whatever frame
is passed in -- effectively RESETS at every segment boundary. Years of
accumulated hysteresis are thrown away and rebuilt from ~80 days of local
context, every rebalance.

This file asks the CLEANEST possible version of that question: with NO
two-asset covariance estimation involved at all, how much does restarting
v4's latch state at a fixed cadence cost, purely as a function of cadence,
on a SINGLE asset (BTC spot)? A true continuous run is ground truth; a
restart harness that mirrors ``_run_leg``'s exact mechanics (fresh
instance + ``run_period`` per segment, stitched end-balance-to-start-
balance) at several cadences isolates the artifact; a cheap "warm-start"
variant -- carrying the ACTUAL ending ``pos``/``state`` forward into the
next segment's initial values, instead of resetting to 0/flat -- tests
whether a minimal patch recovers most of the gap (relevant to B-17's
future multi-asset engine work regardless of what this round concludes
about B-18).

Mechanism, one sentence: identical ``kelly_regime_v4`` decision logic
throughout -- only WHERE the pos/state latch starts at each segment
boundary differs across the three harnesses (continuous: never resets;
naive restart: resets to 0/flat every segment, matching
``kelly_regime_covkelly.py::_run_leg`` today; warm-start: resets to the
prior segment's actual ending pos/state).

Not a duplicate of R-42/R-43/the parallel conservative B-18 branch: those
work on the two-asset covariance allocator directly (does full continuous
replay fix the two-asset flip?). This file never estimates a covariance
matrix, never allocates between two assets, and never reads
``kelly_regime_covkelly.py`` -- it is self-contained on the single-asset
engine side, importing only the registered ``kelly_regime_v4`` and
``tradebot.window.run_period``, so its answer is informative about the
GENERAL restart mechanism even if the parallel branch's two-asset-specific
answer differs.

Sources
-------
- This project's own docs/LEDGER.md R-42, R-43, R-49 and backlog items
  B-17/B-18 (the restart-artifact hypothesis is named there, not invented
  here).
- ``src/tradebot/window.py`` docstring: ``run_period`` feeds a strategy's
  own ``warmup`` bars of prefix before the requested start, and nothing
  more -- the mechanism this file measures the cost of.

Hard rules honored
-------------------
- Only this file is touched; nothing under ``src/tradebot/`` or any other
  ``experiments/*.py`` file is modified. ``KellyRegimeV4WarmStart`` below
  is a local, unregistered subclass -- it re-executes v3/v4's latch loop
  verbatim (read, not imported, from ``kelly_regime_v3.py``) starting from
  caller-supplied ``initial_pos``/``initial_state`` instead of the
  hard-coded 0.0/0.
- Data is HARD-SLICED to <= 2022-12-31 immediately after loading (see
  ``LOAD_CUTOFF``) -- every frame used anywhere in this file is a slice of
  that cut. Grep this file for "2023"/"2024"/"2025"/"2026" to confirm no
  such literal appears outside this docstring and the cutoff comment.
- No lookahead: the warm-start variant's ``prepare()`` uses only
  rolling/ewm/shift causal ops, identical to the registered strategy's own
  (verified framework-wide by ``tests/test_causality_strict.py`` for the
  registered classes; this file's own ``causality_check`` below re-verifies
  the LOCAL subclass specifically, since it is new code the framework test
  never exercises).

Usage::

    python experiments/kelly_regime_v13_restart_isolation.py headline    # step 1+2: continuous vs restart cadences
    python experiments/kelly_regime_v13_restart_isolation.py warmstart   # step 4: warm-start patch at monthly cadence
    python experiments/kelly_regime_v13_restart_isolation.py causality   # mandatory no-lookahead check on the warm-start subclass
    python experiments/kelly_regime_v13_restart_isolation.py all         # everything above, in order
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
from tradebot.data import load_dataset  # noqa: E402
from tradebot.metrics import max_drawdown_pct, sharpe_ratio  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

SPOT = MarketSpec.spot()

# --- Data discipline ----------------------------------------------------
TRAIN_START = "2017-01-01"
TRAIN_END = "2020-12-31"
VALID_START = "2021-01-01"
VALID_END = "2022-12-31"
OOS_START = "2023-01-01"             # never read in this file
LOAD_CUTOFF = "2022-12-31 23:55:00"  # hard slice applied immediately after load

N_EVALUATED = 0  # every distinct backtest configuration evaluated


# ============================================================== data load

def load_btc(data_dir: str = "data") -> pd.DataFrame:
    """BTC spot, HARD-SLICED to <= 2022-12-31 immediately after loading.

    Every other function in this file only ever sees frames derived from
    this function's return value, so there is no path here that can reach
    2023+ data.
    """
    df, _label = load_dataset(data_dir, "spot")
    return df.loc[:LOAD_CUTOFF].copy()


# ===================================================== warm-start subclass

class KellyRegimeV4WarmStart(KellyRegimeV4):
    """v4 whose deadband/regime latch starts from caller-supplied state, not 0/flat.

    Local, file-scoped subclass -- does NOT modify
    ``src/tradebot/strategies/kelly_regime_v3.py`` or ``kelly_regime_v4.py``.
    ``prepare()`` below is a byte-for-byte re-derivation of
    ``KellyRegimeV3.prepare()`` (v4 changes only the anchor horizons,
    inherited via ``__init__``, and adds nothing to ``prepare()`` itself),
    with exactly two changes: the loop's ``pos``/``state`` seeds come from
    ``self.initial_pos``/``self.initial_state`` instead of ``0.0``/``0``,
    and the per-bar ``state`` value is stashed in a ``_state`` column so a
    segment harness can read the ending latch state back out afterwards.
    With the defaults (``initial_pos=0.0, initial_state=0``) this produces
    a ``target`` column bit-identical to plain ``KellyRegimeV4()`` -- see
    ``sanity_check_equivalence`` below, run as part of ``causality``.
    """

    name = "kelly_regime_v4_warmstart"  # not @register'd -- not in the comparison table

    def __init__(self, initial_pos: float = 0.0, initial_state: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.initial_pos = initial_pos
        self.initial_state = initial_state

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

        n = len(df)
        target = np.zeros(n)
        state_series = np.zeros(n, dtype=np.int8)
        pos = float(self.initial_pos)
        state = int(self.initial_state)  # 0 normal, +1 high-vol breakout, -1 low-vol breakout
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
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos
            state_series[i] = state

        df["target"] = target
        df["_state"] = state_series
        return df


# =========================================================== segment runner

def _segment_bounds(start: str, end: str, freq: str) -> list[pd.Timestamp]:
    """Same pattern as ``kelly_regime_covkelly.py::_segment_bounds`` (read,
    not imported -- this file is self-contained on the engine side)."""
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    dates = list(pd.date_range(start_ts, end_ts, freq=freq, tz="UTC"))
    if not dates or dates[0] > start_ts:
        dates = [start_ts] + dates
    dates = [d for d in dates if d <= end_ts]
    dates.append(end_ts + pd.Timedelta(days=1))  # sentinel end
    return dates


def run_continuous(df: pd.DataFrame, start: str, end: str, market: MarketSpec,
                    start_balance: float = 1000.0):
    """Step 1: ONE ``run_period`` call over the whole window -- ground truth,
    no restart of any kind."""
    return run_period(KellyRegimeV4(), df, start=start, end=end,
                       market=market, start_balance=start_balance)


def run_naive_restart(df: pd.DataFrame, start: str, end: str, freq: str,
                       market: MarketSpec, start_balance: float = 1000.0) -> dict:
    """Step 2: fresh ``KellyRegimeV4()`` + ``run_period`` per segment, exactly
    ``kelly_regime_covkelly.py::_run_leg``'s pattern, single asset, no
    reallocation (only one asset here -- this isolates the pure restart
    effect from any capital-reallocation question). Segments stitch
    end-balance-to-start-balance, matching ``run_portfolio``'s stitching."""
    bounds = _segment_bounds(start, end, freq)
    balance = start_balance
    pieces = []
    fees_total = 0.0
    for i in range(len(bounds) - 1):
        seg_start, seg_end = bounds[i], bounds[i + 1] - pd.Timedelta(minutes=5)
        if seg_end < seg_start:
            continue
        seg = df.loc[seg_start:seg_end]
        if balance < 1e-6 or len(seg) == 0:
            idx = seg.index if len(seg) else pd.DatetimeIndex([seg_start])
            pieces.append(pd.Series(max(balance, 0.0), index=idx))
            continue
        result = run_period(KellyRegimeV4(), df, start=seg_start, end=seg_end,
                            market=market, start_balance=balance)
        pieces.append(result.equity)
        fees_total += result.fees_paid
        balance = float(result.equity.iloc[-1]) if len(result.equity) else balance
    equity = pd.concat(pieces).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    return {"equity": equity, "fees_paid": fees_total,
            "final_balance": float(equity.iloc[-1]) if len(equity) else start_balance}


def run_warmstart_restart(df: pd.DataFrame, start: str, end: str, freq: str,
                          market: MarketSpec, start_balance: float = 1000.0) -> dict:
    """Step 4: same segment structure as ``run_naive_restart``, but each new
    segment's ``KellyRegimeV4WarmStart`` is seeded with the ACTUAL ending
    ``pos``/``state`` of the previous segment instead of 0/flat."""
    bounds = _segment_bounds(start, end, freq)
    balance = start_balance
    pieces = []
    fees_total = 0.0
    carry_pos, carry_state = 0.0, 0
    for i in range(len(bounds) - 1):
        seg_start, seg_end = bounds[i], bounds[i + 1] - pd.Timedelta(minutes=5)
        if seg_end < seg_start:
            continue
        seg = df.loc[seg_start:seg_end]
        if balance < 1e-6 or len(seg) == 0:
            idx = seg.index if len(seg) else pd.DatetimeIndex([seg_start])
            pieces.append(pd.Series(max(balance, 0.0), index=idx))
            continue
        strat = KellyRegimeV4WarmStart(initial_pos=carry_pos, initial_state=carry_state)
        result = run_period(strat, df, start=seg_start, end=seg_end,
                            market=market, start_balance=balance)
        pieces.append(result.equity)
        fees_total += result.fees_paid
        balance = float(result.equity.iloc[-1]) if len(result.equity) else balance
        if len(result.df):
            carry_pos = float(result.df["target"].iloc[-1])
            carry_state = int(result.df["_state"].iloc[-1])
    equity = pd.concat(pieces).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    return {"equity": equity, "fees_paid": fees_total,
            "final_balance": float(equity.iloc[-1]) if len(equity) else start_balance}


# ================================================================= metrics

def period_metrics(equity: pd.Series, start: str, end: str) -> dict:
    seg = equity.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")]
    arr = seg.to_numpy(dtype=float)
    return {
        "final_balance": float(arr[-1]) if len(arr) else float("nan"),
        "sharpe": sharpe_ratio(arr),
        "max_dd_pct": max_drawdown_pct(arr),
    }


# =================================================================== step 1+2

CADENCES = {
    "weekly": "W-MON",
    "biweekly": "2W-MON",
    "monthly": "MS",
    "quarterly": "QS",
}


def run_headline(data_dir: str = "data") -> dict:
    """Step 1 (continuous baseline) + step 2 (naive-restart at every cadence),
    train and validation reported separately from the same stitched curve."""
    global N_EVALUATED
    df = load_btc(data_dir)

    N_EVALUATED += 1
    cont = run_continuous(df, TRAIN_START, VALID_END, SPOT)
    cont_train = period_metrics(cont.equity, TRAIN_START, TRAIN_END)
    cont_valid = period_metrics(cont.equity, VALID_START, VALID_END)

    cadence_results = {}
    for label, freq in CADENCES.items():
        N_EVALUATED += 1
        res = run_naive_restart(df, TRAIN_START, VALID_END, freq, SPOT)
        train_m = period_metrics(res["equity"], TRAIN_START, TRAIN_END)
        valid_m = period_metrics(res["equity"], VALID_START, VALID_END)
        cadence_results[label] = {"train": train_m, "valid": valid_m}

    print("=== continuous baseline (kelly_regime_v4, unchanged, one run_period call) ===")
    print(f"train  final={cont_train['final_balance']:.1f}  Sharpe={cont_train['sharpe']:.2f}  "
          f"maxDD={cont_train['max_dd_pct']:.1f}%")
    print(f"valid  final={cont_valid['final_balance']:.1f}  Sharpe={cont_valid['sharpe']:.2f}  "
          f"maxDD={cont_valid['max_dd_pct']:.1f}%")

    print("\n=== naive-restart harness, by cadence ===")
    header = f"{'cadence':<10} {'period':<6} {'final':>10} {'sharpe':>8} {'maxDD%':>8}"
    print(header)
    for label in ("weekly", "biweekly", "monthly", "quarterly"):
        for period in ("train", "valid"):
            m = cadence_results[label][period]
            print(f"{label:<10} {period:<6} {m['final_balance']:>10.1f} "
                  f"{m['sharpe']:>8.2f} {m['max_dd_pct']:>8.1f}")

    print(f"\nconfigs evaluated this call: {1 + len(CADENCES)} "
          f"(N_EVALUATED so far: {N_EVALUATED})")
    return {"continuous": {"train": cont_train, "valid": cont_valid}, "cadences": cadence_results}


# ======================================================================= step 4

def run_warmstart(data_dir: str = "data", freq_label: str = "monthly") -> dict:
    """Step 4: warm-start patch at the given cadence (monthly by default,
    where the restart artifact should bite hardest among the required
    weekly/monthly pair) vs both the naive-restart harness and the
    continuous baseline, same window, same metrics."""
    global N_EVALUATED
    df = load_btc(data_dir)
    freq = CADENCES[freq_label]

    N_EVALUATED += 1
    warm = run_warmstart_restart(df, TRAIN_START, VALID_END, freq, SPOT)
    warm_train = period_metrics(warm["equity"], TRAIN_START, TRAIN_END)
    warm_valid = period_metrics(warm["equity"], VALID_START, VALID_END)

    N_EVALUATED += 1
    naive = run_naive_restart(df, TRAIN_START, VALID_END, freq, SPOT)
    naive_train = period_metrics(naive["equity"], TRAIN_START, TRAIN_END)
    naive_valid = period_metrics(naive["equity"], VALID_START, VALID_END)

    N_EVALUATED += 1
    cont = run_continuous(df, TRAIN_START, VALID_END, SPOT)
    cont_train = period_metrics(cont.equity, TRAIN_START, TRAIN_END)
    cont_valid = period_metrics(cont.equity, VALID_START, VALID_END)

    print(f"\n=== warm-start patch, {freq_label} cadence ({freq}) ===")
    header = f"{'variant':<22} {'period':<6} {'final':>10} {'sharpe':>8} {'maxDD%':>8}"
    print(header)
    for name, table in (("continuous baseline", {"train": cont_train, "valid": cont_valid}),
                        ("naive restart", {"train": naive_train, "valid": naive_valid}),
                        ("warm-start restart", {"train": warm_train, "valid": warm_valid})):
        for period in ("train", "valid"):
            m = table[period]
            print(f"{name:<22} {period:<6} {m['final_balance']:>10.1f} "
                  f"{m['sharpe']:>8.2f} {m['max_dd_pct']:>8.1f}")

    def recovered_fraction(cont_v: float, naive_v: float, warm_v: float) -> float:
        gap = cont_v - naive_v
        if abs(gap) < 1e-9:
            return float("nan")
        return float((warm_v - naive_v) / gap)

    frac_sharpe = recovered_fraction(cont_valid["sharpe"], naive_valid["sharpe"], warm_valid["sharpe"])
    print(f"\nvalidation Sharpe gap recovered by warm-start: {frac_sharpe * 100:.1f}%  "
          f"(continuous={cont_valid['sharpe']:.2f}, naive={naive_valid['sharpe']:.2f}, "
          f"warm-start={warm_valid['sharpe']:.2f})")

    print(f"\nconfigs evaluated this call: 3 (N_EVALUATED so far: {N_EVALUATED})")
    return {
        "continuous": {"train": cont_train, "valid": cont_valid},
        "naive": {"train": naive_train, "valid": naive_valid},
        "warmstart": {"train": warm_train, "valid": warm_valid},
        "recovered_fraction_valid_sharpe": frac_sharpe,
    }


# =============================================================== diagnostics

def sanity_check_equivalence(data_dir: str = "data") -> bool:
    """KellyRegimeV4WarmStart(initial_pos=0.0, initial_state=0) must produce
    a target column bit-identical to plain KellyRegimeV4() -- confirms the
    re-derived prepare() is faithful before any conclusion is drawn from it."""
    df = load_btc(data_dir).iloc[:200_000].copy()  # cheap slice, plenty of bars for both
    a = KellyRegimeV4().prepare(df.copy())["target"].to_numpy()
    b = KellyRegimeV4WarmStart(initial_pos=0.0, initial_state=0).prepare(df.copy())["target"].to_numpy()
    ok = np.array_equal(a, b)
    print(f"sanity check: KellyRegimeV4WarmStart(0.0, 0) == KellyRegimeV4() (bit-identical target): {ok}")
    return ok


def causality_check(data_dir: str = "data") -> bool:
    """Mandatory truncation/tamper causality check on the warm-start subclass
    (the most novel code path here). Cut at a fixed date; build two tampered
    copies where every bar strictly AFTER the cut is multiplied by a large
    constant in one and divided by it in the other; recompute prepare() with
    a NON-zero initial_pos/initial_state (to exercise the warm-start seed
    path too, not just the base loop); confirm target/_state at or before
    the cut are bit-identical across both tampered copies and the original.
    """
    df = load_btc(data_dir)
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    K = 137.0

    def tamper(frame: pd.DataFrame, factor: float) -> pd.DataFrame:
        out = frame.copy()
        mask = out.index > cut
        for col in ("open", "high", "low", "close"):
            out.loc[mask, col] = out.loc[mask, col] * factor
        return out

    kwargs = dict(initial_pos=0.37, initial_state=1)
    base = KellyRegimeV4WarmStart(**kwargs).prepare(df.copy())
    up = KellyRegimeV4WarmStart(**kwargs).prepare(tamper(df, K))
    down = KellyRegimeV4WarmStart(**kwargs).prepare(tamper(df, 1.0 / K))

    pre = base.index <= cut
    cols = ["target", "_state"]
    b = base.loc[pre, cols].to_numpy(dtype=float)
    u = up.loc[pre, cols].to_numpy(dtype=float)
    d = down.loc[pre, cols].to_numpy(dtype=float)
    max_diff_up = np.nanmax(np.abs(b - u))
    max_diff_down = np.nanmax(np.abs(b - d))
    ok = max_diff_up < 1e-9 and max_diff_down < 1e-9

    print(f"causality check (KellyRegimeV4WarmStart, initial_pos=0.37, initial_state=1): "
          f"cut={cut.date()}, K={K}")
    print(f"  max |base - up-tampered| before cut (target, _state): {max_diff_up:.3e}")
    print(f"  max |base - down-tampered| before cut (target, _state): {max_diff_down:.3e}")
    print(f"  PASS (values before cut unchanged): {ok}")

    eq_ok = sanity_check_equivalence(data_dir)
    return ok and eq_ok


# ===================================================================== CLI

def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "headline":
        run_headline()
    elif cmd == "warmstart":
        run_warmstart()
    elif cmd == "causality":
        causality_check()
    elif cmd == "all":
        causality_check()
        run_headline()
        run_warmstart()
        print(f"\ntotal N_EVALUATED (backtest configurations, headline + warmstart): {N_EVALUATED}")
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
