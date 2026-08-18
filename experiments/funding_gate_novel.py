"""Variant B of B-05: continuous funding-adjusted Kelly exposure on kelly_regime_v4.

Pre-registration: experiments/funding_gate_preregistration.md

Mechanism: `kelly_regime_v3`/`v4`'s sizing step computes
``desired = frac * scale`` where ``scale = min(target_vol / vol, max_leverage)``
(a stand-in for the growth-optimal ``mu / vol**2``, calibrated through the
constant ``target_vol`` instead of an estimated ``mu``). A continuously
accruing cost ``phi`` (the EWM-smoothed annualized perp funding rate)
subtracts linearly from that same numerator, the continuous analogue of the
way a one-time transaction fee subtracts from the no-trade band derived in
``kelly_regime_ev``'s docstring (Constantinides 1986):

    exposure_adjusted = max(0, desired - k * phi / vol**2)

floored at 0 because the vote fraction is always in [0, 1] (never short), so
there is no negative exposure to protect against. ``k=1`` is the
no-free-parameter case the derivation gives directly; it and the EWM
smoothing span on ``phi`` are swept below for a plateau, not a peak.

NOT @register-ed: this is a research experiment file, not a promoted
strategy, and must not enter `tradebot run`'s matrix.

Usage (from repo root, venv activated):

    python experiments/funding_gate_novel.py sweep         # train + validation grid
    python experiments/funding_gate_novel.py redundancy    # failure mode (a)
    python experiments/funding_gate_novel.py regimesplit   # failure mode (b)
    python experiments/funding_gate_novel.py causality     # by-hand lookahead check
    python experiments/funding_gate_novel.py all
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

# ---------------------------------------------------------------------------
# Splits (the funding file covers 2020-01-01 .. 2023-12-31 only; see prereg).
# HOLDOUT_START exists only so this file can assert it never crosses it --
# this module must never backtest, print, or inspect anything at or after it.
# ---------------------------------------------------------------------------
INNER_TRAIN = ("2020-01-01", "2021-12-31")
INNER_VALID = ("2022-01-01", "2022-12-31")
HOLDOUT_START = pd.Timestamp("2023-01-01", tz="UTC")


def _assert_before_holdout(end: str | None) -> None:
    if end is not None and pd.Timestamp(end, tz="UTC") >= HOLDOUT_START:
        raise ValueError(f"refusing to backtest up to {end}: touches the funding-holdout")


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class FundingAdjustedKellyV4(KellyRegimeV4):
    """kelly_regime_v4 with exposure reduced by a continuous funding cost term.

    Overrides ``prepare()`` (not ``on_bar``) because the funding subtraction
    must happen to ``desired`` *before* the deadband/latch loop that produces
    the final ``target`` column -- so the deadband sees the funding-adjusted
    desired exposure, not the raw vote-times-scale one. The body below is
    ``KellyRegimeV3.prepare()`` copied verbatim (v4 inherits it unchanged,
    only its default ``horizons`` differ) with exactly one insertion: the
    funding term is computed and subtracted where ``desired = frac[i] * scale``
    is formed.

    Causality of the funding term specifically:

    - ``funding.reindex(df.index, method="ffill").fillna(0.0)`` uses only
      settlements at-or-before each bar's own timestamp (a settlement at
      time T is public information from T onward) -- this is the alignment,
      not a lookahead.
    - annualizing (``rate * 3 * 365.25``, 3 settlements/day) is a pointwise
      scalar multiply, so it cannot introduce lookahead.
    - ``.ewm(span=..., min_periods=1).mean()`` on that already-causal series
      is itself causal: bar i's EWM value is a function of bars <= i only.
    - ``vol`` here is the identical ``r.ewm(...).std().shift(1)`` array v3/v4
      already compute -- the ``shift(1)`` is preserved, not recomputed, so
      bar i never sizes off bar i's own return.
    - no statistic (mean/std/quantile/min/max) is taken over the *whole*
      series and applied to early rows anywhere in this file -- the one
      documented lookahead class (R-21) this repo has actually been burned
      by.

    Degrades to exactly ``KellyRegimeV4`` wherever ``funding`` is ``None``,
    empty, or the bar predates the funding file's first settlement (phi=0
    there by construction, never imputed).
    """

    name = "funding_adjusted_kelly_v4"  # not registered; name kept for logging only

    def __init__(self, funding: pd.Series | None = None, k: float = 1.0,
                 funding_span_days: float = 3.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.funding = funding
        self.k = k
        self.funding_span_days = funding_span_days

    def _phi(self, df: pd.DataFrame) -> np.ndarray:
        """Causal EWM-smoothed annualized funding rate, aligned to df.index."""
        if self.funding is None or len(self.funding) == 0:
            return np.zeros(len(df))
        aligned = self.funding.reindex(df.index, method="ffill").fillna(0.0)
        annualized = aligned * 3.0 * 365.25  # 3 settlements/day -> per-year rate
        span_bars = max(1, int(round(self.funding_span_days * BARS_PER_DAY)))
        phi = annualized.ewm(span=span_bars, min_periods=1).mean()
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

        # Realized vol: identical construction to v3/v4, shift(1) preserved --
        # bar i sizes off returns strictly before bar i.
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

        # --- the one insertion: continuous funding-adjusted exposure -------
        phi = self._phi(df)
        with np.errstate(divide="ignore", invalid="ignore"):
            funding_term = self.k * phi / (vol ** 2)
        valid_vol = np.isfinite(vol) & (vol > 0)
        funding_term = np.where(valid_vol & np.isfinite(funding_term), funding_term, 0.0)
        # ---------------------------------------------------------------------

        n = len(df)
        target = np.zeros(n)
        desired_raw = np.zeros(n)
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
            d = frac[i] * scale
            desired_raw[i] = d
            d_adj = max(0.0, d - funding_term[i])  # floor at 0: never go short
            if abs(d_adj - pos) > self.deadband:
                pos = d_adj
            target[i] = pos

        df["target"] = target
        # Diagnostic columns for the pre-registered failure-mode checks below
        # (redundancy with the vote gate, regime split). Underscore-prefixed
        # per this repo's convention (see kelly_regime_ev's `_ev_vol`).
        df["_frac"] = frac
        df["_vol"] = vol
        df["_phi"] = phi
        df["_funding_term"] = funding_term
        df["_desired_raw"] = desired_raw
        return df


# ---------------------------------------------------------------------------
# Data / backtest plumbing (funding_study.py's `_period` pattern -- run_period
# does not support funding, so it cannot be used here)
# ---------------------------------------------------------------------------

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL_FUNDING = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)

K_GRID = (0.5, 1.0, 1.5, 2.0)
SPAN_DAYS_GRID = (1, 3, 7)
CONFIGS = [(k, s) for k in K_GRID for s in SPAN_DAYS_GRID]  # 4 x 3 = 12

N_EVALUATED = 0


def period(strategy, df, market, start, end, funding, balance=1000.0, data_label=""):
    """Backtest over [start, end], warmed on the bars before it, funding charged."""
    _assert_before_holdout(end)
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, df.iloc[lo - pre: hi], market, balance,
                        trade_start=pre, funding=funding, data_label=data_label)
    trimmed = raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
    return compute_metrics(trimmed), raw.funding_paid


def _eval(strategy, start, end, count=True) -> dict:
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    m, funding_paid = period(strategy, DF, FUTURES, start, end, REAL_FUNDING,
                              data_label=LABEL)
    return {"final": m.final_balance,
            "log_growth": float(np.log(max(m.final_balance, 1e-9) / m.start_balance)),
            "dd": m.max_drawdown_pct, "sharpe": m.sharpe, "trades": m.num_trades,
            "funding_paid": funding_paid}


def _print_row(tag: str, row: dict) -> None:
    print(f"  {tag:22s} final=${row['final']:>10,.0f}  logret={row['log_growth']:>+6.3f}  "
          f"DD={row['dd']:>5.1f}%  sharpe={row['sharpe']:>5.2f}  trades={row['trades']:>4d}  "
          f"funding_paid=${row['funding_paid']:>8,.0f}")


def _table(start: str, end: str, label: str) -> pd.DataFrame:
    print(f"\n{label} ({start} .. {end}), futures 5x, funding CHARGED:")
    rows = []
    base = _eval(get_baseline(), start, end, count=False)
    _print_row("kelly_regime_v4 (base)", base)
    rows.append({"k": None, "span_days": None, "tag": "baseline", **base})
    for k, span in CONFIGS:
        strat = FundingAdjustedKellyV4(funding=REAL_FUNDING, k=k, funding_span_days=span)
        row = _eval(strat, start, end)
        tag = f"k={k:g} span={span}d"
        _print_row(tag, row)
        rows.append({"k": k, "span_days": span, "tag": tag, **row})
    return pd.DataFrame(rows)


def get_baseline() -> KellyRegimeV4:
    return KellyRegimeV4()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def sweep() -> pd.DataFrame:
    """The 4 x 3 grid, on inner-train and inner-validation. Every cell counted."""
    train = _table(*INNER_TRAIN, "INNER-TRAIN")
    valid = _table(*INNER_VALID, "INNER-VALIDATION")
    print(f"\nconfigurations evaluated this run: {N_EVALUATED} backtests "
          f"({len(CONFIGS)} distinct (k, span) configs x 2 periods; "
          f"baseline runs not counted as trials)")
    train["split"] = "train"
    valid["split"] = "valid"
    out = pd.concat([train, valid], ignore_index=True)
    out_dir = ROOT / "reports" / "funding_gate_novel"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "sweep.csv", index=False)
    return out


def redundancy(k: float = 1.0, span_days: float = 3.0) -> None:
    """Failure mode (a): does the funding term mostly fire where the vote
    has already gated exposure down, or does it act independently?

    Computed on inner-train + inner-validation only (never the holdout).
    Runs `prepare()` directly (no backtest needed) to read the diagnostic
    columns it wrote.
    """
    strat = FundingAdjustedKellyV4(funding=REAL_FUNDING, k=k, funding_span_days=span_days)
    lo = int(DF.index.searchsorted(INNER_TRAIN[0]))
    hi = int(DF.index.searchsorted(INNER_VALID[1], side="right"))
    _assert_before_holdout(INNER_VALID[1])
    window = DF.iloc[lo:hi]
    prepared = strat.prepare(window.copy())

    frac = prepared["_frac"].to_numpy()
    term = prepared["_funding_term"].to_numpy()
    desired_raw = prepared["_desired_raw"].to_numpy()

    active = np.abs(desired_raw) > 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(active, term / np.maximum(np.abs(desired_raw), 1e-9), np.nan)
    material = active & (relative > 0.10)  # funding term >10% of desired exposure

    gated = frac < 0.999   # vote has already reduced exposure below full
    ungated = frac >= 0.999  # vote is at full conviction (1.0)

    print(f"\nredundancy check (k={k:g}, span={span_days}d), "
          f"{INNER_TRAIN[0]} .. {INNER_VALID[1]}:")
    print(f"  bars with a nonzero desired exposure: {active.sum():,} / {len(prepared):,}")
    print(f"  mean funding_term while vote UNGATED (frac=1.0): "
          f"{np.nanmean(term[active & ungated]):.4f}")
    print(f"  mean funding_term while vote GATED   (frac<1.0): "
          f"{np.nanmean(term[active & gated]):.4f}")
    print(f"  material (funding_term > 10% of desired), UNGATED bars: "
          f"{material[active & ungated].mean():.1%}")
    print(f"  material (funding_term > 10% of desired), GATED bars:   "
          f"{material[active & gated].mean():.1%}")
    corr = np.corrcoef(term[active], (1.0 - frac[active]))[0, 1]
    print(f"  corr(funding_term, 1-frac) over active bars: {corr:+.3f}  "
          "(near 0 = independent of vote state; near +1 = redundant with it)")


def regimesplit() -> pd.DataFrame:
    """Failure mode (b): does the selected config's edge over v4 hold in
    2020, 2021 (train sub-years) and 2022 (validation) alike, or is it one
    episode (the 2021 mania unwind)?
    """
    years = [("2020-01-01", "2020-12-31"), ("2021-01-01", "2021-12-31"),
             ("2022-01-01", "2022-12-31")]
    rows = []
    print("\nregime-split check (edge vs kelly_regime_v4 baseline, per year):")
    for start, end in years:
        base = _eval(get_baseline(), start, end, count=False)
        print(f"\n  {start[:4]}:")
        _print_row("  kelly_regime_v4", base)
        for k, span in CONFIGS:
            strat = FundingAdjustedKellyV4(funding=REAL_FUNDING, k=k, funding_span_days=span)
            row = _eval(strat, start, end, count=False)
            edge = row["log_growth"] - base["log_growth"]
            rows.append({"year": start[:4], "k": k, "span_days": span,
                         "edge_log_growth": edge, "edge_dd": base["dd"] - row["dd"],
                         **row})
    out = pd.DataFrame(rows)
    print("\n  edge (config log-growth minus baseline log-growth) by year, "
          "median across the 12 configs:")
    for year in ("2020", "2021", "2022"):
        sub = out[out.year == year]["edge_log_growth"]
        print(f"    {year}: median {sub.median():+.3f}  range [{sub.min():+.3f}, "
              f"{sub.max():+.3f}]")
    return out


def causality() -> None:
    """By-hand lookahead check (this file is not @register'd, so the
    registry-parametrized `tests/test_causality_strict.py` never sees it).

    Same tamper procedure as R-28/R-32: bars strictly after a cut are
    multiplied by 3 in one copy and divided by 3 in the other; every
    prepared column and every on_bar decision at or before the cut must be
    byte-identical between the two, which is what catches a full-series
    statistic (mean/std/quantile) applied to early rows as well as a plain
    i+1 peek.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    df = DF.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        s = FundingAdjustedKellyV4(funding=REAL_FUNDING, k=1.0, funding_span_days=3.0)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out, prepared

    a, pa = decisions(up)
    b, pb = decisions(down)
    bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
    status = ("FAIL - reads the future at bars " + str(bad) if bad
              else "PASS - decisions unchanged at and before the cut")
    check_cols = ("target", "_frac", "_vol", "_phi", "_funding_term", "_desired_raw")
    worst = max(float(np.nanmax(np.abs(pa[c].to_numpy()[:cut] - pb[c].to_numpy()[:cut])))
                for c in check_cols)
    print(f"  {status}")
    print(f"  max |delta column| before the cut, over {check_cols}: {worst:.3e}  "
          f"{'PASS' if worst < 1e-9 else 'FAIL'}")


COMMANDS = {"sweep": sweep, "redundancy": redundancy, "regimesplit": regimesplit,
            "causality": causality}


def main() -> None:
    if REAL_FUNDING is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "all":
        sweep()
        redundancy()
        regimesplit()
        causality()
    elif choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python experiments/funding_gate_novel.py [{'|'.join(COMMANDS)}|all]")


if __name__ == "__main__":
    main()
