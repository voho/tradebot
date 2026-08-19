#!/usr/bin/env python
"""Continuous funding-carry haircut on kelly_regime_v4's Kelly numerator (B-05).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered (ROUTINE.md step 5). Promote into
``src/tradebot/strategies/`` only if it clears the promotion bar.

Mechanism (one sentence)
------------------------
Funding is a continuous carry cost proportional to exposure with the same
economic role as a financing rate in a Kelly-optimal leverage formula, so
it belongs inside the growth-optimal fraction's numerator rather than
being handled as a separate all-or-nothing gate — this should produce a
smoother response than a decile cutoff and avoid the whipsaw risk of
trading right at a hard threshold.

Derivation
----------
``kelly_regime_v4`` (and every ancestor in the family) sizes exposure as
fractional-Kelly vol targeting: ``f = target_vol / vol``, capped at
``max_leverage`` and scaled by the regime vote fraction. That rule is the
solution of the classic Kelly/log-growth optimization ``max_f  f*mu -
(sigma^2/2)*f^2``, i.e. ``f* = mu / sigma^2``, evaluated backwards: given
the *chosen* target_vol and the *realized* vol, the rule is implicitly
assuming an annualized drift ``mu_implied = target_vol * vol`` — the drift
that would make ``f = target_vol/vol`` the exact unconstrained Kelly
optimum.

A perpetual future paying funding is economically a levered position
financed at a carry rate ``c`` (funding, annualized) charged on notional
exactly as a repo rate is charged on a financed cash position in classical
portfolio theory (e.g. the financing term in a Merton-style optimal
leverage problem). The position's realized log-growth is then
``f*(mu - c) - (sigma^2/2)*f^2`` instead of ``f*mu - (sigma^2/2)*f^2``, so
the growth-optimal fraction becomes::

    f*_adjusted = (mu - c) / sigma^2 = f* * (1 - c/mu)

Substituting the implied drift above, ``c/mu = c / (target_vol * vol)``,
gives a *continuous, multiplicative* haircut on the existing ``target``
column::

    haircut = clip(1 - funding_k * expected_funding_annualized
                        / (target_vol * vol_of_the_moment),
                    haircut_floor, 1.0)
    target_adjusted = target * haircut

``funding_k`` is a free scale (default 1.0 = the identity substitution
above; the fractional-Kelly discipline this whole strategy family already
uses is itself a similar free scale on the raw Kelly formula, so tuning
``funding_k`` on inner-train/inner-validation is the same kind of choice
this project already makes, not a new one). ``haircut_floor`` (default
0.0) keeps the haircut from flipping the position negative — consistent
with the parent's own behaviour of standing flat rather than shorting a
historically upward-drifting asset.

``expected_funding_annualized`` is a *causal* EWMA of trailing settlement
rates (see ``_merge_expected_funding`` below), never the realized future
rate. Where funding data does not exist (before 2020-01-01, or past
2023-12-31), the haircut defaults to a no-op (multiplier = 1): it behaves
exactly like ``kelly_regime_v4``.

Falsification test (pre-registered)
------------------------------------
Is the haircut curve actually smooth in practice, or does the chosen
normalization scale accidentally saturate the clip almost everywhere,
reproducing a near step-function? Tabulate the haircut multiplier's
10th/25th/50th/75th/90th percentiles (bars with funding data only, selected
config). Bimodal — mass piled at 0 and 1 with a thin middle — falsifies the
"continuous, not a gate" premise this variant exists to test, even if the
backtest numbers look good. See ``falsification_report()``.

Why an EWMA rather than a plain expanding mean
-----------------------------------------------
R-16 (docs/LEDGER.md) found funding's forward-return information decays
over weeks, not years: a full-sample expanding mean would drag in stale
regime information from years earlier and respond far too slowly to the
funding-regime shifts (2020 vs. the 2021 blow-off vs. 2022's bear) that
R-14 shows dominate the annualized cost. An EWMA is this repo's existing
convention for a trailing, regime-adaptive statistic (see ``vol_span`` in
``kelly_regime.py``), so the funding estimate uses the same construction:
an EWMA over trailing *settlements*, shifted so the current settlement's
own rate is never included in the estimate of itself.

Causality, verified three ways (see ``causality_perturbation_check``)
-----------------------------------------------------------------------
1. ``pd.merge_asof(..., direction="backward")`` — a bar only ever sees
   settlements timestamped at or before it.
2. An extra ``.shift(1)`` on the EWMA of that merged column, matching this
   repo's universal per-bar convention (``kelly_regime.py``'s
   ``vol = (...).shift(1)``), on top of (1).
3. An explicit perturbation check: multiply every funding value strictly
   after a cutoff bar by 50x and assert the recomputed ``target`` column
   is byte-identical before the cutoff.
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
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

SETTLEMENTS_PER_YEAR = 3 * 365.25  # perp funding settles every 8h
BARS_PER_SETTLEMENT = 96  # 8h / 5min bars
OOS_START = "2023-01-01"  # never read/evaluate on or after this date


class FundingCarryKelly(KellyRegimeV4):
    """kelly_regime_v4 with a continuous funding-carry haircut on ``target``.

    See the module docstring for the mechanism, the derivation and the
    falsification test. NOT registered on purpose (``@register`` is not
    used) — this is a research variant living under ``experiments/``.
    """

    def __init__(self, funding_k: float = 1.0, funding_ewm_settlements: int = 63,
                 haircut_floor: float = 0.0, data_dir: str | Path = ROOT / "data",
                 funding_series: pd.Series | None = None, pays_funding: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.funding_k = funding_k
        self.funding_ewm_settlements = funding_ewm_settlements
        self.haircut_floor = haircut_floor
        # A strategy's prepare() is not handed the MarketSpec it will
        # trade against (only the OHLCV frame), yet the whole derivation
        # is conditional on actually paying funding. ``pays_funding=False``
        # (pass it when constructing an instance to run on spot) makes the
        # haircut an exact no-op, so a spot backtest of this strategy must
        # reduce byte-for-byte to plain kelly_regime_v4 spot — the sanity
        # check the assignment calls for.
        self.pays_funding = pays_funding
        # Injectable for the causality perturbation check; defaults to the
        # committed real series.
        self._funding = funding_series if funding_series is not None else load_funding(data_dir)
        # Diagnostics stashed by prepare(), read by falsification_report().
        self.last_multiplier: np.ndarray | None = None
        self.last_has_funding: np.ndarray | None = None

    def _merge_expected_funding(self, df: pd.DataFrame) -> np.ndarray:
        """Causal, annualized expected funding rate, one value per bar.

        NaN wherever no funding settlement has occurred yet <= that bar
        (pre-2020, or the EWMA's own min_periods warmup).
        """
        left = pd.DataFrame(index=df.index)
        fdf = self._funding.rename("funding_rate").to_frame()
        # bar timestamps and the committed funding file can carry different
        # datetime64 resolutions (ms vs us); merge_asof requires an exact
        # dtype match, so align the funding index to the bars' resolution.
        fdf.index = fdf.index.astype(df.index.dtype)
        merged = pd.merge_asof(left, fdf, left_index=True, right_index=True,
                                direction="backward")
        raw = merged["funding_rate"]  # bar i sees only settlements <= bar i

        span_bars = max(1, self.funding_ewm_settlements) * BARS_PER_SETTLEMENT
        # EWMA over trailing settlements, then the repo's universal extra
        # per-bar shift(1) so the current bar's own merged value is never
        # part of its own estimate.
        expected_period = (raw.ewm(span=span_bars, min_periods=BARS_PER_SETTLEMENT)
                               .mean().shift(1))
        return (expected_period * SETTLEMENTS_PER_YEAR).to_numpy()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # exact v4 target column, deadband and all
        target = df["target"].to_numpy().copy()
        n = len(df)

        if not self.pays_funding or self._funding is None or len(self._funding) == 0:
            # No funding data committed at all: strict no-op.
            self.last_multiplier = np.ones(n)
            self.last_has_funding = np.zeros(n, dtype=bool)
            df["funding_haircut"] = self.last_multiplier
            return df

        expected_annualized = self._merge_expected_funding(df)
        has_funding = np.isfinite(expected_annualized)

        # vol-of-the-moment, recomputed exactly as the parent computes it
        # (KellyRegime.prepare / KellyRegimeV3.prepare do not store it).
        close = df["close"]
        r = np.log(close).diff()
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        implied_mu = self.target_vol * vol  # see derivation in module docstring

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(implied_mu > 0,
                              self.funding_k * expected_annualized / implied_mu, 0.0)
        ratio = np.where(np.isfinite(ratio) & has_funding, ratio, 0.0)

        multiplier = np.clip(1.0 - ratio, self.haircut_floor, 1.0)
        self.last_multiplier = multiplier
        self.last_has_funding = has_funding

        df["funding_haircut"] = multiplier
        df["target"] = target * multiplier
        return df


# --------------------------------------------------------------------- data

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

INNER_TRAIN = ("2017-01-01", "2020-12-31")
INNER_VALID = ("2021-01-01", "2022-12-31")
FUNDING_WINDOW = ("2020-01-01", "2022-12-31")  # supplementary, still pre-holdout

CONFIGS = [
    dict(funding_k=0.5, funding_ewm_settlements=63),
    dict(funding_k=1.0, funding_ewm_settlements=63),
    dict(funding_k=2.0, funding_ewm_settlements=63),
    dict(funding_k=2.5, funding_ewm_settlements=63),
    dict(funding_k=3.0, funding_ewm_settlements=63),
    dict(funding_k=4.0, funding_ewm_settlements=63),
    dict(funding_k=6.0, funding_ewm_settlements=63),
    dict(funding_k=8.0, funding_ewm_settlements=63),
    dict(funding_k=1.0, funding_ewm_settlements=21),
    dict(funding_k=1.0, funding_ewm_settlements=189),
]


def _period(strategy_factory, market, start=None, end=None, funding=None):
    """Backtest over a date range, warmed on the bars before it.

    Adapted from ``scripts/funding_study.py``'s ``_period`` (per the
    assignment: copy/adapt, don't edit that file). ``tradebot.window.run_period``
    does not forward ``funding=``, so this bypasses it exactly as that
    script does.
    """
    if isinstance(strategy_factory, str):
        strategy = get_strategy(strategy_factory)
    elif callable(strategy_factory):
        strategy = strategy_factory()
    else:
        strategy = strategy_factory
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                        trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw.funding_paid


def _row(label, m, funding_paid=None) -> str:
    paid = f"${funding_paid:>9,.0f}" if funding_paid is not None else " " * 10
    liq = "LIQUIDATED" if m.liquidated else ""
    return (f"{label:34s} ${m.final_balance:>11,.0f} DD={m.max_drawdown_pct:>5.1f}% "
            f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>4d} funding={paid} {liq}")


# ------------------------------------------------------------------- sweep

def sweep_and_select() -> dict:
    """Sweep CONFIGS on inner-train, select on inner-validation only.

    Selection never looks at inner-train numbers, and never at anything
    on/after OOS_START. Returns the selected config dict.
    """
    print(f"Sweeping {len(CONFIGS)} configurations of "
          f"(funding_k, funding_ewm_settlements). Futures 5x, real funding.\n")
    print(f"{'config':28s} | {'inner-train (17-20)':46s} | "
          f"{'inner-validation (21-22)':46s}")
    print("-" * 128)

    train_results, valid_results = [], []
    for cfg in CONFIGS:
        make = lambda cfg=cfg: FundingCarryKelly(**cfg)
        m_train, paid_train = _period(make, FUTURES, *INNER_TRAIN, funding=REAL)
        m_valid, paid_valid = _period(make, FUTURES, *INNER_VALID, funding=REAL)
        train_results.append((cfg, m_train, paid_train))
        valid_results.append((cfg, m_valid, paid_valid))
        tag = f"k={cfg['funding_k']:g} span={cfg['funding_ewm_settlements']}"
        print(f"{tag:28s} | ${m_train.final_balance:>9,.0f} DD={m_train.max_drawdown_pct:>5.1f}% "
              f"S={m_train.sharpe:>5.2f} n={m_train.num_trades:>4d}    | "
              f"${m_valid.final_balance:>9,.0f} DD={m_valid.max_drawdown_pct:>5.1f}% "
              f"S={m_valid.sharpe:>5.2f} n={m_valid.num_trades:>4d}")

    # Select on inner-validation Sharpe, tie-break on drawdown (lower is
    # better), decided now, before looking at anything past 2022-12-31.
    best_cfg, best_m, _ = max(
        valid_results, key=lambda t: (round(t[1].sharpe, 3), -t[1].max_drawdown_pct))
    print(f"\nSelected on inner-validation only: funding_k={best_cfg['funding_k']}, "
          f"funding_ewm_settlements={best_cfg['funding_ewm_settlements']} "
          f"(sharpe={best_m.sharpe:.2f}, DD={best_m.max_drawdown_pct:.1f}%, "
          f"final=${best_m.final_balance:,.0f})")
    return best_cfg


# --------------------------------------------------------------- reporting

def full_report(selected_cfg: dict) -> None:
    windows = [
        ("inner-train 2017-2020", *INNER_TRAIN),
        ("inner-validation 2021-2022", *INNER_VALID),
        ("funding-covered 2020-2022", *FUNDING_WINDOW),
    ]

    for label, start, end in windows:
        print(f"\n=== {label} ({start} .. {end}) ===")
        print("-- futures 5x, real funding charged --")
        m, paid = _period(lambda: FundingCarryKelly(**selected_cfg), FUTURES,
                           start, end, funding=REAL)
        print(_row("funding_carry_kelly (selected)", m, paid))
        m, paid = _period("kelly_regime_v4", FUTURES, start, end, funding=REAL)
        print(_row("kelly_regime_v4 (a: fair baseline)", m, paid))
        m, _ = _period("kelly_regime_v4", FUTURES, start, end, funding=None)
        print(_row("kelly_regime_v4 (b: funding-free, context only)", m))
        m, _ = _period("buy_and_hold", SPOT, start, end, funding=None)
        print(_row("buy_and_hold SPOT (c: promotion bar)", m))

        print("-- spot (sanity: must be ~identical to plain kelly_regime_v4 spot) --")
        m, _ = _period(lambda: FundingCarryKelly(**selected_cfg, pays_funding=False), SPOT, start, end)
        print(_row("funding_carry_kelly spot", m))
        m, _ = _period("kelly_regime_v4", SPOT, start, end)
        print(_row("kelly_regime_v4 spot", m))


def falsification_report(selected_cfg: dict) -> None:
    """Tabulate the haircut multiplier's distribution — the pre-registered test."""
    print("\n=== Falsification test: is the haircut actually continuous? ===")
    strat = FundingCarryKelly(**selected_cfg)
    end_ts = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")
    df = DF[DF.index <= end_ts].copy()
    strat.prepare(df)
    mult = strat.last_multiplier
    has_funding = strat.last_has_funding
    sample = mult[has_funding]
    print(f"bars with funding data (<= {INNER_VALID[1]}): {has_funding.sum():,} / {len(df):,}")
    if len(sample) == 0:
        print("no funding-covered bars in range; cannot evaluate.")
        return
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    pct = np.percentile(sample, [q * 100 for q in qs])
    print("haircut multiplier percentiles (1.0 = no haircut, 0.0 = fully zeroed):")
    for q, v in zip(qs, pct):
        print(f"  p{int(q*100):>2d}  {v:.4f}")
    near0 = float((sample < 0.05).mean())
    near1 = float((sample > 0.95).mean())
    middle = float(((sample >= 0.05) & (sample <= 0.95)).mean())
    print(f"\nmass near 0 (<0.05): {near0:.1%}   mass near 1 (>0.95): {near1:.1%}   "
          f"mass in between: {middle:.1%}")
    if near0 + near1 > 0.85 and middle < 0.15:
        print("VERDICT: bimodal — this looks like a near step-function, NOT continuous. "
              "The 'continuous, not a gate' premise has FAILED its own falsification test.")
    else:
        print("VERDICT: mass is spread across the interval, not clustered at the two "
              "extremes — the haircut behaves as a continuous discount, as designed.")


def causality_perturbation_check(selected_cfg: dict) -> None:
    """Multiply funding strictly after a cutoff by 50x; assert target unchanged before it."""
    print("\n=== Causality perturbation check ===")
    end_ts = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")
    df = DF[DF.index <= end_ts].copy()
    cutoff_ts = pd.Timestamp("2021-06-01", tz="UTC")
    cutoff_i = int(df.index.searchsorted(cutoff_ts, side="right"))

    strat_a = FundingCarryKelly(**selected_cfg, funding_series=REAL)
    out_a = strat_a.prepare(df.copy())["target"].to_numpy()

    perturbed = REAL.copy()
    perturbed.loc[perturbed.index > cutoff_ts] = perturbed.loc[perturbed.index > cutoff_ts] * 50.0
    strat_b = FundingCarryKelly(**selected_cfg, funding_series=perturbed)
    out_b = strat_b.prepare(df.copy())["target"].to_numpy()

    identical_before = np.array_equal(out_a[:cutoff_i], out_b[:cutoff_i])
    changed_after = not np.array_equal(out_a[cutoff_i:], out_b[cutoff_i:])
    print(f"cutoff bar {cutoff_i:,} / {len(df):,}  (ts={cutoff_ts})")
    print(f"target column byte-identical strictly before cutoff: {identical_before}")
    print(f"target column changed on/after cutoff (perturbation had an effect at all): "
          f"{changed_after}")
    if identical_before and changed_after:
        print("PASS: the 50x perturbation after the cutoff has zero effect on any bar "
              "before it, and a nonzero effect after it — no lookahead.")
    else:
        print("FAIL: causality violated or perturbation had no effect anywhere — investigate.")

    # k=0 sanity: the haircut must be an exact no-op reduction to plain v4.
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
    noop = FundingCarryKelly(funding_k=0.0, funding_series=REAL)
    plain = KellyRegimeV4()
    t_noop = noop.prepare(df.copy())["target"].to_numpy()
    t_plain = plain.prepare(df.copy())["target"].to_numpy()
    print(f"funding_k=0.0 reduces exactly to kelly_regime_v4: "
          f"{np.array_equal(t_noop, t_plain)}")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL}); funding: {len(REAL):,} settlements "
          f"{REAL.index[0]:%Y-%m-%d} -> {REAL.index[-1]:%Y-%m-%d}\n", file=sys.stderr)
    # DF spans past OOS_START (it is the full committed dataset), but every
    # call below explicitly bounds end<=2022-12-31 (see the windows lists),
    # so no bar on or after OOS_START is ever read into a backtest here.

    selected = sweep_and_select()
    full_report(selected)
    falsification_report(selected)
    causality_perturbation_check(selected)
