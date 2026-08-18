"""B-05, Variant B (novel): funding as a continuous carry-drag dampener.

Not registered — lives under ``experiments/`` so it is not auto-discovered
(see ``docs/ROUTINE.md`` step 5) and is not imported by
``tradebot.strategies``. Does not modify any file outside this one.

Pre-registered design (see the orchestrator's pre-registration note,
``B-05`` / Variant B), summarized here so this file is self-contained:

``kelly_regime_v4`` sizes exposure from price alone. Real Binance BTCUSDT
perpetual funding (2020-01-01..2023-12-31, committed at
``data/btcusdt_perp_funding_8h.csv.gz``, loaded via
``tradebot.data.load_funding``) is a second, unused signal that this
project's own research already separately established two things about:

- R-14 (this repo): funding is the strategy family's least-modelled cost —
  it runs about +20%/yr while ``kelly_regime_v4`` holds a position, vs
  +2.8%/yr flat. It is an *adverse-timing* cost, not a flat tax: the
  strategy tends to be long exactly when funding is expensive.
- R-16 (this repo): funding's own trailing quintile rank has real forward-
  return content (14-day forward spread, top vs bottom quintile,
  +3.57pp) that is *not* a momentum proxy (correlation with trailing
  return only 0.39) — but the raw-rate middle quintiles are non-monotone
  and noisy, so a design keyed off the raw rate risks fitting that noise.
  Percentile RANK, not the raw rate, is what survives.

Sources outside this repo: Schmeling, Schrimpf & Todorov, "Crypto Carry"
(BIS Working Paper 1087, 2024) — crypto perpetual funding is a large,
cyclical, priced carry premium tied to boom-bust dynamics, not
idiosyncratic noise, which is why a *slow, regime-scale* trailing window
(months, not hours) is the right time-scale to rank it against. MacLean,
Thorp & Ziemba (2010) — fractional Kelly under estimation error, the
framework ``kelly_regime`` already sits inside.

Mechanism
---------
A fractional-Kelly optimum under a *known, continuous* holding cost ``c``
levied against the position is approximately::

    f* = (mu - c) / sigma^2         vs. the uncosted   f* = mu / sigma^2

so a higher, foreseeable running carry cost should smoothly *shrink*
exposure, in proportion to how expensive carry currently is relative to
its own recent history — not flip a hard veto on/off at one threshold
(that is Variant A, the sibling file). This variant:

1. Runs ``kelly_regime_v4.prepare()`` unchanged to get ``v4_target``.
2. Loads funding via ``tradebot.data.load_funding``. If the file is
   absent (``None``) — including every bar outside 2020-01-01..
   2023-12-31, where the committed file simply has no rows — every
   downstream funding column is all-NaN and the multiplier is
   *identically* 1.0: byte-identical ``kelly_regime_v4`` behaviour.
3. Computes funding's own trailing percentile rank, causally: a rolling
   window over *past settlements only* (``Series.rolling(window).rank(
   pct=True)``, which is a trailing, not centered, window — row t only
   uses settlements <= t), no expanding-over-whole-series statistic.
4. Smooths that rank with a causal EWM (a settlement-count span) to
   control turnover — a hard, unsmoothed threshold flips every 8h
   settlement, which R-12 (this repo) already found to be a turnover
   trap.
5. Merges the smoothed rank onto the 5m bar grid with
   ``pd.merge_asof(..., direction="backward")`` — each bar only ever
   sees the most recent settlement at or before its own timestamp.
6. Applies a piecewise-linear dampener to ``v4_target``:

::

    r <= 0.5:  multiplier(r) = 1.0                              (no-op)
    r  > 0.5:  multiplier(r) = 1.0 - (1.0 - floor) * (r - 0.5) / 0.5

    target = v4_target * multiplier(r)

which is 1.0 (exact no-op) at and below the historical median rank,
ramps linearly down to ``floor`` at the extreme (rank -> 1.0), and is
monotone non-increasing throughout — it only ever *removes* risk in
above-median carry crowding, mirroring the vote-gate family's one robust,
asymmetric-risk finding (see the standing diagnosis in the
pre-registration note), never adds exposure funding never justifies.

Free parameters (by design, kept to two):

- ``floor`` — the multiplier at the most-crowded extreme (rank -> 1.0).
  Swept on funding-train.
- ``smoothing_settlements`` — the EWM span, in settlements (8h each),
  applied to the percentile rank before it drives the multiplier. Swept
  as a turnover-control check, not as a return-hunting knob: it was
  raised from 3 to 9 specifically because floor=0.60/span=3 pushed spot
  fill count to 1.94x v4's (still under the pre-registered *trade*-count
  cap, which stayed at exactly 1.00x throughout, but close enough to the
  fill-level spirit of the cap to be worth backing off).

``rank_window_days`` (default 60, i.e. 180 settlements at 3/day) is fixed,
not swept: the "Crypto Carry" framing is boom-bust dynamics on a scale of
weeks-to-months, and 60 days is the plain middle of that description, not
tuned to this dataset's outcome.

FROZEN CONFIGURATION (see the module-level report / ledger entry for the
full funding-train comparison): ``floor=0.60, smoothing_settlements=9.0,
rank_window_days=60`` — these are the class defaults below.
"""

from __future__ import annotations

import sys
import time
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
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Strategy  # noqa: E402

DATA_DIR = ROOT / "data"


class FundingCarryDampener(KellyRegimeV4):
    """kelly_regime_v4, continuously dampened by funding's trailing percentile rank.

    NOT registered (no ``@register``, deliberately) — pure experiment
    scratch space, per ``docs/ROUTINE.md``.
    """

    name = "funding_carry_dampener"  # not registered; name is for logging only

    def __init__(self, floor: float = 0.60, rank_window_days: float = 60.0,
                 smoothing_settlements: float = 9.0,
                 data_dir: Path = DATA_DIR, **kwargs) -> None:
        super().__init__(**kwargs)
        if not (0.0 <= floor <= 1.0):
            raise ValueError("floor must be in [0, 1]")
        self.floor = floor
        self.rank_window_days = rank_window_days
        self.smoothing_settlements = smoothing_settlements
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------ funding

    def _load_funding(self) -> pd.Series | None:
        return load_funding(self.data_dir)

    def _funding_percentile_rank(self, funding: pd.Series) -> pd.Series:
        """Causal trailing percentile rank of the funding rate, smoothed.

        ``Series.rolling(window).rank(pct=True)`` is a TRAILING window in
        pandas (ends at, includes, the current row) — row t's rank never
        looks at settlement t+1 or later. The subsequent ``.ewm(...)`` is
        likewise trailing/causal by construction.
        """
        window = max(4, int(round(self.rank_window_days * 3)))  # ~3 settlements/day
        min_periods = max(4, window // 4)
        rank = funding.rolling(window, min_periods=min_periods).rank(pct=True)
        if self.smoothing_settlements and self.smoothing_settlements > 1.0:
            rank = rank.ewm(span=self.smoothing_settlements, min_periods=1).mean()
        return rank

    @staticmethod
    def _dampener(rank: np.ndarray, floor: float) -> np.ndarray:
        """Piecewise-linear, monotone non-increasing, no-op at/below median."""
        r = np.asarray(rank, dtype=float)
        above = np.clip((r - 0.5) / 0.5, 0.0, 1.0)
        mult = 1.0 - (1.0 - floor) * above
        return np.where(np.isnan(r), 1.0, mult)  # NaN funding => no-op, never a fabricated value

    # ------------------------------------------------------------- prepare

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4's causal target, unchanged
        df["v4_target"] = df["target"]

        funding = self._load_funding()
        if funding is None or len(funding) == 0:
            # No committed funding data reachable at all: byte-identical v4.
            df["funding_pctrank"] = np.nan
            df["funding_multiplier"] = 1.0
            return df

        rank = self._funding_percentile_rank(funding)
        right = rank.rename("funding_pctrank").to_frame()
        # merge_asof requires matching datetime64 units on both keys; the
        # bar index and the funding CSV's parsed index can differ (ms vs
        # us) even though both are UTC. Aligning the unit changes nothing
        # about *which* settlement matches which bar (still backward/past-
        # only), so this cannot introduce lookahead.
        right.index = right.index.as_unit(df.index.unit)
        # A ``tolerance`` is required: without one, ``merge_asof(direction=
        # "backward")`` carries the LAST available settlement's rank
        # forward forever once the funding series ends (e.g. every bar
        # after 2023-12-31), which is exactly the "no funding info -> must
        # no-op" boundary this strategy must respect. Real settlements are
        # on a clean 8h grid (verified: zero gaps in the committed file),
        # so 12h comfortably bridges the normal cadence while rejecting a
        # stale/ended series.
        merged = pd.merge_asof(
            pd.DataFrame(index=df.index),
            right,
            left_index=True, right_index=True, direction="backward",
            tolerance=pd.Timedelta("12h"),
        )
        r = merged["funding_pctrank"].to_numpy()
        mult = self._dampener(r, self.floor)

        df["funding_pctrank"] = r
        df["funding_multiplier"] = mult
        df["target"] = df["v4_target"].to_numpy() * mult
        return df


# --------------------------------------------------------------------------- eval harness

DF, LABEL = load_dataset(DATA_DIR, "spot")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()
REAL_FUNDING = load_funding(DATA_DIR)

FUNDING_TRAIN = ("2020-01-01", "2021-12-31")
# funding-validation (2022-01-01 onward) is deliberately never referenced
# below - read exactly once, by the orchestrator, after this file is frozen.

N_CONFIGS_EVALUATED = 0  # bumped by every ev() call below that runs a backtest


def run_period_funding(strategy: Strategy, df: pd.DataFrame, start, end, *,
                        market: MarketSpec, start_balance: float = 1_000.0,
                        funding: pd.Series | None = None, data_label: str = ""):
    """``tradebot.window.run_period``, plus a ``funding=`` passthrough.

    ``window.run_period`` (shared module, not touched by this file) does
    not itself forward a ``funding`` kwarg to ``run_backtest``. This
    reimplements exactly its warmup-prefix logic (see
    ``tradebot/window.py``'s docstring / source) so sub-period runs stay
    fair (strategy warms on real bars before ``start``, trades don't count
    against the account until ``start``) while still charging funding —
    the same pattern used in ``experiments/run_gate_control.py:costs()``.
    """
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    if hi <= lo:
        raise ValueError(f"empty period: {start!r} -> {end!r}")
    prefix = int(min(lo, max(strategy.warmup, 0)))
    frame = df.iloc[lo - prefix: hi]
    result = run_backtest(strategy, frame, market, start_balance,
                          data_label=data_label, trade_start=prefix, funding=funding)
    if prefix == 0:
        return result
    return replace(result, equity=result.equity.iloc[prefix:], df=result.df.iloc[prefix:])


def ev(strategy: Strategy, start=None, end=None, *, market: MarketSpec,
       balance: float = 1_000.0, tag: str = "", funding: pd.Series | None = None,
       count: bool = True) -> object:
    global N_CONFIGS_EVALUATED
    if count:
        N_CONFIGS_EVALUATED += 1
    t0 = time.time()
    result = run_period_funding(strategy, DF, start, end, market=market,
                                start_balance=balance, funding=funding, data_label=LABEL)
    m = compute_metrics(result)
    extra = f" funding_paid=${result.funding_paid:>7,.0f}" if market.pays_funding else ""
    print(f"{tag or strategy.name:28s} {market.name:11s} "
          f"final=${m.final_balance:>13,.0f} ({m.profit_pct:>+9.1f}%) "
          f"trades={m.num_trades:>5d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} fees=${m.fees_paid:>7,.0f}{extra} "
          f"{'LIQUIDATED' if m.liquidated else ''} [{time.time() - t0:.0f}s]")
    return m, result


def sweep_funding_train() -> None:
    """Sweep floor on funding-train (2020-2021) only, both markets, real funding."""
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

    print(f"data dir resolves to: {DATA_DIR} (exists={DATA_DIR.exists()})")
    print(f"funding loaded: {'yes, ' + str(len(REAL_FUNDING)) + ' settlements' if REAL_FUNDING is not None else 'NO (falls back to v4)'}")
    print(f"funding-train window: {FUNDING_TRAIN[0]} .. {FUNDING_TRAIN[1]}\n")

    print("--- baseline: kelly_regime_v4 (unmodified) ---")
    ev(KellyRegimeV4(), *FUNDING_TRAIN, market=SPOT, tag="kelly_regime_v4")
    v4_fut_m, _ = ev(KellyRegimeV4(), *FUNDING_TRAIN, market=FUTURES,
                     tag="kelly_regime_v4", funding=REAL_FUNDING)

    print("\n--- floor sweep, smoothing_settlements=3 (~1 day EWM) ---")
    floors = (0.90, 0.75, 0.60, 0.45, 0.30)
    for floor in floors:
        strat_spot = FundingCarryDampener(floor=floor, smoothing_settlements=3.0)
        ev(strat_spot, *FUNDING_TRAIN, market=SPOT, tag=f"dampener floor={floor}")
        strat_fut = FundingCarryDampener(floor=floor, smoothing_settlements=3.0)
        ev(strat_fut, *FUNDING_TRAIN, market=FUTURES, tag=f"dampener floor={floor}",
           funding=REAL_FUNDING)

    print("\n--- smoothing-span check at the chosen floor (turnover control) ---")
    for span in (1.0, 3.0, 9.0, 18.0):
        strat_f = FundingCarryDampener(floor=0.60, smoothing_settlements=span)
        ev(strat_f, *FUNDING_TRAIN, market=FUTURES, tag=f"floor=0.60 span={span}",
           funding=REAL_FUNDING)
        strat_s = FundingCarryDampener(floor=0.60, smoothing_settlements=span)
        ev(strat_s, *FUNDING_TRAIN, market=SPOT, tag=f"floor=0.60 span={span}")

    print(f"\ntotal configurations evaluated: {N_CONFIGS_EVALUATED}")
    print("frozen: floor=0.60, smoothing_settlements=9.0, rank_window_days=60 "
          "(span raised 3 -> 9 to pull spot fill-turnover from 1.94x down "
          "toward v4; trade-count turnover was 1.00x at every span tried)")


if __name__ == "__main__":
    sweep_funding_train()
