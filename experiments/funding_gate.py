"""Funding gate on kelly_regime_v4's exposure (backlog B-05).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

The idea, one sentence
----------------------
Gate ``kelly_regime_v4``'s existing exposure to zero (or down-scale it)
when the trailing perpetual funding rate is extremely elevated (top
decile), because R-16 found the top funding decile predicts negative
14-day forward returns and R-14 found funding runs ~+20%/yr while the
strategy holds vs +2.8%/yr while flat — the strategy's own crowding sets
the rate it then pays.

Mechanism
---------
``FundingGateKelly`` is ``kelly_regime_v4`` unchanged (same 20/40/80-day
latched vote, same extreme-only inverse-vol sizer, same 10% deadband) with
one more multiplicative factor applied to the finished per-bar
``target`` column:

    target = v4_target * funding_factor

``funding_factor`` is 1.0 everywhere **except** where a causal, trailing
rolling quantile of the funding series says the most recently *settled*
funding print is in the extreme (top ``quantile``) tail of its own
trailing history — there it is either 0.0 (``mode="zero"``, stand flat)
or ``scale`` (``mode="scale"``, e.g. 0.3x the vote/sizer's answer).

Causality of the gate itself
-----------------------------
For a funding print at settlement time ``s``, the threshold it is
compared against is ``rolling_quantile(funding[< s], window_days)`` —
computed with ``.shift(1)`` before the rolling window so the print being
tested is excluded from its own threshold. That gate value is then
forward-filled onto 5-minute price bars from ``s`` until the next
settlement — i.e. a bar knows only the most recent *already-settled*
funding print, exactly as ``run_backtest``'s own funding charging does.

The critical constraint: no effect outside coverage
-----------------------------------------------------
The committed funding file (``data/btcusdt_perp_funding_8h.csv.gz``, via
``tradebot.data.load_funding``) covers 2020-01-01 through 2023-12-31
only. Forward-filling a boolean flag would otherwise let the *last known*
gate state leak indefinitely into 2024+, which is exactly the kind of
"proxy the missing data out of something else" move docs/LEDGER.md
section C rules out. So ``_funding_factor`` explicitly forces the factor
to 1.0 (no gate, plain v4 behaviour) for every bar strictly before the
funding series' first timestamp or strictly after its last timestamp.
Passing ``funding=None`` (or an empty series) reduces the whole strategy
to exactly ``kelly_regime_v4`` — this is the ETH-control / no-data
fallback path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4


class FundingGateKelly(KellyRegimeV4):
    """kelly_regime_v4, gated to zero/reduced exposure on extreme trailing funding.

    Not a duplicate of R-14 (measured the cost, built no strategy), R-16
    (measured the signal, built no strategy), or R-15 (delta-neutral
    funding harvest, a different mechanism). This is the first strategy in
    the project that actually trades on the funding signal.
    """

    name = "funding_gate_kelly"

    def __init__(
        self,
        funding: pd.Series | None = None,
        window_days: float = 60.0,
        quantile: float = 0.90,
        mode: str = "zero",
        scale: float = 0.3,
        min_obs_frac: float = 0.5,
        horizons: tuple[int, ...] = (20, 40, 80),
        **kwargs,
    ) -> None:
        if mode not in ("zero", "scale"):
            raise ValueError(f"mode must be 'zero' or 'scale', got {mode!r}")
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile!r}")
        super().__init__(horizons=horizons, **kwargs)
        self.funding = funding
        self.window_days = window_days
        self.quantile = quantile
        self.mode = mode
        self.scale = scale
        self.min_obs_frac = min_obs_frac

    # ------------------------------------------------------------- the gate

    def _funding_factor(self, bar_index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """(factor, covered) -- factor is 1.0 except a causal extreme tail,
        and forced back to 1.0 outside the funding file's own coverage
        window, so an unsupplied or exhausted funding series is a strict
        no-op. ``covered`` is a separate boolean so callers can report
        "fraction of bars gate was active" as a share of *covered* bars,
        not diluted by the uncovered ones (pre-registered failure mode b).
        """
        n = len(bar_index)
        funding = self.funding
        if funding is None or len(funding) == 0:
            return np.ones(n), np.zeros(n, dtype=bool)
        funding = funding.sort_index()

        # 8h settlements -> ~3/day; half the expected count in the window
        # is a reasonable settling-in period before the quantile is trusted.
        min_periods = max(4, int(round(self.window_days * 3 * self.min_obs_frac)))
        thresh = (funding.shift(1)
                         .rolling(f"{self.window_days}D", min_periods=min_periods)
                         .quantile(self.quantile))
        active = (funding > thresh).fillna(False)  # NaN threshold -> not gated

        # Forward-fill the *settled* gate state onto the 5m bar grid: a bar
        # only ever sees the most recent settlement strictly at-or-before it.
        g = active.reindex(bar_index, method="ffill").fillna(False).to_numpy()

        # No effect outside the file's coverage: never let a stale ffill (or
        # a missing prefix) leak a gate decision into an uncovered period.
        covered = np.asarray((bar_index >= funding.index.min())
                            & (bar_index <= funding.index.max()))
        g = g & covered

        mult = 0.0 if self.mode == "zero" else self.scale
        return np.where(g, mult, 1.0), covered

    # --------------------------------------------------------------- strategy

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)
        factor, covered = self._funding_factor(df.index)
        df["funding_covered"] = covered.astype(float)
        df["funding_gate_active"] = (factor < 0.999).astype(float)
        df["v4_target"] = df["target"].to_numpy()
        df["target"] = df["target"].to_numpy() * factor
        return df
