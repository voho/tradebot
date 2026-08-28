"""R-168 CONSERVATIVE branch: kelly_regime_v4 with the vote's anchor input
replaced by a causal, equal-weight fusion of Coinbase spot close and
Deribit's independently-transacted perpetual close (BTC-PERPETUAL /
ETH-PERPETUAL). SCALE (the conditional-volatility-target multiplier,
still computed from the ORIGINAL spot return series), the 1% band, the
10% deadband and the latching vol-regime hysteresis are all reused
byte-for-byte, unmodified, from ``kelly_regime_v4`` / ``kelly_regime_v3``
/ ``kelly_regime``. Only the price INPUT to the vote's rolling anchors
changes. See ``experiments/r168_shared.py`` for the frozen
pre-registration this branch is evaluated against.

This module is experiment-only: it does not import
``tradebot.registry.register`` and defines no ``@register``-decorated
strategy, so it is never auto-discovered by ``tradebot run``.

Run directly (``python experiments/r168_conservative_venue_fusion.py``)
to reproduce the causal-safety probe and the full inner-validation
report used in this round's writeup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r168_shared as shared  # noqa: E402

from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402


class _KellyRegimeV4FusedVoteBase(Strategy):
    """Shared mechanism: byte-for-byte ``kelly_regime_v3``/``v4`` sizing
    (conditional-volatility-target SCALE, hysteresis state machine, 10%
    deadband, latch), parameterized identically to v4's own defaults, with
    the vote's anchor input abstracted to :meth:`_vote_close` so the two
    concrete subclasses below (equal-weight fusion, and the perp-only
    bracketing check) can share every other line verbatim.

    Not itself a promotion candidate -- see ``KellyRegimeV4VenueFusion``.
    """

    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, asset: str = "BTC", horizons: tuple[int, ...] = (20, 40, 80),
                 band: float = 0.01, target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 vote_gamma: float = 1.0, anchor_span_days: int = 180,
                 high_in: float = 1.70, high_out: float = 1.20,
                 low_in: float = 0.55, low_out: float = 0.85) -> None:
        assert asset in ("BTC", "ETH"), f"asset must be 'BTC' or 'ETH', got {asset!r}"
        self.asset = asset
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.vote_gamma = vote_gamma
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out

    # -- the ONE axis subclasses vary -----------------------------------
    def _vote_input_close(self, df: pd.DataFrame) -> pd.Series:
        """Return the raw price series (on its own FULL, un-truncated
        timeline -- never pre-sliced to df.index) that the vote's rolling
        anchors are built from. Subclasses implement this; everything else
        in `prepare` is byte-for-byte kelly_regime_v3/v4."""
        raise NotImplementedError

    def _load(self):
        return shared.fused_close_btc() if self.asset == "BTC" else shared.fused_close_eth()

    def _vote_frac(self, df: pd.DataFrame) -> np.ndarray:
        """Compute the vote fraction on the FULL price series' own
        timeline (so every rolling anchor always sees its full causal
        history, never truncated to whatever prefix `df` happens to
        carry), THEN reindex the resulting per-bar fraction onto
        `df.index`. Getting this order backwards -- reindexing the price
        series onto df.index first and rolling on the truncated result --
        silently damages the warmup for any `df` shorter than the full
        committed history (verified by `_warmup_parity_check` below)."""
        vote_close_full = self._vote_input_close(df)
        assert df.index.isin(vote_close_full.index).all(), (
            "prepare() received a df whose index is not a subset of the "
            "freshly-reloaded vote-input series' index"
        )
        frac_full = pd.Series(
            shared.vote_from_close(vote_close_full, horizons=self.horizons, band=self.band),
            index=vote_close_full.index,
        )
        frac = frac_full.reindex(df.index)
        assert frac.notna().all(), "vote fraction has gaps after reindexing onto df.index"
        return frac.to_numpy()

    # -- byte-for-byte kelly_regime_v3 / v4 mechanism from here on ------
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]                       # trading/fill price: spot, untouched
        r = np.log(close).diff()                   # SCALE input: original spot returns, untouched

        frac = self._vote_frac(df)                  # ONLY thing this branch changes
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
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures


class KellyRegimeV4VenueFusion(_KellyRegimeV4FusedVoteBase):
    """CONSERVATIVE R-168 branch: equal-weight venue fusion, zero new
    tunable knobs beyond the ``asset`` selector (which venue/spot pair to
    load -- not a free parameter of the mechanism itself). The vote's
    anchor input is ``(spot_close + perp_close_aligned) / 2``, falling
    back to spot alone wherever Deribit-perp coverage does not yet exist
    (pre-2018-08-14 for BTC, pre-2019-03-14 for ETH) -- exactly
    ``r168_shared.fused_close_btc``/``fused_close_eth``.
    """

    name = "r168_conservative_venue_fusion"

    def _vote_input_close(self, df: pd.DataFrame) -> pd.Series:
        # Deliberately return the FULL, un-truncated fused series (never
        # reindexed onto df.index here -- that happens in `_vote_frac`,
        # AFTER the rolling anchors below are computed on this series' own
        # complete timeline). fused_close_btc/eth always reload the full
        # committed file regardless of what slice of `df` prepare() was
        # called with, so this is insensitive to whatever prefix `df`
        # carries -- what makes the result identical whether prepare()
        # sees a short warmup-prefixed frame (as run_period hands it) or
        # the full history sliced afterward. Also deliberately NOT a
        # value-equality check against `df`: the causal truncation probe
        # intentionally tampers `df`'s OHLC values, and this vote path
        # must stay independent of that tampering (it only ever reads the
        # untouched, freshly-reloaded committed file). See the
        # causal-probe / warmup-parity check at the bottom of this file.
        _spot_df, fused, _perp = self._load()
        return fused


class _KellyRegimeV4PerpOnlyBracket(_KellyRegimeV4FusedVoteBase):
    """NOT a promotion candidate -- item 6's bracketing sanity check only.

    Vote's anchor input is the Deribit perp close alone (weight=1.0),
    degrading to spot wherever perp is unavailable. Used solely to confirm
    the equal-weight fusion (this round's actual branch) sits between
    spot-only (v4 itself, weight=0) and perp-only (weight=1) on the key
    metrics -- the expected pattern for a linear blend, and a cheap bug
    check on the fusion arithmetic above.
    """

    name = "r168_conservative_venue_fusion_perp_only_bracket"

    def _vote_input_close(self, df: pd.DataFrame) -> pd.Series:
        spot_df, _fused, perp = self._load()
        return perp.where(perp.notna(), spot_df["close"])


# --------------------------------------------------------------------------
# Report generation (causal probe, warmup-parity check, 4-cell signal_check
# matrix, and the perp-only bracketing check). Run this file directly.
# --------------------------------------------------------------------------

def _warmup_parity_check(asset: str, market) -> bool:
    """Verify prepare()'s fused-vote construction is insensitive to how
    much history `df` carries: a real run_period call over just the
    inner-validation window (warmup prefix only) must produce the SAME
    `target` values in that window as calling prepare() directly on the
    full history sliced afterward."""
    from tradebot.window import run_period

    spot_df, _fused, _perp = (shared.fused_close_btc() if asset == "BTC"
                               else shared.fused_close_eth())

    # A: real warmup via run_period, only the inner-val window is "live".
    strat_a = KellyRegimeV4VenueFusion(asset=asset)
    res_a = run_period(strat_a, spot_df, start=shared.INNER_VAL_START,
                        end=shared.INNER_VAL_END, market=market, start_balance=1000.0)
    target_a = res_a.df["target"]

    # B: prepare() called on the FULL history, then sliced to the same window.
    strat_b = KellyRegimeV4VenueFusion(asset=asset)
    full_prepared = strat_b.prepare(spot_df.copy())
    target_b = full_prepared.loc[shared.INNER_VAL_START:shared.INNER_VAL_END, "target"]

    common = target_a.index.intersection(target_b.index)
    ok = bool(np.allclose(target_a.loc[common].to_numpy(),
                           target_b.loc[common].to_numpy(), equal_nan=True))
    print(f"  [{asset}] warmup-parity check ({len(common):,} bars compared): "
          f"{'PASS (identical)' if ok else 'FAIL (mismatch)'}")
    return ok


def _print_signal_check(label: str, sc: dict) -> None:
    print(f"\n--- {label} ---")
    print(f"  sharpe: cand={sc['sharpe_cand']:.4f}  v4={sc['sharpe_v4']:.4f}  "
          f"d_sharpe={sc['d_sharpe']:+.4f}")
    print(f"  max_drawdown_pct: cand={sc['dd_cand']:.4f}  v4={sc['dd_v4']:.4f}")
    print(f"  final_balance: cand={sc['final_cand']:.2f}  v4={sc['final_v4']:.2f}")
    print(f"  paired-bootstrap dlog-growth: point={sc['paired_diff']:+.6f}  "
          f"95% CI=[{sc['paired_lo']:+.6f}, {sc['paired_hi']:+.6f}]  "
          f"significant={sc['significant']}")
    print(f"  time_in_market_pct: cand={sc['tim_cand']:.4f}  v4={sc['tim_v4']:.4f}  "
          f"(delta={sc['tim_cand'] - sc['tim_v4']:+.4f} pp)")


if __name__ == "__main__":
    print("=" * 78)
    print("R-168 CONSERVATIVE branch: venue-fusion vote (equal-weight spot+perp)")
    print("=" * 78)

    # ---- causal-safety probe (BTC, sliced to :INNER_VAL_END to stay fast) --
    print("\n[1] Causal truncation probe (BTC, sliced to :INNER_VAL_END)")
    btc_spot_df, _f, _p = shared.fused_close_btc()
    df_probe = btc_spot_df.loc[:shared.INNER_VAL_END].copy()
    causal_ok = shared.causal_truncation_probe(
        lambda: KellyRegimeV4VenueFusion(asset="BTC"), df_probe, shared.FUTURES)
    print(f"  causal_truncation_probe -> {'PASS (no lookahead detected)' if causal_ok else 'FAIL -- LOOKAHEAD DETECTED'}")

    print("\n[2] Warmup-parity check (fused-vote anchors computed on the full")
    print("    reloaded series, reindexed onto df.index afterward)")
    parity_btc = _warmup_parity_check("BTC", shared.FUTURES)
    parity_eth = _warmup_parity_check("ETH", shared.FUTURES)

    if not (causal_ok and parity_btc and parity_eth):
        print("\n*** SAFETY CHECK FAILED -- fix before trusting any numbers below. ***")

    # ---- 4-cell inner-validation signal_check matrix -----------------------
    print("\n[3] Inner-validation signal_check matrix "
          f"({shared.INNER_VAL_START} -> {shared.INNER_VAL_END})")

    eth_spot_df, _f2, _p2 = shared.fused_close_eth()

    results = {}
    results[("BTC", "spot")] = shared.signal_check(
        lambda: KellyRegimeV4VenueFusion(asset="BTC"), btc_spot_df, shared.SPOT,
        shared.INNER_VAL_START, shared.INNER_VAL_END)
    results[("BTC", "futures_5x")] = shared.signal_check(
        lambda: KellyRegimeV4VenueFusion(asset="BTC"), btc_spot_df, shared.FUTURES,
        shared.INNER_VAL_START, shared.INNER_VAL_END)
    results[("ETH", "spot")] = shared.signal_check(
        lambda: KellyRegimeV4VenueFusion(asset="ETH"), eth_spot_df, shared.SPOT,
        shared.INNER_VAL_START, shared.INNER_VAL_END)
    results[("ETH", "futures_5x")] = shared.signal_check(
        lambda: KellyRegimeV4VenueFusion(asset="ETH"), eth_spot_df, shared.FUTURES,
        shared.INNER_VAL_START, shared.INNER_VAL_END)

    for (asset, mkt), sc in results.items():
        _print_signal_check(f"{asset} {mkt}", sc)

    # ---- item 6: perp-only bracketing sanity check (BTC futures_5x only) --
    print("\n[4] Bracketing sanity check: BTC futures_5x, perp-ONLY vote input")
    sc_perp_only = shared.signal_check(
        lambda: _KellyRegimeV4PerpOnlyBracket(asset="BTC"), btc_spot_df, shared.FUTURES,
        shared.INNER_VAL_START, shared.INNER_VAL_END)
    _print_signal_check("BTC futures_5x (perp-only, weight=1.0)", sc_perp_only)

    fused_sharpe = results[("BTC", "futures_5x")]["sharpe_cand"]
    perp_sharpe = sc_perp_only["sharpe_cand"]
    spot_sharpe = results[("BTC", "futures_5x")]["sharpe_v4"]  # v4 IS the weight=0 spot-only arm
    lo, hi = sorted([spot_sharpe, perp_sharpe])
    brackets = lo <= fused_sharpe <= hi
    print(f"\n  spot-only (v4, w=0) sharpe = {spot_sharpe:.4f}")
    print(f"  equal-weight fusion (w=0.5) sharpe = {fused_sharpe:.4f}")
    print(f"  perp-only (w=1.0) sharpe = {perp_sharpe:.4f}")
    print(f"  fusion sits between the two endpoints: {brackets}")

    print("\nDone.")
