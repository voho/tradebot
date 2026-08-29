"""R-185 CONSERVATIVE branch: kelly_regime_v4 with a bounded, never-increase
multiplicative haircut applied to conviction whenever the vote computed
independently on Coinbase spot (`frac_spot`, exactly v4's own vote) and the
vote computed independently on Deribit's perpetual future (`frac_perp`)
DISAGREE. This is a SCALE-factor veto, not a new vote and not a price
fusion: `frac_spot` (which anchor-vote decides bull/bear) is completely
unchanged from v4 -- only how strongly the strategy acts on it changes, and
only downward, never upward. See ``experiments/r185_shared.py`` for the
frozen pre-registration this branch is evaluated against.

This module is experiment-only: it does not import
``tradebot.registry.register`` and defines no ``@register``-decorated
strategy, so it is never auto-discovered by ``tradebot run``.

Run directly (``python experiments/r185_conservative_disagreement_veto.py``)
to reproduce the causal-safety probe, the pre-registered lambda_veto grid
sweep, and the full inner-validation report used in this round's writeup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r185_shared as shared  # noqa: E402

from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

# Pre-registered lambda_veto grid, fixed BEFORE any inner-validation number
# was read. Selection happens on inner-validation only (allowed -- it is not
# the holdout); inner-train is for fit/iteration/debugging only.
LAMBDA_GRID = (0.10, 0.20, 0.30, 0.40, 0.50)


class KellyRegimeV4DisagreementVeto(Strategy):
    """CONSERVATIVE R-185 branch: bounded, never-increase disagreement veto.

    Byte-for-byte ``kelly_regime_v4`` (vote horizons 20/40/80 days, 1% band,
    conditional-volatility-target SCALE with the 180-day-span hysteresis
    latch, 10% deadband) except the final per-bar `desired` exposure is
    multiplied by ``(1.0 - lambda_veto)`` whenever the spot-vote and the
    Deribit-perp-vote (``shared.spot_and_perp_votes``, computed on each
    venue's own FULL reloaded timeline -- never truncated to whatever
    prefix of `df` `prepare()` happens to receive) disagree at that bar.
    `frac_spot` itself -- which anchor-vote decides bull/bear, and the
    scale before the veto multiplier -- is untouched from v4. One new
    tunable parameter: `lambda_veto` in (0, 1).
    """

    name = "r185_conservative_disagreement_veto"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, asset: str = "BTC", horizons: tuple[int, ...] = (20, 40, 80),
                 band: float = 0.01, target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 vote_gamma: float = 1.0, anchor_span_days: int = 180,
                 high_in: float = 1.70, high_out: float = 1.20,
                 low_in: float = 0.55, low_out: float = 0.85,
                 lambda_veto: float = 0.20) -> None:
        assert asset in ("BTC", "ETH"), f"asset must be 'BTC' or 'ETH', got {asset!r}"
        assert 0.0 < lambda_veto < 1.0, f"lambda_veto must be in (0,1), got {lambda_veto!r}"
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
        self.lambda_veto = lambda_veto

    def _spot_and_perp_votes_on_df_index(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Compute frac_spot / frac_perp on each venue's own FULL, freshly
        reloaded timeline (`shared.spot_and_perp_votes` always reloads the
        full committed spot+perp files from disk regardless of what `df`
        prepare() was called with), then reindex both onto `df.index`.
        Getting this backwards -- computing the votes on a df-sliced window
        -- is exactly the truncation bug R-168's own writeup warns about:
        it would silently damage the rolling-anchor warmup for any `df`
        shorter than the full committed history (which is every call
        `run_period` makes)."""
        spot_df, frac_spot_full, frac_perp_full, _mask = shared.spot_and_perp_votes(self.asset)
        assert df.index.isin(spot_df.index).all(), (
            "prepare() received a df whose index is not a subset of the "
            "freshly-reloaded spot series' index"
        )
        fs_full = pd.Series(frac_spot_full, index=spot_df.index)
        fp_full = pd.Series(frac_perp_full, index=spot_df.index)
        fs = fs_full.reindex(df.index)
        fp = fp_full.reindex(df.index)
        assert fs.notna().all() and fp.notna().all(), (
            "vote fraction has gaps after reindexing onto df.index"
        )
        return fs.to_numpy(), fp.to_numpy()

    # -- byte-for-byte kelly_regime_v3 / v4 mechanism from here on, except
    # the veto multiplier on `desired` -----------------------------------
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]                        # trading/fill price: spot, untouched
        r = np.log(close).diff()                    # SCALE input: original spot returns, untouched

        frac_spot, frac_perp = self._spot_and_perp_votes_on_df_index(df)
        disagree = frac_spot != frac_perp            # discrete-vote disagreement, this bar

        frac = frac_spot.copy()                      # v4's OWN vote, completely unchanged
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
            # -- ONLY new line vs kelly_regime_v4: bounded, never-increase
            # multiplicative haircut when the two independent venue votes
            # disagree. lambda_veto in (0,1) so the multiplier is always in
            # (1-lambda_veto, 1] -- never a boost.
            veto_mult = (1.0 - self.lambda_veto) if disagree[i] else 1.0
            desired = frac[i] * scale * veto_mult
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


# --------------------------------------------------------------------------
# Report generation: causal probe, lambda_veto grid sweep (inner-val
# selection), full 4-cell x 5-lambda signal_check matrix, exposure-matching
# check, and the frozen gate verdict. Run this file directly.
# --------------------------------------------------------------------------

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
    n_configs = 0

    print("=" * 78)
    print("R-185 CONSERVATIVE branch: discrete-vote disagreement veto (spot vs perp)")
    print("=" * 78)

    # ---- causal-safety probe (BTC, sliced to :INNER_VAL_END to stay fast) --
    print("\n[1] Causal truncation probe (BTC, sliced to :INNER_VAL_END, lambda_veto=0.20)")
    btc_spot_df, _fs, _fp, _mask = shared.spot_and_perp_votes("BTC")
    df_probe = btc_spot_df.loc[:shared.INNER_VAL_END].copy()
    causal_ok = shared.causal_truncation_probe(
        lambda: KellyRegimeV4DisagreementVeto(asset="BTC", lambda_veto=0.20),
        df_probe)
    print(f"  causal_truncation_probe -> "
          f"{'PASS (no lookahead detected)' if causal_ok else 'FAIL -- LOOKAHEAD DETECTED'}")
    if not causal_ok:
        print("\n*** CAUSAL SAFETY CHECK FAILED -- fix before trusting any numbers below. ***")

    eth_spot_df, _fs2, _fp2, _mask2 = shared.spot_and_perp_votes("ETH")

    # ---- pre-registered lambda_veto grid sweep, all 4 (asset, market) cells,
    # inner-validation only. Selection uses inner-validation (allowed); this
    # IS the report -- the full grid, not just the winner. ------------------
    print(f"\n[2] lambda_veto grid sweep {LAMBDA_GRID}, "
          f"BTC+ETH x spot+futures_5x, inner-validation "
          f"({shared.INNER_VAL_START} -> {shared.INNER_VAL_END})")

    grid_results = {}  # (asset, market, lam) -> signal_check dict
    for asset, spot_df in (("BTC", btc_spot_df), ("ETH", eth_spot_df)):
        for market_name, market in (("spot", shared.SPOT), ("futures_5x", shared.FUTURES)):
            for lam in LAMBDA_GRID:
                sc = shared.signal_check(
                    lambda asset=asset, lam=lam: KellyRegimeV4DisagreementVeto(
                        asset=asset, lambda_veto=lam),
                    spot_df, market, shared.INNER_VAL_START, shared.INNER_VAL_END)
                grid_results[(asset, market_name, lam)] = sc
                n_configs += 1

    print(f"\n  {'asset':<5} {'market':<11} {'lambda':>7} {'sharpe':>8} {'d_sharpe':>9} "
          f"{'dd_cand':>8} {'dd_v4':>8} {'CI_lo':>9} {'CI_hi':>9} {'sig':>5} "
          f"{'tim_cand':>9} {'tim_v4':>8} {'d_tim(pp)':>10}")
    for (asset, market_name, lam), sc in grid_results.items():
        d_tim = sc["tim_cand"] - sc["tim_v4"]
        print(f"  {asset:<5} {market_name:<11} {lam:>7.2f} {sc['sharpe_cand']:>8.4f} "
              f"{sc['d_sharpe']:>+9.4f} {sc['dd_cand']:>8.4f} {sc['dd_v4']:>8.4f} "
              f"{sc['paired_lo']:>+9.6f} {sc['paired_hi']:>+9.6f} "
              f"{str(sc['significant']):>5} {sc['tim_cand']:>9.4f} {sc['tim_v4']:>8.4f} "
              f"{d_tim:>+10.4f}")

    # ---- selection: best lambda_veto by combined BTC+ETH d_sharpe on
    # futures_5x (the primary sizing-mechanism market; spot is capped so the
    # veto's ceiling effect is muted there), subject to the exposure-match
    # constraint (|d_tim| <= 5pp) on both assets/markets. -------------------
    print("\n[3] Selecting best lambda_veto on inner-validation "
          "(BTC+ETH combined d_sharpe, futures_5x + spot, subject to exposure match)")
    lam_scores = {}
    for lam in LAMBDA_GRID:
        cells = [grid_results[(a, m, lam)] for a in ("BTC", "ETH") for m in ("spot", "futures_5x")]
        matched = all(abs(c["tim_cand"] - c["tim_v4"]) <= 5.0 for c in cells)
        combined_d_sharpe = sum(c["d_sharpe"] for c in cells)
        lam_scores[lam] = (combined_d_sharpe, matched)
        print(f"  lambda_veto={lam:.2f}: combined d_sharpe (4 cells)={combined_d_sharpe:+.4f}  "
              f"all 4 cells exposure-matched (<=5pp)={matched}")

    eligible = {lam: v for lam, v in lam_scores.items() if v[1]}
    if eligible:
        best_lambda = max(eligible, key=lambda lam: eligible[lam][0])
    else:
        best_lambda = max(lam_scores, key=lambda lam: lam_scores[lam][0])
    print(f"\n  Selected lambda_veto = {best_lambda:.2f} "
          f"(exposure-matched on all 4 cells: {best_lambda in eligible})")

    # ---- full report for the selected lambda_veto --------------------------
    print(f"\n[4] Full signal_check report at selected lambda_veto={best_lambda:.2f}, "
          f"inner-validation only")
    final_results = {
        (asset, market_name): grid_results[(asset, market_name, best_lambda)]
        for asset in ("BTC", "ETH") for market_name in ("spot", "futures_5x")
    }
    for (asset, market_name), sc in final_results.items():
        _print_signal_check(f"{asset} {market_name} (lambda_veto={best_lambda:.2f})", sc)

    # ---- exposure-matching check (R-33 rule): explicit pass/fail, all 4
    # cells, at the selected lambda. -----------------------------------------
    print("\n[5] Exposure-matching check (R-33 rule: within 5pp of v4's own time-in-market)")
    exposure_ok = True
    for (asset, market_name), sc in final_results.items():
        d_tim = sc["tim_cand"] - sc["tim_v4"]
        ok = abs(d_tim) <= 5.0
        exposure_ok = exposure_ok and ok
        print(f"  {asset} {market_name}: cand={sc['tim_cand']:.2f}%  v4={sc['tim_v4']:.2f}%  "
              f"delta={d_tim:+.2f}pp  matched(<=5pp)={ok}")

    # ---- frozen inner-validation gate, applied verbatim ---------------------
    print(f"\n[6] Frozen gate verdict (lambda_veto={best_lambda:.2f})")
    btc_fut = final_results[("BTC", "futures_5x")]
    eth_fut = final_results[("ETH", "futures_5x")]
    btc_spot_r = final_results[("BTC", "spot")]
    eth_spot_r = final_results[("ETH", "spot")]

    def clears_magnitude(sc):
        return sc["d_sharpe"] >= 0.20

    def same_direction(a, b):
        return (a["d_sharpe"] >= 0) == (b["d_sharpe"] >= 0)

    def ci_excludes_zero(sc):
        return bool(sc["paired_lo"] > 0 or sc["paired_hi"] < 0)

    def ci_excludes_zero_losing_direction(sc):
        # "losing direction": CI entirely on the side that says candidate is
        # WORSE than v4 (paired_diff = cand - v4, so losing = entirely < 0).
        return sc["paired_hi"] < 0

    for label, sc in final_results.items():
        print(f"  {label}: d_sharpe={sc['d_sharpe']:+.4f}  clears +0.20={clears_magnitude(sc)}  "
              f"CI=[{sc['paired_lo']:+.6f},{sc['paired_hi']:+.6f}]  "
              f"CI_excludes_zero={ci_excludes_zero(sc)}  "
              f"CI_excludes_zero_losing_dir={ci_excludes_zero_losing_direction(sc)}")

    print(f"\n  Total configurations evaluated this branch: {n_configs}")
    print("Done.")
