"""R-151 residual decomposition: name what the shared deadband base does
NOT fix, per the pre-registration's PARTIAL clause ("whatever is left is
named, not glossed").

Inner-validation only; no holdout code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from r145_shared import (  # noqa: E402
    CONSERVATIVE_THRESHOLDS,
    INNER_VAL_END,
    INNER_VAL_START,
    fut_market,
    load_btc,
    spot_market,
)
from r145_conservative import FEE_TIERS, make_route_builder, plain_target_slice  # noqa: E402
from r151_shared import arm_kwargs, run_hybrid_backtest_v2  # noqa: E402


def main() -> None:
    df, funding, _ = load_btc()
    target = plain_target_slice(df, INNER_VAL_START, INNER_VAL_END)

    print("=== routing geometry of each threshold (from `target` alone, no broker) ===")
    print(f"target over inner-validation: n={len(target):,} "
          f"min={target.min():.3f} max={target.max():.3f} "
          f"mean={target.mean():.3f} frac>0={np.mean(target > 0):.3f}")
    for thr in CONSERVATIVE_THRESHOLDS:
        spot_frac = np.clip(target, 0.0, thr)
        fut_frac = target - spot_frac
        # The spot leg is unlevered and long-only: `_execute_leg` clamps
        # `target_equiv = min(1.0, frac / leverage)`, so any routed
        # spot_frac above 1.0 is SILENTLY TRUNCATED, not carried.
        clipped = spot_frac > 1.0 + 1e-12
        lost = np.where(clipped, spot_frac - 1.0, 0.0)
        print(f"  thr={thr:.1f}: futures leg engaged {np.mean(fut_frac > 0)*100:6.2f}% of bars "
              f"(mean fut_frac when engaged {fut_frac[fut_frac > 0].mean() if (fut_frac > 0).any() else 0:.3f})"
              f" | spot_frac>1.0 on {np.mean(clipped)*100:6.2f}% of bars, "
              f"mean truncated exposure {lost[clipped].mean() if clipped.any() else 0.0:.3f}x, "
              f"time-weighted {lost.mean():.4f}x")
    print()

    print("=== per-leg throttle behaviour by threshold, arms 'leg' (frozen) and 'shared' (the fix) ===")
    for tier, spot_fee in FEE_TIERS:
        for thr in CONSERVATIVE_THRESHOLDS:
            for arm in ("leg", "shared"):
                r = run_hybrid_backtest_v2(
                    df, make_route_builder(thr), spot_market(spot_fee), fut_market(),
                    funding=funding, start=INNER_VAL_START, end=INNER_VAL_END,
                    **arm_kwargs(arm))
                print(f"  [{tier} thr={thr:.1f} arm={arm:<7}] spot {r.retargets_spot:>4} re-targets / "
                      f"{r.absorbed_spot:>4} absorbed / {r.fills_spot:>4} fills | "
                      f"fut {r.retargets_fut:>4} re-targets / {r.absorbed_fut:>4} absorbed / "
                      f"{r.fills_fut:>4} fills | fees=${r.fees_paid:,.2f} "
                      f"final=${r.final_balance:,.2f}")
    print()

    # The aggregate-throttle hypothesis, MEASURED but deliberately NOT
    # SCORED: an addition after the freeze may only tighten a bar, never
    # loosen one, and a fourth arm that removed the residual would turn
    # this round's own PARTIAL into an ADOPT. So this counts how often the
    # two legs' independent throttle decisions disagree with the single
    # decision the one-venue baseline would have made -- evidence for the
    # diagnosis, not a candidate fix.
    print("=== is the residual an AGGREGATE-vs-PER-LEG throttle problem? (diagnostic, unscored) ===")
    for thr in CONSERVATIVE_THRESHOLDS:
        spot_frac = np.clip(target, 0.0, thr)
        fut_frac = target - spot_frac
        moved = np.abs(np.diff(target, prepend=target[0])) > 1e-9
        d_spot = np.abs(np.diff(spot_frac, prepend=spot_frac[0]))
        d_fut = np.abs(np.diff(fut_frac, prepend=fut_frac[0]))
        d_tot = np.abs(np.diff(target, prepend=target[0]))
        # A re-target is "split" when both legs must move on the same bar:
        # the single-venue baseline throttles |d_tot| once; the hybrid
        # throttles |d_spot| and |d_fut| separately, each strictly smaller.
        split = moved & (d_spot > 1e-9) & (d_fut > 1e-9)
        print(f"  thr={thr:.1f}: {moved.sum():>5} target moves, {split.sum():>5} of them split "
              f"across BOTH legs ({split.sum()/max(moved.sum(),1)*100:5.2f}%); "
              f"on those bars each leg sees on average "
              f"{(np.where(split, np.maximum(d_spot, d_fut), 0).sum() / max(np.where(split, d_tot, 0).sum(), 1e-12))*100:5.1f}% "
              f"of the aggregate move")
    print()


def exposure_paths() -> None:
    """The decisive residual measurement: realized combined notional /
    equity, bar by bar, hybrid vs the plain all-futures baseline.

    R-145's criterion (3) is an EXPOSURE-match test, and fill counts are
    only a proxy for it. Under the fix the hybrid's fill count converges on
    the baseline's at every threshold — so if the volatility mismatch
    survives anyway, the exposure path itself must still differ, and this
    says where.
    """
    from r145_shared import plain_v4_period  # noqa: PLC0415

    df, funding, _ = load_btc()
    plain = plain_v4_period(df, fut_market(), funding, INNER_VAL_START, INNER_VAL_END)
    # The baseline's own realized notional path, from the engine's own
    # position column rather than reconstructed from `target`.
    # `BacktestResult.df` carries no position column, so rebuild the
    # baseline's position path from its own fills: a fill stamped at bar
    # i's timestamp executes at that bar's OPEN, so it is already in the
    # position that marks bar i's CLOSE -- the same bar the equity series
    # records. Forward-fill between fills.
    import pandas as _pd  # noqa: PLC0415

    from tradebot.orders import Side  # noqa: PLC0415

    idx = plain.equity.index
    steps = _pd.Series(0.0, index=idx)
    for f in plain.fills:
        if f.ts in steps.index:
            steps.loc[f.ts] += f.qty if f.side is Side.BUY else -f.qty
    base_pos = steps.cumsum().to_numpy(dtype=float)
    base_exp = (base_pos * plain.df["close"].reindex(idx).to_numpy(dtype=float)
                / plain.equity.to_numpy(dtype=float))

    print("=== realized exposure path (combined notional / equity) vs the baseline ===")
    if base_exp is None:
        print(f"  baseline position column not exposed by BacktestResult.df "
              f"(columns: {list(plain.df.columns)[:12]}...); "
              f"reporting hybrid-vs-hybrid only")
    for tier, spot_fee in FEE_TIERS:
        for thr in CONSERVATIVE_THRESHOLDS:
            row = [f"  [{tier} thr={thr:.1f}]"]
            paths = {}
            for arm in ("leg", "shared"):
                r = run_hybrid_backtest_v2(
                    df, make_route_builder(thr), spot_market(spot_fee), fut_market(),
                    funding=funding, start=INNER_VAL_START, end=INNER_VAL_END,
                    record_exposure=True, **arm_kwargs(arm))
                paths[arm] = r.exposure.to_numpy(dtype=float)
                row.append(f"{arm}: mean_exp={paths[arm].mean():.4f}")
                if base_exp is not None and len(base_exp) == len(paths[arm]):
                    d = paths[arm] - base_exp
                    row.append(f"mean|d_exp|={np.abs(d).mean():.4f} "
                               f"p99|d_exp|={np.percentile(np.abs(d), 99):.4f}")
            if base_exp is not None and len(base_exp) == len(paths["leg"]):
                row.append(f"baseline mean_exp={base_exp.mean():.4f}")
            print("  ".join(row))
    print()


if __name__ == "__main__":
    main()
    exposure_paths()
