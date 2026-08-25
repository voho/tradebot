"""R-132 NOVEL branch — online dual-ascent shadow price on turnover.

Mechanism, as frozen in `r131_shared.py` before any code was written: maintain
a causal shadow price `lambda_t` on turnover, updated every bar by projected
dual ascent,
    lambda_{t+1} = clip(lambda_t + ETA * (turnover_ewm_t - TURNOVER_UPPER), 0, LAMBDA_MAX)
and SHRINK the pending rebalance by `1 / (1 + lambda_t)` rather than skipping
it. `lambda_t` decays back to 0 on its own once trailing turnover falls back
inside the corridor.

Citation: Boyd, Busseti, Diamond, Kahn, Koh, Nystrup & Speth (2017,
Foundations and Trends in Optimization 3(1)), "Multi-Period Trading via Convex
Optimization" — turnover-penalized trading as resource-constrained control.
This branch is the causal, online, dual-variable analogue rather than a
closed-form solve.

Writes `experiments/reports/r132_novel_report.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import r132_eval as E
from r132_mechanisms import NovelTurnoverThrottle
from r131_shared import (
    ETA,
    INNER_VAL_END,
    INNER_VAL_START,
    LAMBDA_MAX,
    STRESS_EPISODES,
    TURNOVER_UPPER,
    V4_NATURAL_TRADES_PER_DAY,
    a2_non_inertness,
)

OUT = Path(__file__).resolve().parent / "reports" / "r132_novel_report.md"

MULTS = [2.0, 2.5, 3.0, 3.5, 4.0]
ETAS = [0.25, 0.5, 1.0]
PRIMARY_MULT = 3.0


def factory(mult: float = PRIMARY_MULT, eta: float = ETA):
    return lambda: NovelTurnoverThrottle(upper=mult * V4_NATURAL_TRADES_PER_DAY, eta=eta)


def lambda_at_episodes(cell: E.Cell) -> pd.DataFrame:
    """The branch's own named diagnostic: does `lambda` spike WITH the turnover
    burst a regime transition causes — i.e. does the throttle engage exactly
    where L-01/R-62 say v4's edge lives?"""
    strat = NovelTurnoverThrottle(upper=TURNOVER_UPPER, eta=ETA)
    strat.prepare(cell.df.copy())
    lam = strat.diag["state_trace"]
    idx = cell.df.index
    lam_s = pd.Series(lam, index=idx)
    rows = []
    for name, date in STRESS_EPISODES:
        t = pd.Timestamp(date, tz=idx.tz)
        win = lam_s.loc[t - pd.Timedelta(days=3): t + pd.Timedelta(days=3)]
        rows.append({
            "episode": name,
            "date": date,
            "lambda_mean_pm3d": round(float(win.mean()), 3) if len(win) else float("nan"),
            "lambda_max_pm3d": round(float(win.max()), 3) if len(win) else float("nan"),
            "frac_throttled_pm3d": round(float((win > 0).mean()), 3) if len(win) else float("nan"),
        })
    out = pd.DataFrame(rows)
    out.attrs["lam_mean_all"] = float(lam_s.mean())
    out.attrs["lam_frac_pos_all"] = float((lam_s > 0).mean())
    return out


def main() -> None:
    cells = E.build_cells()
    lines: list[str] = []
    n_evals = 0

    def w(s: str = "") -> None:
        lines.append(s)
        print(s)

    def dump(t: pd.DataFrame) -> None:
        w("```")
        w(t[[c for c in E.COLS if c in t.columns and not t[c].isna().all()]]
          .to_string(index=False))
        w("```")

    w("# R-132 novel — online dual-ascent turnover throttle on `kelly_regime_v4`")
    w()
    w(f"Frozen: `TURNOVER_UPPER = {TURNOVER_UPPER:.4f}` trades/day (30-day causal "
      f"EWM), `ETA = {ETA}`, `LAMBDA_MAX = {LAMBDA_MAX}`. Inner-validation = "
      f"{INNER_VAL_START} → {INNER_VAL_END}.")
    w()

    # ---------------- A2 ----------------
    w("## A2 — non-inertness gate (run before any performance number is read)")
    w()
    probe = NovelTurnoverThrottle(upper=TURNOVER_UPPER, eta=ETA)
    probe.prepare(cells["btc_spot"].df.copy())
    a2 = a2_non_inertness(probe.diag["n_intervened"])
    w(f"- pending rebalances shrunk by the throttle: **{probe.diag['n_intervened']}** "
      f"of {probe.diag['n_pending']} pending-bars")
    w(f"- `lambda` mean {probe.diag['lam_mean']:.3f}, positive on "
      f"{probe.diag['lam_frac_positive']:.1%} of bars, max seen "
      f"{probe.diag['lam_max_seen']:.2f} (cap {LAMBDA_MAX})")
    w(f"- **A2: {'PASS' if a2['pass'] else 'FAIL'}**")
    w()

    # ---------------- B3 plateau: corridor and eta ----------------
    rows = [E.compare(factory(mult=m), cells["btc_spot"], tag=f"corridor={m}x eta={ETA}")
            for m in MULTS]
    n_evals += len(rows)
    t_corr = E.show(rows, "B3 plateau — corridor multiple (BTC spot, inner-val)")
    w("## B3 — plateau over the corridor multiple (BTC spot, inner-validation)")
    w()
    dump(t_corr)
    w()

    rows_eta = [E.compare(factory(eta=e), cells["btc_spot"],
                          tag=f"corridor={PRIMARY_MULT}x eta={e}")
                for e in ETAS if e != ETA]
    n_evals += len(rows_eta)
    t_eta = E.show(rows_eta, "B3 plateau — dual-ascent step size (BTC spot, inner-val)")
    w("## B3 — plateau over the dual-ascent step size `eta` (BTC spot, inner-validation)")
    w()
    dump(t_eta)
    w()

    b3_pass = bool((t_corr["d_sharpe"] > 0).sum() > len(t_corr) / 2)
    w(f"**B3 (plateau majority positive over the corridor grid): "
      f"{'PASS' if b3_pass else 'FAIL'}** — "
      f"{int((t_corr['d_sharpe'] > 0).sum())} of {len(t_corr)} grid points beat v4.")
    w()

    # ---------------- B1 / B4 / B5 ----------------
    prim = []
    for key in ("btc_spot", "btc_futures", "eth_spot",
                "btc_spot_040", "btc_futures_040", "eth_spot_040"):
        prim.append(E.compare(factory(), cells[key],
                              tag=f"corridor={PRIMARY_MULT}x eta={ETA}"))
    prim.append(E.compare(factory(), cells["btc_spot"], slice_name="inner-train",
                          tag=f"corridor={PRIMARY_MULT}x eta={ETA}"))
    n_evals += len(prim)
    tp = E.show(prim, "B1 / B4 / B5 at the frozen primary")
    w("## B1 (signal, both markets) / B4 (ETH replication) / B5 (0.40% taker)")
    w()
    dump(tp)
    w()

    def row(cell):
        return tp[(tp["cell"] == cell) & (tp["slice"] == "inner-val")].iloc[0]

    b1 = bool(row("btc_spot")["d_sharpe"] > 0 and row("btc_futures")["d_sharpe"] > 0)
    b4 = bool(np.sign(row("eth_spot")["d_sharpe"]) == np.sign(row("btc_spot")["d_sharpe"])
              and row("btc_spot")["d_sharpe"] > 0)
    b5 = bool(row("btc_spot_040")["d_sharpe"] > 0 and row("btc_futures_040")["d_sharpe"] > 0)
    w(f"- **B1**: {'PASS' if b1 else 'FAIL'}")
    w(f"- **B4** (pre-registered falsification test): {'PASS' if b4 else 'FAIL'} "
      f"(BTC {row('btc_spot')['d_sharpe']:+.3f}, ETH {row('eth_spot')['d_sharpe']:+.3f})")
    w(f"- **B5**: {'PASS' if b5 else 'FAIL'}")
    w(f"- **B2** (diagnostic only): BTC spot {row('btc_spot')['d_dd']:+.2f}pp, "
      f"BTC futures {row('btc_futures')['d_dd']:+.2f}pp, "
      f"ETH spot {row('eth_spot')['d_dd']:+.2f}pp")
    w()

    # ---------------- named diagnostic ----------------
    w("## Branch diagnostic — `lambda`'s trajectory through the six stress episodes")
    w()
    le = lambda_at_episodes(cells["btc_spot"])
    w(f"Baseline over the whole BTC frame: `lambda` mean "
      f"{le.attrs['lam_mean_all']:.3f}, positive on "
      f"{le.attrs['lam_frac_pos_all']:.1%} of bars.")
    w()
    w("```")
    w(le.to_string(index=False))
    w("```")
    w()

    verdict = "PROMOTE-candidate" if (a2["pass"] and b1 and b3_pass and b4 and b5) else "NEGATIVE"
    w(f"## Branch verdict: **{verdict}**")
    w()
    w(f"A2={a2['pass']}, B1={b1}, B3={b3_pass}, B4={b4}, B5={b5}.")
    w()
    w(f"Configurations evaluated on this branch: **{n_evals}** candidate backtests "
      f"({len(MULTS) + len(ETAS) - 1} distinct configurations).")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
