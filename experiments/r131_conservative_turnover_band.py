"""R-131 CONSERVATIVE branch — band turnover regularization on `kelly_regime_v4`.

Mechanism, as frozen in `r131_shared.py` before any code was written: track a
causal trailing-turnover EWM of v4's own realized rebalances; inside the
corridor `[0, TURNOVER_UPPER]` trade exactly as v4 does; at or above the upper
edge, DEFER the pending rebalance, unless it is a full de-risking exit
(`desired == 0`) or its magnitude exceeds `OVERRIDE_MULT * TURNOVER_UPPER`.

Citation: Khubiev, Semenov, Podlipnova & Khubieva (2025, arXiv:2509.04541),
"Finance-Grounded Optimization For Algorithmic Trading" — band turnover
regularization, zero inside an admissible range, biting only above it.

Writes `experiments/reports/r131_conservative_report.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import r131_eval as E
from r131_mechanisms import ConservativeTurnoverBand
from r131_shared import (
    INNER_VAL_END,
    INNER_VAL_START,
    STRESS_EPISODES,
    TURNOVER_UPPER,
    V4_NATURAL_TRADES_PER_DAY,
    a2_non_inertness,
)

OUT = Path(__file__).resolve().parent / "reports" / "r131_conservative_report.md"

# Plateau grid: the corridor's upper edge as a multiple of v4's own measured
# natural rate. 3.0 is the frozen primary; the grid is symmetric around it.
MULTS = [2.0, 2.5, 3.0, 3.5, 4.0]
PRIMARY_MULT = 3.0


def factory(mult: float):
    return lambda: ConservativeTurnoverBand(upper=mult * V4_NATURAL_TRADES_PER_DAY)


def stress_episode_diagnostic(cell: E.Cell) -> pd.DataFrame:
    """Did the mechanism defer a rebalance within 3 days of a listed episode?

    The branch's own named diagnostic (reported regardless of B1-B5, never
    gates promotion): v4's edge concentrates around a handful of sudden regime
    transitions, and a throttle that damps trading exactly when turnover
    spikes could be damping it exactly there.
    """
    strat = ConservativeTurnoverBand(upper=PRIMARY_MULT * V4_NATURAL_TRADES_PER_DAY)
    strat.prepare(cell.df.copy())
    idx = cell.df.index
    defer_ts = idx[np.array(strat.defer_bars, dtype=int)] if strat.defer_bars else idx[:0]
    rows = []
    for name, date in STRESS_EPISODES:
        t = pd.Timestamp(date, tz=idx.tz)
        lo, hi = t - pd.Timedelta(days=3), t + pd.Timedelta(days=3)
        n = int(((defer_ts >= lo) & (defer_ts <= hi)).sum())
        rows.append({"episode": name, "date": date, "deferral_bars_within_3d": n})
    return pd.DataFrame(rows)


def main() -> None:
    cells = E.build_cells()
    lines: list[str] = []
    n_evals = 0

    def w(s: str = "") -> None:
        lines.append(s)
        print(s)

    w("# R-131 conservative — turnover-corridor deferral band on `kelly_regime_v4`")
    w()
    w(f"Frozen corridor edge: `TURNOVER_UPPER = {PRIMARY_MULT} x "
      f"{V4_NATURAL_TRADES_PER_DAY:.4f} = {TURNOVER_UPPER:.4f}` trades/day, "
      f"30-day causal EWM. Inner-validation = {INNER_VAL_START} → {INNER_VAL_END}.")
    w()

    # ---------------- A2: does the mechanism ever bind? ----------------
    w("## A2 — non-inertness gate (run before any performance number is read)")
    w()
    probe = ConservativeTurnoverBand(upper=TURNOVER_UPPER)
    probe.prepare(cells["btc_spot"].df.copy())
    a2 = a2_non_inertness(probe.n_defer)
    w(f"- deferrals fired (BTC, full train+inner-val frame): **{probe.n_defer}**")
    w(f"- overrides — full de-risking exit: {probe.n_override_exit}; "
      f"move > {probe.override_mult} x corridor: {probe.n_override_size}")
    w(f"- **A2: {'PASS' if a2['pass'] else 'FAIL'}** — the corridor is reached and "
      f"the mechanism does change v4's behaviour.")
    w()
    w("Note on the size override as frozen: `|desired - current| > OVERRIDE_MULT * "
      "TURNOVER_UPPER` compares a position magnitude (units of equity notional, "
      "range 0-2 here) against a *rate* (trades/day). The frozen text is "
      "implemented literally; the resulting threshold is "
      f"{probe.override_mult * TURNOVER_UPPER:.3f} in position units, so it fires "
      f"{probe.n_override_size} times and the effective safety valve is the "
      "`desired == 0` full-exit clause. Recorded, not silently repaired.")
    w()

    # ---------------- B3: plateau on the primary cell ----------------
    rows = []
    for m in MULTS:
        rows.append(E.compare(factory(m), cells["btc_spot"], tag=f"corridor={m}x"))
        n_evals += 1
    t = E.show(rows, "B3 plateau — corridor multiple, BTC spot, inner-validation")
    w("## B3 — plateau over the corridor multiple (BTC spot, inner-validation)")
    w()
    w("```")
    w(t[[c for c in E.COLS if c in t.columns and not t[c].isna().all()]].to_string(index=False))
    w("```")
    b3_pass = bool((t["d_sharpe"] > 0).sum() > len(t) / 2)
    w(f"\n**B3 (plateau majority positive): {'PASS' if b3_pass else 'FAIL'}** — "
      f"{int((t['d_sharpe'] > 0).sum())} of {len(t)} grid points beat v4 on Sharpe.")
    w()

    # ---------------- B1 / B4 / B5 at the frozen primary ----------------
    prim = []
    for key in ("btc_spot", "btc_futures", "eth_spot",
                "btc_spot_040", "btc_futures_040", "eth_spot_040"):
        prim.append(E.compare(factory(PRIMARY_MULT), cells[key],
                              tag=f"corridor={PRIMARY_MULT}x"))
        n_evals += 1
    prim.append(E.compare(factory(PRIMARY_MULT), cells["btc_spot"],
                          slice_name="inner-train", tag=f"corridor={PRIMARY_MULT}x"))
    n_evals += 1
    tp = E.show(prim, "B1 / B4 / B5 at the frozen primary corridor")
    w("## B1 (signal, both markets) / B4 (ETH replication) / B5 (0.40% taker)")
    w()
    w("```")
    w(tp[[c for c in E.COLS if c in tp.columns and not tp[c].isna().all()]].to_string(index=False))
    w("```")
    w()

    def row(cell):
        return tp[(tp["cell"] == cell) & (tp["slice"] == "inner-val")].iloc[0]

    b1 = bool(row("btc_spot")["d_sharpe"] > 0 and row("btc_futures")["d_sharpe"] > 0)
    b4_sign = np.sign(row("eth_spot")["d_sharpe"]) == np.sign(row("btc_spot")["d_sharpe"])
    b4 = bool(b4_sign and row("btc_spot")["d_sharpe"] > 0)
    b5 = bool(row("btc_spot_040")["d_sharpe"] > 0 and row("btc_futures_040")["d_sharpe"] > 0)
    w(f"- **B1** (beats v4 on both BTC markets, inner-val): "
      f"{'PASS' if b1 else 'FAIL'}")
    w(f"- **B4** (pre-registered falsification — sign of `d_sharpe` replicates on ETH, "
      f"and is positive): {'PASS' if b4 else 'FAIL'} "
      f"(BTC {row('btc_spot')['d_sharpe']:+.3f}, ETH {row('eth_spot')['d_sharpe']:+.3f})")
    w(f"- **B5** (survives a 0.40% taker tier on both BTC markets): "
      f"{'PASS' if b5 else 'FAIL'}")
    w(f"- **B2** (drawdown, diagnostic only, never gates): "
      f"BTC spot {row('btc_spot')['d_dd']:+.2f}pp, "
      f"BTC futures {row('btc_futures')['d_dd']:+.2f}pp, "
      f"ETH spot {row('eth_spot')['d_dd']:+.2f}pp")
    w()

    # ---------------- named diagnostic ----------------
    w("## Branch diagnostic — deferral behaviour at the six stress episodes")
    w()
    w("The failure mode named before any code ran: a throttle that damps trading "
      "when turnover spikes is damping it exactly when a regime transition — where "
      "L-01/R-62 say v4's edge lives — drives a burst of rebalances.")
    w()
    se = stress_episode_diagnostic(cells["btc_spot"])
    w("```")
    w(se.to_string(index=False))
    w("```")
    w()

    verdict = "PROMOTE-candidate" if (a2["pass"] and b1 and b3_pass and b4 and b5) else "NEGATIVE"
    w(f"## Branch verdict: **{verdict}**")
    w()
    w(f"Decision rule as frozen: A2 AND B1 (both markets) AND B3 (plateau majority) "
      f"AND B4 (full, both markets) AND B5 — all must pass. "
      f"A2={a2['pass']}, B1={b1}, B3={b3_pass}, B4={b4}, B5={b5}.")
    w()
    w(f"Configurations evaluated on this branch: **{n_evals}** candidate backtests "
      f"({len(MULTS)} distinct configurations).")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
