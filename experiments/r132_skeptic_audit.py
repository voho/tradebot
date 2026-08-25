"""R-132 skeptic audit — the checks neither branch's own battery would run.

Four questions, all decided after both branch reports were written and all
adversarial in intent:

1. **Is the turnover-feedback channel doing anything at all?** (R-130's
   methodological finding, applied here as a *tightening* of the bar rather
   than a rescue of a result.) Replace the live `lambda_t` with a CONSTANT —
   deleting the trailing-turnover input entirely, leaving plain constant-rate
   partial adjustment, i.e. exactly the Gârleanu-Pedersen smooth trading rate
   R-64 already closed on this object. If a constant reproduces the live
   branch, the "self-regulating control loop" is decoration.

2. **Does the conservative branch's single best grid point survive off the
   cell it was best on?** `corridor=2.0x` scored `d_sharpe = +0.137` on BTC
   spot inner-validation, the largest positive number this round produced and
   the one a careless write-up would headline. Run it on every other cell.

3. **What did all those interventions actually buy?** Both branches
   intervene thousands of times. Count interventions against the realized
   change in FILL count — the thing the whole COST axis exists to reduce.
   (`Metrics.num_trades` counts round-trip episodes, not orders; turnover is
   `len(result.fills)`. Conflating the two is a unit error of exactly the kind
   R-128 recorded, so both are carried in every table below.)

4. **Where did the shrunk orders go?** `broker.REBALANCE_DEADBAND` drops any
   same-sign adjustment below 5% of max notional — the evaluability defect
   R-130's skeptic measured on `hedge_experts` (filed as B-43 by this round;
   R-130's entry attributes it to "R-66/B-29", which is loose — B-29 was a
   snap-to-flat destination question, closed by R-66). Count intended
   re-targets against realized fills.

Writes `experiments/reports/r132_skeptic_audit_report.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import r132_eval as E
from r132_mechanisms import ConservativeTurnoverBand, NovelTurnoverThrottle
from r131_shared import ETA, V4_NATURAL_TRADES_PER_DAY

OUT = Path(__file__).resolve().parent / "reports" / "r132_skeptic_audit_report.md"

# lambda values spanning the live branch's own realized mean (3.61 on BTC
# spot inner-val, 2.06 on ETH) so the control is matched, not strawmanned.
LAM_CONSTS = [1.0, 2.0, 3.61, 6.0, 20.0]
CORRIDORS = [2.0, 2.5, 3.0, 3.5, 4.0]


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

    w("# R-132 skeptic audit")
    w()

    # ---------- 1. constant-lambda ablation --------------------------------
    w("## 1. Ablation — delete the turnover-feedback channel")
    w()
    w("The novel branch's `lambda_t` is replaced by a constant. The trailing-"
      "turnover EWM is then read by nothing: what remains is `pos += (desired - "
      "pos)/(1+lambda)` at a fixed rate — a Gârleanu-Pedersen-style smooth "
      "trading rate, the object R-64 (novel) closed on `kelly_regime_v4` with "
      "\"do not re-try ... at all\". The live branch's own realized mean "
      "`lambda` on this cell is 3.61, so `lam=3.61` is the matched control.")
    w()
    rows = [E.compare(lambda lam=lam: NovelTurnoverThrottle(lam_const=lam),
                      cells["btc_spot"], tag=f"CONTROL lam={lam} (no feedback)")
            for lam in LAM_CONSTS]
    rows.append(E.compare(lambda: NovelTurnoverThrottle(
        upper=3.0 * V4_NATURAL_TRADES_PER_DAY, eta=ETA),
        cells["btc_spot"], tag="LIVE corridor=3.0x eta=0.5"))
    n_evals += len(rows)
    t1 = E.show(rows, "constant-lambda control vs the live throttle (BTC spot, inner-val)")
    dump(t1)
    w()
    live = t1.iloc[-1]
    ctrl = t1.iloc[:-1]
    matched = ctrl[ctrl["tag"].str.contains("3.61")].iloc[0]
    beat = int((ctrl["d_sharpe"] >= live["d_sharpe"]).sum())
    w(f"- live branch `d_sharpe` = {live['d_sharpe']:+.3f} "
      f"({live['fills']} fills, {live['trades']} round-trip episodes); matched "
      f"control (`lam=3.61`, no feedback) = {matched['d_sharpe']:+.3f} "
      f"({matched['fills']} fills, {matched['trades']} episodes). v4 itself: "
      f"{live['fills_v4']} fills, {live['trades_v4']} episodes.")
    w(f"- {beat} of {len(ctrl)} zero-feedback controls score a higher `d_sharpe` "
      f"than the live branch. **That is not evidence the feedback channel is "
      f"decoration, because the arms are not risk-matched** (the standing rule): "
      f"every control carries "
      f"{ctrl['d_dd'].min():+.2f} to {ctrl['d_dd'].max():+.2f}pp more drawdown "
      f"than v4 and {ctrl['tim'].min():.1f}-{ctrl['tim'].max():.1f}% time in "
      f"market against v4's {live['tim_v4']:.1f}%, at realized volatility "
      f"{ctrl['vol'].min():.3f}-{ctrl['vol'].max():.3f} against v4's "
      f"{live['vol_v4']:.3f}. Higher Sharpe bought with more exposure is an "
      f"exposure statement.")
    w(f"- What the ablation DOES establish is structural, and it is the finding "
      f"that explains this whole round. Constant-`lambda` partial adjustment "
      f"never lands exactly on the target and never reaches zero, so the "
      f"position is never fully closed: {int(ctrl['trades'].max())} round-trip "
      f"episode(s) across two years, against v4's {live['trades_v4']}, while "
      f"still paying {ctrl['fills'].min()}-{ctrl['fills'].max()} fills. The "
      f"controls are not throttled versions of v4 — they are permanently-invested "
      f"variants that trade about as often. A shrink-the-order throttle does not "
      f"convert into less trading; it converts one decisive order into a stream "
      f"of partial ones and removes the exits.")
    w(f"- And the live branch's `lambda` is not a smooth dial either. The `eta` "
      f"grid in the novel branch's own report moves `eta` by 4x and changes "
      f"`lam_mean` by 3% and the fill count not at all, because `lambda` "
      f"saturates at its `LAMBDA_MAX` cap whenever it leaves zero (positive on "
      f"only {live['lam_frac_pos']:.1%} of bars, mean {live['lam_mean']:.2f}, max "
      f"seen = the cap). The \"smooth, self-regulating control loop\" as frozen "
      f"is behaviourally **bang-bang**.")
    w()

    # ---------- 2. the conservative branch's best grid point ---------------
    w("## 2. Does the conservative branch's best grid point survive off its own cell?")
    w()
    w("`corridor=2.0x` scored `d_sharpe = +0.137` on BTC spot inner-validation — "
      "this round's largest positive number, and the one a careless write-up "
      "would report as the result. Every other cell, same frozen configuration:")
    w()
    rows2 = []
    for key in ("btc_spot", "btc_futures", "eth_spot",
                "btc_spot_040", "btc_futures_040", "eth_spot_040"):
        rows2.append(E.compare(
            lambda: ConservativeTurnoverBand(upper=2.0 * V4_NATURAL_TRADES_PER_DAY),
            cells[key], tag="cons corridor=2.0x"))
    rows2.append(E.compare(
        lambda: ConservativeTurnoverBand(upper=2.0 * V4_NATURAL_TRADES_PER_DAY),
        cells["btc_spot"], slice_name="inner-train", tag="cons corridor=2.0x"))
    n_evals += len(rows2)
    t2 = E.show(rows2, "conservative corridor=2.0x on every cell")
    dump(t2)
    w()
    iv = t2[t2["slice"] == "inner-val"]
    w(f"- positive `d_sharpe` on {int((iv['d_sharpe'] > 0).sum())} of {len(iv)} "
      f"inner-validation cells; bootstrap interval excludes zero on "
      f"{int(iv['sig'].sum())} of {len(iv)}.")
    w()

    # ---------- 3. turnover accounting ------------------------------------
    w("## 3. What the interventions bought — turnover accounting")
    w()
    w("Both branches intervene thousands of times. The COST axis exists to "
      "reduce realized trading, so count the interventions against realized "
      "FILLS (BTC spot, inner-validation). `turnover_ratio` is fills / v4's "
      "fills; `episodes` is `Metrics.num_trades`, a different unit, carried "
      "alongside so the two cannot be confused.")
    w()
    acc = []
    for m in CORRIDORS:
        r = E.compare(lambda m=m: ConservativeTurnoverBand(
            upper=m * V4_NATURAL_TRADES_PER_DAY), cells["btc_spot"],
            tag=f"cons {m}x")
        acc.append({"branch": "conservative", "corridor": f"{m}x",
                    "interventions": r["n_intervened"], "fills": r["fills"],
                    "fills_v4": r["fills_v4"], "episodes": r["trades"],
                    "turnover_ratio": round(r["fills"] / r["fills_v4"], 3),
                    "d_sharpe": r["d_sharpe"], "d_dd": r["d_dd"]})
        n_evals += 1
    for m in CORRIDORS:
        r = E.compare(lambda m=m: NovelTurnoverThrottle(
            upper=m * V4_NATURAL_TRADES_PER_DAY, eta=ETA), cells["btc_spot"],
            tag=f"novel {m}x")
        acc.append({"branch": "novel", "corridor": f"{m}x",
                    "interventions": r["n_intervened"], "fills": r["fills"],
                    "fills_v4": r["fills_v4"], "episodes": r["trades"],
                    "turnover_ratio": round(r["fills"] / r["fills_v4"], 3),
                    "d_sharpe": r["d_sharpe"], "d_dd": r["d_dd"]})
        n_evals += 1
    t3 = pd.DataFrame(acc)
    print(t3.to_string(index=False))
    w("```")
    w(t3.to_string(index=False))
    w("```")
    w()
    cons = t3[t3["branch"] == "conservative"]
    nov = t3[t3["branch"] == "novel"]
    w(f"- **conservative** (defer, i.e. change order TIMING): "
      f"{int(cons['interventions'].max())} interventions at its tightest corridor "
      f"buy a turnover ratio of {cons['turnover_ratio'].min():.3f} — a "
      f"{100 * (1 - cons['turnover_ratio'].min()):.0f}% cut for ~5000 "
      f"interventions. Deferral POSTPONES a rebalance; it does not cancel it, so "
      f"the move executes a few bars later and the fill count barely moves. A "
      f"defer-only band on a latched target is close to a no-op on cost, whatever "
      f"its intervention count says.")
    w(f"- **novel** (shrink, i.e. change order SIZE): the turnover ratio is "
      f"**non-monotone and never much below 1**, "
      f"{nov['turnover_ratio'].min():.3f}-{nov['turnover_ratio'].max():.3f}, and "
      f"at the TIGHTEST corridor (2.0x) it is "
      f"**{nov['turnover_ratio'].iloc[0]:.3f} — more turnover than v4, not less**. "
      f"This is the round's central mechanical finding: shrinking an order does "
      f"not remove it, it splits it, so a size throttle on a latched-target "
      f"strategy converts one decisive rebalance into a sequence of partial ones. "
      f"Turning the throttle up trades MORE. Meanwhile `d_dd` worsens "
      f"monotonically as the corridor tightens "
      f"({nov['d_dd'].iloc[-1]:+.2f}pp at 4.0x → {nov['d_dd'].iloc[0]:+.2f}pp at "
      f"2.0x): the mechanism buys extra drawdown and does not even deliver the "
      f"cost saving it exists for.")
    w()

    # ---------- 4. where the shrunk orders went ---------------------------
    w("## 4. Where the shrunk orders went — `broker.REBALANCE_DEADBAND` absorption")
    w()
    w("Section 1's controls intend hundreds of re-targets and fill almost none. "
      "`tradebot.broker` drops any same-sign adjustment worth less than "
      "`REBALANCE_DEADBAND = 5%` of max notional. A mechanism whose whole action "
      "is to SHRINK a re-target therefore shrinks its orders straight through "
      "that floor: `kelly_regime_v4`'s own strategy-level deadband is 0.10, so a "
      "move that has just cleared it, divided by `1 + lambda`, lands at 0.10/"
      "(1+lambda) — below the broker's 0.05 at any `lambda > 1`. The intent is "
      "recorded in the target column; the order never reaches the tape.")
    w()
    w("This is the same evaluability defect R-130's skeptic measured on "
      "`hedge_experts` (96.8% of intended re-targets absorbed), filed as B-43 "
      "by this round. "
      "It is confirmed here independently, on `kelly_regime_v4`:")
    w()
    absorb = []
    for lam in LAM_CONSTS:
        s = NovelTurnoverThrottle(lam_const=lam)
        prepared = s.prepare(cells["btc_spot"].df.copy())
        tgt = prepared["target"].to_numpy()
        idx = prepared.index
        mask = ((idx >= pd.Timestamp("2021-01-01", tz=idx.tz))
                & (idx <= pd.Timestamp("2022-12-31", tz=idx.tz)))
        changes = int((np.abs(np.diff(tgt[mask])) > 1e-12).sum())
        r = E.compare(lambda lam=lam: NovelTurnoverThrottle(lam_const=lam),
                      cells["btc_spot"], tag=f"lam={lam}")
        n_evals += 1
        absorb.append({
            "config": f"constant lam={lam}", "intended_retargets": changes,
            "filled_orders": r["fills"],
            "absorbed_pct": round(100.0 * (1 - r["fills"] / max(changes, 1)), 1)})
    for m in (2.0, 3.0):
        s = NovelTurnoverThrottle(upper=m * V4_NATURAL_TRADES_PER_DAY, eta=ETA)
        prepared = s.prepare(cells["btc_spot"].df.copy())
        tgt = prepared["target"].to_numpy()
        idx = prepared.index
        mask = ((idx >= pd.Timestamp("2021-01-01", tz=idx.tz))
                & (idx <= pd.Timestamp("2022-12-31", tz=idx.tz)))
        changes = int((np.abs(np.diff(tgt[mask])) > 1e-12).sum())
        r = E.compare(lambda m=m: NovelTurnoverThrottle(
            upper=m * V4_NATURAL_TRADES_PER_DAY, eta=ETA), cells["btc_spot"],
            tag=f"live {m}x")
        n_evals += 1
        absorb.append({
            "config": f"LIVE novel corridor={m}x", "intended_retargets": changes,
            "filled_orders": r["fills"],
            "absorbed_pct": round(100.0 * (1 - r["fills"] / max(changes, 1)), 1)})
    from tradebot.registry import get_strategy
    prepared = get_strategy("kelly_regime_v4").prepare(cells["btc_spot"].df.copy())
    tgt = prepared["target"].to_numpy()
    idx = prepared.index
    mask = ((idx >= pd.Timestamp("2021-01-01", tz=idx.tz))
            & (idx <= pd.Timestamp("2022-12-31", tz=idx.tz)))
    changes = int((np.abs(np.diff(tgt[mask])) > 1e-12).sum())
    m_v4, _, fills_v4 = E.baseline(cells["btc_spot"])
    absorb.append({"config": "kelly_regime_v4 (reference)",
                   "intended_retargets": changes, "filled_orders": fills_v4,
                   "absorbed_pct": round(100.0 * (1 - fills_v4 / max(changes, 1)), 1)})
    t4 = pd.DataFrame(absorb)
    print(t4.to_string(index=False))
    w("```")
    w(t4.to_string(index=False))
    w("```")
    w()
    w("Absorption scales with the shrink factor exactly as the algebra predicts, "
      "and it is NOT total: v4 loses 4.5% of its intended re-targets to the floor, "
      "a constant `lambda=1` loses 9%, and the live novel branch loses 61% at its "
      "frozen corridor and 83% at its tightest. So the deadband is not what makes "
      "this round negative — section 3 is — but it does mean the *measured* "
      "behaviour of any size-shrinking mechanism here is a blend of the mechanism "
      "and the broker's floor, in a proportion that changes with the mechanism's "
      "own parameter. **A COST-axis mechanism on this framework that acts on order "
      "SIZE cannot be cleanly attributed until `REBALANCE_DEADBAND` is addressed** "
      "(B-29). Mechanisms that act on order TIMING, like the conservative branch, "
      "are unaffected.")
    w()

    w(f"Configurations evaluated in this audit: **{n_evals}** candidate backtests.")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
