"""Freeze and report R-190's predeclared rules without changing tested signals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "experiments")]

import numpy as np
import pandas as pd

from r190_eval import OUT, FILES, CANDIDATES, AUXILIARIES, PARENTS, validate_manifest
from r190_variations import CONFIGS
from tradebot.inference import (annualized_sharpe, bootstrap_interval,
    deflated_sharpe_ratio, max_drawdown_from_returns, moments, paired_bootstrap,
    stationary_bootstrap_indices, total_log_return)

PRIMARY = ("inner_val", "funded_val", "holdout", "funded_holdout")
REFERENCE = {name: parent if parent != "blend" else "kelly_regime_v4"
             for name, parent, _ in CONFIGS}
PRIOR = 1503


def read(stage, matched=False):
    tag = f"{stage}_matched" if matched else stage
    cells = pd.read_csv(OUT / f"{tag}_cells.csv")
    daily = pd.read_csv(OUT / f"{tag}_daily.csv.gz")
    series = {(name, cell): group.set_index("timestamp")["return"]
              for (name, cell), group in daily.groupby(["strategy", "cell"], sort=False)}
    return cells, series


def power():
    """Training-only noise check; the +0.20 floor is inherited from R-20."""
    cells, daily = read("train")
    rows = []
    for cell in ("inner_val", "funded_val"):
        n = len(daily[CANDIDATES[0], cell])
        idx = stationary_bootstrap_indices(n, 30., 2000, np.random.default_rng(190))
        for name in CANDIDATES:
            a, b = daily[name, cell], daily[REFERENCE[name], cell]
            assert a.index.equals(b.index)
            diff = np.log1p(a.to_numpy()) - np.log1p(b.to_numpy())
            se = float(diff[idx].mean(axis=1).std(ddof=1))
            hurdle = .20 * a.std(ddof=1) / np.sqrt(365.25)
            required = n * (2.8 * se / hurdle) ** 2 if hurdle > 0 else float("inf")
            rows.append(dict(strategy=name, cell=cell, days=n, daily_logdiff_sd=diff.std(ddof=1),
                block_mean_se=se, daily_mean_hurdle=hurdle,
                required_days_approx=required, required_years_approx=required / 365.25))
    pd.DataFrame(rows).to_csv(OUT / "training_power.csv", index=False)
    return cells


def freeze():
    cells = power()
    path = OUT / "manifest.json"
    if path.exists():
        validate_manifest()
        return
    own = cells[(cells.cell == "inner_val") & cells.strategy.isin(CANDIDATES + AUXILIARIES)]
    sd = max(.418538, float(own.daily_sharpe.std(ddof=1)))
    ranking = cells[cells.strategy.isin(CANDIDATES) & cells.cell.isin(PRIMARY)]
    ranked = ranking.groupby("strategy").daily_sharpe.mean().rename("mean_validation_sharpe").to_frame()
    ranked["spot_fills"] = cells[cells.cell == "inner_val"].set_index("strategy").fills
    ranked = ranked.sort_values(["mean_validation_sharpe", "spot_fills"], ascending=[False, True])
    paths = ["experiments/r190_protocol.md", "docs/R190_RESEARCH.md",
        "experiments/r190_variations.py", "experiments/r190_eval.py",
        "experiments/r190_matched.py", "experiments/matched_hold.py",
        "src/tradebot/strategies/kelly_regime.py", "src/tradebot/strategies/kelly_regime_v3.py",
        "src/tradebot/strategies/kelly_regime_v4.py", "src/tradebot/strategies/buy_and_hold.py",
        "src/tradebot/engine.py", "src/tradebot/broker.py", "src/tradebot/data.py",
        "src/tradebot/inference.py", "src/tradebot/metrics.py", "src/tradebot/strategy.py"]
    paths += ["data/" + f for f in FILES.values()]
    paths += ["data/btcusdt_deribit_perp_funding_8h.csv.gz"]
    manifest = dict(frozen_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        protocol="experiments/r190_protocol.md", primary_candidates=list(CANDIDATES),
        auxiliary_neighbours=list(AUXILIARIES), prior_consultations_approx=PRIOR,
        core_evaluations=784, core_holdout_evaluations=736, local_configurations=12,
        sd_trials=sd, validation_lead=str(ranked.index[0]),
        validation_ranking=ranked.reset_index().to_dict("records"),
        hashes={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths})
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen; inner-validation lead {ranked.index[0]}; trial Sharpe SD {sd:.6f}")


def inference():
    rows = []
    for stage, cell_names in (("train", PRIMARY[:2]), ("holdout", PRIMARY[2:])):
        cells, daily = read(stage)
        matched, control_daily = read(stage, matched=True)
        for cell in cell_names:
            n = len(daily[CANDIDATES[0], cell])
            idx = stationary_bootstrap_indices(n, 30., 2000, np.random.default_rng(190))
            for name in CANDIDATES:
                a = daily[name, cell]
                final = matched[(matched.reference_candidate == name) & (matched.cell == cell)
                                & matched.final_selected].iloc[0]
                refs = [("parent", REFERENCE[name], daily[REFERENCE[name], cell], .05),
                        ("matched_hold", final.strategy, control_daily[final.strategy, cell], .02)]
                for kind, ref, b, tolerance in refs:
                    assert a.index.equals(b.index), (name, cell, ref)
                    ratio = b.std(ddof=1) / a.std(ddof=1) if a.std(ddof=1) > 0 else np.nan
                    row = dict(strategy=name, cell=cell, control=kind, reference=ref,
                        risk_valid=bool(np.isfinite(ratio) and abs(ratio - 1) <= tolerance),
                        control_to_candidate_vol=ratio, days=n)
                    for label, stat in (("sharpe", annualized_sharpe), ("growth", total_log_return),
                                        ("drawdown", max_drawdown_from_returns)):
                        p = paired_bootstrap(a.to_numpy(), b.to_numpy(), stat, indices=idx)
                        row.update({label: p.stat_a, "control_" + label: p.stat_b,
                            "d_" + label: p.diff.point, "d_" + label + "_lo": p.diff.lo,
                            "d_" + label + "_hi": p.diff.hi})
                    rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "bootstrap.csv", index=False)
    return result


def decide(boot):
    train, _ = read("train")
    hold, daily = read("holdout")
    mt, _ = read("train", True)
    mh, _ = read("holdout", True)
    cells = pd.concat([train, hold], ignore_index=True)
    cells.to_csv(OUT / "cells.csv", index=False)
    manifest = json.loads((OUT / "manifest.json").read_text())
    audit_path = OUT / "audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    extra_train = int(audit.get("train_evaluations", 0))
    extra_holdout = int(audit.get("holdout_evaluations", 0))
    count = len(hold) + len(mh) + extra_holdout
    program_n = PRIOR + count
    indexed = cells.set_index(["strategy", "cell"])
    rows = []
    for name in CANDIDATES:
        own = cells[cells.strategy == name]
        comparisons = boot[boot.strategy == name]
        primary_hold = comparisons[comparisons.cell.isin(PRIMARY[2:])]
        dsrs = []
        local_dsrs = []
        for cell in PRIMARY[2:]:
            r = daily[name, cell].to_numpy()
            sk, ku = moments(r)
            args = (annualized_sharpe(r), len(r), sk, ku)
            dsrs.append(deflated_sharpe_ratio(*args, n_trials=program_n, sd_trials=manifest["sd_trials"]))
            local_dsrs.append(deflated_sharpe_ratio(*args, n_trials=12, sd_trials=manifest["sd_trials"]))
        beta = {}
        for market in ("spot", "perp"):
            valid, wins, raw_wins = 0, 0, 0
            for k in range(24):
                cell = f"beta_{market}_{k:02d}"
                a = indexed.loc[name, cell]
                b = indexed.loc[REFERENCE[name], cell]
                ratio = b.annualized_volatility / a.annualized_volatility if a.annualized_volatility > 0 else np.nan
                match = bool(np.isfinite(ratio) and abs(ratio - 1) <= .05)
                won = a.final_balance > b.final_balance
                valid += match
                wins += match and won
                raw_wins += won
            beta.update({f"beta_{market}_valid": valid, f"beta_{market}_wins": wins,
                         f"beta_{market}_raw_wins": raw_wins})
        family = [name.rsplit("_b", 1)[0] + f"_b{b}" for b in ("05", "10", "20")]
        plateau = True
        for cell in PRIMARY:
            f = indexed.loc[[(n, cell) for n in family]]
            parent_sr = indexed.loc[REFERENCE[name], cell].daily_sharpe
            plateau &= bool((f.final_balance > 1000).all()
                and f.daily_sharpe.max() - f.daily_sharpe.min() <= .20
                and (f.daily_sharpe > parent_sr).sum() >= 2)
        spot = indexed.loc[name, "holdout"]
        gates = dict(
            point_and_risk=bool(comparisons.risk_valid.all() and (comparisons.d_sharpe > .20).all()),
            intervals=bool((primary_hold.d_sharpe_lo > 0).all() and (primary_hold.d_growth_lo > 0).all()
                and all(indexed.loc[name, c].final_balance > indexed.loc[REFERENCE[name], c].final_balance
                        for c in PRIMARY[2:])),
            falsification=bool((own[own.cell.isin(("holdout", "discount_holdout", "eth_holdout", "funded_holdout"))].final_balance > 1000).all()
                and indexed.loc[name, "eth_holdout"].daily_sharpe >= indexed.loc[REFERENCE[name], "eth_holdout"].daily_sharpe
                and not own.liquidated.any()),
            cadence=bool(2 <= spot.fills_per_day <= 6),
            dsr=bool(min(dsrs) >= .95),
            beta=bool(beta["beta_spot_wins"] >= 13 and beta["beta_perp_wins"] >= 13),
            plateau=plateau,
        )
        rows.append(dict(strategy=name, reference=REFERENCE[name], verdict="PROMOTED" if all(gates.values()) else "NEGATIVE",
            **gates, failed_gates=", ".join(k for k, v in gates.items() if not v),
            dsr_spot=dsrs[0], dsr_perp=dsrs[1], local_dsr_spot=local_dsrs[0], local_dsr_perp=local_dsrs[1],
            program_trials_approx=program_n, **beta))
    decisions = pd.DataFrame(rows)
    decisions.to_csv(OUT / "decision.csv", index=False)
    counts = dict(candidate_configurations=10, auxiliary_configurations=2,
        core_evaluations=len(cells), matching_evaluations=len(mt) + len(mh),
        audit_train_evaluations=extra_train, audit_holdout_evaluations=extra_holdout,
        total_evaluations=len(cells) + len(mt) + len(mh) + extra_train + extra_holdout,
        holdout_consultations=count, cumulative_consultations_approx=program_n)
    (OUT / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    return cells, decisions, counts


def chart(cells, boot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, daily = read("holdout")
    names = list(CANDIDATES) + list(PARENTS)
    colors = plt.cm.tab10(np.arange(10))
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")
    for ax, cell, title in zip(axes[0], PRIMARY[2:], ("BTC spot · 40bp taker + 1bp slippage", "BTC perp · 5bp + 1bp, funding charged")):
        for i, name in enumerate(names):
            r = daily[name, cell]
            ax.plot(pd.to_datetime(r.index), 1000 * (1 + r).cumprod(),
                color=colors[i] if i < 10 else ("black", "gray", "saddlebrown")[i - 10],
                ls="-" if i < 10 else "--", alpha=.8, lw=1.2 if i < 10 else 2, label=name)
        if cell == "holdout":
            r = daily["buy_and_hold", cell]
            benchmark, = ax.plot(pd.to_datetime(r.index), 1000 * (1 + r).cumprod(), color="navy", lw=2, ls=":", label="buy_and_hold")
            ax.legend(handles=[benchmark], fontsize=8, loc="upper left")
        ax.set(title=title, ylabel="Account value ($), log scale", yscale="log")
        ax.grid(alpha=.2)
    axes[0, 1].legend(fontsize=7, ncol=2)
    b = boot[(boot.cell == "holdout") & (boot.control == "parent")].set_index("strategy").loc[list(CANDIDATES)]
    y = np.arange(10)
    # A percentile interval need not enclose its original point estimate.
    axes[1, 0].hlines(y, b.d_sharpe_lo, b.d_sharpe_hi, color="tab:blue")
    axes[1, 0].scatter(b.d_sharpe, y, color="tab:blue")
    for i, valid in enumerate(b.risk_valid):
        if not valid:
            axes[1, 0].plot(b.d_sharpe.iloc[i], i, "rx", ms=10)
    axes[1, 0].axvline(0, color="gray", lw=1)
    axes[1, 0].axvline(.20, color="green", ls="--", lw=1)
    axes[1, 0].set(yticks=y, yticklabels=CANDIDATES, title="Spot Δ daily Sharpe vs parent · paired 95% CI\nred × = volatility match invalid", xlabel="Difference in annualized daily Sharpe")
    s = cells[cells.cell == "holdout"].set_index("strategy").loc[list(CANDIDATES)]
    axes[1, 1].barh(y, s.fills_per_day, color=colors, label="Actual fills/day")
    axes[1, 1].scatter(s.round_trips_per_day, y, marker="x", color="black", label="Completed round trips/day")
    axes[1, 1].axvspan(2, 6, color="green", alpha=.1, label="Requested fill cadence")
    axes[1, 1].set(yticks=y, yticklabels=CANDIDATES, xlim=(0, 6), title="Spot activity · no forced trades", xlabel="Per calendar day")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("R-190 · Ten variations of the three promoted Kelly parents", fontsize=17)
    fig.savefig(OUT / "candidates.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "report"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        validate_manifest()
        boot = inference()
        cells, decisions, counts = decide(boot)
        chart(cells, boot)
        print(decisions[["strategy", "verdict", "failed_gates"]].to_string(index=False))
        print(json.dumps(counts, indent=2))
