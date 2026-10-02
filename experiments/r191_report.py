"""Freeze training-only choices and report R191's exhaustive decision rule."""
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

import r191_eval as evaluation
from r191_strategies import PRIMARY_NAMES, ALL_NAMES, FAMILIES, LABELS
from tradebot.inference import (annualized_sharpe, deflated_sharpe_ratio,
    max_drawdown_from_returns, moments, paired_bootstrap,
    stationary_bootstrap_indices, total_log_return)

OUT = ROOT / "reports/r191_strategies"
PRIOR = 2290
PRIMARY_CELLS = ("inner_val", "funded_val", "holdout", "funded_holdout")


def read(stage, matched=False):
    tag = f"{stage}_matched" if matched else stage
    cells = pd.read_csv(OUT / f"{tag}_cells.csv")
    daily = pd.read_csv(OUT / f"{tag}_daily.csv.gz")
    if cells.duplicated(["strategy", "cell"]).any():
        raise ValueError(f"Duplicate evaluation receipts: {tag}")
    if daily.duplicated(["strategy", "cell", "timestamp"]).any():
        raise ValueError(f"Duplicate daily observations: {tag}")
    series = {(name, cell): group.set_index("timestamp")["return"]
              for (name, cell), group in daily.groupby(["strategy", "cell"], sort=False)}
    return cells, series


def power():
    cells, daily = read("train")
    matches, mdaily = read("train", True)
    rows = []
    for cell in PRIMARY_CELLS[:2]:
        n = len(daily[PRIMARY_NAMES[0], cell])
        idx = stationary_bootstrap_indices(n, 30., 2000, np.random.default_rng(191))
        for name in PRIMARY_NAMES:
            match = matches[(matches.reference_candidate == name) & (matches.cell == cell)
                            & matches.final_selected].iloc[0]
            a, b = daily[name, cell], mdaily[match.strategy, cell]
            assert a.index.equals(b.index)
            diff = np.log1p(a.to_numpy()) - np.log1p(b.to_numpy())
            se = float(diff[idx].mean(axis=1).std(ddof=1))
            hurdle = .20 * a.std(ddof=1) / np.sqrt(365.25)
            required = n * (2.8 * se / hurdle) ** 2 if hurdle > 0 else float("inf")
            rows.append(dict(strategy=name, cell=cell, days=n,
                daily_logdiff_sd=diff.std(ddof=1), block_mean_se=se,
                daily_mean_hurdle=hurdle, required_days_approx=required,
                required_years_approx=required / 365.25, matched_valid=match.matched_valid))
    pd.DataFrame(rows).to_csv(OUT / "training_power.csv", index=False)
    return cells


def freeze():
    path = OUT / "manifest.json"
    if path.exists():
        evaluation.validate_manifest()
        return
    cells = power()
    expected = {(name, spec[0]) for name in evaluation.WORKER_NAMES
                for spec in evaluation.specifications("train")}
    if set(zip(cells.strategy, cells.cell)) != expected:
        raise ValueError("Training cells are incomplete; cannot freeze")
    own = cells[(cells.cell == "inner_val") & cells.strategy.isin(ALL_NAMES)]
    sd = max(.418538, float(own.daily_sharpe.std(ddof=1)))
    ranking = cells[cells.strategy.isin(PRIMARY_NAMES) & cells.cell.isin(PRIMARY_CELLS[:2])]
    ranked = ranking.groupby("strategy").daily_sharpe.mean().rename("mean_validation_sharpe").to_frame()
    ranked["spot_fills"] = cells[cells.cell == "inner_val"].set_index("strategy").fills
    ranked = ranked.sort_values(["mean_validation_sharpe", "spot_fills"], ascending=[False, True])
    paths = ["experiments/r191_protocol.md", "experiments/r191_research_notes.md",
        "experiments/r191_strategies.py", "experiments/r191_eval.py",
        "experiments/r191_matched.py", "experiments/r191_report.py",
        "experiments/r190_eval.py", "experiments/r190_variations.py",
        "experiments/matched_hold.py", "src/tradebot/strategies/kelly_regime.py",
        "src/tradebot/strategies/kelly_regime_v3.py", "src/tradebot/strategies/kelly_regime_v4.py",
        "src/tradebot/strategies/buy_and_hold.py", "src/tradebot/engine.py",
        "src/tradebot/broker.py", "src/tradebot/data.py", "src/tradebot/inference.py",
        "src/tradebot/metrics.py", "src/tradebot/orders.py", "src/tradebot/strategy.py"]
    paths += ["data/" + f for f in evaluation.FILES.values()]
    paths += ["data/btcusdt_deribit_perp_funding_8h.csv.gz"]
    manifest = dict(frozen_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        protocol="experiments/r191_protocol.md", primary_candidates=list(PRIMARY_NAMES),
        local_configurations=len(ALL_NAMES), prior_consultations_approx=PRIOR,
        core_evaluations=455, core_holdout_evaluations=404,
        sd_trials=sd, validation_lead=str(ranked.index[0]),
        validation_ranking=ranked.reset_index().to_dict("records"),
        hashes={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths})
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen: training-only lead {ranked.index[0]}; trial SD {sd:.6f}")


def inference():
    rows = []
    for stage, wanted in (("train", PRIMARY_CELLS[:2]), ("holdout", PRIMARY_CELLS[2:])):
        _, daily = read(stage)
        matched, mdaily = read(stage, True)
        for cell in wanted:
            n = len(daily[PRIMARY_NAMES[0], cell])
            idx = stationary_bootstrap_indices(n, 30., 2000, np.random.default_rng(191))
            for name in PRIMARY_NAMES:
                match = matched[(matched.reference_candidate == name) & (matched.cell == cell)
                                & matched.final_selected].iloc[0]
                a, b = daily[name, cell], mdaily[match.strategy, cell]
                assert a.index.equals(b.index)
                row = dict(strategy=name, cell=cell, reference=match.strategy,
                    risk_valid=bool(match.matched_valid), control_c=match.control_c,
                    relative_vol_error=match.relative_vol_error, days=n)
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
    extra_train, extra_hold = int(audit.get("train_evaluations", 0)), int(audit.get("holdout_evaluations", 0))
    count = len(hold) + len(mh) + extra_hold
    indexed = cells.set_index(["strategy", "cell"])
    rows = []
    for name in PRIMARY_NAMES:
        comparisons = boot[boot.strategy == name]
        hb = comparisons[comparisons.cell.isin(PRIMARY_CELLS[2:])]
        dsrs, local = [], []
        for cell in PRIMARY_CELLS[2:]:
            r = daily[name, cell].to_numpy()
            sk, ku = moments(r)
            args = (annualized_sharpe(r), len(r), sk, ku)
            dsrs.append(deflated_sharpe_ratio(*args, n_trials=PRIOR + count, sd_trials=manifest["sd_trials"]))
            local.append(deflated_sharpe_ratio(*args, n_trials=len(ALL_NAMES), sd_trials=manifest["sd_trials"]))
        beta = {}
        for market in ("spot", "perp"):
            valid, wins = 0, 0
            for k in range(24):
                cell = f"beta_{market}_{k:02d}"
                match = mh[(mh.reference_candidate == name) & (mh.cell == cell) & mh.final_selected].iloc[0]
                valid += bool(match.matched_valid)
                wins += bool(match.matched_valid and indexed.loc[name, cell].final_balance > match.final_balance)
            beta.update({f"beta_{market}_valid": valid, f"beta_{market}_wins": wins})
        plateau = True
        for cell in PRIMARY_CELLS:
            f = indexed.loc[[(n, cell) for n in FAMILIES[name]]]
            plateau &= bool((f.final_balance > 1000).all() and f.daily_sharpe.max() - f.daily_sharpe.min() <= .20)
        own = cells[cells.strategy == name]
        gates = dict(
            matched_edge=bool(comparisons.risk_valid.all() and (comparisons.d_sharpe > .20).all()),
            intervals=bool((hb.d_sharpe_lo > 0).all() and (hb.d_growth_lo > 0).all()),
            absolute_growth=bool(all(indexed.loc[name, c].final_balance > indexed.loc["buy_and_hold", c].final_balance for c in PRIMARY_CELLS[2:])),
            falsification=bool(all(indexed.loc[name, c].final_balance > 1000 for c in
                ("holdout", "discount_holdout", "eth_holdout", "funded_holdout"))
                and indexed.loc[name, "eth_holdout"].daily_sharpe >= indexed.loc["buy_and_hold", "eth_holdout"].daily_sharpe
                and not own.liquidated.any()),
            plateau=bool(plateau), dsr=bool(min(dsrs) >= .95),
            beta=bool(beta["beta_spot_wins"] >= 13 and beta["beta_perp_wins"] >= 13))
        rows.append(dict(strategy=name, verdict="PROMOTED" if all(gates.values()) else "NEGATIVE",
            **gates, failed_gates=", ".join(k for k, v in gates.items() if not v),
            dsr_spot=dsrs[0], dsr_perp=dsrs[1], local_dsr_spot=local[0], local_dsr_perp=local[1],
            program_trials_approx=PRIOR + count, **beta))
    decisions = pd.DataFrame(rows)
    decisions.to_csv(OUT / "decision.csv", index=False)
    counts = dict(candidate_configurations=5, auxiliary_configurations=10,
        core_evaluations=len(cells), matching_evaluations=len(mt) + len(mh),
        audit_train_evaluations=extra_train, audit_holdout_evaluations=extra_hold,
        total_evaluations=len(cells) + len(mt) + len(mh) + extra_train + extra_hold,
        holdout_consultations=count, cumulative_consultations_approx=PRIOR + count)
    (OUT / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    return cells, decisions, counts


def chart(cells, boot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, daily = read("holdout")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    for ax, cell, title in zip(axes[0], PRIMARY_CELLS[2:], ("BTC spot · 40bp + 1bp", "BTC perp · 5bp + 1bp + funding")):
        for name in (*PRIMARY_NAMES, "buy_and_hold", "kelly_regime_v4"):
            r = daily[name, cell]
            ax.plot(pd.to_datetime(r.index), 1000 * (1 + r).cumprod(),
                label=LABELS.get(name, name), lw=1.6, ls="-" if name in PRIMARY_NAMES else "--")
        ax.set(title=title, ylabel="Account value ($), log scale", yscale="log")
        ax.grid(alpha=.2)
    axes[0, 1].legend(fontsize=8)
    for ax, cell, title in zip(axes[1], PRIMARY_CELLS[2:], ("Spot", "Funded perpetual")):
        b = boot[boot.cell == cell].set_index("strategy").loc[list(PRIMARY_NAMES)]
        y = np.arange(len(b))
        ax.hlines(y, b.d_sharpe_lo, b.d_sharpe_hi, color="tab:blue")
        ax.scatter(b.d_sharpe, y, color="tab:blue")
        for i, valid in enumerate(b.risk_valid):
            if not valid:
                ax.plot(b.d_sharpe.iloc[i], i, "rx", ms=10)
        ax.axvline(0, color="gray", lw=1)
        ax.axvline(.20, color="green", ls="--", lw=1)
        ax.set(yticks=y, yticklabels=[LABELS[n] for n in PRIMARY_NAMES],
               title=f"{title}: Sharpe difference vs matched passive hold · 95% CI",
               xlabel="Annualized daily Sharpe difference; red × = invalid risk match")
    fig.suptitle("R-191 · Five cost-aware allocation strategies", fontsize=17)
    fig.savefig(OUT / "summary.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "report"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        evaluation.validate_manifest()
        boot = inference()
        cells, decisions, counts = decide(boot)
        chart(cells, boot)
        print(decisions.to_string(index=False))
        print(json.dumps(counts, indent=2))
