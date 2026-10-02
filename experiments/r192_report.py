"""R-192 training power, frozen manifest and exhaustive parent/passive gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "experiments")]

import numpy as np
import pandas as pd

from experiments import r192_eval as evaluation
from experiments import r192_matched as matching
from r192_strategies import PRIMARY_NAMES, ALL_NAMES, FAMILIES
from tradebot.inference import (annualized_sharpe, deflated_sharpe_ratio,
    max_drawdown_from_returns as _drawdown_without_initial, moments, paired_bootstrap,
    stationary_bootstrap_indices, total_log_return)

OUT = evaluation.OUT
PRIOR = 3251
PARENT = "kelly_regime_v4"
PRIMARY_CELLS = ("inner_val", "funded_val", "holdout", "funded_holdout")


def daily_drawdown(rets):
    """Drawdown in percentage points, including the initial capital peak.

    The shared helper starts its peak at the first observed equity close.
    Our daily returns include the first day's PnL, so prepend a zero return
    to retain the starting account value in both original and bootstrap paths.
    """
    rets = np.asarray(rets, dtype=float)
    initial = np.zeros(rets.shape[:-1] + (1,))
    return _drawdown_without_initial(np.concatenate((initial, rets), axis=-1))


STATS = (("sharpe", annualized_sharpe), ("growth", total_log_return),
         ("drawdown", daily_drawdown))


def expected_core(stage):
    return {(name, spec[0]) for name in evaluation.WORKER_NAMES
        for spec in evaluation.specifications(stage)
        if name not in evaluation.AUXILIARIES or not spec[0].startswith("beta_")}


def read(stage, matched=False):
    tag = f"{stage}_matched" if matched else stage
    cells = pd.read_csv(OUT / f"{tag}_cells.csv")
    daily = pd.read_csv(OUT / f"{tag}_daily.csv.gz")
    if cells.duplicated(["strategy", "cell"]).any():
        raise ValueError(f"Duplicate evaluation receipts: {tag}")
    if daily.duplicated(["strategy", "cell", "timestamp"]).any():
        raise ValueError(f"Duplicate daily observations: {tag}")
    if not np.isfinite(daily["return"].to_numpy()).all():
        raise ValueError(f"Nonfinite daily returns: {tag}")
    if matched:
        selected = cells.loc[cells.final_selected]
        expected = {(name, spec[0]) for name in PRIMARY_NAMES for spec in matching.matching_specs(stage)}
        if selected.duplicated(["reference_candidate", "cell"]).any():
            raise ValueError(f"Multiple selected matches: {tag}")
        observed = set(zip(selected.reference_candidate, selected.cell))
    else:
        expected = expected_core(stage)
        observed = set(zip(cells.strategy, cells.cell))
    if observed != expected:
        raise ValueError(f"Incomplete or unexpected evaluation receipts: {tag}")
    series = {}
    for (name, cell), group in daily.groupby(["strategy", "cell"], sort=False):
        r = pd.Series(group["return"].to_numpy(), index=pd.to_datetime(group.timestamp, utc=True)).sort_index()
        series[name, cell] = r
    if set(series) != set(zip(cells.strategy, cells.cell)):
        raise ValueError(f"Daily/metric receipt disagreement: {tag}")
    return cells, series


def align(a, b):
    if not a.index.equals(b.index):
        raise ValueError("Paired return timestamps differ")
    return a.to_numpy(), b.to_numpy()


def risk_ratio(a, b, tolerance):
    av, bv = float(a.std(ddof=1)), float(b.std(ddof=1))
    ratio = bv / av if av > 0 else np.nan
    return ratio, bool(np.isfinite(ratio) and abs(ratio - 1.) <= tolerance)


def references(name, cell, daily, matches, mdaily):
    selected = matches[(matches.reference_candidate == name) & (matches.cell == cell)
                       & matches.final_selected]
    if len(selected) != 1:
        raise ValueError(f"Expected one selected risk match: {name} {cell}")
    match = selected.iloc[0]
    return [("parent", PARENT, daily[PARENT, cell], True, .05, np.nan),
            ("matched_hold", match.strategy, mdaily[match.strategy, cell],
             bool(match.matched_valid), .02, float(match.control_c))]


def comparisons(stage, cells_wanted, *, include_power=False):
    _, daily = read(stage)
    matches, mdaily = read(stage, True)
    rows = []
    for cell in cells_wanted:
        n = len(daily[PRIMARY_NAMES[0], cell])
        idx = stationary_bootstrap_indices(n, 30., 2000, np.random.default_rng(192))
        for name in PRIMARY_NAMES:
            a = daily[name, cell]
            for kind, ref, b, valid, tolerance, c in references(name, cell, daily, matches, mdaily):
                aa, bb = align(a, b)
                if len(a) != n:
                    raise ValueError("Candidate return histories have different lengths")
                ratio, within = risk_ratio(a, b, tolerance)
                row = dict(strategy=name, cell=cell, control=kind, reference=ref,
                    risk_valid=bool(valid and within), control_c=c,
                    control_to_candidate_vol=ratio, days=n)
                for label, stat in STATS:
                    p = paired_bootstrap(aa, bb, stat, indices=idx)
                    row.update({label: p.stat_a, "control_" + label: p.stat_b,
                        "d_" + label: p.diff.point, "d_" + label + "_lo": p.diff.lo,
                        "d_" + label + "_hi": p.diff.hi})
                    if include_power and label in ("sharpe", "drawdown"):
                        draws = stat(aa[idx]) - stat(bb[idx])
                        se = float(draws.std(ddof=1))
                        threshold = .20 if label == "sharpe" else 2.
                        # A stationary-noise planning approximation, not a
                        # guarantee for a nonlinear growing-horizon statistic.
                        required = n * (2.8 * se / threshold) ** 2
                        row.update({f"{label}_bootstrap_se": se,
                            f"{label}_effect_threshold": threshold,
                            f"{label}_required_days_approx": required,
                            f"{label}_required_years_approx": required / 365.25})
                if include_power:
                    diff = np.log1p(np.clip(aa, -1 + 1e-15, None)) - np.log1p(np.clip(bb, -1 + 1e-15, None))
                    row["daily_logdiff_sd"] = float(diff.std(ddof=1))
                    row["block_mean_logdiff_se"] = float(diff[idx].mean(axis=1).std(ddof=1))
                    row["power_limitation"] = "stationary-noise approximation; nonlinear drawdown horizon invalidates exact sample scaling"
                rows.append(row)
    return pd.DataFrame(rows)


def power():
    result = comparisons("train", PRIMARY_CELLS[:2], include_power=True)
    result.to_csv(OUT / "training_power.csv", index=False)
    return result


def freeze():
    path = OUT / "manifest.json"
    if path.exists():
        evaluation.validate_manifest()
        return
    cells, _ = read("train")
    power()
    own = cells[(cells.cell == "inner_val") & cells.strategy.isin(ALL_NAMES)]
    sd = max(.418538, float(own.daily_sharpe.std(ddof=1)))
    ranking = cells[cells.strategy.isin(PRIMARY_NAMES) & cells.cell.isin(PRIMARY_CELLS[:2])]
    ranked = ranking.groupby("strategy").daily_sharpe.mean().rename("mean_validation_sharpe").to_frame()
    ranked["spot_fills"] = cells[cells.cell == "inner_val"].set_index("strategy").fills
    ranked = ranked.sort_values(["mean_validation_sharpe", "spot_fills"], ascending=[False, True])
    # Include all local runtime source: registration can import additional
    # strategies, and a hand-picked dependency list can silently miss them.
    paths = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src/tradebot").rglob("*.py"))
    paths += ["experiments/r192_protocol.md", "experiments/r192_research_notes.md",
        "experiments/r192_strategies.py", "experiments/r192_eval.py",
        "experiments/r192_matched.py", "experiments/r192_report.py",
        "experiments/r190_eval.py", "experiments/r190_variations.py", "experiments/matched_hold.py",
        "tests/test_r192_eval.py", "tests/test_r192_strategies.py", "tests/test_r192_audit.py"]
    paths += ["data/" + f for f in evaluation.FILES.values()]
    paths += ["data/btcusdt_deribit_perp_funding_8h.csv.gz"]
    missing = [p for p in paths if not (ROOT / p).exists()]
    if missing:
        raise ValueError(f"Cannot freeze missing source/test/audit files: {missing}")
    manifest = dict(frozen_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        protocol="experiments/r192_protocol.md", primary_candidates=list(PRIMARY_NAMES),
        local_configurations=len(ALL_NAMES), prior_consultations_approx=PRIOR,
        core_evaluations=len(expected_core("train")) + len(expected_core("holdout")),
        core_holdout_evaluations=len(expected_core("holdout")),
        sd_trials=sd, validation_lead=str(ranked.index[0]),
        validation_ranking=ranked.reset_index().to_dict("records"),
        initialization="common first-globally-eligible positive-target convention; subsequent native callback",
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        hashes={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths})
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen: descriptive training lead {ranked.index[0]}; trial SD {sd:.6f}")


def inference():
    result = pd.concat([comparisons("train", PRIMARY_CELLS[:2]),
                        comparisons("holdout", PRIMARY_CELLS[2:])], ignore_index=True)
    result.to_csv(OUT / "bootstrap.csv", index=False)
    return result


def _metric(daily, name, cell):
    r = daily[name, cell].to_numpy()
    return {label: float(stat(r)) for label, stat in STATS}


def validated_audit():
    """A final verdict requires all four successful fixed-subject replays."""
    path = OUT / "audit.json"
    if not path.exists():
        raise ValueError("Final report requires the completed four-cell audit")
    audit = json.loads(path.read_text())
    subject = "r192_tracking_budget"
    expected = {"inner_val": "train", "funded_val": "train",
                "holdout": "holdout", "funded_holdout": "holdout"}
    records = audit.get("reproductions")
    valid = (audit.get("subject") == subject
        and audit.get("train_evaluations") == 2
        and audit.get("holdout_evaluations") == 2
        and audit.get("discrepancies") == []
        and isinstance(records, list) and len(records) == 4)
    if valid:
        valid = all(isinstance(r, dict) and r.get("strategy") == subject
            and r.get("status") == "completed"
            and r.get("cell") in expected
            and r.get("stage") == expected[r["cell"]] for r in records)
        valid = valid and {r["cell"] for r in records} == set(expected)
    if not valid:
        raise ValueError("Final report requires exactly four completed tracking-budget audit cells, "
                         "two train/two holdout, with no discrepancies")
    return audit


def decide(boot):
    audit = validated_audit()
    train, tdaily = read("train")
    hold, hdaily = read("holdout")
    mt, _ = read("train", True)
    mh, mdaily = read("holdout", True)
    cells = pd.concat([train, hold], ignore_index=True)
    cells.to_csv(OUT / "cells.csv", index=False)
    indexed = cells.set_index(["strategy", "cell"])
    expected = {(n, c, ref) for n in PRIMARY_NAMES for c in PRIMARY_CELLS
                for ref in ("parent", "matched_hold")}
    if boot.duplicated(["strategy", "cell", "control"]).any() or set(
            zip(boot.strategy, boot.cell, boot.control)) != expected:
        raise ValueError("Incomplete or duplicate paired inference")
    manifest = json.loads((OUT / "manifest.json").read_text())
    extra_train = audit["train_evaluations"]
    extra_hold = audit["holdout_evaluations"]
    count = len(hold) + len(mh) + extra_hold
    rows, windows = [], []
    for name in PRIMARY_NAMES:
        b = boot[boot.strategy == name]
        hb = b[b.cell.isin(PRIMARY_CELLS[2:])]
        dsrs, local = [], []
        for cell in PRIMARY_CELLS[2:]:
            r = hdaily[name, cell].to_numpy()
            sk, ku = moments(r)
            args = (annualized_sharpe(r), len(r), sk, ku)
            dsrs.append(deflated_sharpe_ratio(*args, n_trials=PRIOR + count, sd_trials=manifest["sd_trials"]))
            local.append(deflated_sharpe_ratio(*args, n_trials=len(ALL_NAMES), sd_trials=manifest["sd_trials"]))
        beta = {}
        for market in ("spot", "perp"):
            valid, growth, tail = 0, 0, 0
            for k in range(24):
                cell = f"beta_{market}_{k:02d}"
                a = hdaily[name, cell]
                own_metric = _metric(hdaily, name, cell)
                ok, gw, tw = True, True, True
                for kind, ref, control, permitted, tolerance, _ in references(name, cell, hdaily, mh, mdaily):
                    aa, bb = align(a, control)
                    ratio, within = risk_ratio(a, control, tolerance)
                    risk = bool(permitted and within)
                    dg = float(total_log_return(aa) - total_log_return(bb))
                    dd = float(daily_drawdown(aa) - daily_drawdown(bb))
                    ok &= risk
                    gw &= risk and dg > 0
                    tw &= risk and dd < -2. and dg >= 0
                    windows.append(dict(strategy=name, cell=cell, control=kind, reference=ref,
                        risk_valid=risk, control_to_candidate_vol=ratio, d_growth=dg, d_drawdown=dd))
                valid += ok
                growth += gw
                tail += tw
            beta.update({f"beta_{market}_valid": valid, f"beta_{market}_growth_wins": growth,
                         f"beta_{market}_tail_wins": tail})
        eth = _metric(hdaily, name, "eth_holdout")
        eth_refs = [_metric(hdaily, ref, "eth_holdout") for ref in (PARENT, "buy_and_hold")]
        eth_growth = all(eth["sharpe"] >= ref["sharpe"] for ref in eth_refs)
        eth_tail = all(eth["drawdown"] < ref["drawdown"] and eth["growth"] >= ref["growth"] for ref in eth_refs)
        plateau = True
        family = cells[cells.strategy.isin(FAMILIES[name]) & cells.cell.isin(PRIMARY_CELLS)]
        for cell in PRIMARY_CELLS:
            f = indexed.loc[[(n, cell) for n in FAMILIES[name]]]
            plateau &= bool((f.final_balance > 1000).all() and f.daily_sharpe.max() - f.daily_sharpe.min() <= .20)
        own = cells[cells.strategy == name]
        binding = all(indexed.loc[name, c].fill_signature != indexed.loc[PARENT, c].fill_signature
                      for c in PRIMARY_CELLS[:2])
        common = dict(risk_valid=bool(b.risk_valid.all()), mechanism_binding=bool(binding),
            profitable=bool(all(indexed.loc[name, c].final_balance > 1000 for c in
                ("holdout", "discount_holdout", "eth_holdout", "funded_holdout"))),
            no_liquidation=bool(not family.liquidated.any()), plateau=bool(plateau),
            dsr=bool(np.isfinite(dsrs).all() and min(dsrs) >= .95))
        growth_gates = dict(growth_effect=bool((b.d_sharpe > .20).all()),
            growth_intervals=bool((hb.d_sharpe_lo > 0).all() and (hb.d_growth_lo > 0).all()),
            growth_full_passive=bool(all(indexed.loc[name, c].final_balance > indexed.loc["buy_and_hold", c].final_balance
                                        for c in PRIMARY_CELLS[2:])),
            growth_eth=bool(eth_growth), growth_beta=bool(beta["beta_spot_growth_wins"] >= 13 and beta["beta_perp_growth_wins"] >= 13))
        tail_gates = dict(tail_effect=bool((b.d_drawdown < -2.).all()),
            tail_intervals=bool((hb.d_drawdown_hi < 0).all() and (hb.d_growth_lo >= 0).all()),
            tail_eth=bool(eth_tail), tail_beta=bool(beta["beta_spot_tail_wins"] >= 13 and beta["beta_perp_tail_wins"] >= 13))
        growth_pass, tail_pass = all(growth_gates.values()), all(tail_gates.values())
        passed = all(common.values()) and (growth_pass or tail_pass)
        rows.append(dict(strategy=name, verdict="PROMOTED" if passed else "NEGATIVE",
            route=("growth+tail" if growth_pass and tail_pass else "growth" if growth_pass else "tail" if tail_pass else "none"),
            **common, **growth_gates, **tail_gates, growth_route=growth_pass, tail_route=tail_pass,
            failed_common=", ".join(k for k, v in common.items() if not v),
            failed_growth=", ".join(k for k, v in growth_gates.items() if not v),
            failed_tail=", ".join(k for k, v in tail_gates.items() if not v),
            dsr_spot=dsrs[0], dsr_perp=dsrs[1], local_dsr_spot=local[0], local_dsr_perp=local[1],
            program_trials_approx=PRIOR + count, **beta))
    decisions = pd.DataFrame(rows)
    decisions.to_csv(OUT / "decision.csv", index=False)
    pd.DataFrame(windows).to_csv(OUT / "window_comparisons.csv", index=False)
    counts = dict(candidate_configurations=len(PRIMARY_NAMES), auxiliary_configurations=len(ALL_NAMES) - len(PRIMARY_NAMES),
        core_evaluations=len(cells), matching_evaluations=len(mt) + len(mh),
        audit_train_evaluations=extra_train, audit_holdout_evaluations=extra_hold,
        total_evaluations=len(cells) + len(mt) + len(mh) + extra_train + extra_hold,
        holdout_consultations=count, cumulative_consultations_approx=PRIOR + count)
    (OUT / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    return cells, decisions, counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "report"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        evaluation.validate_manifest()
        cells, decisions, counts = decide(inference())
        print(decisions.to_string(index=False))
        print(json.dumps(counts, indent=2))
