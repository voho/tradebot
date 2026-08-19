"""B-14 (R-36): does the matched-risk return advantage survive outside the
2017-2020 bull?

Not registered: lives under ``experiments/`` per ROUTINE.md step 5.

Reuses ``reports/matched_hold/windows.csv`` (R-33's already-published,
seed=42 40-window resample from ``experiments/run_matched_hold.py
windows``) rather than re-running any backtest. The only new computation
is recovering each window's *start date*, by replaying the identical
``rng(seed=42)`` draw sequence ``windows()`` uses -- same warmup, same
``length``/``start`` draws, same order -- and using it to look up the
calendar date in the committed dataset. Splitting the existing per-window
return/vol numbers by that date is the entire new analysis; see the R-36
pre-registration in docs/LEDGER.md, committed before this file was run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as _stats  # noqa: E402

from tradebot.data import load_dataset  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
OUT = ROOT / "reports" / "matched_hold"

INCUMBENT = "kelly_regime_v4"
SEED = 42
TRIALS = 40
SPLIT_DATE = pd.Timestamp("2021-01-01", tz="UTC")


def _binom_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) 95% CI on a win-rate, no extra dependency
    beyond scipy (already a transitive dependency of matplotlib/pandas'
    test stack; if unavailable, fall back to a normal approximation)."""
    lo = _stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = _stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def recover_window_dates() -> pd.DataFrame:
    """Replay windows()'s exact rng sequence to recover each trial's
    calendar start date, without re-running any backtest."""
    # Same contenders list windows() builds warmup from: buy_and_hold,
    # v4, and the three frozen passive arms (all warmup=0). max() is v4's.
    v4 = get_strategy(INCUMBENT)
    warmup = v4.warmup + 10

    rng = np.random.default_rng(SEED)
    rows = []
    for trial in range(1, TRIALS + 1):
        length = int(rng.integers(90, 731) * 288)
        start_idx = int(rng.integers(warmup, len(DF) - length))
        start_date = DF.index[start_idx]
        end_date = DF.index[min(start_idx + length, len(DF) - 1)]
        rows.append({"trial": trial, "start_idx": start_idx,
                     "start_date": start_date, "end_date": end_date,
                     "length_days": length // 288})
    return pd.DataFrame(rows)


def main() -> None:
    dates = recover_window_dates()
    windows = pd.read_csv(OUT / "windows.csv")
    dates.to_csv(OUT / "window_dates.csv", index=False)
    print(f"recovered {len(dates)} window start dates "
          f"(warmup+10={get_strategy(INCUMBENT).warmup + 10}, seed={SEED})")
    print(dates[["trial", "start_date", "end_date", "length_days"]]
          .to_string(index=False))

    pre = set(dates[dates.start_date < SPLIT_DATE].trial)
    post = set(dates[dates.start_date >= SPLIT_DATE].trial)
    print(f"\n{len(pre)} windows start before {SPLIT_DATE.date()} "
          f"(inner-train era), {len(post)} start on/after it "
          "(inner-validation era or later)")

    results = {}
    for mname in ("spot", "futures"):
        sub = windows[windows.market == mname]
        v4 = sub[sub.strategy == INCUMBENT].set_index("trial")["return_pct"]
        matched = (sub[sub.strategy == "per-window matched hold"]
                   .set_index("trial")["return_pct"])
        diff = (v4 - matched).dropna()

        def stat_block(trial_set, label):
            d = diff[diff.index.isin(trial_set)]
            n = len(d)
            k = int((d > 0).sum())
            lo, hi = _binom_ci(k, n) if n else (float("nan"), float("nan"))
            print(f"  {mname:8s} {label:26s} n={n:2d}  win-rate {k}/{n} "
                  f"= {k / n:.1%}  95% CI [{lo:.1%}, {hi:.1%}]  "
                  f"median Δreturn {d.median():+7.1f}pp")
            return {"n": n, "wins": k, "win_rate": k / n if n else float("nan"),
                    "ci_lo": lo, "ci_hi": hi, "median_diff": d.median()}

        print(f"\n{mname}: v4 - per-window-matched-hold, paired return (pp)")
        results[(mname, "pooled")] = stat_block(diff.index, "pooled (D1)")
        results[(mname, "pre-2021 start")] = stat_block(pre, "pre-2021 start (bull-heavy)")
        results[(mname, "post-2021 start")] = stat_block(post, "post-2021 start (falsification)")

    print("\n--- decision, against the rule pre-registered in docs/LEDGER.md (R-36) ---")
    for mname in ("spot", "futures"):
        pooled = results[(mname, "pooled")]
        d1 = "PASS" if pooled["ci_lo"] > 0.5 else "FAIL"
        print(f"  {mname:8s} D1 (pooled CI excludes 50%): {d1}  "
              f"[{pooled['ci_lo']:.1%}, {pooled['ci_hi']:.1%}]")
        post = results[(mname, "post-2021 start")]
        fails_falsification = (post["win_rate"] <= 0.5) or (post["median_diff"] <= 0)
        print(f"  {mname:8s} falsification (post-2021 windows still win): "
              f"{'FAILS -> bull-period artifact' if fails_falsification else 'SURVIVES'}  "
              f"win-rate {post['win_rate']:.1%}, median {post['median_diff']:+.1f}pp")

    out_rows = [{"market": m, "segment": s, **v} for (m, s), v in results.items()]
    pd.DataFrame(out_rows).to_csv(OUT / "regime_breakdown.csv", index=False)
    print(f"\nwritten: {OUT / 'regime_breakdown.csv'}, {OUT / 'window_dates.csv'}")


if __name__ == "__main__":
    main()
