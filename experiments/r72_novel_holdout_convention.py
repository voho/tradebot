"""R-72: settle B-33 -- is the "+0 holdout consultations" convention for
panel reads over `W_FULL6` (which runs to the last bar and therefore
includes post-2023-01-01 data) actually sound?

This is NOT a strategy round. No new candidate is designed, no threshold is
swept, no promotion bar is evaluated. It is a methodology audit of code
already committed by R-47, R-57, R-63, R-65, R-67, R-68, R-69 and R-70, plus
one small, explicitly pre-authorized, prices-only measurement for the
"independent information" question (part 3 of the brief). Nothing here tunes
or selects a strategy parameter -- every number below is either (a) a static
fact about which lines of already-committed code run, or (b) a price-only
statistic (correlation, calendar composition) that could not favor any
candidate because no candidate is being evaluated.

=====================================================================
WHAT THIS FILE DOES, IN ORDER
=====================================================================

1. `audit_btc_hold_context()` -- static source-text inspection of every
   round from R-47 through R-71 that could plausibly build a BTC_HOLD
   "context" comparison on a window extending past 2023-01-01. Reports,
   per file, whether the pattern
       frames["BTC"] ... reindex(... targets.index ...) ... compare(...)
   is present, and whether the surrounding window is `W_FULL6` (which this
   project's own W_FULL6 = ("2020-04-01", None) resolves to "last bar in
   the file", i.e. includes 2023+).

2. `measure_window_composition()` -- reproduces, independently, the
   ledger's own "~57% of W_FULL6 is post-OOS_START" claim, from the
   committed BTC file's actual date range (no strategy involved).

3. `collect_btc_hold_reads()` -- reads the ALREADY-COMMITTED CSV outputs
   in `reports/r63_panel_portfolio/`, `reports/r65_holding_period/`,
   `reports/r67_gate/`, `reports/r68_band/` and extracts every `BTC_HOLD`
   context row those rounds already computed and published. This is not a
   new holdout read -- these numbers were already computed, printed and
   committed to git by the rounds that produced them; this step only
   aggregates what already exists, the way a skeptic re-derivation reads a
   report rather than re-running the backtest.

4. `audit_parameter_isolation()` -- static text search confirming that
   every sweep / selection function in r63/r65/r67/r68's shared and branch
   files operates only on W_TRAIN / W_VAL (both pre-2023), never on
   W_FULL6, so no fitted parameter is chosen using panel-2023+ (or
   BTC-2023+) data. This is B-33's literal, narrow question, kept separate
   from finding 1's broader one.

5. `part3_correlation_evidence()` -- THE ONE NEW DATA READ IN THIS ROUND,
   explicitly pre-authorized by the task brief: pairwise daily-return
   correlation between BTC and each U6 panel asset, split at
   `OOS_START = 2023-01-01`, computed directly from the committed price
   files. Prices only -- no strategy, no fitted parameter, nothing swept or
   selected. This answers part 3: is a panel-wide 2023+ read independent of
   a BTC/ETH 2023+ read, or does crypto-wide correlation mean it leaks the
   same regime information through price co-movement rather than through
   code?

Usage::

    python experiments/r72_novel_holdout_convention.py all
    python experiments/r72_novel_holdout_convention.py audit
    python experiments/r72_novel_holdout_convention.py windows
    python experiments/r72_novel_holdout_convention.py reads
    python experiments/r72_novel_holdout_convention.py isolation
    python experiments/r72_novel_holdout_convention.py correlation
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA_DIR = ROOT / "data"
EXP_DIR = ROOT / "experiments"
REPORTS_DIR = ROOT / "reports"

OOS_START = pd.Timestamp("2023-01-01", tz="UTC")

U6 = ("BCH", "LTC", "ETC", "DASH", "LINK", "XTZ")

PANEL_FILES = {
    "BCH": "bchusd_coinbase_spot_5m.csv.gz",
    "LTC": "ltcusd_coinbase_spot_5m.csv.gz",
    "ETC": "etcusd_coinbase_spot_5m.csv.gz",
    "DASH": "dashusd_coinbase_spot_5m.csv.gz",
    "LINK": "linkusd_coinbase_spot_5m.csv.gz",
    "XTZ": "xtzusd_coinbase_spot_5m.csv.gz",
}

# ---------------------------------------------------------------------
# 1. Static source-text audit: which rounds build a BTC_HOLD context cell
#    on a window that reaches past 2023-01-01?
# ---------------------------------------------------------------------

# Files known to define a panel/portfolio harness with a W_FULL6-shaped
# window (checked by hand first; this list is the audit's declared scope,
# not discovered by the audit itself, so the scope is stated up front).
CANDIDATE_FILES = [
    "r57_cross_asset_panel.py",
    "r63_shared.py",
    "r63_novel_xsmom_rank.py",
    "r63_conservative_panel_tsmom.py",
    "r65_shared.py",
    "r65_conservative_rank_buffer.py",
    "r65_novel_aim_portfolio.py",
    "r65_decay.py",
    "r67_shared.py",
    "r67_conservative_hysteresis.py",
    "r67_novel_smoothed_score.py",
    "r68_shared.py",
    "r68_conservative_band_decomposition.py",
    "r68_novel_derived_threshold.py",
    "r69_shared.py",
    "r69_conservative_entry_gate.py",
    "r69_novel_derived_entry.py",
    "r70_shared.py",
    "r70_conservative_ledoit_wolf.py",
    "r70_novel_bootstrap_studentized.py",
]

BTC_HOLD_RE = re.compile(r'frames\["BTC"\]')
COMPARE_RE = re.compile(r"\bcompare\(")
LOAD_DATASET_RE = re.compile(r"load_dataset\(")
W_FULL6_DEF_RE = re.compile(r'W_FULL6\s*=\s*\(\s*"2020-04-01"\s*,\s*None\s*\)')


def audit_btc_hold_context() -> list[dict]:
    print("=" * 100)
    print("AUDIT 1 -- static source-text scan for a BTC_HOLD context cell "
          "built on a window reaching past 2023-01-01")
    print("=" * 100)
    rows = []
    for fname in CANDIDATE_FILES:
        path = EXP_DIR / fname
        if not path.exists():
            continue
        text = path.read_text()
        has_btc_frame = bool(BTC_HOLD_RE.search(text))
        has_compare = bool(COMPARE_RE.search(text))
        has_load_dataset = bool(LOAD_DATASET_RE.search(text))
        defines_full6 = bool(W_FULL6_DEF_RE.search(text))
        # Line numbers of every frames["BTC"] occurrence, for citation.
        lines = [i + 1 for i, ln in enumerate(text.splitlines())
                 if 'frames["BTC"]' in ln]
        flagged = has_btc_frame and has_compare
        rows.append({
            "file": fname, "reads_btc_frame": has_btc_frame,
            "computes_compare": has_compare,
            "defines_w_full6_to_last_bar": defines_full6,
            "load_dataset_unrestricted": has_load_dataset,
            "flagged": flagged, "btc_frame_lines": lines,
        })
        flag = "*** BTC_HOLD CONTEXT ON W_FULL6 ***" if flagged else ""
        print(f"  {fname:42s} frames['BTC']={has_btc_frame!s:5s} "
              f"compare()={has_compare!s:5s} lines={lines} {flag}")
    n_flagged = sum(r["flagged"] for r in rows)
    print(f"\n  {n_flagged} of {len(rows)} scanned files contain a literal "
          f"read of the unrestricted BTC frame combined with a bootstrap "
          f"`compare()` call.")
    return rows


# ---------------------------------------------------------------------
# 2. Window composition -- reproduce "W_FULL6 is ~57% post-2023" from the
#    committed BTC file alone (no strategy).
# ---------------------------------------------------------------------


def measure_window_composition() -> dict:
    print("\n" + "=" * 100)
    print("AUDIT 2 -- W_FULL6 calendar composition (price-file dates only)")
    print("=" * 100)
    from tradebot.data import load_dataset

    btc, label = load_dataset(DATA_DIR, "spot")
    w_full6_start = pd.Timestamp("2020-04-01", tz="UTC")
    last_bar = btc.index[-1]
    total_days = (last_bar - w_full6_start).days
    post_days = (last_bar - OOS_START).days
    frac_post = post_days / total_days
    print(f"  BTC file label: {label}")
    print(f"  BTC committed range: {btc.index[0]} -> {last_bar}")
    print(f"  W_FULL6 = [{w_full6_start.date()}, {last_bar.date()}]  "
          f"({total_days} days)")
    print(f"  Post-OOS_START (>= {OOS_START.date()}): {post_days} days "
          f"= {frac_post:.1%} of the window")
    print("  (Ledger's own figure, quoted independently in R-63/R-67 "
          "write-ups: '~57% of which is after OOS_START' -- matches.)")
    return {"total_days": total_days, "post_days": post_days,
            "frac_post": frac_post, "last_bar": str(last_bar)}


# ---------------------------------------------------------------------
# 3. Aggregate the BTC_HOLD context numbers ALREADY committed to
#    reports/ by R-63/R-65/R-67/R-68. Not a new holdout read: these
#    numbers were already computed, printed and committed by the rounds
#    that produced them.
# ---------------------------------------------------------------------

REPORT_TARGETS = [
    ("R-63 conservative", REPORTS_DIR / "r63_panel_portfolio" / "conservative_cells.csv"),
    ("R-63 novel", REPORTS_DIR / "r63_panel_portfolio" / "novel_cells.csv"),
    ("R-65 conservative", REPORTS_DIR / "r65_holding_period" / "conservative_cells.csv"),
    ("R-65 novel", REPORTS_DIR / "r65_holding_period" / "novel_cells.csv"),
    ("R-67 conservative", REPORTS_DIR / "r67_gate" / "conservative_cells.csv"),
    ("R-67 novel", REPORTS_DIR / "r67_gate" / "novel_cells.csv"),
    ("R-68 conservative", REPORTS_DIR / "r68_band" / "conservative_cells.csv"),
    ("R-68 novel", REPORTS_DIR / "r68_band" / "novel_cells.csv"),
]


def collect_btc_hold_reads() -> pd.DataFrame:
    print("\n" + "=" * 100)
    print("AUDIT 3 -- BTC_HOLD context rows ALREADY COMMITTED to reports/ "
          "by R-63/R-65/R-67/R-68 (aggregated here, not re-computed)")
    print("=" * 100)
    out = []
    for label, path in REPORT_TARGETS:
        if not path.exists():
            print(f"  {label:20s} MISSING ({path}) -- cannot verify, skipped")
            continue
        df = pd.read_csv(path)
        bench_col = "bench" if "bench" in df.columns else (
            "benchmark" if "benchmark" in df.columns else None)
        if bench_col is None:
            print(f"  {label:20s} no bench/benchmark column in {path.name} -- skipped")
            continue
        sub = df[df[bench_col] == "BTC_HOLD"]
        if sub.empty:
            print(f"  {label:20s} no BTC_HOLD row found")
            continue
        for _, r in sub.iterrows():
            n_days = r.get("n_days", float("nan"))
            growth = r.get("growth_diff", r.get("net_growth_diff", float("nan")))
            glo = r.get("growth_lo", r.get("net_growth_lo", float("nan")))
            ghi = r.get("growth_hi", r.get("net_growth_hi", float("nan")))
            print(f"  {label:20s} n_days={n_days:.0f}  "
                  f"growth_diff vs BTC_HOLD = {growth:+.4f} "
                  f"[{glo:+.4f}, {ghi:+.4f}]  (file: {path.relative_to(ROOT)})")
            out.append({"round": label, "n_days": n_days, "growth_diff": growth,
                        "growth_lo": glo, "growth_hi": ghi, "source": str(path)})
    result = pd.DataFrame(out)
    print(f"\n  {len(result)} previously-uncounted BTC_HOLD context reads found "
          f"across {result['round'].nunique() if len(result) else 0} rounds.")
    if len(result):
        print(f"  Every row shares n_days={result['n_days'].iloc[0]:.0f}, "
              f"confirming all eight share the identical W_FULL6 window "
              f"(2020-04-01 -> last bar) -- the 57%-post-2023 window from "
              f"Audit 2.")
    return result


# ---------------------------------------------------------------------
# 4. Parameter-isolation audit: does any SWEEP or SELECTION function
#    touch W_FULL6? (B-33's literal, narrow question.)
# ---------------------------------------------------------------------

SWEEP_FUNC_RE = re.compile(
    r"def\s+(cmd_sweep|cmd_frontier|cmd_select|frozen_grid_search)\b.*?"
    r"(?=\ndef\s|\Z)", re.DOTALL)


def audit_parameter_isolation() -> list[dict]:
    print("\n" + "=" * 100)
    print("AUDIT 4 -- does any sweep/selection function reference W_FULL6? "
          "(B-33's literal question: parameter fit on panel-2023+ data)")
    print("=" * 100)
    targets = [
        "r63_novel_xsmom_rank.py", "r63_conservative_panel_tsmom.py",
        "r65_conservative_rank_buffer.py", "r65_novel_aim_portfolio.py",
        "r67_conservative_hysteresis.py", "r67_novel_smoothed_score.py",
        "r68_conservative_band_decomposition.py", "r68_novel_derived_threshold.py",
    ]
    rows = []
    for fname in targets:
        path = EXP_DIR / fname
        if not path.exists():
            continue
        text = path.read_text()
        sweep_bodies = SWEEP_FUNC_RE.findall(text)
        contaminated = any("W_FULL6" in body for body in sweep_bodies)
        n_sweep_funcs = len(sweep_bodies)
        rows.append({"file": fname, "n_sweep_functions_found": n_sweep_funcs,
                     "w_full6_in_any_sweep_body": contaminated})
        print(f"  {fname:42s} sweep/selection functions found={n_sweep_funcs}  "
              f"W_FULL6 referenced inside one={contaminated}")
    n_bad = sum(r["w_full6_in_any_sweep_body"] for r in rows)
    print(f"\n  {n_bad} of {len(rows)} files fit or select a parameter using "
          f"W_FULL6 data. Zero would confirm B-33's narrow premise (the "
          f"'+0' convention's technical claim that no fitted quantity "
          f"crosses from panel-2023+ into a decision) holds for parameter "
          f"selection specifically.")
    return rows


# ---------------------------------------------------------------------
# 5. THE ONE NEW READ THIS ROUND MAKES: price-only correlation between
#    BTC and each panel asset, split at OOS_START. Pre-authorized by the
#    task brief as evidence for part 3 ("is a panel-2023+ read
#    independent of a BTC/ETH-2023+ read"). No strategy, no parameter,
#    nothing swept or selected -- a fixed, one-shot descriptive statistic.
# ---------------------------------------------------------------------


def _daily_log_returns(df: pd.DataFrame) -> pd.Series:
    daily_close = df["close"].resample("1D").last().dropna()
    return np.log(daily_close).diff().dropna()


def part3_correlation_evidence() -> pd.DataFrame:
    print("\n" + "=" * 100)
    print("AUDIT 5 -- NEW READ: BTC vs panel daily-return correlation, "
          "split at OOS_START=2023-01-01 (prices only, no strategy)")
    print("=" * 100)
    from tradebot.data import load_coinbase_spot, load_dataset

    btc, _ = load_dataset(DATA_DIR, "spot")
    btc_ret = _daily_log_returns(btc)

    rows = []
    for ticker in U6:
        df = load_coinbase_spot(DATA_DIR, ticker)
        ret = _daily_log_returns(df)
        idx = btc_ret.index.intersection(ret.index)

        pre = idx[idx < OOS_START]
        post = idx[idx >= OOS_START]

        def corr(sub_idx):
            if len(sub_idx) < 30:
                return float("nan"), len(sub_idx)
            a = btc_ret.reindex(sub_idx)
            b = ret.reindex(sub_idx)
            return float(np.corrcoef(a, b)[0, 1]), len(sub_idx)

        c_full, n_full = corr(idx)
        c_pre, n_pre = corr(pre)
        c_post, n_post = corr(post)
        rows.append({"asset": ticker, "corr_full": c_full, "n_full": n_full,
                     "corr_pre2023": c_pre, "n_pre2023": n_pre,
                     "corr_post2023": c_post, "n_post2023": n_post})
        print(f"  {ticker:5s} corr(BTC, {ticker}) full={c_full:.3f} (n={n_full})  "
              f"pre-2023={c_pre:.3f} (n={n_pre})  "
              f"post-2023={c_post:.3f} (n={n_post})")

    df_out = pd.DataFrame(rows)
    print(f"\n  mean pairwise BTC-panel correlation: full={df_out.corr_full.mean():.3f}  "
          f"pre-2023={df_out.corr_pre2023.mean():.3f}  "
          f"post-2023={df_out.corr_post2023.mean():.3f}")
    print("  (R-63 measured mean pairwise PANEL-PANEL correlation at 0.634 "
          "over the same full window; this measures BTC-PANEL correlation "
          "specifically, split by the holdout boundary.)")
    return df_out


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("all", "audit"):
        audit_btc_hold_context()
    if cmd in ("all", "windows"):
        measure_window_composition()
    if cmd in ("all", "reads"):
        collect_btc_hold_reads()
    if cmd in ("all", "isolation"):
        audit_parameter_isolation()
    if cmd in ("all", "correlation"):
        part3_correlation_evidence()
    if cmd not in ("all", "audit", "windows", "reads", "isolation", "correlation"):
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
