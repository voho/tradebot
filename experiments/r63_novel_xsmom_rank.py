"""R-63 NOVEL branch -- cross-sectional relative strength over the panel.

The decision this arm makes is *which asset is strongest right now*, not
*is this asset trending*. That is relative information: it does not exist
in any single price series, and this repo has never used it. Everything
else -- the signal's own shape and the sizer -- is `kelly_regime_v4`'s,
untouched, so the round tests the CROSS-SECTION and not a new indicator.

Construction (frozen before any number was read; one free parameter, `k`):

  1. SCORE. For asset i,
         score_i(t) = mean over h in (20, 40, 80) days of
                      close_i(t) / anchor_{i,h}(t) - 1
     where `anchor_{i,h} = close_i.rolling(h * 288).mean()` -- the same
     anchor `kelly_regime_v4` votes on. This is v4's own vote made
     continuous, deliberately: the round must not also be testing a newly
     invented signal.

  2. SELECTION. Rank assets by `score` at each bar close; hold the top `k`
     assets THAT HAVE A POSITIVE SCORE, equal-weighted. Fewer than `k`
     positive -> hold only those. None positive -> flat. That is a rule,
     not a parameter, and it is what stops the arm being forced long
     through a bear.

  3. SIZING. `kelly_regime_v4`'s own conditional volatility target, at the
     PORTFOLIO level, shipped constants untouched (target_vol=0.55,
     max_leverage=2.0, vol_span=8*288, anchor_span_days=180, high_in=1.70,
     high_out=1.20, low_in=0.55, low_out=0.85, deadband=0.10). The scale
     half of `KellyRegimeV3.prepare()` is copied verbatim below and driven
     by the EQUAL-WEIGHT ALL-N BASKET's log return series -- NOT the top-k
     basket's. Driving it from the top-k basket would make the scale a
     function of the weights the scale itself determines; that circularity
     is a bug, not a design choice.

     v3 latches `desired = frac * scale` through a 0.10 deadband. The
     portfolio analogue of `frac * scale` is the desired TOTAL notional,
     `scale(t) * m(t) / k` with `m` the number of selected assets, so the
     deadband is applied there and the result split equally over the `m`
     holdings. When the latch is not binding this reduces exactly to
     `w_i(t) = scale(t)/k`, as specified. The total is then clipped to
     1.0: long-only spot, unlevered.

WARMUP. The 80-day anchor needs 23,040 bars. Targets are built on an
ALIGNED frame that starts `WARM_DAYS` before the evaluation window (the
cross-section is only a comparison if every asset's anchors span the same
calendar), then both the targets and the price frames are sliced down to
the evaluation window before `simulate_portfolio` sees them.

Windows, universes, costs, decision rules D1-D4, the scramble
falsification and the further-work bar all live in the frozen
pre-registration in `experiments/r63_shared.py`. This file implements a
candidate and measures it; it does not define or relax a rule.

Run as:
    python experiments/r63_novel_xsmom_rank.py checks
    python experiments/r63_novel_xsmom_rank.py sweep
    python experiments/r63_novel_xsmom_rank.py run
    python experiments/r63_novel_xsmom_rank.py scramble
    python experiments/r63_novel_xsmom_rank.py all
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from experiments.r63_shared import (  # noqa: E402
    OUT_DIR,
    SCRAMBLE_SEEDS,
    SPOT_BASE,
    SPOT_REAL,
    UNIVERSE_6,
    UNIVERSE_8,
    W_FULL6,
    W_TRAIN,
    W_VAL,
    align_frames,
    check_against_engine,
    check_causality,
    compare,
    config_count,
    d1_pass,
    d2_pass,
    d3_pass,
    further_work,
    load_universe,
    matched_hold_targets,
    mean_total_notional,
    scramble_targets,
    simulate_portfolio,
    static_hold_equity,
)

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

HORIZONS = (20, 40, 80)
WARM_DAYS = 91  # > 80-day anchor, per the round's warmup instruction

# kelly_regime_v4's shipped constants. Do not tune these -- the round has
# exactly one free parameter and it is `k`.
TARGET_VOL = 0.55
MAX_LEVERAGE = 2.0
VOL_SPAN = 8 * BARS_PER_DAY
ANCHOR_SPAN_DAYS = 180
HIGH_IN, HIGH_OUT = 1.70, 1.20
LOW_IN, LOW_OUT = 0.55, 0.85
DEADBAND = 0.10

K_GRID = (1, 2, 3, 4)

# Frozen after the W_TRAIN sweep and the W_VAL selection, BEFORE the D1
# cell was touched.
#
# Selected on W_VAL ONLY, on the D1 decision statistic (growth_diff vs
# MATCHED_HOLD): k=1 is the best of the four on W_VAL growth (-0.864 vs
# -1.064 / -1.066 / -0.931) AND the best on W_VAL drawdown difference
# (+38.5 vs +46.0 / +47.2 / +45.4), so the selection is unambiguous
# without inventing a tie-break. Note for the record that k=1 is the
# WORST of the four on W_TRAIN (-4.264), which is itself evidence that
# the k axis is noise: every k on every window loses to MATCHED_HOLD by
# a similar large margin. The neighbourhood is a plateau -- a uniformly
# bad one.
K_FROZEN = 1


# ------------------------------------------------------------------ signal


def cross_sectional_score(aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """`score_i(t) = mean_h (close_i(t) / anchor_{i,h}(t) - 1)`.

    Rolling means only: row t uses rows <= t and nothing else. No
    standardization, no ranking over time, no whole-series statistic of
    any kind -- the ranking in :func:`build_targets` is strictly ACROSS
    COLUMNS within one row.
    """
    cols = {}
    for t, df in aligned.items():
        close = df["close"]
        acc = None
        for h in HORIZONS:
            anchor = close.rolling(int(h * BARS_PER_DAY)).mean()
            term = close / anchor - 1.0
            acc = term if acc is None else acc + term
        cols[t] = acc / len(HORIZONS)
    return pd.DataFrame(cols, index=next(iter(aligned.values())).index)


def basket_log_returns(aligned: dict[str, pd.DataFrame]) -> pd.Series:
    """Log return series of the EQUAL-WEIGHT ALL-N basket.

    This, not the top-k basket, is what the portfolio volatility target is
    driven from -- see the module docstring on the circularity.
    """
    acc = None
    for df in aligned.values():
        r = np.log(df["close"]).diff()
        acc = r if acc is None else acc + r
    return acc / len(aligned)


def conditional_vol_scale(r: pd.Series) -> np.ndarray:
    """The scale half of `KellyRegimeV3.prepare()`, copied verbatim.

    Everything below the vote: the 8-day EWM realized vol (shifted one bar,
    as v3 shifts it), its 180-day EWM anchor, the latched high/low
    breakout state machine, and `full` vs `steady` inverse-vol sizing --
    with v4's shipped constants. Returns the raw `scale` array; the 0.10
    deadband is applied by the caller to the desired TOTAL notional, which
    is the portfolio analogue of v3's `frac * scale`.
    """
    vol = (r.ewm(span=VOL_SPAN, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    slow = (pd.Series(vol).ewm(span=ANCHOR_SPAN_DAYS * BARS_PER_DAY,
                               min_periods=BARS_PER_DAY).mean().to_numpy())

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(TARGET_VOL / vol, MAX_LEVERAGE)
        steady = np.minimum(TARGET_VOL / slow, MAX_LEVERAGE)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    n = len(vol)
    scale = np.zeros(n)
    state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > HIGH_IN else (-1 if x < LOW_IN else 0)
            elif state == 1 and x < HIGH_OUT:
                state = 0
            elif state == -1 and x > LOW_OUT:
                state = 0
        scale[i] = full[i] if state != 0 else steady[i]
    return scale


def build_targets(aligned: dict[str, pd.DataFrame], k: int) -> pd.DataFrame:
    """Target weight matrix for the cross-sectional arm, one free param `k`."""
    score = cross_sectional_score(aligned)
    assets = list(score.columns)
    s = score.to_numpy(dtype=float)
    n, n_assets = s.shape

    valid = np.isfinite(s)
    # Rank ACROSS COLUMNS within each row (a cross-section, never a
    # time-series quantile). NaNs sort last and are masked out anyway.
    s_rank = np.where(valid, s, -np.inf)
    order = np.argsort(-s_rank, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order,
                      np.broadcast_to(np.arange(n_assets), (n, n_assets)), axis=1)
    sel = valid & (s > 0.0) & (rank < k)

    m = sel.sum(axis=1)
    scale = conditional_vol_scale(basket_log_returns(aligned))

    desired = scale * (m / float(k))
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        d = desired[i]
        if abs(d - cur) > DEADBAND:
            cur = d
        pos[i] = cur

    total = np.minimum(pos, 1.0)  # long-only spot, unlevered
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(m > 0, total / np.maximum(m, 1), 0.0)
    w = sel * per[:, None]
    return pd.DataFrame(w, index=score.index, columns=assets)


# ------------------------------------------------------------------ cells


def warm_window(window):
    start, end = window
    return (str((pd.Timestamp(start, tz="UTC")
                 - pd.Timedelta(days=WARM_DAYS)).date()), end)


def build_cell(frames, universe, window, k):
    """Aligned prices + targets, both sliced to the evaluation window.

    Returns (aligned_eval, targets_eval, first_bar_warm) where the last is
    True iff every asset has a finite score on the FIRST evaluated bar.
    """
    sub = {t: frames[t] for t in universe}
    warm = align_frames(sub, warm_window(window))
    targets = build_targets(warm, k)

    start = pd.Timestamp(window[0], tz="UTC")
    idx = warm[universe[0]].index
    idx = idx[idx >= start]
    if window[1] is not None:
        hi = pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)
        idx = idx[idx < hi]

    score = cross_sectional_score(warm).loc[idx]
    first_warm = bool(np.isfinite(score.iloc[0].to_numpy()).all())

    aligned_eval = {t: df.loc[idx] for t, df in warm.items()}
    return aligned_eval, targets.loc[idx], first_warm


def measure(cand_eq, bench_eq, targets, label, **extra):
    row = compare(cand_eq, bench_eq)
    row.update(extra)
    row["bench"] = label
    row["mean_total_notional"] = mean_total_notional(targets)
    return row


CELL_FIELDS = [
    "arm", "k", "window", "universe", "market", "fee", "bench",
    "cand_final", "bench_final", "cand_dd", "bench_dd",
    "mean_total_notional", "growth_diff", "growth_lo", "growth_hi",
    "dd_diff", "dd_lo", "dd_hi", "n_days", "n_bars",
    "d1_pass", "d2_pass", "d3_pass", "note",
]


def write_csv(path, fields, rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"  wrote {path}")


def fmt(row):
    return (f"    final {row['cand_final']:>10,.0f} vs {row['bench_final']:>10,.0f}"
            f" | dd {row['cand_dd']:5.1f}% vs {row['bench_dd']:5.1f}%"
            f" | growth {row['growth_diff']:+.3f} [{row['growth_lo']:+.3f},"
            f" {row['growth_hi']:+.3f}]"
            f" | dd_diff {row['dd_diff']:+.2f} [{row['dd_lo']:+.2f},"
            f" {row['dd_hi']:+.2f}]")


# ------------------------------------------------------------------ checks


def perturbation_probe(frames, universe=UNIVERSE_8, k=2, frac_tail=0.4) -> bool:
    """Self-check #4: a whole-series scaler / quantile / mean / std probe.

    Multiply the TAIL of every price series by 10 and rebuild the targets.
    Any statistic computed over the whole series -- a `.rank(pct=True)`, a
    full-sample mean, a StandardScaler -- would move the EARLY rows. A
    strictly causal construction cannot. This is deliberately different
    from `check_causality`'s truncation probe: truncation removes the
    tail, this one corrupts it, and a statistic that ignores series length
    but not series content is only caught by the second.
    """
    warm = align_frames({t: frames[t] for t in universe}, warm_window(W_TRAIN))
    n = len(next(iter(warm.values())))
    cut = int(n * (1.0 - frac_tail))

    bad = {}
    for t, df in warm.items():
        d = df.copy()
        for c in ("open", "high", "low", "close"):
            v = d[c].to_numpy(dtype=float).copy()
            v[cut:] *= 10.0
            d[c] = v
        bad[t] = d

    a = np.nan_to_num(build_targets(warm, k).to_numpy()[:cut], nan=0.0)
    b = np.nan_to_num(build_targets(bad, k).to_numpy()[:cut], nan=0.0)
    return bool(np.allclose(a, b, atol=1e-12, rtol=0.0))


def cmd_checks(frames):
    print("== self-checks ==")
    ok, err = check_against_engine()
    print(f"  check_against_engine: ok={ok} relative_final_balance_error={err:.6f}")

    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    causal = check_causality(lambda a: build_targets(a, 2), warm)
    print(f"  check_causality(k=2): {causal}")
    causal2 = check_causality(lambda a: build_targets(a, 1), warm)
    print(f"  check_causality(k=1): {causal2}")

    probe = perturbation_probe(frames)
    print(f"  perturbation_probe (tail x10, early rows unchanged): {probe}")

    score = cross_sectional_score(warm)
    start = pd.Timestamp(W_TRAIN[0], tz="UTC")
    first = score.loc[score.index >= start].iloc[0]
    print(f"  first evaluated bar {first.name} scores finite for every asset: "
          f"{bool(np.isfinite(first.to_numpy()).all())}")
    print(f"    {dict(first.round(4))}")
    return ok and causal and causal2 and probe


# ------------------------------------------------------------------ sweep


def cmd_sweep(frames):
    print("== k sweep: W_TRAIN then W_VAL, U8, spot 0.10% ==")
    rows = []
    for wname, window in (("W_TRAIN", W_TRAIN), ("W_VAL", W_VAL)):
        ew = None
        for k in K_GRID:
            aligned, targets, warm_ok = build_cell(frames, UNIVERSE_8, window, k)
            if not warm_ok:
                raise RuntimeError(f"{wname} k={k}: first evaluated bar not warm")
            cand = simulate_portfolio(targets, aligned, SPOT_BASE)
            c = mean_total_notional(targets)
            mh = simulate_portfolio(matched_hold_targets(targets.index, UNIVERSE_8, c),
                                    aligned, SPOT_BASE)
            if ew is None:
                ew = static_hold_equity(aligned, UNIVERSE_8, SPOT_BASE)
            row = measure(cand, mh, targets, "MATCHED_HOLD", arm="xsmom_rank", k=k,
                          window=wname, universe="U8", market="spot", fee=0.001,
                          n_bars=len(targets), note="sweep")
            row["ew_final"] = float(ew.iloc[-1])
            row["d1_pass"] = d1_pass(row)
            row["d2_pass"] = d2_pass(row)
            row["d3_pass"] = d3_pass(row)
            row["first_bar_warm"] = warm_ok
            rows.append(row)
            print(f"  {wname} k={k} mtn={c:.3f} ew_final={ew.iloc[-1]:,.0f}")
            print(fmt(row))
            print(f"    growth>0 & excl 0: {row['d1_pass']}  "
                  f"dd<0 & excl 0: {row['d2_pass']}  directional (D3 form): "
                  f"{row['d3_pass']}")

    write_csv(OUT_DIR / "novel_sweep.csv",
              CELL_FIELDS + ["ew_final", "first_bar_warm"], rows)
    return rows


# ------------------------------------------------------------------ run


def cmd_run(frames, k=None):
    k = K_FROZEN if k is None else k
    if k is None:
        raise SystemExit("k is not frozen yet: run `sweep`, set K_FROZEN, or pass --k")
    print(f"== D1/D2/D4 cells: W_FULL6, U6, k={k} (FROZEN) ==")
    rows = []

    aligned, targets, warm_ok = build_cell(frames, UNIVERSE_6, W_FULL6, k)
    if not warm_ok:
        raise RuntimeError("W_FULL6 first evaluated bar not warm")
    print(f"  bars {len(targets):,}  {targets.index[0]} -> {targets.index[-1]}")
    print(f"  first evaluated bar warm for every asset: {warm_ok}")

    c = mean_total_notional(targets)
    cand = simulate_portfolio(targets, aligned, SPOT_BASE)
    mh = simulate_portfolio(matched_hold_targets(targets.index, UNIVERSE_6, c),
                            aligned, SPOT_BASE)
    ew = static_hold_equity(aligned, UNIVERSE_6, SPOT_BASE)

    d1 = measure(cand, mh, targets, "MATCHED_HOLD", arm="xsmom_rank", k=k,
                 window="W_FULL6", universe="U6", market="spot", fee=0.001,
                 n_bars=len(targets), note="D1/D2 primary")
    d1["d1_pass"] = d1_pass(d1)
    d1["d2_pass"] = d2_pass(d1)
    rows.append(d1)
    print(f"  [D1/D2] mtn={c:.3f}")
    print(fmt(d1))
    print(f"    D1 PASS={d1['d1_pass']}   D2 PASS={d1['d2_pass']}")

    ctx_ew = measure(cand, ew, targets, "EW_HOLD", arm="xsmom_rank", k=k,
                     window="W_FULL6", universe="U6", market="spot", fee=0.001,
                     n_bars=len(targets), note="context: vs EW_HOLD")
    rows.append(ctx_ew)
    print("  [context vs EW_HOLD]")
    print(fmt(ctx_ew))

    btc = frames["BTC"]
    btc_on_idx = btc.reindex(btc.index.union(targets.index)).ffill().reindex(targets.index)
    btc_eq = static_hold_equity({"BTC": btc_on_idx}, ["BTC"], SPOT_BASE)
    ctx_btc = measure(cand, btc_eq, targets, "BTC_HOLD", arm="xsmom_rank", k=k,
                      window="W_FULL6", universe="U6", market="spot", fee=0.001,
                      n_bars=len(targets),
                      note="context: vs BTC buy-and-hold (BTC ffilled onto U6 grid)")
    rows.append(ctx_btc)
    print("  [context vs BTC_HOLD]")
    print(fmt(ctx_btc))

    # D4: same cell at 0.40% taker, candidate final balance vs EW_HOLD.
    cand40 = simulate_portfolio(targets, aligned, SPOT_REAL)
    ew40 = static_hold_equity(aligned, UNIVERSE_6, SPOT_REAL)
    d4 = measure(cand40, ew40, targets, "EW_HOLD", arm="xsmom_rank", k=k,
                 window="W_FULL6", universe="U6", market="spot", fee=0.004,
                 n_bars=len(targets), note="D4 cost")
    d4_ok = d4["cand_final"] > d4["bench_final"]
    d4["note"] = f"D4 cost; pass={d4_ok}"
    rows.append(d4)
    print(f"  [D4 @0.40%] cand {d4['cand_final']:,.0f} vs EW_HOLD "
          f"{d4['bench_final']:,.0f} -> D4 PASS={d4_ok}")

    # D3: W_VAL, U8, spot 0.10%, directional gate.
    aligned3, targets3, warm3 = build_cell(frames, UNIVERSE_8, W_VAL, k)
    if not warm3:
        raise RuntimeError("W_VAL first evaluated bar not warm")
    c3 = mean_total_notional(targets3)
    cand3 = simulate_portfolio(targets3, aligned3, SPOT_BASE)
    mh3 = simulate_portfolio(matched_hold_targets(targets3.index, UNIVERSE_8, c3),
                             aligned3, SPOT_BASE)
    d3 = measure(cand3, mh3, targets3, "MATCHED_HOLD", arm="xsmom_rank", k=k,
                 window="W_VAL", universe="U8", market="spot", fee=0.001,
                 n_bars=len(targets3), note="D3 inner-validation")
    d3["d3_pass"] = d3_pass(d3)
    rows.append(d3)
    print(f"  [D3] mtn={c3:.3f}")
    print(fmt(d3))
    print(f"    D3 PASS={d3['d3_pass']}")

    write_csv(OUT_DIR / "novel_cells.csv", CELL_FIELDS, rows)
    return {"d1": d1["d1_pass"], "d2": d1["d2_pass"], "d3": d3["d3_pass"],
            "d4": d4_ok, "d1_row": d1, "targets": targets, "aligned": aligned,
            "mh": mh, "k": k}


# ------------------------------------------------------------------ scramble


def cmd_scramble(frames, k=None, state=None):
    k = K_FROZEN if k is None else k
    if k is None:
        raise SystemExit("k is not frozen yet: run `sweep`, set K_FROZEN, or pass --k")
    print(f"== FALSIFICATION: cross-section scramble, seeds 0..9, D1 cell, k={k} ==")
    if state is None:
        aligned, targets, warm_ok = build_cell(frames, UNIVERSE_6, W_FULL6, k)
        if not warm_ok:
            raise RuntimeError("W_FULL6 first evaluated bar not warm")
        c = mean_total_notional(targets)
        cand = simulate_portfolio(targets, aligned, SPOT_BASE)
        mh = simulate_portfolio(matched_hold_targets(targets.index, UNIVERSE_6, c),
                                aligned, SPOT_BASE)
        real = compare(cand, mh)["growth_diff"]
    else:
        aligned, targets, mh = state["aligned"], state["targets"], state["mh"]
        real = state["d1_row"]["growth_diff"]

    rows = []
    diffs = []
    for seed in SCRAMBLE_SEEDS:
        st = scramble_targets(targets, seed)
        eq = simulate_portfolio(st, aligned, SPOT_BASE)
        r = compare(eq, mh)
        diffs.append(r["growth_diff"])
        rows.append({"arm": "xsmom_rank_scrambled", "k": k, "seed": seed,
                     "window": "W_FULL6", "universe": "U6", "market": "spot",
                     "fee": 0.001, "bench": "MATCHED_HOLD",
                     "mean_total_notional": mean_total_notional(st),
                     **{key: r[key] for key in
                        ("cand_final", "bench_final", "cand_dd", "bench_dd",
                         "growth_diff", "growth_lo", "growth_hi",
                         "dd_diff", "dd_lo", "dd_hi", "n_days")}})
        print(f"  seed {seed}: growth_diff {r['growth_diff']:+.4f}  "
              f"final {r['cand_final']:>10,.0f}  dd {r['cand_dd']:5.1f}%")

    p90 = float(np.percentile(diffs, 90))
    survived = bool(real > p90)
    rows.append({"arm": "xsmom_rank", "k": k, "seed": -1, "window": "W_FULL6",
                 "universe": "U6", "market": "spot", "fee": 0.001,
                 "bench": "MATCHED_HOLD", "growth_diff": real,
                 "mean_total_notional": mean_total_notional(targets),
                 "scramble_p90": p90, "scramble_survived": survived})
    print(f"  real growth_diff {real:+.4f} vs scramble p90 {p90:+.4f} -> "
          f"SURVIVED={survived}")

    write_csv(OUT_DIR / "novel_scramble.csv",
              ["arm", "k", "seed", "window", "universe", "market", "fee", "bench",
               "mean_total_notional", "cand_final", "bench_final", "cand_dd",
               "bench_dd", "growth_diff", "growth_lo", "growth_hi", "dd_diff",
               "dd_lo", "dd_hi", "n_days", "scramble_p90", "scramble_survived"],
              rows)
    return survived


# ------------------------------------------------------------------ diag


def cmd_diag(frames, k=None):
    """Post-hoc diagnostic (changes no decision, but every run is counted).

    The sweep loses by a margin large enough that the negative is worth
    DIAGNOSING rather than merely recording: is the cross-section
    uninformative, or is its information eaten by rank-flip turnover? The
    zero-fee counterfactual separates the two. It is not a decision cell
    and appears in no D-rule; it is written to its own CSV.
    """
    k = K_FROZEN if k is None else k
    print(f"== DIAGNOSTIC (not a decision cell): turnover and zero-fee, k={k} ==")
    from tradebot.broker import MarketSpec
    free = MarketSpec.spot(fee_rate=0.0)

    rows = []
    for wname, window, uni in (("W_TRAIN", W_TRAIN, UNIVERSE_8),
                               ("W_VAL", W_VAL, UNIVERSE_8),
                               ("W_FULL6", W_FULL6, UNIVERSE_6)):
        aligned, targets, _ = build_cell(frames, uni, window, k)
        w = targets.to_numpy()
        dw = np.abs(np.diff(w, axis=0)).sum(axis=1)
        days = (targets.index[-1] - targets.index[0]).total_seconds() / 86400.0
        sel = w > 0
        setchg = int((np.abs(np.diff(sel.astype(int), axis=0)).sum(axis=1) > 0).sum())

        c = mean_total_notional(targets)
        mh0 = simulate_portfolio(matched_hold_targets(targets.index, uni, c),
                                 aligned, free)
        cand0 = simulate_portfolio(targets, aligned, free)
        r0 = compare(cand0, mh0)
        mh1 = simulate_portfolio(matched_hold_targets(targets.index, uni, c),
                                 aligned, SPOT_BASE)
        cand1 = simulate_portfolio(targets, aligned, SPOT_BASE)
        r1 = compare(cand1, mh1)

        row = {
            "arm": "xsmom_rank", "k": k, "window": wname,
            "universe": "U6" if uni is UNIVERSE_6 else "U8",
            "mean_total_notional": c, "days": days,
            "turnover_per_day": float(dw.sum() / days),
            "topk_membership_changes_per_day": setchg / days,
            "implied_fee_drag_logret_per_day_10bps": float(0.001 * dw.sum() / days),
            "growth_diff_at_10bps": r1["growth_diff"],
            "growth_diff_at_0bps": r0["growth_diff"],
            "cand_final_10bps": r1["cand_final"], "cand_final_0bps": r0["cand_final"],
            "bench_final_10bps": r1["bench_final"], "bench_final_0bps": r0["bench_final"],
            "cand_dd_0bps": r0["cand_dd"], "bench_dd_0bps": r0["bench_dd"],
            "growth_lo_0bps": r0["growth_lo"], "growth_hi_0bps": r0["growth_hi"],
        }
        rows.append(row)
        print(f"  {wname}: turnover {row['turnover_per_day']:.2f}x equity/day, "
              f"{row['topk_membership_changes_per_day']:.1f} membership changes/day")
        print(f"    growth_diff @10bps {r1['growth_diff']:+.3f} -> @0bps "
              f"{r0['growth_diff']:+.3f} [{r0['growth_lo']:+.3f}, "
              f"{r0['growth_hi']:+.3f}]  (fee-free final {r0['cand_final']:,.0f} "
              f"vs matched {r0['bench_final']:,.0f})")

    # Fee-free scramble on the D1 cell. The pre-registered falsification
    # runs at 10bps and already failed; this asks the sharper question the
    # zero-fee counterfactual raises -- absent costs, is the RANKING doing
    # anything, or is a fee-free concentrated basket just a fee-free
    # concentrated basket? Diagnostic, post-hoc, and counted.
    aligned, targets, _ = build_cell(frames, UNIVERSE_6, W_FULL6, k)
    c = mean_total_notional(targets)
    mh0 = simulate_portfolio(matched_hold_targets(targets.index, UNIVERSE_6, c),
                             aligned, free)
    real0 = compare(simulate_portfolio(targets, aligned, free), mh0)["growth_diff"]
    sdiffs = []
    for seed in SCRAMBLE_SEEDS:
        eq = simulate_portfolio(scramble_targets(targets, seed), aligned, free)
        sdiffs.append(compare(eq, mh0)["growth_diff"])
    p90 = float(np.percentile(sdiffs, 90))
    print(f"  fee-free scramble on W_FULL6: real {real0:+.3f} vs p90 {p90:+.3f} "
          f"-> survived={real0 > p90}")
    print(f"    seeds: {[round(x, 3) for x in sdiffs]}")
    for r in rows:
        if r["window"] == "W_FULL6":
            r["scramble_free_real"] = real0
            r["scramble_free_p90"] = p90
            r["scramble_free_survived"] = bool(real0 > p90)
            r["scramble_free_seeds"] = ";".join(f"{x:.4f}" for x in sdiffs)

    fields = sorted({key for r in rows for key in r},
                    key=lambda x: (x not in rows[0], x))
    write_csv(OUT_DIR / "novel_diagnostic.csv", fields, rows)
    return rows


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd",
                    choices=["checks", "sweep", "run", "scramble", "diag", "all"])
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    frames = load_universe(UNIVERSE_8)

    if args.cmd == "checks":
        cmd_checks(frames)
    elif args.cmd == "sweep":
        cmd_sweep(frames)
    elif args.cmd == "run":
        cmd_run(frames, args.k)
    elif args.cmd == "scramble":
        cmd_scramble(frames, args.k)
    elif args.cmd == "diag":
        cmd_diag(frames, args.k)
    else:
        cmd_checks(frames)
        cmd_sweep(frames)
        st = cmd_run(frames, args.k)
        surv = cmd_scramble(frames, st["k"], st)
        cmd_diag(frames, st["k"])
        fw = further_work(st["d1"], st["d2"], st["d3"], surv)
        print(f"\n== further_work(d1={st['d1']}, d2={st['d2']}, d3={st['d3']}, "
              f"scramble={surv}) = {fw} ==")
        if fw:
            print("  -> STOP. Report to the operator; the holdout read is theirs.")
        else:
            print("  -> DONE. W_HOLD is NOT read.")

    print(f"\nconfig_count() = {config_count()}")


if __name__ == "__main__":
    main()
