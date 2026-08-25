#!/usr/bin/env python
"""R-136 CONSERVATIVE branch: substitute kelly_regime_v4's fast (8-day EWM)
volatility estimator with `r136_shared.har_rv_vol` -- the a-priori
equal-weight mean of daily/weekly/monthly realized-vol components (Corsi
2009 HAR structure), pure realized-vol, no new data source -- via
`r136_shared.HARVolMixin`, and run the full pre-registered gate battery
(causal truncation, QLIKE descriptive check, B1/B3/B4/B5/B6, bug hunt)
against frozen `kelly_regime_v4`.

See `experiments/r136_shared.py`'s module docstring for the full
pre-registration this file implements verbatim: direction, literature,
mechanism, named failure modes, the falsification test, and the exact
decision rule applied at the bottom of this file's `main()`. Nothing here
re-derives or restates that reasoning; this file only runs the gates it
specifies and reports the numbers.

No bar at or after `OOS_START = 2023-01-01` is read anywhere in this file
(`_assert_no_holdout` is called on every frame this file itself loads, and
every helper reused from `r136_shared` already enforces it on load).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

from experiments.r136_shared import (  # noqa: E402
    B1_PERIODS,
    B3_WINDOW_SETS_DAYS,
    FUTURES,
    FUTURES_HIGH_FEE,
    HARVolMixin,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SPOT,
    SPOT_HIGH_FEE,
    _assert_no_holdout,
    compute_rv_components,
    exposure_by_vol_quartile,
    har_rv_vol,
    load_btc_train,
    load_eth_train,
    log_growth_diff,
    qlike_loss,
    run_baseline,
    run_strategy,
    sharpe_diff,
)

N_BACKTESTS = 0  # incremented by every call to run_baseline/run_strategy below


# ---------------------------------------------------------------------------
# Candidate: HARVolMixin's prepare() (KellyRegimeV3.prepare() copied verbatim
# except the `vol` line) + har_rv_vol as _vol_series, on kelly_regime_v4's own
# (20/40/80-day) anchor ladder. Not @register'd -- experiments/-only.
# ---------------------------------------------------------------------------

class ConservativeHARScale(HARVolMixin, KellyRegimeV4):
    """kelly_regime_v4 with its fast EWM(8-day) vol estimator replaced by the
    a-priori equal-weight HAR (daily/weekly/monthly) realized-vol mean."""

    name = "r136_conservative_har_rv_scale"

    def _vol_series(self, df: pd.DataFrame, r: pd.Series) -> np.ndarray:
        return har_rv_vol(df).to_numpy()


def make_har_variant(windows_days: tuple[int, ...]):
    """A ConservativeHARScale-alike bound to a non-default HAR window set,
    for the B3 plateau sweep. A closure rather than a constructor argument
    on ConservativeHARScale itself so the pre-registered primary candidate's
    own class stays argument-free and unambiguous."""

    class _Variant(HARVolMixin, KellyRegimeV4):
        name = f"r136_conservative_har_rv_scale_w{windows_days}"

        def _vol_series(self, df: pd.DataFrame, r: pd.Series) -> np.ndarray:
            return har_rv_vol(df, windows_days=windows_days).to_numpy()

    return _Variant()


# ---------------------------------------------------------------------------
# 1. Causal truncation probe (mandatory scaffolding).
# ---------------------------------------------------------------------------

def gate_1_truncation_probe(df: pd.DataFrame, label: str) -> bool:
    global N_BACKTESTS
    print("=" * 78)
    print("GATE 1: causal truncation probe (own class instance)")
    print("=" * 78)
    m_full, _ = run_strategy(ConservativeHARScale(), df, SPOT,
                              INNER_TRAIN_START, INNER_TRAIN_END, label)
    N_BACKTESTS += 1
    df_trunc = df.loc[:INNER_VAL_END]
    m_trunc, _ = run_strategy(ConservativeHARScale(), df_trunc, SPOT,
                               INNER_TRAIN_START, INNER_TRAIN_END, label)
    N_BACKTESTS += 1
    ok = np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9)
    print(f"  full-frame final_balance:      {m_full.final_balance:.6f}")
    print(f"  truncated-frame final_balance: {m_trunc.final_balance:.6f}")
    print(f"  PASS={ok}")
    return ok


# ---------------------------------------------------------------------------
# 2. QLIKE descriptive check.
# ---------------------------------------------------------------------------

def ewm8_vol(df: pd.DataFrame) -> pd.Series:
    """kelly_regime_v4's own fast vol estimator (EWM(8-day) of 5m log
    returns, annualized, `.shift(1)`), reused verbatim as the QLIKE baseline
    forecast and as the B6 quartile-split series for the baseline."""
    close = df["close"]
    r = np.log(close).diff()
    vol_span = 8 * BARS_PER_DAY
    return (r.ewm(span=vol_span, min_periods=BARS_PER_DAY).std()
            * np.sqrt(BARS_PER_YEAR)).shift(1)


def next_day_realized_vol(df: pd.DataFrame) -> pd.Series:
    """Ground truth for the QLIKE check: each bar's forward-looking realized
    vol over the NEXT trading day (1-day-ahead), annualized -- the daily
    component of `compute_rv_components` shifted back by one day's bars so
    bar i holds the RV of (i, i+1day] rather than [i-1day, i)."""
    close = df["close"]
    r = np.log(close).diff()
    w = BARS_PER_DAY
    fwd_rv = np.sqrt(r.pow(2).rolling(w, min_periods=w).mean() * BARS_PER_YEAR)
    return fwd_rv.shift(-w)  # bar i now holds RV realized strictly AFTER i


def gate_2_qlike(df: pd.DataFrame) -> dict:
    print("=" * 78)
    print("GATE 2: QLIKE descriptive check (Patton 2011) -- har_rv_vol vs "
          "v4's own EWM(8-day) vol, one-day-ahead, ground truth = next-day RV")
    print("=" * 78)

    har = har_rv_vol(df)
    ewm = ewm8_vol(df)
    fwd = next_day_realized_vol(df)

    out = {}
    for split_name, start, end in (
        ("inner_train", INNER_TRAIN_START, INNER_TRAIN_END),
        ("inner_val", INNER_VAL_START, INNER_VAL_END),
    ):
        sl = slice(start, end)
        h, e, f = har.loc[sl], ewm.loc[sl], fwd.loc[sl]
        mask = np.isfinite(h.to_numpy()) & np.isfinite(e.to_numpy()) & np.isfinite(f.to_numpy()) \
            & (h.to_numpy() > 0) & (e.to_numpy() > 0) & (f.to_numpy() > 0)
        h_v, e_v, f_v = h.to_numpy()[mask], e.to_numpy()[mask], f.to_numpy()[mask]
        q_har = qlike_loss(h_v, f_v)
        q_ewm = qlike_loss(e_v, f_v)
        better = "har_rv_vol BETTER (lower QLIKE)" if q_har < q_ewm else "har_rv_vol WORSE (higher QLIKE)"
        print(f"  [{split_name}] n={mask.sum():,}  QLIKE(har_rv_vol)={q_har:.6f}  "
              f"QLIKE(ewm8)={q_ewm:.6f}  -> {better}")
        out[split_name] = dict(n=int(mask.sum()), qlike_har=q_har, qlike_ewm=q_ewm,
                                har_is_better=bool(q_har < q_ewm))
    return out


# ---------------------------------------------------------------------------
# 3. B1: both markets x three periods, candidate vs baseline.
# ---------------------------------------------------------------------------

def gate_3_b1(df: pd.DataFrame, label: str) -> dict:
    global N_BACKTESTS
    print("=" * 78)
    print("GATE 3 (B1): candidate vs baseline, both markets x B1_PERIODS")
    print("=" * 78)

    out = {}
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        for per_name, start, end in B1_PERIODS:
            m_base, res_base = run_baseline(df, mkt, start, end, label)
            N_BACKTESTS += 1
            m_cand, res_cand = run_strategy(ConservativeHARScale(), df, mkt, start, end, label)
            N_BACKTESTS += 1

            lg = log_growth_diff(res_cand, res_base)
            sh = sharpe_diff(res_cand, res_base)

            print(f"\n  [{mkt_name}/{per_name}]")
            print(f"    baseline : final={m_base.final_balance:>12,.1f}  sharpe={m_base.sharpe:6.3f}  "
                  f"trades={m_base.num_trades:5d}  dd={m_base.max_drawdown_pct:5.1f}%")
            print(f"    candidate: final={m_cand.final_balance:>12,.1f}  sharpe={m_cand.sharpe:6.3f}  "
                  f"trades={m_cand.num_trades:5d}  dd={m_cand.max_drawdown_pct:5.1f}%")
            print(f"    d_log_return (cand-base): {lg.diff}  p(>0)={lg.p_positive:.3f}")
            print(f"    d_sharpe     (cand-base): {sh.diff}  p(>0)={sh.p_positive:.3f}")

            out[(mkt_name, per_name)] = dict(
                base=m_base, cand=m_cand, res_base=res_base, res_cand=res_cand,
                log_growth_diff=lg, sharpe_diff=sh,
            )
    return out


# ---------------------------------------------------------------------------
# 4. B3: plateau sweep across HAR window-neighbourhood grid.
# ---------------------------------------------------------------------------

def gate_4_b3(df: pd.DataFrame, label: str) -> dict:
    global N_BACKTESTS
    print("=" * 78)
    print("GATE 4 (B3): HAR window-neighbourhood plateau sweep, SPOT, full_inner")
    print("=" * 78)

    per_name, start, end = [p for p in B1_PERIODS if p[0] == "full_inner"][0]
    m_base, res_base = run_baseline(df, SPOT, start, end, label)
    N_BACKTESTS += 1

    frozen = B3_WINDOW_SETS_DAYS[0]
    assert frozen == (1, 5, 22), f"expected frozen candidate (1,5,22) first, got {frozen}"

    results = {}
    frozen_sign = None
    for windows in B3_WINDOW_SETS_DAYS:
        variant = make_har_variant(windows)
        m_cand, res_cand = run_strategy(variant, df, SPOT, start, end, label)
        N_BACKTESTS += 1
        lg = log_growth_diff(res_cand, res_base)
        sign = 1 if lg.diff.point > 0 else (-1 if lg.diff.point < 0 else 0)
        if windows == frozen:
            frozen_sign = sign
        results[windows] = dict(final=m_cand.final_balance, sharpe=m_cand.sharpe,
                                 d_log_return=lg.diff.point, sign=sign)
        print(f"  windows_days={windows!s:>14}  final={m_cand.final_balance:>12,.1f}  "
              f"sharpe={m_cand.sharpe:6.3f}  d_log_return={lg.diff.point:+.4f}  sign={sign:+d}")

    n_agree = sum(1 for w, r in results.items() if r["sign"] == frozen_sign)
    majority = n_agree >= 3  # 3 of 5 = majority
    print(f"\n  frozen (1,5,22) sign={frozen_sign:+d}; {n_agree}/5 variants agree in sign; "
          f"majority={majority}")
    return dict(results=results, frozen_sign=frozen_sign, n_agree=n_agree, majority=majority)


# ---------------------------------------------------------------------------
# 5. B4: ETH falsification, SPOT, inner_val.
# ---------------------------------------------------------------------------

def gate_5_b4() -> dict:
    global N_BACKTESTS
    print("=" * 78)
    print("GATE 5 (B4): ETH falsification, SPOT, inner_val")
    print("=" * 78)

    eth = load_eth_train()
    _assert_no_holdout(eth)

    m_base, res_base = run_baseline(eth, SPOT, INNER_VAL_START, INNER_VAL_END, "ETH coinbase spot")
    N_BACKTESTS += 1
    m_cand, res_cand = run_strategy(ConservativeHARScale(), eth, SPOT,
                                     INNER_VAL_START, INNER_VAL_END, "ETH coinbase spot")
    N_BACKTESTS += 1

    lg = log_growth_diff(res_cand, res_base)
    sh = sharpe_diff(res_cand, res_base)

    print(f"  baseline : final={m_base.final_balance:>12,.1f}  sharpe={m_base.sharpe:6.3f}")
    print(f"  candidate: final={m_cand.final_balance:>12,.1f}  sharpe={m_cand.sharpe:6.3f}")
    print(f"  d_log_return: {lg.diff}  d_sharpe: {sh.diff}")

    # BTC sign for the SAME period/market, for the replication comparison.
    btc_df, btc_label = load_btc_train("spot")
    m_base_btc, res_base_btc = run_baseline(btc_df, SPOT, INNER_VAL_START, INNER_VAL_END, btc_label)
    N_BACKTESTS += 1
    m_cand_btc, res_cand_btc = run_strategy(ConservativeHARScale(), btc_df, SPOT,
                                             INNER_VAL_START, INNER_VAL_END, btc_label)
    N_BACKTESTS += 1
    lg_btc = log_growth_diff(res_cand_btc, res_base_btc)

    btc_sign = 1 if lg_btc.diff.point > 0 else (-1 if lg_btc.diff.point < 0 else 0)
    eth_sign = 1 if lg.diff.point > 0 else (-1 if lg.diff.point < 0 else 0)
    replicates = (btc_sign == eth_sign) and (btc_sign != 0)
    print(f"\n  BTC (spot/inner_val) d_log_return sign={btc_sign:+d}  "
          f"ETH sign={eth_sign:+d}  REPLICATES={replicates}")

    return dict(base=m_base, cand=m_cand, log_growth_diff=lg, sharpe_diff=sh,
                btc_sign=btc_sign, eth_sign=eth_sign, replicates=replicates)


# ---------------------------------------------------------------------------
# 6. B5: fee tier survival, full_inner, high-fee SPOT/FUTURES.
# ---------------------------------------------------------------------------

def gate_6_b5(df: pd.DataFrame, label: str, b1_results: dict) -> dict:
    global N_BACKTESTS
    print("=" * 78)
    print("GATE 6 (B5): 0.40% taker fee tier, full_inner")
    print("=" * 78)

    per_name, start, end = [p for p in B1_PERIODS if p[0] == "full_inner"][0]
    out = {}
    for mkt_name, mkt in (("spot_high_fee", SPOT_HIGH_FEE), ("futures_5x_high_fee", FUTURES_HIGH_FEE)):
        m_base, res_base = run_baseline(df, mkt, start, end, label)
        N_BACKTESTS += 1
        m_cand, res_cand = run_strategy(ConservativeHARScale(), df, mkt, start, end, label)
        N_BACKTESTS += 1
        lg = log_growth_diff(res_cand, res_base)
        sign = 1 if lg.diff.point > 0 else (-1 if lg.diff.point < 0 else 0)

        base_mkt_key = "spot" if "spot" in mkt_name else "futures_5x"
        base_sign_std_fee = None
        cell = b1_results.get((base_mkt_key, "full_inner"))
        if cell is not None:
            base_sign_std_fee = 1 if cell["log_growth_diff"].diff.point > 0 else (
                -1 if cell["log_growth_diff"].diff.point < 0 else 0)

        flips = (base_sign_std_fee is not None) and (sign != base_sign_std_fee)
        print(f"\n  [{mkt_name}] baseline final={m_base.final_balance:>12,.1f}  "
              f"candidate final={m_cand.final_balance:>12,.1f}")
        print(f"    d_log_return: {lg.diff}  sign={sign:+d}  "
              f"(std-fee full_inner sign was {base_sign_std_fee}) sign_flip={flips}")
        out[mkt_name] = dict(base=m_base, cand=m_cand, log_growth_diff=lg,
                              sign=sign, sign_flip=flips)
    survives = not any(v["sign_flip"] for v in out.values())
    print(f"\n  survives 0.40% fee tier (no sign flip vs std-fee full_inner): {survives}")
    return dict(cells=out, survives=survives)


# ---------------------------------------------------------------------------
# 7. B6 (mandatory risk-match): time-in-market, realized vol, exposure by
#    vol quartile, on every B1 cell.
# ---------------------------------------------------------------------------

def realized_vol_of_equity(res) -> float:
    """Annualized realized vol of the strategy's OWN equity curve (bar-freq
    simple returns), the risk-match companion to `time_in_market_pct` -- a
    higher-return candidate that also runs hotter vol is not a free lunch."""
    eq = res.equity.to_numpy(dtype=float)
    if len(eq) < 3:
        return float("nan")
    rets = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], np.nan)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return float("nan")
    return float(np.std(rets, ddof=1) * np.sqrt(BARS_PER_YEAR))


def gate_7_b6(df: pd.DataFrame, label: str, b1_results: dict) -> dict:
    print("=" * 78)
    print("GATE 7 (B6, MANDATORY): time-in-market / realized vol on every B1 cell, "
          "+ exposure-by-vol-quartile on full_inner both markets")
    print("=" * 78)

    risk_rows = {}
    for (mkt_name, per_name), cell in b1_results.items():
        m_base, m_cand = cell["base"], cell["cand"]
        res_base, res_cand = cell["res_base"], cell["res_cand"]
        tim_base, tim_cand = m_base.time_in_market_pct, m_cand.time_in_market_pct
        rv_base, rv_cand = realized_vol_of_equity(res_base), realized_vol_of_equity(res_cand)
        print(f"\n  [{mkt_name}/{per_name}]")
        print(f"    time_in_market_pct: baseline={tim_base:6.2f}%  candidate={tim_cand:6.2f}%  "
              f"(delta={tim_cand - tim_base:+.2f}pp)")
        print(f"    realized_vol(equity, annualized): baseline={rv_base:.4f}  candidate={rv_cand:.4f}  "
              f"(delta={rv_cand - rv_base:+.4f})")
        risk_rows[(mkt_name, per_name)] = dict(
            tim_base=tim_base, tim_cand=tim_cand,
            rv_base=rv_base, rv_cand=rv_cand,
        )

    print("\n  --- exposure by vol quartile (full_inner, both markets) ---")
    quartiles = {}
    per_name, start, end = [p for p in B1_PERIODS if p[0] == "full_inner"][0]
    for mkt_name in ("spot", "futures_5x"):
        cell = b1_results[(mkt_name, "full_inner")]
        res_base, res_cand = cell["res_base"], cell["res_cand"]

        # candidate's own vol series (har_rv_vol), on the same (trimmed) frame
        # the candidate's result carries -- res_cand.df is the post-warmup,
        # in-period frame run_period() returns.
        cand_vol = har_rv_vol(res_cand.df).to_numpy()
        cand_target = res_cand.df["target"].to_numpy()
        q_cand = exposure_by_vol_quartile(cand_target, cand_vol)

        base_vol = ewm8_vol(res_base.df).to_numpy()
        base_target = res_base.df["target"].to_numpy()
        q_base = exposure_by_vol_quartile(base_target, base_vol)

        print(f"\n  [{mkt_name}/full_inner] mean |exposure| by vol quartile (q1=lowest vol .. q4=highest)")
        print(f"    baseline (EWM8 vol) : q1={q_base['q1']:.3f} q2={q_base['q2']:.3f} "
              f"q3={q_base['q3']:.3f} q4={q_base['q4']:.3f}")
        print(f"    candidate (har_rv_vol): q1={q_cand['q1']:.3f} q2={q_cand['q2']:.3f} "
              f"q3={q_cand['q3']:.3f} q4={q_cand['q4']:.3f}")
        q4_delta = q_cand["q4"] - q_base["q4"]
        print(f"    q4 (highest-vol) delta (candidate - baseline): {q4_delta:+.3f} "
              f"({'candidate holds MORE in high-vol' if q4_delta > 0 else 'candidate holds LESS in high-vol'})")
        quartiles[mkt_name] = dict(base=q_base, cand=q_cand, q4_delta=q4_delta)

    return dict(risk_rows=risk_rows, quartiles=quartiles)


# ---------------------------------------------------------------------------
# 8. Bug hunt.
# ---------------------------------------------------------------------------

def gate_8_bug_hunt(df: pd.DataFrame) -> dict:
    print("=" * 78)
    print("GATE 8: bug hunt")
    print("=" * 78)

    import inspect

    from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3

    src_v3 = inspect.getsource(KellyRegimeV3.prepare)
    src_mixin = inspect.getsource(HARVolMixin.prepare)

    v3_lines = [ln.strip() for ln in src_v3.splitlines() if ln.strip()]
    mixin_lines = [ln.strip() for ln in src_mixin.splitlines() if ln.strip()]

    # Strip the two lines each version legitimately differs on (the `def`
    # line itself, the vol-computation line(s), and the trailing
    # `df = df.copy()` HARVolMixin adds before assignment) and diff the rest.
    def _strip_vol_and_def(lines: list[str]) -> list[str]:
        out = []
        skip_markers = ("def prepare", "vol = (r.ewm", "vol = np.asarray", "df = df.copy()")
        for ln in lines:
            if any(ln.startswith(m) for m in skip_markers):
                continue
            out.append(ln)
        return out

    v3_rest = _strip_vol_and_def(v3_lines)
    mixin_rest = _strip_vol_and_def(mixin_lines)
    identical_except_vol_line = (v3_rest == mixin_rest)

    print(f"  HARVolMixin.prepare() vs KellyRegimeV3.prepare(): identical apart from the "
          f"`vol` computation line and the added `df = df.copy()`: {identical_except_vol_line}")
    if not identical_except_vol_line:
        print("  *** DIFF FOUND -- lines present in one but not the other: ***")
        set_v3, set_mixin = set(v3_rest), set(mixin_rest)
        for ln in set_v3 - set_mixin:
            print(f"    only in KellyRegimeV3.prepare(): {ln}")
        for ln in set_mixin - set_v3:
            print(f"    only in HARVolMixin.prepare(): {ln}")

    # har_rv_vol's .shift(1): does bar i see its own or a future bar's return?
    sub = df.iloc[:5000].copy()
    close = sub["close"]
    r = np.log(close).diff()
    comp = compute_rv_components(sub, windows_days=(1, 5, 22))
    # Recompute the SAME components WITHOUT the shift, to see what bar i's
    # "raw" (unshifted) window would have included, then confirm the shifted
    # column at bar i equals the unshifted column at bar i-1 exactly (i.e.
    # the shift is a genuine one-bar lag, not a no-op or a lookahead).
    w1 = int(1 * BARS_PER_DAY)
    raw_1d = np.sqrt(r.pow(2).rolling(w1, min_periods=BARS_PER_DAY).mean() * BARS_PER_YEAR)
    shifted_matches_lagged_raw = np.allclose(
        comp["rv_1d"].to_numpy()[w1 + 5:], raw_1d.to_numpy()[w1 + 4: -1],
        equal_nan=True,
    )
    # Directly confirm bar i's shifted value does not depend on r[i]: perturb
    # r at a single interior bar and check only bars > that index change.
    probe_i = 3000
    sub2 = sub.copy()
    sub2.iloc[probe_i, sub2.columns.get_loc("close")] *= 1.10  # shock close[probe_i]
    comp2 = compute_rv_components(sub2, windows_days=(1, 5, 22))
    diffs = (comp2["rv_1d"] - comp["rv_1d"]).to_numpy()
    changed = np.flatnonzero(np.abs(diffs) > 1e-12)
    no_lookahead = len(changed) == 0 or changed.min() > probe_i
    # also confirms bar probe_i itself is untouched (no same-bar leakage)
    same_bar_untouched = not np.isfinite(diffs[probe_i]) or abs(diffs[probe_i]) < 1e-12

    print(f"  har_rv_vol .shift(1) lag check (rv_1d component, unshifted-vs-shifted "
          f"alignment): {'OK' if shifted_matches_lagged_raw else 'MISMATCH'}")
    print(f"  perturbation probe: shocked close[{probe_i}] by 10%; earliest affected "
          f"bar in rv_1d = {int(changed.min()) if len(changed) else 'none'} "
          f"(must be > {probe_i} for no lookahead, and bar {probe_i} itself must be untouched)")
    print(f"    no bar <= {probe_i} affected (no lookahead): {no_lookahead}")
    print(f"    bar {probe_i} itself untouched (no same-bar leakage): {same_bar_untouched}")

    outcome = identical_except_vol_line and no_lookahead and same_bar_untouched
    print(f"\n  BUG HUNT OUTCOME: {'CLEAN' if outcome else 'ISSUE FOUND'}")
    return dict(identical_except_vol_line=identical_except_vol_line,
                no_lookahead=no_lookahead, same_bar_untouched=same_bar_untouched,
                clean=outcome)


# ---------------------------------------------------------------------------
# Decision rule (verbatim from r136_shared.py's module docstring).
# ---------------------------------------------------------------------------

def apply_decision_rule(truncation_ok: bool, b1: dict, b3: dict, b4: dict,
                         b5: dict, b6: dict) -> dict:
    print("=" * 78)
    print("PRE-REGISTERED DECISION RULE (verbatim, r136_shared.py)")
    print("=" * 78)

    clause_a = truncation_ok

    # (b) both markets, full_inner AND inner_val both show d_log_return point
    # estimate > 0 vs frozen v4, with >=1 of the 2 periods' 95% CI excluding
    # zero on >=1 market.
    both_periods_positive = True
    any_ci_excludes_zero = False
    for mkt_name in ("spot", "futures_5x"):
        for per_name in ("full_inner", "inner_val"):
            cell = b1[(mkt_name, per_name)]
            pt = cell["log_growth_diff"].diff.point
            if pt <= 0:
                both_periods_positive = False
            if cell["log_growth_diff"].significant:
                any_ci_excludes_zero = True
    clause_b = both_periods_positive and any_ci_excludes_zero

    clause_c = b3["majority"]
    clause_d = b4["replicates"]
    clause_e = b5["survives"]
    clause_f = True  # B6 is a mandatory REPORT, not a pass/fail threshold; satisfied by reporting above

    print(f"  (a) causal-truncation probe passes:                    {clause_a}")
    print(f"  (b) B1 both markets, both periods d_log_return>0,")
    print(f"      >=1 period's CI excludes zero on >=1 market:       {clause_b}")
    print(f"      [both_periods_positive={both_periods_positive}, "
          f"any_ci_excludes_zero={any_ci_excludes_zero}]")
    print(f"  (c) B3 plateau (majority agrees in sign):              {clause_c}")
    print(f"  (d) B4 ETH sign replication:                           {clause_d}")
    print(f"  (e) B5 no sign flip at 0.40% fee tier:                 {clause_e}")
    print(f"  (f) B6 mandatory risk-match reported:                  {clause_f}")

    promote = all([clause_a, clause_b, clause_c, clause_d, clause_e, clause_f])
    verdict = "PROMOTE-candidate (as kelly_regime_v5, pending operator holdout consultation)" \
        if promote else "NEGATIVE"
    print(f"\n  VERDICT: {verdict}")
    return dict(clause_a=clause_a, clause_b=clause_b, clause_c=clause_c,
                clause_d=clause_d, clause_e=clause_e, clause_f=clause_f,
                promote=promote, verdict=verdict)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    global N_BACKTESTS
    t0 = time.time()

    df, label = load_btc_train("spot")
    _assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  (< {OOS_START})\n")

    truncation_ok = gate_1_truncation_probe(df, label)
    if not truncation_ok:
        print("\n*** CAUSAL TRUNCATION PROBE FAILED. STOPPING. Do not trust any "
              "number below -- this is the headline finding. ***", file=sys.stderr)
        return

    qlike = gate_2_qlike(df)
    b1 = gate_3_b1(df, label)
    b3 = gate_4_b3(df, label)
    b4 = gate_5_b4()
    b5 = gate_6_b5(df, label, b1)
    b6 = gate_7_b6(df, label, b1)
    bug = gate_8_bug_hunt(df)

    verdict = apply_decision_rule(truncation_ok, b1, b3, b4, b5, b6)

    print("\n" + "=" * 78)
    print(f"Total backtests run (baseline + candidate, across gates 1/3/4/5/6): {N_BACKTESTS}")
    print(f"Bug hunt outcome: {'CLEAN' if bug['clean'] else 'ISSUE FOUND'}")
    print(f"FINAL VERDICT: {verdict['verdict']}")
    print(f"Total wall time: {time.time() - t0:.0f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
