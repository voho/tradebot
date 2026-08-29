#!/usr/bin/env python
"""R-180 NOVEL branch (08-29): sequential "testing by betting" confidence
process (Shafer 2021, JRSS-A 184(2); Waudby-Smith & Ramdas 2024, JRSS-B
86(1)) replacing R-179's fixed-window refit-then-predict meta-label
classifier, on `kelly_regime_v4`'s own `frac*scale` decision.

See `experiments/r180_direction.md` ("Mechanism, novel branch") for the
frozen design, non-duplication case and both falsification clauses; this
file only executes that design and records the resulting numbers. Does NOT
edit `experiments/r180_shared.py`, `experiments/r179_shared.py` or
`experiments/r180_direction.md`, and does not touch the sibling
conservative branch or anything under `src/tradebot/`.

Mechanism, one sentence (frozen, unchanged from the pre-registration):
maintain a single continuously-updating capital process -- `wealth=1` at
the start of whatever period the strategy is run on; at each newly-resolved
daily triple-barrier checkpoint, bet a Kelly fraction of current wealth on
the realized {0,1} outcome using the CURRENT online-logistic estimate of
`P(label=1 | macro_stress_z, mvrv_z)`, updated by ONE stochastic-gradient
step per newly-resolved label (no batch refit, no `refit_days` calendar --
the specific difference from R-179's own novel branch and from this
round's conservative branch); use `log(wealth_t)` directly as a continuous,
anytime-valid confidence score, and multiply v4's own `frac*scale` by
`clip(1 + kappa*tanh(log(wealth_t)/3.0), 0, cap)`, capped at
`max_leverage`, with v4's own 10% deadband applied to the FINAL (multiplied)
desired exposure -- exactly as v4 applies it to its own `desired`.

Falsification clause (this branch, frozen in `r180_direction.md` Step 1
Q4 / Step 2, checked FIRST, before any inner-validation number is read):
the terminal training-period (2017-2020) wealth must exceed `1/alpha` at
`alpha=0.05`, i.e. wealth > 20. If it does not, the branch is NEGATIVE by
construction and the fuller sweep is not required (this is disclosed as
NOT the same failure mode as R-174: the pre-registered n-requirement,
~600-1,670 resolved daily checkpoints, is reachable inside the training
period alone per R-179's own measured yield -- see the pre-registration's
Step 1 Q4 for the derivation).

Evaluation protocol (docs/ROUTINE.md Step 3): train/inner-validation only
-- 2023-01-01 onward (`OOS_START`) is never read by this file.

Usage
-----
    python experiments/r180_novel.py
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

from tradebot.inference import daily_returns, paired_bootstrap, total_log_return  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r180_shared import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    build_checkpoint_data,
    conditional_scale,
    macro_stress_z,
    mvrv_z,
    vote_frac,
)

from scripts.experiment import DF, FUTURES, LABEL, SPOT, ev  # noqa: E402

DATA_DIR = ROOT / "data"
TRAIN_END = "2020-12-31"
VAL_START, VAL_END = "2021-01-01", "2022-12-31"
ALPHA = 0.05
WEALTH_THRESHOLD = 1.0 / ALPHA  # 20.0

N_EVALUATED = 0  # distinct parameter configurations run, for the report


# ============================================================================
# (1) the online "testing by betting" engine -- branch-owned, not shared
#     (r180_shared.py exposes only the feature builders / checkpoint data /
#     labels; the online updater and betting process are this branch's own
#     mechanism, per r180_direction.md).
# ============================================================================


def online_betting_process(cpd: dict, *, lr: float = 0.05, kelly_mult: float = 1.0,
                            l2: float = 1e-3, min_bet_n: int = 20,
                            phat_clip: tuple[float, float] = (0.02, 0.98),
                            lambda_cap: float = 1.9, p0: float = 0.5) -> tuple[np.ndarray, dict]:
    """Single-pass, continuously-updating capital process (Shafer 2021;
    Waudby-Smith & Ramdas 2024). wealth=1 at the start of `cpd`'s own index;
    at each checkpoint whose OWN resolve_bar has occurred (the same
    non-strict "resolve_bar <= bar" convention `r179_shared`'s walk-forward
    classifier already uses), bet a Kelly-style fraction of current wealth
    on the realized {0,1} label using the CURRENT online-logistic estimate
    of P(label=1|x), then take ONE stochastic-gradient step to update that
    estimate for the NEXT checkpoint. No batch refit, no `refit_days`
    window -- the model updates strictly incrementally, and both the
    classifier weights and the feature standardization (Welford's online
    mean/std) use only checkpoints whose OWN resolution has already
    occurred as of the CURRENT event; a global scaler fit over the whole
    series and applied to early rows would itself be lookahead
    (ROUTINE.md's own standing warning), so the scaler is online too.

    Betting rule (frozen in `r180_direction.md`'s resolution-time check):
    null p0=0.5 (features carry no information). For a {0,1} bet against a
    p0=0.5 null, the full-Kelly (log-optimal) fraction is
    `lambda* = argmax_lambda E[log(1+lambda*(X-p0))] = 4*(phat-p0)`
    (maximize the expected log-growth; derivative zero at lambda=4*(p-p0)
    for p0=0.5). `kelly_mult` scales this (1.0 = full Kelly, <1.0 =
    fractional Kelly, the project's own convention for a sizing-safety
    constant) -- this IS the "Kelly-fraction sizing constant" the task
    sweeps. `lambda_cap` (fixed at 1.9, not swept: a fixed safety margin
    inside the +-2 bound that keeps every single-bet wealth multiplier
    `1+lambda*(X-p0)` strictly positive for X in {0,1}) and `phat_clip`
    (fixed, not swept) prevent one badly-calibrated early bet from wiping
    out the whole process. `min_bet_n`: the process bets flat (phat=0.5,
    i.e. lambda=0, no wager) until `min_bet_n` prior labels have resolved
    -- the model itself still trains via SGD from the first resolved label,
    only the BETTING is held back during that short warmup, so the warmup
    itself never contributes fabricated "edge" to the wealth process.

    Returns `(log_wealth_per_bar, diag)`. `log_wealth_per_bar[i]` is
    `log(wealth)` as of bar `i`, using only checkpoints whose
    `resolve_bar <= i` -- forward-filled, 0 (wealth=1, neutral) before the
    first resolved checkpoint. `diag['terminal_wealth']` /
    `diag['terminal_log_wealth']` are this process's own final values over
    whatever slice `cpd` was built on -- the falsification-clause read.
    """
    n = cpd["n"]
    checkpoints = cpd["checkpoints"]
    labels = cpd["labels"]
    cp_features = cpd["cp_features"]
    valid = cpd["valid"]
    resolve_bar = cpd["resolve_bar"]
    n_feat = cp_features.shape[1]

    w = np.zeros(n_feat + 1)
    count = 0
    mean = np.zeros(n_feat)
    m2 = np.zeros(n_feat)

    log_wealth_per_bar = np.zeros(n)
    log_wealth = 0.0
    prev_bar = 0

    # events in resolve-time order -- checkpoints (hence resolve_bar) are
    # already monotonically increasing by construction (daily_checkpoints
    # returns ascending bar positions), so no re-sort is needed; filter to
    # valid, in-range events only.
    order = [j for j in range(len(checkpoints)) if valid[j] and resolve_bar[j] < n]

    n_bet_nontrivial = 0
    log_wealth_path = []
    max_abs_lambda = 0.0

    for j in order:
        rb = int(resolve_bar[j])
        x = cp_features[j]
        y = float(labels[j])

        # -- fill the segment strictly before this event with the wealth
        #    as it stood BEFORE this event's own bet (still causal: no bar
        #    before rb ever reflects a bet that resolves at rb).
        log_wealth_per_bar[prev_bar:rb] = log_wealth

        # -- predict with the CURRENT (pre-update) online model/scaler.
        std = np.sqrt(m2 / count) if count > 0 else np.ones(n_feat)
        std = np.where(std > 1e-9, std, 1.0)
        xs = (x - mean) / std
        z = float(np.clip(w[0] + w[1:] @ xs, -30.0, 30.0))
        phat_model = 1.0 / (1.0 + np.exp(-z))

        # -- bet flat during the short warmup so the process cannot fabricate
        #    edge from an under-trained model; the model still trains below.
        phat_bet = phat_model if count >= min_bet_n else 0.5
        phat_bet = float(np.clip(phat_bet, phat_clip[0], phat_clip[1]))
        if count >= min_bet_n:
            n_bet_nontrivial += 1

        lam = float(np.clip(kelly_mult * 4.0 * (phat_bet - p0), -lambda_cap, lambda_cap))
        max_abs_lambda = max(max_abs_lambda, abs(lam))
        growth = max(1.0 + lam * (y - p0), 1e-9)
        log_wealth += float(np.log(growth))
        log_wealth_path.append(log_wealth)
        prev_bar = rb

        # -- ONE stochastic-gradient step on the online logistic model,
        #    using the SAME (pre-update) standardization used to predict
        #    above -- then update the Welford accumulators for next time.
        grad = y - phat_model
        w[0] += lr * grad
        w[1:] += lr * (grad * xs - l2 * w[1:])

        count += 1
        delta = x - mean
        mean = mean + delta / count
        delta2 = x - mean
        m2 = m2 + delta * delta2

    log_wealth_per_bar[prev_bar:n] = log_wealth

    log_wealth_arr = np.array(log_wealth_path) if log_wealth_path else np.array([0.0])
    first_cross = next((i for i, lw in enumerate(log_wealth_path)
                         if lw > np.log(WEALTH_THRESHOLD)), None)
    diag = dict(
        n_events=len(order), n_bet_nontrivial=n_bet_nontrivial,
        terminal_log_wealth=log_wealth, terminal_wealth=float(np.exp(log_wealth)),
        max_log_wealth=float(log_wealth_arr.max()), min_log_wealth=float(log_wealth_arr.min()),
        max_abs_lambda=max_abs_lambda,
        first_cross_alpha05_event=first_cross,
    )
    return log_wealth_per_bar, diag


# ============================================================================
# (2) the Strategy itself
# ============================================================================


class R180NovelBettingConfidence(Strategy):
    """v4's own `frac*scale`, continuously rescaled by a bounded function of
    a sequential "testing by betting" confidence score (Shafer 2021;
    Waudby-Smith & Ramdas 2024) built from an online logistic updater on
    two exogenous features (`macro_stress_z`, `mvrv_z`), instead of a
    fixed-window walk-forward classifier.

    `frac`/`scale` are v4's own factors (`r180_shared.vote_frac`/
    `conditional_scale`, bit-identical to v4). `vol_daily` is the same
    causal EWM realized-vol array `conditional_scale` computes internally,
    reproduced here (it returns only `scale`/`vol_ratio`). Checkpoints/
    labels/features come from `r180_shared.build_checkpoint_data`
    (unedited). `online_betting_process` above turns those into
    `log_wealth_per_bar`; the final sizing is

        multiplier = clip(1 + kappa*tanh(log_wealth_per_bar / 3.0), 0, cap)
        final_desired = min(frac * scale * multiplier, max_leverage)

    v4's own 10% deadband is then applied sequentially to `final_desired`,
    exactly as v4 applies it to `desired = frac*scale`.

    `warmup` matches v4's own 80 days -- NOT larger. `run_backtest`
    (`engine.py`) uses `strategy.warmup` for two DIFFERENT purposes that
    must be kept small enough to actually satisfy: it caps how much prefix
    `run_period` hands `prepare()` (`min(start_pos, warmup)`), AND,
    separately and independently, it is the bar index before which
    `on_bar` is never even called at all (`i >= strategy.warmup` gates the
    per-bar loop in `run_backtest` directly) -- an oversized warmup
    intended only to widen the first effect silently disables `on_bar` for
    the ENTIRE run (a real bug hit once while building this branch: a
    100,000-day warmup produced a `target` column that clearly varied, a
    `diag_` that clearly showed billions of dollars of training-period
    wealth, and ZERO fills, ZERO trades, flat $1,000 equity everywhere --
    because `i` never reached `strategy.warmup` for any bar in the
    dataset). So this branch uses the SAME 80-day warmup as v4/R-179: the
    training-period read (`start=None` in every Phase-0 call below) still
    gets the full available history back to the dataset's own start
    (`prefix = min(0, warmup) = 0`), but every inner-validation call
    cold-starts the online process with only ~80 days of prior checkpoints
    before `VAL_START` -- the SAME per-call cold-start convention R-179's
    own walk-forward classifier used (this harness re-runs `prepare()`
    from scratch on whatever frame it is given for every backtest; no
    strategy in this project persists state across separate `ev()` calls).

    `diag_` (set by `prepare()`) is `online_betting_process`'s own
    diagnostic dict for the slice this instance was last run on -- the
    falsification clause reads `diag_['terminal_wealth']` directly, not any
    trading number.
    """

    name = "r180_novel_betting_confidence"
    warmup = 80 * BARS_PER_DAY + 10  # matches v4/R-179 -- see docstring for why this must stay small

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70, high_out: float = 1.20,
                 low_in: float = 0.55, low_out: float = 0.85,
                 k: float = 1.0, horizon_days: int = 3, embargo_days: int = 3,
                 lr: float = 0.05, kelly_mult: float = 1.0,
                 kappa: float = 1.0, cap: float = 2.0) -> None:
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        self.k = k
        self.horizon_days = horizon_days
        self.embargo_days = embargo_days
        self.lr = lr
        self.kelly_mult = kelly_mult
        self.kappa = kappa
        self.cap = cap
        # populated by prepare(), read back by the caller after a run
        self.diag_: dict | None = None

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        n = len(df)

        frac = vote_frac(close, horizons=self.horizons, band=self.band)
        scale, _vol_ratio = conditional_scale(
            close, target_vol=self.target_vol, max_leverage=self.max_leverage,
            vol_span=self.vol_span, anchor_span_days=self.anchor_span_days,
            high_in=self.high_in, high_out=self.high_out,
            low_in=self.low_in, low_out=self.low_out)

        r = np.log(close).diff()
        vol_daily = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
                     * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()

        macro = macro_stress_z(df.index, DATA_DIR)
        mvrv = mvrv_z(df.index, DATA_DIR)
        feat_full = np.column_stack([macro, mvrv])

        cpd = build_checkpoint_data(df.index, close.to_numpy(dtype=float), vol_daily, feat_full,
                                     k=self.k, horizon_days=self.horizon_days,
                                     embargo_days=self.embargo_days)
        log_wealth_per_bar, diag = online_betting_process(
            cpd, lr=self.lr, kelly_mult=self.kelly_mult)
        self.diag_ = diag

        multiplier = np.clip(1.0 + self.kappa * np.tanh(log_wealth_per_bar / 3.0), 0.0, self.cap)
        final_desired = np.minimum(frac * scale * multiplier, self.max_leverage)

        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = final_desired[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["_log_wealth"] = log_wealth_per_bar
        df["_multiplier"] = multiplier
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures


# ============================================================================
# (3) measurement helpers
# ============================================================================


def mean_notional(result) -> float:
    if "target" not in result.df:
        return float("nan")
    tgt = np.abs(result.df["target"].to_numpy(dtype=float))
    return float(np.mean(np.clip(tgt, 0.0, result.market.leverage)))


def realized_vol(equity: pd.Series) -> float:
    eq = equity.to_numpy(dtype=float)
    if len(eq) < 3:
        return float("nan")
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
    return float(rets.std(ddof=1) * np.sqrt(BARS_PER_YEAR))


def full_measure(strategy: Strategy, start, end, market, count: bool = False) -> dict:
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    result = run_period(strategy, DF, start, end, market=market,
                         start_balance=1_000.0, data_label=LABEL)
    m = compute_metrics(result)
    return dict(metrics=m, vol=realized_vol(result.equity), notional=mean_notional(result),
                result=result, diag=getattr(strategy, "diag_", None))


def compare_to_v4(cand_result, v4_result) -> dict:
    cand_daily = daily_returns(cand_result.equity).to_numpy()
    v4_daily = daily_returns(v4_result.equity).to_numpy()
    n = min(len(cand_daily), len(v4_daily))
    cand_daily, v4_daily = cand_daily[-n:], v4_daily[-n:]
    d_log = float(total_log_return(cand_daily) - total_log_return(v4_daily))
    return dict(d_log_growth=d_log, cand_daily=cand_daily, v4_daily=v4_daily)


# ============================================================================
# (4) PHASE 0: the decisive falsification gate -- terminal training-period
#     wealth, spot only, BEFORE any inner-validation number is read.
# ============================================================================

LR_GRID = (0.01, 0.03, 0.1, 0.3)
KELLY_GRID = (0.25, 0.5, 1.0, 2.0)


def phase0_falsification_gate() -> pd.DataFrame:
    print("\n" + "=" * 100)
    print(f"PHASE 0 (decisive, pre-registered): terminal training-period (2017-{TRAIN_END}) wealth, "
          f"SPOT only -- must exceed 1/alpha={WEALTH_THRESHOLD:.0f} at alpha={ALPHA} to proceed")
    print("=" * 100)
    rows = []
    for lr in LR_GRID:
        for km in KELLY_GRID:
            strat = R180NovelBettingConfidence(lr=lr, kelly_mult=km)
            m = ev(strat, market=SPOT, tag=f"train lr={lr} km={km}", end=TRAIN_END)
            global N_EVALUATED
            N_EVALUATED += 1
            d = strat.diag_
            clears = d["terminal_wealth"] > WEALTH_THRESHOLD
            print(f"    lr={lr:<5} kelly_mult={km:<5} terminal_wealth={d['terminal_wealth']:>12.3f} "
                  f"terminal_log_wealth={d['terminal_log_wealth']:>7.3f} n_events={d['n_events']:>5d} "
                  f"n_bet={d['n_bet_nontrivial']:>5d} max|lambda|={d['max_abs_lambda']:.2f} "
                  f"first_cross_event={d['first_cross_alpha05_event']} CLEARS={clears}")
            rows.append(dict(lr=lr, kelly_mult=km, terminal_wealth=d["terminal_wealth"],
                              terminal_log_wealth=d["terminal_log_wealth"], n_events=d["n_events"],
                              n_bet=d["n_bet_nontrivial"], max_abs_lambda=d["max_abs_lambda"],
                              clears=clears, sharpe=m.sharpe))
    return pd.DataFrame(rows)


# ============================================================================
# (5) PHASE 1: kappa x cap sweep at the best (lr, kelly_mult), inner-val,
#     BOTH markets -- the actual selection grid, only reached if phase 0 clears.
# ============================================================================

KAPPA_GRID = (0.5, 1.0, 2.0)
CAP_GRID = (1.5, 2.0)


def sweep_configs(configs: list[dict], v4_cache: dict) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        for market_name, market in (("spot", SPOT), ("futures_5x", FUTURES)):
            strat = R180NovelBettingConfidence(**cfg)
            out = full_measure(strat, VAL_START, VAL_END, market, count=True)
            m = out["metrics"]
            v4 = v4_cache[market_name]
            cmp_ = compare_to_v4(out["result"], v4["result"])
            row = dict(**cfg, market=market_name,
                       final=m.final_balance, sharpe=m.sharpe, dd=m.max_drawdown_pct,
                       vol=out["vol"], notional=out["notional"],
                       time_in_mkt=m.time_in_market_pct,
                       d_sharpe=m.sharpe - v4["metrics"].sharpe,
                       d_log_growth=cmp_["d_log_growth"],
                       vol_ratio_vs_v4=out["vol"] / v4["vol"] if v4["vol"] else float("nan"),
                       notional_ratio_vs_v4=out["notional"] / v4["notional"] if v4["notional"] else float("nan"))
            rows.append(row)
            print(f"  lr={cfg['lr']:<5} km={cfg['kelly_mult']:<5} kappa={cfg['kappa']:<4} "
                  f"cap={cfg['cap']:<4} {market_name:11s} final=${m.final_balance:>9,.0f} "
                  f"sharpe={m.sharpe:>5.2f} (v4={v4['metrics'].sharpe:>5.2f}, d={row['d_sharpe']:+.3f}) "
                  f"DD={m.max_drawdown_pct:>5.1f}% vol={out['vol']:.3f} (v4={v4['vol']:.3f}) "
                  f"notional={out['notional']:.3f} (v4={v4['notional']:.3f}) "
                  f"d_log_growth={cmp_['d_log_growth']:+.4f}")
    return pd.DataFrame(rows)


# ============================================================================
# main
# ============================================================================


def hr(msg: str) -> None:
    print("\n" + "=" * 100)
    print(msg)
    print("=" * 100)


def main() -> None:
    t0 = time.time()
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(holdout 2023-01-01 onward is NEVER read by this file)")

    # ---- PHASE 0: decisive falsification gate, training period, spot only ----
    phase0 = phase0_falsification_gate()
    any_clears = bool(phase0["clears"].any())
    clearing = phase0[phase0["clears"]].copy()

    hr("PHASE 0 VERDICT")
    if any_clears:
        # Pick the SMALLEST lr (and, among ties, smallest kelly_mult) that
        # clears -- terminal wealth scales explosively with lr (32.7 at
        # lr=0.01 up to ~8e32 by lr=0.3, see the table above): this is a
        # disclosed red flag, not treated as "more real edge" -- an
        # unregularized fixed-step online SGD update can lock onto a
        # near-saturated (phat clipped at [0.02,0.98]) directional bias
        # and then compound a BOUNDED per-bet edge over ~1,400 sequential
        # events, which is genuine Kelly-betting math, not a lookahead bug
        # (independently checked: shuffling the resolved labels collapses
        # terminal wealth to ~1e-7, so the process is not reading anything
        # about its own future -- see this branch's report for the check),
        # but the monotonic explosion with lr is exactly the signature of
        # an unstable, un-decayed online learner rather than a converged
        # estimate of a fixed edge. The mildest (lr, kelly_mult) that still
        # clears the pre-registered bar is the more defensible "primary"
        # choice than the single most extreme corner in the grid.
        best0 = clearing.sort_values(["lr", "kelly_mult"]).iloc[0]
        print("  WEALTH SCALES EXPLOSIVELY WITH lr (32.7 at lr=0.01 to ~1e32+ at lr=0.3) -- a red flag for "
              "online-SGD instability, not stronger evidence of edge; see phase-0 table above. Primary "
              "config chosen as the MILDEST (lr, kelly_mult) that still clears the falsification gate, "
              "not the most extreme corner:")
    else:
        best0 = phase0.loc[phase0["terminal_wealth"].idxmax()]
    print(f"  best terminal training-period wealth: {phase0['terminal_wealth'].max():.3g} "
          f"(threshold: >{WEALTH_THRESHOLD:.0f})")
    print(f"  ANY (lr, kelly_mult) clears the falsification gate: {any_clears}")

    if not any_clears:
        hr("VERDICT: NEGATIVE BY CONSTRUCTION -- Phase 0 falsification clause")
        print("Per r180_direction.md's frozen falsification clause (novel branch, Step 1 Q4): "
              "the terminal training-period wealth must exceed 1/alpha=20 at alpha=0.05 to claim the "
              "process found any real edge at all. No (lr, kelly_mult) combination in the pre-registered "
              "grid cleared it. This is a real, well-powered negative result (NOT R-174's unreachable-n "
              "failure mode -- the pre-registered n-requirement, ~600-1,670 resolved daily checkpoints, "
              "was reachable; see n_events above, several thousand per config). The fuller kappa x cap "
              "sweep on inner-validation is NOT run, per the pre-registration's own permission to skip it "
              "when Phase 0 fails.")
        print(f"\nconfigurations evaluated: {N_EVALUATED} (Phase 0 falsification gate only: "
              f"{len(LR_GRID)}x{len(KELLY_GRID)}={len(LR_GRID) * len(KELLY_GRID)} distinct (lr, kelly_mult) "
              f"tuples, training period, SPOT).")
        print(f"\ntotal wall time: {time.time() - t0:.0f}s")
        return

    # ---- Phase 0 cleared: pick the best (lr, kelly_mult) for the main sweep ----
    lr_star = float(best0["lr"])
    km_star = float(best0["kelly_mult"])
    print(f"\n  => primary (lr, kelly_mult) for the kappa x cap sweep = ({lr_star}, {km_star})")

    # ---- v4-alone control, cached once per market, inner-validation ----
    hr("v4-alone control, inner-validation (2021-01-01 -> 2022-12-31), both markets")
    v4_cache = {}
    for market_name, market in (("spot", SPOT), ("futures_5x", FUTURES)):
        out = full_measure(KellyRegimeV4(), VAL_START, VAL_END, market)
        v4_cache[market_name] = out
        m = out["metrics"]
        print(f"  v4 alone  {market_name:11s} final=${m.final_balance:>9,.0f} "
              f"sharpe={m.sharpe:>5.2f} DD={m.max_drawdown_pct:>5.1f}% "
              f"vol={out['vol']:.3f} notional={out['notional']:.3f} "
              f"time_in_mkt={m.time_in_market_pct:.1f}%")

    # ---- Phase 1: kappa x cap sweep at (lr*, kelly_mult*), inner-validation ----
    hr(f"PHASE 1: kappa x cap sweep at (lr={lr_star}, kelly_mult={km_star}), inner-validation, "
       f"both markets ({len(KAPPA_GRID)}x{len(CAP_GRID)}={len(KAPPA_GRID) * len(CAP_GRID)} configs)")
    primary_configs = [dict(lr=lr_star, kelly_mult=km_star, kappa=kp, cap=c)
                       for kp in KAPPA_GRID for c in CAP_GRID]
    primary_rows = sweep_configs(primary_configs, v4_cache)

    # ---- Phase 2: robustness -- the runner-up (lr, kelly_mult) corners from
    #      Phase 0 at the best (kappa, cap) found above ----
    fut_rows = primary_rows[primary_rows.market == "futures_5x"].reset_index(drop=True)
    spot_rows = primary_rows[primary_rows.market == "spot"].reset_index(drop=True)
    fut_rows["joint_d_sharpe"] = fut_rows["d_sharpe"] + spot_rows["d_sharpe"]
    best_idx = fut_rows["joint_d_sharpe"].idxmax()
    kappa_star = float(fut_rows.loc[best_idx, "kappa"])
    cap_star = float(fut_rows.loc[best_idx, "cap"])
    print(f"\n  => best (kappa, cap) at this corner, by joint (spot+futures) d_sharpe on "
          f"inner-validation = ({kappa_star}, {cap_star})")

    runner_up = phase0[phase0["clears"]].sort_values("terminal_wealth", ascending=False)
    runner_up = runner_up[(runner_up["lr"] != lr_star) | (runner_up["kelly_mult"] != km_star)].head(2)
    robust_configs = [dict(lr=float(r.lr), kelly_mult=float(r.kelly_mult), kappa=kappa_star, cap=cap_star)
                      for r in runner_up.itertuples()]
    if robust_configs:
        hr(f"PHASE 2: robustness -- runner-up (lr, kelly_mult) corners that also cleared Phase 0, at "
           f"kappa={kappa_star}, cap={cap_star} ({len(robust_configs)} configs)")
        robust_rows = sweep_configs(robust_configs, v4_cache)
    else:
        robust_rows = pd.DataFrame()

    all_rows = pd.concat([primary_rows, robust_rows], ignore_index=True) if len(robust_rows) else primary_rows

    # ---- Best overall config selection (joint spot+futures d_sharpe, inner-val) ----
    hr("BEST OVERALL CONFIG (joint spot+futures d_sharpe, inner-validation)")
    fut_all = all_rows[all_rows.market == "futures_5x"].reset_index(drop=True)
    spot_all = all_rows[all_rows.market == "spot"].reset_index(drop=True)
    fut_all["joint_d_sharpe"] = fut_all["d_sharpe"] + spot_all["d_sharpe"]
    best_i = fut_all["joint_d_sharpe"].idxmax()
    best_cfg = dict(lr=float(fut_all.loc[best_i, "lr"]), kelly_mult=float(fut_all.loc[best_i, "kelly_mult"]),
                    kappa=float(fut_all.loc[best_i, "kappa"]), cap=float(fut_all.loc[best_i, "cap"]))
    print(f"  {best_cfg}")

    hr("FULL COMPARISON TABLE: best config vs v4-alone, inner-validation, both markets")
    final_rows = []
    for market_name, market in (("spot", SPOT), ("futures_5x", FUTURES)):
        strat = R180NovelBettingConfidence(**best_cfg)
        cand = full_measure(strat, VAL_START, VAL_END, market)
        v4 = v4_cache[market_name]
        cm, vm = cand["metrics"], v4["metrics"]
        cmp_ = compare_to_v4(cand["result"], v4["result"])
        boot = paired_bootstrap(cmp_["cand_daily"], cmp_["v4_daily"], total_log_return,
                                 mean_block=30.0, n_boot=2_000, seed=180)
        final_rows.append(dict(
            market=market_name,
            cand_final=cm.final_balance, v4_final=vm.final_balance,
            cand_sharpe=cm.sharpe, v4_sharpe=vm.sharpe, d_sharpe=cm.sharpe - vm.sharpe,
            cand_dd=cm.max_drawdown_pct, v4_dd=vm.max_drawdown_pct,
            cand_vol=cand["vol"], v4_vol=v4["vol"],
            cand_notional=cand["notional"], v4_notional=v4["notional"],
            cand_tim=cm.time_in_market_pct, v4_tim=vm.time_in_market_pct,
            d_log_growth=cmp_["d_log_growth"],
            boot_lo=boot.diff.lo, boot_hi=boot.diff.hi, boot_significant=boot.significant,
            boot_p_positive=boot.p_positive,
        ))
        print(f"\n  -- {market_name} --")
        print(f"     final balance:    cand=${cm.final_balance:>10,.0f}   v4=${vm.final_balance:>10,.0f}")
        print(f"     sharpe:           cand={cm.sharpe:>6.2f}          v4={vm.sharpe:>6.2f}   "
              f"d_sharpe={cm.sharpe - vm.sharpe:+.3f}")
        print(f"     max drawdown:     cand={cm.max_drawdown_pct:>6.1f}%         v4={vm.max_drawdown_pct:>6.1f}%")
        print(f"     realized vol:     cand={cand['vol']:.3f}          v4={v4['vol']:.3f}   "
              f"ratio={cand['vol']/v4['vol'] if v4['vol'] else float('nan'):.2f}")
        print(f"     avg notional:     cand={cand['notional']:.3f}          v4={v4['notional']:.3f}   "
              f"ratio={cand['notional']/v4['notional'] if v4['notional'] else float('nan'):.2f}")
        print(f"     time in market:   cand={cm.time_in_market_pct:.1f}%         v4={vm.time_in_market_pct:.1f}%")
        print(f"     d_log_growth (daily, full window) = {cmp_['d_log_growth']:+.4f}")
        print(f"     paired block-bootstrap 95% CI on d_log_growth (daily, 30d mean block, n_boot=2000): "
              f"[{boot.diff.lo:+.4f}, {boot.diff.hi:+.4f}]  significant={boot.significant}  "
              f"p(diff>0)={boot.p_positive:.3f}")

    final_df = pd.DataFrame(final_rows)

    # ---- R-33 risk-matching + promotion-bar gate, applied mechanically ----
    hr("R-33 RISK-MATCHING + PRE-REGISTERED PROMOTION-BAR GATE, applied mechanically")
    vol_ratios = final_df["cand_vol"] / final_df["v4_vol"]
    notional_ratios = final_df["cand_notional"] / final_df["v4_notional"]
    risk_matched = bool(((vol_ratios - 1.0).abs() <= 0.10).all()
                         and ((notional_ratios - 1.0).abs() <= 0.10).all())
    both_sig_positive = bool((final_df["boot_significant"] & (final_df["d_log_growth"] > 0)).all())
    both_beat_noise_floor = bool((final_df["d_sharpe"].abs() > 0.2).all() and (final_df["d_sharpe"] > 0).all())

    print(f"  vol ratios (cand/v4):      {dict(zip(final_df['market'], vol_ratios.round(3)))}")
    print(f"  notional ratios (cand/v4): {dict(zip(final_df['market'], notional_ratios.round(3)))}")
    print(f"  risk-matched (R-33 convention: notional AND realized-vol ratios both in [0.9,1.1] on BOTH "
          f"markets): {risk_matched}")
    print(f"  paired-bootstrap-plausible improvement, same sign, BOTH markets: {both_sig_positive}")
    print(f"  |d_sharpe| exceeds the +/-0.2 noise floor AND is positive, BOTH markets: {both_beat_noise_floor}")

    promote = bool(risk_matched and both_sig_positive)

    hr("VERDICT")
    if not risk_matched:
        print("NEGATIVE (R-33) -- the candidate's realized volatility or average notional differs "
              "materially from v4-alone's on at least one market (see ratios above): this branch "
              "deliberately varies exposure via the wealth-multiplier, and the resulting comparison is "
              "not risk-matched, so any apparent Sharpe/log-growth improvement cannot be distinguished "
              "from simply running hotter or cooler than v4. Not scored as a win.")
    elif promote:
        print("PROMOTE-CANDIDATE (inner-validation only; holdout not read by this branch) -- risk-matched "
              "within +/-10% (notional and vol) on both markets, and the paired-bootstrap 95% CI on "
              "d_log_growth excludes zero on the winning side on BOTH BTC markets.")
    else:
        print("NEGATIVE -- risk is matched but the pre-registered bar (paired-bootstrap plausible "
              "improvement on BOTH markets, risk-matched) is not met. See the comparison table above.")

    n_phase1 = len(KAPPA_GRID) * len(CAP_GRID)
    n_phase2 = len(robust_configs)
    print(f"\nconfigurations evaluated: {N_EVALUATED}")
    print(f"  breakdown: phase 0 (lr x kelly_mult falsification gate, train, SPOT) = "
          f"{len(LR_GRID) * len(KELLY_GRID)}; phase 1 (kappa x cap sweep at chosen (lr,kelly_mult), "
          f"inner-val, both markets) = {n_phase1} distinct configs (x2 markets each); phase 2 (robustness "
          f"-- runner-up (lr,kelly_mult) corners at best (kappa,cap), inner-val, both markets) = "
          f"{n_phase2} distinct configs (x2 markets each). Distinct parameter tuples = "
          f"{len(LR_GRID) * len(KELLY_GRID) + n_phase1 + n_phase2}.")
    print(f"\ntotal wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
