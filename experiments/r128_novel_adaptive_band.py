"""R-128 NOVEL branch: an adaptive, state-dependent EV-band on ``hedge_experts``.

See ``experiments/r128_shared.py`` (frozen, not edited by this file) for the
full pre-registration: direction, non-duplicate argument, falsification test
and decision rule. This module only implements the NOVEL mechanism named
there and the battery of checks the round's protocol requires.

**Mechanism.** ``hedge_experts`` blends ten causal technical experts with a
discounted-Hedge (multiplicative-weights) update into a signal ``x`` in
``[-1, 1]``, then re-targets its position toward ``x`` whenever
``abs(x - pos) > hysteresis`` for a FIXED ``hysteresis = 0.05`` -- a
hand-set number. This module replaces that fixed deadband with the same
quadratic-growth-cost-vs-linear-fee algebra ``kelly_regime_ev`` already
shipped (Constantinides 1986; Davis & Norman 1990):

    band = 2 * fee / (H * sigma**2)

but, unlike the CONSERVATIVE branch (a single frozen constant ``H``), this
branch computes ``H`` ONLINE, every bar, from the blend signal ``x``'s own
trailing persistence:

1. A causal EWM correlation between ``x`` and ``x`` shifted by one bar
   (``x.ewm(span=...).corr(x.shift(1))``) over a trailing window of
   ``ac_lookback_days`` trading days (primary: 20; swept over
   ``{10, 20, 40, 80}`` in B3). ``pandas.Series.ewm(...).corr(...)`` is
   causal by construction -- at row ``i`` it only uses ``x[0..i]`` and
   ``x.shift(1)[0..i] = x[-1..i-1]``, both already known at bar ``i``'s
   close, i.e. no different in kind from any other EWM statistic this
   project already treats as causal (e.g. ``hedge_experts``'s own
   ``sig1 = r.ewm(span=288).std()``).
2. That autocorrelation ``rho`` is converted to an implied AR(1) horizon
   in days via the relation specified for this round:
   ``H_t = -1 / (BARS_PER_DAY * ln(rho))`` for ``0 < rho < 1``, with
   ``rho`` clipped to ``[0.02, 0.98]`` (disclosed, structural -- not swept
   for best results) before the log, and the resulting ``H_t`` clipped to
   ``[0.25, 10]`` days (also disclosed, structural bounds).
3. ``H_t`` replaces the CONSERVATIVE branch's single frozen ``H`` in the
   same ``band = 2*fee/(H_t*sigma**2)`` formula, using
   ``ctx.market.fee_rate`` (the real venue fee) and ``sigma`` =
   ``hedge_experts``'s own already-computed ``sig1`` (EWM realized vol),
   annualized the same way ``kelly_regime_ev``'s ``_ev_vol`` is.

Economically: a highly persistent blend signal (rho near 1, H_t large)
implies holding the wrong exposure compounds its opportunity cost over a
long regime, so the band narrows and the strategy chases the signal more
readily. A signal with little persistence (rho near 0, H_t small) implies
any edge decays before a rebalance could capture it, so the band widens
and the strategy mostly declines to trade. This is the "state-dependence
of the horizon" axis ``r128_shared.py`` names (R-89 finding 11), applied
to the REBALANCE BAND's horizon input via a continuous AR(1)-persistence
estimate -- not a re-run of B-40 (which made the VOTE's own blend weights
state-dependent via a discrete 4-state classifier).

**Architecture, inherited/shared with the CONSERVATIVE branch (disclosed,
not independently invented by this branch):** ``x`` is computed in
``prepare()`` with NO hysteresis at all (the raw Hedge-blend output every
bar). The position update moves from ``prepare()`` into ``on_bar``, using
``ctx.market.fee_rate`` and a live, ``ctx.position``-derived current
fraction of equity (``ctx.position * ctx.close / ctx.equity``), exactly
mirroring ``kelly_regime_ev.on_bar``'s own body (including its "always
allow a full exit to flat when desired == 0" clause, kept here verbatim
because standing flat removes the whole position's risk, a benefit the
quadratic opportunity-cost term does not price in). Trades are placed via
``ctx.order_notional`` (leverage-invariant sizing), NOT
``hedge_experts``'s original ``ctx.order_target`` (leverage-scaled) -- a
structural change shared by both branches, not introduced independently
by this one. The ONLY structural difference between the two branches is
this file's ADAPTIVE ``H_t`` versus the conservative branch's FIXED ``H``.

**Warmup, disclosed choice.** ``warmup = 2500``, identical to
``hedge_experts``'s own value, so every B1 comparison against the
registered baseline gets the SAME warmup budget on both sides (see the
module-level comment above ``_b1_candidate`` for why this matters: giving
the candidate more history than the baseline gets, purely because more
data exists, was empirically confirmed during development to inject a
spurious multi-tenths-of-Sharpe advantage having nothing to do with the
mechanism -- an artifact of the Hedge weight vector's slow, path-dependent
convergence from its zero initialization, not of signal quality). One
consequence: for the wider AC lookbacks in B3 (40, 80 days), the
EWM-correlation's own ``min_periods`` (roughly a quarter of the lookback
in bars) can exceed the 2500-bar warmup budget, so ``H_t`` starts several
of those runs at its fallback floor (``H_MIN_DAYS``) and only becomes a
genuine estimate partway into the measured window. This is disclosed, not
hidden, and does not violate causality (the fallback is itself a fixed,
pre-registered constant, not a peek).

POST-HOC CORRECTION (operator, before any promotion decision was recorded),
two independent issues found on review:

1. **Unit mismatch, shared with the conservative branch.** ``desired_x``
   is already in "fraction of this market's max leverage" units (the
   convention ``hedge_experts``'s own experts and its original
   ``ctx.order_target`` call use). The first version of this file compared
   it against ``current = position*close/equity``, a NOTIONAL-MULTIPLE
   (ranges to +/-5 on 5x futures) -- coincidentally identical to the
   fraction-of-leverage convention on spot (leverage=1) but not on
   futures -- and placed orders via ``order_notional`` (leverage-invariant)
   rather than ``order_target`` (leverage-scaled). Net effect: the
   candidate could not exceed roughly 1x equity notional on futures
   regardless of the real 5x cap, while the baseline could -- an
   EXPOSURE-LEVEL artifact (the R-33 trap this project's ledger names
   repeatedly), not a rebalance-timing effect. Fixed below: ``current`` is
   now ``position*close/(equity*leverage)``, ``_band`` gains one extra
   ``/leverage`` factor (identical to the conservative branch's own
   correction; algebra in that file's docstring), and orders are placed
   via ``ctx.order_target`` again.
2. **``RHO_MAX``/``H_MIN_DAYS`` were mutually inconsistent, discovered
   independently during this branch's own report-writing.** At
   ``BARS_PER_DAY=288``, the maximum unclipped ``H_t`` reachable at
   ``rho=RHO_MAX=0.98`` is ``-1/(288*ln(0.98)) ~= 0.172`` days -- always
   BELOW the ``H_MIN_DAYS=0.25`` floor, so ``H_t`` was mathematically
   forced to the floor on every bar, in every configuration tested; the
   "adaptive" mechanism this branch exists to test never actually varied,
   and B3's four sweep points were byte-identical for exactly this reason.
   Fixed below: ``RHO_MAX`` raised to ``0.999`` (``H_t`` at that bound is
   ``~3.47`` days, inside ``[H_MIN_DAYS, H_MAX_DAYS]`` with room either
   side) -- a structural choice (comfortably below certainty, not fit to
   any result) rather than a value swept for best performance.

SPOT numbers are numerically identical before and after fix 1 (leverage=1
makes the two formulas coincide); fix 2 changes every cell's ``H_t`` path,
so the WHOLE battery was re-run after both fixes, not only the futures
cells. The battery results and report below are from the CORRECTED
version; the first (confounded, degenerate) run's numbers are preserved in
this file's git history and in the R-128 ledger entry's own discussion,
not deleted, per this project's "nothing is deleted, annotate in place"
convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r128_shared import (  # noqa: E402
    BARS_PER_DAY, BARS_PER_YEAR, FUTURES, FUTURES_HIGH_FEE, INNER_TRAIN_END,
    INNER_TRAIN_START, INNER_VAL_END, INNER_VAL_START, SPOT, SPOT_HIGH_FEE,
    b1_signal, load_btc_train, load_eth_train, run_baseline,
)
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategies.hedge_experts import HedgeExperts  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import prefix_bars, run_period  # noqa: E402


class HedgeExpertsAdaptiveBand(HedgeExperts):
    """``hedge_experts`` with a fee/vol/horizon-derived band whose horizon
    is estimated online from the blend signal's own AR(1) persistence.

    Not a registered strategy: this class lives only in ``experiments/``
    per this project's convention (see module docstring above for the
    full derivation). It reuses ``HedgeExperts._experts()`` verbatim so
    the expert construction and Hedge weight update are byte-identical to
    the registered strategy; only the position-update rule changes.
    """

    name = "r128_novel_adaptive_band"
    warmup = 2500  # == HedgeExperts.warmup, disclosed for B1 fairness (see module docstring)

    RHO_MIN = 0.02
    RHO_MAX = 0.999
    H_MIN_DAYS = 0.25
    H_MAX_DAYS = 10.0

    def __init__(self, ac_lookback_days: float = 20.0, eta: float = 0.05,
                 fixed_share: float = 1e-4, fee_rate: float = 0.0005,
                 min_band: float = 0.02, max_band: float = 1.0) -> None:
        super().__init__(eta=eta, fixed_share=fixed_share, hysteresis=0.0,
                          fee_rate=fee_rate)
        self.ac_lookback_days = float(ac_lookback_days)
        self.min_band = min_band
        self.max_band = max_band
        self._recorded_target: np.ndarray | None = None

    # ------------------------------------------------------------- signal

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        r = np.log(df["close"]).diff()
        sig1 = r.ewm(span=288, min_periods=250).std()
        a = self._experts(df, r, sig1)  # (n, N); byte-identical to HedgeExperts
        r_a = r.to_numpy()
        sig_a = sig1.shift(1).to_numpy()

        n, num = a.shape
        x_arr = np.zeros(n)
        logw = np.zeros(num)
        # Identical Hedge weight-update loop to HedgeExperts.prepare(), minus
        # the hysteresis re-target clause -- x_arr holds the RAW blend value
        # every bar (or, when vol is not yet finite, carries the previous
        # value forward, matching HedgeExperts's own `target[i] = pos` on the
        # same condition).
        for i in range(2, n):
            s = sig_a[i]
            if not np.isfinite(s) or s <= 0:
                x_arr[i] = x_arr[i - 1]
                continue
            z_t = min(max(r_a[i] / (3.0 * s), -1.0), 1.0)
            fee_n = min(self.fee_rate / (3.0 * s), 0.25)
            g = np.clip(a[i - 1] * z_t - fee_n * np.abs(a[i - 1] - a[i - 2]), -1.0, 1.0)
            logw += self.eta * g
            logw -= logw.max()
            p = np.exp(logw)
            p /= p.sum()
            p = (1.0 - self.fixed_share) * p + self.fixed_share / num
            logw = np.log(p)
            x_arr[i] = float(p @ a[i])

        x = pd.Series(x_arr, index=df.index)

        # Causal EWM lag-1 autocorrelation of x over a trailing window of
        # `ac_lookback_days` trading days. `.ewm(...).corr(...)` at row i
        # uses only x[0..i] and x.shift(1)[0..i] = x[-1..i-1] -- both known
        # at bar i's close. No centered window, no lookahead.
        span_bars = max(int(round(self.ac_lookback_days * BARS_PER_DAY)), 20)
        min_periods = max(span_bars // 4, 10)
        rho = x.ewm(span=span_bars, min_periods=min_periods).corr(x.shift(1))
        rho_c = rho.clip(lower=self.RHO_MIN, upper=self.RHO_MAX)
        with np.errstate(divide="ignore", invalid="ignore"):
            h_t = -1.0 / (BARS_PER_DAY * np.log(rho_c))
        h_t = h_t.clip(lower=self.H_MIN_DAYS, upper=self.H_MAX_DAYS)
        h_t = h_t.fillna(self.H_MIN_DAYS)  # pre-min_periods / non-finite fallback

        df["x"] = x
        # Same lagged sig1 the expert loop already uses, annualized the way
        # kelly_regime_ev's `_ev_vol` is, so `band`'s units are consistent
        # (horizon in years, variance annualized).
        df["_ab_sigma"] = sig1.shift(1) * np.sqrt(BARS_PER_YEAR)
        df["_ab_H"] = h_t
        df["_ab_rho"] = rho_c

        # Bookkeeping only (does not affect real orders): records the
        # order_target-unit position this strategy's on_bar decides on each
        # bar, so a live run of this class can be replayed through
        # r128_shared.b1_signal's frozen order_target wrapper for the
        # paired-bootstrap comparison against the registered baseline.
        self._recorded_target = np.zeros(n)
        return df

    def _band(self, fee_rate: float, h_days: float, vol: float, leverage: float) -> float:
        """CORRECTED (post-hoc fix, see module docstring addendum): threshold
        on |delta_exposure| below which trading destroys value, re-derived
        for fraction-of-max-leverage units (see conservative branch's own
        addendum for the algebra -- identical extra ``/leverage`` factor)."""
        horizon_years = max(h_days, 1e-9) / 365.25
        variance = max(vol, 1e-6) ** 2
        lev = max(leverage, 1e-9)
        band = 2.0 * fee_rate / (horizon_years * variance * lev)
        return float(np.clip(band, self.min_band, self.max_band))

    # -------------------------------------------------------------- orders

    def on_bar(self, ctx: Context) -> None:
        """CORRECTED (post-hoc fix -- see module docstring addendum):
        ``current`` is now expressed in the same fraction-of-max-leverage
        units as ``desired_x`` itself, and orders are placed via the
        original ``ctx.order_target``, not ``ctx.order_notional``."""
        i = ctx.i
        prev_recorded = float(self._recorded_target[i - 1]) if i > 0 else 0.0

        desired_x = float(ctx.bar["x"])
        vol = float(ctx.bar["_ab_sigma"])
        h_days = float(ctx.bar["_ab_H"])
        if not np.isfinite(vol) or vol <= 0 or not np.isfinite(h_days):
            self._recorded_target[i] = prev_recorded
            return

        equity = ctx.equity
        if equity <= 0:
            self._recorded_target[i] = prev_recorded
            return
        lev = max(ctx.market.leverage, 1e-9)
        current = ctx.position * ctx.close / (equity * lev)

        band = self._band(ctx.market.fee_rate, h_days, vol, lev)

        # Inherited from kelly_regime_ev.on_bar (and shared with the
        # conservative branch): always allow a full exit to flat.
        if desired_x == 0.0 and abs(current) > 1e-9:
            ctx.order_target(0.0)
            self._recorded_target[i] = 0.0
            return

        if abs(desired_x - current) > band:
            ctx.order_target(desired_x)
            self._recorded_target[i] = desired_x
        else:
            self._recorded_target[i] = prev_recorded


# ---------------------------------------------------------------------
# Battery helpers.
#
# b1_signal (r128_shared.py) expects a `candidate_target` ndarray it can
# replay through a frozen `ctx.order_target(t)` wrapper, aligned
# POSITION-FOR-POSITION with the `df` it is handed -- and, importantly,
# with the SAME warmup budget the baseline strategy gets, or the
# comparison is not apples-to-apples (confirmed empirically before this
# module was finalized: replaying hedge_experts's own target array back
# through b1_signal using its full multi-year history reproduces neither
# hedge_experts's own inner-validation Sharpe nor final balance -- off by
# 0.72 Sharpe purely from the Hedge weight vector having converged for
# longer -- while giving it EXACTLY the warmup hedge_experts.warmup
# affords, no more, reproduces the baseline bit-for-bit). `_b1_candidate`
# below reproduces run_period's own prefix arithmetic so that, since this
# class's `warmup` (2500) never exceeds the frozen replay wrapper's own
# warmup (also 2500), the two always align with zero slack.
#
# POST-HOC CORRECTION (operator): `_recorded_target[i]` now records
# `desired_x` directly (not `desired_x / lev`). The `/lev` division in the
# first version of this file was compensating for `on_bar` placing the
# live order via `ctx.order_notional` (leverage-invariant) while this
# replay wrapper always used `ctx.order_target` (leverage-scaled) -- i.e.
# it was reproducing the SAME exposure cap the live order's own unit bug
# introduced, not curing it. Now that `on_bar` places its live order via
# `ctx.order_target(desired_x)` directly (see the class's own
# POST-HOC CORRECTION note above `_band`/`on_bar`), recording `desired_x`
# unchanged makes the replay agree with the live backtest's own real
# fills, both using the same leverage-scaled convention hedge_experts
# itself uses.
# ---------------------------------------------------------------------

def _b1_candidate(df_full: pd.DataFrame, market, start: str, end: str,
                   label: str = "", **kwargs) -> tuple[pd.DataFrame, np.ndarray]:
    strat = HedgeExpertsAdaptiveBand(**kwargs)
    lo = int(df_full.index.searchsorted(start))
    hi = int(df_full.index.searchsorted(end, side="right"))
    prefix = prefix_bars(df_full, lo, strat.warmup)
    frame = df_full.iloc[lo - prefix: hi].copy()
    run_backtest(strat, frame, market, start_balance=1000.0, data_label=label,
                 trade_start=prefix)
    return frame, strat._recorded_target.copy()


def b1_cell(df_full: pd.DataFrame, market, start: str, end: str,
            label: str = "", **kwargs) -> dict:
    frame, target = _b1_candidate(df_full, market, start, end, label, **kwargs)
    return b1_signal(target, frame, market, start=start, end=end)


def run_candidate_direct(df: pd.DataFrame, market, start, end, label: str = "",
                          **kwargs):
    strat = HedgeExpertsAdaptiveBand(**kwargs)
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


PRIMARY_KWARGS = dict(ac_lookback_days=20.0)


# ============================================================ battery ====

def step1_truncation() -> bool:
    df, label = load_btc_train("spot")
    m_full, _ = run_candidate_direct(df, SPOT, INNER_TRAIN_START, INNER_TRAIN_END,
                                      label, **PRIMARY_KWARGS)
    df_trunc = df.loc[:INNER_VAL_END]  # r128_shared's own convention: slice further
    m_trunc, _ = run_candidate_direct(df_trunc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END,
                                       label, **PRIMARY_KWARGS)
    ok = np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9)
    print(f"[1] causal truncation probe: {'PASS' if ok else 'FAIL'} "
          f"({m_full.final_balance} vs {m_trunc.final_balance})")

    # A second, harder truncation point closer to the actual OOS boundary,
    # using the class's own prepare() output directly (checks the AC/H_t
    # columns too, not just final equity).
    df2, _ = load_btc_train("spot")
    cut = "2020-06-30"
    strat_a = HedgeExpertsAdaptiveBand(**PRIMARY_KWARGS)
    prep_a = strat_a.prepare(df2.loc[:cut].copy())
    strat_b = HedgeExpertsAdaptiveBand(**PRIMARY_KWARGS)
    prep_b = strat_b.prepare(df2.copy())
    common = prep_a.index
    close_x = np.allclose(prep_a["x"].to_numpy(),
                           prep_b.loc[common, "x"].to_numpy(), equal_nan=True)
    close_h = np.allclose(prep_a["_ab_H"].to_numpy(),
                           prep_b.loc[common, "_ab_H"].to_numpy(), equal_nan=True,
                           atol=1e-12)
    ok2 = bool(close_x and close_h)
    print(f"[1b] prepare()-column truncation probe (x, H_t): "
          f"{'PASS' if ok2 else 'FAIL'}")
    return bool(ok and ok2)


def step2_b1() -> dict:
    df_btc, label_btc = load_btc_train("spot")
    cells = {}
    for mkt_name, market in (("spot", SPOT), ("futures", FUTURES)):
        for win_name, (s, e) in (
            ("full", (INNER_TRAIN_START, INNER_VAL_END)),
            ("val", (INNER_VAL_START, INNER_VAL_END)),
        ):
            res = b1_cell(df_btc, market, s, e, label_btc, **PRIMARY_KWARGS)
            cells[f"{mkt_name}-{win_name}"] = res
            print(f"[2] B1 {mkt_name}-{win_name}: d_sharpe={res['d_sharpe']:.4f} "
                  f"paired=[{res['paired_lo']:.3f},{res['paired_hi']:.3f}] "
                  f"sig={res['significant']} dd_cand={res['dd_cand']:.1f} "
                  f"dd_base={res['dd_base']:.1f} trades_cand={res['trades_cand']} "
                  f"trades_base={res['trades_base']} final_cand={res['final_cand']:.1f} "
                  f"final_base={res['final_base']:.1f}")
    return cells


def step3_b3() -> tuple[dict, dict]:
    # Matches r128_shared.py's own convention (its __main__ self-test runs
    # both SPOT and FUTURES market specs over the same spot-loaded price
    # series -- no committed real perp file exists, so load_dataset(kind=
    # "perp") would silently fall back to the identical spot series anyway).
    df_fut, label_fut = load_btc_train("spot")
    sweep = {}
    for lookback in (10, 20, 40, 80):
        res = b1_cell(df_fut, FUTURES, INNER_VAL_START, INNER_VAL_END, label_fut,
                       ac_lookback_days=float(lookback))
        sweep[lookback] = res
        print(f"[3] B3 lookback={lookback}d: d_sharpe={res['d_sharpe']:.4f}")

    # Diagnostic: distribution of the realized H_t the PRIMARY config
    # produces on inner-train (not gating).
    df_spot, _ = load_btc_train("spot")
    strat = HedgeExpertsAdaptiveBand(**PRIMARY_KWARGS)
    prep = strat.prepare(df_spot.copy())
    h_inner_train = prep.loc[INNER_TRAIN_START:INNER_TRAIN_END, "_ab_H"]
    diag = {
        "median": float(h_inner_train.median()),
        "p5": float(h_inner_train.quantile(0.05)),
        "p95": float(h_inner_train.quantile(0.95)),
        "frac_at_floor": float((h_inner_train <= HedgeExpertsAdaptiveBand.H_MIN_DAYS + 1e-9).mean()),
        "frac_at_ceiling": float((h_inner_train >= HedgeExpertsAdaptiveBand.H_MAX_DAYS - 1e-9).mean()),
    }
    print(f"[3] H_t diagnostic (primary config, inner-train): "
          f"median={diag['median']:.3f}d p5={diag['p5']:.3f}d p95={diag['p95']:.3f}d "
          f"frac_at_floor={diag['frac_at_floor']:.3f} frac_at_ceiling={diag['frac_at_ceiling']:.3f}")
    return sweep, diag


def step4_b4() -> dict:
    eth = load_eth_train()
    res = b1_cell(eth, SPOT, INNER_VAL_START, INNER_VAL_END, "ETH spot (coinbase)",
                  **PRIMARY_KWARGS)
    print(f"[4] B4 ETH spot inner-val: d_sharpe={res['d_sharpe']:.4f}")
    return res


def step5_b5() -> dict:
    df_btc, label_btc = load_btc_train("spot")
    cells = {}
    for mkt_name, market in (("spot_hf", SPOT_HIGH_FEE), ("futures_hf", FUTURES_HIGH_FEE)):
        for win_name, (s, e) in (
            ("full", (INNER_TRAIN_START, INNER_VAL_END)),
            ("val", (INNER_VAL_START, INNER_VAL_END)),
        ):
            res = b1_cell(df_btc, market, s, e, label_btc, **PRIMARY_KWARGS)
            cells[f"{mkt_name}-{win_name}"] = res
            print(f"[5] B5 {mkt_name}-{win_name}: d_sharpe={res['d_sharpe']:.4f}")
    return cells


if __name__ == "__main__":
    n_configs = 0

    ok_trunc = step1_truncation()
    n_configs += 1

    b1_cells = step2_b1()
    n_configs += 4

    b3_sweep, h_diag = step3_b3()
    n_configs += 4

    b4_cell = step4_b4()
    n_configs += 1

    b5_cells = step5_b5()
    n_configs += 4

    print(f"\n[7] total distinct configurations evaluated: {n_configs}")

    # ---- decision rule (pre-registered in r128_shared.py; not altered) ----
    def clears(res: dict) -> bool:
        return (res["d_sharpe"] > 0.2) or (res["dd_cand"] < res["dd_base"])

    b1_pass = all(clears(b1_cells[k]) for k in b1_cells)
    b3_same_sign = sum(1 for v in b3_sweep.values() if v["d_sharpe"] > 0)
    b3_pass = b3_same_sign >= 3 or (4 - b3_same_sign) >= 3
    btc_val_sign = np.sign(b1_cells["spot-val"]["d_sharpe"])
    eth_sign = np.sign(b4_cell["d_sharpe"])
    b4_pass = bool(btc_val_sign == eth_sign and btc_val_sign != 0)
    normal_signs = {
        "spot-full": np.sign(b1_cells["spot-full"]["d_sharpe"]),
        "spot-val": np.sign(b1_cells["spot-val"]["d_sharpe"]),
        "futures-full": np.sign(b1_cells["futures-full"]["d_sharpe"]),
        "futures-val": np.sign(b1_cells["futures-val"]["d_sharpe"]),
    }
    hf_signs = {
        "spot-full": np.sign(b5_cells["spot_hf-full"]["d_sharpe"]),
        "spot-val": np.sign(b5_cells["spot_hf-val"]["d_sharpe"]),
        "futures-full": np.sign(b5_cells["futures_hf-full"]["d_sharpe"]),
        "futures-val": np.sign(b5_cells["futures_hf-val"]["d_sharpe"]),
    }
    b5_pass = all(normal_signs[k] == hf_signs[k] for k in normal_signs)

    verdict = "PROMOTE-candidate" if (ok_trunc and b1_pass and b3_pass and b4_pass and b5_pass) else "NEGATIVE"
    print(f"\n[decision] truncation={ok_trunc} b1={b1_pass} b3={b3_pass} "
          f"({b3_same_sign}/4 positive) b4={b4_pass} b5={b5_pass} -> {verdict}")
