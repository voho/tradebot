#!/usr/bin/env python
"""Bag the anchor-ladder choice across R-07's validated 18-28d plateau (SIZE/ERR axis).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea
--------
``kelly_regime_v4`` answers "is the regime bullish?" with three latched
anchor votes on a SINGLE doubling ladder (20/40/80 days), chosen once and
shipped as if it were certain. R-07 already swept nine base periods in the
18-28 day range and found the whole region is a validated PLATEAU: every
variant cut max drawdown to 35-39% from 41.8%, and the Sharpe spread
(1.52-1.60) sat inside the +/-0.2 noise floor. Nobody has ever asked what
happens if that plateau is treated as what it is -- several near-equally-
plausible points, not one -- and AVERAGED rather than picked from.

Bootstrap aggregating (Breiman 1996, Machine Learning 24(2)) shows that
averaging a statistic across resamples of the fitting procedure reduces
variance without changing the expected value, precisely when the
individual estimates are unstable but roughly unbiased -- exactly R-07's
description of the 18-28d region. Recent ensemble-averaging work (e.g.
stochastic weight averaging and adaptive-ensembling literature, 2024-2025)
makes the identical variance-reduction argument for hyperparameter/model
choice specifically, not just for data resampling. Here the "bootstrap
resample" is not data -- it is the a-priori, not-fitted choice of which
already-validated ladder base to use.

Mechanism, one sentence: replace v4's single (20,40,80) ladder vote with
the plain, UNWEIGHTED average of the same latched vote fraction computed
independently on a small, fixed-in-advance set of doubling ladders spanning
R-07's 18-28d region, and feed that average into v3/v4's completely
UNCHANGED conditional-vol-target scale.

Constraint attacked: ERR (no error control anywhere in the signal path).
R-07 measured that the anchor-ladder choice sits on a plateau of near-
equally-plausible points; v4 nonetheless treats one point on it as
certain. This is model averaging over an already-validated hyperparameter
plateau -- a form of error control on the CHOICE of anchor ladder, not on
the vote mechanism itself.

Not a duplicate of
-------------------
- R-06 (7-48 anchor sweep) / R-07 (18-28d plateau sweep): both measured
  INDIVIDUAL points on the plateau. Neither ever averaged/ensembled them.
  This file's only job is to test the averaging, using R-07's own already-
  validated region as the ensemble membership -- no new sweep of the
  region itself.
- ``kelly_regime_v2`` (``vote_gamma`` convexity): shrinks exposure based on
  DISAGREEMENT among the 3 anchors WITHIN one ladder. This file averages
  the vote fraction ACROSS multiple different ladders/timescale choices --
  a different axis entirely -- and never touches ``vote_gamma`` (fixed
  implicitly at the v3/v4 default of linear, gamma=1).
- R-34, R-37, R-38 (different signal sources / different parts of the
  sizing formula -- Bayesian posterior margin, per-state Kelly fraction,
  CRRA/drawdown-constrained caps): none of them ensemble the anchor-ladder
  choice itself; all keep v4's single ladder and change something else.
- The companion "novel" branch of this same round,
  ``kelly_regime_v8_uncertainty_shrink.py`` (not read, not written by this
  file): that branch additionally uses cross-ladder DISAGREEMENT as a
  causal uncertainty signal to shrink exposure below the plain average.
  This file is strictly the plain unweighted average -- no disagreement-
  based shrinkage, no formula beyond averaging ``frac`` across ladders.

Causal construction -- read this before the code
--------------------------------------------------
Every ladder's vote is built EXACTLY like ``kelly_regime.py``'s own vote:
a ``close.rolling(days * BARS_PER_DAY).mean()`` anchor (rolling means look
backward only), a latch built with a forward loop using only ``<= t``
information (``ffill().fillna(0.0)`` on a "bull (1) / bear (0) / hold-
previous (nan)" series -- identical to ``kelly_regime.py`` line for line,
run once per ladder). No expanding statistic sees its own future. The only
new arithmetic is ``frac_bagged[t] = mean_k(frac_k[t])``, a plain per-bar
average of already-causal series -- averaging causal quantities at a fixed
bar index cannot introduce lookahead. v3's conditional-vol-target scale
(the high/low-vol-breakout hysteresis state machine) is copied verbatim,
unchanged, from ``kelly_regime_v3.py``.

Pre-registered failure modes (named before any code ran)
------------------------------------------------------------
(a) The bagged frac barely differs from v4's own single-ladder frac (e.g.
    correlation > 0.98) because the 18-28d plateau ladders mostly agree --
    "no effect", a legitimate negative finding, not a failure of
    execution. Checked directly: correlation of each candidate's
    ``frac_bagged`` against the (20,40,80)-only ladder (v4's own vote,
    reproduced verbatim by this same class with ``ladder_bases=(20,)``).
(b) Averaging produces a genuine but tiny improvement inside the +/-0.2
    Sharpe noise floor -- not distinguishable from the incumbent.
(c) The result is an exposure-level artifact: regress the candidate's
    target series against a mean-notional-matched flat rescale of v4's own
    target series and report R^2 (R^2 > 0.95 is this project's standing
    threshold for "this is just a rescale, not a real mechanism" --
    R-33/R-34's diagnostic).
(d) Fails the ETH falsification test (worse than v4 on ETH, or visibly
    worse on ETH than the identical pipeline's BTC control).

Ensemble definitions tested (fixed in advance, not fitted; step 3's only
free axis is WHICH of these four sets, not any per-ladder parameter)
------------------------------------------------------------------------
- ``full6_18-28``          = {18,20,22,24,26,28} -- R-07's region, evenly
  spaced, 6 ladders. The primary, most-representative candidate.
- ``coarse3_18-23-28``     = {18,23,28} -- a coarser 3-ladder subset of the
  same region.
- ``edges2_18-28``         = {18,28} -- the minimal 2-ladder ensemble, just
  the region's own endpoints.
- ``wide_negcontrol_14-21-28`` = {14,21,28} -- a NEGATIVE CONTROL: 14d sits
  below R-07's own reported break point (16/32/64 scored 1.46, well off
  the plateau), so if the plateau's edge shows up here too, this set
  should measurably underperform the other three.

Usage
-----
    python experiments/kelly_regime_v8_ladder_bag.py sweep       # step 3
    python experiments/kelly_regime_v8_ladder_bag.py select      # step 5
    python experiments/kelly_regime_v8_ladder_bag.py fraccorr    # failure mode (a)
    python experiments/kelly_regime_v8_ladder_bag.py artifact    # failure mode (c)
    python experiments/kelly_regime_v8_ladder_bag.py causality   # step 6
    python experiments/kelly_regime_v8_ladder_bag.py eth         # step 7 / failure mode (d)
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

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


# --------------------------------------------------------------------- strategy


class KellyRegimeV8LadderBag(Strategy):
    """Average the latched vote fraction across a fixed set of doubling ladders spanning R-07's plateau.

    Everything about v3/v4's own conditional-vol-target scale (the
    high/low-vol-breakout state machine, ``target_vol``, ``max_leverage``)
    is copied verbatim from ``kelly_regime_v3.py``. The only new thing is
    the vote: instead of one ladder's latched fraction, this computes the
    SAME latch independently for each base period in ``ladder_bases``
    (each expanded to ``(h, 2h, 4h)`` exactly like v4's ``(20,40,80)``) and
    plain-averages the resulting fractions, unweighted, before handing the
    result to v3's unchanged scale.
    """

    name = "kelly_regime_v8_ladder_bag"

    def __init__(self, ladder_bases: tuple[int, ...] = (18, 20, 22, 24, 26, 28),
                 band: float = 0.01, target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70, high_out: float = 1.20,
                 low_in: float = 0.55, low_out: float = 0.85) -> None:
        self.ladder_bases = tuple(int(b) for b in ladder_bases)
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        max_anchor_days = max(4 * b for b in self.ladder_bases)
        self.warmup = int(max_anchor_days * BARS_PER_DAY) + 10

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        idx = df.index
        n = len(df)
        r = np.log(close).diff()

        # ---- per-ladder latched vote fractions, each identical to kelly_regime.py's
        # own vote construction, run once per base period in ladder_bases ----
        ladder_fracs = []
        for base in self.ladder_bases:
            horizons = (base, 2 * base, 4 * base)
            votes = []
            for days in horizons:
                anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
                v = pd.Series(
                    np.where(close > anchor * (1.0 + self.band), 1.0,
                             np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                    index=idx,
                )
                votes.append(v.ffill().fillna(0.0))
            frac_k = (sum(votes) / len(votes)).to_numpy()
            ladder_fracs.append(frac_k)
        # plain, unweighted average across ladders -- the only new arithmetic,
        # and it operates on already-causal per-bar series.
        frac_bagged = np.mean(np.vstack(ladder_fracs), axis=0)

        # ---- v3's own conditional-vol-target scale, verbatim (see kelly_regime_v3.py) ----
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                    min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        scale = np.zeros(n)
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale[i] = full[i] if state != 0 else steady[i]

        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = frac_bagged[i] * scale[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["_frac_bagged"] = frac_bagged
        df["_scale"] = scale
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures


# ------------------------------------------------------------------------ harness

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")

INCUMBENT = "kelly_regime_v4"

# Ensemble definitions -- fixed in advance, not fitted. See module docstring.
LADDER_SETS = {
    "full6_18-28": (18, 20, 22, 24, 26, 28),
    "coarse3_18-23-28": (18, 23, 28),
    "edges2_18-28": (18, 28),
    "wide_negcontrol_14-21-28": (14, 21, 28),
}
PRIMARY = "full6_18-28"  # the a-priori "best/most-representative" candidate

N_EVALUATED = 0  # distinct ensemble configurations searched in step 3

OUT = ROOT / "reports" / "kelly_regime_v8_ladder_bag"


def mean_notional(result) -> float:
    if "target" not in result.df:
        return float("nan")
    tgt = np.abs(result.df["target"].to_numpy(dtype=float))
    return float(np.mean(np.clip(tgt, 0.0, result.market.leverage)))


def realized_vol(equity) -> float:
    eq = equity.to_numpy(dtype=float) if hasattr(equity, "to_numpy") else np.asarray(equity)
    if len(eq) < 3:
        return float("nan")
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
    return float(rets.std(ddof=1) * np.sqrt(BARS_PER_YEAR))


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0, count=False):
    """One backtest -> (metrics, realized vol, mean notional, result)."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                         start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    return m, realized_vol(result.equity), mean_notional(result), result


def line(tag, m, vol, notional, result):
    print(f"  {tag:44s} final=${m.final_balance:>11,.0f} "
          f"vol={vol:5.3f} notional={notional:5.3f} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# --------------------------------------------------------------------------- step 3


def sweep() -> pd.DataFrame:
    """Step 3: measure every fixed ensemble definition on inner-train, both markets, vs v4 control."""
    rows = []
    t0 = time.time()
    for label, bases in LADDER_SETS.items():
        for mi, (mname, market) in enumerate(MARKETS):
            strat = KellyRegimeV8LadderBag(ladder_bases=bases)
            m, vol, notional, res = measure(strat, *TRAIN, market=market, count=(mi == 0))
            rows.append({"label": label, "ladder_bases": str(bases), "market": mname,
                         "final": m.final_balance, "vol": vol, "notional": notional,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "trades": m.num_trades, "fees": m.fees_paid,
                         "liquidated": m.liquidated})
            print(f"[{N_EVALUATED:>2d}] {label:26s} {mname:8s} "
                  f"final=${m.final_balance:>10,.0f} DD={m.max_drawdown_pct:>5.1f}% "
                  f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
                  f"[{time.time() - t0:.0f}s]")
    # v4 control, same period, both markets
    for mname, market in MARKETS:
        m, vol, notional, res = measure(get_strategy(INCUMBENT), *TRAIN, market=market)
        rows.append({"label": "kelly_regime_v4_control", "ladder_bases": "(20,40,80)",
                     "market": mname, "final": m.final_balance, "vol": vol,
                     "notional": notional, "max_dd": m.max_drawdown_pct,
                     "sharpe": m.sharpe, "trades": m.num_trades, "fees": m.fees_paid,
                     "liquidated": m.liquidated})
        print(f"[ctl] {'kelly_regime_v4_control':26s} {mname:8s} "
              f"final=${m.final_balance:>10,.0f} DD={m.max_drawdown_pct:>5.1f}% "
              f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d}")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "sweep_inner_train.csv", index=False)
    print(f"\nconfigurations evaluated (step 3): {N_EVALUATED}")
    print(f"written: {OUT / 'sweep_inner_train.csv'}")
    return out


# --------------------------------------------------------------------------- step 5


def select() -> pd.DataFrame:
    """Step 5: score every ensemble definition on inner-validation, both markets, vs v4 control."""
    rows = []
    for label, bases in LADDER_SETS.items():
        for mname, market in MARKETS:
            strat = KellyRegimeV8LadderBag(ladder_bases=bases)
            m, vol, notional, res = measure(strat, *VALID, market=market)
            rows.append({"label": label, "ladder_bases": str(bases), "market": mname,
                         "final": m.final_balance, "vol": vol, "notional": notional,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "trades": m.num_trades, "fees": m.fees_paid,
                         "liquidated": m.liquidated})
        s = [r for r in rows if r["label"] == label and r["market"] == "spot"][-1]
        f = [r for r in rows if r["label"] == label and r["market"] == "futures"][-1]
        print(f"{label:26s} spot: ${s['final']:>9,.0f} DD{s['max_dd']:>5.1f}% "
              f"sh{s['sharpe']:>5.2f} tr{s['trades']:>4d}   "
              f"fut: ${f['final']:>9,.0f} DD{f['max_dd']:>5.1f}% "
              f"sh{f['sharpe']:>5.2f} tr{f['trades']:>4d}")
    for mname, market in MARKETS:
        m, vol, notional, res = measure(get_strategy(INCUMBENT), *VALID, market=market)
        rows.append({"label": "kelly_regime_v4_control", "ladder_bases": "(20,40,80)",
                     "market": mname, "final": m.final_balance, "vol": vol,
                     "notional": notional, "max_dd": m.max_drawdown_pct,
                     "sharpe": m.sharpe, "trades": m.num_trades, "fees": m.fees_paid,
                     "liquidated": m.liquidated})
    ctl_s = [r for r in rows if r["label"] == "kelly_regime_v4_control" and r["market"] == "spot"][-1]
    ctl_f = [r for r in rows if r["label"] == "kelly_regime_v4_control" and r["market"] == "futures"][-1]
    print(f"{'kelly_regime_v4 (control)':26s} spot: ${ctl_s['final']:>9,.0f} "
          f"DD{ctl_s['max_dd']:>5.1f}% sh{ctl_s['sharpe']:>5.2f} tr{ctl_s['trades']:>4d}   "
          f"fut: ${ctl_f['final']:>9,.0f} DD{ctl_f['max_dd']:>5.1f}% "
          f"sh{ctl_f['sharpe']:>5.2f} tr{ctl_f['trades']:>4d}")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "select_inner_validation.csv", index=False)
    print(f"\nwritten: {OUT / 'select_inner_validation.csv'}")
    return out


# ------------------------------------------------------------------------ failure mode (a)


def frac_correlation() -> None:
    """Failure mode (a): does the bagged frac barely differ from v4's own single-ladder frac?

    ``KellyRegimeV8LadderBag(ladder_bases=(20,))`` reduces exactly to v4's
    own (20,40,80) vote formula -- same class, one-ladder set -- so it is
    used here as the reference rather than re-deriving the formula.
    Restricted to pre-2023 bars only, per this session's data rule.
    """
    pre = DF.loc[:"2022-12-31"]
    ref = KellyRegimeV8LadderBag(ladder_bases=(20,)).prepare(pre.copy())["_frac_bagged"]
    print("frac correlation vs v4's own single-ladder (20,40,80) vote, pre-2023 bars:")
    for label, bases in LADDER_SETS.items():
        cand = KellyRegimeV8LadderBag(ladder_bases=bases).prepare(pre.copy())["_frac_bagged"]
        mask = np.isfinite(ref.to_numpy()) & np.isfinite(cand.to_numpy())
        corr = float(np.corrcoef(ref.to_numpy()[mask], cand.to_numpy()[mask])[0, 1])
        mean_abs_diff = float(np.mean(np.abs(ref.to_numpy()[mask] - cand.to_numpy()[mask])))
        verdict = "collapses to v4 (corr > 0.98)" if corr > 0.98 else "measurably different from v4"
        print(f"  {label:26s} corr={corr:.4f}  mean|diff|={mean_abs_diff:.4f}  {verdict}")


# --------------------------------------------------------------------------- failure mode (c)


def exposure_artifact_check() -> None:
    """Mandatory exposure-artifact check (ROUTINE.md standing rule, sharpened by R-33).

    Build a "flat-rescaled v4" comparator: v4's own unchanged target,
    multiplied by a single constant c chosen so its mean notional matches
    the candidate's mean notional over the SAME period. Report R^2 of the
    candidate's target series against that flat rescale, on inner-
    validation, both markets, for every ensemble definition. R^2 > 0.95
    means "this is the standard exposure-level artifact".
    """
    print("\nexposure-artifact check (inner-validation, mean-notional-matched flat rescale of v4):")
    for label, bases in LADDER_SETS.items():
        print(f" {label}:")
        for mname, market in MARKETS:
            cand = KellyRegimeV8LadderBag(ladder_bases=bases)
            m_c, vol_c, not_c, res_c = measure(cand, *VALID, market=market)
            v4 = get_strategy(INCUMBENT)
            m_v4, vol_v4, not_v4, res_v4 = measure(v4, *VALID, market=market)

            cand_t = res_c.df["target"].to_numpy(dtype=float)
            v4_t = res_v4.df["target"].reindex(res_c.df.index).to_numpy(dtype=float)
            c = not_c / not_v4 if not_v4 > 0 else float("nan")
            flat = c * v4_t

            mask = np.isfinite(cand_t) & np.isfinite(flat)
            x = flat[mask]
            y = cand_t[mask]
            ss_res = float(np.sum((y - x) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")

            verdict = ("EXPOSURE-LEVEL ARTIFACT (R^2 > 0.95)" if np.isfinite(r2) and r2 > 0.95
                        else "not a flat rescale by this test")
            print(f"    {mname}: cand notional={not_c:.3f} v4 notional={not_v4:.3f} c={c:.3f}  "
                  f"corr={corr:.4f}  R^2={r2:.4f}  {verdict}")


# ------------------------------------------------------------------------ causality


def causality() -> None:
    """Step 6: by-hand two-opposite-tampers lookahead probe, on the primary (full6) candidate.

    Same procedure as R-28/R-31/R-33/R-37/R-38: bars after a cut are
    multiplied by 3 in one copy, divided by 3 in another; every decision
    at or before the cut must be bit-identical. Restricted to strictly
    pre-2023 bars, per this session's data rule.
    """
    pre_2023 = DF.loc[:"2022-12-31"]
    df = pre_2023.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    strat_kwargs = dict(ladder_bases=LADDER_SETS[PRIMARY])

    def prepared(frame):
        return KellyRegimeV8LadderBag(**strat_kwargs).prepare(frame.copy())

    pa = prepared(up)
    pb = prepared(down)
    ok = True
    for col in ("target", "_frac_bagged", "_scale"):
        a = pa[col].to_numpy(dtype=float)[:cut]
        b = pb[col].to_numpy(dtype=float)[:cut]
        worst = float(np.nanmax(np.abs(a - b)))
        good = worst < 1e-9
        ok &= good
        print(f"  column={col:16s} max |difference| before the cut = {worst:.3e}  "
              f"{'PASS' if good else 'FAIL'}")

    from tradebot.broker import PaperBroker
    from tradebot.orders import Order

    def decisions(frame):
        s = KellyRegimeV8LadderBag(**strat_kwargs)
        prep = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        broker.execute(Order(target=0.1), prep.index[0], float(prep["open"].iloc[0]))
        out = []
        for i in bars:
            ctx = Context(prep, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down)) if oa != ob]
    ok &= not bad
    print(f"  orders {'match' if not bad else f'DIFFER at bars {bad}'} at the probe bars")

    a = run_backtest(KellyRegimeV8LadderBag(**strat_kwargs), up.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    b = run_backtest(KellyRegimeV8LadderBag(**strat_kwargs), down.iloc[:cut + 1], FUTURES,
                      1_000.0, data_label=LABEL)
    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-6
    print(f"  max |equity difference| before the cut = {worst_eq:.3e}  "
          f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")

    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Step 7: pre-registered falsification -- does every candidate hold on ETH?

    Same venue (Bitfinex), same window as R-17/R-28/R-31/R-33/R-37/R-38,
    both spot and 5x futures, every ensemble definition vs shipped v4
    defaults as the control, on both the BTC control run and the ETH test
    run of the identical pipeline -- this is whole-file, pre-2020 data,
    safe under this session's rule. Falsification rule (fixed before
    running): if a candidate is not at least comparable to v4 on ETH, or is
    visibly worse on ETH than on the BTC control run through the identical
    code, this direction fails.
    """
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            m_v4, vol_v4, not_v4, res_v4 = measure(get_strategy(INCUMBENT), None, None,
                                                    df=df, market=market)
            line(f"    {INCUMBENT} (control)", m_v4, vol_v4, not_v4, res_v4)
            for label, bases in LADDER_SETS.items():
                cand = KellyRegimeV8LadderBag(ladder_bases=bases)
                m_c, vol_c, not_c, res_c = measure(cand, None, None, df=df, market=market)
                line(f"    v8_ladder_bag[{label}]", m_c, vol_c, not_c, res_c)


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "sweep":
        sweep()
    elif choice == "select":
        select()
    elif choice == "fraccorr":
        frac_correlation()
    elif choice == "artifact":
        exposure_artifact_check()
    elif choice == "causality":
        causality()
    elif choice == "eth":
        eth()
    else:
        print("usage: python experiments/kelly_regime_v8_ladder_bag.py "
              "[sweep|select|fraccorr|artifact|causality|eth]")
