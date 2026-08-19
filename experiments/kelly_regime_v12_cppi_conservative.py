#!/usr/bin/env python
"""Replace kelly_regime_v4's vol-target SCALE with a CPPI cushion scale; keep the vote unchanged (SIZE axis).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

R-46 conservative branch. A second, independent "novel" branch is running
in parallel on a disjoint file at the same time (see ROUTINE.md's
parallel-round rules at the bottom) -- this file does not coordinate with
it and does not commit anything; the operator merges both after both
verdicts are in.

The idea
--------
Fourteen independent branches across nine rounds (R-34 through R-45, see
docs/LEDGER.md section C "Ruled out") have all tried to IMPROVE
``kelly_regime_v4``'s vote-and-scale mechanism -- retuning its constants,
bagging ladders, adding on-chain/basis/funding signals as gates or votes,
replacing the vol-target formula with CRRA or risk-constrained-Kelly
variants, robust/walk-forward re-estimation of its constants -- and every
one failed, either as an exposure-level artifact (R^2 > 0.95 against a
flat-rescaled v4) or by failing the pre-2020 ETH/BTC-control falsification
test. The ledger's most recent entry (R-45) explicitly recommends NOT
doing a fifteenth variation "on the incumbent's own vote-and-scale
mechanism, whatever axis it attacks."

So this branch keeps v4's vote ``frac`` completely UNCHANGED -- the exact
same three-anchor doubling-ladder crowd-regime vote, recomputed here
byte-identically from the same formula and the same default horizons, not
modified in any way -- and replaces only the OTHER half of v4's exposure
decision, the volatility-targeting SCALE (``target_vol / realized_vol``,
capped at ``max_leverage``, with v3's hysteresis latch), with a Constant
Proportion Portfolio Insurance (CPPI) cushion-based scale. CPPI is a much
older portfolio-construction mechanism from a different part of the
literature than anything tried in R-34 through R-45 (none of those touched
portfolio-insurance theory; all worked either inside the vol-targeting
formula itself or bolted a new predictive signal onto the vote).

Mechanism, one sentence: track a slowly-growing floor as a fixed fraction
of the strategy's own starting equity, size exposure in proportion to how
far realized equity sits above that floor (the "cushion"), and multiply
that cushion-fraction by v4's own unchanged crowd-regime vote -- so the
strategy delevers only when its own realized account has actually lost
money relative to a floor it can never re-inflate by trading, rather than
when instantaneous realized volatility rises (which on this asset,
R-46's own tested hypothesis below, sits on the wrong side of an already-
validated effect).

Citations
---------
- Perold, A. F. (1986). "Constant Proportion Portfolio Insurance."
  Unpublished manuscript, Harvard Business School -- the original CPPI
  formulation (cushion x fixed multiplier = risky-asset exposure),
  developed for institutional equity/bond allocation with a
  continuously-rebalanced floor guarantee. No transaction costs, no
  crypto-scale volatility, no leverage/funding: a theoretical allocation
  rule, not a backtested trading strategy on this kind of data.
- Perold, A. F. & Sharpe, W. F. (1988). "Dynamic Strategies for Asset
  Allocation." Financial Analysts Journal 44(1), 16-27. Formalizes CPPI
  alongside buy-and-hold and constant-mix as one of three canonical
  dynamic stock/bond allocation rules, and derives CPPI's defining
  "concave in down markets, convex in up markets" payoff shape (a
  levered call on the cushion) -- the same shape this branch's failure
  hypothesis below turns on. Equity/T-bill markets, no fees modeled.
- Black, F. & Jones, R. (1987). "Simplifying Portfolio Insurance."
  Journal of Portfolio Management 14(1), 48-51. Independent, contemporary
  derivation of the identical constant-multiplier cushion rule, and the
  source of "CPPI" as a named strategy in the practitioner literature
  (continuous-time GBM assumption, no discrete rebalancing cost).
- Ko, H., Son, B. & Lee, J. (2024). "Portfolio Insurance Strategy in the
  Cryptocurrency Market." Research in International Business and Finance
  67, 102135. The crypto-specific test this project's own falsification
  step should be read against: on BTC/ETH/major-cap daily data (no
  leverage, no funding, low assumed cost), CPPI-style insurance realizes
  a higher Omega ratio and lower downside risk than buy-and-hold, but
  their own reported mechanism is exactly the one this branch's failure
  hypothesis names -- CPPI underperforms buy-and-hold in strong bull
  runs because it de-risks after a drawdown and re-levers only slowly
  during the recovery. That paper trades daily bars with no leverage and
  no funding; this branch tests the same mechanism on 5m bars, 5x
  futures, with realistic taker fees and (via the existing eth()/BTC-
  control falsification) the same asset this project already has a
  validated volatility-effect finding for.

Constraint attacked: SIZE (costs/behavior that scale with the exposure
decision) -- specifically, this project's own repeated finding that "every
strategy that decides how much to hold makes money" holds for the SCALE
half of v4's mechanism, so this branch asks whether a structurally
different, older, well-established SCALE formula (not a retune of the
existing one, not a new predictive signal) does better or worse than
volatility-targeting at the one job that half of the mechanism does:
converting realized risk-bearing capacity into exposure.

Pre-registered falsification test (fixed before any code ran)
---------------------------------------------------------------
The project's standard one, verbatim from kelly_regime_v11_robust_ladder.py's
eth(): does the selected candidate at least match kelly_regime_v4 on the
pre-2020 BTC control window, and not visibly underperform v4 on ETH more
than it does on that BTC control -- both spot and 5x futures, identical
pipeline for both assets, using the project's OK/WORSE per-cell criterion
(d(Sharpe) > -0.05 and d(profit) > -2.0pp). If the candidate loses to v4 on
the BTC control, or is visibly worse on ETH than on the BTC control, this
direction fails.

Pre-registered failure hypothesis (named before any sweep ran)
------------------------------------------------------------------
CPPI's core mechanism cuts exposure exactly when the cushion is thin --
i.e. after realized losses, which on this dataset correlate with recent
high realized volatility (drawdowns ARE the high-vol episodes here). This
project's OWN validated finding, measured on this exact data in
kelly_regime_v3.py's docstring (Baur & Dimpfl 2018's inverse leverage
effect for BTC, re-measured on this series), is that high-volatility
states carry the HIGHEST forward Sharpe. So CPPI's mechanism should be
expected to fight an edge this project has already found real: predicting
it stays under-invested through recoveries and underperforms v4, possibly
decisively. This is tested directly in ``recovery_windows()`` below by
reporting the candidate's ``cppi_scale`` against v4's own ``scale`` in the
30/60/90-bar windows immediately following each of the largest inner-
validation drawdown troughs. Confirming this hypothesis is a legitimate,
expected negative result, not a failure of execution -- it is reported
even (especially) if it confirms the hypothesis named here in advance.

Not a duplicate of
-------------------
- R-34/R-35/R-38/R-41 (Bayesian posterior, funding gate, risk-constrained
  Kelly cap, CRRA fraction, basis brake/lead): all changed the SCALE
  formula's inputs (added a signal/gate) while keeping a volatility-
  targeting-family functional form. This branch changes the functional
  FORM of the scale itself -- from "target risk / realized risk" to
  "multiplier x (equity - floor) / equity" -- a mechanism with no
  realized-volatility term in it at all.
- R-37/R-40/R-11 (target_vol/max_leverage retune, ladder bagging, cross-
  fold robust ladder/target_vol/max_leverage reselection): all retuned or
  reselected the EXISTING vol-targeting formula's own constants or its
  ladder. This branch does not touch target_vol, vol_span, or the vote's
  ladder/deadband at all; it removes the vol-targeting formula and
  replaces it with a portfolio-insurance rule that has never appeared in
  this project.
- R-45 (this lineage's most recent entry, cross-fold minimax reselection
  of the ladder + target_vol/max_leverage pair): explicitly recommended
  against a fifteenth retune of v4's OWN mechanism. This branch is not a
  retune -- it is a substitution of an entirely different, independently-
  sourced (Perold 1986; Perold & Sharpe 1988; Black & Jones 1987) sizing
  rule for the half of v4 that decides notional, while leaving the vote
  (the OTHER half, which decides direction/conviction) untouched.
- kelly_regime_v3 (the incumbent's own conditional/hysteresis vol-
  targeting): CPPI's floor is a fixed function of ELAPSED TIME, never of
  a trailing peak or of realized volatility, and is deliberately NOT a
  high-water-mark floor -- seed to avoid exactly the reactive
  peak-following, deleverage-after-drawdown behavior v3's docstring's own
  cited finding (Baur & Dimpfl 2018) argues against. A peak-following
  floor would directly reproduce v3's already-diagnosed conflict; the
  textbook slowly-growing floor used here does not, by construction.

Search space (fixed in advance, not widened after seeing results)
--------------------------------------------------------------------
- F0 (floor as a fraction of the strategy's own starting balance) in
  {0.50, 0.65, 0.80}
- g (floor annual growth rate) in {0.00, 0.03}
- m (CPPI multiplier) in {3, 4, 5, 6}
- 3 x 2 x 4 = 24 distinct configurations. Everything else (the vote
  itself: horizons, band, vote_gamma; the deadband) stays at v4's shipped
  defaults. max_leverage is an explicit constructor parameter, held fixed
  at v4's own default (2.0) across the whole grid -- it is the hard
  exposure ceiling this project's futures market already uses, not a
  CPPI-specific axis, so it is not swept.

Data rule for this session (self-verified at the end)
----------------------------------------------------------
Iterate ONLY on inner-train (2017-01-01 to 2020-12-31) and inner-
validation (2021-01-01 to 2022-12-31). No bar dated 2023-01-01 or later is
read anywhere in this file -- grep this file for date literals to check;
the ``eth()`` falsification step deliberately loads whole-file pre-2020
BTC/ETH control data, which is safe under this rule and is the project's
standard falsification data source (see kelly_regime_v11_robust_ladder.py).

Usage
-----
    python experiments/kelly_regime_v12_cppi_conservative.py sweep       # step 3: grid on inner-train + inner-validation
    python experiments/kelly_regime_v12_cppi_conservative.py select      # step 4/5: pick candidate, plateau, TRAIN/VALID vs v4
    python experiments/kelly_regime_v12_cppi_conservative.py artifact    # exposure-artifact R^2 check
    python experiments/kelly_regime_v12_cppi_conservative.py causality   # step 6: tamper probe
    python experiments/kelly_regime_v12_cppi_conservative.py eth         # step 7: ETH/BTC-control falsification
    python experiments/kelly_regime_v12_cppi_conservative.py recovery    # failure-hypothesis check (point 6)
    python experiments/kelly_regime_v12_cppi_conservative.py all         # everything, in order
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
from tradebot.strategy import Context  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


# --------------------------------------------------------------------- strategy


class KellyRegimeV12CPPIConservative(KellyRegimeV4):
    """v4's crowd-regime vote, gated by a CPPI cushion scale instead of vol-targeting.

    ``prepare()`` recomputes v4's own vote formula byte-identically
    (same anchors, same band, same hysteresis-free rolling-mean logic) and
    stores it as ``frac`` -- nothing about the vote changes. It does NOT
    compute ``target``: unlike v3/v4, the scale here depends on realized
    EQUITY, which is only known causally bar-by-bar inside the backtest
    loop, not from the price frame alone. So ``on_bar`` is overridden to
    build the CPPI floor/cushion/scale state itself, using ``ctx.equity``
    (the engine's own causal running-equity figure, recorded from the
    broker BEFORE this bar's on_bar call, so it can never see this bar's
    own order) and ``ctx.i`` (elapsed bars, for the floor's time-growth
    term). State is reset at the start of ``prepare()`` so a reused
    instance across multiple backtests (as the sweep functions below do)
    never carries stale floor/position state between runs.

    Inherited-but-unused: ``target_vol``, ``vol_span``, ``anchor_span_days``
    and the high/low hysteresis thresholds are all set by the v3/v4
    constructor chain this class still calls (for the vote's shared
    plumbing) but play no role in this class's own ``prepare()``/``on_bar()``
    -- CPPI's scale formula has no realized-volatility term at all.
    """

    name = "kelly_regime_v12_cppi_conservative"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 deadband: float = 0.10, vote_gamma: float = 1.0,
                 F0: float = 0.65, g: float = 0.03, m: float = 4.0,
                 max_leverage: float = 2.0, **kwargs) -> None:
        super().__init__(horizons=horizons, band=band, deadband=deadband,
                          vote_gamma=vote_gamma, max_leverage=max_leverage, **kwargs)
        self.F0 = F0
        self.g = g
        self.m = m
        self._t0: int | None = None
        self._floor0: float | None = None
        self._pos: float = 0.0
        self._last_sent: float | None = None
        # Timestamp-keyed logs of this instance's own actual causal
        # decisions, populated by on_bar() as it runs -- used afterward by
        # target_series()/scale_series() so every downstream check (the
        # exposure-artifact R^2, the recovery-window comparison, notional
        # reporting) reads what on_bar() truly decided rather than a
        # second, possibly-diverging reimplementation of the formula.
        self._pos_log: dict = {}
        self._scale_log: dict = {}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """v4's own vote, recomputed unchanged; no scale/target column here (see class docstring)."""
        close = df["close"]
        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma
        df["frac"] = frac

        # Reset causal CPPI state for THIS run -- prepare() runs exactly
        # once per run_backtest call, before any on_bar, so this is the
        # correct per-run reset hook even if the instance is reused.
        self._t0 = None
        self._floor0 = None
        self._pos = 0.0
        self._last_sent = None
        self._pos_log = {}
        self._scale_log = {}
        return df

    def on_bar(self, ctx: Context) -> None:
        equity = ctx.equity  # broker's running equity, recorded before this bar's own order
        if self._t0 is None:
            # First live bar: no order has filled yet (run_backtest guarantees
            # equity == start_balance up to and including this call), so this
            # is exactly "the strategy's own starting balance", captured
            # causally rather than passed in from outside.
            self._t0 = ctx.i
            self._floor0 = self.F0 * equity

        t_years = (ctx.i - self._t0) / BARS_PER_YEAR
        floor = self._floor0 * (1.0 + self.g) ** t_years
        cushion = max(equity - floor, 0.0)
        cppi_scale = 0.0
        if equity > 0:
            cppi_scale = min(max(self.m * cushion / equity, 0.0), self.max_leverage)

        frac = float(ctx.bar["frac"])
        desired = frac * cppi_scale
        if abs(desired - self._pos) > self.deadband:
            self._pos = desired

        self._scale_log[ctx.ts] = cppi_scale
        self._pos_log[ctx.ts] = self._pos

        if self._last_sent is None or abs(self._pos - self._last_sent) > 1e-9:
            ctx.order_notional(self._pos)  # fraction of equity: same risk on spot and futures
            self._last_sent = self._pos


def target_series(strategy: "KellyRegimeV12CPPIConservative", index: pd.Index) -> np.ndarray:
    """The candidate's own actually-recorded causal position at every bar of ``index``.

    Read directly from ``strategy._pos_log``, populated by that instance's
    own ``on_bar()`` during the run just completed -- not a second
    reconstruction of the formula, so it cannot silently diverge from what
    ``on_bar()`` actually decided. Bars before the strategy's first live
    call (warmup, or a cold-started period's pre-trade prefix) get 0.0:
    the account is genuinely flat there (``run_backtest`` guarantees
    equity stays at ``start_balance`` until the first order fills),
    matching every other strategy's own flat-during-warmup convention.
    """
    if not strategy._pos_log:
        return np.zeros(len(index))
    return pd.Series(strategy._pos_log).reindex(index).fillna(0.0).to_numpy(dtype=float)


def scale_series(strategy: "KellyRegimeV12CPPIConservative", index: pd.Index) -> np.ndarray:
    """The candidate's own actually-recorded ``cppi_scale`` (pre-vote, pre-deadband) at every bar."""
    if not strategy._scale_log:
        return np.full(len(index), np.nan)
    return pd.Series(strategy._scale_log).reindex(index).to_numpy(dtype=float)


# ------------------------------------------------------------------------ harness

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
# OOS_START = "2023-01-01"  -- NEVER read in this file, by construction.

INCUMBENT = "kelly_regime_v4"

F0_GRID = (0.50, 0.65, 0.80)
G_GRID = (0.00, 0.03)
M_GRID = (3.0, 4.0, 5.0, 6.0)
CONFIGS = [(f0, g, m) for f0 in F0_GRID for g in G_GRID for m in M_GRID]

OUT = ROOT / "reports" / "kelly_regime_v12_cppi_conservative"

_SEEN: set[tuple] = set()  # distinct configurations evaluated, for the trials count


def make_strategy(F0: float, g: float, m: float) -> KellyRegimeV12CPPIConservative:
    _SEEN.add((F0, g, m))
    return KellyRegimeV12CPPIConservative(F0=F0, g=g, m=m)


def mean_notional(result, strategy=None) -> float:
    """Mean |exposure|, clipped to the market's leverage cap.

    v3/v4 write their causal decision to a ``target`` column in
    ``prepare()``, so it is read straight from ``result.df``. The CPPI
    candidate has no such column (see class docstring) -- for it, pass
    ``strategy`` and its own recorded ``_pos_log`` is used instead via
    ``target_series()``.
    """
    if strategy is not None and hasattr(strategy, "_pos_log"):
        tgt = np.abs(target_series(strategy, result.df.index))
    elif "target" in result.df:
        tgt = np.abs(result.df["target"].to_numpy(dtype=float))
    else:
        return float("nan")
    return float(np.mean(np.clip(tgt, 0.0, result.market.leverage)))


def realized_vol(equity) -> float:
    eq = equity.to_numpy(dtype=float) if hasattr(equity, "to_numpy") else np.asarray(equity)
    if len(eq) < 3:
        return float("nan")
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
    sd = np.std(rets, ddof=1)
    return float(sd * np.sqrt(BARS_PER_YEAR)) if np.isfinite(sd) else float("nan")


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0):
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                         start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    return m, realized_vol(result.equity), mean_notional(result, strategy), result


def line(tag, m, vol, notional, result) -> None:
    print(f"  {tag:44s} final=${m.final_balance:>11,.0f} "
          f"vol={vol:5.3f} notional={notional:5.3f} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# --------------------------------------------------------------------------- step 3


def sweep() -> pd.DataFrame:
    """Evaluate every configuration on inner-train and inner-validation, both markets."""
    rows = []
    t0 = time.time()
    n = 0
    for F0, g, m in CONFIGS:
        n += 1
        for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
            strat = make_strategy(F0, g, m)
            for mname, market in MARKETS:
                mm, vol, notional, res = measure(strat, start, end, market=market)
                rows.append({"F0": F0, "g": g, "m": m, "split": split_name, "market": mname,
                             "final": mm.final_balance, "profit_pct": mm.profit_pct,
                             "vol": vol, "mean_notional": notional,
                             "max_dd": mm.max_drawdown_pct, "sharpe": mm.sharpe,
                             "trades": mm.num_trades, "liquidated": mm.liquidated})
        print(f"[{n:>2d}/{len(CONFIGS)}] F0={F0:.2f} g={g:.2f} m={m:.1f}  "
              f"[{time.time() - t0:.0f}s]")
    # v4 control, same splits, both markets
    for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        for mname, market in MARKETS:
            mm, vol, notional, res = measure(get_strategy(INCUMBENT), start, end, market=market)
            rows.append({"F0": "v4_control", "g": "", "m": "", "split": split_name, "market": mname,
                         "final": mm.final_balance, "profit_pct": mm.profit_pct,
                         "vol": vol, "mean_notional": notional,
                         "max_dd": mm.max_drawdown_pct, "sharpe": mm.sharpe,
                         "trades": mm.num_trades, "liquidated": mm.liquidated})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "sweep.csv", index=False)
    print(f"\nsweep: {len(CONFIGS)} configurations x 2 splits x 2 markets "
          f"= {len(CONFIGS) * 2 * 2} backtests, [{time.time() - t0:.0f}s]")
    print(f"written: {OUT / 'sweep.csv'}")
    return out


# --------------------------------------------------------------------------- step 4/5


def select() -> None:
    """Select on inner-validation Sharpe (spot), report plateau, then TRAIN/VALID vs v4 control."""
    OUT.mkdir(parents=True, exist_ok=True)
    scsv = OUT / "sweep.csv"
    sdf = pd.read_csv(scsv) if scsv.exists() else sweep()

    valid_spot = sdf[(sdf.market == "spot") & (sdf.split == "inner-validation")
                      & (sdf.F0 != "v4_control")].copy()
    valid_spot["F0"] = valid_spot["F0"].astype(float)
    valid_spot["g"] = valid_spot["g"].astype(float)
    valid_spot["m"] = valid_spot["m"].astype(float)
    valid_spot = valid_spot.sort_values("sharpe", ascending=False)

    print("=== top 8 configurations by inner-validation Sharpe (spot) ===")
    print(valid_spot[["F0", "g", "m", "sharpe", "profit_pct", "max_dd", "trades"]]
          .head(8).to_string(index=False))

    winner = valid_spot.iloc[0]
    s_F0, s_g, s_m = float(winner.F0), float(winner.g), float(winner.m)
    print(f"\nselected candidate: F0={s_F0}, g={s_g}, m={s_m}  "
          f"inner-validation Sharpe (spot)={winner.sharpe:.3f}")

    # ---- plateau neighbourhood around the winner ----
    f0_idx = list(F0_GRID).index(s_F0)
    g_idx = list(G_GRID).index(s_g)
    m_idx = list(M_GRID).index(s_m)
    f0_nb = F0_GRID[max(0, f0_idx - 1): f0_idx + 2]
    g_nb = G_GRID[max(0, g_idx - 1): g_idx + 2]
    m_nb = M_GRID[max(0, m_idx - 1): m_idx + 2]
    nb = valid_spot[valid_spot.F0.isin(f0_nb) & valid_spot.g.isin(g_nb) & valid_spot.m.isin(m_nb)]
    print(f"\n=== plateau neighbourhood (F0 in {f0_nb}, g in {g_nb}, m in {m_nb}) ===")
    print(nb[["F0", "g", "m", "sharpe", "profit_pct", "max_dd"]].to_string(index=False))
    print(f"Sharpe range across neighbourhood: [{nb.sharpe.min():.3f}, {nb.sharpe.max():.3f}]  "
          f"(project noise floor is +/-0.2 Sharpe, R-20)")

    # ---- full grid range, for context ----
    print(f"\nfull grid Sharpe range (inner-validation, spot): "
          f"[{valid_spot.sharpe.min():.3f}, {valid_spot.sharpe.max():.3f}]")

    # ---- v4 control, same splits ----
    v4_valid = sdf[(sdf.market == "spot") & (sdf.split == "inner-validation")
                    & (sdf.F0 == "v4_control")].iloc[0]
    print(f"\nkelly_regime_v4 control, inner-validation, spot: "
          f"sharpe={v4_valid.sharpe:.3f} profit={v4_valid.profit_pct:+.1f}% "
          f"maxDD={v4_valid.max_dd:.1f}%")

    # ---- TRAIN / VALID continuous-window report, candidate vs v4 control, both markets ----
    print(f"\n=== inner-train (2017-2020) and inner-validation (2021-2022), "
          f"candidate vs kelly_regime_v4 control ===")
    tv_rows = []
    for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        for mname, market in MARKETS:
            m_c, vol_c, not_c, res_c = measure(
                KellyRegimeV12CPPIConservative(F0=s_F0, g=s_g, m=s_m), start, end, market=market)
            m_v, vol_v, not_v, res_v = measure(get_strategy(INCUMBENT), start, end, market=market)
            line(f"  {split_name}/{mname} candidate", m_c, vol_c, not_c, res_c)
            line(f"  {split_name}/{mname} v4 control", m_v, vol_v, not_v, res_v)
            tv_rows.append({"split": split_name, "market": mname, "arm": "candidate",
                            "final": m_c.final_balance, "profit_pct": m_c.profit_pct,
                            "sharpe": m_c.sharpe, "max_dd": m_c.max_drawdown_pct,
                            "vol": vol_c, "mean_notional": not_c, "trades": m_c.num_trades})
            tv_rows.append({"split": split_name, "market": mname, "arm": "v4_control",
                            "final": m_v.final_balance, "profit_pct": m_v.profit_pct,
                            "sharpe": m_v.sharpe, "max_dd": m_v.max_drawdown_pct,
                            "vol": vol_v, "mean_notional": not_v, "trades": m_v.num_trades})
    pd.DataFrame(tv_rows).to_csv(OUT / "train_valid_candidate_vs_v4.csv", index=False)

    # Trials count: derived from the persisted sweep file's own distinct
    # (F0, g, m) rows rather than the in-process _SEEN set, so it is
    # correct whether `sweep` and `select` run in one process (`all`) or
    # as separate invocations (`_SEEN` does not survive across processes).
    n_configs = int(sdf[sdf.F0 != "v4_control"]
                     .drop_duplicates(subset=["F0", "g", "m"]).shape[0])
    (OUT / "selected_config.txt").write_text(
        f"F0={s_F0}\ng={s_g}\nm={s_m}\nn_configs_evaluated={n_configs}\n")
    print(f"\ndistinct configurations evaluated in total: {n_configs}")
    print(f"wrote {OUT / 'selected_config.txt'}")


def _load_selected() -> tuple[float, float, float]:
    cfg = OUT / "selected_config.txt"
    if cfg.exists():
        kv = dict(line.split("=", 1) for line in cfg.read_text().splitlines() if "=" in line)
        return float(kv["F0"]), float(kv["g"]), float(kv["m"])
    print("(no selected_config.txt yet -- run `select` first; falling back to a grid midpoint)")
    return 0.65, 0.00, 4.0


# --------------------------------------------------------------------------- exposure artifact


def exposure_artifact_check() -> None:
    """Mandatory exposure-artifact check (ROUTINE.md standing rule, sharpened by R-33).

    Build a "flat-rescaled v4" comparator: v4's own unchanged target,
    multiplied by a single constant c chosen so its mean notional matches
    the candidate's mean notional over the SAME period. Report R^2 of the
    candidate's target series against that flat rescale, on inner-
    validation, both markets. R^2 > 0.95 means "this is the standard
    exposure-level artifact". The candidate itself does not write a
    "target" column in prepare() (see class docstring); its causal
    decision is read from ``target_series()``, which reads the candidate
    instance's own ``_pos_log`` -- populated by its actual ``on_bar()``
    calls during the run just completed, not a separate reimplementation
    of the formula.
    """
    F0, g, m = _load_selected()
    print(f"\nexposure-artifact check (inner-validation, mean-notional-matched flat rescale of v4)")
    print(f"candidate: F0={F0} g={g} m={m}")
    for mname, market in MARKETS:
        cand = KellyRegimeV12CPPIConservative(F0=F0, g=g, m=m)
        m_c, vol_c, not_c, res_c = measure(cand, *VALID, market=market)
        v4 = get_strategy(INCUMBENT)
        m_v4, vol_v4, not_v4, res_v4 = measure(v4, *VALID, market=market)

        cand_t = target_series(cand, res_c.df.index)
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
        print(f"  {mname}: cand notional={not_c:.3f} v4 notional={not_v4:.3f} c={c:.3f}  "
              f"corr={corr:.4f}  R^2={r2:.4f}  {verdict}")
        print(f"    cand realized vol={vol_c:.3f}  v4 realized vol={vol_v4:.3f}  "
              f"cand sharpe={m_c.sharpe:.3f}  v4 sharpe={m_v4.sharpe:.3f}")


# ------------------------------------------------------------------------ causality


def causality() -> None:
    """Step 6: by-hand two-opposite-tampers lookahead probe on the selected candidate.

    Same procedure as R-28/R-31/R-33/R-37/R-38/R-40/R-45: bars after a cut
    are multiplied by 3 in one copy, divided by 3 in another; every
    decision at or before the cut must be bit-identical. This class
    introduces a NEW code path relative to every prior branch in this
    lineage -- an equity-dependent stateful loop inside on_bar(), rather
    than a target column precomputed in prepare() -- so this probe is
    especially load-bearing here, not a formality. It checks the "frac"
    column from prepare() (must be untouched, since the vote is
    unchanged), the order decisions on_bar() emits, AND the resulting
    equity path itself, since a leak in the CPPI state loop could show up
    only in accumulated equity rather than in any single order.
    Restricted to strictly pre-2023 bars, per this session's data rule.
    """
    from tradebot.broker import PaperBroker
    from tradebot.orders import Order

    F0, g, m = _load_selected()

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

    def build():
        return KellyRegimeV12CPPIConservative(F0=F0, g=g, m=m)

    print(f"probing candidate: F0={F0} g={g} m={m}")

    pa = build().prepare(up.copy())
    pb = build().prepare(down.copy())
    ok = True
    worst_col = float(np.nanmax(np.abs(pa["frac"].to_numpy(dtype=float)[:cut]
                                        - pb["frac"].to_numpy(dtype=float)[:cut])))
    good = worst_col < 1e-9
    ok &= good
    print(f"  column=frac  max |difference| before the cut = {worst_col:.3e}  "
          f"{'PASS' if good else 'FAIL'}")

    def decisions(frame):
        s = build()
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        broker.execute(Order(target=0.1), prepared.index[0], float(prepared["open"].iloc[0]))
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down)) if oa != ob]
    ok &= not bad
    print(f"  orders {'match' if not bad else f'DIFFER at bars {bad}'} at the probe bars")

    a = run_backtest(build(), up.iloc[:cut + 1], FUTURES, 1_000.0, data_label=LABEL)
    b = run_backtest(build(), down.iloc[:cut + 1], FUTURES, 1_000.0, data_label=LABEL)
    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-6
    print(f"  max |equity difference| before the cut = {worst_eq:.3e}  "
          f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")

    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Step 7: pre-registered falsification -- does the selected candidate hold on ETH?

    Same venue (Bitfinex), same pre-2020 window R-17/R-28/R-31/R-33/R-37/
    R-38/R-40/R-45 used, both spot and 5x futures, candidate vs shipped v4
    defaults as the control, on both the BTC control run and the ETH test
    run of the identical pipeline -- whole-file, pre-2020 data, safe under
    this session's rule. Falsification rule (fixed before running): if the
    candidate is not at least comparable to v4 on ETH, or is visibly worse
    on ETH than on the BTC control run through the identical code, this
    direction fails.
    """
    F0, g, m = _load_selected()
    print(f"candidate: F0={F0} g={g} m={m}")

    rows = []
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            cand = KellyRegimeV12CPPIConservative(F0=F0, g=g, m=m)
            m_c, vol_c, not_c, res_c = measure(cand, None, None, df=df, market=market)
            line(f"    candidate (v12)", m_c, vol_c, not_c, res_c)
            m_v4, vol_v4, not_v4, res_v4 = measure(get_strategy(INCUMBENT), None, None,
                                                    df=df, market=market)
            line(f"    {INCUMBENT} (control)", m_v4, vol_v4, not_v4, res_v4)
            rows.append({"asset": asset, "market": mname, "arm": "candidate",
                         "final": m_c.final_balance, "profit_pct": m_c.profit_pct,
                         "sharpe": m_c.sharpe, "max_dd": m_c.max_drawdown_pct,
                         "vol": vol_c, "liquidated": m_c.liquidated})
            rows.append({"asset": asset, "market": mname, "arm": "v4_control",
                         "final": m_v4.final_balance, "profit_pct": m_v4.profit_pct,
                         "sharpe": m_v4.sharpe, "max_dd": m_v4.max_drawdown_pct,
                         "vol": vol_v4, "liquidated": m_v4.liquidated})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "eth_falsification.csv", index=False)

    print("\n=== falsification verdict, candidate vs v4 control ===")
    verdict_ok = True
    for asset in ("BTC (control)", "ETH (test)"):
        for mname, _ in MARKETS:
            c = out[(out.asset == asset) & (out.market == mname) & (out.arm == "candidate")].iloc[0]
            d = out[(out.asset == asset) & (out.market == mname) & (out.arm == "v4_control")].iloc[0]
            d_sharpe = c.sharpe - d.sharpe
            d_profit = c.profit_pct - d.profit_pct
            d_dd = c.max_dd - d.max_dd
            ok = d_sharpe > -0.05 and d_profit > -2.0
            verdict_ok &= ok if "ETH" in asset else True
            print(f"  {asset:16s} {mname:8s} d(Sharpe)={d_sharpe:+.3f} "
                  f"d(profit)={d_profit:+.1f}pp d(maxDD)={d_dd:+.1f}pp  "
                  f"{'OK' if ok else 'WORSE'}")
    print(f"\nETH falsification: {'PASS' if verdict_ok else 'FAIL'}")
    print(f"wrote {OUT / 'eth_falsification.csv'}")


# ------------------------------------------------------------------------- recovery


def recovery_windows() -> None:
    """Point 6: test the pre-registered failure hypothesis directly.

    On inner-validation (spot), find the largest drawdown episodes in the
    candidate's OWN equity path, then compare the candidate's cppi_scale
    against v4's own vol-target scale in the 30/60/90-bar-per-day windows
    immediately following each trough (i.e. during the recovery). The
    candidate's scale is read from its own ``_scale_log`` (populated by its
    actual ``on_bar()`` calls, not a reimplementation); v4's own scale is
    recovered from its "target"/"frac"-equivalent ratio (v4 does not store
    scale separately, so it is recovered as target / frac where frac > 0,
    undefined -- and skipped -- while frac == 0, since scale is then
    unobservable from target alone).
    """
    F0, g, m = _load_selected()
    print(f"candidate: F0={F0} g={g} m={m}\n")

    cand = KellyRegimeV12CPPIConservative(F0=F0, g=g, m=m)
    m_c, vol_c, not_c, res_c = measure(cand, *VALID, market=SPOT)
    v4 = get_strategy(INCUMBENT)
    m_v4, vol_v4, not_v4, res_v4 = measure(v4, *VALID, market=SPOT)

    eq = res_c.equity.to_numpy(dtype=float)
    cand_scale = scale_series(cand, res_c.equity.index)

    # v4's own scale, recovered from its own target/frac (frac is v4's vote,
    # recomputed with v4's own shipped horizons/band, its own formula)
    v4_target = res_v4.df["target"].to_numpy(dtype=float)
    v4_close = res_v4.df["close"]
    v4_votes = []
    for days in v4.horizons:
        anchor = v4_close.rolling(int(days * BARS_PER_DAY)).mean()
        vv = pd.Series(
            np.where(v4_close > anchor * (1.0 + v4.band), 1.0,
                     np.where(v4_close < anchor * (1.0 - v4.band), 0.0, np.nan)),
            index=v4_close.index)
        v4_votes.append(vv.ffill().fillna(0.0))
    v4_frac = (sum(v4_votes) / len(v4_votes)).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        v4_scale = np.where(v4_frac > 1e-9, v4_target / v4_frac, np.nan)

    # find the 3 largest drawdown troughs in the candidate's equity path
    peaks = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peaks > 0, (peaks - eq) / peaks, 0.0)
    troughs = []
    remaining = dd.copy()
    for _ in range(3):
        i = int(np.argmax(remaining))
        if remaining[i] <= 0:
            break
        troughs.append(i)
        lo, hi = max(0, i - 30 * BARS_PER_DAY), min(len(remaining), i + 30 * BARS_PER_DAY)
        remaining[lo:hi] = -1.0  # exclude neighbourhood so the next pick is a distinct episode
    troughs.sort()

    print("=== recovery-window under-exposure check (inner-validation, spot) ===")
    print(f"(candidate mean scale over full window={np.nanmean(cand_scale):.3f}, "
          f"v4 mean scale over full window={np.nanmean(v4_scale):.3f})\n")
    any_underexposed = False
    for k, i in enumerate(troughs, 1):
        ts = res_c.equity.index[i]
        print(f"trough {k}: bar {i} ({ts:%Y-%m-%d}), drawdown={dd[i]*100:.1f}%")
        for days, label in ((30, "30d"), (60, "60d"), (90, "90d")):
            lo = i
            hi = min(len(cand_scale), i + days * BARS_PER_DAY)
            c_mean = float(np.nanmean(cand_scale[lo:hi]))
            v_mean = float(np.nanmean(v4_scale[lo:hi]))
            under = c_mean < v_mean
            any_underexposed |= under
            print(f"    +{label:>4s} recovery window: candidate cppi_scale={c_mean:.3f}  "
                  f"v4 scale={v_mean:.3f}  {'candidate UNDER-EXPOSED' if under else 'candidate not under-exposed'}")
        print()

    print(f"candidate sharpe={m_c.sharpe:.3f} (DD={m_c.max_drawdown_pct:.1f}%)  "
          f"v4 sharpe={m_v4.sharpe:.3f} (DD={m_v4.max_drawdown_pct:.1f}%)")
    print(f"\nfailure hypothesis {'CONFIRMED' if any_underexposed else 'NOT CONFIRMED'} by this check: "
          f"candidate is {'' if any_underexposed else 'not '}measurably under-exposed relative to v4 "
          f"in at least one post-trough recovery window.")


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    cmds = {"sweep": sweep, "select": select, "artifact": exposure_artifact_check,
            "causality": causality, "eth": eth, "recovery": recovery_windows}

    def all_() -> None:
        sweep()
        select()
        exposure_artifact_check()
        causality()
        eth()
        recovery_windows()

    cmds["all"] = all_
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python {sys.argv[0]} [{'|'.join(cmds)}]")
