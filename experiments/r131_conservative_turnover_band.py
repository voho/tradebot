#!/usr/bin/env python
"""R-131 CONSERVATIVE branch: ``ConservativeTurnoverBand`` -- a hard admissible

turnover CORRIDOR wrapped around ``kelly_regime_v4``'s own already-decided
target: while v4's own causal trailing-turnover EWM of THIS strategy's own
realized rebalances sits inside ``[0, TURNOVER_UPPER]``, rebalance exactly as
v4 does today; once trailing turnover reaches the corridor's upper edge,
DEFER (skip) the pending rebalance, unless a pre-registered override fires.

Full literature grounding (Khubiev, Semenov, Podlipnova & Khubieva
2025/2026, arXiv:2509.04541; Boyd, Busseti, Diamond, Kahn, Koh, Nystrup &
Speth 2017), the non-duplication argument against ``kelly_regime_ev`` /
Gârleanu-Pedersen / Almgren-Chriss / width-profile banding / the
``hedge_experts`` band rounds, the named failure modes, and the
pre-registered decision rule / falsification test all live in
``experiments/r131_shared.py``'s own module docstring (read in full before
this file was written); not re-derived here beyond the summary above. This
file NEVER edits ``r131_shared.py`` (frozen, shared with the parallel NOVEL
branch, a disjoint file this session does not touch), never edits
``src/tradebot/strategies/kelly_regime_v4.py``, and never reads a bar at or
after ``r131_shared.OOS_START`` (2023-01-01) from any data source.

MECHANISM (exact). ``prepare()`` is INHERITED, byte-identical, from
``KellyRegimeV4`` (no override at all -- this file never re-derives v4's
vote/hysteresis/deadband algebra, per the round's own binding instruction to
reuse v4's target/deadband logic "unchanged"). ``target[i]`` is therefore
EXACTLY what v4 would fill to at bar ``i`` if it traded there -- the
counterfactual used by the stress-episode diagnostic below.

``on_bar`` is a stateful, per-bar decision that cannot be precomputed in
``prepare()`` (whether bar ``i`` defers depends on whether bar ``i-1``
deferred, which changes the strategy's own realized-event history, which
feeds the trailing turnover the decision at ``i`` reads):

    # a new v4 DECISION arrives exactly when v4's own (already-deadbanded)
    # target column changes -- v4's own on_bar trigger, byte-identical
    # condition to KellyRegime.on_bar's `abs(t - prev) > 1e-9` check.
    if |ctx.bar["target"] - ctx.prev["target"]| > 1e-9:
        owed_target = ctx.bar["target"]; owed = True

    current = ctx.position * ctx.close / ctx.equity   # actual held fraction
    trailing = <causal EWM of realized rebalance events at bars < i>
    if owed:
        if trailing < TURNOVER_UPPER:
            EXECUTE  (ctx.order_notional(owed_target); event=1; owed=False)
        else:
            override = (owed_target == 0 and current != 0)                    # (a) full exit
                       or (|owed_target - current| > OVERRIDE_MULT*TURNOVER_UPPER)  # (b) large move
            EXECUTE (owed=False) if override else DEFER (owed stays True)

**Why "owed", not a live current-vs-desired comparison.** An earlier
version of this file compared ``ctx.position*ctx.close/ctx.equity`` (the
actual held fraction) against ``desired`` every bar with a ``1e-9``
tolerance and called that "pending." That is wrong and was caught by A2
before any B-step number was trusted: exposure fraction drifts with price
even while base-asset quantity is fixed (equity = cash + position*close, so
their ratio moves with price whenever ``cash != 0``), so the 1e-9 check was
true on ~66% of ALL bars -- a price-noise artifact, not v4 wanting to
rebalance. v4 itself never makes this comparison (``KellyRegime.on_bar``
only checks whether ITS OWN target column moved from the previous bar); the
fix above reproduces exactly that trigger, and separately tracks whether a
triggered move has actually been carried out yet (``owed``), so a deferred
rebalance is retried on every subsequent bar -- rather than silently
vanishing the moment ``target`` next holds still -- until it executes or an
override fires. ``current`` is still read live every bar, but only to
decide the override conditions' magnitudes, never to decide whether a move
is pending at all.

**Causality of the gate, spelled out.** The trailing-turnover reading used
to gate bar ``i``'s decision is built from events realized at bars
``< i`` ONLY -- strictly excluding bar ``i``'s own not-yet-decided event.
This is a deliberately STRICTER reading than ``r131_shared.trailing_turnover_ewm``
itself requires (its own docstring notes a caller MAY read its output at
index ``i`` including event ``i``, when the event at ``i`` is exogenous to
the gate reading it); here the event at bar ``i`` is NOT exogenous -- it is
this branch's own gate output -- so folding it into the same bar's gate
input would be circular. Concretely: this file maintains the exact
recursive form of ``pandas.Series.ewm(adjust=True).mean()`` --
``N_i = e_i + (1-alpha)*N_{i-1}``, ``D_i = 1 + (1-alpha)*D_{i-1}``,
``trailing_i = (N_i/D_i)*BARS_PER_DAY`` -- and READS ``trailing_{i-1}``
(state carried in from the previous call) before computing bar ``i``'s
decision, THEN folds bar ``i``'s own realized event ``e_i`` in for future
bars. ``wiring_self_test_recursive_ewm_matches_shared`` below proves this
recursion is numerically identical to ``r131_shared.trailing_turnover_ewm``'s
own (batched, includes-``i``) convention once the one-step lag is accounted
for, on both a synthetic array and this file's own real, realized-event
trace from an actual backtest -- so the online form is not a new formula
smuggled in, it is the same formula, run incrementally, one step behind.

The online recursion (O(1) per bar) is used INSTEAD of calling
``trailing_turnover_ewm`` fresh on the whole events-so-far array every bar,
which the module docstring's phrasing could be read as literally requesting.
That literal reading is O(n^2) over a bars-per-day=288, ~6-year inner period
(~630K bars) and was measured infeasible; the wiring self-test is what
licenses the substitution -- see the paragraph above.

``ConservativeTurnoverBand`` is NOT ``@register``ed -- experiments/-only,
reached only through this file.

CONFIGURATIONS EVALUATED: 1 (real-run causal-truncation probe) + 1 (A2
non-inertness, BTC spot, full inner period) + 4 (B1: BTC spot/futures x
inner-validation/full-inner-period) + 4 (B3: TURNOVER_UPPER at
0.5x/1x/2x/4x the shared default, BTC spot inner-validation) + 1 (B4: ETH
spot inner-validation) + 2 (B5: BTC spot + futures at the 0.40% taker tier,
inner-validation, the decisive cell) = 13 total backtest configurations.
(The synthetic wiring self-test against ``trailing_turnover_ewm`` is pure
numpy, no backtest, and is not counted in this tally, matching how
``r131_shared.py``'s own ``__main__`` probe of the same function is not
counted anywhere either.)

DECISION RULE (pre-registered, verbatim from ``r131_shared.py``, unaltered
after seeing any number): PROMOTE-candidate only if A2 (non-inertness) AND
B1 (both markets) AND B3 (plateau majority) AND B4 (full, both markets --
ETH spot only, since ETH futures data does not exist) AND B5 all pass.
Anything else is NEGATIVE. B2 (drawdown) is diagnostic only and never gates
promotion by itself. The branch's own named diagnostic (stress-episode
deferral behaviour) is reported regardless of the above and never gates
promotion by itself either.

USAGE
-----
    python experiments/r131_conservative_turnover_band.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import r131_shared as shared
from tradebot.inference import daily_returns, paired_bootstrap, total_log_return
from tradebot.metrics import compute_metrics
from tradebot.strategies.kelly_regime import BARS_PER_DAY
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.strategy import Context
from tradebot.window import run_period

TURNOVER_UPPER = shared.TURNOVER_UPPER
OVERRIDE_MULT = shared.OVERRIDE_MULT
TURNOVER_EWM_SPAN_DAYS = shared.TURNOVER_EWM_SPAN_DAYS
STRESS_EPISODES = shared.STRESS_EPISODES

B3_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)


# ================================================================== (1)
# ConservativeTurnoverBand: KellyRegimeV4.prepare() inherited unchanged.
# The new mechanism lives entirely in on_bar's per-bar state machine.
# NOT @register'd -- experiments/-only.
# ==================================================================

class ConservativeTurnoverBand(KellyRegimeV4):
    """kelly_regime_v4's exact target/deadband logic, wrapped in a hard
    admissible turnover corridor on this strategy's OWN realized rebalances.

    ``prepare()`` is inherited unchanged from ``KellyRegimeV4`` -- this
    class overrides ONLY ``on_bar``. See the module docstring above for the
    exact per-bar decision and the causality argument for the trailing-
    turnover gate. Not ``@register``ed -- experiments/-only, per this
    round's instructions.
    """

    name = "r131_conservative_turnover_band"

    def __init__(self, turnover_upper: float = TURNOVER_UPPER,
                 override_mult: float = OVERRIDE_MULT,
                 span_days: int = TURNOVER_EWM_SPAN_DAYS,
                 horizons: tuple[int, ...] = (20, 40, 80), **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.turnover_upper = turnover_upper
        self.override_mult = override_mult
        self.span_days = span_days
        self._alpha = 2.0 / (int(span_days * BARS_PER_DAY) + 1)

        # online recursive EWM state (adjust=True form), events at bars < i only
        self._N = 0.0
        self._D = 0.0

        # "do we still owe v4's own last-triggered rebalance" -- persists
        # across deferred bars, cleared only on execution. Set whenever
        # v4's own target column changes (v4's own on_bar trigger).
        self._owed = False
        self._owed_target = 0.0

        # diagnostics (reset per instance, i.e. per backtest run)
        self._n_calls = 0
        self._n_pending = 0
        self._n_deferred = 0
        self._n_override_exit = 0
        self._n_override_large = 0
        self._events: list[float] = []        # realized 0/1 events, call order
        self._trailing_pre: list[float] = []   # gate reading used at each call
        self._defer_log: list[tuple] = []      # (ts, desired, current, trailing) per deferral

    # -- the causal EWM of this strategy's OWN realized rebalances --------

    def _trailing_value(self) -> float:
        """State as of the end of the PREVIOUS call -- excludes this bar's
        own not-yet-decided event. See module docstring's causality note."""
        if self._D <= 0.0:
            return 0.0
        return (self._N / self._D) * BARS_PER_DAY

    def _update_trailing(self, event: float) -> None:
        self._N = event + (1.0 - self._alpha) * self._N
        self._D = 1.0 + (1.0 - self._alpha) * self._D

    def on_bar(self, ctx: Context) -> None:
        self._n_calls += 1
        desired = float(ctx.bar["target"])
        prev_target = float(ctx.prev["target"]) if ctx.prev is not None else desired

        # A new v4 DECISION arrives exactly when v4's own (already-
        # deadbanded) target column moves -- byte-identical condition to
        # KellyRegime.on_bar's own trigger. Always chase the LATEST v4
        # decision if one arrives while an earlier one is still owed.
        if abs(desired - prev_target) > 1e-9:
            self._owed_target = desired
            self._owed = True

        equity = ctx.equity
        trailing = self._trailing_value()
        self._trailing_pre.append(trailing)

        event = 0.0
        if self._owed and equity > 0:
            self._n_pending += 1
            current = ctx.position * ctx.close / equity
            owed_target = self._owed_target
            over_corridor = trailing >= self.turnover_upper
            override = False
            if over_corridor:
                full_exit = bool(owed_target == 0.0 and abs(current) > 1e-9)
                large_move = bool(abs(owed_target - current) > self.override_mult * self.turnover_upper)
                override = full_exit or large_move
                if full_exit:
                    self._n_override_exit += 1
                if large_move:
                    self._n_override_large += 1
            if (not over_corridor) or override:
                ctx.order_notional(owed_target)
                event = 1.0
                self._owed = False
            else:
                self._n_deferred += 1
                self._defer_log.append((ctx.ts, owed_target, current, trailing))
        self._events.append(event)
        self._update_trailing(event)


# ================================================================== (2)
# Run/metric helpers. b1_signal parameterized over (start, end), since
# r131_shared.b1_signal hardcodes inner-validation; the "full inner period"
# B1 cell and the A2 full-period run both need arbitrary windows.
# ==================================================================

def run_candidate(factory, df: pd.DataFrame, market, start, end, label: str = ""):
    strat = factory()
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res, strat


def b1_signal_window(factory, df: pd.DataFrame, market, start, end,
                      label: str = "", seed: int = 130) -> dict:
    m_cand, res_cand, strat = run_candidate(factory, df, market, start, end, label)
    m_v4, res_v4, _ = run_candidate(lambda: shared.get_strategy("kelly_regime_v4"),
                                    df, market, start, end, label)
    r_cand = daily_returns(res_cand.equity)
    r_v4 = daily_returns(res_v4.equity)
    n = min(len(r_cand), len(r_v4))
    paired = paired_bootstrap(r_cand.to_numpy()[:n], r_v4.to_numpy()[:n],
                              stat=total_log_return, seed=seed)
    return {
        "sharpe_cand": m_cand.sharpe, "sharpe_v4": m_v4.sharpe,
        "d_sharpe": m_cand.sharpe - m_v4.sharpe,
        "paired_diff": paired.diff.point, "paired_lo": paired.diff.lo, "paired_hi": paired.diff.hi,
        "significant": paired.significant,
        "dd_cand": m_cand.max_drawdown_pct, "dd_v4": m_v4.max_drawdown_pct,
        "trades_cand": m_cand.num_trades, "trades_v4": m_v4.num_trades,
        "strat": strat,
    }


def cell_clears(r: dict) -> bool:
    """Same +/-0.2 noise floor / bootstrap-excludes-zero rule the whole
    SIZE/ERR/COST family (R-109...R-129) has used."""
    return bool(r["d_sharpe"] > 0.2 or r["paired_lo"] > 0.0)


# ================================================================== (3)
# Wiring self-test: the online recursion used inside on_bar is the SAME
# formula as r131_shared.trailing_turnover_ewm, one step lagged. Pure numpy,
# no backtest -- validates the substitution the module docstring's
# "CAUSALITY OF THE GATE" section argues for.
# ==================================================================

def wiring_self_test_recursive_ewm_matches_shared(seed: int = 130, n: int = 500_000,
                                                   p: float = 0.0005) -> dict:
    rng = np.random.default_rng(seed)
    events = (rng.random(n) < p).astype(float)
    batch = shared.trailing_turnover_ewm(events)  # r131_shared's own formula, includes event i at index i

    alpha = 2.0 / (int(TURNOVER_EWM_SPAN_DAYS * BARS_PER_DAY) + 1)
    online = np.empty(n)
    N = D = 0.0
    for i in range(n):
        N = events[i] + (1.0 - alpha) * N
        D = 1.0 + (1.0 - alpha) * D
        online[i] = (N / D) * BARS_PER_DAY
    same_as_batch = bool(np.allclose(batch, online, rtol=1e-9))

    # truncation: does the recursion give an identical prefix run start-to-
    # finish vs cut partway (mirrors r131_shared.py's own __main__ probe,
    # applied to THIS file's reimplementation rather than the shared one).
    cut = 300_000
    trunc = np.empty(cut)
    N = D = 0.0
    for i in range(cut):
        N = events[i] + (1.0 - alpha) * N
        D = 1.0 + (1.0 - alpha) * D
        trunc[i] = (N / D) * BARS_PER_DAY
    n_check = cut - BARS_PER_DAY * (TURNOVER_EWM_SPAN_DAYS + 1)
    causal_under_truncation = bool(np.allclose(online[:n_check], trunc[:n_check],
                                               equal_nan=True, rtol=1e-9))

    return dict(same_as_batch=same_as_batch, causal_under_truncation=causal_under_truncation,
               pass_=bool(same_as_batch and causal_under_truncation))


def gate_lag_consistency(strat: "ConservativeTurnoverBand") -> bool:
    """Ties the wiring self-test to a REAL run: this branch's own realized
    event trace, re-fed through r131_shared.trailing_turnover_ewm in batch,
    must reproduce the SAME on_bar instance's own recorded gate readings
    (self._trailing_pre), one step lagged -- trailing_pre[i+1] (state before
    folding event i+1) must equal trailing_turnover_ewm(events)[i] (state
    after folding events 0..i)."""
    events = np.asarray(strat._events, dtype=np.float64)
    if len(events) < 2:
        return True
    unlagged = shared.trailing_turnover_ewm(events)
    lagged_from_run = np.asarray(strat._trailing_pre[1:], dtype=np.float64)
    return bool(np.allclose(unlagged[:-1], lagged_from_run, rtol=1e-9))


# ================================================================== (4)
# Real-run causal-truncation probe: mirrors R-125/R-129's own convention
# (evaluate the SAME window on the full multi-year df vs a df already
# truncated well past that window's end) -- proves on_bar's sequential
# state isn't secretly reading a stashed full frame.
# ==================================================================

def causal_truncation_probe(df: pd.DataFrame, label: str):
    factory = lambda: ConservativeTurnoverBand()
    m_full, _, _ = run_candidate(factory, df, shared.SPOT,
                                 shared.INNER_TRAIN_START, shared.INNER_TRAIN_END, label)
    df_trunc = df.loc[:shared.INNER_VAL_END]
    m_trunc, _, _ = run_candidate(factory, df_trunc, shared.SPOT,
                                  shared.INNER_TRAIN_START, shared.INNER_TRAIN_END, label)
    ok = bool(np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9))
    return ok, m_full.final_balance, m_trunc.final_balance


# ================================================================== (5)
# Stress-episode diagnostic: did the mechanism ever defer within 3 days of
# any of the six named episodes, and what was v4's own counterfactual fill?
# ==================================================================

def stress_episode_diagnostic(strat: "ConservativeTurnoverBand") -> list[dict]:
    rows = []
    for name, date_str in STRESS_EPISODES:
        episode_ts = pd.Timestamp(date_str, tz="UTC")
        window_lo = episode_ts - pd.Timedelta(days=3)
        window_hi = episode_ts + pd.Timedelta(days=3)
        hits = [(ts, desired, current, trailing) for ts, desired, current, trailing in strat._defer_log
                if window_lo <= ts <= window_hi]
        rows.append(dict(name=name, date=date_str, n_deferrals=len(hits), hits=hits))
    return rows


# ================================================================== (6)
# Main: wiring self-test -> causal probe -> A2 -> B1 -> B3 -> B4 -> B5 ->
# stress diagnostic -> verdict -> pytest.
# ==================================================================

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []
    n_configs = 0

    print("=" * 78)
    print("R-131 CONSERVATIVE: ConservativeTurnoverBand -- hard turnover-corridor")
    print("defer wrapped around kelly_regime_v4's own target/deadband decision.")
    print("=" * 78)

    btc, btc_label = shared.load_btc_train("spot")
    max_ts_seen.append(btc.index.max())
    print(f"\nBTC spot (truncated < {shared.OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    # -------------------------------------------------------------- wiring self-test
    print("\n" + "=" * 78)
    print("STEP 0 -- wiring self-test: online recursion vs r131_shared.trailing_turnover_ewm")
    print("=" * 78)
    wiring = wiring_self_test_recursive_ewm_matches_shared()
    print(f"  same_as_batch (synthetic, incl.-i convention): {wiring['same_as_batch']}")
    print(f"  causal_under_truncation (synthetic): {wiring['causal_under_truncation']}")
    print(f"  WIRING SELF-TEST PASS: {wiring['pass_']}")
    assert wiring["pass_"], "on_bar's online EWM recursion does not match r131_shared.trailing_turnover_ewm"

    # -------------------------------------------------------------- causal truncation probe
    print("\n" + "=" * 78)
    print("STEP 1 -- causal-truncation probe (real run, this file's own on_bar state machine)")
    print("=" * 78)
    probe_ok, full_bal, trunc_bal = causal_truncation_probe(btc, btc_label)
    n_configs += 1
    print(f"  causal_truncation_probe: {'PASS' if probe_ok else 'FAIL'} "
          f"({full_bal:.6f} vs {trunc_bal:.6f})")
    assert probe_ok, "ConservativeTurnoverBand reads ahead of its own truncation point -- aborting"

    primary_factory = lambda: ConservativeTurnoverBand()

    # -------------------------------------------------------------- A2
    print("\n" + "=" * 78)
    print("STEP 2 -- A2: non-inertness, BTC spot, full inner period "
          f"({shared.INNER_TRAIN_START} -> {shared.INNER_VAL_END})")
    print("=" * 78)
    m_a2, res_a2, strat_a2 = run_candidate(primary_factory, btc, shared.SPOT,
                                           shared.INNER_TRAIN_START, shared.INNER_VAL_END, btc_label)
    n_configs += 1
    a2 = shared.a2_non_inertness(strat_a2._n_deferred)
    lag_ok = gate_lag_consistency(strat_a2)
    print(f"  on_bar calls: {strat_a2._n_calls:,}  pending rebalances: {strat_a2._n_pending:,}  "
          f"deferrals: {strat_a2._n_deferred:,}  override(exit): {strat_a2._n_override_exit:,}  "
          f"override(large move): {strat_a2._n_override_large:,}")
    print(f"  A2 non-inertness (n_interventions={a2['n_interventions']}): PASS={a2['pass']}")
    print(f"  gate-lag consistency (real-run events vs shared.trailing_turnover_ewm, lagged): {lag_ok}")
    assert lag_ok, "on_bar's gate reading does not match the lagged shared.trailing_turnover_ewm"

    # -------------------------------------------------------------- B1
    print("\n" + "=" * 78)
    print("STEP 3 -- B1: BTC signal, spot + futures, inner-validation + full inner period")
    print("=" * 78)
    b1_rows = []
    for mkt_name, market in (("spot", shared.SPOT), ("futures", shared.FUTURES)):
        for window_name, start, end in (
            ("val", shared.INNER_VAL_START, shared.INNER_VAL_END),
            ("full", shared.INNER_TRAIN_START, shared.INNER_VAL_END),
        ):
            r = b1_signal_window(primary_factory, btc, market, start, end, btc_label)
            n_configs += 1
            b1_rows.append((mkt_name, window_name, r))
            print(f"  {mkt_name:>8s} {window_name:>4s}  sharpe_cand={r['sharpe_cand']:+.4f}  "
                  f"sharpe_v4={r['sharpe_v4']:+.4f}  d_sharpe={r['d_sharpe']:+.4f}  "
                  f"boot=[{r['paired_lo']:+.4f},{r['paired_hi']:+.4f}]  significant={r['significant']}  "
                  f"dd_cand={r['dd_cand']:.2f}%  dd_v4={r['dd_v4']:.2f}%  "
                  f"trades_cand={r['trades_cand']}  trades_v4={r['trades_v4']}  "
                  f"clears={cell_clears(r)}")
    b1_val = [(m, w, r) for m, w, r in b1_rows if w == "val"]
    b1_pass = all(cell_clears(r) for _, _, r in b1_val)  # decision rule: "B1 (both markets)" == the decisive val cells
    print(f"  B1 PASS (both markets, inner-validation, d_sharpe > +0.2 floor OR CI excludes zero "
          f"positively): {b1_pass}")

    # -------------------------------------------------------------- B3
    print("\n" + "=" * 78)
    print(f"STEP 4 -- B3: TURNOVER_UPPER sweep {B3_MULTIPLIERS} x default "
          f"({TURNOVER_UPPER:.6f} trades/day), BTC spot inner-validation")
    print("=" * 78)
    b3_rows = []
    for m in B3_MULTIPLIERS:
        tu = TURNOVER_UPPER * m
        factory = (lambda tu=tu: ConservativeTurnoverBand(turnover_upper=tu))
        r = b1_signal_window(factory, btc, shared.SPOT, shared.INNER_VAL_START, shared.INNER_VAL_END, btc_label)
        n_configs += 1
        sign = float(np.sign(r["d_sharpe"]))
        row = dict(multiplier=m, turnover_upper=tu, sign=sign, **r)
        b3_rows.append(row)
        print(f"  multiplier={m:g}x  turnover_upper={tu:.6f}  d_sharpe={r['d_sharpe']:+.4f}  "
              f"boot=[{r['paired_lo']:+.4f},{r['paired_hi']:+.4f}]  sign={sign:+.0f}  "
              f"deferrals={r['strat']._n_deferred}")
    signs = [row["sign"] for row in b3_rows]
    majority_count = max((signs.count(s) for s in set(signs)), default=0)
    b3_pass = majority_count >= 3
    print(f"  B3 PASS (>=3/4 same-signed): {b3_pass} ({majority_count}/4)")

    # -------------------------------------------------------------- B4
    print("\n" + "=" * 78)
    print("STEP 5 -- B4: ETH falsification, spot, inner-validation")
    print("=" * 78)
    eth = shared.load_eth_train()
    max_ts_seen.append(eth.index.max())
    eth_val_start_actual = max(pd.Timestamp(shared.INNER_VAL_START, tz=eth.index.tz), eth.index[0])
    truncated_by_data = eth.index[0] > pd.Timestamp(shared.INNER_VAL_START, tz=eth.index.tz)
    print(f"ETH spot (truncated < {shared.OOS_START}): {len(eth):,} bars, "
          f"{eth.index[0]} -> {eth.index[-1]}")
    if truncated_by_data:
        print(f"  NOTE: ETH data starts at {eth.index[0]}, AFTER inner-validation start "
              f"{shared.INNER_VAL_START} -- inner-validation window is truncated by data "
              f"availability to {eth_val_start_actual} -> {shared.INNER_VAL_END}.")
    b4_r = b1_signal_window(primary_factory, eth, shared.SPOT, shared.INNER_VAL_START, shared.INNER_VAL_END,
                            label="eth-spot")
    n_configs += 1
    btc_val_spot = next(r for mkt, win, r in b1_rows if mkt == "spot" and win == "val")
    btc_sign = float(np.sign(btc_val_spot["d_sharpe"]))
    eth_sign = float(np.sign(b4_r["d_sharpe"]))
    b4_pass = bool(btc_sign != 0 and eth_sign == btc_sign)
    print(f"  ETH spot d_sharpe={b4_r['d_sharpe']:+.4f}  boot=[{b4_r['paired_lo']:+.4f},{b4_r['paired_hi']:+.4f}]  "
          f"significant={b4_r['significant']}  deferrals={b4_r['strat']._n_deferred}")
    print(f"  BTC spot (val) sign={btc_sign:+.0f}  ETH spot sign={eth_sign:+.0f}  "
          f"B4 PASS (sign replicates): {b4_pass}")

    # -------------------------------------------------------------- B5
    print("\n" + "=" * 78)
    print("STEP 6 -- B5: fee-tier survival (0.40% taker), decisive (inner-validation) cells")
    print("=" * 78)
    b5_rows = []
    fee_market = {"spot": shared.SPOT_HIGH_FEE, "futures": shared.FUTURES_HIGH_FEE}
    for mkt_name, window_name, r0 in b1_val:
        r1 = b1_signal_window(primary_factory, btc, fee_market[mkt_name],
                              shared.INNER_VAL_START, shared.INNER_VAL_END, btc_label)
        n_configs += 1
        flip = bool(np.sign(r1["d_sharpe"]) != np.sign(r0["d_sharpe"]) and r0["d_sharpe"] != 0)
        b5_rows.append((mkt_name, window_name, r0, r1, flip))
        print(f"  {mkt_name:>8s}  d_sharpe@0.10%={r0['d_sharpe']:+.4f}  "
              f"d_sharpe@0.40%={r1['d_sharpe']:+.4f}  flip={flip}")
    b5_pass = not any(flip for *_, flip in b5_rows)
    print(f"  B5 PASS (no sign flip, either market): {b5_pass}")

    # -------------------------------------------------------------- stress diagnostic
    print("\n" + "=" * 78)
    print("STEP 7 -- named diagnostic: deferral behaviour at the six STRESS_EPISODES "
          "(A2's BTC spot, full inner period run)")
    print("=" * 78)
    stress_rows = stress_episode_diagnostic(strat_a2)
    for row in stress_rows:
        print(f"  {row['name']} ({row['date']}): deferrals within +/-3d = {row['n_deferrals']}")
        for ts, desired, current, trailing in row["hits"]:
            print(f"      {ts}  desired(v4 counterfactual fill)={desired:+.4f}  "
                  f"current={current:+.4f}  trailing={trailing:.4f}")

    # -------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    all_pass = a2["pass"] and b1_pass and b3_pass and b4_pass and b5_pass
    verdict = "PROMOTE-candidate" if all_pass else "NEGATIVE"
    print(f"A2={a2['pass']}  B1={b1_pass}  B2=diagnostic-only  B3={b3_pass}  B4={b4_pass}  B5={b5_pass}")
    print(f"VERDICT: {verdict}")
    if not all_pass:
        failed = [nm for nm, ok in (("A2", a2["pass"]), ("B1", b1_pass), ("B3", b3_pass),
                                    ("B4", b4_pass), ("B5", b5_pass)) if not ok]
        print(f"Reason(s): {', '.join(failed)}")

    # -------------------------------------------------------------- pytest
    print("\n" + "=" * 78)
    print("STEP 8 -- tests/test_causality_strict.py")
    print("=" * 78)
    import subprocess
    proc = subprocess.run(
        ["python", "-m", "pytest", "tests/test_causality_strict.py", "-q"],
        cwd=str(shared.ROOT), capture_output=True, text=True)
    pytest_out = proc.stdout.strip().splitlines()
    pytest_summary = pytest_out[-1] if pytest_out else f"(exit {proc.returncode})"
    print(f"  {pytest_summary}")

    max_ts = max(max_ts_seen)
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(1 causal probe + 1 A2 + {len(b1_rows)} B1 + {len(b3_rows)} B3 + 1 B4 + {len(b5_rows)} B5)")
    print(f"max timestamp read anywhere in this branch: {max_ts} "
          f"(< {shared.OOS_START}: {max_ts < pd.Timestamp(shared.OOS_START, tz='UTC')})")
    print("NO bar at or after 2023-01-01 was ever read by this file.")
    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(
        verdict=verdict, n_configs=n_configs, max_ts=max_ts,
        wiring=wiring, probe_ok=probe_ok, probe_full=full_bal, probe_trunc=trunc_bal,
        a2=a2, gate_lag_ok=lag_ok, strat_a2=strat_a2,
        b1_rows=b1_rows, b1_pass=b1_pass,
        b3_rows=b3_rows, b3_pass=b3_pass, majority_count=majority_count,
        b4=b4_r, b4_pass=b4_pass, btc_sign=btc_sign, eth_sign=eth_sign,
        truncated_by_data=truncated_by_data, eth_range=(eth.index[0], eth.index[-1]),
        b5_rows=b5_rows, b5_pass=b5_pass,
        stress_rows=stress_rows,
        pytest_summary=pytest_summary,
    )


if __name__ == "__main__":
    main()
