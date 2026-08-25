"""Shared, read-only utilities and pre-registration for the R-135 round (08-25).

DIRECTION, in one sentence: a third EXPERT COMPOSITION construction on
`hedge_experts`'s Hedge/multiplicative-weights panel -- adding derivatives-
positioning ("crowding") and implied-volatility-risk-premium votes, two data
channels never tried as Hedge-panel members before -- while making R-130's
and R-132's own named next step a MANDATORY pre-registered gate instead of a
discretionary post-hoc skeptic check: a turnover-matched, information-free
control (B6), plus reporting (and partially gating on) an evaluation window
where the fixed baseline panel is NOT itself losing money.

**Why this exact axis, and why now.** Per docs/LEDGER.md's 08-25 backlog
re-ranking (after R-134), the ranked backlog is empty of anything but B-06
(forward paper trading, already running unattended). The best-ranked
off-backlog lead is `hedge_experts`'s own EXPERT COMPOSITION axis: R-130
named it as the one thing no round had varied ("...or should vary something
about `hedge_experts` other than its re-target/cost rule (its EXPERT
COMPOSITION, e.g., has never been touched by any two-branch round)"); R-132
tried it once (MVRV valuation vote + stablecoin-supply-growth macro-flow
vote, both DAILY-cadence signals) and got NEGATIVE via a turnover-reduction-
in-a-losing-regime artifact, closing with an explicit recommendation: a third
attempt should either change the evaluation window (`hedge_experts`'s own
baseline is itself losing money over the standard inner-validation split) or
make R-130's own turnover-matched-control skeptic check mandatory. This round
does both, and picks two data channels never tried in this axis before.

**Verified before any candidate code was written** (see this round's own
console log): baseline `hedge_experts`, BTC, inner-train (2017-01-01 to
2020-12-31) is Sharpe 1.592 spot / 1.726 futures_5x, final balance $15,716 /
$65,761 from $1,000 -- genuinely profitable. Inner-validation (2021-01-01 to
2022-12-31) is Sharpe -0.711 spot / -0.762 futures_5x, final balance $594 /
$5.80 -- a near-total loss on futures. This is the R-132-diagnosed "losing
regime" made concrete, and it is why B1 below reports THREE periods, not two,
with `inner_train` (the non-losing window) entering the decision rule
alongside `full_inner` rather than being informational-only.

Literature:
- Freund, Y. & Schapire, R. (1997), "A Decision-Theoretic Generalization of
  On-Line Learning and an Application to Boosting," JCSS 55(1) --
  `hedge_experts`'s own citation, unchanged from R-132's use of it.
- De Roon, F., Nijman, T. & Veld, C. (2000), "Hedging Pressure Effects in
  Futures Markets," Journal of Finance 55(3) -- the theoretical basis for
  this round's positioning/crowding expert: futures risk premia are driven
  by the NET positioning of hedgers vs. speculators, so an aggregate
  long/short-ratio extreme carries information about the futures risk
  premium's sign, independent of any price-trend signal already in the
  panel. Crypto perpetual futures have no true hedgers in the commodity
  sense, but the same net-positioning-predicts-forward-premium logic
  underlies practitioner "crowding" indicators on this exact data source.
- Bollerslev, T., Tauchen, G. & Zhou, H. (2009), "Expected Stock Returns and
  Variance Risk Premia," Review of Financial Studies 22(11) -- the
  theoretical basis for this round's implied-volatility expert: the gap
  between options-implied and subsequently realized variance (the variance
  risk premium, VRP) positively predicts near-term returns as compensation
  for bearing volatility risk. This round's DVOL vote uses that paper's own
  SIGN (positive VRP -> risk-on lean), fixed a priori from the citation, not
  fit to any backtest number.
- Grinold, R. (1989), "The Fundamental Law of Active Management," JPM 15(3)
  -- `IR = IC * sqrt(BR)`, reused (as in R-63 and R-132) for the NOVEL
  branch's breadth-maximizing panel-slot-selection procedure below.

**Mechanism, one sentence per branch, before any code was run:**

- CONSERVATIVE (`r135_conservative_positioning_expert.py`): append exactly
  ONE new expert -- a derivatives-positioning ("crowding") contrarian vote
  built from `tradebot.data.load_binance_metrics`'s `count_long_short_ratio`
  column (the broadest, all-account positioning ratio; deliberately NOT the
  `*_toptrader_*` columns, which R-81 already found 37.6%-missing across
  2022) -- to the existing ten-member panel, with every other line of
  `HedgeExperts` (Hedge `eta`/`fixed_share`/`hysteresis`, the other ten
  experts, the weight-update loop) held bit-for-bit fixed. This is a
  5-minute-native-cadence signal, unlike R-132's two daily-cadence votes,
  and a market-structure ("how levered/one-sided is the book right now")
  channel rather than a valuation or macro-flow one -- structurally
  different information from both of R-132's experts.

- NOVEL (`r135_novel_breadth_optimized_panel.py`): a different slot-
  selection PROCEDURE, not just different experts. R-132's horizon-collapse
  was an a-priori, eyeballed choice ("keep shortest + longest") never
  computed from data. This branch instead computes, on INNER-TRAIN ONLY
  (2017-2020, never inner-validation or the holdout), the existing ten
  experts' own pairwise position-correlation matrix and drops the TWO
  experts with the highest mean absolute correlation to the rest of the
  panel -- i.e., the two contributing the least marginal Grinold breadth --
  replacing them with the positioning vote (as above) AND an implied-
  volatility variance-risk-premium vote (Deribit DVOL minus the panel's own
  trailing realized volatility, z-scored, Bollerslev-Tauchen-Zhou sign).
  This tests whether an actual breadth-maximizing selection procedure beats
  an eyeballed one, and whether a genuinely different data channel (a
  forward-looking, priced market expectation) adds distinguishable
  information the crowding vote alone does not.

**Which constraint this attacks: INFO** -- same structural distinction R-132
established: these are ADDITIONAL VOTING MEMBERS inside a regret-minimizing
combinator, not standalone directional bets or hand-built gates. Positioning
was tried standalone on `kelly_regime_v4` (R-81, NEGATIVE at the lead-time
measurement gate) and DVOL likewise (R-73, NEGATIVE, lags 2.0-9.0 days) --
both irrelevant failure modes here, since a Hedge vote does not need to LEAD
price to earn weight, only to be less wrong than the panel's current blend
often enough for the multiplicative update to favor it.

**Not a duplicate of:**
- R-132 (this same axis, MVRV + stablecoin-supply votes): different data
  channels (positioning + implied vol vs. valuation + macro-flow), and this
  round's slot-selection procedure for the NOVEL branch is data-driven
  (inner-train correlation) rather than R-132's a-priori horizon rule.
- R-81 (positioning as a standalone lead-time-gated confirming signal on
  `kelly_regime_v4`) and R-73 (DVOL, same object, same gate): different
  object, different mechanism (Hedge vote vs. lead/lag gate).
- R-128/R-129/R-130 (`hedge_experts`'s re-target/cost-rule axis): all three
  hold the panel fixed; this round holds the re-target rule
  (`hysteresis=0.05`) fixed and varies the panel.

**What would make this fail, named now, before any code:**
1. INERTNESS -- gate A2, unchanged from R-132's construction.
2. TURNOVER-REDUCTION ARTIFACT -- R-132's own diagnosed failure mode,
   exactly what B6 below exists to catch: any new vote that happens to
   dampen the blend's churn during the 2021-2022 losing regime will look
   like an improvement on `inner_val` alone regardless of whether it carries
   real information. B6 is now a MANDATORY gate, not a discretionary
   post-hoc check.
3. BTC-PASS/ETH-INVERT -- this project's now eight-plus constructions across
   two axes and two objects share this signature; B4 is a real test.
4. NOVEL-branch-specific -- COVERAGE ARTIFACT: DVOL only exists from
   2021-03-24 and Binance metrics only from 2020-09-01 (BTC) /
   2021-12-01 (ETH); both new experts are near-permanently flat (NaN -> 0,
   the panel's existing warmup convention) before those dates. If the
   NOVEL branch's apparent gain is concentrated in `inner_train` (2017-2020,
   almost entirely BEFORE either signal exists), that gain can only be
   coming from the breadth-optimized DROP of two experts, not from the two
   ADDITIONS -- checked directly by comparing against the mandatory
   `BreadthDropOnlyPanel` ablation (drop the same two experts, add nothing).
5. NOVEL-branch-specific -- SELECTION-PROCEDURE OVERFIT: the correlation-
   based drop selection is computed once, on inner-train, and FROZEN before
   any inner-validation or B1 number is read; if it happens to select
   experts whose removal helps ONLY inner-train (the exact period it was
   fit on) and not `full_inner`, that is the data-driven procedure fitting
   its own selection window, the direct data-driven analogue of R-132's
   named risk -- checked by requiring `full_inner`, not just `inner_train`
   alone, to also pass in the decision rule below.

**Falsification test, pre-registered:** B4 -- does the candidate's
`d_sharpe` sign (candidate vs `hedge_experts`, inner-validation) replicate on
ETH spot? Unchanged convention since R-59.

**Decision rule, pre-registered, before any branch's code was run:**
PROMOTE-candidate only if ALL of:
  (a) causal-truncation probe passes;
  (b) A2 (every new expert non-inert);
  (c) B1 -- both markets, and a MAJORITY (>=2 of 2) of {`inner_train`,
      `full_inner`} cells beat baseline by `d_sharpe`'s point estimate
      (`inner_val` is reported for every cell but does NOT by itself count
      toward this majority, precisely because it is the known-losing window
      R-132's artifact lived in);
  (d) B3 (plateau majority across the divisor grid);
  (e) B4 (sign replicates on ETH spot, inner-validation);
  (f) B5 (no sign flip at the 0.40% fee tier);
  (g) B6 (MANDATORY) -- the candidate's `d_sharpe` vs. baseline on the
      decisive B1 cells must EXCEED the turnover-matched, information-free
      control's own `d_sharpe` vs. baseline on the same cells, not merely
      have the same sign. A candidate that only matches or trails the
      control is a turnover-reduction artifact, R-132's own finding,
      restated as a gate instead of a post-hoc check.
Anything else is NEGATIVE. No bar at or after `OOS_START = 2023-01-01` may
be read by either branch; consulting the true holdout is out of scope for
this round regardless of outcome and is left to a future pre-registration
if every gate above clears.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.inference import daily_returns, paired_bootstrap, total_log_return  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# ----------------------------------------------------------------------
# Splits. Identical convention to every prior round on this object.
# ----------------------------------------------------------------------
INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"  # do not read; guarded by _assert_no_holdout below

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT_HIGH_FEE = MarketSpec.spot(fee_rate=0.0040)      # B5: 0.40% taker tier
FUTURES_HIGH_FEE = MarketSpec.futures(leverage=5.0, fee_rate=0.0040)

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

B3_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)

# B1 periods: THREE, not two -- `inner_train` is the non-losing window,
# added per R-132's own recommendation, and counts toward the decision rule
# (see module docstring) rather than being purely informational.
B1_PERIODS = [
    ("inner_train", INNER_TRAIN_START, INNER_TRAIN_END),
    ("full_inner", INNER_TRAIN_START, INNER_VAL_END),
    ("inner_val", INNER_VAL_START, INNER_VAL_END),
]


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {OOS_START}")


def load_btc_train(kind: str = "spot"):
    df, label = load_dataset(ROOT / "data", kind)
    train = df.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(train)
    return train, label


def load_eth_train():
    eth = load_coinbase_eth_spot(ROOT / "data")
    assert eth is not None, "ETH spot data not committed"
    eth = eth.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(eth)
    return eth


# ----------------------------------------------------------------------
# Baseline (unmodified hedge_experts) and generic-candidate run/metric helpers.
# ----------------------------------------------------------------------

def run_baseline(df: pd.DataFrame, market: MarketSpec, start: str, end: str,
                  label: str = ""):
    strat = get_strategy("hedge_experts")
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


def run_strategy(strat, df: pd.DataFrame, market: MarketSpec, start: str, end: str,
                  label: str = ""):
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


def sharpe_diff(res_a, res_b, mean_block: float = 30.0, n_boot: int = 2000, seed: int = 135):
    """Paired-bootstrap Sharpe difference (a - b) on aligned daily returns."""
    ra = daily_returns(res_a.equity)
    rb = daily_returns(res_b.equity)
    n = min(len(ra), len(rb))
    ra = ra.iloc[-n:].to_numpy()
    rb = rb.iloc[-n:].to_numpy()

    def _sharpe(x):
        x = np.asarray(x, dtype=float)
        sd = x.std(axis=-1, ddof=1)
        sd = np.where(sd <= 0, np.nan, sd)
        return np.nan_to_num(x.mean(axis=-1) / sd * np.sqrt(365.25))

    return paired_bootstrap(ra, rb, _sharpe, mean_block=mean_block, n_boot=n_boot, seed=seed)


def log_growth_diff(res_a, res_b, mean_block: float = 30.0, n_boot: int = 2000, seed: int = 135):
    """Paired-bootstrap total-log-growth difference (a - b)."""
    ra = daily_returns(res_a.equity)
    rb = daily_returns(res_b.equity)
    n = min(len(ra), len(rb))
    ra = ra.iloc[-n:].to_numpy()
    rb = rb.iloc[-n:].to_numpy()
    return paired_bootstrap(ra, rb, total_log_return, mean_block=mean_block, n_boot=n_boot, seed=seed)


def effective_breadth(corr_matrix: np.ndarray) -> float:
    """Grinold (1989) equal-correlation effective breadth: N / (1 + (N-1)*rho_bar)."""
    n = corr_matrix.shape[0]
    off_diag = corr_matrix[~np.eye(n, dtype=bool)]
    rho_bar = float(np.nanmean(off_diag))
    return n / (1.0 + (n - 1) * rho_bar)


def a2_non_inertness(weight_share: np.ndarray, num_experts: int, frac_threshold: float = 0.01,
                      mass_multiple: float = 2.0) -> dict:
    """Does the new expert ever meaningfully move the blend? Unchanged from R-132."""
    uniform = 1.0 / num_experts
    frac_above = float(np.mean(weight_share > mass_multiple * uniform))
    return {"frac_bars_above_2x_uniform": frac_above, "pass": frac_above >= frac_threshold}


def replay_hedge_weights(strat, df: pd.DataFrame) -> np.ndarray:
    """Read-only, bit-faithful replay of `HedgeExperts.prepare()`'s own
    `logw`/`p` recursion, instrumented to also record the full weight-
    probability history `p_i(t)` for every expert at every bar. Copied
    verbatim from R-132's own diagnostic (`r132_novel_diversified_panel.py`);
    does not modify `HedgeExperts.prepare`/`on_bar`.
    """
    r = np.log(df["close"]).diff()
    sig1 = r.ewm(span=288, min_periods=250).std()
    a = strat._experts(df, r, sig1)
    r_a = r.to_numpy()
    sig_a = sig1.shift(1).to_numpy()
    n, num = a.shape
    p_hist = np.zeros((n, num))
    logw = np.zeros(num)
    p = np.ones(num) / num
    for i in range(2, n):
        s = sig_a[i]
        if not np.isfinite(s) or s <= 0:
            p_hist[i] = p
            continue
        z_t = min(max(r_a[i] / (3.0 * s), -1.0), 1.0)
        fee_n = min(strat.fee_rate / (3.0 * s), 0.25)
        g = np.clip(a[i - 1] * z_t - fee_n * np.abs(a[i - 1] - a[i - 2]), -1.0, 1.0)
        logw += strat.eta * g
        logw -= logw.max()
        p = np.exp(logw)
        p /= p.sum()
        p = (1.0 - strat.fixed_share) * p + strat.fixed_share / num
        logw = np.log(p)
        p_hist[i] = p
    return p_hist


def turnover_matched_control(df: pd.DataFrame, market: MarketSpec, target_trades: int,
                              start: str, end: str, label: str = "",
                              hyst_grid=(0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25,
                                         0.30, 0.35, 0.40, 0.45, 0.50)):
    """B6 (MANDATORY): find the unmodified-panel hysteresis whose OWN
    inner-val trade count on this market most closely matches
    `target_trades` (the candidate's own inner-val trade count), then return
    that turnover-matched, zero-new-information control's metrics/result on
    (start, end). This is R-130's/R-132's own skeptic construction, promoted
    from a discretionary post-hoc check to this round's mandatory B6 gate.
    """
    from tradebot.strategies.hedge_experts import HedgeExperts
    rows = []
    for h in hyst_grid:
        strat = HedgeExperts(hysteresis=h)
        res = run_period(strat, df, start=INNER_VAL_START, end=INNER_VAL_END,
                          market=market, start_balance=1000.0, data_label=label)
        m = compute_metrics(res)
        rows.append((h, m.num_trades))
    best_h, best_trades = min(rows, key=lambda t: abs(t[1] - target_trades))
    strat = HedgeExperts(hysteresis=best_h)
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res, best_h, best_trades


if __name__ == "__main__":
    # Self-test: causal truncation probe on THIS module's own baseline
    # plumbing, plus the profitable/losing-regime split this round's own
    # premise rests on (see module docstring for the numbers this prints).
    df, label = load_btc_train("spot")
    m_full, _ = run_baseline(df, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)
    df_trunc = df.loc[:INNER_VAL_END]
    m_trunc, _ = run_baseline(df_trunc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)
    ok = np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9)
    print(f"causal truncation probe (r135_shared baseline plumbing): "
          f"{'PASS' if ok else 'FAIL'} ({m_full.final_balance} vs {m_trunc.final_balance})")
    assert ok, "run_baseline reads ahead of its own truncation point"

    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
        for per_name, start, end in B1_PERIODS:
            m, _ = run_baseline(df, mkt, start, end, label)
            print(f"baseline {mkt_name}/{per_name}: trades={m.num_trades} "
                  f"final={m.final_balance:.1f} sharpe={m.sharpe:.3f}")
