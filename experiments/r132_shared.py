"""Shared, read-only utilities and pre-registration for the R-132 round (08-25).

DIRECTION, in one sentence: vary `hedge_experts`'s own EXPERT COMPOSITION --
the panel of causal experts feeding its Hedge/multiplicative-weights blend
-- the one axis of this object R-128/R-129/R-130 all named as untouched,
because all three of those rounds varied the re-target/cost rule (WHERE
and HOW a no-trade band or weight-update penalty applies) around a
completely fixed ten-expert panel (four vol-scaled momentum horizons,
MACD, an RSI ramp, 1-bar reversion, a Donchian breakout, always-flat,
buy-and-hold -- see `HedgeExperts._experts`).

Literature:
- Freund, Y. & Schapire, R. (1997), "A Decision-Theoretic Generalization
  of On-Line Learning and an Application to Boosting," JCSS 55(1) --
  already `hedge_experts`'s own citation; the Hedge/multiplicative-weights
  regret bound holds within `2*sqrt(T ln N)` of the BEST expert in
  hindsight, a guarantee stated over the whole expert SET and largely
  insensitive to how bad any one OTHER expert is.
- Kalnishkan, Y. & Vyugin, M.V. (2008), "The Weak Aggregating Algorithm
  and weak mixability," Journal of Computer and System Sciences 74(8) --
  a mixture-of-experts aggregation whose own regret guarantee is stated
  to hold regardless of the individual quality of any one expert in the
  pool, the direct theoretical basis for this round's central question:
  does that guarantee's practical shadow mean an already-failed-standalone
  signal is nonetheless safe (or even useful) to add as one MORE vote,
  since the combinator's own online weight update is supposed to
  down-weight it if it is wrong?
- Grinold, R. (1989), "The Fundamental Law of Active Management," Journal
  of Portfolio Management 15(3) -- `IR = IC * sqrt(BR)`, already used by
  this project's own R-63 to price the multi-asset panel's diversification
  gain; the NOVEL branch below applies the same breadth logic to
  `hedge_experts`'s OWN expert panel rather than to a cross-asset universe.

**Mechanism, one sentence per branch, before any code was run:**

- CONSERVATIVE (`r132_conservative_mvrv_expert.py`): append exactly ONE new
  expert -- an MVRV-valuation mean-reversion vote (R-74's own signal,
  `data/btc_mvrv_daily.csv.gz` / `eth_mvrv_daily.csv.gz`, causally aligned
  via `tradebot.data.align_mvrv_causal`, exactly as R-74 loaded it) -- to
  the existing ten-member panel, with every other line of `HedgeExperts`
  (Hedge `eta`/`fixed_share`/`hysteresis`, the other ten experts, the
  weight-update loop) held bit-for-bit fixed. The narrowest possible test
  of the EXPERT COMPOSITION question: can a signal R-74 already found
  NEGATIVE as a standalone confirming-vote GATE on `kelly_regime_v4` still
  earn its keep once it is merely one more weighted VOTE inside a
  regret-minimizing combinator, where the combinator (not a hand-built
  threshold) decides how much to trust it?

- NOVEL (`r132_novel_diversified_panel.py`): a structural composition
  change rather than an addition -- collapse the panel's four vol-scaled
  momentum horizons (1h/6h/1d/1w, all long/short-symmetric bets on the
  SAME underlying trend factor and therefore highly mutually correlated)
  down to two representative horizons (short + long), and use the two
  freed slots to add MVRV valuation (as above) AND a stablecoin-supply
  growth macro-flow vote (R-54/R-55/R-58's own `growth_14d`-derived
  signal, `data/stablecoin_supply_daily.csv.gz`, causally aligned via
  `tradebot.data.align_stablecoin_causal`). Report the panel's own
  pairwise expert-position correlation matrix and a Grinold-style
  effective-breadth estimate (`BR_eff = N / (1 + (N-1)*rho_bar)`,
  Grinold's own formula for N equally-correlated bets at mean pairwise
  correlation `rho_bar`) BEFORE and AFTER the swap, as a named diagnostic
  independent of B1-B5: does removing momentum-horizon redundancy and
  adding structurally different information actually raise `BR_eff`, or
  does it just trade away two working experts for two that (per R-74,
  R-54-58) were already shown weak standalone?

**Which constraint this attacks: INFO** -- but a structurally different
application than every one of the 19+ prior INFO-axis attempts against
`kelly_regime_v4`: those all evaluated an external signal as a standalone
directional bet or a veto/confirming-vote GATE bolted onto a single vote.
This round is the first to test the SAME already-tried information
sources as ADDITIONAL VOTING MEMBERS inside a regret-minimizing (Hedge)
combinator on a different object (`hedge_experts`), where the combinator's
own online down-weighting -- not a hand-built threshold -- is supposed to
make a weak or wrong expert cheap to include rather than something that
must be pre-filtered by a gate.

**Not a duplicate of:**
- R-74 (MVRV level-and-rate-of-change as a standalone confirming-vote gate
  on `kelly_regime_v4`, NEGATIVE): here MVRV is one Hedge-weighted vote
  among eleven or twelve, never gating anything directly, on a different
  object.
- R-54/R-55/R-58 (stablecoin supply as a standalone hard-veto /
  confirming-vote / persistence-filtered gate on `kelly_regime_v4`, all
  NEGATIVE): same distinction -- embedded as a vote, not a gate, on a
  different object.
- R-128/R-129/R-130 (`hedge_experts`'s re-target/cost-rule axis: EV-bands
  at three application points, weight-update-level lazy L1 shrinkage,
  combinator-wrapping online learning): all three hold the ten-expert
  panel completely fixed and vary only the re-target/cost mechanism around
  it. This round holds the re-target rule (`hysteresis=0.05`, unchanged)
  fixed and is the first to vary the panel itself.
- R-113 (ERR-axis novelty brakes applied to the multi-asset panel):
  different object (six-asset panel) and different axis (a veto/brake on
  an existing signal, not new panel members).

**What would make this fail, named now, before any code:**
1. INERTNESS -- the new expert(s)' own Hedge weight could simply decay
   toward the `fixed_share` floor (1e-4) within weeks if genuinely
   uninformative and never meaningfully move the blend at all; checked by
   gate A2 before any B1 number is read (per ROUTINE.md: "not tested is
   not a negative result").
2. FLOOR DRAG -- even if the new expert's weight is non-trivial at times,
   `fixed_share` re-injects a small non-zero probability mass into EVERY
   expert every bar, including ones the update has already learned are
   wrong; a persistently-wrong expert could keep contributing a small but
   nonzero, wrong-directional drag forever -- the opposite of "cheap to
   include," and the concrete way the Kalnishkan-Vyugin guarantee could
   fail to show up in a finite, non-adversarial, real-market sample rather
   than an adversarial worst case.
3. BTC-PASS/ETH-INVERT -- this project's now seven-plus constructions
   across two axes and two objects (R-115, R-125, R-126, R-127, R-128
   novel, ...) share this exact signature, and MVRV/stablecoin supply are
   BOTH already individually known (R-74, R-54-58) to behave differently
   or invert across BTC/ETH. B4 is a real, likely-binding test here, not a
   formality.
4. NOVEL-branch-specific: collapsing four momentum horizons to two could
   simply REMOVE working signal (the horizon diversity may be contributing
   more than the two new experts add), a net panel-quality loss rather
   than a breadth gain -- checked by reporting a horizons-only ablation
   (four horizons collapsed to two, no new experts added) alongside the
   full novel construction, to separate the two structural changes' own
   individual contributions.

**Falsification test, pre-registered:** B4 -- does the candidate's
`d_sharpe` sign (candidate vs `hedge_experts`, inner-validation) replicate
on ETH spot (`load_coinbase_eth_spot`)? The same test this whole
SIZE/ERR/COST/INFO research programme has used since R-59.

**Decision rule, pre-registered verbatim from the `hedge_experts` family's
own convention (R-128...R-130):** PROMOTE-candidate only if the causal-
truncation probe AND A2 (non-inertness) AND B1 (both markets, full period
AND inner-validation) AND B3 (plateau majority across a natural parameter
grid) AND B4 (sign replicates on ETH) AND B5 (no sign flip at the 0.40%
fee tier) all pass. B2 (drawdown/turnover) and the panel-correlation /
breadth diagnostic are reported but never gate promotion by themselves.

No bar at or after `OOS_START = 2023-01-01` may be read by either branch.
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
# Splits. Identical convention to every prior round: inner-train / inner-
# validation only. The holdout (>= OOS_START) is never read by either branch.
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


def sharpe_diff(res_a, res_b, mean_block: float = 30.0, n_boot: int = 2000, seed: int = 132):
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


def log_growth_diff(res_a, res_b, mean_block: float = 30.0, n_boot: int = 2000, seed: int = 132):
    """Paired-bootstrap total-log-growth difference (a - b) -- B1's other reported stat."""
    ra = daily_returns(res_a.equity)
    rb = daily_returns(res_b.equity)
    n = min(len(ra), len(rb))
    ra = ra.iloc[-n:].to_numpy()
    rb = rb.iloc[-n:].to_numpy()
    return paired_bootstrap(ra, rb, total_log_return, mean_block=mean_block, n_boot=n_boot, seed=seed)


def effective_breadth(corr_matrix: np.ndarray) -> float:
    """Grinold (1989) equal-correlation effective breadth: N / (1 + (N-1)*rho_bar).

    ``rho_bar`` is the mean OFF-DIAGONAL pairwise correlation. Matches the
    formula R-63 already used for the six-asset cross-sectional panel,
    applied here to expert POSITIONS rather than asset returns.
    """
    n = corr_matrix.shape[0]
    off_diag = corr_matrix[~np.eye(n, dtype=bool)]
    rho_bar = float(np.nanmean(off_diag))
    return n / (1.0 + (n - 1) * rho_bar)


def a2_non_inertness(weight_share: np.ndarray, num_experts: int, frac_threshold: float = 0.01,
                      mass_multiple: float = 2.0) -> dict:
    """Does the new expert ever meaningfully move the blend?

    ``weight_share`` is the new expert's own Hedge probability weight
    ``p_i`` at every bar of inner-validation. Pass if it exceeds
    ``mass_multiple / num_experts`` (twice the uniform floor, by default)
    on at least ``frac_threshold`` of bars -- cheap, pre-registered, and
    independent of whether the eventual blend helped or hurt.
    """
    uniform = 1.0 / num_experts
    frac_above = float(np.mean(weight_share > mass_multiple * uniform))
    return {"frac_bars_above_2x_uniform": frac_above, "pass": frac_above >= frac_threshold}


if __name__ == "__main__":
    # Self-test: causal truncation probe on THIS module's own baseline
    # plumbing (hedge_experts.prepare() itself is covered by
    # tests/test_causality_strict.py at the framework level).
    df, label = load_btc_train("spot")
    m_full, _ = run_baseline(df, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)

    df_trunc = df.loc[:INNER_VAL_END]
    m_trunc, _ = run_baseline(df_trunc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)

    ok = np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9)
    print(f"causal truncation probe (r132_shared baseline plumbing): "
          f"{'PASS' if ok else 'FAIL'} ({m_full.final_balance} vs {m_trunc.final_balance})")
    assert ok, "run_baseline reads ahead of its own truncation point"

    m_spot, _ = run_baseline(df, SPOT, None, INNER_TRAIN_END, label)
    m_fut, _ = run_baseline(df, FUTURES, None, INNER_TRAIN_END, label)
    print(f"baseline inner-train spot: trades={m_spot.num_trades} "
          f"final={m_spot.final_balance:.1f} sharpe={m_spot.sharpe:.3f}")
    print(f"baseline inner-train futures: trades={m_fut.num_trades} "
          f"final={m_fut.final_balance:.1f} sharpe={m_fut.sharpe:.3f}")
