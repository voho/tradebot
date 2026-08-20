"""R-62: decompose `kelly_regime_v4`'s two mechanisms -- the directional vote
and the volatility-targeted scale -- and ask which one, alone, produces the
matched-exposure drawdown advantage R-57 found on the six-asset panel.

Shared, frozen infrastructure for a two-branch parallel round. Per
ROUTINE.md's parallelism rules this file is neutral ground: both branches
import from it, neither branch edits it, and it does not itself define a
candidate strategy or run a verdict. It exists so the pre-registration below
is committed once, before either branch reads a single strategy number.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **SIZE** (this is a structural decomposition of the
mechanism that has produced every profitable strategy in this project, not a
retune of its parameters) and, secondarily, **N≈3** (the panel's independent
price paths, per R-57).

Backlog item: **B-27**, filed by R-61 after R-57 -> R-61 left the panel
puzzle with two explanations ruled out and none confirmed:

  - R-57: `kelly_regime_v4`'s matched-exposure drawdown advantage over a
    constant-exposure hold at v4's own mean notional is 6/6 on BTC/ETH
    (a window they share with the panel) and 0/6 on six further Coinbase
    instruments the strategy was never fitted on.
  - R-59, R-60: neither the sizing constant's magnitude (`target_vol`,
    `max_leverage`) nor the vote/gate's timing (anchor horizons, an
    OU-half-life-adaptive rescale, a CUSUM change-point vote) restores the
    property on the panel. SIZE-axis record after these: 0-for-21.
  - R-61: a genuinely different signal family (mean-reversion, not trend) on
    the identical unmodified SIZE machinery also fails to produce a
    promotable edge on the panel, and the panel's own measured Hurst exponent
    (mean 0.601) mildly refutes the "buy-the-dip" explanation R-59/R-60 both
    offered for why the matched-exposure gap exists there at all.

What none of R-57 -> R-61 tested, because none of their pre-registrations
authorized it: `kelly_regime_v4` is not one mechanism, it is two, multiplied
together --

    desired[i] = frac[i] * scale[i]

`frac` is the latched 20/40/80-day multi-anchor vote (trend timing: 0, 1/3,
2/3 or 1). `scale` is the conditional volatility target (`kelly_regime_v3`'s
extreme-only inverse-vol sizing: constant notional through normal vol,
`target_vol / vol` only once vol has broken out of [low_in, high_in] and
until it retraces below [low_out, high_out] -- the same hysteresis idea as
the vote, applied to the risk axis). R-59 varied `scale`'s magnitude and
R-60 varied `frac`'s timing, but no round has asked what either factor does
*alone*, with the other held at its simplest possible constant. That is
exactly what B-27 named as the untested angle: "does the SAME
matched-exposure advantage appear for a strategy that holds a constant
vol-targeted exposure with no directional vote at all ... isolating the SIZE
machinery's own turnover/rebalancing behavior from any signal, trend or
reversion?"

This round answers that question and its mirror image, as a factorial
decomposition against v4's own combination:

  conservative  `frac` forced to 1.0 always (never gated) x v4's own
                unmodified conditional vol-target `scale` -- the literal
                B-27 request. Zero new parameters; nothing removed is
                replaced by anything tuned, a component is simply deleted.

  novel         v4's own unmodified vote `frac` (20/40/80, latched, 1% band)
                x a CONSTANT exposure `c_const = 1.0` whenever the vote says
                "in" -- no volatility adjustment of any kind. The complement
                of the conservative arm: it isolates trend timing with the
                risk-scaling machinery deleted instead of the signal.
                `c_const=1.0` is chosen for being the least arbitrary
                non-fitted value (full notional whenever the trend vote is
                on, the simplest possible binary trend rule) and is not
                swept or selected against any data -- there is no second
                free parameter to protect against here.

Together the two arms bracket v4 = frac(trend) x scale(vol-target): if
EITHER arm alone reproduces the panel property, that identifies which half
of the mechanism carries it. If NEITHER does, that is a third, independent
line of evidence (after R-59's scale-axis and R-60's timing-axis failures)
that the property is not decomposable out of v4's own machinery at all --
strengthening, rather than merely repeating, R-59/R-60's "this looks like a
property of the panel's own price dynamics" reading, because this is the
first round to ask the question at the level of *which ingredient*, not
*which calibration of an ingredient*.

**Not a duplicate of:**
- R-59 (retunes `scale`'s magnitude, keeps `frac` and the vol-target
  functional form), R-60 (retunes `frac`'s timing mechanism, keeps `scale`).
  Both keep BOTH factors; this round deletes one factor entirely from each
  arm, which neither prior round did.
- R-61 (replaces `frac`'s sign rule with reversion, keeps `scale` and a real
  directional signal). This round's conservative arm has no directional
  signal of any kind, gated or not; R-61 always had one.
- R-34, R-37, R-38, R-40 -> R-46, R-53 -> R-56 (SIZE-axis retunes / new
  sizing formulas, all keeping the frac x scale product form).

**Simulable here**: yes. Both arms are `Strategy` subclasses with the same
`prepare`/`on_bar` shape v4 already uses; no order book, no new data. Reuses
R-57's committed panel loaders/harness (`load_candidates`, `select_panel`,
`measure`) and R-33/R-57's `ConstantExposureHold`/`mean_notional` matched-hold
machinery from `experiments/matched_hold.py`. Zero new data fetched.

=====================================================================
LITERATURE (Step 2 sources, read before either branch was coded)
=====================================================================

- Wang, F., & Yan, X. S. (2021), "Downside risk and the performance of
  volatility-managed portfolios," Journal of Banking & Finance, 131, 106198.
  Decomposes volatility-managed portfolio performance into a **volatility
  timing** component (the scaling-by-inverse-vol act itself) and a **return
  timing** component (whether the underlying signal predicts returns), and
  finds the two contribute differently depending on which volatility measure
  scales the position. The direct motivation for testing this project's own
  `frac` (return/trend timing) and `scale` (volatility timing) as separable
  contributors rather than only as a fused product -- the analogy is
  explicit and imperfect: Wang & Yan decompose *mean-return* alpha across a
  cross-section of equity factors, this round decomposes a *drawdown*
  property on one mechanism across six crypto instruments, so the same
  question (does one component or the other carry the effect) is being
  asked of a different statistic on a different asset class. Cited for the
  decomposition logic, not for a transferable magnitude.
- Bongaerts, D., Kang, X., & van Dijk, M. A. (2020), Financial Analysts
  Journal, 76(4) -- already `kelly_regime_v3`'s own citation for the
  conditional/extreme-only volatility targeting the conservative arm below
  reuses byte-identical.
- Baur, D. G., & Dimpfl, T. (2018), Economics Letters, 173 -- BTC's inverse
  leverage effect, already v3/v4's own grounding for why continuous
  vol-targeting is not simply strictly better here; relevant because the
  conservative arm is the purest possible test of the vol-target machinery
  alone, on instruments (the panel) where the inverse-leverage-effect finding
  has never itself been checked.
- Moreira, A., & Muir, T. (2017), Journal of Finance, 72(4) -- the baseline
  "volatility-managed alpha" result the conservative arm is the closest thing
  in this project to actually isolating and testing, since it removes the
  return-timing (`frac`) component Moreira & Muir's own volatility-only
  strategies also lack.
- Harvey, C. R., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M., & Van
  Hemert, O. (2018), Journal of Portfolio Management, 45(1) -- mechanical
  tail protection from vol-targeting, already v3/v4's grounding; the
  conservative arm tests whether that tail protection alone (no trend gate)
  produces v4's specific *matched-exposure* drawdown edge, which is a
  narrower and previously untested claim than "vol-targeting helps at all."

=====================================================================
WINDOWS AND COSTS (fixed before either branch is coded, identical to R-57
so results are directly comparable cell-for-cell)
=====================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.matched_hold import ConstantExposureHold, mean_notional  # noqa: E402
from experiments.r57_cross_asset_panel import (  # noqa: E402
    BEAR22,
    FULL,
    FUT_BASE,
    SPOT_BASE,
    SPOT_REAL,
    Asset,
    binomial_tail,
    load_candidates,
    measure,
    realized_vol,
    select_panel,
)
from tradebot.data import load_coinbase_spot, load_dataset  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns,
    max_drawdown_from_returns,
    paired_bootstrap,
    total_log_return,
)
from tradebot.registry import get_strategy  # noqa: E402

CONTROL_WINDOW = ("2020-04-01", "2022-12-31")  # BTC/ETH, pre-2023: +0 holdout
BOOT_KW = dict(mean_block=30.0, n_boot=2_000, seed=7)

# =====================================================================
# DECISION RULES, FROZEN (default is REJECT)
# =====================================================================
#
# D1 (PRIMARY). FULL window (2020-04-01 -> last bar), spot @0.10%. For each
# panel asset, compute the candidate's own mean notional c_mean (its ACTUAL
# realized average clipped exposure, not a target), build
# ConstantExposureHold(c_mean, deadband=0.10) as ITS OWN matched hold, and
# count assets (of 6) where the candidate's max drawdown is strictly lower
# than that matched hold's. Identical binomial convention to R-57/R-59/R-60/
# R-61 (n=6, one-sided, p=0.5 null):
#     6/6  -> REPLICATES     (p = 0.0156)
#     5/6  -> SUGGESTIVE, not established (p = 0.109)
#     <=4/6 -> FAILS TO REPLICATE
# Paired stationary-block-bootstrap interval on the drawdown difference
# reported per asset (30-day mean block, 2,000 resamples, seed 7), exactly
# as R-57's `cell()` does.
#
# D2 (CONTEXT). The candidate's own mean notional per asset, alongside v4's
# (already on record from R-57's `cells.csv`). Not a gate: it tells us
# whether the matched hold D1 compares against is even a similar risk level
# to v4's, or a different point on the exposure axis entirely (expected: the
# conservative arm's mean notional should be MUCH higher than v4's, since it
# never stands aside; the novel arm's should be lower, since c_const=1.0 with
# no vol-boost is smaller than v4's occasional up-to-2x breakout sizing).
#
# D3 (BTC/ETH CONTROL, +0 holdout). Identical D1 methodology, on BTC and ETH,
# CONTROL_WINDOW = 2020-04-01 -> 2022-12-31 (R-57's own control window, so
# this is directly comparable to R-57's control.csv, which found v4 itself
# 2/2 there). Diagnostic, not a gate: it asks whether deleting one factor
# breaks the property EVERYWHERE (both arms fail D3 too) or only on the
# panel (D3 holds, D1 doesn't) -- the second outcome is a cleaner isolation
# than anything R-59/R-60/R-61 produced, because it would mean the missing
# factor specifically matters off-BTC/ETH and not on them.
#
# D4 (0.40% FEE, CONTEXT). FULL window, spot @0.40%: does the candidate still
# beat buy_and_hold's final balance on a majority (>=5/6) of panel assets?
# Prediction, recorded now: FAILS, per this project's standing fee finding
# (R-12, R-13, R-47, R-57's own D2) -- named so a pass is read as a genuine
# surprise, not confirmation of something assumed.
#
# WHAT WOULD MAKE THIS FAIL (named now, before either branch is coded):
# D1 <= 4/6 for BOTH arms. That would mean neither the vote alone nor the
# vol-target alone reproduces the panel property -- a third independent line
# of evidence (after R-59's scale-axis, R-60's timing-axis) that it is not
# decomposable out of v4's own machinery, strengthening the "property of the
# panel's own price dynamics" reading without itself confirming it (this
# round does not test the panel's price dynamics directly; R-61's Hurst
# measurement already did that).
#
# FURTHER-WORK BAR (this pre-registration does NOT authorize a BTC/ETH
# holdout consultation; that is a separate, later decision exactly as R-61's
# own promotion bar states): an arm is worth a further round only if D1 >= 5/6
# AND D3 remains >= 1/2 (does not break the known BTC/ETH property) AND D4
# beats buy_and_hold in >= 4/6 panel assets. Anything else closes that arm's
# line for this round.
#
# Configurations evaluated: counted via `experiments.r57_cross_asset_panel
# .measure`'s module-level counter, reported honestly by each branch and
# summed across both for the round's total (ROUTINE.md: the trials count is
# the total across all parallel branches).
#
# Holdout cost: +0. No bar dated 2023-01-01 or later is read on BTC/ETH
# anywhere in this round (CONTROL_WINDOW ends 2022-12-31, matching R-57's own
# `cmd_control`); panel reads never touch a reserved holdout at all.


def load_panel() -> list[Asset]:
    """The frozen six-asset panel R-57 selected, loaded the identical way."""
    return select_panel(load_candidates())


def d1_verdict(k: int, n: int = 6) -> str:
    if k == n:
        return "REPLICATES"
    if k == n - 1:
        return "SUGGESTIVE (not established)"
    return "FAILS TO REPLICATE"


def further_work(d1: int, d3: int, d4: int, n_panel: int = 6, n_ctrl: int = 2) -> bool:
    """The further-work bar from the docstring above. NOT a promotion bar."""
    return d1 >= n_panel - 1 and d3 >= 1 and d4 >= n_panel - 2


# ------------------------------------------------------------- shared harness


def cell(strategy, label: str, a: Asset, window, market, rows: list) -> dict:
    """One asset x window x market cell for an arbitrary candidate strategy
    instance: candidate vs buy_and_hold vs candidate's OWN mean-notional-
    matched hold. Mirrors R-57's `cell()`, generalized to take a strategy
    object instead of a registry name so unregistered experimental
    strategies can reuse it.
    """
    from tradebot.window import run_period

    start, end = window
    cand_res = run_period(strategy, a.df, start, end, market=market, start_balance=1_000.0)
    from tradebot.metrics import compute_metrics

    cand = compute_metrics(cand_res)

    hold_res, hold = measure(get_strategy("buy_and_hold"), a.df, start, end, market)
    global _EXTRA_CONFIGS
    _EXTRA_CONFIGS[0] += 1  # count the candidate's own run too

    c_mean = mean_notional(cand_res)
    mh_res, mh = measure(ConstantExposureHold(c_mean), a.df, start, end, market)

    cand_ret = daily_returns(cand_res.equity).to_numpy(dtype=float)
    mh_ret = daily_returns(mh_res.equity).to_numpy(dtype=float)
    n = min(len(cand_ret), len(mh_ret))
    dd_matched = paired_bootstrap(cand_ret[:n], mh_ret[:n],
                                  max_drawdown_from_returns, **BOOT_KW)
    growth_matched = paired_bootstrap(cand_ret[:n], mh_ret[:n],
                                      total_log_return, **BOOT_KW)

    row = {
        "arm": label, "asset": a.ticker, "window": f"{start}:{end}",
        "market": market.name, "fee": market.fee_rate,
        "cand_final": cand.final_balance, "cand_dd": cand.max_drawdown_pct,
        "cand_sharpe": cand.sharpe, "cand_trades": cand.num_trades,
        "cand_liq": cand.liquidated,
        "hold_final": hold.final_balance, "hold_dd": hold.max_drawdown_pct,
        "c_mean_notional": c_mean,
        "mh_final": mh.final_balance, "mh_dd": mh.max_drawdown_pct,
        "cand_vol": realized_vol(cand_res.equity), "mh_vol": realized_vol(mh_res.equity),
        "dd_matched_diff": dd_matched.diff.point,
        "dd_matched_lo": dd_matched.diff.lo, "dd_matched_hi": dd_matched.diff.hi,
        "growth_matched_diff": growth_matched.diff.point,
        "growth_matched_lo": growth_matched.diff.lo,
        "growth_matched_hi": growth_matched.diff.hi,
    }
    rows.append(row)
    print(f"  [{label}] {a.ticker:5s} {market.name:11s} fee={market.fee_rate:.2%}  "
          f"cand ${cand.final_balance:>10,.0f} DD {cand.max_drawdown_pct:5.1f}% | "
          f"hold ${hold.final_balance:>10,.0f} DD {hold.max_drawdown_pct:5.1f}% | "
          f"matched(c={c_mean:.2f}) ${mh.final_balance:>10,.0f} "
          f"DD {mh.max_drawdown_pct:5.1f}% | "
          f"dDD_matched {dd_matched.diff.point:+6.1f}pp "
          f"[{dd_matched.diff.lo:+6.1f},{dd_matched.diff.hi:+6.1f}]")
    return row


_EXTRA_CONFIGS = [0]


def extra_config_count() -> int:
    """Candidate-strategy runs aren't counted by r57's `measure()` (which only
    wraps `run_period` for registered strategies called by name). Add this to
    `experiments.r57_cross_asset_panel.CONFIG_COUNT` for the honest total."""
    return _EXTRA_CONFIGS[0]


def d1_from_rows(rows: list[dict], label: str, market: str, fee: float, n: int = 6) -> tuple[int, "pd.DataFrame"]:
    df = pd.DataFrame(rows)
    d1 = df[(df.arm == label) & (df.market == market) & (df.fee == fee)]
    k = int((d1.cand_dd < d1.mh_dd).sum())
    return k, d1


def d4_from_rows(rows: list[dict], label: str, market: str, fee: float, n: int = 6) -> int:
    df = pd.DataFrame(rows)
    d4 = df[(df.arm == label) & (df.market == market) & (df.fee == fee)]
    return int((d4.cand_final > d4.hold_final).sum())


def run_control(strategy, label: str, rows: list) -> None:
    """D3: identical D1 methodology on BTC and ETH, CONTROL_WINDOW."""
    from tradebot.data import load_coinbase_spot as _lcs

    btc, _ = load_dataset(ROOT / "data", "spot")
    eth = _lcs(ROOT / "data", "ETH")
    for ticker, df in (("BTC", btc), ("ETH", eth)):
        asset = Asset(ticker, df, coverage=1.0, max_gap=pd.Timedelta(0), qualifies=True)
        cell(strategy, label, asset, CONTROL_WINDOW, SPOT_BASE, rows)
