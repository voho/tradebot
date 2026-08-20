"""R-58: is `kelly_regime_v4`'s BTC-calibrated `target_vol`/`max_leverage`
the reason its matched-exposure drawdown property does not travel to other
instruments? (backlog **B-25**, filed by R-57.)

Shared, frozen infrastructure for a two-branch parallel round. Per
ROUTINE.md's parallelism rules this file is neutral ground: both branches
import from it, neither branch edits it, and it does not itself run any
backtest. It exists so the pre-registration below is committed once, before
either branch reads a single strategy number on the panel.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **SIZE** (this strategy family's sizing rule) and
**N≈3** (the panel is the only route to more than two instruments).

R-57 found that `kelly_regime_v4`'s one surviving property — a matched-
exposure drawdown advantage — is present on BTC and ETH and **inverts on 6
of 6** further Coinbase instruments (BCH, LTC, ETC, DASH, LINK, XTZ),
committed to this repo as `data/*_coinbase_spot_5m.csv.gz`. R-57 named, but
did not test, a hypothesis for *why*: `target_vol=0.55` / `max_leverage=2.0`
are constants tuned to BTC's own absolute volatility scale, and on
higher-volatility instruments the `target_vol / realized_vol` ratio is
structurally smaller — measured mean notional 0.18-0.26 on the panel vs.
0.38 BTC / 0.34 ETH over the same shared window — so the sizing term is
small and near-permanently binding, leaving mostly the vote's timing. B-25
asks the obvious next question: does making the exposure scale a function
of *which instrument* is being traded fix it?

**Not a duplicate of sixteen prior SIZE-axis rounds** (R-34, R-37, R-38,
R-40, R-41, R-45, R-46, R-53, R-54, R-55, R-56 — every one NEGATIVE, see
LEDGER.md section C): every one of those retuned a single global,
asset-independent parameter or mechanism, evaluated only on BTC and ETH —
a purely *temporal* axis of variation. This round is the first to vary
exposure sizing *cross-sectionally*, evaluated on the R-57 panel — an axis
that has never been tried on this strategy family. Not a duplicate of R-57
itself, which changed nothing about the strategy at all.

**Simulable here**: yes. Reuses R-57's committed panel loaders
(`tradebot.data.load_coinbase_spot`), R-57's matched-exposure harness
(`experiments/matched_hold.py`'s `ConstantExposureHold`/`mean_notional`,
`experiments/r57_cross_asset_panel.py`'s bootstrap/binomial helpers), and
`tradebot.inference`. No new data.

**What would make this fail, named now, before either branch is coded**:
if per-asset-normalized sizing still does not restore a majority (>= 5 of
6) matched-exposure drawdown advantage on the panel-train window, OR if
fixing generalization regresses v4's own already-established BTC/ETH
matched-exposure numbers (R-57's control cells: BTC dDD -5.6pp
[-20.0,+16.4], ETH -11.5pp [-17.3,+19.6], both vs. the mean-notional-
matched hold, 2020-04..2022-12) by more than 5 percentage points on either
asset. Either outcome is NEGATIVE and closes B-25, extending this family's
SIZE-axis record to 0-for-18 or 0-for-19.

=====================================================================
LITERATURE (Step 2 sources, read and cited before either branch was coded)
=====================================================================

- Baltas & Kosowski (2013/2017, "Demystifying Time-Series Momentum
  Strategies: Volatility Estimators, Trading Rules and Pairwise
  Correlations", J. Investment Management) — cross-instrument trend
  portfolios scale each instrument's bet by its OWN volatility estimate so
  every instrument contributes comparable risk; volatility-estimator choice
  materially changes turnover without degrading performance. This is the
  literature basis for the conservative branch: calibrate the scale
  constant per instrument rather than share one constant across instruments
  with different absolute volatility levels.
- Moskowitz, Ooi & Pedersen (2012, "Time Series Momentum", J. Financial
  Economics) — inverse-volatility position sizing so each instrument
  targets constant ex-ante volatility; already the mechanism `kelly_regime`
  uses *through time* on one asset. This round asks whether the same
  normalization needs to also apply *across* assets.
- Barroso & Santa-Clara (2015, "Momentum Has Its Moments", J. Financial
  Economics 116(1), 111-120) — scale a strategy by the inverse of its OWN
  trailing realized volatility to hit a constant target; risk is
  predictable from its own recent history and doing this nearly doubles
  momentum's Sharpe and suppresses crashes. The mechanism this round's
  novel branch borrows: normalize an asset's current volatility against
  its OWN trailing distribution (dimensionless, mean ~1 by construction)
  *before* applying a single global target, rather than fitting a new
  constant per asset.
- Bongaerts, Kang & van Dijk (2020, FAJ 76(4)) and Baur & Dimpfl (2018,
  Economics Letters 173) — already cited in `kelly_regime_v3`'s docstring;
  the conditional-targeting mechanism and the inverse-leverage-effect
  finding this strategy family is built on are BTC-specific measurements
  and neither paper claims they transfer across instruments, which is
  exactly the gap R-57 found and this round investigates.
- Springer FMPM (2025), "Cryptocurrency momentum has (not) its moments" —
  a direct test of the Barroso-Santa-Clara volatility-management mechanism
  on crypto; noted for context but not load-bearing here (equity/crypto
  momentum factor, not this project's regime-gated sizing mechanism).

=====================================================================
WINDOWS (fixed before either branch is coded)
=====================================================================

PANEL_TRAIN  2020-04-01 -> 2022-12-31   fit / select here ONLY. Includes the
                                        panel's own 2021 top and 2022 bear.
PANEL_TEST   2023-01-01 -> 2026-08-20   panel assets only, held untouched
                                        until each branch's configuration is
                                        frozen on PANEL_TRAIN. Reported as a
                                        generalization check, NOT a gate
                                        (single window per asset, so a fail
                                        here is informative, not fatal, on
                                        its own).
CONTROL      2020-04-01 -> 2022-12-31   BTC and ETH, R-57's own control
                                        window. No 2023+ BTC/ETH bar is read
                                        anywhere in this round.

Same convention as R-57: reading panel-asset data (train or test) costs the
program's BTC/ETH holdout counter **+0** — none of these are the reserved
BTC/ETH 2023+ holdout, they are new-instrument evidence, exactly like
R-47/B-08's ETH read and R-57's panel itself.

=====================================================================
ARMS, COSTS, METHODOLOGY (identical to R-57's D1, mean-notional axis only)
=====================================================================

Arms: candidate strategy; `buy_and_hold`; `ConstantExposureHold(c =
candidate's own mean clipped notional over the SAME window/asset/market)` —
the mean-notional-matched hold, R-57's primary axis. The equal-realized-
volatility axis is NOT re-run here (R-57 already established both axes
agree in direction on this panel); reused only if a branch's PANEL_TRAIN
result is ambiguous enough to need it, and any such use is reported as
robustness, not as the decision rule.

Costs: spot 0.10% primary (`SPOT_BASE`), spot 0.40% Bitstamp falsification
tier (`SPOT_REAL`) — both re-exported below from `r57_cross_asset_panel`.

=====================================================================
DECISION RULES, FROZEN (default is REJECT)
=====================================================================

D1 (PRIMARY). On PANEL_TRAIN, spot @0.10%: count panel assets (of 6) where
the candidate's max drawdown is strictly below the mean-notional-matched
hold's. Same binomial convention as R-57 (n=6, one-sided, p=0.5 null):
    6/6  -> REPLICATES     (p = 0.0156)
    5/6  -> SUGGESTIVE, not established (p = 0.109)
    <=4/6 -> FAILS

D2 (FALSIFICATION, chosen now). CONTROL window, BTC and ETH: the
candidate's matched-exposure drawdown advantage must not be worse than
v4's own R-57 control numbers (BTC -5.6pp, ETH -11.5pp) by more than 5
percentage points on either asset. A fix for six instruments that breaks
the two instruments the mechanism already works on is not a fix.

D3 (GENERALIZATION CHECK, reported, not a gate). PANEL_TEST, spot @0.10%,
same D1 methodology on the panel's 2023-2026 window. Descriptive: the
configuration was frozen on PANEL_TRAIN only, so a pass here is
corroborating, a fail is informative but does not by itself reverse a D1
promotion (n=6 assets x 1 window, same small-n caveat as D1 itself).

D4 (0.40% FALSIFICATION, from ROUTINE step 2's menu). PANEL_TRAIN, spot
@0.40%: candidate beats `buy_and_hold`'s final balance in >= 5 of 6 panel
assets. PREDICTION, recorded now: FAILS — R-13, R-47 and R-57's own D2 all
say the return edge does not survive the real entry tier, and nothing in
this round changes the strategy's return mechanism, only its risk scale.

PROMOTION BAR: D1 >= 5/6 AND D2 passes on BOTH BTC and ETH. Anything else
is NEGATIVE. A branch that clears the bar is a candidate for further work
(inner-validation on BTC/ETH's own pre-2023 data, then — only if that also
clears — a holdout consultation); this round's own pre-registration does
NOT authorize a BTC/ETH 2023+ holdout read, because the promotion bar above
must be cleared on the panel and the control before that consultation is
worth spending.

Configurations evaluated: counted per branch via the same `measure()`-
style counter pattern as R-57, reported honestly including solver
iterations, and summed across both branches for the round's total (per
ROUTINE.md: parallel trials count is the total across all branches, not
per branch).

Holdout cost: +0 (no BTC/ETH 2023+ bar is read anywhere in this round).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.matched_hold import ConstantExposureHold, mean_notional  # noqa: E402,F401
from experiments.r57_cross_asset_panel import (  # noqa: E402,F401
    RANKED_CANDIDATES,
    SPOT_BASE,
    SPOT_REAL,
    Asset,
    binomial_tail,
    load_candidates,
    realized_vol,
    select_panel,
)

PANEL_TRAIN = ("2020-04-01", "2022-12-31")
PANEL_TEST = ("2023-01-01", "2026-08-20")
CONTROL = ("2020-04-01", "2022-12-31")

# R-57's own recorded control-cell results (LEDGER.md R-57), cited here as
# already-published evidence, not a new backtest read.
R57_CONTROL_DD_ADVANTAGE = {"BTC": -5.6, "ETH": -11.5}  # percentage points
D2_REGRESSION_TOLERANCE_PP = 5.0

PANEL_TICKERS = ["BCH", "LTC", "ETC", "DASH", "LINK", "XTZ"]


def load_panel() -> list[Asset]:
    """The frozen six-asset panel R-57 selected, loaded the identical way."""
    return select_panel(load_candidates())


def d1_verdict(k: int, n: int = 6) -> str:
    if k == n:
        return "REPLICATES"
    if k == n - 1:
        return "SUGGESTIVE (not established)"
    return "FAILS"


def d2_passes(candidate_dd_advantage: dict[str, float]) -> bool:
    """candidate_dd_advantage: {'BTC': dDD_pp, 'ETH': dDD_pp}, same sign
    convention as R-57 (negative = candidate draws down less than the
    matched hold, i.e. better)."""
    for ticker, base in R57_CONTROL_DD_ADVANTAGE.items():
        cand = candidate_dd_advantage[ticker]
        if cand > base + D2_REGRESSION_TOLERANCE_PP:
            return False
    return True


def promoted(k1: int, dd_advantage: dict[str, float], n: int = 6) -> bool:
    return k1 >= n - 1 and d2_passes(dd_advantage)
