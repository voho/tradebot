"""R-61: a genuinely different strategy family — mean-reversion, not trend —
sharing `kelly_regime`'s proven fractional-Kelly vol-targeted SIZE machinery,
evaluated on the panel before any BTC/ETH 2023+ holdout is touched.

Shared, frozen infrastructure for a two-branch parallel round. Per
ROUTINE.md's parallelism rules this file is neutral ground: both branches
import from it, neither branch edits it, and it does not itself run any
backtest. It exists so the pre-registration below is committed once, before
either branch reads a single strategy number.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

R-57 -> R-60 spent twenty-one attempts (R-34 -> R-46, R-53 -> R-56, R-59x2,
R-60x2) retuning `kelly_regime_v4`'s own vote-and-scale mechanism to restore
its matched-exposure drawdown property on R-57's six-asset panel (BCH, LTC,
ETC, DASH, LINK, XTZ) and failed every time (SIZE-axis record 0-for-21).
Both R-59 and R-60 converged, independently, on the same alternative
explanation neither had tested as a strategy: "the matched hold's advantage
on these higher-volatility, more mean-reverting instruments looks like a
buy-the-dip effect the panel's own price dynamics reward, not a sizing
mismatch" (R-59), and "the panel's matched-hold advantage ... looks like a
property of the panel's own price dynamics instead [of a v4 miscalibration]"
(R-60). R-60's own re-ranking then states, as its primary recommendation
rather than an alternative: "a genuinely different strategy family, evaluated
on the panel before any holdout consultation is spent, in preference to any
further variant of the incumbent." This round is that direction.

Constraint attacked: **SIZE** (reuses `kelly_regime`'s proven fractional-Kelly
vol-targeting exposure machinery, unchanged, so any result isolates the
*signal*, not the sizing) and **INFO** (a genuinely different read of the
same single price series — local deviation from a short-horizon mean, rather
than position relative to a long-horizon trend anchor).

**Not a duplicate of:**
- R-34 -> R-46, R-53 -> R-56, R-59, R-60 (twenty-one rounds, all retune v4's
  existing TREND vote's scale or timing; none replaces the vote's underlying
  *sign rule* with a reversion rule).
- L-13 `overshoot_fade`, L-24 `attrition_reversion`, L-25 `rsi_reversion`
  (this project's three prior mean-reversion strategies) — none uses
  fractional-Kelly vol-targeted sizing with a latching multi-horizon vote;
  L-25 is a bare RSI in/out rule, L-24 is an inventory-shifted fair-value
  fade needing an order book this project does not have, L-13 fades forced
  liquidations specifically. All three predate this project's SIZE finding
  (L-04, 08-14) and none combines a reversion signal with the sizing
  mechanism that turned out to be what actually works here.
- L-12 `harsanyi_crowd`'s belief-margin, R-34's SIZE-input reuse of it — both
  are still trend/crowding reads, not price-vs-own-mean reversion.

**Simulable here**: yes. Single-instrument `Strategy`/`prepare()`/`on_bar()`
API, no order book, no queue model — same causal rolling-window computation
`kelly_regime_v4` already uses, just a different rolling statistic. Reuses
R-57's committed panel loaders and harness (`tradebot.data.load_coinbase_spot`,
`experiments.r57_cross_asset_panel`'s `measure`/`select_panel`/`load_candidates`,
`experiments.matched_hold`'s `ConstantExposureHold`/`mean_notional`). No new
data.

**What would make this fail, named now, before either branch is coded:**
if the candidate does not beat `buy_and_hold` on a majority (>=5/6) of
PANEL_TRAIN assets (D1), OR does not draw down less than `buy_and_hold` on a
majority of them (D2), OR loses badly enough on the 0.40% fee tier that no
panel asset survives it (D4). Separately and **predicted in advance**: the
candidate is expected to UNDERPERFORM `kelly_regime_v4` on BTC and ETH (D3),
because the literature below places reversal in illiquid names and momentum
in the most liquid ones — BTC/ETH are the most liquid instruments in this
project by a wide margin. D3 failing does not by itself kill the panel
claim; D3 *passing* (candidate competitive with v4 on BTC/ETH) would be the
surprising result.

=====================================================================
LITERATURE (Step 2 sources, read and cited before either branch was coded)
=====================================================================

- Zaremba, A., Bilgin, M. H., Long, H., Mercik, A. R., & Szczygielski, J. J.
  (2021), "Up or down? Short-term reversal, momentum, and liquidity effects
  in cryptocurrency markets," International Review of Financial Analysis,
  78, 101928. Studied >3,600 cryptoassets 2015-2021: daily reversals are
  concentrated in illiquid coins, while the handful of the largest, most
  tradeable coins show daily momentum instead. This is the direct empirical
  motivation for testing a reversion signal on this project's *panel*
  (BCH/LTC/ETC/DASH/LINK/XTZ — all well outside BTC/ETH's liquidity tier)
  while predicting it will NOT beat the incumbent trend strategy on BTC/ETH
  themselves.
- Liu, Y., & Tsyvinski, A. (2021), "Risks and Returns of Cryptocurrency,"
  Review of Financial Studies, 34(6), 2689-2727, and Liu, Y., Tsyvinski, A.,
  & Wu, X. (2022), "Common Risk Factors in Cryptocurrency," Journal of
  Finance, 77(2), 1133-1177. Cited for the standing counter-evidence: at
  weekly cross-sectional-momentum horizons across thousands of coins
  (market/size/momentum three-factor model), momentum dominates and no
  reversal effect is found. Not a contradiction of Zaremba et al. so much as
  a different horizon/instrument mix (weekly cross-sectional momentum across
  a broad universe vs. Zaremba's daily single-asset reversal concentrated in
  the illiquid tail) — named here so the round's prediction is not reading
  one paper and ignoring another that could be seen as opposed.
- Hurst, H. E. (1951), "Long-Term Storage Capacity of Reservoirs," Trans.
  ASCE 116; Mandelbrot, B. B., & Wallis, J. R. (1969), "Robustness of the
  rescaled range R/S in the measurement of noncyclic long run statistical
  dependence," Water Resources Research 5(5); Lo, A. W. (1991),
  "Long-Term Memory in Stock Market Prices," Econometrica 59(5) — the
  classical rescaled-range Hurst exponent (H<0.5 anti-persistent/mean-
  reverting, H>0.5 persistent/trending) and Lo's documented upward bias
  under volatility clustering. This project's own `rolling_causal_hurst()`
  (R-46, `experiments/kelly_regime_v12_cppi_hurst.py`) already implements
  the classical multi-scale R/S estimator and is reused byte-identical by
  the novel branch below, not reimplemented. R-46 measured BTC's own rolling
  Hurst as persistently high (mean ~0.62) — i.e. BTC spends most of its time
  in the *trending* regime a Hurst-gated reversion strategy would sit out —
  which is independent, already-measured evidence for this round's own
  prediction that reversion should not compete with trend on BTC.
- Kelly, J. L. (1956); Bell, R. M., & Cover, T. M. (1980); Cardaliaguet, P.,
  & Lehalle, C.-A. (2018) — already `kelly_regime`'s own citations for the
  fractional-Kelly vol-targeted sizing machinery, reused unchanged by both
  branches below (see `KellyRegime.prepare`'s sizing loop, which neither
  branch edits).

=====================================================================
WINDOWS (fixed before either branch is coded)
=====================================================================

PANEL_TRAIN  2020-04-01 -> 2022-12-31   fit / select here ONLY. Identical to
                                        R-57/R-59/R-60's own window, for
                                        direct comparability.
PANEL_TEST   2023-01-01 -> 2026-08-20   panel assets only, descriptive
                                        generalization check, not a gate.
BTC_INNER_TRAIN  2017-01-01 -> 2020-12-31   BTC, D3 falsification only.
BTC_INNER_VALID  2021-01-01 -> 2022-12-31   BTC, D3 falsification only.
ETH_FULL     full committed ETH series (Coinbase, 2019-03 -> present) —
                                        the standing falsification asset
                                        (R-17/R-47/B-08 convention); not the
                                        reserved BTC holdout.

No bar dated 2023-01-01 or later is read on BTC anywhere in this round. Panel
reads (train or test) and the full ETH series cost the program's BTC/ETH
holdout counter **+0**, per the established R-47/B-08/R-57/R-59/R-60
convention: neither is the reserved BTC 2023+ holdout, and no panel or ETH
bar has ever fitted a parameter for the BTC-registered comparison table.

=====================================================================
MECHANISM (shared by both branches; only the vote's sign rule and, in the
novel branch, a multiplicative gate, differ from `KellyRegime.prepare`)
=====================================================================

`KellyRegime.prepare` builds `frac` (the 0..1 vote average, latched between
crossings) and `scale = min(target_vol / realized_vol, max_leverage)`, then
`desired = frac * scale` with a 10% deadband. That sizing loop is REUSED
BYTE-IDENTICAL by both branches (`target_vol=0.55`, `max_leverage=2.0`, the
shipped v4 defaults — this round does not retune sizing, only the vote). The
only change either branch makes is how `frac` is computed:

Conservative (`r61_conservative_zscore_reversion.py`): for each horizon in
`Z_HORIZONS_DAYS = (1, 3, 7)` (short, matched to Zaremba et al.'s DAILY
reversal timescale — not v4's 20/40/80-day ladder, which is the wrong clock
for a phenomenon reported at daily frequency), compute a rolling z-score
`z = (close - roll_mean(close, window)) / roll_std(close, window)` with
`window = days * BARS_PER_DAY`. Vote bullish (1.0, buy the dip) when
`z < -Z_THRESH`, bearish (0.0, stand aside) when `z > +Z_THRESH`, else hold
the previous verdict (identical hysteresis/ffill to v4). Average the three
horizon votes exactly as `KellyRegime` does. `Z_THRESH` is swept over
`Z_THRESH_GRID = (1.0, 1.5, 2.0)` on PANEL_TRAIN only, selected by D1's own
criterion (count of panel assets beating `buy_and_hold`, ties broken by mean
Delta max-drawdown), and the whole grid is reported so the choice is a
plateau check, not a peak.

Novel (`r61_novel_hurst_gated_reversion.py`): the IDENTICAL z-score vote as
the conservative branch, at conservative's own selected `Z_THRESH` (frozen
from conservative's PANEL_TRAIN selection — the novel branch tests one
additional mechanism, not a second free hyperparameter), multiplicatively
gated by a rolling causal Hurst exponent: `frac_gated = frac * gate`, where
`gate = 1.0 if H(t) < HURST_THRESH else 0.0`, `HURST_THRESH = 0.5` (the
literature's own structural midpoint, not fit), `H(t)` from this project's
own `rolling_causal_hurst(close, hurst_window_days=60)`
(`experiments/kelly_regime_v12_cppi_hurst.py`, imported not reimplemented).
Mechanism in one sentence: only let the reversion vote act while the asset
is *currently measured* to be in an anti-persistent regime; stand flat
through measured trending regimes rather than buying every dip regardless of
context.

=====================================================================
DECISION RULES, FROZEN (default is REJECT)
=====================================================================

D1 (PRIMARY). PANEL_TRAIN, spot @0.10%: count panel assets (of 6) where the
candidate's final balance beats `buy_and_hold`'s. Binomial convention
matching R-57/R-59/R-60 (n=6, one-sided, p=0.5 null):
    6/6  -> REPLICATES     (p = 0.0156)
    5/6  -> SUGGESTIVE, not established (p = 0.109)
    <=4/6 -> FAILS

D2 (RISK CHECK, panel). PANEL_TRAIN, spot @0.10%: count panel assets (of 6)
where the candidate's max drawdown is strictly below `buy_and_hold`'s (the
SIZE mechanism's own signature property, checked on the new signal).

D3 (BTC/ETH FALSIFICATION, chosen now, predicted to FAIL relative to v4 -
see "what would make this fail" above). BTC_INNER_TRAIN + BTC_INNER_VALID
and ETH_FULL, spot @0.10%: candidate vs `buy_and_hold` and vs
`kelly_regime_v4`. This is diagnostic, not a gate on the panel claim - but a
candidate that is not merely worse but *catastrophically* broken on BTC/ETH
(e.g. liquidation-grade drawdown, sub-buy_and_hold by an order of magnitude)
would indicate a bug rather than the predicted liquidity effect, and voids
the round pending a fix.

D4 (0.40% FEE TIER, panel, falsification). PANEL_TRAIN, spot @0.40%: count
panel assets (of 6) where the candidate still beats `buy_and_hold`. This
project's own standing finding (R-12, R-13, the README fee warning) is that
turnover-sensitive edges rarely survive the real entry tier; a reversion
strategy trades more often than a slow trend vote by construction, so this
is a real risk, stated now rather than after looking.

D5 (GENERALIZATION, panel, descriptive). PANEL_TEST, same D1/D2 methodology.
Not a gate (config frozen on PANEL_TRAIN only); reported for the record.

PROMOTION BAR (for "this mechanism deserves further work" - a BTC/ETH
holdout consultation is a SEPARATE, later decision this pre-registration
does NOT authorize): D1 >= 5/6 AND D2 >= 4/6 AND D4 >= 4/6 AND the
Z_THRESH_GRID is a plateau (D1 does not collapse to 0/6 at an adjacent grid
point). Anything else is NEGATIVE for promotion, whatever it says about the
underlying liquidity-conditional-reversal hypothesis (which can be confirmed
or refuted independently of whether either branch is promotable - see
below).

A SEPARATE, EXPLICIT reading, because this round's finding may be genuinely
informative without being promotable: if D1/D2 pass on the panel AND D3
shows the candidate losing to v4 specifically on BTC/ETH, that CONFIRMS
Zaremba et al.'s liquidity-conditional split as an explanation for why
`kelly_regime_v4`'s own mechanism does not travel (R-57's original question)
- a result worth recording even though it produces no new BTC-registrable
strategy, since this project's comparison table and every registered
strategy trade BTC only (README: "Paper-testing framework for BTCUSD
5-minute trading strategies"). The panel was built to fail strategies
cheaply before a holdout read (README), not to be a tradeable universe of
its own; a mechanism that only works off-BTC still answers this project's
standing question about why v4's property doesn't travel, which has been on
the backlog since R-57.

Configurations evaluated: counted via the same `measure()`-style counter
pattern as R-57/R-59/R-60 (`experiments.r57_cross_asset_panel.measure`
increments a module-level counter on every backtest), reported honestly by
each branch and summed across both for the round's total (ROUTINE.md: the
trials count is the total across all parallel branches).

Holdout cost: +0 (no BTC/ETH 2023+ bar is read anywhere in this round; see
"Windows" above).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.matched_hold import ConstantExposureHold, mean_notional  # noqa: E402,F401
from experiments.r57_cross_asset_panel import (  # noqa: E402,F401
    SPOT_BASE,
    SPOT_REAL,
    Asset,
    binomial_tail,
    load_candidates,
    measure,
    realized_vol,
    select_panel,
)

PANEL_TRAIN = ("2020-04-01", "2022-12-31")
PANEL_TEST = ("2023-01-01", "2026-08-20")
BTC_INNER_TRAIN = ("2017-01-01", "2020-12-31")
BTC_INNER_VALID = ("2021-01-01", "2022-12-31")

Z_HORIZONS_DAYS = (1, 3, 7)
Z_THRESH_GRID = (1.0, 1.5, 2.0)
HURST_THRESH = 0.5
HURST_WINDOW_DAYS = 60  # matches R-46's own default, reused not refit

TARGET_VOL = 0.55  # v4's shipped default, unchanged
MAX_LEVERAGE = 2.0  # v4's shipped default, unchanged
DEADBAND = 0.10  # v4's shipped default, unchanged
VOL_SPAN_DAYS = 8  # v4's shipped default (8 * BARS_PER_DAY), unchanged


def load_panel() -> list:
    """The frozen six-asset panel R-57 selected, loaded the identical way."""
    return select_panel(load_candidates())


def d1_verdict(k: int, n: int = 6) -> str:
    if k == n:
        return "REPLICATES"
    if k == n - 1:
        return "SUGGESTIVE (not established)"
    return "FAILS"


def promoted(d1: int, d2: int, d4: int, plateau_ok: bool, n: int = 6) -> bool:
    return d1 >= n - 1 and d2 >= n - 2 and d4 >= n - 2 and plateau_ok
