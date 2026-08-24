"""R-116: does the trend-vote of the six OTHER instruments this project
already has 5m spot data for (BCH, LTC, ETC, DASH, LINK, XTZ) -- run through
`kelly_regime_v4`'s own, unmodified 20/40/80-day anchor-vote construction,
independently per instrument -- carry information about BTC's own regime
that BTC's own vote does not already have?

Shared, frozen infrastructure for a two-branch parallel round, scoped by an
R-116 direction-finding pass (grep of docs/LEDGER.md + a literature search)
run before either branch was coded. Per ROUTINE.md's parallelism rules this
file is neutral ground: both branches import from it, neither branch edits
it, and it does not itself define a candidate strategy or compute a verdict.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **INFO** (primary -- a structurally new *kind* of
information, contemporaneous cross-instrument agreement, not a new external
feed and not a transform of BTC's own price/vote) and **ERR** (secondary --
a fifth notion of uncertainty for the vote's own confidence: cross-ASSET
disagreement, as opposed to the four already closed: sampling significance
R-87/R-104; cross-MODEL-class disagreement on one asset R-105/R-106;
distributional novelty of the market state R-109/R-112/R-115; temporal
duration dependence of the regime's own age R-114).

**Not a duplicate of:**

- R-76 (cointegration / distance-method PAIRS TRADING between two panel
  instruments). That round trades a price spread between two legs; this
  round never trades an alt, and never forms a spread -- it only reads each
  alt's OWN independent trend-vote as an input to BTC's single-asset
  strategy.
- R-63 / R-65 / R-67 / R-68 / R-107 / R-110 / R-111 / R-113 (`xsmom_entry_band`
  and the whole cross-sectional-momentum lineage). All eleven of those
  rounds *trade* the 8-instrument panel as a portfolio (cross-sectional rank,
  timing, allocation, error control on the PANEL's own positions). None of
  them ever feeds the panel's information back into `kelly_regime_v4`'s
  single-asset BTC decision -- the panel here is a witness, not a book.
- R-100 (Binance-vs-Deribit *funding-rate* term-structure divergence) and
  B-05/R-35/R-39 (funding as a COST-axis gate). Both are derivative-market
  quantities on BTC alone; this round uses SPOT price trend on six other
  instruments, not funding, and not BTC's own market.
- R-41 (Deribit BTC spot/perp *basis*, one instrument, one venue-pair).
- R-105 / R-106 (disagreement across independent MODELS on the SAME BTC
  series -- BOCPD, Kalman, CSD, Hawkes, anchor-ladder jackknife). This round
  holds the MODEL fixed (v4's own 20/40/80 anchor vote, byte-identical) and
  varies the ASSET instead -- the flipped construction, and the first time
  this project has asked whether v4's own detector, not a different one,
  agrees with itself when pointed at a different instrument.

Confirmed by grep of docs/LEDGER.md (R-116 direction-finding pass): no prior
round applies `kelly_regime_v4`'s own vote construction to the six-Coinbase-
instrument panel as a CONFIRMING/DISAGREEMENT INPUT to the BTC strategy.

**Literature grounding** (fetched by the R-116 direction-finding pass, not
assumed):

- Zaremba, Szyszka, Karathanasopoulos & Mikutowski (2019/2020), "Herding
  for Profits: Market Breadth and the Cross-Section of Global Equity
  Returns," *Economic Modelling* (SSRN 3444882). 64 countries, 1973-2018:
  market breadth (fraction of advancing constituents) predicts returns
  incrementally over size/style/vol/skew/time-series-momentum signals.
  Equities, no crypto, no cost model verified here -- motivating, not
  load-bearing.
- Mercik, Bedowska-Sojka, Karim & Zaremba (2025), "Cross-sectional
  interactions in cryptocurrency returns," *International Review of
  Financial Analysis* 97(C). Documents real cross-sectional interaction
  structure among crypto returns; post-cost behaviour not verified here.
- Zweig (1986) breadth-thrust: the classical precedent for breadth as a
  CONFIRMING signal (rare, high-conviction agreement/disagreement), not a
  standalone forecast -- exactly the role (discount/confirm an existing
  vote) both branches below test, not the stronger claim that breadth alone
  predicts returns.
- Zarnowitz & Lambros (1987, *JPE*) and Bomberger (1996, *JMCB*): already
  the R-106 ERR-axis citations for disagreement-as-uncertainty; reused here
  for the NOVEL branch's framing, with the panel varied by ASSET instead of
  MODEL.

**Is it simulable here?** Yes, zero new data. `data.load_coinbase_spot()` /
`r63_shared.load_universe(UNIVERSE_6)` are already committed and causal;
this round reads them only, no new file.

**What would make it fail, named now, before any code ran.** R-63 already
measured this exact six-alt panel's mean pairwise daily-return correlation
at 0.634 and its Grinold equal-correlation breadth at 1.47-of-8. That is
direct, already-computed evidence that the six alts carry very little
information *independent* of BTC/ETH -- so the single most likely failure
mode, named before any bar of this round was read, is that a breadth/
disagreement statistic built from them is either (a) near-collinear with
v4's own vote (fails a Step-0 R^2 gate before any economic test can even
run) or (b) genuinely informative but LAGGING, reproducing the exact
failure mode common to 16 of 16 prior INFO-axis attempts (R-44 through
R-115) and the "real but inert" pattern common to 6 of 7 prior ERR-axis
attempts (R-87 through R-114). A clean NEGATIVE closing this as INFO
signal #17 and ERR notion #5 is the fully expected, fully successful
outcome of this round -- see docs/ROUTINE.md's own framing: "a
well-documented negative result is a successful day."

=====================================================================
SHARED INFRASTRUCTURE
=====================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reused, not duplicated: both files are read-only dependencies that
# predate this round and are not touched by it.
from experiments.r63_shared import (  # noqa: E402,F401
    UNIVERSE_6,
    UNIVERSE_8,
    W_TRAIN,
    W_VAL,
    W_HOLD,
    align_frames,
    compare,
    d1_pass,
    d2_pass,
    d3_pass,
    further_work,
    excludes_zero,
    load_universe,
    v4_targets,
    check_causality,
)
from experiments.r82_shared import (  # noqa: E402,F401
    V4_BAND,
    V4_HORIZONS,
    STRESS_EPISODES,
    anchor_majority,
    anchor_votes,
    block_bootstrap_shifts,
    episode_window,
    nearest_transition,
)

from scripts.experiment import OOS_START  # noqa: E402

# Trials counter: report the sum across BOTH branches in the ledger entry,
# per ROUTINE.md's parallelism rules. Each branch increments this on every
# backtest it runs (train, inner-val OR holdout -- count all of them).
_CONFIGS = [0]


def config_count() -> int:
    return _CONFIGS[0]


def note_config(n: int = 1) -> None:
    _CONFIGS[0] += n


# ----------------------------------------------------------------- panel


def build_panel_votes(tickers: tuple[str, ...] = UNIVERSE_8,
                       window: tuple[str, str | None] = W_TRAIN,
                       ) -> pd.DataFrame:
    """`kelly_regime_v4`'s own vote fraction (in {0, 1/3, 2/3, 1}), computed
    INDEPENDENTLY per instrument with v4's byte-identical 20/40/80-day
    anchors and 1% band, for every ticker in `tickers`, aligned onto one
    shared 5m grid inside `window`.

    Causal: `align_frames` only ever forward-fills a PAST bar onto a bar an
    exchange did not print (R-63's own causality-verified primitive); each
    column is then `anchor_majority`, itself a rolling-mean + ffill/latch
    construction that depends only on rows <= i. Composing two causal
    transforms is causal.
    """
    frames = load_universe(tickers)
    aligned = align_frames(frames, window)
    votes = {t: anchor_majority(aligned[t]) for t in tickers}
    return pd.DataFrame(votes)


def agree_frac(panel_votes: pd.DataFrame, home: str) -> pd.Series:
    """Fraction of the OTHER tickers in `panel_votes` whose vote-side
    (bullish: vote >= 0.5, else bearish) agrees with `home`'s vote-side, at
    every bar. In [0, 1]; 1.0 means unanimous agreement with `home`.
    """
    others = panel_votes.drop(columns=[home])
    home_side = (panel_votes[home] >= 0.5).astype(float)
    other_side = (others >= 0.5).astype(float)
    agree = other_side.eq(home_side, axis=0)
    return agree.mean(axis=1)


def panel_disagreement(panel_votes: pd.DataFrame) -> pd.Series:
    """Bomberger (1996)-style disagreement: the cross-sectional standard
    deviation of ALL tickers' vote fractions (including `home`, if present
    in `panel_votes`) at every bar. Continuous, in [0, ~0.5]. High values
    mean the panel is split; 0 means unanimous (whether bullish or
    bearish).
    """
    return panel_votes.std(axis=1, ddof=0)


def r_squared(a: pd.Series, b: pd.Series) -> float:
    """Step-0 collinearity gate: how much of `a` is already explained by a
    flat rescaling of `b`? An R^2 > 0.95 here is this project's own
    standing artifact signature (R-73's flat-rescale failure mode) and
    should be reported and treated as a near-automatic kill, per that
    round's lesson, BEFORE any economic backtest is read.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.std(b) == 0:
        return float("nan")
    corr = np.corrcoef(a, b)[0, 1]
    return float(corr ** 2)


def attach_to_btc(btc_df: pd.DataFrame, signal: pd.Series, col: str) -> pd.DataFrame:
    """Reindex a panel-derived signal (built on the aligned panel grid) onto
    `btc_df`'s own native index, forward-filling only PAST panel values --
    still causal, same guarantee `align_frames` documents. Bars before the
    panel's own first aligned timestamp are filled with 0.0 (neutral: no
    panel confirmation/disagreement available yet, matching the anchor
    vote's own warmup convention of latching to a neutral default).
    """
    out = btc_df.copy()
    reindexed = signal.reindex(out.index, method="ffill")
    out[col] = reindexed.fillna(0.0)
    return out
