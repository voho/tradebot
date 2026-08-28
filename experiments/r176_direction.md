# R-176 pre-registration — dollar-volume activity-clock resampling of `kelly_regime_v4`'s vote (conservative) and a dollar-bar arrival-rate crowding gate (novel)

Status: FROZEN before either branch was dispatched or any real-data
performance number beyond the operator's own infra smoke-test was read.
The two smoke numbers in `experiments/r176_shared.py`'s own development
log (inner-train/inner-val, BTC spot only, default constants, no sweep)
are disclosed below in Step 2 for transparency, exactly as
r175_direction.md discloses its shared-engine self-test numbers — they are
NOT the pre-registered comparison (that runs on the frozen configuration
after each branch's own training-only sweep, per Step 3/4) and neither
branch may treat them as a target to beat.

## Step 0 recap

`git fetch origin main` + `git rev-parse HEAD origin/main` confirmed HEAD ==
origin/main (8dc67b6) before this round started; no `r<nn>_shared.py`
existed without a section B entry (newest was `r175_shared.py`, `grep -c
"R-175" docs/LEDGER.md` = 5). Step 0b's saturation count was 1 consecutive
null pass since R-173's own dispatch ("0-2: normal"). The backlog grep
returns only **B-48** (a documentation/formatting instrument fix, not a
strategy-research item) plus four already-inactionable rows (B-06
de-ranked, B-09 LOW, B-17 PARTIAL, B-28 blocked on data) — so this is a
fresh off-backlog literature-prompted round, the same convention R-160
through R-175 used. An announce commit ("R-176 (WIP): announce in-progress
round") was pushed to `main` before either branch's real code was written,
per the in-flight-discipline convention (R-131/R-133's lesson).

A dedicated research sub-agent (background, no code, no file writes) first
surveyed `docs/LEDGER.md` sections B (R-140 through R-175) and C in full to
find a genuinely untried sub-family rather than a 29th SIZE-axis plug-in or
a 15th ERR-axis validity construction — both flagged by that survey as
heavily picked-over classes. Its finding, confirmed independently by a
direct grep of section B for "dollar bar", "volume bar", "imbalance bar",
"activity time", "tick imbalance", "volume clock" (zero hits before this
entry): every prior INFO-axis round added a NEW EXTERNAL feature computed
from the SAME calendar-time-sampled `close` column; none has ever touched
the `volume` field already present in the committed OHLCV file, and no
round has ever changed the SAMPLING CLOCK (as opposed to the estimator)
underneath `kelly_regime_v4`'s vote or scale.

## The idea, in one sentence

Resample the inputs driving `kelly_regime_v4`'s regime vote (`frac`, the
three latched anchors) from a fixed calendar-day clock to a dollar-activity
clock built from the file's own `close * volume`, per Easley, Lopez de
Prado & O'Hara's "volume clock" and Lopez de Prado's information-driven
bars — conservative branch redefines the vote's anchor WINDOW SIZE in
dollar-time; novel branch derives a dollar-bar ARRIVAL-RATE "crowding"
signal from the same primitive and uses it as an independent multiplicative
haircut on top of the unmodified `frac * scale`.

## Step 1 — the four required questions

**1. Which constraint does it attack?** **INFO**, primarily: the `volume`
column is part of the already-committed OHLCV file but is consumed by
*zero* strategies in the `kelly_regime` family (v4's `prepare()` reads only
`close`); using it to redefine the vote's clock is a genuinely new
information channel by this project's own established convention (the same
one that let `camouflage_flow`/`stealth_trend`'s BVC-based flow signals and
`harsanyi_crowd`'s volume-efficiency crowding term count as attacking a
constraint beyond "another indicator"). Secondary: **SIZE** — the
conservative branch changes the vote (`frac`), which R-62 established
carries v4's entire matched-exposure signature, so a genuine timing change
here is a priori more likely to matter than the ~28 already-closed
scale-focused attempts. Does not attack ERR (no hypothesis test or
multiplicity correction is added) or N≈3 (no change to how regime *events*
are counted or resampled as an evidence base).

**2. Which ledger entries is it not a duplicate of, and why the difference
should matter?**

- **Not R-62** (factor isolation: vote alone vs. scale alone, both forced
  to frozen/constant values to measure which factor carries v4's
  signature). R-62 never substituted either factor's underlying
  *construction* — it held each one fixed or dropped it. This round
  substitutes what `frac`'s own anchor window MEANS (dollar-time instead of
  calendar-time), the natural next question R-62's own finding raises: if
  the vote carries the signature, does *how the vote is timed* matter?
- **Not R-99** (bipower-variation jump/continuous decomposition, an
  event-time construction via Barndorff-Nielsen-Shephard's own asymptotic
  jump test). R-99's "event time" is a statistical test on adjacent-return
  structure within a FIXED calendar-time series, feeding an additive
  regime-timing alarm; it explicitly does not touch `scale` and never
  resamples the underlying clock. This round changes the clock a rolling
  window is measured IN, not a jump-detection statistic layered on top of
  the existing calendar-time series.
- **Not R-102** (realized downside/upside semivariance decomposition of
  volatility, a SIZE-axis substitution for `scale`'s vol input). Different
  object entirely (this round never touches `scale`'s vol estimator; the
  conservative branch's `scale` is v4's own, byte-identical) and different
  data field (return sign vs. traded volume).
- **Not `camouflage_flow`/`stealth_trend`** (Bulk Volume Classification —
  splits volume into signed buy/sell flow via the volatility-standardized
  return, to predict DIRECTION). Neither branch here classifies flow by
  sign or predicts direction from volume at all; `dollar_volume =
  close*volume` is unsigned, and both branches only ever change (a) a
  rolling window's SIZE in dollar-time or (b) an exposure HAIRCUT gated on
  arrival RATE, never a directional signal.
- **Not `harsanyi_crowd`**'s crowding haircut (cuts exposure when a trend is
  OLD **and** volume efficiency — price progress per unit volume — is
  decaying; an Amihud-illiquidity-flavoured statistic conditioned on trend
  age). The novel branch's crowding gate is unconditional on trend age and
  measures a structurally different object: the RATE at which
  average-sized dollar bars arrive per unit calendar time, relative to its
  own causal trailing baseline — high when trading is simply fast, whether
  or not price is making progress or a trend is old.
- **Not R-175/R-136/R-08** (volatility *forecast quality* substitutions on
  `scale`). This round's conservative branch changes `frac`'s window, not
  any volatility estimator; its novel branch's haircut multiplies the
  existing `frac*scale` product from outside, the same application point as
  R-141's LPPLS dampener but built from a structurally different underlying
  statistic (dollar-bar arrival rate, not a bubble-confidence score) — the
  A2 kill-switch below checks this directly rather than merely asserting
  it.
- Never previously named in this ledger: zero hits for "dollar bar",
  "volume bar", "imbalance bar", "activity time", "tick imbalance", "volume
  clock", "adaptive window" (in the anchor sense), or "VWAP" before this
  entry.

**3. Is it simulable here?** Yes, with only data already in the repo. Both
branches read only the committed OHLCV file's own `close` and `volume`
columns at native 5-minute cadence — no order book, no tick-level trade
data, no per-side (buy/sell) attribution. The execution engine and its
bar-close-signal / next-open-fill convention are completely unchanged:
neither branch resamples the TRADING clock (fills still happen on native
5-minute bars) — only the internal WINDOW used to compute the vote's
anchors (conservative) or an exposure-haircut gate (novel) is computed on a
dollar-activity clock and then broadcast back onto the native calendar-time
index, the same "compute in one representation, broadcast onto native
bars" discipline r175_shared.py's daily-MSM engine and r102_shared.py's
signed-semivariance engine both already use.

**4. What would falsify it, named now?**

- **(a) Conservative branch, falsified if either fires:** the paired
  bootstrap CI on `d_log_growth`/`d_sharpe` (dollar-time vote vs. v4's own
  calendar-time vote) excludes zero on the LOSING side on BTC
  inner-validation, on either market; OR the R² between the dollar-time
  vote and v4's own calendar-time vote exceeds 0.98 (mere relabeling — a
  real risk, since BTC's own trading activity has not been perfectly flat
  across 2017-2022 but could still be smooth enough that a dollar clock and
  a calendar clock rarely disagree on which side of a 1% band price sits).
- **(b) Novel branch, two-part, either kills it, mechanism check first and
  decisive independent of any performance number:** the crowding haircut
  must (i) actually bind a non-trivial-but-not-overwhelming fraction of BTC
  inner-train bars (between 2% and 40% of bars in the crowded state — a
  range disclosed now, before any real-data run, wide enough not to be a
  configuration knob in disguise) AND (ii) not merely relabel v4's own
  existing volatility-breakout hysteresis: R² between the haircut's
  crowded-state indicator and v4's own `scale`-state breakout indicator
  (`full[i] != steady[i]` in `conditional_target_scale`) below 0.5. If
  either fails, the branch is falsified on mechanism alone. Only if both
  pass does the standard four-clause promotion bar apply to its
  performance numbers.

## Step 2 — citations and design

**Mechanism citations.**
- Easley, D., Lopez de Prado, M. M., & O'Hara, M. (2012). "The Volume
  Clock: Insights into the High-Frequency Paradigm." *The Journal of
  Portfolio Management* 39(1), 19-29. Establishes that sampling market data
  as a function of trading activity (volume/dollar throughput), rather than
  fixed calendar time, produces observations with more uniform statistical
  properties — the paper this project's own `camouflage_flow` module
  already cites for VPIN/BVC, cited here instead for its OTHER, distinct
  contribution: the sampling-clock argument itself, applied to a rolling
  window rather than to flow classification.
- Lopez de Prado, M. M. (2018). *Advances in Financial Machine Learning*,
  Wiley, ch. 2 ("Financial Data Structures"). Formalizes tick/volume/dollar
  bars and information-driven (imbalance) bars; documents that returns
  sampled in activity time are closer to i.i.d. Gaussian with weaker serial
  correlation than calendar-time bars of the same asset — confirmed live by
  WebSearch this round (multiple independent secondary sources summarizing
  the same claim, dollar bars specifically preferred over volume bars for
  being invariant to the asset's own price level, which matters here given
  BTC's 750x price range across this project's own 2017-2026 sample).
- Baur, D. & Dimpfl, T. (2018). "Asymmetric Volatility in Cryptocurrencies."
  *Economics Letters* 173. Already this project's own citation for
  `kelly_regime_v3`/`v4`; cited here only as the reason the novel branch's
  crowding gate multiplies `scale` from OUTSIDE rather than replacing its
  vol input (R-08/R-136/R-175's now-repeated finding that more *reactive*
  volatility estimates hurt this architecture on BTC's inverse leverage
  effect) — the crowding gate is deliberately built to be orthogonal to vol
  *level*, gating on trading *pace* instead.

**What was checked and rejected before settling on this candidate.**
Canonical tick/volume/dollar IMBALANCE bars (the recursively-estimated
expected-imbalance threshold construction) were considered for the novel
branch directly but require genuine per-trade tick data this project does
not have (only OHLCV+volume); building a faithful implementation from
bar-level data alone would be simulating data the file does not contain,
which Step 1 Q3 and this project's standing rule ("never proxy unavailable
data out of price") both rule out. The bar-level arrival-intensity proxy
below is disclosed as a simplification of that construction, not a claim to
reproduce it. A full re-implementation of the backtest engine's trading
clock itself (executing on dollar bars rather than native 5-minute bars)
was rejected per Step 1 Q3's own instruction not to proxy a missing
simulation capability — instead, both branches compute their
dollar-activity-clock statistic and broadcast it onto the unchanged native
grid, exactly as every prior daily-refit round in this ledger already does.

**Engine parameters, fixed before any real-data comparison** (see
`experiments/r176_shared.py`): 180-day causal trailing MEDIAN daily-dollar
baseline (matching v3/v4's own `anchor_span_days=180` convention), shifted
by one day (no same-day lookahead), minimum 30 days of history before
either branch leaves its cold-start `NaN`/no-op fallback; novel branch's
intensity smoothed over a trailing 1-day window, its crowding threshold a
causal trailing 90-day quantile (enter above the 90th percentile, exit
below the 60th — the same hysteresis SHAPE as v3/v4's own high/low
breakout state machine), haircut fixed at 0.5 while crowded.

**Development-time infra smoke numbers (BTC spot only, unswept defaults,
disclosed per r175_direction.md's own convention — NOT the pre-registered
result):** dollar-time vote alone scored `d_sharpe -0.31` (inner-train,
excludes zero on the losing side) / `-0.38` (inner-val, CI does not exclude
zero: [-0.433, +0.024]); crowding-gate-alone scored `d_sharpe -0.22`
(inner-train, excludes zero) / `-0.12` (inner-val, CI does not exclude
zero: [-0.220, +0.111]). Both are single unswept points, reported for
disclosure only — Step 3's job is a training-only sweep of each branch's
own free parameters (window length for the conservative branch; haircut
magnitude and the two quantile thresholds for the novel branch) against
inner-train/inner-validation, not a verdict on these numbers.

## Conservative variant

**Mechanism, one sentence.** Replace the 20/40/80-calendar-day rolling-mean
anchors inside v4's own latched 3-anchor vote with `dollar_time_anchor`'s
dollar-activity-clock windows (same target dollar volume = `days *`
causal 180-day median daily dollar volume), leaving the band, latching,
`scale`, `target_vol`, `max_leverage` and the 10% deadband byte-identical
to `kelly_regime_v4` today.

**Free parameter for the training-only sweep:** `BASELINE_WINDOW_DAYS`
(the causal baseline's own trailing window — default 180, sweep e.g.
{90, 120, 180, 270, 365}), evaluated on inner-train/inner-validation only,
never the holdout.

## Novel variant

**Mechanism, one sentence.** Multiply v4's own unmodified `frac * scale`
by a latching crowding haircut (0.5 while dollar-bar arrival intensity sits
above its own causal 90th-percentile trailing threshold, 1.0 otherwise,
releasing below the 60th percentile), leaving the vote and scale
byte-identical to `kelly_regime_v4` today.

**Free parameters for the training-only sweep:** `CROWDING_HAIRCUT`
(default 0.5, sweep e.g. {0.25, 0.4, 0.5, 0.7}) and the
`(high_in_q, high_out_q)` threshold pair (default (0.90, 0.60), sweep e.g.
{(0.85,0.55), (0.90,0.60), (0.95,0.70)}), evaluated on
inner-train/inner-validation only, never the holdout.

## Honest prior, stated before any code beyond the shared-engine smoke test ran

The disclosed smoke numbers above are both negative on inner-train
(excluding zero on the losing side, unswept defaults) and inconclusive on
inner-validation. Combined with this ledger's now-repeated finding that
more *reactive* signal-timing constructions on this architecture tend to
add turnover without added edge (R-165's rate/destination axis, closed for
both `frac` and `scale`; R-08/R-136/R-175's forecast-quality inversions),
the prior on either branch clearing the promotion bar is **low, on the
order of 10-15%** — comparable to R-175's own stated prior. The more
informative possible outcome is the novel branch's mechanism check (b):
independent of any Sharpe number, it directly tests whether "the market's
own activity clock has sped up" is a real, non-degenerate, non-relabelled
signal on this data at all — a clean negative on that check alone would be
a useful, disclosed boundary on how far the volume column can be pushed
before this project's INFO well really is exhausted.
