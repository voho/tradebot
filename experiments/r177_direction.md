# R-177 pre-registration — a signed (long-short) exposure for `kelly_regime_v4`

Status: FROZEN before either branch was dispatched or any real-data
performance number was read. This document, `experiments/r177_shared.py`
and its self-test above were committed to `main` before either branch's
code was written, per the in-flight-discipline convention (R-131/R-133's
lesson).

## Step 0 recap

`git fetch origin main` + `git rev-parse HEAD origin/main` confirmed HEAD
== `origin/main` (`b6ca1dc`) before this round started; no `r<nn>_shared.py`
existed without a section B entry (newest was `r176_shared.py`, `grep -c
"R-176" docs/LEDGER.md` = 6). Step 0b's saturation count, run per
ROUTINE.md's own command
(`awk '/^## E\. Verification/,/^### E-archive/' docs/LEDGER.md | grep -E
"^\|" | awk -F'|' '$2 ~ /^ *#/ {next} $2 ~ /—/ {exit} {n++} END {print
n+0}'`), returned **2** consecutive null passes since R-176's own dispatch
— "0–2: normal", a fresh sweep is warranted. The backlog grep returns only
**B-48** (a documentation/formatting instrument fix, not a strategy-research
item) plus four already-inactionable rows (B-06 de-ranked, B-09 LOW, B-17
PARTIAL, B-28 blocked on data) — so this is a fresh off-backlog
literature-prompted round, the same convention R-160 through R-176 used.

A dedicated research sub-agent (background, no code, no file writes) first
surveyed `docs/LEDGER.md` section C's ~75-row ruled-out table in full and
the most recent rounds in section B (R-160 through R-176), then ran real
web searches for a genuinely untried mechanism, before proposing this
direction. Its finding, confirmed independently by the operator with a
direct grep of the full 25,600-line ledger for `\bshort\b` (excluding
"shorter"/"shortfall"/routine prose) and separately for `negative
exposure`/`inverse position`/`long-short`: no round has ever given
`kelly_regime_v4` or any of its SIZE-axis variants a directional short
against its own vote's bear state. This is confirmed at the code level,
not just the ledger's prose level — `kelly_regime.py`'s own class
docstring states plainly that the strategy "stands flat rather than
shorting a historically upward-drifting asset," and
`kelly_regime_v6_state_kelly.py` (R-37's own novel branch) hard-codes
`kelly_f = np.clip(kelly_f, 0.0, None)` with the explicit inline comment
"never short a state; mirrors v4."

## Step 1 — the four-question filter, answered before any branch code runs

**1. Which constraint does it attack?** **SIZE**, primary and only. This
changes nothing about detection (the vote's three anchors, band, and
latching hysteresis are byte-identical to v4 in both branches) — only how
much, and in which direction, is held given an already-validated vote
state. "Another indicator" attacks nothing on this list; this is not
that — it is a reparametrization of the existing signal's own range.

**2. Which ledger entries is this not a duplicate of?**

- **R-37/R-38 (08-19, novel branch, per-state Kelly).** R-37's novel
  branch built the exact per-state causal `mu_state/sigma_state**2`
  estimator this round's `state_kelly_stats` reuses verbatim, and *it
  measured* the bear state's own forward drift at roughly −62%/yr and the
  1/3 state at roughly −101%/yr — genuinely negative, non-trivial numbers.
  But R-37's own construction floors `kelly_f` at zero for every state
  (see quote above) and then multiplies by `frac` (which is itself 0 for
  the bear state), so the measured negative number was never converted
  into a position; it was reported and discarded. This round is the
  specific, named alternative R-37 built the machinery for but declined to
  run. R-38 (Busseti-Ryu-Boyd risk-constrained Kelly gambling) is a
  different sizing rule entirely (a convex-optimization risk budget, not a
  per-state drift estimate) and never touches sign either.
- **L-22 `macd_cross` / L-25 `rsi_reversion`.** Both already mirror a short
  on futures and both are registered NEGATIVE ("fee death", 4,301 and
  4,464 trades). They are not this direction's precedent: both are raw
  threshold-crossing signals traded at full notional on every flip, with
  no Kelly/vol-target sizing, no hysteresis, no deadband — the opposite of
  the low-turnover, risk-managed execution discipline this project's own
  standing diagnosis credits for every strategy that actually makes money
  ("every strategy that decides how much to hold makes money"). Applying
  a raw short to an unmanaged crossover signal and applying a signed,
  vol-targeted, deadbanded Kelly fraction to a 3-anchor latched vote are
  different execution regimes, and this ledger has never tested the
  latter.
- **R-63 (cross-sectional panel momentum).** Its own text notes a
  "long-short... which this round cannot run" — that refers to a
  multi-instrument, shared-margin book (no cross-asset margin model exists
  in `tradebot.multi_engine`), not a net directional short on ONE
  instrument's own futures position, which the existing single-instrument
  `MarketSpec.futures(allow_short=True)` broker already supports natively
  and has supported since before R-01.
- **R-76 (cointegration/distance pairs).** Uses long/short machinery only
  as a market-NEUTRAL spread between two different assets (net exposure
  ≈0); never a net directional short on one asset's own bear state.
- **R-89/R-90 (asymmetric thresholds, trailing stops).** Timing/exit
  mechanisms layered on the existing long-only exposure; neither changes
  the sign of what is held.

**3. Is it simulable here?** Yes, with the one disclosed narrowing above:
5m OHLCV bars, bar-close signals, next-open fills, no order book — but
mechanically restricted to `futures_5x` (`allow_short=True`); `spot`
(`allow_short=False`) is reported only as an identity check (a signed vote
that never goes negative on spot reproduces v4's spot numbers exactly,
since the broker clips the lower bound to 0.0 regardless of what the
strategy requests).

**4. What would make it fail, named now, before any code beyond the
shared engine above has been read against real numbers?** Named
explicitly in `r177_shared.py`'s own disclosed-risks section and repeated
here as the actual pre-registered falsification: **this project's own
evidence predicts this will fail.** BTC's measured stress episodes (the
same six-episode panel every regime-timing round in this ledger has
already characterized) are dominated by violent V-shaped reversals, not
sustained bear trends — twelve prior regime-timing mechanisms have already
failed to time entries/exits around exactly this shape. A short entered
and held on the vote's own multi-day hysteresis schedule is expected to
be disproportionately whipsawed by that V-shape, converting the vote's own
proven "protect capital by standing aside" property into "actively lose
money forecasting a reversal down, then get run over by the recovery" —
the textbook signature of every directional predictor already in this
project's comparison table (macd_cross, rsi_reversion, elliott_wave, and
all seven pure game-theoretic strategies below `champions_council`).
Additionally, funding (R-13/R-14, COST axis, already the one cost proven
to move a verdict in this ledger) is known to swing sharply negative
during capitulation — precisely when a short would otherwise pay off on
price alone — a mirror-image cost this direction's short leg pays and its
long-only predecessor never has.

## Step 2 — branch design

### Conservative branch — sign-symmetric vote, v4's own scale unchanged

**Mechanism.** `target = signed_vote_frac(close) * scale`, where `scale`
is v4's own unmodified `min(target_vol/realized_vol, max_leverage)` — the
ONLY change from `kelly_regime_v4` is the affine remap of `frac` from
`[0,1]` to `[-1,1]` (see `r177_shared.py::signed_vote_frac`). Zero new
parameters. Same 10% deadband, same cap, same hysteresis.

**Pre-registered failure modes:** (a) the whipsaw argument above — the
short leg loses more from V-shaped reversals than the long leg gains from
sustained trends; (b) liquidation risk rises materially on 5x futures
relative to unmodified v4 (v4 itself never liquidates in this project's
own record; a short doubles the ways a position can be wrong-footed); (c)
turnover roughly doubles (the vote now crosses zero at the same points it
used to cross into/out of "flat", which is a real trading event only when
signed).

### Novel branch — the specific alternative R-37 built and declined to run

**Mechanism.** Reuse R-37's exact causal per-state `mu_state/sigma_state**2`
estimator (`r177_shared.py::state_kelly_stats`, `floor_at_zero=False`),
`scale_state = clip(kelly_mult * kelly_f_state, -max_leverage,
max_leverage)`, `target = sign(frac - 0.5 or 0) ... ` — precisely:
`target = clip(kelly_mult * kelly_f_state, -max_leverage, max_leverage)`
directly (the per-state Kelly ratio already carries its own sign once
unfloored, so — unlike the conservative branch — the vote's magnitude
does not additionally multiply it; the vote instead selects WHICH state's
estimate applies, exactly as R-37's own construction already did for the
long side). Same 10% deadband, same `min_obs=2,000` occurrence floor R-37
used, same halflife/kelly_mult grid R-37 swept.

**Pre-registered failure modes:** (a) R-37's own already-measured
data-hunger problem (non-monotone kelly_mult response, a fitted 330-450d
halflife peak rather than a plateau, failure on the BTC control once
routed through ETH's shorter window) inherited verbatim, now on both
signs at once; (b) the bear-state estimate, unfloored, is even noisier
than the already-fragile positive states R-37 measured, because bear
episodes are rarer in this project's own data than bull episodes; (c) the
same whipsaw/funding risk named for the conservative branch, now sized
by a fitted magnitude rather than a fixed vote fraction, which could make
the failure mode either better-controlled (smaller in the least reliable
state) or worse (larger if the estimator is fooled) — named as genuinely
undetermined in advance rather than assumed.

## Falsification rule, frozen for both branches

Promote only if **all** of the project's own standing promotion-bar
clauses hold (ROUTINE.md Step 4): beats `buy_and_hold` out-of-sample on
`futures_5x` after real costs (funding charged); the improvement clears
the **±0.2 Sharpe noise floor** or is a genuine, risk-matched (R-33's
rule) drawdown/tail improvement; survives the falsification test above
(does not fail more on the pre-registered stress panel/ETH than it gains
on the primary comparison); the parameter neighbourhood is a plateau, not
a peak. Anything else is NEGATIVE, reported with the same care as a win,
per ROUTINE.md's own default.

Both branches must report, for `futures_5x` specifically: liquidation
rate across the Monte Carlo stress windows (not just the point estimate),
mean notional/realized vol against v4 (R-33's exposure-match discipline),
and turnover in both trade-episode and fill-count units (R-131's own
lesson about the two units differing 300x on exactly this kind of round).
