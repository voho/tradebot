# kelly_regime_v17_stablecoin_corroborate — R-56 NOVEL branch (08-20), closing B-23 (fix 2 of 2)

Unregistered experiment. Code: `experiments/kelly_regime_v17_stablecoin_corroborate.py`.
Signal module reused read-only, unedited: `experiments/_stablecoin_signal.py`
(`compute_stablecoin_stress`, feature `stablecoin_stress_z`, data
`data/stablecoin_supply_daily.csv.gz`). On-chain data reused read-only,
unedited: `data/btc_onchain_daily.csv.gz` / `data/eth_onchain_daily.csv.gz`
via `tradebot.data.load_onchain_metrics` / `align_onchain_causal` (both
pre-existing, from B-07/R-44). Not `@register`ed, not auto-discovered,
nothing committed by this branch's own choice. This branch does not touch
`kelly_regime_v4.py`, `kelly_regime_v3.py`, `kelly_regime.py`,
`kelly_regime_v14_macro_lead.py`, `kelly_regime_v15_stablecoin_veto.py`,
`kelly_regime_v16_stablecoin_confirm.py`, `kelly_regime_v10_onchain_confirm.py`,
`_stablecoin_signal.py`, `docs/LEDGER.md`, or the disjoint parallel
CONSERVATIVE branch's files (`kelly_regime_v17_stablecoin_shortwindow.py`
and its report — observed running in parallel this session, neither read
nor coordinated with). All evaluation below is restricted to inner-train
(2017-01-01 → 2020-12-31), inner-validation (2021-01-01 → 2022-12-31), and
a 2019-only BTC-vs-ETH falsification pair (the window where price,
stablecoin, *and* on-chain coverage all overlap for ETH). **The 2023+
holdout was never read** — grep proof at the bottom of this report.

## Idea, mechanism, and pre-registration (written before any code ran)

**Idea, one sentence.** Treat a stablecoin-supply-deceleration reading as
noise unless BTC's own active-address growth is *also* decelerating at
the same time — an AND-gate across two structurally independent signals
(dollar capital flow vs. blockchain usage), rather than any further
filter on the stablecoin signal alone.

**Constraint attacked.** INFO. Fifth attempt: B-07/R-44 (on-chain alone),
R-53 (macro), R-54 (stablecoin hard veto), R-55 (stablecoin confirming
vote), now this (same primary signal as R-54/R-55, corroborated by a
*second*, independent channel for the first time).

**Not a duplicate of.** R-54's hard veto and R-55's confirming vote both
used the stablecoin signal *alone*, varying only the combination rule.
R-55's own CONSERVATIVE sibling tried a duration/persistence filter on
that same single series and failed for a specific, diagnosed reason ("the
'transient' onsets don't reverse within a few days ... duration and
precision are not separable axes here" — the 14-day growth window is
already smoothed past what a duration filter can add). This file asks a
different question: does a *second*, differently-sourced signal agree
that something real is happening? B-07/R-44's on-chain branches used
active-address growth and Hash Ribbons *alone*, fed into v4's vote or a
multiplier; neither combined either with a second, independent series.
Full citations and reasoning are in the module docstring of
`kelly_regime_v17_stablecoin_corroborate.py`.

**Choice of corroborating signal.** Two R-44 candidates were screened
against R-54's own 12 matched stablecoin-onset dates *before* any
strategy code was written: Hash Ribbons miner capitulation corroborates
only **1 of 12** onsets (too coarse — a once-a-cycle, 30–60-day-smoothed
signal) and was ruled out; BTC active-address growth (7-day log growth,
180-day trailing z-score, R-44's own fixed, non-swept defaults,
sign-flipped to match `_stablecoin_signal.py`'s convention) operates on a
comparable cadence and was used as the primary corroborating signal — its
own weakness, uncovered by the pre-sweep mechanism check below, turned
out to be the *opposite* one.

**Mechanism.** `stable_vote` and a new `onchain_vote` (both a latched 0/1
hysteresis vote on their own z-score, identical discipline to every prior
round in this lineage) combine as `corrob_vote = 0` ("stress") *only*
when BOTH read stress; `1` ("calm") otherwise, including whenever either
input is unavailable. `corrob_vote` feeds R-55's own precision-weighted
confirming-vote dilution (`frac=(anchor_sum+stable_weight·corrob_vote)/(3+stable_weight)`)
as the PRIMARY candidate — chosen over R-54's hard override because R-55
already established, and this file's own `ablation()` re-confirms fresh,
that the confirming dilution beats an analogous override on this signal.
A second class, `KellyRegimeV17StablecoinCorroborateOverride`, reruns the
identical AND-gate through R-54's hard-override rule as an ablation arm
only — explicitly **not** proposed as this file's own candidate.

**Sources.** BIS WP 1340 (2025), Ahmed & Aldasoro / BIS WP 1270 (2025), NY
Fed Liberty Street Economics (2025), IMF WP 2025/141 (2025) — all four
cited unchanged from `_stablecoin_signal.py`. Web search run for this
round (queries: "on-chain active addresses stablecoin supply
corroborating signal crypto regime detection multi-signal confirmation
2025"; "ensemble on-chain indicators crypto market stress signal
agreement research 2025"): a 2025 MDPI paper ("Temporal Fusion
Transformer-Based Trading Strategy for Multi-Crypto Assets Using On-Chain
and Technical Indicators," *J. Risk Financial Manag.*) combines active
addresses with other on-chain/technical features in one forecasting
framework — general precedent for multi-signal on-chain ensembles, not a
specific corroboration-gate design this mechanism is drawn from;
aggregated 2025–2026 industry commentary describes stablecoin flows as
typically weighted ~15–25% of a combined signal and states that "when
flows, derivatives, and macro point the same way, the probability of a
move tightens" — the general intuition this file tests directly rather
than assumes. Neither source is stronger than this project's own standing
finding across R-53/R-54/R-55: new information channels are real, but
converting them into a strategy has failed on precision/specificity
grounds four times running.

**Pre-registered falsification tests (fixed before any result was read).**
(1) Confirming-vote candidate does not beat v4 on inner-validation Sharpe
(both markets) by more than the ±0.2 noise floor, or does but not on a
plateau. (2) Fails the pre-2020(-adjacent) BTC-vs-ETH differential test.
(3) Corroboration does not beat the identical signal/architecture
*without* corroboration (R-55's own mechanism, reproduced fresh).
(4) Fails the three-pathway causality probe, or the identity check fails.
(5) Is an exposure-artifact (R² > 0.95 vs. a flat rescale of v4).
**Holdout decision rule, fixed in advance:** read the 2023+ holdout ONLY
IF (1) clears the noise floor on a genuine plateau AND (2) passes AND (3)
shows corroboration earning its keep AND (4)/(5) both pass cleanly. If
ANY fail — or if the pre-sweep mechanism check already shows corroboration
doesn't separate signal from noise — report NEGATIVE, holdout never read.

## Configs evaluated: 33

- **21** confirming-vote-dilution configs (`KellyRegimeV17StablecoinCorroborate`):
  1 identity (`stable_weight=0`) + 4 stablecoin thresh/gap points (R-54's
  own primary/tightest/tight-hys/loose) × 4 weights (0.15/0.33/0.5/1.0) at
  the a-priori "matched" onchain threshold (16 configs), plus the primary
  stablecoin point × 4 weights at a stricter "tight" onchain threshold (4
  configs).
- **4** new configs from `ablation()`: the identical primary-point ×
  4-weight grid with `corroborate=False` (R-55's own mechanism,
  reproduced fresh for an in-session, apples-to-apples comparison — the
  `corroborate=True` half of this grid duplicates 4 of the 21 above and
  is not double-counted).
- **8** new configs from `override()`: R-54's hard-override architecture
  fed the identical AND-gate, at 2 stablecoin points (primary, tightest) ×
  2 onchain thresholds (matched, tight), each with `corroborate=True` and
  `corroborate=False` (4 + 4).

Diagnostic re-reads (v4/`buy_and_hold` benchmarks, the plateau table, the
DD/exposure-artifact follow-up measurements on the override finding,
causality probes) are not separately counted, per the R-42/R-44/R-53/
R-54/R-55 convention.

## Pre-sweep mechanism check (run BEFORE any inner-validation sweep, per this round's instruction)

Reproduced R-54's own numbers fresh, for direct comparability: raw
stress-onset flip counts at primary (1.00/0.75) = **12** and tightest
(0.75/0.00) = **24** (both match R-54's `descriptive()` exactly); 12
matched episodes at primary, **9/12** lead the 3-anchor majority, median
**+16.5 days** (matches R-54's `leadtime()` exactly).

**Corroboration rate does not discriminate lead from lag/noise.** At the
a-priori "matched" onchain threshold (same numeric scale as the
stablecoin primary threshold, 1.00/0.75, chosen before any result was
read): onchain-participation stress occupies **16.4%** of all days.
Checking each of the 12 primary stablecoin onsets for onchain
corroboration within ±30 days: **7/9** genuinely-leading episodes
corroborate, and **3/3** lagging/unmatched episodes *also* corroborate —
if anything, the lag episodes corroborate slightly *more* reliably than
the leads. (The two leads that fail to corroborate, both from April/May
2017, fail because the on-chain z-score's 180-day warmup has not yet
elapsed that early in the series — a coverage artifact, not a genuine
disagreement.) The AND-gate is mostly reading "are we broadly in a
multi-month downtrend," a state nearly all stablecoin onsets — genuine
and false alike — already share, rather than corroborating any one
episode specifically.

**The raw flip count is not reduced — it increases.** AND-gating the
tightest config's vote (24 raw flips uncorroborated) with the matched
onchain threshold produces **28** raw flips, not fewer. Two independently
oscillating binary states intersect at more boundary points than either
alone if their "stress" windows overlap only partially rather than
nesting — a mechanical property of AND-gating two noisy series, not a bug.
Of the 9 genuinely-leading matched episodes, only 6 still produce a
corroborated onset within ±14 days of their original date.

**Tightening the onchain threshold trades discrimination for coverage,
not a clean fix.** At a stricter threshold (2.00/0.50, 3.5% of days in
stress): corroboration rate drops to 4/9 on leads vs. 2/3 on lag/unmatched
— still not cleanly separating — while now also failing to corroborate
several of the largest-magnitude genuine leads (including both 2017
episodes and 2019-11-09).

**Hash Ribbons corroborates only 1/12** onsets — confirms the module
docstring's "too coarse, once-a-cycle" reasoning for not using it as the
primary corroborating signal.

**Conclusion of the mechanism check, stated plainly per this round's
instruction, before any Sharpe number was read:** at the natural,
non-cherry-picked threshold, requiring active-address corroboration does
not separate R-54's genuine early leads from its transient false alarms —
it is either too permissive to filter (matched threshold, corroborates
almost everything) or filters real leads along with noise once tightened
enough to filter anything (tight threshold). This predicts the branch
should fail; the full sweep below was still run to quantify by how much,
per this round's brief.

## Inner-train (sweep, spot, 21 configs)

| candidate | final | Sharpe | max DD |
|---|---|---|---|
| `buy_and_hold` | $29,803 | 1.38 | 84.1% |
| `kelly_regime_v4` | $18,477 | 2.03 | 43.3% |
| identity (`stable_weight=0`) | $18,477 | 2.03 | 43.3% |
| loose w=0.50 onchain=matched (best) | $19,486 | 2.03 | 43.4% |
| tight-hys w=1.00 onchain=matched (worst) | $16,498 | 1.91 | 53.0% |

A handful of low-weight configurations edge v4 slightly in-sample; the
unweighted (w=1.00) negative control is worst everywhere, consistent with
the precision-weighting hypothesis but uninformative about whether
corroboration itself helps (identical pattern to R-55's own uncorroborated
sweep).

## Inner-validation vs v4 (both markets, 21 configs) — falsification test (1)

| candidate | market | final | Sharpe | max DD |
|---|---|---|---|---|
| `kelly_regime_v4` (control) | spot | $998 | 0.14 | 33.2% |
| `kelly_regime_v4` (control) | futures 5x | $1,064 | 0.25 | 32.3% |
| identity (`stable_weight=0`) | spot | $998 | 0.14 | 33.2% |
| best non-identity (several tied configs) | spot | $979 | 0.11 | 35.0% |
| best non-identity (tight-hys w=0.50 matched — exact tie) | futures 5x | $1,064 | 0.25 | 34.4% |
| worst (tightest w=1.00 matched) | spot | $899 | −0.01 | 42.4% |

**No non-identity configuration beats v4 on inner-validation spot
Sharpe** (best 0.11 vs. v4's 0.14) — identical qualitative pattern to
R-55's own uncorroborated confirming vote. One futures cell exactly ties
v4 (0.25); no cell exceeds it. **Falsification test (1) fails
decisively**, same as R-55.

Full parameter-neighbourhood table (spot Sharpe, inner-validation, matched
onchain threshold):

| thresh/gap | w=0.15 | w=0.33 | w=0.50 | w=1.00 |
|---|---|---|---|---|
| primary (1.00/0.75) | 0.11 | 0.09 | 0.06 | 0.02 |
| tightest (0.75/0.00) | 0.11 | 0.11 | 0.05 | −0.01 |
| tight-hys (0.75/0.75) | 0.11 | 0.10 | 0.10 | 0.07 |
| loose (1.25/1.25) | 0.11 | 0.08 | 0.05 | 0.00 |
| identity (`w=0`, = v4) | 0.14 | | | |

There is a mild plateau at low weight (0.10–0.11 across three of four
rows) — but every cell sits below v4's own 0.14, describing where the
candidate loses least, not where it wins. Sharpe strictly decreases with
weight in every row (precision-weighting hypothesis qualitatively
confirmed, again without producing a win). Tightening the onchain
threshold at the primary point moves numbers by ≤0.02 Sharpe in either
direction — the onchain-threshold axis does not rescue anything either.

## Falsification test (3): does corroboration beat the same signal/architecture without it?

`ablation()` compares `corroborate=True` vs. `corroborate=False` (R-55's
own confirming-vote mechanism, reproduced fresh) at the primary
stablecoin point, all 4 weights, both splits, both markets:

| weight | split | market | Δ(corrob−plain) Sharpe |
|---|---|---|---|
| 0.15 | VALID | spot | +0.015 |
| 0.15 | VALID | futures | +0.050 |
| 0.33 | VALID | spot | +0.042 |
| 0.33 | VALID | futures | +0.007 |
| 0.50 | VALID | spot | +0.010 |
| 0.50 | VALID | futures | **−0.162** |
| 1.00 | VALID | spot | +0.091 |
| 1.00 | VALID | futures | +0.084 |
| *(all 8 TRAIN cells)* | TRAIN | both | **−0.02 to −0.15**, every cell negative |

On inner-validation, corroboration is a wash-to-mildly-positive on 7 of 8
cells and actively removes R-55's own best (still noise-floor) result at
w=0.50 futures (0.32 → 0.16). On inner-train, corroboration is
consistently *worse* than the plain confirming vote in all 8 cells.
**Falsification test (3) does not clearly pass**: corroboration neither
decisively helps nor decisively hurts the confirming-vote architecture on
the decisive split, and hurts on train — not the clean "earns its keep"
result the pre-registration required.

## A secondary finding: corroboration substantially rescues R-54's hard override — but it is an exposure-level artifact

Not this file's own candidate (explicitly named an ablation arm in the
pre-registration), but reported in full because it is the most striking
number this round produced. `override()` reruns R-54's exact hard-override
architecture (`frac=0` while vetoed) fed the identical AND-gate:

| config | split | market | corroborated Sharpe | plain-override Sharpe | Δ |
|---|---|---|---|---|---|
| primary/matched | VALID | spot | 0.18 | −0.45 | +0.626 |
| primary/matched | VALID | futures | 0.29 | −0.41 | +0.693 |
| tightest/matched | VALID | spot | 0.18 | −0.53 | +0.708 |
| tightest/matched | VALID | futures | 0.29 | −0.52 | +0.809 |
| primary/tight | VALID | spot | 0.14 | −0.45 | +0.587 |
| tightest/tight | VALID | spot | 0.15 | −0.53 | +0.679 |

The plain-override numbers (−0.41 to −0.53) reproduce R-54's own reported
range almost exactly, confirming the ablation class is faithful. Adding
corroboration turns a decisively negative architecture into one that
**ties or marginally beats v4** on 3 of 4 (thresh/gap × onchain-threshold)
points on spot (v4 = 0.14) and futures (v4 = 0.25), with a small,
consistent drawdown improvement on inner-validation (33.2%→32.5% spot,
32.3%→31.8% futures for the best cell) and a larger one on inner-train
(43.3%→39.4% spot, 35.3%→29.4% futures).

**This is fully explained, and closed, by the exposure-artifact check —
not a real edge.** R² of the corroborated-override's `target` series
against a mean-notional-matched flat rescale of v4's own `target`,
inner-validation:

| market | mean\|v4\| | mean\|cand\| | alpha | R² | corr | verdict |
|---|---|---|---|---|---|---|
| spot | 0.289 | 0.288 | 0.996 | **0.9971** | 0.9986 | **EXPOSURE-LEVEL ARTIFACT** |
| futures 5x | 0.289 | 0.288 | 0.996 | **0.9971** | 0.9986 | **EXPOSURE-LEVEL ARTIFACT** |

Mean absolute exposure is within 0.3% of v4's own, and R² clears the 0.95
threshold by a wide margin. The mechanism is exactly the same trap R-33/
R-34 (LEDGER) already named: because the AND-gate corroborates the large
majority of the override's trigger points (per the mechanism check above),
requiring corroboration mostly just *disables* the override — the
"improvement" over the plain override is not a better regime call, it is
corroboration silently rescaling the strategy back toward v4's own
exposure path. The apparent win is v4 relabeled, not a new mechanism
paying off. This candidate would fail the promotion bar on this check
alone even setting aside that its margin over v4 (≤0.04 Sharpe, ≤0.7pp
drawdown on the decisive validation split) is already inside the ±0.2
noise floor and the "genuine tail improvement" bar. Not eligible for
promotion this round in any case — it was never the file's pre-registered
candidate, and ROUTINE.md's goalpost discipline forbids retroactively
promoting an ablation arm that happened to look interesting.

## Falsification test (2): BTC-vs-ETH, primary confirming-vote candidate

2019-01-01 → 2019-12-31, the window where BTC/ETH price, stablecoin
coverage, *and* on-chain coverage all overlap (ETH on-chain data starts
2019-01-01; the committed ETH price file ends 2019-12-31 — the onchain
leg's z-score itself only has a valid reading from roughly mid-2019
onward inside this window, 53,613 of 104,491 ETH bars NaN on the onchain
column, falling back to no-corroboration/no-veto during that half-year by
the module's designed absence-handling).

| segment | cells failing (of 21 per segment) |
|---|---|
| spot | **13 of 21** |
| futures 5x | 5 of 21 |
| **total** | **18 of 42** |

Primary candidate (`thresh=1.00, gap=0.75, stable_weight=0.33,
onchain=matched`): spot BTC ratio 0.999× / ETH ratio 0.974× — **FAIL**;
futures BTC ratio 1.005× / ETH ratio 1.051× — ok. Same qualitative
signature R-55 found on the plain confirming vote (spot-specific,
systematic ETH underperformance not explained by the BTC side alone).
**Falsification test (2) fails.**

## Falsification test (4): causality

Three independent pathways — price OHLCV (×3/÷3), the stablecoin-supply
input (raw CSV copied to a temp dir, ×50/÷50), and the new on-chain
active-address input (×50/÷50) — tampered separately and together on
strictly pre-2023 bars:

| probe | decisions at/before cut | `target`/`v17_frac`/`v17_used_vote`/`v17_stable_vote`/`v17_onchain_vote` max\|diff\| |
|---|---|---|
| PRICE tamper | PASS | 0.000e+00 |
| STABLECOIN tamper | PASS | 0.000e+00 |
| ONCHAIN tamper (new pathway) | PASS | 0.000e+00 |
| all three at once | PASS | 0.000e+00 |
| identity (`stable_weight=0` ≡ v4) | — | 0.000e+00, PASS |

**Falsification test (4) passes cleanly.**

## Falsification test (5): exposure-artifact, primary confirming-vote candidate

| market | mean\|v4\| | mean\|cand\| | alpha | R² | corr | verdict |
|---|---|---|---|---|---|---|
| spot | 0.289 | 0.333 | 1.149 | **0.9230** | 0.9951 | genuinely different exposure shape |
| futures 5x | 0.289 | 0.333 | 1.149 | **0.9230** | 0.9951 | genuinely different exposure shape |

**Passes** (below 0.95), but with less margin than R-55's own uncorroborated
confirming vote (R²=0.9407) and far less than R-54's hard veto
(R²=0.6091) — corroboration pulls the exposure path even closer to v4's
own, the expected direction given how rarely it actually changes the
vote relative to the plain confirming architecture.

## Verdict: NEGATIVE

Two of five pre-registered falsification gates fail decisively for this
file's own candidate (tests 1 and 2); test 3 does not clearly pass either
(mixed-to-negative). The pre-registered holdout decision rule required
ALL of (1)–(5) to hold before consulting 2023+; it does not. **The 2023+
holdout was never read.**

1. **Falsification test (1) FAILS.** No non-identity configuration beats
   v4 on inner-validation spot Sharpe (best 0.11 vs. 0.14); the mild
   plateau at low weight describes where the candidate loses least, not
   where it wins.
2. **Falsification test (2) FAILS.** 18 of 42 cells are visibly worse on
   ETH than on the BTC control, concentrated on spot — the same
   spot-specific asymmetry R-55 already found on the uncorroborated
   confirming vote.
3. **Falsification test (3) does not clearly pass.** Corroboration is a
   wash-to-mildly-positive on inner-validation (7 of 8 cells) and
   decisively negative on inner-train (8 of 8 cells) versus the identical
   architecture without it — not the clean "earns its keep" result
   required.
4. **Falsification test (4) PASSES cleanly.** Zero lookahead on all three
   independent data pathways (price, stablecoin, on-chain) and combined;
   exact identity recovery of v4.
5. **Falsification test (5) PASSES** (R²=0.9230) but with less margin
   than R-55's own uncorroborated confirming vote.
6. **The pre-sweep mechanism check, run before any of the above, already
   predicted this outcome correctly**: at the natural, non-cherry-picked
   corroboration threshold, active-address participation corroborates
   genuine leading episodes and lagging/noise episodes at nearly the same
   rate (7/9 vs. 3/3) — it is not discriminating what it was built to
   discriminate, because both categories mostly just describe "broadly a
   downtrend," a state the stablecoin signal's false alarms already share
   with its genuine leads far more often than not.
7. **A secondary, more striking-looking number — corroboration turning
   R-54's decisively negative hard override into one that ties/marginally
   beats v4 (Δ Sharpe up to +0.81 over the plain override) — is fully
   explained by, and closed by, an exposure-artifact check (R²=0.9971):
   it is v4 relabeled, not a new edge**, and was never this file's
   pre-registered candidate in any case.

**One-line lesson:** corroboration from a second, structurally
independent signal is a materially different mechanism from a duration
filter on the same series (as B-23 asked it to be) — but it inherits the
same underlying problem this project's whole stablecoin-signal research
line has now hit four times: the corroborating evidence available
(participation growth, hash-rate capitulation) either moves on the wrong
timescale to discriminate genuine multi-week stress from few-day noise,
or — once combined with an architecture sensitive enough to look like it
helps — the "improvement" turns out to be nothing more than the gate
disabling itself back toward the incumbent's own exposure, a trap this
project has now caught with an explicit R² check three separate times
(R-33, R-34, this round).

## Configs evaluated

**33** total, this branch. (21 confirming-vote-dilution + 4 new
no-corroboration ablation configs + 8 new hard-override-architecture
configs; diagnostic re-reads including the exposure-artifact follow-up on
the override finding are not separately counted, per convention.) The
parallel CONSERVATIVE branch reports its own count separately — this
round's project-level trials count is the sum of both, to be totaled by
the operator once both reports are in.

## Holdout

**+0.** Never read. Grep proof, every date literal ≥2023 in this branch's
new file:

```
$ grep -n "202[3-9]" experiments/kelly_regime_v17_stablecoin_corroborate.py
145:  markets" (2025); Ahmed & Aldasoro, "Stablecoins and safe asset prices"
146:  (Cleveland Fed / BIS WP 1270, August 2025); NY Fed Liberty Street
147:  Economics, "Stablecoins and Crypto Shocks: An Update" (April 2025); IMF
148:  WP 2025/141, "Decrypting Crypto: How to Estimate International
149:  Stablecoin Flows" (July 2025) -- all four cited unchanged from
154:  multi-signal confirmation 2025"; "ensemble on-chain indicators crypto
155:  market stress signal agreement research 2025"), verified at
157:  - A 2025 MDPI study (*Journal of Risk and Financial Management*,
165:  - Industry-practice commentary (2025-2026, aggregated by web search, not
223:read:** the 2023+ holdout is read ONLY IF (1) clears the noise floor AND
298:OOS_START = "2023-01-01"                 # never read in this file
1057:    cut must be unchanged. Restricted to strictly pre-2023 bars."""
1060:    pre2023 = DF[DF.index < OOS_START]
1061:    df = pre2023.iloc[-300_000:].copy()
```

Every hit above 298 and below is either a literature-citation year in
prose (145–165), the pre-registration prose itself (223, no data read),
or the `OOS_START` constant and its two call sites — both of which are
*exclusive upper bounds* restricting the causality probe to strictly
pre-2023 bars, never a data read past the boundary. The `eth()`
falsification function is independently restricted to
`"2019-01-01":"2019-12-31"` by construction (a literal string slice, not
matched by this grep pattern since neither bound is ≥2023), nowhere near
the boundary. Independently verifiable by re-running the same grep
command above.

## Test suite

`pytest` (from `.venv`): **457 passed**, unchanged from the session's
starting count and from every prior round in this lineage — this branch
touched no file under `src/` or `tests/`; no new tests added (not
required for a NEGATIVE, unregistered experiment per ROUTINE.md).

## Next step

B-23's second named fix is now closed as NEGATIVE, alongside its sibling
(a shorter growth window, tested by this round's parallel CONSERVATIVE
branch). Both of B-23's own candidate directions have now been tried.
Given this project's five-attempt record on the INFO constraint
(B-07/R-44, R-53, R-54, R-55, this round) sharing the same qualitative
failure shape each time — a genuinely new, real information channel
found, but no combination mechanism yet converts it into a working
strategy, and the one time this round found something that *looked* like
an edge, it dissolved into the R-33/R-34 exposure-artifact trap on
inspection — the honest recommendation, consistent with R-55's own, is
that a sixth INFO-axis variant on this same stablecoin-signal research
line is a weaker bet than `scripts/paper_trade.py` (B-06) or a genuinely
different research direction entirely. Not recommended as this project's
next move.
