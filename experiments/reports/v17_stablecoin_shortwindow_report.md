# kelly_regime_v17_stablecoin_shortwindow — R-56 CONSERVATIVE branch (08-20)

Unregistered experiment, closing backlog item **B-23** (filed by R-55,
LOW priority): "a shorter growth window matched to genuine-stress
duration ... applied to the feature itself, rather than a persistence
filter bolted onto the existing 14-day feature." Code:
`experiments/kelly_regime_v17_stablecoin_shortwindow.py`. Reuses
`tradebot.data.load_stablecoin_supply`/`align_stablecoin_causal`
unchanged; does **not** edit `experiments/_stablecoin_signal.py`,
`kelly_regime_v15_stablecoin_veto.py`, `kelly_regime_v16_stablecoin_persist.py`,
`kelly_regime_v16_stablecoin_confirm.py`, `kelly_regime_v4.py`/`_v3.py`/
`kelly_regime.py`, `docs/LEDGER.md`, or this round's disjoint parallel
NOVEL branch's files (not read, not coordinated with). All evaluation
below is restricted to inner-train (2017-01-01 → 2020-12-31),
inner-validation (2021-01-01 → 2022-12-31), a pre-2020 BTC control, and
the standard ETH falsification pair. **The 2023+ holdout was never
read** — grep proof at the bottom of this report.

## Idea, mechanism, and why it is a different mechanism from R-55's two fixes

**Idea, one sentence.** R-54's `stablecoin_stress_z` fixes its growth
window at 14 calendar days a-priori; this branch tests whether a shorter
window (`{2, 3, 5, 7, 10}` days), closer to the duration recent
literature attributes to acute stablecoin-redemption stress, tracks
`kelly_regime_v4`'s own 3-anchor-majority price-gate flip with less lag
than the 14-day feature, and whether that in turn fixes the
false-stress-onset problem R-54 diagnosed and R-55's persistence filter
failed to fix.

**Constraint attacked.** INFO — the fourth consecutive round on this
axis (R-53, R-54, R-55, this round) and the fourth attempt on this
specific signal. Per B-23's own LOW-priority filing, this branch went in
expecting a plausible negative.

**Not a duplicate of, cited precisely:**
- `kelly_regime_v15_stablecoin_veto.py` (R-54): fixed N=14 a-priori,
  never swept. This file's entire contribution is making N a genuinely
  swept parameter — everything else (hard-veto architecture, 365-day
  z-score window, hysteresis grid) is reused unchanged.
- `kelly_regime_v16_stablecoin_persist.py` (R-55 CONSERVATIVE, closed
  B-22 fix 1/2): bolted a minimum-duration *confirmation* filter on top
  of the unmodified 14-day feature. This file changes the feature's own
  time-scale instead of filtering an already-computed vote.
- `kelly_regime_v16_stablecoin_confirm.py` (R-55 NOVEL, closed B-22 fix
  2/2): swapped the *combination rule* (hard override → confirming vote)
  on the unmodified 14-day feature. This file keeps R-54's hard-override
  combination rule byte-for-byte and varies only the feature.
- **B-23**: this file is exactly its first named candidate mechanism.

## Sources

Unchanged from R-54/`_stablecoin_signal.py`: BIS WP 1340 (2025); Ahmed &
Aldasoro, Cleveland Fed conference paper / BIS WP 1270 (Aug 2025); NY Fed
Liberty Street Economics (Apr 2025); IMF WP 2025/141 — base mechanism,
not re-derived.

**New for this round**, found via web search specifically on stablecoin
redemption *timescales* (the question B-23 asks): ESRB, "Crypto-assets
and decentralised finance" (Oct 2025); industry technical write-ups on
stablecoin reserve/liquidity stress (e.g. crypto-economy.com,
"Stablecoins Under Stress: A Technical Dissection of Reserve Architecture
and Liquidity Risk"; 2026 trade-press pieces on stablecoin run risk).
These converge on: acute stablecoin redemption spikes typically run
**48–72 hours**, after which issuers liquidate short-term reserves and
spreads normalize within **about a week** — this is the literature basis
for this round's window grid (2–10 days), chosen before any code ran.
The **named risk**, stated equally plainly before running anything: that
short "acute redemption" timescale and the multi-week *capital-flight*
dynamic R-54's signal is actually being asked to lead (its own confirmed
+16.5-day lead at N=14) are not necessarily the same timescale — a window
tuned to the former could be too short for the latter and mostly track
day-to-day supply-reporting noise instead. **This is exactly what the
result below found.**

## Feature formula

`growth_Nd = log(supply_t) − log(supply_{t−N})`, N ∈ `{2, 3, 5, 7, 10}`
(plus N=14 recomputed in-file as a reference reproduction of R-54's own
number, not new evidence). `stablecoin_stress_z = −1 · zscore(growth_Nd,
trailing 365d, min_periods=60)` — z-score window and `min_periods` held
**identical** across every N (min_periods governs z-score stability, not
the growth window's own length, so fixing it isolates N as the only
varied axis). Hard-veto combination architecture reused byte-for-byte
from `KellyRegimeV15StablecoinVeto`.

## Pre-registered decision procedure (frozen before any code ran)

**Step A — the gate, run first, before any Sharpe number.** A candidate
window passes only if its lead-time result (vs. v4's own 3-anchor
majority, same matching methodology as R-54's `leadtime()`) is **at least
as good** as the N=14 reference on **both** axes: lead fraction ≥ 0.75
(9/12) and median lead ≥ +16.5 days. If no window passes, the
pre-registered decision is **NOT TO PROMOTE**, regardless of any
subsequently computed Sharpe numbers — those are still computed and
reported (the project's convention is to finish the pipeline for a
complete report), but are explicitly diagnostic only once Step A fails.
**Step B — only if Step A passes for ≥1 window:** standard inner-train/
inner-validation sweep, ordinary promotion bar (beats `buy_and_hold`
OOS after costs, Δ Sharpe beyond ±0.2, survives falsification, plateau
not peak). **Falsification (run regardless):** (a) pre-2020 BTC control,
(b) ETH (`data/ethusd_bitfinex_5m.csv.gz`). **Exposure-artifact R²** and
the **two-pathway causality probe** run unconditionally too.

## Step A result: THE GATE — 0/5 windows pass, and the failure is a clean, monotonic flip from lead to lag

| window (days) | onset events | matched | leads | median lead (days) | gate |
|---|---|---|---|---|---|
| 2 | 13 | 13 | 4/13 (31%) | **−15.0** | FAIL |
| 3 | 12 | 12 | 4/12 (33%) | **−12.0** | FAIL |
| 5 | 17 | 17 | 7/17 (41%) | **−8.0** | FAIL |
| 7 | 15 | 15 | 6/15 (40%) | **−7.0** | FAIL |
| 10 | 12 | 12 | 6/12 (50%) | **0.0** | FAIL |
| 14 (reference) | 12 | 12 | 9/12 (75%) | **+16.5** | — |

**This is not a noisy or borderline result — it is monotonic and clean.**
As the growth window shrinks from 14 to 2 days, the lead fraction falls
75%→50%→41%→40%→33%→31% and the median offset moves smoothly from a
+16.5-day **lead** through 0 at N=10 to a −15.0-day **lag** at N=2. Every
one of the 5 candidate windows fails the gate on both axes; none is a
close call. The onset-event *count* is not the driver — it ranges 12–17
with no clean trend — the driver is that shorter windows' onsets
increasingly land *after* the price-anchor majority has already flipped,
not before it.

**Mechanism reading.** This confirms the round's own pre-registered named
risk over its own hypothesis: the 48–72-hour *acute redemption* timescale
recent literature reports and the multi-week *capital-flight* dynamic
R-54's 14-day feature actually leads on are not the same thing. A growth
window shortened toward the former timescale does not track the latter
faster — it substitutes a different, less useful signal (near-term
supply/reporting noise, or perhaps redemption spikes that are themselves
reactive to price stress rather than predictive of it) for the one that
was shown to lead. Per this round's own pre-registered rule, **the
decision is NOT TO PROMOTE as of this point** — everything below is
computed for a complete, honest report, not as grounds to revisit that
decision.

## Configurations evaluated

**15** (`growth_window_days ∈ {2, 3, 5, 7, 10} × gap ∈ {0.0, 0.75, 1.25}`,
`thresh_hi` held at R-54's primary `1.0`). The original pre-registration
specified the full `thresh_hi ∈ {0.75, 1.0, 1.25} × gap` 3×3 grid at
every window (45 configs); after Step A returned its clean, decisive
verdict, and after the 45-config run had not finished within a
reasonable window, the grid was **scoped down mid-session** to the 15
above — thresh_hi fixed at the primary point, gap still swept at every
window, so a real parameter-neighbourhood axis is preserved at every
window rather than collapsing to one point each. This changes no
conclusion (Step A already decided the branch on mechanism grounds, not
a borderline Sharpe call) and is recorded here plainly per ROUTINE.md's
own instruction that a changed procedure be stated explicitly. `leadtime_
by_window()` (Step A, 6 windows) counts 0 configurations, matching R-54/
R-55's own convention for descriptive/diagnostic reads; the exposure-
artifact, causality and pre-2020-control re-reads below reuse config keys
already inside the 15-config grid and add no new ones.

## Inner-train (sweep, spot, 15 configs)

| candidate | final | Sharpe | max DD |
|---|---|---|---|
| `buy_and_hold` | $29,803 | 1.38 | 84.1% |
| `kelly_regime_v4` | $18,477 | 2.03 | 43.3% |
| w=10d, gap=0.75 (best in-sample) | $21,900 | 2.15 | 39.9% |
| w=5d, gap=1.25 (worst) | $14,530 | 1.98 | 40.3% |

Every one of the 15 configs stays close to v4's in-sample Sharpe (1.96–
2.15) — no config collapses on inner-train the way R-54's tightest
thresholds did; the damage shows up entirely on inner-validation.

## Inner-validation vs v4 (both markets, all 15 configs)

| candidate | market | final | Sharpe | max DD |
|---|---|---|---|---|
| `kelly_regime_v4` (control) | spot | $998 | **0.14** | 33.2% |
| `kelly_regime_v4` (control) | futures 5x | $1,064 | **0.25** | 32.3% |
| best by min(train,valid) spot Sharpe: w=3d, gap=0.00 | spot | $971 | 0.09 | 32.3% |
| w=3d, gap=0.00 | futures 5x | $1,038 | 0.21 | 31.4% |
| w=5d, gap=0.75 | spot | $784 | −0.33 | 45.6% |
| w=7d, gap=0.75 | spot | $716 | **−0.54** (worst) | 48.7% |

**No configuration beats v4 on inner-validation Sharpe, in either
market.** The best candidate by the pre-registered selection rule
(min(train, valid) spot Sharpe) is w=3d/gap=0.00, and it still trails v4
by −0.049 spot / −0.044 futures — inside the noise floor in magnitude but
on the wrong side of it. Windows 5, 7 and 10 (at the primary gap=0.75)
are decisively worse (Sharpe −0.21 to −0.54), reproducing R-54's original
"tight threshold fires on noise" failure mode at every window that
actually differs meaningfully from a no-op.

**Parameter-neighbourhood plateau check** (spot, inner-validation Sharpe,
gap swept at fixed thresh_hi=1.00):

| window | gap=0.00 | gap=0.75 | gap=1.25 |
|---|---|---|---|
| 2d | 0.07 | −0.01 | −0.03 |
| 3d | 0.09 | −0.01 | −0.16 |
| 5d | −0.23 | −0.33 | −0.38 |
| 7d | −0.21 | −0.54 | −0.28 |
| 10d | −0.42 | −0.21 | −0.26 |

**Cross-window plateau check** (primary gap=0.75, window varied):

| window | 2d | 3d | 5d | 7d | 10d |
|---|---|---|---|---|---|
| spot Sharpe | −0.01 | −0.01 | −0.33 | −0.54 | −0.21 |

**Not a plateau on either axis.** Every cell in both tables sits below
v4's 0.14 control; the surface is uneven and, for windows 5/7/10,
sharply negative — the promotion bar's plateau requirement fails
independently of the Sharpe result.

## Exposure-artifact check

R² of the candidate's `target` series against a mean-notional-matched
flat rescale of v4's own `target`, inner-validation, both markets
(identical for spot/futures since the rescale is notional-matched):

| candidate | R² | verdict |
|---|---|---|
| w=3d, gap=0.00 (the selected "best") | **0.9813** | **EXPOSURE-LEVEL ARTIFACT** |
| w=3d, gap=0.75 (primary) | **0.9643** | **EXPOSURE-LEVEL ARTIFACT** |
| w=10d, gap=0.75 (closest to R-54's N=14) | 0.5707 | genuinely different exposure shape |

**A second, independent kill, distinct from Step A.** The one config that
comes closest to matching v4 on Sharpe (w=3d) does so because it is
barely distinguishable from v4's own exposure in the first place — a
short, noisy growth window flips the latched vote so often that it
tracks the 3-anchor vote almost in lock-step (raw corr 0.99), so the
near-tied Sharpe is largely a relabeled rescale of v4, not a genuinely
different mechanism. Only the longer window (10d, closest to R-54's
original 14) produces a genuinely different exposure shape (R²=0.57,
comparable to R-54's own reported 0.61 at N=14) — and that is exactly the
window that still loses on Sharpe and fails the falsification test below.
**Shortening the window trades away the one property (a genuinely
different exposure shape) that made R-54's original candidate worth
testing at all, without buying back Sharpe.**

## Causality probe (unregistered strategy, no CI coverage)

Two independent pathways tampered separately and together, on strictly
pre-2023 bars (candidate: w=3d, gap=0.00), plus identity check:

| probe | decisions at/before cut | `target`/`v17_frac`/`v17_stable_vote`/`v17_anchor_sum` max\|diff\| before cut |
|---|---|---|
| PRICE tamper | PASS | 0.000e+00 (all 4 columns) |
| STABLECOIN tamper (new pathway) | PASS | 0.000e+00 (all 4 columns) |
| both at once | PASS | 0.000e+00 (all 4 columns) |
| identity check (`enabled=False` ≡ v4) | — | max\|diff\|=0.000e+00, PASS |

No lookahead on either pathway; the veto mechanism recovers v4 exactly
when disabled.

## Falsification test (a): pre-2020 BTC control

| candidate | market | final | Sharpe |
|---|---|---|---|
| `kelly_regime_v4` (control) | spot | $6,033 | 1.75 |
| `kelly_regime_v4` (control) | futures 5x | $7,938 | 1.94 |
| w=2d primary | spot | $5,466 | 1.68 |
| w=3d primary | spot | $6,286 | 1.79 |
| w=5d primary | spot | $6,369 | 1.80 |
| w=7d primary | spot | $6,107 | 1.77 |
| w=10d primary | spot | $7,151 | 1.92 |

**Passes.** No window loses badly to v4 on data before the stress
episodes the signal is tuned around — every Sharpe sits within ±0.17 of
v4's 1.75, well inside noise. This leg of the falsification test does not
by itself distinguish the windows; the ETH leg below does.

## Falsification test (b): ETH (pre-registered rule identical to R-54's)

ETH-USD Bitfinex spot (2016-03 → 2019-12-31) against USDT-supply coverage
(2017-01-01 →), same rule as R-54's `eth()`: candidate must not be
visibly worse on ETH than on the identical-pipeline BTC control (here,
BTC's full 2017–2022 span, matching R-54's convention). Run on the
per-window primary configs (gap=0.75), both markets, 10 cells:

| config | market | BTC ratio (cand/v4) | ETH ratio (cand/v4) | flag |
|---|---|---|---|---|
| w=2d | spot | 0.821× | 0.991× | ok |
| w=2d | futures 5x | 0.803× | 1.037× | ok |
| w=3d | spot | 0.937× | 1.017× | ok |
| w=3d | futures 5x | 0.960× | 1.036× | ok |
| w=5d | spot | 0.830× | 1.040× | ok |
| w=5d | futures 5x | 0.856× | 1.022× | ok |
| w=7d | spot | 0.749× | 0.992× | ok |
| w=7d | futures 5x | 0.692× | 1.004× | ok |
| **w=10d** | **spot** | **1.041×** | **0.992×** | **FAIL** |
| w=10d | futures 5x | 0.998× | 1.008× | ok |

**Overall verdict: FAIL** (1/10 cells fails outright by the differential
rule). The one config that actually *beats* v4 on the BTC side (w=10d
spot, 1.041×) is exactly the one that shows ETH-specific weakness
(0.992× vs. its own 1.041× BTC ratio) — the same asset-specific
degradation signature R-53's macro candidate showed. The other 9 cells
register "ok" only in the weak sense R-54's own report already flagged:
they add no *further* ETH-specific degradation on top of an
already-large absolute BTC-side shortfall (ratios as low as 0.69×), not
because they are competitive with v4 on either asset.

## Pre-registered decision rule, applied

Step A's gate (0/5 windows preserve or improve the reference lead-time
result) already fixed the outcome before any Sharpe number was read.
Every subsequent check corroborates rather than overturns it: no window
clears v4 on inner-validation Sharpe (best −0.049 spot vs. the required
+0.2 improvement), no plateau exists on either the gap or window axis,
the one near-tied candidate is an exposure-level artifact (R²=0.98), and
ETH falsification fails outright on the one config that does beat v4 on
its own BTC control. **Every leg of the promotion bar fails
independently.**

## Holdout

**Never consulted.** Per the pre-registered rule ("only consult the
2023+ holdout if inner-validation + falsification already look like a
genuine win — do not look just to see"), and since Step A alone already
determined the outcome before any Sharpe or falsification number was
read, the 2023+ holdout was never read. Grep proof, every date literal
≥2023 in this branch's one new file:

```
$ grep -n "202[3-9]" experiments/kelly_regime_v17_stablecoin_shortwindow.py
72:- BIS WP 1340 (2025), ...                              <- literature citation year, prose only
74:  conference paper / BIS WP 1270, Aug 2025); NY Fed ... <- literature citation year, prose only
75:  Economics, "Stablecoins and Crypto Shocks: An Update" (Apr 2025); IMF WP
76:  2025/141 -- ...                                       <- literature citation year, prose only
82:  and decentralised finance" (Oct 2025) and ...          <- literature citation year, prose only
85:  Architecture and Liquidity Risk," and 2026 trade-press pieces on
                                                            <- literature citation year, prose only
223:OOS_START = "2023-01-01"                 # never read in this file
679:    identity-recovery check. Restricted to strictly pre-2023 bars."""
682:    pre2023 = DF[DF.index < OOS_START]         <- exclusive upper bound, causality probe only
683:    df = pre2023.iloc[-300_000:].copy()
```

`OOS_START` is used exclusively as an exclusive upper bound restricting
the causality probe to strictly pre-2023 bars — never a data read past
the boundary. No other file in this branch (there is only the one) was
created or edited.

## Test suite

`pytest`: **457 passed**, unchanged from the session's starting count and
from R-54/R-55's own counts.

## Verdict: NEGATIVE

**B-23's first named mechanism (shorter growth window) does not rescue
the stablecoin signal, and fails faster and more cleanly than either of
R-55's two fixes did.** A shorter growth window does not track genuine
stress episodes with less lag than the 14-day feature — it does the
opposite, monotonically, flipping from a confirmed +16.5-day lead at
N=14 to an outright −15.0-day lag at N=2, with every intermediate window
tested landing strictly between those two points. This is diagnosed as a
timescale mismatch, not a noise problem: recent literature's ~48–72-hour
acute-redemption-stress window and the multi-week capital-flight dynamic
R-54's signal was shown to actually lead are evidently not the same
thing, and shortening the feature toward the former timescale does not
buy earlier detection of the latter — it substitutes a different, less
useful signal for the one already shown to work. A second, independent
finding reinforces the same conclusion from a different angle: the one
window that comes close to matching v4's Sharpe (3 days) does so only
because its vote has degenerated into a near-relabeling of v4's own
anchor vote (exposure-artifact R²=0.98), not because it represents a
genuinely different, competing mechanism; the one window that keeps a
genuinely different exposure shape (10 days, closest to the original 14)
still loses on Sharpe and is the one config that fails ETH falsification
outright.

**One-line lesson:** shortening this signal's growth window trades away
the property that made it worth testing (a confirmed lead over v4's own
gate) without buying back either better timing or a Sharpe improvement —
the fix B-23 named as "matched to genuine-stress duration" assumed
redemption-episode duration and capital-flight-lead duration are the same
clock; they measurably are not, and the a-priori 14-day window R-54 chose
by a different, unrelated reasoning (near-instant on-chain mint/burn)
turns out to be closer to the useful timescale than any of the
literature-motivated shorter candidates tested here.

## Next step

B-23's second named candidate — corroboration from a second, independent
signal rather than any further modification of this one — is the
remaining open half of B-23 and is this round's disjoint parallel NOVEL
branch's job, not re-attempted here. With this round, the stablecoin
signal's research line has now tried four structurally different
combination/feature-timescale variants (R-54's fixed-window hard veto,
R-55's persistence filter, R-55's confirming-vote architecture, this
round's swept growth window) and all four have failed on independent,
well-diagnosed grounds. **Recommend closing B-23's window-mechanism half
without further pursuit**; `scripts/paper_trade.py` (B-06, forward paper
trading) remains this project's standing zero-cost recommendation.

## Configurations evaluated

**15** (this branch's total; the parallel NOVEL branch reports its own
count separately — this round's project-level trials count is the sum of
both, per ROUTINE.md's parallelism rules, to be totaled by the operator
when both reports are in).
