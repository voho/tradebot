# kelly_regime_v15_macro_veto — R-54 CONSERVATIVE branch (08-20)

Unregistered experiment. Code: `experiments/kelly_regime_v15_macro_veto.py`.
Not `@register`ed, not auto-discovered, nothing committed by this branch's
own choice — a human operator merges after this and the parallel novel
branch both report. This file imports `experiments/_macro_signal.py`
unchanged and reimplements (does not import) the veto mechanism standalone
so it has no runtime dependency on either R-53 sibling file; it does not
touch `kelly_regime_v4.py`, `kelly_regime_v3.py`, `kelly_regime.py`,
`src/tradebot/data.py`, `docs/LEDGER.md`, `kelly_regime_v14_macro_lead.py`,
or `kelly_regime_v14_macro_brake.py`. All evaluation is restricted to
inner-train (2017-01-01 → 2020-12-31), inner-validation (2021-01-01 →
2022-12-31), and the standard pre-2020 BTC-control/ETH falsification pair.
**The 2023+ holdout was never read.**

Grep for date literals in the file (pasted verbatim, proof no bar
≥2023-01-01 is ever read):

```
$ grep -n "20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]" experiments/kelly_regime_v15_macro_veto.py
133:Inner-train (2017-01-01 -> 2020-12-31) for iteration, inner-validation
134:(2021-01-01 -> 2022-12-31) for selection, per ROUTINE.md step 3. No bar
135:dated 2023-01-01 or later is read anywhere in this file -- grepped and
179:TRAIN = ("2017-01-01", "2020-12-31")     # inner-train
180:VALID = ("2021-01-01", "2022-12-31")     # inner-validation
181:OOS_START = "2023-01-01"                 # never read in this file
805:    step-2 menu item). ETH-USD Bitfinex spot (2016-03 -> 2019-12-31)
832:    btc_control = DF[DF.index < "2020-01-01"]
```

`OOS_START` is used exactly once, at line 797 of the file (`pre2023 = DF[DF.index < OOS_START]`), exclusively as an upper bound restricting the causality probe to pre-2023 bars — never as a data read for any backtest. `TRAIN`/`VALID` bound every `sweep()`/`select()`/`leadtime()`/`descriptive()`/`artifact()` call. `eth()`'s BTC control is explicitly sliced to `< 2020-01-01`, matching R-53's convention. No other date literal appears anywhere in the file.

## Idea, mechanism, and pre-registration (written before running anything)

**Idea, one sentence.** While VIX/DXY-derived macro stress (`stress_z`,
imported unchanged from the shared `experiments/_macro_signal.py`) is
latched "elevated" (with hysteresis), force `frac = 0` — a full
stand-down, overriding `kelly_regime_v4`'s own three-anchor vote
completely; otherwise `frac` is v4's own unmodified 3-anchor average — no
weighting, no averaging, no continuous multiplicative haircut.

**Constraint(s) attacked.** **INFO** — VIX/DXY are the second genuinely
new, price-independent data channel this project has used (after
on-chain metrics, B-07/R-44). **SIZE** — a hard override forcing
`frac=0` is a decision about *how much* to hold, the project's one
repeatedly-working axis, layered on top of (not replacing) v4's existing
SIZE machinery, which is untouched.

**Not a duplicate of, cited precisely:**
- **L-04/L-01** (`kelly_regime_v4`): the candidate's `frac` degenerates to
  v4's own unmodified 3-anchor average whenever macro data is absent or
  `stress_z` never crosses `thresh_hi` — verified as an explicit identity
  check in `causality()` (forcing `stress_z` all-NaN reproduces v4's
  `target` exactly, max|diff|=0.0).
- **R-53's two rows**: the conservative row
  (`kelly_regime_v14_macro_brake.py`) is a *continuous, never-increase
  multiplicative haircut* fed by the identical signal, and it collapsed
  into the R-34 exposure-artifact failure mode (R²=0.974–0.999) — this
  file's mechanism is architecturally different (binary override on the
  vote, before the SIZE formula runs), which is exactly why the mandatory
  exposure-artifact check below is run rather than assumed safe. The
  novel row (`kelly_regime_v14_macro_lead.py`) is a *precision-weighted
  4th vote averaged with the three anchors*, and it lost to its own hard-
  override ablation arm in 10/12 matched cells by 0.25–0.48 Sharpe — this
  file promotes that ablation arm to be the pre-registered primary
  subject of its own, full falsification battery, rather than a side
  comparison arm inside a round pre-registered for a different mechanism.
- **R-44's on-chain rows** (B-07, hash-ribbon vote): same general
  architecture family (a latched auxiliary vote combined with v4's own
  anchor votes), but a different, BTC-network-specific signal, and the
  opposite sign discipline — R-44's vote only ever pushes exposure UP;
  this veto only ever pushes exposure DOWN to exactly zero.

**Pre-registered falsification test, primary (resolves B-21's own named,
unresolved tension).** R-53's own lead-time finding for the AVERAGED
version of this signal found the macro vote lags the 3-anchor majority on
net (4/12 matched episodes lead, median offset −5.5 days). This file
re-runs the identical `leadtime()` methodology (flip TIMESTAMPS, not
Sharpe) against its own veto latch. **Stated failure outcome, named in
advance:** if the veto's own bear-onset flips do NOT lead the 3-anchor
majority on net (median lead_days ≤ 0), the "faster-flip" rationale does
not hold for the hard-override form either, and the "blunter combination
rule wins despite lagging" hypothesis from B-21's backlog note is
REJECTED unless the candidate clears every other mandatory gate (ETH,
BTC control, causality, R²<0.95, plateau) decisively AND any inner-
validation edge is attributable to something other than faster reaction.

**Secondary, standard-menu test.** Does it survive on ETH (pre-2020,
ROUTINE.md step 2)? Failure outcome: underperforms v4 on ETH spot while
matching/beating v4 on the identical-pipeline BTC control → FAIL.

## Grid

`thresh_hi = 1.0` fixed, never swept. `gap ∈ {0.0, 0.5, 0.75, 1.0, 1.25}`
(the task's pre-declared grid), `gap=0.0` the explicit no-hysteresis
negative control. Primary candidate pre-registered at `gap=0.75` (a real
hysteresis band, matching R-53's own default) — **not** `gap=0.0`, even
though R-53's own ledger row (B-21) happens to cite the `gap=0.0` cell's
numbers as the "unvetted lead" headline. **5 distinct configurations
evaluated** (the `gap` grid only; `thresh_hi` is fixed and not counted).
Diagnostic re-reads (v4/`buy_and_hold` benchmarks, the plateau table,
causality probes, the full-grid artifact sweep, ETH/BTC-control runs) are
not separately counted, per this project's established convention.

## Primary falsification test: lead-time (against the 3-anchor majority)

Corrected `.shift(fill_value=False)` used throughout (the
`~is_target.shift().fillna(False)` bitwise-not-on-object-dtype bug R-53
found and documented is avoided by construction, not merely noted).

| comparison | matched episodes | veto leads | median lead (days) |
|---|---|---|---|
| vs. fastest single (20d) anchor | 12 | 7/12 (58%) | +4.5 |
| vs. 3-anchor MAJORITY (the actual gate-flip proxy) | 12 | 4/12 (33%) | **−5.5** |

**FAIL, and it replicates R-53's averaged-version result almost exactly**
(same episode count, same 4/12 lead fraction, same −5.5-day median offset
against the majority-anchor proxy — because it is fundamentally the same
underlying `stress_z`/hysteresis construction, only the combination rule
differs). The hard veto does not flip earlier than v4's own gate on this
project's available stress episodes. Per the pre-registered rule, the
"blunter combination rule buys back the lead-time loss" hypothesis is
**REJECTED** unless every remaining gate passes decisively and any surviving
edge is explained by something other than faster reaction. It is not, and
they do not — see below.

## Inner-train (sweep, spot, 5 configs)

| candidate | final | Sharpe | max DD |
|---|---|---|---|
| `buy_and_hold` | $29,803 | 1.38 | 84.1% |
| `kelly_regime_v4` | $18,477 | 2.03 | 43.3% |
| `gap=0.00` (naive-nohys) | $24,258 | 2.24 | 33.3% |
| `gap=0.50` | $23,637 | 2.23 | 34.1% |
| `gap=0.75` (primary) | $24,681 | **2.26** | 31.8% |
| `gap=1.00` | $22,932 | 2.25 | 32.8% |
| `gap=1.25` | $19,339 | 2.15 | 44.2% |

Every configuration beats v4 on inner-train.

## Inner-validation vs v4 (both markets, all 5 configs)

| candidate | market | final | Sharpe | max DD |
|---|---|---|---|---|
| `kelly_regime_v4` (control) | spot | $998 | 0.14 | 33.2% |
| `kelly_regime_v4` (control) | futures 5x | $1,064 | 0.25 | 32.3% |
| `gap=0.00` (naive-nohys) | spot | $1,122 | **0.34** | 26.4% |
| `gap=0.00` (naive-nohys) | futures 5x | $1,147 | **0.39** | 27.4% |
| `gap=0.50` | spot | $1,080 | 0.28 | 27.7% |
| `gap=0.50` | futures 5x | $1,133 | 0.36 | 27.9% |
| `gap=0.75` (primary) | spot | $1,105 | 0.32 | 26.0% |
| `gap=0.75` (primary) | futures 5x | $1,163 | 0.41 | 26.0% |
| `gap=1.00` | spot | $975 | 0.09 | 34.7% |
| `gap=1.00` | futures 5x | $1,001 | 0.14 | 36.3% |
| `gap=1.25` | spot | $941 | 0.02 | 37.0% |
| `gap=1.25` | futures 5x | $999 | 0.14 | 36.4% |

Primary config (`gap=0.75`) vs v4, inner-validation: spot ΔSharpe
**+0.177** (just under the ±0.2 noise floor), ΔDD **−7.2pp**; futures
ΔSharpe **+0.159** (also under the floor), ΔDD **−6.3pp**. Neither
market's Sharpe edge clears the noise floor on its own; the drawdown/tail
improvement is real (~6–7pp on both markets) and would, in isolation,
satisfy the promotion bar's alternative clause.

## Parameter-neighbourhood plateau check

| gap | 0.00 | 0.50 | 0.75 | 1.00 | 1.25 |
|---|---|---|---|---|---|
| spot Sharpe (valid) | 0.34 | 0.28 | 0.32 | 0.09 | 0.02 |

Spread across the grid: **0.32** — **NOT a plateau** (≥ the 0.2 noise
floor). This is a genuine peak, not a flat region: performance collapses
by more than the noise floor between `gap=0.75` and `gap=1.00`, and the
single best-scoring cell in the whole grid is `gap=0.00` — the explicit
no-hysteresis negative control, not the pre-registered primary. This
mirrors, almost exactly, the fact already on the record in B-21's own
ledger citation: the "unvetted lead" number the backlog item quotes is
the negative control's own result, not a genuine hysteresis-bearing
candidate's.

## Exposure-artifact check

R² of the candidate's `target` series against a mean-notional-matched
flat rescale of v4's own `target`, inner-validation, both markets:

| config | market | R² | verdict |
|---|---|---|---|
| `gap=0.00` (naive-nohys, **the grid's top scorer**) | both | **0.9544** | **EXPOSURE-LEVEL ARTIFACT** |
| `gap=0.50` | both | 0.8527 | genuinely different exposure shape |
| `gap=0.75` (primary) | both | **0.8410** | genuinely different exposure shape |
| `gap=1.00` | both | 0.7627 | genuinely different exposure shape |
| `gap=1.25` | both | 0.7339 | genuinely different exposure shape |

**The pre-registered primary (`gap=0.75`) passes** (R²=0.841, comfortably
under 0.95) — it is not a relabeled flat rescale of v4. **But the grid's
single best-performing cell, `gap=0.00`, crosses the artifact threshold**
(R²=0.9544 > 0.95). This is a load-bearing finding, not a footnote: the
one configuration whose raw Sharpe number looks most like a win is also
the one this project's own artifact test says to distrust most — R² rises
monotonically as `gap` shrinks toward the no-hysteresis limit, i.e. as the
veto's own behaviour converges toward "hold less, always," exactly the
exposure-level-artifact mechanism this checklist exists to catch (R-33,
R-34, R-41-conservative, R-46-conservative, R-53-conservative).

## Causality probe (both pathways + identity check)

| probe | decisions at/before cut | `target`/`v15_frac`/`v15_veto` max\|diff\| before cut |
|---|---|---|
| PRICE tamper | PASS | 0.000e+00 (all 3 columns) |
| MACRO tamper (stress_z pathway) | PASS | 0.000e+00 (all 3 columns) |
| both at once | PASS | 0.000e+00 (all 3 columns) |
| identity check (no macro data ⇒ candidate ≡ v4) | — | max\|diff\| = 0.000e+00, PASS |

No lookahead on either information pathway; the override mechanism
recovers v4 exactly when macro data is entirely unavailable, as designed.

## Secondary falsification test: ETH (pre-2020) vs BTC control (pre-2020)

Pre-registered rule: underperforms v4 on ETH spot while matching/beating
v4 on the identical-pipeline BTC control → FAIL.

| config | market | BTC ratio (cand/v4) | ETH ratio (cand/v4) | flag |
|---|---|---|---|---|
| `gap=0.00` | spot | 1.220× | 0.990× | **FAIL** |
| `gap=0.00` | futures | 1.171× | 0.960× | **FAIL** |
| `gap=0.50` | spot | 1.284× | 0.989× | **FAIL** |
| `gap=0.50` | futures | 1.208× | 0.940× | **FAIL** |
| `gap=0.75` (primary) | spot | 1.353× | 1.047× | caution |
| `gap=0.75` (primary) | futures | 1.264× | 0.986× | **FAIL** |
| `gap=1.00` | spot | 1.356× | 1.070× | caution |
| `gap=1.00` | futures | 1.274× | 1.040× | caution |
| `gap=1.25` | spot | 1.135× | 1.057× | caution |
| `gap=1.25` | futures | 1.049× | 1.026× | caution |

**Overall verdict: FAIL.** Every config beats v4 handily on the BTC
control (1.05×–1.36×) but the improvement compresses or reverses on ETH
in 5 of 10 cells (down to 0.94×–0.99×), including the primary candidate's
own futures cell (1.264× BTC vs 0.986× ETH). This is exactly the
asset-specific signature the pre-registered rule was written to catch: a
genuinely asset-agnostic, market-wide signal (VIX/DXY) should not produce
a BTC-only edge, and here it does.

## Verdict: NEGATIVE — does not clear the promotion bar, holdout not warranted

The mechanism is rejected on **three independent, pre-registered
grounds**, meeting ROUTINE.md's "all must hold" bar in reverse (any one
failure is sufficient, and three failed here):

1. **Fails the primary pre-registered falsification test (lead-time).**
   Against the 3-anchor majority that actually determines the gate's own
   flip, the veto leads only 4/12 matched episodes (33%), median offset
   **−5.5 days** — it lags, replicating R-53's finding for the averaged
   version almost exactly. **This directly resolves B-21's own named
   tension: making the combination rule blunter does not rescue the
   mechanism's timing.** The veto is built from the identical `stress_z`
   signal and hysteresis discipline as the averaged version; only *how*
   the flip is used downstream differs, and the flip itself arrives no
   earlier.
2. **Fails the plateau check.** The gap-neighbourhood spread is 0.32,
   comfortably above the 0.2 noise floor — a genuine peak, not a plateau —
   and the single best-scoring point in the grid is the explicit
   no-hysteresis negative control (`gap=0.00`), not the pre-registered
   primary candidate.
3. **Fails the secondary ETH falsification test.** 5 of 10 (config,
   market) cells underperform v4 on ETH while beating it decisively on
   the BTC control, including the pre-registered primary's own futures
   cell — an asset-specific pattern a genuinely market-wide signal should
   not produce.

It **passes** two of the mandatory integrity checks: causality (0.0
lookahead on both the price and macro pathways, plus an exact
no-macro-data identity recovery of v4) and the exposure-artifact check
**at the pre-registered primary config only** (R²=0.841). It notably
**fails** the exposure-artifact check at the grid's actual best-performing
point (`gap=0.00`, R²=0.9544) — the honest reading is that the visually
most attractive number in this entire experiment is partly an exposure-
level-artifact effect, not a genuine timing or gate-quality advantage.

The primary candidate's own inner-validation Sharpe edge (+0.177 spot,
+0.159 futures) sits **under** the ±0.2 noise floor on both markets — it
would need the drawdown/tail improvement clause (a real ~6–7pp DD
reduction on both markets) to qualify at all, and even that clause cannot
rescue a candidate that already failed its own pre-registered primary
falsification test and the plateau check.

**This candidate does NOT clear ROUTINE.md's step-4 promotion bar and a
fresh holdout consultation is NOT warranted.** Per ROUTINE.md's own
instruction not to spend the holdout on a candidate that already failed
pre-registered gates, **the 2023+ holdout was never read** (confirmed by
the grep transcript above; program holdout-consultation total unchanged).

**One-line lesson, resolving B-21:** a hard, unweighted macro veto is not
an obviously stronger candidate than the precision-weighted average R-53
already rejected "just because the combination rule is blunter" — the
underlying signal's timing (lag, not lead, against the price-anchor
majority) is the actual constraint, and it is identical in both
mechanisms because both are built on the same `stress_z`/hysteresis
construction; blunting the combination rule changes how a late signal is
used, not how late it arrives. The apparent inner-validation edge that
motivated filing B-21 in the first place traces substantially to the
grid's most extreme (no-hysteresis) point, which independently fails this
project's own exposure-artifact test — the two failure modes (lag, and
artifact-at-the-winning-point) reinforce rather than contradict each
other, since a memoryless threshold both reacts fastest to noise and
converges fastest toward "hold less, always."

**Next step.** B-21 is now answered and should close as NEGATIVE, on the
same footing as R-53's two rows. No further variant of this VIX/DXY
`stress_z` signal combined with `kelly_regime_v4`'s gate — averaged,
weighted, hard-vetoed, or otherwise — is recommended without a materially
different underlying macro signal or a stress-detection method that
demonstrably leads the 3-anchor majority on this project's available
episodes, which none of R-53's or this round's three combination rules
achieved. Per the standing recommendation repeated since R-46/R-52/R-53:
`scripts/paper_trade.py` (B-06, ongoing since R-48) remains the highest-
value zero-cost item for a future session with spare capacity.
