# R-142 (CONSERVATIVE branch) -- dual-quarter futures term-structure SLOPE as a confirming vote (08-25)

Unregistered. Code: `experiments/r142_conservative_slope_vote.py` (appended
after the frozen pre-registration banner; the banner itself is untouched).
Shared, frozen, read-only infrastructure: `experiments/r142_shared.py`
(not edited by this branch). Executed exactly as pre-registered, verbatim,
per docs/ROUTINE.md's step-0 "execute the frozen pre-registration" rule.

## 1. Direction (as frozen, not re-derived here)

Does the SLOPE between Deribit's two nearest, simultaneously-listed
quarterly BTC futures (`ann_basis_next - ann_basis_front`, z-scored on a
20-day trailing baseline) lead `kelly_regime_v4`'s own slow 3-anchor
(20/40/80-day) gate around this project's standing 4-episode BTC stress
table, the same way R-120's front-quarter basis LEVEL and MOMENTUM were
each tested and failed? Full mechanism, citations and the "not a
duplicate of R-120" argument are in the module docstring and are not
repeated here (one citation trail in one place, per this project's own
convention).

## 2. Step A -- the mandatory measurement gate (BTC, 4 usable episodes)

Primary feature `slope_z`: `r142_shared.dual_quarter_slope(...).slope`,
z-scored on its own trailing 20-day mean/std (`r142_shared.slope_zscore`).
Threshold: bidirectional `|slope_z| >= 1.5`. Search window: onset +/- 60
days. Anchor-gate flip: `r142_shared.nearest_transition(direction="down")`
on `r142_shared.anchor_majority` (V4_HORIZONS=(20,40,80), V4_BAND=0.01,
byte-for-byte `kelly_regime_v4`'s own construction). Null:
`r142_shared.block_bootstrap_lead_null`, 1000 draws, **block_days=35**
(1.75x the 20-day z-score window -- long enough that a circular shift
cannot resample the same 20-day-smoothed autocorrelated hump into two
different draws, comfortably inside the 60-day episode window's own
geometry; chosen once, before any real-data lead number, not swept).
Seed=142 (this round's own number).

| episode | anchor flip | slope_z crossing | LEAD (days) | null median | null p90 | lead>0? | lead>p90? | PASS |
|---|---|---|---|---|---|---|---|---|
| 2020-03 COVID crash | 2020-03-08 18:45 UTC | 2020-03-12 02:00 UTC | **-3.30** | -3.22 | -2.69 | No | No | **FAIL** |
| 2021-11 top / 2022 bear | 2021-11-04 19:10 UTC | 2021-11-09 21:55 UTC | **-5.11** | -5.18 | -0.45 | No | No | **FAIL** |
| 2022-05 Terra/Luna | 2022-04-21 23:55 UTC | 2022-05-09 00:00 UTC | **-17.00** | -16.98 | -10.95 | No | No | **FAIL** |
| 2022-11 FTX collapse | 2022-11-08 09:10 UTC | 2022-11-08 00:20 UTC | **+0.37** | +0.39 | +1.02 | Yes | No | **FAIL** |

**Episodes passing: 0/4.** Gate requires `>= r142_shared.MIN_EPISODES_PASS_BTC` (3).

**GATE VERDICT: FAIL.**

Three of four episodes have the slope_z crossing occurring strictly
*after* the anchor gate's own down-flip (negative lead outright); the
fourth (FTX) is nominally positive (+0.37d) but sits below its own null's
90th percentile (+1.02d) -- indistinguishable from an arbitrary
time-shift of the same series, the identical pattern R-81/R-84/R-120
found on their own one nominally-positive cell.

## 3. Per the frozen decision rule (banner item 3): STOP

The gate failed at 0/4 (< 3/4 required). Per the pre-registration, this
branch **stops here**: no confirming-vote strategy was built, ETH was not
touched, and the holdout (`>= 2023-01-01`) was never read. This is the
modal, pre-named outcome (19/19 prior INFO-axis Step-A gates in this
ledger have failed an analogous gate; this makes 20/20).

## 4. Diagnostic note (disclosed, not part of the pre-registered decision)

`slope_z`'s numeric lead/lag values above are, to 2 decimal places,
identical to R-120's own front-quarter basis-LEVEL branch's reported
lead numbers on the same 4 episodes (COVID -3.30d, 2021-top -5.11d,
Terra/Luna -17.00d, FTX +0.37d -- see docs/LEDGER.md's R-120 entry). This
was checked, not assumed: a direct correlation of `slope_z` against
`basis_z` (R-120's own construction, recomputed here from `r120_shared`,
same 20-day window) over their 420,887 jointly-valid bars gives **r =
-0.9963**, and the correlation restricted to the COVID episode's own
+/-61-day window is **r = -0.998**. Mechanically: `front_dte` ranges
0-90 days and `next_dte` ranges 91-181 days in this dataset (a near-fixed
~90-day gap set by the quarterly roll cycle), and the `365.25/dte`
annualization convention (shared, R-120's own choice, reused verbatim)
amplifies the *front* leg's own short-horizon noise so strongly that it
dominates both `ann_basis_front` alone and the `slope = ann_basis_next -
ann_basis_front` difference, with opposite sign. In this specific
instrument/annualization construction, the SLOPE is empirically **not**
the "structurally distinct, cross-sectional" statistic the
pre-registration's citation trail (Bianchi, Fan, Miffre & Zhang 2023)
argued it would be relative to R-120's own LEVEL -- it is close to
`-1 x` a re-scaling of it. This does not change the pre-registered
decision (the gate still fails 0/4 either way, and the near-mirror
relationship is itself consistent with why: both statistics inherit the
same lagging-relative-to-price behaviour from the same front-leg
annualization term), but it is disclosed here because it bears on any
future INFO-axis round considering a curve-shape statistic on this same
quarterly dataset: the LEVEL and SLOPE are not independent evidence on
this instrument, whatever they are in the commodities literature this
round's citations describe.

## 5. Causality checks

- `pytest tests/test_causality_strict.py`: **51 passed** (run before
  trusting any number in this report, per the task's own instruction).
- `pytest -q` (full suite): **516 passed** in 203.9s.
- Causal truncation probe on this branch's own `slope_z` construction
  (`truncation_causality_probe`, 3 checkpoints: bars 150,000 / 250,000 /
  350,000 into the BTC frame): **PASS, PASS, PASS** -- an early row's
  `slope_z` value is bit-identical whether or not later bars exist,
  confirming `dual_quarter_slope`'s row-local column selection plus
  `merge_asof(direction="backward")` plus a trailing `.rolling()` really
  is causal in this construction, not merely assumed.
- Holdout guard: `assert_no_holdout` fired on every frame this file
  touched (BTC spot bars and the BTC quarterly contract file); max
  timestamp read anywhere in this session: `2022-12-31 23:55:00+00:00`,
  strictly before `OOS_START = 2023-01-01`.

## 6. Configurations evaluated

**4** -- the Step-A gate itself: 4 usable BTC episodes x 1 fixed
threshold (`|slope_z| >= 1.5`), a fixed, non-swept measurement gate (this
project's standing accounting convention for this construction, matching
R-53/R-73/R-74/R-79/R-81/R-84/R-120's own Step-A counts). No sweep was
run because Step B was never reached. The causal truncation probe (3
checkpoints) and the diagnostic slope_z/basis_z correlation check are
disclosed above but are not swept configurations against a decision
threshold, so they are not added to this count, matching R-120's own
accounting convention for its own truncation probe.

## 7. Verdict

**NEGATIVE at Step A.** `slope_z` fails this project's Step-A lead-time
gate at 0 of 4 usable BTC episodes (three episodes lag the anchor gate's
own reaction outright; the fourth's nominally-positive lead does not
clear its own block-bootstrap null's 90th percentile) -- below the
required `>= 3/4`, so per the frozen decision rule this branch stops
here: no confirming-vote strategy was built, ETH was never touched, and
the holdout was never read (max timestamp read: 2022-12-31 23:55 UTC).
This is the modal, pre-registered outcome and the 20th INFO-axis signal
in this ledger to fail an analogous gate. A genuine and disclosed
nuance -- this branch's own `slope_z` turns out to be, empirically,
nearly the negative mirror of R-120's own basis-LEVEL `basis_z`
(r=-0.996 to -0.998) on this specific quarterly-futures/annualization
construction, not the independent cross-sectional statistic its citation
trail (Bianchi, Fan, Miffre & Zhang 2023) predicted -- does not move the
decision (both statistics fail the same gate for the same underlying
reason) but is recorded for whichever future round next considers a
curve-shape input on this dataset.
