# B-05 — Funding as a gate on `kelly_regime_v4`

**Date:** 2026-08-18 · **Attacks:** COST · **Verdict: NEGATIVE**

## Idea

Multiply `kelly_regime_v4`'s existing target exposure by a scale factor
that falls when trailing perpetual funding sits in its own top decile
(crowded longs) — unless trailing price momentum is also strongly
positive, in which case the scale factor stays at 1.0. This is the
low-turnover use of R-16 the ledger prescribes: a gate on an existing
sizing decision, never a standalone reversal (the high-turnover form is
where strategies go to die, R-12).

## Sources

- **He, Manela, Ross & von Wachter (2024, "Fundamentals of Perpetual
  Futures", *Journal of Finance*)** — the no-arbitrage relation tying
  funding to the perp-spot basis; already cited in this repo's R-15/B-03,
  the theoretical reason funding is economically meaningful rather than
  noise.
- **Zhang (2026, SSRN 6185958, "Funding Rate Mechanism in Perpetual
  Futures")** — models funding as an algorithmic feedback rule; with
  risk-constrained arbitrageurs *and* momentum speculators, a linear
  funding rule produces an **endogenous mean-reverting basis** in
  equilibrium. Theoretical account of exactly the asymmetry R-16 measured
  empirically: funding pulls price back toward fair value except when
  momentum traders overwhelm the arbitrageurs — which is the override
  case this design tests.
- **Nimmagadda & Sasanka (2019)** — earliest systematic funding-rate
  study (BitMEX): documents heteroskedastic funding dynamics and a
  Granger-causal link from funding to the perp price, i.e. funding
  carries directional information forward in time, not merely
  contemporaneous with price.
- **Presto Research, "Can Funding Rate Predict Price Change?"** (industry
  quantitative research note) — funding-rate *changes* account for
  roughly 12.5% of the variance in price change over the following 7
  days, decaying fast at longer horizons; the authors judge the signal
  more reliable **cross-sectionally** (relative funding across many
  perpetuals) than for single-asset timing. Taken seriously here: it is
  why the variants below stay conservative (partial scale-down / hard
  gate, never a reversal) and why the pre-registered expectation was a
  small effect, not a large one.
- **Crowding/decay of the funding-carry premium, 2020–2025** — already in
  R-15/`VALIDATION.md` (Sharpe 6.45 → 4.06 in 2024 → negative in 2025 as
  the trade crowded, Ethena and exchange-native delta-neutral supply
  absorbing the premium); independently reconfirmed by a 2026 web search
  for this report. Does not bear directly on this gate (which trades
  funding-as-signal, not funding-as-carry) but is the reason this design
  does not lean on funding levels or thresholds persisting unchanged
  beyond 2023.
- Web search also surfaced repeated practitioner commentary (BloFin,
  MetaMask/Consensys academy, industry funding trackers) converging on
  the same qualitative point the sources above make quantitatively:
  funding is a **crowding/cost signal, not a standalone timing tool** —
  useful as a risk filter layered on an existing position, not as a
  reason to flip direction. That is exactly the shape this design uses.

## Constraint attacked and not-a-duplicate-of

**COST.** R-14 measured that `kelly_regime_v4`'s funding bill is
adversely timed (+20.05%/yr while holding vs +2.78%/yr while flat,
2020–2023) because the crowding that produces the trend signal is what
sets the funding rate. This is the first strategy that tries to *avoid*
the adversely-timed part of that cost rather than merely report it.

- **L-04/L-01/L-02/L-03** (`kelly_regime` family) — same sizer and regime
  vote, unchanged; this file only multiplies the existing target by a
  funding-based scale factor.
- **R-16** — measured the funding→forward-return correlation; this is
  the first attempt to turn it into a trading rule, in the ledger's own
  prescribed low-turnover form.
- **L-12 (`harsanyi_crowd`)** — also used a crowding intuition, but as a
  *direction* signal (long/short) and lost. This design only ever scales
  an existing long position down, never flips it.
- **B-03 (funding harvest)** — that is a market-neutral carry strategy
  (long spot, short perp); this is a directional strategy's sizing
  overlay on the same data file. Different mechanism.

## Data constraint (confirmed, not assumed)

```
4,383 settlements  2020-01-01 03:00:00+00:00 -> 2023-12-31 19:00:00+00:00
1,010,889 price bars  2017-01-01 -> 2026-08-12  (data: real)
funding covers 1,460 of 3,510 days (41.6% of the dataset span)
```

The gate is a no-op by construction (`funding_active=False`, scale
forced to 1.0) for every bar outside `[2020-01-01 03:00, 2023-12-31
19:00 + 8h]` — see `experiments/funding_gate.py::_funding_state`. It does
**not** freeze the last known 2023 rate forward into 2024+; coverage ends
exactly one settlement interval past the last observed settlement.

## Design (written before any variant was tuned)

**Mechanism, one sentence.** `target = kelly_regime_v4_target × scale`,
where `scale` drops from 1.0 to `floor` when the trailing rolling
percentile of funding (causal, `.rolling(window).rank(pct=True)`, one
extra settlement of lag) is at or above `pctile_threshold` — unless
trailing `momentum_days`-day price return exceeds `momentum_override`, in
which case `scale` stays at 1.0 even though funding is rich.

**Variants (3 axes, swept as a small grid, not hand-picked):**

| axis | values | what it tests |
|---|---|---|
| `pctile_threshold` | 0.90, 0.95 | top decile vs a stricter top-5% cut |
| `floor` | 0.0, 0.35 | hard gate (stand flat) vs partial scale-down |
| `momentum_override` | None, 0.05, 0.10 | no override vs "unless price is up ≥5%/7d" vs "≥10%/7d" |

Fixed a-priori: `window_days=180` (rolling percentile lookback),
`momentum_days=7` (matching R-16's own 7-day forward/trailing window).
Both are sensitivity-checked afterward (window: 90/365d; the grid above
already covers the other two axes densely).

**Falsification test, chosen now, before any code ran:** the ETH
replication test is not usable here (no ETH funding data is committed),
so — per the assignment's fallback — **does the gate's effect survive
resampled windows drawn from inside the funding-covered span
(2020-01-01→2023-12-31), paired against `kelly_regime_v4` on identical
windows**, the R-19/R-28 design restricted to dates where the mechanism
can possibly fire (windows outside that span would trivially "survive"
without exercising the gate at all). **Named failure mode:** if the paired
return/drawdown differences are centered near zero and have no consistent
sign across windows, the gate is measuring noise, not a real effect —
exactly the caution `VALIDATION.md` already attaches to R-16 ("four
years, one asset... every apparent predictor died out-of-sample").

**What would make it fail, named before any code ran:** (a) the gate
fires rarely enough that it cannot move a full-period holdout result
(funding covers only ~28% of the holdout by construction); (b) the
momentum override, motivated by R-16's own momentum-tercile table, turns
out to be fitted to that same four-year window and reverses sign
out-of-window; (c) the extra rebalancing the gate introduces costs more
in fees than the avoided funding periods save.

## Step 3 — tuning, inner-train / inner-validation only

**Splits used** (per the assignment, anchored to where funding data
starts rather than the dataset's 2017 start): inner-train
`2020-01-01→2020-12-31`, inner-validation `2021-01-01→2022-12-31` (2021
top + 2022 bear), holdout `2023-01-01→` untouched until frozen.

**Configurations evaluated: 14** — the 2×2×3 = 12-point main grid
(threshold × floor × override), each scored on both inner splits (24
backtests, spot only, matching the table's convention of selecting on
spot), plus 2 more points varying `window_days` ∈ {90, 365} around the
selected triple on both inner splits (180d was already in the main
grid). Baselines (`kelly_regime_v4`, `buy_and_hold`) measured for
reference but not counted as searched configurations.

**Main grid (spot), full numbers in `reports/funding_gate/grid.csv`:**

```
inner-train (2020) — baseline kelly_regime_v4: $3,122, DD 18.0%, Sharpe 2.81
  thr  floor  ovr    final    DD    sharpe
  0.90 0.00  none  $2,931  18.0%   2.89
  0.90 0.00  0.05  $3,225  18.0%   2.92
  0.90 0.00  0.10  $3,291  18.0%   3.01   <- best on inner-train
  0.90 0.35  none  $3,044  18.0%   2.96
  0.95 0.00  none  $2,930  18.0%   2.78
  (12 rows total; every config's max DD is IDENTICAL to baseline's 18.0%
   in this split — the 2020 drawdown event does not coincide with a
   top-decile funding period)

inner-validation (2021 top -> 2022 bear) — baseline kelly_regime_v4: $998, DD 33.2%, Sharpe 0.14
  thr  floor  ovr    final    DD    sharpe
  0.90 0.00  none  $1,151  32.1%   0.40   <- best on inner-validation
  0.90 0.35  none  $1,123  32.2%   0.35
  0.95 0.00  none  $1,061  30.0%   0.25
  0.90 0.00  0.05  $  890  38.0%  -0.07   <- WORST on inner-validation
  0.90 0.00  0.10  $  932  36.6%   0.01
```

**The honest finding from step 3: the momentum override reverses sign
between the two splits.** On inner-train, adding the override *helps*
(scale rises from $2,931→$3,291 as override tightens from none→0.10). On
inner-validation — the split with the actual regime change — the
override *hurts*, badly: every overridden config underperforms both the
no-override config and the `kelly_regime_v4` baseline itself, because the
override re-opens exposure during the last leg of the 2021 rally right
before the 2022 crash, which is precisely the worst place to have it.
This is the same caution `VALIDATION.md` already attaches to R-16's own
table ("middle quintiles are non-monotone... a warning about how much of
the rest is noise too"): the theoretically-motivated refinement (grounded
in Zhang 2026's momentum-speculator mechanism and R-16's own
momentum-tercile table) does not survive the one split built to contain a
real regime change. **Selection follows the routine's rule — decide on
inner-validation — so the override is dropped, honestly, against the
mechanism's own motivating citation.**

**Selected on inner-validation:** `pctile_threshold=0.90, floor=0.0
(hard gate), momentum_override=None`. It beats `kelly_regime_v4` on
inner-validation on every axis measured (return $1,151 vs $998, DD 32.1%
vs 33.2%, Sharpe 0.40 vs 0.14) while costing almost nothing on
inner-train ($2,931 vs baseline's $3,122, DD unchanged).

**Window sensitivity (P4 plateau, `reports/funding_gate/window_check.csv`):**

```
inner-validation
  window= 90d  final=$1,108  DD=25.4%  sharpe=0.33
  window=180d  final=$1,151  DD=32.1%  sharpe=0.40   <- selected
  window=365d  final=$1,141  DD=33.2%  sharpe=0.39
```

All three window lengths beat the `kelly_regime_v4` baseline
($998/33.2%/0.14) on inner-validation — a plateau along this axis, and
notably the 90-day window gives the deepest drawdown cut (25.4%) at a
comparable Sharpe. `floor∈{0.0,0.35}` is also a near-plateau at fixed
threshold ($1,151 vs $1,123 at thr=0.90). The axis that is **not** a
plateau is the momentum override, which is the finding reported above,
not a tuning failure to paper over.

## Frozen configuration

```python
FundingGateV4(pctile_threshold=0.90, floor=0.0, momentum_override=None,
              window_days=180.0, momentum_days=7.0)
```

Frozen before any 2023+ data was read — `holdout()` was the first call in
`experiments/run_funding_gate.py` to touch the OOS split, and the frozen
dict above was already the script's default at that point.

## Pre-registered decision rule (written before the holdout was read)

Promote only if **all four** hold, using the project's standing bar with
the baseline specified by the assignment:

- **P1** — holdout spot final balance beats `buy_and_hold` after the
  0.10% fee;
- **P2** — the improvement over **`kelly_regime_v4`** (the incumbent, not
  `buy_and_hold`) exceeds **±0.2 Sharpe** OR is a **≥10pp drawdown
  improvement**;
- **P3** — survives the pre-registered falsification test (resampled
  windows inside the funding-covered span, paired vs `kelly_regime_v4`);
- **P4** — the parameter neighborhood is a plateau, not a single lucky
  peak (reported above, before the holdout was read).

**Stated prediction before looking:** given the gate fires on only
1–6% of bars even inside its active window (`gated_frac` in the grid
above) and funding itself covers only ~28% of the 2023+ holdout by
construction, the expected holdout effect is small — plausibly inside
the noise floor in either direction. P1 is expected to fail regardless
(both `kelly_regime_v4` and `buy_and_hold` are close on this holdout per
R-29/R-30, and this overlay cannot manufacture return the base strategy
doesn't already have). P2 is the live question.

## Holdout, 2023-01-01 → 2026-08-12, $1,000 start, 0.10%/0.05% taker

```
spot                     final       DD      Sharpe   fills   fees
buy_and_hold            $3,839      54.0%     1.03       1     $1
kelly_regime_v4         $3,373      27.8%     1.22     332    $310
funding_gate_v4 (frozen)$3,294      27.8%     1.21     367    $365

futures 5x               final       DD      Sharpe   fills   fees
buy_and_hold           $15,176      60.3%     1.44       1     $2
kelly_regime_v4         $4,901      33.0%     1.36     328    $265
funding_gate_v4 (frozen)$4,938      33.0%     1.38     363    $317
```

**Funding coverage of the holdout: 27.7%** of bars had the gate
`funding_active` (2023 only, out of 2023–2026); **the gate actually
fired (scale<1) on 1.72% of all holdout bars** — the overlay is inactive
by construction for nearly three-quarters of the holdout and, even
inside its active window, changes the position on well under a tenth of
those bars.

**Paired 95% block-bootstrap interval, `funding_gate_v4 − kelly_regime_v4`**
(30-day mean block, 2,000 resamples, 1,319 daily observations,
`reports/funding_gate/intervals.csv`):

```
spot     Δ log growth        ≈ -0.024 [-0.167, +0.114]  P(>0)=0.36
spot     Δ max drawdown (pp) ≈ -0.000 [-2.727, +3.229]   P(>0)=0.47
futures  Δ log growth        ≈ +0.007 [-0.180, +0.203]   P(>0)=0.52
futures  Δ max drawdown (pp) ≈ -0.000 [-3.998, +3.436]   P(>0)=0.59
```

Every interval contains zero. Max drawdown is identical to three
significant figures on both markets — the gate never fires during the
holdout's actual drawdown episodes.

## Cost checks (step 4)

**0.40% Bitstamp entry tier, spot holdout:**

```
                  final     DD      Sharpe   fees
buy_and_hold     $3,827    54.0%    1.03      $4
kelly_regime_v4  $2,445    34.1%    0.94   $1,027
funding_gate_v4  $2,146    34.1%    0.83   $1,132
```

At the real fee tier the gate's extra turnover (367 vs 332 fills)
turns a near-tie into a clear loss against the incumbent: $2,146 vs
$2,445, Sharpe 0.83 vs 0.94. The overlay adds cost without an offsetting
return.

**Real funding charged, futures holdout (coverage through 2023-12):**

```
                  final     DD      Sharpe   funding paid
buy_and_hold     $14,432   61.6%    1.39        $743
kelly_regime_v4   $4,517   33.0%    1.30        $154
funding_gate_v4   $4,632   33.0%    1.34        $113
```

Here the gate pays less funding ($113 vs $154, the mechanism it was
built to exploit) and gains a small edge ($4,632 vs $4,517). This is the
one place the intended mechanism shows up directionally — but it is a
$115 difference on a $4,500 base, and the paired bootstrap above (run
without funding charged, since that is the table's convention) already
shows the futures Δ log growth interval containing zero; charging
funding does not change the drawdown, and the return gap here is well
within the kind of single-path noise this project's own methodology
rows (R-20, R-29) warn against reading as a result.

## Falsification test result

**40 resampled windows (30–365 days), drawn only from inside the
funding-covered span (2020-01-01→2023-12-31), paired against
`kelly_regime_v4` on identical windows** (`reports/funding_gate/windows.csv`):

```
spot:    funding_gate_v4 − kelly_regime_v4
         return median -0.7pp, gate higher in 28% of windows
         DD     median +0.0pp, gate deeper in 10% of windows

futures: funding_gate_v4 − kelly_regime_v4
         return median +0.0pp, gate higher in 35% of windows
         DD     median +0.0pp, gate deeper in 12% of windows
```

**Result: does not blow up, but shows no consistent edge.** The gate is
not distinguishably worse (drawdown is deeper in only 10–12% of windows,
i.e. usually identical), but it is also not distinguishably better —
on spot it is *higher* than the incumbent in barely more than a quarter
of windows, the opposite of what a real, replicable improvement should
look like. This matches the named failure mode written down before the
test ran: the paired differences are centered near zero with no
consistent sign. The falsification test does not kill the idea outright,
but it actively fails to support it.

## Causality / lookahead probe (by hand — this strategy gets no CI protection)

Reproduced `experiments/run_matched_risk.py`'s two-opposite-tampers
procedure directly (`experiments/run_funding_gate.py::causality`), on a
slice chosen to actually fall inside funding coverage
(`2021-01-01→2022-05-31`, 148,608 bars — the dataset's own 2024+ tail is
**outside** funding coverage by design and would pass this check
trivially without exercising the gate logic at all, so it was
deliberately avoided):

```
slice: 2021-01-01 -> 2022-05-31  (148,608 bars), cut at 2021-09-16
orders match
max |column difference| before the cut (target, funding_scale,
funding_pctile) = 0.000e+00
PASS - no decision at or before the cut moves
sanity: min funding_scale before the cut = 0.00
        (gate fired at least once - check is meaningful)
```

Every column examined (`target`, `funding_scale`, `funding_pctile`) is
bit-identical before the cut between the ×3 and ÷3 tampered copies and
the untampered original, and the sanity check confirms the gate actually
fired somewhere in the tested region (min `funding_scale` = 0.0), so the
PASS is not vacuous. This is the check that would catch a full-series
quantile — `_funding_state`'s rolling percentile is computed with
`.rolling(window).rank(pct=True)` on the funding series alone (never on
price), shifted one further settlement, and merge_asof'd backward onto
the bar grid, so nothing about it can see its own future by
construction.

**`pytest` from the repo root: 436 passed**, 0 failed — confirms the
environment and the existing suite are unaffected (no existing file was
modified for this session).

## Verdict: NEGATIVE

| check | result |
|---|---|
| **P1** (beats `buy_and_hold`, spot, 0.10% fee) | **FAIL** — $3,294 vs $3,839 |
| **P2** (>±0.2 Sharpe or ≥10pp DD vs `kelly_regime_v4`) | **FAIL** — Sharpe Δ −0.01 (spot) / +0.02 (futures), both inside the noise floor; DD Δ = 0.0pp on both markets |
| **P3** (falsification: resampled windows, funding-covered span) | **FAIL to support** — no consistent sign, gate ahead in only 28–35% of windows |
| **P4** (plateau, not peak) | **PASS** on threshold/floor/window; but the momentum-override axis reverses sign between inner-train and inner-validation — reported honestly rather than hidden by only showing the selected point |

**Default reject. Nothing was re-argued after the holdout was read**: the
frozen configuration was set by the inner-validation grid before
`holdout()` was ever called, and no threshold was adjusted afterward.

**Why it failed, in one paragraph.** The mechanism is real in direction
— the gate does avoid some of the richest funding (futures funding paid
drops from $154 to $113 under real charging) and R-16's underlying
correlation is a genuine, if weak, empirical regularity — but it cannot
move a result that depends on it firing, and it almost never fires where
it matters. Funding data covers only 27.7% of the 2023+ holdout by the
design constraint this task itself imposed (never extrapolate past the
committed file), and even inside that window the top-decile threshold
by construction gates only a few percent of bars. Max drawdown on the
holdout is **identical to `kelly_regime_v4` to three significant
figures on both markets** — the gate simply never coincides with the
holdout's actual drawdown episodes. What the overlay reliably does
instead is add turnover (367 vs 332 spot fills), which costs more at the
real fee tier ($2,146 vs $2,445 at 0.40%) than the avoided funding ever
saves.

**A secondary, honest finding worth keeping regardless of the negative
verdict:** the "unless price is also strongly bullish" override — the
piece of the design most directly grounded in R-16's own
momentum-tercile table and in Zhang (2026)'s theoretical account of when
funding's mean-reversion signal breaks down — reverses sign between
inner-train (2020, where it helps) and inner-validation (2021 top → 2022
bear, where it hurts, by re-opening exposure into the last leg of the
2021 rally right before the crash). This is exactly the kind of
regime-instability this project's methodology rows (R-20, R-29) exist to
catch, and it is a caution specifically about R-16 itself: the
correlation there (0.39 with trailing return, momentum-tercile spreads
of a few points) is thin enough that a plausible refinement of it
changes sign across a four-year dataset's two halves.

## Configs evaluated and holdout counter

- **Configurations evaluated in step 3: 14** (12-point main grid ×
  2 inner splits = 24 backtests, spot; + 2-point window-sensitivity
  check × 2 inner splits = 4 more backtests). This is this session's
  contribution to the project trials count; combined with R-32's running
  total of 172, the project figure becomes **172 + 14 = 186**.
- **Holdout consultations this session: ~19** — `holdout()` (3 strategies
  × 2 markets = 6), `interval()` (2 strategies × 2 markets, re-run
  independently = 4), `costs()` (3 strategies × 2 fee tiers on spot = 6,
  + 3 strategies with funding charged on futures = 3). The falsification
  windows and the causality probe do **not** touch the 2023+ holdout (by
  design, same convention as R-19/R-28/R-31/R-32). Combined with R-32's
  running total of **~124**, the project figure becomes **~143**. Per
  R-29/R-30/R-31/R-32, no Sharpe-based claim from this dataset is
  supportable at this trial count regardless of sign; this session's
  result does not depend on that ceiling since it is a clean negative on
  point estimates, paired intervals, and the falsification test alike,
  not a marginal one that would need deflation to resolve.

## Next step

The mechanism direction is not contradicted (funding paid does drop when
charged; the sign of the small effects is usually, not always, the
"right" one), but the constraint that kills it — **only 27.7% of the
2023+ holdout has any funding data at all** — is structural, not a
tuning failure, and no amount of threshold search fixes a signal that is
absent for three-quarters of the period being judged. The honest
next step is the one already at the top of the backlog for exactly this
reason: **B-02 (extend the funding series through 2026)** is the
single item that would let this idea be tested on a holdout where it can
actually act on the majority of bars, and B-05 should not be revisited
before that unless B-02 unblocks. Absent new data, this branch's
recommendation agrees with the ledger's own standing note: a session
that finds B-05 unpersuasive should spend itself on **B-06** (forward
paper-trading recorder), still blocked on network access but the only
source of uncontaminated evidence left.

## Files

- `experiments/funding_gate.py` — the `FundingGateV4` strategy (subclasses
  `KellyRegimeV4`), not registered.
- `experiments/run_funding_gate.py` — driver: `rates`, `sweep`,
  `window_check`, `causality`, `holdout`, `interval`, `windows`, `costs`.
- `reports/funding_gate/*.csv` — grid, window_check, holdout, intervals,
  windows — every number in this report is reproducible from these.
