# R-60 conservative branch — reversion-brake overlay on `kelly_regime_v4`

**Verdict: NEGATIVE.** Fails at the first gate (F1, the exact R-40-through-R-46
failure signature) and does not fix the panel (F3, still 0/6). Not ready for a
holdout consultation; no BTC bar dated 2023-01-01 or later was read anywhere
in this branch.

Full pre-registration, mechanism, and code:
`experiments/r60_conservative_reversion_brake.py` (read the module docstring
first — it is the frozen pre-registration, written before any strategy result
in this file was read).

## Mechanism (frozen before any result was read)

`ReversionBrakeV4(KellyRegimeV4)` overrides only `prepare()`: calls
`super().prepare()` for v4's own unmodified `target`, then applies a bounded
overlay gated on a rolling daily variance ratio VR(q) (Lo & MacKinlay 1988)
and a short-EMA extension z-score:

- `VR_WINDOW_DAYS=90`, `Q_DAYS=5`, `VR_THRESHOLD=0.85`
- `EMA_SPAN_DAYS=5`, `DISP_WINDOW_DAYS=30`, `Z_THRESHOLD=1.5`
- `MAX_FADE=1.0`, `COUNTER_FRAC=0.30` (this branch's two free choices)

Trigger: `VR < 0.85 AND |z| > 1.5`. When triggered,
`new_target = target * (1 - fade*(1 + COUNTER_FRAC*sign(z)))`, which is
provably bounded (`|new_target| <= target` for any `fade, COUNTER_FRAC in
[0,1]`, since v4's own target is always >= 0) — the overlay cannot amplify
v4's exposure or exceed its `max_leverage`, and falls back to v4 exactly
(floating-point identical) whenever the trigger is absent. Daily VR/z values
are computed on `close.resample("1D").last()`, then their index is shifted
forward one calendar day before being reindexed onto the 5m grid with
`ffill`, so no bar ever uses a still-forming day's data.

## Causality

Opposite-tamper probe (R-57/R-59 methodology) on BTC and ETH: **PASS** on
both. Boundedness invariant (`|brake target| <= |v4 target|`) checked
numerically over the whole pre-2023 BTC series: **PASS**. Brake reproduces
v4's target exactly on 96.5% of BTC bars pre-2023; active on the remaining
3.5%. `pytest -q`: **461 passed** (unchanged; no registered strategy touched).
`pytest -q -k causality`: **103 passed**.

## F1 — BTC pre-2020 control (2017-01-01..2019-12-31)

| | Sharpe | Max DD | Final |
|---|---|---|---|
| candidate | 1.312 | 43.3% | $3,313 |
| `kelly_regime_v4` | 1.749 | 43.3% | $6,033 |

Delta Sharpe **-0.438**, worse than the -0.2 noise floor. **F1 FAILS** — the
exact signature that killed R-40, R-41, R-42, R-43, R-45, R-46: drawdown is
*identical* to v4's own (43.3%), so the brake gave up return with no
compensating risk reduction on this window. A post-hoc, non-gating
neighbourhood sweep (`MAX_FADE in {0.5,1.0} x COUNTER_FRAC in {0,0.3,0.6}`,
inner-train + inner-validation) confirms this is a plateau, not noise: every
combination reproduces v4's drawdown exactly and Sharpe at or below v4's in
all 12 cells — see `reports/r60_conservative/neighborhood.csv`.

## F2 — ETH replication (full window, spot @0.10%)

| | mean notional | matched-exposure dDD |
|---|---|---|
| candidate | 0.409 | -9.4pp [-23.1,+18.7] |
| `kelly_regime_v4` | 0.412 | -9.9pp [-23.3,+18.5] |

Candidate's matched-exposure drawdown advantage (-9.4pp) is within 0.5pp of
v4's own (-9.9pp) on the identical window — **F2 PASSES (no regression)**,
tolerance was 5.0pp. Mean notional essentially unchanged (0.409 vs 0.412),
so this is not an exposure-relabeling artifact — the brake barely engages on
ETH's own history.

## F3 (PRIMARY) — 6-asset panel D1, FULL window, spot @0.10%

| asset | candidate DD | matched hold DD | dDD (pp) | 95% interval |
|---|---|---|---|---|
| BCH | 54.9% | 46.8% | +8.5 | [-7.5, +44.1] |
| LTC | 74.3% | 43.0% | +32.8 | [+0.4, +52.6] |
| ETC | 52.6% | 29.4% | +25.1 | [+5.5, +46.3] |
| DASH | 55.4% | 29.4% | +27.0 | [+0.2, +39.1] |
| LINK | 47.3% | 37.9% | +12.6 | [-6.0, +37.8] |
| XTZ | 53.4% | 34.9% | +18.3 | [+1.1, +43.9] |

**0/6** (exact binomial p=1.0000); 4/6 bootstrap intervals exclude zero, all
four against the candidate. Magnitudes are essentially unchanged from v4's
own R-57 baseline (BCH +5.2, LTC +33.8, ETC +23.6, DASH +29.8, LINK +13.4,
XTZ +19.3) — the brake does not move the panel result at all. Candidate mean
notional per asset (0.19-0.31) is close to v4's own R-57 range (0.18-0.26),
so this is not an exposure-relabeling artifact either.

## F4 — context only (0.40% Bitstamp tier)

2/6 beat `buy_and_hold` (DASH, XTZ — both cleared only because holding lost
50-87%). **FAILS**, as every strategy in this project's history does at this
tier; not a gate.

## Decision

Pre-registered rule: READY FOR HOLDOUT only if F1 passes AND F2 does not
regress AND F3 >= 5/6. **F1 fails, F3 fails (0/6, need >=5/6) → NEGATIVE.**
Rule was not moved after seeing any number.

## Configurations evaluated: 54

(causality probe 0 — no `measure()` calls; inner 4; F1 2; F2 6; F3 18;
F4 12; neighbourhood 12.)

## Holdout

**0 BTC bars dated 2023-01-01 or later were read anywhere in this branch.**
`load_btc_no_holdout()` truncates the BTC frame at 2022-12-31 23:59:59 UTC
immediately after loading, before any other line touches it. Holdout counter
unaffected by this branch: **+0**.

## Files

- `experiments/r60_conservative_reversion_brake.py` — the branch, its frozen
  pre-registration, and its causality probe.
- `reports/r60_conservative/*.csv` — per-cell numbers (`inner.csv`,
  `f2_eth.csv`, `f3_panel.csv`, `f4_panel_040.csv`, `neighborhood.csv`). Not
  yet git-tracked — `reports/*` is gitignored by default and this round did
  not add an allowlist entry (that would touch `.gitignore`, an existing
  file outside this round's scope); the operator can allowlist it at merge
  time if the CSVs are worth keeping, the same way R-59 did for its own
  branches.
