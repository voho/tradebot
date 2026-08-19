# B-05, "novel" branch — a continuous, derived funding-crowding correction

Files: `experiments/funding_crowding_novel.py` (strategy, unregistered),
`experiments/run_funding_crowding_novel.py` (driver). Independent of, and
disjoint from, the parallel "conservative decile gate" branch of B-05 —
that branch's files were not read, imported, or coordinated with.

## Mechanism (restated in my own words)

`kelly_regime_v4` already sizes exposure as `frac(vote) * min(target_vol /
realized_vol, max_leverage)`, which is the code's empirical stand-in for
the classic fractional-Kelly optimum `f* = mu/sigma^2` (second-order
expansion of log-growth `g(f) = f*mu - (sigma^2/2)*f^2`, maximized at
`f*`). A perpetual funding payment is a real, causally-observed cash flow
proportional to the position held: holding exposure `f` for one period
costs `r_t * f` in expectation, where `r_t` is the current annualized
funding rate. That is the same "cost of holding" object Constantinides
(1986) and Davis & Norman (1990) put into the classic no-trade-band
derivation (already used once in this repo, for a *one-off* trading
friction, in L-05/L-06's `kelly_regime_ev`) — except funding is a
*continuously accruing* holding cost rather than a one-off transaction
cost, and He et al. (2024) give the economic reason it behaves that way:
the funding rate is the no-arbitrage market-clearing price of being on
the crowded side of a perpetual, which is the same crowding
Cardaliaguet & Lehalle (2018) already grounds the regime vote in — except
*observed directly* every 8 hours rather than inferred from price. Adding
the drag term to growth, `g(f) = f*mu - (sigma^2/2)*f^2 - r_t*f`, and
re-maximizing gives

    f*_adjusted = f*_before - r_t / sigma_t^2

implemented by subtracting `funding_scale * r_t / sigma_t^2` (r_t: causal,
annualized funding; sigma_t^2: `kelly_regime_v4`'s own already-computed
realized-volatility estimate, squared — no new estimator) from the
existing post-vote target exposure, every bar, before the existing 10%
deadband and `[0, max_leverage]` clamp are re-applied unchanged.
`funding_scale` (default 1.0) is the only knob, used solely as a
robustness check at 0.5x/2x on the frozen structural form — never swept
for a best value.

## Pre-registered checks (written before running anything, per the task)

**Materiality check (#1).** Compute the distribution of `|r_t/sigma_t^2|`
on inner-train and compare it to `kelly_regime_v4`'s own typical target
exposure (order 0-2, given the 2x cap). If the correction sits three-plus
orders of magnitude below that range almost always, the mechanism is
empirically vacuous — a legitimate, reportable negative — regardless of
how cleanly it was derived.

**Falsification (a) — fee tier.** Re-run inner-train at the Bitstamp
entry taker tier (0.40%). If the extra re-targeting turnover eats the
benefit (the general R-13 finding), the fee-tier result should look
materially worse, relative to baseline, than the 0.10%-tier result.

**Falsification (b) — real funding-cost accounting.** Run the frozen
strategy and the baseline over inner-train+inner-validation combined
(2020-01-01..2022-12-31) with real funding actually charged. The
mechanism's actual point is the COST constraint (R-14): it should reduce
the dollar funding cost paid relative to baseline, without materially
hurting the funding-free gross return. If the funding-free return is
markedly worse, the "correction" is really just an aggressive de-lever in
disguise, not a targeted funding-timing fix.

**Causality probe.** A by-hand lookahead check in the style of R-28:
tamper every bar strictly after a cut (prices x3//3, and the funding
column too, since it is the new input this branch adds) two opposite
ways; every decision and every `prepare()` column at or before the cut
must be bit-identical. This substitutes for `tests/test_causality_strict.py`,
which only covers registered strategies.

---

## Results

All evaluation restricted to 2020-01-01..2022-12-31 (funding data starts
2020-01-01; nothing at or after 2023-01-01 was run, printed, or inspected).
`$1,000` start balance both markets.

### Materiality (inner-train, 2020-01-01..2021-12-31, spot, funding_scale=1.0)

| stat | \|r_t/sigma_t^2\| |
|---|---|
| mean | 0.6089 |
| median | 0.3119 |
| 90th pct | 1.5031 |
| max | 11.69 |

`kelly_regime_v4`'s own mean \|target\| while in-market over the same
window: **0.629** (range 0-2x, deadband 0.10). Mean correction/typical-target
ratio: **97%**; median ratio: **50%**. The correction exceeds the 10%
deadband on **85.3%** of bars.

**Verdict on materiality: emphatically material — not vacuous.** The
back-of-envelope in the task brief (mean funding ~15-20%/yr, vol ~55%,
giving a ratio around 0.4-0.7) turns out right in direction but the
realized distribution is wide and often *larger* than the target itself
(90th pct 1.5 against a 2x cap), because realized funding swings far more
than its long-run mean across the 2020-2021 bull and the 2022 bear. This
is the opposite failure mode from the one flagged as a real risk in the
brief: the correction is large enough to dominate the decision on the
large majority of bars, which raises a different question the sensitivity
sweep below addresses — is a correction this large itself stable, or does
it swing the strategy so hard that it stops looking like `kelly_regime_v4`
at all?

### Step 4 — funding_scale sweep vs baseline

| split | market | config | final | return | trades | max DD | Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| inner-train | spot | kelly_regime_v4 | $4,336 | +333.6% | 40 | 24.3% | 1.99 |
| inner-train | spot | scale=0.5 | $5,125 | +412.5% | 103 | 11.9% | 3.14 |
| inner-train | spot | **scale=1.0 (frozen)** | **$3,621** | **+262.1%** | **113** | **9.7%** | **3.01** |
| inner-train | spot | scale=2.0 | $2,404 | +140.4% | 123 | 12.2% | 2.36 |
| inner-train | futures 5x | kelly_regime_v4 | $5,502 | +450.2% | 40 | 21.5% | 2.22 |
| inner-train | futures 5x | scale=0.5 | $6,256 | +525.6% | 103 | 11.4% | 3.40 |
| inner-train | futures 5x | **scale=1.0 (frozen)** | **$4,201** | **+320.1%** | **113** | **10.5%** | **2.97** |
| inner-train | futures 5x | scale=2.0 | $3,291 | +229.1% | 123 | 15.3% | 2.55 |
| inner-validation | spot | kelly_regime_v4 | $766 | -23.4% | 26 | 28.2% | -1.21 |
| inner-validation | spot | scale=0.5 | $887 | -11.3% | 33 | 19.4% | -0.56 |
| inner-validation | spot | **scale=1.0 (frozen)** | **$1,006** | **+0.6%** | **60** | **16.5%** | **0.12** |
| inner-validation | spot | scale=2.0 | $1,182 | +18.2% | 93 | 13.3% | 0.99 |
| inner-validation | futures 5x | kelly_regime_v4 | $741 | -25.9% | 26 | 30.6% | -1.36 |
| inner-validation | futures 5x | scale=0.5 | $886 | -11.4% | 33 | 19.9% | -0.56 |
| inner-validation | futures 5x | **scale=1.0 (frozen)** | **$1,125** | **+12.5%** | **60** | **14.8%** | **0.69** |
| inner-validation | futures 5x | scale=2.0 | $1,274 | +27.4% | 93 | 15.7% | 1.11 |

**Reading it honestly.** On inner-train (the 2020-2021 bull, where funding
ran richest — this is R-14's "adverse timing" window exactly), the
correction *costs* return relative to baseline ($3,621 vs $4,336 spot)
while roughly halving drawdown (9.7% vs 24.3%) and lifting Sharpe (3.01 vs
1.99) — a trade of upside for a much shallower path, consistent with the
mechanism doing exactly what it was derived to do (de-lever when the
crowding cost is highest). On inner-validation (the 2022 bear, where
`kelly_regime_v4` alone loses money on both markets), the correction
**flips the sign of the return** — spot goes from -23.4% to +0.6%, futures
from -25.9% to +12.5% — while cutting drawdown by 12-16 percentage
points. That is the more interesting number, because inner-validation is
a real, different regime, not a re-test of the same bull.

**Sensitivity is not a clean plateau.** Effect size is monotone but not
flat in `funding_scale` across either period: on inner-train, higher
scale trades progressively more return for progressively less drawdown
(0.5x is a free lunch relative to 1.0x on this slice — more return *and*
comparable/better Sharpe); on inner-validation, higher scale is
monotonically *better* on every axis measured (0.5x is worse than 1.0x,
which is worse than 2.0x). The two periods disagree about which knob
setting is best, which is itself informative: this is not a mechanism
that is indifferent to its one free parameter, so "un-fit" does not mean
"the exact value used doesn't matter" — it means the value used was
derived, not searched, even though the sweep shows the result is
scale-sensitive. That sensitivity, not a plateau, is the honest
qualifier on the size of the effect.

### Falsification (a) — 0.40% Bitstamp taker tier, inner-train spot

| tier | config | final | return | trades | max DD | Sharpe |
|---|---|---:|---:|---:|---:|---:|
| 0.10% | kelly_regime_v4 | $4,336 | +333.6% | 40 | 24.3% | 1.99 |
| 0.10% | scale=1.0 | $3,621 | +262.1% | 113 | 9.7% | 3.01 |
| 0.40% | kelly_regime_v4 | $3,631 | +263.1% | 40 | 27.9% | 1.78 |
| 0.40% | scale=1.0 | $2,120 | +112.0% | 113 | 15.7% | 1.80 |

At 0.40%, the novel arm's raw return falls further behind baseline in
absolute terms ($2,120 vs $3,631, a much wider gap than at 0.10%) —
turnover (113 fills vs 40) is expensive, confirming R-13's general
finding. But Sharpe does **not** collapse below baseline (1.80 vs 1.78,
essentially tied) because the drawdown reduction survives the fee hike
(15.7% vs 27.9%). **Falsification (a) partially triggers**: return-based
comparisons are not robust to the real fee tier (the higher-turnover
correction loses ground on return), but the risk-adjusted and drawdown
story is not erased by it either. Read this as: promote-on-return would
fail at 0.40%; promote-on-drawdown/Sharpe would not.

### Falsification (b) — real funding charged, combined 2020-2022, futures 5x

| config | funding-free final | with-funding final | cost | funding paid |
|---|---:|---:|---:|---:|
| kelly_regime_v4 | $4,218 | $3,060 | -27% | $948 |
| **funding_crowding_novel scale=1.0** | **$4,574** | **$4,310** | **-6%** | **$151** |

This is the strongest result in the file, and it is the one the
mechanism was actually built to produce (R-14's COST constraint, not a
return claim). The frozen arm pays **84% less funding in dollars**
($151 vs $948) than the baseline over the same combined window, and it
does so **without a worse funding-free gross return** — the funding-free
number is actually slightly higher ($4,574 vs $4,218), not a
de-lever-and-hope trade-off. **Falsification (b) does not trigger**: the
correction is not merely an aggressive de-lever wearing the funding
mechanism as a label: it targeted the cost it was derived to target.

### Causality probe

Tampered every bar from index 195,000 of a 200,000-bar slice (prices
x3//3, funding column x3//3, two opposite directions). Every queued order
at cut-1 through cut-1,000 matched between the two tampered runs, and the
max absolute difference in `target`/`funding_correction`/`kelly_vol`
before the cut was `0.000e+00`. **PASS.**

---

## Bookkeeping

**22 distinct (strategy-config, market, period, fee-tier) backtests**
evaluated in this branch: 16 in the funding_scale sweep (2 markets x 2
periods x [baseline + 3 scales]), 2 more for the 0.40% fee tier (baseline
+ scale=1.0; the 0.10%-tier pair was reused from the sweep, not
re-counted), and 4 more for the combined-period funding-cost check
(baseline + scale=1.0, each funding-free and funding-charged). The
materiality check reused two already-counted backtests. Only `scale`
{0.5, 1.0, 2.0} was ever varied as a search axis — three configurations,
not a sweep of the correction's structural form.

## Honest verdict

**The mechanism is not empirically vacuous — it is the opposite: large
enough to dominate the sizing decision on the majority of bars (85%
exceed the existing deadband), and it produces the single strongest
dollar-cost result in the funding line of work to date (84% less funding
paid, at zero funding-free cost).** It also flips inner-validation
(2022 bear) from a losing to a winning period on both markets, and cuts
drawdown materially everywhere it was measured.

Set against that: it is not a free lunch. (1) It gives up return in the
2020-2021 bull specifically — the period R-14 already identified as
where funding runs richest against this strategy — which is the
mechanism working as designed, not a bug, but it means a return-only
promotion bar would not clear at the 0.40% fee tier. (2) The one free
parameter is not on a plateau: inner-train and inner-validation disagree
about which `funding_scale` is best, so "derived, not fit" is true of the
*form* of the correction but the *size* of its effect is scale-sensitive
in a way L-05/L-06's fee-derived band was not (that band collapsed
cleanly to "no rebalance is ever worth it" past a threshold; this one
does not have an equally clean degenerate case). (3) Turnover roughly
triples (40 to 113 fills on inner-train), so any promotion case has to be
made on drawdown/cost grounds, explicitly, not on the 0.10%-tier return
number.

**Recommendation: this is worth spending part of the project's holdout
on, but only as a drawdown/cost claim, not a return claim** — the same
qualifier `kelly_regime_v4` itself carries, and the same one R-28's
strongest number needed (and did not survive matched risk). Before any
holdout read, the promotion bar should be pre-registered on **drawdown
and dollar funding cost paid**, explicitly excluding raw return at the
0.10% fee tier as a criterion, given falsification (a)'s result. This
report does not itself constitute step 4 (pre-registration into
`docs/LEDGER.md` and a holdout read) — it is the step-3 record for
whoever runs step 4, consistent with `docs/ROUTINE.md`'s split between
"iterate on the inner split" and "pre-register, then look once."
