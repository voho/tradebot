# R-101 NOVEL branch — causal expanding-window jackknife confidence on `kelly_regime_v4`

**Mechanism, one sentence.** `desired = frac * scale * conf`, where `conf`
is a Quenouille–Tukey delete-one-group jackknife coefficient of variation
(Quenouille 1949; Tukey 1958; Efron 1979) of `frac`'s (the 3-anchor regime
vote's) realized log-growth edge, recomputed *causally as an
expanding-window statistic* over the six standard stress episodes,
counting only those whose ±60-day window has fully closed by the current
bar, defaulting to `conf = conf_floor` (not an undefined jackknife) while
fewer than two episodes have resolved.

**Attacks:** N≈3 (effective sample size ≈3 regime events) — see the
project's standing diagnosis. Gives that diagnosis an actual number
(`CV(conf)`) instead of a qualitative caveat, and — because it is
recomputed causally rather than frozen once — makes that number itself a
time-varying series a live deployment could have used with zero
lookahead.

**Files.** `experiments/r101_novel_jackknife_causal.py` (strategy +
driver), this report.

---

## Step 0 — kill switches, checked first, no Sharpe read before this

At the pre-registered a-priori cell **k=1.0, conf_floor=0.5**, over
inner-train ∪ inner-validation (2017-01-01 → 2022-12-31, 630,721 bars):

| check | measured | threshold | result |
|---|---|---|---|
| **KS-A** real dispersion — CV(conf) | **0.2664** | ≥ 0.05 | **PASS** |
| **KS-B** not a flat rescale — R²(target vs v4's own target) | **0.8588** | < 0.95 | **PASS** |

`conf` ranges 0.500–0.9627 (mean 0.779, std 0.208) over that window, with
57,919 distinct values — genuinely continuous variation, not a step
function sitting near one level.

**Both kill switches pass.** This branch proceeded to the standard
battery, per its own pre-registration.

### Episode resolution timeline (how many of the six are actually available)

| date | window closes (+60d) |
|---|---|
| 2018-01-17 | 2018-03-18 |
| 2018-12-15 | 2019-02-13 |
| 2020-03-12 | 2020-05-11 |
| 2021-11-10 | 2022-01-09 |
| 2022-05-09 | 2022-07-08 |
| 2022-11-08 | **2023-01-07** |

| as of | episodes resolved |
|---|---|
| end of inner-train (2020-12-31) | 3 of 6 |
| end of inner-validation (2022-12-31) | **5 of 6** |

The sixth episode (FTX, 2022-11-08) does **not** resolve within
inner-validation — its window closes 2023-01-07, seven days into what
would be the holdout — confirmed by pure calendar arithmetic, no bar
dated 2023+ was read to establish this. `conf` sits at `conf_floor`
everywhere before 2018-03-18 (0 resolved), stays at `conf_floor` between
2018-03-18 and 2019-02-13 (1 resolved, still short of the "≥2" bar), and
only becomes a genuine jackknife statistic from 2019-02-13 onward.

---

## Step 3 — the sweep (all pre-2023 data only)

**Grid, named in advance, 10 configurations:** `k ∈ {0, 0.5, 1.0, 2.0}`
(k=0 is the identity harness check, not a real confidence configuration —
see the file's docstring) `× conf_floor ∈ {0.3, 0.5, 0.7}` for k>0 (9
cells) `+` the k=0 identity cell `= 10`.

**Configurations evaluated: 40** (10 configs × 2 markets [spot,
futures 5x] × 2 splits [inner-train, inner-validation] via `ev()`,
counted by `N_EVALUATED` in the file). This is the trials count carried
into any deflated-Sharpe reading of this branch's numbers. Separately,
and **not** included in that count (matching R-28's convention of only
counting the search step): the k=0 identity check reproduced
`kelly_regime_v4` **bit-for-bit** in all four cells (spot/futures ×
train/validation) — final balance, trade count, drawdown and Sharpe
matched to the printed digit — which is the harness sanity check working
as designed.

### Inner-train (2017-01-01 → 2020-12-31)

| market | v4 (unmodified) | best novel cells |
|---|---|---|
| spot | Sharpe 2.03, DD 43.3%, +1747.7% | Sharpe 2.06–2.09, DD 26.2–31.6%, +575–1271% (several `floor≥0.5` cells) |
| futures 5x | Sharpe 2.28, DD 35.3%, +2934.4% | **k=0.5/floor=0.5: Sharpe 2.37, DD 28.1%, +1658%**; k=1/floor=0.5: Sharpe 2.37, DD 26.1%, +1557%; k=2/floor=0.5: Sharpe 2.32, DD 24.0%, +1352% |

On inner-train, `floor=0.5` cells form a genuine **plateau across k**
(0.5/1/2 all land Sharpe 2.32–2.37, DD 24.0–28.1% on futures) — not an
isolated peak — with real Sharpe gains (+0.04 to +0.09, inside the ±0.2
noise floor on their own) and real drawdown cuts (−7.2 to −11.3pp,
outside anything that reads as noise). Every cell trades far less
notional than v4 (return roughly halved to a third), the expected effect
of `conf<1` most of the time.

### Inner-validation (2021-01-01 → 2022-12-31) — the split selection is made on

| market | v4 (unmodified) | best novel cell | delta |
|---|---|---|---|
| spot | Sharpe 0.12, DD 33.2%, −1.4% | k=1/any floor: Sharpe 0.12, DD 32.5%, −1.2% | ΔSharpe 0.00, ΔDD −0.7pp |
| futures 5x | Sharpe 0.17, DD 32.3%, +1.5% | k=2/any floor: Sharpe 0.12, DD 31.5%, −0.9% | ΔSharpe −0.05, ΔDD −0.8pp |

**The plateau that looked real on inner-train does not survive
inner-validation.** Every cell in the 10-config grid is, on the split
that selection is actually made on, statistically indistinguishable from
`kelly_regime_v4` unmodified — differences of a few tenths of a Sharpe
point and about one drawdown percentage point, both comfortably inside
the ±0.2 Sharpe / low-single-digit-pp noise floor this project uses
(R-20). No cell beats v4 by a margin worth reporting as a finding; the
best futures cell (k=2) is flat-to-slightly-worse on both Sharpe and
return, with a drawdown difference (−0.8pp) an order of magnitude smaller
than the −7 to −11pp seen on inner-train. This is the in-sample-only
pattern ROUTINE.md names by number (R-12: "28 of 32 in-sample winners; 0
of 28 out-of-sample").

---

## Pre-registered falsification: does it survive ETH?

Following R-17's established substitution (the Coinbase ETH file this
round's pre-registration names does not exist in this isolated worktree;
Bitfinex BTC+ETH, same venue, same window, is the same substitution
R-28's `eth()` used for the identical reason), evaluated at the a-priori
cell (k=1.0, conf_floor=0.5), full available 2016–2019 Bitfinex range
(entirely pre-2023, `assert`-checked in code):

| asset | market | v4 Sharpe / DD | novel cell Sharpe / DD | ΔSharpe | ΔDD |
|---|---|---|---|---|---|
| **BTC** (control) | spot | 1.86 / 40.1% | 1.94 / 29.0% | **+0.08** | **−11.1pp** |
| **BTC** (control) | futures 5x | 2.19 / 32.1% | 2.04 / 29.8% | −0.15 | **−2.3pp** |
| **ETH** (test) | spot | 1.48 / 36.5% | 1.24 / 47.2% | −0.24 | **+10.7pp** |
| **ETH** (test) | futures 5x | 1.25 / 35.1% | 1.27 / 48.1% | +0.02 | **+13.0pp** |

**This is the falsification firing.** On BTC, the mechanism cuts
drawdown in both markets (the property this project treats as credible —
ROUTINE.md: drawdown/tail improvement "is the property that actually
replicates"). On ETH, under the **identical construction, identical
STRESS_EPISODES calendar, identical a-priori config**, drawdown gets
**meaningfully worse** in both markets — +10.7pp spot, +13.0pp futures,
larger in magnitude than the BTC improvement it is supposed to mirror.
BTC improves, ETH gets worse, on the same cell: this is exactly the
pre-registered kill condition, and exactly the pattern the pre-
registration names as having killed R-53's and R-73's conservative
branches. The mechanism is not general; it is fit, in a way that does not
show up in KS-A/KS-B, to something specific about *which six dates* land
inside BTC's own price history.

---

## Plateau assessment

Two different, conflicting plateau readings, and both matter:

- **Within inner-train**, the `floor=0.5` region is a genuine plateau
  across `k ∈ {0.5,1,2}` — consistent Sharpe/DD across three points, not
  a tuned singleton. Read on its own, this would pass the "plateau, not
  peak" bar.
- **Across splits and assets**, the whole grid is a **plateau of
  nothing**: every cell collapses to "indistinguishable from v4" on
  inner-validation, and every cell that helps BTC's drawdown hurts ETH's
  by a larger margin. A flat neighborhood around a non-effect is not
  evidence for the mechanism; it is evidence the in-train numbers were
  overfit to which six calendar dates happen to sit inside BTC's own
  cycle, not evidence of a real, portable confidence signal.

---

## A harness bug worth recording (caught before it reached any number above)

An earlier version of this file set `strategy.warmup` to a 100-million-bar
sentinel to give the causal jackknife "full history since inception" as
its `run_period` prefix. That broke a second, undocumented use of the
same attribute: `tradebot/engine.py`'s `run_backtest` gates the call to
`on_bar` itself on `i >= strategy.warmup` (not merely order placement, as
`trade_start` does) — so with a 100-million-bar gate, `on_bar` was never
called, on any configuration, including the k=0 identity arm. The result
was **0 trades and an unchanged \$1,000.00 balance on every one of the 40
planned evaluations** — a result that reads as a valid (if flat) run
rather than as a broken harness, which is exactly the failure mode R-21
and R-24 are in this project's memory for ("$3.7e23 with a fully green
suite", "an audit built exactly that strategy"). It never touched a
single holdout bar (it manifested entirely pre-2023) and was caught
before any figure in this report was produced, but it is recorded here
per that same convention: a flat-looking result is a bug report first.
**Fix:** leave `warmup` at v4's own value (governs the vote anchors'
burn-in only) and give the jackknife its long history a different way —
`ev()` builds the evaluation frame as every bar of the dataset from its
true start through the period's end, rather than going through
`run_period`'s `warmup`-bars prefix, applied uniformly to every strategy
evaluated in this file (benchmarks included) so the internal comparison
stays fair. One side effect worth flagging explicitly: because of this,
the `kelly_regime_v4` benchmark numbers printed in this file (e.g.
inner-validation spot Sharpe 0.12, \$986) differ slightly in the second
decimal from numbers `scripts/experiment.py`'s `run_period`-based harness
would print for the identical strategy on the identical split (v4's own
180-day volatility-regime EWM is not fully converged after only an
80-day warmup prefix; feeding it years more history changes its state at
the margin). This does not affect any comparison *within* this file — v4
and every novel cell get the identical, longer prefix — but a skeptic
cross-checking these v4 numbers against a different session's figures
should expect a small, explained discrepancy, not a bug.

---

## Causality

`experiments/r101_novel_jackknife_causal.py causality` (the same
two-opposite-tampers procedure `tests/test_causality_strict.py` runs for
registered strategies, applied here by hand since that suite only
parametrizes over registered names): **PASS**. Every decision at or
before the tamper cut is bit-identical whether bars after the cut are
multiplied or divided by 3; `target`, `conf` and `frac` columns show
`0.000e+00` max difference before the cut.

`pytest tests/test_causality_strict.py -q`: **51 passed**, 0 failed.

Full `pytest -q`: **391 passed**, 0 failed, 0 skipped (207.76s).

**Grep audit for `202[3-9]` literals** (required by this round's
pre-registration) — `grep -n "202[3-9]" experiments/r101_novel_jackknife_causal.py`
returns 7 lines: three are prose inside comments (explaining the FTX
episode's window close date, the paper-trading-appended file, and why
`DF.index[-1]` is never printed), and four are `assert ... < "2023-01-01"`
/ `assert end is not None and str(end) < "2023-01-01"` refusal guards.
**Zero** occurrences are used to read, slice, or index into a dataset —
every literal is either prose or an exclusive-upper-bound refusal check.

**One disclosure, in the spirit of the project's own honesty norm:**
during interactive development the terminal banner
`load_dataset`/`scripts/experiment.py`-style scripts print
(`{len(DF):,} bars {DF.index[0]} -> {DF.index[-1]}`) surfaced the
committed file's literal last-bar **date** (2026-08-12, since the paper-
trading recorder keeps appending to the same committed CSV) several times
in early tool output, before this was noticed and the file's own banner
was rewritten to report only the pre-holdout portion's range. No bar's
**price/volume content** dated 2023+ was ever read, computed on, or
included in any figure in this report or in the strategy's own logic —
every `ev()`/backtest/prepare() call in this file is bounded by an
explicit `end <= 2022-12-31` — but the bare timestamp of the file's last
row was visible in scrollback, and that is recorded here rather than
left unstated.

---

## Verdict

**`further_work = False`.**

Both pre-registered kill switches passed (KS-A CV(conf)=0.2664≥0.05,
KS-B R²=0.8588<0.95) — the causal-expanding-jackknife `conf` signal is
real, time-varying, and not a disguised flat rescale of v4's exposure.
That much of this branch's premise is correct. But the mechanism fails on
both remaining pre-registered bars from ROUTINE.md's own promotion
criteria:

1. **No improvement on the split selection is made on.** Every one of
   the 10×2×2=40 evaluated configurations is statistically
   indistinguishable from `kelly_regime_v4` unmodified on
   inner-validation — the in-train drawdown improvement (−7 to −11pp on
   futures) does not survive to inner-validation (best cell: −0.8pp),
   the textbook in-sample-only pattern this project has a numbered
   ledger row for (R-12).
2. **Fails its own pre-registered falsification test.** BTC's drawdown
   improves under the mechanism; ETH's drawdown gets meaningfully worse
   under the identical construction (+10.7 to +13.0pp) — an
   asset-specific sign flip on the one property (drawdown) this project
   otherwise trusts, matching the exact pattern pre-registered as
   disqualifying.

Per this round's own pre-registration, this branch's further-work bar
requires KS-A, KS-B, ETH falsification, *and* a plausible
noise-floor-exceeding improvement, all four. Two of four hold; two do
not. This is a clean **NEGATIVE**, and — per ROUTINE.md — "a well-
documented negative result is a successful day."

### What this adds to the standing diagnosis

The N≈3 constraint now has one more concrete measurement attached to it:
even a mechanism explicitly built to *quantify* "how much does the vote's
edge depend on any single stress episode" (CV(conf)=0.27, genuinely
dispersed, not a rescale) still cannot turn that quantification into
return or generalizable risk edge, because the quantity being jackknifed
is itself a function of *which six calendar dates* happen to fall inside
one asset's particular price history — precisely the fragility N≈3
describes, one level up. Measuring the fragility does not cure it.

### Next step, if any

Not recommended as a `NEXT` backlog item in its current form. If revisited,
the falsification result suggests the episode calendar itself
(`STRESS_EPISODES`) may be too BTC-specific to serve as a cross-asset
confidence anchor; an asset-relative or volatility-relative event
definition (rather than fixed calendar dates) would be a different
mechanism, not a parameter tweak of this one.
