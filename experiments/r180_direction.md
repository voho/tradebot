# R-180 pre-registration (08-29) — exogenous-feature meta-labeling on `kelly_regime_v4`, conservative fixed-window vs. novel anytime-valid (testing-by-betting) confidence

## Step 0

`git rev-parse HEAD origin/main` are equal (`3879b42`); no undispatched
`r<nn>_shared.py` (newest is `r179_shared.py`, recorded as R-179). Step 0b
consecutive-null-pass count is **0** (R-179 was a dispatched round, not a
null pass — resets the counter) — squarely "0–2: normal," a fresh
literature-sweep round is in scope. Backlog: B-06 (blocked/de-ranked, N≈3
horizon problem, R-78/R-83), B-09 (LOW), B-17 (PARTIAL, blocked on a
strategy candidate that doesn't exist yet), B-28 (HALF-CLOSED, blocked on
data this repo cannot fetch), B-48 (OPEN but a documentation/formatting
instrument fix, not a strategy-research item). No live, unblocked
strategy-research backlog item — confirmed by re-reading the table, not
just the re-ranking prose — so a new off-backlog direction is warranted.

## Step 1 — the four questions

**Idea, one sentence.** Re-run meta-labeling (López de Prado 2018) on
`kelly_regime_v4`'s `frac*scale` decision — the mechanism R-170 and R-179
both tried and both closed NEGATIVE — but for the first time use a feature
family that is genuinely exogenous to price and to the vote (macroeconomic
stress and on-chain valuation, never used as ERR-axis reliability features
before), and, on the novel branch, replace R-179's fixed-window
walk-forward classifier with a sequential, anytime-valid "testing by
betting" confidence process (Shafer 2021; Waudby-Smith & Ramdas 2024) so
the gate's trust in the vote updates continuously instead of on an
arbitrarily-chosen refit calendar.

1. **Which constraint does it attack, and why this is not "another
   indicator"?** **ERR** — "no error control anywhere in the signal path."
   This is not a new *directional* signal competing with the vote (that
   would be INFO, and both candidate feature sources have already been
   tried and killed in that role: VIX/DXY macro stress as a directional
   veto/vote failed R-53/R-54/B-21, and MVRV level/rate-of-change as a
   directional confirming vote failed R-74). Here neither feature is asked
   "which way should the position move" — both are asked "how much should
   I trust the position the vote has *already* chosen," which is a
   downstream reliability question, not a prediction of direction. That is
   the same distinction R-179 itself drew to justify meta-labeling as ERR
   rather than INFO, and it is why a signal that failed as a directional
   input can still be a legitimate meta-labeling feature: R-53/R-54's own
   finding was that macro stress *lags* the price-anchor gate by
   ~5.5 days on average — a bad property for a leading indicator, and an
   entirely irrelevant one for a feature that only has to describe the
   *ambient conditions* under which the vote is currently operating, not
   predict when the vote should flip.

2. **Which ledger entries is it not a duplicate of?**
   - **Not R-170** (funding z-score, DVOL-minus-realized VRP, Amihud
     illiquidity, R-163's episode-excursion — four features, all either a
     derivatives-market cost/positioning proxy or a strategy-own-path
     statistic) and **not R-179** (`vol_ratio`, `vote_strength`,
     `log1p(regime_duration)` — three features, all literal transforms of
     `kelly_regime_v4`'s own anchors/vote/scale). This round's two features
     — a macro risk-off stress z-score (VIX level + DXY level, `data/
     vix_daily.csv.gz`, `data/dxy_daily.csv.gz`) and an on-chain valuation
     z-score (MVRV, `data/btc_mvrv_daily.csv.gz`) — are neither. They are
     the specific feature source this project's own closing lines on both
     prior rounds asked for: R-179's verdict named the requirement
     explicitly ("a feature source that is simultaneously (a) available
     from this project's committed data, (b) structurally independent of
     `kelly_regime_v4`'s own price anchors and volatility estimator, and
     (c) not already one of R-170's four rejected channels") and this
     round adds the same disjointness from R-179's own three. All three
     conditions are checked below.
     - **(a) Available in committed data**: `data/vix_daily.csv.gz`
       (2016-06-01→), `data/dxy_daily.csv.gz` (2016-06-01→), `data/
       btc_mvrv_daily.csv.gz` (2016-01-01→) — all three already fetched
       and used by R-53/R-54 (VIX/DXY) and R-74 (MVRV). Unlike R-170's
       DVOL (which only starts 2021-03-24, inside inner-validation, and
       forced a disclosed inner-train-skipping workaround), **all three
       series here cover the entire 2017-01-01 inner-train start with
       margin**, so this round needs no such workaround.
     - **(b) Structurally independent of price/vote anchors and the vol
       estimator**: VIX and DXY are US equity-vol and dollar-index levels,
       computed from entirely separate markets; MVRV is a blockchain
       cost-basis statistic (aggregate realized price paid by current
       coin holders vs. spot price) computed from on-chain UTXO movement,
       not from `kelly_regime_v4`'s rolling-mean anchors or its EWM
       volatility estimator. None of the three is a transform of BTCUSD
       OHLCV.
     - **(c) Not R-170's four or R-179's three**: confirmed by inspection
       — {funding, VRP, Amihud illiquidity, episode-excursion} ∩ {macro
       stress, MVRV} = ∅; {vol_ratio, vote_strength, regime_duration} ∩
       {macro stress, MVRV} = ∅.
   - **Not R-53/R-54/B-21** (VIX/DXY as a hard veto or a precision-weighted
     directional vote — an INFO-axis lead-time claim, tested and failed at
     4/12 episodes leading) or **R-74** (MVRV level/rate-of-change as a
     directional confirming vote, also INFO-axis, also failed on lead
     time): both used the same raw series to answer "which way," this
     round asks "how much to trust," on a classifier fit against realized
     triple-barrier outcomes rather than against forward direction.
   - **Not R-87** (Adaptive Conformal Inference on the vote's own
     confidence, or on the Kelly-scale dispersion estimator — an
     *unsupervised*, single continuously-updating conformal quantile, no
     labeled outcome, no exogenous feature) and **not R-161/R-167** (RCPS/
     CRC/anytime-valid concentration bounds calibrating a *cap* on SCALE's
     magnitude from SCALE's own historical values — again unsupervised,
     no external feature, no labeled bet outcome).
   - **Not R-174** (Wald SPRT / GROW e-value sequential gate on `frac*
     scale`'s own re-sizing timing, anchored to BTC's raw per-bar drift
     and volatility — closed because that ratio's natural resolution
     timescale is 2–3 orders of magnitude longer than a deadband episode's
     own lifetime). This round's novel branch also uses a sequential
     e-value/betting construction, but on a **structurally different
     statistic**: not BTC's per-bar drift (~8e-6 mean, ~1e-3 std, the
     quantity R-174 showed cannot resolve inside an episode), but a
     **daily-checkpoint Bernoulli-like triple-barrier hit/miss outcome**
     (base rate near 50–55%, R-179's own daily-labeling design), which has
     a coarser, days-to-weeks natural resolution horizon by construction —
     Step 1 Q4 below computes this explicitly rather than assuming it.
   - **Not any prior ERR-axis round on this architecture** (R-28/R-31,
     R-87 ×2, R-104, R-105 ×2, R-114 ×2, R-147 ×2, R-160 ×2, R-172 ×2,
     R-174 ×2, R-170, R-179 — 0 of 17 promoted): every one of those either
     used an unsupervised statistic (significance, disagreement, novelty,
     duration, shrinkage, online-FDR, FCR/PoSI, RCPS/CRC, ACI, sequential
     e-value on raw drift) or a supervised classifier fed price/vote or
     derivatives-market/strategy-own-path features. This round is the
     first supervised classifier fed macro/on-chain features, and the
     first to pair a supervised classifier with a sequential/anytime-valid
     confidence process rather than a fixed-window refit-and-predict loop
     — a combination a web search for "anytime-valid e-value meta-labeling
     trading" (08-29, this round) found no existing literature combining
     directly, so this is a genuine methodological cross of two techniques
     this ledger has each used separately (meta-labeling: R-04/R-80/R-170/
     R-179; testing-by-betting/anytime-valid: R-71/R-83/R-167/R-174), not
     a re-run of either.

3. **Is it simulable here?** Yes. VIX, DXY and MVRV are daily series
   already committed to `data/`; no order book, no new fetch. Both
   features are forward-filled onto the 5m bar grid using only data
   available at or before the bar's own timestamp (the daily print for day
   `d` becomes available and usable starting the bar closest to day `d`'s
   own close, mirroring R-53/R-54/R-74's own causal join convention).
   Labels are causal daily triple-barrier outcomes exactly as
   `r179_shared.py`'s `daily_triple_barrier_labels` already builds them
   (reused, not reimplemented, per Step 0's collision-avoidance
   convention — this round's own shared module imports it rather than
   duplicating it). The novel branch's sequential e-value accumulator
   updates once per bar from already-resolved labels only (no label is
   read before its own horizon + embargo has elapsed) — same causal
   discipline R-179's `walk_forward_meta_prob` already used and passed.
   Signal is decided at bar close, orders fill at next bar open, matching
   every registered strategy in this repo.

4. **What would make it fail — named now, with the implied sample size
   checked against this project's own measured noise, before any code
   beyond the shared scaffold?**

   - **Step-A discriminative-skill gate (both branches, reused from R-170/
     R-179's own convention): does the classifier/accumulator carry any
     real signal at all, before a single backtest number is read?**
     Fixed logistic classifier (conservative): AUC vs. a 1,000-draw
     label-permutation null, on 2017–2020 training-period data only. Must
     clear the null's 95th percentile on a majority of walk-forward
     refits, or the branch is recorded NEGATIVE/inconclusive by
     construction (R-179's clause A, reused verbatim). Sequential
     accumulator (novel): the terminal e-value/wealth at the end of the
     training period must exceed `1/alpha` at `alpha=0.05` (Ville's
     inequality threshold, the standard testing-by-betting rejection
     rule), computed on training-period data only, before touching
     inner-validation.

   - **The power check this project's R-78 finding requires, computed
     rather than assumed.** R-179's own two feature-family results give a
     real prior for what a "real but not tradeable" ERR-axis signal looks
     like on this architecture: its best risk-matched Δlog-growth point
     estimates were spot +0.094 / futures +0.133 (conservative) and spot
     +0.106 / futures +0.118 (novel), against paired block-bootstrap 95%
     CIs on the novel branch of **[−0.129, +0.415]** (spot) and
     **[−0.099, +0.379]** (futures) — a half-width of roughly **±0.25–0.27
     log units** over the 2-year inner-validation window, at n=2,000
     bootstrap resamples with a 30-day block. Assume this round's exogenous
     features produce a comparably-sized effect (there is no reason to
     expect a materially larger one — the features are weaker directional
     predictors than v4's own anchors, per R-53/R-54/R-74's own INFO-axis
     failures, so if anything a smaller true effect is the more likely
     prior). To exclude zero at that same point estimate (~0.10 log units)
     the CI half-width would need to shrink to ~0.10, i.e. by a factor of
     ~2.5–2.7×. Since bootstrap CI width scales as `1/sqrt(n_independent)`
     for a fixed noise process, that requires **~(2.6)² ≈ 6.8× the
     independent evidence** — roughly **6.8 × 2 years ≈ 13.6 years** of
     inner-validation-equivalent data. This project's full pre-holdout
     history (2017–2022, inner-train + inner-validation) is 6 years; the
     holdout (2023–2026) adds 3.6 more; neither individually nor combined
     reaches 13.6 years. **This is named now, before any code, as the
     expected outcome of the Sharpe/log-growth promotion-bar clause**: at
     the effect size this architecture has actually produced twice before
     (R-179), the CI is very unlikely to exclude zero on this dataset,
     for the same structural reason R-78 found the B-06 forward horizon
     unresolvable — the ratio of plausible effect to measured noise is too
     small for the data actually available, not a defect of this round's
     test. **Falsification is therefore pre-registered at the Step-A gate
     as the decisive, well-powered test** (it needs O(hundreds) of daily
     checkpoints and R-179 already showed that count is reached — see
     Result — so it is NOT a repeat of R-78's mistake of pre-registering
     an unreachable bar): if Step-A fails, the branch is NEGATIVE outright
     and the underpowered Sharpe-CI clause is reported for completeness
     only, exactly as ROUTINE.md's promotion bar still requires, but
     without treating a CI-includes-zero result on that clause as new
     information beyond what the Step-A gate already decided.

   - **The novel branch's own resolution-time check (parallel to R-174's,
     computed for THIS statistic rather than reused from R-174's).** The
     accumulator bets on a Bernoulli-like daily label with assumed base
     rate `p0=0.50` (null: features carry no information, meta-label
     right at chance net of triple-barrier drift) against an alternative
     edge `Δp`. Using a Kelly-style log-optimal betting fraction, the
     expected per-checkpoint growth rate of the e-value under a true edge
     `Δp` is approximately `Δp²/(2·p0·(1-p0)) ≈ 2·Δp²` bits-equivalent
     nats per checkpoint (binary KL-divergence, small-`Δp` approximation).
     To reach the `ln(1/alpha) = ln(20) ≈ 3.0` threshold at `alpha=0.05`
     requires **n ≈ 3.0 / (2·Δp²)** resolved checkpoints. At `Δp=0.03`
     (a 3-point hit-rate edge, in line with the magnitude R-170's own
     single-feature AUC deltas implied), that is **n ≈ 1,670** resolved
     checkpoints; at `Δp=0.05`, **n ≈ 600**. Against R-179's own measured
     yield (median `n_at_refit` 743–760 *per refit*, with dozens of refits
     across the training period — i.e. the training period alone produces
     several thousand label-resolution events in total, an order of
     magnitude more than either bound), **this is reachable inside the
     2017–2020 training period alone** — unlike R-174's raw-drift
     accumulator, which needed 10⁴–10⁵ *bars* of one-sided evidence against
     an episode lifetime of hundreds of bars. **Falsification clause
     (novel, specific):** if the terminal training-period e-value does not
     cross `1/alpha` at `alpha=0.05` despite this being a reachable n, the
     branch is NEGATIVE and the failure is attributed to the feature
     carrying no edge, not to an unresolvable test — a real result, unlike
     R-174's inconclusive-by-construction one.

   - **Unmatched risk (R-33).** As in every round on this axis: if either
     branch's realized volatility or average notional differs materially
     from `kelly_regime_v4`-alone's, the comparison is voided per this
     ledger's standing rule, not scored as a win. R-179's own novel branch
     failed exactly this way (19–23% hot realized vol) — this round's
     gates check it explicitly on every surviving config before any Sharpe
     number is quoted.

## Step 2 — design

**Shared, read-only engine (`r180_shared.py`, neither branch may edit):**
imports `daily_triple_barrier_labels` from `r179_shared.py` unedited (per
Step 0's collision-avoidance convention — do not re-derive an already-
validated causal labeling function); adds two new causal feature builders,
`macro_stress_z` (an equal-weight z-score of VIX level and DXY level, each
z-scored against its own trailing 365-day causal window, summed) and
`mvrv_z` (MVRV's own value, z-scored against its trailing 365-day causal
window — deliberately not a rate-of-change, since R-74 already tested and
killed MVRV's rate-of-change as a directional signal; a level z-score
answers a different, ERR-axis question: "is the current on-chain valuation
regime unusual," not "which way is it moving"); both forward-filled from
each series' own daily print onto the 5m grid with no future information;
a Step-0(i) identity check (feature builders reproduce `v4` exactly when
the gate/multiplier degenerates to a no-op) and Step-0(iii) label-
permutation-null gate, reusing R-80's validated design, exactly as R-170/
R-179 required before either branch was dispatched.

**Mechanism, conservative branch — literal reuse of R-179's own validated
architecture, feature swap only.** Same walk-forward fixed-window logistic
classifier, same purge/embargo, same binary bet/no-bet veto gate structure
as R-179's conservative branch — the only change is the two input features
(`macro_stress_z`, `mvrv_z` in place of `vol_ratio`, `vote_strength`,
`log1p(regime_duration)`). This is the minimal, safest possible variant: it
changes exactly one thing (the feature source) and reuses every other
design decision R-179 already froze and validated, so any difference in
outcome is attributable to the feature family, not to a different
architecture.

**Mechanism, novel branch — sequential testing-by-betting confidence
process (Shafer 2021, *Testing by Betting: A Strategy for Statistical and
Scientific Communication*, JRSS-A 184(2); Waudby-Smith & Ramdas 2024,
*Estimating Means of Bounded Random Variables by Betting*, JRSS-B 86(1)).**
Instead of a fixed-window refit-then-predict classifier, run a single
continuously-updating capital process: start `wealth=1`; at each newly-
resolved daily triple-barrier checkpoint, bet a Kelly-fraction of current
wealth on the label's realized outcome using the current best estimate of
`P(label=1 | macro_stress_z, mvrv_z)` from a simple online logistic
updater (stochastic gradient, no batch refit, no arbitrarily-chosen
`refit_days` window — the parameter R-179's own Step-A gate showed
mattered materially, e.g. its winning novel corner needed `refit_days=90`
specifically); use `log(wealth_t)` directly as a continuous, anytime-valid
confidence score, and multiply `v4`'s desired exposure by a bounded
function of it: `multiplier = clip(1 + kappa * tanh(log(wealth_t) / 3.0),
0, cap)` (the `/3.0` normalizes so the multiplier saturates near
`ln(1/alpha)` at `alpha=0.05`, i.e. exactly the point the accumulator
itself calls "significant"). This removes R-179's own named failure
mode (the classifier's confidence tracking a single arbitrarily-tuned
refit calendar) and gives the confidence score a formal, sequential,
anytime-valid interpretation instead of a walk-forward point estimate —
the creative, outside-vanilla-trading mechanism this step asks for,
borrowing the same game-theoretic betting framework this ledger has
already used for continuous risk control (R-71/R-83/R-167) but applying it
for the first time to a *supervised* classifier's own confidence rather
than to an unsupervised statistic.

**Falsification tests:** as named in Step 1 Q4, frozen before either
branch is implemented — Step-A discriminative-skill/e-value-threshold gate
first (the well-powered, decisive test), Sharpe/log-growth promotion-bar
clause reported for completeness with its own power limitation disclosed
in advance, R-33 risk-matching checked on every surviving config.

Configs evaluated by this file: 0 (shared infrastructure only, per R-163/
R-168/R-178/R-179's convention — each branch counts and reports its own).

## Citations

- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  Wiley, ch. 3 — meta-labeling / triple-barrier framework (base
  architecture, reused from R-170/R-179, not re-derived).
- Shafer, G. (2021). "Testing by Betting: A Strategy for Statistical and
  Scientific Communication." *Journal of the Royal Statistical Society,
  Series A*, 184(2), 407–431 — the testing-by-betting principle: evidence
  against a null is the wealth of a fictitious bettor wagering against it
  at fair odds. Borrowed for the novel branch's continuous confidence
  process, in place of a fixed-window refit-and-predict classifier.
- Waudby-Smith, I. & Ramdas, A. (2024). "Estimating Means of Bounded
  Random Variables by Betting." *Journal of the Royal Statistical Society,
  Series B*, 86(1), 1–27 — the specific betting-based capital-process
  construction and its anytime-valid guarantee; this project has already
  used this paper's machinery for a continuous risk control (R-167), this
  round is the first to apply it to a supervised classifier's own
  confidence.
- Grünwald, P., de Heide, R. & Koolen, W. (2024). "Safe Testing." *JRSS-B*
  86(5) — cited for contrast: R-174 already tried this family's GROW
  e-value on `frac*scale`'s raw per-bar timing and found the natural
  resolution timescale unreachable; Step 1 Q4 above computes why this
  round's daily-checkpoint statistic does not inherit that failure.
- Bongaerts, D., Kang, X. & van Dijk, M. (2020). *Financial Analysts
  Journal* 76(4) — unchanged, `kelly_regime_v4`'s own SCALE-factor
  citation, not part of this round's own contribution, listed for
  completeness since the strategy under test is unmodified except for the
  gate/multiplier.
- Mahmudov, M. & Puell, D. (2018), MVRV Z-Score (the on-chain valuation
  construction itself; industry-standard, no peer-reviewed venue —
  disclosed as such). Used here only as a feature source, not re-litigated
  as a directional signal (that is R-74's already-closed finding).
