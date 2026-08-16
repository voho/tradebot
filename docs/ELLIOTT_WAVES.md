# Elliott waves: assessed against this repo's own bar

Asked whether Elliott Wave Theory (EWT), possibly combined with game
theory or neural networks, could improve results here. Short answer: **no,
and the reason is methodological rather than a matter of taste.** The
defensible part of EWT is already implemented in this repo in falsifiable
form; the rest cannot be tested by the machinery that everything else
here had to pass.

## The blocking problem: it is not falsifiable as practitioners use it

This project's causality guarantees rest on strategies being
**deterministic functions of past bars**. That is what
`tests/test_causality_strict.py` checks — the orders a strategy queues
must be byte-identical under two opposite tamperings of the future.

EWT wave counts are not that. Counts are subjective, disputed between
analysts on identical data, and routinely **re-labelled after the fact**
when price invalidates them. Aronson characterises it as a story prone to
subjective revision; the standard critique is precisely that a rule which
can be re-counted retrospectively cannot be falsified. A wave count
assigned "as of" a date by a human — or by an LLM that has read the
subsequent price action — is the exact leak class this repo built the
strict causality suite to catch, and it is worth remembering what that
class is worth: the probe strategy that peeked one bar ahead returned
$3.7e23 from $1,000 at Sharpe 73 with a fully green test suite.

To be testable here, EWT would have to be reduced to a **mechanical wave
counter**: fixed swing detection, fixed impulse/corrective rules, no
discretion, no revision. At that point it is a pattern-matching indicator
over zigzag pivots — and this repo has already established what happens
to pattern indicators on 5m BTC after fees.

## Its one testable component has already been tested and failed

The Fibonacci retracement ratios are the part of EWT that makes a
quantitative claim. Batchelor & Ramyar tested them and found no
significant evidence of Fibonacci ratios in price data. That is a direct
refutation of the component most easily bolted onto a strategy.

## The 2024 neural-network result does not meet this repo's bar

The current state of the art is *ElliottAgents* (Applied Sciences 14(24),
Dec 2024): a multi-agent LLM system with deep reinforcement learning that
reports **73.68% accuracy with backtesting against 57.89% without**, on
BTC/USD from October 2022 to September 2024.

Held to the standards this repo enforces on itself:

- **The sample is 19 cases.** 57.89% and 73.68% are 11/19 and 14/19. The
  headline improvement is **three additional correct calls.**
- **The window is ~2 years of a monotonic rise**, $20K → $70K. Every
  strategy in this suite looks good on that stretch; `buy_and_hold`
  returned +1,418% on 5x futures out-of-sample for exactly this reason.
- **Accuracy is the wrong metric, and this project has the receipts.**
  Several strategies here have respectable hit rates and still lose money
  — `overshoot_fade` has a good win rate and bad tails; `minority_oracle`
  is an honest negative result at 9,039 trades. Directional accuracy
  without a fee-aware, turnover-aware equity curve says nothing.
- **No walk-forward.** The repo's own fee study found 28 of 32
  configurations beating buy-and-hold in-sample and **0 of 28**
  out-of-sample. A result with no out-of-sample split is not evidence
  against that base rate.
- **Noise floor.** The measured ±0.2 Sharpe floor on this data means a
  single 2-year path cannot resolve differences far larger than the one
  claimed.

None of this proves the system does not work. It means the evidence
offered is several orders of magnitude weaker than what this repo demands
of its own strategies before registering them.

## Game theory: there is no theorem to borrow

EWT has no players, no payoffs, no equilibrium concept and no solution
method. It is a descriptive taxonomy of shapes with a psychological
narrative attached. The rigorous relatives of that narrative are already
implemented here:

| EWT's informal claim | the falsifiable version, already in this repo |
|---|---|
| crowd psychology drives price in waves | Cardaliaguet & Lehalle (2018) mean-field game of trade crowding — the grounding of `kelly_regime`, the best strategy here |
| markets are self-similar across degrees | Müller et al. (1997) heterogeneous market hypothesis / Corsi (2009) HAR — the doubling anchor ladder in `kelly_regime_v4` |
| participants act on each other's expectations | minority games, Challet & Zhang (1997) — implemented as `minority_oracle`, a documented negative result |

So the useful kernel of EWT — **multi-timescale structure produced by
crowd behaviour** — is not novel and is not missing. It is the explicit
foundation of the leading strategy, in a form that can be tested, and it
has been.

## Neural networks: the label adds nothing and imports a leak

A network trained to recognise Elliott patterns is a network trained on
price shape. The wave labels are a *human prior about which shapes
matter*, and the network can learn shape directly from the same data. The
labels contribute no information the input lacks; what they do contribute
is a subjective, hindsight-contaminated annotation step.

This repo already ran a deep-learning round and adopted none of it (see
[RESEARCH.md](RESEARCH.md#deep-learning-why-none-was-adopted)). The
binding constraint was never model capacity — it is that with ~3
independent regime events, after fees, on one asset, there is not enough
signal to fit anything that survives out-of-sample. A larger model does
not manufacture independent observations.

## If it is pursued anyway, the honest way to do it

In keeping with this repo's "nothing is deleted" convention, a negative
result is worth having:

1. Implement a **deterministic** wave counter — ZigZag pivots at a fixed
   threshold, then Elliott's impulse/corrective rules applied
   mechanically, no discretion and no revision of past counts.
2. Register it and let the standard suite judge it: strict causality,
   truncation, walk-forward split, Monte Carlo windows, the ±0.2 Sharpe
   noise floor, and fees.
3. Publish whatever comes out, including "it lost", exactly as
   `minority_oracle` and `game_switch` are published.

That is perhaps a day of work and it converts an unfalsifiable debate
into a row in the comparison table. What it will not do is produce an
edge that the twenty existing predictors could not.

## Recommendation

Do not pursue it ahead of the two directions in
[ALTERNATIVES.md](ALTERNATIVES.md). Extending the funding series to test
whether the carry premium survived 2024–2025 is a data fetch that decides
a real question; a second bear on a second asset closes the sample-size
gap in [CROSS_ASSET.md](CROSS_ASSET.md). Both are cheaper than a wave
counter and both can change a conclusion. Elliott waves, tested properly,
would most likely add a twenty-fourth documented way to lose to fees.
