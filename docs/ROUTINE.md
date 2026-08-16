# The daily research routine

The automation prompt for this repo is one line:

> Read `docs/ROUTINE.md` and follow it.

Everything below is that routine. It exists as a file rather than as a
prompt so it can be revised when the evidence changes, and so every
session runs the same procedure.

The routine's job is **not** to produce a strategy every day. It is to
convert one idea per session into a permanent, honest row in
[LEDGER.md](LEDGER.md) — promoted or rejected. A well-documented negative
result is a successful day. This repo's most valuable output so far is a
list of things that do not work.

---

## Step 0 — Load the memory

Read, in this order:

1. `README.md` — the comparison table and the two standing warnings
   (fees, funding).
2. [`docs/LEDGER.md`](LEDGER.md) — everything already tried, and the
   ranked backlog at the bottom.
3. [`docs/FRONTIER.md`](FRONTIER.md) — the four binding constraints.

**Backlog first.** If the backlog has an item marked `NEXT` or `OPEN`
that is not blocked, work that. Invent a new direction only when the
backlog is empty, fully blocked, or stale. Daily novelty-seeking is a
treadmill; the backlog is the actual research plan.

**Multi-day threads are expected.** If today's work does not finish,
write the state into the ledger with verdict `PARKED` and stop. Do not
force a shippable strategy into a single session.

---

## Step 1 — Select, and justify the selection

State the idea in one sentence, then answer all four. If any answer
fails, discard the idea and pick another — this is a filter, not a
formality.

1. **Which constraint does it attack?** One of the four in
   `FRONTIER.md`: information (one price series), effective sample size
   (n≈3), no error control in the signal path, or costs that scale with
   the signal. *"Another indicator" attacks none of them* — that is why
   the bottom of the comparison table looks the way it does.
2. **Which ledger rows is it not a duplicate of?** Cite them by ID. If
   it is a variant of something already tried, say what is different and
   why that difference should matter.
3. **Is it simulable here?** 5m OHLCV bars, bar-close signals, next-open
   fills, no order book, no queue model. If not, today's job is to build
   the missing simulation capability and record *that* — do **not**
   proxy it out of OHLCV. `camouflage_flow`, `stealth_trend` and
   `flow_regime` are what that costs (L-05, L-06, L-11).
4. **What would make it fail?** Name the outcome now, before any code.

Recency is not a criterion. Prefer recent work, but testability beats
novelty every time.

---

## Step 2 — Understand and design

Gather sources and cite them properly (author, year, venue). Record what
the original paper claims, **on what data, at what cost assumption, and
across how many instruments** — that last one killed the deep-learning
round (R-05: the published edge came from diversifying across 88–100
instruments at 2–3bps, against our one instrument at 10bps).

Then design 1–5 variants. Be creative — the interesting ideas in this
repo came from outside trading (game theory, e-values, evolutionary
dynamics). For each variant write down, **before running anything**:

- the **mechanism** in one sentence: why this should make money;
- **one pre-registered falsification test**, chosen now, with the
  outcome that kills it. Pick from: does it survive on ETH
  (`scripts/build_bitfinex_dataset.py`), does it survive funding
  (`scripts/funding_study.py`), does it survive a 0.40% taker
  (`scripts/fee_study.py`), does it survive the Monte Carlo windows
  (`scripts/stress_test.py`).

---

## Step 3 — Implement and tune, on the training period only

All development, parameter sweeps and iteration use data **before
`OOS_START = 2023-01-01`**. Do not look at the holdout during this step.

```python
from scripts.experiment import ev, OOS_START
ev(MyStrategy(param=x), end="2022-12-31")      # train
```

- Use `--max-bars` and the `experiment.py` harness for the iteration
  loop; a full `tradebot run` is for the final registered set only.
- Keep experiments in `scripts/experiment.py` or `experiments/`, not by
  mutating a registered strategy's defaults — the comparison table is a
  stable record.
- **Count every configuration you evaluate.** That number goes in the
  ledger and into the deflated-Sharpe calculation. The fee study ran 32.
- `pytest` must pass, including `test_causality_strict.py`. A result
  that looks too good is a bug report first: a one-day lookahead is
  worth +2.1 Sharpe, and an `i + 1` peek inside `on_bar` returned
  $3.7e23 with a fully green suite (R-24).

---

## Step 4 — Evaluate once

Freeze the configuration. Then run the holdout **a single time**:

```python
ev(MyStrategy(frozen), start=OOS_START)        # one shot
```

Report all of:

| check | how |
|---|---|
| out-of-sample vs `buy_and_hold` | `ev(..., start=OOS_START)` on both markets |
| real fee tier, not just 0.10% | `scripts/fee_study.py` |
| futures **with funding charged** | `scripts/funding_study.py` |
| the pre-registered falsification test | chosen in step 2 |
| path sensitivity | `scripts/beta_test.py --windows 24` / `stress_test.py` |
| trials-adjusted significance | deflated Sharpe using the step-3 count |

**If you go back to step 3 after seeing any of this, the holdout is
burned.** Say so explicitly in the ledger row and downgrade the result to
in-sample. This is the single rule that separates this routine from what
produced 28-of-32 in-sample winners and 0-of-28 out-of-sample (R-15).

### The promotion bar — default is REJECT

Promote only if **all** hold:

- beats `buy_and_hold` out-of-sample, after real costs (funding charged
  on futures, the venue's actual taker tier on spot);
- the improvement exceeds the **±0.2 Sharpe noise floor** (R-25), or is
  a drawdown/tail improvement, which this repo has repeatedly found to
  be the property that actually replicates;
- survives its pre-registered falsification test;
- the parameter neighbourhood is a **plateau, not a peak** — report the
  neighbours, not just the winner.

Anything else is `NEGATIVE`. Write it up with the same care as a win.

---

## Step 5 — Record

Append one row to [`LEDGER.md`](LEDGER.md) using the template at the
bottom of that file: ID, date, idea, constraint attacked, sources,
variants, **configs evaluated**, train result, holdout result,
falsification outcome, verdict, one-line lesson, next step.

Then, by verdict:

- **PROMOTED** → register the strategy, run the full `tradebot run`,
  refresh the README table, `docs/STRATEGIES.md` and
  `docs/VALIDATION.md`. CI fails if a registered strategy is missing
  from the README table.
- **NEGATIVE** → ledger row plus code under `experiments/` (not
  auto-discovered), **unless** the negative is instructive enough to
  earn a table row the way `minority_oracle` and `game_switch` did.
  Registering every failure inflates the table and slows every future
  run; the ledger is the record now.
- **BLOCKED / PARKED** → ledger row with the blocker named and what
  would unblock it.

Finally, **re-rank the backlog** at the bottom of the ledger, then
commit and push.

---

## Standing rules

- **Nothing is deleted.** Registered negative results stay registered.
- **The table's futures column is an upper bound** until funding is
  charged. Never quote it without that caveat.
- **Never proxy unavailable data out of price.** If the information is
  not in the file, it is not in the strategy.
- **Report ranges, not points**, wherever a bootstrap is available.
- One session, one idea, one ledger row.
