# The daily research routine

The automation prompt for this repo is one line:

> Read `docs/ROUTINE.md` and follow it.

Everything below is that routine. It exists as a file rather than as a
prompt so it can be revised when the evidence changes, and so every
session runs the same procedure.

The routine's job is **not** to produce a strategy every day. It is to
convert one idea per session into a permanent, honest entry in
[LEDGER.md](LEDGER.md) — promoted or rejected. A well-documented negative
result is a successful day. This repo's most valuable output so far is a
list of things that do not work.

---

## Step 0 — Load the memory

**First, before the memory: is a round already in flight?** Sessions can end
between freezing a pre-registration and reporting on it, and what they leave
behind is invisible to every other step here — the ledger has no section for
it, so it reads as if nothing happened. Check for it:

```bash
git fetch --unshallow 2>/dev/null || git fetch origin main   # SEE THE WARNING BELOW
ls experiments/*_shared.py | tail -3        # newest frozen pre-registrations
grep -c "R-<nn>" docs/LEDGER.md             # is that round recorded in section B?
git log --oneline -5                        # a "WIP"/"dispatched" commit is the tell
git rev-parse HEAD origin/main              # equal => nothing in flight elsewhere
```

**Unshallow before you trust that last line.** On a shallow clone
`git merge-base HEAD origin/main` returns **empty, with no error**, against a
genuinely-related ref — so the one step-0 check that would catch a real fork
fails silently exactly when it matters. Pass 23 (section E) read a
fetch race this way and spent the session chasing "44 rounds of unmerged work
off `main`" that did not exist. One command rules it out.

An `r<nn>_shared.py` with no matching section B entry is an **undispatched
frozen pre-registration, and executing it outranks both the backlog and any
new idea.** Its thresholds, splits, decision rule and named failure modes were
fixed by an agent that had seen no number from it, which is the strongest form
of pre-registration this project can produce and cannot be recreated once
today's session has looked at the data. Execute it verbatim, and say in the
entry that you did.

R-131 and R-133 are the worked example, and they are the reason this check is
step 0's first line rather than a footnote: **both** found the same
undispatched file on 08-25, both correctly decided to execute it, and neither
knew the other was doing so until push time — and while the second was being
written up, a *third* session claimed its intended ID for an unrelated
direction, so it landed as R-133. Two ID collisions in one day. The duplicated compute is the
cost of not having this check; the accidental independent replication it
produced — same verdict, same gates, the decisive cell agreeing to three
decimals across two different implementations — is the compensation, and it is
not a substitute for looking first. So: **before executing an in-flight
pre-registration, announce it in the working branch** (a commit touching
`docs/LEDGER.md` with a one-line `IN PROGRESS: R-nn <direction>` stub is
enough) **and re-check `origin/main` before writing the entry.** If another
session has landed the round meanwhile, record yours as a replication under
the next ID, say plainly that it is one, and report only what it adds.

Additions *after* the freeze are allowed in exactly one direction — they may
tighten the bar (an extra ablation, an extra cell, a control), never loosen
it — and must be declared as additions in the entry.

Then read, in this order:

1. `README.md` — the comparison table and the four standing warnings
   (fees, funding, that the table's ordering is mostly noise, and that
   its drawdown column mostly measures holding less).
2. [`docs/LEDGER.md`](LEDGER.md) — everything already tried, the four
   binding constraints (the standing diagnosis at the top), and the
   ranked backlog at the bottom. Section B is the research log: **one
   section per round, newest first**, so the rounds immediately below
   that heading are the most recent work and the ones most likely to
   duplicate today's idea. Read those first, then the backlog.

**Backlog first.** If the backlog has an item marked `NEXT` or `OPEN`
that is not blocked, work that. Invent a new direction only when the
backlog is empty, fully blocked, or stale. Daily novelty-seeking is a
treadmill; the backlog is the actual research plan.

Read the **table** for that, not only the re-ranking prose above it. The
prose is one round's summary and can be wrong: R-77's re-ranking said
B-06 was "the only item remaining on the backlog" while B-32 had been
sitting `OPEN` since R-65. Infrastructure and methodology items count —
they are usually the cheapest things on the list and the only ones that
can actually be finished in a session.

**Do not read that warning, agree with it, and then read the prose
anyway.** Run the grep — a struck row is `~~B-nn~~`, a live one is
`**B-nn**`, so one line prints the whole live list and no re-ranking
paragraph can talk you out of it:

```bash
# every backlog item still live, with its status cell
awk '/^## D\. Backlog/,/^## E\./' docs/LEDGER.md \
  | grep -oE "^\| \*\*B-[0-9]+\*\* \|.*" \
  | awk -F'|' '{printf "%-9s %s\n", $2, substr($5,1,70)}'
```

Then **strike what that prints as already done** (`~~B-nn~~`, the file's
own convention) before reading the list, because a row whose status says
`DONE` while its ID is still bold is a live-looking item that is not one.
R-151 found four of them and R-110 had already had to clean up the same
error once: B-42's row sat bold-but-closed through six rounds'
re-rankings.

R-151 is why this is a command and not a paragraph. Four consecutive
sessions (R-147 through R-150, plus both 08-26 verification passes) wrote
"B-06 is the only ranked, unblocked backlog item" while **B-44** sat
`OPEN` in the table underneath them; two of those sessions dispatched
nothing at all as a result. The grep takes a second, B-44 took one
session to finish, and it produced a measurement that annotated an
earlier round's published table. The prose is a summary of one round's
opinion. The table is the state.

**Multi-day threads are expected.** If today's work does not finish,
write the state into the ledger with verdict `PARKED` and stop. Do not
force a shippable strategy into a single session.

---

## Step 0b — The saturation check: is there anything to do at all?

The routine above assumes each firing meets a backlog with work in it. When
it does not, the instruction *"invent a new direction only when the backlog
is empty"* becomes a standing order to manufacture one — and this brief
fires roughly **hourly**, while research ideas do not arrive hourly. The
result is measured in R-158: **twenty consecutive passes, 08-26 through
08-27, evaluated zero configurations between them** and deposited 1,285
lines of prose saying so. That is not a research program running slowly; it
is a loop.

So before Step 1, count. Section E's table is newest-first and every session
adds a row to it — a null pass adds a numbered one, a session that dispatches
a round adds a `—` row naming that round. So the consecutive-null count is
exactly *the numbered rows above the first `—`*, and it needs no git parsing:

```bash
# consecutive null passes since the last dispatched round
awk '/^## E\. Verification/,/^### E-archive/' docs/LEDGER.md \
  | grep -E "^\|" \
  | awk -F'|' '$2 ~ /^ *#/ || $2 ~ /^ *-+ *$/ {next}
               $2 ~ /—/ {exit} {n++} END {print n+0}'
```

Do **not** count these from `git log` subject lines. Verification-pass commit
subjects cite R-numbers themselves (pass 23's names five), so the obvious
`git log --oneline | grep -m1 "R-1[0-9][0-9]"` matches a null pass and reports
0 every time — it is wrong in the direction that keeps the loop running.

Then apply the rule. It is mechanical on purpose — every one of those twenty
passes reasoned its way to "one more sweep can't hurt," and each was locally
right:

| consecutive null passes | what this session does |
|---|---|
| **0–2** | Normal. Proceed to Step 1; a fresh sweep is reasonable. |
| **3+** | **Do not open a new literature sweep.** The prior passes' searches are in section E — read the table, not the archive. Proceed to Step 1 only with a direction that is *not* a literature search: an infrastructure or methodology item, a measurement that annotates an existing round, or a genuine data channel newly become fetchable. If none exists, append one row to section E and **stop**. |
| **5+** | The above, **plus notify the project owner once** that the firing cadence exceeds the rate at which evidence arrives here, with the pass count and the zero-configuration total. Do not re-argue it every pass afterwards — one notification per escalation, and say in the row that it was sent. |

**Stopping is a valid outcome of this routine, and it is the honest one when
the backlog is exhausted.** A pass that appends a row and stops has run the
routine correctly and completely. The failure mode this rule exists to
prevent is not idleness — it is a session that, finding nothing, writes a
long entry *about* finding nothing and thereby looks productive to the next
session, which then repeats it.

Two things remain worth doing at any pass count, because they are cheap and
they are how a null day still earns its keep:

- **Verify, don't re-search.** Confirm the four live backlog rows against
  their table text, confirm no in-flight pre-registration, confirm B-06's
  recorder is still writing. That is the row's content.
- **Fix the instrument.** If the routine, the ledger format or the harness
  has a defect, that is a legitimate round with an R number — R-158 is one,
  and so are R-151 and B-44. An exhausted backlog is the *best* time for it.

---

## Step 1 — Select, and justify the selection

State the idea in one sentence, then answer all four. If any answer
fails, discard the idea and pick another — this is a filter, not a
formality.

1. **Which constraint does it attack?** One of the four in the
   [`LEDGER.md`](LEDGER.md) standing diagnosis: information (one price
   series), effective sample size
   (n≈3), no error control in the signal path, or costs that scale with
   the signal. *"Another indicator" attacks none of them* — that is why
   the bottom of the comparison table looks the way it does.
2. **Which ledger entries is it not a duplicate of?** Cite them by ID. If
   it is a variant of something already tried, say what is different and
   why that difference should matter.
3. **Is it simulable here?** 5m OHLCV bars, bar-close signals, next-open
   fills, no order book, no queue model. If not, today's job is to build
   the missing simulation capability and record *that* — do **not**
   proxy it out of OHLCV. `camouflage_flow`, `stealth_trend` and
   `flow_regime` are what that costs (L-14, L-15, L-16).
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

**Set every threshold against the comparison's own noise, not against a
number that merely sounds big.** A bar written in annualized percent, or
in "bigger than anything we've measured", is not yet a bar — it becomes
one only when you divide by the standard deviation of the thing being
compared. R-78 pre-registered a power check at +0.001/day on the grounds
that ≈+36.5%/yr is far larger than any effect in this file, and it was;
but the paired difference it had to show up in carries **3.0%/day**, so
that "large" effect reaches t ≈ 1.4 after five years and **no correct
test, sequential or fixed-`n`, could have passed it**. The test failed,
and it was the test's fault. Before freezing any threshold, compute the
`n` it implies at the measured noise and check that `n` is one the
experiment can actually reach. This costs two lines and is the difference
between a falsification test and a formality.

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

### Iterate against an INNER split, never the holdout

Iteration needs feedback, and "don't look at the holdout" leaves none —
which in practice means the holdout gets looked at anyway. So split the
training period and iterate against the inner half:

| slice | dates | use |
|---|---|---|
| **inner-train** | 2017-01-01 → 2020-12-31 | fit, sweep, iterate freely |
| **inner-validation** | 2021-01-01 → 2022-12-31 | select between variants |
| *holdout* | *2023-01-01 →* | *do not touch in this step* |

```python
ev(MyStrategy(param=x), end="2020-12-31")                    # inner-train
ev(MyStrategy(param=x), start="2021-01-01", end="2022-12-31")  # inner-validation
```

Inner-validation contains the 2021 top and the 2022 bear, so it is a
real test with a real regime change in it. Select on it as often as you
like — it is a training resource, and burning it costs nothing that the
holdout has not already been protected from.
- `pytest` must pass, including `test_causality_strict.py`. A result
  that looks too good is a bug report first: a one-day lookahead is
  worth +2.1 Sharpe, and an `i + 1` peek inside `on_bar` returned
  $3.7e23 with a fully green suite (R-21).

---

## Step 4 — Pre-register the decision, then evaluate

Freeze the configuration. **Before running the holdout, write down the
decision rule** — the exact thresholds that will promote or reject —
into the ledger entry. Then run it:

```python
ev(MyStrategy(frozen), start=OOS_START)        # holdout
```

Why pre-registration rather than "look only once": *looking* is not what
corrupts a holdout — **moving the goalposts after looking** is. A single
evaluation with the rule invented afterwards is worth less than three
evaluations against a rule fixed in advance. And a literal
evaluate-once rule is not enforceable across a multi-session program:
this holdout has already been read dozens of times (every OOS number in
the ledger came from it), so treating it as pristine would be a
comfortable fiction. Pre-registration is the discipline that still works
after that has happened.

**Increment the holdout counter** in the ledger entry, and add a bullet
at the **top** of the ledger's `Holdout consultations to date` list —
that list is newest-first too — recording how many times this holdout has
been consulted across the whole project to date and what this round added
to it. That number is the trials count for deflated Sharpe at the program
level, and it only goes up. When it gets large enough that nothing can clear the
deflated bar, the honest conclusion is that this dataset is exhausted
and only forward paper trading (B-06) can settle anything — which is the
argument for starting that recorder now rather than later. **R-78
qualified the second half of that sentence and it should be read with the
qualification attached:** the recorder is running, and on the comparison
it was set up to record it needs 18.9 years to never before its own
anytime-valid tool can fire. Forward evidence is still the only
uncontaminated kind; it is not a queued answer. Which comparison you
record decides whether it ever arrives — see B-38.

Report all of:

| check | how |
|---|---|
| out-of-sample vs `buy_and_hold` | `ev(..., start=OOS_START)` on both markets |
| real fee tier, not just 0.10% | `scripts/fee_study.py` |
| futures **with funding charged** | `scripts/funding_study.py` |
| the pre-registered falsification test | chosen in step 2 |
| path sensitivity | `scripts/beta_test.py --windows 24` / `stress_test.py` |
| trials-adjusted significance | deflated Sharpe using the step-3 count |

**A decision rule must partition the outcome space.** Before freezing it,
check that every possible result maps to exactly one verdict — not that
each verdict has a plausible trigger, which is a different and much
weaker property. R-151 froze three clauses (ADOPT / PARTIAL / REJECT)
whose conditions left a gap, and the result landed in it: the median
effect cleared REJECT's bar comfortably while two of six cells missed
PARTIAL's, so no clause fired. There is exactly one honest response —
**report the fall-through as a fall-through, say the rule was
under-specified, and let the raw table carry the finding.** Reaching for
the nearest-looking label after the fact is the goalpost move wearing a
different hat, and it is the more tempting version of it because nothing
was technically changed. Two lines of thought at freeze time ("what
result would satisfy none of these? what would satisfy two?") is the
whole cost.

**If you change the decision rule after seeing any of this, say so
explicitly in the ledger entry and downgrade the result to in-sample.**
Going back to step 3 to fix a *bug* is fine and always was; going back
to find a threshold that turns a rejection into a promotion is the thing
that produced 28-of-32 in-sample winners and 0-of-28 out-of-sample
(R-12). The difference is whether the target moved.

### The promotion bar — default is REJECT

Promote only if **all** hold:

- beats `buy_and_hold` out-of-sample, after real costs (funding charged
  on futures, the venue's actual taker tier on spot);
- the improvement exceeds the **±0.2 Sharpe noise floor** (R-20), or is
  a drawdown/tail improvement, which this repo has repeatedly found to
  be the property that actually replicates;
- survives its pre-registered falsification test;
- the parameter neighbourhood is a **plateau, not a peak** — report the
  neighbours, not just the winner.

Anything else is `NEGATIVE`. Write it up with the same care as a win.

---

## Step 5 — Record

Add **one section** to section B of [`LEDGER.md`](LEDGER.md), at the
**top** of that section — the log is newest-first, so today's round goes
directly above the previous one, never at the bottom of the file and
never inside another round's write-up. Copy the skeleton from
[Appending an entry](LEDGER.md#appending-an-entry) at the bottom of that
file; the shape is fixed:

```markdown
### R-nn · MM-DD · <VERDICT> — <short title, ≤90 chars>

**Direction.** the idea, its citation, the backlog item, the constraint
attacked, and the IDs it is not a duplicate of.

**What was done.** branches and files, data, the pre-registered decision
rule and falsification test as frozen, **configs evaluated** (total across
ALL branches).

**Result.** train / inner-validation / holdout numbers, the decision
rule's outcome, the falsification test's outcome, skeptic reproduction.

**Verdict.** verdict, one-line lesson, **holdout counter** (increment and
running total), whether the decision rule moved, next step.
```

A round that produced a long-form pre-registration or results write-up
keeps it in the same section, under `####` sub-headings (`#### R-nn
pre-registration — …`, `#### R-nn results — …`), as R-28 through R-40 do.

Three rules the format exists to protect, learned by losing them:

- **Prose belongs in a section, never in a table cell.** Section B was a
  table until 08-20 and it broke every way a table can: cells grown to
  10,000 characters, `|basis|`-style notation silently shifting whole
  columns (R-41, R-44), a round appended to the wrong table (R-46), and
  nine rounds appended below the table as raw pipe-text (R-47–R-55).
  Sections A, C and D are short-cell registries and stay tables — if one
  of their cells starts wanting a paragraph, the paragraph goes in the
  round's section in B and the cell gets the ID.
- **Newest first, everywhere.** Section B and the holdout-consultation
  list are both appended at the top. Reading down either one reads
  backwards through the project.
- **Nothing is deleted.** A superseded finding is annotated in place
  (R-28's risk claim, retracted by R-31), never removed.

Then, by verdict:

- **PROMOTED** → register the strategy, run `python scripts/inference.py`
  so it has an interval, then the full `tradebot run` to regenerate the
  README table with that interval in it, and refresh
  `docs/STRATEGIES.md` and `docs/VALIDATION.md`. CI fails if a
  registered strategy is missing from either the README table or
  `reports/inference/bootstrap.csv`.
- **NEGATIVE** → ledger entry plus code under `experiments/` (not
  auto-discovered), **unless** the negative is instructive enough to
  earn a table row the way `minority_oracle` and `game_switch` did.
  Registering every failure inflates the table and slows every future
  run; the ledger is the record now.
- **BLOCKED / PARKED** → ledger entry with the blocker named and what
  would unblock it.
- **No round dispatched at all** (Step 0b sent you here: zero configurations
  evaluated) → **one row in section E**, and nothing anywhere else. Not an
  R number, not a section in B, and *not* a paragraph at the head of section
  D — that is the specific mistake R-158 had to undo, and the reason it went
  unnoticed for twenty passes is that twenty interchangeable prose blobs look
  like diligence. Five short cells: pass number, commit time, step-0 result,
  what was attempted, what came of it. If the outcome genuinely does not fit
  in a cell, it was not a null pass — write it up in B as a round.

Finally, **re-rank the backlog** at the bottom of the ledger, then
commit and push.

**Re-ranking means editing the table, not appending a paragraph above it.**
Section D's header carries one `Re-ranked after R-nn` paragraph per round and
had grown to 4,556 lines — in a section whose whole point is a table that
Step 0 makes you `grep` because nobody reads that far. If a round changes an
item's status, change the item's status cell and strike it if it is closed;
add a paragraph only when *why* it moved will not fit in the note column.

---

## The mechanics: registering a strategy

The scaffold creates `src/tradebot/strategies/<name>.py` with a working
EMA-cross template, auto-discovered on the next run:

```bash
tradebot new my_strategy
pytest                                        # no-lookahead check runs for it automatically
tradebot run --strategies my_strategy buy_and_hold --max-bars 100000   # quick compare
python scripts/inference.py                   # its interval; CI requires one
```

Or write the file by hand:

```python
# src/tradebot/strategies/my_strategy.py
import pandas as pd
from tradebot.indicators import ema
from tradebot.registry import register
from tradebot.strategy import Context, Strategy

@register
class MyStrategy(Strategy):
    """One-line description shown in reports."""

    name = "my_strategy"   # unique
    warmup = 100           # bars skipped before the first on_bar call

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # Called once with the full OHLCV frame. Add indicator columns.
        # MUST be causal (row i may only use rows <= i): rolling / ewm /
        # shift are fine. A framework test verifies this for every
        # registered strategy.
        df["fast"] = ema(df["close"], 20)
        df["slow"] = ema(df["close"], 100)
        return df

    def on_bar(self, ctx: Context) -> None:
        # Called at every bar close; orders fill at the NEXT bar open.
        if ctx.bar["fast"] > ctx.bar["slow"] and not ctx.in_market:
            ctx.order_target(1.0)      # fully long (fraction of equity x leverage)
        elif ctx.bar["fast"] < ctx.bar["slow"] and ctx.position > 0:
            ctx.close_position()       # or ctx.order_target(-1.0) to short (futures)
```

That's it — `tradebot run` picks it up, tests it on the whole matrix,
ranks it in the comparison table and refreshes the README table. `ctx`
also offers `history(n)`, `equity`, `position`, `can_short`, and raw
`buy(qty)` / `sell(qty)`. `ctx.bar` / `ctx.prev` are fast mapping-style
views (`bar["rsi"]`).

Three rules are CI-enforced for every registered strategy (GitHub Actions
runs the suite on each push/PR):

- it **must have a docstring** describing the idea (first line lands in
  the comparison table and `tradebot list`);
- it **must appear in the README comparison table** — run the full
  `tradebot run` after adding a strategy, commit the regenerated
  README + reports, and CI stays green; and
- it **must have a measured interval** in
  `reports/inference/bootstrap.csv`, on both markets and both periods —
  `python scripts/inference.py` writes it, and
  `tests/test_evidence.py` fails without it (R-30). The point is that a
  new row cannot enter the table as a bare point estimate beside rows
  that carry error bars; that asymmetry is how a fresh number gets read
  as a better one. Run the inference script **before** `tradebot run`, so
  the regenerated table already carries the new row's interval.

Registration is the *end* of the routine, not a shortcut past it: a
strategy is registered either because it was **PROMOTED**, or because it
is a documented negative result instructive enough to earn a table row
(step 5). Experiments that are neither live in `experiments/` or
`scripts/experiment.py`.

---

## Running directions in parallel

The routine is written for one session, one idea. Several directions can
be explored at once — but parallelism is a **multiple-testing
multiplier**, not a free speedup, and it has to be paid for:

- **The trials count is the total across all parallel branches**, not
  per branch. Five directions each sweeping 8 configurations is 40
  trials for deflated Sharpe, not 8. The single most likely way to
  manufacture a fake winner here is to run many searches and report the
  best one as though it were the only one.
- **Every branch reports, including the dead ones.** Reporting only the
  branch that worked *is* selection on the holdout, performed by the
  operator rather than the code.
- **Each branch owns disjoint files**, and none of them commits. A
  parallel round is merged and committed once, by the operator, after
  the verdicts are in.
- **An independent skeptic re-runs each surviving claim** before it is
  believed — reading the code rather than the report, re-deriving the
  numbers, and hunting specifically for full-series fits (a scaler,
  quantile, mean or std computed over the *whole* series and applied to
  early rows is lookahead that the truncation test will not catch).
- **Dispatch a skeptic only after the primary reports ≥1 evaluated
  configuration.** An adversarial re-run against an empty task cannot add
  evidence about the direction. R-26 burned five skeptic sessions
  re-measuring the same broken harness the first primary had already
  diagnosed.
- **A synthesis prompt must not contain a conditional naming the result
  it expects.** State the question, not the hoped-for answer. R-27: a
  prompt reading *"if X was found, say so first and plainly"* was handed
  to a synthesizer whose inputs were empty. It refused; a compliant one
  would have printed the headline as fact. Ask "what did the evidence
  show?" and make "nothing" an explicitly acceptable answer.
- **Not tested is not a negative result.** A negative needs a
  measurement. A branch that produced no evaluated configuration goes
  back on the backlog as untried, never into the ruled-out list —
  mislabelling it there is worse than losing the round, because it stops
  the idea being tried again.

## Standing rules

- **Match risk before comparing anything.** Holding less draws down less;
  that is arithmetic, not evidence. Three of this project's findings have
  died of it — R-28's e-process drawdown cut (retired by R-31), R-32's
  gate comparison, and L-04's own headline (retired by R-33, which found
  **88–92%** of "regime-gated sizing cuts drawdown" was the exposure
  level). A comparison whose arms carry different realized volatility is
  a statement about the exposures, whatever else it looks like. If the
  arms cannot be matched, say so and void the cell rather than scoring
  it, and prefer matching **inside each resampled window** over freezing
  an exposure on one period: R-33 froze one on 2021–22 and five of six
  holdout cells failed the match, while its per-window matching landed
  within 0.5%.
- **Cost the plan, not just the mechanism.** Every rule above falsifies an
  *idea*. Nothing here falsifies the **backlog's own top item**, and that
  is how B-06 spent eleven rounds as "the highest-value item on merit"
  without anyone computing how long it needed: the answer turned out to be
  18.9 years to never, and the recorder accumulating it was seeing 10% of
  the strategy's decisions (R-78). So: **when the top of the backlog has
  been named "the answer" for three consecutive rounds without being
  carried to a verdict, the next session's job is to cost it** — how much
  evidence does it need, how fast does that evidence arrive, and is the
  instrument producing it actually measuring what it claims? A plan is a
  claim. Before believing it, check what it would take for it to be wrong.
  This is the same move R-33 made on risk-matching and R-57 made on
  cross-asset scope, applied to the research plan instead of a result, and
  it is cheap: R-78 cost one session and read zero holdout.
- **Match risk in the *controls* too, not only the benchmark.** R-33's rule is
  written about benchmarks, and R-131 found the same trap one level down: its
  ablation controls scored a higher Sharpe than the live branch, and every one
  of them sat **99.5–100% in market against the incumbent's 55.6%**. An
  ablation that changes the exposure has not isolated the mechanism, it has
  replaced it — and because an ablation is supposed to *weaken* a claim, an
  unmatched one fails silently in the direction of a false negative. Report
  time-in-market and realized volatility for every arm, control arms included.
- **Read a metric's definition before dividing by it.** `Metrics.num_trades`
  counts round-trip **episodes** (`build_trades` groups fills); turnover is
  `len(result.fills)`. R-131 wrote a whole results section on the wrong one and
  concluded a mechanism "freezes the strategy — 1 trade in two years" when it
  was in fact filling 240–352 times and merely never fully exiting. The two
  numbers differ by 300x on exactly the mechanisms a COST-axis round studies.
  Carry both units in the table so they cannot be confused, and be suspicious
  of any turnover ratio that looks too good: a mechanism that reduces trading
  by 99% has usually changed what "a trade" means.
- **Nothing is deleted.** Registered negative results stay registered.
- **The table's futures column is an upper bound** until funding is
  charged. Never quote it without that caveat.
- **Never proxy unavailable data out of price.** If the information is
  not in the file, it is not in the strategy.
- **Report ranges, not points**, wherever a bootstrap is available.
- One session, one idea, one ledger entry — a new section at the top of
  the ledger's section B, never a row appended to a table. **Unless the
  session had no idea to run**, in which case it is one row in section E and
  no entry at all: the inverse rule matters just as much, because a session
  with nothing to report that writes a section-B-sized write-up anyway is
  how twenty passes produced 1,285 lines and zero measurements (R-158).
- **The routine can return "nothing to do", and that is a pass, not a
  failure.** Step 0b decides it. Every instruction above is written for a
  session that has work; none of them is a reason to manufacture some.
