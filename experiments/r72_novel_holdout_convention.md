# R-72 — settling B-33: is the panel "+0 holdout consultations" convention sound?

Methodology round. No candidate strategy, no sweep, no promotion bar. Audits
already-committed code (R-47, R-57, R-63, R-65, R-67, R-68, R-69, R-70) plus
one small, pre-authorized, prices-only correlation read. Script:
`experiments/r72_novel_holdout_convention.py` (`python experiments/r72_novel_holdout_convention.py all`).

## B-33's literal question, answered: NOT the leak

B-33 asks whether a fitted parameter/threshold selected using panel data
that extends past 2023-01-01 (`W_FULL6`) ever reaches a BTC/ETH computation.
Audit 4 (static text search of every sweep/selection function in R-63,
R-65, R-67, R-68's eight branch files) found **zero** such references: every
`cmd_sweep` / `cmd_frontier` / `cmd_select` operates only on `W_TRAIN`
(2020-04-01→2021-12-31) and `W_VAL` (2022-01-01→2022-12-31), both strictly
pre-holdout. `W_FULL6` is touched only inside each branch's `cmd_run`, after
the parameter is already frozen. So the convention's literal technical
premise — no fitted quantity crosses from panel-2023+ into a BTC/ETH
computation — **holds**. B-33 as originally framed is answered negatively:
the panel-only channel it worried about is not where the leak is.

## The actual finding: a different, unflagged channel reads real BTC 2023+ data directly

Audit 1 (static scan) and Audit 3 (aggregation of already-committed
`reports/*.csv` output) found that **8 of the 8 branches in R-63, R-65,
R-67 and R-68** build a `BTC_HOLD` "context" cell inside `cmd_run`:

```python
btc = frames["BTC"]                         # unrestricted load_dataset(), 2017–2026
btc_on = btc.reindex(btc.index.union(targets.index)).ffill().reindex(targets.index)
btc_eq = static_hold_equity({"BTC": btc_on}, ["BTC"], SPOT_BASE)
r_btc = compare(cand, btc_eq)                # paired stationary-block bootstrap
```

`targets.index` is `W_FULL6` (2020-04-01 → last bar), which Audit 2
independently confirms is **56.8% post-`OOS_START`** (2,332 evaluated days,
1,319 of them ≥ 2023-01-01) — reproducing the ledger's own "~57%" figure
from the raw BTC file alone. `compare()` runs a real paired bootstrap
(2,000 resamples) on BTC's actual daily returns over that window and
returns a growth-difference point estimate with a 95% interval. This is not
panel data standing in for BTC — it is BTC's own reserved 2023+ price bars,
read, resampled, and reduced to a statistic, in every one of these eight
branches. R-68's novel branch does this twice (its two derived-threshold
configurations `D_A`/`D_B` each build a separate `BTC_HOLD` cell), for
**9 total context-cell reads across 4 rounds**:

| round/branch | growth_diff vs BTC_HOLD | 95% interval | source |
|---|---|---|---|
| R-63 conservative | −1.767 | [−4.568, +1.118] | `reports/r63_panel_portfolio/conservative_cells.csv` |
| R-63 novel | −8.869 | [−14.573, −3.003] | `reports/r63_panel_portfolio/novel_cells.csv` |
| R-65 conservative | −2.124 | [−6.368, +2.592] | `reports/r65_holding_period/conservative_cells.csv` |
| R-65 novel | −0.709 | [−4.297, +3.122] | `reports/r65_holding_period/novel_cells.csv` |
| R-67 conservative | −0.420 | [−4.447, +3.907] | `reports/r67_gate/conservative_cells.csv` |
| R-67 novel | −0.223 | [−4.187, +4.091] | `reports/r67_gate/novel_cells.csv` |
| R-68 conservative | −0.450 | [−4.712, +4.121] | `reports/r68_band/conservative_cells.csv` |
| R-68 novel (D-A) | −0.918 | [−5.103, +3.621] | `reports/r68_band/novel_cells.csv` |
| R-68 novel (D-B) | +0.220 | [−3.708, +4.341] | `reports/r68_band/novel_cells.csv` |

These numbers were not newly computed by this round — they already exist,
committed, in the report CSVs those rounds themselves produced. This audit
only aggregates and cites them.

By contrast: **R-57's** own `cmd_control` explicitly truncates its BTC/ETH
comparison at 2022-12-31 "so no 2023+ BTC bar is read and the holdout
counter stays at +0" — R-57 is genuinely clean, and its authors were
visibly aware of exactly this trap. **R-69** dropped the `BTC_HOLD` context
cell entirely (confirmed: no `frames["BTC"]` literal in either branch file
despite inheriting the same shared harness) and its ledger bullet's claim —
"neither file... references `HOLDOUT`/`OOS_START` anywhere" — holds. **R-70**
built a harness whose windows never resolve past `W_VAL` at all. So this is
not the whole project's practice; it is a regression that entered at R-63
(inherited via `r63_shared.py`'s `BTC_HOLD` benchmark convenience, copied
forward by every branch that imported R-63's harness through R-68) and was
independently fixed by R-69, apparently without anyone naming what R-69 had
fixed.

## Was this leak ever load-bearing on a decision?

Checked directly: every `d1_pass` / `d2_pass` / `d3_pass` / `d4_pass` /
`further_work()` in all eight branches is computed from `MATCHED_HOLD`,
`EW_HOLD`, or `VOLMATCH_HOLD` — never from `BTC_HOLD`. The frozen parameter
in every branch (R-63's `k=1`, R-65's `buffer`/`hold_days` and `a`, R-67's
`delta`, R-68's `d`) was selected on `W_TRAIN`/`W_VAL` *before* `cmd_run`
(and therefore before the `BTC_HOLD` cell) ever executed. So there is no
evidence this specific channel moved a gate, a threshold, or a promotion
decision in any of the four rounds — the mechanical "did it corrupt a
decision" question the deflated-Sharpe correction ultimately protects has a
clean answer here: no.

But the project's own operational definition of "holdout consultation" —
established by its own practice, not by this round — has never been "did it
change a decision." It has been "was the bar read at all." R-57's own
language ("so no 2023+ BTC bar is read") and every "+0" bullet's careful
grep-for-`load_btc()`/`HOLDOUT` audit treat the read itself, independent of
downstream use, as the countable event; that is also the logic ROUTINE.md
gives for pre-registration in the first place ("looking is not what
corrupts a holdout — moving the goalposts after looking is" — which
presupposes the look is being tracked regardless). Judged by the standard
this project has consistently applied to itself everywhere else, these nine
reads are holdout consultations that were never logged.

## Verdict

**(b) — the "+0" convention is NOT justified as applied, but not for the
reason B-33 named.** B-33's panel-fitted-parameter channel is clean (Audit
4: zero contamination). The actual channel is a literal, direct read of
real BTC 2023+ price bars, mislabelled "context" and built into the shared
harness `r63_shared.py` introduced and every downstream round through R-68
inherited unmodified, until R-69 silently dropped it.

**Proposed retroactive correction.** +1 holdout-equivalent consultation per
branch that built and ran a `BTC_HOLD` context cell on `W_FULL6`:

- R-63: **+2** (conservative, novel) — was +0, now ~629 relative to R-63's own baseline
- R-65: **+2** (conservative, novel) — was +0
- R-67: **+2** (conservative, novel) — was +0
- R-68: **+3** (conservative, novel×2 for D_A/D_B) — was +0

**+9 total**, raising the running program-level total from ~627 to
**~636**, applied at R-68's position in the sequence (R-69's and R-70's own
"+0" bullets are unaffected — they are downstream of R-68 in the log and
their own totals were computed as "R-69's ~627" / "R-70's ~627"; each
downstream "~627" should read "~636" once this correction is applied,
though none of R-69/R-70/R-71's own verdicts depended on the exact value).
This is a bookkeeping correction, not a re-opened verdict: nothing above
suggests any of R-63/65/67/68's NEGATIVE verdicts were reached by peeking —
the correction is to the trials count the deflated-Sharpe multiple-testing
adjustment uses, not to any individual finding.

## Part 1b — the researcher-degrees-of-freedom question (honest, not settled)

This is explicitly not answerable by grepping code, and this round does not
claim to close it. Two observations, offered as a named risk rather than a
measurement:

- The nine `BTC_HOLD` numbers above were **printed to stdout and written to
  CSV** at the end of each branch's `cmd_run`, i.e. visible to whichever
  session or skeptic read that round's output — including, in five of nine
  cases, negative point estimates (the candidate loses to BTC buy-and-hold
  even net of the multiple-testing question) and in one case (R-63 novel)
  an interval that excludes zero against the candidate. A researcher
  reading that number before writing the round's verdict has, at minimum,
  seen a real quantified signal about how the candidate would have fared
  against BTC's actual 2023+ path — even though (checked above) it entered
  no gate.
- Separately, and this is the part that cannot be closed by code
  inspection at all: every one of these rounds' authors also had the
  *panel's own* W_FULL6 performance in hand before deciding whether to
  request a `W_HOLD` read from the operator. Whether seeing the panel
  (or, now, the `BTC_HOLD` context number) shifted anyone's *prior* about
  what a real BTC/ETH holdout read would show — and thus shifted the
  decision of whether to ask for one — is not observable from the repository.
  It is named here as an unquantifiable risk, consistent with the brief's
  instruction not to pretend this half of the question is closed.

## Part 3 — is a panel-2023+ read independent of a BTC/ETH-2023+ read?

New read this round (pre-authorized by the task brief, prices only, nothing
swept or selected): daily-return correlation between BTC and each U6 panel
asset, split at `OOS_START`.

| asset | corr(BTC, ·) full | pre-2023 | post-2023 |
|---|---|---|---|
| BCH | 0.603 | 0.628 | 0.559 |
| LTC | 0.622 | 0.655 | 0.556 |
| ETC | 0.527 | 0.493 | 0.609 |
| DASH | 0.514 | 0.552 | 0.463 |
| LINK | 0.565 | 0.547 | 0.600 |
| XTZ | 0.530 | 0.522 | 0.547 |
| **mean** | **0.560** | **0.566** | **0.556** |

Mean BTC-panel correlation is essentially unchanged across the holdout
boundary (0.566 → 0.556) and sits close to R-63's own measured
panel-panel correlation (0.634) over the same window. This directly
supports the backlog item's second question: **a panel-wide 2023+ read is
not independent information from a BTC/ETH 2023+ read.** At ~0.56 mean
correlation, a panel-wide regime read (e.g. "did trend-following still work
anywhere in 2023-2026") carries real information about BTC/ETH's own 2023+
regime through price co-movement alone — no shared code path is needed for
this channel; it is a property of the asset class, not of any experiment's
implementation. This does not change the Audit-4 verdict on *fitted
parameters* (still clean), but it does mean the panel channel B-33 named is
not truly "free information" in the way the "+0" convention's informal
framing suggests, even where no code-level leak exists. It is weaker
evidence than a literal data leak — correlation, not causation, and not
something deflated Sharpe has a standard way to price — but it is a real,
measured, non-zero dependency and should be named as such rather than
implied away by "the reserved holdout is untouched."

## Recommendation for `docs/LEDGER.md`'s convention going forward

1. **Correct the running total** from ~627 to ~636 per the table above, and
   add a bullet (or amend R-68's existing one) explaining the nine
   previously-uncounted `BTC_HOLD` context reads.
2. **Retire the `BTC_HOLD` context cell from `r63_shared.py`'s pattern, or
   truncate it at `2022-12-31` the way R-57's `cmd_control` and R-69 already
   do.** There is no finding in this round that the cell added analytical
   value proportional to its cost — every branch's decision already ran
   through `MATCHED_HOLD`/`EW_HOLD`/`VOLMATCH_HOLD`, and `BTC_HOLD` was
   reported but never gated on. Cheapest fix available: one-line truncation
   in any future round that reuses this harness.
3. **State the panel convention narrowly, not broadly.** "+0 for a
   `W_FULL6` panel read" should be qualified: true for parameter fitting
   (Audit 4, clean), not free of researcher-degrees-of-freedom risk (1b,
   unquantifiable, named), and not fully independent of the BTC/ETH regime
   even at the price level (Part 3, ~0.56 correlation). A round that wants
   genuine "+0" should do what R-57 and R-69 did: prove by grep that no
   `frames["BTC"]`/`frames["ETH"]` object loaded via the unrestricted
   loader is ever reindexed onto or compared against a window whose right
   edge is `None`/last-bar, not just that no *decision* references
   `W_HOLD`.
