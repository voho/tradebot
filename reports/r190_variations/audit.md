# R-190 independent skeptical audit

The independent audit reproduced **eight financial cells: four training
and four holdout**, with no material discrepancy. Holdout work began only
after the operator confirmed the committed manifest freeze (`2da272c`).
The selected held-out subject was the training-only lead,
`r190_v3_b20`; these measurements did not select another default.

The implementation is retained in `tests/test_r190_audit.py`. It derives
the conditional Kelly signal from the parent constants, then simulates an
equivalent quote-cash account independently of `run_backtest`,
`PaperBroker`, `Prepared` and the R-190 evaluation helper. It checks
next-open execution, side-dependent slippage, fees, order minimums, native
broker deadband, actual holdings, funding and the parent initialization.
The checked cells never liquidate; the independent account asserts that
the maintenance and funding-solvency boundaries are not crossed rather
than claiming an independently implemented liquidation path.

| Strategy | Cell | Final balance | Fees | Funding paid | Fills | Completed episodes |
|---|---|---:|---:|---:|---:|---:|
| v4 band 0.10 | spot inner validation | $899.324632 | $239.376412 | $0 | 197 | 45 |
| Native v4 | spot inner validation | $868.828167 | $275.563221 | $0 | 259 | 52 |
| v4 band 0.10 | funded inner validation | $1,005.332482 | $23.495365 | $129.007854 | 122 | 45 |
| Native v4 | funded inner validation | $982.555747 | $26.497888 | $131.043261 | 146 | 51 |
| v3 band 0.20 | spot holdout | $2,226.412226 | $899.050603 | $0 | 241 | 46 |
| Native v3 | spot holdout | $2,210.109949 | $963.801989 | $0 | 297 | 48 |
| v3 band 0.20 | funded holdout | $2,758.546993 | $152.974726 | $626.671114 | 265 | 46 |
| Native v3 | funded holdout | $2,772.212365 | $163.996246 | $643.436934 | 301 | 49 |

Every compared metric agrees within $1e-7 or the corresponding metric
unit; the maximum daily account-value difference is **$1.28e-11**. The
source-reconstructed v4 and v3 target arrays equal the actual parent and
candidate targets exactly. Daily returns include the first measured day's
change from the fresh $1,000 account. The lead's small spot improvement
and funded loss are point arithmetic, not a promotion finding.

The training report has exactly **48 unique cells**, 16 names times
three cells, with all evaluation ends before 2023. Independent training
CSV parsing stops at the cutoff timestamp before parsing later financial
values. The primary worker likewise slices its raw frame before calling
`prepare`. Earlier causal history remains available, while every measured
cell starts with a fresh account. Full-period warmup suppression and
native-parent first-eligible initialization agree with the frozen protocol.

All ten candidates additionally pass real pre-2023 BTC target-identity,
future-perturbation and prefix-equivalence checks after their full warmups.
These target-only probes add **zero financial evaluations**. A separate
small synthetic regression checks the independent cash book's first
next-open fill and exact fee/slippage deduction without reading market
data.

The frozen risk matcher uses the existing constant-exposure strategy,
counts each attempt, retains failed attempts, and checks achieved
volatility. Its explicit-quantity orders intentionally retain a different
rebalance policy from the candidate's ordinary target orders; that
difference is disclosed in the protocol. Parent comparisons and beta
windows validate their own realized-volatility ratios rather than assuming
equal risk from a matching target multiplier. The final promotion decision
still requires the complete frozen gates; reproducing a balance alone
does not validate that decision.

Scope limits remain visible: the generic linear-margin broker and
8-hour funding aggregation are not Deribit's exact inverse-contract and
continuous-funding mechanics; the 0.25-equity target-order floor can
dominate small futures bands; quote minima permit risk-reducing closes;
there is no queue or market-impact model. Bar-level drawdown in the cells
and daily-resampled bootstrap drawdown are different statistics. The
inherited bootstrap helper initializes peaks at its first daily
observation, so it does not separately include the initial cash point.

Exact metrics, numerical errors, source hashes and counted evaluations
are in [audit.json](audit.json). The audit adds **+4** to this round's
holdout consultations and **+8** to its total financial evaluations.
