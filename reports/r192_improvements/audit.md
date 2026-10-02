# R-192 independent audit

Fixed subject: `r192_tracking_budget`.

Targets are reconstructed from rolling anchors, raw volatility states and
a separate tracking-distance accumulator; callbacks use the common startup convention.
Accounting uses a separate quote-cash book without the production engine or broker.
Synthetic tests do not evaluate historical portfolios. Unsupported liquidation
or funding insolvency paths fail explicitly.

Financial replays: **2 training, 2 holdout**.

| Cell | State | Balance | Max target error | Max daily equity error |
|---|---|---|---|---|
| inner_val | completed | 874.204420 | 0 | 1.59e-12 |
| funded_val | completed | 871.810948 | 0 | 2.27e-12 |
| holdout | completed | 2406.118823 | 0 | 3.64e-12 |
| funded_holdout | completed | 3088.339955 | 0 | 1.5e-11 |

Every financial replay is appended and counted before execution. Daily dates,
costs, funding, request/fill/episode counts, realized risk and time in market
must agree with the already-completed primary receipt. Zero discrepancies are
claimed only for completed records; failed or started attempts remain counted.
