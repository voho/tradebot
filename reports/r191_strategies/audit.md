# R-191 independent audit

Fixed subject: `r191_cost_proximal`.

Targets are reconstructed by enumerating the proximal objective's extrema,
using independently identified complete days and NumPy rolling-window moments.
Accounting uses a separate quote-cash book without the production engine or broker.
Synthetic tests do not evaluate historical portfolios. Unsupported liquidation
or funding insolvency paths fail explicitly.

Financial replays: **2 training, 2 holdout**.

| Cell | State | Balance | Max target error | Max daily equity error |
|---|---|---|---|---|
| inner_val | completed | 1303.904991 | 4.55e-15 | 1.59e-12 |
| funded_val | completed | 1118.275216 | 2.55e-15 | 2.05e-12 |
| holdout | completed | 2683.618554 | 4.55e-15 | 6.37e-12 |
| funded_holdout | completed | 2321.697490 | 2.78e-15 | 8.19e-12 |

Every financial replay is appended and counted before execution. Daily dates,
costs, funding, request/fill/episode counts, realized risk and time in market
must agree with the already-completed primary receipt. Zero discrepancies are
claimed only for completed records; failed or started attempts remain counted.

After the four replays, the audit CLI's input paths were changed to use
the committed aggregate cell/daily artifacts filtered to the fixed subject,
instead of ignored per-strategy receipts. Accounting, signal reconstruction
and thresholds did not change, and no financial replay was repeated.
The original executed audit-code hash remains on each reproduction record.
