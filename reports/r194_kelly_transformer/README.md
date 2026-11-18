# R-194 — Kelly v4 x transformer hybrids: results

Frozen protocol: [`experiments/r194_protocol.md`](../../experiments/r194_protocol.md) ·
notes: [`experiments/r194_research_notes.md`](../../experiments/r194_research_notes.md) ·
code: [`experiments/r194_strategies.py`](../../experiments/r194_strategies.py) ·
ledger: R-194 in [`docs/LEDGER.md`](../../docs/LEDGER.md).

**Verdict: NEGATIVE for all three primaries** under R-192's decision rule, applied verbatim
(all common gates and a complete growth or tail route, against both the unchanged v4 parent and a
volatility-matched passive hold). Manifest frozen 2026-11-18T16:04:51Z before any holdout cell.
Training-only lead: `r194_kelly_episodic_memory`. Program trials 5,488.

## Gates

| primary | risk valid | binds (fills differ) | profitable | no liquidation | plateau | DSR ≥ .95 | growth route | tail route | DSR spot / funded | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `r194_panel_cross_attention` | **fail** | pass | pass | pass | **fail** | **fail** | **fail** | **fail** | 0.120 / 0.142 | NEGATIVE |
| `r194_kelly_episodic_memory` | **fail** | pass | pass | pass | **fail** | **fail** | **fail** | **fail** | 0.082 / 0.111 | NEGATIVE |
| `r194_residual_transformer` | **fail** | pass | pass | pass | **fail** | **fail** | **fail** | **fail** | 0.058 / 0.099 | NEGATIVE |

## Paired comparisons (30-day block bootstrap, 2,000 draws, seed 192)

`control_to_candidate_vol` outside 0.95–1.05 makes the parent comparison invalid (R-192's 5% rule).

| primary | cell | control | risk valid | Sharpe | control | ΔSharpe [95% CI] | Δlog growth [95% CI] | Δdaily max DD (pp) |
|---|---|---|---|---:|---:|---|---|---:|
| `r194_panel_cross_attention` | inner_val | parent | yes | -0.112 | -0.077 | -0.035 [-0.382, +0.235] | -0.035 [-0.236, +0.155] | +2.8 |
| `r194_panel_cross_attention` | inner_val | matched_hold | yes | -0.112 | 0.016 | -0.128 [-0.745, +0.506] | -0.083 [-0.445, +0.374] | +1.3 |
| `r194_kelly_episodic_memory` | inner_val | parent | **no** | 0.163 | -0.077 | +0.240 [-0.258, +0.611] | +0.121 [-0.228, +0.493] | +2.2 |
| `r194_kelly_episodic_memory` | inner_val | matched_hold | yes | 0.163 | -0.023 | +0.186 [-0.797, +1.138] | +0.142 [-0.538, +0.920] | -4.3 |
| `r194_residual_transformer` | inner_val | parent | **no** | -0.014 | -0.077 | +0.063 [-0.400, +0.464] | -0.091 [-0.627, +0.511] | +11.9 |
| `r194_residual_transformer` | inner_val | matched_hold | yes | -0.014 | -0.010 | -0.004 [-0.903, +0.934] | -0.003 [-0.804, +0.944] | -3.4 |
| `r194_panel_cross_attention` | funded_val | parent | yes | -0.048 | 0.117 | -0.165 [-0.610, +0.296] | -0.098 [-0.356, +0.165] | +7.4 |
| `r194_panel_cross_attention` | funded_val | matched_hold | yes | -0.048 | -0.031 | -0.017 [-0.630, +0.551] | -0.009 [-0.360, +0.376] | +0.4 |
| `r194_kelly_episodic_memory` | funded_val | parent | yes | 0.368 | 0.117 | +0.250 [-0.380, +0.918] | +0.148 [-0.214, +0.538] | -2.6 |
| `r194_kelly_episodic_memory` | funded_val | matched_hold | yes | 0.368 | -0.066 | +0.434 [-0.485, +1.304] | +0.253 [-0.257, +0.777] | -9.7 |
| `r194_residual_transformer` | funded_val | parent | **no** | -0.116 | 0.117 | -0.233 [-0.863, +0.274] | -0.500 [-1.394, +0.472] | +40.3 |
| `r194_residual_transformer` | funded_val | matched_hold | yes | -0.116 | -0.062 | -0.054 [-1.054, +0.911] | -0.047 [-1.177, +1.303] | +4.8 |
| `r194_panel_cross_attention` | holdout | parent | **no** | 0.949 | 0.931 | +0.019 [-0.224, +0.243] | +0.067 [-0.231, +0.344] | +5.9 |
| `r194_panel_cross_attention` | holdout | matched_hold | yes | 0.949 | 1.026 | -0.077 [-0.493, +0.297] | -0.107 [-0.580, +0.394] | -2.1 |
| `r194_kelly_episodic_memory` | holdout | parent | **no** | 0.836 | 0.931 | -0.095 [-0.376, +0.181] | -0.142 [-0.493, +0.195] | +0.2 |
| `r194_kelly_episodic_memory` | holdout | matched_hold | yes | 0.836 | 1.046 | -0.211 [-0.871, +0.435] | -0.224 [-0.907, +0.492] | -5.0 |
| `r194_residual_transformer` | holdout | parent | yes | 0.740 | 0.931 | -0.191 [-0.486, +0.092] | -0.208 [-0.547, +0.122] | +9.5 |
| `r194_residual_transformer` | holdout | matched_hold | yes | 0.740 | 1.067 | -0.327 [-0.914, +0.243] | -0.389 [-1.060, +0.332] | +2.3 |
| `r194_panel_cross_attention` | funded_holdout | parent | yes | 1.003 | 1.057 | -0.054 [-0.280, +0.143] | -0.033 [-0.340, +0.249] | +9.0 |
| `r194_panel_cross_attention` | funded_holdout | matched_hold | yes | 1.003 | 0.911 | +0.092 [-0.318, +0.466] | +0.134 [-0.390, +0.679] | -4.7 |
| `r194_kelly_episodic_memory` | funded_holdout | parent | **no** | 0.923 | 1.057 | -0.134 [-0.500, +0.223] | -0.050 [-0.633, +0.566] | +8.9 |
| `r194_kelly_episodic_memory` | funded_holdout | matched_hold | yes | 0.923 | 0.894 | +0.029 [-0.701, +0.706] | +0.057 [-1.005, +1.149] | -10.5 |
| `r194_residual_transformer` | funded_holdout | parent | **no** | 0.888 | 1.057 | -0.169 [-0.593, +0.189] | +0.100 [-0.861, +1.109] | +19.5 |
| `r194_residual_transformer` | funded_holdout | matched_hold | **no** | 0.888 | 0.890 | -0.002 [-0.607, +0.583] | +0.121 [-1.021, +1.325] | -4.3 |

## Final balances from $1,000

| strategy | inner_train | inner_val | funded_val | holdout | discount_holdout | funded_holdout | eth_holdout |
|---|---:|---:|---:|---:|---:|---:|---:|
| `buy_and_hold` | $29,913 | $573 | $481 | $3,827 | $3,839 | $3,137 | $1,566 |
| `kelly_regime_v4` | $13,296 | $869 | $983 | $2,454 | $3,392 | $3,193 | $1,662 |
| `r194_kelly_episodic_memory` | $7,446 | $980 | $1,139 | $2,129 | $2,841 | $3,038 | $1,203 |
| `r194_kelly_episodic_memory_high` | $8,334 | $953 | $1,011 | $2,309 | $3,099 | $2,966 | $1,419 |
| `r194_kelly_episodic_memory_low` | $4,916 | $1,066 | $1,223 | $1,750 | $2,389 | $2,675 | $1,087 |
| `r194_panel_cross_attention` | $12,064 | $839 | $891 | $2,623 | $3,312 | $3,091 | $1,571 |
| `r194_panel_cross_attention_high` | $11,793 | $834 | $783 | $2,669 | $3,200 | $2,889 | $1,563 |
| `r194_panel_cross_attention_low` | $13,580 | $855 | $842 | $2,553 | $3,348 | $3,196 | $1,656 |
| `r194_residual_transformer` | $11,115 | $793 | $596 | $1,992 | $2,922 | $3,529 | $1,000 |
| `r194_residual_transformer_high` | $9,508 | $860 | $886 | $2,191 | $3,097 | $4,839 | $910 |
| `r194_residual_transformer_low` | $5,414 | $851 | $834 | $2,347 | $3,248 | $3,509 | $1,434 |

The residual transformer's L=60 neighbour reaches $4,839 on the funded perpetual by running far more
volatility than v4 (the primary's parent volatility ratio is 0.63); its ETH Sharpe is 0.13. At matched
risk the primary's funded edge is −0.002. That is R-33's exposure artifact, not an improvement.

The first training run was discarded: it executed stale bytecode from in-place mutation testing
(receipts kept locally in `discarded_stale_bytecode_train/`, not committed); only its nine
residual-transformer cells differed from the clean rerun.

## Counts

317 core evaluations + 366 matching attempts = **683 financial evaluations**
(plus 33 discarded stale-bytecode training cells); holdout consultations **+638**,
cumulative ~5,488.
