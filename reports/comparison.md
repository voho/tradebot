# Strategy comparison

Period: 2017-01-01 to 2026-08-12 (1,010,889 x 5m bars)  
Data: real, spot (perp proxy)  
Ranked by **final balance** (the primary comparison criterion); rows ordered by each strategy's best config.

Every comparison against `buy_and_hold` carries a 95% paired block-bootstrap interval. A rank is not a result: see the legend under the table for how much of this ordering survives that test. In the per-market detail tables below, ☠ marks a comparison made against a `buy_and_hold` account that was liquidated early and inert for most of the period — on 5x futures it dies in January 2017, so beating it there is a statement about surviving leverage, not about edge (R-22).

| # | strategy | spot | futures_5x | trades | profit | max DD | growth vs hold (spot) | max DD vs hold (spot) |
|---|---|---|---|---|---|---|---|---|
| 🥇1 | [kelly_regime_v4](../src/tradebot/strategies/kelly_regime_v4.py) | 🟢 $66.8K | 🟢 **$156.2K** | 174 | 📈 $155.2K | 35% | ≈ +0.04 [-2.60, +2.85] | ▲ -41.1pp [-54.8, -18.4] |
| 🥈2 | [kelly_regime_v3](../src/tradebot/strategies/kelly_regime_v3.py) | 🟢 $65.8K | 🟢 **$139.5K** | 147 | 📈 $138.5K | 42% | ≈ +0.03 [-2.54, +2.81] | ▲ -36.8pp [-53.4, -16.1] |
| 🥉3 | [kelly_regime_v2](../src/tradebot/strategies/kelly_regime_v2.py) | 🟢 $46.4K | 🟢 **$122.0K** | 113 | 📈 $121.0K | 40% | ≈ -0.32 [-3.15, +2.62] | ▲ -43.4pp [-54.7, -15.3] |
| 4 | [kelly_regime](../src/tradebot/strategies/kelly_regime.py) | 🟢 $42.1K | 🟢 **$108.2K** | 143 | 📈 $107.2K | 43% | ≈ -0.42 [-3.08, +2.36] | ▲ -38.9pp [-50.3, -12.2] |
| 5 | [kelly_regime_ev](../src/tradebot/strategies/kelly_regime_ev.py) | 🟢 $40.9K | 🟢 **$108.0K** | 135 | 📈 $107.0K | 37% | ≈ -0.45 [-3.28, +2.58] | ▲ -40.0pp [-55.5, -16.3] |
| 6 | [kelly_regime_ev_fast](../src/tradebot/strategies/kelly_regime_ev.py) | 🟢 **$71.1K** | 🟢 $70.8K | 34 | 📈 $70.1K | 32% | ≈ +0.11 [-3.08, +3.29] | ▲ -52.9pp [-62.9, -20.7] |
| 7 | [buy_and_hold](../src/tradebot/strategies/buy_and_hold.py) | 🟢 **$66.0K** | 💀 $18.05 | 1 | 📈 $65.0K | 84% ⚠️ | benchmark | benchmark |
| 8 | [champions_council](../src/tradebot/strategies/champions_council.py) | 🟢 $19.3K | 🟢 **$36.8K** | 261 | 📈 $35.8K | 37% | ≈ -1.20 [-4.06, +1.81] | ▲ -49.5pp [-54.7, -19.0] |
| 9 | [hedge_experts](../src/tradebot/strategies/hedge_experts.py) | 🟢 **$13.3K** | 🔴 $258 | 2,044 | 📈 $12.3K | 59% ⚠️ | ≈ -1.57 [-4.01, +0.96] | ▲ -24.1pp [-39.1, -3.0] |
| 10 | [replicator_book](../src/tradebot/strategies/replicator_book.py) | 🟢 **$2,330** | 🔴 $10.58 | 713 | 📈 $1,330 | 38% | ≈ -3.31 [-6.86, +0.28] | ▲ -47.1pp [-57.9, -20.7] |
| 11 | [universal_kelly](../src/tradebot/strategies/universal_kelly.py) | 🟢 **$1,276** | 🟢 $1,227 | 9 | 📈 $276 | 7% | ≈ -3.91 [-8.39, +0.44] | ▲ -76.6pp [-89.1, -52.9] |
| 12 | [harsanyi_crowd](../src/tradebot/strategies/harsanyi_crowd.py) | 🔴 **$888** | 🔴 $429 | 91 | 📉 -$112 | 11% | ≈ -4.28 [-8.88, +0.23] | ▲ -71.9pp [-85.6, -46.0] |
| 13 | [overshoot_fade](../src/tradebot/strategies/overshoot_fade.py) | 🔴 **$662** | 🔴 $33.52 | 189 | 📉 -$338 | 37% | ▼ -4.57 [-9.13, -0.07] | ▲ -46.6pp [-68.5, -19.6] |
| 14 | [camouflage_flow](../src/tradebot/strategies/camouflage_flow.py) | 🔴 **$548** | 🔴 $0.99 | 802 | 📉 -$452 | 53% ⚠️ | ▼ -4.76 [-9.23, -0.29] | ▲ -31.4pp [-57.7, -2.5] |
| 15 | [stealth_trend](../src/tradebot/strategies/stealth_trend.py) | 🔴 **$465** | 🔴 $0.38 | 1,605 | 📉 -$535 | 55% ⚠️ | ▼ -4.92 [-9.26, -0.76] | ≈ -29.0pp [-43.9, +12.1] |
| 16 | [flow_regime](../src/tradebot/strategies/flow_regime.py) | 🔴 **$447** | 🔴 $0.80 | 1,184 | 📉 -$553 | 56% ⚠️ | ▼ -4.96 [-9.43, -0.54] | ≈ -27.3pp [-47.2, +6.3] |
| 17 | [game_council](../src/tradebot/strategies/game_council.py) | 🔴 **$284** | 🔴 $2.00 | 2,541 | 📉 -$716 | 72% ⚠️ | ▼ -5.42 [-9.97, -0.95] | ≈ -11.5pp [-25.9, +12.4] |
| 18 | [elliott_wave](../src/tradebot/strategies/elliott_wave.py) | 🔴 **$272** | 💀 $81.67 | 1,261 | 📉 -$728 | 81% ⚠️ | ▼ -5.46 [-9.80, -1.00] | ≈ -2.9pp [-33.6, +22.3] |
| 19 | [minority_oracle](../src/tradebot/strategies/minority_oracle.py) | 🔴 **$53.36** | 🔴 $3.83 | 9,039 | 📉 -$947 | 95% ⚠️ | ▼ -7.09 [-11.60, -2.52] | ≈ +11.5pp [-3.8, +35.9] |
| 20 | [game_switch](../src/tradebot/strategies/game_switch.py) | 🔴 **$5.00** | 🔴 $1.00 | 6,672 | 📉 -$995 | 99% ⚠️ | ▼ -9.45 [-15.38, -4.06] | ▼ +16.3pp [+1.0, +39.3] |
| 21 | [regret_grid](../src/tradebot/strategies/regret_grid.py) | 🔴 **$5.00** | 🔴 $1.00 | 3,461 | 📉 -$995 | 100% ⚠️ | ▼ -9.46 [-16.20, -3.49] | ▼ +16.3pp [+0.9, +39.9] |
| 22 | [tft_trend](../src/tradebot/strategies/tft_trend.py) | 🔴 **$4.99** | 🔴 $1.00 | 2,538 | 📉 -$995 | 100% ⚠️ | ▼ -9.46 [-14.88, -4.58] | ▼ +16.3pp [+3.0, +39.9] |
| 23 | [macd_cross](../src/tradebot/strategies/macd_cross.py) | 🔴 **$4.99** | 🔴 $1.00 | 4,301 | 📉 -$995 | 100% ⚠️ | ▼ -9.47 [-16.93, -3.21] | ≈ +16.4pp [-1.4, +40.0] |
| 24 | [macd_rsi](../src/tradebot/strategies/macd_rsi.py) | 🔴 **$4.96** | 🔴 $0.94 | 2,454 | 📉 -$995 | 100% ⚠️ | ▼ -9.46 [-15.00, -4.40] | ▼ +16.3pp [+2.6, +39.6] |
| 25 | [attrition_reversion](../src/tradebot/strategies/attrition_reversion.py) | 🔴 **$4.94** | 🔴 $0.99 | 2,930 | 📉 -$995 | 100% ⚠️ | ▼ -9.47 [-14.55, -4.69] | ▼ +16.3pp [+3.1, +39.1] |
| 26 | [rsi_reversion](../src/tradebot/strategies/rsi_reversion.py) | 🔴 **$4.85** | 🔴 $0.77 | 4,464 | 📉 -$995 | 100% ⚠️ | ▼ -9.49 [-13.24, -5.78] | ▼ +16.7pp [+3.5, +39.8] |

_Balances from a $1,000 start · bold = the strategy's better market · 🟢 profit · 🔴 loss · 💀 liquidated · ⚠️ drawdown over 50%. Trades, profit and max drawdown describe that market._

_The last two columns are the only ones that answer **"is this difference real?"** Both are paired differences against `buy_and_hold` on spot over the full period (3,510 daily observations), each with a 95% stationary block-bootstrap interval — 30-day mean block, 2,000 resamples, the identical resample applied to both strategies so the market's own variance cancels instead of swamping the gap. ▲ / ▼ = the interval excludes zero and the strategy is better / worse; **≈ = it contains zero, so the difference from simply holding is not established**._

_**Growth**, not Sharpe, because final balance is what this table ranks by — and the two disagree. **spot**, because leveraged buy-and-hold is a stress case rather than a benchmark: it is liquidated in early 2017, and an account that cannot draw down further is not something to draw down less than (R-22). On this run **0 of 25** strategies are distinguishably better than holding on growth; the drawdown column is where the project's findings actually live._

_Adjacent steps down this ranking that survive the same test: **14 of 25** on spot · **5 of 25** on futures_5x. The order is a display convention, not a result — read the table as buckets._

_Regenerate with `python scripts/inference.py`; the numbers live in `reports/inference/bootstrap.csv`._

## Details per market and starting balance

### futures_5x · start balance $1,000

| strategy | final balance | profit | profit % | trades | win % | best trade | worst trade | max DD % | sharpe | in market % | fees | liq. | Δ sharpe vs hold | Δ max DD vs hold | Δ log growth vs hold | P(growth > hold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [kelly_regime_v4](../src/tradebot/strategies/kelly_regime_v4.py) | $156.2K | $155.2K | +15517.02% | 174 | 14.9 | $53.8K | -$8,817 | 35.3 | 1.59 | 66.4 | $9,532 |  | ☠ +1.89 [+0.97, +2.58] | ☠ -65.1pp [-70.7, +52.7] | ☠ +9.22 [+3.03, +19.34] | 1.00 |
| [kelly_regime_v3](../src/tradebot/strategies/kelly_regime_v3.py) | $139.5K | $138.5K | +13850.95% | 147 | 14.4 | $50.9K | -$9,152 | 41.8 | 1.55 | 66.5 | $8,396 |  | ☠ +1.87 [+0.90, +2.59] | ☠ -58.3pp [-69.4, +55.6] | ☠ +9.11 [+2.89, +19.14] | 1.00 |
| [kelly_regime_v2](../src/tradebot/strategies/kelly_regime_v2.py) | $122.0K | $121.0K | +12099.31% | 113 | 15.2 | $44.7K | -$9,232 | 39.6 | 1.49 | 67.3 | $8,222 |  | ☠ +1.79 [+0.82, +2.50] | ☠ -61.2pp [-68.4, +57.2] | ☠ +8.98 [+2.76, +18.93] | 1.00 |
| [kelly_regime](../src/tradebot/strategies/kelly_regime.py) | $108.2K | $107.2K | +10722.13% | 143 | 14.1 | $44.9K | -$6,782 | 42.6 | 1.42 | 66.3 | $8,000 |  | ☠ +1.72 [+0.78, +2.44] | ☠ -57.6pp [-65.8, +60.2] | ☠ +8.86 [+2.56, +18.86] | 1.00 |
| [kelly_regime_ev](../src/tradebot/strategies/kelly_regime_ev.py) | $108.0K | $107.0K | +10697.13% | 135 | 18.5 | $37.7K | -$7,065 | 36.8 | 1.53 | 62.2 | $4,278 |  | ☠ +1.84 [+0.90, +2.54] | ☠ -63.4pp [-71.6, +52.9] | ☠ +8.85 [+2.79, +18.88] | 1.00 |
| [kelly_regime_ev_fast](../src/tradebot/strategies/kelly_regime_ev.py) | $70.8K | $69.8K | +6978.05% | 50 | 36.0 | $22.8K | -$7,231 | 43.0 | 1.42 | 52.7 | $1,525 |  | ☠ +1.76 [+0.80, +2.49] | ☠ -57.3pp [-71.7, +53.5] | ☠ +8.43 [+2.33, +18.26] | 1.00 |
| [champions_council](../src/tradebot/strategies/champions_council.py) | $36.8K | $35.8K | +3577.35% | 261 | 22.3 | $18.6K | -$2,070 | 37.2 | 1.37 | 97.1 | $2,052 |  | ☠ +1.67 [+0.71, +2.40] | ☠ -62.9pp [-72.1, +52.0] | ☠ +7.78 [+1.90, +17.65] | 1.00 |
| [universal_kelly](../src/tradebot/strategies/universal_kelly.py) | $1,227 | $227 | +22.65% | 20 | 50.0 | $129 | -$15.25 | 9.0 | 0.46 | 30.0 | $1.18 |  | ☠ +0.82 [-0.03, +1.45] | ☠ -90.9pp [-94.8, +13.5] | ☠ +4.38 [-0.01, +13.41] | 0.97 |
| [harsanyi_crowd](../src/tradebot/strategies/harsanyi_crowd.py) | $429 | -$571 | -57.13% | 178 | 28.7 | $34.04 | -$73.42 | 57.7 | -0.86 | 0.2 | $308 |  | ☠ -0.67 [-1.49, -0.00] | ☠ -41.9pp [-61.0, +73.8] | ☠ +3.33 [-1.32, +12.38] | 0.64 |
| [hedge_experts](../src/tradebot/strategies/hedge_experts.py) | $258 | -$742 | -74.25% | 4103 | 11.4 | $53.0K | -$15.9K | 99.9 | 0.96 | 99.6 | $52.1K |  | ☠ +1.14 [+0.15, +1.88] | ☠ +0.9pp [-1.6, +100.0] | ☠ +2.82 [-10.84, +18.21] | 0.62 |
| [elliott_wave](../src/tradebot/strategies/elliott_wave.py) | $81.67 | -$918 | -91.83% | 164 | 51.8 | $512 | -$2,966 | 97.5 | 0.13 | 1.0 | $565 | yes | ☠ +0.49 [-0.12, +0.93] | ☠ -1.7pp [-35.0, +100.0] | ☠ +1.67 [-7.61, +10.85] | 0.66 |
| [overshoot_fade](../src/tradebot/strategies/overshoot_fade.py) | $33.52 | -$966 | -96.65% | 341 | 57.2 | $71.14 | -$115 | 97.0 | -1.16 | 0.3 | $309 |  | ☠ -0.92 [-1.75, -0.17] | ☠ -2.1pp [-14.5, +99.3] | ☠ +0.78 [-4.96, +11.00] | 0.53 |
| [buy_and_hold](../src/tradebot/strategies/buy_and_hold.py) | $18.05 | -$982 | -98.20% | 1 | 0.0 | -$982 | -$982 | 99.0 | -0.19 | 0.3 | $4.50 | yes | benchmark | benchmark | benchmark | 0.00 |
| [replicator_book](../src/tradebot/strategies/replicator_book.py) | $10.58 | -$989 | -98.94% | 1427 | 24.4 | $1,847 | -$681 | 99.8 | 0.33 | 99.6 | $2,215 |  | ☠ +0.54 [-0.24, +1.13] | ☠ +0.7pp [-4.9, +100.0] | ☠ -0.38 [-9.36, +11.50] | 0.43 |
| [minority_oracle](../src/tradebot/strategies/minority_oracle.py) | $3.83 | -$996 | -99.62% | 7065 | 25.4 | $11.19 | -$10.12 | 99.6 | -8.29 | 0.9 | $1,180 |  | ☠ -4.70 [-6.42, -3.15] | ☠ +0.6pp [-4.9, +100.0] | ☠ -1.39 [-8.39, +9.13] | 0.33 |
| [game_council](../src/tradebot/strategies/game_council.py) | $2.00 | -$998 | -99.80% | 2494 | 33.8 | $17.63 | -$55.75 | 99.8 | -1.66 | 2.5 | $297 |  | ☠ -1.84 [-2.93, -1.10] | ☠ +0.8pp [-1.4, +100.0] | ☠ -2.04 [-8.58, +8.12] | 0.27 |
| [macd_cross](../src/tradebot/strategies/macd_cross.py) | $1.00 | -$999 | -99.90% | 1464 | 27.6 | $566 | -$259 | 99.9 | -1.08 | 1.8 | $1,086 |  | ☠ -0.90 [-1.88, +0.00] | ☠ +0.9pp [-5.6, +100.0] | ☠ -2.71 [-10.40, +3.04] | 0.19 |
| [game_switch](../src/tradebot/strategies/game_switch.py) | $1.00 | -$999 | -99.90% | 6449 | 40.1 | $21.63 | -$17.26 | 99.9 | -1.60 | 1.2 | $1,167 |  | ☠ -1.48 [-3.00, -0.47] | ☠ +0.9pp [-5.1, +100.0] | ☠ -2.73 [-11.77, +8.46] | 0.26 |
| [tft_trend](../src/tradebot/strategies/tft_trend.py) | $1.00 | -$999 | -99.90% | 659 | 34.1 | $120 | -$125 | 99.9 | -1.03 | 0.7 | $465 |  | ☠ -1.17 [-2.12, -0.38] | ☠ +0.9pp [-13.1, +100.0] | ☠ -2.74 [-12.45, +7.29] | 0.27 |
| [regret_grid](../src/tradebot/strategies/regret_grid.py) | $1.00 | -$999 | -99.90% | 1389 | 9.4 | $33.17 | -$136 | 99.9 | -2.13 | 2.2 | $537 |  | ☠ -1.94 [-3.12, -0.81] | ☠ +0.9pp [-8.3, +100.0] | ☠ -2.74 [-10.98, +4.56] | 0.21 |
| [camouflage_flow](../src/tradebot/strategies/camouflage_flow.py) | $0.99 | -$999 | -99.90% | 1044 | 34.3 | $101 | -$205 | 99.9 | -0.64 | 3.3 | $190 |  | ☠ -0.69 [-1.57, +0.00] | ☠ +0.9pp [-1.4, +100.0] | ☠ -2.75 [-10.34, +7.47] | 0.24 |
| [attrition_reversion](../src/tradebot/strategies/attrition_reversion.py) | $0.99 | -$999 | -99.90% | 1176 | 60.4 | $103 | -$457 | 99.9 | -0.24 | 2.8 | $549 |  | ☠ -0.40 [-1.15, +0.04] | ☠ +0.9pp [-6.2, +100.0] | ☠ -2.75 [-11.32, +5.94] | 0.25 |
| [macd_rsi](../src/tradebot/strategies/macd_rsi.py) | $0.94 | -$999 | -99.91% | 1239 | 31.9 | $110 | -$150 | 99.9 | -0.94 | 1.8 | $709 |  | ☠ -1.01 [-1.85, -0.34] | ☠ +0.9pp [-5.2, +100.0] | ☠ -2.77 [-10.97, +5.38] | 0.23 |
| [flow_regime](../src/tradebot/strategies/flow_regime.py) | $0.80 | -$999 | -99.92% | 467 | 34.5 | $50.52 | -$107 | 99.9 | -1.04 | 1.3 | $102 |  | ☠ -1.01 [-2.02, -0.11] | ☠ +0.9pp [-5.3, +100.0] | ☠ -2.96 [-12.05, +8.21] | 0.26 |
| [rsi_reversion](../src/tradebot/strategies/rsi_reversion.py) | $0.77 | -$999 | -99.92% | 2264 | 63.1 | $246 | -$346 | 100.0 | 0.68 | 7.4 | $2,074 |  | ☠ +0.41 [-0.46, +1.01] | ☠ +0.9pp [-0.9, +100.0] | ☠ -2.82 [-12.45, +6.75] | 0.26 |
| [stealth_trend](../src/tradebot/strategies/stealth_trend.py) | $0.38 | -$1,000 | -99.96% | 248 | 24.6 | $37.26 | -$144 | 100.0 | -0.85 | 1.3 | $86.66 |  | ☠ -0.88 [-1.75, -0.04] | ☠ +0.9pp [-12.8, +100.0] | ☠ -3.71 [-15.04, +8.40] | 0.25 |

### spot · start balance $1,000

| strategy | final balance | profit | profit % | trades | win % | best trade | worst trade | max DD % | sharpe | in market % | fees | liq. | Δ sharpe vs hold | Δ max DD vs hold | Δ log growth vs hold | P(growth > hold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [kelly_regime_ev_fast](../src/tradebot/strategies/kelly_regime_ev.py) | $71.1K | $70.1K | +7010.24% | 34 | 44.1 | $20.5K | -$5,302 | 31.7 | 1.49 | 43.4 | $2,492 |  | ▲ +0.58 [+0.03, +1.10] | ▲ -52.9pp [-62.9, -20.7] | ≈ +0.11 [-3.08, +3.29] | 0.53 |
| [kelly_regime_v4](../src/tradebot/strategies/kelly_regime_v4.py) | $66.8K | $65.8K | +6579.39% | 174 | 14.4 | $21.7K | -$2,883 | 43.3 | 1.42 | 66.4 | $8,050 |  | ▲ +0.47 [+0.07, +0.87] | ▲ -41.1pp [-54.8, -18.4] | ≈ +0.04 [-2.60, +2.85] | 0.52 |
| [buy_and_hold](../src/tradebot/strategies/buy_and_hold.py) | $66.0K | $65.0K | +6504.41% | 1 | 0.0 | $65.0K | $65.0K | 84.1 | 0.95 | 100.0 | $1.00 |  | benchmark | benchmark | benchmark | 0.00 |
| [kelly_regime_v3](../src/tradebot/strategies/kelly_regime_v3.py) | $65.8K | $64.8K | +6484.44% | 147 | 14.4 | $23.7K | -$3,788 | 47.4 | 1.40 | 66.5 | $7,856 |  | ▲ +0.46 [+0.05, +0.87] | ▲ -36.8pp [-53.4, -16.1] | ≈ +0.03 [-2.54, +2.81] | 0.53 |
| [kelly_regime_v2](../src/tradebot/strategies/kelly_regime_v2.py) | $46.4K | $45.4K | +4542.30% | 113 | 15.2 | $13.9K | -$3,420 | 41.4 | 1.36 | 67.3 | $5,403 |  | ≈ +0.41 [-0.03, +0.84] | ▲ -43.4pp [-54.7, -15.3] | ≈ -0.32 [-3.15, +2.62] | 0.43 |
| [kelly_regime](../src/tradebot/strategies/kelly_regime.py) | $42.1K | $41.1K | +4109.63% | 143 | 14.1 | $14.5K | -$2,428 | 45.3 | 1.29 | 66.3 | $5,445 |  | ≈ +0.33 [-0.08, +0.75] | ▲ -38.9pp [-50.3, -12.2] | ≈ -0.42 [-3.08, +2.36] | 0.39 |
| [kelly_regime_ev](../src/tradebot/strategies/kelly_regime_ev.py) | $40.9K | $39.9K | +3986.30% | 81 | 25.9 | $15.1K | -$3,482 | 44.4 | 1.31 | 57.4 | $2,403 |  | ≈ +0.37 [-0.07, +0.81] | ▲ -40.0pp [-55.5, -16.3] | ≈ -0.45 [-3.28, +2.58] | 0.39 |
| [champions_council](../src/tradebot/strategies/champions_council.py) | $19.3K | $18.3K | +1832.14% | 131 | 14.6 | $8,520 | -$1,150 | 34.6 | 1.23 | 87.9 | $4,050 |  | ≈ +0.25 [-0.18, +0.68] | ▲ -49.5pp [-54.7, -19.0] | ≈ -1.20 [-4.06, +1.81] | 0.23 |
| [hedge_experts](../src/tradebot/strategies/hedge_experts.py) | $13.3K | $12.3K | +1227.66% | 2044 | 7.5 | $11.0K | -$1,469 | 59.3 | 0.87 | 88.0 | $16.7K |  | ≈ -0.07 [-0.42, +0.29] | ▲ -24.1pp [-39.1, -3.0] | ≈ -1.57 [-4.01, +0.96] | 0.11 |
| [replicator_book](../src/tradebot/strategies/replicator_book.py) | $2,330 | $1,330 | +132.98% | 713 | 26.2 | $451 | -$90.85 | 38.4 | 0.52 | 53.0 | $1,630 |  | ▼ -0.44 [-0.86, -0.06] | ▲ -47.1pp [-57.9, -20.7] | ≈ -3.31 [-6.86, +0.28] | 0.03 |
| [universal_kelly](../src/tradebot/strategies/universal_kelly.py) | $1,276 | $276 | +27.60% | 9 | 88.9 | $131 | -$6.99 | 7.4 | 0.62 | 22.7 | $1.33 |  | ≈ -0.28 [-0.88, +0.32] | ▲ -76.6pp [-89.1, -52.9] | ≈ -3.91 [-8.39, +0.44] | 0.04 |
| [harsanyi_crowd](../src/tradebot/strategies/harsanyi_crowd.py) | $888 | -$112 | -11.19% | 91 | 28.6 | $7.93 | -$13.89 | 11.4 | -0.82 | 0.1 | $90.02 |  | ▼ -2.03 [-2.96, -1.02] | ▲ -71.9pp [-85.6, -46.0] | ≈ -4.28 [-8.88, +0.23] | 0.03 |
| [overshoot_fade](../src/tradebot/strategies/overshoot_fade.py) | $662 | -$338 | -33.84% | 189 | 59.8 | $21.29 | -$52.43 | 37.1 | -0.95 | 0.2 | $214 |  | ▼ -2.02 [-2.89, -1.06] | ▲ -46.6pp [-68.5, -19.6] | ▼ -4.57 [-9.13, -0.07] | 0.02 |
| [camouflage_flow](../src/tradebot/strategies/camouflage_flow.py) | $548 | -$452 | -45.24% | 802 | 29.9 | $73.99 | -$40.11 | 52.7 | -0.66 | 2.4 | $950 |  | ▼ -1.73 [-2.75, -0.86] | ▲ -31.4pp [-57.7, -2.5] | ▼ -4.76 [-9.23, -0.29] | 0.02 |
| [stealth_trend](../src/tradebot/strategies/stealth_trend.py) | $465 | -$535 | -53.55% | 1605 | 30.5 | $109 | -$40.68 | 54.8 | -0.31 | 9.9 | $1,542 |  | ▼ -1.32 [-2.06, -0.62] | ≈ -29.0pp [-43.9, +12.1] | ▼ -4.92 [-9.26, -0.76] | 0.01 |
| [flow_regime](../src/tradebot/strategies/flow_regime.py) | $447 | -$553 | -55.33% | 1184 | 38.0 | $102 | -$52.56 | 56.1 | -0.72 | 3.5 | $966 |  | ▼ -1.73 [-2.67, -0.87] | ≈ -27.3pp [-47.2, +6.3] | ▼ -4.96 [-9.43, -0.54] | 0.02 |
| [game_council](../src/tradebot/strategies/game_council.py) | $284 | -$716 | -71.60% | 2541 | 25.9 | $22.92 | -$14.04 | 71.7 | -2.67 | 2.8 | $718 |  | ▼ -3.79 [-4.82, -2.85] | ≈ -11.5pp [-25.9, +12.4] | ▼ -5.42 [-9.97, -0.95] | 0.01 |
| [elliott_wave](../src/tradebot/strategies/elliott_wave.py) | $272 | -$728 | -72.82% | 1261 | 39.4 | $141 | -$113 | 81.0 | -0.57 | 7.6 | $2,097 |  | ▼ -1.63 [-2.43, -0.79] | ≈ -2.9pp [-33.6, +22.3] | ▼ -5.46 [-9.80, -1.00] | 0.01 |
| [minority_oracle](../src/tradebot/strategies/minority_oracle.py) | $53.36 | -$947 | -94.66% | 9039 | 12.2 | $3.00 | -$2.93 | 94.7 | -19.17 | 1.2 | $1,024 |  | ▼ -9.76 [-11.54, -8.09] | ≈ +11.5pp [-3.8, +35.9] | ▼ -7.09 [-11.60, -2.52] | 0.00 |
| [game_switch](../src/tradebot/strategies/game_switch.py) | $5.00 | -$995 | -99.50% | 6672 | 26.6 | $9.00 | -$11.38 | 99.5 | -6.19 | 1.2 | $1,264 |  | ▼ -5.98 [-7.95, -3.96] | ▼ +16.3pp [+1.0, +39.3] | ▼ -9.45 [-15.38, -4.06] | 0.00 |
| [regret_grid](../src/tradebot/strategies/regret_grid.py) | $5.00 | -$995 | -99.50% | 3461 | 8.4 | $13.29 | -$46.98 | 99.5 | -5.11 | 7.7 | $872 |  | ▼ -6.04 [-8.02, -4.03] | ▼ +16.3pp [+0.9, +39.9] | ▼ -9.46 [-16.20, -3.49] | 0.00 |
| [tft_trend](../src/tradebot/strategies/tft_trend.py) | $4.99 | -$995 | -99.50% | 2538 | 26.4 | $36.26 | -$92.03 | 99.5 | -2.78 | 3.1 | $883 |  | ▼ -4.10 [-5.23, -3.05] | ▼ +16.3pp [+3.0, +39.9] | ▼ -9.46 [-14.88, -4.58] | 0.00 |
| [macd_cross](../src/tradebot/strategies/macd_cross.py) | $4.99 | -$995 | -99.50% | 4301 | 30.9 | $42.19 | -$46.33 | 99.5 | -2.02 | 5.3 | $1,269 |  | ▼ -3.27 [-4.74, -1.85] | ≈ +16.4pp [-1.4, +40.0] | ▼ -9.47 [-16.93, -3.21] | 0.00 |
| [macd_rsi](../src/tradebot/strategies/macd_rsi.py) | $4.96 | -$995 | -99.50% | 2454 | 29.5 | $42.97 | -$44.90 | 99.5 | -2.56 | 3.8 | $1,103 |  | ▼ -4.03 [-5.28, -2.89] | ▼ +16.3pp [+2.6, +39.6] | ▼ -9.46 [-15.00, -4.40] | 0.00 |
| [attrition_reversion](../src/tradebot/strategies/attrition_reversion.py) | $4.94 | -$995 | -99.51% | 2930 | 58.6 | $28.72 | -$90.17 | 99.5 | -1.61 | 6.4 | $1,165 |  | ▼ -3.08 [-3.93, -2.20] | ▼ +16.3pp [+3.1, +39.1] | ▼ -9.47 [-14.55, -4.69] | 0.00 |
| [rsi_reversion](../src/tradebot/strategies/rsi_reversion.py) | $4.85 | -$995 | -99.51% | 4464 | 57.0 | $133 | -$159 | 99.8 | -1.07 | 14.5 | $4,882 |  | ▼ -2.60 [-3.37, -1.85] | ▼ +16.7pp [+3.5, +39.8] | ▼ -9.49 [-13.24, -5.78] | 0.00 |
