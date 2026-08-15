# Strategy comparison

Period: 2017-01-01 to 2026-08-12 (1,010,889 x 5m bars)  
Data: real, spot (perp proxy)  
Ranked by **final balance** (the primary comparison criterion); rows ordered by each strategy's best config.

| # | strategy | spot | futures_5x | trades | profit | max DD |
|---|---|---|---|---|---|---|
| 🥇1 | [kelly_regime](../src/tradebot/strategies/kelly_regime.py) | 🟢 $42.1K | 🟢 **$108.2K** | 143 | 📈 $107.2K | 43% |
| 🥈2 | [buy_and_hold](../src/tradebot/strategies/buy_and_hold.py) | 🟢 **$66.0K** | 💀 $18.05 | 1 | 📈 $65.0K | 84% ⚠️ |
| 🥉3 | [champions_council](../src/tradebot/strategies/champions_council.py) | 🟢 $19.3K | 🟢 **$36.8K** | 261 | 📈 $35.8K | 37% |
| 4 | [hedge_experts](../src/tradebot/strategies/hedge_experts.py) | 🟢 **$13.3K** | 🔴 $258 | 2,044 | 📈 $12.3K | 59% ⚠️ |
| 5 | [replicator_book](../src/tradebot/strategies/replicator_book.py) | 🟢 **$2,330** | 🔴 $10.58 | 713 | 📈 $1,330 | 38% |
| 6 | [universal_kelly](../src/tradebot/strategies/universal_kelly.py) | 🟢 **$1,276** | 🟢 $1,227 | 9 | 📈 $276 | 7% |
| 7 | [harsanyi_crowd](../src/tradebot/strategies/harsanyi_crowd.py) | 🔴 **$888** | 🔴 $429 | 91 | 📉 -$112 | 11% |
| 8 | [overshoot_fade](../src/tradebot/strategies/overshoot_fade.py) | 🔴 **$662** | 🔴 $33.52 | 189 | 📉 -$338 | 37% |
| 9 | [camouflage_flow](../src/tradebot/strategies/camouflage_flow.py) | 🔴 **$548** | 🔴 $0.99 | 802 | 📉 -$452 | 53% ⚠️ |
| 10 | [stealth_trend](../src/tradebot/strategies/stealth_trend.py) | 🔴 **$465** | 🔴 $0.38 | 1,605 | 📉 -$535 | 55% ⚠️ |
| 11 | [flow_regime](../src/tradebot/strategies/flow_regime.py) | 🔴 **$447** | 🔴 $0.80 | 1,184 | 📉 -$553 | 56% ⚠️ |
| 12 | [game_council](../src/tradebot/strategies/game_council.py) | 🔴 **$284** | 🔴 $2.00 | 2,541 | 📉 -$716 | 72% ⚠️ |
| 13 | [minority_oracle](../src/tradebot/strategies/minority_oracle.py) | 🔴 **$53.36** | 🔴 $3.83 | 9,039 | 📉 -$947 | 95% ⚠️ |
| 14 | [game_switch](../src/tradebot/strategies/game_switch.py) | 🔴 **$5.00** | 🔴 $1.00 | 6,672 | 📉 -$995 | 99% ⚠️ |
| 15 | [regret_grid](../src/tradebot/strategies/regret_grid.py) | 🔴 **$5.00** | 🔴 $1.00 | 3,461 | 📉 -$995 | 100% ⚠️ |
| 16 | [tft_trend](../src/tradebot/strategies/tft_trend.py) | 🔴 **$4.99** | 🔴 $1.00 | 2,538 | 📉 -$995 | 100% ⚠️ |
| 17 | [macd_cross](../src/tradebot/strategies/macd_cross.py) | 🔴 **$4.99** | 🔴 $1.00 | 4,301 | 📉 -$995 | 100% ⚠️ |
| 18 | [macd_rsi](../src/tradebot/strategies/macd_rsi.py) | 🔴 **$4.96** | 🔴 $0.94 | 2,454 | 📉 -$995 | 100% ⚠️ |
| 19 | [attrition_reversion](../src/tradebot/strategies/attrition_reversion.py) | 🔴 **$4.94** | 🔴 $0.99 | 2,930 | 📉 -$995 | 100% ⚠️ |
| 20 | [rsi_reversion](../src/tradebot/strategies/rsi_reversion.py) | 🔴 **$4.85** | 🔴 $0.77 | 4,464 | 📉 -$995 | 100% ⚠️ |

_Balances from a $1,000 start · bold = the strategy's better market · 🟢 profit · 🔴 loss · 💀 liquidated · ⚠️ drawdown over 50%. Trades, profit and max drawdown describe that market._

## Details per market and starting balance

### futures_5x · start balance $1,000

| strategy | final balance | profit | profit % | trades | win % | best trade | worst trade | max DD % | sharpe | in market % | fees | liq. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [kelly_regime](../src/tradebot/strategies/kelly_regime.py) | $108.2K | $107.2K | +10722.13% | 143 | 14.1 | $44.9K | -$6,782 | 42.6 | 1.42 | 66.3 | $8,000 |  |
| [champions_council](../src/tradebot/strategies/champions_council.py) | $36.8K | $35.8K | +3577.35% | 261 | 22.3 | $18.6K | -$2,070 | 37.2 | 1.37 | 97.1 | $2,052 |  |
| [universal_kelly](../src/tradebot/strategies/universal_kelly.py) | $1,227 | $227 | +22.65% | 20 | 50.0 | $129 | -$15.25 | 9.0 | 0.46 | 30.0 | $1.18 |  |
| [harsanyi_crowd](../src/tradebot/strategies/harsanyi_crowd.py) | $429 | -$571 | -57.13% | 178 | 28.7 | $34.04 | -$73.42 | 57.7 | -0.86 | 0.2 | $308 |  |
| [hedge_experts](../src/tradebot/strategies/hedge_experts.py) | $258 | -$742 | -74.25% | 4103 | 11.4 | $53.0K | -$15.9K | 99.9 | 0.96 | 99.6 | $52.1K |  |
| [overshoot_fade](../src/tradebot/strategies/overshoot_fade.py) | $33.52 | -$966 | -96.65% | 341 | 57.2 | $71.14 | -$115 | 97.0 | -1.16 | 0.3 | $309 |  |
| [buy_and_hold](../src/tradebot/strategies/buy_and_hold.py) | $18.05 | -$982 | -98.20% | 1 | 0.0 | -$982 | -$982 | 99.0 | -0.19 | 0.3 | $4.50 | yes |
| [replicator_book](../src/tradebot/strategies/replicator_book.py) | $10.58 | -$989 | -98.94% | 1427 | 24.4 | $1,847 | -$681 | 99.8 | 0.33 | 99.6 | $2,215 |  |
| [minority_oracle](../src/tradebot/strategies/minority_oracle.py) | $3.83 | -$996 | -99.62% | 7065 | 25.4 | $11.19 | -$10.12 | 99.6 | -8.29 | 0.9 | $1,180 |  |
| [game_council](../src/tradebot/strategies/game_council.py) | $2.00 | -$998 | -99.80% | 2494 | 33.8 | $17.63 | -$55.75 | 99.8 | -1.66 | 2.5 | $297 |  |
| [macd_cross](../src/tradebot/strategies/macd_cross.py) | $1.00 | -$999 | -99.90% | 1464 | 27.6 | $566 | -$259 | 99.9 | -1.08 | 1.8 | $1,086 |  |
| [game_switch](../src/tradebot/strategies/game_switch.py) | $1.00 | -$999 | -99.90% | 6449 | 40.1 | $21.63 | -$17.26 | 99.9 | -1.60 | 1.2 | $1,167 |  |
| [tft_trend](../src/tradebot/strategies/tft_trend.py) | $1.00 | -$999 | -99.90% | 659 | 34.1 | $120 | -$125 | 99.9 | -1.03 | 0.7 | $465 |  |
| [regret_grid](../src/tradebot/strategies/regret_grid.py) | $1.00 | -$999 | -99.90% | 1389 | 9.4 | $33.17 | -$136 | 99.9 | -2.13 | 2.2 | $537 |  |
| [camouflage_flow](../src/tradebot/strategies/camouflage_flow.py) | $0.99 | -$999 | -99.90% | 1044 | 34.3 | $101 | -$205 | 99.9 | -0.64 | 3.3 | $190 |  |
| [attrition_reversion](../src/tradebot/strategies/attrition_reversion.py) | $0.99 | -$999 | -99.90% | 1176 | 60.4 | $103 | -$457 | 99.9 | -0.24 | 2.8 | $549 |  |
| [macd_rsi](../src/tradebot/strategies/macd_rsi.py) | $0.94 | -$999 | -99.91% | 1239 | 31.9 | $110 | -$150 | 99.9 | -0.94 | 1.8 | $709 |  |
| [flow_regime](../src/tradebot/strategies/flow_regime.py) | $0.80 | -$999 | -99.92% | 467 | 34.5 | $50.52 | -$107 | 99.9 | -1.04 | 1.3 | $102 |  |
| [rsi_reversion](../src/tradebot/strategies/rsi_reversion.py) | $0.77 | -$999 | -99.92% | 2264 | 63.1 | $246 | -$346 | 100.0 | 0.68 | 7.4 | $2,074 |  |
| [stealth_trend](../src/tradebot/strategies/stealth_trend.py) | $0.38 | -$1,000 | -99.96% | 248 | 24.6 | $37.26 | -$144 | 100.0 | -0.85 | 1.3 | $86.66 |  |

### spot · start balance $1,000

| strategy | final balance | profit | profit % | trades | win % | best trade | worst trade | max DD % | sharpe | in market % | fees | liq. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [buy_and_hold](../src/tradebot/strategies/buy_and_hold.py) | $66.0K | $65.0K | +6504.41% | 1 | 0.0 | $65.0K | $65.0K | 84.1 | 0.95 | 100.0 | $1.00 |  |
| [kelly_regime](../src/tradebot/strategies/kelly_regime.py) | $42.1K | $41.1K | +4109.63% | 143 | 14.1 | $14.5K | -$2,428 | 45.3 | 1.29 | 66.3 | $5,445 |  |
| [champions_council](../src/tradebot/strategies/champions_council.py) | $19.3K | $18.3K | +1832.14% | 131 | 14.6 | $8,520 | -$1,150 | 34.6 | 1.23 | 87.9 | $4,050 |  |
| [hedge_experts](../src/tradebot/strategies/hedge_experts.py) | $13.3K | $12.3K | +1227.66% | 2044 | 7.5 | $11.0K | -$1,469 | 59.3 | 0.87 | 88.0 | $16.7K |  |
| [replicator_book](../src/tradebot/strategies/replicator_book.py) | $2,330 | $1,330 | +132.98% | 713 | 26.2 | $451 | -$90.85 | 38.4 | 0.52 | 53.0 | $1,630 |  |
| [universal_kelly](../src/tradebot/strategies/universal_kelly.py) | $1,276 | $276 | +27.60% | 9 | 88.9 | $131 | -$6.99 | 7.4 | 0.62 | 22.7 | $1.33 |  |
| [harsanyi_crowd](../src/tradebot/strategies/harsanyi_crowd.py) | $888 | -$112 | -11.19% | 91 | 28.6 | $7.93 | -$13.89 | 11.4 | -0.82 | 0.1 | $90.02 |  |
| [overshoot_fade](../src/tradebot/strategies/overshoot_fade.py) | $662 | -$338 | -33.84% | 189 | 59.8 | $21.29 | -$52.43 | 37.1 | -0.95 | 0.2 | $214 |  |
| [camouflage_flow](../src/tradebot/strategies/camouflage_flow.py) | $548 | -$452 | -45.24% | 802 | 29.9 | $73.99 | -$40.11 | 52.7 | -0.66 | 2.4 | $950 |  |
| [stealth_trend](../src/tradebot/strategies/stealth_trend.py) | $465 | -$535 | -53.55% | 1605 | 30.5 | $109 | -$40.68 | 54.8 | -0.31 | 9.9 | $1,542 |  |
| [flow_regime](../src/tradebot/strategies/flow_regime.py) | $447 | -$553 | -55.33% | 1184 | 38.0 | $102 | -$52.56 | 56.1 | -0.72 | 3.5 | $966 |  |
| [game_council](../src/tradebot/strategies/game_council.py) | $284 | -$716 | -71.60% | 2541 | 25.9 | $22.92 | -$14.04 | 71.7 | -2.67 | 2.8 | $718 |  |
| [minority_oracle](../src/tradebot/strategies/minority_oracle.py) | $53.36 | -$947 | -94.66% | 9039 | 12.2 | $3.00 | -$2.93 | 94.7 | -19.17 | 1.2 | $1,024 |  |
| [game_switch](../src/tradebot/strategies/game_switch.py) | $5.00 | -$995 | -99.50% | 6672 | 26.6 | $9.00 | -$11.38 | 99.5 | -6.19 | 1.2 | $1,264 |  |
| [regret_grid](../src/tradebot/strategies/regret_grid.py) | $5.00 | -$995 | -99.50% | 3461 | 8.4 | $13.29 | -$46.98 | 99.5 | -5.11 | 7.7 | $872 |  |
| [tft_trend](../src/tradebot/strategies/tft_trend.py) | $4.99 | -$995 | -99.50% | 2538 | 26.4 | $36.26 | -$92.03 | 99.5 | -2.78 | 3.1 | $883 |  |
| [macd_cross](../src/tradebot/strategies/macd_cross.py) | $4.99 | -$995 | -99.50% | 4301 | 30.9 | $42.19 | -$46.33 | 99.5 | -2.02 | 5.3 | $1,269 |  |
| [macd_rsi](../src/tradebot/strategies/macd_rsi.py) | $4.96 | -$995 | -99.50% | 2454 | 29.5 | $42.97 | -$44.90 | 99.5 | -2.56 | 3.8 | $1,103 |  |
| [attrition_reversion](../src/tradebot/strategies/attrition_reversion.py) | $4.94 | -$995 | -99.51% | 2930 | 58.6 | $28.72 | -$90.17 | 99.5 | -1.61 | 6.4 | $1,165 |  |
| [rsi_reversion](../src/tradebot/strategies/rsi_reversion.py) | $4.85 | -$995 | -99.51% | 4464 | 57.0 | $133 | -$159 | 99.8 | -1.07 | 14.5 | $4,882 |  |
