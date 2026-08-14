# Strategy comparison

Period: 2017-01-01 to 2026-08-12 (1,010,889 x 5m bars)  
Data: real, spot (perp proxy)  
Ranked by **final balance** (the primary comparison criterion); rows ordered by each strategy's best config.

| # | strategy | spot · $1K | spot · $1M | futures_5x · $1K | futures_5x · $1M | trades | profit | max DD |
|---|---|---|---|---|---|---|---|---|
| 1 | [kelly_regime](../src/tradebot/strategies/kelly_regime.py) | $42.1K | $42.10M | $108.2K | **$108.22M** | 143 | $107.22M | 43% |
| 2 | [buy_and_hold](../src/tradebot/strategies/buy_and_hold.py) | $66.0K | **$66.04M** | $18.05 ! | $18.0K ! | 1 | $65.04M | 84% |
| 3 | [champions_council](../src/tradebot/strategies/champions_council.py) | $19.3K | $19.32M | $36.8K | **$36.77M** | 263 | $35.77M | 37% |
| 4 | [hedge_experts](../src/tradebot/strategies/hedge_experts.py) | $13.3K | **$13.27M** | $258 | $257.4K | 2,159 | $12.27M | 59% |
| 5 | [replicator_book](../src/tradebot/strategies/replicator_book.py) | $2,330 | **$2.33M** | $10.58 | $10.6K | 717 | $1.33M | 38% |
| 6 | [universal_kelly](../src/tradebot/strategies/universal_kelly.py) | $1,276 | **$1.20M** | $1,227 | $1.00M | 1,529 | $202.9K | 7% |
| 7 | [harsanyi_crowd](../src/tradebot/strategies/harsanyi_crowd.py) | $888 | **$888.1K** | $429 | $428.7K | 91 | -$111.9K | 11% |
| 8 | [overshoot_fade](../src/tradebot/strategies/overshoot_fade.py) | $662 | **$661.6K** | $33.52 | $33.5K | 189 | -$338.4K | 37% |
| 9 | [camouflage_flow](../src/tradebot/strategies/camouflage_flow.py) | $548 | **$547.6K** | $0.99 | $127 | 802 | -$452.4K | 53% |
| 10 | [stealth_trend](../src/tradebot/strategies/stealth_trend.py) | $465 | **$464.5K** | $0.38 | $0.84 | 1,605 | -$535.5K | 55% |
| 11 | [flow_regime](../src/tradebot/strategies/flow_regime.py) | $447 | **$446.7K** | $0.80 | $1.46 | 1,184 | -$553.3K | 56% |
| 12 | [game_council](../src/tradebot/strategies/game_council.py) | $284 | **$284.0K** | $2.00 | $34.66 | 2,541 | -$716.0K | 72% |
| 13 | [minority_oracle](../src/tradebot/strategies/minority_oracle.py) | $53.36 | **$53.4K** | $3.83 | $4.29 | 9,039 | -$946.6K | 95% |
| 14 | [rsi_reversion](../src/tradebot/strategies/rsi_reversion.py) | $4.85 | **$365** | $0.77 | $3.41 ! | 5,713 | -$1.00M | 100% |
| 15 | [game_switch](../src/tradebot/strategies/game_switch.py) | **$5.00** | $4.95 | $1.00 | $1.00 | 6,672 | -$995 | 99% |
| 16 | [regret_grid](../src/tradebot/strategies/regret_grid.py) | **$5.00** | $5.00 | $1.00 | $1.00 | 3,461 | -$995 | 100% |
| 17 | [macd_rsi](../src/tradebot/strategies/macd_rsi.py) | $4.96 | **$5.00** | $0.94 | $0.98 | 5,503 | -$1.00M | 100% |
| 18 | [macd_cross](../src/tradebot/strategies/macd_cross.py) | $4.99 | **$5.00** | $1.00 | $0.73 | 7,945 | -$1.00M | 100% |
| 19 | [tft_trend](../src/tradebot/strategies/tft_trend.py) | **$4.99** | $4.97 | $1.00 | $0.94 | 2,538 | -$995 | 100% |
| 20 | [attrition_reversion](../src/tradebot/strategies/attrition_reversion.py) | $4.94 | **$4.97** | $0.99 | $0.95 | 7,221 | -$1.00M | 100% |

_Bold = the strategy's best config · `!` = liquidated. Trades, profit and max drawdown describe that best config; per-config detail is in the tables below._

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

### futures_5x · start balance $1.00M

| strategy | final balance | profit | profit % | trades | win % | best trade | worst trade | max DD % | sharpe | in market % | fees | liq. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [kelly_regime](../src/tradebot/strategies/kelly_regime.py) | $108.22M | $107.22M | +10722.13% | 143 | 14.1 | $44.94M | -$6.78M | 42.6 | 1.42 | 66.3 | $8.00M |  |
| [champions_council](../src/tradebot/strategies/champions_council.py) | $36.77M | $35.77M | +3577.29% | 263 | 22.1 | $18.57M | -$2.07M | 37.2 | 1.37 | 97.1 | $2.05M |  |
| [universal_kelly](../src/tradebot/strategies/universal_kelly.py) | $1.00M | $3,730 | +0.37% | 3047 | 3.8 | $4,923 | -$238 | 0.8 | 0.18 | 97.5 | $722 |  |
| [harsanyi_crowd](../src/tradebot/strategies/harsanyi_crowd.py) | $428.7K | -$571.3K | -57.13% | 178 | 28.7 | $34.0K | -$73.4K | 57.7 | -0.86 | 0.2 | $308.0K |  |
| [hedge_experts](../src/tradebot/strategies/hedge_experts.py) | $257.4K | -$742.6K | -74.26% | 4316 | 11.1 | $52.98M | -$15.87M | 99.9 | 0.96 | 99.8 | $52.05M |  |
| [overshoot_fade](../src/tradebot/strategies/overshoot_fade.py) | $33.5K | -$966.5K | -96.65% | 341 | 57.2 | $71.1K | -$115.2K | 97.0 | -1.16 | 0.3 | $309.1K |  |
| [buy_and_hold](../src/tradebot/strategies/buy_and_hold.py) | $18.0K | -$982.0K | -98.20% | 1 | 0.0 | -$982.0K | -$982.0K | 99.0 | -0.19 | 0.3 | $4,499 | yes |
| [replicator_book](../src/tradebot/strategies/replicator_book.py) | $10.6K | -$989.4K | -98.94% | 1435 | 24.3 | $1.85M | -$680.7K | 99.8 | 0.33 | 99.8 | $2.22M |  |
| [camouflage_flow](../src/tradebot/strategies/camouflage_flow.py) | $127 | -$1.00M | -99.99% | 1676 | 34.5 | $100.9K | -$204.8K | 100.0 | -0.87 | 5.1 | $191.3K |  |
| [game_council](../src/tradebot/strategies/game_council.py) | $34.66 | -$1.00M | -100.00% | 4978 | 31.8 | $17.6K | -$55.8K | 100.0 | -2.38 | 5.2 | $299.9K |  |
| [minority_oracle](../src/tradebot/strategies/minority_oracle.py) | $4.29 | -$1.00M | -100.00% | 15493 | 26.3 | $11.2K | -$10.1K | 100.0 | -13.34 | 2.0 | $1.18M |  |
| [rsi_reversion](../src/tradebot/strategies/rsi_reversion.py) | $3.41 | -$1.00M | -100.00% | 2638 | 63.6 | $246.1K | -$345.7K | 100.0 | 0.53 | 8.6 | $2.07M | yes |
| [flow_regime](../src/tradebot/strategies/flow_regime.py) | $1.46 | -$1.00M | -100.00% | 2135 | 38.6 | $50.5K | -$106.6K | 100.0 | -1.08 | 6.2 | $103.1K |  |
| [game_switch](../src/tradebot/strategies/game_switch.py) | $1.00 | -$1.00M | -100.00% | 11485 | 41.2 | $21.6K | -$17.3K | 100.0 | -2.94 | 2.1 | $1.17M |  |
| [regret_grid](../src/tradebot/strategies/regret_grid.py) | $1.00 | -$1.00M | -100.00% | 4027 | 9.7 | $33.2K | -$136.2K | 100.0 | -2.91 | 4.8 | $538.9K |  |
| [macd_rsi](../src/tradebot/strategies/macd_rsi.py) | $0.98 | -$1.00M | -100.00% | 2120 | 32.9 | $110.2K | -$149.6K | 100.0 | -1.26 | 3.2 | $710.4K |  |
| [attrition_reversion](../src/tradebot/strategies/attrition_reversion.py) | $0.95 | -$1.00M | -100.00% | 2034 | 61.6 | $103.1K | -$456.9K | 100.0 | -0.51 | 4.6 | $550.0K |  |
| [tft_trend](../src/tradebot/strategies/tft_trend.py) | $0.94 | -$1.00M | -100.00% | 1583 | 33.9 | $119.8K | -$125.2K | 100.0 | -1.13 | 1.7 | $465.7K |  |
| [stealth_trend](../src/tradebot/strategies/stealth_trend.py) | $0.84 | -$1.00M | -100.00% | 1285 | 30.1 | $37.3K | -$144.4K | 100.0 | -0.54 | 7.9 | $86.8K |  |
| [macd_cross](../src/tradebot/strategies/macd_cross.py) | $0.73 | -$1.00M | -100.00% | 3306 | 29.6 | $566.0K | -$258.6K | 100.0 | -1.18 | 4.0 | $1.09M |  |

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

### spot · start balance $1.00M

| strategy | final balance | profit | profit % | trades | win % | best trade | worst trade | max DD % | sharpe | in market % | fees | liq. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [buy_and_hold](../src/tradebot/strategies/buy_and_hold.py) | $66.04M | $65.04M | +6504.41% | 1 | 0.0 | $65.04M | $65.04M | 84.1 | 0.95 | 100.0 | $999 |  |
| [kelly_regime](../src/tradebot/strategies/kelly_regime.py) | $42.10M | $41.10M | +4109.63% | 143 | 14.1 | $14.50M | -$2.43M | 45.3 | 1.29 | 66.3 | $5.45M |  |
| [champions_council](../src/tradebot/strategies/champions_council.py) | $19.32M | $18.32M | +1832.12% | 132 | 14.5 | $8.52M | -$1.15M | 34.6 | 1.23 | 87.9 | $4.05M |  |
| [hedge_experts](../src/tradebot/strategies/hedge_experts.py) | $13.27M | $12.27M | +1227.50% | 2159 | 7.3 | $11.03M | -$1.47M | 59.3 | 0.87 | 88.1 | $16.70M |  |
| [replicator_book](../src/tradebot/strategies/replicator_book.py) | $2.33M | $1.33M | +132.99% | 717 | 26.1 | $451.4K | -$90.8K | 38.4 | 0.52 | 53.1 | $1.63M |  |
| [universal_kelly](../src/tradebot/strategies/universal_kelly.py) | $1.20M | $202.9K | +20.29% | 1529 | 2.7 | $123.9K | -$6,509 | 7.2 | 0.49 | 61.7 | $1,838 |  |
| [harsanyi_crowd](../src/tradebot/strategies/harsanyi_crowd.py) | $888.1K | -$111.9K | -11.19% | 91 | 28.6 | $7,926 | -$13.9K | 11.4 | -0.82 | 0.1 | $90.0K |  |
| [overshoot_fade](../src/tradebot/strategies/overshoot_fade.py) | $661.6K | -$338.4K | -33.84% | 189 | 59.8 | $21.3K | -$52.4K | 37.1 | -0.95 | 0.2 | $214.4K |  |
| [camouflage_flow](../src/tradebot/strategies/camouflage_flow.py) | $547.6K | -$452.4K | -45.24% | 802 | 29.9 | $74.0K | -$40.1K | 52.7 | -0.66 | 2.4 | $950.4K |  |
| [stealth_trend](../src/tradebot/strategies/stealth_trend.py) | $464.5K | -$535.5K | -53.55% | 1605 | 30.5 | $108.7K | -$40.7K | 54.8 | -0.31 | 9.9 | $1.54M |  |
| [flow_regime](../src/tradebot/strategies/flow_regime.py) | $446.7K | -$553.3K | -55.33% | 1184 | 38.0 | $101.6K | -$52.6K | 56.1 | -0.72 | 3.5 | $965.9K |  |
| [game_council](../src/tradebot/strategies/game_council.py) | $284.0K | -$716.0K | -71.60% | 2541 | 25.9 | $22.9K | -$14.0K | 71.7 | -2.67 | 2.8 | $717.8K |  |
| [minority_oracle](../src/tradebot/strategies/minority_oracle.py) | $53.4K | -$946.6K | -94.66% | 9039 | 12.2 | $2,995 | -$2,930 | 94.7 | -19.17 | 1.2 | $1.02M |  |
| [rsi_reversion](../src/tradebot/strategies/rsi_reversion.py) | $365 | -$1.00M | -99.96% | 5713 | 54.6 | $133.5K | -$159.0K | 100.0 | -1.63 | 18.6 | $4.89M |  |
| [regret_grid](../src/tradebot/strategies/regret_grid.py) | $5.00 | -$1.00M | -100.00% | 10834 | 4.7 | $13.3K | -$47.0K | 100.0 | -8.77 | 15.0 | $879.8K |  |
| [macd_rsi](../src/tradebot/strategies/macd_rsi.py) | $5.00 | -$1.00M | -100.00% | 5503 | 26.5 | $43.0K | -$44.9K | 100.0 | -4.64 | 8.5 | $1.11M |  |
| [macd_cross](../src/tradebot/strategies/macd_cross.py) | $5.00 | -$1.00M | -100.00% | 7945 | 26.2 | $42.2K | -$46.3K | 100.0 | -3.80 | 9.8 | $1.27M |  |
| [attrition_reversion](../src/tradebot/strategies/attrition_reversion.py) | $4.97 | -$1.00M | -100.00% | 7221 | 58.2 | $28.7K | -$90.2K | 100.0 | -2.80 | 15.9 | $1.17M |  |
| [tft_trend](../src/tradebot/strategies/tft_trend.py) | $4.97 | -$1.00M | -100.00% | 5939 | 24.8 | $36.3K | -$92.0K | 100.0 | -5.10 | 6.8 | $888.2K |  |
| [game_switch](../src/tradebot/strategies/game_switch.py) | $4.95 | -$1.00M | -100.00% | 13899 | 20.1 | $9,001 | -$11.4K | 100.0 | -11.46 | 2.5 | $1.27M |  |
