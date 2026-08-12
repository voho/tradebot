# Strategy comparison

Period: 2017-01-01 to 2026-08-12 (1,010,889 x 5m bars)  
Data: real, spot (perp proxy)  
Ranked by **final balance** (the primary comparison criterion); rows ordered by each strategy's best config.

| strategy | spot · $1K | spot · $1M | futures_5x · $1K | futures_5x · $1M |
|---|---|---|---|---|
| **buy_and_hold**<br>_Buy everything on the first bar and never trade again._<br>[source](../src/tradebot/strategies/buy_and_hold.py) | trades 1<br>profit $65.0K<br>worst $65.0K<br>best $65.0K<br>**after $66.0K** | trades 1<br>profit $65.04M<br>worst $65.04M<br>best $65.04M<br>**after $66.04M** | trades 1<br>profit -$982<br>worst -$982<br>best -$982<br>**after $18.05**<br>LIQUIDATED | trades 1<br>profit -$982.0K<br>worst -$982.0K<br>best -$982.0K<br>**after $18.0K**<br>LIQUIDATED |
| **rsi_reversion**<br>_Mean-reversion: buy oversold dips (RSI < 30), exit on recovery; mirror short overbought on futures._<br>[source](../src/tradebot/strategies/rsi_reversion.py) | trades 4,464<br>profit -$995<br>worst -$159<br>best $133<br>**after $4.85** | trades 5,713<br>profit -$1.00M<br>worst -$159.0K<br>best $133.5K<br>**after $365** | trades 2,264<br>profit -$999<br>worst -$346<br>best $246<br>**after $0.77** | trades 2,638<br>profit -$1.00M<br>worst -$345.7K<br>best $246.1K<br>**after $3.41**<br>LIQUIDATED |
| **macd_rsi**<br>_Trend + timing combo: trade RSI pullback recoveries only in the direction of the MACD trend._<br>[source](../src/tradebot/strategies/macd_rsi.py) | trades 2,454<br>profit -$995<br>worst -$44.90<br>best $42.97<br>**after $4.96** | trades 5,503<br>profit -$1.00M<br>worst -$44.9K<br>best $43.0K<br>**after $5.00** | trades 1,239<br>profit -$999<br>worst -$150<br>best $110<br>**after $0.94** | trades 2,120<br>profit -$1.00M<br>worst -$149.6K<br>best $110.2K<br>**after $0.98** |
| **macd_cross**<br>_Trend-following: long when MACD crosses above its signal line, flat/short on the cross below._<br>[source](../src/tradebot/strategies/macd_cross.py) | trades 4,301<br>profit -$995<br>worst -$46.33<br>best $42.19<br>**after $4.99** | trades 7,945<br>profit -$1.00M<br>worst -$46.3K<br>best $42.2K<br>**after $5.00** | trades 1,464<br>profit -$999<br>worst -$259<br>best $566<br>**after $1.00** | trades 3,306<br>profit -$1.00M<br>worst -$258.6K<br>best $566.0K<br>**after $0.73** |

## Details per market and starting balance

### futures_5x · start balance $1,000

| strategy | final balance | profit | profit % | trades | win % | best trade | worst trade | max DD % | sharpe | in market % | fees | liq. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [buy_and_hold](../src/tradebot/strategies/buy_and_hold.py) | $18.05 | -$982 | -98.20% | 1 | 0.0 | -$982 | -$982 | 99.0 | -0.19 | 0.3 | $4.50 | yes |
| [macd_cross](../src/tradebot/strategies/macd_cross.py) | $1.00 | -$999 | -99.90% | 1464 | 27.6 | $566 | -$259 | 99.9 | -1.08 | 1.8 | $1,086 |  |
| [macd_rsi](../src/tradebot/strategies/macd_rsi.py) | $0.94 | -$999 | -99.91% | 1239 | 31.9 | $110 | -$150 | 99.9 | -0.94 | 1.8 | $709 |  |
| [rsi_reversion](../src/tradebot/strategies/rsi_reversion.py) | $0.77 | -$999 | -99.92% | 2264 | 63.1 | $246 | -$346 | 100.0 | 0.68 | 7.4 | $2,074 |  |

### futures_5x · start balance $1.00M

| strategy | final balance | profit | profit % | trades | win % | best trade | worst trade | max DD % | sharpe | in market % | fees | liq. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [buy_and_hold](../src/tradebot/strategies/buy_and_hold.py) | $18.0K | -$982.0K | -98.20% | 1 | 0.0 | -$982.0K | -$982.0K | 99.0 | -0.19 | 0.3 | $4,499 | yes |
| [rsi_reversion](../src/tradebot/strategies/rsi_reversion.py) | $3.41 | -$1.00M | -100.00% | 2638 | 63.6 | $246.1K | -$345.7K | 100.0 | 0.53 | 8.6 | $2.07M | yes |
| [macd_rsi](../src/tradebot/strategies/macd_rsi.py) | $0.98 | -$1.00M | -100.00% | 2120 | 32.9 | $110.2K | -$149.6K | 100.0 | -1.26 | 3.2 | $710.4K |  |
| [macd_cross](../src/tradebot/strategies/macd_cross.py) | $0.73 | -$1.00M | -100.00% | 3306 | 29.6 | $566.0K | -$258.6K | 100.0 | -1.18 | 4.0 | $1.09M |  |

### spot · start balance $1,000

| strategy | final balance | profit | profit % | trades | win % | best trade | worst trade | max DD % | sharpe | in market % | fees | liq. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [buy_and_hold](../src/tradebot/strategies/buy_and_hold.py) | $66.0K | $65.0K | +6504.41% | 1 | 0.0 | $65.0K | $65.0K | 84.1 | 0.95 | 100.0 | $1.00 |  |
| [macd_cross](../src/tradebot/strategies/macd_cross.py) | $4.99 | -$995 | -99.50% | 4301 | 30.9 | $42.19 | -$46.33 | 99.5 | -2.02 | 5.3 | $1,269 |  |
| [macd_rsi](../src/tradebot/strategies/macd_rsi.py) | $4.96 | -$995 | -99.50% | 2454 | 29.5 | $42.97 | -$44.90 | 99.5 | -2.56 | 3.8 | $1,103 |  |
| [rsi_reversion](../src/tradebot/strategies/rsi_reversion.py) | $4.85 | -$995 | -99.51% | 4464 | 57.0 | $133 | -$159 | 99.8 | -1.07 | 14.5 | $4,882 |  |

### spot · start balance $1.00M

| strategy | final balance | profit | profit % | trades | win % | best trade | worst trade | max DD % | sharpe | in market % | fees | liq. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [buy_and_hold](../src/tradebot/strategies/buy_and_hold.py) | $66.04M | $65.04M | +6504.41% | 1 | 0.0 | $65.04M | $65.04M | 84.1 | 0.95 | 100.0 | $999 |  |
| [rsi_reversion](../src/tradebot/strategies/rsi_reversion.py) | $365 | -$1.00M | -99.96% | 5713 | 54.6 | $133.5K | -$159.0K | 100.0 | -1.63 | 18.6 | $4.89M |  |
| [macd_rsi](../src/tradebot/strategies/macd_rsi.py) | $5.00 | -$1.00M | -100.00% | 5503 | 26.5 | $43.0K | -$44.9K | 100.0 | -4.64 | 8.5 | $1.11M |  |
| [macd_cross](../src/tradebot/strategies/macd_cross.py) | $5.00 | -$1.00M | -100.00% | 7945 | 26.2 | $42.2K | -$46.3K | 100.0 | -3.80 | 9.8 | $1.27M |  |
