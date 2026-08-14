# Walk-forward validation & honest caveats

The comparison table ranks strategies on the **whole** 2017–2026 history.
That single number hides whether an edge is real or an artifact of one
lucky regime, so the leading strategies were re-run on a split:

- **In-sample (IS)**: 2017-01-01 → 2022-12-31 (631k bars) — contains the
  2018 bear (−84%) and the 2022 bear (−77%).
- **Out-of-sample (OOS)**: 2023-01-01 → 2026-08 (380k bars) — a strong
  bull with one ~54% drawdown, and no multi-year bear.

Starting balance $1,000, futures at 5x, spot at 1x.

| strategy | IS futures | OOS futures | OOS max DD | verdict |
|---|---|---|---|---|
| buy_and_hold (spot ref) | $17.2K | $3.8K | 54% | benchmark; liquidates on 5x futures |
| `kelly_regime` | $25.5K | $2.4K (+142%) | 41% | edge real, regime-dependent |
| `champions_council` | $12.9K | $1.9K (+87%) | 29% | lower return, lowest drawdown |

## What this actually says

**The regime filter's edge is concentrated in bear markets.** In-sample —
where two multi-year bears exist to be avoided — `kelly_regime` returns
about 1.5x buy-and-hold. Out-of-sample, in an almost uninterrupted bull,
it **lags** buy-and-hold on raw return (+142% vs +284%) while carrying
noticeably less risk (41% vs 54% max drawdown). That is the classic
trend-following payoff profile, and it is honest to state it plainly:
*this family does not beat holding in a steady bull; it earns its keep by
not participating in the collapses, and by surviving on leverage.*

**Leverage is where the difference compounds.** On 5x futures,
buy-and-hold is liquidated in the January 2017 crash and ends at $18. The
same market with regime-gated fractional-Kelly sizing ends the full
period at $108K from $1K, never liquidating, at Sharpe 1.42 — the highest
in the suite. Position sizing, not signal cleverness, produces that gap.

**Sizing beats prediction.** The three strategies that make money over the
decade (`kelly_regime`, `hedge_experts`, `replicator_book`,
`universal_kelly`) are all *allocators* — they decide how much to hold.
Every pure *predictor* in the suite (MACD/RSI baselines, the flow
followers, the minority-game oracle, the fictitious-play state machine)
loses after fees. On 5-minute BTC, the tradable game-theoretic content is
in growth-optimal sizing and no-regret allocation, not in forecasting the
next bar's sign.

## Parameter honesty

Deliberately **not** the tuned optimum. A sweep of single regime anchors
found 50 days best over the full period ($146K spot vs $66K
buy-and-hold), with 200 days at only $6K — that spread is exactly the
sensitivity that signals curve-fitting. So the shipped strategy:

- votes across **three** anchors (30/50/100 days) rather than picking the
  winner;
- targets **55% annualized volatility**, which is BTC's own long-run
  realized volatility, not a swept value;
- caps leverage at **2x**, comfortably inside fractional-Kelly practice
  (MacLean, Thorp & Ziemba 2010) rather than at the return-maximizing 3x
  (which reached $690K on futures and is not the default).

## Known limitations

- **No funding rates.** Perpetual futures pay/receive funding every 8
  hours; it is invisible in OHLCV and can meaningfully erode a
  held-long leveraged position. Treat futures figures as an upper bound.
- **Spot data as a perp proxy.** No perp series was reachable when the
  dataset was built (see README); the basis is small but the label
  `spot (perp proxy)` is carried through every report for a reason.
- **One asset, one decade.** BTC 2017–2026 is a single, upward-drifting
  sample path. Cross-asset (ETH) and cross-period validation would be the
  next honest step before risking capital.
- **Survivorship in the council.** `champions_council` selects members
  that already performed well on this data. Its OOS split is reported
  above precisely because its in-sample rank is not evidence.
