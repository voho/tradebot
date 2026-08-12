# Game theory & algorithmic trading — research notes

Literature survey behind the game-theoretic strategies in
`src/tradebot/strategies/`. Four areas were researched; each strategy's
docstring carries its own citations, and the mapping is at the bottom.
All input is 5m OHLCV bars only (no order book), so each mechanism below
is stated together with its bar-visible footprint.

## 1. Market microstructure games

- **Kyle (1985), "Continuous Auctions and Insider Trading," Econometrica 53(6)** —
  equilibrium of informed trader vs market makers vs noise traders. The
  insider trades gradually, hiding inside noise volume; price impact
  λ = information/noise ratio. *Footprint: persistent signed flow and
  durable drift; λ estimable per bar as |return|/dollar volume
  (Amihud 2002, J. Fin. Markets).*
- **Glosten & Milgrom (1985), JFE 14(1)** — bid/ask spread as a defense
  against adverse selection. *Footprint: when informed arrivals are few,
  wiggles mean-revert; when many, moves are informational. Bar-based
  spread estimate: Corwin & Schultz (2012), J. Finance 67(2).*
- **Easley, Kiefer, O'Hara & Paperman (1996), J. Finance 51(4)** — PIN, the
  probability a trade is informed. **Easley, López de Prado & O'Hara
  (2012), RFS 25(5)** — VPIN flow toxicity via **Bulk Volume
  Classification**: buy volume = V·Φ(Δp/σ) — computable from bars alone
  (confirmed superior in Easley, López de Prado & O'Hara 2016, JFE 120(2)).
- **Admati & Pfleiderer (1988), RFS 1(1)** — liquidity and informed trading
  pool into the same windows; judge volume against its normal level.
- **Brunnermeier & Pedersen (2005), J. Finance 60(4), "Predatory Trading"** —
  leader–follower game around a forced liquidator: price overshoots
  fundamentals, then recovers. *The one mean-reversion trade the
  microstructure games license: fade identified liquidity events after
  flow exhaustion.*
- **Yang & Zhu (2020), RFS 33(4), "Back-Running"** — following detected
  informed flow is equilibrium-consistent up to the leader's camouflage.

## 2. Minority games & evolutionary game theory

- **Arthur (1994), AER 84(2)** — El Farol: no deductive equilibrium; agents
  hold an ecology of predictors selected by recent fitness.
- **Challet & Zhang (1997), Physica A 246** — the Minority Game: ±1 choices,
  minority wins; strategy tables scored by virtual points over m-bit
  histories.
- **Savit, Manuca & Riolo (1999), PRL 82** and **Challet, Marsili & Zecchina
  (2000), PRL 84** — predictability phase transition at α = 2^m/N ≈ 0.34:
  above it, some history states carry exploitable conditional drift
  ("pockets of predictability", order parameter H).
- **Jefferies, Hart, Hui & Johnson (2001), EPJ B 20** — grand-canonical MG:
  agents abstain unless their edge clears a confidence threshold — the
  game-theoretic version of a fee filter. **Johnson, Lamper, Jefferies,
  Hart & Howison (2001), Physica A 299** — train the game on a real
  binarized price series and read its vote as a forecast. **Lamper,
  Howison & Johnson (2002), PRL 88** — ensemble agreement rises before
  large moves.
- **Marsili (2001), Physica A 299** — contrarian expectations ⇒ minority
  game, trend-following ⇒ majority game; real markets mix both.
  **Andersen & Sornette (2003), EPJ B 31, "The $-game"** — with real P&L
  payoffs agents switch opportunistically between reversion and momentum.
- **Taylor & Jonker (1978), Math. Biosci. 40** — replicator dynamics;
  **Lux & Marchesi (1999), Nature 397** — profit-contagion switching
  between chartists and fundamentalists generates fat tails and regimes;
  **Brock & Hommes (1998), JEDC 22** — logit switching with intensity of
  choice β (too high ⇒ instability; here: fee bleed).
- **Satinover & Sornette (2007), EPJ B 60** — "illusion of control":
  always playing the best-scoring rule can lose to score mean-reversion;
  score updates need decay.

## 3. No-regret learning, equilibria and growth optimality

- **Freund & Schapire (1997), JCSS 55(1)** — Hedge/multiplicative weights:
  within 2√(T ln N) of the best expert in hindsight on adversarial
  sequences (survey: Arora, Hazan & Kale 2012, Theory of Computing 8).
  **Herbster & Warmuth (1998), Machine Learning 32** — fixed-share
  tracking of a drifting best expert.
- **Hart & Mas-Colell (2000), Econometrica 68(5)** — regret matching: play
  ∝ positive regret; empirical play converges to correlated equilibrium
  (via Blackwell 1956 approachability). RM⁺ clipping: Tammelin (2014).
- **Brown (1951)** fictitious play; **Robinson (1951), Annals of Math 54** —
  FP converges in zero-sum games. **von Neumann (1928)** minimax;
  **Freund & Schapire (1999), GEB 29** — no-regret play attains at least
  the zero-sum game value (with a flat action, ≥ doing nothing).
- **Cover (1991), Mathematical Finance 1(1), "Universal Portfolios"** —
  wealth-weighted mixture over constant exposures matches the best fixed
  exposure in hindsight to O(log T), assumption-free; Dirichlet(½)
  refinement Cover & Ordentlich (1996, IEEE IT 42); exact minimax
  Ordentlich & Cover (1998, Math. OR 23); costs Blum & Kalai (1999,
  Machine Learning 35).
- **Kelly (1956), BSTJ 35**; **Breiman (1961)** — log-optimal growth.
  **Bell & Cover (1980), Math. OR 5(2)** — Kelly play is the equilibrium
  of the two-investor zero-sum game. **MacLean, Thorp & Ziemba (2010)** —
  fractional Kelly against estimation error.

## 4. Repeated games, attrition, beliefs and crowding

- **Axelrod (1984), The Evolution of Cooperation** — tit-for-tat wins
  repeated-PD tournaments: nice, retaliatory, forgiving, clear.
  **Friedman (1971), Rev. Econ. Studies 38** — grim-trigger equilibria.
  **Nowak & Sigmund (1992, Nature 355; 1993, Nature 364)** — generosity
  and win-stay/lose-shift beat strict TFT under noise. *Trading reading:
  forgiveness is turnover control; the no-trade band is the equilibrium
  object.*
- **Maynard Smith (1974), J. Theor. Biol. 47** — war of attrition: the ESS
  mixes quitting times so waiting cost matches the prize; **Fudenberg &
  Tirole (1986), Econometrica 54** — as time passes without the rival
  quitting, update beliefs that the rival is strong. *Trading reading:
  time-and-cost stops; non-reversion is information.*
- **Harsanyi (1967–68), Management Science 14** — games of incomplete
  information: estimate the opponent's hidden type by Bayesian updating;
  act on belief margins (with hysteresis, so fees are paid only on
  decisive belief moves).
- **Avellaneda & Stoikov (2008), Quantitative Finance 8(3)** — market
  making around a reservation price r = s − q·γ·σ²: fair value shifted
  against inventory; the optimal spread is the fee-aware no-trade band.
- **Cardaliaguet & Lehalle (2018), Math. Fin. Econ. 12(3)** — mean-field
  game of trade crowding: drift is crowd flow; aged, saturated trends
  (rising volume per unit of price progress) carry a strategic crowding
  cost.

## Strategy → grounding map

| strategy | grounded in |
|---|---|
| `camouflage_flow` | Kyle 1985; Easley–López de Prado–O'Hara 2012/2016 (BVC/VPIN); Yang & Zhu 2020 |
| `stealth_trend` | Kyle 1985; Amihud 2002; Admati & Pfleiderer 1988; Barclay & Warner 1993 |
| `overshoot_fade` | Brunnermeier & Pedersen 2005; Glosten & Milgrom 1985; Corwin & Schultz 2012 |
| `minority_oracle` | Challet & Zhang 1997; Savit et al. 1999; Jefferies et al. 2001 (GCMG); Johnson et al. 2001 |
| `replicator_book` | Taylor & Jonker 1978; Lux & Marchesi 1999; Brock & Hommes 1998; Maynard Smith & Price 1973 |
| `game_switch` | Brown 1951; Marsili 2001; Andersen & Sornette 2003 ($-game); Challet–Marsili–Zecchina 2000 |
| `hedge_experts` | Freund & Schapire 1997; Arora–Hazan–Kale 2012; Herbster & Warmuth 1998 |
| `regret_grid` | Hart & Mas-Colell 2000; Blackwell 1956; von Neumann 1928; Freund & Schapire 1999 |
| `universal_kelly` | Cover 1991; Cover & Ordentlich 1996; Bell & Cover 1980; MacLean–Thorp–Ziemba 2010 |
| `tft_trend` | Axelrod 1984; Nowak & Sigmund 1992; Friedman 1971 |
| `attrition_reversion` | Avellaneda & Stoikov 2008; Maynard Smith 1974; Fudenberg & Tirole 1986 |
| `harsanyi_crowd` | Harsanyi 1967–68; Cardaliaguet & Lehalle 2018 |

**Shared design lesson** from all four areas: Axelrod's forgiveness,
Maynard Smith's calibrated persistence, Harsanyi's belief hysteresis,
the GCMG abstention threshold and Avellaneda–Stoikov's spread are
mechanically the same object — a no-trade band sized so the expected
value of switching state exceeds the certain cost of switching. At a
0.1% round trip on 5m bars, that band decides viability.
