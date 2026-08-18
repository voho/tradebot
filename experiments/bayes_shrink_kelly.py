"""Bayesian-uncertainty-shrunk Kelly fraction (backlog idea, this round's assignment).

Not registered: lives under ``experiments/`` so it is not auto-discovered,
per ROUTINE.md step 5. Promote into ``src/tradebot/strategies/`` only if it
clears the promotion bar.

The idea
--------
``kelly_regime`` (and v2/v3/v4) discount full Kelly by a FIXED constant —
``target_vol=0.55, max_leverage=2.0`` — applied uniformly at every bar,
regardless of whether the regime vote driving the position has recently
been reliable. MacLean, Thorp & Ziemba (2010) justify fractional Kelly
against estimation error in general, but the fraction here has never been
a function of how much estimation error is actually present *right now*.

This variant replaces that fixed discount with a continuously time-varying
multiplier kappa_t in (0, 1], derived from a Beta-Bernoulli posterior,
updated with exponential forgetting, over the recent hit-rate of the
anchor vote itself (Sukhov 2026's Bayesian-Kelly-under-parameter-
uncertainty framing, applied here to the regime vote rather than to the
win-probability of a single bet type; Busseti, Ryu & Boyd 2016's
risk-aware Kelly is the other point of the citation triangle, for putting
drawdown directly in the objective rather than trusting a fixed
fractional constant to do it indirectly).

kappa_t enters as ``desired = frac[i] * scale[i] * kappa[i]``, multiplying
the incumbent's existing product BEFORE the deadband is applied — see
``prepare()`` below, the line building ``desired``.

Constraint attacked: ERR (no error control anywhere in the signal path) —
the same constraint R-28 attacked, by a different mechanism (see below).

Not a duplicate of
-------------------
- ``kelly_regime_v2`` (L-03): reshapes the VOTE nonlinearly
  (``frac ** vote_gamma``), a static function of the current vote's
  composition. Nothing here is a function of the vote's HISTORY.
- ``kelly_regime_v3`` / ``kelly_regime_v4`` (L-02/L-01): change WHICH
  volatility regime re-sizes, or WHICH anchors vote. Neither touches
  confidence-weighting at all; this variant is built as a subclass that
  keeps v4's conditional-vol-targeting sizer AND its 20/40/80 anchors
  exactly as-is, and only inserts kappa_t as a further multiplier.
- **R-28 (e-process gate, B-01)** — the adjacent one, and the one this
  file must not blur into. R-28 ran an anytime-valid SEQUENTIAL HYPOTHESIS
  TEST: a nonnegative wealth martingale against the null "drift is zero",
  with a hard evidence THRESHOLD (``log(1/alpha)``) that the gate divides
  by to get a 0..1 "how close to significant" ratio. It is a testing
  framework wearing a continuous coat: the quantity that drives the gate
  is *evidence accumulated toward rejecting a null*, and the maths that
  license it (Ville's inequality, Type-I error control at arbitrary
  stopping times) are a hypothesis-testing guarantee.
  This file runs no test and has no null hypothesis. It tracks a
  posterior belief about a Bernoulli parameter (the vote's hit-rate) and
  uses the posterior's WIDTH — a credible-interval / variance object, not
  a p-value or an e-value — to shrink exposure continuously. There is no
  threshold anywhere in kappa_t's formula and no notion of "reject" or
  "fail to reject". The two mechanisms are related (both are
  estimation-risk-under-uncertainty devices, both attack ERR) but are
  mathematically different objects, and Section 3 below reports the
  empirical correlation between this file's kappa and R-28's frozen E1
  gate so the distinction is not just asserted.

Falsification test (pre-registered, Step 2, before any tuning)
----------------------------------------------------------------
Does the drawdown-reduction property (if any this strategy has) replicate
on ETH, same Bitfinex BTC/ETH window as R-17/R-28's P3 test
(2016-03-09 -> 2019-12-31)? Convention: BTC is the control (must also cut
drawdown, since it is the same asset the strategy was built on), ETH is
the test asset. The property under test is DRAWDOWN, stated in advance,
because that is what has repeatedly replicated in this project (L-01,
R-17, R-28) while RETURN has repeatedly not.

Stated prediction, written before the holdout was read
--------------------------------------------------------
P1 fails. N ~= 3: an honestly-uncertain Bayesian estimator of the vote's
own hit-rate will, on a mostly-one-way bull holdout, spend a long stretch
of its early history with kappa depressed simply because there has not
been enough decayed evidence yet, and BTC's few genuine regime EVENTS
(not bars) do not deliver enough labelled trials for the posterior width
to collapse quickly. Expect it to hold materially LESS mean exposure than
kelly_regime_v4 into the 2023+ bull, the same shape of failure as R-28,
even though the mechanism generating the caution is different (posterior
width vs an evidence threshold). Expect the drawdown cut to be real (this
is the "SIZE" pattern L-01..L-04 already showed for every honest
uncertainty discount tried on this signal) and the return shortfall to
be the thing that kills P1, again as in R-28.

Indexing / causality (read before trusting any number below)
----------------------------------------------------------------
The anchor vote ``frac[i]`` is causal at the close of bar i (uses
close[i] itself, exactly as kelly_regime/v2/v3/v4 already do — the
project's existing convention: decide on bar-close information, fill at
next open).

The Beta posterior tracks whether being long over bar (i-1 -> i) paid
off, i.e. it is a bet on ``frac[i-1] > 0`` (the vote LIVE going into bar
i), resolved by ``r[i] = log(close[i] / close[i-1])`` (the return realized
BY the close of bar i). That pairing is only knowable once bar i's close
prints, so the earliest bar at which the posterior may reflect this
particular win/loss is bar i itself — and ``kappa[i]`` (built from that
posterior) is what enters the decision at the close of bar i, which fills
at the open of bar i+1. Mapped onto the assignment's own notation (vote
made "at bar t", label "available after bar t+1's close", effect "from
bar t+2 onward at the earliest"): here t = i-1, the label is known at the
close of bar t+1 = i, and it first affects a fill at the open of bar
i+1 = t+2. No extra shift is needed beyond what the existing bar-close
-> next-bar-open contract already enforces, but see the ``causality()``
self-check below, which was run by hand (this file gets none of
``test_causality_strict.py``'s automatic protection) and verifies this by
direct tamper rather than by trusting the derivation above.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402


# --------------------------------------------------------------------- strategy


class KellyRegimeBayesShrink(KellyRegimeV4):
    """v4's anchors/vol-targeting, discounted by a Beta-posterior confidence in the vote.

    Everything about the regime vote (20/40/80-day doubling ladder,
    latched hysteresis) and the conditional volatility targeting
    (constant notional through normal vol, re-size only on breakout) is
    inherited unchanged from ``kelly_regime_v4``. The only addition is
    ``kappa_t``: a Beta(a,b) posterior over "was the vote's most recent
    directional call right", updated causally and with exponential
    forgetting (``halflife_days``) so it stays adaptive, whose WIDTH
    (posterior std) and MEAN jointly set exposure.

    kappa_t = clip(2*(mu_t - 0.5), 0, 1) * clip(1 - sigma_t / sigma_max, 0, 1)

    Trial resolution: DAILY, not per-5m-bar — this file's second documented
    deviation from a literal reading of the brief, discovered empirically
    (see ``_kappa`` docstring) rather than assumed. A per-bar Bernoulli
    trial measures whether bar i's individual 5-minute return was positive
    given the vote live going into it: on this data that hit-rate is
    49.92% conditional on the vote vs. 49.76% unconditional — indistinguishable
    from a coin flip, because BTC's day-to-day drift is a tiny fraction of
    its 5-minute noise. A posterior fit to that stream never leaves its
    uninformative prior: mean_factor requires mu appreciably above 0.5 and
    mu never gets there, so kappa is pinned near 0 forever and the strategy
    never trades (verified: every half-life from 5d to 180d, INNER-TRAIN
    AND INNER-VALIDATION, spot AND futures, zero fills). Resolving the
    trial once per DAY instead (was being long over calendar day D, given
    the vote live at D's start, profitable?) recovers a measurable signal:
    52.95% conditional vs. 52.04% unconditional — small, but no longer
    identical to noise. This also better matches what "N approx 3" means
    in this project's own standing diagnosis (regime EVENTS, not bars) and
    the 20-80 day anchor timescales the vote itself already operates on.

    Two factors, both required, both continuous, neither a threshold:

    - the WIDTH factor (as specified in this round's brief) starts at 0
      when the posterior is at its uninformative prior (sigma_t =
      sigma_max) and rises to 1 as evidence accumulates and the estimate
      sharpens;
    - the MEAN factor is this file's one documented deviation from the
      brief's single-factor starting design. A pure width-based kappa is
      fooled by a *confidently bad* vote: a hit-rate that has decayed to a
      precise 0.2 has LOW sigma (a skewed Beta is tighter than a centered
      one at the same pseudo-count) and would earn a HIGH kappa under
      width alone, releasing exposure exactly when the vote has been
      reliably wrong. The mean factor kills that failure mode: kappa is
      zero whenever the posterior mean hit-rate is at or below 0.5,
      regardless of how tight the posterior is. Both factors are built
      from the same posterior and both are continuous in its two moments,
      so this keeps the "continuous shrinkage from a posterior", not a
      gate — see the module docstring for the reasoning and R-28
      comparison.

    Beta prior: a0 = b0 = 1 (uniform / maximally uncertain), so
    sigma_max = sqrt(1*1 / (2**2 * 3)) = sqrt(1/12) ~= 0.2887 and kappa
    starts at exactly 0 (mu=0.5 AND sigma=sigma_max at time zero) — the
    strategy is flat until evidence has accumulated in BOTH senses, matching
    the brief: "kappa_t starts low (cautious) and rises only as the vote
    accumulates decayed evidence of being right."
    """

    name = "kelly_regime_bayes_shrink"

    def __init__(self, halflife_days: float = 30.0, prior_a: float = 1.0,
                 prior_b: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.halflife_days = halflife_days
        self.prior_a = prior_a
        self.prior_b = prior_b

    def _kappa(self, frac: np.ndarray, close: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
        """Beta-posterior confidence multiplier, broadcast onto every 5m bar.

        The Bernoulli trial is resolved once per CALENDAR DAY, not once per
        5-minute bar — see the class docstring for why: a per-bar trial
        measures 5-minute noise (49.92% conditional hit-rate, statistically
        a coin flip), not the vote's actual edge, which lives in
        day-to-day drift (52.95% conditional).

        For day D: ``elig[D]`` = the vote LIVE at the *start* of day D
        (frac at day D-1's last bar) was directional. ``win[D]`` = the
        close-to-close return over day D was positive. Both are known once
        day D's last bar closes — so the Beta posterior incorporating
        trial D is usable starting with day D+1's bars (one whole extra
        day of lag beyond the minimum the module docstring derives for a
        single-bar trial, deliberately conservative: "when in doubt, use
        an extra bar of lag"). ``kappa`` for every 5-minute bar in day
        D+1 is the constant value the posterior held right after day D's
        trial resolved; it does not update again until D+1 itself closes.
        """
        daily_close = close.resample("1D").last()
        daily_vote = pd.Series(frac, index=index).resample("1D").last()
        daily_ret = np.log(daily_close).diff()

        elig = (daily_vote.shift(1) > 0.0).to_numpy().copy()
        win = (daily_ret > 0.0).to_numpy()
        elig[0] = False  # no prior day to have been eligible on

        rho = 0.5 ** (1.0 / self.halflife_days)
        a0, b0 = self.prior_a, self.prior_b
        sigma_max = np.sqrt((a0 * b0) / ((a0 + b0) ** 2 * (a0 + b0 + 1.0)))

        n_days = len(daily_close)
        kappa_day = np.empty(n_days)
        av, bv = a0, b0
        for d in range(n_days):
            av *= rho
            bv *= rho
            if elig[d]:
                if win[d]:
                    av += 1.0
                else:
                    bv += 1.0
            mu = av / (av + bv)
            sigma = np.sqrt((av * bv) / ((av + bv) ** 2 * (av + bv + 1.0)))
            mean_factor = np.clip(2.0 * (mu - 0.5), 0.0, 1.0)
            width_factor = np.clip(1.0 - sigma / sigma_max, 0.0, 1.0)
            kappa_day[d] = mean_factor * width_factor

        # kappa_day[d] reflects the posterior right after day d's trial
        # resolved; it is valid for every bar in day d+1 onward, until the
        # next day's value supersedes it. Shift the day index forward by
        # one day, then forward-fill onto the 5-minute bar timestamps.
        valid_from = daily_close.index + pd.Timedelta(days=1)
        kappa_by_day = pd.Series(kappa_day, index=valid_from)
        kappa_bar = kappa_by_day.reindex(index, method="ffill")
        return kappa_bar.fillna(0.0).to_numpy()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        kappa = self._kappa(frac, close, df.index)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = frac[i] * scale * kappa[i]  # kappa enters HERE, before the deadband
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["kappa"] = kappa
        df["vote"] = frac
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# ------------------------------------------------------------------------ driver

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS = ("2023-01-01", None)

N_EVALUATED = 0  # distinct configurations evaluated in step 3, for deflated Sharpe


def ev(strategy, start, end, df=None, market=SPOT, tag="", balance=1_000.0, count=True):
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    if start is None and end is None:
        result = run_backtest(strategy, frame, market, balance, data_label=LABEL)
    else:
        result = run_period(strategy, frame, start, end, market=market,
                            start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    print(f"  {tag or strategy.name:34s} {market.name:9s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"fills={len(result.fills):>5d} fees=${m.fees_paid:>8,.0f} "
          f"DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f}{'  LIQUIDATED' if m.liquidated else ''}")
    return m


# The swept knob: decay half-life. 9 values, roughly log-spaced, chosen
# before looking at any result. Prior strength (a0=b0=1, uniform) is held
# fixed at its a-priori uninformative value and NOT swept, to keep the
# configuration count small per ROUTINE.md's "5-12 is plenty".
HALFLIVES = (5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 180.0)


def _benchmarks(start, end, market, label):
    print(f"\n{label} benchmarks:")
    for name in ("buy_and_hold", "kelly_regime_v4"):
        ev(get_strategy(name), start, end, market=market, tag=f"  {name}", count=False)


def sweep() -> None:
    """Step 3: sweep halflife_days on inner-train + inner-validation, both markets."""
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN, "INNER-TRAIN"), (VALID, "INNER-VALIDATION")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            print(f"{split} / {mname} bayes_shrink halflife sweep:")
            for hl in HALFLIVES:
                ev(KellyRegimeBayesShrink(halflife_days=hl), start, end, market=market,
                   tag=f"  hl={hl:g}d", count=(split == "INNER-VALIDATION" and mname == "spot"))
    print(f"\nconfigurations evaluated (distinct, counted once each): {N_EVALUATED}")


def inspect() -> None:
    """kappa's behaviour, mean exposure, and correlation with the incumbent vote / R-28's E1 gate."""
    s = KellyRegimeBayesShrink(halflife_days=10.0)
    prepared = s.prepare(DF.copy())
    kappa, vote = prepared["kappa"], prepared["vote"]

    print("kappa summary (full series):")
    print(f"  mean={kappa.mean():.3f}  median={kappa.median():.3f}  "
          f"fraction at exactly 0={ (kappa <= 1e-9).mean():.1%}  "
          f"fraction > 0.9={(kappa > 0.9).mean():.1%}")

    v4 = get_strategy("kelly_regime_v4")
    v4_prepared = v4.prepare(DF.copy())
    both = pd.DataFrame({"kappa": kappa, "vote": vote,
                         "v4_target": v4_prepared["target"],
                         "own_target": prepared["target"]}).dropna()
    print("\nkappa vs the incumbent's own (v4) latched anchor vote:")
    print(f"  correlation(kappa, vote)            {both['kappa'].corr(both['vote']):.3f}")
    print(f"  mean exposure |target| this strategy {both['own_target'].abs().mean():.3f}")
    print(f"  mean exposure |target| v4             {both['v4_target'].abs().mean():.3f}")
    ratio = both['own_target'].abs().mean() / max(both['v4_target'].abs().mean(), 1e-12)
    print(f"  -> holds {ratio:.2f}x v4's mean exposure")

    try:
        from experiments.eprocess_regime import EProcessRegime

        e1 = EProcessRegime(bet_halflife_days=20.0, gate=True, sizing="fixed")
        e1_prepared = e1.prepare(DF.copy())
        e1_gate = (e1_prepared["evidence"] / np.log(1.0 / e1.alpha)).clip(0, 1)
        cmp = pd.DataFrame({"kappa": kappa, "e1_gate": e1_gate}).dropna()
        print("\nkappa vs R-28's frozen E1 evidence gate (bet_halflife_days=20):")
        print(f"  correlation(kappa, e1_gate)         {cmp['kappa'].corr(cmp['e1_gate']):.3f}")
        print(f"  mean kappa   {cmp['kappa'].mean():.3f}")
        print(f"  mean e1_gate {cmp['e1_gate'].mean():.3f}")
    except Exception as exc:  # pragma: no cover - diagnostic path only
        print(f"\n(could not reconstruct R-28's E1 gate for comparison: {exc})")


def causality() -> None:
    """Two-opposite-tampers check, by hand (this file gets none of
    ``test_causality_strict.py``'s automatic protection since that suite
    only parametrizes over the *registered* strategy registry).

    Bars after a cut are multiplied by 3 in one copy and divided by 3 in
    the other; every decision, and the ``target``/``kappa``/``vote``
    columns, must be bit-identical before the cut in both copies.
    """
    from tradebot.broker import PaperBroker

    df = DF.iloc[-200_000:].copy()
    cuts = [len(df) - 5_000, len(df) // 2, 50_000]  # three different cut points
    FROZEN_HL = 30.0

    for cut in cuts:
        bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000) if cut - k >= 0]

        up, down = df.copy(), df.copy()
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

        def decisions(frame):
            s = KellyRegimeBayesShrink(halflife_days=FROZEN_HL)
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        a, b = decisions(up), decisions(down)
        bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
        print(f"cut at bar {cut:,} of {len(df):,}; checked bars {bars}")
        print("  FAIL - reads the future at bars " + str(bad) if bad
              else "  PASS - every decision at or before the cut is unchanged")

        pa = KellyRegimeBayesShrink(halflife_days=FROZEN_HL).prepare(up.copy())
        pb = KellyRegimeBayesShrink(halflife_days=FROZEN_HL).prepare(down.copy())
        for col in ("target", "kappa", "vote"):
            diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
            worst = float(np.nanmax(diff))
            print(f"    column {col:8s} max |difference| before the cut = {worst:.3e}"
                  f"  {'PASS' if worst < 1e-9 else 'FAIL'}")


# Frozen before the holdout was read. halflife_days=10 selected on
# inner-validation: best or near-best on BOTH markets there (futures
# +8.2%->this was hl=5's edge value, which train/validation disagreed on
# sharply -- see the report's sweep-table discussion; hl=10 is the most
# consistent value across train AND validation, avoiding both the noisy
# 15-30d neighbourhood, where inner-train futures DD spikes to 61-65%,
# and the edge at hl=5d). Prior stays at its a-priori uninformative
# Beta(1,1) default, never swept.
FROZEN = dict(halflife_days=10.0)


def holdout() -> None:
    """Step 4. Configuration frozen above; decision rule (P1-P4) is in the report."""
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"\nHOLDOUT 2023-01-01 -> / {mname}:")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            ev(get_strategy(name), *OOS, market=market, tag=f"  {name}", count=False)
        ev(KellyRegimeBayesShrink(**FROZEN), *OOS, market=market,
           tag="  bayes_shrink_kelly (FROZEN)", count=False)


def eth() -> None:
    """Pre-registered falsification: does the drawdown-cut property survive on ETH?

    Same venue (Bitfinex), same window, only the asset varies (R-17 /
    R-28's P3 design); BTC on this window is the control.
    """
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for market in (SPOT, FUTURES):
            for name in ("buy_and_hold", "kelly_regime_v4"):
                ev(get_strategy(name), None, None, df=df, market=market,
                   tag=f"  {name}", count=False)
            ev(KellyRegimeBayesShrink(**FROZEN), None, None, df=df, market=market,
               tag="  bayes_shrink_kelly (frozen)", count=False)


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"sweep": sweep, "inspect": inspect, "causality": causality,
            "holdout": holdout, "eth": eth}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/bayes_shrink_kelly.py [{'|'.join(cmds)}]")
