#!/usr/bin/env python
"""NOVEL branch, parallel round R-44: Hash Ribbons miner-capitulation as a
FOURTH, structurally independent vote inside kelly_regime_v4's own vote
mechanism -- not a post-hoc multiplier on top of it.

Idea, one sentence
------------------
"Hash Ribbons" (Charles Edwards, Capriole Investments, 2019 -- "Finding
Bitcoin Bottoms Using Miner Capitulation" / "Hash Ribbons and Bitcoin
Bottoms") is a supply-side, miner-economics signal built purely from BTC's
own mining hash rate (30d/60d SMA): when the 30d MA crosses below the 60d
MA, mining profitability has collapsed and inefficient miners are shutting
off ("capitulation"); when the 30d MA recovers back above the 60d MA,
capitulation is over -- historically associated with major cyclical
bottoms. This is added here as a fourth latched vote, precision-weighted
against v4's three existing price-anchor votes, so the vote-GENERATION
mechanism itself has one more genuine degree of freedom -- never touching
v3/v4's conditional-vol-target SIZE formula, which is copied verbatim.

Constraint attacked: INFO (one price series). This is the first genuinely
price-independent information channel in the project -- CoinMetrics'
on-chain hash rate, not derived from any traded price series (unlike every
prior signal in this repo, including R-41's real Deribit basis, which is
still a price transform).

Not a duplicate of
-------------------
- R-34 (harsanyi_crowd posterior as SIZE input), R-37/R-38 (retuned/
  formalized sizing constants), R-40 (anchor-ladder bagging/shrinkage),
  R-41 (basis magnitude brake / basis as early confirming vote): all six
  touch the SIZE/magnitude axis or re-derive from the SAME single spot
  price series. This branch touches neither -- it is a new fourth VOTE
  built from a genuinely different data source (mining hash rate, not
  price), combined into frac via precision-weighting rather than applied
  as a multiplier on top of v4's existing exposure.
- The conservative branch running in parallel this round (on-chain
  participation as a post-hoc confirmation multiplier): architecturally
  different by design -- this file never multiplies v4's target; it
  changes what feeds the vote itself, giving strictly more degrees of
  freedom to the mechanism than a bounded multiplier can. Not read, not
  coordinated with.
- B-07's own warning, sharpened by R-08 (a better volatility forecast made
  this strategy family WORSE, $52K vs $115K, by de-levering more promptly
  into BTC's highest-forward-Sharpe high-vol states -- Baur & Dimpfl 2018
  inverse leverage effect): the mechanism here is built to respect that
  sign discipline throughout -- the hash-ribbon vote only ever pushes
  EXPOSURE UP (recovery = bullish vote), the same direction v4's price
  anchors already push on a confirmed uptrend, never down in response to
  rising activity/volatility. See also Chi, Chu & Hao (2025, arXiv:
  2411.06327) on reading on-chain flows as confidence/regime signals
  rather than raw direction forecasts, consistent with how this vote is
  used here (a quantized regime confirmation, not a continuous overlay).

Pre-registered failure modes (named before any code ran)
------------------------------------------------------------
(a) The hash-ribbon vote fires so rarely (recovery-from-capitulation is a
    once-a-cycle event) that at any defensible weight it changes almost
    nothing -- "no effect", a legitimate negative finding.
(b) A naive, unweighted 4-way average lets 3 fast, noisy price anchors
    dominate a vote that is (by design) mostly static for years at a time,
    so it either does nothing (drowned out) or, if it happens to flip
    during a marginal 2/3-vs-1/3 anchor split, moves the vote a full
    quartile on a single, rare, low-precision daily event -- the "different
    failure mode than a real mechanism" named in this round's brief.
(c) The result is the standard exposure-level artifact: R^2 > 0.95 against
    a mean-notional-matched flat rescale of v4's own target.
(d) Fails the ETH falsification test (worse than v4 on ETH, or visibly
    worse on ETH than the identical pipeline's BTC control) -- ETH's own
    hash rate went proof-of-stake at the Merge (2022-09-15), so this test
    only has genuine HashRate coverage on the pre-Merge, pre-2020 Bitfinex
    ETH file this project already uses for falsification.

Usage
-----
    python experiments/kelly_regime_v10_hashribbon_vote.py descriptive  # step 2b
    python experiments/kelly_regime_v10_hashribbon_vote.py sweep        # step 3
    python experiments/kelly_regime_v10_hashribbon_vote.py select       # step 3 (validation)
    python experiments/kelly_regime_v10_hashribbon_vote.py artifact     # failure mode (c)
    python experiments/kelly_regime_v10_hashribbon_vote.py causality    # lookahead probe
    python experiments/kelly_regime_v10_hashribbon_vote.py eth          # failure mode (d)
    python experiments/kelly_regime_v10_hashribbon_vote.py all          # everything, in order
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
from tradebot.data import (  # noqa: E402
    align_onchain_causal, load_dataset, load_ohlcv_csv, load_onchain_metrics,
)
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.orders import Order  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures 5x", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")     # inner-train
VALID = ("2021-01-01", "2022-12-31")     # inner-validation
OOS_START = "2023-01-01"                 # never read in this file

INCUMBENT = "kelly_regime_v4"

N_EVALUATED = 0  # distinct (band, weight) configurations evaluated, project-trials count
_SEEN_CONFIGS: set[str] = set()


# --------------------------------------------------------------------- data


def build_hashribbon_dataframe(asset: str = "BTC") -> tuple[pd.DataFrame, str]:
    """Canonical spot OHLCV with a causal ``hashrate_visible`` column merged on.

    ``hashrate_visible`` is CoinMetrics' daily hash-rate estimate, aligned
    onto the 5m bar grid via ``tradebot.data.align_onchain_causal`` -- a
    day-D figure only becomes visible from D+1 00:00 UTC, exactly as that
    function's docstring specifies. Nothing is back-filled: bars before the
    first visible day are NaN, and the strategy below falls back to v4's
    exact unmodified vote wherever hash-rate data is unavailable (weight
    contribution is 0 when the column reads NaN throughout its own
    warmup window).
    """
    spot, label = load_dataset(ROOT / "data", "spot")
    onchain = load_onchain_metrics(ROOT / "data", asset)
    if onchain is None:
        raise FileNotFoundError(f"onchain data for {asset} not found under data/")
    aligned = align_onchain_causal(onchain[["HashRate"]], spot)
    out = spot.copy()
    out["hashrate_visible"] = aligned["HashRate"]
    return out, label


DF, LABEL = build_hashribbon_dataframe("BTC")
print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
      f"(data: {LABEL}); hashrate coverage {DF['hashrate_visible'].notna().sum():,} bars "
      f"from {DF['hashrate_visible'].dropna().index[0]:%Y-%m-%d}", file=sys.stderr)


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0,
            config_key: str | None = None):
    """One backtest -> Metrics. ``config_key`` counts a DISTINCT (band, weight)
    configuration exactly once across the whole session, however many
    market/period cells it is subsequently re-scored on (v4 control and
    diagnostic re-reads pass config_key=None and are never counted)."""
    global N_EVALUATED
    if config_key is not None and config_key not in _SEEN_CONFIGS:
        _SEEN_CONFIGS.add(config_key)
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                         start_balance=balance, data_label=LABEL)
    return compute_metrics(result), result


def line(tag, m, market_name=""):
    print(f"  {tag:38s} {market_name:10s} final=${m.final_balance:>11,.0f} "
          f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>7,.0f}{'  LIQUIDATED' if m.liquidated else ''}")


# ---------------------------------------------------------------- the vote


def _anchor_votes(close: pd.Series, horizons=(20, 40, 80), band: float = 0.01) -> dict:
    """Exactly kelly_regime_v4's own per-anchor latched vote."""
    votes = {}
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes[days] = v.ffill().fillna(0.0)
    return votes


def _hash_ribbon_vote(hashrate: pd.Series, capitulation_band: float,
                       ma_fast_days: int = 30, ma_slow_days: int = 60):
    """Latched Hash-Ribbons state (Edwards 2019): 0/1, hysteresis on the
    30d/60d hash-rate SMA ratio.

    Recovery (vote -> 1) fires at the literal Edwards crossover: the
    30d MA crossing back above the 60d MA (ratio > 1.0), exactly as
    described in the original indicator. Capitulation (vote -> 0, arming
    the NEXT recovery to be meaningful) requires the ratio to fall
    ``capitulation_band`` below 1.0, not merely dip under it -- this is
    the state-machine fix that stops the vote being "unconditional
    30d>60d": with capitulation_band=0 there is no dead zone at all and
    the vote collapses to the pointwise, memoryless "ratio>1" indicator
    (kept in the swept grid below, explicitly labelled, as the negative
    control the round's brief asked for). With capitulation_band>0 the
    vote only resets to "not yet recovered" after a genuinely deep hash-
    rate decline, so it stays latched bullish through the ordinary upward
    noise of a secularly growing hash rate, and a "recovery" is then a
    real, comparatively rare regime event, not a coin-flip around 1.0.

    Same np.where + ffill + fillna(0.0) latching style as
    ``_anchor_votes`` and ``kelly_regime.py``'s own vote -- the only
    difference is the ratio being thresholded (hash-rate SMA ratio vs.
    price-vs-anchor) and the asymmetric entry/exit levels.
    """
    ma_fast = hashrate.rolling(int(ma_fast_days * BARS_PER_DAY),
                                min_periods=int(ma_fast_days * BARS_PER_DAY)).mean()
    ma_slow = hashrate.rolling(int(ma_slow_days * BARS_PER_DAY),
                                min_periods=int(ma_slow_days * BARS_PER_DAY)).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ma_fast / ma_slow
    low = 1.0 - capitulation_band
    high = 1.0
    raw = np.where(ratio > high, 1.0, np.where(ratio < low, 0.0, np.nan))
    vote = pd.Series(raw, index=hashrate.index).ffill().fillna(0.0)
    return vote, ratio


class KellyRegimeV10HashribbonVote(KellyRegimeV3):
    """v4's 3-anchor vote plus a 4th, precision-weighted Hash-Ribbons miner-capitulation vote.

    Mechanism: ``frac = (anchor_vote_sum + hr_weight * hr_vote) / (3 +
    hr_weight)``, where ``anchor_vote_sum`` is v4's own three 0/1 latched
    price-anchor votes (unchanged) and ``hr_vote`` is the latched Hash-
    Ribbons state (0/1) above. ``hr_weight`` is a fixed, swept constant
    rather than 1.0 (a naive unweighted 4-way average) precisely because
    the hash-ribbon vote transitions roughly 15-40x less often than any
    single price anchor over the same window (see the ``descriptive``
    step) -- an inverse-variance/precision argument for giving it a
    FRACTION of one full vote's weight until it has more evidence behind
    it, the same N-is-small caution this project already applies to its
    own regime count. ``hr_weight=0`` recovers v4 exactly (identity
    check). Everything else -- v3/v4's conditional-vol-target scale, the
    2x cap, the 10% deadband -- is copied verbatim, unchanged.
    """

    name = "kelly_regime_v10_hashribbon_vote"
    warmup = 80 * BARS_PER_DAY + 10  # same as v4; dominates the 60-day hash-ribbon warmup

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 capitulation_band: float = 0.05, hr_weight: float = 0.33,
                 **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.capitulation_band = capitulation_band
        self.hr_weight = hr_weight

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = _anchor_votes(close, self.horizons, self.band)
        anchor_sum = sum(votes.values())  # 0..3 per bar
        n_anchors = float(len(votes))

        if "hashrate_visible" in df.columns and self.hr_weight > 0:
            hr_vote, hr_ratio = _hash_ribbon_vote(df["hashrate_visible"], self.capitulation_band)
        else:
            hr_vote = pd.Series(0.0, index=df.index)
            hr_ratio = pd.Series(np.nan, index=df.index)

        combined = (anchor_sum + self.hr_weight * hr_vote) / (n_anchors + self.hr_weight)
        frac = combined.to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        # From here down: byte-for-byte v3's conditional vol-targeting sizer,
        # unchanged -- only the vote fraction feeding it differs.
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

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        vstate = 0
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if vstate == 0:
                    vstate = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif vstate == 1 and x < self.high_out:
                    vstate = 0
                elif vstate == -1 and x > self.low_out:
                    vstate = 0
            scale = full[i] if vstate != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["v10_frac"] = frac
        df["v10_hr_vote"] = hr_vote.to_numpy()
        df["v10_hr_ratio"] = hr_ratio.to_numpy()
        df["v10_anchor_sum"] = anchor_sum.to_numpy()
        return df


# ------------------------------------------------------------- the grid


# capitulation_band=0.0 is the explicit negative control: no hysteresis dead
# zone at all, so the vote collapses to the naive, memoryless "ratio>1"
# reading the round's brief warned about. 0.05/0.08 require a genuinely
# deep hash-rate decline before the next recovery counts (see `descriptive`).
BANDS = (0.0, 0.05, 0.08)
# hr_weight=1.0 is the explicit negative control: a literal unweighted
# 4-way average. 0.15/0.33/0.5 are the precision-weighted candidates,
# justified by the transition-count comparison in `descriptive`.
WEIGHTS = (0.15, 0.33, 0.5, 1.0)


def _grid():
    out = []
    for band in BANDS:
        for weight in WEIGHTS:
            label = (f"band={band:.2f}"
                      f"{'(naive-nohys)' if band == 0.0 else ''}"
                      f" w={weight:.2f}{'(unweighted)' if weight == 1.0 else ''}")
            out.append((label, dict(capitulation_band=band, hr_weight=weight)))
    return out


def _config_key(kw: dict) -> str:
    return f"band={kw['capitulation_band']:.3f}|w={kw['hr_weight']:.3f}"


# ------------------------------------------------------- step 2b: descriptive


def descriptive() -> None:
    """Step 2b -- purely descriptive, no free parameter fitted: how often,
    and when, has BTC's hash-ribbon capitulation-recovery event actually
    fired on the inner-train window, and what did forward BTC returns look
    like right after it? Run BEFORE committing to the full strategy.

    Also reports how many times each of v4's own three price-anchor votes
    flips over the identical window, as the empirical basis for treating
    ``hr_weight`` as a small fraction of one full vote rather than 1.0.
    """
    lo, hi = TRAIN
    onchain = load_onchain_metrics(ROOT / "data", "BTC")
    hr = onchain["HashRate"].loc[lo:hi]
    daily_close = DF["close"].resample("1D").last().ffill().loc[lo:hi]

    print(f"descriptive window (inner-train): {lo} -> {hi}  ({len(hr):,} daily hash-rate rows)")
    print("\nprice-anchor vote transition counts over the SAME window (context for hr_weight):")
    close = DF["close"].loc[lo:hi]
    for days in (20, 40, 80):
        anchor = DF["close"].rolling(int(days * BARS_PER_DAY)).mean().loc[lo:hi]
        v = pd.Series(np.where(close > anchor * 1.01, 1.0,
                                np.where(close < anchor * 0.99, 0.0, np.nan)),
                      index=close.index).ffill().fillna(0.0)
        print(f"  {days:>3d}d price anchor: {int(v.ne(v.shift()).sum()):>4d} vote flips")

    for band in BANDS:
        ma_fast = hr.rolling(30, min_periods=30).mean()
        ma_slow = hr.rolling(60, min_periods=60).mean()
        ratio = ma_fast / ma_slow
        low, high = 1.0 - band, 1.0
        raw = np.where(ratio > high, 1.0, np.where(ratio < low, 0.0, np.nan))
        state = pd.Series(raw, index=hr.index).ffill().fillna(0.0)
        ups = state.index[(state == 1) & (state.shift() == 0)]
        downs = int(((state == 0) & (state.shift() == 1)).sum())
        label = f"capitulation_band={band:.2f}" + (" (naive, no hysteresis)" if band == 0.0 else "")
        print(f"\n{label}: {len(ups)} recovery event(s), {downs} capitulation-entry event(s) "
              f"in inner-train")
        for d in ups:
            visible = d + pd.Timedelta(days=1)  # causal: day D visible from D+1 00:00 UTC
            base = daily_close.asof(visible)
            fwd = {}
            for h in (30, 90, 180):
                px = daily_close.asof(visible + pd.Timedelta(days=h))
                fwd[h] = (px / base - 1.0) * 100.0 if pd.notna(px) and pd.notna(base) else float("nan")
            print(f"    event {d.date()} (visible {visible.date()}, price=${base:,.0f})  "
                  f"fwd30d={fwd[30]:>+7.1f}%  fwd90d={fwd[90]:>+7.1f}%  fwd180d={fwd[180]:>+7.1f}%")

    print("\nunconditional (unconditional-on-hash-ribbon) forward BTC return over the same window, for context:")
    for h in (30, 90, 180):
        fwd_ret = daily_close.pct_change(h).shift(-h)
        print(f"  {h:>3d}d: mean={fwd_ret.mean() * 100:>+7.1f}%  median={fwd_ret.median() * 100:>+7.1f}%")

    print("\ndescriptive step: 0 configurations counted toward N_EVALUATED (no free parameter fitted)")


# --------------------------------------------------------------- step 3: sweep


def sweep() -> None:
    """Step 3: every (band, weight) config on inner-train ONLY, spot primary."""
    print(f"\nINNER-TRAIN {TRAIN} / spot -- benchmarks:")
    for name in ("buy_and_hold", INCUMBENT):
        m, _ = measure(get_strategy(name), *TRAIN, market=SPOT)
        line(name, m, "spot")

    print(f"\nINNER-TRAIN {TRAIN} / spot -- candidate configurations:")
    for label, kw in _grid():
        m, _ = measure(KellyRegimeV10HashribbonVote(**kw), *TRAIN, market=SPOT,
                        config_key=_config_key(kw))
        line(label, m, "spot")

    print(f"\nconfigurations evaluated (step 3, distinct band/weight pairs): {N_EVALUATED}")


# -------------------------------------------------------------- step 3: select


def select() -> None:
    """Every config on inner-validation ONLY, BOTH markets, vs v4 control."""
    rows = []
    print(f"\nINNER-VALIDATION {VALID} -- v4 control:")
    ctl = {}
    for mname, market in MARKETS:
        m, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        ctl[mname] = m
        line(f"{INCUMBENT} (control)", m, mname)

    print(f"\nINNER-VALIDATION {VALID} -- candidate configurations:")
    best_label, best_kw, best_sharpe = None, None, -1e9
    for label, kw in _grid():
        cell = {}
        for mname, market in MARKETS:
            m, _ = measure(KellyRegimeV10HashribbonVote(**kw), *VALID, market=market,
                            config_key=_config_key(kw))
            cell[mname] = m
            line(label, m, mname)
            rows.append({"label": label, **kw, "market": mname,
                         "final": m.final_balance, "sharpe": m.sharpe,
                         "max_dd": m.max_drawdown_pct, "trades": m.num_trades})
        if cell["spot"].sharpe > best_sharpe:
            best_label, best_kw, best_sharpe = label, kw, cell["spot"].sharpe

    print(f"\nbest inner-validation SPOT Sharpe: {best_label}  ({best_sharpe:.2f})")
    print(f"v4 control spot Sharpe: {ctl['spot'].sharpe:.2f}   "
          f"v4 control futures Sharpe: {ctl['futures 5x'].sharpe:.2f}")

    print("\nside-by-side: best candidate's TRAIN window vs VALIDATION window (overfitting-signature check):")
    for mname, market in MARKETS:
        m_train, _ = measure(KellyRegimeV10HashribbonVote(**best_kw), *TRAIN, market=market)
        m_valid, _ = measure(KellyRegimeV10HashribbonVote(**best_kw), *VALID, market=market)
        m_train_v4, _ = measure(get_strategy(INCUMBENT), *TRAIN, market=market)
        m_valid_v4, _ = measure(get_strategy(INCUMBENT), *VALID, market=market)
        print(f"  {mname}:")
        print(f"    candidate  TRAIN final=${m_train.final_balance:>11,.0f}  "
              f"VALID final=${m_valid.final_balance:>11,.0f}")
        print(f"    v4         TRAIN final=${m_train_v4.final_balance:>11,.0f}  "
              f"VALID final=${m_valid_v4.final_balance:>11,.0f}")
        print(f"    candidate/v4 ratio:  TRAIN={m_train.final_balance / m_train_v4.final_balance:.3f}x"
              f"   VALID={m_valid.final_balance / m_valid_v4.final_balance:.3f}x")

    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")
    return best_label, best_kw


# ------------------------------------------------------------------ artifact


def artifact(kw: dict | None = None) -> None:
    """Mandatory exposure-artifact check: R^2 of the candidate's target
    series against a mean-notional-matched flat rescale of v4's own
    target, on inner-validation, both markets. R^2 > 0.95 -> artifact."""
    kw = kw or dict(capitulation_band=0.05, hr_weight=0.33)
    print(f"exposure-artifact check, candidate={kw}")
    v4 = get_strategy(INCUMBENT)
    for mname, market in MARKETS:
        cand = KellyRegimeV10HashribbonVote(**kw)
        lo = int(DF.index.searchsorted(VALID[0]))
        hi = int(DF.index.searchsorted(VALID[1], side="right"))
        prefix = min(lo, max(cand.warmup, v4.warmup))
        frame = DF.iloc[lo - prefix:hi]

        v4_prepared = v4.prepare(frame.copy())
        cand_prepared = cand.prepare(frame.copy())
        v4_t = v4_prepared["target"].to_numpy()[prefix:]
        cand_t = cand_prepared["target"].to_numpy()[prefix:]

        mean_abs_v4 = float(np.mean(np.abs(v4_t)))
        mean_abs_cand = float(np.mean(np.abs(cand_t)))
        alpha = mean_abs_cand / mean_abs_v4 if mean_abs_v4 > 0 else 0.0
        rescaled = alpha * v4_t
        ss_res = float(np.sum((cand_t - rescaled) ** 2))
        ss_tot = float(np.sum((cand_t - cand_t.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        corr = float(np.corrcoef(cand_t, v4_t)[0, 1])
        verdict = "EXPOSURE-LEVEL ARTIFACT" if r2 > 0.95 else "genuinely different exposure shape"
        print(f"  {mname:10s} mean|v4|={mean_abs_v4:.3f} mean|cand|={mean_abs_cand:.3f} "
              f"alpha={alpha:.3f}  R^2(cand vs alpha*v4)={r2:.4f}  raw corr={corr:.4f}  {verdict}")


# ------------------------------------------------------------------ causality


def causality(kw: dict | None = None) -> None:
    """Two-opposite-tampers probe: price AND the on-chain hash-rate input
    independently tampered after a cut. Every decision at or before the
    cut must be unchanged. Restricted to strictly pre-2023 bars."""
    kw = kw or dict(capitulation_band=0.05, hr_weight=0.33)

    pre2023 = DF[DF.index < OOS_START]
    df = pre2023.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]
    cut_day = df.index[cut].normalize()

    def run_probe(name, tamper_fn):
        up, down = df.copy(), df.copy()
        tamper_fn(up, down)

        def decisions(frame):
            s = KellyRegimeV10HashribbonVote(**kw)
            prepared = s.prepare(frame.copy())
            broker = _fresh_broker()
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        a, b = decisions(up), decisions(down)
        bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
        print(f"[{name}] tampered from bar {cut:,} of {len(df):,} "
              f"(calendar day {cut_day.date()}); checked bars {bars}")
        print("  FAIL - reads the future at bars " + str(bad) if bad
              else "  PASS - every decision at or before the cut is unchanged")

        pa = KellyRegimeV10HashribbonVote(**kw).prepare(up.copy())
        pb = KellyRegimeV10HashribbonVote(**kw).prepare(down.copy())
        for col in ("target", "v10_frac", "v10_hr_vote", "v10_anchor_sum"):
            diff = np.abs(pa[col].to_numpy()[:cut].astype(float)
                          - pb[col].to_numpy()[:cut].astype(float))
            worst = float(np.nanmax(diff))
            print(f"    column {col:16s} max |difference| before the cut = {worst:.3e}"
                  f"  {'PASS' if worst < 1e-9 else 'FAIL'}")

    def tamper_ohlcv(up, down):
        for col in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def tamper_hashrate(up, down):
        # Tamper the on-chain input from the CALENDAR DAY of the cut bar
        # forward -- align_onchain_causal's own +1-day lag means the true
        # cutover for a strategy decision is the day boundary, not the bar
        # index, so tampering at cut_day (rather than exactly at `cut`) is
        # the conservative, correctly-causal choice.
        col = up.columns.get_loc("hashrate_visible")
        mask = up.index >= cut_day
        up.iloc[mask, col] = up.iloc[mask, col].to_numpy() * 50.0     # extreme "recovered" reading
        down.iloc[mask, col] = down.iloc[mask, col].to_numpy() / 50.0  # extreme "capitulating" reading

    def tamper_both(up, down):
        tamper_ohlcv(up, down)
        tamper_hashrate(up, down)

    run_probe("PRICE tamper (standard)", tamper_ohlcv)
    run_probe("HASH-RATE tamper (the new, on-chain pathway)", tamper_hashrate)
    run_probe("both at once", tamper_both)


def _fresh_broker():
    from tradebot.broker import PaperBroker
    return PaperBroker(market=FUTURES, start_balance=10_000.0)


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Falsification test (pre-registered rule below, fixed before running).

    ETH-USD Bitfinex spot (2016-03 -> 2019-12-31) overlaps ETH's own
    on-chain HashRate coverage (2019-01-01 -> pre-Merge) for calendar-year
    2019 -- fully populated (0 NaN in that overlap), well before the
    2022-09-15 Merge that ended ETH mining. Every candidate config vs v4
    control, on BTC (control, identical pipeline/window) and ETH (test).

    PRE-REGISTERED RULE, fixed now, before results are read: if the
    candidate is not at least comparable to v4 on ETH, or is visibly worse
    on ETH than on the BTC control through the identical code, this
    direction fails.
    """
    eth_spot = load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz")
    eth_onchain = load_onchain_metrics(ROOT / "data", "ETH")
    if eth_onchain is None:
        raise FileNotFoundError("data/eth_onchain_daily.csv.gz not found")
    eth_aligned = align_onchain_causal(eth_onchain[["HashRate"]], eth_spot)
    eth_df = eth_spot.copy()
    eth_df["hashrate_visible"] = eth_aligned["HashRate"]

    overlap = eth_df["hashrate_visible"].dropna()
    print(f"ETH spot file: {len(eth_spot):,} bars  {eth_spot.index[0]:%Y-%m-%d} -> "
          f"{eth_spot.index[-1]:%Y-%m-%d}")
    print(f"ETH on-chain HashRate coverage overlapping ETH spot: {len(overlap):,} bars "
          f"visible from {overlap.index[0]:%Y-%m-%d} to {overlap.index[-1]:%Y-%m-%d}  "
          f"(NaN in this overlap: {eth_df.loc[overlap.index[0]:overlap.index[-1], 'hashrate_visible'].isna().sum()})")

    frames = {"BTC (control)": DF[DF.index < OOS_START], "ETH (test)": eth_df}
    primary_kw = dict(capitulation_band=0.05, hr_weight=0.33)
    results = {}
    for asset, frame in frames.items():
        print(f"\n{asset}  {len(frame):,} bars  {frame.index[0]:%Y-%m-%d} -> {frame.index[-1]:%Y-%m-%d}")
        results[asset] = {}
        for mname, market in MARKETS:
            m_v4, _ = measure(get_strategy(INCUMBENT), None, None, df=frame, market=market)
            line(f"{INCUMBENT} (control)", m_v4, mname)
            asset_results = {"v4": m_v4}
            for label, kw in _grid():
                cand = KellyRegimeV10HashribbonVote(**kw)
                m_c, _ = measure(cand, None, None, df=frame, market=market)
                line(f"v10[{label}]", m_c, mname)
                asset_results[label] = m_c
            results[asset][mname] = asset_results

    print("\nfalsification verdict per config (candidate/v4 final-balance ratio, BTC control vs ETH test):")
    any_fail = False
    for label, kw in _grid():
        for mname, market in MARKETS:
            btc_r = results["BTC (control)"][mname][label].final_balance / results["BTC (control)"][mname]["v4"].final_balance
            eth_r = results["ETH (test)"][mname][label].final_balance / results["ETH (test)"][mname]["v4"].final_balance
            worse_on_eth = eth_r < btc_r - 0.02  # small tolerance for noise
            eth_beats_v4 = eth_r >= 1.0
            flag = "FAIL" if (worse_on_eth and not eth_beats_v4) else ("caution" if worse_on_eth else "ok")
            any_fail = any_fail or flag == "FAIL"
            print(f"  {label:40s} {mname:10s} BTC ratio={btc_r:.3f}x  ETH ratio={eth_r:.3f}x  [{flag}]")
    print(f"\nprimary candidate (band=0.05, w=0.33) checked above; "
          f"overall falsification verdict: {'FAIL' if any_fail else 'no outright FAIL by the pre-registered rule'}")


# --------------------------------------------------------------------- driver


def all_checks() -> None:
    print("=" * 78)
    print("STEP 2b -- descriptive: hash-ribbon capitulation-recovery event study")
    print("=" * 78)
    descriptive()
    print("\n" + "=" * 78)
    print("STEP 3 -- sweep (inner-train)")
    print("=" * 78)
    sweep()
    print("\n" + "=" * 78)
    print("STEP 3 -- select (inner-validation, both markets)")
    print("=" * 78)
    select()
    print("\n" + "=" * 78)
    print("EXPOSURE-ARTIFACT CHECK")
    print("=" * 78)
    artifact()
    print("\n" + "=" * 78)
    print("CAUSALITY / NO-LOOKAHEAD PROBE")
    print("=" * 78)
    causality()
    print("\n" + "=" * 78)
    print("ETH FALSIFICATION TEST")
    print("=" * 78)
    eth()
    print(f"\ntotal distinct configurations evaluated (N_EVALUATED): {N_EVALUATED}")


if __name__ == "__main__":
    cmds = {"descriptive": descriptive, "sweep": sweep, "select": select,
            "artifact": artifact, "causality": causality, "eth": eth, "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/kelly_regime_v10_hashribbon_vote.py [{'|'.join(cmds)}]")
