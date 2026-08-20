"""R-60 conservative branch: Kaufman Efficiency-Ratio-adaptive anchors for
`kelly_regime_v4`'s three-anchor vote (backlog **B-26**). Pre-registration:
`experiments/r60_shared.py` — read it first; this file implements its
"conservative branch" mechanism and nothing else.

Mechanism (the ONLY thing this branch changes)
------------------------------------------------
`KellyRegimeV3`/`KellyRegimeV4`'s vote uses a fixed-window simple moving
average as each of the three anchors: `close.rolling(d * BARS_PER_DAY).mean()`
for d in the horizon ladder (20/40/80 days for v4). This branch replaces
each anchor with a Kaufman (1995, "Smarter Trading") KAMA-style adaptive
average, bracketed around that SAME nominal horizon d, so the anchor speeds
up during efficient/trending price action and slows down during noisy/
choppy action — a structural, asset-agnostic adaptation (the formula is
identical for every asset; nothing here is fit per instrument):

    ER_t   = |close_t - close_{t-d*288}| / sum_{i=t-d*288+1..t} |close_i - close_{i-1}|
    fast_bars = (d/k) * 288,  slow_bars = (d*k) * 288
    fastSC = 2/(fast_bars+1), slowSC = 2/(slow_bars+1)
    SC_t   = (ER_t * (fastSC - slowSC) + slowSC) ** 2
    KAMA_t = KAMA_{t-1} + SC_t * (close_t - KAMA_{t-1})

seeded at the first bar where ER becomes valid (t = d*288), using that
bar's close as KAMA_0. `k` (the bracket-width multiplier) is the only free
parameter and is swept over a small grid {1.5, 2.0, 3.0} — applied
IDENTICALLY to every asset, never fit per instrument, keeping this branch
structural rather than fitted (see r60_shared.py's literature section).

Everything else is copied byte-for-byte from `KellyRegimeV3`/`KellyRegime`:
the 1% band, the latching hysteresis ("hold previous verdict inside the
band"), the vote-fraction averaging across the 3 anchors, and the entire
conditional-volatility-targeting / breakout-hysteresis sizing layer
(`target_vol`, `max_leverage`, the high/low breakout state machine). NONE
of that is touched — this branch is `KellyRegimeV3.prepare` with exactly
one block (the anchor construction) replaced.

Warmup, and why it is longer than v4's
---------------------------------------
A per-instance value, not the class default: `warmup = d_max*288*(1+2k) +
288` where `d_max = max(horizons)`. The first term (`d_max*288`) is the
bars needed before ER is even defined (the seed bar). The second
(`2*d_max*k*288`) gives the recursive filter two of its own slowest time
constants to let the seed's initial condition decay below ~2%
((1 - slowSC)^(2*slow_bars) = e^-4 ~ 1.8% for the standard sc=2/(N+1)
convention). The extra `+288` is one day of margin for the (much shorter)
volatility-targeting indicators inherited from v3. This is deliberately
per-instance (each swept `k` gets its own, shorter warmup for shorter `k`)
rather than a single warmup fixed at the grid's k_max for every instance —
using k_max everywhere would cost every non-maximal k an unnecessarily
cold, needlessly long prefix for no stabilization benefit; using each
instance's own k keeps every configuration's own stabilization guarantee
identical while not needlessly penalizing the smaller-k, faster-converging
brackets. Documented here per the task's request to record this judgment
call.

Implementation is a per-bar Python loop for the recursive KAMA update
(consistent with the existing style in kelly_regime.py's own sizing loop,
and safer than a vectorized recursion for a time-varying-alpha filter);
the Efficiency Ratio itself is vectorized (a rolling sum of absolute bar-
to-bar changes and a fixed-lag absolute price change, both causal).

No BTC or ETH bar past 2022-12-31 is ever loaded or backtested by this
module — both loaders are truncated immediately after reading, before any
other line of code touches the resulting frame.

Usage::

    uv run python experiments/r60_conservative_kama_anchors.py sweep
    uv run python experiments/r60_conservative_kama_anchors.py causality
    uv run python experiments/r60_conservative_kama_anchors.py run     # everything, writes report + CSVs
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.matched_hold import ConstantExposureHold, mean_notional  # noqa: E402
from experiments.r57_cross_asset_panel import (  # noqa: E402
    BOOT_KW,
    SPOT_BASE,
    SPOT_REAL,
    Asset,
    binomial_tail,
)
from experiments.r60_shared import (  # noqa: E402
    CONTROL,
    D2_REGRESSION_TOLERANCE_PP,
    PANEL_TEST,
    PANEL_TRAIN,
    R57_CONTROL_DD_ADVANTAGE,
    d1_verdict,
    d2_passes,
    load_panel,
    promoted,
)
from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.data import load_coinbase_spot, load_dataset  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns,
    max_drawdown_from_returns,
    paired_bootstrap,
    total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "reports" / "r60_conservative"
REPORT_PATH = ROOT / "experiments" / "reports" / "r60_conservative_report.md"

K_GRID = (1.5, 2.0, 3.0)
BOOT = dict(**BOOT_KW)

CONFIG_COUNT = 0


# ------------------------------------------------------------------ helpers


def measure(strategy, df, start, end, market):
    """One backtest. Every call is counted — there is no free evaluation."""
    global CONFIG_COUNT
    CONFIG_COUNT += 1
    result = run_period(strategy, df, start, end, market=market, start_balance=1_000.0)
    return result, compute_metrics(result)


def load_control_assets() -> tuple[Asset, Asset]:
    """BTC and ETH, truncated at CONTROL's end BEFORE any other line touches
    them, so no 2023+ bar of either is ever loaded into a variable this
    module can backtest (the round's holdout guarantee, enforced in code,
    not just by which window string is passed to run_period)."""
    end_ts = pd.Timestamp(CONTROL[1], tz="UTC")
    btc_df, _label = load_dataset(DATA_DIR, "spot")
    btc_df = btc_df[btc_df.index <= end_ts]
    eth_df = load_coinbase_spot(DATA_DIR, "ETH")
    eth_df = eth_df[eth_df.index <= end_ts]
    btc = Asset("BTC", btc_df, coverage=1.0, max_gap=pd.Timedelta(0), qualifies=True)
    eth = Asset("ETH", eth_df, coverage=1.0, max_gap=pd.Timedelta(0), qualifies=True)
    return btc, eth


# --------------------------------------------------------------- the strategy


def _efficiency_ratio(price: np.ndarray, n_bars: int) -> np.ndarray:
    """Kaufman's ER over a causal trailing window of ``n_bars`` bars.

    ER_t = |close_t - close_{t-n_bars}| / sum_{i=t-n_bars+1..t} |close_i - close_{i-1}|

    Both terms are causal (a fixed-lag difference and a rolling sum), so
    this is vectorized safely — only the KAMA recursion itself needs a
    per-bar loop. Undefined for t < n_bars (returned as 0.0, the "no
    trend" / slowest-response default, matching how the vote treats a
    NaN anchor: hold the previous verdict).
    """
    n = len(price)
    abs_diff = np.empty(n, dtype=float)
    abs_diff[0] = 0.0
    abs_diff[1:] = np.abs(np.diff(price))
    cum_abs = pd.Series(abs_diff).rolling(n_bars).sum().to_numpy()

    change = np.full(n, np.nan)
    if n > n_bars:
        change[n_bars:] = np.abs(price[n_bars:] - price[:-n_bars])

    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(cum_abs > 0, change / cum_abs, 0.0)
    er = np.nan_to_num(er, nan=0.0, posinf=0.0, neginf=0.0)
    return er


def kama_anchor(price: np.ndarray, days: int, k: float,
                bars_per_day: int = BARS_PER_DAY) -> np.ndarray:
    """Kaufman KAMA-style adaptive anchor bracketed around ``days``.

    Returns an array the same length as ``price``, NaN before the anchor
    is defined (t < days*bars_per_day — no ER, no seed yet).
    """
    n_bars = int(days * bars_per_day)
    fast_bars = (days / k) * bars_per_day
    slow_bars = (days * k) * bars_per_day
    fast_sc = 2.0 / (fast_bars + 1.0)
    slow_sc = 2.0 / (slow_bars + 1.0)

    n = len(price)
    kama = np.full(n, np.nan)
    if n <= n_bars:
        return kama

    er = _efficiency_ratio(price, n_bars)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    prev = float(price[n_bars])
    kama[n_bars] = prev
    for i in range(n_bars + 1, n):
        prev = prev + sc[i] * (price[i] - prev)
        kama[i] = prev
    return kama


class KellyRegimeV4KamaAnchors(KellyRegimeV3):
    """`kelly_regime_v4` with its 3 fixed-window SMA anchors replaced by
    Kaufman KAMA-style adaptive averages bracketed around the same nominal
    horizons (20/40/80 days). See module docstring for the mechanism and
    the warmup derivation. Plain `Strategy` subclass, NOT `@register`-ed —
    this is an R-60 experiment, not a registered strategy.

    Every other mechanism — the 1% latch band, the vote-fraction average,
    fractional-Kelly sizing, the conditional (breakout-only) volatility
    targeting inherited from `KellyRegimeV3`, the 2x leverage cap, the 10%
    deadband — is unchanged from `KellyRegimeV4`/`KellyRegimeV3`.
    """

    name = "kelly_regime_v4_kama_anchors"

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80),
                 k: float = 2.0, warmup_buffer_days: int = 1, **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        if k <= 1.0:
            raise ValueError(f"k (bracket-width multiplier) must be > 1.0, got {k!r}")
        self.k = float(k)
        d_max = max(self.horizons)
        # See module docstring: ER-validity prefix (d_max*288) + two of the
        # slowest bracket's own time constants (2*d_max*k*288) + 1 day margin.
        self.warmup = int(d_max * BARS_PER_DAY * (1 + 2 * self.k)) \
            + warmup_buffer_days * BARS_PER_DAY

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()
        price = close.to_numpy(dtype=float)

        # --- the ONLY changed block: KAMA-adaptive anchors instead of
        # fixed-window SMAs. Vote logic below this is byte-identical to
        # KellyRegime/KellyRegimeV3's own prepare().
        votes = []
        for days in self.horizons:
            anchor = pd.Series(kama_anchor(price, days, self.k), index=df.index)
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma  # convex confidence response

        # --- unchanged from KellyRegimeV3.prepare from here down ---
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
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
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
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df


# --------------------------------------------------------------- causality


def cmd_causality(k: float, probe_assets: list[Asset]) -> bool:
    """R-57's `cmd_causality` tamper probe, adapted: constructs
    `KellyRegimeV4KamaAnchors(k=k)` directly (this branch's candidate never
    goes through the registry). Two opposite tampers of post-cut bars
    (x3/÷7 and ÷3x7); pre-cut decisions must be byte-identical or the
    round stops here."""
    print("=" * 100)
    print(f"CAUSALITY TAMPER PROBE — kelly_regime_v4_kama_anchors(k={k})")
    print("=" * 100)
    market = MarketSpec.futures(leverage=5.0)
    all_ok = True
    for a in probe_assets:
        tail = a.df.iloc[-60_000:].copy()
        cut = len(tail) - 5_000
        bars = [cut - j for j in (1, 2, 3, 5, 10, 20)]

        variants = {
            "x3/vol x7": (3.0, 7.0),
            "div3/vol div7": (1.0 / 3.0, 1.0 / 7.0),
        }
        tampered = {}
        for label, (price_mult, vol_mult) in variants.items():
            frame = tail.copy()
            for col in ("open", "high", "low", "close"):
                frame.iloc[cut:, frame.columns.get_loc(col)] *= price_mult
            frame.iloc[cut:, frame.columns.get_loc("volume")] *= vol_mult
            tampered[label] = frame

        def decisions(frame):
            s = KellyRegimeV4KamaAnchors(k=k)
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=market, start_balance=10_000.0)
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        up = decisions(tampered["x3/vol x7"])
        down = decisions(tampered["div3/vol div7"])
        ok = all(x == y for x, y in zip(up, down))
        all_ok = all_ok and ok
        print(f"  {a.ticker:5s} decisions identical under opposite post-cut "
              f"tampers (x3/vol x7 vs div3/vol div7): {'PASS' if ok else 'FAIL'}")
    return all_ok


# --------------------------------------------------------------------- cells


def cell(a: Asset, strategy, window, market, label: str, rows: list) -> dict:
    """One asset x window x market cell: candidate, buy_and_hold, matched
    hold, paired-bootstrap intervals. Identical structure to R-57/R-59's
    `cell()`, parameterized on a strategy INSTANCE."""
    start, end = window
    cand_res, cand = measure(strategy, a.df, start, end, market)
    hold_res, hold = measure(get_strategy("buy_and_hold"), a.df, start, end, market)

    c_mean = mean_notional(cand_res)
    mh_res, mh = measure(ConstantExposureHold(c_mean), a.df, start, end, market)

    cand_ret = daily_returns(cand_res.equity).to_numpy(dtype=float)
    mh_ret = daily_returns(mh_res.equity).to_numpy(dtype=float)
    hold_ret = daily_returns(hold_res.equity).to_numpy(dtype=float)
    n = min(len(cand_ret), len(mh_ret), len(hold_ret))
    dd_matched = paired_bootstrap(cand_ret[:n], mh_ret[:n], max_drawdown_from_returns, **BOOT)
    growth_matched = paired_bootstrap(cand_ret[:n], mh_ret[:n], total_log_return, **BOOT)

    row = {
        "asset": a.ticker, "window": label, "market": market.name,
        "fee": market.fee_rate,
        "cand_final": cand.final_balance, "cand_dd": cand.max_drawdown_pct,
        "cand_sharpe": cand.sharpe, "cand_trades": cand.num_trades,
        "cand_liq": cand.liquidated,
        "hold_final": hold.final_balance, "hold_dd": hold.max_drawdown_pct,
        "hold_sharpe": hold.sharpe, "hold_liq": hold.liquidated,
        "c_mean_notional": c_mean,
        "mh_final": mh.final_balance, "mh_dd": mh.max_drawdown_pct,
        "mh_sharpe": mh.sharpe,
        "dd_matched_diff": dd_matched.diff.point,
        "dd_matched_lo": dd_matched.diff.lo, "dd_matched_hi": dd_matched.diff.hi,
        "growth_matched_diff": growth_matched.diff.point,
        "growth_matched_lo": growth_matched.diff.lo,
        "growth_matched_hi": growth_matched.diff.hi,
    }
    rows.append(row)
    print(f"  {a.ticker:5s} {label:9s} {market.name:11s} fee={market.fee_rate:.2%}  "
          f"cand ${cand.final_balance:>10,.0f} DD {cand.max_drawdown_pct:5.1f}% | "
          f"hold ${hold.final_balance:>10,.0f} DD {hold.max_drawdown_pct:5.1f}% | "
          f"matched(c={c_mean:.2f}) ${mh.final_balance:>10,.0f} "
          f"DD {mh.max_drawdown_pct:5.1f}% | "
          f"dDD_matched {dd_matched.diff.point:+6.1f}pp "
          f"[{dd_matched.diff.lo:+6.1f},{dd_matched.diff.hi:+6.1f}]")
    return row


# ------------------------------------------------------------------- sweep


def cmd_sweep(panel: list[Asset]) -> tuple[dict[float, list[dict]], dict[float, int]]:
    """D1 methodology (PANEL_TRAIN, spot @0.10%, matched-exposure drawdown
    count) evaluated for every k in K_GRID. Selection happens ONLY on
    PANEL_TRAIN — this is the sweep, counted honestly."""
    print("=" * 100)
    print(f"K-GRID SWEEP — PANEL_TRAIN, spot @0.10%, k in {K_GRID}")
    print("=" * 100)
    sweep_rows: dict[float, list[dict]] = {}
    k1_by_k: dict[float, int] = {}
    for k in K_GRID:
        rows: list[dict] = []
        print(f"\n-- k={k} --")
        for a in panel:
            strat = KellyRegimeV4KamaAnchors(k=k)
            cell(a, strat, PANEL_TRAIN, SPOT_BASE, "PANEL_TRAIN", rows)
        k1 = int(sum(1 for r in rows if r["cand_dd"] < r["mh_dd"]))
        sweep_rows[k] = rows
        k1_by_k[k] = k1
        print(f"  k={k}: D1 matched-exposure drawdown count = {k1}/{len(panel)} "
              f"-> {d1_verdict(k1, len(panel))}")
    return sweep_rows, k1_by_k


def select_k(k1_by_k: dict[float, int]) -> float:
    """Best D1 count wins; ties broken toward the middle of the grid (k=2.0)
    as the least extreme choice, since a tie means the data does not
    distinguish the brackets and picking the middle is the more
    conservative, less cherry-picked choice."""
    best_k1 = max(k1_by_k.values())
    tied = [k for k, v in k1_by_k.items() if v == best_k1]
    if len(tied) == 1:
        return tied[0]
    return min(tied, key=lambda k: abs(k - 2.0))


# ---------------------------------------------------------------------- run


def cmd_run() -> None:
    panel = load_panel()
    btc, eth = load_control_assets()

    print()
    sweep_rows, k1_by_k = cmd_sweep(panel)
    frozen_k = select_k(k1_by_k)
    print(f"\nFROZEN k = {frozen_k} (selected on PANEL_TRAIN only; "
          f"D1 counts by k: {k1_by_k})")

    print()
    causality_ok = cmd_causality(frozen_k, [btc] + panel[:3])
    if not causality_ok:
        raise SystemExit("CAUSALITY PROBE FAILED — refusing to report D1-D4 "
                         "results until the lookahead bug is fixed.")

    # D1 is already computed by the sweep for the frozen k — reuse it rather
    # than re-running (an identical backtest re-run would inflate the config
    # count for no new information).
    d1_rows = sweep_rows[frozen_k]

    print("\n" + "=" * 100)
    print(f"D2 (FALSIFICATION CONTROL) — CONTROL window, BTC and ETH, k={frozen_k}")
    print("=" * 100)
    d2_rows: list[dict] = []
    for a in (btc, eth):
        strat = KellyRegimeV4KamaAnchors(k=frozen_k)
        cell(a, strat, CONTROL, SPOT_BASE, "CONTROL", d2_rows)

    print("\n" + "=" * 100)
    print(f"D3 (GENERALIZATION, reported not gating) — PANEL_TEST, spot @0.10%, k={frozen_k}")
    print("=" * 100)
    d3_rows: list[dict] = []
    for a in panel:
        strat = KellyRegimeV4KamaAnchors(k=frozen_k)
        cell(a, strat, PANEL_TEST, SPOT_BASE, "PANEL_TEST", d3_rows)

    print("\n" + "=" * 100)
    print(f"D4 (0.40% FEE FALSIFICATION) — PANEL_TRAIN, spot @0.40%, k={frozen_k}")
    print("=" * 100)
    d4_rows: list[dict] = []
    for a in panel:
        strat = KellyRegimeV4KamaAnchors(k=frozen_k)
        cell(a, strat, PANEL_TRAIN, SPOT_REAL, "PANEL_TRAIN", d4_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_all = []
    for k, rows in sweep_rows.items():
        for r in rows:
            sweep_all.append({"k": k, **r})
    pd.DataFrame(sweep_all).to_csv(OUT_DIR / "k_sweep.csv", index=False)
    pd.DataFrame(d1_rows).to_csv(OUT_DIR / "d1_panel_train.csv", index=False)
    pd.DataFrame(d2_rows).to_csv(OUT_DIR / "d2_control.csv", index=False)
    pd.DataFrame(d3_rows).to_csv(OUT_DIR / "d3_panel_test.csv", index=False)
    pd.DataFrame(d4_rows).to_csv(OUT_DIR / "d4_fee_falsification.csv", index=False)

    verdict_text = verdicts(frozen_k, k1_by_k, d1_rows, d2_rows, d3_rows, d4_rows, len(panel))
    print(f"\nTotal backtest configurations evaluated: {CONFIG_COUNT}")
    print("Holdout consultations added by this round: 0 "
          "(BTC/ETH truncated at 2022-12-31 before any other line touches them; "
          "panel-asset reads cost +0 per the pre-registration)")

    write_report(frozen_k, k1_by_k, d1_rows, d2_rows, d3_rows, d4_rows, len(panel), verdict_text)


def verdicts(frozen_k: float, k1_by_k: dict[float, int], d1_rows: list[dict],
             d2_rows: list[dict], d3_rows: list[dict], d4_rows: list[dict],
             n: int) -> str:
    print("\n" + "=" * 100)
    print("PRE-REGISTERED DECISION RULES (experiments/r60_shared.py)")
    print("=" * 100)

    d1 = pd.DataFrame(d1_rows)
    k1 = int((d1.cand_dd < d1.mh_dd).sum())
    excl = int(((d1.dd_matched_lo > 0) | (d1.dd_matched_hi < 0)).sum())
    better_excl = int((d1.dd_matched_hi < 0).sum())
    p1 = binomial_tail(k1, n)
    print(f"D1 (primary, matched-exposure drawdown, PANEL_TRAIN, spot @0.10%, k={frozen_k}): "
          f"{k1}/{n} -> {d1_verdict(k1, n)} (exact binomial p={p1:.4f})")
    print(f"    paired bootstrap: {excl}/{n} intervals exclude zero "
          f"({better_excl}/{n} of them in the candidate's favour)")

    d2 = pd.DataFrame(d2_rows).set_index("asset")
    dd_advantage = {t: float(d2.loc[t, "dd_matched_diff"]) for t in ("BTC", "ETH")}
    print(f"D2 (falsification control, CONTROL window): "
          f"BTC {dd_advantage['BTC']:+.1f}pp (R-57: {R57_CONTROL_DD_ADVANTAGE['BTC']:+.1f}pp), "
          f"ETH {dd_advantage['ETH']:+.1f}pp (R-57: {R57_CONTROL_DD_ADVANTAGE['ETH']:+.1f}pp), "
          f"tolerance {D2_REGRESSION_TOLERANCE_PP:+.1f}pp -> "
          f"{'PASSES' if d2_passes(dd_advantage) else 'FAILS'}")

    d3 = pd.DataFrame(d3_rows)
    k3 = int((d3.cand_dd < d3.mh_dd).sum())
    print(f"D3 (generalization, PANEL_TEST, descriptive): {k3}/{n} -> {d1_verdict(k3, n)}")

    d4 = pd.DataFrame(d4_rows)
    k4 = int((d4.cand_final > d4.hold_final).sum())
    print(f"D4 (0.40% fee falsification, beats buy_and_hold final balance): "
          f"{k4}/{n} -> {'SURVIVES' if k4 >= n - 1 else 'FAILS (as predicted)'}")

    verdict = "PROMOTE-CANDIDATE" if promoted(k1, dd_advantage, n) else "NEGATIVE"
    print(f"\nOVERALL (promoted(k1, dd_advantage) mechanically applied): {verdict}")
    print(f"frozen k = {frozen_k}; sweep D1 counts by k: {k1_by_k}")
    return verdict


def write_report(frozen_k: float, k1_by_k: dict[float, int], d1_rows: list[dict],
                  d2_rows: list[dict], d3_rows: list[dict], d4_rows: list[dict],
                  n: int, verdict: str) -> None:
    d1 = pd.DataFrame(d1_rows)
    k1 = int((d1.cand_dd < d1.mh_dd).sum())
    p1 = binomial_tail(k1, n)
    d2 = pd.DataFrame(d2_rows).set_index("asset")
    dd_advantage = {t: float(d2.loc[t, "dd_matched_diff"]) for t in ("BTC", "ETH")}
    d3 = pd.DataFrame(d3_rows)
    k3 = int((d3.cand_dd < d3.mh_dd).sum())
    d4 = pd.DataFrame(d4_rows)
    k4 = int((d4.cand_final > d4.hold_final).sum())

    def fmt_row(r) -> str:
        return (f"| {r.asset} | {r.cand_dd:.1f}% | {r.mh_dd:.1f}% | "
                f"{r.dd_matched_diff:+.1f}pp | [{r.dd_matched_lo:+.1f}, {r.dd_matched_hi:+.1f}] | "
                f"{'yes' if r.cand_dd < r.mh_dd else 'no'} |")

    lines = []
    lines.append("# R-60 conservative — KAMA-adaptive vote anchors")
    lines.append("")
    lines.append("Pre-registration: `experiments/r60_shared.py`. Mechanism: "
                  "`experiments/r60_conservative_kama_anchors.py`. "
                  "Backlog **B-26**.")
    lines.append("")
    lines.append("## Mechanism")
    lines.append("")
    lines.append("`kelly_regime_v4`'s three fixed-window SMA anchors (20/40/80-day) "
                  "are replaced by Kaufman (1995) KAMA-style adaptive averages "
                  "bracketed around the same nominal horizons, so each anchor's "
                  "effective memory shortens in efficient/trending action and "
                  "lengthens in noise/chop. Formula, bracket multiplier `k` and "
                  "warmup derivation are in the module docstring. Everything else "
                  "(1% latch band, vote-fraction average, fractional-Kelly sizing, "
                  "the conditional volatility-targeting/breakout hysteresis, 2x cap, "
                  "10% deadband) is unchanged from `KellyRegimeV3`/`KellyRegimeV4`.")
    lines.append("")
    lines.append("## k-grid sweep (PANEL_TRAIN only)")
    lines.append("")
    lines.append("| k | D1 count (of 6) | verdict |")
    lines.append("|---|---|---|")
    for k in K_GRID:
        lines.append(f"| {k} | {k1_by_k[k]}/{n} | {d1_verdict(k1_by_k[k], n)} |")
    lines.append("")
    lines.append(f"**Frozen k = {frozen_k}** (best D1 count on PANEL_TRAIN; "
                  "ties broken toward k=2.0, the grid midpoint, as the least "
                  "extreme choice).")
    lines.append("")
    lines.append("## D1 — primary, matched-exposure drawdown, PANEL_TRAIN, spot @0.10%")
    lines.append("")
    lines.append(f"**{k1}/{n} -> {d1_verdict(k1, n)}** (exact one-sided binomial p={p1:.4f})")
    lines.append("")
    lines.append("| asset | candidate DD | matched-hold DD | dDD (matched) | 95% bootstrap CI | candidate better |")
    lines.append("|---|---|---|---|---|---|")
    for r in d1.itertuples():
        lines.append(fmt_row(r))
    lines.append("")
    lines.append("## D2 — falsification control, CONTROL window (BTC/ETH, 2020-04..2022-12)")
    lines.append("")
    lines.append("| asset | candidate dDD (matched) | R-57 v4 dDD | tolerance | within tolerance |")
    lines.append("|---|---|---|---|---|")
    for t in ("BTC", "ETH"):
        within = dd_advantage[t] <= R57_CONTROL_DD_ADVANTAGE[t] + D2_REGRESSION_TOLERANCE_PP
        lines.append(f"| {t} | {dd_advantage[t]:+.1f}pp | {R57_CONTROL_DD_ADVANTAGE[t]:+.1f}pp | "
                      f"+{D2_REGRESSION_TOLERANCE_PP:.1f}pp | {'yes' if within else 'NO'} |")
    lines.append("")
    lines.append(f"**D2 overall: {'PASSES' if d2_passes(dd_advantage) else 'FAILS'}** "
                  "(must pass on both BTC and ETH).")
    lines.append("")
    lines.append("## D3 — generalization check, PANEL_TEST (descriptive, not a gate)")
    lines.append("")
    lines.append(f"**{k3}/{n} -> {d1_verdict(k3, n)}**")
    lines.append("")
    lines.append("| asset | candidate DD | matched-hold DD | dDD (matched) | 95% bootstrap CI | candidate better |")
    lines.append("|---|---|---|---|---|---|")
    for r in d3.itertuples():
        lines.append(fmt_row(r))
    lines.append("")
    lines.append("## D4 — 0.40% fee falsification, PANEL_TRAIN (predicted to fail)")
    lines.append("")
    lines.append(f"Candidate beats `buy_and_hold` final balance in **{k4}/{n}** assets "
                  f"-> {'SURVIVES' if k4 >= n - 1 else 'FAILS (as predicted)'}")
    lines.append("")
    lines.append("| asset | candidate final | hold final | candidate beats hold |")
    lines.append("|---|---|---|---|")
    for r in d4.itertuples():
        lines.append(f"| {r.asset} | ${r.cand_final:,.0f} | ${r.hold_final:,.0f} | "
                      f"{'yes' if r.cand_final > r.hold_final else 'no'} |")
    lines.append("")
    lines.append(f"## Total configurations evaluated by this branch: {CONFIG_COUNT}")
    lines.append("")
    lines.append("(k-grid sweep: 3 k x 6 panel assets x 3 arms (candidate, "
                  "buy_and_hold, matched hold) = 54 backtests; D2: 2 assets x 3 "
                  "arms = 6; D3: 6 assets x 3 arms = 18; D4: 6 assets x 3 arms = "
                  "18. D1 reuses the frozen-k sweep cells rather than re-running "
                  "them.)")
    lines.append("")
    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    lines.append("Promotion bar (pre-registered, `r60_shared.promoted`): D1 >= 5/6 "
                  "AND D2 passes on both BTC and ETH. Anything else is NEGATIVE.")
    lines.append("")
    lines.append("Holdout consultations added by this round: 0 (no BTC/ETH 2023+ bar read).")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nreport written: {REPORT_PATH}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "sweep":
        panel = load_panel()
        _sweep_rows, k1_by_k = cmd_sweep(panel)
        frozen_k = select_k(k1_by_k)
        print(f"\nFrozen k would be: {frozen_k} (D1 counts: {k1_by_k})")
        print(f"Configurations evaluated: {CONFIG_COUNT}")
        return
    if cmd == "causality":
        panel = load_panel()
        btc, _eth = load_control_assets()
        ok = cmd_causality(2.0, [btc] + panel[:3])
        print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
        return
    if cmd == "run":
        cmd_run()
        return
    raise SystemExit(f"unknown command {cmd!r} (sweep | causality | run)")


if __name__ == "__main__":
    main()
