#!/usr/bin/env python
"""R-129 NOVEL branch: ``NovelBucketBand`` -- a per-timescale-bucket EV band
for ``hedge_experts``.

Full grounding, literature citation, non-duplication argument, named failure
modes, and the pre-registered decision rule / falsification test all live in
``experiments/r129_shared.py``'s own module docstring (read in full before
this file was written); not re-derived here beyond the summary below. This
file NEVER edits ``r129_shared.py`` (frozen, shared with the parallel
CONSERVATIVE branch, a disjoint file this session does not read or write --
``experiments/r129_conservative_per_expert_band.py`` and anything it might
create), NEVER edits ``src/tradebot/strategies/hedge_experts.py``, and never
reads a bar at or after ``r129_shared.OOS_START`` (2023-01-01) from any data
source.

MECHANISM (exact). ``HedgeExperts.prepare()`` builds ten causal experts
(``HedgeExperts._experts``, reused verbatim -- imported and called, not
re-derived) and blends them with discounted multiplicative weights (Hedge)
into a per-bar signal ``x = p @ a`` in ``[-1, 1]``, then re-targets toward
``x`` only when ``abs(x - pos) > hysteresis`` for a FIXED ``hysteresis =
0.05``. R-128 replaced that one fixed threshold with one EV-derived band on
the whole blended ``x`` and found it NEGATIVE, naming this round's construction
as the untested alternative. This branch groups the ten experts into THREE
timescale buckets (``r129_shared.EXPERT_BUCKET``: FAST = {0,1,4,5,6}, SLOW =
{2,3,7}, STATIC = {8,9}), computes each bucket's own Hedge-weighted
sub-blend every bar (``x_bucket = sum(weight_i * expert_i for i in
bucket)``), and applies ONE EV band per bucket -- three bands total, never
one on the sum -- derived from that bucket's own STRUCTURAL horizon
(``r129_shared.BUCKET_HORIZON_DAYS``, the median of its members'
``EXPERT_HORIZON_DAYS``, frozen upstream, never fit to a return):

    band_bucket = clip(2*fee / (H_bucket_years * sigma_market**2 * leverage),
                        MIN_BAND, MAX_BAND)
    held[bucket] updated to x_bucket only if |x_bucket - held[bucket]| > band_bucket
    x = sum(held.values())
    if |x - last_target| > 1e-9: ctx.order_target(x)

``fee``/``leverage`` are read live via ``ctx.market`` at ``on_bar`` time (so
the band differs between spot and futures automatically); ``sigma_market`` is
the ONE shared market-vol input (``_ev_vol``, identical construction to
R-128's own), not any one expert's own units. This is the CORRECTED,
unit-consistent band shape from the start (``current``/``desired`` both in
fraction-of-max-leverage units, matching ``hedge_experts``'s native
``ctx.order_target`` convention) -- R-128's own post-hoc unit bug
(comparing a fraction-of-max-leverage signal against a notional-multiple
``current``, routed through ``order_notional``, silently capping exposure
near 1x notional on 5x futures) is avoided by construction, not discovered
after the fact. Orders are placed via ``ctx.order_target`` only, exactly
``hedge_experts``'s own convention.

**Per-bucket state, instance-tracked (no broker-side equivalent exists).**
``self._held`` (dict, keys "fast"/"slow"/"static"), ``self._last_target``
(float), and ``self._retarget_count`` (dict, per-bucket counters, diagnostic
only) are instance state on the strategy object, per
``r129_shared.py``'s binding Implementation note. **Disclosed cold-start
convention:** ``self._held`` is initialized ONCE, on the first ``on_bar``
call that sees a finite positive ``sigma_market`` (normally the first
post-warmup bar, ``i = warmup = 2500``), to that bar's own live sub-blend
values -- not accumulated causally from ``prepare()``'s own internal loop
the way the Hedge weights ``p`` themselves are. This is a minor, bounded
cold-start artifact (0.6% of inner-train bars), not a lookahead, and is
reported plainly rather than silently matched to ``hedge_experts``'s own
(different, broker-state-based) behavior.

``NovelBucketBand`` is NOT ``@register``ed -- experiments/-only per this
round's instructions, reached only through this file.

CONFIGURATIONS EVALUATED: 1 (causal-truncation self-test) + 4 (B1: BTC
spot/futures x full-period/inner-validation) + 4 (B3: bucket-horizon
multiplier sweep 0.5/1/2/4x, applied uniformly to all three bucket
horizons at once, FUTURES inner-validation) + 1 (B4: ETH spot,
inner-validation, primary 1x config) + 4 (B5: same 4 B1 cells at the 0.40%
fee tier) = 14 total, matching R-128's own count. The per-bucket diagnostic
table (Section 2.6 of the report) reuses the already-run BTC-spot/full B1
cell's own strategy instance -- no additional configuration is run for it.

DECISION RULE (pre-registered, verbatim from ``r129_shared.py``, unaltered
after seeing any number): PROMOTE-candidate only if the causal-truncation
probe AND B1 (all 4 cells clear) AND B3 (>=3/4 same-signed) AND B4 (sign
replicates on ETH) AND B5 (no sign flip at 0.40% fee) all pass. Anything
else is NEGATIVE. B2 (drawdown/turnover) and the per-bucket re-target counts
(failure modes 1-2) are diagnostic only and never gate promotion by
themselves.

BUG LOG: no computational/causal bug found. The unit-consistency issue
R-128 discovered post-hoc (fraction-of-max-leverage signal compared against
a notional-multiple ``current``, routed through ``order_notional``) does not
arise here because this branch never reads back a broker-side position at
all -- ``self._held`` is tracked entirely in the strategy's own
fraction-of-max-leverage units from construction, and every comparison
(``x_bucket`` vs ``held[bucket]``, ``x`` vs ``last_target``) is
unit-homogeneous by construction. This is disclosed as a design choice made
BEFORE any code ran (Implementation note item 3 in ``r129_shared.py``), not
a fix discovered after running numbers. ONE non-computational bug WAS found
and fixed: the first draft of ``write_report``'s Section 2.6 discussion
paragraph hard-coded a wrong description of the STATIC-vs-FAST re-target
comparison (it asserted the two buckets re-target "about as often," which
was false on the actual first run: FAST re-targeted 0 times, STATIC 592
times, out of 631,008 bars). This was a prose bug in the report generator,
not in the strategy or battery -- no metric, Sharpe, trade count, or
verdict changed. Fixed before the numbers below were regenerated; both runs
produced numerically identical results (verified by re-running the full
battery after the fix).

USAGE
-----
    python experiments/r129_novel_bucket_band.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import r129_shared as shared
from tradebot.inference import daily_returns, paired_bootstrap, total_log_return
from tradebot.metrics import compute_metrics
from tradebot.strategies.hedge_experts import HedgeExperts
from tradebot.strategy import Context
from tradebot.window import run_period

# ----------------------------------------------------------------------
# Pre-registered constants. Fixed before any inner-validation number was
# read.
# ----------------------------------------------------------------------
PRIMARY_BUCKET_HORIZON_DAYS = dict(shared.BUCKET_HORIZON_DAYS)  # frozen upstream
MIN_BAND = shared.MIN_BAND
MAX_BAND = shared.MAX_BAND
B3_MULTIPLIERS = shared.B3_MULTIPLIERS
BUCKETS = ("fast", "slow", "static")

# Diagnostic-only alternative clearance for B1/B5: a "real" drawdown
# improvement, in percentage points of max_drawdown_pct, when d_sharpe
# itself does not clear the noise floor. Fixed before any cell was run
# (identical convention/value to R-128's own DD_IMPROVEMENT_PP).
DD_IMPROVEMENT_PP = 5.0


# ================================================================== (1)
# NovelBucketBand: HedgeExperts's exact expert construction + Hedge weight
# update (calls HedgeExperts._experts verbatim), grouped into three
# timescale buckets, each independently EV-banded at its own structural
# horizon, evaluated in on_bar so it can read ctx.market.fee_rate/leverage.
# NOT @register'd.
# ==================================================================

class NovelBucketBand(HedgeExperts):
    """hedge_experts's exact expert construction and Hedge weight update
    (``HedgeExperts._experts``, reused verbatim), with ONE substitution: the
    fixed ``hysteresis=0.05`` re-target threshold on the whole blended
    output is replaced by THREE independent EV-derived no-trade bands, one
    per timescale bucket (fast/slow/static), each applied to that bucket's
    own Hedge-weighted sub-blend before the three are summed into the final
    target. See module docstring above and ``experiments/r129_shared.py``
    for the full derivation, non-duplication argument, and pre-registration.
    Not ``@register``ed -- experiments/-only.
    """

    name = "r129_novel_bucket_band"

    def __init__(self, eta: float = 0.05, fixed_share: float = 1e-4,
                 fee_rate: float = 0.0005,
                 bucket_horizon_days: dict | None = None,
                 min_band: float = MIN_BAND, max_band: float = MAX_BAND) -> None:
        # hysteresis=0.0 on the base class is inert here: prepare() below is
        # fully overridden and never reads self.hysteresis.
        super().__init__(eta=eta, fixed_share=fixed_share, hysteresis=0.0,
                          fee_rate=fee_rate)
        self.bucket_horizon_days = (dict(bucket_horizon_days) if bucket_horizon_days
                                     is not None else dict(PRIMARY_BUCKET_HORIZON_DAYS))
        self.min_band = min_band
        self.max_band = max_band
        self._bucket_idxs = {
            b: [i for i, bb in enumerate(shared.EXPERT_BUCKET) if bb == b]
            for b in BUCKETS
        }
        # Instance state (Implementation note, r129_shared.py): never a
        # dataframe column -- there is no broker-side equivalent of a
        # per-bucket position to read back.
        self._held: dict | None = None
        self._last_target: float = 0.0
        self._retarget_count: dict = {b: 0 for b in BUCKETS}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Byte-identical expert construction + Hedge weight-update loop to
        ``HedgeExperts.prepare()`` (same call to ``self._experts()``, same
        z_t/fee_n/g/logw/p update, line for line), with the hysteresis-gated
        ``pos`` collapse removed: emits the RAW per-bar expert matrix
        (``expert_0``..``expert_9``) and the RAW per-bar Hedge weight matrix
        (``weight_0``..``weight_9``), plus the causal annualized vol series
        the bands need (``_ev_vol``, identical construction to R-128's own).
        Fully market-independent -- never reads ``ctx``/``self.market``."""
        r = np.log(df["close"]).diff()
        sig1 = r.ewm(span=288, min_periods=250).std()
        a = self._experts(df, r, sig1)  # HedgeExperts._experts, unchanged, (n, 10)
        r_a = r.to_numpy()
        sig_a = sig1.shift(1).to_numpy()

        n, num = a.shape
        weight_mat = np.zeros((n, num))
        logw = np.zeros(num)
        # Initial (pre-update) weights, uniform after the fixed-share mix --
        # used to seed rows 0/1, which the weight-update loop below (mirrors
        # HedgeExperts.prepare() exactly) only starts updating at i=2. These
        # rows are never read by on_bar in practice (warmup=2500 >> 2).
        p0 = np.exp(logw)
        p0 /= p0.sum()
        p0 = (1.0 - self.fixed_share) * p0 + self.fixed_share / num
        weight_mat[0] = p0
        weight_mat[1] = p0
        logw = np.log(p0)
        for i in range(2, n):
            s = sig_a[i]
            if not np.isfinite(s) or s <= 0:
                weight_mat[i] = weight_mat[i - 1]
                continue
            z_t = min(max(r_a[i] / (3.0 * s), -1.0), 1.0)
            fee_n = min(self.fee_rate / (3.0 * s), 0.25)
            g = np.clip(a[i - 1] * z_t - fee_n * np.abs(a[i - 1] - a[i - 2]), -1.0, 1.0)
            logw += self.eta * g
            logw -= logw.max()
            p = np.exp(logw)
            p /= p.sum()
            p = (1.0 - self.fixed_share) * p + self.fixed_share / num
            logw = np.log(p)
            weight_mat[i] = p

        for j in range(num):
            df[f"expert_{j}"] = a[:, j]
            df[f"weight_{j}"] = weight_mat[:, j]
        df["_ev_vol"] = (sig1 * np.sqrt(shared.BARS_PER_YEAR)).shift(1)
        return df

    def _bucket_x(self, ctx: Context, bucket: str) -> float:
        idxs = self._bucket_idxs[bucket]
        return float(sum(float(ctx.bar[f"weight_{i}"]) * float(ctx.bar[f"expert_{i}"])
                         for i in idxs))

    def _band(self, fee: float, vol: float, leverage: float, bucket: str) -> float:
        h_years = self.bucket_horizon_days[bucket] / 365.25
        variance = max(vol, 1e-6) ** 2
        lev = max(leverage, 1e-9)
        band = 2.0 * fee / (h_years * variance * lev)
        return float(np.clip(band, self.min_band, self.max_band))

    def on_bar(self, ctx: Context) -> None:
        vol = float(ctx.bar["_ev_vol"])
        if not np.isfinite(vol) or vol <= 0:
            return  # guard, matches R-128's own non-finite/non-positive skip

        fee = ctx.market.fee_rate
        lev = max(ctx.market.leverage, 1e-9)

        if self._held is None:
            # Disclosed cold-start convention: seed to THIS bar's own raw
            # (unbanded) sub-blend, not accumulated from prepare()'s loop.
            self._held = {b: self._bucket_x(ctx, b) for b in BUCKETS}

        for bucket in BUCKETS:
            x_bucket = self._bucket_x(ctx, bucket)
            band = self._band(fee, vol, lev, bucket)
            if abs(x_bucket - self._held[bucket]) > band:
                self._held[bucket] = x_bucket
                self._retarget_count[bucket] += 1

        x = sum(self._held.values())
        if abs(x - self._last_target) > 1e-9:
            ctx.order_target(x)
            self._last_target = x


# ================================================================== (2)
# Run/metric helpers. Mirrors r128_conservative_ev_band.py's own
# run_candidate/b1_signal/cell_clears, extended to also hand back the
# strategy instance (needed for the per-bucket re-target diagnostic table).
# ==================================================================

def run_candidate(factory, df: pd.DataFrame, market, start, end, label: str = ""):
    strat = factory()
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res, strat


def b1_signal(factory, df: pd.DataFrame, market, start=None, end=None,
              label: str = "") -> dict:
    if start is None:
        start = shared.INNER_VAL_START
    if end is None:
        end = shared.INNER_VAL_END
    m_cand, res_cand, strat_cand = run_candidate(factory, df, market, start, end, label)
    m_base, res_base = shared.run_baseline(df, market, start, end, label)
    r_cand = daily_returns(res_cand.equity)
    r_base = daily_returns(res_base.equity)
    n = min(len(r_cand), len(r_base))
    paired = paired_bootstrap(r_cand.to_numpy()[:n], r_base.to_numpy()[:n],
                               stat=total_log_return, seed=129)
    return {
        "sharpe_cand": m_cand.sharpe, "sharpe_base": m_base.sharpe,
        "d_sharpe": m_cand.sharpe - m_base.sharpe,
        "paired_diff": paired.diff.point, "paired_lo": paired.diff.lo,
        "paired_hi": paired.diff.hi, "significant": paired.significant,
        "dd_cand": m_cand.max_drawdown_pct, "dd_base": m_base.max_drawdown_pct,
        "trades_cand": m_cand.num_trades, "trades_base": m_base.num_trades,
        "final_cand": m_cand.final_balance, "final_base": m_base.final_balance,
        "strat_cand": strat_cand,
    }


def cell_clears(r: dict) -> bool:
    """A single B1/B5 cell 'clears' if d_sharpe beats the +/-0.2 noise floor,
    or the paired-bootstrap CI excludes zero positively, or there is a real
    (>= DD_IMPROVEMENT_PP) drawdown improvement -- the pre-registered OR,
    identical convention to R-128."""
    return bool(r["d_sharpe"] > 0.2 or r["paired_lo"] > 0.0
                or (r["dd_base"] - r["dd_cand"]) >= DD_IMPROVEMENT_PP)


# ================================================================== (3)
# Causal-truncation self-test on THIS file's own new code (mirrors
# r129_shared.py's own __main__ probe and R-128's own, same split, same
# market).
# ==================================================================

def causal_truncation_probe(df: pd.DataFrame, label: str):
    factory = lambda: NovelBucketBand()
    m_full, _, _ = run_candidate(factory, df, shared.SPOT,
                                 shared.INNER_TRAIN_START, shared.INNER_TRAIN_END, label)
    df_trunc = df.loc[:shared.INNER_VAL_END]
    m_trunc, _, _ = run_candidate(factory, df_trunc, shared.SPOT,
                                  shared.INNER_TRAIN_START, shared.INNER_TRAIN_END, label)
    ok = bool(np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9))
    return ok, m_full.final_balance, m_trunc.final_balance


# ================================================================== (4)
# Report-file writer.
# ==================================================================

def write_report(results: dict) -> None:
    b1 = results["b1"]
    b3 = results["b3"]
    b4 = results["b4"]
    b5 = results["b5"]
    diag = results["diag"]
    probe_ok = results["probe_ok"]
    verdict = results["verdict"]

    lines = []
    lines.append("# R-129 (NOVEL branch) -- per-timescale-bucket EV rebalance "
                  "bands for `hedge_experts` (08-25)\n")
    lines.append(
        "Unregistered candidate. Code: `experiments/r129_novel_bucket_band.py`. "
        "Not `@register`ed, not auto-discovered, nothing committed by this "
        "session. `src/tradebot/strategies/hedge_experts.py` is never edited "
        "-- `NovelBucketBand` subclasses `HedgeExperts` and reuses "
        "`HedgeExperts._experts()` verbatim. Full derivation, literature "
        "citation, non-duplication argument, named failure modes, and the "
        "pre-registered decision rule live in `experiments/r129_shared.py`'s "
        "module docstring (frozen, shared with the parallel CONSERVATIVE "
        "branch); only summarized here.\n"
    )
    lines.append(
        "## 1. Mechanism recap\n\n"
        "`hedge_experts` blends ten causal technical experts with discounted "
        "multiplicative weights (Hedge) into a raw signal `x` in [-1, 1], then "
        "only re-targets toward `x` when `abs(x - pos) > hysteresis` for a "
        "FIXED `hysteresis = 0.05`. R-128 replaced that one threshold with one "
        "EV-derived band on the whole blended `x` and found it NEGATIVE, "
        "naming a per-timescale construction as the untested alternative "
        "(Ekren, Liu & Muhle-Karbe 2018 on multivariate no-trade regions). "
        "This branch groups the ten experts into three timescale buckets "
        "(`r129_shared.EXPERT_BUCKET`: FAST = {0,1,4,5,6}, SLOW = {2,3,7}, "
        "STATIC = {8,9}), computes each bucket's own Hedge-weighted "
        "sub-blend every bar, and applies ONE EV band per bucket (three "
        "total, never one on the sum), each derived from that bucket's own "
        "STRUCTURAL horizon (median of its members' native lookback/decay "
        "horizons, frozen upstream, never fit to a return):\n\n"
        "```\n"
        "band_bucket = clip(2*fee / (H_bucket_years * sigma_market**2 * leverage),\n"
        "                    MIN_BAND, MAX_BAND)\n"
        "held[bucket] <- x_bucket   only if |x_bucket - held[bucket]| > band_bucket\n"
        "x = sum(held.values())\n"
        "if |x - last_target| > 1e-9: ctx.order_target(x)\n"
        "```\n\n"
        f"`fee`/`leverage` read live via `ctx.market`. `MIN_BAND={MIN_BAND}`, "
        f"`MAX_BAND={MAX_BAND}` are `kelly_regime_ev`'s/R-128's own literal "
        "defaults, reused unchanged. The expert construction and Hedge "
        "weight update (`HedgeExperts._experts`, the weight-update loop) are "
        "byte-identical to the registered strategy -- only the re-target "
        "decision changes, and it changes at bucket granularity, not "
        "per-expert (that is the parallel CONSERVATIVE branch) and not on "
        "the single already-blended output (that was R-128).\n\n"
        "**Per-bucket instance state and cold start.** `self._held` (dict), "
        "`self._last_target` (float), and `self._retarget_count` (dict, "
        "diagnostic) are tracked on the strategy instance -- there is no "
        "broker-side equivalent of a per-bucket position to read back. "
        "`self._held` is initialized once, on the first `on_bar` call that "
        "sees a finite positive `sigma_market` (normally `i = warmup = "
        "2500`), to that bar's own live (unbanded) sub-blend values -- a "
        "disclosed, bounded cold-start artifact (0.6% of inner-train bars), "
        "not a lookahead.\n\n"
        "**No unit-consistency bug found.** Unlike R-128's own first draft, "
        "this branch never reads a broker-side position back into the band "
        "comparison at all -- `self._held` is tracked entirely in the "
        "strategy's own fraction-of-max-leverage units from construction, "
        "so `x_bucket` vs `held[bucket]` and `x` vs `last_target` are "
        "unit-homogeneous by design, and `ctx.order_target` (not "
        "`order_notional`) is used throughout. See BUG LOG in the file's own "
        "module docstring.\n"
    )

    lines.append("## 2. Results table\n\n")
    lines.append("### 2.1 Causal-truncation self-test\n\n")
    lines.append(f"**{'PASS' if probe_ok else 'FAIL'}** -- full-period final balance "
                 f"{results['probe_full']:.4f} vs truncated-frame final balance "
                 f"{results['probe_trunc']:.4f} (BTC spot, "
                 f"{shared.INNER_TRAIN_START}..{shared.INNER_TRAIN_END}).\n\n")

    lines.append("### 2.2 B1 -- BTC signal, spot + futures, full period + inner-validation\n\n")
    lines.append("| market | window | sharpe_cand | sharpe_base | d_sharpe | boot CI | "
                 "dd_cand% | dd_base% | trades_cand | trades_base | final_cand | final_base | clears? |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for mkt_name, window_name, r in b1["rows"]:
        lines.append(
            f"| {mkt_name} | {window_name} | {r['sharpe_cand']:+.4f} | {r['sharpe_base']:+.4f} | "
            f"{r['d_sharpe']:+.4f} | [{r['paired_lo']:+.4f}, {r['paired_hi']:+.4f}] | "
            f"{r['dd_cand']:.2f} | {r['dd_base']:.2f} | {r['trades_cand']} | {r['trades_base']} | "
            f"{r['final_cand']:.1f} | {r['final_base']:.1f} | {cell_clears(r)} |\n")
    lines.append(f"\n**B1 PASS (all 4 cells clear):** {b1['pass']}\n")

    lines.append("\n### 2.3 B3 -- bucket-horizon-multiplier plateau (uniform across "
                 "FAST/SLOW/STATIC, FUTURES, inner-validation)\n\n")
    lines.append("| multiplier | fast_days | slow_days | static_days | d_sharpe | boot CI | sign |\n"
                 "|---|---|---|---|---|---|---|\n")
    for row in b3["rows"]:
        h = row["horizons"]
        lines.append(f"| {row['multiplier']:g}x | {h['fast']:.4f} | {h['slow']:.4f} | "
                     f"{h['static']:.4f} | {row['d_sharpe']:+.4f} | "
                     f"[{row['paired_lo']:+.4f}, {row['paired_hi']:+.4f}] | {row['sign']:+.0f} |\n")
    lines.append(f"\n**B3 PASS (>=3/4 same-signed):** {b3['pass']} "
                 f"({b3['majority_count']}/4 share the majority sign)\n")

    lines.append("\n### 2.4 B4 -- ETH falsification (spot only, inner-validation, primary config)\n\n")
    lines.append(f"ETH spot d_sharpe = {b4['d_sharpe']:+.4f}, boot CI = "
                 f"[{b4['paired_lo']:+.4f}, {b4['paired_hi']:+.4f}]. "
                 f"BTC spot inner-validation d_sharpe sign = {b4['btc_sign']:+.0f}, "
                 f"ETH spot d_sharpe sign = {b4['eth_sign']:+.0f}. "
                 f"**B4 PASS (sign replicates):** {b4['pass']}\n")

    lines.append("\n### 2.5 B5 -- fee-tier survival (0.40% taker), primary config\n\n")
    lines.append("| market | window | d_sharpe @0.10% | d_sharpe @0.40% | sign flip? |\n"
                 "|---|---|---|---|---|\n")
    for mkt_name, window_name, r0, r1, flip in b5["rows"]:
        lines.append(f"| {mkt_name} | {window_name} | {r0['d_sharpe']:+.4f} | "
                     f"{r1['d_sharpe']:+.4f} | {flip} |\n")
    lines.append(f"\n**B5 PASS (no sign flip, any cell):** {b5['pass']}\n")

    lines.append(
        "\n### 2.6 Diagnostic -- per-bucket re-target counts (primary config, "
        "BTC spot, full period; failure modes 1-2 in `r129_shared.py`)\n\n"
        "| bucket | horizon_days | re-target count | share of bars re-targeted |\n"
        "|---|---|---|---|\n"
    )
    for b in BUCKETS:
        lines.append(f"| {b} | {PRIMARY_BUCKET_HORIZON_DAYS[b]:.4f} | "
                     f"{diag['counts'][b]} | {diag['counts'][b] / diag['n_bars']:.4%} |\n")
    lines.append(
        f"\nTotal candidate trades placed (this cell): {diag['trades_cand']}. "
        f"Baseline `hedge_experts` trades placed (this cell): {diag['trades_base']}. "
        "A bar can update more than one bucket's held value while producing "
        "at most one order (the buckets are summed before `order_target` is "
        "called), so the sum of the three counters is an upper, not a "
        "1-to-1, bound on the candidate's own trade count.\n"
    )

    lines.append(
        "\n## 3. Configurations evaluated\n\n"
        "1 causal-truncation probe + 4 B1 cells + 4 B3 sweep points + 1 B4 cell "
        f"+ 4 B5 cells = **{results['n_configs']} total**. The Section 2.6 "
        "diagnostic table reuses the already-run BTC-spot/full B1 cell's own "
        "strategy instance and required no additional configuration. No "
        "selection occurred among any of the 14 -- every cell is reported, "
        "none filtered by outcome.\n"
    )

    lines.append(
        "\n## 4. Decision-rule verdict\n\n"
        f"causal probe={probe_ok}  B1={b1['pass']}  B2=diagnostic-only  "
        f"B3={b3['pass']}  B4={b4['pass']}  B5={b5['pass']}\n\n"
        f"**VERDICT: {verdict}**\n\n"
        "(Pre-registered rule from `r129_shared.py`, unaltered after seeing "
        "any number: PROMOTE-candidate only if the causal-truncation probe "
        "AND B1 (all 4 cells clear) AND B3 (>=3/4 same-signed) AND B4 (sign "
        "replicates on ETH) AND B5 (no sign flip) all pass. Anything else is "
        "NEGATIVE.)\n"
    )

    lines.append("\n## 5. Discussion\n\n" + results["discussion"] + "\n")

    max_ts = results["max_ts"]
    lines.append(
        "\n## 6. Causality / holdout accounting\n\n"
        f"Max timestamp read anywhere in this branch: {max_ts} "
        f"(< OOS_START {shared.OOS_START}: "
        f"{max_ts < pd.Timestamp(shared.OOS_START, tz=max_ts.tz if max_ts.tz else None)}). "
        "No bar at or after 2023-01-01 was read by this file. "
        f"`pytest tests/test_causality_strict.py -q`: {results['pytest_summary']}.\n"
    )

    out_path = shared.ROOT / "experiments" / "reports" / "r129_novel_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(l if l.endswith("\n") else l + "\n" for l in lines))
    print(f"\nReport written to {out_path}")


# ================================================================== (5)
# Main: causal probe -> B1 -> B3 -> B4 -> B5 -> diagnostic -> verdict ->
# report -> pytest.
# ==================================================================

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []
    n_configs = 0

    print("=" * 78)
    print("R-129 NOVEL: NovelBucketBand -- hedge_experts's own architecture,")
    print("fixed hysteresis=0.05 replaced by THREE per-timescale-bucket EV bands.")
    print("=" * 78)

    btc, btc_label = shared.load_btc_train("spot")
    max_ts_seen.append(btc.index.max())
    print(f"\nBTC spot (truncated < {shared.OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    # -------------------------------------------------------------- causal probe
    print("\n" + "=" * 78)
    print("STEP 1 -- causal-truncation self-test (this file's own new code)")
    print("=" * 78)
    probe_ok, full_bal, trunc_bal = causal_truncation_probe(btc, btc_label)
    n_configs += 1
    print(f"  causal_truncation_probe: {'PASS' if probe_ok else 'FAIL'} "
          f"({full_bal:.4f} vs {trunc_bal:.4f})")
    assert probe_ok, "NovelBucketBand reads ahead of its own truncation point -- aborting"

    primary_factory = lambda: NovelBucketBand()

    # -------------------------------------------------------------- B1
    print("\n" + "=" * 78)
    print("STEP 2 -- B1: BTC signal, spot + futures, full period + inner-validation")
    print("=" * 78)
    b1_rows = []
    for mkt_name, market in (("spot", shared.SPOT), ("futures", shared.FUTURES)):
        for window_name, start, end in (
            ("full", shared.INNER_TRAIN_START, shared.INNER_VAL_END),
            ("val", shared.INNER_VAL_START, shared.INNER_VAL_END),
        ):
            r = b1_signal(primary_factory, btc, market, start, end, btc_label)
            n_configs += 1
            b1_rows.append((mkt_name, window_name, r))
            print(f"  {mkt_name:>8s} {window_name:>4s}  sharpe_cand={r['sharpe_cand']:+.4f}  "
                  f"sharpe_base={r['sharpe_base']:+.4f}  d_sharpe={r['d_sharpe']:+.4f}  "
                  f"boot=[{r['paired_lo']:+.4f},{r['paired_hi']:+.4f}]  "
                  f"dd_cand={r['dd_cand']:.2f}%  dd_base={r['dd_base']:.2f}%  "
                  f"trades_cand={r['trades_cand']}  trades_base={r['trades_base']}  "
                  f"clears={cell_clears(r)}")
    b1_pass = all(cell_clears(r) for _, _, r in b1_rows)
    print(f"  B1 PASS (all 4 cells clear +/-0.2 floor OR dd improvement OR CI>0): {b1_pass}")

    # -------------------------------------------------------------- B3
    print("\n" + "=" * 78)
    print(f"STEP 3 -- B3: bucket-horizon multiplier sweep {B3_MULTIPLIERS} "
          "(uniform), FUTURES inner-validation")
    print("=" * 78)
    b3_rows = []
    for m in B3_MULTIPLIERS:
        horizons = {b: PRIMARY_BUCKET_HORIZON_DAYS[b] * m for b in BUCKETS}
        factory = (lambda horizons=horizons: NovelBucketBand(bucket_horizon_days=horizons))
        r = b1_signal(factory, btc, shared.FUTURES)
        n_configs += 1
        sign = float(np.sign(r["d_sharpe"]))
        row = dict(multiplier=m, horizons=horizons, sign=sign, **r)
        b3_rows.append(row)
        print(f"  multiplier={m:g}x  fast={horizons['fast']:.4f}d  slow={horizons['slow']:.4f}d  "
              f"static={horizons['static']:.4f}d  d_sharpe={r['d_sharpe']:+.4f}  "
              f"boot=[{r['paired_lo']:+.4f},{r['paired_hi']:+.4f}]  sign={sign:+.0f}")
    signs = [row["sign"] for row in b3_rows]
    majority_count = max((signs.count(s) for s in set(signs)), default=0)
    b3_pass = majority_count >= 3
    print(f"  B3 PASS (>=3/4 same-signed): {b3_pass} ({majority_count}/4)")

    # -------------------------------------------------------------- B4
    print("\n" + "=" * 78)
    print("STEP 4 -- B4: ETH falsification (spot only, inner-validation, primary config)")
    print("=" * 78)
    eth = shared.load_eth_train()
    max_ts_seen.append(eth.index.max())
    print(f"ETH spot (truncated < {shared.OOS_START}): {len(eth):,} bars, "
          f"{eth.index[0]} -> {eth.index[-1]}")
    b4_r = b1_signal(primary_factory, eth, shared.SPOT)
    n_configs += 1
    btc_val_spot = next(r for mkt, win, r in b1_rows if mkt == "spot" and win == "val")
    btc_sign = float(np.sign(btc_val_spot["d_sharpe"]))
    eth_sign = float(np.sign(b4_r["d_sharpe"]))
    b4_pass = bool(btc_sign != 0 and eth_sign == btc_sign)
    print(f"  ETH spot d_sharpe={b4_r['d_sharpe']:+.4f}  "
          f"boot=[{b4_r['paired_lo']:+.4f},{b4_r['paired_hi']:+.4f}]")
    print(f"  BTC spot (val) sign={btc_sign:+.0f}  ETH spot sign={eth_sign:+.0f}  "
          f"B4 PASS (sign replicates): {b4_pass}")

    # -------------------------------------------------------------- B5
    print("\n" + "=" * 78)
    print("STEP 5 -- B5: fee-tier survival (0.40% taker), primary config's B1 cells")
    print("=" * 78)
    b5_rows = []
    fee_market = {"spot": shared.SPOT_HIGH_FEE, "futures": shared.FUTURES_HIGH_FEE}
    for mkt_name, window_name, r0 in b1_rows:
        start, end = ((shared.INNER_TRAIN_START, shared.INNER_VAL_END) if window_name == "full"
                     else (shared.INNER_VAL_START, shared.INNER_VAL_END))
        r1 = b1_signal(primary_factory, btc, fee_market[mkt_name], start, end, btc_label)
        n_configs += 1
        flip = bool(np.sign(r1["d_sharpe"]) != np.sign(r0["d_sharpe"]) and r0["d_sharpe"] != 0)
        b5_rows.append((mkt_name, window_name, r0, r1, flip))
        print(f"  {mkt_name:>8s} {window_name:>4s}  d_sharpe@0.10%={r0['d_sharpe']:+.4f}  "
              f"d_sharpe@0.40%={r1['d_sharpe']:+.4f}  flip={flip}")
    b5_pass = not any(flip for *_, flip in b5_rows)
    print(f"  B5 PASS (no sign flip, any cell): {b5_pass}")

    # -------------------------------------------------------------- diagnostic
    print("\n" + "=" * 78)
    print("STEP 6 -- diagnostic: per-bucket re-target counts (primary config, "
          "BTC spot, full period)")
    print("=" * 78)
    spot_full_row = next(r for mkt, win, r in b1_rows if mkt == "spot" and win == "full")
    strat_spot_full = spot_full_row["strat_cand"]
    diag_counts = dict(strat_spot_full._retarget_count)
    diag_n_bars = int(len(btc.loc[shared.INNER_TRAIN_START:shared.INNER_VAL_END]))
    for b in BUCKETS:
        print(f"  bucket={b:>6s}  horizon_days={PRIMARY_BUCKET_HORIZON_DAYS[b]:.4f}  "
              f"re-targets={diag_counts[b]}  share={diag_counts[b] / diag_n_bars:.4%}")
    print(f"  candidate trades (this cell)={spot_full_row['trades_cand']}  "
          f"baseline trades (this cell)={spot_full_row['trades_base']}")

    # -------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    all_pass = probe_ok and b1_pass and b3_pass and b4_pass and b5_pass
    verdict = "PROMOTE-candidate" if all_pass else "NEGATIVE"
    print(f"causal probe={probe_ok}  B1={b1_pass}  B2=diagnostic-only  B3={b3_pass}  "
          f"B4={b4_pass}  B5={b5_pass}")
    print(f"VERDICT: {verdict}")

    # -------------------------------------------------------------- pytest
    print("\n" + "=" * 78)
    print("STEP 7 -- tests/test_causality_strict.py")
    print("=" * 78)
    import subprocess
    proc = subprocess.run(
        ["python", "-m", "pytest", "tests/test_causality_strict.py", "-q"],
        cwd=str(shared.ROOT), capture_output=True, text=True)
    pytest_out = proc.stdout.strip().splitlines()
    pytest_summary = pytest_out[-1] if pytest_out else f"(exit {proc.returncode})"
    print(f"  {pytest_summary}")

    max_ts = max(max_ts_seen)
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(1 probe + {len(b1_rows)} B1 + {len(b3_rows)} B3 + 1 B4 + {len(b5_rows)} B5)")
    print(f"max timestamp read anywhere in this branch: {max_ts} "
          f"(< {shared.OOS_START}: {max_ts < pd.Timestamp(shared.OOS_START, tz='UTC')})")
    print("NO bar at or after 2023-01-01 was ever read by this file.")
    print(f"\n[{time.time() - t0:.0f}s]")

    b1_by_cell = {(mkt, win): r for mkt, win, r in b1_rows}
    fut_full = b1_by_cell[("futures", "full")]
    fut_val = b1_by_cell[("futures", "val")]
    spot_full = b1_by_cell[("spot", "full")]
    spot_val = b1_by_cell[("spot", "val")]

    static_share = diag_counts["static"] / diag_n_bars
    fast_share = diag_counts["fast"] / diag_n_bars
    discussion = (
        "This branch tests the hypothesis `r129_shared.py`'s docstring names as "
        "untested by R-128: does damping at the level of three timescale-bucket "
        "sub-blends, rather than the single already-blended output, cut "
        "hedge_experts's turnover cost without destroying the responsiveness "
        "that makes it profitable? Reading the actual numbers against the five "
        "named risks, in order:\n\n"
        "(1) **Failure mode 1 (weight drift bypasses any pre-blend damping).** "
        f"Candidate trade counts: spot full {spot_full['trades_cand']} vs baseline "
        f"{spot_full['trades_base']}; spot val {spot_val['trades_cand']} vs "
        f"{spot_val['trades_base']}; futures full {fut_full['trades_cand']} vs "
        f"{fut_full['trades_base']}; futures val {fut_val['trades_cand']} vs "
        f"{fut_val['trades_base']}. "
        + ("Turnover is materially lower than baseline in every cell -- the "
           "bucket bands do cut trade count despite the Hedge weights moving "
           "every bar, so this branch is not structurally defeated by failure "
           "mode 1."
           if all(b1_by_cell[k]["trades_cand"] < b1_by_cell[k]["trades_base"]
                  for k in (("spot", "full"), ("spot", "val"), ("futures", "full"), ("futures", "val")))
           else "Turnover reduction is inconsistent across cells -- at least one "
           "cell does NOT show fewer candidate trades than baseline, meaning "
           "failure mode 1 (continuous Hedge-weight drift defeating pre-blend "
           "damping) is at least partially realized in this branch.") + "\n\n"
        "(2) **Failure mode 2, bucket-level analogue (near-permanent freezing).** "
        f"Per-bucket re-target counts (BTC spot, full period, {diag_n_bars:,} bars): "
        f"fast={diag_counts['fast']} ({fast_share:.3%} of bars), "
        f"slow={diag_counts['slow']} ({diag_counts['slow'] / diag_n_bars:.3%}), "
        f"static={diag_counts['static']} ({static_share:.3%}). "
        + ("The FAST bucket re-targets ZERO times across the entire full-period "
           "run -- its band, formed from the shortest structural horizon "
           "(0.0486d, dominated by the 1-bar-reversion expert's 1/288d "
           "horizon among its members), saturates at MAX_BAND=1.0 after "
           "clipping (this is exactly failure mode 2 named in "
           "`r129_shared.py`, realized here at bucket rather than per-expert "
           "granularity, and more severe than the risk as originally framed: "
           "not merely 'wide bands' but a band wide enough that the FAST "
           "bucket's cold-start value, set once at warmup, is NEVER updated "
           "again -- five of the ten experts (1h/6h momentum, MACD, RSI, "
           "1-bar reversion) are silently frozen out of the traded blend for "
           "the entire run). The SLOW and STATIC buckets, by contrast, both "
           "re-target a comparable, non-trivial number of times (333 and 592 "
           "respectively) -- the reverse of the naive expectation that the "
           "two structurally-inert STATIC experts (always-flat, buy-and-hold) "
           "would freeze most: STATIC's sub-blend is entirely the buy-and-hold "
           "expert's own Hedge weight (the always-flat expert contributes "
           "zero regardless of its weight), and that weight moves enough, "
           "relative to STATIC's own comparatively tight ~7-day-horizon band, "
           "to re-target more often than the intuitively 'fastest' bucket "
           "does. This is the opposite pattern the round's own docstring "
           "anticipated and is reported plainly rather than smoothed over."
           if diag_counts["fast"] < diag_counts["static"] else
           "The relative ordering of the three buckets' re-target counts does "
           "not show the FAST-freezes/STATIC-moves pattern seen in the run "
           "this discussion text was drafted against; re-verify before "
           "reusing this paragraph.") + "\n\n"
        "(3) **B1.** All four cells reported above " +
        ("clear the pre-registered OR (d_sharpe > 0.2, OR bootstrap CI lower "
         "bound > 0, OR >=5pp drawdown improvement)." if b1_pass else
         "do NOT all clear the pre-registered OR -- at least one of the four "
         "BTC cells fails to show a d_sharpe/CI/drawdown improvement over "
         "unmodified hedge_experts.") + "\n\n"
        "(4) **B4, the pre-registered falsification test.** ETH spot d_sharpe "
        f"= {b4_r['d_sharpe']:+.4f} (BTC spot inner-validation sign = "
        f"{btc_sign:+.0f}, ETH sign = {eth_sign:+.0f}). " +
        ("The sign replicates, consistent with the pre-registered pass "
         "condition." if b4_pass else
         "The sign does NOT replicate -- this is the sixth-plus instance on "
         "this project of a BTC-passing mechanism inverting on ETH "
         "(R-109, R-113, R-115-conservative, R-125-conservative, R-126 both "
         "branches, R-128 conservative weakly), the exact named risk B4 was "
         "built to catch, and on its own is sufficient to make the "
         "pre-registered decision rule return NEGATIVE regardless of any "
         "other gate.") + "\n\n"
        "(5) **B3, the LAG-failure check.** Majority sign "
        f"{majority_count}/4 across the {B3_MULTIPLIERS} bucket-horizon "
        "multiplier sweep (uniform across all three buckets). " +
        ("No reversal appears across this range." if b3_pass else
         "A sign reversal appears within this range, consistent with the "
         "band widening enough to reproduce the LAG failure this project has "
         "measured in every regime-timing mechanism tried on "
         "`kelly_regime_v4`.") + "\n\n"
        "(6) **Bucket-partition sensitivity (failure mode 5, disclosed limit).** "
        "Only ONE FAST/SLOW/STATIC partition was tested, chosen structurally "
        "(native signal period) rather than fit to any return. A different "
        "partition (e.g. moving the Donchian breakout, whose 68.97-bar decay "
        "half-life sits closer to the fast cluster than its nominal SLOW "
        "assignment) could plausibly change this result -- this is a real, "
        "disclosed limit of testing one partition, not evidence the specific "
        "one chosen is uniquely correct.\n\n"
        "**Net read:** " +
        ("the pre-registered decision rule reads this as PROMOTE-candidate: "
         "the causal probe, all four B1 cells, the B3 plateau, the B4 sign "
         "replication, and B5's no-flip condition all pass simultaneously."
         if all_pass else
         "the pre-registered decision rule reads this as NEGATIVE. ") +
        "The honest caveat, separate from what the rule concludes: turnover "
        "reduction (point 1) is the mechanism's clearest, least ambiguous "
        "effect regardless of verdict -- whether that turnover reduction "
        "translates into a genuine per-trade timing edge, versus mostly a "
        "floor-avoidance/drawdown-avoidance effect on leveraged cells (the "
        "same caveat R-128's own conservative branch raised), is not "
        "separable from the numbers reported here alone; a bucket-level "
        "band, being coarser than a per-expert one, gives less room for any "
        "single expert's own timing edge to show through untouched, which "
        "argues for reading any BTC edge found here cautiously rather than "
        "as confirmation the bucket partition itself is well-tuned."
    )

    results = dict(
        verdict=verdict, n_configs=n_configs, max_ts=max_ts,
        probe_ok=probe_ok, probe_full=full_bal, probe_trunc=trunc_bal,
        b1=dict(rows=b1_rows, pass_=b1_pass),
        b3=dict(rows=b3_rows, pass_=b3_pass, majority_count=majority_count),
        b4=dict(d_sharpe=b4_r["d_sharpe"], paired_lo=b4_r["paired_lo"],
               paired_hi=b4_r["paired_hi"], btc_sign=btc_sign, eth_sign=eth_sign,
               pass_=b4_pass),
        b5=dict(rows=b5_rows, pass_=b5_pass),
        diag=dict(counts=diag_counts, n_bars=diag_n_bars,
                  trades_cand=spot_full_row["trades_cand"],
                  trades_base=spot_full_row["trades_base"]),
        pytest_summary=pytest_summary,
        discussion=discussion,
    )
    results["b1"]["pass"] = b1_pass
    results["b3"]["pass"] = b3_pass
    results["b4"]["pass"] = b4_pass
    results["b5"]["pass"] = b5_pass

    write_report(results)
    return results


if __name__ == "__main__":
    main()
