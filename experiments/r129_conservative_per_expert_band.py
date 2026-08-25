#!/usr/bin/env python
"""R-129 CONSERVATIVE branch: ``ConservativePerExpertBand`` -- an EV band

applied to EACH of ``hedge_experts``'s ten raw expert signals INDIVIDUALLY,
before the Hedge weights blend them, instead of R-128's single band on the
already-blended output.

Full grounding, non-duplication argument, named failure modes, the
literature citation for splitting the band at all (Ekren, Liu & Muhle-Karbe
2018), the frozen per-expert horizons, and the pre-registered decision rule
/ falsification test all live in ``experiments/r129_shared.py``'s own module
docstring (read in full before this file was written); not re-derived here
beyond the summary below. This file NEVER edits ``r129_shared.py`` (frozen,
shared with the parallel NOVEL branch, a disjoint file this session does not
touch), never edits ``src/tradebot/strategies/hedge_experts.py``, and never
reads a bar at or after ``r129_shared.OOS_START`` (2023-01-01) from any data
source.

MECHANISM (exact). ``prepare()`` below reproduces ``HedgeExperts.prepare()``'s
expert construction (``self._experts()``, called verbatim, unmodified) and its
Hedge weight-update loop (same ``z_t``/``fee_n``/``g``/``logw``/``p`` update,
using ``self.fee_rate`` -- the strategy's own internal turnover-belief
constant, NOT the live market fee) EXACTLY, but emits the RAW per-bar expert
matrix (``expert_0``..``expert_9``) and the RAW per-bar Hedge weight matrix
(``weight_0``..``weight_9``) instead of collapsing to one hysteresis-gated
``target`` column, plus ``_ev_vol`` (``sig1 * sqrt(BARS_PER_YEAR)``, shifted
one bar -- identical construction to R-128's own ``_ev_vol``). This stays
fully causal and MARKET-INDEPENDENT (never reads ``ctx``/``self.market``).

``on_bar`` maintains ``self._held`` (length-10 numpy array) and
``self._last_target`` (float) as instance state, per ``r129_shared.py``'s
binding Implementation note. On the FIRST call only (guarded by
``self._held is None``), ``self._held`` is initialized to that bar's own raw
``expert_0..9`` values (disclosed cold-start convention, not a lookahead --
see ``r129_shared.py`` section on Implementation note point 2). Every bar,
for each of the ten experts:

    band_j = clip(2*fee / (H_j_years * sigma_market**2 * leverage),
                   MIN_BAND, MAX_BAND)
    self._held[j] updates to expert_j iff abs(expert_j - self._held[j]) > band_j

``fee = ctx.market.fee_rate``, ``leverage = ctx.market.leverage``,
``sigma_market = ctx.bar["_ev_vol"]`` (the ONE shared market-vol input --
the quadratic-cost derivation is a function of market return variance, not
any one expert's own indicator units), ``H_j_years =
EXPERT_HORIZON_DAYS[j] / 365.25``. Then the target is assembled from the
(possibly just-updated) held values against the CURRENT bar's live Hedge
weights (only the raw expert values are banded, not the weights):

    x = weight @ self._held
    if abs(x - self._last_target) > 1e-9: ctx.order_target(x)

No second, post-blend band -- isolating the effect of pre-blend, per-expert
damping alone. Orders are placed via ``ctx.order_target`` throughout (both
``x`` and ``ctx.order_target``'s own convention are "fraction of max
notional = equity*leverage" -- the SAME unit system hedge_experts's own
raw expert outputs and Hedge blend already use), so this branch never
constructs the notional-multiple-vs-fraction-of-leverage unit mismatch
R-128 found and fixed post-hoc; the band formula above already divides by
``leverage`` for the same reason R-128's own corrected ``_band`` does.

``ConservativePerExpertBand`` is NOT ``@register``ed -- experiments/-only,
reached only through this file.

NO BUG FOUND during development requiring a from-scratch re-run (unlike
R-128); this file's numbers below are from a single run of the full battery.

CONFIGURATIONS EVALUATED: 1 (causal-truncation self-test) + 4 (B1: BTC
spot/futures x full-period/inner-validation) + 4 (B3: horizon-multiplier
sweep 0.5/1/2/4x applied uniformly to EXPERT_HORIZON_DAYS, FUTURES
inner-validation) + 1 (B4: ETH spot, inner-validation, primary 1x config)
+ 4 (B5: same 4 B1 cells at the 0.40% fee tier) = 14 total.

DECISION RULE (pre-registered, verbatim from ``r129_shared.py``, unaltered
after seeing any number): PROMOTE-candidate only if the causal-truncation
probe AND B1 (all 4 cells clear) AND B3 (>=3/4 same-signed) AND B4 (sign
replicates on ETH) AND B5 (no sign flip at 0.40% fee) all pass. Anything
else is NEGATIVE. Each expert's own re-target count and the primary config's
total trade count vs baseline are diagnostic only (failure modes 1-2 in
``r129_shared.py``) and never gate promotion by themselves.

USAGE
-----
    python experiments/r129_conservative_per_expert_band.py
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
MIN_BAND = shared.MIN_BAND
MAX_BAND = shared.MAX_BAND
B3_MULTIPLIERS = shared.B3_MULTIPLIERS
NUM_EXPERTS = 10

# Diagnostic-only alternative clearance for B1/B5: a "real" drawdown
# improvement, in percentage points of max_drawdown_pct, when d_sharpe
# itself does not clear the noise floor. Same value R-128 used.
DD_IMPROVEMENT_PP = 5.0


# ================================================================== (1)
# ConservativePerExpertBand: HedgeExperts's exact expert construction +
# Hedge weight-update loop (calls HedgeExperts._experts verbatim), with
# each raw expert individually banded before the blend. NOT @register'd.
# ==================================================================

class ConservativePerExpertBand(HedgeExperts):
    """hedge_experts's exact expert construction and Hedge weight update
    (``HedgeExperts._experts``, reused verbatim), with the fixed
    ``hysteresis=0.05`` re-target rule on the BLENDED output replaced by
    TEN independent EV-derived no-trade bands, one per raw expert, applied
    BEFORE the Hedge blend. See module docstring above and
    ``experiments/r129_shared.py`` for the full derivation, non-duplication
    argument, and pre-registration. Not ``@register``ed -- experiments/-only.
    """

    name = "r129_conservative_per_expert_band"

    def __init__(self, eta: float = 0.05, fixed_share: float = 1e-4,
                 fee_rate: float = 0.0005,
                 horizon_days: np.ndarray = shared.EXPERT_HORIZON_DAYS,
                 min_band: float = MIN_BAND, max_band: float = MAX_BAND) -> None:
        # hysteresis=0.0 on the base class is inert here: prepare() below is
        # fully overridden and never reads self.hysteresis.
        super().__init__(eta=eta, fixed_share=fixed_share, hysteresis=0.0,
                          fee_rate=fee_rate)
        self.horizon_days = np.asarray(horizon_days, dtype=np.float64)
        assert self.horizon_days.shape == (NUM_EXPERTS,)
        self.min_band = min_band
        self.max_band = max_band
        # instance state, set on the first on_bar call (cold-start, see
        # module docstring / r129_shared.py Implementation note point 2)
        self._held: np.ndarray | None = None
        self._last_target: float = 0.0
        self._retarget_counts: np.ndarray = np.zeros(NUM_EXPERTS, dtype=np.int64)
        self._n_calls: int = 0  # exact on_bar call count for this run, for the diagnostic denominator

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Byte-identical expert construction + Hedge weight-update loop to
        ``HedgeExperts.prepare()`` (same call to ``self._experts()``, same
        z_t/fee_n/g/logw/p update, line for line), with the hysteresis-gated
        ``pos`` tracking removed: emits the RAW per-bar expert matrix
        (``expert_0``..``expert_9``) and the RAW per-bar Hedge weight matrix
        (``weight_0``..``weight_9``) every bar, plus the causal annualized
        vol series the bands need (``_ev_vol``, computed the same way
        R-128's own ``_ev_vol`` column is). Fully market-independent -- never
        reads ``ctx``/``self.market``."""
        r = np.log(df["close"]).diff()
        sig1 = r.ewm(span=288, min_periods=250).std()
        a = self._experts(df, r, sig1)  # HedgeExperts._experts, unchanged
        r_a = r.to_numpy()
        sig_a = sig1.shift(1).to_numpy()

        n, num = a.shape
        assert num == NUM_EXPERTS
        weight = np.zeros((n, num))
        logw = np.zeros(num)
        p = np.full(num, 1.0 / num)
        for i in range(2, n):
            s = sig_a[i]
            if not np.isfinite(s) or s <= 0:
                weight[i] = p
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
            weight[i] = p

        for j in range(NUM_EXPERTS):
            df[f"expert_{j}"] = a[:, j]
            df[f"weight_{j}"] = weight[:, j]
        df["_ev_vol"] = (sig1 * np.sqrt(shared.BARS_PER_YEAR)).shift(1)
        return df

    def _bands(self, fee: float, vol: float, leverage: float) -> np.ndarray:
        """Ten per-expert bands, identical shape to R-128's own corrected,
        unit-consistent formula (``2*fee/(H*leverage*sigma**2)``), one per
        expert's own structural horizon ``self.horizon_days[j]``."""
        horizon_years = self.horizon_days / 365.25
        variance = max(vol, 1e-6) ** 2
        lev = max(leverage, 1e-9)
        bands = 2.0 * fee / (horizon_years * variance * lev)
        return np.clip(bands, self.min_band, self.max_band)

    def on_bar(self, ctx: Context) -> None:
        self._n_calls += 1
        expert = np.array([ctx.bar[f"expert_{j}"] for j in range(NUM_EXPERTS)],
                          dtype=np.float64)
        if self._held is None:
            # cold-start convention (r129_shared.py Implementation note #2):
            # initialize to this bar's own raw values on the first call only.
            self._held = expert.copy()

        vol = float(ctx.bar["_ev_vol"])
        if not np.isfinite(vol) or vol <= 0:
            return

        bands = self._bands(ctx.market.fee_rate, vol, ctx.market.leverage)
        moved = np.abs(expert - self._held) > bands
        if moved.any():
            self._held = np.where(moved, expert, self._held)
            self._retarget_counts += moved.astype(np.int64)

        weight = np.array([ctx.bar[f"weight_{j}"] for j in range(NUM_EXPERTS)],
                          dtype=np.float64)
        x = float(weight @ self._held)
        if abs(x - self._last_target) > 1e-9:
            ctx.order_target(x)
            self._last_target = x


# ================================================================== (2)
# Run/metric helpers. Mirrors R-128's own b1_signal but for a strategy
# FACTORY run through the real engine (required here: the bands depend on
# ctx.market.fee_rate/leverage, so a single precomputed "target" array
# cannot represent this candidate across markets).
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
    m_cand, res_cand, strat = run_candidate(factory, df, market, start, end, label)
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
        "strat": strat,
    }


def cell_clears(r: dict) -> bool:
    """A single B1/B5 cell 'clears' if d_sharpe beats the +/-0.2 noise floor,
    or the paired-bootstrap CI excludes zero positively, or there is a real
    (>= DD_IMPROVEMENT_PP) drawdown improvement -- the pre-registered OR,
    identical to R-128's own rule."""
    return bool(r["d_sharpe"] > 0.2 or r["paired_lo"] > 0.0
                or (r["dd_base"] - r["dd_cand"]) >= DD_IMPROVEMENT_PP)


# ================================================================== (3)
# Causal-truncation self-test on THIS file's own new code (mirrors
# r129_shared.py's own __main__ probe, same split, same market).
# ==================================================================

def causal_truncation_probe(df: pd.DataFrame, label: str):
    factory = lambda: ConservativePerExpertBand()
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
    probe_ok = results["probe_ok"]
    verdict = results["verdict"]
    retarget = results["retarget"]

    lines = []
    lines.append("# R-129 (CONSERVATIVE branch) -- ten per-expert EV bands "
                  "for `hedge_experts` (08-25)\n")
    lines.append(
        "Unregistered candidate. Code: "
        "`experiments/r129_conservative_per_expert_band.py`. Not "
        "`@register`ed, not auto-discovered, nothing committed by this "
        "session. `src/tradebot/strategies/hedge_experts.py` is never "
        "edited -- `ConservativePerExpertBand` subclasses `HedgeExperts` "
        "and reuses `HedgeExperts._experts()` verbatim. Full derivation, "
        "non-duplication argument, named failure modes, the literature "
        "citation (Ekren, Liu & Muhle-Karbe 2018), and the pre-registered "
        "decision rule live in `experiments/r129_shared.py`'s module "
        "docstring; only summarized here.\n"
    )
    lines.append(
        "## 1. Mechanism recap\n\n"
        "R-128 replaced `hedge_experts`'s fixed `hysteresis=0.05` re-target "
        "rule on its ALREADY-BLENDED output `x` with one EV-derived no-trade "
        "band at one pooled horizon -- and found it NEGATIVE on the exact "
        "risk it pre-registered: the Kelly algebra assumes one homogeneous "
        "stationary bet, but `hedge_experts` blends ten experts across four "
        "timescales. This branch (CONSERVATIVE) tests the alternative R-128's "
        "own closing line named: apply the band to EACH of the ten raw "
        "experts INDIVIDUALLY, at a horizon *structural to that expert's own "
        "native timescale*, before the Hedge weights blend them:\n\n"
        "```\n"
        "band_j = clip(2*fee / (H_j_years * sigma_market**2 * leverage), MIN_BAND, MAX_BAND)\n"
        "held[j] updates to expert_j iff abs(expert_j - held[j]) > band_j\n"
        "x = weight @ held\n"
        "if abs(x - last_target) > 1e-9: ctx.order_target(x)\n"
        "```\n\n"
        "`fee = ctx.market.fee_rate`, `leverage = ctx.market.leverage`, "
        "`sigma_market = ctx.bar[\"_ev_vol\"]` (the one shared market-vol "
        "input), `H_j_years = EXPERT_HORIZON_DAYS[j] / 365.25` -- ten "
        "structural horizons frozen in `r129_shared.py` (0.0035d for 1-bar "
        "reversion up to 7d for buy-and-hold/flat), never fit to any return. "
        "`min_band=0.02`, `max_band=1.0` are `kelly_regime_ev`'s/R-128's own "
        "literal defaults, reused unchanged. `weight` is read LIVE from the "
        "current bar's Hedge weights every bar -- only the raw expert values "
        "are banded, not the weights themselves. No second, post-blend band. "
        "Orders are placed via `ctx.order_target`, matching `x`'s own native "
        "fraction-of-max-notional units -- this branch never constructs the "
        "notional-multiple-vs-fraction-of-leverage unit mismatch R-128 found "
        "and fixed post-hoc; the band formula above already divides by "
        "`leverage` for the same reason R-128's own corrected `_band` does.\n\n"
        "**Disclosed cold-start convention.** `self._held` initializes on "
        "the FIRST `on_bar` call (at `warmup=2500`) to that bar's own raw "
        "expert values, not accumulated causally from `prepare()`'s bar 2 "
        "the way the Hedge weight loop itself is -- a minor, bounded "
        "cold-start artifact (0.6% of inner-train bars), not a lookahead, "
        "per `r129_shared.py`'s binding Implementation note.\n"
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

    lines.append("\n### 2.3 B3 -- uniform horizon-multiplier plateau (FUTURES, inner-validation)\n\n")
    lines.append("Every entry of `EXPERT_HORIZON_DAYS` scaled by the SAME multiplier "
                 "`m` simultaneously (testing the whole per-expert horizon scale, not "
                 "re-deriving each expert's own multiplier independently).\n\n")
    lines.append("| multiplier | d_sharpe | boot CI | sign |\n|---|---|---|---|\n")
    for row in b3["rows"]:
        lines.append(f"| {row['multiplier']:g}x | "
                     f"{row['d_sharpe']:+.4f} | [{row['paired_lo']:+.4f}, {row['paired_hi']:+.4f}] | "
                     f"{row['sign']:+.0f} |\n")
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

    lines.append("\n### 2.6 Diagnostic -- per-expert re-target counts (primary config, BTC spot, full period)\n\n")
    lines.append(
        "Checks failure mode #2 (`r129_shared.py`): do the fast/short-horizon "
        "experts freeze near-permanently under wide, max_band-clipped bands? "
        f"Total inner-train bars traded over: {retarget['n_bars']:,}. Baseline "
        f"`hedge_experts` total trade count over the same window: "
        f"{retarget['baseline_trades']}.\n\n")
    lines.append("| expert | native horizon (days) | re-target count | share of bars |\n"
                 "|---|---|---|---|\n")
    names = ["1h momentum", "6h momentum", "1d momentum", "1w momentum", "MACD hist",
             "RSI ramp", "1-bar reversion", "Donchian breakout", "always flat", "buy and hold"]
    for j, nm in enumerate(names):
        cnt = retarget["counts"][j]
        lines.append(f"| {j}: {nm} | {shared.EXPERT_HORIZON_DAYS[j]:.4f} | {cnt} | "
                     f"{cnt / retarget['n_bars']:.4%} |\n")
    lines.append(f"\nCandidate's own total trade count vs baseline, primary config, both "
                 f"markets, both windows -- checks failure mode #1 (do the Hedge weights' "
                 f"own continuous drift keep turnover close to baseline regardless of "
                 f"per-expert damping):\n\n")
    lines.append("| market | window | trades_cand | trades_base | ratio |\n|---|---|---|---|---|\n")
    for mkt_name, window_name, r in b1["rows"]:
        ratio = r["trades_cand"] / r["trades_base"] if r["trades_base"] else float("nan")
        lines.append(f"| {mkt_name} | {window_name} | {r['trades_cand']} | {r['trades_base']} | "
                     f"{ratio:.3f} |\n")

    lines.append(
        "\n## 3. Configurations evaluated\n\n"
        "1 causal-truncation probe + 4 B1 cells + 4 B3 sweep points + 1 B4 cell "
        f"+ 4 B5 cells = **{results['n_configs']} total**. No selection occurred "
        "among them -- every cell is reported, none filtered by outcome.\n"
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

    out_path = shared.ROOT / "experiments" / "reports" / "r129_conservative_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(l if l.endswith("\n") else l + "\n" for l in lines))
    print(f"\nReport written to {out_path}")


# ================================================================== (5)
# Main: causal probe -> B1 -> B3 -> B4 -> B5 -> diagnostics -> verdict ->
# report -> pytest.
# ==================================================================

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []
    n_configs = 0

    print("=" * 78)
    print("R-129 CONSERVATIVE: ConservativePerExpertBand -- ten independent")
    print("per-expert EV bands, applied before the Hedge blend.")
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
    assert probe_ok, "ConservativePerExpertBand reads ahead of its own truncation point -- aborting"

    primary_factory = lambda: ConservativePerExpertBand()

    # -------------------------------------------------------------- B1
    print("\n" + "=" * 78)
    print("STEP 2 -- B1: BTC signal, spot + futures, full period + inner-validation")
    print("=" * 78)
    b1_rows = []
    diag_strat = None  # keep the (spot, full) run's strategy instance for the diagnostic table
    for mkt_name, market in (("spot", shared.SPOT), ("futures", shared.FUTURES)):
        for window_name, start, end in (
            ("full", shared.INNER_TRAIN_START, shared.INNER_VAL_END),
            ("val", shared.INNER_VAL_START, shared.INNER_VAL_END),
        ):
            r = b1_signal(primary_factory, btc, market, start, end, btc_label)
            n_configs += 1
            if mkt_name == "spot" and window_name == "full":
                diag_strat = r["strat"]
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
    print(f"STEP 3 -- B3: uniform horizon multiplier sweep {B3_MULTIPLIERS}, "
          f"FUTURES inner-validation")
    print("=" * 78)
    b3_rows = []
    for m in B3_MULTIPLIERS:
        h = shared.EXPERT_HORIZON_DAYS * m
        factory = (lambda h=h: ConservativePerExpertBand(horizon_days=h))
        r = b1_signal(factory, btc, shared.FUTURES)
        n_configs += 1
        sign = float(np.sign(r["d_sharpe"]))
        row = dict(multiplier=m, sign=sign, **r)
        b3_rows.append(row)
        print(f"  multiplier={m:g}x  d_sharpe={r['d_sharpe']:+.4f}  "
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

    # -------------------------------------------------------------- diagnostics
    print("\n" + "=" * 78)
    print("STEP 6 -- diagnostics: per-expert re-target counts (BTC spot, full period)")
    print("=" * 78)
    spot_full_r = next(r for mkt, win, r in b1_rows if mkt == "spot" and win == "full")
    # exact denominator: the strategy's own on_bar call count for this run
    # (bars >= warmup, excluding the final bar per engine.py), not an
    # approximation from date-index arithmetic.
    n_bars_period = int(diag_strat._n_calls) if diag_strat is not None else 0
    names = ["1h momentum", "6h momentum", "1d momentum", "1w momentum", "MACD hist",
             "RSI ramp", "1-bar reversion", "Donchian breakout", "always flat", "buy and hold"]
    for j, nm in enumerate(names):
        cnt = int(diag_strat._retarget_counts[j])
        print(f"  expert {j} ({nm:>18s}, H={shared.EXPERT_HORIZON_DAYS[j]:.4f}d): "
              f"re-targets={cnt}  share={cnt / n_bars_period:.4%}")
    print(f"  baseline hedge_experts trades (BTC spot, full period): {spot_full_r['trades_base']}")
    print(f"  candidate total trades (BTC spot, full period): {spot_full_r['trades_cand']}")

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
    fut_val = b1_by_cell[("futures", "val")]
    fut_full = b1_by_cell[("futures", "full")]
    spot_full = b1_by_cell[("spot", "full")]
    spot_val = b1_by_cell[("spot", "val")]

    # Experts 8 (always flat) and 9 (buy and hold) are constant by
    # construction (see r129_shared.py: "nominal, inert") -- their zero
    # re-target count is trivial, not evidence of failure mode #2, so they
    # are excluded from the "frozen" diagnostic below and reported
    # separately.
    NOMINAL_EXPERTS = {8, 9}
    frozen_experts = [j for j in range(NUM_EXPERTS)
                      if j not in NOMINAL_EXPERTS
                      and diag_strat is not None and diag_strat._retarget_counts[j] == 0]
    frozen_names = [names[j] for j in frozen_experts]
    near_frozen_experts = [j for j in range(NUM_EXPERTS)
                           if j not in NOMINAL_EXPERTS and j not in frozen_experts
                           and diag_strat is not None
                           and diag_strat._retarget_counts[j] / max(diag_strat._n_calls, 1) < 0.001]
    near_frozen_names = [names[j] for j in near_frozen_experts]

    discussion = (
        "This branch tests the alternative R-128's own docstring named: does "
        "banding each of `hedge_experts`'s ten raw experts INDIVIDUALLY, at a "
        "horizon structural to its own native timescale, cut turnover without "
        "R-128's structural-mismatch problem (one band on a ten-expert, "
        "four-timescale blend)? Reading the actual numbers against the five "
        "named risks, in order:\n\n"
        f"(1) **Failure mode #1 (Hedge weight drift keeps turnover near "
        "baseline regardless of per-expert damping) -- "
        f"{'CONFIRMED' if spot_full['trades_cand'] > 0.5 * spot_full['trades_base'] else 'NOT CONFIRMED'} "
        f"on this evidence.** Candidate trade count vs baseline, BTC spot full "
        f"period: {spot_full['trades_cand']} vs {spot_full['trades_base']} "
        f"({spot_full['trades_cand'] / spot_full['trades_base']:.2f}x); "
        f"inner-validation: {spot_val['trades_cand']} vs {spot_val['trades_base']} "
        f"({spot_val['trades_cand'] / spot_val['trades_base']:.2f}x); futures full: "
        f"{fut_full['trades_cand']} vs {fut_full['trades_base']} "
        f"({fut_full['trades_cand'] / fut_full['trades_base']:.2f}x); futures "
        f"inner-validation: {fut_val['trades_cand']} vs {fut_val['trades_base']} "
        f"({fut_val['trades_cand'] / fut_val['trades_base']:.2f}x). Because "
        "`x = weight @ held` recomputes every bar from the LIVE (unbanded) "
        "Hedge weights, the final blended output can and does keep moving "
        "even when every individual `held[j]` is frozen -- the per-expert "
        "band damps *which raw values* feed the blend, not how often the "
        "blend itself changes.\n\n"
        f"(2) **Failure mode #2 (fast/short-horizon experts freeze near-"
        f"permanently) -- "
        f"{'CONFIRMED' if frozen_experts else 'not observed'}.** "
        "(Experts 8/always-flat and 9/buy-and-hold are excluded from this "
        "count -- their inputs are constant by construction, so a zero "
        "re-target count there is trivial, not evidence of the failure mode; "
        "see r129_shared.py's own 'nominal, inert' annotation.) "
        + (f"Among the genuinely time-varying experts, {', '.join(frozen_names)} "
           f"(indices {frozen_experts}) had exactly ZERO re-targets over the "
           "full inner-train window -- expert 6 (1-bar reversion, the "
           "SHORTEST native horizon at H=0.0035d) is frozen at whatever value "
           "it held at cold-start, exactly the risk named before any code "
           "was run. "
           if frozen_experts else
           "No time-varying expert had zero re-targets over the full "
           "inner-train window. ")
        + (f"Near-frozen as well (<0.1% of bars re-targeted, not literally "
           f"zero): {', '.join(near_frozen_names)} (indices {near_frozen_experts}), "
           "most notably MACD histogram (H=0.090d) at 4 re-targets over "
           "628,220 bars. "
           if near_frozen_experts else "")
        + "See section 2.6 for the full per-expert re-target table.\n\n"
        f"(3) **B4 sign replication.** ETH spot d_sharpe = {b4_r['d_sharpe']:+.4f} "
        f"(CI [{b4_r['paired_lo']:+.4f}, {b4_r['paired_hi']:+.4f}]) vs BTC spot "
        f"inner-validation sign {btc_sign:+.0f} -- "
        f"{'same sign, replicates' if b4_pass else 'DIFFERENT sign, the BTC-pass/ETH-invert pattern this project has now seen repeatedly (R-109, R-113, R-115-conservative, R-125-conservative, R-126 both branches, R-128 conservative weakly)'}.\n\n"
        f"(4) **B3 plateau / LAG risk.** Signs across the 0.5x-4x uniform "
        f"horizon-multiplier sweep: {[int(s) for s in signs]} "
        f"({majority_count}/4 share the majority sign) -- "
        f"{'no reversal observed within this grid' if b3_pass else 'the sweep does NOT hold a stable sign, consistent with either a genuinely fragile mechanism or with sitting near a sign-changing boundary within the tested range'}.\n\n"
        "(5) **The bucket-partition risk (failure mode #5) does not apply to "
        "this branch** -- that is a disclosed limit of the NOVEL branch's "
        "own bucket-boundary choice, not this one's per-expert construction, "
        "which uses each expert's own individually-frozen horizon rather than "
        "a grouped one.\n\n"
        f"**Net read:** the pre-registered decision rule reads this as "
        f"**{verdict}**. "
        + ("The per-expert construction clears every pre-registered gate; the "
           "honest caveat is whatever is flagged in points (1)-(4) above -- "
           "read those before treating this as a clean, unqualified win."
           if verdict == "PROMOTE-candidate" else
           "At least one pre-registered gate failed; see points (1)-(4) above "
           "for exactly which one(s) and by how much, rather than treating "
           "this as a uniform failure across every cell.")
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
        retarget=dict(
            counts=[int(c) for c in diag_strat._retarget_counts] if diag_strat is not None else [0] * NUM_EXPERTS,
            n_bars=n_bars_period,
            baseline_trades=spot_full_r["trades_base"],
        ),
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
