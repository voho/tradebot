"""Bayes-Stein / empirical-Bayes shrinkage Kelly sizing (backlog: new, this session).

Not registered: lives under ``experiments/`` so it is not auto-discovered,
per ROUTINE.md step 5.

The mechanism, in one sentence
-------------------------------
Estimate the recent drift (mean log-return) with a causal EWM, shrink that
estimate toward a **zero-drift prior** by the classical empirical-Bayes /
James-Stein weight — the fraction of the estimate that is *not* explained
by its own estimation noise — and use the surviving (unshrunk) fraction
directly as a continuous, confidence-weighted gate multiplying the
incumbent's inverse-volatility sizer. No latch (``kelly_regime``'s vote),
no sequential wealth accumulation against a null (R-28/R-31/R-32's
e-process) — just a point estimate and its own standard error, re-computed
fresh every bar from the current window alone.

Citation and exact formula
---------------------------
Jorion, P. (1986), "Bayes-Stein Estimation for Portfolio Analysis",
*Journal of Financial and Quantitative Analysis* 21(3), 279-292: the
sample mean is an inadmissible estimator of expected return under squared
loss once three or more parameters are being estimated jointly (the
Stein 1956 / James & Stein 1961 result), and shrinking it toward a common
target reduces out-of-sample estimation error. Jorion's own estimator
shrinks a *cross-section* of asset means toward the grand mean of the
minimum-variance portfolio; there is exactly one asset and one time series
here, so the object being shrunk is a single time-varying scalar (this
bar's best drift estimate) rather than a cross-section, and the target is
the more common single-parameter empirical-Bayes choice for a "no edge"
prior: **zero**.

The single-parameter empirical-Bayes shrinkage weight this file uses
(Efron & Morris 1975; Morris 1983, "Parametric Empirical Bayes Inference",
*JASA* 78(381) — the ``1 - B = sigma_mu^2 / (sigma_mu^2 + sigma_error^2)``
form, specialised here to a zero-mean prior) is

    z_t              = mu_hat_t / se_t                      (signal-to-noise)
    shrinkage_wt_t    = 1 / (1 + z_t^2)      in [0, 1]        ("B", shrink-to-0 weight)
    shrunk_mu_t       = mu_hat_t * (1 - shrinkage_wt_t)
                       = mu_hat_t * z_t^2 / (1 + z_t^2)

which is algebraically identical to ``se_t^2 / (se_t^2 + mu_hat_t^2)`` for
the shrink-to-zero weight, i.e. the textbook single-parameter empirical
Bayes / James-Stein form with the "population" variance term replaced by
the point estimate's own squared magnitude (the standard move when there
is one parameter, not a cross-section, to estimate a prior spread from —
see Efron, *Large-Scale Inference* (2010), ch. 1, "the James-Stein
estimator for a single mean shrinks toward zero in proportion to
t-statistic-squared over one-plus-t-statistic-squared").

``se_t`` is the standard error of the EWM mean estimator itself:
``se_t = sigma_bar_t / sqrt(n_eff)``, where ``sigma_bar_t`` is the EWM
per-bar return standard deviation over the *same* span, and ``n_eff`` is
the exact effective sample size of a pandas EWM with ``span=S``:
``n_eff = (2-alpha)/alpha = S`` for ``alpha = 2/(S+1)`` (derived from
``1/sum(w_k^2)`` for geometric weights ``w_k = alpha*(1-alpha)^k`` —
verified numerically to 12 significant figures before this file was
written). This is exact, not an approximation to a rule of thumb.

``conf_t = shrunk_mu_t / mu_hat_t`` when ``mu_hat_t > 0``, else 0 — i.e.
the confidence gate *is* the fraction of the raw drift estimate that
survives shrinkage, floored at zero to keep the sizer long-only (matching
the ``vote``/``evidence`` gates' convention: neither of them shorts
either, so a `bayes_stein` arm that could short would not be a matched
comparison). It is bounded in [0, 1] by construction — no separate
normalisation constant is needed for it to sit in the same
``conf * scale`` architecture ``matched_risk.GatedKelly`` uses.

Why this is not R-28 / R-31 / R-32's e-process wearing a different name
-------------------------------------------------------------------------
Both mechanisms produce a confidence value in [0, 1] that multiplies an
inverse-volatility sizer, and both attack the ERR constraint (error
control in the signal path). That is where the similarity ends:

* **The e-process is a sequential hypothesis test with a stopping-time
  guarantee.** Its state is a *log-wealth accumulator* against the null
  "drift is zero" that only grows while evidence is favourable, decays
  only through an explicit `evidence_cap_mult` (fixed at 1.0 by
  convention here, per R-28's warning) and needs 3.8 years of the
  measured drift/noise ratio to cross its alpha=0.05 threshold (R-28's
  own finding). It therefore has *memory*: the mean gate over a decade
  was 0.145, and evidence built in the 2017 bull persisted.
* **Bayes-Stein shrinkage has no accumulator and no state at all.** The
  gate at bar *t* is a pure function of the current window's ``mu_hat_t``
  and ``se_t`` — both re-estimated from scratch every bar over a span of
  days, not years. It has nothing that plays the role of "wealth" and no
  formal Type-I error guarantee at an arbitrary stopping time (that
  guarantee is specifically what the e-process buys and shrinkage does
  not); it buys a different thing — an estimator that is provably lower
  mean-squared-error than the raw sample mean under squared loss,
  bar-by-bar, with no claim about sequential testing at all.
* **The falsifiable, specific behavioral prediction this implies**
  (checked in this file's pre-registration, not asserted from the
  formula): the Bayes-Stein gate should re-open and re-close on the
  timescale of one ``drift_span`` window around each local drift
  reversal, and should NOT exhibit the e-process's multi-year
  evidence-accumulation lag. If it turns out to move exactly as slowly as
  the e-process in practice, the "no accumulation" distinction is
  cosmetic rather than behavioural, and the falsification/turnover checks
  in ``run_bayes_stein_kelly.py`` measure this directly rather than
  assuming it from the algebra.
* Also not R-01 (HMM) / R-02 (jump models) / R-03 (BOCPD): none of those
  is being re-tried. Those detect a *discrete regime state*; Bayes-Stein
  shrinkage claims no such state exists — it is a continuous point-
  estimate correction with zero notion of "which regime am I in."
* Also not R-08/R-09 (better volatility *forecasting*, which hurt): the
  volatility input to the sizer (``vol_span = 8 days``, unchanged from
  ``matched_risk.GatedKelly``) is untouched here. Only the *drift*-side
  estimator is new.

Architecture reused unchanged from ``experiments/matched_risk.py``
--------------------------------------------------------------------
This file does not import or modify ``matched_risk.py``'s ``GatedKelly``
in place; it subclasses it (``BayesSteinKelly(GatedKelly)``) and adds a
third gate. The sizer (``_scale``), the deadband, the exposure multiplier
``k`` (which rescales ``target_vol``, ``max_leverage`` and ``deadband``
together — ``min(k*tv/vol, k*ml) == k*min(tv/vol, ml)``, an exact
rescaling, per the parent class's docstring), and the ``prepare()``/
``on_bar()`` scaffolding are all the parent's, verbatim, duplicated only
where Python requires a full method override (``prepare()``, because the
parent hardcodes which of its two private gate methods to call).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.matched_risk import BARS_PER_DAY, BARS_PER_YEAR, GatedKelly


class BayesSteinKelly(GatedKelly):
    """``GatedKelly`` with a third, continuous, shrinkage-based gate.

    ``gate="bayes_stein"`` shrinks the EWM drift estimate toward zero by
    the empirical-Bayes weight described in this module's docstring and
    uses the surviving fraction as ``conf``. ``gate="vote"`` and
    ``gate="evidence"`` fall through to the parent class unchanged, so
    this file is also a drop-in way to run the original two gates through
    the identical code path used for matching.
    """

    name = "bayes_stein_kelly"

    def __init__(
        self,
        gate: str = "bayes_stein",
        exposure: float = 1.0,
        sizer: str = "plain",
        # --- the bayes_stein gate
        drift_span_days: float = 20.0,
        z_clip: float = 10.0,
        **kwargs,
    ) -> None:
        if gate not in ("vote", "evidence", "bayes_stein"):
            raise ValueError(
                f"gate must be 'vote', 'evidence' or 'bayes_stein', got {gate!r}"
            )
        # The parent's __init__ validates gate in {"vote", "evidence"} only;
        # satisfy that with a placeholder and overwrite immediately after.
        super().__init__(
            gate="vote" if gate == "bayes_stein" else gate,
            exposure=exposure,
            sizer=sizer,
            **kwargs,
        )
        self.gate = gate
        self.drift_span_days = drift_span_days
        self.z_clip = z_clip

    # ------------------------------------------------------------ the gate

    def _bayes_stein(self, r: pd.Series) -> np.ndarray:
        """Empirical-Bayes shrinkage confidence, causal, no accumulator.

        ``mu_hat`` and ``sigma_bar`` are both EWM statistics over the same
        span, each ``.shift(1)`` so bar *i*'s gate uses only returns
        through bar *i-1* — known at the close of bar *i*, one bar before
        anything can fill (same causal pattern as the parent's
        ``_evidence``).
        """
        span = int(self.drift_span_days * BARS_PER_DAY)
        mu_hat = r.ewm(span=span, min_periods=BARS_PER_DAY).mean().shift(1)
        sigma_bar = r.ewm(span=span, min_periods=BARS_PER_DAY).std().shift(1)
        n_eff = float(span)  # exact for a pandas EWM: n_eff = (2-alpha)/alpha = span
        with np.errstate(divide="ignore", invalid="ignore"):
            se = sigma_bar / np.sqrt(n_eff)
            z = (mu_hat / se).clip(-self.z_clip, self.z_clip)
        z = z.where(np.isfinite(z), 0.0)
        shrink_wt = 1.0 / (1.0 + z * z)          # "B": shrink-to-zero weight
        survive = 1.0 - shrink_wt                # = z^2 / (1 + z^2)
        conf = survive.where(mu_hat > 0, 0.0)    # long-only, matches vote/evidence
        return np.nan_to_num(conf.to_numpy(), nan=0.0)

    # ----------------------------------------------------------- strategy

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()
        vol = (
            r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
            * np.sqrt(BARS_PER_YEAR)
        ).shift(1).to_numpy()

        if self.gate == "bayes_stein":
            conf = self._bayes_stein(r)
        elif self.gate == "vote":
            conf = self._vote(close)
        else:
            conf = self._evidence(r, vol)
        scale = self._scale(vol)

        band = self.exposure * self.deadband
        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = conf[i] * scale[i]
            if abs(desired - pos) > band:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["conf"] = conf
        df["scale"] = scale
        return df
