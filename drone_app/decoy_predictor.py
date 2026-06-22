"""
Sequential decoy-budget estimator.

Two math families compose here, as outlined:

A) POISSON / NEG-BIN regression sizes the WAVE.
   Expected total decoys for the week:
       λ = exp(β0 + β1·log(weekly_budget) + ...)
   With a clean budget anchor and historical share r ≈ 0.35:
       λ ≈ r · weekly_budget   (linear approximation good enough at our N)

B) BETA-BINOMIAL posterior updates the share as the week unfolds.
   Prior Beta(α0, β0) with α0/(α0+β0) = r and strength B_prior:
       α0 = r · B_prior
       β0 = (1−r) · B_prior
   After observing K decoys in L launches mid-week:
       α_post = α0 + K
       β_post = β0 + L − K
       posterior_mean = α_post / (α_post + β_post)

   This is robust if Russia front-loads decoys (Bayesian shrinkage pulls
   the posterior toward the prior; the prior is informed by the budget
   anchor and the previous week's actuals).

C) The CLASSIFIER decision threshold uses asymmetric costs:
       threshold* = C_FP / (C_FP + C_FN)
   C_FP = wasted interceptor on a decoy ($500K)
   C_FN = missed warhead ($1–3M damage + lives)
   So threshold* is small (~0.15–0.30) — fire on most ambiguous tracks.

These compose: the posterior decoy share is the per-track PRIOR fed to
the classifier. As the week progresses, the prior shifts and the
classifier's effective threshold shifts with it (you become more
willing to engage when the wave is decoy-heavy and you've used up
your decoys-this-week budget).
"""
from __future__ import annotations
from dataclasses import dataclass
from math import exp, log
from typing import Optional


# Historical Shahed share = 0.65 → decoy share = 0.35 (mean across 15
# pre-May-4 night summaries with explicit counts; σ = 0.036).
HISTORICAL_DECOY_SHARE = 0.35


@dataclass
class DecoyWeekState:
    """Snapshot of a week's decoy accounting at time t."""
    weekly_budget: int                  # total launches expected this week
    launched_so_far: int                # cumulative launches observed
    decoys_identified_so_far: int       # confirmed/classified decoys
    historical_share: float = HISTORICAL_DECOY_SHARE
    prior_strength: int = 800           # Bayesian "fake-N" for the prior

    # ---------- Poisson / linear-approx sizing of the week ----------
    @property
    def expected_decoys_total_week(self) -> float:
        """Prior expectation, sized off the budget anchor."""
        return self.historical_share * self.weekly_budget

    @property
    def expected_shaheds_total_week(self) -> float:
        return (1 - self.historical_share) * self.weekly_budget

    # ---------- Beta-Binomial posterior ----------
    @property
    def prior_alpha(self) -> float:
        return self.historical_share * self.prior_strength

    @property
    def prior_beta(self) -> float:
        return (1 - self.historical_share) * self.prior_strength

    @property
    def posterior_alpha(self) -> float:
        return self.prior_alpha + self.decoys_identified_so_far

    @property
    def posterior_beta(self) -> float:
        return self.prior_beta + (self.launched_so_far - self.decoys_identified_so_far)

    @property
    def posterior_decoy_share(self) -> float:
        """Best estimate of P(next track is decoy), given observations."""
        a, b = self.posterior_alpha, self.posterior_beta
        return a / (a + b) if (a + b) > 0 else self.historical_share

    @property
    def posterior_std(self) -> float:
        """Beta posterior std-dev — uncertainty in P(decoy)."""
        a, b = self.posterior_alpha, self.posterior_beta
        n = a + b
        if n <= 1:
            return 0.0
        return ((a * b) / ((n ** 2) * (n + 1))) ** 0.5

    # ---------- The user's headline formula ----------
    @property
    def launched_remaining(self) -> int:
        return max(self.weekly_budget - self.launched_so_far, 0)

    @property
    def expected_decoys_remaining(self) -> float:
        """E[D_remaining] = L_remaining · posterior_decoy_share.

        This is the headline number to budget interceptors against.
        Self-corrects: if Russia front-loaded decoys, posterior_decoy_share
        drops and the remaining-decoy estimate shrinks accordingly.
        """
        return self.launched_remaining * self.posterior_decoy_share

    @property
    def expected_shaheds_remaining(self) -> float:
        return self.launched_remaining - self.expected_decoys_remaining

    # ---------- Confidence band (90% credible interval) ----------
    def remaining_decoys_band(self, conf: float = 0.90) -> tuple[float, float]:
        """Approximate credible interval via beta-posterior std × z."""
        from scipy.stats import beta
        a, b = self.posterior_alpha, self.posterior_beta
        lo, hi = beta.ppf((1 - conf) / 2, a, b), beta.ppf(1 - (1 - conf) / 2, a, b)
        return self.launched_remaining * lo, self.launched_remaining * hi


def optimal_threshold(c_fp: float, c_fn: float) -> float:
    """Bayes-optimal classifier threshold for asymmetric costs.

    Fire on tracks with P(warhead) >= threshold. Equivalently, fire on
    tracks with P(decoy) <= 1 − threshold.

    With C_FN >> C_FP, threshold is small — Ukraine should over-engage.
    """
    if c_fp + c_fn <= 0:
        return 0.5
    return c_fp / (c_fp + c_fn)


def cost_weighted_engagement(
    p_warhead: float,
    c_fp: float,
    c_fn: float,
    interceptor_cost: float,
    damage_if_missed_mid: float,
) -> dict:
    """Should I engage this track?

    Expected cost of FIRING:
        E[fire] = p_decoy · c_fp + 0      (decoy intercepted = cost spent)
              ≈ p_decoy · interceptor_cost

    Expected cost of HOLDING fire:
        E[hold] = p_warhead · c_fn + 0
              ≈ p_warhead · damage_if_missed_mid

    Fire when E[fire] < E[hold].
    """
    p_decoy = 1 - p_warhead
    e_fire = p_decoy * c_fp
    e_hold = p_warhead * c_fn
    return {
        'p_warhead': p_warhead,
        'p_decoy': p_decoy,
        'expected_cost_if_fire': e_fire,
        'expected_cost_if_hold': e_hold,
        'optimal_action': 'FIRE' if e_fire < e_hold else 'HOLD',
        'threshold': optimal_threshold(c_fp, c_fn),
    }
