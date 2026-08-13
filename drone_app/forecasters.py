"""
Alternative weekly forecasters — swapping the naive 2-point CAGR for
a proper family of models.  Response to a reader critique that
exponential-growth extrapolation has no asymptote, so a 52-week
projection off an 11-week fit hands back a number that exceeds
physical build capacity.

Each forecaster implements:
  predict_next_week(daily_df, week_start) -> dict with keys
      predicted_total     int      point forecast
      lower_90            int      90% prediction-interval lower
      upper_90            int      90% PI upper
      method_note         str      how it was derived
      diagnostics         dict     model-specific fit info

The dict shape is stable across all forecasters so the multi-model
lock can iterate uniformly.

MODELS
------
  1. log_linear_ols     — same exponential form, uses every point,
                          gives you SE(β̂). Strict upgrade over 2-pt CAGR.
  2. damped_trend_ets   — Gardner's damped Holt (φ ≈ 0.9). Exponential
                          that runs out of steam. Winner of M-competitions.
  3. gompertz           — explicit carrying-capacity K. Warning printed
                          when the K estimate is unstable (pre-inflection).
  4. nb_glm             — Negative Binomial GLM, log link, time trend.
                          Correct error model for overdispersed counts.
  5. rolling_baseline   — kept for direct comparison to the naive baseline.

HYPOTHESES REGISTERED
---------------------
H3: damped_trend_ets outperforms rolling+cagr at 4-8 week horizons
H4: log_linear_ols beats 2-point CAGR at every horizon (uses all data)
H5: nb_glm 90% PIs cover the actual for 85-95% of weeks
H6: gompertz K has CI width > 50% of K until we're past inflection
H7: optimal choice by horizon —
        1w:      nb_glm or rolling
        2-3w:    rolling / log-linear (variance-limited)
        4-12w:   damped_trend_ets (bias-limited begins here)
        13-52w:  scenario bands, not point estimate (policy-driven)

All constants below are LOCKED — changes require a public post first.
"""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Constants (locked; change = new public commitment)
# -----------------------------------------------------------------------------
# Damped-trend Gardner φ. 0.85-0.98 is the operational range; 0.90 is a
# reasonable prior for a production process that is starting to plateau.
DAMPED_PHI            = 0.90

# ETS smoothing parameters (kept mid-range so weights aren't degenerate)
ALPHA_LEVEL           = 0.40
BETA_TREND            = 0.20

# NB-GLM dispersion floor — protects against numerical instability when
# the sample variance/mean is small.
NB_DISPERSION_FLOOR   = 1.0

# Gompertz stability threshold — if K CI half-width > this fraction of
# the point estimate, we flag the model as pre-inflection unreliable.
GOMPERTZ_K_INSTABILITY_RATIO = 0.50

# Prediction-interval z-score for 90% coverage
Z90 = 1.6449


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _prior_history(daily_df: pd.DataFrame, week_start: date) -> pd.DataFrame:
    """Return only nights BEFORE the week starting on `week_start`."""
    d = daily_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    return d[d["date"] < pd.Timestamp(week_start)].sort_values("date").reset_index(drop=True)


def _weekly_totals(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate nightly data into ISO weeks (Mon–Sun)."""
    d = daily_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["week_start"] = (d["date"] - pd.to_timedelta(d["date"].dt.weekday, unit="D")).dt.normalize()
    w = d.groupby("week_start")["launched"].sum().reset_index()
    return w.sort_values("week_start").reset_index(drop=True)


def _pack(pt: float, lo: float, hi: float, note: str, diagnostics: dict) -> dict:
    return {
        "predicted_total": int(round(max(0, pt))),
        "lower_90":        int(round(max(0, lo))),
        "upper_90":        int(round(max(0, hi))),
        "method_note":     note,
        "diagnostics":     {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                             for k, v in diagnostics.items()},
    }


# -----------------------------------------------------------------------------
# 1. Rolling baseline (unchanged — for comparison only)
# -----------------------------------------------------------------------------
def rolling_baseline(daily_df: pd.DataFrame, week_start: date) -> dict:
    prior = _prior_history(daily_df, week_start)
    if len(prior) < 14:
        return _pack(0, 0, 0, "insufficient history",
                     {"n": len(prior)})
    tail14 = prior.tail(14)["launched"]
    daily_mean = tail14.mean()
    pt = daily_mean * 7 * 1.10
    # Prediction interval from residuals
    sd = tail14.std(ddof=1)
    pi_half = Z90 * sd * np.sqrt(7)   # scale nightly sd to weekly
    return _pack(pt, pt - pi_half, pt + pi_half,
                 f"rolling: 14d mean {daily_mean:.1f} × 7 × 1.10 buffer",
                 {"n": len(prior), "daily_mean_14d": daily_mean,
                  "nightly_sd_14d": sd})


# -----------------------------------------------------------------------------
# 2. Log-linear OLS — proper regression on log(launched)
# -----------------------------------------------------------------------------
def log_linear_ols(daily_df: pd.DataFrame, week_start: date) -> dict:
    weekly = _weekly_totals(_prior_history(daily_df, week_start))
    if len(weekly) < 4:
        return _pack(0, 0, 0, "need ≥4 weeks of history",
                     {"n_weeks": len(weekly)})

    # Fit log(y) = a + b*t. t in weeks since first.
    y = weekly["launched"].values.astype(float)
    y = np.clip(y, 1, None)          # avoid log(0)
    t = np.arange(len(y), dtype=float)
    logy = np.log(y)

    n = len(t)
    t_mean = t.mean(); logy_mean = logy.mean()
    Sxx = ((t - t_mean) ** 2).sum()
    Sxy = ((t - t_mean) * (logy - logy_mean)).sum()
    b = Sxy / Sxx
    a = logy_mean - b * t_mean
    resid = logy - (a + b * t)
    sigma2 = (resid ** 2).sum() / max(n - 2, 1)
    se_b = np.sqrt(sigma2 / Sxx)

    # Forecast for next week (t = n)
    t_next = float(n)
    log_pt = a + b * t_next
    pt = float(np.exp(log_pt))
    # PI: log-scale ± Z*sqrt(σ² (1 + 1/n + (t_next - t̄)² / Sxx))
    pi_log_half = Z90 * np.sqrt(sigma2 * (1 + 1/n + (t_next - t_mean)**2 / Sxx))
    lo = float(np.exp(log_pt - pi_log_half))
    hi = float(np.exp(log_pt + pi_log_half))

    weekly_growth = np.exp(b) - 1
    note = (f"log-lin OLS on {n} weeks: β̂={b:.4f} (SE {se_b:.4f}), "
            f"weekly growth {weekly_growth*100:+.2f}%")
    return _pack(pt, lo, hi, note,
                 {"n_weeks": n, "beta_hat": b, "se_beta": se_b,
                  "weekly_growth_pct": weekly_growth * 100,
                  "log_intercept": a, "residual_sigma2": sigma2})


# -----------------------------------------------------------------------------
# 3. Damped-trend ETS (Gardner) — Holt's linear with φ damping
# -----------------------------------------------------------------------------
def damped_trend_ets(daily_df: pd.DataFrame, week_start: date,
                       phi: float = DAMPED_PHI,
                       alpha: float = ALPHA_LEVEL,
                       beta: float = BETA_TREND) -> dict:
    """One-step ahead forecast of next week's total using Gardner's
    damped-trend method fit on WEEKLY totals.

    Recursion:
        L_t = alpha * y_t + (1 - alpha) * (L_{t-1} + phi * b_{t-1})
        b_t = beta * (L_t - L_{t-1}) + (1 - beta) * phi * b_{t-1}
        ŷ_{t+h} = L_t + sum_{i=1..h} phi^i * b_t

    We forecast h=1 week ahead.
    """
    weekly = _weekly_totals(_prior_history(daily_df, week_start))
    if len(weekly) < 3:
        return _pack(0, 0, 0, "need ≥3 weeks of history",
                     {"n_weeks": len(weekly)})

    y = weekly["launched"].values.astype(float)
    n = len(y)
    L = np.zeros(n); B = np.zeros(n)
    L[0] = y[0]
    B[0] = y[1] - y[0] if n > 1 else 0.0
    for i in range(1, n):
        L_prev = L[i-1]; b_prev = B[i-1]
        L[i] = alpha * y[i] + (1 - alpha) * (L_prev + phi * b_prev)
        B[i] = beta * (L[i] - L_prev) + (1 - beta) * phi * b_prev

    pt = L[-1] + phi * B[-1]     # h=1 forecast
    resid = y[1:] - (L[:-1] + phi * B[:-1])
    sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else float(np.std(y) or 1)
    pi_half = Z90 * sd

    return _pack(pt, pt - pi_half, pt + pi_half,
                 f"damped-trend ETS (φ={phi:.2f}, α={alpha:.2f}, β={beta:.2f}) "
                 f"on {n} weeks",
                 {"n_weeks": n, "level_last": L[-1], "trend_last": B[-1],
                  "phi": phi, "residual_sd": sd})


# -----------------------------------------------------------------------------
# 4. Gompertz — asymmetric growth with explicit carrying capacity K
# -----------------------------------------------------------------------------
def gompertz(daily_df: pd.DataFrame, week_start: date) -> dict:
    """Fit y(t) = K * exp(-b * exp(-c*t)) via nonlinear least squares.

    Honest warning: K is nearly unidentifiable pre-inflection. When the
    numerical fit produces a K estimate with wide relative CI, we flag
    the result as unreliable in the note.
    """
    weekly = _weekly_totals(_prior_history(daily_df, week_start))
    if len(weekly) < 6:
        return _pack(0, 0, 0, "need ≥6 weeks of history",
                     {"n_weeks": len(weekly)})

    from scipy.optimize import curve_fit
    y = weekly["launched"].values.astype(float)
    t = np.arange(len(y), dtype=float)

    def gompertz_fn(t, K, b, c):
        return K * np.exp(-b * np.exp(-c * t))

    # Initial guesses: K ≈ 2 × max observed, b so inflection is roughly in
    # the middle of the observed window, c small.
    K0 = float(y.max() * 2)
    b0 = 2.0
    c0 = 0.1
    try:
        popt, pcov = curve_fit(gompertz_fn, t, y, p0=[K0, b0, c0],
                                 maxfev=5000, bounds=([1, 0.01, 0.001], [K0 * 20, 20, 2]))
        K, b, c = popt
        perr = np.sqrt(np.diag(pcov))
        se_K, se_b, se_c = perr
        pt = gompertz_fn(len(y), K, b, c)
        # Approximate PI: propagate residual sigma
        resid = y - gompertz_fn(t, K, b, c)
        sigma = float(np.std(resid, ddof=1))
        pi_half = Z90 * sigma
        # Flag K instability
        k_ci_ratio = float(se_K / K) if K > 0 else float("inf")
        stable = k_ci_ratio < GOMPERTZ_K_INSTABILITY_RATIO
        note = (f"Gompertz fit on {len(y)} weeks: K̂={K:.0f} "
                f"(±{se_K:.0f} = {k_ci_ratio*100:.0f}% of K̂), "
                f"{'STABLE' if stable else 'UNSTABLE — pre-inflection, K unidentifiable'}")
        return _pack(pt, pt - pi_half, pt + pi_half, note,
                     {"n_weeks": len(y), "K": K, "se_K": se_K,
                      "k_ci_ratio": k_ci_ratio, "stable": stable,
                      "b": b, "c": c, "sigma": sigma})
    except Exception as e:
        return _pack(0, 0, 0,
                     f"Gompertz fit failed: {type(e).__name__}: {e}",
                     {"n_weeks": len(y), "error": str(e)})


# -----------------------------------------------------------------------------
# 5. Negative Binomial GLM — correct error model for count data
# -----------------------------------------------------------------------------
def nb_glm(daily_df: pd.DataFrame, week_start: date) -> dict:
    """Fit weekly totals as NegativeBinomial with log link and time trend.

    Uses statsmodels. Returns honest prediction intervals (accounting
    for overdispersion) rather than the artificially narrow Gaussian PIs.
    """
    weekly = _weekly_totals(_prior_history(daily_df, week_start))
    if len(weekly) < 4:
        return _pack(0, 0, 0, "need ≥4 weeks of history",
                     {"n_weeks": len(weekly)})

    try:
        import statsmodels.api as sm
        y = weekly["launched"].values.astype(float)
        t = np.arange(len(y), dtype=float)
        X = sm.add_constant(t)

        # Estimate dispersion first via Poisson fit, then fit NB with alpha
        poisson_fit = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        mu = poisson_fit.mu
        # Method-of-moments alpha
        alpha_hat = float(max(NB_DISPERSION_FLOOR / max(mu.mean(), 1),
                              ((y - mu) ** 2 - mu).sum() / (mu ** 2).sum())) \
                    if len(mu) > 2 else NB_DISPERSION_FLOOR
        alpha_hat = max(alpha_hat, 1e-4)

        nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_hat)).fit()
        t_next = np.array([[1.0, float(len(y))]])
        pt = float(np.exp(nb.predict(t_next))[0])
        # NB variance: mu + alpha * mu^2
        var = pt + alpha_hat * pt ** 2
        se = float(np.sqrt(var))
        return _pack(pt, pt - Z90 * se, pt + Z90 * se,
                     f"NB-GLM (log link) on {len(y)} weeks, α̂={alpha_hat:.3f}, "
                     f"β̂_t={float(nb.params[1]):.3f}",
                     {"n_weeks": len(y), "alpha_hat": alpha_hat,
                      "beta_t": float(nb.params[1]),
                      "predicted_variance": var,
                      "converged": bool(nb.converged)})
    except Exception as e:
        return _pack(0, 0, 0,
                     f"NB-GLM fit failed: {type(e).__name__}: {e}",
                     {"n_weeks": len(weekly), "error": str(e)})


# -----------------------------------------------------------------------------
# Registry — a stable list the multi-model lock iterates over
# -----------------------------------------------------------------------------
FORECASTERS: dict[str, Callable] = {
    "rolling_baseline":  rolling_baseline,
    "log_linear_ols":    log_linear_ols,
    "damped_trend_ets":  damped_trend_ets,
    "gompertz":          gompertz,
    "nb_glm":            nb_glm,
}


if __name__ == "__main__":
    # Ad-hoc smoke test on current data
    from datetime import date
    D = Path(__file__).parent / "data"
    daily = pd.read_csv(D / "daily_totals.csv").dropna(subset=["launched"])
    daily["date"] = pd.to_datetime(daily["date"])
    week = daily["date"].max().date() + timedelta(days=1)
    week = week - timedelta(days=week.weekday())   # next Monday
    print(f"Forecasting week of {week}:\n")
    for name, fn in FORECASTERS.items():
        r = fn(daily, week)
        print(f"  {name:<20}  pt={r['predicted_total']:>5,}  "
              f"[{r['lower_90']:>5,}, {r['upper_90']:>5,}]")
        print(f"    {r['method_note']}")
