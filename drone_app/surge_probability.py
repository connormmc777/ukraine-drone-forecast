"""
Daily surge probability model.

MATHEMATICAL FRAMEWORK
======================
Binary classification: is tomorrow a "surge day"?
Surge = daily launches ≥ threshold (default 300).

Model form (logistic regression):

    z_t = β0 + β1·x1 + β2·x2 + β3·x3 + β4·x4 + β5·x5
    P(surge_t) = 1 / (1 + exp(-z_t))

where the features x_i are:
    x1 = days_since_last_surge      (hazard-rate proxy — hazard grows with time)
    x2 = rolling_14day_average      (baseline tempo; higher tempo = more surge freq)
    x3 = buffer_estimate            (production − launches over trailing 14 days)
    x4 = trend_slope_7day           (positive = accelerating; increases surge risk)
    x5 = post_surge_penalty         (=1 if surge occurred yesterday; stockpile drained)

The five features together encode:
    - "when did we last surge?"    (x1)
    - "what's the ambient tempo?"  (x2)
    - "how much stockpile is available?"  (x3)
    - "which direction is momentum?"      (x4)
    - "did we just deplete the buffer?"   (x5)

TRAINING
========
Coefficients β can be:
  (a) Fit via maximum-likelihood estimation on historical (features, label) pairs
      using scipy.optimize.minimize on the negative log-likelihood, OR
  (b) Hand-set from operational intuition, which gives sensible ordinal behavior
      even without a fit.

This module implements BOTH:
  - `fit_coefficients(daily_series, threshold)` — MLE fit on your data
  - `DEFAULT_COEFFS` — sensible hand-tuned defaults if you have very little data

ALTERNATE FRAMINGS
==================
The same problem could be modeled as:
  - Poisson regression for surge COUNTS in a window
  - Cox proportional hazards (survival analysis) treating surges as failure events
  - Neural net with the same features (learns nonlinearity, needs much more data)

Logistic is chosen because (i) we have ≤ 80 daily observations so far, (ii) the
features are near-linear in log-odds, and (iii) coefficients are interpretable.

USAGE
=====
    from surge_probability import SurgeModel
    m = SurgeModel(threshold=300)
    m.fit(daily_df)                    # daily_df: date, launched
    result = m.predict_next(daily_df, target_date)
    # -> {'p_surge': 0.087, 'features': {...}, 'contributions': {...}}
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# Hand-tuned defaults (used if no fit is performed).
# Derived from operational intuition + spot-checks on the current data.
DEFAULT_COEFFS = {
    'intercept':                -2.8,   # base: sigmoid(-2.8) ≈ 5.7% ambient
    'days_since_last_surge':    0.18,   # +18% log-odds per day since last surge
    'rolling_14day_avg':        0.005,  # slight positive: higher tempo = more surges
    'buffer_estimate':          0.0009, # buffer accumulation raises surge risk
    'trend_slope_7day':         0.020,  # positive slope raises surge risk
    'post_surge_penalty':      -1.60,   # just-surged: log-odds drop, stockpile drained
}


DEFAULT_SURGE_THRESHOLD = 300         # drones / day
DEFAULT_PRODUCTION_RATE = 283         # Russia's reverse-engineered belt throughput


@dataclass
class SurgeFeatures:
    days_since_last_surge: int
    rolling_14day_avg: float
    buffer_estimate: float
    trend_slope_7day: float
    post_surge_penalty: float  # 1.0 if yesterday was a surge, else 0.0

    def as_array(self) -> np.ndarray:
        return np.array([
            self.days_since_last_surge,
            self.rolling_14day_avg,
            self.buffer_estimate,
            self.trend_slope_7day,
            self.post_surge_penalty,
        ])

    def as_dict(self) -> dict:
        return {
            'days_since_last_surge': self.days_since_last_surge,
            'rolling_14day_avg': round(self.rolling_14day_avg, 2),
            'buffer_estimate': round(self.buffer_estimate, 1),
            'trend_slope_7day': round(self.trend_slope_7day, 2),
            'post_surge_penalty': self.post_surge_penalty,
        }


def _extract_features(daily_df: pd.DataFrame, target_date: date,
                       threshold: int = DEFAULT_SURGE_THRESHOLD,
                       prod_rate: int = DEFAULT_PRODUCTION_RATE) -> SurgeFeatures:
    """Compute the 5-feature vector for `target_date` using only data
    strictly BEFORE that date (no look-ahead)."""
    hist = daily_df[daily_df['date'] < pd.Timestamp(target_date)].sort_values('date')

    # Feature 1: days since last surge (capped at 30 to avoid runaway hazard)
    surges = hist[hist['launched'] >= threshold]
    if surges.empty:
        days_since = 30
    else:
        last_surge_date = surges.iloc[-1]['date']
        days_since = min(30, (pd.Timestamp(target_date) - last_surge_date).days)

    # Feature 2: 14-day rolling average of daily launches
    last14 = hist['launched'].tail(14)
    rolling_14 = float(last14.mean()) if len(last14) else 0.0

    # Feature 3: buffer estimate = production − launches over trailing 14 days
    consumed_14 = float(last14.sum())
    produced_14 = prod_rate * len(last14)
    buffer = produced_14 - consumed_14

    # Feature 4: linear-regression slope of last-7-days launches
    last7 = hist['launched'].tail(7).values
    if len(last7) >= 3:
        x = np.arange(len(last7))
        slope, _ = np.polyfit(x, last7, 1)
    else:
        slope = 0.0

    # Feature 5: was yesterday a surge?
    if not hist.empty:
        yesterday_launched = float(hist.iloc[-1]['launched'])
        post_surge = 1.0 if yesterday_launched >= threshold else 0.0
    else:
        post_surge = 0.0

    return SurgeFeatures(
        days_since_last_surge=int(days_since),
        rolling_14day_avg=rolling_14,
        buffer_estimate=float(buffer),
        trend_slope_7day=float(slope),
        post_surge_penalty=post_surge,
    )


def _sigmoid(z: float) -> float:
    if z >= 0:
        e = np.exp(-z)
        return float(1.0 / (1.0 + e))
    e = np.exp(z)
    return float(e / (1.0 + e))


def _neg_log_likelihood(beta: np.ndarray, X: np.ndarray, y: np.ndarray,
                          l2: float = 0.5) -> float:
    """L2-regularized logistic loss."""
    z = X @ beta
    # numerically stable log(1 + exp(z))
    log1p_exp_z = np.where(z > 0,
                            z + np.log1p(np.exp(-z)),
                            np.log1p(np.exp(z)))
    loss = np.mean(log1p_exp_z - y * z)
    loss += l2 * (beta[1:] ** 2).sum()  # don't penalize intercept
    return loss


class SurgeModel:
    """Daily surge classifier using logistic regression on 5 features."""

    def __init__(self, threshold: int = DEFAULT_SURGE_THRESHOLD,
                 production_rate: int = DEFAULT_PRODUCTION_RATE):
        self.threshold = threshold
        self.production_rate = production_rate
        self.coeffs: dict = dict(DEFAULT_COEFFS)
        self.fitted: bool = False
        self.training_stats: dict = {}

    # ---------- Prediction ----------
    def _z_from_features(self, feats: SurgeFeatures) -> float:
        c = self.coeffs
        return (c['intercept']
                + c['days_since_last_surge'] * feats.days_since_last_surge
                + c['rolling_14day_avg']    * feats.rolling_14day_avg
                + c['buffer_estimate']      * feats.buffer_estimate
                + c['trend_slope_7day']     * feats.trend_slope_7day
                + c['post_surge_penalty']   * feats.post_surge_penalty)

    def predict_next(self, daily_df: pd.DataFrame,
                       target_date: date | None = None) -> dict:
        """Predict surge probability for `target_date` (defaults to tomorrow)."""
        daily_df = daily_df.copy()
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        target_date = target_date or (daily_df['date'].max().date() + timedelta(days=1))
        feats = _extract_features(daily_df, target_date,
                                    threshold=self.threshold,
                                    prod_rate=self.production_rate)
        z = self._z_from_features(feats)
        p = _sigmoid(z)

        c = self.coeffs
        contribs = {
            'intercept':               c['intercept'],
            'days_since_last_surge':   c['days_since_last_surge'] * feats.days_since_last_surge,
            'rolling_14day_avg':       c['rolling_14day_avg']    * feats.rolling_14day_avg,
            'buffer_estimate':         c['buffer_estimate']      * feats.buffer_estimate,
            'trend_slope_7day':        c['trend_slope_7day']     * feats.trend_slope_7day,
            'post_surge_penalty':      c['post_surge_penalty']   * feats.post_surge_penalty,
        }

        return {
            'target_date': str(target_date),
            'p_surge': p,
            'z': z,
            'threshold': self.threshold,
            'features': feats.as_dict(),
            'contributions_to_z': {k: round(v, 3) for k, v in contribs.items()},
            'fitted': self.fitted,
        }

    def predict_next_week(self, daily_df: pd.DataFrame,
                           start_date: date | None = None,
                           n_days: int = 7) -> list[dict]:
        """Roll forward n_days from start_date (uses today's actuals for day 1)."""
        daily_df = daily_df.copy()
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        start_date = start_date or (daily_df['date'].max().date() + timedelta(days=1))
        results = []
        for i in range(n_days):
            d = start_date + timedelta(days=i)
            r = self.predict_next(daily_df, target_date=d)
            results.append(r)
        return results

    # ---------- Fitting ----------
    def _build_training_matrix(self, daily_df: pd.DataFrame
                                ) -> tuple[np.ndarray, np.ndarray, list]:
        """Build (X, y) pairs — for each historical day, use only prior data."""
        daily_df = daily_df.copy()
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df = daily_df.sort_values('date').reset_index(drop=True)

        rows, labels, dates = [], [], []
        for i in range(3, len(daily_df)):
            target_date = daily_df.iloc[i]['date'].date()
            feats = _extract_features(daily_df, target_date,
                                        threshold=self.threshold,
                                        prod_rate=self.production_rate)
            label = 1 if daily_df.iloc[i]['launched'] >= self.threshold else 0
            rows.append([1.0, feats.days_since_last_surge,
                          feats.rolling_14day_avg, feats.buffer_estimate,
                          feats.trend_slope_7day, feats.post_surge_penalty])
            labels.append(label)
            dates.append(target_date)
        return np.array(rows), np.array(labels), dates

    def fit(self, daily_df: pd.DataFrame, l2: float = 0.5) -> dict:
        """MLE fit via scipy.optimize.minimize. Falls back to hand-tuned
        coefficients if the dataset is too small or degenerate."""
        X, y, dates = self._build_training_matrix(daily_df)
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)

        if len(X) < 20 or n_pos < 3 or n_neg < 3:
            # Not enough data; keep hand-tuned defaults
            self.fitted = False
            self.training_stats = {
                'note': 'fit skipped (insufficient data)',
                'n_days': int(len(X)),
                'n_surges': n_pos,
                'n_non_surges': n_neg,
            }
            return self.training_stats

        try:
            from scipy.optimize import minimize
            beta0 = np.array([DEFAULT_COEFFS['intercept'],
                              DEFAULT_COEFFS['days_since_last_surge'],
                              DEFAULT_COEFFS['rolling_14day_avg'],
                              DEFAULT_COEFFS['buffer_estimate'],
                              DEFAULT_COEFFS['trend_slope_7day'],
                              DEFAULT_COEFFS['post_surge_penalty']])
            res = minimize(_neg_log_likelihood, beta0, args=(X, y, l2),
                            method='BFGS')
            beta = res.x
            self.coeffs = {
                'intercept':               float(beta[0]),
                'days_since_last_surge':   float(beta[1]),
                'rolling_14day_avg':       float(beta[2]),
                'buffer_estimate':         float(beta[3]),
                'trend_slope_7day':        float(beta[4]),
                'post_surge_penalty':      float(beta[5]),
            }
            # Compute in-sample accuracy
            z = X @ beta
            preds = (z > 0).astype(int)
            acc = float((preds == y).mean())
            self.fitted = True
            self.training_stats = {
                'n_days_trained_on': int(len(X)),
                'n_surges': n_pos,
                'n_non_surges': n_neg,
                'base_rate': round(n_pos / len(y), 3),
                'in_sample_accuracy': round(acc, 3),
                'converged': bool(res.success),
                'final_loss': round(float(res.fun), 4),
            }
        except Exception as e:
            self.fitted = False
            self.training_stats = {
                'note': f'fit failed: {type(e).__name__}: {e}',
                'n_days': int(len(X)),
            }
        return self.training_stats
