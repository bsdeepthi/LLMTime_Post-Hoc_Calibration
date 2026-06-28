import numpy as np
from conformal.scores import symmetric_score, asymmetric_scores


def _cp_level(n, alpha):
    """
    Finite-sample conformal quantile level.
    Guarantees empirical coverage >= 1-alpha by construction.
    level = min(1, ceil((1-alpha)(n+1) / n))
    """
    return min(1.0, np.ceil((1 - alpha) * (n + 1)) / n)


def fit_symmetric(q_lo_cal, q_hi_cal, actuals_cal, alpha):
    """
    Fit symmetric CQR threshold on calibration set.

    Args:
        q_lo_cal, q_hi_cal : ndarray [n_cal]  calibrated lower/upper quantiles
        actuals_cal        : ndarray [n_cal]
        alpha              : float            miscoverage rate (e.g. 0.20 for 80%)

    Returns:
        Q      : float   conformal threshold
        scores : ndarray nonconformity scores
    """
    scores = symmetric_score(q_lo_cal, q_hi_cal, actuals_cal)
    level  = _cp_level(len(scores), alpha)
    Q      = float(np.quantile(scores, level))
    return Q, scores


def apply_symmetric(q_lo, q_hi, Q):
    """
    Expand quantile interval by conformal threshold Q.
    Returns (lo_cp, hi_cp).
    """
    return q_lo - Q, q_hi + Q


def fit_asymmetric(q_lo_cal, q_hi_cal, actuals_cal, alpha):
    """
    Fit asymmetric CQR: separate lower/upper thresholds.
    Corrects each tail independently — useful when the LLM's distribution is skewed.

    Returns:
        Q_lo, Q_hi : float   separate thresholds
        s_lo, s_hi : ndarray nonconformity scores per side
    """
    s_lo, s_hi = asymmetric_scores(q_lo_cal, q_hi_cal, actuals_cal)
    level       = _cp_level(len(s_lo), alpha)
    Q_lo        = float(np.quantile(s_lo, level))
    Q_hi        = float(np.quantile(s_hi, level))
    return (Q_lo, Q_hi), s_lo, s_hi


def apply_asymmetric(q_lo, q_hi, Q_lo, Q_hi):
    """
    Expand quantile interval with separate per-side corrections.
    Returns (lo_cp, hi_cp).
    """
    return q_lo - Q_lo, q_hi + Q_hi
