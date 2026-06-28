import numpy as np
from scipy.optimize import minimize_scalar


def _pinball_loss(quantiles, actuals, levels):
    total = 0.0
    for j, tau in enumerate(levels):
        r = actuals - quantiles[:, j]
        total += np.mean(np.where(r >= 0, tau * r, (tau - 1) * r))
    return total / len(levels)


def fit_temperature(raw_quantiles, actuals, levels):
    """
    Find scalar T that minimises mean pinball loss on the calibration set.

    Scales each quantile around the per-step median:
        q_cal_tau = median + T * (q_raw_tau - median)

    Args:
        raw_quantiles : ndarray [n_cal, n_levels]
        actuals       : ndarray [n_cal]
        levels        : list of float (must contain 0.50)

    Returns:
        T : float
    """
    median_idx = levels.index(0.50)
    medians = raw_quantiles[:, median_idx : median_idx + 1]  # [n_cal, 1]

    def loss(T):
        return _pinball_loss(medians + T * (raw_quantiles - medians), actuals, levels)

    result = minimize_scalar(loss, bounds=(0.01, 30.0), method="bounded")
    return float(result.x)


def apply_temperature(raw_quantiles, T, levels):
    """
    Apply temperature scaling to an array of raw quantile estimates.

    Args:
        raw_quantiles : ndarray [n, n_levels]
        T             : float  (fitted temperature)
        levels        : list of float (must contain 0.50)

    Returns:
        ndarray [n, n_levels]
    """
    median_idx = levels.index(0.50)
    medians = raw_quantiles[:, median_idx : median_idx + 1]
    return medians + T * (raw_quantiles - medians)


def apply_temperature_to_samples(samples, T):
    """
    Scale individual forecast samples around their per-step median.
    Used to produce a fan plot that reflects the calibrated spread.

    Args:
        samples : ndarray [n_samples, n_steps]
        T       : float

    Returns:
        ndarray [n_samples, n_steps]
    """
    medians = np.median(samples, axis=0, keepdims=True)  # [1, n_steps]
    return medians + T * (samples - medians)
