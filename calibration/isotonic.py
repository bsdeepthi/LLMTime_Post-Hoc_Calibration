import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(raw_quantiles, actuals, levels):
    """
    Fit one isotonic regression per quantile level, mapping raw quantile
    estimates to calibrated values.

    For each level tau the model learns:
        calibrated_q_tau = isotonic_regression(raw_q_tau)
    using (raw_q_tau[t], actual[t]) pairs from the calibration set.

    Args:
        raw_quantiles : ndarray [n_cal, n_levels]
        actuals       : ndarray [n_cal]
        levels        : list of float

    Returns:
        calibrators : dict { tau -> fitted IsotonicRegression }
    """
    calibrators = {}
    for j, tau in enumerate(levels):
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        ir.fit(raw_quantiles[:, j], actuals)
        calibrators[tau] = ir
    return calibrators


def apply_isotonic(raw_quantiles, calibrators, levels):
    """
    Apply fitted isotonic calibrators to raw quantile estimates.
    Monotonicity across levels is enforced with a final sort.

    Args:
        raw_quantiles : ndarray [n, n_levels]
        calibrators   : dict { tau -> IsotonicRegression }
        levels        : list of float

    Returns:
        ndarray [n, n_levels]
    """
    result = np.empty_like(raw_quantiles)
    for j, tau in enumerate(levels):
        result[:, j] = calibrators[tau].predict(raw_quantiles[:, j])
    # Enforce cross-level monotonicity (may break after per-level correction)
    result = np.sort(result, axis=1)
    return result
