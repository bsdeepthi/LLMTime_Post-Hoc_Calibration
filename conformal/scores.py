import numpy as np


def symmetric_score(q_lo, q_hi, y):
    """
    CQR symmetric nonconformity score.
    s_i = max(q_lo_i - y_i,  y_i - q_hi_i)
    Positive when y falls outside [q_lo, q_hi]; magnitude = how far outside.
    """
    return np.maximum(q_lo - y, y - q_hi)


def asymmetric_scores(q_lo, q_hi, y):
    """
    Separate lower and upper nonconformity scores.
    s_lo_i > 0  when y < q_lo  (lower bound too high)
    s_hi_i > 0  when y > q_hi  (upper bound too low)

    Returns:
        s_lo : ndarray  (q_lo - y)
        s_hi : ndarray  (y - q_hi)
    """
    return q_lo - y, y - q_hi
