import numpy as np
from conformal.scores import symmetric_score
from conformal.split_cp import _cp_level


def assign_tod_group(timestamps):
    """
    Assign each timestamp a time-of-day group.

    Groups:
        night     : 00-05
        morning   : 06-11
        afternoon : 12-17
        evening   : 18-23
    """
    hours = timestamps.hour
    groups = np.empty(len(hours), dtype=object)
    groups[(hours >= 0)  & (hours < 6)]  = "night"
    groups[(hours >= 6)  & (hours < 12)] = "morning"
    groups[(hours >= 12) & (hours < 18)] = "afternoon"
    groups[(hours >= 18) & (hours < 24)] = "evening"
    return groups


def assign_dow_group(timestamps):
    """
    Assign each daily timestamp a day-of-week group.

    Groups:
        monday / tuesday / wednesday / thursday / friday / weekend
    """
    dow = timestamps.dayofweek   # 0=Mon … 6=Sun
    names = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    # Clip to valid index range before fancy-indexing; np.where evaluates
    # both branches unconditionally, so Saturday(5)/Sunday(6) would be OOB.
    safe_idx = np.minimum(dow, 4)
    groups = np.where(dow < 5, np.array(names)[safe_idx], "weekend")
    return groups


def fit_mondrian(q_lo_cal, q_hi_cal, actuals_cal, groups_cal, alpha, min_group_size=3):
    """
    Fit per-group conformal thresholds (Mondrian CP).
    Groups with fewer than `min_group_size` calibration points fall back
    to the global threshold.

    Args:
        q_lo_cal, q_hi_cal : ndarray [n_cal]
        actuals_cal        : ndarray [n_cal]
        groups_cal         : ndarray [n_cal]  string group labels
        alpha              : float
        min_group_size     : int

    Returns:
        thresholds : dict { group -> Q }   (includes 'global' key)
        group_info : dict { group -> n_cal }
    """
    scores_all = symmetric_score(q_lo_cal, q_hi_cal, actuals_cal)

    # Global fallback
    level_global       = _cp_level(len(scores_all), alpha)
    Q_global           = float(np.quantile(scores_all, level_global))
    thresholds         = {"global": Q_global}
    group_info         = {}

    for g in np.unique(groups_cal):
        mask = groups_cal == g
        n    = mask.sum()
        group_info[g] = int(n)
        if n < min_group_size:
            thresholds[g] = Q_global          # fallback
        else:
            scores_g   = scores_all[mask]
            level_g    = _cp_level(n, alpha)
            thresholds[g] = float(np.quantile(scores_g, level_g))

    return thresholds, group_info


def apply_mondrian(q_lo_test, q_hi_test, groups_test, thresholds):
    """
    Apply per-group conformal thresholds to the test quantile estimates.

    Args:
        q_lo_test, q_hi_test : ndarray [n_test]
        groups_test          : ndarray [n_test]  string group labels
        thresholds           : dict { group -> Q }

    Returns:
        lo_cp, hi_cp : ndarray [n_test]
        Q_used       : ndarray [n_test]  threshold applied at each step
    """
    Q_used = np.array([
        thresholds.get(g, thresholds["global"]) for g in groups_test
    ], dtype=float)
    return q_lo_test - Q_used, q_hi_test + Q_used, Q_used
