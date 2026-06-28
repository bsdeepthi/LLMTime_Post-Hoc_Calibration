import numpy as np
from collections import deque
from conformal.scores import symmetric_score


class RollingCP:
    """
    Adaptive conformal predictor with a rolling calibration window.

    Starts with an initial pool of calibration scores (from the cal set),
    then optionally grows the pool as each test step's actual value is revealed.
    This simulates operational deployment where ground truth arrives with a lag.

    Args:
        alpha  : float   miscoverage rate (e.g. 0.20 for 80% coverage target)
        window : int | None
            None  → growing window: all observed scores are retained.
            int   → fixed-size window: only the most recent `window` scores.
    """

    def __init__(self, alpha, window=None):
        self.alpha  = alpha
        self.window = window
        self._pool  = deque(maxlen=window) if window is not None else []

    def fit(self, cal_scores):
        """Seed the pool with calibration-set nonconformity scores."""
        for s in cal_scores:
            self._pool.append(float(s))

    def _threshold(self):
        pool  = list(self._pool)
        n     = len(pool)
        level = min(1.0, np.ceil((1 - self.alpha) * (n + 1)) / n)
        return float(np.quantile(pool, level))

    def predict(self, q_lo, q_hi):
        """
        Return the CP-adjusted interval and current threshold Q̂
        for one test step. Call update() afterwards with the realized value.

        Args:
            q_lo, q_hi : float   calibrated quantile estimates for this step

        Returns:
            lo, hi, Q : floats
        """
        Q  = self._threshold()
        return q_lo - Q, q_hi + Q, Q

    def update(self, q_lo, q_hi, actual):
        """
        Observe the realized value for this step and add its nonconformity
        score to the pool before predicting the next step.

        Args:
            q_lo, q_hi : float   the pre-CP calibrated quantiles used for this step
            actual     : float   observed ground-truth value
        """
        score = float(symmetric_score(
            np.array([q_lo]), np.array([q_hi]), np.array([actual])
        )[0])
        self._pool.append(score)
