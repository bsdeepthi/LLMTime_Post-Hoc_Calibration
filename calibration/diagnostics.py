import numpy as np
import matplotlib.pyplot as plt


def compute_coverage(quantiles, actuals, levels):
    """
    For each quantile level tau, compute the empirical fraction of actuals
    that fall below the predicted quantile (empirical CDF).

    A perfectly calibrated model satisfies: empirical_coverage(tau) == tau.

    Args:
        quantiles : ndarray [n, n_levels]
        actuals   : ndarray [n]
        levels    : list of float

    Returns:
        dict { tau -> empirical_coverage }
    """
    return {
        tau: float(np.mean(actuals <= quantiles[:, j]))
        for j, tau in enumerate(levels)
    }


def compute_ece(coverage_dict, levels):
    """
    Expected Calibration Error: mean absolute deviation of empirical
    coverage from the nominal level.

    Args:
        coverage_dict : dict { tau -> empirical_coverage }
        levels        : list of float

    Returns:
        float
    """
    return float(np.mean([abs(coverage_dict[tau] - tau) for tau in levels]))


def plot_reliability_diagram(coverage_dict, levels, title, ax):
    """
    Plot a reliability diagram on a given Axes.

    Args:
        coverage_dict : dict { tau -> empirical_coverage }
        levels        : list of float
        title         : str
        ax            : matplotlib Axes
    """
    expected = list(levels)
    observed = [coverage_dict[tau] for tau in levels]

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.fill_between(expected, expected, observed, alpha=0.20, color="orange")
    ax.plot(expected, observed, "o-", color="steelblue",
            linewidth=2, markersize=6, label="Model")

    ece = compute_ece(coverage_dict, levels)
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Expected coverage (tau)")
    ax.set_ylabel("Empirical coverage")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
