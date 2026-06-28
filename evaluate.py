"""
evaluate.py — Probabilistic forecast evaluation metrics

Metrics
-------
- CRPS           : Continuous Ranked Probability Score (sample-based)
- PIT            : Probability Integral Transform values + histogram
- Coverage       : Empirical coverage at multiple nominal levels (50/80/90/95%)
- Width-coverage : Interval width vs coverage curve
- SMAPE          : Symmetric Mean Absolute Percentage Error
- ECE            : Expected Calibration Error (via quantile reliability)
"""

import numpy as np
import matplotlib.pyplot as plt
from calibration.diagnostics import compute_coverage, compute_ece, plot_reliability_diagram


# ── CRPS ──────────────────────────────────────────────────────────────────────

def crps_samples(samples, actuals):
    """
    Energy-score CRPS for sample-based ensemble forecasts.

    Args:
        samples : ndarray [n_samples, n_steps]
        actuals : ndarray [n_steps]

    Returns:
        crps_t  : ndarray [n_steps]  per-step CRPS
        mean_crps : float
    """
    K, T = samples.shape
    crps_t = np.zeros(T)
    k_idx = np.arange(K, dtype=float)

    for t in range(T):
        y = actuals[t]
        x = np.sort(samples[:, t])
        term1 = np.mean(np.abs(x - y))
        # Efficient: 0.5*E[|X-X'|] = (1/K^2) * sum_{j<k} |x_j - x_k|
        #            = (1/K^2) * sum_k x_k * (2k - K + 1)  [0-indexed]
        spread = np.dot(x, 2 * k_idx - K + 1) / (K ** 2)
        crps_t[t] = term1 - spread

    return crps_t, float(np.mean(crps_t))


# ── PIT ───────────────────────────────────────────────────────────────────────

def pit_values(samples, actuals):
    """
    Probability Integral Transform: fraction of samples at or below the actual.

    Args:
        samples : ndarray [n_samples, n_steps]
        actuals : ndarray [n_steps]

    Returns:
        pit : ndarray [n_steps]  values in [0, 1]
    """
    K, T = samples.shape
    return np.array([np.mean(samples[:, t] <= actuals[t]) for t in range(T)])


def plot_pit_histogram(pit_vals, n_bins=10, title="PIT Histogram", ax=None):
    """
    Plot PIT histogram.  Uniform distribution = well-calibrated forecast.
    Left-heavy (U-shaped right) = over-dispersed; right-heavy = under-dispersed.
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(5, 4))

    ax.hist(pit_vals, bins=n_bins, range=(0, 1), density=True,
            edgecolor="black", color="steelblue", alpha=0.7)
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Ideal uniform")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if created:
        plt.tight_layout()
    return ax


# ── Multi-level coverage ───────────────────────────────────────────────────────

COVERAGE_LEVELS = [0.50, 0.80, 0.90, 0.95]


def coverage_at_levels(samples, actuals, nominal_levels=None):
    """
    Empirical coverage at each nominal level using symmetric quantile intervals.

    Args:
        samples        : ndarray [n_samples, n_steps]
        actuals        : ndarray [n_steps]
        nominal_levels : list of floats, defaults to [0.50, 0.80, 0.90, 0.95]

    Returns:
        dict { level -> empirical_coverage }
    """
    if nominal_levels is None:
        nominal_levels = COVERAGE_LEVELS
    result = {}
    for level in nominal_levels:
        alpha = 1.0 - level
        q_lo = np.quantile(samples, alpha / 2.0, axis=0)
        q_hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
        result[level] = float(np.mean((actuals >= q_lo) & (actuals <= q_hi)))
    return result


def interval_coverage_at_levels(q_lo, q_hi, actuals, lo_level, hi_level):
    """
    Empirical coverage for a fixed interval [q_lo, q_hi] treated as a (hi_level - lo_level) interval.
    Convenience wrapper for CP interval evaluation.
    """
    return float(np.mean((actuals >= q_lo) & (actuals <= q_hi)))


# ── Width–coverage curve ───────────────────────────────────────────────────────

def width_coverage_curve(samples, actuals, levels=None):
    """
    Sweep over nominal levels and compute (empirical coverage, avg interval width).

    Args:
        samples : ndarray [n_samples, n_steps]
        actuals : ndarray [n_steps]
        levels  : iterable of floats in (0, 1); defaults to 50 equally spaced values

    Returns:
        nominal   : ndarray   nominal coverage levels
        empirical : ndarray   observed coverage at each level
        widths    : ndarray   average interval width at each level
    """
    if levels is None:
        levels = np.linspace(0.05, 0.99, 50)
    levels = np.asarray(levels)
    empirical = np.zeros_like(levels)
    widths = np.zeros_like(levels)

    for i, level in enumerate(levels):
        alpha = 1.0 - level
        q_lo = np.quantile(samples, alpha / 2.0, axis=0)
        q_hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
        empirical[i] = float(np.mean((actuals >= q_lo) & (actuals <= q_hi)))
        widths[i] = float(np.mean(q_hi - q_lo))

    return levels, empirical, widths


def plot_width_coverage(nominal, empirical, widths, label="", ax=None):
    """
    Dual-axis width–coverage plot.
    Left axis: coverage vs nominal (calibration diagonal).
    Right axis: avg interval width vs nominal.
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(nominal, empirical, "o-", markersize=3, linewidth=1.5,
            label=f"{label} coverage", color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage", color="steelblue")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax2 = ax.twinx()
    ax2.plot(nominal, widths, "s--", markersize=3, linewidth=1.5,
             color="darkorange", label=f"{label} width")
    ax2.set_ylabel("Avg interval width", color="darkorange")

    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    if created:
        plt.tight_layout()
    return ax


# ── Point accuracy ─────────────────────────────────────────────────────────────

def mae(actuals, forecasts):
    return float(np.mean(np.abs(actuals - forecasts)))


def rmse(actuals, forecasts):
    return float(np.sqrt(np.mean((actuals - forecasts) ** 2)))


def smape(actuals, forecasts):
    """Symmetric Mean Absolute Percentage Error (in %)."""
    denom = (np.abs(actuals) + np.abs(forecasts)) / 2.0
    safe_denom = np.where(denom == 0.0, 1.0, denom)
    return float(100.0 * np.mean(np.abs(actuals - forecasts) / safe_denom))


# ── Omnibus summary ────────────────────────────────────────────────────────────

def compute_all_metrics(samples, actuals, quantile_levels, nominal_levels=None):
    """
    Compute the full evaluation suite in one call.

    Args:
        samples        : ndarray [n_samples, n_steps]
        actuals        : ndarray [n_steps]
        quantile_levels: list of float  (e.g. [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        nominal_levels : list of float  coverage levels to test (default: COVERAGE_LEVELS)

    Returns:
        dict with keys: crps, pit, coverage, smape, mae, rmse, ece, pit_vals, crps_t
    """
    if nominal_levels is None:
        nominal_levels = COVERAGE_LEVELS

    crps_t, mean_crps = crps_samples(samples, actuals)
    pit_vals = pit_values(samples, actuals)
    cov_at_levels = coverage_at_levels(samples, actuals, nominal_levels)

    # Quantile-based metrics (use empirical quantiles from samples)
    q_matrix = np.array([
        np.quantile(samples, tau, axis=0) for tau in quantile_levels
    ]).T  # [n_steps, n_levels]

    med_idx = quantile_levels.index(0.50) if 0.50 in quantile_levels else len(quantile_levels) // 2
    median_forecast = q_matrix[:, med_idx]

    cov_dict = compute_coverage(q_matrix, actuals, quantile_levels)
    ece_val = compute_ece(cov_dict, quantile_levels)

    return {
        "crps":         mean_crps,
        "crps_t":       crps_t,
        "pit_vals":     pit_vals,
        "coverage":     cov_at_levels,
        "mae":          mae(actuals, median_forecast),
        "rmse":         rmse(actuals, median_forecast),
        "smape":        smape(actuals, median_forecast),
        "ece":          ece_val,
        "cov_dict":     cov_dict,
        "q_matrix":     q_matrix,
    }


def crps_quantiles(q_matrix, actuals, levels):
    """
    CRPS approximation from discrete quantile levels (quantile decomposition).
    CRPS ≈ (2/L) * sum_k pinball_tau_k(q_k, y)

    Useful when you have calibrated quantiles but not raw samples.

    Args:
        q_matrix : ndarray [n_steps, n_levels]
        actuals  : ndarray [n_steps]
        levels   : list of float (sorted)

    Returns:
        crps_t    : ndarray [n_steps]
        mean_crps : float
    """
    T, L = q_matrix.shape
    crps_t = np.zeros(T)
    for j, tau in enumerate(levels):
        q = q_matrix[:, j]
        crps_t += np.where(actuals >= q,
                           tau * (actuals - q),
                           (1.0 - tau) * (q - actuals))
    crps_t *= 2.0 / L
    return crps_t, float(np.mean(crps_t))


def print_metrics_table(metrics_by_method, nominal_levels=None):
    """
    Pretty-print a comparison table of metrics across methods.
    metrics_by_method : dict { method_label -> metrics_dict }
    """
    if nominal_levels is None:
        nominal_levels = COVERAGE_LEVELS

    cov_headers = " ".join(f"Cov@{int(l*100):2d}%  " for l in nominal_levels)
    header = f"{'Method':<30} {'CRPS':>8} {'MAE':>8} {'SMAPE':>8} {'ECE':>8} {cov_headers}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for label, m in metrics_by_method.items():
        cov_str = " ".join(f"{m['coverage'].get(l, float('nan')):>8.1%}" for l in nominal_levels)
        crps_s  = f"{m['crps']:>8.4f}" if not np.isnan(m['crps'])  else "    —   "
        mae_s   = f"{m['mae']:>8.2f}"  if not np.isnan(m['mae'])   else "    —   "
        smape_s = f"{m['smape']:>7.2f}%" if not np.isnan(m['smape']) else "    —   "
        ece_s   = f"{m['ece']:>8.4f}" if not np.isnan(m['ece'])   else "    —   "
        print(f"{label:<30} {crps_s} {mae_s} {smape_s} {ece_s} {cov_str}")

    print("=" * len(header))
