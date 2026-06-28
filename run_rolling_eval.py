"""
run_rolling_eval.py — Rolling-origin evaluation pipeline
=========================================================

Two-layer pipeline (Layer 1 calibration + Layer 2 CP) evaluated with a
proper rolling-origin protocol across multiple horizons.

Rolling-origin protocol
-----------------------
  At each origin i:
    context_cal  : series[origin_i - TRAIN_WINDOW : origin_i]
    cal_target   : series[origin_i                : origin_i + CAL_SIZE]
    context_test : series[origin_i + CAL_SIZE - TRAIN_WINDOW : origin_i + CAL_SIZE]
    test_target  : series[origin_i + CAL_SIZE     : origin_i + CAL_SIZE + H]

LLM samples are cached per (dataset, origin_idx, split, horizon).
Delete a cache file to force a fresh API call for that origin.

Usage
-----
  python run_rolling_eval.py                              # ETTh1, 10 origins, H=1/6/24
  python run_rolling_eval.py --dataset exchange           # Exchange Rate (daily)
  python run_rolling_eval.py --n_origins 5               # fewer rolling windows
  python run_rolling_eval.py --horizons 6 24             # specific horizons
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from serialize import SerializerSettings
from llmtime import get_llmtime_predictions_data
from calibration.temperature import fit_temperature, apply_temperature
from calibration.isotonic import fit_isotonic, apply_isotonic
from calibration.diagnostics import compute_coverage, compute_ece, plot_reliability_diagram
from conformal.split_cp import fit_symmetric, apply_symmetric, fit_asymmetric, apply_asymmetric, _cp_level
from conformal.rolling import RollingCP
from conformal.mondrian import assign_tod_group, assign_dow_group, fit_mondrian, apply_mondrian
from datasets.loaders import load_etth1, load_exchange_rate, dataset_summary
from evaluate import (
    crps_samples, crps_quantiles, pit_values, plot_pit_histogram,
    coverage_at_levels, width_coverage_curve, plot_width_coverage,
    mae, rmse, smape, COVERAGE_LEVELS,
)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL        = "gpt-4-turbo"
NUM_SAMPLES  = 10
ALPHA        = 0.20          # target 80% coverage

QUANTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
LO_IDX  = QUANTILE_LEVELS.index(0.10)
HI_IDX  = QUANTILE_LEVELS.index(0.90)
MED_IDX = QUANTILE_LEVELS.index(0.50)

TRAIN_WINDOW = 168   # context steps sent to LLM (1 week of hourly data)
_CAL_SIZE_DEFAULT = 24
CAL_SIZE = _CAL_SIZE_DEFAULT  # may be overridden after args are parsed below

gpt4_hypers = dict(
    alpha=0.3, basic=True, temp=1.0, top_p=0.8,
    settings=SerializerSettings(
        base=10, prec=3, signed=True,
        time_sep=", ", bit_sep="", minus_sign="-",
    ),
)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",    default="etth1", choices=["etth1", "exchange"],
                    help="Dataset to evaluate (default: etth1)")
parser.add_argument("--model",      default="gpt-4-turbo",
                    choices=["gpt-4-turbo", "mistral-small"],
                    help="LLM to use (default: gpt-4-turbo)")
parser.add_argument("--n_origins",  type=int,   default=10,
                    help="Number of rolling origins (default 10)")
parser.add_argument("--horizons",   nargs="+",  type=int, default=[1, 6, 24])
parser.add_argument("--cal_size",   type=int,   default=None,
                    help="Calibration window size (default: 24, or H when --cal_size 0)")
parser.add_argument("--output_dir", default="outputs/rolling")
args = parser.parse_args()

MODEL      = args.model
N_ORIGINS  = args.n_origins
HORIZONS   = args.horizons
OUTPUT_DIR = args.output_dir
if args.cal_size is not None and args.cal_size != 0:
    CAL_SIZE = args.cal_size
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────

if args.dataset == "etth1":
    print("\n[Dataset] ETTh1-OT")
    series = load_etth1(target_col="OT")
    DATASET_LABEL = "ETTh1-OT"
elif args.dataset == "exchange":
    print("\n[Dataset] Exchange Rate")
    series = load_exchange_rate()
    DATASET_LABEL = "Exchange-Rate"

dataset_summary(series)

# Mondrian stratification function depends on data frequency
_freq = pd.infer_freq(series.index[:50]) or ""
_is_daily = _freq.startswith(("D", "B", "W"))
_assign_group = assign_dow_group if _is_daily else assign_tod_group

max_H = max(HORIZONS)
min_len_needed = TRAIN_WINDOW + CAL_SIZE + max_H
if len(series) < min_len_needed:
    raise ValueError(
        f"Series too short ({len(series)}). Need >= {min_len_needed}."
    )

# Space origins evenly over the second half of the dataset (avoid cold-start)
half = len(series) // 2
start_idx = max(TRAIN_WINDOW, half)
end_idx   = len(series) - CAL_SIZE - max_H - 1
origin_points = np.linspace(start_idx, end_idx, N_ORIGINS, dtype=int).tolist()

print(f"  Origins  : {origin_points}")
print(f"  Horizons : {HORIZONS}")

# ── Cache helpers ─────────────────────────────────────────────────────────────

_MODEL_TAG = {
    "gpt-4-turbo":   "",            # no suffix — preserves existing cache filenames
    "mistral-small": "_mistral_sm",
}


def _cache_path(origin_idx, split, horizon, cal_size=None):
    model_tag = _MODEL_TAG.get(MODEL, f"_{MODEL}")
    # Include cal_size in filename when it differs from the 24-step default,
    # so different CAL_SIZE runs don't share (potentially corrupt) cache files.
    cs = cal_size if cal_size is not None else CAL_SIZE
    cal_tag = "" if cs == _CAL_SIZE_DEFAULT else f"_c{cs}"
    # Cal split: shared across horizons when cal_size is fixed; per-horizon when cal_size=H
    if split == "cal":
        h_tag = str(horizon) if args.cal_size == 0 else "shared"
    else:
        h_tag = str(horizon)
    return os.path.join(
        OUTPUT_DIR,
        f"cache_{args.dataset}{model_tag}{cal_tag}_o{origin_idx}_{split}_h{h_tag}.csv",
    )


def _get_samples(train_series, target_series, origin_idx, split, horizon):
    path = _cache_path(origin_idx, split, horizon)
    if os.path.exists(path):
        print(f"    [cache] {os.path.basename(path)}")
        df = pd.read_csv(path, index_col=0)
        return df.values  # [K, steps]

    print(f"    [API  ] origin={origin_idx} {split} H={horizon} ...")
    result = get_llmtime_predictions_data(
        train=train_series, test=target_series,
        model=MODEL, num_samples=NUM_SAMPLES, **gpt4_hypers,
    )
    df = result["samples"]
    df.to_csv(path)
    return df.values


# ── Interval coverage/width helper ────────────────────────────────────────────

def _interval_metrics(lo, hi, act):
    cov   = float(np.mean((act >= lo) & (act <= hi)))
    width = float(np.mean(hi - lo))
    return cov, width


# ── Main evaluation loop ──────────────────────────────────────────────────────

all_rows = []

for H in HORIZONS:
    # --cal_size 0 means "match calibration window to test horizon"
    if args.cal_size == 0:
        CAL_SIZE = H
    print(f"\n{'='*65}")
    print(f"  Horizon H = {H}  (CAL_SIZE={CAL_SIZE})")
    print(f"{'='*65}")

    # Accumulators across origins
    pool_test_samples = []   # list of [K, test_steps] arrays
    pool_actuals_test = []   # list of [test_steps] arrays
    pool_pit_raw      = []
    pool_pit_l1       = []
    pool_crps_raw_t   = []
    pool_crps_l1_t    = []

    # Per-origin CP (existing: fit per-origin, 24 cal scores each)
    cp_lo  = {m: [] for m in ["sym", "asym", "roll", "mond"]}
    cp_hi  = {m: [] for m in ["sym", "asym", "roll", "mond"]}
    cp_act = []

    # Storage for pooled CP (fit once on all 10×24 = 240 cal scores)
    per_origin_data = []

    for i, origin in enumerate(origin_points):
        print(f"\n  -- Origin {i+1}/{N_ORIGINS}  (index {origin}) --")

        # Fixed-length context window ending at cal boundary / test boundary
        ctx_cal  = series.iloc[origin - TRAIN_WINDOW          : origin]
        ctx_test = series.iloc[origin + CAL_SIZE - TRAIN_WINDOW : origin + CAL_SIZE]
        cal_s    = series.iloc[origin                          : origin + CAL_SIZE]
        test_s   = series.iloc[origin + CAL_SIZE              : origin + CAL_SIZE + H]

        actuals_cal  = cal_s.values.astype(float)
        actuals_test = test_s.values.astype(float)

        # LLM samples: cal period (forecasting CAL_SIZE steps)
        try:
            cal_samples  = _get_samples(ctx_cal,  cal_s,  i, "cal",  H)
            test_samples = _get_samples(ctx_test, test_s, i, "test", H)
        except Exception as exc:
            print(f"    [SKIP] origin {i} H={H} failed after all retries: {exc}")
            continue

        # Clip to actual lengths in case LLM returned fewer tokens
        n_cal  = min(cal_samples.shape[1],  len(actuals_cal))
        n_test = min(test_samples.shape[1], len(actuals_test))
        cal_samples   = cal_samples[:,  :n_cal]
        test_samples  = test_samples[:, :n_test]
        actuals_cal   = actuals_cal[:n_cal]
        actuals_test  = actuals_test[:n_test]

        if n_cal < 3 or n_test < 1:
            print(f"    Skipping — insufficient steps (cal={n_cal}, test={n_test})")
            continue

        # ── Layer 1: temperature + isotonic calibration ───────────────────

        raw_q_cal  = np.array([np.quantile(cal_samples[:,  t], QUANTILE_LEVELS)
                                for t in range(n_cal)])   # [n_cal, 7]
        raw_q_test = np.array([np.quantile(test_samples[:, t], QUANTILE_LEVELS)
                                for t in range(n_test)])  # [n_test, 7]

        T_temp   = fit_temperature(raw_q_cal, actuals_cal, QUANTILE_LEVELS)
        iso_cals = fit_isotonic(raw_q_cal,    actuals_cal, QUANTILE_LEVELS)

        iso_q_cal  = apply_isotonic(
            apply_temperature(raw_q_cal,  T_temp, QUANTILE_LEVELS),
            iso_cals, QUANTILE_LEVELS)
        iso_q_test = apply_isotonic(
            apply_temperature(raw_q_test, T_temp, QUANTILE_LEVELS),
            iso_cals, QUANTILE_LEVELS)

        q_lo_cal  = iso_q_cal[:, LO_IDX];   q_hi_cal  = iso_q_cal[:, HI_IDX]
        q_lo_test = iso_q_test[:, LO_IDX];  q_hi_test = iso_q_test[:, HI_IDX]

        # ── Layer 2: per-origin CP variants ──────────────────────────────

        Q_sym, scores_sym               = fit_symmetric(q_lo_cal, q_hi_cal, actuals_cal, ALPHA)
        lo_sym, hi_sym                  = apply_symmetric(q_lo_test, q_hi_test, Q_sym)

        (Q_lo_a, Q_hi_a), s_lo_a, s_hi_a = fit_asymmetric(
            q_lo_cal, q_hi_cal, actuals_cal, ALPHA)
        lo_a, hi_a = apply_asymmetric(q_lo_test, q_hi_test, Q_lo_a, Q_hi_a)

        rcp = RollingCP(alpha=ALPHA, window=None)
        rcp.fit(scores_sym)
        lo_r, hi_r = [], []
        for t in range(n_test):
            l, h, _ = rcp.predict(q_lo_test[t], q_hi_test[t])
            lo_r.append(l); hi_r.append(h)
            rcp.update(q_lo_test[t], q_hi_test[t], actuals_test[t])
        lo_r = np.array(lo_r);  hi_r = np.array(hi_r)

        gc  = _assign_group(cal_s.index[:n_cal])
        gt  = _assign_group(test_s.index[:n_test])
        thr, _ = fit_mondrian(q_lo_cal, q_hi_cal, actuals_cal, gc, ALPHA)
        lo_m, hi_m, _ = apply_mondrian(q_lo_test, q_hi_test, gt, thr)

        # ── Accumulate per-origin CP intervals ────────────────────────────

        for k, lo, hi in [("sym",  lo_sym, hi_sym),
                           ("asym", lo_a,   hi_a),
                           ("roll", lo_r,   hi_r),
                           ("mond", lo_m,   hi_m)]:
            cp_lo[k].extend(lo.tolist())
            cp_hi[k].extend(hi.tolist())
        cp_act.extend(actuals_test.tolist())

        # ── Store data needed for pooled CP (second pass) ─────────────────

        per_origin_data.append(dict(
            q_lo_test    = q_lo_test.copy(),
            q_hi_test    = q_hi_test.copy(),
            actuals_test = actuals_test.copy(),
            scores_sym   = scores_sym.copy(),
            s_lo_a       = s_lo_a.copy(),
            s_hi_a       = s_hi_a.copy(),
            gc           = gc.copy(),
            gt           = gt.copy(),
        ))

        # ── Other metric accumulators ─────────────────────────────────────

        pool_test_samples.append(test_samples)
        pool_actuals_test.append(actuals_test)

        pool_pit_raw.extend(pit_values(test_samples, actuals_test).tolist())

        for t in range(n_test):
            pit_val = float(np.interp(
                actuals_test[t],
                iso_q_test[t, :],
                QUANTILE_LEVELS,
            ))
            pool_pit_l1.append(pit_val)

        crps_t_raw, _ = crps_samples(test_samples, actuals_test)
        pool_crps_raw_t.append(crps_t_raw)

        crps_t_l1, _ = crps_quantiles(iso_q_test, actuals_test, QUANTILE_LEVELS)
        pool_crps_l1_t.append(crps_t_l1)

    # ── Aggregate across origins ──────────────────────────────────────────────

    if not pool_actuals_test:
        print(f"  No valid origins — skipping H={H}")
        continue

    all_actuals = np.concatenate(pool_actuals_test)
    all_samples = np.hstack(pool_test_samples)           # [K, total_steps]

    q_mat_raw = np.array([np.quantile(all_samples, tau, axis=0)
                           for tau in QUANTILE_LEVELS]).T  # [total, 7]

    crps_raw = float(np.mean(np.concatenate(pool_crps_raw_t)))
    crps_l1  = float(np.mean(np.concatenate(pool_crps_l1_t)))

    cov_dict_raw = compute_coverage(q_mat_raw, all_actuals, QUANTILE_LEVELS)
    ece_raw      = compute_ece(cov_dict_raw, QUANTILE_LEVELS)
    med_raw      = q_mat_raw[:, MED_IDX]
    mae_raw      = mae(all_actuals, med_raw)
    smape_raw    = smape(all_actuals, med_raw)
    cov_multi    = coverage_at_levels(all_samples, all_actuals)

    pit_raw_arr = np.array(pool_pit_raw)
    pit_l1_arr  = np.array(pool_pit_l1)
    cp_act_arr  = np.array(cp_act)

    # ── Pooled CP: fit once on all origins' cal scores (10×24=240 scores) ────

    all_scores_sym = np.concatenate([d["scores_sym"] for d in per_origin_data])
    all_s_lo_a     = np.concatenate([d["s_lo_a"]     for d in per_origin_data])
    all_s_hi_a     = np.concatenate([d["s_hi_a"]     for d in per_origin_data])
    all_gc         = np.concatenate([d["gc"]         for d in per_origin_data])

    n_pool = len(all_scores_sym)
    Q_sym_pool = float(np.quantile(all_scores_sym, _cp_level(n_pool, ALPHA)))
    Q_lo_pool  = float(np.quantile(all_s_lo_a,     _cp_level(len(all_s_lo_a), ALPHA)))
    Q_hi_pool  = float(np.quantile(all_s_hi_a,     _cp_level(len(all_s_hi_a), ALPHA)))

    # Mondrian pooled: per-group threshold from all origins combined
    from conformal.scores import symmetric_score as _sym_score
    thr_pool = {"global": Q_sym_pool}
    for g in np.unique(all_gc):
        mask = all_gc == g
        n_g  = mask.sum()
        if n_g >= 3:
            scores_g = all_scores_sym[mask]
            thr_pool[g] = float(np.quantile(scores_g, _cp_level(n_g, ALPHA)))
        else:
            thr_pool[g] = Q_sym_pool

    # Pooled Rolling CP: initialise with all cal scores, then test sequentially
    rcp_pool = RollingCP(alpha=ALPHA, window=None)
    rcp_pool.fit(all_scores_sym)

    cp_pool_lo = {m: [] for m in ["sym", "asym", "roll", "mond"]}
    cp_pool_hi = {m: [] for m in ["sym", "asym", "roll", "mond"]}

    for d in per_origin_data:
        qlo = d["q_lo_test"]; qhi = d["q_hi_test"]; act = d["actuals_test"]

        lo_sp, hi_sp = apply_symmetric(qlo, qhi, Q_sym_pool)
        lo_ap, hi_ap = apply_asymmetric(qlo, qhi, Q_lo_pool, Q_hi_pool)

        lo_rp, hi_rp = [], []
        for t in range(len(act)):
            l, h, _ = rcp_pool.predict(qlo[t], qhi[t])
            lo_rp.append(l); hi_rp.append(h)
            rcp_pool.update(qlo[t], qhi[t], act[t])

        lo_mp, hi_mp, _ = apply_mondrian(qlo, qhi, d["gt"], thr_pool)

        cp_pool_lo["sym"].extend(lo_sp.tolist())
        cp_pool_hi["sym"].extend(hi_sp.tolist())
        cp_pool_lo["asym"].extend(lo_ap.tolist())
        cp_pool_hi["asym"].extend(hi_ap.tolist())
        cp_pool_lo["roll"].extend(lo_rp)
        cp_pool_hi["roll"].extend(hi_rp)
        cp_pool_lo["mond"].extend(lo_mp.tolist())
        cp_pool_hi["mond"].extend(hi_mp.tolist())

    cp_pool_act_arr = cp_act_arr   # same actuals

    # ── Print results ─────────────────────────────────────────────────────────

    print(f"\n  === Aggregate Results  H={H}  ({N_ORIGINS} origins) ===")
    sep = "-" * 95
    hdr = (f"  {'Method':<33} {'CRPS':>8} {'MAE':>8} {'SMAPE%':>7} "
           f"{'ECE':>7} {'Cov50':>7} {'Cov80':>7} {'Cov90':>7} {'Cov95':>7}")
    print(sep); print(hdr); print(sep)

    def _print_row(label, crps_v, mae_v, smape_v, ece_v, cov_d, cov80_override=None):
        c50 = cov_d.get(0.50, float("nan"))
        c80 = cov80_override if cov80_override is not None else cov_d.get(0.80, float("nan"))
        c90 = cov_d.get(0.90, float("nan"))
        c95 = cov_d.get(0.95, float("nan"))
        all_rows.append(dict(
            dataset=DATASET_LABEL, horizon=H, method=label,
            crps=crps_v, mae=mae_v, smape=smape_v, ece=ece_v,
            cov_50=c50, cov_80=c80, cov_90=c90, cov_95=c95
        ))
        def _f(v):
            return f"{v:>7.4f}" if not (v != v) else "     — "
        def _p(v):
            return f"{v:>7.1%}" if not (v != v) else "     — "
        print(f"  {label:<33} {_f(crps_v)} {_f(mae_v)} {_f(smape_v)} "
              f"{_f(ece_v)} {_p(c50)} {_p(c80)} {_p(c90)} {_p(c95)}")

    _print_row("Raw LLM", crps_raw, mae_raw, smape_raw, ece_raw, cov_multi)
    _print_row("L1: Isotonic", crps_l1, float("nan"), float("nan"), float("nan"),
               {}, cov80_override=cov_multi.get(0.80))

    print(f"  {'--- Per-origin CP (24 cal scores each) ---':<33}")
    for k, label in [("sym",  "L2: Sym CP"),
                     ("asym", "L2: Asym CP"),
                     ("roll", "L2: Adaptive CP"),
                     ("mond", "L2: Mondrian CP")]:
        lo = np.array(cp_lo[k]);  hi = np.array(cp_hi[k])
        c80_cp = float(np.mean((cp_act_arr >= lo) & (cp_act_arr <= hi)))
        _print_row(label, float("nan"), float("nan"), float("nan"), float("nan"),
                   {}, cov80_override=c80_cp)

    print(f"  {'--- Pooled CP ({} cal scores total) ---'.format(n_pool):<33}")
    for k, label in [("sym",  "L2: Pooled Sym CP"),
                     ("asym", "L2: Pooled Asym CP"),
                     ("roll", "L2: Pooled Adaptive CP"),
                     ("mond", "L2: Pooled Mondrian CP")]:
        lo = np.array(cp_pool_lo[k]); hi = np.array(cp_pool_hi[k])
        c80_cp = float(np.mean((cp_pool_act_arr >= lo) & (cp_pool_act_arr <= hi)))
        _print_row(label, float("nan"), float("nan"), float("nan"), float("nan"),
                   {}, cov80_override=c80_cp)

    print(sep)
    print(f"  Pooled CP uses {n_pool} calibration scores "
          f"({N_ORIGINS} origins × {n_pool // N_ORIGINS} steps each)")

    # ── Plots ─────────────────────────────────────────────────────────────────

    fig = plt.figure(figsize=(17, 10))
    fig.suptitle(f"{DATASET_LABEL}  —  H={H}  ({N_ORIGINS} rolling origins)",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    # [0,0] PIT — Raw LLM
    ax = fig.add_subplot(gs[0, 0])
    plot_pit_histogram(pit_raw_arr, title=f"PIT Histogram — Raw LLM  (H={H})", ax=ax)

    # [0,1] PIT — After Layer 1
    ax = fig.add_subplot(gs[0, 1])
    plot_pit_histogram(pit_l1_arr, title=f"PIT Histogram — L1 Calibrated  (H={H})", ax=ax)

    # [0,2] Width–coverage curve
    ax = fig.add_subplot(gs[0, 2])
    nom, emp, wid_arr = width_coverage_curve(all_samples, all_actuals)
    plot_width_coverage(nom, emp, wid_arr, label="Raw LLM", ax=ax)
    ax.set_title(f"Width–Coverage Curve  (H={H})")

    # [1,0] Reliability diagram — raw
    ax = fig.add_subplot(gs[1, 0])
    plot_reliability_diagram(cov_dict_raw, QUANTILE_LEVELS, "Reliability — Raw LLM", ax)

    # [1,1] CP 80% coverage bar chart — per-origin vs pooled
    ax = fig.add_subplot(gs[1, 1])
    cp_keys   = ["sym", "asym", "roll", "mond"]
    cp_labels = ["Sym", "Asym", "Adaptive", "Mondrian"]
    per_covs  = [float(np.mean(
                     (cp_act_arr >= np.array(cp_lo[k])) &
                     (cp_act_arr <= np.array(cp_hi[k]))
                 )) for k in cp_keys]
    pool_covs = [float(np.mean(
                     (cp_pool_act_arr >= np.array(cp_pool_lo[k])) &
                     (cp_pool_act_arr <= np.array(cp_pool_hi[k]))
                 )) for k in cp_keys]
    x = np.arange(len(cp_keys))
    bars1 = ax.bar(x - 0.2, per_covs,  0.35, label="Per-origin",  color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + 0.2, pool_covs, 0.35, label="Pooled (240)", color="darkorange", alpha=0.8)
    ax.axhline(0.80, color="red", linestyle="--", linewidth=1.5, label="Target 80%")
    ax.set_ylim(0, 1.10); ax.set_ylabel("80% Empirical Coverage")
    ax.set_xticks(x); ax.set_xticklabels(cp_labels)
    ax.set_title(f"Per-origin vs Pooled CP  (H={H})")
    ax.legend(fontsize=7)
    for bar, c in zip(list(bars1) + list(bars2), per_covs + pool_covs):
        ax.text(bar.get_x() + bar.get_width() / 2, min(c + 0.02, 1.04),
                f"{c:.0%}", ha="center", va="bottom", fontsize=7)

    # [1,2] Multi-level coverage (raw LLM)
    ax = fig.add_subplot(gs[1, 2])
    nom_lvls = COVERAGE_LEVELS
    emp_lvls = [cov_multi.get(l, float("nan")) for l in nom_lvls]
    x2 = np.arange(len(nom_lvls))
    ax.bar(x2 - 0.2, nom_lvls, 0.35, alpha=0.5, color="lightgray", label="Nominal")
    ax.bar(x2 + 0.2, emp_lvls, 0.35, alpha=0.85, color="steelblue", label="Empirical (Raw)")
    ax.set_xticks(x2)
    ax.set_xticklabels([f"{int(l*100)}%" for l in nom_lvls])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Coverage")
    ax.set_title(f"Coverage @ 50/80/90/95%  (H={H})")
    ax.legend(fontsize=8)

    plot_path = os.path.join(OUTPUT_DIR, f"eval_{args.dataset}_H{H}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot: {plot_path}")

# ── Save results CSV ──────────────────────────────────────────────────────────

results_df = pd.DataFrame(all_rows)
csv_path = os.path.join(OUTPUT_DIR, f"results_{args.dataset}.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nResults CSV: {csv_path}")

# ── Final printout ────────────────────────────────────────────────────────────

print(f"\n{'='*90}")
print(f"  FULL RESULTS — {DATASET_LABEL}")
print(f"{'='*90}")
with pd.option_context("display.float_format", "{:.4f}".format,
                       "display.max_columns", 20,
                       "display.width", 120):
    print(results_df.to_string(index=False))
print(f"{'='*90}\nDone.")
