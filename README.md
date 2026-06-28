# Probabilistic Time Series Forecasting with LLMs: A Post-Hoc Calibration Framework

**MSc Research — Liverpool John Moores University**  
**Author:** B S Deepthi  
**Models:** GPT-4-Turbo (primary), Mistral-Small (exploratory)  
**Datasets:** ETTh1-OT (primary), Exchange Rate SGD/USD (cross-validation)

---

## Overview

This repository implements and evaluates a three-layer post-hoc calibration framework that converts raw GPT-4-Turbo zero-shot ensemble forecasts into statistically reliable prediction intervals.

| Layer | Component | Purpose |
|-------|-----------|---------|
| **Layer 0** | LLMTime (GPT-4-Turbo) | Zero-shot ensemble sampling via serialised decimal text |
| **Layer 1** | Temperature Scaling + Isotonic Regression | Distributional recalibration of quantile estimates |
| **Layer 2** | Split Conformal Prediction (4 variants) | Distribution-free 80% coverage guarantee |

Four conformal prediction variants are compared: **Symmetric CP**, **Asymmetric CP**, **Adaptive Rolling CP**, and **Mondrian CP** (time-of-day or day-of-week stratified).

---

## Reproducing Results — No API Key Required

All GPT-4-Turbo API responses are cached as CSV files and committed to this repository. You can reproduce every table and figure in the thesis **without an OpenAI API key**:

```bash
git clone https://github.com/bsdeepthi/LLMTime_Post-Hoc_Calibration.git
cd LLMTime_Post-Hoc_Calibration
pip install -r requirements.txt
pip install python-docx        # for thesis generation only

# Reproduce ETTh1 results (uses cached samples — no API call)
python run_rolling_eval.py

# Reproduce Exchange Rate results (uses cached samples — no API call)
python run_rolling_eval.py --dataset exchange
```

An API key is only needed if you want to **re-query the model** (e.g., delete a cache file and re-run that origin).

---

## Key Results

### ETTh1-OT (Primary — Hourly, Seasonal)

| Horizon | Raw LLM Cov80% | Best Method | Best Cov80% | Gap |
|---------|---------------|-------------|-------------|-----|
| H = 1   | 50.0%         | Mondrian CP | 60.0%       | −20 pp |
| H = 6   | 46.7%         | Adaptive CP | 51.7%       | −28 pp |
| H = 24  | 50.4%         | Adaptive CP | **70.0%**   | −10 pp |

### Exchange Rate SGD/USD (Cross-Validation — Daily, Stationary)

| Horizon | Raw LLM Cov80% | Mondrian CP Cov80% | Pooled Sym CP Cov80% |
|---------|---------------|-------------------|---------------------|
| H = 1   | 50.0%         | **80.0% ✓**       | **80.0% ✓**         |
| H = 6   | 43.3%         | 66.7%             | 78.3%               |
| H = 24  | 55.4%         | 47.9%             | 50.0%               |

**Primary finding:** The coverage gap on ETTh1 is caused by **seasonal non-stationarity** violating conformal prediction's exchangeability assumption — not by calibration pool size. On the stationary Exchange Rate series, Mondrian CP and all three Pooled CP variants achieve the nominal 80% guarantee at H=1.

---

## Repository Structure

```
.
├── run_rolling_eval.py          # Main evaluation pipeline (start here)
├── generate_plots.py            # Generate all thesis figures from cached results
├── generate_thesis.py           # Generate thesis Word document
│
├── llmtime.py                   # LLMTime serialisation and sampling wrapper
├── serialize.py                 # Numerical serialisation (base-10, prec=3)
├── evaluate.py                  # CRPS, ECE, PIT, coverage metrics
│
├── models/
│   └── llms.py                  # GPT-4-Turbo and Mistral-Small API wrappers
│
├── calibration/
│   ├── temperature.py           # Temperature scaling (Layer 1a)
│   ├── isotonic.py              # Isotonic regression calibration (Layer 1b)
│   └── diagnostics.py           # PIT, reliability diagram utilities
│
├── conformal/
│   ├── split_cp.py              # Symmetric and Asymmetric CP
│   ├── rolling.py               # Adaptive Rolling CP
│   ├── mondrian.py              # Mondrian CP (time-of-day and day-of-week strata)
│   └── scores.py                # Nonconformity score functions
│
├── datasets/
│   └── loaders.py               # ETTh1, Exchange Rate, Electricity, M4 loaders
│
├── outputs/
│   ├── rolling/
│   │   ├── cache_etth1_o*_*.csv      # Cached GPT-4-Turbo samples (ETTh1)
│   │   ├── cache_exchange_o*_*.csv   # Cached GPT-4-Turbo samples (Exchange Rate)
│   │   ├── results_etth1.csv         # Aggregated ETTh1 results
│   │   ├── results_exchange.csv      # Aggregated Exchange Rate results
│   │   └── eval_*_H*.png             # Per-horizon diagnostic plots
│   └── figures/                      # Thesis figures (from generate_plots.py)
│
├── requirements.txt
├── .env.example                 # API key template (copy to .env)
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/bsdeepthi/LLMTime_Post-Hoc_Calibration.git
cd LLMTime_Post-Hoc_Calibration
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. API key (only needed to re-query the model)

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-...        # for GPT-4-Turbo
MISTRAL_API_KEY=...          # for Mistral-Small (optional)
```

> All results in the thesis were produced with `gpt-4-turbo`. The Mistral-Small option is exploratory — see the Limitations section of the thesis.

### 3. Datasets

The ETTh1 and Exchange Rate datasets are downloaded automatically on first run and cached to `Data/`. No manual download is needed.

| Dataset | Description | Direct URL |
|---------|-------------|------------|
| **ETTh1** | Electricity Transformer Temperature, hourly, Jul 2016 – Jun 2018 (Zhou et al., 2021) | https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv |
| **Exchange Rate** | 8 currency rates vs USD, daily, 1990–2010 (Lai et al., 2018) | https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz |

To download manually, pass the URLs above to `datasets/loaders.py:_download()`, or simply run any evaluation script and the files will be fetched and cached to `Data/` automatically.

---

## Running the Evaluation

### CLI arguments

```bash
python run_rolling_eval.py [--dataset {etth1,exchange}] [--model {gpt-4-turbo,mistral-small}]
                           [--n_origins N] [--horizons H [H ...]]
                           [--cal_size C] [--output_dir DIR]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `etth1` | Dataset: `etth1` (hourly, seasonal) or `exchange` (daily, stationary) |
| `--model` | `gpt-4-turbo` | LLM: `gpt-4-turbo` or `mistral-small` |
| `--n_origins` | `10` | Number of rolling evaluation origins |
| `--horizons` | `1 6 24` | Forecast horizons to evaluate |
| `--cal_size` | `24` | Calibration window size; `0` = match horizon H |
| `--output_dir` | `outputs/rolling` | Directory for cache files and plots |

### Reproduce thesis results (uses cache — no API calls)

```bash
# ETTh1-OT: 10 origins, H=1/6/24  (primary results, Tables 5.1–5.6)
python run_rolling_eval.py

# Exchange Rate: 10 origins, H=1/6/24  (cross-validation, Table 5.7)
python run_rolling_eval.py --dataset exchange
```

### Re-run with fresh API calls

Delete the relevant cache file(s), then run normally:

```bash
# Example: re-run ETTh1 origin 0, H=24 test split
del outputs\rolling\cache_etth1_o0_test_h24.csv
python run_rolling_eval.py --horizons 24 --n_origins 1
```

### Quick spot-check (1 origin, 1 horizon — ~30 seconds)

```bash
python run_rolling_eval.py --n_origins 1 --horizons 1
```

**Cache naming convention:**
```
cache_{dataset}_{model_tag}_o{origin_idx}_{split}_h{horizon}.csv
# model_tag is empty for gpt-4-turbo (default), _mistral_sm for Mistral
# cal split uses hshared (same calibration samples used across all horizons)
```

## Extending the Framework

### Add a new dataset

Implement a loader in `datasets/loaders.py` following the `load_etth1()` pattern. Return a `pd.Series` with a `DatetimeIndex`. Register the dataset name in the `--dataset` argument in `run_rolling_eval.py`. The pipeline auto-detects hourly vs daily frequency and selects time-of-day or day-of-week Mondrian stratification accordingly.

### Add a new CP variant

Implement `fit_*()` and `apply_*()` functions in `conformal/` following the interface in `conformal/split_cp.py`. Register it in `run_rolling_eval.py`'s results loop.

### Add a new LLM

Add a completion function to `models/llms.py` following the `_gpt4_completion()` pattern (single API call with `n=num_samples`, same prompt format from Gruver et al. 2023). Register it in `completion_fns`, `tokenization_fns`, and `context_lengths`. Add it to `--model` choices in `run_rolling_eval.py`.

> **Note on model scale:** The framework requires the LLM to generate multi-step numeric sequences reliably. Models below ~70B parameters (e.g., Mistral-Small 7B) tend to stop generating after 3–4 values, making the calibration pipeline inapplicable without redesign.

---

## Configuration Reference

Key constants in `run_rolling_eval.py` (override via CLI args or edit directly):

| Constant | Value | Description |
|----------|-------|-------------|
| `TRAIN_WINDOW` | 168 | Context steps sent to LLM |
| `CAL_SIZE` | 24 | Calibration target size (steps) |
| `ALPHA` | 0.20 | CP significance level → 80% nominal coverage |
| `NUM_SAMPLES` | 10 | LLM samples per call (`n=10` in one API request) |
| `gpt4_hypers.alpha` | 0.3 | LLMTime scaling quantile (basic mode) |
| `gpt4_hypers.temp` | 1.0 | Sampling temperature |
| `gpt4_hypers.top_p` | 0.8 | Nucleus sampling parameter |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `openai >= 1.0.0` | GPT-4-Turbo and Mistral API access |
| `numpy`, `pandas` | Numerical computation and data handling |
| `scipy` | Temperature scaling optimisation |
| `scikit-learn` | Isotonic regression |
| `matplotlib` | Figure generation |
| `python-dotenv` | API key loading |
| `tiktoken` | Token counting for dynamic `max_tokens` budget |
| `tqdm` | Progress bars |
| `httpx` | HTTP timeout configuration for OpenAI client |
| `python-docx` | Thesis Word document generation |

---

## Citation

If you use this framework or results in your work, please cite:

```
B S Deepthi (2026). Probabilistic Time Series Forecasting with Large Language Models:
A Post-Hoc Calibration Framework. MSc Dissertation, Liverpool John Moores University.
```

---
