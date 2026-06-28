"""
datasets/loaders.py — Dataset loaders for benchmark time series.

Supported datasets
------------------
  ETTh1         Electricity Transformer Temperature (hourly), 2016-2018
  ExchangeRate  8-currency daily exchange rates vs USD, 1990-2016
  Electricity   UCI Electricity Load Diagrams (hourly, client MT_320)
  M4_Hourly     M4 competition hourly series (any series ID H1-H414)

Downloaded files are cached in the Data/ directory at the workspace root.
"""

import os
import gzip
import urllib.request
import pandas as pd
import numpy as np

# Workspace root = parent of this package directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "Data")


def _download(url, dest):
    """Download url to dest if the file is not already present."""
    if os.path.exists(dest):
        return
    print(f"  Downloading {os.path.basename(dest)} ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


# ── ETTh1 ─────────────────────────────────────────────────────────────────────

_ETTH1_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
    "main/ETT-small/ETTh1.csv"
)


def load_etth1(target_col="OT", data_dir=None):
    """
    Load ETTh1 dataset (17,420 hourly rows, Jul 2016 to Jun 2018).
    Target column defaults to 'OT' (oil temperature).
    Columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT

    Returns pd.Series with DatetimeIndex.
    """
    if data_dir is None:
        data_dir = _DATA
    path = os.path.join(data_dir, "ETTh1.csv")
    _download(_ETTH1_URL, path)

    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found. Available: {list(df.columns)}")
    series = df[target_col].astype(float)
    series.index.name = "timestamp"
    return series


# ── Exchange Rate ─────────────────────────────────────────────────────────────

# Original source: Lai et al. (2018) LSTNet paper.
# 7,588 daily rows, 8 currency exchange rates vs USD (no header, gzipped txt).
_EXCHANGE_URL = (
    "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/"
    "master/exchange_rate/exchange_rate.txt.gz"
)

# Currency columns in the Lai et al. (2018) ordering
_EXCHANGE_COLS = ["AUD", "GBP", "CAD", "CHF", "CNY", "JPY", "NZD", "SGD"]


def load_exchange_rate(target_col="SGD", data_dir=None):
    """
    Load the 8-currency daily exchange rate dataset (7,588 rows, ~1990–2010).
    Source: Lai et al. (2018) LSTNet benchmark (laiguokun/multivariate-time-series-data).
    No date column in the raw file — a synthetic daily DatetimeIndex is created.

    Currencies (vs USD): AUD, GBP, CAD, CHF, CNY, JPY, NZD, SGD.
    Default target: SGD (Singapore Dollar) — relatively stationary, non-seasonal.

    Returns pd.Series with DatetimeIndex (daily frequency).
    """
    if data_dir is None:
        data_dir = _DATA
    gz_path  = os.path.join(data_dir, "exchange_rate.txt.gz")
    txt_path = os.path.join(data_dir, "exchange_rate.txt")

    _download(_EXCHANGE_URL, gz_path)

    if not os.path.exists(txt_path):
        print(f"  Decompressing exchange_rate.txt.gz ...")
        with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
            f_out.write(f_in.read())

    arr = np.loadtxt(txt_path, delimiter=",")   # shape [7588, 8]
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    n_rows, n_cols = arr.shape
    cols = _EXCHANGE_COLS[:n_cols]

    start = pd.Timestamp("1990-01-01")
    idx   = pd.date_range(start, periods=n_rows, freq="D")
    df    = pd.DataFrame(arr, index=idx, columns=cols)
    df.index.name = "timestamp"

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found. Available: {list(df.columns)}")

    return df[target_col].astype(float)


# ── Electricity (UCI ECL) ─────────────────────────────────────────────────────

_ECL_URL = (
    "https://raw.githubusercontent.com/thuml/Time-Series-Library/"
    "main/dataset/electricity/electricity.csv"
)


def load_electricity(client_col="MT_320", data_dir=None):
    """
    Load UCI Electricity dataset — single client column (hourly, 2012-2014).
    Preprocessed file from Time-Series-Library: columns date + MT_001..MT_321.
    Defaults to MT_320.

    Returns pd.Series with DatetimeIndex.
    """
    if data_dir is None:
        data_dir = _DATA
    path = os.path.join(data_dir, "electricity.csv")
    _download(_ECL_URL, path)

    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    if client_col not in df.columns:
        avail = [c for c in df.columns if c.startswith("MT_")]
        raise ValueError(f"Column '{client_col}' not found. Sample available: {avail[:5]}")
    series = df[client_col].astype(float)
    series.index.name = "timestamp"
    return series


# ── M4 Hourly ─────────────────────────────────────────────────────────────────

_M4_TRAIN_URL = (
    "https://raw.githubusercontent.com/M4Competition/M4-methods/"
    "master/Dataset/Train/Hourly-train.csv"
)
_M4_TEST_URL = (
    "https://raw.githubusercontent.com/M4Competition/M4-methods/"
    "master/Dataset/Test/Hourly-test.csv"
)


def load_m4_hourly(series_id="H1", data_dir=None, return_test=False):
    """
    Load a single M4 hourly series (H1 to H414, lengths 700-960).
    A synthetic hourly DatetimeIndex starting 2000-01-01 is created.

    Args:
        series_id  : str   e.g. 'H1', 'H100'
        return_test: bool  if True return (train_series, test_series);
                           otherwise concatenate into a single series.

    Returns pd.Series (or tuple) with DatetimeIndex.
    """
    if data_dir is None:
        data_dir = _DATA
    train_path = os.path.join(data_dir, "M4_Hourly_train.csv")
    test_path  = os.path.join(data_dir, "M4_Hourly_test.csv")
    _download(_M4_TRAIN_URL, train_path)
    _download(_M4_TEST_URL,  test_path)

    train_df = pd.read_csv(train_path, index_col=0)
    test_df  = pd.read_csv(test_path,  index_col=0)

    if series_id not in train_df.index:
        raise ValueError(f"Series '{series_id}' not found. IDs run from H1 to H414.")

    train_vals = train_df.loc[series_id].dropna().astype(float).values
    test_vals  = test_df.loc[series_id].dropna().astype(float).values

    start      = pd.Timestamp("2000-01-01")
    train_idx  = pd.date_range(start, periods=len(train_vals), freq="h")
    test_idx   = pd.date_range(
        train_idx[-1] + pd.Timedelta(hours=1), periods=len(test_vals), freq="h"
    )

    train_s = pd.Series(train_vals, index=train_idx, name=series_id)
    test_s  = pd.Series(test_vals,  index=test_idx,  name=series_id)

    return (train_s, test_s) if return_test else pd.concat([train_s, test_s])


# ── Utility ───────────────────────────────────────────────────────────────────

def dataset_summary(series):
    """Print basic statistics for a loaded series."""
    freq = pd.infer_freq(series.index[:50]) if len(series) >= 50 else "unknown"
    print(f"  Length   : {len(series):,} steps")
    print(f"  Range    : {series.index[0]}  to  {series.index[-1]}")
    print(f"  Freq     : {freq or 'irregular'}")
    print(f"  Min/Max  : {series.min():.3f} / {series.max():.3f}")
    print(f"  Mean±Std : {series.mean():.3f} ± {series.std():.3f}")
    print(f"  NaN count: {series.isna().sum()}")
