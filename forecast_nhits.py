#!/usr/bin/env python3
"""Train an NHITS model (via Darts) to forecast the daily player count.

Usage:
    python forecast_nhits.py [--plot]

Prerequisites (install once inside an activated virtual environment):
    pip install "u8darts[torch]>=0.28.0" matplotlib pandas tqdm

This script performs:
    1. Data loading & cleaning
    2. Train/validation split (last 365 days held out)
    3. Baseline seasonal naïve model for comparison
    4. NHITS training & forecasting
    5. Accuracy reporting (MAPE, MAE, RMSE)
    6. Optional plotting of forecasts
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import torch

# Ensure Torch uses float32 (needed for Apple Silicon GPU/MPS)
torch.set_default_dtype(torch.float32)

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mape, rmse
from darts.models import NaiveSeasonal, NHiTSModel

# --------------------------- Configuration ---------------------------------- #
DATA_PATH = Path(__file__).with_name("steamdb_chart_570.csv")
# Forecast horizon (days)
HORIZON = 365  # last year reserved for evaluation and forecasting
# Input history length fed to NHITS (2 years recommended for yearly seasonality)
INPUT_LENGTH = 365 * 2
# Output chunk length (= forecast horizon for a single shot)
OUTPUT_LENGTH = 30  # model predicts in 30-day chunks internally
# Number of training epochs (increase if you have GPU / time)
EPOCHS = 300
RANDOM_STATE = 42

# ---------------------------------------------------------------------------- #

def _read_and_clean(path: Path) -> pd.Series:
    """Read the CSV and return a cleaned pandas Series indexed by Daily datetime.

    The raw file contains a leading "|" and wrapping quotes; rows may be
    blank or have missing player counts.
    """
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # strip leading pipe if present
            if line.startswith("|"):
                line = line[1:]
            # remove trailing comma (due to empty last field)
            if line.endswith(","):
                line = line[:-1]
            # remove outer quotes if they wrap the whole line
            if line.startswith("\"") and line.endswith("\""):
                line = line[1:-1]
            parts = [p.strip().strip("\"") for p in line.split(",")]
            rows.append(parts)

    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)
    # Remove potential BOM characters from column names
    df.columns = [col.lstrip("\ufeff").strip("\"") for col in df.columns]
    # Coerce dtypes
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df["Players"] = pd.to_numeric(df["Players"], errors="coerce")

    # Drop rows with invalid datetime
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime")

    # Build continuous daily index
    series = df.set_index("DateTime")["Players"]
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_idx)

    # Impute missing values via linear interpolation (forward/backfill as fallback)
    series = series.interpolate(method="linear")
    series = series.ffill().bfill()

    # Ensure float32 dtype for compatibility with MPS backend
    series = series.astype(np.float32)

    return series


def _train_test_split(ts: TimeSeries, horizon: int = HORIZON) -> Tuple[TimeSeries, TimeSeries]:
    """Split the TimeSeries into train/test sets."""
    return ts[:-horizon], ts[-horizon:]


def main(plot: bool = False, horizon: int = 365, save_path: str | None = None, future: bool = False) -> None:
    print("Loading and cleaning data …")
    series_pd = _read_and_clean(DATA_PATH)

    # Create Darts TimeSeries
    ts = TimeSeries.from_series(series_pd)

    # Decide training/test split depending on "future" flag
    if future:
        train_ts = ts
        test_ts = None
    else:
        train_ts, test_ts = _train_test_split(ts, horizon)

    # Scale data (improves neural network convergence)
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_ts).astype(np.float32)

    # ---------------------- Baseline model ---------------------------------- #
    print("Training Seasonal Naïve baseline …")
    naive_model = NaiveSeasonal(K=7)  # weekly seasonality
    naive_model.fit(train_ts)
    naive_forecast = naive_model.predict(horizon)

    # ----------------------- NHITS model ------------------------------------ #
    print("Training NHITS model … (this can take a few minutes)")
    nhits = NHiTSModel(
        input_chunk_length=INPUT_LENGTH,
        output_chunk_length=OUTPUT_LENGTH,
        n_epochs=EPOCHS,
        random_state=RANDOM_STATE,
        batch_size=32,
        num_stacks=3,
        num_blocks=3,
        # Use default architecture parameters; tweak via grid search if needed
    )
    nhits.fit(train_scaled, verbose=True)
    # Forecast
    nhits_forecast_scaled = nhits.predict(horizon)
    nhits_forecast = scaler.inverse_transform(nhits_forecast_scaled)

    # ----------------------- Evaluation / Output --------------------------- #
    if not future and test_ts is not None:
        print("Evaluating …")
        for name, pred in {
            "Seasonal Naïve": naive_forecast,
            "NHITS": nhits_forecast,
        }.items():
            print(
                f"{name} -> MAPE: {mape(test_ts, pred):.2f} %, "
                f"MAE: {mae(test_ts, pred):.2f}, RMSE: {rmse(test_ts, pred):.2f}"
            )
    else:
        print(f"Generated {horizon}-day forecast beyond available data.")

    if plot or save_path:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 6))
        ts.plot(label="Actual")
        naive_forecast.plot(label="Naïve forecast", lw=1)
        nhits_forecast.plot(label="NHITS forecast", lw=2)
        if not future and test_ts is not None:
            plt.axvline(test_ts.start_time(), color="grey", linestyle="--", alpha=0.6)
        plt.legend()
        plt.title("Player count forecast (NHITS)")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        if plot:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast player counts with NHITS.")
    parser.add_argument("--plot", action="store_true", help="Show forecast plots interactively.")
    parser.add_argument("--horizon", type=int, default=365, help="Forecast horizon in days (default: 365)")
    parser.add_argument("--future", action="store_true", help="Forecast beyond available data (train on full history and skip evaluation)")
    parser.add_argument("--save", metavar="PNG", type=str, help="File path to save the forecast plot as PNG")
    args = parser.parse_args()

    main(plot=args.plot, horizon=args.horizon, save_path=args.save, future=args.future)

