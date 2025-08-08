# Steam Player Count Forecast

This repository trains a state-of-the-art N-HiTS time-series model (via the [Darts](https://github.com/unit8co/darts) library) on **`steamdb_chart_570.csv`**, which contains almost 14 years of daily player-count data. The model can forecast player numbers years into the future and save the resulting charts.

---

## 1. Environment setup

```bash
# Clone the repo (if you haven’t already)
# git clone <repo-url> && cd steam

# 1. Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.9 or newer is recommended.  A GPU (e.g., Apple Silicon MPS, CUDA) will accelerate training, but the dataset is small enough to run comfortably on CPU.

---

## 2. Basic usage

### 2.1 Evaluate on the last year of data and plot results
```bash
python forecast_nhits.py --plot
```
This will:
1. Clean and pre-process the CSV
2. Reserve the most recent 365 days for testing
3. Train a weekly **Seasonal Naïve** baseline and an **N-HiTS** model
4. Print MAPE / MAE / RMSE for both models and display an interactive plot

### 2.2 Forecast the next 5 years (1 825 days) and save the figure
```bash
python forecast_nhits.py \
       --horizon 1825 \        # forecast length in days
       --future \              # train on the full history; skip back-test
       --save forecast_5yr.png # file path to save the PNG chart
```
The script will write **`forecast_5yr.png`** to the project root.

### 2.3 Model persistence
On the very first run the script trains the N-HiTS model and stores two artefacts under `models/`:

* `models/nhits_model` – the trained network weights (saved with Darts)
* `models/scaler.pkl`   – the fitted data scaler (saved with joblib)

If these files already exist, the script will load them and skip retraining, so subsequent forecasts start almost instantly.

To force a fresh training simply delete the `models/` directory.

---

## 3. CLI options
| Flag | Default | Description |
|------|---------|-------------|
| `--plot`    | off | Show the forecast plot in a window. |
| `--horizon` | 365 | Forecast horizon (days). |
| `--future`  | off | Train on **all** available history and generate an out-of-sample forecast (no test split). |
| `--save`    | —   | Path to save the plot as PNG. |

Run `python forecast_nhits.py -h` for the full list.

---

## 4. Project structure
```
steam/
├── forecast_nhits.py      # main training/forecast script
├── steamdb_chart_570.csv  # raw daily player counts (14 years)
├── models/                # saved model + scaler (created on first run)
├── requirements.txt       # Python dependencies
└── README.md              # you are here
```

---

## 5. Notes
* The script automatically detects Apple Silicon “MPS” GPUs and converts all tensors to **`float32`** to avoid dtype issues.
* Feel free to tune hyper-parameters (`EPOCHS`, `INPUT_LENGTH`, etc.) in `forecast_nhits.py` for even higher accuracy.

Happy forecasting!

