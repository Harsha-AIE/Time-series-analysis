"""
FULL TIME SERIES ANALYSIS PIPELINE
Monash Electricity Demand (Hourly)

This script satisfies:

1. Data Description & Problem Formulation
2. Exploratory Visualization & Statistical Analysis
3. Linear Model Implementation & Evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"e:\sem 6\Time series analysis\Project")
DATA_DIR = PROJECT_ROOT / "train" / "monash_elecdemand_dataset_hourly"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SEASONAL_PERIOD = 24
TRAIN_SPLIT = 0.8
ADF_ALPHA = 0.05

# ============================================================
# DATA DESCRIPTION (Printed in console)
# ============================================================

def describe_dataset():
    print("\n" + "="*60)
    print("DATA DESCRIPTION & PROBLEM FORMULATION")
    print("="*60)
    print("Dataset: Monash Electricity Demand (Hourly)")
    print("Frequency: Hourly observations")
    print("Seasonality: 24-hour daily cycle")
    print("Variable: Electricity demand (GWh)")
    print("Problem: Forecast future electricity demand")
    print("Motivation: Energy planning & grid optimization")
    print("="*60)

# ============================================================
# STATIONARITY TEST
# ============================================================

def adf_test(series):
    result = adfuller(series, autolag="AIC")
    return result[0], result[1], result[1] < ADF_ALPHA

# ============================================================
# METRICS
# ============================================================

def compute_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return rmse, mae, mape

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

def train_test_split(series):
    split = int(len(series) * TRAIN_SPLIT)
    return series[:split], series[split:]

# ============================================================
# MAIN PIPELINE
# ============================================================

def process_series(file_path):

    series_id = int(file_path.stem.split("_")[-1])
    data = np.load(file_path)
    series = pd.Series(data)

    print(f"\nProcessing Series {series_id}")

    # -------------------------
    # PART 2: EXPLORATORY ANALYSIS
    # -------------------------

    # ADF Test (original)
    adf_stat, p_value, stationary = adf_test(series)
    print(f"ADF p-value (original): {p_value:.6f}")

    # ADF Test (1st difference)
    diff_series = series.diff().dropna()
    adf_stat_diff, p_value_diff, stationary_diff = adf_test(diff_series)
    print(f"ADF p-value (1st diff): {p_value_diff:.6f}")

    # Determine differencing order
    d = 0 if stationary else 1
    print(f"Selected differencing order d = {d}")

    # ACF/PACF (used for manual order justification)
    fig, axes = plt.subplots(2, 1, figsize=(10,6))
    plot_acf(series, ax=axes[0])
    plot_pacf(series, ax=axes[1])
    plt.tight_layout()
    plt.close()

    # Seasonal Decomposition
    seasonal_decompose(series, model="additive", period=SEASONAL_PERIOD)

    # -------------------------
    # PART 3: MODELING
    # -------------------------

    train, test = train_test_split(series)

    # ARIMA
    arima_model = ARIMA(train, order=(1,d,1)).fit()
    arima_forecast = arima_model.forecast(len(test))
    arima_rmse, arima_mae, arima_mape = compute_metrics(test, arima_forecast)

    # SARIMA
    sarima_model = ARIMA(train,
                         order=(1,d,1),
                         seasonal_order=(1,0,1,SEASONAL_PERIOD)).fit()
    sarima_forecast = sarima_model.forecast(len(test))
    sarima_rmse, sarima_mae, sarima_mape = compute_metrics(test, sarima_forecast)

    # Auto-ARIMA
    auto_model = auto_arima(train,
                            seasonal=True,
                            m=SEASONAL_PERIOD,
                            stepwise=True,
                            suppress_warnings=True)
    auto_forecast = auto_model.predict(len(test))
    auto_rmse, auto_mae, auto_mape = compute_metrics(test, auto_forecast)

    # Determine best model
    rmses = {
        "ARIMA": arima_rmse,
        "SARIMA": sarima_rmse,
        "Auto-ARIMA": auto_rmse
    }

    best_model = min(rmses, key=rmses.get)

    print("RMSE Comparison:")
    print(f"ARIMA: {arima_rmse:.4f}")
    print(f"SARIMA: {sarima_rmse:.4f}")
    print(f"Auto-ARIMA: {auto_rmse:.4f}")
    print(f"Best Model: {best_model}")

    return {
        "series_id": series_id,
        "arima_rmse": arima_rmse,
        "sarima_rmse": sarima_rmse,
        "auto_rmse": auto_rmse,
        "best_model": best_model
    }

# ============================================================
# EXECUTION
# ============================================================

def main():
    describe_dataset()

    files = sorted(DATA_DIR.glob("*.npy"))
    results = []

    for f in files:
        results.append(process_series(f))

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "full_analysis_summary.csv", index=False)

    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()