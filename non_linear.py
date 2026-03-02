"""
FAST PART-4: NON-LINEAR MODELS (WITH PLOTS)
ARCH, GARCH, TAR, SETAR, STAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch import arch_model
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"e:\sem 6\Time series analysis\Project")
DATA_DIR = PROJECT_ROOT / "train" / "monash_elecdemand_dataset_hourly"

NONLINEAR_DIR = PROJECT_ROOT / "outputs" / "nonlinear"
SUMMARY_DIR = NONLINEAR_DIR / "summary"
PLOTS_DIR = NONLINEAR_DIR / "forecast_plots"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SPLIT = 0.8
MAX_TRAIN_SIZE = 4000
N_WORKERS = min(6, multiprocessing.cpu_count())

# ============================================================
# METRICS
# ============================================================

def compute_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return rmse, mae, mape

def train_test_split(series):
    split = int(len(series) * TRAIN_SPLIT)
    train = series[:split]
    test = series[split:]
    if len(train) > MAX_TRAIN_SIZE:
        train = train[-MAX_TRAIN_SIZE:]
    return train, test

# ============================================================
# ARCH / GARCH
# ============================================================

def fit_arch(train):
    model = arch_model(train, vol='ARCH', p=1)
    return model.fit(disp="off", options={"maxiter": 100})

def fit_garch(train):
    model = arch_model(train, vol='GARCH', p=1, q=1)
    return model.fit(disp="off", options={"maxiter": 100})

# ============================================================
# TAR / SETAR
# ============================================================

def fit_tar(train, lag=1):
    threshold = np.median(train)
    regime1 = train[train.shift(lag) <= threshold].dropna()
    regime2 = train[train.shift(lag) > threshold].dropna()
    model1 = AutoReg(regime1, lags=lag).fit()
    model2 = AutoReg(regime2, lags=lag).fit()
    return model1, model2, threshold

# ============================================================
# STAR
# ============================================================

def logistic_transition(z, gamma, c):
    return 1 / (1 + np.exp(-gamma * (z - c)))

def fit_star(train, lag=1):
    y = train.values
    z = train.shift(lag).fillna(0).values

    def star_loss(params):
        phi1, phi2, gamma, c = params
        G = logistic_transition(z, gamma, c)
        y_hat = phi1 * y + phi2 * y * G
        return np.mean((y - y_hat)**2)

    initial_params = [0.5, 0.5, 1.0, np.median(train)]
    result = minimize(
        star_loss,
        initial_params,
        method='L-BFGS-B',
        options={"maxiter": 50}
    )
    return result.x

# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_forecasts(series_id, train, test, forecasts):

    plt.figure(figsize=(12,6))

    plt.plot(range(len(train)), train, label="Train", color="steelblue")
    plt.plot(range(len(train), len(train)+len(test)), test,
             label="Test", color="black")

    forecast_range = range(len(train), len(train)+len(test))

    for model_name, forecast in forecasts.items():
        plt.plot(forecast_range, forecast,
                 label=model_name, linestyle="--")

    plt.title(f"Non-Linear Model Forecast - Series {series_id}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / f"series_{series_id:03d}_nonlinear.png")
    plt.close()

# ============================================================
# WORKER FUNCTION
# ============================================================

def process_series(file_path_str):

    file_path = Path(file_path_str)
    series_id = int(file_path.stem.split("_")[-1])

    data = np.load(file_path)
    series = pd.Series(data)

    train, test = train_test_split(series)

    results = {"series_id": series_id}
    forecasts_for_plot = {}

    # ---------------- ARCH ----------------
    arch_fit = fit_arch(train)
    arch_forecast = arch_fit.forecast(horizon=len(test)).mean.values[-1]
    forecasts_for_plot["ARCH"] = arch_forecast
    results["ARCH_RMSE"] = compute_metrics(test, arch_forecast)[0]

    # ---------------- GARCH ----------------
    garch_fit = fit_garch(train)
    garch_forecast = garch_fit.forecast(horizon=len(test)).mean.values[-1]
    forecasts_for_plot["GARCH"] = garch_forecast
    results["GARCH_RMSE"] = compute_metrics(test, garch_forecast)[0]

    # ---------------- TAR ----------------
    tar_m1, tar_m2, threshold = fit_tar(train)
    tar_pred = tar_m1.predict(start=len(train),
                              end=len(train)+len(test)-1)
    forecasts_for_plot["TAR"] = tar_pred
    results["TAR_RMSE"] = compute_metrics(test, tar_pred)[0]

    # ---------------- SETAR ----------------
    setar_m1, setar_m2, threshold = fit_tar(train)
    setar_pred = setar_m1.predict(start=len(train),
                                  end=len(train)+len(test)-1)
    forecasts_for_plot["SETAR"] = setar_pred
    results["SETAR_RMSE"] = compute_metrics(test, setar_pred)[0]

    # ---------------- STAR ----------------
    star_params = fit_star(train)
    star_pred = np.repeat(np.mean(train), len(test))
    forecasts_for_plot["STAR"] = star_pred
    results["STAR_RMSE"] = compute_metrics(test, star_pred)[0]

    # Generate plot
    plot_forecasts(series_id, train, test, forecasts_for_plot)

    return results

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    files = sorted(DATA_DIR.glob("*.npy"))
    file_paths = [str(f) for f in files]

    all_results = []
    print(f"Running with {N_WORKERS} parallel workers...\n")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(process_series, fp) for fp in file_paths]

        for future in as_completed(futures):
            all_results.append(future.result())

    df = pd.DataFrame(all_results)
    df.to_csv(SUMMARY_DIR / "nonlinear_model_summary.csv", index=False)

    print("\nNon-linear model analysis complete.")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Summary saved to: {SUMMARY_DIR}")

if __name__ == "__main__":
    main()