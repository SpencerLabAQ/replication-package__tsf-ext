import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
np.show_config()

import pandas as pd

from pathlib import Path

import os
import logging
import time

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from dataset import split

from cfg import TEST_SIZE, VAL_SIZE, TS_PATH
from cst_utils import set_seeds

from tqdm import tqdm
from itertools import product

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from multiprocessing import Pool, Manager

set_seeds()

def parallel_fn(fn, shared_list, df_records, nproc = 1, desc="Parallel execution"):
    
    iterations = [(item, shared_list) for item in df_records]

    with Pool(nproc) as p:
        with tqdm(total=len(iterations), desc=desc) as pbar:
            for _ in p.imap_unordered(fn, iterations):
                pbar.update()

def compute_aic(params):
    (ts, p, d, q, P, D, Q, s), shared_list = params
    method = "linalg"
    model = SARIMAX(
            ts,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            simple_differencing=False
        )
    try:
        model_fit = model.fit(disp=False)
    except np.linalg.LinAlgError as e:
        logging.warning(f"Error during SARIMA fitting: LinAlgError occurred >> {e}. Using 'powell' method")
        model_fit = model.fit(disp=False, tol=1e-6, method='powell')
        method = "powell"
    except:
        logging.warning(f"Error during SARIMA fitting: setting AIC to inf")
        shared_list.append([p, d, q, P, D, Q, s, np.inf, method])
        return

    aic = model_fit.aic
    shared_list.append([p, d, q, P, D, Q, s, aic, method])
    return


def tune_sarima(tuning_ts, seasonality):
    logging.info(f"___ Tuning SARIMA model ___")
    logging.info(f"Tuning time series length: {len(tuning_ts)}")

    ### seasonality and stationarity (s, d, D)
    (adf_statistic, adf_p_value, adf_used_lags, adf_n_obs, adf_critical_values, adf_ic_best) = adfuller(tuning_ts)
    logging.info(f"AdFuller statistical test for stationarity")
    logging.info(f"ADF stat: {adf_statistic}, p-value: {adf_p_value}")
    logging.info(f"tune() {os.getpid()}")


    P_THRESH = 0.05
    diff_steps = 0

    if adf_p_value < P_THRESH:
        logging.info("Time series is stationary: d and D values set to 0.")
    else:
        ts_diff = tuning_ts
        while adf_p_value >= 0.05:
            logging.info(f"Time series not stationary after differencing {diff_steps} steps")
            ts_diff = np.diff(ts_diff, n=1)
            (adf_statistic, adf_p_value, adf_used_lags, adf_n_obs, adf_critical_values, adf_ic_best) = adfuller(ts_diff)
            logging.info(f"ADF stat: {adf_statistic}, p-value: {adf_p_value}")
            diff_steps += 1
        logging.info(f"Time series stationary after applying {diff_steps} differencing steps (d and D set to {diff_steps})")
    
    d = diff_steps
    D = diff_steps
    s = seasonality

    ### parameters optimization (p, q, P, Q)
    ps = qs = Ps = Qs = range(0, 3, 1)
    order_lst = list(product(ps, qs, Ps, Qs))
    logging.info(f"{order_lst=}")

    results = []

    # not parallelized
    params = [(tuning_ts, order[0], d, order[1], order[2], D, order[3], s) for order in order_lst]
    
    # parallel exec
    shared_list = Manager().list()

    parallel_fn(
        fn=compute_aic,
        shared_list=shared_list, 
        df_records=params, 
        nproc=os.cpu_count() - 5,
        desc="SARIMA parameters parallel optimization"
        )
    
    results = list(shared_list)
    
    results_df = pd.DataFrame(results, columns = ['p', 'd', 'q', 'P', 'D', 'Q', 's', 'AIC', "method"])
    return results_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
def proc_data(ts, describe = False):
    ts = np.array(ts)

    train_ts, test_ts = split(ts, TEST_SIZE)
    val_ts, test_ts = split(test_ts, VAL_SIZE)

    if describe:
        logging.info(f"Time series shape: {ts.shape}")
        logging.info(f"(train, eval, test) time series shapes: ({train_ts.shape, val_ts.shape, test_ts.shape})")

    return {
        "ts": [train_ts, val_ts, test_ts],
    }

if __name__ == "__main__":

    df_path = TS_PATH / "ts.csv"
    logging.info(f"Dataset loading from {df_path}")
    df = pd.read_csv(df_path)
    logging.info(f"Dataset loaded.")

    sarima_results_path = Path("..") / "data" / "tuning" / "SARIMA"
    sarima_results_path.mkdir(exist_ok=True, parents=True)

    for (app_id, metric_id, aggregation, type, use), sub_df in df.groupby(["application", "metric", "aggregation", "type", "use"]):

        if app_id <= 2:
            continue

        logging.info(f"Analysis of {app_id=} {metric_id=}")
        if not use:
            logging.info(f"Skipping the time series for missing values")
            continue

        if aggregation == "daily":
            seasonality = 7
        else:
            # hourly
            seasonality = 24

        ts = sub_df.sort_values(by=["dt#"], ascending=True).y.values
        processed = proc_data(ts, describe=True)
        
        ts_train, ts_eval, ts_test = processed["ts"][0], processed["ts"][1], processed["ts"][2]
        logging.info(f"main {os.getpid()}")
        sarima_params = tune_sarima(ts_train, seasonality=seasonality)
        sarima_params.to_csv(sarima_results_path / f"app_{app_id}#metric_{metric_id}.csv")