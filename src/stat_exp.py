import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from pathlib import Path

import logging
import argparse

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from dataset import windowed, proc_data
from datetime import datetime

from forecasting_models import SARIMA, ProphetWrapper, ETS

from cfg import DAILY_WINDOW, DAILY_CONTEXT, DAILY_SEASONALITY, HOURLY_WINDOW, HOURLY_CONTEXT, HOURLY_SEASONALITY, PRED_PATH, TS_PATH, DATA_PATH
from cst_utils import set_seeds

set_seeds()

def get_sarima_params(app_id, metric_id):
    sarima_results_path = DATA_PATH / "tuning" / "SARIMA"
    sarima_params = pd.read_csv(sarima_results_path / f"app_{app_id}#metric_{metric_id}.csv")
    sarima_params.sort_values(by=["AIC"], ascending=True, inplace=True)
    best_params = sarima_params.iloc[0]
    return {
        "p" : best_params["p"],
        "d" : best_params["d"],
        "q" : best_params["q"],
        "P" : best_params["P"],
        "D" : best_params["D"],
        "Q" : best_params["Q"],
        "s" : best_params["s"]
    }

def create_statistical_models(application, metric, aggregation, window, context, seasonality):

    # statistical models
    prophet_model = ProphetWrapper(window=window, context=context, aggregation='d' if aggregation == "daily" else 'h')
    
    ets_model = ETS(seasonality = seasonality, window = window, context = context)

    best_sarima_params = get_sarima_params(application, metric)
    sarima_model = SARIMA(
        seasonality = seasonality, 
        window = window, 
        context = context, 
        p = best_sarima_params['p'], 
        d = best_sarima_params["d"], 
        q = best_sarima_params["q"], 
        P = best_sarima_params["P"], 
        D = best_sarima_params["D"], 
        Q = best_sarima_params["Q"]
    )

    return prophet_model, ets_model, sarima_model

if __name__ == "__main__":

    # --- args ---
    parser = argparse.ArgumentParser(description="Time series programmatic analysis")
    parser.add_argument('--aggregation', default='daily', type=str, required=False)
    args = parser.parse_args()

    assert args.aggregation == 'daily', f"Aggregation not supported {args.aggregation}"

    df_path = TS_PATH / "ts.csv"
    logging.info(f"Dataset loading from {df_path}")
    df = pd.read_csv(df_path)
    logging.info(f"Dataset loaded.")

    exp_metadata = []
    
    for (application, metric, aggregation, type, use), sub_df in df.groupby(["application", "metric", "aggregation", "type", "use"]):

        if not use or aggregation != args.aggregation:
            continue

        logging.info(f"Analysis of {application=} {metric=} {aggregation=} {type=} {use=}")

        ts = sub_df.sort_values(by=["dt#"], ascending=True).y.values
        
        (window, context, seasonality) = (DAILY_WINDOW, DAILY_CONTEXT, DAILY_SEASONALITY)
        horizon = window - context

        models = create_statistical_models(application = application, metric = metric, aggregation = aggregation, window = window, context = context, seasonality = seasonality)

        processed = proc_data(ts, describe=True)

        ts_train, ts_eval, ts_test = processed["ts"][0], processed["ts"][1], processed["ts"][2]

        ts_context = np.concatenate((ts_train, ts_eval))
        logging.info(f"{ts_context.shape=}")

        X_test, y_test = windowed(ts_test, window=window, context=context, reshape=False)

        # experiment using models
        for model in models:

            mt = {
                'application': application,
                'metric': metric,
                'aggregation': aggregation,
                'type': type,
                'model': model.get_name()
            }

            logging.info(f"Experiment with {model.get_name()}")

            logging.info(f"Start fitting for {model.get_name()}")
            model.fit(ts_context)
            logging.info(f"End fitting for {model.get_name()}")

            logging.info(f"Start predict for {model.get_name()}")
            start_prediction_time = datetime.now()
            y_pred = model.predict(ts_test)
            end_prediction_time = datetime.now()
            prediction_time = end_prediction_time - start_prediction_time
            logging.info(f"End predict for {model.get_name()}")
            
            # dump predictions
            pred_path = PRED_PATH / args.aggregation
            pred_path.mkdir(exist_ok=True, parents=True)
            res_df = pd.concat([pd.DataFrame(X_test, columns=[f"X_test_{i}" for i in range(context)]), pd.DataFrame(y_pred, columns=[f"y_pred_{i}" for i in range(horizon)]), pd.DataFrame(y_test, columns=[f"y_{i}" for i in range(horizon)])], axis = 1)
            res_df["metric"] = metric
            res_df["application"] = application
            res_df["model"] = model.get_name()
            res_df.reset_index(names = "test_window", inplace=True)
            res_df.set_index(["metric", "application", "model", "test_window"]).to_csv(pred_path / f"app_{application}#metric_{metric}#model_{model.get_name()}.csv")

            mt.update({'fit_time': None, 'pred_time': prediction_time.total_seconds() * 1000})
            exp_metadata.append(mt)
    
    save_in = DATA_PATH / "results"
    save_in.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(exp_metadata).to_csv(save_in / f"stat_exp_{args.aggregation}.csv", index=None)