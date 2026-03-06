import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import argparse

from pathlib import Path

import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from sklearn.preprocessing import StandardScaler

from dataset import split, windowed
from datetime import datetime

from forecasting_models import RNN

from cfg import TEST_SIZE, VAL_SIZE, DAILY_WINDOW, DAILY_CONTEXT, DAILY_HORIZON, DAILY_SEASONALITY, HOURLY_WINDOW, HOURLY_CONTEXT, HOURLY_HORIZON, HOURLY_SEASONALITY, EPOCHS, MODELS_PATH, PRED_PATH, ES_PATIENCE, ROP_PATIENCE, BATCH_SIZE, TS_PATH, DATA_PATH
from cst_utils import set_seeds

set_seeds()

def create_models(window, context, aggregation):
    rnn_model = RNN(model = "RNN", window = window, context = context, aggregation = aggregation)
    lstm_model = RNN(model = "LSTM", window = window, context = context, aggregation = aggregation)
    gru_model = RNN(model = "GRU", window = window, context = context, aggregation = aggregation)

    return rnn_model, lstm_model, gru_model

def proc_data(ts, scaler, window, context, describe = False):

    ts = np.array(ts)
    scaled_ts = scaler.fit_transform(ts.reshape(-1, 1))
    scaled_ts = scaled_ts.reshape(scaled_ts.shape[0])

    train_ts, test_ts = split(scaled_ts, TEST_SIZE)
    val_ts, test_ts = split(test_ts, VAL_SIZE)

    X_train, y_train = windowed(train_ts, window, context)
    X_val, y_val = windowed(val_ts, window, context)
    X_test, y_test = windowed(test_ts, window, context)

    if describe:
        logging.info(f"Time series shape: {ts.shape}")
        logging.info(f"(train, eval, test) time series shapes: ({train_ts.shape, val_ts.shape, test_ts.shape})")
        logging.info(f"Train splits shape: [X={X_train.shape}, y={y_train.shape}]")
        logging.info(f"Val splits shape: [X={X_val.shape}, y={y_val.shape}]")
        logging.info(f"Test splits shape: [X={X_test.shape}, y={y_test.shape}]")


    return {
        "ts": [train_ts, val_ts, test_ts],
        "X": [X_train, X_val, X_test],
        "y": [y_train, y_val, y_test]
    }

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

        ml_models = create_models(window, context, aggregation)

        scaler = StandardScaler()
        processed = proc_data(ts, scaler, window = window, context = context, describe = True)

        X_train, X_val, X_test = processed["X"][0], processed["X"][1], processed["X"][2]
        y_train, y_val, y_test = processed["y"][0], processed["y"][1], processed["y"][2]

        # experiment using models
        for model in ml_models:

            mt = {
                'application': application,
                'metric': metric,
                'aggregation': aggregation,
                'type': type,
                'model': model.get_name()
            }

            logging.info(f"Experiment with {model.get_name()}")

            # model.load()

            logging.info(f"Start fitting for {model.get_name()}")
            start_fitting_time = datetime.now()
            model.fit(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, verbose = 2, save_param=f"app_{application}#metric_{metric}")
            end_fitting_time = datetime.now()
            logging.info(f"End fitting for {model.get_name()}")
            fitting_time = end_fitting_time - start_fitting_time

            logging.info(f"Start predict for {model.get_name()}")
            start_prediction_time = datetime.now()
            y_pred = model.predict(X_test)
            end_prediction_time = datetime.now()
            prediction_time = end_prediction_time - start_prediction_time
            logging.info(f"End predict for {model.get_name()}")
            y_pred = scaler.inverse_transform(y_pred)

            X_test_rescaled = scaler.inverse_transform(X_test.squeeze())
            y_true_rescaled = scaler.inverse_transform(y_test)
            
            # dump predictions
            pred_path = PRED_PATH / args.aggregation
            pred_path.mkdir(exist_ok=True, parents=True)
            res_df = pd.concat([pd.DataFrame(X_test_rescaled, columns=[f"X_test_{i}" for i in range(horizon)]), pd.DataFrame(y_pred, columns=[f"y_pred_{i}" for i in range(horizon)]), pd.DataFrame(y_true_rescaled, columns=[f"y_{i}" for i in range(horizon)])], axis = 1)
            res_df["metric"] = metric
            res_df["application"] = application
            res_df["model"] = model.get_name()
            res_df.reset_index(names = "test_window", inplace=True)
            res_df.set_index(["metric", "application", "model", "test_window"]).to_csv(pred_path / f"app_{application}#metric_{metric}#model_{model.get_name()}.csv")

            mt.update({'fit_time': fitting_time.total_seconds() * 1000, 'pred_time': prediction_time.total_seconds() * 1000})
            exp_metadata.append(mt)

    save_in = DATA_PATH / "results"
    save_in.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(exp_metadata).to_csv(save_in / f"rnn_exp_{args.aggregation}.csv", index=None)