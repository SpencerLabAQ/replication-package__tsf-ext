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

from cfg import TEST_SIZE, VAL_SIZE, WINDOW, CONTEXT, HORIZON, SEASONALITY, EPOCHS, MODELS_PATH, PRED_PATH, ES_PATIENCE, ROP_PATIENCE, BATCH_SIZE, TS_PATH, DATA_PATH
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

    X_all_windows, y_all_windows = windowed(scaled_ts, window, context)

    return {
        "ts": [train_ts, val_ts, test_ts],
        "X": [X_train, X_val, X_test],
        "y": [y_train, y_val, y_test],
        "all_windows": [X_all_windows, y_all_windows] # both train and test (for online training)
    }

def check_alignment(X, y, Xt, yt):
    # print("lengths X, y, Xt, yt:", len(X), len(y), len(Xt), len(yt))

    # print("X first elements:",  X[0].squeeze())
    # print("y first elements:",  y[0].squeeze())
    # print("Xt first elements:", Xt[0].squeeze())
    # print("yt first elements:", yt[0].squeeze())

    assert len(y) == len(yt), "y and yt must have the same length"

    for idx in range(len(y)):
        assert np.isclose(y[idx][0], yt[idx][-1]), (
            f"Misalignment at idx={idx}: "
            f"y[-1]={y[idx][0]} != yt[0]={yt[idx][-1]}"
        )

    logging.info("X, y, Xt, and yt arrays are aligned for online training.")

if __name__ == "__main__":

    def_aggregation = 'daily'

    df_path = TS_PATH / "ts.csv"
    logging.info(f"Dataset loading from {df_path}")
    df = pd.read_csv(df_path)
    logging.info(f"Dataset loaded.")

    exp_metadata = []
    
    for (application, metric, aggregation, type, use), sub_df in df.groupby(["application", "metric", "aggregation", "type", "use"]):

        if not use or aggregation != def_aggregation:
            continue

        logging.info(f"Analysis of {application=} {metric=} {aggregation=} {type=} {use=}")

        ts = sub_df.sort_values(by=["dt#"], ascending=True).y.values

        (window, context, seasonality) = (WINDOW, CONTEXT, SEASONALITY)
        horizon = window - context

        ml_models = create_models(window, context, aggregation)

        scaler = StandardScaler()
        processed = proc_data(ts, scaler, window = window, context = context, describe = True)

        X_train, X_val, X_test = processed["X"][0], processed["X"][1], processed["X"][2]
        y_train, y_val, y_test = processed["y"][0], processed["y"][1], processed["y"][2]


        shift = context - 1
        X_all, y_all = processed["all_windows"][0], processed["all_windows"][1]
        # print(X_all, y_all)
        Xh_test, yh_test = X_all[-len(X_test)-shift:-shift], y_all[-len(y_test)-shift:-shift]
        
        check_alignment(X_test, y_test, Xh_test, yh_test)

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

            model.load(load_param=f"app_{application}#metric_{metric}")

            logging.info(f"Start fitting for {model.get_name()}")
            start_fitting_time = datetime.now()
            model.fit(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, verbose = 2, save_param=f"app_{application}#metric_{metric}")
            end_fitting_time = datetime.now()
            logging.info(f"End fitting for {model.get_name()}")
            fitting_time = end_fitting_time - start_fitting_time

            logging.info(f"Start predict for {model.get_name()}")
            start_prediction_time = datetime.now()
            y_pred = model.predict_with_online_training(X = X_test, y = y_test, Xh = Xh_test, yh = yh_test)
            end_prediction_time = datetime.now()
            prediction_time = end_prediction_time - start_prediction_time
            logging.info(f"End predict for {model.get_name()}")
            y_pred = scaler.inverse_transform(y_pred)

            X_test_rescaled = scaler.inverse_transform(X_test.squeeze())
            y_true_rescaled = scaler.inverse_transform(y_test)
            
            # dump predictions
            pred_path = PRED_PATH / aggregation
            pred_path.mkdir(exist_ok=True, parents=True)
            res_df = pd.concat([pd.DataFrame(X_test_rescaled, columns=[f"X_test_{i}" for i in range(horizon)]), pd.DataFrame(y_pred, columns=[f"y_pred_{i}" for i in range(horizon)]), pd.DataFrame(y_true_rescaled, columns=[f"y_{i}" for i in range(horizon)])], axis = 1)
            res_df["metric"] = metric
            res_df["application"] = application
            res_df["model"] = model.get_name()
            res_df.reset_index(names = "test_window", inplace=True)
            res_df.set_index(["metric", "application", "model", "test_window"]).to_csv(pred_path / f"app_{application}#metric_{metric}#model_{model.get_name()}.csv")

            mt.update({'online_fit_and_pred_time': prediction_time.total_seconds() * 1000})
            exp_metadata.append(mt)

    save_in = DATA_PATH / "results"
    save_in.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(exp_metadata).to_csv(save_in / f"rnn_exp_online_training_{def_aggregation}.csv", index=None)