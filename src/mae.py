'''
Compute mean absolute error from predictions
'''

import pandas as pd
from pathlib import Path

from errors import mae

from cfg import DAILY_HORIZON

def _degradation(err_fn, y_pred, y_true):
    errors = {}
    for i in range(DAILY_HORIZON):
        err_params = (y_true[f"y_{i}"], y_pred[f"y_pred_{i}"])
        error_function = globals()[err_fn]
        error_i = error_function(*err_params)
        errors[f"{err_fn}_{str(i).zfill(2)}"] = error_i
    return errors

def _next_step_mae(pred_df):
    errors_df = []
    for (metric, application, model, metric_class), sub_df in pred_df.groupby(["metric", "application", "model", "metric_class"]):
        print(metric, application, model, metric_class)
        
        errors_df_row = {
            "ts": metric_class + "#" + str(application).zfill(2),
            "metric_class": metric_class, 
            "model": model
        }
        
        errors_df_row.update({
            "mae": mae(sub_df["y_0"], sub_df["y_pred_0"]),
        })
        errors_df.append(errors_df_row)
    
    return pd.DataFrame(errors_df)

def _horizon_mae(pred_df):
    errors_df = []
    for (metric, application, model, metric_class), sub_df in pred_df.groupby(["metric", "application", "model", "metric_class"]):
        print(metric, application, model, metric_class)
        
        errors_df_row = {
            "ts": metric_class + "#" + str(application),
            "metric_class": metric_class, 
            "model": model
        }

        y_pred = sub_df[[col for col in sub_df.columns if col.startswith("y_pred")]]
        y_true = sub_df[[col for col in sub_df.columns if col.startswith("y_") and "pred" not in col]]

        degradation_errors = _degradation(err_fn = "mae", y_pred = y_pred, y_true = y_true)
        errors_df_row.update(degradation_errors)
        
        errors_df.append(errors_df_row)
    
    return pd.DataFrame(errors_df)

def _format(df):
    # Models
    df["model"] = df["model"].str.replace("SNAIVE", "sNaive")
    df["model"] = df["model"].str.replace("SMM", "sMM")
    df["model"] = df["model"].str.replace("SARIMA", "SARIMA")
    df["model"] = df["model"].str.replace("RNN_RNN", "FC-RNN")
    df["model"] = df["model"].str.replace("RNN_LSTM", "LSTM")
    df["model"] = df["model"].str.replace("RNN_GRU", "GRU")

    return df

if __name__ == "__main__":


    '''
    Use this version for the initial setting (no online RNN training)
    '''
    # pred_df = pd.concat([pd.read_csv(pred_file) for pred_file in Path(f"../data/results/pred/daily/").glob("*.csv")])


    
    '''
    [UPDATE] Use this version for the online RNN training
    '''
    base_path = Path("../data/results/pred/daily/")
    foundations = list(base_path.glob("*Chronos.csv")) + list(base_path.glob("*TimesFM.csv"))
    pred_df_foundation = pd.concat([pd.read_csv(pred_file) for pred_file in foundations])

    statistic = list(base_path.glob("*SARIMA.csv")) + list(base_path.glob("*ETS.csv")) + list(base_path.glob("*Prophet.csv")) + list(base_path.glob("*SMM.csv")) + list(base_path.glob("*SNAIVE.csv"))
    pred_df_statistic = pd.concat([pd.read_csv(pred_file) for pred_file in statistic])

    base_path = Path("../review/data/results/pred/daily/")
    rnn = list(base_path.glob("*RNN_GRU.csv")) + list(base_path.glob("*RNN_LSTM.csv")) + list(base_path.glob("*RNN_RNN.csv"))
    pred_df_rnn = pd.concat([pd.read_csv(pred_file) for pred_file in rnn])
    
    pred_df = pd.concat([pred_df_foundation, pred_df_statistic, pred_df_rnn])




    pred_df["metric_class"] = pred_df["metric"].str.split('_').str[0]

    next_step_mae_df = _format(_next_step_mae(pred_df))
    next_step_mae_df.to_csv("../data/results/next_step_mae.csv", index=None)

    horizon_mae_df = _format(_horizon_mae(pred_df))
    horizon_mae_df.to_csv("../data/results/horizon_mae.csv", index=None)