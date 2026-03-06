import numpy as np
from sklearn.metrics import mean_absolute_error

def mae(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE)
    """
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(y_true, y_pred):
    """
    Compute Symmetric Mean Absolute Percentage Error (SMAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def mean_smape(y_true, y_pred):
    """
    Compute the Mean of Symmetric Mean Absolute Percentage Error (SMAPE)
    for multiple windows.
    """
    smape_values = [smape(yt, yp) for yt, yp in zip(y_true, y_pred)]
    return np.mean(smape_values)

def median_smape(y_true, y_pred):
    """
    Compute the Median of Symmetric Mean Absolute Percentage Error (SMAPE)
    for multiple windows.
    """
    smape_values = [smape(yt, yp) for yt, yp in zip(y_true, y_pred)]
    return np.median(smape_values)

def mase(forecast: np.ndarray, insample: np.ndarray, outsample: np.ndarray, frequency: int) -> np.ndarray:
    """
    MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

    :param forecast: Forecast values. Shape: batch, time_o   -> y_pred (batch_size, horizon)
    :param insample: Insample values. Shape: batch, time_i   -> X_test (batch_size, history_len)
    :param outsample: Target values. Shape: batch, time_o    -> y_true (batch_size, horizon)
    :param frequency: Frequency value
    :return: Same shape array with error calculated for each time step
    """
    return np.mean(np.abs(forecast - outsample)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:]))