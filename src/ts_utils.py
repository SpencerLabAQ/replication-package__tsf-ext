import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

from scipy.stats import boxcox

def test_boxcox_suitability(time_series):
    try:
        _, lmbda = boxcox(time_series)
        return True
    except ValueError:
        return False

def find_seasonality(data, threshold=0.5, max_lag=None):
    if len(data) < 2:
        return None  
    
    if max_lag is None:
        max_lag = len(data) // 2
    
    autocorr = acf(data, nlags=max_lag, fft=True)
    
    for lag in range(1, len(autocorr)):
        if autocorr[lag] > threshold:
            return lag
    
    return None


def find_trend_type(time_series, period):

    if period < 2:
        return "add" 
    
    stl = STL(time_series, period=period, robust=True)
    result = stl.fit()

    trend = result.trend
    residual = result.resid
    
    residual_ratio = residual / trend
    
    std_residual = np.std(residual)
    std_residual_ratio = np.std(residual_ratio)

    if std_residual_ratio < std_residual * 0.8:
        return 'mul'
    elif std_residual_ratio > std_residual * 1.2:
        return 'add'
    else:
        return 'unk'
    
def check_stationarity(time_series, significance_level=0.05):
    result = adfuller(time_series)
    is_stationary = result[1] < significance_level
    return bool(is_stationary)