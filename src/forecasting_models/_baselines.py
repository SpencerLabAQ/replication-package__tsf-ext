import numpy as np

from dataset import windowed

import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class SNAIVE:
    ''' Seasonal Naive Baseline '''

    def __init__(self, window, context):
        self.window = window
        self.context = context
        self.horizon = self.window - self.context

    def get_name(self):
        return self.__class__.__name__

    def fit(self, X = None):
        pass

    def predict(self, test):
        X_test, _ = windowed(test, window=self.window, context=self.context, reshape=False)
        return X_test

class SMM:
    ''' Seasonal Monthly Mean Baseline '''

    def __init__(self, window, context, seasonality, context_perdiods = 4):
        self.context_perdiods = context_perdiods
        
        self.window = window
        self.context = context
        self.horizon = self.window - self.context
        self.seasonality = seasonality

    def fit(self, X):
        self.X = np.array(X)

    def get_name(self):
        return self.__class__.__name__

    def __forecasting_helper(self, X, horizon):
        assert len(X) > self.seasonality * self.context_perdiods, "Too few context datapoints for this baseline"
        
        forecasting = []

        for i in range(horizon):
            forecasting.append(np.mean([X[(-self.seasonality)*(j+1)+(i%self.seasonality)] for j in range(self.context_perdiods)]))
        
        return forecasting

    def predict(self, test):
        horizon = self.window - self.context
        X_test, y_test = windowed(test, window=self.window, context=self.context, reshape=False)

        _past_ctx = self.X.tolist() + X_test[0].tolist()

        y_true, y_pred = [], []

        for i, test_window in enumerate(y_test):

            y_pred_ith = self.__forecasting_helper(_past_ctx, horizon)

            y_true.append(test_window)
            y_pred.append(y_pred_ith)

            _past_ctx.append(test_window[0])

        return np.array(y_pred)