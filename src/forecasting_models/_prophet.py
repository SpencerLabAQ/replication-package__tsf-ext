
import sys
import os
import logging

import pandas as pd

from dataset import windowed

import numpy as np

import matplotlib.pyplot as plt

class ProphetWrapper():

    def __init__(self, window = 28, context = 14, aggregation = "D", seasonality = 7):
        # Initialize the model
        self.window = window
        self.context = context
        self.horizon = self.window - self.context
        self.seasonality = seasonality

        self.aggregation = aggregation

    def tune(self, X, y):
        raise NotImplementedError(f"tune() is not supported by {self.__class__.__name__}")
        
    
    def fit(self, X):

        self.X = X

    def get_name(self):
        return "Prophet"
        
    def predict(self, X):
        from prophet import Prophet

        X_test, y_test = windowed(X, window=self.window, context=self.context, reshape=False)

        _past_ctx = self.X.tolist() + X_test[0].tolist()

        y_true, y_pred = [], []

        for i, test_window in enumerate(y_test):

            self.model = Prophet()

            time_index = pd.date_range(start='2023-01-01', periods=len(_past_ctx), freq=self.aggregation)

            fit_df = pd.DataFrame({'ds': time_index, 'y': _past_ctx})

            self.model.fit(fit_df)
            
            # predict step
            future = self.model.make_future_dataframe(periods=self.horizon, freq=self.aggregation)
            forecast = self.model.predict(future)
            y_pred_ith = forecast['yhat'].iloc[-self.horizon:].to_numpy()

            y_true.append(test_window)
            y_pred.append(y_pred_ith)

            _past_ctx.append(test_window[0])

        return np.array(y_pred)
        
    def predict_proba(self, X):
        # Predict class probabilities for given data.
        raise NotImplementedError(f"predict_proba() is not supported by {self.__class__.__name__}")

    def dump(self, filename):
        # Save the model to a path.
        raise NotImplementedError(f"dump() is not supported by {self.__class__.__name__}")

    def load(self, filename):
        # Load the model from path
        raise NotImplementedError(f"load() is not supported by {self.__class__.__name__}")