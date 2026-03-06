'''Exponential smoothing (ETS)'''
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

from ts_utils import find_trend_type, test_boxcox_suitability

from dataset import windowed
import numpy as np

import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ETS():

    def __init__(self, seasonality = None, window = 28, context = 14):
        # Initialize the model
        self.window = window
        self.context = context
        self.horizon = self.window - self.context

        self.seasonality = seasonality
        assert seasonality is not None, "Seasonality required for ETS model"

    def tune(self, X, y):
        best_params, best_model, best_mse = self.__tune_exponential_smoothing(X, y, seasonal_periods=self.seasonality)
        return best_params

    def get_name(self):
        return self.__class__.__name__

    def fit(self, X):
        self.X = X
        
        self.trend_type = find_trend_type(self.X, self.seasonality)
        self.boxcox = test_boxcox_suitability(self.X)
        
    def predict(self, X):
        X_test, y_test = windowed(X, window=self.window, context=self.context, reshape=False)

        _past_ctx = self.X.tolist() + X_test[0].tolist()

        y_true, y_pred = [], []

        for i, test_window in enumerate(y_test):

            try:
                self.model = ExponentialSmoothing(
                    _past_ctx, 
                    seasonal="add", 
                    seasonal_periods=self.seasonality,
                    trend=self.trend_type, 
                    use_boxcox=self.boxcox,
                    initialization_method='estimated'
                )
                self.model_fit = self.model.fit()
            except Exception as e:
                logging.warning(f"{e}. Using additive model")
                self.model = ExponentialSmoothing(
                    _past_ctx, 
                    seasonal="add", 
                    seasonal_periods=self.seasonality,
                    trend='additive', 
                    use_boxcox=self.boxcox,
                    initialization_method='estimated'
                )
                self.model_fit = self.model.fit()
            y_pred_ith = self.model_fit.forecast(steps = self.horizon)

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