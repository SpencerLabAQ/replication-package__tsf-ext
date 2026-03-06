from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from itertools import product
import pandas as pd
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import time

from dataset import windowed

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class SARIMA():

    def __init__(self, seasonality, window = 28, context = 14, p = None, d = None, q = None, P = None, D = None, Q = None, max_context = 100):
        # Initialize the model
        self.window = window
        self.context = context

        self.max_context = max_context

        self.horizon = self.window - self.context

        self.seasonality = seasonality
        if self.seasonality is None:
            raise ValueError("Seasonality is required for SARIMA model")
        
        self.p, self.d, self.q, self.P, self.D, self.Q = p, d, q, P, D, Q

    def fit(self, X):
        assert self.p is not None and self.d is not None and self.q is not None and self.P is not None and self.D is not None and self.Q is not None, "Initialize or tune (p, q, d, P, D, Q) parameters before fitting"

        logging.info(f"Fitting SARIMA with parameters [{self.p=} {self.d=} {self.q=} {self.P=} {self.D=} {self.Q=}]")
        self.X = X

        assert not np.isnan(self.X).any(), "Error in SARIMA fitting. nan values in X"
        
    def predict(self, X):

        X_test, y_test = windowed(X, window=self.window, context=self.context, reshape=False)

        _past_ctx = self.X.tolist() + X_test[0].tolist()

        y_true, y_pred = [], []

        for i, test_window in tqdm(enumerate(y_test), total=len(y_test), desc="Forecasting values with SARIMA model"):

            self.model = SARIMAX(
                _past_ctx,
                order = (self.p, self.d, self.q),
                seasonal_order = (self.P, self.D, self.Q, self.seasonality),
                simple_differencing = False
            )

            # Check for LinAlg Exception
            try:
                self.model_fit = self.model.fit(disp=False, tol=1e-6)
            except np.linalg.LinAlgError as e:
                logging.warning(f"Error during SARIMA fitting: LinAlgError occurred >> {e}. Using 'powell' method")
                self.model_fit = self.model.fit(disp=False, tol=1e-6, method='powell')

            y_pred_ith = self.model_fit.get_forecast(self.horizon, dynamic = True).predicted_mean

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

    def get_name(self):
        return "SARIMA"

    def set_seeds(self, seed):
        # Set the seed for all necessary random number generators.
        raise NotImplementedError(f"set_seeds() is not supported by {self.__class__.__name__}")
        