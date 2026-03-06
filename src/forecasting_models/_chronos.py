'''Chronos forecasting'''
import pandas as pd  # requires: pip install pandas
import torch
from chronos import BaseChronosPipeline

from dataset import windowed

import numpy as np

from tqdm import tqdm

import logging

class Chronos():

    def __init__(self, window = 28, context = 14, max_context = 512):
        # Initialize the model
        self.context = context
        self.window = window
        self.horizon = self.window - self.context
        self.max_context = max_context
        self.pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map="cpu",  # use "cpu" for CPU inference
            torch_dtype=torch.bfloat16,
        )

    def fit(self, X):
        # Fit the model from path
        raise NotImplementedError(f"fit() is not supported by {self.__class__.__name__}. The model is already pre-trained.")
        
    def predict(self, train_ts, test_ts):
        X_test, y_test = windowed(test_ts, window=self.window, context=self.context, reshape=False)

        _past_ctx = train_ts.tolist() + X_test[0].tolist()

        y_true, y_pred = [], []

        input_ctxs = []
        for i, test_window in enumerate(y_test):
            input_ctxs.append(_past_ctx[-len(train_ts.tolist()):])
            y_true.append(test_window)
            _past_ctx.append(test_window[0])

        input_ctxs = np.array(input_ctxs).squeeze()

        logging.info(f"Chronos contexts shape {input_ctxs.shape}")

        # # Version 1
        # y_pred_quantiles, y_pred_mean = self.pipeline.predict_quantiles(
        #     context=torch.tensor(input_ctxs),
        #     prediction_length=self.horizon,
        #     quantile_levels=[0.1, 0.5, 0.9],
        # )

        # Version 2
        y_pred_quantiles, y_pred_mean = self.pipeline.predict_quantiles(
            inputs=torch.tensor(input_ctxs),
            prediction_length=self.horizon,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        y_pred = y_pred_mean.numpy().squeeze()
        return y_pred

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
        return "Chronos"
