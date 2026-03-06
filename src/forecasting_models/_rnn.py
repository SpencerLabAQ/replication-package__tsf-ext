import numpy as np
import tensorflow as tf
import pandas as pd

from pathlib import Path

from cfg import EPOCHS, ES_PATIENCE, ROP_PATIENCE, BATCH_SIZE, ES_START_FROM_EPOCH

class RNN():

    metrics = [
        tf.keras.metrics.MeanAbsoluteError(),
        tf.keras.metrics.MeanAbsolutePercentageError(),
        tf.keras.metrics.MeanSquaredError(),
        tf.keras.metrics.RootMeanSquaredError(),
        tf.keras.metrics.R2Score(
            class_aggregation="uniform_average", num_regressors=0, name="r2_score", dtype=None
        )
    ]

    def __init__(self, window, context, aggregation = 'daily', model = "LSTM"):
        self.loss = 'mae'
        # self.optimizer = 'adam'
        self.model_type = model

        self.window = window
        self.context = context
        self.horizon = self.window - self.context

        self.aggregation = aggregation

        self.model_save_path = Path("..") / "data" / "models" / "RNN" / self.aggregation
        self.model_save_path.mkdir(exist_ok=True, parents=True)

        self.model_training_history_path = Path("..") / "data" / "training" / "RNN" / self.aggregation
        self.model_training_history_path.mkdir(exist_ok=True, parents=True)

        self.model = tf.keras.models.Sequential()
        if model == "LSTM":
            self.model.add(tf.keras.layers.LSTM(self.context, input_shape=(self.context, 1), return_sequences=True))
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.context, return_sequences=False)))
        elif model == "GRU":
            self.model.add(tf.keras.layers.GRU(self.context, input_shape=(self.context, 1), return_sequences=True))
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.context, return_sequences=False)))
        elif model == "RNN":
            self.model.add(tf.keras.layers.SimpleRNN(self.context, input_shape=(self.context, 1), return_sequences=True))
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(self.context, return_sequences=False)))

        # self.model.add(tf.keras.layers.Dense(WINDOW, activation = 'linear'))
        self.model.add(tf.keras.layers.Dense(self.horizon, activation = 'tanh'))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-03, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss=self.loss, optimizer=optimizer, metrics=self.__class__.metrics)

    def __fit_validate(self, X_train, y_train, X_val, y_val):
        assert not np.isnan(X_train).any(), "nan in X_train"
        assert not np.isnan(y_train).any(), "nan in y_train"
        assert not np.isnan(X_val).any(), "nan in X_val"
        assert not np.isnan(y_val).any(), "nan in y_val"

    def fit(self, X_train, X_val, y_train, y_val, epochs = EPOCHS, batch_size = BATCH_SIZE, es_patience = ES_PATIENCE, es_start_from_epoch = ES_START_FROM_EPOCH, rop_patience = ROP_PATIENCE, verbose = 0, save_param = ""):
        log_path = Path("..") / "logs" / "training" / self.aggregation
        log_path.mkdir(exist_ok=True, parents=True)
        
        es_cb = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", min_delta = 0.001, patience = es_patience, start_from_epoch = es_start_from_epoch)
        cv_cb = tf.keras.callbacks.CSVLogger(log_path / f"{save_param}#{self.get_name()}_training.csv", separator = ',', append = False)
        rl_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode='min', factor=0.5, patience=rop_patience, min_lr=1e-4, verbose=verbose)
        ck_cb = tf.keras.callbacks.ModelCheckpoint(self.model_save_path / f"{save_param}#{self.model_type}.keras", monitor="val_loss", mode="min", save_best_only = True, verbose=verbose)

        self.__fit_validate(X_train, y_train, X_val, y_val)

        history = self.model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = epochs, batch_size = batch_size, verbose = verbose, callbacks = [es_cb, cv_cb, rl_cb, ck_cb])
        
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(self.model_training_history_path / f"{save_param}#{self.model_type}.csv")
        
        self.model = tf.keras.models.load_model(self.model_save_path / f"{save_param}#{self.model_type}.keras")

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def dump(self, path, parameter = ""):
        path = Path(path)

        path.mkdir(exist_ok=True, parents=True)

        model_path = path / f"{parameter}#{self.get_name()}.keras"

        self.model.save(model_path)

    def get_name(self):
        return f"{self.__class__.__name__}_{self.model_type}"