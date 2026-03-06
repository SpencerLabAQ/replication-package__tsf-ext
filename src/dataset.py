import pandas as pd
import numpy as np

import logging
# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)


import more_itertools as m_iter

from cfg import DAILY_WINDOW, DAILY_HORIZON, HOURLY_WINDOW, HOURLY_HORIZON, TEST_SIZE, VAL_SIZE

def proc_data(ts, describe = False):

    ts = np.array(ts)

    train_ts, test_ts = split(ts, TEST_SIZE)
    val_ts, test_ts = split(test_ts, VAL_SIZE)

    if describe:
        logging.info(f"Time series shape: {ts.shape}")
        logging.info(f"(train, eval, test) time series shapes: ({train_ts.shape, val_ts.shape, test_ts.shape})")

    return {
        "ts": [train_ts, val_ts, test_ts],
    }

def windowed(ts, window, context, reshape = True):
    windowed_ts = list(m_iter.windowed(ts, n=window, step=1))
    X = pd.DataFrame(windowed_ts).iloc[:, 0:context].to_numpy()
    y = pd.DataFrame(windowed_ts).iloc[:, context:window].to_numpy()

    # useful for NN input shape
    if reshape:
        X = np.expand_dims(X, axis=-1)
    
    return X, y

def split(ts, test_size = .2):
    return ts[:-int(len(ts)*test_size)], ts[-int(len(ts)*test_size):]

def extract_features(df, aggregation):
    window, horizon = DAILY_WINDOW, DAILY_HORIZON if aggregation == 'daily' else HOURLY_WINDOW, HOURLY_HORIZON
    y = df[["y{}".format(i) for i in range(horizon)]].to_numpy()
    X = df[["x{}".format(i) for i in range(window-horizon)]].to_numpy()
    return X.astype('float32'), y.astype('float32')