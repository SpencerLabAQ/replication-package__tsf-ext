from pathlib import Path

# generic
RANDOM_STATE = 42

# dataset
TEST_SIZE, VAL_SIZE = .2, .5

# experiment setting
DAILY_WINDOW, DAILY_CONTEXT = 28, 14
DAILY_HORIZON = DAILY_WINDOW - DAILY_CONTEXT
DAILY_SEASONALITY = 7 # weekly

HOURLY_WINDOW, HOURLY_CONTEXT = 48, 24
HOURLY_HORIZON = HOURLY_WINDOW - HOURLY_CONTEXT
HOURLY_SEASONALITY = 24

# training and testing for RNN
EPOCHS = 500
BATCH_SIZE = 4
ES_PATIENCE = 20
ROP_PATIENCE = 20
ES_START_FROM_EPOCH = 20

# path
DATA_PATH = Path("path-to-your-experiment-data")

MODELS_PATH = DATA_PATH / "models" 
PRED_PATH = DATA_PATH / "results" / "pred"

TS_PATH = Path("path-to-your-timseries-data")

MODELS_ORDER = ["sNaive", "sMM", "SARIMA", "ETS", "Prophet", "FC-RNN", "LSTM", "GRU", "TimesFM", "Chronos"]