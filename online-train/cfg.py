from pathlib import Path

# generic
RANDOM_STATE = 42

# dataset
TEST_SIZE, VAL_SIZE = .2, .5

# experiment setting
WINDOW, CONTEXT = 28, 14
HORIZON = WINDOW - CONTEXT
SEASONALITY = 7 # weekly

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