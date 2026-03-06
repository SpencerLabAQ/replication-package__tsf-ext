import os
import random as rn
import numpy as np

def set_seeds(seed = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    rn.seed(seed)
    np.random.seed(seed)
