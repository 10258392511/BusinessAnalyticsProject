import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor

CONFIG = {
    "model_cls": HistGradientBoostingRegressor,
    "params": {
        "categorical_features": None
    },
    "param_space": {
        "max_iter": [100, 200, 300, 500],
        "learning_rate": [.01, .05, .1, .2],
        "max_depth": [None, 3, 5, 10, 15],
        "min_samples_leaf": [5, 10, 20, 30],
        "l2_regularization": [0, .01, .1, 1]
    }
}
