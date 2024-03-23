import numpy as np

from sklearn.ensemble import RandomForestRegressor


CONFIG = {
    "model_cls": RandomForestRegressor,
    "params": {
        "n_estimators": 75,
        "max_depth": None,
        "max_features": None,
        "min_samples_leaf": 10,
        "min_samples_split": 10,
        "verbose": 1,
        "n_jobs": 5
    },
    "param_space": {
        "n_estimators": [2, 5, 10, 50, 75],
        "max_depth": [None, 3, 5, 7],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "max_features": [None, "sqrt", "log2", 1.],
    }
}
