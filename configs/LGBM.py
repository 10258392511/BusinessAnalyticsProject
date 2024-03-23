import numpy as np

from lightgbm import LGBMRegressor

CONFIG = {
    "model_cls": LGBMRegressor,
    "params": {
        "categorical_feature": ""
    },
    "param_space": {
        "num_leaves": [31, 63, 127, 255],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_child_samples": [20, 50, 100],
        "max_depth": [-1, 10, 20, 40],
        "subsample": [0.8, 0.9, 1.],
        "colsample_bytree": [0.8, 0.9, 1.],
        "reg_alpha": [0, 0.01, 0.1, 1.],
        "reg_lambda": [0, 0.01, 0.1, 1]
    }
}
