from catboost import CatBoostRegressor


CONFIG = {
    "model_cls": CatBoostRegressor,
    "params": {
        "cat_features": None,
        "train_dir": None
    },
    "param_space": {
        "iterations": [50, 100, 200, 500, 1000],
        "depth": [None, 4, 6, 8, 10],
        "subsample": [.6, .7, .8, .9, 1.],
        "colsample_bylevel": [.6, .7, .8, .9, 1.],
        "min_child_samples": [10, 20, 50],
        "learning_rate": [.01, .05, .1, .5, 1],
        "reg_lambda": [.01, .05, .1, .5, 1, 5],
        "border_count": [15, 31, 63, 127, 255],
        "random_strength": [1, 10, 20]
    }
}
