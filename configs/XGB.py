from xgboost import XGBRegressor


CONFIG = {
    "model_cls": XGBRegressor,
    "params": {
        "objective": "reg:squarederror"
    },
    "param_space": {
        "n_estimators": [50, 100, 250, 500],
        "learning_rate": [.01, .05, .1, .2, .3],
        "max_depth": [3, 6, 9, 12, None],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [.6, .7, .8, .9, 1.],
        "colsample_bytree": [.6, .7, .8, .9, 1.],
        "gamma": [0., .1, .2, .3, .4, .5],
        "reg_alpha": [0, .01, .05, .1, .5, 1, 5],
        "reg_lambda": [0, .01, .05, .1, .5, 1, 5]
    }
}
