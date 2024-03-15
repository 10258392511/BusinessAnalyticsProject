import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import pickle

from scipy.stats.mstats import winsorize
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from skopt import BayesSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error
from typing import List, Union


np.int = int

ROOT = os.path.abspath(__file__)
for _ in range(2):
    ROOT = os.path.dirname(ROOT)

PATH = os.path.dirname(ROOT)
if PATH not in sys.path:
    sys.path.append(PATH)


from BusinessAnalyticsProject.configs import add_prefix_to_feature_names
from BusinessAnalyticsProject.configs import load_global_config

GlOBAL_CONFIG = load_global_config()


def identity(x):
    return x


def winsorization(df: pd.DataFrame, quantile_cut=0.02, if_plot=False):
    df_out = df.apply(lambda x: winsorize(x, limits=(quantile_cut, quantile_cut)).data, axis=0)

    if if_plot:
        fig, axis = plt.subplots()
        (df_out != df).mean().plot(kind="bar", ax=axis)

        return df_out, fig, axis

    return df_out


def create_pipeline(model_cls, param_dict, if_winsorization=False, if_data_normalization=False,
                    if_to_one_hot=False) -> Pipeline:
    """
    Parameters
    ----------
    model_cls: a model class (not an instance) or None
    param_dict: dict or None
        Parameters to initialize a model_cls
    categorical_cols: list of str
        Categorical columns
    if_winsorization: bool
        Whether to do winsorization
    if_data_normalization: bool
        Whether to do mean-std normalization
    if_to_one_hot: bool
        Whether to do one-hot coding for categorical features

    Returns
    -------
    Pipeline
    """
    cate_cols_selector = make_column_selector(dtype_include="category")
    cont_cols_selector = make_column_selector(dtype_exclude="category")

    cate_cols_steps = []
    if if_to_one_hot:
        cate_cols_steps.append(("one_hot", OneHotEncoder()))
    else:
        cate_cols_selector.append(("dummy", FunctionTransformer(identity, feature_names_out="one-to-one")))
    pipeline_cate = Pipeline(cate_cols_steps)

    cont_cols_steps = []
    if if_winsorization:
        cont_cols_steps.append(("winsorization", FunctionTransformer(winsorization, feature_names_out="one-to-one")))
    if if_data_normalization:
        cont_cols_steps.append(("standard_norm", StandardScaler()))
    if len(cont_cols_steps) == 0:
        cont_cols_steps.append(("dummy", FunctionTransformer(identity, feature_names_out="one-to-one")))
    pipeline_cont = Pipeline(cont_cols_steps)

    col_transformer = ColumnTransformer(
        [
            ("categorical", pipeline_cate, cate_cols_selector),
            ("continuous", pipeline_cont, cont_cols_selector)
        ]
    )

    if model_cls is None and param_dict is None:
        return col_transformer

    model = model_cls(**param_dict)
    pipeline_out = Pipeline([
        ("data_preprocessing", col_transformer),
        ("model", model)
    ])

    return pipeline_out


def hyperparam_tuning(
        pipeline: Pipeline,
        param_space: dict,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        weights_train: pd.Series,
        X_test: Union[pd.DataFrame, None] = None,
        y_test: Union[pd.Series, None] = None,
        weights_test: Union[pd.Series, None] = None,
        save_dir=None):
    model_step_name = list(pipeline.named_steps.keys())[-1]
    param_space_with_prefix = add_prefix_to_feature_names(model_step_name, param_space)
    cv_params = GlOBAL_CONFIG["cv_params"]
    opt = BayesSearchCV(pipeline, param_space_with_prefix,
                        fit_params={f"{model_step_name}__sample_weight": weights_train}, **cv_params)
    opt.fit(X_train, y_train)

    metrics_out = None
    if X_test is not None and y_test is not None and weights_test is not None:
        metrics_out = metrics(opt, X_test, y_test, weights_test)

    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs((save_dir))
        df_cv_results = pd.DataFrame(opt.cv_results_)
        df_cv_results.to_csv(os.path.join(save_dir, "cv_results.csv"))
        with open(os.path.join(save_dir, "cv_model_and_metrics.pkl"), "wb") as wf:
            pickle.dump({
                "model": opt,
                "metrics": metrics_out
            }, wf)

    return opt, metrics_out


def metrics(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, weights: pd.Series):
    y = y.values
    y_pred = pipeline.predict(X)
    out_dict = {
        "r2_score": r2_score(y, y_pred),
        "rmse": root_mean_squared_error(y, y_pred),
    }
    out_dict["rwmse"] = (weights @ (y - y_pred) ** 2).mean() ** 0.5

    return out_dict
