import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
import os
import pickle

from statsmodels.regression.linear_model import RegressionResults
from sklearn.pipeline import Pipeline
from eli5.sklearn import PermutationImportance


def create_benchmark_plot(results: RegressionResults, conf_level=0.05, **kwargs):
    """
    kwargs: figsize
    """
    figsize = kwargs.get("figsize", (18, 7.2))

    p_vals = results.pvalues
    sig_mask = p_vals <= conf_level
    sig_features = p_vals[sig_mask].sort_values()
    insig_features = p_vals[~sig_mask].sort_values(ascending=False)

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    sig_features.plot(kind="bar", rot=90, ax=axes[0], title="Significant Features", ylabel="p-value")
    insig_features.plot(kind="bar", rot=90, ax=axes[1], title="Inignificant Features", ylabel="p-value")

    fig.tight_layout()

    return fig, axes


def create_feature_importance_plot(pipeline: Pipeline, **kwargs):
    """
    kwargs: figsize
    """
    figsize = kwargs.get("figsize", (18, 6))

    model = pipeline.named_steps["model"]
    feature_importances = model.feature_importances_.astype(float)
    feature_importances /= feature_importances.sum()
    df = pd.DataFrame({
        "feature_name": pipeline.named_steps["data_preprocessing"].get_feature_names_out(),
        "feature_importance": feature_importances
    })

    fig, axis = plt.subplots(figsize=figsize)
    df.sort_values("feature_importance", ascending=False).plot(kind="bar", x="feature_name", y="feature_importance", rot=90, ax=axis, ylabel="Relative importance")

    return fig, axis


def show_weights_permutation_importance(model, X: pd.DataFrame, y: pd.DataFrame, save_dir=None, **kwargs):
    """
    "model" should be fitted.
    """
    figsize = kwargs.get("figsize", (18, 6))
    perm = PermutationImportance(model).fit(X, y)
    # exp = eli5.show_weights(perm, feature_names=X.columns.tolist())
    exp_df = eli5.explain_weights_df(perm, feature_names=X.columns.tolist())

    fig, axis = plt.subplots(figsize=figsize)
    exp_df.sort_values("weight", ascending=False).plot(x="feature", y="weight", kind="bar", yerr="std", ax=axis)

    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        # with open(os.path.join(save_dir, "perm_model.pkl"), "wb") as wf:
        #     pickle.dump(perm, wf)
        exp_df.to_csv(os.path.join(save_dir, "perm_model.csv"))

    return fig, axis
