import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.regression.linear_model import RegressionResults
from sklearn.pipeline import Pipeline


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
    feature_importances = model.feature_importances_
    feature_importances /= feature_importances.sum()
    df = pd.DataFrame({
        "feature_name": pipeline.named_steps["data_preprocessing"].get_feature_names_out(),
        "feature_importance": feature_importances
    })

    fig, axis = plt.subplots(figsize=figsize)
    df.plot(kind="bar", x="feature_name", y="feature_importance", rot=90, ax=axis, ylabel="Relative importance")

    return fig, axis
