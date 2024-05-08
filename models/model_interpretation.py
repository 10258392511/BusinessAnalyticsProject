import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
import os
import glob
import pickle

from statsmodels.regression.linear_model import RegressionResults
from sklearn.pipeline import Pipeline
from eli5.sklearn import PermutationImportance
from eli5.permutation_importance import get_score_importances
from sklearn.metrics import r2_score
from typing import List, Iterable


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
    insig_features.plot(kind="bar", rot=90, ax=axes[1], title="Insignificant Features", ylabel="p-value")

    fig.tight_layout()

    return fig, axes

def create_benchmark_plot_with_p(pval, results: RegressionResults, conf_level=0.05, **kwargs):
    """
    kwargs: figsize
    """
    figsize = kwargs.get("figsize", (18, 7.2))

    p_vals = pd.Series(pval)
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


def show_weights_permutation_importance_non_sklearn(model, X: pd.DataFrame, y: pd.DataFrame, save_dir=None, **kwargs):
    """
    "model" should be fitted.
    """
    figsize = kwargs.get("figsize", (18, 6))
    def score(X_in, y_in):
        X_in = pd.DataFrame(X_in, index=X.index, columns=X.columns)
        y_pred = model.predict(X_in)

        return r2_score(y, y_pred)

    base_score, score_decreases = get_score_importances(score, X.values, y.values)
    exp_df = __create_permutation_importance_df(score_decreases, X.columns)

    fig, axis = plt.subplots(figsize=figsize)
    exp_df.sort_values("weight", ascending=False).plot(x="feature", y="weight", kind="bar", yerr="std", ax=axis)

    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        exp_df.to_csv(os.path.join(save_dir, "perm_model.csv"))

    return fig, axis


def __create_permutation_importance_df(
        score_decreases: List[np.ndarray],
        col_names: Iterable
):
    score_decreases_df = pd.DataFrame(score_decreases, columns=col_names)
    exp_df = pd.DataFrame(index=col_names)
    exp_df["weight"] = score_decreases_df.mean()
    exp_df["std"] = score_decreases_df.std()
    exp_df.index.name = "feature"
    exp_df.reset_index(inplace=True)

    return exp_df


def show_prediction_best_worst(
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.DataFrame,
        num_samples=3
):
    y_pred = pipeline.predict(X)
    y_all = pd.DataFrame()
    y_all["label"] = y
    y_all["pred"] = y_pred
    y_all["mse"] = (y_all["pred"] - y_all["label"]) ** 2
    y_all.sort_values("mse", ascending=True, inplace=True)

    X_in = pipeline[:-1].fit_transform(X)
    X_in = pd.DataFrame(X_in, index=X.index, columns=X.columns)
    all_exp = {}
    for i in range(num_samples):
        all_exp[i] = (eli5.explain_prediction(pipeline.named_steps["model"], X_in.loc[y_all.index[i]]),
                      y_all.iloc[i], X.loc[y_all.index[i]])
        all_exp[-i] = (eli5.explain_prediction(pipeline.named_steps["model"], X_in.loc[y_all.index[-i]]),
                       y_all.iloc[-i], X.loc[y_all.index[-i]])

    return all_exp


def show_prediction_time_series(
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.DataFrame,
        date_time=None,
        save_dir=None,
        **kwargs
):
    figsize = kwargs.get("figsize", (18, 6))
    y_pred = pipeline.predict(X)
    y_pred_df = pd.DataFrame({"pred": y_pred, "date": date_time})
    y_pred_df["label"] = np.NaN
    y_pred_df.iloc[:len(y), y_pred_df.columns.get_loc("label")] = y
    mask = ~y_pred_df.label.isna()
    r2_score_val = r2_score(y_pred_df.label[mask], y_pred_df.pred[mask])
    fig, axis = plt.subplots(figsize=figsize)
    y_pred_avg_df = (
        y_pred_df
        .groupby("date")
        .mean()
    )
    # y_pred_avg_df.plot(ax=axis, xlabel="Date", ylabel="Avg weekly sales", title=r"$R^2 = $" + f"{r2_score_val: .3f}")
    y_pred_avg_df.plot(ax=axis, xlabel="Date", ylabel="Avg weekly sales")
    axis.axvline(pd.to_datetime("20111231"), color="r", linewidth=3, linestyle="--")

    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fig.savefig(os.path.join(save_dir, "pred_time_series.png"))

    return fig, axis


def show_permutation_importance_corr(
        results_dir: str,
        **kwargs
):
    figsize = kwargs.get("figsize", (6, 6))
    filenames = glob.glob(os.path.join(results_dir, "*/perm_model.csv"))
    perm_all_df = None
    for filename in filenames:
        model_name = os.path.basename(os.path.dirname(filename))
        perm_df = pd.read_csv(filename, index_col=[0]).set_index("feature")
        if perm_all_df is None:
            perm_all_df = pd.DataFrame(index=perm_df.index)
        perm_all_df[model_name] = perm_df["weight"]

    corr_df = perm_all_df.corr(method="spearman")
    fig, axis = plt.subplots(figsize=figsize)
    sns.heatmap(corr_df, ax=axis, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    axis.set_title("Rank Correlation of Permutation Importance")

    return fig, axis


def show_metrics(
        results_dir: str
):
    filenames = glob.glob(os.path.join(results_dir, "*/*.pkl"))
    metrics_all = {}
    for filename in filenames:
        model_name = os.path.basename(os.path.dirname(filename))
        with open(filename, "rb") as rf:
            models_all = pickle.load(rf)
        metrics_all[model_name] = models_all["metrics"]

    metrics_df = pd.DataFrame(metrics_all).T  # (metric) | model1, ... -> (model) | metric1

    return metrics_df.sort_values("r2_score", ascending=False)


def show_cv_test_results(
        results_dir: str
):
    dirnames = glob.glob(os.path.join(results_dir, "*/"))
    metrics_all = {}
    for dirname in dirnames:
        model_name = os.path.basename(os.path.dirname(dirname))
        model_filename = glob.glob(os.path.join(dirname, "*.pkl"))[0]
        with open(model_filename, "rb") as rf:
            models_all = pickle.load(rf)
        cv_res_df = pd.read_csv(os.path.join(dirname, "cv_results.csv"))
        metrics_all[model_name] = {
            "cross_val": cv_res_df["mean_test_score"].max(),
            "test_set": models_all["metrics"]["r2_score"]
        }

    metrics_all = pd.DataFrame(metrics_all).T.sort_values("cross_val", ascending=False)

    return metrics_all


def submission(
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        all_test: pd.DataFrame,
        save_dir=None
):
    y_pred = pipeline.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=["Weekly_Sales"])
    y_pred_df["Id"] = all_test["Store"].astype(str) + "_" + all_test["Dept"].astype(str) + "_" + all_test["Date"].dt.strftime("%Y-%m-%d")
    y_pred_df.set_index("Id", inplace=True)

    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        y_pred_df.to_csv(os.path.join(save_dir, "submission.csv"), float_format="%.2f")

    return y_pred_df
