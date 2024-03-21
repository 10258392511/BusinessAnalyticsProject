import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.tools as tls
import plotly.io as pio
import os
import sys

ROOT = os.path.abspath(__file__)
for _ in range(2):
    ROOT = os.path.dirname(ROOT)

PATH = os.path.dirname(ROOT)
if PATH not in sys.path:
    sys.path.append(PATH)

from scipy.stats import pearsonr
from BusinessAnalyticsProject.configs import load_global_config
from BusinessAnalyticsProject.dataset.feature_engineering import combine_holiday_cols
from tqdm import tqdm
from typing import Union


GLOBAL_CONFIG = load_global_config()


def count_duplicates(df: pd.DataFrame):
    sizes = df.groupby(["Store", "Dept", "Date"]).size()

    return sizes


def count_missing_values(df: pd.DataFrame, **kwargs):
    missing_vals_ratio = df.isna().mean()
    figsize = kwargs.get("figsize", (12, 12))
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    sns.heatmap(df.isna(), cbar=False, ax=axes[0])
    axes[0].set_xlabel("Feature name")
    axes[0].set_yticklabels([])
    axes[0].set_ylabel("Obs no.,\nsorted by (Date, Store, Dept)")
    axes[0].set_title("Missing values are indicated by 1")

    sns.barplot(missing_vals_ratio, ax=axes[1])
    axes[1].set_xticks(axes[1].get_xticks(), axes[1].get_xticklabels(), rotation=90)
    axes[1].set_xlabel("Feature name")
    axes[1].set_ylabel("Missing portion")
    axes[1].set_title("Missing value ratio")

    fig.tight_layout()

    return missing_vals_ratio, fig, axes


def __preprocess_features(features_df: pd.DataFrame, sep_date=Union[None, str]) -> pd.DataFrame:
    """
    (1). Combine holiday columns and use text labels.
    (2). Convert categorical feature columns to dtype "category".
    """
    cate_cols = ["Dept", "Type", "Holiday_type"]
    cols_drop = ["Store", "Weight"]  # "Store": similar to an index col
    features_df = combine_holiday_cols(features_df)
    holidays_decode = {val: key for key, val in GLOBAL_CONFIG["holiday_category"].items()}
    holidays_decode[0] = "not_a_holiday"
    holidays_decode = {key: " ".join(map(lambda s: s.capitalize(), val.split("_"))) for key, val in
                       holidays_decode.items()}
    features_df["Holiday_type"] = features_df["Holiday_type"].apply(lambda enc: holidays_decode[enc])

    for col in cate_cols:
        features_df[col] = features_df[col].astype("category")

    features_df = features_df.drop(columns=cols_drop)

    if sep_date is not None:
        sep_date = pd.Timestamp(sep_date)
        return features_df[features_df["Date"] <= sep_date]

    return features_df


def create_time_series_plot(features_df: pd.DataFrame, y="Weekly_Sales", x="Date", **kwargs):
    """
    Avg weekly sales vs time
    """
    hue = "Holiday_type"
    figsize = kwargs.get("figsize", (18, 6))
    fig, axis = plt.subplots(figsize=figsize)
    avg_tgt_df = (
        features_df[[x, y, hue]]
        .groupby(x)
        .apply(lambda df: pd.DataFrame({y: [df[y].mean()], hue: [df[hue].iloc[0]]}))
        .reset_index()
        .drop(columns="level_1")
    )

    axis = sns.scatterplot(avg_tgt_df[avg_tgt_df[hue] != "Not A Holiday"], x=x, y=y, hue=hue, ax=axis, s=200, marker="*")
    axis = avg_tgt_df.plot(x=x, y=y, figsize=figsize, ax=axis)
    # fig = tls.mpl_to_plotly(fig)

    return fig, axis


def create_line_fit_plot(features_df: pd.DataFrame, y: str, x: str, **kwargs):
    features_df = features_df[[x, y]].dropna()
    figsize = kwargs.get("figsize", (4.8, 4.8))
    fig, axis = plt.subplots(figsize=figsize)
    axis = sns.regplot(features_df, x=x, y=y, ax=axis, line_kws={"color": "red"})
    corr_coeff_res = pearsonr(features_df[x], features_df[y])
    r_square = corr_coeff_res.statistic ** 2
    axis.set_title(r"$R^2 = $" + f"{r_square: .3f}")

    return fig, axis


def create_violin_plot(features_df: pd.DataFrame, y: str, x: str, **kwargs):
    features_df = features_df[[x, y]].dropna()
    figsize = kwargs.get("figsize", (18, 6))
    fig, axis = plt.subplots(figsize=figsize)
    axis = sns.violinplot(features_df, x=x, y=y, ax=axis)

    return fig, axis


def create_corr_plot(features_df: pd.DataFrame, tgt_col: Union[str, None] = None, **kwargs):
    figsize = kwargs.get("figsize", (18, 18))
    if tgt_col is None:
        tgt_col = "Weekly_Sales"

    features_df = features_df.drop(columns=["Date"])
    features_df = features_df.select_dtypes(include="number")
    corr_df = features_df.corr(method="spearman")

    # fig = plt.figure(constrained_layout=True)
    # widths = [1, 2]
    # nrows, ncols = 1, 2
    # spec = fig.add_gridspec(nrows=nrows, ncols=ncols, width_ratios=widths)
    # axis = fig.add_subplot(spec[0, 0])

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    sns.heatmap(corr_df, ax=axes[0], annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    axes[0].set_title("Rank Correlation")
    (
        corr_df
        .drop(columns=tgt_col)
        .loc[tgt_col]
        .sort_values(ascending=False)
        .plot(kind="bar", ax=axes[1], ylabel="Rank Correlation with Weekly Sales")
    )

    fig.tight_layout()

    return fig, axes


def create_full_tearsheet(features_df: pd.DataFrame, save_dir: Union[str, None] = None, **kwargs):
    sep_date = kwargs.get("sep_date", "2011-12-31")
    features_df = __preprocess_features(features_df, sep_date)
    tgt_col = "Weekly_Sales"
    fig_dict = {}

    fig_dict["time_series"] = create_time_series_plot(features_df)
    plt.show()

    features_num_df = features_df.select_dtypes(include="number")
    features_cate_df = features_df.select_dtypes(include="category")
    features_cate_df[tgt_col] = features_df[tgt_col]

    pbar = tqdm(features_num_df.columns, leave=False)
    for col in pbar:
        if col == tgt_col:
            continue
        pbar.set_description(f"Column: {col}")
        fig_dict[col] = create_line_fit_plot(features_df, tgt_col, col)
        plt.show()

    pbar = tqdm(features_cate_df.columns, leave=False)
    for col in pbar:
        if col == tgt_col:
            continue
        pbar.set_description(f"Column: {col}")
        fig_dict[col] = create_violin_plot(features_df, tgt_col, col)
        plt.show()

    fig_dict["corr"] = create_corr_plot(features_df, tgt_col)
    plt.show()

    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for name, (fig_iter, _) in fig_dict.items():
            fig_iter.savefig(os.path.join(save_dir, f"{name}.png"))
