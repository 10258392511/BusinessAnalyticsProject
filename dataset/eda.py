import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
