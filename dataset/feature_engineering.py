import numpy as np
import pandas as pd
import sys
import os


ROOT = os.path.abspath(__file__)
for _ in range(2):
    ROOT = os.path.dirname(ROOT)

PATH = os.path.dirname(ROOT)
if PATH not in sys.path:
    sys.path.append(PATH)

from BusinessAnalyticsProject.configs import load_global_config

GlOBAL_CONFIG = load_global_config()


def get_holiday_df():
    """
    holiday | 2010, 2011, 2012, 2013
    """
    all_df = []
    for holiday_iter in GlOBAL_CONFIG["holiday_category"]:
        df_iter = pd.DataFrame(GlOBAL_CONFIG[holiday_iter], index=[holiday_iter])
        all_df.append(df_iter)

    all_df = pd.concat(all_df)
    all_df.rename(columns={col: int(col) for col in all_df.columns}, inplace=True)

    return all_df


def __contruct_features(features_df: pd.DataFrame):
    features_df["Temperature"] = (features_df["Temperature"] - 32) * 5 / 9  # Farenheit to Celsius
    features_df["Weight"] = features_df["IsHoliday"].apply(lambda val: GlOBAL_CONFIG["holiday_weight"] if val else 1)
    markdown_cols = features_df.loc[:, features_df.columns.str.contains("MarkDown")]
    features_df["MarkDownMean"] = np.nanmean(markdown_cols.values, axis=1)
    features_df["MarkDownStd"] = np.nanstd(markdown_cols.values, axis=1)

    holiday_df = get_holiday_df()
    year = features_df["Date"].dt.isocalendar().year
    week = features_df["Date"].dt.isocalendar().week

    for holiday_iter in GlOBAL_CONFIG["holiday_category"]:
        holiday_week = holiday_df.loc[holiday_iter, year.values]  # (N_train,)
        features_df[f"Is_{holiday_iter}"] = (week.values == holiday_week.dt.isocalendar().week.values)

    return features_df


def join_tables(train_or_test_path: str, features_path: str, stores_path: str, save_path=None):
    """
    Join tables and construct all features; without filling missing values.
    """
    train_or_test = pd.read_csv(train_or_test_path, parse_dates=["Date"])
    features = pd.read_csv(features_path, parse_dates=["Date"])
    stores = pd.read_csv(stores_path)

    out_table = pd.merge(train_or_test, features.drop(columns=["IsHoliday"]), "left", ["Store", "Date"])
    out_table = pd.merge(out_table, stores, "left", ["Store"])
    out_table = out_table.sort_values((["Date", "Store", "Dept"]))

    out_table = __contruct_features(out_table)

    if save_path is not None:
        assert ".csv" in save_path
        dirname = os.path.dirname(save_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        out_table.to_csv(save_path)

    return out_table


def select_features(features_df: pd.DataFrame, if_train=True):
    """
    - Fill missing values.
        + MarkDown*: 0 (including mean and std)
        + CPI and Unemployment: mean
    - Select columns used for features and labels (if_train == True)

    Parameters
    ----------
    features_df: pd.DataFrame
        "data/processed/all_train.csv" or "data/processed/all_test.csv"

    Returns
    -------
    training: out_df, labels
    testing: out_df, weights
    """
    markdown_cols = features_df.columns[features_df.columns.str.contains("MarkDown")]
    features_df[markdown_cols] = features_df[markdown_cols].fillna(0.)
    other_missing_val_cols = ["CPI", "Unemployment"]
    features_df[other_missing_val_cols] = features_df[other_missing_val_cols].fillna(
        features_df[other_missing_val_cols].mean()
    )

    labels = None
    cols_drop = ["Store", "Date"]
    if if_train:
        labels = features_df["Weekly_Sales"]
        cols_drop.append("Weekly_Sales")
    out_df = features_df.drop(columns=cols_drop)

    if not if_train:
        weights = features_df["Weight"]
        out_df = out_df.drop(columns=["Weight"])

        return out_df, weights

    return out_df, labels


def train_test_split(all_train_df: pd.DataFrame):
    """
    Split: Use 2012 data for testing

    Parameters
    ----------
    all_train_df: pd.DataFrame
        Without feature selection. This is "data/processed/all_train.csv".

    Returns
    -------
    X_train, y_train, weights_train, X_test, y_test, weights_test
    """
    all_train_df["Date"] = pd.to_datetime(all_train_df["Date"])
    train_mask = all_train_df["Date"] <= pd.to_datetime("20120101")
    all_train_df, all_labels = select_features(all_train_df, True)
    X_train = all_train_df[train_mask]
    y_train = all_labels[train_mask]
    weights_train = X_train["Weight"]
    X_train = X_train.drop(columns=["Weight"])

    X_test = all_train_df[~train_mask]
    y_test = all_labels[~train_mask]
    weights_test = X_test["Weight"]
    X_test = X_test.drop(columns=["Weight"])

    return X_train, y_train, weights_train, X_test, y_test, weights_test
