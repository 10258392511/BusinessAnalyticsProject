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


def combine_holiday_cols(df_in: pd.DataFrame):
    """
    Combine IsHoliday and Is_{holiday_name} into one categorical column.
    """
    df = df_in.copy()
    holiday_enc_dict = GlOBAL_CONFIG["holiday_category"]
    df["Holiday_type"] = 0
    for colname, code in holiday_enc_dict.items():
        code = int(code)
        if code == 0:
            continue
        colname = f"Is_{colname}"
        mask = df[colname]
        df.loc[mask, "Holiday_type"] = code

    df.drop(columns=df.columns[df.columns.str.contains("Is")], inplace=True)

    return df


def encode_type_col(df_in: pd.DataFrame):
    df = df_in.copy()
    type_enc_dict = GlOBAL_CONFIG["type_category"]
    df["Type"] = df["Type"].apply(lambda val: type_enc_dict[val])

    return df


def train_test_split(all_train_df: pd.DataFrame, split_date="20120101", if_train=True):
    """
    - Split: Use 2012 data for testing
    - Features are selected here.
    - Combine holiday columns into one categorical column.

    Parameters
    ----------
    all_train_df: pd.DataFrame
        Without feature selection. This is "data/processed/all_train.csv".

    Returns
    -------
    X_train, y_train, weights_train, X_test, y_test, weights_test
    """
    cate_cols = ["Dept", "Type", "Holiday_type"]
    all_train_df["Date"] = pd.to_datetime(all_train_df["Date"])
    train_mask = all_train_df["Date"] <= pd.to_datetime(split_date)
    all_train_df, all_labels = select_features(all_train_df, if_train)  # all_labels: labels or weights
    X_train = all_train_df[train_mask]
    y_train = all_labels[train_mask]
    weights_train = None
    if if_train:
        weights_train = X_train["Weight"]
        X_train = X_train.drop(columns=["Weight"])
    X_train = combine_holiday_cols(X_train)
    X_train = encode_type_col(X_train)
    X_train[cate_cols] = X_train[cate_cols].astype("category")

    X_test = all_train_df[~train_mask]
    if X_test.shape[0] == 0:
        return X_train, y_train, weights_train, None, None, None

    y_test = all_labels[~train_mask]
    weights_test = X_test["Weight"]
    X_test = X_test.drop(columns=["Weight"])
    X_test = combine_holiday_cols(X_test)
    X_test = encode_type_col(X_test)
    X_test[cate_cols] = X_test[cate_cols].astype("category")

    return X_train, y_train, weights_train, X_test, y_test, weights_test
