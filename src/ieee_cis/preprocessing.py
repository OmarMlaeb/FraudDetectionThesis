import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


def load_ieee_cis(transaction_path, identity_path):
    transaction_df = pd.read_csv(transaction_path)
    identity_df = pd.read_csv(identity_path)

    df = transaction_df.merge(identity_df, on="TransactionID", how="left")

    return df


def remove_high_missing_columns(df, threshold=0.80):
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index
    df = df.drop(columns=cols_to_drop)

    return df


def sort_ieee_cis_temporally(df):
    if "TransactionDT" in df.columns:
        sort_columns = ["TransactionDT"]
        if "TransactionID" in df.columns:
            sort_columns.append("TransactionID")
        return df.sort_values(sort_columns, kind="mergesort")

    return df


def split_dataframe_temporally(df, train_ratio=0.80, val_ratio=0.10):
    df = sort_ieee_cis_temporally(df)

    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def fit_ieee_cis_preprocessor(train_df, missing_threshold=0.80):
    missing_ratio = train_df.isnull().mean()
    columns_to_drop = set(missing_ratio[missing_ratio > missing_threshold].index)
    columns_to_drop.update(["isFraud", "TransactionID"])

    feature_columns = [
        column for column in train_df.columns if column not in columns_to_drop
    ]
    X_train = train_df[feature_columns].copy()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]

    category_maps = {}
    for column in categorical_columns:
        values = X_train[column].fillna("Unknown").astype(str)
        category_map = {
            value: index for index, value in enumerate(pd.unique(values))
        }
        if "Unknown" not in category_map:
            category_map["Unknown"] = len(category_map)
        category_maps[column] = category_map

    medians = X_train[numerical_columns].median() if numerical_columns else pd.Series()
    X_train_numeric = X_train[numerical_columns].fillna(medians)

    scaler = StandardScaler()
    if numerical_columns:
        scaler.fit(X_train_numeric)
    else:
        scaler = None

    return {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "category_maps": category_maps,
        "medians": medians,
        "scaler": scaler,
    }


def transform_ieee_cis_features(df, preprocessor):
    feature_columns = preprocessor["feature_columns"]
    categorical_columns = preprocessor["categorical_columns"]
    numerical_columns = preprocessor["numerical_columns"]

    X = df[feature_columns].copy()

    for column in categorical_columns:
        category_map = preprocessor["category_maps"][column]
        unknown_code = category_map["Unknown"]
        X[column] = (
            X[column]
            .fillna("Unknown")
            .astype(str)
            .map(category_map)
            .fillna(unknown_code)
        )

    if numerical_columns:
        X[numerical_columns] = X[numerical_columns].fillna(preprocessor["medians"])
        X[numerical_columns] = preprocessor["scaler"].transform(X[numerical_columns])

    y = df["isFraud"].values.astype(np.float32)

    return X.values.astype(np.float32), y


def preprocess_ieee_cis_train_val_test(df, train_ratio=0.80, val_ratio=0.10):
    train_df, val_df, test_df = split_dataframe_temporally(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    preprocessor = fit_ieee_cis_preprocessor(train_df)

    X_train, y_train = transform_ieee_cis_features(train_df, preprocessor)
    X_val, y_val = transform_ieee_cis_features(val_df, preprocessor)
    X_test, y_test = transform_ieee_cis_features(test_df, preprocessor)

    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_ieee_cis_with_train_fit(df, train_ratio=0.80, val_ratio=0.10):
    df = sort_ieee_cis_temporally(df)
    train_df, _, _ = split_dataframe_temporally(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    preprocessor = fit_ieee_cis_preprocessor(train_df)

    return transform_ieee_cis_features(df, preprocessor)


def preprocess_ieee_cis(df):
    return preprocess_ieee_cis_with_train_fit(df)


def temporal_train_val_test_split(X, y, train_ratio=0.80, val_ratio=0.10):
    n = len(X)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test
