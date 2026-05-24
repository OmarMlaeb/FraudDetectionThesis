import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler


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


def preprocess_ieee_cis(df):
    df = remove_high_missing_columns(df)

    y = df["isFraud"].values

    drop_columns = ["isFraud", "TransactionID"]
    X = df.drop(columns=[col for col in drop_columns if col in df.columns])

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    for col in categorical_cols:
        X[col] = X[col].fillna("Unknown")
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))

    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].median())

    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X.values.astype(np.float32), y.astype(np.float32)


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