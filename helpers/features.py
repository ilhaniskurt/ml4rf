import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from helpers.spliters import create_frequency_based_split


def __preprocess_frequency(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    X_train_processed["freq_log"] = np.log10(X_train_processed["freq"])
    X_test_processed["freq_log"] = np.log10(X_test_processed["freq"])

    freq_scaler = MinMaxScaler(feature_range=(-1, 1))
    freq_scaler.fit(X_train_processed[["freq_log"]])

    X_train_processed["freq_log_norm"] = freq_scaler.transform(
        X_train_processed[["freq_log"]]
    )
    X_test_processed["freq_log_norm"] = freq_scaler.transform(
        X_test_processed[["freq_log"]]
    )

    for data in [X_train_processed, X_test_processed]:
        data["freq_band_1"] = (data["freq"] < 1e9).astype(
            int
        )  # < 1 GHz (UHF and below)
        data["freq_band_2"] = ((data["freq"] >= 1e9) & (data["freq"] < 6e9)).astype(
            int
        )  # 1-6 GHz (L/S bands)
        data["freq_band_3"] = ((data["freq"] >= 6e9) & (data["freq"] < 20e9)).astype(
            int
        )  # 6-20 GHz (C/X/Ku bands)
        data["freq_band_4"] = ((data["freq"] >= 20e9) & (data["freq"] < 40e9)).astype(
            int
        )  # 20-40 GHz (K/Ka bands)
        data["freq_band_5"] = (data["freq"] >= 40e9).astype(
            int
        )  # > 40 GHz (Q/V/W bands)

    # 4. Add polynomial frequency terms for non-linear relationships
    for data in [X_train_processed, X_test_processed]:
        data["freq_squared"] = data["freq_log_norm"] ** 2
        data["freq_inv"] = 1.0 / (
            data["freq_log_norm"] + 1e-10
        )  # Add small epsilon to avoid division by zero

    for data in [X_train_processed, X_test_processed]:
        for device_length in ["DEV_L_0_9um", "DEV_L_2_5um", "DEV_L_5_0um"]:
            if device_length in data.columns:
                data[f"freq_{device_length}_interaction"] = (
                    data["freq_log_norm"] * data[device_length]
                )

    band_boundaries = []

    # Determine band boundaries from training data only
    band_boundaries = [
        (X_train_processed["freq"].min(), 1e9),  # Band 1
        (1e9, 6e9),  # Band 2
        (6e9, 20e9),  # Band 3
        (20e9, 40e9),  # Band 4
        (40e9, X_train_processed["freq"].max()),  # Band 5
    ]

    # Process each dataset using the same boundaries
    for data in [X_train_processed, X_test_processed]:
        for i in range(1, 6):
            band_col = f"freq_band_{i}"
            col_name = f"freq_pos_in_band_{i}"

            # Initialize the column with NaNs or zeros
            data[col_name] = np.nan

            mask = data[band_col] == 1
            if mask.sum() > 0:
                lower_bound, upper_bound = band_boundaries[i - 1]
                data.loc[mask, col_name] = (data.loc[mask, "freq"] - lower_bound) / (
                    upper_bound - lower_bound
                )

                if data is X_test_processed:
                    data.loc[mask, col_name] = data.loc[mask, col_name].clip(0, 1)
    return X_train_processed, X_test_processed, freq_scaler


def __encode_dev_geom_l(df: pd.DataFrame):
    df = df.copy()
    lengths = [0.9, 2.5, 5.0]
    for length in lengths:
        col_name = f"DEV_L_{str(length).replace('.', '_')}um"
        df[col_name] = (df["DEV_GEOM_L"] == length).astype(int)
    return df.drop("DEV_GEOM_L", axis=1)


def __encode_num_of_trans_rf(df: pd.DataFrame):
    df = df.copy()
    values = [1, 2, 4]
    for val in values:
        df[f"TRANS_{val}"] = (df["NUM_OF_TRANS_RF"] == val).astype(int)
    return df.drop("NUM_OF_TRANS_RF", axis=1)


def __add_vb_vc_features(df: pd.DataFrame):
    df = df.copy()
    df["vb_is_zero"] = (df["vb"] == 0).astype(int)
    df["vb_is_high"] = ((df["vb"] >= 0.7) & (df["vb"] <= 0.9)).astype(int)
    df["vc_is_zero"] = (df["vc"] == 0).astype(int)
    df["vc_is_1_2V"] = ((df["vc"] >= 1.1) & (df["vc"] <= 1.3)).astype(int)
    df["vc_is_1_5V"] = ((df["vc"] >= 1.4) & (df["vc"] <= 1.6)).astype(int)
    return df


def process_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    feature_columns = ["freq", "vb", "vc", "DEV_GEOM_L", "NUM_OF_TRANS_RF"]
    label_columns = [
        "S_deemb(1,1)_real",
        "S_deemb(1,1)_imag",
        "S_deemb(1,2)_real",
        "S_deemb(1,2)_imag",
        "S_deemb(2,1)_real",
        "S_deemb(2,1)_imag",
        "S_deemb(2,2)_real",
        "S_deemb(2,2)_imag",
    ]

    df_clean = df.dropna(subset=feature_columns + label_columns)

    X = df_clean[feature_columns].copy()
    Y = df_clean[label_columns].copy()
    train_mask, test_mask = create_frequency_based_split(
        df_clean, test_size=test_size, random_state=random_state, mute=True
    )

    X_raw_train = X[train_mask].copy()
    X_raw_test = X[test_mask].copy()
    Y_raw_train = Y[train_mask].copy()
    Y_raw_test = Y[test_mask].copy()

    X_train = __add_vb_vc_features(X_raw_train)
    X_test = __add_vb_vc_features(X_raw_test)

    voltage_scaler = MinMaxScaler(feature_range=(-1, 1))
    voltage_scaler.fit(X_train[["vb", "vc"]])

    X_train[["vb", "vc"]] = voltage_scaler.transform(X_train[["vb", "vc"]])
    X_test[["vb", "vc"]] = voltage_scaler.transform(X_test[["vb", "vc"]])

    X_train = __encode_dev_geom_l(X_train)
    X_test = __encode_dev_geom_l(X_test)

    X_train = __encode_num_of_trans_rf(X_train)
    X_test = __encode_num_of_trans_rf(X_test)

    X_train, X_test, freq_scaler = __preprocess_frequency(X_train, X_test)

    # Fill NaN values with 0 for freq_pos_in_band columns
    for i in range(1, 6):
        X_train[f"freq_pos_in_band_{i}"] = X_train[f"freq_pos_in_band_{i}"].fillna(0)
        if X_test is not None:
            X_test[f"freq_pos_in_band_{i}"] = X_test[f"freq_pos_in_band_{i}"].fillna(0)

    # Fill any remaining NaN values in other columns
    X_train = X_train.fillna(0)
    if X_test is not None:
        X_test = X_test.fillna(0)

    return (
        X_train,
        pd.DataFrame(Y_raw_train),
        X_test,
        pd.DataFrame(Y_raw_test),
        voltage_scaler,
        freq_scaler,
    )
