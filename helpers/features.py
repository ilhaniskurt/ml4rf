import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from helpers.spliters import create_extrapolation_split, create_frequency_based_split


def __preprocess_frequency(
    X_train: pd.DataFrame, X_test: pd.DataFrame, mode="interpolation"
):
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

    if mode == "extrapolation":
        max_freq = X_train["freq"].max()
        X_train_processed["freq_ratio"] = X_train["freq"] / max_freq
        X_test_processed["freq_ratio"] = X_test["freq"] / max_freq

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


def __identify_frequency_features(columns, mute=False):
    """Identify frequency-related features in the dataset."""
    freq_features = [
        i
        for i, col in enumerate(columns)
        if "freq" in col.lower() or "band" in col.lower()
    ]
    other_features = [i for i in range(len(columns)) if i not in freq_features]

    if not mute:
        print(
            f"Identified {len(freq_features)} frequency-related features and {len(other_features)} other features"
        )
    return freq_features, other_features


def process_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    split_mode="interpolation",
    mute=False,
):
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

    if split_mode == "extrapolation":
        train_mask, test_mask = create_extrapolation_split(df_clean, mute=mute)
    else:
        train_mask, test_mask = create_frequency_based_split(
            df_clean, test_size=test_size, random_state=random_state, mute=mute
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

    X_train, X_test, freq_scaler = __preprocess_frequency(
        X_train, X_test, mode=split_mode
    )

    # Fill NaN values with 0 for freq_pos_in_band columns
    for i in range(1, 6):
        X_train[f"freq_pos_in_band_{i}"] = X_train[f"freq_pos_in_band_{i}"].fillna(0)
        if X_test is not None:
            X_test[f"freq_pos_in_band_{i}"] = X_test[f"freq_pos_in_band_{i}"].fillna(0)

    # Fill any remaining NaN values in other columns
    X_train = X_train.fillna(0)
    if X_test is not None:
        X_test = X_test.fillna(0)

    # Identify frequency-related features (Assuming column orders are the same for train and test)
    freq_idx, other_idx = __identify_frequency_features(X_train.columns, mute=mute)

    return (
        X_train,
        pd.DataFrame(Y_raw_train),
        X_test,
        pd.DataFrame(Y_raw_test),
        voltage_scaler,
        freq_scaler,
        freq_idx,
        other_idx,
    )


def get_min_max_sparam(X_train: pd.DataFrame, Y_train: pd.DataFrame):
    s21_real_train = Y_train["S_deemb(2,1)_real"].values
    s21_imag_train = Y_train["S_deemb(2,1)_imag"].values

    X_train_s21 = X_train.copy()
    X_train_s21["S21_real"] = s21_real_train
    X_train_s21["S21_imag"] = s21_imag_train

    # Calculate different bounds for real and imaginary parts
    # For real part - tighter bounds due to problems with this component
    real_p10 = np.percentile(s21_real_train, 10)  # type: ignore
    real_p90 = np.percentile(s21_real_train, 90)  # type: ignore
    real_range = real_p90 - real_p10
    real_min = real_p10 - 0.2 * real_range  # Tighter bound for real
    real_max = real_p90 + 0.2 * real_range

    # For imaginary part - more relaxed bounds since it's behaving better
    imag_p05 = np.percentile(s21_imag_train, 5)  # type: ignore
    imag_p95 = np.percentile(s21_imag_train, 95)  # type: ignore
    imag_range = imag_p95 - imag_p05
    imag_min = imag_p05 - 0.3 * imag_range  # More relaxed bound
    imag_max = imag_p95 + 0.3 * imag_range

    print("Setting component-specific bounds:")
    print(f"  Real: [{real_min:.6f}, {real_max:.6f}]")
    print(f"  Imaginary: [{imag_min:.6f}, {imag_max:.6f}]")

    return real_min, real_max, imag_min, imag_max
