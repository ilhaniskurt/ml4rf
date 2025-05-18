import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_frequency(X_train, X_test=None, fit_mode=True):
    """
    Comprehensive frequency preprocessing using domain knowledge-based bands

    Parameters:
    -----------
    X_train : pandas DataFrame
        Training DataFrame containing 'freq' column and other features
    X_test : pandas DataFrame, optional
        Test DataFrame to transform using parameters from training data
    fit_mode : bool, default=True
        If True, fit scalers on X_train and transform both X_train and X_test
        If False, load pre-fitted scalers and transform X_train only

    Returns:
    --------
    X_train_processed : pandas DataFrame
        Training DataFrame with added frequency features
    X_test_processed : pandas DataFrame or None
        Test DataFrame with added frequency features (if X_test was provided)
    """
    # Create copies to avoid modifying originals
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy() if X_test is not None else None

    # 1. Log transform to handle wide frequency range
    X_train_processed["freq_log"] = np.log10(X_train_processed["freq"])
    if X_test_processed is not None:
        X_test_processed["freq_log"] = np.log10(X_test_processed["freq"])

    # 2. Scale log-transformed frequency to [-1, 1] range
    if fit_mode:
        # Fit on training data only
        freq_scaler = MinMaxScaler(feature_range=(-1, 1))
        freq_scaler.fit(X_train_processed[["freq_log"]])
        # Save the scaler for later use
        joblib.dump(freq_scaler, "freq_log_scaler.pkl")
    else:
        # Load pre-fitted scaler
        freq_scaler = joblib.load("freq_log_scaler.pkl")

    # Transform both datasets with the same scaler
    X_train_processed["freq_log_norm"] = freq_scaler.transform(
        X_train_processed[["freq_log"]]
    )
    if X_test_processed is not None:
        X_test_processed["freq_log_norm"] = freq_scaler.transform(
            X_test_processed[["freq_log"]]
        )

    # 3. Create domain knowledge-based frequency bands
    # RF engineering standard bands
    for data in [X_train_processed, X_test_processed]:
        if data is None:
            continue

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
        if data is None:
            continue

        data["freq_squared"] = data["freq_log_norm"] ** 2
        data["freq_inv"] = 1.0 / (
            data["freq_log_norm"] + 1e-10
        )  # Add small epsilon to avoid division by zero

    # 5. Create interaction terms between frequency and device parameters
    for data in [X_train_processed, X_test_processed]:
        if data is None:
            continue

        if "Cmu_abs_log" in data.columns:
            data["freq_Cmu_interaction"] = data["freq_log_norm"] * data["Cmu_abs_log"]

        if "gm_abs_log" in data.columns:
            data["freq_gm_interaction"] = data["freq_log_norm"] * data["gm_abs_log"]

    # 6. Interactions with device geometry
    for data in [X_train_processed, X_test_processed]:
        if data is None:
            continue

        for device_length in ["DEV_L_0_9um", "DEV_L_2_5um", "DEV_L_5_0um"]:
            if device_length in data.columns:
                data[f"freq_{device_length}_interaction"] = (
                    data["freq_log_norm"] * data[device_length]
                )

    # 7. Create band-specific features
    # Critical: use training data boundaries for both datasets!
    band_boundaries = []

    if fit_mode:
        # Determine band boundaries from training data only
        band_boundaries = [
            (X_train_processed["freq"].min(), 1e9),  # Band 1
            (1e9, 6e9),  # Band 2
            (6e9, 20e9),  # Band 3
            (20e9, 40e9),  # Band 4
            (40e9, X_train_processed["freq"].max()),  # Band 5
        ]
        # Save boundaries for future use
        joblib.dump(band_boundaries, "freq_band_boundaries.pkl")
    else:
        # Load pre-computed boundaries
        band_boundaries = joblib.load("freq_band_boundaries.pkl")

    # Process each dataset using the same boundaries
    for data in [X_train_processed, X_test_processed]:
        if data is None:
            continue

        for i in range(1, 6):
            band_col = f"freq_band_{i}"
            mask = data[band_col] == 1

            if mask.sum() > 0:  # Only if there are samples in this band
                lower_bound, upper_bound = band_boundaries[i - 1]

                # Create normalized position within band (0 to 1)
                data.loc[mask, f"freq_pos_in_band_{i}"] = (
                    data.loc[mask, "freq"] - lower_bound
                ) / (upper_bound - lower_bound)

                # Clip values to [0,1] range in case test data exceeds training boundaries
                if data is X_test_processed:
                    col_name = f"freq_pos_in_band_{i}"
                    data.loc[mask, col_name] = data.loc[mask, col_name].clip(0, 1)

    # 8. Create interaction terms between bands and key parameters
    for data in [X_train_processed, X_test_processed]:
        if data is None:
            continue

        # Band-specific GM interactions
        if "gm_abs_log" in data.columns:
            for i in range(1, 6):
                data[f"band_{i}_gm"] = data[f"freq_band_{i}"] * data["gm_abs_log"]

        # Band-specific capacitance interactions
        if "Cmu_abs_log" in data.columns and "Cpi_abs_log" in data.columns:
            for i in range(1, 6):
                data[f"band_{i}_Cmu"] = data[f"freq_band_{i}"] * data["Cmu_abs_log"]
                data[f"band_{i}_Cpi"] = data[f"freq_band_{i}"] * data["Cpi_abs_log"]

        # Band-specific impedance interactions
        if "Zin_real_log" in data.columns and "Zin_imag_log" in data.columns:
            for i in range(1, 6):
                data[f"band_{i}_Zin_real"] = (
                    data[f"freq_band_{i}"] * data["Zin_real_log"]
                )
                data[f"band_{i}_Zin_imag"] = (
                    data[f"freq_band_{i}"] * data["Zin_imag_log"]
                )

    if X_test_processed is not None:
        return X_train_processed, X_test_processed
    else:
        return X_train_processed
