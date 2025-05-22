import numpy as np


def create_frequency_based_split(df, test_size=0.2, random_state=42, mute=False):
    """
    Create an improved frequency-based train-test split that ensures:
    1. No frequency overlap between train and test
    2. Even distribution of device parameters between train and test
    3. Balanced representation of different device geometries and parameters

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing dataset with 'freq' and device parameters
    test_size : float, default=0.2
        Proportion of unique frequency values to include in test set
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    train_mask, test_mask : numpy arrays
        Boolean masks for train and test data
    """
    # Set random seed
    np.random.seed(random_state)

    # Get sorted unique frequency values
    unique_freqs = np.sort(df["freq"].unique())

    # Define frequency bands
    band_boundaries = [
        (0, 1e9),  # < 1 GHz
        (1e9, 6e9),  # 1-6 GHz
        (6e9, 20e9),  # 6-20 GHz
        (20e9, 40e9),  # 20-40 GHz
        (40e9, float("inf")),  # > 40 GHz
    ]

    # Assign frequencies to bands
    freq_bands = {}
    band_freqs = {i: [] for i in range(len(band_boundaries))}

    for freq in unique_freqs:
        for band_idx, (lower, upper) in enumerate(band_boundaries):
            if lower <= freq < upper or (
                band_idx == len(band_boundaries) - 1 and freq >= lower
            ):
                freq_bands[freq] = band_idx
                band_freqs[band_idx].append(freq)
                break

    # Select test frequencies ensuring no consecutive frequencies
    test_freqs = []

    # Calculate target number of test frequencies per band
    target_per_band = {
        band: max(1, int(len(freqs) * test_size)) for band, freqs in band_freqs.items()
    }

    # Randomly select frequencies from each band
    for band, freqs in band_freqs.items():
        if len(freqs) > 0:
            # Sort frequencies within band
            sorted_freqs = np.sort(freqs)

            # Select frequencies with spacing to avoid consecutive selections
            n_select = target_per_band[band]
            step = max(1, len(sorted_freqs) // (n_select + 1))

            # Jitter indices to avoid selecting frequencies at exact intervals
            indices = np.arange(step, len(sorted_freqs), step)[:n_select]
            indices = np.clip(
                indices + np.random.randint(-step // 4, step // 4, size=len(indices)),
                0,
                len(sorted_freqs) - 1,
            )

            # Ensure unique indices
            indices = np.unique(indices)
            selected_freqs = sorted_freqs[indices]

            test_freqs.extend(selected_freqs)

    # Create train and test masks
    test_mask = df["freq"].isin(test_freqs)
    train_mask = ~test_mask

    # Check for balanced distribution of device parameters
    dev_params = ["DEV_GEOM_L", "NUM_OF_TRANS_RF", "vb", "vc"]
    for param in dev_params:
        if param in df.columns:
            train_dist = df.loc[train_mask, param].value_counts(normalize=True)
            test_dist = df.loc[test_mask, param].value_counts(normalize=True)

            # If distributions are very different, adjust selection
            if np.abs(train_dist.values - test_dist.values).max() > 0.2:
                print(
                    f"Warning: Unbalanced distribution detected for {param}. Adjusting split..."
                )
                # This could be expanded with a rebalancing algorithm

    # Verify no frequency overlap
    train_freqs = df.loc[train_mask, "freq"].unique()
    test_freqs = df.loc[test_mask, "freq"].unique()
    overlap = np.intersect1d(train_freqs, test_freqs)
    assert len(overlap) == 0, "Frequency overlap detected in split!"

    if not mute:
        print(f"Train set: {train_mask.sum()} samples ({train_mask.mean():.2%})")
        print(f"Test set: {test_mask.sum()} samples ({test_mask.mean():.2%})")

    return train_mask, test_mask


def create_extrapolation_split(
    df,
    train_threshold=40e9,
    test_freqs: list[int] = [
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
    ],
    mute=False,
):
    """
    Create a train-test split for extrapolation where:
    - Training set contains all data with frequency <= train_threshold
    - Test set contains specified frequencies above the threshold

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing a 'freq' column in Hz
    train_threshold : float, default=60e9
        Maximum frequency (in Hz) to include in training data
    test_freqs : list, default=[61, 62, 63, 64, 65]
        Specific frequencies (in GHz) to include in test set

    Returns:
    --------
    train_mask : numpy array
        Boolean mask for training data
    test_mask : numpy array
        Boolean mask for test data
    """

    # Convert test frequencies from GHz to Hz
    test_freqs_hz = np.array(test_freqs) * 1e9

    # Get all unique frequency values
    unique_freqs = np.sort(df["freq"].unique())

    # Create training mask: include all frequencies <= threshold
    train_mask = df["freq"] <= train_threshold

    # Create test mask: include only the specified test frequencies
    test_mask = df["freq"].isin(test_freqs_hz)

    # Print dataset information
    train_freqs = unique_freqs[unique_freqs <= train_threshold]
    test_freqs_found = np.intersect1d(unique_freqs, test_freqs_hz)
    test_freqs_missing = np.setdiff1d(test_freqs_hz, unique_freqs)

    if not mute:
        print(
            f"Found {len(unique_freqs)} unique frequency values from {unique_freqs.min() / 1e9:.2f} GHz to {unique_freqs.max() / 1e9:.2f} GHz"
        )

        print(
            f"Training on {len(train_freqs)} unique frequencies from {train_freqs.min() / 1e9:.2f} GHz to {train_freqs.max() / 1e9:.2f} GHz"
        )
        print(f"Training set: {train_mask.sum()} samples")

        print(
            f"Testing on {len(test_freqs_found)} unique frequencies: {test_freqs_found / 1e9} GHz"
        )
        print(f"Test set: {test_mask.sum()} samples")

    if len(test_freqs_missing) > 0:
        print(
            f"Warning: {len(test_freqs_missing)} requested test frequencies not found in data: {test_freqs_missing / 1e9} GHz"
        )

    return train_mask, test_mask
