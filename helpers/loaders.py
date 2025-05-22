import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def prepare_data_for_interpolation(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    batch_size=256,
    scale_y=False,
):
    scaler_y = None

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32)

    if scale_y:
        scaler_y = StandardScaler()
        Y_train_scaled = scaler_y.fit_transform(Y_train_tensor)  # type: ignore
        Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
        Y_test_scaled = scaler_y.transform(Y_test_tensor)
        Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, Y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return (
        X_train_tensor,
        Y_train_tensor,
        X_test_tensor,
        Y_test_tensor,
        train_loader,
        scaler_y,
    )


def prepare_data_for_extrapolation(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    batch_size=256,
):
    non_freq_cols = [col for col in X_train.columns if "freq" not in col.lower()]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    Y_train_scaled = Y_train.copy()
    Y_test_scaled = Y_test.copy()

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled[non_freq_cols] = x_scaler.fit_transform(X_train[non_freq_cols])
    X_test_scaled[non_freq_cols] = x_scaler.transform(X_test[non_freq_cols])
    Y_train_scaled = y_scaler.fit_transform(Y_train_scaled)
    Y_test_scaled = y_scaler.transform(Y_test_scaled)

    X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, Y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    return (
        X_train_tensor,
        Y_train_tensor,
        X_test_tensor,
        Y_test_tensor,
        train_loader,
        x_scaler,
        y_scaler,
    )
