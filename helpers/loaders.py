import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def prepare_data_for_pytorch(
    X_train, Y_train, X_test, Y_test, batch_size=256, scale_y=False
):
    scaler_y = None

    # Extract freq tensors before dropping any columns
    freq_train_tensor = torch.tensor(X_train["freq"].values, dtype=torch.float32)
    freq_test_tensor = torch.tensor(X_test["freq"].values, dtype=torch.float32)

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

    # Now include freq in training dataset
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor, freq_train_tensor)
    train_loader = DataLoader(
        train_dataset,
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
        freq_test_tensor,
    )
