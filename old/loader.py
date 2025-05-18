import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def prepare_data_for_pytorch(
    X_train, Y_train, X_test, Y_test, components, batch_size, scale_y=True
):
    """Prepare data for PyTorch models with optional Y-scaling."""

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    X_test_tensor = torch.FloatTensor(X_test.values)

    # Handle Y data scaling if requested
    if scale_y:
        # Create scaler for Y values
        y_scaler = StandardScaler()
        Y_train_values = Y_train[components].values
        Y_test_values = Y_test[components].values

        # Fit scaler and transform data
        Y_train_scaled = y_scaler.fit_transform(Y_train_values)
        Y_test_scaled = y_scaler.transform(Y_test_values)

        # Convert to tensors
        Y_train_tensor = torch.FloatTensor(Y_train_scaled)
        Y_test_tensor = torch.FloatTensor(Y_test_scaled)

        # Save scaler for later use
        # component_str = "_".join(components)
        # joblib.dump(y_scaler, f"freq_aware_results/{component_str}_scaler.pkl")

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        return (
            X_train_tensor,
            Y_train_tensor,
            X_test_tensor,
            Y_test_tensor,
            train_loader,
            y_scaler,
        )
