import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def evaluate_model(model, X_test, Y_test):
    model.eval()
    device = next(model.parameters()).device  # detect model device
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        y_true = Y_test.to(device).cpu().numpy()
    return {
        "R2": r2_score(y_true, preds, multioutput="raw_values"),
        "MAE": mean_absolute_error(y_true, preds, multioutput="raw_values"),
        "RMSE": np.sqrt(mean_squared_error(y_true, preds, multioutput="raw_values")),
    }


def __symmetric_mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-10):
    """Calculate SMAPE with protection against division by zero."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
    numerator = np.abs(y_true - y_pred)
    smape = numerator / denominator
    return np.mean(smape) * 100


def __mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-10):
    """Calculate MAPE with protection against division by zero."""
    non_zero = np.abs(y_true) > epsilon
    if non_zero.sum() == 0:
        return np.nan
    percentage_errors = (
        np.abs(
            (y_true[non_zero] - y_pred[non_zero]) / (np.abs(y_true[non_zero]) + epsilon)
        )
        * 100
    )
    return np.mean(percentage_errors)


def evaluate_model_with_scaling(
    model: nn.Module,
    X_test_tensor: torch.Tensor,
    Y_test: pd.DataFrame,
    labels: list[str],
    device: torch.device,
    y_scaler: StandardScaler | None = None,
):
    """Evaluate a trained model and calculate performance metrics."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        predictions = model(X_test_tensor.to(device)).cpu().numpy()

    # Inverse transform if scaler was used
    if y_scaler is not None:
        predictions_original = y_scaler.inverse_transform(predictions)
    else:
        predictions_original = predictions

    y_test_original = Y_test[labels].values

    # Calculate metrics
    metrics = {}

    for i, component in enumerate(labels):
        y_true = y_test_original[:, i]
        y_pred = predictions_original[:, i]

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        # Use SMAPE instead of MAPE for S12
        if "S12" in component or "S_deemb(1,2)" in component:
            smape_val = __symmetric_mean_absolute_percentage_error(y_true, y_pred)
            metrics[component] = {
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "mae": mae,
                "smape": smape_val,
            }
        else:
            # Regular MAPE for other S-parameters
            metrics[component] = {
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "mae": mae,
                "mape": __mean_absolute_percentage_error(y_true, y_pred),
            }

    # Calculate average metrics
    avg_metrics = {
        "rmse": np.mean([metrics[comp]["rmse"] for comp in labels]),
        "r2": np.mean([metrics[comp]["r2"] for comp in labels]),
        "mae": np.mean([metrics[comp]["mae"] for comp in labels]),
    }

    # Add SMAPE or MAPE average depending on which components were evaluated
    if any("S12" in comp or "S_deemb(1,2)" in comp for comp in labels):
        avg_metrics["smape"] = np.mean([metrics[comp]["smape"] for comp in labels])
    else:
        avg_metrics["mape"] = np.mean([metrics[comp]["mape"] for comp in labels])

    return metrics, avg_metrics, predictions_original


def evaluate_component_model(
    model: nn.Module,
    X_test_tensor: torch.Tensor,
    Y_test: pd.DataFrame,
    labels: list[str],
    model_name: str,
    real_min,
    real_max,
    imag_min,
    imag_max,
    device: torch.device,
    y_scaler: StandardScaler,
):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        predictions = model(X_test_tensor.to(device)).cpu().numpy()

    predictions = y_scaler.inverse_transform(predictions)

    if model_name == "S21_real":
        predictions = np.clip(predictions, real_min, real_max)
    else:
        predictions = np.clip(predictions, imag_min, imag_max)

    y_true = Y_test[labels].values
    y_pred = predictions.flatten()

    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": __mean_absolute_percentage_error(y_true, y_pred),
    }

    return metrics, predictions
