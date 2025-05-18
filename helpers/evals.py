import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
