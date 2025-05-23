import argparse
import json
from pathlib import Path

import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from helpers.features import process_dataset
from helpers.loaders import prepare_data_for_extrapolation
from helpers.loss import huber_logcosh_loss
from helpers.models import ComponentModel2
from helpers.trainers import train_component_model
from helpers.types import ActivationTypes, SchedulerTypes

DATASET_FILE_PATH = "dataset.csv"

MODELS = "models"
SUBFOLDER = "extrapolation"

BATCH_SIZE = 2048
EPOCHS = 300
PATIENCE = 40  # Irrelevant when using Optuna Pruner

TRIALS = 200
PRUNER = optuna.pruners.SuccessiveHalvingPruner()
SAMPLER = optuna.samplers.TPESampler()


def tune_model(
    selected_model: dict[str, str],
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    freq_idx: list[int],
    other_idx: list[int],
):
    print(f"[{selected_model['model_name']}] Starting Optuna tuning...")

    # Split training data for tuning (80% train, 20% val)
    X_train_part, X_val_part, Y_train_part, Y_val_part = train_test_split(
        X_train,
        Y_train[[selected_model["label"]]],
        test_size=0.2,
        random_state=42,
    )

    (
        X_train_tensor,
        Y_train_tensor,
        X_val_tensor,
        Y_val_tensor,
        loader,
        x_scaler,
        y_scaler,
    ) = prepare_data_for_extrapolation(
        X_train_part,
        Y_train_part,
        X_val_part,
        Y_val_part,
        batch_size=BATCH_SIZE,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.Trial) -> float:
        depth = trial.suggest_int("depth", 3, 9, step=2)  # Must be odd
        base_width = trial.suggest_categorical("base_width", [128, 256, 384, 512])
        peak_multiplier = trial.suggest_int("peak_multiplier", 2, 8)

        freq_depth = trial.suggest_int("freq_depth", 2, 6)
        freq_width = trial.suggest_categorical("freq_width", [128, 256, 384, 512])

        other_depth = trial.suggest_int("other_depth", 2, 6)
        other_width = trial.suggest_categorical(
            "other_width", [128, 256, 384, 512, 1024]
        )

        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

        # They are fixed for now
        # batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
        # epochs = 300
        # patience = trial.suggest_int("patience", 20, 60)

        lr_scheduler_type = trial.suggest_categorical(
            "lr_scheduler_type",
            [
                SchedulerTypes.REDUCE_ON_PLATEAU,
                SchedulerTypes.COSINE_ANNEALING,
                SchedulerTypes.ONE_CYCLE,
            ],
        )

        activation = trial.suggest_categorical(
            "activation",
            [ActivationTypes.RELU, ActivationTypes.GELU, ActivationTypes.SILU],
        )

        model = ComponentModel2(
            depth,
            base_width,
            dropout_rate,
            freq_idx,
            other_idx,
            activation,
            selected_model["model_name"],
            freq_depth,
            freq_width,
            other_depth,
            other_width,
            peak_multiplier,
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        if selected_model["model_name"] == "S21_real":
            criterion = nn.SmoothL1Loss(beta=0.05)
        else:
            criterion = huber_logcosh_loss

        trained_model, loss = train_component_model(
            model,
            selected_model["model_name"],
            loader,
            X_val_tensor,
            Y_val_tensor,
            criterion,
            optimizer,
            device,
            EPOCHS,
            PATIENCE,
            lr_scheduler_type,
            tqdm_disable=False,
            tqdm_position=1,
            suffix=f"Trial: {trial.number}/{TRIALS}",
            trial=trial,
        )

        return loss

    study = optuna.create_study(
        direction="minimize",
        pruner=PRUNER,
        sampler=SAMPLER,
    )
    study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)
    pruned_trials = [
        t.value for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print(f"{selected_model['model_name']} study statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print(f"  User attrs: {trial.user_attrs}")
    print(f"  System attrs: {trial.system_attrs}")
    print(f"  Intermediate values: {trial.intermediate_values}")
    print(f"  Duration: {trial.duration}")

    output_dir = Path(MODELS) / SUBFOLDER
    output_dir.mkdir(parents=True, exist_ok=True)
    param_file = output_dir / f"{selected_model['model_name']}_params.json"
    with open(param_file, "w") as f:
        json.dump(study.best_trial.params, f, indent=2)


if __name__ == "__main__":
    models_to_train = [
        {
            "model_name": "S11_real",
            "label": "S_deemb(1,1)_real",
        },
        {
            "model_name": "S11_imag",
            "label": "S_deemb(1,1)_imag",
        },
        {
            "model_name": "S12_real",
            "label": "S_deemb(1,2)_real",
        },
        {
            "model_name": "S12_imag",
            "label": "S_deemb(1,2)_imag",
        },
        {
            "model_name": "S21_real",
            "label": "S_deemb(2,1)_real",
        },
        {
            "model_name": "S21_imag",
            "label": "S_deemb(2,1)_imag",
        },
        {
            "model_name": "S22_real",
            "label": "S_deemb(2,2)_real",
        },
        {
            "model_name": "S22_imag",
            "label": "S_deemb(2,2)_imag",
        },
    ]

    parser = argparse.ArgumentParser(description="Select model to tune by index.")
    parser.add_argument(
        "model_idx",
        type=int,
        help=f"Index of model to tune (0 to {len(models_to_train) - 1})",
    )
    args = parser.parse_args()

    if args.model_idx < 0 or args.model_idx >= len(models_to_train):
        raise ValueError(
            f"Model index '{args.model_idx}' is out of range. "
            f"Available indices: 0 to {len(models_to_train) - 1}"
        )

    selected_model = models_to_train[args.model_idx]

    print(f"Selected model: {selected_model['model_name']}")
    print(f"Loading dataset from: {DATASET_FILE_PATH}")
    df = pd.read_csv(DATASET_FILE_PATH)
    (
        X_train,
        Y_train,
        X_test,
        Y_test,
        voltage_scaler,
        freq_scaler,
        freq_idx,
        other_idx,
    ) = process_dataset(df, split_mode="extrapolation", mute=True)

    tune_model(selected_model, X_train, Y_train, freq_idx, other_idx)
