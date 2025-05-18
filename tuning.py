import argparse
import json
from multiprocessing import Process, set_start_method
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

from helpers.base.loaders import prepare_data_for_pytorch
from helpers.base.models import FrequencyAwareNetwork
from helpers.base.trainers import train_model
from helpers.evals import evaluate_model
from helpers.spliters import create_frequency_based_split

# Load and prepare data
df = pd.read_csv("dataset.csv")
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
df = df.dropna(subset=feature_columns + label_columns)
X = pd.get_dummies(
    df[feature_columns].copy(), columns=["DEV_GEOM_L", "NUM_OF_TRANS_RF"], dtype=int
)
Y = df[label_columns].copy()

train_mask, test_mask = create_frequency_based_split(
    df, test_size=0.2, random_state=42, mute=True
)
X_train, X_test = X[train_mask], X[test_mask]
Y_train, Y_test = Y[train_mask], Y[test_mask]

freq_idx = [X_train.columns.get_loc("freq")]
other_idx = [i for i in range(X_train.shape[1]) if i not in freq_idx]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label groups
model_labels = [
    ("S_deemb(1,1)_real", "S_deemb(1,1)_imag"),
    ("S_deemb(1,2)_real", "S_deemb(1,2)_imag"),
    ("S_deemb(2,1)_real", "S_deemb(2,1)_imag"),
    ("S_deemb(2,2)_real", "S_deemb(2,2)_imag"),
]

arch_options = {
    "a": [64, 128, 256],
    "b": [128, 256, 512],
    "c": [256, 512, 1024],
    "d": [256, 512, 1024, 512],
    "e": [512, 1024, 2048, 1024],
    "f": [1024, 2048, 4096, 2048],
    "g": [256, 512, 1024, 512, 256],
    "h": [512, 1024, 2048, 1024, 512],
    "i": [1024, 2048, 4096, 2048, 1024],
    "j": [256, 512, 1024, 512, 256, 128],
    "k": [512, 1024, 2048, 1024, 512, 256],
    "l": [1024, 2048, 4096, 2048, 1024, 512],
}


def tune_model(label_pair, suffix):
    y_train_pair = Y_train[list(label_pair)]
    y_test_pair = Y_test[list(label_pair)]

    def objective(trial):
        arch_key = trial.suggest_categorical("arch", list(arch_options.keys()))
        hidden_sizes = arch_options[arch_key]

        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
        activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu"])
        lr_scheduler_type = trial.suggest_categorical(
            "lr_scheduler_type",
            [
                "step",
                "exponential",
                "reduce_on_plateau",
                "cosine_annealing",
                "one_cycle",
                "none",
            ],
        )
        patience = trial.suggest_int("patience", 30, 40)
        epochs = 300

        print(f"[{suffix}] Trial {trial.number} starting...")
        print(
            f"[{suffix}] Params: hidden_sizes={hidden_sizes}, dropout={dropout_rate:.2f}, "
            f"lr={learning_rate:.5f}, batch={batch_size}, activation={activation}, "
            f"scheduler={lr_scheduler_type}"
        )

        # Prepare data
        X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, loader, _ = (
            prepare_data_for_pytorch(
                X_train,
                y_train_pair,
                X_test,
                y_test_pair,
                batch_size=batch_size,
                scale_y=False,
            )
        )

        # Build model
        model = FrequencyAwareNetwork(
            freq_idx, other_idx, hidden_sizes, dropout_rate, activation
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        trained_model = train_model(
            model,
            loader,
            X_test_tensor,
            Y_test_tensor,
            criterion,
            optimizer,
            device,
            epochs=epochs,
            patience=patience,
            scheduler_str=lr_scheduler_type,
            tqdm_position=0 if suffix in ["S1", "S3"] else 1,
        )

        metrics = evaluate_model(trained_model, X_test_tensor, Y_test_tensor)
        mae_mean = float(np.mean(metrics["MAE"]))  # average across outputs
        print(f"[{suffix}] Trial {trial.number} finished - MAE: {mae_mean:.5f}")
        return mae_mean

    print(f"[{suffix}] Starting Optuna tuning...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"[{suffix}] Best trial MAE: {study.best_value:.5f}")
    print(f"[{suffix}] Best params: {study.best_params}")

    Path("models").mkdir(exist_ok=True)
    with open(f"models/{suffix}_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)


if __name__ == "__main__":
    set_start_method("spawn", force=True)

    # Argument parser for controlling parallelism
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for S-parameter models."
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=1,
        help="Number of models to tune in parallel (default: 1 = serial)",
    )
    args = parser.parse_args()
    parallel_jobs = args.parallel

    # Define all S-parameter groups
    label_groups = [
        ("S_deemb(1,1)_real", "S_deemb(1,1)_imag"),
        ("S_deemb(1,2)_real", "S_deemb(1,2)_imag"),
        ("S_deemb(2,1)_real", "S_deemb(2,1)_imag"),
        ("S_deemb(2,2)_real", "S_deemb(2,2)_imag"),
    ]

    print(f"Starting tuning with {parallel_jobs} parallel job(s)...")

    # Loop through label groups in chunks of size `parallel_jobs`
    for i in range(0, len(label_groups), parallel_jobs):
        processes = []
        for j in range(parallel_jobs):
            idx = i + j
            if idx < len(label_groups):
                label_pair = label_groups[idx]
                suffix = f"S{idx + 1}"
                p = Process(target=tune_model, args=(label_pair, suffix))
                p.start()
                processes.append(p)

        for p in processes:
            p.join()
