from typing import TypedDict

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from helpers.base.models import FrequencyAwareNetwork
from helpers.evals import evaluate_model


class Config(TypedDict):
    hidden_sizes: list[int]
    dropout_rate: float
    learning_rate: float
    activation: str
    lr_scheduler_type: str
    epochs: int
    patience: int


def train_model(
    model,
    train_loader,
    X_test,
    Y_test,
    criterion,
    optimizer,
    device,
    epochs=150,
    patience=15,
    scheduler_str: str | None = None,
    tqdm_position=0,
    tqdm_disable=False,
    suffix=None,
):
    model.to(device)
    best_model_state = None
    best_loss = float("inf")
    patience_counter = 0

    # Initialize scheduler with sensible defaultss
    scheduler = None
    if scheduler_str:
        if scheduler_str == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.1
            )
        elif scheduler_str == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        elif scheduler_str == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.1
            )
        elif scheduler_str == "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
        elif scheduler_str == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]["lr"],
                steps_per_epoch=len(train_loader),
                epochs=epochs,
                pct_start=0.3,
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=1e4,
                three_phase=False,
            )
        elif scheduler_str == "none":
            scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_str}")

    with tqdm(
        range(epochs),
        desc="Training Epochs" if not suffix else f"Training Epochs ({suffix})",
        position=tqdm_position,
        disable=tqdm_disable,
    ) as pbar:
        for epoch in pbar:
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

                # Step OneCycleLR after each batch
                if scheduler and isinstance(
                    scheduler, torch.optim.lr_scheduler.OneCycleLR
                ):
                    scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_preds = model(X_test.to(device))
                val_loss = criterion(val_preds, Y_test.to(device)).item()

            # Step other schedulers after each epoch
            if scheduler and not isinstance(
                scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    pbar.write("Early stopping triggered.")
                    break

            # Update progress bar
            pbar.set_postfix(
                {
                    "Epoch": epoch + 1,
                    "Val Loss": f"{val_loss:.6f}",
                    "Best": f"{best_loss:.6f}",
                    "LR": optimizer.param_groups[0]["lr"],
                }
            )

    # Load and save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


class ChainedPredictor:
    def __init__(
        self,
        targets,
        freq_idx,
        hidden_sizes,
        dropout_rate=0.2,
        activation="silu",
        device="cuda",
    ):
        self.models = {}
        self.targets = targets
        self.freq_idx = freq_idx
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.device = device

    def train_chain(
        self,
        X_train_tensor,
        Y_train_tensor,
        X_test_tensor,
        Y_test_tensor,
        criterion,
        device,
        learning_rate=1e-3,
        epochs=150,
        patience=15,
        batch_size=128,
        scheduler_str: str | None = None,
    ):
        X_train_chain = X_train_tensor.clone().to(device)
        X_test_chain = X_test_tensor.clone().to(device)
        Y_train_tensor = Y_train_tensor.to(device)
        Y_test_tensor = Y_test_tensor.to(device)

        for i, target_group in enumerate(self.targets):
            if isinstance(target_group, tuple):
                output_dim = len(target_group)
                target_names = list(target_group)
                suffix = "+".join(target_names)
            else:
                output_dim = 1
                target_names = [target_group]
                suffix = target_group

            print(f"\n🔁 Training {suffix} ({i + 1}/{len(self.targets)})")

            other_idx = [
                j for j in range(X_train_chain.shape[1]) if j not in self.freq_idx
            ]
            model = FrequencyAwareNetwork(
                freq_idx=self.freq_idx,
                other_idx=other_idx,
                hidden_sizes=self.hidden_sizes,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                output_dim=output_dim,
            ).to(self.device)

            flat_target_list = [
                t
                for group in self.targets
                for t in (group if isinstance(group, tuple) else [group])
            ]
            target_indices = [flat_target_list.index(t) for t in target_names]
            y_train_target = Y_train_tensor[:, target_indices]
            y_test_target = Y_test_tensor[:, target_indices]

            train_loader = DataLoader(
                TensorDataset(X_train_chain, y_train_target),
                batch_size=batch_size,
                shuffle=True,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            model = train_model(
                model,
                train_loader,
                X_test_chain,
                y_test_target,
                criterion,
                optimizer,
                device,
                epochs=epochs,
                patience=patience,
                scheduler_str=scheduler_str,
                tqdm_disable=False,
                suffix=suffix,
            )

            for j, t in enumerate(target_names):
                self.models[t] = model

            with torch.no_grad():
                train_pred = model(X_train_chain)
                test_pred = model(X_test_chain)

                if train_pred.dim() == 1:
                    train_pred = train_pred.unsqueeze(1)
                if test_pred.dim() == 1:
                    test_pred = test_pred.unsqueeze(1)

            scores = evaluate_model(model, X_test_chain, y_test_target)
            print(f"R2: {scores['R2']}, MAE: {scores['MAE']}, RMSE: {scores['RMSE']}")

        print("\n✅ Chained training complete.")
