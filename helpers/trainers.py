from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int = 150,
    patience: int = 15,
    scheduler_str: Optional[str] = None,
    tqdm_position: int = 0,
    tqdm_disable: bool = False,
    suffix: Optional[str] = None,
    warmup_epochs: int = 5,
    max_grad_norm: float = 1.0,
) -> nn.Module:
    model.to(device)
    best_model_state = None
    best_loss = float("inf")
    patience_counter = 0
    initial_lr = optimizer.param_groups[0]["lr"]

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
                max_lr=initial_lr,
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
            # Warm-up learning rate (if no OneCycleLR)
            if (
                warmup_epochs > 0
                and epoch < warmup_epochs
                and scheduler_str != "one_cycle"
            ):
                lr_scale = (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = initial_lr * lr_scale

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()

                # ✅ Gradient clipping
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=max_grad_norm
                    )

                optimizer.step()

                # ✅ Step OneCycleLR scheduler per batch
                if scheduler and isinstance(
                    scheduler, torch.optim.lr_scheduler.OneCycleLR
                ):
                    scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_preds = model(X_test.to(device))
                val_loss = criterion(val_preds, Y_test.to(device)).item()

            # ✅ Step other schedulers per epoch
            if scheduler and not isinstance(
                scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # ✅ Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    pbar.write("Early stopping triggered.")
                    break

            pbar.set_postfix(
                {
                    "Epoch": epoch + 1,
                    "Val Loss": f"{val_loss:.6f}",
                    "Best": f"{best_loss:.6f}",
                    "LR": optimizer.param_groups[0]["lr"],
                }
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model
