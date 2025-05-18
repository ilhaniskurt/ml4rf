import torch
from tqdm import tqdm


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
    freq_test=None,  # NEW: for validation loss
):
    model.to(device)
    best_model_state = None
    best_loss = float("inf")
    patience_counter = 0

    # Initialize scheduler with sensible defaults
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
            for batch in train_loader:
                if len(batch) == 3:
                    xb, yb, freq = batch
                    xb, yb, freq = xb.to(device), yb.to(device), freq.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb, freq)
                else:
                    xb, yb = batch
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if scheduler and isinstance(
                    scheduler, torch.optim.lr_scheduler.OneCycleLR
                ):
                    scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_preds = model(X_test.to(device))
                if freq_test is not None:  # required for custom validation loss
                    val_loss = criterion(
                        val_preds, Y_test.to(device), freq_test.to(device)
                    ).item()
                else:
                    val_loss = criterion(val_preds, Y_test.to(device)).item()

            # Scheduler step
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

            # Progress bar update
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
