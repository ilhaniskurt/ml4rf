import time

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(
    model,
    train_loader,
    X_test_tensor,
    Y_test_tensor,
    criterion,
    optimizer,
    device,
    epochs=100,
    early_stopping_patience=15,
    verbose=True,
    lr_scheduler_type="reduce_on_plateau",
    warmup_epochs=5,
):
    """Train a PyTorch model with early stopping and learning rate scheduling."""
    model = model.to(device)

    # Set up learning rate scheduler based on specified type
    if lr_scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.85, patience=5, verbose=verbose, min_lr=5e-7
        )
    elif lr_scheduler_type == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif lr_scheduler_type == "one_cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            steps_per_epoch=len(train_loader),
            epochs=epochs,
        )
    else:
        scheduler = None

    # For early stopping
    best_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    # Track losses and learning rates for plotting
    train_losses = []
    val_losses = []
    learning_rates = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Apply learning rate warmup if needed
        if warmup_epochs > 0 and epoch < warmup_epochs and scheduler is None:
            lr_multiplier = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = optimizer.param_groups[0]["lr"] * lr_multiplier

        # Record current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Step OneCycleLR scheduler here if being used
            if lr_scheduler_type == "one_cycle":
                scheduler.step()

            running_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor.to(device))
            val_loss = criterion(val_outputs, Y_test_tensor.to(device)).item()
            val_losses.append(val_loss)

        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.8f}"
            )

        # Learning rate scheduler step (except for OneCycleLR which is done per iteration)
        if scheduler is not None:
            if lr_scheduler_type == "reduce_on_plateau":
                scheduler.step(val_loss)
            elif lr_scheduler_type == "cosine_annealing":
                scheduler.step()

        # Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Plot learning rate schedule
    # plt.figure(figsize=(10, 4))
    # plt.plot(learning_rates)
    # plt.xlabel("Epochs")
    # plt.ylabel("Learning Rate")
    # plt.title("Learning Rate Schedule")
    # plt.yscale("log")
    # plt.savefig("freq_aware_results/learning_rate_schedule.png")
    # plt.close()

    return model, train_losses, val_losses


def train_frequency_aware_models(
    X_train, X_test, Y_train, Y_test, hyperparameters=None, selected_features=None
):
    """
    Train frequency-aware models for each S-parameter with conditional scaling.
    """
    # S-parameter definitions
    s_parameter_models = {
        "S21": ["S_deemb(2,1)_real", "S_deemb(2,1)_imag"],
    }

    # 'S12': ['S_deemb(1,2)_real', 'S_deemb(1,2)_imag']

    # Set default hyperparameters if not provided
    if hyperparameters is None:
        hyperparameters = {
            "hidden_sizes": [64, 128, 256],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 150,
            "early_stopping_patience": 15,
            "activation": "gelu",
            "lr_scheduler_type": "one_cycle",
        }

    # Filter features if requested
    if selected_features is not None:
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        print(f"Using {len(selected_features)} selected features")

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Identify frequency-related features
    freq_indices, other_indices = identify_frequency_features(X_train.columns)

    # Store results and models
    models = {}
    all_results = {}
    all_predictions = {}
    scalers = {}  # Store scalers for each model

    # Record start time
    start_time = time.time()

    # Train a model for each S-parameter
    for model_name, components in s_parameter_models.items():
        print(f"\n{'=' * 50}")
        print(f"Training frequency-aware model for {model_name}")
        print(f"{'=' * 50}")

        # Decide whether to scale Y data (only for S12)
        scale_y = model_name == "S12"

        # Prepare data with conditional scaling
        prep_results = prepare_data_for_pytorch_with_scaling(
            X_train,
            Y_train,
            X_test,
            Y_test,
            components,
            hyperparameters["batch_size"],
            scale_y=scale_y,
        )

        if scale_y:
            (
                X_train_tensor,
                Y_train_tensor,
                X_test_tensor,
                Y_test_tensor,
                train_loader,
                y_scaler,
            ) = prep_results
            scalers[model_name] = y_scaler
            print("Applied StandardScaler to Y values for S12")
        else:
            (
                X_train_tensor,
                Y_train_tensor,
                X_test_tensor,
                Y_test_tensor,
                train_loader,
                _,
            ) = prep_results

        # Initialize model
        model = FrequencyAwareNetwork(
            len(freq_indices),
            len(other_indices),
            hyperparameters["hidden_sizes"],
            hyperparameters["dropout_rate"],
            hyperparameters.get("activation", "gelu"),
        )
        model.set_feature_indices(freq_indices, other_indices)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

        # Train model (use your existing train_model function)
        trained_model, train_losses, val_losses = train_model(
            model,
            train_loader,
            X_test_tensor,
            Y_test_tensor,
            criterion,
            optimizer,
            device,
            hyperparameters["epochs"],
            hyperparameters["early_stopping_patience"],
            lr_scheduler_type=hyperparameters.get("lr_scheduler_type", "one_cycle"),
        )

        # Plot learning curves
        plot_learning_curves(train_losses, val_losses, model_name)

        # Evaluate model with proper scaling handling
        metrics, avg_metrics, predictions = evaluate_model_with_scaling(
            trained_model,
            X_test_tensor,
            Y_test_tensor,
            Y_test,
            components,
            device,
            scalers.get(model_name),
        )

        # Plot predictions and error distributions
        plot_predictions(Y_test, predictions, components, model_name)
        plot_error_distribution(Y_test, predictions, components, model_name)

        # Print results
        print(f"\nPerformance metrics for {model_name}:")
        for component, metric in metrics.items():
            print(f"  {component}:")
            print(f"    RMSE: {metric['rmse']:.6f}")
            print(f"    R²: {metric['r2']:.6f}")
            print(f"    MAE: {metric['mae']:.6f}")
            if "smape" in metric:
                print(f"    SMAPE: {metric['smape']:.2f}%")
            else:
                print(f"    MAPE: {metric['mape']:.2f}%")

        print(f"\nAverage metrics for {model_name}:")
        print(f"  R²: {avg_metrics['r2']:.6f}")
        print(f"  RMSE: {avg_metrics['rmse']:.6f}")
        print(f"  MAE: {avg_metrics['mae']:.6f}")
        if "smape" in avg_metrics:
            print(f"  SMAPE: {avg_metrics['smape']:.2f}%")
        else:
            print(f"  MAPE: {avg_metrics['mape']:.2f}%")

        # Store results
        models[model_name] = trained_model
        all_results[model_name] = {
            "component_metrics": metrics,
            "avg_metrics": avg_metrics,
        }
        all_predictions[model_name] = predictions

    # Record total training time
    train_time = time.time() - start_time
    print(f"\nTotal training time: {train_time:.2f} seconds")

    # Save models
    for model_name, model in models.items():
        torch.save(model.state_dict(), f"freq_aware_results/{model_name}_model.pth")

    print("Models and results saved to freq_aware_results/")

    return models, all_results, all_predictions, scalers
