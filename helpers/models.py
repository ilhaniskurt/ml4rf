import torch
import torch.nn as nn


class FrequencyAwareNetwork(nn.Module):
    def __init__(
        self,
        freq_features,
        other_features,
        hidden_sizes,
        dropout_rate,
        activation,
    ):
        super().__init__()

        if activation == "silu":
            activation_fn = nn.SiLU()
        elif activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "gelu":
            activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Frequency-specific processing branch
        freq_layers = []
        prev_size = freq_features
        for h_size in hidden_sizes[:2]:  # First two hidden sizes for branches
            freq_layers.append(nn.Linear(prev_size, h_size))
            freq_layers.append(
                activation_fn
            )  # Using SiLU (Swish) activation for better performance
            freq_layers.append(nn.BatchNorm1d(h_size))
            freq_layers.append(nn.Dropout(dropout_rate))
            prev_size = h_size

        self.freq_branch = nn.Sequential(*freq_layers)

        # Other parameters branch
        other_layers = []
        prev_size = other_features
        for h_size in hidden_sizes[:2]:
            other_layers.append(nn.Linear(prev_size, h_size))
            other_layers.append(activation_fn)
            other_layers.append(nn.BatchNorm1d(h_size))
            other_layers.append(nn.Dropout(dropout_rate))
            prev_size = h_size

        self.other_branch = nn.Sequential(*other_layers)

        # Combined processing with residual connections
        combined_layers = []
        prev_size = hidden_sizes[1] * 2  # Output size from both branches combined

        for h_size in hidden_sizes[2:]:
            combined_layers.append(nn.Linear(prev_size, h_size))
            combined_layers.append(activation_fn)
            combined_layers.append(nn.BatchNorm1d(h_size))
            combined_layers.append(nn.Dropout(dropout_rate))
            prev_size = h_size

        # Final output layer for real and imaginary components
        combined_layers.append(nn.Linear(prev_size, 2))

        self.combined = nn.Sequential(*combined_layers)

        # Store feature indices for processing
        self.freq_indices = None
        self.other_indices = None

    def forward(self, x):
        # Split input into frequency and other features
        if self.freq_indices is None or self.other_indices is None:
            raise ValueError(
                "Feature indices not set. Call set_feature_indices() first."
            )

        freq_input = x[:, self.freq_indices]
        other_input = x[:, self.other_indices]

        # Process through branches
        freq_features = self.freq_branch(freq_input)
        other_features = self.other_branch(other_input)

        # Combine and output
        combined = torch.cat([freq_features, other_features], dim=1)
        return self.combined(combined)

    def set_feature_indices(self, freq_indices, other_indices):
        """Set indices for frequency and other features."""
        self.freq_indices = freq_indices
        self.other_indices = other_indices


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = FrequencyAwareNetwork(
        checkpoint["freq_idx"],
        checkpoint["other_idx"],
        checkpoint["hidden_sizes"],
        checkpoint["dropout_rate"],
        checkpoint["activation"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
