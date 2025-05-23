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
            freq_layers.append(activation_fn)
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


class ComponentModel(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        dropout_rate,
        freq_indices,
        other_indices,
        activation,
        model_name: str,
        freq_hidden_sizes: list[int] | None = None,
        other_hidden_sizes: list[int] | None = None,
    ):
        super().__init__()
        self.freq_indices = freq_indices
        self.other_indices = other_indices

        if freq_hidden_sizes is None:
            freq_hidden_sizes = [64, 128]
        if other_hidden_sizes is None:
            other_hidden_sizes = [64, 128]

        activation_fn = None
        if activation == "silu":
            activation_fn = nn.SiLU()
        elif activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "gelu":
            activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Frequency branch with custom architecture
        freq_layers = []
        input_size_freq = len(freq_indices)
        for i in range(len(freq_hidden_sizes)):
            out_size = freq_hidden_sizes[i]
            freq_layers.append(nn.Linear(input_size_freq, out_size))
            freq_layers.append(nn.BatchNorm1d(out_size))
            freq_layers.append(activation_fn)
            freq_layers.append(nn.Dropout(dropout_rate))
            input_size_freq = out_size
        self.freq_layers = nn.Sequential(*freq_layers)

        # Other parameters branch with custom architecture
        other_layers = []
        input_size_other = len(other_indices)
        for i in range(len(other_hidden_sizes)):
            out_size = other_hidden_sizes[i]
            other_layers.append(nn.Linear(input_size_other, out_size))
            other_layers.append(nn.BatchNorm1d(out_size))
            other_layers.append(activation_fn)
            other_layers.append(nn.Dropout(dropout_rate))
            input_size_other = out_size
        self.other_layers = nn.Sequential(*other_layers)

        # Attention mechanism for better integration of branches
        self.attention = nn.Sequential(
            nn.Linear(freq_hidden_sizes[-1] + other_hidden_sizes[-1], 64),
            activation_fn,
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )

        # Combined layers
        combined_size = freq_hidden_sizes[-1] + other_hidden_sizes[-1]

        combined_layers = []
        input_size_combined = combined_size
        for i in range(len(hidden_sizes) - 1):
            combined_layers.append(nn.Linear(input_size_combined, hidden_sizes[i]))
            combined_layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            combined_layers.append(activation_fn)
            combined_layers.append(nn.Dropout(dropout_rate))
            input_size_combined = hidden_sizes[i]

        # Output layer
        combined_layers.append(
            nn.Linear(
                hidden_sizes[-2] if len(hidden_sizes) > 1 else input_size_combined,
                1,
            )
        )

        # Apply tanh only to real component to constrain outputs
        if model_name == "S21_real":
            combined_layers.append(nn.Tanh())

        self.combined_layers = nn.Sequential(*combined_layers)

    def forward(self, x):
        # Extract frequency and other inputs
        freq_input = x[:, self.freq_indices]
        other_input = x[:, self.other_indices]

        # Process through respective branches
        freq_features = self.freq_layers(freq_input)
        other_features = self.other_layers(other_input)

        # Combine features
        combined = torch.cat([freq_features, other_features], dim=1)

        # Apply attention mechanism
        attention_weights = self.attention(combined)
        weighted_freq = freq_features * attention_weights[:, 0].unsqueeze(1)
        weighted_other = other_features * attention_weights[:, 1].unsqueeze(1)

        # New combined features with attention
        combined_attention = torch.cat([weighted_freq, weighted_other], dim=1)

        # Final processing
        return self.combined_layers(combined_attention)


class ComponentModel2(nn.Module):
    def __init__(
        self,
        depth: int,
        base_width: int,
        dropout_rate: float,
        freq_indices: list[int],
        other_indices: list[int],
        activation: str,
        model_name: str,
        freq_depth: int = 2,
        freq_width: int = 128,
        other_depth: int = 2,
        other_width: int = 128,
        peak_multiplier: int = 4,
    ):
        super().__init__()
        self.freq_indices = freq_indices
        self.other_indices = other_indices

        # Activation
        activation_fn = {
            "silu": nn.SiLU(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
        }.get(activation.lower(), None)

        if activation_fn is None:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Frequency branch
        self.freq_layers = self._build_branch(
            len(freq_indices), freq_depth, freq_width, dropout_rate, activation_fn
        )
        # Other branch
        self.other_layers = self._build_branch(
            len(other_indices), other_depth, other_width, dropout_rate, activation_fn
        )

        combined_size = freq_width + other_width

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(combined_size, 64),
            activation_fn,
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )

        # Build symmetric pyramid hidden_sizes
        widths = self._make_symmetric_hidden_sizes(depth, base_width, peak_multiplier)

        # Combined MLP
        layers = []
        input_size = combined_size
        for w in widths:
            layers.append(nn.Linear(input_size, w))
            layers.append(nn.BatchNorm1d(w))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_rate))
            input_size = w

        layers.append(nn.Linear(input_size, 1))
        if model_name == "S21_real":
            layers.append(nn.Tanh())
        self.combined_layers = nn.Sequential(*layers)

    def _make_symmetric_hidden_sizes(self, depth, base, peak_mult):
        if depth % 2 == 0:
            raise ValueError("Only odd depths supported for symmetric architecture")
        mid = depth // 2
        sizes = []
        for i in range(depth):
            dist = abs(i - mid)
            factor = peak_mult / (2**dist)
            sizes.append(int(base * factor))
        return sizes

    def _build_branch(self, in_dim, depth, width, dropout, activation_fn):
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout))
            in_dim = width
        return nn.Sequential(*layers)

    def forward(self, x):
        freq = self.freq_layers(x[:, self.freq_indices])
        other = self.other_layers(x[:, self.other_indices])
        combined = torch.cat([freq, other], dim=1)

        attn = self.attention(combined)
        weighted = torch.cat(
            [freq * attn[:, 0].unsqueeze(1), other * attn[:, 1].unsqueeze(1)], dim=1
        )

        return self.combined_layers(weighted)
