import torch
import torch.nn as nn


class FrequencyAwareNetwork(nn.Module):
    def __init__(
        self,
        freq_indices,
        other_indices,
        hidden_sizes=[64, 128, 256],
        dropout_rate=0.2,
        activation="silu",
    ):
        super().__init__()

        self.freq_indices = freq_indices
        self.other_indices = other_indices

        # Activation function
        if activation == "silu":
            activation_fn = nn.SiLU()
        elif activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "gelu":
            activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Frequency-specific branch
        freq_layers = []
        prev_size = len(freq_indices)
        for h_size in hidden_sizes[:2]:
            freq_layers.append(nn.Linear(prev_size, h_size))
            freq_layers.append(activation_fn)
            freq_layers.append(nn.BatchNorm1d(h_size))
            freq_layers.append(nn.Dropout(dropout_rate))
            prev_size = h_size
        self.freq_branch = nn.Sequential(*freq_layers)

        # Other input branch
        other_layers = []
        prev_size = len(other_indices)
        for h_size in hidden_sizes[:2]:
            other_layers.append(nn.Linear(prev_size, h_size))
            other_layers.append(activation_fn)
            other_layers.append(nn.BatchNorm1d(h_size))
            other_layers.append(nn.Dropout(dropout_rate))
            prev_size = h_size
        self.other_branch = nn.Sequential(*other_layers)

        # Combined branch
        combined_layers = []
        prev_size = hidden_sizes[1] * 2
        for h_size in hidden_sizes[2:]:
            combined_layers.append(nn.Linear(prev_size, h_size))
            combined_layers.append(activation_fn)
            combined_layers.append(nn.BatchNorm1d(h_size))
            combined_layers.append(nn.Dropout(dropout_rate))
            prev_size = h_size

        # Final output: 8 values (4 S-parameters × real/imag)
        combined_layers.append(nn.Linear(prev_size, 8))
        self.combined = nn.Sequential(*combined_layers)

    def forward(self, x):
        freq_input = x[:, self.freq_indices]
        other_input = x[:, self.other_indices]
        freq_features = self.freq_branch(freq_input)
        other_features = self.other_branch(other_input)
        combined = torch.cat([freq_features, other_features], dim=1)
        return self.combined(combined)
