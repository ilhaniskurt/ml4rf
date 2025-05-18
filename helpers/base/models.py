import torch
import torch.nn as nn


class FrequencyAwareNetwork(nn.Module):
    def __init__(
        self, freq_idx, other_idx, hidden_sizes, dropout_rate, activation, output_dim=2
    ):
        super().__init__()
        self.freq_idx = freq_idx
        self.other_idx = other_idx

        if activation == "silu":
            activation_fn = nn.SiLU()
        elif activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "gelu":
            activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.freq_net = nn.Sequential(
            nn.Linear(len(freq_idx), hidden_sizes[0]),
            activation_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            activation_fn,
        )

        self.other_net = nn.Sequential(
            nn.Linear(len(other_idx), hidden_sizes[0]),
            activation_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            activation_fn,
        )

        self.combined_net = nn.Sequential(
            nn.Linear(2 * hidden_sizes[1], hidden_sizes[2]),
            activation_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[2], output_dim),
        )

    def forward(self, x):
        freq_in = x[:, self.freq_idx]
        other_in = x[:, self.other_idx]
        freq_out = self.freq_net(freq_in)
        other_out = self.other_net(other_in)
        combined = torch.cat((freq_out, other_out), dim=1)
        return self.combined_net(combined)


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
