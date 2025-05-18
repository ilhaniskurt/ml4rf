import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScientificDerivedLoss(nn.Module):
    def __init__(self, Z0=50):
        super(ScientificDerivedLoss, self).__init__()
        self.Z0 = Z0

        # Normalization factors
        self.norm = {
            "gm": 0.01,  # 10 mS
            "Cpi": 0.1e-12,  # 0.1 pF
            "Cmu": 0.1e-12,  # 0.1 pF
            "fT": 100e9,  # 100 GHz
            "Zin_real": 50,
            "Zin_imag": 100,
            "Zout_real": 50,
            "Zout_imag": 100,
            "Mason_U": 10.0,  # normalized target value
        }

    def s_to_y(self, s):
        # s: (batch, 8) → S11r, S21r, S12r, S22r, S11i, S21i, S12i, S22i
        S11 = torch.complex(s[:, 0], s[:, 4])
        S21 = torch.complex(s[:, 1], s[:, 5])
        S12 = torch.complex(s[:, 2], s[:, 6])
        S22 = torch.complex(s[:, 3], s[:, 7])

        delta = (1 - S11) * (1 - S22) - S12 * S21

        Y11 = ((1 - S11) * (1 + S22) + S12 * S21) / (self.Z0 * delta)
        Y12 = -2 * S12 / (self.Z0 * delta)
        Y21 = -2 * S21 / (self.Z0 * delta)
        Y22 = ((1 + S11) * (1 - S22) + S12 * S21) / (self.Z0 * delta)

        return S11, S12, S21, S22, Y11, Y12, Y21, Y22

    def compute_physical_params(self, S11, S12, S21, S22, Y11, Y12, Y21, Y22, freq):
        omega = 2 * np.pi * freq
        omega = omega.to(dtype=Y11.dtype, device=Y11.device)

        Zin = 1 / Y11
        Zout = 1 / Y22

        gm = Y21.real
        Cpi = Y11.imag / omega
        Cmu = Y12.imag / omega
        fT = gm / (2 * np.pi * (Cpi + Cmu + 1e-15))

        # Mason's Gain Factor U
        mag_S21_sq = torch.abs(S21) ** 2
        mag_S12_sq = torch.abs(S12) ** 2
        denom = (1 - torch.abs(S11) ** 2) * (1 - torch.abs(S22) ** 2) + 1e-12
        mason_U = (mag_S21_sq - mag_S12_sq) / denom

        return {
            "gm": gm,
            "Cpi": Cpi,
            "Cmu": Cmu,
            "fT": fT,
            "Zin_real": Zin.real,
            "Zin_imag": Zin.imag,
            "Zout_real": Zout.real,
            "Zout_imag": Zout.imag,
            "Mason_U": mason_U,
        }

    def forward(self, pred_s, target_s, freq):
        # Convert S-parameters to Y-parameters and extract S
        (
            pred_S11,
            pred_S12,
            pred_S21,
            pred_S22,
            pred_Y11,
            pred_Y12,
            pred_Y21,
            pred_Y22,
        ) = self.s_to_y(pred_s)
        (
            targ_S11,
            targ_S12,
            targ_S21,
            targ_S22,
            targ_Y11,
            targ_Y12,
            targ_Y21,
            targ_Y22,
        ) = self.s_to_y(target_s)

        pred_params = self.compute_physical_params(
            pred_S11,
            pred_S12,
            pred_S21,
            pred_S22,
            pred_Y11,
            pred_Y12,
            pred_Y21,
            pred_Y22,
            freq,
        )
        targ_params = self.compute_physical_params(
            targ_S11,
            targ_S12,
            targ_S21,
            targ_S22,
            targ_Y11,
            targ_Y12,
            targ_Y21,
            targ_Y22,
            freq,
        )

        loss = 0
        for key in self.norm:
            p_pred = pred_params[key] / self.norm[key]
            p_true = targ_params[key] / self.norm[key]

            if torch.is_complex(p_pred):
                loss += F.mse_loss(p_pred.real, p_true.real)
                loss += F.mse_loss(p_pred.imag, p_true.imag)
            else:
                loss += F.mse_loss(p_pred, p_true)

        return loss / len(self.norm)
