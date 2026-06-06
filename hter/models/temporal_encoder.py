import torch
from torch import nn


class TemporalResidualEncoder(nn.Module):
    """Lightweight temporal refinement for pre-extracted modality sequences."""

    def __init__(self, dim, enabled=False, kernel_size=3, dropout=0.1, scale_init=0.01):
        super().__init__()
        self.enabled = enabled
        if not enabled:
            return

        self.temporal_block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x):
        if not self.enabled:
            return x
        residual = self.temporal_block(x.transpose(1, 2)).transpose(1, 2)
        return x + self.scale * residual
