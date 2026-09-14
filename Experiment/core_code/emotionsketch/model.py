from __future__ import annotations

import torch
from torch import nn


class EmotionSketchAdapter(nn.Module):
    """Residual adapter that preserves Diff-BGM's visual condition dimensionality."""

    def __init__(self, sketch_dim: int = 16, cond_dim: int = 512, num_labels: int = 4):
        super().__init__()
        self.sketch_proj = nn.Sequential(
            nn.LayerNorm(sketch_dim),
            nn.Linear(sketch_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.label_emb = nn.Embedding(num_labels, cond_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, cond_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        visual: torch.Tensor,
        emotion_sketch: torch.Tensor,
        emotion_label: torch.Tensor,
        use_sketch: bool = True,
        use_label: bool = True,
        label_scale: float = 1.0,
    ) -> torch.Tensor:
        if use_sketch:
            sketch_context = self.sketch_proj(emotion_sketch)
        else:
            sketch_context = torch.zeros_like(visual)
        if use_label:
            label_context = self.label_emb(emotion_label).unsqueeze(1).expand(-1, visual.shape[1], -1)
            label_context = label_context * label_scale
        else:
            label_context = torch.zeros_like(visual)
        delta = self.out(sketch_context + label_context)
        gate = torch.sigmoid(self.gate_logit)
        return visual + gate * delta
