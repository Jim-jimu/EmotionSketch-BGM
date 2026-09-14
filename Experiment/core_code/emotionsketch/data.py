from __future__ import annotations

import json
from pathlib import Path

import torch


def _safe_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.linalg.vector_norm(x.float(), dim=dim, keepdim=True)


def _batch_style_features(prmat: torch.Tensor) -> torch.Tensor:
    """Return compact symbolic style features used for EMOPIA-prior labeling."""
    batch_size = prmat.shape[0]
    prmat = prmat.float()
    beat_roll = prmat.reshape(batch_size, 32, 4, 128).clamp(min=0)
    density = beat_roll.mean(dim=(1, 2, 3))
    pitch_index = torch.linspace(0, 1, 128, device=prmat.device).view(1, 1, 1, 128)
    pitch_mass = beat_roll.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    pitch_mean = (beat_roll * pitch_index).sum(dim=(1, 2, 3)) / pitch_mass
    high_ratio = beat_roll[:, :, :, 64:].sum(dim=(1, 2, 3)) / pitch_mass
    low_ratio = beat_roll[:, :, :, :48].sum(dim=(1, 2, 3)) / pitch_mass
    pitch_active = beat_roll.sum(dim=(1, 2)).clamp(min=0)
    pitch_prob = pitch_active / pitch_active.sum(dim=1, keepdim=True).clamp_min(1e-6)
    pitch_axis = torch.linspace(0, 1, 128, device=prmat.device).view(1, 128)
    pitch_var = ((pitch_axis - pitch_mean.view(-1, 1)) ** 2 * pitch_prob).sum(dim=1)
    pitch_spread = pitch_var.clamp_min(0).sqrt()
    return torch.stack([density, pitch_mean, high_ratio, low_ratio, pitch_spread], dim=1)


class EmotionLabelProvider:
    """Label provider for EmotionSketch experiments.

    Modes:
    - proxy: median-threshold labels from current batch statistics.
    - emopia_prior: nearest EMOPIA quadrant centroid in compact symbolic feature space.
    """

    def __init__(
        self,
        mode: str = "proxy",
        prior_path: str | Path | None = None,
        device: torch.device | None = None,
    ):
        self.mode = mode
        self.prior_path = Path(prior_path) if prior_path else None
        self.device = device
        self._centroids: torch.Tensor | None = None
        if self.mode not in {"proxy", "emopia_prior"}:
            raise ValueError(f"Unsupported emotion label mode: {self.mode}")
        if self.mode == "emopia_prior":
            if self.prior_path is None:
                raise ValueError("prior_path is required for emopia_prior label mode")
            payload = json.loads(self.prior_path.read_text(encoding="utf-8"))
            centroids = payload["centroids"]
            ordered = [centroids[f"Q{i}"] for i in range(1, 5)]
            self._centroids = torch.tensor(ordered, dtype=torch.float32, device=device)

    def labels_from_batch(
        self,
        prmat: torch.Tensor,
        arousal_proxy: torch.Tensor,
        valence_proxy: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "proxy":
            song_arousal = arousal_proxy.mean(dim=(1, 2))
            song_valence = valence_proxy.mean(dim=(1, 2))
            arousal_bit = (song_arousal >= song_arousal.median()).long()
            valence_bit = (song_valence >= song_valence.median()).long()
            return arousal_bit * 2 + valence_bit
        if self._centroids is None:
            raise RuntimeError("EMOPIA prior centroids are not loaded")
        centroids = self._centroids.to(prmat.device)
        features = _batch_style_features(prmat)
        distances = torch.cdist(features.float(), centroids.float(), p=2)
        return distances.argmin(dim=1).long()


def build_emotion_sketch_from_batch(
    prmat: torch.Tensor,
    chord: torch.Tensor,
    visual: torch.Tensor,
    caption: torch.Tensor,
    shot_cnt: torch.Tensor,
    label_provider: EmotionLabelProvider | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic EmotionSketch for adapter experiments.

    The sketch is segment-aligned and derived from existing Diff-BGM batch
    tensors. Labels can come from simple proxy statistics or from an EMOPIA
    symbolic-prior label provider. EMOPIA-prior labels are traceable pseudo
    targets, not ground-truth BGM909 affect annotations.
    """
    batch_size = prmat.shape[0]
    prmat = prmat.float()
    chord = chord.float()
    visual = visual.float()
    caption = caption.float()
    beat_roll = prmat.reshape(batch_size, 32, 4, 128)
    activity = beat_roll.clamp(min=0)
    density = activity.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
    pitch_index = torch.linspace(0, 1, 128, device=prmat.device).view(1, 1, 1, 128)
    pitch_mass = activity.sum(dim=(2, 3), keepdim=False).unsqueeze(-1).clamp_min(1e-6)
    pitch_mean = (activity * pitch_index).sum(dim=(2, 3), keepdim=False).unsqueeze(-1) / pitch_mass
    high_ratio = activity[:, :, :, 64:].mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
    low_ratio = activity[:, :, :, :48].mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
    pitch_spread = activity.std(dim=(2, 3), keepdim=False).unsqueeze(-1)
    chord_root = chord[:, :, :12].sum(dim=-1, keepdim=True)
    chord_chroma = chord[:, :, 12:24].sum(dim=-1, keepdim=True)
    chord_bass = chord[:, :, 24:36].sum(dim=-1, keepdim=True)
    visual_norm = _safe_norm(visual).clamp(max=100.0) / 100.0
    caption_norm = _safe_norm(caption).clamp(max=100.0) / 100.0
    shot_feature = shot_cnt.float().view(batch_size, 1, 1).expand(-1, 32, -1).clamp(max=16.0) / 16.0
    arousal_proxy = (density * 6.0 + visual_norm).clamp(0, 1)
    valence_proxy = (pitch_mean + high_ratio * 4.0 - low_ratio * 2.0).clamp(0, 1)
    sketch = torch.cat(
        [
            density,
            pitch_mean,
            high_ratio,
            low_ratio,
            pitch_spread,
            chord_root,
            chord_chroma,
            chord_bass,
            visual_norm,
            caption_norm,
            shot_feature,
            arousal_proxy,
            valence_proxy,
            density - low_ratio,
            high_ratio - low_ratio,
            pitch_mean * density,
        ],
        dim=-1,
    )
    provider = label_provider or EmotionLabelProvider(mode="proxy")
    labels = provider.labels_from_batch(prmat, arousal_proxy, valence_proxy)
    return sketch, labels
