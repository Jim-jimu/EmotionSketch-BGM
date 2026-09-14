#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "Experiment" / "core_code"
DIFFBGM_PKG = ROOT / "Experiment" / "code_references" / "Diff-BGM" / "diffbgm"
OUT_PATH = ROOT / "Experiment" / "analysis" / "PARAMETER_EFFICIENCY_2026-05-09.json"


def build_baseline_model(device: torch.device):
    from models.model_sdf import diffbgm_SDF
    from params.params_sdf_chd8bar import params
    from stable_diffusion.latent_diffusion import LatentDiffusion
    from stable_diffusion.model.unet import UNetModel

    unet = UNetModel(
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        channels=params.channels,
        attention_levels=params.attention_levels,
        n_res_blocks=params.n_res_blocks,
        channel_multipliers=params.channel_multipliers,
        n_heads=params.n_heads,
        tf_layers=params.tf_layers,
        d_cond=params.d_cond,
    )
    ldm = LatentDiffusion(
        linear_start=params.linear_start,
        linear_end=params.linear_end,
        n_steps=params.n_steps,
        latent_scaling_factor=params.latent_scaling_factor,
        autoencoder=None,
        unet_model=unet,
    )
    return diffbgm_SDF(
        ldm,
        cond_type=params.cond_type,
        concat_ratio=params.concat_ratio if hasattr(params, "concat_ratio") else 1 / 8,
    ).to(device)


def count_parameters(module) -> int:
    return int(sum(param.numel() for param in module.parameters()))


def main() -> None:
    os.chdir(ROOT)
    sys.path.insert(0, str(DIFFBGM_PKG))
    sys.path.insert(0, str(CORE))
    from emotionsketch.model import EmotionSketchAdapter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = build_baseline_model(device)
    adapter = EmotionSketchAdapter().to(device)
    baseline_params = count_parameters(baseline)
    adapter_params = count_parameters(adapter)
    result = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "PARAMETER_EFFICIENCY_OK",
        "baseline_params": baseline_params,
        "adapter_params": adapter_params,
        "adapter_percent_of_baseline": adapter_params / baseline_params * 100.0,
        "adapter_to_baseline_ratio": adapter_params / baseline_params,
        "method_note": "Parameter count compares trainable EmotionSketch adapter size to the frozen Diff-BGM SDF model skeleton.",
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PARAMETER_EFFICIENCY_WRITTEN", OUT_PATH.relative_to(ROOT))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
