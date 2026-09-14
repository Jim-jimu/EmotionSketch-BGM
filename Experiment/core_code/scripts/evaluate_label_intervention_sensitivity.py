#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "Experiment" / "core_code"
DIFFBGM_PKG = ROOT / "Experiment" / "code_references" / "Diff-BGM" / "diffbgm"
BASELINE_CKPT = CORE / "checkpoints" / "baseline_1000steps" / "step_001000.pt"
EMOPIA_PRIOR = ROOT / "Experiment" / "datasets" / "derived" / "emopia_emotion_prior.json"
OUT_PATH = ROOT / "Experiment" / "analysis" / "LABEL_INTERVENTION_SENSITIVITY_2026-05-09.json"


ADAPTERS = {
    "full": CORE / "checkpoints" / "emotionsketch_adapter_emopia_prior_100steps" / "emotionsketch_adapter_emopia_prior_100steps.pt",
    "sketch_only": CORE
    / "checkpoints"
    / "emotionsketch_adapter_emopia_prior_sketch_only_100steps"
    / "emotionsketch_adapter_emopia_prior_sketch_only_100steps.pt",
    "label_only": CORE
    / "checkpoints"
    / "emotionsketch_adapter_emopia_prior_label_only_100steps"
    / "emotionsketch_adapter_emopia_prior_label_only_100steps.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure adapter sensitivity to forced Q1-Q4 label interventions.")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--skip-eps", action="store_true", help="Only measure adapted-condition sensitivity.")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--out", default=str(OUT_PATH))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=False) for tensor in batch)


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
    model = diffbgm_SDF(
        ldm,
        cond_type=params.cond_type,
        concat_ratio=params.concat_ratio if hasattr(params, "concat_ratio") else 1 / 8,
    ).to(device)
    checkpoint = torch.load(BASELINE_CKPT, map_location=device)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_adapter(path: Path, device: torch.device):
    from emotionsketch.model import EmotionSketchAdapter

    adapter = EmotionSketchAdapter().to(device)
    payload = torch.load(path, map_location=device)
    adapter.load_state_dict(payload["adapter"], strict=True)
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad_(False)
    return adapter


def pairwise_rmse(tensors: list[torch.Tensor]) -> list[float]:
    values = []
    for i, j in combinations(range(len(tensors)), 2):
        diff = tensors[i].float() - tensors[j].float()
        values.append(float(diff.pow(2).mean().sqrt().detach().cpu().item()))
    return values


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "n": int(arr.size),
    }


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(DIFFBGM_PKG))
    sys.path.insert(0, str(CORE))
    set_seed(args.seed)

    from data.dataloader import get_train_val_dataloaders
    from emotionsketch.data import EmotionLabelProvider, build_emotion_sketch_from_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _train_dl, val_dl = get_train_val_dataloaders(
        args.batch_size,
        num_workers=0,
        pin_memory=False,
        use_track=[0, 1, 2],
    )
    base_model = None if args.skip_eps else build_baseline_model(device)
    adapters = {name: load_adapter(path, device) for name, path in ADAPTERS.items()}
    label_provider = EmotionLabelProvider("emopia_prior", EMOPIA_PRIOR, device=device)
    mode_flags = {
        "full": {"use_sketch": True, "use_label": True},
        "sketch_only": {"use_sketch": True, "use_label": False},
        "label_only": {"use_sketch": False, "use_label": True},
    }
    cond_values = {name: [] for name in adapters}
    eps_values = {name: [] for name in adapters}
    seen = 0
    for batch in val_dl:
        if seen >= args.samples:
            break
        prmat2c, _pnotree, chord, prmat, visual, caption, shot_cnt = to_device(batch, device)
        current = min(prmat2c.shape[0], args.samples - seen)
        prmat2c = prmat2c[:current]
        chord = chord[:current]
        prmat = prmat[:current]
        visual = visual[:current]
        caption = caption[:current]
        shot_cnt = shot_cnt[:current]
        sketch, _labels = build_emotion_sketch_from_batch(prmat, chord, visual, caption, shot_cnt, label_provider)
        if base_model is not None:
            t = torch.full((current,), args.timestep, device=device, dtype=torch.long)
            noise = torch.randn_like(prmat2c)
            xt = base_model.ldm.q_sample(prmat2c, t, eps=noise)
        else:
            t = None
            xt = None
        with torch.no_grad():
            for name, adapter in adapters.items():
                conds = []
                eps_preds = []
                for label_id in range(4):
                    forced = torch.full((current,), label_id, dtype=torch.long, device=device)
                    cond = adapter(visual, sketch, forced, **mode_flags[name])
                    conds.append(cond)
                    if base_model is not None:
                        eps_preds.append(base_model.ldm.eps_model(xt, t, cond))
                cond_values[name].extend(pairwise_rmse(conds))
                if eps_preds:
                    eps_values[name].extend(pairwise_rmse(eps_preds))
        seen += current

    result = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "LABEL_INTERVENTION_SENSITIVITY_OK",
        "samples": seen,
        "batch_size": args.batch_size,
        "timestep": args.timestep,
        "skip_eps": args.skip_eps,
        "label_mapping": {"0": "Q1", "1": "Q2", "2": "Q3", "3": "Q4"},
        "methods": {
            name: {
                "condition_pairwise_rmse": summarize(cond_values[name]),
                "eps_prediction_pairwise_rmse": summarize(eps_values[name]) if not args.skip_eps else None,
                "adapter_checkpoint": str(ADAPTERS[name].relative_to(ROOT)),
            }
            for name in adapters
        },
        "method_note": (
            "For each validation sample, force Q1-Q4 labels while keeping visual/sketch/noise fixed. "
            "Positive pairwise RMSE means the adapter label path changes the adapted condition and/or UNet "
            "denoising prediction. This is controllability-path evidence, not proof of perceived emotion quality."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("LABEL_INTERVENTION_SENSITIVITY_WRITTEN", out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
