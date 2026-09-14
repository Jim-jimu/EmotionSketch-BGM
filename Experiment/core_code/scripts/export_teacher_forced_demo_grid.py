#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "Experiment" / "core_code"
DIFFBGM_PKG = ROOT / "Experiment" / "code_references" / "Diff-BGM" / "diffbgm"
BASELINE_CKPT = CORE / "checkpoints" / "baseline_1000steps" / "step_001000.pt"
EMOPIA_PRIOR = ROOT / "Experiment" / "datasets" / "derived" / "emopia_emotion_prior.json"
OUT_DIR = ROOT / "Experiment" / "outputs" / "demo_midi_grid_2026-05-09"
FULL_ADAPTER = CORE / "checkpoints" / "emotionsketch_adapter_emopia_prior_100steps" / "emotionsketch_adapter_emopia_prior_100steps.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search teacher-forced demo MIDI export settings.")
    parser.add_argument("--clips", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timesteps", default="100,500,900")
    parser.add_argument("--thresholds", default="density,0.2,0.4")
    parser.add_argument("--label-scales", default="1,4")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser.parse_args()


def parse_list(text: str, cast):
    return [cast(x) for x in text.split(",") if x]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=False) for tensor in batch)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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


def load_adapter(device: torch.device):
    from emotionsketch.model import EmotionSketchAdapter

    adapter = EmotionSketchAdapter().to(device)
    payload = torch.load(FULL_ADAPTER, map_location=device)
    adapter.load_state_dict(payload["adapter"], strict=True)
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad_(False)
    return adapter


def denoise_pred_x0(base_model, prmat2c, cond, timestep: int, noise: torch.Tensor) -> torch.Tensor:
    batch_size = prmat2c.shape[0]
    t = torch.full((batch_size,), timestep, device=prmat2c.device, dtype=torch.long)
    with torch.no_grad():
        xt = base_model.ldm.q_sample(prmat2c, t, eps=noise)
        eps_theta = base_model.ldm.eps_model(xt, t, cond)
        alpha_bar = base_model.ldm.alpha_bar[t].view(-1, 1, 1, 1)
        return (xt - (1.0 - alpha_bar).sqrt() * eps_theta) / alpha_bar.sqrt()


def binarize(pred_x0: torch.Tensor, target: torch.Tensor, threshold_spec: str) -> torch.Tensor:
    pred = pred_x0.detach().float().clamp(0, 1)
    if threshold_spec == "density":
        output = torch.zeros_like(pred)
        flat_pred = pred.view(pred.shape[0], -1)
        flat_target = (target.detach().float() > 0.5).view(target.shape[0], -1)
        for index in range(pred.shape[0]):
            keep = int(flat_target[index].sum().item())
            if keep <= 0:
                continue
            keep = min(keep, flat_pred.shape[1])
            threshold = torch.topk(flat_pred[index], keep).values.min()
            output[index] = (pred[index] >= threshold).float()
        return output
    return (pred > float(threshold_spec)).float()


def write_prmat2c(path: Path, tensor: torch.Tensor) -> None:
    from utils import prmat2c_to_midi_file

    path.parent.mkdir(parents=True, exist_ok=True)
    prmat2c_to_midi_file(tensor.detach().cpu().numpy(), str(path))


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(DIFFBGM_PKG))
    sys.path.insert(0, str(CORE))
    set_seed(args.seed)

    from data.dataloader import get_train_val_dataloaders
    from emotionsketch.data import EmotionLabelProvider, build_emotion_sketch_from_batch

    timesteps = parse_list(args.timesteps, int)
    thresholds = [x for x in args.thresholds.split(",") if x]
    label_scales = parse_list(args.label_scales, float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _train_dl, val_dl = get_train_val_dataloaders(
        args.batch_size,
        num_workers=0,
        pin_memory=False,
        use_track=[0, 1, 2],
    )
    base_model = build_baseline_model(device)
    adapter = load_adapter(device)
    label_provider = EmotionLabelProvider("emopia_prior", EMOPIA_PRIOR, device=device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    seen = 0
    for batch in val_dl:
        if seen >= args.clips:
            break
        prmat2c, _pnotree, chord, prmat, visual, caption, shot_cnt = to_device(batch, device)
        for idx in range(prmat2c.shape[0]):
            if seen >= args.clips:
                break
            clip = f"clip_{seen:02d}"
            one_prmat2c = prmat2c[idx : idx + 1]
            one_chord = chord[idx : idx + 1]
            one_prmat = prmat[idx : idx + 1]
            one_visual = visual[idx : idx + 1]
            one_caption = caption[idx : idx + 1]
            one_shot = shot_cnt[idx : idx + 1]
            sketch, pseudo_label = build_emotion_sketch_from_batch(
                one_prmat,
                one_chord,
                one_visual,
                one_caption,
                one_shot,
                label_provider,
            )
            target_path = out_dir / f"{clip}_target.mid"
            write_prmat2c(target_path, one_prmat2c)
            manifest.append(
                {
                    "clip": clip,
                    "method": "target",
                    "path": display_path(target_path),
                    "label": "target",
                    "timestep": None,
                    "threshold": "target",
                    "label_scale": None,
                    "pseudo_label": int(pseudo_label.item()),
                }
            )
            for timestep in timesteps:
                noise = torch.randn_like(one_prmat2c)
                baseline_x0 = denoise_pred_x0(base_model, one_prmat2c, one_visual, timestep, noise)
                for threshold in thresholds:
                    baseline_binary = binarize(baseline_x0, one_prmat2c, threshold)
                    path = out_dir / f"{clip}_baseline_t{timestep}_th{threshold}.mid"
                    write_prmat2c(path, baseline_binary)
                    manifest.append(
                        {
                            "clip": clip,
                            "method": "baseline",
                            "path": display_path(path),
                            "label": "baseline",
                            "timestep": timestep,
                            "threshold": threshold,
                            "label_scale": None,
                            "pseudo_label": int(pseudo_label.item()),
                        }
                    )
                for label_scale in label_scales:
                    for label_id in range(4):
                        forced_label = torch.full((1,), label_id, dtype=torch.long, device=device)
                        with torch.no_grad():
                            cond = adapter(
                                one_visual,
                                sketch,
                                forced_label,
                                use_sketch=True,
                                use_label=True,
                                label_scale=label_scale,
                            )
                        pred_x0 = denoise_pred_x0(base_model, one_prmat2c, cond, timestep, noise)
                        for threshold in thresholds:
                            pred_binary = binarize(pred_x0, one_prmat2c, threshold)
                            path = out_dir / f"{clip}_full_Q{label_id + 1}_scale{label_scale:g}_t{timestep}_th{threshold}.mid"
                            write_prmat2c(path, pred_binary)
                            manifest.append(
                                {
                                    "clip": clip,
                                    "method": "full",
                                    "path": display_path(path),
                                    "label": f"Q{label_id + 1}",
                                    "label_id": label_id,
                                    "timestep": timestep,
                                    "threshold": threshold,
                                    "label_scale": label_scale,
                                    "pseudo_label": int(pseudo_label.item()),
                                }
                            )
            seen += 1
    payload = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "DEMO_MIDI_GRID_EXPORTED",
        "clips": seen,
        "timesteps": timesteps,
        "thresholds": thresholds,
        "label_scales": label_scales,
        "entries": manifest,
        "method_note": "Teacher-forced demo grid: timestep, threshold, and label_scale sweep. Not full DDPM sampling.",
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DEMO_MIDI_GRID_EXPORTED", display_path(manifest_path), "COUNT", len(manifest))
    print(json.dumps({k: payload[k] for k in ["status", "clips", "timesteps", "thresholds", "label_scales"]}, sort_keys=True))


if __name__ == "__main__":
    main()
