#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
DIFFBGM_PKG = ROOT / "Experiment" / "code_references" / "Diff-BGM" / "diffbgm"
BASELINE_CKPT = ROOT / "Experiment" / "core_code" / "checkpoints" / "baseline_1000steps" / "step_001000.pt"
EMOPIA_PRIOR = ROOT / "Experiment" / "datasets" / "derived" / "emopia_emotion_prior.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EmotionSketch adapter prototype.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--val-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--baseline-ckpt", default=str(BASELINE_CKPT))
    parser.add_argument("--label-mode", choices=["proxy", "emopia_prior", "emopia_prior_shuffled"], default="proxy")
    parser.add_argument("--adapter-mode", choices=["full", "sketch_only", "label_only"], default="full")
    parser.add_argument("--emopia-prior", default=str(EMOPIA_PRIOR))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def next_batch(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def to_device(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=False) for tensor in batch)


def build_baseline_model(device: torch.device, ckpt_path: Path):
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
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, params


def loss_with_adapter(base_model, adapter, batch, label_provider, adapter_mode: str, shuffle_labels: bool = False):
    prmat2c, _pnotree, chord, prmat, visual, caption, shot_cnt = batch
    from emotionsketch.data import build_emotion_sketch_from_batch

    sketch, labels = build_emotion_sketch_from_batch(prmat, chord, visual, caption, shot_cnt, label_provider)
    if shuffle_labels and labels.numel() > 1:
        labels = labels[torch.randperm(labels.numel(), device=labels.device)]
    adapted_visual = adapter(
        visual,
        sketch,
        labels,
        use_sketch=adapter_mode != "label_only",
        use_label=adapter_mode != "sketch_only",
    )
    loss = base_model.ldm.loss(prmat2c, adapted_visual)
    return loss, sketch, labels


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(DIFFBGM_PKG))
    sys.path.insert(0, str(ROOT / "Experiment" / "core_code"))
    (ROOT / "result" / "symmv").mkdir(parents=True, exist_ok=True)
    (ROOT / "demo").mkdir(exist_ok=True)
    set_seed(args.seed)

    from data.dataloader import get_train_val_dataloaders
    from emotionsketch.data import EmotionLabelProvider
    from emotionsketch.model import EmotionSketchAdapter

    started_at = datetime.now(timezone.utc).replace(microsecond=0)
    run_name = args.run_name or f"emotionsketch_adapter_{args.label_mode}_{args.adapter_mode}_{args.steps}steps"
    run_dir = ROOT / "Experiment" / "core_code" / "runs" / run_name
    ckpt_dir = ROOT / "Experiment" / "core_code" / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("RUN_NAME", run_name, flush=True)
    print("DEVICE", device, flush=True)
    if torch.cuda.is_available():
        print("CUDA_DEVICE", torch.cuda.get_device_name(0), flush=True)
    train_dl, val_dl = get_train_val_dataloaders(
        args.batch_size,
        num_workers=0,
        pin_memory=False,
        use_track=[0, 1, 2],
    )
    print("TRAIN_BATCHES", len(train_dl), flush=True)
    print("VAL_BATCHES", len(val_dl), flush=True)
    base_model, params = build_baseline_model(device, Path(args.baseline_ckpt))
    provider_mode = "emopia_prior" if args.label_mode == "emopia_prior_shuffled" else args.label_mode
    label_provider = EmotionLabelProvider(provider_mode, args.emopia_prior, device=device)
    shuffle_labels = args.label_mode == "emopia_prior_shuffled"
    adapter = EmotionSketchAdapter().to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    print("LABEL_MODE", args.label_mode, flush=True)
    print("ADAPTER_MODE", args.adapter_mode, flush=True)
    print("ADAPTER_PARAM_COUNT", sum(p.numel() for p in adapter.parameters()), flush=True)
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_iter = iter(train_dl)
    val_iter = iter(val_dl)
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "phase", "loss", "grad_norm", "label_hist", "cuda_max_memory_mb"])
        writer.writeheader()
        for step in range(1, args.steps + 1):
            adapter.train()
            batch, train_iter = next_batch(train_iter, train_dl)
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _sketch, labels = loss_with_adapter(
                base_model,
                adapter,
                batch,
                label_provider,
                args.adapter_mode,
                shuffle_labels,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=5.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            grad_value = float(grad_norm.detach().cpu().item())
            train_losses.append(loss_value)
            label_hist = torch.bincount(labels.detach().cpu(), minlength=4).tolist()
            max_mem = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2) if torch.cuda.is_available() else 0.0
            writer.writerow(
                {
                    "step": step,
                    "phase": "train",
                    "loss": loss_value,
                    "grad_norm": grad_value,
                    "label_hist": json.dumps(label_hist),
                    "cuda_max_memory_mb": max_mem,
                }
            )
            handle.flush()
            if step == 1 or step % 10 == 0:
                print(
                    f"TRAIN_STEP {step}/{args.steps} loss={loss_value:.6f} "
                    f"grad_norm={grad_value:.6f} label_hist={label_hist} max_mem_mb={max_mem:.2f}",
                    flush=True,
                )
            if step % args.val_every == 0 or step == args.steps:
                adapter.eval()
                current_val = []
                with torch.no_grad():
                    for val_index in range(1, args.val_steps + 1):
                        batch, val_iter = next_batch(val_iter, val_dl)
                        batch = to_device(batch, device)
                        val_loss, _sketch, val_labels = loss_with_adapter(
                            base_model,
                            adapter,
                            batch,
                            label_provider,
                            args.adapter_mode,
                            shuffle_labels,
                        )
                        val_value = float(val_loss.detach().cpu().item())
                        current_val.append(val_value)
                        val_losses.append(val_value)
                        writer.writerow(
                            {
                                "step": step,
                                "phase": f"val_{val_index}",
                                "loss": val_value,
                                "grad_norm": "",
                                "label_hist": json.dumps(torch.bincount(val_labels.detach().cpu(), minlength=4).tolist()),
                                "cuda_max_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
                                if torch.cuda.is_available()
                                else 0.0,
                            }
                        )
                    handle.flush()
                print(f"VAL_AT_STEP {step} mean_loss={float(np.mean(current_val)):.6f} n={len(current_val)}", flush=True)
    summary = {
        "run_name": run_name,
        "created_at": started_at.isoformat(),
        "steps": args.steps,
        "val_every": args.val_every,
        "val_steps": args.val_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "baseline_ckpt": str(Path(args.baseline_ckpt).relative_to(ROOT)),
        "label_mode": args.label_mode,
        "adapter_mode": args.adapter_mode,
        "emopia_prior": str(Path(args.emopia_prior).relative_to(ROOT)) if args.label_mode.startswith("emopia_prior") else None,
        "shuffle_labels": shuffle_labels,
        "train_loss_first": train_losses[0],
        "train_loss_last": train_losses[-1],
        "train_loss_mean": float(np.mean(train_losses)),
        "val_loss_mean": float(np.mean(val_losses)) if val_losses else None,
        "cuda_max_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2) if torch.cuda.is_available() else 0.0,
        "status": "EMOTIONSKETCH_ADAPTER_TRAIN_OK",
    }
    ckpt_path = ckpt_dir / f"{run_name}.pt"
    torch.save(
        {
            "summary": summary,
            "adapter": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "params": dict(params),
        },
        ckpt_path,
    )
    summary["checkpoint"] = str(ckpt_path.relative_to(ROOT))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("CHECKPOINT", ckpt_path.relative_to(ROOT), flush=True)
    print("EMOTIONSKETCH_ADAPTER_TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
