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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded Diff-BGM baseline training run.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-steps", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(device: torch.device):
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
    return model, params


def next_batch(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def to_device(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=False) for tensor in batch)


def open_tensorboard(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
    except Exception as exc:  # noqa: BLE001 - tensorboard is optional for this runner
        print(f"TENSORBOARD_DISABLED {type(exc).__name__}: {exc}")
        return None


def save_checkpoint(path: Path, model, optimizer, params, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "summary": summary,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "params": dict(params),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(DIFFBGM_PKG))
    (ROOT / "result" / "symmv").mkdir(parents=True, exist_ok=True)
    (ROOT / "demo").mkdir(exist_ok=True)
    set_seed(args.seed)

    from data.dataloader import get_train_val_dataloaders

    started_at = datetime.now(timezone.utc).replace(microsecond=0)
    run_name = args.run_name or f"baseline_bounded_{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ROOT / "Experiment" / "core_code" / "runs" / run_name
    ckpt_dir = ROOT / "Experiment" / "core_code" / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"
    writer = open_tensorboard(run_dir / "tensorboard")

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

    model, params = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("MODEL_PARAM_COUNT", sum(p.numel() for p in model.parameters()), flush=True)

    train_iter = iter(train_dl)
    val_iter = iter(val_dl)
    train_losses: list[float] = []
    val_losses: list[float] = []
    fieldnames = ["step", "phase", "loss", "grad_norm", "lr", "cuda_max_memory_mb"]
    with csv_path.open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
        csv_writer.writeheader()

        for step in range(1, args.steps + 1):
            model.train()
            batch, train_iter = next_batch(train_iter, train_dl)
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.get_loss_dict(batch, step=step)["loss"]
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            grad_value = float(grad_norm.detach().cpu().item())
            train_losses.append(loss_value)
            max_memory_mb = (
                round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
                if torch.cuda.is_available()
                else 0.0
            )
            row = {
                "step": step,
                "phase": "train",
                "loss": loss_value,
                "grad_norm": grad_value,
                "lr": args.lr,
                "cuda_max_memory_mb": max_memory_mb,
            }
            csv_writer.writerow(row)
            handle.flush()
            if writer is not None:
                writer.add_scalar("loss/train", loss_value, step)
                writer.add_scalar("grad_norm/train", grad_value, step)
                writer.add_scalar("cuda/max_memory_mb", max_memory_mb, step)

            if step == 1 or step % 10 == 0:
                print(
                    f"TRAIN_STEP {step}/{args.steps} loss={loss_value:.6f} "
                    f"grad_norm={grad_value:.6f} max_mem_mb={max_memory_mb:.2f}",
                    flush=True,
                )

            if step % args.val_every == 0 or step == args.steps:
                model.eval()
                current_val_losses = []
                with torch.no_grad():
                    for val_index in range(1, args.val_steps + 1):
                        batch, val_iter = next_batch(val_iter, val_dl)
                        batch = to_device(batch, device)
                        val_loss = model.get_loss_dict(batch, step=step)["loss"]
                        val_value = float(val_loss.detach().cpu().item())
                        current_val_losses.append(val_value)
                        val_losses.append(val_value)
                        csv_writer.writerow(
                            {
                                "step": step,
                                "phase": f"val_{val_index}",
                                "loss": val_value,
                                "grad_norm": "",
                                "lr": args.lr,
                                "cuda_max_memory_mb": (
                                    round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
                                    if torch.cuda.is_available()
                                    else 0.0
                                ),
                            }
                        )
                    handle.flush()
                val_mean = float(np.mean(current_val_losses))
                if writer is not None:
                    writer.add_scalar("loss/val_mean", val_mean, step)
                print(
                    f"VAL_AT_STEP {step} mean_loss={val_mean:.6f} "
                    f"n={len(current_val_losses)}",
                    flush=True,
                )

            if step % args.ckpt_every == 0 or step == args.steps:
                summary = {
                    "run_name": run_name,
                    "created_at": started_at.isoformat(),
                    "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "steps_requested": args.steps,
                    "steps_completed": step,
                    "val_every": args.val_every,
                    "val_steps": args.val_steps,
                    "ckpt_every": args.ckpt_every,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "seed": args.seed,
                    "train_loss_first": train_losses[0],
                    "train_loss_last": train_losses[-1],
                    "train_loss_mean": float(np.mean(train_losses)),
                    "val_loss_mean": float(np.mean(val_losses)) if val_losses else None,
                    "cuda_max_memory_mb": (
                        round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                    "status": "running" if step < args.steps else "BASELINE_BOUNDED_TRAIN_OK",
                }
                ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
                save_checkpoint(ckpt_path, model, optimizer, params, summary)
                summary["checkpoint"] = str(ckpt_path.relative_to(ROOT))
                summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
                print("CHECKPOINT", ckpt_path.relative_to(ROOT), flush=True)

    if writer is not None:
        writer.flush()
        writer.close()
    print("SUMMARY_PATH", summary_path.relative_to(ROOT), flush=True)
    print("CSV_PATH", csv_path.relative_to(ROOT), flush=True)
    print("BASELINE_BOUNDED_TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
