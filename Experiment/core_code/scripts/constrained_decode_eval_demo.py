#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pretty_midi as pm
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "Experiment" / "core_code"
DIFFBGM_PKG = ROOT / "Experiment" / "code_references" / "Diff-BGM" / "diffbgm"
OUT_DIR = ROOT / "Experiment" / "outputs" / "constrained_decode_2026-05-09"
OUT_JSON = ROOT / "Experiment" / "analysis" / "CONSTRAINED_DECODE_V3_2026-05-09.json"
V3_CKPT = CORE / "checkpoints" / "emopia_midi_quadrant_evaluator_v3.pt"
EMOPIA_PRIOR = ROOT / "Experiment" / "datasets" / "derived" / "emopia_emotion_prior.json"

PROFILES = {
    0: {"name": "q1_bright_dense", "density": 0.115, "pitch_min": 48, "pitch_max": 88, "dilate": 1, "velocity": 104, "tempo": 150},
    1: {"name": "q2_dense_wide", "density": 0.145, "pitch_min": 44, "pitch_max": 92, "dilate": 1, "velocity": 86, "tempo": 96},
    2: {"name": "q3_sparse_long", "density": 0.035, "pitch_min": 36, "pitch_max": 76, "dilate": 5, "velocity": 46, "tempo": 64},
    3: {"name": "q4_sparse_narrow", "density": 0.025, "pitch_min": 51, "pitch_max": 59, "dilate": 2, "velocity": 34, "tempo": 104},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label-specific constrained symbolic decoding before MIDI export.")
    parser.add_argument("--clips", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timesteps", default="50,100,200,500,900")
    parser.add_argument("--label-scales", default="1,4,8,16")
    parser.add_argument("--noise-seeds", default="0,1")
    parser.add_argument("--rerank-k", type=int, default=31)
    parser.add_argument("--labels", default="0,1,2,3", help="Comma-separated zero-based labels to run.")
    parser.add_argument("--condition-source", choices=["adapter", "baseline"], default="adapter")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    return parser.parse_args()


def parse_list(text: str, cast):
    return [cast(value) for value in text.split(",") if value != ""]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def to_device(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=False) for tensor in batch)


def load_v3(path: Path, k: int) -> dict:
    payload = torch.load(path, map_location="cpu")
    return {
        "k": k,
        "train_x": payload["train_x"].float().numpy(),
        "train_y": payload["train_y"].long().numpy(),
        "centroids": payload["centroids"].float().numpy(),
        "mean": payload["mean"].float().numpy(),
        "std": np.maximum(payload["std"].float().numpy(), 1e-6),
        "summary": payload.get("summary", {}),
    }


def score_feature(feature: np.ndarray, evaluator: dict) -> dict:
    x = (feature - evaluator["mean"]) / evaluator["std"]
    distances = np.linalg.norm(evaluator["train_x"] - x.reshape(1, -1), axis=1)
    order = np.argsort(distances)[: int(evaluator["k"])]
    scores = np.zeros(4, dtype=np.float64)
    for label, distance in zip(evaluator["train_y"][order], distances[order]):
        scores[int(label)] += 1.0 / max(float(distance), 1e-6)
    probs = scores / scores.sum() if scores.sum() else np.ones(4) / 4.0
    pred = int(scores.argmax())
    centroid_distances = np.linalg.norm(evaluator["centroids"] - x.reshape(1, -1), axis=1)
    centroid_scores = 1.0 / np.maximum(centroid_distances, 1e-6)
    centroid_probs = centroid_scores / centroid_scores.sum()
    centroid_pred = int(centroid_scores.argmax())
    return {
        "pred": pred,
        "pred_label": f"Q{pred + 1}",
        "probs": [round(float(v), 6) for v in probs.tolist()],
        "centroid_pred": centroid_pred,
        "centroid_pred_label": f"Q{centroid_pred + 1}",
        "centroid_probs": [round(float(v), 6) for v in centroid_probs.tolist()],
    }


def constrained_prmat2c(scores: torch.Tensor, label_id: int) -> torch.Tensor:
    profile = PROFILES[label_id]
    x = scores.detach().float().clamp(0, 1)
    constrained = torch.zeros_like(x)
    pitch_count = x.shape[-1]
    pitch_mask = torch.zeros(pitch_count, dtype=torch.bool, device=x.device)
    pitch_min = max(0, int(profile["pitch_min"]))
    pitch_max = min(pitch_count - 1, int(profile["pitch_max"]))
    pitch_mask[pitch_min : pitch_max + 1] = True
    masked = x.clone()
    masked[..., ~pitch_mask] = -1.0
    flat = masked.view(masked.shape[0], -1)
    allowed = int(masked.shape[1] * masked.shape[2] * int(pitch_mask.sum().item()))
    keep = max(8, int(round(allowed * float(profile["density"]))))
    keep = min(keep, flat.shape[1])
    for index in range(flat.shape[0]):
        values, indices = torch.topk(flat[index], keep)
        valid = values > 0
        if valid.any():
            out = constrained[index].view(-1)
            out[indices[valid]] = 1.0
    dilate = int(profile["dilate"])
    if dilate > 1:
        pad = dilate // 2
        constrained = F.max_pool2d(constrained, kernel_size=(dilate, 1), stride=1, padding=(pad, 0))
        constrained = constrained[..., : x.shape[-2], :]
    return constrained.clamp(0, 1)


def write_prmat2c_controlled(path: Path, tensor: torch.Tensor, label_id: int) -> None:
    profile = PROFILES[label_id]
    prmat2c = tensor.detach().cpu().numpy()
    midi = pm.PrettyMIDI(initial_tempo=float(profile["tempo"]))
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    instrument = pm.Instrument(program=piano_program)
    n_step = prmat2c.shape[2]
    t = 0.0
    t_bar = int(n_step / 8)
    for bars in prmat2c:
        onset = bars[0]
        sustain = bars[1]
        for step_ind, step in enumerate(onset):
            for key, on in enumerate(step):
                if int(round(float(on))) <= 0:
                    continue
                dur = 1
                while step_ind + dur < n_step:
                    if not (int(round(float(sustain[step_ind + dur, key]))) > 0):
                        break
                    dur += 1
                note = pm.Note(
                    velocity=int(profile["velocity"]),
                    pitch=int(key),
                    start=t + step_ind * 1 / 8,
                    end=min(t + (step_ind + dur) * 1 / 8, t + t_bar),
                )
                instrument.notes.append(note)
        t += t_bar
    midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def summarize(entries: list[dict]) -> dict:
    counter = Counter(entry["pred_label"] for entry in entries)
    centroid_counter = Counter(entry["centroid_pred_label"] for entry in entries)
    total = len(entries)
    hits = sum(1 for entry in entries if int(entry["label_id"]) == int(entry["pred"]))
    centroid_hits = sum(1 for entry in entries if int(entry["label_id"]) == int(entry["centroid_pred"]))
    return {
        "n": total,
        "pred_counts": {f"Q{i + 1}": int(counter.get(f"Q{i + 1}", 0)) for i in range(4)},
        "centroid_pred_counts": {f"Q{i + 1}": int(centroid_counter.get(f"Q{i + 1}", 0)) for i in range(4)},
        "forced_label_agreement": float(hits / total) if total else 0.0,
        "centroid_forced_label_agreement": float(centroid_hits / total) if total else 0.0,
        "mean_target_prob": float(np.mean([entry["target_prob"] for entry in entries])) if entries else None,
    }


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(DIFFBGM_PKG))
    sys.path.insert(0, str(CORE))
    sys.path.insert(0, str(CORE / "scripts"))
    set_seed(args.seed)

    from data.dataloader import get_train_val_dataloaders
    from emotionsketch.data import EmotionLabelProvider, build_emotion_sketch_from_batch
    from export_teacher_forced_demo_grid import build_baseline_model, denoise_pred_x0, load_adapter
    from train_emopia_midi_quadrant_evaluator_v3 import midi_features_v3

    timesteps = parse_list(args.timesteps, int)
    label_scales = parse_list(args.label_scales, float)
    noise_seeds = parse_list(args.noise_seeds, int)
    labels = parse_list(args.labels, int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = load_v3(V3_CKPT, args.rerank_k)
    _train_dl, val_dl = get_train_val_dataloaders(args.batch_size, num_workers=0, pin_memory=False, use_track=[0, 1, 2])
    base_model = build_baseline_model(device)
    adapter = load_adapter(device) if args.condition_source == "adapter" else None
    label_provider = EmotionLabelProvider("emopia_prior", EMOPIA_PRIOR, device=device)
    out_dir = Path(args.out_dir)
    candidates_dir = out_dir / "candidates"
    selected_dir = out_dir / "selected"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    all_candidates = []
    selected = []
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
                one_prmat, one_chord, one_visual, one_caption, one_shot, label_provider
            )
            for label_id in labels:
                label_candidates = []
                forced_label = torch.full((1,), label_id, dtype=torch.long, device=device)
                for label_scale in label_scales:
                    with torch.no_grad():
                        if args.condition_source == "baseline":
                            cond = one_visual
                        else:
                            cond = adapter(
                                one_visual,
                                sketch,
                                forced_label,
                                use_sketch=True,
                                use_label=True,
                                label_scale=label_scale,
                            )
                    for timestep in timesteps:
                        for noise_seed in noise_seeds:
                            generator = torch.Generator(device=device)
                            generator.manual_seed(args.seed + seen * 10000 + label_id * 1000 + noise_seed)
                            noise = torch.randn(one_prmat2c.shape, device=device, generator=generator)
                            pred_x0 = denoise_pred_x0(base_model, one_prmat2c, cond, timestep, noise)
                            constrained = constrained_prmat2c(pred_x0, label_id)
                            path = candidates_dir / (
                                f"{clip}_Q{label_id + 1}_scale{label_scale:g}_t{timestep}_seed{noise_seed}_"
                                f"{PROFILES[label_id]['name']}.mid"
                            )
                            write_prmat2c_controlled(path, constrained, label_id)
                            feature = midi_features_v3(path)
                            if feature is None:
                                continue
                            score = score_feature(feature, evaluator)
                            row = {
                                "clip": clip,
                                "method": "constrained_decode",
                                "path": display_path(path),
                                "label": f"Q{label_id + 1}",
                                "label_id": label_id,
                                "pseudo_label": int(pseudo_label.item()),
                                "label_scale": label_scale,
                                "timestep": timestep,
                                "noise_seed": noise_seed,
                                "profile": PROFILES[label_id]["name"],
                                **score,
                            }
                            row["target_prob"] = 0.5 * float(row["probs"][label_id]) + 0.5 * float(row["centroid_probs"][label_id])
                            row["target_hit"] = int(int(row["pred"]) == label_id)
                            row["centroid_target_hit"] = int(int(row["centroid_pred"]) == label_id)
                            all_candidates.append(row)
                            label_candidates.append(row)
                best = sorted(
                    label_candidates,
                    key=lambda item: (
                        item["target_hit"],
                        item["centroid_target_hit"],
                        item["target_prob"],
                    ),
                    reverse=True,
                )[0]
                source = ROOT / best["path"]
                selected_path = selected_dir / f"{clip}_selected_Q{label_id + 1}.mid"
                selected_path.write_bytes(source.read_bytes())
                best = dict(best)
                best["selected_path"] = display_path(selected_path)
                selected.append(best)
            seen += 1

    result = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "CONSTRAINED_DECODE_V3_DONE",
        "clips": seen,
        "candidate_count": len(all_candidates),
        "selected_count": len(selected),
        "profiles": PROFILES,
        "timesteps": timesteps,
        "label_scales": label_scales,
        "noise_seeds": noise_seeds,
        "labels": labels,
        "condition_source": args.condition_source,
        "candidate_summary": summarize(all_candidates),
        "candidate_by_label": {
            f"Q{label_id + 1}": summarize([entry for entry in all_candidates if int(entry["label_id"]) == label_id])
            for label_id in range(4)
        },
        "selected_summary": summarize(selected),
        "selected_by_label": {
            f"Q{label_id + 1}": summarize([entry for entry in selected if int(entry["label_id"]) == label_id])
            for label_id in range(4)
        },
        "selected": selected,
        "manifest": display_path(out_dir / "manifest.json"),
        "method_note": (
            "Applies label-specific constraints directly to decoded prmat2c tensors and uses label-aware "
            "decode-to-MIDI velocity/tempo at export time. This is not post-hoc editing of an existing MIDI file."
        ),
    }
    manifest = {"status": "CONSTRAINED_DECODE_EXPORTED", "entries": selected}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CONSTRAINED_DECODE_V3_DONE", display_path(Path(args.out_json)), "CANDIDATES", len(all_candidates))
    print(json.dumps({"candidate_summary": result["candidate_summary"], "selected_summary": result["selected_summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
