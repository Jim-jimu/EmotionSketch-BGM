#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "Experiment" / "core_code"
DIFFBGM_PKG = ROOT / "Experiment" / "code_references" / "Diff-BGM" / "diffbgm"
OUT_DIR = ROOT / "Experiment" / "outputs" / "guided_decode_rerank_2026-05-09"
OUT_JSON = ROOT / "Experiment" / "analysis" / "GUIDED_DECODE_RERANK_V3_2026-05-09.json"
V3_CKPT = CORE / "checkpoints" / "emopia_midi_quadrant_evaluator_v3.pt"
EMOPIA_PRIOR = ROOT / "Experiment" / "datasets" / "derived" / "emopia_emotion_prior.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluator-guided reranking over teacher-forced Diff-BGM candidates.")
    parser.add_argument("--clips", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timesteps", default="50,100,200,500,900")
    parser.add_argument("--thresholds", default="density,0.15,0.25,0.35,0.45")
    parser.add_argument("--label-scales", default="1,4,8,16")
    parser.add_argument("--noise-seeds", default="0,1,2,3")
    parser.add_argument("--rerank-k", type=int, default=31)
    parser.add_argument("--rerank-objective", choices=["knn", "centroid", "hybrid"], default="hybrid")
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


def load_v3_evaluator(path: Path, rerank_k: int) -> dict:
    payload = torch.load(path, map_location="cpu")
    return {
        "k": rerank_k,
        "train_x": payload["train_x"].float().numpy(),
        "train_y": payload["train_y"].long().numpy(),
        "centroids": payload["centroids"].float().numpy(),
        "mean": payload["mean"].float().numpy(),
        "std": np.maximum(payload["std"].float().numpy(), 1e-6),
        "summary": payload.get("summary", {}),
    }


def score_feature_v3(feature: np.ndarray, evaluator: dict) -> dict:
    x = (feature - evaluator["mean"]) / evaluator["std"]
    distances = np.linalg.norm(evaluator["train_x"] - x.reshape(1, -1), axis=1)
    order = np.argsort(distances)[: int(evaluator["k"])]
    scores = np.zeros(4, dtype=np.float64)
    for label, distance in zip(evaluator["train_y"][order], distances[order]):
        scores[int(label)] += 1.0 / max(float(distance), 1e-6)
    total = float(scores.sum())
    probs = scores / total if total > 0 else np.ones(4, dtype=np.float64) / 4.0
    pred = int(scores.argmax())
    centroid_distances = np.linalg.norm(evaluator["centroids"] - x.reshape(1, -1), axis=1)
    centroid_scores = 1.0 / np.maximum(centroid_distances, 1e-6)
    centroid_probs = centroid_scores / centroid_scores.sum()
    centroid_pred = int(centroid_scores.argmax())
    return {
        "pred": pred,
        "pred_label": f"Q{pred + 1}",
        "scores": [round(float(value), 6) for value in scores.tolist()],
        "probs": [round(float(value), 6) for value in probs.tolist()],
        "confidence": round(float(probs[pred]), 6),
        "centroid_pred": centroid_pred,
        "centroid_pred_label": f"Q{centroid_pred + 1}",
        "centroid_distances": [round(float(value), 6) for value in centroid_distances.tolist()],
        "centroid_probs": [round(float(value), 6) for value in centroid_probs.tolist()],
    }


def summarize(entries: list[dict]) -> dict:
    counter = Counter(entry["pred_label"] for entry in entries)
    centroid_counter = Counter(entry.get("centroid_pred_label", "NA") for entry in entries)
    total = len(entries)
    hits = sum(1 for entry in entries if int(entry["label_id"]) == int(entry["pred"]))
    centroid_hits = sum(1 for entry in entries if int(entry["label_id"]) == int(entry.get("centroid_pred", -1)))
    return {
        "n": total,
        "pred_counts": {f"Q{i + 1}": int(counter.get(f"Q{i + 1}", 0)) for i in range(4)},
        "centroid_pred_counts": {f"Q{i + 1}": int(centroid_counter.get(f"Q{i + 1}", 0)) for i in range(4)},
        "pred_ratios": {f"Q{i + 1}": float(counter.get(f"Q{i + 1}", 0) / total) if total else 0.0 for i in range(4)},
        "forced_label_agreement": float(hits / total) if total else 0.0,
        "centroid_forced_label_agreement": float(centroid_hits / total) if total else 0.0,
        "mean_target_prob": float(np.mean([entry["target_prob"] for entry in entries])) if entries else None,
    }


def objective_score(row: dict, label_id: int, mode: str) -> float:
    knn_prob = float(row["probs"][label_id])
    centroid_prob = float(row["centroid_probs"][label_id])
    if mode == "knn":
        return knn_prob
    if mode == "centroid":
        return centroid_prob
    return 0.5 * knn_prob + 0.5 * centroid_prob


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(DIFFBGM_PKG))
    sys.path.insert(0, str(CORE))
    sys.path.insert(0, str(CORE / "scripts"))
    set_seed(args.seed)

    from data.dataloader import get_train_val_dataloaders
    from emotionsketch.data import EmotionLabelProvider, build_emotion_sketch_from_batch
    from export_teacher_forced_demo_grid import (
        binarize,
        build_baseline_model,
        denoise_pred_x0,
        load_adapter,
        write_prmat2c,
    )
    from train_emopia_midi_quadrant_evaluator_v3 import midi_features_v3

    timesteps = parse_list(args.timesteps, int)
    thresholds = [value for value in args.thresholds.split(",") if value]
    label_scales = parse_list(args.label_scales, float)
    noise_seeds = parse_list(args.noise_seeds, int)
    labels = parse_list(args.labels, int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = load_v3_evaluator(Path(V3_CKPT), args.rerank_k)
    _train_dl, val_dl = get_train_val_dataloaders(
        args.batch_size,
        num_workers=0,
        pin_memory=False,
        use_track=[0, 1, 2],
    )
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
                one_prmat,
                one_chord,
                one_visual,
                one_caption,
                one_shot,
                label_provider,
            )
            for label_id in labels:
                label_candidates = []
                forced_label = torch.full((1,), label_id, dtype=torch.long, device=device)
                for label_scale in label_scales:
                    if args.condition_source == "baseline":
                        cond = one_visual
                    else:
                        with torch.no_grad():
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
                            for threshold in thresholds:
                                pred_binary = binarize(pred_x0, one_prmat2c, threshold)
                                name = (
                                    f"{clip}_Q{label_id + 1}_scale{label_scale:g}_"
                                    f"t{timestep}_seed{noise_seed}_th{threshold}.mid"
                                )
                                path = candidates_dir / name
                                write_prmat2c(path, pred_binary)
                                feature = midi_features_v3(path)
                                if feature is None:
                                    continue
                                score = score_feature_v3(feature, evaluator)
                                row = {
                                    "clip": clip,
                                    "method": "guided_candidate",
                                    "path": display_path(path),
                                    "label": f"Q{label_id + 1}",
                                    "label_id": label_id,
                                    "pseudo_label": int(pseudo_label.item()),
                                    "label_scale": label_scale,
                                    "timestep": timestep,
                                    "noise_seed": noise_seed,
                                    "threshold": threshold,
                                    **score,
                                }
                                row["target_prob_knn"] = float(row["probs"][label_id])
                                row["target_prob_centroid"] = float(row["centroid_probs"][label_id])
                                row["target_prob"] = objective_score(row, label_id, args.rerank_objective)
                                row["target_hit"] = int(int(row["pred"]) == label_id)
                                row["centroid_target_hit"] = int(int(row["centroid_pred"]) == label_id)
                                label_candidates.append(row)
                                all_candidates.append(row)
                best = sorted(
                    label_candidates,
                    key=lambda item: (
                        item["target_hit"],
                        item["centroid_target_hit"],
                        item["target_prob"],
                        item["target_prob_knn"],
                        item["target_prob_centroid"],
                    ),
                    reverse=True,
                )[0]
                source = ROOT / best["path"]
                selected_path = selected_dir / f"{clip}_selected_Q{label_id + 1}.mid"
                selected_path.write_bytes(source.read_bytes())
                best_selected = dict(best)
                best_selected["selected_path"] = display_path(selected_path)
                selected.append(best_selected)
            seen += 1

    candidate_by_label = {
        f"Q{label_id + 1}": summarize([entry for entry in all_candidates if int(entry["label_id"]) == label_id])
        for label_id in range(4)
    }
    selected_by_label = {
        f"Q{label_id + 1}": summarize([entry for entry in selected if int(entry["label_id"]) == label_id])
        for label_id in range(4)
    }
    manifest = {
        "status": "GUIDED_DECODE_RERANK_EXPORTED",
        "entries": selected,
    }
    result = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "GUIDED_DECODE_RERANK_V3_DONE",
        "clips": seen,
        "candidate_count": len(all_candidates),
        "selected_count": len(selected),
        "timesteps": timesteps,
        "thresholds": thresholds,
        "label_scales": label_scales,
        "noise_seeds": noise_seeds,
        "labels": labels,
        "rerank_evaluator": display_path(V3_CKPT),
        "rerank_k": args.rerank_k,
        "rerank_objective": args.rerank_objective,
        "condition_source": args.condition_source,
        "evaluator_validation": evaluator["summary"].get("best"),
        "candidate_summary": summarize(all_candidates),
        "candidate_by_label": candidate_by_label,
        "selected_summary": summarize(selected),
        "selected_by_label": selected_by_label,
        "selected": selected,
        "manifest": display_path(out_dir / "manifest.json"),
        "method_note": (
            "For each clip and requested label, generate multiple teacher-forced denoising candidates over "
            "timesteps, noise seeds, thresholds, and label scales, then select the MIDI with the highest V3 "
            "target-class score. No MIDI postprocessing is applied in this script."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("GUIDED_DECODE_RERANK_V3_DONE", display_path(Path(args.out_json)), "CANDIDATES", len(all_candidates))
    print(json.dumps({"candidate_summary": result["candidate_summary"], "selected_summary": result["selected_summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
