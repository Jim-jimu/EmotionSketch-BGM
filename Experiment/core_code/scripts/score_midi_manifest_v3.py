#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from train_emopia_midi_quadrant_evaluator_v3 import CKPT_PATH, ROOT, midi_features_v3

DEFAULT_MANIFEST = ROOT / "Experiment" / "outputs" / "demo_midi_postprocess_q34_2026-05-09" / "manifest.json"
OUT_PATH = ROOT / "Experiment" / "analysis" / "DEMO_MIDI_POSTPROCESS_Q34_V3_2026-05-09.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a MIDI manifest with evaluator v3.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--checkpoint", default=str(CKPT_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        if count:
            p = count / total
            value -= p * math.log2(p)
    return value


def load_evaluator(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    return {
        "method": payload["method"],
        "k": payload.get("k"),
        "train_x": payload["train_x"].float().numpy(),
        "train_y": payload["train_y"].long().numpy(),
        "centroids": payload["centroids"].float().numpy(),
        "mean": payload["mean"].float().numpy(),
        "std": np.maximum(payload["std"].float().numpy(), 1e-6),
        "summary": payload.get("summary", {}),
    }


def predict(feature: np.ndarray, evaluator: dict) -> tuple[int, list[float]]:
    x = (feature - evaluator["mean"]) / evaluator["std"]
    if evaluator["method"] == "prototype":
        distances = np.linalg.norm(evaluator["centroids"] - x.reshape(1, -1), axis=1)
        scores = -distances
    else:
        distances = np.linalg.norm(evaluator["train_x"] - x.reshape(1, -1), axis=1)
        order = np.argsort(distances)[: int(evaluator["k"])]
        scores = np.zeros(4, dtype=np.float64)
        for label, distance in zip(evaluator["train_y"][order], distances[order]):
            scores[int(label)] += 1.0 / max(float(distance), 1e-6)
    shifted = scores - scores.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()
    return int(scores.argmax()), [float(value) for value in probs.tolist()]


def score_entry(entry: dict, evaluator: dict) -> dict | None:
    path = ROOT / entry["path"]
    feature = midi_features_v3(path)
    if feature is None:
        return None
    pred, probs = predict(feature, evaluator)
    scored = dict(entry)
    scored["pred"] = pred
    scored["pred_label"] = f"Q{pred + 1}"
    scored["probs"] = [round(value, 6) for value in probs]
    scored["confidence"] = max(scored["probs"])
    return scored


def summarize(entries: list[dict]) -> dict:
    counter = Counter(entry["pred_label"] for entry in entries)
    counts = {f"Q{i + 1}": int(counter.get(f"Q{i + 1}", 0)) for i in range(4)}
    total = len(entries)
    label_total = 0
    label_hits = 0
    for entry in entries:
        if "target_label_id" in entry:
            label_total += 1
            label_hits += int(int(entry["target_label_id"]) == int(entry["pred"]))
        elif "label_id" in entry:
            label_total += 1
            label_hits += int(int(entry["label_id"]) == int(entry["pred"]))
    return {
        "n": total,
        "pred_counts": counts,
        "pred_ratios": {key: counts[key] / total if total else 0.0 for key in counts},
        "unique_preds": sum(1 for value in counts.values() if value > 0),
        "entropy": entropy(counts),
        "mean_confidence": float(np.mean([entry["confidence"] for entry in entries])) if entries else None,
        "forced_label_agreement": label_hits / label_total if label_total else None,
    }


def examples_by_pred(entries: list[dict], limit: int = 12) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        key = entry["pred_label"]
        if len(grouped[key]) < limit:
            grouped[key].append(entry)
    return {f"Q{i + 1}": grouped.get(f"Q{i + 1}", []) for i in range(4)}


def main() -> None:
    args = parse_args()
    evaluator = load_evaluator(Path(args.checkpoint))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    raw_entries = manifest.get("entries", [])
    scored = []
    skipped = []
    for entry in raw_entries:
        try:
            item = score_entry(entry, evaluator)
        except Exception as exc:
            skipped.append({"path": entry.get("path"), "reason": repr(exc)})
            continue
        if item is None:
            skipped.append({"path": entry.get("path"), "reason": "no_non_drum_notes"})
            continue
        scored.append(item)

    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in scored:
        if "target" in entry:
            key = str(entry["target"])
        elif entry.get("method") == "full":
            key = f"full|Q{int(entry.get('label_id', -1)) + 1}|t={entry.get('timestep')}|th={entry.get('threshold')}|scale={entry.get('label_scale')}"
        else:
            key = str(entry.get("method", "unknown"))
        groups[key].append(entry)
    result = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "MIDI_MANIFEST_V3_SCORED",
        "manifest": display_path(Path(args.manifest)),
        "checkpoint": display_path(Path(args.checkpoint)),
        "evaluator_method": evaluator["method"],
        "evaluator_k": evaluator["k"],
        "evaluator_validation": evaluator["summary"].get("best"),
        "scored_entries": len(scored),
        "skipped": skipped,
        "overall": summarize(scored),
        "groups": {key: summarize(value) for key, value in sorted(groups.items())},
        "scored_examples": scored[:40],
        "examples_by_pred": examples_by_pred(scored),
        "method_note": "Scores exported MIDI using v3 EMOPIA MIDI evaluator. V3 remains a diagnostic transfer evaluator, not human affect ground truth.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("MIDI_MANIFEST_V3_SCORED", display_path(out), "N", len(scored))
    print(json.dumps({"overall": result["overall"], "validation": result["evaluator_validation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
