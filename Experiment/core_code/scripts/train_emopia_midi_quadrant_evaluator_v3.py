#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pretty_midi
import torch

from train_emopia_midi_quadrant_classifier import EMOPIA_MIDIS, ROOT, confusion_matrix, macro_f1

RUN_DIR = ROOT / "Experiment" / "core_code" / "runs" / "emopia_midi_quadrant_evaluator_v3"
CKPT_PATH = ROOT / "Experiment" / "core_code" / "checkpoints" / "emopia_midi_quadrant_evaluator_v3.pt"

FEATURE_NAMES = [
    "density_beats",
    "onset_density",
    "note_count_norm",
    "notes_per_second_norm",
    "pitch_mean",
    "pitch_std",
    "pitch_range",
    "pitch_q10",
    "pitch_q25",
    "pitch_q50",
    "pitch_q75",
    "pitch_q90",
    "high_ratio",
    "mid_ratio",
    "low_ratio",
    "pitch_entropy",
    "pitch_class_entropy",
    "duration_mean",
    "duration_std",
    "duration_q10",
    "duration_q25",
    "duration_q50",
    "duration_q75",
    "duration_q90",
    "long_note_ratio",
    "short_note_ratio",
    "velocity_mean",
    "velocity_std",
    "velocity_q10",
    "velocity_q25",
    "velocity_q50",
    "velocity_q75",
    "velocity_q90",
    "tempo_mean",
    "tempo_std",
    "onset_polyphony_mean",
    "onset_polyphony_std",
    "onset_polyphony_max",
    "multi_onset_ratio",
    "ioi_mean",
    "ioi_std",
    "ioi_q25",
    "ioi_q50",
    "ioi_q75",
    "interval_abs_mean",
    "interval_abs_std",
    "up_interval_ratio",
    "down_interval_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train richer MIDI affective evaluator v3 on EMOPIA.")
    parser.add_argument("--midi-dir", default=str(EMOPIA_MIDIS))
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--ks", default="1,3,5,7,11,15,21,31")
    return parser.parse_args()


def safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def entropy_from_bins(values: np.ndarray, bins: int, value_range: tuple[int, int] | None = None) -> float:
    if values.size == 0:
        return 0.0
    if value_range is None:
        hist = np.bincount(values.astype(np.int64), minlength=bins).astype(np.float32)
    else:
        hist, _ = np.histogram(values, bins=bins, range=value_range)
        hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    prob = hist / total
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum() / math.log2(bins))


def midi_features_v3(path: Path) -> np.ndarray | None:
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    if not notes:
        return None
    notes = sorted(notes, key=lambda note: (note.start, note.pitch, note.end))
    pitches = np.asarray([note.pitch for note in notes], dtype=np.float32)
    velocities = np.asarray([note.velocity for note in notes], dtype=np.float32)
    starts = np.asarray([note.start for note in notes], dtype=np.float32)
    ends = np.asarray([note.end for note in notes], dtype=np.float32)
    tempos = midi.get_tempo_changes()[1]
    if len(tempos) == 0:
        tempos = np.asarray([120.0], dtype=np.float32)
    tempo_mean_raw = float(np.mean(tempos))
    beat_duration = 60.0 / max(tempo_mean_raw, 1e-6)
    beat_times = midi.get_beats()
    beat_count = max(len(beat_times), 1)
    total_time = max(float(ends.max() - starts.min()), 1e-3)
    durations_beats = np.maximum(ends - starts, 1e-3) / beat_duration
    quantized_starts = np.round(starts / 0.05).astype(np.int64)
    unique_onsets, onset_counts = np.unique(quantized_starts, return_counts=True)
    onset_count = max(len(unique_onsets), 1)
    sorted_pitch_by_onset = pitches[np.argsort(starts)]
    intervals = np.diff(sorted_pitch_by_onset) if len(sorted_pitch_by_onset) > 1 else np.asarray([], dtype=np.float32)
    unique_starts = np.sort(np.unique(starts))
    ioi_beats = np.diff(unique_starts) / beat_duration if len(unique_starts) > 1 else np.asarray([], dtype=np.float32)

    pitch_norm = np.clip(pitches / 127.0, 0.0, 1.0)
    velocity_norm = np.clip(velocities / 127.0, 0.0, 1.0)
    duration_norm = np.clip(durations_beats / 4.0, 0.0, 1.0)
    ioi_norm = np.clip(ioi_beats / 4.0, 0.0, 1.0) if ioi_beats.size else np.asarray([0.0], dtype=np.float32)
    abs_intervals_norm = np.clip(np.abs(intervals) / 24.0, 0.0, 1.0) if intervals.size else np.asarray([0.0], dtype=np.float32)

    return np.asarray(
        [
            min(len(notes) / (beat_count * 8.0), 1.0),
            min(onset_count / (beat_count * 4.0), 1.0),
            min(len(notes) / 2000.0, 1.0),
            min((len(notes) / total_time) / 24.0, 1.0),
            float(pitch_norm.mean()),
            float(pitch_norm.std()),
            float((pitches.max() - pitches.min()) / 127.0),
            safe_quantile(pitch_norm, 0.10),
            safe_quantile(pitch_norm, 0.25),
            safe_quantile(pitch_norm, 0.50),
            safe_quantile(pitch_norm, 0.75),
            safe_quantile(pitch_norm, 0.90),
            float((pitches >= 64).mean()),
            float(((pitches >= 48) & (pitches < 64)).mean()),
            float((pitches < 48).mean()),
            entropy_from_bins(pitches.astype(np.int64), 128),
            entropy_from_bins((pitches.astype(np.int64) % 12), 12),
            float(duration_norm.mean()),
            float(duration_norm.std()),
            safe_quantile(duration_norm, 0.10),
            safe_quantile(duration_norm, 0.25),
            safe_quantile(duration_norm, 0.50),
            safe_quantile(duration_norm, 0.75),
            safe_quantile(duration_norm, 0.90),
            float((durations_beats >= 1.0).mean()),
            float((durations_beats <= 0.25).mean()),
            float(velocity_norm.mean()),
            float(velocity_norm.std()),
            safe_quantile(velocity_norm, 0.10),
            safe_quantile(velocity_norm, 0.25),
            safe_quantile(velocity_norm, 0.50),
            safe_quantile(velocity_norm, 0.75),
            safe_quantile(velocity_norm, 0.90),
            min(tempo_mean_raw / 240.0, 1.0),
            min(float(np.std(tempos)) / 120.0, 1.0),
            min(float(onset_counts.mean()) / 10.0, 1.0),
            min(float(onset_counts.std()) / 10.0, 1.0),
            min(float(onset_counts.max()) / 10.0, 1.0),
            float((onset_counts > 1).mean()),
            float(ioi_norm.mean()),
            float(ioi_norm.std()),
            safe_quantile(ioi_norm, 0.25),
            safe_quantile(ioi_norm, 0.50),
            safe_quantile(ioi_norm, 0.75),
            float(abs_intervals_norm.mean()),
            float(abs_intervals_norm.std()),
            float((intervals > 0).mean()) if intervals.size else 0.0,
            float((intervals < 0).mean()) if intervals.size else 0.0,
        ],
        dtype=np.float32,
    )


def load_dataset(midi_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = []
    labels = []
    paths = []
    for path in sorted(midi_dir.glob("Q[1-4]_*.mid")):
        feature = midi_features_v3(path)
        if feature is None:
            continue
        features.append(feature)
        labels.append(int(path.name[1]) - 1)
        paths.append(str(path.relative_to(ROOT)))
    return np.stack(features), np.asarray(labels, dtype=np.int64), paths


def stratified_split(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    train_indices = []
    val_indices = []
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        by_label[label].append(index)
    for indices in by_label.values():
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    return np.asarray(sorted(train_indices), dtype=np.int64), np.asarray(sorted(val_indices), dtype=np.int64)


def knn_predict(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, k: int) -> np.ndarray:
    predictions = []
    for row in query_x:
        distances = np.linalg.norm(train_x - row.reshape(1, -1), axis=1)
        order = np.argsort(distances)[:k]
        weights = 1.0 / np.maximum(distances[order], 1e-6)
        scores = np.zeros(4, dtype=np.float64)
        for label, weight in zip(train_y[order], weights):
            scores[int(label)] += float(weight)
        predictions.append(int(scores.argmax()))
    return np.asarray(predictions, dtype=np.int64)


def prototype_predict(centroids: np.ndarray, query_x: np.ndarray) -> np.ndarray:
    predictions = []
    for row in query_x:
        distances = np.linalg.norm(centroids - row.reshape(1, -1), axis=1)
        predictions.append(int(distances.argmin()))
    return np.asarray(predictions, dtype=np.int64)


def evaluate_predictions(pred: np.ndarray, target: np.ndarray) -> dict:
    return {
        "accuracy": float((pred == target).mean()),
        "macro_f1": macro_f1(pred, target),
        "confusion": confusion_matrix(pred, target),
        "pred_counts": {f"Q{i + 1}": int((pred == i).sum()) for i in range(4)},
    }


def per_class_recall(confusion: list[list[int]]) -> dict:
    out = {}
    for label, row in enumerate(confusion):
        total = sum(row)
        out[f"Q{label + 1}"] = row[label] / total if total else 0.0
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    features, labels, paths = load_dataset(Path(args.midi_dir))
    train_idx, val_idx = stratified_split(labels, args.val_ratio, args.seed)
    mean = features[train_idx].mean(axis=0)
    std = features[train_idx].std(axis=0) + 1e-6
    norm = (features - mean) / std
    train_x = norm[train_idx]
    train_y = labels[train_idx]
    val_x = norm[val_idx]
    val_y = labels[val_idx]
    ks = [int(value) for value in args.ks.split(",") if value]
    candidates = []
    for k in ks:
        pred = knn_predict(train_x, train_y, val_x, k)
        metrics = evaluate_predictions(pred, val_y)
        metrics["method"] = "knn"
        metrics["k"] = k
        metrics["per_class_recall"] = per_class_recall(metrics["confusion"])
        candidates.append(metrics)
        print("KNN", json.dumps(metrics, sort_keys=True), flush=True)
    centroids = np.stack([train_x[train_y == label].mean(axis=0) for label in range(4)])
    pred = prototype_predict(centroids, val_x)
    proto_metrics = evaluate_predictions(pred, val_y)
    proto_metrics["method"] = "prototype"
    proto_metrics["k"] = None
    proto_metrics["per_class_recall"] = per_class_recall(proto_metrics["confusion"])
    candidates.append(proto_metrics)
    print("PROTOTYPE", json.dumps(proto_metrics, sort_keys=True), flush=True)
    best = sorted(candidates, key=lambda item: (item["macro_f1"], item["accuracy"]), reverse=True)[0]
    best_pred = knn_predict(train_x, train_y, val_x, best["k"]) if best["method"] == "knn" else prototype_predict(centroids, val_x)
    summary = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "EMOPIA_MIDI_QUADRANT_EVALUATOR_V3_OK",
        "source": str(Path(args.midi_dir).relative_to(ROOT)),
        "feature_names": FEATURE_NAMES,
        "samples": int(len(labels)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "class_counts": {f"Q{i + 1}": int((labels == i).sum()) for i in range(4)},
        "candidates": candidates,
        "best": best,
        "val_examples": [
            {"path": paths[int(index)], "target": int(labels[index]), "pred": int(pred)}
            for index, pred in zip(val_idx[:40], best_pred[:40])
        ],
        "checkpoint": str(CKPT_PATH.relative_to(ROOT)),
        "method_note": "Evaluator v3 uses richer hand-engineered MIDI statistics and a standardized distance classifier; no generated examples are used for training.",
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(
        {
            "method": best["method"],
            "k": best["k"],
            "train_x": torch.tensor(train_x, dtype=torch.float32),
            "train_y": torch.tensor(train_y, dtype=torch.long),
            "centroids": torch.tensor(centroids, dtype=torch.float32),
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
            "feature_names": FEATURE_NAMES,
            "summary": summary,
            "paths": [paths[int(index)] for index in train_idx],
        },
        CKPT_PATH,
    )
    print("SUMMARY", json.dumps(summary["best"], sort_keys=True), flush=True)
    print("EMOPIA_MIDI_QUADRANT_EVALUATOR_V3_OK", flush=True)


if __name__ == "__main__":
    main()
