#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pretty_midi
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
EMOPIA_MIDIS = ROOT / "Experiment" / "datasets" / "emopia" / "EMOPIA_2.2" / "midis"
RUN_DIR = ROOT / "Experiment" / "core_code" / "runs" / "emopia_midi_quadrant_classifier"
CKPT_PATH = ROOT / "Experiment" / "core_code" / "checkpoints" / "emopia_midi_quadrant_classifier.pt"


FEATURE_NAMES = [
    "density",
    "pitch_mean",
    "high_ratio",
    "low_ratio",
    "pitch_spread",
    "pitch_range",
    "pitch_entropy",
    "duration_mean",
    "duration_std",
    "velocity_mean",
    "velocity_std",
    "tempo_mean",
    "tempo_std",
    "note_count_norm",
    "onset_density",
    "mean_onset_polyphony",
]


class QuadrantClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stronger MIDI-derived EMOPIA quadrant classifier.")
    parser.add_argument("--midi-dir", default=str(EMOPIA_MIDIS))
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser.parse_args()


def midi_features(path: Path) -> np.ndarray | None:
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    if not notes:
        return None
    pitches = np.asarray([note.pitch for note in notes], dtype=np.float32)
    velocities = np.asarray([note.velocity for note in notes], dtype=np.float32)
    tempos = midi.get_tempo_changes()[1]
    if len(tempos) == 0:
        tempos = np.asarray([120.0], dtype=np.float32)
    tempo_mean_raw = float(np.mean(tempos))
    beat_times = midi.get_beats()
    beat_count = max(len(beat_times), 1)
    beat_duration = 60.0 / max(tempo_mean_raw, 1e-6)
    durations_beats = np.asarray([(note.end - note.start) / beat_duration for note in notes], dtype=np.float32)
    pitch_norm = np.clip(pitches / 127.0, 0.0, 1.0)
    hist = np.bincount(pitches.astype(np.int64), minlength=128).astype(np.float32)
    prob = hist / max(hist.sum(), 1.0)
    entropy = float(-(prob[prob > 0] * np.log2(prob[prob > 0])).sum() / 7.0)
    starts = np.asarray([note.start for note in notes], dtype=np.float32)
    quantized_starts = np.round(starts / 0.05).astype(np.int64)
    _, onset_counts = np.unique(quantized_starts, return_counts=True)
    density = min(len(notes) / (beat_count * 8.0), 1.0)
    onset_density = min(len(onset_counts) / (beat_count * 4.0), 1.0)
    return np.asarray(
        [
            density,
            float(pitch_norm.mean()),
            float((pitches >= 64).mean()),
            float((pitches < 48).mean()),
            float(pitch_norm.std()),
            float((pitches.max() - pitches.min()) / 127.0),
            entropy,
            min(float(durations_beats.mean()) / 4.0, 1.0),
            min(float(durations_beats.std()) / 4.0, 1.0),
            float(velocities.mean() / 127.0),
            float(velocities.std() / 127.0),
            min(tempo_mean_raw / 240.0, 1.0),
            min(float(np.std(tempos)) / 120.0, 1.0),
            min(len(notes) / 2000.0, 1.0),
            onset_density,
            min(float(onset_counts.mean()) / 10.0, 1.0),
        ],
        dtype=np.float32,
    )


def load_dataset(midi_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = []
    labels = []
    paths = []
    for path in sorted(midi_dir.glob("Q[1-4]_*.mid")):
        feature = midi_features(path)
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


def confusion_matrix(pred: np.ndarray, target: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((4, 4), dtype=np.int64)
    for gold, guess in zip(target.tolist(), pred.tolist()):
        matrix[gold, guess] += 1
    return matrix.tolist()


def macro_f1(pred: np.ndarray, target: np.ndarray) -> float:
    scores = []
    for label in range(4):
        tp = int(((pred == label) & (target == label)).sum())
        fp = int(((pred == label) & (target != label)).sum())
        fn = int(((pred != label) & (target == label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, list[list[int]]]:
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1).cpu().numpy()
    target = y.cpu().numpy()
    return float((pred == target).mean()), macro_f1(pred, target), confusion_matrix(pred, target)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    features, labels, paths = load_dataset(Path(args.midi_dir))
    train_idx, val_idx = stratified_split(labels, args.val_ratio, args.seed)
    mean = features[train_idx].mean(axis=0)
    std = features[train_idx].std(axis=0) + 1e-6
    features_norm = (features - mean) / std
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.tensor(features_norm[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(labels[train_idx], dtype=torch.long, device=device)
    x_val = torch.tensor(features_norm[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(labels[val_idx], dtype=torch.long, device=device)
    counts = np.bincount(labels[train_idx], minlength=4).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    model = QuadrantClassifier(features.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    best = {"val_macro_f1": -1.0, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            train_acc, train_f1, _ = evaluate(model, x_train, y_train)
            val_acc, val_f1, val_conf = evaluate(model, x_val, y_val)
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu().item()),
                "train_accuracy": train_acc,
                "train_macro_f1": train_f1,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
            history.append(row)
            print("EPOCH", json.dumps(row, sort_keys=True), flush=True)
            if val_f1 > best["val_macro_f1"]:
                best = {
                    "val_macro_f1": val_f1,
                    "epoch": epoch,
                    "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "val_confusion": val_conf,
                }
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    train_acc, train_f1, train_conf = evaluate(model, x_train, y_train)
    val_acc, val_f1, val_conf = evaluate(model, x_val, y_val)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "EMOPIA_MIDI_QUADRANT_CLASSIFIER_OK",
        "source": str(Path(args.midi_dir).relative_to(ROOT)),
        "feature_names": FEATURE_NAMES,
        "samples": int(len(labels)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "class_counts": {f"Q{i + 1}": int((labels == i).sum()) for i in range(4)},
        "train_accuracy": train_acc,
        "train_macro_f1": train_f1,
        "train_confusion": train_conf,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "val_confusion": val_conf,
        "best_epoch": int(best["epoch"]),
        "best_val_macro_f1": float(best["val_macro_f1"]),
        "checkpoint": str(CKPT_PATH.relative_to(ROOT)),
        "method_note": "MIDI-derived EMOPIA evaluator trained with the same feature extractor used for POP909 calibration.",
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RUN_DIR / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save({"model": model.state_dict(), "mean": torch.tensor(mean), "std": torch.tensor(std), "summary": summary, "paths": paths}, CKPT_PATH)
    print("SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("EMOPIA_MIDI_QUADRANT_CLASSIFIER_OK", flush=True)


if __name__ == "__main__":
    main()
