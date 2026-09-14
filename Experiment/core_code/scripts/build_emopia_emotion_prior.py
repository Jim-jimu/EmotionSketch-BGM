#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EMOPIA_CP = ROOT / "Experiment" / "datasets" / "emopia" / "EMOPIA_2.2" / "CP_events"
OUT_PATH = ROOT / "Experiment" / "datasets" / "derived" / "emopia_emotion_prior.json"


def parse_number(value: object, prefix: str) -> float | None:
    if not isinstance(value, str) or not value.startswith(prefix):
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)", value)
    return float(match.group(1)) if match else None


def features_from_cp_events(path: Path) -> np.ndarray | None:
    events = pickle.load(path.open("rb"))
    pitches: list[float] = []
    beat_count = 0
    for event in events:
        event_type = event.get("type")
        if event_type == "Metrical":
            bar_beat = event.get("bar-beat")
            if isinstance(bar_beat, str) and bar_beat.startswith("Beat_"):
                beat_count += 1
        elif event_type == "Note":
            pitch = parse_number(event.get("pitch"), "Note_Pitch_")
            if pitch is not None:
                pitches.append(pitch)
    if not pitches:
        return None
    pitches_arr = np.asarray(pitches, dtype=np.float32)
    pitch_norm = np.clip(pitches_arr / 127.0, 0.0, 1.0)
    beat_count = max(beat_count, 1)
    density = min(len(pitches) / (beat_count * 8.0), 1.0)
    return np.asarray(
        [
            density,
            float(pitch_norm.mean()),
            float((pitches_arr >= 64).mean()),
            float((pitches_arr < 48).mean()),
            float(pitch_norm.std()),
        ],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact EMOPIA quadrant prior centroids.")
    parser.add_argument("--emopia-cp", default=str(EMOPIA_CP))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    cp_dir = Path(args.emopia_cp)
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    skipped = 0
    for path in sorted(cp_dir.glob("Q[1-4]_*.pkl")):
        label = path.name.split("_", 1)[0]
        features = features_from_cp_events(path)
        if features is None:
            skipped += 1
            continue
        groups[label].append(features)

    centroids = {}
    stds = {}
    counts = {}
    for q in [f"Q{i}" for i in range(1, 5)]:
        values = np.stack(groups[q], axis=0)
        centroids[q] = values.mean(axis=0).round(6).tolist()
        stds[q] = values.std(axis=0).round(6).tolist()
        counts[q] = int(values.shape[0])

    payload = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": str(cp_dir.relative_to(ROOT)),
        "feature_names": ["density", "pitch_mean", "high_ratio", "low_ratio", "pitch_spread"],
        "label_mapping": {
            "0": "Q1",
            "1": "Q2",
            "2": "Q3",
            "3": "Q4",
        },
        "counts": counts,
        "skipped_files": skipped,
        "centroids": centroids,
        "std": stds,
        "total_files": int(sum(counts.values())),
        "file_label_counts": dict(Counter({q: counts[q] for q in counts})),
        "method_note": (
            "Centroids are built from EMOPIA CP_events using compact symbolic features. "
            "They support traceable pseudo-labeling for BGM909 wiring experiments, not ground-truth BGM909 emotion labels."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("EMOPIA_PRIOR_WRITTEN", out.relative_to(ROOT))
    print("COUNTS", json.dumps(counts, sort_keys=True))
    print("CENTROIDS", json.dumps(centroids, sort_keys=True))


if __name__ == "__main__":
    main()
