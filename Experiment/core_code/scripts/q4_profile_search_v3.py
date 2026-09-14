#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pretty_midi as pm
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "Experiment" / "core_code"
DIFFBGM_PKG = ROOT / "Experiment" / "code_references" / "Diff-BGM" / "diffbgm"
OUT_DIR = ROOT / "Experiment" / "outputs" / "q4_profile_search_2026-05-09"
OUT_JSON = ROOT / "Experiment" / "analysis" / "Q4_PROFILE_SEARCH_V3_2026-05-09.json"
V3_CKPT = CORE / "checkpoints" / "emopia_midi_quadrant_evaluator_v3.pt"
EMOPIA_PRIOR = ROOT / "Experiment" / "datasets" / "derived" / "emopia_emotion_prior.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q4-only constrained decoding profile search under V3 evaluator.")
    parser.add_argument("--clips", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timesteps", default="100,500")
    parser.add_argument("--label-scales", default="1,4")
    parser.add_argument("--noise-seeds", default="0")
    parser.add_argument("--max-profiles", type=int, default=96)
    parser.add_argument("--rerank-k", type=int, default=31)
    parser.add_argument("--profile-mode", choices=["broad", "focused"], default="broad")
    parser.add_argument("--q3-margin-lambda", type=float, default=0.35)
    parser.add_argument("--condition-source", choices=["adapter", "baseline"], default="adapter")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    return parser.parse_args()


def parse_list(text: str, cast):
    return [cast(value) for value in text.split(",") if value != ""]


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=False) for tensor in batch)


def generate_profiles(max_profiles: int, seed: int, mode: str = "broad") -> list[dict]:
    # Designed from V3 Q4-vs-Q3 diagnostics: mid-range pitches, fewer high notes,
    # long and stable durations, moderate density, and controlled velocity/tempo.
    if mode == "focused":
        pitch_windows = [(40, 72), (44, 70), (45, 69), (48, 68)]
        densities = [0.04, 0.06, 0.08, 0.10]
        dilations = [3, 5, 7]
        velocities = [34, 42, 50, 58, 66]
        tempos = [88, 104, 120]
    else:
        pitch_windows = [(40, 72), (44, 70), (45, 69), (48, 68), (50, 66), (52, 68), (54, 70)]
        densities = [0.04, 0.06, 0.08, 0.10, 0.12]
        dilations = [3, 5, 7]
        velocities = [34, 42, 50, 58, 66, 76]
        tempos = [72, 88, 104, 120]
    raw = []
    for density, (pitch_min, pitch_max), dilate, velocity, tempo in itertools.product(
        densities,
        pitch_windows,
        dilations,
        velocities,
        tempos,
    ):
        raw.append(
            {
                "density": density,
                "pitch_min": pitch_min,
                "pitch_max": pitch_max,
                "dilate": dilate,
                "velocity": velocity,
                "tempo": tempo,
            }
        )
    rng = random.Random(seed)
    rng.shuffle(raw)
    # Always include a few hand-picked diagnostics before the random subset.
    anchors = [
        {"density": 0.08, "pitch_min": 45, "pitch_max": 69, "dilate": 5, "velocity": 50, "tempo": 104},
        {"density": 0.10, "pitch_min": 48, "pitch_max": 68, "dilate": 5, "velocity": 58, "tempo": 104},
        {"density": 0.06, "pitch_min": 44, "pitch_max": 70, "dilate": 7, "velocity": 42, "tempo": 88},
        {"density": 0.12, "pitch_min": 52, "pitch_max": 68, "dilate": 3, "velocity": 66, "tempo": 120},
    ]
    seen = set()
    profiles = []
    for profile in anchors + raw:
        key = tuple(profile.items())
        if key in seen:
            continue
        seen.add(key)
        profiles.append(profile)
        if len(profiles) >= max_profiles:
            break
    return profiles


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


def score_feature(feature: np.ndarray, evaluator: dict, q3_margin_lambda: float = 0.35) -> dict:
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
    q3_score = float(0.5 * probs[2] + 0.5 * centroid_probs[2])
    q4_score = float(0.5 * probs[3] + 0.5 * centroid_probs[3])
    return {
        "pred": pred,
        "pred_label": f"Q{pred + 1}",
        "probs": [round(float(v), 6) for v in probs.tolist()],
        "centroid_pred": centroid_pred,
        "centroid_pred_label": f"Q{centroid_pred + 1}",
        "centroid_probs": [round(float(v), 6) for v in centroid_probs.tolist()],
        "q3_score": q3_score,
        "q4_score": q4_score,
        "q4_margin_score": float(q4_score - q3_margin_lambda * q3_score),
    }


def constrain(scores: torch.Tensor, profile: dict) -> torch.Tensor:
    x = scores.detach().float().clamp(0, 1)
    out = torch.zeros_like(x)
    pitch_count = x.shape[-1]
    pitch_mask = torch.zeros(pitch_count, dtype=torch.bool, device=x.device)
    pitch_mask[int(profile["pitch_min"]) : int(profile["pitch_max"]) + 1] = True
    masked = x.clone()
    masked[..., ~pitch_mask] = -1.0
    flat = masked.view(masked.shape[0], -1)
    allowed = int(masked.shape[1] * masked.shape[2] * int(pitch_mask.sum().item()))
    keep = max(8, min(flat.shape[1], int(round(allowed * float(profile["density"])))))
    for index in range(flat.shape[0]):
        values, indices = torch.topk(flat[index], keep)
        valid = values > 0
        if valid.any():
            row = out[index].view(-1)
            row[indices[valid]] = 1.0
    dilate = int(profile["dilate"])
    if dilate > 1:
        pad = dilate // 2
        out = F.max_pool2d(out, kernel_size=(dilate, 1), stride=1, padding=(pad, 0))
        out = out[..., : x.shape[-2], :]
    return out.clamp(0, 1)


def write_controlled(path: Path, tensor: torch.Tensor, profile: dict) -> None:
    prmat2c = tensor.detach().cpu().numpy()
    midi = pm.PrettyMIDI(initial_tempo=float(profile["tempo"]))
    instrument = pm.Instrument(program=pm.instrument_name_to_program("Acoustic Grand Piano"))
    n_step = prmat2c.shape[2]
    t_bar = int(n_step / 8)
    t = 0.0
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
                instrument.notes.append(
                    pm.Note(
                        velocity=int(profile["velocity"]),
                        pitch=int(key),
                        start=t + step_ind * 1 / 8,
                        end=min(t + (step_ind + dur) * 1 / 8, t + t_bar),
                    )
                )
        t += t_bar
    midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def summarize(rows: list[dict]) -> dict:
    counter = Counter(row["pred_label"] for row in rows)
    centroid_counter = Counter(row["centroid_pred_label"] for row in rows)
    return {
        "n": len(rows),
        "pred_counts": {f"Q{i + 1}": int(counter.get(f"Q{i + 1}", 0)) for i in range(4)},
        "centroid_pred_counts": {f"Q{i + 1}": int(centroid_counter.get(f"Q{i + 1}", 0)) for i in range(4)},
        "mean_q4_score": float(np.mean([row["q4_score"] for row in rows])) if rows else None,
        "max_q4_score": float(max([row["q4_score"] for row in rows])) if rows else None,
        "mean_q4_margin_score": float(np.mean([row["q4_margin_score"] for row in rows])) if rows else None,
        "max_q4_margin_score": float(max([row["q4_margin_score"] for row in rows])) if rows else None,
        "q4_hit_rate": float(counter.get("Q4", 0) / len(rows)) if rows else 0.0,
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

    profiles = generate_profiles(args.max_profiles, args.seed, args.profile_mode)
    timesteps = parse_list(args.timesteps, int)
    label_scales = parse_list(args.label_scales, float)
    noise_seeds = parse_list(args.noise_seeds, int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = load_v3(V3_CKPT, args.rerank_k)
    _train_dl, val_dl = get_train_val_dataloaders(args.batch_size, num_workers=0, pin_memory=False, use_track=[0, 1, 2])
    base_model = build_baseline_model(device)
    adapter = load_adapter(device) if args.condition_source == "adapter" else None
    label_provider = EmotionLabelProvider("emopia_prior", EMOPIA_PRIOR, device=device)
    out_dir = Path(args.out_dir)
    candidate_dir = out_dir / "candidates"
    selected_dir = out_dir / "selected"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    rows = []
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
            sketch, pseudo_label = build_emotion_sketch_from_batch(
                prmat[idx : idx + 1],
                chord[idx : idx + 1],
                visual[idx : idx + 1],
                caption[idx : idx + 1],
                shot_cnt[idx : idx + 1],
                label_provider,
            )
            forced_label = torch.full((1,), 3, dtype=torch.long, device=device)
            for label_scale in label_scales:
                with torch.no_grad():
                    if args.condition_source == "baseline":
                        cond = visual[idx : idx + 1]
                    else:
                        cond = adapter(
                            visual[idx : idx + 1],
                            sketch,
                            forced_label,
                            use_sketch=True,
                            use_label=True,
                            label_scale=label_scale,
                        )
                for timestep in timesteps:
                    for noise_seed in noise_seeds:
                        generator = torch.Generator(device=device)
                        generator.manual_seed(args.seed + seen * 10000 + int(label_scale * 100) + timestep + noise_seed)
                        noise = torch.randn(one_prmat2c.shape, device=device, generator=generator)
                        pred_x0 = denoise_pred_x0(base_model, one_prmat2c, cond, timestep, noise)
                        for profile_index, profile in enumerate(profiles):
                            constrained = constrain(pred_x0, profile)
                            path = candidate_dir / (
                                f"{clip}_q4_s{label_scale:g}_t{timestep}_n{noise_seed}_p{profile_index:03d}.mid"
                            )
                            write_controlled(path, constrained, profile)
                            feature = midi_features_v3(path)
                            if feature is None:
                                continue
                            score = score_feature(feature, evaluator, args.q3_margin_lambda)
                            row = {
                                "clip": clip,
                                "path": display_path(path),
                                "label": "Q4",
                                "label_id": 3,
                                "pseudo_label": int(pseudo_label.item()),
                                "label_scale": label_scale,
                                "timestep": timestep,
                                "noise_seed": noise_seed,
                                "profile_index": profile_index,
                                **profile,
                                **score,
                            }
                            rows.append(row)
            seen += 1

    def q4_rank_key(row: dict) -> tuple:
        return (
            int(row["pred"] == 3),
            int(row["centroid_pred"] == 3),
            row["q4_margin_score"],
            row["q4_score"],
        )

    selected = sorted(
        rows,
        key=q4_rank_key,
        reverse=True,
    )[:12]
    for rank, row in enumerate(selected):
        source = ROOT / row["path"]
        target = selected_dir / f"q4_rank{rank:02d}_{source.name}"
        target.write_bytes(source.read_bytes())
        row["selected_path"] = display_path(target)

    by_clip: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_clip[row["clip"]].append(row)
    best_by_clip = {}
    for clip, clip_rows in sorted(by_clip.items()):
        best = sorted(clip_rows, key=q4_rank_key, reverse=True)[0]
        best_by_clip[clip] = best
    q4_covered_clips = sorted([clip for clip, row in best_by_clip.items() if row["pred_label"] == "Q4"])
    candidate_count_formula = (
        "clips * len(label_scales) * len(timesteps) * len(noise_seeds) * profile_count"
    )
    result = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "Q4_PROFILE_SEARCH_V3_DONE",
        "clips": seen,
        "profiles": profiles,
        "profile_count": len(profiles),
        "profile_mode": args.profile_mode,
        "q3_margin_lambda": args.q3_margin_lambda,
        "condition_source": args.condition_source,
        "candidate_count": len(rows),
        "candidate_count_formula": candidate_count_formula,
        "budget_context": {
            "clips": seen,
            "timesteps": timesteps,
            "label_scales": label_scales,
            "noise_seeds": noise_seeds,
            "profile_count": len(profiles),
            "candidate_count_formula": candidate_count_formula,
        },
        "summary": summarize(rows),
        "best_by_clip": best_by_clip,
        "q4_covered_clips": q4_covered_clips,
        "q4_clip_coverage": float(len(q4_covered_clips) / len(best_by_clip)) if best_by_clip else 0.0,
        "top_selected": selected,
        "top_q4_score": selected[0]["q4_score"] if selected else None,
        "method_note": (
            "Q4-only search generated base Diff-BGM/EmotionSketch candidates once, then swept label-specific "
            "decode-to-MIDI profiles informed by V3 Q4-vs-Q3 feature deltas."
        ),
    }
    manifest = {"status": "Q4_PROFILE_SEARCH_SELECTED", "entries": selected}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Q4_PROFILE_SEARCH_V3_DONE", display_path(Path(args.out_json)), "CANDIDATES", len(rows))
    print(json.dumps({"summary": result["summary"], "top": selected[:3]}, sort_keys=True))


if __name__ == "__main__":
    main()
