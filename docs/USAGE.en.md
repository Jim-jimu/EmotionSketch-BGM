# Usage & Reproduction

[Back to the project](../README.en.md) · [简体中文](USAGE.md)

## Run the text router first

Run from the repository root. No Diff-BGM download or PyTorch installation is required:

```bash
python3 Experiment/core_code/scripts/train_prompt_q_router.py --help
PYTHONPATH=Experiment/core_code python3 - <<'PY'
from emotionsketch.prompt_router import rule_predict
print(rule_predict("温暖平静的背景音乐"))
PY
```

For the trainable router, prepare the following JSONL files. Each line contains `text` and `label`, with optional `id`; use labels `Q1` through `Q4`. Keep training, validation, and stress-test splits separate.

```text
Experiment/datasets/prompt_q_router/train.jsonl
Experiment/datasets/prompt_q_router/val.jsonl
Experiment/datasets/prompt_q_router/stress_test_gold.jsonl
```

```bash
python3 Experiment/core_code/scripts/train_prompt_q_router.py
```

The model is saved to `Experiment/core_code/checkpoints/prompt_q_router.pkl`. Load pickle files and model weights from trusted sources.

## Prepare the music environment

The repository retains the original `Experiment/core_code/` hierarchy. Scripts use it to locate resources and outputs. Execute commands from the repository root.

Music modules directly depend on NumPy, PyTorch, and pretty_midi, listed in [requirements.txt](../requirements.txt). This is a direct-dependency list; the complete Diff-BGM environment also requires upstream installation instructions. GPU training and a pinned compatibility matrix have not been revalidated for this release.

```bash
python3 -m pip install -r requirements.txt
git clone https://github.com/sizhelee/Diff-BGM.git Experiment/code_references/Diff-BGM
git -C Experiment/code_references/Diff-BGM checkout c5a0e8a2d589142ce5951d99c1627fd7d523411c
git -C Experiment/code_references/Diff-BGM apply ../../../patches/diffbgm-shot-count.patch
bash Experiment/core_code/scripts/prepare_remote_layout.sh
```

The patch restores the data-loader fix used in the project: appending each sample's shot count to the batch. Third-party source is prepared separately. The pinned commit is the Diff-BGM revision used in the local experiments.

Required resources:

- Follow the [Diff-BGM preparation instructions](https://github.com/sizhelee/Diff-BGM#2-training) for BGM909/POP909 features, pretrained components, and split files. The layout script creates directories and symlinks; it does not download data.
- Obtain [EMOPIA](https://github.com/annahung31/EMOPIA) and place `EMOPIA_2.2/midis/` and `EMOPIA_2.2/CP_events/` under `Experiment/datasets/emopia/`.
- Train or supply baseline, adapter, and MFAE weights. Full music experiments require the associated datasets and compute resources.

## Training and decoding entrypoints

The commands below require those resources. They document the experimental invocation pattern; datasets and trained weights are prepared separately.

```bash
# EMOPIA quadrant prior and MFAE evaluator
python3 Experiment/core_code/scripts/build_emopia_emotion_prior.py
python3 Experiment/core_code/scripts/train_emopia_midi_quadrant_evaluator_v3.py

# Run names match the default checkpoint paths used by decoding scripts
python3 Experiment/core_code/scripts/baseline_bounded_train.py \
  --steps 1000 --run-name baseline_1000steps
python3 Experiment/core_code/scripts/emotionsketch_adapter_train.py \
  --steps 100 --label-mode emopia_prior \
  --run-name emotionsketch_adapter_emopia_prior_100steps

# Quadrant-specific candidate generation
python3 Experiment/core_code/scripts/guided_decode_rerank_demo.py --labels 0,1
python3 Experiment/core_code/scripts/constrained_decode_eval_demo.py --labels 2
python3 Experiment/core_code/scripts/q4_profile_search_v3.py --profile-mode focused
```

Shared model loading, denoising, and MIDI export are implemented in [export_teacher_forced_demo_grid.py](../Experiment/core_code/scripts/export_teacher_forced_demo_grid.py). Evaluator training imports shared metrics from [train_emopia_midi_quadrant_classifier.py](../Experiment/core_code/scripts/train_emopia_midi_quadrant_classifier.py). Additional diagnostics include [label interventions](../Experiment/core_code/scripts/evaluate_label_intervention_sensitivity.py), [parameter counting](../Experiment/core_code/scripts/summarize_parameter_efficiency.py), and [MIDI manifest scoring](../Experiment/core_code/scripts/score_midi_manifest_v3.py).

## Runtime and evaluation notes

Music scripts use teacher-forced conditional denoising: noise is added to reference symbolic music, then pre-extracted features condition candidate generation and selection. MFAE scores measure agreement under the evaluator's quadrant criterion; perceptual validation requires independent listening evaluation.

Release checks cover Python/Shell syntax, project-local imports, directory resolution, and router training, prediction, and serialization. Full GPU training and music generation were not rerun for this documentation update.

Resources retain their upstream usage terms. This repository has not specified an open-source license.
