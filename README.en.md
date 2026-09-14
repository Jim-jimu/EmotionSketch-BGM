<div align="center">

# EmotionSketch-BGM

### Natural-language emotion control for symbolic video background music

**Emotion sketch adaptation · Symbolic music generation · Evaluator-guided decoding**

[简体中文](README.md) | **English**

[Contributions](#contributions) · [Method & Pipeline](#method) · [Results](#results) · [Quick Start](#quickstart) · [Code](#code)

</div>

---

EmotionSketch-BGM studies how the same video can support different musical emotions. It connects **text-to-emotion routing, lightweight condition adaptation, and MIDI-based evaluation** into a research workflow with trainable components and explicit controls.

On a frozen Diff-BGM backbone, the adapter introduces **537,121 trainable parameters**. Across 100 validation clips with matched candidate budgets, target-quadrant agreement rises from **32.17% to 33.70%** before selection. The text front end reaches **0.8335 Macro-F1** on 80 manually curated Chinese stress-test prompts.

| Lightweight adaptation | Text emotion routing | Candidate distribution | Evaluation scale |
| :---: | :---: | :---: | :---: |
| **1.295%** | **0.8335** | **+1.53 pp** | **20,400 × 2** |
| Adapter / frozen-backbone parameters | Chinese stress-test Macro-F1 | MFAE target-hit improvement | Candidates per adapter / baseline condition |

> Results come from the project's May 2026 experiments. Target hits are defined by MFAE and measured before best-candidate selection; `pp` denotes percentage points. [Aggregate data and source records →](docs/results-summary.json)

<a id="contributions"></a>
## Contributions

**01 · A parameter-efficient emotion interface**<br>
EmotionSketch Adapter projects a 16-dimensional segment sketch and four-quadrant label embeddings into a gated residual on the 512-dimensional visual condition. It adds 537k trainable parameters while preserving the frozen backbone and its conditioning dimensions.

**02 · Feature diagnostics translated into decoding policies**<br>
An EMOPIA-derived, 48-feature MIDI Feature Affective Evaluator (MFAE) scores pitch, duration, velocity, and rhythm statistics. Q1/Q2 reranking, Q3 controlled export, and Q4 focused profile search form a reusable four-quadrant decoding workflow.

**03 · A lightweight natural-language front end**<br>
A character n-gram linear softmax classifier maps Chinese emotion descriptions to a Q1–Q4 probability distribution for the adapter's label interface. Router inference uses the Python standard library and requires no online LLM calls.

**04 · Experiments that separate condition response from selection gains**<br>
Label interventions, a matched-budget no-adapter control, and pre-selection candidate statistics measure three distinct properties: whether the label branch responds, whether the adapter shifts candidate distributions, and how the complete selection protocol covers target quadrants.

<a id="method"></a>
## Method & Pipeline

![EmotionSketch-BGM pipeline: text routing, sketch adaptation, a frozen backbone, quadrant-specific decoding, and MFAE selection.](assets/pipeline.svg)

*The diagram shows component interfaces. Current music experiments use pre-extracted features and reference symbolic music for teacher-forced conditional denoising; router predictions connect through the label interface.*

### 1. Express emotion through a shared control signal

| Quadrant | Valence | Arousal | Example intent | Label ID |
| --- | --- | --- | --- | :---: |
| Q1 | Positive | High | Joyful, excited, triumphant | 0 |
| Q2 | Negative | High | Tense, oppressive, conflict-driven | 1 |
| Q3 | Negative | Low | Sad, lonely, melancholic | 2 |
| Q4 | Positive | Low | Warm, calm, soothing | 3 |

EMOPIA quadrant centroids supply pseudo-labels during adapter training. At use time, a target can be predicted by Prompt-to-Q or specified explicitly. The `proxy` mode is a within-batch threshold baseline; its IDs do not directly represent EMOPIA quadrants.

### 2. Adapt visual conditions with a gated residual

Each sample receives a **32 × 16** sketch combining note activity, register statistics, chord summaries, visual/caption feature norms, shot count, and valence/arousal proxies. For visual conditions $V$, sketch $S$, and target label $q$:

```math
V' = V + \sigma(g)\,f_{\mathrm{out}}\!\left(f_{\mathrm{sketch}}(S) + \alpha E(q)\right)
```

$E(q)$ is broadcast over time, $\alpha$ corresponds to `label_scale`, and $g$ is a learned gate. Input and output both have shape **[B, 32, 512]**. Training updates the adapter while retaining the backbone's denoising objective.

### 3. Use emotion features to guide candidate selection

| Target | Decoding policy | Main controls |
| --- | --- | --- |
| Q1 / Q2 | Conditional candidates + MFAE reranking | Label scale, noise, timestep, binarization threshold |
| Q3 | Symbolic constraints + label-aware export | Note density, duration, velocity, tempo |
| Q4 | Diagnostic feature-profile search | Pitch window, mid-register ratio, duration, velocity, tempo |

MFAE builds its scoring space from 48 EMOPIA MIDI features. Decoding combines neighborhood probabilities and centroid scores to select target candidates, connecting model conditioning to interpretable symbolic feature diagnostics.

<a id="results"></a>
## Experimental Results

### A. Parameter efficiency and condition response

| Measurement | Result |
| --- | ---: |
| Frozen Diff-BGM SDF backbone parameters | 41,479,098 |
| EmotionSketch trainable parameters | **537,121** |
| Adapter / backbone parameter ratio | **1.295%** |
| Full adapter: label-intervention condition RMSE | **0.099496** |
| Label-only branch: label-intervention condition RMSE | 0.110717 |
| Sketch-only branch: label-intervention condition RMSE | 0.000000 |

With visual and sketch inputs fixed, forcing Q1–Q4 produces a measurable condition change in the full adapter. Removing the label path makes that response zero, isolating the role of the explicit emotion-label branch.

### B. Target-emotion candidates under matched budgets

**Protocol: 100 validation clips × four quadrants; 20,400 candidates per condition, with identical per-quadrant budgets and decoding policies.** Rates below measure MFAE target hits before best-candidate selection.

| Target | Candidates / condition | No-adapter baseline | EmotionSketch | Change |
| --- | ---: | ---: | ---: | ---: |
| Q1 | 2,400 | 33.08% | **36.04%** | **+2.96 pp** |
| Q2 | 2,400 | 59.25% | **60.33%** | **+1.08 pp** |
| Q3 · controlled export | 4,000 | 100.00% | 100.00% | 0.00 pp |
| Q4 · focused search | 11,600 | 2.98% | **4.84%** | **+1.85 pp** |
| **Overall** | **20,400** | **32.17%** | **33.70%** | **+1.53 pp** |

The adapter produces **312 additional target-hit candidates** at the same budget, including a Q4 increase from **346 to 561**. This supports a shift toward the requested emotion in the candidate distribution. The overall rate is weighted by the reported per-quadrant budgets; differences are computed from unrounded counts.

The complete MFAE-guided selection workflow produces **400/400** target-matching outputs in both conditions. Adapter contribution is therefore measured by pre-selection distribution changes; 400/400 describes four-quadrant coverage under the complete selection protocol.

### C. Chinese Prompt-to-Q routing

The router uses **400** training prompts, **100** cleaned validation prompts, and **80** manually curated stress-test prompts.

| Split | Method | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| Validation | Keyword rules | 0.6600 | 0.6430 |
| Validation | **Prompt-to-Q** | **1.0000** | **1.0000** |
| Stress test | Keyword rules | 0.8125 | 0.8242 |
| Stress test | **Prompt-to-Q** | **0.8250** | **0.8335** |

MFAE itself reaches **0.6279 Accuracy / 0.6285 Macro-F1** on **215** EMOPIA validation samples. Music-generation results use this evaluator's diagnostic criterion; text routing and MIDI emotion scoring are evaluated separately. [Full aggregates, confusion matrices, and source fingerprints →](docs/results-summary.json)

<a id="quickstart"></a>
## Quick Start

Run the Chinese rule-based routing example without additional dependencies:

```bash
PYTHONPATH=Experiment/core_code python3 - <<'PY'
from emotionsketch.prompt_router import rule_predict
print(rule_predict("温暖平静的背景音乐"))  # Q4
PY
```

This example uses the rule baseline; Prompt-to-Q results above use the trained classifier. Music modules use **PyTorch / NumPy / pretty_midi**. The full workflow requires Diff-BGM, datasets, and model weights.

**[Installation, router training, adapter training, and decoding commands →](docs/USAGE.en.md)**

<a id="code"></a>
## Code Map

| Capability | Implementation |
| --- | --- |
| Gated emotion adaptation | [model.py](Experiment/core_code/emotionsketch/model.py) |
| 16-dimensional sketches and EMOPIA pseudo-labels | [data.py](Experiment/core_code/emotionsketch/data.py) |
| Chinese prompt training and prediction | [prompt_router.py](Experiment/core_code/emotionsketch/prompt_router.py) |
| Adapter training on a frozen backbone | [emotionsketch_adapter_train.py](Experiment/core_code/scripts/emotionsketch_adapter_train.py) |
| MIDI evaluator training / scoring | [MFAE training](Experiment/core_code/scripts/train_emopia_midi_quadrant_evaluator_v3.py) · [Manifest scoring](Experiment/core_code/scripts/score_midi_manifest_v3.py) |
| Quadrant-guided decoding | [Q1/Q2](Experiment/core_code/scripts/guided_decode_rerank_demo.py) · [Q3](Experiment/core_code/scripts/constrained_decode_eval_demo.py) · [Q4](Experiment/core_code/scripts/q4_profile_search_v3.py) |
| Intervention and parameter analysis | [Label intervention](Experiment/core_code/scripts/evaluate_label_intervention_sensitivity.py) · [Parameter count](Experiment/core_code/scripts/summarize_parameter_efficiency.py) |

The repository contains **18 core source files** with project and reproduction documentation. Raw data, weights, candidate MIDI files, logs, and third-party repositories are prepared separately. Reported metrics are archived experiment results; training was not rerun for this documentation update.

## Acknowledgements

The video-music backbone builds on [Diff-BGM](https://github.com/sizhelee/Diff-BGM), and emotion data comes from [EMOPIA](https://github.com/annahung31/EMOPIA). The project-specific [shot-count patch](patches/diffbgm-shot-count.patch) is included. Third-party resources retain their own usage terms; this repository has not specified an open-source license.
