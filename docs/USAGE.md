# 使用与复现

[返回项目首页](../README.md) · [English](USAGE.en.md)

## 先运行文本路由器

在仓库根目录运行，无需下载 Diff-BGM 或安装 PyTorch：

```bash
python3 Experiment/core_code/scripts/train_prompt_q_router.py --help
PYTHONPATH=Experiment/core_code python3 - <<'PY'
from emotionsketch.prompt_router import rule_predict
print(rule_predict("温暖平静的背景音乐"))
PY
```

训练可学习路由器时，自行准备以下 JSONL 文件，每行含 `text`、`label`，可选 `id`；标签使用 `Q1` 至 `Q4`。训练、验证、压力测试集应独立划分。

```text
Experiment/datasets/prompt_q_router/train.jsonl
Experiment/datasets/prompt_q_router/val.jsonl
Experiment/datasets/prompt_q_router/stress_test_gold.jsonl
```

```bash
python3 Experiment/core_code/scripts/train_prompt_q_router.py
```

模型输出到 `Experiment/core_code/checkpoints/prompt_q_router.pkl`。仅加载可信来源的 pickle 和模型权重。

## 准备音乐模型环境

本仓库保留原项目的 `Experiment/core_code/` 层级，脚本据此定位数据和输出目录。请在仓库根目录执行命令。

核心音乐模块直接使用 NumPy、PyTorch、pretty_midi，见 [requirements.txt](../requirements.txt)。该文件是直接依赖清单，完整 Diff-BGM 环境还需遵循上游安装说明；本次发布未重新验证 GPU 训练环境或固定兼容版本组合。

```bash
python3 -m pip install -r requirements.txt
git clone https://github.com/sizhelee/Diff-BGM.git Experiment/code_references/Diff-BGM
git -C Experiment/code_references/Diff-BGM checkout c5a0e8a2d589142ce5951d99c1627fd7d523411c
git -C Experiment/code_references/Diff-BGM apply ../../../patches/diffbgm-shot-count.patch
bash Experiment/core_code/scripts/prepare_remote_layout.sh
```

补丁恢复本项目使用的数据加载修正：将每条样本的镜头计数加入批次。第三方源码不随本仓库分发；上述提交来自本地实验所用的 Diff-BGM 版本。

另外需要：

- 按 [Diff-BGM 上游说明](https://github.com/sizhelee/Diff-BGM#2-training)准备 BGM909/POP909 特征、预训练组件及划分文件。布局脚本只创建目录与链接，不下载数据。
- 准备 [EMOPIA](https://github.com/annahung31/EMOPIA) 数据，将 `EMOPIA_2.2/midis/` 与 `EMOPIA_2.2/CP_events/` 放在 `Experiment/datasets/emopia/` 下。
- 自行训练或准备基线、适配器及 MFAE 权重。完整训练涉及大模型和数据，需相应计算资源。

## 训练与解码入口

以下命令依赖上一节的资源。它们展示原实验的调用方式，并不表示数据或已训练模型包含在仓库中。

```bash
# EMOPIA 象限先验与 MFAE 评价器
python3 Experiment/core_code/scripts/build_emopia_emotion_prior.py
python3 Experiment/core_code/scripts/train_emopia_midi_quadrant_evaluator_v3.py

# 基线与情绪适配器：名称与解码脚本的默认权重路径一致
python3 Experiment/core_code/scripts/baseline_bounded_train.py \
  --steps 1000 --run-name baseline_1000steps
python3 Experiment/core_code/scripts/emotionsketch_adapter_train.py \
  --steps 100 --label-mode emopia_prior \
  --run-name emotionsketch_adapter_emopia_prior_100steps

# 四象限对应的候选生成路径
python3 Experiment/core_code/scripts/guided_decode_rerank_demo.py --labels 0,1
python3 Experiment/core_code/scripts/constrained_decode_eval_demo.py --labels 2
python3 Experiment/core_code/scripts/q4_profile_search_v3.py --profile-mode focused
```

共享的模型加载、去噪与 MIDI 导出实现位于 [export_teacher_forced_demo_grid.py](../Experiment/core_code/scripts/export_teacher_forced_demo_grid.py)。评价器训练所需的公共指标函数保留在 [train_emopia_midi_quadrant_classifier.py](../Experiment/core_code/scripts/train_emopia_midi_quadrant_classifier.py)。标签干预及参数量检查分别见 [evaluate_label_intervention_sensitivity.py](../Experiment/core_code/scripts/evaluate_label_intervention_sensitivity.py) 和 [summarize_parameter_efficiency.py](../Experiment/core_code/scripts/summarize_parameter_efficiency.py)；MIDI 清单评分见 [score_midi_manifest_v3.py](../Experiment/core_code/scripts/score_midi_manifest_v3.py)。

## 运行与评估说明

当前音乐脚本使用 teacher-forced 条件去噪：从参考符号音乐加噪，结合预提取特征生成并筛选候选。MFAE 分数衡量评价器定义下的目标象限匹配；听感验证需要独立的聆听评估。

发布检查覆盖 Python/Shell 语法、项目内脚本依赖、目录定位，以及路由器训练、预测和保存/加载。完整 GPU 训练与音乐生成未在此次文档更新中重跑。

相关资源遵循上游使用条件；本仓库暂未指定开源许可证。
