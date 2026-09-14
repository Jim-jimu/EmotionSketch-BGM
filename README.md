# EmotionSketch-BGM

**自然语言情绪控制的视频背景音乐生成 · SRTP 核心代码**

本项目研究如何在 Diff-BGM 视频配乐骨干上加入可解释的情绪控制：Prompt-to-Q 路由器将文本映射到情绪象限，EmotionSketch 适配器将符号音乐草图和标签注入视觉条件分支，MFAE 评价器用于 MIDI 情绪评分和候选选择。

这是结题项目的精简研究代码。包含适配器、路由器、评价器及必要的训练与解码入口；数据集、模型权重、实验结果、论文、答辩材料和第三方源码需要另行准备。

## 核心模块

| 模块 | 实现 |
| --- | --- |
| EmotionSketch Adapter | [model.py](Experiment/core_code/emotionsketch/model.py)：16 维草图与四类标签经门控残差注入 512 维视觉条件。 |
| 草图与伪标签 | [data.py](Experiment/core_code/emotionsketch/data.py)：从符号音乐及多模态特征构建分段草图；支持 EMOPIA 象限质心伪标签。 |
| Prompt-to-Q | [prompt_router.py](Experiment/core_code/emotionsketch/prompt_router.py)：字符 n-gram 线性分类器及规则基线，仅依赖 Python 标准库。 |
| MFAE / V3 | [train_emopia_midi_quadrant_evaluator_v3.py](Experiment/core_code/scripts/train_emopia_midi_quadrant_evaluator_v3.py)：48 维 MIDI 特征、近邻与质心分类评估。 |
| 情绪引导解码 | [Q1/Q2 候选排序](Experiment/core_code/scripts/guided_decode_rerank_demo.py)、[Q3 受控导出](Experiment/core_code/scripts/constrained_decode_eval_demo.py)、[Q4 参数搜索](Experiment/core_code/scripts/q4_profile_search_v3.py)。 |

标签顺序为 Q1（正效价、高唤醒）、Q2（负效价、高唤醒）、Q3（负效价、低唤醒）、Q4（正效价、低唤醒），模型编号为 0–3。`proxy` 模式是批内阈值基线，其编号不能直接解释为上述 EMOPIA 象限；正式象限实验使用 `emopia_prior`。

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

核心音乐模块直接使用 NumPy、PyTorch、pretty_midi，见 [requirements.txt](requirements.txt)。该文件是直接依赖清单，完整 Diff-BGM 环境还需遵循上游安装说明；本次发布未重新验证 GPU 训练环境或固定兼容版本组合。

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

共享的模型加载、去噪与 MIDI 导出实现位于 [export_teacher_forced_demo_grid.py](Experiment/core_code/scripts/export_teacher_forced_demo_grid.py)。评价器训练所需的公共指标函数保留在 [train_emopia_midi_quadrant_classifier.py](Experiment/core_code/scripts/train_emopia_midi_quadrant_classifier.py)。标签干预及参数量检查分别见 [evaluate_label_intervention_sensitivity.py](Experiment/core_code/scripts/evaluate_label_intervention_sensitivity.py) 和 [summarize_parameter_efficiency.py](Experiment/core_code/scripts/summarize_parameter_efficiency.py)；MIDI 清单评分见 [score_midi_manifest_v3.py](Experiment/core_code/scripts/score_midi_manifest_v3.py)。

## 实验范围

当前解码入口从已有符号音乐样本加噪后进行条件去噪，并结合阈值、受控导出或候选搜索。它们是 teacher-forced 诊断流程，不能当作任意原始视频输入到音乐的完整采样应用。

适配器提供条件控制接口；最终象限匹配还受到解码策略和评价器引导选择的影响。评价器得分不能直接等同于听众感知的情绪。本仓库不附实验结果，因此不在此声明性能指标。

## 发布检查与来源

此次整理检查了 Python 语法、项目内脚本依赖、目录定位、Shell 语法及文本路由器的训练/预测/保存/加载；完整音乐训练与生成未在此次发布中重跑。

本项目基于 [Diff-BGM](https://github.com/sizhelee/Diff-BGM)，情绪象限及相关训练数据来自 [EMOPIA](https://github.com/annahung31/EMOPIA)。第三方代码、数据和权重的使用条件以各自项目为准。本仓库暂未指定开源许可证。
