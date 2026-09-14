<div align="center">

# EmotionSketch-BGM

### 同一视频，不同情绪的符号音乐配乐

情绪草图适配器 · 自然语言情绪控制 · MIDI 评价引导

**简体中文** · [English](README.en.md)

**[核心贡献](#contributions)** &nbsp; / &nbsp; **[方法设计](#method)** &nbsp; / &nbsp; **[实验结果](#results)** &nbsp; / &nbsp; **[快速开始](#quickstart)** &nbsp; / &nbsp; **[代码导航](#code)**

</div>

EmotionSketch-BGM 在冻结的 **Diff-BGM** 上，以轻量情绪草图适配器连接自然语言情绪意图与符号音乐生成。项目围绕三个模块展开：**MFAE 情绪评价、EmotionSketch 条件适配、Prompt-to-Q 文本路由**。

<table align="center">
<tr>
<td align="center" width="300">
<h3>537,121</h3>
<b>可训练参数</b><br>
<sub>门控残差适配器</sub>
</td>
<td align="center" width="300">
<h3>1.295%</h3>
<b>相对骨干参数量</b><br>
<sub>冻结 Diff-BGM</sub>
</td>
<td align="center" width="300">
<h3>32.17% → 33.70%</h3>
<b>候选目标命中率</b><br>
<sub>同预算提升 <b>1.53 pp</b></sub>
</td>
</tr>
</table>

![EmotionSketch 适配器结构：情绪草图与 Q-label 编码、特征融合、门控残差注入及冻结的 Diff-BGM 骨干。](assets/adapter-architecture.png)

*适配器结构图：连续情绪草图与离散标签融合后，经可学习门控注入视觉条件。点击图片可查看原图。*

> **实验依据** · 100 个验证视频片段，Adapter 与 baseline 各生成 20,400 个 MIDI 候选。目标命中率由 MFAE 定义，统计于最佳候选筛选之前；pp 为百分点。[2026 年 5 月实验汇总 →](docs/results-summary.json)

---

<a id="contributions"></a>
## 01 · 核心贡献

### 01 · MFAE MIDI 情绪评价器

基于 EMOPIA MIDI 的 48 维特征训练 KNN 四象限情绪分类器，提供统一的情绪评分依据，用于比较情绪草图适配器与无适配器 baseline，并支持目标 MIDI 候选选择。

### 02 · 参数高效的 EmotionSketch-BGM 情绪草图适配器

将 16 维连续情绪草图与离散 Q-label 嵌入映射到 512 维条件空间，通过可学习门控，以残差形式注入冻结 Diff-BGM 的视觉条件路径。适配器包含 537,121 个可训练参数，占冻结骨干参数量的 1.295%。

### 03 · Prompt-to-Q 自然语言路由器

实现字符 n-gram 与线性 softmax 分类器，将“明亮、兴奋”“紧张、激烈”“忧伤、低落”“平静、舒缓”等自然语言描述转为 Q1–Q4 概率分布，再映射为适配器的情绪标签。路由器使用 Python 标准库运行，无需在线调用大语言模型。

---

<a id="method"></a>
## 02 · 方法设计与 Pipeline

![EmotionSketch-BGM 方法流程：文本路由、草图适配、冻结骨干、MIDI 候选生成与 MFAE 筛选。](assets/pipeline.svg)

*图中展示组件接口关系。当前音乐实验基于预提取特征与参考符号音乐的 teacher-forced 条件去噪；文本路由器输出通过标签接口接入。*

### 1. 将情绪意图转成统一控制信号

| 象限 | 效价 | 唤醒度 | 情绪示例 | 标签编号 |
| --- | --- | --- | --- | :---: |
| Q1 | 正 | 高 | 欢快、兴奋、胜利 | 0 |
| Q2 | 负 | 高 | 紧张、压迫、冲突 | 1 |
| Q3 | 负 | 低 | 悲伤、孤独、忧郁 | 2 |
| Q4 | 正 | 低 | 温暖、平静、舒缓 | 3 |

<details>
<summary>训练标签与编号说明</summary>

训练阶段通过 EMOPIA 象限质心提供伪标签；使用阶段可由 Prompt-to-Q 预测或显式指定目标标签。`proxy` 是批内阈值基线，其编号不直接对应 EMOPIA 象限。

</details>

### 2. 用门控残差适配视觉条件

每个样本构建 **32 × 16** 草图，涵盖音符活动、音区分布、和弦汇总、视觉/字幕特征范数、镜头计数与效价/唤醒代理量。设视觉条件为 `V`、草图为 `S`、目标标签为 `q`：

```math
V' = V + \sigma(g)\,f_{\mathrm{out}}\!\left(f_{\mathrm{sketch}}(S) + \alpha E(q)\right)
```

`E(q)` 在时间维广播，α 对应 `label_scale`，`g` 是可学习门控。输入与输出均为 **[B, 32, 512]**；训练更新适配器，保留骨干去噪目标。

<details>
<summary>参数量明细</summary>

| 项目 | 参数量 |
| --- | ---: |
| EmotionSketch Adapter | **537,121** |
| 冻结 Diff-BGM SDF 骨干 | 41,479,098 |
| Adapter 占比 | **1.295%** |

</details>

### 3. 将情绪特征转成候选选择依据

MFAE 使用 **EMOPIA MIDI 的 48 维特征**建立情绪评分依据，涵盖音高、时值、力度、节奏等统计信息，并通过 KNN 分类器进行 Q1–Q4 情绪分类。

在候选评价阶段，结合邻域概率与质心分数衡量 MIDI 与目标情绪的匹配程度，选择符合情绪意图的候选。MFAE 同时作为统一评价器，比较 Adapter 与 no-adapter baseline 的目标象限命中情况，连接轻量条件控制与可诊断的输出评价。

---

<a id="results"></a>
## 03 · 实验与结果

| 验证问题 | 关键结果 |
| --- | --- |
| **评价器能否识别情绪？** | MFAE：Accuracy **0.6279**，Macro-F1 **0.6285** |
| **Q-label 能否改变条件输出？** | full RMSE **0.099496**；关闭标签分支后为 **0** |
| **候选是否更偏向目标情绪？** | 相同预算下增加 **312** 个目标命中候选 |
| **文本能否映射为情绪标签？** | 100 条清洗验证数据上，Prompt-to-Q Accuracy **1.0000** |

### MFAE：MIDI 四象限情绪分类

**实验设置。** 使用 EMOPIA MIDI 特征训练 KNN 分类器，在验证集上评价四象限情绪识别能力，并将其作为后续 Adapter 对照实验的统一评价器。

| 指标 | 数值 |
| --- | ---: |
| Accuracy | **0.6279** |
| Macro-F1 | **0.6285** |

<details>
<summary>各象限召回率</summary>

| Quadrant | Recall |
| --- | ---: |
| Q1 | 0.7755 |
| Q2 | 0.6038 |
| Q3 | 0.5294 |
| Q4 | 0.6129 |

</details>

Accuracy 衡量分类准确率，Macro-F1 为四类 F1 的宏平均，Recall 为各象限召回率。MFAE 为生成候选提供一致的情绪分类与评价依据。

### 实验 1：Q-label 情绪控制路径是否能够干预生成结果

**实验设置。** 固定同一个视频样本的视觉特征和情绪草图，强制将输入标签切换为 Q1/Q2/Q3/Q4，检查 Adapter 的条件输出是否随标签变化。对比完整适配器、仅标签分支和仅草图分支三种设置。

| Adapter 设置 | Label path | Mean pairwise condition RMSE |
| --- | --- | ---: |
| **full（label + sketch）** | enabled | **0.099496** |
| label-only | enabled | 0.110717 |
| sketch-only | disabled | 0.000000 |

**指标说明。** 对不同 Q-label 下的条件输出进行两两比较，计算均方根误差（RMSE）并取平均。数值越大，表示标签切换对条件输出的影响越大。

**实验结论。** 强制切换 Q1–Q4 标签时，full 和 label-only 的条件输出发生变化，sketch-only 严格不变，说明 **Q-label branch 确实进入了生成条件路径**，提供了可显式干预的情绪控制接口。

### 实验 2：Adapter vs no-adapter，候选分布更偏向目标象限

**实验设置。** 使用 BGM909 的 **100 个验证视频片段**，为每个片段指定目标 Q-label，分别使用 Adapter 与 no-adapter baseline 生成 MIDI 候选。MFAE 对候选进行情绪分类，统计预测象限与目标标签是否一致。两组保持相同候选预算，每组共 **20,400 个 MIDI 候选**；命中率统计于最佳候选筛选之前。

| Target | Budget | Adapter hit / rate | Baseline hit / rate | Delta |
| --- | ---: | ---: | ---: | ---: |
| Q1 | 2,400 | **865 / 0.3604** | 794 / 0.3308 | **+2.96 pp** |
| Q2 | 2,400 | **1,448 / 0.6033** | 1,422 / 0.5925 | **+1.08 pp** |
| Q3 | 4,000 | 4,000 / 1.0000 | 4,000 / 1.0000 | 0.00 pp |
| Q4 | 11,600 | **561 / 0.0484** | 346 / 0.0298 | **+1.85 pp** |
| **Overall** | **20,400** | **6,874 / 0.3370** | **6,562 / 0.3217** | **+1.53 pp** |

**实验结论。** 在 100 个验证片段、每组 20,400 个 MIDI 输出候选上，Adapter 将整体 target-quadrant hit rate 从 **0.3217 提升到 0.3370**，相同预算下增加 **312 个**目标命中候选。结果表明，情绪草图适配器使候选分布更偏向指定的目标象限。

*Overall 按各象限候选数加权；pp 为百分点，差值由原始计数计算。*

### Prompt-to-Q：自然语言情绪标签分类

**实验设置。** 使用 LLM 辅助生成自然语言情绪描述及对应 Q-label，经清洗后按 Q1–Q4 均衡构造数据集。使用 **400 条训练数据**训练轻量路由器，在 **100 条清洗后的验证数据**上与关键词规则 baseline 比较。

| Split | Model | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| validation | **lightweight router** | **1.0000** | **1.0000** |
| validation | rule baseline | 0.6600 | 0.6430 |

**实验结论。** 在该清洗验证集上，轻量路由器准确完成自然语言描述到情绪标签的映射，为用户情绪意图提供结构化入口，并与 Adapter 的 Q-label 控制接口连接。

### 项目结论

EmotionSketch-BGM 将视频配乐的情绪控制组织为三个可验证模块：**MFAE 提供情绪分类，Adapter 提供轻量条件控制路径，Prompt-to-Q 提供自然语言入口**。标签干预实验验证了控制分支的作用，候选对照实验进一步表明该控制路径能够使 MIDI 候选分布更偏向目标情绪。

[实验汇总、混淆矩阵与来源记录 →](docs/results-summary.json)

---

<a id="quickstart"></a>
## 04 · 快速开始

先运行无需额外依赖的中文规则路由示例：

```bash
PYTHONPATH=Experiment/core_code python3 - <<'PY'
from emotionsketch.prompt_router import rule_predict
print(rule_predict("温暖平静的背景音乐"))  # Q4
PY
```

这是规则基线入口；上表中的 Prompt-to-Q 为训练后的分类器。音乐模块使用 **PyTorch / NumPy / pretty_midi**，完整流程需准备 Diff-BGM、数据与模型权重。

**[安装、路由训练、适配器训练和解码命令 →](docs/USAGE.md)**

---

<a id="code"></a>
## 05 · 代码导航

| 功能 | 核心实现 |
| --- | --- |
| 门控情绪适配 | [model.py](Experiment/core_code/emotionsketch/model.py) |
| 16 维草图与 EMOPIA 伪标签 | [data.py](Experiment/core_code/emotionsketch/data.py) |
| 中文提示训练与预测 | [prompt_router.py](Experiment/core_code/emotionsketch/prompt_router.py) |
| 冻结骨干上的适配器训练 | [emotionsketch_adapter_train.py](Experiment/core_code/scripts/emotionsketch_adapter_train.py) |
| MIDI 评价器训练 / 评分 | [MFAE 训练](Experiment/core_code/scripts/train_emopia_midi_quadrant_evaluator_v3.py) · [清单评分](Experiment/core_code/scripts/score_midi_manifest_v3.py) |
| 候选生成与选择 | [候选生成](Experiment/core_code/scripts/guided_decode_rerank_demo.py) · [受控导出](Experiment/core_code/scripts/constrained_decode_eval_demo.py) · [特征搜索](Experiment/core_code/scripts/q4_profile_search_v3.py) |
| 干预与参数分析 | [标签干预](Experiment/core_code/scripts/evaluate_label_intervention_sensitivity.py) · [参数统计](Experiment/core_code/scripts/summarize_parameter_efficiency.py) |

仓库保留 **18 个核心源码文件**和展示/复现说明。原始数据、权重、候选 MIDI、日志及第三方仓库另行准备；实验汇总为已归档结果，不代表本次文档更新重新执行训练。

## 致谢

视频配乐骨干基于 [Diff-BGM](https://github.com/sizhelee/Diff-BGM)，情绪数据来自 [EMOPIA](https://github.com/annahung31/EMOPIA)。本仓库包含项目使用的[镜头计数补丁](patches/diffbgm-shot-count.patch)。第三方资源遵循各自使用条件；本仓库暂未指定开源许可证。

<p align="center">
<a href="#emotionsketch-bgm">回到顶部</a> · <a href="docs/USAGE.md">使用指南</a> · <a href="docs/results-summary.json">实验记录</a>
</p>
