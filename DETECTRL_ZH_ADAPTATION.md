# GTA 迁移到 DetectRL-ZH：实现与研究边界

## 1. 检测对象

本适配执行的是**文档级二分类**：

- `0 = 人类撰写`
- `1 = 机器生成`

输出的预测或 `P(AI)`（若未来增加概率头）只能解释为“该文档属于机器类的模型分数/概率”，不能解释为 AI 写作占比、作者贡献比例或学术违规结论。

DetectRL-ZH 的 Attack 是对机器生成文本施加自动化后处理：混入少于 50% 的人类片段、英译再回译、少于 5% 的形近字扰动。它不是“人类原文经过轻度 AI 润色”的直接代理。

## 2. GTA 到检测任务的映射

```text
<guess>人类撰写|机器生成</guess>
<think>基于整篇文本语言模式进行简短复核</think>
<answer>人类撰写|机器生成</answer>
```

- Guess：用 gold 文档标签做监督交叉熵。
- Think/Answer 及格式 token：使用 GRPO。
- RL 仅屏蔽 Guess 的内容 token，与 GTA 论文 Figure 2 对齐。
- 格式合法奖励 `+1`；最终 Answer 正确奖励 `+1`。
- SFT/RL 梯度冲突时按论文公式只保留 RL 梯度。

## 3. 数据适配

官方训练数据为 JSON 数组，每条包含 `text/label/model/source`；dev/test 主要包含 `text/label`，测试标签版另有 `id`。

本地已核验的官方数据快照：

- train：32,400，机器 24,300、人类 8,100，比例 3:1；三个来源各 10,800，三个生成模型各 8,100。
- dev：2,800，机器 1,700、人类 1,100。
- test：11,000，机器 5,499、人类 5,501。

主实验应保留官方训练分布；1:1 下采样只能作为明确标注的消融，不能与主实验混写。

## 4. 评测

主指标是三个场景 Macro-F1 的平均值：

- Normal：ID 1-4000
- Attack：ID 4001-7000
- Varying Lengths：ID 7001-11000

代码还报告 Mixed、Paraphrase、Perturbation 和 64/128/256/512 字符细分。场景划分必须按 `id`，不能依赖 JSON 数组位置。

生成式 GTA 没有天然的连续机器概率。本实现报告 Answer/Guess/Fallback 使用率和严格格式率，不把硬标签或候选投票比例包装成 `P(AI)`。

## 5. 与旧版实验的关系

既有 Qwen2.5-3B LoRA GTA 实验的 DetectRL-ZH 三场景平均 Macro-F1 约为 0.7343；其中 Varying-64 的 AI recall 仅约 0.158，说明短文本是首要失效点。本适配增加严格解析、Attack 细分和长度细分，后续比较必须同时报告：

1. Normal/Attack/Varying 三场景 Macro-F1；
2. 64/128/256/512 字符 AI recall 与 Macro-F1；
3. 生成格式有效率；
4. Guess 到 Answer 的纠错率，而不只是最终总分。

## 6. 当前完成层级

- 已完成：数据加载/校验、中文 prompt、标签映射、奖励复用、严格生成解析、官方分场景评测、真实 Qwen LoRA 单步 smoke。
- 尚未声称完成：Qwen2.5-3B 全量 GRPO 训练和新指标复现。论文级配置需要多卡 GPU；学习率、clip epsilon、reference 更新周期仍是显式假设。

本机真实数据 smoke（Qwen2.5-0.5B-Instruct，LoRA 135,168 个可训练参数，2 个 prompt、每组 2 个候选）得到：SFT loss `1.3725`、平均奖励 `1.5`、SFT/RL 梯度点积 `-2.2627`，因此该步按论文规则仅保留 RL 梯度。这证明中文标签、真实 tokenizer、双损失和冲突裁决链路已经连通，但不是效果实验。
