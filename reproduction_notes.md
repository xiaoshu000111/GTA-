# GTA 复现边界与实现映射

## 已确认来源

- 论文：Min Zeng et al., *GTA: Supervised-Guided Reinforcement Learning for Text Classification with Large Language Models*, EMNLP 2025 Findings, ACL Anthology ID `2025.findings-emnlp.56`, arXiv `2509.12108v2`。
- 截至本次检索，ACL 页面、arXiv HTML 均未提供代码链接；GitHub 以 arXiv ID 搜索无仓库结果。

## 论文明确描述的实现

1. 输出为 `<guess>...</guess><think>...</think><answer>...</answer>`。
2. Guess 内容使用交叉熵监督；除 Guess 外的 Think 与 Answer token 使用 GRPO。
3. 格式正确奖励为 1，最终分类正确奖励为 1，总奖励为两者相加。
4. 每个 prompt 采样 16 个候选；importance-sampling reuse factor 为 4；KL 系数 `beta=0.01`。
5. `lambda_sft=lambda_rl=1`；最大长度 4096；bf16；每卡 batch size 4；训练 3-4 epoch。
6. 若 SFT 与 RL 梯度点积为正，使用总损失；否则丢弃 SFT 梯度，只保留 RL。
7. 参考模型不是静态 base model，而是周期性用当前模型更新。

## 论文未报告、无法严格复现的细节

- 学习率、优化器、warmup、scheduler、weight decay、随机种子。
- GRPO clipping epsilon、参考模型更新周期、最大生成长度、采样温度/top-p。
- SFT Guess 的 teacher-forcing 样本怎样与在线 RL completion 拼成同一批次。
- 分布式采样/更新的具体同步顺序，以及 reuse factor 4 的精确实现。
- 数据清洗、模板中的空格规范、评测解码失败策略。

因此，本仓库把这些值暴露为配置，不能把一次本地运行称为论文数字的严格复现。

## 当前代码覆盖

- `formatting.py`：严格 GTA 格式解析与论文式 prompt。
- `masking.py`：只保留 Guess 内容的 SFT mask；RL mask 覆盖整个 completion（含标签 token），但屏蔽 Guess 内容，严格对应 Figure 2。
- `rewards.py`：格式奖励、准确奖励和总奖励。
- `losses.py`：masked causal CE、组内标准化 advantage、clipped GRPO、论文 KL 估计器。
- `gradients.py`：公式 (11) 的梯度冲突检测与“冲突时仅保留 RL”规则。
- `data.py`：Table 1 四个数据集的元信息与加载入口。
- `scripts/smoke_core.py` 与 `tests/test_core.py`：无需下载模型即可验证核心数学与 mask。
- `trainer.py` 与 `scripts/hf_offline_one_step.py`：对已经生成的候选组执行一次真实 GTA 优化；本地缓存小模型可离线验证。

## 推荐复现层级

1. **Level 0（当前机器可做）**：运行核心单测与 smoke，验证公式、奖励、mask、梯度选择。
2. **Level 1（单卡 24-48GB）**：用 Qwen2.5-0.5B/1.5B、LoRA、Emotion 小子集验证 GTA 比纯 GRPO 更快收敛的趋势。
3. **Level 2（论文级）**：4x L40 48GB/节点、Qwen2.5-3B 全量数据、16 generations、4 次 reuse，复现 Table 2 与 Figure 5/6。

## 本机验证记录（2026-10-09）

- Python 3.9.6、PyTorch 2.8.0、Transformers 4.57.6、PEFT 0.17.1。
- `scripts/smoke_core.py`：奖励为 `[2.0, 1.0]`；构造的冲突梯度点积为 `-4.0`，仅保留 RL；同向梯度点积为 `8.0`，使用合并梯度。
- `python3 -m unittest discover -s tests -v`：4/4 通过。
- 本地缓存 `Qwen/Qwen2.5-0.5B-Instruct` + LoRA 的单步离线验证通过：可训练参数 135,168，SFT loss 0.0601，平均奖励 1.5，梯度点积 0.3503，使用合并梯度。
- 该真实模型 smoke 使用 group size 2、reuse 1、无独立 reference model，目的仅为验证实现链路；它不对应论文实验结果。
