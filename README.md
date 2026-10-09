# GTA 文本分类复现

这是对 EMNLP 2025 Findings 论文 **GTA: Supervised-Guided Reinforcement Learning for Text Classification with Large Language Models** 的独立、可审计复现骨架。作者没有公开代码，因此本仓库区分：

- 论文明确给出的算法与超参数；
- 论文未报告、只能作为假设配置的实现细节；
- 当前 Mac 可验证的核心数值行为；
- 需要多卡 GPU 才能执行的论文级实验。

论文原文：<https://aclanthology.org/2025.findings-emnlp.56/>

DetectRL-ZH 官方任务仓库：<https://github.com/NLP2CT/NLPCC-2025-Task1>

## DetectRL-ZH 中文大模型文本检测适配

当前仓库已经加入 DetectRL-ZH 专用适配，不再只是 SST-5 等通用分类：

- 文档级标签：`0=人类撰写`、`1=机器生成`；
- 官方 JSON 数据加载与 3:1 训练分布审计；
- 中文 Guess-Think-Answer prompt；
- Normal / Attack / Varying Lengths Macro-F1；
- Mixed / Paraphrase / Perturbation 与 64/128/256/512 字符细分；
- 严格格式率、Answer/Guess/Fallback 解析统计。

详细映射与研究边界见 [DETECTRL_ZH_ADAPTATION.md](DETECTRL_ZH_ADAPTATION.md)。

仓库内已固定本次可行性实验使用的数据副本：

```text
data/detectrl_zh/train.json
data/detectrl_zh/dev.json
data/detectrl_zh/test_with_label.json
```

来源、用途限制和 SHA-256 见 [data/detectrl_zh/README.md](data/detectrl_zh/README.md)。
其中 `dev.json` 仅用于调参与模型选择，不能混入训练；`test_with_label.json`
仅用于最终分析，不能参与优化或阈值选择。

本地中文检测单步验证：

```bash
PYTHONPATH=src python3 scripts/hf_detectrlzh_one_step.py \
  --local-files-only \
  --data /path/to/DetectRL-ZH/train.json
```

Colab 中可以直接克隆代码与本次固定数据：

```bash
!git clone https://github.com/xiaoshu000111/GTA-.git
%cd GTA-
!pip install -e '.[train]'
```

部分 Colab 镜像预装的 `torchao==0.10.0` 与新版 PEFT 不兼容；若出现
“only versions above 0.16.0 are supported”，先执行：

```bash
!pip uninstall -y torchao
```

小规模真实生成 rollout 实验（每步都把 LoRA、优化器状态、生成文本和指标
保存到 Google Drive，可从 `checkpoints/step_XXXX` 恢复）：

```bash
!PYTHONPATH=src python scripts/hf_detectrlzh_microtrain.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/detectrl_zh/train.json \
  --output-dir /content/drive/MyDrive/GTA_AIGTD/runs/<RUN_ID>/microtrain_05b \
  --device cuda --num-samples 8 --steps 4 --batch-size 2 --group-size 4
```

这个脚本用于验证真实策略采样、奖励差异、GTA 更新和断点恢复，不代表已得到
可报告的 DetectRL-ZH 检测性能；正式结论仍需固定训练方案并在 dev/test 上评测。

已有预测的官方分场景评测：

```bash
PYTHONPATH=src python3 scripts/evaluate_detectrlzh.py \
  --data /path/to/test_with_label.json \
  --result /path/to/result.json \
  --out outputs/detectrl_metrics.json
```

`result.json` 可以包含逐样本 `preds`，也可以包含 GTA `completions`；后一种情况会同时统计格式解析失败率。

## 方法对应关系

模型依次输出：

```text
<guess>初步标签</guess><think>反思与推理</think><answer>最终标签</answer>
```

- `Guess` 内容：监督交叉熵（SFT）。
- 整个 completion（包含 GTA 标签 token）：GRPO，但 Guess 的内容 token 从 RL loss 中屏蔽。
- 奖励：格式正确 `+1`，最终标签正确 `+1`。
- 梯度冲突：SFT/RL 梯度点积 `> 0` 时相加，否则只保留 RL。

完整的论文参数、缺失参数与复现边界见 [reproduction_notes.md](reproduction_notes.md)。

## 本地验证

GTA 损失核心使用 PyTorch，DetectRL-ZH 评测使用 NumPy 与 scikit-learn：

```bash
python3 -m pip install -e .
python3 scripts/smoke_core.py
python3 -m unittest discover -s tests -v
```

smoke 输出应满足：正确且格式合法的 completion 奖励为 `2`，错误但格式合法的 completion 奖励为 `1`；构造的冲突梯度只保留 RL。

如果本机已经缓存 `Qwen/Qwen2.5-0.5B-Instruct`，可以对真实因果语言模型做一次完全离线的 LoRA 更新：

```bash
PYTHONPATH=src python3 scripts/hf_offline_one_step.py --local-files-only
```

这一步验证真实 tokenizer、Guess/RL token mask、SFT/GRPO 双前向和梯度冲突选择能够连通；它不是论文规模训练。

## 论文级环境

论文使用 Qwen2.5-3B、Qwen3-4B、Llama3.2-3B，并在每节点 4 张 NVIDIA L40S 48GB、DeepSpeed ZeRO-2 环境运行。其明确的 Qwen2.5/SST-5 参数已经写入：

```text
configs/paper_qwen25_3b_sst5.yaml
```

训练依赖可用以下命令安装：

```bash
python3 -m pip install -e '.[train]'
```

注意：论文没有报告学习率、GRPO clip epsilon、参考模型更新频率等参数；配置中这些值均标为 `ASSUMPTION`，不能据此声称严格复现论文表格数字。

## 数据集

| 配置名 | Hugging Face ID | 论文训练/测试规模 | 类别数 |
|---|---|---:|---:|
| `sst5` | `SetFit/sst5` | 8,544 / 2,210 | 5 |
| `amazon_massive` | `mteb/amazon_massive_scenario` | 11,514 / 2,974 | 18 |
| `emotion` | `dair-ai/emotion` | 16,000 / 2,000 | 6 |
| `bbc_news` | `SetFit/bbc-news` | 1,225 / 1,000 | 5 |

`gta_repro.data.load_paper_dataset()` 会检查 train/test split 与字段，但第三方托管数据集可能更新，正式实验前应固定 revision 并记录数据 SHA。
