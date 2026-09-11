# 跟着 stanford 动手学大模型

本系列以斯坦福（Stanford）大学公开课 [**CS 336: Language Modeling from Scratch**](https://cs336.stanford.edu/) 为基底，以初学者友好、内容聚焦、讲解接地气的方式，介绍现代大语言模型的原理与代码实现。

整体讲解逻辑遵循 CS 336 课程作业文档，以任务点（Problem）的形式引导大家写出代码块 0、代码块 1……最终拼出大模型训练与推理所需要的一切代码。在讲解内容上会额外补充文档中略过的背景知识、基础概念与代码细节。

## 为什么不看原版课程？

1. **访问门槛**：原版课程需要科学上网（B 站有双语字幕版，有基础的同学强烈建议食用）
2. **基础要求高**：原版默认有基础，讲得干练简洁，无相关基础容易从入门到放弃
3. **周期过长**：原版课程横跨模型、数据、系统等多层，完整看完性价比不高

## 相较于原版的不同点

| 原版 | 本系列 |
|------|--------|
| 默认有基础 | 帮大家把必要基础补上，遵循现学现用 |
| 覆盖面广 | 聚焦最通配的核心内容（如 Transformer 结构），学习周期更短 |
| 纯板书/讲义 | 结合代码示例、配图讲解，降低学习难度 |

## 适合谁看？

- 对大模型感兴趣，想手搓自己大模型的朋友（零基础也能跟上）
- 想短期内掌握大模型核心内容与代码的同学（本科/研 0 / 转码）
- 想做一个大模型相关入门项目（考研复试 / 实习简历项目均可）

## 学完你能获得什么？

1. 入门大模型所需要的基础知识
2. 动手搭建完整大模型系统的基础代码能力（涵盖数据、训练、评估和推理）
3. 一段扎实的项目经历

## 项目结构

```
easy-llm/
├── 代码文档/
│   ├── 第零章_大模型基础/
│   │   └── 0_基础知识.ipynb
│   └── 第一章_构建一个Transformer模型/
│       ├── 1_transformer语言模型/
│       │   ├── 1.2_tensor基本运算.ipynb
│       │   ├── 1.3_基础模块.ipynb
│       │   ├── 1.4_层归一化&旋转编码.ipynb
│       │   ├── 1.4_多头注意力.ipynb
│       │   ├── 1.4_前馈网络.ipynb
│       │   └── 1.5_transformer模型.ipynb
│       └── 2_模型训练/
│           ├── 2.1_训练损失.ipynb
│           ├── 2.2_优化器.ipynb
│           └── 2.3_训练循环.ipynb
├── scripts/
│   ├── transformer.py             # 完整 Transformer 语言模型
│   ├── training_utils.py          # 损失、优化器、调度、数据与 checkpoint 工具
│   └── train_llm.py               # 完整训练入口与 W&B 实验记录
├── DATA/                          # 本地数据目录，不提交到仓库
├── results/                       # 本地 checkpoint 目录，不提交到仓库
├── environment.yml               # Conda 环境配置
└── README.md
```

## 内容导航

### 第零章：大模型基础

- [0 基础知识](代码文档/第零章_大模型基础/0_基础知识.ipynb)

### 第一章：构建 Transformer 模型

模型结构：

- [1.2 Tensor 基本运算](代码文档/第一章_构建一个Transformer模型/1_transformer语言模型/1.2_tensor基本运算.ipynb)
- [1.3 基础模块](代码文档/第一章_构建一个Transformer模型/1_transformer语言模型/1.3_基础模块.ipynb)
- [1.4 层归一化与旋转位置编码](代码文档/第一章_构建一个Transformer模型/1_transformer语言模型/1.4_层归一化&旋转编码.ipynb)
- [1.4 多头注意力](代码文档/第一章_构建一个Transformer模型/1_transformer语言模型/1.4_多头注意力.ipynb)
- [1.4 前馈网络](代码文档/第一章_构建一个Transformer模型/1_transformer语言模型/1.4_前馈网络.ipynb)
- [1.5 Transformer 模型](代码文档/第一章_构建一个Transformer模型/1_transformer语言模型/1.5_transformer模型.ipynb)

模型训练：

- [2.1 训练损失](代码文档/第一章_构建一个Transformer模型/2_模型训练/2.1_训练损失.ipynb)
- [2.2 优化器](代码文档/第一章_构建一个Transformer模型/2_模型训练/2.2_优化器.ipynb)
- [2.3 训练循环](代码文档/第一章_构建一个Transformer模型/2_模型训练/2.3_训练循环.ipynb)

## 环境配置

本项目运行环境为 Conda（推荐使用 [Miniconda](https://docs.anaconda.com/miniconda/)），Python 3.10 + PyTorch 2.5（CUDA 12.1）。

```bash
# 1. 安装 Miniconda（如已安装可跳过）
#    下载地址：https://docs.anaconda.com/miniconda/
#    安装完成后打开终端（Anaconda Prompt 或系统终端）

# 2. 基于 environment.yml 创建环境
conda env create -f environment.yml

# 3. 激活环境
conda activate nanogpt

# 4. 运行代码（以 Jupyter Notebook 为例）
jupyter notebook
```

## 准备数据并启动训练

训练数据不随仓库分发。请将已经 token 化的一维 `.npy` 文件放到项目根目录的 `DATA/` 中，默认文件名为：

```text
DATA/
├── TinyStoriesV2-GPT4-train.npy
└── TinyStoriesV2-GPT4-valid.npy
```

首次使用 W&B 时先完成登录，然后从项目根目录启动训练：

```bash
wandb login
python scripts/train_llm.py
```

可以通过命令行覆盖实验参数，例如：

```bash
python scripts/train_llm.py --batch_size 16 --max_lr 3e-4 --min_lr 3e-5 --seed 42
```

如果没有传入 `--run_name`，脚本会根据数据集、batch size、学习率区间和随机种子自动生成 W&B run 名。也可以手动指定：

```bash
python scripts/train_llm.py --run_name tinystories-baseline --seed 42
```

checkpoint 默认保存在 `results/checkpoints/`，文件名同时包含 run 名、W&B run ID 和训练步数，重复实验不会相互覆盖。

## 参考资料

- [CS 336 课程官网](https://cs336.stanford.edu/)
- [CS 336 课程回放（2026 春季）](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
- [CS 336 课程作业文档](https://github.com/stanford-cs336)
- [Let's reproduce GPT-2 (124M) — Andrej Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU)
