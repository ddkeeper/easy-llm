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
│       ├── 2_模型训练/
│       │   ├── 2.1_训练损失.ipynb
│       │   ├── 2.2_优化器.ipynb
│       │   └── 2.3_训练循环.ipynb
│       └── 3_实验/
│           ├── 3.0_训练资源估算.ipynb
│           └── 3.3_文本生成.ipynb
├── scripts/
│   ├── transformer.py             # 完整 Transformer 语言模型
│   ├── training_utils.py          # 损失、优化器、调度、数据与 checkpoint 工具
│   ├── train_llm.py               # 完整训练入口与 W&B 实验记录
│   └── run_experiments.sh         # 实验脚本：lr / batch size 扫描与消融实验
├── data/                          # 数据集目录，从 Release 下载，不提交到仓库
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

模型实验：

- [3.0 训练资源估算](代码文档/第一章_构建一个Transformer模型/3_实验/3.0_训练资源估算.ipynb)
- [3.3 文本生成](代码文档/第一章_构建一个Transformer模型/3_实验/3.3_文本生成.ipynb)

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

## 数据集

本项目使用 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) V2-GPT4，
训练集与验证集都已用课程实现的 BPE 分词器（词表 10000）编码成 uint16 的 numpy 数组：

| 文件 | 大小 | token 数 |
|------|------|----------|
| `TinyStoriesV2-GPT4-train.npy` | 1013.5 MB | 531,386,205 |
| `TinyStoriesV2-GPT4-valid.npy` | 10.2 MB | 5,365,116 |

训练集超过 GitHub 的单文件 100 MB 上限，因此托管在 Release 上，按需下载：

```bash
mkdir -p data/TinyStories && cd data/TinyStories
base=https://github.com/ddkeeper/easy-llm/releases/download/data-v1
curl -LO $base/TinyStoriesV2-GPT4-train.npy
curl -LO $base/TinyStoriesV2-GPT4-valid.npy
```

下载完成后 `scripts/train_llm.py` 的默认路径即可直接命中，不必再传
`--train_data_path` / `--val_data_path`。sha256 校验值见
[Release 说明](https://github.com/ddkeeper/easy-llm/releases/tag/data-v1)。

## 运行训练与实验

脚本约定在**项目根目录**、已激活 `nanogpt` 环境的前提下执行，所有路径都相对项目根目录。

```bash
# 训练一个基准模型（数据集默认取 data/TinyStories/）
python scripts/train_llm.py --amp --device cuda

# 批量跑实验：先取消 run_experiments.sh 底部「实验命令」中要跑的行注释，再执行整个文件
bash scripts/run_experiments.sh
```

实验名会自动拼成 `{实验类型}-lr{峰值}_{谷值}-bs{batch_size}-s{种子}`，训练日志同步到 W&B，
检查点保存到 `results/checkpoints/{数据集}/{实验名}/`。

## 参考资料

- [CS 336 课程官网](https://cs336.stanford.edu/)
- [CS 336 课程回放（2026 春季）](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
- [CS 336 课程作业文档](https://github.com/stanford-cs336)
- [Let's reproduce GPT-2 (124M) — Andrej Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU)
