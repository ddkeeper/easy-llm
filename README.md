本项目为[**【斯坦福 cs336】零基础手搓大模型系列视频**](https://www.bilibili.com/video/BV1NV8t6AE4T/?vd_source=0034741cbe350a95fe85ffdab1bdc34a)的配套项目。

## 系列视频介绍
本系列以斯坦福（Stanford）大学公开课 [**CS 336: Language Modeling from Scratch**](https://cs336.stanford.edu/) 为基底，以初学者友好、内容聚焦、讲解接地气的方式，介绍现代大语言模型的原理与代码实现。

整体讲解逻辑遵循 CS 336 课程作业文档，以任务点（Problem）的形式引导大家写出代码块 0、代码块 1……最终拼出大模型训练与推理所需要的一切代码。在讲解内容上会额外补充文档中略过的背景知识、基础概念与代码细节。

**Q：为什么不直接看原版课程？**
A：原版默认有基础、体系庞大、学习周期长，容易从入门到放弃。

**Q：相比原版有什么不同？**
A：初学者友好（补基础再切入）、内容聚焦核心（如 Transformer 结构）、讲解接地气（配代码与配图）。

**Q：适合谁看？**
A：想动手实现大模型的零基础朋友、想短期掌握核心内容的本科/研 0/转码同学、想做入门项目的同学。

**Q：能收获什么？**
A：入门大模型的基础知识、完整系统的代码能力（数据/训练/评估/推理），以及可用于考研复试、实习简历的项目经历。

## 项目结构

```
easy-llm/
├── 代码文档/
│   ├── 第零章_大模型基础/
│   └── 第一章_构建一个Transformer模型/
│       ├── 1_transformer语言模型/
│       ├── 2_模型训练/
│       ├── 3_实验/
│       └── 补充1_BPE分词器/
├── scripts/
│   ├── transformer.py             # 完整 Transformer 语言模型
│   ├── train_llm.py               # 训练入口与 W&B 实验记录
│   ├── bpe_tokenizer.py           # 字节级 BPE 分词器
│   └── ...
├── data/                          # 数据集
├── results/                       # 模型 checkpoint 与训练好的分词器
├── environment.yml                # Conda 环境配置
├── cs336_assignment1_basics.pdf   # CS336 作业文档
└── README.md
```

## 环境配置

本项目运行环境为 Conda（推荐使用 [Miniconda](https://docs.anaconda.com/miniconda/)），Python 3.10 + PyTorch 2.5（CUDA 12.1）。

```bash
# 1. 安装 Miniconda（如已安装可跳过），下载地址：https://docs.anaconda.com/miniconda/
# 2. 基于 environment.yml 创建环境
cd easy-llm 
conda env create -f environment.yml

# 3. 激活环境
conda activate nanogpt
```

## 数据集

数据放入 `data/<数据集名>/`，对应的分词器放在 `results/tokenizer/<数据集名>-train/`。

- **TinyStories**：[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)。本仓库提供[分词后的 numpy 版本](https://github.com/ddkeeper/easy-llm/releases/tag/data-v1)（词表 10000，uint16），下载后放入 `data/TinyStories/`。
> 该分词器（`results/tokenizer/TinyStoriesV2-GPT4-train/`）已随仓库提供，用法见 [3.3 文本生成](代码文档/第一章_构建一个Transformer模型/3_实验/3.3_文本生成.ipynb)。分词器的实现与训练将在补充 1 里介绍，在学习 3.3 文本生成时可以先用后学。
- **OpenWebText**：[stanford-cs336/owt-sample](https://huggingface.co/datasets/stanford-cs336/owt-sample)（CS336 官方抽样版）。

也可以在终端使用 wget 命令下载原始文本：

```sh
mkdir -p data
cd data

# TinyStories
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_train.txt.gz owt_valid.txt.gz

cd ..
```

## 训练与实验

```bash
conda activate nanogpt
cd easy-llm

# 1) 训练 BPE 分词器（可在脚本里修改相关路径）
python scripts/train_bpe.py

# 2) 用分词器把原始文本编码成训练用的 .npy 数组（可在脚本里修改相关路径）
python scripts/encode_data.py

# 3) 训练一个基准模型（数据集默认取 data/TinyStories/）
python scripts/train_llm.py --amp --device cuda

# 批量跑实验（Windows 上需在 Git Bash 中执行）：
# 先取消 scripts/run_experiments.sh 底部「实验命令」中要跑的行注释，再执行整个文件
bash scripts/run_experiments.sh
```

## 参考资料

- [CS 336 课程官网](https://cs336.stanford.edu/)
- [CS 336 课程回放（2026 春季）](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
- [CS 336 课程作业文档](https://github.com/stanford-cs336)
- [Let's reproduce GPT-2 (124M) — Andrej Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU)
