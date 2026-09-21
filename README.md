# 跟着 stanford 动手学大模型

本系列以斯坦福（Stanford）大学公开课 [**CS 336: Language Modeling from Scratch**](https://cs336.stanford.edu/) 为基底，以初学者友好、内容聚焦、讲解接地气的方式，介绍现代大语言模型的原理与代码实现。

整体讲解逻辑遵循 CS 336 课程作业文档，以任务点（Problem）的形式引导大家写出代码块 0、代码块 1……最终拼出大模型训练与推理所需要的一切代码。在讲解内容上会额外补充文档中略过的背景知识、基础概念与代码细节。

## 为什么不看原版课程？

1. ~~看原版课需要科学上网~~（B 站搜 cs336 有双语字幕版，强烈建议有基础的朋友们食用）
2. 原版课程默认有基础，讲的比较干练简洁，对于无相关基础的同学来说容易从入门到放弃
3. 原版课程体系庞大，横跨模型，数据，系统等多层，学习周期长，完整看完的性价比不高

## 我的视频相较于原版有什么不同点吗？
1. 初学者友好：帮大家把必要的基础补上，随后快速切入核心内容，遵循现学现用的原则
2. 内容聚焦：关注原版课程中最通配的那部分内容（比如 transformer 结构），学习周期更短
3. 讲解接地气：结合代码示例、配图等进行讲解，尽量降低大家的学习难度

## 哪些人适合看我的系列视频？
1. 像我一样对大模型感兴趣，想手搓自己大模型的朋友们（零基础的朋友们也能跟上）
2. 想要短期内掌握大模型核心内容与代码的的同学（本科/研 0 的同学、转码的同学）
3. 想要做一个大模型相关的入门项目（非开发方向，不适合用作求职项目）

## 看完我的视频后大家能获得什么？
1. 入门大模型所需要的基础知识
2. 动手搭建完整大模型系统的基础代码能力（涵盖数据、训练、评估和推理）
3. 一段扎实的项目经历（考研复试项目、实习简历项目都可以用）

## 项目结构

```
easy-llm/
├── 代码文档/
│   ├── 第零章_大模型基础/
│   └── 第一章_构建一个Transformer模型/
│       ├── 1_transformer语言模型/
│       ├── 2_模型训练/
│       └── 3_实验/
├── scripts/
│   ├── transformer.py             # 完整 Transformer 语言模型
│   ├── training_utils.py          # 损失、优化器、调度、数据与 checkpoint 工具
│   ├── bpe_tokenizer.py           # 字节级 BPE 分词器
│   ├── train_llm.py               # 训练入口与 W&B 实验记录
│   └── run_experiments.sh         # 实验脚本：lr / batch size 扫描与消融实验
├── data/                          # 数据集
├── results/tokenizer/             # 训练好的 BPE 分词器
├── environment.yml                # Conda 环境配置
├── cs336_assignment1_basics.pdf   # CS336 作业文档
└── README.md
```

## 环境配置

本项目运行环境为 Conda（推荐使用 [Miniconda](https://docs.anaconda.com/miniconda/)），Python 3.10 + PyTorch 2.5（CUDA 12.1）。

```bash
# 1. 安装 Miniconda（如已安装可跳过），下载地址：https://docs.anaconda.com/miniconda/
# 2. 基于 environment.yml 创建环境
conda env create -f environment.yml

# 3. 激活环境
conda activate nanogpt
```

## 数据集

每个数据集一节，数据放入 `data/<数据集名>/`，对应的分词器放在 `results/tokenizer/<数据集名>-train/`。

### 1. TinyStories V2-GPT4

原始数据集：[TinyStories V2-GPT4](https://huggingface.co/datasets/roneneldan/TinyStories)。

本仓库提供的是分词后的版本：训练集与验证集已用课程实现的 BPE 分词器（词表大小 10000）编码为 uint16 的 numpy 数组。

下载地址：https://github.com/ddkeeper/easy-llm/releases/tag/data-v1

下载后放入 `data/TinyStories/`，`scripts/train_llm.py` 的默认路径即可直接命中。

分词器已随仓库提供：`results/tokenizer/TinyStoriesV2-GPT4-train/`，用法见 [3.3 文本生成](代码文档/第一章_构建一个Transformer模型/3_实验/3.3_文本生成.ipynb)。

> 分词器的实现及其训练将在补充部分介绍，我们在这里先用后学。

## 训练与实验

```bash
conda activate nanogpt
cd easy-llm
# 训练一个基准模型（数据集默认取 data/TinyStories/）
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
