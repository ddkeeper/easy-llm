# -*- coding: utf-8 -*-
"""
encode_data.py —— 用训练好的 BPE 分词器把原始文本编码成 token ID 数组，存成 npy。

用法（在项目根目录、已激活 nanogpt 环境）：
    python scripts/encode_data.py

编码结果存成 uint16 的 numpy 数组（.npy），供 scripts/train_llm.py 直接加载训练。
数据与分词器路径、输出路径都在下面 __main__ 里改。
"""
import time

import numpy as np

from bpe_tokenizer import Tokenizer


def encode_data(input_path, vocab_filepath, merges_filepath, output_path, special_tokens):
    """把 input_path 里的文本编码成 token ID 序列，写入 output_path（uint16 .npy）。"""
    t_start = time.time()
    tokenizer = Tokenizer()
    tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    # 逐行编码，全部 token 收进一个列表，最后一次性 np.save 存成 uint16 数组。
    with open(input_path, "r", encoding="utf-8") as f:
        ids = list(tokenizer.encode_iterable(text.rstrip("\n") for text in f))
    np.save(output_path, np.array(ids, dtype=np.uint16))

    print(f"token 总数: {len(ids):,}，已写入 {output_path}，耗时 {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    # 在项目根目录、已激活 nanogpt 环境执行：python scripts/encode_data.py
    # 换数据集的话，改下面这些路径即可；特殊 token 名要与 vocab.json 里注册的一致。

    # TinyStories（特殊 token 是 "<|end_of_text|>"）
    input_path = "data/TinyStories/TinyStoriesV2-GPT4-train.txt"
    vocab_filepath = "results/tokenizer/TinyStoriesV2-GPT4-train/vocab.json"
    merges_filepath = "results/tokenizer/TinyStoriesV2-GPT4-train/merge.txt"
    output_path = "data/TinyStories/TinyStoriesV2-GPT4-train.npy"
    special_tokens = ["<|end_of_text|>"]

    # OpenWebText（词表 32000，特殊 token 是 "<|endoftext|>"）
    # input_path = "data/openwebtext/owt_train.txt"
    # vocab_filepath = "results/tokenizer/owt_train/vocab.json"
    # merges_filepath = "results/tokenizer/owt_train/merge.txt"
    # output_path = "data/openwebtext/owt_train.npy"
    # special_tokens = ["<|endoftext|>"]

    encode_data(input_path, vocab_filepath, merges_filepath, output_path, special_tokens)
