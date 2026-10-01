# -*- coding: utf-8 -*-
"""
train_bpe.py —— 训练字节级 BPE 分词器（补充 1「BPE 训练」的实现）。

包含三个函数：
    preprocess_chunks   预分词 + 计数（按特殊 token 切块 + GPT-2 正则）
    merge_and_update    应用一次 BPE 合并，并同步更新 tokens / stats / pair_to_ids
    train_bpe           把上面两步串起来，训练出一个 byte-level BPE 分词器

用法（在项目根目录、已激活 nanogpt 环境）：
    python scripts/train_bpe.py

训练产物会写到 results/tokenizer/<数据集名>/ 下（vocab.json 与 merge.txt），
供 scripts/bpe_tokenizer.py 里的 Tokenizer.from_files 加载。

注：本文件里的 train_bpe 是逐行串行读文件的实现；真实数据集很大时，
工程实现会按特殊 token 边界把语料切成若干段并行预分词（见 references/pretokenization_example.py
里的 find_chunk_boundaries），这只是性能优化，不改变 BPE 算法本身。
"""
import os
import json
import time
import regex as re
from collections import defaultdict


# GPT-2 的预分词正则（来自 tiktoken，这里不 import，直接复制模式）。
# 匹配顺序是：英文缩写、字母、数字、标点/符号，最后是空白；前面的可选空格
# 会跟在后面的词或标点一起保留。这样 BPE 只在这些片段内部合并，不跨片段连接。
# 其中 \p{L}+ 匹配字母，\p{N}+ 匹配数字，[^...] 匹配标点/符号，\s+ 匹配空白。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def preprocess_chunks(chunk_text: str, special_tokens_pattern: str) -> dict[str, int]:
    """先按特殊 token 切分，再用 GPT-2 正则统计普通片段频次。"""
    chunk_dict = {}
    # 第一步：按特殊 token 边界切分，特殊 token 不进入普通片段，也不参与 pair 统计。
    for chunk1 in re.split(special_tokens_pattern, chunk_text):
        # 第二步：对每个普通片段应用 GPT-2 正则，并统计片段频次。
        for chunk2 in re.findall(PAT, chunk1):
            chunk_dict[chunk2] = chunk_dict.get(chunk2, 0) + 1
    return chunk_dict


def merge_and_update(p0, p1, merge_id, tokens, stats, pair_to_ids):
    """应用一次合并 (p0, p1) -> merge_id，并同步更新 tokens / stats / pair_to_ids。

    tokens      : {字节元组: 频次}，当前所有「片段级」token 序列及其计数
    stats       : {(id_a, id_b): 频次}，全局相邻 pair 的计数
    pair_to_ids : {(id_a, id_b): {含该 pair 的序列集合}}，反向索引，避免每轮全文重扫
    """
    pair_to_ids_add = defaultdict(set)   # 本轮待新增的 pair -> 序列
    pair_to_ids_del = defaultdict(set)   # 本轮待删除的 pair -> 序列
    for ids in pair_to_ids[(p0, p1)]:    # 只处理「包含 (p0, p1)」的序列，其余不受影响
        times = tokens[ids]
        # 把 ids 中所有相邻的 (p0, p1) 替换成 merge_id
        new_ids = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == p0 and ids[i + 1] == p1:
                new_ids.append(merge_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        # 旧序列里的所有 pair 计数减去 times，并登记待删除
        for pair in zip(ids, ids[1:]):
            stats[pair] -= times
            pair_to_ids_del[pair].add(ids)
        # 新序列里的所有 pair 计数加上 times，并登记待新增
        for pair in zip(new_ids, new_ids[1:]):
            stats[pair] = stats.get(pair, 0) + times
            pair_to_ids_add[pair].add(tuple(new_ids))
        # 更新 tokens：新序列替换旧序列
        tokens[tuple(new_ids)] = tokens.get(tuple(new_ids), 0) + times
        tokens.pop(ids)
    # 延迟统一增删索引，避免边遍历 set 边修改 set
    for pair, idss in pair_to_ids_del.items():
        for ids in idss:
            pair_to_ids[pair].discard(ids)
    for pair, idss in pair_to_ids_add.items():
        for ids in idss:
            pair_to_ids[pair].add(ids)


def train_bpe(input_path, vocab_size, special_tokens, output_path=None):
    """训练一个 byte-level BPE 分词器。

    input_path    : str  训练语料文本文件路径
    vocab_size    : int  最终词表大小的上限（含 256 个初始字节 + 特殊 token + 合并出的 token）
    special_tokens: list[str]  要加入词表的特殊 token，训练时作为硬边界、不参与 pair 统计

    返回 (vocab, merges)：
        vocab  : dict[int, bytes]  token ID -> 对应字节串
        merges : list[tuple[bytes, bytes]]  按创建顺序排列的合并规则
    """
    # 1) 词表初始化：256 个字节 + 特殊 token
    vocab = {i: bytes([i]) for i in range(256)}
    for i, st in enumerate(special_tokens):
        vocab[i + 256] = st.encode("utf-8")

    merges = []
    tokens = {}
    # 2) 预分词：按特殊 token 切块 + GPT-2 正则，统计片段计数
    special_pattern = "|".join(re.escape(st) for st in special_tokens) if special_tokens else r"(?!)"
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            for chunk, times in preprocess_chunks(line, special_pattern).items():
                ids = tuple(chunk.encode("utf-8"))
                tokens[ids] = tokens.get(ids, 0) + times

    # 3) 初始化 pair 统计与反向索引
    stats = {}
    pair_to_ids = defaultdict(set)
    for ids, times in tokens.items():
        for p0, p1 in zip(ids, ids[1:]):
            stats[(p0, p1)] = stats.get((p0, p1), 0) + times
            pair_to_ids[(p0, p1)].add(ids)

    # 4) 迭代合并，直到词表满 vocab_size
    t_start = time.time()
    for i in range(len(vocab), vocab_size):
        # max 的 key 用 lambda 为每个 pair 生成比较键：先比频次，平局时再按
        # 左、右 token 的字节序比较，保证每轮选择都是确定的。
        p0, p1 = max(stats, key=lambda k: (stats[k], vocab[k[0]], vocab[k[1]]))
        vocab[i] = vocab[p0] + vocab[p1]
        merges.append((vocab[p0], vocab[p1]))
        merge_and_update(p0, p1, i, tokens, stats, pair_to_ids)
    print(f"训练总耗时: {time.time() - t_start:.2f}s")

    # 5) 后处理：保存 vocab（bytes 转 hex）为 json 文件，保存 merges（每行一对）为 txt 文件；结果目录为：output_path 或者 results/tokenizer/<数据集名>/
    if output_path is not None:
        results_dir = output_path
    else:
        dataset_name = os.path.splitext(os.path.basename(input_path))[0]
        results_dir = os.path.join("./results/tokenizer", dataset_name)

    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({k: v.hex() for k, v in vocab.items()}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(results_dir, "merge.txt"), "w", encoding="utf-8") as f:
        for pair in merges:
            f.write(f"{list(pair)}\n")
    return vocab, merges


if __name__ == "__main__":
    # 在项目根目录、已激活 nanogpt 环境执行：python scripts/train_bpe.py
    # 训练数据与词表大小按需改这里的路径与参数即可（训练产物写到 results/tokenizer/<数据集名>/）。

    # TinyStories：词表 10,000，特殊 token 用于分隔文档
    output_path = "results/Tinystories_try"
    train_bpe("data/TinyStories/TinyStoriesV2-GPT4-train.txt", 300, ["<|endoftext|>"], output_path=output_path) # 标准大小：10000

    # OpenWebText：词表 32,000，同一套代码只改 vocab_size 与输入路径
    # train_bpe("data/openwebtext/owt_train.txt", 32000, ["<|endoftext|>"])
