# -*- coding: utf-8 -*-
"""
bpe_tokenizer.py —— 第 2 章实现的字节级 BPE 分词器，供后续 notebook 直接导入复用。

这里只包含「使用」一个已经训练好的分词器所需的部分：
    Tokenizer.from_files     从 vocab.json / merge.txt 里恢复词表与合并规则
    Tokenizer.encode         文本 -> token ID 列表
    Tokenizer.encode_iterable
    Tokenizer.decode         token ID 列表 -> 文本

训练分词器本身（train_bpe、预分词、pair 计数与合并）属于第 2 章的内容，见对应的 notebook。

用法（本项目里 results/tokenizer/TinyStoriesV2-GPT4-train/ 下有一份 10k 词表）：
    tokenizer = Tokenizer()
    tokenizer.from_files(".../vocab.json", ".../merge.txt", ["<|end_of_text|>"])
    ids = tokenizer.encode("Once upon a time")
    print(tokenizer.decode(ids))      # 还原回原文
"""
import ast
import json
import regex as re
from typing import Iterable, Iterator


# GPT-2 的预分词正则：先把文本切成「英文缩写 / 词 / 数字 / 标点 / 空白」若干片段，
# 再在片段内部做 BPE 合并。BPE 的合并规则只在单个片段内生效，不会跨片段合并。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer():
    def __init__(self, vocab=None, merges=None, special_tokens=None):
        '''
        vocab: dict[int, bytes]  词表，token ID -> 该 token 对应的字节串
        merges: list[tuple[bytes, bytes]]  按创建顺序排列的合并规则
        special_tokens: list[str] | None  特殊 token（如 "<|end_of_text|>"）：
            它们永远不会被切开，始终整体作为一个 token
        '''
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = list(special_tokens) if special_tokens else []
        self._build_pattern()

        self.vocab_rev = None   # 反向词表：字节串 -> token ID
        if self.vocab:
            self.vocab_rev = {v: k for k, v in self.vocab.items()}

    def _build_pattern(self):
        '''把特殊 token 拼成一个正则，供 re.split 用。

        注意要优先匹配更长的特殊 token：比如 "<|end_of_text|><|end_of_text|>" 应该先被当成
        两个特殊 token，而不是被短的模式先切走一小段。
        '''
        if self.special_tokens:
            escaped = [re.escape(st) for st in sorted(self.special_tokens, key=len, reverse=True)]
            self.pattern = "|".join(escaped)
        else:
            # 没有任何特殊 token 时给一个必定匹配失败的正则，使得 re.split 等价于不切分
            self.pattern = r"(?!)"

    # ------------------------------------------------------------ 载入已训练好的分词器
    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        '''从 train_bpe 保存下来的 vocab.json 与 merge.txt 恢复分词器。'''
        if special_tokens:
            self.special_tokens = list(special_tokens)
            self._build_pattern()

        # 读 vocab：保存时 bytes 被转成了十六进制字符串，这里转回来
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.vocab = {int(k): bytes.fromhex(v) for k, v in raw.items()}
        self.vocab_rev = {v: k for k, v in self.vocab.items()}

        # 读 merges：每行形如 [b' ', b't']，用 literal_eval 安全地解析回 (bytes, bytes)
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    merges.append(tuple(ast.literal_eval(line)))
        self.merges = merges

    # ------------------------------------------------------------ 编解码
    def _get_stats(self, ids):
        '''统计相邻 token 对的出现次数：{(id_a, id_b): 次数}'''
        stats = {}
        for pair in zip(ids, ids[1:]):
            stats[pair] = stats.get(pair, 0) + 1
        return stats

    def _merge(self, ids, pair, idx):
        '''把 ids 中所有相邻的 pair 替换成合并后的新 ID idx'''
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def encode(self, text: str) -> list[int]:
        '''文本 -> token ID 列表'''
        encoded_ids = []
        # 预分词第一步：按特殊 token 切分。这里用捕获组 "(...)"，让特殊 token 本身也保留在结果里
        # （训练时不保留，因为特殊 token 不参与合并；但编码时要保留，否则解码就还原不出来了）
        for chunk1 in re.split("(" + self.pattern + ")", text):
            # 当前块本身就是特殊 token：直接查表取 ID，不做任何字节切分与合并
            if self.special_tokens and chunk1 in self.special_tokens:
                encoded_ids.append(self.vocab_rev[chunk1.encode("utf-8")])
                continue

            # 预分词第二步：GPT-2 正则，在片段内部再做 BPE
            for chunk2 in re.findall(PAT, chunk1):
                byte_seq = chunk2.encode("utf-8")

                # 初始状态：每个字节各是一个 token，用反向词表查出它们各自的基础 ID
                # （不能直接把 byte 当成整数 ID：前 256 个 ID 恰好与字节值一一对应，
                #   但特殊 token 插在 256 号位之后，词表未必满足 id == byte 的关系）
                ids = [self.vocab_rev[byte_seq[i:i + 1]] for i in range(len(byte_seq))]

                # 反复找出「合并优先级最高」的相邻对并合并，直到片段内没有任何可合并的对
                while len(ids) > 1:
                    stats = self._get_stats(ids)
                    # 合并优先级 = 该对合并后得到的新 token ID，ID 越小说明它越早被合并出来、优先级越高；
                    # 用 vocab_rev 查不到的（不在词表里的对）给无穷大，等价于不可合并
                    top_pair = min(
                        stats,
                        key=lambda p: self.vocab_rev.get(
                            b"".join([self.vocab[p[0]], self.vocab[p[1]]]), float("inf")
                        ),
                    )
                    top_pair_bytes = b"".join([self.vocab[top_pair[0]], self.vocab[top_pair[1]]])

                    if top_pair_bytes not in self.vocab_rev:   # 词表里没有这个合并结果，收工
                        break

                    ids = self._merge(ids, top_pair, self.vocab_rev[top_pair_bytes])
                encoded_ids.extend(ids)
        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''逐条编码一个可迭代对象（如按行读取的大文件），用生成器避免一次性把结果拼成大列表'''
        for text in iterable:
            for id in self.encode(text):
                yield id

    def decode(self, ids: list[int]) -> str:
        '''token ID 列表 -> 文本'''
        tokens = b"".join(self.vocab[idx] for idx in ids)
        # errors="replace"：如果收到的是一串不完整的字节序列（模型偶尔会生成非法 token 组合），
        # 用替换字符兜底而不是直接抛异常，方便调试采样结果
        return tokens.decode("utf-8", errors="replace")
