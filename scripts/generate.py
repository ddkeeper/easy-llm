# -*- coding: utf-8 -*-
"""
generate.py —— 文本生成（3.3 解码的实现，供 3.3 / 3.4 的代码文档复用）。

只做一件事：给定模型 + 提示词，自回归采样出新 token，直到碰到结束符或达到长度上限。

对应 PDF Problem (decoding)：温度缩放 + top-p 核采样。
"""
import torch


def decode(model, prompt, eot_token_id=None, max_new_tokens=256,
           temperature=0.5, top_p=0.9, context_length=256) -> list[int]:
    '''
    从语言模型采样生成，返回新生成的 token ID（包含 prompt）

    model: torch.nn.Module  输入 (batch, seq_len)，输出 (batch, seq_len, vocab_size)
    prompt: list[int]  已编码的 token ID 列表
    eot_token_id: int  结束符 <|endoftext|> 的 ID；None 表示不提前停止
    max_new_tokens: int  最多新生成的 token 数
    temperature: float  温度参数 τ，越小分布越尖锐（趋近贪心），越大越平缓
    top_p: float  Top-p 核采样阈值，只保留累计概率达到 p 的那一小撮候选
    context_length: int  模型最大上下文长度，每一步只保留最后这么多个 token 作为输入
    '''
    ids = list(prompt)                               # 提示词长度为 prompt_length
    x = torch.tensor(ids).unsqueeze(0)               # 补上 batch 维：(seq_len,) -> (1, seq_len)

    model.eval()
    torch.set_grad_enabled(False)                    # 全局开关，关闭梯度计算

    # 每轮产出一个 token：前向 → 取最后一个位置上的分布 → 采样 → 拼回输入
    for _ in range(max_new_tokens):
        x_input = x[:, -context_length:]            # 只保留最后 context_length 个 token
        pred_y = model(x_input)[:, -1, :]           # (1, vocab_size) 当前最后一个位置的预测

        # 温度缩放：τ→0 时最大的 logit 一枝独秀（趋近贪心），τ 越大分布越平、越敢选低概率词
        probs = torch.softmax(pred_y / temperature, dim=-1).squeeze(0)  # (vocab_size,)

        # top-p：按概率从大到小排序并求前缀和，累计到 p 的那一段就是要保留的"核"
        sorted_probs, sorted_indices = probs.sort(descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        cutoff_idx = torch.searchsorted(cumsum, top_p)
        sorted_mask = torch.arange(cumsum.shape[-1], device=probs.device) <= cutoff_idx

        # mask 按"排序后顺序"创建，需映射回原始下标再作用到 probs 上
        probs[sorted_indices[~sorted_mask]] = 0
        probs = probs / probs.sum()                  # 丢掉尾部后重新归一化
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        ids.append(next_token_id)
        x = torch.cat([x, torch.tensor([[next_token_id]], device=x.device)], dim=1)
        if eot_token_id is not None and next_token_id == eot_token_id:  # 生成结束符，提前停止
            break

    return ids
