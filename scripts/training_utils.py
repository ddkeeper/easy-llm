# -*- coding: utf-8 -*-
"""
training_utils.py —— 汇总本节之前实现好的组件，供 2.3 及之后的 notebook 直接导入复用。

内容来源：
    cross_entropy           2.1 交叉熵损失
    cal_perplexity          2.1 困惑度
    SGD                     2.2.1 随机梯度下降
    AdamW                   2.2.2 优化器
    learning_rate_schedule  2.2.3 带预热的余弦学习率调度
    gradient_clipping       2.2.4 梯度裁剪
    get_batch               2.3.1 数据加载
    save_checkpoint         2.3.2 检查点保存
    load_checkpoint         2.3.2 检查点加载

具体推导与示例见对应的 notebook，这里只保留可直接调用的实现。
"""
import os
import math
import typing
import numpy as np
import torch

from collections.abc import Callable, Iterable
from typing import Optional
from torch.nn import Module
from torch.optim import Optimizer


# ---------------------------------------------------------------- 2.1 交叉熵损失
# Problem 11: 实现交叉熵损失 (1 分)
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''
    logits: (..., seq_len, vocab_size), 通常为 (batch_size, seq_len, vocab_size)
    targets: (..., seq_len) (B, S)
    '''
    # 交叉熵损失的计算：-log softmax(o)[target] = -(o[target] - max(o)) + log Σ exp(o - max(o))
    # 先减最大值再取 exp 避免上溢，log 与 exp 相互抵消

    # 1.默认不保留归约后的维度；2. max 返回的是元组 (最大值, 最大值对应的下标)
    max_logits = logits.max(dim=-1)[0] # (B, S, V) -> (B, S)

    # max_logits.unsqueeze(-1) -> (B, S, 1); 等价于 max_logits[..., None]
    # max_logits.unsqueeze(-1).squeeze(-1) -> (B, S); squeeze(-1)：删除最后维度，仅当该维度大小等于 1

    # 在最后一维收集目标标签对应的 logit 值: logits[..., targets]; logits[targets] 只能在第一维取元素，与需求不符
    target_logit = torch.gather(logits, -1, targets[..., None]).squeeze(-1) # (B, S, V), (B, S, 1) -> (B, S)

    term1 = -(target_logit - max_logits) # (B, S), (B, S)  -> (B, S)

    exp_sum = torch.exp(logits - max_logits[..., None]).sum(dim=-1) # (B, S, V), (B, S, V) -> (B, S)

    # 按照题目要求需要直接返回平均损失 loss.mean()，这里为了方便示例的比较，返回的是各个位置的损失
    return term1 + torch.log(exp_sum) # (B, S), (B, S) -> (B, S)


# 困惑度 perplexity：交叉熵用于训练足够了，评估模型时更习惯报告困惑度
# 可以理解为"平均意义下模型在多少个 token 之间犹豫"，数值越小越好
def cal_perplexity(loss: torch.Tensor) -> torch.Tensor:
    return torch.exp(loss.mean(dim=-1)) # (B, S) -> (B,)：先对序列维取平均，再取指数


# ---------------------------------------------------------------- 2.2 优化器
# Problem 12: 学习率超参调优 (1 分)
class SGD(torch.optim.Optimizer):
    '''
    PyTorch 中的优化器统一继承 torch.optim.Optimizer，需要实现两个方法：
    __init__(self, params, ...): 使用基类的 init 方法保存待优化参数和默认超参；
    step(self): 执行一次参数更新
    '''

    def __init__(self, params, lr=1e-3):
        """
        自定义SGD优化器构造函数
        :param params: 待优化的参数集合，可以是模型全部参数，也可以是分组参数
        :param lr: 学习率，超参数
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr} # 以字典的形式存储超参名（key）与超参默认值（value)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        执行单步参数更新，反向传播完成 grad 计算之后调用此函数
        :param closure: 可选的闭包函数，用于重新计算 loss 并返回 loss 值，部分优化器需要使用（我们不用）
        :return: 如果传入 closure，返回 loss，否则返回 None
        """
        loss = None if closure is None else closure()
        for group in self.param_groups: # 遍历不同的参数组: 不同组可以指定不同的超参数，比如第一层参数用小一点的 lr，最后一层用大一点的 lr
            lr = group["lr"] # group 也是一个字典，取其中的 lr
            for p in group["params"]: # 遍历该组内的每个待优化参数 p
                if p.grad is None: # 梯度还没有计算出来，grad 属性为 None，跳过更新
                    continue

                state = self.state[p]  # self.state 字典：保存每个参数对应的状态信息，如迭代步数、动量等；state[p] 获取参数 p 对应的状态字典
                t = state.get("t", 0)  # 读取迭代计数 t；如果key "t" 不存在，返回默认值 0
                grad = p.grad.data  # 获取损失 loss 关于参数 p 的梯度张量
                p.data -= lr / math.sqrt(t + 1) * grad  # 原地更新参数：学习率随迭代步数做平方根衰减
                state["t"] = t + 1  # 更新状态里的迭代计数，步数 +1

        return loss


# Problem 13:  实现 AdamW (2 分)
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr:float = 1e-3, betas:tuple[float, float] = (0.9, 0.999), eps:float = 1e-8, weight_decay:float = 0.0):
        '''
        lr: float  α, learning‑rate 学习率
        betas: tuple[float, float] (β1, β2), 一阶矩、二阶矩估计的指数衰减系数
        eps: float ε, 保证数值稳定性的小常数
        weight_decay: float λ, weight‑decay 权重衰减系数
        '''
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        # defaults 字典保存优化器的默认超参数
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}

        # 调用父类 Optimizer 的构造函数
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        执行单步 AdamW 参数更新，反向传播计算完梯度后调用
        :param closure: 可选闭包函数，用于重计算 loss
        :return: loss，若传入 closure 返回 loss，否则返回 None
        """
        # 如果提供 closure，则执行闭包获取 loss，否则 loss 置为 None
        loss = None if closure is None else closure()

        # 遍历每一组参数 group
        for group in self.param_groups:
            # 获取本组参数使用的各种超参
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # 遍历当前参数组内每一个待优化参数 p
            for p in group["params"]:
                # 如果该参数没有梯度张量，跳过更新
                if p.grad is None:
                    continue
                state = self.state[p]  # 获取参数 p 对应的状态字典，保存 m/v/t 等状态变量
                t = state.get("t", 0)  # 从状态字典读取迭代步数，初值为 0
                m = state.get("m", 0) # 一阶矩向量（first moment vector）
                v = state.get("v", 0) # 二阶矩向量（second moment vector）
                grad = p.grad.data  # 获取损失关于参数 p 的梯度张量

                t += 1 # AdawW 里面 t 的取值范围为 1~T，代码里它的初值是 0，所以需要先加 1，再执行更新

                # 执行 AdamW 单步更新
                p.data -= lr * weight_decay * p.data # AdamW 的权重衰减：单独对权重做衰减

                m_t = beta1 * m + (1 - beta1) * grad # 更新一阶矩估计
                v_t = beta2 * v + (1 - beta2) * grad ** 2 # 更新二阶矩估计
                m_hat = m_t / (1 - beta1 ** t) # 用 α 修正因子的分母调节一阶矩估计
                v_hat = v_t / (1 - beta2 ** t) # 用 α 修正因子的分子调节二阶矩估计
                p.data -= lr * m_hat / (torch.sqrt(v_hat) + eps) # 执行 moment-adjusted 的权重更新

                state["t"] = t  # 更新状态字典中的迭代步数
                state["m"] = m_t # 保存最新一阶矩到状态
                state["v"] = v_t # 保存最新二阶矩到状态

        return loss


# Problem 14:  实现带预热的余弦学习率调度 (1 分)
def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    '''
    it: int  当前迭代步数 t (t 从 0 开始); 实际训练时 t 应该是从 1 开始的
    max_learning_rate: float  α_max，最大学习率
    min_learning_rate: float  α_min，最小（最终）学习率
    warmup_iters: int  T_w，预热迭代数
    cosine_cycle_iters: int  T_c，余弦退火的结束步数
    return: float，第 it 步使用的学习率 α_t
    '''

    # 预热阶段：学习率从 0 线性升到 max_learning_rate
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    # 余弦退火阶段：按余弦曲线从 max_learning_rate 衰减到 min_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    # 退火结束：保持 min_learning_rate
    else:
        return min_learning_rate


# Problem 15:  实现梯度裁剪 (1 分)
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps:float = 1e-6) -> None:

    # model.parameters() 返回一次性迭代器；先保存下来，后续才能分别计算范数和缩放梯度
    parameters = [p for p in parameters if p.grad is not None]

    # 把所有参数的梯度当成一个大向量，计算整体的 ℓ2 范数
    # 1. p.grad.pow(2).sum().item(): 对单个参数 梯度张量内的所有元素求平方和，得到该参数张量 梯度的平方和
    # 2. sum(... for p in parameters): 对所有参数张量的梯度平方和求和，得到所有参数的梯度的平方和
    l2_norm = math.sqrt(sum(p.grad.pow(2).sum().item() for p in parameters))

    # 只在范数超过阈值时整体按比例缩放，方向保持不变（原地修改）
    if l2_norm > max_l2_norm:
        for p in parameters:
            with torch.no_grad():
                p.grad *= max_l2_norm / (l2_norm + eps)


# ---------------------------------------------------------------- 2.3 训练循环
# Problem 16: 实现数据加载 (2 分)
def get_batch(x: np.ndarray,
              batch_size: int,
              context_length: int,
              device: str) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    x: np.ndarray  integer array with token IDs  一维整数数组，元素为 token ID
    batch_size: int  B，批大小
    context_length: int  m，每条样本的上下文长度
    device: str  device string such as 'cpu' or 'cuda:0'  设备字符串
    returns (input_seq, target_seq), shape (batch_size, context_length)
    '''
    # 随机采样 B 个起始下标：取值范围 [0, len(x) - m)，保证往后取 m 个 token 不会越界
    indices = np.random.randint(0, len(x) - context_length, size=batch_size)

    # indices[:, None] 将 indices 的形状从 (B,) 变成 (B, 1), 与 (L,) 进行广播相加后形状变为 (B, L)
    input_seq = x[indices[:, None] + np.arange(0, context_length)]   # (B, m)

    indices += 1 # target_seq 的起始下标全部加 1
    target_seq = x[indices[:, None] + np.arange(0, context_length)]  # (B, m)

    # 转成 torch 张量并放到指定设备上
    input_seq_tensor = torch.tensor(input_seq, device=device)
    target_seq_tensor = torch.tensor(target_seq, device=device)

    return (input_seq_tensor, target_seq_tensor)


# Problem 17: 实现模型检查点保存与加载 (1 分)
def save_checkpoint(model: Module,
                    optimizer: Optimizer,
                    iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    '''
    model: torch.nn.Module  模型
    optimizer: torch.optim.Optimizer  优化器
    iteration: int  当前的迭代步数
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]  保存路径或文件对象
    '''
    # state_dict() 返回保存全部可学习权重的字典；优化器同理（AdamW 的 m / v 等状态也在里面）
    torch.save(
        {
            "model": model.state_dict(),      # 模型权重
            "optim": optimizer.state_dict(),  # 优化器状态
            "iter_num": iteration,            # 迭代步数：恢复学习率调度要用
        }
        , out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: Module,
                    optimizer: Optimizer = None) -> int:
    '''
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]  检查点路径或文件对象
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    returns: int，检查点中保存的迭代步数
    '''
    device = next(model.parameters()).device   # 加载到模型当前所在的设备上
    state_dic = torch.load(src, map_location=device, weights_only=True)
    model.load_state_dict(state_dic["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state_dic["optim"])

    return state_dic["iter_num"]
