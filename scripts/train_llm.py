# -*- coding: utf-8 -*-
"""
2.3.3 训练脚本：把各节实现的组件组装成一个完整的训练循环。

结构（先单步、再循环）：
    train_step(...)  单步训练：设置学习率 → 前向传播 → 计算损失 → 反向传播 → 梯度裁剪 → 更新参数
    train_llm(...)   训练循环：反复调用 train_step，中间穿插学习率调度、验证集评估、检查点保存与日志记录

运行示例：
    python train_llm.py                                  # 用默认超参与 TinyStories 数据训练
    python train_llm.py --total_iters 200 --batch_size 8 # 覆盖部分超参做对比实验
"""

import os
import time
import argparse
import numpy as np
import torch
import wandb

from training_utils import (
    cross_entropy,
    AdamW,
    learning_rate_schedule,
    gradient_clipping,
    get_batch,
    save_checkpoint,
)
from transformer import TransformerLM


DATA_DIR = "DATA" # 默认数据路径：项目根目录下的 DATA/

# ------------------------------------------------------------------ 单步训练
def train_step(model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               x: torch.Tensor,
               y: torch.Tensor,
               lr: float,
               max_l2_norm: float = 1.0) -> float:
    '''
    执行一步训练，返回本步的损失值（标量）

    model: torch.nn.Module  语言模型
    optimizer: torch.optim.Optimizer  优化器
    x: (batch_size, context_length)  输入 token ID，需为 long 类型
    y: (batch_size, context_length)  目标 token ID，即 x 整体右移一格
    lr: float  本步使用的学习率（由余弦调度给出）
    max_l2_norm: float  M，梯度裁剪允许的最大 ℓ2 范数
    '''
    optimizer.param_groups[0]["lr"] = lr   # 余弦调度给出的学习率，必须在 step 之前写进去

    logits = model(x)                      # (B, m, vocab_size) 前向传播
    loss = cross_entropy(logits, y).mean() # 对所有位置取平均，得到一个标量损失

    optimizer.zero_grad()                          # 清空上一步的梯度
    loss.backward()                                # 反向传播，计算梯度
    gradient_clipping(model.parameters(), max_l2_norm)  # 梯度裁剪
    optimizer.step()                               # 更新参数

    return loss.item()


# ------------------------------------------------------------------ 训练循环
def train_llm(model: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              train_data: np.ndarray,
              val_data: np.ndarray,
              args) -> None:
    '''
    训练循环：反复调用 train_step，并按间隔做验证、保存检查点与记录日志

    train_data / val_data: np.ndarray  token ID 数组（大数据集建议用 np.load(路径, mmap_mode="r") 打开）
    args: 超参集合，字段见 parse_args()
    '''

    run = wandb.init(project="easy-llm", name=args.run_name, config={
        "dataset_name": args.dataset_name,
        "batch_size": args.batch_size,
        "max_lr": args.max_lr,
        "min_lr": args.min_lr,
        "seed": args.seed,
    })
    os.makedirs(args.save_dir, exist_ok=True)

    start_time = time.time()

    for i in range(1, args.total_iters + 1):
        model.train()   # 切到训练模式（启用 dropout 等）

        # 1. 取一批训练数据，并算出本步的学习率
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        lr = learning_rate_schedule(i, args.max_lr, args.min_lr,
                                    args.warmup_iters, args.cosine_cycle_iters)

        # 2. 单步训练：前向 → 损失 → 反向 → 裁剪 → 更新
        loss = train_step(model, optimizer, x.long(), y.long(), lr=lr, max_l2_norm=args.max_l2_norm)

        # 3. 同一步的训练/验证指标放进同一条记录，避免 W&B 横轴错位
        metrics = {"train_loss": loss, "lr": lr}

        # 4. 定期在验证集上评估：torch.no_grad() 下不建图，也不更新参数
        if i % args.val_interval == 0:
            model.eval()   # 切到评估模式（关闭 dropout 等）
            with torch.no_grad():
                val_x, val_y = get_batch(val_data, args.batch_size, args.context_length, args.device)
                val_loss = cross_entropy(model(val_x.long()), val_y.long()).mean().item()
            metrics["val_loss"] = val_loss
            print(f"iter {i:5d}  val_loss  ={val_loss:.4f}")

        metrics["wall_time"] = time.time() - start_time
        run.log(metrics, step=i)

        # 5. 定期保存检查点（最后一步一定保存）
        if i % args.save_interval == 0 or i == args.total_iters:
            checkpoint_path = os.path.join(
                args.save_dir, f"{args.run_name}-{run.id}-iter{i:06d}.pt"
            )
            save_checkpoint(model, optimizer, i, checkpoint_path)
            print(f"iter {i:5d}  已保存检查点 → {checkpoint_path}")

    run.finish()


# ------------------------------------------------------------------ 超参与入口
def parse_args():
    parser = argparse.ArgumentParser(description="训练一个 Transformer 语言模型")

    # 数据
    parser.add_argument("--dataset_name", type=str, default="TinyStoriesV2-GPT4")
    parser.add_argument("--train_data_path", type=str, default=os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.npy"))
    parser.add_argument("--val_data_path", type=str, default=os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.npy"))

    # 优化器超参
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)

    # 训练超参
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_iters", type=int, default=5000)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--cosine_cycle_iters", type=int, default=5000)
    parser.add_argument("--max_l2_norm", type=float, default=1.0)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="results/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run 显示名称；不传时根据数据集、batch size、学习率区间和 seed 自动生成")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 模型结构基准值（CS336 Assignment 1, §7.2）
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.run_name is None:
        args.run_name = (
            f"{args.dataset_name}-bs{args.batch_size}"
            f"-lr{args.min_lr:.0e}_{args.max_lr:.0e}-s{args.seed}"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # mmap_mode="r"：以内存映射方式打开，不会一次性把整个数据集读进内存
    train_data = np.load(args.train_data_path, mmap_mode="r")
    val_data = np.load(args.val_data_path, mmap_mode="r")
    print(f"run {args.run_name}｜训练集 {train_data.shape}，验证集 {val_data.shape}，设备 {args.device}")

    # SwiGLU 中间层维度取约 8/3 * d_model，并向上取整到 64 的整数倍
    d_ff = int(args.d_model * 8 / 3 + 63) // 64 * 64

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)

    optimizer = AdamW(model.parameters(),
                      lr=args.max_lr,
                      betas=(args.beta1, args.beta2),
                      weight_decay=args.weight_decay)

    train_llm(model, optimizer, train_data, val_data, args)

if __name__ == "__main__":
    main()
