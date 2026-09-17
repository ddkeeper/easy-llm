# -*- coding: utf-8 -*-
"""
2.3.3 训练脚本：把各节实现的组件组装成一个完整的训练循环。

结构与 §3.2 实验里真正跑的那份脚本一致：
    train_step(...)  单步训练：设置学习率 → 前向传播 → 计算损失 → 反向传播 → 梯度裁剪 → 更新参数
    train_llm(...)   训练循环：反复调用 train_step，中间穿插学习率调度、验证集评估、检查点保存与日志记录

相比纯讲解版本，这里额外支持：
    · 消融开关：--no_rmsnorm / --post_norm / --no_gated_ffn / --rope_theta 0 / --tie_embedding
    · 混合精度：--amp（仅在 CUDA 上生效，CPU 自动忽略）
    · 实验管理：实验脚本统一用 --run_name 传「类型-lr峰值_谷值-bs-种子」（如 lr_sweep-lr3e-4_3e-5-bs64-s2026），
      不传时按「数据集-bs-学习率-消融名-种子」自动拼；超参全量存档到 wandb
    · 检查点分层：保存到 {save_dir}/{数据集}/{实验名}/iter{步数:06d}.pt，换数据集自动分开存

数据路径默认就指向本项目的 TinyStories 分词结果，手跑时不必再写
--train_data_path / --val_data_path：
    data/TinyStories/TinyStoriesV2-GPT4-train.npy
    data/TinyStories/TinyStoriesV2-GPT4-valid.npy
路径都相对项目根目录（= 执行脚本时的当前目录），换数据集时再用这两个参数覆盖；
完整命令行（含路径覆盖写法）见 scripts/run_experiments.sh。

运行示例（Windows / Linux 通用）：
    python scripts/train_llm.py --amp --device cuda                            # 基准配置，完整训练 5000 步
    python scripts/train_llm.py --total_iters 1000 --batch_size 8 --val_interval 50 --run_name try_in_python_123  # 覆盖部分超参做快速对比
    python scripts/train_llm.py --post_norm --ablation_name post_norm          # 消融实验：Post-Norm
"""

import os
import time
import argparse
import numpy as np
import torch
import wandb   # Weights & Biases：实验记录与可视化平台，本脚本只用到 init / log / finish 三个 API

from training_utils import (
    cross_entropy,
    AdamW,
    learning_rate_schedule,
    gradient_clipping,
    get_batch,
    save_checkpoint,
)
from transformer import TransformerLM


# ------------------------------------------------------------------ 单步训练
def train_step(model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               x: torch.Tensor,
               y: torch.Tensor,
               lr: float,
               max_l2_norm: float = 1.0,
               scaler: torch.amp.GradScaler = None) -> float:
    '''
    执行一步训练，返回本步的损失值（标量）

    model: torch.nn.Module  语言模型
    optimizer: torch.optim.Optimizer  优化器
    x: (batch_size, context_length)  输入 token ID，需为 long 类型
    y: (batch_size, context_length)  目标 token ID，即 x 整体右移一格
    lr: float  本步使用的学习率（由余弦调度给出）
    max_l2_norm: float  M，梯度裁剪允许的最大 ℓ2 范数
    scaler: torch.amp.GradScaler | None  混合精度用的梯度缩放器；传 None 表示走普通 fp32 流程
    '''

    optimizer.param_groups[0]["lr"] = lr   # 余弦调度给出的学习率，必须在 step 之前写进去
    use_amp = scaler is not None           # 是否走混合精度

    # 混合精度：矩阵乘在 autocast 上下文里自动降到 bfloat16 计算，其余算子保持 fp32
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=use_amp):
        logits = model(x)                       # (B, m, vocab_size) 前向传播
        loss = cross_entropy(logits, y).mean()  # 对所有位置取平均，得到一个标量损失

    optimizer.zero_grad()                       # 清空上一步的梯度

    if use_amp:
        scaler.scale(loss).backward()           # 反向传播：先把损失放大，避免小梯度在 bf16 下变成 0
        scaler.unscale_(optimizer)              # 裁剪前必须还原尺度，否则裁剪阈值是对放大后的梯度生效的
    else:
        loss.backward()                         # 反向传播，计算梯度

    gradient_clipping(model.parameters(), max_l2_norm)  # 梯度裁剪

    if use_amp:
        scaler.step(optimizer)                  # 优化器更新（若检测到 NaN/Inf 会自动跳过这一步）
        scaler.update()                         # 根据是否跳过动态调整缩放系数
    else:
        optimizer.step()                        # 更新参数

    return loss.item()


# ------------------------------------------------------------------ 训练循环
def train_llm(model: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              train_data: np.ndarray,
              val_data: np.ndarray,
              args,
              scaler: torch.amp.GradScaler = None) -> None:
    '''
    训练循环：反复调用 train_step，并按间隔做验证、保存检查点与记录日志

    train_data / val_data: np.ndarray  token ID 数组（大数据集建议用 np.load(路径, mmap_mode="r") 打开）
    args: 超参集合，字段见 parse_args()
    scaler: torch.amp.GradScaler | None  混合精度缩放器，由 main() 创建好后一路传进来
    '''

    # wandb.init：开启一次"实验运行"（run），网页端会新建一个同名项目页面来承接它
    #   project / name 决定它出现在哪个项目的哪一行
    #   config=vars(args) 把全部命令行超参一起存档，之后可在网页端按超参筛选、排序、对比不同 run
    #   返回的 run 是这次运行的句柄：run.log 写数据、run.id 可在网页端定位到这次运行
    #   mode 默认 online：日志实时上传到 wandb 云端（需要能联网且已经 wandb login）
    #   想只在本地留一份，就在命令前加 WANDB_MODE=offline，之后可用 wandb sync 补传
    
    run = wandb.init(project="easy-llm", name=args.run_name, config=vars(args), mode=args.wandb_mode)
    os.makedirs(args.save_dir, exist_ok=True)   # args.save_dir 已在 main() 里拼成「根目录/数据集/实验名」

    start_time = time.time()

    for i in range(1, args.total_iters + 1):
        model.train()   # 切到训练模式（启用 dropout 等）

        # 1. 取一批训练数据，并算出本步的学习率
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        lr = learning_rate_schedule(i, args.max_lr, args.min_lr,
                                    args.warmup_iters, args.cosine_cycle_iters)

        # 2. 单步训练：前向 → 损失 → 反向 → 裁剪 → 更新
        loss = train_step(model, optimizer, x.long(), y.long(), lr=lr,
                          max_l2_norm=args.max_l2_norm, scaler=scaler)

        # 3. 同一步的训练/验证指标放进同一条记录，避免 W&B 横轴错位
        metrics = {"train_loss": loss, "lr": lr}

        # 4. 定期在验证集上评估：torch.no_grad() 下不建图，也不更新参数
        if i % args.val_interval == 0:
            model.eval()   # 切到评估模式（关闭 dropout 等）
            with torch.no_grad():
                with torch.autocast(device_type=args.device.type, dtype=torch.bfloat16, enabled=scaler is not None):
                    val_x, val_y = get_batch(val_data, args.batch_size, args.context_length, args.device)
                    val_loss = cross_entropy(model(val_x.long()), val_y.long()).mean().item()
            metrics["val_loss"] = val_loss
            print(f"iter {i:5d}  train_loss={loss:.4f}  val_loss={val_loss:.4f}")

        metrics["wall_time"] = time.time() - start_time

        # run.log：把一个 dict 追加到这条 run 的时间线上，网页端会自动画出曲线
        #   功能上近似 print(metrics)，额外多了"持久化 + 自动绘图 + 跨 run 对比"
        #   step=i 显式指定横轴为迭代数；不传的话 wandb 会自己维护一个自增步数
        run.log(metrics, step=i)

        # 5. 定期保存检查点（最后一步一定保存）
        #    文件名只留迭代步数：数据集与实验名已经体现在目录上，不用再挤进文件名里
        if i % args.save_interval == 0 or i == args.total_iters:
            checkpoint_path = os.path.join(args.save_dir, f"iter{i:06d}.pt")
            save_checkpoint(model, optimizer, i, checkpoint_path)
            print(f"iter {i:5d}  已保存检查点 → {checkpoint_path}")

    # run.finish：结束本次运行，把缓冲区里还没上传的数据刷到服务器
    #   脚本正常退出时 wandb 会自动收尾，显式写上能让异常中断时也尽量保住已记录的曲线
    run.finish()


# ------------------------------------------------------------------ 超参与入口
def parse_args():
    parser = argparse.ArgumentParser(description="训练一个 Transformer 语言模型")

    # 数据
    parser.add_argument("--dataset_name", type=str, default="TinyStoriesV2-GPT4")
    # 数据路径：默认指向项目里的 TinyStories 分词结果（相对项目根目录），
    # 换数据集时用这两个参数覆盖即可。
    parser.add_argument("--train_data_path", type=str,
                        default="data/TinyStories/TinyStoriesV2-GPT4-train.npy")
    parser.add_argument("--val_data_path", type=str,
                        default="data/TinyStories/TinyStoriesV2-GPT4-valid.npy")

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
    # 检查点根目录（相对项目根目录）；实际保存位置是 {save_dir}/{dataset_name}/{run_name}/，见 main() 里的拼接
    parser.add_argument("--save_dir", type=str, default="results/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None,
                        help="实验名，同时用作 W&B run 名与检查点目录名；实验脚本统一传 "
                             "「类型-lr峰值_谷值-bs-种子」（如 lr_sweep-lr3e-4_3e-5-bs64-s2026），"
                             "不传时按 数据集-bs-学习率[-消融名]-种子 自动生成")
    parser.add_argument("--wandb_mode", type=str, default=os.getenv("WANDB_MODE", "online"),
                        help="wandb 运行模式：online 实时上传到云端（需能联网且已 wandb login）；"
                             "offline 只写本地 ./wandb，之后可 wandb sync 补传。"
                             "也认环境变量 WANDB_MODE，方便在实验脚本里统一开关")
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备：auto 自动挑 cuda/cpu，也可显式写 cuda、cuda:0、cpu")
    parser.add_argument("--amp", action="store_true",
                        help="开启 bf16 混合精度；设备不是 CUDA 时自动忽略")

    # 模型超参，默认使用基准值（CS336 Assignment 1, §7.2）
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--rope_theta", type=float, default=10000.0,
                        help="RoPE 基数频率；传 0 即关闭位置编码（NoPE 消融）")
    # 默认 0.02 而不是 1.0：嵌入层参与残差主路径，标准差太大会让 logits 一开始就爆（尤其开了
    # --tie_embedding 时，输出头直接复用嵌入矩阵，std=1.0 下初始 loss 会到几百）
    parser.add_argument("--embed_init_std", type=float, default=0.02,
                        help="嵌入层初始化标准差；调小可抑制训练初期 loss 尖刺")

    # ---- 消融实验开关：默认全关，即标准 Pre-Norm + RMSNorm + SwiGLU 结构 ----
    parser.add_argument("--no_rmsnorm", action="store_true", help="消融：去掉全部 RMSNorm") # python train_llm.py 
    parser.add_argument("--post_norm", action="store_true", help="消融：Pre-Norm 改成 Post-Norm")
    parser.add_argument("--no_gated_ffn", action="store_true", help="消融：SwiGLU 改成普通 SiLU 前馈网络")
    parser.add_argument("--tie_embedding", action="store_true", help="性能优化：输出头与输入嵌入共享权重")
    parser.add_argument("--ablation_name", type=str, default=None,
                        help="本次消融的简短代号（如 post_norm），会写进 run_name 方便网页端分组查看")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 设备：auto 时优先 CUDA，没有就退回 CPU，Windows 无显卡时也能跑通 ----
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = torch.device(args.device)

    if args.device.type == "cuda":
        # 允许 fp32 矩阵乘走 tf32：Ampere 及以上架构的张量核心格式，提速明显且几乎不掉精度
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 混合精度只在 CUDA 上生效：CPU 上的 bf16 支持不完善，强行开反而更慢
    use_amp = args.amp and args.device.type == "cuda"

    # ---- run_name：实验脚本一般会显式传（{类型}-lr{峰值}_{谷值}-bs{bs}-s{种子}），
    #      直接手跑、没传时按「数据集-bs-学习率[-消融名]-种子」兜底拼一个，保证网页端名字可区分 ----
    if args.run_name is None:
        args.run_name = (
            f"{args.dataset_name}-bs{args.batch_size}"
            f"-lr{args.max_lr:.1e}_{args.min_lr:.1e}"
            f"{'-' + args.ablation_name if args.ablation_name else ''}"
            f"-s{args.seed}"
        )

    # ---- 检查点目录：{save_dir}/{数据集}/{实验名}/ ----
    # 数据集和实验名交给目录承载，文件名只留迭代步数，这样：
    #   1. 换数据集自动分开存，不会和别的实验结果混在一起
    #   2. 同一组配置重跑会直接覆盖旧文件，不会像以前那样靠 run.id 越积越多
    args.save_dir = os.path.normpath(os.path.join(args.save_dir, args.dataset_name, args.run_name))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ---- 数据：mmap_mode="r" 以内存映射方式打开，不会一次性把整个数据集读进内存 ----
    # 默认指向项目里的 TinyStories
    train_data = np.load(args.train_data_path, mmap_mode="r")
    val_data = np.load(args.val_data_path, mmap_mode="r")
    print(f"run {args.run_name}｜训练集 {train_data.shape}，验证集 {val_data.shape}，设备 {args.device}"
          f"｜混合精度 {use_amp}")

    # 中间层维度按是否门控分别取，统一对齐到 64 的整数倍：
    #   gated（SwiGLU）取 8/3 * d_model；plain SiLU 取 4 * d_model，让两者的参数量大致相当，
    #   免得「有没有门控」和「参数多少」两个变量搅在一起，消融结论不好解释。
    if args.no_gated_ffn:
        d_ff = int(args.d_model * 4 + 63) // 64 * 64      # plain SiLU：4 * d_model
    else:
        d_ff = 64 * (args.d_model * 8 // 3 // 64)         # SwiGLU：8/3 * d_model

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        embed_init_std=args.embed_init_std,
        use_rmsnorm=not args.no_rmsnorm,
        pre_norm=not args.post_norm,
        ffn_gated=not args.no_gated_ffn,
        tie_embedding=args.tie_embedding,
    ).to(args.device)

    optimizer = AdamW(model.parameters(),
                      lr=args.max_lr,
                      betas=(args.beta1, args.beta2),
                      weight_decay=args.weight_decay)

    # GradScaler 只在真的要走混合精度时才创建
    scaler = torch.amp.GradScaler(args.device.type, enabled=use_amp) if use_amp else None

    train_llm(model, optimizer, train_data, val_data, args, scaler=scaler)

if __name__ == "__main__":
    main()
