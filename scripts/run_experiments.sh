#!/bin/bash
# OpenWebText(owt) 实验脚本：只定义 run_exp 一个函数，跑哪些实验由底部的命令决定
# 用法（Linux 下可以直接执行；windows 下要先安装 git，在自带的 gitbash 终端里执行）:
#   1. 把底部「实验命令」里要跑的行取消注释
#   2. nohup bash scripts/run_experiments.sh > logs/owt_ablation.txt 2>&1 &
#   ⚠ 不要直接 source：source 会立刻把底部未注释的行全跑一遍；要手动调用 run_exp，
#     先把底部实验命令全部注释掉，再 source scripts/run_experiments.sh
#   WANDB_MODE=offline bash scripts/run_experiments.sh   # 不上传云端，只写本地 ./wandb

export CUDA_VISIBLE_DEVICES=0
# bs=128 + 32000 词表时，(128,256,32000) 的 logits 中间张量很大，不加这行会因显存碎片 OOM
# 加上后实测峰值 22.8G / 24G（bs=96: 17.3G，bs=64: 11.9G）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=${PYTHON:-/home/wangshijin/anaconda3/envs/nanogpt/bin/python}    # 跑实验用的虚拟环境 python；可用环境变量 PYTHON 覆盖

# ========== 公共超参 ==========
# owt_train 265,537,732 token / owt_valid 64,681,244 token（uint16，全量扫描 max id = 31999 → 词表取 32,000）
train_data=data/owt/owt_train.npy
val_data=data/owt/owt_valid.npy
dataset_name=owt
vocab_size=32000
context_length=256
wandb_project=easy-llm-owt   # 换数据集同时换 wandb project，网页端不会和 TinyStories 的 run 混在一起

d_model=512
num_layers=4
num_heads=16
rope_theta=10000

weight_decay=0.1
beta1=0.9
beta2=0.95
max_l2_norm=1.0
seed=2026

# 公共步数：(3) 消融与 (4) 基准直接用这里的值（16000 步 × bs64 × 256 ≈ 262M ≈ 1 epoch）；
# (2) bs 扫描会在自己那几行用 extra_args 覆盖 total_iters/warmup/val/save（见下方表格）
warmup_iters=1600      # 10% 的 total_iters，与 TinyStories 的 500/5000 同比例
total_iters=16000
val_interval=100
save_interval=4000     # 16000 步存 4 个；若按 500 存是 32 个 × 518MB = 16.6GB/组，21 组会撑爆磁盘
save_dir=results/checkpoints

embed_init_std=0.02

# ========== 核心训练函数 ==========
# run_exp <exp_name> <max_lr> <min_lr> <batch_size> [额外参数...]
# 额外参数可传: --total_iters 40000 --warmup_iters 4000 --cosine_cycle_iters 40000
# 实验名统一拼成 {exp_name}-lr{max_lr}_{min_lr}-bs{bs}-s{seed}，由 --run_name 传给训练脚本，
# 训练脚本据此建目录 checkpoints/{dataset_name}/{run_name}/，wandb 上也用同一个名字
run_exp() {
    local exp_name=$1; shift
    local max_lr=$1; shift
    local min_lr=$1; shift
    local bs=$1; shift
    local extra_args="$@"

    local run_name="${exp_name}-lr${max_lr}_${min_lr}-bs${bs}-s${seed}"
    echo "=== ${run_name} ==="

    $PYTHON scripts/train_llm.py \
        --run_name ${run_name} \
        --dataset_name ${dataset_name} \
        --wandb_project ${wandb_project} \
        --vocab_size ${vocab_size} \
        --context_length ${context_length} \
        --d_model ${d_model} \
        --num_layers ${num_layers} \
        --num_heads ${num_heads} \
        --rope_theta ${rope_theta} \
        --weight_decay ${weight_decay} \
        --beta1 ${beta1} \
        --beta2 ${beta2} \
        --max_l2_norm ${max_l2_norm} \
        --seed ${seed} \
        --warmup_iters ${warmup_iters} \
        --cosine_cycle_iters ${total_iters} \
        --total_iters ${total_iters} \
        --max_lr ${max_lr} \
        --min_lr ${min_lr} \
        --batch_size ${bs} \
        --train_data_path ${train_data} \
        --val_data_path ${val_data} \
        --save_dir ${save_dir} \
        --val_interval ${val_interval} \
        --save_interval ${save_interval} \
        --amp --embed_init_std ${embed_init_std} \
        ${extra_args}
}

# ========== 实验命令 ==========
# 【当前状态】(3) 消融 5 组 + (2) bs 扫描 6 组 = 11 组已取消注释，直接 bash 就会依次跑完这 11 组！
# 执行顺序 = 文本顺序：(3) 消融 5 组 → (2) bs 扫描 6 组（编号保持 1/2/3/4 不变），跑完约 8 小时。
# (1) lr 扫描已跑完并全部注释（结论在下方），(0) 和 (4) 也是注释状态。
# 公共 16000 步被 (2) 用 extra_args 覆盖：argparse 对重复参数取最后一个值，extra_args 拼在命令末尾所以能覆盖成功。

# (0) 通路测试：200 步约 40 秒，验证数据/模型/保存没报错（自带 total_iters/warmup 覆盖，不看 lr 结论）
# run_exp "try_in_bash"  3e-4  3e-5  128 --total_iters 200 --warmup_iters 50 --cosine_cycle_iters 200 --tie_embedding

# 【已跑完 2026-10-04，结论见下】9 组全部 16000 步，末段 val_loss 排序：
#   2.4e-3 (4.1211) > 1.2e-3 (4.1329) > 4.8e-3 (4.1601) > 6e-4 > 1e-2 > 3e-4 > 2e-2 > 2e-4 > 1e-4 (4.8750)
#   区间已夹住（两端 1e-4、2e-2 明显劣化），最优区间 1.2e-3 ~ 4.8e-3，取几何中心 2.4e-3；
#   前三名差距 < 末段噪声 std≈0.05（val 只用 1 个随机 batch），统计上等价，故不必再细化。
# run_exp "lr_sweep"  1e-4  1e-5  64
# run_exp "lr_sweep"  2e-4  2e-5  64
# run_exp "lr_sweep"  3e-4  3e-5  64
# run_exp "lr_sweep"  6e-4  6e-5  64
# run_exp "lr_sweep"  1.2e-3 1.2e-4 64
# run_exp "lr_sweep"  2.4e-3 2.4e-4 64
# run_exp "lr_sweep"  4.8e-3 4.8e-4 64
# run_exp "lr_sweep"  1e-2  1e-3  64
# run_exp "lr_sweep"  2e-2  2e-3  64

# ============ 执行顺序：先 (3) 消融、再 (2) bs 扫描（编号保持不变）============
# (3) 消融实验（lr=2.4e-3→2.4e-4，(1) 扫出的最优；bs=64，只改结构开关；步数跟随公共的 16000）
#     注意：原来硬编码的 1e-2 在 owt 上排第 5、比最优差 0.13，必须换成 2.4e-3，否则结构结论会被 lr 污染
# run_exp "ablation_no_rmsnorm"  2.4e-3 2.4e-4 64 --no_rmsnorm   --ablation_name no_rmsnorm
# run_exp "ablation_post_norm"   2.4e-3 2.4e-4 64 --post_norm    --ablation_name post_norm
# run_exp "ablation_nope"        2.4e-3 2.4e-4 64 --rope_theta 0 --ablation_name nope
# run_exp "ablation_no_gated"    2.4e-3 2.4e-4 64 --no_gated_ffn --ablation_name no_gated
# run_exp "ablation_tied"        2.4e-3 2.4e-4 64 --tie_embedding --ablation_name tied

# (2) bs 扫描（lr=2.4e-3→2.4e-4，(1) 扫出的最优）
#     token 预算对齐 (1)(3)(4)：262.14M = 16000×64×256 ≈ 1 epoch（原来固定 81.92M 只有 0.31 epoch，与其它组不同源）
#     bs≥16 按 iters = 262.14M/(bs×256) 保证数据量相同；
#     bs=8 / bs=1 若按比例要 128 万 / 32 万步，太费时 → 两组都封顶用 bs=16 的 64000 步：
#         bs   iters   token    epoch   预计耗时
#           1   64000    16.4M   0.06    2.7 min   ← 数据量最少，结论要区分「batch 小」还是「见得少」
#           8   64000   131.1M   0.49     25 min   ← 同上，为省时折中
#          16   64000   262.1M   0.99     50 min
#          32   32000   262.1M   0.99     50 min
#          64   16000   262.1M   0.99     50 min   ← 与 (1)(3)(4) 完全一致，可直接对照
#         128    8000   262.1M   0.99     50 min
#     warmup 统一 10%；val_interval = iters/160 → 每组 160 个验证点（单点噪声都来自同一 16k token batch，
#     末段 5 点均值可横向比）；save_interval = iters/4 → 每组 5 个 ckpt
#     注意：2.4e-3 是 bs=64 下的最优，按 linear scaling rule 其它 bs 的最优 lr 本应不同，
#     固定 lr 正是为了让组间差异只来自 batch，解读时记得这点
#run_exp "bs_sweep"  2.4e-3 2.4e-4   1 --total_iters 64000  --warmup_iters 6400 --cosine_cycle_iters 64000  --val_interval 400 --save_interval 16000
# run_exp "bs_sweep"  2.4e-3 2.4e-4 128 --total_iters  8000  --warmup_iters  800 --cosine_cycle_iters  8000  --val_interval  50 --save_interval 2000
# run_exp "bs_sweep"  2.4e-3 2.4e-4  8 --total_iters 32000  --warmup_iters 3200 --cosine_cycle_iters 32000  --val_interval 200 --save_interval 8000
# run_exp "bs_sweep"  2.4e-3 2.4e-4  16 --total_iters 32000  --warmup_iters 3200 --cosine_cycle_iters 32000  --val_interval 200 --save_interval 8000
# run_exp "bs_sweep"  2.4e-3 2.4e-4  32 --total_iters 32000  --warmup_iters 3200 --cosine_cycle_iters 32000  --val_interval 200 --save_interval 8000
#run_exp "bs_sweep"  2.4e-3 2.4e-4  64 --total_iters 16000  --warmup_iters 1600 --cosine_cycle_iters 16000  --val_interval 100 --save_interval 4000


# (5) 训练加速对比（3.4.2 leaderboard：amp 作基础，逐项叠加 Muon / torch.compile / 权重共享）
#     全组 lr=2.4e-3→2.4e-4、bs=64、16000 步（与 (1)(3)(4) 同源）；amp 由 run_exp 自带，视为基础优化。
#     变量一次只动一个，才能讲清「每项各带来多少 + 全叠加的上限」：
#   ① amp（基准，已跑）
# run_exp "accel_base"   2.4e-3 2.4e-4 64
#   ② amp + 权重共享（已跑）
# run_exp "accel_tied"   2.4e-3 2.4e-4 64 --tie_embedding --ablation_name tied
#   ③ amp + Muon（单看 Muon 的收益）
run_exp "accel_muon"   2.4e-3 2.4e-4 64 --optimizer muon
#   ④ amp + Muon + 权重共享 + torch.compile（全优化）
run_exp "accel_full"   2.4e-3 2.4e-4 64 --optimizer muon --tie_embedding --torch_compile
#     注：torch.compile 首次编译慢、只在 CUDA 生效，单列一组收益不如叠加到 ④ 里一起看。
