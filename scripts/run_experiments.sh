#!/bin/bash
# OpenWebText(owt) 实验脚本：只定义 run_exp 一个函数，跑哪些实验由底部的命令决定
# 用法（Windows 需在 Git Bash 里执行；Linux 直接执行）:
#   nohup bash scripts/run_experiments.sh > logs/owt.txt 2>&1 &
#   WANDB_MODE=offline bash scripts/run_experiments.sh   # 不上传云端，只写本地 ./wandb
# 默认跑全部 4 组实验，不需要的实验自行注释掉对应行即可。

export CUDA_VISIBLE_DEVICES=0
# bs=128 + 32000 词表时中间张量很大，不加这行会因显存碎片 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=${PYTHON:-/home/wangshijin/anaconda3/envs/nanogpt/bin/python}   # 可用环境变量 PYTHON 覆盖

# ========== 公共超参 ==========
train_data=data/owt/owt_train.npy
val_data=data/owt/owt_valid.npy
dataset_name=owt
vocab_size=32000
context_length=256
wandb_project=easy-llm-owt # 换数据集同时换 wandb project，网页端不同数据集上的实验可以独立开来

d_model=512
num_layers=4
num_heads=16
rope_theta=10000

weight_decay=0.1
beta1=0.9
beta2=0.95
max_l2_norm=1.0
seed=2026

warmup_iters=1600
total_iters=16000
val_interval=100
save_interval=4000
save_dir=results/checkpoints

embed_init_std=0.02

# ========== 核心训练函数 ==========
# run_exp <exp_name> <max_lr> <min_lr> <batch_size> [额外参数...]
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
# 4 组实验默认全部执行，不需要的自行注释掉。执行顺序 = 文本顺序。

# (1) lr 扫描
run_exp "lr_sweep"  1e-4  1e-5  64
run_exp "lr_sweep"  2e-4  2e-5  64
run_exp "lr_sweep"  3e-4  3e-5  64
run_exp "lr_sweep"  6e-4  6e-5  64
run_exp "lr_sweep"  1.2e-3 1.2e-4 64
run_exp "lr_sweep"  2.4e-3 2.4e-4 64
run_exp "lr_sweep"  4.8e-3 4.8e-4 64
run_exp "lr_sweep"  1e-2  1e-3  64
run_exp "lr_sweep"  2e-2  2e-3  64

# (2) batch_size 扫描（lr=2.4e-3→2.4e-4；token 预算对齐 = 16000×64×256 ≈ 1 epoch）
run_exp "bs_sweep"  2.4e-3 2.4e-4   1 --total_iters 64000  --warmup_iters 6400 --cosine_cycle_iters 64000  --val_interval 400 --save_interval 16000
run_exp "bs_sweep"  2.4e-3 2.4e-4   8 --total_iters 32000  --warmup_iters 3200 --cosine_cycle_iters 32000  --val_interval 200 --save_interval 8000
run_exp "bs_sweep"  2.4e-3 2.4e-4  16 --total_iters 32000  --warmup_iters 3200 --cosine_cycle_iters 32000  --val_interval 200 --save_interval 8000
run_exp "bs_sweep"  2.4e-3 2.4e-4  32 --total_iters 32000  --warmup_iters 3200 --cosine_cycle_iters 32000  --val_interval 200 --save_interval 8000
run_exp "bs_sweep"  2.4e-3 2.4e-4  64 --total_iters 16000  --warmup_iters 1600 --cosine_cycle_iters 16000  --val_interval 100 --save_interval 4000
run_exp "bs_sweep"  2.4e-3 2.4e-4 128 --total_iters  8000  --warmup_iters  800 --cosine_cycle_iters  8000  --val_interval  50 --save_interval 2000

# (3) 消融实验（lr=2.4e-3→2.4e-4、bs=64，只改结构开关）
run_exp "ablation_no_rmsnorm"  2.4e-3 2.4e-4 64 --no_rmsnorm   --ablation_name no_rmsnorm
run_exp "ablation_post_norm"   2.4e-3 2.4e-4 64 --post_norm    --ablation_name post_norm
run_exp "ablation_nope"        2.4e-3 2.4e-4 64 --rope_theta 0 --ablation_name nope
run_exp "ablation_no_gated"    2.4e-3 2.4e-4 64 --no_gated_ffn --ablation_name no_gated
run_exp "ablation_tied"        2.4e-3 2.4e-4 64 --tie_embedding --ablation_name tied

# (4) 训练优化对比（amp 作基础，逐项叠加 Muon / torch.compile / 权重共享）
run_exp "accel_base"   2.4e-3 2.4e-4 64
run_exp "accel_tied"   2.4e-3 2.4e-4 64 --tie_embedding --ablation_name tied
run_exp "accel_muon"   2.4e-3 2.4e-4 64 --optimizer muon
# run_exp "accel_full"   2.4e-3 2.4e-4 64 --optimizer muon --tie_embedding --torch_compile # 当前代码版本与 torch.compile(model) 并不兼容，需要修改后才能使用
