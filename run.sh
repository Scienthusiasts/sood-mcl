#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=4,5
# export CUDA_VISIBLE_DEVICES=6,7

# training 
/home/yht/.conda/envs/sood-mcl run -n sood-mcl
cd /data/yht/code/sood-mcl




'''mcl'''
# 10per_mcl_train DOTA1.5 debug
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/mcl/mcl_fcos_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/DOTA1.0/debug

# 10per_mcl_train DOTA1.5
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29564\
#     train.py configs_dota15/mcl/mcl_fcos_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/mcl/DOTA1.5/10per_burn-in-12800

# 10per_mcl_train DOTA1.0
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29562\
#     train.py configs_dota15/mcl/mcl_fcos_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/mcl/DOTA1.0/10per




'''dt'''
# 10per_dtbaseline_train DOTA1.0 debug
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29450\
#     train.py configs_dota15/denseteacher/denseteacher_fcos_dota10_10p_debug.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/DOTA1.0/debug

# 10per_dt_train DOTA1.5
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29558\
#     train.py configs_dota15/denseteacher/denseteacher_fcos_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/dt/DOTA1.5/10per

# 10per_dt_train DOTA1.0
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29558\
#     train.py configs_dota15/denseteacher/denseteacher_fcos_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/dt/DOTA1.0/10per_switch-aug

# full_dt_train DOTA1.0
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29556\
#     train.py configs_dota15/denseteacher/denseteacher_fcos_dota10_full.py \
#     --launcher pytorch \
#     --work-dir log/dt/DOTA1.0/full




'''dtbaseline'''
'''debug'''
# 10per_dtbaseline_train DOTA1.0 debug # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_refineORCNNhead_dota15_10p_debug.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/debug


# 单卡
# /home/yht/.conda/envs/sood-mcl/bin/python \
#     train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_refineORCNNhead_dota15_10p_debug.py \
#     --work-dir log/dtbaseline/debug



# 10per_dtbaseline_train DOTA1.0
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29560\
#     train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/DOTA1.0/10per_prototype/only-update_wo-random-init

# full_dtbaseline_train DOTA1.0
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29560\
#     train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_dota10_full.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/DOTA1.0/full

# 10per_dtbaseline_train DOTA1.5
/home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29564\
    train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_refineORCNNhead_dota15_10p.py \
    --launcher pytorch \
    --work-dir log/dtbaseline/DOTA1.5/10per_denoise/global-w/ss/joint-score-beta-2.0_burn-in-12800_orcnn-head_all-refine-loss_box-O2M-loss_detach_GA_7_ssloss-box-flip0.1-w1.0






