#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=2,3

# training 
/home/kpn/anaconda3/bin/conda run -n sood-mcl
cd /data/yht/code/sood-mcl




# 10per_mcl_train DOTA1.5
# /home/kpn/anaconda3/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29558\
#     train.py configs_dota15/mcl/mcl_fcos_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/mcl/DOTA1.5/10per

# 10per_mcl_train DOTA1.0
# /home/kpn/anaconda3/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29558\
#     train.py configs_dota15/mcl/mcl_fcos_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/mcl/DOTA1.0/10per_unsup-weight-0




# 10per_dt_train DOTA1.5
# /home/kpn/anaconda3/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29558\
#     train.py configs_dota15/denseteacher/denseteacher_fcos_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/dt/DOTA1.5/10per

# 10per_dt_train DOTA1.0
# /home/kpn/anaconda3/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29558\
#     train.py configs_dota15/denseteacher/denseteacher_fcos_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/dt/DOTA1.0/10per_switch-aug

# full_dt_train DOTA1.0
# /home/kpn/anaconda3/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29556\
#     train.py configs_dota15/denseteacher/denseteacher_fcos_dota10_full.py \
#     --launcher pytorch \
#     --work-dir log/dt/DOTA1.0/full




# 10per_dtbaseline_train DOTA1.0 debug
# /home/kpn/anaconda3/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29458\
#     train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_dota10_10p_debug.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/DOTA1.0/debug

# 10per_dtbaseline_train DOTA1.0
# /home/kpn/anaconda3/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29560\
#     train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/DOTA1.0/10per_prototype/only-update

# full_dtbaseline_train DOTA1.0
/home/kpn/anaconda3/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29560\
    train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_dota10_full.py \
    --launcher pytorch \
    --work-dir log/dtbaseline/DOTA1.0/full