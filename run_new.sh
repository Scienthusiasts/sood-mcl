#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=4,5
# export CUDA_VISIBLE_DEVICES=6,7
# export CUDA_VISIBLE_DEVICES=0,3

# training 
/home/yht/.conda/envs/sood-mcl run -n sood-mcl
cd /data/yht/code/sood-mcl










"""sparsely annotated 稀疏标注任务(去除有监督分支, 且只保留强弱增强的数据)"""

'''debug'''
# 10per_sparse_fnmining
# CUDA_LAUNCH_BLOCKING=1  /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550 --use_env\
#     train.py configs_dota15/sparse_new_idea/globalw_fcos_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/new_sparse/debug

# 10per_sparse_fnmining_gihead CUDA_LAUNCH_BLOCKING=1  
/home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550 --use_env\
    train.py configs_dota15/sparse_new_idea/globalw_fcos_gihead_dota10_10p.py \
    --launcher pytorch \
    --work-dir log/sparse_fnmining_gihead/debug

# 10per_sparse_fnmining_gihead_wo_reggt CUDA_LAUNCH_BLOCKING=1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550 --use_env\
#     train.py configs_dota15/sparse_new_idea/globalw_fcos_gihead_dota10_10p_wo_reggt.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead/debug




'''sparsely globalw dota1.0'''
# 10per_sparse_fnmining # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining/1.0/burn-in-12800_ga_sfpm-thres0.05-fn-allweight-thres1.0-beta5.0_10per/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29564 --use_env\
#     train.py configs_dota15/sparse_new_idea/globalw_fcos_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining/1.0/burn-in-12800_ga_sfpm-thres0.05-fn-allweight-thres1.0-beta5.0_10per

# 10per_sparse_fnmining_gihead_wo_reggt # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining_gihead/1.0/burn-in-12800_ga_sfpm-thres0.1-fn-allweight-thres1.0-beta5.0_only-update-giheadplus-posthr0.7-noclsloss_10per/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29564 --use_env\
#     train.py configs_dota15/sparse_new_idea/globalw_fcos_gihead_dota10_10p_wo_reggt.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead/1.0/burn-in-12800_ga_sfpm-thres0.1-fn-allweight-thres1.0-beta5.0_only-update-giheadplus-posthr0.7-noclsloss_10per

# 10per_sparse_fnmining_gihead # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining_gihead/1.0/burn-in-12800_ga_sfpm-thres0.1-fn-allweight-thres1.0-beta5.0_giheadplus-posthr0.7-noclsloss_reggt-thr0.9_10per/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29564 --use_env\
#     train.py configs_dota15/sparse_new_idea/globalw_fcos_gihead_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead/1.0/burn-in-12800_ga_sfpm-thres0.1-fn-allweight-thres1.0-beta5.0_giheadplus-posthr0.7-noclsloss_reggt-thr0.9_10per



'''sparsely globalw dota1.5'''
# 10per_globalw # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining/1.5/burn-in-120000_ga_1per/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29566 --use_env\
#     train.py configs_dota15/sparse_new_idea/globalw_fcos_ss_gihead_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining/1.5/burn-in-120000_ga_1per


