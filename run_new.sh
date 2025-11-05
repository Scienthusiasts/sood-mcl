#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=4,5
# export CUDA_VISIBLE_DEVICES=6,7

# training 
/home/yht/.conda/envs/sood-mcl run -n sood-mcl
cd /data/yht/code/sood-mcl










"""sparsely annotated 稀疏标注任务(去除有监督分支, 且只保留强弱增强的数据)"""

'''debug'''
# 10per_sparse_baseline
# CUDA_LAUNCH_BLOCKING=1  /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550 --use_env\
#     train.py configs_dota15/sparse_new_idea/fcos_baseline_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/new_sparse/debug

# 10per_sparse_fnmining
# CUDA_LAUNCH_BLOCKING=1  /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550 --use_env\
#     train.py configs_dota15/sparse_new_idea/fcos_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/new_sparse/debug

# 10per_sparse_fnmining_gihead CUDA_LAUNCH_BLOCKING=1  
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550 --use_env\
#     train.py configs_dota15/sparse_new_idea/fcos_gihead_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead/debug

# 10per_sparse_fnmining_gihead_wo_reggt CUDA_LAUNCH_BLOCKING=1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550 --use_env\
#     train.py configs_dota15/sparse_new_idea/fcos_gihead_dota10_10p_wo_reggt.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead/debug

# # 10per_unbaisedteacher_baseline # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550 --use_env\
#     train.py configs_dota15/sparse_new_idea_PECL/unbaisedteacher_orientedrcnn_baseline_dota10.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead/debug

# 10per_unbaisedteacher_gihead # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550 --use_env\
#     train.py configs_dota15/sparse_new_idea_PECL/unbaisedteacher_orientedrcnn_gi_dota10.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead/debug




'''sparsely  dota1.0 (s2teacher)'''
# 10per_sparse_fnmining # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining/1.0/burn-in-12800_ga_sfpm-thres0.05-fn-allweight-thres1.0-beta5.0_10per/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29564 --use_env\
#     train.py configs_dota15/sparse_new_idea/fcos_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining/1.0/burn-in-12800_ga_sfpm-thres0.05-fn-allweight-thres1.0-beta5.0_10per

# 10per_sparse_fnmining_gihead_wo_reggt # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining_gihead/1.0/burn-in-12800_ga_sfpm-thres0.1-fn-allweight-thres1.0-beta5.0_only-update-giheadplus-posthr0.7-noclsloss_10per/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29564 --use_env\
#     train.py configs_dota15/sparse_new_idea/fcos_gihead_dota10_10p_wo_reggt.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead/1.0/burn-in-12800_ga_sfpm-thres0.1-fn-allweight-thres1.0-beta5.0_only-update-giheadplus-posthr0.7-noclsloss_10per

# 10per_sparse_fnmining_gihead # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining_gihead/1.0/burn-in-12800_ga_sfpm-thres0.1-fn-allweight-thres1.0-beta5.0_gihead0712-posthr0.7-noclsloss_reggt-thr0.9_10per/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29576 --use_env\
#     train.py configs_dota15/sparse_new_idea/fcos_gihead_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead/1.0/burn-in-12800_ga_sfpm-thres0.1-fn-allweight-thres1.0-beta5.0_gihead0712-posthr0.7-noclsloss_reggt-thr0.9_10per



'''sparsely fcos dota1.0 (PECL)'''
# 10per_sparse_baseline # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining_gihead_PECL/1.0/burn-in-120000_ga_10per_train/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29554 --use_env\
#     train.py configs_dota15/sparse_new_idea_PECL/fcos_baseline_dota10.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead_PECL/1.0/burn-in-120000_ga_10per_train


# 10per_sparse_fnmining # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining_gihead_PECL/1.0/burn-in-12800_ga_sfnm-thres0.1-fn-allweight-thres1.0-beta5.0_5per_trainval2/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29566 --use_env\
#     train.py configs_dota15/sparse_new_idea_PECL/fcos_fnmining_dota10.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead_PECL/1.0/burn-in-12800_ga_sfnm-thres0.1-fn-allweight-thres1.0-beta5.0_5per_trainval2

# 10per_sparse_fnmining_gihead # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining_gihead_PECL/1.0/burn-in-12800_ga_sfnm-thres0.1-fn-allweight-thres1.0-beta5.0_gihead0712-posthr0.7-noclsloss_reggt-thr0.9_5per_trainval/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29564 --use_env\
#     train.py configs_dota15/sparse_new_idea_PECL/fcos_fnmining_gihead_dota10.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead_PECL/1.0/burn-in-12800_ga_sfnm-thres0.1-fn-allweight-thres1.0-beta5.0_gihead0712-posthr0.7-noclsloss_reggt-thr0.9_5per_trainval



'''sparsely orcnn dota1.0 (PECL)'''
# 10per_unbaisedteacher_baseline # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining_gihead_PECL/1.0/unbiased-orcnn_burn-in-12800_mining-thr0.1-roihead-negw5.0_5per_contgt-proj-posw5.0-loss0.1_lr1e-2_trainval/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29564 --use_env\
#     train.py configs_dota15/sparse_new_idea_PECL/unbaisedteacher_orientedrcnn_baseline_dota10.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead_PECL/1.0/unbiased-orcnn_burn-in-12800_mining-thr0.1-roihead-negw5.0_5per_contgt-proj-posw5.0-loss0.1_lr1e-2_trainval


# 10per_unbaisedteacher_gihead # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining_gihead_PECL/1.0/unbiased-orcnn_burn-in-12800_mining-thr0.1-roihead-negw5.0_ss-thr0.01-allloss1.0-0.5_5per_lr1e-2_trainval/terminal_log.log 2>&1
/home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29560 --use_env\
    train.py configs_dota15/sparse_new_idea_PECL/unbaisedteacher_orientedrcnn_gi_dota10.py \
    --launcher pytorch \
    --work-dir log/sparse_fnmining_gihead_PECL/1.0/unbiased-orcnn_burn-in-12800_mining-thr0.1-roihead-negw5.0_ss-thr0.01-allloss1.0-0.5_5per_lr1e-2_trainval


# 10per_unbaisedteacher_baseline_zhang # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining_gihead_PECL/1.0/zhang_burn-in-12800_thr0.2_5per_trainval_resume_wo-suploss/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29576 --use_env\
#     train.py configs_dota15/sparse_new_idea_PECL/semi_orcnn_sparse_ann_dota10p_le90.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead_PECL/1.0/zhang_burn-in-12800_thr0.2_5per_trainval_resume_wo-suploss






'''sparsely dota1.0 (s2teacher)'''

# 10per_unbaisedteacher_baseline # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining_gihead_s2teacher/1.0/unbiased-orcnn_burn-in-12800_mining-thr0.1-roihead-negw5.0_10per_lr1e-2_trainval/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29566 --use_env\
#     train.py configs_dota15/sparse_new_idea_s2teacher/unbaisedteacher_orientedrcnn_baseline_dota10.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_gihead_s2teacher/1.0/unbiased-orcnn_burn-in-12800_mining-thr0.1-roihead-negw5.0_10per_lr1e-2_trainval











'''sparsely dota1.5 (semi_ann)'''

# 10per_unbaisedteacher_gihead # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/sparse_fnmining_semi/1.5/unbiased-orcnn_burn-in-120000_mining-thr0.1-roihead-allw5.0-thr0.5_10per_train/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29564 --use_env\
#     train.py configs_dota15/sparse_new_idea_semi/unbaisedteacher_orientedrcnn_gi_dota15.py \
#     --launcher pytorch \
#     --work-dir log/sparse_fnmining_semi/1.5/unbiased-orcnn_burn-in-120000_mining-thr0.1-roihead-allw5.0-thr0.5_10per_train
