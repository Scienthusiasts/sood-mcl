#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=4,5
# export CUDA_VISIBLE_DEVICES=6,7

# training 
/home/yht/.conda/envs/sood-mcl run -n sood-mcl
cd /data/yht/code/sood-mcl








"""debug"""

'''denseteacher'''
# 10per_denseteacher # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/new_idea/denseteacher_fcos_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/debug

# 10per_denseteacher_ss # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/new_idea/denseteacher_fcos_ss_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/debug

# 10per_denseteacher_gihead # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29552\
#     train.py configs_dota15/new_idea/denseteacher_fcos_gihead_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/debug

# 10per_denseteacher_ss_gihead # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/new_idea/denseteacher_fcos_ss_gihead_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/debug

'''mcl'''
# 10per_mcl # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/new_idea/mcl_fcos_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/debug

# 10per_mcl_ss # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/new_idea/mcl_fcos_ss_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/debug

# 10per_mcl_gihead # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/new_idea/mcl_fcos_gihead_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/debug

# 10per_mcl_ss_gihead # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/new_idea/mcl_fcos_ss_gihead_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/debug

'''global-w'''
# # 10per_globalw_gihead_ss 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550 --use_env\
#     train.py configs_dota15/new_idea/globalw_fcos_ss_gihead_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/debug






"""train"""

'''denseteacher'''
# 10per_denseteacher 
# sh run_new.sh > log/new/denseteacher/baseline/burn-in-6400_top0.01_boxloss/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29562\
#     train.py configs_dota15/new_idea/denseteacher_fcos_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/denseteacher/baseline/burn-in-6400_top0.01_boxloss

# 10per_denseteacher_gihead
# sh run_new.sh > log/new/denseteacher/gihead/burn-in-6400_top0.03_O2M-only-boxloss_refine-allloss_avgpoolroi/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/new_idea/denseteacher_fcos_gihead_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/denseteacher/gihead/burn-in-6400_top0.03_O2M-only-boxloss_refine-allloss_avgpoolroi

# 10per_denseteacher_ss 
# sh run_new.sh > log/new/denseteacher/ss/burn-in-6400_top0.03_joint-jsdloss-dim0-w0.05_roiuloss-w0.05/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29574\
#     train.py configs_dota15/new_idea/denseteacher_fcos_ss_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/denseteacher/ss/burn-in-6400_top0.03_joint-jsdloss-dim0-w0.05_roiuloss-w0.05

# 10per_denseteacher_gihead_ss 
# sh run_new.sh > log/new/denseteacher/ss_gihead/burn-in-6400_top0.03_O2M-only-boxloss_refine-allloss_avgpoolroi_joint-jsdloss-dim0-w0.1_roiuloss-w0.1/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29562\
#     train.py configs_dota15/new_idea/denseteacher_fcos_ss_gihead_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/denseteacher/ss_gihead/burn-in-6400_top0.03_O2M-only-boxloss_refine-allloss_avgpoolroi_joint-jsdloss-dim0-w0.1_roiuloss-w0.1



'''mcl'''
# 10per_mcl 
# sh run_new.sh > log/new/mcl/baseline/burn-in-6400/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29562\
#     train.py configs_dota15/new_idea/mcl_fcos_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/mcl/baseline/burn-in-6400

# 10per_mcl_gihead 
# sh run_new.sh > log/new/mcl/gihead/burn-in-6400_O2M-only-boxloss_refine-allloss_avgpoolroi/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29566\
#     train.py configs_dota15/new_idea/mcl_fcos_gihead_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/mcl/gihead/burn-in-6400_O2M-only-boxloss_refine-allloss_avgpoolroi

# 10per_mcl_ss 
# sh run_new.sh > log/new/mcl/ss/burn-in-6400_joint-jsdloss-dim0-w0.1_roiuloss-w0.1/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29560\
#     train.py configs_dota15/new_idea/mcl_fcos_ss_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/mcl/ss/burn-in-6400_joint-jsdloss-dim0-w0.1_roiuloss-w0.1

# 10per_mcl_gihead_ss 
# sh run_new.sh > log/new/mcl/ss_gihead/burn-in-6400_top0.03_O2M-only-boxloss_refine-allloss_avgpoolroi_joint-jsdloss-dim0-w0.1_roiuloss-w0.1/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29562\
#     train.py configs_dota15/new_idea/mcl_fcos_ss_gihead_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/mcl/ss_gihead/burn-in-6400_top0.03_O2M-only-boxloss_refine-allloss_avgpoolroi_joint-jsdloss-dim0-w0.1_roiuloss-w0.1



'''globalw'''
# 10per_globalw 
# sh run_new.sh > log/new/globalw/baseline/burn-in-6400/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29562\
#     train.py configs_dota15/new_idea/globalw_fcos_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/mcl/baseline/burn-in-6400

# 10per_globalw_gihead 
# sh run_new.sh > log/new/globalw/gihead/burn-in-6400_O2M-only-boxloss_refine-allloss_sharefcheadroi_addnoise-p0.5_pe/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29562\
#     train.py configs_dota15/new_idea/globalw_fcos_gihead_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/globalw/gihead/burn-in-6400_O2M-only-boxloss_refine-allloss_sharefcheadroi_addnoise-p0.5_pe

# 10per_globalw_ss # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/new/globalw/ss/burn-in-6400_joint-jsdloss-dim0-w0.05-bilinear_roiuloss-w0.05/terminal_log.log 2>&1
# CUDA_LAUNCH_BLOCKING=1 /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29566 --use_env\
#     train.py configs_dota15/new_idea/globalw_fcos_ss_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/globalw/ss/burn-in-6400_joint-jsdloss-dim0-w0.05-bilinear_roiuloss-w0.05

# 10per_globalw_ss_clip_distill 
# sh run_new.sh > log/new/globalw/ss_clipdistill/burn-in-6400_joint-jsdloss-dim0-w0.05_roiuloss-w0.05_ga_distill-l1loss-w1.0_1layer-proj-linear3/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29560 --use_env\
#     train.py configs_dota15/new_idea/globalw_fcos_ss_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/globalw/ss_clipdistill/burn-in-6400_joint-jsdloss-dim0-w0.05_roiuloss-w0.05_ga_distill-l1loss-w1.0_1layer-proj-linear3


# 10per_globalw_gihead_ss 
# sh run_new.sh > log/new/globalw/ss_gihead/burn-in-6400_top0.03_O2M-only-boxloss_refine-allloss_sharefcheadroi_addnoise-p0.5_joint-jsdloss-dim0-w0.1_roiuloss-w0.1_ga_pe2/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29560\
#     train.py configs_dota15/new_idea/globalw_fcos_ss_gihead_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/new/globalw/ss_gihead/burn-in-6400_top0.03_O2M-only-boxloss_refine-allloss_sharefcheadroi_addnoise-p0.5_joint-jsdloss-dim0-w0.1_roiuloss-w0.1_ga_pe2













 
"""sparsely annotated 稀疏标注任务"""

# 10per_globalw_gihead_ss 
# sh run_new.sh > log/sparse_ann/globalw/ss_gihead/burn-in-6400_top0.03_O2M-only-boxloss_refine-allloss_sharefcheadroi_joint-jsdloss-dim0-w0.1_roiuloss-w0.1_ga_pe_alltrain-unsup2/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29560 --use_env\
#     train.py configs_dota15/new_idea/sparse_ann/globalw_fcos_ss_gihead_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/sparse_ann/globalw/ss_gihead/burn-in-6400_top0.03_O2M-only-boxloss_refine-allloss_sharefcheadroi_joint-jsdloss-dim0-w0.1_roiuloss-w0.1_ga_pe_alltrain-unsup2

# 10per_globalw_ss 
# sh run_new.sh > log/sparse_ann/globalw/ss/burn-in-6400_top0.03_joint-jsdloss-dim0-w0.1_roiuloss-w0.1_ga_pe_alltrain-unsup_wo-nomal-unsuploss/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29564 --use_env\
#     train.py configs_dota15/new_idea/sparse_ann/globalw_fcos_ss_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/sparse_ann/globalw/ss/burn-in-6400_top0.03_joint-jsdloss-dim0-w0.1_roiuloss-w0.1_ga_pe_alltrain-unsup_wo-nomal-unsuploss


















"""sparsely annotated without sup branch 稀疏标注任务(去除有监督分支, 且只保留强弱增强的数据)"""

'''debug'''
# 10per_globalw 
# CUDA_LAUNCH_BLOCKING=1  /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550 --use_env\
#     train.py configs_dota15/sparse_new_idea/globalw_fcos_ss_gihead_dota10_10p_wosupbranch.py \
#     --launcher pytorch \
#     --work-dir log/new_sparse/debug

# sh run_new.sh > log/new_sparse/debug/terminal_log.log 2>&1
# CUDA_LAUNCH_BLOCKING=1  /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550 --use_env\
#     train.py configs_dota15/sparse_new_idea/globalw_fcos_ss_gihead_dota15_10p_wosupbranch.py \
#     --launcher pytorch \
#     --work-dir log/new_sparse/debug




'''sparsely globalw dota1.0'''
# 10per_globalw # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/new_sparse/1.0/globalw_burn-in-12800_ga_sfpm-thres0.05-fn-allweight-thres1.0-beta5.0_10per/terminal_log.log 2>&1
/home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29564 --use_env\
    train.py configs_dota15/sparse_new_idea/globalw_fcos_ss_gihead_dota10_10p_wosupbranch.py \
    --launcher pytorch \
    --work-dir log/new_sparse/1.0/globalw_burn-in-12800_ga_sfpm-thres0.05-fn-allweight-thres1.0-beta5.0_10per



'''sparsely globalw dota1.5'''
# 10per_globalw # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/new_sparse/1.5/globalw_burn-in-120000_ga_1per/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29566 --use_env\
#     train.py configs_dota15/sparse_new_idea/globalw_fcos_ss_gihead_dota10_10p_wosupbranch.py \
#     --launcher pytorch \
#     --work-dir log/new_sparse/1.5/globalw_burn-in-120000_ga_1per


