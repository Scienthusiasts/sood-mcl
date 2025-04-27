#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=4,5
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





'''debug'''
# # 10per_dtbaseline_refinehead_train DOTA1.5 debug # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_refineORCNNhead_dota15_10p_debug.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/debug


# # 10per_dtbaseline_refinehead_ss_train DOTA1.5 debug # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_refineORCNNhead_ss_dota15_10p_debug.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/debug


# 10per_dtbaseline_train DOTA1.5 debug # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_dota15_10p_debug.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/debug/dtbaseline

# 10per_dtbaseline_gihead_ss_train DOTA1.5 debug # CUDA_LAUNCH_BLOCKING=1 
/home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
    train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_refineGIhead_ss_dota15_10p_debug.py \
    --launcher pytorch \
    --work-dir log/dtbaseline_gi/debug

# 10per_denseteacher_train DOTA1.5 debug # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/denseteacher/denseteacher_ss_fcos_dota15_10p_debug.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/debug/denseteacher

# 10per_mcl_train DOTA1.5 debug # CUDA_LAUNCH_BLOCKING=1 
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29550\
#     train.py configs_dota15/mcl/mcl_fcos_ss_dota15_10p_debug.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/debug/mcl

# 单卡
# /home/yht/.conda/envs/sood-mcl/bin/python \
#     train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_refineORCNNhead_dota15_10p_debug.py \
#     --work-dir log/dtbaseline/debug












'''dtbaseline'''
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





'''full_supervised'''
# denseteacher_train DOTA1.5
# sh run.sh > log/dtbaseline/DOTA1.5/full_sup/denseteacher/total-120000-iter/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29560\
#     train.py configs_dota15/denseteacher/denseteacher_fcos_dota15_full.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/DOTA1.5/full_sup/denseteacher/total-120000-iter

# mcl_train DOTA1.5
# sh run.sh > log/dtbaseline/DOTA1.5/full_sup/mcl/total-120000-iter/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29562\
#     train.py configs_dota15/mcl/mcl_fcos_dota15_full.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/DOTA1.5/full_sup/mcl/total-120000-iter




'''ss-branch'''
# 10per_dtbaseline_orcnnhead_train DOTA1.5
# sh run.sh > log/dtbaseline/DOTA1.5/ss-branch/global-w_refinehead/joint-score-sigmoid_burn-in-12800_orcnn-head_all-refine-loss_box-O2M-loss_detach_GA_ssloss-joint-jsd-dim0-w1.0_10_modify-batch-nms/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29562\
#     train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_refineORCNNhead_ss_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/DOTA1.5/ss-branch/global-w_refinehead/joint-score-sigmoid_burn-in-12800_orcnn-head_all-refine-loss_box-O2M-loss_detach_GA_ssloss-joint-jsd-dim0-w1.0_10_modify-batch-nms

# 10per_dtbaseline_gihead_train DOTA1.5
# # sh run.sh > log/dtbaseline/DOTA1.5/ss-branch/global-w_gihead/joint-score-sigmoid_burn-in-12800_gi-head_all-refine-loss_box-O2M-loss_GA_detach-fpnfeat_ssloss-joint-jsd-dim0-w1.0_avgpoolroi3/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29560\
#     train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_refineGIhead_ss_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/DOTA1.5/ss-branch/global-w_gihead/joint-score-sigmoid_burn-in-12800_gi-head_all-refine-loss_box-O2M-loss_GA_detach-fpnfeat_ssloss-joint-jsd-dim0-w1.0_avgpoolroi3


# 10per_dtbaseline_train DOTA1.5 
# 跑这个需要先把refinehead那部分注释掉
# sh run.sh > log/dtbaseline/DOTA1.5/ss-branch/global-w_no-refinehead/joint-score-sigmoid_burn-in-12800_GA_ssloss-cls-jsd-dim0-w1.0/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29560\
#     train.py configs_dota15/denseteacher_baseline/denseteacher_fcos_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/DOTA1.5/ss-branch/global-w_no-refinehead/joint-score-sigmoid_burn-in-12800_GA_ssloss-cls-jsd-dim0-w1.0

# 10per_denseteacher_train DOTA1.5
# sh run.sh > log/dtbaseline/DOTA1.5/ss-branch/denseteacher/joint-score-sigmoid_burn-in-6400_ssloss-joint-jsd-dim0-w0.01/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29562\
#     train.py configs_dota15/denseteacher/denseteacher_ss_fcos_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/DOTA1.5/ss-branch/denseteacher/joint-score-sigmoid_burn-in-6400_ssloss-joint-jsd-dim0-w0.01

# 10per_mcl_train DOTA1.5
# sh run.sh > log/dtbaseline/DOTA1.5/ss-branch/mcl/joint-score-sigmoid_burn-in-6400_ssloss-joint-jsd-dim0-w0.1/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29562\
#     train.py configs_dota15/mcl/mcl_fcos_ss_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/DOTA1.5/ss-branch/mcl/joint-score-sigmoid_burn-in-6400_ssloss-joint-jsd-dim0-w0.1

# 10per_mcl_train DOTA1.5
# sh run.sh > log/dtbaseline/DOTA1.5/ss-branch/mcl/joint-score-sigmoid_burn-in-6400_ssloss-joint-jsd-dim0-w0.1/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29562\
#     train.py configs_dota15/mcl/mcl_fcos_ss_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/dtbaseline/DOTA1.5/ss-branch/mcl/joint-score-sigmoid_burn-in-6400_ssloss-joint-jsd-dim0-w0.1





