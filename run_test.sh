#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=4,5
# export CUDA_VISIBLE_DEVICES=6,7

# training 
/home/yht/.conda/envs/sood-mcl run -n sood-mcl
cd /data/yht/code/sood-mcl








'''SSOD'''
# 10per_unbaisedteacher(SPL) # CUDA_LAUNCH_BLOCKING=1 
# sh run_new.sh > log/semi_PECL/1.0/unbaisedteacher_burn-in-12800_10per/terminal_log.log 2>&1
/home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29540 --use_env\
    train.py configs_dota15/sparse_new_idea_PECL/unbaisedteacher_orientedrcnn_baseline_dota10.py \
    --launcher pytorch \
    --work-dir log/debug