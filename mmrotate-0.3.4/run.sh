#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=4,5
export CUDA_VISIBLE_DEVICES=6,7

# training 
/home/yht/.conda/envs/sood-mcl run -n sood-mcl
cd /data/yht/code/sood-mcl




# fcos_sparse_12e_10per # CUDA_LAUNCH_BLOCKING=1 
# sh run.sh > log/fcos_sparse_12e_10per2/terminal_log.log 2>&1
# /home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29560 --use_env\
#     ./tools/train.py configs/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90_sparse_10per.py \
#     --launcher pytorch \
#     --work-dir log/fcos_sparse_12e_10per2



# fcos_sparse_gi # CUDA_LAUNCH_BLOCKING=1 
# sh run.sh > log/orcnn_gi0728-onlyupdate-detach_3x_train-5per/terminal_log.log 2>&1
/home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29566 --use_env\
    ./mmrotate-0.3.4/tools/train.py mmrotate-0.3.4/configs/oriented_rcnn/oriented_rcnn_r50_fpn_gi_3x_dota_le90.py \
    --launcher pytorch \
    --work-dir mmrotate-0.3.4/log/orcnn_gi0728-onlyupdate-detach_3x_train-5per
