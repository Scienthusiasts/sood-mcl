#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=2,3





# training 
/usr/local/anaconda3/bin/conda run -n /data/yht/env/sood_mcl
cd /data/yht/code/sood-mcl

# 10per_mcl_train DOTA1.5
# /data/yht/env/sood_mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29558\
#     train.py configs_dota15/mcl/mcl_fcos_dota15_10p.py \
#     --launcher pytorch \
#     --work-dir log/mcl/DOTA1.5/10per


# 10per_mcl_train DOTA1.0
# /data/yht/env/sood_mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29558\
#     train.py configs_dota15/mcl/mcl_fcos_dota10_10p.py \
#     --launcher pytorch \
#     --work-dir log/mcl/DOTA1.0/10per_unsup-weight-0


# 10per_dt_train DOTA1.0
/data/yht/env/sood_mcl/bin/python -m torch.distributed.launch --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --nnodes=1 --master_port=29558\
    train.py configs_dota15/denseteacher/denseteacher_fcos_dota10_10p.py \
    --launcher pytorch \
    --work-dir log/dt/DOTA1.0/10per_unsup-weight-0
