#!/bin/bash

        # --total_batch_size 524288 \
        # --total_batch_size 491520 \
        # --total_batch_size 983040 \
        # --total_batch_size=589824 \
        # --input_bin="dev/data/fineweb10B/fineweb_train_*.bin" \
        # --input_val_bin=dev/data/fineweb10B/fineweb_val_000000.bin \
        # --num_iterations=20000 \
        # 786432
        # 589824
        # 655360

# python \
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=

# export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES,0
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES,1
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES,2
# export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES,3
# export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES,4
# export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES,5


export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:1}
echo $CUDA_VISIBLE_DEVICES


time torchrun --standalone --nproc_per_node 2 \
    train.py \
        --tokens_per_step=589824 \
        --position_encoding='bam' \
        --model_size='l12' \
        --sequence_length=1024 \
        --batch_size=18 \
        --weight_decay=0.1  \
        --learning_rate_decay_frac=0.1 \
        --compile \
        --tensorcores \
        --val_loss_every=32 \
        --dtype=bfloat16 \
        --learning_rate 1e-3 \


