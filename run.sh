#!/bin/bash

        # --total_batch_size 524288 \
        # --total_batch_size 491520 \
        # --total_batch_size 983040 \
        # --total_batch_size=589824 \
        # --input_bin="dev/data/fineweb10B/fineweb_train_*.bin" \
        # --input_val_bin=dev/data/fineweb10B/fineweb_val_000000.bin \

# python \
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,2,3,4,5
time torchrun --standalone --nproc_per_node 5 \
    train.py \
        --input_bin="10B" \
        --total_batch_size=491520 \
        --sequence_length=1024 \
        --batch_size=32 \
        --weight_decay=0.1  \
        --num_iterations=20000 \
        --model=l6 \
        --warmup_iters=400 \
        --learning_rate_decay_frac=0.1 \
        --output_dir=logs \
        --compile=1 \
        --tensorcores=1 \
        --dtype=bfloat16 \
        --val_loss_every=10 \
