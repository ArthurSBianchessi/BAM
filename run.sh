#!/bin/bash

        # --total_batch_size 524288 \
        # --total_batch_size 491520 \
        # --total_batch_size 983040 \
        # --total_batch_size=589824 \
        # --input_bin="dev/data/fineweb10B/fineweb_train_*.bin" \
        # --input_val_bin=dev/data/fineweb10B/fineweb_val_000000.bin \
        # --num_iterations=20000 \

# python \
export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=1,2,3,4,5
# export CUDA_VISIBLE_DEVICES=2,3,4,5
# export CUDA_VISIBLE_DEVICES=0,1,2
# export CUDA_VISIBLE_DEVICES=3,4,5
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=4,5


time torchrun --standalone --nproc_per_node 5 \
    train.py \
        --tokens_per_step=655360 \
        --position_encoding='bam' \
        --model_size='l6' \
        --sequence_length=1024 \
        --batch_size=32 \
        --weight_decay=0.1  \
        --warmup_iters=400 \
        --learning_rate_decay_frac=0.1 \
        --compile=1 \
        --tensorcores=1 \
        --val_loss_every=32 \
        --dtype=bfloat16 \
        --shape_lr=1e-1 \
        --scale_lr=1e-1 \
        --loc_lr=1e-1 \



time torchrun --standalone --nproc_per_node 5 \
    train.py \
        --tokens_per_step=655360 \
        --position_encoding='bam' \
        --model_size='l6' \
        --sequence_length=1024 \
        --batch_size=32 \
        --weight_decay=0.1  \
        --warmup_iters=400 \
        --learning_rate_decay_frac=0.1 \
        --compile=1 \
        --tensorcores=1 \
        --val_loss_every=32 \
        --dtype=bfloat16 \
        # --shape_lr=1e-1 \
        # --scale_lr=1e-1 \
        # --loc_lr=1e-1 \

        

time torchrun --standalone --nproc_per_node 5 \
    train.py \
        --tokens_per_step=655360 \
        --position_encoding='alibi' \
        --model_size='l24' \
        --sequence_length=1024 \
        --batch_size=4 \
        --weight_decay=0.1  \
        --warmup_iters=400 \
        --learning_rate_decay_frac=0.1 \
        --compile=1 \
        --tensorcores=1 \
        --val_loss_every=32 \
        --dtype=bfloat16 \

    # parser.add_argument("--shape_lr", type=float, default=1e-4, help="learning rate for shape exponent")
    # parser.add_argument("--scale_lr", type=float, default=1e-4, help="learning rate for scale multiplier")
    # parser.add_argument("--loc_lr", type=float, default=1e-4, help="learning rate for location sum")





# """
# Olá, Arthur,

# Boa tarde.

 

# As disciplinas indicadas - Teoria da Computação (54685-02) e Logicas e Álgebras Computacionais (34648-02) não serão oferecidas na modalidade GPG, foram publicadas equivocadamente no site GPG e já solicitamos a correção.

 

# Por favor indique outras disciplinas de seu interesse para que possamos efetuar a reserva de matrícula, e caso alguma disciplina estiver esgotada, nos sinalize para abertura de vaga.

 

# Atenciosamente, Vanessa Ruschel
# """"
# Realizei a pré-matrícula para ambas as disciplinas em Janeiro. Como recebi esse email ontem as 17 horas, neccessito de mais 4 créditos de eletivas para me formar esse semestre.