#!/bin/bash
DATA="../../../ILSVRC2012"
python -m torch.distributed.launch --master_addr=localhost --nproc_per_node=8 --master_port=8051 ./train.py \
    ${DATA} \
    --model "resnext101_64x4d.tv_in1k" \
    --num-classes 1000 \
    --pretrained \
    --batch-size 128 \
    --lr 0.175 \
    --epochs 180 \
    --warmup-epochs 0 \
    --cooldown-epochs 20 \
    --do-prune \
    --do-distillation \
    --target-sparsity 0.75 \
    --pruning-pattern "2x1" \
    --update-frequency-on-step 2000 \
    --distillation-loss-weight "1.0" \
    --output ./resnext101_pruned_models/ \
    --distributed
