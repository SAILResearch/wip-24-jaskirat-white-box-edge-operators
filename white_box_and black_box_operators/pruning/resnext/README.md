# Step by Step
This document describes the step-by-step instructions for pruning resnext101_64x4d on ImageNet dataset. The example refers **pytorch-image-model[](https://github.com/huggingface/pytorch-image-models)**, a popular package for PyTorch image models.

# Prerequisite
## Environment
First, please make sure that you have successfully installed neural_compressor.
```bash
# install dependencies
pip install -r requirements.txt
```
## Prepare Dataset
Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:
```bash
ls /path/to/imagenet
train  val
```

# Pruning
Go to the script run_resnext101_prune.sh. Please get familiar with some parameters of pruning by referring to our [Pruning API README](https://github.com/intel/neural-compressor/tree/master/neural_compressor/compression/pruner)

```bash
DATA="/path/to/your/dataset/"
python -m torch.distributed.launch --master_addr=localhost --nproc_per_node=8 --master_port=8050 \
    ./train.py \
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
    --output ./path/save/your/models/ \
    --distributed
```
After configs are settled, just run:
```bash
sh run_resnext101_prune.sh
```
