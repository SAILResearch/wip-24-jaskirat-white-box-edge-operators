Step-by-Step
============

This document presents step-by-step instructions for pruning Huggingface models using the Intel® Neural Compressor.

# Prerequisite

## 1. Environment

PyTorch 1.8 or higher version is needed with pytorch_fx backend.

```shell
pip install -r examples/pytorch/nlp/huggingface_models/text-classification/pruning/eager/requirements.txt
```

## 2. Prepare Dataset

The dataset will be downloaded automatically from the datasets Hub.
See more about loading [huggingface dataset](https://huggingface.co/docs/datasets/loading_datasets.html)


### Run 
```shell
python3 run_glue_no_trainer.py --model_name_or_path Intel/bert-base-uncased-mrpc --task_name mrpc --max_length 128 --per_device_train_batch_size 16 --learning_rate 5e-5 --distill_loss_weight 2.0 --num_train_epochs 15 --weight_decay 5e-5 --cooldown_epochs 5 --sparsity_warm_epochs 1 --lr_scheduler_type constant --do_prune --output_dir ./sparse_mrpc_bertbase --target_sparsity 0.9 --pruning_pattern 4x1 --pruning_frequency 500
```