Step-by-Step
============
This document list steps of reproducing Intel Optimized PyTorch roberta models quantization and benchmarking results via Neural Compressor with quantization aware training.
Our example comes from [Huggingface/transformers](https://github.com/huggingface/transformers)

# Prerequisite
## Environment
Python 3.6 or higher version is recommended.
The dependent packages are listed in `requirements.txt`, please install them as follows,
```shell
cd examples/pytorch/nlp/huggingface_models/text-classification/quantization/qat/fx
pip install -r requirements.txt
```
# Run
## 1. Enable Intel/roberta-base-mrpc example with the auto QAT operator of Neural Compressor
  The changes made are as the following:
  * edit run_glue.py:  
    - For quantization, We used neural_compressor in it.  
    - For training, we enabled early stop strategy.  
## 2. To get tuned model and its accuracy: 
```shell
    bash run_quant.sh --input_model=Intel/roberta-base-mrpc  --output_model=saved_results
```

or

``` shell
    python run_glue.py \
        --model_name_or_path ${input_model} \
        --task_name ${task_name} \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_eval_batch_size ${batch_size} \
        --per_device_train_batch_size ${batch_size} \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --output_dir ${output_model} --overwrite_output_dir \
        --eval_steps 300 \
        --save_steps 300 \
        --greater_is_better True \
        --load_best_model_at_end True \
        --evaluation_strategy steps \
        --save_strategy steps \
        --metric_for_best_model f1 \
        --save_total_limit 1 \
        --tune
```