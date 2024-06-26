# Prerequisite

## Environment

Recommend python 3.6 or higher version.

```shell
pip install -r requirements.txt
```

# Run Distillation pipeline

## MRPC task

```bash
python run_glue_no_trainer_distillation.py --task_name mrpc \
      --model_name_or_path lyeonii/bert-small \
      --teacher_model_name_or_path Intel/bert-base-uncased-mrpc \
      --do_distillation --max_seq_length 128 --per_device_train_batch_size 32 \
      --learning_rate 1e-4 --num_train_epochs 9 --output_dir ./bert-small \
      --loss_weights 0 1 --temperature 2 --seed 5143
```
