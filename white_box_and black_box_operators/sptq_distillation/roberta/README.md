# Prerequisite

## 1. Environment

```shell
pip install neural-compressor
pip install -r requirements.txt
```

> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

```bash
python prepare_model.py  --input_model=Intel/roberta-base-mrpc  --output_model=roberta-base-mrpc.onnx
```

## 3. Prepare Dataset

Download the GLUE data with `prepare_data.sh` script.

```shell
export GLUE_DIR=/path/to/glue_data
export TASK_NAME=MRPC 

bash prepare_data.sh --data_dir=$GLUE_DIR --task_name=$TASK_NAME
```

# Run

## 1. Quantization

SPTQ of Distilled models:

```bash
bash run_quant.sh --input_model=/path/to/model \ # model path as *.onnx
                   --output_model=/path/to/model_tune \
                   --dataset_location=path/to/glue/data \
                   --quant_format="QDQ"
```