# Prerequisite

## 1. Environment

```shell
pip install -r requirements.txt
```

> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

```shell
python prepare_model.py --input_model='Intel/roberta-base-mrpc' --output_model='roberta-base-mrpc.onnx'
```

```shell
python prepare_model.py --input_model='Intel/bert-base-uncased-mrpc' --output_model='bert-base-uncased-mrpc.onnx'
```


# Run

##  Partitioning

```shell
python bert_textual_model_split.py
python bert_sptq_textual_model_split.py
```
## SPTQ Partitioning

```shell
python roberta_textual_model_split.py
python roberta_sptq_textual_model_split.py
```


