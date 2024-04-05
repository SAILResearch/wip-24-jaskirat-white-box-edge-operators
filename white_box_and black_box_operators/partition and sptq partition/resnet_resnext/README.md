# Prerequisite

## 1. Environment

```shell
pip install -r requirements.txt
```

> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

```shell
python prepare_resnet_original_model.py --output_model='wide-resnet101.onnx'
```

```shell
python prepare_resnext_original_model.py --output_model='resnext101.onnx'
```


# Run

##  Partitioning

```shell
python resnet_partition.py
python resnext_partition.py
```
## SPTQ Partitioning

```shell
python resnet_sptq_partition.py
python resnext_sptq_partition.py
```


