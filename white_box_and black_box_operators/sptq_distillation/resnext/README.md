Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch tuning results with Intel® Neural Compressor.

# Prerequisite

## 1. Environment

PyTorch 1.8 or higher version is needed with pytorch_fx backend.

```shell
pip install -r requirements.txt
```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

> Note: All torchvision model names can be passed as long as they are included in `torchvision.models`, below are some examples.

### 1. SPTQ Quantization of Distilled Resnet

```shell
python sptq_of_resnext50_student.py -t -a resnext50_32x4d --pretrained /path/to/imagenet
```
