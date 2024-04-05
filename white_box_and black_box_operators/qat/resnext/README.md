Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch Resnext101 QAT tuning results with Intel® Neural Compressor.

# Prerequisite

### 1. Environment

PyTorch 1.8 or higher version is needed with pytorch_fx backend.

```Shell
cd examples/pytorch/image_recognition/torchvision_models/quantization/qat/fx
pip install -r requirements.txt
```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment).

### 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

### resnext101_64x4d

```Shell
python main.py -t -a resnext101_64x4d --pretrained /path/to/imagenet
```
