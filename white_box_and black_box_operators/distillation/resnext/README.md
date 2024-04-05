Step-by-Step
============

This is an example to show the usage of distillation.

# Prerequisite

## 1. Environment
```shell
pip install -r requirements.txt
```

## 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

Command is shown as below:

```shell
bash run_resnext_distillation.sh
```
> Note: `--topology` is the student model and `--teacher` is the teacher model.
