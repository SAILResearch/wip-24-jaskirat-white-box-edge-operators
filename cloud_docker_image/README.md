# Prerequisite

## 1. Environment

```shell
pip install -r requirements.txt
```

## 3. Prepare Dataset

Download Validation data
1. [ILSVR2012 Imagenet dataset](http://www.image-net.org/challenges/LSVRC/2012/downloads).
Create a folder imagenet and store the validation images in it

## 3. Create a Network Bridge for Cloud Docker Container

```shell
docker network create cloud_network
```

## 4. Prepare Cloud Docker Container

```shell
docker build -t cloud_inference .
docker run -it --net=cloud_network --cap-add=NET_ADMIN --name cloud_inference_container --gpus device=0 -e PYTHONUNBUFFERED=1 --cpus="16" --memory="64g" -p 5002:5002 -d cloud_inference
```

## 5. Run Accuracy Scripts for textual operators within Cloud container

```bash
# benchmark ONNX model
bash run_benchmark.sh --input_model=[/path/to/.onnx] --dataset_location=[dataset_name] --tokenizer=[model_name_or_path] --mode=[accuracy] --batch_size=[16]
# benchmark PyTorch model
bash run_benchmark.sh --input_model=[model_name_or_path|/path/to/saved_results] --dataset_location=[dataset_name] --mode=[accuracy] --int8=[true|false] --batch_size=[16]
```

## 6. Run Accuracy Scripts for Image-based operators within Cloud container
```bash
python resnet_accuracy.py
python resnext_accuracy.py
```


