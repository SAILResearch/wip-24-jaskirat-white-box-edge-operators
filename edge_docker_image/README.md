# Prerequisite

## 1. Prepare Dataset

Download Validation data
1. [ILSVR2012 Imagenet dataset](http://www.image-net.org/challenges/LSVRC/2012/downloads).

Create a folder imagenet and store the validation images in it

## 2. Create a Network Bridge for connecting Mobile and Edge Docker Containers in a shared server

```shell
docker network create mobile_edge_network
```

## 3. Prepare Edge Docker Container 

```shell
docker build -t edge_inference .
docker run -it --net=mobile_edge_network --cap-add=NET_ADMIN --name edge_inference_container -e PYTHONUNBUFFERED=1 --cpus="8" --memory="16g" -p 5001:5001 -d edge_inference
```
## 4. SSh tunneling of the shared server with the external server to exchange API requests between Edge and Cloud Docker containers

```shell
ssh -o ServerAliveInterval=60 -f -N -L :5002:localhost:5002 username@servername
```

## 5. Run Accuracy Scripts for textual operators within Edge container
```bash
# benchmark ONNX model
bash run_benchmark.sh --input_model=[/path/to/.onnx] --dataset_location=[dataset_name] --tokenizer=[model_name_or_path] --mode=[accuracy] --batch_size=[16]
# benchmark PyTorch model
bash run_benchmark.sh --input_model=[model_name_or_path|/path/to/saved_results] --dataset_location=[dataset_name] --mode=[accuracy] --int8=[true|false] --batch_size=[16]
```
## 6. Run Accuracy Scripts for Image-based operators within Edge container
```bash
python resnet_accuracy.py
python resnext_accuracy.py
```
