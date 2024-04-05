import torch
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime as ort   # to inference ONNX models, we use the ONNX Runtime
import json
from PIL import Image
import base64
from io import BytesIO
from torchvision import transforms as T
resnet_preprocess_image = T.Compose([T.Resize(232), T.CenterCrop(224), T.ToTensor(
), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class ResNet():
    def __init__(self):
        self.labels = self.load_labels(
            'resnet_models/imagenet-simple-labels.json')

    def load_labels(self, path):
        with open(path) as f:
            data = json.load(f)
        return data

    def preprocess(self, input):
        input_data = resnet_preprocess_image(input)
        input_data = np.expand_dims(input_data, 0)
        return input_data

    def postprocess(self, result):
        result = self.labels[np.argmax(result)]
        return {"Prediction: ": result}


resnet_model = ResNet()
providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})]


def resnet_single_inference(img):
    # Run the model on the backend
    image = Image.open(BytesIO(base64.b64decode(
        img.encode('utf-8')))).convert('RGB')
    # get the name of the first input of the model
    input_data = resnet_model.preprocess(image)
    session = ort.InferenceSession(
        'resnet_models/wide-resnet101.onnx', providers=providers)
    result = session.run(
        [session.get_outputs()[0].name], {session.get_inputs()[0].name: input_data})
    result = resnet_model.postprocess(result)
    return result

def resnet_int8_sptq_single_inference(img):
    # Run the model on the backend
    image = Image.open(BytesIO(base64.b64decode(
        img.encode('utf-8')))).convert('RGB')

    # get the name of the first input of the model
    input_data = resnet_model.preprocess(image)
    session = ort.InferenceSession(
        'resnet_models/wide-resnet101_int8_sptq.onnx', providers=providers)
    result = session.run(
        [session.get_outputs()[0].name], {session.get_inputs()[0].name: input_data})
    result = resnet_model.postprocess(result)
    return result


def resnet_split_first_half_single_inference(img):
    # Run the model on the backend
    image = Image.open(BytesIO(base64.b64decode(
        img.encode('utf-8')))).convert('RGB')
    # get the name of the first input of the model
    input_data = resnet_model.preprocess(image)
    session = ort.InferenceSession(
        'resnet_models/wide-resnet101_split_part1.onnx', providers=providers)
    result = session.run([session.get_outputs()[0].name], {
        session.get_inputs()[0].name: input_data})
    return result

def resnet_split_second_half_single_inference(img):
    # Run the model on the backend
    # get the name of the first input of the model
    session = ort.InferenceSession(
        'resnet_models/wide-resnet101_split_part2.onnx', providers=providers)
    result = session.run([session.get_outputs()[0].name], {
        session.get_inputs()[0].name: img[0]})
    result = resnet_model.postprocess(result)
    return result

def resnet_int8_sptq_split_first_half_single_inference(img):
    # Run the model on the backend
    image = Image.open(BytesIO(base64.b64decode(
        img.encode('utf-8')))).convert('RGB')
    # get the name of the first input of the model
    input_data = resnet_model.preprocess(image)
    session = ort.InferenceSession(
        'resnet_models/wide-resnet101_int8_sptq_split_part1.onnx', providers=providers)
    result = session.run([session.get_outputs()[0].name], {
        session.get_inputs()[0].name: input_data})
    return result

def resnet_int8_sptq_split_second_half_single_inference(img):
    # Run the model on the backend
    # get the name of the first input of the model
    session = ort.InferenceSession(
        'resnet_models/wide-resnet101_int8_sptq_split_part2.onnx', providers=providers)
    result = session.run([session.get_outputs()[0].name], {
        session.get_inputs()[0].name: img[0]})
    result = resnet_model.postprocess(result)
    return result

def resnet_pruned_single_inference(img):
    # Run the model on the backend
    image = Image.open(BytesIO(base64.b64decode(
        img.encode('utf-8')))).convert('RGB')
    # get the name of the first input of the model
    input_data = resnet_model.preprocess(image)
    session = ort.InferenceSession(
        'resnet_models/wide-resnet101_pruned.onnx', providers=providers)
    result = session.run(
        [session.get_outputs()[0].name], {session.get_inputs()[0].name: input_data})
    result = resnet_model.postprocess(result)
    return result

def resnet_int8_qat_single_inference(img):
    # Run the model on the backend
    image = Image.open(BytesIO(base64.b64decode(
        img.encode('utf-8')))).convert('RGB')
    # get the name of the first input of the model
    input_data = resnet_model.preprocess(image)
    session = ort.InferenceSession(
        'resnet_models/wide-resnet101_int8_qat.onnx', providers=providers)
    result = session.run(
        [session.get_outputs()[0].name], {session.get_inputs()[0].name: input_data})
    result = resnet_model.postprocess(result)
    return result

def resnet_distilled_single_inference(img):
    # Run the model on the backend
    image = Image.open(BytesIO(base64.b64decode(
        img.encode('utf-8')))).convert('RGB')
    # get the name of the first input of the model
    input_data = resnet_model.preprocess(image)
    session = ort.InferenceSession(
        'resnet_models/wide-resnet50_distilled.onnx', providers=providers)
    result = session.run(
        [session.get_outputs()[0].name], {session.get_inputs()[0].name: input_data})
    result = resnet_model.postprocess(result)
    return result

def resnet_distilled_int8_sptq_single_inference(img):
    # Run the model on the backend
    image = Image.open(BytesIO(base64.b64decode(
        img.encode('utf-8')))).convert('RGB')
    # get the name of the first input of the model
    input_data = resnet_model.preprocess(image)
    session = ort.InferenceSession(
        'resnet_models/wide-resnet50_distilled_int8_sptq.onnx', providers=providers)
    result = session.run(
        [session.get_outputs()[0].name], {session.get_inputs()[0].name: input_data})
    result = resnet_model.postprocess(result)
    return result


