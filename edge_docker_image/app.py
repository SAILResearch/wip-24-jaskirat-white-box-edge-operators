from flask import Flask, request
import numpy as np
from resnet_inference import *
from resnext_inference import *
from bert_inference import *
from roberta_inference import *
import requests
import json
import base64
app = Flask(__name__)


@app.route('/run_cloud_single_inference_resnet', methods=['GET', 'POST'])
def run_cloud_single_inference_resnet():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_resnet',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_resnet_int8_qat', methods=['GET', 'POST'])
def run_cloud_single_inference_resnet_int8_qat():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_resnet_int8_qat',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_resnet_int8_sptq', methods=['GET', 'POST'])
def run_cloud_single_inference_resnet_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_resnet_int8_sptq',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_resnet_pruned', methods=['GET', 'POST'])
def run_cloud_single_inference_resnet_pruned():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_resnet_pruned',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_resnet_distilled', methods=['GET', 'POST'])
def run_cloud_single_inference_resnet_distilled():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_resnet_distilled',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_resnet_distilled_int8_sptq', methods=['GET', 'POST'])
def run_cloud_single_inference_resnet_distilled_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_resnet_distilled_int8_sptq',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)
        
@app.route('/run_cloud_single_inference_resnext', methods=['GET', 'POST'])
def run_cloud_single_inference_resnext():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_resnext',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_resnext_int8_sptq', methods=['GET', 'POST'])
def run_cloud_single_inference_resnext_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_resnext_int8_sptq',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)
        
@app.route('/run_cloud_single_inference_resnext_int8_qat', methods=['GET', 'POST'])
def run_cloud_single_inference_resnext_int8_qat():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_resnext_int8_qat',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_resnext_pruned', methods=['GET', 'POST'])
def run_cloud_single_inference_resnext_pruned():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_resnext_pruned',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_resnext_distilled', methods=['GET', 'POST'])
def run_cloud_single_inference_resnext_distilled():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_resnext_distilled',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_resnext_distilled_int8_sptq', methods=['GET', 'POST'])
def run_cloud_single_inference_resnext_distilled_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_resnext_distilled_int8_sptq',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_bert', methods=['GET', 'POST'])
def run_cloud_single_inference_bert():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_bert',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_bert_int8_sptq', methods=['GET', 'POST'])
def run_cloud_single_inference_bert_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_bert_int8_sptq',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_bert_int8_qat', methods=['GET', 'POST'])
def run_cloud_single_inference_bert_int8_qat():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_bert_int8_qat',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_bert_pruned', methods=['GET', 'POST'])
def run_cloud_single_inference_bert_pruned():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_bert_pruned',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_bert_distilled', methods=['GET', 'POST'])
def run_cloud_single_inference_bert_distilled():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_bert_distilled',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_bert_distilled_int8_sptq', methods=['GET', 'POST'])
def run_cloud_single_inference_bert_distilled_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_bert_distilled_int8_sptq',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_roberta', methods=['GET', 'POST'])
def run_cloud_single_inference_roberta():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_roberta',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_roberta_int8_sptq', methods=['GET', 'POST'])
def run_cloud_single_inference_roberta_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_roberta_int8_sptq',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_roberta_int8_qat', methods=['GET', 'POST'])
def run_cloud_single_inference_roberta_int8_qat():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_roberta_int8_qat',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)
        
@app.route('/run_cloud_single_inference_roberta_pruned', methods=['GET', 'POST'])
def run_cloud_single_inference_roberta_pruned():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_roberta_pruned',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_roberta_distilled', methods=['GET', 'POST'])
def run_cloud_single_inference_roberta_distilled():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_roberta_distilled',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_cloud_single_inference_roberta_distilled_int8_sptq', methods=['GET', 'POST'])
def run_cloud_single_inference_roberta_distilled_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = requests.post('http://172.17.0.1:5002/run_cloud_single_inference_roberta_distilled_int8_sptq',
                                   json=request.json, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}).json()
            return json.dumps(result)

@app.route('/run_mobile_edge_split_single_inference_resnet', methods=['GET', 'POST'])
def run_mobile_edge_split_single_inference_resnet():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = request.json
            result = [np.frombuffer(base64.b64decode(i), 'float32').reshape(
                1, 1024, 14, 14) for i in result]
            result = resnet_split_second_half_single_inference(result)
            return json.dumps(result)

@app.route('/run_mobile_edge_split_single_inference_resnet_int8_sptq', methods=['GET', 'POST'])
def run_mobile_edge_split_single_inference_resnet_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = request.json
            result = [np.frombuffer(base64.b64decode(i), 'float32').reshape(
                1, 1024, 14, 14) for i in result]
            result = resnet_int8_sptq_split_second_half_single_inference(
                result)
            return json.dumps(result)

@app.route('/run_mobile_edge_split_single_inference_bert', methods=['GET', 'POST'])
def run_mobile_edge_split_single_inference_bert():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = request.json
            result = [np.frombuffer(base64.b64decode(result[0]), 'float32').reshape(1,128,768), np.frombuffer(base64.b64decode(result[1]), 'float32').reshape(1,1,1,128)]
            result = bert_split_second_half_single_inference(result)
            return json.dumps(result)

@app.route('/run_mobile_edge_split_single_inference_bert_int8_sptq', methods=['GET', 'POST'])
def run_mobile_edge_split_single_inference_bert_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = request.json
            result = [np.frombuffer(base64.b64decode(result[0]), 'float32').reshape(1,128,768), np.frombuffer(base64.b64decode(result[1]), 'int32').reshape(1,128)]
            result = bert_int8_sptq_split_second_half_single_inference(result)
            return json.dumps(result)

@app.route('/run_mobile_edge_split_single_inference_roberta', methods=['GET', 'POST'])
def run_mobile_edge_split_single_inference_roberta():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = request.json
            result = [np.frombuffer(base64.b64decode(result[0]), 'float32').reshape(1,128,768), np.frombuffer(base64.b64decode(result[1]), 'float32').reshape(1,1,1,128)]
            result = roberta_split_second_half_single_inference(result)
            return json.dumps(result)

@app.route('/run_mobile_edge_split_single_inference_roberta_int8_sptq', methods=['GET', 'POST'])
def run_mobile_edge_split_single_inference_roberta_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = request.json
            result = [np.frombuffer(base64.b64decode(result[0]), 'float32').reshape(1,128,768), np.frombuffer(base64.b64decode(result[1]), 'int32').reshape(1,128)]
            result = roberta_int8_sptq_split_second_half_single_inference(result)
            return json.dumps(result)

@app.route('/run_edge_single_inference_resnet', methods=['GET', 'POST'])
def run_edge_single_inference_resnet():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = resnet_single_inference(request.json[0])
            return json.dumps(result)

@app.route('/run_edge_single_inference_resnet_pruned', methods=['GET', 'POST'])
def run_edge_single_inference_resnet_pruned():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = resnet_pruned_single_inference(request.json[0])
            return json.dumps(result)

@app.route('/run_edge_single_inference_resnet_int8_sptq', methods=['GET', 'POST'])
def run_edge_single_inference_resnet_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = resnet_int8_sptq_single_inference(request.json[0])
            return json.dumps(result)

@app.route('/run_edge_single_inference_resnet_int8_qat', methods=['GET', 'POST'])
def run_edge_single_inference_resnet_int8_qat():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = resnet_int8_qat_single_inference(request.json[0])
            return json.dumps(result)

@app.route('/run_edge_single_inference_resnet_distilled', methods=['GET', 'POST'])
def run_edge_single_inference_resnet_distilled():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = resnet_distilled_single_inference(request.json[0])
            return json.dumps(result)

@app.route('/run_edge_single_inference_resnet_distilled_int8_sptq', methods=['GET', 'POST'])
def run_edge_single_inference_resnet_distilled_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = resnet_distilled_int8_sptq_single_inference(request.json[0])
            return json.dumps(result)

@app.route('/run_mobile_edge_split_single_inference_resnext', methods=['GET', 'POST'])
def run_mobile_edge_split_single_inference_resnext():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = request.json
            result = [np.frombuffer(base64.b64decode(i), 'float32').reshape(
                1, 1024, 14, 14) for i in result]
            result = resnext_split_second_half_single_inference(
                result)
            return json.dumps(result)

@app.route('/run_mobile_edge_split_single_inference_resnext_int8_sptq', methods=['GET', 'POST'])
def run_mobile_edge_split_single_inference_resnext_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = request.json
            result = [np.frombuffer(base64.b64decode(i), 'float32').reshape(
                1, 1024, 14, 14) for i in result]
            result = resnext_int8_sptq_split_second_half_single_inference(
                result)
            return json.dumps(result)

@app.route('/run_edge_single_inference_resnext', methods=['GET', 'POST'])
def run_edge_single_inference_resnext():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = resnext_single_inference(request.json[0])
            return json.dumps(result)

@app.route('/run_edge_single_inference_resnext_int8_sptq', methods=['GET', 'POST'])
def run_edge_single_inference_resnext_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = resnext_int8_sptq_single_inference(request.json[0])
            return json.dumps(result)

@app.route('/run_edge_single_inference_resnext_pruned', methods=['GET', 'POST'])
def run_edge_single_inference_resnext_pruned():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = resnext_pruned_single_inference(request.json[0])
            return json.dumps(result)

@app.route('/run_edge_single_inference_resnext_int8_qat', methods=['GET', 'POST'])
def run_edge_single_inference_resnext_int8_qat():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = resnext_int8_qat_single_inference(request.json[0])
            return json.dumps(result)

@app.route('/run_edge_single_inference_resnext_distilled', methods=['GET', 'POST'])
def run_edge_single_inference_resnext_distilled():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = resnext_distilled_single_inference(request.json[0])
            return json.dumps(result)

@app.route('/run_edge_single_inference_resnext_distilled_int8_sptq', methods=['GET', 'POST'])
def run_edge_single_inference_resnext_distilled_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = resnext_distilled_int8_sptq_single_inference(request.json[0])
            return json.dumps(result)

@app.route('/run_edge_single_inference_bert', methods=['GET', 'POST'])
def run_edge_single_inference_bert():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = bert_single_inference(request.json)
            return json.dumps(result)

@app.route('/run_edge_single_inference_bert_int8_sptq', methods=['GET', 'POST'])
def run_edge_single_inference_bert_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = bert_int8_sptq_single_inference(request.json)
            return json.dumps(result)

@app.route('/run_edge_single_inference_bert_int8_qat', methods=['GET', 'POST'])
def run_edge_single_inference_bert_int8_qat():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = bert_int8_qat_single_inference(request.json)
            return json.dumps(result)

@app.route('/run_edge_single_inference_bert_pruned', methods=['GET', 'POST'])
def run_edge_single_inference_bert_pruned():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = bert_pruned_single_inference(request.json)
            return json.dumps(result)

@app.route('/run_edge_single_inference_bert_distilled', methods=['GET', 'POST'])
def run_edge_single_inference_bert_distilled():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = bert_distilled_single_inference(request.json)
            return json.dumps(result)

@app.route('/run_edge_single_inference_bert_distilled_int8_sptq', methods=['GET', 'POST'])
def run_edge_single_inference_bert_distilled_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = bert_distilled_int8_sptq_single_inference(request.json)
            return json.dumps(result)

@app.route('/run_edge_single_inference_roberta', methods=['GET', 'POST'])
def run_edge_single_inference_roberta():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = roberta_single_inference(request.json)
            return json.dumps(result)

@app.route('/run_edge_single_inference_roberta_int8_sptq', methods=['GET', 'POST'])
def run_edge_single_inference_roberta_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = roberta_int8_sptq_single_inference(request.json)
            return json.dumps(result)

@app.route('/run_edge_single_inference_roberta_int8_qat', methods=['GET', 'POST'])
def run_edge_single_inference_roberta_int8_qat():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = roberta_int8_qat_single_inference(request.json)
            return json.dumps(result)

@app.route('/run_edge_single_inference_roberta_pruned', methods=['GET', 'POST'])
def run_edge_single_inference_roberta_pruned():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = roberta_pruned_single_inference(request.json)
            return json.dumps(result)

@app.route('/run_edge_single_inference_roberta_distilled', methods=['GET', 'POST'])
def run_edge_single_inference_roberta_distilled():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = roberta_distilled_single_inference(request.json)
            return json.dumps(result)

@app.route('/run_edge_single_inference_roberta_distilled_int8_sptq', methods=['GET', 'POST'])
def run_edge_single_inference_roberta_distilled_int8_sptq():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            result = roberta_distilled_int8_sptq_single_inference(request.json)
            return json.dumps(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
