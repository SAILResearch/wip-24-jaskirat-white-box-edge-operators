import torch
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime as ort   # to inference ONNX models, we use the ONNX Runtime
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Intel/bert-base-uncased-mrpc')

class Bert():
    def __init__(self):
        pass

    def preprocess(self, text):
        inputs = tokenizer(text[0], text[1], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return self.to_numpy(inputs['input_ids']), self.to_numpy(inputs['attention_mask']),self.to_numpy(inputs['token_type_ids']) 

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def postprocess(self, result):
        res = np.argmax(result[0])
        if (res == 0):
            result = {"Prediction: ": "Not Equivalent"}
        elif (res == 1):
            result = {"Prediction: ": "Equivalent"}
        return result

bert_model = Bert()

def bert_single_inference(text):
    input_ids, attention_mask, token_type_ids = bert_model.preprocess(text)
    session = ort.InferenceSession(
        'bert_models/bert-base-uncased-mrpc.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run(
       [i.name for i in session.get_outputs()], {session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask,session.get_inputs()[2].name:token_type_ids})
    result = bert_model.postprocess(raw_result)
    return result

def bert_pruned_single_inference(text):
    input_ids, attention_mask, token_type_ids = bert_model.preprocess(text)
    session = ort.InferenceSession(
        'bert_models/bert-base-uncased-mrpc-pruned.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run(
       [i.name for i in session.get_outputs()], {session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask,session.get_inputs()[2].name:token_type_ids})
    result = bert_model.postprocess(raw_result)
    return result

def bert_distilled_single_inference(text):
    input_ids, attention_mask, token_type_ids = bert_model.preprocess(text)
    session = ort.InferenceSession(
        'bert_models/bert-small-distilled-mrpc.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run(
       [i.name for i in session.get_outputs()], {session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask,session.get_inputs()[2].name:token_type_ids})
    result = bert_model.postprocess(raw_result)
    return result

def bert_distilled_int8_sptq_single_inference(text):
    input_ids, attention_mask, token_type_ids = bert_model.preprocess(text)
    session = ort.InferenceSession(
        'bert_models/bert-small-distilled-mrpc-int8-sptq.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run(
       [i.name for i in session.get_outputs()], {session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask,session.get_inputs()[2].name:token_type_ids})
    result = bert_model.postprocess(raw_result)
    return result

def bert_int8_sptq_single_inference(text):
    input_ids, attention_mask, token_type_ids = bert_model.preprocess(text)
    session = ort.InferenceSession(
        'bert_models/bert-base-uncased-mrpc-int8-sptq.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run(
       [i.name for i in session.get_outputs()], {session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask,session.get_inputs()[2].name:token_type_ids})
    result = bert_model.postprocess(raw_result)
    return result

def bert_int8_qat_single_inference(text):
    input_ids, attention_mask, token_type_ids = bert_model.preprocess(text)
    session = ort.InferenceSession(
        'bert_models/bert-base-uncased-mrpc-int8-qat.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run(
       [i.name for i in session.get_outputs()], {session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask,session.get_inputs()[2].name:token_type_ids})
    result = bert_model.postprocess(raw_result)
    return result

def bert_split_first_half_single_inference(text):
    input_ids, attention_mask, token_type_ids = bert_model.preprocess(text)
    session = ort.InferenceSession(
        'bert_models/bert-base-uncased-mrpc-part1.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run([i.name for i in session.get_outputs()], {
        session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask,session.get_inputs()[2].name:token_type_ids})
    return raw_result

def bert_split_second_half_single_inference(text):
    session = ort.InferenceSession(
        'bert_models/bert-base-uncased-mrpc-part2.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run([i.name for i in session.get_outputs()], {
        session.get_inputs()[0].name: text[0], session.get_inputs()[1].name: text[1]})
    result = bert_model.postprocess(raw_result)
    return result

def bert_int8_sptq_split_first_half_single_inference(text):
    input_ids, attention_mask, token_type_ids = bert_model.preprocess(text)
    session = ort.InferenceSession(
        'bert_models/bert-base-uncased-mrpc-int8-sptq-part1.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run([i.name for i in session.get_outputs()], {
        session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask,session.get_inputs()[2].name:token_type_ids})
    return raw_result


def bert_int8_sptq_split_second_half_single_inference(text):
    session = ort.InferenceSession(
        'bert_models/bert-base-uncased-mrpc-int8-sptq-part2.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run([i.name for i in session.get_outputs()], {
        session.get_inputs()[0].name: text[0], session.get_inputs()[1].name: text[1]})
    result = bert_model.postprocess(raw_result)
    return result
