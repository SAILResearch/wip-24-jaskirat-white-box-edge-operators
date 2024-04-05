import torch
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime as ort   # to inference ONNX models, we use the ONNX Runtime
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Intel/roberta-base-mrpc')
tokenizer_student=AutoTokenizer.from_pretrained('haisongzhang/roberta-tiny-cased')

class Roberta():
    def __init__(self):
        pass

    def preprocess(self, text):
        inputs = tokenizer(text[0], text[1], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return self.to_numpy(inputs['input_ids']), self.to_numpy(inputs['attention_mask'])

    def preprocess_for_student(self, text):
        inputs = tokenizer_student(text[0], text[1], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
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


roberta_model = Roberta()


def roberta_single_inference(text):
    input_ids, attention_mask = roberta_model.preprocess(text)
    session = ort.InferenceSession(
        'roberta_models/roberta-base-mrpc.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run(
       [i.name for i in session.get_outputs()], {session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask})
    result = roberta_model.postprocess(raw_result)
    return result

def roberta_pruned_single_inference(text):
    input_ids, attention_mask = roberta_model.preprocess(text)
    session = ort.InferenceSession(
        'roberta_models/roberta-base-mrpc-pruned.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run(
       [i.name for i in session.get_outputs()], {session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask})
    result = roberta_model.postprocess(raw_result)
    return result

def roberta_distilled_single_inference(text):
    input_ids, attention_mask, token_type_ids = roberta_model.preprocess_for_student(text)
    session = ort.InferenceSession(
        'roberta_models/roberta-tiny-distilled-mrpc.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run(
       [i.name for i in session.get_outputs()], {session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask,session.get_inputs()[2].name:token_type_ids})
    result = roberta_model.postprocess(raw_result)
    return result

def roberta_distilled_int8_sptq_single_inference(text):
    input_ids, attention_mask, token_type_ids = roberta_model.preprocess_for_student(text)
    session = ort.InferenceSession(
        'roberta_models/roberta-tiny-distilled-mrpc-int8-sptq.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run(
       [i.name for i in session.get_outputs()], {session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask,session.get_inputs()[2].name:token_type_ids})
    result = roberta_model.postprocess(raw_result)
    return result

def roberta_int8_sptq_single_inference(text):
    input_ids, attention_mask = roberta_model.preprocess(text)
    session = ort.InferenceSession(
        'roberta_models/roberta-base-mrpc-int8-sptq.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run(
       [i.name for i in session.get_outputs()], {session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask})
    result = roberta_model.postprocess(raw_result)
    return result

def roberta_int8_qat_single_inference(text):
    input_ids, attention_mask = roberta_model.preprocess(text)
    session = ort.InferenceSession(
        'roberta_models/roberta-base-mrpc-int8-qat.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run(
       [i.name for i in session.get_outputs()], {session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask})
    result = roberta_model.postprocess(raw_result)
    return result

def roberta_split_first_half_single_inference(text):
    input_ids, attention_mask = roberta_model.preprocess(text)
    session = ort.InferenceSession(
        'roberta_models/roberta-base-mrpc-part1.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run([i.name for i in session.get_outputs()], {
        session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask})
    return raw_result

def roberta_split_second_half_single_inference(text):
    session = ort.InferenceSession(
        'roberta_models/roberta-base-mrpc-part2.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run([i.name for i in session.get_outputs()], {
        session.get_inputs()[0].name: text[0], session.get_inputs()[1].name: text[1]})
    result = roberta_model.postprocess(raw_result)
    return result

def roberta_int8_sptq_split_first_half_single_inference(text):
    input_ids, attention_mask = roberta_model.preprocess(text)
    session = ort.InferenceSession(
        'roberta_models/roberta-base-mrpc-int8-sptq-part1.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run([i.name for i in session.get_outputs()], {
        session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name:attention_mask})
    return raw_result


def roberta_int8_sptq_split_second_half_single_inference(text):
    session = ort.InferenceSession(
        'roberta_models/roberta-base-mrpc-int8-sptq-part2.onnx', providers=['CPUExecutionProvider'])
    raw_result = session.run([i.name for i in session.get_outputs()], {
        session.get_inputs()[0].name: text[0], session.get_inputs()[1].name: text[1]})
    result = roberta_model.postprocess(raw_result)
    return result
