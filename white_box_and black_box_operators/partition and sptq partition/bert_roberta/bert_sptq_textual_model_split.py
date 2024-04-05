import onnx

input_path="bert-base-uncased-mrpc-int8-sptq.onnx"
output_path="bert-base-uncased-mrpc-int8-sptq-part1.onnx"
input_names=['input_ids','attention_mask','token_type_ids']
output_names=['/bert/embeddings/Add_1_output_0','attention_mask_int32']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)

output_path="bert-base-uncased-mrpc-int8-sptq-part2.onnx"
input_names=['/bert/embeddings/Add_1_output_0','attention_mask_int32']
output_names=['logits']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)


