import onnx


input_path="bert-base-uncased-mrpc.onnx"
output_path="bert-base-uncased-mrpc_part1.onnx"
input_names=['input_ids','attention_mask','token_type_ids']
output_names=['/bert/embeddings/Add_1_output_0','/bert/Sub_output_0']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)

output_path="bert-base-uncased-mrpc_part2.onnx"
input_names=['/bert/embeddings/Add_1_output_0','/bert/Sub_output_0']
output_names=['logits']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)
