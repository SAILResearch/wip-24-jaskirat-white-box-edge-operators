import onnx
input_path="roberta-base-mrpc.onnx"
output_path="roberta-base-mrpc_part1.onnx"
input_names=['input_ids','attention_mask']
output_names=['/roberta/embeddings/Add_2_output_0','/roberta/Sub_output_0']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)

output_path="roberta-base-mrpc_part2.onnx"
input_names=['/roberta/embeddings/Add_2_output_0','/roberta/Sub_output_0']
output_names=['logits']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)