import onnx
input_path = "wide-resnet101_int8_sptq.onnx"
output_path = "wide-resnet101_int8_sptq_split_part1.onnx"
input_names = ['input']
output_names = ['/layer3/layer3.16/Add_output_0']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)

output_path = "wide-resnet101_int8_sptq_split_part1.onnx"
input_names=['/layer3/layer3.16/Add_output_0']
output_names = ['output']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)