import torch
import torchvision.models.resnet as models
checkpoint_path = "best_model.pt"
checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
model = models.resnext50_32x4d(weights=None)
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

model_state_dict = remove_module_prefix(checkpoint)
model.load_state_dict(model_state_dict)
model.eval()
batch_size = 1
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        'resnext50_distilled.onnx',  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=17,  # the ONNX version to export the model to, please ensure at least 11.
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {
                0: 'batch_size'
            },  # variable length axes
            'output': {
                0: 'batch_size'
            }
        })
