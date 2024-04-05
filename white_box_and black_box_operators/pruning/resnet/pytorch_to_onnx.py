import torch
import torchvision.models as models
checkpoint_path = "model_best.pth.tar"
checkpoint = torch.load(checkpoint_path)
model = models.wide_resnet101_2(weights=None)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
torch.onnx.export(
        model,  # model being run
        torch.randn(1, 3, 224, 224, requires_grad=True),  # model input (or a tuple for multiple inputs)
        'wide-resnet101_pruned.onnx',  # where to save the model (can be a file or file-like object)
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

