import os
from neural_compressor.utils.pytorch import load
import torchvision.models.resnet as models
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
model=models.resnext101_64x4d(weights=None)
valdir = os.path.join('../../../ILSVRC2012', 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_dataset = datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),transforms.ToTensor(),normalize]))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
new_model = load(os.path.abspath(os.path.expanduser('saved_results')), model, dataloader=val_loader)
new_model.eval()
batch_size = 1
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
torch.onnx.export(
        new_model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        'resnext101_int8_qat.onnx',  # where to save the model (can be a file or file-like object)
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
