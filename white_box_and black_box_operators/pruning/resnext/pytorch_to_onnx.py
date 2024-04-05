import torch
import torchvision.models as models
checkpoint_path = "model_best.pth.tar"
checkpoint = torch.load(checkpoint_path)
model = models.resnext101_64x4d(weights=None)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
torch.onnx.export(model,
                 torch.randn(1, 3, 224, 224,requires_grad=True), 
                 'resnext101_pruned.onnx',
                  export_params=True,
                  opset_version=17,
                  do_constant_folding=True,
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={'input': {0: 'batch_size'},'output': {0: 'batch_size'}}
)
