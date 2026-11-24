"""Export the official ImageNet ResNet-18 backbone. Python is not used at C++ inference time."""
from pathlib import Path
import torch
from torchvision.models import resnet18, ResNet18_Weights
import onnx

root = Path(__file__).resolve().parents[2]
torch.hub.set_dir(str(root / 'models' / '.cache'))
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Identity()
model.eval()
output = root / 'models' / 'resnet18_embedding.onnx'
output.parent.mkdir(exist_ok=True)
with torch.inference_mode():
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(output),
                      input_names=['image'], output_names=['embedding'],
                      opset_version=17, dynamo=False)
onnx.checker.check_model(onnx.load(output))
print(f'Exported {output}: [1,3,224,224] -> [1,512] (fc removed, avgpool retained)')
