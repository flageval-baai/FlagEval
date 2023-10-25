import timm
import torch.nn as nn

class Resnet50(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Resnet50, self).__init__()
        self.model = timm.create_model(
            'resnet50',
        )
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)
 
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        return [x1, x2, x3, x4]

def get_model():
    resnet50 = Resnet50()
    return resnet50 
