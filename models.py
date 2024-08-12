import torch
import torch.nn as nn

import config


# resnet-50
class ResNet50(nn.Module):
    def __init__(self, out_dim=10):
        super(ResNet50, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=config.PRETRAIN)
        self.model.fc = nn.Linear(2048, out_dim)

    def forward(self, x):
        return self.model(x)

model_class = {
    'resnet-50': ResNet50,
}