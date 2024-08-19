import torch
import torch.nn as nn
from torchvision import models

import config


# resnet-50
class ResNet50(nn.Module):
    def __init__(self, out_dim=10):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(2048, out_dim)

    def forward(self, x):
        return self.model(x)

    def get_params(self) -> torch.Tensor:
        return self.model.parameters()


# resnet-18
class ResNet18(nn.Module):
    def __init__(self, out_dim=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.model(x)

    def get_params(self) -> torch.Tensor:
        return self.model.parameters()