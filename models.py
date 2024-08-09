import torch
import torch.nn as nn

from client import Client
from threading import Thread
import config

model_class = {
    'resnet-50': ResNet50,
}

# a unified interface for all models
class Model:
    def __init__(self, out_dim, model_name):
        self.backbone = model_class[model_name](out_dim=out_dim)
        self.client = Client(out_dim, self.backbone)

    def forward(self, x):
        return self.backbone(x)

    def communicate(self):
        '''
        To be implemented later...
        :return: None
        '''
        Thread(target=self.client.communicate).start()


# resnet-50
class ResNet50(nn.Module):
    def __init__(self, out_dim=10):
        super(ResNet50, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=config.PRETRAIN)
        self.model.fc = nn.Linear(2048, out_dim)

    def forward(self, x):
        return self.model(x)
