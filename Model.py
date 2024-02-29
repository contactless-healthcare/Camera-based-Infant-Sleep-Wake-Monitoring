import os
from torch import nn, optim
import torch
import timm
from torchvision import models
from torch.nn import functional as F


class SupCon_Inception_v3(nn.Module):
    def __init__(self):
        super(SupCon_Inception_v3, self).__init__()

        self.inception_v3 = timm.create_model('inception_v3', pretrained=True)
        for param in self.inception_v3.parameters():
            param.requires_grad = True

        self.task_output = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(2048, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 2),
        )

        self.supCon_Output = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(2048, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(32, 32)
        )

    def forward(self, x):
        embedding = self.inception_v3.forward_features(x)
        embedding = self.inception_v3.global_pool(embedding)
        embedding = torch.flatten(embedding, start_dim=1)

        task_output = self.task_output(embedding)


        supCon_output = self.supCon_Output(embedding)
        supCon_output = F.normalize(supCon_output)

        return {"output": task_output, "embedding": embedding, "supCon_output": supCon_output}


