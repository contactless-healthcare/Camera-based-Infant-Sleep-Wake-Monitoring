####################################################################################################
#  The best model (Inception_v3) and the best lightweight model (mobilenetv2) will be published    #
####################################################################################################

import os
from torch import nn, optim
import torch
import timm
from torchvision import models
from torch.nn import functional as F

import torchvision.models as torchvision_models
from timm.models.layers import SelectAdaptivePool2d


class SupCon_Inception_v3(nn.Module):
    def __init__(self):
        super(SupCon_Inception_v3, self).__init__()

        # # ResNet18， 输出嵌入层修改为128维，并冻结前面层的参数
        self.inception_v3 = timm.create_model('inception_v3', pretrained=True)
        for param in self.inception_v3.parameters():
            param.requires_grad = True

        # 任务输出
        self.task_output = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(2048, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 2),
        )

        # 对比嵌入层
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

        # 任务输出
        task_output = self.task_output(embedding)

        # 对比嵌输出
        supCon_output = self.supCon_Output(embedding)
        supCon_output = F.normalize(supCon_output)

        return {"output": task_output, "embedding": embedding, "supCon_output": supCon_output}

        # return task_output


class SupCon_mobilenetv2_100(nn.Module):
    def __init__(self):
        super(SupCon_mobilenetv2_100, self).__init__()

        # # ResNet18， 输出嵌入层修改为128维，并冻结前面层的参数
        self.mobilenetv2_100 = timm.create_model('mobilenetv2_100', pretrained=True)
        for param in self.mobilenetv2_100.parameters():
            param.requires_grad = True

        # 任务输出
        self.task_output = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(1280, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 2),
        )

        # 对比嵌入层
        self.supCon_Output = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(1280, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(32, 32)
        )

    def forward(self, x):
        embedding = self.mobilenetv2_100.forward_features(x)
        embedding = self.mobilenetv2_100.global_pool(embedding)
        embedding = torch.flatten(embedding, start_dim=1)

        # 任务输出
        task_output = self.task_output(embedding)

        # 对比嵌输出
        supCon_output = self.supCon_Output(embedding)
        supCon_output = F.normalize(supCon_output)

        return {"output": task_output, "embedding": embedding, "supCon_output": supCon_output}





