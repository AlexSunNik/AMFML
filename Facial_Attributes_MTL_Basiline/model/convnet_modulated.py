import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modulate_bn import *

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=0),
            afrm_BN(3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc1 = nn.Linear(36300, 32)  
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x, task_id):
        for m in self.modules():
            if isinstance(m, afrm_BN):
                m.set_task_id(task_id)
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out[:, task_id]