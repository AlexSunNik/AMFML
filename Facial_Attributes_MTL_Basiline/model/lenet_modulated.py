import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modulate_bn import *

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            afrm_BN(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            afrm_BN(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = 2, stride = 2))
        # self.fc = nn.Linear(18432, 120)  
        self.fc = nn.Linear(44944, 120)    
        # self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x, task_id):
        for m in self.modules():
            if isinstance(m, afrm_BN):
                m.set_task_id(task_id)
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out[:, task_id]