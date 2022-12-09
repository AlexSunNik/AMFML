
import torch.nn as nn
import torch
# from dataset.config import *

class afrm_BN(nn.Module):
    def __init__(self, chn_num):
        super(afrm_BN, self).__init__()
        self.task_id = 0
        self.num_tasks = 2
        self.bn_modules = nn.ModuleList([nn.BatchNorm2d(chn_num).cuda() for _ in range(self.num_tasks)])
        # The below line receives no gradient update when backward
        # self.bn_modules = [nn.BatchNorm2d(chn_num).cuda() for _ in range(NUM_CLASS)]
        for m in self.bn_modules:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def set_task_id(self, id):
        self.task_id = id

    def forward(self, x, id=None):
        if id is None:
            return self.bn_modules[self.task_id](x)
        else:
            return self.bn_modules[id](x)
    
    def return_loss(self):
        weight_loss = (self.bn_modules[0].weight - self.bn_modules[1].weight) ** 2
        bias_loss = (self.bn_modules[0].bias - self.bn_modules[1].bias) ** 2
        return torch.sum(weight_loss), torch.sum(bias_loss)