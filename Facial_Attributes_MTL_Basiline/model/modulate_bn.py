
import torch.nn as nn
from dataset.config import *

class afrm_BN(nn.Module):
    def __init__(self, chn_num):
        super(afrm_BN, self).__init__()
        self.task_id = 0
        self.num_tasks = NUM_CLASS
        self.bn_modules = nn.ModuleList([nn.BatchNorm2d(chn_num).cuda() for _ in range(NUM_CLASS)])
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