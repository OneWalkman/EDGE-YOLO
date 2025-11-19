# 文件：ultralytics/nn/modules/getitem.py
import torch.nn as nn

class GetItem(nn.Module):
    def __init__(self, index=0):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]
