import torch
from torch import nn
from torchvision.transforms.v2 import functional as ttf

def consistencyLoss(x, y):
    return -(y.softmax(1) * x.log_softmax(1)).sum(1)


