"""
Builds upon: https://github.com/qinenergy/cotta
Corresponding paper: https://arxiv.org/abs/2203.13591

CoTTA:
    - mean-teacher model
    - weight-and-augmentation-averaged pseudo-labels
    - explicit information preservation from source model
"""

import torch
import torch.jit
from torch import nn, optim
from copy import deepcopy
from torchvision.transforms import v2 as tt

from .util import consistencyLoss

class CoTTA():
    name = "CoTTA"
    def __init__(self, model, hparams, device):
        super(CoTTA).__init__()
        self.student = model.to(device)
        self.source = deepcopy(model).to(device)
        self.teacher = deepcopy(model).to(device)
        for param in self.source.parameters():
            param.detach_()
        for param in self.teacher.parameters():
            param.detach_()
        self.optimizer = optim.SGD(self.student.parameters(), lr=hparams.lr, momentum=0.9)
        self.alpha = hparams.adapt_alpha
        self.transforms = tt.Compose([
            tt.ColorJitter(),
            tt.Pad(112),
            tt.RandomAffine([-8,8]),
            tt.GaussianBlur(5),
            tt.CenterCrop(224),
            tt.RandomHorizontalFlip(),
            tt.GaussianNoise(),
        ])
        self.n_augs = hparams.adapt_n_augs
        self.threshold = hparams.adapt_restore_threshold

    @torch.enable_grad()
    def __call__(self, x):
        logits = self.student(x)
        logits_teacher = torch.stack([self.teacher(self.transforms(x)) for _ in range(self.n_augs)]).mean(0)

        # adapt student model
        loss = consistencyLoss(logits, logits_teacher).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # adapt teacher model
        for tparam, sparam in zip(self.teacher.parameters(), self.student.parameters()):
            tparam = self.alpha * tparam + (1 - self.alpha) * sparam

        # stochastic restore
        with torch.no_grad():
            for sparam, tparam in zip(self.source.parameters(), self.student.parameters()):
                tparam = torch.where(torch.rand(*sparam.shape).to("cuda") < self.threshold, sparam, tparam)

        return logits_teacher
