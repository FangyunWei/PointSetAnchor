import torch
import torch.nn as nn
from ..registry import LOSSES


@LOSSES.register_module
class L1Loss(nn.L1Loss):

    def __init__(self, loss_weight=1.0):
        super(L1Loss, self).__init__(None, None, 'none')
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, avg_factor=1.0):
        res = super(L1Loss, self).forward(pred, target)
        loss = self.loss_weight * torch.sum(weight * res)[None] / avg_factor
        return loss
