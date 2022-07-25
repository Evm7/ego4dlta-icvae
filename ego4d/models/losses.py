#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional
import numpy as np

def kl_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class FocalLoss(nn.modules.loss._Loss):
    """Focal Loss with binary cross-entropy
    Implement the focal loss with class-balanced loss, using binary cross-entropy as criterion
    Following paper "Class-Balanced Loss Based on Effective Number of Samples" (CVPR2019)
    """
    def __init__(self,
            beta = 0.9, #0.9, 0.99, 0.999, 0.9999
            samples_per_cls= None,
            no_of_classes = 136,
            size_average=None,
            reduce=None,
            reduction: str = "mean",
            device = None
    ):
        super(FocalLoss, self).__init__(size_average, reduce, reduction)
        self.beta = beta
        if samples_per_cls is None:
            self.loss_base = nn.CrossEntropyLoss(reduction="mean")
        else:
            effective_num = 1.0 - np.power(beta, samples_per_cls)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * no_of_classes
            weights = torch.tensor(weights).float().to(device)
            self.loss_base = nn.CrossEntropyLoss(reduction="mean", weight=weights)


    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return self.loss_base(input, target)

_LOSSES = {
    "cross_entropy_balanced": FocalLoss,
    "cross_entropy": nn.CrossEntropyLoss(reduction="mean"),
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
    'smooth_l1': nn.SmoothL1Loss,
    "l2": nn.MSELoss(reduction="mean"),
    "kl": kl_loss,
}





def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
