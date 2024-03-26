"""
Project: DUDES
Authors: Steven Landgraf, Kira Wursthorn, Markus Hillemann, Markus Ulrich
"""

from statistics import mean, stdev

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchmetrics import Metric
from torch.optim.lr_scheduler import _LRScheduler


class ClassUncertainty(Metric):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.uncertainty_list = []

        self.add_state('class_uncertainties', default=torch.zeros(size=[self.num_classes]), dist_reduce_fx='mean')

    
    def update(self, probability_map, uncertainty_map):
        one_hot_prediction_map = F.one_hot(torch.argmax(probability_map, dim=1), num_classes=self.num_classes)    # [batch_size, 1024, 2048, self.num_classes]
        one_hot_prediction_map = torch.permute(one_hot_prediction_map, (0, 3, 1, 2))                # [batch_size, self.num_classes, 1024, 2048]

        one_hot_uncertainty_map = torch.where(one_hot_prediction_map == 1., uncertainty_map, torch.zeros_like(uncertainty_map))   # [batch_size, self.num_classes, 1024, 2048]
        one_hot_uncertainty_map[one_hot_uncertainty_map == 0] = float('nan')

        class_uncertainties = torch.empty(size=[self.num_classes]).cuda()
        for i in range(self.num_classes):
            one_hot_class_uncertainty = one_hot_uncertainty_map[:, i, :, :]     # [batch_size, 1024, 2048]
            class_uncertainties[i] = torch.mean(one_hot_class_uncertainty[~one_hot_class_uncertainty.isnan()])

        self.uncertainty_list.append(class_uncertainties.detach().cpu().numpy())


    def compute(self):
        self.class_uncertainties = torch.nan_to_num(torch.from_numpy(np.nanmean(np.asarray(self.uncertainty_list), axis=0))).cuda()
        return self.class_uncertainties


    def reset(self):
        self.uncertainty_list = []


class PolyLR(_LRScheduler):
    """LR = Initial_LR * (1 - iter / max_iter)^0.9"""
    def __init__(self, optimizer, max_iterations, power=0.9):
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.power = power
        super().__init__(optimizer)

    def get_lr(self):
        self.current_iteration += 1
        return [base_lr * (1 - self.current_iteration / self.max_iterations) ** self.power for base_lr in self.base_lrs]


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
    

class Mask_RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, pred, actual, mask):
        loss = self.mse(torch.log(pred + 1), torch.log(actual + 1))
        loss[mask == 1] = 0
        loss = torch.sum(loss) / (torch.numel(loss) - torch.sum(mask))
        return torch.sqrt(loss)


if __name__ == '__main__':
    torch.manual_seed(42)

    loss = Mask_RMSLELoss()
    pred = torch.rand(16, 1, 256, 256)
    actual = torch.rand(16, 1, 256, 256)
    mask = torch.rand(16, 1, 256, 256)
    mask[mask > 0.5] = 1
    print(loss(pred, actual, mask))

    loss = RMSLELoss()
    print(loss(pred, actual))
