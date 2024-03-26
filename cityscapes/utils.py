"""
DUDES: Deep Uncertainty Distillation using Ensembles for Segmentation
Authors: Steven Landgraf, Kira Wursthorn, Markus Hillemann, Markus Ulrich
"""

import time
import multiprocessing as mp
from statistics import mean, stdev

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchmetrics import Metric
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from __init__ import DEVICE
from dataset import SupervisedCityscapes
from model import DUDESLabV3Plus


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


def evaluate_inference_time_single_model(model, repetitions):
    """
    It takes a model and a number of repetitions, and returns the average inference time of the model on
    a single image.    
    """
    # Cityscapes image as dummy input
    dataset = SupervisedCityscapes(root_dir='./data/cityscapes', split='val')
    image, _ = dataset[0]
    image = image.unsqueeze(0).to(device=DEVICE)

    # inference time measurement
    total_time = 0
    with torch.no_grad():
        for _ in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            
            starter.record()
            _ = model(image)
            ender.record()

            torch.cuda.synchronize()

            current_time = starter.elapsed_time(ender)
            total_time += current_time            

    return (total_time / repetitions)


def evaluate_inference_time_ensemble(ensemble: list, repetitions):
    """
    It takes an ensemble of models and a number of repetitions, and returns the average inference time
    of the ensemble.
    """
    # Cityscapes image as dummy input
    dataset = SupervisedCityscapes(root_dir='./data/cityscapes', split='val')
    image, _ = dataset[0]
    image = image.unsqueeze(0).to(device=DEVICE)

    # inference time measurement
    total_time = 0
    with torch.no_grad():
        for repetition in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            
            starter.record()
            ensemble_outputs = torch.empty(size=[len(ensemble), image.shape[0], 19, image.shape[2], image.shape[3]], device=DEVICE) 
            for i in range(len(ensemble)):
                ensemble_outputs[i] = ensemble[i](image)
            _ = torch.mean(torch.softmax(ensemble_outputs, dim=2), dim=0) 
            _ = torch.std(torch.softmax(ensemble_outputs, dim=2), dim=0) 
            ender.record()

            torch.cuda.synchronize()

            current_time = starter.elapsed_time(ender)
            total_time += current_time

    return (total_time / repetitions)


def print_model_parameters(baseline_model, student_model):
    baseline_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    student_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters:')
    print(f'Baseline: {baseline_params}:')
    print(f'Teacher : {10 * baseline_params}:')
    print(f'Student : {student_params}:')


if __name__ == '__main__':
    ## Initialize Models 
    baseline_model = smp.DeepLabV3Plus(
        encoder_name='resnet18',
        encoder_weights=None,
        in_channels=3,
        classes=19,
    ).to(device=DEVICE).eval()
    
    teacher_models = []
    for i in range(10):
        teacher_models.append(smp.DeepLabV3Plus(
            encoder_name='resnet18',
            encoder_weights=None,
            in_channels=3,
            classes=19,
        ).to(device=DEVICE).eval())

    student_model = DUDESLabV3Plus(
        uncertainty_classes=1
    ).to(device=DEVICE).eval()

    ## Count Parameters
    print_model_parameters(baseline_model, student_model)

    ## Inference Time Comparison
    inference_times = []
    for i in range(25):
        inference_times.append(evaluate_inference_time_single_model(student_model, 100))
    
    print(f'Mean Inference Time: {mean(inference_times)}')
    print(f'Std Inference Time: {stdev(inference_times)}')
    
    inference_times = []
    for i in range(25):
        inference_times.append(evaluate_inference_time_ensemble(teacher_models, 100))
    
    print(f'Mean Inference Time: {mean(inference_times)}')
    print(f'Std Inference Time: {stdev(inference_times)}')

    inference_times = []
    for i in range(25):
        inference_times.append(evaluate_inference_time_single_model(baseline_model, 100))
    
    print(f'Mean Inference Time: {mean(inference_times)}')
    print(f'Std Inference Time: {stdev(inference_times)}')
