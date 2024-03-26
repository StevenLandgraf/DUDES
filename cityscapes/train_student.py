"""
DUDES: Deep Uncertainty Distillation using Ensembles for Segmentation
Authors: Steven Landgraf, Kira Wursthorn, Markus Hillemann, Markus Ulrich
"""

import os
import logging
from statistics import mean

import torch
import wandb
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from __init__ import DEVICE
from model import DUDESLabV3Plus
from utils import PolyLR, ClassUncertainty, RMSLELoss
from dataset import SupervisedCityscapes, decode_segmentation_map


class DUDES(pl.LightningModule):
    def __init__(self, ensemble_models_path, num_models, learning_rate):
        super().__init__()

        self.ensemble = self._load_ensemble(ensemble_models_path, num_models)
        self.model = DUDESLabV3Plus(uncertainty_classes=1)

        self.learning_rate = learning_rate
        self.segmentation_criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        self.uncertainty_criterion = RMSLELoss()
        self.optimizer = torch.optim.SGD(params=[
            {'params': self.model.encoder.parameters(), 'lr': self.learning_rate},
            {'params': self.model.decoder.parameters(), 'lr': 10 * self.learning_rate},
            # {'params': self.model.uncertainty_decoder.parameters(), 'lr': 10 * self.learning_rate},
            {'params': self.model.segmentation_head.parameters(), 'lr': 10 * self.learning_rate},
            {'params': self.model.uncertainty_head.parameters(), 'lr': 10 * self.learning_rate},            
        ], lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)

        self.train_iou = torchmetrics.JaccardIndex('multiclass', num_classes=19, ignore_index=255)
        self.val_iou = torchmetrics.JaccardIndex('multiclass', num_classes=19, ignore_index=255)
        self.student_class_uncertainties = ClassUncertainty(num_classes=19)
        self.teacher_class_uncertainties = ClassUncertainty(num_classes=19)


    def _load_ensemble(self, ensemble_models_path, num_models):
        """
        It loads the models from the ensemble_models_path directory and returns a list of the models
        """
        ensemble_list = []
        for path in os.listdir(ensemble_models_path):
            if path.endswith('.ckpt'):
                ensemble_model = smp.DeepLabV3Plus(
                    encoder_name='resnet18',
                    encoder_weights=None,
                    in_channels=3,
                    classes=19,
                )
                checkpoint = torch.load(ensemble_models_path + path, map_location='cpu')
                
                # removes 'model' prefix in state dict key names
                new_state_dict = {}
                old_state_dict = checkpoint['state_dict'].copy()
                for key in checkpoint['state_dict'].keys():
                    new_state_dict[key[6:]] = old_state_dict.pop(key)

                ensemble_model.load_state_dict(new_state_dict)
                ensemble_list.append(ensemble_model)

            if len(ensemble_list) == num_models:
                logging.info(f'Loaded {len(ensemble_list)} models into the ensemble')
                return ensemble_list
        
        logging.info(f'Loaded {len(ensemble_list)} models into the ensemble')
        return ensemble_list


    def training_step(self, batch, batch_index):
        images, labels = batch

        # Forward pass teacher (ensemble)
        with torch.no_grad():
            ensemble_outputs = torch.empty(size=[len(self.ensemble), images.shape[0], 19, images.shape[2], images.shape[3]], device=self.device)
            for i, model in enumerate(self.ensemble):
                model.cuda()
                model.eval()
                ensemble_outputs[i] = model(images)

        probability_label = torch.mean(torch.softmax(ensemble_outputs, dim=2), dim=0)
        ensemble_uncertainty_map = torch.std(torch.softmax(ensemble_outputs, dim=2), dim=0)
        uncertainty_label = torch.empty(size=[images.shape[0], images.shape[2], images.shape[3]], device=DEVICE)
        for i in range(19):
            uncertainty_label = torch.where(torch.argmax(probability_label, dim=1) == i, ensemble_uncertainty_map[:, i, :, :], uncertainty_label)

        # Forward pass student
        segmentation_prediction, uncertainty_prediction = self.model(images)

        # Loss and metrics
        segmentation_loss = self.segmentation_criterion(segmentation_prediction, labels.squeeze())
        uncertainty_loss = self.uncertainty_criterion(uncertainty_prediction, uncertainty_label.unsqueeze(1))
        loss = segmentation_loss + uncertainty_loss
        
        self.train_iou(torch.argmax(torch.softmax(segmentation_prediction, dim=1), dim=1), labels.squeeze())

        # Logging
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_seg_loss', segmentation_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_unc_loss', uncertainty_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_index):
        images, labels = batch

        # Forward pass teacher (ensemble)
        with torch.no_grad():
            ensemble_outputs = torch.empty(size=[len(self.ensemble), images.shape[0], 19, images.shape[2], images.shape[3]], device=self.device)
            for i, model in enumerate(self.ensemble):
                model.cuda()
                model.eval()
                ensemble_outputs[i] = model(images)

        probability_label = torch.mean(torch.softmax(ensemble_outputs, dim=2), dim=0)
        ensemble_uncertainty_map = torch.std(torch.softmax(ensemble_outputs, dim=2), dim=0)
        uncertainty_label = torch.empty(size=[images.shape[0], 1024, 2048], device=DEVICE)
        for i in range(19):
            uncertainty_label = torch.where(torch.argmax(probability_label, dim=1) == i, ensemble_uncertainty_map[:, i, :, :], uncertainty_label)

        # Forward pass student
        segmentation_prediction, uncertainty_prediction = self.model(images)

        segmentation_loss = self.segmentation_criterion(segmentation_prediction, labels.squeeze())
        uncertainty_loss = self.uncertainty_criterion(uncertainty_prediction, uncertainty_label.unsqueeze(1))
        loss = segmentation_loss + uncertainty_loss

        self.val_iou(torch.argmax(torch.softmax(segmentation_prediction, dim=1), dim=1), labels.squeeze())
        self.student_class_uncertainties.update(torch.softmax(segmentation_prediction, dim=1), uncertainty_prediction.repeat(1, 19, 1, 1))
        self.teacher_class_uncertainties.update(probability_label, ensemble_uncertainty_map)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_seg_loss', segmentation_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_unc_loss', uncertainty_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True)


    def validation_epoch_end(self, outputs):
        student_class_uncertainties = self.student_class_uncertainties.compute()
        teacher_class_uncertainties = self.teacher_class_uncertainties.compute()

        self.student_class_uncertainties.reset()
        self.teacher_class_uncertainties.reset()

        self.log('student_unc', torch.mean(student_class_uncertainties), on_step=False, on_epoch=True, prog_bar=False)
        self.log('teacher_unc', torch.mean(teacher_class_uncertainties), on_step=False, on_epoch=True, prog_bar=False)

        class_difference = torch.sqrt(torch.pow((student_class_uncertainties - teacher_class_uncertainties), 2))
        self.log('class_unc_error', torch.mean(class_difference), on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f'class_unc_error')

        for i in range(19):
            self.log(f'class_unc_difference{(i+1):02d}', class_difference[i], on_step=False, on_epoch=True, prog_bar=False, metric_attribute=f'class_unc_difference_{(i+1):02d}')


    def configure_optimizers(self):
        iterations_per_epoch = len(self.train_dataloader())
        total_iterations = iterations_per_epoch * self.trainer.max_epochs
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    
    def train_dataloader(self):
        train_dataset = SupervisedCityscapes(root_dir='./data/cityscapes', split='train')
        return DataLoader(train_dataset, batch_size=16, num_workers=24, persistent_workers=True, pin_memory=True, shuffle=True)

    
    def val_dataloader(self):
        val_dataset = SupervisedCityscapes(root_dir='./data/cityscapes', split='val')
        return DataLoader(val_dataset, batch_size=4, num_workers=24, persistent_workers=True, pin_memory=True, shuffle=False)


if __name__ == '__main__':
    pl.seed_everything(42, workers=True)

    wandb_logger = WandbLogger(
        entity = 'dudes',
        project = 'Student Model with ImageNet Init',
        log_model = True,
    )

    model = DUDES(
        ensemble_models_path='./ensemble_models/resnet18_random/',
        num_models=10,
        learning_rate=0.01,
    )
    
    trainer = pl.Trainer(
        max_epochs = 200,
        accelerator = 'gpu',
        devices=[DEVICE],
        check_val_every_n_epoch = 1,
        num_sanity_val_steps = 0,
        logger = wandb_logger,
        enable_checkpointing=True,
        callbacks = [ModelCheckpoint(monitor='val_iou', mode='max', save_top_k=5, dirpath='./wandb/checkpoints')],
        precision = 16,
    )

    trainer.fit(model)
    wandb.finish()