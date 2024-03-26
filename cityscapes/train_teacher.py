"""
DUDES: Deep Uncertainty Distillation using Ensembles for Segmentation
Authors: Steven Landgraf, Kira Wursthorn, Markus Hillemann, Markus Ulrich
"""

import multiprocessing as mp
from multiprocessing import Process

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader

from dataset import SupervisedCityscapes
from utils import PolyLR


class Model(pl.LightningModule):
    def __init__(self, learning_rate, random_seed):
        super().__init__()

        # Initialize DeepLabV3+ with ResNet18 backbone with random weights
        pl.seed_everything(random_seed, workers=True)
        self.model = smp.DeepLabV3Plus(
            encoder_name='resnet18',
            encoder_weights=None,
            in_channels=3,
            classes=19,
        )

        # Training Setup
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        self.optimizer = torch.optim.SGD(params=[
            {'params': self.model.encoder.parameters(), 'lr': self.learning_rate},
            {'params': self.model.decoder.parameters(), 'lr': 10 * self.learning_rate},
            {'params': self.model.segmentation_head.parameters(), 'lr': 10 * self.learning_rate},
        ], lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)

        # Metrics
        self.train_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=19, ignore_index=255)
        self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=19, ignore_index=255)

    def training_step(self, batch, batch_index):
        images, labels = batch
        prediction = self.model(images)
        loss = self.criterion(prediction, labels.squeeze())
        return {'loss': loss, 'prediction': prediction, 'labels': labels}

    def validation_step(self, batch, batch_index):
        images, labels = batch
        prediction = self.model(images)
        loss = self.criterion(prediction, labels.squeeze())
        return {'loss': loss, 'prediction': prediction, 'labels': labels}

    def training_step_end(self, outputs):
        segmentation_map = torch.argmax(torch.softmax(outputs['prediction'], 1), 1)
        train_iou = self.train_iou(segmentation_map, outputs['labels'].squeeze())
        self.log('train_loss', outputs['loss'], on_step=False, on_epoch=True, prog_bar=True)    
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        for param_group in self.optimizer.param_groups:
            self.log('learning_rate', param_group['lr']) 

    def validation_step_end(self, outputs):
        segmentation_map = torch.argmax(torch.softmax(outputs['prediction'], 1), 1)
        val_iou = self.val_iou(segmentation_map, outputs['labels'].squeeze())
        self.log('val_loss', outputs['loss'], on_step=False, on_epoch=True, prog_bar=True)    
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        total_iterations = len(self.train_dataloader()) * self.trainer.max_epochs
        lr_scheduler = PolyLR(self.optimizer, total_iterations, power=0.9)
        return [self.optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        
    def train_dataloader(self):
        train_dataset = SupervisedCityscapes(root_dir='./data/cityscapes', split='train')
        train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=24, persistent_workers=True, pin_memory=True, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = SupervisedCityscapes(root_dir='./data/cityscapes', split='val')
        val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=24, persistent_workers=True, pin_memory=True, shuffle=False)
        return val_dataloader


def train_supervised(seed, device):
    wandb_logger = WandbLogger(
        entity = 'dudes',
        project = 'Teacher Models with Random Init',
        log_model = True,
    )

    model = Model(
        learning_rate = 0.1,
        random_seed = seed,
    )

    trainer = pl.Trainer(
        max_epochs = 200,
        accelerator = 'gpu',
        devices = [device],
        enable_checkpointing = True,
        check_val_every_n_epoch = 1,
        logger = wandb_logger,
        callbacks = [ModelCheckpoint(monitor='val_iou', mode='max', dirpath='./wandb/checkpoints')],  # Early Stopping
        precision = 16,
    )

    trainer.fit(model)
    wandb.finish()


if __name__ == '__main__':
    devices = [1, 2, 3]
    
    # train on a single GPU
    train_supervised(42, devices[0])

    # Train on 3 GPUs in parallel
    # p1 = Process(target=train_supervised, args=(100, devices[0]))
    # p1.start()
    # p2 = Process(target=train_supervised, args=(101, devices[1]))
    # p2.start()
    # p3 = Process(target=train_supervised, args=(102, devices[2]))
    # p3.start()

    # p1.join()
    # p2.join()
    # p3.join()