"""
Project: DUDES
Authors: Steven Landgraf, Kira Wursthorn, Markus Hillemann, Markus Ulrich
"""

import math
from typing import Any

import torch
import torchmetrics
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from segmentation_models_pytorch.base.heads import SegmentationHead
from segmentation_models_pytorch.decoders.unet.model import Unet

import config
from utils import RMSLELoss, ClassUncertainty, PolyLR, Mask_RMSLELoss
from model import MCDropoutViT
from dataset import PascalVOCDataModule


class DUDES_Net(Unet):
    def __init__(self):
        super().__init__(
            encoder_name='mit_b5',
            encoder_weights='imagenet',
            classes=config.NUM_CLASSES,
        )

        self.uncertainty_head = SegmentationHead(
            in_channels=16,
            out_channels=1,     # only one output channel for the predictive uncertainty
            activation='sigmoid',
            kernel_size=1,      # 1x1 convolution kernel size for the uncertainty head (default: 3x3)
            upsampling=1,
        )


    def forward(self, x):
        super().check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        segmentation_prediction = self.segmentation_head(decoder_output)
        uncertainty_prediction = self.uncertainty_head(decoder_output)

        return segmentation_prediction, uncertainty_prediction


class DUDES(L.LightningModule):
    def __init__(self, checkpoint_path):
        super().__init__()

        # Initialize the teacher model
        self.teacher = self._load_mcdropout_model(checkpoint_path)

        # Initialize the student model
        self.model = DUDES_Net()

        # Initialize the loss function
        self.segmentation_criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        self.uncertainty_criterion = RMSLELoss()

        # Initialize the optimizer
        self.optimizer = torch.optim.SGD(params=[
            {'params': self.model.encoder.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decoder.parameters(), 'lr': 10 * config.LEARNING_RATE},
            {'params': self.model.uncertainty_head.parameters(), 'lr': 10 * config.LEARNING_RATE},
            {'params': self.model.segmentation_head.parameters(), 'lr': 10 * config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

        # Initialize the metrics
        self.train_iou = torchmetrics.JaccardIndex('multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.val_iou = torchmetrics.JaccardIndex('multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.student_class_uncertainties = ClassUncertainty(num_classes=config.NUM_CLASSES)
        self.teacher_class_uncertainties = ClassUncertainty(num_classes=config.NUM_CLASSES)


    def _load_mcdropout_model(self, checkpoint_path):
        model = MCDropoutViT()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        new_state_dict = {}
        old_state_dict = checkpoint['state_dict'].copy()
        for key in checkpoint['state_dict'].keys():
            new_state_dict[key[6:]] = old_state_dict.pop(key)
        
        model.load_state_dict(new_state_dict)
        print('Loaded MCDropout model')

        return model
        

    def training_step(self, batch, batch_index):
        images, labels = batch

        # Forward pass teacher (MCD model)
        with torch.no_grad():
            self.teacher.cuda()
            self.teacher.eval()
            for layer in self.teacher.modules():
                if isinstance(layer, torch.nn.Dropout):
                    layer.train()

            teacher_outputs = torch.empty(size=[10, images.shape[0], config.NUM_CLASSES, images.shape[2], images.shape[3]], device=self.device)
            for i in range(10):
                teacher_outputs[i] = self.teacher(images)

        uncertainty_map = torch.std(torch.softmax(teacher_outputs, dim=2), dim=0)
        teacher_uncertainty = torch.empty(size=[images.shape[0], images.shape[2], images.shape[3]], device=self.device)
        for i in range(config.NUM_CLASSES):
            teacher_uncertainty = torch.where(torch.argmax(torch.mean(torch.softmax(teacher_outputs, dim=2), dim=0), dim=1) == i, uncertainty_map[:, i], teacher_uncertainty)
        
        # Forward pass student
        logits, student_uncertainty = self.model(images)

        # Calculate loss
        segmentation_loss = self.segmentation_criterion(logits, labels.squeeze())
        uncertainty_loss = self.uncertainty_criterion(student_uncertainty, teacher_uncertainty.unsqueeze(1))
        loss = segmentation_loss + uncertainty_loss

        # Calculate metrics
        self.train_iou(torch.argmax(torch.softmax(logits, dim=1), dim=1), labels.squeeze())

        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_seg_loss', segmentation_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_unc_loss', uncertainty_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch, batch_index):
        images, labels = batch

        logits, student_uncertainty = self.model(images)

        self.val_iou(torch.argmax(torch.softmax(logits, dim=1), dim=1), labels.squeeze())
        self.student_class_uncertainties.update(torch.softmax(logits, dim=1), student_uncertainty.repeat(1, config.NUM_CLASSES, 1, 1))
        
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def on_validation_epoch_end(self):
        student_class_uncertainties = self.student_class_uncertainties.compute()

        self.student_class_uncertainties.reset()

        self.log('student_mUnc', torch.mean(student_class_uncertainties), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


    def configure_optimizers(self):
        iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * len(config.DEVICES)))
        total_iterations = iterations_per_epoch * self.trainer.max_epochs
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=0.9)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]   


if __name__ == '__main__':
    L.seed_everything(42, workers=True)

    model = DUDES('./mcd_models/pascal_200_50%.ckpt')

    data_module = PascalVOCDataModule(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        distill=True,
    )

    trainer = L.Trainer(
        max_epochs=config.NUM_EPOCHS,
        accelerator='gpu',
        devices=config.DEVICES,
        precision=config.PRECISION,
        log_every_n_steps = 47,
        check_val_every_n_epoch=1, 
        logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=True),
        callbacks=[
            # ModelCheckpoint(dirpath=f'./checkpoints/{config.RUN_NAME}', filename='model-{epoch}-{val_iou:.3f}', monitor='val_iou', mode='max', save_top_k=3),
            ModelCheckpoint(dirpath=f'./checkpoints/{config.RUN_NAME}', filename='model-{epoch}-{val_iou:.3f}', monitor='student_mUnc', mode='max', save_top_k=20),
            LearningRateMonitor(logging_interval='epoch'),
        ],
    )

    trainer.fit(model, data_module)
