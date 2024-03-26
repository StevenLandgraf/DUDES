"""
Project: DUDES
Authors: Steven Landgraf, Kira Wursthorn, Markus Hillemann, Markus Ulrich
"""

import math

import torch
import torchmetrics
import lightning as L
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from functools import partial
from segmentation_models_pytorch.decoders.unet import Unet
from segmentation_models_pytorch.encoders.mix_transformer import MixVisionTransformerEncoder, get_pretrained_cfg

import config
from utils import PolyLR


class MCD_ViT(L.LightningModule):
    def __init__(self):
        super().__init__()

        # Initialize the model
        self.model = MCDropoutViT()

        # Initialize the loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)

        # Initialize the optimizer
        self.optimizer = torch.optim.SGD(params=[
            {'params': self.model.encoder.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decoder.parameters(), 'lr': 10 * config.LEARNING_RATE},
            {'params': self.model.segmentation_head.parameters(), 'lr': 10 * config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

        # Initialize the metrics
        self.train_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)


    def training_step(self, batch, batch_index):
        images, labels = batch

        logits = self.model(images)

        loss = self.criterion(logits, labels.squeeze())

        self.train_iou(torch.softmax(logits, dim=1), labels.squeeze())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_index):
        images, labels = batch

        logits = self.model(images)

        self.val_iou(torch.softmax(logits, dim=1), labels.squeeze())

        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


    def configure_optimizers(self):
        iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * len(config.DEVICES)))
        total_iterations = iterations_per_epoch * self.trainer.max_epochs
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=0.9)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]        


class MCDropoutViT(Unet):
    def __init__(self):
        super().__init__(
            # encoder_name = 'mit_b3',
            encoder_name = 'mit_b5',
            encoder_weights = 'imagenet',
            classes = config.NUM_CLASSES,
        )

        # "mit_b5"
        params = {
            "encoder": MixVisionTransformerEncoder,
            "pretrained_settings": {
                "imagenet": get_pretrained_cfg("mit_b5"),
            },
            "params": dict(
                out_channels=(3, 0, 64, 128, 320, 512),
                patch_size=4,
                embed_dims=[64, 128, 320, 512],
                num_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4],
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                depths=[3, 6, 40, 3],
                sr_ratios=[8, 4, 2, 1],
                drop_rate=config.DROPOUT_RATE,
                drop_path_rate=config.DROPOOUT_PATH_RATE,
            ),
        }

        self.encoder = MixVisionTransformerEncoder(**params["params"])
        self.encoder.load_state_dict(model_zoo.load_url(params["pretrained_settings"]["imagenet"]["url"]))


if __name__ == '__main__':
    model = MCDropoutViT()

    # dummy input
    x = torch.randn(4, 3, 512, 512)

    # forward pass in train mode
    model.train()
    y = model(x)
    print(y[0][0][0][0])
    y = model(x)
    print(y[0][0][0][0])

    # forward pass in eval mode
    model.eval()
    y = model(x)
    print(y[0][0][0][0])
    y = model(x)
    print(y[0][0][0][0])

    # print output shape
    print(y.shape)