"""
Project: DUDES
Authors: Steven Landgraf, Kira Wursthorn, Markus Hillemann, Markus Ulrich
"""

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import config
from dataset import PascalVOCDataModule
from model import MCD_ViT


if __name__ == '__main__':
    L.seed_everything(42, workers=True)

    model = MCD_ViT()

    data_module = PascalVOCDataModule(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    trainer = L.Trainer(
        max_epochs=config.NUM_EPOCHS,
        accelerator='gpu',
        strategy='ddp',
        devices=config.DEVICES,
        precision=config.PRECISION,
        log_every_n_steps = 47,
        check_val_every_n_epoch=1, 
        logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=True),
        callbacks=[
            ModelCheckpoint(dirpath=f'./checkpoints/{config.RUN_NAME}', filename='model-{epoch}-{val_iou:.3f}', monitor='val_iou', mode='max', save_top_k=3),
            LearningRateMonitor(logging_interval='epoch'),
        ],
    )

    trainer.fit(model, data_module)
