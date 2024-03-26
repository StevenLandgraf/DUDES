"""
DUDES: Deep Uncertainty Distillation using Ensembles for Segmentation
Authors: Steven Landgraf, Kira Wursthorn, Markus Hillemann, Markus Ulrich
"""

import logging

import pytorch_lightning as pl


logging.basicConfig(format='%(asctime)s | %(levelname)s | %(funcName)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
pl.seed_everything(42)

# Set DEVICE index to your preferred GPU
DEVICE = 3