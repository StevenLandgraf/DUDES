"""
Project: DUDES
Authors: Steven Landgraf, Kira Wursthorn, Markus Hillemann, Markus Ulrich
"""

import argparse

parser = argparse.ArgumentParser(description='DUDES Parser')

parser.add_argument('--entity', type=str, default='-')
parser.add_argument('--project', type=str, default='DUDES (Default)')
parser.add_argument('--run', type=str, default='DUDES (Default)')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dataset', type=str, default='PascalVOC')
parser.add_argument('--dropout', type=float, default=0.2)

args = parser.parse_args()

## W&B Logging
ENTITY = args.entity
PROJECT = args.project
RUN_NAME = args.run

## Training + Hyperparameters
DEVICES = [3]               # GPUs to use
NUM_EPOCHS = args.epochs    # Number of epochs to train for
LEARNING_RATE = args.lr     # linearly scale learning rate with number of GPUs
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_UNC_SAMPLES = 10        # Number of samples to draw from MC Dropout for uncertainty estimation
PRECISION = '16-mixed'

## MCD ViT Specific
DROPOUT_RATE = args.dropout
DROPOOUT_PATH_RATE = 0.1

## Dataset
DATASET = args.dataset      # Add more datasets later

if DATASET == 'Cityscapes':
    NUMBER_TRAIN_IMAGES = 2975
    NUMBER_VAL_IMAGES = 500
    BATCH_SIZE = 16
    NUM_CLASSES = 19
    IGNORE_INDEX = 255
    NUM_WORKERS = 8         

elif DATASET == 'PascalVOC':
    NUMBER_TRAIN_IMAGES = 1464
    NUMBER_VAL_IMAGES = 1449
    BATCH_SIZE = 16
    NUM_CLASSES = 21
    IGNORE_INDEX = 255
    NUM_WORKERS = 4       