"""
Project: Deep Uncertainty Distillation using Ensembles for Segmentation
Authors: Steven Landgraf, Kira Wursthorn, Markus Hillemann, Markus Ulrich
"""

import torch
from segmentation_models_pytorch.base.heads import SegmentationHead
from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3Plus, DeepLabV3PlusDecoder

from __init__ import DEVICE
from dataset import SupervisedCityscapes


class DUDESLabV3Plus(DeepLabV3Plus):
    def __init__(self, uncertainty_classes):
        super().__init__(
            encoder_name = 'resnet18',
            encoder_depth = 5,
            encoder_weights = 'imagenet',
            # encoder_weights = None,   # If you want to try without pretrained weights (works, but not recommended as it converges slower)
            encoder_output_stride = 16,
            decoder_channels = 256,
            decoder_atrous_rates = (12, 24, 36),
            in_channels = 3,
            classes = 19,
            activation = None,
            upsampling = 4,
            aux_params = None,
        )

        # Use the regular segmentation head with sigmoid activations for uncertainty head
        # self.uncertainty_decoder = DeepLabV3PlusDecoder(
        #     encoder_channels=self.encoder.out_channels,
        #     out_channels=256,
        #     atrous_rates=(12, 24, 36),
        #     output_stride=16,
        # )

        self.uncertainty_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=uncertainty_classes,
            activation='sigmoid',
            kernel_size=1,
            upsampling=4,
        )

    def forward(self, x):
        super().check_input_shape(x)

        features = self.encoder(x)

        decoder_output = self.decoder(*features)
        # uncertainty_decoder_output = self.uncertainty_decoder(*features)

        segmentation_prediction = self.segmentation_head(decoder_output)
        uncertainty_prediction = self.uncertainty_head(decoder_output)

        return segmentation_prediction, uncertainty_prediction


if __name__ == '__main__':
    dataset = SupervisedCityscapes(root_dir='./data/cityscapes', split='train')
    
    image, mask = dataset[0]
    image, mask = image.to(device=DEVICE), mask.to(device=DEVICE)

    model = DUDESLabV3Plus().to(device=DEVICE)
    model.eval()

    segmentation_prediction, uncertainty_prediction = model(image.unsqueeze(0))

    print(segmentation_prediction.shape, uncertainty_prediction.shape)
