import argparse
from typing import Callable, Dict, Optional, Union

from src.utils.model_utils import smp_encoders

from ..base import ClassificationHead, SegmentationHead, SegmentationModel
from ..encoders import get_encoder
from .decoder import PANDecoder


def add_predefined_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--encoder", default="resnet18", choices=smp_encoders, help="Name of encoder to use with smp models"
    )
    parser.add_argument("--encoder_weights", default="imagenet", help="Set of pre-trained encoder weights")
    parser.add_argument("--in_channels", default=3, help="Number of input channels")
    parser.add_argument("--classes", default=1, help="Number of classes to predict")


class Model(SegmentationModel):
    """Implementation of PAN_ (Pyramid Attention Network).

    Note:
        Currently works with shape of input tensor >= [B x C x 128 x 128] for pytorch <= 1.1.0
        and with shape of input tensor >= [B x C x 256 x 256] for pytorch == 1.3.1

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride: 16 or 32, if 16 use dilation in encoder last layer.
            Doesn't work with ***ception***, **vgg***, **densenet*`** backbones.Default is 16.
        decoder_channels: A number of convolution layer filters in decoder blocks
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to
                return logits)

    Returns:
        ``torch.nn.Module``: **PAN**

    .. _PAN:
        https://arxiv.org/abs/1805.10180

    """

    NAME = "PAN"

    def __init__(
        self,
        encoder: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: int = 16,
        decoder_channels: int = 32,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        **kwargs: Dict,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        if encoder_output_stride not in [16, 32]:
            raise ValueError("PAN support output stride 16 or 32, got {}".format(encoder_output_stride))

        self.encoder = get_encoder(
            encoder,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )

        self.decoder = PANDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=upsampling,
        )

        self.classification_head: Optional[ClassificationHead] = None
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)

        self.name = "pan-{}".format(encoder)
        self.initialize()
