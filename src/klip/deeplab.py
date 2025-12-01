import torch
import torch.nn as nn
import torch.nn.functional as F
import orthogonium.layers as ol
import math
from typing import Literal

from deel.torchlip import SpectralConv2d, SpectralConvTranspose2d
from orthogonium.layers import AdaptiveOrthoConv2d as AOC2d
from orthogonium.layers import AdaptiveOrthoConvTranspose2d as AOCTr2d
from orthogonium.layers import MaxMin
from orthogonium.layers.normalization import BatchCentering2D
from functools import partial

from .layers import (
    WrapLipFn,
    ScaledModule,
    ConcatResidual,
    AddResidual,
    Identity,
    InputNormalize,
)


class SimpleEncoder(nn.Module):
    def __init__(
        self,
        conv,
        in_channels=3,
        dim_repeats=[(64, 3), (128, 3), (256, 3)],
        learnable_coeffs=False,
    ):
        super(SimpleEncoder, self).__init__()
        prev_channels = in_channels
        layers = []
        norm_layer = BatchCentering2D

        for dim, nb in dim_repeats:
            layers.append(
                ScaledModule(
                    conv(
                        prev_channels,
                        dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    adjust_to_rms=True,
                    learnable_coeffs=learnable_coeffs,
                )
            )
            layers.append(WrapLipFn(MaxMin()))
            layers.append(WrapLipFn(norm_layer(dim)))
            for i in range(nb - 1):
                layers.append(
                    ScaledModule(
                        conv(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                        adjust_to_rms=True,
                        learnable_coeffs=learnable_coeffs,
                    )
                )
                layers.append(WrapLipFn(MaxMin()))
                layers.append(WrapLipFn(norm_layer(dim)))
            prev_channels = dim
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling without global pooling branch.
    After concatenation, divides by sqrt(number of branches).
    """

    def __init__(
        self,
        conv,
        in_channels,
        out_channels=256,
        atrous_rates=[6, 12, 18],
        learnable_coeffs=False,
    ):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            ScaledModule(
                conv(in_channels, out_channels, kernel_size=1, bias=False),
                adjust_to_rms=True,
                learnable_coeffs=learnable_coeffs,
            )
        )
        for rate in atrous_rates:
            modules.append(
                ScaledModule(
                    conv(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rate,
                        padding_mode="zeros",
                        dilation=rate,
                        bias=False,
                    ),
                    adjust_to_rms=True,
                    learnable_coeffs=learnable_coeffs,
                )
            )

        self.convs = ConcatResidual(modules, learnable_coeffs=learnable_coeffs)
        self.project = nn.Sequential(
            ScaledModule(
                conv(
                    len(modules) * out_channels, out_channels, kernel_size=1, bias=False
                ),
                adjust_to_rms=True,
                learnable_coeffs=learnable_coeffs,
            ),
            WrapLipFn(MaxMin()),
            WrapLipFn(BatchCentering2D(out_channels)),
        )

    def forward(self, x):
        x = self.convs(x)
        out = self.project(x)
        return out


class KLipDeepLabV3(nn.Module):
    """
    Simple DeepLabV3 with a basic encoder and ASPP module,
    using AOC2d and MaxMin, and transposed conv for upsampling.
    """

    def __init__(
        self,
        mean,
        std,
        conv=AOC2d,
        conv_tr=AOCTr2d,
        in_channels=3,
        num_classes=21,
        encoder_layers=[(64, 3), (128, 3), (256, 3)],
        aspp_out=256,
        learnable_coeffs=False,
    ):
        super(KLipDeepLabV3, self).__init__()
        assert len(mean) == in_channels
        assert len(std) == in_channels

        encoder_channels = encoder_layers[-1][0]

        self.input_norm = InputNormalize(mean, std)
        self.encoder = SimpleEncoder(
            conv,
            in_channels=in_channels,
            dim_repeats=encoder_layers,
            learnable_coeffs=learnable_coeffs,
        )
        self.aspp = ASPP(
            conv,
            in_channels=encoder_channels,
            out_channels=aspp_out,
            learnable_coeffs=learnable_coeffs,
        )
        norm_layer = BatchCentering2D

        convtr_layers = []
        previous = aspp_out
        for i in range(len(encoder_layers)):
            convtr_layers.append(
                ScaledModule(
                    conv_tr(
                        previous,
                        previous // 2,
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        bias=False,
                    ),
                    adjust_to_rms=True,
                    learnable_coeffs=learnable_coeffs,
                )
            )
            convtr_layers.append(WrapLipFn(MaxMin()))
            convtr_layers.append(WrapLipFn(norm_layer(previous // 2)))
            previous = previous // 2

        self.upsampling = nn.Sequential(
            ScaledModule(
                conv(aspp_out, aspp_out, kernel_size=3, padding=1, bias=False),
                adjust_to_rms=True,
                learnable_coeffs=learnable_coeffs,
            ),
            WrapLipFn(MaxMin()),
            *convtr_layers,
        )
        self.classifier = ScaledModule(
            conv(previous, num_classes, kernel_size=1, stride=1),
            adjust_to_rms=True,
            learnable_coeffs=learnable_coeffs,
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.encoder(x)
        x = self.aspp(x)
        x = self.upsampling(x)
        x = self.classifier(x)
        return x


def deeplabv3_klipschitz(
    mean,
    std,
    in_channels,
    num_classes,
    config,
    param: Literal["ortho", "spectral"] = "ortho",
    learnable_coeffs: bool = False,
):
    """
    Param can be "ortho" or "spectral" for orthogonal or spectral convolutions.
    Config can be "S", "M1", "M2", "L", or "XL" for different model sizes.
    """
    conv = {
        "ortho": AOC2d,
        "spectral": SpectralConv2d,
    }[param]
    conv_tr = {
        "ortho": AOCTr2d,
        "spectral": SpectralConvTranspose2d,
    }[param]
    configs = {
        "S": [(32, 3), (64, 3), (128, 3)],
        "M1": [(64, 5), (128, 5), (256, 5)],
        "M2": [(64, 5), (128, 5), (256, 5), (512, 5)],
        "L": [(64, 7), (128, 7), (256, 7), (512, 7)],
        "XL": [(128, 5), (256, 5), (512, 5), (1024, 5), (2048, 5)],
    }
    encoder_layers = configs[config]
    aspp_out = encoder_layers[-1][0]
    model = KLipDeepLabV3(
        mean,
        std,
        conv,
        conv_tr,
        in_channels,
        num_classes,
        encoder_layers,
        aspp_out,
        learnable_coeffs,
    )
    return model


if __name__ == "__main__":
    for config in ["S", "M1", "M2", "L", "XL"]:
        for param in ["spectral", "ortho"]:
            print(f"Testing config {config} with param {param}")
            model = deeplabv3_klipschitz(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                in_channels=3,
                num_classes=21,
                config=config,
                param=param,
            )
            x = torch.randn(2, 3, 224, 224)
            y = model(x)
            print(y[0].shape)  # should be (2, 21, 224, 224)
            print("Init lipconstant: ", y[1])
            assert y[0].shape == (2, 21, 224, 224)
