import torch
import torch.nn as nn
import torch.nn.functional as F
import orthogonium.layers as ol
import math

from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, ReLU


class SimpleEncoder(nn.Module):
    def __init__(self, conv, in_channels=3, dim_repeats=[(64, 3), (128, 3), (256, 3)]):
        super(SimpleEncoder, self).__init__()
        prev_channels = in_channels
        layers = []
        norm_layer = BatchNorm2d

        for dim, nb in dim_repeats:
            layers.append(
                conv(prev_channels, dim, kernel_size=3, stride=2, padding=1, bias=False)
            )
            layers.append(ReLU())
            layers.append(norm_layer(dim))
            for i in range(nb - 1):
                layers.append(
                    conv(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
                )
                layers.append(ReLU())
                layers.append(norm_layer(dim))
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

    def __init__(self, conv, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []
        modules.append(conv(in_channels, out_channels, kernel_size=1, bias=False))
        for rate in atrous_rates:
            modules.append(
                conv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=rate,
                    padding_mode="zeros",
                    dilation=rate,
                    bias=False,
                )
            )
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            conv(len(modules) * out_channels, out_channels, kernel_size=1, bias=False),
            ReLU(),
            BatchNorm2d(out_channels),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        x_cat = torch.cat(res, dim=1)
        x_scaled = x_cat
        out = self.project(x_scaled)
        return out


class DeepLabV3(nn.Module):
    """
    DeepLabV3 with a basic encoder and ASPP module,
    using classic Conv2d and ReLU, and ConvTranspose2d for upsampling.
    """

    def __init__(
        self,
        mean,
        std,
        conv=Conv2d,
        conv_tr=ConvTranspose2d,
        in_channels=3,
        num_classes=21,
        encoder_layers=[(64, 3), (128, 3), (256, 3)],
        aspp_out=256,
    ):
        super(DeepLabV3, self).__init__()
        encoder_channels = encoder_layers[-1][0]

        self.input_norm = Normalize(mean, std)
        self.encoder = SimpleEncoder(
            conv, in_channels=in_channels, dim_repeats=encoder_layers
        )
        self.aspp = ASPP(conv, in_channels=encoder_channels, out_channels=aspp_out)
        norm_layer = BatchNorm2d

        convtr_layers = []
        previous = aspp_out
        for i in range(len(encoder_layers)):
            convtr_layers.append(
                conv_tr(
                    previous,
                    previous // 2,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=False,
                )
            )
            convtr_layers.append(ReLU())
            convtr_layers.append(norm_layer(previous // 2))
            previous = previous // 2

        self.upsampling = nn.Sequential(
            conv(aspp_out, aspp_out, kernel_size=3, padding=1, bias=False),
            ReLU(),
            *convtr_layers,
        )
        self.classifier = conv(previous, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.input_norm(x)
        feats = self.encoder(x)
        x = self.aspp(feats)
        x = self.upsampling(x)
        x = self.classifier(x)
        return x


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        assert len(mean) == len(std)
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        assert x.ndim == 4
        x = x - self.mean.view(1, -1, 1, 1).to(x.device)
        x = x / self.std.view(1, -1, 1, 1).to(x.device)
        return x


def deeplabv3_unconstrained(mean, std, in_channels, num_classes, config):
    configs = {
        "S": [(32, 3), (64, 3), (128, 3)],
        "M1": [(64, 5), (128, 5), (256, 5)],
        "M2": [(64, 5), (128, 5), (256, 5), (512, 5)],
        "L": [(128, 5), (256, 5), (512, 5), (1024, 5)],
        "XL": [(128, 5), (256, 5), (512, 5), (1024, 5), (2048, 5)],
    }
    encoder_layers = configs[config]
    aspp_out = encoder_layers[-1][0]
    model = DeepLabV3(
        mean,
        std,
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_layers=encoder_layers,
        aspp_out=aspp_out,
    )
    return model
