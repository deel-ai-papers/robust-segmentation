import torch
import torch.nn as nn
import math

from deel.torchlip import SpectralConv2d, SpectralConvTranspose2d
from orthogonium.layers import AdaptiveOrthoConv2d as AOC2d
from orthogonium.layers import AdaptiveOrthoConvTranspose2d as AOCTr2d


def sqrt_fan_ratio_conv2d(m: nn.Conv2d) -> float:
    return math.sqrt((m.out_channels * m.groups) / m.in_channels)


def sqrt_fan_ratio_convtr2d(m: nn.ConvTranspose2d) -> float:
    return math.sqrt((m.in_channels * m.groups) / m.out_channels)


class WrapLipFn(nn.Module):
    def __init__(self, module, lipconstant=1.0):
        super().__init__()
        self.module = module
        self.register_buffer(
            "lipconstant", torch.as_tensor(float(lipconstant), dtype=torch.float32)
        )

    def forward(self, x_lip):
        assert isinstance(x_lip, tuple)
        x, prev_coeffs = x_lip
        return self.module(x), self.lipconstant * prev_coeffs


class UnscaleModule(nn.Module):
    def __init__(self, module, lipconstant=1.0):
        super().__init__()
        self.mod = module
        self.register_buffer(
            "lipconstant", torch.as_tensor(float(lipconstant), dtype=torch.float32)
        )

    def forward(self, x_lip, return_lipconstant=False):
        assert isinstance(x_lip, tuple)
        x, prev_coeffs = x_lip
        return self.mod(x) / (prev_coeffs * self.lipconstant + 1e-10)


class ScaledModule(nn.Module):
    def __init__(
        self,
        module,
        init_val=1.0,
        adjust_to_rms=False,
        learnable_coeffs=True,
    ):
        super().__init__()
        self.mod = module

        init_val = 1.0
        if adjust_to_rms:
            assert self.mod.weight.ndim == 4
            if hasattr(self.mod, "transposed"):
                if self.mod.transposed:
                    init_val = sqrt_fan_ratio_convtr2d(self.mod)
                else:
                    init_val = sqrt_fan_ratio_conv2d(self.mod)
            else:
                raise ValueError(
                    "Only Conv2d and ConvTranspose2d supported for adjust_to_rms"
                )

        val = torch.tensor(float(init_val), dtype=torch.get_default_dtype())
        self.coeff = torch.nn.Parameter(val, requires_grad=learnable_coeffs)

    def forward(self, x_lip):
        assert isinstance(x_lip, tuple)
        x, prev_coeffs = x_lip
        return self.coeff * self.mod(x), prev_coeffs * self.coeff.abs()


class AddResidual(nn.Module):
    def __init__(self, modules, learnable_coeffs=True):
        super().__init__()
        self.module_list = nn.ModuleList(modules)
        if learnable_coeffs:
            self.coeffs = torch.nn.ParameterList(
                [torch.tensor(1.0 / len(modules)) for _ in range(len(modules))]
            )
        else:
            self.coeffs = [
                torch.tensor(1.0 / len(modules)) for _ in range(len(modules))
            ]

    def forward(self, x_lip):
        assert isinstance(x_lip, tuple)
        _, prev_coeffs = x_lip
        sum, lipconstant = 0, 0.0
        for coef, mod in zip(self.coeffs, self.module_list):
            y, l = mod(x_lip)
            sum += coef * y
            lipconstant += coef.abs() * (l / prev_coeffs)
        return sum, prev_coeffs * lipconstant


class ConcatResidual(nn.Module):
    def __init__(self, modules, learnable_coeffs=True):
        super().__init__()
        self.module_list = nn.ModuleList(modules)
        if learnable_coeffs:
            self.coeffs = torch.nn.ParameterList(
                [
                    torch.tensor(1.0 / math.sqrt(len(modules)))
                    for _ in range(len(modules))
                ]
            )
        else:
            self.coeffs = [
                torch.tensor(1.0 / math.sqrt(len(modules))) for _ in range(len(modules))
            ]

    def forward(self, x_lip):
        assert isinstance(x_lip, tuple)
        _, prev_coeffs = x_lip
        out, lipconstant = [], 0.0
        for coef, mod in zip(self.coeffs, self.module_list):
            y, l = mod(x_lip)
            out.append(coef * y)
            lipconstant += (coef.abs() * (l / prev_coeffs)) ** 2
        out = torch.cat(out, dim=1)
        return out, prev_coeffs * torch.sqrt(lipconstant)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.id = nn.Identity()

    def forward(self, x_lip):
        assert isinstance(x_lip, tuple)
        x, prev_coeffs = x_lip
        return x, prev_coeffs


class InputNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        assert len(mean) == len(std)
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def get_lipconstant(self):
        return (1 / torch.min(self.std)).abs()

    def forward(self, x):
        assert x.ndim == 4
        x = x - self.mean.view(1, -1, 1, 1).to(x.device)
        x = x / self.std.view(1, -1, 1, 1).to(x.device)
        return x, self.get_lipconstant()
