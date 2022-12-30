import torch
from torch import nn
from mmdet.models.builder import BACKBONES
from timm.models.layers import DropPath, to_2tuple
from ..meta_arch import LayerNorm2d, MetaArch


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path, layer_scale_init_value, kernel_size=7, **kwargs):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2,
                                groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x.permute(0, 2, 3, 1).contiguous()
        x = self.dwconv(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = shortcut + self.drop_path(x)

        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


"""
To distinguish between the original convnext block, we use ConvNeXtV3 to denote the model we use in our paper
"""
class ConvNeXtV3Block(nn.Module):
    # double res + in/out proj
    def __init__(self, dim, drop_path, layer_scale_init_value, kernel_size=7, **kwargs):
        super().__init__()
        self.dw_norm = LayerNorm2d(dim, eps=1e-6)
        self.dw_input_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.dwconv = nn.Conv2d(dim, dim,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2,
                                groups=dim)
        self.dw_out_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        # pointwise/1x1 convs, implemented with linear layers
        self.pw_norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((1, 1, 1, dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(self.dw_input_proj(self.dw_norm(x)))
        x = self.dw_out_proj(x)
        if self.gamma_1 is not None:
            x = self.gamma_1 * x
        x = shortcut + self.drop_path(x)

        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()

        shortcut = x
        x = self.pw_norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma_2 is not None:
            x = self.gamma_2 * x
        x = shortcut + self.drop_path(x)

        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

class ConvNeXtStem(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4),
            LayerNorm2d(out_channels, eps=1e-6)
        )
        self.grid_size = (img_size[0] // 4, img_size[1] // 4)

    def forward(self, x):
        return self.stem(x)


class ConvNeXtDownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.reduction = nn.Sequential(
            LayerNorm2d(in_channels, eps=1e-6),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.reduction(x)


@BACKBONES.register_module()
class UnifiedConvNeXt(MetaArch):
    def __init__(self, *args, **kwargs):
        kwargs['block_type'] = ConvNeXtV3Block
        super().__init__(*args, **kwargs)
