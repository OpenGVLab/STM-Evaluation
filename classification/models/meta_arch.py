import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from torch import nn
from timm.models.layers import to_2tuple, trunc_normal_
from .blocks.pvt import PvtBlock


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class Stem(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 img_size,
                 norm_layer,
                 act_layer,
                 ratio=0.5,
                 **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.grid_size = (img_size[0] // 4, img_size[1] // 4)

        # input_shape: B x C x H x W
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels * ratio),
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            norm_layer(int(out_channels * ratio)),
            act_layer(),
            nn.Conv2d(int(out_channels * ratio), out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            norm_layer(out_channels)
        )

    def forward(self, x):
        return self.stem(x)


class DownsampleLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_layer,
                 **kwargs):
        super().__init__()

        # input_shape: B x C x H x W
        self.reduction = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1),
                      bias=False),
            norm_layer(out_channels),
        )

    def forward(self, x):
        return self.reduction(x)


class MetaArch(nn.Module):
    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 num_classes=1000,
                 depths=(3, 3, 9, 3),
                 dims=(96, 192, 384, 768),
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 stem_type=Stem,
                 stem_kwargs=None,
                 block_type=None,
                 block_kwargs=None,
                 downsample_type=DownsampleLayer,
                 downsample_kwargs=None,
                 extra_transform=True,
                 extra_transform_ratio=1.5,
                 norm_layer=LayerNorm2d,
                 norm_every_stage=True,
                 norm_after_avg=False,
                 act_layer=nn.GELU,
                 forward_kwargs=None,
                 use_checkpoint=False,
                 label_map_path=None,
                 **kwargs,
                 ):
        super().__init__()

        stem_kwargs = stem_kwargs or {}
        block_kwargs = block_kwargs or {}
        downsample_kwargs = downsample_kwargs or {}
        forward_kwargs = forward_kwargs or {}

        self.depths = depths
        self.block_type = block_type
        self.forward_kwargs = forward_kwargs
        self.use_checkpoint = use_checkpoint
        self.label_map_path = label_map_path

        if label_map_path is not None and num_classes != 1000:
            with open(label_map_path, 'r', encoding='utf-8') as file:
                self.label_map = [int(i) for i in file.readlines()]

        # stem + downsample_layers
        stem = stem_type(in_channels=in_channels,
                         out_channels=dims[0],
                         img_size=img_size,
                         norm_layer=norm_layer,
                         norm_first=False,
                         act_layer=act_layer,
                         **stem_kwargs)
        # H, W
        self.patch_grid = stem.grid_size
        self.downsample_layers = nn.ModuleList([stem])
        for i in range(3):
            self.downsample_layers.append(downsample_type(in_channels=dims[i],
                                                          out_channels=dims[i+1],
                                                          norm_layer=norm_layer,
                                                          norm_first=True,
                                                          img_size=(self.patch_grid[0] // (2 ** i), self.patch_grid[1] // (2 ** i)),
                                                          **downsample_kwargs))

        # blocks
        cur = 0
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        self.stage_norms = nn.ModuleList()

        for i, (depth, dim) in enumerate(zip(depths, dims)):
            self.stages.append(nn.ModuleList(
                [block_type(dim=dim,
                            drop_path=dp_rates[cur + j],
                            stage=i,
                            depth=j,
                            total_depth=cur+j,
                            input_resolution=(self.patch_grid[0] // (2 ** i), self.patch_grid[1] // (2 ** i)),
                            layer_scale_init_value=layer_scale_init_value,
                            **block_kwargs)
                 for j in range(depth)]
            ))
            self.stage_norms.append(norm_layer(dim) if norm_every_stage else nn.Identity())
            cur += depths[i]

        self.stage_end_norm = nn.Identity() if norm_every_stage or norm_after_avg else norm_layer(dims[-1])

        self.conv_head = nn.Sequential(
            nn.Conv2d(dims[-1], int(dims[-1] * extra_transform_ratio), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(dims[-1] * extra_transform_ratio)),
            act_layer()
        ) if extra_transform else nn.Identity()

        features = int(dims[-1] * extra_transform_ratio) if extra_transform else dims[-1]
        self.avg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            norm_layer(features) if norm_after_avg else nn.Identity(),
            nn.Flatten(1),
        )

        if num_classes > 0:
            self.head = nn.Linear(features, num_classes)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)

    @ torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @ torch.jit.ignore
    def no_weight_decay(self):
        # from swin v1
        no_weight_decay = {'absolute_pos_embed'}
        for name, _ in self.named_parameters():
            if 'relative_position_bias_table' in name:
                no_weight_decay.add(name)

        return no_weight_decay

    def forward_features(self, x):
        extra_inputs = None
        if hasattr(self.block_type, 'extra_inputs'):  # dcn_v3
            extra_inputs = self.block_type.extra_inputs(x, **self.forward_kwargs)

        # shape: (B, C, H, W)
        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)

            if hasattr(self.block_type, 'pre_stage_transform'):  # halonet
                x = self.block_type.pre_stage_transform(x)

            x = x if extra_inputs is None else (x, extra_inputs[i])
            for blk in self.stages[i]:
                if self.use_checkpoint:
                    x = cp.checkpoint(blk, x)
                else:
                    x = blk(x)

            if hasattr(self.block_type, 'post_stage_transform'):
                x = self.block_type.post_stage_transform(x)

            x = x if extra_inputs is None else x[0]
            x = self.stage_norms[i](x)

        x = self.stage_end_norm(x)

        x = self.conv_head(x)
        x = self.avg_head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def load_state_dict(self, state_dict, strict: bool = True):
        new_state_dict = {}
        for key, value in state_dict.items():
            if 'relative_position_index' in key:
                continue

            if 'attn_mask' in key:
                continue

            # swin pos embed
            if 'relative_position_bias_table' in key:
                L1, nH1 = value.shape
                S1 = int(L1 ** 0.5)

                L2, nH2 = self.state_dict()[key].shape
                S2 = int(L2 ** 0.5)

                assert nH1 == nH2

                value = value.permute(1, 0).view(1, nH1, S1, S1)
                value = F.interpolate(value,
                                      size=(S2, S2),
                                      mode='bicubic')
                value = value.view(nH2, L2).permute(1, 0)

            # halonet pos embed
            if 'height_rel' in key or 'width_rel' in key:
                win_size = self.state_dict()[key].shape[0]
                value = value.permute(1, 0).contiguous().unsqueeze(0)
                value = F.interpolate(value,
                                      size=win_size,
                                      mode='linear',
                                      align_corners=True)
                value = value.squeeze(0).permute(1, 0).contiguous()

            if 'head' in key and 'conv' not in key:
                if self.state_dict()[key].shape[0] != value.shape[0]:
                    with open(self.label_map_path, 'r', encoding='utf-8') as file:
                        label_map = [int(i) for i in file.readlines()]
                    value = value[label_map]

            if self.block_type is PvtBlock and 'pos_embed' in key:
                value = F.interpolate(value,
                                      size=self.state_dict()[key].shape[-2:],
                                      mode='bilinear',
                                      align_corners=True)

            new_state_dict[key] = value

        if strict:
            ckpt_keys = new_state_dict.keys()
            model_keys = self.state_dict().keys()

            for key in ckpt_keys:
                assert key in model_keys

            for key in model_keys:
                if 'relative_position_index' in key:
                    continue

                if 'attn_mask' in key:
                    continue

                assert key in ckpt_keys

        return super().load_state_dict(new_state_dict, False)
