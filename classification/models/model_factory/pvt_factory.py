"""
Model Factory for PVT v2 and PVT
PVT v2:
PVT:
"""

from timm.models import register_model
from ..meta_arch import MetaArch
from ..blocks.pvt import PvtBlock
from ..blocks.pvt_v2 import PvtV2Block, OverlapPatchEmbed


@register_model
def official_pvt_v2_b0(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[2, 2, 2, 2],
                     dims=[32, 64, 160, 256],
                     stem_type=OverlapPatchEmbed,
                     stem_kwargs=dict(patch_size=7, stride=4),
                     block_type=PvtV2Block,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     downsample_type=OverlapPatchEmbed,
                     downsample_kwargs=dict(patch_size=3, stride=2),
                     extra_transform=False,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_pvt_micro(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[2, 2, 3, 2],
                     dims=[32, 64, 160, 256],
                     block_type=PvtBlock,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     drop_path_rate=0,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_pvt_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 4, 9, 3],
                     dims=[64, 128, 320, 512],
                     block_type=PvtBlock,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     drop_path_rate=0.1,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_pvt_small(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 4, 21, 3],
                     dims=[64, 128, 320, 512],
                     block_type=PvtBlock,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     drop_path_rate=0.3,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_pvt_base(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 8, 45, 3],
                     dims=[64, 128, 320, 512],
                     block_type=PvtBlock,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[4, 4, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     drop_path_rate=0.5,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

@register_model
def unified_pvt_v2_micro(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[2, 2, 3, 2],
                     dims=[32, 64, 160, 256],
                     block_type=PvtV2Block,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     drop_path_rate=0,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_pvt_v2_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 9, 3],
                     dims=[64, 128, 320, 512],
                     block_type=PvtV2Block,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     drop_path_rate=0.1,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_pvt_v2_small(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 21, 3],
                     dims=[64, 128, 320, 512],
                     block_type=PvtV2Block,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     drop_path_rate=0.3,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_pvt_v2_base(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 6, 45, 3],
                     dims=[64, 128, 320, 512],
                     block_type=PvtV2Block,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[4, 4, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     drop_path_rate=0.5,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model
