from timm.models import register_model
from ..meta_arch import MetaArch
from ..blocks.convnext import (
    ConvNeXtBlock, UnifiedConvNeXtBlock,
    ConvNeXtStem, ConvNeXtDownsampleLayer,
)

@ register_model
def unified_convnext_micro(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 8, 3],
                     dims=[32, 64, 128, 256],
                     block_type=UnifiedConvNeXtBlock,
                     drop_path_rate=0,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[2, 2, 9, 2],
                     dims=[96, 192, 384, 768],
                     block_type=UnifiedConvNeXtBlock,
                     drop_path_rate=0.1,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_small(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[2, 2, 24, 2],
                     dims=[96, 192, 384, 768],
                     block_type=UnifiedConvNeXtBlock,
                     drop_path_rate=0.4,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_base(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[2, 2, 24, 2],
                     dims=[128, 256, 512, 1024],
                     block_type=UnifiedConvNeXtBlock,
                     drop_path_rate=0.5,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model