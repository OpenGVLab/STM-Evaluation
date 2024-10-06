from timm.models import register_model
from ..meta_arch import MetaArch
from ..blocks.dcn_v3 import DCNv3Block, DCNv3SingleResBlock


@register_model
def unified_dcn_v3_micro(pretrained=False, **kwargs):
    dims = [32 * 2 ** i for i in range(4)]
    depths = [2, 2, 9, 2]
    num_heads = [2, 4, 8, 16]
    deform_points = 9
    deform_padding = True
    kernel_size = 3

    model = MetaArch(depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=deform_points, kernel_size=kernel_size, deform_padding=deform_padding),
                     forward_kwargs=dict(deform_points=deform_points, deform_padding=deform_padding),
                     drop_path_rate=0,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_dcn_v3_tiny(pretrained=False, **kwargs):
    dims = [64 * 2 ** i for i in range(4)]
    depths = [4, 4, 18, 4]
    num_heads = [4, 8, 16, 32]
    deform_points = 9
    deform_padding = True
    kernel_size = 3

    model = MetaArch(depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=deform_points, kernel_size=kernel_size, deform_padding=deform_padding),
                     forward_kwargs=dict(deform_points=deform_points, deform_padding=deform_padding),
                     drop_path_rate=0.1,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_dcn_v3_small(pretrained=False, **kwargs):
    dims = [80 * 2 ** i for i in range(4)]
    depths = [4, 4, 21, 4]
    num_heads = [5, 10, 20, 40]
    deform_points = 9
    deform_padding = True
    kernel_size = 3

    model = MetaArch(depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=deform_points, kernel_size=kernel_size, deform_padding=deform_padding),
                     forward_kwargs=dict(deform_points=deform_points, deform_padding=deform_padding),
                     drop_path_rate=0.3,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_dcn_v3_base(pretrained=False, **kwargs):
    dims = [112 * 2 ** i for i in range(4)]
    depths = [4, 4, 21, 4]
    num_heads = [7, 14, 28, 56]
    deform_points = 9
    deform_padding = True
    kernel_size = 3

    model = MetaArch(depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=deform_points, kernel_size=kernel_size, deform_padding=deform_padding),
                     forward_kwargs=dict(deform_points=deform_points, deform_padding=deform_padding),
                     drop_path_rate=0.5,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

@register_model
def unified_dcn_v3_large(pretrained=False, **kwargs):
    dims = [160 * 2 ** i for i in range(4)]
    depths = [5, 5, 22, 5]
    num_heads = [10, 20, 40, 80]
    deform_points = 9
    deform_padding = True
    kernel_size = 3

    model = MetaArch(depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=deform_points, kernel_size=kernel_size, deform_padding=deform_padding),
                     forward_kwargs=dict(deform_points=deform_points, deform_padding=deform_padding),
                     drop_path_rate=0.5,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model
