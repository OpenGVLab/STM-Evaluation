_base_ = [
    '../mask_rcnn_meta.py'
]

dims = [80 * 2 ** i for i in range(4)]
depths = [4, 4, 21, 4]
num_heads = [5, 10, 20, 40]
deform_points = 9
deform_padding = True
kernel_size = 3
pretrained = '/mnt/petrelfs/wangweiyun/model_ckpt/unified_dcn_v3_small.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='UnifiedDCNv3',
        dims=dims,
        depths=depths,
        block_kwargs=dict(
            num_heads=num_heads,
            deform_points=deform_points,
            kernel_size=kernel_size,
            deform_padding=deform_padding),
        forward_kwargs=dict(
            deform_points=deform_points,
            deform_padding=deform_padding),
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        pretrained=pretrained),
    neck=dict(in_channels=dims),
)
