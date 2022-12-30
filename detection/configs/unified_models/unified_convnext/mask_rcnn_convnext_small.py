_base_ = [
    '../mask_rcnn_meta.py'
]

depths = [2, 2, 24, 2]
dims = [96, 192, 384, 768]
pretrained = '/mnt/petrelfs/wangweiyun/model_ckpt/unified_convnext_v3_small.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='UnifiedConvNeXt',
        dims=dims,
        depths=depths,
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        pretrained=pretrained),
    neck=dict(in_channels=dims),
)
