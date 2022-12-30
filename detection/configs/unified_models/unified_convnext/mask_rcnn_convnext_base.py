_base_ = [
    '../mask_rcnn_meta.py'
]

depths = [2, 2, 24, 2]
dims = [128, 256, 512, 1024]
pretrained = '/mnt/petrelfs/wangweiyun/model_ckpt/unified_convnext_v3_base.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='UnifiedConvNeXt',
        dims=dims,
        depths=depths,
        drop_path_rate=0.3,
        out_indices=(0, 1, 2, 3),
        pretrained=pretrained),
    neck=dict(in_channels=dims),
)
