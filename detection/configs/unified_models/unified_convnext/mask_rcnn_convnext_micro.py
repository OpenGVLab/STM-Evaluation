_base_ = [
    '../mask_rcnn_meta.py'
]

depths = [3, 3, 8, 3]
dims = [32, 64, 128, 256]
pretrained = '/mnt/petrelfs/wangweiyun/model_ckpt/unified_convnext_v3_micro.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='UnifiedConvNeXt',
        dims=dims,
        depths=depths,
        drop_path_rate=0.1,
        out_indices=(0, 1, 2, 3),
        pretrained=pretrained),
    neck=dict(in_channels=dims),
)
