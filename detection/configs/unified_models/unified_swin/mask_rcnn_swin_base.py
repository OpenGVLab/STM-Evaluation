_base_ = [
    '../mask_rcnn_meta.py'
]

dims = [128 * 2 ** i for i in range(4)]
depths = [2, 2, 18, 2]
num_heads = [4, 8, 16, 32]
window_size = 7
pretrained = '/mnt/petrelfs/wangweiyun/model_ckpt/unified_swin_base.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='UnifiedSwinTransformer',
        dims=dims,
        depths=depths,
        block_kwargs=dict(
            num_heads=num_heads,
            window_size=window_size,
        ),
        drop_path_rate=0.3,
        out_indices=(0, 1, 2, 3),
        pretrained=pretrained),
    neck=dict(in_channels=dims),
)
