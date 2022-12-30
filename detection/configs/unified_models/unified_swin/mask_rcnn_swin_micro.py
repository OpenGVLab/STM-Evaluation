_base_ = [
    '../mask_rcnn_meta.py'
]

dims = [32 * 2 ** i for i in range(4)]
depths = [2, 2, 9, 2]
num_heads = [1, 2, 4, 8]
window_size = 7
pretrained = '/mnt/petrelfs/wangweiyun/model_ckpt/unified_swin_micro.pth'

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
        drop_path_rate=0.1,
        out_indices=(0, 1, 2, 3),
        pretrained=pretrained),
    neck=dict(in_channels=dims),
)
