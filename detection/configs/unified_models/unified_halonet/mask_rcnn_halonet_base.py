_base_ = [
    '../mask_rcnn_meta.py'
]

dims = [128 * 2 ** i for i in range(4)]
depths = [2, 2, 18, 2]
num_heads = [4, 8, 16, 32]
block_size = 7
halo_size = 3
pretrained = '/mnt/petrelfs/wangweiyun/model_ckpt/unified_halonet_v2_base.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='UnifiedHalonet',
        dims=dims,
        depths=depths,
        block_kwargs=dict(
            num_heads=num_heads,
            block_size=block_size,
            halo_size=halo_size,
        ),
        drop_path_rate=0.3,
        out_indices=(0, 1, 2, 3),
        pretrained=pretrained),
    neck=dict(in_channels=dims),
)
