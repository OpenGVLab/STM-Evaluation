_base_ = [
    '../mask_rcnn_meta.py'
]

dims = [32, 64, 160, 256]
depths = [2, 2, 3, 2]
num_heads = [1, 2, 5, 8]
mlp_ratios = [8, 8, 4, 4]
qkv_bias = True
sr_ratios = [8, 4, 2, 1]
pretrained = '/mnt/petrelfs/wangweiyun/model_ckpt/unified_pvt_micro.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='UnifiedPVT',
        dims=dims,
        depths=depths,
        block_kwargs=dict(
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            sr_ratios=sr_ratios,
        ),
        drop_path_rate=0.1,
        out_indices=(0, 1, 2, 3),
        pretrained=pretrained),
    neck=dict(in_channels=dims),
)
