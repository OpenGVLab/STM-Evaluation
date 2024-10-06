# Object Detection
We use Mask-RCNN to evaluate the down-stream performance. Please refer to the paper for detailed settings.

## Requirements
First follow the installation guidance in the main `readme` file to prepare the environment. For object detection, additional packages shall be installed:
+ mmcv (1.x.0) [installation guidance](https://mmcv.readthedocs.io/en/latest/)
+ mmdet (1.x.0) [installation guidance](https://mmdetection.readthedocs.io/en/latest/)
Other version may also work.

## Results
| Scale |       Model        | Box AP | Mask AP |                                                          checkpoint                                                           |
| :---: | :----------------: | :----: | :-----: | :---------------------------------------------------------------------------------------------------------------------------: |
| Micro |     U-HaloNet      |  40.3  |  37.3   |   [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_halonet_micro_1x.pth)   |
|       |       U-PVT        |  35.9  |  34.2   |     [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_pvt_micro_1x.pth)     |
|       | U-Swin Transformer |  36.6  |  34.6   |    [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_swin_micro_1x.pth)     |
|       |     U-ConvNeXt     |  39.2  |  36.4   |  [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_convnext_micro_1x.pth)   |
|       |   U-InternImage    |  39.5  |  36.6   | [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_internimage_micro_1x.pth) |
| Tiny  |     U-HaloNet      |  46.9  |  42.4   |   [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_halonet_tiny_1x.pth)    |
|       |       U-PVT        |  44.2  |  40.6   |     [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_pvt_tiny_1x.pth)      |
|       | U-Swin Transformer |  44.3  |  40.5   |     [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_swin_tiny_1x.pth)     |
|       |     U-ConvNeXt     |  44.3  |  40.5   |   [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_convnext_tiny_1x.pth)   |
|       |   U-InternImage    |  47.2  |  42.5   | [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_internimage_tiny_1x.pth)  |
| Small |     U-HaloNet      |  48.2  |  43.3   |   [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_halonet_small_1x.pth)   |
|       |       U-PVT        |  46.1  |  41.9   |     [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_pvt_small_1x.pth)     |
|       | U-Swin Transformer |  46.4  |  42.1   |    [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_swin_small_1x.pth)     |
|       |     U-ConvNeXt     |  45.6  |  41.2   |  [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_convnext_small_1x.pth)   |
|       |   U-InternImage    |  47.8  |  43.0   | [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_internimage_small_1x.pth) |
| Base  |     U-HaloNet      |  49.0  |  43.8   |   [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_halonet_base_1x.pth)    |
|       |       U-PVT        |  46.4  |  42.3   |     [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_pvt_base_1x.pth)      |
|       | U-Swin Transformer |  47.0  |  42.2   |     [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_swin_base_1x.pth)     |
|       |     U-ConvNeXt     |  46.7  |  42.2   |   [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_convnext_base_1x.pth)   |
|       |   U-InternImage    |  48.7  |  43.8   | [Download](https://github.com/OpenGVLab/STM-Evaluation/releases/download/det-ckpt/mask_rcnn_unified_internimage_base_1x.pth)  |

## Usage
### Training
- To train a mask rcnn based on unified-swin with slurm
  - remember to modify the path to the pertained ckpt in `./configs/unified_models/`
```bash
# MODEL_TYPE: halonet, pvt, swin, convnext
# SCALE: micro, tiny, small, base
bash shells/train.py [MODEL_TYPE] [SCALE]
```

- For training on a single machine, run the following command:

```shell
bash shells/dist_train.py [CONFIG] [NUM_GPUS] --auto-scale-lr
```

### Evaluation

- To test the trained model with slurm

```bash
bash shells/test.py [MODEL_TYPE] [SCALE]
```

- For single machine evaluation, run the following command:

```shell
bash shells/dist_test.py [CONFIG] [CKPT_PATH] [NUM_GPUS]
```


