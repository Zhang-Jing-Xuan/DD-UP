_base_ = './mask-rcnn_r50_fpn_8xb8-amp-lsj-200e_coco.py'

model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='/root/autodl-tmp/mmdetection/imagenet-res18.pth')),
    neck=dict(in_channels=[64, 128, 256, 512]))
