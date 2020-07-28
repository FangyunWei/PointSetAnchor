# model settings
import os
tmp_this_dir = os.path.dirname(__file__)

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='PointSetAnchorPoseDetector',
    pretrained='open-mmlab://msra/hrnetv2_w32',
    extra_stage_num=1,
    stage2_oks_thre=0.99,
    use_predict_bbx=True,
    heat_reg_group=True,
    backbone=dict(
        type='HRNet',
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))),
    neck=dict(
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256,
        stride=2,
        num_outs=3,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='PointSetAnchorPoseHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        norm_cfg=norm_cfg,
        use_shape_index_feature=True,
        modulated_dcn=False,
        fea_point_index=[0, 7, 8, 9, 10, 13, 14, 15, 16],
        anchor_scales=[16],
        anchor_strides=[8, 16, 32],
        target_means=[.0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0,
                      .0, .0, .0, .0, .0, .0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        use_out_scale=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_reg=dict(type='L1Loss', loss_weight=10.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    heat_head=dict(
        type='HeatmapMultitaskHead',
        stacked_heat_convs=1,
        stacked_offset_convs=1,
        in_channels=256,
        feat_channels=256,
        stride=[4],
        add_deconv=True,
        deconv_with_bias=False,
        deconv_num_layers=1,
        deconv_num_filters=[256],
        deconv_num_kernels=[4],
        separate_out_conv=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        with_offset=True,
        loss_offset=dict(type='L1Loss', loss_weight=10),
        use_heatmap=True,
        loss_heatmap='tradeoff_l2_loss',
        bg_weight=0.1,
        guassian_sigma=1,
        num_points=17),
    heat_branch_weight=0.05,
)
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxOksIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        assign_fg_per_anchor=True,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.001,
    #nms=dict(type='nms', iou_thr=0.5),
    nms=dict(type='oks_nms', iou_thr=0.5,
              sigmas=[.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoPoseDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
    dict(
        type='CenterRandomCropXiao',
        scale_factor=0.5,
        rot_factor=0,
        patch_width=640,
        patch_height=640),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # single-scale test
        #img_scale=(1333, 800),
        # multi-scale test
        img_scale=[(666, 400), (1000, 600), (1333, 800), (1666, 1000), (2000, 1200), (2333, 1400)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='Adam', lr=0.0004, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1.0 / 3,
    step=[95, 110])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
evaluation = dict(interval=5)
# runtime settings
total_epochs = 115
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = os.path.join(tmp_this_dir, '../../../../../output/TemplateShapeCNN/mmdetection/work_dirs/psa/psa_pose_final')
load_from = None
resume_from = None
workflow = [('train', 1)]