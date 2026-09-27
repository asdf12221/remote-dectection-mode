import os

_DATA_ROOT = os.getenv('FPBA_DATA_ROOT', 'data/finaldatav5')
_WORK_ROOT = os.getenv('FPBA_WORK_ROOT', 'work_dirs')
_INTERNIMAGE_CKPT = os.getenv(
    'FPBA_INTERNIMAGE_CKPT', 'weights/internimage_l_22k_192to384.pth')
_DOTA_CKPT = os.getenv('FPBA_DOTA_CKPT') or None

custom_hooks = [
    dict(momentum=0.0001, priority=49, type='EMAHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet_custom.models.backbones.intern_image',
        'mmdet_custom.custom_layer_decay_optimizer_constructor',
    ])
data_root = _DATA_ROOT
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
image_scales = [
    (
        512,
        512,
    ),
    (
        640,
        640,
    ),
    (
        800,
        800,
    ),
]
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
launcher = 'pytorch'
load_from = _DOTA_CKPT
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 24
metainfo = dict(
    classes=(
        'HM',
        'LQS',
        'QHS',
        'MS',
        'A1_SU-35',
        'A2_C-130',
        'A3_C-17',
        'A4_C-5',
        'A5_F-16',
        'A6_TU-160',
        'A7_E-3',
        'A8_B-52',
        'A9_P-3C',
        'A10_B-1B',
        'A11_E-8',
        'A12_TU-22',
        'A13_F-15',
        'A14_KC-135',
        'A15_F-22',
        'A16_FA-18',
        'A17_TU-95',
        'A18_KC-10',
        'A19_SU-34',
        'A20_SU-24',
        'FSC',
    ))
model = dict(
    backbone=dict(
        channels=160,
        core_op='DCNv3',
        depths=[
            5,
            5,
            22,
            5,
        ],
        drop_path_rate=0.4,
        frozen_stages=2,
        groups=[
            10,
            20,
            40,
            80,
        ],
        init_cfg=dict(
            checkpoint=
            _INTERNIMAGE_CKPT,
            type='Pretrained'),
        layer_scale=1.0,
        mlp_ratio=4.0,
        norm_layer='LN',
        offset_scale=2.0,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        post_norm=True,
        type='InternImage',
        with_cp=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            160,
            320,
            640,
            1280,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=[
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                conv_out_channels=256,
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=10.0, type='GIoULoss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=True),
                norm_cfg=dict(requires_grad=True, type='SyncBN'),
                num_classes=25,
                num_shared_convs=4,
                num_shared_fcs=1,
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                roi_feat_size=7,
                type='ConvFCBBoxHead'),
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.05,
                        0.05,
                        0.1,
                        0.1,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                conv_out_channels=256,
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=10.0, type='GIoULoss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=True),
                norm_cfg=dict(requires_grad=True, type='SyncBN'),
                num_classes=25,
                num_shared_convs=4,
                num_shared_fcs=1,
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                roi_feat_size=7,
                type='ConvFCBBoxHead'),
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.033,
                        0.033,
                        0.067,
                        0.067,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                conv_out_channels=256,
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=10.0, type='GIoULoss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=True),
                norm_cfg=dict(requires_grad=True, type='SyncBN'),
                num_classes=25,
                num_shared_convs=4,
                num_shared_fcs=1,
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                roi_feat_size=7,
                type='ConvFCBBoxHead'),
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        num_stages=3,
        stage_loss_weights=[
            1,
            0.5,
            0.25,
        ],
        type='CascadeRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111, loss_weight=1.0, type='SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    ignore_iof_thr=0.5,
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=0.5,
                    match_low_quality=False,
                    min_pos_iou=0.6,
                    neg_iou_thr=0.6,
                    pos_iou_thr=0.6,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=0.5,
                    match_low_quality=False,
                    min_pos_iou=0.7,
                    neg_iou_thr=0.7,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
        ],
        rpn=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=0.5,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='CascadeRCNN')
num_classes = 25
optim_wrapper = dict(
    clip_grad=None,
    constructor='CustomLayerDecayOptimizerConstructor',
    loss_scale='dynamic',
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        depths=[
            5,
            5,
            22,
            5,
        ],
        layer_decay_rate=0.8,
        num_layers=37,
        offset_lr_scale=0.01),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            16,
            20,
        ],
        type='MultiStepLR'),
]
pretrained = _INTERNIMAGE_CKPT
resume = False
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        800,
        800,
    ), type='Resize'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(size_divisor=32, type='Pad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations_train.json',
        data_prefix=dict(img='images/all/'),
        data_root=_DATA_ROOT,
        filter_cfg=dict(filter_empty_gt=False),
        metainfo=dict(
            classes=(
                'HM',
                'LQS',
                'QHS',
                'MS',
                'A1_SU-35',
                'A2_C-130',
                'A3_C-17',
                'A4_C-5',
                'A5_F-16',
                'A6_TU-160',
                'A7_E-3',
                'A8_B-52',
                'A9_P-3C',
                'A10_B-1B',
                'A11_E-8',
                'A12_TU-22',
                'A13_F-15',
                'A14_KC-135',
                'A15_F-22',
                'A16_FA-18',
                'A17_TU-95',
                'A18_KC-10',
                'A19_SU-34',
                'A20_SU-24',
                'FSC',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                keep_ratio=True,
                scales=[
                    (
                        512,
                        512,
                    ),
                    (
                        640,
                        640,
                    ),
                    (
                        800,
                        800,
                    ),
                ],
                type='RandomChoiceResize'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        keep_ratio=True,
        scales=[
            (
                512,
                512,
            ),
            (
                640,
                640,
            ),
            (
                800,
                800,
            ),
        ],
        type='RandomChoiceResize'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(size_divisor=32, type='Pad'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations_val.json',
        data_prefix=dict(img='images/all/'),
        data_root=_DATA_ROOT,
        metainfo=dict(
            classes=(
                'HM',
                'LQS',
                'QHS',
                'MS',
                'A1_SU-35',
                'A2_C-130',
                'A3_C-17',
                'A4_C-5',
                'A5_F-16',
                'A6_TU-160',
                'A7_E-3',
                'A8_B-52',
                'A9_P-3C',
                'A10_B-1B',
                'A11_E-8',
                'A12_TU-22',
                'A13_F-15',
                'A14_KC-135',
                'A15_F-22',
                'A16_FA-18',
                'A17_TU-95',
                'A18_KC-10',
                'A19_SU-34',
                'A20_SU-24',
                'FSC',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                800,
            ), type='Resize'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=os.path.join(_DATA_ROOT, 'annotations_val.json'),
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_dataloader = val_dataloader
test_cfg = val_cfg
test_evaluator = val_evaluator
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = os.path.join(_WORK_ROOT, 'cascade_merged_25cls_v3_ce')
