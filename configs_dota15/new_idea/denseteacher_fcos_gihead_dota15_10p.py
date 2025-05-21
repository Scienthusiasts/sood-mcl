import torchvision.transforms as transforms
from copy import deepcopy


'''重要的参数写在前面:'''
# DOTA数据集版本(1.0 or 1.5)
version = 1.5
# 数据集路径
train_sup_image_dir =   f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/train_10per/{version}/labeled/images/'
train_sup_label_dir =   f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/train_10per/{version}/labeled/annfiles/'
train_unsup_image_dir = f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/train_10per/{version}/unlabeled/images/'
train_unsup_label_dir = f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/train_10per/{version}/unlabeled/empty_annfiles/'
# full:
# train_sup_image_dir = f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/train/images/'
# train_sup_label_dir = f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/train/{version}/annfiles/'
val_image_dir =         f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/val/images'
val_label_dir =         f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/val/{version}/annfiles'
test_image_dir =        f'/data/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/test/images'

angle_version = 'le90'
# 类别数
nc = 16
# 伪标签筛选超参
semi_loss = dict(type='RotatedDTBLGIHeadLoss', cls_channels=nc, loss_type='origin', bbox_loss_type='l1', 
                 # 'topk', 'top_dps', 'catwise_top_dps', 'global_w', 'sla'
                 p_selection = dict(mode='topk', k=0.03, beta=2.0), # 当mode=='top_dps'时, beta为S_pds的权重系数
                 # p_selection = dict(mode='sla', k=0.01, beta=1.0),
                 )

# 无监督分支权重
unsup_loss_weight = 1.0
# 是否使用高斯椭圆标签分配 (注意GA分配得搭配QualityFocalLoss)
bbox_head_type = 'SemiRotatedBLFCOSHead'
loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)

# 是否开启选择一致性自监督分支
use_ss_branch=False
ss_branch = None

# 是否开启refine head
use_refine_head=True
roi_head=dict(
    type='GIRoIHead', # ORCNNRoIHead GIRoIHead
    bbox_roi_extractor=dict(
        type='RotatedSingleRoIExtractor',
        roi_layer=dict(
            type='RoIAlignRotated',
            out_size=7,
            sample_num=2,
            clockwise=True),
        out_channels=256,
        featmap_strides=[8, 16, 32, 64, 128]),
    bbox_coder=dict(
        type='DeltaXYWHAOBBoxCoder',
        angle_range=angle_version,
        norm_factor=None,
        edge_swap=True,
        proj_xy=True,
        target_means=(.0, .0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
    nc=nc,
    add_noise_p=0.5,
    # 'share_head' 'avg_pool' 'share_fchead'
    roi_pooling = 'avg_pool', 
    assigner='HungarianWithIoUMatching',
)


burn_in_steps = 6400
# 是否导入权重
load_from = 'log/new/denseteacher/gihead/burn-in-6400_top0.03_O2M-only-boxloss_refine-allloss_avgpoolroi/latest.pth'
# load_from = None








# model settings
detector = dict(
    type='SemiRotatedBLRefineFCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type=bbox_head_type,
        num_classes=nc,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=loss_cls,
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # 这部分充当去噪微调模块:
    # (roi_head, train_cfg, test_cfg): reference: /data/yht/code/sood-mcl/mmrotate-0.3.4/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py
    roi_head=roi_head,
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1, class_agnostic=True), # 阈值越小越苛刻
            max_per_img=2000),
        # 原本就有的:
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000,
    )
)

model = dict(
    type="RotatedDTBaselineGISS",
    model=detector,
    nc=nc,
    # 核心部分:
    use_ss_branch=use_ss_branch,
    ss_branch=ss_branch,
    use_refine_head=use_refine_head,

    semi_loss=semi_loss,
    train_cfg=dict(
        iter_count=0,
        burn_in_steps=burn_in_steps,
        sup_weight=1.0,
        unsup_weight=unsup_loss_weight,
        weight_suppress="linear",
        logit_specific_weights=dict(),
    ),
    test_cfg=dict(inference_on="teacher"),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
common_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                    'img_norm_cfg', 'tag')
         )
]
strong_pipeline = [
    dict(type='DTToPILImage'),
    dict(type='DTRandomApply', operations=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    dict(type='DTRandomGrayscale', p=0.2),
    dict(type='DTRandomApply', operations=[
        dict(type='DTGaussianBlur', rad_range=[0.1, 2.0])
    ]),
    # dict(type='DTRandCrop'),
    dict(type='DTToNumpy'),
    dict(type="ExtraAttrs", tag="unsup_strong"),
]
weak_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type="ExtraAttrs", tag="unsup_weak"),
]
unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    dict(type="LoadEmptyAnnotations", with_bbox=True),
    dict(type="STMultiBranch", unsup_strong=deepcopy(strong_pipeline), unsup_weak=deepcopy(weak_pipeline),
         common_pipeline=common_pipeline, is_seq=True), 
]
sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type="ExtraAttrs", tag="sup_weak"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'img_norm_cfg', 'tag')
         )
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

dataset_type = 'DOTADataset'  
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane')
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=5,
    train=dict(
        type="SemiDataset",
        sup=dict(
            type=dataset_type,
            ann_file=train_sup_label_dir,
            img_prefix=train_sup_image_dir,
            classes=classes,
            pipeline=sup_pipeline,
        ),
        unsup=dict(
            type=dataset_type,
            ann_file=train_unsup_label_dir,
            img_prefix=train_unsup_image_dir,
            classes=classes,
            pipeline=unsup_pipeline,
            filter_empty_gt=False,
        ),
    ),
    val=dict(
        type=dataset_type,
        img_prefix=val_image_dir,
        ann_file=val_label_dir,
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        img_prefix=val_image_dir,
        ann_file=val_label_dir,
        classes=classes,
        pipeline=test_pipeline,
    ),
    sampler=dict(
        train=dict(
            type="MultiSourceSampler",
            sample_ratio=[2, 1],
            seed=42
        )
    ),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.9996, interval=1, start_steps=3200),
]

# evaluation
evaluation = dict(type="SubModulesDistEvalHook", interval=3200, metric='mAP',
                  save_best='mAP')
# 单卡调试时推理报分布式的错，是BN的问题，在配置文件里加一个broadcast_这个参数
# evaluation = dict(type="SubModulesDistEvalHook", interval=3200, metric='mAP',
#                   save_best='mAP', broadcast_bn_buffer=False)


# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=120000)
# 120k iters is enough for DOTA
runner = dict(type="IterBasedRunner", max_iters=120000)
checkpoint_config = dict(by_epoch=False, interval=3200, max_keep_ckpts=50)

# Default: disable fp16 training
# fp16 = dict(loss_scale="dynamic")

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook'),
        # dict(
        #     type="WandbLoggerHook",
        #     init_kwargs=dict(
        #         project="rotated_DenseTeacher_10percent",
        #         name="default_bce4cls",
        #     ),
        #     by_epoch=False,
        # ),
    ],
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]   # mode, iters

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'