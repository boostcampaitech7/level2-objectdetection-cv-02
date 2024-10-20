# base_config.py 

dataset_type = 'CocoDataset'

data_root = 'data/coco/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True)

# custom augmentation이 정의된 파일을 임포트
custom_imports = dict(
    imports=['mmdet.datasets.pipelines.custom_augmentation'],  # custom_augmentation.py의 경로
    allow_failed_imports=False
)

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     # 공통된 augmentation
#     dict(type='Resize', img_scale=[(1024, 1024)], keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='RandomBrightness', brightness_delta=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]

train_pipeline = [
    dict(type='LoadImageFromFile'),  # 이미지 로딩
    dict(type='LoadAnnotations', with_bbox=True),  # BBox 로딩
    # MultiImageMixDataset에서 Mosaic과 MixUp을 적용
    dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            pipeline=[
                dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
            ]
        ),
        pipeline=[
            dict(type='Mosaic', img_scale=(1024, 1024), pad_val=114.0),
            dict(type='MixUp', img_scale=(1024, 1024), ratio_range=(0.8, 1.6)),
            dict(type='RandomAffine', scaling_ratio_range=(0.5, 1.5), border=(-512, -512)),  # Affine 변환 적용
        ]
    ),
    dict(type='RandomBrightness', brightness_delta=32),  # 이후 밝기 조정만 별도로 추가
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]