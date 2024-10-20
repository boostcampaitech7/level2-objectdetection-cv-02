# base_config.py 

dataset_type = 'CocoDataset'

data_root = 'data/coco/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# custom augmentation이 정의된 파일을 임포트
custom_imports = dict(
    imports=['mmdet.datasets.pipelines.custom_augmentation'],  # custom_augmentation.py의 경로
    allow_failed_imports=False
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # 공통된 augmentation
    dict(type='Resize', img_scale=[(1024, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='RandomBrightness', brightness_delta=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]