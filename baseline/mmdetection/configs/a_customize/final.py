_base_ = [
          # './models/customize_example.py',
          '../swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py',
          'customize_datasets.py', 
          # 'customize_schedule.py', 
          'customize_runtime.py']

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step', # 어떤 scheduler 쓸건지 선택
    warmup='linear', # warmup을 할지 안할지
    warmup_iters=1000, # warmup interation 얼마나 줄건지
    warmup_ratio=0.001,
    step=[27, 33]) # step은 얼마마다 밟는지
# lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)

# swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py