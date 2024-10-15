# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001) # lr 조정
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step', # 어떤 scheduler 쓸건지 선택
    warmup='linear', # warmup을 할지 안할지
    warmup_iters=500, # warmup interation 얼마나 줄건지
    warmup_ratio=0.001,
    step=[8, 11]) # step은 얼마마다 밟는지
runner = dict(type='EpochBasedRunner', max_epochs=12) #max epochs값 조절