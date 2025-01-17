_base_ = ["../_base_/default_runtime.py"]
# misc custom setting
batch_size = 12 * 2  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = False
enable_amp = True
num_worker = 12 * 4

# model settings
model = dict(
    type="Predictor_6_towards",
    backbone=dict(
        type="SpUNet-MLP",
        in_channels=3,
        num_classes=6,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# scheduler settings
epoch = 300
eval_epoch = 30 * 10
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type="PolyLR")

# dataset settings
dataset_type = "S3DISDataset"
data_root = "data/cylinders_normal"

    # # dict(type="CheckpointLoader"),
    # # dict(type="IterationTimer", warmup_iter=2),
    # # dict(type="InformationWriter"),
    # dict(type="PredictorEvaluator"),
    # # dict(type="CheckpointSaver", save_freq=None),
    # # dict(type="PreciseEvaluator", test_last=False),


hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="PredictorEvaluator_6_towards",
    ),
    dict(type="CheckpointSaver", save_freq=None),
]

test = dict(type="PredictorTester_6")

gd_size= 0.03

data = dict(
    train=dict(
        type=dataset_type,
        split=("Area_1", "Area_2", "Area_3", "Area_4","Area_5","Area_6"),
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=gd_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "normal", "color", "vector_attr_0"),
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=100000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord","normal","color", "vector_attr_0"),
                feat_keys=["color"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        # split=("test","Area_5"),
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=gd_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "normal", "color","vector_attr_0"),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord","normal","color", "vector_attr_0"),
                feat_keys=["color"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type='S3DISDataset',
        split='test',
        data_root='data/cylinders_normal',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='GridSample',
                grid_size=0.03,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'normal', 'color','vector_attr_0'),
                return_grid_coord=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'normal', 'color', "vector_attr_0"),
                feat_keys=['color'])
        ],
        test_mode=False
    ),
)

