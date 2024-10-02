from pointcept.models import build_model
from pointcept.datasets import build_dataset, collate_fn
from collections import OrderedDict
import os
import numpy as np
import glob
from addict import Dict
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


# model settings
model_cfg = dict(
    type="DefaultSegmentorV2",
    num_classes=5,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D", "Factory"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)
model_cfg = Dict(model_cfg)
test_dict = dict(
    type="PCatDataset",
    split="test",
    data_root=None,
    transform=[
        dict(type="CenterShift", apply_z=True),
        dict(type="NormalizeColor"),
    ],
    test_mode=True,
    test_cfg=dict(
        voxelize=dict(
            type="GridSample",
            grid_size=0.001,
            hash_type="fnv",
            mode="test",
            keys=("coord", "color"),
            return_grid_coord=True,
        ),
        crop=None,
        post_transform=[
            # dict(type="Copy", keys_dict={"grid_size": 0.005}),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_size", "index"),
                feat_keys=("coord", "color"),
            ),
        ],
        aug_transform=[
            [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[0],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                )
            ],
            [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[1 / 2],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                )
            ],
            [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[1],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                )
            ],
            [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[3 / 2],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                )
            ],
            [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[0],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                ),
                dict(type="RandomScale", scale=[0.95, 0.95]),
            ],
            [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[1 / 2],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                ),
                dict(type="RandomScale", scale=[0.95, 0.95]),
            ],
            [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[1],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                ),
                dict(type="RandomScale", scale=[0.95, 0.95]),
            ],
            [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[3 / 2],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                ),
                dict(type="RandomScale", scale=[0.95, 0.95]),
            ],
            [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[0],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                ),
                dict(type="RandomScale", scale=[1.05, 1.05]),
            ],
            [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[1 / 2],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                ),
                dict(type="RandomScale", scale=[1.05, 1.05]),
            ],
            [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[1],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                ),
                dict(type="RandomScale", scale=[1.05, 1.05]),
            ],
            [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[3 / 2],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                ),
                dict(type="RandomScale", scale=[1.05, 1.05]),
            ],
            [dict(type="RandomFlip", p=1)],
        ],
    ),
)
test_dict = Dict(test_dict)


def collate_fn_t(batch):
    return batch


def SemSegTest(data, n_clusters=80, checkpoint=None):
    # data cluster
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    kmeans.fit(data[:, :3])
    patch_data = []
    _cluster = kmeans.labels_
    inverse = np.zeros(data.shape[0], dtype=np.int32)
    bs = 0
    for x in np.unique(_cluster):
        idx = np.argwhere(_cluster == x).reshape(-1)
        patch_data.append(data[idx])
        inverse[idx] = np.arange(bs, bs + idx.shape[0])
        bs += idx.shape[0]

    # load checkpoint
    model = build_model(model_cfg).cuda()
    checkpoint = torch.load(checkpoint)
    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        key = key[7:]
        weight[key] = value
    model.load_state_dict(weight, strict=True)

    # data_loader
    test_dict.data_root = patch_data
    test_dataset = build_dataset(test_dict)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        sampler=None,
        collate_fn=collate_fn_t,
    )

    # test
    model.eval()
    all_pred = []
    for idx, data_dict in enumerate(test_loader):
        print(f"Test: {idx:04d}")
        data_dict = data_dict[0]
        fragment_list = data_dict.pop("fragment_list")
        data_name = data_dict.pop("name")
        segment = data_dict.pop("segment")
        pred = torch.zeros((segment.size, model_cfg.num_classes)).cuda()
        for i in range(len(fragment_list)):
            fragment_batch_size = 1
            s_i, e_i = i * fragment_batch_size, min(
                (i + 1) * fragment_batch_size, len(fragment_list)
            )
            input_dict = collate_fn(fragment_list[s_i:e_i])
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            idx_part = input_dict["index"]
            with torch.no_grad():
                pred_part = model(input_dict)["seg_logits"]
                pred_part = F.softmax(pred_part, -1)
                bs = 0
                for be in input_dict["offset"]:
                    pred[idx_part[bs:be], :] += pred_part[bs:be]
                    bs = be
        pred = pred.max(1)[1].data.cpu().numpy()
        all_pred.append(pred)
    all_pred = np.hstack(all_pred)
    all_pred = all_pred[inverse]
    return all_pred


if __name__ == "__main__":
    all_data = []
    segment = []
    patch_name = sorted(glob.glob(os.path.join("datasets/factory_data/test", "*")))
    # print(patch_name)
    for patch in patch_name[:2]:
        coord = np.load(os.path.join(patch, "coord.npy"))
        color = np.load(os.path.join(patch, "color.npy"))
        seg = np.load(os.path.join(patch, "segment.npy"))
        all_data.append(np.hstack([coord, color]))
        segment.append(seg)
    all_data = np.vstack(all_data)
    segment = np.hstack(segment).reshape(-1)
    all_pred = SemSegTest(
        data=all_data,
        n_clusters=2,
        checkpoint="exp/factory/factory-new-semseg-pt-v3m1-0-base/model/model_best.pth",
    )
