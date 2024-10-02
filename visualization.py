import numpy as np
import os
from pathlib import Path
import open3d as o3d
from tqdm import tqdm
import argparse

def save_multi_channel_pcd(xyz, rgb, pred, gt, save_path):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point['positions'] = o3d.core.Tensor(xyz, dtype=o3d.core.Dtype.Float32)
    pcd.point['colors'] = o3d.core.Tensor(rgb / 255.0 , dtype=o3d.core.Dtype.Float32)
    pcd.point['pred'] = o3d.core.Tensor(pred.reshape(-1, 1), dtype=o3d.core.Dtype.Int32)
    pcd.point['gt'] = o3d.core.Tensor(gt.reshape(-1, 1), dtype=o3d.core.Dtype.Int32)
    equals = np.zeros_like(gt, dtype=np.int32)

    cnt = 0
    print(f"cnt: gt -> pred")
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            cnt += 1
            mask = (gt == i) & (pred == j)
            equals[mask] = cnt
            print(f"{mask.sum()}: {i} -> {j}")
    pcd.point['equals'] = o3d.core.Tensor(equals.reshape(-1, 1), dtype=o3d.core.Dtype.Int32)
    o3d.t.io.write_point_cloud(save_path, pcd)


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--exp_name', type=str)
parser.add_argument('-d', '--dataset_root', type=str)
parser.add_argument('-s', '--split', type=str)
args = parser.parse_args()
data_root = Path(args.dataset_root) / args.split
exp_name = args.exp_name
split = args.split
data_list = sorted([str(i) for i in data_root.glob("patch_*") if i.is_dir()])
all_coords = []
all_gt = []
all_pred = []
all_colors = []
all_equals = []

for i, patch in tqdm(enumerate(data_list)):
    # print(patch)
    idx = os.path.basename(patch).split("_")[1]
    # print(idx)
    coords = np.load(os.path.join(patch, "coord.npy")).reshape([-1, 3])
    colors = np.load(os.path.join(patch, "color.npy")).reshape([-1, 3])
    gt = np.load(os.path.join(patch, "segment.npy")).reshape([-1, 1])
    # print(clusters.shape)
    pred = np.load(f"exp/factory/{exp_name}/result/{idx}_pred.npy").reshape([-1, 1])

    all_coords.append(coords)
    all_colors.append(colors)
    all_gt.append(gt)
    all_pred.append(pred)

all_coords = np.vstack(all_coords)
all_colors = np.vstack(all_colors)
all_gt = np.vstack(all_gt)
all_pred = np.vstack(all_pred)
save_path = f"visualization/{exp_name}_{split}.pcd"
save_multi_channel_pcd(all_coords, all_colors, all_pred, all_gt, save_path)
