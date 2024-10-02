import numpy as np
from typing import Optional
import os
import glob
from pathlib import Path
import open3d as o3d
from tqdm import tqdm

def save_multi_channel_pcd(xyz, rgb, pred, gt, save_path):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point['positions'] = o3d.core.Tensor(xyz, dtype=o3d.core.Dtype.Float32)
    pcd.point['colors'] = o3d.core.Tensor(rgb, dtype=o3d.core.Dtype.Float32)
    pcd.point['pred'] = o3d.core.Tensor(pred.reshape(-1, 1), dtype=o3d.core.Dtype.Int32)
    pcd.point['gt'] = o3d.core.Tensor(gt.reshape(-1, 1), dtype=o3d.core.Dtype.Int32)
    pcd.point['equals'] = o3d.core.Tensor((pred == gt).reshape(-1, 1), dtype=o3d.core.Dtype.Int32)
    o3d.t.io.write_point_cloud(save_path, pcd)

data_list = sorted([str(i) for i in Path("/data/lyq/factory_new/test").glob("patch_*") if i.is_dir()])
all_coords = np.zeros((0, 3))
all_gt = np.zeros((0))
all_pred = np.zeros((0))
all_colors = np.zeros((0, 3))

for patch in tqdm(data_list):
    idx = os.path.basename(patch).split("_")[1]
    # print(idx)
    coords = np.load(os.path.join(patch, "coord.npy"))
    colors = np.load(os.path.join(patch, "color.npy"))
    gt = np.load(os.path.join(patch, "segment.npy"))
    pred = np.load(f"exp/factory/factory-new-semseg-pt-v3m1-0-base/result_last/{idx}_pred.npy")

    all_coords = np.concatenate([all_coords, coords], axis=0)
    all_colors = np.concatenate([all_colors, colors], axis=0)
    all_gt = np.concatenate([all_gt, gt], axis=0)
    all_pred = np.concatenate([all_pred, pred], axis=0)

save_path = "/data/lyq/factory_new/vis/H2_val_last.pcd"
save_multi_channel_pcd(all_coords, all_colors, all_pred, all_gt, save_path)
