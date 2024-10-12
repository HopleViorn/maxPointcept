import numpy as np
from sklearn.cluster import HDBSCAN
import open3d as o3d
from glob import glob
import os
from knn import KNN
from tqdm import tqdm


COLOR_MAP = {
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

COLOR_MAP = list(COLOR_MAP.values())

def save_pcd(xyz, colors, normals, instance, save_path):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point['positions'] = o3d.core.Tensor(xyz, dtype=o3d.core.Dtype.Float32)
    pcd.point['colors'] = o3d.core.Tensor(colors, dtype=o3d.core.Dtype.Float32)
    pcd.point['normals'] = o3d.core.Tensor(normals, dtype=o3d.core.Dtype.Float32)
    pcd.point['instances'] = o3d.core.Tensor(instance.reshape(-1,1), dtype=o3d.core.Dtype.Int32)

    o3d.t.io.write_point_cloud(save_path, pcd)

if __name__=="__main__":
    res_dir = "exp/cylinders_normal/test/result"
    fns = sorted(glob(os.path.join(res_dir, "*.npy")))

    cnt = 1
    save_dir = "visualization/cluster_res"
    os.makedirs(save_dir, exist_ok=True)
    for i, fn in tqdm(enumerate(fns)):
        data = np.load(fn)
        coords, normals = data[:, :3], data[:, 3:]
        hdbscan = HDBSCAN(min_cluster_size=100, min_samples=5, cluster_selection_epsilon=0.02)
        hdbscan.fit(coords + normals)
        clabel = hdbscan.labels_
        if np.all(clabel == -1):
            clabel[:] = 0
        for _ in range(3):
            noise = coords[clabel == -1]
            if noise.size == 0:
                break
            others = coords[clabel >= 0]
            dists, idx = KNN.minibatch_nn(noise, others)
            _tmp = clabel[clabel >= 0][idx]
            _tmp[dists > 0.05] = -1
            clabel[clabel == -1] = _tmp
        # clabel[clabel >= 0] = clabel[clabel >= 0] + cnt
        # cnt += len(np.unique(clabel[clabel >= 0]))
        colors = np.zeros_like(coords)
        for j, ins in enumerate(np.unique(clabel)):
            mask = clabel == ins
            if ins == -1:
                colors[mask] = [0, 0, 0]
            colors[mask] = np.array(COLOR_MAP[j % 38])
        save_pcd(coords, colors / 255.0, normals, clabel, os.path.join(save_dir, f"res_{i:02d}.pcd"))



    

