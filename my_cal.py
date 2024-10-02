from pointcept.utils.misc import intersection_and_union
import numpy as np
from pathlib import Path
from tqdm import tqdm
NUM_CLASS = 5

names=[
    "others",
    "pipe",
    "piperack",
    "tank",
    "steel",
]

gt_path = sorted([i for i in Path("datasets/factory_data/test").glob("patch_*") if i.is_dir()])
record = {}
for idx, patch in enumerate(gt_path):
    # idx = patch.stem.split("_")[1]
    segment = np.load(patch / "segment.npy")
    pred = np.load(f"/home/lyq/Pointcept/exp/factory/my_test/pred_{idx:04d}.npy")
    # segment[np.where(segment==3)[0]] = 1
    # pred[np.where(pred==3)[0]] = 1
    intersection, union, target = intersection_and_union(pred, segment, NUM_CLASS, -1)
    record[patch.stem] = dict(
        intersection=intersection, union=union, target=target
    )

intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
iou_class = intersection / (union + 1e-10)
print(len(iou_class))
for i, name in enumerate(names):
    print(f"Class_{i} - {name} Result: iou {iou_class[i]:.4f}")