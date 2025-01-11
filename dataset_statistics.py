from datasets import ShapeNet55Dataset
from easydict import EasyDict
import torch
from tqdm import tqdm as tq

if __name__ == "__main__":
    shapenet_ds = ShapeNet55Dataset.ShapeNet(
        config=EasyDict(
            {
                "DATA_PATH": "/export/datasets/public/Point_Cloud_Processing/data/ShapeNet55-34/ShapeNet-55",
                "PC_PATH": "/export/datasets/public/Point_Cloud_Processing/data/ShapeNet55-34/shapenet_pc",
                "subset": "train",
                "N_POINTS": 8192,
                "npoints": 1024
            }
        )
    )
    shapenet_ds.sampling_method = "fps"
    shapenet_ds.stat_norm = False

    all_radii = []
    for i in tq(range(len(shapenet_ds))):
        point = shapenet_ds[i][2]
        radius = torch.linalg.norm(point, axis=1)

        all_radii.append(radius)

    all_radii = torch.cat(all_radii, dim=0)
    print(all_radii.mean())
    print(all_radii.std())
