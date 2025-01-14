from datasets import ShapeNet55Dataset
from easydict import EasyDict
import torch
from tqdm import tqdm as tq
import numpy as np
from utils import misc


def dataset_radius_stat():
    shapenet_ds = ShapeNet55Dataset.ShapeNet(
        config=EasyDict(
            {
                "DATA_PATH": "/export/datasets/public/Point_Cloud_Processing/data/ShapeNet55-34/ShapeNet-55",
                "PC_PATH": "/export/datasets/public/Point_Cloud_Processing/data/ShapeNet55-34/shapenet_pc",
                "subset": "train",
                "N_POINTS": 8192,
                "npoints": 1024,
            }
        )
    )
    shapenet_ds.sampling_method = "fps"

    radiuses = []
    for item in tq(shapenet_ds):
        pc = item[2]
        radiuses.append(torch.norm(pc, dim=1, keepdim=True))

    radiuses = torch.cat(radiuses, dim=0)
    print(radiuses.mean(dim=(0)))
    print(radiuses.std(dim=(0)))
    # std_xyz = torch.tensor([0.1780, 0.1526, 0.1679])
    # mean_radius = torch.tensor([0.5302])
    # std_radius = torch.tensor([0.2267])


def norm_helper():
    shapenet_ds = ShapeNet55Dataset.ShapeNet(
        config=EasyDict(
            {
                "DATA_PATH": "/export/datasets/public/Point_Cloud_Processing/data/ShapeNet55-34/ShapeNet-55",
                "PC_PATH": "/export/datasets/public/Point_Cloud_Processing/data/ShapeNet55-34/shapenet_pc",
                "subset": "train",
                "N_POINTS": 8192,
                "npoints": 1024,
            }
        )
    )
    shapenet_ds.sampling_method = "fps"
    
    save_path = "experiments/lab/pc_norm"    
    ds_length = len(shapenet_ds)
    ds_idxes = torch.randperm(ds_length)[:10]
    for idx in ds_idxes:
        pc = shapenet_ds[idx][2]
        np.savetxt(f"{save_path}/{idx:02d}_org.xyz", pc)

        pc = pc.unsqueeze(0)
        pc_7d = misc.point_xyz_to_7D(pc)

        mean_radius = torch.tensor([[[0.5302, 0, 0, 0, 0, 0, 0]]])
        std_radius = torch.tensor([[[0.2267, 1, 1, 1, 1, 1, 1]]])

        pc_7d_normed = (pc_7d - mean_radius) / std_radius

        pc_normed = misc.point_7D_to_xyz(pc_7d_normed)
        np.savetxt(f"{save_path}/{idx:02d}_pc_normed.xyz", pc_normed[0].numpy())

        pc_7d_unnormed = pc_7d_normed * std_radius + mean_radius

        pc_unnormed = misc.point_7D_to_xyz(pc_7d_unnormed)
        np.savetxt(f"{save_path}/{idx:02d}_pc_unnormed.xyz", pc_unnormed[0].numpy())


if __name__ == "__main__":
    # dataset_radius_stat()
    norm_helper()
