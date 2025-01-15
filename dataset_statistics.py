from datasets import ShapeNet55Dataset
from easydict import EasyDict
import torch
from tqdm import tqdm as tq
import numpy as np
from utils import misc
import os


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

    shapenet_dl = torch.utils.data.DataLoader(
        shapenet_ds, batch_size=256, num_workers=16, shuffle=False
    )

    rep_7D = []
    for _, _, pc in tq(shapenet_dl):
        rep_7D.append(misc.point_xyz_to_7D(pc))

    rep_7D = torch.cat(rep_7D, dim=0)
    print(rep_7D.mean(dim=(0, 1)))
    print(rep_7D.std(dim=(0, 1)))
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

    save_path_radius = "experiments/lab/pc_norm_radius"
    save_path_rotation = "experiments/lab/pc_norm_rotation"
    save_path_full = "experiments/lab/pc_norm_full"

    if os.path.exists(save_path_radius):
        os.system(f"rm -r {save_path_radius}")
    os.makedirs(save_path_radius)
    if os.path.exists(save_path_rotation):
        os.system(f"rm -r {save_path_rotation}")
    os.makedirs(save_path_rotation)
    if os.path.exists(save_path_full):
        os.system(f"rm -r {save_path_full}")
    os.makedirs(save_path_full)

    ds_length = len(shapenet_ds)
    ds_idxes = torch.randperm(ds_length)[:10]

    def save_normalized(pc, path, norm_func, inverse_norm_func):
        np.savetxt(f"{path}/{idx:02d}_org.xyz", pc)

        pc = pc.unsqueeze(0)
        pc_7d = misc.point_xyz_to_7D(pc)

        pc_7d_normed = norm_func(pc_7d)

        pc_normed = misc.point_7D_to_xyz(pc_7d_normed)
        np.savetxt(f"{path}/{idx:02d}_pc_normed.xyz", pc_normed[0].numpy())

        pc_7d_unnormed = inverse_norm_func(pc_7d_normed)

        pc_unnormed = misc.point_7D_to_xyz(pc_7d_unnormed)
        np.savetxt(f"{path}/{idx:02d}_pc_unnormed.xyz", pc_unnormed[0].numpy())

    for idx in ds_idxes:
        pc = shapenet_ds[idx][2]

        save_normalized(
            pc,
            save_path_radius,
            misc.normalize_7D_radius,
            misc.inverse_normalize_7D_radius,
        )
        save_normalized(
            pc,
            save_path_rotation,
            misc.normalize_7D_rotation,
            misc.inverse_normalize_7D_rotation,
        )
        save_normalized(
            pc, save_path_full, misc.normalize_7D, misc.inverse_normalize_7D
        )


if __name__ == "__main__":
    # dataset_radius_stat()
    norm_helper()
