from datasets import ShapeNet55Dataset
from easydict import EasyDict
import torch
from tqdm import tqdm as tq
import numpy as np
from utils import misc
import os
import shutil


def dataset_stat(rep_type, dl):
    """
    Compute dataset statistics for the specified representation type.

    Args:
        rep_type (str): Representation type, either "xyz", "7D" or "5D".

    Returns:
        None
    """
    if rep_type not in ["7D", "5D"]:
        raise ValueError("Invalid representation type. Use '7D' or '5D'.")

    # Compute representation
    representation = []
    for _, _, pc in tq(dl):
        if rep_type == "7D":
            representation.append(misc.point_xyz_to_7D(pc))
        elif rep_type == "5D":
            representation.append(misc.point_xyz_to_5D(pc))
        elif rep_type == "xyz":
            representation.append(pc)
        else:
            raise ValueError("Invalid representation type.")

    representation = torch.cat(representation, dim=0)

    # Print statistics
    print(representation.mean(dim=(0, 1)).tolist())
    print(representation.std(dim=(0, 1)).tolist())


def norm_helper(rep_type, ds):
    """
    Helper function to normalize point clouds and save the results.

    Args:
        rep_type (str): Representation type (e.g., "7D", "5D", "xyz").
        ds (Dataset): Dataset containing the point clouds.

    Returns:
        None
    """

    base_path = f"experiments/lab/{rep_type}"
    paths = {
        "full": f"{base_path}/pc_norm_full",
        "rotation": f"{base_path}/pc_norm_rotation",
        "radius": f"{base_path}/pc_norm_radius",
    }

    # Ensure directories are fresh
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    # Select random samples from the dataset
    ds_length = len(ds)
    ds_idxes = torch.randperm(ds_length)[:10]

    def save_normalized(
        pc,
        path,
        norm_func,
        inverse_norm_func,
        to_rep_func,
        from_rep_func,
        idx,
        normalize_mode,
    ):
        """Normalize and save point cloud data."""
        file_prefix = f"{path}/{idx:02d}"

        # Save original point cloud
        np.savetxt(f"{file_prefix}_org.xyz", pc.cpu().numpy())

        # Convert and normalize
        pc = pc.unsqueeze(0)
        pc_rep = to_rep_func(pc)
        pc_rep_normed = norm_func(pc_rep, normalize_mode=normalize_mode)
        pc_normed = from_rep_func(pc_rep_normed)

        # Save normalized point cloud
        np.savetxt(f"{file_prefix}_pc_normed.xyz", pc_normed[0].cpu().numpy())

        # Inverse normalization
        pc_rep_unnormed = inverse_norm_func(
            pc_rep_normed, normalize_mode=normalize_mode
        )
        pc_unnormed = from_rep_func(pc_rep_unnormed)

        # Save unnormalized point cloud
        np.savetxt(f"{file_prefix}_pc_unnormed.xyz", pc_unnormed[0].cpu().numpy())

    # Set functions based on representation type
    if rep_type == "7D":
        to_rep_func, from_rep_func = misc.point_xyz_to_7D, misc.point_7D_to_xyz
        norm_func, inverse_norm_func = misc.normalize_7D, misc.inverse_normalize_7D
    elif rep_type == "5D":
        to_rep_func, from_rep_func = misc.point_xyz_to_5D, misc.point_5D_to_xyz
        norm_func, inverse_norm_func = misc.normalize_5D, misc.inverse_normalize_5D
    elif rep_type == "xyz":
        to_rep_func, from_rep_func = lambda x: x, lambda x: x
        norm_func, inverse_norm_func = misc.normalize_xyz, misc.inverse_normalize_xyz
    else:
        raise ValueError("Unsupported representation type.")

    # Process and save for each normalization mode
    for idx in ds_idxes:
        pc = ds[idx][2]
        for normalize_mode in ["radius", "rotation", "full"]:
            if rep_type == "xyz" and normalize_mode != "full":
                continue
            save_normalized(
                pc,
                paths[normalize_mode],
                norm_func,
                inverse_norm_func,
                to_rep_func,
                from_rep_func,
                idx,
                normalize_mode,
            )


# def norm_helper(rep_type, ds):
#     save_path_radius = f"experiments/lab/{rep_type}/pc_norm_radius"
#     save_path_rotation = f"experiments/lab/{rep_type}/pc_norm_rotation"
#     save_path_full = f"experiments/lab/{rep_type}/pc_norm_full"

#     if os.path.exists(save_path_radius):
#         os.system(f"rm -r {save_path_radius}")
#     os.makedirs(save_path_radius)
#     if os.path.exists(save_path_rotation):
#         os.system(f"rm -r {save_path_rotation}")
#     os.makedirs(save_path_rotation)
#     if os.path.exists(save_path_full):
#         os.system(f"rm -r {save_path_full}")
#     os.makedirs(save_path_full)

#     ds_length = len(ds)
#     ds_idxes = torch.randperm(ds_length)[:10]

#     def save_normalized(
#         pc, path, norm_func, inverse_norm_func, to_rep_func, from_rep_func
#     ):
#         np.savetxt(f"{path}/{idx:02d}_org.xyz", pc)

#         pc = pc.unsqueeze(0)
#         pc_rep = to_rep_func(pc)

#         pc_rep_normed = norm_func(pc_rep)

#         pc_normed = from_rep_func(pc_rep_normed)
#         np.savetxt(f"{path}/{idx:02d}_pc_normed.xyz", pc_normed[0].numpy())

#         pc_rep_unnormed = inverse_norm_func(pc_rep_normed)

#         pc_unnormed = from_rep_func(pc_rep_unnormed)
#         np.savetxt(f"{path}/{idx:02d}_pc_unnormed.xyz", pc_unnormed[0].numpy())

#     if rep_type == "7D":
#         to_rep_func, from_rep_func = (
#             misc.point_xyz_to_7D,
#             misc.point_7D_to_xyz,
#         )
#         norm_func_radius, inverse_norm_func_radius = (
#             misc.normalize_7D_radius,
#             misc.inverse_normalize_7D_radius,
#         )
#         norm_func_rotation, inverse_norm_func_rotation = (
#             misc.normalize_7D_rotation,
#             misc.inverse_normalize_7D_rotation,
#         )
#         norm_func_full, inverse_norm_func_full = (
#             misc.normalize_7D,
#             misc.inverse_normalize_7D,
#         )
#     for idx in ds_idxes:
#         pc = ds[idx][2]

#         save_normalized(
#             pc,
#             save_path_radius,
#             norm_func_radius,
#             inverse_norm_func_radius,
#             to_rep_func,
#             from_rep_func,
#         )
#         save_normalized(
#             pc,
#             save_path_rotation,
#             norm_func_rotation,
#             inverse_norm_func_rotation,
#             to_rep_func,
#             from_rep_func,
#         )
#         save_normalized(
#             pc,
#             save_path_full,
#             norm_func_full,
#             inverse_norm_func_full,
#             to_rep_func,
#             from_rep_func,
#         )


if __name__ == "__main__":
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

    # dataset_stat("7D", shapenet_dl)
    # dataset_stat("5D", shapenet_dl)
    # dataset_stat("xyz", shapenet_dl)
    norm_helper("5D", shapenet_ds)
