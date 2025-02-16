import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import abc

from pytorch3d.transforms.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

# from pointnet2_ops import pointnet2_utils
from pytorch3d.ops import sample_farthest_points

from easydict import EasyDict


def add_args_to_easydic_as_object(easydic_obj, args, args_key="args"):
    """
    Add all arguments from args to the EasyDict object as a nested object.

    Args:
        easydic_obj (EasyDict): The EasyDict object to update.
        args (Namespace): The arguments from argparse.
        args_key (str): The key under which to store the arguments.

    Returns:
        None
    """
    easydic_obj[args_key] = EasyDict(vars(args))
    return easydic_obj


def fps(data, number):
    """
    data B N 3
    number int
    """
    # fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    # fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()

    fps_data, _ = sample_farthest_points(points=data, K=number)

    return fps_data


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_lambda_sche(opti, config):
    if config.get("decay_step") is not None:
        lr_lbmd = lambda e: max(
            config.lr_decay ** (e / config.decay_step), config.lowest_decay
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd)
    else:
        raise NotImplementedError()
    return scheduler


def build_lambda_bnsche(model, config):
    if config.get("decay_step") is not None:
        bnm_lmbd = lambda e: max(
            config.bn_momentum * config.bn_decay ** (e / config.decay_step),
            config.lowest_decay,
        )
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd)
    else:
        raise NotImplementedError()
    return bnm_scheduler


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model).__name__)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)


def seprate_point_cloud(xyz, num_points, crop, fixed_points=None, padding_zeros=False):
    """
    seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    """
    _, n, c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None

    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop, list):
            num_crop = random.randint(crop[0], crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:
            center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1).cuda()
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3).cuda()

        distance_matrix = torch.norm(
            center.unsqueeze(2) - points.unsqueeze(1), p=2, dim=-1
        )  # 1 1 2048

        idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0, 0]  # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0)  # 1 N 3

        crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop, list):
            INPUT.append(fps(input_data, 2048))
            CROP.append(fps(crop_data, 2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT, dim=0)  # B N 3
    crop_data = torch.cat(CROP, dim=0)  # B M 3

    return input_data.contiguous(), crop_data.contiguous()


def get_ptcloud_img(ptcloud, roll, pitch):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    ax = fig.gca(projection=Axes3D.name, adjustable="box")
    ax.axis("off")
    # ax.axis('scaled')
    ax.view_init(roll, pitch)
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir="z", c=y, cmap="jet")

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def visualize_KITTI(
    path,
    data_list,
    titles=["input", "pred"],
    cmap=["bwr", "autumn"],
    zdir="y",
    xlim=(-1, 1),
    ylim=(-1, 1),
    zlim=(-1, 1),
):
    fig = plt.figure(figsize=(6 * len(data_list), 6))
    cmax = data_list[-1][:, 0].max()

    for i in range(len(data_list)):
        data = data_list[i][:-2048] if i == 1 else data_list[i]
        color = data[:, 0] / cmax
        ax = fig.add_subplot(1, len(data_list), i + 1, projection="3d")
        ax.view_init(30, -120)
        b = ax.scatter(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            zdir=zdir,
            c=color,
            vmin=-1,
            vmax=1,
            cmap=cmap[0],
            s=4,
            linewidth=0.05,
            edgecolors="black",
        )
        ax.set_title(titles[i])

        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0)
    if not os.path.exists(path):
        os.makedirs(path)

    pic_path = path + ".png"
    fig.savefig(pic_path)

    np.save(os.path.join(path, "input.npy"), data_list[0].numpy())
    np.save(os.path.join(path, "pred.npy"), data_list[1].numpy())
    plt.close(fig)


def random_dropping(pc, e):
    up_num = max(64, 768 // (e // 50 + 1))
    pc = pc
    random_num = torch.randint(1, up_num, (1, 1))[0, 0]
    pc = fps(pc, random_num)
    padding = torch.zeros(pc.size(0), 2048 - pc.size(1), 3).to(pc.device)
    pc = torch.cat([pc, padding], dim=1)
    return pc


def random_scale(partial, scale_range=[0.8, 1.2]):
    scale = torch.rand(1).cuda() * (scale_range[1] - scale_range[0]) + scale_range[0]
    return partial * scale


def radius_normalization(pc, mean=0.5302, std=0.2267):
    # Compute radius of the vectors
    # centroid = torch.mean(pc, dim=1, keepdim=True)
    # pc = pc - centroid
    radius = torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True))  # Shape: (..., 1)

    radius_mean = radius.mean(dim=-1)
    radius_std = radius.std(dim=-1)

    # Normalize the radius
    normalized_radius = (radius - mean) / std  # Shape: (N, 1)

    # Adjust points to reflect normalized radius
    normalized_points = pc * (normalized_radius / (radius + 1e-6)) - torch.tensor(
        [[[mean, mean, mean]]], device=pc.device
    )

    return normalized_points


def inverse_radius_normalization(normalized_points, mean=0.5302, std=0.2267):
    # Compute the normalized radius of the vectors
    normalized_radius = torch.sqrt(
        torch.sum(normalized_points**2, dim=-1, keepdim=True)
    )  # Shape: (N, 1)

    # Reverse the normalization of the radius
    original_radius = normalized_radius * std + mean  # Shape: (N, 1)

    # Adjust points to reflect the original radius
    original_points = normalized_points * (
        original_radius / normalized_radius
    )  # Broadcasting happens correctly

    return original_points


def pc_scale(pc, range=(-1, 1)):
    """pc: NxC, return NxC"""
    # assert len(pc.shape) == 3, "The shape of the point cloud should be 3."
    range_center = (range[1] + range[0]) / 2
    range_radius = (range[1] - range[0]) / 2
    centroid = torch.mean(pc, dim=-2, keepdim=True)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1)), dim=-1, keepdim=True)[0]
    pc = ((pc / m) * range_radius) + range_center
    return pc


def pc_to_tensor(pc):
    """pc: NxC, return NxC"""
    pc = torch.from_numpy(pc).float()
    return pc


def normalize_xyz(
    pc: torch.tensor,
    mean: list = [0, 0, 0],
    std: list = [0.356, 0.3052, 0.3358],
    normalize_mode="full",
):
    assert pc.shape[-1] == 3, "The last dimension of the point cloud should be 3."
    if normalize_mode == "none":
        return pc
    elif normalize_mode == "full":
        mean = torch.tensor(mean, device=pc.device).expand_as(pc)
        std = torch.tensor(std, device=pc.device).expand_as(pc)
        return (pc - mean) / std
    else:
        raise ValueError("Invalid normalization mode.")


def inverse_normalize_xyz(
    pc: torch.tensor,
    mean: list = [0, 0, 0],
    std: list = [0.356, 0.3052, 0.3358],
    normalize_mode="full",
):
    assert pc.shape[-1] == 3, "The last dimension of the point cloud should be 3."
    if normalize_mode == "none":
        return pc
    elif normalize_mode == "full":
        mean = torch.tensor(mean, device=pc.device).expand_as(pc)
        std = torch.tensor(std, device=pc.device).expand_as(pc)
        return pc * std + mean
    else:
        raise ValueError("Invalid normalization mode.")


# code is copied from MAE
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: Output dimension for each position (must be even).
    pos: A tensor of positions to be encoded, shape (B, N).
    Returns:
    A tensor of positional embeddings with shape (B, N, embed_dim).
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."

    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)  # (D/2,)

    out = pos.unsqueeze(-1) * omega  # (B, N, D/2)

    emb_sin = torch.sin(out)  # (B, N, D/2)
    emb_cos = torch.cos(out)  # (B, N, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (B, N, D)
    return emb


def get_3d_sincos_pos_embed(embed_dim, points):
    """
    embed_dim: Output embedding dimension (must be divisible by 3).
    points: A tensor of 3D positions, shape (B, N, 3).
    Returns:
    A tensor of positional embeddings with shape (B, N, embed_dim).
    """
    assert embed_dim % 3 == 0, "Embedding dimension must be divisible by 3."
    assert points.shape[-1] == 3, "Input points must have shape (B, N, 3)."

    # Split the embedding dimension equally among X, Y, and Z
    dim_per_axis = embed_dim // 3

    # Extract X, Y, Z coordinates
    x_embed = get_1d_sincos_pos_embed_from_grid(
        dim_per_axis, points[..., 0]
    )  # (B, N, dim_per_axis)
    y_embed = get_1d_sincos_pos_embed_from_grid(
        dim_per_axis, points[..., 1]
    )  # (B, N, dim_per_axis)
    z_embed = get_1d_sincos_pos_embed_from_grid(
        dim_per_axis, points[..., 2]
    )  # (B, N, dim_per_axis)

    # Concatenate embeddings along the last dimension
    pos_embed = torch.cat([x_embed, y_embed, z_embed], dim=-1)  # (B, N, embed_dim)
    return pos_embed


def point_xyz_to_spherical(pc):
    """
    Convert a batch of 3D points to spherical coordinates (radius, azimuth, spherical angle).

    Args:
        pc (torch.Tensor): Tensor of shape (B, N, 3) with (x, y, z) coordinates.

    Returns:
        torch.Tensor: Tensor of shape (B, N, 3) with (radius, azimuth, spherical angle).
    """
    # Extract x, y, z coordinates
    x = pc[..., 0]
    y = pc[..., 1]
    z = pc[..., 2]

    # Compute radius
    radius = torch.sqrt(torch.sum(pc**2, dim=-1) + 1e-7)

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    safe_radius = torch.clamp(radius, min=epsilon)

    # Compute azimuth (theta): atan2(y, x)
    azimuth = torch.atan2(y, x)

    # Compute spherical angle (phi): acos(z / r)
    spherical_angle = torch.acos(torch.clamp(z / safe_radius, min=-1.0, max=1.0))

    # Handle points with zero radius (origin)
    spherical_angle[radius == 0] = 0.0  # Set phi to 0 for origin
    azimuth[radius == 0] = 0.0  # Set theta to 0 for origin

    # Combine into a single tensor
    spherical_coordinates = torch.stack((radius, azimuth, spherical_angle), dim=-1)

    return spherical_coordinates


def point_spherical_to_xyz(spherical_coordinates):
    """
    Convert spherical coordinates to XYZ coordinates.

    Args:
        spherical_coordinates (torch.Tensor): Tensor of shape (B, N, 3) with radius, azimuth, and spherical angle.

    Returns:
        torch.Tensor: Tensor of shape (B, N, 3) with reconstructed XYZ coordinates.
    """
    # Extract radius, azimuth, and spherical angle
    radius = spherical_coordinates[..., 0]
    azimuth = spherical_coordinates[..., 1]
    spherical_angle = spherical_coordinates[..., 2]

    # Compute direction vector from radius, azimuth, and spherical angle
    x = radius * torch.sin(spherical_angle) * torch.cos(azimuth)
    y = radius * torch.sin(spherical_angle) * torch.sin(azimuth)
    z = radius * torch.cos(spherical_angle)

    # Combine into a single tensor
    xyz = torch.stack((x, y, z), dim=-1)

    return xyz


def normalize_spherical(
    pc,
    normalize_mode: str,
    mean: list = [0.5302362442016602, 0, 0],
    std: list = [0.2266872227191925, 1, 1],
):
    """
    Normalize a 5D point cloud based on the specified mode.

    Args:
        pc (torch.Tensor): Input point cloud.
        normalize_mode (str): Mode of normalization ("none", "radius", "rotation", "full").
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        torch.Tensor: Normalized point cloud.
    """
    if normalize_mode == "none":
        return pc
    elif normalize_mode == "radius":
        mean_tensor = torch.tensor([mean], device=pc.device)
        std_tensor = torch.tensor([std], device=pc.device)
        return (pc - mean_tensor) / std_tensor
    else:
        raise ValueError("Invalid normalization mode.")


def inverse_normalize_spherical(
    pc,
    normalize_mode: str,
    mean: list = [0.5302362442016602, 0, 0],
    std: list = [0.2266872227191925, 1, 1],
):
    """
    Inverse normalize a 5D point cloud based on the specified mode.

    Args:
        pc (torch.Tensor): Input normalized point cloud.
        normalize_mode (str): Mode of normalization ("none", "radius", "rotation", "full").
        mean (list): Mean values for inverse normalization.
        std (list): Standard deviation values for inverse normalization.

    Returns:
        torch.Tensor: Inverse normalized point cloud.
    """
    if normalize_mode == "none":
        return pc
    elif normalize_mode == "radius":
        mean_tensor = torch.tensor([mean], device=pc.device)
        std_tensor = torch.tensor([std], device=pc.device)
        return pc * std_tensor + mean_tensor
    else:
        raise ValueError("Invalid normalization mode.")


def chamfer_loss_spherical(source, target):
    """
    Compute the Chamfer Loss for spherical coordinates (radius, azimuth, spherical angle).

    Args:
        source (torch.Tensor): Source spherical coordinates of shape (B, N, 3) with [r, theta, phi].
        target (torch.Tensor): Target spherical coordinates of shape (B, M, 3) with [r, theta, phi].

    Returns:
        torch.Tensor: Scalar tensor representing the Chamfer Loss.
    """
    # Ensure inputs are valid
    assert source.shape[-1] == 3, (
        "Source points should be in spherical space [r, theta, phi]."
    )
    assert target.shape[-1] == 3, (
        "Target points should be in spherical space [r, theta, phi]."
    )

    # Extract components
    r_s, theta_s, phi_s = source[..., 0], source[..., 1], source[..., 2]
    r_t, theta_t, phi_t = target[..., 0], target[..., 1], target[..., 2]

    # Compute pairwise distances for radius
    radius_dist = (r_s.unsqueeze(-1) - r_t.unsqueeze(-2)) ** 2  # Shape: (B, N, M)

    # Compute angular distance for azimuth (theta)
    azimuth_diff = torch.atan2(
        torch.sin(theta_s.unsqueeze(-1) - theta_t.unsqueeze(-2)),
        torch.cos(theta_s.unsqueeze(-1) - theta_t.unsqueeze(-2)),
    )  # Shape: (B, N, M)
    azimuth_dist = azimuth_diff**2

    # Compute angular distance for spherical angle (phi)
    spherical_diff = torch.atan2(
        torch.sin(phi_s.unsqueeze(-1) - phi_t.unsqueeze(-2)),
        torch.cos(phi_s.unsqueeze(-1) - phi_t.unsqueeze(-2)),
    )  # Shape: (B, N, M)
    spherical_dist = spherical_diff**2

    # Total pairwise distance
    total_dist = radius_dist + azimuth_dist + spherical_dist  # (B, N, M)

    # Compute Chamfer Loss
    src_to_tgt_dist = torch.min(total_dist, dim=-1).values  # (B, N)
    tgt_to_src_dist = torch.min(total_dist, dim=-2).values  # (B, M)

    loss = src_to_tgt_dist.mean() + tgt_to_src_dist.mean()
    return loss


def spherical_to_continues(spherical_coordinates):
    """
    Convert spherical coordinates to a continues 5D representation.

    Args:
        spherical_coordinates (torch.Tensor): Tensor of shape (B, N, 3) with radius, azimuth, and spherical angle.

    Returns:
        torch.Tensor: Tensor of shape (B, N, 5) with radius, sin(azimuth), cos(azimuth), sin(spherical angle), cos(spherical angle).
    """
    # Extract radius, azimuth, and spherical angle
    radius = spherical_coordinates[..., 0]
    azimuth = spherical_coordinates[..., 1]
    spherical_angle = spherical_coordinates[..., 2]

    # Compute the sin and cos of the azimuth and spherical angle
    sin_azimuth = torch.sin(azimuth)
    cos_azimuth = torch.cos(azimuth)
    sin_spherical_angle = torch.sin(spherical_angle)
    cos_spherical_angle = torch.cos(spherical_angle)

    # Combine into a single tensor
    spherical_continues = torch.stack(
        (radius, sin_azimuth, cos_azimuth, sin_spherical_angle, cos_spherical_angle),
        dim=-1,
    )

    return spherical_continues


def continues_to_spherical(spherical_continues):
    """
    Convert a continuous 5D representation back to spherical coordinates.

    Args:
        spherical_continues (torch.Tensor): Tensor of shape (B, N, 5) with
                                            [radius, sin(azimuth), cos(azimuth), sin(spherical), cos(spherical)].

    Returns:
        torch.Tensor: Tensor of shape (B, N, 3) with [radius, azimuth, spherical angle].
    """
    # Extract components
    radius = spherical_continues[..., 0]  # Radius (B, N)
    sin_azimuth = spherical_continues[..., 1]
    cos_azimuth = spherical_continues[..., 2]
    sin_spherical = spherical_continues[..., 3]
    cos_spherical = spherical_continues[..., 4]

    # Compute azimuth (theta) using atan2
    azimuth = torch.atan2(sin_azimuth, cos_azimuth)  # (B, N)

    # Compute spherical angle (phi) using atan2
    spherical_angle = torch.atan2(sin_spherical, cos_spherical)  # (B, N)

    # Combine into spherical coordinates
    spherical_coordinates = torch.stack(
        (radius, azimuth, spherical_angle), dim=-1
    )  # (B, N, 3)

    return spherical_coordinates


def point_xyz_to_5D(pc):
    """
    Convert a 3D point cloud to a 5D representation (radius, sin(azimuth), cos(azimuth), sin(spherical), cos(spherical)).

    Args:
        pc (torch.Tensor): Tensor of shape (B, N, 3) with (x, y, z) coordinates.

    Returns:
        torch.Tensor: Tensor of shape (B, N, 5) with 5D representation.
    """
    assert pc.shape[-1] == 3, "The last dimension of the point cloud should be 3."
    assert len(pc.shape) == 3, (
        "The input point cloud should have 3 dimensions (B, N, 3)."
    )

    spherical_coordinates = point_xyz_to_spherical(pc)
    spherical_continues = spherical_to_continues(spherical_coordinates)

    return spherical_continues


def point_5D_to_xyz(pc_5d):
    """
    Convert a 5D point cloud representation back to 3D XYZ coordinates.

    Args:
        pc_5d (torch.Tensor): Tensor of shape (B, N, 5) containing radius and 4D representation.

    Returns:
        torch.Tensor: Tensor of shape (B, N, 3) with reconstructed XYZ coordinates.
    """
    assert pc_5d.shape[-1] == 5, "Expected input shape (B, N, 5)."

    spherical_continues = pc_5d
    spherical_coordinates = continues_to_spherical(spherical_continues)
    xyz = point_spherical_to_xyz(spherical_coordinates)

    return xyz


def normalize_5D(
    pc,
    normalize_mode: str = "none",
    mean: list = [
        0.5302362442016602,
        0.008457111194729805,
        -0.0002034295175690204,
        0.7485858798027039,
        0.005146338604390621,
    ],
    std: list = [
        0.2266872227191925,
        0.6612724661827087,
        0.7500981092453003,
        0.2791619598865509,
        0.6013826727867126,
    ],
):
    """
    Normalize a 5D point cloud based on the specified mode.

    Args:
        pc (torch.Tensor): Input point cloud.
        normalize_mode (str): Mode of normalization ("none", "radius", "rotation", "full").
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        torch.Tensor: Normalized point cloud.
    """

    def normalize(pc, mean, std):
        mean_tensor = torch.tensor([mean], device=pc.device)
        std_tensor = torch.tensor([std], device=pc.device)
        return (pc - mean_tensor) / std_tensor

    if normalize_mode == "none":
        return pc
    elif normalize_mode == "radius":
        return normalize(pc, [mean[0], 0, 0, 0, 0], [std[0], 1, 1, 1, 1])
    elif normalize_mode == "rotation":
        return normalize(pc, [0, *mean[1:]], [1, *std[1:]])
    elif normalize_mode == "full":
        return normalize(pc, mean, std)
    else:
        raise ValueError("Invalid normalization mode.")


def inverse_normalize_5D(
    pc,
    normalize_mode: str = "none",
    mean: list = [
        0.5302362442016602,
        0.008457111194729805,
        -0.0002034295175690204,
        0.7485858798027039,
        0.005146338604390621,
    ],
    std: list = [
        0.2266872227191925,
        0.6612724661827087,
        0.7500981092453003,
        0.2791619598865509,
        0.6013826727867126,
    ],
):
    """
    Inverse normalize a 5D point cloud based on the specified mode.

    Args:
        pc (torch.Tensor): Input normalized point cloud.
        normalize_mode (str): Mode of normalization ("none", "radius", "rotation", "full").
        mean (list): Mean values for inverse normalization.
        std (list): Standard deviation values for inverse normalization.

    Returns:
        torch.Tensor: Inverse normalized point cloud.
    """

    def inverse_normalize(pc, mean, std):
        mean_tensor = torch.tensor([mean], device=pc.device)
        std_tensor = torch.tensor([std], device=pc.device)
        return pc * std_tensor + mean_tensor

    if normalize_mode == "none":
        return pc
    elif normalize_mode == "radius":
        return inverse_normalize(pc, [mean[0], 0, 0, 0, 0], [std[0], 1, 1, 1, 1])
    elif normalize_mode == "rotation":
        return inverse_normalize(pc, [0, *mean[1:]], [1, *std[1:]])
    elif normalize_mode == "full":
        return inverse_normalize(pc, mean, std)
    else:
        raise ValueError("Invalid normalization mode.")


def chamfer_loss_5D(source, target):
    """
    Compute the Chamfer Loss for 5D point cloud representations.

    Args:
        source (torch.Tensor): Source point cloud of shape (B, N, 5).
        target (torch.Tensor): Target point cloud of shape (B, M, 5).

    Returns:
        torch.Tensor: Scalar tensor representing the Chamfer Loss.
    """
    # Ensure inputs are valid
    assert source.shape[-1] == 5, "Source points should be in 5D space."
    assert target.shape[-1] == 5, "Target points should be in 5D space."

    # Extract radius and angular components
    r_s, sin_theta_s, cos_theta_s, sin_phi_s, cos_phi_s = torch.split(source, 1, dim=-1)
    r_t, sin_theta_t, cos_theta_t, sin_phi_t, cos_phi_t = torch.split(target, 1, dim=-1)

    # Compute pairwise distances for radius
    radius_dist = (r_s - r_t.transpose(1, 2)) ** 2  # Shape: (B, N, M)

    # Compute pairwise distances for angles
    sin_theta_dist = (sin_theta_s - sin_theta_t.transpose(1, 2)) ** 2
    cos_theta_dist = (cos_theta_s - cos_theta_t.transpose(1, 2)) ** 2
    sin_phi_dist = (sin_phi_s - sin_phi_t.transpose(1, 2)) ** 2
    cos_phi_dist = (cos_phi_s - cos_phi_t.transpose(1, 2)) ** 2

    # Total angular distance
    angular_dist = (
        sin_theta_dist + cos_theta_dist + sin_phi_dist + cos_phi_dist
    )  # (B, N, M)

    # Total distance in 5D space
    total_dist = radius_dist + angular_dist  # (B, N, M)

    # For each point in source, find the closest point in target
    src_to_tgt_dist = torch.min(total_dist, dim=2).values  # (B, N)

    # For each point in target, find the closest point in source
    tgt_to_src_dist = torch.min(total_dist, dim=1).values  # (B, M)

    # Symmetric Chamfer Loss
    loss = src_to_tgt_dist.mean() + tgt_to_src_dist.mean()

    return loss


def point_to_euler_radius(pc):
    """
    Compute the Euler angles (yaw, pitch, roll) and radius for each point in a batch of point clouds.

    Args:
        pc (torch.Tensor): Tensor of shape (B, N, 3) with (x, y, z) coordinates.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Euler angles of shape (B, N, 3) and radius of shape (B, N, 1).
    """
    # Extract x, y, z coordinates
    x = pc[..., 0]
    y = pc[..., 1]
    z = pc[..., 2]

    # Compute yaw (azimuth): atan2(y, x)
    yaw = torch.atan2(y, x)

    # Compute pitch (elevation): atan2(z, sqrt(x^2 + y^2))
    pitch = torch.atan2(z, torch.sqrt(x**2 + y**2 + 1e-7))

    # Set roll to 0 (undefined for individual points)
    roll = torch.zeros_like(yaw)

    # Combine yaw, pitch, and roll into a single tensor
    euler_angles = torch.stack((yaw, pitch, roll), dim=-1)

    # Compute radius: sqrt(x^2 + y^2 + z^2)
    radius = torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True) + 1e-7)

    return euler_angles, radius


def euler_radius_to_point(euler_angles, radius):
    """
    Convert Euler angles and radius to XYZ coordinates.

    Args:
        euler_angles (torch.Tensor): Tensor of shape (B, N, 3) with yaw, pitch, and roll.
        radius (torch.Tensor): Tensor of shape (B, N, 1) with radius values.

    Returns:
        torch.Tensor: Tensor of shape (B, N, 3) with reconstructed XYZ coordinates.
    """
    # Extract yaw and pitch
    yaw = euler_angles[..., 0]
    pitch = euler_angles[..., 1]

    # Debug Euler angles
    assert not torch.isnan(yaw).any(), "NaN in yaw!"
    assert not torch.isnan(pitch).any(), "NaN in pitch!"

    # Compute direction vector from yaw and pitch
    x = torch.cos(pitch) * torch.cos(yaw)
    y = torch.cos(pitch) * torch.sin(yaw)
    z = torch.sin(pitch)

    direction_vector = torch.stack((x, y, z), dim=-1)  # (B, N, 3)

    # Debug direction vector
    assert not torch.isnan(direction_vector).any(), "NaN in direction vector!"

    # Clamp radius to avoid instability
    radius = torch.clamp(radius, min=1e-7, max=1e3)

    # Scale direction vector by radius
    xyz = radius * direction_vector

    # Debug final output
    assert not torch.isnan(xyz).any(), "NaN in reconstructed XYZ!"

    return xyz


def point_xyz_to_7D(pc):
    """
    Convert a 3D point cloud to a 7D representation (radius + 6D rotation).

    Args:
        pc (torch.Tensor): Tensor of shape (B, N, 3) with (x, y, z) coordinates.

    Returns:
        torch.Tensor: Tensor of shape (B, N, 7) with 1 radius and 6D rotation.
    """
    assert pc.shape[-1] == 3, "The last dimension of the point cloud should be 3."
    assert len(pc.shape) == 3, (
        "The input point cloud should have 3 dimensions (B, N, 3)."
    )

    euler, radius = point_to_euler_radius(pc)

    # Convert Euler angles to rotation matrix and then to 6D representation
    matrix_rotation = euler_angles_to_matrix(euler, convention="ZYX")
    rotation_6d = matrix_to_rotation_6d(matrix_rotation)

    return torch.cat([radius, rotation_6d], dim=-1)


def point_7D_to_xyz(pc_7d):
    """
    Convert a 7D point cloud representation back to 3D XYZ coordinates.

    Args:
        pc_7d (torch.Tensor): Tensor of shape (B, N, 7) containing radius and 6D rotation.

    Returns:
        torch.Tensor: Tensor of shape (B, N, 3) with reconstructed XYZ coordinates.
    """
    assert pc_7d.shape[-1] == 7, "Expected input shape (B, N, 7)."

    # Extract radius and 6D rotation representation
    radius = pc_7d[..., 0:1]  # (B, N, 1)
    rotation_6d = pc_7d[..., 1:]  # (B, N, 6)

    # Convert 6D rotation to rotation matrix
    rotation_matrix = rotation_6d_to_matrix(rotation_6d)  # (B, N, 3, 3)

    # Debug rotation matrix
    assert not torch.isnan(rotation_matrix).any(), "NaN in rotation matrix!"
    assert torch.allclose(
        torch.det(rotation_matrix),
        torch.ones_like(torch.det(rotation_matrix)),
        atol=1e-6,
    ), "Invalid rotation matrix!"

    # Convert rotation matrix to Euler angles
    euler = matrix_to_euler_angles(rotation_matrix, convention="ZYX")  # (B, N, 3)

    # Debug Euler angles
    assert not torch.isnan(euler).any(), "NaN in Euler angles!"

    # Reconstruct XYZ coordinates
    xyz = euler_radius_to_point(euler, radius)

    return xyz


def chamfer_loss_7d(p, q, weight_radius=1.0, weight_rotation=1.0):
    """
    Compute Chamfer Loss for 7D point cloud representation.

    Args:
        p (torch.Tensor): Tensor of shape (B, N, 7), the predicted point cloud.
        q (torch.Tensor): Tensor of shape (B, M, 7), the target point cloud.
        weight_radius (float): Weight for the radius component.
        weight_rotation (float): Weight for the rotation component.

    Returns:
        torch.Tensor: Chamfer Loss (scalar).
    """
    assert p.shape[-1] == 7 and q.shape[-1] == 7, "Input tensors must have 7D features."

    # Separate radius and rotation components
    radius_p, rotation_p = p[..., :1], p[..., 1:]
    radius_q, rotation_q = q[..., :1], q[..., 1:]

    # Pairwise distances for radius
    dist_radius = torch.cdist(radius_p, radius_q) ** 2

    # Pairwise distances for rotation (6D vectors)
    dist_rotation = torch.cdist(rotation_p, rotation_q) ** 2

    # Combined distance
    dist_combined = weight_radius * dist_radius + weight_rotation * dist_rotation

    # Forward Chamfer Loss: For each point in p, find the closest point in q
    loss_p_to_q = torch.mean(torch.min(dist_combined, dim=-1)[0])

    # Backward Chamfer Loss: For each point in q, find the closest point in p
    loss_q_to_p = torch.mean(torch.min(dist_combined.transpose(1, 2), dim=-1)[0])

    # Final Chamfer Loss
    loss = loss_p_to_q + loss_q_to_p
    return loss


def normalize_7D(
    pc,
    normalize_mode="none",
    mean: list = [
        0.5302363038063049,
        -0.00013076931645628065,
        -0.008457111194729805,
        0.00080042117042467,
        0.0010340140433982015,
        -0.00020342960488051176,
        0.022919561713933945,
    ],
    std: list = [
        0.22668714821338654,
        0.6049997806549072,
        0.6612724661827087,
        0.4434205889701843,
        0.521811306476593,
        0.7500981092453003,
        0.4056345522403717,
    ],
):
    """
    Normalize a 7D point cloud representation based on the specified mode.

    Args:
        pc (torch.Tensor): Input point cloud.
        normalize_mode (str): Mode of normalization ("none", "radius", "rotation", "full").
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        torch.Tensor: Normalized point cloud.
    """

    def normalize(pc, mean, std):
        mean_tensor = torch.tensor([[[*mean]]], device=pc.device)
        std_tensor = torch.tensor([[[*std]]], device=pc.device)
        return (pc - mean_tensor) / std_tensor

    if normalize_mode == "full":
        return normalize(pc, mean, std)
    elif normalize_mode == "radius":
        return normalize(pc, [mean[0], 0, 0, 0, 0, 0, 0], [std[0], 1, 1, 1, 1, 1, 1])
    elif normalize_mode == "rotation":
        return normalize(pc, [0, *mean[1:]], [1, *std[1:]])
    elif normalize_mode == "none":
        return pc
    else:
        raise ValueError("Invalid normalization mode.")


def inverse_normalize_7D(
    pc,
    normalize_mode="none",
    mean: list = [
        0.5302363038063049,
        -0.00013076931645628065,
        -0.008457111194729805,
        0.00080042117042467,
        0.0010340140433982015,
        -0.00020342960488051176,
        0.022919561713933945,
    ],
    std: list = [
        0.22668714821338654,
        0.6049997806549072,
        0.6612724661827087,
        0.4434205889701843,
        0.521811306476593,
        0.7500981092453003,
        0.4056345522403717,
    ],
):
    """
    Inverse normalize a 7D point cloud representation based on the specified mode.

    Args:
        pc (torch.Tensor): Input normalized point cloud.
        normalize_mode (str): Mode of normalization ("none", "radius", "rotation", "full").
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        torch.Tensor: Inverse normalized point cloud.
    """

    def inverse_normalize(pc, mean, std):
        mean_tensor = torch.tensor([[[*mean]]], device=pc.device)
        std_tensor = torch.tensor([[[*std]]], device=pc.device)
        return pc * std_tensor + mean_tensor

    if normalize_mode == "full":
        return inverse_normalize(pc, mean, std)
    elif normalize_mode == "radius":
        return inverse_normalize(
            pc, [mean[0], 0, 0, 0, 0, 0, 0], [std[0], 1, 1, 1, 1, 1, 1]
        )
    elif normalize_mode == "rotation":
        return inverse_normalize(pc, [0, *mean[1:]], [1, *std[1:]])
    elif normalize_mode == "none":
        return pc
    else:
        raise ValueError("Invalid normalization mode.")
