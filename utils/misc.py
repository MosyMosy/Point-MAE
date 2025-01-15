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
    pc: torch.tensor, mean: list = [0, 0, 0], std: list = [0.356, 0.3052, 0.3358]
):
    assert pc.shape[-1] == 3, "The last dimension of the point cloud should be 3."
    mean = torch.tensor(mean, device=pc.device).expand_as(pc)
    std = torch.tensor(std, device=pc.device).expand_as(pc)
    torch.tensor([[[0.356, 0.3052, 0.3358]]])
    return (pc - mean) / std


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


def compute_angles_to_axes(points):
    """
    Compute angles between each point vector and the X, Y, Z axes.

    Args:
        points: Tensor of shape (B, N, 3), where B is batch size, N is number of points,
                and 3 corresponds to (x, y, z) coordinates.

    Returns:
        angles: Tensor of shape (B, N, 3), where the last dimension corresponds to
                angles with the X, Y, and Z axes (in radians).
    """
    # Compute the norm of each point vector: (B, N)
    norms = torch.linalg.norm(points, dim=-1, keepdim=True)  # (B, N, 1)

    # Normalize points to unit vectors to avoid dividing by zero
    norms = torch.clamp(norms, min=1e-8)  # Avoid division by zero

    # Compute the cosine of angles with each axis
    cos_angles = points / norms  # (B, N, 3)

    # Compute the angles in radians
    angles = torch.acos(cos_angles)  # (B, N, 3)

    return angles


def compute_polar_pose_embed(embed_dim, points):
    polar_points = compute_angles_to_axes(points)
    polar_embed = get_3d_sincos_pos_embed(embed_dim, polar_points)
    return polar_embed


def compute_polar_coordinates(pc):
    """
    Convert a batch of 3D points to polar coordinates (radius, azimuth, polar angle).

    Args:
        point_cloud (torch.Tensor): Tensor of shape (B, N, 3), where
                                     B is the batch size,
                                     N is the number of points,
                                     3 corresponds to (x, y, z) coordinates.

    Returns:
        torch.Tensor: Polar coordinates tensor of shape (B, N, 3), where
                      the last dimension represents (radius, azimuth, polar_angle).
    """
    # Extract x, y, z coordinates
    x = pc[..., 0]
    y = pc[..., 1]
    z = pc[..., 2]

    # Compute radius
    radius = torch.sqrt(x**2 + y**2 + z**2)

    # Compute azimuth (theta): atan2(y, x)
    azimuth = torch.atan2(y, x)

    # Compute polar angle (phi): acos(z / r)
    polar_angle = torch.acos(
        torch.clamp(z / radius, min=-1.0, max=1.0)
    )  # Clamp to avoid numerical issues

    # Combine into a single tensor
    polar_coordinates = torch.stack((radius, azimuth, polar_angle), dim=-1)

    return polar_coordinates


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


def normalize_7D_radius(
    pc, mean: float = 0.5302363038063049, std: float = 0.22668714821338654
):
    mean_radius = torch.tensor([[[mean, 0, 0, 0, 0, 0, 0]]], device=pc.device)
    std_radius = torch.tensor([[[std, 1, 1, 1, 1, 1, 1]]], device=pc.device)

    pc_7d_normed = (pc - mean_radius) / std_radius
    return pc_7d_normed


def inverse_normalize_7D_radius(
    pc_7d_normed, mean: float = 0.5302363038063049, std: float = 0.22668714821338654
):
    mean_radius = torch.tensor([[[mean, 0, 0, 0, 0, 0, 0]]], device=pc_7d_normed.device)
    std_radius = torch.tensor([[[std, 1, 1, 1, 1, 1, 1]]], device=pc_7d_normed.device)

    pc_7d_unnormed = pc_7d_normed * std_radius + mean_radius
    return pc_7d_unnormed


def normalize_7D_rotation(
    pc,
    mean: list = [
        -0.00013076931645628065,
        -0.008457111194729805,
        0.00080042117042467,
        0.0010340140433982015,
        -0.00020342960488051176,
        0.022919561713933945,
    ],
    std: list = [
        0.6049997806549072,
        0.6612724661827087,
        0.4434205889701843,
        0.521811306476593,
        0.7500981092453003,
        0.4056345522403717,
    ],
):
    mean_rotation = torch.tensor([[[0, *mean]]], device=pc.device)
    std_rotation = torch.tensor([[[1, *std]]], device=pc.device)

    pc_7d_normed = (pc - mean_rotation) / std_rotation
    return pc_7d_normed


def inverse_normalize_7D_rotation(
    pc_7d_normed,
    mean: list = [
        -0.00013076931645628065,
        -0.008457111194729805,
        0.00080042117042467,
        0.0010340140433982015,
        -0.00020342960488051176,
        0.022919561713933945,
    ],
    std: list = [
        0.6049997806549072,
        0.6612724661827087,
        0.4434205889701843,
        0.521811306476593,
        0.7500981092453003,
        0.4056345522403717,
    ],
):
    mean_rotation = torch.tensor([[[0, *mean]]], device=pc_7d_normed.device)
    std_rotation = torch.tensor([[[1, *std]]], device=pc_7d_normed.device)

    pc_7d_unnormed = pc_7d_normed * std_rotation + mean_rotation
    return pc_7d_unnormed


def normalize_7D(
    pc,
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
    pc_7d_normed = normalize_7D_radius(pc, mean[0], std[0])
    pc_7d_normed = normalize_7D_rotation(pc_7d_normed, mean[1:], std[1:])
    return pc_7d_normed


def inverse_normalize_7D(
    pc_7d_normed,
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
    pc_7d_unnormed = inverse_normalize_7D_rotation(pc_7d_normed, mean[1:], std[1:])
    pc_7d_unnormed = inverse_normalize_7D_radius(pc_7d_unnormed, mean[0], std[0])
    return pc_7d_unnormed
