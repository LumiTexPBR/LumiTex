import torch
import math
import numpy as np
from typing import Union, Optional, Tuple, List
import trimesh


def intr_to_proj(intr_mtx:torch.Tensor, near=0.01, far=1000.0, perspective=False):
    proj_mtx = torch.zeros((*intr_mtx.shape[:-2], 4, 4), dtype=intr_mtx.dtype, device=intr_mtx.device)
    intr_mtx = intr_mtx.clone()
    if perspective:
        proj_mtx[..., 0, 0] = 2 * intr_mtx[..., 0, 0]
        proj_mtx[..., 1, 1] = 2 * intr_mtx[..., 1, 1]
        proj_mtx[..., 2, 2] = -(far + near) / (far - near)
        proj_mtx[..., 0, 2] = 2 * intr_mtx[..., 0, 2] - 1
        proj_mtx[..., 1, 2] = 2 * intr_mtx[..., 1, 2] - 1
        proj_mtx[..., 3, 2] = -1.0
        proj_mtx[..., 2, 3] = -2.0 * far * near / (far - near)
    else:
        proj_mtx[..., 0, 0] = intr_mtx[..., 0, 0]
        proj_mtx[..., 1, 1] = intr_mtx[..., 1, 1]
        proj_mtx[..., 2, 2] = -2.0 / (far - near)
        proj_mtx[..., 3, 3] = 1.0
        proj_mtx[..., 0, 3] = -(2 * intr_mtx[..., 0, 2] - 1)
        proj_mtx[..., 1, 3] = -(2 * intr_mtx[..., 1, 2] - 1)
        proj_mtx[..., 2, 3] = - (far + near) / (far - near)
    proj_mtx[..., 1, :] = -proj_mtx[..., 1, :]  # for nvdiffrast
    return proj_mtx


def generate_intrinsics(f_x: float, f_y: float, fov=True, degree=False):
    '''
    f_x, f_y: 
        * focal length divide width/height for perspective camera
        * fov degree or radians for perspective camera
        * scale for orthogonal camera
    intrinsics: [3, 3], normalized
    '''
    if fov:
        if degree:
            f_x = math.radians(f_x)
            f_y = math.radians(f_y)
        f_x_div_W = 1 / (2 * math.tan(f_x / 2))
        f_y_div_H = 1 / (2 * math.tan(f_y / 2))
    else:
        f_x_div_W = f_x
        f_y_div_H = f_y
    return torch.as_tensor([
        [f_x_div_W, 0.0, 0.5],
        [0.0, f_y_div_H, 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

def c2w_to_w2c(c2w:torch.Tensor):
    # y = Rx + t, x = R_inv(y - t)
    w2c = torch.zeros((*c2w.shape[:-2], 4, 4), dtype=c2w.dtype, device=c2w.device)
    c2w = c2w.clone()
    w2c[..., :3, :3] = c2w[..., :3, :3].transpose(-1, -2)
    w2c[..., :3, 3:] = -c2w[..., :3, :3].transpose(-1, -2) @ c2w[..., :3, 3:]
    w2c[..., 3, 3] = 1.0
    return w2c


def lookat_to_matrix(lookat:torch.Tensor) -> torch.Tensor:
    '''
    lookat: [..., 3], camera locations looking at origin
    c2ws: [..., 4, 4]
        * world: x forward, y right, z up, need to transform xyz to zxy
        * camera: z forward, x right, y up
    '''
    batch_shape = lookat.shape[:-1]
    e2 = torch.as_tensor([0.0, 1.0, 0.0], dtype=lookat.dtype, device=lookat.device)
    e3 = torch.as_tensor([0.0, 0.0, 1.0], dtype=lookat.dtype, device=lookat.device)
    zzzo = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=lookat.dtype, device=lookat.device)
    xyz_to_zxy = torch.as_tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=lookat.dtype, device=lookat.device)
    # NOTE: camera locations are opposite from ray directions
    z_axis = torch.nn.functional.normalize(lookat, dim=-1)
    x_axis = torch.linalg.cross(e3.expand_as(z_axis), z_axis, dim=-1)
    x_axis_mask = (x_axis == 0).all(dim=-1, keepdim=True)
    if x_axis_mask.sum() > 0:
        # NOTE: top and down is not well defined, hard code here
        x_axis = torch.where(x_axis_mask, e2, x_axis)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    rots = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    c2ws = torch.cat([
        torch.cat([rots, lookat.unsqueeze(-1)], dim=-1),
        zzzo.expand(batch_shape + (1, -1)),
    ], dim=1)
    # NOTE: world f/r/u is x/y/z, camera f/r/u is z/x/y, so we need to transform x/y/z to z/x/y for c2ws
    c2ws = torch.matmul(xyz_to_zxy, c2ws)
    return c2ws


def lookat_to_matrix_fixed(lookat: torch.Tensor) -> torch.Tensor:
    '''
    lookat: [..., 3], camera locations looking at origin
    c2ws: [..., 4, 4]
    '''
    batch_shape = lookat.shape[:-1]
    e2 = torch.as_tensor([0.0, 1.0, 0.0], dtype=lookat.dtype, device=lookat.device)
    e3 = torch.as_tensor([0.0, 0.0, 1.0], dtype=lookat.dtype, device=lookat.device)
    zzzo = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=lookat.dtype, device=lookat.device)
    xyz_to_zxy = torch.as_tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=lookat.dtype, device=lookat.device)
    z_axis = torch.nn.functional.normalize(lookat, dim=-1)
    x_axis = torch.linalg.cross(e3.expand_as(z_axis), z_axis, dim=-1)
    x_axis = torch.nn.functional.normalize(x_axis, dim=-1)
    x_axis_mask = (torch.sum(x_axis * x_axis, dim=-1, keepdim=True) < 1e-6)
    if x_axis_mask.sum() > 0:
        x_axis = torch.where(x_axis_mask, e2, x_axis)
        x_axis = torch.nn.functional.normalize(x_axis, dim=-1)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = torch.nn.functional.normalize(y_axis, dim=-1)
    
    rots = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    c2ws = torch.cat([
        torch.cat([rots, lookat.unsqueeze(-1)], dim=-1),
        zzzo.expand(batch_shape + (1, -1)),
    ], dim=1)
    
    c2ws = torch.matmul(xyz_to_zxy, c2ws)
    return c2ws


def generate_orbit_views_c2ws(num_views: int, radius: float = 1.0, height: float = 0.0, theta_0: float = 0.0, degree=False):
    if degree:
        theta_0 = math.radians(theta_0)
    projected_radius = math.sqrt(radius ** 2 - height ** 2)
    theta = torch.linspace(theta_0, 2.0 * math.pi + theta_0, num_views, dtype=torch.float32)
    x = projected_radius * torch.cos(theta)
    y = projected_radius * torch.sin(theta)
    z = torch.full((num_views,), fill_value=height, dtype=torch.float32)
    xyz = torch.stack([x, y, z], dim=-1)
    c2ws = lookat_to_matrix(xyz)
    return c2ws


def sample_point_on_sphere(
    radius: float, 
    theta: float = None, 
    phi: float = None
):
    """
    Sample a point on the unit sphere.
    pytorch3d (up-axis: Y, forward-axis: -Z)
    blender (up-axis: Z, forward-axis: Y)
    ref: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/cameras.html#camera_position_from_spherical_angles
    """
    # Convert inputs to torch tensors to handle both scalars and arrays
    theta = torch.as_tensor(theta, dtype=torch.float32)
    phi = torch.as_tensor(phi, dtype=torch.float32)
    
    x = radius * torch.cos(phi) * torch.sin(theta)
    y = radius * torch.sin(phi)
    z = radius * torch.cos(phi) * torch.cos(theta)
    # pytorch3d (up-axis: Y, forward-axis: -Z) -> blender (up-axis: Z, forward-axis: Y)
    x, y, z = z, x, y
    return x, y, z


def generate_orbit_views_c2ws_from_elev_azim(radius: float = 2.0, elevation: List[float] = None, azimuth: List[float] = None):
    ele = np.deg2rad(elevation)
    azi = np.deg2rad(azimuth)
    x, y, z = sample_point_on_sphere(
        radius,
        theta=azi,
        phi=ele,
    )
    xyz = torch.stack([x, y, z], dim=-1)
    c2ws = lookat_to_matrix_fixed(xyz)
    return c2ws


def scale_to_sphere(vertices):
    center = vertices.mean(dim=0)
    vertices = vertices - center
    scale = vertices.norm(dim=1).max()
    vertices = vertices / scale
    return vertices



def get_mv_matrix(elev, azim, camera_distance, center=None):
    elev = -elev
    azim += 90

    elev_rad = math.radians(elev)
    azim_rad = math.radians(azim)

    camera_position = np.array(
        [
            camera_distance * math.cos(elev_rad) * math.cos(azim_rad),
            camera_distance * math.cos(elev_rad) * math.sin(azim_rad),
            camera_distance * math.sin(elev_rad),
        ]
    )

    if center is None:
        center = np.array([0, 0, 0])
    else:
        center = np.array(center)

    lookat = center - camera_position
    lookat = lookat / np.linalg.norm(lookat)

    up = np.array([0, 0, 1.0])
    right = np.cross(lookat, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, lookat)
    up = up / np.linalg.norm(up)

    c2w = np.concatenate([np.stack([right, up, -lookat], axis=-1), camera_position[:, None]], axis=-1)

    w2c = np.zeros((4, 4))
    w2c[:3, :3] = np.transpose(c2w[:3, :3], (1, 0))
    w2c[:3, 3:] = -np.matmul(np.transpose(c2w[:3, :3], (1, 0)), c2w[:3, 3:])
    w2c[3, 3] = 1.0

    return c2w.astype(np.float32), w2c.astype(np.float32)


def convert_orbit_c2w_to_get_mv_format(c2w_orbit: torch.Tensor) -> np.ndarray:
    """
    Convert orbit-view style c2w to get_mv_matrix-style c2w.

    Args:
        c2w_orbit: torch.Tensor of shape [4, 4]

    Returns:
        numpy.ndarray of shape [3, 4]
    """
    R = c2w_orbit[:3, :3]
    t = c2w_orbit[:3, 3]

    # reorder basis vectors: orbit [x, y, z] -> get_mv [z, x, y]
    R_new = torch.stack([
        R[:, 2],  # z -> x
        R[:, 0],  # x -> y
        R[:, 1],  # y -> z
    ], dim=-1)

    c2w_mv = torch.cat([R_new, t.unsqueeze(-1)], dim=-1)
    return c2w_mv.cpu().numpy()


def verify_scaling_issue():
    """验证原函数的缩放问题"""
    import torch
    
    # 测试一个非垂直的相机位置
    lookat = torch.tensor([[1.0, 1.0, 1.0]])  # 不平行于z轴
    
    # 原函数
    c2w_original = lookat_to_matrix(lookat)
    R_original = c2w_original[0, :3, :3]
    
    # 检查列向量的长度
    col_norms = [torch.norm(R_original[:, i]).item() for i in range(3)]
    print(f"Original function column norms: {col_norms}")
    
    # 修正函数
    c2w_fixed = lookat_to_matrix_fixed(lookat)
    R_fixed = c2w_fixed[0, :3, :3]
    
    col_norms_fixed = [torch.norm(R_fixed[:, i]).item() for i in range(3)]
    print(f"Fixed function column norms: {col_norms_fixed}")


if __name__ == "__main__":
    elev, azim = 20, 0
    radius = 2.0
    # c2w, w2c = get_mv_matrix(elev, azim, radius)
    # print("TEST get_mv_matrix")
    # print(w2c)
    # print(c2w)
    
    # print("TEST generate_orbit_views_c2ws_from_elev_azim")
    # c2ws = generate_orbit_views_c2ws_from_elev_azim(radius, [elev], [azim])
    # print(c2ws)
    
    # print("\n" + "="*50)
    # print("Testing conversion function:")
    # success = test_conversion()
    # print(f"Conversion test {'PASSED' if success else 'FAILED'}")

    verify_scaling_issue()