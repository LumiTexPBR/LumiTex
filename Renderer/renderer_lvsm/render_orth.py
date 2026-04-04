"""
Single Orthographic Rendering Script

This script renders GLB models from multiple viewpoints using orthographic projection.
Outputs include RGB images, normal maps, coordinate maps, and camera parameters.
"""

import os
import json
import time
from typing import Tuple, Optional

# Set environment variables for CUDA
os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_ext"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["TORCH_CUDA_BUILD_VERBOSE"] = "0"
os.environ["NINJA_STATUS"] = ""
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import nvdiffrast.torch as dr
from tqdm import tqdm

from .cam import Camera
from .mesh import Mesh


def diffrast_w2c_to_opencv_w2c(diffrast_w2c: np.ndarray) -> np.ndarray:
    """
    Convert diffrast world-to-camera matrix to OpenCV format.
    
    Args:
        diffrast_w2c: Diffrast world-to-camera transformation matrix
        
    Returns:
        OpenCV compatible world-to-camera matrix
    """
    A = np.eye(4)
    A[:3, :3] = np.diag([1, -1, -1])
    opencv_w2c = A @ diffrast_w2c @ A
    return opencv_w2c


@torch.no_grad()
def export_lvsm_condition(
    cadidate_views_path: str,
    model_path: str,
    output_dir: str,
    geometry_scale: float = 0.90,
    H: int = 512,
    W: int = 512,
) -> None:
    """
    Render a GLB model from multiple viewpoints using orthographic projection.
    
    Args:
        model_path: Path to the GLB model file
        output_dir: Directory to save rendered images
        H: Image height
        W: Image width
    """
    
    # Setup camera with predefined viewpoints
    camera = Camera()
    with open(cadidate_views_path, "r") as f:
        data = json.load(f)
    
    azimuths = data["b"]
    elevations = data["a"]
    camera.generate_camera_sequence_ortho(
        radius=2.8, 
        ortho_scale=1.0, 
        near=0.1, 
        far=100.0, 
        azimuths=azimuths, 
        elevations=elevations
    )
    
    # Load 3D mesh
    try:
        mesh = Mesh.load(model_path, bound=geometry_scale)
    except Exception as e:
        print(f"Mesh loading error for {model_path}: {e}")
        return
    
    # Setup rendering context
    glctx = dr.RasterizeCudaContext()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # Create output directories
    # os.makedirs(os.path.join(output_dir, 'render'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'ccm'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'normal'), exist_ok=True)
    
    # Check if mesh has valid albedo texture
    if mesh.albedo is None or mesh.albedo.shape[-1] != 3:
        print(f"No valid albedo texture found for {model_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    
    # Render from each viewpoint
    for idx, (pose, proj, w2c) in enumerate(zip(camera.pose, camera.proj, camera.w2c)):
        # Convert numpy arrays to torch tensors
        pose = torch.from_numpy(pose.astype(np.float32)).cuda()
        proj = torch.from_numpy(proj.astype(np.float32)).cuda()
        
        # Transform vertices to camera space
        v_cam = torch.matmul(
            F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), 
            pose.T
        ).float().unsqueeze(0)
        
        # Project to clip space
        v_clip = v_cam @ proj.T
        
        # Rasterize
        rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (H, W))
        alpha = (rast[..., 3:] > 0).float()
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).squeeze(0).clamp(0, 1)
        alpha = alpha.permute(2, 0, 1)[0:1, :, :]
        
        # Render position map (coordinate map)
        pos_map, _ = dr.interpolate(mesh.v[None], rast, mesh.f)
        pos_map = pos_map[0]
        
        # Render normal map
        normals, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
        normal = normals[0]
        normal = normal / normal.norm(dim=2, keepdim=True).clamp(min=1e-20)
        
        # Calculate view directions for orthographic projection
        R = pose[:3, :3]
        view_dirs = -R[:, 2]
        view_dirs = view_dirs / torch.norm(view_dirs)
        view_dirs = view_dirs.view(1, 1, 3).expand(H, W, 3)
        
        torchvision.utils.save_image(
            torch.cat(((normal.permute(2, 0, 1) + 1.0) / 2.0, alpha), dim=0),
            os.path.join(output_dir, 'normal', f"{idx:03d}.png")
        )
        
        torchvision.utils.save_image(
            torch.cat(((pos_map.permute(2, 0, 1) + 1.0) / 2.0, alpha), dim=0),
            os.path.join(output_dir, 'ccm', f"{idx:03d}.png")
        )
        
        # Store camera parameters
        frame = {
            "w": W,
            "h": H,
            "fx": 2.0,
            "fy": 2.0,
            "cx": 2.0,
            "cy": 2.0,
            "w2c": w2c.tolist(),
            "file_path": os.path.join('render_type', f"{idx:03d}.png"),
        }
        frames.append(frame)
    
    # Save camera parameters to JSON
    with open(os.path.join(output_dir, 'opencv_cameras.json'), 'w') as f:
        json.dump(frames, f, indent=2)



def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Render GLB model with orthographic projection")
    parser.add_argument("--folder", type=str, required=True, help="Folder name containing the GLB model")
    parser.add_argument("--res", type=int, required=True, help="Rendering resolution (height and width)")
    parser.add_argument("--cadidate_views_path", type=str, required=True, help="Path to the candidate views file")
    return parser.parse_args()


def main(args):
    """Main function to execute the rendering pipeline."""
    folder = args.folder
    resolution = args.res
    cadidate_views_path = args.cadidate_views_path
    # Setup paths
    base_dir = os.environ.get("/cephfs/hongzechen/FLUX_TEST", ".")
    model_path = os.path.join(base_dir, "assets", folder, "mesh.glb")
    render_path = os.path.join(base_dir, "assets", folder, "tmp", "lvsm")
    
    # Create output directory
    os.makedirs(render_path, exist_ok=True)
    
    # Execute rendering
    export_lvsm_condition(cadidate_views_path=cadidate_views_path, model_path=model_path, output_dir=render_path, H=resolution, W=resolution)


if __name__ == '__main__':
    args = parse_args()
    main(args)
