"""UV atlas generation and mesh preprocessing."""
import os
from typing import Tuple, Union

import numpy as np
import open3d as o3d
import torch
import trimesh
import xatlas


def preprocess_blank_mesh(
    input_obj_path: str,
    output_obj_path: str,
    min_faces: int = 20_000,
    max_faces: int = 200_000,
    scale: float = 1.0,
):
    """Load, simplify/subdivide, UV-unwrap, and save a mesh for texture baking."""
    mesh_o3d = o3d.io.read_triangle_mesh(input_obj_path, enable_post_processing=False)

    # GLB coordinate system: Z-up → Y-up
    if input_obj_path.lower().endswith(".glb"):
        mesh_o3d.transform(np.array([
            [1, 0,  0, 0],
            [0, 0, -1, 0],
            [0, 1,  0, 0],
            [0, 0,  0, 1],
        ]))

    mesh_o3d = _preprocess_blank_mesh_o3d(mesh_o3d, min_faces=min_faces, max_faces=max_faces, scale=scale)

    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
    o3d.io.write_triangle_mesh(
        output_obj_path, mesh_o3d,
        write_ascii=False, compressed=False,
        write_vertex_normals=False, write_vertex_colors=False,
        write_triangle_uvs=True,
    )


def _preprocess_blank_mesh_o3d(
    mesh: o3d.geometry.TriangleMesh,
    min_faces: int = 20_000,
    max_faces: int = 200_000,
    scale: float = 1.0,
) -> o3d.geometry.TriangleMesh:
    """Simplify or subdivide *mesh* to target face range, then compute UV atlas."""
    device_o3d = o3d.core.Device("CPU:0")

    num_faces = len(mesh.triangles)
    if num_faces < min_faces:
        target_faces = min(min_faces, max_faces)
        subdivision_iters = max(1, int(np.ceil(np.log2(target_faces / num_faces) / 2)))
        mesh = mesh.subdivide_midpoint(number_of_iterations=subdivision_iters)
    elif num_faces > max_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max_faces)

    # Align to bounding box
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_max_extent() / (2.0 * scale)
    mesh.translate(-center)
    mesh.scale(1.0 / extent, center=np.zeros(3))

    # Remove degenerate / unreferenced geometry
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()

    # Compute UV atlas via Open3D
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh, device=device_o3d)
    mesh_t.compute_uvatlas(size=2048, gutter=4.0, max_stretch=0.1667, parallel_partitions=4, nthreads=8)
    mesh_legacy = mesh_t.to_legacy()
    mesh.triangle_uvs = mesh_legacy.triangle_uvs

    return mesh
