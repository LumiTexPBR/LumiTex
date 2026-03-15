import hashlib
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import trimesh
import xatlas
from PIL import Image, ImageFilter
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial, SimpleMaterial


def _to_uint8_color(
    color: Optional[np.ndarray], channels: int, default_alpha: int = 255
) -> Optional[Tuple[int, ...]]:
    """Convert a color-like value to an uint8 tuple with fixed channel count."""
    if color is None:
        return None

    color_array = np.asarray(color, dtype=np.float32).reshape(-1)
    if color_array.size == 0:
        return None

    if color_array.max() <= 1.0:
        color_array = color_array * 255.0

    if channels == 4:
        if color_array.size == 3:
            color_array = np.concatenate([color_array, np.array([default_alpha], dtype=np.float32)])
        elif color_array.size > 4:
            color_array = color_array[:4]
        elif color_array.size < 4:
            return None
    elif channels == 3:
        if color_array.size > 3:
            color_array = color_array[:3]
        elif color_array.size < 3:
            return None
    else:
        return None

    color_u8 = color_array.clip(0.0, 255.0).astype(np.uint8)
    return tuple(int(x) for x in color_u8.tolist())


def parse_texture_visuals(
    texture_visuals: TextureVisuals,
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image]]:
    """
    Parse texture visuals and return `(albedo_map, metallic_roughness_map, normal_map)`.
    """
    if isinstance(texture_visuals.material, SimpleMaterial):
        map_kd = texture_visuals.material.image
        if map_kd is None:
            color_kd = _to_uint8_color(texture_visuals.material.diffuse, channels=4)
            if color_kd is not None:
                map_kd = Image.new(mode="RGBA", size=(4, 4), color=color_kd)
        map_ks = None
        map_normal = None
    elif isinstance(texture_visuals.material, PBRMaterial):
        map_kd = texture_visuals.material._data.get("baseColorTexture", None)
        if map_kd is None:
            map_kd = texture_visuals.material._data.get("emissiveTexture", None)
        if map_kd is None:
            color_kd = texture_visuals.material._data.get("baseColorFactor", None)
            if color_kd is None:
                color_kd = texture_visuals.material._data.get("emissiveFactor", None)
            color_kd = _to_uint8_color(color_kd, channels=4)
            if color_kd is not None:
                map_kd = Image.new(mode="RGBA", size=(4, 4), color=color_kd)

        map_ks = texture_visuals.material._data.get("metallicRoughnessTexture", None)
        if map_ks is None:
            color_m = texture_visuals.material._data.get("metallicFactor", None)
            color_r = texture_visuals.material._data.get("roughnessFactor", None)
            if color_m is not None or color_r is not None:
                if color_m is None:
                    color_m = 0.0
                if color_r is None:
                    color_r = 1.0
                color_ks = np.asarray([1.0, color_r, color_m], dtype=np.float32)
                color_ks = (color_ks.clip(0.0, 1.0) * 255).astype(np.uint8)
                map_ks = Image.new(mode="RGB", size=(4, 4), color=tuple(int(x) for x in color_ks))

        map_normal = texture_visuals.material._data.get("normalTexture", None)
    else:
        map_kd = None
        map_ks = None
        map_normal = None

    return map_kd, map_ks, map_normal


def parse_mesh_info(mesh_path: str) -> dict:
    """Parse mesh metadata and return a dictionary including face count `F` when available."""
    mesh_ext = os.path.splitext(mesh_path)[1].lower()
    num_faces = None

    if mesh_ext == ".ply":
        with open(mesh_path, "rb") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line_text = line.decode("utf-8", errors="ignore").strip()
                if line_text.startswith("element face"):
                    parts = line_text.split()
                    if len(parts) >= 3:
                        num_faces = int(parts[2])
                if line_text == "end_header":
                    break
    elif mesh_ext == ".off":
        with open(mesh_path, "r", encoding="utf-8", errors="ignore") as file:
            first_line = file.readline().strip()
            if first_line in ("OFF", "COFF"):
                second_line = file.readline().strip()
                while second_line.startswith("#"):
                    second_line = file.readline().strip()
                parts = second_line.split()
                if len(parts) >= 2:
                    num_faces = int(parts[1])
    elif mesh_ext == ".stl":
        with open(mesh_path, "rb") as file:
            file.seek(80)
            face_bytes = file.read(4)
            if len(face_bytes) == 4:
                num_faces = int.from_bytes(face_bytes, byteorder="little", signed=False)
    elif mesh_ext == ".obj":
        with open(mesh_path, "r", encoding="utf-8", errors="ignore") as file:
            num_faces = 0
            for line in file:
                if line.startswith("f "):
                    num_faces += 1

    return {"F": num_faces}


def convert_to_whole_mesh(scene_or_mesh: trimesh.Scene) -> trimesh.Trimesh:
    """Convert a loaded trimesh scene/mesh object into a single trimesh.Trimesh."""
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh

    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            raise ValueError("Loaded scene contains no geometry.")

        merged_mesh = scene_or_mesh.dump(concatenate=True)
        if isinstance(merged_mesh, trimesh.Trimesh):
            return merged_mesh

        if isinstance(merged_mesh, (list, tuple)):
            mesh_list = [mesh for mesh in merged_mesh if isinstance(mesh, trimesh.Trimesh)]
            if len(mesh_list) == 0:
                raise ValueError("Scene dump did not contain trimesh.Trimesh objects.")
            return trimesh.util.concatenate(mesh_list)

        raise TypeError(f"Unsupported scene dump type: {type(merged_mesh)}")

    raise TypeError(f"Unsupported loaded type: {type(scene_or_mesh)}")


def load_whole_mesh(mesh_path: str, limited_faces: Optional[int] = 10_000_000) -> trimesh.Trimesh:
    """
    Load mesh as a single Trimesh with optional face-count guard.
    """
    if limited_faces is not None:
        num_faces = parse_mesh_info(mesh_path)["F"]
        if num_faces is not None:
            assert (
                num_faces <= limited_faces
            ), f"num faces {num_faces} is larger than limited_faces {limited_faces}"

    scene = trimesh.load(mesh_path, process=False)
    mesh = convert_to_whole_mesh(scene)
    return mesh


def standardize_mesh_for_uv(
    mesh: trimesh.Trimesh, weld_tolerance: float = 1e-8
) -> Tuple[trimesh.Trimesh, np.ndarray, str]:
    """
    Canonicalize mesh geometry for deterministic UV parameterization.

    The mesh is cleaned, normalized, sorted deterministically, and hashed so atlas generation
    remains stable across runs.
    """
    standardized_mesh = mesh.copy()

    try:
        standardized_mesh.remove_degenerate_faces()
        standardized_mesh.remove_duplicate_faces()
    except Exception:
        pass

    try:
        standardized_mesh.merge_vertices(radius=weld_tolerance)
    except Exception:
        pass

    vertices = standardized_mesh.vertices.view(np.ndarray).copy()
    faces = standardized_mesh.faces.view(np.ndarray).copy()

    center = vertices.mean(axis=0)
    vertices_centered = vertices - center
    covariance = np.cov(vertices_centered.T)
    u_matrix, _, _ = np.linalg.svd(covariance)
    rotation = u_matrix
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1

    vertices_projected = vertices_centered @ rotation
    scale = np.max(np.linalg.norm(vertices_projected.max(axis=0) - vertices_projected.min(axis=0)))
    if scale > 0:
        vertices_projected = vertices_projected / scale

    vertex_order = np.lexsort(
        (vertices_projected[:, 2], vertices_projected[:, 1], vertices_projected[:, 0])
    )
    old_to_new = np.empty(len(vertices_projected), dtype=np.int64)
    old_to_new[vertex_order] = np.arange(len(vertices_projected))
    vertices_sorted = vertices_projected[vertex_order]
    faces_reindexed = old_to_new[faces]

    normalized_faces = []
    for triangle in faces_reindexed:
        triangle = np.asarray(triangle)
        min_index_position = int(np.argmin(triangle))
        rotated = np.concatenate([triangle[min_index_position:], triangle[:min_index_position]])
        b_index, c_index = int(rotated[1]), int(rotated[2])
        if b_index > c_index:
            rotated[1], rotated[2] = rotated[2], rotated[1]
        normalized_faces.append(rotated)
    normalized_faces = np.asarray(normalized_faces, dtype=np.int64)

    face_order = np.lexsort(
        (normalized_faces[:, 2], normalized_faces[:, 1], normalized_faces[:, 0])
    )
    normalized_faces = normalized_faces[face_order]

    standardized_mesh.vertices = vertices_sorted
    standardized_mesh.faces = normalized_faces
    if hasattr(standardized_mesh.visual, "uv"):
        standardized_mesh.visual.uv = None

    new_to_old_vertex_index = vertex_order

    hasher = hashlib.sha256()
    hasher.update(np.round(vertices_sorted, 8).tobytes())
    hasher.update(normalized_faces.tobytes())
    geometry_hash = hasher.hexdigest()
    return standardized_mesh, new_to_old_vertex_index, geometry_hash


def parameterize_uv_with_xatlas(
    vertices: np.ndarray, faces: np.ndarray, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run xatlas UV parameterization with deterministic chart and pack options.

    Returns `(vertex_mapping, triangle_indices, uv_coordinates)`.
    """
    np.random.seed(seed)

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)

    chart_options = xatlas.ChartOptions()
    chart_options.max_chart_area = 0.0
    chart_options.max_boundary_length = 0.0
    chart_options.normal_deviation_weight = 2.0
    chart_options.roundness_weight = 0.01
    chart_options.straightness_weight = 6.0
    chart_options.normal_seam_weight = 4.0
    chart_options.texture_seam_weight = 0.5
    chart_options.max_cost = 16.0
    chart_options.max_iterations = 1
    chart_options.use_input_mesh_uvs = False
    chart_options.fix_winding = True

    pack_options = xatlas.PackOptions()
    pack_options.max_chart_size = 0
    pack_options.padding = 4
    pack_options.texels_per_unit = 0.0
    pack_options.resolution = 2048
    pack_options.bilinear = True
    pack_options.blockAlign = True
    pack_options.bruteForce = False
    pack_options.create_image = False
    pack_options.rotate_charts_to_axis = False
    pack_options.rotate_charts = False

    atlas.generate(chart_options, pack_options, verbose=False)
    vertex_mapping, triangle_indices, uv_coordinates = atlas.get_mesh(0)
    return vertex_mapping, triangle_indices, uv_coordinates


def extract_albedo_map(mesh: trimesh.Trimesh) -> Optional[Image.Image]:
    """Extract albedo map from mesh texture visuals when available."""
    if not hasattr(mesh, "visual") or mesh.visual is None:
        return None

    if not isinstance(mesh.visual, TextureVisuals):
        return None

    albedo_map, _, _ = parse_texture_visuals(mesh.visual)
    return albedo_map


def _compute_barycentric_coords_gpu(
    points: torch.Tensor, triangle: torch.Tensor
) -> torch.Tensor:
    """Compute barycentric coordinates for many 2D points against one triangle."""
    v0 = triangle[2] - triangle[0]
    v1 = triangle[1] - triangle[0]
    v2 = points - triangle[0].unsqueeze(0)

    dot00 = torch.dot(v0, v0)
    dot01 = torch.dot(v0, v1)
    dot11 = torch.dot(v1, v1)

    dot02 = torch.sum(v0.unsqueeze(0) * v2, dim=1)
    dot12 = torch.sum(v1.unsqueeze(0) * v2, dim=1)

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-8)
    u_coord = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v_coord = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w_coord = 1.0 - u_coord - v_coord
    return torch.stack([w_coord, v_coord, u_coord], dim=1)


def _sample_texture_bilinear_gpu(
    texture: torch.Tensor, x_coord: torch.Tensor, y_coord: torch.Tensor
) -> torch.Tensor:
    """Sample RGB values from texture using batched bilinear interpolation on GPU."""
    tex_h, tex_w = texture.shape[:2]

    x_coord = torch.clamp(x_coord, 0, tex_w - 1)
    y_coord = torch.clamp(y_coord, 0, tex_h - 1)

    x0 = torch.floor(x_coord).long()
    x1 = torch.clamp(x0 + 1, 0, tex_w - 1)
    y0 = torch.floor(y_coord).long()
    y1 = torch.clamp(y0 + 1, 0, tex_h - 1)

    wx = x_coord - x0.float()
    wy = y_coord - y0.float()

    c00 = texture[y0, x0]
    c01 = texture[y0, x1]
    c10 = texture[y1, x0]
    c11 = texture[y1, x1]

    c0 = c00 * (1 - wx).unsqueeze(-1) + c01 * wx.unsqueeze(-1)
    c1 = c10 * (1 - wx).unsqueeze(-1) + c11 * wx.unsqueeze(-1)
    color = c0 * (1 - wy).unsqueeze(-1) + c1 * wy.unsqueeze(-1)
    return color.to(torch.uint8)


def rasterize_triangles_on_gpu(
    triangle_uvs: torch.Tensor,
    source_uvs: torch.Tensor,
    albedo_texture: torch.Tensor,
    output_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Rasterize UV triangles on GPU and sample source albedo into output texture buffer.

    Returns an RGBA uint8 texture buffer of shape `[output_size, output_size, 4]`.
    """
    texture_buffer = torch.zeros((output_size, output_size, 4), dtype=torch.uint8, device=device)
    texture_h, texture_w = albedo_texture.shape[:2]
    triangle_pixels = triangle_uvs * (output_size - 1)

    batch_size = 50000
    for batch_start in range(0, len(triangle_uvs), batch_size):
        batch_end = min(batch_start + batch_size, len(triangle_uvs))
        batch_triangle_pixels = triangle_pixels[batch_start:batch_end]
        batch_source_uvs = source_uvs[batch_start:batch_end]

        min_coords = torch.floor(batch_triangle_pixels.min(dim=1)[0]).long()
        max_coords = torch.ceil(batch_triangle_pixels.max(dim=1)[0]).long()
        min_coords = torch.clamp(min_coords, 0, output_size - 1)
        max_coords = torch.clamp(max_coords, 0, output_size - 1)

        for triangle_pixels_2d, triangle_source_uvs, min_coord, max_coord in zip(
            batch_triangle_pixels, batch_source_uvs, min_coords, max_coords
        ):
            min_x, min_y = min_coord
            max_x, max_y = max_coord
            if min_x >= max_x or min_y >= max_y:
                continue

            bbox_h = max_y - min_y + 1
            bbox_w = max_x - min_x + 1
            if bbox_h <= 0 or bbox_w <= 0:
                continue

            bbox_y, bbox_x = torch.meshgrid(
                torch.arange(min_y, max_y + 1, device=device),
                torch.arange(min_x, max_x + 1, device=device),
                indexing="ij",
            )
            bbox_pixels = torch.stack([bbox_x, bbox_y], dim=-1).float()
            bbox_flat = bbox_pixels.reshape(-1, 2) + 0.5

            barycentric = _compute_barycentric_coords_gpu(bbox_flat, triangle_pixels_2d)
            inside_mask = torch.all(barycentric >= -1e-6, dim=1)
            if not inside_mask.any():
                continue

            valid_barycentric = barycentric[inside_mask]
            valid_positions = bbox_flat[inside_mask]

            interpolated_uvs = torch.sum(
                valid_barycentric.unsqueeze(-1) * triangle_source_uvs.unsqueeze(0), dim=1
            )
            u_coords = interpolated_uvs[:, 0] * (texture_w - 1)
            v_coords = (1 - interpolated_uvs[:, 1]) * (texture_h - 1)
            sampled_colors = _sample_texture_bilinear_gpu(albedo_texture, u_coords, v_coords)

            pixel_positions = valid_positions.long()
            valid_mask = (
                (pixel_positions[:, 0] >= 0)
                & (pixel_positions[:, 0] < output_size)
                & (pixel_positions[:, 1] >= 0)
                & (pixel_positions[:, 1] < output_size)
            )
            if valid_mask.any():
                valid_pixel_positions = pixel_positions[valid_mask]
                valid_sampled_colors = sampled_colors[valid_mask]
                texture_buffer[
                    valid_pixel_positions[:, 1], valid_pixel_positions[:, 0], :3
                ] = valid_sampled_colors
                texture_buffer[
                    valid_pixel_positions[:, 1], valid_pixel_positions[:, 0], 3
                ] = 255

    return texture_buffer


def postprocess_texture_image(texture_buffer: np.ndarray, output_size: int) -> Image.Image:
    """
    Post-process baked texture with inpainting, filtering, and final resize.

    Returns an RGB PIL image.
    """
    image_bgra = texture_buffer.copy()
    if image_bgra.shape[2] == 3:
        alpha_channel = (image_bgra.sum(axis=2) > 0).astype(np.uint8) * 255
        image_bgra = np.dstack([image_bgra, alpha_channel])

    mask = (image_bgra[:, :, 3] == 0).astype(np.uint8) * 255
    image_bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)
    inpainted_bgr = cv2.inpaint(image_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    inpainted_bgra = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2BGRA)
    image_texture = Image.fromarray(inpainted_bgra[::-1])

    image_texture = image_texture.filter(ImageFilter.MedianFilter(size=3))
    image_texture = image_texture.filter(ImageFilter.GaussianBlur(radius=1))
    image_texture = image_texture.resize((output_size, output_size), Image.LANCZOS)
    if image_texture.mode == "RGBA":
        image_texture = image_texture.convert("RGB")
    return image_texture


def save_uv_wireframe_image(mesh: trimesh.Trimesh, output_path: str, image_size: int = 1024) -> None:
    """Save UV wireframe visualization for quick topology inspection."""
    if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
        return

    uv_coords = mesh.visual.uv
    faces = mesh.faces
    canvas = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

    for face in faces:
        if all(idx < len(uv_coords) for idx in face):
            triangle_uv = uv_coords[face]
            triangle_pixels = (triangle_uv * (image_size - 1)).astype(int)
            triangle_pixels = np.clip(triangle_pixels, 0, image_size - 1)
            for i in range(3):
                pt1 = tuple(triangle_pixels[i])
                pt2 = tuple(triangle_pixels[(i + 1) % 3])
                cv2.line(canvas, pt1, pt2, (0, 0, 0), 1)

    canvas = cv2.flip(canvas, 0)
    Image.fromarray(canvas).save(output_path)
