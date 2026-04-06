"""Mesh loading, UV wrapping, and PBR material linking."""
import json
import os
from typing import Optional, Union

import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.material import PBRMaterial
import xatlas


# ============ Mesh loading ============

def load_whole_mesh(mesh_path: str, limited_faces: Optional[int] = 10_000_000) -> trimesh.Trimesh:
    """Load a mesh file and merge it into a single ``trimesh.Trimesh``.

    Raises ``AssertionError`` if face count exceeds *limited_faces*.
    """
    if limited_faces is not None:
        info = _parse_mesh_info(mesh_path)
        assert info["F"] <= limited_faces, (
            f"num faces {info['F']} is larger than limited_faces {limited_faces}"
        )
    scene = trimesh.load(mesh_path, process=False)
    return _convert_to_whole_mesh(scene)


def _convert_to_whole_mesh(scene: Union[trimesh.Trimesh, trimesh.Scene]) -> trimesh.Trimesh:
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    elif isinstance(scene, trimesh.Scene):
        geometry = scene.dump()
        mesh = geometry[0] if len(geometry) == 1 else trimesh.util.concatenate(geometry)
    else:
        raise ValueError(f"Unknown mesh type: {type(scene)}")
    mesh.merge_vertices(merge_tex=False, merge_norm=True)
    return mesh


# ============ UV wrapping ============

def mesh_uv_wrap(mesh: Union[trimesh.Trimesh, trimesh.Scene]) -> trimesh.Trimesh:
    """Parametrise *mesh* with xatlas and assign UV coordinates."""
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if len(mesh.faces) > 500_000_000:
        raise ValueError("Mesh exceeds 500M faces — not supported.")
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs
    return mesh


# ============ PBR material linking ============

def link_rgb_to_mesh(
    src_path: Union[str, trimesh.Trimesh],
    rgb_path: Union[str, Image.Image],
    dst_path: Optional[str] = None,
) -> trimesh.Trimesh:
    """Attach an RGB albedo texture to *mesh* and optionally save."""
    mesh = trimesh.load(src_path, process=False, force="mesh") if isinstance(src_path, str) else src_path
    rgb = Image.open(rgb_path) if isinstance(rgb_path, str) else rgb_path
    mesh.visual.material = PBRMaterial(
        baseColorTexture=rgb.transpose(Image.FLIP_TOP_BOTTOM),
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    mesh.merge_vertices()
    if dst_path is not None:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        mesh.export(dst_path)
    return mesh


def link_pbr_to_mesh(
    src_path: Union[str, trimesh.Trimesh],
    albedo_path: Union[str, Image.Image],
    metallic_roughness_path: Union[str, Image.Image],
    bump_path: Union[str, Image.Image],
    dst_path: Optional[str] = None,
) -> trimesh.Trimesh:
    """Attach full PBR material (albedo + MR + normal) and optionally save."""
    mesh = trimesh.load(src_path, process=False, force="mesh") if isinstance(src_path, str) else src_path
    albedo = Image.open(albedo_path) if isinstance(albedo_path, str) else albedo_path
    mr = Image.open(metallic_roughness_path) if isinstance(metallic_roughness_path, str) else metallic_roughness_path
    bump = Image.open(bump_path) if isinstance(bump_path, str) else bump_path
    mesh.visual.material = PBRMaterial(
        baseColorTexture=albedo.transpose(Image.FLIP_TOP_BOTTOM),
        metallicRoughnessTexture=mr.transpose(Image.FLIP_TOP_BOTTOM),
        normalTexture=bump.transpose(Image.FLIP_TOP_BOTTOM),
    )
    mesh.merge_vertices()
    mesh.fix_normals()
    _ = mesh.face_normals   # force computation
    _ = mesh.vertex_normals
    if dst_path is not None:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        mesh.export(dst_path)
    return mesh


# ============ GLB header parser (for face-count gating) ============

_GLTF_MAGIC = {"gltf": 1179937895, "json": 1313821514, "bin": 5130562}


def _load_mesh_header(mesh_path: str) -> dict:
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".glb":
        with open(mesh_path, "rb") as f:
            head = np.frombuffer(f.read(20), dtype="<u4")
            if head[0] != _GLTF_MAGIC["gltf"]:
                raise ValueError("incorrect header on GLB file")
            if head[1] != 2:
                raise NotImplementedError(f"only GLTF 2 is supported, got v{head[1]}")
            _, chunk_length, chunk_type = head[2:]
            if chunk_type != _GLTF_MAGIC["json"]:
                raise ValueError("no initial JSON header")
            raw = f.read(int(chunk_length))
            return json.loads(raw if isinstance(raw, str) else trimesh.util.decode_text(raw))
    if ext == ".gltf":
        with open(mesh_path, "r") as f:
            header = json.loads(f.read())
            header.pop("buffers", None)
            return header
    # unsupported extension — return empty
    return {"meshes": []}


def _parse_mesh_info(mesh_path: str) -> dict:
    h = _load_mesh_header(mesh_path)
    vl = fl = 0
    for m in h.get("meshes", []):
        for p in m.get("primitives", []):
            vl += h["accessors"][p["attributes"]["POSITION"]]["count"]
            fl += h["accessors"][p["indices"]]["count"]
    nm = len(h.get("materials", []))
    return {"V": vl, "F": fl // 3, "NC": len(h.get("meshes", [])), "NM": nm}
