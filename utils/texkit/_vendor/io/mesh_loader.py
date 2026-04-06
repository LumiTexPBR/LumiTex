from typing import Optional, Union
import trimesh
from .mesh_header_loader import parse_mesh_info
import xatlas

def convert_to_whole_mesh(scene:Union[trimesh.Trimesh, trimesh.Scene]):
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    elif isinstance(scene, trimesh.Scene):
        # NOTE: bake scene.graph to scene.geometry
        geometry = scene.dump()
        if len(geometry) == 1:
            mesh = geometry[0]
        else:  # NOTE: missing some attributes
            mesh = trimesh.util.concatenate(geometry)
    else:
        raise ValueError(f"Unknown mesh type.")
    mesh.merge_vertices(merge_tex=False, merge_norm=True)
    return mesh


def load_whole_mesh(mesh_path, limited_faces:Optional[int]=10_000_000) -> trimesh.Trimesh:
    # NOTE: skip large file by header
    if limited_faces is not None:
        num_faces = parse_mesh_info(mesh_path)['F']
        assert num_faces <= limited_faces, \
            f'num faces {num_faces} is larger than limited_faces {limited_faces}'
    scene = trimesh.load(mesh_path, process=False)
    mesh = convert_to_whole_mesh(scene)
    return mesh

def mesh_uv_wrap_deterministic(mesh, seed=42):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if len(mesh.faces) > 500000000:
        raise ValueError("The mesh has more than 500,000,000 faces, which is not supported.")

    options = xatlas.ChartOptions()
    options.max_cost = 2.0
    options.max_boundary_length = 0.0
    options.normal_deviation_weight = 2.0
    options.roundness_weight = 0.01
    options.straightness_weight = 6.0
    options.normal_seam_weight = 4.0
    options.texture_seam_weight = 0.5
    
    pack_options = xatlas.PackOptions()
    pack_options.resolution = 0  # 自动分辨率
    pack_options.bias_angle = 0.0
    pack_options.max_chart_size = 0
    pack_options.padding = 1
    
    atlas = xatlas.Atlas()
    atlas.set_progress_callback(None)
    
    atlas.add_mesh(mesh.vertices, mesh.faces)
    
    atlas.generate(chart_options=options, pack_options=pack_options)
    
    vmapping, indices, uvs = atlas[0]
    
    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs

    return mesh


def mesh_uv_wrap(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if len(mesh.faces) > 500000000:
        raise ValueError("The mesh has more than 500,000,000 faces, which is not supported.")

    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs

    return mesh
