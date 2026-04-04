"""
Blender script to render images of 3D models.

jingzhibao Modified from https://github.com/3DTopia/Phidias-Diffusion/blob/main/scripts/blender_script.py
                         https://github.com/3DTopia/MaterialAnything/blob/main/rendering_scripts/blender_script_material.py
                         https://github.com/Tencent/Tencent-XR-3DGen/tree/main/data/main_data_pipeline/render_color_depth_normal_helper.py
zhenwwang  Modified from https://github.com/cvlab-columbia/zero123/blob/main/objaverse-rendering/scripts/blender_script.py

Example usage:

blender -b -P blender_script.py -- \
        --object_path test_render/toymakers-goggles.fbx \
        --output_dir output_dir \
        --num_lightenvs 1 \
        --resolution 1024 \
        --unit_sphere True \
        --camera_type ORTHO \
        --num_images 4  \
        --start_azimuth 0 \
        --start_elevation 0 \
        --material_type PBR
"""

import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np
import bpy
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple
import bmesh
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from blender_utils_helper import *

logger = get_logger("logger", log_file="logs/blender.log")

########################################################

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--num_images", type=int, default=32)
parser.add_argument("--resolution", type=int, default=1024)
parser.add_argument("--unit_sphere", type=bool, default=True)
parser.add_argument("--fov", type=int, default=30)
parser.add_argument("--start_azimuth", type=float, default=0)
parser.add_argument("--start_elevation", type=float, default=0)
parser.add_argument("--camera_type", type=str, default="ORTHO", choices=["PERSP", "ORTHO"])
parser.add_argument("--envmap_dir", type=str, required=True)
parser.add_argument("--num_lightenvs", type=int, default=1, help="Number of lighting environments to use")
parser.add_argument("--material_type", type=str, default=None, choices=[None, "PBR", "bump"], help="Material type to render, None for RGBA")
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print("===================", args.engine, "===================")

render_material = args.material_type is not None
context = bpy.context
scene = context.scene
render = scene.render
view_layer = context.view_layer
# NEW: ENABLE OPTIX
device_type = "CUDA" # or "OPTIX"
# get all devices
cycles_prefs = context.preferences.addons["cycles"].preferences
cycles_prefs.compute_device_type = device_type
cycles_prefs.get_devices()
n_dev = len(list(cycles_prefs.devices))
# logger.info(f"Found {n_dev} devices")
optix_devices = []
for dev in cycles_prefs.devices:
    # logger.info(f"Device: {dev.name} ({dev.type})")
    dev.use = False
    if dev.type == device_type:
        optix_devices.append(dev)
    if dev.type == 'CPU':
        cpu_device = dev
    # if 'Intel' in dev["name"] or 'AMD' in dev["name"]:
    #     dev["use"] = 0
    # else:
    #     optix_devices.append(dev)
    # logger.info(d["name"], ",", d["id"], ",", d["type"], ",", d["use"])

optix_devices[args.gpu_id].use = True
# cpu_device.use = True # also use cpu device

# for normal, depth, ccm rendering
scene.use_nodes = True
view_layer.use_pass_diffuse_color = True # for diffuse rendering
view_layer.use_pass_normal = True  # for normal rendering
view_layer.use_pass_z = True  # for depth rendering
view_layer.use_pass_position = True  # for ccm rendering
if render_material:
    scene.view_settings.view_transform = "Raw"  # for omr
else:
    scene.view_settings.view_transform = "Standard"  # for rgba

render.engine = args.engine
render.image_settings.color_depth = "8"
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100

SAMPLES = 128

scene.cycles.device = "GPU"
scene.cycles.samples = SAMPLES
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True # background w/o environment map


def compose_RT(R, T):
    return np.hstack((R, T.reshape(-1, 1)))


# def sample_point_on_sphere(
#     radius: float,
#     theta: float = None,
#     phi: float = None
# ) -> Tuple[float, float, float]:
#     """
#     Sample a point on the unit sphere.
#     """
#     if theta is None:
#         theta = random.random() * 2 * math.pi
#     if phi is None:
#         phi = math.acos(2 * random.random() - 1)
#     # print(theta, phi)
#     return (
#         radius * math.sin(phi) * math.cos(theta),
#         radius * math.sin(phi) * math.sin(theta),
#         radius * math.cos(phi),
#     )


# def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
#     correct = False
#     while not correct:
#         vec = np.random.uniform(-1, 1, 3)
#         #         vec[2] = np.abs(vec[2])
#         radius = np.random.uniform(radius_min, radius_max, 1)

#         vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
#         if maxz > vec[2] > minz:
#             correct = True
#     return vec


# def set_camera_location(camera, elevation, azimuth, radius):

#     ele = np.deg2rad(elevation)
#     azi = np.deg2rad(azimuth)
#     x, y, z = sample_point_on_sphere_pytorch(
#         radius,
#         theta=azi, # np.deg2rad(-90) + azi,
#         phi=ele, # np.deg2rad(90) - ele,
#     )
#     logger.debug(f"elevation: {elevation}, azimuth: {azimuth}, (x, y, z): {(x, y, z)}")

#     camera.location = x, y, z

#     # adjust orientation
#     direction = -camera.location
#     rot_quat = direction.to_track_quat("-Z", "Y")
#     camera.rotation_euler = rot_quat.to_euler()
#     return camera, (x, y, z)


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
    x = radius * math.cos(phi) * math.sin(theta)
    y = radius * math.sin(phi)
    z = radius * math.cos(phi) * math.cos(theta)
    # pytorch3d (up-axis: Y, forward-axis: -Z) -> blender (up-axis: Z, forward-axis: Y)
    x, y, z = x, -z, y
    return x, y, z


def set_camera_location(camera, elevation, azimuth, radius):
    ele = np.deg2rad(elevation)
    azi = np.deg2rad(azimuth)
    x, y, z = sample_point_on_sphere(
        radius,
        theta=azi,
        phi=ele,
    )
    # logger.debug(f"elevation: {elevation}, azimuth: {azimuth}, (x, y, z): {(x, y, z)}")

    camera.location = x, y, z

    # adjust orientation
    look_at(camera, Vector((0, 0, 0)))
    return camera, (x, y, z)


def set_camera_location_xyz(camera, _xyz):
    x, y, z = _xyz[0], _xyz[1], _xyz[2]

    camera.location = x, y, z

    # adjust orientation
    look_at(camera, Vector((0, 0, 0)))
    return camera


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    use_contact_shadow: bool = False,
    specular_factor: float = 1.0,
    radius: float = 0.25,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.use_contact_shadow = use_contact_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    if light_type=="SUN":
        light_data.angle = 0.5
    if light_type=="POINT":
        light_data.shadow_soft_size = radius
    if light_type=="AREA":
        light_data.size = radius

    return light_object


def randomize_lighting():
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """
    # Add random angle offset in 0-90
    angle_offset = random.uniform(0, math.pi / 2)

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398 + angle_offset),  # 45 0 -45
        energy=random.choice([2.5, 3.25, 4]),
        use_shadow=True,
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699 + angle_offset),  # -45 0 -225
        energy=random.choice([2.5, 3.25, 4]),
        use_shadow=True,
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619 + angle_offset),  # 45 0 135
        energy=random.choice([2, 3, 3.5]),
    )

    # Create small light
    small_light1 = _create_light(
        name="S1_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(1.57079, 0, 0.785398 + angle_offset),  # 90 0 45
        energy=random.choice([0.25, 0.5, 1]),
    )

    small_light2 = _create_light(
        name="S2_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(1.57079, 0, 3.92699 + angle_offset),  # 90 0 45
        energy=random.choice([0.25, 0.5, 1]),
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),  # 180 0 0
        energy=random.choice([1, 2, 3]),
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
        small_light1=small_light1,
        small_light2=small_light2,
    )


def add_lighting(option: str) -> None:
    assert option in ["fixed", "random"]

    # delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()

    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light = bpy.data.lights["Area"]

    if option == "fixed":
        light.energy = 30000
        bpy.data.objects["Area"].location[0] = 0
        bpy.data.objects["Area"].location[1] = 1
        bpy.data.objects["Area"].location[2] = 0.5

    elif option == "random":
        light.energy = random.uniform(80000, 120000)
        bpy.data.objects["Area"].location[0] = random.uniform(-2.0, 2.0)
        bpy.data.objects["Area"].location[1] = random.uniform(-2.0, 2.0)
        bpy.data.objects["Area"].location[2] = random.uniform(1.0, 3.0)

    # set light scale
    bpy.data.objects["Area"].scale[0] = 200
    bpy.data.objects["Area"].scale[1] = 200
    bpy.data.objects["Area"].scale[2] = 200


def delete_light():
    world = bpy.context.scene.world
    if world:
        bpy.data.worlds.remove(world, do_unlink=True)
    # delete the default light
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()


def random_envmap(envmap_paths: List[str], log: bool = False) -> None:
    # delete the default light
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    envmap_path = random.choice(envmap_paths)
    change_hdr_map_path(envmap_path)
    if log:
        logger.info(f"Changing HDR map path to {envmap_path}")


def change_pointlights(locations: List[List[Tuple[float, float]]], raidus = 2.0, energy: float = 1000) -> None:
    # remove the default world
    world = bpy.context.scene.world
    if world:
        bpy.data.worlds.remove(world, do_unlink=True)
    # delete the default light
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    # add point light
    for elevation, azimuth in locations:
        ele = np.deg2rad(elevation)
        azi = np.deg2rad(azimuth)
        x, y, z = sample_point_on_sphere(
            radius=raidus,
            theta=azi,
            phi=ele,
        )
        _create_light(
            name="Point_Light",
            light_type="POINT",
            location=(x, y, z),
            rotation=(0, 0, 0),
            energy=energy,
            use_shadow=True,
        )
    

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str, z_up: bool = False) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        # ! I am not sure if should assign forward_axis and up_axis
        if z_up:
            bpy.ops.wm.obj_import(filepath=object_path, forward_axis="Y", up_axis="Z")
        else:
            bpy.ops.wm.obj_import(filepath=object_path, forward_axis="NEGATIVE_Z", up_axis="Y")
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    max_dist = -math.inf
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
            max_dist = max(max_dist, (coord - Vector((0.0, 0.0, 0.0))).length_squared)
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max), np.sqrt(max_dist)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene_box(box_scale: float):
    bbox_min, bbox_max, _ = scene_bbox()
    scale = box_scale / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()
    bbox_min, bbox_max, _ = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")


def normalize_scene_sphere(radius: float):
    normalize_scene_box(1)

    bbox_min, bbox_max, max_dist = scene_bbox()
    center = (bbox_min + bbox_max) / 2
    scale = radius / max_dist
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()

    bbox_min, bbox_max, _ = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    # logger.debug(f"bbox_min: {bbox_min}, bbox_max: {bbox_max}, offset: {offset}")
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")


def setup_camera(camera_type="PERSP"):
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)

    # cam.data.lens = 24
    cam.data.sensor_width = 32
    cam.data.sensor_height = (
        32  # affects instrinsics calculation, should be set explicitly
    )

    # Convert FOV to radians
    assert args.fov == 30
    fov_radians = np.deg2rad(args.fov)

    cam.data.type = camera_type
    if camera_type == "PERSP":
        cam.data.angle = fov_radians # set FOV
    elif camera_type == "ORTHO":
        cam.data.ortho_scale = 1.8 # 2.0 # [-1, 1]^3

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint


def main_pipeline(
    object_file: str,
    output_dir: str,
    envmap_dir: str = "None",
    num_lightenvs: int = 1,
    extra_outs=["normal", "ccm"],
    unit_sphere=True,
    camera_type="PERSP",
    dump_pose=False,
    material_type=None,
) -> None:
    """Saves rendered images of the object in the scene."""
    reset_scene()

    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    if object_uid == "model":
        object_uid = object_file.split("/")[-3]
    object_uid = "_".join(object_uid.lower().split())

    # prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    photometric = envmap_dir == "None"
    output_dir = os.path.join(output_dir, object_uid, "pm" if photometric else "hdr")
    os.makedirs(output_dir, exist_ok=True)
    num_lightenvs = num_lightenvs
    num_images = args.num_images
    logger.info(f"Rendering {object_uid} ({'photometric' if photometric else 'hdr'}): {num_lightenvs} lightenvs, {num_images} images")
    
    # meshes
    meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            meshes.append(obj)

    try:
        # switch character to rest mode, i.e. A-pose in most case
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.loc_clear()
        bpy.ops.pose.rot_clear()
        bpy.ops.pose.scale_clear()
        bpy.ops.object.posemode_toggle()
    except:
        # just pass this if pose toggle failed
        logger.warning('posemode_toggle failed')


    # material shader nodes
    if material_type:
        for mesh in meshes:
            fix_bump_color_space(mesh)
            if material_type == "PBR":
                fix_material_space(mesh, input_type="Metallic")
                fix_material_space(mesh, input_type="Roughness")
                # TODO: normal map may not consistent with new shader.
                pbr_render_shader(mesh)
                metallic_dir = os.path.join(output_dir, "metallic")
                roughness_dir = os.path.join(output_dir, "roughness")
                os.makedirs(metallic_dir, exist_ok=True)
                os.makedirs(roughness_dir, exist_ok=True)
            elif material_type == 'bump':
                raise NotImplementedError("Bump rendering is not implemented yet")
                # bump_render_shader(mesh)
            else:
                logger.error(f"Unsupported material type {material_type}")
                raise ValueError(f"Unsupported material type {material_type}")
    else:
        for mesh in meshes:
            fix_bump_color_space(mesh)
            fix_material_space(mesh, "Metallic")
            fix_material_space(mesh, "Specular")
            fix_material_space(mesh, "Roughness")


    radius_sphere = 1.0
    if unit_sphere:
        normalize_scene_sphere(radius=radius_sphere) # r = 1.0
    else:
        normalize_scene_box(box_scale=2.0) # [-1, 1]^3
    # Mesh processing and normalization completed!

    # setup cameras & lighting
    # add_lighting(option="random")
    # randomize_lighting()
    camera, cam_constraint = setup_camera(camera_type)
    
    lighting_func = None
    if envmap_dir != "None":
        envmap_paths = os.listdir(envmap_dir)
        # shuffle the envmap paths
        random.shuffle(envmap_paths)
        logger.info(f"Using envmaps: {envmap_paths[:num_lightenvs]}")
        envmap_paths = [os.path.join(envmap_dir, envmap_path) for envmap_path in envmap_paths]
        lighting_func = change_hdr_map_path
        lighting_attr = envmap_paths[:num_lightenvs]
    elif photometric:
        lighting_func = change_pointlights
        # prepare lighting_attr
        pointlight_elevs = [-30, 0, 30]
        pointlight_azims = [315, 0, 45, 90, 135, 180, 225, 270]
        all_locations = []
        for azim in pointlight_azims:
            for elev in pointlight_elevs:
                all_locations.append([elev, azim])
        num_locations = len(all_locations)
        
        lighting_attr = []
        for i in range(num_lightenvs):
            # sample two locations, one for each hemisphere
            lighting_attr.append(
                [
                    random.choice(all_locations[:num_locations//2]),
                    random.choice(all_locations[num_locations//2:]),
                ]
            )
        logger.info(f"Sampling {len(lighting_attr)} pointlights.")
    
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    empty.location = 0, 0, 0
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # prepare to save
    if render_material:
        omr_dir = os.path.join(output_dir, "omr") # m, r
        os.makedirs(omr_dir, exist_ok=True)
    else:
        img_dir = os.path.join(output_dir, "rgba")
        os.makedirs(img_dir, exist_ok=True)
    if dump_pose:
        pose_dir = os.path.join(output_dir, "pose")
        os.makedirs(pose_dir, exist_ok=True)

    # bbox_min, bbox_max, distance = scene_bbox()
    # logger.debug(f"Object bounding box min: {bbox_min}, max: {bbox_max}, distance: {distance}")

    scene.use_nodes = False
    context.view_layer.use_pass_normal = False  # for normal rendering
    context.view_layer.use_pass_position = False  # for ccm rendering
    context.view_layer.use_pass_z = False  # for depth rendering
    camera_xyz = []
    time_0 = time.time()
    # render rgba / omr
    # TODO: adjust the elevations and azimuths here
    render_id = 0
    for light_i in range(num_lightenvs):
        if material_type == "PBR":
            delete_light()
        else:
            lighting_func(lighting_attr[light_i])
        
        # set the camera position
        fixed_elevations = np.array(
            # [0.0, 0.0, 0.0, 0.0, 20, 20, 20, 20]
            # np.arange(0, args.num_images) / (args.num_images - 1) * 45
            [20] * 3 + [0] * 3
        ) # rotate at X-axis
        fixed_azimuths = np.array(
            [0.0, 30, 330] * 2
        ) # rotate at Z-axis
        
        sample_camera_steps = num_images
    
        for i in range(sample_camera_steps):
            render_id += 1
            if i < len(fixed_elevations):
                elevation = fixed_elevations[i]
                azimuth = fixed_azimuths[i]
            else:
                while True:
                    elevation = np.random.uniform(-20, 60)
                    azimuth = np.random.uniform(0, 360)
                    if np.any((np.abs(fixed_azimuths - int(azimuth))) < 5) and np.any(
                        (np.abs(fixed_elevations - int(elevation))) < 5
                    ):
                        continue
                    else:
                        break
            radius = radius_sphere / np.tan(np.radians(args.fov / 2)) if camera_type == "PERSP" else 2.0
            camera, _xyz = set_camera_location(
                camera,
                elevation + args.start_elevation,
                azimuth + args.start_azimuth,
                radius,
            )
            # logger.info(f"Camera position: {_xyz}")
            camera_xyz.append(_xyz)

            # render the image (RGBA)
            render_path = os.path.join(omr_dir if material_type == "PBR" else img_dir, f"{render_id:03d}.png")
            scene.render.filepath = render_path
            bpy.ops.render.render(write_still=True)
            
            if material_type == "PBR":
                # parse metallic and roughness from RGBA
                metallic_path = os.path.join(metallic_dir, f"{render_id:03d}.png")
                roughness_path = os.path.join(roughness_dir, f"{render_id:03d}.png")
                # rgba_img = bpy.data.images.load(render_path) # axis not aligned!
                rgba_img = np.array(Image.open(render_path))
                
                # Extract G channel for metallic
                m_0 = rgba_img[:, :, 1]
                metallic_img = np.stack([m_0, m_0, m_0, np.ones_like(m_0)*255], axis=-1).astype(np.uint8)
                
                # Extract B channel for roughness
                r_0 = rgba_img[:, :, 2]
                roughness_img = np.stack([r_0, r_0, r_0, np.ones_like(r_0)*255], axis=-1).astype(np.uint8)
                
                # Save the extracted channels as separate images
                Image.fromarray(metallic_img).save(metallic_path)
                Image.fromarray(roughness_img).save(roughness_path)

                # Clean up
                # bpy.data.images.remove(rgba_img)

            # save camera RT matrix (C2W)
            if dump_pose:
                # location, rotation = camera.matrix_world.decompose()[0:2]
                # RT = compose_RT(rotation.to_matrix(), np.array(location))
                # RT_path = os.path.join(pose_dir, f"{render_id:03d}_mat.npy")
                # np.save(RT_path, RT)

                # raw camera pose: elevation, azimuth, radius
                pose = [elevation, azimuth, radius]
                pose_path = os.path.join(pose_dir, f"{render_id:03d}_raw.npy")
                np.save(pose_path, pose)

    time_0_ = time.time() - time_0

    scene.use_nodes = True
    context.view_layer.use_pass_normal = True  # for normal rendering
    context.view_layer.use_pass_position = True  # for ccm rendering
    context.view_layer.use_pass_z = True  # for depth rendering

    extra_dirs = {}
    for t in extra_outs:
        temp_dir = os.path.join(output_dir, t)
        os.makedirs(temp_dir, exist_ok=True)
        extra_dirs[t] = temp_dir

    # create input render layer node
    render_layers = scene.node_tree.nodes.new("CompositorNodeRLayers")
    render_layers.label = "Custom Outputs"
    render_layers.name = "Custom Outputs"
    
    if "diffuse" in extra_outs:
        # create diffuse output node
        diffuse_file_output = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        diffuse_file_output.label = "diffuse"
        diffuse_file_output.name = "diffuse"
        diffuse_file_output.base_path = extra_dirs["diffuse"]

        # add alpha channel
        set_alpha_node = scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
        set_alpha_node.mode = "REPLACE_ALPHA"
        scene.node_tree.links.new(
            render_layers.outputs["DiffCol"], set_alpha_node.inputs["Image"]
        )
        scene.node_tree.links.new(
            render_layers.outputs["Alpha"], set_alpha_node.inputs["Alpha"]
        )

        scene.node_tree.links.new(
            set_alpha_node.outputs["Image"], diffuse_file_output.inputs["Image"]
        )

    if "normal" in extra_outs:
        # create normal output node
        normal_file_output = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = "normal"
        normal_file_output.name = "normal"
        normal_file_output.base_path = extra_dirs["normal"]

        # Create a Separate RGB node
        separate_rgb_node = scene.node_tree.nodes.new(type="CompositorNodeSepRGBA")
        scene.node_tree.links.new(
            render_layers.outputs["Normal"], separate_rgb_node.inputs["Image"]
        )

        # Create a Combine RGBA node
        combine_rgba_node = scene.node_tree.nodes.new(type="CompositorNodeCombRGBA")
        reversed_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
        reversed_G.operation = "MULTIPLY"
        reversed_G.inputs[0].default_value = -1

        # y -> -y
        scene.node_tree.links.new(separate_rgb_node.outputs["G"], reversed_G.inputs[1])

        # map normal range from [-1, 1] to [0, 1]
        # normal = (normal + 1) / 2
        # channel R
        bias_node_R = scene.node_tree.nodes.new(type="CompositorNodeMath")
        bias_node_R.operation = "ADD"
        bias_node_R.inputs[0].default_value = 1

        scale_node_R = scene.node_tree.nodes.new(type="CompositorNodeMath")
        scale_node_R.operation = "MULTIPLY"
        scale_node_R.inputs[0].default_value = 0.5

        scene.node_tree.links.new(separate_rgb_node.outputs["R"], bias_node_R.inputs[1])
        scene.node_tree.links.new(bias_node_R.outputs[0], scale_node_R.inputs[1])

        # channel G
        bias_node_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
        bias_node_G.operation = "ADD"
        bias_node_G.inputs[0].default_value = 1

        scale_node_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
        scale_node_G.operation = "MULTIPLY"
        scale_node_G.inputs[0].default_value = 0.5

        scene.node_tree.links.new(reversed_G.outputs[0], bias_node_G.inputs[1])
        scene.node_tree.links.new(bias_node_G.outputs[0], scale_node_G.inputs[1])

        # channel B
        bias_node_B = scene.node_tree.nodes.new(type="CompositorNodeMath")
        bias_node_B.operation = "ADD"
        bias_node_B.inputs[0].default_value = 1

        scale_node_B = scene.node_tree.nodes.new(type="CompositorNodeMath")
        scale_node_B.operation = "MULTIPLY"
        scale_node_B.inputs[0].default_value = 0.5

        scene.node_tree.links.new(separate_rgb_node.outputs["B"], bias_node_B.inputs[1])
        scene.node_tree.links.new(bias_node_B.outputs[0], scale_node_B.inputs[1])

        # Combine RGB
        scene.node_tree.links.new(
            combine_rgba_node.inputs["R"], scale_node_R.outputs[0]
        )
        scene.node_tree.links.new(
            combine_rgba_node.inputs["G"], scale_node_B.outputs[0]
        )
        scene.node_tree.links.new(
            combine_rgba_node.inputs["B"], scale_node_G.outputs[0]
        )
        scene.node_tree.links.new(
            combine_rgba_node.inputs["A"], separate_rgb_node.outputs["A"]
        )

        # add alpha channel
        set_alpha_node = scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
        set_alpha_node.mode = "REPLACE_ALPHA"
        scene.node_tree.links.new(
            combine_rgba_node.outputs["Image"], set_alpha_node.inputs["Image"]
        )
        scene.node_tree.links.new(
            render_layers.outputs["Alpha"], set_alpha_node.inputs["Alpha"]
        )

        scene.node_tree.links.new(
            set_alpha_node.outputs["Image"], normal_file_output.inputs["Image"]
        )

    if "ccm" in extra_outs:
        # create CCM output node
        ccm_file_output = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        ccm_file_output.label = "ccm"
        ccm_file_output.name = "ccm"
        ccm_file_output.base_path = extra_dirs["ccm"]

        # Create a Separate RGB node
        separate_xyz_node = scene.node_tree.nodes.new(type="CompositorNodeSeparateXYZ")
        scene.node_tree.links.new(
            render_layers.outputs["Position"], separate_xyz_node.inputs["Vector"]
        )

        # Create a Combine RGBA node
        combine_rgba_node = scene.node_tree.nodes.new(type="CompositorNodeCombRGBA")
        reversed_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
        reversed_G.operation = "MULTIPLY"
        reversed_G.inputs[0].default_value = -1

        # y -> -y
        scene.node_tree.links.new(separate_xyz_node.outputs["Y"], reversed_G.inputs[1])

        # map ccm range from [-1, 1] to [0, 1] (normalize scence to unit box)
        # map ccm range from [-0.5, 0.5] to [0, 1] (normalize scence to unit sphere)
        # upd(unit sphere): [-1, 1] -> [0, 1]
        add = 1
        mul = 0.5

        # channel R
        bias_node_R = scene.node_tree.nodes.new(type="CompositorNodeMath")
        bias_node_R.operation = "ADD"
        bias_node_R.inputs[0].default_value = add

        scale_node_R = scene.node_tree.nodes.new(type="CompositorNodeMath")
        scale_node_R.operation = "MULTIPLY"
        scale_node_R.inputs[0].default_value = mul

        scene.node_tree.links.new(separate_xyz_node.outputs["X"], bias_node_R.inputs[1])
        scene.node_tree.links.new(bias_node_R.outputs[0], scale_node_R.inputs[1])

        # channel G
        bias_node_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
        bias_node_G.operation = "ADD"
        bias_node_G.inputs[0].default_value = add

        scale_node_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
        scale_node_G.operation = "MULTIPLY"
        scale_node_G.inputs[0].default_value = mul

        scene.node_tree.links.new(reversed_G.outputs[0], bias_node_G.inputs[1])
        scene.node_tree.links.new(bias_node_G.outputs[0], scale_node_G.inputs[1])

        # channel B
        bias_node_B = scene.node_tree.nodes.new(type="CompositorNodeMath")
        bias_node_B.operation = "ADD"
        bias_node_B.inputs[0].default_value = add

        scale_node_B = scene.node_tree.nodes.new(type="CompositorNodeMath")
        scale_node_B.operation = "MULTIPLY"
        scale_node_B.inputs[0].default_value = mul

        scene.node_tree.links.new(separate_xyz_node.outputs["Z"], bias_node_B.inputs[1])
        scene.node_tree.links.new(bias_node_B.outputs[0], scale_node_B.inputs[1])

        # Combine RGB
        scene.node_tree.links.new(
            combine_rgba_node.inputs["R"], scale_node_R.outputs[0]
        )
        scene.node_tree.links.new(
            combine_rgba_node.inputs["G"], scale_node_B.outputs[0]
        )
        scene.node_tree.links.new(
            combine_rgba_node.inputs["B"], scale_node_G.outputs[0]
        )

        # add alpha channel
        set_alpha_node = scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
        set_alpha_node.mode = "REPLACE_ALPHA"
        scene.node_tree.links.new(
            combine_rgba_node.outputs["Image"], set_alpha_node.inputs["Image"]
        )
        scene.node_tree.links.new(
            render_layers.outputs["Alpha"], set_alpha_node.inputs["Alpha"]
        )

        scene.node_tree.links.new(
            set_alpha_node.outputs["Image"], ccm_file_output.inputs["Image"]
        )

    if "Z" in extra_outs:
        # create Z output node
        z_file_output = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        z_file_output.label = "Z"
        z_file_output.name = "Z"
        z_file_output.base_path = extra_dirs["Z"]
        z_file_output.format.file_format = "OPEN_EXR"
        z_file_output.format.color_depth = "32"  # Use 32-bit float for depth
        z_file_output.format.color_mode = "RGB"

        # Directly connect the Depth output
        # Depth output is usually positive Z values in camera space
        # Use regular CombRGBA but only connect the R channel, leaving alpha disconnected
        combine_rgb_node = scene.node_tree.nodes.new(type="CompositorNodeCombRGBA")
        combine_rgb_node.inputs["A"].default_value = 1.0
        scene.node_tree.links.new(
            render_layers.outputs["Depth"], combine_rgb_node.inputs["R"]
        )
        scene.node_tree.links.new(
            render_layers.outputs["Depth"], combine_rgb_node.inputs["G"]
        )
        scene.node_tree.links.new(
            render_layers.outputs["Depth"], combine_rgb_node.inputs["B"]
        )
        
        scene.node_tree.links.new(
            combine_rgb_node.outputs["Image"], z_file_output.inputs["Image"]
        )

    time_2 = time.time()
    if len(extra_outs) > 0:
        scene.view_settings.view_transform = "Raw"
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False
        for i, _xyz in enumerate(camera_xyz):
            # set the camera position
            camera = set_camera_location_xyz(camera, _xyz)

            # render the image
            scene.render.filepath = ".cache/tmp.png"

            for out in extra_outs:
                # render_path = os.path.join(extra_dirs[out], f"{i:03d}_")
                # logger.info(f"Render path: {render_path}")
                # scene.node_tree.nodes[out].file_slots[0].path = render_path
                filename_prefix = f"{i:03d}_"
                scene.node_tree.nodes[out].file_slots[0].path = filename_prefix

            bpy.ops.render.render(write_still=True)
        scene.view_settings.view_transform = "Standard"
        scene.cycles.samples = SAMPLES
        scene.cycles.use_denoising = True
    time_2_ = time.time() - time_2

    # print("0: Time taken for rendering", time_0_, "seconds")
    # print("2: Time taken for rendering", time_2_, "seconds")
    # save the camera intrinsics
    if dump_pose:
        intrinsics = get_calibration_matrix_K_from_blender(
            camera.data, return_principles=True
        )
        with open(
            os.path.join(output_dir, "intrinsics.npy"), "wb"
        ) as f_intrinsics:
            np.save(f_intrinsics, intrinsics)


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


def get_calibration_matrix_K_from_blender(camera, return_principles=False):
    """
    Get the camera intrinsic matrix from Blender camera.
    Return also numpy array of principle parameters if specified.

    Intrinsic matrix K has the following structure in pixels:
        [fx  0 cx]
        [0  fy cy]
        [0   0  1]

    Specified principle parameters are:
        [fx, fy] - focal lengths in pixels
        [cx, cy] - optical centers in pixels
        [width, height] - image resolution in pixels

    """
    # Render resolution
    render = bpy.context.scene.render
    width = render.resolution_x * render.pixel_aspect_x
    height = render.resolution_y * render.pixel_aspect_y

    # Camera parameters
    focal_length = camera.lens  # Focal length in millimeters
    sensor_width = camera.sensor_width  # Sensor width in millimeters
    sensor_height = camera.sensor_height  # Sensor height in millimeters

    # Calculate the focal length in pixel units
    focal_length_x = width * (focal_length / sensor_width)
    focal_length_y = height * (focal_length / sensor_height)

    # Assuming the optical center is at the center of the sensor
    optical_center_x = width / 2
    optical_center_y = height / 2

    # Constructing the intrinsic matrix
    K = np.array(
        [
            [focal_length_x, 0, optical_center_x],
            [0, focal_length_y, optical_center_y],
            [0, 0, 1],
        ]
    )

    if return_principles:
        return np.array(
            [
                [focal_length_x, focal_length_y],
                [optical_center_x, optical_center_y],
                [width, height],
            ]
        )
    else:
        return K


if __name__ == "__main__":
    start_i = time.time()
    if args.object_path.startswith("http"):
        local_path = download_object(args.object_path)
    else:
        local_path = args.object_path
    main_pipeline(
        object_file=local_path, 
        output_dir=args.output_dir,
        envmap_dir=args.envmap_dir,
        num_lightenvs=args.num_lightenvs,
        extra_outs=[],
        unit_sphere=args.unit_sphere, 
        camera_type=args.camera_type,
        dump_pose=True,
        material_type=args.material_type
    )
    end_i = time.time()
    logger.info(f"Finished {local_path} in {end_i - start_i:.2f} seconds")
    # delete the object if it was downloaded
    if args.object_path.startswith("http"):
        os.remove(local_path)
