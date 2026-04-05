import numpy as np
from typing import Tuple
import logging
import bpy
import os
from utils import get_logger

def fix_bump_color_space(object):
    """
    Fix bump map color space to non-color space to avoid color shift.
    :param object: blender object containing the material to be converted
    """
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    if len(node.inputs["Normal"].links) > 0:
                        l = node.inputs["Normal"].links[0]
                        if l.from_socket.name == 'Normal':
                            normal_vector_node = l.from_node
                            if len(normal_vector_node.inputs["Color"].links) > 0:
                                l_bump = normal_vector_node.inputs["Color"].links[0]
                                if l_bump.from_socket.name == 'Color':
                                    bump_iamge_node = l_bump.from_node
                                    bump_iamge_node.image.colorspace_settings.name = "Non-Color"


def fix_material_space(object, input_type="Metallic"):
    """
    Fix input material image color space to non-color space to avoid color shift.
    :param object: blender object containing the material to be converted
    :param input_type: input type of the bump map, can be 'Metallic', 'Roughness', 'Specular'
    """
    version_info = bpy.app.version
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    # blender changes BSDF specular api after version 4.0
                    if version_info[0] >= 4:
                        if input_type == "Specular":
                            input_type = "Specular IOR Level"
                    if len(node.inputs[input_type].links) > 0:
                        l = node.inputs[input_type].links[0]
                        if l.from_socket.name == 'Color':
                            material_image_node = l.from_node
                            material_image_node.image.colorspace_settings.name = "Non-Color"


def remove_image_linkage(object, material_input_type: str = "Roughness",
                         remove_tex_image: bool = True):
    """
    Remove the image linkage and the tex_image node (if needed) from the object's material
    :param object: blender object containing the material to be converted
    :param material_input_type: the name of the input slot on BSDF node that needs to be removed
    :param remove_tex_image: if true, remove the tex_image node as well
    """
    version_info = bpy.app.version
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            nodes = node_tree.nodes
            links = node_tree.links
            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    # blender changes BSDF specular api after version 4.0
                    if version_info[0] >= 4:
                        if material_input_type == "Specular":
                            material_input_type = "Specular IOR Level"
                    if len(node.inputs[material_input_type].links) > 0:
                        l = node.inputs[material_input_type].links[0]
                        original_tex_image_node = l.from_node
                        if l is not None:
                            links.remove(l)
                        if remove_tex_image:
                            if original_tex_image_node is not None:
                                nodes.remove(original_tex_image_node)
                    if isinstance(node.inputs[material_input_type].default_value, float):
                        node.inputs[material_input_type].default_value = 0
                    else:
                        node.inputs[material_input_type].default_value = (0, 0, 0, 1)


# https://github.com/Tencent/Tencent-XR-3DGen/tree/main/data/main_data_pipeline/render_color_depth_normal_helper.py#L231
def pbr_render_shader(object):
    """
    Generate rendering shader for roughness and metallic;
    rendered image has color [R (always 1.0), G (roughness), B (metallic)].
    :param object: blender object containing the material to be converted
    """
    version_info = bpy.app.version
    remove_image_linkage(object, material_input_type='Base Color', remove_tex_image=False)
    # blender changed BSDF emission api after version 4.0
    if version_info[0] > 3:
        remove_image_linkage(object, material_input_type='Emission Color', remove_tex_image=False)
    else:
        remove_image_linkage(object, material_input_type='Emission', remove_tex_image=False)

    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            newly_made_bsdf_list = []
            for node in node_tree.nodes:
                if "BSDF" in node.type:
                    if node in newly_made_bsdf_list:
                        continue
                    new_bsdf_shader = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
                    newly_made_bsdf_list.append(new_bsdf_shader)

                    if not node.outputs["BSDF"].is_linked:
                        continue
                    output_link = node.outputs["BSDF"].links[0]
                    output_socket = output_link.to_socket
                    node_tree.links.remove(output_link)
                    node_tree.links.new(new_bsdf_shader.outputs["BSDF"], output_socket) # BSDF -> Material Output

                    # follow the common standard to use one RGB image as material image
                    combine_node = node_tree.nodes.new(type="ShaderNodeCombineColor") # [R, G, B] -> Color
                    combine_node.inputs['Red'].default_value = 1.0
                    if "Roughness" in node.inputs:
                        if node.inputs["Roughness"].is_linked:
                            color_origin_socket = node.inputs["Roughness"].links[0].from_socket
                            node_tree.links.new(color_origin_socket, combine_node.inputs['Green'])
                        else:
                            combine_node.inputs['Green'].default_value = node.inputs["Roughness"].default_value
                    if "Metallic" in node.inputs:
                        if node.inputs["Metallic"].is_linked:
                            color_origin_socket = node.inputs["Metallic"].links[0].from_socket
                            node_tree.links.new(color_origin_socket, combine_node.inputs['Blue'])
                        else:
                            combine_node.inputs['Blue'].default_value = node.inputs["Metallic"].default_value

                    new_bsdf_shader.inputs["Base Color"].default_value = (0, 0, 0, 1)
                    new_bsdf_shader.inputs["Emission Strength"].default_value = 1
                    if version_info[0] > 3:
                        node_tree.links.new(combine_node.outputs["Color"], new_bsdf_shader.inputs["Emission Color"])
                    else:
                        node_tree.links.new(combine_node.outputs["Color"], new_bsdf_shader.inputs["Emission"])

                    if "Alpha" in node.inputs:
                        if node.inputs["Alpha"].is_linked:
                            color_origin_socket = node.inputs["Alpha"].links[0].from_socket
                            node_tree.links.new(color_origin_socket, new_bsdf_shader.inputs['Alpha'])
                        else:
                            new_bsdf_shader.inputs['Alpha'].default_value = node.inputs["Alpha"].default_value


def change_hdr_map_path(hdr_map_path: str):
    """
    Change current hdr map in blender world to the given path.
    :param hdr_map_path: path to the hdr map, should be an .hdr file, for instance: https://polyhaven.com/hdris
    """
    hdr_file = hdr_map_path
    if len(hdr_file) < 1 or not os.path.exists(hdr_file):
        hdr_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'irrmaps/aerodynamics_workshop_2k.hdr')
        
    hdr_image = bpy.data.images.load(hdr_file)

    # setup scene (world) texture
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # remove current nodes as we need to reduce effects of other output heads
    for node in nodes:
        nodes.remove(node)

    # create new nodes for env map
    environment_texture_node = nodes.new("ShaderNodeTexEnvironment")
    environment_texture_node.image = hdr_image
    output_node = nodes.new("ShaderNodeOutputWorld")

    # connect the nodes
    links.new(environment_texture_node.outputs["Color"], output_node.inputs["Surface"])
    bpy.context.scene.render.film_transparent = True


def join_list_of_mesh(mesh_list: list):
    """
    join a list of meshes into a single mesh
    :param mesh_list: list of meshes
    """
    if len(mesh_list) <= 0:
        raise ValueError("mesh_list must contain at least one element")
    if len(mesh_list) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for ind, obj in enumerate(mesh_list):
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        bpy.ops.object.join()
        joint_mesh = bpy.context.object
    else:
        joint_mesh = mesh_list[0]
    return joint_mesh


# change transformation materix in opencv coordinates to blender coordinates
def opencv_to_blender(T):
    """
    change transformation materix in opencv coordinates to blender coordinates
    transform a point like: new_point = np.matmul(output_transform, old_point)
    :param T: transformation matrix in opencv coordinate system, 4 * 4 numpy array
    :returns: transformation matrix in blender coordinate system, 4 * 4 numpy array
    """
    origin = np.array(((1, 0, 0, 0),
                       (0, -1, 0, 0),
                       (0, 0, -1, 0),
                       (0, 0, 0, 1)))
    return np.matmul(T, origin)  # T * origin


def blender_to_opencv(T):
    """
    change transformation materix in blender coordinates to opencv coordinates
    transform a point like: new_point = np.matmul(output_transform, old_point)
    :param T: transformation matrix in blender coordinate system, 4 * 4 numpy array
    :returns: transformation matrix in opencv coordinate system, 4 * 4 numpy array
    """
    transform = np.array(((1, 0, 0, 0),
                          (0, -1, 0, 0),
                          (0, 0, -1, 0),
                          (0, 0, 0, 1)))
    return np.matmul(T, transform)  # T * transform


def look_at(obj_camera, point):
    """
    calculate lookat matrix of a camera
    :param obj_camera: blender camera object
    :param point: point to look at (mostly center of a sphere)
    """
    loc_camera = obj_camera.location
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


if __name__ == "__main__":
    logger = get_logger("logger")
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")
