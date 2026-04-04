import bpy
import os
import numpy as np
from tqdm import tqdm
import logging
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


def uv_unwrap(obj):
    # https://github.com/princeton-vl/infinigen/blob/59a2574f3d6a2ab321f3e50573dddecd31b15095/infinigen/tools/export.py#L263
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    obj.data.uv_layers.new(name="ExportUV")
    bpy.context.object.data.uv_layers["ExportUV"].active = True

    logging.info("UV Unwrapping")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        # https://docs.blender.org/api/current/bpy.ops.uv.html#bpy.ops.uv.smart_project
        bpy.ops.uv.smart_project(angle_limit=1.1513)
        # bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.0078)
    except RuntimeError:
        logging.info("UV Unwrap failed, skipping mesh")
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(False)
        return False
    logging.info("UV Unwrap finished")
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return True

# Function to draw a line between two points (Bresenham's line algorithm)
def draw_line(image, x1, y1, x2, y2, color=(0.5, 0.5, 0.5)):
    """Draws an anti-aliased line between two UV points in the image (Gray)."""
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
    err = dx - dy

    while True:
        if 0 <= x1 < W and 0 <= y1 < H:  # Ensure inside image bounds
            index = y1 * W + x1  # Corrected indexing
            image[index][:3] = color
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

# Function to draw a dot at UV vertices
def draw_dot(image, x, y, color=(0, 0, 0)):
    """Draws a small black dot at UV vertices."""
    dot_size = 1  # Size of the dot
    for i in range(-dot_size, dot_size+1):
        for j in range(-dot_size, dot_size+1):
            xi, yj = x + i, y + j
            if 0 <= xi < W and 0 <= yj < H:
                index = yj * W + xi  # Corrected indexing
                image[index][:3] = color  # Set RGB to black

def make_uvgrid(W, H, pattern="shade"):
    uv_pixels = None
    if pattern == 'shade':
        grid_w, grid_h = np.meshgrid(np.linspace(0.2, 1.0, W), np.linspace(0.2, 1.0, H))
        grid_color = np.stack((grid_w, grid_h, np.ones_like(grid_w)), axis=2) 
        grid_alpha = np.ones((H, W, 1), dtype=np.float32)
        uv_pixels = np.concatenate((grid_color, grid_alpha), axis=2).reshape(H * W, 4)  # RGBA

    if pattern == 'checkerboard':
        checker_size = W // 16
        checkerboard = np.zeros((H, W, 3), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                    checkerboard[y, x] = [0.3, 0.3, 0.3]
                else:
                    checkerboard[y, x] = [0.6, 0.6, 0.6]

        grid_alpha = np.ones((H, W, 1), dtype=np.float32)
        uv_pixels = np.concatenate((checkerboard, grid_alpha), axis=2).reshape(H * W, 4)  # RGBA

    if uv_pixels is None:
        logging.info("Unsupported UV grid pattern")
    return uv_pixels

if __name__ == '__main__':
    
    obj_path = "../assets/mesh.obj"
    uv_output_path = os.path.join(os.path.dirname(obj_path), "uv_layout.png")
    obj_output_path = os.path.join(os.path.dirname(obj_path), "model.obj")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    # Import the OBJ file
    bpy.ops.wm.obj_import(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    if not uv_unwrap(obj):
        exit()

    # ==========================
    # 🎨 Generate UV Gradient Texture
    # ==========================
    W, H = 3072, 3072

    uv_pixels = make_uvgrid(W, H, pattern="checkerboard")

    uv_layer = obj.data.uv_layers.active.data

    print("🎨 Drawing UV layout...")

    # Draw UV edges
    for poly in tqdm(obj.data.polygons):
        num_vertices = len(poly.loop_indices)
        for i in range(num_vertices):
            loop_index_1 = poly.loop_indices[i]
            loop_index_2 = poly.loop_indices[(i + 1) % num_vertices]
            uv1 = uv_layer[loop_index_1].uv
            uv2 = uv_layer[loop_index_2].uv
            x1, y1 = int(uv1.x * W), int(uv1.y * H)
            x2, y2 = int(uv2.x * W), int(uv2.y * H)
            draw_line(uv_pixels, x1, y1, x2, y2, color=(0.906, 0.488, 0.051))

    # Draw UV vertices (Black)
    for poly in obj.data.polygons:
        for loop_index in poly.loop_indices:
            uv = uv_layer[loop_index].uv
            x, y = int(uv.x * W), int(uv.y * H)
            draw_dot(uv_pixels, x, y, color=(0, 0, 0))

    # * Save the Image
    # Create an image in Blender
    uv_image = bpy.data.images.new(name="UV_Unwrap", width=W, height=H, alpha=True)
    uv_pixels_blurred = cv2.GaussianBlur(uv_pixels.reshape(H, W, 4), (5, 5), 0)
    # uv_image.pixels = uv_pixels_blurred.flatten().tolist()
    uv_image.pixels = uv_pixels.flatten().tolist()
    uv_image.filepath_raw = uv_output_path
    uv_image.file_format = 'PNG'
    uv_image.save()
    logging.info(f"UV layout saved to: {uv_output_path}")
    
    # * export the obj
    bpy.ops.wm.obj_export(filepath=obj_output_path, export_uv=True)
    logging.info(f"Model exported to: {obj_output_path}")
    
    # * Create and export temporary material file
    mtl_output_path = obj_output_path.replace('.obj', '.mtl')
    with open(mtl_output_path, 'w') as f:
        f.write("newmtl material_0\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write(f"map_Kd {os.path.basename(uv_output_path)}\n")
    logging.info(f"Material file exported to: {mtl_output_path}")
