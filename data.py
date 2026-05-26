import json
import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def resize_and_crop(image, target_size, fxfycxcy):
    """
    Resize and crop image to target_size, adjusting camera parameters accordingly.
    
    Args:
        image: PIL Image
        target_size: (width, height) tuple
        fxfycxcy: [fx, fy, cx, cy] list
    
    Returns:
        tuple: (resized_cropped_image, adjusted_fxfycxcy)
    """
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    fx, fy, cx, cy = fxfycxcy
    
    # Calculate scale factor to fill target size (resize to cover)
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale = max(scale_x, scale_y)  # Use larger scale to ensure it covers the target area
    
    # Resize image
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate crop box for center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    # Crop image
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    # Adjust camera parameters
    # Scale focal lengths and principal points
    new_fx = fx * scale
    new_fy = fy * scale
    new_cx = cx * scale - left
    new_cy = cy * scale - top
    
    return cropped_image, [new_fx, new_fy, new_cx, new_cy]


class NVSDataset(Dataset):
    def __init__(self, data_path, num_views, image_size):
        """
        image_size is (w, h) or just a int (as size).
        """
        self.base_dir = data_path
        # self.data_point_paths = json.load(open(data_path, "r"))
        self.data_point_paths = os.listdir(data_path)

        self.num_views = num_views
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

    def __len__(self):
        return len(self.data_point_paths)
    
    def __getitem__(self, index):
        data_point_base_dir = os.path.join(self.base_dir, self.data_point_paths[index])
        json_path = os.path.join(data_point_base_dir, "opencv_cameras.json")
        with open(json_path, "r") as f:
            images_info = json.load(f)
        
        indices = random.sample(range(len(images_info)), self.num_views)
        
        fxfycxcy_list = []
        c2w_list = []
        image_list = []
        normal_list = []
        ccm_list = []
        
        for index in indices:
            info = images_info[index]
            
            fxfycxcy = [info["fx"], info["fy"], info["cx"], info["cy"]]
            
            w2c = torch.tensor(info["w2c"])
            c2w = torch.inverse(w2c)
            c2w_list.append(c2w)
            
            # Load image from file_path using PIL and convert to torch tensor
            image_path = os.path.join(data_point_base_dir, info["file_path"])
            s = info["file_path"]
            
            normal_path = os.path.join(data_point_base_dir, s.replace("render", "normal"))
            ccm_path = os.path.join(data_point_base_dir, s.replace("render", "ccm"))
            image = Image.open(image_path)
            normal = Image.open(normal_path)
            ccm = Image.open(ccm_path)
            
            image, fxfycxcy = resize_and_crop(image, self.image_size, fxfycxcy)
            normal, _ = resize_and_crop(normal, self.image_size, fxfycxcy)
            ccm, _ = resize_and_crop(ccm, self.image_size, fxfycxcy)

            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                # Create a white background and paste the RGBA image on it
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = rgb_image

            elif image.mode != 'RGB':
                # Convert any other mode to RGB
                image = image.convert('RGB')

            if normal.mode == 'RGBA':
                # Create a white background and paste the RGBA image on it
                rgb_normal = Image.new('RGB', normal.size, (255, 255, 255))
                rgb_normal.paste(normal, mask=normal.split()[-1])
                normal = rgb_normal
            elif normal.mode != 'RGB':
                # Convert any other mode to RGB
                normal = normal.convert('RGB')

            if ccm.mode == 'RGBA':
                # Create a white background and paste the RGBA image on it
                rgb_ccm = Image.new('RGB', ccm.size, (255, 255, 255))
                rgb_ccm.paste(ccm, mask=ccm.split()[-1])
                ccm = rgb_ccm   
            elif ccm.mode != 'RGB':
                # Convert any other mode to RGB
                ccm = ccm.convert('RGB')
            
            fxfycxcy_list.append(fxfycxcy)
            image_list.append(transforms.ToTensor()(image))
            normal_list.append(transforms.ToTensor()(normal))
            ccm_list.append(transforms.ToTensor()(ccm))
        
        return {
            "fxfycxcy": torch.tensor(fxfycxcy_list),
            "c2w": torch.stack(c2w_list),
            "image": torch.stack(image_list),
            "normal": torch.stack(normal_list),
            "ccm": torch.stack(ccm_list)
        }
