import argparse
import random
import os
import json
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
import imageio
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from transformers.optimization import get_cosine_schedule_with_warmup
from PIL import Image

# from data import NVSDataset
from .model_vae import LaCTLVSM

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

def getitem(path, size, mode):
    data_point_base_dir = path
    json_path = os.path.join(data_point_base_dir, "opencv_cameras.json")
    with open(json_path, "r") as f:
        images_info = json.load(f)
    
    fxfycxcy_list = []
    c2w_list = []
    image_list = []
    normal_list = []
    ccm_list = []
    latent1_list = []
    latent2_list = []
    latent3_list = []
    
    image_size = (size, size)

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1",  # 你也可以换成其它VAE模型
        subfolder="vae"
    ).to("cuda")


    preprocess = transforms.Compose([
        transforms.ToTensor(),  # (C, H, W), C=3
        transforms.Normalize([0.5], [0.5]),  # [-1, 1]
    ])

    for index in range(len(images_info)):
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
        if index < 6:
            image_path = image_path.replace("render/render", mode)
        image = Image.open(image_path)
        normal = Image.open(normal_path)
        ccm = Image.open(ccm_path)
        # image = image.resize((512, 512), Image.LANCZOS)
        image, fxfycxcy = resize_and_crop(image, image_size, fxfycxcy)
        normal, _ = resize_and_crop(normal, image_size, fxfycxcy)
        ccm, _ = resize_and_crop(ccm, image_size, fxfycxcy)

        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            # Create a white background and paste the RGBA image on it
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
            image = rgb_image

        elif image.mode != 'RGB':
            # Convert any other mode to RGB
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=normal.split()[-1])  # Use alpha channel as mask
            image = rgb_image

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
        img_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            latents = vae.encode(img_tensor.cuda()).latent_dist.sample().squeeze(0)  # (1, 4, 64, 64)
        image_list.append(latents[:3, :, :])
        latent1_list.append(latents[3, :, :])
        
        img_tensor = preprocess(normal).unsqueeze(0)
        with torch.no_grad():
            latents = vae.encode(img_tensor.cuda()).latent_dist.sample().squeeze(0)  # (1, 4, 64, 64)
        normal_list.append(latents[:3, :, :])
        latent2_list.append(latents[3, :, :])

        img_tensor = preprocess(ccm).unsqueeze(0)
        with torch.no_grad():
            latents = vae.encode(img_tensor.cuda()).latent_dist.sample().squeeze(0)
        ccm_list.append(latents[:3, :, :])
        latent3_list.append(latents[3, :, :])
    
    return {
        "fxfycxcy": torch.tensor(fxfycxcy_list),
        "c2w": torch.stack(c2w_list),
        "image": torch.stack(image_list),
        "normal": torch.stack(normal_list),
        "ccm": torch.stack(ccm_list),
        "latent1": torch.stack(latent1_list),
        "latent2": torch.stack(latent2_list),
        "latent3": torch.stack(latent3_list),
    }

parser = argparse.ArgumentParser()
# Basic info
parser.add_argument("--config", type=str, default="config/lact_l24_d768_ttt2x.yaml")
parser.add_argument("--load", type=str, default="/cephfs/hongzechen/FLUX_TEST/ckpt/model_0010000.pth")
parser.add_argument("--folder", type=str, default="case1")
parser.add_argument("--mode", type=str, default="albedos")
parser.add_argument("--num_all_views", type=int, default=24)

parser.add_argument("--num_input_views", type=int, default=6)
parser.add_argument("--num_target_views", type=int, default=18)
parser.add_argument("--image_size", type=int, default=768, help="Image size W, H")

args = parser.parse_args()
if args.num_target_views is None:
    args.num_target_views = args.num_all_views - args.num_input_views
model_config = omegaconf.OmegaConf.load(args.config)



# Seed everything
seed = 95
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
model = LaCTLVSM(**model_config).cuda()

if args.image_size == 512:
    modelpath = "ckpt/lvsm512.pth"
else:
    modelpath = "ckpt/lvsm768.pth"
# Load checkpoint
print(f"Loading checkpoint from {modelpath}...")
checkpoint = torch.load(modelpath, map_location="cpu")
model.load_state_dict(checkpoint["model"])

mode = args.mode
base_dir = os.environ.get("LUMITEX_DATA_DIR", ".")
data_path = os.path.join(base_dir, "assets", args.folder, "render")
data_dict = getitem(data_path, args.image_size, mode)
for key, value in data_dict.items():
    if isinstance(value, torch.Tensor):
        data_dict[key] = value.unsqueeze(0)
data_dict = {key: value.cuda() for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
input_data_dict = {key: value[:, :args.num_input_views] for key, value in data_dict.items()}
target_data_dict = {key: value[:, -args.num_target_views:] for key, value in data_dict.items()}

with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True) and torch.no_grad():
    rendering = model(input_data_dict, target_data_dict)
    target = target_data_dict["image"]
    # psnr = -10.0 * torch.log10(F.mse_loss(rendering, target)).item()

    # Save rendered images if output directory is specified
    def save_image_rgb(tensor, filepath):
        """Save tensor as RGB image."""
        numpy_image = tensor.permute(1, 2, 0).cpu().numpy()
        numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(numpy_image, mode='RGB')
        # img = img.resize((768, 768), Image.LANCZOS)
        img.save(filepath)

    batch_size, num_views = rendering.shape[:2]
    for batch_idx in range(batch_size):
        for view_idx in range(num_views):
            # Save rendered and target images
            img_tensor = rendering[batch_idx, view_idx]
            image_path = data_path.replace("render", mode)
            filename = f"{view_idx+6:03d}.png"
            save_image_rgb(img_tensor, os.path.join(image_path, filename))
        
        # print(f"Saved images for sample {sample_idx} to {output_dir}")


            
                