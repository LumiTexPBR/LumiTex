"""
Large-scale View Synthesis Model (LVSM) Inference Script

Performs multi-view synthesis inference using a pre-trained LVSM model.
Loads rendered images, normal maps, and coordinate maps, then generates novel views.
"""

import argparse
import os
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import omegaconf
import torch
import torchvision.transforms as transforms
from PIL import Image

from .model import LaCTLVSM

import warnings
warnings.filterwarnings("ignore")

NUM_INPUT_VIEWS = 6
SEED = 95


def resize_and_crop(image: Image.Image, target_size: Tuple[int, int], fxfycxcy: List[float]) -> Tuple[Image.Image, List[float]]:
    """
    Resize and crop image to target size, adjusting camera parameters accordingly.

    Args:
        image: PIL Image to resize and crop
        target_size: (width, height) tuple for target dimensions
        fxfycxcy: Camera intrinsics [fx, fy, cx, cy]

    Returns:
        Tuple of (resized_cropped_image, adjusted_camera_intrinsics)
    """
    original_width, original_height = image.size
    target_width, target_height = target_size
    fx, fy, cx, cy = fxfycxcy

    scale = max(target_width / original_width, target_height / original_height)

    new_size = (int(round(original_width * scale)), int(round(original_height * scale)))
    resized_image = image.resize(new_size, Image.LANCZOS)

    left = (new_size[0] - target_width) // 2
    top = (new_size[1] - target_height) // 2
    cropped_image = resized_image.crop((left, top, left + target_width, top + target_height))

    adjusted_intrinsics = [
        fx * scale,
        fy * scale,
        cx * scale - left,
        cy * scale - top
    ]

    return cropped_image, adjusted_intrinsics


def convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == 'RGB':
        return image
    elif image.mode == 'RGBA':
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[-1])
        return rgb_image
    else:
        return image.convert('RGB')


def load_multiview_data(data_dir: str, image_size: int, material_type: str = "albedo") -> Dict[str, torch.Tensor]:
    """
    Load multi-view images, normals, and coordinate maps from directory.

    Args:
        data_dir: Base directory containing rendered data
        image_size: Target image size (assumes square images)
        material_type: Rendering material_type for input views

    Returns:
        Dictionary containing loaded data tensors
    """
    json_path = os.path.join(data_dir, "opencv_cameras.json")
    with open(json_path, "r") as f:
        images_info = json.load(f)

    data_lists = {
        'fxfycxcy': [],
        'c2w': [],
        'image': [],
        'normal': [],
        'ccm': []
    }

    target_size = (image_size, image_size)

    for index, info in enumerate(images_info):
        fxfycxcy = [info["fx"], info["fy"], info["cx"], info["cy"]]

        w2c = torch.tensor(info["w2c"])
        c2w = torch.inverse(w2c)
        data_lists['c2w'].append(c2w)

        base_path = info["file_path"]
        paths = {
            'image': os.path.join(data_dir, base_path.replace("render_type", material_type) if index < NUM_INPUT_VIEWS else base_path),
            'normal': os.path.join(data_dir, base_path.replace("render_type", "normal")),
            'ccm': os.path.join(data_dir, base_path.replace("render_type", "ccm"))
        }

        processed_images = {}
        for img_type, img_path in paths.items():
            if img_type == 'image' and index >= NUM_INPUT_VIEWS:
                image = Image.new('RGB', target_size, (255, 255, 255))
            else:
                image = Image.open(img_path)

            if img_type == 'image':
                image, fxfycxcy = resize_and_crop(image, target_size, fxfycxcy)
            else:
                image, _ = resize_and_crop(image, target_size, fxfycxcy)

            image = convert_to_rgb(image)
            processed_images[img_type] = transforms.ToTensor()(image)

        data_lists['fxfycxcy'].append(fxfycxcy)
        for key, tensor in processed_images.items():
            data_lists[key].append(tensor)

    return {
        "fxfycxcy": torch.tensor(data_lists['fxfycxcy']),
        "c2w": torch.stack(data_lists['c2w']),
        "image": torch.stack(data_lists['image']),
        "normal": torch.stack(data_lists['normal']),
        "ccm": torch.stack(data_lists['ccm'])
    }


def save_rendered_images(rendering: torch.Tensor, output_dir: str, start_index: int = NUM_INPUT_VIEWS) -> None:
    """
    Save rendered images to disk.

    Args:
        rendering: Rendered image tensor [batch_size, num_views, 3, H, W]
        output_dir: Directory to save images
        start_index: Starting index for filename numbering
    """
    os.makedirs(output_dir, exist_ok=True)

    batch_size, num_views = rendering.shape[:2]
    for batch_idx in range(batch_size):
        for view_idx in range(num_views):
            img_tensor = rendering[batch_idx, view_idx]
            numpy_image = img_tensor.permute(1, 2, 0).cpu().numpy()
            numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)

            image = Image.fromarray(numpy_image, mode='RGB')
            filename = f"{view_idx + start_index:03d}.png"
            image.save(os.path.join(output_dir, filename))


def setup_model(config_path: str, image_size: int, model_dir: str = "ckpt/FLUX-MV-Shaded") -> LaCTLVSM:
    """
    Initialize and load the LVSM model.

    Args:
        config_path: Path to model configuration file
        image_size: Input image size to determine which checkpoint to load

    Returns:
        Loaded LVSM model
    """
    model_config = omegaconf.OmegaConf.load(config_path)
    model = LaCTLVSM(**model_config).cuda()

    checkpoint_path = os.path.join(model_dir, f"lvsm{image_size}.pth")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    return model


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LVSM Multi-view Synthesis Inference")

    parser.add_argument("--config", type=str, default="config/lact_l24_d768_ttt2x.yaml",
                        help="Path to model configuration file")
    parser.add_argument("--folder", type=str, default="case1",
                        help="Folder name containing input data")
    parser.add_argument("--mode", type=str, default="albedos",
                        help="Rendering mode for input views")
    parser.add_argument("--image_size", type=int, default=768,
                        help="Image size (assumes square images)")

    return parser.parse_args()


def main():
    """Main inference pipeline."""
    args = parse_arguments()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    model = setup_model(args.config, args.image_size)

    data_dir = f"{args.folder}/render"
    if not os.path.exists(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(parent_dir, data_dir)

    data_dict = load_multiview_data(data_dir, args.image_size, args.mode)

    for key, value in data_dict.items():
        data_dict[key] = value.unsqueeze(0).cuda()

    input_data = {key: value[:, :NUM_INPUT_VIEWS] for key, value in data_dict.items()}
    target_data = {key: value[:, NUM_INPUT_VIEWS:] for key, value in data_dict.items()}

    print("Running inference...")
    with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True), torch.no_grad():
        rendering = model(input_data, target_data)

    output_path = data_dir.replace("render", args.mode)
    save_rendered_images(rendering, output_path)
    print(f"Saved rendered images to {output_path}")


if __name__ == "__main__":
    main()
