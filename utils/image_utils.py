from PIL import Image
from typing import List, Union, Tuple
import torch
import numpy as np

mode_to_dtype = {
    "I": np.int32,
    "I;16": np.int16,
    "I;16B": np.int16,
    "F": np.float32,
}
dtype_to_fmax = {
    np.uint8: 255.0,
    np.int16: 65535.0,
    np.int32: 4294967295.0,
    np.float32: 1.0,
}


def _fast_save(image: Image.Image, path: str, format: str = None, quality: int = 95):
    """Fast save image with optional format auto-detection from path extension."""
    if format is None:
        format = path.rsplit('.', 1)[-1] if '.' in path else "webp"
    format = format.lower()
    if format == "png":
        try:
            import fpng_py # pylint: disable=import-error
            HAS_FPNG = True
        except ImportError:
            HAS_FPNG = False
        if HAS_FPNG:
            img_rgb = image.convert("RGB") if image.mode != "RGB" else image
            img_np = np.ascontiguousarray(np.array(img_rgb, dtype=np.uint8))
            h, w, c = img_np.shape
            png_bytes = fpng_py.fpng_encode_image_to_memory(img_np.tobytes(), w, h, c)
            with open(path, "wb") as f:
                f.write(png_bytes)
        else:
            image.save(path, "PNG", compress_level=1)
    elif format in ("jpg", "jpeg"):
        rgb_image = image.convert("RGB") if image.mode == "RGBA" else image
        rgb_image.save(path, "JPEG", quality=quality, subsampling=0)
    elif format == "webp":
        image.save(path, "WEBP", quality=quality, method=0)
    else:
        image.save(path)


def make_image_grid(images: List[Image.Image], rows: int, cols: int, resize: int = None, background: str = "white") -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h), color=background)

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def grid_to_images(grid_img, n_rows=2, n_cols=3):
    if grid_img.mode == 'L':
        grid_img = grid_img.convert('RGB')
    w, h = grid_img.size
    img_h, img_w = h // n_rows, w // n_cols
    images = []
    for row in range(n_rows):
        for col in range(n_cols):
            left = col * img_w
            top = row * img_h
            right = left + img_w
            bottom = top + img_h
            images.append(grid_img.crop((left, top, right, bottom)))
    return images


def image_to_tensor(image:Union[Image.Image, List[Image.Image], Tuple[Image.Image]], device='cuda') -> torch.Tensor:
    if isinstance(image, Image.Image):
        dtype = mode_to_dtype.get(image.mode, np.uint8)
        # NOTE: division may cause overflow if do not convert to float32 here
        image = np.array(image, dtype=np.float32)
        image = image / dtype_to_fmax[dtype]
        if image.ndim == 2:
            image = np.tile(image[:, :, None], (1, 1, 3))
        elif image.ndim == 3:
            if image.shape[-1] == 1:
                image = np.tile(image, (1, 1, 3))
            if image.shape[-1] == 2:
                image = np.concatenate([image, np.ones_like(image[..., [0]])], axis=-1)
            if image.shape[-1] > 3:
                image = image[:, :, :3]
        else:
            raise NotImplementedError(f'image.ndim {image.ndim} is not supported')
        tensor = torch.as_tensor(image, dtype=torch.float32, device=device)
    elif isinstance(image, (List, Tuple)):
        dtype = mode_to_dtype.get(image[0].mode, np.uint8)
        # NOTE: division may cause overflow if do not convert to float32 here
        image = np.stack([np.array(im, dtype=np.float32) for im in image], axis=0)
        image = image / dtype_to_fmax[dtype]
        if image.ndim == 3:
            image = np.tile(image[:, :, :, None], (1, 1, 1, 3))
        elif image.ndim == 4:
            if image.shape[-1] == 1:
                image = np.tile(image, (1, 1, 1, 3))
            if image.shape[-1] == 2:
                image = np.concatenate([image, np.ones_like(image[..., [0]])], axis=-1)
            if image.shape[-1] > 3:
                image = image[:, :, :, :3]
        else:
            raise NotImplementedError(f'image.ndim {image.ndim} is not supported')
        tensor = torch.as_tensor(image, dtype=torch.float32, device=device)
    return tensor


def tensor_to_image(tensor:torch.Tensor) -> Union[Image.Image, List[Image.Image]]:
    if tensor.ndim == 3:
        image = tensor.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        if image.shape[-1] == 1:
            image = Image.fromarray(image[:, :, 0], mode='L')
        elif image.shape[-1] == 3:
            image = Image.fromarray(image, mode='RGB')
        elif image.shape[-1] == 4:
            image = Image.fromarray(image, mode='RGBA')
        else:
            raise NotImplementedError(f'num of channels error: {image.shape[-1]}')
    elif tensor.ndim == 4:
        image = tensor.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        if image.shape[-1] == 1:
            image = [Image.fromarray(im[:, :, 0], mode='L') for im in image]
        elif image.shape[-1] == 3:
            image = [Image.fromarray(im, mode='RGB') for im in image]
        elif image.shape[-1] == 4:
            image = [Image.fromarray(im, mode='RGBA') for im in image]
        else:
            raise NotImplementedError(f'num of channels error: {image.shape[-1]}')
    return image


def extract_channel(image: Union[Image.Image, List[Image.Image]], channel: int) -> Union[Image.Image, List[Image.Image]]:
    if isinstance(image, Image.Image):
        image = np.array(image)
        return Image.fromarray(image[:, :, channel])
    elif isinstance(image, List):
        return [Image.fromarray(np.array(im)[:, :, channel]) for im in image]


# # For physical luminance (actual light intensity)
# def srgb_to_linear(x, gamma=2.2):
#     return x ** gamma

# def linear_to_srgb(x, gamma=2.2):
#     return x ** (1/gamma)

def srgb_to_linear(x: Image.Image) -> Image.Image:
    """sRGB [0,255] -> Linear [0,255], preserves alpha channel if present"""
    arr = np.array(x, dtype=np.float32) / 255.0
    if arr.ndim == 3 and arr.shape[2] == 4:  # RGBA
        rgb = arr[..., :3]
        alpha = arr[..., 3:4]
        linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        linear = np.concatenate([linear_rgb, alpha], axis=-1)
    else:  # RGB or grayscale
        linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    return Image.fromarray((linear * 255).clip(0, 255).astype(np.uint8))

def linear_to_srgb(x: Image.Image) -> Image.Image:
    """Linear [0,255] -> sRGB [0,255], preserves alpha channel if present"""
    arr = np.array(x, dtype=np.float32) / 255.0
    if arr.ndim == 3 and arr.shape[2] == 4:  # RGBA
        linear_rgb = arr[..., :3]
        alpha = arr[..., 3:4]
        srgb_rgb = np.where(linear_rgb <= 0.0031308, linear_rgb * 12.92, 1.055 * (linear_rgb ** (1/2.4)) - 0.055)
        srgb = np.concatenate([srgb_rgb, alpha], axis=-1)
    else:  # RGB or grayscale
        srgb = np.where(arr <= 0.0031308, arr * 12.92, 1.055 * (arr ** (1/2.4)) - 0.055)
    return Image.fromarray((srgb * 255).clip(0, 255).astype(np.uint8))


def get_luminance(img: Image.Image, mask: Image.Image = None, to_linear: bool = True, gamma: float = 2.2) -> float:
    """
    Args:
        img: PIL.Image
        mask: PIL.Image (optional)
        to_linear: bool, calculate in linear space
    Returns:
        float: mean luminance [0, 1]
    """
    img_arr = np.array(img, dtype=np.float32) / 255.0
    
    # PIL.Image to np.array
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    elif img_arr.shape[-1] == 4:
        img_arr = img_arr[:, :, :3]
    if to_linear:
        img_arr = srgb_to_linear(img_arr, gamma=gamma)
    
    # Luminance: ITU-R BT.709
    Y = 0.2126 * img_arr[:, :, 0] + 0.7152 * img_arr[:, :, 1] + 0.0722 * img_arr[:, :, 2]
    
    if mask is not None:
        mask_arr = np.array(mask, dtype=np.float32)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        mask_arr = mask_arr / mask_arr.max() if mask_arr.max() > 0 else mask_arr
        valid_pixels = mask_arr > 0.5
        if valid_pixels.sum() > 0:
            luminance = (Y * valid_pixels).sum() / valid_pixels.sum()
        else:
            luminance = 0.0
    else:
        luminance = Y.mean()
    
    return float(luminance)


def batch_luminance_scaling(
    images: List[Image.Image],
    masks: List[Image.Image],
    ref_image: Image.Image = None,
    ref_mask: Image.Image = None,
    target_luminance: float = None,
    gamma: float = 2.2,
) -> Tuple[List[Image.Image], float]:
    """
    Scale a batch of images to match a target luminance.
    All images are scaled by the same factor to preserve relative brightness.
    
    Args:
        images: List[PIL.Image], batch of RGB images
        masks: List[PIL.Image], corresponding masks (required)
        ref_image: PIL.Image (optional), reference image to get target luminance
        ref_mask: PIL.Image (optional), mask for reference image
        target_luminance: float (optional), if not provided, will use ref_image's luminance
        gamma: float, gamma value for sRGB conversion
    
    Returns:
        Tuple[List[PIL.Image], float]: scaled images and the scale factor used
    """
    assert len(images) == len(masks), "images and masks must have the same length"
    
    # Get target luminance from reference image or use provided value
    if target_luminance is None:
        if ref_image is not None:
            target_luminance = get_luminance(ref_image, mask=ref_mask, to_linear=True, gamma=gamma)
        else:
            raise ValueError("Either target_luminance or ref_image must be provided")
    
    # Calculate combined luminance of all batch images
    total_luminance_sum = 0.0
    total_valid_pixels = 0
    
    for img, mask in zip(images, masks):
        img_arr = np.array(img, dtype=np.float32) / 255.0
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        elif img_arr.shape[-1] == 4:
            img_arr = img_arr[:, :, :3]
        
        # Convert to linear space
        img_linear = srgb_to_linear(img_arr, gamma=gamma)
        Y = 0.2126 * img_linear[:, :, 0] + 0.7152 * img_linear[:, :, 1] + 0.0722 * img_linear[:, :, 2]
        
        # Apply mask
        mask_arr = np.array(mask, dtype=np.float32)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        mask_arr = mask_arr / mask_arr.max() if mask_arr.max() > 0 else mask_arr
        valid_pixels = mask_arr > 0.5
        
        total_luminance_sum += (Y * valid_pixels).sum()
        total_valid_pixels += valid_pixels.sum()
    
    # Calculate current average luminance
    if total_valid_pixels > 0:
        current_luminance = total_luminance_sum / total_valid_pixels
    else:
        current_luminance = 0.0
    
    # Calculate scale factor
    if current_luminance > 0:
        scale_factor = target_luminance / current_luminance
    else:
        scale_factor = 1.0
    
    # Apply scale factor to all images
    scaled_images = []
    for img in images:
        img_arr = np.array(img, dtype=np.float32) / 255.0
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        elif img_arr.shape[-1] == 4:
            alpha = img_arr[:, :, 3:4]
            img_arr = img_arr[:, :, :3]
        else:
            alpha = None
        
        # Convert to linear, scale, then back to sRGB
        img_linear = srgb_to_linear(img_arr, gamma=gamma)
        img_scaled = (img_linear * scale_factor).clip(0, 1)
        img_srgb = linear_to_srgb(img_scaled, gamma=gamma)
        
        # Convert back to PIL.Image
        img_uint8 = (img_srgb * 255).clip(0, 255).astype(np.uint8)
        if alpha is not None:
            alpha_uint8 = (alpha * 255).clip(0, 255).astype(np.uint8)
            img_uint8 = np.concatenate([img_uint8, alpha_uint8], axis=-1)
            scaled_images.append(Image.fromarray(img_uint8, mode='RGBA'))
        else:
            scaled_images.append(Image.fromarray(img_uint8, mode='RGB'))
    
    return scaled_images, float(scale_factor)


def adjust_albedo_tone_mapping(albedos, masks, target_luminance=0.4, min_threshold=0.05, scale_factor=None, gamma=2.2):
    """
    Adjust albedo values to have consistent tone mapping for intrinsic decomposition.
    Converts to linear space for accurate processing, then back to sRGB.
    
    Args:
        albedos: torch.Tensor (B, 3, H, W), albedo values [0, 1] in sRGB space
        masks: torch.Tensor (B, 1, H, W), object masks [0, 1]
        target_luminance: float, target average luminance for albedos (linear space)
        min_threshold: float, pixels below this value are considered naturally dark (linear space)
        gamma: float, gamma value for sRGB conversion
    
    Returns:
        torch.Tensor (B, 3, H, W), tone-mapped albedos in sRGB space
    """
    # First convert sRGB albedos to linear space for accurate processing
    linear_albedos = srgb_to_linear(albedos)
    
    B, C, H, W = linear_albedos.shape
    adjusted_albedos = linear_albedos.clone()
    
    for b in range(B):
        # Get current albedo and mask for this batch item
        albedo = linear_albedos[b]  # (3, H, W) in linear space
        mask = masks[b]             # (1, H, W)
        
        # Calculate current luminance in linear space (now this is accurate!)
        Y = 0.2126 * albedo[0] + 0.7152 * albedo[1] + 0.0722 * albedo[2]
        
        # Only consider pixels within the mask
        valid_pixels = mask[0] > 0
        if valid_pixels.sum() == 0:
            continue
            
        # Calculate mean luminance of valid pixels
        curr_lum = (Y * valid_pixels.float()).sum() / valid_pixels.sum()
        
        # Create a mask for naturally dark pixels (threshold also in linear space)
        dark_mask = (Y < min_threshold) & valid_pixels
        
        # Create a mask for pixels that should be scaled
        scale_mask = (Y >= min_threshold) & valid_pixels
        
        if scale_mask.sum() > 0 and curr_lum > 0:
            if scale_factor is not None:
                final_scale_factor = scale_factor
            else:
                final_scale_factor = target_luminance / curr_lum
                print("Image-wise tone mapping may cause inconsistency.")
            
            for c in range(3):
                adjusted_albedos[b, c] = torch.where(
                    scale_mask,
                    (albedo[c] * final_scale_factor).clamp(0, 1),
                    albedo[c]
                )
    
    # Convert back to sRGB space for output
    srgb_adjusted = linear_to_srgb(adjusted_albedos, gamma=gamma)
    return srgb_adjusted

