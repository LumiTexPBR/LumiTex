"""Image ↔ Tensor conversion utilities."""
from typing import List, Tuple, Union

import numpy as np
from PIL import Image
import torch


_MODE_TO_DTYPE = {
    "I": np.int32,
    "I;16": np.int16,
    "I;16B": np.int16,
    "F": np.float32,
}

_DTYPE_TO_FMAX = {
    np.uint8: 255.0,
    np.int16: 65535.0,
    np.int32: 4294967295.0,
    np.float32: 1.0,
}


def image_to_tensor(
    image: Union[Image.Image, List[Image.Image], Tuple[Image.Image]],
    device: str = "cuda",
) -> torch.Tensor:
    """Convert PIL Image(s) to float32 tensor in [0, 1] on *device*."""
    if isinstance(image, Image.Image):
        dtype = _MODE_TO_DTYPE.get(image.mode, np.uint8)
        arr = np.array(image, dtype=np.float32) / _DTYPE_TO_FMAX[dtype]
        arr = _normalise_channels(arr)
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    if isinstance(image, (list, tuple)):
        dtype = _MODE_TO_DTYPE.get(image[0].mode, np.uint8)
        arr = np.stack([np.array(im, dtype=np.float32) for im in image]) / _DTYPE_TO_FMAX[dtype]
        arr = _normalise_channels_batch(arr)
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    raise TypeError(f"Unsupported type: {type(image)}")


def tensor_to_image(tensor: torch.Tensor) -> Union[Image.Image, List[Image.Image]]:
    """Convert float32 tensor in [0, 1] to PIL Image(s)."""
    arr = tensor.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)

    if tensor.ndim == 3:
        return _array_to_pil(arr)
    if tensor.ndim == 4:
        return [_array_to_pil(frame) for frame in arr]

    raise ValueError(f"Expected 3D or 4D tensor, got {tensor.ndim}D")


# ---- internal helpers ----

def _normalise_channels(arr: np.ndarray) -> np.ndarray:
    """Ensure HWC with 3 channels."""
    if arr.ndim == 2:
        return np.tile(arr[:, :, None], (1, 1, 3))
    if arr.ndim == 3:
        c = arr.shape[-1]
        if c == 1:
            return np.tile(arr, (1, 1, 3))
        if c == 2:
            return np.concatenate([arr, np.ones_like(arr[..., :1])], axis=-1)
        if c > 3:
            return arr[..., :3]
    return arr


def _normalise_channels_batch(arr: np.ndarray) -> np.ndarray:
    """Ensure BHWC with 3 channels."""
    if arr.ndim == 3:
        return np.tile(arr[:, :, :, None], (1, 1, 1, 3))
    if arr.ndim == 4:
        c = arr.shape[-1]
        if c == 1:
            return np.tile(arr, (1, 1, 1, 3))
        if c == 2:
            return np.concatenate([arr, np.ones_like(arr[..., :1])], axis=-1)
        if c > 3:
            return arr[..., :3]
    return arr


def _array_to_pil(arr: np.ndarray) -> Image.Image:
    c = arr.shape[-1]
    if c == 1:
        return Image.fromarray(arr[:, :, 0], mode="L")
    if c == 3:
        return Image.fromarray(arr, mode="RGB")
    if c == 4:
        return Image.fromarray(arr, mode="RGBA")
    raise NotImplementedError(f"Unsupported channel count: {c}")
