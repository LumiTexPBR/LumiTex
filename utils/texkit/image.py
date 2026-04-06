"""Image preprocessing (background removal, centering, cropping)."""
from typing import List, Optional, Tuple, Union

import numpy as np
import rembg
import rembg.sessions
from PIL import Image, ImageOps
import torch


def get_bbox(mask: np.ndarray) -> np.ndarray:
    """Return [x1, y1, x2, y2] bounding box of non-zero region in a 2-D mask."""
    assert mask.ndim == 2
    row = mask.sum(-1)
    col = mask.sum(-2)
    row_idx = np.where(row > 0)[0]
    col_idx = np.where(col > 0)[0]
    return np.array([col_idx.min(), row_idx.min(), col_idx.max(), row_idx.max()])


def preprocess(
    image: Image.Image,
    alpha: Optional[Image.Image] = None,
    H: int = 2048,
    W: int = 2048,
    scale: float = 0.8,
    color: str = "white",
    return_alpha: bool = False,
    rembg_session=None,
) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
    """Remove background, centre and resize *image* to (W, H).

    Parameters
    ----------
    image : PIL.Image
        Input image (RGB or RGBA).
    alpha : PIL.Image, optional
        Pre-computed alpha mask. If ``None``, computed via *rembg_session*.
    rembg_session
        A ``rembg.sessions.BaseSession`` **or** any callable that returns an
        RGBA ``PIL.Image`` (e.g. the ``RMBG2`` wrapper in inference.py).
    """
    image = ImageOps.exif_transpose(image)
    rgb = image.convert("RGB")

    if alpha is None:
        if (
            image.mode == "RGBA"
            and np.sum(np.array(image.getchannel("A")) > 0) < image.size[0] * image.size[1] - 8
        ):
            alpha = image.getchannel("A")
        else:
            if isinstance(rembg_session, rembg.sessions.BaseSession):
                rgba = rembg.remove(image, alpha_matting=True, session=rembg_session)
            else:
                rgba = rembg_session(image)
            alpha = rgba.getchannel("A")

    bbox = get_bbox(np.array(alpha))
    x1, y1, x2, y2 = bbox
    dy, dx = y2 - y1, x2 - x1
    s = min(H * scale / dy, W * scale / dx)
    Ht, Wt = int(dy * s), int(dx * s)
    ox, oy = int((W - Wt) / 2), int((H - Ht) / 2)
    target_box = np.array([ox, oy, ox + Wt, oy + Ht])

    rgbc = rgb.crop(bbox).resize((Wt, Ht))
    alphac = alpha.crop(bbox).resize((Wt, Ht))
    alphat = Image.new("L", (W, H))
    alphat.paste(alphac, target_box)

    out = Image.new("RGBA", (W, H), color)
    out.paste(rgbc, target_box, alphac)
    out.putalpha(alphat)

    if return_alpha:
        return out, alpha
    return out
