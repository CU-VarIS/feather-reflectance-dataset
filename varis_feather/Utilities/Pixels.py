
import numpy as np
import cv2 as cv

from .ImageIO import RGL_tonemap_uint8

def marker_mask_from_image(image_exr_or_tonemapped: np.ndarray, morph=None, morph_amount: int = 7, margin: int = 128) -> np.ndarray:
    """
    Args:
        image_exr_or_tonemapped: Either a float EXR image or a tonemapped uint8 image
    """

    # Mask to capture just the border
    h, w, _ = image_exr_or_tonemapped.shape
    border_mask = np.ones((h, w), dtype=bool)
    border_mask[margin:-margin, margin:-margin] = False

    image_border = image_exr_or_tonemapped[border_mask]
    image_tn = image_border if image_border.dtype == np.uint8 else RGL_tonemap_uint8(image_border)
    # Extra channel for opencv
    image_tn = image_tn[:, None]
    image_tn_intensity = cv.cvtColor(image_tn, cv.COLOR_RGB2GRAY)

    # Otsu thresholding
    _, border_white_mask = cv.threshold(image_tn_intensity, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # To bool and drop opencv's channel
    border_white_mask = border_white_mask.astype(bool)[:, 0]

    # Put it back into the original shape
    full_mask = np.zeros((h, w), dtype=bool)
    full_mask[border_mask] = border_white_mask

    # Morphology open to get rid of straggler pixels
    if morph is not None:
        full_mask = cv.morphologyEx(
            full_mask.astype(np.uint8), 
            morph,
            np.ones((morph_amount, morph_amount), np.uint8)
        ).astype(bool)

    return full_mask
