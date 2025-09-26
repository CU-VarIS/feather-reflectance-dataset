
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2 as cv
import numpy as np
from pandas import DataFrame
from tqdm import tqdm

from ..Utilities.ImageIO import RGL_tonemap_uint8
from ..Utilities.Pixels import marker_mask_from_image


def _capture_cleanup_shadows_per_frame(capture: "VarisCapture", frame: "CaptureFrame", mask_all: np.ndarray, mask_all_area: int):
    frame_measurement = capture.read_measurement_image(frame, cache=False)
    frame_tonemapped = RGL_tonemap_uint8(frame_measurement)
    frame_mask = marker_mask_from_image(frame_tonemapped, morph=cv.MORPH_OPEN, margin=128)

    board_mask_intersection = np.logical_and(frame_mask, mask_all)
    white_marker_visible_fraction = np.count_nonzero(board_mask_intersection) / mask_all_area

    marker_white_pixels = frame_measurement[board_mask_intersection]

    marker_white_intensity = np.mean(marker_white_pixels, axis=1)
    marker_white_intensity_mean = np.mean(marker_white_intensity)
    marker_white_intensity_std = np.std(marker_white_intensity)

    marker_white_rgb_mean = np.mean(marker_white_pixels, axis=0)

    # return fr_id, board_visible_fraction, marker_white_intensity_mean, marker_white_intensity_std, marker_white_rgb_mean

    frame.white_marker_intensity_mean = marker_white_intensity_mean
    frame.white_marker_intensity_std = marker_white_intensity_std
    frame.white_marker_rgb_mean = marker_white_rgb_mean
    frame.white_marker_visible_fraction = white_marker_visible_fraction
    frame.is_valid = white_marker_visible_fraction > 0.8

def capture_calculate_marker_brightness(capture: "VarisCapture"):
    with ThreadPoolExecutor() as pool:
        futures = []

        for wiid, sp in capture.stage_poses.items():
            mask_all = marker_mask_from_image(sp.all_lights_image, morph=cv.MORPH_ERODE)
            mask_all_area = np.count_nonzero(mask_all)

            for i in sp.frame_indices:
                futures.append(pool.submit(_capture_cleanup_shadows_per_frame, capture, capture.frames[i], mask_all, mask_all_area))


        for _ in tqdm(as_completed(futures), total=len(futures), desc="Extracting marker brightness"):
            ...

    capture._update_table_from_frames()