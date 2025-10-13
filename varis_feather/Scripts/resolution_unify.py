
from pathlib import Path

import cv2 as cv
from tqdm import tqdm

from .. import load_standard_capture
from ..Space.file_index import FileIndex
from ..Utilities.ImageIO import readImage, writeImage

_high_res_capture_names = {
    "FeatherNorthernFlicker",
    "FeatherNorthernFlickerVentral", # weird shape
    "FeatherBlueJay",
    "FeatherRedTailedHawk",
}

_OUT_RES = (1152, 1408)

def _downsample_images(dest_dir: Path, file_index: FileIndex):
    dest_dir.mkdir(parents=True, exist_ok=True)

    for entry in tqdm(file_index, desc=f"Resizing to {dest_dir}"):

        if not entry.is_source or not entry.present_local:
            # Skip caches
            continue

        name = entry.path_local.name

        if name.endswith(".exr") or name.endswith(".jpg") or name.endswith(".png") or name.endswith(".webp"):
            img = readImage(entry.path_local)

            if img.shape[1] / img.shape[0] == _OUT_RES[1] / _OUT_RES[0]:
                img_small = cv.resize(img, _OUT_RES)
                writeImage(dest_dir / name, img_small)

            else:
                print(f"Unexpected aspect ratio {img.shape} for {entry.path_local}")



def resolution_unify(scenes: str = ""):
    captures = scenes.split(",") if scenes else _high_res_capture_names

    kw = dict(use_index=False, drop_outliers=False)

    for cap_name in captures:
        retro, olat = load_standard_capture(cap_name, mode="rectifiedHighRes", retro_kwargs=kw, olat_kwargs=kw)

        if olat:
            _downsample_images(
                dest_dir=olat.dir_src.parent / olat.dir_src.name.replace("HighRes", ""), 
                file_index=olat.file_index()
            )

        if retro:
            _downsample_images(
                dest_dir=retro.dir_src.parent / retro.dir_src.name.replace("HighRes", ""), 
                file_index=retro.file_index()
            )
