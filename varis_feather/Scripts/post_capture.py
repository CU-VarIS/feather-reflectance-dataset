

from ..Paths import SCENES
from .. import load_standard_capture
from ..Cleanup.board_markers_brightness import capture_calculate_marker_brightness

def post_capture(scenes: str = "all"):
    """Starting from original captured data, apply all cleanup and caching"""

    scene_names = SCENES if scenes == "all" else scenes.split(",")

    kw = dict(use_index=False)

    for sc in scene_names:
        print(f"Post-capture for scene {sc}")

        retro, olat = load_standard_capture(sc, retro_kwargs=kw, olat_kwargs=kw)

        for capture in (retro, olat):
            capture_calculate_marker_brightness(capture)
            capture.save_index()
