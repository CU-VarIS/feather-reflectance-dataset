

from ..Paths import SCENES
from .. import load_standard_capture
from ..Cleanup.board_markers_brightness import capture_calculate_marker_brightness
from ..Cleanup.brightness_calibration import correct_light_brightness_and_board_normal
from ..Space.demo_visualization import visualize_olat, visualize_retro
from ..Space.olat_subcrops import OLATRegionView

def post_capture(scenes: str = "all", num_workers: int = 8):
    """Starting from original captured data, apply all cleanup and caching"""

    scene_names = SCENES if scenes == "all" else scenes.split(",")

    kw = dict(use_index=False)

    for sc in scene_names:
        print(f"Post-capture for scene {sc}")

        retro, olat = load_standard_capture(sc, retro_kwargs=kw, olat_kwargs=kw)

        # Visualize state before correction
        OLATRegionView.extract_region_cache_olat(olat)
        visualize_retro(retro, dir_out=retro.dir_src / "visualizations_uncorrected")
        visualize_olat(olat, dir_out=olat.dir_src / "visualizations_uncorrected")

        # Cleanup and save index
        for capture in (retro, olat):
            capture_calculate_marker_brightness(capture, num_workers=num_workers)
            correct_light_brightness_and_board_normal(capture, plot=capture.dir_src / "visualizations")
            capture.save_index()

        # Rebuild region cache and visualize after correction
        OLATRegionView.extract_region_cache_olat(olat)
        visualize_retro(retro, dir_out=retro.dir_src / "visualizations")
        visualize_olat(olat, dir_out=olat.dir_src / "visualizations")