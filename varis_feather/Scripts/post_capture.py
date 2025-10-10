

from varis_feather.Utilities.ImageIO import writeImage
from .. import load_standard_capture
from ..Cleanup.board_markers_brightness import capture_calculate_marker_brightness
from ..Cleanup.brightness_calibration import correct_light_brightness_and_board_normal
from ..Paths import SCENES
from ..Space.capture import VarisCapture
from ..Space.demo_visualization import visualize_olat, visualize_retro
from ..Space.olat_subcrops import OLATRegionView

from ..Space.gradient_normals import GradientNormalVisitor


def _correct_capture(capture: VarisCapture, num_workers: int = 8):
    capture_calculate_marker_brightness(capture, num_workers=num_workers)
    correct_light_brightness_and_board_normal(capture, plot=capture.dir_src / "visualizations")
    capture.save_index()

def post_capture(scenes: str = "all", num_workers: int = 8):
    """Starting from original captured data, apply all cleanup and caching"""

    scene_names = SCENES if scenes == "all" else scenes.split(",")

    kw = dict(use_index=False)

    for sc in scene_names:
        print(f"Post-capture for scene {sc}")

        retro, olat = load_standard_capture(sc, retro_kwargs=kw, olat_kwargs=kw)

        normal_visitors = [
            GradientNormalVisitor(wiid) for wiid in olat.stage_poses.keys()
        ]
        olat.visit(normal_visitors, num_threads=num_workers)
        for v in normal_visitors:
            writeImage(v.normals_img, olat.dir_src / "visualizations_uncorrected" / f"gradient_normals_wiid{v.wiid[0]}_{v.wiid[1]}.png")
            # writeImage(olat.dir_src / "cache" / f"gradient_normals_wiid{v.wiid[0]}_{v.wiid[1]}.exr", v.normals)

        # Visualize state before correction
        OLATRegionView.extract_region_cache_olat(olat)
        visualize_olat(olat, dir_out=olat.dir_src / "visualizations_uncorrected")
        _correct_capture(olat, num_workers=num_workers)
        # Rebuild region cache and visualize after correction
        OLATRegionView.extract_region_cache_olat(olat)
        visualize_olat(olat, dir_out=olat.dir_src / "visualizations")

        if retro:
            visualize_retro(retro, dir_out=retro.dir_src / "visualizations_uncorrected")
            _correct_capture(retro, num_workers=num_workers)
            visualize_retro(retro, dir_out=retro.dir_src / "visualizations")

        