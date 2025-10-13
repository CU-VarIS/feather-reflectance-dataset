

from varis_feather.Utilities.ImageIO import writeImage
from .. import load_standard_capture
from ..Cleanup.board_markers_brightness import capture_calculate_marker_brightness
from ..Cleanup.brightness_calibration import correct_light_brightness_and_board_normal
from ..Paths import SCENES
from ..Space.capture import VarisCapture
from ..Space.demo_visualization import visualize_olat, visualize_retro
from ..Space.olat_subcrops import OLATRegionView

from ..Space.gradient_normals import GradientNormalVisitor


def _rebuild_caches(capture: VarisCapture, num_workers: int = 8, vis_normal_dir=None):
    visitors_normals = [
        GradientNormalVisitor(wiid) for wiid in capture.stage_poses.keys()
    ]
    visitors = [
        region.visitor_extract_region() 
        for region in capture.named_region_views.values()
    ] + visitors_normals
    capture.visit(visitors, num_threads=num_workers)

    # return visitors_normals

    if vis_normal_dir:
        vis_normal_dir.mkdir(exist_ok=True, parents=True)
        for v in visitors_normals:
            writeImage(v.normals_img, vis_normal_dir / f"gradient_normals_wiid{v.wiid[0]}_{v.wiid[1]}.png")


def _correct_capture(capture: VarisCapture, num_workers: int = 8):
    capture_calculate_marker_brightness(capture, num_workers=num_workers)
    correct_light_brightness_and_board_normal(capture, plot=capture.dir_src / "visualizations")
    capture.save_index()


def process_scene(scene: str, num_workers: int = 8):
    print(f"Post-capture for scene {scene}")
    kw = dict(use_index=False)

    retro, olat = load_standard_capture(scene, retro_kwargs=kw, olat_kwargs=kw)
    
    # Visualize state before correction
    _rebuild_caches(olat, num_workers=num_workers, vis_normal_dir=olat.dir_src / "visualizations_uncorrected")
    visualize_olat(olat, dir_out=olat.dir_src / "visualizations_uncorrected")

    _correct_capture(olat, num_workers=num_workers)
    
    # Rebuild region cache and visualize after correction
    _rebuild_caches(olat, num_workers=num_workers, vis_normal_dir=olat.dir_src / "visualizations")
    visualize_olat(olat, dir_out=olat.dir_src / "visualizations")

    if retro:
        visualize_retro(retro, dir_out=retro.dir_src / "visualizations_uncorrected")
        _correct_capture(retro, num_workers=num_workers)
        visualize_retro(retro, dir_out=retro.dir_src / "visualizations")

        

def post_capture(scenes: str = "all", num_workers: int = 8):
    """Starting from original captured data, apply all cleanup and caching"""

    scene_names = SCENES if scenes == "all" else scenes.split(",")


    for sc in scene_names:
        try:
            process_scene(sc, num_workers=num_workers)
        except Exception as e:
            print(f"Failed to process scene {sc}: {e}")