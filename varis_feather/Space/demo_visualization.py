


from pathlib import Path
from typing import Union

from matplotlib import pyplot

from .olat import OLATCapture
from .retro import RetroreflectionCapture


def visualize_olat(capture: OLATCapture, dir_out: Union[str, Path, None] = None):
    """All OLAT plots."""

    dir_out = Path(dir_out or (capture.dir_src / "visualizations"))
    dir_out.mkdir(exist_ok=True, parents=True)

    for view_name in capture.named_region_views.keys():
        for sp_wiid in capture.stage_poses.keys():
            for outliers in (False, True):
                fig = capture.plot_crops_in_theta_phi(view_name, pose=sp_wiid, show_outliers=outliers, out_path_override=dir_out)
                pyplot.close(fig)


def visualize_retro(capture: RetroreflectionCapture, dir_out: Union[str, Path, None] = None):
    """All Retro plots."""

    dir_out = Path(dir_out or (capture.dir_src / "visualizations"))
    dir_out.mkdir(exist_ok=True, parents=True)

    fig = capture.plot_board_normals_gradient_vs_theoretical()
    fig.savefig(dir_out / "board_normals_gradient_vs_theoretical.png")
