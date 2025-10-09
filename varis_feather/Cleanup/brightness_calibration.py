
"""
Removing frames with arm shadows
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

import cv2 as cv
import h5py
import numpy as np
from imageio import imwrite
from matplotlib import pyplot
from tqdm import tqdm
from scipy.spatial.transform import Rotation


from ..Space.capture import VarisCapture
from ..Space.olat_subcrops import OLATRegionView
from ..Utilities.ImageIO import RGL_tonemap, RGL_tonemap_uint8
from ..Utilities.Pixels import marker_mask_from_image
from ..Utilities.show_image import image_montage_same_shape, show






@dataclass
class FrameBrightnessStatistics:
    board_visible_fraction: np.ndarray
    marker_white_rgb_mean: np.ndarray
    marker_white_rgb_std: np.ndarray

    @cached_property
    def marker_white_mean(self) -> np.ndarray:
        return np.mean(self.marker_white_rgb_mean, axis=1)

    @cached_property
    def marker_white_std(self) -> np.ndarray:
        return np.std(self.marker_white_rgb_mean, axis=1)

    @classmethod
    def from_num_frames(cls, num_frames: int) -> "FrameBrightnessStatistics":
        return cls(
            board_visible_fraction=np.zeros(num_frames, dtype=np.float32),
            marker_white_rgb_mean=np.zeros((num_frames, 3), dtype=np.float32),
            marker_white_rgb_std=np.zeros((num_frames, 3), dtype=np.float32)
        )

    def write_to_h5(self, path: Path):
        with h5py.File(path, "w") as f:
            f.create_dataset("board_visible_fraction", data=self.board_visible_fraction)
            f.create_dataset("marker_white_rgb_mean", data=self.marker_white_rgb_mean)
            f.create_dataset("marker_white_rgb_std", data=self.marker_white_rgb_std)

    @classmethod
    def from_h5(cls, path: Path) -> "FrameBrightnessStatistics":
        with h5py.File(path, "r") as f:
            return cls(
                board_visible_fraction=f["board_visible_fraction"][:],
                marker_white_rgb_mean=f["marker_white_rgb_mean"][:],
                marker_white_rgb_std=f["marker_white_rgb_std"][:]
            )

   
    @staticmethod
    def _capture_cleanup_shadows_per_frame(capture: VarisCapture, fr_id: int, mask_all: np.ndarray, mask_all_area: int):
        frame_measurement = capture.read_measurement_image(fr_id, cache=False)
        frame_tonemapped = RGL_tonemap_uint8(frame_measurement)
        frame_mask = marker_mask_from_image(frame_tonemapped, morph=cv.MORPH_OPEN, margin=128)

        board_mask_intersection = np.logical_and(frame_mask, mask_all)
        board_visible_fraction = np.count_nonzero(board_mask_intersection) / mask_all_area

        marker_white_pixels = frame_measurement[board_mask_intersection]

        marker_white_intensity = np.mean(marker_white_pixels, axis=1)
        marker_white_intensity_mean = np.mean(marker_white_intensity)
        marker_white_intensity_std = np.std(marker_white_intensity)

        marker_white_rgb_mean = np.mean(marker_white_pixels, axis=0)
        marker_white_rgb_std = np.std(marker_white_pixels, axis=0)

        return fr_id, board_visible_fraction, marker_white_intensity_mean, marker_white_intensity_std, marker_white_rgb_mean, marker_white_rgb_std

    @classmethod
    def from_markers(cls, capture: VarisCapture) -> "FrameBrightnessStatistics":
        num_frames = len(capture)
        stats = FrameBrightnessStatistics.from_num_frames(num_frames)

        with ThreadPoolExecutor() as pool:
            futures = []

            for wiid, sp in capture.stage_poses.items():
                mask_all = marker_mask_from_image(sp.all_lights_image, morph=cv.MORPH_ERODE)
                mask_all_area = np.count_nonzero(mask_all)

                for i in sp.frame_indices:
                    futures.append(pool.submit(cls._capture_cleanup_shadows_per_frame, capture, i, mask_all, mask_all_area))


            for i, fraction in tqdm(enumerate(as_completed(futures)), total=len(futures), desc="Extracting marker brightness"):
                fr_id, board_visible_fraction, marker_white_intensity_mean, marker_white_intensity_std, marker_white_rgb_mean, marker_white_rgb_std = fraction.result()
                stats.board_visible_fraction[fr_id] = board_visible_fraction
                stats.marker_white_rgb_mean[fr_id] = marker_white_rgb_mean
                stats.marker_white_rgb_std[fr_id] = marker_white_rgb_std


        path_out = capture._brightness_statistics_path()
        path_out.parent.mkdir(exist_ok=True)
        stats.write_to_h5(path_out)

        return stats
    
    @classmethod
    def from_view(cls, view: OLATRegionView) -> "FrameBrightnessStatistics":
        num_frames = len(view.capture)
        stats = FrameBrightnessStatistics.from_num_frames(num_frames)

        stats.board_visible_fraction[:] = 1.0  # Assume full visibility if using region view

        def process_frame(idx):
            mean = np.mean(view[idx], axis=(0, 1))
            std = np.std(view[idx], axis=(0, 1))
            stats.marker_white_rgb_mean[idx] = mean
            stats.marker_white_rgb_std[idx] = std
            
        with ThreadPoolExecutor() as pool:
            for _ in tqdm(as_completed([pool.submit(process_frame, i) for i in range(num_frames)])):
                pass

        return stats
    

def plot_brightness(theta_cos, rgb_mean, rgb_std):
    theta_deg = np.rad2deg(np.arccos(theta_cos))

    brightness_mean = np.mean(rgb_mean, axis=1)
    brightness_std = np.mean(rgb_std, axis=1)
    color = RGL_tonemap(rgb_mean)

    fig, ax = pyplot.subplots(1, 2, figsize=(16, 10))
    ax[0].errorbar(theta_cos, brightness_mean, yerr=np.abs(brightness_std), color="grey", linestyle="none", zorder=1)
    ax[0].scatter(theta_cos, brightness_mean, color=color, marker="o")
    ax[0].set_title("White Level vs Theta (Cosine)")
    ax[0].set_xlabel("$\\cos(\\theta)$")
    ax[0].set_ylabel("Marker brightness")

    # seaborn.pointplot(x=theta_deg, y=stats_white_mean, ax=ax[0], yerr=stats_white_std)
    # seaborn.pointplot(x=theta_deg, y=stats_white_mean, ax=ax[0], errorbar=("ci", stats_white_std))
    ax[1].errorbar(theta_deg, brightness_mean, yerr=np.abs(brightness_std), color="grey", linestyle="none", zorder=1)
    ax[1].scatter(theta_deg, brightness_mean, color=color, marker="o")
    ax[1].set_title("White Level vs Theta (Degrees)")
    ax[1].set_xlabel("$\\theta [^O]$")
    ax[1].set_ylabel("Marker brightness")

    fig.tight_layout()
    return fig

def plot_brightness_statistics(capture: VarisCapture, mask = None, theta_cos = None):

    stats: FrameBrightnessStatistics = capture.stats_frame_brightness
    
    fr_idx_mask_valid = mask if mask is not None else stats.board_visible_fraction > 0.85

    if theta_cos is None:
        wos = capture.frame_wo[fr_idx_mask_valid]
        theta_cos = wos[:, 2]

    plot_brightness(theta_cos, stats.marker_white_rgb_mean[fr_idx_mask_valid], stats.marker_white_rgb_std[fr_idx_mask_valid])


def plot_per_light(theta_cos_adjusted, frame_brightness, per_light_indices):
    per_light_slope = np.ones(len(per_light_indices), dtype=np.float32)

    # Plot each light's frame brightnesses as a line
    fig, ax = pyplot.subplots(figsize=(10, 6))
    for light_idx, frame_indices in enumerate(per_light_indices):
        if len(frame_indices) == 0:
            per_light_slope[light_idx] = 1
            continue
        light_brightness = frame_brightness[frame_indices]
        theta_cos_subset = theta_cos_adjusted[frame_indices]

        # polyfit is with intercept
        # per_light_slope[light_idx] = np.polyfit(theta_cos_subset, light_brightness, 1)[0]

        a_no_intercept, _, _, _ = np.linalg.lstsq(theta_cos_subset[:, None], light_brightness, rcond=None)
        a_no_intercept = a_no_intercept[0]
        per_light_slope[light_idx] = a_no_intercept

        order = np.argsort(theta_cos_subset)
        ax.plot(theta_cos_subset[order], light_brightness[order], marker='.')

    ax.set_xlabel("$\\cos(\\theta)$")
    ax.set_ylabel("Brightness")
    # ax.set_title("Light Brightness Over Time")
    # ax.legend()
    fig.tight_layout()
    fig.show()

    return per_light_slope

def plot_both(theta_cos, rgb_mean, rgb_std, per_light_indices):
    brightness_mean = np.mean(rgb_mean, axis=1)
    brightness_std = np.mean(rgb_std, axis=1)
    color = RGL_tonemap(rgb_mean)

    fig, (ax_fit, ax_lights) = pyplot.subplots(1, 2, figsize=(16, 10))
    ax_fit.errorbar(theta_cos, brightness_mean, yerr=np.abs(brightness_std), color="grey", linestyle="none", zorder=1)
    ax_fit.scatter(theta_cos, brightness_mean, color=color, marker="o")
    ax_fit.set_title("White Level vs Theta (Cosine)")
    ax_fit.set_xlabel("$\\cos(\\theta)$")
    ax_fit.set_ylabel("Marker brightness")

    a_no_intercept, sq_err, _, _ = np.linalg.lstsq(theta_cos[:, None], brightness_mean, rcond=None)
    a_no_intercept = a_no_intercept[0]

    theta_max = np.max(theta_cos)
    sq_err = sq_err[0] / len(theta_cos)
    print(f"Residual: {sq_err}")

    ax_fit.plot([0, theta_max], [0, a_no_intercept * theta_max], color="black", linestyle="--", label=f"Fit err={sq_err}")


    ax_fit.legend()


    per_light_slope = np.ones(len(per_light_indices), dtype=np.float32)
    for light_idx, frame_indices in enumerate(per_light_indices):
        if len(frame_indices) == 0:
            per_light_slope[light_idx] = 1
            continue
        light_brightness = brightness_mean[frame_indices]
        theta_cos_subset = theta_cos[frame_indices]

        a_no_intercept, _, _, _ = np.linalg.lstsq(theta_cos_subset[:, None], light_brightness, rcond=None)
        a_no_intercept = a_no_intercept[0]
        per_light_slope[light_idx] = a_no_intercept

        order = np.argsort(theta_cos_subset)
        ax_lights.plot(theta_cos_subset[order], light_brightness[order], marker='.')

    ax_lights.set_xlabel("$\\cos(\\theta)$")
    ax_lights.set_ylabel("Brightness")

    fig.tight_layout()
    


    return fig


def light_find_slope(theta_cos, frame_brightness, per_light_indices):
    per_light_slope = np.ones(len(per_light_indices), dtype=np.float32)

    for light_idx, frame_indices in enumerate(per_light_indices):
        if len(frame_indices) == 0:
            per_light_slope[light_idx] = 1
            continue
        light_brightness = frame_brightness[frame_indices]
        theta_cos_subset = theta_cos[frame_indices]
       
        a_no_intercept, _, _, _ = np.linalg.lstsq(theta_cos_subset[:, None], light_brightness, rcond=None)
        a_no_intercept = a_no_intercept[0]
        
        # Can't be negative!
        if a_no_intercept <= 0:
            a_no_intercept = 1.0
        
        per_light_slope[light_idx] = a_no_intercept




    return per_light_slope



def rotate_stage_pose_by_board_normal_keeping_view_x_zero(board_normal: np.ndarray, stage_view_dir: np.ndarray) -> Rotation:
    """
    Find a rotation that adjusts view and light directions in sample space given a recovered board normal.

    Args:
        board_normal: [3] Normal of the board plane in sample coordinates
        stage_view_dir: [3] View direction is sample coordinates

    Returns:
        A rotation that sends board_normal -> [0,0,1] 
        but keeps the rotated stage_view_dir x at 0 to maintain the convention.
    """

    eps = 1e-12
    board_normal = np.asarray(board_normal, dtype=np.float64)
    stage_view_dir = np.asarray(stage_view_dir, dtype=np.float64)
    stage_view_dir /= np.linalg.norm(stage_view_dir)

    # Check that stage_view_dir x is close to 0
    if abs(stage_view_dir[0]) > 1e-6:
        raise ValueError("stage_view_dir x component must be close to 0")

    # Build the rotation matrix columns
    # unit R_z
    R_z = board_normal / np.linalg.norm(board_normal)

    # choose x axis orthogonal to both R_z and stage_view_dir
    R_x = np.cross(R_z, stage_view_dir)
    if (R_x_norm := np.linalg.norm(R_x)) > eps:
        R_x /= R_x_norm
    else:
        # stage_view_dir parallel (or nearly) to board_normal: pick any stable perpendicular vector
        perp = np.array([1.0, 0.0, 0.0]) if abs(R_z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        R_x = np.cross(perp, R_z)
        R_x /= np.linalg.norm(R_x)

    # The y axis is normal to both x and z
    R_y = np.cross(R_z, R_x)

    S = np.column_stack([R_x, R_y, R_z]) # source basis columns
    R_adjust = Rotation.from_matrix(S.T) # maps source basis -> canonical basis

    assert np.allclose(R_adjust.apply(board_normal), np.array([0.0, 0.0, 1.0]), atol=1e-8)
    assert abs(R_adjust.apply(stage_view_dir)[0]) < 1e-6

    return R_adjust

def dataset_opt(capture: VarisCapture, n_iters = 3, stats_override: Optional[FrameBrightnessStatistics] = None, b_plot: Union[bool, str, Path] = True):

    # Lights
    _, light_id_unique, light_ref_count = np.unique(capture.frame_dmx_light_ids, axis=0, return_inverse=True, return_counts=True)
    num_lights = light_ref_count.shape[0]

    # Remove outliers
    stats: FrameBrightnessStatistics = stats_override or capture.stats_frame_brightness
    fr_idx_mask_valid = (stats.board_visible_fraction > 0.8) & (stats.marker_white_mean > 0.05 )

    print(f"Valid frames: {np.count_nonzero(fr_idx_mask_valid)} / {len(fr_idx_mask_valid)}")

    light_id_unique = light_id_unique[fr_idx_mask_valid]
    # For each unique light, find all frame indices
    per_light_indices = [[] for _ in range(num_lights)]
    for fr_idx, light_id in enumerate(light_id_unique):
        per_light_indices[light_id].append(fr_idx)
    per_light_indices = [np.array(indices) for indices in per_light_indices]


    # Brightness
    stats_rgb_mean = stats.marker_white_rgb_mean[fr_idx_mask_valid]
    stats_rgb_std = stats.marker_white_rgb_std[fr_idx_mask_valid]
    stats_white_mean = np.mean(stats_rgb_mean, axis=1)
    stats_white_std = np.mean(stats_rgb_std, axis=1)
    wos = capture.frame_wo[fr_idx_mask_valid]


    theta_cos = wos[:, 2]
    light_slope = np.ones(num_lights, dtype=np.float32)
    brightness = stats_white_mean

    def plot(step= 0):
        if b_plot:
            fig = plot_both(theta_cos, stats_rgb_mean / light_slope[light_id_unique][:, None], stats_white_std / light_slope[light_id_unique][:, None], per_light_indices)
            if isinstance(b_plot, (str, Path)):
                path_out = Path(b_plot) / f"{capture.name}_brightness_correction_iter{step:02d}"
                path_out.parent.mkdir(exist_ok=True, parents=True)
                for fmt in ["png", "pdf"]:
                    fig.savefig(path_out.with_suffix(f".{fmt}"))

    plot()

    for step in range(n_iters):
        # plot_per_light(theta_cos, stats_white_mean, per_light_indices)
        light_slope = light_find_slope(theta_cos, stats_white_mean, per_light_indices)
        brightness = stats_white_mean / light_slope[light_id_unique]

        plot(step*2+2)

        # Solve for initial board rotation normal, such that
        # wos dot n is linear to stack_mean
        n_optimized = np.linalg.lstsq(wos, brightness, rcond=None)[0]
        n_optimized /= np.linalg.norm(n_optimized)
        print(f"Initial board rotation normal: {n_optimized}")
        theta_cos = np.dot(wos, n_optimized)

        plot(step*2+1)



    # TODO rotate the frames!


    for fr in capture.frames:
        ...
    

    return n_optimized, light_slope, 1. / light_slope[light_id_unique]



def correct_light_brightness_and_board_normal(capture: VarisCapture, n_iters = 3, visible_fraction_threshold:float=0.8, brightness_threshold:float= 0.05, plot: Union[bool, str, Path] = True):


    # Pick valid
    white_marker_visible_fraction = capture.frame_table["white_marker_visible_fraction"].to_numpy()
    fr_idx_mask_valid = (white_marker_visible_fraction > visible_fraction_threshold) & (capture.white_marker_intensity_mean > brightness_threshold )
    stats_rgb_mean = capture.white_marker_rgb_mean[fr_idx_mask_valid]
    stats_white_std = capture.white_marker_intensity_std[fr_idx_mask_valid]
    stats_white_mean = capture.white_marker_intensity_mean[fr_idx_mask_valid]
    wos = capture.sample_wo[fr_idx_mask_valid]
    print(f"Valid frames: {np.count_nonzero(fr_idx_mask_valid)} / {len(fr_idx_mask_valid)}")

    # Lights
    _, light_id_unique, light_ref_count = np.unique(capture.frame_dmx_light_ids, axis=0, return_inverse=True, return_counts=True)
    num_lights = light_ref_count.shape[0]
    # For each unique light, find all frame indices
    light_id_unique = light_id_unique[fr_idx_mask_valid]
    per_light_indices = [[] for _ in range(num_lights)]
    for fr_idx, light_id in enumerate(light_id_unique):
        per_light_indices[light_id].append(fr_idx)
    per_light_indices = [np.array(indices) for indices in per_light_indices]

    # Optimization variables
    n_optimized = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    light_slope = np.ones(num_lights, dtype=np.float32)

    theta_cos = wos[:, 2]
    brightness = stats_white_mean

    def plot_step(step= 0):
        if plot:
            fig = plot_both(theta_cos, stats_rgb_mean / light_slope[light_id_unique][:, None], stats_white_std / light_slope[light_id_unique][:, None], per_light_indices)
            if isinstance(plot, (str, Path)):
                path_out = Path(plot) / f"{capture.name}_brightness_correction_iter{step:02d}"
                path_out.parent.mkdir(exist_ok=True, parents=True)
                for fmt in ["png", "pdf"]:
                    fig.savefig(path_out.with_suffix(f".{fmt}"))


    # for step in range(n_iters):
    #     # plot_per_light(theta_cos, stats_white_mean, per_light_indices)
    #     light_slope = light_find_slope(theta_cos, stats_white_mean, per_light_indices)
    #     brightness = stats_white_mean / light_slope[light_id_unique]

    #     plot(step*2+1)

    #     # Solve for initial board rotation normal, such that
    #     # wos dot n is linear to stack_mean
    #     n_optimized = np.linalg.lstsq(wos, brightness, rcond=None)[0]
    #     n_optimized /= np.linalg.norm(n_optimized)
    #     print(f"Initial board rotation normal: {n_optimized}")
    #     theta_cos = np.dot(wos, n_optimized)

    #     plot(step*2+2)


    def opt_normal():
        # Solve for initial board rotation normal, such that
        # wos dot n is linear to stack_mean
        n_optimized = np.linalg.lstsq(wos, brightness, rcond=None)[0]
        n_optimized /= np.linalg.norm(n_optimized)
        print(f"Initial board rotation normal: {n_optimized}")
        theta_cos = np.dot(wos, n_optimized)
        return n_optimized, theta_cos

    def opt_light_brightness():
        # plot_per_light(theta_cos, stats_white_mean, per_light_indices)
        light_slope = light_find_slope(theta_cos, stats_white_mean, per_light_indices)
        brightness = stats_white_mean / light_slope[light_id_unique]
        return light_slope, brightness


    plot_step()

    for step in range(n_iters):
        n_optimized, theta_cos = opt_normal()
        plot_step(step*2+1)

        light_slope, brightness = opt_light_brightness()
        plot_step(step*2+2)


    brightness_correction_all_frames = np.ones(len(capture), dtype=np.float32)
    brightness_correction_all_frames[fr_idx_mask_valid] = 1. / light_slope[light_id_unique]

    # Rotation is specific to stage poses
    rotation_by_stage_pose_index : dict[tuple[int, int], Rotation] = {}

    for fr, is_valid, brightness_correction in zip(capture.frames, fr_idx_mask_valid, brightness_correction_all_frames):
        fr.brightness_correction = brightness_correction

        R = rotation_by_stage_pose_index[fr.wiid] = (
            rotation_by_stage_pose_index.get(fr.wiid) 
            or rotate_stage_pose_by_board_normal_keeping_view_x_zero(n_optimized, fr.sample_wi)
        )
        fr.sample_wi = R.apply(fr.sample_wi)
        fr.sample_wo = R.apply(fr.sample_wo)

        fr.is_valid = is_valid and fr.sample_wo[2] > 0.01  # also remove cases where light ends up being below horizon


    print(f"{rotation_by_stage_pose_index=}")

    capture._update_table_from_frames()



