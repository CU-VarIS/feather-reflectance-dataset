from dataclasses import dataclass
from enum import Enum
from functools import cached_property, lru_cache
from pathlib import Path
import re
from typing import Any, Literal, Optional, Union

import einops
from matplotlib import pyplot
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import seaborn
from tqdm import tqdm
import cv2 as cv

from ..Utilities.ImageIO import RGL_tonemap_uint8, readImage, writeImage



class ThetaDistribution:
    MODE_ARC_COS = "arc_cos"
    MODE_UNIFORM = "uniform"
    MODE_U_TO_THETA = "u_to_theta"

    def __init__(self, mode: Literal[MODE_ARC_COS, MODE_UNIFORM, MODE_U_TO_THETA], offset_rad: float = 0.):
        self.mode = mode
        self.offset_rad = offset_rad

    

@dataclass
class OLATFrame:
    """
    An photo in the OLAT (one light at a time) capture
    """

    # Which single light was on
    dmx_id: int = -1
    light_id: int = -1

    # Incident direction
    wiid: tuple[int, int] = (-1, 0)
    theta_i: float = -1.0
    phi_i: float = -1.0

    # Transformation from dome space to sample space
    r_dome_to_sample: Rotation | None = None
    # Indicent direction in sample space
    sample_wi: np.ndarray | None = None
    # Outgoing direction in sample space
    sample_wo: np.ndarray | None = None

    image_path: Path | None = None
    image_cache: np.ndarray | None = None

    is_valid_macro_shadow: bool = True
    white_marker_visible_fraction: float | None = None
    white_marker_intensity_mean: float | None = None
    white_marker_intensity_std: float | None = None
    # white_marker_color_mean: np.ndarray | None = None
    # white_marker_color_std: np.ndarray | None = None

    @property
    def is_valid(self) -> bool:
        return self.is_valid_macro_shadow
    
    @property
    def incident_direction_preset_idx(self):
        return self.wiid

@dataclass
class OLATStagePose:
    """
    Description of a particular pose of the stage shared by a set of images.
    For a given pose, we capture an image for each light, resulting in around 200 images.

    Fields:
        all_lights_image: Each pose also has an image with all lights on for better alignment.
        normals_image: Likewise each pose has a normal map estimated from subtration of gradient images.
    """

    theta_i_index: int
    phi_i_index: int
    theta_i: float
    phi_i: float

    name: str = ""

    all_lights_image_path: Path | None = None
    normals_image_path: Path | None = None
    anchor_image_path: Path | None = None

    # Indices into the OLATCapture.frames array, for frames sharing this stage pose
    frame_indices: np.ndarray = None
    # Index to lookup frames by outgoing direction in sample space
    wo_kdtree: KDTree = None

    R_to_macronormal: Rotation | None = None

    # disparity_direction: np.ndarray | None = None
    # disparity_magnitude: float = -1.

    # Extra homography for manual adjustment
    registration_manual_homography: np.ndarray | None = None

    alignment_transform: Optional["itk.ParameterObject"] = None
    alignment_transform_paths: list[Path] | None = None

    @property
    def wiid(self):
        return (self.theta_i_index, self.phi_i_index)

    @cached_property
    def all_lights_image(self) -> np.ndarray:
        """
        Photo with all lights on in JPG format, captured for each theta-phi pair.
        Loaded on demand when first accessed and then cached.
        """
        return self._manual_homography_apply(readImage(str(self.all_lights_image_path)))

    @cached_property
    def normals_image(self) -> np.ndarray:
        """
        Normals estimated using gradient patterns, captured for each theta-phi pair.
        Loaded on demand when first accessed and then cached.
        """
        return self._manual_homography_apply(readImage(str(self.normals_image_path)))
    
    @cached_property
    def anchor_image(self) -> np.ndarray:
        """
        Normals estimated using gradient patterns, captured for each theta-phi pair.
        Loaded on demand when first accessed and then cached.
        """
        return self._manual_homography_apply(readImage(str(self.anchor_image_path)))

    @cached_property
    def disparity_direction(self) -> np.ndarray:
        phi_adjusted = self.phi_i - np.pi / 2
        return np.array([np.cos(phi_adjusted), np.sin(phi_adjusted)], dtype=np.float32)

    @property
    def disparity_magnitude(self) -> float:
        return np.tan(self.theta_i)
    
    def _manual_homography_apply(self, img: np.ndarray) -> np.ndarray:
        if self.registration_manual_homography is not None:
            h, w = img.shape[:2]
            return cv.warpPerspective(img, self.registration_manual_homography, (w, h))
        return img

    def set_manual_homography_from_corners(self, corners_target_xy: np.ndarray, corners_source_xy: np.ndarray):
        """Add a homography adjustment to the image."""
        self.registration_manual_homography, _ = cv.findHomography(corners_source_xy, corners_target_xy)

    @cached_property
    def board_white_mask(self) -> np.ndarray:
        board_white_mask = np.mean(self.anchor_image, axis=2) > 150
        board_white_mask[128:-128, 128:-128] = False
        return board_white_mask
    
    @cached_property
    def board_gradient_normal_mean_and_std(self) -> tuple[np.ndarray, np.ndarray]:
        normals = 2*self.normals_image - 1
        normals_on_white = normals[self.board_white_mask]
        normal_mean = np.mean(normals_on_white, axis=0)
        normal_std = np.std(normals_on_white, axis=0)
        return normal_mean, normal_std

    @cached_property
    def board_gradient_normal(self) -> np.ndarray:
        n, _ = self.board_gradient_normal_mean_and_std
        return n / np.linalg.norm(n)
    
    @cached_property
    def normals_smooth(self) -> np.ndarray:
        return self.normals_smooth_configurable()

    def normals_smooth_configurable(self, ksize:int = 5, steps:int = 2) -> np.ndarray:
        img_normals = self.normals_image
        img_normals_smooth = img_normals
        for s in range(steps):
            img_normals_smooth = cv.medianBlur(img_normals_smooth, ksize)

        val_normals_smooth = (img_normals_smooth * 2 - 1)
        val_normals_smooth /= np.linalg.norm(val_normals_smooth, axis=2, keepdims=True)
        return val_normals_smooth


class VarisCapture:
    frame_wo: np.ndarray
    frame_wi: np.ndarray
    # Array of `wiid`
    frame_wi_index: np.ndarray

    # The set of 8 default in
    # olat_default_incident_angles = np.pi * 0.5 - np.arccos(np.linspace(0, 1, 8))
    # To be set by set_incident_angles
    olat_theta_i: np.ndarray
    olat_phi_i: np.ndarray
    # width x height in mm
    img_full_size_mm: tuple[float, float]
    pix_per_mm: float
    mm_per_tile: float
    name: str
    dir_src: Path

    @property
    def num_theta_i(self) -> int:
        return len(self.olat_theta_i)

    @property
    def num_phi_i(self) -> int:
        return len(self.olat_phi_i)

    @cached_property
    def is_isotropic(self) -> bool:
        return self.num_phi_i == 1

    @property
    def is_symetric_along_X(self) -> bool:
        """Whether to duplicate light directions along the Y axis for isotropic captures."""
        return False

    def _set_angles(self, num_theta_i, num_phi_i, theta_distribution: ThetaDistribution):
        raise NotImplementedError("Provide implementation for self.olat_theta_i and self.olat_phi_i")

    def _handle_file_all_lights(self, path: Path, match: re.Match, is_iso=False):
        if is_iso:
            name, theta_id = match.groups()
            phi_id = "0"
        else:
            name, theta_id, phi_id = match.groups()
        wi_id = (int(theta_id), int(phi_id))
        self.get_stage_pose(wi_id, name=name, create=True).all_lights_image_path = path

    def _handle_file_grad_normals(self, path: Path, match: re.Match, is_iso=False):
        if is_iso:
            name, theta_id = match.groups()
            phi_id = "0"
        else:
            name, theta_id, phi_id = match.groups()
        wi_id = (int(theta_id), int(phi_id))
        self.get_stage_pose(wi_id, name=name, create=True).normals_image_path = path

    def _handle_file_anchor(self, path: Path, match: re.Match, is_iso=False):
        if is_iso:
            name, theta_id = match.groups()
            phi_id = "0"
        else:
            name, theta_id, phi_id = match.groups()

        wi_id = (int(theta_id), int(phi_id))
        self.get_stage_pose(wi_id, name=name, create=True).anchor_image_path = path

    def _handle_file_unmatched(self, path: Path):
        self._unmatched_files.setdefault(re.sub(r"\d", "#", path.name), []).append(path)

    def _ingest_files(self, dir_src: Path, handlers: dict[str, callable]):
        for file in sorted(dir_src.iterdir()):
            handled = False
            for pattern, handler in handlers.items():
                if m := re.match(pattern, file.name):
                    handler(file, m)
                    handled = True
                    break

            if not handled and file.is_file():
                self._handle_file_unmatched(file)

    def _update_name(self, frame_name: str):
        if self.name == "uninitialized":
            self.name = frame_name
        elif self.name != frame_name:
            print(f"Frame name '{frame_name}' does not match previously loaded '{self.name}'")

    def get_stage_pose(
        self, wi_id: tuple[int, int], name: str = "", create: bool = False
    ) -> OLATStagePose | None:
        sp = self.stage_poses.get(wi_id)

        if sp is None and create:
            theta_i, phi_i = wi_id
            sp = OLATStagePose(
                theta_i_index=theta_i,
                theta_i=self.olat_theta_i[theta_i],
                phi_i_index=phi_i,
                phi_i=self.olat_phi_i[phi_i],
                name=name,
            )
            self.stage_poses[wi_id] = sp

        self._update_name(name)

        return sp

    def __init__(
        self,
        dir_src: Path,
        num_theta_i=8,
        num_phi_i=8,
        theta_distribution: ThetaDistribution = ThetaDistribution(mode=ThetaDistribution.MODE_UNIFORM),
        img_full_size_mm: tuple[float, float] = (9 * 7, 11 * 7),
        pix_per_mm: float = 128 / 7,
        mm_per_tile: float = 7,
    ):
        self.name = "uninitialized"
        self.dir_src = dir_src
        self._set_angles(num_theta_i, num_phi_i, theta_distribution=theta_distribution)
        self.stage_poses: dict[tuple[int, int], OLATStagePose] = {}
        self.frames: list[OLATFrame] = []
        self.img_full_size_mm = img_full_size_mm
        self.pix_per_mm = pix_per_mm
        self.mm_per_tile = mm_per_tile

        self._unmatched_files = {}


    def tile_to_pix(self, tx: float, ty: float) -> tuple[int, int]:
        r = self.mm_per_tile * self.pix_per_mm
        return round(tx * r), round(ty * r)

    def report(self) -> str:
        unmatched_file_report = "\n".join([
            f"    {pattern} x {len(paths)}" if len(paths) > 1 else f"    {paths[0].name}" for pattern, paths in self._unmatched_files.items()
        ])

        return f"""Capture {self.name} at {self.dir_src}
  {self.num_theta_i} thetas x {self.num_phi_i} phis
  {len(self.stage_poses)} stage poses
  {len(self.frames)} measurements
  Unmatched files:\n{unmatched_file_report}
"""

    # @cached_property
    # def image_shape(self):
    #     w, h = self.img_full_size_mm

    #     shape = (
    #         round(self.pix_per_mm * h), 
    #         round(self.pix_per_mm * w), 
    #         3,
    #     )

    #     assert self.stage_poses[(0, 0)].all_lights_image.shape == shape

    #     return shape
    
    # def crop_mm_slice(self, roi_tl_mm: tuple[float, float], roi_wh_mm: tuple[float, float]) -> tuple[slice, slice]:
    #     roi_tl, roi_wh = self.crop_mm_to_pix(roi_tl_mm, roi_wh_mm)
    #     # return img[roi_tl[1]:roi_tl[1]+roi_wh[1], roi_tl[0]:roi_tl[0]+roi_wh[0]]
    #     return slice(roi_tl[1], roi_tl[1]+roi_wh[1]), slice(roi_tl[0], roi_tl[0]+roi_wh[0])
    
    # def crop_tile_slice(self, roi_tl_mm: tuple[float, float], roi_wh_mm: tuple[float, float]) -> tuple[slice, slice]:
    #     roi_tl, roi_wh = self.crop_mm_to_pix(roi_tl_mm, roi_wh_mm)
    #     # return img[roi_tl[1]:roi_tl[1]+roi_wh[1], roi_tl[0]:roi_tl[0]+roi_wh[0]]
    #     return slice(roi_tl[1], roi_tl[1]+roi_wh[1]), slice(roi_tl[0], roi_tl[0]+roi_wh[0])

    def read_measurement_image(self, frame: Union[OLATFrame, int], cache=False) -> np.ndarray:
        frame = frame if isinstance(frame, OLATFrame) else self.frames[int(frame)]

        if cache and frame.image_cache is not None:
            return frame.image_cache

        img = readImage(frame.image_path)

        if cache:
            frame.image_cache = img


        return img



    @staticmethod
    def _rotation_to_z(vec):
        vec = vec / np.linalg.norm(vec)
        axis = np.cross(vec, [0, 0, 1])
        axis /= np.linalg.norm(axis)
        angle = np.arccos(vec[2])
        return Rotation.from_rotvec(axis * angle)

    def make_R_domeEnvYUp_to_sample(cls, theta_rad: float, phi_rad: float, is_isotropic: bool = True) -> Rotation:
        """
        FOR ISOTROPIC
        The 0..1 range for cosine is divided evenly into 8 directions.
        The last one is the fully grazing angle so it is not captured.
        wi_0 to wi_6 are the 7 captured directions.

        To get the angles, we arrange cosides in 0 ... 1 then apply arccos.
        The angles are denser along the normal.
        Therefore we do
            90 deg - arccos(i)
        
        FOR ANISOTROPIC
        Same theta_i angles.
        Several phi_i rotations 0..360 deg about same macronormal after the theta_i rotation.
        """
       
        #print(f'make_R_domeEnvYUp_to_sample(): {is_isotropic=}')
        if is_isotropic:
            # For isotropic capture, 2 maps to the biggest arm ('Y')
            # return DomeCoordinates.make_R_domeEnvYUp_to_sample(arm_2_angle=theta_rad)
            #return Rotation.from_euler("y", np.pi + theta_rad)
            return Rotation.from_matrix(
                #Rotation.from_euler("z", np.pi).as_matrix()          # also better but not sure if better than pi/2?
                Rotation.from_euler("z", np.pi * 0.5).as_matrix()   # seems much better, rotate in phi-ish dimension after main rotation
                #Rotation.from_euler("z", 0.0).as_matrix()          # no influence
                @ Rotation.from_euler("y", np.pi + theta_rad).as_matrix()
            )

        else:
            # JRB NOTE: Not rotating 'phi' since don't want to affect view dir by phi, should be same for a single theta.
            # JRB TODO: Why pi+theta?
            return Rotation.from_matrix(
                #Rotation.from_euler("x", np.pi + theta_rad).as_matrix()
                #Rotation.from_euler("z", phi_rad).as_matrix()
                #@
                Rotation.from_euler("x", np.pi + theta_rad).as_matrix()
            )
     

    def _derive_frame_coordinates(self, invalid: str = "error"):
        """
        Processes `self.frames` to determine the sample space transform, incident and outgoing directions.

        Args:
            invalid: what to do if an invalid (light under sample horizon) frame is found
                "error" - throw an exception, use this mode to verify that a stored OLAT is correctly loaded
                "remove" - remove the frame, use this mode to filter valid frames from naively generated synthetic frames
        """

        valid_indices = []

        if invalid not in {"error", "remove", "ignore"}:
            raise ValueError(
                f"invalid must be 'error' / 'remove' / 'ignore', got {invalid}"
            )

        for i, frame in enumerate(self.frames):
            # Check if directions are above horizon
            is_valid = frame.sample_wo[2] > -0.05 and frame.sample_wi[2] > -0.
            # TODO Should I bump up this threshold?  I like the images that the -0.05 check deems invalid...
            

            if is_valid:
                valid_indices.append(i)
            elif invalid == "error":
                raise ValueError(
                    f"Frame {i} has invalid directions {frame.sample_wo=} (for LED {frame.dmx_id=}, {frame.light_id=}) {frame.sample_wi=})"
                )
            elif invalid == "ignore":
                valid_indices.append(i)
                wiid = frame.wiid
                print(
                    f"Frame {wiid} has invalid directions {frame.sample_wo=}  (for LED {frame.dmx_id=}, {frame.light_id=}) {frame.sample_wi=})"
                )

        # Remove frames with directions under the horizon
        self.frames = [self.frames[i] for i in valid_indices]
        # Build efficient packed arrays of directions
        self.frame_wo = np.stack(
            [frame.sample_wo for frame in self.frames], axis=0, dtype=np.float32
        )
        self.frame_wi = np.stack(
            [frame.sample_wi for frame in self.frames], axis=0, dtype=np.float32
        )
        self.frame_wi_index = np.array(
            [frame.wiid for frame in self.frames], dtype=np.int32
        )
        self.frame_dmx_light_ids = np.array(
            [[frame.dmx_id, frame.light_id] for frame in self.frames], dtype=np.int32
        )

        # Replace the direction vectors with slices into the packed array
        for i, frame in enumerate(self.frames):
            frame.sample_wo = self.frame_wo[i]
            frame.sample_wi = self.frame_wi[i]

    def cropped_image_view( self, crop_tl_yx=(0, 0), crop_br_yx=(None, None), mask=None) -> "VarisCaptureCroppedView":
        """
        A cached loader of images in this capture, with an optional crop.
        """
        return VarisCaptureCroppedView(
            [frame.image_path for frame in self.frames],
            crop_tl_yx,
            crop_br_yx,
            mask=mask,
        )

    @classmethod
    def _normal_remove_invariants(cls, normal_orig, normal_rotated):
        x_orig, y_orig, z_orig = (normal_orig[..., i] for i in (0, 1, 2))
        inv_mask_orig = (z_orig > np.abs(x_orig)) & (z_orig > np.abs(y_orig))

        x_r, y_r, z_r = (normal_rotated[..., i] for i in (0, 1, 2))
        inv_mask_rot = (z_r < np.abs(x_r)) | (z_r < np.abs(y_r))

        inv_mask = inv_mask_rot & inv_mask_orig
        if np.count_nonzero(inv_mask) < np.prod(inv_mask.shape) * 0.35:
            normal_rotated[inv_mask] = normal_orig[inv_mask]

        return normal_rotated

    def _determine_macro_normal(self, probe_crop_tl=(0, 0), probe_crop_br=(None, None)):
        self.probe_crop = (
            slice(probe_crop_tl[0], probe_crop_br[0]),
            slice(probe_crop_tl[1], probe_crop_br[1]),
        )

        for sp in self.stage_poses.values():
            normal_mean = np.mean(
                self.normal_map_per_pose(
                    sp.wiid,
                    rotate_theta=False,
                    rotate_mean=False,
                    rotate_phi=False,
                    probe=True,
                ),
                axis=(0, 1),
            )
            normal_mean_after_theta = Rotation.from_euler("x", sp.theta_i).apply(
                normal_mean
            )
            sp.R_to_macronormal = self._rotation_to_z(normal_mean_after_theta)


    def normal_map_for_pose(
        self,
        wiid: tuple[int, int],
        rotate_theta=True,
        rotate_mean=True,
        rotate_phi=True,
        probe=None,
    ) -> np.ndarray:
        sp = self.stage_poses[wiid]
        normal = sp.normals_image.astype(np.float32) * 2 - 1

        if probe:
            normal = normal[self.probe_crop]

        h, w = normal.shape[:2]
        normals_flat = einops.rearrange(normal, "h w c -> (h w) c", c=3)

        R = np.eye(3, dtype=np.float32)

        if rotate_theta:
            R_theta = Rotation.from_euler("x", sp.theta_i)
            R = R_theta.as_matrix() @ R
            # normals_flat = R_theta.apply(normals_flat)
            # print(f"{dt1} -> {normals_flat.dtype}")

            # normals_flat = self.R_macro_to_z.apply(normals_flat)

        if rotate_mean:
            # normals_flat = self.frame_R_to_macronormal[idx].apply(normals_flat)
            R = sp.R_to_macronormal.as_matrix() @ R

        if rotate_phi:
            R_phi = Rotation.from_euler("z", np.pi - sp.phi_i)
            # normals_flat = R_phi.apply(normals_flat)
            R = R_phi.as_matrix() @ R

        normals_flat = Rotation.from_matrix(R).apply(normals_flat).astype(np.float32)

        # if rotate_mean:
        #     assert rotate_theta and rotate_phi
        #     # TODO use crop here
        #     mean = np.mean(normals_flat, axis=0) if probe else np.mean(normals_flat[self.probe_mask.reshape(-1)], axis=0)
        #     mean /= np.linalg.norm(mean)
        #     # Rotation vector from `mean` to [0, 0, 1]
        #     R_mean = self._rotation_to_z(mean)

        #     normals_flat = R_mean.apply(normals_flat)

        # print(f"{dt1} -> {normals_flat.dtype}")

        normals_flat /= np.linalg.norm(normals_flat, axis=1)[:, None]
        normals_out = einops.rearrange(normals_flat, "(h w) c -> h w c", h=h, w=w, c=3)
        # normals_out = np.clip(normals_out, -1, 1)

        normals_out = self._normal_remove_invariants(normal, normals_out)

        return normals_out

    def write_refined_normal(self, wiid, path=None):
        normal = self.normal_map(wiid)
        sp = self.stage_poses[wiid]

        path = (
            path
            or sp.normals_image_path.parent
            / f"NormalRefined_Retro_{sp.name}_theta{sp.theta_i_index:03d}-phi{sp.phi_i_index:03d}.exr"
        )
        writeImage(normal, path)

    def write_refined_normals(self):
        for i in tqdm(range(len(self.frames))):
            self.write_refined_normal(i)


    @staticmethod
    def _mirror_light_directions_for_isotropic(wos: np.ndarray):
        """Extend light directions to add a copy mirrored around the Y axis.

        Camera direction is in the Y=0 plane.
        Isotropic material is therefore symmetric about the Y axis.        
        """
        wos_iso_symmetric = wos.copy()
        wos_iso_symmetric[:, 1] *= -1
        return np.concatenate([wos, wos_iso_symmetric], axis=0)

    def _marker_brightness_path(self, wiid: tuple[int, int]) -> Path:
        return self.dir_src / "009_white_level" / f"MarkerBrightness_theta{wiid[0]:03d}_phi{wiid[1]:03d}.h5"

    def plot_directions(self, theta_i=None, phi_i=None, color="wiwo", plot_wi=True, plot_wo=True, plot_gradient_board_normals=False):
        plot = VarisPlot()
        if color == "wiwo":
            colors = plot.plotly_colors_sequence_for_labels(self.frame_wi_index[:, 0])
        elif color == "dmx":
            colors = plot.plotly_colors_sequence_for_labels(
                np.array([frame.dmx_id for frame in self.frames])
            )

        mask = np.ones(len(self.frame_wi_index), dtype=bool)
        if theta_i is not None:
            mask = mask & (self.frame_wi_index[:, 0] == theta_i)
        if phi_i is not None:
            mask = mask & (self.frame_wi_index[:, 1] == phi_i)

        print(f"{theta_i=} {phi_i=}: {np.sum(mask)} frames")

        wis = self.frame_wi[mask]
        wos = self.frame_wo[mask]
        colors = colors[mask]

        # TODO expose as param
        if self.is_symetric_along_X:
            wos = self._mirror_light_directions_for_isotropic(wos)
            colors = np.concatenate([colors, colors], axis=0)

        if plot_wi:
            plot.add_points(wis, name="wi", marker=dict(size=10, color=colors))
        if plot_wo:
            plot.add_points(wos, name="wo", marker=dict(size=3, color=colors))

        if plot_gradient_board_normals:
            sps = [sp for wid, sp in sorted(self.stage_poses.items())]
            # grad_board_normals = np.array([
            #     self.frames[sp.frame_indices[0]].r_dome_to_sample.apply(sp.board_gradient_normal) 
            #     for sp in sps
            # ])
            grad_board_normals = np.array([sp.board_gradient_normal for sp in sps])
            plot.add_points(grad_board_normals, name="grad normals", marker=dict(size=5, color=plot.plotly_colors_sequence(len(grad_board_normals))))

        return plot

    def _plot_board_gradient_normals_for_sp(self, sp: OLATStagePose):
        anchor_img = sp.anchor_image

        radius = round(min(anchor_img.shape[:2]) // 2 - 64 - 128)

        normals_img = sp.normals_image
        normals = 2*normals_img - 1

        # white_board_mask = np.mean(anchor_img, axis=2) > 150
        # white_board_mask[128:-128, 128:-128] = False

        normals_on_white = normals[sp.board_white_mask]
        normal_mean = np.mean(normals_on_white, axis=0)
        # print(normal_mean, [int(n) for n in normal_mean])
        normal_std = np.std(normals_on_white, axis=0)

        # show([normals_img, anchor_img, white_board_mask])

        img_demo = np.zeros_like(anchor_img)
        img_demo[sp.board_white_mask] = (np.clip(normals_on_white * 0.5 + 0.5, 0., 1.) * 255).astype(np.uint8)

        # Circle in center of image based on its size
        img_center_tuple = (img_demo.shape[1]//2, img_demo.shape[0]//2)

        cv.circle(img_demo, img_center_tuple, radius, (255, 255, 255), 2)
        cv.circle(img_demo, img_center_tuple, radius//25, (255, 255, 255), -1)

        # Circle offset from center of by the normal XY
        img_normal_tuple = (img_center_tuple[0] + int(normal_mean[0]*radius), img_center_tuple[1] + int(normal_mean[1]*radius))
        normal_color = tuple(int(n*255) for n in normal_mean * 0.5 + 0.5)

        cv.ellipse(img_demo, img_normal_tuple, (int(normal_std[0]*radius), int(normal_std[1]*radius)), 0, 0, 360, normal_color, -1)
        return img_demo

    def plot_board_gradient_normals(self, path_out: Path = None):
        from moviepy.editor import ImageSequenceClip

        path_out = path_out or self.dir_src/"003_board_normals.mp4"
        dir_frames = path_out.parent / path_out.stem
        dir_frames.mkdir(exist_ok=True)

        frames = []
        for wiid, sp in self.stage_poses.items():
            t, p = wiid
            img = self._plot_board_gradient_normals_for_sp(sp)
            frames.append(img)
            writeImage(img, dir_frames/f"NormalDemo_thetaI{t:03d}_phiI{p:03d}.webp", verbose=False)

        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(str(path_out), codec="libx264", fps=10)

    def plot_board_normals_gradient_vs_theoretical(self):
        sps = [sp for wid, sp in sorted(self.stage_poses.items())]
        
        # grad_board_normals = np.array([
        #     self.frames[sp.frame_indices[0]].r_dome_to_sample.apply(sp.board_gradient_normal) 
        #     for sp in sps
        # ])
        board_normals_grad = np.array([sp.board_gradient_normal for sp in sps])

        z = np.array([0, 0, 1], dtype=np.float32)
        
        board_normals_theoretical = np.array([self.frames[sp.frame_indices[0]].r_dome_to_sample.inv().apply(z) for sp in sps])
        board_normals_theoretical[:, 2] *= -1

        fig, ax = pyplot.subplots(1, 1, figsize=(10, 10))
        ax.plot(board_normals_grad[:, 0], board_normals_grad[:, 2], 'rx', label="Gradient $n$")
        ax.plot(board_normals_theoretical[:, 0], board_normals_theoretical[:, 2], 'b+', label="Theoretical $n$")


        thetas_theoretical = np.array([sp.theta_i for sp in sps])
        thetas_grad = np.arctan2(board_normals_grad[:, 0], board_normals_grad[:, 2])

        theta_diffs = thetas_grad - thetas_theoretical
        theta_diff_mean = np.mean(theta_diffs)
        theta_diff_median = np.median(theta_diffs)
        theta_diff_std = np.std(theta_diffs)

        print("θ theo:", "  ".join(f"{th:.1f}°" for th in np.rad2deg(thetas_theoretical)))
        print("θ grad:", "  ".join(f"{th:.1f}°" for th in np.rad2deg(thetas_grad)))
        print(f"(θ grad - θ theo) = ({np.rad2deg(theta_diff_mean):.2f} +- {np.rad2deg(theta_diff_std):.2f})°")
        print(f"median = {np.rad2deg(theta_diff_median):.2f}°")

        def plot_thetas_adjusted(name, offset, radius=1.0, fmt="g."):
            thetas_adjusted = thetas_theoretical + offset
            print(f"{name}: ", "  ".join(f"{th:.1f}°" for th in np.rad2deg(thetas_adjusted)))
            ax.plot(radius*np.sin(thetas_adjusted), radius*np.cos(thetas_adjusted),  fmt, label=name)

        plot_thetas_adjusted("θ theo + mean", theta_diff_mean, radius=0.98, fmt="g.")
        plot_thetas_adjusted("θ theo + median", theta_diff_median, radius=0.95, fmt="g+")
        fig.legend()

        return fig


    def plot_crops_in_theta_phi(
            self, 
            view, 
            pose: Union[Literal["all"], tuple[int, int]] = "all", 
            tonemap: bool=True, 
            tile_px:float=20, 
            allow_symmetrization: bool = False, 
            show_outliers: bool = False,
            out_path_override: str | Path | None = None,
            correction_factors = None,
    ):
        seaborn.set_theme()
        fig, ax = pyplot.subplots(1, 1, figsize=(20, 10))

        if pose == "all":
            sp_indices = range(len(self.frames))
            wi = None
        elif isinstance(pose, tuple):
            sp_indices = self.stage_poses[pose].frame_indices
            wi = self.frame_wi[sp_indices[0]]
        else:
            raise ValueError(f"Invalid pose: {pose}")

        wos = self.frame_wo[sp_indices]

        if allow_symmetrization and self.is_symetric_along_X:
            sp_indices = np.concatenate([sp_indices, sp_indices], axis=0)
            wos = self._mirror_light_directions_for_isotropic(wos)

        def w_to_theta_phi(w, allow_negatives=False, deg=True):
            thetas = np.arccos(w[..., 2])

            # Chosen so that camera direction is phi=0
            phis = np.arctan2(w[..., 0], w[..., 1])

            if not allow_negatives:
                # Cast phis to 0...360 range
                phis = (phis + (2 * np.pi)) % (2 * np.pi)
            
            if deg:
                thetas = np.rad2deg(thetas)
                phis = np.rad2deg(phis)
            
            return thetas, phis

            
        thetas, phis = w_to_theta_phi(wos)


        ax.scatter(phis, thetas)


        color_stack = np.stack([view[fr_ind] for fr_ind in sp_indices], axis=0)

        if correction_factors is not None:
            color_stack *= correction_factors[:, None, None, None]

        print(color_stack.shape)
        if tonemap:
            color_stack = RGL_tonemap_uint8(color_stack)
        else:
            color_stack = (np.clip(color_stack, 0, 1)*255).astype(np.uint8)

        for fr_ind, theta, phi, crop_img in zip(sp_indices, thetas, phis, color_stack):
            plot_pos = (phi, theta)
            fr = self.frames[fr_ind]

            

            if fr.is_valid or show_outliers:
                ax.add_artist(
                    AnnotationBbox(
                        OffsetImage(crop_img, zoom=tile_px/crop_img.shape[0]), 
                        plot_pos, 
                        frameon = not fr.is_valid,
                        bboxprops = dict(facecolor="red"),
                        zorder=1,
                    ),
                )

        

        if wi is not None:
            wi_theta, wi_phi = w_to_theta_phi(wi)

            color_camera = "orange"
            # Draw a star at (wi_phi, wi_theta)
            ax.scatter(wi_phi, wi_theta, marker="*", color=color_camera, s=100, label="Camera direction", zorder=2)
            # Add "Camera direction" text next to the star
            ax.text(wi_phi, wi_theta, "Camera direction", fontsize=12, ha="left", va="bottom", color=color_camera)

            # reflect wi around Z
            wi_ref = wi.copy()
            wi_ref[:2] *= -1
            wir_theta, wir_phi = w_to_theta_phi(wi_ref)

            # Draw a star at (wir_phi, wir_theta)
            ax.scatter(wir_phi, wir_theta, marker="*", color="red", s=100, label="Reflected direction", zorder=2)
            # Add "Reflected camera direction" text next to the star
            ax.text(wir_phi, wir_theta, "Reflected camera direction", fontsize=12, ha="left", va="bottom", color="red")

        if name := getattr(view, "region_name", ""):
            title = f"{self.name} - region {name}"
            if wi is not None:
                title += f" - $\\theta_i$ = {wi_theta:.1f}$^O$"
            ax.set_title(title)

        # ax.set_xlim(-180, 180)
        # ax.set_ylim(0, 90)
        ax.invert_yaxis()
        # ax.set_xlabel("φ")
        # ax.set_ylabel("θ")
        ax.set_xlabel("$\\phi_o$ [$^O$]")
        ax.set_ylabel("$\\theta_o$ [$^O$]")
        ax.set_facecolor('grey')
        fig.tight_layout()
        # ax.set_xlim(-np.pi, np.pi)
        # ax.set_ylim(0, 0.5*np.pi)

        dir_out = Path(out_path_override) if out_path_override else self.dir_src / "005_plots"
        dir_out.mkdir(exist_ok=True)
        pose_str = pose if isinstance(pose, str) else "".join(str(i) for i in pose)
        outlier_str = "_outliers" if show_outliers else ""

        for fmt in ("png", "pdf"):
            fig.savefig(dir_out / f"{self.name}_wi{pose_str}__crops_{name}_theta_phi{outlier_str}.{fmt}")

        return fig

    def plot_crops_in_3d(self, view, tonemap:bool=True):
        
        plot = VarisPlot()

        sp_indices = range(len(self.frames))
        wos = self.frame_wo

        crop_means = np.array([np.mean(view[i], axis=(0, 1)) for i in sp_indices])
        crop_means = (np.clip(crop_means, 0, 1) * 255).astype(np.uint8)

        colors = plot.plotly_colors_from_rgb(crop_means)

        plot.add_points(wos, marker=dict(size=3, color=colors))


        return plot

    # def plot_white_brightness(self, ax=None):
    #     if ax is None:
    #         fig, ax = pyplot.subplots()
    #     thetas, intensity = self.measure_white_brightness
    #     intensity = np.mean(intensity, axis=-1)

    #     ax.scatter(np.rad2deg(thetas), intensity)
    #     ax.set_xlabel('Theta')
    #     ax.set_ylabel('Mean intensity of white in all-lights')
    #     ax.set_title('Brightness vs Theta')
    #     # Start Y axis from 0
    #     ax.set_ylim(bottom=0)
    #     return ax

@lru_cache(maxsize=2048)
def _view_read_image(path: Path, crop_tl: tuple[int, int], crop_br: tuple[int, int]):
    crop = slice(crop_tl[0], crop_br[0]), slice(crop_tl[1], crop_br[1])
    return np.ascontiguousarray(
        readImage(str(path))[crop]
    )

class VarisCaptureCroppedView:
    def __init__(self, image_paths, crop_tl=(0, 0), crop_br=(None, None), mask=None):
        """
        By default get the whole image `img[0:None, 0:None]`
        """
        self.crop_tl = crop_tl
        self.crop_br = crop_br
        self.crop = slice(crop_tl[0], crop_br[0]), slice(crop_tl[1], crop_br[1])
        self.image_paths = image_paths
        self.mask = mask

        self._cache = [None] * len(image_paths)

    def __getitem__(self, index):
        if self._cache[index] is None:
            img_crop = _view_read_image(self.image_paths[index], self.crop_tl, self.crop_br)
            if self.mask is not None:
                img_crop = img_crop[self.mask].mean(axis=0)
            self._cache[index] = img_crop

        return self._cache[index]

def load_standard_capture(cap_name, retro_kwargs: None | dict[str, Any] = None, olat_kwargs: None | dict[str, Any] = None) -> tuple[VarisCapture, VarisCapture]:
    from Paths import PATH_DATA
    from Space.olat import OLATCapture, ThetaDistribution
    from Space.retro import RetroreflectionCapture

    dir_retro = PATH_DATA / "Retroreflection" / "128x1" / cap_name
    dir_olat = PATH_DATA / "FullScattering" / cap_name


    retro = RetroreflectionCapture(
        dir_src = dir_retro,
        num_theta_i=128,
        num_phi_i=1,
        # theta_distribution=ThetaDistribution(mode=ThetaDistribution.MODE_U_TO_THETA, offset_rad=np.deg2rad(10.43)),
        frame_below_horizon="ignore",
        **(retro_kwargs or {}),
    )
    print(retro.report())

    olat = OLATCapture(
        dir_src = dir_olat,
        num_theta_i=8,
        num_phi_i=1,    
        # theta_distribution=ThetaDistribution(mode=ThetaDistribution.MODE_UNIFORM, offset_rad=np.deg2rad(16.35)),
        # frame_below_horizon="remove",
        theta_distribution=ThetaDistribution(mode=ThetaDistribution.MODE_UNIFORM),
        frame_below_horizon="ignore",
        **(olat_kwargs or {}),
    )
    print(olat.report())
    return retro, olat
