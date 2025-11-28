import re
from concurrent.futures import as_completed
from dataclasses import dataclass
from functools import cached_property, partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas
from matplotlib import pyplot
from scipy.spatial import KDTree
from tqdm import tqdm

from ..Utilities.ImageIO import RGL_tonemap_uint8, readImage, writeImage
from ..Utilities.show_image import image_montage_same_shape
from .capture import CaptureFrame, CaptureStagePose, ThetaDistribution, VarisCapture
from .light_index import DomeLightIndex
from .olat_subcrops import OLATPixelMaskView


@dataclass
class LightQueryResult:
    # For a query of NN nearest neighbours and K wos:

    # wo directions we queried by, Kx3
    query_directions: np.ndarray
    # OLAT frames which are closest by cartesian distance in wo in sample space, K x NN
    frame_indices: np.ndarray
    # Distances in wo in sample space, K x NN
    distances: np.ndarray
    # Weights for blending the frames, K x NN
    weights: np.ndarray
    # Result OLAT frames but indicated by DMX and light ID, K x NN x 2
    frame_dmx_and_light_ids: np.ndarray

    @property
    def num_frames(self):
        return self.frame_indices.shape[1]

    @property
    def num_query_points(self):
        return self.query_directions.shape[0]



class OLATCapture(VarisCapture):
    """
    * Iterate incident and outgoing directions for OLAT capture
    * Iterate directions for retroreflection capture
    * Convert between image filenames and incident/outgoing directions
    * Load stored captures from file system


    Apply theta offset:
    ```
    capture_adjusted = OLATCapture(
        dir_src = dir_src,
        num_theta_i=8,
        num_phi_i=1,    
        frame_below_horizon="ignore", # maybe remove?
        theta_distribution=ThetaDistribution(mode=ThetaDistribution.MODE_UNIFORM, offset_rad=np.deg2rad(16.35)),
    )
    ```
    """

    @property
    def is_symetric_along_X(self) -> bool:
        return self._symmetrize_light_directions_if_isotropic and self.is_isotropic

    def __init__(
            self, 
            dir_src: Path,
            num_theta_i: int = 8,
            num_phi_i: int = 1,
            symmetrize_light_directions_if_isotropic=True,
            conflate_phis=False,
            drop_outliers=True,
            use_index=True,
            theta_distribution: ThetaDistribution=ThetaDistribution(mode=ThetaDistribution.MODE_ARC_COS, offset_rad=0.0),
        ):
            
        dir_src = Path(dir_src)
        super().__init__(dir_src, num_theta_i, num_phi_i, theta_distribution=theta_distribution)
        self.conflate_phis = conflate_phis
        self._symmetrize_light_directions_if_isotropic = symmetrize_light_directions_if_isotropic

        if (not use_index) or  not self.load_index():
        # if True:
            self._load_frames_from_dir_without_index(dir_src)
            self._derive_frame_coordinates()

        # if (not use_index) or  not self.load_index():
            # pass

        if drop_outliers:
            self._drop_outliers()

        self._build_stage_pose_index()

        # if drop_outliers:
            # self._drop_outliers()

        # self._build_query_index()

        self.load_named_regions(dir_src)

        print(f"OLAT: {len(self.frames)} frames in {dir_src}")

        

    def _add_frame(self, path:Path, name:str, theta_id:str, phi_id:str, dmx_id:str, light_id:str):
        wi_id = (int(theta_id), int(phi_id))

        if self.conflate_phis:
            phi_id = 0
               
        theta_i = self.olat_theta_i[wi_id[0]]
        phi_i = self.olat_phi_i[wi_id[1]]
        dmx_id = int(dmx_id)
        light_id = int(light_id)

        self.frames.append(
            CaptureFrame(
                dmx_id=dmx_id,
                light_id=light_id,
                theta_i=theta_i,
                phi_i=phi_i,
                wiid=wi_id,
                image_path=path,
            )
        )
        # ensure the stage pose is allocated
        self._get_stage_pose(wi_id, name=name, create=True)


    def _handle_file_olat(self, path: Path, match: re.Match, is_iso=False):
        if is_iso:
            name, theta_id, dmx_id, light_id = match.groups()
            phi_id = "0"
        else:
            name, theta_id, phi_id, dmx_id, light_id = match.groups()

        self._add_frame(path, name, theta_id, phi_id, dmx_id, light_id)


    def _load_frames_from_dir_without_index(self, dir_src: Path):
        RE_OLAT_FILE = r"Full_([a-zA-Z]+)_wi(\d+)_dmx(\d+)_light(\d+)_\.exr"
        RE_OLAT_FILE_ANISO = r"Full_([a-zA-Z]+)_thetaI(\d+)-phiI(\d+)_dmx(\d+)_light(\d+)_\.exr"
        # Photo with all lights on, for alignment, ie "Full_ButterflySwallowtailAniso_thetaI000-phiI000.jpg"
        RE_ALL_LIGHTS = r"Full_([a-zA-Z]+)_wi(\d+)_gradientA_\.exr"
        RE_ALL_LIGHTS_ANISO = r"Full_([a-zA-Z]+)_thetaI(\d+)-phiI(\d+)\.jpg"
        # Single exposure photo per stage pose
        RE_ANCHOR = r"Full_([a-zA-Z]+)_wi(\d+)_rectify\.jpg"
        RE_ANCHOR_ANISO = r"Full_([a-zA-Z]+)_thetaI(\d+)-phiI(\d+)_rectify\.jpg"
        # Normals estimated using gradient patterns
        RE_NORMALS_FROM_GRAD = r"Full_([a-zA-Z]+)_wi(\d+)_normal01\.exr"
        RE_NORMALS_FROM_GRAD_ANISO = r"Normal_Full_([a-zA-Z]+)_thetaI(\d+)-phiI(\d+)\.exr"

        RE_GRADIENT = r"Full_([a-zA-Z]+)_wi(\d+)_gradient([AXYZ])_\.exr"

        self._ingest_files(Path(dir_src), {
            RE_OLAT_FILE: partial(self._handle_file_olat, is_iso=True),
            RE_OLAT_FILE_ANISO: partial(self._handle_file_olat, is_iso=False),
            RE_ANCHOR: partial(self._handle_file_anchor, is_iso=True),
            RE_ANCHOR_ANISO: partial(self._handle_file_anchor, is_iso=False),
            RE_ALL_LIGHTS: partial(self._handle_file_all_lights, is_iso=True),
            RE_ALL_LIGHTS_ANISO: partial(self._handle_file_all_lights, is_iso=False),
            RE_NORMALS_FROM_GRAD: partial(self._handle_file_grad_normals, is_iso=True),
            RE_NORMALS_FROM_GRAD_ANISO: partial(self._handle_file_grad_normals, is_iso=False),
            RE_GRADIENT: partial(self._handle_file_gradient, is_iso=True),
        })

    def _load_frames_synthetic(self, dir_src: Path):
        """
        Produce all valid OLAT frames - that is
        the 7 valid camera directions (ignoring the 90deg full grazing angle)
        and all lights above the sample horizon.
        """
        # workdir = Path(workdir)
        # dir_rectified = workdir / "rectified"
        name = dir_src.name

        di = DomeLightIndex.default_instance()

        self.frames = [
            CaptureFrame(
                dmx_id=dmx_id,
                light_id=light_id,
                theta_i=self.olat_theta_i[theta_id],
                phi_i=self.olat_phi_i[phi_id],
                wiid=(theta_id, phi_id),
                image_path=dir_src /
                f"Full_{name}_thetaI{theta_id:03d}-phiI{phi_id:03d}_dmx{dmx_id}_light{light_id:03d}_.exr",
                # / f"Full_{name}_wi{incident_idx:02d}_dmx{dmx_id}_light{light_id:03d}_.exr",
            )
            for dmx_id, light_id in di.lights_dmx_and_index
            for phi_id in range(self.num_phi_i)
            for theta_id in range(self.num_theta_i)
        ]

        self.stage_poses = {
            (theta_id, phi_id): CaptureStagePose(theta_i_index=theta_id, phi_i_index=phi_id)
            for phi_id in range(self.num_phi_i)
            for theta_id in range(self.num_theta_i)
        }         

    def _set_angles(self, num_theta_i, num_phi_i, theta_distribution: ThetaDistribution):
        if theta_distribution.mode == ThetaDistribution.MODE_ARC_COS:
            # Cosine-based theta distribution
            self.olat_theta_i = np.pi * 0.5 - np.arccos(np.linspace(0, 1, num_theta_i))
        elif theta_distribution.mode == ThetaDistribution.MODE_UNIFORM:
            # Angularly uniform theta distribution
            self.olat_theta_i = np.linspace(0, np.pi * 0.5, num_theta_i, endpoint=True)
        else: 
            raise NotImplementedError(theta_distribution)
            
        self.olat_theta_i += theta_distribution.offset_rad

        # Uniformly spaced phi distribution
        self.olat_phi_i = np.linspace(-np.pi, np.pi, num_phi_i, endpoint=False)
        
    def _derive_frame_coordinates(self):
        di = DomeLightIndex.default_instance()

        for frame in self.frames:
            # Transform from stage rotation
            # wi_theta_idx, wi_phi_idx = frame.wiid
            r_to_sample = self.make_R_domeEnvYUp_to_sample(frame.theta_i, frame.phi_i)
            # print('rotation matrix', r_to_sample.as_matrix())
            frame.r_dome_to_sample = r_to_sample

        for frame in self.frames:
            r_to_sample = self.make_R_domeEnvYUp_to_sample(frame.theta_i, frame.phi_i)
            frame.r_dome_to_sample = r_to_sample
            frame.sample_wo = frame.sample_wi = r_to_sample.apply(np.array([0, 0, -1]))

            # Find and transform light direction
            table_idx = di.get_index(frame.dmx_id, frame.light_id)
            # get light coordinates in cartesian dome space
            light_envMapYUp = di.lights_envMapYUp[table_idx]
            # rotate to sample space
            light_sample = r_to_sample.apply(light_envMapYUp)
            frame.sample_wo = light_sample

            # Transform camera direction
            frame.sample_wi = r_to_sample.apply(np.array([0, 0, -1]))

            # if self.is_isotropic:
            #     # Camera direction is in the Y=0 plane
            #     # Isotropic material is therefore symmetric about the Y axis
            #     frame.sample_wo[1] = np.abs(frame.sample_wo[1])
            #     print("isotropic ", frame.sample_wi)

        super()._derive_frame_coordinates()






    def _build_stage_pose_index(self):
        """
        Each wi has a whole hemisphere of OLAT frames.
        For each frame, we will build a tree for lookups.
        All frames are in the single array `self.frames`, so we will find subsets.

        Frame subsets for each wi index - which frame indices belong to a given wi
        Find unique entries in `self.frame_wi_index` and group indices by them
        This pandas operation yields a dictionary exactly like we need:
         {(0, 0): [0, 1, 2, ...], (0, 1): [3, 4, 5, ...], (0, 2): [6, 7, 8, ...]
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
        """
        frame_indices_by_wiid = self.frame_table.reset_index(drop=True).groupby(["theta_id", "phi_id"]).groups

        for wiid, frame_indices in frame_indices_by_wiid.items():
            sp = self._get_stage_pose(wiid, create=True)
            sp.frame_indices = np.array(frame_indices)

    # def _drop_outliers(self):
    #     for sp in self.stage_poses.values():
    #         mask = np.array([self.frames[i].is_valid for i in sp.frame_indices])
    #         print(f"Wiid{sp.wiid}: dropped outliers {np.count_nonzero(~mask)}/{len(mask)}")
    #         sp.frame_indices = sp.frame_indices[mask]



    def _build_query_index(self):
        for sp in self.stage_poses.values():
            wos = self.frame_wo[sp.frame_indices]
            if self.is_symetric_along_X:
                # Add extra entries for mirrored light directions since BRDF is supposed to be symmetric about Y
                wos = self._mirror_light_directions_for_isotropic(wos)

            sp.wo_kdtree = KDTree(wos.astype(np.float64))

    def _frame_blending_weights(self, distances: np.ndarray):
        """
        Args:
            distances: N x num_lights
        """

        # Careful about inf distances
        mask = np.isfinite(distances)
        distance_sum = np.sum(distances, axis=1, where=mask)
        
        # Reweight arrays of weights to add up to 1, avoiding the areas out of bounds (with the inf weight).
        weights = 1.0 - distances / distance_sum[:, None]
        
        #max_weights = weights[:, 0]   # first neighbor from the KD-tree
        #weights = weights / max_weights[:, None]   # now closest is weighed as 1.0
       
        sum_weights = np.sum(weights, axis=1, where=mask)
        weights = weights / sum_weights[:, None]   # now closest is weighed as 1.0
        
        # print('new sum weights (should be 1s)',  np.sum(weights, axis=1, where=mask))

        # 1-inf yields -inf so we filter that out
        weights = weights.clip(0, 1)
        return weights

    def query_whole_olat_frames_by_wos(self,
                                       wi_index: tuple[int, int],
                                       wo: np.ndarray,
                                       num_lights: int = 3,
                                       max_angle_rad: float = np.deg2rad(15),
                                       ) -> LightQueryResult:
        """
        Args:
            wi_index: integer index of theta rotation 0..7
            wo: query light directions in sample space, N x 3
            num_lights: number of nearest neighbours to find
        """
        if not isinstance(wo, np.ndarray):
            # Convert list to array
            wo = np.array(wo)

        if len(wo.shape) == 1:
            # Convert 1D vector into [1x3]
            wo = wo[None, :]

        # ensure Nx3 shape
        assert wo.shape[1] == 3

        # Normalize direction
        wo = wo / np.linalg.norm(wo, axis=1)[:, None]

        # Select frames and lookup tree for this wi
        sp = self.stage_poses[wi_index]
       
        # TODO How to handle a bad stage pose (when kdtree is None)?
        if sp.wo_kdtree is None:
            bad = np.ones(1) * -1
            return LightQueryResult(wo, bad, bad, bad)

        # Query the tree in parallel
        result_distances, result_ind = sp.wo_kdtree.query(
            wo,  # query points
            k=num_lights,  # number of nearest neighbours
            workers=( 1 if len(wo) == 1 else -1 ),  # use multiple cores if multiple queries
            # Maximum angle difference to maximum distance on sphere surface
            # Ignore curvature and approximate with arc length which is `angle_rad * radius`.
            distance_upper_bound = max_angle_rad,
        )

        # For isotropic, we extended the kdtree with a mirrored copy of the light directions.
        # For N frames, we have 0..N-1 as the original frames and N..2N-1 as the mirrored frames.
        # The modulo N operation will convert to the original frame indices.
        if self.is_symetric_along_X:
            result_ind = result_ind % len(sp.frame_indices)

        # Now grab the final frame indices relevant to this query.
        chosen_indices = sp.frame_indices[result_ind]

        # if max_angle_rad > 0.:
        #     # Cosine distances
        #     result_wos = self.frame_wo[chosen_indices]
        #     # Dot product of wo and wos
        #     cos_distances = np.dot(wo, result_wos.T)
        #     # Filter out angles which are too big - cosines too small
        #     mask = cos_distances >= np.cos(max_angle_rad)
        #     # If all results are too far, still return the first one
        #     if np.count_nonzero(mask) == 0:
        #         mask[0] = True
        
        #     result_ind = result_ind[mask]
        #     chosen_indices = chosen_indices[mask]
        #     result_distances = result_distances[mask]



        return LightQueryResult(
            query_directions=wo,
            # We were querying a subset of frames (which match the wi), so we convert to global indices in `self.frames`
            frame_indices=chosen_indices,
            distances=result_distances,
            frame_dmx_and_light_ids=self.frame_dmx_light_ids[chosen_indices],
            # If 1 neighbour, then weight = [1], else calculate weights inverse to distance
            weights=(
                self._frame_blending_weights(result_distances)
                if num_lights > 1
                else np.ones(wo.shape[0])
            ),
        )

    @cached_property
    def file_path_list(self) -> list[Path]:
        """List of all OLAT measurement image paths."""
        frame_paths = [f.image_path for f in self.frames]
        frame_paths.sort()
        return frame_paths
