
import re
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..Utilities.ImageIO import RGL_tonemap, readImage, writeImage
from .capture import CaptureFrame, ThetaDistribution, VarisCapture
from .sample_coordinates import u_to_theta

# class RetroAllLightNaming:
#     """Photo with all lights on, for alignment
#         ie "Full_ButterflySwallowtailAniso_thetaI000-phiI000.jpg"
#     """
#     JUST_JPG = r"Retro_([a-zA-Z]+)_theta(\d+)-phi(\d+)\.jpg"
#     GRAD_A = r"Retro_([a-zA-Z]+)_theta(\d+)-phi(\d+)_gradientA255\.jpg"


class RetroreflectionCapture(VarisCapture):
    def _set_angles(self, num_theta_i, num_phi_i, theta_distribution):
        # The last theta (theta_idx=15) is not 90 degrees but the one before that
        if theta_distribution.mode != ThetaDistribution.MODE_U_TO_THETA:
            raise NotImplementedError(f"Retro must have u_to_theta distribution but has {theta_distribution.mode}")

        self.olat_theta_i = u_to_theta(np.linspace(0, 1, num_theta_i, endpoint=False))

        self.olat_theta_i += theta_distribution.offset_rad

        if num_phi_i == 1:
            self.olat_phi_i = np.array([np.pi], dtype=np.float32)
        else:
            self.olat_phi_i = np.linspace(0, 2 * np.pi, num_phi_i, endpoint=False)

    def _handle_file_measure(self, path: Path, match: re.Match, is_iso=False):
        if is_iso:
            name, theta_id = match.groups()
            phi_id = "0"
        else:
            name, theta_id, phi_id = match.groups()
        wi_id = (int(theta_id), int(phi_id))

        self.frames.append(
            CaptureFrame(
                wiid=wi_id,
                theta_i=self.olat_theta_i[wi_id[0]],
                phi_i=self.olat_phi_i[wi_id[1]],
                image_path=path,
            )
        )
        # ensure the stage pose is allocated
        sp = self._get_stage_pose(wi_id, name=name, create=True)

        assert sp.frame_indices is None
        sp.frame_indices = np.array([len(self.frames) - 1], dtype=np.int32)


    def _load_frames_from_dir_without_index(self, dir_src: Path):
        """
        Kind_Name_Pose_Image

        * Kind 
            * `Retro` for retroreflection
            * `Full` for OLAT reflectance
        * Name
            * `Spectralon`
            * `Feather{Species}`
            * `ButterflySwallowtail`
        * Pose
            * `rot000` ... `rot127` for isotropic retroreflection
            * `wi000` ... `wi007` for isotropic OLAT
            * `thetaI000-phiI000` ... `thetaI015-phiI003` for anisotropic OLAT
            * `theta000-phi000` ... `theta015-phi003` for anisotropic retroreflection
        * Image
            * `_.exr` for HDR reflectance
            * `_rectify.jpg` for single exposure JPG photo with all lights on, for alignment
            * `_gradientA_.exr` for photo with all lights on
            * `_gradientX_.exr` for photo with gradient pattern (X, Y, Z)
            * `_normal01.exr` for estimated normals using gradient patterns
        """

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

        prefix = r"Retro_([a-zA-Z]+)_rot(\d+)"
        prefix_aniso = r"Retro_([a-zA-Z]+)_theta(\d+)-phi(\d+)"


        self._ingest_files(dir_src, {
            # Retroreflection brightness
            fr"{prefix}_\.exr": partial(self._handle_file_measure, is_iso=True),
            fr"{prefix_aniso}_\.exr": partial(self._handle_file_measure, is_iso=False),
            # Single exposure photo per stage pose
            fr"{prefix}_rectify\.jpg": partial(self._handle_file_anchor, is_iso=True),
            fr"{prefix_aniso}_rectify\.jpg": partial(self._handle_file_anchor, is_iso=False),
            # All lights
            fr"{prefix}_gradientA_\.exr": partial(self._handle_file_all_lights, is_iso=True),
            fr"{prefix_aniso}_gradientA255\.jpg": partial(self._handle_file_all_lights, is_iso=False),
            # Normals estimated using gradient patterns
            fr"{prefix}_normal01\.exr": partial(self._handle_file_grad_normals, is_iso=True),
            fr"{prefix_aniso}_normal01\.exr": partial(self._handle_file_grad_normals, is_iso=False),
            r"Normal_Retro_([a-zA-Z]+)_theta(\d+)-phi(\d+)\.exr": partial(self._handle_file_grad_normals, is_iso=False),
        })



    def __init__(self, 
            dir_src, 
            num_theta_i,     # TODO rename to num_theta_r and num_phi_r
            num_phi_i,
            use_index=True,
            frame_below_horizon="error",
            theta_distribution=ThetaDistribution(mode=ThetaDistribution.MODE_U_TO_THETA),

        ): #, naming_all_light=RetroAllLightNaming.JUST_JPG):
        dir_src = Path(dir_src)
        super().__init__(dir_src=dir_src, num_theta_i=num_theta_i, num_phi_i=num_phi_i, theta_distribution=theta_distribution)

        if (not use_index) or not self.load_index():
            self._load_frames_from_dir_without_index(dir_src)
            self._derive_frame_coordinates()



    def _derive_frame_coordinates(self):

        for frame in self.frames:
            r_to_sample = self.make_R_domeEnvYUp_to_sample(frame.theta_i, frame.phi_i)
            frame.r_dome_to_sample = r_to_sample
            frame.sample_wo = frame.sample_wi = r_to_sample.apply(np.array([0, 0, -1]))

        super()._derive_frame_coordinates()

    def measure_image_for_pose(self, wiid: tuple[int, int], view=None, tonemap=False) -> np.ndarray:
        """
        Photos with all lights on in JPG format, captured for each theta-phi pair.
        Loaded on demand when first accessed and then cached.
        """

        view = view or self.cropped_image_view()
        img = view[self.stage_poses[wiid].frame_indices[0]]
        if tonemap:
            img = (RGL_tonemap(img) * 255).astype(np.uint8)
        return img
    
    def extract_stacks(self, named_tlxywh: dict[str, tuple[int, int, int, int]]):        
        num_frames = len(self.frames)
        stacks = {name: np.zeros((num_frames, h, w, 3), dtype=np.float32) for name, (tl_x, tl_y, w, h) in named_tlxywh.items()}
        slices = {name: (slice(tl_y, tl_y + h), slice(tl_x, tl_x + w)) for name, (tl_x, tl_y, w, h) in named_tlxywh.items()}

        frames_theta_sorted = sorted(self.frames, key=lambda f: f.theta_i)
        thetas = []

        for i in tqdm(range(num_frames), desc="Extracting stacks"):
            img = self.read_measurement_image(frames_theta_sorted[i])
            for name, sl in slices.items():
                stacks[name][i] = img[sl]
            thetas.append(frames_theta_sorted[i].theta_i)

        return stacks, np.array(thetas)

    def extract_stack(self, tl_xy, wh):
        tl_x, tl_y = tl_xy
        w, h = wh
        stacks, thetas = self.extract_stacks({"single": (tl_x, tl_y, w, h)})
        return stacks["single"], thetas

    # @cached_property
    # def measure_white_brightness(self):
    #     intensity = []
    #     thetas = []

    #     for sp_idx, sp in tqdm(self.stage_poses.items(), total=len(self.stage_poses)):
    #         fr = self.frames[sp.frame_indices[0]]
    #         self.read_measurement_image(fr) # this sets the white level on the frame
    #         intensity.append(fr.white_marker_intensity_mean)
    #         thetas.append(sp.theta_i)

    #     return np.array(thetas), np.array(intensity)




    # def retro_file(self):
    #     from PostProduction.EstimateRetroNDF_Standalone import runAllRetroFunctions
    #     runAllRetroFunctions(
    #         workingDir=,
    #         retroShape=,
    #         sessionName=,
    #         dowResW=,
    #         downResH=,            
    #     )
