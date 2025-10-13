from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Collection

import cv2 as cv
import imageio
import numpy as np
import pandas
from matplotlib import pyplot
from scipy.spatial import KDTree, SphericalVoronoi

from .dome_coordinates import DomeCoordinates

DIR_THIS = Path(__file__).absolute().parent
DIR_RESOURCES = DIR_THIS / "resources"


@dataclass
class LightQueryResult:
    query_directions: np.ndarray
    light_indices: np.ndarray
    distances: np.ndarray
    weights: np.ndarray

    @property
    def num_lights(self):
        return self.indices.shape[1]

    @property
    def num_query_points(self):
        return self.query_directions.shape[0]


@dataclass
class GradientResult:
    # direction towards the highest intensity
    gradient_dir_envMapYUp: np.ndarray
    # intesities
    intensities: np.ndarray
    # intensity indexed by (dmx, light_id)
    intensities_by_dmx: dict[tuple[int, int], float]


class DomeLightIndex:
    """ """

    light_info_table: pandas.DataFrame
    _idx_by_dmx_and_lightID: dict[int, dict[int, int]]

    @classmethod
    @cache
    def default_instance(cls) -> "DomeLightIndex":
        return cls()

    def __init__(
        self,
        table_csv_path: Path = DIR_RESOURCES / "DomeLightCoords.csv",
        env_map_image_path: Path = DIR_RESOURCES / "light_env_map_annotated.jpg",
    ):
        self.light_info_table = pandas.read_csv(table_csv_path)
        self._build_index()
        self.env_map_image_path = env_map_image_path

    @cached_property
    def lights_dmx_and_index(self):
        return self.light_info_table[["DMX ID", "Light ID"]].values.astype(np.int32)

    @cached_property
    def lights_dmx(self):
        return self.lights_dmx_and_index[:, 0]

    @cached_property
    def lights_envMapYUp(self):
        return self.light_info_table[["Dome X", "Dome Y", "Dome Z"]].values

    @cached_property
    def lights_domeYUp(self):
        return DomeCoordinates.t_envMapYUp_to_domeYUp(self.lights_envMapYUp)

    @cached_property
    def lights_sampleW0(self):
        return DomeCoordinates.t_domeYUp_to_sampleW0(self.lights_domeYUp)

    @cached_property
    def lights_domeThetaPhi(self):
        return self.light_info_table[["Dome Theta", "Dome Phi"]].values

    @cached_property
    def lights_envMapUV(self):
        return self.light_info_table[["Pixel U", "Pixel V"]].values

    @cached_property
    def env_map_image(self):
        return imageio.imread(self.env_map_image_path)

    def area_correction(self, removed: Collection[tuple[int, int]] = frozenset()) -> dict[tuple[int, int], float]:
        removed = set(removed)

        mask = np.array(
            [
                (dmx_id, light_id) not in removed
                for dmx_id, light_id in self.lights_dmx_and_index
            ]
        )

        # Calculate the spherical Voronoi diagram:
        sv = SphericalVoronoi(self.lights_envMapYUp[mask], radius=1, center=np.array([0, 0, 0]))
        # Calculate area of each Voronoi region
        areas = sv.calculate_areas()
        # Correction is averaged around 1
        areas_normalized = areas / np.mean(areas)

        return {
            tuple(map(int, dmx_index)): float(area)
            for dmx_index, area in zip(self.lights_dmx_and_index[mask], areas_normalized)
        }

    # def area_density_correction(self) -> np.ndarray:
    #     # Calculate the spherical Voronoi diagram:
    #     sv = SphericalVoronoi(self.lights_envMapYUp, radius=1, center=np.array([0, 0, 0]))
    #     # Calculate area of each Voronoi region
    #     areas = sv.calculate_areas()
    #     # Correction is averaged around 1
    #     return areas / np.mean(areas)


    def get_index(self, dmx_id: int, light_id: int) -> int:
        return self._idx_by_dmx_and_lightID[dmx_id][light_id]

    def _determine_envMap_to_domeThetaPhi_rotation(self):
        phi_envMap = self.light_info_table["Pixel U"] * (2 * np.pi)
        phi_dome = self.light_info_table["Dome Phi"]
        print(np.unique(np.round(phi_dome - phi_envMap, decimals=2)))

    def light_blending_weights(self, distances: np.ndarray):
        """
        Args:
                distances: N x num_lights
        """
        return 1.0 - distances / np.sum(distances, axis=1)[:, None]

    def _build_index(self):
        # Build a KDTree index of the light positions
        # Casting to float64 probably not needed but I don't know what type the CSV loads
        self._kdtree_domeYUp = KDTree(self.lights_domeYUp.astype(np.float64))

        lights_by_dmx = {}
        for i, (dmx, light_id) in enumerate(self.lights_dmx_and_index):
            lights_by_dmx.setdefault(dmx, {})[light_id] = i
        self._idx_by_dmx_and_lightID = lights_by_dmx

    def query_lights(self, dir_domeYUp: np.ndarray, num_lights: int = 3):
        """
        Args:
                dir_envMapYUp: Nx3
        """
        if not isinstance(dir_domeYUp, np.ndarray):
            # Convert list to array
            dir_domeYUp = np.array(dir_domeYUp)

        if len(dir_domeYUp.shape) == 1:
            # Convert 1D vector into [1x3]
            dir_domeYUp = dir_domeYUp[None, :]

        # ensure Nx3 shape
        assert dir_domeYUp.shape[1] == 3

        # Normalize direction
        dir_normalized = dir_domeYUp / np.linalg.norm(dir_domeYUp, axis=1)[:, None]

        # Query the tree in parallel
        result_distances, result_ind = self._kdtree_domeYUp.query(
            dir_normalized,  # query points
            k=num_lights,  # number of nearest neighbours
            workers=(
                1 if len(dir_normalized) == 1 else -1
            ),  # use multiple cores if multiple queries
        )

        return LightQueryResult(
            query_directions=dir_normalized,
            light_indices=result_ind,
            distances=result_distances,
            weights=self.light_blending_weights(result_distances),
        )

    def describe_blend_query(self, result: LightQueryResult):
        """ """
        for i in range(result.num_query_points):

            query_dir = result.query_directions[i]
            light_indices = result.light_indices[i]
            lights = self.lights_dmx_and_index[light_indices]
            weights = result.weights[i]

            print(f"Query {i}: {query_dir}")
            for (dmx, light_id), weight in zip(lights, weights):
                print(f"\tDMX_{dmx:02d}/{light_id:03d}, weight: {weight}")

    def plot_blend_query(self, result: LightQueryResult):
        """
        Args:
        """

        query_dirs_envMapUV = DomeCoordinates.t_domeYUp_to_envMapUV(
            result.query_directions
        )

        canvas = self.env_map_image.copy()
        ih, iw = canvas.shape[:2]

        def uv_to_image(uv):
            return (int(uv[0] * iw), int((uv[1]) * ih))

        for i in range(result.num_query_points):
            query_dir_envMapUV = query_dirs_envMapUV[i]
            light_indices = result.light_indices[i]
            lights_envMapUV = self.lights_envMapUV[light_indices]
            weights = result.weights[i]

            for light_uv, weight in zip(lights_envMapUV, weights):
                # Draw a line between the query direction and the light
                cv.line(
                    canvas,
                    uv_to_image(query_dir_envMapUV),
                    uv_to_image(light_uv),
                    (255, 255, 255),
                    thickness=1,
                )

                # Draw light, circle area proportional to weight
                cv.circle(
                    canvas,
                    uv_to_image(light_uv),
                    int(20 * np.sqrt(weight)),
                    (255, 0, 0),
                    thickness=2,
                )

            # Draw query point
            cv.circle(
                canvas,
                uv_to_image(query_dir_envMapUV),
                2,
                (0, 255, 0),
                thickness=2,
            )

        return canvas

    def gradient_generate(self, gradient_dir_envMapYUp) -> GradientResult:
        # normalize gradient dir
        gradient_dir = np.array(gradient_dir_envMapYUp, dtype=np.float64)
        gradient_dir /= np.linalg.norm(gradient_dir)
        assert gradient_dir.shape == (3,)

        # dot product
        light_coords = self.lights_envMapYUp
        dir_dot_light = light_coords @ gradient_dir[:, None]

        # intensity from 0 to 1
        intensity = 0.5 * dir_dot_light + 0.5

        # dict by dmx
        return GradientResult(
            gradient_dir_envMapYUp=gradient_dir,
            intensities=intensity,
            intensities_by_dmx={
                tuple(map(int, dmx_index)): float(intensity)
                for dmx_index, intensity in zip(self.lights_dmx_and_index, intensity)
            },
        )

    def gradient_serialize(self, gradient: GradientResult, path: Path):
        path = Path(path)

        # Build CSV table with columns: DMX ID, Light ID, Intensity
        dmx_ids = []
        light_ids = []
        intensities = []

        for (dmx_id, light_id), intensity in gradient.intensities_by_dmx.items():
            dmx_ids.append(dmx_id)
            light_ids.append(light_id)
            intensities.append(intensity)

        table = pandas.DataFrame(
            {"DMX ID": dmx_ids, "Light ID": light_ids, "Intensity": intensities}
        )
        table.to_csv(path, index=False)
        return table

    def gradient_plot(self, gradient_result: GradientResult, save=None):
        fig, ax = pyplot.subplots(1, 1, figsize=(14, 6))
        fig.suptitle(f"Gradient for {gradient_result.gradient_dir_envMapYUp}")

        # The Y values needed to be inverted to match, the UVs must be calculated from top to bottom?
        ax.imshow(self.env_map_image, extent=[0, 1, 0, 1], aspect="auto")

        # scatter, color by intensity, red-blue colormap, big dots
        uvs = self.lights_envMapUV
        sc = ax.scatter(
            uvs[:, 0],
            1.0 - uvs[:, 1],
            c=gradient_result.intensities,
            s=50,
            cmap="magma",
        )

        # display colorbar
        fig.colorbar(sc, ax=ax)

        fig.tight_layout()

        pyplot.show()
        if save:
            fig.savefig(save)
        pyplot.close(fig)


    def voronoi_area_correction_demo(self):
        from matplotlib import pyplot
        from scipy.spatial import geometric_slerp
        from mpl_toolkits.mplot3d import proj3d

        points = self.lights_envMapYUp

        # Calculate the spherical Voronoi diagram:
        sv = SphericalVoronoi(points, radius=1, center=np.array([0, 0, 0]))

        # Plot
        # sort vertices (optional, helpful for plotting)
        sv.sort_vertices_of_regions()

        t_vals = np.linspace(0, 1, 2000)

        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')

        # plot the unit sphere for reference (optional)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='y', alpha=0.1)

        # plot generator points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
        # plot Voronoi vertices
        ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], c='g')

        # indicate Voronoi regions (as Euclidean polygons)
        for region in sv.regions:
            n = len(region)
            for i in range(n):
                start = sv.vertices[region][i]
                end = sv.vertices[region][(i + 1) % n]
                result = geometric_slerp(start, end, t_vals)
                ax.plot(result[..., 0], result[..., 1], result[..., 2], c='k')

        ax.azim = 10
        ax.elev = 40

        _ = ax.set_xticks([])
        _ = ax.set_yticks([])
        _ = ax.set_zticks([])
        fig.set_size_inches(4, 4)
        fig.tight_layout()
        pyplot.show()


