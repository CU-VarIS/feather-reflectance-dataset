
from dataclasses import dataclass
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from xml.dom import minidom

import numpy as np
from tqdm import tqdm

from ..Utilities.ImageIO import readImage, writeImage
from .capture import CaptureFrame, FrameVisitor, FunctionFrameVisitor, VarisCapture
from .retro import RetroreflectionCapture
from .file_index import FileIndex

@dataclass
class BRDFMeasurementBundle:
    name: str
    wis: np.ndarray
    wos: np.ndarray
    thetas: np.ndarray
    normals: np.ndarray

    measurements_rgb: np.ndarray

    def __len__(self) -> int:
        b, h, w, c = self.measurements_rgb.shape
        return b

    @property
    def image_wh(self) -> tuple[int, int]:
        b, h, w, c = self.measurements_rgb.shape
        return w, h

class OLATRegionView:
    def __init__(self, region_name: str, crop_tl_xy: tuple[int, int], crop_wh: tuple[int, int], capture: VarisCapture, dir_storage: Path | None = None):
        self.region_name = region_name
        self.crop_tl_xy = crop_tl_xy
        self.crop_wh = crop_wh
        self.crop_slice = (slice(crop_tl_xy[1], crop_tl_xy[1] + crop_wh[1]), slice(crop_tl_xy[0], crop_tl_xy[0] + crop_wh[0]))

        self.dir_storage = dir_storage or capture.dir_src / "cache"


        self.capture = capture
        self.capture_name = capture.name
        # For converting between linear frame index and wiid / dmx / light
        self.frame_wi_index = capture.frame_wi_index
        self.frame_dmx_light_ids = capture.frame_dmx_light_ids

        self._cache_block_by_wiid = {}
        self._cache_allow_missing = False


    def _get_cache_block(self, wiid: tuple[int, int]) -> np.ndarray:
        """Cache blocks store the crops alongside each other in a 2D grid where 
            row = DMX
            col = Light ID / 5

            DMX is in range of 0-9
            LightID/5 is in range of 1-40
        """
        NUM_ROWS = 10  
        NUM_COLS = 40
        ti, pi = wiid
        wiid = (int(ti), int(pi))

        if wiid not in self._cache_block_by_wiid:
            w, h = self.crop_wh
            block_shape = (NUM_ROWS * h, NUM_COLS * w, 3)

            cache_path = self._cache_file_path(wiid)

            self.capture.ensure_local_file(cache_path.relative_to(self.capture.dir_src), required=False)

            if cache_path.is_file():
                block_cached =  readImage(cache_path)
                assert block_cached.shape == block_shape, f"Block shape mismatch cached={block_cached.shape} expected={block_shape}"
                self._cache_block_by_wiid[wiid] = block_cached
            elif self._cache_allow_missing:
                self._cache_block_by_wiid[wiid] = np.zeros(block_shape, dtype=np.float32)
            else:
                raise FileNotFoundError(f"No cache at {self._cache_file_path(wiid)}, run `extract_region_cache_olat` first")
                      
        return self._cache_block_by_wiid[wiid]
    
    def _block_slice_view(self, wiid: tuple[int, int], dmx_id: int, light_id: int) -> np.ndarray:
        cache_block = self._get_cache_block(wiid)
        w, h = self.crop_wh
        row = dmx_id * h
        col = (light_id // 5 - 1) * w
        return cache_block[row:row+h, col:col+w]

    def __getitem__(self, wiid_dmx_light: tuple[int, int, int, int]) -> np.ndarray:
        # wiid = self.frame_wi_index[idx]
        # dmx_id, light_id = self.frame_dmx_light_ids[idx]
        theta_id, phi_id, dmx_id, light_id = wiid_dmx_light
        return np.ascontiguousarray(self._block_slice_view((theta_id, phi_id), dmx_id, light_id))

    def __len__(self):
        return len(self.frame_wi_index)
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
    
    def _extract_crop(self, frame_full_image: np.ndarray) -> np.ndarray:
        return frame_full_image[self.crop_slice]

    def add_sample(self, frame: CaptureFrame, frame_full_image: np.ndarray) -> np.ndarray:
        crop = self._extract_crop(frame_full_image)
        self._block_slice_view(frame.wiid, frame.dmx_id, frame.light_id)[:] = crop
        return crop

    def _cache_file_path(self, wiid: tuple[int, int]) -> Path:
        theta_i, phi_i = wiid
        return self.dir_storage / f"RegionCache_{self.capture_name}_{self.region_name}_thetaI{theta_i:03d}_phiI{phi_i:03d}.exr"

    def save(self):
        self._cache_file_path((0, 0)).parent.mkdir(parents=True, exist_ok=True)
        for wiid, cache_block in tqdm(self._cache_block_by_wiid.items(), desc=f"Writing cache for {self.region_name}"):
            writeImage(cache_block, self._cache_file_path(wiid))

    def olat_stack(self, wiid: tuple[int, int]) -> np.ndarray:
        sp = self.capture.stage_poses[wiid]
        return np.stack([self[fr_idx] for fr_idx in sp.frame_indices])

    def olat_bundle(self, wiid: tuple[int, int]) -> BRDFMeasurementBundle:
        sp = self.capture.stage_poses[wiid]
        fr_idx = sp.frame_indices

        wis = self.capture.frame_wi[fr_idx]
        wos = self.capture.frame_wo[fr_idx]
        thetas = np.array([self.capture.frames[fi].theta_i for fi in fr_idx])

        return BRDFMeasurementBundle(
            name = f"{self.capture.name} wi{wiid[0]}-{wiid[1]}",
            wis = wis,
            wos = wos,
            thetas = thetas,
            normals = np.full((len(wis), 3), [0, 0, 1], dtype=np.float32),
            measurements_rgb = self.olat_stack(wiid),
        )    

    def retro_stack(self, retro: RetroreflectionCapture):
        # Load from cache if exists
        cache_path = retro.dir_src / f"RegionCache_{self.capture_name}_{self.region_name}_RetroStack.npy"
        w, h = self.crop_wh

        if cache_path.is_file():
            # return readImage(cache_path).astype(np.float32).reshape(-1, retro.num_theta_i, retro.num_phi_i, h, w, 3)
            return np.load(cache_path)

        retro_stack = np.zeros((retro.num_theta_i, retro.num_phi_i, h, w, 3), dtype=np.float32)

        for fr in tqdm(retro.frames, desc=f"Extracting retro stack {self.region_name}"):
            theta_i, phi_i = fr.wiid
            img = retro.read_measurement_image(fr)
            retro_stack[theta_i, phi_i] = self._extract_crop(img)

        # writeImage(retro_stack, cache_path)
        np.save(cache_path, retro_stack)

        return retro_stack

    @classmethod
    def regions_from_svg(cls, capture: VarisCapture, path_svg: Path, dir_storage = None) -> list["RegionView"]:
        dir_storage = dir_storage or path_svg.parent
        regions = []

        # Read the SVG and extract rectangles
        doc = minidom.parse(str(path_svg))
        rects = doc.getElementsByTagName("rect")
        for rect in rects:
            region_name = rect.getAttribute("inkscape:label")
            x, y, w, h = xml_node_get_numerical(rect, ("x", "y", "width", "height"))
            w += w % 2
            h += h % 2
            print(f"Region {region_name} tl_xy={x, y}, wh={w, h}")

            regions.append(cls(
                region_name = region_name, 
                crop_tl_xy=(x, y), 
                crop_wh=(w, h), 
                capture=capture,
                dir_storage=dir_storage,
            ))


        return regions

    class _Visitor(FrameVisitor):
        def __init__(self, region: "OLATRegionView", write_small_files: bool = False):
            self.region = region
            self.write_small_files = write_small_files

        def start(self, capture: VarisCapture):
            del capture
            self.region._cache_allow_missing = True

        def visit_frame(self, frame: CaptureFrame, img: np.ndarray):
            crop = self.region.add_sample(frame, img)

            if self.write_small_files:
                dir_small = self.region.dir_storage / "processing" / self.region.region_name
                dir_small.mkdir(parents=True, exist_ok=True)
                writeImage(crop, dir_small / frame.image_path.name)

        def finalize(self):
            self.region._cache_allow_missing = False
            self.region.save()


    def visitor_extract_region(self, write_small_files=False) -> FrameVisitor:
        return OLATRegionView._Visitor(self, write_small_files=write_small_files)

    @classmethod
    def extract_region_cache_olat(cls, capture: VarisCapture, write_small_files=False, num_workers=8):
        # TODO reread if original cache not present
        capture.visit(
            visitors=[
                region.visitor_extract_region(write_small_files=write_small_files) 
                for region in capture.named_region_views.values()
            ],
            num_threads=num_workers,
        )


    # @classmethod
    # def extract_region_cache_retro(cls, capture: VarisCapture, write_small_files=False):
    #     regions = list(capture.named_region_views.values())

    #     for frame in tqdm(capture.frames, desc="Reading OLAT frames"):
    #         wiid = frame.wiid
    #         sp = capture.stage_poses[wiid]
    #         img = sp._manual_homography_apply(readImage(frame.image_path))
    #         for region in regions:
    #             crop = region.add_sample(frame, img)

    #             if write_small_files:
    #                 dir_small = region.dir_storage / "processing" / region.region_name
    #                 dir_small.mkdir(parents=True, exist_ok=True)
    #                 writeImage(crop, dir_small / frame.image_path.name)

    #     for region in regions:
    #         region.save()

    def file_index(self) -> FileIndex:
        """Return all files in the region view."""
        file_index = FileIndex(name=self.capture.name, dir_src=self.capture.dir_src)

        # Cache files
        for wiid in self.capture.stage_poses:
            cache_path = self._cache_file_path(wiid)
            cache_path_relative = cache_path.relative_to(self.capture.dir_src)
            file_index.add(cache_path, name=str(cache_path_relative), is_source=False)

        # Materials
        mat_path = self.capture.dir_src / "006_materials_manual" / f"{self.region_name}.bsdf"
        file_index.add(mat_path, name=str(mat_path.relative_to(self.capture.dir_src)), is_source=False)

        return file_index





class OLATPixelMaskView(OLATRegionView):
    """Cache an average of pixels selected by a pixel mask.
        The mask is assumed to be small < 100 pixels.
    """
    def __init__(self, region_name: str, mask: np.ndarray, dir_storage: Path, capture: VarisCapture | None = None):
        super().__init__(region_name=region_name, crop_tl_xy=(0, 0), crop_wh=(1, 1), dir_storage=dir_storage, capture=capture)
        self.mask = mask
        self.crop_slice = np.where(mask)

    def _extract_crop(self, frame_full_image: np.ndarray) -> np.ndarray:
        pixels = frame_full_image[self.crop_slice]
        # average the pixels, add fake 1x1 height and width
        return np.mean(pixels, axis=0)[None, None]

    


def xml_node_get_numerical(node, names: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(round(float(node.getAttribute(name))) for name in names)


def main(mask_file):
    """Given a JPG mask Masks_*.jpg convert to 000_submask_*.png"""
    # Read the mask file
    mask = readImage(mask_file)

    print(f"Mask {mask_file}:", np.unique(mask, return_counts=True))

    # Ensure the mask is binary using the red channel
    mask[mask[:, :, 0] < 127] = 0
    mask[mask[:, :, 0] >= 127] = 255


    # Save the binary mask as PNG
    output_file = mask_file.replace("Masks_", "000_submask_").replace(".jpg", ".png")
    print(f"Mask {output_file}:", np.unique(mask, return_counts=True))
    writeImage(mask, output_file)



if __name__ == "__main__":
    from fire import Fire
    Fire(main)