
import numpy as np
from .light_index import DomeLightIndex
from .capture import CaptureFrame, VarisCapture, FrameVisitor

class GradientNormalVisitor(FrameVisitor):
    def __init__(self, wiid: tuple[int, int]):
        self.wiid = wiid
        self._gradient_accumulators: dict[str, np.ndarray] = {}
        self.normals: np.ndarray | None = None

    def start(self, capture: VarisCapture):

        sp = capture.stage_poses[self.wiid]

        removed_lights = set()

        for fr_idx in sp.frame_indices:
            frame = capture.frames[fr_idx]
            if not frame.is_valid:
                removed_lights.add((frame.dmx_id, frame.light_id))

        di = DomeLightIndex.default_instance()
        self._area_correction = di.area_correction(removed_lights)



    def _accumulate_grad(self, dim: str, contribution: np.ndarray):
        if (acc := self._gradient_accumulators.get(dim)) is None:
            acc = self._gradient_accumulators[dim] = np.zeros_like(contribution)

        acc += contribution

    def visit_frame(self, frame: CaptureFrame, img: np.ndarray):
        if frame.wiid != self.wiid or not frame.is_valid:
            return
                
        # Correct by inverse density
        corr = self._area_correction[(frame.dmx_id, frame.light_id)]

        for dim_i, dim_name in enumerate(("x", "y", "z")):
            weight = (0.5*frame.sample_wo[dim_i] + 0.5) * corr
            self._accumulate_grad(dim_name, img * weight)
        
        self._accumulate_grad("all", img * corr)

        

    def finalize(self):
        g_all = self._gradient_accumulators["all"]
        self.normals = np.stack([
            np.mean(self._gradient_accumulators[dim] / g_all, axis=2) * 2 - 1
            for dim in ("x", "y", "z")
        ], axis=-1)
        self.normals /= np.linalg.norm(self.normals, axis=-1, keepdims=True)
        self.normals_img = ((self.normals + 1) * 128).astype(np.uint8)
        return self.normals



