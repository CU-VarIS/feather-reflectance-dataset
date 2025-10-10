
import numpy as np
from .light_index import DomeLightIndex
from .capture import CaptureFrame, VarisCapture, FrameVisitor

class GradientNormalVisitor(FrameVisitor):
    def __init__(self, wiid: tuple[int, int]):
        self.wiid = wiid
        self._gradient_accumulators: dict[str, np.ndarray] = {}
        self.normals: np.ndarray | None = None

    def start(self, capture: VarisCapture):
        pass

    def _accumulate_grad(self, dim: str, contribution: np.ndarray):
        if (acc := self._gradient_accumulators.get(dim)) is None:
            acc = self._gradient_accumulators[dim] = np.zeros_like(contribution)

        acc += contribution

    def visit_frame(self, frame: CaptureFrame, img: np.ndarray):
        if frame.wiid != self.wiid:
            return
        
        di = DomeLightIndex.default_instance()
        
        # Correct by inverse density
        corr = di.area_density_correction[di.get_index(frame.dmx_id, frame.light_id)]

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



