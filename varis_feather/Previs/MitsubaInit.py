
from functools import lru_cache
from typing import Literal, Optional
import mitsuba as mi

@lru_cache(maxsize=1)
def mitsuba_cuda_available() -> bool:
    try:
        # Try to set the CUDA mode
        mi.set_variant("cuda_ad_rgb")

        # There may still be failure to load OptiX when creating a scene
        mi.load_dict({
            "type": "roughconductor", 
            "distribution": "ggx", 
            "alpha_u": {"type": "uniform", "value": 0.5}, 
            "alpha_v": {"type": "uniform", "value": 0.5}, 
            "specular_reflectance": {"type": "rgb", "value": [0, 0.5, 0.75]}
        })

        return True

    except Exception as e:
        print("Mitsuba: CUDA not available", e)
        return False

def _mitsuba_default_init():
    """Set the Mitsuba architecture to a default.
        We prefer CUDA if available, otherwise LLVM.    
    """

    # Mitsuba already initialized
    if mi.variant():
        return

    if mitsuba_cuda_available():
        mi.set_variant("cuda_ad_rgb")
    else:
        mi.set_variant("llvm_ad_rgb")

    print(f"Mitsuba = {mi.variant()}")

def mitsuba_arch_colormode() -> tuple[str, str]:
    """Get the Mitsuba architecture and color mode
    
    llvm_ad_rgb -> llvm, rgb
    cuda_ad_rgb -> cuda, rgb
    scalar_rgb -> Error, since scalar mode is slow an we do not use it
    """

    variant = mi.variant()
    try:
        arch, ad, colormode = variant.split('_')
        return arch, colormode
    except Exception as e:
        print("Mituba variant not following ARCH_ad_COLORMODE format:", e)
        raise e

def mitsuba_set_mode(arch: Optional[Literal["llvm", "cuda"]] = None, colormode: Optional[Literal["rgb", "spectral"]]=None ):
    """Set the architecture or spectral mode.
    
    * Set architecture to `arch` = 'llvm' or 'cuda', or keep unchanged if None.
    * Set color mode to `colormode` = 'rgb' or 'spectral', or keep unchanged if None.
    """
    # Get current variant
    arch_prev, colormode_prev = mitsuba_arch_colormode()
    # If None, keep previous:
    arch = arch or arch_prev
    colormode = colormode or colormode_prev

    if (arch_prev != arch) or (colormode_prev != colormode):
        mi.set_variant(f"{arch}_ad_{colormode}")
    
_mitsuba_default_init()

# @lru_cache(maxsize=32)
# def _mitsuba_type_by_arch(t: str, arch: Literal["llvm", "cuda"]):
#     drjit_by_arch = getattr(dr, arch)
#     return getattr(drjit_by_arch, t)

# def mitsuba_Array2f(data: np.ndarray):
#     arch, _ = mitsuba_arch_colormode()
#     Array2f = _mitsuba_type_by_arch("Array2f", arch=arch)
#     return Array2f(data.T)

# def mitsuba_Array3f(data: np.ndarray):
#     arch, _ = mitsuba_arch_colormode()
#     Array3f = _mitsuba_type_by_arch("Array3f", arch=arch)
#     return Array3f(data.T)
