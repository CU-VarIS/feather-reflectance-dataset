

from typing import Any

from .Paths import DIR_DATASET
from .Space.olat import OLATCapture, ThetaDistribution
from .Space.retro import RetroreflectionCapture


def load_standard_capture(cap_name, mode: str = "rectified", retro_variant="128x1", olat_variant="iso", retro_kwargs: None | dict[str, Any] = None, olat_kwargs: None | dict[str, Any] = None) -> tuple[RetroreflectionCapture, OLATCapture]:
 
    if mode == "legacy":
        rec_mode = "rectified"
        dir_retro = DIR_DATASET / "Retroreflection" / "128x1" / cap_name / rec_mode
        dir_olat = DIR_DATASET / "FullScattering" / cap_name / rec_mode
    else:
        dir_retro = DIR_DATASET / "captures" / cap_name / f"retro_{retro_variant}_{mode}"
        dir_olat = DIR_DATASET / "captures" / cap_name / f"olat_{olat_variant}_{mode}"


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
