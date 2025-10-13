

from typing import Any

from varis_feather.Space.file_index import FileIndex

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


    if (olat_kwargs or {}).get("use_index", True):
        FileIndex.download_starter(cap_name, dir_olat)

    olat = OLATCapture(
        dir_src = dir_olat,
        num_theta_i=8,
        num_phi_i=1,    
        # theta_distribution=ThetaDistribution(mode=ThetaDistribution.MODE_UNIFORM, offset_rad=np.deg2rad(16.35)),
        # frame_below_horizon="remove",
        theta_distribution=ThetaDistribution(mode=ThetaDistribution.MODE_UNIFORM),
        **(olat_kwargs or {}),
    )
    print(olat.report())


    try:
        if (retro_kwargs or {}).get("use_index", True):
            FileIndex.download_starter(cap_name, dir_retro)

        retro = RetroreflectionCapture(
            dir_src = dir_retro,
            num_theta_i=128,
            num_phi_i=1,
            # theta_distribution=ThetaDistribution(mode=ThetaDistribution.MODE_U_TO_THETA, offset_rad=np.deg2rad(10.43)),
            frame_below_horizon="ignore",
            **(retro_kwargs or {}),
        )
        print(retro.report())


        if olat.named_region_views and not retro.named_region_views:
            # We usually keep the named regions with the OLAT
            retro.load_named_regions(olat.dir_src)
    except Exception as e:
        print(f"Could not load retro capture from {dir_retro}: {e}")
        retro = None

    return retro, olat
