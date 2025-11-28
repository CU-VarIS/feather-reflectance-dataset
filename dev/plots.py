
# %% [markdown]
# # CU-Varis Feather - plots

# %%

import sys
sys.path.append("..")

from varis_feather.Utilities.show_image import show, image_montage_same_shape
from varis_feather.Utilities.ImageIO import readImage, writeImage, RGL_tonemap_uint8, RGL_tonemap
from varis_feather.Space.capture import CaptureFrame
from varis_feather.Space.olat import OLATCapture

from varis_feather import load_standard_capture


# %%

retro, olat = load_standard_capture("FeatherHyacinthMacaw")

# %%

olat.plot_crops_in_theta_phi("barb64", (3, 0), out_path_override="assets/")
olat.plot_crops_in_theta_phi("barb64", (5, 0), out_path_override="assets/")

# %%

olat.stage_poses[(0, 0)].normals_image_path
# %%

index = olat.file_index()
index.entries
# %%
