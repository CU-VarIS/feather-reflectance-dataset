
# Dataset maintenance

## High resolution rectified images

The original photos taken by the camera are in 6k resolution.
For older captures the rectified image was in that size too.
However the area going into the rectified is only a small part of the overall photo.
Later on we have switched to rectifying to `1152 x 1408`.

Captures where the rectified resolutions are too big:
* FeatherNorthernFlicker
* FeatherNorthernFlickerVentral
* FeatherBlueJay
* FeatherRedTailedHawk


To unify the resolution we downsample to `1152 x 1408` with the command:
```
python -m varis_feather resolution_unify
```
