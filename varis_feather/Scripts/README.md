
# Dataset maintenance

Edit `Paths.py` to activate or deactivate captures.

## Applying corrections and generating caches

Convert from older naming scheme to the one in the storage:
```
python -m varis_feather rename DIR_OLD
```


Apply calibration of light brightness, board normals. Then regenerate region caches.

```
python -m varis_feather post_capture
```

## Upload to storage

* Enter the storage key in `cu_varis_settings.json` or set env `CU_VARIS_STORAGE_KEY_ID` and `CU_VARIS_STORAGE_KEY`.
```json
{
 "key": "...",
 "key_id": "...",
}
```
* Differential upload with `python -m varis_feather upload`


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
