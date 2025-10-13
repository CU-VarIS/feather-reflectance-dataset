from pathlib import Path
import os, json

DIR_PROJECT = Path(__file__).absolute().parents[1]

DIR_DATASET = DIR_PROJECT / "dataset"

SCENES = [
    "Spectralon",
    "FeatherHyacinthMacaw",
    "ButterflySwallowtail",
    ## "FeatherBlackChinnedHummingbird", # incomplete!
    "FeatherBlackVulture",
    "FeatherBlueJay",
    "FeatherGreatBlueHeron",
    "FeatherGreatHornedOwl",
    "FeatherNorthernFlicker",
    "FeatherNorthernFlickerVentral",
    "FeatherOstrich",
    "FeatherRedCrownedAmazon",
    "FeatherRedTailedHawk",
    "FeatherRockDove",
]


STORAGE_BUCKET = "cu-varis-feather-v1-dev"
STORAGE_URL = "https://s3.us-east-005.backblazeb2.com"

STORAGE_ID_AND_KEY = (None, None)

# Try getting key from file
if (key_path := DIR_PROJECT / "storage_key.json").is_file():
    content = json.loads(key_path.read_text())
    STORAGE_ID_AND_KEY = (
        content["key_id"],
        content["key"],
    )
# Try getting from env
else:
    STORAGE_ID_AND_KEY = (
        os.getenv("STORAGE_KEY_ID", None),
        os.getenv("STORAGE_KEY", None),
    )

