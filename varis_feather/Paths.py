from pathlib import Path
import os, json

DIR_PROJECT = Path(__file__).absolute().parents[1]

if (settings_path := DIR_PROJECT / "cu_varis_settings.json").is_file():
    settings_content = json.loads(settings_path.read_text())
else:
    settings_content = {}

DIR_DATASET = Path(os.getenv("CU_VARIS_FEATHER_DIR", settings_content.get("feather_dir", DIR_PROJECT / "dataset"))).resolve()

SCENES = [
    "Spectralon",
    "FeatherHyacinthMacaw",
    "FeatherRedCrownedAmazon",
    "FeatherOstrich",
    "FeatherRockDove",
    "FeatherBlackVulture",
    "ButterflySwallowtail",
    # "FeatherBlackChinnedHummingbird", # incomplete!
    # "FeatherBlueJay", # weird shape
    # "FeatherGreatBlueHeron",
    # "FeatherGreatHornedOwl",
    # "FeatherNorthernFlicker",
    # "FeatherNorthernFlickerVentral", # weird shape
    # "FeatherRedTailedHawk",
]



STORAGE_BUCKET = "cu-varis-feather-v1-dev"
STORAGE_URL = "https://s3.us-east-005.backblazeb2.com"

# Try getting key from env or file
STORAGE_ID_AND_KEY = (
    os.getenv("CU_VARIS_STORAGE_KEY_ID", settings_content.get("key_id", None)),
    os.getenv("CU_VARIS_STORAGE_KEY", settings_content.get("key", None)),
)
