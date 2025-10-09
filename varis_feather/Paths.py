from pathlib import Path

DIR_PROJECT = Path(__file__).absolute().parents[1]

DIR_DATASET = DIR_PROJECT / "dataset"

SCENES = [
    # "Spectralon",
    # "FeatherHyacinthMacaw",
    # "ButterflySwallowtail",
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