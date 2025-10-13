
from fire import Fire

from .Scripts.post_capture import post_capture
from .Scripts.rename import main as rename_main
from .Scripts.resolution_unify import resolution_unify
from .Scripts.upload import upload

if __name__ == "__main__":
    Fire({
        "post_capture": post_capture,
        "upload": upload,
        "rename": rename_main,
        "resolution_unify": resolution_unify,
    })
