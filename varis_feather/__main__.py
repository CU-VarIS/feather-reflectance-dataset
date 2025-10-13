
from fire import Fire

from .Scripts.post_capture import post_capture
from .Scripts.upload import upload
from .Scripts.rename import main as rename_main

if __name__ == "__main__":
    Fire({
        "post_capture": post_capture,
        "upload": upload,
        "rename": rename_main,
    })
