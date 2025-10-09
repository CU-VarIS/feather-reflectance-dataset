
from fire import Fire

from .Scripts.post_capture import post_capture

if __name__ == "__main__":
    Fire({
        "post_capture": post_capture,
    })
