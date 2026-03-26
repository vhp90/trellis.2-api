from __future__ import annotations

import os


HF_ENV_KEYS = [
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_API_KEY",
    "HUGGING_FACE_API_KEY",
]


def configure_huggingface_auth() -> str | None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    token = None
    for key in HF_ENV_KEYS:
        value = os.getenv(key)
        if value:
            token = value.strip()
            break

    if not token:
        return None

    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False, skip_if_logged_in=True)
    except Exception:
        # Downloads still work via env vars in most cases, so don't fail startup here.
        pass

    return token
