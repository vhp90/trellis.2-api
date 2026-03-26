from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT_DIR / "outputs"))
CACHE_DIR = Path(os.getenv("HF_HOME", ROOT_DIR / ".cache" / "huggingface"))


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings:
    app_name = "TRELLIS.2 API"
    model_id = os.getenv("TRELLIS_MODEL_ID", "microsoft/TRELLIS.2-4B")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    low_vram = os.getenv("TRELLIS_LOW_VRAM", "0") != "0"
    preload_model = os.getenv("PRELOAD_MODEL", "1") != "0"
    device = os.getenv("TRELLIS_DEVICE", "cuda")
    default_pipeline_type = os.getenv("TRELLIS_DEFAULT_PIPELINE", "1024_cascade")
    public_base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    allow_origins = _split_csv(
        os.getenv(
            "CORS_ALLOW_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
        )
    )


settings = Settings()
