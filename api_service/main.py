from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

import time
import uuid
from io import BytesIO
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool

from .config import OUTPUT_DIR, settings
from .pipeline_manager import pipeline_manager

torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _bool_from_form(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "on"}


def _build_sampler_params(
    *,
    steps: int | None,
    guidance_strength: float | None,
    rescale_t: float | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if steps is not None:
        params["steps"] = steps
    if guidance_strength is not None:
        params["guidance_strength"] = guidance_strength
    if rescale_t is not None:
        params["rescale_t"] = rescale_t
    return params


def _public_output_url(request: Request, filename: str) -> str:
    if settings.public_base_url:
        return f"{settings.public_base_url}/outputs/{filename}"
    return str(request.url_for("outputs", path=filename))


def _decode_uploaded_image(raw_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(raw_bytes)).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc


@app.on_event("startup")
def _startup() -> None:
    if settings.preload_model:
        pipeline_manager.load()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": pipeline_manager.loaded,
        "cuda_available": torch.cuda.is_available(),
        "device": settings.device,
        "model_id": settings.model_id,
        "low_vram": settings.low_vram,
    }


@app.get("/config")
def config() -> dict[str, Any]:
    return {
        "model_id": settings.model_id,
        "default_pipeline_type": settings.default_pipeline_type,
        "allow_origins": settings.allow_origins,
        "outputs_path": str(OUTPUT_DIR),
        "low_vram": settings.low_vram,
    }


@app.post("/rembg")
async def remove_background(
    request: Request,
    image: UploadFile = File(...),
) -> dict[str, Any]:
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        raw_bytes = await image.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    pil_image = _decode_uploaded_image(raw_bytes)
    request_id = uuid.uuid4().hex
    output_name = f"{request_id}-rembg.png"
    output_path = OUTPUT_DIR / output_name

    try:
        processed = await run_in_threadpool(pipeline_manager.remove_background, pil_image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {exc}") from exc

    processed.save(output_path, format="PNG")
    return {
        "id": request_id,
        "filename": output_name,
        "content_type": "image/png",
        "download_url": _public_output_url(request, output_name),
    }


@app.post("/generate")
async def generate(
    request: Request,
    image: UploadFile = File(...),
    seed: int = Form(42),
    num_samples: int = Form(1),
    pipeline_type: str = Form(settings.default_pipeline_type),
    preprocess_image: str = Form("true"),
    max_num_tokens: int = Form(49152),
    simplify_target: int = Form(1000000),
    texture_size: int = Form(2048),
    remesh: str = Form("true"),
    remesh_band: float = Form(1.0),
    remesh_project: float = Form(0.0),
    ss_steps: str | None = Form(None),
    ss_guidance_strength: str | None = Form(None),
    ss_rescale_t: str | None = Form(None),
    shape_steps: str | None = Form(None),
    shape_guidance_strength: str | None = Form(None),
    shape_rescale_t: str | None = Form(None),
    tex_steps: str | None = Form(None),
    tex_guidance_strength: str | None = Form(None),
    tex_rescale_t: str | None = Form(None),
) -> dict[str, Any]:
    if num_samples < 1:
        raise HTTPException(status_code=400, detail="num_samples must be at least 1.")
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        raw_bytes = await image.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    pil_image = _decode_uploaded_image(raw_bytes)

    request_id = uuid.uuid4().hex
    output_name = f"{request_id}.glb"
    output_path = OUTPUT_DIR / output_name

    sparse_structure_sampler_params = _build_sampler_params(
        steps=_optional_int(ss_steps),
        guidance_strength=_optional_float(ss_guidance_strength),
        rescale_t=_optional_float(ss_rescale_t),
    )
    shape_slat_sampler_params = _build_sampler_params(
        steps=_optional_int(shape_steps),
        guidance_strength=_optional_float(shape_guidance_strength),
        rescale_t=_optional_float(shape_rescale_t),
    )
    tex_slat_sampler_params = _build_sampler_params(
        steps=_optional_int(tex_steps),
        guidance_strength=_optional_float(tex_guidance_strength),
        rescale_t=_optional_float(tex_rescale_t),
    )

    started_at = time.time()
    try:
        result = await run_in_threadpool(
            pipeline_manager.generate,
            image=pil_image,
            output_path=output_path,
            num_samples=num_samples,
            seed=seed,
            pipeline_type=pipeline_type,
            preprocess_image=_bool_from_form(preprocess_image),
            max_num_tokens=max_num_tokens,
            simplify_target=simplify_target,
            texture_size=texture_size,
            remesh=_bool_from_form(remesh),
            remesh_band=remesh_band,
            remesh_project=remesh_project,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            shape_slat_sampler_params=shape_slat_sampler_params,
            tex_slat_sampler_params=tex_slat_sampler_params,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    elapsed = time.time() - started_at
    return {
        "id": request_id,
        "model_id": settings.model_id,
        "pipeline_type": pipeline_type,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 2),
        "download_url": _public_output_url(request, output_name),
        "filename": output_name,
        "num_vertices": result["num_vertices"],
        "num_faces": result["num_faces"],
    }


@app.post("/generate/file")
async def generate_file(
    request: Request,
    image: UploadFile = File(...),
    seed: int = Form(42),
    pipeline_type: str = Form(settings.default_pipeline_type),
) -> FileResponse:
    result = await generate(
        request=request,
        image=image,
        seed=seed,
        num_samples=1,
        pipeline_type=pipeline_type,
        preprocess_image="true",
        max_num_tokens=49152,
        simplify_target=1000000,
        texture_size=2048,
        remesh="true",
        remesh_band=1.0,
        remesh_project=0.0,
        ss_steps=None,
        ss_guidance_strength=None,
        ss_rescale_t=None,
        shape_steps=None,
        shape_guidance_strength=None,
        shape_rescale_t=None,
        tex_steps=None,
        tex_guidance_strength=None,
        tex_rescale_t=None,
    )
    filename = result["filename"]
    return FileResponse(
        path=str(OUTPUT_DIR / filename),
        media_type="model/gltf-binary",
        filename=filename,
    )
