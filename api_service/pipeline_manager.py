from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

from .config import OUTPUT_DIR, CACHE_DIR, settings
from .hf_auth import configure_huggingface_auth


class PipelineManager:
    def __init__(self) -> None:
        self._pipeline: Trellis2ImageTo3DPipeline | None = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> Trellis2ImageTo3DPipeline:
        if self._pipeline is not None:
            return self._pipeline

        with self._load_lock:
            if self._pipeline is not None:
                return self._pipeline

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

            configure_huggingface_auth()
            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(settings.model_id)
            pipeline.low_vram = settings.low_vram
            if settings.device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("TRELLIS API is configured for CUDA, but no GPU is available.")
                pipeline.cuda()
            else:
                pipeline.to(torch.device(settings.device))
            self._pipeline = pipeline
            return pipeline

    def remove_background(self, image: Image.Image) -> Image.Image:
        pipeline = self.load()

        with self._run_lock:
            if pipeline.rembg_model is None:
                raise RuntimeError("Background removal model is not available.")

            if pipeline.low_vram:
                pipeline.rembg_model.to(pipeline.device)
            output = pipeline.rembg_model(image.convert("RGB"))
            if pipeline.low_vram:
                pipeline.rembg_model.cpu()
            return output

    def generate(
        self,
        image: Image.Image,
        *,
        output_path: Path,
        num_samples: int,
        seed: int,
        pipeline_type: str,
        preprocess_image: bool,
        max_num_tokens: int,
        simplify_target: int,
        texture_size: int,
        remesh: bool,
        remesh_band: float,
        remesh_project: float,
        sparse_structure_sampler_params: dict[str, Any],
        shape_slat_sampler_params: dict[str, Any],
        tex_slat_sampler_params: dict[str, Any],
    ) -> dict[str, Any]:
        pipeline = self.load()

        with self._run_lock:
            meshes = pipeline.run(
                image=image,
                num_samples=num_samples,
                seed=seed,
                sparse_structure_sampler_params=sparse_structure_sampler_params,
                shape_slat_sampler_params=shape_slat_sampler_params,
                tex_slat_sampler_params=tex_slat_sampler_params,
                preprocess_image=preprocess_image,
                pipeline_type=pipeline_type,
                max_num_tokens=max_num_tokens,
            )
            if not meshes:
                raise RuntimeError("The pipeline returned no meshes.")

            mesh = meshes[0]
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=simplify_target,
                texture_size=texture_size,
                remesh=remesh,
                remesh_band=remesh_band,
                remesh_project=remesh_project,
                verbose=True,
            )
            glb.export(output_path, file_type="glb")

            return {
                "output_path": str(output_path),
                "num_vertices": int(mesh.vertices.shape[0]),
                "num_faces": int(mesh.faces.shape[0]),
            }


pipeline_manager = PipelineManager()
