# lightning.ai-api

TRELLIS.2 inference has been converted into a FastAPI service that is intended to run directly inside Lightning AI Studio.

## What is included

- `api_service/` with a FastAPI app, Hugging Face auth bootstrap, and a singleton pipeline manager
- `trellis2/` inference runtime
- `o-voxel/` local package for mesh export
- `requirements.txt` for the API and native runtime dependencies
- `install_lightning.sh` and `start_api.sh` for Lightning startup

## Lightning notes

- Put your Hugging Face key in Lightning secrets as one of:
  - `HF_TOKEN`
  - `HUGGINGFACE_HUB_TOKEN`
  - `HUGGINGFACE_API_KEY`
- The service uses Lightning's own public app URL. No tunnel or ngrok is needed.
- The API defaults to `xformers` attention, `flex_gemm` sparse convolution, and `TRELLIS_LOW_VRAM=0` so models stay on GPU for faster repeated inference.
- If you hit GPU memory limits, you can set `TRELLIS_LOW_VRAM=1` to turn model offloading back on.

## Start in Lightning Studio

```bash
bash install_lightning.sh
bash start_api.sh
```

Then open the Lightning-provided URL for the running app.

## Build behavior

- Builds are stored in a workspace-local virtual environment at `.venv/`
- Completed install steps are tracked in `.build_state/`
- On the next run, the installer checks imports and skips steps that already succeeded
- If a late native build step fails, the earlier successful steps are reused on retry

This is designed to avoid losing a long build because one package fails near the end.

## Endpoints

- `GET /health`
- `GET /config`
- `POST /rembg`
- `POST /generate`
- `POST /generate/file`
- `GET /outputs/{filename}`

## Recommended frontend flow

1. Upload the original image to `POST /rembg`
2. Show the returned processed PNG to the user
3. Send that processed image to `POST /generate`
4. Set `preprocess_image=false` when generating from the already-cleaned image

## Example request

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -F "image=@input.png" \
  -F "seed=42" \
  -F "pipeline_type=1024_cascade" \
  -F "preprocess_image=false" \
  -F "shape_steps=50" \
  -F "tex_steps=50"
```

## Example rembg request

```bash
curl -X POST "http://127.0.0.1:8000/rembg" \
  -F "image=@input.png"
```
