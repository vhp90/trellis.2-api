# Frontend API Guide

This guide explains how to use the deployed TRELLIS.2 API from your frontend.

## Base URL

After deployment on Lightning Studio, use your Lightning app URL as the API base URL.

Example:

```text
https://your-lightning-app-url.lightning.ai
```

Do not use ngrok or any other tunnel. The Lightning URL is the public API URL.

## Flow

Recommended user flow:

1. User uploads an image
2. Frontend sends it to `/rembg`
3. Frontend shows the returned transparent PNG preview
4. Frontend sends the processed image to `/generate`
5. Frontend receives a `.glb` download URL and uses it in the UI

## Endpoints

### `GET /health`

Use this to check whether the API is up.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "cuda_available": true,
  "device": "cuda",
  "model_id": "microsoft/TRELLIS.2-4B",
  "low_vram": false
}
```

### `GET /config`

Use this to inspect runtime defaults.

### `POST /rembg`

Uploads an image and returns a processed PNG with background removed.

Form field:

- `image`: image file

Example response:

```json
{
  "id": "abc123",
  "filename": "abc123-rembg.png",
  "content_type": "image/png",
  "download_url": "https://your-lightning-app-url.lightning.ai/outputs/abc123-rembg.png"
}
```

### `POST /generate`

Uploads an image and starts 3D generation. Returns metadata and a `.glb` URL.

Required form field:

- `image`: image file

Important fields:

- `seed`: integer
- `pipeline_type`: one of `512`, `1024`, `1024_cascade`, `1536_cascade`
- `preprocess_image`: `true` or `false`
- `num_samples`: usually keep this at `1`
- `max_num_tokens`
- `simplify_target`
- `texture_size`
- `remesh`: `true` or `false`
- `remesh_band`
- `remesh_project`
- `ss_steps`, `shape_steps`, `tex_steps`
- `ss_guidance_strength`, `shape_guidance_strength`, `tex_guidance_strength`
- `ss_rescale_t`, `shape_rescale_t`, `tex_rescale_t`

If the image already came from `/rembg`, send:

- `preprocess_image=false`

Example response:

```json
{
  "id": "xyz789",
  "model_id": "microsoft/TRELLIS.2-4B",
  "pipeline_type": "1024_cascade",
  "seed": 42,
  "elapsed_seconds": 28.4,
  "download_url": "https://your-lightning-app-url.lightning.ai/outputs/xyz789.glb",
  "filename": "xyz789.glb",
  "num_vertices": 120340,
  "num_faces": 240100
}
```

### `POST /generate/file`

Returns the generated `.glb` file directly instead of JSON.

## JavaScript Example

```javascript
const API_BASE = "https://your-lightning-app-url.lightning.ai";

export async function removeBackground(file) {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch(`${API_BASE}/rembg`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`rembg failed: ${response.status}`);
  }

  return await response.json();
}

export async function generateModel(file, options = {}) {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("seed", String(options.seed ?? 42));
  formData.append("pipeline_type", options.pipelineType ?? "1024_cascade");
  formData.append("preprocess_image", String(options.preprocessImage ?? false));

  if (options.shapeSteps != null) formData.append("shape_steps", String(options.shapeSteps));
  if (options.texSteps != null) formData.append("tex_steps", String(options.texSteps));
  if (options.ssSteps != null) formData.append("ss_steps", String(options.ssSteps));

  const response = await fetch(`${API_BASE}/generate`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`generate failed: ${response.status} ${text}`);
  }

  return await response.json();
}
```

## Suggested Frontend Integration

- Call `/health` once when the app loads
- Call `/rembg` immediately after image upload
- Show the processed PNG preview to the user
- Send the processed file to `/generate`
- Use the returned `download_url` for download buttons or model loaders

## Notes

- Generation is GPU-heavy and usually should be treated as a single active request per workspace.
- Keep `num_samples=1` unless you know the GPU has enough memory.
- If generation is slow, use the default GPU-resident setup and avoid enabling low-VRAM mode.
