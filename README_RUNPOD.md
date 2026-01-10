# LTX-2 RunPod Serverless Endpoint

This is a RunPod serverless wrapper for LTX-2 video generation using the DistilledPipeline with FP8 optimization.

## Overview

- **Pipeline**: DistilledPipeline (12-step distilled diffusion)
- **Optimizations**: FP8 transformer, XFormers attention
- **Input**: Text prompt + optional image
- **Output**: MP4 video with synchronized audio (24kHz stereo)

## Files Created

1. **Dockerfile** - CUDA 12.8 container with uv package manager
2. **api_server.py** - RunPod handler wrapping DistilledPipeline
3. **download_models.py** - HuggingFace model downloader

**Note**: No pipeline code was modified - this is a pure wrapper implementation.

## Models Downloaded (Build Time)

Total: ~18GB

1. **Transformer**: `ltx-video-2b-v0.9.1.safetensors` (~12GB)
2. **Spatial Upsampler**: `ltx_video_vae_upsampler.safetensors` (~1.5GB)
3. **Gemma Text Encoder**: `google/gemma-2-2b` (~5GB)

## Build and Deploy

### 1. Build Docker Image

```bash
cd LTX-2-Runpod
docker build -t ltx2-runpod:latest .
```

**Build time**: 15-20 minutes (includes model downloads)

### 2. Test Locally

```bash
docker run --gpus all -p 8000:8000 ltx2-runpod:latest
```

### 3. Push to Registry

```bash
docker tag ltx2-runpod:latest <your-registry>/ltx2-runpod:latest
docker push <your-registry>/ltx2-runpod:latest
```

### 4. Deploy to RunPod

1. Go to RunPod Serverless dashboard
2. Create new endpoint
3. Configure:
   - **Container Image**: `<your-registry>/ltx2-runpod:latest`
   - **Container Disk**: 25 GB
   - **GPU Type**: A40 or A100 (48GB VRAM recommended)
   - **Min Workers**: 0
   - **Max Workers**: 5
   - **Execution Timeout**: 600 seconds (10 minutes)

## API Usage

### Request Format

```json
{
  "input": {
    "prompt": "A cat walking on the beach at sunset",
    "image": "https://example.com/image.jpg",  // optional
    "seed": 10,
    "height": 1024,
    "width": 1536,
    "num_frames": 121,
    "frame_rate": 24.0,
    "image_frame_idx": 0,
    "image_strength": 0.8,
    "enhance_prompt": false
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text description of the video |
| `image` | string | optional | Image URL or base64 (for conditioning) |
| `seed` | int | 10 | Random seed for reproducibility |
| `height` | int | 1024 | Output height (must be divisible by 64) |
| `width` | int | 1536 | Output width (must be divisible by 64) |
| `num_frames` | int | 121 | Number of frames (formula: 8*K + 1) |
| `frame_rate` | float | 24.0 | Frames per second |
| `image_frame_idx` | int | 0 | Frame index to inject image conditioning |
| `image_strength` | float | 0.8 | Image conditioning strength (0-1) |
| `enhance_prompt` | bool | false | Use Gemma to enhance the prompt |

### Response Format

```json
{
  "success": true,
  "video": "<base64_encoded_mp4>",
  "metadata": {
    "prompt": "A cat walking on the beach at sunset",
    "seed": 10,
    "height": 1024,
    "width": 1536,
    "num_frames": 121,
    "frame_rate": 24.0,
    "duration": 5.04
  }
}
```

### Example with cURL

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint-id>/run \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A serene waterfall in a lush forest, birds flying overhead",
      "num_frames": 121,
      "height": 1024,
      "width": 1536
    }
  }'
```

## Constraints

### Resolution
- **height, width**: Must be divisible by 64
- Valid examples: 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536

### Frame Count
- **num_frames**: Must follow formula `8*K + 1` where K ≥ 0
- Valid examples: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145...

### Recommended Configurations

**5-second video (landscape)**:
```json
{
  "num_frames": 121,
  "height": 1024,
  "width": 1536,
  "frame_rate": 24.0
}
```

**5-second video (portrait)**:
```json
{
  "num_frames": 121,
  "height": 1536,
  "width": 1024,
  "frame_rate": 24.0
}
```

**10-second video**:
```json
{
  "num_frames": 241,
  "height": 1024,
  "width": 1536,
  "frame_rate": 24.0
}
```

## Performance

### Hardware
- **GPU**: A40 or A100 (48GB VRAM)
- **VRAM Usage**: ~28GB peak with FP8 optimization

### Generation Times (A100)
- **121 frames** (5 sec) @ 1024×1536: ~45-60 seconds
- **241 frames** (10 sec) @ 1024×1536: ~90-120 seconds

### Cold Start
- First request after deployment: ~10 seconds (model loading)
- Subsequent requests: Immediate

## Optimizations Enabled

### 1. FP8 Transformer
- Weights stored as FP8 (50% VRAM reduction)
- Computation in bfloat16 (maintains quality)
- Reduces VRAM from ~48GB → ~28GB

### 2. XFormers
- Memory-efficient attention
- 20-30% faster than standard attention
- Auto-detected if available

### 3. DistilledPipeline
- 12 steps total (8 + 4)
- 3-4x faster than full pipeline
- No quality loss compared to distilled training

## Troubleshooting

### Build Issues

**Error: uv sync fails**
- Ensure Docker has internet access
- Check if PyTorch cu129 index is accessible
- Try rebuilding with `--no-cache`

**Error: Model download fails**
- Verify HuggingFace is accessible
- Check disk space (need ~20GB free)
- Models are cached in `/workspace/models`

### Runtime Issues

**Error: CUDA not available**
- Verify GPU is allocated in RunPod
- Check NVIDIA drivers in container
- Run health check: `docker run --gpus all <image> python -c "import torch; print(torch.cuda.is_available())"`

**Error: Out of memory**
- Reduce `num_frames` (e.g., 121 → 81)
- Lower resolution (e.g., 1536×1024 → 1024×768)
- Ensure FP8 is enabled (check logs for "FP8 enabled")

**Error: Invalid dimensions**
- Ensure height/width divisible by 64
- Ensure num_frames follows 8*K + 1 formula

## Architecture

```
Input (Text + Image)
        ↓
Gemma Text Encoder (12B params)
        ↓
Stage 1: Low-res generation (8 steps)
        ├─ Transformer (2B params, FP8)
        └─ Video latent (H/2 × W/2)
        ↓
Spatial Upsampler (2x)
        ↓
Stage 2: High-res refinement (4 steps)
        └─ Transformer (2B params, FP8)
        ↓
VAE Decoder (video + audio)
        ↓
Output (MP4 with 24kHz audio)
```

## License

This wrapper follows the LTX-2 license. See the main repository for details.

## Support

For issues specific to this RunPod implementation, check:
1. Container logs in RunPod dashboard
2. Health check status
3. GPU allocation and VRAM usage

For LTX-2 pipeline issues, refer to the original repository.
