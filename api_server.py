#!/usr/bin/env python3
"""
LTX-2 RunPod Serverless API Handler
Wraps DistilledPipeline for text+image to video+audio generation
"""
import os
import sys
import base64
import tempfile
import traceback
import shutil
import warnings
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)

# Add .venv packages to path
sys.path.insert(0, "/workspace/.venv/lib/python3.10/site-packages")

import torch
import requests
import runpod
from ltx_pipelines.distilled import DistilledPipeline
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.utils.media_io import encode_video

# Import download functions for automatic model downloading
from download_models import (
    download_transformer_checkpoint,
    download_spatial_upsampler,
    download_gemma_encoder
)


# Logging utilities
def log_info(message):
    print(f"ℹ️  [INFO] {message}")

def log_success(message):
    print(f"✅ [SUCCESS] {message}")

def log_warning(message):
    print(f"⚠️  [WARNING] {message}")

def log_error(message):
    print(f"❌ [ERROR] {message}")

def log_step(message):
    print(f"🔄 [STEP] {message}")


# Model configuration
MODEL_CONFIG = {
    "checkpoint_path": "/workspace/models/checkpoints/ltx-2-19b-distilled-fp8.safetensors",
    "spatial_upsampler_path": "/workspace/models/checkpoints/ltx-2-spatial-upscaler-x2-1.0.safetensors",
    "gemma_root": "/workspace/models/gemma",
}


class LTX2API:
    """LTX-2 API Handler for RunPod Serverless"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.models_loaded = False

        # GPU check
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available. LTX-2 requires GPU acceleration.")

        log_info(f"GPU: {torch.cuda.get_device_name(0)}")
        log_info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

        # Verify models exist
        self._verify_models()

        log_success("LTX2 API initialized successfully")

    def _verify_models(self):
        """Verify that all required models are present, download if missing"""
        log_step("Verifying model files...")

        required_files = [
            MODEL_CONFIG["checkpoint_path"],
            MODEL_CONFIG["spatial_upsampler_path"],
            os.path.join(MODEL_CONFIG["gemma_root"], "config.json"),
        ]

        missing_files = [f for f in required_files if not os.path.exists(f)]

        if missing_files:
            log_warning("Missing required model files, downloading...")
            for f in missing_files:
                log_warning(f"  - {f}")

            # Download missing models
            try:
                log_step("Starting model downloads (this may take 15-20 minutes)...")

                # Download transformer if missing
                if not os.path.exists(MODEL_CONFIG["checkpoint_path"]):
                    if not download_transformer_checkpoint():
                        raise RuntimeError("Failed to download transformer checkpoint")

                # Download spatial upsampler if missing
                if not os.path.exists(MODEL_CONFIG["spatial_upsampler_path"]):
                    if not download_spatial_upsampler():
                        raise RuntimeError("Failed to download spatial upsampler")

                # Download Gemma encoder if missing
                if not os.path.exists(os.path.join(MODEL_CONFIG["gemma_root"], "config.json")):
                    if not download_gemma_encoder():
                        raise RuntimeError("Failed to download Gemma text encoder")

                log_success("All models downloaded successfully")

            except Exception as e:
                log_error(f"Model download failed: {str(e)}")
                raise RuntimeError(
                    f"Failed to download required models: {str(e)}"
                )

        log_success("All model files verified")

    def load_models(self):
        """Load DistilledPipeline with FP8 + XFormers optimizations"""
        if self.models_loaded:
            log_info("Models already loaded")
            return

        log_step("Loading LTX-2 DistilledPipeline...")

        try:
            self.pipeline = DistilledPipeline(
                checkpoint_path=MODEL_CONFIG["checkpoint_path"],
                gemma_root=MODEL_CONFIG["gemma_root"],
                spatial_upsampler_path=MODEL_CONFIG["spatial_upsampler_path"],
                loras=[],  # No LoRAs for base implementation
                device=self.device,
                fp8transformer=True  # Enable FP8 optimization
            )

            self.models_loaded = True
            log_success("Pipeline loaded successfully")
            log_info("Optimizations: FP8 enabled, XFormers auto-detected")

        except Exception as e:
            log_error(f"Failed to load pipeline: {str(e)}")
            raise

    def _prepare_input_file(self, input_data: str, suffix: str) -> str:
        """Prepare input file from base64 string or URL"""
        if input_data.startswith('http://') or input_data.startswith('https://'):
            log_info(f"Downloading from URL...")
            return self._download_file(input_data, suffix)
        else:
            log_info("Decoding from base64...")
            return self._load_file_from_base64(input_data, suffix)

    def _load_file_from_base64(self, base64_string: str, suffix: str) -> str:
        """Load file from base64 string"""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:'):
                base64_string = base64_string.split(',')[1]

            file_data = base64.b64decode(base64_string)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(file_data)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            raise ValueError(f"Failed to decode base64: {str(e)}")

    def _download_file(self, url: str, suffix: str) -> str:
        """Download file from URL"""
        try:
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(response.content)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            raise ValueError(f"Failed to download from URL: {str(e)}")

    def generate_video(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        seed: int = 10,
        height: int = 1024,
        width: int = 1536,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        image_frame_idx: int = 0,
        image_strength: float = 0.8,
        enhance_prompt: bool = False
    ) -> str:
        """
        Generate video using DistilledPipeline

        Args:
            prompt: Text description
            image_path: Optional image conditioning path
            seed: Random seed
            height: Output height (must be divisible by 64)
            width: Output width (must be divisible by 64)
            num_frames: Number of frames (formula: 8*K + 1)
            frame_rate: FPS
            image_frame_idx: Frame index for image conditioning
            image_strength: Conditioning strength (0-1)
            enhance_prompt: Use Gemma to enhance prompt

        Returns:
            Path to output MP4 file
        """
        # Load models if not loaded
        if not self.models_loaded:
            self.load_models()

        # Prepare image conditioning
        images = []
        if image_path:
            images = [(image_path, image_frame_idx, image_strength)]

        # Create temp output
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "output.mp4")

        try:
            log_step("Generating video...")
            log_info(f"Prompt: {prompt}")
            log_info(f"Resolution: {width}x{height}, Frames: {num_frames}, FPS: {frame_rate}")

            # TilingConfig for memory-efficient VAE decoding
            tiling_config = TilingConfig.default()

            # Run pipeline
            video_iterator, audio = self.pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=enhance_prompt
            )

            # Calculate video chunks for encoding
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

            # Encode to MP4
            log_step("Encoding video to MP4...")
            encode_video(
                video=video_iterator,
                fps=frame_rate,
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=output_path,
                video_chunks_number=video_chunks_number
            )

            log_success(f"Video generated: {output_path}")
            return output_path

        except Exception as e:
            log_error(f"Generation failed: {str(e)}")
            # Cleanup temp dir on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _video_to_base64(self, video_path: str) -> str:
        """Convert video file to base64 string"""
        try:
            with open(video_path, 'rb') as video_file:
                video_data = video_file.read()
                base64_string = base64.b64encode(video_data).decode()
                return base64_string
        except Exception as e:
            raise ValueError(f"Failed to convert video to base64: {str(e)}")


# Initialize API
api = LTX2API()


def handler(job):
    """
    RunPod handler function

    Expected input format:
    {
        "prompt": "A cat walking on the beach",  # required
        "image": "data:image/jpeg;base64,..." or "https://...",  # optional
        "seed": 10,  # optional, default: 10
        "height": 1024,  # optional, default: 1024 (must be divisible by 64)
        "width": 1536,  # optional, default: 1536 (must be divisible by 64)
        "num_frames": 121,  # optional, default: 121 (formula: 8*K + 1)
        "frame_rate": 24.0,  # optional, default: 24.0
        "image_frame_idx": 0,  # optional, default: 0
        "image_strength": 0.8,  # optional, default: 0.8
        "enhance_prompt": false  # optional, default: false
    }

    Returns:
    {
        "success": true,
        "video": "base64_encoded_mp4_data",
        "metadata": {
            "prompt": "...",
            "seed": 10,
            "height": 1024,
            "width": 1536,
            "num_frames": 121,
            "frame_rate": 24.0,
            "duration": 5.04  # seconds
        }
    }
    """
    try:
        input_data = job["input"]

        # Validate required fields
        if "prompt" not in input_data:
            return {"success": False, "error": "prompt is required"}

        prompt = input_data["prompt"]

        # Extract optional parameters with defaults
        seed = input_data.get("seed", 10)
        height = input_data.get("height", 1024)
        width = input_data.get("width", 1536)
        num_frames = input_data.get("num_frames", 121)
        frame_rate = input_data.get("frame_rate", 24.0)
        image_frame_idx = input_data.get("image_frame_idx", 0)
        image_strength = input_data.get("image_strength", 0.8)
        enhance_prompt = input_data.get("enhance_prompt", False)

        # Validate dimensions
        if height % 64 != 0 or width % 64 != 0:
            return {
                "success": False,
                "error": "height and width must be divisible by 64"
            }

        # Validate num_frames formula: 8*K + 1
        if (num_frames - 1) % 8 != 0:
            return {
                "success": False,
                "error": "num_frames must follow formula: 8*K + 1 (e.g., 9, 17, 25, ..., 121)"
            }

        # Prepare image if provided
        image_path = None
        if "image" in input_data:
            log_step("Preparing image input...")
            image_path = api._prepare_input_file(input_data["image"], suffix=".jpg")

        # Generate video
        output_video_path = api.generate_video(
            prompt=prompt,
            image_path=image_path,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            image_frame_idx=image_frame_idx,
            image_strength=image_strength,
            enhance_prompt=enhance_prompt
        )

        # Convert to base64
        log_step("Encoding output video to base64...")
        video_base64 = api._video_to_base64(output_video_path)

        # Cleanup
        try:
            if image_path:
                os.remove(image_path)
            temp_dir = os.path.dirname(output_video_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

        # Calculate duration
        duration = num_frames / frame_rate

        log_success("Request completed successfully")

        return {
            "success": True,
            "video": video_base64,
            "metadata": {
                "prompt": prompt,
                "seed": seed,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "duration": round(duration, 2)
            }
        }

    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        log_error(error_msg)
        log_error(traceback.format_exc())
        return {
            "success": False,
            "error": error_msg
        }


# Start RunPod serverless handler
log_info("Starting RunPod serverless handler...")
runpod.serverless.start({"handler": handler})
