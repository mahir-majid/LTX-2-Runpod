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
from typing import Dict, Any, Optional, List

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)

# Add .venv packages to path
sys.path.insert(0, "/workspace/.venv/lib/python3.10/site-packages")

import torch
import torch.nn.functional as F
import torchaudio
import requests
import runpod
import gc
from ltx_pipelines.distilled import DistilledPipeline
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.types import AudioLatentShape
from huggingface_hub import hf_hub_download

# Import download functions for automatic model downloading
from download_models import (
    download_transformer_checkpoint,
    download_spatial_upsampler,
    download_gemma_encoder
)


# ============ MEMORY DIAGNOSTICS ============
def log_memory(label: str):
    """Log GPU memory usage at a specific point"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"🔍 [MEMORY] {label}")
        print(f"           Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Peak: {max_allocated:.2f} GB")

# Patch cleanup_memory to log
from ltx_pipelines.utils import helpers as ltx_helpers
_original_cleanup = ltx_helpers.cleanup_memory
def logged_cleanup():
    log_memory("Before cleanup_memory()")
    _original_cleanup()
    log_memory("After cleanup_memory()")
ltx_helpers.cleanup_memory = logged_cleanup

# Patch ModelLedger methods to log
from ltx_pipelines.utils.model_ledger import ModelLedger
_model_ledger_methods = ['text_encoder', 'transformer', 'video_encoder', 'video_decoder', 'audio_decoder', 'vocoder', 'spatial_upsampler']
for method_name in _model_ledger_methods:
    if hasattr(ModelLedger, method_name):
        original = getattr(ModelLedger, method_name)
        def make_logged(name, orig):
            def logged(self):
                log_memory(f"Before ModelLedger.{name}()")
                result = orig(self)
                log_memory(f"After ModelLedger.{name}()")
                return result
            return logged
        setattr(ModelLedger, method_name, make_logged(method_name, original))
# ============ END MEMORY DIAGNOSTICS ============


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


# ============ AUDIO REFERENCE UTILITIES ============
def load_and_encode_reference_audio(
    audio_path: str,
    audio_encoder,
    target_frames: int,
    device: torch.device,
    noise_scale: float = 0.75
) -> torch.Tensor:
    """
    Load reference audio, encode to latent space, and prepare for initialization.

    Args:
        audio_path: Path to reference audio file
        audio_encoder: Audio VAE encoder
        target_frames: Target number of audio frames for generation
        device: Torch device
        noise_scale: Noise scale for blending (0.0-1.0)

    Returns:
        Audio latent tensor ready for use as initial_audio_latent
    """
    log_step(f"Loading reference audio: {audio_path}")

    # Load audio with torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 16kHz if needed (LTX-2 audio VAE expects 16kHz)
    if sample_rate != AUDIO_SAMPLE_RATE:
        log_info(f"Resampling audio from {sample_rate}Hz to {AUDIO_SAMPLE_RATE}Hz")
        resampler = torchaudio.transforms.Resample(sample_rate, AUDIO_SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Move to device
    waveform = waveform.to(device)

    log_step("Encoding reference audio to latent space...")

    # Encode audio to latent space
    # Audio encoder expects spectrogram input, so we need to use AudioProcessor
    from ltx_core.model.audio_vae import AudioProcessor

    # Get audio parameters from encoder (with fallback defaults)
    sample_rate = getattr(audio_encoder, 'sample_rate', AUDIO_SAMPLE_RATE)
    mel_bins = getattr(audio_encoder, 'mel_bins', 64)
    mel_hop_length = getattr(audio_encoder, 'mel_hop_length', 160)
    n_fft = getattr(audio_encoder, 'n_fft', 1024)

    audio_processor = AudioProcessor(
        sample_rate=sample_rate,
        mel_bins=mel_bins,
        mel_hop_length=mel_hop_length,
        n_fft=n_fft
    )

    # Convert waveform to spectrogram (mel spectrogram)
    spectrogram = audio_processor.waveform_to_mel(
        waveform.unsqueeze(0),
        waveform_sample_rate=sample_rate
    )
    spectrogram = spectrogram.to(device)

    # Encode to latent
    with torch.no_grad():
        reference_latent = audio_encoder(spectrogram)

    # Get reference dimensions
    ref_frames = reference_latent.shape[2]

    log_info(f"Reference audio: {ref_frames} frames → Target: {target_frames} frames")

    # Upsample or downsample to match target duration
    if ref_frames != target_frames:
        log_step(f"Resampling reference latent from {ref_frames} to {target_frames} frames")
        reference_latent = F.interpolate(
            reference_latent,
            size=(target_frames, reference_latent.shape[3]),
            mode='bilinear',
            align_corners=False
        )

    log_success(f"Reference audio encoded: shape={reference_latent.shape}, noise_scale={noise_scale}")

    return reference_latent

# ============ END AUDIO REFERENCE UTILITIES ============


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
        self.current_lora_config = None  # Track current LoRA config

        # GPU check
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available. LTX-2 requires GPU acceleration.")

        log_info(f"GPU: {torch.cuda.get_device_name(0)}")
        log_info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        
        # Verify PyTorch CUDA allocator config
        pytorch_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "NOT SET")
        log_info(f"PYTORCH_CUDA_ALLOC_CONF: {pytorch_alloc_conf}")

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

    def _get_lora_path(self, camera_lora: Optional[str]) -> Optional[str]:
        """Get LoRA file path, downloading from HuggingFace if needed"""
        if camera_lora is None:
            return None
        
        if camera_lora == "Static":
            repo_id = "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static"
            filename = "ltx-2-19b-lora-camera-control-static.safetensors"
            local_path = f"/workspace/models/checkpoints/{filename}"
            
            # Check if already downloaded
            if os.path.exists(local_path):
                log_info(f"LoRA file already exists: {local_path}")
                return local_path
            
            # Download from HuggingFace
            try:
                log_step(f"Downloading LoRA from HuggingFace: {repo_id}")
                hf_token = os.environ.get("HF_TOKEN")
                
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir="/workspace/models/checkpoints",
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    token=hf_token
                )
                
                log_success(f"LoRA downloaded successfully: {downloaded_path}")
                return downloaded_path
            except Exception as e:
                log_error(f"Failed to download LoRA: {str(e)}")
                raise RuntimeError(f"Failed to download LoRA from HuggingFace: {str(e)}")
        else:
            log_warning(f"Unknown camera_lora value: {camera_lora}. Supported values: 'Static'")
            return None

    def load_models(self, loras: Optional[List[LoraPathStrengthAndSDOps]] = None):
        """Load DistilledPipeline with FP8 optimization and optional LoRAs"""
        # Check if LoRA config changed
        lora_config_key = tuple(sorted([(l.path, l.strength) for l in (loras or [])]))
        
        if self.models_loaded and self.current_lora_config == lora_config_key:
            log_info("Models already loaded with same LoRA config")
            return
        
        # If LoRA config changed, unload existing pipeline
        if self.models_loaded and self.current_lora_config != lora_config_key:
            log_step("LoRA config changed, reloading pipeline...")
            self.pipeline = None
            self.models_loaded = False
            gc.collect()
            torch.cuda.empty_cache()

        log_step("Loading LTX-2 DistilledPipeline...")
        if loras:
            log_info(f"Using {len(loras)} LoRA(s)")
            for lora in loras:
                log_info(f"  - LoRA: {lora.path} (strength: {lora.strength})")
        else:
            log_info("No LoRAs configured")
        
        log_memory("Before pipeline creation")

        try:
            self.pipeline = DistilledPipeline(
                checkpoint_path=MODEL_CONFIG["checkpoint_path"],
                gemma_root=MODEL_CONFIG["gemma_root"],
                spatial_upsampler_path=MODEL_CONFIG["spatial_upsampler_path"],
                loras=loras or [],
                device=self.device,
                fp8transformer=True  # Enable FP8 optimization
            )

            self.models_loaded = True
            self.current_lora_config = lora_config_key
            log_memory("After pipeline creation")
            log_success("Pipeline loaded successfully")
            log_info("Optimizations: FP8 enabled")
            if loras:
                log_info(f"LoRA(s) loaded: {len(loras)}")

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
        reference_audio_path: Optional[str] = None,
        audio_noise_scale: float = 0.75,
        seed: int = 10,
        height: int = 1024,
        width: int = 1536,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        image_frame_idx: int = 0,
        image_strength: float = 0.8,
        enhance_prompt: bool = False,
        camera_lora: Optional[str] = "Static",
        camera_lora_strength: float = 1.0
    ) -> str:
        """
        Generate video using DistilledPipeline with optional reference audio for voice consistency

        Args:
            prompt: Text description
            image_path: Optional image conditioning path
            reference_audio_path: Optional path to reference audio for voice consistency (Approach 1)
            audio_noise_scale: Noise scale for reference audio blending (0.0-1.0, default: 0.75)
                              Lower values = stronger voice consistency, higher = more variation
            seed: Random seed
            height: Output height (must be divisible by 64)
            width: Output width (must be divisible by 64)
            num_frames: Number of frames (formula: 8*K + 1)
            frame_rate: FPS
            image_frame_idx: Frame index for image conditioning
            image_strength: Conditioning strength (0-1)
            enhance_prompt: Use Gemma to enhance prompt
            camera_lora: Camera LoRA type ("Static" for static camera, None to disable LoRA, default: "Static")
            camera_lora_strength: LoRA strength (0.0-1.0, default: 1.0)

        Returns:
            Path to output MP4 file
        """
        log_memory("Start of generate_video()")
        
        # Get LoRA path if requested
        lora_path = self._get_lora_path(camera_lora)
        loras = []
        if lora_path:
            loras = [LoraPathStrengthAndSDOps(lora_path, camera_lora_strength, LTXV_LORA_COMFY_RENAMING_MAP)]
            log_info(f"🎥 Using Camera LoRA: {camera_lora} (path: {lora_path}, strength: {camera_lora_strength})")
        else:
            log_info("🎥 No Camera LoRA configured")
        
        # Load models with LoRAs
        self.load_models(loras=loras if loras else None)

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

            # Process reference audio if provided
            reference_audio_latent = None
            if reference_audio_path:
                log_info(f"🎤 Using reference audio for voice consistency (noise_scale={audio_noise_scale})")

                # Calculate target audio frames
                # Audio latent frames = (num_video_frames - 1) based on LTX-2 architecture
                target_audio_frames = num_frames

                # Load audio encoder (not in ModelLedger, must load separately)
                from ltx_trainer.model_loader import load_audio_vae_encoder
                audio_encoder = load_audio_vae_encoder(
                    checkpoint_path=MODEL_CONFIG["checkpoint_path"],
                    device=self.device,
                    dtype=torch.bfloat16
                )

                # Encode reference audio
                reference_audio_latent = load_and_encode_reference_audio(
                    audio_path=reference_audio_path,
                    audio_encoder=audio_encoder,
                    target_frames=target_audio_frames,
                    device=self.device,
                    noise_scale=audio_noise_scale
                )

                # Clean up encoder to save memory
                del audio_encoder
                gc.collect()
                torch.cuda.empty_cache()
            else:
                log_info("🎤 No reference audio provided, generating audio from scratch")

            # TilingConfig for memory-efficient VAE decoding
            tiling_config = TilingConfig.default()

            # Reset peak memory stats before pipeline call
            torch.cuda.reset_peak_memory_stats()
            log_memory("Before pipeline call")

            # Run pipeline and encode video with autograd disabled
            # This is critical for memory - without no_grad, PyTorch retains
            # computation graphs and FP8 upcast tensors, causing OOM
            with torch.no_grad():
                video_iterator, audio = self._generate_with_reference_audio(
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    images=images,
                    tiling_config=tiling_config,
                    enhance_prompt=enhance_prompt,
                    reference_audio_latent=reference_audio_latent,
                    audio_noise_scale=audio_noise_scale
                )

                log_memory("After pipeline call")

                # Calculate video chunks for encoding
                video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

                # Encode to MP4
                log_step("Encoding video to MP4...")
                log_memory("Before video encoding")
                encode_video(
                    video=video_iterator,
                    fps=frame_rate,
                    audio=audio,
                    audio_sample_rate=AUDIO_SAMPLE_RATE,
                    output_path=output_path,
                    video_chunks_number=video_chunks_number
                )
                log_memory("After video encoding")

            log_success(f"Video generated: {output_path}")
            return output_path

        except Exception as e:
            log_error(f"Generation failed: {str(e)}")
            # Cleanup temp dir on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _generate_with_reference_audio(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list,
        tiling_config: TilingConfig,
        enhance_prompt: bool,
        reference_audio_latent: Optional[torch.Tensor],
        audio_noise_scale: float
    ):
        """
        Custom wrapper around DistilledPipeline that injects reference audio latent.
        This implements Approach 1 (audio latent initialization + partial noising).
        """
        from ltx_pipelines.utils.helpers import (
            denoise_audio_video,
            image_conditionings_by_replacing_latent,
            euler_denoising_loop,
            simple_denoising_func,
            cleanup_memory
        )
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.model.upsampler import upsample_video
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.helpers import generate_enhanced_prompt

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        text_encoder = self.pipeline.model_ledger.text_encoder()
        if enhance_prompt:
            image_path_for_prompt = images[0][0] if len(images) > 0 else None
            prompt = generate_enhanced_prompt(text_encoder, prompt, image_path_for_prompt)
        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p

        torch.cuda.synchronize()
        del text_encoder
        cleanup_memory()

        # Stage 1: Initial low resolution video generation
        video_encoder = self.pipeline.model_ledger.video_encoder()
        transformer = self.pipeline.model_ledger.transformer()
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        def denoising_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        # Downsample reference audio latent for Stage 1 (half resolution)
        stage_1_reference_audio = None
        if reference_audio_latent is not None:
            # Stage 1 operates at half resolution, so we keep the same audio latent
            # (audio is 1D temporal, not affected by spatial resolution)
            stage_1_reference_audio = reference_audio_latent

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline.pipeline_components,
            dtype=dtype,
            device=self.device,
            initial_audio_latent=stage_1_reference_audio,  # ← Approach 1 injection
            noise_scale=audio_noise_scale
        )

        # Stage 2: Upsample and refine
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.pipeline.model_ledger.spatial_upsampler()
        )

        torch.cuda.synchronize()
        cleanup_memory()

        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        stage_2_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width,
            height=height,
            fps=frame_rate
        )
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=stage_2_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,  # Re-use Stage 1 audio
        )

        torch.cuda.synchronize()
        del transformer
        del video_encoder
        cleanup_memory()

        decoded_video = vae_decode_video(
            video_state.latent,
            self.pipeline.model_ledger.video_decoder(),
            tiling_config
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent,
            self.pipeline.model_ledger.audio_decoder(),
            self.pipeline.model_ledger.vocoder()
        )

        return decoded_video, decoded_audio

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
        "reference_audio": "data:audio/wav;base64,..." or "https://...",  # optional - for voice consistency
        "audio_noise_scale": 0.75,  # optional, default: 0.75. Lower = stronger voice consistency (0.0-1.0)
        "seed": 10,  # optional, default: 10
        "height": 1024,  # optional, default: 1024 (must be divisible by 64)
        "width": 1536,  # optional, default: 1536 (must be divisible by 64)
        "num_frames": 121,  # optional, default: 121 (formula: 8*K + 1)
        "frame_rate": 24.0,  # optional, default: 24.0
        "image_frame_idx": 0,  # optional, default: 0
        "image_strength": 0.8,  # optional, default: 0.8
        "enhance_prompt": false,  # optional, default: false
        "camera_lora": "Static",  # optional, default: "Static". Use "Static" for static camera LoRA, null to disable
        "camera_lora_strength": 1.0  # optional, default: 1.0. LoRA strength (0.0-1.0)
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
        camera_lora = input_data.get("camera_lora", "Static")
        camera_lora_strength = input_data.get("camera_lora_strength", 1.0)
        audio_noise_scale = input_data.get("audio_noise_scale", 0.75)

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

        # Validate camera_lora_strength
        if camera_lora_strength < 0.0 or camera_lora_strength > 1.0:
            return {
                "success": False,
                "error": "camera_lora_strength must be between 0.0 and 1.0"
            }

        # Validate audio_noise_scale
        if audio_noise_scale < 0.0 or audio_noise_scale > 1.0:
            return {
                "success": False,
                "error": "audio_noise_scale must be between 0.0 and 1.0"
            }

        # Prepare image if provided
        image_path = None
        if "image" in input_data:
            log_step("Preparing image input...")
            image_path = api._prepare_input_file(input_data["image"], suffix=".jpg")

        # Prepare reference audio if provided
        reference_audio_path = None
        if "reference_audio" in input_data:
            log_step("Preparing reference audio input...")
            reference_audio_path = api._prepare_input_file(input_data["reference_audio"], suffix=".wav")

        # Generate video
        output_video_path = api.generate_video(
            prompt=prompt,
            image_path=image_path,
            reference_audio_path=reference_audio_path,
            audio_noise_scale=audio_noise_scale,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            image_frame_idx=image_frame_idx,
            image_strength=image_strength,
            enhance_prompt=enhance_prompt,
            camera_lora=camera_lora,
            camera_lora_strength=camera_lora_strength
        )

        # Convert to base64
        log_step("Encoding output video to base64...")
        video_base64 = api._video_to_base64(output_video_path)

        # Cleanup
        try:
            if image_path:
                os.remove(image_path)
            if reference_audio_path:
                os.remove(reference_audio_path)
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
