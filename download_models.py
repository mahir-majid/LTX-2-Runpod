#!/usr/bin/env python3
"""
Download script for LTX-2 models
This script downloads all necessary models for the RunPod API server.
"""
import os
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

# Get HuggingFace token from environment variable (required)
HF_TOKEN = os.environ["HF_TOKEN"]


def log_info(message):
    """Log info message"""
    print(f"ℹ️  [INFO] {message}")

def log_success(message):
    """Log success message"""
    print(f"✅ [SUCCESS] {message}")

def log_warning(message):
    """Log warning message"""
    print(f"⚠️  [WARNING] {message}")

def log_error(message):
    """Log error message"""
    print(f"❌ [ERROR] {message}")

def log_step(message):
    """Log step message"""
    print(f"🔄 [STEP] {message}")


def get_directory_size(path):
    """Get directory size in GB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024**3)
    except Exception:
        return 0


def create_directories():
    """Create necessary model directories"""
    log_step("Creating model directories...")
    directories = [
        "/workspace/models/checkpoints",
        "/workspace/models/gemma"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        log_info(f"Created/verified: {directory}")

    log_success("All directories created")


def download_transformer_checkpoint():
    """Download LTX-2 transformer checkpoint"""
    log_step("Downloading LTX-2 transformer checkpoint...")
    log_info("Repository: Lightricks/LTX-2")

    checkpoint_path = "/workspace/models/checkpoints/ltx-2-19b-distilled-fp8.safetensors"

    if os.path.exists(checkpoint_path):
        size = os.path.getsize(checkpoint_path) / (1024**3)
        log_info(f"Transformer checkpoint already exists ({size:.1f} GB)")
        log_success("Transformer checkpoint found")
        return True

    try:
        log_info("Downloading ltx-2-19b-distilled-fp8.safetensors from HuggingFace...")
        log_step("Starting download...")
        start_time = time.time()

        downloaded_path = hf_hub_download(
            repo_id="Lightricks/LTX-2",
            filename="ltx-2-19b-distilled-fp8.safetensors",
            local_dir="/workspace/models/checkpoints",
            local_dir_use_symlinks=False,
            resume_download=True,
            token=HF_TOKEN
        )

        download_time = time.time() - start_time
        size = os.path.getsize(checkpoint_path) / (1024**3)

        log_success(f"Transformer checkpoint downloaded successfully")
        log_info(f"Download time: {download_time:.1f} seconds")
        log_info(f"File size: {size:.1f} GB")
        log_info(f"Path: {downloaded_path}")
        return True
    except Exception as e:
        log_error(f"Error downloading transformer checkpoint: {e}")
        log_info("\n📋 Troubleshooting:")
        log_info("1. Check your internet connection")
        log_info("2. Visit: https://huggingface.co/Lightricks/LTX-2")
        log_info("3. Ensure you have sufficient disk space (~20GB)")
        return False


def download_spatial_upsampler():
    """Download spatial upsampler model"""
    log_step("Downloading spatial upsampler...")
    log_info("Repository: Lightricks/LTX-2")

    upsampler_path = "/workspace/models/checkpoints/ltx-2-spatial-upscaler-x2-1.0.safetensors"

    if os.path.exists(upsampler_path):
        size = os.path.getsize(upsampler_path) / (1024**3)
        log_info(f"Spatial upsampler already exists ({size:.1f} GB)")
        log_success("Spatial upsampler found")
        return True

    try:
        log_step("Starting download...")
        start_time = time.time()

        downloaded_path = hf_hub_download(
            repo_id="Lightricks/LTX-2",
            filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
            local_dir="/workspace/models/checkpoints",
            local_dir_use_symlinks=False,
            resume_download=True,
            token=HF_TOKEN
        )

        download_time = time.time() - start_time
        size = os.path.getsize(upsampler_path) / (1024**3)

        log_success(f"Spatial upsampler downloaded successfully")
        log_info(f"Download time: {download_time:.1f} seconds")
        log_info(f"File size: {size:.1f} GB")
        log_info(f"Path: {downloaded_path}")
        return True
    except Exception as e:
        log_error(f"Error downloading spatial upsampler: {e}")
        log_info("\n📋 Troubleshooting:")
        log_info("1. Check your internet connection")
        log_info("2. Visit: https://huggingface.co/Lightricks/LTX-2")
        return False


def download_gemma_encoder():
    """Download Gemma-3-12B text encoder"""
    log_step("Downloading Gemma-3-12B text encoder...")
    log_info("Repository: google/gemma-3-12b-it-qat-q4_0-unquantized")

    gemma_dir = "/workspace/models/gemma"

    # Check if already downloaded (check for key files)
    required_files = [
        "config.json",
        "tokenizer.json"
    ]

    all_exist = all(os.path.exists(os.path.join(gemma_dir, f)) for f in required_files)

    if all_exist:
        size = get_directory_size(gemma_dir)
        log_info(f"Gemma text encoder already exists ({size:.1f} GB)")
        log_success("Gemma text encoder found")
        return True

    try:
        log_step("Starting download...")
        start_time = time.time()

        snapshot_download(
            repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
            local_dir=gemma_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Skip non-PyTorch formats
            token=HF_TOKEN
        )

        download_time = time.time() - start_time
        size = get_directory_size(gemma_dir)

        log_success(f"Gemma text encoder downloaded successfully")
        log_info(f"Download time: {download_time:.1f} seconds")
        log_info(f"Model size: {size:.1f} GB")
        return True
    except Exception as e:
        log_error(f"Error downloading Gemma text encoder: {e}")
        log_info("\n📋 Troubleshooting:")
        log_info("1. Check your internet connection")
        log_info("2. Visit: https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized")
        log_info("3. Ensure you have sufficient disk space (~10GB)")
        return False


def verify_model_structure():
    """Verify that all required model files are present"""
    log_step("Verifying model structure...")

    required_paths = [
        '/workspace/models/checkpoints/ltx-2-19b-distilled-fp8.safetensors',
        '/workspace/models/checkpoints/ltx-2-spatial-upscaler-x2-1.0.safetensors',
        '/workspace/models/gemma/config.json',
        '/workspace/models/gemma/tokenizer.json'
    ]

    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
        else:
            log_info(f"✓ Found: {path}")

    if missing_paths:
        log_error("Missing model files:")
        for path in missing_paths:
            log_error(f"   - {path}")
        return False
    else:
        log_success("All required model files found")
        return True


def check_disk_space():
    """Check available disk space"""
    log_step("Checking disk space...")

    # Estimate required space
    required_gb = 20  # Conservative estimate (12GB + 1.5GB + 5GB + buffer)

    try:
        stat = os.statvfs('/workspace')
        available_gb = (stat.f_frsize * stat.f_bavail) / (1024**3)

        log_info(f"Available: {available_gb:.1f} GB")
        log_info(f"Required: ~{required_gb} GB")

        if available_gb < required_gb:
            log_warning(f"Low disk space ({available_gb:.1f} GB available)")
            log_warning("Download may fail if insufficient space")
            return False
        else:
            log_success("Sufficient disk space available")
            return True
    except Exception as e:
        log_warning(f"Could not check disk space: {e}")
        return True  # Assume OK if we can't check


def main():
    """Main download function"""
    print("🚀 LTX-2 Model Downloader")
    print("=" * 50)

    # Check disk space
    if not check_disk_space():
        log_warning("Proceeding with download despite low disk space")

    # Create directories
    create_directories()

    # Download models
    success = True

    if not download_transformer_checkpoint():
        success = False

    if not download_spatial_upsampler():
        success = False

    if not download_gemma_encoder():
        success = False

    # Verify structure
    if success and not verify_model_structure():
        success = False

    # Final status
    print("\n" + "=" * 50)
    if success:
        log_success("All models downloaded successfully!")
        log_info("Ready to run the LTX-2 API server")

        # Show model sizes
        try:
            total_size = 0
            for root, dirs, files in os.walk('/workspace/models'):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)

            log_info(f"Total model size: {total_size / (1024**3):.1f} GB")

            # Show available disk space after download
            stat = os.statvfs('/workspace')
            available_gb = (stat.f_frsize * stat.f_bavail) / (1024**3)
            log_info(f"Remaining disk space: {available_gb:.1f} GB")

        except Exception as e:
            log_warning(f"Could not calculate total size: {e}")
    else:
        log_error("Model download failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
