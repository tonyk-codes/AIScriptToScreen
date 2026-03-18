"""Configuration for Smart Nike Shoe Ad Studio."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
VIDEOS_DIR = ARTIFACTS_DIR / "videos"
IMAGES_DIR = ARTIFACTS_DIR / "images"
TEMP_DIR = ARTIFACTS_DIR / "tmp"

PROFILE_EMBEDDING_MODEL_ID = os.getenv(
    "PROFILE_EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
)
PRODUCT_CLIP_MODEL_ID = os.getenv("PRODUCT_CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
SLOGAN_GENERATION_MODEL_ID = os.getenv("SLOGAN_GENERATION_MODEL_ID", "Qwen/Qwen3.5-2B")
SCRIPT_GENERATION_MODEL_ID = os.getenv("SCRIPT_GENERATION_MODEL_ID", "Qwen/Qwen3.5-2B")
SCENE_IMAGE_MODEL_ID = os.getenv("SCENE_IMAGE_MODEL_ID", "F16/z-image-turbo-sda")
VIDEO_MODEL_ID = os.getenv("VIDEO_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B")

HF_TOKEN = os.getenv("HF_TOKEN", "")

USE_HF_INFERENCE_API_FOR_VIDEO = os.getenv("USE_HF_INFERENCE_API_FOR_VIDEO", "true").lower() == "true"
USE_LOCAL_DIFFUSERS_VIDEO = os.getenv("USE_LOCAL_DIFFUSERS_VIDEO", "false").lower() == "true"
FORCE_MOCK_MODE = os.getenv("FORCE_MOCK_MODE", "false").lower() == "true"

DEFAULT_VIDEO_DURATION_SECONDS = int(os.getenv("DEFAULT_VIDEO_DURATION_SECONDS", "4"))
DEFAULT_VIDEO_WIDTH = int(os.getenv("DEFAULT_VIDEO_WIDTH", "854"))
DEFAULT_VIDEO_HEIGHT = int(os.getenv("DEFAULT_VIDEO_HEIGHT", "480"))
DEFAULT_VIDEO_FPS = int(os.getenv("DEFAULT_VIDEO_FPS", "20"))
FINAL_SLOGAN_FRAME_SECONDS = float(os.getenv("FINAL_SLOGAN_FRAME_SECONDS", "1.5"))


def ensure_artifact_dirs() -> None:
    """Create all required artifact directories."""
    for path in (ARTIFACTS_DIR, VIDEOS_DIR, IMAGES_DIR, TEMP_DIR):
        path.mkdir(parents=True, exist_ok=True)
