from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image, ImageDraw, ImageFont


# =========================================================
# 1) Configuration (formerly config.py)
# =========================================================
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
VIDEOS_DIR = ARTIFACTS_DIR / "videos"
IMAGES_DIR = ARTIFACTS_DIR / "images"
TEMP_DIR = ARTIFACTS_DIR / "tmp"

SLOGAN_GENERATION_MODEL_ID = os.getenv("SLOGAN_GENERATION_MODEL_ID", "Qwen/Qwen3.5-2B")
SCRIPT_GENERATION_MODEL_ID = os.getenv("SCRIPT_GENERATION_MODEL_ID", "Qwen/Qwen3.5-2B")
SCENE_IMAGE_MODEL_ID = os.getenv("SCENE_IMAGE_MODEL_ID", "F16/z-image-turbo-sda")
VIDEO_MODEL_ID = os.getenv("VIDEO_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B")

HF_TOKEN = os.getenv("HF_TOKEN", "")
USE_HF_INFERENCE_API_FOR_VIDEO = os.getenv("USE_HF_INFERENCE_API_FOR_VIDEO", "true").lower() == "true"
FORCE_MOCK_MODE = os.getenv("FORCE_MOCK_MODE", "false").lower() == "true"

DEFAULT_VIDEO_DURATION_SECONDS = int(os.getenv("DEFAULT_VIDEO_DURATION_SECONDS", "4"))
DEFAULT_VIDEO_WIDTH = int(os.getenv("DEFAULT_VIDEO_WIDTH", "854"))
DEFAULT_VIDEO_HEIGHT = int(os.getenv("DEFAULT_VIDEO_HEIGHT", "480"))
DEFAULT_VIDEO_FPS = int(os.getenv("DEFAULT_VIDEO_FPS", "20"))
FINAL_SLOGAN_FRAME_SECONDS = float(os.getenv("FINAL_SLOGAN_FRAME_SECONDS", "1.5"))


def ensure_artifact_dirs() -> None:
    for path in (ARTIFACTS_DIR, VIDEOS_DIR, IMAGES_DIR, TEMP_DIR):
        path.mkdir(parents=True, exist_ok=True)


ensure_artifact_dirs()

st.set_page_config(
    page_title="AI Smart Marketing: Personalized Nike Video Advertisements",
    layout="wide",
    initial_sidebar_state="expanded",
)

secret_hf_token = st.secrets.get("HF_TOKEN", "")
if secret_hf_token:
    HF_TOKEN = secret_hf_token


# =========================================================
# 2) Interfaces and data contracts (formerly interfaces.py)
# =========================================================
@dataclass(slots=True)
class CustomerProfile:
    name: str
    age: int
    gender: str
    nationality: str
    language: str
    product_id: str
    additional_notes: str = ""


@dataclass(slots=True)
class ProductInfo:
    product_id: str
    name: str
    category: str
    gender: str
    image_path_or_url: str
    product_url: str
    attributes: dict[str, str] | None = None


@dataclass(slots=True)
class MarketingAssets:
    slogan: str
    headline: str
    script: str
    storyline: str = ""
    scene_image_paths: list[str] = field(default_factory=list)
    final_slogan_text: str = ""
    video_path: str | None = None
    pipeline_models: dict[str, str] = field(default_factory=dict)
    debug_metadata: dict[str, Any] | None = None


class SloganGenerator(Protocol):
    def generate(self, profile: CustomerProfile, product: ProductInfo) -> str:
        ...


class StorylineGenerator(Protocol):
    def generate(self, profile: CustomerProfile, product: ProductInfo, slogan: str) -> str:
        ...


class SceneGenerator(Protocol):
    def generate(
        self,
        profile: CustomerProfile,
        product: ProductInfo,
        storyline: str,
        image_count: int = 3,
    ) -> list[str]:
        ...


class VideoGenerator(Protocol):
    def generate(self, profile: CustomerProfile, product: ProductInfo, assets: MarketingAssets) -> str:
        ...


# =========================================================
# 3) Nike catalog (formerly nike_catalog.py)
# =========================================================
_CATALOG: list[dict[str, str]] = [
    {
        "product_id": "pegasus-41-m",
        "name": "Nike Air Zoom Pegasus 41",
        "category": "Running",
        "gender": "Men",
        "image_path_or_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=1200&q=80",
        "product_url": "https://www.nike.com.hk/man/shoe/list.htm?intpromo=PNTP",
    },
    {
        "product_id": "vomero-17-f",
        "name": "Nike ZoomX Vomero 17",
        "category": "Running",
        "gender": "Women",
        "image_path_or_url": "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?auto=format&fit=crop&w=1200&q=80",
        "product_url": "https://www.nike.com.hk/woman/shoe/list.htm?intpromo=PNTP",
    },
    {
        "product_id": "dunk-low-unisex",
        "name": "Nike Dunk Low",
        "category": "Lifestyle",
        "gender": "Unisex",
        "image_path_or_url": "https://images.unsplash.com/photo-1600185365483-26d7a4cc7519?auto=format&fit=crop&w=1200&q=80",
        "product_url": "https://www.nike.com.hk/man/shoe/list.htm?intpromo=PNTP",
    },
    {
        "product_id": "ja-2-basketball",
        "name": "Nike Ja 2",
        "category": "Basketball",
        "gender": "Unisex",
        "image_path_or_url": "https://images.unsplash.com/photo-1514989940723-e8e51635b782?auto=format&fit=crop&w=1200&q=80",
        "product_url": "https://www.nike.com.hk/man/shoe/list.htm?intpromo=PNTP",
    },
    {
        "product_id": "metcon-9-training",
        "name": "Nike Metcon 9",
        "category": "Training",
        "gender": "Unisex",
        "image_path_or_url": "https://images.unsplash.com/photo-1463100099107-aa0980c362e6?auto=format&fit=crop&w=1200&q=80",
        "product_url": "https://www.nike.com.hk/man/shoe/list.htm?intpromo=PNTP",
    },
]

_DEFAULT_ATTRIBUTES = {
    "Running": {"benefit": "responsive cushioning", "feel": "lightweight"},
    "Lifestyle": {"benefit": "all-day comfort", "feel": "street-ready"},
    "Basketball": {"benefit": "court grip", "feel": "explosive"},
    "Training": {"benefit": "stable support", "feel": "versatile"},
}


def _to_product(raw: dict[str, str]) -> ProductInfo:
    category = raw["category"]
    return ProductInfo(
        product_id=raw["product_id"],
        name=raw["name"],
        category=category,
        gender=raw["gender"],
        image_path_or_url=raw["image_path_or_url"],
        product_url=raw["product_url"],
        attributes=_DEFAULT_ATTRIBUTES.get(category, {}).copy(),
    )


@st.cache_data(show_spinner=False)
def list_products() -> list[ProductInfo]:
    return [_to_product(row) for row in _CATALOG]


def get_product_by_name(product_name: str) -> ProductInfo:
    for row in _CATALOG:
        if row["name"] == product_name:
            return _to_product(row)
    raise ValueError(f"Unknown product name: {product_name}")


# =========================================================
# 4) Media utilities (formerly media_utils.py)
# =========================================================
def sanitize_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value.strip().lower())
    return "-".join(part for part in safe.split("-") if part)[:80] or "asset"


def save_video_bytes(video_bytes: bytes, prefix: str = "nike-ad") -> str:
    ensure_artifact_dirs()
    digest = hashlib.sha1(video_bytes).hexdigest()[:12]
    output_path = VIDEOS_DIR / f"{sanitize_filename(prefix)}-{digest}.mp4"
    output_path.write_bytes(video_bytes)
    return str(output_path)


def _resize_and_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _load_title_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ["arialbd.ttf", "Arial Bold.ttf", "segoeuib.ttf", "HelveticaNeue-Bold.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _build_end_card_frames(final_slogan_text: str, width: int, height: int, fps: int) -> list[np.ndarray]:
    frame_count = max(1, int(round(FINAL_SLOGAN_FRAME_SECONDS * fps)))
    title_font = _load_title_font(42)
    label_font = _load_title_font(18)
    frames: list[np.ndarray] = []

    for idx in range(frame_count):
        progress = (idx + 1) / frame_count
        canvas = Image.new("RGB", (width, height), color=(8, 8, 8))
        overlay = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        accent_top = int(height * 0.18)
        accent_bottom = int(height * 0.82)
        draw.rectangle((width - 28, accent_top, width - 18, accent_bottom), fill=(255, 255, 255, 210))
        draw.rectangle((0, height - 72, width, height), fill=(18, 18, 18, 235))

        text_alpha = int(255 * min(1.0, progress * 1.35))
        text = final_slogan_text.strip() or "Move in your own way"
        bbox = draw.multiline_textbbox((0, 0), text, font=title_font, spacing=8)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = max(32, (width - text_width) // 2)
        text_y = max(48, (height - text_height) // 2 - 18)
        draw.multiline_text(
            (text_x, text_y),
            text,
            font=title_font,
            fill=(255, 255, 255, text_alpha),
            spacing=8,
            align="center",
        )
        draw.text((36, height - 52), "PERSONALIZED NIKE-STYLE AD", font=label_font, fill=(196, 196, 196, text_alpha))
        composed = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        frames.append(np.asarray(composed))

    return frames


def compose_final_ad_video(source_video_path: str, output_stem: str, final_slogan_text: str) -> str:
    ensure_artifact_dirs()
    width = DEFAULT_VIDEO_WIDTH
    height = DEFAULT_VIDEO_HEIGHT
    default_fps = DEFAULT_VIDEO_FPS
    output_path = VIDEOS_DIR / f"{sanitize_filename(output_stem)}.mp4"

    with VideoFileClip(source_video_path) as clip:
        fps = int(round(getattr(clip, "fps", 0) or default_fps))
        frames = [
            np.asarray(_resize_and_crop(Image.fromarray(frame).convert("RGB"), width, height))
            for frame in clip.iter_frames(fps=fps, dtype="uint8")
        ]

    frames.extend(_build_end_card_frames(final_slogan_text=final_slogan_text, width=width, height=height, fps=fps))

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(output_path), codec="libx264", audio=False, preset="ultrafast", logger=None)
    clip.close()
    return str(output_path)


def _placeholder_image(path: Path, title: str) -> None:
    canvas = Image.new("RGB", (1280, 720), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 580, 1280, 720), fill=(238, 29, 35))
    draw.text((60, 80), "SMART NIKE SHOE AD STUDIO", fill=(255, 255, 255), font=ImageFont.load_default())
    draw.text((60, 130), title[:70], fill=(230, 230, 230), font=ImageFont.load_default())
    canvas.save(path)


def ensure_local_image(image_path_or_url: str, name_hint: str = "product") -> str:
    ensure_artifact_dirs()
    if not image_path_or_url:
        target = IMAGES_DIR / f"{sanitize_filename(name_hint)}-placeholder.png"
        _placeholder_image(target, name_hint)
        return str(target)

    parsed = urlparse(image_path_or_url)
    if parsed.scheme in ("http", "https"):
        stem = sanitize_filename(name_hint)
        suffix = Path(parsed.path).suffix or ".jpg"
        target = IMAGES_DIR / f"{stem}{suffix}"
        if not target.exists():
            try:
                urlretrieve(image_path_or_url, target)
            except Exception:
                _placeholder_image(target, name_hint)
        return str(target)

    path = Path(image_path_or_url)
    if path.exists():
        return str(path)

    target = IMAGES_DIR / f"{sanitize_filename(name_hint)}-placeholder.png"
    if not target.exists():
        _placeholder_image(target, name_hint)
    return str(target)


def create_fallback_banner_video(
    image_path_or_url: str,
    slogan: str,
    headline: str,
    output_stem: str,
    final_slogan_text: str = "",
    duration_seconds: int | None = None,
) -> str:
    ensure_artifact_dirs()
    image_path = ensure_local_image(image_path_or_url, name_hint=output_stem)
    width = DEFAULT_VIDEO_WIDTH
    height = DEFAULT_VIDEO_HEIGHT
    fps = DEFAULT_VIDEO_FPS
    duration = duration_seconds or DEFAULT_VIDEO_DURATION_SECONDS
    total_frames = max(1, duration * fps)

    base = Image.open(image_path).convert("RGB")
    frames: list[np.ndarray] = []

    for idx in range(total_frames):
        t = idx / max(1, total_frames - 1)
        scale = 1.0 + 0.08 * t
        scaled_w = int(base.width * scale)
        scaled_h = int(base.height * scale)
        scaled = base.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

        left = max(0, (scaled_w - width) // 2)
        top = max(0, (scaled_h - height) // 2)
        frame = scaled.crop((left, top, left + width, top + height))

        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, height - 130, width, height), fill=(0, 0, 0, 150))

        slogan_text = slogan[:70] if slogan else "Move with confidence"
        headline_text = headline[:100] if headline else "Personalized Nike Shoe Story"
        draw.text((30, height - 110), slogan_text, fill=(255, 255, 255, 255), font=ImageFont.load_default())
        draw.text((30, height - 80), headline_text, fill=(225, 225, 225, 255), font=ImageFont.load_default())

        composed = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
        frames.append(np.asarray(composed))

    end_text = final_slogan_text or slogan
    frames.extend(_build_end_card_frames(final_slogan_text=end_text, width=width, height=height, fps=fps))

    clip = ImageSequenceClip(frames, fps=fps)
    output_path = VIDEOS_DIR / f"{sanitize_filename(output_stem)}.mp4"
    clip.write_videofile(str(output_path), codec="libx264", audio=False, preset="ultrafast", logger=None)
    clip.close()
    return str(output_path)


# =========================================================
# 5) Shared generation helpers
# =========================================================
def _deterministic_pick(options: list[str], key: str) -> str:
    if not options:
        return ""
    idx = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % len(options)
    return options[idx]


def _age_group(age: int) -> str:
    if age < 18:
        return "teenager"
    if age < 30:
        return "person in their 20s"
    if age < 40:
        return "person in their 30s"
    if age < 50:
        return "person in their 40s"
    return "mature adult"


def build_profile_summary(profile: CustomerProfile) -> str:
    notes = profile.additional_notes.strip() or "No additional notes"
    return (
        f"Name: {profile.name}; Age: {profile.age}; Gender: {profile.gender}; "
        f"Nationality: {profile.nationality or 'Not specified'}; Language: {profile.language}; "
        f"Product ID: {profile.product_id}; Notes: {notes}."
    )


def build_slogan_prompt(
    profile_summary: str,
    product: ProductInfo,
    product_attributes: dict[str, str],
    language: str = "English",
) -> str:
    attrs = ", ".join(f"{k}: {v}" for k, v in product_attributes.items()) or "No attributes provided"
    lang_instruction = "Write the slogan in English."
    if language == "Traditional Chinese":
        lang_instruction = "Write the slogan in Traditional Chinese only. Do not use Simplified Chinese."
    return (
        "You are a Nike marketing slogan expert. "
        "Create ONE short, inspiring, and motivating slogan (at most 10 words) to make the product attractive and persuade the customer to buy it. "
        "Do NOT include the customer's name in the generated text. "
        "Return ONLY the slogan text, no explanation, no quotation marks.\n\n"
        f"{lang_instruction}\n"
        f"Customer profile: {profile_summary}\n"
        f"Shoe: {product.name} - {product.category}\n"
        f"Product attributes: {attrs}\n"
        "Make it tailored, energetic, and highly attractive."
    )


def parse_generated_slogan(raw_output: str) -> str:
    raw_output = (raw_output or "").strip()
    if not raw_output:
        return "Move Beyond Limits"
    slogan = raw_output.splitlines()[0].strip().strip('"')
    words = slogan.split()
    if len(words) > 10:
        slogan = " ".join(words[:10])
    return slogan[:80] or "Move Beyond Limits"


def build_storyline_prompt(profile: CustomerProfile, product: ProductInfo, slogan: str) -> str:
    scene_styles = [
        "urban dawn training route with reflective streets",
        "sunset rooftop run with warm cinematic haze",
        "indoor court lights with dramatic shadows and speed",
        "city crosswalk rush-hour moment with crisp focus",
        "coastal boardwalk stride with golden-hour glow",
    ]
    mood_styles = ["confident and premium", "youthful and fearless", "focused and determined", "playful but elite", "bold and emotional"]
    variation_key = f"{profile.name}|{profile.age}|{profile.gender}|{profile.nationality}|{product.name}|{slogan}"
    scene_style = _deterministic_pick(scene_styles, variation_key)
    mood_style = _deterministic_pick(mood_styles, f"mood:{variation_key}")
    return (
        "You are creating a concise Nike-style ad storyline for text-to-image and text-to-video generation. "
        "Return ONLY one sentence. Keep it concrete, cinematic, and realistic.\n\n"
        f"Required realistic character profile: {profile.age}-year-old {profile.gender} from {profile.nationality}.\n"
        "The character must remain visually consistent, explicitly showing their face and body.\n"
        f"Product: {product.name} ({product.category}).\n"
        f"Approved slogan: {slogan}.\n"
        f"Primary scene direction: {scene_style}.\n"
        f"Emotional tone: {mood_style}.\n"
        "Style hint: dynamic slow motion, bold marketing style, high realism."
    )


def _fallback_slogan(profile: CustomerProfile, product: ProductInfo, product_attributes: dict[str, str]) -> str:
    benefit = (
        product_attributes.get("benefit")
        or product_attributes.get("performance")
        or product_attributes.get("feel")
        or "everyday performance"
    )
    key = f"{profile.name}|{profile.age}|{profile.gender}|{profile.nationality}|{product.name}|{profile.language}|{benefit}"

    if profile.language == "Traditional Chinese":
        templates = [
            "{nationality}力量，{benefit}每一步",
            "為{age}世代而生，{benefit}現在啟動",
            "{gender}風格全開，{product_tail}帶你突破",
            "穿上{product_tail}，把日常走成高光時刻",
            "從{nationality}到全場，{benefit}一路領先",
        ]
        return _deterministic_pick(templates, key).format(
            nationality=profile.nationality,
            age=profile.age,
            gender=profile.gender,
            product_tail=product.name.split()[-1],
            benefit=benefit,
        )

    templates = [
        "{nationality} grit, {benefit} in every stride",
        "Built for {age}, ready for every challenge",
        "{gender} energy, next-level {product_tail} momentum",
        "From daily hustle to spotlight runs, stay unstoppable",
        "Own your pace with {benefit} confidence",
    ]
    return _deterministic_pick(templates, key).format(
        nationality=profile.nationality,
        age=profile.age,
        gender=profile.gender,
        product_tail=product.name.split()[-1],
        benefit=benefit,
    )


def _fallback_storyline(profile: CustomerProfile, product: ProductInfo) -> str:
    templates = [
        "{age}-year-old {gender} from {nationality}, full face and body visible, bursts from a metro staircase in {product} as rain mist catches neon light, cinematic Nike-style momentum",
        "{age}-year-old {gender} from {nationality}, clear face and full body, drives a fast break on an indoor court wearing {product}, dramatic rim light and bold ad realism",
        "{age}-year-old {gender} from {nationality}, face and body in focus, powers through a sunrise bridge run in {product}, wind-swept motion and premium campaign energy",
        "{age}-year-old {gender} from {nationality}, full body and expressive face visible, transitions from office streetwear to performance stride in {product}, dynamic lifestyle-to-sport cinematic arc",
        "{age}-year-old {gender} from {nationality}, face and body clearly shown, leads a twilight rooftop training session in {product}, high-detail realism and aspirational brand mood",
    ]
    key = f"story:{profile.name}|{profile.age}|{profile.gender}|{profile.nationality}|{product.name}"
    return _deterministic_pick(templates, key).format(
        age=profile.age,
        gender=profile.gender,
        nationality=profile.nationality,
        product=product.name,
    )


def _create_scene_placeholder(path: Path, prompt: str, index: int) -> None:
    canvas = Image.new("RGB", (1024, 576), color=(16, 26, 39))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 500, 1024, 576), fill=(232, 63, 42))
    draw.text((24, 26), f"SCENE {index}", fill=(245, 245, 245), font=ImageFont.load_default())
    draw.text((24, 58), prompt[:130], fill=(220, 220, 220), font=ImageFont.load_default())
    draw.text((24, 82), prompt[130:260], fill=(220, 220, 220), font=ImageFont.load_default())
    draw.text((24, 106), "Fallback image (inference unavailable)", fill=(196, 196, 196), font=ImageFont.load_default())
    canvas.save(path)


def _save_generated_image(image_obj: Any, output_path: Path) -> str:
    if isinstance(image_obj, Image.Image):
        image_obj.save(output_path)
        return str(output_path)
    if isinstance(image_obj, bytes):
        output_path.write_bytes(image_obj)
        return str(output_path)
    if hasattr(image_obj, "save"):
        cast(Any, image_obj).save(output_path)
        return str(output_path)
    if isinstance(image_obj, dict) and "image" in image_obj:
        image_data = image_obj["image"]
        if isinstance(image_data, bytes):
            output_path.write_bytes(image_data)
            return str(output_path)
    raise ValueError("Unsupported generated image format")


# =========================================================
# 6) Hugging Face implementations (formerly hf_pipelines.py)
# =========================================================
@st.cache_resource(show_spinner=False)
def _load_hf_inference_client(model_id: str, token: str):
    return cast(Any, InferenceClient(model=model_id, token=token))


def _generate_text_via_hf_api(
    model_id: str,
    token: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.8,
) -> str:
    if not token:
        raise ValueError("HF token is required for API-only text generation")

    client = _load_hf_inference_client(model_id, token)
    if hasattr(client, "text_generation"):
        try:
            result = cast(Any, client).text_generation(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                return_full_text=False,
            )
            return str(result or "").strip()
        except StopIteration:
            pass

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "return_full_text": False,
        },
    }
    raw = cast(Any, client).post(json=payload)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore").strip()
    return str(raw or "").strip()


class HFSloganGenerator(SloganGenerator):
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or SLOGAN_GENERATION_MODEL_ID
        self.token = HF_TOKEN

    def generate(self, profile: CustomerProfile, product: ProductInfo) -> str:
        prompt = build_slogan_prompt(
            profile_summary=build_profile_summary(profile),
            product=product,
            product_attributes=product.attributes or {},
            language=profile.language,
        )
        try:
            raw = _generate_text_via_hf_api(
                model_id=self.model_id,
                token=self.token,
                prompt=prompt,
                max_new_tokens=220,
                temperature=0.8,
            )
        except Exception:
            raw = ""

        slogan = parse_generated_slogan(raw)
        if not raw.strip() or slogan == "Move Beyond Limits":
            slogan = _fallback_slogan(profile, product, product.attributes or {})
        return f"{slogan}, {profile.name}"


class HFStorylineGenerator(StorylineGenerator):
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or SCRIPT_GENERATION_MODEL_ID
        self.token = HF_TOKEN

    def generate(self, profile: CustomerProfile, product: ProductInfo, slogan: str) -> str:
        prompt = build_storyline_prompt(profile, product, slogan)
        try:
            raw = _generate_text_via_hf_api(
                model_id=self.model_id,
                token=self.token,
                prompt=prompt,
                max_new_tokens=110,
                temperature=0.95,
            )
            storyline = raw.strip().splitlines()[0].strip() if raw else ""
        except Exception:
            storyline = ""

        if not storyline:
            storyline = _fallback_storyline(profile, product)
        return storyline[:260]


class HFSceneGenerator(SceneGenerator):
    def __init__(self, model_id: str | None = None, token: str | None = None) -> None:
        self.model_id = model_id or SCENE_IMAGE_MODEL_ID
        self.token = token or HF_TOKEN

    def _scene_prompt(self, profile: CustomerProfile, product: ProductInfo, storyline: str, index: int) -> str:
        camera_map = {
            1: "wide shot, full body visible, clear face, establishing scene",
            2: "mid shot, visible face and body, strong motion blur, shoe emphasis",
            3: "close-up hero shot, detailed face and body, cinematic rim light",
        }
        shot = camera_map.get(index, "cinematic commercial shot, full body and clear face")
        return (
            f"Realistic {profile.age}-year-old {profile.gender} from {profile.nationality} showing clear face and full body portrait, "
            f"wearing {product.name}, {storyline}. {shot}. "
            "photorealistic, high detail, natural skin texture, advertising photography, expressive face"
        )

    def _generate_single_image(self, prompt: str, target: Path, index: int) -> str:
        if self.token:
            try:
                client = _load_hf_inference_client(self.model_id, self.token)
                if hasattr(client, "text_to_image"):
                    result = cast(Any, client).text_to_image(prompt=prompt)
                    return _save_generated_image(result, target)
                if hasattr(client, "post"):
                    result = cast(Any, client).post(json={"inputs": prompt})
                    if isinstance(result, bytes):
                        return _save_generated_image(result, target)
            except Exception:
                pass

        _create_scene_placeholder(target, prompt, index)
        return str(target)

    def generate(
        self,
        profile: CustomerProfile,
        product: ProductInfo,
        storyline: str,
        image_count: int = 3,
    ) -> list[str]:
        ensure_artifact_dirs()
        safe_stem = sanitize_filename(f"{profile.name}-{product.product_id}")
        image_paths: list[str] = []
        for idx in range(1, max(1, image_count) + 1):
            prompt = self._scene_prompt(profile, product, storyline, idx)
            target = IMAGES_DIR / f"{safe_stem}-scene-{idx}.png"
            image_paths.append(self._generate_single_image(prompt, target, idx))
        return image_paths


class HFVideoGenerator(VideoGenerator):
    def __init__(self, model_id: str | None = None, token: str | None = None) -> None:
        self.model_id = model_id or VIDEO_MODEL_ID
        self.token = token or HF_TOKEN

    @staticmethod
    def _build_video_prompt(profile: CustomerProfile, product: ProductInfo, assets: MarketingAssets) -> str:
        city = profile.nationality or "a modern city"
        age_group = _age_group(profile.age)
        scene_text = assets.storyline or assets.script
        return (
            f"Use the reference image to preserve the same realistic character: {age_group} {profile.gender} from {city}. "
            f"Storyline: {scene_text}. "
            f"The character wears {product.name}, cinematic premium sportswear ad, smooth motion, high realism, 480p."
        )

    def _try_inference_api(self, prompt: str, reference_image_path: str | None) -> bytes | None:
        if not USE_HF_INFERENCE_API_FOR_VIDEO or not self.token:
            return None

        try:
            client = _load_hf_inference_client(self.model_id, self.token)
            if reference_image_path and hasattr(client, "image_to_video"):
                with open(reference_image_path, "rb") as image_file:
                    image_bytes = image_file.read()
                result = cast(Any, client).image_to_video(image=image_bytes, prompt=prompt)
                if isinstance(result, bytes):
                    return result
            if hasattr(client, "text_to_video"):
                result = cast(Any, client).text_to_video(prompt)
                if isinstance(result, bytes):
                    return result
            return None
        except Exception:
            return None

    def generate(self, profile: CustomerProfile, product: ProductInfo, assets: MarketingAssets) -> str:
        prompt = self._build_video_prompt(profile, product, assets)
        output_stem = sanitize_filename(f"{profile.name}-{product.product_id}-ad")
        final_slogan_text = assets.final_slogan_text or assets.slogan
        reference_image = assets.scene_image_paths[0] if assets.scene_image_paths else None

        video_bytes = self._try_inference_api(prompt, reference_image)
        if video_bytes:
            raw_video_path = save_video_bytes(video_bytes, prefix=f"{output_stem}-raw")
            return compose_final_ad_video(
                source_video_path=raw_video_path,
                output_stem=output_stem,
                final_slogan_text=final_slogan_text,
            )

        return create_fallback_banner_video(
            image_path_or_url=reference_image or product.image_path_or_url,
            slogan=assets.slogan,
            headline=assets.headline,
            output_stem=output_stem,
            final_slogan_text=final_slogan_text,
        )


# =========================================================
# 7) Mock implementations (formerly mock_implementations.py)
# =========================================================
class MockSloganGenerator(SloganGenerator):
    def generate(self, profile: CustomerProfile, product: ProductInfo) -> str:
        attrs = product.attributes or {}
        benefit = attrs.get("benefit") or attrs.get("performance") or "all-day comfort"
        seed = f"{profile.name}|{profile.age}|{profile.gender}|{profile.nationality}|{profile.language}|{product.name}|{benefit}"

        if profile.language == "Traditional Chinese":
            templates = [
                "{nationality}節奏，由{product_tail}開跑",
                "{age}歲也能火力全開，{benefit}一路相隨",
                "{gender}風格上場，步步都是主場",
                "穿上{product_tail}，把今天走成高光",
                "從第一步到最後一步，{benefit}不妥協",
            ]
            slogan = _deterministic_pick(templates, seed).format(
                nationality=profile.nationality,
                age=profile.age,
                gender=profile.gender,
                product_tail=product.name.split()[-1],
                benefit=benefit,
            )
            return f"{slogan}, {profile.name}"

        templates = [
            "{nationality} pace, unstoppable in {product_tail}",
            "Built for {age}, made to move fearlessly",
            "{gender} energy meets {benefit} comfort",
            "Turn everyday miles into your highlight reel",
            "Own every step with bold {product_tail} momentum",
        ]
        slogan = _deterministic_pick(templates, seed).format(
            nationality=profile.nationality,
            age=profile.age,
            gender=profile.gender,
            product_tail=product.name.split()[-1],
            benefit=benefit,
        )
        return f"{slogan}, {profile.name}"


class MockStorylineGenerator(StorylineGenerator):
    def generate(self, profile: CustomerProfile, product: ProductInfo, slogan: str) -> str:
        templates = [
            "{age}-year-old {gender} from {nationality}, face and full body visible, sprints through a dawn city bridge in {product}, dynamic slow motion and premium ad realism",
            "{age}-year-old {gender} from {nationality}, clear face and body shown, commands an indoor court wearing {product}, explosive movement, cinematic sports-commercial look",
            "{age}-year-old {gender} from {nationality}, visible face and body, transitions from commute to training in {product}, lifestyle-to-performance narrative with bold lighting",
            "{age}-year-old {gender} from {nationality}, full body and expressive face in frame, runs a rooftop interval session in {product}, golden-hour atmosphere and high-detail realism",
            "{age}-year-old {gender} from {nationality}, face and body clearly visible, powers through rainy neon streets in {product}, energetic momentum and modern brand storytelling",
        ]
        seed = f"story:{profile.name}|{profile.age}|{profile.gender}|{profile.nationality}|{product.name}|{slogan}"
        template = _deterministic_pick(templates, seed)
        return template.format(
            age=profile.age,
            gender=profile.gender,
            nationality=profile.nationality,
            product=product.name,
        )


class MockSceneGenerator(SceneGenerator):
    @staticmethod
    def _write_scene(path: Path, text: str, idx: int) -> None:
        canvas = Image.new("RGB", (1024, 576), color=(18, 30, 44))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 502, 1024, 576), fill=(236, 72, 45))
        draw.text((26, 28), f"MOCK SCENE {idx}", fill=(245, 245, 245), font=ImageFont.load_default())
        draw.text((26, 62), text[:140], fill=(225, 225, 225), font=ImageFont.load_default())
        draw.text((26, 86), text[140:280], fill=(225, 225, 225), font=ImageFont.load_default())
        canvas.save(path)

    def generate(
        self,
        profile: CustomerProfile,
        product: ProductInfo,
        storyline: str,
        image_count: int = 3,
    ) -> list[str]:
        ensure_artifact_dirs()
        stem = sanitize_filename(f"mock-{profile.name}-{product.product_id}")
        images: list[str] = []
        for idx in range(1, max(1, image_count) + 1):
            path = IMAGES_DIR / f"{stem}-scene-{idx}.png"
            self._write_scene(path, storyline, idx)
            images.append(str(path))
        return images


class MockVideoGenerator(VideoGenerator):
    def generate(self, profile: CustomerProfile, product: ProductInfo, assets: MarketingAssets) -> str:
        stem = sanitize_filename(f"mock-{profile.name}-{product.product_id}")
        return create_fallback_banner_video(
            image_path_or_url=assets.scene_image_paths[0] if assets.scene_image_paths else product.image_path_or_url,
            slogan=assets.slogan,
            headline=assets.headline,
            output_stem=stem,
            final_slogan_text=assets.final_slogan_text or assets.slogan,
        )


# =========================================================
# 8) App orchestration and UI
# =========================================================
PIPELINE_MODELS = {
    1: SLOGAN_GENERATION_MODEL_ID,
    2: SCRIPT_GENERATION_MODEL_ID,
    3: SCENE_IMAGE_MODEL_ID,
    4: VIDEO_MODEL_ID,
}

PIPELINE_NAMES = {
    1: "Personalized Slogan Generation (Text Generation)",
    2: "Personalized Storyline Generation (Text Generation)",
    3: "Storyline Scene Generation (Text-to-Image Generation)",
    4: "Personalized Marketing Video Generation (Image+Text-to-Video Generation)",
}


def init_state() -> None:
    st.session_state.setdefault("assets", None)
    st.session_state.setdefault("video_path", None)
    st.session_state.setdefault("scene_images", [])


def validate_name(name: str) -> bool:
    return bool(name and name.strip())


def get_backends() -> tuple[SloganGenerator, StorylineGenerator, SceneGenerator, VideoGenerator, bool]:
    use_mock = FORCE_MOCK_MODE or not HF_TOKEN
    if use_mock:
        return (MockSloganGenerator(), MockStorylineGenerator(), MockSceneGenerator(), MockVideoGenerator(), True)
    return (
        HFSloganGenerator(),
        HFStorylineGenerator(),
        HFSceneGenerator(),
        HFVideoGenerator(model_id=VIDEO_MODEL_ID),
        False,
    )


def render_css() -> None:
    st.markdown(
        """
<style>
:root {
    --owid-blue: #002147;
    --owid-blue-2: #083a72;
    --owid-red: #d73a31;
    --owid-bg: #f2f4f7;
    --owid-text: #223145;
    --owid-border: #d8dee7;
    --owid-top-border: #0e2c50;
    --owid-card-bg: #ffffff;
    --owid-table-border: #ecf0f5;
    --owid-heading: #163555;
    --owid-note-bg: #fff7f4;
    --owid-sidebar-bg: #f6f8fb;
    --owid-sidebar-border: #dbe3ed;
}

.stApp { background: var(--owid-bg); color: var(--owid-text); }
[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding-top: 5rem; padding-bottom: 1.2rem; }

.top-header {
    border: 1px solid var(--owid-top-border);
    background: linear-gradient(90deg, var(--owid-blue), var(--owid-blue-2));
    border-radius: 8px;
    color: #f8fbff;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}
.top-header h1 { font-family: Georgia, "Times New Roman", serif; font-size: 2.1rem; margin: 0; font-weight: 600; }
.top-header p { margin-top: 0.4rem; color: #ffffff; opacity: 0.92; font-size: 0.95rem; }

.pipeline-grid, .result-card {
    border: 1px solid var(--owid-border);
    border-radius: 8px;
    background: var(--owid-card-bg);
    margin-bottom: 1rem;
    padding: 0.8rem 0.9rem;
}

.pipeline-grid table { width: 100%; border-collapse: collapse; }
.pipeline-grid th, .pipeline-grid td {
    border-bottom: 1px solid var(--owid-table-border);
    padding: 0.55rem 0.45rem;
    font-size: 0.9rem;
}
.pipeline-grid th { text-align: left; color: var(--owid-heading); font-weight: 700; }
.result-card h3 { margin: 0 0 0.5rem 0; color: var(--owid-heading); font-size: 1.06rem; }

.full-video [data-testid="stVideo"] { width: 100% !important; }
.full-video [data-testid="stVideo"] > div { width: 100% !important; }
.full-video [data-testid="stVideo"] video, .full-video [data-testid="stVideo"] iframe {
    width: 100% !important;
    height: auto !important;
    display: block;
}

[data-testid="stSidebar"] {
    background: var(--owid-sidebar-bg);
    border-right: 1px solid var(--owid-sidebar-border);
}

@media (max-width: 900px) {
    .top-header h1 { font-size: 1.6rem; }
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
<div class="top-header">
    <h1>AI Smart Marketing: Personalized Nike Video Advertisements</h1>
    <p>This application generates personalized Nike campaign assets by chaining 4 AI stages: slogan, storyline, scene images, and final short video.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_pipeline_table() -> None:
    st.markdown(
        f"""
<div class="pipeline-grid">
  <table>
    <thead>
      <tr><th>Pipeline</th><th>Task Type</th><th>Model</th></tr>
    </thead>
    <tbody>
      <tr><td>{PIPELINE_NAMES[1]}</td><td>Text Generation</td><td>{PIPELINE_MODELS[1]}</td></tr>
      <tr><td>{PIPELINE_NAMES[2]}</td><td>Text Generation</td><td>{PIPELINE_MODELS[2]}</td></tr>
      <tr><td>{PIPELINE_NAMES[3]}</td><td>Text-to-Image Generation</td><td>{PIPELINE_MODELS[3]}</td></tr>
      <tr><td>{PIPELINE_NAMES[4]}</td><td>Image+Text-to-Video Generation</td><td>{PIPELINE_MODELS[4]}</td></tr>
    </tbody>
  </table>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar(products: list[ProductInfo]) -> tuple[str, int, str, str, str, str, bool]:
    product_names = [item.name for item in products]
    with st.sidebar:
        st.header("Nike Online Store Customer Profile")
        name = st.text_input("Name", value="")
        age = int(st.number_input("Age", min_value=10, max_value=90, value=25, step=1))
        gender = cast(str, st.selectbox("Gender", ["Male", "Female"]))
        nationality = cast(str, st.selectbox("Nationality", ["Chinese", "USA"]))
        language = cast(str, st.selectbox("Language", ["English", "Traditional Chinese"]))
        selected_product_name = cast(str, st.selectbox("Product", options=product_names))

        if FORCE_MOCK_MODE or not HF_TOKEN:
            st.warning("HF token missing or mock mode forced: app runs in deterministic mock mode.")

        generate_clicked = st.button("Generate Assets", type="primary", use_container_width=True)

    return name, age, gender, nationality, language, selected_product_name, generate_clicked


def generate_assets_flow(
    customer: CustomerProfile,
    selected_product: ProductInfo,
    language: str,
    progress: Any,
    status_log: Any,
    slogan_slot: Any,
    storyline_slot: Any,
    scene_slot: Any,
) -> tuple[MarketingAssets, list[str]]:
    slogan_generator, storyline_generator, scene_generator, video_generator, use_mock = get_backends()

    logs: list[str] = []
    with st.status("Running pipelines...", expanded=True) as status:
        logs.append(f"[1/4] {PIPELINE_NAMES[1]}")
        status_log.code("\n".join(logs), language="text")

        slogan = slogan_generator.generate(customer, selected_product)
        progress.progress(25, text="Pipeline 1/4 complete")
        slogan_slot.markdown(
            f"""
<div class="result-card">
  <h3>Pipeline 1 Result: Personalized Slogan</h3>
  <p><b>Model:</b> {PIPELINE_MODELS[1]}</p>
  <p><b>Language:</b> {language}</p>
  <p style="font-size:1.06rem; margin-top:0.4rem;">{slogan}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        logs.append(f"[2/4] {PIPELINE_NAMES[2]}")
        status_log.code("\n".join(logs), language="text")
        storyline = storyline_generator.generate(customer, selected_product, slogan)
        progress.progress(50, text="Pipeline 2/4 complete")
        storyline_slot.markdown(
            f"""
<div class="result-card">
  <h3>Pipeline 2 Result: Personalized Storyline</h3>
  <p><b>Model:</b> {PIPELINE_MODELS[2]}</p>
  <p>{storyline}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        logs.append(f"[3/4] {PIPELINE_NAMES[3]} (3 images)")
        status_log.code("\n".join(logs), language="text")
        scene_images = scene_generator.generate(customer, selected_product, storyline, image_count=3)
        progress.progress(75, text="Pipeline 3/4 complete")
        with scene_slot.container():
            st.markdown(
                f"""
<div class="result-card">
  <h3>Pipeline 3 Result: Generated Scene Images</h3>
  <p><b>Model:</b> {PIPELINE_MODELS[3]}</p>
</div>
""",
                unsafe_allow_html=True,
            )
            cols = st.columns(3)
            for idx, image_path in enumerate(scene_images):
                cols[idx % 3].image(image_path, caption=f"Scene {idx + 1}", use_container_width=True)

        logs.append(f"[4/4] {PIPELINE_NAMES[4]}")
        status_log.code("\n".join(logs), language="text")
        assets = MarketingAssets(
            slogan=slogan,
            headline=f"{selected_product.name} personalized campaign",
            script=storyline,
            storyline=storyline,
            scene_image_paths=scene_images,
            final_slogan_text=slogan,
            pipeline_models={
                "pipeline_1": PIPELINE_MODELS[1],
                "pipeline_2": PIPELINE_MODELS[2],
                "pipeline_3": PIPELINE_MODELS[3],
                "pipeline_4": PIPELINE_MODELS[4],
            },
            debug_metadata={
                "runtime_mode": "mock" if use_mock else "huggingface",
                "customer": asdict(customer),
            },
        )
        assets.video_path = video_generator.generate(customer, selected_product, assets)

        progress.progress(100, text="All pipelines complete")
        logs.append("[done] All generated assets are ready")
        status_log.code("\n".join(logs), language="text")
        status.update(label="Generation complete", state="complete")

    return assets, scene_images


def render_saved_results(
    assets: MarketingAssets,
    slogan_slot: Any,
    storyline_slot: Any,
    scene_slot: Any,
    video_slot: Any,
) -> None:
    slogan_slot.markdown(
        f"""
<div class="result-card">
  <h3>Pipeline 1 Result: Personalized Slogan</h3>
  <p><b>Model:</b> {assets.pipeline_models.get('pipeline_1', PIPELINE_MODELS[1])}</p>
  <p style="font-size:1.06rem; margin-top:0.4rem;">{assets.slogan}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    storyline_slot.markdown(
        f"""
<div class="result-card">
  <h3>Pipeline 2 Result: Personalized Storyline</h3>
  <p><b>Model:</b> {assets.pipeline_models.get('pipeline_2', PIPELINE_MODELS[2])}</p>
  <p>{assets.storyline or assets.script}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    with scene_slot.container():
        st.markdown(
            f"""
<div class="result-card">
  <h3>Pipeline 3 Result: Generated Scene Images</h3>
  <p><b>Model:</b> {assets.pipeline_models.get('pipeline_3', PIPELINE_MODELS[3])}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for idx, image_path in enumerate(assets.scene_image_paths):
            cols[idx % 3].image(image_path, caption=f"Scene {idx + 1}", use_container_width=True)

    with video_slot.container():
        st.markdown(
            f"""
<div class="result-card full-video">
  <h3>Pipeline 4 Result: Personalized Marketing Video</h3>
  <p><b>Model:</b> {assets.pipeline_models.get('pipeline_4', PIPELINE_MODELS[4])}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if assets.video_path:
            st.video(assets.video_path)


def main() -> None:
    init_state()
    render_css()
    render_header()
    render_pipeline_table()

    products = list_products()
    name, age, gender, nationality, language, selected_product_name, generate_clicked = render_sidebar(products)

    progress = st.progress(0, text="Waiting to start generation")
    status_log = st.empty()
    slogan_slot = st.empty()
    storyline_slot = st.empty()
    scene_slot = st.empty()
    video_slot = st.empty()

    if generate_clicked:
        if not validate_name(name):
            st.error("Name is required. Please enter a non-empty name.")
        else:
            selected_product = get_product_by_name(selected_product_name)
            customer = CustomerProfile(
                name=name.strip(),
                age=age,
                gender=gender,
                nationality=nationality,
                language=language,
                product_id=selected_product.product_id,
            )

            assets, scene_images = generate_assets_flow(
                customer=customer,
                selected_product=selected_product,
                language=language,
                progress=progress,
                status_log=status_log,
                slogan_slot=slogan_slot,
                storyline_slot=storyline_slot,
                scene_slot=scene_slot,
            )

            st.session_state["assets"] = assets
            st.session_state["video_path"] = assets.video_path
            st.session_state["scene_images"] = scene_images

            with video_slot.container():
                st.markdown(
                    f"""
<div class="result-card full-video">
  <h3>Pipeline 4 Result: Personalized Marketing Video</h3>
  <p><b>Model:</b> {PIPELINE_MODELS[4]}</p>
</div>
""",
                    unsafe_allow_html=True,
                )
                if assets.video_path:
                    st.video(assets.video_path)

    assets = cast(MarketingAssets | None, st.session_state.get("assets"))
    if assets and not generate_clicked:
        render_saved_results(
            assets=assets,
            slogan_slot=slogan_slot,
            storyline_slot=storyline_slot,
            scene_slot=scene_slot,
            video_slot=video_slot,
        )


if __name__ == "__main__":
    main()
