import os
import traceback
from pathlib import Path
from dataclasses import dataclass

import streamlit as st
import fal_client as fal
from dotenv import load_dotenv
from transformers import pipeline, BitsAndBytesConfig
from huggingface_hub import InferenceClient
import torch

# =========================================================
# 1) Configuration & Setup
# =========================================================
# Load environment variables (e.g., HF_TOKEN)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
VIDEOS_DIR = ARTIFACTS_DIR / "videos"
IMAGES_DIR = ARTIFACTS_DIR / "images"
TMP_DIR = ARTIFACTS_DIR / "tmp"
HF_MODEL_CACHE_DIR = TMP_DIR / "hf-model-cache"
ASSETS_DIR = BASE_DIR / "assets"
ICON_DIR = ASSETS_DIR / "icon"

# Ensure directories exist for saved media to prevent write errors
for path in (VIDEOS_DIR, IMAGES_DIR, TMP_DIR, HF_MODEL_CACHE_DIR):
    path.mkdir(parents=True, exist_ok=True)


def _read_secret_from_streamlit_secrets_file(secret_key: str) -> str:
    """Read a single secret from .streamlit/secrets.toml without extra dependencies."""
    secrets_path = BASE_DIR / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return ""

    try:
        for raw_line in secrets_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == secret_key:
                return value.strip().strip('"').strip("'")
    except Exception as e:
        print(f"Failed reading .streamlit/secrets.toml: {type(e).__name__}: {e}")

    return ""

# Application state and fixed model configuration.
HF_TOKEN = ""
try:
    # Prefer Streamlit secrets when running via `streamlit run`.
    HF_TOKEN = str(st.secrets.get("HF_TOKEN", "")).strip()
except Exception:
    HF_TOKEN = ""

# Fallback: read .streamlit/secrets.toml directly.
if not HF_TOKEN:
    HF_TOKEN = _read_secret_from_streamlit_secrets_file("HF_TOKEN")

# Fallback to .env / shell environment for local runs.
if not HF_TOKEN:
    HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Keep token available to libraries that read from process env.
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

FAL_KEY = ""
try:
    FAL_KEY = str(st.secrets.get("FAL_KEY", "")).strip()
except Exception:
    FAL_KEY = ""

if not FAL_KEY:
    FAL_KEY = _read_secret_from_streamlit_secrets_file("FAL_KEY")

if FAL_KEY:
    os.environ["FAL_KEY"] = FAL_KEY

SLOGAN_MODEL = "erichflam-hkust/Qwen2.5-VL-7B-Instruct-NIKE-Finetuned"
SCRIPT_MODEL = "zai-org/GLM-4.7-Flash:novita"
VIDEO_MODEL = "fal-ai/ltx-2.3/image-to-video/fast"

PIPELINE1_LOAD_ERROR = ""
PIPELINE1_LAST_ERROR = ""
PIPELINE1_BACKEND = ""
PIPELINE1_API_ERROR = ""
PIPELINE1_MODEL_USED = ""
PIPELINE1_INITIALIZED = False

pipeline1_pipe = None
pipeline1_processor = None
pipeline1_model = None


def _set_pipeline1_error(message: str):
    global PIPELINE1_LAST_ERROR
    PIPELINE1_LAST_ERROR = message


def _set_pipeline1_load_error(message: str):
    global PIPELINE1_LOAD_ERROR
    PIPELINE1_LOAD_ERROR = message


def _set_pipeline1_backend(name: str):
    global PIPELINE1_BACKEND
    PIPELINE1_BACKEND = name


def _set_pipeline1_api_error(message: str):
    global PIPELINE1_API_ERROR
    PIPELINE1_API_ERROR = message


def _set_pipeline1_model_used(model: str):
    global PIPELINE1_MODEL_USED
    PIPELINE1_MODEL_USED = model


def _set_pipeline1_initialized(initialized: bool):
    global PIPELINE1_INITIALIZED
    PIPELINE1_INITIALIZED = initialized

# No longer caching local models as we are using inference API
# def load_slogan_model() and _ensure_pipeline1_loaded() removed below since we delegate directly
PIPELINE1_INITIALIZED = True

try:
    icon_path = ICON_DIR / "nike_icon.png"
    if icon_path.exists():
        st.set_page_config(page_title="AI Smart Marketing", layout="wide", page_icon=str(icon_path))
    else:
        st.set_page_config(page_title="AI Smart Marketing", layout="wide")
except:
    st.set_page_config(page_title="AI Smart Marketing", layout="wide")

# =========================================================
# 2) Data Models & Catalog
# =========================================================
# Define simple robust data structures for Customer and Product
@dataclass
class Customer:
    name: str
    age: int
    gender: str
    nationality: str

@dataclass
class Product:
    id: str
    name: str
    shoe_type: str  # e.g., "Running Shoe", "Basketball Shoe", etc.

# A simple catalog mimicking a real database of products
CATALOG = [
    Product("air-force-1", "Nike Air Force 1 '07 LV8", "Casual Shoe"),
    Product("acg-ultrafly", "Nike ACG Ultrafly Trail", "Trail Shoe"),
    Product("vomero-plus", "Nike Vomero Plus", "Running Shoe"),
    Product("kobe-3-protro", "Kobe III Protro", "Basketball Shoe"),
    Product("tiempo-maestro", "Nike Tiempo Maestro Elite LV8", "Football Shoe"),
    Product("sb-dunk-low", "Nike SB Dunk Low Pro Premium", "Skateboarding Shoe"),
]

# =========================================================
# 3) AI Generators
# =========================================================


def normalize_video_output(output):
    """Normalize video output from various formats to usable file path."""
    if output is None:
        return None

    # URL or file path
    if isinstance(output, (str, Path)):
        s = str(output)
        if s.startswith(("http://", "https://")) or Path(s).exists():
            return s
        return None

    # Has URL attribute
    if hasattr(output, "url"):
        url = output.url
        if isinstance(url, str) and url:
            return url

    # Dict handling
    if isinstance(output, dict):
        # Nested `data` payload used by fal responses
        data_payload = output.get("data")
        if isinstance(data_payload, dict):
            for k in ["url", "video_url", "file", "path"]:
                v = data_payload.get(k)
                if isinstance(v, str) and v:
                    return v

            nested_video = data_payload.get("video")
            if isinstance(nested_video, dict):
                for k in ["url", "video_url"]:
                    v = nested_video.get(k)
                    if isinstance(v, str) and v:
                        return v

        # Top-level URLs
        for k in ["url", "video_url", "file", "path"]:
            v = output.get(k)
            if isinstance(v, str) and v:
                return v
        
        # Nested video
        video_data = output.get("video")
        if video_data is not None:
            if isinstance(video_data, dict):
                for k in ["url", "video_url"]:
                    v = video_data.get(k)
                    if isinstance(v, str) and v:
                        return v

    return None


def _is_video_source_playable(video_source: str | None) -> bool:
    """Validate whether a video source is playable by Streamlit."""
    if not video_source:
        return False
    if video_source.startswith("http://") or video_source.startswith("https://"):
        return True
    return Path(video_source).exists()


def _normalize_messages_for_chat_api(messages: list[dict]) -> list[dict]:
    """Normalize chat messages for HF chat APIs while preserving remote image URLs."""
    normalized = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_value = str(part.get("text", "")).strip()
                    if text_value:
                        parts.append({"type": "text", "text": text_value})
                elif part.get("type") == "image":
                    image_url = str(part.get("url", "")).strip()
                    if image_url.startswith("http://") or image_url.startswith("https://"):
                        parts.append({"type": "image_url", "image_url": {"url": image_url}})
                    else:
                        # Local file URIs/paths cannot be fetched by remote API compute.
                        parts.append({"type": "text", "text": "[local image omitted for remote inference]"})
            normalized.append({"role": role, "content": parts})
            continue
        else:
            normalized.append({"role": role, "content": str(content)})
    return normalized


def _extract_text_from_chat_content(content) -> str:
    """Extract text from chat message content variants returned by providers."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                # Common formats: {"type":"text","text":"..."} or {"text":"..."}
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
                # Some providers nest text under {"type":"output_text","text":"..."}
                content_text = item.get("content")
                if isinstance(content_text, str) and content_text.strip():
                    chunks.append(content_text.strip())
        return "\n".join(chunks).strip()

    return ""


def _messages_to_plain_prompt(messages: list[dict]) -> str:
    """Flatten mixed chat messages into a single plain-text prompt."""
    lines = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip() or "user"
        content = msg.get("content", "")
        prefix = f"{role}: "

        if isinstance(content, str):
            text = content.strip()
            if text:
                lines.append(prefix + text)
            continue

        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    t = str(item.get("text", "")).strip()
                    if t:
                        parts.append(t)
                elif item_type == "image":
                    url = str(item.get("url", "")).strip()
                    if url.startswith("http://") or url.startswith("https://"):
                        parts.append(f"[image_url: {url}]")
            merged = "\n".join(parts).strip()
            if merged:
                lines.append(prefix + merged)
            continue

        text = str(content).strip()
        if text:
            lines.append(prefix + text)

    return "\n".join(lines).strip()


def _format_messages_for_image_text_to_text(messages: list[dict]) -> list[dict]:
    """Format messages for image-text-to-text pipeline to use content list structure."""
    formatted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # If content is already a list (has type/text structure), use as-is
        if isinstance(content, list):
            formatted.append({"role": role, "content": content})
        # If content is a string, wrap it in the expected list structure
        elif isinstance(content, str):
            formatted.append({
                "role": role,
                "content": [{"type": "text", "text": content}]
            })
        else:
            formatted.append(msg)
    
    return formatted


def _extract_text_from_text_generation_output(output) -> str:
    """Extract assistant text from transformers text-generation pipeline outputs."""
    if output is None:
        return ""

    first = output
    if isinstance(output, list):
        if not output:
            return ""
        first = output[0]

    if isinstance(first, dict):
        generated = first.get("generated_text", "")
        if isinstance(generated, str):
            return generated.strip()

        # Chat template output can be a list of role/content messages.
        if isinstance(generated, list):
            for item in reversed(generated):
                if not isinstance(item, dict):
                    continue
                if item.get("role") == "assistant":
                    content = item.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

        text = first.get("text", "")
        if isinstance(text, str):
            return text.strip()

    if isinstance(first, str):
        return first.strip()

    return ""


def _run_pipeline_text_api(messages: list[dict], max_new_tokens: int, model: str, base_url: str = None) -> str:
    """Run text generation using Hugging Face InferenceClient chat API in stream mode."""
    if not HF_TOKEN:
        print("HF_TOKEN is not set. Inference will fail.")
        return ""

    normalized_messages = _normalize_messages_for_chat_api(messages)

    # Single path: InferenceClient streaming chat completions.
    try:
        kwargs = {"api_key": HF_TOKEN}
        if base_url:
            kwargs["base_url"] = base_url
            
        client = InferenceClient(**kwargs)
        stream = client.chat.completions.create(
            model=model,
            messages=normalized_messages,
            max_tokens=max_new_tokens,
            temperature=0.7,
            stream=True,
        )

        chunks: list[str] = []
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                chunks.append(content)
            elif content:
                extracted = _extract_text_from_chat_content(content)
                if extracted:
                    chunks.append(extracted)

        text = "".join(chunks).strip()
        if text:
            return text
        print(f"InferenceClient chat returned no usable text for model {model}.")
        return ""
    except Exception as e:
        err = f"InferenceClient chat failed for model {model}: {type(e).__name__}: {e}"
        print(err)
        # Re-raise to let the caller handle and display it if necessary
        raise RuntimeError(err)

def _run_pipeline1_text(messages: list[dict], max_new_tokens: int) -> str:
    """Run Pipeline 1 model with Inference API (no transformers needed locally)."""
    # Use the dedicated dedicated HF Inference Endpoint for Pipeline 1
    custom_endpoint = "https://atm0kc5pzw8g9pck.us-east-1.aws.endpoints.huggingface.cloud"
    return _run_pipeline_text_api(messages, max_new_tokens, SLOGAN_MODEL, base_url=custom_endpoint)


def _run_pipeline2_text(messages: list[dict], max_new_tokens: int) -> str:
    """Run Pipeline 2 model using Inference API."""
    import traceback
    if not HF_TOKEN:
        print("HF_TOKEN is not set. Inference will fail.")
        return ""
        
    # Flatten the messages to simple strings (GLM model may fail with lists/dictionaries)
    flat_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = "\n".join(text_parts).strip()
        flat_messages.append({"role": role, "content": str(content)})
    
    try:
        print(f"\n[Pipeline 2] Sending request to model: {SCRIPT_MODEL}")
        client = InferenceClient(api_key=HF_TOKEN)
        completion = client.chat.completions.create(
            model=SCRIPT_MODEL,
            messages=flat_messages,
            max_tokens=max_new_tokens,
        )
        
        # Debug the raw response
        print(f"\n[Pipeline 2] RAW COMPLETION: {completion}")
        
        # Extract content
        if not completion.choices:
            raise RuntimeError(f"No choices returned by the API. Raw response: {completion}")
            
        message = completion.choices[0].message
        print(f"\n[Pipeline 2] MESSAGE PAYLOAD: {message}")
        
        content = message.content
        if not content:
            raise RuntimeError(f"Response content was empty or null. Message payload: {message}")
            
        if isinstance(content, str):
            return content.strip()
        return _extract_text_from_chat_content(content) if content else ""
        
    except Exception as e:
        err_msg = (
            f"\n{'='*60}\n"
            f"❌ PIPELINE 2 FAILED ❌\n"
            f"Model: {SCRIPT_MODEL}\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Details: {str(e)}\n\n"
            f"Full Traceback:\n{traceback.format_exc()}\n"
            f"Payload Sent: {flat_messages}\n"
            f"{'='*60}\n"
        )
        print(err_msg)
        raise RuntimeError(err_msg) from e

def generate_slogan_and_description(
    customer: Customer,
    product: Product,
    negative_prompt: str,
    video_duration: int,
    product_image_path: str | None = None,
) -> tuple[str, str]:
    """
    Generates a personalized slogan and product description based on customer profile,
    product type, and generation config.
    
    Args:
        customer: Customer profile (name, age, gender, nationality)
        product: Product info (name, shoe_type)
        negative_prompt: Elements to avoid in downstream video output
        video_duration: Duration of generated video in seconds
        product_image_path: Path to the product image
    
    Returns:
        Tuple of (slogan, product_description)
    """
    # Slogan generation (image + text for Pipeline 1)
    slogan_prompt = (
        f"Write a short, engaging Nike slogan (max 10 words) for a {customer.age}yo {customer.nationality} {customer.gender} "
        f"named {customer.name} buying a {product.shoe_type}. "
        f"The slogan should fit a {video_duration}-second high-energy ad concept. "
        f"Avoid concepts related to: {negative_prompt}. "
        f"Make it motivational and empowering. "
        f"Do not mention any product name in the slogan. "
        f"End the slogan with a comma and the customer's name exactly, like '..., {customer.name}'."
    )
    slogan = ""
    try:
        slogan_content = []
        slogan_content.append({"type": "text", "text": slogan_prompt})
        if product_image_path:
            # When deploying behind proxy/API, it usually requires a URL or base64 data uri.
            # Local assets fallback for public models require base64.
            import base64
            with open(product_image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                # Add base64 data prefix (assuming png since assets are .png)
                b64_url = f"data:image/png;base64,{encoded_string}"
            slogan_content.append({"type": "image_url", "image_url": {"url": b64_url}})
        
        messages = [{"role": "user", "content": slogan_content}]
        res = _run_pipeline1_text(messages, max_new_tokens=50)
        if res:
            slogan = res
    except Exception as e:
        print(f"Slogan generation error: {e}")

    if not slogan:
        raise RuntimeError(
            f"Pipeline 1 failed to generate slogan using model {SLOGAN_MODEL}. "
            f"Root cause: {PIPELINE1_LAST_ERROR or PIPELINE1_LOAD_ERROR or 'Unknown error'}"
        )

    # Product description generation (image + text for Pipeline 1)
    description_prompt = (
        f"Write a compelling 2-sentence product description for {product.name} ({product.shoe_type}) "
        f"targeting a {customer.age}yo {customer.nationality} {customer.gender}. "
        f"Focus on performance and design. "
        f"The copy should match a {video_duration}-second cinematic ad and avoid: {negative_prompt}. "
        f"Be vivid and marketing-focused."
    )
    description = ""
    try:
        desc_content = []
        desc_content.append({"type": "text", "text": description_prompt})
        if product_image_path:
            # Re-read and attach base64 representation of the product image for remote vision api
            import base64
            with open(product_image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                b64_url = f"data:image/png;base64,{encoded_string}"
            desc_content.append({"type": "image_url", "image_url": {"url": b64_url}})
            
        messages = [{"role": "user", "content": desc_content}]
        res = _run_pipeline1_text(messages, max_new_tokens=100)
        if res:
            description = res
    except Exception as e:
        print(f"Product description generation error: {e}")

    if not description:
        raise RuntimeError(
            f"Pipeline 1 failed to generate product description using model {SLOGAN_MODEL}. "
            f"Root cause: {PIPELINE1_LAST_ERROR or PIPELINE1_LOAD_ERROR or 'Unknown error'}"
        )

    return slogan, description


def generate_cinematic_script(
    customer: Customer,
    product: Product,
    product_description: str,
    negative_prompt: str,
    video_duration: int,
) -> str:
    """
    Generates a detailed cinematic script for video generation using the reference prompt structure.
    
    Args:
        customer: Customer profile
        product: Product info with shoe_type
        product_description: Description from Pipeline 1
        negative_prompt: Elements to avoid in video
        video_duration: Duration of video in seconds
    
    Returns:
        Detailed cinematic script for video generation
    """
    system_prompt = f"""You are a world-class cinematic prompt engineer and Nike advertising creative director.

Your job is to generate ONE single, extremely detailed, ready-to-use text prompt for high-end image-text-to-video model.

CRITICAL: You MUST follow this exact layered structure and style (never deviate):

[Subject / Hero Shot]:
[Scene & Environment]:
[Motion & Dynamics]:
[Camera & Cinematography]:
[Lighting & Mood]:
[Personalization Layer]:
[Style & Quality Boosters]:

Use highly vivid, professional advertising language with cinematic terms (tracking pan, dolly zoom, orbiting crane shot, low-angle side-to-front reveal, anamorphic lenses, ARRI Alexa 65, 60fps slow-motion bursts, volumetric god rays, lens flares, etc.).

Essential requirements:
- Emphasize visible Swoosh branding on clothing and billboards
- High-energy athletic motion and dynamic action
- Futuristic neon city aesthetics with golden hour sunset lighting
- Motivational and empowering atmosphere tailored to the customer profile
- {product.shoe_type} in prominent focus throughout the video
- Video duration: {video_duration} seconds
- Avoid: {negative_prompt}

Personalization is CRITICAL: Tailor the energy, tone, appeal, and cultural resonance specifically for a {customer.age}-year-old {customer.gender} named {customer.name} from {customer.nationality}. Make it feel empowering and perfectly matched to this demographic.

Product context: {product.name} ({product.shoe_type}) - {product_description}

Output ONLY the final prompt text — nothing else. No explanations, no JSON, no markdown, no extra words. Start directly with "[Subject / Hero Shot]:"."""

    user_message = f"""Analyze this Nike product and {product.shoe_type} type, then create the perfect cinematic video prompt.
Product: {product.name}
Type: {product.shoe_type}
Description: {product_description}
Target: {customer.age}yo {customer.gender} from {customer.nationality}
Negative constraints: {negative_prompt}

Generate the cinematic script now."""

    script = ""
    last_err = ""
    try:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_message}]},
        ]
        res = _run_pipeline2_text(messages, max_new_tokens=500)
        if res:
            script = res
    except Exception as e:
        last_err = str(e)
        print(f"Cinematic script generation error: {e}")

    if not script:
        raise RuntimeError(f"Pipeline 2 failed to generate cinematic script using model {SCRIPT_MODEL}. Error: {last_err}")

    return script


def get_product_image(product: Product) -> str | None:
    """
    Retrieves the product image path from the assets folder.
    
    Args:
        product: Product object
    
    Returns:
        Path to product image or None if not found
    """
    # Try multiple naming conventions for product images
    # First try: product_id.png
    img_path = ASSETS_DIR / f"{product.id}.png"
    if img_path.exists():
        return str(img_path)
    
    # Second try: product_name.png (with spaces)
    img_path = ASSETS_DIR / f"{product.name}.png"
    if img_path.exists():
        return str(img_path)
    
    # Third try: product_name with underscores instead of spaces
    img_name_underscore = product.name.replace(" ", "_")
    img_path = ASSETS_DIR / f"{img_name_underscore}.png"
    if img_path.exists():
        return str(img_path)
    
    return None


def generate_video(product_image_path: str | None, cinematic_script: str, slogan: str, 
                  customer: Customer, product: Product, video_duration: int) -> str:
    """
    Generates a video from product image with cinematic effects and slogan overlay.
    Uses the product image as the base and generates a video with the slogan embedded at the end.
    
    Args:
        product_image_path: Path to product image or None
        cinematic_script: Detailed cinematic script for video description
        slogan: Slogan to embed at the end of video
        customer: Customer profile
        product: Product info
        video_duration: Duration of video in seconds
    
    Returns:
        Path to generated video file
    """
    if not product_image_path or not Path(product_image_path).exists():
        raise RuntimeError(
            f"Pipeline 3 failed: product image not found for {product.name}."
        )

    if not FAL_KEY:
        raise RuntimeError("Pipeline 3 failed: FAL_KEY is missing in Streamlit secrets.")

    try:
        image_url = fal.upload_file(product_image_path)
    except Exception as e:
        raise RuntimeError(f"Pipeline 3 failed uploading image to fal.ai storage: {type(e).__name__}: {e}") from e

    video_prompt = (
        f"{cinematic_script}\n"
        f"Integrate this closing brand line naturally in the final beat: {slogan}.\n"
        f"Target duration: {video_duration} seconds."
    )

    def _on_queue_update(update):
        try:
            if isinstance(update, dict) and update.get("status") == "IN_PROGRESS":
                for log in update.get("logs", []):
                    message = log.get("message") if isinstance(log, dict) else None
                    if message:
                        print(f"Pipeline 3 queue: {message}")
        except Exception:
            pass

    try:
        result = fal.subscribe(
            VIDEO_MODEL,
            arguments={
                "image_url": image_url,
                "prompt": video_prompt,
            },
            with_logs=True,
            on_queue_update=_on_queue_update,
        )
    except Exception as e:
        raise RuntimeError(f"Pipeline 3 failed calling fal.ai model {VIDEO_MODEL}: {type(e).__name__}: {e}") from e

    video_source = normalize_video_output(result)
    if not video_source:
        raise RuntimeError(f"Pipeline 3 failed: fal.ai returned no usable video source for model {VIDEO_MODEL}.")

    return video_source

# =========================================================
# 4) Main Streamlit Application
# =========================================================
def main():
    # Force dark theme globally via CSS overrides
    st.markdown("""
        <style>
        /* Main background and text */
        .stApp {
            background-color: #0e1117 !important;
            color: #ffffff !important;
        }
        
        /* Top header transparency */
        .stApp header {
            background-color: transparent !important;
        }

        /* Sidebar background and text */
        [data-testid="stSidebar"] {
            background-color: #262730 !important;
            color: #ffffff !important;
        }

        /* Text colors */
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, span {
            color: #ffffff !important;
        }

        /* Input fields (text, number, selectbox) */
        .stTextInput>div>div>input, 
        .stNumberInput>div>div>input,
        .stTextArea textarea,
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
            border-color: #444444 !important;
        }

        /* Ensure number and textarea controls in sidebar match product select style */
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] .stTextArea textarea {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #444444 !important;
        }
        
        /* Primary button styling */
        .stButton>button {
            background-color: #ff4b4b !important;
            color: white !important;
            border: none !important;
        }
        .stButton>button:hover {
            background-color: #ff6b6b !important;
        }
        
        /* Info/Success/Warning boxes */
        [data-testid="stAlert"] {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
            border: 1px solid #444444 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 1. Header Details Setup
    st.markdown("## AI Smart Marketing: Personalized Nike Video Advertisements\n"
                "This app generates personalized Nike campaign toolsets utilizing multi-modal GenAI.")

    # 2. Sidebar Configuration inputs matching the app
    with st.sidebar:
        st.header("Customer Profile")
        name = st.text_input("Name", "Alex")
        age = st.number_input("Age", 10, 90, 25)
        # Selectbox input provides rigid variables easing prompt structuring
        gender = st.selectbox("Gender", ["Male", "Female"])
        nationality = st.selectbox("Nationality", ["USA", "Chinese"])
        
        st.header("Generation Config")
        
        video_duration = st.slider(
            "Video Duration (seconds)",
            min_value=1,
            max_value=5,
            value=5,
            step=1
        )
        
        negative_prompt = st.text_area(
            "Negative Prompt for Video",
            value="blurry, low quality, artifacts, deformed, static, text, watermark, ugly, distorted, overexposed",
            height=100
        )
        
        st.header("Product Selection")
        product_name = st.selectbox("Product", [p.name for p in CATALOG])
        
        # Display product image if available
        selected_product = next(p for p in CATALOG if p.name == product_name)
        product_image = get_product_image(selected_product)
        if product_image:
            # Note: the `use_container_width` parameter was formerly known as `use_column_width` or just `width="stretch"` in older Streamlit. `use_container_width=True` is the proper way.
            st.image(product_image, use_container_width=True)
        else:
            st.info(f"No image found for {selected_product.name}")

        if not st.secrets.get("HF_TOKEN") and not os.environ.get("HF_TOKEN"):
            st.warning("HF_TOKEN is missing! Inference will likely fail.")
            
        generate_btn = st.button("Generate Assets", type="primary", use_container_width=True)

    # 3. Execution Pipeline tracking State and Output
    if generate_btn and name:
        customer = Customer(name, int(age), gender, nationality)
        product = next(p for p in CATALOG if p.name == product_name)
        
        prog = st.progress(0, "Initiating pipeline...")
        
        # Phase 1: Retrieve marketing text (Slogan + Product Description)
        try:
            slogan, product_description = generate_slogan_and_description(
                customer,
                product,
                negative_prompt,
                video_duration,
                product_image_path=product_image
            )
        except Exception as e:
            st.error(str(e))
            if PIPELINE1_LAST_ERROR or PIPELINE1_LOAD_ERROR:
                st.caption(f"Pipeline 1 root cause: {PIPELINE1_LAST_ERROR or PIPELINE1_LOAD_ERROR}")
            st.stop()

        prog.progress(25, "Pipeline 1: Slogan & Product Description generated!")
        st.write("**Pipeline 1 (Slogan & Product Description):**")
        st.caption(f"Model used: {SLOGAN_MODEL}")
        if PIPELINE1_MODEL_USED:
            st.caption(f"Pipeline 1 model runtime: {PIPELINE1_MODEL_USED}")
        st.caption(f"Pipeline 1 backend: {PIPELINE1_BACKEND or 'unavailable'}")
        st.success(f"**Slogan:** {slogan}\n\n**Description:** {product_description}")
        
        # Phase 2: Generate cinematic script for video
        st.write("**Pipeline 2 (Cinematic Script Generation):**")
        try:
            cinematic_script = generate_cinematic_script(
                customer,
                product,
                product_description,
                negative_prompt,
                video_duration,
            )
        except Exception as e:
            st.error(str(e))
            st.stop()

        st.caption(f"Model used: {SCRIPT_MODEL}")
        st.info(cinematic_script)
        prog.progress(50, "Pipeline 2: Cinematic script generated!")
        
        # Phase 3: Generate video with product image and slogan overlay
        st.write("**Pipeline 3 (Video Generation with Slogan Overlay):**")
        st.caption(f"Model used: {VIDEO_MODEL}")
        product_image_path = get_product_image(product)
        
        try:
            with st.spinner(f"Generating {video_duration}s video with slogan overlay..."):
                vid_path = generate_video(
                    product_image_path, cinematic_script, slogan,
                    customer, product, video_duration
                )
        except Exception as e:
            st.error(str(e))
            st.stop()
        
        if _is_video_source_playable(vid_path):
            st.video(vid_path)
            prog.progress(100, "All pipelines completed!")
            st.success("Video generated successfully with slogan embedded at the end!")
        else:
            st.error("Failed to generate video. Please check the logs.")
        prog.progress(100, "All pipelines completed!")

if __name__ == "__main__":
    main()
