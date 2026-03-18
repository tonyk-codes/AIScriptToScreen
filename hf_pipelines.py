"""Hugging Face-backed pipeline implementations for Smart Nike Shoe Ad Studio."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import streamlit as st
import torch
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
)

import config
from interfaces import (
    CustomerProfile,
    EncodedProduct,
    EncodedProfile,
    MarketingAssets,
    ProductEncoder,
    ProductInfo,
    ProfileEncoder,
    SceneGenerator,
    ScriptGenerator,
    SloganGenerator,
    StorylineGenerator,
    VideoGenerator,
)
from media_utils import (
    compose_final_ad_video,
    create_fallback_banner_video,
    ensure_local_image,
    save_video_bytes,
    sanitize_filename,
)


def _deterministic_embedding(text: str, size: int = 384) -> list[float]:
    """Return deterministic pseudo-embedding for graceful degradation."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < size:
        for byte in digest:
            values.append((byte / 255.0) * 2.0 - 1.0)
            if len(values) == size:
                break
        digest = hashlib.sha256(digest).digest()
    return values


def build_profile_summary(profile: CustomerProfile) -> str:
    """Pure helper used for profile encoding and prompt construction."""
    notes = profile.additional_notes.strip() or "No additional notes"
    return (
        f"Name: {profile.name}; Age: {profile.age}; Gender: {profile.gender}; "
        f"Nationality: {profile.nationality or 'Not specified'}; Language: {profile.language}; "
        f"Product ID: {profile.product_id}; Notes: {notes}."
    )


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


def _deterministic_pick(options: list[str], key: str) -> str:
    if not options:
        return ""
    idx = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % len(options)
    return options[idx]


def _fallback_slogan(
    profile: CustomerProfile,
    product: ProductInfo,
    product_attributes: dict[str, str],
) -> str:
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
        template = _deterministic_pick(templates, key)
        return template.format(
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
    template = _deterministic_pick(templates, key)
    return template.format(
        nationality=profile.nationality,
        age=profile.age,
        gender=profile.gender,
        product_tail=product.name.split()[-1],
        benefit=benefit,
    )


def build_script_prompt(
    profile_summary: str,
    slogan: str,
    product: ProductInfo,
    product_attributes: dict[str, str],
) -> str:
    attrs = ", ".join(f"{k}: {v}" for k, v in product_attributes.items()) or "No attributes provided"
    return (
        "You are a Nike marketing copywriter. Write personalized ad copy in English only. "
        "Create output as strict JSON with keys: headline, script. "
        "headline should be one line. script should be 3-5 sentences and suitable for 15-40 second narration.\n\n"
        f"Customer profile: {profile_summary}\n"
        f"Approved slogan: {slogan}\n"
        f"Product name: {product.name}\n"
        f"Product category: {product.category}\n"
        f"Product audience: {product.gender}\n"
        f"Product attributes: {attrs}\n"
        f"Product URL: {product.product_url}\n\n"
        "Keep the tone aspirational, energetic, and premium. "
        "Mention practical fit for the customer profile. "
        "Future multilingual adaptation note: only output English for now."
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


def parse_generated_script(raw_output: str) -> tuple[str, str]:
    raw_output = (raw_output or "").strip()
    if not raw_output:
        return (
            "Designed for your next step.",
            "These Nike shoes are tailored to your lifestyle and goals. Move with confidence and style every day.",
        )

    try:
        parsed = json.loads(raw_output)
        headline = str(parsed.get("headline", "Designed for your next step.")).strip()
        script = str(parsed.get("script", "Feel the energy in every stride.")).strip()
        return headline, script
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            headline = str(parsed.get("headline", "Designed for your next step.")).strip()
            script = str(parsed.get("script", "Feel the energy in every stride.")).strip()
            return headline, script
        except json.JSONDecodeError:
            pass

    lines = [line.strip("-• ") for line in raw_output.splitlines() if line.strip()]
    headline = lines[0] if lines else "Designed for your next step."
    script = " ".join(lines[1:]) if len(lines) > 1 else raw_output
    return headline[:120], script


def build_storyline_prompt(
    profile: CustomerProfile,
    product: ProductInfo,
    slogan: str,
) -> str:
    scene_styles = [
        "urban dawn training route with reflective streets",
        "sunset rooftop run with warm cinematic haze",
        "indoor court lights with dramatic shadows and speed",
        "city crosswalk rush-hour moment with crisp focus",
        "coastal boardwalk stride with golden-hour glow",
    ]
    mood_styles = [
        "confident and premium",
        "youthful and fearless",
        "focused and determined",
        "playful but elite",
        "bold and emotional",
    ]
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


def _fallback_storyline(profile: CustomerProfile, product: ProductInfo) -> str:
    templates = [
        "{age}-year-old {gender} from {nationality}, full face and body visible, bursts from a metro staircase in {product} as rain mist catches neon light, cinematic Nike-style momentum",
        "{age}-year-old {gender} from {nationality}, clear face and full body, drives a fast break on an indoor court wearing {product}, dramatic rim light and bold ad realism",
        "{age}-year-old {gender} from {nationality}, face and body in focus, powers through a sunrise bridge run in {product}, wind-swept motion and premium campaign energy",
        "{age}-year-old {gender} from {nationality}, full body and expressive face visible, transitions from office streetwear to performance stride in {product}, dynamic lifestyle-to-sport cinematic arc",
        "{age}-year-old {gender} from {nationality}, face and body clearly shown, leads a twilight rooftop training session in {product}, high-detail realism and aspirational brand mood",
    ]
    key = f"story:{profile.name}|{profile.age}|{profile.gender}|{profile.nationality}|{product.name}"
    template = _deterministic_pick(templates, key)
    return template.format(
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


@st.cache_resource(show_spinner=False)
def _load_sentence_transformer(model_id: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


@st.cache_resource(show_spinner=False)
def _load_clip_components(model_id: str):
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def _load_text2text_components(model_id: str):
    is_gguf = "gguf" in model_id.lower()
    gguf_kwargs = {}
    if is_gguf:
        gguf_kwargs["gguf_file"] = "Qwen3.5-2B.Q4_K_M.gguf"
        
    tokenizer = AutoTokenizer.from_pretrained(model_id, **gguf_kwargs)
    try:
        if is_gguf:
            raise ValueError("GGUF uses CausalLM")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    except Exception:
        # Fallback to CausalLM for models like Qwen or GGUF
        model = AutoModelForCausalLM.from_pretrained(model_id, **gguf_kwargs)
    return tokenizer, model


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
            result = cast(
                Any,
                client,
            ).text_generation(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                return_full_text=False,
            )
            return str(result or "").strip()
        except StopIteration:
            # Some hub/provider combinations return no mapping and raise StopIteration.
            # Fall back to a raw inference POST payload instead of crashing the app.
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


def _generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
        )
    
    if len(outputs) == 0:
        return ""
        
    # Check if it's a causal LM (decoder-only). If so, ignore the input tokens.
    if getattr(model.config, "is_encoder_decoder", False):
        generated_ids = outputs[0]
    else:
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_length:]
        
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


class HFProfileEncoder(ProfileEncoder):
    """Pipeline 1: profile text to embedding vector."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or config.PROFILE_EMBEDDING_MODEL_ID

    def encode(self, profile: CustomerProfile) -> EncodedProfile:
        summary = build_profile_summary(profile)
        try:
            model = _load_sentence_transformer(self.model_id)
            embedding = model.encode(summary).tolist()
        except Exception:
            embedding = _deterministic_embedding(summary)
        return EncodedProfile(profile_summary=summary, embedding=embedding)


class HFProductEncoder(ProductEncoder):
    """Pipeline 2: product image+text to embedding vector."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or config.PRODUCT_CLIP_MODEL_ID

    @staticmethod
    def _infer_attributes(product: ProductInfo) -> dict[str, str]:
        baseline = product.attributes.copy() if product.attributes else {}
        category_map = {
            "running": {"support": "smooth transitions", "performance": "distance-ready"},
            "basketball": {"support": "lateral stability", "performance": "explosive cuts"},
            "lifestyle": {"support": "all-day comfort", "performance": "street style"},
            "training": {"support": "multi-direction control", "performance": "gym versatility"},
        }
        for key, value in category_map.items():
            if key in product.category.lower():
                baseline.update(value)
                break
        if "feel" not in baseline:
            baseline["feel"] = "confident and energetic"
        return baseline

    def encode(self, product: ProductInfo) -> EncodedProduct:
        local_image = ensure_local_image(product.image_path_or_url, product.name)
        attributes = self._infer_attributes(product)
        summary = (
            f"{product.name}; category: {product.category}; audience: {product.gender}; "
            f"attributes: {', '.join(f'{k} {v}' for k, v in attributes.items())}."
        )

        try:
            processor, model = _load_clip_components(self.model_id)
            image = Image.open(local_image).convert("RGB")

            processor_any = cast(Any, processor)
            model_any = cast(Any, model)
            text_inputs = processor_any(text=[summary], return_tensors="pt", padding=True, truncation=True)
            image_inputs = processor_any(images=image, return_tensors="pt")

            with torch.no_grad():
                text_features = model_any.get_text_features(**text_inputs)
                image_features = model_any.get_image_features(**image_inputs)

            merged = cast(torch.Tensor, (text_features + image_features) / 2.0)
            embedding = merged[0].cpu().tolist()
        except Exception:
            embedding = _deterministic_embedding(summary, size=512)

        return EncodedProduct(product_summary=summary, embedding=embedding, attributes=attributes)


class HFSloganGenerator(SloganGenerator):
    """Pipeline 3a: profile+product context to short slogan."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or config.SLOGAN_GENERATION_MODEL_ID
        self.token = config.HF_TOKEN

    def generate(
        self,
        profile: CustomerProfile,
        encoded_profile: EncodedProfile,
        product: ProductInfo,
        encoded_product: EncodedProduct,
    ) -> str:
        prompt = build_slogan_prompt(
            profile_summary=encoded_profile.profile_summary,
            product=product,
            product_attributes=encoded_product.attributes,
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
            slogan = _fallback_slogan(profile, product, encoded_product.attributes)
        return f"{slogan}, {profile.name}"


class HFStorylineGenerator(StorylineGenerator):
    """Pipeline 2: generate a concise storyline with required realistic character details."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or config.SCRIPT_GENERATION_MODEL_ID
        self.token = config.HF_TOKEN

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

        required = f"{profile.age}-year-old {profile.gender} from {profile.nationality}"
        if required.lower() not in storyline.lower():
            storyline = f"{storyline}"

        return storyline[:260]


class HFSceneGenerator(SceneGenerator):
    """Pipeline 3: generate three realistic scene images from storyline."""

    def __init__(self, model_id: str | None = None, token: str | None = None) -> None:
        self.model_id = model_id or config.SCENE_IMAGE_MODEL_ID
        self.token = token or config.HF_TOKEN

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
                    payload = {"inputs": prompt}
                    result = cast(Any, client).post(json=payload)
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
        config.ensure_artifact_dirs()
        safe_stem = sanitize_filename(f"{profile.name}-{product.product_id}")
        image_paths: list[str] = []
        for idx in range(1, max(1, image_count) + 1):
            prompt = self._scene_prompt(profile, product, storyline, idx)
            target = config.IMAGES_DIR / f"{safe_stem}-scene-{idx}.png"
            image_paths.append(self._generate_single_image(prompt, target, idx))
        return image_paths


class HFScriptGenerator(ScriptGenerator):
    """Pipeline 3b: profile+product context to headline and long-form script."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or config.SCRIPT_GENERATION_MODEL_ID

    def generate(
        self,
        profile: CustomerProfile,
        encoded_profile: EncodedProfile,
        product: ProductInfo,
        encoded_product: EncodedProduct,
        slogan: str,
    ) -> MarketingAssets:
        tokenizer, model = _load_text2text_components(self.model_id)
        prompt = build_script_prompt(
            profile_summary=encoded_profile.profile_summary,
            slogan=slogan,
            product=product,
            product_attributes=encoded_product.attributes,
        )
        raw = _generate_text(model, tokenizer, prompt, max_new_tokens=220)
        headline, script = parse_generated_script(raw)

        debug = {
            "model_id": self.model_id,
            "raw_text_output": raw,
            "profile_embedding_size": len(encoded_profile.embedding),
            "product_embedding_size": len(encoded_product.embedding),
            "profile": asdict(profile),
        }
        return MarketingAssets(slogan=slogan, headline=headline, script=script, debug_metadata=debug)


class HFVideoGenerator(VideoGenerator):
    """Pipeline 4: text+image to promotional short video, with robust fallback."""

    def __init__(self, model_id: str | None = None, token: str | None = None) -> None:
        self.model_id = model_id or config.VIDEO_MODEL_ID
        self.token = token or config.HF_TOKEN

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
        if not config.USE_HF_INFERENCE_API_FOR_VIDEO or not self.token:
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

        # Streamlit Cloud-safe fallback when inference or local TI2V is unavailable.
        return create_fallback_banner_video(
            image_path_or_url=reference_image or product.image_path_or_url,
            slogan=assets.slogan,
            headline=assets.headline,
            output_stem=output_stem,
            final_slogan_text=final_slogan_text,
        )
