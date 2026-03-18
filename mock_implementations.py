"""Deterministic mock backends for Smart Nike Shoe Ad Studio."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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
from media_utils import create_fallback_banner_video, sanitize_filename


def _mock_embedding(key: str, size: int) -> list[float]:
    seed = hashlib.sha1(key.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < size:
        for byte in seed:
            values.append((byte / 255.0) * 2.0 - 1.0)
            if len(values) == size:
                break
        seed = hashlib.sha1(seed).digest()
    return values


def _deterministic_pick(options: list[str], key: str) -> str:
    if not options:
        return ""
    idx = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % len(options)
    return options[idx]


class MockProfileEncoder(ProfileEncoder):
    def encode(self, profile: CustomerProfile) -> EncodedProfile:
        summary = (
            f"Name: {profile.name}; Age: {profile.age}; Gender: {profile.gender}; "
            f"Nationality: {profile.nationality or 'Not specified'}; Language: {profile.language}; "
            f"Product ID: {profile.product_id}; Notes: {profile.additional_notes or 'No notes'}."
        )
        return EncodedProfile(profile_summary=summary, embedding=_mock_embedding(summary, 384))


class MockProductEncoder(ProductEncoder):
    def encode(self, product: ProductInfo) -> EncodedProduct:
        attributes = (product.attributes or {}).copy()
        if "benefit" not in attributes:
            attributes["benefit"] = "all-day comfort"
        summary = f"{product.name} in {product.category} for {product.gender}."
        return EncodedProduct(
            product_summary=summary,
            embedding=_mock_embedding(f"{product.product_id}:{summary}", 512),
            attributes=attributes,
        )


class MockSloganGenerator(SloganGenerator):
    def generate(
        self,
        profile: CustomerProfile,
        encoded_profile: EncodedProfile,
        product: ProductInfo,
        encoded_product: EncodedProduct,
    ) -> str:
        benefit = (
            encoded_product.attributes.get("benefit")
            or encoded_product.attributes.get("performance")
            or "all-day comfort"
        )
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


class MockScriptGenerator(ScriptGenerator):
    def generate(
        self,
        profile: CustomerProfile,
        encoded_profile: EncodedProfile,
        product: ProductInfo,
        encoded_product: EncodedProduct,
        slogan: str,
    ) -> MarketingAssets:
        headline = f"{product.name} built for your {product.category.lower()} rhythm"
        script = (
            f"Meet {profile.name}'s personalized Nike moment. "
            f"The {product.name} blends {encoded_product.attributes.get('benefit', 'performance support')} "
            f"with a style that fits a {profile.age}-year-old {profile.gender.lower()} customer. "
            "Every stride feels lighter, more confident, and ready for daily wins. "
            "Lace up, move happy, and turn your routine into a statement."
        )
        return MarketingAssets(
            slogan=slogan,
            headline=headline,
            script=script,
            final_slogan_text=slogan,
            debug_metadata={"mode": "mock", "profile_summary": encoded_profile.profile_summary},
        )


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
        config.ensure_artifact_dirs()
        stem = sanitize_filename(f"mock-{profile.name}-{product.product_id}")
        images: list[str] = []
        for idx in range(1, max(1, image_count) + 1):
            path = config.IMAGES_DIR / f"{stem}-scene-{idx}.png"
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
