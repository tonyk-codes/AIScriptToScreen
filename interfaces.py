"""Core data contracts and registries for Smart Nike Shoe Ad Studio."""

from dataclasses import dataclass, field
from typing import Any, Protocol


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
class EncodedProfile:
    profile_summary: str
    embedding: list[float]


@dataclass(slots=True)
class EncodedProduct:
    product_summary: str
    embedding: list[float]
    attributes: dict[str, str] = field(default_factory=dict)


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


class ProfileEncoder(Protocol):
    def encode(self, profile: CustomerProfile) -> EncodedProfile:
        ...


class ProductEncoder(Protocol):
    def encode(self, product: ProductInfo) -> EncodedProduct:
        ...


class SloganGenerator(Protocol):
    def generate(
        self,
        profile: CustomerProfile,
        encoded_profile: EncodedProfile,
        product: ProductInfo,
        encoded_product: EncodedProduct,
    ) -> str:
        ...


class ScriptGenerator(Protocol):
    def generate(
        self,
        profile: CustomerProfile,
        encoded_profile: EncodedProfile,
        product: ProductInfo,
        encoded_product: EncodedProduct,
        slogan: str,
    ) -> MarketingAssets:
        ...


class StorylineGenerator(Protocol):
    def generate(
        self,
        profile: CustomerProfile,
        product: ProductInfo,
        slogan: str,
    ) -> str:
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
    def generate(
        self,
        profile: CustomerProfile,
        product: ProductInfo,
        assets: MarketingAssets,
    ) -> str:
        ...


_PROFILE_ENCODERS: dict[str, type[ProfileEncoder]] = {}
_PRODUCT_ENCODERS: dict[str, type[ProductEncoder]] = {}
_SLOGAN_GENERATORS: dict[str, type[SloganGenerator]] = {}
_SCRIPT_GENERATORS: dict[str, type[ScriptGenerator]] = {}
_VIDEO_GENERATORS: dict[str, type[VideoGenerator]] = {}


def register_profile_encoder(name: str, implementation: type[ProfileEncoder]) -> None:
    _PROFILE_ENCODERS[name] = implementation


def register_product_encoder(name: str, implementation: type[ProductEncoder]) -> None:
    _PRODUCT_ENCODERS[name] = implementation


def register_slogan_generator(name: str, implementation: type[SloganGenerator]) -> None:
    _SLOGAN_GENERATORS[name] = implementation


def register_script_generator(name: str, implementation: type[ScriptGenerator]) -> None:
    _SCRIPT_GENERATORS[name] = implementation


def register_video_generator(name: str, implementation: type[VideoGenerator]) -> None:
    _VIDEO_GENERATORS[name] = implementation


def get_profile_encoder(name: str) -> type[ProfileEncoder]:
    return _PROFILE_ENCODERS[name]


def get_product_encoder(name: str) -> type[ProductEncoder]:
    return _PRODUCT_ENCODERS[name]


def get_slogan_generator(name: str) -> type[SloganGenerator]:
    return _SLOGAN_GENERATORS[name]


def get_script_generator(name: str) -> type[ScriptGenerator]:
    return _SCRIPT_GENERATORS[name]


def get_video_generator(name: str) -> type[VideoGenerator]:
    return _VIDEO_GENERATORS[name]


def list_registered_stages() -> dict[str, list[str]]:
    return {
        "profile_encoders": sorted(_PROFILE_ENCODERS.keys()),
        "product_encoders": sorted(_PRODUCT_ENCODERS.keys()),
        "slogan_generators": sorted(_SLOGAN_GENERATORS.keys()),
        "script_generators": sorted(_SCRIPT_GENERATORS.keys()),
        "video_generators": sorted(_VIDEO_GENERATORS.keys()),
    }
