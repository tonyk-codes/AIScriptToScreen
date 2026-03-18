from __future__ import annotations

from typing import cast

import streamlit as st

import config
from hf_pipelines import HFSceneGenerator, HFSloganGenerator, HFStorylineGenerator, HFVideoGenerator
from interfaces import CustomerProfile, EncodedProduct, EncodedProfile, MarketingAssets, ProductInfo
from mock_implementations import MockSceneGenerator, MockSloganGenerator, MockStorylineGenerator, MockVideoGenerator
from nike_catalog import get_product_by_name, list_products

config.ensure_artifact_dirs()

st.set_page_config(page_title="AI Smart Marketing: Personalized Video Advertisements", layout="wide", initial_sidebar_state="expanded")

hf_token = st.secrets.get("HF_TOKEN", "")
if hf_token:
    config.HF_TOKEN = hf_token

PIPELINE_MODELS = {
    1: config.SLOGAN_GENERATION_MODEL_ID,
    2: config.SCRIPT_GENERATION_MODEL_ID,
    3: config.SCENE_IMAGE_MODEL_ID,
    4: "Wan-AI/Wan2.2-TI2V-5B",
}

PIPELINE_NAMES = {
    1: "Personalized Slogan Generation (Text Generation)",
    2: "Personalized Storyline Generation (Text Generation)",
    3: "Storyline Scene Generation (Text-to-Image Generation)",
    4: "Personalized Marketing Video Generation (Image+Text-to-Video Generation)",
}

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
}

.stApp {
    background: var(--owid-bg);
    color: var(--owid-text);
}

[data-testid="collapsedControl"] {
    display: none !important;
}

.block-container {
    padding-top: 5rem;
    padding-bottom: 1.2rem;
}

.top-header {
    border: 1px solid #0e2c50;
    background: linear-gradient(90deg, var(--owid-blue), var(--owid-blue-2));
    border-radius: 8px;
    color: #f8fbff;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}

.top-header h1 {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 2.1rem;
    margin: 0;
    font-weight: 600;
}

.top-header p {
    margin-top: 0.4rem;
    opacity: 0.92;
    font-size: 0.95rem;
}

.pipeline-grid {
    border: 1px solid var(--owid-border);
    border-radius: 8px;
    background: #ffffff;
    margin-bottom: 1rem;
    padding: 0.8rem 0.9rem;
}

.pipeline-grid table {
    width: 100%;
    border-collapse: collapse;
}

.pipeline-grid th, .pipeline-grid td {
    border-bottom: 1px solid #ecf0f5;
    padding: 0.55rem 0.45rem;
    font-size: 0.9rem;
}

.pipeline-grid th {
    text-align: left;
    color: #163555;
    font-weight: 700;
}

.result-card {
    border: 1px solid var(--owid-border);
    border-radius: 8px;
    background: #ffffff;
    margin-bottom: 0.9rem;
    padding: 1rem;
}

.result-card h3 {
    margin: 0 0 0.5rem 0;
    color: #163555;
    font-size: 1.06rem;
}

.info-note {
    border-left: 4px solid var(--owid-red);
    padding: 0.55rem 0.8rem;
    background: #fff7f4;
    margin-bottom: 1rem;
    font-size: 0.88rem;
}

.full-video [data-testid="stVideo"] {
    width: 100% !important;
}

.full-video [data-testid="stVideo"] > div {
    width: 100% !important;
}

.full-video [data-testid="stVideo"] video,
.full-video [data-testid="stVideo"] iframe {
    width: 100% !important;
    height: auto !important;
    display: block;
}

[data-testid="stSidebar"] {
    background: #f6f8fb;
    border-right: 1px solid #dbe3ed;
}

@media (max-width: 900px) {
    .top-header h1 {
        font-size: 1.6rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_catalog() -> list[ProductInfo]:
    return list_products()


def init_state() -> None:
    defaults = {
        "assets": None,
        "video_path": None,
        "scene_images": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_backends() -> tuple[
    HFSloganGenerator | MockSloganGenerator,
    HFStorylineGenerator | MockStorylineGenerator,
    HFSceneGenerator | MockSceneGenerator,
    HFVideoGenerator | MockVideoGenerator,
    bool,
]:
    use_mock = config.FORCE_MOCK_MODE or not config.HF_TOKEN
    if use_mock:
        return (
            MockSloganGenerator(),
            MockStorylineGenerator(),
            MockSceneGenerator(),
            MockVideoGenerator(),
            True,
        )
    return HFSloganGenerator(), HFStorylineGenerator(), HFSceneGenerator(), HFVideoGenerator(model_id="Wan-AI/Wan2.2-TI2V-5B"), False


def validate_name(name: str) -> bool:
    return bool(name and name.strip())


def build_profile_summary(profile: CustomerProfile) -> str:
    return (
        f"Name: {profile.name}; Age: {profile.age}; Gender: {profile.gender}; "
        f"Nationality: {profile.nationality}; Language: {profile.language}; Product ID: {profile.product_id}."
    )


def render_pipeline_table() -> None:
    st.markdown(
        f"""
<div class="pipeline-grid">
  <table>
    <thead>
      <tr>
        <th>Pipeline</th>
        <th>Task Type</th>
        <th>Hugging Face Model</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>{PIPELINE_NAMES[1]}</td><td>Text Generation</td><td>{PIPELINE_MODELS[1]}</td></tr>
      <tr><td>{PIPELINE_NAMES[2]}</td><td>Text Generation</td><td>{PIPELINE_MODELS[2]}</td></tr>
      <tr><td>{PIPELINE_NAMES[3]}</td><td>Text-to-Image Generation (3 Images)</td><td>{PIPELINE_MODELS[3]}</td></tr>
      <tr><td>{PIPELINE_NAMES[4]}</td><td>Image+Text-to-Video Generation</td><td>Wan-AI/Wan2.2-TI2V-5B</td></tr>
    </tbody>
  </table>
</div>
""",
        unsafe_allow_html=True,
    )


init_state()
products = load_catalog()
product_names = [item.name for item in products]

st.markdown(
    """
<div class="top-header">
    <h1>AI Smart Marketing: Personalized Video Advertisements</h1>
    <p>This application generates machine learning-driven, personalized promotional videos for Nike marketing campaigns. By leveraging customer profile data from the Nike online store, the app uses chained Hugging Face text and video pipelines to craft a unique storyline, catchy slogan, and detailed storyboard. Ultimately, it produces customized marketing video materials tailored specifically to each individual customer.</p>
</div>
""",
    unsafe_allow_html=True,
)

render_pipeline_table()

with st.sidebar:
    st.header("Customer Profile")
    name = st.text_input("Name", value="")
    age = int(st.number_input("Age", min_value=10, max_value=90, value=25, step=1))
    gender = cast(str, st.selectbox("Gender", ["Male", "Female"]))
    nationality = cast(str, st.selectbox("Nationality", ["Chinese", "USA"]))
    language = cast(str, st.selectbox("Language", ["English", "Traditional Chinese"]))
    selected_product_name = cast(str, st.selectbox("Product", options=product_names))

    if config.FORCE_MOCK_MODE or not config.HF_TOKEN:
        st.warning("HF token missing or mock mode forced: app runs in deterministic mock mode.")

    generate_clicked = st.button("Generate Assets", type="primary", use_container_width=True)

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

        slogan_generator, storyline_generator, scene_generator, video_generator, use_mock = get_backends()

        logs: list[str] = []
        with st.status("Running pipelines...", expanded=True) as status:
            logs.append(f"[1/4] {PIPELINE_NAMES[1]}")
            status_log.code("\n".join(logs), language="text")

            encoded_profile = EncodedProfile(profile_summary=build_profile_summary(customer), embedding=[])
            encoded_product = EncodedProduct(
                product_summary=selected_product.name,
                embedding=[],
                attributes=selected_product.attributes or {},
            )

            slogan = slogan_generator.generate(customer, encoded_profile, selected_product, encoded_product)
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
                    "pipeline_4": "Wan-AI/Wan2.2-TI2V-5B",
                },
                debug_metadata={"runtime_mode": "mock" if use_mock else "huggingface"},
            )
            video_path = video_generator.generate(customer, selected_product, assets)
            assets.video_path = video_path

            progress.progress(100, text="All pipelines complete")
            logs.append("[done] All generated assets are ready")
            status_log.code("\n".join(logs), language="text")
            status.update(label="Generation complete", state="complete")

        st.session_state["assets"] = assets
        st.session_state["video_path"] = assets.video_path
        st.session_state["scene_images"] = scene_images

        with video_slot.container():
            st.markdown(
                """
<div class="result-card full-video">
  <h3>Pipeline 4 Result: Personalized Marketing Video</h3>
  <p><b>Model:</b> Wan-AI/Wan2.2-TI2V-5B</p>
</div>
""",
                unsafe_allow_html=True,
            )
            if assets.video_path:
                st.video(assets.video_path)

assets: MarketingAssets | None = st.session_state.get("assets")
if assets and not generate_clicked:
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
  <p><b>Model:</b> {assets.pipeline_models.get('pipeline_4', 'Wan-AI/Wan2.2-TI2V-5B')}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if assets.video_path:
            st.video(assets.video_path)
