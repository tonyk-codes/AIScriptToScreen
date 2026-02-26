"""Streamlit front-end for the AI Script-to-Screen pipeline.

Run with:
    streamlit run app.py
"""

import importlib
import os

import streamlit as st
from dotenv import load_dotenv

from interfaces import ScriptRefiner, StoryboardGenerator, VideoGenerator

# Load .env so API keys are available before any model class is instantiated.
load_dotenv()

# ---------------------------------------------------------------------------
# Model registry – add new backends here as you integrate real AI services.
# ---------------------------------------------------------------------------

_SCRIPT_REFINERS: dict[str, tuple[str, str]] = {
    "mock": ("mock_implementations", "MockScriptRefiner"),
}

_STORYBOARD_GENERATORS: dict[str, tuple[str, str]] = {
    "mock": ("mock_implementations", "MockStoryboardGen"),
}

_VIDEO_GENERATORS: dict[str, tuple[str, str]] = {
    "mock": ("mock_implementations", "MockVideoGenerator"),
}


def _load_class(registry: dict, model_name: str):
    """Import and return the class registered under *model_name*."""
    if model_name not in registry:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(registry.keys())}"
        )
    module_name, class_name = registry[model_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Script to Screen",
    page_icon="🎬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar – pipeline configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")
    st.divider()

    st.subheader("Pipeline Models")
    script_model = st.selectbox(
        "Script Refiner",
        list(_SCRIPT_REFINERS.keys()),
        help="Model used to split raw text into structured scene descriptions.",
    )
    storyboard_model = st.selectbox(
        "Storyboard Generator",
        list(_STORYBOARD_GENERATORS.keys()),
        help="Model used to generate a visual reference image for each scene.",
    )
    video_model = st.selectbox(
        "Video Generator",
        list(_VIDEO_GENERATORS.keys()),
        help="Model used to assemble the storyboard images into a video.",
    )

    st.divider()
    st.caption("Add new AI backends by registering them in the registry dicts inside `app.py`.")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("🎬 AI Script to Screen")
st.caption(
    "Transform your story or script into storyboard images and a final video — powered by AI."
)
st.divider()

raw_text = st.text_area(
    "✍️ Enter your script or story",
    placeholder=(
        "A robot explores a jungle. "
        "It discovers an ancient ruin. "
        "The robot makes contact with alien life."
    ),
    height=200,
)

run_btn = st.button("🚀 Run Pipeline", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if run_btn:
    if not raw_text.strip():
        st.warning("Please enter some text before running the pipeline.")
        st.stop()

    # Instantiate the selected backends.
    assert script_model is not None
    assert storyboard_model is not None
    assert video_model is not None
    refiner: ScriptRefiner = _load_class(_SCRIPT_REFINERS, script_model)()
    storyboard_gen: StoryboardGenerator = _load_class(_STORYBOARD_GENERATORS, storyboard_model)()
    video_gen: VideoGenerator = _load_class(_VIDEO_GENERATORS, video_model)()

    # ------------------------------------------------------------------
    # Step 1 – Script refinement
    # ------------------------------------------------------------------
    with st.status("Step 1 — Refining script…", expanded=True) as step1:
        scenes = refiner.refine(raw_text)
        step1.update(
            label=f"✅ Step 1 — Script refined into **{len(scenes)}** scene(s)",
            state="complete",
        )

    st.subheader("📝 Scenes")
    for i, scene in enumerate(scenes, 1):
        with st.expander(f"Scene {i}", expanded=True):
            st.write(scene)

    st.divider()

    # ------------------------------------------------------------------
    # Step 2 – Storyboard generation
    # ------------------------------------------------------------------
    with st.status("Step 2 — Generating storyboard…", expanded=True) as step2:
        image_paths = storyboard_gen.generate(scenes)
        step2.update(
            label=f"✅ Step 2 — **{len(image_paths)}** storyboard image(s) generated",
            state="complete",
        )

    st.subheader("🖼️ Storyboard")
    if image_paths:
        num_cols = min(len(image_paths), 4)
        cols = st.columns(num_cols)
        for idx, path in enumerate(image_paths):
            with cols[idx % num_cols]:
                st.caption(f"Scene {idx + 1}")
                if os.path.exists(path):
                    st.image(path, use_column_width="auto")
                else:
                    st.info(f"`{path}`\n\n*(placeholder – no file on disk)*")
    else:
        st.info("No storyboard images were produced.")

    st.divider()

    # ------------------------------------------------------------------
    # Step 3 – Video generation
    # ------------------------------------------------------------------
    with st.status("Step 3 — Generating video…", expanded=True) as step3:
        video_path = video_gen.generate(image_paths)
        step3.update(
            label=f"✅ Step 3 — Video generated: `{video_path}`",
            state="complete",
        )

    st.subheader("🎥 Output Video")
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.success(
            f"Pipeline complete! Output: `{video_path}`\n\n"
            "*(Replace the mock backends with real AI services to produce an actual video file.)*"
        )
