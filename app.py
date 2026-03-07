"""Streamlit front-end for the AI storyboard and animation pipeline.

Run with:
    streamlit run app.py
"""

import importlib
import html
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


def _inject_styles() -> None:
    """Apply a more polished visual treatment to the Streamlit app."""
    st.markdown(
        """
        <style>
            :root {
                --surface: rgba(18, 23, 31, 0.9);
                --surface-strong: rgba(12, 16, 23, 0.96);
                --surface-soft: rgba(28, 35, 46, 0.78);
                --border: rgba(159, 178, 202, 0.16);
                --text: #edf3fb;
                --muted: #9eafc4;
                --accent: #6ab7ff;
                --accent-soft: rgba(106, 183, 255, 0.14);
                --accent-strong: #8fd0ff;
                --success: #7dd6b4;
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(76, 148, 255, 0.20), transparent 24%),
                    radial-gradient(circle at left top, rgba(34, 209, 238, 0.10), transparent 18%),
                    linear-gradient(180deg, #081019 0%, #0d1520 42%, #101824 100%);
                color: var(--text);
            }

            [data-testid="stAppViewContainer"] .main .block-container {
                max-width: 1180px;
                padding-top: 2.4rem;
                padding-bottom: 2.5rem;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(7, 11, 17, 0.98) 0%, rgba(12, 18, 28, 0.98) 100%);
                border-right: 1px solid rgba(143, 208, 255, 0.12);
            }

            [data-testid="stSidebar"] * {
                color: var(--text);
            }

            [data-testid="stSidebar"] .stSelectbox label,
            [data-testid="stSidebar"] .stMarkdown,
            [data-testid="stSidebar"] .stCaption {
                color: var(--muted);
            }

            .hero-shell,
            .panel,
            .scene-card,
            .stat-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 22px;
                box-shadow: 0 20px 55px rgba(2, 8, 18, 0.45);
                backdrop-filter: blur(10px);
            }

            .hero-shell {
                background:
                    linear-gradient(135deg, rgba(23, 30, 42, 0.96) 0%, rgba(13, 18, 27, 0.96) 100%);
                padding: 2.2rem 2.2rem;
                margin-bottom: 1.3rem;
            }

            .hero-kicker {
                display: inline-block;
                font-size: 0.72rem;
                letter-spacing: 0.16em;
                text-transform: uppercase;
                font-weight: 700;
                color: var(--accent);
                background: var(--accent-soft);
                border-radius: 999px;
                padding: 0.4rem 0.7rem;
                margin-bottom: 1rem;
            }

            .hero-title {
                font-size: clamp(2.2rem, 4vw, 3.5rem);
                line-height: 1.02;
                font-weight: 700;
                margin: 0;
                color: var(--text);
                letter-spacing: -0.04em;
            }

            .hero-copy,
            .section-copy,
            .panel-copy,
            .muted-text {
                color: var(--muted);
                font-size: 1rem;
                line-height: 1.65;
            }

            .panel {
                padding: 1.35rem 1.4rem;
                margin-bottom: 1rem;
            }

            .panel-title,
            .section-title {
                font-size: 1.08rem;
                font-weight: 650;
                margin: 0 0 0.45rem 0;
                color: var(--text);
                letter-spacing: -0.02em;
            }

            .mini-list {
                margin: 0.85rem 0 0 0;
                padding: 0;
                list-style: none;
            }

            .mini-list li {
                margin: 0.45rem 0;
                color: var(--muted);
                padding-left: 1rem;
                position: relative;
            }

            .mini-list li::before {
                content: "";
                position: absolute;
                left: 0;
                top: 0.7rem;
                width: 0.42rem;
                height: 0.42rem;
                border-radius: 999px;
                background: var(--accent);
                box-shadow: 0 0 12px rgba(106, 183, 255, 0.45);
            }

            .stat-card {
                padding: 1rem 1.1rem;
                height: 100%;
                background: linear-gradient(180deg, rgba(23, 30, 42, 0.92) 0%, rgba(16, 22, 32, 0.92) 100%);
            }

            .stat-label {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: var(--muted);
                margin-bottom: 0.45rem;
            }

            .stat-value {
                font-size: 1.55rem;
                font-weight: 700;
                color: var(--text);
                letter-spacing: -0.03em;
            }

            .scene-card {
                padding: 1.1rem 1.2rem;
                margin-bottom: 0.9rem;
            }

            .scene-label {
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                color: var(--accent);
                margin-bottom: 0.45rem;
                font-weight: 700;
            }

            .stButton > button {
                border-radius: 999px;
                border: 1px solid transparent;
                background: linear-gradient(135deg, #3f90e8 0%, #1b6fd1 100%);
                color: #eef7ff;
                font-weight: 650;
                min-height: 3rem;
                box-shadow: 0 14px 32px rgba(27, 111, 209, 0.32);
            }

            .stButton > button:hover {
                border-color: rgba(143, 208, 255, 0.18);
                background: linear-gradient(135deg, #4a9bf2 0%, #2a7de0 100%);
            }

            .stTextArea textarea,
            .stSelectbox [data-baseweb="select"] > div {
                border-radius: 16px;
            }

            .stTextArea textarea {
                background: rgba(9, 14, 21, 0.88);
                color: var(--text);
                border: 1px solid rgba(143, 208, 255, 0.14);
            }

            .stTextArea textarea::placeholder {
                color: #73859b;
            }

            .stSelectbox [data-baseweb="select"] > div,
            [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
                background: rgba(10, 14, 21, 0.9);
                border: 1px solid rgba(143, 208, 255, 0.14);
                color: var(--text);
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
            }

            .stTabs [data-baseweb="tab"] {
                background: rgba(255, 255, 255, 0.06);
                border-radius: 999px;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            [data-testid="stStatusWidget"] {
                background: rgba(10, 15, 22, 0.82);
                border: 1px solid rgba(143, 208, 255, 0.14);
            }

            [data-testid="stAlert"] {
                background: rgba(16, 23, 34, 0.94);
                color: var(--text);
                border: 1px solid rgba(143, 208, 255, 0.12);
            }

            hr {
                border-color: rgba(143, 208, 255, 0.10);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Storyboard Studio",
    layout="wide",
)

_inject_styles()

# ---------------------------------------------------------------------------
# Sidebar – pipeline configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Production Setup")
    st.divider()

    st.subheader("Pipeline Engines")
    script_model = st.selectbox(
        "Refiner Engine",
        list(_SCRIPT_REFINERS.keys()),
        help="Engine used to convert a topic brief into structured scene and script-ready refinements.",
    )
    storyboard_model = st.selectbox(
        "Storyboard Engine",
        list(_STORYBOARD_GENERATORS.keys()),
        help="Engine used to generate sketch frames and storyboard material for the workbook package.",
    )
    video_model = st.selectbox(
        "Animation Engine",
        list(_VIDEO_GENERATORS.keys()),
        help="Engine used to render the final animation with voice, sound design, and music layers.",
    )

    st.divider()
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Deliverable Scope</div>
            <div class="panel-copy">
                This workspace is structured for topic intake, narrative refinement, storyboard workbook output,
                and final animation delivery.
            </div>
            <ul class="mini-list">
                <li>Input: topic with supporting description</li>
                <li>Storyboard workbook: script, shot notes, sketches, and frame descriptions</li>
                <li>Final render: animation with dialogue, effects, ambience, and music</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown(
    """
    <section class="hero-shell">
        <div class="hero-kicker">Pre-Production to Final Render</div>
        <h1 class="hero-title">AI Storyboard and Animation Studio</h1>
        <p class="hero-copy">
            Start from a topic and short description, refine it into structured scenes and script notes,
            assemble storyboard sketches into an Excel workbook, and prepare the final animation output
            with sound, voice, and music direction.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

intro_col, workflow_col = st.columns([1.35, 1], gap="large")

with intro_col:
    st.markdown(
        """
        <div class="panel">
            <div class="section-title">Input Brief</div>
            <div class="section-copy">
                Enter a topic with several sentences of description. The pipeline is framed around creative
                development for storyboard production, workbook export, and final animation packaging.
            </div>
            <ul class="mini-list">
                <li>Refiner output: clearer story structure, shot intent, and script-ready scene breakdown</li>
                <li>Storyboard output: sketches and frame descriptions for an Excel storyboard file</li>
                <li>Final output: animation render intended for sound design, dialogue, and background music</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with workflow_col:
    stage_a, stage_b, stage_c, stage_d = st.columns(4, gap="small")
    with stage_a:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Input</div>
                <div class="stat-value">Topic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stage_b:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Stage 1</div>
                <div class="stat-value">Refiner</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stage_c:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Stage 2</div>
                <div class="stat-value">Storyboard</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stage_d:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Stage 3</div>
                <div class="stat-value">Animation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

raw_text = st.text_area(
    "Topic and Description",
    placeholder=(
        "Topic: A young inventor builds a rescue drone for a flooded town. "
        "Description: The story should feel hopeful and cinematic, with clear escalation, "
        "emotional beats, and visuals that can be translated into storyboard sketches and a final animation."
    ),
    height=200,
)

run_btn = st.button("Generate Production Package", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if run_btn:
    if not raw_text.strip():
        st.warning("Please enter a topic and description before running the pipeline.")
        st.stop()

    # Instantiate the selected backends.
    assert script_model is not None
    assert storyboard_model is not None
    assert video_model is not None
    refiner: ScriptRefiner = _load_class(_SCRIPT_REFINERS, script_model)()
    storyboard_gen: StoryboardGenerator = _load_class(_STORYBOARD_GENERATORS, storyboard_model)()
    video_gen: VideoGenerator = _load_class(_VIDEO_GENERATORS, video_model)()

    # ------------------------------------------------------------------
    # Step 1 – Narrative refinement
    # ------------------------------------------------------------------
    with st.status("Step 1: Refining topic into structured scenes and script notes", expanded=True) as step1:
        scenes = refiner.refine(raw_text)
        step1.update(
            label=f"Step 1 complete: refiner produced {len(scenes)} structured scene(s)",
            state="complete",
        )

    summary_cols = st.columns(3, gap="small")
    with summary_cols[0]:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Refined Scenes</div>
                <div class="stat-value">{len(scenes)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="panel">
            <div class="section-title">Refined Narrative Breakdown</div>
            <div class="section-copy">
                The refiner stage converts the original topic brief into clearer scene units that can feed
                storyboard descriptions, shot planning, and script development.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for i, scene in enumerate(scenes, 1):
        scene_text = html.escape(scene).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="scene-card">
                <div class="scene-label">Scene {i:02d}</div>
                <div>{scene_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ------------------------------------------------------------------
    # Step 2 – Storyboard asset generation
    # ------------------------------------------------------------------
    with st.status("Step 2: Generating sketches and storyboard workbook assets", expanded=True) as step2:
        image_paths = storyboard_gen.generate(scenes)
        step2.update(
            label=f"Step 2 complete: {len(image_paths)} storyboard sketch asset(s) prepared",
            state="complete",
        )

    with summary_cols[1]:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Sketch Frames</div>
                <div class="stat-value">{len(image_paths)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="panel">
            <div class="section-title">Storyboard Workbook (分鏡)</div>
            <div class="section-copy">
                This stage is framed to build an Excel storyboard file containing shot descriptions,
                sketches, and script lines for each frame. The current mock backend produces sketch placeholders,
                but the UI now reflects the intended workbook deliverable.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if image_paths:
        num_cols = min(len(image_paths), 4)
        cols = st.columns(num_cols)
        for idx, path in enumerate(image_paths):
            with cols[idx % num_cols]:
                st.caption(f"Sketch {idx + 1}")
                if os.path.exists(path):
                    st.image(path, use_column_width="auto")
                else:
                    st.info(f"`{path}`\n\n*(placeholder sketch asset – no file on disk)*")
    else:
        st.info("No storyboard sketch assets were produced.")

    st.divider()

    # ------------------------------------------------------------------
    # Step 3 – Animation rendering
    # ------------------------------------------------------------------
    with st.status("Step 3: Rendering final animation and sound package", expanded=True) as step3:
        video_path = video_gen.generate(image_paths)
        step3.update(
            label=f"Step 3 complete: animation output generated at {video_path}",
            state="complete",
        )

    with summary_cols[2]:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Final Delivery</div>
                <div class="stat-value">Ready</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="panel">
            <div class="section-title">Final Animation Output</div>
            <div class="section-copy">
                The final delivery target is an animation render that can include dialogue, sound effects,
                ambience, and background music. The current mock backend returns a placeholder video path.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Animation Preview")
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.success(
            f"Pipeline complete! Output: `{video_path}`\n\n"
            "*(Replace the mock backends with real AI services to produce the storyboard workbook and final animation assets.)*"
        )
