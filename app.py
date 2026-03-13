"""Streamlit front-end for the Smart IKEA Assembly Assistant."""

import os
import streamlit as st
from dotenv import load_dotenv

from interfaces import AssemblyStep
from hf_pipelines import LocalExtractor, LocalWriter, LocalNarration, LocalAnimation, APIExtractor, APIWriter, APINarration, APIAnimation
import mock_implementations

load_dotenv()

st.set_page_config(page_title="Smart IKEA Assembly Assistant", layout="wide")

st.markdown("""
<style>
/* CSS from old app.py simplified */
.stApp { background: #0e1117; color: #fff; }
.card { background: #1a1c23; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #333; }
.title { font-size: 1.5rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

_EXTRACTORS = {"mock": mock_implementations.MockExtractor, "hf_local": LocalExtractor, "hf_api": APIExtractor}
_WRITERS = {"mock": mock_implementations.MockWriter, "hf_local": LocalWriter, "hf_api": APIWriter}
_NARRATORS = {"mock": mock_implementations.MockNarration, "hf_local": LocalNarration, "hf_api": APINarration}
_ANIMATORS = {"mock": mock_implementations.MockAnimation, "hf_local": LocalAnimation, "hf_api": APIAnimation}

with st.sidebar:
    st.title("Setup")
    pdf_file = st.file_uploader("Upload IKEA Assembly PDF", type=["pdf"])
    url_input = st.text_input("IKEA Product URL (optional)")
    user_notes = st.text_area("User Notes (optional)")
    hf_token = st.text_input("Hugging Face Profile Token", type="password")
    
    if hf_token:
        st.session_state["hf_token"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

    st.subheader("Model Settings")
    extractor_choice = st.selectbox("Extractor Backend", list(_EXTRACTORS.keys()))
    writer_choice = st.selectbox("Writer Backend", list(_WRITERS.keys()))
    narrator_choice = st.selectbox("TTS Backend", list(_NARRATORS.keys()))
    animator_choice = st.selectbox("Video Backend", list(_ANIMATORS.keys()))

st.title("Smart IKEA Assembly Assistant")
st.markdown("Convert assembly PDF manuals into an annotated, narrated timeline using Hugging Face AI pipelines.")

if st.button("Generate Assembly Guide", type="primary"):
    if not pdf_file and extractor_choice != "mock":
        st.warning("Please upload a PDF file.")
        st.stop()

    if pdf_file:
        pdf_path = f"target_{pdf_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
    else:
        pdf_path = "mock.pdf"

    extractor = _EXTRACTORS[extractor_choice]()
    writer = _WRITERS[writer_choice]()
    narrator = _NARRATORS[narrator_choice]()
    animator = _ANIMATORS[animator_choice]()

    with st.status("Running Assembly AI Pipeline...", expanded=True) as status:
        st.write("Extracting and captioning panels (Image-to-Text)...")
        steps = extractor.extract(pdf_path)
        
        st.write("Generating step-by-step instructions (Text-to-Text)...")
        steps = writer.write(steps)
        
        st.write("Synthesizing narration voices (Text-to-Speech)...")
        steps = narrator.generate(steps)
        
        st.write("Creating visual timeline animations (Image-to-Video)...")
        steps = animator.generate(steps)
        status.update(label="Pipeline processing complete!", state="complete")

    st.session_state["assembly_steps"] = steps

if "assembly_steps" in st.session_state:
    st.subheader("Assembly Timeline")
    for step in st.session_state["assembly_steps"]:
        with st.container():
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="title">Step {step.step_id}: {step.title}</div>', unsafe_allow_html=True)
            
            cols = st.columns([1, 2])
            with cols[0]:
                for pth in step.image_paths:
                    if os.path.exists(pth):
                        st.image(pth)
                    else:
                        st.info(f"Mock Image Path: {pth}")
                        
                if step.video_path:
                    if os.path.exists(step.video_path):
                        st.video(step.video_path)
                    else:
                        st.info(f"Mock Video Path: {step.video_path}")

            with cols[1]:
                st.markdown(f"**Action detected:** {step.raw_caption}")
                st.markdown(f"**Instruction:** {step.detailed_instruction}")
                
                if step.audio_path:
                    if os.path.exists(step.audio_path):
                        st.audio(step.audio_path)
                    else:
                        st.info(f"Mock Audio Path: {step.audio_path}")
            st.markdown('</div>', unsafe_allow_html=True)
