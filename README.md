# Machine Learning for Personalized Marketing

Personalized Nike-style product ads built with chained Hugging Face pipelines and a Streamlit storefront preview.

This Streamlit app generates a personalized Nike-style marketing video experience from a customer profile:
- Name, Age, Gender, Nationality, Language, Product
- Optional customer notes at the data-model level for future extension

The UI mimics a Nike Hong Kong-style online storefront and replaces the hero banner with a generated promotional video when the user clicks Generate Marketing Video.

## Business Objective

This project demonstrates how profile-level personalization can be transformed into targeted, product-specific ad creative:
- The customer profile drives personalization intent.
- A curated Nike shoe catalog anchors product context.
- Separate Hugging Face pipelines generate a slogan, script, and cinematic video asset.

This project is designed for:
- ISOM5240 Deep Learning coursework (Hugging Face-only model stack)
- Portfolio-quality software architecture
- Future production extensibility

## Pipeline Architecture

Core flow (4 stages):
1. Personalized Slogan Generation
2. Personalized Storyline Generation
3. Storyline Scene Generation (Image generation)
4. Cinematic Video Generation and End Card Composition

Logical diagram:

```text
Customer Profile + Selected Nike Product
				|
				v
[Pipeline 1] Personalized Slogan Generation (Text Generation)
				|
				v
[Pipeline 2] Personalized Storyline Generation (Text Generation)
				|
				v
[Pipeline 3] Storyline Scene Generation (Text-to-Image Generation)
				|
				v
[Pipeline 4] Personalized Marketing Video Generation (Image+Text-to-Video Generation)
				|
				v
[Pipeline 4] Video Generator (Wan TI2V prompt + 480p composition)
				|
				v
[Pipeline 5] Final Slogan End Card (moviepy)
				|
				v
Nike-style storefront hero banner video + slogan + script
```

Video behavior:
- Target output resolution is 854x480 by default.
- If the generated clip is not already 480p, it is resized and center-cropped to 480p before saving.
- Every final MP4 appends a 1.5 second branded end card showing `"<SLOGAN>, <Name>"`.
- Video prompts are written to request realistic, cinematic footage of a synthetic person matching the customer profile.

## Hugging Face Models

Default model ids are configurable via `.env`.

- Qwen/Qwen3.5-2B
	- Role: personalized slogan generation (Pipeline 1)
	- Target format: Text Generation

- Qwen/Qwen3.5-2B
	- Role: personalized storyline generation (Pipeline 2)
	- Target format: Text Generation

- F16/z-image-turbo-sda
	- Role: storyline scene generation (Pipeline 3)
	- Target format: Text-to-Image Generation

- Wan-AI/Wan2.2-TI2V-5B
	- Role: personalized marketing video generation target model (Pipeline 4)
	- Target format: Image-to-Video Generation

## Fine-Tuned Model Hook

Pipelines 1 and 2 are designed to be replaced by your own fine-tuned marketing models.

Where to plug in:
- Update SLOGAN_GENERATION_MODEL_ID and SCRIPT_GENERATION_MODEL_ID at the top of `app.py`
- Or securely set environment variables SLOGAN_GENERATION_MODEL_ID and SCRIPT_GENERATION_MODEL_ID in `.env`

Example target model id:
- my-username/nike-marketing-flan-t5-base

## Project Structure

```text
app.py                  Complete logic (Streamlit UI, AI workflows, and data contracts)
requirements.txt        Python dependencies
```

## Run Locally

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Create .env from .env.example and add Hugging Face token

```bash
HF_TOKEN=your_huggingface_token_here
```

3. Launch app

```bash
streamlit run app.py
```

## Streamlit Cloud Notes

This app is optimized for constrained environments:
- Heavy video generation defaults to Hugging Face Inference API mode when token is available.
- If token/video inference is unavailable, app falls back to a generated Ken Burns style MP4 at 480p with a branded end card so the full flow still works.
- Cache usage:
	- @st.cache_data for catalog and deterministic image prep
	- @st.cache_resource for model/client loaders

## UI Behavior

- Sidebar inputs:
	- Name, Age, Gender, Nationality, Language, Product
- Action:
	- Generate Marketing Video
- Main panel:
	- Pipeline overview and live processing log
	- Nike-style storefront layout in the right column
	- Hero banner becomes a generated 480p video with a final slogan frame
	- Displays personalized slogan, headline, and script

## Limitations

- Real TI2V API behavior can vary by hosted endpoint and account capability.
- Inference latency depends on model availability and queue time.
- Current text generation enforces English output for consistency.
- Catalog is curated demo data, not a live Nike commerce feed.
- When hosted video generation is unavailable, the fallback clip cannot fully reproduce synthetic human motion from the prompt.

## Future Work

1. Add multilingual copy generation and language-specific model routing.
2. Enable optional TTS voice-over and audio-video muxing.
3. Add A/B ad prompt experimentation and conversion analytics hooks.
4. Expand product coverage and dynamic catalog ingestion pipeline.