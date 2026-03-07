# AIScriptToScreen

Transform a story or script into storyboard images and a final video — powered by AI.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Add API keys to a .env file
# OPENAI_API_KEY=...

# 3. Launch the Streamlit app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Project Structure

```
app.py                  # Streamlit front-end (entry point)
interfaces.py           # Abstract base classes for each pipeline stage
mock_implementations.py # Drop-in mock backends (no API keys needed)
config.yaml             # Default model selection
requirements.txt        # Python dependencies
```

## Pipeline Stages

| Stage | Interface | What it does |
|---|---|---|
| Script Refiner | `ScriptRefiner` | Splits raw text into ordered scene descriptions |
| Storyboard Generator | `StoryboardGenerator` | Produces one reference image per scene |
| Video Generator | `VideoGenerator` | Assembles images into a final video clip |

## Adding Real AI Backends

1. Create a class in a new file that inherits from the relevant interface in `interfaces.py`.
2. Register it in the corresponding `_*` registry dict at the top of `app.py`.
3. Select it from the sidebar dropdown in the app.