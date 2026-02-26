"""Mock implementations of the pipeline interfaces.

These classes return deterministic placeholder data so that the full pipeline
can be exercised immediately, without any real API keys or model endpoints.
Replace any mock with a real implementation when you are ready to call an
actual AI service.
"""

from typing import List

from interfaces import ScriptRefiner, StoryboardGenerator, VideoGenerator


class MockScriptRefiner(ScriptRefiner):
    """Returns a fixed list of scene descriptions derived from the input text.

    The mock splits the raw text on sentence-ending punctuation and wraps
    each chunk in a minimal scene template, giving realistic-looking output
    without calling any external service.
    """

    def refine(self, raw_text: str) -> List[str]:
        import re

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_text) if s.strip()]
        if not sentences:
            sentences = [raw_text.strip()]
        return [f"[Scene {i + 1}] {sentence}" for i, sentence in enumerate(sentences)]


class MockStoryboardGen(StoryboardGenerator):
    """Returns placeholder image paths instead of generating real images.

    Each returned path follows the pattern ``mock_image_<n>.png`` so
    downstream stages receive a believable list of file references.
    """

    def generate(self, scenes: List[str]) -> List[str]:
        return [f"mock_image_{i + 1}.png" for i in range(len(scenes))]


class MockVideoGenerator(VideoGenerator):
    """Returns a placeholder video path instead of rendering a real video."""

    def generate(self, image_paths: List[str]) -> str:
        return "mock_output_video.mp4"
