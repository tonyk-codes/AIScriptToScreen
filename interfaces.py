"""Abstract base classes for the AI video generation pipeline.

Each interface represents one stage in the Text -> Storyboard -> Video pipeline.
Swap out any stage by creating a new class that inherits from the relevant ABC.
"""

from abc import ABC, abstractmethod
from typing import List


class ScriptRefiner(ABC):
    """Refines raw input text into a list of structured scene descriptions."""

    @abstractmethod
    def refine(self, raw_text: str) -> List[str]:
        """Take raw text and return an ordered list of scene descriptions.

        Args:
            raw_text: Unstructured story / script text.

        Returns:
            A list of scene description strings, one per scene.
        """


class StoryboardGenerator(ABC):
    """Generates storyboard images from scene descriptions."""

    @abstractmethod
    def generate(self, scenes: List[str]) -> List[str]:
        """Generate one image per scene description.

        Args:
            scenes: Ordered list of scene description strings.

        Returns:
            An ordered list of image file paths or URLs, one per scene.
        """


class VideoGenerator(ABC):
    """Assembles storyboard images into a final video clip."""

    @abstractmethod
    def generate(self, image_paths: List[str]) -> str:
        """Generate a video from an ordered sequence of storyboard images.

        Args:
            image_paths: Ordered list of image file paths or URLs.

        Returns:
            The file path (or URL) of the generated video.
        """
