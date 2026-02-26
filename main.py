"""Pipeline orchestrator for the AI video generation system.

Usage
-----
Run with the default config:

    python main.py --text "A robot explores a jungle."

Or specify a custom config file:

    python main.py --config my_config.yaml --text "A hero saves the day."

You can also drive the pipeline programmatically::

    from main import Pipeline
    pipeline = Pipeline.from_config("config.yaml")
    result = pipeline.run("Once upon a time …")
    print(result)
"""

import argparse
import importlib
import os
from typing import Optional

import yaml
from dotenv import load_dotenv

from interfaces import ScriptRefiner, StoryboardGenerator, VideoGenerator

# Load environment variables from .env (if present) so API keys are available
# before any model class is instantiated.
load_dotenv()

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Maps the string model name (as written in config.yaml) to the Python class
# that implements the corresponding interface.  Add a new entry here whenever
# you integrate a new AI backend.
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
            f"Unknown model '{model_name}'. Available: {list(registry.keys())}"
        )
    module_name, class_name = registry[model_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Chains ScriptRefiner -> StoryboardGenerator -> VideoGenerator."""

    def __init__(
        self,
        script_refiner: ScriptRefiner,
        storyboard_generator: StoryboardGenerator,
        video_generator: VideoGenerator,
    ) -> None:
        self.script_refiner = script_refiner
        self.storyboard_generator = storyboard_generator
        self.video_generator = video_generator

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "Pipeline":
        """Build a Pipeline from a YAML configuration file.

        The YAML file must contain top-level keys ``script_refiner``,
        ``storyboard_generator``, and ``video_generator``, each with at
        least a ``model`` sub-key, e.g.::

            script_refiner:
              model: mock
            storyboard_generator:
              model: mock
            video_generator:
              model: mock
        """
        with open(config_path, "r") as fh:
            config = yaml.safe_load(fh)

        refiner_cls = _load_class(
            _SCRIPT_REFINERS, config["script_refiner"]["model"]
        )
        storyboard_cls = _load_class(
            _STORYBOARD_GENERATORS, config["storyboard_generator"]["model"]
        )
        video_cls = _load_class(
            _VIDEO_GENERATORS, config["video_generator"]["model"]
        )

        return cls(refiner_cls(), storyboard_cls(), video_cls())

    def run(self, raw_text: str) -> str:
        """Execute the full pipeline and return the path to the final video.

        Steps:
        1. Refine the raw text into scene descriptions.
        2. Generate a storyboard image for each scene.
        3. Assemble the images into a video.

        Args:
            raw_text: The unstructured story / script to process.

        Returns:
            The file path (or URL) of the generated video.
        """
        print("[Pipeline] Step 1 – Refining script …")
        scenes = self.script_refiner.refine(raw_text)
        print(f"[Pipeline]   → {len(scenes)} scene(s) produced")
        for i, scene in enumerate(scenes, 1):
            print(f"[Pipeline]     Scene {i}: {scene}")

        print("[Pipeline] Step 2 – Generating storyboard …")
        image_paths = self.storyboard_generator.generate(scenes)
        print(f"[Pipeline]   → {len(image_paths)} image(s): {image_paths}")

        print("[Pipeline] Step 3 – Generating video …")
        video_path = self.video_generator.generate(image_paths)
        print(f"[Pipeline]   → Video: {video_path}")

        return video_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Script-to-Screen pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="The raw story / script text to process",
    )
    return parser


def main(argv: Optional[list] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    pipeline = Pipeline.from_config(args.config)
    video = pipeline.run(args.text)
    print(f"\n✅  Done! Output video: {video}")


if __name__ == "__main__":
    main()
