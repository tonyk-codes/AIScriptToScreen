"""Tests for the AI Script-to-Screen pipeline."""

import sys
import os
import pytest

# Ensure the project root is on the path so imports resolve correctly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from interfaces import ScriptRefiner, StoryboardGenerator, VideoGenerator
from mock_implementations import MockScriptRefiner, MockStoryboardGen, MockVideoGenerator
from main import Pipeline


# ---------------------------------------------------------------------------
# Interface contract tests
# ---------------------------------------------------------------------------


def test_interfaces_are_abstract():
    """All three ABCs must not be directly instantiable."""
    with pytest.raises(TypeError):
        ScriptRefiner()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        StoryboardGenerator()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        VideoGenerator()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Mock implementation unit tests
# ---------------------------------------------------------------------------


class TestMockScriptRefiner:
    def test_returns_list(self):
        refiner = MockScriptRefiner()
        result = refiner.refine("A hero walks into a bar.")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_scene_labels(self):
        refiner = MockScriptRefiner()
        result = refiner.refine("First sentence. Second sentence.")
        assert result[0].startswith("[Scene 1]")
        assert result[1].startswith("[Scene 2]")

    def test_empty_string_handled(self):
        refiner = MockScriptRefiner()
        result = refiner.refine("")
        # Should return at least one (possibly empty) scene without crashing.
        assert isinstance(result, list)

    def test_single_sentence(self):
        refiner = MockScriptRefiner()
        result = refiner.refine("Just one sentence.")
        assert len(result) == 1
        assert "Just one sentence." in result[0]


class TestMockStoryboardGen:
    def test_returns_one_image_per_scene(self):
        gen = MockStoryboardGen()
        scenes = ["Scene 1", "Scene 2", "Scene 3"]
        images = gen.generate(scenes)
        assert len(images) == len(scenes)

    def test_image_paths_are_strings(self):
        gen = MockStoryboardGen()
        images = gen.generate(["A scene"])
        assert all(isinstance(p, str) for p in images)

    def test_empty_scenes(self):
        gen = MockStoryboardGen()
        images = gen.generate([])
        assert images == []

    def test_image_naming_pattern(self):
        gen = MockStoryboardGen()
        images = gen.generate(["s1", "s2"])
        assert images[0] == "mock_image_1.png"
        assert images[1] == "mock_image_2.png"


class TestMockVideoGenerator:
    def test_returns_string(self):
        gen = MockVideoGenerator()
        result = gen.generate(["img1.png", "img2.png"])
        assert isinstance(result, str)

    def test_returns_mp4(self):
        gen = MockVideoGenerator()
        result = gen.generate(["img1.png"])
        assert result.endswith(".mp4")

    def test_empty_input(self):
        gen = MockVideoGenerator()
        result = gen.generate([])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Pipeline integration test
# ---------------------------------------------------------------------------


class TestPipeline:
    def _make_pipeline(self):
        return Pipeline(MockScriptRefiner(), MockStoryboardGen(), MockVideoGenerator())

    def test_run_returns_video_path(self):
        pipeline = self._make_pipeline()
        result = pipeline.run("A robot explores a jungle.")
        assert isinstance(result, str)
        assert result.endswith(".mp4")

    def test_run_multi_sentence(self):
        pipeline = self._make_pipeline()
        result = pipeline.run("First scene. Second scene. Third scene.")
        assert isinstance(result, str)

    def test_from_config(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            "script_refiner:\n  model: mock\n"
            "storyboard_generator:\n  model: mock\n"
            "video_generator:\n  model: mock\n"
        )
        pipeline = Pipeline.from_config(str(config))
        assert isinstance(pipeline, Pipeline)
        result = pipeline.run("Test input.")
        assert isinstance(result, str)

    def test_from_config_unknown_model_raises(self, tmp_path):
        config = tmp_path / "bad_config.yaml"
        config.write_text(
            "script_refiner:\n  model: nonexistent\n"
            "storyboard_generator:\n  model: mock\n"
            "video_generator:\n  model: mock\n"
        )
        with pytest.raises(ValueError, match="Unknown model"):
            Pipeline.from_config(str(config))
