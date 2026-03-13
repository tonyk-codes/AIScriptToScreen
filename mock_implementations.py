import os
from PIL import Image
from interfaces import AssemblyStep

class MockExtractor:
    def extract(self, pdf_path: str) -> list[AssemblyStep]:
        return [
            AssemblyStep(
                step_id=1,
                title="Mock Step 1",
                raw_caption="insert wooden dowels into the side panel",
                detailed_instruction=None,
                source_pages=[1],
                image_paths=["mock_panel_1.jpg"]
            ),
            AssemblyStep(
                step_id=2,
                title="Mock Step 2",
                raw_caption="attach top board using cam locks",
                detailed_instruction=None,
                source_pages=[2],
                image_paths=["mock_panel_2.jpg"]
            )
        ]

class MockWriter:
    def write(self, steps: list[AssemblyStep]) -> list[AssemblyStep]:
        for step in steps:
            step.detailed_instruction = f"Detailed: {step.raw_caption} carefully."
        return steps

class MockNarration:
    def generate(self, steps: list[AssemblyStep]) -> list[AssemblyStep]:
        for step in steps:
            step.audio_path = "mock_audio.wav"
        return steps

class MockAnimation:
    def generate(self, steps: list[AssemblyStep]) -> list[AssemblyStep]:
        for step in steps:
            step.video_path = "mock_video.mp4"
        return steps
