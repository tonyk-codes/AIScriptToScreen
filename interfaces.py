from dataclasses import dataclass
from typing import Protocol

@dataclass
class AssemblyStep:
    step_id: int
    title: str
    raw_caption: str
    detailed_instruction: str | None
    source_pages: list[int]
    image_paths: list[str]
    audio_path: str | None = None
    video_path: str | None = None

class InstructionExtractor(Protocol):
    def extract(self, pdf_path: str) -> list[AssemblyStep]:
        ...

class InstructionWriter(Protocol):
    def write(self, steps: list[AssemblyStep]) -> list[AssemblyStep]:
        ...

class NarrationGenerator(Protocol):
    def generate(self, steps: list[AssemblyStep]) -> list[AssemblyStep]:
        ...

class AnimationGenerator(Protocol):
    def generate(self, steps: list[AssemblyStep]) -> list[AssemblyStep]:
        ...
