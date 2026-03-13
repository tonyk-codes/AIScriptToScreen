import os
import streamlit as st
from PIL import Image
from interfaces import AssemblyStep, InstructionExtractor, InstructionWriter, NarrationGenerator, AnimationGenerator
import json

class BasePipeline:
    def get_token(self):
        return os.environ.get("HF_TOKEN")

#
# Local Transformers/Diffusers Implementations
#
class LocalExtractor(BasePipeline, InstructionExtractor):
    def extract(self, pdf_path: str) -> list[AssemblyStep]:
        # Implementation of PDF to image and then local VQA/captioning model
        from transformers import pipeline
        import fitz
        
        captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        
        doc = fitz.open(pdf_path)
        steps = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            caption = captioner(img)[0]['generated_text']
            
            steps.append(AssemblyStep(
                step_id=i+1, title=f"Action on Page {i+1}", raw_caption=caption,
                detailed_instruction=None, source_pages=[i], image_paths=[f"page_{i}.png"]
            ))
            
            img.save(f"page_{i}.png")
        return steps

class LocalWriter(BasePipeline, InstructionWriter):
    def write(self, steps: list[AssemblyStep]) -> list[AssemblyStep]:
        from transformers import pipeline
        
        # TODO: Replace 'google/flan-t5-base' with your custom fine-tuned model ID (e.g., 'your-username/ikea-instructions')
        generator = pipeline("text2text-generation", model="google/flan-t5-base")
        
        for step in steps:
            prompt = f"You are an IKEA assembly expert. Turn this caption into a clear, step-by-step instruction for a beginner: {step.raw_caption}"
            result = generator(prompt, max_length=150)
            step.detailed_instruction = result[0]['generated_text']
        return steps

class LocalNarration(BasePipeline, NarrationGenerator):
    def generate(self, steps: list[AssemblyStep]) -> list[AssemblyStep]:
        from transformers import VitsModel, AutoTokenizer
        import torch
        import scipy.io.wavfile

        model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        
        for step in steps:
            inputs = tokenizer(step.detailed_instruction, return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs).waveform
            path = f"audio_{step.step_id}.wav"
            scipy.io.wavfile.write(path, rate=model.config.sampling_rate, data=output[0].numpy())
            step.audio_path = path

        return steps

class LocalAnimation(BasePipeline, AnimationGenerator):
    def generate(self, steps: list[AssemblyStep]) -> list[AssemblyStep]:
        # Implement a stub or lightweight Image2Video pipeline
        for step in steps:
            step.video_path = f"animation_{step.step_id}.mp4"
        return steps

#
# Inference API Implementations
#
class APIExtractor(BasePipeline, InstructionExtractor):
    def extract(self, pdf_path: str) -> list[AssemblyStep]:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model="Salesforce/blip-image-captioning-base", token=self.get_token())
        # Processing PDF for inference API call
        return []

class APIWriter(BasePipeline, InstructionWriter):
    def write(self, steps: list[AssemblyStep]) -> list[AssemblyStep]:
        from huggingface_hub import InferenceClient
        # TODO: Replace 'google/flan-t5-base' with your custom fine-tuned model ID
        client = InferenceClient(model="google/flan-t5-base", token=self.get_token())
        for step in steps:
            prompt = f"You are an expert. Explain this: {step.raw_caption}"
            step.detailed_instruction = client.text_generation(prompt, max_new_tokens=150)
        return steps

class APINarration(BasePipeline, NarrationGenerator):
    def generate(self, steps: list[AssemblyStep]) -> list[AssemblyStep]:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model="facebook/mms-tts-eng", token=self.get_token())
        for step in steps:
            audio = client.text_to_speech(step.detailed_instruction)
            path = f"audio_{step.step_id}.wav"
            with open(path, "wb") as f:
                f.write(audio)
            step.audio_path = path
        return steps

class APIAnimation(BasePipeline, AnimationGenerator):
    def generate(self, steps: list[AssemblyStep]) -> list[AssemblyStep]:
        # Typically Stability SVD image-to-video inference
        return steps
