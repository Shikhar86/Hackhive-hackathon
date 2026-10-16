"""Offline image captioning for surveillance event keyframes."""

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.config import DEVICE


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


@lru_cache(maxsize=1)
def _load_model():
    """Load the already-cached model once; never access the network."""
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf-cache"))

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).to(DEVICE)
    model.eval()
    return processor, model, torch


def caption_image(image_bgr: np.ndarray, prompt: str, max_new_tokens: int = 64) -> str:
    """Generate one short factual caption for a BGR OpenCV frame."""
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        raise ValueError("A non-empty BGR image is required for VLM captioning")
    if not prompt.strip():
        raise ValueError("A caption prompt is required")

    processor, model, torch = _load_model()
    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]
    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=chat_prompt, images=[image], return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    caption = processor.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()
    if not caption:
        raise RuntimeError("SmolVLM returned an empty caption")
    return " ".join(caption.split())


def summarize_image(image_bgr: np.ndarray) -> str:
    """Describe visible evidence in a keyframe without seeing heuristic text."""
    prompt = (
        "Describe the visible people and their posture or position in one short factual sentence. "
        "Only describe what can be seen in this image. Do not infer identity, intent, injury, "
        "cause, or actions outside this frame. Return only the sentence."
    )
    return caption_image(image_bgr, prompt)
