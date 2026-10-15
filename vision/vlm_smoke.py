"""Load a small local VLM and caption one bundled test image."""

import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf-cache"))

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

from core.config import DEVICE


MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def main():
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        attn_implementation="eager",
    ).to(DEVICE)
    print(f"Loaded {MODEL_ID} on {DEVICE} in {time.perf_counter() - start:.1f}s", flush=True)

    image_path = ROOT / ".venv" / "Lib" / "site-packages" / "ultralytics" / "assets" / "bus.jpg"
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the people and scene in one short sentence."}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    caption = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    print(f"Caption: {caption}")
    print(f"Generation: {time.perf_counter() - start:.1f}s")
    if DEVICE == "cuda":
        print(f"Peak GPU allocation: {torch.cuda.max_memory_allocated() / 1048576:.0f} MiB")
    assert caption, "VLM returned an empty caption"


if __name__ == "__main__":
    main()
