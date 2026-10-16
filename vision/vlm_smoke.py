"""Smoke test the cached local VLM on one bundled test image."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import cv2

from vision.vlm import MODEL_ID, caption_image


MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def main():
    image_path = ROOT / ".venv" / "Lib" / "site-packages" / "ultralytics" / "assets" / "bus.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Bundled smoke-test image not found: {image_path}")
    caption = caption_image(image, "Describe the people and scene in one short factual sentence.", max_new_tokens=48)
    print(f"Model: {MODEL_ID}")
    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
