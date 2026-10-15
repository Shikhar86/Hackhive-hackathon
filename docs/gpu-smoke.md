# Main laptop GPU smoke test

Tested on 2026-10-15 on the main Windows laptop.

| Item | Result |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3050 A Laptop GPU, 4094 MiB VRAM |
| Driver / reported CUDA compatibility | 592.82 / 13.1 |
| Python | 3.11.5 |
| PyTorch | 2.14.1+cu126; `torch.cuda.is_available()` true |
| Ultralytics | 8.4.177 |
| YOLO model | `yolov8n.pt` on CUDA |
| Real-image check | Four people detected in bundled `bus.jpg`; model-reported inference 5.5 ms |
| Synthetic warm runs | 127.0, 46.7, 46.4, 48.1, 18.8 ms end-to-end Python calls after warm-up |
| VLM | HuggingFaceTB/SmolVLM-256M-Instruct on CUDA |
| VLM smoke output | "There are three men standing on the road. Two of them are holding cameras. There is a bus with some text on it." |
| VLM generation | 2.3 seconds for one image and up to 48 new tokens |
| VLM peak allocated GPU memory | 1984 MiB |

The VLM caption is unverified and may contain errors. The YOLO and VLM tests ran in separate processes, so this does not prove simultaneous inference fits in 4 GB VRAM. A short real video and a combined-model run are still needed in Phase 1/2.

## Repeat commands

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -m vision.vlm_smoke
```

The VLM downloads into `data/hf-cache` on first run. Models are open-source local assets; no paid API is used.
