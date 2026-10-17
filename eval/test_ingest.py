"""Integration check: write a real video file and verify sampling and timestamps."""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from vision.ingest import sampled_frames


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "ten_seconds.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (320, 240))
        assert writer.isOpened(), "OpenCV could not create the test video"
        for index in range(250):
            image = np.full((240, 320, 3), index % 256, dtype=np.uint8)
            cv2.putText(image, str(index), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            writer.write(image)
        writer.release()

        frames = list(sampled_frames(str(path), "test-camera", 2.0))
        assert len(frames) == 20, f"Expected 20 samples, got {len(frames)}"
        assert frames[0].frame_index == 0 and frames[0].video_time_s == 0
        assert 9.5 <= frames[-1].video_time_s < 9.5 + 1 / 25 + 1e-9
        for sample_number, frame in enumerate(frames):
            expected_time = sample_number / 2
            assert expected_time <= frame.video_time_s < expected_time + 1 / 25 + 1e-9
            assert frame.image.shape == (240, 320, 3)
        with_last = list(sampled_frames(str(path), "test-camera", 2.0, include_last_frame=True))
        assert len(with_last) == 21
        assert with_last[-1].frame_index == 249
        assert abs(with_last[-1].video_time_s - 249 / 25) < 1e-9
        print(f"PASS: {len(frames)} samples from 250 frames / 10 seconds at 25 FPS")
        print("First five samples:")
        for frame in frames[:5]:
            print(f"  frame {frame.frame_index:3d} -> {frame.video_time_s:.2f}s")
        print(f"Last sample: frame {frames[-1].frame_index} -> {frames[-1].video_time_s:.2f}s")


if __name__ == "__main__":
    main()
