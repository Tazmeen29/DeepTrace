from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class FrameAnalysisError(ValueError):
    """Raised when a video cannot be decoded for sampled-frame analysis."""


@dataclass(frozen=True)
class SampledVideoFrame:
    frame_number: int
    timestamp_seconds: float | None
    image_rgb: np.ndarray


@dataclass(frozen=True)
class VideoMetadata:
    filename: str
    duration_seconds: float | None
    fps: float | None
    frame_count: int
    width: int
    height: int


def extract_sampled_video_frames(
    video_path: str | Path,
    sample_count: int = 3,
) -> tuple[VideoMetadata, list[SampledVideoFrame]]:
    if not isinstance(video_path, (str, Path)) or not str(video_path).strip():
        raise FrameAnalysisError("Choose a video file to analyze.")
    if sample_count < 1:
        raise ValueError("sample_count must be at least one.")

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FrameAnalysisError("The uploaded video could not be opened.")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = raw_fps if np.isfinite(raw_fps) and raw_fps > 0 else None
        if frame_count < 1:
            raise FrameAnalysisError("No readable frames were found in the video.")

        requested_samples = min(sample_count, frame_count)
        if requested_samples == 1:
            sample_indices = {0}
        else:
            sample_indices = {
                round(index * (frame_count - 1) / (requested_samples - 1))
                for index in range(requested_samples)
            }

        frames: list[SampledVideoFrame] = []
        for frame_index in range(frame_count):
            success, frame_bgr = capture.read()
            if not success:
                break
            if frame_index not in sample_indices:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp = frame_index / fps if fps is not None else None
            frames.append(
                SampledVideoFrame(
                    frame_number=frame_index + 1,
                    timestamp_seconds=timestamp,
                    image_rgb=frame_rgb,
                )
            )

        if len(frames) != len(sample_indices):
            raise FrameAnalysisError(
                "The video ended before all selected frames could be read."
            )

        duration = frame_count / fps if fps is not None else None
        metadata = VideoMetadata(
            filename=path.name,
            duration_seconds=duration,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )
        return metadata, frames
    finally:
        capture.release()
