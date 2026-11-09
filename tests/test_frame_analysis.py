import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from frame_analysis import FrameAnalysisError, extract_sampled_video_frames


class FrameAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_path = Path(self.temp_dir.name) / "sample.avi"
        writer = cv2.VideoWriter(
            str(self.video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            5.0,
            (64, 48),
        )
        if not writer.isOpened():
            writer.release()
            self.temp_dir.cleanup()
            self.skipTest("OpenCV MJPG video writer is not available.")

        for frame_index in range(7):
            frame = np.full(
                (48, 64, 3),
                fill_value=(10 + frame_index * 2, 40 + frame_index * 2, 120 + frame_index * 2),
                dtype=np.uint8,
            )
            writer.write(frame)
        writer.release()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_extracts_three_evenly_spaced_rgb_frames_and_metadata(self) -> None:
        metadata, frames = extract_sampled_video_frames(self.video_path)

        self.assertEqual(metadata.filename, "sample.avi")
        self.assertEqual(metadata.frame_count, 7)
        self.assertEqual((metadata.width, metadata.height), (64, 48))
        self.assertEqual(metadata.fps, 5.0)
        self.assertAlmostEqual(metadata.duration_seconds, 1.4)
        self.assertEqual([frame.frame_number for frame in frames], [1, 4, 7])
        self.assertEqual(
            [frame.timestamp_seconds for frame in frames],
            [0.0, 0.6, 1.2],
        )
        self.assertEqual(frames[0].image_rgb.shape, (48, 64, 3))
        self.assertEqual(frames[0].image_rgb.dtype, np.uint8)
        self.assertGreater(
            int(frames[0].image_rgb[0, 0, 0]),
            int(frames[0].image_rgb[0, 0, 2]),
        )

    def test_sampling_count_is_capped_by_available_frame_count(self) -> None:
        _, frames = extract_sampled_video_frames(self.video_path, sample_count=10)

        self.assertEqual(len(frames), 7)

    def test_rejects_missing_or_unreadable_video(self) -> None:
        with self.assertRaises(FrameAnalysisError):
            extract_sampled_video_frames(
                Path(self.temp_dir.name) / "missing.avi"
            )

    def test_rejects_zero_sample_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            extract_sampled_video_frames(self.video_path, sample_count=0)


if __name__ == "__main__":
    unittest.main()
