import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import inference_2
from face_detection import FaceRegion
from frame_analysis import SampledVideoFrame


class ConstantImageModel(nn.Module):
    def __init__(self, real_score: float, fake_score: float) -> None:
        super().__init__()
        self.scores = (real_score, fake_score)
        self.input_shapes: list[tuple[int, ...]] = []
        self.conv = nn.Conv2d(3, 1, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.input_shapes.append(tuple(image.shape))
        feature_signal = self.conv(image.permute(0, 3, 1, 2)).mean(dim=(1, 2, 3))
        scores = image.new_tensor(self.scores).repeat(image.shape[0], 1)
        return scores + feature_signal.unsqueeze(1) * 0


class InferencePredictionTests(unittest.TestCase):
    def test_image_without_face_has_no_authenticity_score(self) -> None:
        with patch.object(inference_2, "detect_faces", return_value=[]):
            prediction = inference_2.analyze_image(
                np.zeros((32, 32, 3), dtype=np.uint8)
            )

        self.assertEqual(prediction.verdict, "NO_FACE")
        self.assertIsNone(prediction.fake_score)
        self.assertEqual(prediction.summary, "NO FACE — N/A")

    def test_image_uses_full_frame_scores_and_returns_top_class_percentage(self) -> None:
        model = ConstantImageModel(0.55, 0.45)
        with (
            patch.object(
                inference_2,
                "detect_faces",
                return_value=[FaceRegion(8, 8, 16, 16)],
            ),
            patch.object(inference_2, "img_model", model),
        ):
            prediction = inference_2.analyze_image(
                np.full((32, 32, 3), 127, dtype=np.uint8)
            )

        self.assertEqual(prediction.verdict, "REAL")
        self.assertEqual(prediction.summary, "REAL — 55.0%")
        self.assertEqual(model.input_shapes, [(1, 256, 256, 3)])

    def test_video_prediction_uses_sampled_faces_and_scores(self) -> None:
        frames = [
            SampledVideoFrame(
                frame_number=1,
                timestamp_seconds=0.0,
                image_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            )
        ]
        prediction = inference_2.ImagePrediction("FAKE", 1, 0.3, 0.7)
        with (
            patch.object(
                inference_2,
                "extract_sampled_video_frames",
                return_value=(None, frames),
            ),
            patch.object(inference_2, "analyze_image", return_value=prediction),
        ):
            result = inference_2.deepfakes_video_predict("sample.mp4")

        self.assertEqual(result, "FAKE — 70.0%")

    def test_video_uses_mean_class_scores_and_returns_top_class_percentage(self) -> None:
        frames = [
            SampledVideoFrame(
                frame_number=index + 1,
                timestamp_seconds=float(index),
                image_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            )
            for index in range(3)
        ]
        predictions = [
            inference_2.ImagePrediction("FAKE", 1, 0.3, 0.7),
            inference_2.ImagePrediction("REAL", 1, 0.8, 0.2),
            inference_2.ImagePrediction("REAL", 1, 0.8, 0.2),
        ]
        with (
            patch.object(
                inference_2,
                "extract_sampled_video_frames",
                return_value=(None, frames),
            ),
            patch.object(
                inference_2,
                "analyze_image",
                side_effect=predictions,
            ),
        ):
            result = inference_2.deepfakes_video_predict("sample.mp4")

        self.assertEqual(result, "REAL — 63.3%")


if __name__ == "__main__":
    unittest.main()
