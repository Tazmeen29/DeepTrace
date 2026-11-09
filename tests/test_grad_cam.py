import unittest

import numpy as np
import torch
from torch import nn

from grad_cam import explain_face_prediction


class SpatialModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, 0, 0] = 1

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        red_features = self.conv(image.permute(0, 3, 1, 2))
        region_score = red_features.mean(dim=(1, 2, 3))
        return torch.stack(
            (0.5 + region_score, 0.5 - region_score),
            dim=1,
        )


class GradCamTests(unittest.TestCase):
    def test_generates_gradcam_overlay_and_ranked_facial_zones(self) -> None:
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:32, :32, 0] = 255
        image = torch.from_numpy(pixels.astype(np.float32) / 255).unsqueeze(0)

        result = explain_face_prediction(
            pixels,
            image,
            SpatialModel(),
            target_class=None,
            face_bounds=(0, 0, 64, 64),
        )

        self.assertEqual(result.overlay_rgb.shape, pixels.shape)
        self.assertEqual(len(result.features), 4)
        self.assertEqual(sorted(feature.rank for feature in result.features), [1, 2, 3, 4])
        self.assertEqual(result.features[0].name, "Left eye zone (approx.)")
        self.assertGreater(float(np.abs(result.overlay_rgb.astype(int) - pixels).sum()), 0)
        self.assertEqual(result.target_class, 0)
        self.assertEqual(len(result.class_scores), 2)


if __name__ == "__main__":
    unittest.main()
