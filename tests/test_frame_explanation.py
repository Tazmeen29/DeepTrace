import unittest

import numpy as np
import torch
from torch import nn

from frame_explanation import explain_image_model_sensitivity


class RegionSensitiveModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        region_score = image[:, :4, :4, 0].mean(dim=(1, 2))
        return torch.stack((region_score, 1.0 - region_score), dim=1)


class ConstantModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = image.new_tensor([0.7, 0.3])
        return output.unsqueeze(0).repeat(image.shape[0], 1)


class FrameExplanationTests(unittest.TestCase):
    def setUp(self) -> None:
        pixels = np.zeros((16, 16, 3), dtype=np.uint8)
        pixels[:8, :8, 0] = 255
        self.image = pixels
        self.model_input = torch.from_numpy(pixels.astype(np.float32) / 255).unsqueeze(0)

    def test_highlights_a_region_that_changes_the_selected_output(self) -> None:
        result = explain_image_model_sensitivity(
            self.image,
            self.model_input,
            RegionSensitiveModel(),
            grid_size=4,
            batch_size=4,
        )

        self.assertEqual(result.marked_frame.shape, self.image.shape)
        self.assertIsNotNone(result.zoomed_region)
        self.assertIn("Masking the highlighted grid cell", result.note)
        self.assertIn("not evidence of manipulation", result.note)
        self.assertLess(result.zoomed_region.shape[0], self.image.shape[0])

    def test_reports_when_no_localized_sensitivity_is_measured(self) -> None:
        result = explain_image_model_sensitivity(
            self.image,
            self.model_input,
            ConstantModel(),
            grid_size=4,
            batch_size=4,
        )

        self.assertIsNone(result.zoomed_region)
        self.assertIn("No clearly localized", result.note)

    def test_rejects_non_rgb_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "RGB image"):
            explain_image_model_sensitivity(
                self.image[:, :, :1],
                self.model_input,
                ConstantModel(),
            )


if __name__ == "__main__":
    unittest.main()
