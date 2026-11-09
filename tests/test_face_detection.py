import unittest
from pathlib import Path

import cv2
import numpy as np

from face_detection import detect_faces


class FaceDetectionTests(unittest.TestCase):
    def test_returns_no_faces_for_blank_image(self) -> None:
        image = np.zeros((128, 128, 3), dtype=np.uint8)

        self.assertEqual(detect_faces(image), [])

    def test_detects_the_face_in_a_project_image_fixture(self) -> None:
        image_path = Path(__file__).parent.parent / "images" / "fake_image.jpg"
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            self.fail(f"Could not load test image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        faces = detect_faces(image_rgb)

        self.assertTrue(any(face.width >= 80 and face.height >= 80 for face in faces))

    def test_rejects_non_rgb_input(self) -> None:
        image = np.zeros((32, 32), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "RGB image"):
            detect_faces(image)


if __name__ == "__main__":
    unittest.main()
