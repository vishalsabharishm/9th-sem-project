import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_gradcam import generate_gradcam


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "data" / "person_fixture.mp4"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "test_yolo_gradcam_person.jpg"


@unittest.skipUnless(FIXTURE_PATH.is_file(), "Requires data/person_fixture.mp4.")
class YOLOGradCAMTests(unittest.TestCase):
    def test_generates_gradcam_for_a_real_person_detection(self):
        capture = cv2.VideoCapture(str(FIXTURE_PATH))
        success, frame = capture.read()
        capture.release()
        self.assertTrue(success)

        explanation = generate_gradcam(frame, class_name="person", output_path=OUTPUT_PATH)

        self.assertEqual(explanation.class_name, "person")
        self.assertGreater(explanation.confidence, 0.25)
        self.assertGreater(explanation.target_score, 0.0)
        self.assertGreater(float(np.ptp(explanation.heatmap)), 0.0)
        self.assertTrue(explanation.output_path.is_file())
        self.assertIsNotNone(cv2.imread(str(explanation.output_path)))


if __name__ == "__main__":
    unittest.main()
