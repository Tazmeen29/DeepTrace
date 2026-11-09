
import onnxruntime as ort
import numpy as np
from PIL import Image
import sys

MODEL_PATH = "models/best_xception_inference.onnx"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    output = session.run(
        [output_name],
        {input_name: image_array}
    )[0]

    real_score = float(output.reshape(-1)[0])
    fake_score = 1.0 - real_score

    label = "REAL" if real_score >= 0.5 else "FAKE"

    print("\n===== DEEPTRACE XCEPTION RESULT =====")
    print("Prediction:", label)
    print(f"Real score: {real_score * 100:.2f}%")
    print(f"Fake score: {fake_score * 100:.2f}%")
    print("=====================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_xception_onnx.py image.jpg")
    else:
        predict_image(sys.argv[1])
