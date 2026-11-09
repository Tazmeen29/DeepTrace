
import os
import numpy as np
import onnxruntime as ort
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_xception_inference.onnx"
)

_session = None


def load_model():
    global _session

    if _session is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Xception ONNX model not found: {MODEL_PATH}"
            )

        _session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )

    return _session


def predict_image(image):
    session = load_model()

    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)

    image = image.convert("RGB").resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    output = session.run(
        [output_name],
        {input_name: image_array}
    )[0]

    real_score = float(output.reshape(-1)[0])
    fake_score = 1.0 - real_score

    label = "REAL" if real_score >= 0.5 else "FAKE"

    return {
        "label": label,
        "real_score": real_score,
        "fake_score": fake_score
    }
