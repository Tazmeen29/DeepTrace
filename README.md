# DeepTrace – Deepfake Video Detection Using Deep Learning

An AI-powered application for analyzing potentially manipulated images and videos.

---

## About DeepTrace

**DeepTrace** is a deep-learning-based application designed to identify potentially manipulated facial content in digital images and videos.

The project uses trained and fine-tuned **Xception deep-learning model**, developed using the DeepFake Detection Challenge (DFDC) dataset in Google Colab.

It combines deep-learning-based prediction with video processing, frame extraction, face analysis, and an interactive web interface to provide an accessible deepfake detection workflow.

The goal is to make deepfake analysis easier while providing useful information about model predictions.

## Key Features

- **Xception-Based Deepfake Detection:** Independently trained and fine-tuned Xception model using the DFDC dataset.
- **Image Analysis:** Analyze uploaded images using the configured detection model.
- **Video Analysis:** Process uploaded videos through frame extraction and analysis.
- **Frame Extraction:** Extract representative frames from uploaded videos using OpenCV.
- **Face Detection:** Detect and preprocess facial regions for analysis.
- **Authentication:** User registration, login, and protected application access.
- **Explainable AI:** Grad-CAM-related functionality for visual interpretation, subject to model compatibility.
- **Interactive Interface:** Gradio-based application for uploading and analyzing media.

*Feature availability depends on successful integration and testing of the corresponding modules.*

## System Architecture

The proposed system follows this processing pipeline:

```text
                  USER
                    |
                    v
             Web Interface
                    |
                    v
                Backend API
                    |
                    v
             Video Processing
                    |
                    v
              Frame Extraction
                    |
                    v
          Face Detection & Alignment
                    |
                    v
                Preprocessing
                    |
                    v
           Xception Deep Learning Model
                    |
                    v
             Frame Predictions
                    |
                    v
             Result Aggregation
                    |
                    v
           Real / Deepfake Prediction
                    |
                    v
             Confidence Information
```

The actual implementation may differ from the proposed architecture as development progresses.

## Model Development and Training

The core detection model was independently trained and fine-tuned using the **Xception architecture**.

### Training Environment

- Platform: Google Colab
- Framework: TensorFlow / Keras
- Architecture: Xception
- Dataset: DeepFake Detection Challenge (DFDC)
- Input image size: 224 × 224
- Learning approach: Transfer learning and fine-tuning

### Training Process

1. Prepared and organized real and manipulated facial images.
2. Preprocessed images to match the model's input requirements.
3. Initialized Xception using ImageNet-pretrained weights.
4. Trained a classification head for real and fake prediction.
5. Fine-tuned selected layers of the Xception network.
6. Evaluated model performance using validation data.
7. Saved the fine-tuned model checkpoint.

**Model checkpoint:** `best_xception_finetuned.keras`

The checkpoint should be made available separately or placed in the configured model directory before running Xception inference.

### Model Architecture

The Xception network uses depthwise separable convolutions to learn image features efficiently.

The classification pipeline consists of:

- Xception feature extraction
- Global Average Pooling
- Dropout regularization
- Fully connected classification layer
- Sigmoid output for binary prediction

### Model Limitations

The model's predictions are estimates and may be affected by image quality, compression, lighting, face alignment, unseen manipulation techniques, and differences between training and testing datasets.

A prediction score should not be interpreted as a guarantee that a media file is genuine or manipulated.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| TensorFlow / Keras | Deep-learning model training and inference |
| Xception | Deepfake classification |
| OpenCV | Video processing and frame extraction |
| Gradio | Web application interface |
| NumPy | Numerical operations |
| Pillow | Image processing |
| Google Colab | Model training environment |
| Git / GitHub | Version control |

## Project Structure

The following is a representative project structure. Your actual repository may contain additional modules and directories.

```text
DeepTrace/
│
├── app.py
├── inference.py
├── inference_2.py
├── auth_store.py
├── auth_web.py
│
├── face_detection.py
├── frame_analysis.py
├── frame_explanation.py
├── grad_cam.py
│
├── models/
├── checkpoints/
│
├── tests/
│   └── test_auth_store.py
│
├── frontendhtml/
├── images/
├── videos/
├── utils/
│
├── requirements.txt
├── .gitignore
├── .env.example
├── LICENSE
└── README.md
```

Only modules required by the final Xception-based implementation should be considered part of the active inference pipeline.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Tazmeen29/DeepTrace.git
```

### 2. Navigate to the project directory

```bash
cd DeepTrace
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows:**

```powershell
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Configure environment variables

Use `.env.example` as a reference for the required configuration.

Create a local `.env` file and provide the required values. Do not upload real credentials or secret keys to GitHub.

### 7. Configure the model checkpoint

Place the independently trained Xception checkpoint in the model directory expected by the inference configuration.

For example:

```text
models/
└── best_xception_finetuned.keras
```

Ensure the inference code uses the same preprocessing, image dimensions, class mapping, and prediction threshold as the training pipeline.

### 8. Run the application

```bash
python app.py
```

Open the local URL displayed in the terminal, typically:

```text
http://127.0.0.1:7860
```

## Authentication

DeepTrace includes an authentication module designed to provide:

- User registration
- User login
- Password hashing
- Session-based access
- Protected application functionality

Authentication tests are available in the `tests/` directory.

Run the tests using:

```bash
pytest -q
```

## Explainable AI

The project includes Grad-CAM-related functionality to support visual interpretation of model predictions.

Grad-CAM can highlight image regions that influence a model's classification.

For the Xception model, the Grad-CAM implementation must be compatible with TensorFlow/Keras and the model's actual computational graph.

## Future Enhancements

- Improved video-level prediction and temporal analysis.
- Frame-level prediction visualization.
- Suspicious-frame highlighting.
- Downloadable detection reports.
- Detection history and user dashboard.
- Improved explainability and model interpretation.
- Further evaluation on independent datasets.
- Performance and generalization improvements.

## Project Status

DeepTrace is an academic deep-learning project under development.

Its detection results should be treated as model predictions rather than definitive proof of media authenticity.

## Dataset and Acknowledgements

The Xception model was independently trained using the DeepFake Detection Challenge (DFDC) dataset.

Dataset source:

https://ai.meta.com/datasets/dfdc/

All datasets, pretrained weights, third-party libraries, and any adapted source-code components remain subject to their respective licences and attribution requirements.

