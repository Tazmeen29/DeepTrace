import argparse
from dataclasses import dataclass
import threading

import cv2
import onnx
import torch
import numpy as np
from models.TMC import ETMC
from models import image
from face_detection import detect_faces
from frame_analysis import FrameAnalysisError, extract_sampled_video_frames
from grad_cam import FeatureAttribution, explain_face_prediction

from onnx2pytorch import ConvertModel

onnx_model = onnx.load('checkpoints/efficientnet.onnx')
pytorch_model = ConvertModel(onnx_model)

#Set random seed for reproducibility.
torch.manual_seed(42)


# Define the audio_args dictionary
audio_args = {
    'nb_samp': 64600,
    'first_conv': 1024,
    'in_channels': 1,
    'filts': [20, [20, 20], [20, 128], [128, 128]],
    'blocks': [2, 4],
    'nb_fc_node': 1024,
    'gru_node': 1024,
    'nb_gru_layer': 3,
    'nb_classes': 2
}


def get_args(parser):
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="datasets/train/fakeavceleb*")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=1024)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="MMDF")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--savedir", type=str, default="./savepath/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--pretrained_image_encoder", type=bool, default = False)
    parser.add_argument("--freeze_image_encoder", type=bool, default = False)
    parser.add_argument("--pretrained_audio_encoder", type = bool, default=False)
    parser.add_argument("--freeze_audio_encoder", type = bool, default = False)
    parser.add_argument("--augment_dataset", type = bool, default = True)

    for key, value in audio_args.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)

def model_summary(args):
    '''Prints the model summary.'''
    model = ETMC(args)

    for name, layer in model.named_modules():
        print(name, layer)

def load_multimodal_model(args):
    '''Load multimodal model'''
    model = ETMC(args)
    ckpt = torch.load('checkpoints/model.pth', map_location = torch.device('cpu'))
    model.load_state_dict(ckpt, strict = True)
    model.eval()
    return model

def load_img_modality_model(args):
    '''Loads image modality model.'''
    rgb_encoder = pytorch_model

    ckpt = torch.load('checkpoints/model.pth', map_location = torch.device('cpu'))
    rgb_encoder.load_state_dict(ckpt['rgb_encoder'], strict = True)
    rgb_encoder.eval()
    return rgb_encoder

def load_spec_modality_model(args):
    spec_encoder = image.RawNet(args)
    ckpt = torch.load('checkpoints/model.pth', map_location = torch.device('cpu'))
    spec_encoder.load_state_dict(ckpt['spec_encoder'], strict = True)
    spec_encoder.eval()
    return spec_encoder


#Load models.
parser = argparse.ArgumentParser(description="Inference models")
get_args(parser)
args, _ = parser.parse_known_args()

spec_model = load_spec_modality_model(args)

img_model = load_img_modality_model(args)


MODEL_INFERENCE_LOCK = threading.RLock()


@dataclass(frozen=True)
class ImagePrediction:
    verdict: str
    face_count: int
    real_score: float | None
    fake_score: float | None
    explanation_image_rgb: np.ndarray | None = None
    feature_attributions: tuple[FeatureAttribution, ...] = ()
    explanation_class: str | None = None

    @property
    def summary(self) -> str:
        if self.verdict == "NO_IMAGE":
            return "NO IMAGE — N/A"
        if self.verdict == "NO_FACE":
            return "NO FACE — N/A"
        score = self.fake_score if self.verdict == "FAKE" else self.real_score
        if score is None:
            raise ValueError(f"{self.verdict} prediction is missing its class score.")
        return f"{self.verdict} — {score:.1%}"


def _verdict_for_scores(real_score: float, fake_score: float) -> str:
    return "REAL" if real_score >= fake_score else "FAKE"


def preprocess_img(face):
    face = face / 255
    face = cv2.resize(face, (256, 256))
    # face = face.transpose(2, 0, 1) #(W, H, C) -> (C, W, H)
    face_pt = torch.unsqueeze(torch.Tensor(face), dim = 0) 
    return face_pt

def preprocess_audio(audio_file):
    audio_pt = torch.unsqueeze(torch.Tensor(audio_file), dim = 0)
    return audio_pt

def deepfakes_spec_predict(input_audio):
    x, _ = input_audio
    audio = preprocess_audio(x)
    spec_grads = spec_model.forward(audio)
    spec_grads_inv = np.exp(spec_grads.cpu().detach().numpy().squeeze())

    # multimodal_grads = multimodal.spec_depth[0].forward(spec_grads)

    # out = nn.Softmax()(multimodal_grads)
    # max = torch.argmax(out, dim = -1) #Index of the max value in the tensor.
    # max_value = out[max] #Actual value of the tensor.
    max_value = np.argmax(spec_grads_inv)

    if max_value > 0.5:
        preds = round(100 - (max_value*100), 3)
        text2 = f"The audio is REAL."

    else:
        preds = round(max_value*100, 3)
        text2 = f"The audio is FAKE."

    return text2

def deepfakes_image_predict(input_image):
    return analyze_image(input_image).summary


def analyze_image(
    input_image: np.ndarray | None,
    with_explanation: bool = False,
) -> ImagePrediction:
    if input_image is None:
        return ImagePrediction("NO_IMAGE", 0, None, None)

    faces = detect_faces(input_image)
    if not faces:
        return ImagePrediction("NO_FACE", 0, None, None)

    face_scores = []
    feature_attributions = []
    explanation_image = input_image.copy() if with_explanation else None
    explanation_classes = []
    with MODEL_INFERENCE_LOCK:
        for face_index, face_region in enumerate(faces, start=1):
            model_input = preprocess_img(input_image)
            if with_explanation:
                explanation = explain_face_prediction(
                    explanation_image,
                    model_input,
                    img_model,
                    target_class=None,
                    face_bounds=(
                        face_region.x,
                        face_region.y,
                        face_region.x + face_region.width,
                        face_region.y + face_region.height,
                    ),
                )
                outputs = np.asarray(explanation.class_scores, dtype=np.float32)
            else:
                with torch.inference_mode():
                    outputs = (
                        img_model(model_input)
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                explanation = None
            if outputs.size != 2 or not np.isfinite(outputs).all():
                raise ValueError("The image model did not return two finite class scores.")
            if np.any(outputs < 0) or np.any(outputs > 1):
                raise ValueError("The image model returned scores outside the expected range.")
            face_scores.append(outputs)
            if explanation is not None:
                explanation_image = explanation.overlay_rgb
                explanation_classes.append(
                    "fake" if explanation.target_class == 1 else "real"
                )
                feature_attributions.extend(
                    FeatureAttribution(
                        name=f"Face {face_index} · {feature.name}",
                        score=feature.score,
                        rank=feature.rank,
                    )
                    for feature in explanation.features
                )

    real_score, fake_score = np.mean(face_scores, axis=0).tolist()
    verdict = _verdict_for_scores(real_score, fake_score)
    return ImagePrediction(
        verdict=verdict,
        face_count=len(faces),
        real_score=float(real_score),
        fake_score=float(fake_score),
        explanation_image_rgb=explanation_image,
        feature_attributions=tuple(feature_attributions),
        explanation_class=(
            ", ".join(dict.fromkeys(explanation_classes))
            if explanation_classes
            else None
        ),
    )


def preprocess_video(input_video, n_frames = 3):
    v_cap = cv2.VideoCapture(input_video)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pick 'n_frames' evenly spaced frames to sample
    if n_frames is None:
        sample = np.arange(0, v_len)
    else:
        sample = np.linspace(0, v_len - 1, n_frames).astype(int)

    #Loop through frames.
    frames = []
    for j in range(v_len):
        success = v_cap.grab()
        if j in sample:
            # Load frame
            success, frame = v_cap.retrieve()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = preprocess_img(frame)
            frames.append(frame)
    v_cap.release()
    return frames


def deepfakes_video_predict(input_video):
    try:
        _, frames = extract_sampled_video_frames(input_video, sample_count=8)
    except FrameAnalysisError as exc:
        return str(exc)

    predictions = [
        analyze_image(frame.image_rgb, with_explanation=False) for frame in frames
    ]
    return summarize_video_predictions(predictions, len(frames))


def summarize_video_predictions(
    predictions: list[ImagePrediction],
    sampled_frame_count: int,
) -> str:
    predictions_with_scores = [
        prediction
        for prediction in predictions
        if prediction.fake_score is not None and prediction.real_score is not None
    ]
    if not predictions_with_scores:
        return (
            f"No face detected in the {sampled_frame_count} sampled video frames. "
            "Authenticity was not assessed; no real/fake score is available."
        )

    real_score, fake_score = np.mean(
        [
            (prediction.real_score, prediction.fake_score)
            for prediction in predictions_with_scores
        ],
        axis=0,
    )
    verdict = _verdict_for_scores(float(real_score), float(fake_score))
    confidence = fake_score if verdict == "FAKE" else real_score
    return f"{verdict} — {confidence:.1%}"
