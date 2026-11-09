from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FeatureAttribution:
    name: str
    score: float
    rank: int


@dataclass(frozen=True)
class GradCamExplanation:
    overlay_rgb: np.ndarray
    features: tuple[FeatureAttribution, ...]
    class_scores: tuple[float, ...]
    target_class: int


FACIAL_ZONES = (
    ("Left eye zone (approx.)", (0.12, 0.18, 0.49, 0.46), (75, 102, 245)),
    ("Right eye zone (approx.)", (0.51, 0.18, 0.88, 0.46), (75, 102, 245)),
    ("Mouth zone (approx.)", (0.18, 0.50, 0.82, 0.72), (233, 137, 67)),
    ("Jawline zone (approx.)", (0.12, 0.73, 0.88, 0.98), (83, 171, 126)),
)


def _cam_for_class(
    model: nn.Module,
    model_input: torch.Tensor,
    target_class: int | None,
) -> tuple[np.ndarray, tuple[float, ...], int]:
    convolution_layers = [
        layer for layer in model.modules() if isinstance(layer, nn.Conv2d)
    ]
    if not convolution_layers:
        raise ValueError("Grad-CAM requires a model with a convolution layer.")

    activation: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def capture_activation(*hook_args: object) -> None:
        output = hook_args[-1]
        if not isinstance(output, torch.Tensor):
            raise TypeError("The convolution layer did not return a tensor.")
        activation.append(output)
        output.register_hook(gradients.append)

    hook = convolution_layers[-1].register_forward_hook(capture_activation)
    model.zero_grad(set_to_none=True)
    try:
        grad_input = model_input.detach().requires_grad_(True)
        output = model(grad_input)
        if output.ndim != 2 or output.shape[0] != 1:
            raise ValueError("The image model returned an unsupported Grad-CAM output.")
        if target_class is None:
            target_class = int(output[0].argmax().item())
        if target_class < 0 or target_class >= output.shape[1]:
            raise ValueError("The requested Grad-CAM class is outside the model output.")
        if not torch.isfinite(output).all():
            raise ValueError("The image model returned non-finite class scores.")
        class_scores = tuple(float(score) for score in output[0].detach().cpu())
        output[0, target_class].backward()
    finally:
        hook.remove()

    if not activation or not gradients:
        raise RuntimeError("Grad-CAM did not receive activations and gradients.")
    if activation[0].ndim != 4 or gradients[0].shape != activation[0].shape:
        raise ValueError("Grad-CAM requires matching NCHW activations and gradients.")

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation[0]).sum(dim=1))[0]
    cam_array = cam.detach().cpu().numpy()
    maximum = float(cam_array.max())
    if not np.isfinite(cam_array).all():
        raise ValueError("Grad-CAM returned non-finite attribution values.")
    if maximum > 0:
        cam_array = cam_array / maximum

    input_height, input_width = model_input.shape[1:3]
    return (
        cv2.resize(
            cam_array.astype(np.float32),
            (input_width, input_height),
            interpolation=cv2.INTER_LINEAR,
        ),
        class_scores,
        target_class,
    )


def explain_face_prediction(
    image_rgb: np.ndarray,
    model_input: torch.Tensor,
    model: nn.Module,
    target_class: int | None,
    face_bounds: tuple[int, int, int, int],
) -> GradCamExplanation:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with three color channels.")
    if model_input.ndim != 4 or model_input.shape[0] != 1:
        raise ValueError("Expected one preprocessed image in NHWC format.")

    cam, class_scores, target_class = _cam_for_class(
        model,
        model_input,
        target_class,
    )
    frame_height, frame_width = image_rgb.shape[:2]
    face_x1, face_y1, face_x2, face_y2 = face_bounds
    face_x1 = max(0, min(face_x1, frame_width))
    face_y1 = max(0, min(face_y1, frame_height))
    face_x2 = max(face_x1, min(face_x2, frame_width))
    face_y2 = max(face_y1, min(face_y2, frame_height))
    face_width = face_x2 - face_x1
    face_height = face_y2 - face_y1
    if face_width < 1 or face_height < 1:
        raise ValueError("The detected face region is empty.")

    frame_cam = cv2.resize(
        cam,
        (frame_width, frame_height),
        interpolation=cv2.INTER_LINEAR,
    )
    region_values: list[tuple[str, float, tuple[int, int, int, int], tuple[int, int, int]]] = []
    for name, (left, top, right, bottom), color in FACIAL_ZONES:
        zone_x1 = face_x1 + round(left * face_width)
        zone_y1 = face_y1 + round(top * face_height)
        zone_x2 = face_x1 + round(right * face_width)
        zone_y2 = face_y1 + round(bottom * face_height)
        zone_cam = frame_cam[zone_y1:zone_y2, zone_x1:zone_x2]
        score = float(zone_cam.mean()) if zone_cam.size else 0.0
        region_values.append(
            (
                name,
                score,
                (
                    zone_x1,
                    zone_y1,
                    zone_x2,
                    zone_y2,
                ),
                color,
            )
        )

    maximum_region_score = max((region[1] for region in region_values), default=0.0)
    ordered_indices = sorted(
        range(len(region_values)),
        key=lambda index: region_values[index][1],
        reverse=True,
    )
    ranks = {index: rank + 1 for rank, index in enumerate(ordered_indices)}
    features = tuple(
        FeatureAttribution(
            name=name,
            score=score / maximum_region_score if maximum_region_score > 0 else 0.0,
            rank=ranks[index],
        )
        for index, (name, score, _, _) in enumerate(region_values)
    )

    heatmap = cv2.applyColorMap(
        np.round(np.clip(frame_cam, 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    alpha = (np.clip(frame_cam, 0, 1) * 0.48)[..., np.newaxis]
    overlay = np.round(
        np.ascontiguousarray(image_rgb, dtype=np.uint8).astype(np.float32)
        * (1 - alpha)
        + heatmap.astype(np.float32) * alpha
    ).astype(np.uint8)

    for index in ordered_indices:
        name, _, (x1, y1, x2, y2), color = region_values[index]
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(x1 + 1, min(x2, frame_width))
        y2 = max(y1 + 1, min(y2, frame_height))
        attribution = features[index]
        label = f"#{attribution.rank} {name} {attribution.score:.0%}"
        cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        text_y = max(16, y1 - 6)
        cv2.putText(
            overlay,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return GradCamExplanation(overlay, features, class_scores, target_class)
