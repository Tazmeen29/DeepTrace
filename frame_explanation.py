from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class OcclusionExplanation:
    marked_frame: np.ndarray
    zoomed_region: np.ndarray | None
    note: str


def explain_image_model_sensitivity(
    image_rgb: np.ndarray,
    model_input: torch.Tensor,
    model: nn.Module,
    grid_size: int = 5,
    batch_size: int = 8,
) -> OcclusionExplanation:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with three color channels.")
    if model_input.ndim != 4 or model_input.shape[0] != 1:
        raise ValueError("Expected one preprocessed image in NHWC format.")
    if grid_size < 1 or batch_size < 1:
        raise ValueError("grid_size and batch_size must be at least one.")

    height, width = model_input.shape[1:3]
    original_model_input = model_input[0].detach().cpu().numpy().copy()
    replacement_color = original_model_input.mean(axis=(0, 1))

    with torch.inference_mode():
        baseline = model(model_input).detach().cpu().numpy().reshape(-1)
    if baseline.size < 2 or not np.isfinite(baseline[:2]).all():
        raise ValueError("The image model did not return two finite class outputs.")

    predicted_class = 0 if baseline[0] > 0.5 else 1
    baseline_score = float(baseline[predicted_class])
    regions: list[tuple[int, int, int, int]] = []
    for row in range(grid_size):
        y1 = row * height // grid_size
        y2 = (row + 1) * height // grid_size
        for column in range(grid_size):
            x1 = column * width // grid_size
            x2 = (column + 1) * width // grid_size
            regions.append((x1, y1, x2, y2))

    score_changes: list[float] = []
    for start in range(0, len(regions), batch_size):
        batch_regions = regions[start : start + batch_size]
        masked_inputs = np.repeat(
            original_model_input[np.newaxis, ...],
            len(batch_regions),
            axis=0,
        )
        for masked_image, (x1, y1, x2, y2) in zip(masked_inputs, batch_regions):
            masked_image[y1:y2, x1:x2] = replacement_color

        model_batch = torch.as_tensor(
            masked_inputs,
            dtype=model_input.dtype,
            device=model_input.device,
        )
        with torch.inference_mode():
            masked_outputs = model(model_batch).detach().cpu().numpy()
        masked_scores = masked_outputs.reshape(len(batch_regions), -1)
        if masked_scores.shape[1] < 2 or not np.isfinite(
            masked_scores[:, :2]
        ).all():
            raise ValueError("The image model returned invalid occlusion outputs.")
        score_changes.extend(
            baseline_score - float(scores[predicted_class])
            for scores in masked_scores
        )

    sensitivity = np.abs(np.asarray(score_changes, dtype=np.float32)).reshape(
        grid_size,
        grid_size,
    )
    max_change = float(sensitivity.max())
    frame = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    if max_change <= 1e-4:
        return OcclusionExplanation(
            marked_frame=frame.copy(),
            zoomed_region=None,
            note="No clearly localized score-sensitive area was found in the tested regions.",
        )

    strongest_index = int(np.argmax(sensitivity))
    strongest_row, strongest_column = divmod(strongest_index, grid_size)
    model_x1 = strongest_column * width // grid_size
    model_x2 = (strongest_column + 1) * width // grid_size
    model_y1 = strongest_row * height // grid_size
    model_y2 = (strongest_row + 1) * height // grid_size

    frame_height, frame_width = frame.shape[:2]
    x1 = model_x1 * frame_width // width
    x2 = (model_x2 * frame_width + width - 1) // width
    y1 = model_y1 * frame_height // height
    y2 = (model_y2 * frame_height + height - 1) // height

    sensitivity_map = cv2.resize(
        sensitivity,
        (frame_width, frame_height),
        interpolation=cv2.INTER_LINEAR,
    )
    normalized = np.clip(sensitivity_map / max_change, 0.0, 1.0)
    heatmap = cv2.applyColorMap(
        np.round(normalized * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    alpha = (normalized * 0.5)[..., np.newaxis]
    marked_frame = np.round(
        frame.astype(np.float32) * (1.0 - alpha)
        + heatmap.astype(np.float32) * alpha
    ).astype(np.uint8)
    cv2.rectangle(marked_frame, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 3)
    cv2.rectangle(marked_frame, (x1, y1), (x2 - 1, y2 - 1), (220, 35, 45), 2)

    margin_x = max(8, (x2 - x1) // 2)
    margin_y = max(8, (y2 - y1) // 2)
    crop_x1 = max(0, x1 - margin_x)
    crop_y1 = max(0, y1 - margin_y)
    crop_x2 = min(frame_width, x2 + margin_x)
    crop_y2 = min(frame_height, y2 + margin_y)
    zoomed_region = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    change_direction = (
        "decreased"
        if score_changes[strongest_index] >= 0
        else "increased"
    )
    note = (
        f"Masking the highlighted grid cell {change_direction} the selected "
        f"class output by {max_change:.4f} model-output units. This is a "
        "sensitivity visualization, not evidence of manipulation or a causal explanation."
    )
    return OcclusionExplanation(marked_frame, zoomed_region, note)
