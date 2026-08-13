"""Grad-CAM explanations for individual YOLOv8 detection predictions.

This module explains a YOLO class score, not a downstream rule-based event or
risk level. It is deliberately independent of the detection/tracking pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

try:
    from config import MODELS_DIR, OUTPUTS_DIR, ensure_directories
    from detection import DetectionError, load_yolo_model
except ImportError:  # pragma: no cover - supports package execution
    from src.config import MODELS_DIR, OUTPUTS_DIR, ensure_directories
    from src.detection import DetectionError, load_yolo_model


DEFAULT_MODEL_PATH = MODELS_DIR / "yolov8s.pt"
DEFAULT_OUTPUT_PATH = OUTPUTS_DIR / "yolo_gradcam_person.jpg"


class XAIError(RuntimeError):
    """Raised when a valid YOLO Grad-CAM explanation cannot be generated."""


@dataclass(frozen=True)
class GradCAMExplanation:
    """Metadata and artifact path for one class-score Grad-CAM explanation."""

    class_name: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    raw_prediction_index: int
    target_score: float
    heatmap: np.ndarray
    output_path: Path


def _class_id(model_names, class_name: str) -> int:
    items = model_names.items() if isinstance(model_names, dict) else enumerate(model_names)
    for class_id, name in items:
        if str(name) == class_name:
            return int(class_id)
    raise XAIError(f"YOLO model does not contain class '{class_name}'.")


def _network_input(frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Resize to YOLO's stride while retaining scale factors for the overlay."""
    height, width = frame.shape[:2]
    input_height = int(np.ceil(height / 32) * 32)
    input_width = int(np.ceil(width / 32) * 32)
    if (input_width, input_height) == (width, height):
        return frame, 1.0, 1.0
    resized = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    return resized, width / input_width, height / input_height


def _select_detection(result, class_id: int):
    """Return the highest-confidence post-NMS detection for the requested class."""
    if result.boxes is None or len(result.boxes) == 0:
        raise XAIError("YOLO produced no detections for this frame.")
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    matching = np.flatnonzero(classes == class_id)
    if matching.size == 0:
        raise XAIError("YOLO produced no detection for the requested class.")
    selected = int(matching[np.argmax(confidences[matching])])
    return result.boxes.xyxy.cpu().numpy()[selected], float(confidences[selected])


def generate_gradcam(
    frame: np.ndarray,
    class_name: str = "person",
    model_path: Path = DEFAULT_MODEL_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    confidence_threshold: float = 0.25,
    target_layer_index: int = 21,
) -> GradCAMExplanation:
    """Create Grad-CAM for an actual post-NMS YOLO class detection.

    The target scalar is the requested class score from the raw YOLO prediction
    whose center is closest to the selected post-NMS detection's center.
    """
    if frame is None or frame.size == 0:
        raise XAIError("Cannot explain an empty frame.")
    ensure_directories()
    inference_frame, scale_x, scale_y = _network_input(frame)
    model = load_yolo_model(model_path)
    class_id = _class_id(model.names, class_name)

    try:
        result = model.predict(
            source=inference_frame,
            imgsz=(inference_frame.shape[0], inference_frame.shape[1]),
            conf=confidence_threshold,
            verbose=False,
        )[0]
    except Exception as error:
        raise XAIError(f"YOLO inference failed while selecting an XAI target: {error}") from error

    selected_box, confidence = _select_detection(result, class_id)
    # Ultralytics ``predict`` runs under inference mode and caches decoded
    # anchors in its detection head. Backpropagating through those tensors is
    # invalid, so the Grad-CAM forward pass uses a fresh copy of the same local
    # weights with no inference-mode cache.
    network = load_yolo_model(model_path).model
    if not hasattr(network, "model") or target_layer_index >= len(network.model):
        raise XAIError(f"YOLO target layer {target_layer_index} is unavailable.")
    target_layer = network.model[target_layer_index]
    device = next(network.parameters()).device
    original_grad_flags = [parameter.requires_grad for parameter in network.parameters()]
    activations = []
    gradients = []
    hook = target_layer.register_forward_hook(lambda _, __, output: activations.append(output))

    try:
        for parameter in network.parameters():
            parameter.requires_grad_(True)
        network.zero_grad(set_to_none=True)
        tensor = torch.from_numpy(inference_frame[:, :, ::-1].copy()).permute(2, 0, 1)
        tensor = tensor.float().unsqueeze(0).to(device) / 255.0
        with torch.enable_grad():
            raw_output = network(tensor)[0]
            if not activations or not isinstance(activations[0], torch.Tensor):
                raise XAIError("Selected YOLO layer did not produce a tensor activation.")
            activations[0].register_hook(lambda gradient: gradients.append(gradient))

            target_center = (selected_box[0] + selected_box[2]) / 2, (selected_box[1] + selected_box[3]) / 2
            raw_centers = raw_output[0, 0:2, :]
            distances = (raw_centers[0] - target_center[0]).square() + (raw_centers[1] - target_center[1]).square()
            raw_prediction_index = int(torch.argmin(distances).item())
            target_score = raw_output[0, 4 + class_id, raw_prediction_index]
            target_score.backward()

        if not gradients:
            raise XAIError("No gradients reached the selected YOLO feature layer.")
        activation = activations[0].detach()[0]
        gradient = gradients[0].detach()[0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=0)).cpu().numpy()
    finally:
        hook.remove()
        for parameter, requires_grad in zip(network.parameters(), original_grad_flags):
            parameter.requires_grad_(requires_grad)

    if not np.isfinite(cam).all() or cam.max() <= cam.min():
        raise XAIError("Grad-CAM computation produced no usable spatial variation.")
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    color_map = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.55, color_map, 0.45, 0)

    x1, y1, x2, y2 = selected_box
    bounding_box = (
        int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
    )
    cv2.rectangle(overlay, bounding_box[:2], bounding_box[2:], (255, 255, 255), 2)
    cv2.putText(
        overlay,
        f"Grad-CAM: {class_name} {confidence:.2f}",
        (bounding_box[0], max(24, bounding_box[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), overlay):
        raise XAIError(f"Could not write Grad-CAM image: {destination}")

    return GradCAMExplanation(
        class_name=class_name,
        confidence=confidence,
        bounding_box=bounding_box,
        raw_prediction_index=raw_prediction_index,
        target_score=float(target_score.detach().cpu().item()),
        heatmap=heatmap,
        output_path=destination,
    )
