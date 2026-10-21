"""PyTorch integration of upstream YOLOv8, not an original YOLO implementation.

Conv/C2f/SPPF/Detect/loss are provided by Ultralytics 8.3.221 (AGPL-3.0).
Source: https://github.com/ultralytics/ultralytics/tree/v8.3.221/ultralytics/nn
Course acceptance of this component packaging remains unconfirmed.
"""
from runtime import NAMES, check_runtime


def build_detector(cfg, weights=None, verbose=False):
    """The trainer calls this function to construct the actual two-class network."""
    check_runtime()
    from ultralytics.nn.tasks import DetectionModel
    model = DetectionModel(cfg, nc=2, ch=3, verbose=verbose)
    if weights is not None:
        model.load(weights)
    model.names = NAMES.copy()
    if model.model[-1].nc != 2:
        raise ValueError('Detection head does not have two class outputs')
    return model


def load_trained(checkpoint):
    check_runtime()
    from ultralytics import YOLO
    model = YOLO(str(checkpoint), task='detect')
    if model.names != NAMES or model.model.model[-1].nc != 2:
        raise ValueError('Use a new two-class checkpoint, not an old three-class or COCO checkpoint')
    return model
