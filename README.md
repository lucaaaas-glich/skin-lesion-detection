# Skin Lesion Detection with YOLOv8

皮肤病灶定位与良恶性分类研究：使用 YOLOv8、分组数据划分和误差分析，探索检测效果与误警报之间的取舍。

A research prototype comparing YOLOv8n and YOLOv8s on dermoscopic images.
This repository presents the implementation and aggregate experiment results;
raw images, patient metadata, trained weights and local credentials are excluded.

## What I built

- A data workflow for checking mask-derived boxes, matching class labels and
  grouping images by recorded lesion/patient identifiers before splitting.
- Training and inference entry points around Ultralytics, with fixed configs,
  checkpoint checksums and consistent preprocessing/evaluation.
- Evaluation separating localization, class prediction, missed malignant lesions
  and false alerts on benign images, plus model-size and augmentation comparisons.
- Tests for data leakage checks, preprocessing, checkpoint selection and metrics.

## Results at a glance

The main cohort contains 1,098 images (768 training, 165 validation and
165 supplementary reassessment). Both YOLOv8n and YOLOv8s completed 60 epochs.

The light-geometry augmentation candidate improved malignant detections from
13/41 to 18/41 on reused validation images, while remaining within the chosen
validation false-alert budget. A second seed showed the same direction
(12/41 to 15/41). These are validation findings, not a new independent test.

The original n/s models remained weak on supplementary reassessment: malignant
detections were 10/41 and 6/41 at their separately fixed operating thresholds.
The n model exceeded its validation false-alert budget on reassessment.
These limitations are retained in [aggregate results](results/summary.json).

## Code map

| Area | Files |
| --- | --- |
| Dataset and grouping | `prepare_dataset.py`, `plan_group_splits.py`, `dataset.py` |
| Model and training | `modules.py`, `transfer.py`, `train.py`, `schedule.py` |
| Evaluation and inference | `evaluation.py`, `predict.py`, `preprocessing.py` |
| Reproducibility | `runtime.py`, `checkpoints.py`, `weights_registry.json` |
| Verification | `test_*.py`, `smoke_test.py`, `preflight.py` |

## Setup and use

Create an isolated Python environment and install the pinned requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -p 'test_*.py'
python predict.py --help
```

The dependency file records the original tested environment; choose the PyTorch
build appropriate to your hardware. Real training requires separately obtained
ISIC data, verified manifests and a locked experiment configuration. The saved
`experiment.json` is a historical draft and deliberately does not start full
training. This code snapshot does not include a ready-to-run trained detector.
Prediction accepts a checkpoint and its matching config/validation lock; see
`python predict.py --help` for the required arguments.

## Scope and attribution

- Data comes from ISIC 2018 Task 1 images/masks and matched official metadata.
  Obtain it from the [ISIC challenge](https://challenge.isic-archive.com/data/).
- Network layers, pretrained COCO assets and the trainer are provided by
  [Ultralytics](https://github.com/ultralytics/ultralytics); this project implements
  the surrounding data, experiment, evaluation and verification workflow.
- Missing patient IDs prevent a full patient-independence claim. Repeated use of
  validation images and prior image exposure limit generalization conclusions.
- Confidence is not a calibrated cancer probability. This research prototype
  is not clinically validated and is not a diagnostic product.
- Codex assisted with implementation, checks, experiment operations and drafting.
  Upstream code and data retain their respective licenses and terms.

`SOURCE_MANIFEST.json` records checksums of the copied implementation files.
